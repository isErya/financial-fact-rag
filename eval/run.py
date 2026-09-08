"""Evaluate the pipeline on a labelled question set.

  python -m eval.run --retrieval-only --set tuning [--index-dir index]
                     [--out eval/results/retrieval.json] [--ablate]
  python -m eval.run --report BEFORE.json AFTER.json
  python -m eval.run --grade RESULTS.json

Layer 1 (retrieval only, no model request): for each row, run ask() as a
dry run and score what reached the context: the resolved tickers against
the expected set, each expected section and file against the ones used,
every expected passage (expected_substrings, or the older single
expected_substring) against the rendered excerpts, the status of refusal
rows, and the token estimate against the budget. --ablate repeats the run
with lexical only, dense only, hybrid, and hybrid without pinning, and
prints one summary row per mode; dense modes need an index built with
--dense 1, which is why the tuning-file index is the usual target.

--report reads two saved result files, prints a before, after and delta per
metric, and prints one row shaped for the table in docs/PROMPT-LOG.md with the
metric cell filled in. It reads the metric names out of the files rather than
from a list here, so a layer 2 result file reports without a change. It refuses
to difference two files that do not share a question set, an index and a shape.

--grade walks a result file row by row and takes the four rubric scores from
eval/notes.md, 0 to 2 each, graded by a person. The grades are stored back into
the result file, so a session stopped part way is resumed rather than restarted.

Neither --report nor --grade makes a model request.

The held-out set is read only with --final, so no prompt or rule is tuned
against it by accident.
"""

import argparse
import dataclasses
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service"))

import config  # noqa: E402
import index as index_module  # noqa: E402
import llm_client  # noqa: E402
from ask import ask  # noqa: E402
from corpus import normalize  # noqa: E402

SETS = {"tuning": os.path.join(ROOT, "eval", "tuning.jsonl"),
        "heldout": os.path.join(ROOT, "eval", "heldout.jsonl")}
ABLATION = [("bm25", "bm25", True), ("dense", "dense", True), ("hybrid", "hybrid", True),
            ("hybrid-nopin", "hybrid", False)]


def read_rows(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def expected_substrings(row: dict) -> list[str]:
    """The passages a row expects in the rendered context. Newer rows list
    them under expected_substrings; older rows carry one expected_substring,
    which reads as a one-item list."""
    listed = row.get("expected_substrings")
    if listed:
        return list(listed)
    single = row.get("expected_substring")
    return [single] if single else []


def score_row(row: dict, payload: dict) -> dict:
    plan = payload["plan"]
    context = payload.get("context", [])
    resolved = sorted(c["ticker"] for c in plan.companies)
    used_items = {r["item"] for r in context}
    used_files = {r["file"] for r in context}
    rendered = normalize(payload.get("rendered", ""))
    expected_items = row.get("expected_items") or []
    expected_files = row.get("expected_files") or []
    substrings = expected_substrings(row)
    substrings_missing = [s for s in substrings if normalize(s) not in rendered]
    budget = payload.get("budget", {})
    return {
        "id": row["id"],
        "tickers_ok": resolved == sorted(row.get("expected_tickers") or []),
        "resolved": resolved,
        "items_hit": sum(1 for i in expected_items if i in used_items),
        "items_n": len(expected_items),
        "items_missing": [i for i in expected_items if i not in used_items],
        "files_hit": sum(1 for f in expected_files if f in used_files),
        "files_n": len(expected_files),
        "files_missing": [f for f in expected_files if f not in used_files],
        # A passage hit needs every listed substring; one missing figure
        # in a three-figure brief is a miss.
        "substring_hit": (not substrings_missing) if substrings else None,
        "substrings_missing": substrings_missing,
        "status": payload["status"],
        "status_ok": payload["status"] == row.get("expected_status", "ok"),
        "abstain_row": bool(row.get("must_abstain")),
        "tokens": budget.get("tokens_est", 0),
        "budget_ok": budget.get("tokens_est", 0) <= budget.get("budget_tokens", 0) if budget else True,
        "chunks": len(context),
    }


def run_set(rows: list[dict], loaded, companies: dict, mode: str, pin: bool) -> list[dict]:
    return [score_row(row, ask(row["question"], loaded, companies, dry_run=True, mode=mode, pin=pin))
            for row in rows]


def summarize(scores: list[dict]) -> dict:
    items_hit = sum(s["items_hit"] for s in scores)
    items_n = sum(s["items_n"] for s in scores)
    files_hit = sum(s["files_hit"] for s in scores)
    files_n = sum(s["files_n"] for s in scores)
    passages = [s for s in scores if s["substring_hit"] is not None]
    abstain = [s for s in scores if s["abstain_row"]]
    answered = [s for s in scores if s["status"] == "ok"]
    return {
        "section_hit": (items_hit, items_n),
        "file_coverage": (files_hit, files_n),
        "passage_hit": (sum(1 for s in passages if s["substring_hit"]), len(passages)),
        "abstain": (sum(1 for s in abstain if s["status_ok"]), len(abstain)),
        "status_ok": (sum(1 for s in scores if s["status_ok"]), len(scores)),
        "budget_ok": (sum(1 for s in answered if s["budget_ok"]), len(answered)),
        "mean_tokens": round(statistics.mean(s["tokens"] for s in answered)) if answered else 0,
    }


def _rate(pair: tuple[int, int]) -> str:
    hit, n = pair
    return "%d/%d (%d%%)" % (hit, n, round(100.0 * hit / n)) if n else "-"


def summary_line(summary: dict) -> str:
    return ("section hit rate %s | file coverage %s | passage hit rate %s | abstain hit rate %s | "
            "status match %s | within budget %s | mean tokens %d" % (
                _rate(summary["section_hit"]), _rate(summary["file_coverage"]),
                _rate(summary["passage_hit"]), _rate(summary["abstain"]), _rate(summary["status_ok"]),
                _rate(summary["budget_ok"]), summary["mean_tokens"]))


def table(scores: list[dict]) -> str:
    lines = ["| id | tickers | items | files | substring | status | tokens |",
             "|---|---|---|---|---|---|---|"]
    for s in scores:
        tickers = "ok" if s["tickers_ok"] else "got " + (",".join(s["resolved"]) or "none")
        items = "%d/%d" % (s["items_hit"], s["items_n"]) if s["items_n"] else "-"
        if s["items_missing"]:
            items += " (missing " + ", ".join(s["items_missing"]) + ")"
        files = "%d/%d" % (s["files_hit"], s["files_n"]) if s["files_n"] else "-"
        if s["files_missing"]:
            files += " (missing " + ", ".join(f.split("_full")[0] for f in s["files_missing"]) + ")"
        substring = "-" if s["substring_hit"] is None else ("hit" if s["substring_hit"] else "MISS")
        if s.get("substrings_missing"):
            substring += " (missing " + ", ".join(s["substrings_missing"]) + ")"
        status = s["status"] if s["status_ok"] else "%s (expected other)" % s["status"]
        tokens = "%d%s" % (s["tokens"], "" if s["budget_ok"] else " OVER") if s["status"] == "ok" else "-"
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            s["id"], tickers, items, files, substring, status, tokens))
    return "\n".join(lines)


# --- report: the change between two saved result files ---------------------

# The friendly names summary_line already uses, so a metric reads the same
# in the log as it does in a run. Anything a later layer adds falls back to
# its key with the underscores opened up.
REPORT_LABELS = {
    "section_hit": "section hit rate",
    "file_coverage": "file coverage",
    "passage_hit": "passage hit rate",
    "abstain": "abstain hit rate",
    "status_ok": "status match",
    "budget_ok": "within budget",
    "mean_tokens": "mean tokens",
}

RUBRIC = (
    ("addresses", "addresses the question"),
    ("follows", "conclusions follow from the cited excerpts"),
    ("complete", "complete against the expected facts"),
    ("comparability", "comparability and gaps stated where they should be"),
)


def refuse(message: str) -> None:
    print(message)
    sys.exit(2)


def load_results(path: str) -> dict:
    """One saved result file, or a refusal that names the problem. Both new
    modes take a path typed on the command line, so a failure here is a typo
    or a file written by something else, and the caller needs to hear which."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        refuse("no such result file: %s" % path)
    except json.JSONDecodeError as exc:
        refuse("%s is not JSON: %s" % (path, exc))
    except OSError as exc:
        refuse("cannot read %s: %s" % (path, exc))
    if not isinstance(data, dict):
        refuse("%s is not a result file: its top level is a %s, not an object" % (
            path, type(data).__name__))
    return data


def measured(container: dict) -> dict:
    """One summary plus any measurement kept beside it, as one object to read
    metrics from. The mode comparison keeps its rank numbers next to the
    summary rather than inside it, and a layer 2 file may keep cost and
    latency the same way; the keys named here say what the run was, not how
    it scored, so they are left out."""
    identity_keys = {"set", "index", "index_dir", "mode", "pin", "rows", "summary", "modes",
                     "layer", "prompt_version"}
    source = dict(container.get("summary") or {})
    for key, value in container.items():
        if key not in identity_keys and key not in source:
            source[key] = value
    return source


def result_summaries(data: dict, path: str) -> tuple[str, dict]:
    """The shape of a result file and the summaries in it, keyed by name. A
    single run holds one; --ablate and the mode comparison hold one per mode.
    No metric name is listed here on purpose: layer 2 writes more metrics into
    the same summary object, and the report prints whatever the file carries."""
    if isinstance(data.get("summary"), dict):
        return "single run", {"overall": measured(data)}
    modes = data.get("modes")
    if isinstance(modes, dict):
        named = {name: measured(entry) for name, entry in modes.items()
                 if isinstance(entry, dict) and isinstance(entry.get("summary"), dict)}
        if named:
            return "per-mode comparison", named
    refuse("%s holds no summary to report on: expected a summary object, or a modes object of them" % path)


def is_pair(value) -> bool:
    """A hit/total pair as JSON gives it back: summarize writes tuples and
    json reads them as two-item lists."""
    return (isinstance(value, list) and len(value) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) for v in value))


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def metric_label(key: str) -> str:
    return REPORT_LABELS.get(key, key.replace("_", " "))


def flatten_metrics(summary: dict, prefix: str = "") -> dict:
    """Every metric in a summary, including any nested a level down: layer 1
    writes them flat, and a layer 2 summary may group the evidence checks
    under a key of their own. Pairs and numbers are metrics; a note or a list
    of flags is not, and is left out rather than differenced."""
    metrics = {}
    for key, value in summary.items():
        label = ("%s %s" % (prefix, metric_label(key))).strip()
        if is_pair(value) or is_number(value):
            metrics[label] = value
        elif isinstance(value, dict):
            metrics.update(flatten_metrics(value, label))
    return metrics


def format_metric(value) -> str:
    if is_pair(value):
        return _rate((value[0], value[1]))
    if isinstance(value, float):
        return "%.3f" % value
    return str(value)


def metric_delta(before, after) -> str:
    """The change, in the units the metric is read in: percentage points for
    a rate, plain difference for a count. A changed denominator is named
    beside the change, because a rate over a different number of rows is a
    different measurement and the points between them are not a like move."""
    if is_pair(before) and is_pair(after):
        parts = []
        if before[1] and after[1]:
            parts.append("%+d pts" % (round(100.0 * after[0] / after[1])
                                      - round(100.0 * before[0] / before[1])))
        else:
            parts.append("no rate")
        if before[1] != after[1]:
            parts.append("denominator %d -> %d" % (before[1], after[1]))
        return ", ".join(parts)
    if is_number(before) and is_number(after):
        change = after - before
        if isinstance(before, int) and isinstance(after, int):
            return "%+d" % change
        return "%+.3f" % change
    return "same" if before == after else "changed"


def index_fingerprint(data: dict) -> str:
    """What the file says about the index it was run against. The directory
    name and the corpus counts are both fingerprints; a file may carry
    either, and a report over two different indexes is the failure the notes
    describe, so an unstated one is not treated as a match."""
    parts = []
    if data.get("index_dir"):
        parts.append("dir=%s" % data["index_dir"])
    counts = data.get("index")
    if isinstance(counts, dict):
        parts.extend("%s=%s" % (key, counts[key]) for key in sorted(counts))
    elif counts:
        parts.append(str(counts))
    return " ".join(parts) if parts else "not stated"


def row_ids(data: dict) -> list[str]:
    rows = data.get("rows")
    if not isinstance(rows, list):
        for entry in (data.get("modes") or {}).values():
            if isinstance(entry, dict) and isinstance(entry.get("rows"), list):
                rows = entry["rows"]
                break
    if not isinstance(rows, list):
        return []
    return [str(row.get("id")) for row in rows if isinstance(row, dict)]


def identity(data: dict, shape: str) -> dict:
    ids = row_ids(data)
    return {
        "question set": str(data.get("set") or "not stated"),
        "index": index_fingerprint(data),
        "rows": ("%d rows: %s" % (len(ids), ",".join(ids))) if ids else "not stated",
        "shape": shape,
    }


def print_metrics(metrics: dict, indent: str = "  ") -> None:
    width = max(len(label) for label in metrics)
    for label, value in metrics.items():
        print("%s%-*s  %s" % (indent, width, label, format_metric(value)))


def report(before_path: str, after_path: str) -> None:
    before = load_results(before_path)
    after = load_results(after_path)
    before_shape, before_summaries = result_summaries(before, before_path)
    after_shape, after_summaries = result_summaries(after, after_path)
    before_id = identity(before, before_shape)
    after_id = identity(after, after_shape)
    shared = [name for name in before_summaries if name in after_summaries]

    print("before: %s" % before_path)
    print("after:  %s" % after_path)
    print()

    mismatched = [field for field in before_id if before_id[field] != after_id[field]]
    if not shared:
        mismatched.append("summaries")
        before_id["summaries"] = ", ".join(before_summaries) or "none"
        after_id["summaries"] = ", ".join(after_summaries) or "none"
    if mismatched:
        print("NOT COMPARABLE: these two files did not measure the same thing.")
        for field in mismatched:
            print("  %-13s before: %s" % (field, before_id[field]))
            print("  %-13s after:  %s" % ("", after_id[field]))
        print()
        print("no delta printed. differencing these numbers would read as a change something in the")
        print("pipeline made, when it is a change of what was measured, and a wrong row in the log is")
        print("worse than a missing one. each file's own numbers, undifferenced:")
        for path, summaries in ((before_path, before_summaries), (after_path, after_summaries)):
            print()
            print("%s" % path)
            for name, summary in summaries.items():
                metrics = flatten_metrics(summary)
                if not metrics:
                    continue
                print("  %s" % name)
                print_metrics(metrics, indent="    ")
        sys.exit(2)

    print("both files measured the same thing:")
    for field, value in before_id.items():
        print("  %-13s %s" % (field, value))

    log_rows = []
    for name in shared:
        before_metrics = flatten_metrics(before_summaries[name])
        after_metrics = flatten_metrics(after_summaries[name])
        labels = list(before_metrics) + [extra for extra in after_metrics
                                         if extra not in before_metrics]
        print()
        print("%s" % ("metrics" if name == "overall" else "mode %s" % name))
        width = max(len(label) for label in labels) if labels else 1
        print("  %-*s  %-14s %-14s %s" % (width, "metric", "before", "after", "delta"))
        moved = []
        for label in labels:
            if label not in before_metrics or label not in after_metrics:
                only = "before" if label in before_metrics else "after"
                value = before_metrics.get(label, after_metrics.get(label))
                print("  %-*s  %s" % (width, label, "only in the %s file: %s" % (
                    only, format_metric(value))))
                # A metric one file does not carry is a measurement that
                # started or stopped, which the log row has to say rather than
                # pass over as if nothing about the measurement changed.
                moved.append("%s (%s) %s" % (
                    label, "no longer measured" if only == "before" else "newly measured",
                    format_metric(value)))
                continue
            old, new = before_metrics[label], after_metrics[label]
            print("  %-*s  %-14s %-14s %s" % (width, label, format_metric(old),
                                              format_metric(new), metric_delta(old, new)))
            if old != new:
                moved.append("%s %s -> %s" % (label, format_metric(old), format_metric(new)))
        cell = "; ".join(moved) if moved else "no metric moved"
        if name != "overall":
            cell = "mode %s: %s" % (name, cell)
        log_rows.append(cell)

    only_before = [name for name in before_summaries if name not in after_summaries]
    only_after = [name for name in after_summaries if name not in before_summaries]
    if only_before:
        print()
        print("only in the before file: %s" % ", ".join(only_before))
    if only_after:
        print("only in the after file: %s" % ", ".join(only_after))

    print()
    print("for the table in docs/PROMPT-LOG.md, metric cell filled, the rest for you to write:")
    for cell in log_rows:
        print("| (version) | (what changed) | (the failure that prompted it: row id, one line) | %s | "
              "(keep or revert) |" % cell)


# --- grade: the hand-scored rubric ----------------------------------------


def save_results(path: str, data: dict) -> None:
    """Write through a temporary file beside the original: a session stopped
    part way through a write must not leave the grades from earlier rows
    half written."""
    temporary = path + ".tmp"
    with open(temporary, "w") as fh:
        json.dump(data, fh, indent=1)
    os.replace(temporary, path)


def stored_grade(row: dict) -> dict | None:
    """The grades already in a row. A partly written or hand-edited entry
    counts as ungraded rather than being averaged, so a value that is not a
    score cannot move a mean without anyone noticing."""
    grade = row.get("rubric")
    if not isinstance(grade, dict):
        return None
    scores = {}
    for key, _ in RUBRIC:
        value = grade.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1, 2):
            return None
        scores[key] = value
    return scores


def format_grade(scores: dict) -> str:
    return ", ".join("%s %d" % (key, scores[key]) for key, _ in RUBRIC)


def question_text(data: dict) -> dict:
    """Question text by row id, so the grader reads the question and not just
    its id. Layer 1 rows store scores only, so the text comes from the set the
    run names; a set file that is not there is not fatal."""
    path = SETS.get(str(data.get("set")))
    if not path or not os.path.exists(path):
        return {}
    try:
        return {str(row["id"]): row.get("question", "") for row in read_rows(path)}
    except (OSError, KeyError, json.JSONDecodeError):
        return {}


def trim(text, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def answer_lines(row: dict, path: str, number: int) -> list[str]:
    """Where the grader reads the answer and its citations. A layer 2 row
    carries the answer itself or the path of the file it was written to; a
    layer 1 row carries neither, and saying so is better than a blank line
    the grader has to interpret."""
    answer = row.get("answer")
    if isinstance(answer, str) and answer.strip():
        return ["    answer: " + trim(answer, 600),
                "    full text of the answer: %s, rows[%d]" % (path, number - 1)]
    if isinstance(answer, dict):
        lines = []
        for sentence in answer.get("summary") or []:
            if isinstance(sentence, dict):
                lines.append("    summary: " + trim(sentence.get("text", ""), 300))
            elif isinstance(sentence, str):
                lines.append("    summary: " + trim(sentence, 300))
        for claim in answer.get("claims") or []:
            if not isinstance(claim, dict):
                continue
            lines.append("    %s cites %s: %s" % (
                claim.get("id", "?"), ",".join(claim.get("citations") or []) or "nothing",
                trim(claim.get("text", ""), 200)))
            if claim.get("quote"):
                lines.append("      quote: " + trim(claim["quote"], 200))
        for gap in answer.get("gaps") or []:
            lines.append("    gap stated: " + trim(gap, 200))
        for item in answer.get("not_comparable") or []:
            if isinstance(item, dict):
                lines.append("    not comparable: %s (%s)" % (
                    item.get("dimension", ""), trim(item.get("reason", ""), 160)))
        lines.append("    full text of the answer: %s, rows[%d]" % (path, number - 1))
        return lines
    for key in ("answer_file", "answer_path", "answer_json"):
        if row.get(key):
            return ["    answer and citations: %s" % row[key]]
    context = " (%d chunks in the context)" % row["chunks"] if isinstance(row.get("chunks"), int) else ""
    return ["    answer: not in this file, which holds retrieval scores only%s." % context,
            "    read the answer for %s where that run recorded it before scoring." % row.get("id")]


def read_reply(prompt: str, allowed: set, choices: str) -> str | None:
    """One answer from the grader, or None when the input ends. Anything not
    in the allowed set asks again, so a typo never lands as a score."""
    while True:
        try:
            reply = input(prompt).strip().lower()
        except EOFError:
            print()
            print("input ended; keeping every grade already saved")
            return None
        if reply in allowed:
            return reply
        print("    answer %s" % choices)


def grade(path: str) -> None:
    data = load_results(path)
    rows = data.get("rows")
    if not isinstance(rows, list) or not rows:
        if isinstance(data.get("modes"), dict) and data["modes"]:
            refuse("--grade needs a single run file: %s holds one set of rows per retrieval mode (%s), "
                   "so each question appears once per mode and a single score could not say which "
                   "context it was given for. grade the run file the shipped mode wrote."
                   % (path, ", ".join(sorted(data["modes"]))))
        refuse("%s has no rows to grade: expected a rows list of scored questions" % path)
    if not all(isinstance(row, dict) and row.get("id") for row in rows):
        refuse("%s is not a result file this can grade: every row needs an id" % path)

    questions = question_text(data)
    print("grading %d rows in %s" % (len(rows), path))
    print("four criteria, 0, 1 or 2 each:")
    for key, wording in RUBRIC:
        print("  %-13s %s" % (key, wording))
    print("s leaves a row ungraded to come back to, q stops. every graded row is saved as it is entered,")
    print("so a session stopped part way is resumed by running the same command again.")

    stopped = False
    for number, row in enumerate(rows, start=1):
        print()
        print("[%d/%d] %s" % (number, len(rows), row["id"]))
        print("    question: %s" % (questions.get(str(row["id"])) or "not in the question set file"))
        status = row.get("status", "not stated")
        if "status_ok" in row:
            status += " (the expected status)" if row["status_ok"] else " (not the expected status)"
        print("    status: %s" % status)
        for line in answer_lines(row, path, number):
            print(line)

        existing = stored_grade(row)
        if existing:
            reply = read_reply("    already graded: %s. enter keeps it, r regrades, q stops: "
                               % format_grade(existing), {"", "r", "q"},
                               "with enter to keep, r to regrade, or q to stop")
            if reply is None or reply == "q":
                stopped = True
                break
            if reply == "":
                continue

        scores = {}
        for key, wording in RUBRIC:
            reply = read_reply("    %s (0-2): " % wording, {"0", "1", "2", "s", "q"},
                               "0, 1 or 2, or s to leave this row ungraded, or q to stop")
            if reply is None or reply == "q":
                stopped = True
                break
            if reply == "s":
                scores = None
                break
            scores[key] = int(reply)
        if stopped:
            break
        if scores is None:
            print("    left ungraded")
            continue
        row["rubric"] = scores
        save_results(path, data)
        print("    saved: %s" % format_grade(scores))

    graded = [stored_grade(row) for row in rows]
    graded = [scores for scores in graded if scores]
    print()
    if stopped:
        print("stopped early. run the same command again to grade the rest.")
    print("graded %d of %d rows in %s" % (len(graded), len(rows), path))
    width = max(len(wording) for _, wording in RUBRIC)
    for key, wording in RUBRIC:
        average = "%.2f" % statistics.mean(s[key] for s in graded) if graded else "-"
        print("  %-*s  %s" % (width, wording, average))
    overall = ("%.2f" % statistics.mean(s[key] for s in graded for key, _ in RUBRIC)) if graded else "-"
    print("  %-*s  %s" % (width, "overall", overall))
    print("ungraded rows: %d" % (len(rows) - len(graded)))


def checks_as_numbers(checks) -> dict:
    """The EvidenceChecks pairs as plain numbers a summary can add up."""
    if checks is None:
        return {}
    d = dataclasses.asdict(checks) if dataclasses.is_dataclass(checks) else dict(checks)
    out = {}
    for key, value in d.items():
        if isinstance(value, (tuple, list)) and len(value) == 2 and all(isinstance(v, int) for v in value):
            out[key + "_hit"], out[key + "_n"] = value
        elif isinstance(value, int):
            out[key] = value
        elif isinstance(value, list):
            out[key + "_count"] = len(value)
    return out


def layer2_row(row: dict, payload: dict) -> dict:
    """Layer 1 scored on the same payload, plus what the one request cost and
    what the checks established. Nothing here says an answer is right; the
    rubric is the only measure of that."""
    score = score_row(row, payload)
    timing = payload.get("timing_ms") or {}
    usage = payload.get("usage") or {}
    answer = payload.get("answer")
    score.update({
        "layer": 2,
        "prompt_version": payload.get("prompt_version"),
        "model": payload.get("model"),
        "backend": payload.get("backend"),
        "request_id": payload.get("request_id"),
        "llm_attempts": payload.get("llm_attempts", 0),
        "llm_completed": payload.get("llm_completed", 0),
        "llm_error": payload.get("llm_error"),
        "parse_ok": bool(answer is not None and not payload.get("llm_error")),
        "n_claims": len(getattr(answer, "claims", []) or []) if answer is not None else 0,
        "timing_ms": timing,
        "total_ms": sum(v for v in timing.values() if isinstance(v, (int, float))),
        "input_tokens": usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
                        + usage.get("cache_read_input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cost_usd": payload.get("cost_usd"),
        "checks": checks_as_numbers(payload.get("checks")),
        "replayed": bool(payload.get("replayed")),
        # The reply itself, so the outputs can be reviewed and graded from
        # this file alone; a results file with only counts in it cannot be
        # read for whether the answer was any good.
        "answer": answer.model_dump() if answer is not None and hasattr(answer, "model_dump") else None,
        "raw_text": payload.get("raw_text"),
        "flags": [dict(f) for f in getattr(payload.get("checks"), "flags", []) or []],
        "notes": [dict(n) for n in getattr(payload.get("checks"), "notes", []) or []],
    })
    return score


def summarize_layer2(scores: list[dict]) -> dict:
    """Layer-1 summary plus request cost and what the checks established,
    over the rows that made a request. Latency is reported as median and
    worst, never a mean, since one slow request is the demo risk."""
    summary = summarize(scores)
    called = [s for s in scores if s.get("llm_attempts", 0) > 0]
    latencies = sorted(s["total_ms"] for s in called)
    agg = {}
    for key in ("quotes_located", "figures_in_quote", "columns_matched", "units_matched"):
        hit = sum(s["checks"].get(key + "_hit", 0) for s in called)
        n = sum(s["checks"].get(key + "_n", 0) for s in called)
        agg[key] = (hit, n)
    summary.update({
        "requests": len(called),
        "parse_first_try": (sum(1 for s in called if s["parse_ok"]), len(called)),
        "latency_ms_median": latencies[len(latencies) // 2] if latencies else 0,
        "latency_ms_worst": latencies[-1] if latencies else 0,
        "under_30s": (sum(1 for l in latencies if l <= 30000), len(latencies)),
        "mean_input_tokens": round(statistics.mean(s["input_tokens"] for s in called)) if called else 0,
        "mean_output_tokens": round(statistics.mean(s["output_tokens"] for s in called)) if called else 0,
        "total_cost_usd": round(sum(s["cost_usd"] or 0 for s in called), 4),
        "flags_total": sum(s["checks"].get("flags_count", 0) for s in called),
        "unlinked_total": sum(s["checks"].get("unlinked_count", 0) for s in called),
        **agg,
    })
    return summary


def layer2_line(summary: dict) -> str:
    return ("requests %d | parse first try %s | latency median %.1fs worst %.1fs | under 30 s %s | "
            "quotes located %s | figures in quote %s | columns matched %s | units matched %s | "
            "flags %d | mean tokens in %d out %d | cost $%.2f" % (
                summary["requests"], _rate(summary["parse_first_try"]),
                summary["latency_ms_median"] / 1000, summary["latency_ms_worst"] / 1000,
                _rate(summary["under_30s"]), _rate(summary["quotes_located"]),
                _rate(summary["figures_in_quote"]), _rate(summary["columns_matched"]),
                _rate(summary["units_matched"]), summary["flags_total"],
                summary["mean_input_tokens"], summary["mean_output_tokens"], summary["total_cost_usd"]))


def run_layer2(args) -> None:
    """One model request per row on the configured backend. Refuses the fake
    backend unless asked, since a fake run would write a results file that
    reads like a measurement."""
    import prompts
    if config.LLM_MODEL_BACKEND == "fake" and not args.allow_fake:
        print("refusing a layer-2 run on the fake backend: set LLM_MODEL_BACKEND=anthropic, or pass "
              "--allow-fake to exercise the harness with stand-in answers")
        sys.exit(2)
    rows = read_rows(SETS[args.set])
    loaded = index_module.load(args.index_dir)
    companies = index_module.load_registry(config.COMPANIES_FILE)
    version = prompts.PROMPT_VERSION
    out = args.out or os.path.join(ROOT, "eval", "results", "%s-%s.json" % (args.set, version))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print("layer 2 | set %s | prompt %s | backend %s | model %s | retrieval %s" % (
        args.set, version, config.LLM_MODEL_BACKEND, config.LLM_MODEL, loaded.effective_mode()[0]))
    scores = []
    for row in rows:
        payload = ask(row["question"], loaded, companies, dry_run=False)
        score = layer2_row(row, payload)
        scores.append(score)
        print("%-4s %-18s %6.1fs  claims %2d  parse %s  checks q%s f%s c%s u%s flags %d  $%.3f" % (
            score["id"], score["status"], score["total_ms"] / 1000, score["n_claims"],
            "ok" if score["parse_ok"] else ("n/a" if score["llm_attempts"] == 0 else "FAIL"),
            _pair(score["checks"], "quotes_located"), _pair(score["checks"], "figures_in_quote"),
            _pair(score["checks"], "columns_matched"), _pair(score["checks"], "units_matched"),
            score["checks"].get("flags_count", 0), score["cost_usd"] or 0), flush=True)
        if args.save_fixtures and payload.get("llm_result") is not None and payload.get("answer") is not None:
            os.makedirs(args.save_fixtures, exist_ok=True)
            path = os.path.join(args.save_fixtures, row["id"] + ".json")
            llm_client.save_fixture(path, row["question"], payload["llm_result"], version)
            print("     fixture saved: %s" % path)
    summary = summarize_layer2(scores)
    print()
    print(summary_line(summary))
    print(layer2_line(summary))
    with open(out, "w") as fh:
        json.dump({"set": args.set, "index_dir": args.index_dir, "layer": 2, "prompt_version": version,
                   "backend": config.LLM_MODEL_BACKEND, "model": config.LLM_MODEL,
                   "mode": loaded.effective_mode()[0], "rows": scores, "summary": summary}, fh, indent=1)
    print("wrote", out)


def _pair(checks: dict, key: str) -> str:
    return "%d/%d" % (checks.get(key + "_hit", 0), checks.get(key + "_n", 0))


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--retrieval-only", action="store_true", help="layer 1: no model request")
    parser.add_argument("--set", default="tuning", choices=sorted(SETS))
    parser.add_argument("--index-dir", default=config.INDEX_DIR)
    parser.add_argument("--out", default=None)
    parser.add_argument("--ablate", action="store_true", help="bm25 / dense / hybrid / hybrid without pinning")
    parser.add_argument("--final", action="store_true", help="allow the held-out set (one final run)")
    parser.add_argument("--save-fixtures", default=None, metavar="DIR",
                        help="layer 2: store each parsed reply so the fake backend can replay it")
    parser.add_argument("--allow-fake", action="store_true",
                        help="layer 2: permit the fake backend (harness check only, not a measurement)")
    parser.add_argument("--report", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="print the change between two saved result files")
    parser.add_argument("--grade", metavar="RESULTS",
                        help="take the hand-graded rubric over a saved result file")
    args = parser.parse_args(argv)

    # The two reading modes come first: neither one builds or loads an index,
    # and neither one needs the held-out guard, which is about running rows
    # rather than reading a file that was already written.
    if args.report and args.grade:
        print("one mode at a time: --report reads two files, --grade writes into one")
        sys.exit(2)
    if args.report:
        report(args.report[0], args.report[1])
        return
    if args.grade:
        grade(args.grade)
        return

    if args.set == "heldout" and not args.final:
        print("refusing --set heldout without --final: the held-out rows are scored once, with the "
              "final prompt and rules; reading them while tuning would turn them into a second "
              "tuning set.")
        sys.exit(2)
    if not args.retrieval_only:
        run_layer2(args)
        return

    rows = read_rows(SETS[args.set])
    loaded = index_module.load(args.index_dir)
    companies = index_module.load_registry(config.COMPANIES_FILE)
    out = args.out or os.path.join(ROOT, "eval", "results",
                                   "retrieval_ablation.json" if args.ablate else "retrieval.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if args.ablate:
        results = {}
        print("| mode | section hit | file coverage | passage hit | abstain | mean tokens |")
        print("|---|---|---|---|---|---|")
        for name, mode, pin in ABLATION:
            scores = run_set(rows, loaded, companies, mode, pin)
            summary = summarize(scores)
            results[name] = {"mode": mode, "pin": pin, "rows": scores, "summary": summary}
            print("| %s | %s | %s | %s | %s | %d |" % (
                name, _rate(summary["section_hit"]), _rate(summary["file_coverage"]),
                _rate(summary["passage_hit"]), _rate(summary["abstain"]), summary["mean_tokens"]))
        with open(out, "w") as fh:
            json.dump({"set": args.set, "index_dir": args.index_dir, "modes": results}, fh, indent=1)
        print("wrote", out)
        return

    scores = run_set(rows, loaded, companies, None, True)
    summary = summarize(scores)
    print(table(scores))
    print()
    print(summary_line(summary))
    with open(out, "w") as fh:
        json.dump({"set": args.set, "index_dir": args.index_dir, "mode": loaded.effective_mode()[0],
                   "pin": True, "rows": scores, "summary": summary}, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
