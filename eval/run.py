"""Evaluate the pipeline on a labelled question set.

  python -m eval.run --retrieval-only --set tuning [--index-dir index]
                     [--out eval/results/retrieval.json] [--ablate]

Layer 1 (retrieval only, no model request): for each row, run ask() as a
dry run and score what reached the context: the resolved tickers against
the expected set, each expected section and file against the ones used,
every expected passage (expected_substrings, or the older single
expected_substring) against the rendered excerpts, the status of refusal
rows, and the token estimate against the budget. --ablate repeats the run
with lexical only, dense only, hybrid, and hybrid without pinning, and
prints one summary row per mode; dense modes need an index built with
--dense 1, which is why the tuning-file index is the usual target.

The held-out set is read only with --final, so no prompt or rule is tuned
against it by accident.
"""

import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service"))

import config  # noqa: E402
import index as index_module  # noqa: E402
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
    return ("section hit rate %s | file coverage %s | passage hit rate %s | abstain correctness %s | "
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


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--retrieval-only", action="store_true", help="layer 1: no model request")
    parser.add_argument("--set", default="tuning", choices=sorted(SETS))
    parser.add_argument("--index-dir", default=config.INDEX_DIR)
    parser.add_argument("--out", default=None)
    parser.add_argument("--ablate", action="store_true", help="bm25 / dense / hybrid / hybrid without pinning")
    parser.add_argument("--final", action="store_true", help="allow the held-out set (one final run)")
    args = parser.parse_args(argv)

    if args.set == "heldout" and not args.final:
        print("refusing --set heldout without --final: the held-out rows are scored once, with the "
              "final prompt and rules; reading them while tuning would turn them into a second "
              "tuning set.")
        sys.exit(2)
    if not args.retrieval_only:
        print("only --retrieval-only is implemented in this milestone; the model request arrives in milestone 4")
        sys.exit(2)

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

    scores = run_set(rows, loaded, companies, "hybrid", True)
    summary = summarize(scores)
    print(table(scores))
    print()
    print(summary_line(summary))
    with open(out, "w") as fh:
        json.dump({"set": args.set, "index_dir": args.index_dir, "mode": "hybrid", "pin": True,
                   "rows": scores, "summary": summary}, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
