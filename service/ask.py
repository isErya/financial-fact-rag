"""Answer one question about the filings, or say why it cannot be answered.

  python service/ask.py "<question>" [--dry-run] [--json] [--index-dir DIR] [--save-fixture PATH]

ask() is the whole pipeline in order: the deterministic scope (rules), the
retrieval inside that scope (retrieve), and one model request whose reply
is parsed and checked against the excerpts (prompts, llm_client, parsing,
resolver). The first two steps are prepare(), which runs once per request:
the CLI prints the scope from its result and, unless --dry-run, continues
from the same result to the model step, so nothing is planned or retrieved
twice. A question the corpus cannot answer stops at the plan with a refusal
payload and never reaches the model. A dry run stops after retrieval and
returns the scope, the coverage block and the numbered excerpts.

The model step makes exactly one request: a RequestGuard created per ask
is handed to the backend, and a reply that does not parse is reported as
llm_error with its raw text rather than repaired by a second request.
--save-fixture stores the reply so the fake backend can replay it.
"""

import argparse
import dataclasses
import json
import sys
import time

import config
import index as index_module
import llm_client
import parsing
import prompts
import resolver
import retrieve
import rules
from models import Answer, Context, Plan


@dataclasses.dataclass
class Prepared:
    """Everything decided before the model step. `context` is None when
    the plan refused the question, and then `coverage` is the refusal's
    coverage text and `budget` is None."""

    plan: Plan
    context: Context | None
    coverage: str
    budget: dict | None


def prepare(question: str, index, companies: dict, mode: str | None = None, pin: bool = True) -> Prepared:
    """The plan, and the context when the plan allows one. `mode` and
    `pin` exist for the retrieval ablation and keep their defaults
    everywhere else."""
    plan = rules.make_plan(question, companies, index.files, rules.DEFAULT_BUDGET)
    if plan.status != "ok":
        return Prepared(plan, None, rules.refusal_coverage(plan, companies), None)
    context = retrieve.retrieve(plan, index, mode=mode, pin=pin)
    budget = {"tokens_est": context.n_tokens, "budget_tokens": context.budget_tokens,
              "chunks_dropped": context.chunks_dropped}
    return Prepared(plan, context, context.coverage, budget)


def scope_payload(prepared: Prepared) -> dict:
    """The refusal or dry-run payload: what is known before any request."""
    plan, context = prepared.plan, prepared.context
    if context is None:
        return {"status": plan.status, "plan": plan, "coverage": prepared.coverage,
                "llm_attempts": 0, "llm_completed": 0}
    return {"status": "ok", "plan": plan, "context": context_rows(context),
            "rendered": context.rendered, "coverage": prepared.coverage, "budget": prepared.budget,
            "llm_attempts": 0, "llm_completed": 0}


def model_step(prepared: Prepared, model=None, question: str = "", companies: dict | None = None,
               timing_ms: dict | None = None) -> dict:
    """One model request over the prepared context, parsed and checked.

    `model` defaults to the backend the environment names; `companies` is
    the registry, whose tickers the gap check knows; `timing_ms` carries the
    stages that ran before this one.
    """
    plan, context = prepared.plan, prepared.context
    model = model or llm_client.from_env()
    timing = dict(timing_ms or {})
    chunks_by_cid = dict(zip(context.cids, context.chunks))

    started = time.monotonic()
    system, user = prompts.render(question, context, plan)
    timing["render"] = _ms(started)

    # The guard is created here, once per ask, so the count of requests
    # this ask made is the guard's count and a second one raises.
    started = time.monotonic()
    guard = llm_client.RequestGuard()
    result = model.generate(system, user, Answer, guard=guard)
    timing["model"] = _ms(started)

    started = time.monotonic()
    answer = result.parsed
    llm_error = result.error
    if answer is None:
        try:
            answer = parsing.parse_answer(result.raw_text)
            llm_error = None
        except parsing.ParseError as exc:
            llm_error = "%s; %s" % (llm_error, exc) if llm_error else str(exc)
    timing["parse"] = _ms(started)

    started = time.monotonic()
    checks = None
    if answer is not None:
        known = set((companies or {}).get("companies", {})) if companies else None
        checks = resolver.check(answer, context, chunks_by_cid, plan, known)
    timing["check"] = _ms(started)

    budget = dict(prepared.budget)
    budget["input_tokens_actual"] = result.usage.get("input_tokens")
    return {
        "status": "ok", "plan": plan, "context": context_rows(context), "rendered": context.rendered,
        "coverage": prepared.coverage, "budget": budget,
        "answer": answer, "checks": checks, "llm_error": llm_error, "raw_text": result.raw_text,
        "usage": result.usage, "cost_usd": cost_usd(result.usage, result.model),
        "timing_ms": timing, "llm_attempts": result.attempts, "llm_completed": result.completed,
        "request_id": result.request_id, "model": result.model, "backend": result.backend,
        "stop_reason": result.stop_reason, "prompt": {"system": system, "user": user},
        "prompt_version": prompts.PROMPT_VERSION, "replayed": result.replayed,
        "llm_result": result,
    }


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def cost_usd(usage: dict, model: str) -> float | None:
    """The bill for one request from config.PRICES, or None for a model
    without a price line (the fake backend, an id added since the prices
    were read)."""
    price = config.PRICES.get(model)
    if not price or not usage:
        return None
    rate_in, rate_out = price["input"], price["output"]
    dollars = (
        (usage.get("input_tokens") or 0) * rate_in
        + (usage.get("cache_creation_input_tokens") or 0) * rate_in * config.CACHE_WRITE_FACTOR
        + (usage.get("cache_read_input_tokens") or 0) * rate_in * config.CACHE_READ_FACTOR
        + (usage.get("output_tokens") or 0) * rate_out
    ) / 1e6
    return round(dollars, 6)


def ask(question: str, index, companies: dict, model=None, dry_run: bool = False,
        mode: str | None = None, pin: bool = True) -> dict:
    """One request, top to bottom: scope and retrieval, then either the
    payload that describes them or the model step over them."""
    started = time.monotonic()
    prepared = prepare(question, index, companies, mode=mode, pin=pin)
    timing = {"plan_and_retrieve": _ms(started)}
    if dry_run or prepared.context is None:
        payload = scope_payload(prepared)
        payload["timing_ms"] = timing
        return payload
    return model_step(prepared, model, question=question, companies=companies, timing_ms=timing)


def context_rows(context: Context) -> list[dict]:
    """One row per excerpt; `pinned` names why a seat was placed
    ("section lead", "row label") and is None for a chunk search found."""
    return [{"cid": cid, "chunk_id": c.chunk_id, "file": c.file, "ticker": c.ticker, "form": c.form,
             "fiscal_label": c.fiscal_label, "item": c.item, "note_title": c.note_title, "kind": c.kind,
             "seq": c.seq, "n_tokens": c.n_tokens, "pinned": context.pinned_reason.get(c.chunk_id),
             "header": c.header}
            for cid, c in zip(context.cids, context.chunks)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def describe(payload: dict) -> str:
    """The interpreted scope, as printed before any model request."""
    plan = payload["plan"]
    out = []
    out.append("status: %s | period: %s | comparison: %s | timeline: %s" % (
        payload["status"], plan.period_mode, "yes" if plan.comparison else "no",
        "yes" if plan.timeline else "no"))
    out.append("companies:")
    for c in plan.companies:
        out.append("  %-5s %-36s matched \"%s\"%s" % (
            c["ticker"], c["name"], c["matched_alias"], "  (stale)" if c["ticker"] in plan.stale else ""))
    if not plan.companies:
        out.append("  (none)")
    out.append("unresolved names: %s" % (", ".join(plan.unresolved) or "(none)"))
    if payload["status"] != "ok":
        out.append("coverage:")
        out.extend("  " + line for line in payload["coverage"].splitlines())
        return "\n".join(out)
    out.append("buckets:")
    for b in plan.buckets:
        out.append("  %s %s (period end %s)" % (b["ticker"], b["label"], b["period_end"]))
        for f in b["files"]:
            ended = "fiscal year ended" if f["form"] == "10-K" else "quarter ended"
            out.append("    %-19s %-40s %s %-10s %s %s, filed %s" % (
                f["reason"], f["file"], f["form"], f["fiscal_label"], ended, f["period_end"], f["filing_date"]))
    weights = ", ".join("%s %.1f" % (k, v) for k, v in plan.sections.items() if k != "*")
    out.append("section weights: %s, other %.1f | note intent: %s" % (
        weights, plan.sections.get("*", 0.0), plan.note_intent or "(none)"))
    out.append("sub-queries:")
    for n, q in enumerate(plan.sub_queries, 1):
        out.append("  %d. %s" % (n, q))
    out.append("quotas:")
    for q in plan.quotas:
        out.append("  %s / %s: %d chunks, pinned positions %s, row-label seats %s" % (
            q["ticker"], q["bucket"], q["chunks"], q["pinned"], q.get("row_label", [])))
    budget = payload["budget"]
    out.append("budget: %d tokens; context %d tokens in %d chunks; %d candidates dropped" % (
        budget["budget_tokens"], budget["tokens_est"], len(payload["context"]), budget["chunks_dropped"]))
    out.append("coverage:")
    out.extend("  " + line for line in payload["coverage"].splitlines())
    out.append("context:")
    for row in payload["context"]:
        out.append("  [%s] %s (%d tokens%s)" % (
            row["cid"], row["header"], row["n_tokens"], ", pinned: %s" % row["pinned"] if row["pinned"] else ""))
    return "\n".join(out)


def describe_checks(checks) -> list[str]:
    """The evidence-check counts, then the flags and the notes. Every pair
    reads matched / checkable with the count of items the check could not run
    on beside it, never folded in: an unchecked item is never a pass."""
    lines = ["evidence checks: quotes located %d/%d | figures in quote %d/%d, unchecked %d | "
             "columns matched %d/%d, unverified %d | units matched %d/%d, unchecked %d" % (
                 *checks.quotes_located, *checks.figures_in_quote, checks.figures_unchecked,
                 *checks.columns_matched, checks.columns_unverified,
                 *checks.units_matched, checks.units_unchecked)]
    for name, rows in (("flagged", checks.flags), ("unchecked and noted", checks.notes)):
        if not rows:
            lines.append("  %s: (none)" % name)
        for row in rows:
            source = " [source: %s]" % row["source_string"] if row["source_string"] else ""
            lines.append("  - %s %s: %s%s" % (row["kind"], row["where"], row["detail"], source))
    return lines


def describe_answer(payload: dict) -> str:
    """The answer as the reader sees it: summary, table, what is not
    comparable, gaps, then the evidence-check counts, flags and notes."""
    out = []
    out.append("answer: prompt %s | %s %s | %d attempt(s), %d completed | replayed: %s%s" % (
        payload["prompt_version"], payload["backend"], payload["model"], payload["llm_attempts"],
        payload["llm_completed"], "yes" if payload["replayed"] else "no",
        " | request %s" % payload["request_id"] if payload["request_id"] else ""))
    answer = payload["answer"]
    if answer is None:
        out.append("  no answer: %s" % payload["llm_error"])
        out.append("  raw text (first 600 chars): %s" % (payload["raw_text"] or "")[:600])
    else:
        out.append("summary:")
        for s in answer.summary:
            out.append("  - %s [%s]" % (s.text, ", ".join(s.claim_ids) or "no claim"))
        out.append("claims:")
        for c in answer.claims:
            out.append("  %s %s | %s | %s %s | cites %s" % (
                c.id, c.text, ", ".join(c.tickers) or "-", c.period_end, c.period_kind,
                ", ".join(c.citations) or "nothing"))
            out.append("     quote: %s" % c.quote)
        out.append("table:" if answer.table else "table: (none)")
        for row in answer.table:
            out.append("  %s" % row.dimension)
            for cell in row.cells:
                out.append("    %-6s %s [%s]" % (cell.column, cell.text, ", ".join(cell.claim_ids) or "no claim"))
        out.append("not comparable:" if answer.not_comparable else "not comparable: (none)")
        for nc in answer.not_comparable:
            out.append("  - %s (%s): %s" % (nc.dimension, ", ".join(nc.tickers), nc.reason))
        out.append("gaps:" if answer.gaps else "gaps: (none)")
        for gap in answer.gaps:
            out.append("  - %s" % gap)
        out.extend(describe_checks(payload["checks"]))
    usage = payload["usage"]
    cost = payload["cost_usd"]
    price = config.PRICES.get(payload["model"], {})
    out.append("usage: input %s, output %s, cache write %s, cache read %s | cost %s | timing ms %s" % (
        usage.get("input_tokens"), usage.get("output_tokens"), usage.get("cache_creation_input_tokens"),
        usage.get("cache_read_input_tokens"),
        "$%.4f (prices checked %s)" % (cost, price.get("checked")) if cost is not None else "n/a",
        ", ".join("%s %d" % kv for kv in payload["timing_ms"].items())))
    return "\n".join(out)


def to_json(payload: dict) -> str:
    plain = dict(payload)
    plain["plan"] = dataclasses.asdict(payload["plan"])
    if isinstance(plain.get("answer"), Answer):
        plain["answer"] = plain["answer"].model_dump()
    if plain.get("checks") is not None:
        plain["checks"] = dataclasses.asdict(plain["checks"])
    plain.pop("llm_result", None)
    return json.dumps(plain, indent=1)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("question")
    parser.add_argument("--dry-run", action="store_true", help="stop after retrieval; print the scope")
    parser.add_argument("--json", action="store_true", help="dump the payload as JSON")
    parser.add_argument("--index-dir", default=config.INDEX_DIR)
    parser.add_argument("--save-fixture", default=None, metavar="PATH",
                        help="store the model reply as a fixture the fake backend replays")
    args = parser.parse_args(argv)

    # One load per process; the dense matrix is memory-mapped, so this is
    # the chunk list and the lexical index.
    loaded = index_module.load(args.index_dir)
    companies = index_module.load_registry(config.COMPANIES_FILE)

    # One request's flow: plan and retrieve once, show the scope, then
    # carry the same prepared state into the model step.
    started = time.monotonic()
    prepared = prepare(args.question, loaded, companies)
    timing = {"plan_and_retrieve": _ms(started)}
    payload = scope_payload(prepared)
    if not args.json:
        print(describe(payload))
    if args.dry_run or prepared.context is None:
        if args.json:
            print(to_json(payload))
        return
    try:
        payload = model_step(prepared, question=args.question, companies=companies, timing_ms=timing)
    except (llm_client.ModelUnavailable, llm_client.ModelProtocolError, llm_client.ModelConfigError) as exc:
        print("model request failed (%s): %s" % (type(exc).__name__, exc))
        sys.exit(4)
    print(to_json(payload) if args.json else describe_answer(payload))
    if args.save_fixture:
        llm_client.save_fixture(args.save_fixture, args.question, payload["llm_result"], payload["prompt_version"])
        print("saved fixture:", args.save_fixture)


if __name__ == "__main__":
    main(sys.argv[1:])
