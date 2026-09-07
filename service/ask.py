"""Answer one question about the filings, or say why it cannot be answered.

  python service/ask.py "<question>" [--dry-run] [--json] [--index-dir DIR]

ask() is the whole pipeline in order: the deterministic scope (rules), the
retrieval inside that scope (retrieve), and, from milestone 4, one model
request whose reply is parsed and checked against the excerpts. The first
two steps are prepare(), which runs once per request: the CLI prints the
scope from its result and, unless --dry-run, continues from the same
result to the model step, so nothing is planned or retrieved twice. A
question the corpus cannot answer stops at the plan with a refusal payload
and never reaches the model. A dry run stops after retrieval and returns
the scope, the coverage block and the numbered excerpts.
"""

import argparse
import dataclasses
import json
import sys

import config
import index as index_module
import retrieve
import rules
from models import Context, Plan


@dataclasses.dataclass
class Prepared:
    """Everything decided before the model step. `context` is None when
    the plan refused the question, and then `coverage` is the refusal's
    coverage text and `budget` is None."""

    plan: Plan
    context: Context | None
    coverage: str
    budget: dict | None


def prepare(question: str, index, companies: dict, mode: str = "hybrid", pin: bool = True) -> Prepared:
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


def model_step(prepared: Prepared, model=None) -> dict:
    """One model request over the prepared context, parsed and checked."""
    # milestone 4 plugs in here: prompt -> one model request -> parsing -> resolver
    raise NotImplementedError("model call arrives in milestone 4")


def ask(question: str, index, companies: dict, model=None, dry_run: bool = False,
        mode: str = "hybrid", pin: bool = True) -> dict:
    """One request, top to bottom: scope and retrieval, then either the
    payload that describes them or the model step over them."""
    prepared = prepare(question, index, companies, mode=mode, pin=pin)
    if dry_run or prepared.context is None:
        return scope_payload(prepared)
    return model_step(prepared, model)


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


def to_json(payload: dict) -> str:
    plain = dict(payload)
    plain["plan"] = dataclasses.asdict(payload["plan"])
    return json.dumps(plain, indent=1)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("question")
    parser.add_argument("--dry-run", action="store_true", help="stop after retrieval; print the scope")
    parser.add_argument("--json", action="store_true", help="dump the payload as JSON")
    parser.add_argument("--index-dir", default=config.INDEX_DIR)
    args = parser.parse_args(argv)

    # One load per process; the dense matrix is memory-mapped, so this is
    # the chunk list and the lexical index.
    loaded = index_module.load(args.index_dir)
    companies = index_module.load_registry(config.COMPANIES_FILE)

    # One request's flow: plan and retrieve once, show the scope, then
    # carry the same prepared state into the model step.
    prepared = prepare(args.question, loaded, companies)
    payload = scope_payload(prepared)
    print(to_json(payload) if args.json else describe(payload))
    if args.dry_run or prepared.context is None:
        return
    try:
        model_step(prepared)
    except NotImplementedError as stop:
        print("stopped before the model request:", stop)
        sys.exit(3)


if __name__ == "__main__":
    main(sys.argv[1:])
