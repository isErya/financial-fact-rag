"""Prompt templates for the one model request an ask makes.

The policy is the system message. The question and the excerpts are text
the service did not write (a filing can hold a sentence shaped like an
instruction, and so can a question), so they travel in the user message
inside named blocks and the system message is the only place instructions
live. Every version that was ever run stays in PROMPTS under its version
key, so a stored result can name the exact prompt that produced it;
PROMPT_VERSION is the one ask.py renders. Later versions are added here
during iteration and logged in docs/PROMPT-LOG.md with the failure that
prompted them.
"""

from models import Context, Plan

PROMPT_VERSION = "v1"

# The schema hint spells out the Answer fields in words. The request also
# carries the JSON schema itself (llm_client sets it on the request), so the
# hint's job is the meaning of each field: what a claim is, what a quote
# must be, which period_kind values exist.
SCHEMA_HINT = """The required structure is a JSON object with these fields:
- summary: list of {text, claim_ids}. Two to six sentences that answer the question. Each sentence names the ids of the claims that back it.
- claims: list of {id, text, tickers, period_end, period_kind, citations, quote}. One claim per fact. id is "K1", "K2", ... in order. tickers lists the companies the fact is about. period_end is the date the figure is as of or the period ends on, as YYYY-MM-DD. period_kind is one of quarter, nine_months, fiscal_year, point_in_time. citations lists the excerpt ids the fact comes from, such as "C3". quote is a passage copied word for word from one cited excerpt (a table row or a sentence, at most 40 words) that carries every figure the claim text states.
- table: list of {dimension, cells: [{column, text, claim_ids}]}. One row per metric compared and one cell per company (column is the ticker). Empty when the question compares nothing.
- not_comparable: list of {dimension, tickers, reason}. Metrics the excerpts report on different bases, for different periods, or for only some of the companies.
- gaps: list of strings. What the question asks for that the excerpts do not hold. Describe the missing item; never supply a substitute figure or a subject the excerpts do not cover.
Figures keep the units the excerpt states. A figure taken from a table must come from the column whose period the claim names."""

PROMPTS = {
    "v1": {
        "system": (
            "You are an analyst supporting a private equity deal team. Answer the question using "
            "only the excerpts. Cite excerpt ids. Return the required structure.\n\n" + SCHEMA_HINT
        ),
        "user": (
            "<coverage>\n{coverage}\n</coverage>\n\n"
            "<question>\n{question}\n</question>\n\n"
            "<excerpts>\n{excerpts}\n</excerpts>\n\n"
            "Answer in the required structure."
        ),
    },
}


def fence(text: str, tag: str) -> str:
    """Keep untrusted text from closing its own block: a literal closing
    tag inside the text is broken with a space so the model still sees one
    block. A filing quoted from such a spot would then differ from the
    corpus by that one space, which the evidence checks tolerate."""
    return text.replace("</%s>" % tag, "< /%s>" % tag)


def render(question: str, context: Context, plan: Plan, version: str = PROMPT_VERSION) -> tuple[str, str]:
    """(system_text, user_text) for one request. The coverage block opens
    with the tickers in scope so the model has the exact strings the
    claims' tickers field expects."""
    template = PROMPTS[version]
    scope = ", ".join("%s (%s)" % (c["ticker"], c["name"]) for c in plan.companies) or "(none)"
    coverage = "Companies in scope: %s\n%s" % (scope, context.coverage)
    user = template["user"].format(
        coverage=fence(coverage, "coverage"),
        question=fence(question.strip(), "question"),
        excerpts=fence(context.rendered, "excerpts"),
    )
    return template["system"], user
