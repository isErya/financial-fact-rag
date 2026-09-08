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

import os

from models import Context, Plan

# Read from the environment so an evaluation run can select a version
# without editing this file; the shipped default is the version the log
# names as the winner.
PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "v3")

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
    # CANDIDATE, not yet an iteration: PROMPT_VERSION still points at v1 and
    # nothing in docs/PROMPT-LOG.md claims this version until it has been run
    # against the tuning set and the numbers written down by hand.
    #
    # Why it exists: v1 sets no shape, and measured on the panel's own example
    # question (three companies compared on risk) it produced 32 claims and
    # 37,555 output tokens in 278 seconds on the mid-tier model. A demo cannot
    # wait four and a half minutes, and output length is what costs the time.
    # v2 changes one thing, the shape of the answer, and leaves every grounding
    # rule of v1 untouched so the two are comparable.
    "v2": {
        "system": (
            "You are an analyst supporting a private equity deal team. Answer the question using "
            "only the excerpts. Cite excerpt ids. Return the required structure.\n\n" + SCHEMA_HINT +
            "\n\nKeep the answer short enough to read on one screen. Write at most four claims "
            "per company and at most twelve in total, choosing the ones that carry the figures and "
            "the disclosures the question asks about. Write at most one summary sentence per "
            "company plus one closing sentence. Give the table at most five rows. Say less rather "
            "than repeating a figure that is already in the table. None of this loosens the rules "
            "above: a shorter answer still cites, still quotes, still names the period and the "
            "units, and still records what the excerpts do not support."
        ),
        "user": (
            "<coverage>\n{coverage}\n</coverage>\n\n"
            "<question>\n{question}\n</question>\n\n"
            "<excerpts>\n{excerpts}\n</excerpts>\n\n"
            "Answer in the required structure."
        ),
    },
    # v3: v2's shape caps plus two rules, each written against a failure the
    # evidence checks flagged on the v1 run of 2026-09-08 (eval/results/
    # tuning-v1.json). q01 claim K2 quoted four section headings joined with
    # ellipses, which no excerpt contains as one passage; q02 claims K6, K7
    # and K16 paraphrased their passages, so the percentages they state fell
    # outside the quote and their columns could not be read; q02's summary
    # restated $26,974 million as $26.97 billion, so the sentence could not
    # be linked to its own claim. Neither rule is about length.
    "v3": {
        "system": (
            "You are an analyst supporting a private equity deal team. Answer the question using "
            "only the excerpts. Cite excerpt ids. Return the required structure.\n\n" + SCHEMA_HINT +
            "\n\nKeep the answer short enough to read on one screen. Write at most four claims "
            "per company and at most twelve in total, choosing the ones that carry the figures and "
            "the disclosures the question asks about. Write at most one summary sentence per "
            "company plus one closing sentence. Give the table at most five rows. Say less rather "
            "than repeating a figure that is already in the table.\n\n"
            "Two rules about fidelity. A quote is one contiguous passage copied exactly from a "
            "single excerpt: no ellipsis, no paraphrase, no joining of separate lines or headings. "
            "If a fact needs two passages, write two claims. And when a summary sentence or a table "
            "cell restates a figure, write it exactly as its claim states it, with the same digits "
            "and the same units; never convert millions to billions or round a figure the claim "
            "gives in full.\n\n"
            "None of this loosens the rules above: a shorter answer still cites, still quotes, "
            "still names the period and the units, and still records what the excerpts do not "
            "support."
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


def render(question: str, context: Context, plan: Plan, version: str | None = None) -> tuple[str, str]:
    # Resolved here rather than as a default argument, which would bind the
    # value at import and ignore a version chosen for one run.
    version = version or PROMPT_VERSION
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
