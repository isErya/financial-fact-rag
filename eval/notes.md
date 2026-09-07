# How quality was evaluated

## What is measured

Two question sets share one schema (`id`, `question`, expected tickers, sections, files,
substring, period end, duration, abstain status, not-comparable flag, notes):

- `eval/tuning.jsonl`, 12 rows: the three questions from the brief, the bank brief, three
  provenance traps (a prior-year column, a three-month versus nine-month column, share counts
  in thousands), an annual-baseline case, a footnote case, two refusals, one alias trap. Used
  for every prompt iteration.
- `eval/heldout.jsonl`, 12 rows: paraphrases, unexpected company pairs, a mixed fiscal
  calendar, a partly covered window, a two-part question, an injection, a bare ticker, a stale
  filer. Run once with the final prompt; never read while tuning.

Layer 1 (retrieval only, no model request, seconds): were the expected companies resolved, were
the expected sections and filings in the assembled context, was the expected passage present,
did refusals fire with the right status, did the context stay inside its token budget. An
ablation runs the same rows with lexical only, dense only, hybrid, and hybrid without pinning.

Layer 2 (one model request per row): the evidence checks (quotes found in their cited excerpt,
figures present in their quote, table figures matched to the column the claim names, units
declared), tokens estimated against tokens billed, cost, latency, whether the structured
response parsed on the first try, and a prompt-leak check. Then a rubric I grade by hand per
row: addresses the question (0-2), conclusions follow from the cited excerpts (0-2), complete
against the expected facts (0-2), comparability and gaps stated where they should be (0-2).

## What the numbers prove, and what they do not

The evidence checks establish presence and provenance: that a quoted passage exists in the
cited excerpt, that a figure sits inside that quote, and which column of which table it came
from. They do not establish that the answer is right, that the right row was chosen, or that
nothing important was left out. Only the graded rubric speaks to those, and it is one person's
reading of 24 rows.

## Results

<filled in from eval/results/*.json during milestones 6 and 8: layer-1 table, ablation table,
layer-2 table per prompt version, held-out table, rubric scores, the stranger test log>

## Sample size

24 labelled questions plus the live stranger test. Enough to catch the failure classes named
above; nowhere near enough to quote a rate with error bars, and the notes say so.
