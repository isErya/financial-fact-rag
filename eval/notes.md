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
did refusals fire with the right status, did the context stay inside its token budget. The same
rows are then run three ways over the full index, lexical only, dense only and the two fused, to
see what the retrieval mode is worth. The pinning ablation was run only on a small index and is
not quoted, for the reason given under Results.

Layer 2 (one model request per row): the evidence checks (quotes found in their cited excerpt,
figures present in their quote, table figures matched to the column the claim names, units
declared), tokens estimated against tokens billed, cost, latency, whether the structured
response parsed on the first try, and a prompt-leak check. Then a four-part rubric per row:
addresses the question (0-2), conclusions follow from the cited excerpts (0-2), complete against
the expected facts (0-2), comparability and gaps stated where they should be (0-2).

The answers are written by Claude and the rubric is scored by GPT. The grader is a different model
family from the generator, so nothing here grades its own work. The grader sees the question, the
answer with its quotes and excerpt ids, and an answer key naming the figures and the traps the row
exists to catch. It never sees which system produced the answer. The exact instructions and the
score anchors it was given are kept verbatim, so the grading can be repeated or disputed.

## What the numbers prove, and what they do not

The evidence checks establish presence and provenance: that a quoted passage exists in the
cited excerpt, that a figure sits inside that quote, and which column of which table it came
from. They do not establish that the answer is right, that the right row was chosen, or that
nothing important was left out. Only the graded rubric speaks to those, and it is one model's
reading of 24 rows against a fixed set of anchors.

## Results

### Layer 1, tuning set, lexical index

From `eval/results/retrieval.json`. Expected sections present 10 of 10, expected filings present
15 of 15, expected passage present 6 of 6, refusal rows with the expected status 2 of 2, rows with
the expected status overall 12 of 12, rows inside their token budget 10 of 10, mean context 11,733
tokens.

### What the retrieval mode is worth

Two measurements, and they disagreed.

The first ran the 12 tuning rows over the full 64,612-chunk index in three modes. Every hit-rate
measure above came out identical, and on the rank of the first correct passage dense retrieval
looked better (MRR 0.671 against 0.501 lexical) on ten gradeable rows, which is too few to act on.
That run is kept in `eval/results/retrieval_modes_full.json`.

The second was built to discriminate. Sixty rows were written from the filings, each carrying a
literal string copied out of the passage that answers it, so the set of correct passages is
small and a rank means something. Ten are paraphrases that share no vocabulary with their
answer. Two independent reviews against the filings dropped twelve rows: some at the lexical
ceiling where every retriever scored rank 1, some with labels the filing did not support. The 48
survivors are in `eval/results/retrieval_modes_bigset.json`, scored with a never-reached passage
counted as zero so that every candidate shares one denominator.

| Retriever | MRR, all 48 | Never found | Paraphrase, 10 rows | Literal, 32 rows |
|---|---|---|---|---|
| Lexical only | 0.321 | 7 | 0.019, 6 never found | 0.422 |
| BGE base, dense only | 0.439 | 0 | 0.700 | 0.381 |
| BGE base fused with lexical | 0.396 | 0 | 0.186 | 0.452 |
| Qwen3 0.6B, dense only | 0.655 | 0 | 0.538 | 0.642 |

Three things follow. Lexical search fails outright on six of ten paraphrase questions, which is
what a person types, so dense retrieval is the default. Fusion makes that case worse, because
reciprocal rank fusion averages in a lexical ranking that is close to random on those rows, so
the shipped mode is dense alone. And on literal questions lexical search remains competitive,
which is why the lexical index still ships behind `DENSE=0` for a machine without a GPU.

BGE base was chosen over Qwen3 on three grounds: it scores highest where the demo lives, the
paraphrase rows; it runs in the inference library already in the image, where Qwen3 would need
a second stack; and it costs 130 ms a query against 413 ms plus a nine-second model load.

What this does not establish: 48 rows separates lexical from dense, and separates the paraphrase
case clearly, but it does not reliably separate the four dense models from each other on the overall
figure. It also measures retrieval rank, not answer quality; the rubric below is
the only measure of that.

### Layer 2

Not yet run. One model request per row is needed for the evidence-check rates, tokens billed
against tokens estimated, cost and latency, the first-try parse rate, the held-out set's single
final run, and the rubric scores. Those land here and in `docs/PROMPT-LOG.md` once the prompt
version is frozen.

## Sample size

24 labelled questions plus the live stranger test. Enough to catch the failure classes named
above; nowhere near enough to quote a rate with error bars, and the notes say so.
