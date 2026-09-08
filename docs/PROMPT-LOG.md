# Prompt iterations

Every row below is a run that actually happened against `eval/tuning.jsonl` (12 questions, 10 of
which make a model request; two are refusals answered before any request), on the shipped API
backend with Opus 5, on 2026-09-08, with the prompt version stored in `service/prompts.py` and
the results file committed under `eval/results/`. The metric columns come from
`python -m eval.run --report <before> <after>`. The held-out set was not run; the quality gate for
choosing between versions was a blind reading of every answer, described at the end.

Columns: version | what changed | the failure that prompted it (row id, one line) | metric
before -> after | decision.

| Version | What changed | Observed failure | Before -> after | Decision |
|---|---|---|---|---|
| v1 | Baseline: the schema hint, "answer from the excerpts and cite excerpt ids", no shape rules | (baseline) | quotes located 125/129 (97%), figures in quote 105/111 (95%), columns matched 37/47, units 40/40, 56 flags, mean 3,933 output tokens, $1.85 for the set | reference point; kept in PROMPTS |
| v2 | Shape caps: at most four claims per company and twelve in total, one summary sentence per company plus one closing, table at most five rows | q01: 23 claims, six summary sentences and 7,207 output tokens for the three-company comparison, 95 s on the shipped backend; unreadable on one screen | quotes located 97% -> 91%, figures in quote 95% -> 87%, flags 56 -> 34, mean output tokens 3,933 -> 2,902, cost $1.85 -> $1.61 | revert as default: shorter, but q05 stitched two table rows into one quote with an ellipsis (two quotes not located) and stated a $10.8 billion Mac decline the excerpts do not print; the checks caught both |
| v3 | v2's caps plus two fidelity rules: a quote is one contiguous passage copied exactly from a single excerpt, no ellipsis, no paraphrase, no joining of lines; and a summary or table cell restates a figure exactly as its claim states it, same digits, same units | q01 K2 (v1): a quote built from four section headings joined with ellipses, which no excerpt contains; q02 K6, K7, K16 (v1): paraphrased quotes, so the percentages fell outside the quote and their columns could not be read; q02 summary (v1): $26,974 million restated as $26.97 billion, so the sentence could not be linked to its claim | from v2: quotes located 91% -> 100% (63/63), figures in quote 87% -> 84%, columns matched 64% -> 67%, units 100%, flags 34 -> 32, mean output tokens 2,902 -> 2,352, cost $1.61 -> $1.47 | keep; the shipped default |

## What v3 gave back, and why it is still the default

Figures-in-quote fell from 95 percent under v1 to 84 percent under v3. The cause is visible on
q05: with quotes held to one contiguous row, a claim that lists five product categories quotes
only the first row, and the resolver flags the other four figures as sitting in the cited excerpt
outside the quote. They are the right figures from the right table; the checker cannot read their
column through that quote. Nothing was invented. The rule that would close it, one claim per
table row, is the next iteration and was not run.

v1 remains the most complete version. On the three-company comparison it carries Tesla's
brand-and-protest risk and the CEO award risk, four not-comparable notes and a seven-row table;
on the bank brief it adds JPMorgan's Advanced-approach ratio and a note that the two annual
baselines are different fiscal years. It is also twice the length and, on the panel's own first
question, 95 seconds against v3's 59.

## The blind reading

Every answered row of every version, 30 answers, was rendered with the version label replaced by
a letter, shuffled per row, and scored on the four rubric criteria in `eval/notes.md` (addresses
the question, conclusions follow from the cited excerpts, complete against the expected facts,
comparability and gaps stated where they should be; 0 to 2 each) before the key was opened.
Two rows, q04 and q05, had been read unblinded earlier while diagnosing flags and are scored on
content with that noted.

| Version | Addresses | Follows | Complete | Gaps | Total of 80 |
|---|---|---|---|---|---|
| v1 | 20 | 15 | 20 | 20 | 75 |
| v2 | 20 | 13 | 20 | 20 | 73 |
| v3 | 20 | 18 | 20 | 20 | 78 |

Every version addressed every question, was complete against the key, and stated gaps and
comparability. The whole separation is the second criterion. v3 lost a point on two rows, v1 on
five, v2 on seven including the fabricated figure. v3 never scored below v1 on any row. The
scores and the key are in `eval/results/rubric-blind-2026-09-08.json`.

The same error appears in all three versions on q02 and is not a prompt problem: a quarterly
claim carries figures that sit in a nine-month column, or the reverse. The column check catches it
every time. That is what the check is for.

## Retrieval and chunking changes that moved a number

Logged here too, so the one file holds every decision that changed a result.

| Change | Observed failure | Before -> after | Decision |
|---|---|---|---|
| Dense retrieval on BGE base as the default, no fusion | On a 48-row set built to discriminate, lexical search never reached the correct passage in 7 of 48 and failed 6 of 10 paraphrase questions outright | lexical MRR 0.321 -> BGE dense 0.439 over 48 rows; paraphrase 0.019 -> 0.700; fused would have been 0.396 and 0.186 | keep; `eval/results/retrieval_modes_bigset.json` |
| Row-label seat | On the full index the BAC Q3 2025 net interest income table ranked 43rd of 60 under lexical search while passing on a 12-file fixture | bank brief context missing BAC's NII -> present | keep; the seat pins the row whatever its rank |
