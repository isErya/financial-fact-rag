# Prompt iterations

Every row below is a run that actually happened against `eval/tuning.jsonl` (12 questions),
with the prompt version stored in `service/prompts.py` and the results file committed under
`eval/results/`. The metric columns come from `python -m eval.run --report <before> <after>`.
The held-out set (`eval/heldout.jsonl`) was run once, with the final version, and never used
to choose a change.

Columns: version | what changed | the failure that prompted it (row id, one line) | metric
before -> after | keep or revert.

| Version | What changed | Observed failure | Before -> after | Decision |
|---|---|---|---|---|
| v1 | Baseline: excerpts plus "answer the question and cite excerpt ids" | (baseline) | | keep as the reference point |

## Retrieval and chunking changes that moved a number

Logged here too, so the one file holds every decision that changed a result.

| Change | Observed failure | Before -> after | Decision |
|---|---|---|---|
