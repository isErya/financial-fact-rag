# Financial facts

**Ask one question about a portfolio company's SEC filings and get a brief where every figure
names its filing, its period, its units, and the row and column it was read from.**

The corpus is 246 10-K and 10-Q filings from 54 companies. I parse each filing into its items and
notes, chunk it so every table keeps its caption, its unit line and its column headers, and index
the chunks with BM25 (dense embeddings are an optional flag). A question is scoped in code before
anything is retrieved: which companies, which filings, which periods, which sections, with no
model call and the result shown on screen. Retrieval runs only inside that scope. One model
request then writes the answer into a fixed schema of claims, each with a quote and the excerpt
ids it came from. Code takes it from there and checks each quote against the excerpt it cites,
each figure against the located passage, each figure's column against the period the claim states,
and each unit against the table's own unit line. The rule the whole build follows: code decides
what the model reads, the model writes, code checks what it wrote against the source.

```
edgar_corpus.zip -> parse -> chunk (tables keep headers, units, columns) -> index (BM25, dense optional)
question -> scope (companies, filings, periods; no model) -> retrieve -> ONE model request -> parse -> evidence checks -> page :8804
```

## Deliverables

| # | Asked for | Where it is |
|---|---|---|
| 1 | README with setup and run instructions | this file |
| 2 | Indexing and retrieval code | `service/corpus.py`, `service/chunk.py`, `service/index.py`, `service/rules.py`, `service/retrieve.py` |
| 3 | A log of prompt iterations | `docs/PROMPT-LOG.md`, with every version that was run kept in `service/prompts.py` and its run under `eval/results/` |
| 4 | The final prompt template | `service/prompts.py` (`PROMPT_VERSION` names the one `ask.py` renders) |
| 5 | A front end | `service/web.py` and `service/static/` (the page at http://localhost:8804) |
| 6 | An example request ready to execute | `examples/request.json` and `examples/ask.sh`, the curl below, the four example questions on the page, and `python service/ask.py "<question>"` |
| 7 | Notes on how quality was evaluated | `eval/notes.md`, `eval/run.py`, `eval/tuning.jsonl`, `eval/heldout.jsonl` |

## Run it

Needs Docker with Compose v2 and about 4 GB of memory available to Docker. The filings ship with
the repo at `data/edgar_corpus.zip` (19.5 MB of public-domain EDGAR text), so there is no download
step. The commands are written for bash (Linux, macOS, or WSL on Windows).

**Building the index needs an NVIDIA GPU, and so do the tests.** Retrieval is dense, and the first
start embeds all 64,612 passages on the device: measured at 77 seconds on an RTX 5080, 109
seconds for the whole index including parsing. The indexer
is built on the GPU image and refuses to start a CPU build rather than silently taking the nine
hours that would cost; the web server stays on the CPU image, since it embeds one query per
question in about 130 ms and has to run on a laptop. The host needs the NVIDIA container toolkit
so Compose can hand the device through; nothing else is installed on the host.

Without a GPU, set `DENSE=0` in `.env`; the index then builds in about two minutes and retrieval
is lexical. Every other part of the system behaves the same way, and `/health` reports which
retrieval mode is actually being served, so a lexical index is never mistaken for a dense one.

```bash
git clone https://github.com/isErya/financial-sourcing-rag.git && cd financial-sourcing-rag
cp .env.example .env      # paste the key into ANTHROPIC_API_KEY, set LLM_MODEL_BACKEND=anthropic
docker compose up --build
```

Then open **http://localhost:8804**.

The key is optional. `LLM_MODEL_BACKEND` defaults to `fake`, so a clone with no `.env` at all comes
up and answers: the fake backend replays a stored response when the question matches one in
`eval/fixtures/`, and otherwise returns a stand-in answer that quotes the opening of the first
excerpt. The page says which of the two happened. Everything except the model's words is real on
that path, so the scope band, the retrieval, the evidence checks and the source cards all behave
as they do with a key.

A first start builds the image (`python:3.12-slim`, the service requirements, and the embedding
model baked in at build time so the containers never reach the network; the web image is 1.42 GB
on this machine, of which the model is 209 MB, and the indexer and test images are 5.2 GB because
they carry the CUDA 13 runtime and cuDNN as pip packages) and then runs the indexer once. The indexer's own phase times from the build in
this repo's `index/fingerprint.json`, built on an RTX 5080: parse 17.9 s, chunk 6.6 s, write 3.1 s,
BM25 4.3 s, dense 77.0 s, so under two minutes of indexing before the page can answer. Every later `up` compares the corpus
fingerprint with the one in the volume, prints `index up to date` and exits in a second.

The embedding model is BGE base (`BAAI/bge-base-en-v1.5`, 768 dimensions), chosen from five
candidates on a measurement described under Evaluation. Embedding one query at request time costs
about 130 ms on this CPU, and retrieval for a five-company question is under three seconds, so
the GPU requirement is a build-time cost and not an answer-time one. Two things the service does
about the build-then-serve split, because vectors built on one machine and queried on another can
silently disagree: it prepends the model's query instruction itself, since the library does not,
and at startup it re-embeds a few stored passages and refuses to serve dense retrieval if they do
not match their stored vectors. `/health` shows the result of that check.

### Check it worked

```bash
docker compose ps                     # web is up and healthy; indexer exits 0 when it is done
docker compose logs indexer           # "index up to date", or the phase line from a real build
curl -s localhost:8804/health | python3 -m json.tool
```

`/health` on the stack as it ships, with a key configured:

```json
{"chunks": 64612, "files": 246, "tickers": 54, "index_fingerprint": "2cf3859c...",
 "index_built_at": "2026-09-08T08:31:02", "index_error": null, "dense": true,
 "search_mode": "dense", "search_mode_effective": "dense", "search_mode_note": null,
 "embed_model": "BAAI/bge-base-en-v1.5", "encoder_error": null,
 "backend": "anthropic", "model": "claude-opus-5", "llm_ready": true,
 "api_key_env": "ANTHROPIC_API_KEY", "prices_checked": "2026-09-07"}
```

`llm_ready` says only that the configured backend has what it needs, which is a key for the API
backend and nothing for the fake. No route pings the provider. `examples/preflight.sh` runs the
same checks in order and ends with GO or NO-GO; it is the one place that does ask the provider
whether a key is live, through a models-list GET with no completion request.

### Look at the results

| | |
|---|---|
| http://localhost:8804 | the page: question, interpreted scope, answer, sources |
| `sh examples/ask.sh` | one full request from `examples/request.json` through `curl`, printed as JSON |
| `curl -s localhost:8804/ask -H 'content-type: application/json' -d '{"question": "What regulatory risks do the major pharmaceutical companies face?", "dry_run": true}'` | a dry run: the scope, the coverage block and the excerpts, with no model request |
| `docker compose exec web python ask.py "<question>"` | the same pipeline from the terminal, with `--dry-run` and `--json` |
| `curl -s localhost:8804/coverage` | every company, every filing it holds, and the fiscal label given to each |

The page paints in two passes. The first is a dry run that shows what will be read before anything
is sent: the companies with the text that matched them, the filings chosen with their period ends
and the reason each was picked, the favoured sections, and the token budget. The second pass makes
the model request and paints the answer, the claims with their evidence-check badges, and the
source cards with the quoted passage highlighted and the matched column underlined. Clicking a
figure scrolls to the card it came from.

Below the answer, the details section carries the instrumentation, open and in full: the per-chunk
token table against the budget, the quotas and pinned chunks, the sub-queries, the evidence-check
counts and every flag, the request and response tokens with the cost and the latency by stage, and
a "show the request" card holding the exact system and user text that was sent, the request id, and
the attempt and completion counters. Nothing on the page is hidden behind a control. The page has
two: the example questions, which fill the box, and the ask button.

Screenshots (taken against the running stack on the fake backend):

![The first screen: the four example questions, the question box, the ask button, and the index
line in the footer](docs/ui-first.png)

![The interpreted scope band for the three-company risk question, showing each company with the
alias that matched it and the filings that will be read](docs/ui.png)

![The bank disclosure brief, scoped to the two banks' Q3 2025 10-Qs with each bank's newest
10-K as an annual baseline](docs/ui-bank.png)

- `docs/ui-scope.png` is the answer band for the three-company risk question: the summary, its
  citation tags, and the start of the claims. The whole page runs past 30,000 pixels once every
  source card is on it, which is why it is not captured as one image.
- `docs/ui-mobile.png` is the same page at 390 px wide.

### Stop it

```bash
docker compose down        # keep the index volume; the next start is seconds
docker compose down -v     # also drop the index; the next start rebuilds it (about two minutes)
```

### If it does not come up

| Symptom | Cause and fix |
|---|---|
| `port is already allocated` | **8804** is the only port published to the host. Put `WEB_PORT=<free port>` in `.env` and `docker compose up -d` again. |
| 503 from `/ask` or the page, `index not built; run: docker compose up indexer` | The web process started without a built index in the volume. Run `docker compose up indexer`, then reload. |
| The page shows the `no api key` banner and answers as `fake` | No key reached the container. Set `LLM_MODEL_BACKEND=anthropic` and `ANTHROPIC_API_KEY` in `.env` next to the compose file, then `docker compose up -d --force-recreate web`. The scope, the excerpts and the source cards work without a key. |
| The indexer exits 137 | Docker killed it for memory. Raise Docker's memory limit to about 4 GB (Docker Desktop: Settings, Resources). |
| `model request failed (ModelUnavailable)` after a long wait | The client's timeout is 300 s and it makes one request with no retry, so a timeout is reported instead of being paid for twice. Ask again. |
| The scope band says `needs_company` | The question named no company the registry knows. Name one, or open `/coverage` to see the 54 that are held. |
| `docker compose` not found | Compose v2 is required. `docker-compose` (v1, hyphenated) will not read this file. |

## Tests

```bash
docker compose run --rm tests      # same base image as the web service, so only Docker is needed
```

The suite runs against the real corpus zip rather than a fixture of it. It covers the parser (all 246
filings parse, section boundaries land outside the table of contents, fiscal labels for the
calendars that trip people up, running headers removed while table header rows survive), the
chunker (a statement's rows keep their labels, a long table splits with its caption and headers
repeated on every piece, columns and units parsed), the scope rules (alias tiers and their traps,
group phrases, period windows, the annual-baseline rule, the three refusal statuses), retrieval
and assembly, the evidence checks (each attack that an adversarial review landed is its own named
regression test), one request per answer, the web routes on the fake backend, and the copy rules in
`tests/test_copy.py`.

A full run collects 150 tests. On the GPU host it takes about a minute; the run behind this
paragraph was 146 passed, 3 skipped and 1 failed on an RTX 5080 in 59 seconds, and the failure was
a test asserting a stricter rank than the shipped design relies on, corrected in the same commit.
Parsing all 246 filings and building a small index is most of the remaining time.

The dense tests build one small dense index per run, the 692 passages of a single JPMorgan 10-Q,
and share it. On a laptop CPU that build is the slowest part of the suite by a wide margin. Two
tests skip by design rather than failing: they need the full-corpus index rather than the small
one the suite builds for itself. Each prints the reason it did not run, so a skip is never
mistaken for a pass. A further conditional skip lives in the copy test, described next.

`tests/test_copy.py` reads its confidential-term list from the path in `COPY_DENYLIST_FILE`. That
list lives outside the repo, so a public clone leaves the variable unset, and the check prints why
it did not run instead of failing. The other three checks (no smart punctuation, ASCII only, and no
wording that claims an answer is established) always run.

## Evaluation

The method, the two question sets and their schema are in `eval/notes.md`. Layer 1 scores retrieval
alone with no model request. Layer 2 makes one request per row, scores the evidence checks and the
cost, and is then scored against a four-part rubric. The answers are written by Claude and the
rubric is scored by GPT, so the grader is a different model family from the generator and
nothing grades its own work.

Layer 1 on the 12-row tuning set, from `eval/results/retrieval.json` (the shipped lexical index):

| Measure | Result |
|---|---|
| Expected sections present in the assembled context | 10 of 10 |
| Expected filings present | 15 of 15 |
| Expected passage present | 6 of 6 |
| Refusal rows with the expected status | 2 of 2 |
| Rows with the expected status overall | 12 of 12 |
| Rows inside their token budget | 10 of 10 |
| Mean context size | 11,733 tokens |

### Does dense retrieval earn its place

Yes, and the first measurement said no. The 12-row tuning set scored every retrieval mode
identically on the hit-rate measures above, and it is small enough that this read as a tie. So a
48-row set was written specifically to separate them: every row carries a literal string copied
from a filing, which keeps the set of correct passages small enough that rank means something,
and ten of the rows are questions phrased in a reader's own words that share no vocabulary with
the passage that answers them. Two independent reviews against the filings dropped 12 of the 60
rows authored. The scoring is in `eval/results/retrieval_modes_bigset.json`, over the full
64,612-chunk index, with a passage the retriever never reached counted as zero so that every
candidate is scored over the same denominator.

| Retriever | MRR, all 48 | Never found the passage | MRR, paraphrase rows |
|---|---|---|---|
| Lexical only | 0.321 | 7 of 48 | 0.019, missing 6 of 10 |
| BGE base, dense only | 0.439 | 0 | 0.700 |
| BGE base fused with lexical | 0.396 | 0 | 0.186 |

Lexical search fails outright on six of the ten paraphrase questions, which is the kind of question
a person types. Fusing the two rankings makes that case worse, not better, because reciprocal rank
fusion averages in a lexical ordering that is close to random there. On questions that share
vocabulary with the filing, lexical search is competitive (0.422 against 0.381 dense). So the
shipped configuration is dense retrieval alone, with the lexical index kept for `DENSE=0`.

Five embedding models were scored the same way. The largest, Qwen3 at 1024 dimensions, ranked
best overall (0.655) but is not supported by the inference library in the image and costs 413 ms
per query; BGE base scored highest on the paraphrase rows, which is the case the demo turns on,
and runs in the image as shipped. Vector files run 99 MB to 265 MB across the five and are built
by the indexer rather than committed.

### Layer 2: one request per row, three prompt versions

Run on 2026-09-08 on the shipped backend, Opus 5, over the 12 tuning rows; ten make a request
and two are refused before any request. Results in `eval/results/tuning-v1.json`, `tuning-v2.json`
and `tuning-v3.json`; the changes and the reasons are in `docs/PROMPT-LOG.md`.

| Version | Quotes located | Figures in quote | Columns matched | Units matched | Flags | Mean output tokens | Cost for the set |
|---|---|---|---|---|---|---|---|
| v1, baseline | 125 of 129 | 105 of 111 | 37 of 47 | 40 of 40 | 56 | 3,933 | $1.85 |
| v2, shape caps | 61 of 67 | 62 of 71 | 23 of 36 | 21 of 21 | 34 | 2,902 | $1.61 |
| v3, caps plus fidelity rules | 63 of 63 | 64 of 76 | 24 of 36 | 25 of 25 | 32 | 2,352 | $1.47 |

Every reply parsed on the first try in every version. v3 is the shipped default: it is the only
version in which every quote is located, it invented nothing where v2 invented one figure, and
it is a third shorter than the baseline. Figures-in-quote is the one number it gave back, and
the prompt log says exactly why.

The rubric was applied blind: all 30 answers rendered with the version label replaced by a
letter, shuffled per row, and scored before the key was opened. Out of 80, v1 scored 75, v2
scored 73, v3 scored 78. Every version addressed every question, was complete against the key,
and stated gaps and comparability; the whole difference was whether conclusions rested on the
cited excerpts. Scores and key: `eval/results/rubric-blind-2026-09-08.json`. The held-out set
was not run.

What these numbers do not prove is in `eval/notes.md` and bears repeating here. The layer-2 counts
establish presence and provenance. Whether an answer is right is the rubric's business, the rubric
is one model's reading, and 24 labelled questions is enough to catch the failure classes I named
and nowhere near enough to quote a rate with error bars.

