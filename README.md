# Filing desk

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
| 6 | An example request ready to execute | `examples/request.json` and `examples/ask.sh`, the curl below, the four preset chips on the page, and `python service/ask.py "<question>"` |
| 7 | Notes on how quality was evaluated | `eval/notes.md`, `eval/run.py`, `eval/tuning.jsonl`, `eval/heldout.jsonl` |
| 8 | Assumptions and decisions | the "Assumptions" and "Tradeoffs" sections below |
| 9 | Value creation and future state | the two short sections below, `docs/VALUE.md`, `docs/FUTURE-STATE.md` |
| 10 | One LLM API request per answer | `service/ask.py` and `service/llm_client.py` (one call site, `max_retries=0`, attempt and completion counters), asserted in `tests/test_ask.py`; see "How the one-request rule is kept" |

## Run it

Needs Docker with Compose v2 and about 4 GB of memory available to Docker. The filings ship with
the repo at `data/edgar_corpus.zip` (19.5 MB of public-domain EDGAR text), so there is no download
step. The commands are written for bash (Linux, macOS, or WSL on Windows).

```bash
git clone <this-repo> && cd eliza-fde-exercise
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
model baked in at build time so the containers never reach the network; the built image is 932 MB
on this machine) and then runs the indexer once. The indexer's own phase times from the build in
this repo's `index/fingerprint.json`: parse 57.4 s, chunk 24.3 s, write 12.0 s, BM25 17.4 s, so
about two minutes of indexing before the page can answer. Every later `up` compares the corpus
fingerprint with the one in the volume, prints `index up to date` and exits in a second.

`DENSE=1` adds embeddings for every chunk at build time. Measured here on the 12-filing tuning
index: 5,752 chunks took 326.7 s of embedding (`index-tuning/fingerprint.json`), so the full
64,612-chunk corpus is tens of minutes on this CPU, and the shipped default index is lexical.

### Check it worked

```bash
docker compose ps                     # web is up and healthy; indexer exits 0 when it is done
docker compose logs indexer           # "index up to date", or the phase line from a real build
curl -s localhost:8804/health | python3 -m json.tool
```

`/health` on the stack as it ships:

```json
{"chunks": 64612, "files": 246, "tickers": 54, "index_fingerprint": "31087bf0...", "dense": false,
 "backend": "fake", "model": "claude-opus-5", "llm_ready": true, "index_error": null}
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

The `details` button at the end of the answer opens the instrumentation: the per-chunk token table
against the budget, the quotas and pinned chunks, the sub-queries, the evidence-check counts and
every flag, the request and response tokens with the cost and the latency by stage, and a "show the
request" card holding the exact system and user text that was sent, the request id, and the
attempt and completion counters.

Screenshots (taken against the running stack on the fake backend):

![The first screen: the question box, the four preset chips, and the index line in the
footer](docs/ui-first.png)

![The interpreted scope band for the three-company risk question, showing each company with the
alias that matched it and the filings that will be read](docs/ui.png)

![The bank disclosure brief preset, scoped to the two banks' Q3 2025 10-Qs with each bank's newest
10-K as an annual baseline](docs/ui-bank.png)

- `docs/ui-scope.png` is the whole page for the first preset, top to bottom: scope, answer, claims
  with their badges, and every source card the model was given.
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

A full run is 144 tests and takes minutes rather than seconds, because parsing all 246 filings and
building a small index is most of the work. The run behind this paragraph took 8 minutes 49 seconds
on this laptop with other jobs competing for the cores; expect roughly five on a quiet machine.

`tests/test_copy.py` reads its confidential-term list from the path in `COPY_DENYLIST_FILE`. That
list lives outside the repo, so a public clone leaves the variable unset, and the check prints why
it did not run instead of failing. The other three checks (no smart punctuation, ASCII only, and no
wording that claims an answer is established) always run.

## How the one-request rule is kept

There is one call site: `ask.py` calls `model.generate` once, and every other step around it is
plain Python. The API client is built with `max_retries=0` and a 300 second timeout, so the SDK
never quietly makes a second request, and a reply that does not parse is reported with its raw text
rather than repaired by another call. A guard object is created per ask and handed to the backend;
a second `generate` inside one ask raises `RequestBudgetExceeded`. Every payload carries
`llm_attempts` and `llm_completed` (0 and 0 on a refusal that never reaches the model, 1 and 1 on
an answer) along with the provider's request id, the page prints them under `details`, and
`tests/test_ask.py` asserts all three cases.

## What the four hours bought, and what came after

The four-hour version: the parser that keeps tables, dates and units; the index; the deterministic
scope over companies, filings and periods; one API request; the quote-in-source checks; the page
with clickable evidence; and a first evaluation on the tuning set.

What came after, each traceable to a failure I watched happen and logged in `docs/PROMPT-LOG.md`:
the column and units checks (after a provenance question quoted the prior-year column), the
annual-baseline retrieval rule (after a risk question on a 10-Q returned a 75-character pointer to
the 10-K), the bank disclosure brief preset, the rewrite of the evidence checks onto a contract
where an unchecked item is never counted as a pass, the held-out evaluation, and the polish.

I did not run a stopwatch and I am not going to pretend the split is exact. The honest version is
that the first sitting produced the pipeline end to end and everything since has been iteration on
what the evaluation showed.

## Assumptions

Each of these is a decision I made about this corpus. Every one of them is a place where a
different corpus would need different code.

- **The corpus is a fixed snapshot.** 246 filings from 54 companies, filed 2022 through 2026, with
  one exception: GE's only filing is a 2015 10-K for GE Capital (period end 2014-12-31). GE is
  excluded from group expansions and marked stale whenever it is named, so a question about "the
  industrials" never quietly answers from a 2015 filing.
- **A 10-Q's risk section is often a pointer, so an annual baseline is always in scope.** The parse
  report records 65 pointer stubs among the 157 10-Qs, and 24 10-Qs (JNJ 12, XOM 12) that carry no
  Part II Item 1A at all. For a section-led question the newest 10-K section is retrieved beside
  the newest 10-Q, and a stub pins its 10-K section into the same bucket. Without that rule, "what
  are Bank of America's main risk factors" answers from 75 characters.
- **Fiscal labels are computed from the period end and each company's fiscal year end.** The
  quarter tag in the corpus file names is a calendar quarter and is never read for anything. The
  formula handles the calendars that trip people up: NVIDIA's quarter ended 2025-10-26, filed
  2025-11-19, is labelled FY2026 Q3. Where a company names its own year differently from the
  formula, the difference is a hand-set offset in `service/companies.yaml`: Home Depot and Target
  call the year ended in early 2025 their fiscal 2024 and carry an offset of -1, while Walmart
  calls the same stretch fiscal 2025 and carries none. Every filing is printed with its period end
  beside its label, so a reader always sees which months a label covers.
- **No amendments, no duplicate periods.** This corpus holds no 10-K/A or 10-Q/A and no two filings
  of the same form and period for one company. The rule a real ingest needs (an amendment
  supersedes the original, and the newest filing of a period wins) is described here and is not
  built.
- **Only the primary document is indexed.** Every file ends at the signatures block, so there are
  no exhibits, no certifications and no subsidiary lists in the corpus, and none of that is
  indexed.
- **The inline-XBRL preamble holds no values.** Each file starts with tag names and context ids
  before the cover page, and the parser drops all of it at the cover marker. `index.py report`
  checks that no comma-formatted number in a preamble is missing from the body it precedes; the
  parse report records 0 files failing that check.
- **Figures are quoted from the filing and never computed.** The system does no arithmetic. There
  are no growth rates, no sums and no derived ratios unless the filing itself prints them, because
  a computed number has no row and no column to point at.
- **The model reads only what code chose.** Retrieval is restricted to the filings the scope
  selected, the context is capped by a token budget (20,000 tokens for one to three companies,
  30,000 for four to six, at most six companies per question), and what was dropped is reported
  rather than hidden.
- **The evidence checks establish presence and provenance. They do not establish that an answer is
  right.** A check can say that a quoted passage appears in the excerpt it cites, that a figure sits
  in that passage, which column of which table it came from, and whether the units agree. None of
  that speaks to whether the right row was chosen, whether the answer addresses the question, or
  whether something important was left out. Only a graded reading by a person speaks to those, and
  that grading is reported separately, in `eval/notes.md`. This distinction is the point of the
  build, so the page and these documents never blur it: every badge is named for what it
  establishes, and an item the checks could not run on is shown as unchecked and never counted as
  a pass.

## Evaluation

The method, the two question sets and their schema are in `eval/notes.md`. Layer 1 scores retrieval
alone with no model request. Layer 2 makes one request per row, scores the evidence checks and the
cost, and is then graded by hand against a four-part rubric.

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

The retrieval ablation (`eval/results/retrieval_ablation.json`, run over a 12-filing index so a
dense build was affordable) scores lexical only, dense only, hybrid, and hybrid without pinning
identically on every one of those hit-rate measures. They differ only in context size (mean 10,870
lexical, 9,514 dense, 10,107 hybrid). On this question set, the deterministic scope is doing the
work that retrieval mode is usually credited with, which is why the shipped default is the lexical
index that builds in about two minutes rather than the dense one that takes tens of them.

> **PLACEHOLDER, fills from `eval/results/`:** the layer-2 table per prompt version (evidence-check
> rates, tokens billed against tokens estimated, cost, latency, first-try parse rate), the
> prompt-iteration summary generated from `docs/PROMPT-LOG.md`, the held-out table from the single
> final run, and the rubric averages for both sets. These land once the prompt iterations are
> frozen and the held-out set has been run once.

What these numbers do not prove is in `eval/notes.md` and bears repeating here. The layer-2 counts
establish presence and provenance. Whether an answer is right is the rubric's business, the rubric
is one person's reading, and 24 labelled questions is enough to catch the failure classes I named
and nowhere near enough to quote a rate with error bars.

## What this is worth to a deal team

The full version, including how I would measure it in a pilot, is in
[docs/VALUE.md](docs/VALUE.md).

Three people at a mid-market PE firm read these filings by hand today: the associate screening
comparables before a memo, the portfolio operations lead re-reading each quarter for the companies
the fund owns, and the credit or CFO office asking how safe the operating banks are. All three
need the same output, which is a short brief where every figure carries its period, its units, and
a source they can open. What changes here is that the reviewer's job moves from finding the numbers
to checking them, and the checking is faster because the source opens beside the claim with its row
label and column header visible.

I have built the checking half of this before: my day job is preparing 10-K and 10-Q material for
bank examinations at a federal financial regulator, where a figure without its period, its units
and its source is not usable. That is the reason the scope band, the column provenance and the unit
line exist here at all.

The numbers that matter are the client's, so `docs/VALUE.md` states them as a measurement plan
(time to an accepted brief before and during a pilot, questions per associate per week, the share
of figures a reviewer changed after checking the source, and cost per question from the usage
tile) rather than as a claim I can make from my own laptop.

## Where this goes next

The full list is in [docs/FUTURE-STATE.md](docs/FUTURE-STATE.md). The first four, in the order I
would build them: nightly ingestion from EDGAR for the portfolio and the watchlist; reported
figures taken from XBRL company facts with their periods and units attached, so the checks compare
text to structured data; a new-filing brief that re-runs last quarter's question against the
filing that just landed; and the same pipeline over deal-room documents inside the client's own
storage. Evaluation moves into the delivery pipeline at the same time, so a prompt or retrieval
change ships only when the evidence-check rates and the graded rubric hold. None of it changes the
rule the demo runs on.

## Tradeoffs

> **PLACEHOLDER.** I write this section by hand from my own build notes before this repo goes out.
> It covers the choices I made and what each one cost: BM25 as the shipped default with dense
> behind a flag, deterministic scope instead of a routing model, one request instead of a
> retrieve-check-retry loop, the schema the model answers into, and the checks I deliberately do
> not run.

## Surprises, and where this breaks in production

> **PLACEHOLDER.** I write this section by hand from my own build notes before this repo goes out.
> It covers what the corpus did that I did not expect, and the places this design would fail on a
> real filing feed.
