"""Serve layer: the page, the health and coverage routes, and /ask.

Read-only over the index. The indexer container writes the index once;
this process loads it at startup and never writes anything.

    GET /            the page (static/index.html)
    GET /static/...  the page's script and styles
    GET /health      index counts and fingerprint, the backend and model,
                     and llm_ready: whether the configured backend has
                     what it needs (the key variable for the API backend;
                     always true for the fake). No provider ping.
    GET /coverage    every company in the registry with its filings, the
                     fiscal label the index gave each, and a stale note for
                     a filer whose newest filing is years behind the rest.
    POST /ask        {"question": str, "dry_run": bool} -> the ask() payload.
                     A dry run stops after retrieval and returns the scope,
                     the coverage block and the excerpts; the full run adds
                     the answer and the evidence checks.

The model request runs through run_in_threadpool so the event loop keeps
answering /health while a request is in flight (one request can take a
minute or more at the output cap). Model strings are never rendered here:
every route returns JSON and the page inserts text with textContent.

A missing index is reported as 503 with the command that builds it, which
is the state a fresh `docker compose up web` without the indexer lands in.
"""

import contextlib
import dataclasses
import json
import os
import zipfile

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import ask
import config
import corpus
import index as index_module
import rules

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
INDEX_MISSING = "index not built; run: docker compose up indexer"
# The key variable name lives in config; the fallback keeps this module
# importable against an older config.py.
API_KEY_ENV = getattr(config, "LLM_API_KEY_ENV", "ANTHROPIC_API_KEY")
# Bytes of each zip member read for its header block. The header is a few
# hundred bytes; the body starts after a rule of "=" characters.
HEADER_BYTES = 4096
# HTTP status per model failure class, by class name so this module does
# not import the model client. The classes are documented in llm_client.
MODEL_FAILURE_STATUS = {
    "ModelUnavailable": 503,
    "ModelProtocolError": 502,
    "ModelConfigError": 500,
    "RequestBudgetExceeded": 500,
}


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    dry_run: bool = False


# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------


def filing_urls(zip_path: str) -> dict[str, str]:
    """EDGAR URL per corpus member, read from the header block only. The
    header is the first few hundred bytes of each file, so this is a
    fraction of a second for the whole zip and needs no parse."""
    urls: dict[str, str] = {}
    if not os.path.exists(zip_path):
        return urls
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".txt"):
                continue
            with zf.open(name) as fh:
                head = fh.read(HEADER_BYTES).decode("utf-8", "replace")
            urls[name] = corpus.parse_header(head).get("URL", "")
    return urls


def load_fingerprint(index_dir: str) -> dict:
    path = os.path.join(index_dir, "fingerprint.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def load_fixture_dates(fixtures_dir: str) -> dict[str, str]:
    """Saved date per fixture question, keyed by the question with its
    whitespace collapsed and casefolded, so a replayed answer can say
    which day its response was stored."""
    dates: dict[str, str] = {}
    if not os.path.isdir(fixtures_dir):
        return dates
    for name in sorted(os.listdir(fixtures_dir)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(fixtures_dir, name)) as fh:
                record = json.load(fh)
        except (OSError, ValueError):
            continue
        question = record.get("question")
        if question:
            dates[question_key(question)] = record.get("saved") or ""
    return dates


def question_key(question: str) -> str:
    return " ".join(question.split()).casefold()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Index and registry load once per process. A missing index is kept as
    the error string the routes report, so the process still serves the
    page and /health while the indexer runs."""
    state = app.state
    state.index = None
    state.index_error = None
    state.fingerprint = {}
    try:
        state.index = index_module.load(config.INDEX_DIR)
        state.fingerprint = load_fingerprint(config.INDEX_DIR)
    except FileNotFoundError:
        state.index_error = INDEX_MISSING
    state.companies = index_module.load_registry(config.COMPANIES_FILE)
    state.urls = filing_urls(config.CORPUS_ZIP)
    state.fixture_dates = load_fixture_dates(config.FIXTURES_DIR)
    # The query encoder takes about 19 s to load and is otherwise loaded
    # lazily on the first dense query, which would put that on the clock of
    # the first question someone asks. Loading it here also runs the check
    # that it matches the stored vectors, so a mismatched index fails at
    # startup rather than on a question. A failure is recorded and the
    # process still serves the page, /health and the lexical path.
    state.encoder_error = None
    if state.index is not None and getattr(state.index, "dense", None) is not None:
        try:
            state.index.dense_scores("warm up the query encoder")
        except Exception as exc:  # noqa: BLE001 - reported, never raised at startup
            state.encoder_error = str(exc)
    yield


app = FastAPI(title="financial facts", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def llm_ready() -> bool:
    """Whether the configured backend can take a request: the fake needs
    nothing, the API backend needs its key variable set."""
    if config.LLM_MODEL_BACKEND == "fake":
        return True
    if config.LLM_MODEL_BACKEND == "anthropic":
        return bool(os.environ.get(API_KEY_ENV))
    return False


def loaded_index(request: Request):
    loaded = request.app.state.index
    if loaded is None:
        raise HTTPException(status_code=503, detail=request.app.state.index_error or INDEX_MISSING)
    return loaded


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def page() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health(request: Request) -> dict:
    loaded = request.app.state.index
    record = request.app.state.fingerprint
    return {
        "chunks": len(loaded.chunks) if loaded else 0,
        "files": len(loaded.files) if loaded else 0,
        "tickers": len(loaded.ticker_ranges) if loaded else 0,
        "index_fingerprint": record.get("sha256"),
        "index_built_at": record.get("built_at"),
        "index_error": request.app.state.index_error,
        "dense": bool(loaded is not None and loaded.dense is not None),
        "search_mode": config.SEARCH_MODE,
        "search_mode_effective": loaded.effective_mode()[0] if loaded else None,
        # None unless the index cannot serve the configured mode.
        "search_mode_note": loaded.effective_mode()[1] if loaded else None,
        "embed_model": config.EMBED_MODEL,
        # None when the encoder loaded and matched the stored vectors.
        "encoder_error": getattr(request.app.state, "encoder_error", None),
        "backend": config.LLM_MODEL_BACKEND,
        "model": config.LLM_MODEL,
        "llm_ready": llm_ready(),
        "api_key_env": API_KEY_ENV,
        "prices_checked": config.PRICES.get(config.LLM_MODEL, {}).get("checked"),
    }


@app.get("/coverage")
def coverage(request: Request) -> dict:
    """Per company: name, ticker, year end, filings with the fiscal label
    the index assigned, and a stale note where the rules would add one."""
    loaded = loaded_index(request)
    registry = request.app.state.companies
    companies = registry.get("companies", {})
    labels = {r["file"]: r["fiscal_label"] for r in loaded.files}
    stale = rules.stale_tickers(companies)
    out = []
    for ticker in sorted(companies):
        entry = companies[ticker]
        filings = [{
            "file": f["file"], "form": f["form"], "fiscal_label": labels.get(f["file"]),
            "period_end": f["period_end"], "filing_date": f["filing_date"],
        } for f in sorted(entry.get("filings", []), key=lambda f: f["period_end"])]
        note = None
        if ticker in stale and filings:
            newest = filings[-1]
            note = "newest filing is the %s %s, filed %s; treat it as stale" % (
                newest["fiscal_label"] or newest["period_end"], newest["form"], newest["filing_date"])
        out.append({"ticker": ticker, "name": entry.get("name"), "fye_month": entry.get("fye_month"),
                    "sector": entry.get("sector") or None, "filings": filings, "stale": note})
    return {"companies": out, "count": len(out), "files": sum(len(c["filings"]) for c in out)}


@app.post("/ask")
async def ask_route(body: AskRequest, request: Request) -> Response:
    """The ask() payload as JSON. The dry run is prepare() and the scope
    payload; the full run is ask.ask(), which makes the one model request.
    Both run off the event loop."""
    loaded = loaded_index(request)
    state = request.app.state
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question is empty")
    if body.dry_run:
        payload = await run_in_threadpool(dry_run_payload, question, loaded, state.companies)
    else:
        try:
            payload = await run_in_threadpool(ask.ask, question, loaded, state.companies, dry_run=False)
        except NotImplementedError:
            raise HTTPException(status_code=501, detail="answer step not wired yet")
        except Exception as exc:
            status = MODEL_FAILURE_STATUS.get(type(exc).__name__)
            if status is None:
                raise
            raise HTTPException(status_code=status, detail="model request failed (%s): %s" % (
                type(exc).__name__, exc))
    decorate(payload, question, loaded, state)
    return Response(content=ask.to_json(payload), media_type="application/json")


def dry_run_payload(question: str, loaded, companies: dict) -> dict:
    prepared = ask.prepare(question, loaded, companies)
    payload = ask.scope_payload(prepared)
    payload["timing_ms"] = {"plan_and_retrieve": None}
    return payload


def decorate(payload: dict, question: str, loaded, state) -> None:
    """What the page needs beyond the pipeline's own payload: the excerpt
    text, units, columns and EDGAR URL per context row (the rows carry
    only the header), the retry setting the request was made with, and
    the saved date of a replayed fixture."""
    payload["max_retries"] = config.LLM_MAX_RETRIES
    payload["dry_run"] = "answer" not in payload
    payload["sources"] = [source_of(row, loaded, state.urls) for row in payload.get("context", [])]
    if payload.get("replayed"):
        payload["replayed_date"] = state.fixture_dates.get(question_key(question)) or None


def source_of(row: dict, loaded, urls: dict[str, str]) -> dict:
    chunk = loaded.by_id[row["chunk_id"]]
    return {
        "cid": row["cid"], "chunk_id": chunk.chunk_id, "file": chunk.file, "ticker": chunk.ticker,
        "company": chunk.company, "form": chunk.form, "fiscal_label": chunk.fiscal_label,
        "period_end": chunk.period_end, "filing_date": chunk.filing_date, "part": chunk.part,
        "item": chunk.item, "item_title": chunk.item_title, "note_title": chunk.note_title,
        "kind": chunk.kind, "units": chunk.units, "units_source": chunk.units_source,
        "columns": [dataclasses.asdict(c) for c in chunk.columns], "column_source": chunk.column_source,
        "header": chunk.header, "text": chunk.text, "n_tokens": chunk.n_tokens,
        "url": urls.get(chunk.file, ""),
    }
