"""Settings for the filing service, read once from the environment.

Every default works on a laptop checkout and inside compose, so a fresh clone
runs `python service/index.py report` with no setup. Values that a deployment
changes come from environment variables; values that only change when the
code changes (CHUNKER_VERSION, budgets) are plain constants so a bump is a
reviewed diff, never a quiet environment drift.
"""

import os

# Where the raw filings live and where derived artifacts are written.
CORPUS_ZIP = os.environ.get("CORPUS_ZIP", "data/edgar_corpus.zip")
INDEX_DIR = os.environ.get("INDEX_DIR", "index")
COMPANIES_FILE = os.environ.get("COMPANIES_FILE", "service/companies.yaml")

# Retrieval settings. Dense retrieval is the default: measured over the full
# index on a 48-question set, lexical search never reached the correct passage
# at all in 7 of them, and on questions phrased in the reader's own words
# rather than the filing's it reached it in 4 of 10. Those numbers are in
# eval/notes.md. SEARCH_MODE "dense" is not a fusion: reciprocal rank fusion
# with the lexical ranking measured worse on exactly those questions, because
# it averages in a ranking that is close to random there.
DENSE = os.environ.get("DENSE", "1")
SEARCH_MODE = os.environ.get("SEARCH_MODE", "dense")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
# BGE is asymmetric: the query carries an instruction and the passage does
# not. The library does not add this, so the query path must, or query and
# passage vectors come from two different distributions and retrieval
# degrades with nothing in the logs to say so.
EMBED_QUERY_PREFIX = os.environ.get(
    "EMBED_QUERY_PREFIX", "Represent this sentence for searching relevant passages: ")
# "cpu" or "cuda". The dense build is the only heavy embedding work, and on
# CPU it takes about nine hours for this corpus against under two minutes on
# a GPU, so the indexer-gpu service sets this to cuda. Query embedding at
# request time is 130 ms on CPU and stays there.
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")
# The model reads 512 tokens. A table chunk's caption, unit line and header
# rows sit ahead of that cut, and the header line is embedded first.
EMBED_MAX_TOKENS = 512
# Sequences per ONNX call. A 256-sequence batch at 512 tokens allocated
# 9 GB on this host; never raise this above 64.
EMBED_BATCH = 32
# Embedding processes; each loads its own copy of the model.
# Overridable because the GPU build wants exactly one: each worker loads its
# own copy of the model, which on a single GPU is four copies competing for
# the same device.
EMBED_WORKERS = int(os.environ.get("EMBED_WORKERS", max(1, min(4, (os.cpu_count() or 4) // 4))))
# fastembed's model cache. The repo checkout ships the model files here so a
# dense build never downloads anything.
FASTEMBED_CACHE = os.environ.get(
    "FASTEMBED_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".fastembed"))

# Bumped by hand whenever chunk boundaries or chunk metadata change, so an
# index built by older code is rebuilt instead of silently reused.
CHUNKER_VERSION = "2"

WEB_PORT = int(os.environ.get("WEB_PORT", "8804"))

# Model client settings. "fake" answers from stored fixtures or a canned
# minimal answer so the whole pipeline runs with no credentials; "anthropic"
# is the one API backend. The request shape (effort, output cap, timeout,
# retries) is constant so a change to it is a reviewed diff.
LLM_MODEL_BACKEND = os.environ.get("LLM_MODEL_BACKEND", "fake")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-opus-5")
LLM_EFFORT = "medium"
LLM_MAX_TOKENS = 16000
LLM_TIMEOUT_SECONDS = 300
# Zero SDK retries: an ask makes exactly one request, so a failure is
# reported as a failure instead of being retried at a second cost.
LLM_MAX_RETRIES = 0
# Stored model replies the fake backend serves, keyed by question.
FIXTURES_DIR = os.environ.get("FIXTURES_DIR", "eval/fixtures")

# Context budgets in tokens, chosen per question shape in milestone 3.
BUDGET_TOKENS_SMALL = 20000
BUDGET_TOKENS_LARGE = 30000
MAX_COMPANIES = 6

# USD per million tokens from the claude-api skill's model notes, read on
# the date in "checked": Claude Opus 5 is listed at $5/$25 per MTok. The
# same notes give no per-token price for claude-sonnet-5, so it is left
# out. Cache writes bill at 1.25x the input rate and cache reads at 0.1x
# (same notes, "Prompt Caching"). A model absent here reports cost_usd
# None rather than a guess.
PRICES: dict[str, dict] = {
    "claude-opus-5": {"input": 5.00, "output": 25.00, "checked": "2026-09-07"},
}
CACHE_WRITE_FACTOR = 1.25
CACHE_READ_FACTOR = 0.10

# The environment variable the API backend reads its key from. The web
# layer reports llm_ready from its presence alone; nothing here pings the
# provider.
LLM_API_KEY_ENV = "ANTHROPIC_API_KEY"
