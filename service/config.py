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

# Retrieval settings. DENSE is off by default because a full-corpus dense
# build takes tens of minutes on a laptop; `index.py build --dense 1` turns
# it on for one build. The model is fastembed's fp32 MiniLM-L6: 384
# dimensions, six layers, about 17 chunks per second per process here.
DENSE = os.environ.get("DENSE", "0")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# The model's tokenizer file truncates at 128 tokens; 256 keeps a table
# chunk's caption, unit line and header rows ahead of the cut. The header
# line is embedded first for the same reason.
EMBED_MAX_TOKENS = 256
# Sequences per ONNX call. A 256-sequence batch at 512 tokens allocated
# 9 GB on this host; never raise this above 64.
EMBED_BATCH = 32
# Embedding processes; each loads its own copy of the model.
EMBED_WORKERS = max(1, min(4, (os.cpu_count() or 4) // 4))
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
