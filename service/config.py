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

# Model client settings. "fake" answers from canned text so the whole pipeline
# runs with no credentials.
LLM_MODEL_BACKEND = os.environ.get("LLM_MODEL_BACKEND", "fake")
LLM_MODEL = os.environ.get("LLM_MODEL", "")

# Context budgets in tokens, chosen per question shape in milestone 3.
BUDGET_TOKENS_SMALL = 20000
BUDGET_TOKENS_LARGE = 30000
MAX_COMPANIES = 6

# Per-million-token prices keyed by model name, filled in with the date they
# were checked once the backend is chosen. Empty means cost is not reported.
PRICES: dict[str, dict[str, float]] = {}
