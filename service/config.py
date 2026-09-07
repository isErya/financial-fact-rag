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

# Retrieval settings; consumed from milestone 2 on.
DENSE = os.environ.get("DENSE", "1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Bumped by hand whenever chunk boundaries or chunk metadata change, so an
# index built by older code is rebuilt instead of silently reused.
CHUNKER_VERSION = "1"

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
