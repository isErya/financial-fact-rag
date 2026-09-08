"""Shared fixtures for the milestone 3 tests.

The BM25-only index over the tuning-set files is built once per session
(about ten seconds) and read by both the rules and the retrieval tests.
The rules tests also need the whole corpus's filing list, since "Apple in
2019" is refused by looking at every Apple filing; that list comes from
the registry (companies.yaml holds every filing's form and period end)
with fiscal labels assigned by the corpus code, and the section layout of
the twelve indexed files merged in.
"""

import json
import os

import pytest

import config
from corpus import assign_fiscal_labels
from index import build, load, load_registry
from models import Filing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNING = os.path.join(ROOT, "eval", "tuning.jsonl")


def model_cache_dir() -> str:
    """Directory fastembed fills when it downloads config.EMBED_MODEL.

    fastembed names the cache after the Hugging Face repo it pulls from, and
    that repo is the model's registered source (qdrant/all-MiniLM-L6-v2-onnx
    for the MiniLM id), so a path derived from the model id itself points at
    a directory a fresh download never creates. The dense tests check this
    path to decide whether to run, so the mapping has to be the real one.
    """
    repo = config.EMBED_MODEL
    try:
        from fastembed import TextEmbedding
        for entry in TextEmbedding.list_supported_models():
            if entry.get("model") == config.EMBED_MODEL:
                repo = (entry.get("sources") or {}).get("hf") or repo
                break
    except Exception:
        pass
    return os.path.join(config.FASTEMBED_CACHE, "models--" + repo.replace("/", "--"))


MODEL_CACHE = model_cache_dir()


def tuning_files() -> list[str]:
    names = set()
    with open(TUNING) as fh:
        for line in fh:
            if line.strip():
                names.update(json.loads(line).get("expected_files") or [])
    return sorted(names)


@pytest.fixture(scope="session")
def tuning_index_dir(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("index-tuning-bm25"))
    build(path, dense=False, files=tuning_files())
    return path


@pytest.fixture(scope="session")
def tuning_index(tuning_index_dir):
    return load(tuning_index_dir)


@pytest.fixture(scope="session")
def registry():
    return load_registry(os.path.join(ROOT, config.COMPANIES_FILE))


@pytest.fixture(scope="session")
def corpus_files(registry, tuning_index):
    """files.json-shaped records for every filing in the registry, with the
    sections of the twelve tuning-set files taken from the real index."""
    companies = registry["companies"]
    filings = []
    for ticker, entry in companies.items():
        for f in entry["filings"]:
            filings.append(Filing(
                file=f["file"], cik=entry["cik"], ticker=ticker, company=entry["name"], form=f["form"],
                filing_date=f["filing_date"], period_end=f["period_end"], period_source="registry",
                fiscal_year=0, fiscal_quarter=None, fiscal_label="", fye_month=entry["fye_month"],
                url="", body=""))
    assign_fiscal_labels(filings, companies)
    indexed = {r["file"]: r for r in tuning_index.files}
    records = []
    for f in filings:
        record = indexed.get(f.file)
        if record is None:
            record = {"file": f.file, "ticker": f.ticker, "form": f.form, "period_end": f.period_end,
                      "fiscal_label": f.fiscal_label, "filing_date": f.filing_date,
                      "chunk_start": 0, "chunk_end": 0, "sections": []}
        else:
            assert record["fiscal_label"] == f.fiscal_label, f.file
        records.append(record)
    return records
