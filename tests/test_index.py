"""Index build and search over three filings.

The BM25 index is built once per module into a temp dir; the dense index
is built once more in its own dir and skipped with a reason when the model
cache is absent. Both searches must put a JPMorgan chunk holding the
third-quarter net interest income figure near the top for a question that
names the figure's period. JPMorgan prints that row in several tables
("Net interest income | 23,966" in the financial highlights, "Net interest
income - reported(a) | $23,966" in the reconciliation), and any of them is
the right chunk to hand the model.
"""

import os
import re
import time

import numpy as np
import pytest

from index import build, load

FILES = [
    "JPM_10Q_2025Q3_2025-11-04_full.txt",
    "BAC_10Q_2025Q3_2025-10-31_full.txt",
    "AAPL_10K_2024Q3_2024-11-01_full.txt",
]
QUERY = "net interest income three months ended September 30 2025"
ROW_RE = re.compile(r"^Net interest income[^|\n]*\| \$?23,966(?: \||$)", re.M)
import config
from conftest import MODEL_CACHE


def holds_the_row(index, hits) -> bool:
    return any(ROW_RE.search(index.by_id[chunk_id].text) for chunk_id, *_ in hits)


@pytest.fixture(scope="module")
def bm25_dir(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("index-bm25"))
    build(path, dense=False, files=FILES)
    return path


def jpm_ids(index):
    start, end = index.ticker_ranges["JPM"]
    return np.arange(start, end)


def test_layout(bm25_dir):
    for name in ("chunks.jsonl.gz", "bm25", "ticker_ranges.json", "files.json", "fingerprint.json"):
        assert os.path.exists(os.path.join(bm25_dir, name)), name
    assert not os.path.exists(os.path.join(bm25_dir, "dense.npy"))
    index = load(bm25_dir)
    assert index.dense is None
    assert len(index.files) == 3
    assert sorted(index.ticker_ranges) == ["AAPL", "BAC", "JPM"]
    # Ticker ranges tile the chunk list in order, with no gaps.
    spans = sorted(index.ticker_ranges.values())
    assert spans[0][0] == 0 and spans[-1][1] == len(index.chunks)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    for record in index.files:
        for section in record["sections"]:
            assert record["chunk_start"] <= section["chunk_start"] <= section["chunk_end"] <= record["chunk_end"]


def test_bm25_search_finds_the_jpm_row(bm25_dir):
    index = load(bm25_dir)
    hits = index.search(QUERY, jpm_ids(index), 3)
    assert len(hits) == 3
    assert holds_the_row(index, hits), hits
    assert all(dense_rank is None for _c, _b, dense_rank, _r in hits)


def test_search_stays_inside_the_given_ids(bm25_dir):
    index = load(bm25_dir)
    start, end = index.ticker_ranges["AAPL"]
    hits = index.search(QUERY, np.arange(start, end), 5)
    assert all(index.by_id[chunk_id].ticker == "AAPL" for chunk_id, *_ in hits)
    assert index.search(QUERY, np.array([], dtype=np.int64), 5) == []


def test_all_stopword_query_scores_zero(bm25_dir):
    index = load(bm25_dir)
    scores = index.bm25_scores("the of and")
    assert scores.shape == (len(index.chunks),) and not scores.any()


def test_fingerprint_short_circuits_a_rebuild(bm25_dir, capsys):
    before = os.path.getmtime(os.path.join(bm25_dir, "chunks.jsonl.gz"))
    record = build(bm25_dir, dense=False, files=FILES)
    assert "index up to date" in capsys.readouterr().out
    assert os.path.getmtime(os.path.join(bm25_dir, "chunks.jsonl.gz")) == before
    assert record["chunks"] == len(load(bm25_dir).chunks)


@pytest.mark.skipif(not os.path.isdir(MODEL_CACHE), reason="embedding model cache missing at " + MODEL_CACHE)
def test_dense_hybrid_search(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("index-dense"))
    started = time.perf_counter()
    record = build(path, dense=True, files=FILES)
    wall = time.perf_counter() - started
    print("dense build for %d files: %.1fs wall, phases %s" % (len(FILES), wall, record["phase_seconds"]))
    index = load(path)
    assert index.dense is not None
    assert index.dense.shape == (len(index.chunks), 384)
    assert np.allclose(np.linalg.norm(index.dense[:50], axis=1), 1.0, atol=1e-3)
    hits = index.search(QUERY, jpm_ids(index), 5)
    assert holds_the_row(index, hits), hits
    assert all(dense_rank is not None for _c, _b, dense_rank, _r in hits)
    assert "dense" in record["phase_seconds"]


def test_model_cache_path_follows_fastembed_source():
    """The dense tests skip when this directory is absent, so it must be the
    one fastembed fills (named after the source repo, not the model id)."""
    from fastembed import TextEmbedding
    sources = {m["model"]: m.get("sources") or {} for m in TextEmbedding.list_supported_models()}
    repo = sources[config.EMBED_MODEL].get("hf", config.EMBED_MODEL)
    assert os.path.basename(MODEL_CACHE) == "models--" + repo.replace("/", "--")
