"""Dry runs over the tuning-set index: the context each question gets.

The index holds the twelve files the tuning rows expect, BM25 only, built
once per session. Dense assertions run only when the embedding model
cache is present.
"""

import os
import re

import pytest

import config
from ask import ask
from conftest import MODEL_CACHE, ROOT
from index import build, load
from test_rules import BANK_BRIEF

FULL_INDEX_DIR = os.path.join(ROOT, config.INDEX_DIR)
FULL_INDEX_BUILT = os.path.exists(os.path.join(FULL_INDEX_DIR, "fingerprint.json"))


def nii_row(figure: str) -> re.Pattern:
    """A "Net interest income" table row one of whose cells holds `figure`:
    the label is the first pipe cell (a footnote marker may follow it) and
    the figure fills a later cell, with or without a dollar sign."""
    return re.compile(r"^Net interest income[^|\n]*(?:\|[^|\n]*)*?\|\s*\$?" + re.escape(figure) + r"\s*(?:\||$)", re.M)


def dry_run(question, index, registry, **kw):
    payload = ask(question, index, registry, dry_run=True, **kw)
    assert payload["status"] == "ok", payload["status"]
    return payload


def rows_of(payload, ticker):
    return [r for r in payload["context"] if r["ticker"] == ticker]


def test_q01_dry_run_is_balanced_and_numbered(tuning_index, registry):
    payload = dry_run("What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do "
                      "they compare?", tuning_index, registry)
    for ticker in ("AAPL", "TSLA", "JPM"):
        assert len(rows_of(payload, ticker)) >= 6, ticker
    assert [r["cid"] for r in payload["context"]] == ["C%d" % n for n in range(1, len(payload["context"]) + 1)]
    assert payload["budget"]["tokens_est"] <= 20000
    assert payload["llm_attempts"] == 0 and payload["llm_completed"] == 0
    assert "[C1] " in payload["rendered"]


def test_bank_brief_reaches_both_nii_rows(tuning_index, registry):
    payload = dry_run(BANK_BRIEF, tuning_index, registry)
    by_id = tuning_index.by_id
    jpm = [by_id[r["chunk_id"]].text for r in rows_of(payload, "JPM")]
    bac = [by_id[r["chunk_id"]].text for r in rows_of(payload, "BAC")]
    assert any(nii_row("23,966").search(t) for t in jpm)
    assert any(nii_row("15,233").search(t) for t in bac)
    coverage = payload["coverage"]
    for name in ("10-Q FY2025 Q3", "10-K FY2025", "10-K FY2024"):
        assert name in coverage, coverage
    assert coverage.count("10-Q FY2025 Q3") == 2


@pytest.fixture(scope="module")
def full_index():
    return load(FULL_INDEX_DIR)


@pytest.mark.skipif(not FULL_INDEX_BUILT, reason="full index not built")
def test_bank_brief_on_the_full_index_seats_both_nii_tables(full_index, registry):
    # The twelve-file index ranks the BAC summary income statement near the
    # top for "net interest income"; over 64,612 chunks the same table sat
    # 43rd of 60 and missed the quota. The row-label seat is what carries
    # it on the full corpus, so this is the index the guard runs on.
    payload = dry_run(BANK_BRIEF, full_index, registry)
    by_id = full_index.by_id
    jpm = [by_id[r["chunk_id"]] for r in rows_of(payload, "JPM")]
    bac_rows = rows_of(payload, "BAC")
    assert any(nii_row("23,966").search(c.text) for c in jpm)
    bac_hits = [r for r in bac_rows if nii_row("15,233").search(by_id[r["chunk_id"]].text)]
    assert bac_hits, [r["header"] for r in bac_rows]
    # 15,233 is the quarter's figure: the chunk's parsed columns name a
    # three-months column ending on the quarter's period end.
    assert any(col.duration == "three_months" and col.period_end == "2025-09-30"
               for r in bac_hits for col in by_id[r["chunk_id"]].columns)
    assert any(r["pinned"] == "row label" for r in bac_hits)
    assert any(q["ticker"] == "BAC" and q["row_label"] for q in payload["plan"].quotas)
    assert "1,548.1" in payload["rendered"]


def test_bac_risk_question_uses_the_10k_item_1a(tuning_index, registry):
    payload = dry_run("What are Bank of America's main risk factors?", tuning_index, registry)
    assert any(r["file"] == "BAC_10K_2025-02-25_full.txt" and r["item"] == "1A" for r in payload["context"])
    assert "Part II Item 1A points to the annual report" in payload["coverage"]


def test_nvda_segment_question_reaches_a_segment_note(tuning_index, registry):
    payload = dry_run("How does NVIDIA break down its revenue by segment?", tuning_index, registry)
    assert any(r["note_title"] and "Segment" in r["note_title"] for r in payload["context"])


def test_apple_fiscal_2023_net_sales_figure_is_present(tuning_index, registry):
    payload = dry_run("What were Apple's total net sales in fiscal 2023?", tuning_index, registry)
    assert "383,285" in payload["rendered"]
    assert all(r["file"] == "AAPL_10K_2023Q3_2023-11-03_full.txt" for r in payload["context"])


def test_timeline_mode_keeps_two_nvda_fiscal_years(tuning_index, registry):
    payload = dry_run("How has NVIDIA's revenue and growth outlook changed over the last two years?",
                      tuning_index, registry)
    assert payload["plan"].timeline
    assert {r["fiscal_label"] for r in payload["context"]} >= {"FY2024", "FY2025"}
    # Chronological order in timeline mode: the older year comes first.
    labels = [r["fiscal_label"] for r in payload["context"]]
    assert labels.index("FY2024") < labels.index("FY2025")


def test_refusals_carry_no_context(tuning_index, registry):
    payload = ask("What did Wells Fargo say about credit risk?", tuning_index, registry, dry_run=True)
    assert payload["status"] == "not_covered" and "context" not in payload
    assert "Wells Fargo" in payload["coverage"]
    payload = ask("What did Apple report in 2019?", tuning_index, registry, dry_run=True)
    assert payload["status"] == "period_not_covered"


def test_no_pin_mode_drops_the_pinned_lead(tuning_index, registry):
    pinned = dry_run("What are Bank of America's main risk factors?", tuning_index, registry)
    unpinned = dry_run("What are Bank of America's main risk factors?", tuning_index, registry, pin=False)
    assert any(r["pinned"] for r in pinned["context"])
    assert not any(r["pinned"] for r in unpinned["context"])


@pytest.mark.skipif(not os.path.isdir(MODEL_CACHE), reason="embedding model cache missing at " + MODEL_CACHE)
def test_dense_mode_on_a_dense_index(tmp_path_factory, registry):
    # One filing keeps the dense build under a minute; the quarter bucket
    # then has no 10-K baseline, which the assertion does not need.
    files = ["JPM_10Q_2025Q3_2025-11-04_full.txt"]
    path = str(tmp_path_factory.mktemp("index-dense-small"))
    build(path, dense=True, files=files)
    index = load(path)
    payload = dry_run("What was JPMorgan's net interest income for the third quarter of 2025?",
                      index, registry, mode="dense")
    assert "23,966" in payload["rendered"]
