"""Parser acceptance tests against the real corpus zip.

The corpus is parsed once per module (about a minute) and every test reads
from that one list, so adding a test costs nothing at run time. Rates are
asserted against the thresholds the milestone promises; single-file checks
pin the cases that drove a parsing rule (a pointer stub, a title-only
filer, a 52/53-week fiscal calendar) so a regression names the rule.
"""

import datetime as dt
import os
import re
import zipfile

import pytest

from corpus import (
    HEADER_RULE,
    SPAN_BOUNDS,
    body_start,
    load_corpus,
    normalize,
    toc_span,
)

ZIP = os.path.join(os.path.dirname(__file__), "..", "data", "edgar_corpus.zip")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# A delimited comma-formatted number. The preamble glues XBRL values into
# digit runs ("false1,785,288,846176.61,435"), which are fragments, never
# facts, so a match may not touch another digit, comma, or period.
COMMA_NUMBER_RE = re.compile(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?![\d.,])")


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(ZIP)


def one(corpus, prefix):
    hits = [f for f in corpus if f.file.startswith(prefix)]
    assert len(hits) == 1, prefix
    return hits[0]


def section(filing, item):
    for s in filing.sections:
        if s.item == item:
            return s
    return None


def tenk(corpus):
    return [f for f in corpus if f.form == "10-K"]


# --- whole-corpus invariants -------------------------------------------------


def test_every_filing_parses_with_a_dated_body(corpus):
    assert len(corpus) == 246
    for f in corpus:
        assert f.body, f.file
        assert DATE_RE.match(f.period_end), f.file
        assert f.period_source in {"header", "cover", "url"}, f.file
        gap = dt.date.fromisoformat(f.filing_date) - dt.date.fromisoformat(f.period_end)
        assert dt.timedelta(0) <= gap <= dt.timedelta(days=120), (f.file, gap)


def test_no_certification_bodies(corpus):
    assert all("I have reviewed this" not in f.body for f in corpus)


# --- fiscal labels -----------------------------------------------------------


@pytest.mark.parametrize(
    "prefix,label",
    [
        ("AAPL_10Q_2025Q4_2026-01-30", "FY2026 Q1"),
        ("AAPL_10K_2024Q3_2024-11-01", "FY2024"),
        ("NVDA_10Q_2025Q4_2025-11-19", "FY2026 Q3"),
        ("NVDA_10K_2025-02-26", "FY2025"),
        ("MSFT_10K_2024Q2", "FY2024"),
    ],
)
def test_fiscal_labels(corpus, prefix, label):
    assert one(corpus, prefix).fiscal_label == label


def test_ge_period_end_comes_from_the_cover(corpus):
    ge = one(corpus, "GE_")
    assert ge.period_end == "2014-12-31"
    assert ge.period_source == "cover"


# --- 10-K sections -----------------------------------------------------------


def test_10k_risk_factors_found_in_bounds(corpus):
    ks = tenk(corpus)
    lo, hi = SPAN_BOUNDS[("10-K", "1A")]
    ok = [f for f in ks if section(f, "1A") and lo <= section(f, "1A").end - section(f, "1A").start <= hi]
    assert len(ok) >= 0.95 * len(ks), [f.file for f in ks if f not in ok]


def test_10k_risk_factors_start_after_the_toc(corpus):
    # MCD_10K_2025 is the one file in the corpus whose page-referenced index
    # sits at the end of the report instead of the front; the region then
    # lies entirely after the body headings, which is the other acceptable
    # side. Every other 10-K with a region passes on the first clause.
    for f in tenk(corpus):
        s = section(f, "1A")
        start, end = toc_span(f.body, f.form)
        if s and end:
            assert s.start >= end or start > s.start, (f.file, s.start, (start, end))


def test_10k_mdna_found(corpus):
    ks = tenk(corpus)
    found = [f for f in ks if section(f, "7")]
    assert len(found) >= 0.95 * len(ks), [f.file for f in ks if f not in found]


def test_10k_financial_statements_found(corpus):
    ks = tenk(corpus)
    found = [f for f in ks if section(f, "8")]
    assert len(found) >= 0.90 * len(ks), [f.file for f in ks if f not in found]


# --- 10-Q sections -----------------------------------------------------------


def test_bac_10q_has_mdna_and_a_pointer_stub_for_risk_factors(corpus):
    bac = one(corpus, "BAC_10Q_2025Q3")
    assert section(bac, "I.2") is not None
    assert section(bac, "II.1A").is_pointer_stub


def test_jnj_10q_has_no_part_ii_risk_factors(corpus):
    assert section(one(corpus, "JNJ_10Q_2025Q3"), "II.1A") is None


def test_jpm_10q_risk_factors_is_a_pointer_stub(corpus):
    assert section(one(corpus, "JPM_10Q_2025Q3"), "II.1A").is_pointer_stub


# --- notes -------------------------------------------------------------------


def test_10k_item_8_carries_notes(corpus):
    with_8 = [f for f in tenk(corpus) if section(f, "8")]
    enough = [f for f in with_8 if len(section(f, "8").notes) >= 5]
    assert len(enough) >= 0.90 * len(with_8), [
        (f.file, len(section(f, "8").notes)) for f in with_8 if f not in enough]


def test_aapl_has_a_revenue_note(corpus):
    titles = [n.title for n in section(one(corpus, "AAPL_10K_2024Q3"), "8").notes]
    assert any(t.startswith("Revenue") for t in titles), titles


# --- noise -------------------------------------------------------------------


def test_ms_running_headers_removed_but_table_headers_kept(corpus):
    lines = one(corpus, "MS_10K_2026-02-19").body.split("\n")
    assert not [l for l in lines if l.startswith("Table of Contents |")]
    assert any(l.startswith("$ in millions | 2025 | 2024 | 2023") for l in lines)


@pytest.mark.parametrize(
    "prefix",
    ["AAPL_10K_2024Q3", "JPM_10K_2026-02-13", "NVDA_10K_2025-02-26", "PFE_10Q_2025Q3"],
)
def test_preamble_holds_no_financial_values(prefix):
    with zipfile.ZipFile(ZIP) as zf:
        name = [n for n in zf.namelist() if n.startswith(prefix)][0]
        text = normalize(zf.read(name).decode("utf-8"))
    start = body_start(text)
    preamble = text[text.index(HEADER_RULE) + len(HEADER_RULE):start]
    body = text[start:]
    missing = [n for n in set(COMMA_NUMBER_RE.findall(preamble)) if n not in body]
    assert missing == []


# --- normalize ---------------------------------------------------------------


def test_normalize_folds_unicode_variants():
    assert normalize("a\u00a0b") == "a b"
    assert normalize("\u201cq\u201d \u2018s\u2019") == '"q" \'s\''
    assert normalize("x\u2014y \u2013 z") == "x-y - z"
    assert normalize("a  \t b\nc") == "a b\nc"
