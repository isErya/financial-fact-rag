"""make_plan: company resolution, periods, buckets, sub-queries, quotas.

Every case is a question a reader of the spec would type, with the scope
it must produce. The file list is the whole corpus (from the registry) so
the refusals and windows see every Apple and NVIDIA filing.
"""

import json
import time

import pytest

import rules

BANK_BRIEF = ("Prepare a Q3 2025 bank-disclosure brief for a PE portfolio CFO. Compare JPMorgan and "
              "Bank of America on CET1 ratio, estimated uninsured deposits, and third-quarter net "
              "interest income. State the reporting period and units, cite each figure, and flag "
              "disclosures that are not comparable.")


@pytest.fixture(scope="module")
def tuning_rows():
    with open("eval/tuning.jsonl") as fh:
        return {r["id"]: r for r in (json.loads(l) for l in fh if l.strip())}


def plan(question, registry, corpus_files):
    return rules.make_plan(question, registry, corpus_files)


def tickers(p):
    return [c["ticker"] for c in p.companies]


def bucket(p, ticker, label):
    hits = [b for b in p.buckets if b["ticker"] == ticker and b["label"] == label]
    assert len(hits) == 1, [(b["ticker"], b["label"]) for b in p.buckets]
    return hits[0]


# --- company resolution ------------------------------------------------------


@pytest.mark.parametrize("question, expected", [
    ("What revenue target does Apple set for services?", ["AAPL"]),
    ("chase growth in deposits", []),
    ("meta-analysis of drug trials", []),
    ("MS", []),
    ("$MS", ["MS"]),
    ("JPMorgan", ["JPM"]),
    ("Morgan Stanley", ["MS"]),
    ("Bank of America and JPMorgan", ["BAC", "JPM"]),
])
def test_aliases_and_tickers(question, expected, registry, corpus_files):
    assert tickers(plan(question, registry, corpus_files)) == expected


def test_group_phrases(registry, corpus_files):
    assert set(tickers(plan("the big banks", registry, corpus_files))) == {"JPM", "BAC", "GS", "MS"}
    pharma = {"JNJ", "PFE", "MRK", "LLY", "ABBV"}
    assert set(tickers(plan("major pharmaceutical companies", registry, corpus_files))) == pharma
    assert set(tickers(plan("drugmakers", registry, corpus_files))) == pharma
    assert tickers(plan("a bank's risk", registry, corpus_files)) == []
    # GE sits in the industrials group so the stale rule has something to
    # exclude: the group phrase skips it, a direct mention still resolves
    # it, and both paths mark it stale.
    assert "GE" in registry["groups"]["industrials"]["tickers"]
    assert "GE" in rules.stale_tickers(registry["companies"])
    industrials = tickers(plan("industrial companies", registry, corpus_files))
    assert industrials and "GE" not in industrials
    assert {"CAT", "DE"} <= set(industrials)
    named = plan("What is General Electric's outlook?", registry, corpus_files)
    assert tickers(named) == ["GE"] and named.stale == ["GE"]
    assert any("GE" in n and "stale" in n for n in named.notes)


def test_matched_alias_is_recorded(registry, corpus_files):
    p = plan("Bank of America and JPMorgan", registry, corpus_files)
    assert [c["matched_alias"] for c in p.companies] == ["Bank of America", "JPMorgan"]
    p = plan("the big banks", registry, corpus_files)
    assert p.companies[0]["matched_alias"] == "big banks"
    p = plan("What regulatory risks do the major pharmaceutical companies face?", registry, corpus_files)
    assert p.companies[0]["matched_alias"] == "major pharmaceutical companies"


def test_tuning_questions_resolve_expected_tickers(tuning_rows, registry, corpus_files):
    for qid in ("q01", "q02", "q03"):
        row = tuning_rows[qid]
        assert tickers(plan(row["question"], registry, corpus_files)) == row["expected_tickers"], qid


# --- periods and buckets -----------------------------------------------------


def test_q02_is_a_two_year_timeline(tuning_rows, registry, corpus_files):
    p = plan(tuning_rows["q02"]["question"], registry, corpus_files)
    assert p.timeline and not p.comparison and p.period_mode == "window"
    nvda_years = sorted({rules.fiscal_parts(r["fiscal_label"])[0] for r in corpus_files if r["ticker"] == "NVDA"})
    labels = [b["label"] for b in p.buckets]
    # The newest two fiscal years in the corpus are covered, and the two
    # newest complete years (those with a 10-K) are among the buckets.
    assert {"FY%d" % y for y in nvda_years[-2:]} <= set(labels)
    tenk_years = sorted({rules.fiscal_parts(r["fiscal_label"])[0] for r in corpus_files
                         if r["ticker"] == "NVDA" and r["form"] == "10-K"})
    assert {"FY%d" % y for y in tenk_years[-2:]} <= set(labels)


def test_bac_latest_pins_the_10k_risk_section(registry, corpus_files):
    p = plan("What are Bank of America's main risk factors?", registry, corpus_files)
    b = bucket(p, "BAC", "latest")
    reasons = {f["file"]: f["reason"] for f in b["files"]}
    assert reasons["BAC_10K_2025-02-25_full.txt"] == "annual_baseline"
    assert reasons["BAC_10Q_2025Q3_2025-10-31_full.txt"] == "newest_10q"
    tenk = next(r for r in corpus_files if r["file"] == "BAC_10K_2025-02-25_full.txt")
    item_1a = next(s for s in tenk["sections"] if s["item"] == "1A")
    assert p.quotas[0]["pinned"] == [item_1a["chunk_start"], item_1a["chunk_start"] + 1]
    assert any("Item 1A" in n and "10-K" in n for n in p.notes)


def test_apple_2019_is_refused_with_the_available_labels(registry, corpus_files):
    p = plan("What did Apple report in 2019?", registry, corpus_files)
    assert p.status == "period_not_covered"
    available = p.companies[0]["available"]
    assert available[0].startswith("FY2022") and available[-1] == "FY2026 Q1"
    assert p.buckets == []


def test_apple_window_2019_to_2025(registry, corpus_files):
    p = plan("Summarize Apple from 2019 to 2025", registry, corpus_files)
    assert p.status == "ok" and p.period_mode == "window"
    assert [b["label"] for b in p.buckets] == ["FY2022", "FY2023", "FY2024", "FY2025"]
    assert p.companies[0]["missing_years"] == [2019, 2020, 2021]
    assert any("2019, 2020, 2021" in n for n in p.not_covered)


def test_bare_year_follows_each_fiscal_calendar(registry, corpus_files):
    p = plan("How did Apple and NVIDIA do in 2024?", registry, corpus_files)
    assert bucket(p, "AAPL", "FY2024")["period_end"] == "2024-09-28"
    assert bucket(p, "NVDA", "FY2024")["period_end"] == "2024-01-28"
    assert p.comparison and not p.timeline


def test_jpm_quarter_bucket_and_bank_brief_sub_queries(registry, corpus_files):
    p = plan("JPMorgan net interest income for the third quarter of 2025", registry, corpus_files)
    b = bucket(p, "JPM", "FY2025 Q3")
    reasons = {f["file"]: f["reason"] for f in b["files"]}
    assert reasons["JPM_10Q_2025Q3_2025-11-04_full.txt"] == "newest_10q"
    assert reasons["JPM_10K_2026-02-13_full.txt"] == "annual_baseline"
    brief = plan(BANK_BRIEF, registry, corpus_files)
    # Three metric sub-queries; "third-quarter" is dropped from the last
    # because the quarter bucket already fixes the period.
    assert brief.sub_queries[1:] == ["CET1 ratio", "estimated uninsured deposits", "net interest income"]
    assert tickers(brief) == ["JPM", "BAC"] and brief.period_mode == "quarter"


def test_wells_fargo_is_not_covered_and_ge_is_stale(registry, corpus_files):
    p = plan("What did Wells Fargo say about credit risk?", registry, corpus_files)
    assert p.status == "not_covered" and p.unresolved == ["Wells Fargo"]
    p = plan("What is GE's outlook for 2024?", registry, corpus_files)
    assert tickers(p) == ["GE"] and p.stale == ["GE"]


# --- sections, quotas, budget ------------------------------------------------


def test_section_intents(registry, corpus_files):
    risk = plan("What are Apple's risk factors?", registry, corpus_files).sections
    assert risk["1A"] == 1.6 and risk["II.1A"] == 1.6 and risk["*"] == 0.6
    segment = plan("How does NVIDIA break down revenue by segment?", registry, corpus_files)
    assert segment.sections["8"] == 1.6 and segment.note_intent == "segment"
    default = plan("Tell me about Apple", registry, corpus_files).sections
    assert default["1A"] == 1.2 and default["*"] == 0.7


def test_quota_math_for_three_companies(registry, corpus_files):
    p = plan("Compare Apple, Tesla and JPMorgan", registry, corpus_files)
    assert p.budget_tokens == 20000 and len(p.quotas) == 3
    assert [q["chunks"] for q in p.quotas] == [16, 16, 16]
    assert rules.quota_chunks(20000, 1) == 24 and rules.quota_chunks(20000, 20) == 6


def test_more_than_six_companies_are_narrowed(registry, corpus_files):
    p = plan("Compare the big banks and the major pharmaceutical companies on capital", registry, corpus_files)
    assert len(p.companies) == 6 and p.budget_tokens == 30000
    assert any("narrowed" in n for n in p.notes)


def test_make_plan_is_fast(registry, corpus_files):
    started = time.perf_counter()
    for _ in range(5):
        plan(BANK_BRIEF, registry, corpus_files)
    assert (time.perf_counter() - started) / 5 < 0.1
