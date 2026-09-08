"""The evidence checks over the chunks the tuning questions really retrieve.

Every case here is built from real excerpts: `retrieved` runs ask.prepare()
for q04, q05, q06, q07 and q09, and each test pulls the chunk it needs out of
the context that question produced, so a table's columns, unit line and row
text are the ones the pipeline puts in front of the model rather than a
hand-made fixture that cannot go stale.

The block after the positive cases is one test per attack an adversarial
reviewer landed on the previous resolver: each is named for the defect it
would be, and asserts the outcome the contract requires.
"""

import dataclasses
import json
import os

import pytest

import resolver
from ask import prepare
from conftest import ROOT
from models import Answer, Chunk, Column, Context

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
QUESTIONS = {
    "q04": ("Prepare a Q3 2025 bank-disclosure brief for a PE portfolio CFO. Compare JPMorgan and Bank of "
            "America on CET1 ratio, estimated uninsured deposits, and third-quarter net interest income. "
            "State the reporting period and units, cite each figure, and flag disclosures that are not "
            "comparable."),
    "q05": "What were Apple's total net sales in fiscal 2023?",
    "q06": "What was JPMorgan's net interest income for the third quarter of 2025?",
    "q07": "How many diluted shares did Apple use in computing earnings per share in fiscal 2024?",
    "q09": "How does NVIDIA break down its revenue by segment?",
}

# Chunk ids, with the question whose context carries them.
NII = "55e0c02f62d5d7dc"          # q06: JPM Q3 2025 summary income statement, "except" unit line
CAPITAL = "1ae8975b3361a7c4"      # q04: JPM Note 20 capital table, Standardized/Advanced x Co./Bank
UNINSURED = "cf75f0872077d7af"    # q04: JPM prose, uninsured deposits, non-breaking spaces
BAC_PROSE = "f1593455cf409bc3"    # q04: BAC prose, "$606.8 billion and $116.6 billion"
APPLE_OPS = "328dc22e456d59fb"    # q07: FY2024 statement of operations, shares excepted into thousands
APPLE_EPS = "e72876a1e7dfffdc"    # q07: FY2024 EPS table, unit line inherited from an earlier excerpt
APPLE_2023 = "3204ddb0ccb5f6f2"   # q05: FY2023 MD&A segments, FY2022 and FY2021 columns carry no date
NVDA_2025 = "6d74cb8a253b7b93"    # q09: FY2025 segment table, clean "USD millions" unit line, seq 308
NVDA_2024 = "575bbdcb04aa0ac3"    # q09: the same table for the prior year, seq 309
# The q05 fixture reads the FY2023 figure out of the FY2024 10-K, which the
# q05 question itself does not retrieve; it is loaded from the index.
APPLE_MDA_2024 = "cffc6bfbcc7ea05f"

NII_ROW = "Net interest income | 23,966 | 23,405 | 2 | 70,448 | 69,233 | 2"
CET1_ROW = "CET1 capital ratio | 14.8% | 15.7% | 14.9% | 16.8%"
ROTCE_ROW = "Return on tangible common equity | 20 | 19 | 21 | 23"
DILUTED_ROW = "Diluted | 15,408,095 | 15,812,547 | 16,325,819"
APPLE_SALES_ROW = "Total net sales | 391,035 | 383,285 | 394,328"
APPLE_2023_ROW = "Total net sales | $383,285 | (3)% | $394,328 | 8% | $365,817"
NVDA_ROW = "Revenue | $116,193 | $14,304 | - | $130,497"
NVDA_2024_ROW = "Revenue | $47,405 | $13,517 | - | $60,922"


@pytest.fixture(scope="session")
def retrieved(tuning_index, registry):
    """Each tuning question's Prepared, so the chunks below are the ones
    retrieval really seated."""
    out = {}
    for key, question in QUESTIONS.items():
        prepared = prepare(question, tuning_index, registry)
        assert prepared.context is not None, key
        out[key] = prepared
    return out


def chunk_from(retrieved, key: str, chunk_id: str) -> Chunk:
    by_id = {c.chunk_id: c for c in retrieved[key].context.chunks}
    assert chunk_id in by_id, "%s did not retrieve %s" % (key, chunk_id)
    return by_id[chunk_id]


def one_chunk(chunk: Chunk, coverage: str = "") -> tuple[Context, dict]:
    context = Context(chunks=[chunk], n_tokens=chunk.n_tokens, cids=["C1"], coverage=coverage)
    return context, {"C1": chunk}


def claim(text: str, quote: str, period_end: str = "2025-09-30", kind: str = "quarter", ticker: str = "JPM",
          cid: str = "C1", claim_id: str = "K1") -> dict:
    return {"id": claim_id, "text": text, "tickers": [ticker], "period_end": period_end,
            "period_kind": kind, "citations": [cid], "quote": quote}


def answer_of(*claims: dict, summary=None, table=None, gaps=None, not_comparable=None) -> Answer:
    return Answer.model_validate({
        "summary": summary if summary is not None else [{"text": "s", "claim_ids": [c["id"] for c in claims]}],
        "claims": list(claims), "table": table or [], "not_comparable": not_comparable or [],
        "gaps": gaps or []})


def load_fixture(name: str, cid_of: dict[str, str]) -> Answer:
    with open(os.path.join(FIXTURES, name)) as fh:
        data = json.load(fh)
    for claim_data in data["claims"]:
        claim_data["citations"] = [cid_of[c[6:]] if c.startswith("chunk:") else c
                                   for c in claim_data["citations"]]
    return Answer.model_validate(data)


def flags(checks) -> list[str]:
    return [f["kind"] for f in checks.flags]


def notes(checks) -> list[str]:
    return [n["kind"] for n in checks.notes]


def detail(checks, kind: str) -> str:
    return next(r["detail"] for r in list(checks.flags) + list(checks.notes) if r["kind"] == kind)


# ---------------------------------------------------------------------------
# The outcomes that must stay right.
# ---------------------------------------------------------------------------


def test_the_jpm_quarterly_row_matches_its_quote_figure_and_column(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q06", NII))
    checks = resolver.check(answer_of(claim(
        "JPMorgan's net interest income was $23,966 million for the three months ended September 30, 2025.",
        NII_ROW)), context, by_cid)
    assert flags(checks) == []
    assert checks.quotes_located == (1, 1) and checks.figures_in_quote == (1, 1)
    assert checks.columns_matched == (1, 1) and checks.columns_unverified == 0
    # The filer's unit line excepts per-share data and ratios, so no scale is
    # established for any row of this table and the scale word goes unchecked.
    assert checks.units_matched == (0, 0) and checks.units_unchecked == 1
    assert notes(checks) == ["units_unchecked"]


def test_the_nvidia_segment_row_matches_all_four_checks(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q09", NVDA_2025))
    checks = resolver.check(answer_of(claim(
        "NVIDIA's Compute & Networking revenue was $116,193 million in fiscal 2025.", NVDA_ROW,
        period_end="2025-01-26", kind="fiscal_year", ticker="NVDA")), context, by_cid)
    assert flags(checks) == [] and notes(checks) == []
    assert checks.quotes_located == (1, 1) and checks.figures_in_quote == (1, 1)
    assert checks.columns_matched == (1, 1) and checks.units_matched == (1, 1)
    assert checks.figures_unchecked == 0 and checks.units_unchecked == 0


def test_a_restatement_in_billions_is_reported_as_a_unit_conversion(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q09", NVDA_2025))
    checks = resolver.check(answer_of(claim(
        "NVIDIA's Consolidated revenue was $130.5 billion in fiscal 2025.", NVDA_ROW,
        period_end="2025-01-26", kind="fiscal_year", ticker="NVDA")), context, by_cid)
    assert flags(checks) == [] and notes(checks) == ["unit_converted"]
    assert checks.units_matched == (1, 1)
    assert "$130,497 in millions reads as $130.5 billion" in detail(checks, "unit_converted")


def test_the_uninsured_deposit_sentence_with_a_non_breaking_space_is_located(retrieved):
    chunk = chunk_from(retrieved, "q04", UNINSURED)
    assert chunk.kind == "prose"
    context, by_cid = one_chunk(chunk)
    # The source prints non-breaking spaces inside these dates; the quote
    # carries them and still has to locate.
    quote = ("At September\u00a030, 2025 and December\u00a031, 2024, Firmwide estimated uninsured "
             "deposits were $1,548.1 billion and $1,414.0 billion, respectively")
    checks = resolver.check(answer_of(claim(
        "JPMorgan's estimated uninsured deposits were $1,548.1 billion at September 30, 2025.", quote,
        kind="point_in_time")), context, by_cid)
    assert flags(checks) == [] and notes(checks) == []
    assert checks.quotes_located == (1, 1) and checks.figures_in_quote == (1, 1)
    # Prose carries no column, so the column check neither matches nor runs.
    assert checks.columns_matched == (0, 0) and checks.columns_unverified == 0
    # The scale word beside the number in the source is the source's own.
    assert checks.units_matched == (1, 1)


def test_a_curly_apostrophe_and_an_en_dash_still_match(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q04", UNINSURED))
    # Written as escapes so this file stays ASCII.
    quote = ("Refer to the Firm\u2019s Consolidated Balance Sheets Analysis and the Business Segment & "
             "Corporate Results on pages 15\u201316")
    checks = resolver.check(answer_of(claim("The filing points to its balance sheet analysis.", quote,
                                            kind="point_in_time")), context, by_cid)
    assert checks.quotes_located == (1, 1) and flags(checks) == []


def test_a_comparative_sentence_reports_the_other_column_without_flagging_it(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q06", NII))
    checks = resolver.check(answer_of(claim(
        "Net interest income was $23,966 million, up from $23,405 million a year earlier.", NII_ROW)),
        context, by_cid)
    assert flags(checks) == []
    assert checks.figures_in_quote == (2, 2) and checks.columns_matched == (1, 2)
    assert "$23,405 million sits in the column for 2024-09-30" == detail(checks, "column_other_period")


def test_a_quote_across_a_chunk_boundary_is_located(retrieved):
    here = chunk_from(retrieved, "q09", NVDA_2025)
    after = chunk_from(retrieved, "q09", NVDA_2024)
    assert after.seq == here.seq + 1 and after.file == here.file
    context = Context(chunks=[here, after], n_tokens=here.n_tokens, cids=["C1", "C2"])
    quote = here.text[-40:] + "\n" + after.text[:40]
    checks = resolver.check(answer_of(claim("NVIDIA reported segment results for both years.", quote,
                                            period_end="2025-01-26", kind="fiscal_year", ticker="NVDA")),
                            context, {"C1": here, "C2": after})
    assert checks.quotes_located == (1, 1) and flags(checks) == []


def test_the_q06_fixture_reports_one_outcome_per_claim(retrieved, registry):
    prepared = retrieved["q06"]
    cid_of = {c.chunk_id: cid for cid, c in zip(prepared.context.cids, prepared.context.chunks)}
    assert NII in cid_of
    answer = load_fixture("q06_answer.json", cid_of)
    by_cid = dict(zip(prepared.context.cids, prepared.context.chunks))
    checks = resolver.check(answer, prepared.context, by_cid, prepared.plan, set(registry["companies"]))
    by_claim = {}
    for row in checks.flags:
        by_claim.setdefault(row["where"], set()).add(row["kind"])
    assert by_claim.get("K1", set()) == set()
    assert by_claim["K2"] == {"duration_mismatch"}
    # K3 restates 23,966 as $24.0 billion, and the unit line excepts rows it
    # does not name, so no scale is established to convert through.
    assert by_claim["K3"] == {"figure_not_in_chunk"}
    assert by_claim["K4"] == {"citation_unknown", "quote_not_found"}
    assert by_claim["K5"] == {"figure_not_in_chunk"}
    assert checks.quotes_located == (4, 5)
    assert checks.figures_in_quote == (2, 4) and checks.figures_unchecked == 1
    assert checks.columns_matched == (1, 2) and checks.columns_unverified == 0
    assert checks.units_matched == (0, 0) and checks.units_unchecked == 2


def test_the_q05_fixture_reads_a_column_that_carries_no_period(tuning_index):
    chunk = tuning_index.by_id[APPLE_MDA_2024]
    assert chunk.column_source == "parsed" and chunk.columns[2].label == "FY2023"
    assert chunk.columns[2].period_end is None
    context, by_cid = one_chunk(chunk, "AAPL Apple Inc: 10-K FY2024")
    answer = load_fixture("q05_answer.json", {APPLE_MDA_2024: "C1"})
    checks = resolver.check(answer, context, by_cid)
    # Both claims read the same undated middle column, so neither the fiscal
    # 2023 reading nor the fiscal 2024 reading is matched or contradicted.
    assert flags(checks) == []
    assert checks.columns_matched == (0, 0) and checks.columns_unverified == 2
    assert detail(checks, "column_unverified") == "$383,285 million: the column carries no period"


@pytest.mark.parametrize("text, expected", [
    ("Net interest income was $23,966 million in 2025, up 2% from 23,405.", ["$23,966 million", "2%", "23,405"]),
    ("At September 30, 2025 the ratio was 14.8% across 3 segments.", ["14.8%"]),
    ("Revenue was $130.5 billion (C12, K3) for the year ended 2025-01-26.", ["$130.5 billion"]),
    ("Per Note 27 and excerpt 13, in a 52-week year income was $23,966 million.", ["$23,966 million"]),
    ("As of 9/30/2025 (30 September 2025) deposits were $2.5 trillion, the 3rd rise.", ["$2.5 trillion"]),
    ("Net sales fell (3)% in FY2023 and 2% in Q4 2024.", ["(3)%", "2%"]),
])
def test_dates_counts_and_ids_are_not_claim_figures(text, expected):
    assert [f.text for f in resolver.figures_in(text)] == expected


def test_a_column_date_matches_a_month_end_the_filer_closed_early():
    column = Column(0, "Three Months Ended Oct 26, 2025", "2025-10-26", "three_months")
    assert resolver.period_agrees(column, "2025-10-31") is True
    assert resolver.period_agrees(column, "2025-10-26") is True
    # A claim date that is not a month end has to be the column's own date.
    assert resolver.period_agrees(column, "2025-10-30") is False
    assert resolver.period_agrees(Column(0, "x", None, None), "2025-10-31") is False


# ---------------------------------------------------------------------------
# One test per attack the reviewer landed, named for the defect.
# ---------------------------------------------------------------------------


def test_a_figure_only_in_the_models_quote_string_is_not_counted(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q06", NII))
    invented = "Net interest income | 99,999 | 23,405 | 2 | 70,448 | 69,233 | 2"
    checks = resolver.check(answer_of(claim("Net interest income was $99,999 million.", invented)),
                            context, by_cid)
    assert flags(checks) == ["quote_not_found"] and notes(checks) == ["figures_unchecked"]
    assert checks.figures_in_quote == (0, 0) and checks.figures_unchecked == 1
    assert checks.columns_matched == (0, 0)


def test_a_figure_the_source_prints_only_outside_the_quote_is_not_a_quote_match(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q06", NII))
    checks = resolver.check(answer_of(claim("Net income was $14,393 million for the quarter.", NII_ROW)),
                            context, by_cid)
    assert flags(checks) == ["figure_elsewhere_in_chunk"]
    assert checks.figures_in_quote == (0, 1)
    # Nothing downstream of the failed figure check is credited.
    assert checks.columns_matched == (0, 0) and checks.columns_unverified == 0


def test_a_dollar_claim_against_a_percent_cell_is_not_a_figure_or_a_column_match(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q04", CAPITAL))
    checks = resolver.check(answer_of(claim("CET1 capital was $14.8 million.", CET1_ROW,
                                            kind="point_in_time")), context, by_cid)
    assert flags(checks) == ["figure_not_in_chunk"]
    assert checks.figures_in_quote == (0, 1) and checks.columns_matched == (0, 0)


def test_a_year_a_day_a_cik_or_a_footnote_marker_is_not_a_figure_match(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q06", NII))
    header_row = ("Three months ended September 30, | Nine months ended September 30,\n"
                  "2025 | 2024 | Change | 2025 | 2024 | Change")
    for text, quote in [
        ("Net interest income was $2,025 million.", header_row),
        ("Net interest income was $2,024 million.", NII_ROW),
        ("Net interest income was $19,617 million.", "JPMorgan Chase & Co (JPM, CIK 19617)"),
        # "2" is printed in the change cells; "$2.0 million" is not those digits.
        ("Net interest income rose $2.0 million.", NII_ROW),
    ]:
        checks = resolver.check(answer_of(claim(text, quote)), context, by_cid)
        assert flags(checks) == ["figure_not_in_chunk"], text
        assert checks.figures_in_quote == (0, 1), text
    context, by_cid = one_chunk(chunk_from(retrieved, "q09", NVDA_2025))
    # "(1)" is the footnote marker on the row label, not a figure of the row.
    checks = resolver.check(answer_of(claim("Other segment items were $1 million.",
                                            "Other segment items (1) | 33,318 | 9,219",
                                            period_end="2025-01-26", kind="fiscal_year", ticker="NVDA")),
                            context, by_cid)
    assert flags(checks) == ["figure_not_in_chunk"]


def test_a_quote_lying_wholly_in_an_uncited_neighbour_is_not_located(retrieved):
    here = chunk_from(retrieved, "q09", NVDA_2025)
    after = chunk_from(retrieved, "q09", NVDA_2024)
    context = Context(chunks=[here, after], n_tokens=here.n_tokens, cids=["C1", "C2"])
    quote = "Year Ended Jan 28, 2024\n" + NVDA_2024_ROW
    assert NVDA_2024_ROW in after.text and NVDA_2024_ROW not in here.text
    # The claim cites C1 alone, so text that lies only in C2 locates nothing.
    checks = resolver.check(answer_of(claim("Compute & Networking revenue was $47,405 million.", quote,
                                            period_end="2024-01-28", kind="fiscal_year", ticker="NVDA")),
                            context, {"C1": here, "C2": after})
    assert flags(checks) == ["quote_not_found"]
    assert checks.quotes_located == (0, 1) and checks.figures_unchecked == 1


def test_a_per_share_row_under_an_except_unit_line_leaves_units_unchecked(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q07", APPLE_OPS))
    checks = resolver.check(answer_of(claim("Apple used 15,408,095 thousand diluted shares.", DILUTED_ROW,
                                            period_end="2024-09-28", kind="fiscal_year", ticker="AAPL")),
                            context, by_cid)
    # The figure and the column are established; the scale word is not, and
    # the figure match alone never reads as a match on the units.
    assert flags(checks) == [] and notes(checks) == ["units_unchecked"]
    assert checks.figures_in_quote == (1, 1) and checks.columns_matched == (1, 1)
    assert checks.units_matched == (0, 0) and checks.units_unchecked == 1
    assert "excepts rows it does not name" in detail(checks, "units_unchecked")
    # A money row of the same table is no better established.
    checks = resolver.check(answer_of(claim("Total net sales were $391,035 million.", APPLE_SALES_ROW,
                                            period_end="2024-09-28", kind="fiscal_year", ticker="AAPL")),
                            context, by_cid)
    assert flags(checks) == [] and checks.units_unchecked == 1


def test_a_bare_ratio_row_is_never_credited_as_millions(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q06", NII))
    checks = resolver.check(answer_of(claim(
        "JPMorgan's return on tangible common equity was $20 million.", ROTCE_ROW)), context, by_cid)
    assert flags(checks) == [] and notes(checks) == ["column_unverified", "units_unchecked"]
    assert checks.columns_matched == (0, 0) and checks.columns_unverified == 1
    assert checks.units_matched == (0, 0) and checks.units_unchecked == 1
    assert "has 4 cells for 6 columns" in detail(checks, "column_unverified")


def test_swapped_comparative_figures_are_a_column_mismatch(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q06", NII))
    checks = resolver.check(answer_of(claim(
        "Net interest income was $23,405 million, up from $23,966 million a year earlier.", NII_ROW)),
        context, by_cid)
    assert flags(checks) == ["column_mismatch"]
    assert "$23,405 million sits in the column for 2024-09-30" in detail(checks, "column_mismatch")
    # The second figure does sit in the claim's own column; the claim is still
    # wrong, and the flag on the figure that anchors the period says so.
    assert checks.columns_matched == (1, 2)


def test_an_inherited_unit_line_leaves_units_unchecked(retrieved):
    chunk = chunk_from(retrieved, "q07", APPLE_EPS)
    assert chunk.units == "USD millions" and chunk.units_source == "inherited"
    context, by_cid = one_chunk(chunk)
    checks = resolver.check(answer_of(claim("Apple's net income was $93,736 million in fiscal 2024.",
                                            "Net income | $93,736 | $96,995 | $99,803",
                                            period_end="2024-09-28", kind="fiscal_year", ticker="AAPL")),
                            context, by_cid)
    assert flags(checks) == [] and checks.columns_matched == (1, 1)
    assert checks.units_matched == (0, 0) and checks.units_unchecked == 1
    assert "carried in from an earlier excerpt" in detail(checks, "units_unchecked")


def test_a_column_naming_another_entity_or_basis_is_unverified(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q04", CAPITAL))
    for text, missing in [
        ("The Standardized CET1 capital ratio was 14.9%.", "advanced"),
        ("JPMorgan Chase & Co.'s Standardized CET1 capital ratio was 15.7%.", "bank"),
    ]:
        checks = resolver.check(answer_of(claim(text, CET1_ROW, kind="point_in_time")), context, by_cid)
        assert flags(checks) == [], text
        assert checks.columns_matched == (0, 0) and checks.columns_unverified == 1, text
        assert missing in detail(checks, "column_unverified"), text
    # The same figure under the label the claim does name is matched.
    checks = resolver.check(answer_of(claim("JPMorgan Chase & Co.'s Standardized CET1 capital ratio was 14.8%.",
                                            CET1_ROW, kind="point_in_time")), context, by_cid)
    assert flags(checks) == [] and checks.columns_matched == (1, 1)


def test_a_segment_column_claimed_as_another_segment_is_unverified(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q09", NVDA_2025))
    for text in ["NVIDIA's Graphics revenue was $116,193 million in fiscal 2025.",
                 "NVIDIA's revenue was $130,497 million in fiscal 2025."]:
        checks = resolver.check(answer_of(claim(text, NVDA_ROW, period_end="2025-01-26",
                                                kind="fiscal_year", ticker="NVDA")), context, by_cid)
        assert flags(checks) == [], text
        assert checks.columns_matched == (0, 0) and checks.columns_unverified == 1, text
    # Two segments named in one sentence are each read in their own column.
    checks = resolver.check(answer_of(claim(
        "Compute & Networking revenue was $116,193 million and Graphics revenue was $14,304 million.",
        NVDA_ROW, period_end="2025-01-26", kind="fiscal_year", ticker="NVDA")), context, by_cid)
    assert flags(checks) == [] and checks.columns_matched == (2, 2)


def test_a_claim_that_does_not_name_the_row_is_unverified(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q06", NII))
    checks = resolver.check(answer_of(claim("Net income was $23,966 million for the quarter.", NII_ROW)),
                            context, by_cid)
    assert flags(checks) == []
    assert checks.columns_matched == (0, 0) and checks.columns_unverified == 1
    assert "does not name the row 'Net interest income'" in detail(checks, "column_unverified")


def test_a_figure_filling_two_cells_of_its_row_is_unverified(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q06", NII))
    row = "Book value per share | $124.96 | $115.15 | 9 | $124.96 | $115.15 | 9"
    checks = resolver.check(answer_of(claim("Book value per share was $124.96.", row)), context, by_cid)
    assert flags(checks) == [] and checks.columns_unverified == 1
    assert "$124.96 fills 2 cells" in detail(checks, "column_unverified")


def test_a_table_whose_column_shape_was_not_established_is_unverified(retrieved):
    chunk = chunk_from(retrieved, "q06", NII)
    for source in ("unverified_shape", None):
        untrusted = dataclasses.replace(chunk, column_source=source)
        context, by_cid = one_chunk(untrusted)
        checks = resolver.check(answer_of(claim("Net interest income was $23,966 million.", NII_ROW)),
                                context, by_cid)
        assert flags(checks) == [], source
        assert checks.figures_in_quote == (1, 1) and checks.columns_matched == (0, 0), source
        assert checks.columns_unverified == 1, source
        assert "column shape was not established" in detail(checks, "column_unverified")


def test_a_fiscal_year_label_column_with_no_period_is_never_a_calendar_year_match(retrieved):
    chunk = chunk_from(retrieved, "q05", APPLE_2023)
    assert chunk.columns[2].label == "FY2022" and chunk.columns[2].period_end is None
    context, by_cid = one_chunk(chunk)
    checks = resolver.check(answer_of(claim("Total net sales were $394,328 million in fiscal 2022.",
                                            APPLE_2023_ROW, period_end="2022-09-24", kind="fiscal_year",
                                            ticker="AAPL")), context, by_cid)
    assert flags(checks) == [] and checks.columns_unverified == 1
    assert detail(checks, "column_unverified").endswith("the column carries no period")
    # The dated column of the same table matches.
    checks = resolver.check(answer_of(claim("Total net sales were $383,285 million in fiscal 2023.",
                                            APPLE_2023_ROW, period_end="2023-09-30", kind="fiscal_year",
                                            ticker="AAPL")), context, by_cid)
    assert flags(checks) == [] and checks.columns_matched == (1, 1)


def test_the_apple_fy2023_claim_matches_and_the_fy2024_period_does_not(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q07", APPLE_OPS))
    checks = resolver.check(answer_of(claim("Apple's total net sales were $383,285 million in fiscal 2023.",
                                            APPLE_SALES_ROW, period_end="2023-09-30", kind="fiscal_year",
                                            ticker="AAPL")), context, by_cid)
    assert flags(checks) == [] and checks.columns_matched == (1, 1)
    checks = resolver.check(answer_of(claim("Apple's total net sales were $383,285 million in fiscal 2024.",
                                            APPLE_SALES_ROW, period_end="2024-09-28", kind="fiscal_year",
                                            ticker="AAPL")), context, by_cid)
    assert flags(checks) == ["column_mismatch"] and checks.columns_matched == (0, 1)
    assert "the claim says 2024-09-28" in detail(checks, "column_mismatch")


def test_a_nine_month_figure_stated_as_a_quarter_is_a_duration_mismatch(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q06", NII))
    checks = resolver.check(answer_of(claim("Net interest income was $70,448 million for the quarter.",
                                            NII_ROW)), context, by_cid)
    assert flags(checks) == ["duration_mismatch"] and checks.columns_matched == (0, 1)


def test_a_claim_that_cites_nothing_is_flagged_and_checks_nothing(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q06", NII))
    data = claim("Net interest income was $23,966 million.", NII_ROW)
    data["citations"] = []
    checks = resolver.check(answer_of(data), context, by_cid)
    assert flags(checks) == ["no_citation", "quote_not_found"]
    assert checks.quotes_located == (0, 1) and checks.figures_in_quote == (0, 0)
    assert checks.figures_unchecked == 1


def test_a_ticker_outside_the_question_scope_is_flagged(retrieved):
    prepared = retrieved["q06"]
    by_cid = dict(zip(prepared.context.cids, prepared.context.chunks))
    cid = next(c for c, chunk in by_cid.items() if chunk.chunk_id == NII)
    answer = answer_of(claim("Net interest income was $23,966 million.", NII_ROW, ticker="BAC", cid=cid))
    checks = resolver.check(answer, prepared.context, by_cid, prepared.plan)
    assert flags(checks) == ["ticker_out_of_scope"]


def test_a_summary_sentence_or_cell_stating_an_unlinked_figure_is_flagged(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q06", NII))
    answer = answer_of(
        claim("Net interest income was $23,966 million.", NII_ROW),
        summary=[{"text": "Net interest income was $23,966 million.", "claim_ids": ["K1"]},
                 {"text": "Net interest income was $99,999 million.", "claim_ids": ["K1"]},
                 {"text": "orphan", "claim_ids": []},
                 {"text": "phantom", "claim_ids": ["K9"]}],
        table=[{"dimension": "NII", "cells": [{"column": "JPM", "text": "$30,000", "claim_ids": ["K1"]},
                                              {"column": "JPM Q3", "text": "23,966", "claim_ids": ["K1"]}]}])
    checks = resolver.check(answer, context, by_cid)
    assert [f["kind"] for f in checks.flags if f["where"] != "K1"] == [
        "unlinked_figure", "unlinked_sentence", "unlinked_sentence", "unlinked_figure"]
    assert checks.unlinked == ["orphan", "phantom"]


def test_a_gap_or_a_not_comparable_note_outside_the_coverage_block_is_flagged(retrieved, registry):
    coverage = "JPM JPMorgan Chase & Co: 10-Q FY2025 Q3 (quarter ended 2025-09-30, filed 2025-11-04)"
    context, by_cid = one_chunk(chunk_from(retrieved, "q06", NII), coverage)
    answer = answer_of(
        claim("Net interest income was $23,966 million.", NII_ROW),
        gaps=["The excerpts hold no 2019 figures.", "No Bank of America (BAC) excerpt is present.",
              "The excerpts state no 2025 target."],
        not_comparable=[{"dimension": "anything", "tickers": ["JPM", "WFC", "ZZZZ"], "reason": "no reason"}])
    checks = resolver.check(answer, context, by_cid, None, set(registry["companies"]))
    grounded = [(f["where"], f["detail"]) for f in checks.flags if f["kind"] == "ungrounded_gap"]
    assert [g[0] for g in grounded] == ["gap 1", "gap 2", "not comparable 1"]
    assert "2019" in grounded[0][1] and "BAC" in grounded[1][1]
    assert "WFC, ZZZZ" in grounded[2][1]


def test_an_approximate_quote_is_flagged_with_the_passage_it_found(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q04", BAC_PROSE))
    doctored = ("At December 31, 2023, the Corporation's deposits totaled $1.92 trillion, of which total "
                "estimated uninsured U.S. and non-U.S. deposits were $616.8 billion and $116.6 billion.")
    checks = resolver.check(answer_of(claim("Uninsured U.S. deposits were $616.8 billion.", doctored,
                                            period_end="2023-12-31", kind="point_in_time", ticker="BAC")),
                            context, by_cid)
    assert flags(checks) == ["approximate_quote", "figure_not_in_chunk"]
    passage = next(f["source_string"] for f in checks.flags if f["kind"] == "approximate_quote")
    assert "606.8" in passage and checks.figures_in_quote == (0, 1)


def test_a_source_number_printed_negative_is_flagged_against_a_positive_claim(retrieved):
    context, by_cid = one_chunk(chunk_from(retrieved, "q05", APPLE_2023))
    checks = resolver.check(answer_of(claim("Total net sales grew 3% in fiscal 2023.", APPLE_2023_ROW,
                                            period_end="2023-09-30", kind="fiscal_year", ticker="AAPL")),
                            context, by_cid)
    assert "sign_differs" in flags(checks)
    assert "(3)%" in detail(checks, "sign_differs")
