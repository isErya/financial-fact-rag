"""The evidence checks over real chunks of the tuning-set index.

The q06 fixture cites the JPMorgan Q3 2025 summary income statement by
chunk id; the test binds that id to whatever C-id retrieval gave it, so
the fixture stays hand-written while the numbering stays retrieval's. The
q05 fixture is checked over a one-chunk context built by hand from the
FY2024 10-K, because the q05 question itself retrieves the FY2023 10-K.
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
Q06 = "What was JPMorgan's net interest income for the third quarter of 2025?"
NII_CHUNK = "55e0c02f62d5d7dc"
APPLE_FY2024_NET_SALES_CHUNK = "cffc6bfbcc7ea05f"
UNINSURED_CHUNK = "cf75f0872077d7af"
NII_ROW = "Net interest income | 23,966 | 23,405 | 2 | 70,448 | 69,233 | 2"


def load_fixture(name: str, cid_of: dict[str, str]) -> Answer:
    with open(os.path.join(FIXTURES, name)) as fh:
        data = json.load(fh)
    for claim in data["claims"]:
        claim["citations"] = [cid_of[c[6:]] if c.startswith("chunk:") else c for c in claim["citations"]]
    return Answer.model_validate(data)


def flags_by_claim(checks) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for f in checks.flags:
        out.setdefault(f["claim_id"], set()).add(f["kind"])
    return out


def one_chunk_context(chunk: Chunk, coverage: str = "") -> tuple[Context, dict]:
    context = Context(chunks=[chunk], n_tokens=chunk.n_tokens, cids=["C1"], coverage=coverage)
    return context, {"C1": chunk}


def claim(cid: str, text: str, quote: str, period_end: str = "2025-09-30", kind: str = "quarter",
          ticker: str = "JPM", cid_id: str = "K1") -> dict:
    return {"id": cid_id, "text": text, "tickers": [ticker], "period_end": period_end,
            "period_kind": kind, "citations": [cid], "quote": quote}


def answer_of(*claims: dict, gaps: list[str] | None = None, summary: list | None = None) -> Answer:
    return Answer.model_validate({
        "summary": summary if summary is not None else [{"text": "s", "claim_ids": [c["id"] for c in claims]}],
        "claims": list(claims), "table": [], "not_comparable": [], "gaps": gaps or []})


def test_q06_fixture_claims_yield_their_flags(tuning_index, registry):
    prepared = prepare(Q06, tuning_index, registry)
    cid_of = {c.chunk_id: cid for cid, c in zip(prepared.context.cids, prepared.context.chunks)}
    assert NII_CHUNK in cid_of
    answer = load_fixture("q06_answer.json", cid_of)
    checks = resolver.check(answer, prepared.context, dict(zip(prepared.context.cids, prepared.context.chunks)),
                            prepared.plan, set(registry["companies"]))
    by_claim = flags_by_claim(checks)
    assert by_claim.get("K1", set()) == set()
    assert by_claim["K2"] == {"duration_mismatch"}
    assert by_claim["K3"] == {"unit_converted"}
    assert by_claim["K4"] == {"citation_unknown"}
    assert by_claim["K5"] == {"figure_not_in_chunk"}
    # K4 has no excerpt to check against, so its quote counts as not found
    # and its figure as not in the quote; K5's figure is nowhere.
    assert checks.quotes_found == (4, 5)
    assert checks.figures_in_quote == (3, 5)
    # K1 and K3 land in the three-month column; K2 reached a parsed column
    # and mismatched; K4 and K5 never reached a column.
    assert checks.columns_matched == (2, 3)
    assert checks.columns_unverified == 0
    assert checks.units_declared == (4, 4)
    converted = next(f for f in checks.flags if f["kind"] == "unit_converted")
    assert converted["source_string"] == "23,966" and "million to billion" in converted["detail"]


def test_q05_fixture_middle_column_matches_fy2023_only(tuning_index):
    chunk = tuning_index.by_id[APPLE_FY2024_NET_SALES_CHUNK]
    assert chunk.column_source == "parsed"
    context, by_cid = one_chunk_context(chunk, "AAPL Apple Inc: 10-K FY2024")
    answer = load_fixture("q05_answer.json", {APPLE_FY2024_NET_SALES_CHUNK: "C1"})
    checks = resolver.check(answer, context, by_cid)
    by_claim = flags_by_claim(checks)
    assert by_claim.get("K1", set()) == set()
    assert by_claim["K2"] == {"column_mismatch"}
    assert checks.columns_matched == (1, 2)
    mismatch = next(f for f in checks.flags if f["kind"] == "column_mismatch")
    assert mismatch["source_string"] == "FY2023"


def test_uninsured_deposit_sentence_with_nbsp_is_found_in_source(tuning_index):
    chunk = tuning_index.by_id[UNINSURED_CHUNK]
    assert chunk.kind == "prose"
    context, by_cid = one_chunk_context(chunk)
    quote = ("At September\u00a030, 2025 and December\u00a031, 2024, Firmwide estimated uninsured deposits "
             "were $1,548.1 billion and $1,414.0 billion, respectively")
    answer = answer_of(claim("C1", "JPMorgan's estimated uninsured deposits were $1,548.1 billion at "
                             "September 30, 2025.", quote, kind="point_in_time"))
    checks = resolver.check(answer, context, by_cid)
    assert checks.quotes_found == (1, 1)
    assert checks.figures_in_quote == (1, 1)
    # A figure found in prose gets no column check and no column flag.
    assert checks.columns_matched == (0, 0) and checks.columns_unverified == 0
    assert checks.flags == []


def test_curly_quote_and_dash_variant_still_matches(tuning_index):
    chunk = tuning_index.by_id[UNINSURED_CHUNK]
    context, by_cid = one_chunk_context(chunk)
    # A curly apostrophe and an en dash, written as escapes so the file
    # itself stays ASCII.
    quote = ("Refer to the Firm\u2019s Consolidated Balance Sheets Analysis and the Business Segment & "
             "Corporate Results on pages 15\u201316")
    answer = answer_of(claim("C1", "The filing points to its balance sheet analysis.", quote, kind="point_in_time"))
    checks = resolver.check(answer, context, by_cid)
    assert checks.quotes_found == (1, 1)
    assert checks.flags == []


def forty_row_chunk() -> Chunk:
    """A synthetic 40-row income statement with JPMorgan's column layout;
    the net interest income row sits deep in the table."""
    columns = [
        Column(0, "Three months ended September 30, 2025", "2025-09-30", "three_months"),
        Column(1, "Three months ended September 30, 2024", "2024-09-30", "three_months"),
        Column(2, "Change", None, None),
        Column(3, "Nine months ended September 30, 2025", "2025-09-30", "nine_months"),
        Column(4, "Nine months ended September 30, 2024", "2024-09-30", "nine_months"),
        Column(5, "Change", None, None),
    ]
    lines = ["(in millions) | Three months ended September 30, | Nine months ended September 30,",
             "2025 | 2024 | Change | 2025 | 2024 | Change"]
    for n in range(40):
        if n == 36:
            lines.append(NII_ROW)
            continue
        base = 1000 + n * 7
        lines.append("Line item %d | %d | %d | 3 | %d | %d | 4" % (n + 1, base, base + 1, base + 2, base + 3))
    text = "\n".join(lines)
    return Chunk(chunk_id="synthetic40", file="JPM_10Q_2025Q3_2025-11-04_full.txt", cik="0000019617",
                 ticker="JPM", company="JPMorgan Chase & Co", form="10-Q", part="I", item="I.2",
                 item_title="Management's Discussion and Analysis", note_title=None, kind="table",
                 period_end="2025-09-30", fiscal_year=2025, fiscal_quarter=3, fiscal_label="FY2025 Q3",
                 filing_date="2025-11-04", seq=999, char_start=0, char_end=len(text), units="USD millions",
                 units_source="declared", columns=columns, column_source="parsed",
                 header="JPMorgan Chase & Co (JPM, CIK 19617) | 10-Q FY2025 Q3", text=text, n_tokens=900)


def test_forty_row_table_finds_the_three_and_nine_month_columns():
    chunk = forty_row_chunk()
    context, by_cid = one_chunk_context(chunk)
    answer = answer_of(
        claim("C1", "Net interest income was $23,966 million for the quarter.", NII_ROW, cid_id="K1"),
        claim("C1", "Net interest income was $70,448 million for the nine months.", NII_ROW, kind="nine_months",
              cid_id="K2"))
    checks = resolver.check(answer, context, by_cid)
    assert checks.quotes_found == (2, 2)
    assert checks.columns_matched == (2, 2)
    assert checks.flags == []


def test_column_unverified_for_ambiguous_rows_and_untrusted_column_sources(tuning_index):
    chunk = tuning_index.by_id[NII_CHUNK]
    row = "Book value per share | $124.96 | $115.15 | 9 | $124.96 | $115.15 | 9"
    # 124.96 fills two columns of its row, so no single column can be named.
    context, by_cid = one_chunk_context(chunk)
    checks = resolver.check(answer_of(claim("C1", "Book value per share was $124.96.", row)), context, by_cid)
    assert [f["kind"] for f in checks.flags] == ["column_unverified"]
    assert checks.columns_matched == (0, 0) and checks.columns_unverified == 1
    # The chunker's display-only labels and a missing header both give the
    # same verdict: the figure is real, its column is not read.
    for source in ("unverified_shape", None):
        untrusted = dataclasses.replace(chunk, column_source=source)
        context, by_cid = one_chunk_context(untrusted)
        checks = resolver.check(answer_of(claim("C1", "Net interest income was $23,966 million.", NII_ROW)),
                                context, by_cid)
        assert [f["kind"] for f in checks.flags] == ["column_unverified"], source
        assert checks.quotes_found == (1, 1) and checks.figures_in_quote == (1, 1)
        assert checks.columns_matched == (0, 0)


def test_figure_absent_from_quote_but_present_in_chunk(tuning_index):
    chunk = tuning_index.by_id[NII_CHUNK]
    context, by_cid = one_chunk_context(chunk)
    answer = answer_of(claim("C1", "Net income was $14,393 million for the quarter.", NII_ROW))
    checks = resolver.check(answer, context, by_cid)
    kinds = [f["kind"] for f in checks.flags]
    assert kinds == ["figure_not_in_quote"]
    assert checks.figures_in_quote == (0, 1)
    # The figure was located in the chunk, so its column is still read.
    assert checks.columns_matched == (1, 1)


def test_approximate_quote_reports_the_closest_passage(tuning_index):
    chunk = tuning_index.by_id[UNINSURED_CHUNK]
    context, by_cid = one_chunk_context(chunk)
    quote = ("Firmwide estimated uninsured deposits were $1,548.1 billion and $1,414.0 billion, respectively, "
             "primarily reflecting wholesale operating deposits at the Firm")
    answer = answer_of(claim("C1", "Uninsured deposits were $1,548.1 billion.", quote, kind="point_in_time"))
    checks = resolver.check(answer, context, by_cid)
    assert checks.quotes_found == (0, 1)
    flag = next(f for f in checks.flags if f["kind"] == "approximate_quote")
    assert "wholesale operating deposits" in flag["source_string"]


def test_quote_not_found_and_too_long(tuning_index):
    chunk = tuning_index.by_id[NII_CHUNK]
    context, by_cid = one_chunk_context(chunk)
    answer = answer_of(claim("C1", "Net interest income rose.", "word " * 45))
    kinds = {f["kind"] for f in resolver.check(answer, context, by_cid).flags}
    assert kinds == {"quote_too_long", "quote_not_found"}


def test_units_mismatch_when_the_claim_states_the_wrong_scale(tuning_index):
    chunk = tuning_index.by_id[NII_CHUNK]
    context, by_cid = one_chunk_context(chunk)
    answer = answer_of(claim("C1", "Net interest income was $23,966 billion.", NII_ROW))
    kinds = [f["kind"] for f in resolver.check(answer, context, by_cid).flags]
    assert kinds == ["units_mismatch"]


def test_unlinked_sentences_and_ungrounded_gaps(tuning_index, registry):
    chunk = tuning_index.by_id[NII_CHUNK]
    coverage = "JPM JPMorgan Chase & Co: 10-Q FY2025 Q3 (quarter ended 2025-09-30, filed 2025-11-04)"
    context, by_cid = one_chunk_context(chunk, coverage)
    answer = answer_of(
        claim("C1", "Net interest income was $23,966 million.", NII_ROW),
        summary=[{"text": "backed", "claim_ids": ["K1"]}, {"text": "orphan", "claim_ids": []},
                 {"text": "phantom", "claim_ids": ["K9"]}],
        gaps=["The excerpts hold no 2019 figures.", "No Bank of America (BAC) excerpt is present.",
              "The excerpts do not state a target."])
    checks = resolver.check(answer, context, by_cid, None, set(registry["companies"]))
    assert checks.unlinked_sentences == ["orphan", "phantom"]
    ungrounded = [f["source_string"] for f in checks.flags if f["kind"] == "ungrounded_gap"]
    assert ungrounded == ["The excerpts hold no 2019 figures.", "No Bank of America (BAC) excerpt is present."]


def test_ticker_outside_the_plan_is_flagged(tuning_index, registry):
    prepared = prepare(Q06, tuning_index, registry)
    by_cid = dict(zip(prepared.context.cids, prepared.context.chunks))
    cid = next(c for c, chunk in by_cid.items() if chunk.chunk_id == NII_CHUNK)
    answer = answer_of(claim(cid, "Net interest income was $23,966 million.", NII_ROW, ticker="BAC"))
    checks = resolver.check(answer, prepared.context, by_cid, prepared.plan)
    assert [f["kind"] for f in checks.flags] == ["ticker_out_of_scope"]


@pytest.mark.parametrize("text, expected", [
    ("Net interest income was $23,966 million in 2025, up 2% from 23,405.", ["$23,966 million", "2%", "23,405"]),
    ("At September 30, 2025 the ratio was 14.8% across 3 segments.", ["14.8%"]),
    ("Revenue was $130.5 billion (C12, K3) for the year ended 2025-01-26.", ["$130.5 billion"]),
])
def test_figures_in_claim_text(text, expected):
    assert [f.text for f in resolver.figures_in(text)] == expected


# ---------------------------------------------------------------------------
# Regression cases from the milestone 4 audit: every false green and false
# red the breaker found, plus the right outcomes that must stay right.
# ---------------------------------------------------------------------------

CAPITAL_CHUNK = "1ae8975b3361a7c4"      # JPM Note 20 capital table: Standardized/Advanced x Co./Bank
DEPOSITS_CHUNK = "5d193f80d41dec3f"     # JPM deposits table, USD billions
BAC_PROSE_CHUNK = "f1593455cf409bc3"    # BAC 10-K prose with "$606.8 billion and $116.6 billion"
APPLE_FY2023_SEGMENTS_CHUNK = "3204ddb0ccb5f6f2"   # FY2023 10-K: FY2022 and FY2021 are label-only columns
APPLE_FY2024_OPERATIONS_CHUNK = "328dc22e456d59fb"  # FY2024 10-K statement of operations, shares in thousands
CET1_ROW = "CET1 capital ratio | 14.8% | 15.7% | 14.9% | 16.8%"
DILUTED_ROW = "Diluted | 15,408,095 | 15,812,547 | 16,325,819"


def kinds(checks) -> list[str]:
    return [f["kind"] for f in checks.flags]


def segment_chunk(seq: int, period_end: str, rows: str) -> Chunk:
    """A segment table in NVIDIA's Note 13 layout: three columns for one
    period, labelled by segment."""
    columns = [Column(0, "Compute & Networking Three Months Ended %s" % period_end, period_end, "three_months"),
               Column(1, "Graphics Three Months Ended %s" % period_end, period_end, "three_months"),
               Column(2, "Total Three Months Ended %s" % period_end, period_end, "three_months")]
    text = "Compute & Networking | Graphics | Total\n(In millions)\nThree Months Ended %s\n%s" % (period_end, rows)
    return Chunk(chunk_id="segments-%d" % seq, file="NVDA_10Q_2026Q3_2025-11-19_full.txt", cik="0001045810",
                 ticker="NVDA", company="NVIDIA Corporation", form="10-Q", part="I", item="I.1",
                 item_title="Financial Statements", note_title="Note 13 - Segment Information", kind="table",
                 period_end="2025-10-26", fiscal_year=2026, fiscal_quarter=3, fiscal_label="FY2026 Q3",
                 filing_date="2025-11-19", seq=seq, char_start=0, char_end=len(text), units="USD millions",
                 units_source="declared", columns=columns, column_source="parsed",
                 header="NVIDIA Corporation (NVDA, CIK 1045810) | 10-Q FY2026 Q3", text=text, n_tokens=80)


SEGMENTS_2025 = segment_chunk(66, "2025-10-26", "Revenue | $50,908 | $6,098 | $57,006\n"
                                                "Other segment items (1) | 15,187 | 3,552 | 18,739\n"
                                                "Operating income (loss) | $35,721 | $2,546 | $38,267")
SEGMENTS_2024 = segment_chunk(67, "2024-10-27", "Revenue | $31,036 | $4,046 | $35,082\n"
                                                "Other segment items (1) | 8,955 | 2,544 | 11,499\n"
                                                "Operating income (loss) | $22,081 | $1,502 | $23,583")
REVENUE_ROW = "Revenue | $50,908 | $6,098 | $57,006"


def nvda_claim(text: str, quote: str, cid: str = "C1", period_end: str = "2025-10-26", cid_id: str = "K1") -> dict:
    return claim(cid, text, quote, period_end=period_end, ticker="NVDA", cid_id=cid_id)


def test_figures_are_credited_only_against_located_text(tuning_index):
    # An invented row carrying an invented number: the quote is nowhere,
    # so the figure is checked against the chunk, where it is not.
    context, by_cid = one_chunk_context(tuning_index.by_id[NII_CHUNK])
    invented = "Net interest income | 99,999 | 23,405 | 2 | 70,448 | 69,233 | 2"
    checks = resolver.check(answer_of(claim("C1", "Net interest income was $99,999 million.", invented)), context, by_cid)
    assert kinds(checks) == ["quote_not_found", "figure_not_in_chunk"]
    assert checks.figures_in_quote == (0, 1) and checks.columns_matched == (0, 0)
    # A real sentence with one digit changed: the closest passage is the
    # chunk's own text, which holds 606.8, so 616.8 is nowhere.
    context, by_cid = one_chunk_context(tuning_index.by_id[BAC_PROSE_CHUNK])
    doctored = ("At December 31, 2023, the Corporation's deposits totaled $1.92 trillion, of which total estimated "
                "uninsured U.S. and non-U.S. deposits were $616.8 billion and $116.6 billion.")
    checks = resolver.check(answer_of(claim("C1", "Uninsured U.S. deposits were $616.8 billion.", doctored,
                                            period_end="2023-12-31", kind="point_in_time", ticker="BAC")),
                            context, by_cid)
    assert kinds(checks) == ["approximate_quote", "figure_not_in_chunk"]
    assert checks.figures_in_quote == (0, 1)


def test_source_number_types_are_compared(tuning_index):
    # A ratio cell is never a dollar amount.
    context, by_cid = one_chunk_context(tuning_index.by_id[CAPITAL_CHUNK])
    checks = resolver.check(answer_of(claim("C1", "CET1 capital was $14.8 million.", CET1_ROW, kind="point_in_time")),
                            context, by_cid)
    assert kinds(checks) == ["figure_not_in_chunk"]
    assert checks.figures_in_quote == (0, 1) and checks.columns_matched == (0, 0)
    # A bare cell in a money column is never a percentage.
    context, by_cid = one_chunk_context(tuning_index.by_id[NII_CHUNK])
    checks = resolver.check(answer_of(claim("C1", "Net interest income was 23,966%.", NII_ROW)), context, by_cid)
    assert kinds(checks) == ["figure_type_mismatch"]
    assert checks.columns_matched == (0, 1)
    # A change column holds percentages, so "$2.0 million" cannot come from
    # its "2".
    checks = resolver.check(answer_of(claim("C1", "Net interest income rose $2.0 million.", NII_ROW)), context, by_cid)
    assert kinds(checks) == ["figure_not_in_chunk"]
    # A percent claim against the change column stays what it was: found,
    # and unverified because 2 fills two columns.
    checks = resolver.check(answer_of(claim("C1", "Net interest income grew 2% year over year.", NII_ROW)),
                            context, by_cid)
    assert kinds(checks) == ["column_unverified"] and checks.figures_in_quote == (1, 1)


def test_years_days_and_ids_in_the_source_are_never_figures(tuning_index):
    context, by_cid = one_chunk_context(tuning_index.by_id[NII_CHUNK])
    header_row = ("Three months ended September 30, | Nine months ended September 30,\n"
                  "2025 | 2024 | Change | 2025 | 2024 | Change")
    for text, quote in [
        ("Net interest income was $2,025 million.", header_row),
        ("Net interest income was $19,617 million.", "JPMorgan Chase & Co (JPM, CIK 19617)"),
        ("Net interest income was $2,024 million.", NII_ROW),
        ("Net interest income was $2.0 billion.", NII_ROW),
    ]:
        checks = resolver.check(answer_of(claim("C1", text, quote)), context, by_cid)
        assert kinds(checks) == ["figure_not_in_chunk"], text
        assert checks.figures_in_quote == (0, 1), text
    context, by_cid = one_chunk_context(tuning_index.by_id[DEPOSITS_CHUNK])
    checks = resolver.check(answer_of(claim("C1", "Deposits rose $30 million.",
                                            "as of September 30, 2025 and December 31, 2024", kind="point_in_time")),
                            context, by_cid)
    assert kinds(checks) == ["figure_not_in_chunk"]


def test_column_labels_name_the_entity_and_basis(tuning_index):
    context, by_cid = one_chunk_context(tuning_index.by_id[CAPITAL_CHUNK])
    # 15.7% is the Bank, N.A. column; 14.9% is the Advanced column.
    for text, reads_as in [
        ("JPMorgan Chase & Co.'s Standardized CET1 capital ratio was 15.7%.", "Standardized JPMorganChase & Co."),
        ("The Standardized CET1 capital ratio was 14.9%.", "Standardized JPMorganChase & Co."),
    ]:
        checks = resolver.check(answer_of(claim("C1", text, CET1_ROW, kind="point_in_time")), context, by_cid)
        assert kinds(checks) == ["column_mismatch"], text
        assert reads_as in checks.flags[0]["detail"]
        assert checks.columns_matched == (0, 1)
    # The same figures under their own labels pass.
    for text in ["JPMorgan Chase & Co.'s Standardized CET1 capital ratio was 14.8%.",
                 "The Advanced CET1 capital ratio of JPMorgan Chase Bank, N.A. was 16.8%."]:
        checks = resolver.check(answer_of(claim("C1", text, CET1_ROW, kind="point_in_time")), context, by_cid)
        assert checks.flags == [], text
        assert checks.columns_matched == (1, 1)


def test_column_labels_name_the_segment():
    context, by_cid = one_chunk_context(SEGMENTS_2025)
    for text in ["Graphics segment revenue was $50,908 million.", "Compute & Networking revenue was $57,006 million."]:
        checks = resolver.check(answer_of(nvda_claim(text, REVENUE_ROW)), context, by_cid)
        assert kinds(checks) == ["column_mismatch"], text
        assert checks.columns_matched == (0, 1)
    checks = resolver.check(answer_of(nvda_claim("Graphics revenue was $6,098 million.", REVENUE_ROW)), context, by_cid)
    assert checks.flags == [] and checks.columns_matched == (1, 1)
    # A claim that names no segment cannot be matched to one.
    checks = resolver.check(answer_of(nvda_claim("Revenue was $57,006 million.", REVENUE_ROW)), context, by_cid)
    assert kinds(checks) == ["column_unverified"] and checks.columns_unverified == 1
    # Two figures naming two segments in one sentence are read as their
    # own columns.
    checks = resolver.check(answer_of(nvda_claim(
        "Compute & Networking revenue was $50,908 million and Graphics revenue was $6,098 million.", REVENUE_ROW)),
        context, by_cid)
    assert checks.flags == [] and checks.columns_matched == (2, 2)


def test_row_label_is_read(tuning_index):
    context, by_cid = one_chunk_context(tuning_index.by_id[NII_CHUNK])
    checks = resolver.check(answer_of(claim("C1", "Net income was $23,966 million for the quarter.", NII_ROW)),
                            context, by_cid)
    assert kinds(checks) == ["row_mismatch"]
    assert "reads as the row 'Net income'" in checks.flags[0]["detail"]
    assert checks.columns_matched == (0, 1)
    # Both rows named in one sentence: each figure is read in its own row.
    checks = resolver.check(answer_of(claim(
        "C1", "Net interest income was $23,966 million and net income was $14,393 million.", NII_ROW)),
        context, by_cid)
    assert kinds(checks) == ["figure_not_in_quote"] and checks.columns_matched == (2, 2)


def test_share_counts_use_the_except_clause_of_the_unit_line(tuning_index):
    context, by_cid = one_chunk_context(tuning_index.by_id[APPLE_FY2024_OPERATIONS_CHUNK])
    checks = resolver.check(answer_of(claim("C1", "Apple used 15,408,095 thousand diluted shares.", DILUTED_ROW,
                                            period_end="2024-09-28", kind="fiscal_year", ticker="AAPL")),
                            context, by_cid)
    assert checks.flags == [] and checks.columns_matched == (1, 1)
    checks = resolver.check(answer_of(claim("C1", "Apple used 15.41 billion diluted shares.", DILUTED_ROW,
                                            period_end="2024-09-28", kind="fiscal_year", ticker="AAPL")),
                            context, by_cid)
    assert kinds(checks) == ["unit_converted"] and "thousand to billion" in checks.flags[0]["detail"]
    # Money rows of the same table stay in millions.
    checks = resolver.check(answer_of(claim("C1", "Total net sales were $391,035 billion.",
                                            "Total net sales | 391,035 | 383,285 | 394,328",
                                            period_end="2024-09-28", kind="fiscal_year", ticker="AAPL")),
                            context, by_cid)
    assert kinds(checks) == ["units_mismatch"]


def test_comparative_claims_keep_the_earlier_column(tuning_index):
    context, by_cid = one_chunk_context(tuning_index.by_id[NII_CHUNK])
    for text in [
        "Net interest income was $23,966 million, up from $23,405 million a year earlier.",
        "Net interest income was $23,966 million for the three months ended September 30, 2025 versus "
        "$23,405 million for the three months ended September 30, 2024.",
    ]:
        checks = resolver.check(answer_of(claim("C1", text, NII_ROW)), context, by_cid)
        assert checks.flags == [], text
        assert checks.figures_in_quote == (2, 2) and checks.columns_matched == (2, 2)
    # Alone, the earlier column's figure is a period mismatch; and a
    # nine-month comparative is still a duration mismatch.
    checks = resolver.check(answer_of(claim("C1", "Net interest income was $23,405 million.", NII_ROW)), context, by_cid)
    assert kinds(checks) == ["column_mismatch"]
    checks = resolver.check(answer_of(claim(
        "C1", "Net interest income was $23,966 million, up from $69,233 million a year earlier.", NII_ROW)),
        context, by_cid)
    assert kinds(checks) == ["duration_mismatch"]


def test_prose_counts_dates_and_large_scales_are_not_false_reds(tuning_index):
    context, by_cid = one_chunk_context(tuning_index.by_id[NII_CHUNK])
    for text in ["Per Note 27 and excerpt 13, in a 52-week year net interest income was $23,966 million.",
                 "As of 9/30/2025 (30 September 2025) net interest income was $23,966 million."]:
        checks = resolver.check(answer_of(claim("C1", text, NII_ROW)), context, by_cid)
        assert checks.flags == [], text
        assert checks.figures_in_quote == (1, 1) and checks.columns_matched == (1, 1)
    context, by_cid = one_chunk_context(tuning_index.by_id[DEPOSITS_CHUNK])
    checks = resolver.check(answer_of(claim("C1", "Deposits were $2.5 trillion.", "Deposits | $2,548.5 | $2,406.0",
                                            kind="point_in_time")), context, by_cid)
    assert kinds(checks) == ["unit_converted"] and checks.columns_matched == (1, 1)
    # Prose carries its own scale word, so a restatement converts against it.
    context, by_cid = one_chunk_context(tuning_index.by_id[BAC_PROSE_CHUNK])
    checks = resolver.check(answer_of(claim(
        "C1", "Uninsured deposits were $606,800 million.",
        "total estimated uninsured U.S. and non-U.S. deposits were $606.8 billion and $116.6 billion",
        period_end="2023-12-31", kind="point_in_time", ticker="BAC")), context, by_cid)
    assert kinds(checks) == ["unit_converted"] and checks.figures_in_quote == (1, 1)


def test_missing_citation_is_flagged(tuning_index):
    context, by_cid = one_chunk_context(tuning_index.by_id[NII_CHUNK])
    data = claim("C1", "Net interest income was $99,999 million.", NII_ROW)
    data["citations"] = []
    checks = resolver.check(answer_of(data), context, by_cid)
    assert kinds(checks) == ["no_citation"]
    assert checks.quotes_found == (0, 1) and checks.figures_in_quote == (0, 1)


def test_summary_and_table_figures_must_be_figures_of_their_claims(tuning_index):
    context, by_cid = one_chunk_context(tuning_index.by_id[NII_CHUNK])
    answer = Answer.model_validate({
        "summary": [{"text": "NII was $30 billion.", "claim_ids": ["K1"]},
                    {"text": "NII was $24.0 billion.", "claim_ids": ["K1"]},
                    {"text": "phantom", "claim_ids": ["K9"]}],
        "claims": [claim("C1", "Net interest income was $23,966 million.", NII_ROW)],
        "table": [{"dimension": "NII", "cells": [{"column": "JPM", "text": "$99,999", "claim_ids": ["K1"]},
                                                 {"column": "JPM Q3", "text": "23,966", "claim_ids": ["K1"]}]}],
        "not_comparable": [], "gaps": []})
    checks = resolver.check(answer, context, by_cid)
    unbacked = [f["source_string"] for f in checks.flags if f["kind"] == "unbacked_figure"]
    assert unbacked == ["NII was $30 billion.", "$99,999"]
    assert checks.unlinked_sentences == ["phantom"]


def test_not_comparable_notes_and_gap_wording_are_checked(tuning_index, registry):
    chunk = tuning_index.by_id[NII_CHUNK]
    coverage = "JPM JPMorgan Chase & Co: 10-Q FY2025 Q3 (quarter ended 2025-09-30, filed 2025-11-04)"
    context, by_cid = one_chunk_context(chunk, coverage)
    answer = Answer.model_validate({
        "summary": [{"text": "s", "claim_ids": ["K1"]}],
        "claims": [claim("C1", "Net interest income was $23,966 million.", NII_ROW)],
        "table": [],
        "not_comparable": [{"dimension": "anything", "tickers": ["JPM", "WFC", "TSLA", "ZZZZ"],
                            "reason": "no reason, 2019 data"}],
        "gaps": ["No FY2019 data.", "Tesla is not covered.", "The excerpts do not state a target."]})
    names = {t: e["name"] for t, e in registry["companies"].items()}
    checks = resolver.check(answer, context, by_cid, None, set(registry["companies"]), names)
    gaps = [(f["detail"], f["source_string"]) for f in checks.flags if f["kind"] == "ungrounded_gap"]
    assert [g[1] for g in gaps] == ["No FY2019 data.", "Tesla is not covered."]
    assert "2019" in gaps[0][0] and "Tesla" in gaps[1][0]
    note = next(f for f in checks.flags if f["kind"] == "ungrounded_not_comparable")
    assert "WFC, TSLA, ZZZZ, 2019" in note["detail"]


def test_neighbour_quote_is_noted_and_a_straddling_quote_is_found():
    context = Context(chunks=[SEGMENTS_2025, SEGMENTS_2024], n_tokens=160, cids=["C1", "C2"])
    by_cid = {"C1": SEGMENTS_2025, "C2": SEGMENTS_2024}
    # The quote lies wholly in the uncited neighbour: found, and said so;
    # the figure and column are then read from that neighbour.
    checks = resolver.check(answer_of(nvda_claim("Compute & Networking revenue was $31,036 million.",
                                                 "Three Months Ended 2024-10-27\nRevenue | $31,036 | $4,046 | $35,082",
                                                 period_end="2024-10-27")), context, by_cid)
    assert kinds(checks) == ["quote_in_neighbour"] and checks.flags[0]["cid"] == "C2"
    assert checks.quotes_found == (1, 1) and checks.columns_matched == (1, 1)
    # A quote across the boundary is an exact quote of the cited chunk.
    straddle = SEGMENTS_2025.text[-60:] + "\n" + SEGMENTS_2024.text[:40]
    checks = resolver.check(answer_of(nvda_claim("Total operating income was $38,267 million.", straddle)),
                            context, by_cid)
    assert checks.quotes_found == (1, 1) and checks.figures_in_quote == (1, 1)
    assert not [f for f in checks.flags if f["kind"].startswith("quote")]


def test_fiscal_year_label_columns_need_the_filers_year_end_month(tuning_index):
    chunk = tuning_index.by_id[APPLE_FY2023_SEGMENTS_CHUNK]
    assert [c.label for c in chunk.columns][2] == "FY2022" and chunk.columns[2].period_end is None
    context, by_cid = one_chunk_context(chunk)
    row = "Total net sales | $383,285 | (3)% | $394,328 | 8% | $365,817"
    checks = resolver.check(answer_of(claim("C1", "Total net sales were $394,328 million in fiscal 2022.", row,
                                            period_end="2022-12-31", kind="fiscal_year", ticker="AAPL")),
                            context, by_cid)
    assert kinds(checks) == ["column_mismatch"] and checks.flags[0]["source_string"] == "FY2022"
    checks = resolver.check(answer_of(claim("C1", "Total net sales were $394,328 million in fiscal 2022.", row,
                                            period_end="2022-09-30", kind="fiscal_year", ticker="AAPL")),
                            context, by_cid)
    assert checks.flags == [] and checks.columns_matched == (1, 1)


def test_month_end_slack_is_one_week():
    chunk = SEGMENTS_2025
    column = Column(0, "Three Months Ended Oct 26, 2025", "2025-10-26", "three_months")
    assert resolver.period_matches(column, "2025-10-31", chunk) is True
    assert resolver.period_matches(Column(0, "x", "2025-10-24", "three_months"), "2025-10-31", chunk) is False
    assert resolver.period_matches(column, "2025-10-30", chunk) is False


def test_sign_and_short_quotes(tuning_index):
    context, by_cid = one_chunk_context(tuning_index.by_id[APPLE_FY2023_SEGMENTS_CHUNK])
    checks = resolver.check(answer_of(claim("C1", "Total net sales grew 3% in fiscal 2023.",
                                            "Total net sales | $383,285 | (3)% | $394,328 | 8% | $365,817",
                                            period_end="2023-09-30", kind="fiscal_year", ticker="AAPL")),
                            context, by_cid)
    assert "sign_mismatch" in kinds(checks)
    checks = resolver.check(answer_of(claim("C1", "Total net sales fell 3% in fiscal 2023.",
                                            "Total net sales | $383,285 | (3)% | $394,328 | 8% | $365,817",
                                            period_end="2023-09-30", kind="fiscal_year", ticker="AAPL")),
                            context, by_cid)
    assert "sign_mismatch" not in kinds(checks)
    context, by_cid = one_chunk_context(tuning_index.by_id[NII_CHUNK])
    checks = resolver.check(answer_of(claim("C1", "Net interest income was $23,966 million.", "23,966")), context, by_cid)
    assert kinds(checks) == ["quote_too_short", "figure_not_in_quote"]
    assert checks.quotes_found == (0, 1) and checks.figures_in_quote == (0, 1)


@pytest.mark.parametrize("text, expected", [
    ("Per Note 27 and excerpt 13, in a 52-week year net interest income was $23,966 million.", ["$23,966 million"]),
    ("As of 9/30/2025 (30 September 2025) deposits were $2.5 trillion, the 3rd rise.", ["$2.5 trillion"]),
    ("Net sales fell (3)% in FY2023 and 2% in Q4 2024.", ["(3)%", "2%"]),
])
def test_figures_in_claim_text_masks_counts_and_dates(text, expected):
    assert [f.text for f in resolver.figures_in(text)] == expected
