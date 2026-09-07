"""Chunker acceptance tests against the real corpus zip.

The corpus is parsed and chunked once per module (about a minute and a
half) and every test reads from that one dict. Single-table checks pin the
cases that drove a parsing rule: Apple's three fiscal-year columns and its
"shares in thousands" exception clause, the three-month and nine-month
columns in one JPMorgan row, Morgan Stanley's unit and years in one header
row. Corpus-wide rates are asserted against the thresholds the milestone
promises.
"""

import os
import re

import pytest

from chunk import (
    chunk_filing,
    estimate_tokens,
    format_cells,
    parse_units,
    shape_of_text,
)
from corpus import load_corpus
from index import load_companies
from models import Filing, Section

ZIP = os.path.join(os.path.dirname(__file__), "..", "data", "edgar_corpus.zip")
COMPANIES = os.path.join(os.path.dirname(__file__), "..", "service", "companies.yaml")
REQUIRED = ("file", "cik", "ticker", "form", "period_end", "fiscal_label", "item",
            "header", "chunk_id")
# A table-of-contents row: an item number or a section title, then a page
# reference and nothing else. "Item 1A. | Risk Factors | 9-31",
# "Consolidated Statements of Operations | 28". A data row such as
# "U.S. equity funds | 115" has the same shape and a different label.
TOC_ROW_RE = re.compile(
    r"^(?:Item\s+\d{1,2}[A-C]?\.?|PART\s+[IV]+|(?:Consolidated|Notes to|Reports? of|Risk Factors"
    r"|Business|Properties|Legal Proceedings|Management's Discussion|Financial Statements"
    r"|Controls and Procedures|Exhibits|Signatures|Quantitative and Qualitative)[^|]*)"
    r"\s*\|(?:[^|]*\|)?\s*(?:Pages?\s*)?(?:[A-Z]-)?\d{1,3}(?:\s*-\s*\d{1,3})?$")


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(ZIP, load_companies(COMPANIES))


@pytest.fixture(scope="module")
def chunked(corpus):
    return {f.file: chunk_filing(f) for f in corpus}


def one(corpus, prefix):
    hits = [f for f in corpus if f.file.startswith(prefix)]
    assert len(hits) == 1, prefix
    return hits[0]


def chunks_with(chunked, corpus, prefix, needle):
    return [c for c in chunked[one(corpus, prefix).file] if needle in c.text]


def cells_after_label(chunk, row_start):
    """The value cells of the row that starts with `row_start`, in order."""
    row = next(l for l in chunk.text.split("\n") if l.startswith(row_start))
    return row.split(" | ")[1:]


# --- cell and unit helpers (no corpus) ----------------------------------------


def test_format_cells_merges_affixes():
    assert format_cells("Products | $ | 294,866 | | | $ | 298,085 |") == ["Products", "$294,866", "$298,085"]
    assert format_cells("Equity | $ | (72,431 | ) | | 5,293 |") == ["Equity", "$(72,431)", "5,293"]
    assert format_cells("Noninterest revenue | $ | 22,461 | | 17 | % |") == ["Noninterest revenue", "$22,461", "17%"]
    # In a header row "$" and "%" are column titles, so they stay cells.
    assert format_cells("(Dollars in millions) | | 2025 | | $ | | %") == ["(Dollars in millions)", "2025", "$", "%"]


def test_parse_units_normalizes_declarations():
    assert parse_units("(In millions, except per share amounts)") == "USD millions, except per share amounts"
    assert parse_units("$ in millions | 2025 | 2024 | 2023") == "USD millions"
    assert parse_units("(Dollars in billions)") == "USD billions"
    assert parse_units("(MILLIONS, EXCEPT PER SHARE DATA)") == "USD millions, except per share data"
    assert parse_units("(thousands of barrels daily)") == "thousands of barrels daily"
    assert parse_units("We repaid $1.2 billion of notes.") is None


def test_estimate_tokens_rounds_up():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


# --- pinned tables -------------------------------------------------------------


def test_aapl_income_statement_columns_and_units(chunked, corpus):
    hits = chunks_with(chunked, corpus, "AAPL_10K_2024Q3", "Products | $294,866 | $298,085 | $316,199")
    assert hits, "income statement row not found in one chunk"
    chunk = hits[0]
    assert chunk.kind == "table"
    assert chunk.column_source == "parsed"
    assert [c.period_end for c in chunk.columns] == ["2024-09-28", "2023-09-30", "2022-09-24"]
    assert {c.duration for c in chunk.columns} == {"fiscal_year"}
    assert chunk.units_source == "declared"
    assert chunk.units.startswith("USD millions")
    assert "shares" in chunk.units and "thousands" in chunk.units
    assert "CONSOLIDATED STATEMENTS OF OPERATIONS" in chunk.header
    assert chunk.header.startswith("Apple Inc (AAPL, CIK 320193) | 10-K FY2024, fiscal year ended 2024-09-28")


def test_jpm_three_and_nine_month_columns(chunked, corpus):
    hits = chunks_with(chunked, corpus, "JPM_10Q_2025Q3", "\nNet interest income | 23,966")
    assert hits
    chunk = hits[0]
    values = cells_after_label(chunk, "Net interest income | 23,966")
    three = chunk.columns[values.index("23,966")]
    nine = chunk.columns[values.index("70,448")]
    assert three.duration == "three_months" and three.period_end == "2025-09-30"
    assert nine.duration == "nine_months" and nine.period_end == "2025-09-30"
    assert chunk.units.startswith("USD millions")


def test_ms_unit_and_years_in_one_header_row(chunked, corpus):
    """Every table whose header row is "$ in millions | 2025 | 2024 | 2023"
    declares USD millions and three year columns. The same line also
    appears inside prose chunks, where a two-row table sits under the
    three-row minimum and packs as prose, so only table chunks count."""
    hits = [c for c in chunks_with(chunked, corpus, "MS_10K_2026-02-19",
                                   "\n$ in millions | 2025 | 2024 | 2023\n") if c.kind == "table"]
    assert len(hits) >= 10
    for chunk in hits:
        assert chunk.units == "USD millions" and chunk.units_source == "declared"
        assert chunk.column_source == "parsed"
        assert len(chunk.columns) % 3 == 0
        for year in ("2025", "2024", "2023"):
            assert any(year in column.label for column in chunk.columns), chunk.columns
    plain = [c for c in hits if len(c.columns) == 3]
    assert len(plain) >= 10
    for chunk in plain:
        for column, year in zip(chunk.columns, ("2025", "2024", "2023")):
            assert year in column.label


def test_nvda_segment_table_period_under_dimension_header(chunked, corpus):
    """Note 16 prints "Compute & Networking | Graphics | All Other |
    Consolidated" once and "Year Ended Jan 26, 2025" as a one-cell row
    under it: the period applies to every segment column, and the four
    columns stay four. The later "Year Ended Jan 28, 2024" rows open
    their own chunks under their own period."""
    hits = [c for c in chunks_with(chunked, corpus, "NVDA_10K_2025-02-26",
                                   "\nRevenue | $116,193 | $14,304 | - | $130,497")
            if c.note_title and "Note 16" in c.note_title]
    assert len(hits) == 1
    chunk = hits[0]
    assert chunk.column_source == "parsed"
    assert [c.label.split(" FY")[0] for c in chunk.columns] == [
        "Compute & Networking", "Graphics", "All Other", "Consolidated"]
    assert {c.period_end for c in chunk.columns} == {"2025-01-26"}
    assert {c.duration for c in chunk.columns} == {"fiscal_year"}
    values = cells_after_label(chunk, "Revenue | $116,193")
    assert "Consolidated" in chunk.columns[values.index("$130,497")].label
    assert "Year Ended Jan 28, 2024" not in chunk.text
    later = [c for c in chunks_with(chunked, corpus, "NVDA_10K_2025-02-26",
                                    "\nRevenue | $47,405 | $13,517 | - | $60,922")
             if c.note_title and "Note 16" in c.note_title]
    assert len(later) == 1 and later[0].column_source == "parsed"
    assert {c.period_end for c in later[0].columns} == {"2024-01-28"}


def test_jpm_reconciliation_period_cell_beside_dimension_labels(chunked, corpus):
    """"As of or for the three monthsended September 30,(in millions,
    except ratios) | Corporate | Reconciling Items(a) | Total" over six year
    cells: the glued period applies to every column and the three names
    take two year cells each."""
    hits = chunks_with(chunked, corpus, "JPM_10Q_2025Q3",
                       "\nNet interest income | 1,406 | 2,915 | (105) | (120) | 23,966 | 23,405")
    assert len(hits) == 1
    chunk = hits[0]
    assert chunk.column_source == "parsed" and len(chunk.columns) == 6
    values = cells_after_label(chunk, "Net interest income | 1,406")
    total = chunk.columns[values.index("23,966")]
    assert total.label.endswith("Total")
    assert total.duration == "three_months" and total.period_end == "2025-09-30"
    assert chunk.columns[values.index("1,406")].label.endswith("Corporate")
    assert chunk.columns[values.index("23,405")].period_end == "2024-09-30"


def test_bac_table_5_quarter_names_and_nine_month_years(chunked, corpus):
    """"2025 Quarters | 2024 Quarters" over "Third | Second | First |
    Fourth | Third", with "Nine Months Ended September 30" over the two
    year cells beside them. Bank of America's fiscal year ends in
    December, so a named quarter resolves to a calendar quarter end."""
    hits = chunks_with(chunked, corpus, "BAC_10Q_2025Q3",
                       "\nNet interest income | $15,233 | $14,670 | $14,443 | $14,359 | $13,967 | $44,346 | $41,701")
    assert len(hits) == 1
    chunk = hits[0]
    assert chunk.column_source == "parsed" and len(chunk.columns) == 7
    values = cells_after_label(chunk, "Net interest income | $15,233")
    third = chunk.columns[values.index("$15,233")]
    assert third.duration == "three_months" and third.period_end == "2025-09-30"
    assert chunk.columns[values.index("$14,359")].period_end == "2024-12-31"
    assert chunk.columns[values.index("$13,967")].period_end == "2024-09-30"
    nine = chunk.columns[values.index("$44,346")]
    assert nine.duration == "nine_months" and nine.period_end == "2025-09-30"
    assert chunk.columns[values.index("$41,701")].period_end == "2024-09-30"


def test_long_table_repeats_caption_and_headers():
    rows = ["Revenue from operating segment number %d of the company | %d,%03d | %d,%03d"
            % (n, 10 + n, n * 7 % 1000, 20 + n, n * 3 % 1000) for n in range(40)]
    body = ("CONSOLIDATED STATEMENTS OF INCOME (In millions)\n"
            "| Years ended\n"
            "| December 31, 2025 | | December 31, 2024\n" + "\n".join(rows) + "\n")
    filing = Filing(file="TEST_10K.txt", cik="0000000001", ticker="TST", company="Test Co",
                    form="10-K", filing_date="2026-02-01", period_end="2025-12-31",
                    period_source="header", fiscal_year=2025, fiscal_quarter=None,
                    fiscal_label="FY2025", fye_month=12, url="", body=body,
                    sections=[Section(part=None, item="8", title="Financial Statements",
                                      start=0, end=len(body), is_pointer_stub=False)])
    chunks = chunk_filing(filing)
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) >= 2
    for chunk in tables:
        lines = chunk.text.split("\n")
        assert lines[0] == "CONSOLIDATED STATEMENTS OF INCOME (In millions)"
        assert lines[1] == "Years ended"
        assert lines[2] == "December 31, 2025 | December 31, 2024"
        assert [c.period_end for c in chunk.columns] == ["2025-12-31", "2024-12-31"]
        assert chunk.units == "USD millions"
    found = [l for c in tables for l in c.text.split("\n") if l.startswith("Revenue from")]
    assert len(found) == 40


# --- corpus-wide invariants ----------------------------------------------------


def test_every_chunk_carries_its_identity(chunked):
    for chunks in chunked.values():
        for chunk in chunks:
            for name in REQUIRED:
                assert getattr(chunk, name), (chunk.file, chunk.seq, name)
            assert chunk.kind in ("prose", "table", "cover")
            assert chunk.n_tokens == estimate_tokens(chunk.header + "\n" + chunk.text)


def test_prose_chunk_sizes(chunked):
    """Prose chunks are 400 to 1900 chars except the last of a run: a run
    ends at a table or the section end, and only there may a chunk be
    short. Contiguity (the next chunk starts inside or right after this
    one) is what tells a run's middle from its end."""
    for chunks in chunked.values():
        for chunk, following in zip(chunks, chunks[1:] + [None]):
            if chunk.kind == "table":
                continue
            assert 300 <= len(chunk.text) <= 1900, (chunk.file, chunk.seq, len(chunk.text))
            contiguous = (following is not None and following.kind == chunk.kind
                          and following.item == chunk.item
                          and following.char_start <= chunk.char_end + 1)
            if contiguous:
                assert len(chunk.text) >= 400, (chunk.file, chunk.seq, len(chunk.text))


def test_no_chunk_is_mostly_table_of_contents(chunked):
    for chunks in chunked.values():
        for chunk in chunks:
            lines = chunk.text.split("\n")
            toc_rows = sum(1 for l in lines if TOC_ROW_RE.match(l))
            assert toc_rows < 3 or toc_rows <= 0.5 * len(lines), (chunk.file, chunk.seq)


def test_chunk_ids_are_stable(corpus):
    filing = one(corpus, "AAPL_10K_2024Q3")
    first = [c.chunk_id for c in chunk_filing(filing)]
    second = [c.chunk_id for c in chunk_filing(filing)]
    assert first == second
    assert len(set(first)) == len(first)


def test_corpus_wide_table_rates(chunked):
    """"parsed" now means the column count equals the chunk's own value-cell
    count, so a figure can be read by position; columns that a header row
    yielded without that match are kept under "unverified_shape"."""
    tables = [c for chunks in chunked.values() for c in chunks if c.kind == "table"]
    parsed = sum(1 for c in tables if c.column_source == "parsed")
    labeled = sum(1 for c in tables if c.column_source in ("parsed", "unverified_shape"))
    declared = sum(1 for c in tables if c.units_source == "declared")
    assert parsed >= 0.55 * len(tables), (parsed, len(tables))
    assert labeled >= 0.80 * len(tables), (labeled, len(tables))
    assert declared >= 0.60 * len(tables), (declared, len(tables))
    for chunk in tables:
        assert (chunk.column_source in ("parsed", "unverified_shape")) == bool(chunk.columns)
        assert (chunk.units is None) == (chunk.units_source is None)
        assert ("columns:" in chunk.header) == (chunk.column_source == "parsed")


def test_parsed_columns_match_the_rows_they_label(chunked):
    """Every parsed chunk's column count equals the modal value-cell count of
    its own labeled data rows, and no parsed chunk collapses a multi-cell
    table into one column. Both are what "parsed" means now, so the shares
    are exact rather than thresholds."""
    parsed = [c for chunks in chunked.values() for c in chunks
              if c.kind == "table" and c.column_source == "parsed"]
    assert parsed
    mismatched = [c for c in parsed if shape_of_text(c.text) != len(c.columns)]
    assert not mismatched, [(c.file, c.chunk_id) for c in mismatched[:5]]
    collapsed = [c for c in parsed if len(c.columns) == 1 and (shape_of_text(c.text) or 0) >= 2]
    assert len(collapsed) < 0.01 * len(parsed), [(c.file, c.chunk_id) for c in collapsed[:5]]
