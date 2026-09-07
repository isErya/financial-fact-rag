"""Cut a parsed Filing into retrievable chunks.

Flow per section: split the section text into prose runs and table blocks,
pack each into chunks, then stamp every chunk with the filing metadata and a
header line naming the company, the filing, the item, the table caption, the
units and the columns. The header line is part of the text the embedder and
the model read, so a chunk quoted on its own still says which filing, which
statement, which column and which unit it came from.

Tables are the reason this module exists. A 10-K income statement shows three
fiscal years side by side and a 10-Q shows three-month and nine-month columns
in one row; a chunk that loses the header row leaves unlabeled digits, and
the unit line ("In millions, except per-share amounts") is declared once per
statement. Every table chunk therefore repeats the caption, the unit line and
the header rows, and carries a parsed `columns` list so the milestone 3
resolver can check which column a quoted figure came from.

Where each decision is made:
  split_blocks     what counts as a table block versus prose
  find_captions    the title lines above a table that travel with it
  format_cells     cell cleanup ("$" and "(" merge into the number they open)
  read_row         caption row, header row, sub-heading or data row
  classify_rows    the prefix rows every chunk of a table repeats
  distribute       which header label stands over which column cell
  with_change_columns  a "% Change" label whose column has no header cell
  parse_columns    header rows to Column records (label, period_end, duration)
  data_shape       the value-cell count a column list has to match
  period_groups    a period sub-heading inside the data restarts the columns
  parse_units      a unit declaration to "USD millions, except ..."
  pack_table       table chunk sizes
  pack_prose       sentence packing with a one-sentence overlap
  header_line      the metadata line embedded in every chunk

Offsets in a Chunk index into Filing.body, the same text the Section offsets
use. Two calls on the same Filing produce the same chunk ids, because an id
is a hash of the file name and the chunk's character span.

Failure policy: a Filing whose period_end or fiscal_year is missing raises
in build_column, because a column stamped with a made-up date would be
cited as fact. Column labeling itself never raises; it degrades. A table
whose header rows cannot be laid over its data rows with certainty keeps
its columns for display under column_source "unverified_shape", or drops
them when no header row carries a date, and a label that would have to be
guessed (a period cell whose count does not divide the columns, a quarter
name under a filer whose year does not end in December) leaves period_end
and duration None. A later resolver compares a quoted figure's column to
the claim's period, so a wrong label turns a right answer red or a wrong
one green, while a missing label only costs the check.
"""

import calendar
import datetime as dt
import hashlib
import re
from collections import Counter
from dataclasses import dataclass

from models import Chunk, Column, Filing, Note, Section

# Chunk sizes in characters of chunk text; the header line is not counted.
SOFT_MAX = 1400
HARD_MAX = 1900
PROSE_MIN = 400
CHUNK_MIN = 300
OVERLAP_MAX = 200

# Table block shape.
MIN_TABLE_ROWS = 3
INTERLEAVE_MAX = 80
CAPTION_MAX_CHARS = 250
CAPTION_MAX_LINES = 2
MAX_CAPTION_ROWS = 2
MAX_HEADER_ROWS = 4
HEADER_ROW_MAX = 500
TOC_SHARE = 0.5
CAPTION_IN_HEADER_MAX = 80


def estimate_tokens(text: str) -> int:
    """Length over four, rounded up. The one place a token count is estimated,
    so milestone 4 can swap in a tokenizer here and every budget follows."""
    return (len(text) + 3) // 4


# ---------------------------------------------------------------------------
# Lines and blocks
# ---------------------------------------------------------------------------


@dataclass
class Line:
    start: int
    end: int
    text: str


def _lines(body: str, start: int, end: int) -> list[Line]:
    out = []
    pos = start
    for text in body[start:end].split("\n"):
        out.append(Line(pos, pos + len(text), text))
        pos += len(text) + 1
    return out


def split_blocks(lines: list[Line]) -> list[tuple[str, list[Line]]]:
    """Alternate ("prose", lines) and ("table", lines) blocks.

    A table block is a maximal run of lines containing "|". One interleaved
    line without a pipe is allowed inside a run when it is shorter than
    INTERLEAVE_MAX chars, which is how a sub-caption such as "ASSETS:" or a
    blank line sits between two rows of the same statement. A run with fewer
    than MIN_TABLE_ROWS pipe lines is prose.
    """
    blocks: list[tuple[str, list[Line]]] = []
    prose: list[Line] = []
    table: list[Line] = []
    gap: Line | None = None

    def close_table() -> None:
        nonlocal prose, table, gap
        if sum(1 for l in table if "|" in l.text) >= MIN_TABLE_ROWS:
            if prose:
                blocks.append(("prose", prose))
                prose = []
            blocks.append(("table", table))
        else:
            prose.extend(table)
        if gap is not None:
            prose.append(gap)
        table, gap = [], None

    for line in lines:
        if "|" in line.text:
            if gap is not None:
                table.append(gap)
                gap = None
            table.append(line)
        elif table and gap is None and len(line.text) < INTERLEAVE_MAX:
            gap = line
        else:
            if table:
                close_table()
            prose.append(line)
    if table:
        close_table()
    if prose:
        blocks.append(("prose", prose))
    return blocks


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

# A number as it lands in a cell after merging: "$65,821", "(6,659)", "(9)%",
# "0.01". Years are tested before this so "2025" reads as a year.
NUMBER_RE = re.compile(r"^\$?\(?\d[\d,]*(?:\.\d+)?\)?\s?%?$")
# A cell standing in a numeric column without a number: a dash for zero or
# none, "NM" or "NA" for a ratio with no meaning in that period.
PLACEHOLDER_RE = re.compile(r"(?i)^(?:-{1,2}|n/?[am]|n\.[am]\.|\*)$")
NUMBER_START_RE = re.compile(r"^[($]*\d")
NUMBER_END_RE = re.compile(r"[\d)]$")
# "2025", "FY2025", "Fiscal 2024", "2025(a)", "2024E"
YEAR_RE = re.compile(r"^(?:FY\s?|Fiscal\s+)?((?:19|20)\d{2})(?:\s*\([a-z0-9]{1,2}\)|[A-Za-z])?$", re.I)

_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"]
_MONTH_ALTS = "|".join(_MONTHS + [m[:3] for m in _MONTHS] + ["sept"])
_MONTH_ALTS_TITLE = "|".join([m.capitalize() for m in _MONTHS]
                             + [m[:3].capitalize() for m in _MONTHS] + ["Sept"])
# A month name, then an optional day and an optional year. The month may be
# glued to the word before it ("Three Months EndedMarch 31,") when it keeps
# its capital. The day may not run into the year ("December 2025" has no day).
DATE_RE = re.compile(
    r"(?:(?<![A-Za-z])(?i:" + _MONTH_ALTS + r")|(?<=[a-z])(?:" + _MONTH_ALTS_TITLE + r"))"
    r"\.?(?![a-z])\s*(?:(\d{1,2})(?!\d)(?:st|nd|rd|th)?)?\s*,?\s*((?:19|20)\d{2})?"
)
# "months" may run straight into "ended" ("three monthsended September 30,")
# when a filer's line break was dropped; nothing else may follow it.
DURATION_MONTHS_RE = re.compile(
    r"(?i)(?<![a-z])(three|six|nine|twelve|3|6|9|12)[\s-]*months?(?:(?![a-z])|(?=ended))")
# "Third", "Third Quarter", "Fourth Quarter 2024": a quarter named by its
# ordinal, which is a three-month column whose year comes from a label above.
QUARTER_NAME_RE = re.compile(
    r"(?i)^(first|second|third|fourth)(?:\s+quarter)?(?:\s+((?:19|20)\d{2}))?$")
# "2025 Quarters": the year a run of quarter-name columns belongs to.
QUARTERS_LABEL_RE = re.compile(r"(?i)^((?:19|20)\d{2})\s+quarters$")
_QUARTERS = {"first": 1, "second": 2, "third": 3, "fourth": 4}
# "Less than 12 Months | 12 Months or Greater" are aging buckets, never periods.
DURATION_EXCLUDE_RE = re.compile(r"(?i)less than|or greater|or more|more than|or less|\bover\b|\bunder\b")
QUARTER_RE = re.compile(r"(?i)quarter\s*ended|(?<![a-z])(?:first|second|third|fourth)\s+quarter")
YEAR_ENDED_RE = re.compile(
    r"(?i)(?<![a-z])(?:fiscal\s+)?years?\s+end(?:ed|ing)|(?<![a-z])fiscal(?:\s+years?)?(?![a-z])")
POINT_RE = re.compile(r"(?i)^\s*(?:as\s+of|at)(?![a-z])")
# "change" anywhere except inside "exchange"; the other words stand alone.
CHANGE_RE = re.compile(
    r"(?i)(?<![x])change|(?<![a-z])(?:vs\.?|versus|increase|decrease|variance|better|worse)(?![a-z])")
_DURATIONS = {"three": "three_months", "3": "three_months", "six": "six_months",
              "6": "six_months", "nine": "nine_months", "9": "nine_months",
              "twelve": "twelve_months", "12": "twelve_months"}
PERIOD_KINDS = {"year", "date", "period"}
HEADER_KINDS = {"year", "date", "period", "change", "unit"}


@dataclass
class Cell:
    """One header cell with what the column parser reads from it."""

    text: str
    kind: str  # "year", "date", "period", "quarter", "quarters", "change", "unit", "number", "other"
    year: int | None
    month: int | None
    day: int | None
    duration: str | None
    point: bool
    quarter: int | None = None


def format_cells(text: str) -> list[str]:
    """Split a row on "|", drop empty cells, and merge the affix cells.

    Filers put "$", "(", ")" and "%" in cells of their own next to the number
    they belong to. "(" and ")" merge with the number they enclose; "$" and
    "%" merge only when a number sits next to them, because in a header row
    "$" and "%" are column titles ("2025 vs. 2024 Change | $ | %"). A "$" in
    front of a "-" placeholder is dropped.
    """
    raw = [c.strip() for c in text.split("|")]
    merged: list[str] = []
    for c in raw:
        if not c:
            continue
        if c == ")" and merged and NUMBER_START_RE.match(merged[-1]):
            merged[-1] += c
        elif merged and merged[-1] == "(" and NUMBER_START_RE.match(c):
            merged[-1] += c
        else:
            merged.append(c)
    out: list[str] = []
    skip = False
    for i, c in enumerate(merged):
        if skip:
            skip = False
            continue
        nxt = merged[i + 1] if i + 1 < len(merged) else ""
        if c == "$" and NUMBER_START_RE.match(nxt):
            out.append(c + nxt)
            skip = True
        elif c == "$" and nxt == "-":
            continue
        elif c == "%" and out and NUMBER_END_RE.search(out[-1]):
            out[-1] += c
        else:
            out.append(c)
    return out


def date_parts(text: str) -> tuple[int | None, int | None, int | None]:
    """(month, day, year) from the first date phrase, each None when absent.
    A month with neither day nor year counts only behind "As of" or "At"."""
    m = DATE_RE.search(text)
    if not m:
        return None, None, None
    month = [x[:3] for x in _MONTHS].index(m.group(0)[:3].lower()) + 1
    day = int(m.group(1)) if m.group(1) else None
    year = int(m.group(2)) if m.group(2) else None
    if day is None and year is None and not POINT_RE.match(text):
        return None, None, None
    return month, day, year


def duration_of(text: str) -> str | None:
    if DURATION_EXCLUDE_RE.search(text):
        return None
    m = DURATION_MONTHS_RE.search(text)
    if m:
        return _DURATIONS[m.group(1).lower()]
    if QUARTER_RE.search(text):
        return "three_months"
    if YEAR_ENDED_RE.search(text):
        return "fiscal_year"
    return None


GLUED_DAY_YEAR_RE = re.compile(r"\b(\d{1,2})((?:19|20)\d{2})\b")


def read_cell(text: str) -> Cell:
    """Classify one header cell. Order matters: a bare year is a year before
    it is a number, and a period phrase carrying a date is a period. A day
    glued to its year ("December 312024") is split first."""
    text = GLUED_DAY_YEAR_RE.sub(r"\1 \2", text)
    ym = YEAR_RE.match(text)
    if ym:
        return Cell(text, "year", int(ym.group(1)), None, None, duration_of(text), False)
    if NUMBER_RE.match(text):
        return Cell(text, "number", None, None, None, None, False)
    qm = QUARTER_NAME_RE.match(text)
    if qm:
        year = int(qm.group(2)) if qm.group(2) else None
        return Cell(text, "quarter", year, None, None, "three_months", False,
                    _QUARTERS[qm.group(1).lower()])
    qs = QUARTERS_LABEL_RE.match(text)
    if qs:
        return Cell(text, "quarters", int(qs.group(1)), None, None, None, False)
    duration = duration_of(text)
    point = bool(POINT_RE.match(text))
    month, day, year = date_parts(text)
    if duration or point:
        return Cell(text, "period", year, month, day, duration, point)
    if month:
        return Cell(text, "date", year, month, day, None, False)
    if parse_units(text):
        return Cell(text, "unit", None, None, None, None, False)
    if CHANGE_RE.search(text):
        return Cell(text, "change", None, None, None, None, False)
    return Cell(text, "other", None, None, None, None, False)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


@dataclass
class Row:
    line: Line
    cells: list[Cell]
    kind: str  # "blank", "text", "caption", "header", "sub", "data"
    text: str
    first_filled: bool  # the cell before the first "|" holds text


TABLE_N_RE = re.compile(r"(?i)^table\s+\d+[a-z]?$")


def read_row(line: Line) -> Row:
    """One block line to a Row.

    "data" is any row carrying a number; "header" a row without numbers
    whose cells are titles or dates; "sub" a single filled cell in the first
    position ("Net sales:"); "caption" a numbered title row ("Table 1 |
    Summary Income Statement") or an interleaved line that reads as a title
    ("Consolidated Statement of Income"); "text" any other interleaved line
    or a pipe row too long to be a header (a paragraph laid out in cells).
    Whether a "sub" or "caption" row is a caption or a sub-heading depends
    on where it sits; classify_rows decides that.
    """
    if "|" not in line.text:
        text = line.text.strip()
        if not text:
            return Row(line, [], "blank", "", True)
        return Row(line, [], "caption" if caption_text(text) else "text", text, True)
    first_filled = bool(line.text.split("|", 1)[0].strip())
    texts = format_cells(line.text)
    if not texts:
        return Row(line, [], "blank", "", first_filled)
    cells = [read_cell(t) for t in texts]
    joined = " | ".join(texts)
    if any(c.kind == "number" for c in cells):
        return Row(line, cells, "data", joined, first_filled)
    if len(line.text) > HEADER_ROW_MAX:
        return Row(line, cells, "text", joined, first_filled)
    if first_filled and TABLE_N_RE.match(texts[0]):
        return Row(line, cells, "caption", joined, first_filled)
    if len(texts) == 1 and first_filled and (cells[0].kind not in HEADER_KINDS or joined.endswith(":")):
        return Row(line, cells, "sub", joined if ":" in joined else joined + ":", first_filled)
    return Row(line, cells, "header", joined, first_filled)


def is_dated_header(row: Row) -> bool:
    """A header row that opens a table: it carries a year or a date and is
    laid out across cells, so a period sub-heading in the label column
    ("July 31, 2022 to August 27, 2022:") does not count."""
    return (row.kind == "header" and (len(row.cells) >= 2 or not row.first_filled)
            and any(_dated(c) for c in row.cells))


def split_fused(rows: list[Row]) -> list[list[Row]]:
    """Cut a block into one list of rows per table.

    A short caption line between two tables is under the interleave limit,
    so split_blocks fuses them into one block; the chunks of the second
    table would then repeat the first table's headers and columns. A dated
    header row that follows data rows therefore starts a new table, taking
    with it the caption and sub-heading rows just above it. Plain text rows
    at that seam stay with the table before them as its footnotes. A
    trailing table without data rows is dropped.
    """
    tables: list[list[Row]] = []
    cur: list[Row] = []
    seen_data = False
    tail_start: int | None = None
    for row in rows:
        if row.kind == "data":
            cur.append(row)
            seen_data, tail_start = True, None
        elif seen_data and is_dated_header(row):
            split_at = len(cur) if tail_start is None else tail_start
            while split_at < len(cur) and cur[split_at].kind in ("text", "blank"):
                split_at += 1
            tables.append(cur[:split_at])
            cur = cur[split_at:] + [row]
            seen_data, tail_start = False, None
        else:
            if tail_start is None:
                tail_start = len(cur)
            cur.append(row)
    if seen_data:
        tables.append(cur)
    return tables


PAGE_REF_RE = re.compile(r"^(?:Pages?\s*)?(?:[A-Z]-)?\d{1,3}(?:\s*-\s*\d{1,3})?$")


def is_toc(rows: list[Row]) -> bool:
    """A table of contents: most rows end in a page reference and carry at
    most three cells ("Item 1A. | Risk Factors | 9-31")."""
    pipe_rows = [r for r in rows if r.cells]
    if not pipe_rows:
        return False
    refs = sum(1 for r in pipe_rows if len(r.cells) <= 3 and PAGE_REF_RE.match(r.cells[-1].text))
    return refs > TOC_SHARE * len(pipe_rows)


EXHIBIT_RE = re.compile(r"(?i)\bexhibits?\b|incorporated by reference")


def is_exhibit_list(rows: list[Row]) -> bool:
    """An exhibit index ("Exhibit Number | Description | Form | Filing
    Date"): a list of documents with no figure to quote, so it packs as
    prose the way a table of contents is left out."""
    head = [r for r in rows if r.cells][:3]
    return any(EXHIBIT_RE.search(r.text) for r in head)


def classify_rows(rows: list[Row]) -> tuple[list[Row], list[Row], list[Row]]:
    """Split a block into (caption rows, header rows, data rows).

    The leading rows without numbers form the prefix every chunk repeats. A
    "sub" row before any header row is a caption inside the table
    ("Financial performance of JPMorganChase | | |"); after a header row it
    is a sub-heading of the data ("Net sales: | | |") and the prefix ends.
    """
    captions: list[Row] = []
    headers: list[Row] = []
    for i, row in enumerate(rows):
        if row.kind == "blank":
            continue
        if row.kind in ("sub", "caption") and not headers and len(captions) < MAX_CAPTION_ROWS:
            captions.append(row)
        elif row.kind == "header" and len(headers) < MAX_HEADER_ROWS:
            headers.append(row)
        else:
            return captions, headers, [r for r in rows[i:] if r.kind != "blank"]
    return captions, headers, []


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

# "(In millions, except ...)", "(dollars in millions)", "$ in millions",
# "(MILLIONS)", "(millions of dollars)", "(shares in thousands)",
# "(thousands of barrels daily)". The scale word may not follow a digit, so
# "$1.2 billion" in prose never declares a unit.
UNIT_RE = re.compile(
    r"(?i)(?<![\d.,])(?<![\d.,] )"
    r"(?:\(|\$|;|(?<![a-z])(?:tabular|dollars?|amounts?|shares|units|in)(?![a-z]))[\s$]*"
    r"(?:(?:tabular|dollars?|amounts?|shares|units|in|and)\s+)*"
    r"(millions?|thousands?|billions?)(?![a-z])"
    r"(?:\s+of\s+(?:u\.?s\.?\s+)?([a-z]+(?:\s+[a-z]+)?))?"
)
# The clause ends at the closing parenthesis, the line end, or the next
# cell ("in millions, except per share data | 2025 | 2024").
EXCEPT_RE = re.compile(r"(?i)[^)\n|]*?(except\b[^)\n|]*)")
_SCALES = {"million": "millions", "thousand": "thousands", "billion": "billions"}


def parse_units(text: str) -> str | None:
    """The unit declaration in a line, normalized, or None.

    Money becomes "USD millions"; share counts "shares thousands"; anything
    else ("thousands of barrels daily") is kept as written. An exception
    clause is appended verbatim in lowercase, since "except per-share
    amounts" is exactly what a reader of the figure needs to know.
    """
    m = UNIT_RE.search(text)
    if not m:
        return None
    scale = _SCALES[m.group(1).lower().rstrip("s")]
    noun = (m.group(2) or "").lower()
    prefix = m.group(0).lower()
    if noun and not noun.startswith("dollar"):
        units = "%s of %s" % (scale, noun)
    elif "dollar" in prefix or "$" in prefix:
        # "Dollars and shares in millions" is money first.
        units = "USD " + scale
    elif "share" in prefix:
        units = "shares " + scale
    elif "unit" in prefix:
        units = "units " + scale
    else:
        units = "USD " + scale
    tail = EXCEPT_RE.match(text[m.end():])
    if tail:
        units += ", " + " ".join(tail.group(1).lower().split()).rstrip(" ,;")
    return units


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------

# A statement title at the end of a line, glued to the sentence before it:
# "...accompanying notes. Apple Inc.CONSOLIDATED STATEMENTS OF OPERATIONS(In
# millions, except ...)". Two or more uppercase words, then an optional
# parenthetical, then the line end.
TRAILING_TITLE_RE = re.compile(
    r"((?:[A-Z][A-Z&',./-]*[ ]+){1,}[A-Z][A-Z&',./-]*(?:\s*\([^()]*\))?)\s*$")
TRAILING_TITLE_MIN = 12
TITLE_WORD_RE = re.compile(r"(?i)statement|\btable\b|summary")
# "(a)For other income ...", "(1) Includes ...": a footnote of the table
# above, never the caption of the table below.
FOOTNOTE_RE = re.compile(r"^\s*\(?[a-z0-9]{1,2}\)")


def caption_text(line: str) -> str | None:
    """The caption inside a line above a table, or None for ordinary prose.

    A long line qualifies only through a trailing uppercase title; a short
    line qualifies whole when it is mostly uppercase or names a statement,
    table, summary or unit. A footnote line never qualifies.
    """
    if FOOTNOTE_RE.match(line):
        return None
    m = TRAILING_TITLE_RE.search(line)
    if m and TRAILING_TITLE_MIN <= len(m.group(1)) <= CAPTION_MAX_CHARS:
        return m.group(1)
    if not line.strip() or len(line) > CAPTION_MAX_CHARS:
        return None
    letters = [ch for ch in line if ch.isalpha()]
    if len(letters) >= 6 and sum(ch.isupper() for ch in letters) >= 0.6 * len(letters):
        return line.strip()
    if TITLE_WORD_RE.search(line) or parse_units(line):
        return line.strip()
    return None


def find_captions(prose: list[Line], block_start: int) -> list[Line]:
    """Take up to CAPTION_MAX_LINES title lines off the end of `prose` and
    return them in reading order.

    A title glued to the end of a longer line is cut off that line and the
    rest stays in the prose run; no further line above is then a caption.
    Blank lines are skipped; a line more than CAPTION_MAX_CHARS above the
    first row ends the search.
    """
    found: list[Line] = []
    while prose and len(found) < CAPTION_MAX_LINES:
        line = prose[-1]
        if not line.text.strip():
            prose.pop()
            continue
        if block_start - line.end > CAPTION_MAX_CHARS:
            break
        text = caption_text(line.text)
        if text is None:
            break
        prose.pop()
        at = line.text.rfind(text)
        found.append(Line(line.start + at, line.start + at + len(text), text))
        rest = line.text[:at]
        if rest.strip():
            prose.append(Line(line.start, line.start + len(rest), rest))
            break
    found.reverse()
    return found


def short_caption(text: str) -> str:
    """Caption as it appears in the header line: no unit parenthetical,
    trimmed to CAPTION_IN_HEADER_MAX chars. The tail is kept when trimming
    because the statement name sits at the end of a glued heading."""
    text = re.sub(r"\([^()]*\)", " ", text)
    text = " ".join(text.split()).strip(" :")
    if len(text) > CAPTION_IN_HEADER_MAX:
        text = text[-CAPTION_IN_HEADER_MAX:]
        text = text[text.find(" ") + 1:] if " " in text else text
    return text


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


def label_cells(row: Row) -> tuple[Cell | None, list[Cell]]:
    """(lead cell, cells that stand over columns) for a header row.

    A filled first cell is a lead when it is a unit ("$ in millions | 2025
    | 2024") or a caption over cells that are all dated: "Year Ended
    December 31, | 2022 | 2021" and "Segment | 2025 | 2024" both have two
    columns. When the other cells are names too ("JPMorganChase & Co. |
    JPMorganChase Bank, N.A. | ..."), the first cell is a column like the
    rest, and a period beside dimension names ("As of or for the three
    months ended September 30, | Corporate | Total") stays with them so
    that `distribute` can lay it over every column.
    """
    cells = row.cells
    if row.first_filled and len(cells) > 1:
        first = cells[0]
        rest_dated = all(c.kind in PERIOD_KINDS or c.kind in ("quarter", "quarters") for c in cells[1:])
        if first.kind == "unit" or (rest_dated and (first.kind in ("other", "change")
                                                    or (first.kind in PERIOD_KINDS and not first.year))):
            return first, cells[1:]
    return None, cells


def _labels_of(lead: Cell | None, cells: list[Cell]) -> list[Cell]:
    """The labels a header row contributes to the columns under it: its
    cells, and its lead when that is a period ("Year Ended December 31,"
    applies to every year beside it; a unit or a caption word does not)."""
    if lead is not None and lead.kind in PERIOD_KINDS:
        return [lead] + cells
    return cells


def _dated(cell: Cell) -> bool:
    """A cell that pins a period: a year, a date, a period phrase carrying
    a year or a month, or a "2025 Quarters" label."""
    return (cell.kind in PERIOD_KINDS or cell.kind == "quarters") and bool(cell.year or cell.month)


def quarter_runs(quarters: list[int]) -> list[list[int]]:
    """Split quarter-name columns into the runs that belong to one year.

    "Third | Second | First | Fourth | Third" is the last five quarters,
    newest first: a run breaks where the sequence turns around, so the
    first three positions are one year and the next two the year before.
    A repeated quarter also breaks the run. Returns lists of positions.
    """
    runs: list[list[int]] = []
    cur: list[int] = []
    direction = 0
    for pos, q in enumerate(quarters):
        if cur:
            step = q - quarters[cur[-1]]
            if step == 0 or (direction and (step > 0) != (direction > 0)):
                runs.append(cur)
                cur, direction = [], 0
            elif direction == 0:
                direction = step
        cur.append(pos)
    if cur:
        runs.append(cur)
    return runs


def distribute(labels: list[Cell], columns: list[Cell]) -> list[list[Cell]]:
    """Spread the labels of one header row over the column cells; one list
    per column cell, since a row that carries both a period and dimension
    names gives a cell both ("Three months ended September 30," and
    "Total" over the same year cell).

    Period labels (durations, dates, "As of") stand over the period cells
    of the column row when it has any, otherwise over every cell: "Nine
    Months Ended September 30" over "Third | Second | First | Fourth |
    Third | 2025 | 2024" covers the two year cells only, while "Three
    months ended September 30," over six year cells covers all six. One
    period label covers all of its targets; several split them into equal
    groups when the counts divide (two durations over "2025 | 2024 |
    Change | 2025 | 2024 | Change" give a three-month pair and a
    nine-month pair, and each Change cell keeps no period). Dimension
    labels (approach, segment, "Change") stand over the non-period cells
    when the row also carries a period, otherwise over every cell, again
    only in equal groups: three segment names over six year cells give two
    each. A row of "<year> Quarters" labels is matched to quarter-name
    cells by `quarter_runs`. When the counts do not divide, a label goes
    to no column rather than to a guessed one; raw cell positions would
    not settle it, because filers pad the rows of one table with different
    numbers of empty cells.
    """
    n = len(columns)
    out: list[list[Cell]] = [[] for _ in range(n)]
    if not labels or not n:
        return out

    def spread(items: list[Cell], slots: list[int]) -> None:
        m, k = len(items), len(slots)
        if m == 1:
            for slot in slots:
                out[slot].append(items[0])
        elif k and k % m == 0:
            for j, slot in enumerate(slots):
                out[slot].append(items[j // (k // m)])

    period_labels = [c for c in labels if c.kind in PERIOD_KINDS]
    other_labels = [c for c in labels if c.kind not in PERIOD_KINDS]
    period_slots = [i for i, c in enumerate(columns) if c.kind in PERIOD_KINDS]
    other_slots = [i for i, c in enumerate(columns) if c.kind not in PERIOD_KINDS]
    every = list(range(n))
    if other_labels and all(c.kind == "quarters" for c in other_labels):
        quarter_slots = [i for i, c in enumerate(columns) if c.kind == "quarter"]
        runs = quarter_runs([columns[i].quarter or 0 for i in quarter_slots])
        if quarter_slots and len(runs) == len(other_labels):
            for label, run in zip(other_labels, runs):
                for pos in run:
                    out[quarter_slots[pos]].append(label)
            other_labels = []
    if period_labels:
        spread(period_labels, period_slots or every)
    if other_labels:
        spread(other_labels, other_slots if other_slots and period_labels else every)
    return out


def _fiscal_year_of(filing: Filing, day: dt.date) -> int:
    """Fiscal year of a column's period end, counted in whole years back from
    the filing's own period end. 52/53-week calendars move a year end by a
    few days, which the rounding absorbs."""
    own = dt.date.fromisoformat(filing.period_end)
    return filing.fiscal_year - round((own - day).days / 365.25)


def _iso_date(year: int | None, month: int | None, day: int | None) -> str | None:
    if year is None or month is None:
        return None
    if day is None:
        # "As of December" over a year row: the month end is the only date
        # the header supports.
        day = calendar.monthrange(year, month)[1]
    try:
        return dt.date(year, month, day).isoformat()
    except ValueError:
        return None


PAREN_RE = re.compile(r"\([^()]*\)")
GLUED_ENDED_RE = re.compile(r"(?i)(months?|years?|quarters?)(ended)")
GLUED_MONTH_RE = re.compile(r"(?<=[a-z])(?=(?:" + _MONTH_ALTS_TITLE + r")(?![a-z]))")


def _label_text(text: str) -> str:
    """A column label as a reader should see it: unit parentheticals out
    ("(in millions, except ratios)" is the table's unit line, not part of
    the column), dropped line breaks restored ("monthsended September",
    "EndedSeptember"), whitespace collapsed, trailing comma gone."""
    text = PAREN_RE.sub(lambda m: " " if parse_units(m.group(0)) else m.group(0), text)
    text = GLUED_ENDED_RE.sub(r"\1 \2", text)
    text = GLUED_MONTH_RE.sub(" ", text)
    return " ".join(text.split()).rstrip(",")


def build_column(index: int, cell: Cell, uppers: list[Cell], lowers: list[Cell],
                 filing: Filing) -> Column:
    """One Column from a column-row cell, the labels stacked above it and
    the single-cell rows below it that apply to every column.

    The nearest source wins: the cell itself, then the header rows from the
    one just above it upward, then the rows below. A quarter name takes its
    month from the quarter number and never from a date above it, and only
    for a filer whose fiscal year ends in December, because "Third" under
    "2025 Quarters" is September 30 for a calendar filer and unknown for
    any other. A bare year under no duration in an annual report is read
    as a fiscal year; in a quarterly report it stays unknown. A change
    column ("Change", "2025 vs. 2024") never carries a period, so a figure
    quoted from it cannot pass as a period figure.
    """
    period_uppers = [u for u in uppers if u.kind in PERIOD_KINDS]
    other_uppers = [u for u in uppers if u.kind not in PERIOD_KINDS]
    period_lowers = [l for l in lowers if l.kind in PERIOD_KINDS]
    if cell.kind == "change":
        label = " ".join([u.text for u in other_uppers] + [cell.text])
        return Column(index=index, label=_label_text(label), period_end=None, duration=None)

    sources = [cell] + list(reversed(uppers)) + list(lowers)
    duration = next((s.duration for s in sources if s.duration), None)
    if duration is None and any(s.point for s in sources):
        duration = "point_in_time"
    year = next((s.year for s in sources if s.year), None)
    if cell.kind == "quarter":
        month, day = None, None
        if year is not None and cell.quarter and filing.fye_month == 12:
            month = 3 * cell.quarter
            day = calendar.monthrange(year, month)[1]
    else:
        dated = next((s for s in sources if s.month), None)
        month, day = (dated.month, dated.day) if dated else (None, None)
    period_end = _iso_date(year, month, day)
    if period_end is None and year is not None and month is None and cell.kind != "quarter":
        if duration is None and filing.form == "10-K":
            duration = "fiscal_year"
        if duration in ("fiscal_year", "twelve_months") and year == filing.fiscal_year:
            period_end = filing.period_end

    if duration == "fiscal_year" and (period_end or year):
        if period_end:
            fy = _fiscal_year_of(filing, dt.date.fromisoformat(period_end))
            period_text = "FY%d (%s)" % (fy, period_end)
        else:
            period_text = "FY%d" % year
    else:
        parts = [u.text for u in period_uppers]
        if cell.kind in PERIOD_KINDS:
            parts.append(cell.text)
        parts.extend(l.text for l in period_lowers)
        period_text = " ".join(parts)
    dim_parts = [u.text for u in other_uppers]
    if cell.kind not in PERIOD_KINDS:
        dim_parts.append(cell.text)
    dim_text = " ".join(dim_parts)
    if period_lowers and not period_uppers and cell.kind not in PERIOD_KINDS:
        # The period sits under the dimension header, so the label reads
        # the way the table does: "Graphics FY2025 (2025-01-26)".
        label = dim_text + " " + period_text
    else:
        label = period_text + " " + dim_text
    return Column(index=index, label=_label_text(label), period_end=period_end, duration=duration)


def data_shape(rows: list[Row]) -> int | None:
    """The modal number of value cells in the labeled data rows, or None
    when no row has both a label and a number.

    A value cell is a number or a placeholder standing in a numeric column
    ("-", "NM"). A row's label is its first cell when that cell is not a
    value; a row that opens with a value is a continuation and does not
    vote. Ties go to the wider count, because the narrow rows of a table
    are its sparse ones (a segment line with two of four cells filled).
    This is the count a column list has to match before a chunk can be
    read by position: empty cells are dropped when a row is formatted, so
    in a row whose value count differs from the column count the k-th
    value is under no known column.
    """
    counts: list[int] = []
    for row in rows:
        if row.kind != "data" or not row.cells:
            continue
        first = row.cells[0]
        if first.kind == "number" or PLACEHOLDER_RE.match(first.text):
            continue
        values = [c for c in row.cells[1:] if c.kind == "number" or PLACEHOLDER_RE.match(c.text)]
        if not any(c.kind == "number" for c in values):
            continue
        counts.append(len(values))
    if not counts:
        return None
    tally = Counter(counts)
    return max(tally, key=lambda count: (tally[count], count))


def shape_of_text(text: str) -> int | None:
    """`data_shape` over the lines of a chunk's text, for a check that only
    holds the chunk. The prefix rows read back as header rows and do not
    vote."""
    lines = text.split("\n")
    return data_shape([read_row(Line(0, len(line), line)) for line in lines])


def with_change_columns(uppers: list[Cell], columns: list[Cell], shape: int) -> list[Cell] | None:
    """The column cells plus the change columns the row above announces.

    "% Change" has no cell of its own in the date row under it ("Quarter
    Ended | % Change | Nine Months Ended | % Change" over "July 2, 2022 |
    July 3, 2021 | July 2, 2022 | July 3, 2021"), yet every data row holds
    a change value after each pair of dates. When the row above holds only
    period and change labels, and the date cells plus the change labels
    count up to the value count, a change column is placed after the group
    its label follows. None when the counts do not work out.
    """
    changes = [c for c in uppers if c.kind == "change"]
    periods = [c for c in uppers if c.kind in PERIOD_KINDS]
    if not changes or len(changes) + len(periods) != len(uppers) or len(columns) + len(changes) != shape:
        return None
    if not periods:
        return columns + [Cell("", "change", None, None, None, None, False) for _ in changes]
    if len(columns) % len(periods):
        return None
    group = len(columns) // len(periods)
    out: list[Cell] = []
    pos = 0
    for label in uppers:
        if label.kind == "change":
            out.append(Cell("", "change", None, None, None, None, False))
        else:
            out.extend(columns[pos:pos + group])
            pos += group
    return out


def parse_columns(headers: list[Row], data: list[Row], filing: Filing) -> tuple[list[Column], str | None]:
    """(columns, column_source) from the header rows, or ([], None) when no
    row carries a date.

    The column row is the last header row with a title cell whose column
    cell count equals the data rows' value-cell count (`data_shape`); the
    rows above it are spread over it by `distribute`, and the rows below
    it too, which is how a single-cell period row under a segment header
    ("Year Ended Jan 26, 2025" under "Compute & Networking | Graphics |
    All Other | Consolidated") gives every segment column its period. Such
    a match is "parsed", as is one dated cell over rows of several values.
    When no header row matches the shape, the last dated row (or a wider
    row below it) still yields columns for display, stamped
    "unverified_shape" so a resolver never reads a figure's column from
    them.
    """
    readings = [label_cells(r) for r in headers]
    dated = [k for k, (lead, cells) in enumerate(readings) if any(_dated(c) for c in _labels_of(lead, cells))]
    if not dated:
        return [], None
    shape = data_shape(data)

    def matches(reads: list[tuple[Cell | None, list[Cell]]]) -> list[int]:
        return [k for k, (lead, cells) in enumerate(reads)
                if any(c.kind != "unit" for c in cells) and shape is not None and len(cells) == shape]

    matching = matches(readings)
    if not matching:
        # The other reading of a filled first cell: "Year ended December
        # 31, | Amount | Percent" over two-value rows is a caption over two
        # columns, which the rule in label_cells cannot tell from a third
        # column; the value count can. A dated first cell is read that way
        # only beside names, never beside other dates.
        alternate = [(cells[0], cells[1:]) if lead is None and r.first_filled and len(cells) > 1
                     and (not _dated(cells[0]) or not any(_dated(c) for c in cells[1:]))
                     else (lead, cells)
                     for r, (lead, cells) in zip(headers, readings)]
        matching = matches(alternate)
        if matching:
            readings = alternate
    if not matching and shape is not None:
        for k in reversed(dated):
            if k == 0:
                continue
            cells = with_change_columns(_labels_of(*readings[k - 1]), readings[k][1], shape)
            if cells is not None:
                readings[k] = (readings[k][0], cells)
                matching = [k]
                break
    titled = [k for k, (lead, cells) in enumerate(readings) if any(c.kind != "unit" for c in cells)]
    if matching:
        col, source = matching[-1], "parsed"
    elif (shape is not None and shape >= 2 and len(titled) == 1 and len(readings[titled[0]][1]) == 1
          and readings[titled[0]][1][0].kind in ("date", "period") and readings[titled[0]][1][0].month):
        # One dated cell over rows of several values ("December 31, 2024"
        # over "Spot rates | 4.50% | 4.49% | 4.07%"): the names of the
        # columns are lost with the block above, the period is not, and
        # the period is what a resolver checks. A bare year is left out
        # because such a row often heads four quarters.
        cell = readings[titled[0]][1][0]
        return [build_column(i, cell, [], [], filing) for i in range(shape)], "parsed"
    else:
        col, source = dated[-1], "unverified_shape"
        for k in range(col + 1, len(readings)):
            if len(readings[k][1]) > len(readings[col][1]):
                col = k
    lead, columns = readings[col]
    above = [distribute(_labels_of(*readings[k]), columns) for k in range(col)]
    if lead is not None and lead.kind in PERIOD_KINDS:
        above.append(distribute([lead], columns))
    below = [distribute(_labels_of(*readings[k]), columns) for k in range(col + 1, len(readings))]
    out = []
    for i, cell in enumerate(columns):
        uppers = [c for layer in above for c in layer[i]]
        lowers = [c for layer in below for c in layer[i]]
        out.append(build_column(i, cell, uppers, lowers, filing))
    if not any(c.period_end or c.duration for c in out):
        return [], None
    return out, source


def _lone_dated(row: Row) -> bool:
    """A row whose one cell names a period: "Year Ended Jan 28, 2024 | | |"
    or "December 31, 2024:"."""
    return row.kind in ("header", "sub") and len(row.cells) == 1 and _dated(row.cells[0])


def period_groups(headers: list[Row], data: list[Row]) -> list[tuple[list[Row], list[Row], Row | None]]:
    """(header rows, data rows, opening row) per period group of one table.

    A segment table prints its dimension header once and then a period
    sub-heading before each block of rows ("Year Ended Jan 26, 2025 | | |"
    ... "Year Ended Jan 28, 2024 | | |"). The first such row is a header
    row and gives every column its period; the later ones sit among the
    data and would leave their rows under the first period's columns. So
    when the header rows hold a single-cell dated row, each later
    single-cell dated row opens a group whose header repeats the other
    header rows with that row in the dated row's place. A table without
    such a header row is one group.
    """
    lone = [k for k, r in enumerate(headers) if _lone_dated(r)]
    if not lone:
        return [(headers, data, None)]
    slot = lone[-1]
    groups: list[tuple[list[Row], list[Row], Row | None]] = []
    cur_headers, cur_data, opener = headers, [], None
    for row in data:
        if _lone_dated(row):
            if cur_data:
                groups.append((cur_headers, cur_data, opener))
            cur_headers = headers[:slot] + [row] + headers[slot + 1:]
            cur_data, opener = [], row
        else:
            cur_data.append(row)
    if cur_data:
        groups.append((cur_headers, cur_data, opener))
    return groups or [(headers, data, None)]


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


@dataclass
class Piece:
    """A chunk before it is stamped with filing metadata."""

    kind: str
    text: str
    char_start: int
    char_end: int
    caption: str | None = None
    units: str | None = None
    units_source: str | None = None
    columns: list[Column] | None = None
    column_source: str | None = None


def pack_table(rows: list[Line], prose: list[Line], filing: Filing,
               inherited: str | None) -> tuple[list[Piece], list[Line], str | None]:
    """Chunks for one table block.

    Returns (pieces, lines handed back to prose, units the block declared
    or None). A table of contents yields nothing; the check runs on the
    whole block first because page references such as "K-71" are not
    numbers and such a block has no data rows. An exhibit index or a block
    with no row carrying a number is handed back whole so it packs as
    prose. Otherwise each table in the block is judged on its own: a table
    of contents fused ahead of a statement yields nothing, a table too small
    to stand is handed back to prose, and the rest are packed. The caption
    lines above the block are taken off the end of `prose` for the first
    table only; when an index precedes a statement, the lines above the
    index are not the statement's title.
    """
    read = [read_row(l) for l in rows]
    if is_toc(read):
        return [], [], None
    tables = [] if is_exhibit_list(read) else split_fused(read)
    if not tables:
        return [], rows, None
    pieces: list[Piece] = []
    leftover: list[Line] = []
    declared: str | None = None
    for k, table in enumerate(tables):
        if is_toc(table):
            continue
        if not table_stands(table):
            leftover.extend(r.line for r in table)
            continue
        caption_lines = find_captions(prose, rows[0].start) if k == 0 else []
        new_pieces, units = _pack_one_table(table, caption_lines, filing, declared or inherited)
        pieces.extend(new_pieces)
        declared = units or declared
    return pieces, leftover, declared


def table_stands(rows: list[Row]) -> bool:
    """A table worth a chunk of its own: a header or caption row over at
    least one data row, or three data rows. Anything smaller is a stray
    footer or page-index row that packs as prose."""
    captions, headers, data = classify_rows(rows)
    n_data = sum(1 for r in data if r.kind == "data")
    return n_data >= 3 or (n_data >= 1 and bool(captions or headers))


def _pack_one_table(read: list[Row], caption_lines: list[Line], filing: Filing,
                    inherited: str | None) -> tuple[list[Piece], str | None]:
    """Chunks for one table; every chunk starts with the caption lines and
    the header rows again. Returns (pieces, units declared here or None).

    Columns are parsed once per period group, then every chunk is checked
    against its own rows: a chunk whose value-cell count differs from the
    column count is stamped "unverified_shape" even when the table as a
    whole matched, because those rows are the ones a resolver would read
    by position.
    """
    caption_rows, header_rows, data_rows = classify_rows(read)

    units, units_source = None, None
    for text in [r.text for r in header_rows + caption_rows] + [l.text for l in caption_lines]:
        units = parse_units(text)
        if units:
            units_source = "declared"
            break
    if units is None and inherited:
        units, units_source = inherited, "inherited"

    caption = None
    for text in [l.text for l in reversed(caption_lines)] + [r.text for r in caption_rows]:
        caption = short_caption(text)
        if caption:
            break

    caption_texts = [l.text for l in caption_lines]
    block_start = caption_lines[0].start if caption_lines else read[0].line.start
    pieces: list[Piece] = []

    def flush(cur: list[Row], prefix: str, columns: list[Column], source: str | None,
              start: int) -> None:
        text = prefix + "\n".join(r.text for r in cur)
        here = source
        if source == "parsed" and data_shape(cur) != len(columns):
            here = "unverified_shape"
        pieces.append(Piece("table", text, start, cur[-1].line.end, caption,
                            units, units_source, columns, here))

    for group_headers, group_rows, opener in period_groups(header_rows, data_rows):
        columns, source = parse_columns(group_headers, group_rows, filing)
        prefix_lines = caption_texts + [r.text for r in caption_rows + group_headers]
        prefix = "\n".join(prefix_lines) + ("\n" if prefix_lines else "")
        first_start = block_start if not pieces else (opener.line.start if opener else None)
        cur: list[Row] = []
        size = len(prefix)
        for row in group_rows:
            if cur and size + len(row.text) + 1 > SOFT_MAX:
                flush(cur, prefix, columns, source, first_start or cur[0].line.start)
                first_start = None
                cur, size = [], len(prefix)
            cur.append(row)
            size += len(row.text) + 1
        if cur:
            flush(cur, prefix, columns, source, first_start or cur[0].line.start)
    return pieces, units if units_source == "declared" else None


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Sentence:
    start: int
    end: int
    text: str
    new_line: bool


def _sentences(lines: list[Line]) -> list[Sentence]:
    out = []
    for line in lines:
        pos = 0
        first = True
        for part in SENTENCE_RE.split(line.text):
            at = line.text.find(part, pos)
            pos = at + len(part)
            if not part.strip():
                continue
            out.append(Sentence(line.start + at, line.start + at + len(part), part, first))
            first = False
    return out


def _hard_split(s: Sentence) -> list[Sentence]:
    """Cut one overlong sentence at word boundaries into pieces under
    SOFT_MAX. Keeping every sentence under SOFT_MAX is what lets a chunk
    always reach PROSE_MIN before its next sentence can overflow HARD_MAX."""
    pieces: list[Sentence] = []
    pos = 0
    text = s.text
    while len(text) - pos > SOFT_MAX:
        cut = text.rfind(" ", pos, pos + SOFT_MAX)
        if cut <= pos:
            cut = pos + SOFT_MAX
        pieces.append(Sentence(s.start + pos, s.start + cut, text[pos:cut], s.new_line and not pieces))
        pos = cut + 1 if text[cut:cut + 1] == " " else cut
    pieces.append(Sentence(s.start + pos, s.end, text[pos:], s.new_line and not pieces))
    return pieces


def _join(sentences: list[Sentence]) -> str:
    out = []
    for k, s in enumerate(sentences):
        if k:
            out.append("\n" if s.new_line else " ")
        out.append(s.text)
    return "".join(out)


def pack_prose(lines: list[Line], kind: str) -> list[Piece]:
    """Sentence-packed chunks of about SOFT_MAX chars.

    A chunk grows past SOFT_MAX only while it is under PROSE_MIN and the
    next sentence still fits under HARD_MAX. A short tail merges into the
    chunk before it when the result stays under HARD_MAX, otherwise it takes
    sentences back from that chunk until it reaches PROSE_MIN. Each chunk
    after the first then opens with the previous chunk's last sentence when
    that sentence is under OVERLAP_MAX chars and leaves room, so a figure and
    the sentence introducing it are not separated by a chunk boundary.
    """
    sentences: list[Sentence] = []
    for s in _sentences(lines):
        sentences.extend(_hard_split(s) if len(s.text) > SOFT_MAX else [s])
    groups: list[list[Sentence]] = []
    cur: list[Sentence] = []
    size = 0
    for s in sentences:
        grow = len(s.text) + (1 if cur else 0)
        if cur and size + grow > SOFT_MAX and not (size < PROSE_MIN and size + grow <= HARD_MAX):
            groups.append(cur)
            cur, size, grow = [], 0, len(s.text)
        cur.append(s)
        size += grow
    if cur:
        groups.append(cur)
    if len(groups) >= 2 and len(_join(groups[-1])) < PROSE_MIN:
        prev, tail = groups[-2], groups[-1]
        if len(_join(prev + tail)) <= HARD_MAX:
            prev.extend(tail)
            groups.pop()
        else:
            while len(_join(tail)) < PROSE_MIN and len(_join(prev[:-1])) >= PROSE_MIN:
                tail.insert(0, prev.pop())
    pieces = []
    for k, group in enumerate(groups):
        text, start = _join(group), group[0].start
        if k:
            last = groups[k - 1][-1]
            if len(last.text) < OVERLAP_MAX and len(last.text) + 1 + len(text) <= HARD_MAX:
                text = last.text + ("\n" if group[0].new_line else " ") + text
                start = last.start
        pieces.append(Piece(kind, text, start, group[-1].end))
    return pieces


# ---------------------------------------------------------------------------
# Filing to chunks
# ---------------------------------------------------------------------------


def header_line(filing: Filing, section: Section, note: str | None, caption: str | None,
                units: str | None, columns: list[Column]) -> str:
    """The metadata line embedded at the top of every chunk."""
    cik = filing.cik.lstrip("0") or "0"
    period = "fiscal year ended" if filing.form == "10-K" else "quarter ended"
    if section.item == "COVER":
        where = "Cover"
    elif filing.form == "10-Q":
        where = "Part %s Item %s %s" % (section.part, section.item.split(".")[-1], section.title)
    else:
        where = "Item %s %s" % (section.item, section.title)
    if note:
        where += " > " + note
    if caption:
        where += " > " + caption
    parts = [
        "%s (%s, CIK %s)" % (filing.company, filing.ticker, cik),
        "%s %s, %s %s, filed %s" % (filing.form, filing.fiscal_label, period,
                                     filing.period_end, filing.filing_date),
        where,
    ]
    if units:
        parts.append("units: " + units)
    if columns:
        parts.append("columns: " + ", ".join(c.label for c in columns))
    return " | ".join(parts)


def _note_title(notes: list[Note], pos: int) -> str | None:
    for note in notes:
        if note.start <= pos < note.end:
            if note.number is not None:
                return "Note %d - %s" % (note.number, note.title)
            return note.title
    return None


def section_pieces(filing: Filing, section: Section) -> list[Piece]:
    """All pieces of one section in reading order.

    Prose runs and table blocks alternate. A table's caption lines leave the
    prose run they came from, and units declared by a table are inherited by
    later tables in the same section that declare none. The cover section
    goes through the same path because a filer whose item headings the
    parser could not place leaves its whole body there; its prose gets the
    kind "cover" and its tables stay tables.
    """
    lines = _lines(filing.body, section.start, section.end)
    prose_kind = "cover" if section.item == "COVER" else "prose"
    pieces: list[Piece] = []
    pending: list[Line] = []
    last_units: str | None = None
    for kind, block in split_blocks(lines):
        if kind == "prose":
            pending.extend(block)
            continue
        table_pieces, leftover, declared = pack_table(block, pending, filing, last_units)
        if table_pieces:
            if pending:
                pieces.extend(pack_prose(pending, prose_kind))
            pending = list(leftover)
            pieces.extend(table_pieces)
            if declared:
                last_units = declared
        else:
            pending.extend(leftover)
    if pending:
        pieces.extend(pack_prose(pending, prose_kind))
    return pieces


def chunk_filing(filing: Filing) -> list[Chunk]:
    """Every chunk of a filing, in body order, numbered by `seq`.

    Prose under CHUNK_MIN chars is dropped: a fragment that short is a stray
    line between tables. A table that `table_stands` accepted is kept at any
    length, because a header over three data rows is exactly what a figure
    question needs.
    """
    chunks: list[Chunk] = []
    for section in filing.sections:
        for piece in section_pieces(filing, section):
            if piece.kind != "table" and len(piece.text) < CHUNK_MIN:
                continue
            note = _note_title(section.notes, piece.char_start)
            columns = piece.columns or []
            # The header line is text the model quotes from, so only columns
            # whose shape matched the rows are named there; the rest stay in
            # the record for display and the raw header rows are in the text.
            shown = columns if piece.column_source == "parsed" else []
            header = header_line(filing, section, note, piece.caption, piece.units, shown)
            chunk_id = hashlib.sha1(
                ("%s:%d:%d" % (filing.file, piece.char_start, piece.char_end)).encode()
            ).hexdigest()[:16]
            chunks.append(Chunk(
                chunk_id=chunk_id,
                file=filing.file,
                cik=filing.cik,
                ticker=filing.ticker,
                company=filing.company,
                form=filing.form,
                part=section.part,
                item=section.item,
                item_title=section.title,
                note_title=note,
                kind=piece.kind,
                period_end=filing.period_end,
                fiscal_year=filing.fiscal_year,
                fiscal_quarter=filing.fiscal_quarter,
                fiscal_label=filing.fiscal_label,
                filing_date=filing.filing_date,
                seq=len(chunks),
                char_start=piece.char_start,
                char_end=piece.char_end,
                units=piece.units,
                units_source=piece.units_source,
                columns=columns,
                column_source=piece.column_source,
                header=header,
                text=piece.text,
                n_tokens=estimate_tokens(header + "\n" + piece.text),
            ))
    return chunks
