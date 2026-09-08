"""Check an Answer against the excerpts it cites.

The model writes claims with a quote and citations; nothing in the Answer
says whether the quote is in the excerpt, whether the figures in the claim
text are in the source, or which row and column of a table a figure came
from. This module establishes each of those from the chunks themselves and
reports what it found as counts plus a flag list. There is no score and
no aggregate: a reader sees "quotes found 4/5" and the one flag, and
decides.

Every comparison runs on corpus.normalize() output on both sides, with
pipe characters and whitespace runs collapsed to single spaces and the
text casefolded, so a curly quote, a no-break space or a table cell
boundary can never break a match.

The governing rule: a check never claims more than it establishes. A
figure is credited only against text that was itself located in a chunk,
never against the quote string the model wrote; a number in the source
is matched by its type ($, %, sign, scale) and never by digits alone; a
column is matched only when its period, duration, and the words of its
label and of its row agree with the claim. When the evidence is short of
that the flag is column_unverified or figure_not_in_quote, never a match.

Where each decision is made, per claim:
  locate_quote        the quote's chunk: exact substring of a cited chunk,
                      of a cited chunk joined to its neighbour (a quote can
                      straddle a boundary), or of the neighbour alone (noted
                      as quote_in_neighbour); else the closest passage (long
                      common substring or 90% of the tokens); else missing
  figures_in          which number tokens of the claim text are figures
                      (years, dates, small counts, note and excerpt numbers,
                      week counts and citation ids are not)
  find_figure         the figure among the typed number tokens of the
                      located passage, then of the chunk text, with
                      scale-aware equivalence against the row's units
  section_of          the label-only line a table row sits under ("Per
                      common share information:", "Capital ratios:"),
                      read from the chunk text since a quoted row arrives
                      without it; it is part of the row's label for every
                      scale and type decision
  row_scale           the scale a row is stated in, from a unit line the
                      chunk declares itself: the line's scale, the one its
                      "except ... shares ... thousands" clause gives that
                      row, "unscaled" for a per-share row, or unknown for
                      a ratio row, an inherited unit line, or a clause
                      that names the row without a scale word
  column_of_figure    the table column a figure sits in, by cell position
  period_matches      the column's period against the claim's period_end
  row_reading         whether the claim names the figure's row, or reads
                      as another row of the same table
  column_reading      whether the claim names the column's entity, basis
                      or segment, or reads as a sibling column's; words
                      of the figure's own row label never count for a
                      sibling, so "ratio" in the claim cannot pick a
                      "Regulatory Minimum ratios" column for a ratio row
Per summary sentence and table cell: the claim ids it names must exist,
and every figure it states must be a figure of one of those claims.
Per gap and not_comparable note: nothing named outside the coverage block.

Column provenance is trusted only when the chunker stamped column_source
"parsed": that is the one state where the header row's cell count matched
the data rows, so a cell position maps to a Column. "unverified_shape"
(labels kept for display after a count mismatch) and None (no dated
header) give column_unverified, never a match and never a mismatch.

Known limits, stated so nobody reads more into a pass than it holds: a
percent sign is printed once per column in most statements, so a percent
claim against a bare cell is judged by the column (a change column or a
ratio row carries it; a money column does not); a bare four-digit number
without a comma in the source is read as a year, so a filer that prints
"1935" for 1,935 is not matched; a row is named when the claim carries
any of its label words, so a claim that uses an abbreviation the table
does not ("NII") is unverified; a comparative figure is accepted from an
earlier column only when a comparative phrase attaches to that figure
("up from $X", "$X a year earlier", "$X in Q3 2024" with the column's
year) and another figure of the claim anchors the period, so the same
sentence with its two figures swapped is a column mismatch; the
direction of a comparison ("up from" beside a higher figure) is not
checked; a column that carries a date and no duration (a capital or
balance sheet column) matches only a point-in-time claim and is
unverified for a claim stated as a quarter or a year; a claim's scale
word is checked only against a scale the chunk's own unit line
establishes for that row, and is reported as units_unverified when the
unit line was inherited from an earlier chunk, names the row without a
scale word, or is absent; a section line that carries a ratio word makes
every row under it a ratio row, so a money row filed under "Selected
ratios and metrics:" is not matched to a money claim; gaps are checked
for years, registry tickers and registry company names, so a company
the registry does not hold is not caught.
"""

import calendar
import datetime as dt
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

from chunk import format_cells
from corpus import normalize
from models import Answer, Chunk, Claim, Column, Context, EvidenceChecks, Plan

MAX_QUOTE_WORDS = 40
# A quote shorter than this locates nothing: "23,966" is in every table
# that holds the number, so finding it says nothing about the source.
MIN_QUOTE_TOKENS = 3
# An approximate quote: this many characters in common, or this share of
# the quote's tokens present in the chunk.
LCS_MIN_CHARS = 60
TOKEN_SHARE_MIN = 0.9
# Integers up to this are counts in prose ("three companies", "12 months")
# and are not checked as figures.
SMALL_INT_MAX = 12
# A 52/53-week filer closes its quarter in the last week of the month, so
# a claim naming the month end may sit this many days after the column's
# own date and no more.
MONTH_END_SLACK_DAYS = 6

SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
SCALE_WORDS = r"thousand|million|billion|trillion"
# Claim period_kind to the chunker's column durations.
DURATIONS_FOR = {
    "quarter": ("three_months",),
    "nine_months": ("nine_months",),
    "fiscal_year": ("fiscal_year", "twelve_months"),
    "point_in_time": ("point_in_time",),
}

MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|december"
          "|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")
# Dates are periods, and the period check is the column check; the day
# and year inside a date are masked before numbers are read so "30" in
# "September 30, 2025", "30 September 2025" or "9/30/2025" is never
# looked for in a table row. Identifiers (a CIK, a note or item number,
# a form name, a page) and counts ("52-week") are masked the same way.
DATE_ISO_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
DATE_TEXT_RE = re.compile(r"\b(?:%s)\.?\s*\d{1,2}(?:\s*,?\s*\d{4})?\b" % MONTHS, re.I)
DATE_DAY_FIRST_RE = re.compile(r"\b\d{1,2}\s+(?:%s)\.?(?:\s*,?\s*\d{4})?\b" % MONTHS, re.I)
DATE_SLASH_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
ID_RE = re.compile(
    r"\b(?:cik|notes?|items?|excerpts?|citations?|claims?|tables?|exhibits?|schedules?|pages?|parts?|footnotes?"
    r"|lines?|sections?|levels?|tiers?|q|fy|fiscal)"
    r"\s*#?\s*\d+[a-z]?\b|\b\d+-[kq]\b", re.I)
COUNT_RE = re.compile(r"\b\d{1,3}[- ](?:week|day|month|year)s?\b|\b\d+(?:st|nd|rd|th)\b", re.I)
# A figure: optional $ and opening parenthesis or minus, digits with commas
# and a decimal part, optional closing parenthesis and %, optional scale
# word. The digit run ends on a digit so the comma after "2025," in prose
# stays punctuation. The lookbehind keeps "C12" and "K3" (citation and
# claim ids) and "CET1" out.
DIGITS = r"\d(?:[\d,]*\d)?(?:\.\d+)?"
FIGURE_RE = re.compile(
    r"(?<![A-Za-z\d.])(\$)?\s*(\()?(-)?(" + DIGITS + r")(\))?\s*(%)?(?:\s*(" + SCALE_WORDS + r")s?\b)?", re.I)
YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
CELL_RE = re.compile(r"^(\$)?\s*(\()?(-)?(\d[\d,]*(?:\.\d+)?)(\))?\s*(%)?$")
SCALE_WORD_RE = re.compile(r"\b(" + SCALE_WORDS + r")s?\b", re.I)
FY_LABEL_RE = re.compile(r"\bFY(\d{4})\b")
GAP_YEAR_RE = re.compile(r"(?<![\d-])(?:19|20)\d{2}(?![\d-])")
GAP_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
FOOTNOTE_MARK_RE = re.compile(r"\(\w{1,2}\)")
WORD_RE = re.compile(r"[a-z&][a-z&.']*")
# Words of a column label that describe its period, so what is left names
# its entity, basis or segment.
PERIOD_WORDS = set(MONTHS.split("|")) | {
    "ended", "ending", "end", "month", "months", "three", "six", "nine", "twelve", "year", "years",
    "fiscal", "fy", "quarter", "quarters", "as", "of", "at", "the", "and", "for", "first", "second",
    "third", "fourth", "period", "periods"}
STOP_WORDS = {
    "the", "of", "and", "for", "in", "at", "as", "a", "an", "to", "by", "on", "per", "with", "from", "or",
    "its", "it", "was", "were", "is", "are", "be", "been", "has", "had", "have", "than", "that", "this",
    "which", "used"}
# A row stated in ratio terms holds no money scale, whatever the unit line
# says, so "$15.8 million" can never match a "Tier 1 capital ratio" cell.
RATIO_WORDS = {"ratio", "rate", "margin", "yield", "percent", "percentage", "return", "roe", "rotce",
               "coverage", "mix", "leverage", "efficiency", "basis"}
EXCEPT_STOP = {"number", "of", "which", "are", "is", "reflected", "in", "data", "amount", "amounts", "the",
               "for", "and", "otherwise", "noted", "where", "as", "indicated", "per", "except", "stated",
               "presented", "expressed"}
# Wording that makes an earlier-period figure a comparison in the claim's
# own sentence. A prefix phrase stands before the figure it excuses ("up
# from $23,405 million"); a postfix phrase stands after it ("$23,405
# million a year earlier"). Which figure a phrase attaches to decides the
# check, so the same sentence with its figures swapped is not excused.
PREFIX_COMPARATIVE_RE = re.compile(
    r"\b(?:(?:up|down|increased?|decreased?|rose|fell|grew|declined|improved|higher|lower|changed?|rising|falling)"
    r"\s+from|compared\s+(?:with|to)|versus|vs\.?|(?:from|than|against)\s+(?:the\s+|a\s+)?"
    r"(?:prior|previous|year[- ]earlier|year[- ]ago|same)[- ](?:year|period|quarter)(?:'s)?)\b", re.I)
POSTFIX_COMPARATIVE_RE = re.compile(
    r"\b(?:a\s+year\s+(?:earlier|ago)|(?:in|for|of|during)\s+(?:the\s+)?(?:prior|previous|year[- ]earlier"
    r"|year[- ]ago|same|comparable)[- ](?:year|period|quarter)|year[- ]over[- ]year|respectively)\b", re.I)
# A table row's section line: a label-only line above it ("Per common
# share information:", "Capital ratios(a)(b):"), read as part of the
# row's label. A unit line ("(in millions, except ratios)") is never a
# section. A quoted row fragment must be this long before it is looked
# up among the chunk's lines, so "1.06" alone finds no section.
SECTION_MAX_WORDS = 6
MIN_FRAGMENT_CHARS = 12
PER_SHARE_RE = re.compile(r"\bper\s+(?:\w+\s+){0,2}share\b")
SHARE_COUNT_RE = re.compile(r"\b(?:shares?\s+(?:used|outstanding|issued)|number\s+of\s+shares|weighted[- ]average"
                            r"|average\s+(?:\w+\s+){0,3}shares)\b")
# The scale of a per-share row: plain dollars, no scale word. Distinct
# from None, which is a scale the chunk does not establish.
UNSCALED = "unscaled"
# A column word the claim may name by another word, used only when no
# sibling column carries the other word itself.
COLUMN_SYNONYMS = {"consolidated": ("total", "overall", "firmwide", "consolidated")}
# A table cell that states the absence of a figure needs no claim.
NO_DATA_RE = re.compile(r"^(?:not\s+(?:disclosed|stated|reported|available|comparable|applicable|given)|n/?a"
                        r"|none|-+)\.?$", re.I)
DECREASE_RE = re.compile(
    r"\b(?:down|decline[ds]?|decrease[ds]?|fell|fall|lower|loss|losses|negative|reduc\w+|shrank|drop\w*"
    r"|contract\w+|worsen\w*)\b", re.I)


def canon(text: str) -> str:
    return " ".join(normalize(text).replace("|", " ").split()).casefold()


def mask_non_figures(text: str) -> str:
    """The text with dates, identifiers and counts blanked, so the number
    grammar sees only amounts. Each mask is the length of what it hides,
    so a figure's offsets still index the unmasked text."""
    for pattern in (DATE_ISO_RE, DATE_SLASH_RE, DATE_TEXT_RE, DATE_DAY_FIRST_RE, ID_RE, COUNT_RE):
        text = pattern.sub(lambda m: " " * len(m.group(0)), text)
    return text


def words_of(text: str) -> set[str]:
    """Content words for naming a row or a column: casefolded, possessive
    and trailing punctuation dropped, plurals trimmed, stop words out."""
    out = set()
    for w in WORD_RE.findall(canon(text)):
        w = w[:-2] if w.endswith("'s") else w
        w = w.rstrip(".,'")
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if w and w not in STOP_WORDS:
            out.add(w)
    return out


def naming_words(text: str) -> set[str]:
    """The words a claim can name a row or column with: its own words and
    the same text with hyphens closed up, so "non-interest" names a
    "Noninterest" row."""
    return words_of(text) | words_of(text.replace("-", ""))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


@dataclass
class Figure:
    """One number the claim text states, as written and as a value."""

    text: str
    value: float
    scale: str | None
    percent: bool
    dollar: bool
    negative: bool
    # Significant digits shown: the precision a converted source value is
    # rounded to before the comparison.
    sig: int
    # Offsets in the normalized claim text, so a comparative phrase can be
    # tied to the figure it stands beside.
    start: int = 0
    end: int = 0


@dataclass
class Token:
    """One number in source text, with the type marks printed around it.
    `position` is the value-cell index when the token came from a table
    row, so its column can be read."""

    text: str
    value: float
    dollar: bool
    percent: bool
    negative: bool
    scale: str | None
    position: int | None = None


@dataclass
class Hit:
    """Where a claim figure was found: the token, the line it sits in, that
    line's row label and section line, the scale the line is stated in
    (None when the chunk does not establish it), whether the match needed
    a unit conversion, and whether the source is negative where the claim
    reads positive."""

    token: Token
    line: str
    label: str
    section: str
    scale: str | None
    converted: bool
    sign_clash: bool


def figure_of(groups: tuple, text: str, span: tuple[int, int] = (0, 0)) -> Figure:
    dollar, opened, minus, digits, closed, percent, scale = groups
    sig = len(digits.replace(",", "").replace(".", "").lstrip("0")) or 1
    return Figure(text=text.strip(), value=float(digits.replace(",", "")),
                  scale=scale.lower() if scale else None, percent=bool(percent), dollar=bool(dollar),
                  negative=bool(minus) or bool(opened and closed), sig=sig, start=span[0], end=span[1])


def is_bare(groups: tuple) -> bool:
    dollar, opened, minus, digits, closed, percent, scale = groups
    return not dollar and not percent and not scale and not opened and not minus and "." not in digits \
        and "," not in digits


def figures_in(text: str) -> list[Figure]:
    masked = mask_non_figures(normalize(text))
    out = []
    for m in FIGURE_RE.finditer(masked):
        groups = m.groups()
        if is_bare(groups) and (YEAR_RE.match(groups[3]) or int(groups[3]) <= SMALL_INT_MAX):
            continue
        out.append(figure_of(groups, m.group(0), m.span()))
    return out


def tokens_in(line: str) -> list[Token]:
    """The typed number tokens of one line of source text. A table row is
    read cell by cell, so a "$" or "%" cell of its own has already merged
    into its number and the cell position is known; prose is read by the
    figure grammar. A bare four-digit number without a comma is a year."""
    out: list[Token] = []
    if "|" in line:
        cells = format_cells(normalize(line))
        values = cells[1:] if cells and cell_value(cells[0]) is None else cells
        for i, cell in enumerate(values):
            m = CELL_RE.match(mask_non_figures(cell).strip())
            if not m:
                continue
            dollar, opened, minus, digits, closed, percent = m.groups()
            if not dollar and not percent and not opened and not minus and "," not in digits \
                    and "." not in digits and YEAR_RE.match(digits):
                continue
            out.append(Token(text=cell.strip(), value=float(digits.replace(",", "")), dollar=bool(dollar),
                             percent=bool(percent), negative=bool(minus) or bool(opened and closed),
                             scale=None, position=i))
        return out
    for m in FIGURE_RE.finditer(mask_non_figures(normalize(line))):
        groups = m.groups()
        if is_bare(groups) and YEAR_RE.match(groups[3]):
            continue
        f = figure_of(groups, m.group(0))
        out.append(Token(text=f.text, value=f.value, dollar=f.dollar, percent=f.percent, negative=f.negative,
                         scale=f.scale))
    return out


def base_scale(units: str | None) -> str | None:
    if not units:
        return None
    head = re.split(r",?\s*\bexcept\b", units, maxsplit=1)[0]
    m = SCALE_WORD_RE.search(head)
    return m.group(1).lower() if m else None


def is_unit_line(line: str) -> bool:
    return bool(SCALE_WORD_RE.search(line)) or "except" in line.casefold()


def is_section_line(line: str) -> bool:
    """A label-only line of a table: no cells, and either a trailing colon
    or a few words with no digit ("Memo:", "Numerator:", "Years ended")."""
    stripped = line.strip()
    if not stripped or "|" in stripped or is_unit_line(stripped):
        return False
    return stripped.endswith(":") or (not any(ch.isdigit() for ch in stripped)
                                      and len(stripped.split()) <= SECTION_MAX_WORDS)


@lru_cache(maxsize=512)
def sections_of(text: str) -> tuple[tuple[str, str], ...]:
    """(canonical line, section label) for each line of a chunk's text
    that is not itself a section line."""
    out = []
    section = ""
    for line in normalize(text).splitlines():
        if not line.strip():
            continue
        if is_section_line(line):
            section = line.strip().rstrip(":").strip()
            continue
        out.append((canon(line), section))
    return tuple(out)


def section_of(chunk: Chunk, line: str) -> str:
    """The section label of the chunk line that `line` is, or is a long
    enough fragment of; "" when no line of the chunk holds it."""
    c = canon(line)
    if not c:
        return ""
    for chunk_line, section in sections_of(chunk.text):
        if c == chunk_line or (len(c) >= MIN_FRAGMENT_CHARS and c in chunk_line):
            return section
    return ""


def is_ratio_row(label: str, section: str) -> bool:
    return bool((words_of(label) | words_of(section)) & RATIO_WORDS)


def per_share_row(label: str, section: str) -> bool:
    """A row stated per share: its label says so, or its section does
    while the label is not a share count ("Diluted" under "Earnings per
    share:" is per share; under "Shares used in computing earnings per
    share:" it is a count)."""
    for text in (canon(label), canon(section)):
        if SHARE_COUNT_RE.search(text):
            return False
        if PER_SHARE_RE.search(text):
            return True
    return False


def money_row(label: str, section: str) -> bool:
    """Whether a bare cell of this row can be a money figure: a per-share
    row is dollars even under a ratios section; any other row under a
    ratio word is a ratio."""
    return per_share_row(label, section) or not is_ratio_row(label, section)


def row_scale(chunk: Chunk, row_label: str, section: str = "") -> tuple[str | None, str]:
    """(scale, why): the scale a row's numbers are stated in, or None
    with the reason it is not established. Only a unit line the chunk
    declares itself counts: an inherited line was read off an earlier
    chunk and may not govern this table. The line's scale applies unless
    its "except" clause names something the row or its section mentions
    (shares in thousands, ratios), in which case that clause's own scale
    applies, or nothing when the clause carries no scale word. A per-share
    row is in plain dollars; a ratio row holds no money scale."""
    if not chunk.units:
        return None, "the excerpt carries no unit line"
    if chunk.units_source != "declared":
        return None, ("the excerpt carries no unit line of its own; the units shown were %s from an earlier "
                      "excerpt" % (chunk.units_source or "carried"))
    if per_share_row(row_label, section):
        return UNSCALED, "per share"
    if is_ratio_row(row_label, section):
        return None, "the row '%s' is a ratio row" % row_label
    parts = re.split(r",?\s*\bexcept\b", chunk.units, maxsplit=1)
    base = base_scale(chunk.units)
    if len(parts) < 2:
        return base, "unit line"
    mention = words_of(row_label) | words_of(section)
    for clause in re.split(r",?\s+and\s+", parts[1]):
        keys = {w for w in words_of(clause) if w not in EXCEPT_STOP and w not in SCALE}
        if keys & mention:
            m = SCALE_WORD_RE.search(clause)
            if m:
                return m.group(1).lower(), "except clause"
            return None, "the unit line '%s' names the row '%s' in its except clause without a scale word" % (
                chunk.units, row_label)
    return base, "unit line"


def scale_for(chunk: Chunk, row_label: str, section: str = "") -> str | None:
    return row_scale(chunk, row_label, section)[0]


def row_label_of(line: str) -> str:
    if "|" not in line:
        return ""
    cells = format_cells(normalize(line))
    return cells[0] if cells and cell_value(cells[0]) is None else ""


def same_at_precision(value: float, target: Figure) -> bool:
    return ("%.*g" % (target.sig, value)) == ("%.*g" % (target.sig, target.value))


def type_compatible(figure: Figure, token: Token, label: str, section: str = "") -> bool:
    """Whether a source token can stand for the claim figure at all: a
    percent cell is never a dollar amount, a dollar cell is never a
    percent, and a ratio row (by its label or its section line) holds no
    money."""
    if token.percent and (figure.dollar or figure.scale):
        return False
    if figure.percent and token.dollar:
        return False
    if (figure.dollar or figure.scale) and not token.dollar and not money_row(label, section):
        return False
    return True


def change_column_at(chunk: Chunk, line: str, position: int | None) -> bool:
    """Whether the cell at `position` of `line` sits under a change column
    of a parsed table. A change column holds percentages, so a dollar or
    scaled claim figure cannot come from it even when the digits agree."""
    if position is None or chunk.column_source != "parsed" or not chunk.columns:
        return False
    cells = format_cells(normalize(line))
    values = cells[1:] if cells and cell_value(cells[0]) is None else cells
    if len(values) != len(chunk.columns):
        return False
    column = next((c for c in chunk.columns if c.index == position), None)
    return column is not None and column.period_end is None and column.duration is None


def find_figure(figure: Figure, passage: str, chunk: Chunk, claim_text: str) -> Hit | None:
    """The first token of `passage` that is the figure, judged line by
    line so each table row is read in its own scale. A direct match is
    numeric equality between type-compatible tokens. When the claim states
    a scale word and the row is stated in a different one, the token is
    converted into the claim's scale and rounded to the claim's precision:
    23,966 in USD millions is $24.0 billion at three significant digits."""
    for line in normalize(passage).splitlines():
        label = row_label_of(line)
        section = section_of(chunk, line) if label else ""
        line_scale = scale_for(chunk, label, section)
        for token in tokens_in(line):
            if not type_compatible(figure, token, label, section):
                continue
            if (figure.dollar or figure.scale) and change_column_at(chunk, line, token.position):
                continue
            scale = token.scale or line_scale
            # No conversion into or out of a scale the chunk does not
            # establish, and none for a per-share row: 1.06 per share is
            # never $1.06 billion.
            convert = figure.scale is not None and scale not in (None, UNSCALED) and figure.scale != scale
            sign_clash = token.negative and not figure.negative and not DECREASE_RE.search(claim_text)
            if token.value == figure.value:
                return Hit(token, line, label, section, scale, False, sign_clash)
            if convert and same_at_precision(token.value * SCALE[scale] / SCALE[figure.scale], figure):
                return Hit(token, line, label, section, scale, True, sign_clash)
    return None


def near_miss(figure: Figure, cited: list[tuple[str, Chunk]]) -> str | None:
    """Why a figure that is nowhere may still look present: its digits sit
    in a ratio row, or would equal it under a scale the chunk does not
    establish. For the flag text only; nothing is credited."""
    for cid, chunk in cited:
        for line in normalize(chunk.text).splitlines():
            label = row_label_of(line)
            if not label:
                continue
            section = section_of(chunk, line)
            under = " under '%s'" % section if section else ""
            scale, why = row_scale(chunk, label, section)
            base = base_scale(chunk.units)
            for token in tokens_in(line):
                if token.value == figure.value and (figure.dollar or figure.scale) and not token.dollar \
                        and not money_row(label, section):
                    return "%s in %s sits in the row '%s'%s, a ratio row that holds no money figure" % (
                        token.text, cid, label, under)
                if scale is None and base and figure.scale and figure.scale != base \
                        and same_at_precision(token.value * SCALE[base] / SCALE[figure.scale], figure):
                    return "%s in %s would read as %s in %ss, but %s" % (token.text, cid, figure.text, base, why)
    return None


def same_figure(a: Figure, b: Figure) -> bool:
    """Whether two stated figures are one number: equal as written, or
    equal after both scale words at the coarser precision."""
    if a.value == b.value and a.percent == b.percent:
        return True
    if a.scale and b.scale:
        sig = min(a.sig, b.sig)
        return ("%.*g" % (sig, a.value * SCALE[a.scale])) == ("%.*g" % (sig, b.value * SCALE[b.scale]))
    return False


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------


def chunk_body(chunk: Chunk) -> str:
    # The header line is part of what the model reads, so a quote taken
    # from it still counts as found.
    return canon(chunk.header + "\n" + chunk.text)


def closest_window(quote_tokens: list[str], body_tokens: list[str]) -> tuple[float, str]:
    """(share of quote tokens present, passage) for the window of the
    body, as long as the quote, that holds the most of its tokens."""
    if not quote_tokens or not body_tokens:
        return 0.0, ""
    wanted = set(quote_tokens)
    width = min(len(quote_tokens), len(body_tokens))
    best_share, best_start = 0.0, 0
    for start in range(0, len(body_tokens) - width + 1):
        window = body_tokens[start:start + width]
        share = len(wanted & set(window)) / len(wanted)
        if share > best_share:
            best_share, best_start = share, start
    return best_share, " ".join(body_tokens[best_start:best_start + width])


def neighbours(chunk: Chunk, context: Context) -> list[tuple[str, Chunk]]:
    """The context chunks adjacent to `chunk` in its own filing."""
    return [(cid, other) for cid, other in zip(context.cids, context.chunks)
            if other.file == chunk.file and abs(other.seq - chunk.seq) == 1]


def locate_quote(quote: str, cited: list[tuple[str, Chunk]], context: Context) -> tuple[str | None, str, str | None]:
    """(cid, tier, passage): the chunk holding the quote. tier is "exact"
    (a substring of a cited chunk, or of a cited chunk joined to its
    neighbour when the quote straddles the boundary), "neighbour" (a
    substring of the uncited neighbour alone), "approximate" (long common
    substring or most tokens of a cited chunk; the passage is the closest
    text) or "missing"."""
    q = canon(quote)
    if not q:
        return None, "missing", None
    for cid, chunk in cited:
        if q in chunk_body(chunk):
            return cid, "exact", None
    cited_ids = {cid for cid, _c in cited}
    for cid, chunk in cited:
        own = canon(chunk.text)
        for ncid, other in neighbours(chunk, context):
            if ncid in cited_ids:
                continue
            if q in chunk_body(other):
                return ncid, "neighbour", None
            joined = own + " " + canon(other.text) if other.seq > chunk.seq else canon(other.text) + " " + own
            if q in joined:
                return cid, "exact", None
    best: tuple[float, str, str] | None = None
    for cid, chunk in cited:
        body = chunk_body(chunk)
        match = SequenceMatcher(None, body, q, autojunk=False).find_longest_match(0, len(body), 0, len(q))
        if match.size >= LCS_MIN_CHARS:
            start = max(0, match.a - match.b)
            passage = body[start:start + len(q)]
            score = match.size / len(q)
            if best is None or score > best[0]:
                best = (score, cid, passage)
            continue
        share, passage = closest_window(q.split(), body.split())
        if share >= TOKEN_SHARE_MIN and (best is None or share > best[0]):
            best = (share, cid, passage)
    if best is not None:
        return best[1], "approximate", best[2]
    return None, "missing", None


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


def cell_value(cell: str) -> float | None:
    m = CELL_RE.match(cell.strip())
    return float(m.group(4).replace(",", "")) if m else None


def table_rows(chunk: Chunk) -> list[tuple[str, str, list[str]]]:
    """(line, label, value cells) for every row of the chunk that carries
    a label and at least one number."""
    out = []
    for line in chunk.text.splitlines():
        if "|" not in line:
            continue
        cells = format_cells(line)
        if not cells or cell_value(cells[0]) is not None:
            continue
        values = cells[1:]
        if any(cell_value(c) is not None for c in values):
            out.append((line, cells[0], values))
    return out


def find_row(chunk: Chunk, quote: str, value: float) -> tuple[str, list[str]] | None:
    """(label, value cells) of the row the figure came from: the row the
    quote covers when that row holds the figure, else the first row
    holding it. Cells are split the way chunk.py split them, so a "$" or
    "(" cell has already merged into its number."""
    q = canon(quote)
    rows = [(canon(line), label, values) for line, label, values in table_rows(chunk)
            if any(cell_value(c) == value for c in values)]
    chosen = next(((label, values) for line, label, values in rows if line in q or q in line), None)
    if chosen is None and rows:
        chosen = rows[0][1], rows[0][2]
    return chosen


def column_of_figure(chunk: Chunk, quote: str, token: Token) -> tuple[Column | None, str | None, str]:
    """(column, reason the column is unknown, row label). A column comes
    back only when the row's value cells match the parsed column count
    and the figure fills exactly one of them."""
    if chunk.column_source != "parsed" or not chunk.columns:
        return None, "column_source is %s" % (chunk.column_source or "none"), ""
    row = find_row(chunk, quote, token.value)
    if row is None:
        return None, "no table row holds %s" % token.text, ""
    label, values = row
    if len(values) != len(chunk.columns):
        return None, "row has %d value cells for %d columns" % (len(values), len(chunk.columns)), label
    positions = [i for i, v in enumerate(values) if cell_value(v) == token.value]
    if len(positions) != 1:
        return None, "%s fills %d columns of the row" % (token.text, len(positions)), label
    by_index = {c.index: c for c in chunk.columns}
    column = by_index.get(positions[0])
    if column is None:
        return None, "no column at position %d" % positions[0], label
    return column, None, label


def is_month_end(date: str) -> bool:
    try:
        y, m, d = (int(p) for p in date.split("-"))
        return calendar.monthrange(y, m)[1] == d
    except (ValueError, TypeError):
        return False


def period_matches(column: Column, period_end: str, chunk: Chunk) -> bool | None:
    """Whether the column's period is the claim's. Exact date, or a column
    date up to MONTH_END_SLACK_DAYS before a claim date that is a month
    end (a 52/53-week filer closes in the last week). A column the chunker
    labelled only by fiscal year ("FY2022" from a bare-year header) matches
    a claim whose year is the label's and whose month is the filing's own
    year-end month, since a fiscal year is named for the calendar year it
    ends in and every fiscal year of one filer ends in the same month.
    None when the column carries no period at all."""
    if column.period_end:
        if column.period_end == period_end:
            return True
        if not is_month_end(period_end) or column.period_end[:7] != period_end[:7]:
            return False
        try:
            gap = (dt.date.fromisoformat(period_end) - dt.date.fromisoformat(column.period_end)).days
        except ValueError:
            return False
        return 0 <= gap <= MONTH_END_SLACK_DAYS
    fy = FY_LABEL_RE.search(column.label)
    if fy:
        return period_end[:4] == fy.group(1) and period_end[5:7] == (chunk.period_end or "")[5:7]
    return None


def column_year(column: Column) -> str | None:
    if column.period_end:
        return column.period_end[:4]
    fy = FY_LABEL_RE.search(column.label)
    return fy.group(1) if fy else None


def is_earlier(column: Column, period_end: str) -> bool:
    year = column_year(column)
    if column.period_end:
        return column.period_end < period_end
    return year is not None and year < period_end[:4]


def names(text_words: set[str], word: str) -> bool:
    """Whether the text carries `word`: as a word of its own, or as a
    five-letter stem shared with one ("estimate" for "estimated",
    "jpmorgan" for "jpmorganchase"). A word inside another word never
    counts, so "another" does not name an "All Other" column."""
    if word in text_words:
        return True
    return any(min(len(w), len(word)) >= 5 and (w.startswith(word) or word.startswith(w)) for w in text_words)


def row_reading(chunk: Chunk, label: str, claim_text: str, other_labels: set[str]) -> tuple[str, str | None]:
    """("ok" | "unnamed" | "mismatch", the row the claim reads as). The
    claim names its row when it carries any word of the row's label. It
    reads as another row when that row's label is wholly in the claim and
    the words that set the figure's row apart from it are absent: "Net
    income was 23,966" against the "Net interest income" row. Rows other
    figures of the same claim sit in are left out, so a two-figure
    sentence naming both rows is not read as either."""
    mine = words_of(FOOTNOTE_MARK_RE.sub(" ", label))
    if not mine:
        return "ok", None
    claim_words = naming_words(claim_text)
    if not any(names(claim_words, w) for w in mine):
        return "unnamed", None
    for _line, other_label, _values in table_rows(chunk):
        if other_label == label or other_label in other_labels:
            continue
        other = words_of(FOOTNOTE_MARK_RE.sub(" ", other_label))
        if not other or other == mine:
            continue
        if all(names(claim_words, w) for w in other) and not any(names(claim_words, w) for w in mine - other):
            return "mismatch", other_label
    return "ok", None


def dimension_words(column: Column, columns: list[Column]) -> set[str]:
    """The words of a column label that set it apart from its siblings
    once period words are gone: "Standardized", "Bank", "N.A." for a
    capital table; nothing for a plain three-months column."""
    own = {w for w in words_of(column.label) if w not in PERIOD_WORDS and not w.isdigit()}
    shared = None
    for c in columns:
        theirs = {w for w in words_of(c.label) if w not in PERIOD_WORDS and not w.isdigit()}
        shared = theirs if shared is None else shared & theirs
    return own - (shared or set())


def column_reading(chunk: Chunk, column: Column, claim_text: str, taken: set[int],
                   row_label: str = "") -> tuple[str, Column | None]:
    """("ok" | "unnamed" | "mismatch", the sibling the claim reads as).
    Each column is scored by how many of its distinguishing words the
    claim carries; the figure's column must score highest and above zero.
    Words of the figure's own row label are left out of every column's
    words, since a claim naming its row ("CET1 capital ratio") must not
    read as a sibling column that repeats one of them ("Regulatory
    Minimum ratios"). Columns other figures of the claim sit in are left
    out of the contest, so a sentence naming two segments is not read as
    either."""
    row_words = words_of(FOOTNOTE_MARK_RE.sub(" ", row_label))
    dims = {c.index: dimension_words(c, chunk.columns) - row_words for c in chunk.columns}
    if not dims[column.index]:
        return "ok", None
    claim_words = naming_words(claim_text)
    everyone = set().union(*dims.values())

    def named(word: str) -> bool:
        if names(claim_words, word):
            return True
        # "total revenue" names a Consolidated column, unless a sibling
        # column is itself the Total.
        return any(names(claim_words, s) for s in COLUMN_SYNONYMS.get(word, ()) if s not in everyone)

    score = {index: sum(named(w) for w in words) for index, words in dims.items()}
    own = score[column.index]
    rival = max(((score[c.index], c) for c in chunk.columns if c.index != column.index and c.index not in taken),
                key=lambda s: s[0], default=(0, None))
    if rival[0] > own or (rival[0] == own and own > 0):
        return "mismatch", rival[1]
    if own == 0:
        return "unnamed", None
    return "ok", None


class Comparatives:
    """Where the comparative phrases of a claim stand, so an earlier
    column's figure is excused only when a phrase attaches to that figure:
    a prefix phrase between the previous figure and it, a postfix phrase
    between it and the next figure, or the column's own year beside it
    with no other figure between."""

    def __init__(self, text: str, figures: list[Figure], period_end: str | None):
        self.text = text
        self.spans = sorted((f.start, f.end) for f in figures)
        self.prefix = [m.end() for m in PREFIX_COMPARATIVE_RE.finditer(text)]
        self.postfix = [m.start() for m in POSTFIX_COMPARATIVE_RE.finditer(text)]
        self.claim_year = (period_end or "")[:4]

    def cover(self, figure: Figure, column: Column) -> bool:
        start, end = figure.start, figure.end
        before = max((e for s, e in self.spans if e <= start), default=0)
        after = min((s for s, e in self.spans if s >= end), default=len(self.text))
        if any(before <= c <= start for c in self.prefix):
            return True
        if any(end <= c <= after for c in self.postfix):
            return True
        year = column_year(column)
        if year and year != self.claim_year:
            return re.search(r"(?<!\d)%s(?!\d)" % year, self.text[before:after]) is not None
        return False


def scale_phrase(hit: Hit) -> str:
    if hit.scale == UNSCALED:
        return "'%s' is stated per share, without a scale" % hit.label
    return "is in %ss" % hit.scale


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


@dataclass
class Located:
    """One claim figure and where it was found."""

    figure: Figure
    hit: Hit
    cid: str
    chunk: Chunk
    in_quote: bool


def scale_reason(loc: Located) -> str:
    """Why the scale of the line a figure was found in is not established:
    the row's reason from the unit line, or, for prose, the absence of a
    scale word beside the number."""
    if not loc.hit.label and not loc.hit.token.scale:
        return "the excerpt text states no scale word beside %s" % loc.hit.token.text
    return row_scale(loc.chunk, loc.hit.label, loc.hit.section)[1]


class Checker:
    """State for one Answer. The flag list and the counters accumulate as
    each claim is checked; `result` packs them."""

    def __init__(self, context: Context, chunks_by_cid: dict[str, Chunk], plan: Plan | None,
                 known_tickers: set[str] | None, known_names: dict[str, str] | None):
        self.context = context
        self.chunks = chunks_by_cid
        self.in_scope = ({c["ticker"] for c in plan.companies} if plan
                         else {c.ticker for c in chunks_by_cid.values()})
        self.known_tickers = known_tickers or set()
        self.known_names = known_names or {}
        self.flags: list[dict] = []
        self.unlinked: list[str] = []
        self.quotes = [0, 0]
        self.figures = [0, 0]
        self.columns = [0, 0]
        self.columns_unverified = 0
        self.units = [0, 0]

    def flag(self, kind: str, claim_id: str | None, detail: str, source_string: str | None = None,
             cid: str | None = None) -> None:
        self.flags.append({"kind": kind, "claim_id": claim_id, "detail": detail,
                           "source_string": source_string, "cid": cid})

    # -- claims -------------------------------------------------------------

    def check_claim(self, claim: Claim) -> None:
        cited = []
        for cid in claim.citations:
            if cid in self.chunks:
                cited.append((cid, self.chunks[cid]))
            else:
                self.flag("citation_unknown", claim.id, "citation %s is not an excerpt id" % cid, cid=cid)
        figures = figures_in(claim.text)
        self.quotes[1] += 1
        self.figures[1] += len(figures)
        for ticker in claim.tickers:
            if ticker not in self.in_scope:
                self.flag("ticker_out_of_scope", claim.id, "%s is not a company the question scoped" % ticker)
        if not claim.citations:
            self.flag("no_citation", claim.id, "the claim cites no excerpt; nothing in it was checked")
        if not cited:
            # Nothing to compare against: the quote and the figures stay
            # unchecked and count as not found in the totals.
            return

        words = len(claim.quote.split())
        if words > MAX_QUOTE_WORDS:
            self.flag("quote_too_long", claim.id, "%d words; the limit is %d" % (words, MAX_QUOTE_WORDS))
        if len(canon(claim.quote).split()) < MIN_QUOTE_TOKENS:
            self.flag("quote_too_short", claim.id,
                      "%d tokens locate no passage; the limit is %d" % (len(canon(claim.quote).split()),
                                                                        MIN_QUOTE_TOKENS))
            quote_cid, tier, passage = None, "missing", None
        else:
            quote_cid, tier, passage = locate_quote(claim.quote, cited, self.context)
        if tier == "exact":
            self.quotes[0] += 1
        elif tier == "neighbour":
            self.quotes[0] += 1
            self.flag("quote_in_neighbour", claim.id,
                      "the quote is in %s, next to the cited %s in the same filing; figures and columns are read "
                      "from %s" % (quote_cid, ", ".join(cid for cid, _c in cited), quote_cid), cid=quote_cid)
        elif tier == "approximate":
            self.flag("approximate_quote", claim.id, "closest passage in %s" % quote_cid, passage, quote_cid)
        elif tier == "missing" and len(canon(claim.quote).split()) >= MIN_QUOTE_TOKENS:
            tried = [cid for cid, _c in cited]
            for _cid, chunk in cited:
                tried.extend(n for n, _c in neighbours(chunk, self.context) if n not in tried)
            self.flag("quote_not_found", claim.id, "not in %s" % ", ".join(tried), cid=cited[0][0])
        quote_chunk = self.chunks[quote_cid] if quote_cid else cited[0][1]
        quote_cid = quote_cid or cited[0][0]
        # Only text located in a chunk can vouch for a figure: the quote
        # when it was found verbatim, the chunk's own closest passage when
        # it was approximate, nothing when it was missing.
        grounded = claim.quote if tier in ("exact", "neighbour") else passage if tier == "approximate" else None

        if figures:
            self.units[1] += 1
            if quote_chunk.units_source == "declared":
                self.units[0] += 1
        located: list[Located] = []
        for figure in figures:
            found = self.locate_figure(claim, figure, grounded, quote_cid, quote_chunk, cited)
            if found is not None:
                located.append(found)
        self.check_columns(claim, figures, located)
        self.check_units(claim, figures, located)

    def locate_figure(self, claim: Claim, figure: Figure, grounded: str | None, quote_cid: str,
                      quote_chunk: Chunk, cited: list[tuple[str, Chunk]]) -> Located | None:
        """Find one figure: in the grounded passage first, then in the
        chunk texts, the quote's chunk ahead of the rest."""
        hit = find_figure(figure, grounded, quote_chunk, claim.text) if grounded else None
        if hit is not None:
            self.figures[0] += 1
            where_cid, where, in_quote = quote_cid, quote_chunk, True
        else:
            where_cid, where, in_quote = None, None, False
            order = [(quote_cid, quote_chunk)] + [(cid, c) for cid, c in cited if cid != quote_cid]
            for cid, chunk in order:
                hit = find_figure(figure, chunk.text, chunk, claim.text)
                if hit is not None:
                    where_cid, where = cid, chunk
                    break
            if hit is None:
                why = near_miss(figure, order)
                self.flag("figure_not_in_chunk", claim.id,
                          "%s is in neither the quote nor the cited excerpts%s" % (
                              figure.text, "; " + why if why else ""), cid=quote_cid)
                return None
            self.flag("figure_not_in_quote", claim.id,
                      "%s is in %s but the quote does not carry it" % (figure.text, where_cid),
                      hit.token.text, where_cid)
        if hit.converted:
            self.flag("unit_converted", claim.id,
                      "%s read as %s (%s to %s)" % (hit.token.text, figure.text, hit.scale, figure.scale),
                      hit.token.text, where_cid)
        if hit.sign_clash:
            self.flag("sign_mismatch", claim.id,
                      "%s is negative in the source; the claim reads it as positive" % hit.token.text,
                      hit.token.text, where_cid)
        return Located(figure, hit, where_cid, where, in_quote)

    def check_columns(self, claim: Claim, figures: list[Figure], located: list[Located]) -> None:
        """Column provenance for every figure found in a table, judged
        together so a comparative sentence can carry an earlier column's
        figure next to the one that anchors the claim's period."""
        seated = []
        for loc in located:
            if loc.chunk.kind != "table":
                continue
            column, reason, label = column_of_figure(loc.chunk, claim.quote, loc.hit.token)
            if column is None:
                self.columns_unverified += 1
                self.flag("column_unverified", claim.id, "%s: %s" % (loc.figure.text, reason), cid=loc.cid)
                continue
            period = period_matches(column, claim.period_end, loc.chunk)
            if period is None:
                self.columns_unverified += 1
                self.flag("column_unverified", claim.id,
                          "%s: the column carries no period" % loc.figure.text, column.label, loc.cid)
                continue
            if column.duration is None and claim.period_kind not in (None, "point_in_time"):
                # A dated column without a duration is a capital or balance
                # sheet column the chunker could not place in time: it
                # can seat a point-in-time figure, and nothing else.
                self.columns_unverified += 1
                self.flag("column_unverified", claim.id,
                          "%s: the column carries a date and no duration; the claim says %s" % (
                              loc.figure.text, claim.period_kind), column.label, loc.cid)
                continue
            seated.append((loc, column, label, period))
        anchored = any(period for _loc, _column, _label, period in seated)
        comparatives = Comparatives(normalize(claim.text), figures, claim.period_end)
        for loc, column, label, period in seated:
            other_rows = {lab for other, _c, lab, _p in seated if other is not loc}
            taken = {c.index for other, c, _l, _p in seated if other is not loc and other.chunk is loc.chunk}
            row_state, row_as = row_reading(loc.chunk, label, claim.text, other_rows)
            col_state, col_as = column_reading(loc.chunk, column, claim.text, taken, label)
            if "mismatch" not in (row_state, col_state) and "unnamed" in (row_state, col_state):
                self.columns_unverified += 1
                what = "row '%s'" % label if row_state == "unnamed" else "column '%s'" % column.label
                self.flag("column_unverified", claim.id,
                          "%s: the claim names nothing of the %s" % (loc.figure.text, what), column.label, loc.cid)
                continue
            self.columns[1] += 1
            ok = True
            if not period:
                if anchored and is_earlier(column, claim.period_end) and comparatives.cover(loc.figure, column):
                    pass
                else:
                    ok = False
                    self.flag("column_mismatch", claim.id,
                              "%s sits in a column for %s; the claim says %s" % (
                                  loc.figure.text, column.period_end or column.label, claim.period_end),
                              column.label, loc.cid)
            allowed = DURATIONS_FOR.get(claim.period_kind)
            if column.duration is not None and (allowed is None or column.duration not in allowed):
                ok = False
                self.flag("duration_mismatch", claim.id,
                          "%s sits in a %s column; the claim says %s" % (
                              loc.figure.text, column.duration, claim.period_kind or "(no period_kind)"),
                          column.label, loc.cid)
            if loc.figure.percent and not loc.hit.token.percent and loc.hit.scale:
                ok = False
                self.flag("figure_type_mismatch", claim.id,
                          "%s is a percentage; %s sits in a column stated in %s" % (
                              loc.figure.text, loc.hit.token.text, loc.chunk.units), column.label, loc.cid)
            if row_state == "mismatch":
                ok = False
                self.flag("row_mismatch", claim.id,
                          "%s sits in the row '%s'; the claim reads as the row '%s'" % (
                              loc.figure.text, label, row_as), label, loc.cid)
            if col_state == "mismatch":
                ok = False
                self.flag("column_mismatch", claim.id,
                          "%s sits in the column '%s'; the claim reads as the column '%s'" % (
                              loc.figure.text, column.label, col_as.label if col_as else "?"),
                          column.label, loc.cid)
            if ok:
                self.columns[0] += 1

    def check_units(self, claim: Claim, figures: list[Figure], located: list[Located]) -> None:
        """A scale word on a claim figure must be the scale of the row it
        was found in, or the figure must have matched through the
        conversion between them. A figure without its own scale word
        takes the claim's one scale word when the text has exactly one
        and no figure of the claim owns it ("23,966 (in millions)"; in
        "23,966, roughly $24 billion" the word belongs to $24). A scale
        the source does not establish for the row is reported as
        units_unverified, never passed over."""
        loose = [m.group(1).lower() for m in SCALE_WORD_RE.finditer(normalize(claim.text))]
        free = loose[0] if len(loose) == 1 and not any(f.scale for f in figures) else None
        for loc in located:
            stated = loc.figure.scale or free
            if not stated or loc.figure.percent or loc.hit.converted:
                continue
            if loc.hit.scale is None:
                self.flag("units_unverified", claim.id,
                          "the claim says %s for %s; %s, so the scale word was not checked" % (
                              stated, loc.figure.text, scale_reason(loc)), loc.chunk.units, loc.cid)
            elif stated != loc.hit.scale:
                self.flag("units_mismatch", claim.id,
                          "the claim says %s; the excerpt row %s" % (stated, scale_phrase(loc.hit)),
                          loc.chunk.units, loc.cid)

    # -- sentences, cells, gaps ---------------------------------------------

    def check_links(self, answer: Answer) -> None:
        claims = {c.id: c for c in answer.claims}
        for sentence in answer.summary:
            self.check_link(sentence.text, sentence.claim_ids, claims, "summary sentence")
        for row in answer.table:
            for cell in row.cells:
                self.check_link(cell.text, cell.claim_ids, claims, "table cell %s / %s" % (row.dimension, cell.column))

    def check_link(self, text: str, claim_ids: list[str], claims: dict[str, Claim], what: str) -> None:
        """The ids must exist, and every figure the text states must be a
        figure one of its claims states, so the reader-facing layer can
        never carry a number the claims do not."""
        unknown = [k for k in claim_ids if k not in claims]
        if not claim_ids and what.startswith("table cell") and NO_DATA_RE.match(text.strip()):
            # "Not disclosed" states that no figure exists; there is no
            # claim it could cite.
            return
        if not claim_ids or unknown:
            detail = "%s names no claim" % what if not claim_ids else "%s names unknown claim %s" % (
                what, ", ".join(unknown))
            self.flag("unlinked_sentence", None, detail, text)
            self.unlinked.append(text)
        backing = [f for k in claim_ids if k in claims for f in figures_in(claims[k].text)]
        for figure in figures_in(text):
            if not any(same_figure(figure, b) for b in backing):
                self.flag("unbacked_figure", None,
                          "%s states %s; its claims %s state %s" % (
                              what, figure.text, ", ".join(k for k in claim_ids if k in claims) or "(none)",
                              ", ".join(b.text for b in backing) or "no figure"), text)

    def coverage_words(self) -> tuple[set[str], str]:
        coverage = normalize(self.context.coverage)
        words = set(re.findall(r"[A-Za-z0-9]+", coverage))
        # "FY2026" in the coverage block covers a gap that says "2026".
        words |= set(FY_LABEL_RE.findall(coverage))
        return words, coverage

    def ungrounded_in(self, text: str) -> list[str]:
        """Years, registry tickers and registry company names the text
        names that the coverage block does not."""
        words, coverage = self.coverage_words()
        text = normalize(text)
        found = [y for y in GAP_YEAR_RE.findall(text) if y not in words]
        found += [t for t in GAP_TICKER_RE.findall(text) if t in self.known_tickers and t not in words]
        for ticker, name in self.known_names.items():
            phrase = company_phrase(name)
            if not phrase or ticker in found:
                continue
            # The registry's own capitalization, so "Target" the filer is
            # caught and "a target" in plain prose is left alone.
            pattern = r"(?<![A-Za-z])" + re.escape(phrase) + r"(?![A-Za-z])"
            if re.search(pattern, text) and not re.search(pattern, coverage):
                found.append(name)
        return found

    def check_gaps(self, answer: Answer) -> None:
        for gap in answer.gaps:
            named = self.ungrounded_in(gap)
            if named:
                self.flag("ungrounded_gap", None, "names %s outside the coverage block" % ", ".join(named), gap)
        for note in answer.not_comparable:
            outside = [t for t in note.tickers if t not in self.in_scope]
            named = self.ungrounded_in(note.dimension + " " + note.reason)
            if outside or named:
                self.flag("ungrounded_not_comparable", None,
                          "names %s outside the question's scope" % ", ".join(outside + named),
                          "%s: %s" % (note.dimension, note.reason))

    def result(self) -> EvidenceChecks:
        return EvidenceChecks(
            quotes_found=(self.quotes[0], self.quotes[1]),
            figures_in_quote=(self.figures[0], self.figures[1]),
            columns_matched=(self.columns[0], self.columns[1]),
            columns_unverified=self.columns_unverified,
            units_declared=(self.units[0], self.units[1]),
            flags=self.flags,
            unlinked_sentences=self.unlinked,
        )


COMPANY_SUFFIXES = {"inc", "inc.", "corp", "corp.", "corporation", "co", "co.", "company", "ltd", "ltd.",
                    "plc", "&", "and", "the", "holdings", "group", "limited", "llc", "lp", "l.p.", "n.a.", "sa",
                    "s.a.", "ag", "nv", "n.v."}


def company_phrase(name: str) -> str:
    """The words of a registry name a gap would use, in the registry's
    capitalization: "Wells Fargo" from "Wells Fargo & Company"."""
    words = [w for w in normalize(name).replace(",", " ").split() if w.casefold() not in COMPANY_SUFFIXES]
    return " ".join(words[:3])


def check(answer: Answer, context: Context, chunks_by_cid: dict[str, Chunk], plan: Plan | None = None,
          known_tickers: set[str] | None = None, known_names: dict[str, str] | None = None) -> EvidenceChecks:
    """The evidence checks for one Answer over the context it was written
    from. `plan` supplies the tickers in scope (the context's own tickers
    stand in without it); `known_tickers` and `known_names` (ticker to
    company name) are the registry, for the gap check."""
    checker = Checker(context, chunks_by_cid, plan, known_tickers, known_names)
    for claim in answer.claims:
        checker.check_claim(claim)
    checker.check_links(answer)
    checker.check_gaps(answer)
    return checker.result()


def describe(checks: EvidenceChecks) -> list[str]:
    """The counts line and one line per flag, for the CLI. "figures in
    quote" counts figures found in a quote that was itself located in a
    chunk; a figure found only against the model's own unlocated quote
    is never counted."""
    lines = ["evidence checks: quotes found %d/%d | figures in quote %d/%d | columns matched %d/%d, "
             "unverified %d | units declared %d/%d" % (
                 *checks.quotes_found, *checks.figures_in_quote, *checks.columns_matched,
                 checks.columns_unverified, *checks.units_declared)]
    if not checks.flags:
        lines.append("  flags: (none)")
    for f in checks.flags:
        who = " %s" % f["claim_id"] if f["claim_id"] else ""
        source = " [source: %s]" % f["source_string"] if f["source_string"] else ""
        lines.append("  - %s%s: %s%s" % (f["kind"], who, f["detail"], source))
    return lines
