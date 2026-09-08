"""Check an Answer against the excerpts it cites.

Nothing in the model's reply says whether its quote is in the excerpt, whether
the numbers of its claim are printed in the source, which column of a table a
number sits in, or what scale that column is stated in. This module settles
each of those from the chunks and reports every item as matched, flagged, or
unchecked with the reason the check could not run.

The checks, in the order they run for one claim, and the sentence each
establishes:

  quote   The claim's quote is text of a chunk the claim cites; a quote may
          run on into the next chunk of the same filing, never begin there.
  figure  Each number of the claim text is printed inside the located span, in
          a source number whose printed form ($, %, parentheses, digits) can
          stand for the way the claim writes it.
  column  The cell a figure sits in belongs to a parsed column whose period,
          duration, label words and row label the claim all name.
  units   The scale word the claim uses is the scale the chunk's own unit line
          states, or the source number converts into it exactly.
  scope   Summary sentences, table cells, gaps and not-comparable notes name
          only claims that exist, figures those claims carry, tickers the
          question scoped, and periods the coverage block holds.

Failure policy. A check reports matched, flagged or unchecked; silence is
never a pass. Nothing downstream of a check that did not pass is credited: an
unlocated quote leaves that claim's figures, columns and units unchecked, and
a figure found outside the located span is flagged and reaches no column.
column_unverified and units_unchecked are unchecked outcomes, counted beside
the matched pair and never inside its denominator, so "columns matched 1/1,
unverified 3" cannot read as four matches. The only type signal a source
number carries is its own printed form; nothing here reads meaning from a row
label, so no row is called money, a ratio or per-share, and no comparative
wording excuses a figure: each figure's column is reported and the reader
judges. Numbers are compared as their digits are written, a row label must be
named in full, and a unit line carrying any "except" clause establishes no
scale for any row, because deciding which rows that clause covers means
reading the row labels.
"""

import calendar
import re
from collections import namedtuple
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

from chunk import format_cells
from corpus import normalize
from models import Answer, Chunk, Claim, Column, Context, EvidenceChecks, Plan

# A two-token quote locates nothing: "23,966" sits in every table that holds
# the number, so finding it says nothing about the source.
MIN_QUOTE_TOKENS = 3
# The approximate tier: this many characters in common with the chunk, or this
# share of the quote's tokens inside one window of it.
LCS_MIN_CHARS = 60
TOKEN_SHARE_MIN = 0.9
# Integers up to this are counts in prose ("three companies"), not figures.
SMALL_INT_MAX = 12
# A scale word this many words after a bare figure is still that figure's
# ("23,966, stated in millions"); further off it belongs to another.
SCALE_ATTACH_WORDS = 4
SCALE_WORDS = r"thousand|million|billion|trillion"
SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
DURATIONS_FOR = {"quarter": ("three_months",), "nine_months": ("nine_months",),
                 "fiscal_year": ("fiscal_year", "twelve_months"), "point_in_time": ()}
MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|december"
          "|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")

# Dates, identifiers and counts are blanked before numbers are read, so the "30"
# of "September 30, 2025", a CIK's digits and the "52" of "52-week" are never
# looked for. A mask is as long as what it hides, so offsets still line up.
MASKS = tuple(re.compile(p, re.I) for p in (
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b(?:%s)\.?\s*\d{1,2}(?:\s*,?\s*\d{4})?\b" % MONTHS,
    r"\b\d{1,2}\s+(?:%s)\.?(?:\s*,?\s*\d{4})?\b" % MONTHS,
    r"\b(?:cik|notes?|items?|excerpts?|citations?|claims?|tables?|exhibits?|schedules?|pages?|parts?"
    r"|footnotes?|lines?|sections?|levels?|tiers?|q|fy|fiscal)\s*#?\s*\d+[a-z]?\b|\b\d+-[kq]\b",
    r"\b\d{1,3}[- ](?:week|day|month|year)s?\b|\b\d+(?:st|nd|rd|th)\b"))
# A number as a claim writes it. The lookbehind keeps "C12", "K3" (citation
# and claim ids) and "CET1" out of the number grammar.
FIGURE_RE = re.compile(r"(?<![A-Za-z\d.])(\$)?\s*(\()?(-)?(\d(?:[\d,]*\d)?(?:\.\d+)?)(\))?\s*(%)?"
                       r"(?:\s*(" + SCALE_WORDS + r")s?\b)?", re.I)
# A number as a table cell prints it, after chunk.format_cells has merged the
# "$", "(" and "%" cells into the number they belong to. The empty last group
# stands where FIGURE_RE has its scale word, which a cell never carries, so one
# reader handles both patterns.
CELL_RE = re.compile(r"^(\$)?\s*(\()?(-)?(\d[\d,]*(?:\.\d+)?)(\))?\s*(%)?()$")
YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
SCALE_WORD_RE = re.compile(r"\b(" + SCALE_WORDS + r")s?\b", re.I)
EXCEPT_RE = re.compile(r"\bexcept\b", re.I)
FOOTNOTE_MARK_RE = re.compile(r"\(\w{1,2}\)")
WORD_RE = re.compile(r"[a-z&][a-z&.']*")
GAP_YEAR_RE = re.compile(r"(?<![\d-])(?:19|20)\d{2}(?![\d-])")
GAP_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
# The words of a column label that describe its period; what is left is what
# distinguishes the column: an entity, a basis, an approach, a segment.
PERIOD_WORDS = set(MONTHS.split("|")) | set(
    "ended ending end month months three six nine twelve year years fiscal fy quarter quarters as of at "
    "the and for first second third fourth period periods change".split())
STOP_WORDS = set("the of and & for in at as a an to by on per with from or its it was were is are be been "
                 "has had have than that this which used".split())


def canon_with_offsets(text: str) -> tuple[str, list[int]]:
    """(comparison text, source index per character). Pipes and whitespace runs
    collapse to one space and the rest casefolds, so a cell boundary, a
    no-break space or a curly quote cannot break a match, and a match can still
    be mapped back to the characters it covers."""
    out: list[str] = []
    src: list[int] = []
    for i, ch in enumerate(text):
        folded = " " if ch == "|" or ch.isspace() else ch.casefold()
        if folded == " " and (not out or out[-1] == " "):
            continue
        out.extend(folded)
        src.extend([i] * len(folded))
    while out and out[-1] == " ":
        del out[-1], src[-1]
    return "".join(out), src


def canon(text: str) -> str:
    return canon_with_offsets(text)[0]


def mask_non_figures(text: str) -> str:
    for pattern in MASKS:
        text = pattern.sub(lambda m: " " * len(m.group(0)), text)
    return text


def words_of(text: str) -> set[str]:
    """Content words for naming a row or a column: casefolded, possessives
    and trailing punctuation dropped, plurals trimmed, stop words out."""
    out = set()
    for w in WORD_RE.findall(canon(FOOTNOTE_MARK_RE.sub(" ", text))):
        w = (w[:-2] if w.endswith("'s") else w).rstrip(".,'")
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if w and w not in STOP_WORDS:
            out.add(w)
    return out


def unnamed(wanted: set[str], claim_text: str) -> list[str]:
    """The words of `wanted` the claim does not carry. A word counts when the
    claim has it or a word sharing a five-character stem with it, so "JPMorgan
    Chase" names a "JPMorganChase" column; a word inside another never counts,
    so "another" does not name "All Other"; hyphens close up, so "non-interest"
    names a "Noninterest" row."""
    claim_words = words_of(claim_text) | words_of(claim_text.replace("-", ""))
    return sorted(w for w in wanted if w not in claim_words and not any(
        min(len(c), len(w)) >= 5 and (c.startswith(w) or w.startswith(c)) for c in claim_words))


@dataclass
class Number:
    """One number as written, on either side of the check. `cell` is its
    value-cell index inside a table row and None otherwise; `twins` counts the
    cells of that row printing the same digits."""

    text: str
    digits: str
    scale: str | None
    percent: bool
    dollar: bool
    negative: bool
    sig: int
    start: int = 0
    end: int = 0
    cell: int | None = None
    row: str = ""
    cells: int = 0
    twins: int = 1


def read_number(groups: tuple, text: str, start: int, end: int) -> Number:
    """A Number from the seven groups both number patterns share."""
    dollar, opened, minus, digits, closed, percent, scale = groups
    plain = digits.replace(",", "")
    return Number(text=text.strip(), digits=plain, scale=scale.lower() if scale else None,
                  percent=bool(percent), dollar=bool(dollar), negative=bool(minus) or bool(opened and closed),
                  sig=len(plain.replace(".", "").lstrip("0")) or 1, start=start, end=end)


def is_bare(groups: tuple) -> bool:
    dollar, opened, minus, digits, closed, percent, scale = groups
    return not (dollar or percent or scale or opened or minus or "." in digits or "," in digits)


def figures_in(text: str) -> list[Number]:
    """The numbers of a claim, with dates, identifiers, ordinals, counts, years
    and small integers left out. A scale word apart from its number ("23,966,
    in millions") attaches to the figure beside it, within four words and with
    no other figure between, never to every figure of the sentence."""
    masked = mask_non_figures(normalize(text))
    out = [read_number(m.groups(), m.group(0), m.start(), m.end()) for m in FIGURE_RE.finditer(masked)
           if not (is_bare(m.groups()) and (YEAR_RE.match(m.group(4)) or int(m.group(4)) <= SMALL_INT_MAX))]
    for i, figure in enumerate(out):
        if figure.scale is None:
            stop = out[i + 1].start if i + 1 < len(out) else len(masked)
            near = SCALE_WORD_RE.search(" ".join(masked[figure.end:stop].split()[:SCALE_ATTACH_WORDS]))
            figure.scale = near.group(1).lower() if near else None
    return out


def cell_number(cell: str) -> Number | None:
    """A cell that prints a number, else None. A bare four-digit number is a year
    and "(a)" is a footnote mark; neither is a figure."""
    m = CELL_RE.match(mask_non_figures(cell).strip())
    if not m:
        return None
    number = read_number(m.groups(), cell, 0, 0)
    return None if is_bare(m.groups()) and YEAR_RE.match(number.digits) else number


def line_numbers(line: str, offset: int) -> list[Number]:
    """The numbers of one line. A table row is split the way chunk.py split it,
    so a "$" or "(" cell has already merged into the number it opens and each
    number's column position is known; a row led by a number has no label."""
    out: list[Number] = []
    if "|" not in line:
        for m in FIGURE_RE.finditer(mask_non_figures(line)):
            if is_bare(m.groups()) and YEAR_RE.match(m.group(4)):
                continue
            out.append(read_number(m.groups(), m.group(0), offset + m.start(4), offset + m.end(4)))
        return out
    cells = format_cells(line)
    label, values = (cells[0], cells[1:]) if cells and cell_number(cells[0]) is None else ("", cells)
    # The cells come back merged, so each one's offset inside the line is found
    # by walking left to right for its own digits.
    pos = 0
    for i, cell in enumerate(values):
        core = cell.strip("$()% ")
        at = line.find(core, pos) if core else -1
        if at >= 0:
            pos = at + len(core)
        number = cell_number(cell)
        if number is None or at < 0:
            continue
        number.start, number.end, number.cell = offset + at, offset + at + len(core), i
        number.row, number.cells = label, len(values)
        out.append(number)
    for number in out:
        number.twins = sum(1 for other in out if other.digits == number.digits)
    return out


@lru_cache(maxsize=64)
def source_numbers(text: str) -> tuple[Number, ...]:
    out, offset = [], 0
    for line in text.split("\n"):
        out.extend(line_numbers(line, offset))
        offset += len(line) + 1
    return tuple(out)


def excerpt_text(chunk: Chunk) -> str:
    """The chunk as the model reads it, header line included: that line names
    the filing, the units and the columns, and the model can quote it."""
    return normalize(chunk.header + "\n" + chunk.text)


def match_figure(figure: Number, sources, scale: str | None) -> tuple[Number | None, bool]:
    """(source number, converted): the printed number the claim figure is. A
    percent claim takes only a percent cell, a claim written with a currency
    symbol or a scale word takes only a cell that is not a percent, and a bare
    claim takes either. The digits as written match first; then, only where the
    source's own scale is established and the claim states a different one, the
    number that rescales into the claim at its precision (130,497 in millions
    is $130.5 billion)."""
    def stands_for(source: Number) -> bool:
        if figure.percent:
            return source.percent
        return not (source.percent and (figure.dollar or figure.scale))

    for source in sources:
        if stands_for(source) and source.digits == figure.digits:
            return source, False
    if not scale or not figure.scale or figure.scale == scale or figure.percent:
        return None, False
    claimed = float(figure.digits)
    for source in sources:
        rescaled = float(source.digits) * SCALE[scale] / SCALE[figure.scale]
        if stands_for(source) and ("%.*g" % (figure.sig, rescaled)) == ("%.*g" % (figure.sig, claimed)):
            return source, True
    return None, False


# Where a claim's quote was found: the excerpt it is in, that excerpt's own
# text, and the character range of the passage inside it.
Span = namedtuple("Span", "cid chunk text start end tier")


def locate(quote: str, cited: list[tuple[str, Chunk]], context: Context) -> tuple[Span | None, str]:
    """(span, reason it is missing). An exact substring of a cited chunk is
    located, and so is one that begins in the cited chunk and runs on into the
    next chunk of the same filing; a quote lying wholly in that neighbour is
    not, because then the model cited the wrong excerpt. Otherwise the closest
    passage of a cited chunk, by longest common substring or token containment,
    is located_approximate."""
    q = canon(normalize(quote))
    if len(q.split()) < MIN_QUOTE_TOKENS:
        return None, "the quote is %d tokens; fewer than %d locate nothing" % (len(q.split()), MIN_QUOTE_TOKENS)
    for cid, chunk in cited:
        own = excerpt_text(chunk)
        after = next((c for c in context.chunks if c.file == chunk.file and c.seq == chunk.seq + 1), None)
        text, offsets = canon_with_offsets(own + ("\n" + normalize(after.text) if after else ""))
        at = text.find(q)
        if at >= 0 and offsets[at] < len(own):
            return Span(cid, chunk, own, offsets[at], min(offsets[at + len(q) - 1] + 1, len(own)), "located"), ""
    best: tuple[float, Span] | None = None
    for cid, chunk in cited:
        own = excerpt_text(chunk)
        text, offsets = canon_with_offsets(own)
        if not text:
            continue
        match = SequenceMatcher(None, text, q, autojunk=False).find_longest_match(0, len(text), 0, len(q))
        if match.size >= LCS_MIN_CHARS:
            at = max(0, match.a - match.b)
            score, start, end = match.size / len(q), at, min(len(text), at + len(q))
        else:
            # Token containment: the window of the chunk as long as the quote
            # that holds the most of the quote's own tokens.
            spans = [m.span() for m in re.finditer(r"\S+", text)]
            wanted = set(q.split())
            width = min(len(wanted), len(spans))
            hits = [(len({text[a:b] for a, b in w} & wanted) / len(wanted), w[0][0], w[-1][1])
                    for w in (spans[i:i + width] for i in range(0, len(spans) - width + 1)) if width]
            score, start, end = max(hits, key=lambda h: h[0], default=(0.0, 0, 0))
            if score < TOKEN_SHARE_MIN:
                continue
        if best is None or score > best[0]:
            best = (score, Span(cid, chunk, own, offsets[start], offsets[end - 1] + 1, "located_approximate"))
    if best is not None:
        return best[1], ""
    return None, "not in %s" % ", ".join(cid for cid, _c in cited)


def period_agrees(column: Column, period_end: str) -> bool:
    """Whether the column's date is the claim's: the same date, or the same
    month of the same year when the claim names a month end and the filer
    closed its period a few days earlier."""
    if not column.period_end or not re.match(r"\d{4}-\d{2}-\d{2}$", period_end or ""):
        return False
    year, month, day = (int(p) for p in period_end.split("-"))
    return column.period_end == period_end or (
        calendar.monthrange(year, month)[1] == day and column.period_end[:7] == period_end[:7])


def row_scale(chunk: Chunk) -> tuple[str | None, str]:
    """(scale, why it is not established) for every row of the chunk. Only a
    unit line the chunk declares itself, naming one scale word and excepting
    nothing, establishes a scale: "USD millions, except number of shares, which
    are reflected in thousands" would have to be read against each row label
    first, and that is a guess."""
    if not chunk.units:
        return None, "the excerpt carries no unit line"
    if chunk.units_source != "declared":
        return None, "the unit line was carried in from an earlier excerpt"
    if EXCEPT_RE.search(chunk.units):
        return None, "the unit line '%s' excepts rows it does not name" % chunk.units
    found = {m.group(1).lower() for m in SCALE_WORD_RE.finditer(chunk.units)}
    if len(found) != 1:
        return None, "the unit line '%s' names %d scale words" % (chunk.units, len(found))
    return found.pop(), ""


# One claim figure and the printed number it was found in.
Hit = namedtuple("Hit", "figure source converted")


class Checker:
    """State for one Answer. Each counter pair is [matched, checkable]."""

    def __init__(self, context: Context, chunks: dict[str, Chunk], in_scope: set[str], tickers: set[str]):
        self.context, self.chunks, self.in_scope = context, chunks, in_scope
        self.known_tickers = tickers
        self.flags, self.notes, self.unlinked = [], [], []
        self.quotes, self.figures, self.columns, self.units = [0, 0], [0, 0], [0, 0], [0, 0]
        self.figures_unchecked = self.columns_unverified = self.units_unchecked = 0

    def flag(self, where: str, kind: str, detail: str, source_string: str | None = None) -> None:
        self.flags.append({"where": where, "kind": kind, "detail": detail, "source_string": source_string})

    def note(self, where: str, kind: str, detail: str, source_string: str | None = None) -> None:
        self.notes.append({"where": where, "kind": kind, "detail": detail, "source_string": source_string})

    def check_claim(self, claim: Claim) -> None:
        where = claim.id
        self.quotes[1] += 1
        figures = figures_in(claim.text)
        for ticker in claim.tickers:
            if ticker not in self.in_scope:
                self.flag(where, "ticker_out_of_scope", "%s is not a company the question scoped" % ticker)
        cited = [(cid, self.chunks[cid]) for cid in claim.citations if cid in self.chunks]
        for cid in [c for c in claim.citations if c not in self.chunks]:
            self.flag(where, "citation_unknown", "citation %s is not an excerpt id" % cid, cid)
        if not claim.citations:
            self.flag(where, "no_citation", "the claim cites no excerpt")
        span, why = locate(claim.quote, cited, self.context) if cited else (None, "no excerpt to search")
        if span is None:
            self.flag(where, "quote_not_found", why, claim.quote[:120])
            self.figures_unchecked += len(figures)
            if figures:
                self.note(where, "figures_unchecked", "quote not located, %d figure(s) unchecked" % len(figures))
            return
        self.quotes[0] += 1
        if span.tier == "located_approximate":
            self.flag(where, "approximate_quote", "the closest passage of %s, not the quote as written"
                      % span.cid, span.text[span.start:span.end])
        self.figures[1] += len(figures)
        hits = [hit for hit in (self.find_figure(where, f, span, cited) for f in figures) if hit is not None]
        self.check_columns(where, claim, span.chunk, hits)
        self.check_units(where, span.chunk, hits)

    def find_figure(self, where: str, figure: Number, span: Span, cited: list[tuple[str, Chunk]]) -> Hit | None:
        """Search the located span first, then the rest of the cited chunks. A
        figure found outside the span is flagged and reaches no later check."""
        inside = [n for n in source_numbers(span.text) if span.start <= n.start and n.end <= span.end]
        source, converted = match_figure(figure, inside, row_scale(span.chunk)[0])
        if source is not None:
            self.figures[0] += 1
            if source.percent and not figure.percent:
                self.note(where, "bare_figure_from_percent_cell", "%s carries no percent sign; the cell it "
                          "matched is printed %s" % (figure.text, source.text), source.text)
            if source.negative != figure.negative:
                self.flag(where, "sign_differs", "%s is printed %s in %s" % (figure.text, source.text, span.cid))
            return Hit(figure, source, converted)
        for cid, chunk in cited:
            rest = [n for n in source_numbers(excerpt_text(chunk))
                    if chunk is not span.chunk or n.end <= span.start or n.start >= span.end]
            if match_figure(figure, rest, row_scale(chunk)[0])[0] is not None:
                self.flag(where, "figure_elsewhere_in_chunk", "%s is printed in %s outside the quote, so its "
                          "column and units were not read" % (figure.text, cid))
                return None
        self.flag(where, "figure_not_in_chunk", "%s is in neither the quote nor the cited excerpts" % figure.text)
        return None

    def check_columns(self, where: str, claim: Claim, chunk: Chunk, hits: list[Hit]) -> None:
        """One column per figure. The claim's period_end is held against the
        column of its first figure; a later figure whose column carries another
        period is reported as a fact, since whether a sentence licenses that
        comparison is not this module's to decide."""
        anchored = False
        for hit in hits:
            source, figure = hit.source, hit.figure
            if source.cell is None:
                continue
            column = self.column_of(where, claim.text, figure, source, chunk)
            if column is None:
                continue
            allowed = DURATIONS_FOR.get(claim.period_kind)
            if column.duration is None:
                if claim.period_kind != "point_in_time":
                    self.unverified(where, figure, "the column carries no duration; the claim says %s"
                                    % (claim.period_kind or "none"), column.label)
                    continue
            elif allowed is None or column.duration not in allowed:
                self.columns[1] += 1
                self.flag(where, "duration_mismatch", "%s sits in a %s column; the claim says %s"
                          % (figure.text, column.duration, claim.period_kind or "none"), column.label)
                continue
            self.columns[1] += 1
            if period_agrees(column, claim.period_end):
                self.columns[0] += 1
            elif not anchored:
                self.flag(where, "column_mismatch", "%s sits in the column for %s; the claim says %s"
                          % (figure.text, column.period_end, claim.period_end), column.label)
            else:
                self.note(where, "column_other_period", "%s sits in the column for %s"
                          % (figure.text, column.period_end), column.label)
            anchored = True

    def column_of(self, where: str, claim_text: str, figure: Number, source: Number,
                  chunk: Chunk) -> Column | None:
        """The column a figure sits in, or None with the reason the excerpt does
        not settle it. Every reason here is column_unverified: an unread column
        is never a mismatch."""
        if chunk.column_source != "parsed" or not chunk.columns:
            return self.unverified(where, figure, "the table's column shape was not established")
        if source.cells != len(chunk.columns):
            return self.unverified(where, figure, "the row '%s' has %d cells for %d columns"
                                   % (source.row, source.cells, len(chunk.columns)))
        if source.twins > 1:
            return self.unverified(where, figure, "%s fills %d cells of the row '%s'"
                                   % (figure.text, source.twins, source.row))
        column = next((c for c in chunk.columns if c.index == source.cell), None)
        if column is None:
            return self.unverified(where, figure, "the table has no column at position %d" % source.cell)
        if unnamed(words_of(source.row), claim_text):
            return self.unverified(where, figure, "the claim does not name the row '%s'" % source.row, source.row)
        # What is left of the label once its period words and digits are
        # gone is what the column is: "Standardized", "Bank, N.A.", "Graphics".
        distinguishing = {w for w in words_of(column.label)
                          if w not in PERIOD_WORDS and not w.replace(".", "").isdigit()}
        missing = unnamed(distinguishing, claim_text)
        if missing:
            return self.unverified(where, figure, "the label says %s and the claim does not"
                                   % ", ".join(missing), column.label)
        if column.period_end is None:
            return self.unverified(where, figure, "the column carries no period", column.label)
        return column

    def unverified(self, where: str, figure: Number, detail: str, source_string: str | None = None) -> None:
        self.columns_unverified += 1
        self.note(where, "column_unverified", "%s: %s" % (figure.text, detail), source_string)
        return None

    def check_units(self, where: str, chunk: Chunk, hits: list[Hit]) -> None:
        """A claim's scale word against the scale the excerpt states for the
        number. Percent and basis points are not scales and never convert."""
        for hit in hits:
            claimed, source = hit.figure.scale, hit.source
            scale, why = row_scale(chunk) if source.cell is not None else (
                source.scale, "the excerpt prints no scale word beside %s" % source.text)
            unread = ("the claim says %s for %s; %s" % (claimed, hit.figure.text, why) if claimed
                      else "the claim states no scale word for %s" % hit.figure.text)
            if claimed is None or scale is None:
                self.units_unchecked += 1
                self.note(where, "units_unchecked", unread, chunk.units)
                continue
            self.units[1] += 1
            if scale != claimed and not hit.converted:
                self.flag(where, "units_mismatch", "the claim says %s; the row is stated in %ss"
                          % (claimed, scale), chunk.units)
                continue
            self.units[0] += 1
            if hit.converted:
                self.note(where, "unit_converted", "%s in %ss reads as %s in %ss"
                          % (source.text, scale, hit.figure.text, claimed), source.text)

    def check_links(self, answer: Answer) -> None:
        """The claim ids a summary sentence or a table cell names must exist,
        and every figure it states must be written the same way in one of those
        claims, so the reader-facing text can never carry a number no claim
        carries and a rounded restatement is flagged rather than assumed."""
        claims = {c.id: c for c in answer.claims}
        rows = [("summary sentence %d" % i, s.text, s.claim_ids) for i, s in enumerate(answer.summary, 1)]
        rows += [("table cell %s / %s" % (row.dimension, cell.column), cell.text, cell.claim_ids)
                 for row in answer.table for cell in row.cells]
        for where, text, claim_ids in rows:
            unknown = [k for k in claim_ids if k not in claims]
            if not claim_ids or unknown:
                self.flag(where, "unlinked_sentence", "names no claim" if not claim_ids else
                          "names unknown claim %s" % ", ".join(unknown), text)
                self.unlinked.append(text)
            backing = [f for k in claim_ids if k in claims for f in figures_in(claims[k].text)]
            for figure in figures_in(text):
                if not any(figure.digits == b.digits and figure.percent == b.percent for b in backing):
                    self.flag(where, "unlinked_figure", "states %s; its claims state %s" % (
                        figure.text, ", ".join(b.text for b in backing) or "no figure"), text)

    def ungrounded_in(self, text: str) -> list[str]:
        """The years and the registry tickers a text names that the coverage
        block does not. A ticker is read only when the registry knows it, so
        an ordinary uppercase word is not mistaken for a company."""
        words = set(re.findall(r"[A-Za-z0-9]+", normalize(self.context.coverage)))
        text = normalize(text)
        return [y for y in GAP_YEAR_RE.findall(text) if y not in words] + \
               [t for t in GAP_TICKER_RE.findall(text) if t in self.known_tickers and t not in words]

    def check_gaps(self, answer: Answer) -> None:
        """A gap or a not-comparable note may name only what the coverage block
        holds; a ticker or a year outside it states something about the corpus
        that the corpus does not carry."""
        coverage = normalize(self.context.coverage)
        rows = [("gap %d" % i, gap, self.ungrounded_in(gap)) for i, gap in enumerate(answer.gaps, 1)]
        for i, n in enumerate(answer.not_comparable, 1):
            named = self.ungrounded_in(n.dimension + " " + n.reason)
            rows.append(("not comparable %d" % i, "%s: %s" % (n.dimension, n.reason),
                         named + [t for t in n.tickers if t not in coverage and t not in named]))
        for where, text, named in rows:
            if named:
                self.flag(where, "ungrounded_gap", "names %s, outside the coverage block" % ", ".join(named), text)

    def result(self) -> EvidenceChecks:
        return EvidenceChecks(
            quotes_located=tuple(self.quotes), figures_in_quote=tuple(self.figures),
            figures_unchecked=self.figures_unchecked, columns_matched=tuple(self.columns),
            columns_unverified=self.columns_unverified, units_matched=tuple(self.units),
            units_unchecked=self.units_unchecked, flags=self.flags, notes=self.notes, unlinked=self.unlinked)


def check(answer: Answer, context: Context, chunks_by_cid: dict[str, Chunk], plan: Plan | None = None,
          known_tickers: set[str] | None = None) -> EvidenceChecks:
    """The evidence checks for one Answer over the context it was written from.
    `plan` supplies the tickers in scope (the context's own stand in without
    it); `known_tickers` is the registry's, for the gap check."""
    in_scope = ({c["ticker"] for c in plan.companies} if plan else {c.ticker for c in chunks_by_cid.values()})
    checker = Checker(context, chunks_by_cid, in_scope, known_tickers or set())
    for claim in answer.claims:
        checker.check_claim(claim)
    checker.check_links(answer)
    checker.check_gaps(answer)
    return checker.result()
