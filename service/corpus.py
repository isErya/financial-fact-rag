"""Parse the filing corpus: one zip member in, one Filing out.

Flow for a member: normalize the text, read the header block, find the cover
page, strip page furniture from the body, cut the signatures block, locate the
item headings, then the notes inside the financial statements. Fiscal labels
are assigned afterwards across the whole corpus because the fiscal year-end
month of a company comes from its newest annual report, not from the file at
hand.

Where each decision is made:
  normalize        every comparison in later milestones runs on text that went
                   through this same function, so it lives here and nowhere else
  body_start       the cover marker is the one string all 246 files share
  strip_noise      page furniture rules, each with a name so the report can
                   count what was removed
  cut_signatures   runs before detection, so nothing after the signatures
                   block can become a heading
  detect_sections  the three candidate forms ("Item N" headings, title-only
                   headings, the auditor's report as the start of Item 8),
                   the TOC region, the rejection rules, the greedy choice
                   with span validation, and the Item 15 hand-off for filers
                   who put the statements there
  detect_notes     note heading styles, the footnote filter, and the
                   ascending-number chain
  assign_fiscal_labels  the only place fiscal years and quarters are computed

The "Quarter:" header line and the quarter tag in the file name are the
calendar quarter of the period end, which is the wrong axis for a company with
a fiscal year ending in September. parse_header drops that line on purpose and
nothing here reads the file name beyond using it as an identifier.

Failure policy: a member with no cover marker or no recoverable period end
raises, because a silently mislabeled filing would poison every answer built
on it. A missing section is not an error: the item is simply absent from
Filing.sections and the report shows the gap.
"""

import datetime as dt
import re
import unicodedata
import zipfile
from collections import Counter

from models import Filing, Note, Section

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_CHAR_MAP = [
    ("\u00a0", " "),  # no-break space
    ("\u2009", " "),  # thin space
    ("\u202f", " "),  # narrow no-break space
    ("\u200b", ""),  # zero-width space
    ("\u00ad", ""),  # soft hyphen
    ("\u2018", "'"),
    ("\u2019", "'"),
    ("\u201c", '"'),
    ("\u201d", '"'),
    ("\u2013", "-"),  # en dash
    ("\u2014", "-"),  # em dash
    ("\u2212", "-"),  # minus sign
]
_SPACE_RUN_RE = re.compile(r"[ \t]{2,}")


def normalize(text: str) -> str:
    """Fold the Unicode variants the filings use into plain ASCII forms.

    NFKC first so ligatures and full-width forms collapse, then the explicit
    map for the characters NFKC leaves alone (curly quotes, dashes, the
    no-break spaces that sit inside dates on cover pages). Runs of spaces and
    tabs collapse within a line; newlines stay because table rows and
    headings are line-based.
    """
    text = unicodedata.normalize("NFKC", text)
    for old, new in _CHAR_MAP:
        text = text.replace(old, new)
    return _SPACE_RUN_RE.sub(" ", text)


# ---------------------------------------------------------------------------
# Header block and cover page
# ---------------------------------------------------------------------------

HEADER_RULE = "=" * 60
COVER_RE = re.compile(r"securities\s+and\s+exchange\s+commission", re.I)


def parse_header(text: str) -> dict:
    """Read the "Key: value" lines above the rule of sixty "=" characters.

    The "Quarter" line is dropped here so no later code can read it by
    accident; it is a calendar quarter and the fiscal quarter is computed.
    """
    rule = text.find(HEADER_RULE)
    if rule < 0:
        raise ValueError("header rule not found")
    header = {}
    for line in text[:rule].split("\n"):
        key, sep, value = line.partition(":")
        if not sep or key.strip() == "Quarter":
            continue
        header[key.strip()] = value.strip()
    return header


def body_start(text: str) -> int:
    """Index of the cover page, the first text after the inline-XBRL preamble."""
    match = COVER_RE.search(text)
    if match is None:
        raise ValueError("cover marker not found")
    return match.start()


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------

# A page break lands as "Table of Contents" glued into the running text,
# sometimes with the page number in front: "10Table of ContentsCompliance",
# once misspelled "Table of ContentItem 2". It is replaced by a line break,
# so whatever followed the page break starts a line the way a heading would.
# The lookahead refuses a lowercase letter so the optional "s" cannot be
# given up to match "Table of Content" inside "Table of Contents |".
GLUED_TOC_RE = re.compile(r"\d{0,3}(?i:table of contents?)(?=[A-Z0-9(]|\s*\n)")

# Footer shapes seen in the corpus, in the order they are applied. Inline
# forms are replaced with a space; line forms remove the whole line.
_FOOTER_INLINE = [
    # "Apple Inc. | 2024 Form 10-K | 29", "Apple Inc. | Q1 2026 Form 10-Q | 12"
    ("company | year Form 10-K | page",
     re.compile(r"(?:[A-Z][A-Za-z&.,']*\s){1,4}\|\s*(?:Q\d\s+)?\d{4}\s+Form\s+10-[KQ]\s*\|\s*\d{1,3}\s*\|?")),
    # "McDonald's Corporation 2024 Annual Report    6"
    ("company year Annual Report page",
     re.compile(r"(?:[A-Z][A-Za-z&.,']*\s){1,4}\d{4}\s+Annual\s+Report\s+\d{1,3}(?=\s|[A-Z]|$)")),
]
_FOOTER_LINE = [
    # "Table of Contents |", "Table of Contents | Part | Item | Page": page-top
    # links and TOC header rows, never headings or data
    ("Table of Contents | ...",
     re.compile(r"^\s*(?i:table of contents)\s*\|[ \t|A-Za-z]*$")),
    # "| 29 | December 2025 Form 10-K", "28 |  | Goldman Sachs 2024 Form 10-K"
    ("page | month year Form 10-K",
     re.compile(r"^\s*\|?\s*\d{1,3}\s*\|\s*\|?\s*[A-Za-z ]{0,40}\d{4}\s+Form\s+10-[KQ]\s*\|?\s*$")),
    # "December 2025 Form 10-K | 12 |", "Goldman Sachs 2024 Form 10-K |  | 45"
    ("month year Form 10-K | page",
     re.compile(r"^\s*[A-Za-z ]{0,40}\d{4}\s+Form\s+10-[KQ]\s*\|\s*\|?\s*\d{1,3}\s*\|?\s*$")),
    # "JPMorgan Chase & Co./2025 Form 10-K |  | 45"
    ("company/year Form 10-K | page",
     re.compile(r"^[^\n|]{0,40}/\d{4}\s+Form\s+10-[KQ]\s*\|\s*\|?\s*\d{1,3}\s*$")),
    # "2025 Annual Report | 41", "2025 Annual Report on Form 10-K | 18 |"
    ("year Annual Report | page",
     re.compile(r"^\s*\d{4}\s+Annual\s+Report(?:\s+on\s+Form\s+10-K)?\s*\|\s*\d{1,3}\s*\|?\s*$")),
    # "GECC 2014 FORM 10-K  140"
    ("CODE year FORM 10-K page",
     re.compile(r"^\s*[A-Z ]{2,20}\d{4}\s+FORM\s+10-K\s+\d{1,3}\s*$")),
    # "27Bank of America |  |"
    ("pagecompany | |",
     re.compile(r"^\s*\d{1,3}[A-Z][A-Za-z ]{2,40}\|(?:\s*\|)*\s*$")),
    # bare page numbers: "35", "8 |", "| 10", "K-23"
    ("bare page number",
     re.compile(r"^\s*\|?\s*(?:[A-Z]-)?\d{1,3}\s*\|?\s*$")),
]

# Running headers repeat on every page. A line is dropped only when it repeats
# at least this often AND matches one of the header shapes AND has no digits
# beyond a trailing page number. Table header rows repeat too ("| 2024 | 2023",
# "| Year Ended") and must survive, which the digit and keep rules guarantee.
# The first copy of a running header is the real heading it was copied from
# ("Management's Discussion and Analysis" at the top of the MD&A), so it
# stays; "Table of Contents" lines are page-top links and all of them go.
RUNNING_HEADER_MIN_REPEATS = 5
TOC_LINK_RE = re.compile(r"table of contents", re.I)
RUNNING_HEADER_RE = re.compile(
    r"table of contents|form 10-[kq]|notes to consolidated|management's discussion|^\d{1,3}$",
    re.I,
)
KEEP_LINE_RE = re.compile(
    r"\d{4}|january|february|march|april|may|june|july|august|september|october"
    r"|november|december|months ended",
    re.I,
)
_TRAILING_PAGE_RE = re.compile(r"\s*\|?\s*\d{1,3}\s*\|?\s*$")


def strip_noise(body: str, stats: dict | None = None) -> str:
    """Remove page furniture and leave every table row and heading in place.

    `stats`, when given, receives per-rule removal counts for the report.
    """
    counts: Counter = Counter()

    body, n = GLUED_TOC_RE.subn("\n", body)
    counts["glued Table of Contents"] += n
    for name, rx in _FOOTER_INLINE:
        body, n = rx.subn(" ", body)
        counts["footer: " + name] += n

    lines = body.split("\n")
    keys = Counter(
        _TRAILING_PAGE_RE.sub("", line).strip() for line in lines if len(line) < 120
    )
    kept = []
    seen: set[str] = set()
    for line in lines:
        dropped = False
        for name, rx in _FOOTER_LINE:
            if rx.match(line):
                counts["footer: " + name] += 1
                dropped = True
                break
        if dropped:
            continue
        if len(line) < 120 and not KEEP_LINE_RE.search(line):
            key = _TRAILING_PAGE_RE.sub("", line).strip()
            if (
                key
                and keys[key] >= RUNNING_HEADER_MIN_REPEATS
                and not any(ch.isdigit() for ch in key)
                and RUNNING_HEADER_RE.search(line.strip())
            ):
                if key in seen or TOC_LINK_RE.search(key):
                    counts["running header lines"] += 1
                    continue
                seen.add(key)
        kept.append(line)
    body = "\n".join(kept)
    body = _SPACE_RUN_RE.sub(" ", body)
    if stats is not None:
        for name, n in counts.items():
            stats[name] = stats.get(name, 0) + n
    return body


SIGNATURES_RE = re.compile(r"(?<![A-Za-z])SIGNATURES?(?![a-z])")


def cut_signatures(body: str) -> str:
    """Drop the signatures block at the end of a filing.

    The word also appears in the table of contents, so only the last
    occurrence counts and only when it sits in the final fifth of the body.
    Runs before section detection, which is how a candidate after the
    signatures block is kept out of the running.
    """
    last = None
    for match in SIGNATURES_RE.finditer(body):
        last = match.start()
    if last is not None and last >= 0.8 * len(body):
        return body[:last]
    return body


# ---------------------------------------------------------------------------
# Period end
# ---------------------------------------------------------------------------

_MONTHS = {
    name: i
    for i, name in enumerate(
        ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"],
        start=1,
    )
}
# "for the fiscal year ended September 28, 2024", also glued ("endedDecember")
# and with the space after the comma missing ("June 30,2025").
COVER_PERIOD_RE = re.compile(
    r"for\s+the\s+(?:fiscal\s+)?(?:year|quarterly\s+period|quarter|period)\s+ended\s*:?\s*"
    r"([A-Za-z]+)\s*(\d{1,2}),?\s*(\d{4})",
    re.I,
)
URL_DATE_RE = re.compile(r"(\d{8})[^/]*\.htm")
COVER_SEARCH_CHARS = 20000


def period_end(header: dict, body: str, url: str) -> tuple[str, str]:
    """Return (YYYY-MM-DD, source) with source in header, cover, url."""
    value = header.get("Report Period")
    if value:
        dt.date.fromisoformat(value)
        return value, "header"
    match = COVER_PERIOD_RE.search(body[:COVER_SEARCH_CHARS])
    if match:
        month = _MONTHS.get(match.group(1).lower())
        if month:
            day = dt.date(int(match.group(3)), month, int(match.group(2)))
            return day.isoformat(), "cover"
    match = URL_DATE_RE.search(url)
    if match:
        raw = match.group(1)
        day = dt.date(int(raw[:4]), int(raw[4:6]), int(raw[6:]))
        return day.isoformat(), "url"
    raise ValueError("period end not found in header, cover, or url")


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

# Canonical items in filing order: (item, title, title regex). Each regex
# matches the start of the heading and an optional tail, so the title-only
# form can require that the whole heading is followed by a capital letter or
# a line end, while the "Item" form accepts any filer wording after it.
_MDA = (r"Management'?s\s+Discussion\s+and\s+Analysis"
        r"(?:\s+of\s+Financial\s+Condition\s+and\s+Results\s+of\s+Operations)?")
_QQ = r"Quantitative\s+and\s+Qualitative\s+Disclosures?\s+About\s+(?:Market\s+)?Risk"
_FS_SUPP = r"(?:\w+\s+){0,3}Financial\s+Statements\s*and\s+Supplementa(?:ry|l)\s+Data"

ITEMS_10K = [
    ("1", "Business", r"Business(?:\s+Description)?"),
    ("1A", "Risk Factors", r"Risk\s+Factors"),
    ("1B", "Unresolved Staff Comments", r"Unresolved\s+Staff\s+Comments"),
    ("1C", "Cybersecurity", r"Cybersecurity"),
    ("2", "Properties", r"Properties"),
    ("3", "Legal Proceedings", r"Legal\s+Proceedings"),
    ("4", "Mine Safety Disclosures", r"Mine\s+Safety\s+Disclosures?"),
    ("5", "Market for Registrant's Common Equity",
     r"Market\s+for\s+(?:the\s+)?(?:Registrant'?s|Company'?s|Our)\s+Common\s+(?:Equity|Stock)"
     r"(?:,\s+Related\s+(?:Stockholder|Shareholder)\s+Matters,?\s+and\s+Issuer\s+Purchases"
     r"\s+of\s+Equity\s+Securities)?"),
    ("6", "Reserved / Selected Financial Data",
     r"\[?\s*Reserved\s*\]?|Selected\s+(?:Consolidated\s+)?Financial\s+Data"),
    ("7", "Management's Discussion and Analysis", _MDA),
    ("7A", "Quantitative and Qualitative Disclosures About Market Risk", _QQ),
    ("8", "Financial Statements and Supplementary Data", _FS_SUPP),
    ("9", "Changes in and Disagreements with Accountants",
     r"Changes\s+in\s+and\s+Disagreements(?:\s+with\s+Accountants\s+on\s+Accounting"
     r"\s+and\s+Financial\s+Disclosure)?"),
    ("9A", "Controls and Procedures", r"Controls\s+and\s+Procedures"),
    ("9B", "Other Information", r"Other\s+Information"),
    ("9C", "Disclosure Regarding Foreign Jurisdictions",
     r"Disclosure\s+Regarding\s+Foreign\s+Jurisdictions(?:\s+that\s+Prevent\s+Inspections)?"),
    ("10", "Directors, Executive Officers and Corporate Governance",
     r"Directors,?\s+Executive\s+Officers(?:,?\s+and\s+Corporate\s+Governance)?"),
    ("11", "Executive Compensation", r"Executive\s+Compensation"),
    ("12", "Security Ownership",
     r"Security\s+Ownership(?:\s+of\s+Certain\s+Beneficial\s+Owners\s+and\s+Management"
     r"\s+and\s+Related\s+(?:Stockholder|Shareholder)\s+Matters)?"),
    ("13", "Certain Relationships and Related Transactions",
     r"Certain\s+Relationships(?:\s+and\s+Related\s+Transactions,?\s+and\s+Director"
     r"\s+Independence)?"),
    ("14", "Principal Accountant Fees and Services",
     r"Principal\s+Account(?:ant|ing)\s+Fees(?:\s+and\s+Services)?"),
    ("15", "Exhibits and Financial Statement Schedules",
     r"Exhibits?(?:,?\s+(?:and\s+)?Financial\s+Statement\s+Schedules?)?"),
    ("16", "Form 10-K Summary", r"Form\s+10-K\s+Summary"),
]

# 10-Q items carry their part; item strings are "I.1", "II.1A".
ITEMS_10Q = [
    ("I", "1", "Financial Statements",
     r"(?:Condensed\s+)?(?:Consolidated\s+)?(?:Interim\s+)?Financial\s+Statements"),
    ("I", "2", "Management's Discussion and Analysis", _MDA),
    ("I", "3", "Quantitative and Qualitative Disclosures About Market Risk", _QQ),
    ("I", "4", "Controls and Procedures", r"Controls\s+and\s+Procedures"),
    ("II", "1", "Legal Proceedings", r"Legal\s+Proceedings"),
    ("II", "1A", "Risk Factors", r"Risk\s+Factors"),
    ("II", "2", "Unregistered Sales of Equity Securities",
     r"Unregistered\s+Sales(?:\s+of\s+Equity\s+Securities\s+and\s+Use\s+of\s+Proceeds)?"),
    ("II", "3", "Defaults Upon Senior Securities", r"Defaults?\s+Upon\s+Senior\s+Securities"),
    ("II", "4", "Mine Safety Disclosures", r"Mine\s+Safety\s+Disclosures?"),
    ("II", "5", "Other Information", r"Other\s+Information"),
    ("II", "6", "Exhibits", r"Exhibits?"),
]

# Span bounds in characters for the sections that answers depend on. A
# candidate whose span falls outside is a cross-reference or a TOC row, so it
# is dropped and the next candidate for the same item is tried.
SPAN_BOUNDS = {
    ("10-K", "1A"): (5_000, 400_000),
    ("10-K", "7"): (10_000, 600_000),
    ("10-K", "8"): (10_000, 1_500_000),
    ("10-Q", "I.1"): (10_000, 1_500_000),
    ("10-Q", "I.2"): (10_000, 600_000),
}

# The items later milestones retrieve from; the rest only delimit them.
CORE_ITEMS = {"10-K": ["1A", "7", "8"], "10-Q": ["I.1", "I.2", "II.1A"]}

# Every "Item 1A" / "ITEM 7" token in the body, found in one pass. Headings
# are often glued to the preceding text ("INFORMATIONItem 1.", a running
# header's "AnalysisItem 7."), so nothing is required before the word; the
# capitalized word plus a number is specific enough. The number must end the
# token so "Item 1" cannot be read out of "Item 1A" or "Item 10".
ITEM_TOKEN_RE = re.compile(r"(?:Item|ITEM)\s*(\d{1,2}[A-Ca-c]?)(?![0-9A-Za-z])")

# A table-of-contents page reference: "| 9", "| 9-31", "| K-24", "| Pages37-51".
PAGE_REF_RE = re.compile(r"\|\s*(?:Pages?\s*)?(?:[A-Z]-)?\d{1,3}(?:\s*-\s*\d{1,3})?\s*(?:\||\n|$)")
PAGE_REF_WINDOW = 80

# Words that introduce a cross-reference ("see Item 1A", "under Part I,
# Item 7"), checked against the text right before a candidate.
LEAD_IN_RE = re.compile(
    r"(?:\bsee|refer\s+to|\bunder|\bwith|described\s+in|discussed\s+in"
    r"|\bin|\bof|\band|\bto|\bthe)"
    r"\s*[:,]?\s*[\"'(]?\s*(?:part\s+[iv1-4]+\s*[,.]?\s*)?$",
    re.I,
)
LEAD_IN_WINDOW = 40

# What follows a cross-reference's title and never follows a heading:
# "Item 1A. Risk Factors of the 2025 Form 10-K", "Item 8: Financial
# Statements and Supplementary Data." in quotes, "Risk Factors above".
XREF_TAIL_RE = re.compile(
    r"\.?\s*(?:[,;\")]|(?:of|in|to|on|for)\s+(?:this|the|our|its|Part)\b"
    r"|(?:above|below|herein|beginning|contained|included)\b)"
)

TOC_MIN_ITEMS = 5
TOC_MAX_GAP = 4000

# "Part II" heading that starts the second half of a 10-Q. The TOC row is
# excluded by the page-reference rule like any other candidate.
PART_II_RE = re.compile(r"(?<![a-z])PART\s+II\b[\s.:\-|]*OTHER\s+INFORMATION", re.I)

# The audited statements block opens with the auditor's report, whatever
# item heading (if any) the filer put in front of it. Used as the last-resort
# Item 8 candidate and to tell whether Item 15 holds the statements.
AUDITOR_REPORT_RE = re.compile(
    r"(?:(?<=\n)|(?<=\.)|(?<=\|)|(?<=\| )|(?<=\d)|(?<=\d ))(?=[A-Z])"
    r"(?i:Report\s+of\s+Independent\s+Registered\s+Public\s+Accounting\s+Firm)"
    r"(?=\s?[A-Z(]|\s*\|?\s*\n)"
)
STATEMENTS_RE = re.compile(r"(?i)consolidated\s+(?:statements?|balance\s+sheets?)")
STATEMENTS_WINDOW = 20000


def _after_item_re(title: str) -> re.Pattern:
    """Separator plus title, matched right after an "Item N" token."""
    return re.compile(r"[.:|\-\s]*(?i:" + title + r")")


def _title_only_re(specs: list[tuple]) -> re.Pattern:
    """One regex for every title-only heading in a form.

    For filers that omit the "Item" prefix. The heading must start a line or
    follow a period, begin with a capital, and run straight into a capital
    letter or the end of the line, which is how a heading glued to its first
    sentence looks ("Risk FactorsFor a discussion"). Group g<k> names the
    k-th spec.
    """
    alts = "|".join("(?P<g%d>%s)" % (k, spec[3]) for k, spec in enumerate(specs))
    return re.compile(r"(?<=[\n.])(?=[A-Z])(?i:" + alts + r")(?=[A-Z]|\s*\|?\s*\n)")


def _line_title_re(specs: list[tuple]) -> re.Pattern:
    """Titles at a line start, used only to recognize TOC rows."""
    alts = "|".join("(?P<g%d>%s)" % (k, spec[3]) for k, spec in enumerate(specs))
    return re.compile(r"(?m)^\s*(?=[A-Z])(?i:" + alts + r")")


def _page_ref_near(body: str, pos: int) -> re.Match | None:
    """The page reference within PAGE_REF_WINDOW chars after pos, or None."""
    return PAGE_REF_RE.search(body, pos, min(len(body), pos + PAGE_REF_WINDOW))


def _page_ref_on_line(body: str, pos: int) -> bool:
    """Page reference after pos on the same line.

    A real heading can be followed by an index table on the next lines
    ("Item 8. Financial Statements\\n| | Page\\nConsolidated Statements | 63"),
    so a reference on a later line does not make the heading a TOC row.
    """
    newline = body.find("\n", pos)
    limit = pos + PAGE_REF_WINDOW if newline < 0 else min(newline + 1, pos + PAGE_REF_WINDOW)
    return PAGE_REF_RE.search(body, pos, limit) is not None


def _has_lead_in(body: str, pos: int) -> bool:
    return LEAD_IN_RE.search(body[max(0, pos - LEAD_IN_WINDOW):pos]) is not None


def toc_region(body: str, specs: list[tuple], tokens: list[tuple]) -> tuple[int, int]:
    """Locate the table of contents as a cluster of page-referenced headings.

    A hit is an "Item N" token or a line-start title with a page reference
    close behind it. The region starts at the first place where at least
    TOC_MIN_ITEMS distinct items hit within TOC_MAX_GAP chars and runs
    through the page reference of the last hit in that cluster. Files with
    no such cluster get (0, 0).
    """
    hits = []
    for start, end, number in tokens:
        ref = _page_ref_near(body, end)
        if ref:
            hits.append((start, ref.end(), "item:" + number))
    for match in _line_title_re(specs).finditer(body):
        ref = _page_ref_near(body, match.end())
        if ref:
            hits.append((match.start(), ref.end(), "title:" + match.lastgroup))
    hits.sort()
    for i in range(len(hits)):
        window = {key for pos, _end, key in hits[i:] if pos - hits[i][0] <= TOC_MAX_GAP}
        if len(window) < TOC_MIN_ITEMS:
            continue
        j = i
        while j + 1 < len(hits) and hits[j + 1][0] - hits[j][0] <= TOC_MAX_GAP:
            j += 1
        return hits[i][0], hits[j][1]
    return 0, 0


def _survives(body: str, pos: int, end: int, toc: tuple[int, int]) -> bool:
    """The TOC, page-reference, and cross-reference rejection rules.

    A cross-reference is caught from either side: a lead-in word before it
    ("see Item 1A") or the sentence continuing after the title ("Risk
    Factors of the 2025 Form 10-K").
    """
    if toc[0] <= pos < toc[1]:
        return False
    if _page_ref_on_line(body, end):
        return False
    if _has_lead_in(body, pos):
        return False
    if XREF_TAIL_RE.match(body, end):
        return False
    return True


def _choose(body: str, form: str, order: list[str], cands: dict[str, list[int]],
            part_of: dict[str, str | None], part2: int | None) -> list[tuple[str, int]]:
    """Greedy choice in canonical order with span validation.

    A 10-K is read in canonical order: each item takes the first surviving
    candidate after the previous accepted heading. A delimiting item may not
    jump past the next core heading: Morgan Stanley files Items 1B through 5
    after the financial statements, and accepting that 1B would push Items 7
    and 8 off the end.

    A 10-Q is read per part. Bank of America files Part I as Items 2, 3, 4
    and then 1, so within a part each item takes its first surviving
    candidate and the headings are sorted by position afterwards; the Part II
    marker keeps the two parts from mixing.

    Then measure each bounded span; the first span outside its bounds rejects
    that candidate and the whole pass repeats. Every pass rejects one
    candidate, so the loop ends. A 10-K Item 8 that only points at Item 15
    keeps its short span when the Item 15 span holds the auditor's report;
    detect_sections then hands the Item 15 span to Item 8.
    """
    core = CORE_ITEMS[form]
    rejected: set[tuple[str, int]] = set()

    def first_surviving(item: str, prev: int) -> int | None:
        for pos in cands[item]:
            if pos <= prev or (item, pos) in rejected:
                continue
            if part2 is not None and part_of[item] == "I" and pos >= part2:
                continue
            if part2 is not None and part_of[item] == "II" and pos < part2:
                continue
            return pos
        return None

    while True:
        chosen: list[tuple[str, int]] = []
        prev = -1
        for k, item in enumerate(order):
            if form == "10-Q":
                prev = part2 if part_of[item] == "II" and part2 is not None else -1
            pos = first_surviving(item, prev)
            if pos is None:
                continue
            if item not in core:
                limits = [first_surviving(c, prev) for c in order[k + 1:] if c in core]
                limit = next((x for x in limits if x is not None), None)
                if limit is not None and pos >= limit:
                    continue
            chosen.append((item, pos))
            prev = pos
        chosen.sort(key=lambda pair: pair[1])
        spans = _spans(chosen, len(body))
        bad = None
        for item, pos in chosen:
            lo, hi = SPAN_BOUNDS.get((form, item), (0, len(body) + 1))
            start, end = spans[item]
            if item == "8" and end - start < lo and _item_15_holds_statements(body, spans, lo):
                continue
            if not lo <= end - start <= hi:
                bad = (item, pos)
                break
        if bad is None:
            return chosen
        rejected.add(bad)


def _spans(chosen: list[tuple[str, int]], body_len: int) -> dict[str, tuple[int, int]]:
    out = {}
    for k, (item, pos) in enumerate(chosen):
        end = chosen[k + 1][1] if k + 1 < len(chosen) else body_len
        out[item] = (pos, end)
    return out


def _item_15_holds_statements(body: str, spans: dict[str, tuple[int, int]], lo: int) -> bool:
    if "15" not in spans:
        return False
    start, end = spans["15"]
    return end - start >= lo and AUDITOR_REPORT_RE.search(body, start, end) is not None


def _specs(form: str) -> list[tuple]:
    """(item, part, title, title regex) for every canonical item of a form."""
    if form == "10-K":
        return [(item, None, title, rx) for item, title, rx in ITEMS_10K]
    return [(part + "." + item, part, title, rx) for part, item, title, rx in ITEMS_10Q]


def _item_tokens(body: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(1).upper()) for m in ITEM_TOKEN_RE.finditer(body)]


def toc_span(body: str, form: str) -> tuple[int, int]:
    """The table-of-contents region detect_sections skips; (0, 0) if none."""
    return toc_region(body, _specs(form), _item_tokens(body))


def detect_sections(body: str, form: str) -> list[Section]:
    """Split the body into canonical items. Missing items are absent."""
    specs = _specs(form)
    order = [s[0] for s in specs]
    part_of = {s[0]: s[1] for s in specs}
    titles = {s[0]: s[2] for s in specs}

    tokens = _item_tokens(body)
    toc = toc_region(body, specs, tokens)

    part2 = None
    if form == "10-Q":
        for match in PART_II_RE.finditer(body):
            if _survives(body, match.start(), match.end(), toc):
                part2 = match.start()
                break

    # Candidate lists are scanned in order, so the "Item N" form comes first
    # and the fallbacks are only reached when no "Item N" candidate yields a
    # valid span: the title-only form (10-K filers that omit "Item"), then for
    # Item 8 the auditor's report that opens the statements wherever the
    # filer put them (after Item 16 as a "Financial Section", or inside 7A).
    cands: dict[str, list[int]] = {item: [] for item in order}
    for item, _part, _title, title_rx in specs:
        number = item.split(".")[-1]
        after = _after_item_re(title_rx)
        for start, end, token_number in tokens:
            if token_number != number:
                continue
            match = after.match(body, end)
            if match and _survives(body, start, match.end(), toc):
                cands[item].append(start)
    if form == "10-K":
        for match in _title_only_re(specs).finditer(body):
            item = specs[int(match.lastgroup[1:])][0]
            if _survives(body, match.start(), match.end(), toc):
                cands[item].append(match.start())
        for match in AUDITOR_REPORT_RE.finditer(body):
            followed = STATEMENTS_RE.search(body, match.end(), match.end() + STATEMENTS_WINDOW)
            if followed and _survives(body, match.start(), match.end(), toc):
                cands["8"].append(match.start())

    chosen = _choose(body, form, order, cands, part_of, part2)

    sections = []
    first = chosen[0][1] if chosen else len(body)
    sections.append(_section(body, None, "COVER", "Cover", 0, first))
    for k, (item, pos) in enumerate(chosen):
        end = chosen[k + 1][1] if k + 1 < len(chosen) else len(body)
        sections.append(_section(body, part_of[item], item, titles[item], pos, end))
    if form == "10-K":
        _statements_under_item_15(body, sections)
    return sections


def _statements_under_item_15(body: str, sections: list[Section]) -> None:
    """Give Item 8 the Item 15 span when Item 8 only points there.

    Some filers put the audited statements and notes after the Item 15
    heading and leave a one-line Item 8. Retrieval asks for Item 8, so the
    long span is labeled 8 and the pointer line is dropped.
    """
    by_item = {s.item: s for s in sections}
    stub, exhibits = by_item.get("8"), by_item.get("15")
    lo = SPAN_BOUNDS[("10-K", "8")][0]
    spans = {s.item: (s.start, s.end) for s in sections}
    if stub and exhibits and stub.end - stub.start < lo and _item_15_holds_statements(body, spans, lo):
        exhibits.item = "8"
        exhibits.title = "Financial Statements and Supplementary Data (filed under Item 15)"
        sections.remove(stub)


POINTER_STUB_MAX_CHARS = 2500
# A pointer stub is a short section that sends the reader elsewhere. Most
# name the annual report as "Form 10-K"; UNH writes "our 2021 10-K" and MCD
# once writes 'refer to the "Risk Factors" section in Part I, Item 2 of this
# report', so the rule accepts those two wordings as well. The wording is
# kept this narrow because "of this report" alone also matches every
# Exhibits list ("filed as part of this report"), which is short and points
# nowhere.
POINTER_STUB_RE = re.compile(r"Form 10-K|\b(?:19|20)\d{2} 10-K\b|refer to[^.]{0,120}of this report")


def _section(body: str, part: str | None, item: str, title: str, start: int, end: int) -> Section:
    text = body[start:end]
    stub = len(text) < POINTER_STUB_MAX_CHARS and POINTER_STUB_RE.search(text) is not None
    return Section(part=part, item=item, title=title, start=start, end=end,
                   is_pointer_stub=stub, notes=[])


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

# Heading styles, each capturing the note number. The title starts at the
# match end. Whitespace inside a heading never crosses a line, so the
# patterns use "[ \t]"; that keeps "Note 4.\n\nIntangible" (a cross-reference
# ending a sentence) from reading as a heading.
NOTE_STYLES = [
    # "Note 2 - Revenue", "Note 2. Revenue", "NOTE 2: REVENUE", also glued to
    # the block heading ("Financial StatementsNote 1. Description")
    ("Note N - Title", re.compile(r"(?:Note|NOTE)[ \t]+(\d{1,2})[ \t]*[-.:][ \t]*(?=[A-Z])")),
    # "Note 2 Revenue" or "NOTE 1SUMMARY" with no separator; the capital
    # refuses "Note 2 to the"
    ("Note N Title", re.compile(r"(?:Note|NOTE)[ \t]+(\d{1,2})[ \t]*(?=[A-Z])")),
    # "2. Significant Accounting Policies" at a line start, or glued to the
    # end of the previous paragraph or block heading: "Statements1. Summary",
    # "policies.2. Revenues", "respectively. 24. Accumulated"
    ("N. Title", re.compile(r"(?:(?<=\n)[ \t]*|(?<=[A-Za-z).])[ \t]?)(\d{1,2})\.[ \t]*(?=[A-Z])")),
    # "(1)Significant accounting policies" at a line start or glued
    ("(N) Title", re.compile(r"(?:(?<=\n)[ \t]*|(?<=[A-Za-z)]))\((\d{1,2})\)[ \t]*(?=[A-Z])")),
    # "1Description of the Business", number glued to a title-case word
    ("NTitle", re.compile(r"(?:(?<=\n)[ \t]*|(?<=[a-z).])[ \t]?)(\d{1,2})(?=[A-Z][a-z]{2})")),
]
# "Level 2. Inputs", "Statement 3. Trade receivables", "December 31. Fiscal"
# have the shape of a glued numbered heading and are not one. Singular
# forms only: "Notes to Consolidated Financial Statements1. Summary" is the
# most common real heading glue.
NOTE_NOT_HEADING_RE = re.compile(
    r"(?i)\b(?:level|statement|note|schedule|item|part|section|agreement|tier|phase"
    r"|step|topic|asc|asu|page|chapter|january|february|march|april|may|june|july|august"
    r"|september|october|november|december)[ \t]?$"
)
# Table footnotes look like glued headings ("1. Includes $568 million of
# proceeds", "(1)The estimated increase is after income taxes") and sit right
# before the real Note 1. A heading title has no digits, dollar or percent
# signs, and does not open like a sentence.
NOTE_TITLE_BAD_RE = re.compile(
    r"[\d$%]|^(?:The|This|These|We|Our|As|In|For|At|On|During|See|Refer|Includes?"
    r"|Excludes?|Represents?|Reflects?|Amounts?|Consists?|Based|Primarily|Net)\b"
)
NOTE_MAX_GAP = 2
NOTE_TITLE_MAX = 100
# A title ends at a line or cell break, a colon, the glue point where the
# first sentence starts ("Income TaxesEuropean"), or a sentence opener.
_TITLE_END_RE = re.compile(
    r"\n|\||:|(?<=[a-z\)])(?=[A-Z])|(?<=[A-Z]{2})(?=[A-Z][a-z])"
    r"|\s(?=(?:The|We|Our|In|As|On|At|For|During|This|These)\s)"
)
NOTE_LEAD_IN_RE = re.compile(
    r"(?:\bsee|refer\s+to|\bunder|\bin|\bof|\band|\bto|\bwith|\bfrom|\bper)\s*[\"'(]?\s*$", re.I
)


def _note_title(body: str, pos: int) -> str:
    raw = body[pos:pos + NOTE_TITLE_MAX]
    match = _TITLE_END_RE.search(raw)
    title = raw[:match.start()] if match else raw
    return title.strip(" .-")


def detect_notes(body: str, section: Section) -> list[Note]:
    """Notes inside one section, as an ascending chain of numbered headings.

    Every style's matches are pooled and sorted. Dropped before the chain:
    TOC-style rows (a page reference on the same line), cross-references (a
    lead-in word close in front), numbers that count something other than a
    note ("Level 2."), and table footnotes (a title with digits or a sentence
    opener). The chain starts at the first heading numbered 1 or 2 and each
    later heading must raise the number by at most NOTE_MAX_GAP, so a stray
    "Note 17" quoted in the text cannot skip the real notes.
    """
    found = []
    for _name, rx in NOTE_STYLES:
        for match in rx.finditer(body, section.start, section.end):
            number = int(match.group(1))
            pos = match.start(1)
            before = body[max(section.start, match.start() - LEAD_IN_WINDOW):match.start()]
            title = _note_title(body, match.end())
            if _page_ref_on_line(body, match.end()):
                continue
            if NOTE_LEAD_IN_RE.search(before) or NOTE_NOT_HEADING_RE.search(before):
                continue
            if NOTE_TITLE_BAD_RE.search(title):
                continue
            found.append((pos, number, title))
    found.sort()

    chain: list[Note] = []
    last = None
    for pos, number, title in found:
        if last is None:
            if number > 2:
                continue
        elif not last < number <= last + NOTE_MAX_GAP:
            continue
        if chain:
            chain[-1].end = pos
        chain.append(Note(number=number, title=title, start=pos, end=section.end))
        last = number
    return chain


# ---------------------------------------------------------------------------
# Filings
# ---------------------------------------------------------------------------

NOTE_SECTIONS = {"10-K": "8", "10-Q": "I.1"}


def parse_filing(name: str, raw_text: str, stats: dict | None = None) -> Filing:
    """Parse one zip member. Fiscal fields are zero until assign_fiscal_labels."""
    text = normalize(raw_text)
    header = parse_header(text)
    form = header["Filing Type"].split(" ")[0]
    body = text[body_start(text):]
    body = strip_noise(body, stats)
    body = cut_signatures(body)
    end, source = period_end(header, body, header.get("URL", ""))

    sections = detect_sections(body, form)
    for section in sections:
        if section.item == NOTE_SECTIONS[form]:
            section.notes = detect_notes(body, section)

    return Filing(
        file=name,
        cik=header["CIK"],
        ticker=header["Ticker"],
        company=header["Company"],
        form=form,
        filing_date=header["Filing Date"],
        period_end=end,
        period_source=source,
        fiscal_year=0,
        fiscal_quarter=None,
        fiscal_label="",
        fye_month=0,
        url=header.get("URL", ""),
        body=body,
        sections=sections,
    )


def load_corpus(zip_path: str, companies: dict | None = None,
                stats: dict | None = None) -> list[Filing]:
    """Parse every .txt member of the zip, then label fiscal periods."""
    filings = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".txt"):
                continue
            filings.append(parse_filing(name, zf.read(name).decode("utf-8"), stats))
    assign_fiscal_labels(filings, companies)
    return filings


# ---------------------------------------------------------------------------
# Fiscal labels
# ---------------------------------------------------------------------------

FISCAL_ANCHOR_DAYS = 7


def fiscal_anchor(day: dt.date) -> dt.date:
    """The month a period end belongs to, for 52/53-week calendars.

    A year that ends on the Saturday nearest December 31 can end on January 2;
    a quarter can end on October 1. Treating the first few days of a month as
    the end of the previous month keeps those on the same fiscal axis as the
    plain month-end filers.
    """
    if day.day <= FISCAL_ANCHOR_DAYS:
        return day - dt.timedelta(days=day.day)
    return day


def assign_fiscal_labels(filings: list[Filing], companies: dict | None = None) -> None:
    """Fill fye_month, fiscal_year, fiscal_quarter, fiscal_label in place.

    fye_month per ticker is the anchored month of the newest 10-K period end.
    fiscal_year = anchored year + 1 when the anchored month is past the fiscal
    year end, + the hand-set fy_label_offset from companies.yaml (retailers
    name the year ending February 2025 "fiscal 2024"). fiscal_quarter counts
    months since the year end in threes.
    """
    companies = companies or {}
    fye: dict[str, int] = {}
    newest: dict[str, str] = {}
    for f in filings:
        if f.form == "10-K" and f.period_end > newest.get(f.ticker, ""):
            newest[f.ticker] = f.period_end
            fye[f.ticker] = fiscal_anchor(dt.date.fromisoformat(f.period_end)).month
    for f in filings:
        if f.ticker not in fye:
            fye[f.ticker] = fiscal_anchor(dt.date.fromisoformat(f.period_end)).month

    for f in filings:
        anchor = fiscal_anchor(dt.date.fromisoformat(f.period_end))
        offset = int((companies.get(f.ticker) or {}).get("fy_label_offset", 0) or 0)
        f.fye_month = fye[f.ticker]
        f.fiscal_year = anchor.year + (1 if anchor.month > f.fye_month else 0) + offset
        if f.form == "10-K":
            f.fiscal_quarter = None
            f.fiscal_label = "FY%d" % f.fiscal_year
        else:
            f.fiscal_quarter = ((anchor.month - f.fye_month) % 12) // 3
            f.fiscal_label = "FY%d Q%d" % (f.fiscal_year, f.fiscal_quarter)
