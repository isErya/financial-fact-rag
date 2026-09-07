"""Decide the scope of a question before anything is retrieved.

make_plan turns a question into a Plan: which companies it names, which
filings stand for the period it asks about, which sections carry the answer,
how many chunks each (company, period) pair may contribute, and whether the
question has to be refused because the corpus cannot answer it. Every
decision is a rule over the question text and the index's file list. No
model is involved, so the scope can be printed and argued with before a
single token is spent, and the same question always gets the same scope.

Where each decision is made:
  resolve_companies  tickers, aliases, group phrases, unresolved names
  detect_period      explicit years and quarters, windows, quarterly, timeline words
  company_buckets    which filings stand for each requested period, a reason each
  section_weights    keyword intent to per-item weights
  list_metrics       enumerated metrics that become sub-queries
  pin_positions      the two lead chunks pinned for each quota
  make_plan          status, budget, quotas

Fiscal labels come from the index's file records, which were computed from
each company's own year end, so "2024" for Apple is the year ended September
2024 and for NVIDIA the year ended January 2024. The calendar quarter in a
file name is never read here.

Sub-queries carry no company name, a deliberate departure from section 6 of
the milestone spec: each quota already restricts search to one company's
filings, and measured on the bank brief a name prefix moved the JPMorgan
net interest income table from rank 2 to rank 150.

Failure policy: a fiscal label the index wrote in an unexpected form raises
in fiscal_parts, and a period mode this module did not produce raises in
company_buckets, because either means the code and its inputs disagree and
a plan built over that would name the wrong filings. A company with no
filings for the requested period is a note and a gap, never an error: the
plan lists it under not_covered, the coverage block prints the fiscal
labels the corpus does hold, and a question with no covered company at
all becomes a refusal status.
"""

import re
from collections import defaultdict
from functools import lru_cache

import config
from corpus import normalize
from models import Plan

DEFAULT_BUDGET = {
    "small": config.BUDGET_TOKENS_SMALL,
    "large": config.BUDGET_TOKENS_LARGE,
    "max_companies": config.MAX_COMPANIES,
}
# Companies a budget of "small" covers; the seventh company and up are cut.
SMALL_BUDGET_COMPANIES = 3

# Quota sizing. 400 tokens is a large chunk (the index mean is about 320),
# so budget // (quotas * 400) is the count that fills the budget with room
# for the coverage block; the clamp keeps one-company questions from
# reading 50 chunks and ten-bucket timelines from reading two.
TOKENS_PER_CHUNK = 400
MIN_CHUNKS = 6
MAX_CHUNKS = 24
LEAD_CHUNKS = 2
QUARTERLY_DEFAULT = 8
MAX_BUCKETS_PER_COMPANY = 10
DEFAULT_WINDOW_YEARS = 2
# A filer whose newest filing is this many years behind the corpus is
# stale: it is named on request and never expanded from a group phrase.
STALE_YEARS = 3

# ---------------------------------------------------------------------------
# Company resolution
# ---------------------------------------------------------------------------

TICKER_RE = re.compile(r"(\$?)\b([A-Z]{1,5})\b")
# Group phrases fire in the plural or behind one of these qualifiers, so
# "the banks" and "major pharmaceutical companies" expand and "a bank's
# risk" does not.
QUALIFIER_RE = re.compile(r"\b(?:major|large|largest|the big|big|leading|top|main)\b", re.I)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9&.'-]*")
# Capitalised words that are never company names. The list is short on
# purpose: an unresolved name only ever reaches the coverage block as
# "not in the corpus", and a question that names nothing else is refused
# with the same list, so a false name costs one odd line.
NAME_STOPWORDS = {
    "What", "Which", "How", "Why", "When", "Where", "Who", "Whose", "Compare", "Prepare",
    "Summarize", "Summarise", "Describe", "Explain", "List", "Give", "Show", "Tell", "State",
    "Discuss", "Analyze", "Analyse", "Assess", "Evaluate", "Draft", "Write", "Outline",
    "Does", "Do", "Did", "Is", "Are", "Was", "Were", "Has", "Have", "Had", "Can", "Could",
    "Should", "Would", "Will", "The", "A", "An", "And", "Or", "In", "On", "For", "Of", "To",
    "From", "With", "By", "As", "At", "Its", "Their", "I", "We", "You", "It", "This", "That",
    "These", "Those", "There", "Please", "Item", "Part", "Note", "Table", "Form",
    "CFO", "CEO", "CTO", "COO", "PE", "VC", "FY", "EPS", "GAAP", "SEC", "US", "USA", "U.S.",
    "U.S", "UK", "EU", "AI", "IT", "NII", "ROE", "ROA", "ROTCE", "EBITDA", "IPO", "ETF", "MD&A",
    "YoY", "QoQ", "COVID", "COVID-19", "FDA", "GDP", "LLM", "R&D", "M&A",
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December", "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep",
    "Sept", "Oct", "Nov", "Dec",
}
NAME_CONNECTORS = {"of", "&"}
MAX_NAME_WORDS = 3


def _phrase_re(phrase: str, flags: int = 0) -> re.Pattern:
    """Word-bounded match of `phrase`. Alnum lookarounds instead of \\b so
    an alias that ends in punctuation ("AT&T", "J.P. Morgan") still bounds."""
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(phrase) + r"(?![A-Za-z0-9])", flags)


@lru_cache(maxsize=8)
def _alternation(phrases: tuple[str, ...], flags: int) -> re.Pattern:
    """One word-bounded pattern over all `phrases`, longest first so the
    match at a position is the longest phrase there ("JPMorgan Chase"
    before "JPMorgan"). One compile per registry instead of one per alias:
    216 separate compiles cost the first call about 110 ms."""
    ordered = sorted(set(phrases), key=lambda p: (-len(p), p))
    body = "|".join(re.escape(p) for p in ordered)
    return re.compile(r"(?<![A-Za-z0-9])(?:" + body + r")(?![A-Za-z0-9])", flags)


def _overlaps(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(s < end and start < e for s, e in spans)


def newest_period_end(entry: dict) -> str:
    return max((f["period_end"] for f in entry.get("filings", [])), default="")


def stale_tickers(companies: dict) -> set[str]:
    """Tickers whose newest filing is STALE_YEARS behind the newest in the
    registry. In this corpus that is GE, whose one filing is a FY2014
    annual report of a subsidiary."""
    newest = max((newest_period_end(e) for e in companies.values()), default="")
    if not newest:
        return set()
    cutoff = "%04d%s" % (int(newest[:4]) - STALE_YEARS, newest[4:])
    return {t for t, e in companies.items() if newest_period_end(e) and newest_period_end(e) < cutoff}


def resolve_companies(text: str, companies: dict, groups: dict) -> tuple[list[dict], list[str], bool]:
    """(hits in first-mention order, unresolved names, whether a group fired).

    Resolution runs in order of how specific the evidence is: a ticker, a
    case-insensitive alias, a case-sensitive alias, a group phrase. Each
    hit consumes its span so "Bank" inside "Bank of America" cannot fire
    the banks group, and the leftover capitalised words are reported as
    names the corpus does not have.
    """
    stale = stale_tickers(companies)
    consumed: list[tuple[int, int]] = []
    hits: dict[str, dict] = {}
    bare: list[tuple[int, str]] = []

    def add(ticker: str, alias: str, start: int, end: int) -> None:
        consumed.append((start, end))
        if ticker not in hits:
            hits[ticker] = {"ticker": ticker, "name": companies[ticker]["name"],
                            "matched_alias": alias, "position": start}

    # Tickers. Three letters and up stand on their own; one- and two-letter
    # tickers (T, V, MA, GE, DE, MS, ...) collide with ordinary words and
    # abbreviations, so they need a "$" or a question that resolves nothing
    # else.
    for m in TICKER_RE.finditer(text):
        token = m.group(2)
        if token not in companies:
            continue
        if len(token) >= 3 or m.group(1) == "$":
            add(token, m.group(0), m.start(), m.end())
        else:
            bare.append((m.start(), token))

    # Aliases: case-insensitive ones (plus the registry name), then the
    # ones whose lowercase form is an English word ("Apple", "Target",
    # "Chase"), which only count as written.
    ci_owner = {}
    cs_owner = {}
    for ticker, entry in companies.items():
        for alias in list(entry.get("aliases") or []) + [entry["name"]]:
            ci_owner[alias.lower()] = ticker
        for alias in entry.get("aliases_cs") or []:
            cs_owner[alias] = ticker
    for owner, flags in ((ci_owner, re.I), (cs_owner, 0)):
        if not owner:
            continue
        for m in _alternation(tuple(owner), flags).finditer(text):
            if not _overlaps(consumed, m.start(), m.end()):
                key = m.group(0).lower() if flags else m.group(0)
                add(owner[key], m.group(0), m.start(), m.end())

    # Group phrases. The alternation is longest-first, so "big banks" is
    # the match at its position rather than "banks".
    group_fired = False
    phrase_group = {}
    for name, spec in (groups or {}).items():
        for phrase in spec.get("phrases", []):
            phrase_group[phrase.lower()] = name
    if phrase_group:
        for m in _alternation(tuple(phrase_group), re.I).finditer(text):
            if _overlaps(consumed, m.start(), m.end()):
                continue
            start = _group_start(text, m)
            if start is None:
                continue
            group_fired = True
            consumed.append((start, m.end()))
            for ticker in groups[phrase_group[m.group(0).lower()]].get("tickers", []):
                if ticker in companies and ticker not in stale:
                    add(ticker, text[start:m.end()], start, m.end())

    # Bare one- and two-letter tickers: the last resort, and only inside a
    # question that has other words. A one-token question such as "MS" has
    # nothing that says it is a ticker rather than an abbreviation.
    if not hits and bare and len(text.split()) > 1:
        for start, token in bare:
            add(token, token, start, start + len(token))

    unresolved = _unresolved_names(text, consumed)
    ordered = sorted(hits.values(), key=lambda h: h["position"])
    return ordered, unresolved, group_fired


def _group_start(text: str, m: re.Match) -> int | None:
    """Where the group mention starts, or None when the phrase does not
    fire. A qualifier within three words ("major", "the big") is part of
    the mention, so it is consumed and shown with the phrase."""
    matched = m.group(0)
    head = text[:m.start()]
    tail = re.search(r"(?:\S+\s+){0,2}\S+\s*$", head)
    if tail is not None:
        qualifier = None
        for qualifier in QUALIFIER_RE.finditer(tail.group(0)):
            pass
        if qualifier is not None:
            return tail.start() + qualifier.start()
    if matched.lower().endswith("s") or QUALIFIER_RE.match(matched):
        return m.start()
    return None


def _unresolved_names(text: str, consumed: list[tuple[int, int]]) -> list[str]:
    """Capitalised runs of one to three words outside every consumed span,
    skipping sentence starts, months and the stopword list."""
    names: list[str] = []
    run: list[str] = []
    run_end = -1
    run_at_sentence_start = False

    def close() -> None:
        nonlocal run
        if run and not run_at_sentence_start:
            words = [w for w in run if w not in NAME_CONNECTORS][:MAX_NAME_WORDS]
            name = " ".join(words)
            if name and name not in names:
                names.append(name)
        run = []

    for m in WORD_RE.finditer(text):
        word = m.group(0).rstrip(".,;:'")
        if word.endswith("'s"):
            word = word[:-2]
        adjacent = run and text[run_end:m.start()].strip() == ""
        if not adjacent:
            close()
        if _overlaps(consumed, m.start(), m.end()):
            close()
            continue
        if word in NAME_CONNECTORS and run:
            run.append(word)
            run_end = m.end()
            continue
        capitalised = word[:1].isupper() and not any(ch.isdigit() for ch in word)
        if not capitalised or word in NAME_STOPWORDS:
            close()
            continue
        if not run:
            before = text[:m.start()].rstrip()
            run_at_sentence_start = before == "" or before[-1] in ".?!"
        run.append(word)
        run_end = m.end()
    close()
    return names


# ---------------------------------------------------------------------------
# Period detection
# ---------------------------------------------------------------------------

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10}
ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}
_YEAR = r"(20\d\d|\d\d)"
_FISCAL = r"(?:fiscal\s+(?:year\s+)?|fy\s?)?"
QUARTER_RES = [
    re.compile(r"\b([1-4])q\s?" + _YEAR + r"\b"),
    re.compile(r"\bq([1-4])(?:\s+(?:of\s+)?" + _FISCAL + _YEAR + r")?\b"),
    re.compile(r"\b(first|second|third|fourth|1st|2nd|3rd|4th)[\s-]quarter"
               r"(?:\s+(?:of\s+|ended\s+)?" + _FISCAL + _YEAR + r")?\b"),
]
WINDOW_RELATIVE_RE = re.compile(
    r"\b(?:(?:over|in|during|across)\s+the\s+|the\s+)?(?:last|past|previous|trailing)\s+(\w+)\s+"
    r"(?:fiscal\s+)?years?\b")
WINDOW_ABSOLUTE_RES = [
    re.compile(r"\bbetween\s+" + _FISCAL + r"(20\d\d)\s+and\s+" + _FISCAL + r"(20\d\d)\b"),
    re.compile(r"\b(?:from\s+)?" + _FISCAL + r"(20\d\d)\s+(?:to|through|until|thru)\s+" + _FISCAL + r"(20\d\d)\b"),
    re.compile(r"\b(20\d\d)\s*-\s*(20\d\d)\b"),
]
WINDOW_SINCE_RE = re.compile(r"\bsince\s+" + _FISCAL + r"(20\d\d)\b")
YEAR_RE = re.compile(r"\b(?:fiscal\s+(?:year\s+)?|fy\s?)?(20\d\d)\b|\bfy\s?(\d\d)\b")
QUARTERLY_RE = re.compile(
    r"\b(?:each\s+quarter|every\s+quarter|quarterly|quarter[\s-]over[\s-]quarter|by\s+quarter|"
    r"per\s+quarter|quarter\s+by\s+quarter)\b")
TIMELINE_RE = re.compile(
    r"\b(?:changed|changes|changing|trend|trends|trended|over\s+time|evolved|evolve|evolution|"
    r"how\s+has|how\s+have|growth\s+outlook|trajectory)\b")


def _year(token: str) -> int:
    return int(token) if len(token) == 4 else 2000 + int(token)


def detect_period(text: str) -> dict:
    """What period the question asks about, before any company is applied.

    Returns mode plus its arguments: "quarter" (quarter, year or None),
    "window" ((y1, y2) absolute, ("last", n) relative, ("since", y)),
    "year" (years), "quarterly", or "latest". `quarterly` and `timeline`
    ride along as flags. Quarter and window spans are blanked before the
    year scan so the 2025 in "Q3 2025" is not also a fiscal year.
    """
    low = text.lower()
    blank = low
    spans: list[tuple[int, int]] = []
    period: dict = {"mode": "latest", "years": [], "quarter": None, "window": None,
                    "quarterly": bool(QUARTERLY_RE.search(low)),
                    "timeline": bool(TIMELINE_RE.search(low)), "spans": spans}

    def consume(m: re.Match) -> None:
        nonlocal blank
        spans.append((m.start(), m.end()))
        blank = blank[:m.start()] + " " * (m.end() - m.start()) + blank[m.end():]

    quarter = None
    for pattern in QUARTER_RES:
        for m in pattern.finditer(low):
            q_token, y_token = m.group(1), m.group(2)
            q = ORDINALS.get(q_token, None) or int(q_token)
            year = _year(y_token) if y_token else None
            if quarter is None or (quarter[1] is None and year is not None):
                quarter = (q, year)
            consume(m)
    if quarter is not None:
        period["mode"] = "quarter"
        period["quarter"] = quarter
        return period

    window = None
    for pattern in WINDOW_ABSOLUTE_RES:
        m = pattern.search(blank)
        if m:
            y1, y2 = int(m.group(1)), int(m.group(2))
            window = (min(y1, y2), max(y1, y2))
            consume(m)
            break
    if window is None:
        m = WINDOW_SINCE_RE.search(blank)
        if m:
            window = ("since", int(m.group(1)))
            consume(m)
    if window is None:
        m = WINDOW_RELATIVE_RE.search(blank)
        if m:
            count = NUMBER_WORDS.get(m.group(1)) or (int(m.group(1)) if m.group(1).isdigit() else None)
            if count:
                window = ("last", count)
                consume(m)

    years = []
    for m in YEAR_RE.finditer(blank):
        year = _year(m.group(1) or m.group(2))
        if year not in years:
            years.append(year)
        spans.append((m.start(), m.end()))
    period["years"] = years

    if window is not None:
        period["mode"] = "window"
        period["window"] = window
    elif years:
        period["mode"] = "year"
    elif period["quarterly"]:
        period["mode"] = "quarterly"
    elif period["timeline"]:
        # "How has X changed" with no window named: the newest two fiscal years.
        period["mode"] = "window"
        period["window"] = ("last", DEFAULT_WINDOW_YEARS)
    if period["quarterly"] and period["mode"] in ("window", "year"):
        period["mode"] = "quarterly"
    return period


# ---------------------------------------------------------------------------
# Buckets: which filings stand for a period
# ---------------------------------------------------------------------------


def fiscal_parts(label: str) -> tuple[int, int | None]:
    """"FY2025 Q3" -> (2025, 3); "FY2025" -> (2025, None)."""
    m = re.fullmatch(r"FY(\d{4})(?: Q([1-4]))?", label)
    if m is None:
        raise ValueError("unexpected fiscal label %r" % label)
    return int(m.group(1)), (int(m.group(2)) if m.group(2) else None)


def files_by_ticker(files) -> dict[str, list[dict]]:
    """Index file records grouped by ticker, newest first, each copied with
    fiscal_year and fiscal_quarter read off its fiscal label."""
    records = files.values() if isinstance(files, dict) else files
    out: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        fy, fq = fiscal_parts(r["fiscal_label"])
        out[r["ticker"]].append({**r, "fiscal_year": fy, "fiscal_quarter": fq})
    for lst in out.values():
        lst.sort(key=lambda r: (r["period_end"], r["form"] == "10-K"), reverse=True)
    return out


def _entry(record: dict, reason: str) -> dict:
    return {"file": record["file"], "form": record["form"], "fiscal_label": record["fiscal_label"],
            "period_end": record["period_end"], "filing_date": record["filing_date"], "reason": reason}


def _bucket(ticker: str, label: str, entries: list[dict], notes: list[str] | None = None) -> dict:
    return {"ticker": ticker, "label": label, "period_end": entries[0]["period_end"],
            "files": entries, "notes": notes or []}


def _newest_10k(recs: list[dict], at_or_before: str | None = None) -> dict | None:
    for r in recs:
        if r["form"] == "10-K" and (at_or_before is None or r["period_end"] <= at_or_before):
            return r
    return None


def _year_entries(recs: list[dict], year: int) -> list[dict]:
    """All filings of one fiscal year, the 10-K first, then 10-Qs newest first."""
    own = sorted((r for r in recs if r["fiscal_year"] == year), key=lambda r: r["period_end"], reverse=True)
    own.sort(key=lambda r: r["form"] != "10-K")
    return [_entry(r, "newest_10k" if r["form"] == "10-K" else "newest_10q") for r in own]


def _window_years(recs: list[dict], window) -> tuple[list[int], list[int]]:
    """(years to bucket, years asked for) for one company.

    A relative window ("last two years") counts back from the newest fiscal
    year that has a 10-K, since a year is only complete once its annual
    report is out, and adds the year in progress when 10-Qs newer than that
    10-K exist, because those carry the current outlook. An absolute window
    is the years as named.
    """
    present = sorted({r["fiscal_year"] for r in recs})
    if not present:
        return [], []
    if window[0] == "last":
        newest_k = _newest_10k(recs)
        anchor = newest_k["fiscal_year"] if newest_k else present[-1]
        asked = list(range(anchor - window[1] + 1, anchor + 1))
        in_progress = [y for y in present if y > anchor]
        return asked + in_progress, asked
    if window[0] == "since":
        asked = list(range(window[1], present[-1] + 1))
        return asked, asked
    asked = list(range(window[0], window[1] + 1))
    return asked, asked


def company_buckets(ticker: str, recs: list[dict], period: dict, fye_month: int,
                    notes: list[str]) -> tuple[list[dict], list[int]]:
    """(buckets, missing years) for one company under the detected period.
    An empty bucket list means the company has nothing for that period."""
    mode = period["mode"]
    if not recs:
        return [], []

    if mode == "latest":
        newest = recs[0]
        entries = [_entry(newest, "newest_10k" if newest["form"] == "10-K" else "newest_10q")]
        if newest["form"] == "10-Q":
            baseline = _newest_10k(recs)
            if baseline is not None:
                entries.append(_entry(baseline, "annual_baseline"))
        return [_bucket(ticker, "latest", entries)], []

    if mode == "year":
        buckets = []
        missing = []
        for year in period["years"]:
            entries = _year_entries(recs, year)
            if not entries:
                # A later 10-K shows the asked year as a prior-year column of
                # its statements (two years back on the income statement).
                later = [r for r in recs if r["form"] == "10-K" and r["fiscal_year"] in (year + 1, year + 2)]
                if later:
                    later.sort(key=lambda r: r["fiscal_year"])
                    entries = [_entry(later[0], "comparative_columns")]
                    notes.append("%s: no filing for fiscal %d; the %s 10-K shows %d as a prior-year column"
                                 % (ticker, year, later[0]["fiscal_label"], year))
            if entries:
                buckets.append(_bucket(ticker, "FY%d" % year, entries))
            else:
                missing.append(year)
        return buckets, missing

    if mode == "quarter":
        q, year = period["quarter"]
        if q == 4:
            # Fourth quarters are reported inside the 10-K, never as a 10-Q.
            tenks = [r for r in recs if r["form"] == "10-K" and (year is None or r["fiscal_year"] == year)]
            if not tenks:
                return [], [year] if year else []
            notes.append("%s: the fourth quarter is reported in the %s 10-K" % (ticker, tenks[0]["fiscal_label"]))
            return [_bucket(ticker, tenks[0]["fiscal_label"], [_entry(tenks[0], "newest_10k")])], []
        tenqs = [r for r in recs if r["form"] == "10-Q" and r["fiscal_quarter"] == q
                 and (year is None or r["fiscal_year"] == year)]
        if not tenqs:
            return [], [year] if year else []
        tenq = tenqs[0]
        entries = [_entry(tenq, "newest_10q")]
        baseline = _newest_10k(recs, at_or_before=tenq["period_end"])
        if baseline is None:
            # The only annual report is a later one; it covers the full year
            # that contains the quarter.
            after = [r for r in recs if r["form"] == "10-K" and r["period_end"] > tenq["period_end"]]
            baseline = after[-1] if after else None
            if baseline is not None:
                notes.append("%s: no 10-K on or before %s in the corpus; the %s 10-K is the annual baseline"
                             % (ticker, tenq["period_end"], baseline["fiscal_label"]))
        if baseline is not None:
            entries.append(_entry(baseline, "annual_baseline"))
        if fye_month != 12:
            notes.append("%s: quarters are fiscal; %s is the quarter ended %s"
                         % (ticker, tenq["fiscal_label"], tenq["period_end"]))
        return [_bucket(ticker, tenq["fiscal_label"], entries)], []

    if mode == "window":
        years, asked = _window_years(recs, period["window"])
        buckets = []
        missing = []
        for year in years:
            entries = _year_entries(recs, year)
            if entries:
                buckets.append(_bucket(ticker, "FY%d" % year, entries))
            elif year in asked:
                missing.append(year)
        return buckets, missing

    if mode == "quarterly":
        years = None
        if period["window"] is not None:
            years, _asked = _window_years(recs, period["window"])
        elif period["years"]:
            years = period["years"]
        tenqs = [r for r in recs if r["form"] == "10-Q" and (years is None or r["fiscal_year"] in years)]
        if years is None:
            tenqs = tenqs[:QUARTERLY_DEFAULT]
        tenqs = tenqs[:MAX_BUCKETS_PER_COMPANY - 1]
        if not tenqs:
            return [], list(years or [])
        buckets = [_bucket(ticker, r["fiscal_label"], [_entry(r, "newest_10q")]) for r in reversed(tenqs)]
        baseline = _newest_10k(recs)
        if baseline is not None:
            buckets.append(_bucket(ticker, baseline["fiscal_label"], [_entry(baseline, "annual_baseline")]))
        return buckets, []

    raise ValueError("unknown period mode %r" % mode)


# ---------------------------------------------------------------------------
# Section intent
# ---------------------------------------------------------------------------

# Each intent: a name, the keyword pattern, weights by 10-K item, and the
# note-title pattern that earns note chunks an extra factor. 10-Q items
# take the weight of their 10-K counterpart (I.2 is the MD&A, I.1 the
# statements, II.1A the risk factors, II.1 legal proceedings).
INTENTS = [
    ("risk", r"\brisk(?:s|y)?\b|\buncertaint(?:y|ies)\b",
     {"1A": 1.6, "7": 1.0, "1": 0.9}, None),
    ("revenue", r"\brevenues?\b|\bsales\b|\bgrowth\b|\boutlook\b|\bguidance\b|\bresults\b|"
                r"\bmargins?\b|\bearnings\b|\bprofits?\b|\bprofitab\w*",
     {"7": 1.5, "8": 1.3, "1A": 0.8}, "revenue|segment|disaggregat"),
    ("segment", r"\bsegments?\b|\bbusiness units?\b",
     {"8": 1.6, "7": 1.2}, "segment"),
    ("regulatory", r"\bregulat\w*|\bfda\b|\bcompliance\b|\bantitrust\b|\blegal\b|\blitigation\b|"
                   r"\blawsuits?\b",
     {"1": 1.4, "1A": 1.4, "3": 1.3, "7": 0.9}, None),
    ("capital", r"\bcapital\b|\bcet1\b|\btier 1\b|\bliquidity\b|\bdeposits?\b|\buninsured\b|"
                r"\bnet interest income\b|\ballowance\b|\bcredit loss(?:es)?\b|\bheld-to-maturity\b|"
                r"\bsecurities portfolio\b",
     {"7": 1.5, "8": 1.5, "1A": 0.7}, None),
    ("cyber", r"\bcyber\w*", {"1C": 1.6, "1A": 1.2}, None),
    ("competition", r"\bcompet\w*|\bmarket share\b", {"1": 1.5, "1A": 1.0, "7": 1.0}, None),
]
INTENT_RES = [(name, re.compile(pattern, re.I), weights, note) for name, pattern, weights, note in INTENTS]
DEFAULT_WEIGHTS = {"1A": 1.2, "7": 1.2, "8": 1.0, "1": 0.9}
DEFAULT_OTHER = 0.7
INTENT_OTHER = 0.6
TENQ_ITEMS = {"1A": "II.1A", "7": "I.2", "8": "I.1", "3": "II.1"}
# Items that exist in both forms, for choosing the pin target of a filing.
TENK_ITEMS = {v: k for k, v in TENQ_ITEMS.items()}


def section_weights(text: str) -> tuple[dict, str | None, list[str]]:
    """(weights by item with "*" as the catch-all, note intent, intents fired).
    Several intents may fire; an item takes the highest weight offered."""
    merged: dict[str, float] = {}
    note_intent = None
    note_rank = 0.0
    fired = []
    for name, pattern, weights, note in INTENT_RES:
        if not pattern.search(text):
            continue
        fired.append(name)
        for item, weight in weights.items():
            merged[item] = max(merged.get(item, 0.0), weight)
        # Note chunks live in the statements (Item 8), so when two intents
        # with note patterns fire ("revenue by segment"), the one that
        # weights Item 8 higher owns the note pattern.
        if note is not None and weights.get("8", 0.0) > note_rank:
            note_intent, note_rank = note, weights["8"]
    if not fired:
        merged = dict(DEFAULT_WEIGHTS)
    for item in list(merged):
        if item in TENQ_ITEMS:
            merged[TENQ_ITEMS[item]] = merged[item]
    merged["*"] = INTENT_OTHER if fired else DEFAULT_OTHER
    return merged, note_intent, fired


# ---------------------------------------------------------------------------
# Sub-queries
# ---------------------------------------------------------------------------

LIST_ANCHOR_RE = re.compile(r"\b(?:on|including|such as|across)\s+([^.?!;:]+)", re.I)
LIST_BARE_RE = re.compile(r"([^,.?!;:]+(?:,\s*[^,.?!;:]+)+,?\s+(?:and|or)\s+[^,.?!;:]+)")
LIST_SPLIT_RE = re.compile(r",\s*(?:and\s+|or\s+)?|\s+and\s+|\s+or\s+")
LEADING_FILLER_RE = re.compile(r"^(?:the|its|their|each|both|all|any)\s+", re.I)
MAX_METRIC_WORDS = 6


def list_metrics(text: str, consumed: list[tuple[int, int]]) -> list[str]:
    """The metrics a question enumerates, for one sub-query each.

    A list counts when it has two or more short noun phrases after "on",
    "including" or "such as", or as a bare "a, b, and c" run. A list whose
    items are company mentions ("Apple, Tesla, and JPMorgan") is a list of
    companies, which the resolver already handled, so it is skipped.
    """
    for pattern, group in ((LIST_ANCHOR_RE, 1), (LIST_BARE_RE, 1)):
        for m in pattern.finditer(text):
            start = m.start(group)
            items = []
            offset = start
            ok = True
            for raw in LIST_SPLIT_RE.split(m.group(group)):
                piece = raw.strip()
                at = text.find(piece, offset) if piece else -1
                if piece and at >= 0:
                    if _overlaps(consumed, at, at + len(piece)):
                        ok = False
                        break
                    offset = at + len(piece)
                piece = LEADING_FILLER_RE.sub("", piece).strip(" ,")
                if piece and len(piece.split()) <= MAX_METRIC_WORDS:
                    items.append(piece)
            if ok and len(items) >= 2:
                return items
    return []


SCAFFOLD_RE = re.compile(
    r"^(?:please\s+)?(?:what|which|how|why|when|where|who)\b"
    r"(?:\s+(?:is|was|were|are|did|does|do|has|have|had|much|many|would|could|should|can|will))*\s*"
    r"|^(?:please\s+)?(?:summarize|summarise|describe|explain|compare|tell me about|give me|show me|"
    r"list|prepare|discuss|analyze|analyse)\b\s*", re.I)
TAIL_RE = re.compile(
    r"\s*\b(?:and\s+)?how\s+(?:do|does|did|are|is|has|have|were|was)\s+(?:they|it|these|those|each)\b.*$"
    r"|\s*\b(?:changed|change|compare|compares|evolved|do|did|report|reported|say|said|state|disclose|"
    r"disclosed|look like|perform|performed|fare|fared)\b\s*$", re.I)
# Words trimmed from the edges of a fragment once its neighbours (a
# company mention, a period phrase) are gone.
EDGE_WORDS = {"and", "or", "the", "a", "an", "of", "for", "in", "on", "at", "to", "by", "its", "their",
              "with", "from", "during", "between", "over", "s"}
FRAGMENT_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9&./-]*")


def focus_query(text: str, spans: list[tuple[int, int]]) -> str | None:
    """The question with its company mentions, period phrases and question
    scaffolding removed: "What was JPMorgan's net interest income for the
    third quarter of 2025?" becomes "net interest income". Run beside the
    full question, it lets the row label outrank the prose that repeats
    the company's name and the period. None when nothing is left."""
    fragments = []
    last = 0
    for start, end in sorted(spans):
        if start > last:
            fragments.append(text[last:start])
        last = max(last, end)
    fragments.append(text[last:])
    cleaned = []
    for fragment in fragments:
        words = FRAGMENT_WORD_RE.findall(re.sub(r"'s\b", "", fragment))
        while words and words[0].lower() in EDGE_WORDS:
            words.pop(0)
        while words and words[-1].lower() in EDGE_WORDS:
            words.pop()
        if words:
            cleaned.append(" ".join(words))
    out = SCAFFOLD_RE.sub("", " ".join(cleaned))
    out = TAIL_RE.sub("", out).strip()
    words = out.split()
    while words and words[-1].lower() in EDGE_WORDS:
        words.pop()
    out = " ".join(words)
    if not out or out.lower() == text.lower():
        return None
    return out


PAIR_MAX_WORDS = 3


def strip_period_words(phrase: str) -> str:
    """A metric phrase without its period words: "third-quarter net interest
    income" becomes "net interest income". The bucket already fixes the
    period and the period-column factor prefers its tables, while "third"
    and "quarter" in the query only reward prose that repeats them."""
    low = phrase.lower()
    blank = low
    for pattern in QUARTER_RES + WINDOW_ABSOLUTE_RES + [WINDOW_SINCE_RE, WINDOW_RELATIVE_RE, YEAR_RE, QUARTERLY_RE]:
        for m in pattern.finditer(low):
            blank = blank[:m.start()] + " " * (m.end() - m.start()) + blank[m.end():]
    # Blanking keeps offsets, so word spans found in the blanked copy read
    # the original phrase with its case intact.
    words = [phrase[m.start():m.end()] for m in FRAGMENT_WORD_RE.finditer(blank)]
    while words and words[0].lower() in EDGE_WORDS:
        words.pop(0)
    while words and words[-1].lower() in EDGE_WORDS:
        words.pop()
    return " ".join(words)


def split_pair(phrase: str) -> list[str]:
    """"revenue and growth outlook" -> ["revenue", "growth outlook"]: two
    short noun phrases joined by "and" are two metrics. Longer sides are
    clauses ("...face and how are they addressing them") and stay whole."""
    sides = [side.strip() for side in re.split(r"\s+(?:and|or)\s+", phrase)]
    if len(sides) == 2 and all(0 < len(side.split()) <= PAIR_MAX_WORDS for side in sides):
        return sides
    return [phrase]


def _company_spans(text: str, hits: list[dict]) -> list[tuple[int, int]]:
    spans = []
    for hit in hits:
        for m in _phrase_re(hit["matched_alias"], re.I).finditer(text):
            spans.append((m.start(), m.end()))
    return spans


# ---------------------------------------------------------------------------
# Pinning and quotas
# ---------------------------------------------------------------------------


def _section(record: dict, item: str) -> dict | None:
    for s in record.get("sections", []):
        if s["item"] == item:
            return s
    return None


def _lead(section: dict) -> list[int]:
    return list(range(section["chunk_start"], min(section["chunk_end"], section["chunk_start"] + LEAD_CHUNKS)))


def pin_positions(bucket: dict, weights: dict, recs_by_file: dict, recs: list[dict],
                  notes: list[str]) -> list[int]:
    """Chunk positions pinned for a bucket: the first two chunks of the
    highest-weight section of the bucket's primary filing. When a 10-Q's
    top section is a pointer stub or absent (most 10-Q risk sections say
    "see the 10-K"), the newest 10-K's counterpart section is pinned
    instead and that 10-K joins the bucket so the coverage block lists it.
    """
    primary = recs_by_file.get(bucket["files"][0]["file"])
    if primary is None:
        return []
    ranked = sorted((item for item in weights if item != "*"), key=lambda i: -weights[i])
    tenq = primary["form"] == "10-Q"
    for item in ranked:
        if tenq != (item in TENK_ITEMS):
            continue
        section = _section(primary, item)
        if section is not None and not section["is_pointer_stub"] and section["chunk_end"] > section["chunk_start"]:
            return _lead(section)
        if not tenq:
            continue
        counterpart = TENK_ITEMS[item]
        tenk = _newest_10k(recs, at_or_before=primary["period_end"]) or _newest_10k(recs)
        if tenk is None:
            continue
        k_section = _section(tenk, counterpart)
        if k_section is None or k_section["chunk_end"] <= k_section["chunk_start"]:
            continue
        why = "points to the annual report" if section is not None else "is absent"
        notes.append("%s: %s Part %s Item %s %s; Item %s of the %s 10-K is pinned in its place" % (
            bucket["ticker"], primary["fiscal_label"], item.split(".")[0], item.split(".")[1],
            why, counterpart, tenk["fiscal_label"]))
        if all(f["file"] != tenk["file"] for f in bucket["files"]):
            bucket["files"].append(_entry(tenk, "annual_baseline"))
        return _lead(k_section)
    return []


def quota_chunks(budget_tokens: int, n_quotas: int) -> int:
    return max(MIN_CHUNKS, min(MAX_CHUNKS, budget_tokens // (max(1, n_quotas) * TOKENS_PER_CHUNK)))


# ---------------------------------------------------------------------------
# make_plan
# ---------------------------------------------------------------------------


def split_registry(companies: dict, groups: dict | None = None) -> tuple[dict, dict]:
    """(companies map, groups map) from either the whole parsed
    companies.yaml or just its companies map plus an explicit groups map."""
    if "companies" in companies and isinstance(companies["companies"], dict):
        return companies["companies"], groups if groups is not None else companies.get("groups", {}) or {}
    return companies, groups or {}


def make_plan(question: str, companies: dict, files, budget: dict | None = None,
              groups: dict | None = None) -> Plan:
    """The scope of `question` over the registry and the index file list.

    `companies` is the parsed companies.yaml (its companies map and its
    phrase-to-ticker groups), or the companies map alone with `groups`
    passed separately; `files` the index's files.json; `budget` the token
    budgets and the company cap.
    """
    budget = budget or DEFAULT_BUDGET
    companies, groups = split_registry(companies, groups)
    text = normalize(question).strip()
    notes: list[str] = []

    hits, unresolved, group_fired = resolve_companies(text, companies, groups)
    stale = sorted(t for t in stale_tickers(companies) if any(h["ticker"] == t for h in hits))
    for ticker in stale:
        notes.append("%s: newest filing is %s; treat it as stale" % (
            ticker, newest_period_end(companies[ticker])))
    if len(hits) > budget["max_companies"]:
        dropped = [h["ticker"] for h in hits[budget["max_companies"]:]]
        hits = hits[:budget["max_companies"]]
        notes.append("narrowed to the first %d companies; left out %s" % (
            budget["max_companies"], ", ".join(dropped)))

    period = detect_period(text)
    by_ticker = files_by_ticker(files)
    recs_by_file = {r["file"]: r for recs in by_ticker.values() for r in recs}
    weights, note_intent, _fired = section_weights(text)

    plan_companies = []
    buckets: list[dict] = []
    not_covered: list[str] = []
    for hit in hits:
        recs = by_ticker.get(hit["ticker"], [])
        fye_month = int(companies[hit["ticker"]].get("fye_month") or 12)
        own, missing = company_buckets(hit["ticker"], recs, period, fye_month, notes)
        available = [r["fiscal_label"] for r in reversed(recs)]
        plan_companies.append({**hit, "fye_month": fye_month, "available": available,
                               "missing_years": missing, "covered": bool(own)})
        if missing:
            phrase = "fiscal " + ", ".join(str(y) for y in missing)
            not_covered.append("%s: no filings for %s" % (hit["ticker"], phrase))
            notes.append("%s: no filings for %s" % (hit["ticker"], phrase))
        buckets.extend(own)

    if not hits:
        status = "not_covered" if unresolved else "needs_company"
        if status == "needs_company":
            for ticker, entry in sorted(companies.items()):
                recs = by_ticker.get(ticker, [])
                span = "%s to %s" % (recs[-1]["fiscal_label"], recs[0]["fiscal_label"]) if recs else "no filings indexed"
                notes.append("%s %s: %s" % (ticker, entry["name"], span))
        not_covered = list(unresolved)
    elif not buckets:
        status = "period_not_covered"
    else:
        status = "ok"

    n_companies = len(hits)
    budget_tokens = budget["small"] if n_companies <= SMALL_BUDGET_COMPANIES else budget["large"]
    timeline = n_companies == 1 and (period["mode"] in ("window", "quarterly") or period["timeline"])
    comparison = n_companies >= 2 and not timeline

    for bucket in buckets:
        recs = by_ticker.get(bucket["ticker"], [])
        bucket["pinned"] = pin_positions(bucket, weights, recs_by_file, recs, notes)

    chunks = quota_chunks(budget_tokens, len(buckets))
    # "pinned" holds the section-lead positions decided here; "row_label"
    # is filled by retrieve, which is where chunk text is available, with
    # the table chunks seated because a data row is labelled with a metric.
    quotas = [{"ticker": b["ticker"], "bucket": b["label"], "chunks": chunks, "pinned": b["pinned"],
               "row_label": []}
              for b in buckets]

    # Sub-queries run inside each quota's own filings, so the company name
    # adds nothing to them; the enumerated metrics, or the focused form of a
    # single-metric question, are what steer the lexical side to the rows.
    # The reason is measured: prefixing "JPMorgan" to "net interest income"
    # dropped the JPM Q3 2025 NII table from rank 2 to rank 150, because
    # the prefix rewards every chunk that repeats the company's name and
    # the quota's id range already restricts search to that company.
    company_spans = _company_spans(text, hits)
    metrics = list_metrics(text, company_spans)
    if not metrics:
        focus = focus_query(text, company_spans + period["spans"])
        metrics = split_pair(focus) if focus else []
    metrics = [m for m in (strip_period_words(m) for m in metrics) if m]
    sub_queries = [text]
    for metric in metrics:
        if metric.lower() != text.lower() and metric not in sub_queries:
            sub_queries.append(metric)

    return Plan(
        companies=plan_companies,
        unresolved=unresolved,
        not_covered=not_covered,
        stale=stale,
        status=status,
        period_mode=period["mode"],
        buckets=buckets,
        sections=weights,
        note_intent=note_intent,
        comparison=comparison,
        timeline=timeline,
        quotas=quotas,
        sub_queries=sub_queries,
        budget_tokens=budget_tokens,
        notes=notes,
    )


def refusal_coverage(plan: Plan, companies: dict) -> str:
    """The coverage text a refusal carries: what the corpus does hold, so
    the reader can rephrase instead of guessing."""
    lines = []
    if plan.status == "period_not_covered":
        for c in plan.companies:
            available = ", ".join(c["available"]) if c["available"] else "nothing indexed"
            lines.append("%s %s: filings in the corpus cover %s" % (c["ticker"], c["name"], available))
        for note in plan.notes:
            lines.append(note)
        return "\n".join(lines)
    if plan.unresolved:
        lines.append("Not in the corpus: " + ", ".join(plan.unresolved))
    companies, _groups = split_registry(companies)
    lines.append("Covered companies (%d):" % len(companies))
    if plan.status == "needs_company":
        lines.extend("  " + note for note in plan.notes)
    else:
        lines.extend("  %s %s" % (t, e["name"]) for t, e in sorted(companies.items()))
    return "\n".join(lines)
