"""Command-line entry points for the corpus.

  python service/index.py report                  -> eval/parse_report.md
  python service/index.py gen-companies [--merge] -> service/companies.yaml
  python service/index.py build                   -> milestone 2

`report` is the parser's acceptance test in prose: summary rates against the
thresholds the milestone promises, then one row per file so a reviewer can
spot-check a boundary. `gen-companies` writes the per-ticker facts the parser
measured and leaves the hand-written fields empty; with --merge the hand
fields already in the file survive a regeneration.

Failure policy: any parse error propagates. A report over a partial corpus
would hide exactly the file that needs attention.
"""

import argparse
import datetime as dt
import os
import re
import sys
import time
import zipfile
from collections import Counter

import yaml

import config
from corpus import (
    HEADER_RULE,
    body_start,
    detect_notes,
    load_corpus,
    normalize,
)
from models import Filing

MIN_NOTES = 5
OFFSET_SCAN_CHARS = 60000
# A delimited comma-formatted number. The preamble glues XBRL values into
# digit runs ("false1,785,288,846176.61,435"), which are fragments, never
# facts, so a match may not touch another digit, comma, or period.
COMMA_NUMBER_RE = re.compile(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?![\d.,])")


def load_registry(path: str) -> dict:
    """The whole companies.yaml: companies plus the hand-written groups map."""
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def load_companies(path: str) -> dict:
    """The companies.yaml mapping keyed by ticker, or {} when absent."""
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("companies", {})


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def section_of(filing: Filing, item: str):
    for section in filing.sections:
        if section.item == item:
            return section
    return None


def span_ok(filing: Filing, item: str, lo: int, hi: int) -> bool:
    section = section_of(filing, item)
    return section is not None and lo <= section.end - section.start <= hi


def preamble_check(zip_path: str) -> list[str]:
    """Files where a comma-formatted number in the preamble is absent from
    the body. The preamble carries tag names and context ids, never values,
    so this is expected to be empty; a hit means the cover marker landed in
    the wrong place."""
    bad = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".txt"):
                continue
            text = normalize(zf.read(name).decode("utf-8"))
            start = body_start(text)
            preamble = text[text.find(HEADER_RULE) + len(HEADER_RULE):start]
            body = text[start:]
            if any(num not in body for num in set(COMMA_NUMBER_RE.findall(preamble))):
                bad.append(name)
    return bad


def offset_candidates(filings: list[Filing]) -> list[str]:
    """Tickers whose newest 10-K names the previous year as its fiscal year
    more often than the computed one, so fy_label_offset likely needs -1."""
    newest: dict[str, Filing] = {}
    for f in filings:
        if f.form == "10-K" and (f.ticker not in newest or f.period_end > newest[f.ticker].period_end):
            newest[f.ticker] = f
    out = []
    for ticker, f in sorted(newest.items()):
        head = f.body[:OFFSET_SCAN_CHARS].lower()
        this_year = head.count("fiscal %d" % f.fiscal_year)
        prior_year = head.count("fiscal %d" % (f.fiscal_year - 1))
        if prior_year > this_year:
            out.append("%s (fiscal %d x%d vs fiscal %d x%d)" % (
                ticker, f.fiscal_year - 1, prior_year, f.fiscal_year, this_year))
    return out


def no_ii1a_by_ticker(tenq: list[Filing]) -> str:
    """Tickers whose 10-Qs carry no Part II Item 1A heading, with counts.

    The spec's descriptive figure (69 of 144 II.1A sections are stubs) does
    not reproduce from the corpus: all 157 10-Q members contain the string
    "Item 1A" somewhere, 133 carry a Part II heading, and no heading count
    lands on 144. The 24 without a heading are JNJ (the item is omitted)
    and XOM (Part I refers to "Item 1A. Risk Factors of ExxonMobil's Form
    10-K" and Part II has no such item), so this line names them.
    """
    missing = Counter(f.ticker for f in tenq if section_of(f, "II.1A") is None)
    return ", ".join("%s %d" % kv for kv in sorted(missing.items())) or "none"


def notes_before_auditor_report(tenk: list[Filing]) -> list[str]:
    """10-Ks whose Item 8 holds no notes while the section before it holds
    at least MIN_NOTES. The filer printed the auditor's report after the
    notes, so the auditor's-report fallback opened Item 8 too late and the
    statements sit in the preceding span. Listed so a later milestone can
    decide where the chunker should read the notes from."""
    out = []
    for f in tenk:
        eight = section_of(f, "8")
        if eight is None or eight.notes:
            continue
        before = [s for s in f.sections if s.end == eight.start]
        if not before:
            continue
        found = len(detect_notes(f.body, before[0]))
        if found >= MIN_NOTES:
            out.append("%s (%d in Item %s)" % (f.ticker, found, before[0].item))
    return out


def build_report(filings: list[Filing], stats: dict, parse_seconds: float,
                 preamble_bad: list[str]) -> str:
    tenk = [f for f in filings if f.form == "10-K"]
    tenq = [f for f in filings if f.form == "10-Q"]

    def pct(n: int, d: int) -> str:
        return "%d/%d (%.1f%%)" % (n, d, 100.0 * n / d if d else 0.0)

    k_1a = sum(1 for f in tenk if section_of(f, "1A"))
    k_1a_ok = sum(1 for f in tenk if span_ok(f, "1A", 5_000, 400_000))
    k_7 = sum(1 for f in tenk if section_of(f, "7"))
    k_8 = sum(1 for f in tenk if section_of(f, "8"))
    q_i2 = sum(1 for f in tenq if section_of(f, "I.2"))
    q_ii1a = sum(1 for f in tenq if section_of(f, "II.1A"))
    q_stub = sum(1 for f in tenq if section_of(f, "II.1A") and section_of(f, "II.1A").is_pointer_stub)
    k_notes = sum(1 for f in tenk if section_of(f, "8") and len(section_of(f, "8").notes) >= MIN_NOTES)
    sources = Counter(f.period_source for f in filings)

    lines = [
        "# Parse report",
        "",
        "Generated %s by `python service/index.py report`. Parse wall time %.1fs." % (
            dt.datetime.now().strftime("%Y-%m-%d %H:%M"), parse_seconds),
        "",
        "## Summary",
        "",
        "| Measure | Value | Threshold |",
        "|---|---|---|",
        "| Files parsed | %d | 246 |" % len(filings),
        "| Cover marker hits | %d | all |" % len(filings),
        "| Period end source | %s | |" % ", ".join("%s %d" % kv for kv in sorted(sources.items())),
        "| 10-Ks with Item 1A | %s | >= 95%% |" % pct(k_1a, len(tenk)),
        "| 10-Ks with Item 1A span 5k-400k | %s | >= 95%% |" % pct(k_1a_ok, len(tenk)),
        "| 10-Ks with Item 7 | %s | >= 95%% |" % pct(k_7, len(tenk)),
        "| 10-Ks with Item 8 | %s | >= 90%% |" % pct(k_8, len(tenk)),
        "| 10-Qs with I.2 | %s | |" % pct(q_i2, len(tenq)),
        "| 10-Qs with II.1A | %s | |" % pct(q_ii1a, len(tenq)),
        "| 10-Q II.1A pointer stubs | %d | |" % q_stub,
        "| 10-K Item 8 with >= %d notes | %s | >= 90%% |" % (MIN_NOTES, pct(k_notes, k_8)),
        "| Running-header lines removed | %d | |" % stats.get("running header lines", 0),
        "| Footer lines/tokens removed | %d | |" % sum(v for k, v in stats.items() if k.startswith("footer")),
        "| Glued Table of Contents tokens removed | %d | |" % stats.get("glued Table of Contents", 0),
        "| Preamble numeric-fact check (files failing) | %d | 0 |" % len(preamble_bad),
        "",
        "Fiscal label offset candidates (newest 10-K names the prior year more often): %s" % (
            ", ".join(offset_candidates(filings)) or "none"),
        "",
        "10-Qs with no Part II Item 1A heading, by ticker: %s" % no_ii1a_by_ticker(tenq),
        "",
        "10-K Item 8 with no notes while the section before it holds notes "
        "(auditor's report printed after the notes): %s" % (
            ", ".join(notes_before_auditor_report(tenk)) or "none"),
        "",
    ]
    if preamble_bad:
        lines += ["Preamble check failures: " + ", ".join(preamble_bad), ""]

    lines += [
        "## Per file",
        "",
        "| File | Form | Period end | Fiscal | Body chars | Sections item:length | Stubs | Notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f in filings:
        secs = " ".join("%s:%d" % (s.item, s.end - s.start) for s in f.sections)
        stubs = " ".join(s.item for s in f.sections if s.is_pointer_stub) or "-"
        notes = sum(len(s.notes) for s in f.sections)
        lines.append("| %s | %s | %s (%s) | %s | %d | %s | %s | %d |" % (
            f.file, f.form, f.period_end, f.period_source, f.fiscal_label,
            len(f.body), secs, stubs, notes))
    lines.append("")
    return "\n".join(lines)


def cmd_report(args: argparse.Namespace) -> None:
    stats: dict = {}
    started = time.perf_counter()
    filings = load_corpus(config.CORPUS_ZIP, load_companies(config.COMPANIES_FILE), stats)
    parse_seconds = time.perf_counter() - started
    preamble_bad = preamble_check(config.CORPUS_ZIP)
    text = build_report(filings, stats, parse_seconds, preamble_bad)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(text)
    # The summary block is short enough to echo so the numbers land in the
    # terminal as well as the file.
    print(text.split("## Per file")[0])
    print("wrote", args.out)


# ---------------------------------------------------------------------------
# gen-companies
# ---------------------------------------------------------------------------

HAND_FIELDS = ("aliases", "aliases_cs", "sector", "groups", "notes", "fy_label_offset")

COMPANIES_HEADER = """\
# Company registry, one entry per ticker.
#
# Generated by `python service/index.py gen-companies` (add --merge to keep the
# hand-written fields below): name, cik, fye_month, filings.
# Hand-written, kept across --merge: aliases (case-insensitive), aliases_cs
# (case-sensitive: the lowercase form is an English word), sector, groups,
# notes, fy_label_offset. The top-level `groups` map phrases to tickers. Set fy_label_offset to -1 for a company that names its
# fiscal year by the calendar year it mostly falls in (a year ending
# February 2025 called "fiscal 2024").
"""


def cmd_gen_companies(args: argparse.Namespace) -> None:
    existing = load_companies(args.out) if args.merge else {}
    # The phrase-to-ticker groups are hand-written too; carry them across a merge.
    groups = load_registry(args.out).get("groups", {}) if args.merge else {}
    filings = load_corpus(config.CORPUS_ZIP, existing)
    companies: dict = {}
    for f in filings:
        entry = companies.get(f.ticker)
        if entry is None:
            old = existing.get(f.ticker, {})
            entry = {
                "name": f.company,
                "cik": f.cik,
                "fye_month": f.fye_month,
                "fy_label_offset": old.get("fy_label_offset", 0),
                "filings": [],
                "aliases": old.get("aliases", []),
                "aliases_cs": old.get("aliases_cs", []),
                "sector": old.get("sector", ""),
                "groups": old.get("groups", []),
                "notes": old.get("notes", ""),
            }
            companies[f.ticker] = entry
        entry["filings"].append({
            "file": f.file,
            "form": f.form,
            "period_end": f.period_end,
            "filing_date": f.filing_date,
        })
    for entry in companies.values():
        entry["filings"].sort(key=lambda x: x["period_end"])
    ordered = {ticker: companies[ticker] for ticker in sorted(companies)}
    with open(args.out, "w") as fh:
        fh.write(COMPANIES_HEADER)
        fh.write("\n")
        yaml.safe_dump({"companies": ordered, "groups": groups}, fh, sort_keys=False,
                       allow_unicode=False, width=100)
    print("wrote", args.out, "with", len(ordered), "companies")


def cmd_build(args: argparse.Namespace) -> None:
    raise NotImplementedError("build arrives with milestone 2")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_report = sub.add_parser("report", help="write eval/parse_report.md")
    p_report.add_argument("--out", default="eval/parse_report.md")
    p_report.set_defaults(func=cmd_report)

    p_gen = sub.add_parser("gen-companies", help="write service/companies.yaml")
    p_gen.add_argument("--merge", action="store_true", help="keep hand-written fields")
    p_gen.add_argument("--out", default=config.COMPANIES_FILE)
    p_gen.set_defaults(func=cmd_gen_companies)

    p_build = sub.add_parser("build", help="build the retrieval index (milestone 2)")
    p_build.set_defaults(func=cmd_build)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
