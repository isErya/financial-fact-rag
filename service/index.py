"""Command-line entry points for the corpus, and the retrieval index.

  python service/index.py report                  -> eval/parse_report.md
  python service/index.py gen-companies [--merge] -> service/companies.yaml
  python service/index.py build [--index-dir DIR] [--dense 0|1] [--files a.txt,b.txt]
                                                  -> DIR/chunks.jsonl.gz, DIR/bm25/,
                                                     DIR/dense.npy, DIR/files.json,
                                                     DIR/ticker_ranges.json,
                                                     DIR/fingerprint.json

`report` is the parser's acceptance test in prose: summary rates against the
thresholds the milestone promises, then one row per file so a reviewer can
spot-check a boundary. `gen-companies` writes the per-ticker facts the parser
measured and leaves the hand-written fields empty; with --merge the hand
fields already in the file survive a regeneration.

`build` parses the corpus, chunks every filing, and writes a lexical index
(bm25s) and, when asked, a dense one (fastembed MiniLM-L6). Chunks are laid
out ticker by ticker so a company's chunks are one contiguous id range and
a search over one ticker is a slice. `load` reads the directory back into
an Index whose `search` fuses the two rankings with reciprocal rank fusion.

Failure policy: any parse error propagates. A report or an index over a
partial corpus would hide exactly the file that needs attention.
"""

import argparse
import dataclasses
import datetime as dt
import gzip
import hashlib
import json
import multiprocessing
import os
import re
import sys
import time
import zipfile
from collections import Counter

import bm25s
import numpy as np
import Stemmer
import yaml

import config
from chunk import chunk_filing
from corpus import (
    HEADER_RULE,
    body_start,
    detect_notes,
    load_corpus,
    normalize,
)
from models import Chunk, Column, Filing

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
    q_i1 = sum(1 for f in tenq if section_of(f, "I.1"))
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
        "| 10-Qs with I.1 | %s | |" % pct(q_i1, len(tenq)),
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


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

STEMMER = Stemmer.Stemmer("english")
# Chunks handed to one embedding worker per task: a few batches, so the
# progress line moves and no worker sits on one giant task at the end.
EMBED_SLAB = config.EMBED_BATCH * 8
# Below this many chunks one process embeds faster than a pool can start.
POOL_MIN_TEXTS = 1000
PROGRESS_EVERY = 1000
RRF_K = 60


def index_text(chunk: Chunk) -> str:
    """What both indexes see: the header line first, so a truncated
    embedding loses chunk tail, never the filing, item, units or columns."""
    return chunk.header + "\n" + chunk.text


# Stopwords and stemming are fixed in these two functions so the index and
# every query agree on the vocabulary.


def tokenize_documents(texts: list[str]):
    return bm25s.tokenize(texts, stopwords="en", stemmer=STEMMER, show_progress=False)


def tokenize_query(query: str) -> list[str]:
    return bm25s.tokenize(query, stopwords="en", stemmer=STEMMER, show_progress=False,
                          return_ids=False)[0]


def fingerprint(zip_path: str, dense: bool, files: list[str] | None) -> str:
    """One hash over everything that changes the index: the zip bytes, the
    chunker version, the embedding model, the dense flag, the file subset."""
    digest = hashlib.sha256()
    with open(zip_path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    tail = "|%s|%s|%d|%s" % (config.CHUNKER_VERSION, config.EMBED_MODEL, dense,
                             ",".join(sorted(files or [])))
    digest.update(tail.encode())
    return digest.hexdigest()


def embedder(threads: int | None = None):
    """The fastembed model, loaded from the local cache only.

    Imported here so the web process and the BM25-only build never pay for
    onnxruntime. The tokenizer file ships with truncation and padding fixed
    at 128 tokens for MiniLM; truncation is raised to EMBED_MAX_TOKENS and
    padding switched to the longest sequence in the batch, since a fixed
    128 pad under a 256 cut gives ragged batches. fastembed offers no
    argument for either, which is also why the parallel build below runs
    its own worker pool instead of fastembed's `parallel=`. `threads` caps
    the onnxruntime session so several workers share the cores instead of
    each claiming all of them.
    """
    from fastembed import TextEmbedding

    # EMBED_DEVICE=cuda asks onnxruntime for its CUDA provider, which exists
    # only in the GPU image (onnxruntime-gpu) on a host that exposes a GPU to
    # the container. Anywhere else fastembed raises rather than quietly
    # running on CPU, which is the right failure: a nine-hour build that
    # was meant to take two minutes should not start silently.
    cuda = config.EMBED_DEVICE == "cuda"
    if cuda:
        # The CUDA 13 runtime and cuDNN arrive as pip packages beside
        # onnxruntime-gpu; this puts them on the loader path so the CUDA
        # provider is available when the session below is created.
        import onnxruntime
        onnxruntime.preload_dlls()
    model = TextEmbedding(config.EMBED_MODEL, cache_dir=config.FASTEMBED_CACHE,
                          local_files_only=True, threads=threads, cuda=cuda)
    try:
        providers = model.model.model.get_providers()
    except AttributeError:
        providers = ["unknown"]
    if cuda and not any("CUDA" in p for p in providers):
        raise RuntimeError("EMBED_DEVICE=cuda but onnxruntime gave %s; the GPU is not reaching the "
                           "container" % providers)
    embedder.providers = providers
    tokenizer = model.model.tokenizer
    tokenizer.enable_truncation(max_length=config.EMBED_MAX_TOKENS)
    padding = dict(tokenizer.padding or {})
    padding["length"] = None
    tokenizer.enable_padding(**padding)
    return model


_WORKER_MODEL = None


def _init_worker(threads: int) -> None:
    global _WORKER_MODEL
    _WORKER_MODEL = embedder(threads)


def _embed_slab(texts: list[str]) -> np.ndarray:
    return np.stack(list(_WORKER_MODEL.embed(texts, batch_size=config.EMBED_BATCH)))


def embed_texts(texts: list[str], workers: int) -> np.ndarray:
    """float32 [n, 384] unit vectors for `texts`, in order.

    Workers use the spawn start method, since onnxruntime's thread pool
    does not survive a fork; each worker loads the model once in
    `_init_worker` with a share of the cores. On the 10-core laptop this
    was developed on, one session already saturates the cores and the pool
    adds little (21 versus 22 chunks per second); it earns its keep on a
    host with more cores. Under POOL_MIN_TEXTS chunks one process wins
    outright, since spawning workers costs more than embedding that many.
    """
    slabs = [texts[i:i + EMBED_SLAB] for i in range(0, len(texts), EMBED_SLAB)]
    parts: list[np.ndarray] = []
    done = 0
    next_mark = PROGRESS_EVERY

    def took(vecs: np.ndarray) -> None:
        nonlocal done, next_mark
        parts.append(vecs)
        done += len(vecs)
        if done >= next_mark:
            print("embedded %d / %d chunks" % (done, len(texts)), flush=True)
            next_mark += PROGRESS_EVERY

    if workers <= 1 or len(texts) < POOL_MIN_TEXTS:
        model = embedder()
        for slab in slabs:
            took(np.stack(list(model.embed(slab, batch_size=config.EMBED_BATCH))))
    else:
        threads = max(1, (os.cpu_count() or workers) // workers)
        context = multiprocessing.get_context("spawn")
        with context.Pool(workers, initializer=_init_worker, initargs=(threads,)) as pool:
            for vecs in pool.imap(_embed_slab, slabs):
                took(vecs)
    dense = np.concatenate(parts).astype(np.float32)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    return dense / np.maximum(norms, 1e-12)


def build(index_dir: str, dense: bool = False, files: list[str] | None = None) -> dict:
    """Build the index into `index_dir`; returns the fingerprint record.

    Skips the whole build when the directory already holds an index with
    the same fingerprint, so a container start or a test can call it
    unconditionally.
    """
    os.makedirs(index_dir, exist_ok=True)
    sha = fingerprint(config.CORPUS_ZIP, dense, files)
    record_path = os.path.join(index_dir, "fingerprint.json")
    if os.path.exists(record_path):
        with open(record_path) as fh:
            record = json.load(fh)
        if record.get("sha256") == sha:
            print("index up to date")
            return record

    phases: dict[str, float] = {}
    started = time.perf_counter()
    filings = load_corpus(config.CORPUS_ZIP, load_companies(config.COMPANIES_FILE),
                          files=set(files) if files else None)
    phases["parse"] = round(time.perf_counter() - started, 1)

    started = time.perf_counter()
    filings.sort(key=lambda f: (f.ticker, f.file))
    chunks: list[Chunk] = []
    file_records = []
    ticker_ranges: dict[str, list[int]] = {}
    for filing in filings:
        first = len(chunks)
        own = chunk_filing(filing)
        chunks.extend(own)
        sections = []
        for section in filing.sections:
            members = [k for k, c in enumerate(own) if c.item == section.item]
            sections.append({
                "item": section.item,
                "title": section.title,
                "is_pointer_stub": section.is_pointer_stub,
                "chunk_start": first + members[0] if members else first + len(own),
                "chunk_end": first + members[-1] + 1 if members else first + len(own),
            })
        file_records.append({
            "file": filing.file,
            "ticker": filing.ticker,
            "form": filing.form,
            "period_end": filing.period_end,
            "fiscal_label": filing.fiscal_label,
            "filing_date": filing.filing_date,
            "chunk_start": first,
            "chunk_end": len(chunks),
            "sections": sections,
        })
        span = ticker_ranges.setdefault(filing.ticker, [first, first])
        span[1] = len(chunks)
    phases["chunk"] = round(time.perf_counter() - started, 1)

    started = time.perf_counter()
    with gzip.open(os.path.join(index_dir, "chunks.jsonl.gz"), "wt") as fh:
        for chunk in chunks:
            fh.write(json.dumps(dataclasses.asdict(chunk)) + "\n")
    with open(os.path.join(index_dir, "ticker_ranges.json"), "w") as fh:
        json.dump(ticker_ranges, fh, indent=1)
    with open(os.path.join(index_dir, "files.json"), "w") as fh:
        json.dump(file_records, fh, indent=1)
    phases["write"] = round(time.perf_counter() - started, 1)

    started = time.perf_counter()
    texts = [index_text(c) for c in chunks]
    retriever = bm25s.BM25()
    retriever.index(tokenize_documents(texts), show_progress=False)
    retriever.save(os.path.join(index_dir, "bm25"))
    phases["bm25"] = round(time.perf_counter() - started, 1)

    dense_path = os.path.join(index_dir, "dense.npy")
    if dense:
        started = time.perf_counter()
        np.save(dense_path, embed_texts(texts, config.EMBED_WORKERS))
        phases["dense"] = round(time.perf_counter() - started, 1)
    elif os.path.exists(dense_path):
        os.remove(dense_path)

    tables = [c for c in chunks if c.kind == "table"]
    with_columns = sum(1 for c in tables if c.column_source == "parsed")
    record = {
        "sha256": sha,
        "chunks": len(chunks),
        "files": len(filings),
        "dense": dense,
        "built_at": dt.datetime.now().isoformat(timespec="seconds"),
        "phase_seconds": phases,
    }
    with open(record_path, "w") as fh:
        json.dump(record, fh, indent=1)
    print("parsed %d filings | %d chunks | tables with columns %d%% | bm25 in %ss | dense in %s" % (
        len(filings), len(chunks), round(100.0 * with_columns / len(tables)) if tables else 0,
        phases["bm25"], "%ss" % phases["dense"] if dense else "off"))
    print("phase seconds:", json.dumps(phases))
    return record


def cmd_build(args: argparse.Namespace) -> None:
    files = [f for f in args.files.split(",") if f] if args.files else None
    build(args.index_dir, dense=args.dense == "1", files=files)


# ---------------------------------------------------------------------------
# load and search
# ---------------------------------------------------------------------------


def _ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank of each score, best first; ties keep index order."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


class Index:
    """A loaded index. Scores are over all chunk ids; `search` restricts to
    a caller-chosen id set (a ticker's range, a section's range) and ranks
    inside it, so a small company is never crowded out by a large one."""

    def __init__(self, chunks: list[Chunk], ticker_ranges: dict, files: list[dict],
                 bm25: bm25s.BM25, dense: np.ndarray | None):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.ticker_ranges = ticker_ranges
        self.files = files
        self.bm25 = bm25
        self.dense = dense
        self._model = None
        self._score_cache: dict[str, np.ndarray] = {}

    def bm25_scores(self, query: str) -> np.ndarray:
        tokens = tokenize_query(query)
        if not tokens:
            # Every query word was a stopword; bm25s raises on an empty list.
            return np.zeros(len(self.chunks), dtype=np.float32)
        return self.bm25.get_scores(tokens)

    def dense_scores(self, query: str) -> np.ndarray:
        """Cosine of `query` against every chunk, cached per query string.

        One question runs this once per quota per sub-query, ten times for a
        five-company question, and every call is identical work: the same
        encoder pass and the same pass over all 198 MB of vectors. Caching by
        query text took a five-company question from 7.1 s to under a second.
        The cache is small and per process, since a web worker only ever holds
        the handful of sub-queries belonging to the question in flight.
        """
        hit = self._score_cache.get(query)
        if hit is not None:
            return hit
        scores = self._dense_scores_uncached(query)
        if len(self._score_cache) >= 8:
            self._score_cache.pop(next(iter(self._score_cache)))
        self._score_cache[query] = scores
        return scores

    def _dense_scores_uncached(self, query: str) -> np.ndarray:
        if self.dense is None:
            raise RuntimeError("index was built without dense vectors")
        if self._model is None:
            # Lazy so a web process that never sees a dense query starts fast.
            self._model = embedder()
            self._check_encoder_matches_vectors()
        # query_embed does not add the model's query instruction, so this
        # does. embed() rather than query_embed() because the prefix has to
        # be inside the encoded text, not alongside it.
        text = config.EMBED_QUERY_PREFIX + query
        vector = np.asarray(list(self._model.embed([text]))[0], dtype=np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        return np.asarray(self.dense @ vector)

    def _check_encoder_matches_vectors(self, positions: tuple = (0, 1, 5000, 30000)) -> None:
        """Re-embed a few stored chunks and compare against their rows.

        The vectors are built once, on a GPU, and served here by a different
        library on CPU. Pooling, normalisation, truncation length, quantisation
        and the model revision can all differ between the two without raising
        anything, and the only symptom is worse retrieval. This turns that
        silent failure into a loud one at first use.
        """
        usable = [p for p in positions if p < len(self.chunks) and p < self.dense.shape[0]]
        if not usable:
            return
        texts = [index_text(self.chunks[p]) for p in usable]
        fresh = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        fresh /= np.maximum(np.linalg.norm(fresh, axis=1, keepdims=True), 1e-12)
        stored = np.asarray(self.dense[usable], dtype=np.float32)
        stored /= np.maximum(np.linalg.norm(stored, axis=1, keepdims=True), 1e-12)
        worst = float(np.min(np.sum(fresh * stored, axis=1)))
        if worst < 0.99:
            raise RuntimeError(
                "the encoder serving queries does not match the stored vectors "
                "(worst cosine %.4f over chunks %s). The index was built with a "
                "different model, window or library than %s is loading now; "
                "rebuild the index or set EMBED_MODEL to the one that built it."
                % (worst, usable, config.EMBED_MODEL))

    def effective_mode(self, requested: str | None = None) -> tuple[str, str | None]:
        """The mode a search will run, and why it differs from the configured
        one when it does.

        An explicit request is honoured as asked, so an ablation that asks for
        dense on a lexical index fails rather than quietly measuring bm25.
        The configured default is different: DENSE=0 is a documented way to
        run without a GPU, and an index built that way must still serve.
        It serves lexical, and says so on /health instead of in a log line.
        """
        if requested is not None:
            return requested, None
        mode = config.SEARCH_MODE
        if mode in ("dense", "hybrid") and self.dense is None:
            return "bm25", ("SEARCH_MODE is %s but this index holds no vectors (built with DENSE=0); "
                            "serving lexical retrieval" % mode)
        return mode, None

    def search(self, query: str, ids: np.ndarray, k: int,
               mode: str | None = None) -> list[tuple[str, int, int | None, float]]:
        """Top k of `ids` by reciprocal rank fusion of the BM25 and dense
        rankings computed within `ids`. Each hit is (chunk_id, bm25_rank,
        dense_rank or None, rrf). `mode` is "hybrid" (both rankings when
        the index has dense vectors), "bm25" or "dense"; the last two exist
        for the retrieval ablation and each fuses a single ranking."""
        mode, _note = self.effective_mode(mode)
        if mode not in ("hybrid", "bm25", "dense"):
            raise ValueError("unknown search mode %r" % mode)
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size == 0:
            return []
        bm25_rank = _ranks(self.bm25_scores(query)[ids])
        rrf = np.zeros(ids.size)
        if mode != "dense":
            rrf = rrf + 1.0 / (RRF_K + bm25_rank)
        dense_rank = None
        if mode != "bm25" and (self.dense is not None or mode == "dense"):
            dense_rank = _ranks(self.dense_scores(query)[ids])
            rrf = rrf + 1.0 / (RRF_K + dense_rank)
        order = np.argsort(-rrf, kind="stable")[:k]
        return [(self.chunks[ids[j]].chunk_id, int(bm25_rank[j]),
                 None if dense_rank is None else int(dense_rank[j]), float(rrf[j]))
                for j in order]


def read_chunks(index_dir: str) -> list[Chunk]:
    chunks = []
    with gzip.open(os.path.join(index_dir, "chunks.jsonl.gz"), "rt") as fh:
        for line in fh:
            record = json.loads(line)
            record["columns"] = [Column(**c) for c in record["columns"]]
            chunks.append(Chunk(**record))
    return chunks


def load(index_dir: str) -> Index:
    """Read an index directory. The dense matrix is memory-mapped, so load
    time does not grow with the corpus."""
    chunks = read_chunks(index_dir)
    with open(os.path.join(index_dir, "ticker_ranges.json")) as fh:
        ticker_ranges = json.load(fh)
    with open(os.path.join(index_dir, "files.json")) as fh:
        files = json.load(fh)
    retriever = bm25s.BM25.load(os.path.join(index_dir, "bm25"))
    dense_path = os.path.join(index_dir, "dense.npy")
    dense = np.load(dense_path, mmap_mode="r") if os.path.exists(dense_path) else None
    return Index(chunks, ticker_ranges, files, retriever, dense)


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

    p_build = sub.add_parser("build", help="build the retrieval index")
    p_build.add_argument("--index-dir", default=config.INDEX_DIR)
    p_build.add_argument("--dense", default=config.DENSE, choices=["0", "1"],
                         help="also embed every chunk (slow on a full corpus)")
    p_build.add_argument("--files", default="",
                         help="comma-separated corpus members; default is the whole zip")
    p_build.set_defaults(func=cmd_build)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
