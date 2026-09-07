"""Search inside the plan's filings and assemble a budgeted, balanced context.

retrieve runs every sub-query over each quota's chunk ids, fuses the
rankings by summing reciprocal-rank contributions, multiplies by the plan's
section weights, puts the pinned lead chunks first and keeps a short
candidate list per quota. assemble then picks from those lists under the
token budget: every quota gets a floor before any quota takes more,
boilerplate repeated across a company's filings is dropped, and the
survivors are ordered company by company, filing by filing, in reading
order, and numbered C1, C2, ... The excerpts block and the coverage block
are rendered here, so a dry run shows the exact text the model will read.

Searching inside a quota's ids is what keeps a wordy filer from crowding
out a terse one: JPMorgan's 10-K has 1,160 chunks and Lilly's 350, and a
ranking over both would be mostly JPMorgan.

Row-label seats (row_label_seats) are the one rule here that reads chunk
text rather than a ranking: for each metric sub-query, up to two table
chunks of the bucket whose data rows are labelled with that metric are
seated inside the quota ahead of everything search found. They are
recorded on the quota as "row_label" positions, beside the section-lead
"pinned" positions, so the printed plan shows both kinds of seat.

Failure policy: a quota whose (ticker, bucket) pair has no bucket in the
plan raises, since a plan that disagrees with itself would otherwise
retrieve nothing for that company without a trace. A bucket file that is
absent from the index contributes no ids and shows up in the coverage
block as a filing with no sections used; a chunk that does not fit the
remaining budget is skipped and counted in chunks_dropped.
"""

import re
from collections import defaultdict

import numpy as np

from corpus import normalize
from models import Chunk, Context, Plan
from rules import EDGE_WORDS

SEARCH_K = 60
# Seats per sub-query ahead of the fused order. Summing reciprocal ranks
# favours a chunk that is middling for every metric over the chunk that
# is best for one, and a brief that lists three metrics needs the best
# chunk for each; so each sub-query's top three, after the weights, are
# seated first and the fused order fills the rest.
SUB_QUERY_FLOOR = 3
# Table chunks seated per metric sub-query because a data row is labelled
# with the metric. An examiner reading a filing for "net interest income"
# goes to the row that carries that label, so a table row labelled with
# the metric the question names outranks prose that merely mentions it.
# Rank alone does not get there: on the full corpus the BAC Q3 2025
# summary income statement ranked 43rd of 60 for "net interest income"
# (the phrase is common across 64,612 chunks, so its BM25 weight is low)
# and missed a 24-chunk quota, while the 12-file test index ranked the
# same chunk near the top. Two seats cover the summary table and the
# statement itself; the rest of the quota stays with search.
ROW_LABEL_SEATS = 2
LABEL_WORD_RE = re.compile(r"[a-z0-9]+")
# Candidates kept per quota beyond its chunk count, so assemble still fills
# the quota after cross-quota dedupe and boilerplate drops.
SLACK = 6
NOTE_FACTOR = 1.3
# A table whose parsed columns carry the bucket's period end holds that
# period's figures; the factor lifts it above prose that repeats the
# metric's name. This is where the column provenance from the chunker
# first earns its keep.
PERIOD_FACTOR = 1.3
# Balance floor: chunks every quota receives before any quota takes more.
FLOOR = 6
# Boilerplate: a chunk whose 300-character shingles overlap a kept chunk
# from another filing of the same company by this much is a restatement
# (risk factors repeat almost verbatim year to year) and is dropped.
SHINGLE_CHARS = 300
BOILERPLATE_OVERLAP = 0.6


def file_ranges(index) -> dict[str, tuple[int, int]]:
    return {r["file"]: (r["chunk_start"], r["chunk_end"]) for r in index.files}


def bucket_ids(bucket: dict, ranges: dict) -> np.ndarray:
    """Chunk positions of every file in the bucket, all sections; the
    section weights do the steering."""
    spans = [np.arange(*ranges[f["file"]]) for f in bucket["files"] if f["file"] in ranges]
    return np.concatenate(spans) if spans else np.array([], dtype=np.int64)


def chunk_weight(plan: Plan, chunk: Chunk, period_end: str | None = None) -> float:
    """The plan's weight for a chunk: its section's weight, times the note
    factor when the note title matches the note intent, times the period
    factor when a table column ends on the bucket's period end."""
    weight = plan.sections.get(chunk.item, plan.sections.get("*", 1.0))
    if plan.note_intent and chunk.note_title and re.search(plan.note_intent, chunk.note_title, re.I):
        weight *= NOTE_FACTOR
    if period_end and chunk.kind == "table" and any(c.period_end == period_end for c in chunk.columns):
        weight *= PERIOD_FACTOR
    return weight


def content_words(phrase: str) -> list[str]:
    """The words of a metric phrase that carry its meaning: "estimated
    uninsured deposits" keeps all three, "allowance for credit losses"
    drops "for"."""
    return [w for w in LABEL_WORD_RE.findall(normalize(phrase).lower()) if w not in EDGE_WORDS]


def row_labels(text: str) -> list[str]:
    """The label of every data row in a table chunk. A table row is a line
    whose first pipe cell is the label; it is a data row when a later cell
    holds a digit, which leaves out title lines ("Table 1 | Summary Income
    Statement") and column-header lines."""
    labels = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = line.split("|")
        if any(ch.isdigit() for cell in cells[1:] for ch in cell):
            labels.append(cells[0].strip())
    return labels


def label_matches(label: str, phrase: str, words: list[str]) -> bool:
    """Whether a row label names the metric: it starts with the phrase
    ("Net interest income - reported(a)") or holds every content word of
    it ("CET1 capital ratio" for "CET1 ratio"). Both sides go through
    corpus.normalize and lowercase, so a curly quote or a no-break space
    in either cannot break the match."""
    label = normalize(label).lower().strip()
    if label.startswith(phrase):
        return True
    own = set(LABEL_WORD_RE.findall(label))
    return bool(words) and all(w in own for w in words)


def row_label_seats(bucket: dict, metrics: list[str], index, ranges: dict) -> list[int]:
    """Positions of the table chunks seated for the bucket's metrics.

    Per metric, up to ROW_LABEL_SEATS table chunks of the bucket whose
    data rows carry a label matching the metric phrase, preferring chunks
    with parsed columns (their figures can be tied to a period), then the
    bucket's own filing (files[0], the one the bucket stands for), then
    the newest of the rest, then filing order. The bucket's own filing
    comes before "newest" because an annual baseline can post-date the
    quarter it backs: JPMorgan's Q3 2025 bucket carries the FY2025 10-K,
    whose year end is later than the quarter end. The union over the
    metrics is returned in seating order, each position once.
    """
    primary = bucket["files"][0]["file"]
    tables = []
    for f in bucket["files"]:
        if f["file"] not in ranges:
            continue
        start, end = ranges[f["file"]]
        rank = (f["file"] != primary, -int(f["period_end"].replace("-", "")))
        for pos in range(start, end):
            chunk = index.chunks[pos]
            if chunk.kind == "table":
                tables.append(((chunk.column_source != "parsed",) + rank + (pos,), row_labels(chunk.text)))
    seats: list[int] = []
    for metric in metrics:
        phrase = normalize(metric).lower().strip()
        words = content_words(metric)
        if not phrase:
            continue
        found = sorted(key for key, labels in tables
                       if any(label_matches(label, phrase, words) for label in labels))
        for key in found[:ROW_LABEL_SEATS]:
            if key[-1] not in seats:
                seats.append(key[-1])
    return seats


def retrieve(plan: Plan, index, sub_queries: list[str] | None = None,
             mode: str = "hybrid", pin: bool = True) -> Context:
    """Candidates per quota, then assemble. `mode` and `pin` exist for the
    retrieval ablation; the product path uses the defaults. With `pin`
    off, neither the section leads nor the row-label seats are placed."""
    ranges = file_ranges(index)
    buckets = {(b["ticker"], b["label"]): b for b in plan.buckets}
    queries = sub_queries if sub_queries is not None else plan.sub_queries
    # The first query is the whole question; the rest are the metric phrases.
    metrics = queries[1:]
    per_quota = []
    for quota in plan.quotas:
        bucket = buckets.get((quota["ticker"], quota["bucket"]))
        if bucket is None:
            raise ValueError("quota %s / %s has no bucket in the plan" % (quota["ticker"], quota["bucket"]))
        ids = bucket_ids(bucket, ranges)
        # The ids restrict every query to this quota's filings, which is how
        # a sub-query is tied to its company; the company's name is left out
        # of the query text, where it only rewarded chunks that repeat it.
        weight = {}
        fused: dict[str, float] = defaultdict(float)
        seated: list[str] = []
        for query in queries:
            hits = index.search(query, ids, SEARCH_K, mode=mode)
            for chunk_id, _bm25_rank, _dense_rank, rrf in hits:
                if chunk_id not in weight:
                    weight[chunk_id] = chunk_weight(plan, index.by_id[chunk_id], bucket["period_end"])
                fused[chunk_id] += rrf
            own = sorted(hits, key=lambda h: -h[3] * weight[h[0]])
            seated.extend(h[0] for h in own[:SUB_QUERY_FLOOR] if h[0] not in seated)
        for chunk_id in fused:
            fused[chunk_id] *= weight[chunk_id]
        # The row-label positions are written back to the quota so the plan
        # printed after retrieval shows them beside the section leads; the
        # plan cannot place them itself because it never reads chunk text.
        quota["row_label"] = row_label_seats(bucket, metrics, index, ranges) if pin else []
        leads = [index.chunks[pos].chunk_id for pos in quota["pinned"]] if pin else []
        labelled = [index.chunks[pos].chunk_id for pos in quota["row_label"]]
        labelled = [cid for cid in labelled if cid not in leads]
        # Seating order: section leads, row-label tables, each sub-query's
        # top three, then the fused order. The cut at chunks + SLACK is what
        # makes a seat replace the lowest fused candidates.
        ranked = leads + labelled
        ranked += [cid for cid in seated if cid not in ranked]
        ranked += [cid for cid, _s in sorted(fused.items(), key=lambda kv: -kv[1]) if cid not in ranked]
        per_quota.append({"quota": quota, "bucket": bucket, "pinned": leads, "row_label": labelled,
                          "candidates": ranked[:quota["chunks"] + SLACK]})
    return assemble(plan, per_quota, index)


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------


def shingles(text: str) -> set[int]:
    # hash() of a str is salted per process, which is fine here: shingle
    # sets are only ever compared with each other inside one request.
    flat = " ".join(text.split())
    if len(flat) <= SHINGLE_CHARS:
        return {hash(flat)}
    return {hash(flat[i:i + SHINGLE_CHARS]) for i in range(len(flat) - SHINGLE_CHARS + 1)}


def overlap(new: set[int], kept: set[int]) -> float:
    return len(new & kept) / len(new) if new else 0.0


def assemble(plan: Plan, per_quota: list[dict], index) -> Context:
    """Pick chunks under the budget, order them, number them, render."""
    # Dedupe across quotas: the first quota that lists a chunk keeps it.
    seen: set[str] = set()
    lists: list[list[str]] = []
    for pq in per_quota:
        own = [cid for cid in pq["candidates"] if cid not in seen]
        seen.update(own)
        lists.append(own)
    # Why each seat was placed, for the context rows. Section leads also
    # sort first within their filing; a row-label table keeps its reading
    # position, since the seat only decides whether it is read.
    reason: dict[str, str] = {}
    for pq in per_quota:
        for cid in pq["pinned"]:
            reason.setdefault(cid, "section lead")
        for cid in pq.get("row_label", []):
            reason.setdefault(cid, "row label")
    pinned_ids = {cid for pq in per_quota for cid in pq["pinned"]}

    kept: list[str] = []
    kept_shingles: dict[str, list[tuple[str, set[int]]]] = defaultdict(list)
    tokens = 0
    counts = [0] * len(lists)
    cursors = [0] * len(lists)

    def boilerplate(chunk: Chunk) -> bool:
        own = None
        for other_file, other in kept_shingles[chunk.ticker]:
            if other_file == chunk.file:
                continue
            own = own if own is not None else shingles(chunk.text)
            if overlap(own, other) >= BOILERPLATE_OVERLAP:
                return True
        return False

    def take_next(i: int) -> bool:
        """Move quota i's cursor to the next chunk that fits and keep it."""
        nonlocal tokens
        while cursors[i] < len(lists[i]):
            chunk = index.by_id[lists[i][cursors[i]]]
            cursors[i] += 1
            if tokens + chunk.n_tokens > plan.budget_tokens:
                continue
            # In timeline mode the year-to-year restatement is the point.
            if not plan.timeline and boilerplate(chunk):
                continue
            kept.append(chunk.chunk_id)
            kept_shingles[chunk.ticker].append((chunk.file, shingles(chunk.text)))
            tokens += chunk.n_tokens
            counts[i] += 1
            return True
        return False

    for i, pq in enumerate(per_quota):
        while counts[i] < min(FLOOR, pq["quota"]["chunks"]) and take_next(i):
            pass
    progress = True
    while progress:
        progress = False
        for i, pq in enumerate(per_quota):
            if counts[i] < pq["quota"]["chunks"] and take_next(i):
                progress = True

    order = {c["ticker"]: n for n, c in enumerate(plan.companies)}
    sign = 1 if plan.timeline else -1

    def sort_key(cid: str):
        c = index.by_id[cid]
        return (order.get(c.ticker, len(order)), sign * int(c.period_end.replace("-", "")),
                c.file, 0 if cid in pinned_ids else 1, c.seq)

    ordered = [index.by_id[cid] for cid in sorted(kept, key=sort_key)]
    cids = ["C%d" % (n + 1) for n in range(len(ordered))]
    rendered = "\n\n".join("[%s] %s\n%s" % (cid, c.header, c.text) for cid, c in zip(cids, ordered))
    return Context(
        chunks=ordered,
        n_tokens=tokens,
        cids=cids,
        pinned=[c.chunk_id for c in ordered if c.chunk_id in reason],
        pinned_reason={c.chunk_id: reason[c.chunk_id] for c in ordered if c.chunk_id in reason},
        coverage=coverage_block(plan, ordered),
        rendered=rendered,
        budget_tokens=plan.budget_tokens,
        chunks_dropped=sum(len(l) for l in lists) - len(kept),
    )


# ---------------------------------------------------------------------------
# coverage block
# ---------------------------------------------------------------------------


def section_name(chunk: Chunk) -> str:
    if chunk.item == "COVER":
        return "Cover"
    if chunk.form == "10-Q":
        return "Part %s Item %s" % (chunk.part, chunk.item.split(".")[-1])
    return "Item %s" % chunk.item


def _item_key(name: str):
    m = re.search(r"Item (\d+)([A-C]?)", name)
    part = "Part II" in name
    return (part, int(m.group(1)) if m else 0, m.group(2) if m else "")


def coverage_block(plan: Plan, chunks: list[Chunk]) -> str:
    """One line per company naming every filing in scope, its period end and
    filing date, and the sections the context drew from; then the names
    the corpus does not have and the period notes."""
    used: dict[str, set[str]] = defaultdict(set)
    for c in chunks:
        used[c.file].add(section_name(c))
    lines = []
    for company in plan.companies:
        files: dict[str, dict] = {}
        for bucket in plan.buckets:
            if bucket["ticker"] == company["ticker"]:
                for f in bucket["files"]:
                    files.setdefault(f["file"], f)
        if not files:
            lines.append("%s %s: no filing in scope" % (company["ticker"], company["name"]))
            continue
        entries = sorted(files.values(), key=lambda f: f["period_end"], reverse=not plan.timeline)
        parts = []
        for f in entries:
            ended = "fiscal year ended" if f["form"] == "10-K" else "quarter ended"
            sections = ", ".join(sorted(used[f["file"]], key=_item_key)) or "none"
            parts.append("%s %s (%s %s, filed %s) [used: %s]" % (
                f["form"], f["fiscal_label"], ended, f["period_end"], f["filing_date"], sections))
        lines.append("%s %s: %s" % (company["ticker"], company["name"], "; ".join(parts)))
    if plan.unresolved:
        lines.append("Not in the corpus: " + ", ".join(plan.unresolved))
    if plan.notes:
        lines.append("Period notes: " + "; ".join(plan.notes))
    return "\n".join(lines)
