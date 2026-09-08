"""Data types for the filing question-answering pipeline.

Flow: corpus.py parses each zip member into a Filing whose body carries
Sections and their Notes as character offsets. Milestone 2 cuts Sections into
Chunks and indexes them. Milestone 3 turns a question into a Plan, gathers a
Context of chunks, and asks the model for an Answer; the model reply travels
inside an LLMResult and its citations are checked into EvidenceChecks.

Pipeline records are stdlib dataclasses because nothing validates them at a
boundary. The Answer tree is pydantic because it is the one shape the model
must produce and a malformed reply has to fail loudly at parse time.

Imports nothing else from the service, so any module can import this one.
"""

from dataclasses import dataclass, field

from pydantic import BaseModel


@dataclass
class Column:
    """One numeric column of a financial table, as far as the header says."""

    index: int
    label: str
    period_end: str | None
    duration: str | None


@dataclass
class Note:
    """A note to the financial statements. Offsets index into Filing.body."""

    number: int | None
    title: str
    start: int
    end: int


@dataclass
class Section:
    """One item of a filing, from its heading to the next accepted heading.

    `item` is "1A", "7", "8" for a 10-K and "I.1", "II.1A" for a 10-Q, with
    "COVER" for the text before the first heading. `is_pointer_stub` marks a
    section that only points at the last annual report instead of restating it.
    """

    part: str | None
    item: str
    title: str
    start: int
    end: int
    is_pointer_stub: bool
    notes: list[Note] = field(default_factory=list)


@dataclass
class Filing:
    """One parsed filing. `body` is the cleaned text every offset refers to."""

    file: str
    cik: str
    ticker: str
    company: str
    form: str
    filing_date: str
    period_end: str
    period_source: str
    fiscal_year: int
    fiscal_quarter: int | None
    fiscal_label: str
    fye_month: int
    url: str
    body: str
    sections: list[Section] = field(default_factory=list)


@dataclass
class Chunk:
    """A retrievable slice of a Section. Populated from milestone 2 on."""

    chunk_id: str
    file: str
    cik: str
    ticker: str
    company: str
    form: str
    part: str | None
    item: str
    item_title: str
    note_title: str | None
    kind: str
    period_end: str
    fiscal_year: int
    fiscal_quarter: int | None
    fiscal_label: str
    filing_date: str
    seq: int
    char_start: int
    char_end: int
    units: str | None
    units_source: str | None
    columns: list[Column]
    column_source: str | None
    header: str
    text: str
    n_tokens: int


@dataclass
class Plan:
    """What the question resolved to before any retrieval. Milestone 3."""

    companies: list[dict]
    unresolved: list[str]
    not_covered: list[str]
    stale: list[str]
    status: str
    period_mode: str
    buckets: list[dict]
    sections: dict
    note_intent: str | None
    comparison: bool
    timeline: bool
    quotas: list[dict]
    sub_queries: list[str]
    budget_tokens: int
    # Scope notes the coverage block prints: missing years, comparative
    # columns, pointer stubs pinned to a 10-K, fiscal-quarter notes, stale
    # filers, the narrowing to six companies.
    notes: list[str] = field(default_factory=list)


@dataclass
class Context:
    """The chunks handed to the model for one question. Milestone 3."""

    chunks: list[Chunk]
    n_tokens: int
    # Filled by retrieve.assemble: the C-id per chunk (same order as
    # `chunks`), the ids that were seated rather than found by search and
    # why ("section lead" or "row label"), the coverage block and the
    # excerpts block exactly as the model will see them, and the budget
    # accounting the dry run reports.
    cids: list[str] = field(default_factory=list)
    pinned: list[str] = field(default_factory=list)
    pinned_reason: dict[str, str] = field(default_factory=dict)
    coverage: str = ""
    rendered: str = ""
    budget_tokens: int = 0
    chunks_dropped: int = 0


@dataclass
class LLMResult:
    """One model call, kept whole so a bad answer can be replayed."""

    parsed: object
    raw_text: str
    stop_reason: str | None
    usage: dict
    model: str
    backend: str
    request_id: str | None
    latency_ms: int
    attempts: int
    completed: int
    error: str | None
    # True when the fake backend served a stored result instead of
    # composing one, so a replayed answer is never mistaken for a fresh
    # request in a report.
    replayed: bool = False


@dataclass
class EvidenceChecks:
    """How many of the answer's citations and figures checked out."""

    quotes_found: tuple[int, int]
    figures_in_quote: tuple[int, int]
    columns_matched: tuple[int, int]
    columns_unverified: int
    units_declared: tuple[int, int]
    flags: list[dict]
    unlinked_sentences: list[str]


# Model-facing answer schema. All fields are required; lists may be empty.


class Sentence(BaseModel):
    text: str
    claim_ids: list[str]


class Claim(BaseModel):
    id: str
    text: str
    tickers: list[str]
    period_end: str
    period_kind: str
    citations: list[str]
    quote: str


class Cell(BaseModel):
    column: str
    text: str
    claim_ids: list[str]


class Row(BaseModel):
    dimension: str
    cells: list[Cell]


class NotComparable(BaseModel):
    dimension: str
    tickers: list[str]
    reason: str


class Answer(BaseModel):
    summary: list[Sentence]
    claims: list[Claim]
    table: list[Row]
    not_comparable: list[NotComparable]
    gaps: list[str]
