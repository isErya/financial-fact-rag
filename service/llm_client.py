"""The model backends behind the TextModel protocol ask.py calls.

Two backends ship. AnthropicModel is the one API class: one request per
generate call, no SDK retries, no fallback model, no repair request. The
structured output is requested on that same request through the SDK's
JSON-schema output format, so the reply is schema-shaped whenever the model
finished, and parsing.py only ever runs as the fallback on the raw text.
FakeModel serves a stored reply for a question it knows (eval/fixtures) and
a minimal canned Answer for any other, so every test and a keyless laptop
run the whole pipeline, evidence checks included.

No other module names the SDK. Swapping providers means one more class with
a `name` and a `generate(system, user, schema, guard)`.

Failure classes, so the caller can react to each:

- ModelUnavailable: the backend could not be reached or did not answer
  (network, timeout, 5xx, rate limit). The request may not have happened.
- ModelProtocolError: the backend answered, but the response is not usable
  (no content, no text block on a finished reply).
- ModelConfigError: the request itself is wrong (4xx: bad key, unknown
  model, unsupported parameter) or the backend name is unknown. Never
  succeeds on a retry.
- RequestBudgetExceeded: generate was called a second time inside one ask.
  The guard that raises it is created per ask and handed to generate, so a
  code path that would quietly make a second request fails loudly instead.

The SDK's own parse helper (client.messages.parse) validates the reply
text against the schema before returning and raises when the text is cut
off at max_tokens, which would lose the stop reason and the partial text.
The request therefore goes through client.messages.create with the same
schema the parse helper would have sent (anthropic.transform_schema over
the pydantic JSON schema), and validation happens here, where a failure
keeps the body.
"""

import glob
import hashlib
import json
import os
import re
import time
from typing import Protocol

from pydantic import BaseModel, ValidationError

import config
from corpus import normalize
from models import LLMResult


class ModelUnavailable(Exception):
    """The backend could not be reached or did not answer in time."""


class ModelProtocolError(Exception):
    """The backend answered, but the response is not usable."""


class ModelConfigError(Exception):
    """The request or the configuration is wrong; a retry cannot help."""


class RequestBudgetExceeded(Exception):
    """A second model request inside one ask."""


class RequestGuard:
    """One per ask call. generate takes a unit before each request, so the
    count of requests an ask made is the guard's count and a second
    request raises before it is sent."""

    def __init__(self, limit: int = 1):
        self.limit = limit
        self.taken = 0

    def take(self, backend: str) -> None:
        if self.taken >= self.limit:
            raise RequestBudgetExceeded(
                "%s: request %d of an ask that allows %d" % (backend, self.taken + 1, self.limit))
        self.taken += 1


class TextModel(Protocol):
    name: str

    def generate(self, system: str, user: str, schema: type[BaseModel],
                 guard: RequestGuard | None = None) -> LLMResult: ...


# ---------------------------------------------------------------------------
# The API backend
# ---------------------------------------------------------------------------

# Status codes the SDK would have retried had retries been on (README,
# "Retries"): the request may succeed later, so they are "unavailable",
# and every other 4xx is a request that will fail again the same way.
TRANSIENT_STATUS = {408, 409, 429}


class AnthropicModel:
    backend = "anthropic"

    def __init__(self, model: str = config.LLM_MODEL, api_key: str | None = None,
                 effort: str = config.LLM_EFFORT, max_tokens: int = config.LLM_MAX_TOKENS,
                 timeout: float = config.LLM_TIMEOUT_SECONDS, client=None):
        # Imported here so the fake backend runs where the SDK is absent.
        try:
            import anthropic
        except ImportError as exc:
            raise ModelConfigError("the anthropic SDK is not installed: %s" % exc) from exc
        self._sdk = anthropic
        self.name = model
        self.effort = effort
        self.max_tokens = max_tokens
        # README "Client Initialization" and "Retries" / "Timeouts": the key
        # resolves from ANTHROPIC_API_KEY when api_key is None; max_retries=0
        # makes one request per call and timeout is in seconds.
        self.client = client or anthropic.Anthropic(
            api_key=api_key, max_retries=config.LLM_MAX_RETRIES, timeout=timeout)

    def generate(self, system: str, user: str, schema: type[BaseModel],
                 guard: RequestGuard | None = None) -> LLMResult:
        if guard is not None:
            guard.take(self.backend)
        attempts = 1
        started = time.monotonic()
        # README "Manual Cache Control": the system text is a block with an
        # ephemeral cache breakpoint, so repeated asks over the same policy
        # read the policy from cache. tool-use.md "Raw Schema": the reply
        # format is a JSON schema on output_config; README "Extended
        # Thinking": effort rides on the same output_config.
        try:
            response = self.client.messages.create(
                model=self.name,
                max_tokens=self.max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema",
                               "schema": self._sdk.transform_schema(schema.model_json_schema())},
                },
            )
        except self._sdk.APIConnectionError as exc:
            # APITimeoutError is a subclass (README "Timeouts").
            raise ModelUnavailable("%s: %s" % (type(exc).__name__, exc)) from exc
        except self._sdk.APIStatusError as exc:
            if exc.status_code >= 500 or exc.status_code in TRANSIENT_STATUS:
                raise ModelUnavailable("HTTP %d: %s" % (exc.status_code, exc.message)) from exc
            raise ModelConfigError("HTTP %d: %s" % (exc.status_code, exc.message)) from exc
        latency_ms = int((time.monotonic() - started) * 1000)
        return self._result(response, schema, attempts, latency_ms)

    def _result(self, response, schema: type[BaseModel], attempts: int, latency_ms: int) -> LLMResult:
        """An LLMResult from a response that arrived, whatever it says.
        completed is 1 from here on: the request happened."""
        content = getattr(response, "content", None)
        if content is None:
            raise ModelProtocolError("response carries no content list")
        raw_text = "".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")
        stop_reason = getattr(response, "stop_reason", None)
        parsed = None
        error = None
        if stop_reason == "end_turn":
            if not raw_text:
                raise ModelProtocolError("finished reply carries no text block")
            try:
                parsed = schema.model_validate_json(raw_text)
            except ValidationError as exc:
                error = "reply is not the schema: " + first_line(str(exc))
        else:
            # README "Stop Reasons": max_tokens is a cut-off body, refusal is
            # a decline; either way the answer did not arrive and the raw
            # text is kept for the record.
            error = "stop reason %s" % stop_reason
        usage = getattr(response, "usage", None)
        return LLMResult(
            parsed=parsed,
            raw_text=raw_text,
            stop_reason=stop_reason,
            usage=usage_dict(usage),
            model=getattr(response, "model", self.name),
            backend=self.backend,
            # README "Response Helpers": _request_id is public despite the
            # underscore and holds the request-id header.
            request_id=getattr(response, "_request_id", None),
            latency_ms=latency_ms,
            attempts=attempts,
            completed=1,
            error=error,
        )


def usage_dict(usage) -> dict:
    """The four token counts a bill is computed from, zero when absent."""
    return {key: int(getattr(usage, key, 0) or 0)
            for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                        "cache_read_input_tokens")}


def first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


# ---------------------------------------------------------------------------
# The fake backend
# ---------------------------------------------------------------------------

QUESTION_BLOCK_RE = re.compile(r"<question>\n(.*?)\n</question>", re.S)
EXCERPT_C1_RE = re.compile(r"\[C1\] ([^\n]*)\n(.*?)(?=\n\n\[C2\] |\Z)", re.S)
HEADER_TICKER_RE = re.compile(r"\(([A-Z][A-Z.-]*), CIK \d+\)")
HEADER_PERIOD_RE = re.compile(r"(quarter|fiscal year) ended (\d{4}-\d{2}-\d{2})")
CANNED_QUOTE_WORDS = 12


def question_key(question: str) -> str:
    """The fixture key: a hash of the question after the corpus
    normalization, whitespace collapse and casefold, so a re-typed question
    with a curly apostrophe or a double space still replays."""
    flat = " ".join(normalize(question).split()).casefold()
    return hashlib.sha1(flat.encode("utf-8")).hexdigest()[:16]


def question_of(user: str) -> str:
    """The question inside the rendered user text, or the whole text when
    the prompt did not use the question block."""
    m = QUESTION_BLOCK_RE.search(user)
    return m.group(1) if m else user


def load_fixtures(fixtures_dir: str) -> dict[str, dict]:
    """Every stored reply under the directory, keyed by question."""
    records = {}
    for path in sorted(glob.glob(os.path.join(fixtures_dir, "*.json"))):
        with open(path) as fh:
            record = json.load(fh)
        key = record.get("question_key") or question_key(record["question"])
        record["path"] = path
        records[key] = record
    return records


def save_fixture(path: str, question: str, result: LLMResult, prompt_version: str) -> None:
    """Store one model reply so the fake backend can replay it. parsed is
    written as plain JSON; the reader validates it back into the schema."""
    parsed = result.parsed
    if isinstance(parsed, BaseModel):
        parsed = parsed.model_dump()
    record = {
        "question": question,
        "question_key": question_key(question),
        "prompt_version": prompt_version,
        "saved": time.strftime("%Y-%m-%d"),
        "result": {
            "parsed": parsed,
            "raw_text": result.raw_text,
            "stop_reason": result.stop_reason,
            "usage": result.usage,
            "model": result.model,
            "backend": result.backend,
            "request_id": result.request_id,
            "latency_ms": result.latency_ms,
            "attempts": result.attempts,
            "completed": result.completed,
            "error": result.error,
        },
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(record, fh, indent=1)


class FakeModel:
    """Answers without a network. `raw_text`, when given, is returned as an
    unparsed reply on every call so the parsing fallback can be exercised
    with fenced, wrapped or cut-off text."""

    backend = "fake"

    def __init__(self, fixtures_dir: str = config.FIXTURES_DIR, model: str = "fake",
                 raw_text: str | None = None):
        self.name = model
        self.fixtures = load_fixtures(fixtures_dir) if os.path.isdir(fixtures_dir) else {}
        self.raw_text = raw_text

    def generate(self, system: str, user: str, schema: type[BaseModel],
                 guard: RequestGuard | None = None) -> LLMResult:
        if guard is not None:
            guard.take(self.backend)
        usage = {"input_tokens": estimate(system) + estimate(user), "output_tokens": 0,
                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        if self.raw_text is not None:
            usage["output_tokens"] = estimate(self.raw_text)
            return self._fresh(None, self.raw_text, usage)
        record = self.fixtures.get(question_key(question_of(user)))
        if record is not None:
            return self._replay(record, schema)
        answer = canned_answer(user, schema)
        raw_text = answer.model_dump_json(indent=1)
        usage["output_tokens"] = estimate(raw_text)
        return self._fresh(answer, raw_text, usage)

    def _fresh(self, parsed, raw_text: str, usage: dict) -> LLMResult:
        return LLMResult(parsed=parsed, raw_text=raw_text, stop_reason="end_turn", usage=usage,
                         model=self.name, backend=self.backend, request_id=None, latency_ms=0,
                         attempts=1, completed=1, error=None)

    def _replay(self, record: dict, schema: type[BaseModel]) -> LLMResult:
        stored = dict(record["result"])
        if stored.get("parsed") is not None:
            stored["parsed"] = schema.model_validate(stored["parsed"])
        return LLMResult(replayed=True, **stored)


def estimate(text: str) -> int:
    # Four characters per token is the usual English average; the fake
    # backend only needs a plausible bill line.
    return (len(text) + 3) // 4


def canned_answer(user: str, schema: type[BaseModel]):
    """The minimal Answer for an unknown question: one claim citing C1 with
    a quote copied from the first words of C1's text, so the resolver has a
    real citation to check. The claim text carries no figure on purpose;
    the point of the canned answer is the plumbing, never the content."""
    m = EXCERPT_C1_RE.search(user)
    header, body = (m.group(1), m.group(2)) if m else ("", "")
    ticker = HEADER_TICKER_RE.search(header)
    period = HEADER_PERIOD_RE.search(header)
    quote = " ".join(body.split()[:CANNED_QUOTE_WORDS])
    tickers = [ticker.group(1)] if ticker else []
    claim = {
        "id": "K1",
        "text": "Excerpt C1%s opens with the quoted passage." % (" for " + tickers[0] if tickers else ""),
        "tickers": tickers,
        "period_end": period.group(2) if period else "",
        "period_kind": "quarter" if period and period.group(1) == "quarter" else "fiscal_year",
        "citations": ["C1"] if m else [],
        "quote": quote,
    }
    return schema.model_validate({
        "summary": [{"text": "No model was asked; this stand-in answer quotes the opening of the "
                             "first excerpt so the pipeline can be exercised end to end.",
                     "claim_ids": ["K1"]}],
        "claims": [claim],
        "table": [],
        "not_comparable": [],
        "gaps": ["The stand-in backend does not read the excerpts; the question is unanswered."],
    })


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def from_env(backend: str | None = None, model: str | None = None) -> TextModel:
    """The backend LLM_MODEL_BACKEND names ("anthropic" or "fake"), with the
    model id from LLM_MODEL. Arguments override the environment for tests."""
    backend = backend or config.LLM_MODEL_BACKEND
    model = model or config.LLM_MODEL
    if backend == "fake":
        return FakeModel(config.FIXTURES_DIR)
    if backend == "anthropic":
        return AnthropicModel(model=model)
    raise ModelConfigError("unknown LLM_MODEL_BACKEND %r (use anthropic or fake)" % backend)
