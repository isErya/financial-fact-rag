"""Turn raw model text into an Answer.

With the JSON schema set on the request the reply is normally parsed by
the client; this module is the fallback for the cases that still reach the
raw text: a fenced block, JSON wrapped in prose, a trailing comma, claims
without ids. It salvages what is unambiguous and raises ParseError with a
one-line reason for anything else. A body cut off before its closing brace
is reported as "truncated" and never completed by guesswork, because a
repaired answer would carry claims the model did not finish writing.

Whether a citation exists or a quote is in the excerpt needs the context,
so resolver.py checks that.
"""

import json
import re

from pydantic import ValidationError

from models import Answer

FENCE_RE = re.compile(r"```[A-Za-z0-9_-]*[ \t]*\n?|```")
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
CID_RE = re.compile(r"^\[?\s*(C\d+)\s*\]?$")


class ParseError(Exception):
    """Model output could not be salvaged. The message is one line."""


def strip_fences(text: str) -> str:
    return FENCE_RE.sub("", text)


def first_object(text: str) -> str:
    """The first balanced top-level JSON object in `text`, string-aware so
    a brace inside a quoted passage does not close the object."""
    start = text.find("{")
    if start < 0:
        raise ParseError("no JSON object in the reply")
    depth = 0
    in_string = False
    escaped = False
    for pos in range(start, len(text)):
        ch = text[pos]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:pos + 1]
    raise ParseError("truncated")


def load_object(body: str) -> dict:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        try:
            data = json.loads(TRAILING_COMMA_RE.sub(r"\1", body))
        except json.JSONDecodeError as exc:
            raise ParseError("invalid JSON: %s" % exc.msg) from exc
    if not isinstance(data, dict):
        raise ParseError("top level is not an object")
    return data


def tidy(data: dict) -> dict:
    """Claim ids in "K1".. order where missing, and only C-ids kept as
    citations. Anything else stays as written for the schema to judge."""
    claims = data.get("claims")
    if isinstance(claims, list):
        for n, claim in enumerate(claims, 1):
            if not isinstance(claim, dict):
                continue
            if not str(claim.get("id") or "").strip():
                claim["id"] = "K%d" % n
            citations = claim.get("citations")
            if isinstance(citations, list):
                kept = []
                for cid in citations:
                    m = CID_RE.match(str(cid).strip())
                    if m:
                        kept.append(m.group(1))
                claim["citations"] = kept
    return data


def parse_answer(raw_text: str) -> Answer:
    body = first_object(strip_fences(raw_text or ""))
    data = tidy(load_object(body))
    try:
        return Answer.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ())) or "answer"
        raise ParseError("%s: %s" % (where, first.get("msg", "invalid"))) from exc
