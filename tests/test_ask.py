"""The model step end to end with the fake backend: one request per ask,
refusals before any request, the parsing fallback, replayed fixtures.
No test here reaches a network."""

import json
import os

import pytest

import config
import llm_client
import parsing
from ask import ask, prepare
from conftest import ROOT
from index import load
from llm_client import FakeModel, RequestBudgetExceeded, save_fixture
from models import Answer, EvidenceChecks, LLMResult
from test_resolver import NII, load_fixture
from test_retrieve import FULL_INDEX_BUILT, FULL_INDEX_DIR

Q01 = "What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?"
Q06 = "What was JPMorgan's net interest income for the third quarter of 2025?"
Q10 = "What did Wells Fargo say about credit risk?"
Q11 = "What did Apple report in 2019?"

MINIMAL = {"summary": [{"text": "One claim.", "claim_ids": ["K1"]}],
           "claims": [{"id": "K1", "text": "Apple lists supply risk.", "tickers": ["AAPL"],
                       "period_end": "2025-09-27", "period_kind": "fiscal_year", "citations": ["C1"],
                       "quote": "Item 1A. Risk Factors"}],
           "table": [], "not_comparable": [], "gaps": []}


@pytest.fixture
def fake(tmp_path):
    return FakeModel(str(tmp_path))


def test_full_ask_on_q01_makes_one_request(tuning_index, registry, fake):
    payload = ask(Q01, tuning_index, registry, model=fake)
    assert payload["status"] == "ok"
    assert payload["llm_attempts"] == 1 and payload["llm_completed"] == 1
    assert isinstance(payload["answer"], Answer)
    assert isinstance(payload["checks"], EvidenceChecks)
    # The shipped default, by name from prompts.py rather than a literal, so a
    # new winner in the prompt log does not fail this test.
    import prompts
    assert payload["prompt_version"] == prompts.PROMPT_VERSION and payload["replayed"] is False
    assert payload["llm_error"] is None and payload["cost_usd"] is None
    assert isinstance(payload["budget"]["input_tokens_actual"], int)
    assert set(payload["timing_ms"]) == {"plan_and_retrieve", "render", "model", "parse", "check"}
    assert "<question>\n%s\n</question>" % Q01 in payload["prompt"]["user"]
    assert "private equity" in payload["prompt"]["system"]
    # The canned claim quotes the first words of C1, so the quote is located.
    assert payload["checks"].quotes_located == (1, 1)


def test_a_second_generate_inside_one_ask_raises(tuning_index, registry, fake):
    class Twice:
        name = "twice"

        def generate(self, system, user, schema, guard=None):
            fake.generate(system, user, schema, guard=guard)
            return fake.generate(system, user, schema, guard=guard)

    with pytest.raises(RequestBudgetExceeded):
        ask(Q01, tuning_index, registry, model=Twice())


def test_refusals_never_reach_the_model(tuning_index, registry):
    class Never:
        name = "never"

        def generate(self, system, user, schema, guard=None):
            raise AssertionError("a refused question reached the model")

    for question, status in ((Q10, "not_covered"), (Q11, "period_not_covered")):
        payload = ask(question, tuning_index, registry, model=Never())
        assert payload["status"] == status
        assert payload["llm_attempts"] == 0 and payload["llm_completed"] == 0
        assert "answer" not in payload


def test_fenced_json_with_trailing_prose_still_parses(tuning_index, registry, tmp_path):
    raw = "Here is the answer:\n```json\n%s\n```\nLet me know if you need more." % json.dumps(MINIMAL, indent=1)
    payload = ask(Q01, tuning_index, registry, model=FakeModel(str(tmp_path), raw_text=raw))
    assert payload["llm_error"] is None
    assert payload["answer"].claims[0].id == "K1"
    assert payload["llm_attempts"] == 1


def test_truncated_object_yields_llm_error_without_a_second_request(tuning_index, registry, tmp_path):
    raw = '{"summary": [{"text": "One claim.", "claim_ids": ["K1"]}], "claims": [{"id": "K1", "text": "cut'
    payload = ask(Q01, tuning_index, registry, model=FakeModel(str(tmp_path), raw_text=raw))
    assert payload["answer"] is None and payload["checks"] is None
    assert payload["llm_error"] == "truncated"
    assert payload["llm_attempts"] == 1 and payload["llm_completed"] == 1
    assert payload["raw_text"] == raw


def test_replayed_fixture_sets_replayed(tuning_index, registry, tmp_path):
    stored = LLMResult(parsed=Answer.model_validate(MINIMAL), raw_text=json.dumps(MINIMAL), stop_reason="end_turn",
                       usage={"input_tokens": 12000, "output_tokens": 300, "cache_creation_input_tokens": 0,
                              "cache_read_input_tokens": 0},
                       model="claude-opus-5", backend="anthropic", request_id="req_fixture", latency_ms=4200,
                       attempts=1, completed=1, error=None)
    save_fixture(str(tmp_path / "q01.json"), Q01, stored, "v1")
    payload = ask(Q01, tuning_index, registry, model=FakeModel(str(tmp_path)))
    assert payload["replayed"] is True
    assert payload["request_id"] == "req_fixture" and payload["model"] == "claude-opus-5"
    assert payload["answer"].claims[0].quote == "Item 1A. Risk Factors"
    # Replayed usage is billed at the stored model's price: 12000 input at
    # $5 per million plus 300 output at $25 per million.
    assert payload["cost_usd"] == pytest.approx(0.0675)


def test_default_model_comes_from_the_environment(tuning_index, registry, monkeypatch):
    monkeypatch.setattr(config, "LLM_MODEL_BACKEND", "fake")
    payload = ask("What are Apple's risk factors?", tuning_index, registry)
    assert payload["backend"] == "fake" and payload["llm_attempts"] == 1


def test_parse_answer_salvage_rules():
    data = json.loads(json.dumps(MINIMAL))
    data["claims"][0]["id"] = ""
    data["claims"][0]["citations"] = ["[C1]", "C7", "page 4", "K2"]
    text = json.dumps(data)[:-1] + ",}"
    answer = parsing.parse_answer(text)
    assert answer.claims[0].id == "K1"
    assert answer.claims[0].citations == ["C1", "C7"]
    with pytest.raises(parsing.ParseError, match="no JSON object"):
        parsing.parse_answer("no structure here")
    with pytest.raises(parsing.ParseError, match="truncated"):
        parsing.parse_answer('{"summary": [{"text": "brace } inside a string", "claim_ids": []}')
    with pytest.raises(parsing.ParseError, match="claims"):
        parsing.parse_answer('{"summary": [], "table": [], "not_comparable": [], "gaps": []}')


@pytest.mark.skipif(not FULL_INDEX_BUILT, reason="full index not built")
def test_q06_fixture_binds_on_the_full_index(registry):
    # The one full-corpus check of the model step: the row-label seat still
    # carries the summary income statement into the q06 context over
    # 64,612 chunks, so the fixture's cited chunk is there to check.
    index = load(FULL_INDEX_DIR)
    prepared = prepare(Q06, index, registry)
    cid_of = {c.chunk_id: cid for cid, c in zip(prepared.context.cids, prepared.context.chunks)}
    assert NII in cid_of
    answer = load_fixture("q06_answer.json", cid_of)
    import resolver
    checks = resolver.check(answer, prepared.context, dict(zip(prepared.context.cids, prepared.context.chunks)),
                            prepared.plan, set(registry["companies"]))
    assert not [f for f in checks.flags if f["where"] == "K1"]
    assert checks.columns_matched == (1, 2)
