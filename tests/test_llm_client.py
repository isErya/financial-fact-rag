"""The API class is built with one request per call and a 300 second
timeout; its reply handling is exercised through a stub client so no test
reaches a network. The fake backend is what from_env("fake") returns."""

import types

import httpx2
import pytest

import anthropic
import config
from llm_client import (AnthropicModel, FakeModel, ModelConfigError, ModelUnavailable, RequestBudgetExceeded,
                        RequestGuard, from_env, question_key)
from models import Answer

SCHEMA_JSON = ('{"summary": [{"text": "s", "claim_ids": ["K1"]}], "claims": [{"id": "K1", "text": "t", '
               '"tickers": ["JPM"], "period_end": "2025-09-30", "period_kind": "quarter", "citations": ["C1"], '
               '"quote": "q"}], "table": [], "not_comparable": [], "gaps": []}')


def test_anthropic_client_is_built_with_no_retries_and_a_300s_timeout():
    model = AnthropicModel(api_key="test-key")
    assert model.client.max_retries == 0
    assert model.client.timeout == 300
    assert model.name == config.LLM_MODEL == "claude-opus-5"
    assert model.effort == "medium" and model.max_tokens == 16000


def test_from_env_fake_returns_the_fake_backend():
    assert isinstance(from_env("fake"), FakeModel)
    with pytest.raises(ModelConfigError):
        from_env("other")


def test_guard_allows_one_request():
    guard = RequestGuard()
    guard.take("fake")
    with pytest.raises(RequestBudgetExceeded):
        guard.take("fake")


class StubClient:
    """Records the request and answers with what the test hands it."""

    def __init__(self, response=None, error=None):
        self.calls = []
        self.messages = types.SimpleNamespace(create=self._create)
        self.response = response
        self.error = error

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def response(stop_reason: str, text: str | None):
    content = [types.SimpleNamespace(type="text", text=text)] if text is not None else []
    usage = types.SimpleNamespace(input_tokens=9000, output_tokens=400, cache_creation_input_tokens=800,
                                  cache_read_input_tokens=0)
    return types.SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage, model="claude-opus-5",
                                 _request_id="req_stub")


def test_generate_sends_one_schema_shaped_request_and_reads_the_reply():
    stub = StubClient(response("end_turn", SCHEMA_JSON))
    model = AnthropicModel(api_key="test-key", client=stub)
    guard = RequestGuard()
    result = model.generate("policy", "question", Answer, guard=guard)
    assert len(stub.calls) == 1 and guard.taken == 1
    request = stub.calls[0]
    assert request["model"] == "claude-opus-5" and request["max_tokens"] == 16000
    assert request["system"] == [{"type": "text", "text": "policy", "cache_control": {"type": "ephemeral"}}]
    assert request["messages"] == [{"role": "user", "content": "question"}]
    assert request["output_config"]["effort"] == "medium"
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["output_config"]["format"]["schema"]["required"] == [
        "summary", "claims", "table", "not_comparable", "gaps"]
    assert "thinking" not in request and "fallbacks" not in request
    assert isinstance(result.parsed, Answer)
    assert result.attempts == 1 and result.completed == 1
    assert result.request_id == "req_stub" and result.stop_reason == "end_turn"
    assert result.usage == {"input_tokens": 9000, "output_tokens": 400, "cache_creation_input_tokens": 800,
                            "cache_read_input_tokens": 0}
    with pytest.raises(RequestBudgetExceeded):
        model.generate("policy", "question", Answer, guard=guard)


def test_max_tokens_and_refusal_keep_the_body_with_no_answer():
    cut = SCHEMA_JSON[:60]
    result = AnthropicModel(api_key="k", client=StubClient(response("max_tokens", cut))).generate("p", "u", Answer)
    assert result.parsed is None and result.raw_text == cut
    assert result.stop_reason == "max_tokens" and result.completed == 1 and result.attempts == 1
    result = AnthropicModel(api_key="k", client=StubClient(response("refusal", None))).generate("p", "u", Answer)
    assert result.parsed is None and result.raw_text == "" and result.stop_reason == "refusal"
    assert result.completed == 1


def test_sdk_failures_map_to_typed_errors():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    timeout = anthropic.APITimeoutError(request=request)
    with pytest.raises(ModelUnavailable):
        AnthropicModel(api_key="k", client=StubClient(error=timeout)).generate("p", "u", Answer)
    overloaded = anthropic.APIStatusError("overloaded", response=httpx2.Response(529, request=request), body=None)
    with pytest.raises(ModelUnavailable):
        AnthropicModel(api_key="k", client=StubClient(error=overloaded)).generate("p", "u", Answer)
    bad_key = anthropic.APIStatusError("bad key", response=httpx2.Response(401, request=request), body=None)
    with pytest.raises(ModelConfigError):
        AnthropicModel(api_key="k", client=StubClient(error=bad_key)).generate("p", "u", Answer)


def test_fake_model_canned_answer_quotes_the_first_twelve_words_of_c1(tmp_path):
    user = ("<coverage>\nJPM\n</coverage>\n\n<question>\nWhat happened?\n</question>\n\n<excerpts>\n"
            "[C1] JPMorgan Chase & Co (JPM, CIK 19617) | 10-Q FY2025 Q3, quarter ended 2025-09-30, filed 2025-11-04\n"
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen\n\n"
            "[C2] Other | 10-K FY2025, fiscal year ended 2025-12-31\nmore text\n</excerpts>\n\nAnswer.")
    result = FakeModel(str(tmp_path)).generate("policy", user, Answer)
    claim = result.parsed.claims[0]
    assert claim.quote == "one two three four five six seven eight nine ten eleven twelve"
    assert claim.citations == ["C1"] and claim.tickers == ["JPM"]
    assert claim.period_end == "2025-09-30" and claim.period_kind == "quarter"
    assert result.replayed is False and result.attempts == 1


def test_question_key_ignores_typography_and_spacing():
    typed = "What was JPMorgan\u2019s  net interest income?"
    assert question_key(typed) == question_key("what was jpmorgan's net interest income?")
