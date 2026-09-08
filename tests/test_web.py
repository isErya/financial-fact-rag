"""The serve layer through FastAPI's TestClient: the page, /health,
/coverage, /ask in both modes, and the failure state a fresh stack can
land in (no index yet). The fake backend answers every full run, so
nothing here reaches a network.

The app reads config at startup (lifespan), so each client below points
config at the session's small tuning index before the lifespan runs.
"""

import os

import pytest
from fastapi.testclient import TestClient

import config
import web
from conftest import ROOT

Q01 = "What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?"
Q_UNKNOWN = "What did Wells Fargo say about credit risk?"


@pytest.fixture(scope="module")
def settings(tuning_index_dir, tmp_path_factory):
    """Config pointed at the tuning index, the fake backend and an empty
    fixtures directory (so no stored reply is replayed), for every client
    in this module. Restored when the module is done."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LLM_MODEL_BACKEND", "fake")
        mp.setattr(config, "LLM_MODEL_BACKEND", "fake")
        mp.setattr(config, "INDEX_DIR", tuning_index_dir)
        mp.setattr(config, "CORPUS_ZIP", os.path.join(ROOT, "data", "edgar_corpus.zip"))
        mp.setattr(config, "COMPANIES_FILE", os.path.join(ROOT, "service", "companies.yaml"))
        mp.setattr(config, "FIXTURES_DIR", str(tmp_path_factory.mktemp("no-fixtures")))
        yield mp


@pytest.fixture(scope="module")
def client(settings):
    with TestClient(web.app) as c:
        yield c


def test_page_and_static_files_are_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "filing desk" in page.text
    assert 'class="bh"' in page.text
    script = client.get("/static/app.js")
    assert script.status_code == 200
    # The page inserts model and filing text with textContent only.
    assert "innerHTML" not in script.text
    assert "textContent" in script.text
    assert client.get("/static/bauhaus.css").status_code == 200


def test_health_reports_the_index_and_the_fake_backend(client):
    body = client.get("/health").json()
    assert body["chunks"] > 0 and body["files"] > 0 and body["tickers"] > 0
    assert body["index_fingerprint"] and body["index_error"] is None
    assert body["dense"] is False
    assert body["backend"] == "fake" and body["llm_ready"] is True
    assert body["model"] == config.LLM_MODEL


def test_llm_ready_follows_the_key_variable_for_the_api_backend(client, monkeypatch):
    monkeypatch.setattr(config, "LLM_MODEL_BACKEND", "anthropic")
    monkeypatch.delenv(config.LLM_API_KEY_ENV, raising=False)
    assert client.get("/health").json()["llm_ready"] is False
    monkeypatch.setenv(config.LLM_API_KEY_ENV, "sk-test")
    assert client.get("/health").json()["llm_ready"] is True


def test_coverage_lists_every_company_with_the_stale_note(client):
    body = client.get("/coverage").json()
    assert body["count"] == 54 and len(body["companies"]) == 54
    by_ticker = {c["ticker"]: c for c in body["companies"]}
    ge = by_ticker["GE"]
    assert ge["stale"] and "stale" in ge["stale"]
    assert all(c["stale"] is None for t, c in by_ticker.items() if t != "GE")
    jpm = by_ticker["JPM"]
    assert jpm["name"] and jpm["fye_month"] == 12
    assert {"file", "form", "fiscal_label", "period_end", "filing_date"} <= set(jpm["filings"][0])
    # Filings the tuning index holds carry the label the index assigned.
    labelled = [f for c in body["companies"] for f in c["filings"] if f["fiscal_label"]]
    assert labelled


def test_dry_run_returns_scope_sources_and_no_model_fields(client):
    res = client.post("/ask", json={"question": Q01, "dry_run": True})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok" and body["dry_run"] is True
    assert body["llm_attempts"] == 0 and body["llm_completed"] == 0
    assert "answer" not in body and "checks" not in body and "prompt" not in body
    assert [c["ticker"] for c in body["plan"]["companies"]] == ["AAPL", "TSLA", "JPM"]
    assert body["context"] and body["budget"]["budget_tokens"] > 0
    assert body["max_retries"] == config.LLM_MAX_RETRIES
    # One source card per context row, with the text the model reads.
    assert [s["cid"] for s in body["sources"]] == [r["cid"] for r in body["context"]]
    first = body["sources"][0]
    assert first["text"] and first["header"] and first["company"]
    assert first["url"].startswith("https://www.sec.gov/")


def test_full_run_returns_the_answer_and_checks(client):
    res = client.post("/ask", json={"question": Q01})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "ok" and body["dry_run"] is False
    assert body["llm_attempts"] == 1 and body["llm_completed"] == 1
    assert body["backend"] == "fake" and body["replayed"] is False
    assert body["answer"]["claims"] and body["answer"]["summary"]
    assert "flags" in body["checks"] and "quotes_located" in body["checks"]
    assert body["prompt"]["system"] and Q01 in body["prompt"]["user"]
    assert body["sources"] and body["budget"]["input_tokens_actual"] is not None
    assert body.get("llm_result") is None


def test_unknown_company_is_refused_before_any_request(client):
    res = client.post("/ask", json={"question": Q_UNKNOWN})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "not_covered"
    assert body["llm_attempts"] == 0 and body["llm_completed"] == 0
    assert "answer" not in body and body["sources"] == []
    assert "Wells Fargo" in body["plan"]["unresolved"]
    assert body["coverage"]


def test_blank_question_is_rejected(client):
    assert client.post("/ask", json={"question": "   ", "dry_run": True}).status_code == 422
    assert client.post("/ask", json={"dry_run": True}).status_code == 422


def test_missing_index_answers_503_with_the_build_command(settings, monkeypatch, tmp_path):
    # Re-enters the shared app's lifespan over an empty directory, which
    # replaces the loaded index on app.state; this test stays last in the
    # file so the module client is never used after it.
    monkeypatch.setattr(config, "INDEX_DIR", str(tmp_path / "no-index"))
    with TestClient(web.app) as c:
        health = c.get("/health").json()
        assert health["chunks"] == 0 and health["index_error"] == web.INDEX_MISSING
        res = c.post("/ask", json={"question": Q01, "dry_run": True})
        assert res.status_code == 503
        assert res.json()["detail"] == "index not built; run: docker compose up indexer"
        assert c.get("/coverage").status_code == 503
        # The page itself still loads, so the browser can show the banner.
        assert c.get("/").status_code == 200
