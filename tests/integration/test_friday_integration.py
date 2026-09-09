"""Integration tests for FRIDAY Integration: FRIDAY Peer Integration and Security Boundary."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.orchestrator import orchestrator
from app.main import app
from app.memory.sqlite import SQLiteMemory
from app.providers.base import ProviderResponse


@pytest.fixture
def friday_client(tmp_path):
    test_db = str(tmp_path / "test_friday_integration.db")
    memory = SQLiteMemory(db_path=test_db)
    orchestrator.memory = memory
    settings.FRIDAY_UNIVERSE_API_KEY = "test_friday_secret_key_12345"

    with TestClient(app) as client:
        yield client


@pytest.mark.asyncio
async def test_friday_ask_authenticated(friday_client):
    client = friday_client

    mock_llm_response = ProviderResponse(
        content="FRIDAY consultation: Recommended architecture strategy validated.",
        model="gemini-2.5-flash",
        provider="gemini",
        total_tokens=45,
        latency_seconds=0.35
    )

    with patch("app.agents.debate.model_gateway.execute", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_llm_response

        headers = {"X-FRIDAY-API-Key": "test_friday_secret_key_12345"}
        resp = client.post(
            "/v1/friday/ask",
            headers=headers,
            json={
                "question": "Assess memory isolation risks for FRIDAY subagents.",
                "caller_id": "friday_executive",
                "max_latency": 10.0
            }
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"].startswith("task_")
        assert "FRIDAY consultation" in data["answer"]
        assert data["confidence"] > 0.8
        assert data["provenance"]["caller_id"] == "friday_executive"
        assert data["provenance"]["platform"] == "Inference"


@pytest.mark.asyncio
async def test_friday_debate_provenance_and_dissent(friday_client):
    client = friday_client

    mock_llm_response = ProviderResponse(
        content="FRIDAY debate consensus: Enforce cryptographic capabilities across process boundaries.",
        model="gemini-2.5-pro",
        provider="gemini",
        total_tokens=95,
        latency_seconds=0.60
    )

    with patch("app.agents.debate.model_gateway.execute", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_llm_response

        headers = {"X-FRIDAY-API-Key": "test_friday_secret_key_12345"}
        resp = client.post(
            "/v1/friday/debate",
            headers=headers,
            json={
                "question": "Should FRIDAY deploy untrusted plugins inside separate sandbox workers?",
                "caller_id": "friday_security_core"
            }
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["mode_used"] in ["debate", "consensus", "collaboration"]
        assert data["confidence"] >= 0.85
        assert isinstance(data["unresolved_disagreements"], list)
        assert len(data["key_evidence"]) > 0
        assert data["provenance"]["rounds_completed"] in [2, 6]
        assert data["provenance"]["caller_id"] == "friday_security_core"


@pytest.mark.asyncio
async def test_friday_authentication_failure_missing_header(friday_client):
    client = friday_client
    # No header provided
    resp = client.post("/v1/friday/ask", json={"question": "Test unauthorized inquiry"})
    assert resp.status_code == 401
    assert "Missing 'X-Inference-API-KEY' or 'X-FRIDAY-API-Key'" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_friday_authentication_failure_invalid_key(friday_client):
    client = friday_client
    headers = {"X-FRIDAY-API-Key": "invalid_wrong_secret_key"}
    resp = client.post("/v1/friday/ask", headers=headers, json={"question": "Test forbidden inquiry"})
    assert resp.status_code == 403
    assert "Forbidden: Invalid API Key provided." in resp.json()["detail"]



@pytest.mark.asyncio
async def test_friday_status_endpoint(friday_client):
    client = friday_client
    headers = {"X-FRIDAY-API-Key": "test_friday_secret_key_12345"}

    # Test authenticated request
    resp = client.get("/v1/friday/status", headers=headers)
    assert resp.status_code == 200
    data = resp.json()

    # Validate schema fields
    assert "active_agents" in data
    assert "configured_providers" in data
    assert "available_models" in data

    # Validate active agent roles list
    assert isinstance(data["active_agents"], list)
    assert len(data["active_agents"]) >= 10
    assert "Architect" in data["active_agents"]
    assert "Coder" in data["active_agents"]
    assert "Critic" in data["active_agents"]
    assert "Trading Analyst" in data["active_agents"]

    # Validate configured providers & models
    assert isinstance(data["configured_providers"], list)
    assert isinstance(data["available_models"], list)
    assert len(data["available_models"]) > 0

    # Test unauthorized request (missing header)
    unauth_resp = client.get("/v1/friday/status")
    assert unauth_resp.status_code == 401


@pytest.mark.asyncio
async def test_friday_ask_l1_cache_hit_and_bypass(friday_client):
    client = friday_client
    headers = {"X-FRIDAY-API-Key": "test_friday_secret_key_12345"}
    mock_llm_response = ProviderResponse(
        content="Ultra-fast cached answer.",
        model="openai/gpt-oss-120b",
        provider="groq",
        total_tokens=20,
        latency_seconds=0.25,
    )

    with patch("app.agents.debate.model_gateway.execute", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_llm_response

        # 1. First invocation (cache miss)
        q = "Unique latency probe question 98765"
        resp1 = client.post(
            "/v1/friday/ask",
            headers=headers,
            json={"question": q, "caller_id": "test_caller"},
        )
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1["provenance"]["cached"] is False

        # 2. Second invocation (cache hit -> sub-millisecond)
        resp2 = client.post(
            "/v1/friday/ask",
            headers=headers,
            json={"question": q, "caller_id": "test_caller"},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["provenance"]["cached"] is True
        assert data2["latency_seconds"] <= 0.002
        assert data2["answer"] == data1["answer"]

        # 3. Third invocation with no_cache=True (bypasses cache)
        resp3 = client.post(
            "/v1/friday/ask",
            headers=headers,
            json={"question": q, "caller_id": "test_caller", "no_cache": True},
        )
        assert resp3.status_code == 200
        data3 = resp3.json()
        assert data3["provenance"]["cached"] is False


@pytest.mark.asyncio
async def test_friday_streaming_sse(friday_client):
    client = friday_client
    headers = {"X-FRIDAY-API-Key": "test_friday_secret_key_12345"}

    async def mock_stream_gen(provider, request, stage_name="general_stream"):
        for token in ["Hello", " ", "FRIDAY", " ", "streaming!"]:
            yield token

    with patch("app.api.friday_routes.model_gateway.stream", side_effect=mock_stream_gen):
        resp = client.post(
            "/v1/friday/stream",
            headers=headers,
            json={"question": "Stream me a greeting", "caller_id": "test_stream"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.text
        assert "data: " in body
        assert "FRIDAY" in body
        assert '"done": true' in body
