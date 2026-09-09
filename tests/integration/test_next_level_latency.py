"""Integration tests for Next-Level Ultra-Low Latency Ecosystem Upgrades:
- Universal L1 Cache in UnifiedProviderManager
- ModelGateway Speculative Fastest-Provider Racing
- Inter-Agent Fast Gateway (/v1/agent/assist & /v1/agent/stream & /v1/agent/stats)
- Real-Time SSE Code Streaming for FORGE (/v1/forge/stream-code)
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.performance_cache import perf_cache
from app.providers.base import ProviderMessage, ProviderRequest, ProviderResponse
from app.providers.gateway import ModelGateway
from app.providers.unified_manager import (
    UnifiedExecutionRequest,
    unified_provider_manager,
)


@pytest.fixture(autouse=True)
def clean_cache():
    perf_cache.clear()
    yield
    perf_cache.clear()


@pytest.fixture
def test_client():
    settings.FRIDAY_UNIVERSE_API_KEY = "test_friday_secret_key_12345"
    with TestClient(app) as client:
        yield client


@pytest.mark.asyncio
async def test_unified_provider_l1_cache():
    """Verify that UnifiedProviderManager caches responses and returns in < 1ms on hit."""
    prompt = "Design a low-latency cache system"
    req = UnifiedExecutionRequest(
        provider="gemini",
        agent_role="system_architect",
        prompt=prompt,
        max_tokens=100,
    )

    mock_resp = ProviderResponse(
        content="Cache architecture design specification.",
        model="gemini-3.6-flash",
        provider="gemini",
        total_tokens=50,
        latency_seconds=0.3,
    )

    with patch("app.providers.unified_manager.get_provider") as mock_get_prov:
        mock_prov_instance = AsyncMock()
        mock_prov_instance.generate.return_value = mock_resp
        mock_get_prov.return_value = mock_prov_instance

        # 1st execution: Miss -> calls provider
        res1 = await unified_provider_manager.execute(req)
        assert res1.status == "success"
        assert res1.content == "Cache architecture design specification."
        assert mock_prov_instance.generate.call_count == 1

        # 2nd execution: Hit -> sub-millisecond cache hit
        res2 = await unified_provider_manager.execute(req)
        assert res2.status == "cache_hit"
        assert res2.content == "Cache architecture design specification."
        assert res2.latency_ms < 10.0  # Typically < 0.05ms
        assert mock_prov_instance.generate.call_count == 1  # Not called again

        # 3rd execution with no_cache=True -> Bypasses cache
        req_nocache = UnifiedExecutionRequest(
            provider="gemini",
            agent_role="system_architect",
            prompt=prompt,
            max_tokens=100,
            no_cache=True,
        )
        res3 = await unified_provider_manager.execute(req_nocache)
        assert res3.status == "success"
        assert mock_prov_instance.generate.call_count == 2


@pytest.mark.asyncio
async def test_model_gateway_speculative_racing():
    """Verify that ModelGateway speculative racing launches parallel calls and picks winner."""
    gw = ModelGateway()

    req = ProviderRequest(
        messages=[ProviderMessage(role="user", content="Hello")],
        model="auto",
    )

    resp_groq = ProviderResponse(
        content="Groq won ultra fast",
        model="openai/gpt-oss-120b",
        provider="groq",
        latency_seconds=0.08,
    )

    async def mock_execute(prov, *args, **kwargs):
        if prov == "groq":
            return resp_groq
        # simulate slower gemini
        import asyncio
        await asyncio.sleep(0.5)
        return ProviderResponse(
            content="Gemini slower response",
            model="gemini-3.6-flash",
            provider="gemini",
            latency_seconds=0.5,
        )

    with patch.object(gw, "execute", side_effect=mock_execute):
        winner = await gw.execute_speculative(["groq", "gemini"], req)
        assert winner.content == "Groq won ultra fast"
        assert winner.raw_response is not None
        assert winner.raw_response["speculative_race"]["winner"] == "groq"


def test_agent_assist_endpoint_and_cache(test_client):
    """Test POST /v1/agent/assist with caller identification and caching."""
    mock_resp = ProviderResponse(
        content="Code generated for FORGE: def solve(): pass",
        model="openai/gpt-oss-120b",
        provider="groq",
        total_tokens=60,
        latency_seconds=0.15,
    )

    headers = {"X-FRIDAY-API-Key": "test_friday_secret_key_12345"}
    payload = {
        "caller_agent": "forge",
        "task_type": "code",
        "prompt": "Implement fast sort algorithm in python",
        "fast_lane": True,
    }

    with patch("app.providers.unified_manager.get_provider") as mock_get_prov:
        mock_prov_instance = AsyncMock()
        mock_prov_instance.generate.return_value = mock_resp
        mock_get_prov.return_value = mock_prov_instance

        # 1st call
        res1 = test_client.post("/v1/agent/assist", json=payload, headers=headers)
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["caller_agent"] == "forge"
        assert data1["agent_role"] == "code_generator"
        assert data1["cache_hit"] is False
        assert "Code generated for FORGE" in data1["response"]

        # 2nd call: cache hit
        res2 = test_client.post("/v1/agent/assist", json=payload, headers=headers)
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["cache_hit"] is True
        assert data2["status"] == "cache_hit"


def test_agent_stats_endpoint(test_client):
    """Test GET /v1/agent/stats returning telemetry and cache stats."""
    headers = {"X-FRIDAY-API-Key": "test_friday_secret_key_12345"}
    res = test_client.get("/v1/agent/stats", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert "agents" in data
    assert "cache" in data
    assert "timestamp" in data


def test_forge_stream_code_endpoint(test_client):
    """Test POST /v1/forge/stream-code real-time SSE streaming."""
    async def mock_chunks(*args, **kwargs):
        yield "def "
        yield "calculate_latency(): "
        yield "return 0.1"

    headers = {"X-FRIDAY-API-Key": "test_friday_secret_key_12345"}
    payload = {
        "filename": "metrics.py",
        "file_type": "python",
        "requirements": ["calculate latency"],
    }

    with patch("app.services.code_generation.model_gateway.stream", side_effect=mock_chunks):
        res = test_client.post("/v1/forge/stream-code", json=payload, headers=headers)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]

        lines = [line.strip() for line in res.text.split("\n") if line.strip()]
        chunks_received = []
        for line in lines:
            if line.startswith("data: "):
                data = json.loads(line[6:])
                chunks_received.append(data.get("chunk", ""))

        joined_code = "".join(chunks_received)
        assert "def calculate_latency(): return 0.1" in joined_code


def test_agent_stream_endpoint(test_client):
    """Test POST /v1/agent/stream universal SSE streaming."""
    async def mock_agent_chunks(*args, **kwargs):
        yield "Strategic "
        yield "Analysis "
        yield "Complete"

    headers = {"X-FRIDAY-API-Key": "test_friday_secret_key_12345"}
    payload = {
        "caller_agent": "sentinel",
        "task_type": "security",
        "prompt": "Analyze boundary attack vector",
    }

    with patch("app.providers.gateway.model_gateway.stream", side_effect=mock_agent_chunks):
        res = test_client.post("/v1/agent/stream", json=payload, headers=headers)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]

        lines = [line.strip() for line in res.text.split("\n") if line.strip()]
        tokens = []
        for line in lines:
            if line.startswith("data: "):
                data = json.loads(line[6:])
                tokens.append(data.get("token", ""))

        joined = "".join(tokens)
        assert "Strategic Analysis Complete" in joined
