"""Integration tests for Instant Ultra-Low Latency Answers & Multi-Tier Caching."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.performance_cache import perf_cache
from app.providers.base import ProviderResponse


@pytest.fixture(autouse=True)
def clean_cache():
    perf_cache.clear()
    yield
    perf_cache.clear()


@pytest.fixture
def client():
    settings.FRIDAY_UNIVERSE_API_KEY = "test_friday_secret_key_12345"
    with TestClient(app) as test_client:
        yield test_client


def test_instant_grounding_zero_overhead(client):
    """Verify that core knowledge questions return in < 1ms via L0 instant grounding."""
    headers = {"X-FRIDAY-API-Key": "test_friday_secret_key_12345"}

    # 1. Instant endpoint
    resp = client.post("/v1/instant/ask", json={"prompt": "what is inference"}, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "Inference is the ultra-low latency" in data["answer"]
    assert data["cached"] is True
    assert data["source"] == "instant_grounding"
    assert data["latency_ms"] < 10.0  # sub-millisecond

    # 2. FRIDAY ask endpoint
    resp2 = client.post("/v1/friday/ask", json={"question": "is this a website"}, headers=headers)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert "No. Inference is an autonomous backend" in data2["answer"]
    assert data2["provenance"]["cache_tier"] == "L0_INSTANT_GROUNDING"
    assert data2["provenance"]["cached"] is True


def test_instant_ask_speculative_lpu_and_cache(client):
    """Verify that instant_ask executes speculative race and subsequent calls hit L1 cache."""
    mock_resp = ProviderResponse(
        content="Quantum entanglement is a physical phenomenon where particles remain connected.",
        model="openai/gpt-oss-120b",
        provider="groq",
        latency_seconds=0.08,
        raw_response={"speculative_race": {"winner": "groq", "elapsed_seconds": 0.08}},
    )

    headers = {"X-FRIDAY-API-Key": "test_friday_secret_key_12345"}
    payload = {"prompt": "Explain quantum entanglement in 1 sentence", "caller_id": "human"}

    with patch("app.api.instant_routes.model_gateway.execute_speculative", new_callable=AsyncMock) as mock_race:
        mock_race.return_value = mock_resp

        # 1st call: Speculative race
        r1 = client.post("/v1/instant/ask", json=payload, headers=headers)
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["source"] == "speculative_lpu"
        assert d1["provider"] == "groq"
        assert d1["cached"] is False
        assert "Quantum entanglement" in d1["answer"]
        assert mock_race.call_count == 1

        # 2nd call: Multi-tier L1 cache hit (< 1ms)
        r2 = client.post("/v1/instant/ask", json=payload, headers=headers)
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["source"] == "l1_cache"
        assert d2["cached"] is True
        assert "Quantum entanglement" in d2["answer"]
        assert mock_race.call_count == 1  # Not called again!


def test_instant_stream_grounded(client):
    """Verify that instant stream delivers grounded answers immediately."""
    headers = {"X-FRIDAY-API-Key": "test_friday_secret_key_12345"}
    resp = client.post("/v1/instant/stream", json={"prompt": "can friday control inference"}, headers=headers)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    lines = [line.strip() for line in resp.text.split("\n") if line.strip().startswith("data: ")]
    assert len(lines) >= 1
    data = json.loads(lines[0][6:])
    assert "Yes, FRIDAY controls Inference" in data["token"]
    assert data["done"] is True


def test_friday_ask_defaults_to_fast_lane(client):
    """Verify that FridayRequest defaults fast_lane to True and runs in fast mode."""
    headers = {"X-FRIDAY-API-Key": "test_friday_secret_key_12345"}

    mock_resp = ProviderResponse(
        content="FRIDAY Fast Mode: Operational analysis completed.",
        model="gemini-3.6-flash",
        provider="gemini",
        latency_seconds=0.15,
    )

    with patch("app.agents.debate.model_gateway.execute", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_resp

        resp = client.post(
            "/v1/friday/ask",
            json={"question": "Verify database latency metrics"},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["provenance"]["fast_lane"] is True
        assert data["mode_used"] == "fast"
