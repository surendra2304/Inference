"""Tests for WebSocket Bi-directional Deliberation Gateway and Streaming Endpoints."""

import json
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.providers.base import ProviderResponse
from app.providers.gateway import ModelGateway


@pytest.fixture(autouse=True)
def mock_gateway_stream(monkeypatch):
    """Mocks model_gateway.stream to return predictable tokens for fast deterministic tests."""
    async def _mock_stream(_self, provider, request, stage_name="general_stream"):
        tokens = ["Deliberation ", "consensus ", "achieved ", "cleanly."]
        for t in tokens:
            yield t

    mock_resp = ProviderResponse(content="Deliberation consensus achieved cleanly.", model="openai/gpt-oss-120b", provider="groq")
    monkeypatch.setattr(ModelGateway, "stream", _mock_stream)
    monkeypatch.setattr(ModelGateway, "execute", AsyncMock(return_value=mock_resp))


def test_ws_deliberation_connect_and_handshake(auth):
    client = TestClient(app)
    api_key = auth.get("X-INFERENCE-API-KEY") or auth.get("X-API-Key")

    with client.websocket_connect("/v1/deliberate/ws") as ws:
        # Handshake
        ws.send_text(json.dumps({
            "type": "connect",
            "session_id": "session_test_001",
            "agent_id": "friday",
            "role": "orchestrator",
            "api_key": api_key,
        }))

        res = json.loads(ws.receive_text())
        assert res["type"] == "connected"
        assert res["session_id"] == "session_test_001"
        assert res["agent_id"] == "friday"
        assert res["status"] == "READY"


def test_ws_deliberation_unauthorized_rejected():
    client = TestClient(app)
    with client.websocket_connect("/v1/deliberate/ws") as ws:
        ws.send_text(json.dumps({
            "type": "connect",
            "session_id": "session_bad_auth",
            "agent_id": "rogue_agent",
            "api_key": "invalid_untrusted_key_12345",
        }))
        res = json.loads(ws.receive_text())
        assert res["type"] == "error"
        assert res["error"] == "UNAUTHORIZED"


def test_ws_deliberation_turn_streaming(auth):
    client = TestClient(app)
    api_key = auth.get("X-INFERENCE-API-KEY") or auth.get("X-API-Key")

    with client.websocket_connect("/v1/deliberate/ws") as ws:
        # Connect
        ws.send_text(json.dumps({
            "type": "connect",
            "session_id": "session_delib_turn",
            "agent_id": "stratex",
            "role": "strategy_engine",
            "api_key": api_key,
        }))
        assert json.loads(ws.receive_text())["type"] == "connected"

        # Submit turn
        ws.send_text(json.dumps({
            "type": "deliberate_turn",
            "turn_id": "turn_1",
            "prompt": "Evaluate market regime transition risks.",
            "max_tokens": 100,
        }))

        # Read chunks until turn_complete
        chunks = []
        turn_complete = None
        for _ in range(20):
            msg = json.loads(ws.receive_text())
            if msg["type"] == "chunk":
                chunks.append(msg["token"])
            elif msg["type"] == "turn_complete":
                turn_complete = msg
                break

        assert len(chunks) > 0
        assert turn_complete is not None
        assert turn_complete["turn_id"] == "turn_1"
        assert turn_complete["sender"] == "stratex"
        assert "Deliberation consensus" in turn_complete["synthesis"]


def test_ws_deliberation_consensus_request(auth):
    client = TestClient(app)
    api_key = auth.get("X-INFERENCE-API-KEY") or auth.get("X-API-Key")

    with client.websocket_connect("/v1/deliberate/ws") as ws:
        ws.send_text(json.dumps({
            "type": "connect",
            "session_id": "session_consensus",
            "agent_id": "cortex",
            "api_key": api_key,
        }))
        assert json.loads(ws.receive_text())["type"] == "connected"

        # Request consensus
        ws.send_text(json.dumps({
            "type": "consensus_request",
        }))

        verdict_msg = json.loads(ws.receive_text())
        assert verdict_msg["type"] == "consensus_verdict"
        assert verdict_msg["session_id"] == "session_consensus"
        assert "convergence_score" in verdict_msg
        assert verdict_msg["convergence_score"] > 0.0


def test_ws_deliberation_ping_pong(auth):
    client = TestClient(app)
    api_key = auth.get("X-INFERENCE-API-KEY") or auth.get("X-API-Key")

    with client.websocket_connect("/v1/deliberate/ws") as ws:
        ws.send_text(json.dumps({
            "type": "connect",
            "session_id": "session_ping",
            "agent_id": "friday",
            "api_key": api_key,
        }))
        assert json.loads(ws.receive_text())["type"] == "connected"

        ws.send_text(json.dumps({"type": "ping"}))
        pong = json.loads(ws.receive_text())
        assert pong["type"] == "pong"
        assert "timestamp" in pong


def test_ws_direct_stream_endpoint(auth):
    client = TestClient(app)
    api_key = auth.get("X-INFERENCE-API-KEY") or auth.get("X-API-Key")

    with client.websocket_connect("/v1/stream") as ws:
        ws.send_text(json.dumps({
            "prompt": "Stream this test response",
            "api_key": api_key,
            "provider": "groq",
            "max_tokens": 50,
        }))

        tokens = []
        is_done = False
        for _ in range(10):
            msg = json.loads(ws.receive_text())
            if msg["type"] == "token":
                tokens.append(msg["token"])
            elif msg["type"] == "done":
                is_done = True
                break

        assert len(tokens) > 0
        assert is_done is True


def test_http_v1_stream_post(auth):
    client = TestClient(app)
    headers = {**auth, "Content-Type": "application/json"}

    response = client.post(
        "/v1/stream",
        json={"prompt": "Stream hello", "provider": "groq", "max_tokens": 50},
        headers=headers,
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    lines = [line for line in response.text.split("\n") if line.startswith("data: ")]
    assert len(lines) >= 2
    events = [json.loads(line[len("data: "):]) for line in lines]
    assert any("token" in e for e in events)
    assert events[-1].get("done") is True
