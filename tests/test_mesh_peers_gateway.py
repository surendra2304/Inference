"""Tests for FRIDAY Universe Peer Mesh Discovery, Status, and Probe Gateway."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def auth_headers():
    return {
        "X-INFERENCE-API-KEY": "test_inference_api_key_not_a_secret",
        "Content-Type": "application/json",
    }


@pytest.mark.asyncio
async def test_mesh_peers_endpoint_requires_auth():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/v1/mesh/peers")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_mesh_peers_returns_all_9_agents(auth_headers):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/v1/mesh/peers", headers=auth_headers)

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_peers"] == 9
    peers = {p["agent_id"]: p for p in data["peers"]}

    expected_agents = ["friday", "memora", "forge", "cortex", "sentinel", "stratex", "intelx", "futuris", "vyralx"]
    for agent_id in expected_agents:
        assert agent_id in peers
        peer = peers[agent_id]
        assert "url" in peer
        assert "role" in peer
        assert "authenticated" in peer
        assert "quota" in peer
        assert "primary_model" in peer
        assert "fallback_model" in peer

    # Forge coding model check
    assert peers["forge"]["primary_model"] == "qwen/qwen-2.5-coder-32b-instruct"
    # Cortex reasoning model check
    assert peers["cortex"]["primary_model"] == "deepseek/deepseek-r1"


@pytest.mark.asyncio
async def test_mesh_status_telemetry(auth_headers):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/v1/mesh/status", headers=auth_headers)

    assert resp.status_code == 200
    data = resp.json()
    assert data["mesh_network"] == "friday_mesh"
    assert data["node_id"] == "inference"
    assert data["agent_number"] == "2"
    assert data["status"] == "OPTIMAL"
    assert data["active_reasoning_model"] == "deepseek/deepseek-r1"
    assert data["active_coding_model"] == "qwen/qwen-2.5-coder-32b-instruct"
    assert "provider_pools" in data
    assert "persistence_authority" in data
    assert "PostgreSQL 18" in data["persistence_authority"]["primary"]


@pytest.mark.asyncio
async def test_mesh_ping_unknown_agent_returns_404(auth_headers):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/v1/mesh/ping/unknown_rogue_bot", headers=auth_headers)

    assert resp.status_code == 404
    assert "not a recognized" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_mesh_ping_known_agent_mocked_success(auth_headers):
    transport = ASGITransport(app=app)
    mock_resp = AsyncMock()
    mock_resp.status_code = 200

    mock_http_client = AsyncMock()
    mock_http_client.get = AsyncMock(return_value=mock_resp)

    with patch("app.api.mesh_routes.get_shared_client", new=AsyncMock(return_value=mock_http_client)):
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post("/v1/mesh/ping/forge", headers=auth_headers)

    assert resp.status_code == 200
    data = resp.json()
    assert data["agent_id"] == "forge"
    assert data["reachable"] is True
    assert data["status_code"] == 200
    assert "latency_ms" in data
