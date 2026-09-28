"""Dashboard page and read-only API contract coverage."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_operations_dashboard_is_served_without_running_model_calls():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/ui")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Inference control room" in response.text
    assert "get('/health/detailed')" in response.text
    assert "get('/health/providers')" in response.text
    assert "get('/agents')" in response.text
    assert "submitQuery" not in response.text
    assert "refreshProviders" not in response.text
    assert "No provider calls have been recorded" in response.text


@pytest.mark.asyncio
async def test_dashboard_read_endpoints_return_expected_shapes():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        detailed = await client.get("/health/detailed")
        providers = await client.get("/health/providers")
        agents = await client.get("/agents")

    assert detailed.status_code == 200
    assert {"status", "service", "version", "performance", "cache", "concurrency"} <= detailed.json().keys()
    assert providers.status_code == 200
    assert {"providers", "priority_chain"} <= providers.json().keys()
    assert agents.status_code == 200
    assert isinstance(agents.json(), list)
