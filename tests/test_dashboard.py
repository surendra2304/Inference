"""Dashboard page and read-only API contract coverage."""

from datetime import datetime

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
    assert 'id="health-evidence"' in response.text
    assert "d.evidence_class" in response.text
    assert "d.evidence_scope" in response.text
    assert "health.value.observed_at" in response.text
    assert "new Date().toLocaleTimeString()" not in response.text
    assert "p.evidence_class" in response.text
    assert "p.observed_at" in response.text


@pytest.mark.asyncio
async def test_dashboard_read_endpoints_return_expected_shapes():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        basic = await client.get("/health")
        ready = await client.get("/health/ready")
        detailed = await client.get("/health/detailed")
        providers = await client.get("/health/providers")
        agents = await client.get("/agents")

    assert basic.status_code == 200
    basic_payload = basic.json()
    assert basic_payload["status"] == "responding"
    assert basic_payload["evidence_class"] == "process_liveness"
    assert datetime.fromisoformat(basic_payload["observed_at"])
    assert ready.status_code == 200
    ready_payload = ready.json()
    assert ready_payload["evidence_class"] == "agent_registry_configuration"
    assert "provider, memory, and database readiness are not checked" in ready_payload["evidence_scope"]
    assert datetime.fromisoformat(ready_payload["observed_at"])
    assert detailed.status_code == 200
    detailed_payload = detailed.json()
    assert {"status", "evidence_class", "evidence_scope", "observed_at", "service", "version", "performance", "cache", "concurrency"} <= detailed_payload.keys()
    assert detailed_payload["status"] == "responding"
    assert detailed_payload["evidence_class"] == "process_runtime_snapshot"
    assert datetime.fromisoformat(detailed_payload["observed_at"])
    assert providers.status_code == 200
    providers_payload = providers.json()
    assert {"providers", "priority_chain", "evidence_class", "evidence_scope", "observed_at"} <= providers_payload.keys()
    assert providers_payload["evidence_class"] == "process_observed_provider_calls"
    assert datetime.fromisoformat(providers_payload["observed_at"])
    assert agents.status_code == 200
    assert isinstance(agents.json(), list)
