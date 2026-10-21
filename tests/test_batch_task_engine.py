"""Tests for Universal Task Batch Execution Engine."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.providers.unified_manager import UnifiedExecutionResponse


@pytest.fixture
def auth_headers():
    return {
        "X-INFERENCE-API-KEY": "test_inference_api_key_not_a_secret",
        "Content-Type": "application/json",
    }


@pytest.fixture
def mock_unified():
    return UnifiedExecutionResponse(
        provider_used="groq",
        model_used="openai/gpt-oss-120b",
        agent_role="software_architect",
        content="```python\ndef run():\n    return 'success'\n```",
        latency_ms=15.0,
        timestamp=1000.0,
        token_usage={"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
        status="success",
        degraded=False,
    )


@pytest.mark.asyncio
async def test_batch_task_execution_success(auth_headers, mock_unified, monkeypatch):
    monkeypatch.setattr(
        "app.api.universal_task_routes.unified_provider_manager.execute",
        AsyncMock(return_value=mock_unified),
    )

    batch_payload = {
        "source_agent": "forge",
        "concurrency_limit": 3,
        "fail_fast": False,
        "tasks": [
            {
                "task_id": "batch_task_1",
                "action": "code_generation",
                "payload": {"prompt": "Write a retry utility function."},
            },
            {
                "task_id": "batch_task_2",
                "action": "code_generation",
                "payload": {"prompt": "Write a token bucket limiter."},
            },
            {
                "task_id": "batch_task_3",
                "action": "code_generation",
                "payload": {"prompt": "Write an LRU cache decorator."},
            },
        ],
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/v1/task/batch", json=batch_payload, headers=auth_headers)

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_tasks"] == 3
    assert data["succeeded_tasks"] == 3
    assert data["failed_tasks"] == 0
    assert len(data["results"]) == 3
    assert data["summary"]["success_rate"] == 1.0
    for r in data["results"]:
        assert r["status"] == "SUCCESS"
        assert "code" in r["result"]


@pytest.mark.asyncio
async def test_batch_task_error_isolation(auth_headers, mock_unified, monkeypatch):
    calls = 0

    async def _failing_execute(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Upstream provider connection timeout")
        return mock_unified

    monkeypatch.setattr(
        "app.api.universal_task_routes.unified_provider_manager.execute",
        _failing_execute,
    )

    batch_payload = {
        "source_agent": "vyralx",
        "concurrency_limit": 2,
        "fail_fast": False,
        "tasks": [
            {"task_id": "task_a", "payload": {"prompt": "Hook 1"}},
            {"task_id": "task_b", "payload": {"prompt": "Hook 2"}},
            {"task_id": "task_c", "payload": {"prompt": "Hook 3"}},
        ],
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/v1/task/batch", json=batch_payload, headers=auth_headers)

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_tasks"] == 3
    assert data["succeeded_tasks"] == 2
    assert data["failed_tasks"] == 1
    assert data["summary"]["success_rate"] == 0.667


@pytest.mark.asyncio
async def test_batch_task_fail_fast_aborts_subsequent(auth_headers, monkeypatch):
    calls = 0

    async def _fail_immediately(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("Fatal database connection error")

    monkeypatch.setattr(
        "app.api.universal_task_routes.unified_provider_manager.execute",
        _fail_immediately,
    )

    batch_payload = {
        "source_agent": "intelx",
        "concurrency_limit": 1,
        "fail_fast": True,
        "tasks": [
            {"task_id": "f_1", "payload": {"prompt": "Claim 1"}},
            {"task_id": "f_2", "payload": {"prompt": "Claim 2"}},
            {"task_id": "f_3", "payload": {"prompt": "Claim 3"}},
        ],
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/v1/task/batch", json=batch_payload, headers=auth_headers)

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_tasks"] == 3
    assert data["succeeded_tasks"] == 0
    # First failed, remaining aborted
    assert any(r["status"] == "ABORTED" for r in data["results"])
    assert data["summary"]["aborted_count"] >= 1
