from __future__ import annotations

import pytest

from app.performance_cache import perf_cache
from app.providers.base import ProviderResponse
from app.providers.gateway import model_gateway
from app.providers.unified_manager import UnifiedExecutionRequest, UnifiedProviderManager


@pytest.fixture(autouse=True)
def clear_response_cache():
    perf_cache.clear()
    yield
    perf_cache.clear()


def _request(**overrides) -> UnifiedExecutionRequest:
    values = {
        "provider": "auto",
        "agent_role": "cache_scope_regression_role",
        "prompt": "Summarize the reporting strategy",
        "context": {},
        "max_tokens": 200,
        "temperature": 0.4,
    }
    values.update(overrides)
    return UnifiedExecutionRequest(**values)


@pytest.fixture
def fake_gateway(monkeypatch):
    calls = []

    async def execute(*, provider_name, request, capability="general", stage_name="general"):
        calls.append({
            "provider": provider_name,
            "content": request.messages[0].content,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "system_instruction": request.system_instruction,
        })
        return ProviderResponse(
            content=(
                f"context={request.messages[0].content};"
                f"max_tokens={request.max_tokens};temperature={request.temperature}"
            ),
            model=request.model or "test-model",
            provider=provider_name,
            latency_seconds=0.01,
            total_tokens=4,
        )

    monkeypatch.setattr(model_gateway, "execute", execute)
    return calls


@pytest.mark.asyncio
async def test_identical_unified_requests_reuse_cache(fake_gateway):
    manager = UnifiedProviderManager()
    request = _request(context={"workspace": "alpha", "policy": "standard"})

    first = await manager.execute(request)
    second = await manager.execute(request)

    assert first.content == second.content
    assert second.status == "cache_hit"
    assert len(fake_gateway) == 1


@pytest.mark.asyncio
async def test_unified_cache_does_not_cross_private_context(fake_gateway):
    manager = UnifiedProviderManager()
    first = await manager.execute(_request(context={"tenant": "alpha", "private_note": "project A"}))
    second = await manager.execute(_request(context={"tenant": "beta", "private_note": "project B"}))

    assert "project A" in first.content
    assert "project B" in second.content
    assert first.content != second.content
    assert second.status == "success"
    assert len(fake_gateway) == 2


@pytest.mark.asyncio
async def test_unified_cache_separates_generation_parameters(fake_gateway):
    manager = UnifiedProviderManager()
    first = await manager.execute(_request(max_tokens=120, temperature=0.1))
    second = await manager.execute(_request(max_tokens=240, temperature=0.9))

    assert "max_tokens=120" in first.content
    assert "max_tokens=240" in second.content
    assert "temperature=0.1" in first.content
    assert "temperature=0.9" in second.content
    assert second.status == "success"
    assert len(fake_gateway) == 2


@pytest.mark.asyncio
async def test_unified_cache_separates_fast_lane_from_standard_execution(fake_gateway):
    manager = UnifiedProviderManager()
    standard = await manager.execute(_request(fast_lane=False))
    fast = await manager.execute(_request(fast_lane=True))

    assert standard.content != fast.content
    assert fast.status == "success"
    assert len(fake_gateway) == 2
