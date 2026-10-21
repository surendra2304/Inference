from __future__ import annotations

import asyncio
import math
import time

import pytest

import app.providers
from app.core.config import settings
from app.providers.base import ProviderMessage, ProviderRequest
from app.providers.gateway import KeyPool, ModelGateway, ProviderRateLimiter


@pytest.mark.parametrize(
    "timeout",
    [0, -1, math.nan, math.inf, -math.inf, True, "not-a-number"],
)
@pytest.mark.asyncio
async def test_execute_rejects_invalid_request_deadlines_before_provider_work(timeout):
    gateway = ModelGateway()
    request = ProviderRequest(
        messages=[ProviderMessage(role="user", content="deadline validation")],
        extra_params={"timeout": timeout},
    )

    with pytest.raises(ValueError, match="request timeout must be a finite positive number"):
        await gateway.execute("groq", request)


@pytest.mark.asyncio
async def test_execute_deadline_bounds_provider_limiter_wait(monkeypatch):
    monkeypatch.setattr(settings, "PRIMARY_ATTEMPT_FRACTION", 0.5)
    monkeypatch.setattr(settings, "LOCAL_PREFERRED", False)
    monkeypatch.setattr(settings, "LOCAL_ENABLED", False)
    monkeypatch.setattr(
        "app.providers.gateway.provider_has_credentials",
        lambda _provider: True,
    )

    gateway = ModelGateway()
    gateway.key_pools["test-provider"] = KeyPool("test-provider", ["test-key"])
    limiter = ProviderRateLimiter(
        "test-provider",
        requests_per_second=0.001,
        max_concurrency=1,
    )
    # Simulate a limiter whose token bucket cannot refill within this request's
    # primary attempt slice. The completion path must time out *while acquiring* it.
    limiter.tokens = 0.0
    limiter.updated = time.monotonic()
    gateway.rate_limiters["test-provider"] = limiter

    provider_calls = 0
    fallback_called = asyncio.Event()

    class _Provider:
        async def generate(self, _request):
            nonlocal provider_calls
            provider_calls += 1
            raise AssertionError("provider call must not start without a limiter slot")

    monkeypatch.setattr(app.providers, "get_provider", lambda *_args, **_kwargs: _Provider())

    async def fallback(**_kwargs):
        fallback_called.set()
        raise RuntimeError("fallback reached after primary deadline")

    monkeypatch.setattr(gateway, "_execute_dynamic_fallback", fallback)
    request = ProviderRequest(
        messages=[ProviderMessage(role="user", content="deadline test")],
        extra_params={"timeout": 0.1},
    )

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="fallback reached after primary deadline"):
        await gateway.execute("test-provider", request)
    elapsed = time.monotonic() - started

    assert fallback_called.is_set()
    assert provider_calls == 0
    assert elapsed < 0.5, f"request exceeded its 0.1s deadline by waiting {elapsed:.3f}s"
    # The timed-out acquire returns its concurrency slot rather than leaking it.
    assert limiter._sem._value == 1


@pytest.mark.asyncio
async def test_execute_uses_remaining_deadline_after_limiter_acquisition(monkeypatch):
    monkeypatch.setattr(settings, "PRIMARY_ATTEMPT_FRACTION", 0.5)
    monkeypatch.setattr(settings, "LOCAL_PREFERRED", False)
    monkeypatch.setattr(settings, "LOCAL_ENABLED", False)
    monkeypatch.setattr(
        "app.providers.gateway.provider_has_credentials",
        lambda _provider: True,
    )

    gateway = ModelGateway()
    gateway.key_pools["test-provider"] = KeyPool("test-provider", ["test-key"])
    limiter = ProviderRateLimiter("test-provider", requests_per_second=100.0, max_concurrency=1)
    gateway.rate_limiters["test-provider"] = limiter

    call_started = asyncio.Event()
    provider_cancelled = asyncio.Event()

    class _SlowProvider:
        async def generate(self, _request):
            call_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                provider_cancelled.set()
                raise

    monkeypatch.setattr(app.providers, "get_provider", lambda *_args, **_kwargs: _SlowProvider())

    async def fallback(**_kwargs):
        raise RuntimeError("fallback after timed-out provider call")

    monkeypatch.setattr(gateway, "_execute_dynamic_fallback", fallback)
    request = ProviderRequest(
        messages=[ProviderMessage(role="user", content="deadline test")],
        extra_params={"timeout": 0.1},
    )

    with pytest.raises(RuntimeError, match="fallback after timed-out provider call"):
        await gateway.execute("test-provider", request)

    assert call_started.is_set()
    assert provider_cancelled.is_set()
    assert limiter._sem._value == 1
