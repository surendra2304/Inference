import asyncio
import time

import pytest

import app.providers
from app.providers.base import ProviderMessage, ProviderRequest, ProviderResponse
from app.providers.errors import GatewayError, RateLimitError
from app.providers.gateway import KeyPool, ModelGateway


class _Limiter:
    async def acquire(self):
        return None

    def release(self):
        return None


class _Health:
    def __init__(self):
        self.successes = []
        self.failures = []

    def record_success(self, provider, latency):
        self.successes.append((provider, latency))

    def record_failure(self, provider, error, **kwargs):
        self.failures.append((provider, error, kwargs))


class _OpenRouter:
    def __init__(self, key):
        self.key = key

    async def get_best_free_model(self, _capability):
        return "test/free-model"

    async def generate(self, request):
        return ProviderResponse(
            content="ok", model=request.model, provider="openrouter", raw_response={}
        )


def _gateway(keys):
    gateway = ModelGateway.__new__(ModelGateway)
    gateway.key_pools = {"openrouter": KeyPool("openrouter", keys)}
    gateway.rate_limiters = {"openrouter": _Limiter()}
    gateway.health_tracker = _Health()
    return gateway


def _request():
    return ProviderRequest(messages=[ProviderMessage(role="user", content="hello")], model="model")


def test_openrouter_fallback_uses_one_rotated_key(monkeypatch):
    gateway = _gateway(["first-test-key", "second-test-key"])
    constructed = []

    def get_provider(name, **kwargs):
        constructed.append((name, kwargs.get("api_key")))
        return _OpenRouter(kwargs.get("api_key"))

    monkeypatch.setattr(app.providers, "get_provider", get_provider)
    response = asyncio.run(
        gateway._execute_dynamic_fallback(
            failed_provider="gemini",
            request=_request(),
            capability="general",
            stage_name="test",
            last_error=GatewayError("primary failed", provider="gemini"),
            deadline=time.monotonic() + 10,
        )
    )

    assert constructed == [("openrouter", "first-test-key")]
    assert response.provider == "openrouter"
    assert gateway.health_tracker.successes[0][0] == "openrouter"


def test_openrouter_fallback_does_not_bypass_all_key_quarantines(monkeypatch):
    gateway = _gateway(["first-test-key", "second-test-key"])
    pool = gateway.key_pools["openrouter"]
    pool.quarantine("first-test-key", duration_seconds=60)
    pool.quarantine("second-test-key", duration_seconds=60)
    provider_calls = []

    monkeypatch.setattr(
        app.providers,
        "get_provider",
        lambda *args, **kwargs: provider_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "app.providers.gateway.ProviderSwitchingPolicy.get_fallback_provider",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("app.providers.gateway.settings.INFERENCE_LITELLM_ENABLED", False)
    monkeypatch.setattr(
        "app.providers.gateway.settings.INFERENCE_LITELLM_FALLBACK_ENABLED", False
    )
    primary_error = GatewayError("primary failed", provider="gemini")

    with pytest.raises(GatewayError, match="primary failed"):
        asyncio.run(
            gateway._execute_dynamic_fallback(
                failed_provider="gemini",
                request=_request(),
                capability="general",
                stage_name="test",
                last_error=primary_error,
                deadline=time.monotonic() + 10,
            )
        )

    assert provider_calls == []
    assert pool.get_active_keys_count() == 0


def test_openrouter_fallback_quarantines_only_the_key_that_was_rate_limited(monkeypatch):
    gateway = _gateway(["first-test-key", "second-test-key"])
    pool = gateway.key_pools["openrouter"]
    constructed = []

    class _RateLimitedOpenRouter(_OpenRouter):
        async def generate(self, _request):
            raise RateLimitError("429 rate limit", provider="openrouter")

    def get_provider(name, **kwargs):
        constructed.append((name, kwargs.get("api_key")))
        return _RateLimitedOpenRouter(kwargs.get("api_key"))

    monkeypatch.setattr(app.providers, "get_provider", get_provider)
    monkeypatch.setattr(
        "app.providers.gateway.ProviderSwitchingPolicy.get_fallback_provider",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("app.providers.gateway.settings.INFERENCE_LITELLM_ENABLED", False)
    monkeypatch.setattr(
        "app.providers.gateway.settings.INFERENCE_LITELLM_FALLBACK_ENABLED", False
    )

    with pytest.raises(GatewayError, match="primary failed"):
        asyncio.run(
            gateway._execute_dynamic_fallback(
                failed_provider="gemini",
                request=_request(),
                capability="general",
                stage_name="test",
                last_error=GatewayError("primary failed", provider="gemini"),
                deadline=time.monotonic() + 10,
            )
        )

    assert constructed == [("openrouter", "first-test-key")]
    assert pool.get_quarantined_keys_count() == 1
    assert pool.get_active_keys_count() == 1
