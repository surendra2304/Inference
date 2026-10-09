"""A configured self-hosted model must answer when no cloud provider is configured.

Found by driving the agent against the local rig (``scripts/local_model_server.py``) with
``LOCAL_ENABLED=true`` and no cloud keys. [FACT] measured before the fix: the rig answered
``GET /v1/models``, the agent was configured to use it, and every ``POST /v1/ask`` came back
``DEGRADED`` with ``Provider 'groq' has no configured credential``. The local tier was never
consulted.

Cause: ``ModelGateway.execute`` raised ``ProviderUnconfiguredError`` from its pre-flight
credential gate, which sits *before* the fallback ladder. The ladder is the only place the
self-hosted tier is tried as the terminal rung, so an unconfigured primary made the local model
unreachable.

Each test below is falsified against the pre-fix code:

* ``test_local_tier_answers_when_the_primary_has_no_credential`` raised instead of answering.
* ``test_no_local_tier_still_raises_the_same_unconfigured_error`` pins the unchanged contract:
  with nothing able to answer, the caller still receives ``ProviderUnconfiguredError``.
"""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.providers import gateway as gateway_module
from app.providers.base import ProviderMessage, ProviderRequest, ProviderResponse
from app.providers.errors import ProviderUnconfiguredError


class FakeLocalProvider:
    """Stands in for the OpenAI-compatible local server; records what it was asked."""

    def __init__(self) -> None:
        self.calls: list[ProviderRequest] = []

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.calls.append(request)
        return ProviderResponse(
            content="served by the self-hosted tier",
            model=request.model or "rig-8b",
            provider="local",
            prompt_tokens=3,
            completion_tokens=5,
            total_tokens=8,
            latency_seconds=0.01,
        )


@pytest.fixture(autouse=True)
def _restore_global_health_state():
    """The gateway's health tracker is process-wide. The broken-local test below records a
    ``local`` failure into it, and that leaked into ``test_self_optimizing_router`` (which runs
    later and reads the same tracker). Measured: the router test failed only in the full suite.
    Snapshot and restore it so this file has no effect on any other test's ordering."""
    import copy

    from app.monitoring import monitor

    tracker = gateway_module.provider_health_tracker
    saved_tracker = copy.deepcopy(tracker._stats)
    # The router's evidence is the monitor's per-provider call record, not the tracker. A
    # successful local call recorded there is what made test_self_optimizing_router see
    # "observed" health and adapt weights when it ran later in the suite.
    saved_monitor = copy.deepcopy(dict(monitor.provider_stats))
    yield
    tracker._stats.clear()
    tracker._stats.update(saved_tracker)
    monitor.provider_stats.clear()
    monitor.provider_stats.update(saved_monitor)


@pytest.fixture
def cloud_unconfigured(monkeypatch):
    """Every cloud provider lacks a credential; the local tier is configured."""
    monkeypatch.setattr(gateway_module, "provider_has_credentials", lambda name: name == "local")
    monkeypatch.setattr(settings, "LOCAL_ENABLED", True)
    monkeypatch.setattr(settings, "LOCAL_PREFERRED", False)


@pytest.fixture
def local_provider(monkeypatch):
    import app.providers as providers_pkg

    fake = FakeLocalProvider()
    real_get = providers_pkg.get_provider

    def get_provider(name: str):
        return fake if name == "local" else real_get(name)

    monkeypatch.setattr(providers_pkg, "get_provider", get_provider)
    return fake


def _request() -> ProviderRequest:
    return ProviderRequest(messages=[ProviderMessage(role="user", content="Is 7 prime?")], max_tokens=64)


@pytest.mark.asyncio
async def test_local_tier_answers_when_the_primary_has_no_credential(cloud_unconfigured, local_provider):
    response = await gateway_module.model_gateway.execute("groq", _request(), capability="general")

    assert response.content == "served by the self-hosted tier"
    assert response.provider == "local"
    assert local_provider.calls, "the self-hosted tier must actually be called"
    provenance = (response.raw_response or {}).get("fallback_provenance", {})
    assert provenance.get("requested_provider") == "groq"
    assert provenance.get("actual_provider") == "local"


@pytest.mark.asyncio
async def test_no_local_tier_still_raises_the_same_unconfigured_error(monkeypatch):
    """Nothing can answer -> the caller gets the same error it always got, not a generic one."""
    monkeypatch.setattr(gateway_module, "provider_has_credentials", lambda name: False)
    monkeypatch.setattr(settings, "LOCAL_ENABLED", False)

    with pytest.raises(ProviderUnconfiguredError) as caught:
        await gateway_module.model_gateway.execute("groq", _request(), capability="general")

    assert "has no configured credential" in str(caught.value)
    assert caught.value.provider == "groq"


@pytest.mark.asyncio
async def test_local_tier_failure_still_surfaces_unconfigured_not_a_crash(monkeypatch):
    """A local tier that is enabled but broken must not turn into an unhandled exception."""
    import app.providers as providers_pkg

    class BrokenLocal:
        async def generate(self, request):  # noqa: ARG002 - interface stand-in
            raise ConnectionError("connection refused")

    real_get = providers_pkg.get_provider
    monkeypatch.setattr(providers_pkg, "get_provider",
                        lambda name: BrokenLocal() if name == "local" else real_get(name))
    monkeypatch.setattr(gateway_module, "provider_has_credentials", lambda name: name == "local")
    monkeypatch.setattr(settings, "LOCAL_ENABLED", True)
    monkeypatch.setattr(settings, "LOCAL_PREFERRED", False)

    with pytest.raises(ProviderUnconfiguredError):
        await gateway_module.model_gateway.execute("groq", _request(), capability="general")
