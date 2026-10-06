"""Regression tests: total provider failure must NEVER be reported as success.

Root cause under test (CRITICAL-1 / HIGH-5):
  app/providers/unified_manager.py used to return a synthesized template answer with
  a hardcoded token count of 120 and status "fallback_success" whenever every
  provider call failed, and every downstream caller reported that as
  status=SUCCESS / confidence=0.92 / "Empirical provider verification".

These tests assert the inverse: when no model can run, the API must return an
explicitly degraded status, zero confidence, empty evidence, and no fabricated
token counts.
"""

import pytest
from httpx import ASGITransport, AsyncClient

import app.providers.unified_manager as unified_manager_module
from app.core.config import settings
from app.main import app
from app.providers.unified_manager import UnifiedExecutionRequest, UnifiedExecutionResponse


def _no_provider_keys(monkeypatch):
    """Force an environment with zero configured provider credentials."""
    monkeypatch.setattr(settings, "GEMINI_API_KEYS", None)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)
    monkeypatch.setattr(settings, "GROQ_API_KEYS", None)
    monkeypatch.setattr(settings, "GROQ_API_KEY", None)
    monkeypatch.setattr(settings, "MISTRAL_API_KEY", None)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", None)
    monkeypatch.setattr(settings, "COHERE_API_KEY", None)
    monkeypatch.setattr(settings, "HUGGINGFACE_API_KEY", None)
    monkeypatch.setattr(settings, "NVIDIA_API_KEY", None)
    # Patch the class method: pydantic BaseSettings forbids per-instance setattr
    # for anything that is not a declared field.
    monkeypatch.setattr(
        type(settings), "get_provider_keys", lambda self, provider_name: []
    )


@pytest.mark.asyncio
async def test_unified_manager_marks_degradation_instead_of_fabricating(monkeypatch):
    """Provider failure -> explicit degraded response, empty content, no fake tokens."""
    _no_provider_keys(monkeypatch)

    resp = await unified_manager_module.unified_provider_manager.execute(
        UnifiedExecutionRequest(prompt="What is 2+2?", agent_role="system_architect")
    )

    assert isinstance(resp, UnifiedExecutionResponse)
    assert resp.degraded is True, "degradation must be flagged to callers"
    assert resp.status == "degraded"
    assert resp.content == "", "no model ran, so no answer may be synthesized"
    assert resp.token_usage == {}, "token counts must never be invented"
    assert resp.error, "the real provider error must be surfaced"
    # The old bug hardcoded exactly this number.
    assert resp.token_usage.get("total_tokens", 0) != 120


@pytest.mark.asyncio
async def test_v1_ask_returns_degraded_not_success_with_confidence(monkeypatch, auth):
    """POST /v1/ask must not claim SUCCESS/0.92/evidence when nothing was computed."""
    _no_provider_keys(monkeypatch)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://testserver", headers=auth) as client:
        r = await client.post("/v1/ask", json={"prompt": "What is the capital of France?"})

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "DEGRADED"
    assert body["failure_state"], "a failure reason must be reported"
    assert body["confidence"] == 0.0
    assert body["uncertainty"] == 1.0
    assert body["evidence"] == [], "no empirical evidence exists when no model ran"
    # The specific fabricated strings from the original bug report.
    assert not any("Empirical provider verification" in e for e in body["evidence"])
    assert "SUCCESS" != body["status"]


@pytest.mark.asyncio
async def test_v1_ask_degraded_hides_internal_system_prompt(monkeypatch, auth):
    """Degraded answers must not echo the internal ASTRA instruction scaffold."""
    _no_provider_keys(monkeypatch)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://testserver", headers=auth) as client:
        r = await client.post("/v1/ask", json={"prompt": "Explain caching."})

    body = r.json()
    assert "ASTRA DELIBERATIVE REASONING COUNCIL" not in body["answer"]
    assert "[ISOLATED" not in body["answer"]


@pytest.mark.asyncio
async def test_astra_reason_degraded_has_zero_confidence(monkeypatch, auth):
    _no_provider_keys(monkeypatch)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://testserver", headers=auth) as client:
        r = await client.post("/v1/astra/reason", json={"prompt": "weigh in on X"})

    assert r.status_code == 200
    body = r.json()
    assert body["confidence"] == 0.0, "a failed reasoning call cannot be 0.95 confident"


@pytest.mark.asyncio
async def test_task_execute_reports_degraded_status(monkeypatch, auth):
    _no_provider_keys(monkeypatch)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://testserver", headers=auth) as client:
        r = await client.post(
            "/v1/task/execute",
            json={
                "task_id": "t_test_degraded",
                "source_agent": "friday",
                "action": "reason",
                "payload": {"prompt": "hello"},
            },
        )

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "DEGRADED"
    assert body["error"], "the underlying provider error must be reported"
    assert body["result"]["token_usage"] == {}


@pytest.mark.asyncio
async def test_peer_assist_reports_degraded_status(monkeypatch, auth):
    _no_provider_keys(monkeypatch)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://testserver", headers=auth) as client:
        r = await client.post(
            "/v1/agent/assist",
            json={
                "caller_agent": "trading_bot",
                "task_type": "general",
                "prompt": "summarize this",
                "no_cache": True,
            },
        )

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["response"] == "", "a failure is not an answer"
    assert body["error"]


@pytest.mark.asyncio
async def test_degraded_results_are_not_cached_for_reuse(monkeypatch, auth):
    """A provider outage must not be replayed from cache after recovery."""
    _no_provider_keys(monkeypatch)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://testserver", headers=auth) as client:
        first = await client.post(
            "/v1/agent/assist",
            json={"caller_agent": "trading_bot", "task_type": "general", "prompt": "unique-prompt-xyz"},
        )
        assert first.json()["status"] == "degraded"

    # Now restore provider keys in settings and force a successful model call.
    from app.providers.base import ProviderResponse

    class _FakeProvider:
        async def generate(self, request):
            return ProviderResponse(
                content="recovered answer",
                model="fake-model",
                provider="gemini",
                total_tokens=5,
                prompt_tokens=2,
                completion_tokens=3,
            )

    async def _unused(*a, **k):  # pragma: no cover - placeholder never called
        return None

    def _fake_get_provider(name, **kwargs):
        # get_provider is synchronous: return a provider instance, not a coroutine.
        return _FakeProvider()

    monkeypatch.setattr(unified_manager_module, "get_provider", _fake_get_provider)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "restored-key")
    monkeypatch.setattr(
        type(settings), "get_provider_keys", lambda self, provider_name: ["restored-key"]
    )

    async with AsyncClient(transport=transport, base_url="http://testserver", headers=auth) as client:
        second = await client.post(
            "/v1/agent/assist",
            json={"caller_agent": "trading_bot", "task_type": "general", "prompt": "unique-prompt-xyz"},
        )

    assert second.json()["status"] == "success"
    assert second.json()["response"] == "recovered answer"
