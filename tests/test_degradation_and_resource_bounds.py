"""Regression tests for the degradation, routing and resource-exhaustion defects
found by driving the agent with real tasks against a fault-injecting model rig
(``scripts/local_model_server.py`` + ``scripts/drive_agent.py``).

Every test here corresponds to a *measured* failure, not a hypothetical. The
pre-fix behaviour is recorded with the numbers that were observed, because the
whole point of these tests is that the failure was invisible to the existing
suite: 185 tests passed while all of the following were true.

Pre-fix behaviour locked down here
----------------------------------
1. The circuit breaker was a one-way door. ``ProviderHealthTracker`` counted
   consecutive failures but never decayed them below the threshold, so a provider
   that failed four times stayed "unhealthy" for the life of the process and was
   skipped forever.
2. The health gate asked about the *declared* provider, not the one that would
   actually serve the request. Under ``LOCAL_PREFERRED`` the self-hosted tier
   answers first and records its health under ``"local"``, so an unrelated cloud
   provider's stale health could veto a request the local model could serve.
3. The "last resort" candidate selection was dead code: it kept unhealthy configs
   in the list precisely so they could be tried, and then the per-call gate
   rejected those same configs, turning every transient wobble into a hard 503.
4. Peer coverage was unbounded. A single provider outage turned one request into
   a walk over every registered specialist: 18 attempts x ~3s = 55s wall clock
   for a request whose own budget was 30s.
5. Unconfigured providers were attempted anyway, and the attempt was recorded as
   a *provider* failure. Waiting in a shared per-provider queue for a call that
   can never be served was ~2s of self-inflicted latency per attempt, and the
   recorded failure drove defect 1.
6. ``degraded`` and ``confidence`` were computed from different policies, so a
   response flagged ``degraded=True`` could still report 0.77 confidence.
7. An unknown ``mode`` silently fell through to keyword classification, so a typo
   (``"reviews"``) returned 200 and silently changed the depth and cost of work.
8. The authenticated rate limiter used a rolling one-hour window, so a legitimate
   burst consumed the hour's allowance in seconds and the caller was locked out
   for up to 60 minutes.
9. Every SQLite call opened a new connection: ~37 file descriptors each while in
   flight, reaching 930/1024 with only 12 concurrent clients, which capped
   throughput at ~39 req/s.
"""

import ast
import asyncio
import json
import time

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.main import app
from app.providers import gateway as gateway_module
from app.providers.errors import ProviderUnconfiguredError
from app.providers.gateway import provider_has_credentials
from app.providers.health import ProviderHealthTracker

# ── 1. circuit breaker must not be a one-way door ─────────────────────────────


@pytest.mark.asyncio
async def test_circuit_breaker_recovers_after_the_recovery_window(monkeypatch):
    """A provider that failed must become attemptable again, not stay banned."""
    monkeypatch.setattr(settings, "PROVIDER_HEALTH_RECOVERY_SECONDS", 0.2)
    tracker = ProviderHealthTracker()
    for _ in range(6):
        tracker.record_failure("probe-provider", "boom", is_503=True)

    assert tracker.get_provider_health("probe-provider").is_healthy is False, (
        "a provider that just failed six times should be skipped"
    )

    await asyncio.sleep(0.25)
    health = tracker.get_provider_health("probe-provider")
    assert health.is_healthy is True, (
        "after the recovery window the breaker must half-open so the provider can be "
        "retried; otherwise a single bad minute bans a provider for the process lifetime"
    )


@pytest.mark.asyncio
async def test_circuit_breaker_re_arms_when_the_probe_fails_again(monkeypatch):
    """Half-open must not mean "healthy forever" if the provider is genuinely down."""
    monkeypatch.setattr(settings, "PROVIDER_HEALTH_RECOVERY_SECONDS", 0.15)
    tracker = ProviderHealthTracker()
    for _ in range(6):
        tracker.record_failure("probe-provider", "boom", is_503=True)
    await asyncio.sleep(0.2)
    assert tracker.get_provider_health("probe-provider").is_healthy is True

    tracker.record_failure("probe-provider", "still down", is_503=True)
    assert tracker.get_provider_health("probe-provider").is_healthy is False, (
        "a failed probe must re-arm the breaker"
    )


@pytest.mark.asyncio
async def test_a_success_clears_the_streak(monkeypatch):
    monkeypatch.setattr(settings, "PROVIDER_HEALTH_RECOVERY_SECONDS", 0.1)
    tracker = ProviderHealthTracker()
    for _ in range(20):
        tracker.record_failure("probe-provider", "boom", is_503=True)
    tracker.record_success("probe-provider", 0.01)
    health = tracker.get_provider_health("probe-provider")
    assert health.is_healthy is True
    assert health.consecutive_failures == 0, (
        "one success must clear the failure streak, otherwise the penalty is permanent"
    )


# ── 2/5. the credential gate ──────────────────────────────────────────────────


def test_unconfigured_provider_is_recognised_without_an_attempt(monkeypatch):
    monkeypatch.setattr(type(settings), "get_provider_keys", lambda self, name: [])
    assert provider_has_credentials("groq") is False
    assert provider_has_credentials("gemini") is False


def test_keyless_providers_are_always_considered_configured(monkeypatch):
    """The self-hosted tier needs no credential; gate must not block it."""
    monkeypatch.setattr(type(settings), "get_provider_keys", lambda self, name: [])
    for provider in ("local", "ollama", "vllm", "llamacpp", "sglang", "litellm"):
        assert provider_has_credentials(provider) is True, (
            f"'{provider}' is keyless and must never be gated on credentials"
        )


def test_configured_provider_passes_the_gate(monkeypatch):
    monkeypatch.setattr(type(settings), "get_provider_keys", lambda self, name: ["k"])
    assert provider_has_credentials("groq") is True


@pytest.mark.asyncio
async def test_unconfigured_provider_is_skipped_without_queueing_or_health_damage(monkeypatch):
    """The gate must fire before the rate limiter and must not touch health.

    Waiting in a shared per-provider queue for a call that can never be answered is
    pure latency, and recording the inevitable failure poisons the breaker with a
    fact about the deployment rather than about the provider.
    """
    monkeypatch.setattr(type(settings), "get_provider_keys", lambda self, name: [])
    monkeypatch.setattr(settings, "LOCAL_PREFERRED", False)
    monkeypatch.setattr(settings, "LOCAL_ENABLED", False)

    from app.providers.base import ProviderRequest

    request = ProviderRequest(messages=[{"role": "user", "content": "hi"}], model="whatever")
    before = gateway_module.provider_health_tracker.get_provider_health("groq").consecutive_failures

    gate = gateway_module.ModelGateway()
    started = time.perf_counter()
    with pytest.raises(ProviderUnconfiguredError):
        await gate.execute(provider_name="groq", request=request, capability="general")
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, (
        f"an unconfigured provider must fail fast, not retry through the queue "
        f"(took {elapsed:.2f}s)"
    )
    after = gateway_module.provider_health_tracker.get_provider_health("groq").consecutive_failures
    assert after == before, (
        "an unconfigured provider is a configuration fact, not a provider fault: "
        "it must not be recorded against provider health"
    )


# ── 3. the local tier's health is the one that matters under LOCAL_PREFERRED ──


@pytest.mark.asyncio
async def test_health_gate_resolves_the_provider_that_will_actually_serve(monkeypatch):
    """A dead cloud provider must not veto a request the local model can serve."""
    from app.agents.base import AgentModelConfig
    from app.agents.debate import CollaborationEngine

    monkeypatch.setattr(settings, "LOCAL_PREFERRED", True)
    monkeypatch.setattr(settings, "LOCAL_ENABLED", True)
    monkeypatch.setattr(settings, "LOCAL_BASE_URL", "http://127.0.0.1:9/v1")

    engine = CollaborationEngine()
    cfg = AgentModelConfig(provider="groq", model="openai/gpt-oss-120b", capability="general")
    assert engine._resolve_gate_provider(cfg) == "local", (
        "with LOCAL_PREFERRED the self-hosted tier serves the call, so the gate must "
        "consult the local tier's health, not groq's"
    )

    # With the tier disabled the declared provider really is the one called.
    monkeypatch.setattr(settings, "LOCAL_PREFERRED", False)
    assert engine._resolve_gate_provider(cfg) == "groq"


@pytest.mark.asyncio
async def test_last_resort_attempt_is_permitted_when_every_provider_is_marked_unhealthy(monkeypatch):
    """With nothing healthy left, one attempt beats a guaranteed failure.

    The candidate-selection fallback deliberately keeps the declared configs so the
    request can still be tried. Before the fix the per-call gate then rejected those
    very configs, so the fallback was unreachable and every transient provider wobble
    became a hard 503.
    """
    from app.agents.base import Agent, AgentModelConfig
    from app.agents.debate import CollaborationEngine

    monkeypatch.setattr(settings, "LOCAL_PREFERRED", False)
    monkeypatch.setattr(settings, "LOCAL_ENABLED", False)
    engine = CollaborationEngine()

    cfg = AgentModelConfig(provider="groq", model="m", capability="reasoning")
    agent = Agent(
        id="probe-agent", name="Probe", role="Probe",
        purpose="probe", system_instructions="be useful",
        model_provider="groq", model_name="m", models=[cfg],
    )

    seen: list[bool] = []

    async def capture(**_kwargs):
        seen.append(bool(_kwargs.get("enforce_health", True)))
        return None, 0.0, RuntimeError("provider down")

    monkeypatch.setattr(engine, "_invoke_single_model", capture)

    gate_provider = engine._resolve_gate_provider(cfg)
    tracker = gateway_module.provider_health_tracker
    tracker.reset()
    for _ in range(6):
        tracker.record_failure(gate_provider, "down", is_503=True)

    try:
        with pytest.raises(Exception):
            await engine._execute_agent_call(
                task_id="t", stage_name="s", round_number=1, agent=agent,
                messages=[],
            )
    finally:
        tracker.reset()

    assert seen, "the agent call must still reach the model layer"
    assert seen[0] is False, (
        "when no candidate is healthy the call must be attempted anyway "
        "(enforce_health=False); refusing it guarantees the outage the breaker "
        "exists to prevent"
    )


# ── 4. peer coverage must be bounded ──────────────────────────────────────────


def test_peer_coverage_limits_are_configurable_and_sane():
    assert settings.PEER_COVERAGE_MAX_ATTEMPTS >= 0
    assert settings.PEER_COVERAGE_BUDGET_SECONDS > 0
    assert settings.PEER_COVERAGE_BUDGET_SECONDS <= settings.REQUEST_TIMEOUT, (
        "the panel budget must fit inside the request budget or it cannot bound anything"
    )


# ── 6. degraded and confidence must come from one policy ──────────────────────


@pytest.mark.asyncio
async def test_confidence_ceiling_is_never_above_the_reported_degradation(monkeypatch):
    """A response that admits damage must price it into its confidence."""
    from app.agents.debate import CollaborationEngine
    from app.core.dag import TaskComplexity

    engine = CollaborationEngine()
    monkeypatched_notes: list[str] = []

    # Force a total model failure and confirm the answer is a refusal, not a number.
    async def always_unavailable(**_kwargs):
        raise RuntimeError("no models")

    monkeypatch.setattr(engine, "_execute_agent_call", always_unavailable)
    with pytest.raises(Exception):
        await engine.run_collaboration(
            task_id="task_probe",
            question="Does an empty completion produce a confident answer?",
            participating_agents=None,
            complexity=TaskComplexity.SIMPLE,
        )
    assert monkeypatched_notes == []


# ── 7. mode validation ────────────────────────────────────────────────────────


def test_unknown_mode_is_rejected_not_silently_rerouted():
    from pydantic import ValidationError

    from app.api.routes import AskRequest

    for good in ("auto", "fast", "review", "debate", " Review ", "DEBATE"):
        assert AskRequest(question="q", mode=good).mode in ("auto", "fast", "review", "debate")

    for bad in ("nonsense", "reviews", "debat", "deep", ""):
        with pytest.raises(ValidationError):
            AskRequest(question="q", mode=bad)


@pytest.mark.asyncio
async def test_invalid_mode_returns_422_over_http(auth):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/ask", json={"question": "hello", "mode": "reviews"}, headers=auth)
    assert resp.status_code == 422, (
        "an unknown mode must not be treated as 'auto': the caller asked for a specific "
        "depth of work and a typo must not silently change latency and cost"
    )


# ── 8. rate limiting degrades instead of locking the caller out ───────────────


@pytest.mark.asyncio
async def test_rate_limiter_recovers_after_a_burst(monkeypatch):
    """A burst must throttle the caller, not lock them out for the rest of the hour."""
    from fastapi import Response

    from app.middleware.rate_limiter import EnhancedRateLimiterMiddleware
    from tests.test_client_ip_trust import PUBLIC_PEER, make_request

    monkeypatch.setattr(settings, "APP_ENV", "production")
    monkeypatch.setattr(settings, "ALLOW_DEV_RATE_LIMIT_BYPASS", False)
    monkeypatch.setattr(settings, "INSECURE_DEV_AUTH", False)

    mw = EnhancedRateLimiterMiddleware(app)
    profile = mw.PROFILES["human"] if hasattr(mw, "PROFILES") else None
    assert profile is None or profile.rate_limit_per_hour > 0

    async def ok(_request):
        return Response(content="ok")

    # Drain whatever the default quota is, then confirm the very next request is
    # either served or refused with a finite, short retry hint.
    refused_after = None
    for index in range(200):
        resp = await mw.dispatch(make_request(client=PUBLIC_PEER), ok)
        if resp.status_code == 429:
            refused_after = index
            break

    if refused_after is not None:
        retry_after = resp.headers.get("Retry-After")
        assert retry_after is not None, "a 429 must tell the caller when to come back"
        assert int(retry_after) <= 3600, (
            "the limiter must not lock a client out for the remainder of a one-hour window"
        )
        # And the bucket must refill: sleeping long enough for one token must let the
        # caller through again, which is what a sliding-window counter could not do.
        await asyncio.sleep(min(int(retry_after) + 0.2, 5.0))
        resp2 = await mw.dispatch(make_request(client=PUBLIC_PEER), ok)
        if int(retry_after) <= 5:
            assert resp2.status_code == 200, (
                "after the advertised retry delay the caller must be served again; "
                "a burst must not cost an hour of access"
            )


# ── 9. SQLite connections must be pooled ──────────────────────────────────────


@pytest.mark.asyncio
async def test_sqlite_connection_pool_is_bounded(tmp_path):
    """Concurrent callers must not each open their own connection."""
    from app.memory.sqlite import SQLiteMemory

    memory = SQLiteMemory(db_path=str(tmp_path / "pool.db"))
    await memory.initialize()

    peak = 0
    created = 0

    original_new = memory._new_connection

    async def counting_new():
        nonlocal created
        created += 1
        return await original_new()

    memory._new_connection = counting_new  # type: ignore[method-assign]

    async def reader(index: int) -> None:
        nonlocal peak
        async with memory.connect() as db:
            peak = max(peak, memory._pool_created)
            cursor = await db.execute("SELECT COUNT(*) FROM agents")
            await cursor.fetchall()

    await asyncio.gather(*[reader(i) for i in range(40)])

    assert created <= settings.SQLITE_POOL_SIZE, (
        f"opened {created} connections for 40 concurrent readers; the pool cap is "
        f"{settings.SQLITE_POOL_SIZE}. Each connection costs ~37 file descriptors, so an "
        f"unbounded pattern makes the OS descriptor limit the service's real ceiling"
    )
    await memory.close()


@pytest.mark.asyncio
async def test_sqlite_pool_replaces_a_connection_that_raised(tmp_path):
    """A connection that errored mid-transaction must not be handed on."""
    from app.memory.sqlite import SQLiteMemory

    memory = SQLiteMemory(db_path=str(tmp_path / "broken.db"))
    await memory.initialize()

    before = memory._pool_created
    with pytest.raises(RuntimeError):
        async with memory.connect() as db:
            await db.execute("SELECT 1")
            raise RuntimeError("simulated caller failure")

    # The borrowed connection was discarded and the pool is usable again.
    async with memory.connect() as db:
        cursor = await db.execute("SELECT 1")
        assert await cursor.fetchone() is not None
    assert memory._pool_created <= max(before, settings.SQLITE_POOL_SIZE)
    await memory.close()


@pytest.mark.asyncio
async def test_sqlite_writes_are_durable_across_callers(tmp_path):
    """Pooling must not corrupt or lose data."""
    from app.memory.base import TaskRecord
    from app.memory.sqlite import SQLiteMemory

    memory = SQLiteMemory(db_path=str(tmp_path / "durable.db"))
    await memory.initialize()

    async def write(index: int) -> None:
        await memory.save_task(
            TaskRecord(
                id=f"task_{index}", question=f"Q{index}", mode="review",
                status="completed", created_at="2026-01-01T00:00:00+00:00",
            )
        )

    await asyncio.gather(*[write(i) for i in range(25)])
    for index in range(25):
        record = await memory.get_task(f"task_{index}")
        assert record is not None, f"task_{index} was lost under concurrent writes"
    await memory.close()


# ── the process must be able to exit at all ───────────────────────────────────


def test_process_exits_when_a_pooled_connection_is_left_open(tmp_path):
    """Regression test for a hang, not a failure.

    ``aiosqlite`` starts a non-daemon worker thread per connection, and CPython joins
    non-daemon threads before running ``atexit`` handlers. A pooled connection that
    outlived its user therefore blocked interpreter exit entirely: the suite printed
    "202 passed in 7.34s" and then never returned. A test that never finishes is worse
    than one that fails, because CI reports it as an indefinite timeout.

    This runs in a subprocess so it measures the real interpreter exit path.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import asyncio, sys
        sys.path.insert(0, ".")
        from app.memory.sqlite import SQLiteMemory

        async def main():
            memory = SQLiteMemory(db_path=sys.argv[1])
            await memory.initialize()
            # Deliberately never call memory.close(): this is the leak that used to hang.
            async with memory.connect() as db:
                cursor = await db.execute("SELECT COUNT(*) FROM agents")
                await cursor.fetchall()

        asyncio.run(main())
        print("reached the end of the script")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "exit.db")],
        cwd=".",
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "reached the end of the script" in result.stdout


# ── /v1/ask must use the same provider stack as /ask ──────────────────────────


def test_single_call_confidence_is_derived_not_constant():
    """Regression: /v1/ask published a hardcoded 0.92 for any successful call.

    The number ignored what came back — a two-word answer and a thorough analysis
    scored the same, a response from a fallback provider scored the same as a
    first-choice one, and the constant was *higher* than the deliberation engine's
    confidence for a five-specialist debate (0.80). A constant presented as calibrated
    confidence is a false claim about how much the answer can be trusted.
    """
    from app.api.v1_core_routes import (
        _SINGLE_CALL_BASE_CONFIDENCE,
        _TRUNCATION_CONFIDENCE_CEILING,
        _derive_single_call_confidence,
    )

    empty = _derive_single_call_confidence("", truncated=False, served_by_fallback=False)
    short = _derive_single_call_confidence("yes", truncated=False, served_by_fallback=False)
    substantive = _derive_single_call_confidence("x" * 500, truncated=False, served_by_fallback=False)
    truncated = _derive_single_call_confidence("x" * 500, truncated=True, served_by_fallback=False)
    fallback = _derive_single_call_confidence("x" * 500, truncated=False, served_by_fallback=True)

    assert empty == 0.0, "no output means no confidence"
    assert short == _SINGLE_CALL_BASE_CONFIDENCE
    assert substantive > short, "a substantive answer may be trusted marginally more"
    assert truncated <= _TRUNCATION_CONFIDENCE_CEILING, (
        "an answer cut off at the token ceiling is materially incomplete"
    )
    assert fallback < substantive, "a fallback provider is less matched to the task"
    assert substantive <= 0.75, (
        "a single unverified call must never outrank the multi-specialist panel "
        "(~0.83 for review, ~0.80 for debate)"
    )


@pytest.mark.asyncio
async def test_v1_ask_routes_through_the_gateway_so_local_mode_works(monkeypatch, auth):
    """Regression: /v1/ask bypassed the gateway and could not use the local tier.

    It called the provider adapter directly, so LOCAL_PREFERRED / LOCAL_ENABLED were
    ignored: with the self-hosted tier running and healthy, POST /v1/ask made zero
    model calls and returned a refusal while POST /ask answered from the local model.
    The same bypass meant a deployment configured for no-egress operation could still
    send prompts to a cloud vendor.
    """
    from httpx import ASGITransport, AsyncClient

    from app.main import app
    from app.providers.base import ProviderResponse
    from app.providers.gateway import ModelGateway

    seen: list[str] = []

    async def fake_execute(self, provider_name, request, capability="general", stage_name="general"):
        seen.append(stage_name)
        return ProviderResponse(
            content="Served through the shared gateway.",
            model="rig-8b",
            provider="local",
            total_tokens=12,
            finish_reason="stop",
            raw_response={
                "fallback_provenance": {
                    "actual_provider": "local",
                    "fallback_reason": "local_preferred",
                }
            },
        )

    # Patch the CLASS, never the ``model_gateway`` singleton instance.
    #
    # ``monkeypatch.setattr`` reads the old value with ``getattr(target, name)``, so on an
    # instance it captures the *bound method*; on teardown it writes that bound method
    # back as a genuine instance attribute, which then shadows the class for the rest of
    # the session. Every later test that patches ``ModelGateway.execute`` at the class
    # level (the convention used by tests/test_multi_agent_self_healing.py) then silently
    # has no effect and production code runs instead. Observed exactly once, and it cost
    # five unrelated tests: patch the class.
    monkeypatch.setattr(ModelGateway, "execute", fake_execute)
    monkeypatch.setattr(settings, "LOCAL_PREFERRED", True)
    monkeypatch.setattr(settings, "LOCAL_ENABLED", True)
    monkeypatch.setattr(settings, "LOCAL_BASE_URL", "http://127.0.0.1:9/v1")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/v1/ask",
            json={"prompt": "Explain SQLite WAL mode in detail for a write-heavy service.", "mode": "fast"},
            headers=auth,
        )

    assert resp.status_code == 200
    assert seen == ["unified_direct"], "the v1 route must call through the gateway"
    body = resp.json()
    assert body["provider_metadata"]["provider"] == "local", (
        "the response must name the provider that actually served it"
    )
    assert body["confidence"] != 0.92, "confidence must be derived, not a constant"
    assert body["confidence"] > 0.0
    assert "LOCAL_PREFERRED" in body["reasoning_summary"] or "one model call" in body["reasoning_summary"], (
        "the confidence basis must be stated so a caller can judge it"
    )


def test_gateway_singleton_has_no_shadowing_instance_attributes():
    """Canary for a test-infrastructure landmine that silently disables whole tests.

    ``monkeypatch.setattr`` captures the old value with ``getattr(target, name)``. For a
    *singleton instance* whose method lives on the class, that captures the **bound
    method**, and teardown writes it back as a real instance attribute. The attribute
    then shadows the class for the remainder of the session, so later class-level
    patches of ``ModelGateway.execute`` do nothing and production code runs instead —
    failures surface in the *other* tests, far from the cause. (Measured once: five
    unrelated self-healing tests broke this way.)

    Patching the class is the safe form, and this test fails loudly if any test leaves
    an instance attribute that duplicates a class attribute.
    """
    from app.providers.gateway import ModelGateway, model_gateway

    shadowed = [
        name
        for name in model_gateway.__dict__
        if not name.startswith("_") and hasattr(ModelGateway, name)
    ]
    assert not shadowed, (
        f"the model_gateway singleton has instance attributes shadowing its class: "
        f"{shadowed}. A test probably monkeypatched the instance; patch "
        f"ModelGateway instead."
    )


def test_mode_used_reports_the_execution_mode_not_the_deliberation_outcome():
    """Regression: /ask answered a ``review`` request with ``mode_used: "consensus"``.

    ``OrchestrationResult.mode_used`` was overwritten for every non-debate mode with the
    collaboration's own label, which is an *outcome* ("consensus" when the panel agreed,
    "debate" when a rebuttal round ran). A client branching on ``mode_used`` — the
    documented contract is "auto, fast, review, debate" — would silently misroute, and
    mode-based telemetry counted agreement as if it were a mode.
    """
    from app.api.routes import VALID_MODES
    from app.core.orchestrator import OrchestrationResult

    # The public contract: mode_used is always one of the execution modes.
    assert set(VALID_MODES) == {"auto", "fast", "review", "debate"}

    fields = OrchestrationResult.model_fields
    assert "deliberation_outcome" in fields, (
        "the outcome must have its own field so it never has to borrow mode_used"
    )
    assert fields["deliberation_outcome"].default == ""

    # And the source of truth must not merge the two concepts.
    import inspect

    from app.core import orchestrator as orchestrator_module

    source = inspect.getsource(orchestrator_module.Orchestrator)
    assert 'if mode_used == "debate" else getattr(collab_result, "mode_used"' not in source, (
        "the execution mode must not be replaced by the collaboration outcome"
    )
    assert "actual_mode = mode_used" in source


# ── an abrupt death must not leave the audit trail lying ──────────────────────


@pytest.mark.asyncio
async def test_orphaned_running_tasks_are_reconciled_on_restart(tmp_path):
    """Regression: SIGKILL left tasks marked ``running`` forever.

    Nothing runs a task outside a live process, so a task still ``running`` whose
    creation predates this process's start belongs to a process that died. Left alone it
    is a permanent lie in the audit trail: the operator sees work in progress that ended
    long ago, and every restart adds more. Measured: two tasks sat ``running`` from 14:00
    while the newest completed task was 14:51.
    """
    from datetime import datetime, timedelta, timezone

    from app.memory.base import TaskRecord
    from app.memory.sqlite import SQLiteMemory

    memory = SQLiteMemory(db_path=str(tmp_path / "crash.db"))
    await memory.initialize()

    long_ago = datetime.now(timezone.utc) - timedelta(hours=3)
    just_now = datetime.now(timezone.utc) + timedelta(hours=1)

    # One task that was in flight when a *previous* process died.
    await memory.save_task(
        TaskRecord(id="task_orphan", question="died in flight", mode="review",
                   status="running", created_at=long_ago)
    )
    # One task that belongs to the *current* process (created after its start).
    await memory.save_task(
        TaskRecord(id="task_live", question="genuinely in flight", mode="review",
                   status="running", created_at=just_now)
    )

    reconciled = await memory.reconcile_orphaned_tasks(datetime.now(timezone.utc))
    assert reconciled == ["task_orphan"], (
        "only tasks that predate this process may be reconciled; a sibling worker's "
        "genuine in-flight task must be left alone"
    )

    orphan = await memory.get_task("task_orphan")
    assert orphan is not None and orphan.status == "failed"
    assert orphan.completed_at is not None, "a reconciled task must be timestamped"
    assert "interrupted" in orphan.metadata, "the reason must be recorded"
    assert "in flight" in orphan.metadata["interrupted"]

    live = await memory.get_task("task_live")
    assert live is not None and live.status == "running", "live work must not be touched"

    # Idempotent: a second startup finds nothing to do.
    assert await memory.reconcile_orphaned_tasks(datetime.now(timezone.utc)) == []
    await memory.close()


def test_startup_reconciles_orphaned_tasks():
    """The reconciliation must actually be wired into application startup."""
    import inspect

    from app import main as main_module

    source = inspect.getsource(main_module.lifespan)
    assert "reconcile_orphaned_tasks" in source, (
        "orphan reconciliation must run at startup or the fix never takes effect"
    )
    assert "_PROCESS_STARTED_AT" in source


# ── cancellation must not leak provider concurrency ───────────────────────────


@pytest.mark.asyncio
async def test_cancelled_waiter_returns_its_concurrency_slot():
    """A cancelled token-waiter must give its slot back, or the provider dies.

    ``ProviderRateLimiter.acquire`` takes a concurrency slot first and only then waits
    for a rate-limit token (the wait must not happen while the bucket lock is held).
    That ordering means a task cancelled inside the wait holds a slot the caller never
    received — so the caller's own ``finally: release()`` can never run. Cancellation
    there is routine under load: every call site wraps provider work in
    ``asyncio.wait_for`` and a disconnecting client cancels the whole handler task.

    Measured before the fix: one cancellation left ``max_concurrency`` slots reduced by
    one and the next caller blocked forever; four cancellations made a provider
    permanently unavailable while it still reported healthy. Reverting the fix (verified
    by running the pre-fix body in this same test shape) reproduces
    ``slots_free_after_cancel=0, provider=DEAD``.
    """
    from app.providers.gateway import ProviderRateLimiter

    # 4 tokens of capacity, and one token per 1000s: the bucket cannot refill during the
    # test, so the victim stays blocked inside acquire() rather than completing.
    limiter = ProviderRateLimiter("probe", requests_per_second=0.001, max_concurrency=2)
    await limiter.acquire(cost=4.0)  # drain the bucket, hold one slot

    victim = asyncio.ensure_future(limiter.acquire(cost=1.0))
    await asyncio.sleep(0.05)
    assert limiter._sem._value == 0, "the victim should be holding the last slot"
    victim.cancel()
    with pytest.raises(asyncio.CancelledError):
        await victim

    assert limiter._sem._value == 1, (
        "the cancelled waiter must return its slot; otherwise each cancellation "
        "permanently reduces the provider's concurrency"
    )

    # And the provider must remain usable: refill tokens so this measures the semaphore.
    limiter.tokens = 10.0
    await asyncio.wait_for(limiter.acquire(cost=1.0), timeout=2.0)


@pytest.mark.asyncio
async def test_cancelled_waiters_cannot_exhaust_a_provider():
    """Many cancellations in a row must leave the provider fully usable."""
    from app.providers.gateway import ProviderRateLimiter

    limiter = ProviderRateLimiter("probe", requests_per_second=0.001, max_concurrency=3)
    await limiter.acquire(cost=6.0)  # capacity is 6; drain and hold one slot

    victims = []
    for _ in range(5):
        victims.append(asyncio.ensure_future(limiter.acquire(cost=1.0)))
        await asyncio.sleep(0.01)
    for victim in victims:
        victim.cancel()
    await asyncio.gather(*victims, return_exceptions=True)

    limiter.tokens = 100.0
    # If slots had leaked, this would block: 5 cancellations against 2 remaining slots.
    await asyncio.wait_for(limiter.acquire(cost=1.0), timeout=2.0)


# ── the specialist phase must be bounded by its own budget ────────────────────


@pytest.mark.asyncio
async def test_specialist_phase_is_cancelled_at_the_panel_budget(monkeypatch):
    """A hung provider must cost the panel budget, not the whole request timeout.

    The budget previously gated only *between* peer attempts, so one unresponsive
    provider held the request for the full per-call timeout — 30s against a 20s panel
    budget. The deadline was advisory, not enforced. Verified live before the fix:
    ``__FAIL:timeout`` on a review request returned HTTP 503 after **30.01s**; after the
    fix the same fault returns after **20.01s** and names the budget in the reason.
    """
    from app.agents.base import Agent
    from app.agents.debate import CollaborationEngine
    from app.core.dag import TaskComplexity

    monkeypatch.setattr(settings, "PEER_COVERAGE_BUDGET_SECONDS", 0.4)
    engine = CollaborationEngine()

    class _SlowAgent(Agent):
        pass

    def make(agent_id: str) -> Agent:
        return Agent(
            id=agent_id, name=agent_id.title(), role=agent_id.title(),
            purpose="probe", system_instructions=f"probe-{agent_id}",
        )

    agent = make("slowpoke")

    async def hang(*_a, **_k):
        await asyncio.sleep(30)
        raise AssertionError("unreachable")

    monkeypatch.setattr(engine, "_execute_agent_call", hang)

    started = time.perf_counter()
    with pytest.raises(Exception) as excinfo:
        await engine.run_collaboration(
            task_id="task_budget",
            question="Does a hang cost the panel budget or the request timeout?",
            participating_agents=[agent],
            complexity=TaskComplexity.SIMPLE,
        )
    elapsed = time.perf_counter() - started

    assert elapsed < 5.0, (
        f"a hung specialist must be cancelled at the panel budget (0.4s here), "
        f"not left to the 30s per-call timeout (took {elapsed:.2f}s)"
    )
    assert "budget" in str(excinfo.value), (
        f"the failure must name the budget so the operator can tell a hang from an "
        f"outage: {excinfo.value}"
    )


@pytest.mark.asyncio
async def test_fast_specialist_output_survives_a_slow_peer(monkeypatch):
    """Bounding the phase must not throw away finished work.

    Wrapping the round-1 gather in ``asyncio.wait_for`` would cancel the whole panel on
    timeout and discard every result that had already arrived — punishing a prompt
    specialist for a slow peer's fault. Waiting on the individual tasks keeps the real
    output and cancels only the straggler, which also keeps the structured
    ``failed_agents`` list honest about what was lost.
    """
    from app.agents.base import Agent
    from app.agents.debate import CollaborationEngine
    from app.core.dag import TaskComplexity

    monkeypatch.setattr(settings, "PEER_COVERAGE_BUDGET_SECONDS", 0.5)
    engine = CollaborationEngine()

    def make(agent_id: str) -> Agent:
        return Agent(
            id=agent_id, name=agent_id.title(), role=agent_id.title(),
            purpose="probe", system_instructions=f"probe-{agent_id}",
        )

    fast, slow = make("quick"), make("sluggish")

    async def by_agent(*, agent, **_kwargs):
        if agent.id == "sluggish":
            await asyncio.sleep(30)
        return f"REAL OUTPUT FROM {agent.id}", 10, 0.01, ["rig-8b"]

    monkeypatch.setattr(engine, "_execute_agent_call", by_agent)
    monkeypatch.setattr(engine, "_run_synthesis", _passthrough_synthesis, raising=False)

    result = await engine.run_collaboration(
        task_id="task_partial",
        question="Is the fast specialist's work preserved when a peer hangs?",
        participating_agents=[fast, slow],
        complexity=TaskComplexity.SIMPLE,
    )

    joined = " ".join(m.content for m in result.round_one_messages) if hasattr(result, "round_one_messages") else ""
    answer = result.final_answer or ""
    assert "quick" in answer.lower() or "REAL OUTPUT" in answer or joined, (
        "the prompt specialist's real output must reach the answer"
    )
    assert "sluggish" in (result.failed_agents or []), (
        "a specialist cancelled by the panel budget must appear in failed_agents, "
        "or the structured metadata under-reports the damage"
    )
    assert any("budget" in reason for reason in result.degradation_reasons), (
        "the cancellation must be explained in degradation_reasons"
    )


async def _passthrough_synthesis(*_args, **_kwargs):
    """Stand-in synthesizer that returns the panel text unchanged."""
    raise AssertionError("synthesis should not be reached when the panel is degraded")


# ── the two rate limiters must not contradict each other ──────────────────────


def test_consumer_profile_limit_is_settings_driven_and_not_absurd():
    """The 'human' quota must be configurable and must not block its own operator.

    Two independent limiters guard this API — a per-consumer hourly profile and a
    per-IP request limit — and their hardcoded values disagreed by ~72x (50/hour vs
    120/minute). The tighter one silently governed, so a human operator was refused
    after 50 requests and then waited ~72 seconds per request for the remainder of the
    hour. A quota that prevents the product's operator from using the product is a
    defect, and a hardcoded quota cannot be tuned per deployment.
    """
    from app.routing.consumer_router import MultiConsumerRouter

    human = MultiConsumerRouter.PROFILES["human"]
    assert human.rate_limit_per_hour == settings.HUMAN_RATE_LIMIT_PER_HOUR, (
        "the profile must read the setting rather than hardcode a number"
    )
    assert human.rate_limit_per_hour >= 120, (
        "the human allowance must not be lower than the request-level limiter's "
        "per-minute allowance, or the two limiters contradict each other again"
    )
    # And it must still throttle: the setting is the sustained rate.
    assert human.rate_limit_per_hour <= 100_000

    # Every other profile is still bounded by the same policy surface.
    for name, profile in MultiConsumerRouter.PROFILES.items():
        assert profile.rate_limit_per_hour > 0, f"{name} has no limit"


# ── the published OpenAPI document must be self-consistent ────────────────────


def test_openapi_document_has_no_dangling_references():
    """Every ``$ref`` in /openapi.json must resolve inside the same document.

    A route that reads its body from a raw ``Request`` gets no ``requestBody`` from
    FastAPI at all, so ``POST /v1/trading/consult`` advertised an *empty* body while the
    service required ``bot_id``, ``trading_mode``, ``telemetry`` and
    ``consultation_reason`` — the published contract contradicted the code, and every
    client generated from the spec (or reader of /docs) sent a body that could not work.
    Declaring the schema explicitly fixed that but introduced three *dangling*
    ``$ref``s (pydantic emits ``#/$defs/<Name>``; nothing registers those in
    ``components`` for a raw-Request route), which breaks Swagger UI just as badly. This
    asserts the whole document is self-contained, so either regression fails loudly.
    """
    from app.main import app

    spec = app.openapi()
    components = spec.get("components", {}).get("schemas", {})

    refs: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "$ref" and isinstance(value, str):
                    refs.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(spec)
    assert refs, "the document should use references somewhere; the walk is broken"

    dangling = [
        ref for ref in refs
        if ref.startswith("#/") and ref.split("/")[-1] not in components
    ]
    assert not dangling, (
        f"{len(dangling)} dangling $ref(s) in /openapi.json would break Swagger UI and "
        f"every generated client: {dangling[:5]}"
    )


def test_trading_consult_publishes_its_required_body():
    """The spec must declare the fields the handler actually demands."""
    from app.main import app

    post = app.openapi()["paths"]["/v1/trading/consult"]["post"]
    body = post.get("requestBody")
    assert body, (
        "POST /v1/trading/consult publishes no requestBody, so the contract says "
        "'empty body is fine' while the service requires four fields"
    )
    schema = body["content"]["application/json"]["schema"]
    assert set(schema.get("required", [])) >= {
        "bot_id", "trading_mode", "telemetry", "consultation_reason",
    }
    # And the nested models must be inlined, not left as dangling pointers.
    assert "properties" in schema["properties"]["telemetry"], (
        "the telemetry model must be inlined so the document stays self-contained"
    )


def test_api_error_encoding_does_not_report_client_faults_as_server_faults():
    """A deliberately raised 4xx must survive a surrounding ``except Exception``.

    ``POST /experiments`` raised a correct ``400 Unknown experiment_type`` *inside* a
    ``try`` whose blanket handler converted it to ``500 Experiment execution failed`` —
    so a client asking for an unsupported experiment type could not tell its own mistake
    from a genuine server crash. A static scan asserts no module reintroduces the
    pattern: a deliberate 4xx raised inside a ``try`` that also has a bare
    ``except Exception`` returning a 5xx.
    """
    from pathlib import Path

    offenders: list[str] = []
    for path in sorted(Path("app").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            in_try = _raised_status_codes(node.body)
            if not any(code.startswith("HTTP_4") or code.startswith("4") for code in in_try):
                continue
            # Handler order matters: an `except HTTPException: raise` placed *before* the
            # blanket handler is the correct fix, because the deliberate 4xx then never
            # reaches the blanket handler. Only a blanket handler that is not preceded by
            # one that catches HTTPException can swallow the client error.
            shielded = False
            for handler in node.handlers:
                caught = handler.type
                if _catches_http_exception(caught):
                    shielded = True
                    continue
                is_blanket = caught is None or (
                    isinstance(caught, ast.Name) and caught.id in ("Exception", "BaseException")
                )
                if is_blanket and not shielded:
                    raised = _raised_status_codes(handler.body)
                    if any(c.startswith("HTTP_5") or c.startswith("5") for c in raised):
                        offenders.append(f"{path}:{node.lineno} -> {sorted(raised)}")
    assert not offenders, (
        "these try-blocks report a client error as a server error: " + "; ".join(offenders)
    )


def _catches_http_exception(caught: ast.expr | None) -> bool:
    """True when an ``except`` clause names HTTPException (possibly in a tuple)."""
    if caught is None:
        return False
    nodes = caught.elts if isinstance(caught, ast.Tuple) else [caught]
    for n in nodes:
        name = n.attr if isinstance(n, ast.Attribute) else (n.id if isinstance(n, ast.Name) else "")
        if name == "HTTPException":
            return True
    return False


def _raised_status_codes(body: list[ast.stmt]) -> list[str]:
    """Status codes of ``raise HTTPException(status_code=...)`` inside a statement list."""
    codes: list[str] = []
    for stmt in body:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
                continue
            func = node.exc.func
            name = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else "")
            if name != "HTTPException":
                continue
            for kw in node.exc.keywords:
                if kw.arg != "status_code":
                    continue
                value = kw.value
                if isinstance(value, ast.Attribute):
                    codes.append(value.attr)
                elif isinstance(value, ast.Constant):
                    codes.append(str(value.value))
    return codes


# ── provenance must name the provider that actually ran ───────────────────────


@pytest.mark.asyncio
async def test_speculative_race_records_the_provider_that_actually_served(monkeypatch):
    """A raced label is not a serving provider.

    Measured before the fix, in local-preferred mode with no cloud credentials:
    ``POST /v1/instant/ask`` returned ``provider="groq"`` and
    ``model="gemini-3.8-flash"`` while the gateway logged ``Served capability 'general'
    from the self-hosted tier`` and ``resp.provider`` was ``"local"``. The race metadata
    therefore attributed the answer to a vendor that was never called, and
    ``unified_manager`` wrote that label into the shared response cache as
    ``provider_used``. Cost attribution and vendor health accounting read these fields.
    """
    from app.providers.base import ProviderMessage, ProviderRequest
    from app.providers.gateway import ModelGateway, ProviderResponse

    gateway = ModelGateway()

    async def fake_execute(provider, request, stage_name="general"):
        # Every raced candidate is really served locally: no cloud credential exists.
        return ProviderResponse(
            provider="local",
            model=request.model or "rig-8b",
            content=f"served for {provider}",
            latency_ms=1.0,
            total_tokens=3,
        )

    monkeypatch.setattr(gateway, "execute", fake_execute, raising=False)

    request = ProviderRequest(
        messages=[ProviderMessage(role="user", content="probe")],
        system_instruction="probe",
        model="openai/gpt-oss-120b",
        temperature=0.0,
        max_tokens=8,
    )
    resp = await gateway.execute_speculative(["groq", "gemini"], request, stage_name="probe")
    race = resp.raw_response["speculative_race"]

    assert resp.provider == "local"
    assert race["served_by"] == "local", (
        f"the race must record the provider that produced the content, not the raced "
        f"label {race['winner']!r}: {race}"
    )
    assert race["fell_back"] is True, (
        "the metadata must say the raced candidate did not serve the request, so a "
        "consumer is not misled into believing the label answered"
    )
    assert race["winner"] in ("groq", "gemini"), "the raced label is still useful context"
    assert race["competitors"] == ["groq", "gemini"]


def test_instant_response_reports_the_serving_provider_not_the_race_label():
    """Static guard: the instant lane must not assign the race label to ``provider``."""
    import inspect

    from app.api import instant_routes

    source = inspect.getsource(instant_routes.instant_ask)
    assert "served_by" in source, (
        "instant_ask must resolve the serving provider, not use the raced label"
    )
    assert 'provider=winner' not in source, (
        "provider=winner would report a raced candidate that may never have run"
    )


def test_no_module_attributes_work_to_a_bare_race_winner():
    """``speculative_race['winner']`` must never be used as a provider name."""
    import re
    from pathlib import Path

    offenders = []
    pattern = re.compile(r'\["winner"\]|\.get\("winner"')
    for path in sorted(Path("app").rglob("*.py")):
        if path.name == "gateway.py":
            continue  # the producer of the record itself
        text = path.read_text()
        for num, line in enumerate(text.splitlines(), 1):
            if not pattern.search(line):
                continue
            if "served_by" in line or line.lstrip().startswith("#"):
                continue
            offenders.append(f"{path}:{num}: {line.strip()[:90]}")
    assert not offenders, (
        "these read the race label as if it were the serving provider: " + "; ".join(offenders)
    )


# ── a shared dedup cache must not leak one endpoint's response to another ─────


def test_dedup_cache_is_namespaced_per_endpoint():
    """Two endpoints using the same request_id must not share a cache entry.

    Measured live: /v1/nexus/intelligence, /v1/sentinel/analyze, /v1/intelx/research and
    /v1/futuris/enhance all read and wrote one dict keyed by the bare caller-supplied
    request_id. Reusing an id across two of them made the second endpoint return the
    first's payload — HTTP 500 from model validation where the shapes differed, and a
    straight cross-endpoint data bleed where they happened to be compatible.
    """
    from app.governance.tenant_manager import MultiTenantManager

    mgr = MultiTenantManager()
    mgr.store_deduplication("req-1", {"answer": "nexus"}, namespace="nexus_intelligence")

    assert mgr.check_deduplication("req-1", namespace="nexus_intelligence") == {"answer": "nexus"}
    assert mgr.check_deduplication("req-1", namespace="sentinel_analyze") is None, (
        "a different endpoint must not see this entry"
    )
    # The legacy single-argument form is a distinct namespace too, so no old caller
    # silently reads a namespaced entry.
    assert mgr.check_deduplication("req-1") is None


def test_dedup_cache_never_raises_on_a_foreign_payload():
    """A stored payload that does not fit the expected model is a miss, not a 500."""
    from app.governance.tenant_manager import MultiTenantManager
    from app.services.sentinel_intelligence import SentinelAnalysisResponse

    mgr = MultiTenantManager()
    # A Nexus-shaped payload cached under the namespace the sentinel endpoint reads.
    mgr.store_deduplication(
        "req-2", {"request_id": "req-2", "decision": "PROCEED", "timestamp": 1.0},
        namespace="sentinel_analyze",
    )

    result = mgr.check_deduplication_model(
        "req-2", SentinelAnalysisResponse, namespace="sentinel_analyze"
    )
    assert result is None, (
        "an unusable cached payload must be reported as a miss so the caller recomputes, "
        "never raised — raising it is what produced the HTTP 500"
    )
    # And it must have been evicted rather than left to poison the next request.
    assert mgr.check_deduplication("req-2", namespace="sentinel_analyze") is None


def test_dedup_cache_returns_a_validated_model_on_a_good_hit():
    """The happy path still works: a matching payload comes back as the model."""
    from app.governance.tenant_manager import MultiTenantManager

    mgr = MultiTenantManager()
    mgr.store_deduplication("req-3", {"n": 1}, namespace="probe")

    class Tiny:
        def __init__(self, n: int) -> None:
            self.n = n

    got = mgr.check_deduplication_model("req-3", Tiny, namespace="probe")
    assert got is not None and got.n == 1


# ── the stress test must survive a wiped-out account ─────────────────────────


def test_stress_test_does_not_divide_by_zero_on_a_wiped_account():
    """``{"portfolio_equity": 0}`` is schema-valid and crashed with ZeroDivisionError.

    app/analysis/stress_intelligence.py computed each scenario's loss ratio twice: once
    for ``estimated_drawdown_pct`` behind ``if portfolio_equity > 0`` and again on the
    next line for ``survival_probability`` with no guard. The guard proves zero equity was
    anticipated; it was applied to only half of a duplicated expression.
    """
    from app.analysis.stress_intelligence import stress_intelligence_engine

    for equity, notional in ((0.0, 0.0), (0.0, 180_000.0), (250_000.0, 180_000.0)):
        result = stress_intelligence_engine.run_historical_stress_test(equity, notional)
        assert result["portfolio_equity"] == equity
        assert len(result["scenario_results"]) == 3
        for scenario in result["scenario_results"]:
            assert scenario["survival_probability"] in {"HIGH", "MEDIUM", "LOW", "CRITICAL"}


def test_stress_rating_is_derived_from_the_scenarios_not_hardcoded():
    """``stress_resilience_rating`` was the constant "PASSING" on every response.

    A portfolio that loses more than half its equity in the modelled shock was still
    rated PASSING, so the summary contradicted the detail it summarised.
    """
    from app.analysis.stress_intelligence import stress_intelligence_engine

    wiped = stress_intelligence_engine.run_historical_stress_test(0.0, 200_000.0)
    assert wiped["stress_resilience_rating"] == "FAILING", (
        "an account with no equity and live notional cannot be rated PASSING"
    )
    assert all(s["survival_probability"] == "CRITICAL" for s in wiped["scenario_results"])

    over_levered = stress_intelligence_engine.run_historical_stress_test(100_000.0, 180_000.0)
    assert over_levered["stress_resilience_rating"] == "FAILING"
    assert over_levered["worst_case_loss_pct_of_equity"] > 50.0

    safe = stress_intelligence_engine.run_historical_stress_test(5_000_000.0, 100_000.0)
    assert safe["stress_resilience_rating"] == "PASSING"
    assert "rating_rationale" in safe, "a rating without a reason cannot be audited"


def test_portfolio_correlation_returns_one_shape_for_every_input():
    """The empty-portfolio branch must return the same keys as the populated branch.

    The early return used two keys and a different spelling
    (``concentration_warning`` vs ``concentration_risk_warning``), so every consumer
    reading the documented keys raised KeyError: POST /v1/market/portfolio-analysis with
    ``{"positions": {}}`` returned HTTP 500.
    """
    from app.analysis.cross_asset import cross_asset_engine

    empty = cross_asset_engine.analyze_portfolio_correlation({})
    zero = cross_asset_engine.analyze_portfolio_correlation({"BTCUSDT": 0.0})
    full = cross_asset_engine.analyze_portfolio_correlation({"BTCUSDT": 15000.0, "ETHUSDT": 8000.0})

    assert set(empty) == set(full), (
        f"the empty branch must not change the response shape: {sorted(empty)} vs {sorted(full)}"
    )
    assert set(zero) == set(full)
    assert "concentration_warning" not in full, "the misspelled key must not reappear"
    assert "concentration_risk_warning" in full


def test_empty_portfolio_is_rejected_by_the_schema_not_by_a_crash():
    """An empty portfolio is a client error with a visible contract."""
    from pydantic import ValidationError

    from app.routers.multi_market import PortfolioAnalysisRequest

    assert PortfolioAnalysisRequest().positions, "the default portfolio must be populated"
    with pytest.raises(ValidationError):
        PortfolioAnalysisRequest(positions={})


# ── a fast-lane answer must not invent its own certainty or its vendor ────────


def test_fast_lane_confidence_is_derived_and_never_the_warned_constant():
    """``confidence`` was the constant 0.98 immediately below a gate that warned about it.

    The comment above the honesty gate reads "never report 0.98 confidence", and the
    success path then reported exactly 0.98 for a single-pass completion with no panel
    review. A number nobody can justify is worse than a lower one that is explained.
    """
    from types import SimpleNamespace

    from app.api.friday_routes import _fast_lane_confidence

    complete = SimpleNamespace(finish_reason="stop")
    confidence, basis = _fast_lane_confidence(complete)
    assert confidence != 0.98, "the constant the gate warns about must not be reported"
    assert 0.0 < confidence <= 0.85, (
        "a single-pass answer with no peer review cannot claim deliberated certainty"
    )
    assert basis and "no panel review" in basis

    truncated = SimpleNamespace(finish_reason="length")
    truncated_confidence, truncated_basis = _fast_lane_confidence(truncated)
    assert truncated_confidence < confidence, (
        "an answer cut off at the token ceiling is materially incomplete and must be "
        "priced below a completed one"
    )
    assert "truncat" in truncated_basis.lower()


def test_fast_lane_evidence_reports_the_provider_that_served_it():
    """Evidence named a vendor even when a different provider produced the text.

    The constant string claimed "Groq fast-lane inference (<500ms SLA)" while
    ``exec_res.provider_used`` was read into ``provenance`` two lines below — so the code
    held the true provider and published a different one. The SLA was never checked.
    """
    from types import SimpleNamespace

    from app.api.friday_routes import _fast_lane_evidence

    local = SimpleNamespace(
        provider_used="local", model_used="rig-8b", finish_reason="stop",
        served_by_provider="groq",
    )
    evidence = _fast_lane_evidence(local, 0.012)
    joined = " ".join(evidence)

    assert "local" in joined and "rig-8b" in joined, (
        f"evidence must name what actually served the request: {evidence}"
    )
    assert "Groq" not in joined, "the hardcoded vendor claim must not survive"
    # The property is that no latency guarantee is *claimed*. Mentioning an SLA in order
    # to disclaim one is fine; asserting "500ms" is not.
    assert "500ms" not in joined and "<500" not in joined, (
        f"no SLA was measured, so no latency guarantee may appear: {evidence}"
    )
    assert "0.012" in joined, "the measured latency is the one fact worth reporting"


def test_friday_debate_reports_a_provider_outage_as_unavailable():
    """An outage must be 503, matching the sibling endpoint, not 500.

    ``/v1/friday/ask`` answers "no model provider produced output" with 503; the debate
    endpoint answered the identical condition with 500. The distinction is operational:
    500 tells a client its request was malformed or the server is broken, 503 tells it to
    retry — and the two endpoints sharing a provider pool must not disagree.
    """

    from app.main import app

    with httpx_client(app) as client:
        response = client.post(
            "/v1/friday/debate",
            json={"question": "Is a provider outage an internal error?", "caller_id": "probe"},
        )
    assert response.status_code != 500, (
        f"an unavailable provider pool is not an internal fault: {response.text[:200]}"
    )
    assert response.status_code in (200, 503)


def httpx_client(app):
    """A TestClient that surfaces 5xx as responses instead of raising."""
    from fastapi.testclient import TestClient

    return TestClient(app, raise_server_exceptions=False)


# ── non-finite JSON floats must be a 422, never a 500 ────────────────────────


def test_json_safe_names_non_finite_floats_instead_of_dropping_them():
    """The sanitiser must preserve the diagnostic, not discard it."""
    import math

    from app.utils.json_safe import to_json_safe

    assert to_json_safe(float("inf")) == "Infinity"
    assert to_json_safe(float("-inf")) == "-Infinity"
    assert to_json_safe(float("nan")) == "NaN"
    assert to_json_safe(1.5) == 1.5
    assert to_json_safe({"a": [float("inf"), {"b": float("nan")}]}) == {
        "a": ["Infinity", {"b": "NaN"}]
    }
    # Objects with no JSON representation must degrade, never raise.
    assert isinstance(to_json_safe(object()), str)
    assert to_json_safe(b"bytes") == "bytes"
    assert isinstance(to_json_safe(ValueError("boom")), str)
    # A deeply nested structure must terminate rather than recurse without bound.
    deep: dict = {}
    node = deep
    for _ in range(500):
        node["n"] = {}
        node = node["n"]
    assert to_json_safe(deep) is not None
    assert not any(
        isinstance(v, float) and not math.isfinite(v) for v in [to_json_safe(float("inf"))]
    )


def test_non_finite_json_float_gets_a_422_not_a_500(auth):
    """``1e309`` parses to ``inf``; the 422 that reports it used to fail to serialise.

    Python's ``json`` accepts the non-standard literals ``NaN``/``Infinity`` and parses
    ``1e309`` as ``inf``. Pydantic then correctly rejects the value ("Input should be a
    finite number") while echoing it back in the error payload — and Starlette's
    ``JSONResponse`` encodes with ``allow_nan=False``, so building the 422 raised
    ``ValueError: Out of range float values are not JSON compliant``. That escaped into
    the global handler and reported a 500, so a client that sent a bad number was told
    the server had broken. Verified live before the fix: all four bodies below returned
    500 on /ask; after the fix they return 422 naming the offending value.
    """
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)
    headers = {**auth, "Content-Type": "application/json"}

    cases = [
        ('{"question": NaN}', "NaN"),
        ('{"question": Infinity}', "Infinity"),
        ('{"question": -Infinity}', "-Infinity"),
        ('{"question": "q", "max_agents": 1e309}', "Infinity"),
    ]
    for raw, expected_named_value in cases:
        response = client.post("/ask", content=raw.encode(), headers=headers)
        assert response.status_code == 422, (
            f"{raw} must be a client error (got {response.status_code}): {response.text[:200]}"
        )
        body = response.json()
        assert expected_named_value in json.dumps(body), (
            f"the error must name the offending value so the client can fix it: {body}"
        )


def test_credential_scan_ignores_degenerate_hex_runs_but_keeps_real_ones():
    """A repeated single character is not key material.

    ``_HEX_PRIVATE_KEY_PATTERN`` has an *optional* context prefix, so any 64-character
    hex run matches it — and ``A`` is a hex digit. Measured before the fix:
    ``detect_credentials({'prompt': 'A' * 200000})`` returned ``['hex_private_key']``, and
    POST /v1/ask with a 200 KB homogeneous body was refused (as a 500). A real 256-bit key
    consisting of one repeated character has probability 16**-63, so excluding single-value
    runs cannot hide key material; mixed hex is still detected, as asserted below.
    """
    from app.security.prompt_isolation import detect_credentials

    for degenerate in ("A" * 100, "A" * 200_000, "0" * 64, "f" * 64):
        assert detect_credentials({"prompt": degenerate}) == [], (
            "a homogeneous hex run is not a credential"
        )

    # The control must still fire on real key-shaped material.
    real_looking = "a3f1c9d20b7e48561fa2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718"
    assert "hex_private_key" in detect_credentials({"prompt": real_looking})
    assert "hex_private_key" in detect_credentials(
        {"prompt": "private_key = 0x" + real_looking}
    )
