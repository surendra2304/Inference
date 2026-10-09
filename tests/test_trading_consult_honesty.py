"""H6: trading consultation must not present fallback text as deliberation.

Pre-fix behaviour locked down here:
  * RunRecord was written with status="completed" even when the provider
    raised, while simultaneously carrying an `error` string.
  * Empty provider content silently became canned prose still recorded as
    a completed model run.
  * Decision confidence was a hardcoded 0.88 whether or not any specialist
    ever produced model output.
  * debate_summary claimed "Multi-Agent Deliberation" and named specialists
    whose only contribution was deterministic fallback text.
"""

import sys

import pytest

from app.memory.sqlite import SQLiteMemory
from app.optimization import circuit_breaker, telemetry_cache
from app.providers.base import ProviderResponse
from app.schemas.trading_consult import TradingConsultRequest, TradingTelemetry
from app.services.trading_consult_service import TradingConsultService

FAILING_STAGES = {"trading_analysis", "strategic_comparison"}


class FakeGateway:
    """Stands in for model_gateway.execute with scripted availability."""

    def __init__(self, mode: str) -> None:
        self.mode = mode

    async def execute(self, provider_name, request, capability="general", stage_name="general"):
        if self.mode == "all_fail":
            raise RuntimeError(f"{provider_name} unreachable: connection refused")
        if self.mode == "partial_fail" and stage_name in FAILING_STAGES:
            raise RuntimeError(f"{provider_name} unreachable: connection refused")
        if self.mode == "empty_content":
            return ProviderResponse(content="", model="m", provider=provider_name)
        return ProviderResponse(
            content=f"Model analysis for {stage_name} from {provider_name}.",
            model="m",
            provider=provider_name,
        )


def make_request(*, total_trades: int = 50) -> TradingConsultRequest:
    """Unhealthy telemetry so synthesis reaches the RECOMMENDATION path."""
    return TradingConsultRequest(
        bot_id="bot_h6_test",
        trading_mode="PAPER",
        consultation_reason="DRAWDOWN_EVENT",
        telemetry=TradingTelemetry(
            equity=1000.0,
            unrealized_pnl=-25.0,
            realized_pnl=-180.0,
            win_rate=0.40,
            profit_factor=0.90,
            max_drawdown_pct=8.0,
            consecutive_losses=5,
            total_trades=total_trades,
        ),
    )


@pytest.fixture(autouse=True)
def _isolate_shared_state():
    """Circuit breaker and telemetry cache are process-global."""
    saved_counts = dict(circuit_breaker.failure_counts)
    saved_opened = dict(circuit_breaker.opened_at)
    saved_cache = dict(telemetry_cache.cache)
    circuit_breaker.failure_counts.clear()
    circuit_breaker.opened_at.clear()
    telemetry_cache.cache.clear()
    yield
    circuit_breaker.failure_counts.clear()
    circuit_breaker.failure_counts.update(saved_counts)
    circuit_breaker.opened_at.clear()
    circuit_breaker.opened_at.update(saved_opened)
    telemetry_cache.cache.clear()
    telemetry_cache.cache.update(saved_cache)


@pytest.fixture
def service(tmp_path):
    return TradingConsultService(memory=SQLiteMemory(db_path=str(tmp_path / "h6.db")))


async def _consult(service, monkeypatch, mode: str, req: TradingConsultRequest):
    # NOTE: app.services.__init__ does `from app.services.trading_consult_service
    # import trading_consult_service`, which rebinds the package attribute
    # `app.services.trading_consult_service` to the singleton instance. Resolve
    # the real module through sys.modules so we patch the global _invoke_agent
    # actually reads.
    module = sys.modules["app.services.trading_consult_service"]
    monkeypatch.setattr(module, "model_gateway", FakeGateway(mode))
    await service.memory.initialize()
    return await service.consult(req)


async def test_all_specialists_down_yields_degraded_honest_decision(service, monkeypatch):
    decision = await _consult(service, monkeypatch, "all_fail", make_request())

    assert decision.degraded is True, "a panel that produced no model output must be flagged"
    assert set(decision.agent_coverage.values()) == {"fallback"}
    assert decision.confidence == 0.45, (
        "zero model participation must not report the full-panel 0.88"
    )
    assert decision.degradation_reasons, "reasons must be populated when degraded"
    assert len(decision.degradation_reasons) == 4
    assert all(r.strip() for r in decision.degradation_reasons), "every reason must say something"
    assert any("unreachable" in r for r in decision.degradation_reasons), (
        "the provider error must be surfaced verbatim"
    )


async def test_degraded_debate_summary_does_not_claim_deliberation(service, monkeypatch):
    decision = await _consult(service, monkeypatch, "all_fail", make_request())

    assert "DEGRADED" in decision.debate_summary
    assert "[FALLBACK]" in decision.debate_summary, "fallback entries must be marked"
    assert "unanimously" not in decision.debate_summary
    assert "confirmed healthy metrics" not in decision.debate_summary


async def test_healthy_panel_keeps_full_confidence(service, monkeypatch):
    decision = await _consult(service, monkeypatch, "success", make_request())

    assert decision.degraded is False
    assert set(decision.agent_coverage.values()) == {"model"}
    assert decision.confidence == 0.88, "a fully model-backed panel keeps the baseline"
    assert "[FALLBACK]" not in decision.debate_summary
    assert "DEGRADED" not in decision.debate_summary


async def test_partial_participation_scales_confidence_between_bounds(service, monkeypatch):
    decision = await _consult(service, monkeypatch, "partial_fail", make_request())

    assert decision.degraded is True
    assert "model" in set(decision.agent_coverage.values())
    assert "fallback" in set(decision.agent_coverage.values())
    assert 0.45 < decision.confidence < 0.88, (
        f"2/4 participation should land between the bounds, got {decision.confidence}"
    )


async def test_run_records_never_claim_completed_on_failure(service, monkeypatch):
    await _consult(service, monkeypatch, "all_fail", make_request())

    async with service.memory.connect() as db:
        async with db.execute("SELECT status, error FROM runs") as cursor:
            rows = await cursor.fetchall()

    assert rows, "the panel should have recorded runs"
    for row in rows:
        assert row["status"] == "failed", (
            f"failed invocation recorded as status={row['status']!r}"
        )
        assert row["error"], "a failed run must carry the reason it failed"


async def test_empty_provider_content_is_not_recorded_as_completed_run(service, monkeypatch):
    await _consult(service, monkeypatch, "empty_content", make_request())

    async with service.memory.connect() as db:
        async with db.execute("SELECT status, error FROM runs") as cursor:
            rows = await cursor.fetchall()

    assert rows, "runs should have been recorded"
    for row in rows:
        assert row["status"] == "failed", "empty content is a failed model invocation"
        assert "empty content" in (row["error"] or "")


async def test_degraded_decision_is_not_cached(service, monkeypatch):
    await _consult(service, monkeypatch, "all_fail", make_request())
    assert telemetry_cache.cache == {}, (
        "degraded advice must not be served later as if it were fresh"
    )

    # Model breaker recovery so the healthy call is not itself blocked by the
    # outage simulated above (production recovers after recovery_time).
    circuit_breaker.failure_counts.clear()
    circuit_breaker.opened_at.clear()

    await _consult(service, monkeypatch, "success", make_request())
    assert telemetry_cache.cache, "a healthy decision should still be cached"


async def test_task_record_documents_specialist_coverage(service, monkeypatch):
    decision = await _consult(service, monkeypatch, "all_fail", make_request())

    async with service.memory.connect() as db:
        async with db.execute(
            "SELECT status, confidence, metadata_json FROM tasks WHERE mode = 'trading_consult'"
        ) as cursor:
            rows = await cursor.fetchall()

    assert rows, "a task record should have been persisted"
    import json

    last = rows[-1]
    raw = last["metadata_json"]
    metadata = json.loads(raw) if isinstance(raw, str) else raw
    assert metadata.get("model_backed_specialists") == 0
    assert metadata.get("degraded") is True
    assert metadata.get("specialist_coverage")
    assert last["confidence"] == decision.confidence


async def test_insufficient_data_path_discloses_missing_panel(service, monkeypatch):
    decision = await _consult(service, monkeypatch, "all_fail", make_request(total_trades=10))

    assert decision.status == "INSUFFICIENT_DATA"
    assert decision.degraded is True
    assert "no model output" in decision.debate_summary
    assert "no panel deliberation" in decision.debate_summary
    assert "unanimously" not in decision.debate_summary


async def test_open_circuit_substitutes_a_healthy_provider_instead_of_giving_up(
    service, monkeypatch
):
    """An open circuit must not mean "no answer" while a healthy provider sits idle.

    [FACT] Before this, ``_invoke_agent`` returned deterministic fallback text the moment the
    agent's preferred provider was unavailable, even with other providers configured, healthy
    and unused — while ``/v1/admin/routing/status`` advertised routing weights that should have
    made exactly this decision. The request is now re-routed by weight and the substitution is
    recorded, including on the run record.
    """
    from app.routing.self_optimizer import self_optimizing_router

    module = sys.modules["app.services.trading_consult_service"]
    monkeypatch.setattr(module, "model_gateway", FakeGateway("success"))
    monkeypatch.setattr(module.circuit_breaker, "is_available", lambda name: name != "groq")
    monkeypatch.setattr(module, "provider_has_credentials", lambda name: name == "gemini")

    await service.memory.initialize()
    await service.consult(make_request())

    async with service.memory.connect() as db:
        async with db.execute("SELECT provider, status, error FROM runs ORDER BY created_at") as cursor:
            rows = await cursor.fetchall()
    providers = {row["provider"] for row in rows}
    assert rows, "runs must be persisted"
    assert "groq" not in providers, "the unavailable provider was used anyway"
    assert "gemini" in providers, "the healthy substitution candidate was never tried"
    assert all(
        row["status"] == "completed" for row in rows if row["provider"] == "gemini"
    )
    # The substitution must be visible on the run record, so an operator can tell a substituted
    # success from a first-choice one.
    substituted = [row for row in rows if row["provider"] == "gemini"]
    assert substituted and all(
        (row["error"] is None or "routed to" in (row["error"] or "")) for row in substituted
    )
    assert self_optimizing_router.decisions, "the router must record that it decided this"


async def test_no_substitution_when_nothing_else_is_available(service, monkeypatch):
    """With every provider unavailable the honest outcome is still a degraded decision."""
    module = sys.modules["app.services.trading_consult_service"]
    monkeypatch.setattr(module, "model_gateway", FakeGateway("all_fail"))
    monkeypatch.setattr(module.circuit_breaker, "is_available", lambda name: False)
    monkeypatch.setattr(module, "provider_has_credentials", lambda name: False)

    await service.memory.initialize()
    decision = await service.consult(make_request())
    assert decision.degraded is True
    async with service.memory.connect() as db:
        async with db.execute("SELECT status, error FROM runs ORDER BY created_at") as cursor:
            rows = await cursor.fetchall()
    assert rows
    assert all(row["status"] == "failed" for row in rows)
    assert all("no other configured provider" in (row["error"] or "") for row in rows)


# ------------------------------------------------------------------------------------
# The per-bot rate limit: configurable, and it tells the client when to come back.
# ------------------------------------------------------------------------------------


def test_rate_limit_is_read_from_settings_not_a_frozen_constant(monkeypatch):
    """A hardcoded 20/hour could only be changed by editing the router.

    Measured in the mixed soak: the endpoint answered 92 x 429 in a 60 s round (every
    attempt, since the 12-shape workload revisits it at a fixed cadence) while the operator
    had no way to tune it and the response carried no Retry-After.
    """
    import app.routers.trading as trading_router
    from app.core.config import settings

    monkeypatch.setattr(settings, "TRADING_CONSULT_RATE_LIMIT_PER_HOUR", 7)
    assert trading_router.rate_limit_max_requests() == 7


def test_rate_limit_falls_back_safely_when_configuration_is_broken(monkeypatch):
    import app.routers.trading as trading_router
    from app.core.config import settings

    monkeypatch.setattr(settings, "TRADING_CONSULT_RATE_LIMIT_PER_HOUR", "not-a-number")
    assert trading_router.rate_limit_max_requests() == trading_router.DEFAULT_RATE_LIMIT_MAX_REQUESTS


def test_exceeding_the_limit_answers_429_with_retry_after(monkeypatch):
    import time

    import pytest as _pytest

    import app.routers.trading as trading_router
    from app.core.config import settings

    monkeypatch.setattr(settings, "TRADING_CONSULT_RATE_LIMIT_PER_HOUR", 2)
    bot = f"retry-after-bot-{time.time()}"
    trading_router._check_rate_limit(bot)
    trading_router._check_rate_limit(bot)

    with _pytest.raises(Exception) as caught:
        trading_router._check_rate_limit(bot)

    error = caught.value
    assert getattr(error, "status_code", None) == 429
    headers = getattr(error, "headers", {}) or {}
    retry_after = int(headers.get("Retry-After"))
    assert 1 <= retry_after <= 3601, f"Retry-After must be inside the window, got {retry_after}"
    assert "TRADING_CONSULT_RATE_LIMIT_PER_HOUR" in error.detail, (
        "the message must name the setting that changes this, so the operator is not stuck"
    )
