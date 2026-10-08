"""Whatever breaks inside must not be published to the caller.

Measured before the fix, on ``/v1/trading/consult``: a raised ``RuntimeError`` carrying
``"secret path /srv/private/db.sqlite: password=hunter2"`` came back verbatim in the 500 body.
Three sibling routes had the same shape, and the v1 ask path additionally publishes
*per-provider* failure reasons in ``failure_state`` — the text most likely to contain a
credential, because provider libraries routinely echo the key they rejected.

These tests inject a failure carrying a marked secret into each route and require that the
secret never appears in the response, that the status class is honest, and that a correlation
id *is* present so the failure can still be found in the logs. The helper itself is tested for
the same properties plus scrubbing, so a future caller cannot use it as a pass-through.
"""

from __future__ import annotations

import logging

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.utils.errors import internal_error, unavailable_detail

#: Deliberately full of the things a leak would expose: a path, a credential, a table name.
MARKER = "s3cr3t-marker-2b1c"
SECRET_TEXT = f"connection to /srv/private/db.sqlite failed for user admin password={MARKER}"


class Boom(RuntimeError):
    """A failure of the kind a provider or the database would raise."""


@pytest.fixture
def client(auth):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver",
                       headers=auth, timeout=60)


# -- the helper --------------------------------------------------------------------


def test_helper_returns_a_safe_detail_and_logs_the_cause(caplog):
    logger = logging.getLogger("test.errors")
    with caplog.at_level(logging.ERROR):
        detail, reference = internal_error(logger, Boom(SECRET_TEXT), doing_what="unit test")
    assert MARKER not in detail
    assert reference in detail
    assert reference.startswith("err_")
    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert "Boom" in joined, "the exception type must be findable in the log"


def test_helper_scrubs_credentials_out_of_the_log_too():
    """Defence in depth: the log is safer than a response, not a place to dump a key."""
    logger = logging.getLogger("test.errors.scrub")
    detail, reference = internal_error(
        logger, Boom("failed with api_key='AKIAIOSFODNN7EXAMPLE' and password=topsecret123"),
        doing_what="scrub check",
    )
    assert reference in detail
    assert "AKIAIOSFODNN7EXAMPLE" not in detail


def test_unavailable_detail_keeps_agent_ids_and_error_kinds_but_not_provider_text():
    exc = Boom("no specialist produced output")
    detail = unavailable_detail(
        exc,
        failures=[
            "researcher: RuntimeError: 401 from https://api.example.com (key=leak-me-please)",
            "synthesizer: TimeoutError: took too long",
        ],
        prefix="panel",
    )
    assert "researcher" in detail and "synthesizer" in detail
    assert "RuntimeError" in detail and "TimeoutError" in detail
    assert "leak-me-please" not in detail
    assert "api.example.com" not in detail


# -- the routes --------------------------------------------------------------------


class FakeDegradedResult:
    """The shape ``orchestrator.process_task`` returns when every provider failed."""

    degraded = True
    degradation_reasons = [
        f"groq: AuthenticationError: 401 from https://api.groq.com key=sk-{MARKER}",
        "gemini: ProviderUnconfiguredError: no credential configured",
    ]
    provider_used = "groq"
    models_used: list[str] = []
    total_tokens = 0
    unresolved_disagreements: list[str] = []
    agent_coverage: dict = {}
    failed_agents: list[str] = []
    mode_used = "review"
    run_id = "run_fake_1"
    deliberation_outcome = "multi_agent_council"
    key_evidence: list = []
    agents_used: list[str] = ["researcher"]
    failed_agents: list[str] = []
    answer = "degraded answer"
    confidence = 0.0
    uncertainty = 1.0
    evidence: list = []
    recommendations: list = []
    proposed_actions: list = []
    authorization_required = False
    reasoning_summary = "deliberation halted"
    latency_seconds = 0.01
    completed_at = None


async def test_v1_ask_scrubs_provider_error_text_out_of_failure_state(client, monkeypatch):
    """``failure_state`` is published per request; a rejected key must not travel with it."""
    import app.core.orchestrator as orchestrator_module

    async def fake_process_task(*_args, **_kwargs):
        return FakeDegradedResult()

    monkeypatch.setattr(orchestrator_module.orchestrator, "process_task", fake_process_task)
    response = await client.post("/v1/ask", json={"question": "does this leak?", "mode": "review"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert MARKER not in response.text, "the provider's rejected key reached the client"
    assert body.get("failure_state"), "the failure must still be reported, just safely"
    assert "ProviderUnconfiguredError" in body["failure_state"], (
        "the reason must stay useful: naming the failure kind is the point of the field"
    )


async def test_v1_ask_does_not_publish_an_internal_failure_as_a_stack_trace(client, monkeypatch):
    import app.core.orchestrator as orchestrator_module

    async def explode(*_args, **_kwargs):
        raise Boom(SECRET_TEXT)

    monkeypatch.setattr(orchestrator_module.orchestrator, "process_task", explode)
    response = await client.post("/v1/ask", json={"question": "does this leak?", "mode": "review"})

    assert MARKER not in response.text
    assert "/srv/private" not in response.text
    assert "Traceback" not in response.text


async def test_debate_does_not_publish_an_internal_failure(client, monkeypatch):
    import app.core.orchestrator as orchestrator_module

    async def explode(*_args, **_kwargs):
        raise Boom(SECRET_TEXT)

    monkeypatch.setattr(orchestrator_module.orchestrator, "process_task", explode)
    response = await client.post("/v1/debate", json={"question": "does this leak?",
                                                     "mode": "debate"})

    # The v1 debate contract answers 200 and reports the failure in band (``status=ERROR``,
    # ``failure_state``); what matters here is that the failure text is not published.
    assert MARKER not in response.text
    assert "/srv/private" not in response.text
    assert "Traceback" not in response.text
    body = response.json()
    if response.status_code == 200:
        assert body["status"] == "ERROR"
        assert "correlation id" in body["failure_state"]
        assert "Provider 'groq' has no configured credential" not in body["answer"], (
            "the per-specialist provider text must not be copied into the answer"
        )
    else:
        assert response.status_code >= 400


async def test_trading_consult_does_not_publish_an_internal_failure(client, monkeypatch):
    import app.routers.trading as trading_router
    from app.memory.sqlite import SQLiteMemory
    from app.services.trading_consult_service import TradingConsultService

    service = TradingConsultService(memory=SQLiteMemory(":memory:"))

    async def explode(_request):
        raise Boom(SECRET_TEXT)

    monkeypatch.setattr(service, "consult", explode)
    monkeypatch.setattr(trading_router, "trading_consult_service", service)

    response = await client.post(
        "/v1/trading/consult",
        json={
            "bot_id": "leak-bot", "trading_mode": "PAPER", "consultation_reason": "SCHEDULED",
            "telemetry": {"equity": 1000.0, "unrealized_pnl": 0.0, "realized_pnl": 0.0,
                          "win_rate": 0.5, "profit_factor": 1.0, "max_drawdown_pct": 1.0,
                          "consecutive_losses": 0, "total_trades": 10},
        },
    )

    assert response.status_code == 500, response.text
    assert MARKER not in response.text
    assert "correlation id" in response.json()["detail"]


# -- the scrubber itself: a provider echoes the key it rejected --------------------------


@pytest.mark.parametrize(
    "leaked",
    [
        "Invalid API key: sk-abcdefghijklmnop12345",
        "401 from gemini: AIzaSyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q",
        "groq rejected gsk_abcdefghijklmnopqrstuvwxyz01",
        "nvidia said nvapi-AbCdEfGhIjKlMnOpQrStUvWx",
        "openrouter: sk-or-v1-abcdefghijklmnopqrstuvwx",
        "cohere-cohere-abcdefghijklmnopqrstuv",
    ],
)
def test_provider_key_literals_are_redacted(leaked):
    """A 401 body quotes the key it rejected, without an ``api_key=`` label in front."""
    from app.security.prompt_isolation import scrub_credentials

    scrubbed = scrub_credentials(leaked)
    for token in ("sk-abcdefghijklmnop12345", "AIzaSyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q",
                  "gsk_abcdefghijklmnopqrstuvwxyz01", "nvapi-AbCdEfGhIjKlMnOpQrStUvWx",
                  "sk-or-v1-abcdefghijklmnopqrstuvwx", "abcdefghijklmnopqrstuv"):
        assert token not in scrubbed, f"{token!r} survived scrubbing of {leaked!r}"
    assert "REDACTED_CREDENTIAL" in scrubbed


@pytest.mark.parametrize(
    "benign",
    [
        "task-1234567890abcdef is not a key",
        "risk-assessment-of-the-write-path",
        "sk-learn model comparison",
        "The write path needs sharding by tenant.",
    ],
)
def test_benign_text_is_not_mangled_by_key_scrubbing(benign):
    """The historical failure mode of this module was over-redaction refusing real requests;
    the provider-prefix patterns keep a leading word boundary so ordinary prose survives."""
    from app.security.prompt_isolation import scrub_credentials

    assert scrub_credentials(benign) == benign


async def test_task_record_metadata_does_not_carry_raw_exception_text(client, monkeypatch):
    """``GET /tasks/{id}`` serves the stored metadata back, so the *write* must be clean too.

    ``ask_v1`` stores ``{"error": str(exc)}`` on the durable task record; ``get_task_status``
    returns ``task_record.model_dump()`` and the route publishes it. Measured in this test
    before the fix: a caller fetching the task by id received the credential marker that the
    original response had been scrubbed of.
    """
    import app.core.orchestrator as orchestrator_module

    calls: list[dict] = []

    async def record_outcome(task_id, question, result, **kwargs):
        calls.append({"task_id": task_id, "result": result, "metadata": kwargs.get("metadata")})

    monkeypatch.setattr(orchestrator_module.orchestrator, "_record_task_outcome",
                        record_outcome, raising=False)
    monkeypatch.setattr("app.api.v1_core_routes._record_task_outcome", record_outcome)

    async def explode(*_args, **_kwargs):
        raise Boom(SECRET_TEXT)

    monkeypatch.setattr(orchestrator_module.orchestrator, "process_task", explode)
    await client.post("/v1/ask", json={"question": "record a failure", "mode": "review"})

    assert calls, "the failure path must still record the task outcome"
    stored = calls[-1]["metadata"] or {}
    assert MARKER not in (stored.get("error") or ""), (
        f"raw exception text was stored on the task record: {stored.get('error')!r}"
    )
    assert calls[-1]["result"], "a result string is still recorded for the audit trail"


def test_reading_a_task_record_scrubs_metadata_keys_on_the_way_out():
    """A writer that forgets must not become a published credential."""
    from app.security.prompt_isolation import scrub_credentials_dict

    stored = {
        "trace_id": "trace_x",
        "error": "provider rejected api_key=AKIAIOSFODNN7EXAMPLE",
        "model": "should-survive",
    }
    clean = scrub_credentials_dict(stored)
    assert "AKIAIOSFODNN7EXAMPLE" not in clean["error"]
    assert clean["model"] == "should-survive"
    assert clean["trace_id"] == "trace_x"
