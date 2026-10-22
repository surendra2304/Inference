"""Tests for Milestone 1: Mesh Contract & Health Probe Standardization.

Covers:
1. GET & HEAD /health liveness probe (200, evidence_class, providers).
2. GET & HEAD /ready readiness probe:
   - 200 when healthy with structured component statuses (database, model_routing).
   - 503 when SQLite connection or model routing is degraded.
   - Zero outbound network calls made.
   - Public unauthenticated access.
3. Standard error envelopes across HTTP statuses (401, 404, 422, 429, 500, 413):
   {"error": "<error_code>", "message": "<detail>", "detail": "<detail>", "success": false}
4. CORS and Port standardization:
   - CORS headers for friday-universe.duckdns.org, 127.0.0.1:8004, localhost variants.
   - Settings.CORS_ALLOWED_ORIGINS field_validator parses comma-separated strings and preserves canonical union.
   - Entrypoint uvicorn runner honors PORT and HOST defaults.
"""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings, settings
from app.health import _active_specialist_count
from app.main import app
from app.providers.gateway import model_gateway


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Return valid auth headers using the server's configured key."""
    key = settings.get_friday_api_key() or "test-m1-key"
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def async_client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver")


# ==============================================================================
# 1. Liveness Probe (GET & HEAD /health)
# ==============================================================================


@pytest.mark.asyncio
async def test_health_liveness_probe_returns_200(async_client):
    """GET /health must return HTTP 200 with process liveness and active providers."""
    async with async_client as client:
        resp = await client.get("/health")

    assert resp.status_code == status.HTTP_200_OK
    body = resp.json()
    assert body["status"] in ("healthy", "ok")
    assert body["service"] in ("inference", "inference-api")
    assert body["evidence_class"] == "process_liveness"
    assert "observed_at" in body
    assert "version" in body
    assert isinstance(body.get("providers"), list)
    for p in ("groq", "cerebras", "openrouter", "gemini"):
        assert p in body["providers"]
    assert body["active_specialist_agents"] == _active_specialist_count()


@pytest.mark.asyncio
async def test_health_head_method_supported(async_client):
    """HEAD /health must return HTTP 200 with empty body."""
    async with async_client as client:
        resp = await client.head("/health")

    assert resp.status_code == status.HTTP_200_OK
    assert resp.text == ""


@pytest.mark.asyncio
async def test_health_unauthenticated_access(async_client):
    """GET /health must be accessible without any Authorization header."""
    async with async_client as client:
        resp = await client.get("/health", headers={})
    assert resp.status_code == status.HTTP_200_OK


# ==============================================================================
# 2. Readiness Probe (GET & HEAD /ready)
# ==============================================================================


@pytest.mark.asyncio
async def test_ready_probe_healthy_returns_200(async_client):
    """GET /ready returns HTTP 200 with structured component statuses when healthy."""
    async with async_client as client:
        resp = await client.get("/ready")

    assert resp.status_code == status.HTTP_200_OK
    body = resp.json()
    assert body["status"] == "ready"
    assert body["ready"] is True
    assert body["service"] == "inference"
    assert body["evidence_class"] == "service_readiness"
    assert "observed_at" in body
    assert "version" in body

    # Structured component status: database
    assert isinstance(body["database"], dict)
    assert body["database"]["status"] == "connected"
    assert body["database"]["connected"] is True
    assert "journal_mode" in body["database"]

    # Structured component status: model_routing
    assert isinstance(body["model_routing"], dict)
    assert body["model_routing"]["status"] == "ready"
    assert body["model_routing"]["initialized"] is True
    assert isinstance(body["model_routing"]["providers"], list)
    assert len(body["model_routing"]["providers"]) > 0


@pytest.mark.asyncio
async def test_ready_head_method_supported(async_client):
    """HEAD /ready returns HTTP 200 with empty body."""
    async with async_client as client:
        resp = await client.head("/ready")
    assert resp.status_code == status.HTTP_200_OK
    assert resp.text == ""


@pytest.mark.asyncio
async def test_ready_unauthenticated_access(async_client):
    """GET /ready must be open to mesh health scrapers without credentials."""
    async with async_client as client:
        resp = await client.get("/ready", headers={})
    assert resp.status_code in (status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE)
    assert "status" in resp.json()


@pytest.mark.asyncio
async def test_ready_probe_simulated_db_failure_returns_503(async_client):
    """GET /ready returns HTTP 503 when SQLite database connection fails."""
    from app.core.orchestrator import orchestrator

    @asynccontextmanager
    async def broken_connect():
        if True:
            raise RuntimeError("Simulated SQLite disk I/O lock failure")
        yield None

    with patch.object(orchestrator.memory, "connect", broken_connect):
        async with async_client as client:
            resp = await client.get("/ready")

    assert resp.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["ready"] is False
    assert body["database"]["status"] == "unready"
    assert body["database"]["connected"] is False
    assert "connection_failed" in body["database"]["detail"]


@pytest.mark.asyncio
async def test_ready_probe_simulated_model_routing_unready_returns_503(async_client):
    """GET /ready returns HTTP 503 when model gateway key pools are uninitialized."""
    with patch.object(model_gateway, "key_pools", {}):
        async with async_client as client:
            resp = await client.get("/ready")

    assert resp.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["ready"] is False
    assert body["model_routing"]["status"] == "unready"
    assert body["model_routing"]["initialized"] is False


@pytest.mark.asyncio
async def test_ready_probe_does_not_call_external_llm_apis(async_client):
    """GET /ready must inspect local state only and never invoke external LLM APIs."""
    mock_execute = MagicMock()
    with patch.object(type(model_gateway), "execute", mock_execute):
        async with async_client as client:
            resp = await client.get("/ready")
        assert resp.status_code == status.HTTP_200_OK
        mock_execute.assert_not_called()


# ==============================================================================
# 3. Standardized Error Envelopes (401, 404, 422, 429, 500, 413)
# ==============================================================================


def _assert_valid_envelope(
    body: dict, expected_error: str, expected_message_substring: str | None = None
) -> None:
    """Helper asserting strict conformance to the standard error envelope contract."""
    assert "error" in body, f"Envelope missing 'error': {body}"
    assert body["error"] == expected_error
    assert "message" in body, f"Envelope missing 'message': {body}"
    assert isinstance(body["message"], str) and len(body["message"]) > 0
    if expected_message_substring:
        assert expected_message_substring.lower() in body["message"].lower()
    assert "detail" in body, f"Envelope missing 'detail': {body}"
    assert "success" in body, f"Envelope missing 'success': {body}"
    assert body["success"] is False


@pytest.mark.asyncio
async def test_error_envelope_on_401_unauthorized(async_client):
    """401 Unauthorized returns standard envelope {"error": "UNAUTHORIZED", ...}."""
    async with async_client as client:
        # Calling protected endpoint without any credentials returns 401
        resp = await client.post(
            "/v1/debate",
            headers={},
            json={"question": "Test question"},
        )

    assert resp.status_code == status.HTTP_401_UNAUTHORIZED
    _assert_valid_envelope(resp.json(), "UNAUTHORIZED", "missing API key")
    assert resp.headers.get("X-Correlation-ID")


@pytest.mark.asyncio
async def test_error_envelope_on_403_forbidden(async_client):
    """403 Forbidden returns standard envelope {"error": "FORBIDDEN", ...}."""
    async with async_client as client:
        # Calling protected endpoint with an invalid key returns 403
        resp = await client.post(
            "/v1/debate",
            headers={"Authorization": "Bearer invalid-wrong-key-999"},
            json={"question": "Test question"},
        )

    assert resp.status_code == status.HTTP_403_FORBIDDEN
    _assert_valid_envelope(resp.json(), "FORBIDDEN", "invalid API key")
    assert resp.headers.get("X-Correlation-ID")


@pytest.mark.asyncio
async def test_error_envelope_on_404_not_found(async_client):
    """404 Not Found returns standard envelope {"error": "NOT_FOUND", "detail": "Not Found", ...}."""
    async with async_client as client:
        resp = await client.get("/v1/this-endpoint-does-not-exist-at-all")

    assert resp.status_code == status.HTTP_404_NOT_FOUND
    body = resp.json()
    _assert_valid_envelope(body, "NOT_FOUND", "Not Found")
    # Must preserve Starlette router-level "Not Found" for backward compatibility
    assert body["detail"] == "Not Found"
    assert resp.headers.get("X-Correlation-ID")


@pytest.mark.asyncio
async def test_error_envelope_on_422_validation_error(async_client, auth_headers):
    """422 Unprocessable Entity returns standard envelope {"error": "VALIDATION_ERROR", ...}."""
    async with async_client as client:
        # Send non-integer max_tokens to /stream
        resp = await client.post(
            "/stream",
            headers=auth_headers,
            json={"max_tokens": "definitely-not-an-integer"},
        )

    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    body = resp.json()
    _assert_valid_envelope(body, "VALIDATION_ERROR", "validation")
    # Detail must retain structured validation errors list
    assert isinstance(body["detail"], list)
    assert resp.headers.get("X-Correlation-ID")


@pytest.mark.asyncio
async def test_error_envelope_on_429_rate_limited(async_client):
    """429 Too Many Requests returns standard envelope {"error": "RATE_LIMIT_EXCEEDED", ...}."""
    from app.security.api_security import security_manager

    with patch.object(security_manager, "check_rate_limit", return_value=False):
        async with async_client as client:
            resp = await client.get("/v1/governance/tenants/tenant_trading")

    assert resp.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    body = resp.json()
    _assert_valid_envelope(body, "RATE_LIMIT_EXCEEDED", "too many requests")


@pytest.mark.asyncio
async def test_error_envelope_on_500_internal_error(async_client):
    """500 Internal Server Error returns standard envelope {"error": "INTERNAL_SERVER_ERROR", ...}."""
    import app.main as main_mod

    # Simulate an unexpected unhandled exception inside a route handler
    def explosion():
        raise RuntimeError("Simulated unhandled critical failure")

    with patch.object(main_mod, "get_dashboard_html", explosion):
        async with async_client as client:
            resp = await client.get("/ui")

    assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    body = resp.json()
    _assert_valid_envelope(body, "INTERNAL_SERVER_ERROR", "internal server error")
    assert resp.headers.get("X-Correlation-ID")
    # Verify no raw exception text was leaked to client
    assert "Simulated unhandled critical failure" not in resp.text


@pytest.mark.asyncio
async def test_error_envelope_on_413_payload_too_large(async_client):
    """413 Payload Too Large returns standard envelope {"error": "PAYLOAD_TOO_LARGE", ...}."""
    async with async_client as client:
        resp = await client.post(
            "/stream",
            headers={"Content-Length": str(3 * 1024 * 1024)},  # 3MB > 2MB limit
            content=b"x" * 10,
        )

    assert resp.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    body = resp.json()
    _assert_valid_envelope(body, "PAYLOAD_TOO_LARGE", "2mb limit")


# ==============================================================================
# 4. Port and CORS Standardization
# ==============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "origin",
    [
        "https://friday-universe.duckdns.org",
        "http://127.0.0.1:8004",
        "http://localhost:8004",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
)
async def test_cors_preflight_allows_canonical_origins(async_client, origin: str):
    """CORS preflight OPTIONS requests must allow canonical mesh and localhost origins."""
    async with async_client as client:
        resp = await client.options(
            "/health",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )

    assert resp.status_code == status.HTTP_200_OK
    assert resp.headers.get("access-control-allow-origin") == origin


@pytest.mark.asyncio
async def test_cors_rejects_unauthorized_external_origin(async_client):
    """CORS preflight OPTIONS must not reflect unapproved external domains."""
    async with async_client as client:
        resp = await client.options(
            "/health",
            headers={
                "Origin": "https://malicious-attacker.com",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert resp.headers.get("access-control-allow-origin") != "https://malicious-attacker.com"


def test_cors_settings_field_validator_parses_comma_separated_strings():
    """Settings.CORS_ALLOWED_ORIGINS field_validator must parse comma-separated strings."""
    raw_csv = "https://custom1.domain.com, http://custom2.local:9000"
    s = Settings(CORS_ALLOWED_ORIGINS=raw_csv)
    origins = s.CORS_ALLOWED_ORIGINS

    assert "https://custom1.domain.com" in origins
    assert "http://custom2.local:9000" in origins
    # Mandatory canonical origins must be preserved via union
    assert "https://friday-universe.duckdns.org" in origins
    assert "http://127.0.0.1:8004" in origins
    assert "http://localhost:8004" in origins


def test_cors_settings_field_validator_parses_json_list():
    """Settings.CORS_ALLOWED_ORIGINS field_validator must parse JSON array strings."""
    json_str = '["https://json-domain.org", "http://internal-mesh:8004"]'
    s = Settings(CORS_ALLOWED_ORIGINS=json_str)
    origins = s.CORS_ALLOWED_ORIGINS

    assert "https://json-domain.org" in origins
    assert "http://internal-mesh:8004" in origins
    assert "https://friday-universe.duckdns.org" in origins
    assert "http://127.0.0.1:8004" in origins


def test_app_main_entrypoint_reads_port_and_host_defaults():
    """Verify app/main.py entrypoint defines PORT fallback 8000 and HOST default 0.0.0.0."""
    import inspect

    import app.main as main_module

    source = inspect.getsource(main_module)
    assert 'if __name__ == "__main__":' in source
    assert 'os.environ.get("PORT"' in source or 'os.environ.get("PORT",' in source
    assert 'os.environ.get("HOST"' in source or 'os.environ.get("HOST",' in source
    assert "8000" in source
    assert "0.0.0.0" in source
