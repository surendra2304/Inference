"""Empirical Adversarial Challenge Harness for Milestone 1.

Milestone 1: Mesh Contract & Health Probe Standardization
Audits and stress-tests:
1. Malformed JSON payloads (syntax errors, invalid types, non-finite floats, empty body) -> HTTP 422 envelope.
2. Missing, empty, or invalid Bearer tokens on protected routes -> HTTP 401 / 403 envelopes.
3. Non-existent routes and subpaths -> HTTP 404 envelope.
4. Simulated unhandled exceptions in handlers and dependencies -> HTTP 500 envelope with zero traceback leak.
5. Simulated database and gateway failures on GET & HEAD /ready -> HTTP 503 {"status": "not_ready", "ready": false}.
6. Strict compliance of all error envelopes to:
   {"error": "<error_code>", "message": "<msg>", "detail": "<detail>", "success": false}
"""

import sqlite3
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.core.orchestrator import orchestrator
from app.main import app
from app.providers.gateway import model_gateway
from app.security.api_security import security_manager


@pytest.fixture
def auth_headers() -> dict[str, str]:
    key = settings.get_friday_api_key() or "test-m1-key"
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def async_client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver")


def assert_mesh_error_envelope(
    data: dict,
    expected_error: str | None = None,
    expected_message_substring: str | None = None,
) -> None:
    """Strictly assert that a response dict conforms to the mesh error envelope contract."""
    assert isinstance(data, dict), f"Response body is not a dict: {type(data)}"
    assert "error" in data, f"Missing 'error' field in response: {data}"
    assert isinstance(data["error"], str) and len(data["error"]) > 0, (
        f"'error' must be a non-empty string: {data['error']}"
    )
    if expected_error:
        assert data["error"] == expected_error, (
            f"Expected error '{expected_error}', got '{data['error']}'"
        )

    assert "message" in data, f"Missing 'message' field in response: {data}"
    assert isinstance(data["message"], str) and len(data["message"]) > 0, (
        f"'message' must be a non-empty string: {data['message']}"
    )
    if expected_message_substring:
        assert expected_message_substring.lower() in data["message"].lower(), (
            f"Expected '{expected_message_substring}' in message: '{data['message']}'"
        )

    assert "detail" in data, f"Missing 'detail' field in response: {data}"
    assert "success" in data, f"Missing 'success' field in response: {data}"
    assert data["success"] is False, f"'success' must be False, got {data['success']}"


# ==============================================================================
# Suite 1: Malformed JSON Payloads (Triggering 422)
# ==============================================================================


@pytest.mark.asyncio
async def test_adversarial_malformed_json_syntax_error(async_client, auth_headers):
    """Sending broken JSON syntax with application/json header must yield 422 envelope."""
    async with async_client as client:
        resp = await client.post(
            "/stream",
            headers={**auth_headers, "Content-Type": "application/json"},
            content=b'{"prompt": "hello", "max_tokens": ',  # incomplete JSON
        )

    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    body = resp.json()
    assert_mesh_error_envelope(body, expected_error="VALIDATION_ERROR")
    assert resp.headers.get("X-Correlation-ID")


@pytest.mark.asyncio
async def test_adversarial_malformed_json_trailing_garbage(async_client, auth_headers):
    """Sending valid JSON with trailing garbage must yield 422 envelope."""
    async with async_client as client:
        resp = await client.post(
            "/v1/debate",
            headers={**auth_headers, "Content-Type": "application/json"},
            content=b'{"question": "test"} trailing garbage',
        )

    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    body = resp.json()
    assert_mesh_error_envelope(body, expected_error="VALIDATION_ERROR")


@pytest.mark.asyncio
async def test_adversarial_malformed_json_raw_binary(async_client, auth_headers):
    """Sending non-utf8 random binary bytes as JSON must yield 422 envelope without crashing."""
    async with async_client as client:
        resp = await client.post(
            "/stream",
            headers={**auth_headers, "Content-Type": "application/json"},
            content=b"\x00\xff\xfe\xfd\x80\x81\x82\x83",
        )

    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    body = resp.json()
    assert_mesh_error_envelope(body, expected_error="VALIDATION_ERROR")


@pytest.mark.asyncio
async def test_adversarial_malformed_json_wrong_type(async_client, auth_headers):
    """Sending non-integer string in integer field must yield 422 envelope."""
    async with async_client as client:
        resp = await client.post(
            "/stream",
            headers=auth_headers,
            json={"prompt": "test", "max_tokens": "not_an_int"},
        )

    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    body = resp.json()
    assert_mesh_error_envelope(body, expected_error="VALIDATION_ERROR", expected_message_substring="validation")
    assert isinstance(body["detail"], list)


@pytest.mark.asyncio
async def test_adversarial_malformed_json_non_finite_float(async_client, auth_headers):
    """Sending non-finite float (1e309 = Inf) must be cleanly converted to 422, not 500."""
    async with async_client as client:
        resp = await client.post(
            "/stream",
            headers={**auth_headers, "Content-Type": "application/json"},
            content=b'{"prompt": "test", "max_tokens": 1e309}',
        )

    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    body = resp.json()
    assert_mesh_error_envelope(body, expected_error="VALIDATION_ERROR")


@pytest.mark.asyncio
async def test_adversarial_malformed_json_empty_body(async_client, auth_headers):
    """Sending empty body with Content-Type: application/json to route requiring body yields 422."""
    async with async_client as client:
        resp = await client.post(
            "/stream",
            headers={**auth_headers, "Content-Type": "application/json"},
            content=b"",
        )

    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    body = resp.json()
    assert_mesh_error_envelope(body, expected_error="VALIDATION_ERROR")


@pytest.mark.asyncio
async def test_adversarial_malformed_json_array_instead_of_object(async_client, auth_headers):
    """Sending JSON array when route expects model object yields 422."""
    async with async_client as client:
        resp = await client.post(
            "/stream",
            headers=auth_headers,
            json=[1, 2, 3, "unexpected array"],
        )

    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    body = resp.json()
    assert_mesh_error_envelope(body, expected_error="VALIDATION_ERROR")


# ==============================================================================
# Suite 2: Missing or Invalid Bearer Tokens (Triggering 401 & 403)
# ==============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("POST", "/v1/debate", {"question": "Hello world"}),
        ("POST", "/stream", {"prompt": "Hello world"}),
        ("POST", "/v1/task/execute", {}),
        ("POST", "/v1/trading/consult", {}),
        ("GET", "/v1/governance/tenants/tenant_trading", None),
        ("GET", "/models", None),
        ("POST", "/v1/forge/generate-code", {}),
    ],
)
async def test_adversarial_missing_token_triggers_401_on_protected_routes(
    async_client, method: str, path: str, payload: dict | None
):
    """Completely missing credentials must return HTTP 401 with standard envelope across routes."""
    async with async_client as client:
        kwargs = {}
        if payload is not None:
            kwargs["json"] = payload
        resp = await client.request(method, path, headers={}, **kwargs)

    assert resp.status_code == status.HTTP_401_UNAUTHORIZED
    body = resp.json()
    assert_mesh_error_envelope(body, expected_error="UNAUTHORIZED", expected_message_substring="missing API key")
    assert resp.headers.get("X-Correlation-ID")


@pytest.mark.asyncio
async def test_adversarial_empty_bearer_token_triggers_401(async_client):
    """Authorization header with empty Bearer token must trigger 401."""
    async with async_client as client:
        resp = await client.post(
            "/v1/debate",
            headers={"Authorization": "Bearer "},
            json={"question": "Test"},
        )

    assert resp.status_code == status.HTTP_401_UNAUTHORIZED
    body = resp.json()
    assert_mesh_error_envelope(body, expected_error="UNAUTHORIZED", expected_message_substring="missing API key")


@pytest.mark.asyncio
async def test_adversarial_whitespace_bearer_token_triggers_401(async_client):
    """Authorization header with whitespace-only Bearer token must trigger 401."""
    async with async_client as client:
        resp = await client.post(
            "/v1/debate",
            headers={"Authorization": "Bearer    "},
            json={"question": "Test"},
        )

    assert resp.status_code == status.HTTP_401_UNAUTHORIZED
    body = resp.json()
    assert_mesh_error_envelope(body, expected_error="UNAUTHORIZED", expected_message_substring="missing API key")


@pytest.mark.asyncio
async def test_adversarial_invalid_bearer_token_triggers_403(async_client):
    """Authorization header with wrong token value must trigger 403 Forbidden with standard envelope."""
    async with async_client as client:
        resp = await client.post(
            "/v1/debate",
            headers={"Authorization": "Bearer wrong-token-value-adversarial-probe-12345"},
            json={"question": "Test"},
        )

    assert resp.status_code == status.HTTP_403_FORBIDDEN
    body = resp.json()
    assert_mesh_error_envelope(body, expected_error="FORBIDDEN", expected_message_substring="invalid API key")
    assert resp.headers.get("X-Correlation-ID")


@pytest.mark.asyncio
async def test_adversarial_invalid_alternative_headers_trigger_403(async_client):
    """Presenting invalid credentials via X-INFERENCE-API-KEY or X-FRIDAY-API-Key must trigger 403 envelope."""
    async with async_client as client:
        resp1 = await client.post(
            "/v1/debate",
            headers={"X-INFERENCE-API-KEY": "bad_inference_key"},
            json={"question": "Test"},
        )
        resp2 = await client.post(
            "/v1/debate",
            headers={"X-FRIDAY-API-Key": "bad_friday_key"},
            json={"question": "Test"},
        )

    assert resp1.status_code == status.HTTP_403_FORBIDDEN
    assert_mesh_error_envelope(resp1.json(), expected_error="FORBIDDEN")
    assert resp2.status_code == status.HTTP_403_FORBIDDEN
    assert_mesh_error_envelope(resp2.json(), expected_error="FORBIDDEN")


# ==============================================================================
# Suite 3: Non-Existent Routes (Triggering 404)
# ==============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/this-route-does-not-exist-at-all"),
        ("POST", "/v1/nonexistent/subresource"),
        ("DELETE", "/api/v1/invalid/deep/nested"),
        ("GET", "/health/unknown_subpath"),
    ],
)
async def test_adversarial_non_existent_route_triggers_404(async_client, method: str, path: str):
    """Calling non-existent routes must return HTTP 404 with standardized error envelope."""
    async with async_client as client:
        resp = await client.request(method, path)

    assert resp.status_code == status.HTTP_404_NOT_FOUND
    body = resp.json()
    assert_mesh_error_envelope(body, expected_error="NOT_FOUND", expected_message_substring="not found")
    assert body["detail"] == "Not Found"
    assert resp.headers.get("X-Correlation-ID")


# ==============================================================================
# Suite 4: Simulated Unhandled Exception (Triggering 500)
# ==============================================================================


@pytest.mark.asyncio
async def test_adversarial_unhandled_exception_triggers_500_without_leak(async_client):
    """Simulated unhandled exception in route must yield 500 envelope without leaking exception text."""
    import app.main as main_mod

    secret_trace = "SUPER_SECRET_INTERNAL_DB_PASSWORD_XYZ123"

    def fatal_crash():
        raise RuntimeError(f"Database crash with secret {secret_trace}")

    with patch.object(main_mod, "get_dashboard_html", fatal_crash):
        async with async_client as client:
            resp = await client.get("/ui")

    assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    body = resp.json()
    assert_mesh_error_envelope(
        body,
        expected_error="INTERNAL_SERVER_ERROR",
        expected_message_substring="internal server error",
    )
    # Ensure sensitive traceback text was NOT leaked
    assert secret_trace not in resp.text
    assert resp.headers.get("X-Correlation-ID")


# ==============================================================================
# Suite 5: Simulated Database Failure on GET & HEAD /ready (Triggering 503)
# ==============================================================================


@pytest.mark.asyncio
async def test_adversarial_ready_probe_db_connect_failure_returns_503(async_client):
    """GET /ready must return HTTP 503 with {'status': 'not_ready', 'ready': false} when connect fails."""
    @asynccontextmanager
    async def broken_connect():
        if True:
            raise sqlite3.OperationalError("Simulated SQLite lock error: database is locked")
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
    assert "database is locked" in body["database"]["detail"]


@pytest.mark.asyncio
async def test_adversarial_ready_probe_db_query_failure_returns_503(async_client):
    """GET /ready must return HTTP 503 when db execute raises an exception during SELECT 1."""
    mock_db = MagicMock()
    mock_db.execute.side_effect = sqlite3.DatabaseError("Corrupt database disk image")

    @asynccontextmanager
    async def broken_query_connect():
        yield mock_db

    with patch.object(orchestrator.memory, "connect", broken_query_connect):
        async with async_client as client:
            resp = await client.get("/ready")

    assert resp.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["ready"] is False
    assert body["database"]["connected"] is False


@pytest.mark.asyncio
async def test_adversarial_ready_probe_head_method_under_failure_returns_503(async_client):
    """HEAD /ready under database failure must return HTTP 503 with empty body."""
    @asynccontextmanager
    async def broken_connect():
        if True:
            raise RuntimeError("Database unreachable")
        yield None

    with patch.object(orchestrator.memory, "connect", broken_connect):
        async with async_client as client:
            resp = await client.head("/ready")

    assert resp.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert resp.text == ""


@pytest.mark.asyncio
async def test_adversarial_ready_probe_routing_failure_returns_503(async_client):
    """GET /ready must return HTTP 503 when model_gateway key pools are empty."""
    with patch.object(model_gateway, "key_pools", {}):
        async with async_client as client:
            resp = await client.get("/ready")

    assert resp.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["ready"] is False
    assert body["model_routing"]["status"] == "unready"
    assert body["model_routing"]["initialized"] is False


# ==============================================================================
# Suite 6: Middleware Error Surfaces (429 Rate Limiting & 413 Payload Too Large)
# ==============================================================================


@pytest.mark.asyncio
async def test_adversarial_rate_limiter_middleware_envelope(async_client):
    """Rate limit exhaustion in ProductionSecurityMiddleware returns standard 429 envelope."""
    with patch.object(security_manager, "check_rate_limit", return_value=False):
        async with async_client as client:
            resp = await client.get("/v1/governance/tenants/tenant_trading")

    assert resp.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    body = resp.json()
    assert_mesh_error_envelope(body, expected_error="RATE_LIMIT_EXCEEDED")


@pytest.mark.asyncio
async def test_adversarial_oversized_payload_middleware_envelope(async_client):
    """Payload exceeding 2MB returns standard 413 envelope."""
    async with async_client as client:
        resp = await client.post(
            "/stream",
            headers={"Content-Length": str(5 * 1024 * 1024)},
            content=b"test",
        )

    assert resp.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    body = resp.json()
    assert_mesh_error_envelope(body, expected_error="PAYLOAD_TOO_LARGE")


@pytest.mark.asyncio
async def test_adversarial_malformed_content_length_treated_as_oversized(async_client):
    """Malformed non-integer Content-Length must not raise 500, but reject as 413."""
    async with async_client as client:
        resp = await client.post(
            "/stream",
            headers={"Content-Length": "not-an-integer-header"},
            content=b"test",
        )

    assert resp.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    body = resp.json()
    assert_mesh_error_envelope(body, expected_error="PAYLOAD_TOO_LARGE")


# ==============================================================================
# Suite 7: Additional Edge Cases (CORS on errors, Correlation ID, Custom Detail)
# ==============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("status_target_path", ["/v1/debate", "/nonexistent", "/stream"])
async def test_adversarial_cors_headers_preserved_on_error_responses(async_client, status_target_path: str):
    """CORS headers must be preserved on 401, 404, and 422 error responses."""
    origin = "https://friday-universe.duckdns.org"
    async with async_client as client:
        resp = await client.post(
            status_target_path,
            headers={"Origin": origin, "Content-Type": "application/json"},
            content=b'{"invalid_json":',
        )

    assert resp.status_code in (
        status.HTTP_401_UNAUTHORIZED,
        status.HTTP_404_NOT_FOUND,
        status.HTTP_422_UNPROCESSABLE_ENTITY,
    )
    assert resp.headers.get("access-control-allow-origin") == origin
    assert_mesh_error_envelope(resp.json())


@pytest.mark.asyncio
async def test_adversarial_correlation_id_propagation(async_client):
    """Client-provided X-Correlation-ID must be echoed in response headers and envelope body."""
    custom_cid = "adversarial-cid-999-testing"
    async with async_client as client:
        resp = await client.get("/nonexistent-route", headers={"X-Correlation-ID": custom_cid})

    assert resp.status_code == status.HTTP_404_NOT_FOUND
    body = resp.json()
    assert_mesh_error_envelope(body, expected_error="NOT_FOUND")
    assert resp.headers.get("X-Correlation-ID") == custom_cid
    assert body.get("correlation_id") == custom_cid


@pytest.mark.asyncio
async def test_adversarial_server_unconfigured_key_triggers_503(async_client):
    """When server has no keys configured and INSECURE_DEV_AUTH is False, returns 503 envelope."""
    with patch("app.core.security._configured_keys", return_value=[]), \
         patch.object(settings, "INSECURE_DEV_AUTH", False):
        async with async_client as client:
            resp = await client.post(
                "/v1/debate",
                headers={"Authorization": "Bearer some-key"},
                json={"question": "Test"},
            )

    assert resp.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    body = resp.json()
    assert_mesh_error_envelope(
        body,
        expected_error="SERVICE_UNAVAILABLE",
        expected_message_substring="INFERENCE_API_KEY",
    )

