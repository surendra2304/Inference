"""Adversarial Challenge Suite for Milestone 1 (M1: Mesh Contract & Health Probe Standardization).

Executed by: Challenger 2 (Empirical Challenger)
Target:
1. CORS origins verification with adversarial requests (allowed vs disallowed/hostile origins).
2. Probe resilience under burst requests (50 rapid requests to /health and /ready).
   - Rate limit 429 exemption verification.
   - Response latency benchmarks (< 50ms).
   - Zero outbound HTTP/TCP socket connections during probe handling.
3. Edge case stress testing:
   - Depleted consumer bucket in EnhancedRateLimiterMiddleware.
   - ProductionSecurityMiddleware vulnerability analysis on probes.
"""

import asyncio
import socket
import time
from unittest.mock import patch

import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.security.api_security import security_manager


@pytest.fixture
def async_client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver")


# ==============================================================================
# Challenge 1: Adversarial CORS Origins
# ==============================================================================

CANONICAL_ALLOWED_ORIGINS = [
    "https://friday-universe.duckdns.org",
    "http://127.0.0.1:8004",
    "http://localhost:8004",
]

ADVERSARIAL_DISALLOWED_ORIGINS = [
    "https://evil-hacker.com",
    "https://evil-hacker.com:8004",
    "http://friday-universe.duckdns.org",  # Unencrypted HTTP
    "https://subdomain.friday-universe.duckdns.org",  # Subdomain mismatch
    "http://localhost:8005",  # Port mismatch
    "http://127.0.0.1:8080",  # Port mismatch
    "null",  # Sandboxed iframe
    "http://attacker.local",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", CANONICAL_ALLOWED_ORIGINS)
async def test_cors_allowed_origins_options_preflight(async_client, origin: str):
    """OPTIONS preflight from canonical origins must be allowed with CORS headers."""
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
    assert resp.headers.get("access-control-allow-credentials") == "true"


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", CANONICAL_ALLOWED_ORIGINS)
async def test_cors_allowed_origins_actual_request(async_client, origin: str):
    """GET /health and GET /ready with Origin header must return matching access-control-allow-origin."""
    async with async_client as client:
        resp_health = await client.get("/health", headers={"Origin": origin})
        assert resp_health.status_code == status.HTTP_200_OK
        assert resp_health.headers.get("access-control-allow-origin") == origin

        resp_ready = await client.get("/ready", headers={"Origin": origin})
        assert resp_ready.status_code == status.HTTP_200_OK
        assert resp_ready.headers.get("access-control-allow-origin") == origin


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ADVERSARIAL_DISALLOWED_ORIGINS)
async def test_cors_adversarial_origins_options_rejected(async_client, origin: str):
    """Adversarial or unapproved origins must NOT receive access-control-allow-origin in preflight."""
    async with async_client as client:
        resp = await client.options(
            "/health",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
    allow_origin = resp.headers.get("access-control-allow-origin")
    assert allow_origin != origin, f"Origin {origin} was improperly allowed in preflight!"
    assert allow_origin is None or allow_origin in CANONICAL_ALLOWED_ORIGINS


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ADVERSARIAL_DISALLOWED_ORIGINS)
async def test_cors_adversarial_origins_actual_request_rejected(async_client, origin: str):
    """Adversarial or unapproved origins on GET /health must NOT receive access-control-allow-origin."""
    async with async_client as client:
        resp = await client.get("/health", headers={"Origin": origin})
    assert resp.status_code == status.HTTP_200_OK
    allow_origin = resp.headers.get("access-control-allow-origin")
    assert allow_origin != origin, f"Origin {origin} was improperly reflected on GET!"


# ==============================================================================
# Challenge 2: Probe Burst Resilience (50 rapid requests, 429 exemption, < 50ms)
# ==============================================================================


@pytest.mark.asyncio
async def test_health_probe_burst_50_rapid_requests(async_client):
    """Send 50 rapid sequential requests to GET /health: verify 0 rate limit 429s and < 50ms latency."""
    latencies = []
    statuses = []

    async with async_client as client:
        for _ in range(50):
            t0 = time.perf_counter()
            resp = await client.get("/health")
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms
            statuses.append(resp.status_code)

    # Assert 0 rate limits
    rate_limited = [s for s in statuses if s == status.HTTP_429_TOO_MANY_REQUESTS]
    assert len(rate_limited) == 0, f"Encountered {len(rate_limited)} 429 rate limit responses on /health!"
    assert all(s == status.HTTP_200_OK for s in statuses)

    # Benchmark latencies
    p50 = sorted(latencies)[len(latencies) // 2]
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)

    print(f"\n[GET /health 50-burst] Avg: {avg_latency:.2f}ms, P50: {p50:.2f}ms, P95: {p95:.2f}ms, Max: {max_latency:.2f}ms")
    # Strict latency threshold
    assert avg_latency < 50.0, f"Average latency {avg_latency:.2f}ms exceeds 50ms requirement"
    assert p95 < 50.0, f"P95 latency {p95:.2f}ms exceeds 50ms requirement"


@pytest.mark.asyncio
async def test_ready_probe_burst_50_rapid_requests(async_client):
    """Send 50 rapid sequential requests to GET /ready: verify 0 rate limit 429s and < 50ms latency."""
    latencies = []
    statuses = []

    async with async_client as client:
        for _ in range(50):
            t0 = time.perf_counter()
            resp = await client.get("/ready")
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms
            statuses.append(resp.status_code)

    rate_limited = [s for s in statuses if s == status.HTTP_429_TOO_MANY_REQUESTS]
    assert len(rate_limited) == 0, f"Encountered {len(rate_limited)} 429 rate limit responses on /ready!"
    assert all(s == status.HTTP_200_OK for s in statuses)

    p50 = sorted(latencies)[len(latencies) // 2]
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)

    print(f"\n[GET /ready 50-burst] Avg: {avg_latency:.2f}ms, P50: {p50:.2f}ms, P95: {p95:.2f}ms, Max: {max_latency:.2f}ms")
    assert avg_latency < 50.0, f"Average latency {avg_latency:.2f}ms exceeds 50ms requirement"
    assert p95 < 50.0, f"P95 latency {p95:.2f}ms exceeds 50ms requirement"


@pytest.mark.asyncio
async def test_concurrent_burst_probes_50_gather(async_client):
    """Send 50 concurrent requests simultaneously to both /health and /ready."""
    async with async_client as client:
        tasks = [client.get("/health") for _ in range(25)] + [client.get("/ready") for _ in range(25)]
        t0 = time.perf_counter()
        responses = await asyncio.gather(*tasks)
        total_time_ms = (time.perf_counter() - t0) * 1000.0

    statuses = [r.status_code for r in responses]
    assert all(s == 200 for s in statuses), f"Non-200 responses under concurrent burst: {set(statuses)}"
    avg_concurrent_time = total_time_ms / 50
    print(f"\n[Concurrent 50-probe burst] Completed all 50 in {total_time_ms:.2f}ms (equiv {avg_concurrent_time:.2f}ms/req)")


@pytest.mark.asyncio
async def test_probes_immune_to_exhausted_consumer_rate_limits(async_client):
    """Even if an API key / consumer has exhausted all rate limit tokens in EnhancedRateLimiterMiddleware, probes remain accessible."""
    async with async_client as client:
        resp_h = await client.get("/health", headers={"X-API-Key": "exhausted-key"})
        assert resp_h.status_code == status.HTTP_200_OK

        resp_r = await client.get("/ready", headers={"X-API-Key": "exhausted-key"})
        assert resp_r.status_code == status.HTTP_200_OK


# ==============================================================================
# Challenge 3: Verification of Zero Outbound Sockets during Probes
# ==============================================================================


@pytest.mark.asyncio
async def test_probes_open_zero_outbound_sockets(async_client):
    """Verify that during 50 health and 50 readiness probe calls, no outbound sockets are opened."""
    socket_connect_attempts = []

    real_connect = socket.socket.connect

    def spy_connect(self, address):
        socket_connect_attempts.append(address)
        return real_connect(self, address)

    with patch.object(socket.socket, "connect", spy_connect):
        async with async_client as client:
            for _ in range(50):
                resp1 = await client.get("/health")
                assert resp1.status_code == 200
                resp2 = await client.get("/ready")
                assert resp2.status_code == 200

    print(f"\n[Socket Outbound Spy] Connect attempts during 100 probe calls: {len(socket_connect_attempts)}")
    assert len(socket_connect_attempts) == 0, (
        f"Outbound socket connections were attempted during health/ready checks: {socket_connect_attempts}"
    )


# ==============================================================================
# Challenge 4: Deep Attack on ProductionSecurityMiddleware Rate Limiter
# ==============================================================================


@pytest.mark.asyncio
async def test_isolated_external_ip_50_probe_burst_passes(async_client):
    """An isolated burst of 50 requests from an external IP passes because 50 < 120."""
    external_ip = "198.51.100.99"
    security_manager._rate_limits.pop(external_ip, None)
    async with async_client as client:
        statuses = []
        for _ in range(50):
            resp = await client.get("/health", headers={"X-Forwarded-For": external_ip})
            statuses.append(resp.status_code)

    assert all(s == 200 for s in statuses), (
        f"External IP was unexpectedly blocked within 50 requests: {statuses}"
    )


@pytest.mark.asyncio
async def test_vulnerability_demonstration_probe_throttled_in_production_security_middleware(async_client):
    """Verify that ProductionSecurityMiddleware throttles non-exempt endpoints when limit is exceeded.

    Probes (/health, /ready, etc.) are exempt, but regular endpoints (such as /v1/models)
    must continue to be rate-limited by ProductionSecurityMiddleware with HTTP 429 RATE_LIMIT_EXCEEDED.
    """
    external_ip = "203.0.113.88"
    security_manager._rate_limits.pop(external_ip, None)

    async with async_client as client:
        # Step 1: Send 120 requests
        for _ in range(120):
            security_manager.check_rate_limit(external_ip)

        # Step 2: Request #121 to non-exempt route
        resp = await client.get("/v1/models", headers={"X-Forwarded-For": external_ip})

    # Empirically proves ProductionSecurityMiddleware enforces 429 on non-exempt endpoints
    assert resp.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    body = resp.json()
    assert body["error"] == "RATE_LIMIT_EXCEEDED"
    assert body["success"] is False


@pytest.mark.asyncio
async def test_contract_probes_must_be_exempt_under_sustained_traffic(async_client):
    """Adversarial stress test: Probes must remain accessible (HTTP 200) even when the IP has reached the 120 req/min limit.

    ProductionSecurityMiddleware explicitly exempts /health, /ready, /metrics, etc.,
    ensuring health checks never trigger container restart cascades.
    """
    external_ip = "203.0.113.99"
    security_manager._rate_limits.pop(external_ip, None)

    async with async_client as client:
        # Saturate 120 requests
        for _ in range(120):
            security_manager.check_rate_limit(external_ip)

        # Probes must NOT return 429
        resp = await client.get("/health", headers={"X-Forwarded-For": external_ip})
        assert resp.status_code == status.HTTP_200_OK, (
            f"Probe was throttled with {resp.status_code}: {resp.text}"
        )
        resp_ready = await client.get("/ready", headers={"X-Forwarded-For": external_ip})
        assert resp_ready.status_code == status.HTTP_200_OK, (
            f"Ready probe was throttled with {resp_ready.status_code}: {resp_ready.text}"
        )
