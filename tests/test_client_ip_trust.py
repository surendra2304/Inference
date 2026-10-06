"""H7: client IP resolution and the loopback rate-limit bypass.

Pre-fix behaviour locked down here:
  * Both middlewares keyed on `request.client.host`, so every caller behind a
    reverse proxy shared one bucket and received the proxy's address.
  * `X-Forwarded-For` was never consulted anywhere.
  * The development bypass triggered on `"pytest" in sys.modules`, which any
    process that happened to import pytest would satisfy, and on
    `APP_ENV in ("development", ...)` combined with a loopback peer - the exact
    shape of a local reverse proxy, i.e. rate limiting silently off for all.
  * An absent client defaulted to 127.0.0.1.
  * A malformed Content-Length raised out of the middleware as a 500.
"""

from fastapi import Response
from starlette.requests import Request

from app.core.config import settings
from app.core.security import resolve_client_ip
from app.middleware.rate_limiter import EnhancedRateLimiterMiddleware
from app.security.api_security import ProductionSecurityMiddleware, security_manager

PUBLIC_PEER = ("93.184.216.34", 40000)
LOOPBACK_PEER = ("127.0.0.1", 50000)


def make_request(path="/ask", headers=None, client=PUBLIC_PEER, method="GET") -> Request:
    hdrs = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "scheme": "http",
        "server": ("testserver", 80),
        "client": client,
        "headers": hdrs,
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


async def _ok_next(request):
    return Response(content="ok")


# ── resolve_client_ip ─────────────────────────────────────────────────────────

def test_public_peer_cannot_forge_x_forwarded_for():
    req = make_request(headers={"X-Forwarded-For": "1.2.3.4"}, client=PUBLIC_PEER)
    assert resolve_client_ip(req) == PUBLIC_PEER[0], (
        "a client on the public internet must not be able to pick its own bucket"
    )


def test_loopback_peer_honours_last_forwarded_hop():
    req = make_request(
        headers={"X-Forwarded-For": "203.0.113.9, 198.51.100.7"},
        client=LOOPBACK_PEER,
    )
    assert resolve_client_ip(req) == "198.51.100.7"


def test_loopback_peer_without_forwarding_returns_peer():
    req = make_request(client=LOOPBACK_PEER)
    assert resolve_client_ip(req) == "127.0.0.1"


def test_trust_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", False)
    req = make_request(headers={"X-Forwarded-For": "203.0.113.9"}, client=LOOPBACK_PEER)
    assert resolve_client_ip(req) == "127.0.0.1"


def test_explicit_trusted_proxy_cidr_is_honoured(monkeypatch):
    monkeypatch.setattr(settings, "TRUSTED_PROXY_CIDRS", ["10.0.0.0/8"])
    req = make_request(headers={"X-Forwarded-For": "203.0.113.9"}, client=("10.1.2.3", 50000))
    assert resolve_client_ip(req) == "203.0.113.9"

    # ...but an address outside the configured ranges still is not trusted.
    other = make_request(headers={"X-Forwarded-For": "203.0.113.9"}, client=("172.16.0.9", 50000))
    assert resolve_client_ip(other) == "172.16.0.9"


def test_missing_client_is_unknown_not_loopback():
    req = make_request()
    req.scope["client"] = None
    assert resolve_client_ip(req) == "unknown", (
        "an unidentified client must not inherit 127.0.0.1"
    )


def test_x_real_ip_used_only_behind_trusted_proxy():
    trusted = make_request(headers={"X-Real-IP": "203.0.113.44"}, client=LOOPBACK_PEER)
    assert resolve_client_ip(trusted) == "203.0.113.44"

    untrusted = make_request(headers={"X-Real-IP": "203.0.113.44"}, client=PUBLIC_PEER)
    assert resolve_client_ip(untrusted) == PUBLIC_PEER[0]


# ── EnhancedRateLimiterMiddleware ─────────────────────────────────────────────

async def test_rate_limiter_keys_on_peer_not_forged_header():
    mw = EnhancedRateLimiterMiddleware(app=None)
    req = make_request(path="/v1/ask", headers={"X-Forwarded-For": "1.2.3.4"}, client=PUBLIC_PEER)
    resp = await mw.dispatch(req, _ok_next)
    assert resp.status_code == 200
    assert mw._request_history, "the request should have been recorded"
    assert all(PUBLIC_PEER[0] in k for k in mw._request_history), (
        f"bucket keyed on a forged header: {list(mw._request_history)}"
    )
    assert not any("1.2.3.4" in k for k in mw._request_history)


async def test_rate_limiter_splits_callers_behind_a_local_proxy():
    mw = EnhancedRateLimiterMiddleware(app=None)
    for ip in ("203.0.113.10", "203.0.113.11", "203.0.113.12"):
        req = make_request(
            path="/v1/ask",
            headers={"X-Forwarded-For": f"198.51.100.1, {ip}"},
            client=LOOPBACK_PEER,
        )
        resp = await mw.dispatch(req, _ok_next)
        assert resp.status_code == 200

    keys = list(mw._request_history)
    assert len(keys) == 3, f"proxy case collapsed into {len(keys)} bucket(s): {keys}"
    for ip in ("203.0.113.10", "203.0.113.11", "203.0.113.12"):
        assert any(ip in k for k in keys), f"no bucket for {ip}"


async def test_dev_bypass_does_not_open_up_for_remote_peers(monkeypatch):
    """APP_ENV=development must not disable rate limiting for the internet."""
    monkeypatch.setattr(settings, "APP_ENV", "development")
    mw = EnhancedRateLimiterMiddleware(app=None)
    req = make_request(path="/v1/ask", client=PUBLIC_PEER)
    resp = await mw.dispatch(req, _ok_next)
    assert resp.status_code == 200
    assert mw._request_history, (
        "a remote peer must be rate limited even when APP_ENV=development"
    )
    # The bypass would have stamped the synthetic 10000 allowance header.
    assert resp.headers.get("X-RateLimit-Limit") != "10000"


async def test_loopback_peer_still_bypasses_in_explicit_dev_mode(monkeypatch):
    monkeypatch.setattr(settings, "APP_ENV", "development")
    mw = EnhancedRateLimiterMiddleware(app=None)
    req = make_request(path="/v1/ask", client=LOOPBACK_PEER)
    resp = await mw.dispatch(req, _ok_next)
    assert resp.headers.get("X-RateLimit-Limit") == "10000"
    assert mw._request_history == {}, "explicit local dev bypass should record nothing"


# ── ProductionSecurityMiddleware / APISecurityManager ─────────────────────────

def test_loopback_rate_limit_bypass_is_opt_in(monkeypatch):
    monkeypatch.setattr(settings, "APP_ENV", "production")
    monkeypatch.setattr(settings, "INSECURE_DEV_AUTH", False)
    monkeypatch.setattr(settings, "ALLOW_DEV_RATE_LIMIT_BYPASS", False)
    # The test runner marks this process as a test environment; drop that mark
    # so the assertion below observes genuine production settings.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    security_manager._rate_limits.clear()

    # Under production settings the loopback shortcut must not apply, so
    # traffic from 127.0.0.1 is counted like anyone else's.
    for _ in range(security_manager._rate_limit_max_requests):
        assert security_manager.check_rate_limit("127.0.0.1") is True
    assert security_manager.check_rate_limit("127.0.0.1") is False, (
        "loopback traffic should count against the limit in production"
    )

    security_manager._rate_limits.clear()
    monkeypatch.setattr(settings, "ALLOW_DEV_RATE_LIMIT_BYPASS", True)
    assert security_manager.check_rate_limit("127.0.0.1") is True
    security_manager._rate_limits.clear()


def test_testclient_identity_is_always_exempt():
    security_manager._rate_limits.clear()
    assert security_manager.check_rate_limit("testclient") is True


async def test_malformed_content_length_does_not_crash():
    mw = ProductionSecurityMiddleware(app=None)
    req = make_request(path="/v1/ask", headers={"Content-Length": "not-a-number"}, method="POST")
    resp = await mw.dispatch(req, _ok_next)
    assert resp.status_code == 413, (
        f"malformed length should be rejected as oversized, got {resp.status_code}"
    )


async def test_oversized_payload_is_rejected():
    mw = ProductionSecurityMiddleware(app=None)
    req = make_request(path="/v1/ask", headers={"Content-Length": str(3 * 1024 * 1024)}, method="POST")
    resp = await mw.dispatch(req, _ok_next)
    assert resp.status_code == 413


async def test_middleware_resolves_real_client_for_rate_key(monkeypatch):
    monkeypatch.setattr(settings, "APP_ENV", "production")
    security_manager._rate_limits.clear()
    mw = ProductionSecurityMiddleware(app=None)
    req = make_request(
        path="/v1/ask",
        headers={"X-Forwarded-For": "203.0.113.77"},
        client=PUBLIC_PEER,
    )
    await mw.dispatch(req, _ok_next)
    assert PUBLIC_PEER[0] in security_manager._rate_limits, (
        f"expected the peer to be tracked, got {list(security_manager._rate_limits)}"
    )
    assert not any("203.0.113.77" in k for k in security_manager._rate_limits)
    security_manager._rate_limits.clear()
