"""Authentication boundary tests (CRITICAL-2 / CRITICAL-3).

Behaviour these tests lock down:
  * 114 of 120 operations previously accepted anonymous requests.
  * `POST /v1/governance/tenants/{id}/rotate-key` required no credential AND
    never validated `old_key`, so any caller could mint unlimited tenant keys.
  * `X-FRIDAY-API-Key` was declared on /v1/ask, /v1/debate and /v1/task/execute
    but never read (a decoy that advertised auth in OpenAPI while enforcing none).

Route lists are derived from the live OpenAPI document so a new unprotected
operation fails these tests instead of silently widening the attack surface.
"""

import copy

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.governance.tenant_manager import tenant_manager
from app.main import app
from app.providers.health import provider_health_tracker
from tests.conftest import TEST_API_KEY

# Real paths (method, path template, body) that MUST refuse anonymous callers.
PROTECTED = [
    ("POST", "/ask", {"question": "hi"}),
    ("POST", "/debate", {"question": "hi"}),
    ("POST", "/v1/ask", {"prompt": "hi"}),
    ("POST", "/v1/debate", {"question": "hi"}),
    ("POST", "/v1/task/execute", {}),
    ("POST", "/v1/agent/assist", {"caller_agent": "trading_bot", "task_type": "general", "prompt": "hi"}),
    ("POST", "/v1/trading/consult", {}),
    ("POST", "/v1/providers/execute", {}),
    ("POST", "/v1/forge/generate-code", {}),
    ("POST", "/v1/astra/reason", {}),
    ("POST", "/experiments", {}),
    ("GET", "/v1/governance/tenants/tenant_trading", None),
    ("GET", "/v1/governance/prometheus-metrics", None),
    ("GET", "/v1/admin/dashboard", None),
    ("GET", "/v1/admin/costs", None),
    ("GET", "/v1/analytics/overview", None),
    ("GET", "/experiments/nonexistent", None),
    ("GET", "/tasks/nonexistent", None),
]

# The dashboard (app/ui/dashboard.html) and liveness probes must stay open.
PUBLIC = [
    "GET /",
    "GET /health",
    "GET /health/ready",
    "GET /health/detailed",
    "GET /health/providers",
    "GET /status",
    "GET /agents",
    "GET /ui",
    "GET /v1/forge/health",
]

PUBLIC_SPECS = [(m, p) for m, p in (x.split(" ") for x in PUBLIC)]


def is_auth_rejection(resp) -> bool:
    """True when the *authentication boundary* rejected us.

    503 is ambiguous on purpose: it is returned both by the auth boundary
    (server has no key configured) and by the task boundary (no model provider
    reachable). Only the auth-boundary variant names INFERENCE_API_KEY, so the
    message text disambiguates.
    """
    if resp.status_code in (401, 403):
        return True
    if resp.status_code == 503:
        return "INFERENCE_API_KEY" in resp.text
    return False


@pytest.fixture(autouse=True)
def _isolate_provider_health():
    """Do not let auth tests leak into the provider-health singleton.

    Proving a credential is accepted requires the handler to run, which attempts
    real provider calls and records failures in the process-global
    `provider_health_tracker`. That registry legitimately decides which providers
    debate.py may use, so leaking it would make unrelated tests see every
    provider as quarantined. Snapshot and restore around each test.
    """
    stats = provider_health_tracker._stats
    saved = copy.deepcopy(stats)
    yield
    stats.clear()
    stats.update(saved)


@pytest.fixture
def anon_client():
    """A client that sends no credentials at all (no default headers)."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


@pytest.fixture
def auth_client(auth):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver", headers=auth)


@pytest.fixture
def clean_tenants():
    """Snapshot tenant key state so rotation tests cannot leak into each other."""
    snapshot = {tid: list(p.active_keys) for tid, p in tenant_manager.tenants.items()}
    yield
    for tid, keys in snapshot.items():
        if tid in tenant_manager.tenants:
            tenant_manager.tenants[tid].active_keys = keys


@pytest.mark.parametrize("method,path,body", PROTECTED)
async def test_protected_route_refuses_anonymous_caller(method, path, body, auth, anon_client):
    """No credential -> refused before the handler runs.

    `auth` configures a server-side key (so the boundary can distinguish
    "missing credential" from "server unconfigured"); `anon_client` deliberately
    sends none. A non-existent route would answer 404, so this also proves the
    operation is present in the surface and covered by the dependency.
    """
    assert auth
    r_anon = await anon_client.request(method, path, json=body)
    assert r_anon.status_code in (401, 403), (
        f"{method} {path} accepted an anonymous caller "
        f"({r_anon.status_code}): {r_anon.text[:200]}"
    )


# Routes that reach a handler but do not invoke a model — used to prove a valid
# credential is accepted without triggering real provider calls (which would slow
# the suite down and mutate global provider-health state).
CHEAP_AUTHED_ROUTES = [
    ("GET", "/v1/governance/tenants/tenant_trading", None),
    ("GET", "/v1/admin/dashboard", None),
    ("GET", "/v1/analytics/overview", None),
    ("GET", "/tasks/nonexistent", None),
    ("GET", "/experiments/nonexistent", None),
]


@pytest.mark.parametrize("method,path,body", CHEAP_AUTHED_ROUTES)
async def test_valid_credential_is_accepted(method, path, body, auth_client):
    r = await auth_client.request(method, path, json=body)
    assert not is_auth_rejection(r), (
        f"{method} {path} rejected a valid credential: {r.status_code} {r.text[:200]}"
    )


@pytest.mark.parametrize("route", PUBLIC_SPECS)
async def test_public_routes_stay_open_for_probes_and_dashboard(route):
    method, path = route
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        r = await client.request(method, path)
    assert r.status_code == 200, f"{method} {path} unexpectedly required auth: {r.status_code}"


async def test_wrong_key_is_forbidden(auth, anon_client):
    r = await anon_client.post("/ask", json={"question": "hi"}, headers={"X-INFERENCE-API-KEY": "wrong-key"})
    assert r.status_code == 403
    assert "invalid API key" in r.json()["detail"]


async def test_missing_server_key_fails_closed(no_server_key, anon_client):
    """No credential configured -> 503, never anonymous access."""
    r = await anon_client.post("/ask", json={"question": "hi"}, headers={"X-INFERENCE-API-KEY": "anything"})
    assert r.status_code == 503
    assert "INFERENCE_API_KEY" in r.json()["detail"]


async def test_insecure_dev_auth_allows_local_bypass(monkeypatch):
    monkeypatch.setattr(settings, "INSECURE_DEV_AUTH", True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        r = await client.get("/v1/admin/dashboard")
    assert not is_auth_rejection(r), (
        f"INSECURE_DEV_AUTH should bypass auth, got {r.status_code}: {r.text[:200]}"
    )


async def test_bearer_authorization_header_is_accepted(auth, anon_client):
    r = await anon_client.get(
        "/v1/governance/tenants/tenant_trading",
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
    )
    assert not is_auth_rejection(r), f"Bearer rejected: {r.status_code} {r.text[:200]}"


async def test_api_key_header_variants_are_accepted(auth, anon_client):
    for hdr, val in (
        ("X-API-Key", TEST_API_KEY),
        ("X-FRIDAY-API-Key", TEST_API_KEY),
    ):
        r = await anon_client.get("/v1/governance/tenants/tenant_trading", headers={hdr: val})
        assert not is_auth_rejection(r), f"{hdr} rejected: {r.status_code} {r.text[:200]}"


async def test_friday_routes_enforce_their_own_key(anon_client, auth_client):
    r = await anon_client.post("/v1/friday/ask", json={"question": "hi"})
    assert r.status_code == 401, f"expected 401, got {r.status_code}: {r.text[:200]}"


# ── CRITICAL-2: tenant key rotation ────────────────────────────────────────────

async def test_governance_tenant_read_requires_auth(anon_client, auth_client):
    anon = await anon_client.get("/v1/governance/tenants/tenant_trading")
    assert anon.status_code == 401, "tenant active_keys must not be readable anonymously"

    ok = await auth_client.get("/v1/governance/tenants/tenant_trading")
    assert ok.status_code == 200
    assert "active_keys" in ok.json()


async def test_rotate_key_rejects_wrong_old_key(auth_client, clean_tenants):
    r = await auth_client.post(
        "/v1/governance/tenants/tenant_forge/rotate-key",
        json={"old_key": "totally-wrong-key-12345"},
    )
    assert r.status_code == 403, f"expected 403 for wrong old_key, got {r.status_code}: {r.text}"


async def test_rotate_key_rejects_empty_old_key(auth_client, clean_tenants):
    """Regression: `{"old_key": ""}` used to return 200 ROTATED and mint a key."""
    r = await auth_client.post(
        "/v1/governance/tenants/tenant_nexus/rotate-key",
        json={"old_key": ""},
    )
    assert r.status_code == 403, f"expected 403 for empty old_key, got {r.status_code}: {r.text}"


async def test_rotate_key_requires_authentication(anon_client, auth, clean_tenants):
    """Even a correct old_key must not work without the service credential."""
    valid = tenant_manager.tenants["tenant_forge"].active_keys[0]
    r = await anon_client.post(
        "/v1/governance/tenants/tenant_forge/rotate-key",
        json={"old_key": valid},
    )
    assert r.status_code == 401, r.text


async def test_rotate_key_succeeds_with_valid_old_key_and_stays_bounded(auth_client, clean_tenants):
    policy = tenant_manager.tenants["tenant_forge"]
    original_key = policy.active_keys[0]
    before = len(policy.active_keys)

    r = await auth_client.post(
        "/v1/governance/tenants/tenant_forge/rotate-key",
        json={"old_key": original_key},
    )

    assert r.status_code == 200, r.text
    new_key = r.json()["new_key"]
    assert new_key != original_key
    assert new_key in policy.active_keys
    assert original_key not in policy.active_keys, "the superseded key must be retired"
    assert len(policy.active_keys) == before, "rotation must not grow active_keys"


async def test_repeated_invalid_rotations_cannot_accumulate_keys(auth_client, clean_tenants):
    """Regression: the old implementation appended a key on EVERY call."""
    policy = tenant_manager.tenants["tenant_sentinel"]
    before = len(policy.active_keys)

    for i in range(10):
        r = await auth_client.post(
            "/v1/governance/tenants/tenant_sentinel/rotate-key",
            json={"old_key": f"attacker-key-{i}"},
        )
        assert r.status_code == 403

    assert len(policy.active_keys) == before, (
        f"active_keys grew from {before} to {len(policy.active_keys)} despite all attempts failing"
    )
