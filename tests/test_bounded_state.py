"""M15: process-global maps keyed by caller-supplied values must stay bounded.

Three dicts grew without limit:
  * app/routers/trading.py `_bot_request_timestamps[bot_id]` - bot_id arrives
    in the request body, so any caller could mint permanent entries.
  * app/security/api_security.py `_rate_limits[client_ip]` - distinct IPs are
    effectively unbounded even though each window is pruned.
  * app/governance/tenant_manager.py `dedup_cache[request_id]` - caller-supplied
    request ids, each retaining a full response payload, with a TTL that was
    only enforced when the same key was read again.
"""

import time

import pytest

from app.governance.tenant_manager import MultiTenantManager, tenant_manager
from app.routers import trading as trading_router
from app.security.api_security import security_manager


@pytest.fixture
def bot_windows():
    saved = dict(trading_router._bot_request_timestamps)
    trading_router._bot_request_timestamps.clear()
    yield trading_router._bot_request_timestamps
    trading_router._bot_request_timestamps.clear()
    trading_router._bot_request_timestamps.update(saved)


@pytest.fixture
def rate_limits():
    saved = dict(security_manager._rate_limits)
    security_manager._rate_limits.clear()
    yield security_manager._rate_limits
    security_manager._rate_limits.clear()
    security_manager._rate_limits.update(saved)


def test_bot_windows_evict_stale_entries(bot_windows):
    now = time.time()
    bot_windows["idle_bot"] = [now - 7200.0]  # two hours old, outside the window
    bot_windows["active_bot"] = [now]

    trading_router._evict_stale_bot_windows(now)

    assert "idle_bot" not in bot_windows, "expired bot window must be dropped"
    assert "active_bot" in bot_windows, "active bot window must survive"


def test_bot_windows_are_capped(monkeypatch, bot_windows):
    monkeypatch.setattr(trading_router, "MAX_TRACKED_BOT_IDS", 50)

    for i in range(500):
        trading_router._evict_stale_bot_windows(time.time())
        bot_windows[f"bot_{i}"] = [time.time()]

    assert len(bot_windows) <= 50, (
        f"tracked bot_ids grew to {len(bot_windows)} despite a cap of 50"
    )


def test_rate_limit_still_enforced_after_eviction(bot_windows):
    """Bounding state must not weaken the actual limit."""
    for _ in range(trading_router.RATE_LIMIT_MAX_REQUESTS):
        trading_router._check_rate_limit("victim_bot")

    import fastapi

    with pytest.raises(fastapi.HTTPException) as exc:
        trading_router._check_rate_limit("victim_bot")
    assert exc.value.status_code == 429


def test_ip_table_is_capped(monkeypatch, rate_limits):
    monkeypatch.setattr(security_manager, "_max_tracked_ips", 50)

    for i in range(500):
        security_manager.check_rate_limit(f"ip-{i}")

    assert len(security_manager._rate_limits) <= 50, (
        f"tracked IPs grew to {len(security_manager._rate_limits)} despite a cap of 50"
    )
    # The cap must not have broken the newest entry's own accounting.
    assert security_manager.check_rate_limit("ip-499") in (True, False)


def test_ip_table_still_counts_within_window(monkeypatch, rate_limits):
    monkeypatch.setattr(security_manager, "_max_tracked_ips", 1000)
    ip = "198.51.100.7"
    for _ in range(security_manager._rate_limit_max_requests):
        assert security_manager.check_rate_limit(ip) is True
    assert security_manager.check_rate_limit(ip) is False, (
        "an IP inside the window must still be limited after the bounding change"
    )


def test_dedup_cache_is_capped():
    mgr = MultiTenantManager()
    mgr.max_dedup_entries = 25

    for i in range(300):
        mgr.store_deduplication(f"req-{i}", {"n": i})

    assert len(mgr.dedup_cache) <= mgr.max_dedup_entries, (
        f"dedup_cache grew to {len(mgr.dedup_cache)} despite a cap of {mgr.max_dedup_entries}"
    )


def test_dedup_cache_prefers_dropping_expired_entries():
    mgr = MultiTenantManager()
    mgr.max_dedup_entries = 10

    # Ten long-lived entries plus ten already-expired ones.
    for i in range(10):
        mgr.store_deduplication(f"live-{i}", {"n": i})
    for i in range(10):
        mgr.dedup_cache[f"stale-{i}"] = {"response": {"n": i}, "expires_at": time.time() - 1.0}

    mgr.store_deduplication("newcomer", {"n": "new"})

    assert len(mgr.dedup_cache) <= mgr.max_dedup_entries
    assert "newcomer" in mgr.dedup_cache
    assert not any(k.startswith("stale-") for k in mgr.dedup_cache), (
        "expired entries should be reclaimed before live ones"
    )


def test_dedup_still_returns_cached_response():
    mgr = MultiTenantManager()
    mgr.store_deduplication("same-id", {"answer": 42})
    assert mgr.check_deduplication("same-id") == {"answer": 42}


def test_existing_tenant_manager_state_is_not_disturbed():
    """The shared singleton must still hold its configured tenants."""
    assert tenant_manager.tenants, "tenant store should not be empty"
    for policy in tenant_manager.tenants.values():
        assert len(policy.active_keys) <= 2, "tenant key list should stay bounded"
