from __future__ import annotations

import asyncio
import math

import pytest

from app.inference_runtime.budget import BudgetLedger
from app.inference_runtime.cache import TenantCache
from app.inference_runtime.circuit import CircuitBreaker, CircuitState
from app.inference_runtime.dedupe import RequestDeduper


@pytest.mark.asyncio
async def test_cancelling_one_waiter_does_not_cancel_shared_work():
    deduper = RequestDeduper()
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def factory():
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return "shared result"

    first = asyncio.create_task(deduper.run("same-key", factory))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    second = asyncio.create_task(deduper.run("same-key", factory))

    for _ in range(100):
        entry = deduper._tasks.get("same-key")
        if entry is not None and entry.waiters == 2:
            break
        await asyncio.sleep(0)
    else:
        pytest.fail("the second caller never joined the in-flight operation")

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    entry = deduper._tasks.get("same-key")
    assert entry is not None
    assert not entry.task.cancelled()
    assert entry.waiters == 1

    release.set()
    assert await asyncio.wait_for(second, timeout=1.0) == "shared result"
    assert calls == 1
    assert deduper._tasks == {}


@pytest.mark.asyncio
async def test_last_cancelled_waiter_cancels_work_and_cleans_entry():
    deduper = RequestDeduper()
    started = asyncio.Event()
    saw_cancel = asyncio.Event()

    async def factory():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            saw_cancel.set()
            raise

    waiter = asyncio.create_task(deduper.run("orphan", factory))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    waiter.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiter

    await asyncio.wait_for(saw_cancel.wait(), timeout=1.0)
    assert deduper._tasks == {}


@pytest.mark.asyncio
async def test_deduper_removes_failed_task_before_retry():
    deduper = RequestDeduper()
    calls = 0

    async def fail_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient failure")
        return "recovered"

    with pytest.raises(RuntimeError, match="transient failure"):
        await deduper.run("retry-key", fail_once)
    assert deduper._tasks == {}

    assert await deduper.run("retry-key", fail_once) == "recovered"
    assert calls == 2
    assert deduper._tasks == {}


@pytest.mark.parametrize("amount", [-1.0, math.nan, math.inf, -math.inf, True, "1"])
def test_budget_reservation_rejects_negative_nonfinite_and_non_numeric_amounts(amount):
    ledger = BudgetLedger(default_budget=5.0)
    with pytest.raises(ValueError):
        ledger.reserve("tenant-a", amount)


@pytest.mark.parametrize("budget", [-1.0, math.nan, math.inf, -math.inf, True])
def test_budget_constructor_rejects_invalid_limits(budget):
    with pytest.raises(ValueError):
        BudgetLedger(default_budget=budget)


@pytest.mark.parametrize(
    ("estimated", "actual"),
    [
        (math.nan, 0.1),
        (math.inf, 0.1),
        (-1.0, 0.1),
        (0.1, math.nan),
        (0.1, math.inf),
        (0.1, -0.1),
        (0.1, True),
    ],
)
def test_budget_reconciliation_rejects_invalid_values_without_mutating_ledger(estimated, actual):
    ledger = BudgetLedger(default_budget=5.0)
    ledger.reserve("tenant-a", 1.0)
    remaining_before = ledger.remaining("tenant-a")

    with pytest.raises(ValueError):
        ledger.reconcile("tenant-a", estimated, actual)

    assert ledger.remaining("tenant-a") == pytest.approx(remaining_before)


@pytest.mark.parametrize("ttl", [math.nan, math.inf, -math.inf, True, "5"])
def test_cache_rejects_nonfinite_or_invalid_ttls(ttl):
    cache = TenantCache()
    with pytest.raises(ValueError):
        cache.put("key", "value", "tenant-a", ttl)
    assert cache.get("key") is None


def test_nonpositive_cache_ttl_evicts_existing_entry():
    cache = TenantCache()
    cache.put("key", "value", "tenant-a", 30.0)
    assert cache.get("key") == "value"

    cache.put("key", "ignored", "tenant-a", 0.0)
    assert cache.get("key") is None


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True])
def test_cache_rejects_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        TenantCache(max_items=capacity)


def test_circuit_breaker_admits_only_one_half_open_probe(monkeypatch):
    now = {"value": 100.0}
    monkeypatch.setattr("app.inference_runtime.circuit.time.monotonic", lambda: now["value"])
    breaker = CircuitBreaker(threshold=2, recovery_seconds=10.0)

    breaker.failure()
    breaker.failure()
    assert breaker.state is CircuitState.OPEN
    assert breaker.allow() is False

    now["value"] = 111.0
    assert breaker.allow() is True
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.allow() is False, "a second recovery probe must not run concurrently"

    breaker.success()
    assert breaker.state is CircuitState.CLOSED
    assert breaker.allow() is True


def test_failed_half_open_probe_reopens_breaker_and_restarts_recovery_window(monkeypatch):
    now = {"value": 100.0}
    monkeypatch.setattr("app.inference_runtime.circuit.time.monotonic", lambda: now["value"])
    breaker = CircuitBreaker(threshold=1, recovery_seconds=10.0)

    breaker.failure()
    now["value"] = 111.0
    assert breaker.allow() is True

    breaker.failure()
    assert breaker.state is CircuitState.OPEN
    assert breaker.opened_at == 111.0
    assert breaker.allow() is False

    now["value"] = 122.0
    assert breaker.allow() is True


def test_cancelled_half_open_probe_releases_permit(monkeypatch):
    now = {"value": 50.0}
    monkeypatch.setattr("app.inference_runtime.circuit.time.monotonic", lambda: now["value"])
    breaker = CircuitBreaker(threshold=1, recovery_seconds=5.0)
    breaker.failure()
    now["value"] = 56.0
    assert breaker.allow() is True
    assert breaker.allow() is False

    breaker.release_probe()
    assert breaker.state is CircuitState.OPEN
    assert breaker.allow() is True


@pytest.mark.parametrize("threshold", [0, -1, 1.5, True])
def test_circuit_breaker_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        CircuitBreaker(threshold=threshold)


@pytest.mark.parametrize("recovery", [-1.0, math.nan, math.inf, -math.inf, True])
def test_circuit_breaker_rejects_invalid_recovery_window(recovery):
    with pytest.raises(ValueError):
        CircuitBreaker(recovery_seconds=recovery)
