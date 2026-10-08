"""The allocator high-water guard: traffic-aware ``malloc_trim`` for a long-lived process.

Measured need (``scripts/memory_forensics.py``, 1,600 ``/ask`` requests at concurrency 6,
tracemalloc off): live Python blocks returned to their baseline exactly — 403,456 at rest,
698,136 under load, 404,371 after 45 s idle — while RSS ended 1.3 MB higher. Nothing is
retained by the code; glibc simply keeps freed arenas mapped. The guard returns that memory
to the kernel, but *only* when the resident set is above its threshold and no request is in
flight, so these tests check the policy as strictly as the trim.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.main import app
from app.utils import memory_guard as guard_module
from app.utils.memory_guard import MemoryGuard, process_rss_mb, trim_allocator


class FakeTrim:
    """Records calls and pretends to reclaim ``reclaim_mb``."""

    def __init__(self, reclaim_mb: float = 12.5) -> None:
        self.calls = 0
        self.reclaim_mb = reclaim_mb

    def __call__(self) -> dict:
        self.calls += 1
        return {
            "supported": True,
            "before_mb": 900.0,
            "after_mb": 900.0 - self.reclaim_mb,
            "reclaimed_mb": self.reclaim_mb,
        }


@pytest.fixture
def frozen_rss(monkeypatch):
    """Pin the resident set size the policy sees; no real arenas involved."""

    state = {"mb": 512.0}
    monkeypatch.setattr(guard_module, "process_rss_mb", lambda: state["mb"])
    return state


# -- the platform primitives --------------------------------------------------------


def test_rss_is_reported_or_absent_but_never_a_guess():
    value = process_rss_mb()
    assert value is None or value > 0, f"RSS must be a real number or None, got {value!r}"


def test_trim_never_raises_and_reports_its_shape():
    result = trim_allocator()
    for key in ("supported", "before_mb", "after_mb", "reclaimed_mb"):
        assert key in result, result
    assert result["reclaimed_mb"] >= 0.0
    if not result["supported"]:
        # A platform without glibc must say so rather than pretend it trimmed.
        assert "detail" in result


# -- policy -------------------------------------------------------------------------


def test_guard_refuses_to_trim_while_a_request_is_in_flight(frozen_rss):
    guard = MemoryGuard(threshold_mb=100.0, cooldown_seconds=0.0)
    guard.request_started()
    allowed, reason = guard.should_trim()
    assert allowed is False
    assert "in flight" in reason


def test_guard_trims_when_idle_and_above_threshold(frozen_rss, monkeypatch):
    fake = FakeTrim()
    monkeypatch.setattr(guard_module, "trim_allocator", fake)
    guard = MemoryGuard(threshold_mb=100.0, cooldown_seconds=0.0)

    result = guard.maybe_trim()

    assert fake.calls == 1
    assert result is not None and result["reclaimed_mb"] == 12.5
    assert guard.describe()["trims"] == 1
    assert guard.describe()["reclaimed_total_mb"] == 12.5


def test_guard_leaves_a_small_resident_set_alone(frozen_rss, monkeypatch):
    fake = FakeTrim()
    monkeypatch.setattr(guard_module, "trim_allocator", fake)
    frozen_rss["mb"] = 50.0
    guard = MemoryGuard(threshold_mb=400.0, cooldown_seconds=0.0)

    assert guard.maybe_trim() is None
    assert fake.calls == 0


def test_cooldown_stops_a_trim_storm(frozen_rss, monkeypatch):
    fake = FakeTrim()
    monkeypatch.setattr(guard_module, "trim_allocator", fake)
    guard = MemoryGuard(threshold_mb=100.0, cooldown_seconds=60.0)

    assert guard.maybe_trim() is not None            # first trim runs
    assert guard.maybe_trim() is None                # immediately after: cooldown
    allowed, reason = guard.should_trim()
    assert allowed is False and "cooldown" in reason
    assert fake.calls == 1


def test_disabled_guard_never_trims(frozen_rss, monkeypatch):
    fake = FakeTrim()
    monkeypatch.setattr(guard_module, "trim_allocator", fake)
    guard = MemoryGuard(enabled=False, threshold_mb=1.0, cooldown_seconds=0.0)

    assert guard.maybe_trim() is None
    assert fake.calls == 0
    assert guard.should_trim()[1] == "disabled"


def test_completions_only_consider_a_trim_when_the_process_is_idle(frozen_rss, monkeypatch):
    """Two requests finish; the second one's completion is the first idle moment, not the first."""
    fake = FakeTrim()
    monkeypatch.setattr(guard_module, "trim_allocator", fake)
    monkeypatch.setattr(guard_module, "CHECK_EVERY", 1)
    guard = MemoryGuard(threshold_mb=100.0, cooldown_seconds=0.0)

    guard.request_started()
    guard.request_started()
    guard.request_finished()                 # one request still in flight
    assert fake.calls == 0
    guard.request_finished()                 # now idle
    assert fake.calls == 1


def test_describe_is_json_serialisable_state(frozen_rss):
    guard = MemoryGuard(threshold_mb=123.0, cooldown_seconds=5.0)
    described = guard.describe()
    assert described["threshold_mb"] == 123.0
    assert described["in_flight"] == 0
    assert described["last_trim"] is None


# -- the live surface ---------------------------------------------------------------


@pytest.fixture
def client(auth):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver", headers=auth)


async def test_trim_action_is_available_without_the_diagnostics_flag(client, monkeypatch):
    """Diagnostics are off by default, but reclaiming arenas is an operational right."""
    monkeypatch.setattr(settings, "ENABLE_MEMORY_DIAGNOSTICS", False)
    response = await client.get("/memory/diagnostics?action=trim")
    assert response.status_code == 200, response.text
    body = response.json()
    assert "supported" in body and "reclaimed_mb" in body
    assert body.get("reason") == "diagnostics endpoint"


async def test_snapshot_is_still_refused_without_the_flag(client, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_MEMORY_DIAGNOSTICS", False)
    response = await client.get("/memory/diagnostics?action=snapshot")
    assert response.status_code == 404


async def test_blocks_action_answers_whether_objects_are_retained(client, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_MEMORY_DIAGNOSTICS", True)
    response = await client.get("/memory/diagnostics?action=blocks")
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body["allocated_blocks"], int) and body["allocated_blocks"] > 0
    assert body["live_object_count"] > 0
    assert isinstance(body["live_object_types"], dict) and body["live_object_types"]


async def test_runtime_metrics_publish_the_guard(client):
    response = await client.get("/metrics/runtime")
    assert response.status_code == 200, response.text
    guard = response.json()["memory_guard"]
    assert guard["enabled"] is True
    assert guard["threshold_mb"] > 0
    assert guard["in_flight"] == 0
