"""Operational and telemetry endpoints for Inference runtime."""

import os
from typing import Any

from fastapi import APIRouter, HTTPException

from app.providers.gateway import model_gateway
from app.providers.health import provider_health_tracker
from app.utils.bounded_store import audit_bounds
from app.utils.memory_guard import memory_guard
from app.version import __version__

operational_router = APIRouter(tags=["Operational"])

#: Previous tracemalloc snapshot, used to answer "what grew between snapshots?" — the only
#: question that separates a cache sitting at its ceiling (big but flat) from a real leak
#: (small but growing). Only populated by the opt-in diagnostics endpoint.
_last_snapshot: Any = None


@operational_router.get("/health/providers")
async def get_providers_health() -> dict[str, Any]:
    """Returns runtime health status and latency snapshots for all registered providers without credentials."""
    snapshots = provider_health_tracker.get_all_health()
    active_keys = {
        prov: {
            "total_keys": pool.total_keys_count,
            "active_keys": pool.get_active_keys_count(),
            "quarantined_keys": pool.get_quarantined_keys_count(),
        }
        for prov, pool in model_gateway.key_pools.items()
    }
    return {
        "status": "healthy",
        "version": __version__,
        "providers": snapshots,
        "key_pools": active_keys,
    }


@operational_router.get("/models")
async def list_models() -> dict[str, Any]:
    """Returns available model registry capabilities and supported configurations."""
    all_models = [
        {"provider": "gemini", "model": "gemini-3.8-flash", "capabilities": ["chat", "stream", "json", "tools", "vision", "thinking"], "context_window": 2097152},
        {"provider": "gemini", "model": "gemini-3.7-flash", "capabilities": ["chat", "stream", "json", "tools", "vision", "thinking"], "context_window": 2097152},
        {"provider": "gemini", "model": "gemini-3.6-flash", "capabilities": ["chat", "stream", "json", "tools", "vision"], "context_window": 2097152},
        {"provider": "gemini", "model": "gemini-3.5-flash", "capabilities": ["chat", "stream", "json", "tools", "vision"], "context_window": 1048576},
        {"provider": "groq", "model": "openai/gpt-oss-120b", "capabilities": ["chat", "stream", "json", "tools"], "context_window": 131072},
        {"provider": "groq", "model": "qwen/qwen3.8-27b", "capabilities": ["chat", "stream", "json", "tools"], "context_window": 131072},
        {"provider": "groq", "model": "openai/gpt-oss-20b", "capabilities": ["chat", "stream", "json", "tools"], "context_window": 131072},
        {"provider": "nvidia", "model": "nvidia/nemotron-3-super-120b-a12b", "capabilities": ["chat", "stream", "json"], "context_window": 131072},
        {"provider": "nvidia", "model": "nvidia/nemotron-3-ultra-550b-a55b", "capabilities": ["chat", "stream", "json"], "context_window": 131072},
        {"provider": "nvidia", "model": "z-ai/glm-5.3", "capabilities": ["chat", "stream", "json"], "context_window": 131072},
        {"provider": "openrouter", "model": "liquid/lfm-2.5-2.6b:free", "capabilities": ["chat", "stream"], "context_window": 32768},
        {"provider": "openrouter", "model": "qwen/qwen3.8-27b:free", "capabilities": ["chat", "stream", "json"], "context_window": 131072},
        {"provider": "openrouter", "model": "nvidia/nemotron-3.5-lightning:free", "capabilities": ["chat", "stream", "json", "tools"], "context_window": 131072},
        {"provider": "vllm", "model": "local-model", "capabilities": ["chat", "stream", "json", "tools"], "context_window": 32768},
        {"provider": "sglang", "model": "local-sglang-model", "capabilities": ["chat", "stream", "json"], "context_window": 32768},
        {"provider": "llamacpp", "model": "local-llama", "capabilities": ["chat", "stream", "json"], "context_window": 8192},
    ]
    return {
        "models": all_models,
        "count": len(all_models),
    }


def _process_resources() -> dict[str, Any]:
    """RSS / open FDs / thread count of this process, when the OS exposes them.

    Served next to the retention audit below because the two questions arrive together:
    "memory is high — is something unbounded, or is it a cache at its ceiling?" The
    audit answers the first, this answers the second.
    """
    info: dict[str, Any] = {"rss_mb": None, "open_fds": None, "threads": None}
    try:
        with open("/proc/self/status") as handle:
            status = handle.read()
        info["rss_mb"] = round(int(status.split("VmRSS:")[1].split()[0]) / 1024.0, 1)
        info["threads"] = int(status.split("Threads:")[1].split()[0])
        info["open_fds"] = len(os.listdir("/proc/self/fd"))
    except Exception:  # non-Linux or restricted /proc: report nothing rather than guess
        pass
    return info


@operational_router.get("/memory/diagnostics")
async def memory_diagnostics(action: str = "snapshot", limit: int = 12) -> dict[str, Any]:
    """Allocation attribution for leak hunting, off unless explicitly enabled.

    Answering "the process grows, which line allocates?" needs tracemalloc running *inside*
    the server; measuring from a client process attributes the client's own growth instead
    (measured: a probe driving 4,000 requests grew itself by 1.8 MB while the server grew
    14 MB, and an in-process ASGI probe attributed everything to its own driver frames).

    Disabled by default: ``ENABLE_MEMORY_DIAGNOSTICS=true`` is required, because a snapshot
    exposes internal file paths and line numbers. ``action`` is ``start``, ``stop`` or
    ``snapshot`` (start/diff workflow: start, drive traffic, snapshot).
    """
    import gc
    import sys
    import tracemalloc

    from app.core.config import settings

    global _last_snapshot

    # A trim is available regardless of the diagnostics flag: it exposes no file paths, it is
    # the operator-facing cure for allocator high-water, and hiding it behind a debug switch
    # would mean a production container cannot reclaim its own arenas.
    if action.lower() == "trim":
        return memory_guard.force_trim("diagnostics endpoint")

    if not getattr(settings, "ENABLE_MEMORY_DIAGNOSTICS", False):
        raise HTTPException(
            status_code=404,
            detail=(
                "Memory diagnostics are disabled. Set ENABLE_MEMORY_DIAGNOSTICS=true to "
                "enable them; the endpoint reports internal allocation sites."
            ),
        )

    action = action.lower()
    if action in ("start", "reset"):
        # reset gives a clean attribution of "what the workload that follows allocates and
        # keeps", instead of a table dominated by start-up imports.
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        gc.collect()
        _last_snapshot = None
        tracemalloc.start(30)
        return {"status": "started", "tracing": True, "reset": action == "reset"}
    if action == "stop":
        tracemalloc.stop()
        return {"status": "stopped", "tracing": False}
    if action == "blocks":
        # Attribution that does *not* need tracemalloc: count the live Python blocks and, if
        # they are growing, which types hold them. Measured need: a run with tracing off grew
        # anonymous heap by ~90 KB/request while the traced delta was nil, so the next question
        # is whether the retention is Python objects at all (blocks grow) or C-level memory
        # (blocks flat, e.g. sqlite3/malloc arenas) — comparing two snapshots answers it.
        gc.collect()
        counters: dict[str, int] = {}
        for obj in gc.get_objects():
            name = type(obj).__name__
            counters[name] = counters.get(name, 0) + 1
        return {
            "tracing": tracemalloc.is_tracing(),
            "allocated_blocks": sys.getallocatedblocks(),
            "gc_counts": list(gc.get_count()),
            "gc_collections": [row["collections"] for row in gc.get_stats()],
            "gc_collected": [row["collected"] for row in gc.get_stats()],
            "live_object_types": dict(sorted(counters.items(), key=lambda kv: -kv[1])[:15]),
            "live_object_count": sum(counters.values()),
        }
    if action != "snapshot":
        raise HTTPException(status_code=422, detail="action must be start, reset, stop, snapshot or blocks")

    if not tracemalloc.is_tracing():
        return {
            "tracing": False,
            "hint": "call ?action=start, drive the workload, then request a snapshot",
        }

    gc.collect()
    snapshot = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()

    # Growth since the previous snapshot: request two snapshots around a workload and this
    # list is the leak, with the allocation site. A store at its ceiling shows up with a
    # near-zero delta even when its absolute size is large; a leak shows a positive delta
    # that keeps returning.
    growth: list[dict[str, Any]] = []
    if _last_snapshot is not None:
        for stat in snapshot.compare_to(_last_snapshot, "traceback")[: max(1, min(limit, 50))]:
            if stat.size_diff <= 0:
                continue
            frames = [f"{frame.filename}:{frame.lineno}" for frame in reversed(stat.traceback)]
            growth.append(
                {
                    "size_diff_kib": round(stat.size_diff / 1024.0, 1),
                    "object_diff": stat.count_diff,
                    "allocation_site": frames[0],
                    "frames_nearest_first": frames[:8],
                }
            )
    _last_snapshot = snapshot
    top = []
    for entry in snapshot.statistics("traceback")[: max(1, min(limit, 50))]:
        # Newest frame first: the allocation site is what matters, and it is the *last* entry
        # in tracemalloc's (oldest-first) traceback. Printing the outer frames first hid the
        # application frames behind asyncio/uvicorn boilerplate.
        frames = [f"{frame.filename}:{frame.lineno}" for frame in reversed(entry.traceback)]
        top.append(
            {
                "size_kib": round(entry.size / 1024.0, 1),
                "objects": entry.count,
                "allocation_site": frames[0],
                "frames_nearest_first": frames[:8],
            }
        )
    process = _process_resources()
    return {
        "tracing": True,
        "traced_current_mb": round(current / 1e6, 2),
        "traced_peak_mb": round(peak / 1e6, 2),
        "rss_mb": process.get("rss_mb"),
        "growth_since_previous_snapshot": growth,
        "has_previous_snapshot": bool(growth),
        "top_allocations": top,
        "note": (
            "Sizes are live Python allocations grouped by allocation site. Compare snapshots "
            "before and after a workload to see what actually grew; a bounded store at its "
            "ceiling shows up as a large but flat allocation."
        ),
    }


@operational_router.get("/memory/stores")
async def list_bounded_stores() -> dict[str, Any]:
    """One row per in-process accumulator: ceiling, occupancy and evictions.

    ``/metrics/runtime`` reports only the stores that are already full. Answering the question
    an operator actually asks — "the process is growing; is that a store still filling up, or
    something unbounded?" — needs the *remaining* capacity of every store, including the ones
    that are nowhere near their ceiling yet.
    """
    return audit_bounds()


@operational_router.get("/metrics/runtime")
async def get_runtime_metrics() -> dict[str, Any]:
    """Returns runtime telemetry metrics, spend budgets, and circuit breaker states."""
    breakers = {
        prov: {
            "threshold": getattr(limiter, "max_concurrency", 4),
            "rate_rps": getattr(limiter, "rate", 5.0),
        }
        for prov, limiter in model_gateway.rate_limiters.items()
    }
    audit = audit_bounds()
    return {
        "metrics": {
            "providers_tracked": len(model_gateway.rate_limiters),
            "rate_limiters": breakers,
            "system_version": __version__,
        },
        # Every in-process accumulator with its ceiling and eviction count. A store that is
        # at its ceiling and evicting is working as designed; a store that is *over* its
        # ceiling would be a bug, which ``within_bound`` reports directly.
        "retention": {
            "store_count": audit["store_count"],
            "within_bound": audit["within_bound"],
            "overflowing": audit["overflowing"],
            "total_evictions": audit["total_evictions"],
            "stores_at_capacity": [
                {"name": row["name"], "used": row["used"], "max_entries": row["max_entries"],
                 "evicted": row["evicted"]}
                for row in audit["stores"]
                if row["max_entries"] and row["used"] >= row["max_entries"]
            ],
        },
        "process": _process_resources(),
        # Allocator high-water is not visible in the store audit: glibc keeps freed arenas
        # mapped. The guard's counters say whether that residue is being returned to the OS.
        "memory_guard": memory_guard.describe(),
    }
