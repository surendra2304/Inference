"""Bound the resident set of a long-lived process when the allocator, not the code, holds it.

Measured on this service (``scripts/memory_forensics.py``, 1,600 ``/ask`` requests at
concurrency 6, tracemalloc off): the resident set ends 1.3 MB above where it started while the
live Python block count returns *exactly* to its baseline — 403,456 blocks at rest, 698,136
under load, 404,371 after 45 s idle. Nothing is retained at the object level. The residue is
allocator high-water: glibc keeps freed arenas mapped instead of returning them to the kernel,
so RSS drifts upward with every burst of traffic and never comes back down.

That is not a leak, but it still ends in an OOM kill on a small container, and the cure is an
explicit ``malloc_trim`` once the traffic goes quiet. This module provides it:

* :func:`process_rss_mb` — the resident set, from ``/proc/self/status`` when available.
* :func:`trim_allocator` — ``malloc_trim(0)`` through ``ctypes``, with the RSS before/after.
* :class:`MemoryGuard` — counts in-flight requests and trims only when the process is *idle*
  and above a threshold, so a trim never runs while a request is being served.

Everything here fails soft: on a platform without glibc the guard reports ``supported: False``
and does nothing, and a trim error is recorded rather than raised. Memory hygiene must never
be able to turn into an outage.
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

#: Trimming is cheap but not free; and calling it on every completed request would be silly.
#: The guard only *considers* a trim every ``CHECK_EVERY`` completions.
CHECK_EVERY = 25


def process_rss_mb() -> float | None:
    """Resident set size of this process in MB, or None when the OS does not expose it."""
    try:
        with open("/proc/self/status") as handle:
            status = handle.read()
        return round(int(status.split("VmRSS:")[1].split()[0]) / 1024.0, 1)
    except Exception:  # non-Linux, restricted /proc, unexpected format
        return None


def trim_allocator() -> dict[str, Any]:
    """Ask glibc to return free arena memory to the kernel.

    Returns a dict with ``supported``, ``before_mb``, ``after_mb`` and ``reclaimed_mb``. Never
    raises: a missing ``libc`` (musl, Windows, macOS) or a failed call is reported, not
    propagated.
    """
    before = process_rss_mb()
    result: dict[str, Any] = {"supported": False, "before_mb": before, "after_mb": before,
                              "reclaimed_mb": 0.0}
    try:
        libc = ctypes.CDLL("libc.so.6")
        trim = libc.malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
    except Exception as exc:  # pragma: no cover - platform dependent
        result["detail"] = f"malloc_trim unavailable: {type(exc).__name__}: {exc}"
        return result

    try:
        trim(0)
    except Exception as exc:  # pragma: no cover - defensive
        result["detail"] = f"malloc_trim failed: {type(exc).__name__}: {exc}"
        return result

    after = process_rss_mb()
    result["supported"] = True
    result["after_mb"] = after
    if before is not None and after is not None:
        result["reclaimed_mb"] = round(max(0.0, before - after), 1)
    return result


class MemoryGuard:
    """Trim the allocator when (and only when) the process is idle and above a threshold.

    The guard is traffic-aware: :meth:`request_started`/:meth:`request_finished` bracket every
    served request (wired into ``RequestMetricsMiddleware``), and a trim is only attempted on
    completion, when the in-flight count is zero. So the process never calls ``malloc_trim``
    in the middle of building a response.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        threshold_mb: float = 400.0,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self.enabled = enabled
        self.threshold_mb = float(threshold_mb)
        self.cooldown_seconds = float(cooldown_seconds)
        self._in_flight = 0
        self._completions = 0
        self._last_trim: float = 0.0
        self._trims = 0
        self._reclaimed_total_mb = 0.0
        self._last_result: dict[str, Any] | None = None
        self._lock = threading.Lock()

    # -- traffic hooks -----------------------------------------------------
    def request_started(self) -> None:
        with self._lock:
            self._in_flight += 1

    def request_finished(self) -> None:
        """Decrement the in-flight count and consider a trim once the process goes idle."""
        with self._lock:
            if self._in_flight > 0:
                self._in_flight -= 1
            self._completions += 1
            should_consider = self._completions % CHECK_EVERY == 0
            idle = self._in_flight == 0
        if should_consider and idle:
            self.maybe_trim()

    # -- policy ------------------------------------------------------------
    def should_trim(self, rss_mb: float | None = None, now: float | None = None) -> tuple[bool, str]:
        """The decision, exposed so it can be tested without allocating real arenas."""
        if not self.enabled:
            return False, "disabled"
        with self._lock:
            if self._in_flight > 0:
                return False, f"{self._in_flight} request(s) in flight"
            last = self._last_trim
        rss = process_rss_mb() if rss_mb is None else rss_mb
        if rss is None:
            return False, "resident set size unavailable on this platform"
        if rss < self.threshold_mb:
            return False, f"rss {rss:.1f} MB below threshold {self.threshold_mb:.1f} MB"
        moment = time.monotonic() if now is None else now
        if last and (moment - last) < self.cooldown_seconds:
            remaining = self.cooldown_seconds - (moment - last)
            return False, f"cooldown active for another {remaining:.0f}s"
        return True, f"rss {rss:.1f} MB is above threshold and the process is idle"

    def maybe_trim(self) -> dict[str, Any] | None:
        """Trim when policy allows; return the trim result (or None when it did not run)."""
        allowed, reason = self.should_trim()
        if not allowed:
            return None
        return self.force_trim(reason)

    def force_trim(self, reason: str = "requested") -> dict[str, Any]:
        """Trim regardless of policy (used by the diagnostics endpoint and tests)."""
        result = trim_allocator()
        with self._lock:
            self._last_trim = time.monotonic()
            self._trims += 1
            self._reclaimed_total_mb += float(result.get("reclaimed_mb") or 0.0)
            result["reason"] = reason
            self._last_result = result
        if result.get("reclaimed_mb"):
            logger.info(
                "memory guard returned %.1f MB to the OS (rss %.1f -> %.1f MB, %s)",
                result["reclaimed_mb"], result.get("before_mb") or 0.0,
                result.get("after_mb") or 0.0, reason,
            )
        return result

    def describe(self) -> dict[str, Any]:
        """State for ``/metrics/runtime`` — enough to audit the guard without guessing."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "threshold_mb": self.threshold_mb,
                "cooldown_seconds": self.cooldown_seconds,
                "in_flight": self._in_flight,
                "completions": self._completions,
                "trims": self._trims,
                "reclaimed_total_mb": round(self._reclaimed_total_mb, 1),
                "last_trim": self._last_result,
            }


#: Process-wide guard. Defaults are deliberately conservative: a trim is attempted only after
#: the resident set passes the threshold *and* the process has been idle for the cooldown, so
#: a normal-sized deployment never pays for this at all.
memory_guard = MemoryGuard()
