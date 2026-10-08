"""Request-level performance recording for every route.

``PerformanceMonitor`` existed with a 5000-sample latency window, but the only caller of
``record_request`` was the trading-consult service (``app/services/trading_consult_service.py``),
so ``/health`` and the governance snapshot published ``total_requests``/``error_rate_pct``/
``p50..p99`` for a subset of traffic while reading as if they covered the API. Measured on a
process that had served thousands of ``/ask`` requests, the block reported only the handful
of consult calls.

This middleware records every completed request exactly once, so the latency window and the
error rate describe the whole process. It is deliberately tiny and never fails a request:
a metrics bug must not become an outage.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.utils.memory_guard import memory_guard

#: Paths that would otherwise dominate the latency window with liveness polling and make
#: the percentiles describe monitoring traffic instead of served work.
_EXCLUDED_PREFIXES = (
    "/health",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/favicon.ico",
)


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    """Record latency and success/failure for every served request into ``monitor``."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path
        if path.startswith(_EXCLUDED_PREFIXES):
            return await call_next(request)

        # Bracket the request for the memory guard: it counts in-flight work so that a
        # malloc_trim (see app/utils/memory_guard.py) can never run while a response is
        # being built. Cost is two integer updates.
        memory_guard.request_started()
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # The exception handlers turn this into a 500; record the failure here so the
            # error rate reflects it either way, then let the framework deal with it.
            self._record(time.perf_counter() - started, success=False)
            memory_guard.request_finished()
            raise
        self._record(time.perf_counter() - started, success=response.status_code < 500)
        memory_guard.request_finished()
        return response

    @staticmethod
    def _record(latency_sec: float, *, success: bool) -> None:
        try:
            from app.monitoring import monitor

            monitor.record_request(latency_sec, success=success)
        except Exception:  # noqa: BLE001 - telemetry must never break a response
            pass
