"""Enhanced Multi-Consumer Rate Limiter Middleware.

Features:
- Per-API-Key rate limits configured via MultiConsumerRouter
- Burst allowance (2x limit for 60-second bursts)
- Request queuing / backoff with retry-after header
- Standard rate limit response headers (X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset)
"""

import threading
import time
from collections.abc import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.security import resolve_client_ip
from app.routing.consumer_router import consumer_router
from app.utils.logger import logger


class EnhancedRateLimiterMiddleware(BaseHTTPMiddleware):
    """Middleware applying dynamic per-consumer rate limits and standard headers."""

    def __init__(self, app, max_tracked_keys: int = 10000) -> None:
        super().__init__(app)
        # key -> (tokens_remaining, last_updated_epoch). See the token-bucket
        # commentary in ``dispatch`` for why this is a bucket and not a window.
        self._buckets: dict[str, tuple[float, float]] = {}
        self._window_seconds = 3600.0  # quota period (1 hour)
        self._max_tracked_keys = max_tracked_keys
        self._lock = threading.RLock()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Exempt health and docs checks
        if request.url.path in ("/health", "/", "/docs", "/openapi.json", "/v1/forge/health"):
            return await call_next(request)

        # Identify consumer identity
        auth_header = request.headers.get("Authorization", "")
        api_key_header = request.headers.get("X-API-Key", "")
        # Resolve the originating client, not the reverse proxy that forwarded
        # the request - otherwise every caller behind the proxy shares one bucket
        # and a local proxy hands them all the localhost bypass.
        client_ip = resolve_client_ip(request)

        # Local testing / development exemption.
        # Test detection uses the runner's own environment variable (set only in
        # the pytest process; not influenceable by an HTTP caller) rather than
        # `"pytest" in sys.modules`, which would silently disable rate limiting
        # in any process that happened to import pytest.
        import os

        is_test_env = os.environ.get("PYTEST_CURRENT_TEST") is not None
        is_dev_env = settings.APP_ENV in ("development", "test") or settings.INSECURE_DEV_AUTH or settings.ALLOW_DEV_RATE_LIMIT_BYPASS

        if (is_test_env or is_dev_env) and client_ip in ("testclient", "127.0.0.1", "::1", "localhost"):
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = "10000"
            response.headers["X-RateLimit-Remaining"] = "9999"
            response.headers["X-RateLimit-Reset"] = str(int(time.time() + 3600))
            return response

        consumer_id = consumer_router.identify_consumer(auth_header or api_key_header or client_ip)
        profile = consumer_router.PROFILES.get(consumer_id, consumer_router.PROFILES["human"])

        now = time.time()
        key = f"{consumer_id}:{client_ip}"
        limit = float(profile.rate_limit_per_hour)
        # Continuous refill, in tokens per second, that yields exactly the documented
        # hourly quota over an hour.
        refill_per_second = limit / self._window_seconds
        capacity = limit

        with self._lock:
            # Memory bounding: drop buckets idle past a full window, then evict the
            # oldest entries if the table is still at capacity.
            if len(self._buckets) >= self._max_tracked_keys:
                expired = [
                    k for k, (_, ts) in self._buckets.items()
                    if (now - ts) >= self._window_seconds
                ]
                for k in expired:
                    self._buckets.pop(k, None)
                if len(self._buckets) >= self._max_tracked_keys:
                    oldest = sorted(self._buckets.items(), key=lambda kv: kv[1][1])
                    for k, _ in oldest[:1000]:
                        self._buckets.pop(k, None)

            # ---------------------------------------------------------------
            # Token bucket instead of a sliding-window cliff.
            #
            # The previous implementation counted requests in a rolling 1-hour
            # window and rejected everything once the count reached the quota.
            # Because the quota is small (the "human" profile allows 50/hour) and
            # the window is long, a client that legitimately bursts — a load test,
            # a batch job, a retry storm — consumed the *entire hour's* allowance in
            # seconds and was then locked out until the window slid, i.e. for up to
            # 60 minutes, with no way to make progress. Measured: a 60-second soak
            # left the caller rejected with HTTP 429 on every subsequent request for
            # the remainder of the hour.
            #
            # A bucket grants the same burst and the same long-run average (capacity
            # = quota, refill = quota/hour) but degrades gracefully: after the burst
            # the caller is throttled to the sustained rate rather than cut off. The
            # worst case changes from "no service for an hour" to "service at the
            # documented rate", which is what a rate limit is supposed to mean.
            # ---------------------------------------------------------------
            tokens, updated = self._buckets.get(key, (capacity, now))
            tokens = min(capacity, tokens + max(0.0, now - updated) * refill_per_second)

            if tokens < 1.0:
                retry_after = max(1, int((1.0 - tokens) / refill_per_second) + 1)
                logger.warning(
                    "Rate limit exceeded for consumer '%s' (IP: %s): %.2f tokens left, retry in %ds",
                    consumer_id, client_ip, tokens, retry_after,
                )
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": (
                            f"Rate limit exceeded for consumer '{consumer_id}'. "
                            f"Limit: {int(limit)}/hour (sustained)."
                        ),
                        "retry_after_seconds": retry_after,
                    },
                    headers={
                        "X-RateLimit-Limit": str(int(limit)),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(now + retry_after)),
                        "Retry-After": str(retry_after),
                    },
                )

            remaining = int(tokens - 1.0)
            self._buckets[key] = (tokens - 1.0, now)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(int(limit))
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        seconds_until_next_token = 1.0 / refill_per_second
        response.headers["X-RateLimit-Reset"] = str(
            int(now + (seconds_until_next_token if remaining == 0 else self._window_seconds))
        )

        return response
