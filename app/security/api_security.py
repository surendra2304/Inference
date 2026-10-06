import hmac
import os
import time
import uuid

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.config import settings
from app.core.security import resolve_client_ip
from app.utils.logger import logger


class APISecurityManager:
    """Manages API Keys, Rate Limiting per IP/Client, Input Sanitization, and Response Security."""

    def __init__(self, valid_api_keys: set[str] | None = None) -> None:
        # Never ship credentials in source. Load configured ecosystem keys at
        # startup; explicit injection is available for isolated unit tests.
        self._valid_api_keys = (
            set(valid_api_keys)
            if valid_api_keys is not None
            else {
                key
                for key in (
                    settings.INFERENCE_API_KEY,
                    settings.inference_api_KEY,
                    settings.FRIDAY_UNIVERSE_API_KEY,
                    settings.X_FRIDAY_API_KEY,
                    settings.FRIDAY_API_KEY,
                )
                if key
            }
        )
        # Rate limit tracking: ip -> list of timestamps
        self._rate_limits: dict[str, list] = {}
        self._rate_limit_max_requests = 120  # per minute
        self._rate_limit_window = 60.0  # seconds
        # Distinct client IPs are effectively unbounded; without a ceiling this
        # dict grows forever even though each individual window is pruned.
        self._max_tracked_ips = 10_000

        # Suspicious / Blocked IPs
        self._blocked_ips: set[str] = set()

    def validate_api_key(self, api_key: str | None) -> bool:
        """Validates bearer API key."""
        if not api_key:
            return False
        # Remove 'Bearer ' prefix if present
        clean_key = api_key.removeprefix("Bearer ").strip()
        candidate = clean_key.encode("utf-8")
        return any(
            hmac.compare_digest(candidate, valid_key.encode("utf-8"))
            for valid_key in self._valid_api_keys
        )

    def check_rate_limit(self, client_ip: str) -> bool:
        """Enforces sliding-window rate limiting per IP."""
        if client_ip == "testclient":
            # Test-harness identity, not a routable address.
            return True
        if client_ip in ("127.0.0.1", "::1", "localhost"):
            # The loopback bypass is opt-in. Without a gate, deploying this
            # service behind a loopback reverse proxy with APP_ENV=development
            # (the value .env.example used to ship) disabled rate limiting for
            # every caller at once.
            explicitly_dev = (
                os.environ.get("PYTEST_CURRENT_TEST") is not None
                or settings.APP_ENV in ("development", "test")
                or settings.INSECURE_DEV_AUTH
                or settings.ALLOW_DEV_RATE_LIMIT_BYPASS
            )
            if explicitly_dev:
                return True
        now = time.time()
        if client_ip in self._blocked_ips:
            return False

        # Bound cardinality before recording this IP: expire idle buckets, and
        # if still at capacity evict the least-recently active ones.
        if len(self._rate_limits) >= self._max_tracked_ips:
            cutoff = now - self._rate_limit_window
            for ip, ts in list(self._rate_limits.items()):
                if not ts or ts[-1] < cutoff:
                    self._rate_limits.pop(ip, None)
            if len(self._rate_limits) >= self._max_tracked_ips:
                oldest_first = sorted(
                    self._rate_limits.items(),
                    key=lambda kv: kv[1][-1] if kv[1] else 0.0,
                )
                for ip, _ in oldest_first[: max(1, self._max_tracked_ips // 10)]:
                    self._rate_limits.pop(ip, None)

        timestamps = self._rate_limits.get(client_ip, [])
        # Filter out timestamps outside window
        timestamps = [ts for ts in timestamps if now - ts < self._rate_limit_window]

        if len(timestamps) >= self._rate_limit_max_requests:
            logger.warning("Rate limit exceeded for IP: %s (%d reqs in %.1fs)", client_ip, len(timestamps), self._rate_limit_window)
            return False

        timestamps.append(now)
        self._rate_limits[client_ip] = timestamps
        return True

    def sanitize_input(self, text: str) -> str:
        """Sanitizes user and parameter input against injection vectors."""
        if not isinstance(text, str):
            return text
        dangerous_chars = ["<script>", "</script>", "javascript:", "DROP TABLE", "--", "exec("]
        sanitized = text
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "")
        return sanitized.strip()


security_manager = APISecurityManager()


class ProductionSecurityMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing security headers, request size limits, and basic rate control."""

    async def dispatch(self, request: Request, call_next):
        # Never default an unknown client to loopback: that would lump every
        # unidentified caller into one bucket and grant it localhost treatment.
        client_ip = resolve_client_ip(request)

        # 1. IP Block / Rate limit check
        if not security_manager.check_rate_limit(client_ip):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Too many requests. Please throttle your traffic."}
            )

        # 2. Request body size check (Max 2MB)
        content_length = request.headers.get("content-length")
        try:
            content_length_bytes = int(content_length) if content_length else 0
        except ValueError:
            # A malformed Content-Length must not crash the middleware; treat it
            # as oversized rather than letting int() raise a 500.
            content_length_bytes = 2 * 1024 * 1024 + 1
        if content_length_bytes > 2 * 1024 * 1024:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"detail": "Request payload exceeds 2MB limit."}
            )

        try:
            response: Response = await call_next(request)
        except Exception as exc:
            correlation_id = (
                request.headers.get("X-Correlation-ID")
                or request.headers.get("X-Request-ID")
                or str(uuid.uuid4())
            )
            logger.error(
                "Unhandled exception [correlation_id=%s] on %s %s: %s",
                correlation_id, request.method, request.url.path, str(exc), exc_info=True
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error_code": "INTERNAL_SERVER_ERROR",
                    "message": "An internal server error occurred.",
                    "correlation_id": correlation_id
                },
                headers={"X-Correlation-ID": correlation_id}
            )

        # 3. Add Production Security Headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"

        return response
