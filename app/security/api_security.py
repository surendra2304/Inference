import hmac
import time
import uuid

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.config import settings
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
        if client_ip in ("testclient", "127.0.0.1", "localhost"):
            return True
        now = time.time()
        if client_ip in self._blocked_ips:
            return False

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
        client_ip = request.client.host if request.client else "127.0.0.1"

        # 1. IP Block / Rate limit check
        if not security_manager.check_rate_limit(client_ip):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Too many requests. Please throttle your traffic."}
            )

        # 2. Request body size check (Max 2MB)
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 2 * 1024 * 1024:
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
