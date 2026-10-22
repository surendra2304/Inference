"""Normalized Provider Exception Hierarchy for Inference Gateway."""

from typing import Any, Optional


class GatewayError(Exception):
    """Base class for all provider gateway errors."""

    def __init__(
        self, message: str, provider: str = "unknown", model: Optional[str] = None, raw_error: Optional[Any] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model
        self.raw_error = raw_error

    def is_retryable(self) -> bool:
        return False


class RateLimitError(GatewayError):
    """Provider returned 429 Too Many Requests or rate-limit quota exhaustion."""

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        model: Optional[str] = None,
        retry_after: float = 60.0,
        raw_error: Optional[Any] = None,
    ) -> None:
        super().__init__(message, provider=provider, model=model, raw_error=raw_error)
        self.retry_after = retry_after

    def is_retryable(self) -> bool:
        return True


class TemporaryUnavailableError(GatewayError):
    """Provider returned 502/503 Service Unavailable or server overload."""

    def is_retryable(self) -> bool:
        return True


class TimeoutError(GatewayError):
    """Provider call exceeded time ceiling or connection timeout."""

    def is_retryable(self) -> bool:
        return True


class ProviderUnconfiguredError(GatewayError):
    """The provider has no credential and cannot operate keyless.

    This is a *configuration* fact, not a provider fault: no amount of retrying,
    key-rotation, or waiting will make an unconfigured provider succeed. It is
    deliberately **not** retryable, is never recorded against provider health
    (recording it would poison the circuit breaker with information about the
    deployment rather than about the provider), and is raised *before* the
    per-provider rate limiter is acquired — waiting in a queue for a call that can
    never be served is pure latency added to a request that is already doomed.
    """

    def is_retryable(self) -> bool:
        return False


class AuthenticationError(GatewayError):
    """Provider returned 401 Unauthorized or invalid API key."""

    def is_retryable(self) -> bool:
        return False


class InvalidRequestError(GatewayError):
    """Provider returned 400 Bad Request or invalid model/parameters."""

    def is_retryable(self) -> bool:
        return False


class ContentPolicyViolationError(GatewayError):
    """Provider rejected input or output due to safety/content policies."""

    def is_retryable(self) -> bool:
        return False


class CapabilityMismatchError(GatewayError):
    """Target or fallback model does not fulfill requested capability."""

    def is_retryable(self) -> bool:
        return False


def normalize_provider_exception(exc: Exception, provider: str, model: Optional[str] = None) -> GatewayError:
    """Classifies any raw Python or HTTP exception into a typed GatewayError."""
    if isinstance(exc, GatewayError):
        return exc

    err_str = str(exc).lower()

    if "429" in err_str or "rate limit" in err_str or "quota" in err_str or "too many requests" in err_str:
        return RateLimitError(str(exc), provider=provider, model=model, raw_error=exc)

    if (
        "401" in err_str
        or "unauthorized" in err_str
        or "authentication" in err_str
        or "invalid api key" in err_str
        or "forbidden" in err_str
        or "403" in err_str
    ):
        return AuthenticationError(str(exc), provider=provider, model=model, raw_error=exc)

    if (
        "503" in err_str
        or "502" in err_str
        or "504" in err_str
        or "unavailable" in err_str
        or "overloaded" in err_str
        or "bad gateway" in err_str
    ):
        return TemporaryUnavailableError(str(exc), provider=provider, model=model, raw_error=exc)

    if "timeout" in err_str or "timed out" in err_str or "deadline" in err_str:
        return TimeoutError(str(exc), provider=provider, model=model, raw_error=exc)

    if "safety" in err_str or "content policy" in err_str or "blocked" in err_str or "harmful" in err_str:
        return ContentPolicyViolationError(str(exc), provider=provider, model=model, raw_error=exc)

    if "400" in err_str or "bad request" in err_str or "not found" in err_str or "404" in err_str:
        return InvalidRequestError(str(exc), provider=provider, model=model, raw_error=exc)

    return GatewayError(str(exc), provider=provider, model=model, raw_error=exc)
