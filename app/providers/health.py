"""Provider Health Tracking Subsystem for Inference."""

import time
from typing import Any

from pydantic import BaseModel, Field

from app.core.config import settings


class ProviderHealthReport(BaseModel):
    """Health metrics and operational status for an individual provider."""

    provider_name: str
    is_healthy: bool
    health_score: float = Field(ge=0.0, le=1.0, description="Health score between 0.0 (dead) and 1.0 (perfect)")
    success_count: int = 0
    failure_count: int = 0
    rate_limit_429_count: int = 0
    service_error_503_count: int = 0
    consecutive_failures: int = 0
    average_latency_seconds: float = 0.0
    last_latency_seconds: float = 0.0
    last_success_timestamp: float | None = None
    last_failure_timestamp: float | None = None
    last_error_message: str | None = None
    active_keys_count: int = 0
    quarantined_keys_count: int = 0


class ProviderHealthTracker:
    """Tracks latency, success rate, 429 rate limits, and health scores across providers."""

    def __init__(self) -> None:
        self._stats: dict[str, dict[str, Any]] = {}

    def _get_or_create(self, provider_name: str) -> dict[str, Any]:
        prov = provider_name.lower().strip()
        if prov not in self._stats:
            self._stats[prov] = {
                "success_count": 0,
                "failure_count": 0,
                "429_count": 0,
                "503_count": 0,
                "consecutive_failures": 0,
                "total_latency": 0.0,
                "last_latency": 0.0,
                "last_success": None,
                "last_failure": None,
                "last_error": None,
                "active_keys": 0,
                "quarantined_keys": 0,
            }
        return self._stats[prov]

    def record_success(self, provider_name: str, latency_seconds: float) -> None:
        """Record a successful provider request."""
        stats = self._get_or_create(provider_name)
        stats["success_count"] += 1
        stats["consecutive_failures"] = 0
        stats["total_latency"] += latency_seconds
        stats["last_latency"] = latency_seconds
        stats["last_success"] = time.time()

    def record_failure(
        self, provider_name: str, error: str, is_429: bool = False, is_503: bool = False, latency_seconds: float = 0.0
    ) -> None:
        """Record a failed provider request with specific error classifications."""
        stats = self._get_or_create(provider_name)
        stats["failure_count"] += 1
        stats["consecutive_failures"] += 1
        stats["last_error"] = error
        stats["last_failure"] = time.time()
        if latency_seconds > 0:
            stats["last_latency"] = latency_seconds

        if is_429:
            stats["429_count"] += 1
        if is_503:
            stats["503_count"] += 1

    def update_key_counts(self, provider_name: str, active_count: int, quarantined_count: int) -> None:
        """Update active and quarantined key counts for health reporting."""
        stats = self._get_or_create(provider_name)
        stats["active_keys"] = active_count
        stats["quarantined_keys"] = quarantined_count

    def get_provider_health(self, provider_name: str) -> ProviderHealthReport:
        """Calculates and returns live health status for a specific provider.

        Recovery semantics matter as much as the penalty calculation here. Two
        independent paths could previously banish a provider **permanently**:

        1. ``consecutive_failures >= 4`` → unhealthy. The only thing that resets that
           counter is :meth:`record_success`, but the callers that record successes are
           the same callers that skip unhealthy providers
           (``debate.py::_invoke_single_model``), so the counter never resets and the
           provider is never called again for the lifetime of the process.
        2. A low *lifetime* success rate keeps ``health_score`` under 0.3 even after a
           fresh success, so the score gate alone can hold a recovered provider out.

        Both are fixed by treating health as a circuit breaker rather than a verdict:
        a provider that has succeeded since its last failure is healthy again, and one
        that has been quiet for the recovery window is offered a half-open probe. A
        failed probe re-arms the window; a successful probe clears the failure count.
        """
        prov = provider_name.lower().strip()
        stats = self._get_or_create(prov)

        total_requests = stats["success_count"] + stats["failure_count"]
        success_rate = (stats["success_count"] / total_requests) if total_requests > 0 else 1.0
        avg_latency = (stats["total_latency"] / stats["success_count"]) if stats["success_count"] > 0 else 0.0

        # Calculate dynamic health score
        # Penalties: consecutive failures (-0.2 each), 429s, 503s
        penalty = min(0.8, stats["consecutive_failures"] * 0.2)
        if stats["429_count"] > 0 and total_requests > 0:
            penalty += min(0.2, (stats["429_count"] / total_requests) * 0.4)

        health_score = max(0.0, min(1.0, success_rate - penalty))

        last_success = stats["last_success"]
        last_failure = stats["last_failure"]

        # (1) Proven recovery: a success more recent than the last failure means the
        # provider is answering right now, whatever the lifetime ratio says.
        recovered_since_failure = (
            last_success is not None and (last_failure is None or last_success >= last_failure)
        )

        # (2) Half-open probe: after the cooldown, report healthy so a caller will
        # actually try the provider. Without this the failure counter can never be
        # cleared, because nothing ever gets through to record a success.
        cooldown_elapsed = (
            last_failure is not None
            and (time.time() - last_failure) >= float(settings.PROVIDER_HEALTH_RECOVERY_SECONDS)
            and stats["consecutive_failures"] > 0
        )

        is_healthy = (
            (health_score >= 0.3 and stats["consecutive_failures"] < 4)
            or recovered_since_failure
            or cooldown_elapsed
        )

        return ProviderHealthReport(
            provider_name=prov,
            is_healthy=is_healthy,
            health_score=round(health_score, 2),
            success_count=stats["success_count"],
            failure_count=stats["failure_count"],
            rate_limit_429_count=stats["429_count"],
            service_error_503_count=stats["503_count"],
            consecutive_failures=stats["consecutive_failures"],
            average_latency_seconds=round(avg_latency, 3),
            last_latency_seconds=round(stats["last_latency"], 3),
            last_success_timestamp=stats["last_success"],
            last_failure_timestamp=stats["last_failure"],
            last_error_message=stats["last_error"],
            active_keys_count=stats["active_keys"],
            quarantined_keys_count=stats["quarantined_keys"],
        )

    def get_all_health(self) -> dict[str, ProviderHealthReport]:
        """Returns health reports for all tracked providers."""
        return {prov: self.get_provider_health(prov) for prov in self._stats.keys()}

    def reset(self) -> None:
        """Reset all tracked stats (useful for test isolation)."""
        self._stats.clear()


# Global singleton instance
provider_health_tracker = ProviderHealthTracker()
