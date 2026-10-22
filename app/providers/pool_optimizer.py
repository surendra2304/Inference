"""Provider Pool Optimizer with task specialization, health-aware load balancing, and demotion."""

import time
from typing import Any

from app.utils.logger import logger


class ProviderPoolOptimizer:
    """Dynamically routes code-gen, architecture, and debugging to best performing providers."""

    def __init__(self) -> None:
        # Task type -> Best providers ranked
        self.specializations: dict[str, list[str]] = {
            "code_generation": ["groq", "gemini", "nvidia", "openrouter"],
            "architecture": ["nvidia", "gemini", "groq", "openrouter"],
            "debugging": ["gemini", "groq", "nvidia", "openrouter"],
            "review": ["gemini", "groq", "nvidia", "openrouter"],
            "documentation": ["gemini", "groq", "nvidia", "openrouter"],
        }
        # Provider demotions: provider -> demoted_until_timestamp
        self._demotions: dict[str, float] = {}
        self._consecutive_failures: dict[str, int] = {}
        self._performance: dict[str, dict[str, Any]] = {
            p: {"successes": 10, "failures": 0, "avg_latency_ms": 250.0}
            for p in ["gemini", "groq", "nvidia", "openrouter"]
        }

    def get_optimal_provider(self, task_type: str = "code_generation") -> str:
        """Selects the best non-demoted provider for the given task."""
        now = time.time()
        candidates = self.specializations.get(task_type, ["gemini", "groq", "openrouter"])

        for p in candidates:
            demote_until = self._demotions.get(p, 0.0)
            if now >= demote_until:
                return p

        # Fallback to first candidate if all demoted
        return candidates[0]

    def record_provider_result(self, provider: str, success: bool, latency_ms: float) -> None:
        """Tracks provider success rate and handles automatic demotion on 3 consecutive failures."""
        stats = self._performance.setdefault(provider, {"successes": 0, "failures": 0, "avg_latency_ms": 200.0})
        if success:
            stats["successes"] += 1
            self._consecutive_failures[provider] = 0
            # Update moving average latency
            stats["avg_latency_ms"] = round((stats["avg_latency_ms"] * 0.8) + (latency_ms * 0.2), 2)
        else:
            stats["failures"] += 1
            failures = self._consecutive_failures.get(provider, 0) + 1
            self._consecutive_failures[provider] = failures

            if failures >= 3:
                # Demote for 1 hour
                self._demotions[provider] = time.time() + 3600.0
                logger.warning("Provider %s demoted for 1 hour after 3 consecutive failures.", provider)

    def get_performance_report(self) -> dict[str, Any]:
        """Returns comprehensive performance metrics per provider."""
        return {
            "provider_stats": self._performance,
            "active_demotions": {p: round(t - time.time(), 1) for p, t in self._demotions.items() if t > time.time()},
            "task_routing_matrix": self.specializations,
        }


provider_pool_optimizer = ProviderPoolOptimizer()
