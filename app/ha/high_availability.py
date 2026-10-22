"""High Availability Architecture with Multi-Provider Failovers and Health Monitoring."""

import time
from typing import Any

from app.utils.logger import logger


class HighAvailabilityManager:
    """Manages Provider Failover Chains, Disaster Recovery health checks, and Node Redundancy."""

    def __init__(self) -> None:
        self.provider_chain = ["groq", "openrouter", "gemini", "nvidia"]
        self.provider_health: dict[str, dict[str, Any]] = {
            p: {"status": "HEALTHY", "consecutive_failures": 0, "last_failure_ts": 0.0}
            for p in self.provider_chain
        }

    def record_provider_result(self, provider: str, success: bool) -> None:
        """Updates provider health based on invocation results."""
        if provider not in self.provider_health:
            self.provider_health[provider] = {"status": "HEALTHY", "consecutive_failures": 0, "last_failure_ts": 0.0}

        rec = self.provider_health[provider]
        if success:
            rec["consecutive_failures"] = 0
            rec["status"] = "HEALTHY"
        else:
            rec["consecutive_failures"] += 1
            rec["last_failure_ts"] = time.time()
            if rec["consecutive_failures"] >= 3:
                rec["status"] = "DEGRADED"
                logger.error("HA Manager marked provider %s as DEGRADED due to %d consecutive failures.", provider, rec["consecutive_failures"])

    def get_healthy_provider_chain(self) -> list[str]:
        """Returns ordered list of active healthy providers, moving degraded nodes to the tail."""
        healthy = [p for p in self.provider_chain if self.provider_health[p]["status"] == "HEALTHY"]
        degraded = [p for p in self.provider_chain if self.provider_health[p]["status"] != "HEALTHY"]
        return healthy + degraded

    def get_ha_status(self) -> dict[str, Any]:
        """Returns overall high-availability telemetry."""
        healthy_count = sum(1 for p in self.provider_health.values() if p["status"] == "HEALTHY")
        return {
            "ha_mode": "ACTIVE_REDUNDANT",
            "healthy_providers_count": healthy_count,
            "total_providers_count": len(self.provider_chain),
            "failover_order": self.get_healthy_provider_chain(),
            "node_status": "OPTIMAL" if healthy_count >= 2 else "DEGRADED"
        }


ha_manager = HighAvailabilityManager()
