from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from .errors import RateLimitExceeded


@dataclass
class Quota:
    rpm: int
    tpm: int
    requests: int = 0
    tokens: int = 0
    window_start: float = 0.0
    allow_burst: bool = False


# Default specialist agent token & request limits per minute (FRIDAY Universe mesh)
DEFAULT_AGENT_QUOTAS: dict[str, tuple[int, int]] = {
    "forge": (60, 80_000),      # Code generation & architecture
    "vyralx": (60, 60_000),     # Viral scriptwriting
    "intelx": (60, 60_000),     # Research & claim extraction
    "sentinel": (60, 60_000),   # Security & threat detection
    "cortex": (60, 60_000),     # Context & reasoning
    "stratex": (60, 60_000),    # Strategic decision engine
    "memora": (60, 60_000),     # Long-term memory recall
    "futuris": (60, 60_000),    # Predictive forecasting
    "friday": (120, 120_000),   # Core gateway orchestrator
}


class TenantQuotaManager:
    """Enforces dynamic elastic token and request buckets per minute for FRIDAY Universe agents.

    Allows bursting peer agents to borrow unused capacity from the cluster pool,
    throttling only if aggregate cluster capacity reaches 85% to protect vendor limits.
    """

    def __init__(self, default_rpm: int = 60, default_tpm: int = 50_000, aggregate_ceiling_ratio: float = 0.85) -> None:
        self.default_rpm = max(1, default_rpm)
        self.default_tpm = max(1, default_tpm)
        self.aggregate_ceiling_ratio = aggregate_ceiling_ratio
        self._q: dict[str, Quota] = {}
        self._lock = threading.RLock()
        self._init_defaults()

    def _init_defaults(self) -> None:
        for agent, (rpm, tpm) in DEFAULT_AGENT_QUOTAS.items():
            self._q[agent.lower()] = Quota(rpm, tpm, window_start=time.time(), allow_burst=True)

    def configure(self, tenant: str, rpm: int, tpm: int, allow_burst: bool = False) -> None:
        """Configures or overrides a tenant quota. allow_burst=False for strict isolated enforcement."""
        with self._lock:
            self._q[tenant.lower()] = Quota(max(1, rpm), max(1, tpm), window_start=time.time(), allow_burst=allow_burst)

    def admit(self, tenant: str, estimated_tokens: int = 1) -> None:
        norm_tenant = tenant.lower().strip()
        with self._lock:
            q = self._q.get(norm_tenant)
            if q is None:
                # Auto-initialize with default quota bucket and burst elasticity allowed
                q = Quota(self.default_rpm, self.default_tpm, window_start=time.time(), allow_burst=True)
                self._q[norm_tenant] = q

            now = time.time()
            if now - q.window_start >= 60.0:
                q.requests = 0
                q.tokens = 0
                q.window_start = now

            # Refresh all active windows for cluster aggregate capacity calculation
            for item in self._q.values():
                if now - item.window_start >= 60.0:
                    item.requests = 0
                    item.tokens = 0
                    item.window_start = now

            # Check RPM limit
            if q.requests + 1 > q.rpm:
                if not q.allow_burst:
                    raise RateLimitExceeded(
                        f"Request rate limit exceeded for calling agent '{tenant}' ({q.rpm} req/min). Window resets in {int(60 - (now - q.window_start))}s."
                    )
                # Check aggregate cluster RPM capacity
                total_rpm_cap = max(1, sum(item.rpm for item in self._q.values()))
                total_rpm_used = sum(item.requests for item in self._q.values())
                if (total_rpm_used + 1) / total_rpm_cap > self.aggregate_ceiling_ratio:
                    raise RateLimitExceeded(
                        f"Request rate limit exceeded for calling agent '{tenant}' ({q.rpm} req/min) - cluster aggregate capacity reached {int(self.aggregate_ceiling_ratio * 100)}%. Window resets in {int(60 - (now - q.window_start))}s."
                    )

            # Check TPM limit
            if q.tokens + estimated_tokens > q.tpm:
                if not q.allow_burst:
                    raise RateLimitExceeded(
                        f"Token quota exceeded for calling agent '{tenant}' ({q.tpm} tokens/min). Window resets in {int(60 - (now - q.window_start))}s."
                    )
                # Check aggregate cluster TPM capacity
                total_tpm_cap = max(1, sum(item.tpm for item in self._q.values()))
                total_tpm_used = sum(item.tokens for item in self._q.values())
                if (total_tpm_used + estimated_tokens) / total_tpm_cap > self.aggregate_ceiling_ratio:
                    raise RateLimitExceeded(
                        f"Token quota exceeded for calling agent '{tenant}' ({q.tpm} tokens/min) - cluster aggregate capacity reached {int(self.aggregate_ceiling_ratio * 100)}%. Window resets in {int(60 - (now - q.window_start))}s."
                    )

            q.requests += 1
            q.tokens += estimated_tokens

    def get_status(self, tenant: str) -> dict[str, Any]:
        norm = tenant.lower().strip()
        with self._lock:
            q = self._q.get(norm)
            now = time.time()
            if not q:
                return {
                    "tenant": tenant,
                    "rpm_limit": self.default_rpm,
                    "tpm_limit": self.default_tpm,
                    "requests_used": 0,
                    "tokens_used": 0,
                    "window_remaining_seconds": 60,
                    "allow_burst": True,
                }
            if now - q.window_start >= 60.0:
                requests_used = 0
                tokens_used = 0
                rem_sec = 60
            else:
                requests_used = q.requests
                tokens_used = q.tokens
                rem_sec = int(60 - (now - q.window_start))

            return {
                "tenant": tenant,
                "rpm_limit": q.rpm,
                "tpm_limit": q.tpm,
                "requests_used": requests_used,
                "tokens_used": tokens_used,
                "window_remaining_seconds": rem_sec,
                "allow_burst": q.allow_burst,
            }


# Process-wide singleton quota manager
agent_quota_manager = TenantQuotaManager()
