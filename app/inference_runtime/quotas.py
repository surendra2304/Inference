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
    """Enforces strict token and request buckets per minute for FRIDAY Universe agents.

    Prevents recursive runaway loops, denial of service, and vendor quota exhaustion.
    """

    def __init__(self, default_rpm: int = 60, default_tpm: int = 50_000) -> None:
        self.default_rpm = max(1, default_rpm)
        self.default_tpm = max(1, default_tpm)
        self._q: dict[str, Quota] = {}
        self._lock = threading.RLock()
        self._init_defaults()

    def _init_defaults(self) -> None:
        for agent, (rpm, tpm) in DEFAULT_AGENT_QUOTAS.items():
            self._q[agent.lower()] = Quota(rpm, tpm, window_start=time.time())

    def configure(self, tenant: str, rpm: int, tpm: int) -> None:
        with self._lock:
            self._q[tenant.lower()] = Quota(max(1, rpm), max(1, tpm), window_start=time.time())

    def admit(self, tenant: str, estimated_tokens: int = 1) -> None:
        norm_tenant = tenant.lower().strip()
        with self._lock:
            q = self._q.get(norm_tenant)
            if q is None:
                # Auto-initialize with default quota bucket
                q = Quota(self.default_rpm, self.default_tpm, window_start=time.time())
                self._q[norm_tenant] = q

            now = time.time()
            if now - q.window_start >= 60.0:
                q.requests = 0
                q.tokens = 0
                q.window_start = now

            if q.requests + 1 > q.rpm:
                raise RateLimitExceeded(
                    f"Request rate limit exceeded for calling agent '{tenant}' ({q.rpm} req/min). Window resets in {int(60 - (now - q.window_start))}s."
                )

            if q.tokens + estimated_tokens > q.tpm:
                raise RateLimitExceeded(
                    f"Token quota exceeded for calling agent '{tenant}' ({q.tpm} tokens/min). Window resets in {int(60 - (now - q.window_start))}s."
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
            }


# Process-wide singleton quota manager
agent_quota_manager = TenantQuotaManager()
