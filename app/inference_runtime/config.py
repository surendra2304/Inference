from __future__ import annotations

import os
from dataclasses import dataclass


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class RuntimeConfig:
    request_timeout_seconds: float = _float("INFERENCE_REQUEST_TIMEOUT", 60.0)
    max_retries: int = _int("INFERENCE_MAX_RETRIES", 2)
    max_concurrency: int = _int("INFERENCE_MAX_CONCURRENCY", 64)
    rpm_limit: int = _int("INFERENCE_RPM_LIMIT", 600)
    cache_ttl_seconds: int = _int("INFERENCE_CACHE_TTL", 60)
    default_budget_usd: float = _float("INFERENCE_DEFAULT_BUDGET_USD", 10.0)
    fail_closed_budget: bool = os.getenv("INFERENCE_FAIL_CLOSED_BUDGET", "true").lower() in {"1", "true", "yes"}
    health_interval_seconds: int = _int("INFERENCE_HEALTH_INTERVAL", 30)
