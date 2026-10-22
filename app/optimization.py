"""Production Performance Optimizations for Inference."""

import asyncio
import hashlib
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from app.config_production import production_config
from app.schemas.trading_consult import AIUniverseDecision, TradingConsultRequest
from app.utils.logger import logger


class TelemetryCache:
    """In-memory LRU-style cache for caching decisions on similar telemetry patterns."""

    def __init__(self, ttl_seconds: int = 86400, max_entries: int = 5000) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self.cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self.hits = 0
        self.misses = 0

    def _generate_key(self, req: TradingConsultRequest) -> str:
        """Computes a stable hash based on telemetry risk bucket and current parameters."""
        t = req.telemetry
        # Normalize continuous values to discrete buckets to capture similar scenarios
        wr_bucket = round(t.win_rate, 2)
        pf_bucket = round(t.profit_factor, 1)
        dd_bucket = round(t.max_drawdown_pct, 1)
        trades_bucket = t.total_trades if t.total_trades < 20 else int(t.total_trades / 10) * 10
        losses = t.consecutive_losses
        mode = req.trading_mode
        arm = req.experiment_group or "NONE"

        key_raw = f"{mode}:{arm}:{wr_bucket}:{pf_bucket}:{dd_bucket}:{trades_bucket}:{losses}:{req.consultation_reason}"
        return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()

    def get(self, req: TradingConsultRequest) -> AIUniverseDecision | None:
        if not production_config.CACHE_ENABLED:
            return None
        key = self._generate_key(req)
        now = time.time()
        if key in self.cache:
            timestamp, decision_dict = self.cache[key]
            if now - timestamp < self.ttl:
                self.hits += 1
                return AIUniverseDecision(**decision_dict)
            else:
                del self.cache[key]
        self.misses += 1
        return None

    def set(self, req: TradingConsultRequest, decision: AIUniverseDecision) -> None:
        if not production_config.CACHE_ENABLED:
            return
        if len(self.cache) >= self.max_entries:
            # Evict oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
            del self.cache[oldest_key]
        key = self._generate_key(req)
        self.cache[key] = (time.time(), decision.model_dump())

    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return round((self.hits / total) * 100.0, 2) if total > 0 else 0.0


class ProviderCircuitBreaker:
    """Circuit breaker for failing external provider APIs."""

    def __init__(self, failure_threshold: int = 3, recovery_time: float = 60.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.failure_counts: dict[str, int] = defaultdict(int)
        self.opened_at: dict[str, float] = {}

    def is_available(self, provider_name: str) -> bool:
        now = time.time()
        if provider_name in self.opened_at:
            if now - self.opened_at[provider_name] > self.recovery_time:
                # Half-open: allow probe
                del self.opened_at[provider_name]
                self.failure_counts[provider_name] = 0
                return True
            return False
        return True

    def record_success(self, provider_name: str) -> None:
        self.failure_counts[provider_name] = 0
        if provider_name in self.opened_at:
            del self.opened_at[provider_name]

    def record_failure(self, provider_name: str) -> None:
        self.failure_counts[provider_name] += 1
        if self.failure_counts[provider_name] >= self.failure_threshold:
            self.opened_at[provider_name] = time.time()
            logger.warning("Circuit breaker OPENED for provider '%s' (3 consecutive failures)", provider_name)


class ConcurrencyController:
    """Limits the number of concurrent consultations to maintain low latency SLAs."""

    def __init__(self, max_concurrent: int = 100) -> None:
        self.max_concurrent = max_concurrent
        self._semaphores: dict[int, asyncio.Semaphore] = {}
        self.active_count = 0

    def _get_semaphore(self) -> asyncio.Semaphore:
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
        except RuntimeError:
            loop_id = 0
        if loop_id not in self._semaphores:
            self._semaphores[loop_id] = asyncio.Semaphore(self.max_concurrent)
        return self._semaphores[loop_id]

    async def run(self, coro_func: Callable, *args, **kwargs) -> Any:
        sem = self._get_semaphore()
        async with sem:
            self.active_count += 1
            try:
                return await coro_func(*args, **kwargs)
            finally:
                self.active_count -= 1


# Global singletons
telemetry_cache = TelemetryCache(ttl_seconds=production_config.CACHE_TTL_SECONDS)
circuit_breaker = ProviderCircuitBreaker()
concurrency_controller = ConcurrencyController(max_concurrent=production_config.MAX_CONCURRENT_REQUESTS)
