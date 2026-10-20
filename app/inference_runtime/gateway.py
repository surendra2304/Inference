from __future__ import annotations

import asyncio
import time
from typing import Protocol

from app.security.prompt_isolation import scrub_credentials

from .budget import BudgetLedger
from .cache import TenantCache
from .circuit import CircuitBreaker
from .config import RuntimeConfig
from .contracts import CompletionRequest, CompletionResult, ProviderEndpoint
from .dedupe import RequestDeduper
from .errors import ProviderUnavailable
from .health import ProviderHealth
from .key_pool import SafeKeyPool
from .retry import with_retries
from .router import CapabilityRouter
from .structured import validate_json
from .telemetry import TraceRecorder


class ProviderTransport(Protocol):
    async def complete(self, request: CompletionRequest, endpoint: ProviderEndpoint, api_key: str | None): ...
    async def health(self) -> bool: ...


class HardenedGateway:
    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()
        self.health = ProviderHealth()
        self.router = CapabilityRouter(self.health)
        self.budget = BudgetLedger(self.config.default_budget_usd)
        self.cache = TenantCache()
        self.dedupe = RequestDeduper()
        self.trace = TraceRecorder()
        self.transports: dict[str, ProviderTransport] = {}
        self.keys: dict[str, SafeKeyPool] = {}
        self.breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    def register(self, endpoint: ProviderEndpoint, transport: ProviderTransport, keys: list[str] | None = None) -> None:
        self.router.register(endpoint)
        self.transports[endpoint.provider] = transport
        self.keys[endpoint.provider] = SafeKeyPool(endpoint.provider, keys)
        self.breakers.setdefault(endpoint.provider, CircuitBreaker())

    def _estimate_cost(self, request: CompletionRequest, endpoint: ProviderEndpoint) -> float:
        chars = sum(len(m.content) if isinstance(m.content, str) else len(str(m.content)) for m in request.messages)
        prompt = max(1, chars // 4)
        out = request.max_tokens or 512
        return (prompt / 1000) * endpoint.cost_per_1k_input + (out / 1000) * endpoint.cost_per_1k_output

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.trace.emit("admission", request.request_id, model=request.model, tenant=request.tenant_id)
        decision = self.router.choose(request)
        ep = decision.endpoint
        estimate = self._estimate_cost(request, ep)
        self.budget.reserve(request.tenant_id, estimate)
        try:
            cache_key = (
                self.cache.key(
                    request.tenant_id,
                    {
                        "model": request.model,
                        "messages": [m.__dict__ for m in request.messages],
                        "temperature": request.temperature,
                        "max_tokens": request.max_tokens,
                        "schema": request.response_schema,
                    },
                )
                if request.cacheable and not request.stream
                else None
            )
            if cache_key:
                hit = self.cache.get(cache_key)
                if hit is not None:
                    self.trace.emit("cache_hit", request.request_id, ep.provider, ep.model)
                    return CompletionResult(**hit, cached=True)
            if request.stream:
                raise ValueError("Use stream() for streaming requests")
            result = await self.dedupe.run(
                request.request_id or self.dedupe.fingerprint(request.__dict__), lambda: self._execute(request, ep)
            )
            actual = result.cost_usd
            self.budget.reconcile(request.tenant_id, estimate, actual)
            if cache_key:
                self.cache.put(cache_key, result.__dict__, request.tenant_id, self.config.cache_ttl_seconds)
            self.trace.emit(
                "completed",
                request.request_id,
                ep.provider,
                ep.model,
                latency=result.latency_seconds,
                tokens=result.total_tokens,
                cost=actual,
            )
            if request.response_schema:
                validate_json(result.text, dict(request.response_schema))
            return result
        except Exception:
            self.budget.reconcile(request.tenant_id, estimate, 0.0)
            raise

    async def _execute(self, request: CompletionRequest, ep: ProviderEndpoint) -> CompletionResult:
        provider = ep.provider
        transport = self.transports.get(provider)
        breaker = self.breakers.get(provider)
        if transport is None:
            raise ProviderUnavailable(f"No transport registered for {provider}")
        if breaker and not breaker.allow():
            raise ProviderUnavailable(f"Circuit open for {provider}")
        pool = self.keys.get(provider)
        key = pool.choose() if pool else None

        async def call():
            started = time.perf_counter()
            try:
                raw = await asyncio.wait_for(
                    transport.complete(request, ep, key),
                    timeout=request.timeout_seconds or self.config.request_timeout_seconds,
                )
                latency = time.perf_counter() - started
                self.health.success(provider, latency)
                self.router.record(ep, True, latency)
                breaker and breaker.success()
                return raw
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                latency = time.perf_counter() - started
                scrubbed = scrub_credentials(str(exc))
                self.health.failure(provider, scrubbed, latency)
                self.router.record(ep, False, latency)
                breaker and breaker.failure()
                pool and key and pool.quarantine(key)
                raise

        return await with_retries(call, self.config.max_retries)
