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
from .errors import ProviderUnavailable, SchemaViolation
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

    def _cache_key(self, request: CompletionRequest) -> str | None:
        if not request.cacheable or request.stream:
            return None
        return self.cache.key(
            request.tenant_id,
            {
                "model": request.model,
                "messages": [m.__dict__ for m in request.messages],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "schema": request.response_schema,
                "capabilities": sorted(capability.value for capability in request.capabilities),
                "extra": request.extra,
            },
        )

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.trace.emit("admission", request.request_id, model=request.model, tenant=request.tenant_id)
        if request.stream:
            raise ValueError("Use stream() for streaming requests")

        decision = self.router.choose(request)
        ep = decision.endpoint
        cache_key = self._cache_key(request)

        # A cache hit does not spend provider budget and therefore must not reserve
        # money. Validate cached values just as strictly as fresh provider responses:
        # stale or malformed entries are evicted and recomputed rather than trusted.
        if cache_key:
            hit = self.cache.get(cache_key)
            if hit is not None:
                try:
                    cached_fields = dict(hit)
                    cached_fields["cached"] = True
                    cached_result = CompletionResult(**cached_fields)
                    if request.response_schema:
                        validate_json(cached_result.text, dict(request.response_schema))
                except (TypeError, ValueError, SchemaViolation):
                    self.cache.delete(cache_key)
                else:
                    self.trace.emit("cache_hit", request.request_id, ep.provider, ep.model)
                    return cached_result

        estimate = self._estimate_cost(request, ep)
        self.budget.reserve(request.tenant_id, estimate)
        reservation_held = True
        try:
            # Caller-supplied request IDs are not globally unique. Include the tenant
            # and full request payload so a reused ID cannot make one tenant or prompt
            # receive another request's in-flight result.
            dedupe_key = self.dedupe.fingerprint(
                {
                    "tenant_id": request.tenant_id,
                    "request": request.__dict__,
                }
            )
            result = await self.dedupe.run(dedupe_key, lambda: self._execute(request, ep))

            # Never persist a result or finalize spend until the response contract has
            # been validated. A malformed answer must not poison subsequent cache hits.
            if request.response_schema:
                validate_json(result.text, dict(request.response_schema))

            self.budget.reconcile(request.tenant_id, estimate, result.cost_usd)
            reservation_held = False

            if cache_key:
                self.cache.put(cache_key, result.__dict__, request.tenant_id, self.config.cache_ttl_seconds)
            self.trace.emit(
                "completed",
                request.request_id,
                ep.provider,
                ep.model,
                latency=result.latency_seconds,
                tokens=result.total_tokens,
                cost=result.cost_usd,
            )
            return result
        except BaseException:
            # Cancellation is a BaseException in supported Python versions. Refund a
            # pending reservation for cancellation as well as ordinary failures.
            if reservation_held:
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
