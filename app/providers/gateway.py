"""Provider Gateway Subsystem for Inference.

Features:
- Global Key Pool with Round-Robin key rotation per provider.
- Per-Provider Rate Limiting (Token Bucket / Concurrency Limiter) with complete provider isolation.
- Automatic 60-second key blacklisting/quarantine on 429/503.
- Provider Health Tracking integration.
- Dynamic Capability-Based OpenRouter fallback.
"""

import asyncio
import threading
import time
from collections.abc import AsyncIterator
from typing import Any

from app.core.config import settings
from app.core.policies import ProviderSwitchingPolicy, SwitchReason
from app.providers.base import (
    ProviderRequest,
    ProviderResponse,
)
from app.providers.errors import (
    GatewayError,
    ProviderUnconfiguredError,
    RateLimitError,
    TemporaryUnavailableError,
    normalize_provider_exception,
)
from app.providers.health import provider_health_tracker
from app.utils.logger import logger

#: Providers that are legitimately able to serve a request with no credential at
#: all — self-hosted servers and transport shims that carry their own auth.
#: Everything else needs a key from ``settings`` before it can do anything.
KEYLESS_PROVIDERS: frozenset[str] = frozenset({"local", "litellm", "llamacpp", "vllm", "sglang", "ollama"})


def provider_has_credentials(prov_name: str) -> bool:
    """Return whether ``prov_name`` could possibly serve a request right now.

    Used as a pre-flight gate. Note that this asks only about *configured
    credentials*, not about health: an unhealthy-but-configured provider is still
    worth attempting (see the circuit-breaker semantics in
    ``ProviderHealthTracker``), whereas an unconfigured one can never answer.
    """
    normalized = prov_name.lower().strip()
    if normalized in KEYLESS_PROVIDERS:
        return True
    try:
        return bool(settings.get_provider_keys(normalized))
    except Exception:  # noqa: BLE001 - configuration lookup must never break routing
        return False


class KeyPool:
    """Manages round-robin key rotation and 60-second quarantines for a provider."""

    def __init__(self, provider_name: str, keys: list[str] | None = None, quarantine_seconds: float = 60.0) -> None:
        self.provider_name = provider_name.lower().strip()
        self._keys: list[str] = [k.strip() for k in (keys or []) if k.strip()]
        self._index: int = 0
        self._quarantined_until: dict[str, float] = {}  # key -> timestamp until quarantined
        self._quarantine_seconds = quarantine_seconds
        self._lock = threading.RLock()

    def set_keys(self, keys: list[str]) -> None:
        with self._lock:
            self._keys = [k.strip() for k in keys if k.strip()]
            self._index = 0

    @property
    def total_keys_count(self) -> int:
        with self._lock:
            return len(self._keys)

    def get_active_keys_count(self) -> int:
        with self._lock:
            now = time.monotonic()
            return sum(1 for k in self._keys if self._quarantined_until.get(k, 0) <= now)

    def get_quarantined_keys_count(self) -> int:
        with self._lock:
            now = time.monotonic()
            return sum(1 for k in self._keys if self._quarantined_until.get(k, 0) > now)

    def choose(self) -> str | None:
        """
        Returns the next available un-quarantined key in round-robin sequence.
        Fails closed: if all keys are quarantined, returns None (never hammers quarantined keys).
        """
        with self._lock:
            if not self._keys:
                return None
            now = time.monotonic()
            for _ in range(len(self._keys)):
                key = self._keys[self._index % len(self._keys)]
                self._index = (self._index + 1) % len(self._keys)
                if self._quarantined_until.get(key, 0) <= now:
                    return key
            return None

    async def get_next_key(self) -> str | None:
        return self.choose()

    def quarantine(self, key: str, duration_seconds: float | None = None) -> None:
        with self._lock:
            dur = duration_seconds if duration_seconds is not None else self._quarantine_seconds
            self._quarantined_until[key] = time.monotonic() + dur
            masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "***"
            logger.warning(
                "KEY POOL: Quarantining key '%s' on provider '%s' for %.0fs", masked_key, self.provider_name, dur
            )

    async def quarantine_key(self, key: str, duration_seconds: float = 60.0) -> None:
        self.quarantine(key, duration_seconds)

    def next_available_delay(self) -> float:
        with self._lock:
            if not self._keys:
                return float("inf")
            now = time.monotonic()
            delays = [self._quarantined_until.get(k, 0) - now for k in self._keys]
            return max(0.0, min(delays))


class ProviderRateLimiter:
    """Per-provider concurrency-safe token bucket and concurrency limiter."""

    def __init__(self, provider_name: str, requests_per_second: float = 5.0, max_concurrency: int = 4) -> None:
        self.provider_name = provider_name
        self.rate = max(requests_per_second, 0.001)
        self.capacity = float(max(max_concurrency * 2, 2))
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self.max_concurrency = max_concurrency
        self._lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(max_concurrency)

    async def acquire(self, cost: float = 1.0) -> None:
        """Acquire a concurrency slot and a rate-limit token, without leaking either.

        Runaway-safe: this method takes a concurrency slot *first* and only then waits
        for a token, because the wait must not happen while the bucket lock is held. That
        ordering creates a window in which the caller holds a slot but has not yet been
        handed the limiter — and if the task is cancelled inside that window the slot
        would never come back. Cancellation there is not exotic: every caller in this
        module wraps a provider call in ``asyncio.wait_for`` and a disconnecting client
        cancels the whole handler task, so the interruption lands on whatever await point
        the task happens to be at — frequently this one, precisely because the bucket is
        exhausted when load is high.

        Measured cost of the leak: one cancelled waiter left ``max_concurrency`` slots
        permanently reduced; the next caller blocked forever, and with four cancellations
        the provider became permanently unavailable while still appearing healthy. The
        ``finally`` below returns the slot on every path that does not hand it to the
        caller, which makes cancellation safe to occur at any point.
        """
        await self._sem.acquire()
        handed_over = False
        try:
            while True:
                delay = 0.0
                async with self._lock:
                    now = time.monotonic()
                    self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                    self.updated = now
                    if self.tokens >= cost:
                        self.tokens -= cost
                        handed_over = True
                        return
                    delay = (cost - self.tokens) / self.rate
                # Never hold lock while sleeping!
                await asyncio.sleep(delay)
        finally:
            if not handed_over:
                # Cancelled, timed out, or otherwise interrupted while waiting for a
                # token: the slot is ours to give back, and nobody else can do it —
                # the caller never received the limiter, so its own ``finally`` (which
                # calls release()) is never reached.
                self._sem.release()

    def release(self) -> None:
        """Release the concurrency semaphore slot."""
        self._sem.release()


class ModelGateway:
    """
    Central gateway coordinating LLM calls with:
    - Per-provider rate limiting & isolation.
    - Round-robin key rotation with 60s quarantine on 429/503.
    - Provider health tracking.
    - Dynamic capability-based OpenRouter fallback.
    """

    def __init__(self) -> None:
        self.key_pools: dict[str, KeyPool] = {}
        self.rate_limiters: dict[str, ProviderRateLimiter] = {}
        self.health_tracker = provider_health_tracker
        self._initialize_pools()

    def _initialize_pools(self) -> None:
        """Initialize key pools and rate limiters for all known providers."""
        all_providers = ["gemini", "groq", "mistral", "openrouter", "cohere", "huggingface", "nvidia"]
        if settings.LOCAL_ENABLED:
            # Self-hosted tier: keyless, unmetered, and therefore the correct final
            # rung of the fallback ladder. Registered only when enabled so that a
            # deployment with no local server never pays a connection-refused
            # penalty on every fallback attempt.
            all_providers.append("local")
        for prov in all_providers:
            keys = settings.get_provider_keys(prov)
            self.key_pools[prov] = KeyPool(prov, keys)
            # Default rate limits per provider
            if prov == "local":
                # A local server has finite compute slots but **no quota**, so the
                # limiting factor must be concurrency, never rate. An earlier revision
                # set this to 10 requests/second by analogy with a cloud vendor; because
                # one panel request fans out to 8+ model calls, the token bucket drained
                # after two requests and every subsequent request ran at 1/rate per call.
                # Measured: p50 latency stepped from 0.11s to 0.80s and stayed there, so
                # the zero-cost tier became the slowest tier in the system.
                self.rate_limiters[prov] = ProviderRateLimiter(
                    prov,
                    requests_per_second=float(settings.LOCAL_MAX_RPS),
                    max_concurrency=int(settings.LOCAL_MAX_CONCURRENCY),
                )
                continue
            rpm = 10.0 if prov in ("gemini", "cohere") else 20.0
            self.rate_limiters[prov] = ProviderRateLimiter(prov, requests_per_second=rpm / 60.0, max_concurrency=4)

    def refresh_keys(self) -> None:
        """Reload keys from environment/settings."""
        for prov, pool in self.key_pools.items():
            keys = settings.get_provider_keys(prov)
            pool.set_keys(keys)

    def get_provider_health(self, provider_name: str) -> Any:
        """Return live health metrics for a provider."""
        pool = self.key_pools.get(provider_name.lower())
        if pool:
            self.health_tracker.update_key_counts(
                provider_name,
                active_count=pool.get_active_keys_count(),
                quarantined_count=pool.get_quarantined_keys_count(),
            )
        return self.health_tracker.get_provider_health(provider_name)

    async def execute(
        self, provider_name: str, request: ProviderRequest, capability: str = "general", stage_name: str = "general"
    ) -> ProviderResponse:
        """
        Executes an LLM request through the gateway:
        1. Selects key from round-robin pool with fail-closed quarantine semantics.
        2. Applies isolated per-provider rate limiter without holding locks during sleep.
        3. Enforces an overall request deadline across primary retries and fallbacks.
        4. Retries across alternate keys only for transient errors (429/503/timeout).
        5. If all keys exhausted/quarantined, routes through capability-matched fallback.
        6. Updates health metrics and records provenance.
        """
        timeout_budget = float(request.extra_params.get("timeout", settings.REQUEST_TIMEOUT or 60.0))
        deadline = time.monotonic() + timeout_budget
        # Reserve the tail of the budget for the fallback ladder. Without this, a
        # primary provider that times out (the single most common provider failure)
        # consumes the entire deadline, `_execute_dynamic_fallback` finds no time
        # remaining, and re-raises the original error — so the ladder is unreachable
        # precisely when it is needed. The reservation is what turns "fallbacks are
        # configured" into "fallbacks actually run".
        primary_deadline = time.monotonic() + (timeout_budget * settings.PRIMARY_ATTEMPT_FRACTION)

        prov_name = provider_name.lower().strip()
        if prov_name == "litellm":
            from app.providers.gateway_litellm import execute_via_litellm

            start_time = time.perf_counter()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Request deadline ({timeout_budget:.1f}s) exceeded before LiteLLM call.")
            try:
                resp = await asyncio.wait_for(execute_via_litellm(request), timeout=remaining)
                latency = time.perf_counter() - start_time
                self.health_tracker.record_success("litellm", latency)
                return resp
            except Exception as exc:
                latency = time.perf_counter() - start_time
                typed_err = normalize_provider_exception(exc, provider="litellm", model=request.model)
                self.health_tracker.record_failure("litellm", str(exc), latency_seconds=latency)
                raise typed_err from exc

        # Air-gapped / offline mode. When LOCAL_PREFERRED is set, the self-hosted tier
        # answers first and no outbound provider call is attempted at all — which is
        # what makes this usable as a privacy control (nothing leaves the machine) and
        # as an offline development mode, not merely as a cost optimisation.
        local_tier_attempted = False
        if settings.LOCAL_PREFERRED and settings.LOCAL_ENABLED and prov_name != "local":
            local_tier_attempted = True
            local_resp = await self._execute_local_tier(prov_name, request, capability, None, deadline)
            if local_resp is not None:
                return local_resp
            logger.warning(
                "GATEWAY: LOCAL_PREFERRED is enabled but the self-hosted tier failed; "
                "falling through to cloud providers. Set LOCAL_ENABLED only when a local "
                "server is actually running if you need a hard guarantee of no egress."
            )

        # Pre-flight credential gate. An unconfigured provider cannot succeed, so the
        # only effects of attempting it are (a) latency, and (b) a bogus failure
        # recorded against that provider's health — which is how a fresh deployment
        # with one working provider ends up believing that all of them are down.
        # Raising here, before the rate limiter, keeps the shared per-provider queue
        # free for calls that can actually be answered.
        if not provider_has_credentials(prov_name):
            raise ProviderUnconfiguredError(
                f"Provider '{prov_name}' has no configured credential; skipping without an attempt.",
                provider=prov_name,
                model=request.model,
            )

        pool = self.key_pools.get(prov_name)
        if not pool or pool.total_keys_count == 0:
            keys = settings.get_provider_keys(prov_name)
            if keys:
                if not pool:
                    pool = KeyPool(prov_name, keys)
                    self.key_pools[prov_name] = pool
                else:
                    pool.set_keys(keys)

        limiter = self.rate_limiters.get(prov_name)
        if not limiter:
            limiter = ProviderRateLimiter(prov_name)
            self.rate_limiters[prov_name] = limiter

        # Fail closed immediately if all keys are quarantined
        last_error: GatewayError | Exception | None = None
        if pool and pool.total_keys_count > 0 and pool.get_active_keys_count() == 0:
            logger.warning(
                "KEY POOL: All %d keys for '%s' are quarantined. Failing closed.", pool.total_keys_count, prov_name
            )
            last_error = TemporaryUnavailableError(
                f"All credentials for provider '{prov_name}' are currently quarantined.", provider=prov_name
            )
            attempts = 0
        else:
            attempts = max(1, pool.total_keys_count if pool else 1)
            last_error = None

        current_key = None

        for _ in range(attempts):
            remaining = primary_deadline - time.monotonic()
            if remaining <= 0:
                # The primary slice is spent. Escalate to the ladder rather than
                # raising: the fallback budget is still intact by construction.
                last_error = last_error or TemporaryUnavailableError(
                    f"Primary attempt slice ({timeout_budget * settings.PRIMARY_ATTEMPT_FRACTION:.1f}s of "
                    f"{timeout_budget:.1f}s) exhausted for provider '{prov_name}' before a call could be made.",
                    provider=prov_name,
                )
                logger.warning(
                    "GATEWAY: Primary attempt slice exhausted for '%s' before a call was made; "
                    "escalating to the fallback ladder.",
                    prov_name,
                )
                break

            current_key = pool.choose() if pool else None
            if pool and pool.total_keys_count > 0 and current_key is None:
                # No active key available
                break

            start_time = time.perf_counter()

            try:
                import app.providers

                provider_instance = (
                    app.providers.get_provider(prov_name, api_key=current_key)
                    if current_key
                    else app.providers.get_provider(prov_name)
                )

                # Acquire isolated rate limiter slot
                await limiter.acquire()
                try:
                    call_timeout = min(remaining, timeout_budget)
                    resp = await asyncio.wait_for(provider_instance.generate(request), timeout=call_timeout)
                finally:
                    limiter.release()

                latency = time.perf_counter() - start_time
                self.health_tracker.record_success(prov_name, latency)
                return resp

            except (asyncio.TimeoutError, TimeoutError) as exc:
                # A timeout is the most likely provider failure and must NOT abort the
                # request: six other providers may be perfectly healthy. Record it,
                # then break out of the key loop so the code below escalates to
                # _execute_dynamic_fallback with the reserved budget. Retrying the
                # same provider on another key would not help — a timeout is a
                # property of the provider or the prompt, not of the credential.
                latency = time.perf_counter() - start_time
                self.health_tracker.record_failure(
                    prov_name,
                    f"timeout after {latency:.2f}s",
                    is_503=True,
                    latency_seconds=latency,
                )
                last_error = TimeoutError(
                    f"Request deadline exceeded for provider '{prov_name}' after {latency:.1f}s"
                )
                logger.warning(
                    "GATEWAY: Provider '%s' timed out after %.1fs (%s, model=%s). Escalating to the "
                    "fallback ladder instead of failing the request.",
                    prov_name,
                    latency,
                    type(exc).__name__,
                    request.model,
                )
                break
            except Exception as exc:
                latency = time.perf_counter() - start_time
                typed_err = normalize_provider_exception(exc, provider=prov_name, model=request.model)
                last_error = typed_err
                err_str = str(exc)
                is_429 = isinstance(typed_err, RateLimitError)
                is_503 = isinstance(typed_err, TemporaryUnavailableError)

                self.health_tracker.record_failure(
                    prov_name, err_str, is_429=is_429, is_503=is_503, latency_seconds=latency
                )

                if pool and current_key and typed_err.is_retryable():
                    pool.quarantine(current_key, duration_seconds=60.0)
                    logger.warning(
                        "GATEWAY: Provider '%s' [%s] retrying with next key in pool. Detail: %s",
                        prov_name,
                        type(typed_err).__name__,
                        err_str.split("\n")[0],
                    )
                    remaining = primary_deadline - time.monotonic()
                    if remaining > 0.5:
                        await asyncio.sleep(min(0.5, remaining))
                    continue
                else:
                    break

        # Primary provider failed on all keys -> Fallback logic
        logger.warning(
            "GATEWAY: Primary provider '%s' failed. Initiating capability fallback for '%s'.", prov_name, capability
        )

        fallback_resp = await self._execute_dynamic_fallback(
            failed_provider=prov_name,
            request=request,
            capability=capability,
            stage_name=stage_name,
            last_error=last_error,
            deadline=deadline,
            skip_local_tier=local_tier_attempted,
        )
        return fallback_resp

    async def _execute_local_tier(
        self,
        failed_provider: str,
        request: ProviderRequest,
        capability: str,
        last_error: Exception | None,
        deadline: float | None,
    ) -> ProviderResponse | None:
        """Attempt the self-hosted tier, returning ``None`` instead of raising on failure.

        Two callers share this one implementation so the local tier cannot drift:

        * the **terminal rung of the fallback ladder** — so a request survives total
          cloud failure when a local server is available, and
        * **air-gapped mode** (``LOCAL_PREFERRED``) — where it is the *first* rung and
          the operator intends no outbound provider call at all.

        Returning ``None`` (rather than raising) lets the caller decide the
        escalation order without a try/except at every call site.
        """
        if not settings.LOCAL_ENABLED or failed_provider == "local":
            return None
        remaining = (deadline - time.monotonic()) if deadline else float(settings.REQUEST_TIMEOUT or 60.0)
        if remaining <= 0:
            logger.warning("GATEWAY: No budget left for the self-hosted tier; skipping it.")
            return None

        try:
            import app.providers

            local_prov = app.providers.get_provider("local")
            local_limiter = self.rate_limiters.get("local")
            if not local_limiter:
                # Same sizing rationale as ``_initialize_pools``: concurrency bounds the
                # host, rate must not bound throughput.
                local_limiter = ProviderRateLimiter(
                    "local",
                    requests_per_second=float(settings.LOCAL_MAX_RPS),
                    max_concurrency=int(settings.LOCAL_MAX_CONCURRENCY),
                )
                self.rate_limiters["local"] = local_limiter

            local_req = ProviderRequest(
                messages=request.messages,
                system_instruction=request.system_instruction,
                model=request.model or settings.LOCAL_MODEL,
                temperature=request.temperature,
                max_tokens=request.max_tokens or 1024,
                response_schema=request.response_schema,
            )
            await local_limiter.acquire()
            try:
                rem = (deadline - time.monotonic()) if deadline else remaining
                if rem <= 0:
                    raise TimeoutError("Deadline exceeded before the self-hosted tier call.")
                resp = await asyncio.wait_for(local_prov.generate(local_req), timeout=rem)
            finally:
                local_limiter.release()

            if not resp.raw_response:
                resp.raw_response = {}
            resp.raw_response["fallback_provenance"] = {
                "requested_provider": failed_provider,
                "requested_model": request.model,
                "actual_provider": "local",
                "actual_model": local_req.model,
                "capability": capability,
                "fallback_reason": str(last_error) if last_error else "local_preferred",
            }
            self.health_tracker.record_success("local", resp.latency_seconds)
            logger.info(
                "GATEWAY: Served capability '%s' from the self-hosted tier (model=%s, %.2fs).",
                capability,
                local_req.model,
                resp.latency_seconds,
            )
            return resp
        except Exception as local_exc:  # noqa: BLE001 - the tier must never abort the request
            self.health_tracker.record_failure("local", str(local_exc))
            logger.error("GATEWAY: Self-hosted tier failed: %s", str(local_exc))
            return None

    async def _execute_dynamic_fallback(
        self,
        failed_provider: str,
        request: ProviderRequest,
        capability: str,
        stage_name: str,
        last_error: Exception | None,
        deadline: float | None = None,
        skip_local_tier: bool = False,
    ) -> ProviderResponse:
        """Executes fallback with dynamic capability matching, rate-limiting, and deadline bounds.

        ``skip_local_tier`` is set by :meth:`execute` when the self-hosted tier was
        already tried as the first choice under ``LOCAL_PREFERRED``. Re-trying it at
        the end of the ladder doubles its failure count against the health tracker —
        which shortens the circuit breaker for the one provider that is supposed to
        keep the system alive — and adds a second round trip for nothing.
        """
        # 1. If failed provider is NOT openrouter, use OpenRouter with dynamic capability model discovery
        remaining = (deadline - time.monotonic()) if deadline else 60.0
        if remaining <= 0:
            if last_error:
                raise last_error
            raise TimeoutError("Request deadline exceeded before fallback execution.")

        if failed_provider != "openrouter":
            openrouter_key: str | None = None
            openrouter_pool = self.key_pools.get("openrouter")
            try:
                import app.providers

                configured_keys = settings.get_provider_keys("openrouter")
                if openrouter_pool is None:
                    openrouter_pool = KeyPool("openrouter", configured_keys)
                    self.key_pools["openrouter"] = openrouter_pool
                elif openrouter_pool.total_keys_count == 0 and configured_keys:
                    openrouter_pool.set_keys(configured_keys)

                if openrouter_pool.total_keys_count:
                    openrouter_key = openrouter_pool.choose()
                    if openrouter_key is None:
                        raise TemporaryUnavailableError(
                            "All OpenRouter credentials are currently quarantined.",
                            provider="openrouter",
                        )
                    openrouter_prov = app.providers.get_provider(
                        "openrouter", api_key=openrouter_key
                    )
                else:
                    # Some free OpenRouter routes permit unauthenticated calls.
                    openrouter_prov = app.providers.get_provider("openrouter")
                if hasattr(openrouter_prov, "get_best_free_model"):
                    dynamic_model = await openrouter_prov.get_best_free_model(capability)
                elif hasattr(openrouter_prov, "find_model_by_capability"):
                    dynamic_model = await openrouter_prov.find_model_by_capability(capability)
                else:
                    dynamic_model = "nvidia/nemotron-3.5-lightning:free"

                logger.info(
                    "GATEWAY FALLBACK: Routed to OpenRouter (Dynamic Model: %s) for capability '%s'",
                    dynamic_model,
                    capability,
                )

                fallback_req = ProviderRequest(
                    messages=request.messages,
                    system_instruction=request.system_instruction,
                    model=dynamic_model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens or 1024,
                    response_schema=request.response_schema,
                    extra_params=request.extra_params,
                )

                openrouter_limiter = self.rate_limiters.get("openrouter")
                if not openrouter_limiter:
                    openrouter_limiter = ProviderRateLimiter("openrouter")
                    self.rate_limiters["openrouter"] = openrouter_limiter

                await openrouter_limiter.acquire()
                try:
                    rem = (deadline - time.monotonic()) if deadline else 60.0
                    if rem <= 0:
                        raise TimeoutError("Deadline exceeded waiting for OpenRouter rate limit.")
                    resp = await asyncio.wait_for(openrouter_prov.generate(fallback_req), timeout=rem)
                finally:
                    openrouter_limiter.release()

                # Attach fallback provenance
                if not resp.raw_response:
                    resp.raw_response = {}
                resp.raw_response["fallback_provenance"] = {
                    "requested_provider": failed_provider,
                    "requested_model": request.model,
                    "actual_provider": "openrouter",
                    "actual_model": dynamic_model,
                    "capability": capability,
                    "fallback_reason": str(last_error) if last_error else "primary_exhausted",
                }

                self.health_tracker.record_success("openrouter", resp.latency_seconds)
                return resp

            except Exception as fb_exc:
                typed_error = normalize_provider_exception(
                    fb_exc, provider="openrouter", model=request.model
                )
                self.health_tracker.record_failure(
                    "openrouter",
                    type(typed_error).__name__,
                    is_429=isinstance(typed_error, RateLimitError),
                    is_503=isinstance(typed_error, TemporaryUnavailableError),
                )
                if openrouter_key and typed_error.is_retryable() and openrouter_pool:
                    openrouter_pool.quarantine(openrouter_key, duration_seconds=60.0)
                logger.error(
                    "OpenRouter dynamic fallback failed (%s)", type(typed_error).__name__
                )

        # 2. Check standard policy matrix fallback as second safeguard
        remaining = (deadline - time.monotonic()) if deadline else 60.0
        if remaining > 0:
            fallback_route = ProviderSwitchingPolicy.get_fallback_provider(
                failed_provider, SwitchReason.TIMEOUT, stage=stage_name
            )
            if fallback_route and fallback_route.fallback_provider != failed_provider:
                try:
                    import app.providers

                    sec_name = fallback_route.fallback_provider
                    if not provider_has_credentials(sec_name):
                        # Same reasoning as the primary pre-flight gate: a policy-matrix
                        # route to a provider nobody configured is not a fallback, it is
                        # a delay. Skip it without touching the limiter or health state.
                        logger.warning(
                            "GATEWAY FALLBACK: Skipping secondary provider '%s' — no credential configured.",
                            sec_name,
                        )
                        raise ProviderUnconfiguredError(
                            f"Secondary fallback provider '{sec_name}' has no configured credential.",
                            provider=sec_name,
                        )
                    sec_prov = app.providers.get_provider(sec_name)
                    sec_limiter = self.rate_limiters.get(sec_name)
                    if not sec_limiter:
                        sec_limiter = ProviderRateLimiter(sec_name)
                        self.rate_limiters[sec_name] = sec_limiter

                    sec_req = ProviderRequest(
                        messages=request.messages,
                        system_instruction=request.system_instruction,
                        model=fallback_route.fallback_model,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens or 1024,
                    )
                    await sec_limiter.acquire()
                    try:
                        rem = (deadline - time.monotonic()) if deadline else 60.0
                        if rem <= 0:
                            raise TimeoutError(f"Deadline exceeded for secondary fallback {sec_name}.")
                        resp = await asyncio.wait_for(sec_prov.generate(sec_req), timeout=rem)
                    finally:
                        sec_limiter.release()

                    if not resp.raw_response:
                        resp.raw_response = {}
                    resp.raw_response["fallback_provenance"] = {
                        "requested_provider": failed_provider,
                        "requested_model": request.model,
                        "actual_provider": fallback_route.fallback_provider,
                        "actual_model": fallback_route.fallback_model,
                        "capability": capability,
                        "fallback_reason": str(last_error) if last_error else "primary_exhausted",
                    }
                    self.health_tracker.record_success(sec_name, resp.latency_seconds)
                    return resp
                except Exception as sec_exc:
                    logger.error(
                        "Secondary fallback provider '%s' failed: %s", fallback_route.fallback_provider, str(sec_exc)
                    )

        # 3. Self-hosted terminal tier. If a local server is running it has no quota,
        #    no key, and no network dependency, which makes it the most reliable rung
        #    of the ladder — but also usually the slowest, so it is tried after the
        #    free cloud peers and before LiteLLM (which itself needs cloud credentials).
        if skip_local_tier:
            logger.info(
                "GATEWAY FALLBACK: Self-hosted tier already attempted as primary choice; "
                "not repeating it at the end of the ladder."
            )
        else:
            local_resp = await self._execute_local_tier(
                failed_provider, request, capability, last_error, deadline
            )
            if local_resp is not None:
                return local_resp

        # 4. Optional LiteLLM unified transport fallback if enabled
        remaining = (deadline - time.monotonic()) if deadline else 60.0
        if (
            remaining > 0
            and settings.INFERENCE_LITELLM_FALLBACK_ENABLED
            and settings.INFERENCE_LITELLM_ENABLED
            and failed_provider != "litellm"
        ):
            try:
                from app.providers.gateway_litellm import execute_via_litellm

                litellm_resp = await asyncio.wait_for(execute_via_litellm(request), timeout=remaining)
                if not litellm_resp.raw_response:
                    litellm_resp.raw_response = {}
                litellm_resp.raw_response["fallback_provenance"] = {
                    "requested_provider": failed_provider,
                    "requested_model": request.model,
                    "actual_provider": "litellm",
                    "actual_model": litellm_resp.model,
                    "capability": capability,
                    "fallback_reason": str(last_error) if last_error else "primary_exhausted",
                }
                self.health_tracker.record_success("litellm", litellm_resp.latency_seconds)
                return litellm_resp
            except Exception as litellm_exc:
                logger.error("LiteLLM fallback failed: %s", str(litellm_exc))

        # If all fallbacks failed, raise the original error
        if last_error:
            raise last_error
        raise GatewayError(f"Provider {failed_provider} and all fallback routes failed.", provider=failed_provider)

    async def execute_speculative(
        self,
        providers: list[str],
        request: ProviderRequest,
        stage_name: str = "speculative_race",
        timeout: float = 10.0,
    ) -> ProviderResponse:
        """
        Ultra-low latency speculative racing:
        Launches parallel requests to specified providers concurrently.
        Returns the first successful response and cancels remaining slower tasks.
        """
        if not providers:
            raise GatewayError("No providers provided for speculative race.")
        if len(providers) == 1:
            request.extra_params["timeout"] = timeout
            return await self.execute(providers[0], request, stage_name=stage_name)

        start_time = time.monotonic()

        async def _call_provider(prov: str) -> tuple[str, ProviderResponse]:
            prov_req = request
            # Auto-select appropriate model for each provider if needed
            if prov == "groq" and (not request.model or "gemini" in request.model):
                prov_req = ProviderRequest(
                    messages=request.messages,
                    system_instruction=request.system_instruction,
                    model="openai/gpt-oss-120b",
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    extra_params={"timeout": timeout},
                )
            elif prov == "gemini" and (not request.model or "groq" in request.model or "gpt" in request.model):
                prov_req = ProviderRequest(
                    messages=request.messages,
                    system_instruction=request.system_instruction,
                    model="gemini-3.8-flash",
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    extra_params={"timeout": timeout},
                )
            else:
                prov_req.extra_params["timeout"] = timeout
            resp = await self.execute(prov, prov_req, stage_name=stage_name)
            return prov, resp

        tasks = [asyncio.create_task(_call_provider(p)) for p in providers]
        first_error: Exception | None = None

        while tasks:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                try:
                    winning_prov, winning_resp = d.result()
                    for p in pending:
                        p.cancel()
                    elapsed = time.monotonic() - start_time
                    # The candidate LABEL that returned first is not necessarily the
                    # provider that produced the text. When a raced candidate has no
                    # usable credential (or the self-hosted tier is preferred) the call is
                    # served locally, yet callers were told the winning *label* answered:
                    # an instant request served by the local rig reported
                    # provider="groq"/model="gemini-3.8-flash". Recording the true
                    # serving provider alongside the label is what makes the race
                    # metadata usable for cost attribution instead of misleading.
                    served_by = winning_resp.provider or winning_prov
                    logger.info(
                        "SPECULATIVE RACE: '%s' returned first among %s in %.3fs; served_by=%s",
                        winning_prov,
                        providers,
                        elapsed,
                        served_by,
                    )
                    if not winning_resp.raw_response:
                        winning_resp.raw_response = {}
                    winning_resp.raw_response["speculative_race"] = {
                        "winner": winning_prov,
                        "served_by": served_by,
                        "fell_back": served_by != winning_prov,
                        "competitors": providers,
                        "elapsed_seconds": round(elapsed, 4),
                    }
                    return winning_resp
                except Exception as exc:
                    logger.warning("SPECULATIVE RACE: Provider attempt failed: %s", exc)
                    first_error = exc
            tasks = list(pending)

        if first_error:
            raise first_error
        raise GatewayError("All speculative race candidates failed.")

    async def stream(
        self,
        provider: str,
        request: ProviderRequest,
        stage_name: str = "general_stream",
    ) -> AsyncIterator[str]:
        """Stream token chunks directly from the requested provider adapter."""
        import app.providers

        prov_name = provider.lower().strip()
        pool = self.key_pools.get(prov_name)
        current_key = pool.choose() if pool else None

        provider_instance = (
            app.providers.get_provider(prov_name, api_key=current_key)
            if current_key
            else app.providers.get_provider(prov_name)
        )
        async for chunk in provider_instance.stream(request):
            yield chunk


# Global default gateway instance
model_gateway = ModelGateway()
