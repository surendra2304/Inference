"""Multi-Tenant Isolation, API Governance, Row-Level Tenant Security, and Deduplication."""

import secrets
import time
from typing import Any

from pydantic import BaseModel, Field

from app.utils.bounded_store import BoundedStore
from app.utils.logger import logger


class TenantPolicy(BaseModel):
    tenant_id: str
    name: str
    rate_limit_per_hour: int = 5000
    daily_budget_usd: float = 999999.0
    current_daily_spend_usd: float = 0.0
    hard_cutoff_enabled: bool = False
    active_keys: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)


class MultiTenantManager:
    """Manages tenant isolation, API key rotation, budget cutoffs, and request deduplication."""

    def __init__(self) -> None:
        self.tenants: dict[str, TenantPolicy] = {
            "tenant_trading": TenantPolicy(
                tenant_id="tenant_trading",
                name="Algorithmic Trading Bot",
                rate_limit_per_hour=5000,
                daily_budget_usd=999999.0,
                active_keys=["key_trading_live_01", "key_trading_test_02"]
            ),
            "tenant_forge": TenantPolicy(
                tenant_id="tenant_forge",
                name="FORGE Code Engine",
                rate_limit_per_hour=5000,
                daily_budget_usd=999999.0,
                active_keys=["key_forge_prod_01"]
            ),
            "tenant_nexus": TenantPolicy(
                tenant_id="tenant_nexus",
                name="Nexus Decision Subsystem",
                rate_limit_per_hour=5000,
                daily_budget_usd=999999.0,
                active_keys=["key_nexus_prod_01"]
            ),
            "tenant_sentinel": TenantPolicy(
                tenant_id="tenant_sentinel",
                name="Sentinel Security Engine",
                rate_limit_per_hour=5000,
                daily_budget_usd=999999.0,
                active_keys=["key_sentinel_prod_01"]
            ),
            "tenant_intelx": TenantPolicy(
                tenant_id="tenant_intelx",
                name="IntelX Research Engine",
                rate_limit_per_hour=5000,
                daily_budget_usd=999999.0,
                active_keys=["key_intelx_prod_01"]
            ),
            "tenant_futuris": TenantPolicy(
                tenant_id="tenant_futuris",
                name="Futuris Forecasting Engine",
                rate_limit_per_hour=5000,
                daily_budget_usd=999999.0,
                active_keys=["key_futuris_prod_01"]
            ),
            "tenant_default": TenantPolicy(
                tenant_id="tenant_default",
                name="Default Sandbox Tenant",
                rate_limit_per_hour=5000,
                daily_budget_usd=999999.0,
                active_keys=["key_default_sandbox"]
            )
        }
        # In-memory deduplication cache: namespaced-key -> (response_payload, expiry).
        #
        # The key is f"{namespace}\x00{request_id}". It used to be the bare
        # caller-supplied request_id, shared by every endpoint that deduplicates:
        # /v1/nexus/intelligence, /v1/sentinel/analyze, /v1/intelx/research and
        # /v1/futuris/enhance all read and wrote this one dict. A client reusing a single
        # request_id across two of them could therefore be served another endpoint's
        # payload. Measured live: POST /v1/sentinel/analyze with a request_id previously
        # used by nexus returned HTTP 500, because the Nexus response failed
        # SentinelAnalysisResponse validation
        # (app/services/sentinel_intelligence.py:190). Wherever the two shapes were
        # compatible it would have returned the other endpoint's data outright. An empty
        # request_id made every endpoint collide on the key "".
        self.dedup_ttl_seconds = 300.0  # 5 minutes idempotency window
        # request_id is caller-supplied and each entry retains a full response payload, so
        # this needs a hard ceiling as well as a TTL. It is a BoundedStore rather than a
        # plain dict so that (a) it appears in the /metrics/runtime retention report with
        # its ceiling and eviction count, and (b) the ceiling is enforced by one shared,
        # tested implementation.
        #
        # Measured: driving 12,000 unique-id POST /v1/nexus/intelligence requests with the
        # old 5,000-entry ceiling left the process at 123.4 MB RSS, and the dedup cache was
        # invisible in the retention report (12 registered stores, none of them this one).
        # A deduplicated response is an optimisation, never a correctness requirement: a
        # repeat request that misses the window is recomputed. 1,000 entries keeps a
        # five-minute burst idempotent at a fifth of the retained payload memory.
        self.dedup_cache: BoundedStore[dict[str, Any]] = BoundedStore(
            "governance.dedup_cache", max_entries=_dedup_ceiling()
        )

    @property
    def max_dedup_entries(self) -> int:
        """Ceiling of the deduplication window (the store's own, so it cannot drift)."""
        return self.dedup_cache.max_entries

    @max_dedup_entries.setter
    def max_dedup_entries(self, value: int) -> None:
        self.dedup_cache.max_entries = value

    def describe_dedup_cache(self) -> dict[str, Any]:
        """Retention state of the idempotency window, for audit endpoints."""
        return self.dedup_cache.describe()

    def dedup_cache_miss_detail(self, request_id: str, namespace: str = "default") -> str:
        """Explain a dedup miss: never seen, or dropped from the window."""
        from app.utils.bounded_store import missing_entry_detail

        return missing_entry_detail(
            self.dedup_cache, self._dedup_key(request_id, namespace), "cached response"
        )

    def extract_tenant_id(self, auth_header: str | None, api_key: str | None) -> str:
        token = (api_key or auth_header or "").strip()
        for t_id, policy in self.tenants.items():
            if any(k in token for k in policy.active_keys) or t_id in token.lower():
                return t_id
        if "forge" in token.lower():
            return "tenant_forge"
        elif "trading" in token.lower() or "bot" in token.lower():
            return "tenant_trading"
        elif "nexus" in token.lower():
            return "tenant_nexus"
        elif "sentinel" in token.lower():
            return "tenant_sentinel"
        elif "intelx" in token.lower():
            return "tenant_intelx"
        elif "futuris" in token.lower():
            return "tenant_futuris"
        return "tenant_default"

    def check_tenant_budget(self, tenant_id: str, estimated_cost_usd: float = 0.001) -> bool:
        """Tracks tenant spend without enforcing artificial budget cutoffs."""
        policy = self.tenants.get(tenant_id, self.tenants["tenant_default"])
        policy.current_daily_spend_usd += estimated_cost_usd
        return True

    def rotate_tenant_key(self, tenant_id: str, old_key: str) -> str:
        """Rotates a tenant API key after proving possession of the current one.

        The caller MUST present a key that is presently active for the tenant.
        Previously any value was accepted (and silently appended), which meant an
        unauthenticated caller could mint unlimited tenant keys. Now:
          * unknown tenant        -> ValueError    (404/400)
          * old_key not active    -> PermissionError (403)
          * success               -> exactly one key swapped, so `active_keys`
                                     cannot grow without bound.
        """
        policy = self.tenants.get(tenant_id)
        if not policy:
            raise ValueError(f"Tenant '{tenant_id}' not found.")

        if old_key not in policy.active_keys:
            logger.warning(
                "Tenant key rotation rejected: presented key is not active for tenant '%s'",
                tenant_id,
            )
            raise PermissionError(
                "Presented 'old_key' is not an active key for this tenant."
            )

        # CSPRNG: a credential must not be derived from a predictable timestamp.
        new_key = f"key_{tenant_id}_{secrets.token_hex(8)}"
        policy.active_keys.remove(old_key)
        policy.active_keys.append(new_key)
        logger.info("Tenant '%s' API key rotated successfully.", tenant_id)
        return new_key

    @staticmethod
    def _dedup_key(request_id: str, namespace: str) -> str:
        """Compose the cache key so entries can never cross endpoint boundaries."""
        return f"{namespace}\x00{request_id}"

    def check_deduplication(
        self, request_id: str, namespace: str = "default"
    ) -> dict[str, Any] | None:
        """Return the cached response for this (namespace, request_id) if still fresh.

        ``namespace`` defaults only so pre-existing callers keep working; every endpoint
        passes its own name. Without a namespace the response cached by one endpoint is
        returned to a different endpoint for the same request_id.
        """
        key = self._dedup_key(request_id, namespace)
        entry = self.dedup_cache.get(key)
        if entry:
            if time.time() < entry["expires_at"]:
                logger.info(
                    "[DEDUP HIT] Returning cached response for namespace=%s request_id=%s",
                    namespace,
                    request_id,
                )
                return entry["response"]
            else:
                del self.dedup_cache[key]
        return None

    def check_deduplication_model(
        self, request_id: str, model: type[Any], namespace: str
    ) -> Any | None:
        """Return the cached payload **as ``model``**, or ``None`` if there is no usable hit.

        A cache must never be able to fail a request. Rebuilding a response with
        ``Model(**cached)`` raised ``ValidationError`` out of the endpoint and surfaced as
        HTTP 500 whenever the stored payload did not match the model — which is exactly
        what happened when one endpoint read another's entry
        (``app/services/sentinel_intelligence.py:190``). Here a payload that does not
        validate is logged, evicted, and treated as a miss so the caller recomputes,
        instead of turning a cache inconsistency into a server fault.
        """
        from pydantic import ValidationError

        payload = self.check_deduplication(request_id, namespace=namespace)
        if payload is None:
            return None
        try:
            return model(**payload)
        except ValidationError as exc:
            logger.error(
                "Discarding deduplication entry namespace=%s request_id=%s: cached payload "
                "does not validate as %s (%s)",
                namespace,
                request_id,
                getattr(model, "__name__", model),
                exc.error_count() if hasattr(exc, "error_count") else "validation error",
            )
            self.invalidate_deduplication(request_id, namespace=namespace)
            return None

    def invalidate_deduplication(self, request_id: str, namespace: str = "default") -> None:
        """Drop a cached entry (used when a stored payload turns out to be unusable)."""
        self.dedup_cache.pop(self._dedup_key(request_id, namespace), None)

    def store_deduplication(
        self, request_id: str, response_payload: dict[str, Any], namespace: str = "default"
    ) -> None:
        """Cache a response under this endpoint's namespace (see ``check_deduplication``)."""
        # Drop expired entries first, so the freshest idempotency window survives; the
        # store's own LRU eviction covers the remaining overflow.
        now = time.time()
        for rid, entry in list(self.dedup_cache.items()):
            if now >= entry["expires_at"]:
                del self.dedup_cache[rid]

        self.dedup_cache[self._dedup_key(request_id, namespace)] = {
            "response": response_payload,
            "expires_at": time.time() + self.dedup_ttl_seconds
        }


def _dedup_ceiling() -> int:
    """Ceiling for the deduplication window, overridable by configuration."""
    try:
        from app.core.config import settings

        value = int(getattr(settings, "DEDUP_CACHE_MAX_ENTRIES", 1000))
        return value if value > 0 else 1000
    except Exception:  # noqa: BLE001 - configuration must never break the cache
        return 1000


tenant_manager = MultiTenantManager()
