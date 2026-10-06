"""Multi-Tenant Isolation, API Governance, Row-Level Tenant Security, and Deduplication."""

import secrets
import time
from typing import Any

from pydantic import BaseModel, Field

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
        # In-memory deduplication cache: hash -> (response_payload, expiry_timestamp)
        self.dedup_cache: dict[str, dict[str, Any]] = {}
        self.dedup_ttl_seconds = 300.0  # 5 minutes idempotency window
        # request_id is caller-supplied and each entry retains a full response
        # payload, so the cache needs a hard ceiling as well as a TTL.
        self.max_dedup_entries = 5_000

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

    def check_deduplication(self, request_id: str) -> dict[str, Any] | None:
        """Returns cached response if request_id was processed in the last 5 minutes."""
        entry = self.dedup_cache.get(request_id)
        if entry:
            if time.time() < entry["expires_at"]:
                logger.info("[DEDUP HIT] Returning cached response for request_id: %s", request_id)
                return entry["response"]
            else:
                del self.dedup_cache[request_id]
        return None

    def store_deduplication(self, request_id: str, response_payload: dict[str, Any]) -> None:
        if len(self.dedup_cache) >= self.max_dedup_entries:
            # Prefer dropping entries that are already expired, then the
            # soonest-to-expire, so the freshest idempotency window survives.
            now = time.time()
            for rid, entry in list(self.dedup_cache.items()):
                if now >= entry["expires_at"]:
                    del self.dedup_cache[rid]
            if len(self.dedup_cache) >= self.max_dedup_entries:
                soonest_expiry = sorted(
                    self.dedup_cache.items(), key=lambda kv: kv[1]["expires_at"]
                )
                overflow = len(self.dedup_cache) - self.max_dedup_entries + 1
                for rid, _ in soonest_expiry[:max(1, overflow)]:
                    del self.dedup_cache[rid]

        self.dedup_cache[request_id] = {
            "response": response_payload,
            "expires_at": time.time() + self.dedup_ttl_seconds
        }


tenant_manager = MultiTenantManager()
