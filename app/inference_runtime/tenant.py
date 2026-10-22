from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TenantPolicy:
    tenant_id: str
    allowed_providers: frozenset[str] = frozenset()
    allowed_models: frozenset[str] = frozenset()
    max_tokens: int = 8192
    cache_enabled: bool = False

    def allows(self, provider: str, model: str, max_tokens: int | None) -> bool:
        if self.allowed_providers and provider not in self.allowed_providers:
            return False
        if self.allowed_models and model not in self.allowed_models:
            return False
        return (max_tokens or 0) <= self.max_tokens


class TenantRegistry:
    def __init__(self):
        self._items = {}

    def register(self, p: TenantPolicy) -> None:
        self._items[p.tenant_id] = p

    def get(self, tenant: str) -> TenantPolicy:
        return self._items.get(tenant, TenantPolicy(tenant))
