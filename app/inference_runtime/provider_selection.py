from __future__ import annotations

from dataclasses import dataclass

from .contracts import Capability, ProviderEndpoint


@dataclass(frozen=True)
class SelectionFilter:
    provider_allow: list[str] | None = None
    provider_deny: list[str] | None = None
    model_allow: list[str] | None = None
    require_caps: frozenset[Capability] = frozenset()

    def match(self, ep: ProviderEndpoint) -> bool:
        if self.provider_allow and ep.provider not in self.provider_allow:
            return False
        if self.provider_deny and ep.provider in self.provider_deny:
            return False
        if self.model_allow and ep.model not in self.model_allow:
            return False
        return self.require_caps.issubset(ep.capabilities)


class ProviderSelector:
    def filter(self, endpoints: list[ProviderEndpoint], f: SelectionFilter) -> list[ProviderEndpoint]:
        return [e for e in endpoints if f.match(e)]
