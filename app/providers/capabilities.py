from __future__ import annotations

from app.inference_runtime.contracts import Capability, ProviderEndpoint


def endpoint(provider: str, model: str, **kwargs) -> ProviderEndpoint:
    caps = kwargs.pop("capabilities", None) or frozenset({Capability.CHAT, Capability.STREAM})
    return ProviderEndpoint(provider=provider, model=model, capabilities=frozenset(caps), **kwargs)


def supports(ep: ProviderEndpoint, *wanted: Capability) -> bool:
    return set(wanted).issubset(ep.capabilities)
