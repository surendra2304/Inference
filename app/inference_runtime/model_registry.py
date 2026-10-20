from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .contracts import Capability, ProviderEndpoint


@dataclass
class ModelRecord:
    endpoint: ProviderEndpoint
    enabled: bool = True
    aliases: set[str] = field(default_factory=set)
    metadata: dict[str, object] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)


# Deterministic failover waterfall rungs
WATERFALL_MODELS: list[tuple[str, str, frozenset[Capability], int]] = [
    ("groq", "llama-3.3-70b", frozenset({Capability.CHAT, Capability.STREAM, Capability.JSON, Capability.TOOLS}), 128_000),
    ("mistral", "codestral-latest", frozenset({Capability.CHAT, Capability.STREAM, Capability.JSON, Capability.TOOLS}), 128_000),
    ("openai", "gpt-4o", frozenset({Capability.CHAT, Capability.STREAM, Capability.JSON, Capability.TOOLS, Capability.VISION}), 128_000),
    ("gemini", "gemini-3.8-flash", frozenset({Capability.CHAT, Capability.STREAM, Capability.JSON, Capability.TOOLS, Capability.VISION}), 1_000_000),
]


class ModelRegistry:
    def __init__(self, populate_defaults: bool = True) -> None:
        self._models: dict[tuple[str, str], ModelRecord] = {}
        self._aliases: dict[str, tuple[str, str]] = {}
        self._lock = threading.RLock()
        if populate_defaults:
            self._register_defaults()

    def _register_defaults(self) -> None:
        for prov, model, caps, ctx in WATERFALL_MODELS:
            ep = ProviderEndpoint(
                provider=prov,
                model=model,
                capabilities=caps,
                max_context=ctx,
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
            )
            self.upsert(ModelRecord(endpoint=ep, aliases={f"{prov}/{model}", model}))

    def upsert(self, record: ModelRecord) -> None:
        with self._lock:
            self._models[(record.endpoint.provider, record.endpoint.model)] = record
            for a in record.aliases:
                self._aliases[a] = (record.endpoint.provider, record.endpoint.model)

    def disable(self, provider: str, model: str) -> None:
        with self._lock:
            if r := self._models.get((provider, model)):
                r.enabled = False
                r.updated_at = time.time()

    def enable(self, provider: str, model: str) -> None:
        with self._lock:
            if r := self._models.get((provider, model)):
                r.enabled = True
                r.updated_at = time.time()

    def resolve(self, name: str, provider: str | None = None) -> ModelRecord | None:
        with self._lock:
            key = self._aliases.get(name)
            if key:
                rec = self._models.get(key)
                if rec and rec.enabled:
                    return rec
            candidates = [
                r
                for r in self._models.values()
                if r.endpoint.model == name and (provider is None or r.endpoint.provider == provider)
            ]
            return next((r for r in candidates if r.enabled), None)

    def list_enabled(self) -> list[ModelRecord]:
        with self._lock:
            return [r for r in self._models.values() if r.enabled]

    def get_waterfall_sequence(self, start_provider: str | None = None) -> list[ModelRecord]:
        """Returns enabled models in the deterministic failover waterfall sequence.

        Order: Groq -> Mistral -> OpenAI -> Gemini.
        If start_provider is specified, cycles through remaining ladder rungs.
        """
        with self._lock:
            enabled_map = {r.endpoint.provider: r for r in self._models.values() if r.enabled}
            waterfall_providers = ["groq", "mistral", "openai", "gemini"]

            if start_provider and start_provider.lower() in waterfall_providers:
                idx = waterfall_providers.index(start_provider.lower())
                ordered_names = waterfall_providers[idx + 1 :] + waterfall_providers[:idx]
            else:
                ordered_names = waterfall_providers

            results: list[ModelRecord] = []
            for name in ordered_names:
                if name in enabled_map:
                    results.append(enabled_map[name])
            return results

    def resolve_fallback(self, current_provider: str, current_model: str) -> ModelRecord | None:
        """Finds the next eligible model in the deterministic failover waterfall."""
        ladder = self.get_waterfall_sequence(start_provider=current_provider)
        for r in ladder:
            if r.endpoint.provider != current_provider and r.enabled:
                return r
        return None
