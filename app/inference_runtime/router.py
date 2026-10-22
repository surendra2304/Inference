from __future__ import annotations

from dataclasses import dataclass

from .contracts import CompletionRequest, ProviderEndpoint, RoutingDecision
from .health import ProviderHealth


@dataclass
class ProviderStats:
    successes: int = 0
    failures: int = 0
    latency: float = 0.0

    def success_rate(self) -> float:
        return self.successes / max(1, self.successes + self.failures)


class CapabilityRouter:
    def __init__(self, health: ProviderHealth) -> None:
        self.health = health
        self.endpoints: list[ProviderEndpoint] = []
        self.stats: dict[tuple[str, str], ProviderStats] = {}

    def register(self, endpoint: ProviderEndpoint) -> None:
        if endpoint not in self.endpoints:
            self.endpoints.append(endpoint)
            self.stats.setdefault((endpoint.provider, endpoint.model), ProviderStats())

    def choose(self, request: CompletionRequest) -> RoutingDecision:
        candidates = []
        for ep in self.endpoints:
            if request.model and ep.model != request.model and request.model != "auto":
                continue
            if not request.capabilities.issubset(ep.capabilities):
                continue
            h = self.health.snapshot(ep.provider)
            if not h.healthy:
                continue
            st = self.stats[(ep.provider, ep.model)]
            score = (
                (st.success_rate() * 5.0)
                + (ep.weight * 1.0)
                - (h.ewma_latency * 0.05)
                - (ep.cost_per_1k_input + ep.cost_per_1k_output) * 0.1
            )
            candidates.append((score, ep))
        if not candidates:
            raise RuntimeError("No provider endpoint satisfies requested model/capabilities")
        score, ep = max(candidates, key=lambda x: (x[0], x[1].provider, x[1].model))
        return RoutingDecision(
            ep,
            score,
            (
                f"success_rate={self.stats[(ep.provider, ep.model)].success_rate():.3f}",
                f"latency={self.health.snapshot(ep.provider).ewma_latency:.3f}",
            ),
        )

    def record(self, ep: ProviderEndpoint, success: bool, latency: float) -> None:
        st = self.stats[(ep.provider, ep.model)]
        st.failures += not success
        st.successes += success
        st.latency = latency if not st.latency else 0.8 * st.latency + 0.2 * latency
