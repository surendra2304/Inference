"""Latency SLA Predictor and Dynamic Performance Estimator for Inference."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .contracts import Capability, ProviderEndpoint


@dataclass
class SlaProfile:
    """Latency profile for a single provider/model combination."""

    provider: str
    model: str
    samples_count: int = 0
    successes_count: int = 0
    base_overhead_ms: float = 120.0
    ms_per_output_token: float = 18.0
    p95_latency_ms: float = 500.0
    recent_latencies: list[float] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


class SlaPredictor:
    """Predicts completion latencies and recommends providers to satisfy caller SLAs."""

    MAX_HISTORY = 100

    def __init__(self) -> None:
        self._profiles: dict[tuple[str, str], SlaProfile] = {}
        self._lock = threading.RLock()
        self._seed_default_profiles()

    def _seed_default_profiles(self) -> None:
        """Seed empirical baseline profiles for known providers."""
        defaults = [
            ("groq", "openai/gpt-oss-120b", 60.0, 8.5, 300.0),
            ("groq", "llama-3.3-70b-versatile", 50.0, 7.0, 250.0),
            ("openrouter", "meta-llama/llama-3.3-70b-instruct", 180.0, 16.0, 650.0),
            ("openrouter", "google/gemini-2.5-flash", 160.0, 14.0, 550.0),
            ("gemini", "gemini-flash-latest", 140.0, 12.0, 480.0),
            ("nvidia", "meta/llama-3.1-70b-instruct", 200.0, 18.0, 750.0),
        ]
        for prov, model, base, per_tok, p95 in defaults:
            self._profiles[(prov, model)] = SlaProfile(
                provider=prov,
                model=model,
                base_overhead_ms=base,
                ms_per_output_token=per_tok,
                p95_latency_ms=p95,
            )

    def _get_or_create(self, provider: str, model: str) -> SlaProfile:
        key = (provider.lower(), model.lower())
        if key not in self._profiles:
            self._profiles[key] = SlaProfile(provider=provider, model=model)
        return self._profiles[key]

    def record_observation(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """Records an observed completion latency to refine predictive models."""
        with self._lock:
            prof = self._get_or_create(provider, model)
            prof.samples_count += 1
            if success:
                prof.successes_count += 1
            prof.last_updated = time.time()

            # Bounded rolling history
            prof.recent_latencies.append(latency_ms)
            if len(prof.recent_latencies) > self.MAX_HISTORY:
                prof.recent_latencies.pop(0)

            # Recalculate P95
            sorted_lats = sorted(prof.recent_latencies)
            p95_idx = int(math.ceil(0.95 * len(sorted_lats))) - 1
            prof.p95_latency_ms = sorted_lats[max(0, p95_idx)]

            # Refine ms_per_output_token using EWMA
            if completion_tokens > 5:
                estimated_output_ms = max(10.0, latency_ms - prof.base_overhead_ms)
                observed_rate = estimated_output_ms / completion_tokens
                alpha = 0.15
                prof.ms_per_output_token = (alpha * observed_rate) + ((1 - alpha) * prof.ms_per_output_token)

    def predict_latency_ms(
        self,
        provider: str,
        model: str,
        prompt_tokens: int = 50,
        completion_tokens: int = 150,
    ) -> float:
        """Predicts expected completion latency in milliseconds."""
        with self._lock:
            prof = self._get_or_create(provider, model)
            # Prompt ingest overhead: ~0.05ms per prompt token
            prompt_overhead = prompt_tokens * 0.05
            expected = prof.base_overhead_ms + prompt_overhead + (completion_tokens * prof.ms_per_output_token)
            return round(expected, 2)

    def recommend_routing(
        self,
        target_sla_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        required_capabilities: frozenset[Capability] | set[Capability] | None = None,
        candidates: list[ProviderEndpoint] | None = None,
    ) -> list[tuple[ProviderEndpoint, float, bool]]:
        """Ranks candidate endpoints based on ability to satisfy target SLA.

        Returns a list of tuples: (endpoint, predicted_latency_ms, meets_sla)
        """
        req_caps = frozenset(required_capabilities or set())
        results: list[tuple[ProviderEndpoint, float, bool]] = []

        if not candidates:
            return []

        with self._lock:
            for ep in candidates:
                if not req_caps.issubset(ep.capabilities):
                    continue
                predicted = self.predict_latency_ms(
                    ep.provider, ep.model, prompt_tokens, completion_tokens
                )
                meets = predicted <= target_sla_ms
                results.append((ep, predicted, meets))

        # Sort: first by meets_sla (True first), then by predicted latency ascending
        results.sort(key=lambda item: (not item[2], item[1]))
        return results

    def get_profile_summary(self) -> dict[str, Any]:
        """Returns diagnostic summary of all learned SLA profiles."""
        with self._lock:
            return {
                f"{k[0]}/{k[1]}": {
                    "provider": v.provider,
                    "model": v.model,
                    "samples": v.samples_count,
                    "success_rate": round(v.successes_count / max(1, v.samples_count), 3),
                    "base_overhead_ms": round(v.base_overhead_ms, 1),
                    "ms_per_output_token": round(v.ms_per_output_token, 2),
                    "p95_latency_ms": round(v.p95_latency_ms, 1),
                }
                for k, v in self._profiles.items()
            }


sla_predictor = SlaPredictor()
