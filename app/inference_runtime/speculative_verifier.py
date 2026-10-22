"""Speculative Decoding Draft Token Verifier and Acceleration Telemetry."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SpeculativeVerificationResult:
    """Outcome of verifying draft tokens against target model tokens."""

    draft_tokens_count: int
    accepted_tokens_count: int
    acceptance_rate: float
    mismatch_index: int | None
    draft_latency_ms: float
    target_latency_ms: float
    effective_speedup: float
    verified_content: str
    is_accelerated: bool


@dataclass
class PairSpeculationMetrics:
    """Historical speculation metrics between a specific draft and target model."""

    draft_model: str
    target_model: str
    total_runs: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    cumulative_speedup: float = 0.0
    last_verified: float = field(default_factory=time.time)


class SpeculativeVerifier:
    """Evaluates draft candidate tokens against primary model completions."""

    def __init__(self) -> None:
        self._pair_metrics: dict[tuple[str, str], PairSpeculationMetrics] = {}
        self._lock = threading.RLock()

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace and punctuation word-level tokenization for verification."""
        import re
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    def verify_completion(
        self,
        draft_text: str,
        target_text: str,
        draft_latency_ms: float,
        target_latency_ms: float,
        draft_model: str = "draft",
        target_model: str = "target",
    ) -> SpeculativeVerificationResult:
        """Compares draft text against target text, identifying accepted prefix length."""
        draft_tokens = self._tokenize(draft_text)
        target_tokens = self._tokenize(target_text)

        draft_len = len(draft_tokens)
        if draft_len == 0:
            return SpeculativeVerificationResult(
                draft_tokens_count=0,
                accepted_tokens_count=0,
                acceptance_rate=0.0,
                mismatch_index=None,
                draft_latency_ms=draft_latency_ms,
                target_latency_ms=target_latency_ms,
                effective_speedup=1.0,
                verified_content=target_text,
                is_accelerated=False,
            )

        # Count common prefix tokens
        accepted_count = 0
        mismatch_idx: int | None = None

        min_len = min(len(draft_tokens), len(target_tokens))
        for i in range(min_len):
            if draft_tokens[i].lower() == target_tokens[i].lower():
                accepted_count += 1
            else:
                mismatch_idx = i
                break

        if mismatch_idx is None and len(target_tokens) < len(draft_tokens):
            mismatch_idx = len(target_tokens)

        acceptance_rate = round(accepted_count / max(1, draft_len), 4)

        # Estimate effective speedup
        # Baseline: target alone would have taken ~ target_latency_ms
        # Combined: draft_latency + fraction of target latency for remaining tokens
        total_time = draft_latency_ms + target_latency_ms
        speedup = round(target_latency_ms / max(1.0, total_time), 3) if total_time > 0 else 1.0
        # If acceptance is high (> 80%), speedup can be modeled as favorable
        if acceptance_rate >= 0.8 and draft_latency_ms < target_latency_ms:
            speedup = max(speedup, round(1.0 + (acceptance_rate * 0.5), 2))

        is_accel = speedup > 1.05 and acceptance_rate > 0.65

        # Record metrics
        with self._lock:
            key = (draft_model.lower(), target_model.lower())
            if key not in self._pair_metrics:
                self._pair_metrics[key] = PairSpeculationMetrics(
                    draft_model=draft_model, target_model=target_model
                )
            m = self._pair_metrics[key]
            m.total_runs += 1
            m.total_draft_tokens += draft_len
            m.total_accepted_tokens += accepted_count
            m.cumulative_speedup += speedup
            m.last_verified = time.time()

        return SpeculativeVerificationResult(
            draft_tokens_count=draft_len,
            accepted_tokens_count=accepted_count,
            acceptance_rate=acceptance_rate,
            mismatch_index=mismatch_idx,
            draft_latency_ms=draft_latency_ms,
            target_latency_ms=target_latency_ms,
            effective_speedup=speedup,
            verified_content=target_text,
            is_accelerated=is_accel,
        )

    def should_speculate(self, draft_model: str, target_model: str) -> bool:
        """Determines whether speculation is historically beneficial between model pairs."""
        with self._lock:
            key = (draft_model.lower(), target_model.lower())
            m = self._pair_metrics.get(key)
            if not m or m.total_runs < 3:
                # Give new pairs an exploration chance
                return True
            avg_acceptance = m.total_accepted_tokens / max(1, m.total_draft_tokens)
            avg_speedup = m.cumulative_speedup / max(1, m.total_runs)
            return avg_acceptance >= 0.5 and avg_speedup >= 1.0

    def get_telemetry(self) -> dict[str, Any]:
        """Returns summary of speculative verification telemetry."""
        with self._lock:
            return {
                f"{k[0]}->{k[1]}": {
                    "runs": v.total_runs,
                    "avg_acceptance_rate": round(
                        v.total_accepted_tokens / max(1, v.total_draft_tokens), 3
                    ),
                    "avg_speedup": round(v.cumulative_speedup / max(1, v.total_runs), 2),
                    "total_draft_tokens": v.total_draft_tokens,
                    "total_accepted_tokens": v.total_accepted_tokens,
                }
                for k, v in self._pair_metrics.items()
            }


speculative_verifier = SpeculativeVerifier()
