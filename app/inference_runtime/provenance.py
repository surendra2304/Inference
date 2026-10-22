from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ProviderAttempt:
    provider: str
    model: str
    attempt: int
    started_at: float
    latency_seconds: float = 0
    outcome: str = "started"
    error_code: str | None = None


@dataclass
class Provenance:
    request_id: str
    selected_provider: str
    selected_model: str
    attempts: list[ProviderAttempt]
    fallback_count: int = 0
    # note: kept mutable for incremental trace construction


class ProvenanceRecorder:
    def __init__(self, request_id: str, provider: str, model: str):
        self.value = Provenance(request_id, provider, model, [])

    def start(self, provider: str, model: str) -> ProviderAttempt:
        item = ProviderAttempt(provider, model, len(self.value.attempts) + 1, time.time())
        self.value.attempts.append(item)
        return item

    def finish(self, item: ProviderAttempt, outcome: str, error_code: str | None = None) -> None:
        item.latency_seconds = time.time() - item.started_at
        item.outcome = outcome
        item.error_code = error_code

    def export(self) -> dict[str, Any]:
        return asdict(self.value)
