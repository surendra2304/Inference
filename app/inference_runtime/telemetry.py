from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TraceEvent:
    ts: float
    request_id: str
    event: str
    provider: str | None = None
    model: str | None = None
    data: dict[str, Any] | None = None


class TraceRecorder:
    def __init__(self, max_events: int = 10000) -> None:
        self.max_events = max_events
        self.events: list[TraceEvent] = []
        self._lock = threading.RLock()

    def emit(
        self, event: str, request_id: str | None = None, provider: str | None = None, model: str | None = None, **data
    ) -> None:
        with self._lock:
            self.events.append(TraceEvent(time.time(), request_id or str(uuid.uuid4()), event, provider, model, data))
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events :]

    def export_json(self) -> str:
        with self._lock:
            return json.dumps([asdict(e) for e in self.events], default=str)
