from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class ProbeStatus:
    provider: str
    ok: bool
    latency_seconds: float
    error: str | None = None
    checked_at: float = 0.0


class HealthService:
    def __init__(self, transports: dict[str, Any]):
        self.transports = transports
        self.status: dict[str, ProbeStatus] = {}

    async def probe_one(self, name, transport):
        start = time.perf_counter()
        try:
            ok = bool(await transport.health())
            s = ProbeStatus(name, ok, time.perf_counter() - start, None, time.time())
        except Exception as exc:
            s = ProbeStatus(name, False, time.perf_counter() - start, str(exc)[:300], time.time())
        self.status[name] = s
        return s

    async def probe_all(self) -> dict[str, ProbeStatus]:
        return {
            s.provider: s for s in await asyncio.gather(*(self.probe_one(n, t) for n, t in self.transports.items()))
        }
