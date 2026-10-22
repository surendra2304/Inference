from __future__ import annotations

from typing import Any

from .health import ProviderHealth


class HealthProbe:
    def __init__(self, health: ProviderHealth) -> None:
        self.health = health

    async def probe(self, providers: dict[str, Any]) -> dict[str, bool]:
        out = {}
        for name, p in providers.items():
            try:
                ok = bool(await p.health())
                out[name] = ok
                self.health.success(name, 0.0) if ok else self.health.failure(name, "health=false")
            except Exception as exc:
                out[name] = False
                self.health.failure(name, str(exc))
        return out
