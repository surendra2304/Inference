from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ConfigurationSnapshot:
    version: str
    providers: tuple[str, ...]
    feature_flags: tuple[tuple[str, bool], ...]
    limits: tuple[tuple[str, float], ...]
    captured_at: float

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()


def snapshot(
    version: str, providers: list[str], flags: dict[str, bool], limits: dict[str, float]
) -> ConfigurationSnapshot:
    return ConfigurationSnapshot(
        version, tuple(sorted(providers)), tuple(sorted(flags.items())), tuple(sorted(limits.items())), time.time()
    )
