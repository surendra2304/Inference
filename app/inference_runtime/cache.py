from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class Entry:
    value: Any
    expires_at: float
    tenant: str


class TenantCache:
    def __init__(self, max_items: int = 1000) -> None:
        if max_items < 1:
            raise ValueError("max_items must be at least 1")
        self.max_items = max_items
        self._data: dict[str, Entry] = {}
        self._lock = threading.RLock()

    def key(self, tenant: str, payload: Any) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        return tenant + ":" + hashlib.sha256(canonical).hexdigest()

    def get(self, key: str) -> Any | None:
        with self._lock:
            e = self._data.get(key)
            if not e:
                return None
            if e.expires_at <= time.monotonic():
                self._data.pop(key, None)
                return None
            return e.value

    def put(self, key: str, value: Any, tenant: str, ttl: float) -> None:
        if ttl <= 0:
            self.delete(key)
            return
        with self._lock:
            if key not in self._data and len(self._data) >= self.max_items:
                self._data.pop(next(iter(self._data)))
            self._data[key] = Entry(value, time.monotonic() + ttl, tenant)

    def delete(self, key: str) -> None:
        """Evict one entry, including a malformed value discovered during validation."""
        with self._lock:
            self._data.pop(key, None)

    def invalidate_tenant(self, tenant: str) -> None:
        with self._lock:
            for k in [k for k, v in self._data.items() if v.tenant == tenant]:
                self._data.pop(k, None)
