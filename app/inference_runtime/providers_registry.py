from __future__ import annotations

import threading
from typing import Any


class TransportRegistry:
    def __init__(self):
        self._items = {}
        self._lock = threading.RLock()

    def register(self, name: str, transport: Any) -> None:
        n = name.lower().strip()
        if not n:
            raise ValueError("provider name required")
        with self._lock:
            self._items[n] = transport

    def get(self, name: str) -> Any | None:
        with self._lock:
            return self._items.get(name.lower().strip())

    def all(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._items)

    def remove(self, name: str) -> None:
        with self._lock:
            self._items.pop(name.lower().strip(), None)
