from __future__ import annotations

import threading
from collections import defaultdict


class KeyedLocks:
    def __init__(self) -> None:
        self._master = threading.Lock()
        self._locks: dict[str, threading.RLock] = defaultdict(threading.RLock)

    def lock(self, key: str) -> threading.RLock:
        with self._master:
            return self._locks[key]


GLOBAL_LOCKS = KeyedLocks()
