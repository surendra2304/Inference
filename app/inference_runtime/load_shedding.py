from __future__ import annotations

import asyncio

from .errors import AdmissionRejected


class LoadShedder:
    def __init__(self, max_queue: int = 256):
        self.max_queue = max_queue
        self._active = 0
        self._lock = asyncio.Lock()

    async def enter(self):
        async with self._lock:
            if self._active >= self.max_queue:
                raise AdmissionRejected("service overloaded")
            self._active += 1

    async def leave(self):
        async with self._lock:
            self._active = max(0, self._active - 1)
