from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass
class StreamChunk:
    text: str
    provider: str
    model: str
    index: int
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


class StreamGuard:
    def __init__(self, timeout_seconds: float = 60.0, max_chunks: int = 20000, max_chars: int = 2_000_000) -> None:
        self.timeout = timeout_seconds
        self.max_chunks = max_chunks
        self.max_chars = max_chars

    async def guard(self, source: AsyncIterator[Any], provider: str, model: str) -> AsyncIterator[StreamChunk]:
        started = time.monotonic()
        idx = 0
        chars = 0
        async for raw in source:
            if time.monotonic() - started > self.timeout:
                raise TimeoutError("stream deadline exceeded")
            if idx >= self.max_chunks:
                raise RuntimeError("stream chunk limit exceeded")
            text = getattr(raw, "text", None) if not isinstance(raw, str) else raw
            text = text if text is not None else str(raw)
            chars += len(text)
            if chars > self.max_chars:
                raise RuntimeError("stream output limit exceeded")
            yield StreamChunk(text, provider, model, idx)
            idx += 1
