from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: tuple[tuple[float, ...], ...]
    model: str
    provider: str
    dimensions: int


class EmbeddingBatcher:
    def __init__(self, max_batch: int = 32):
        self.max_batch = max(1, max_batch)

    def batches(self, items: Sequence[str]):
        for i in range(0, len(items), self.max_batch):
            yield items[i : i + self.max_batch]

    @staticmethod
    def validate(vectors: Sequence[Sequence[float]], expected_dim: int | None = None) -> None:
        if not vectors:
            return
        dim = len(vectors[0])
        if dim == 0:
            raise ValueError("embedding dimension cannot be zero")
        if expected_dim is not None and dim != expected_dim:
            raise ValueError("embedding dimension mismatch")
        if any(len(v) != dim for v in vectors):
            raise ValueError("inconsistent embedding dimensions")
