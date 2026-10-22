from __future__ import annotations

from dataclasses import dataclass

from .contracts import CompletionRequest
from .security import is_safe_cache_content


@dataclass(frozen=True)
class CacheDecision:
    allowed: bool
    reason: str


class CachePolicy:
    def __init__(self, default_ttl: float = 60):
        self.default_ttl = default_ttl

    def decide(self, request: CompletionRequest) -> CacheDecision:
        if request.stream:
            return CacheDecision(False, "stream")
        if not request.cacheable:
            return CacheDecision(False, "request_not_cacheable")
        if request.response_schema is None and any(
            isinstance(m.content, str) and not is_safe_cache_content(m.content) for m in request.messages
        ):
            return CacheDecision(False, "secret_like_content")
        return CacheDecision(True, "explicitly_cacheable")
