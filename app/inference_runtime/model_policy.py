from __future__ import annotations

import re
from dataclasses import dataclass

from .contracts import Capability, CompletionRequest, ProviderEndpoint

_KEY_LEAK_PATTERN = re.compile(
    r"\b(?:sk-(?:proj-|or-v1-)?[A-Za-z0-9_-]{16,}|gsk_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{35})\b"
)


@dataclass(frozen=True)
class PolicyResult:
    allowed: bool
    reason: str


class ModelPolicy:
    def __init__(self, deny_models: set[str] | None = None, max_context: int = 1_000_000):
        self.deny = deny_models or set()
        self.max_context = max_context

    def check(self, request: CompletionRequest, endpoint: ProviderEndpoint) -> PolicyResult:
        if endpoint.model in self.deny:
            return PolicyResult(False, "model denied by policy")
        if endpoint.max_context > self.max_context:
            return PolicyResult(False, "advertised context exceeds configured safety bound")
        if request.response_schema and Capability.JSON not in endpoint.capabilities:
            return PolicyResult(False, "structured output capability not declared")
        if request.stream and Capability.STREAM not in endpoint.capabilities:
            return PolicyResult(False, "stream capability not declared")
        return PolicyResult(True, "ok")

    def can_failover(
        self,
        from_provider: str,
        to_provider: str,
        stage: str | None = None,
        allow_consequential: bool = False,
    ) -> PolicyResult:
        """Enforces policy invariants on provider failovers."""
        consequential_stages = {"cross_review_critique", "consensus_synthesis", "evidence_check"}
        if stage in consequential_stages and not allow_consequential:
            return PolicyResult(False, f"mid-stage failover restricted for consequential stage '{stage}'")
        if from_provider.lower() == to_provider.lower():
            return PolicyResult(False, "cannot fail over to the same provider")
        return PolicyResult(True, "ok")

    def validate_request_safety(self, request: CompletionRequest) -> PolicyResult:
        """Inspects request content to verify no raw credentials or prompt leak vectors exist."""
        for msg in request.messages:
            content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
            if _KEY_LEAK_PATTERN.search(content_str):
                return PolicyResult(False, "request contains unredacted provider credential")
        return PolicyResult(True, "ok")
