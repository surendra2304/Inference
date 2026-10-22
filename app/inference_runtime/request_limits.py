from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestLimits:
    max_prompt_chars: int = 500_000
    max_output_tokens: int = 32_768
    max_messages: int = 512
    max_schema_bytes: int = 128_000

    def validate(self, request) -> None:
        if len(request.messages) > self.max_messages:
            raise ValueError("message count exceeds limit")
        chars = sum(len(m.content) if isinstance(m.content, str) else len(str(m.content)) for m in request.messages)
        if chars > self.max_prompt_chars:
            raise ValueError("prompt too large")
        if request.max_tokens and request.max_tokens > self.max_output_tokens:
            raise ValueError("max_tokens exceeds limit")
        if request.response_schema and len(str(request.response_schema).encode()) > self.max_schema_bytes:
            raise ValueError("response schema too large")
