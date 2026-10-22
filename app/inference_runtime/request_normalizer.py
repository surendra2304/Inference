from __future__ import annotations

from .contracts import CompletionRequest, Message


class RequestNormalizer:
    allowed_roles = {"system", "user", "assistant", "tool"}

    def normalize(self, request: CompletionRequest) -> CompletionRequest:
        messages = []
        for m in request.messages:
            role = m.role.strip().lower()
            if role not in self.allowed_roles:
                raise ValueError(f"unsupported message role: {role}")
            content = m.content if isinstance(m.content, str) else list(m.content)
            messages.append(Message(role, content, m.name))
        temp = min(2.0, max(0.0, request.temperature))
        max_tokens = None if request.max_tokens is None else max(1, request.max_tokens)
        return CompletionRequest(
            model=request.model,
            messages=tuple(messages),
            temperature=temp,
            max_tokens=max_tokens,
            timeout_seconds=max(0.1, request.timeout_seconds),
            capabilities=request.capabilities,
            response_schema=request.response_schema,
            tenant_id=request.tenant_id,
            request_id=request.request_id,
            cacheable=request.cacheable,
            stream=request.stream,
            extra=dict(request.extra),
        )
