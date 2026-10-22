from __future__ import annotations

import contextvars
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("inference_request_id", default=None)
_tenant_id: contextvars.ContextVar[str] = contextvars.ContextVar("inference_tenant_id", default="default")


@dataclass
class RequestContext:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = "default"
    started_at: float = field(default_factory=time.monotonic)
    deadline: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def remaining(self) -> float | None:
        return None if self.deadline is None else max(0.0, self.deadline - time.monotonic())


def install(ctx: RequestContext):
    a = _request_id.set(ctx.request_id)
    b = _tenant_id.set(ctx.tenant_id)
    return a, b


def reset(tokens):
    a, b = tokens
    _request_id.reset(a)
    _tenant_id.reset(b)


def request_id() -> str | None:
    return _request_id.get()


def tenant_id() -> str:
    return _tenant_id.get()
