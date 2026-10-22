from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class Capability(str, Enum):
    CHAT = "chat"
    STREAM = "stream"
    JSON = "json"
    TOOLS = "tools"
    VISION = "vision"
    EMBEDDING = "embedding"


@dataclass(frozen=True)
class Message:
    role: str
    content: str | list[dict[str, Any]]
    name: str | None = None


@dataclass(frozen=True)
class CompletionRequest:
    model: str
    messages: tuple[Message, ...]
    temperature: float = 0.2
    max_tokens: int | None = None
    timeout_seconds: float = 60.0
    capabilities: frozenset[Capability] = field(default_factory=lambda: frozenset({Capability.CHAT}))
    response_schema: Mapping[str, Any] | None = None
    tenant_id: str = "default"
    request_id: str = ""
    cacheable: bool = False
    stream: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompletionResult:
    text: str
    provider: str
    model: str
    latency_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    fallback: bool = False
    finish_reason: str | None = "stop"
    raw: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ProviderEndpoint:
    provider: str
    model: str
    capabilities: frozenset[Capability]
    weight: float = 1.0
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    max_context: int = 128_000
    base_latency_seconds: float = 1.0


@dataclass(frozen=True)
class RoutingDecision:
    endpoint: ProviderEndpoint
    score: float
    reasons: tuple[str, ...] = ()
