"""Base LLM Provider interface and data contracts."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ProviderMessage(BaseModel):
    """Message payload for provider requests."""

    role: str = Field(description="Role of the author: system, user, assistant, or tool")
    content: str = Field(description="Textual message content")
    name: str | None = Field(default=None, description="Optional author or agent name")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        clean = v.lower().strip()
        if clean not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Invalid message role '{v}'. Supported: system, user, assistant, tool")
        return clean


class ProviderRequest(BaseModel):
    """Standardized request sent to any LLM provider."""

    messages: list[ProviderMessage] = Field(description="List of conversation/debate messages")
    system_instruction: str | None = Field(default=None, description="System-level prompt instructions")
    model: str | None = Field(default=None, description="Target model name override")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, description="Max output tokens to generate")
    response_schema: dict[str, Any] | None = Field(default=None, description="Optional structured JSON schema")
    extra_params: dict[str, Any] = Field(default_factory=dict, description="Provider-specific tuning parameters")

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[ProviderMessage]) -> list[ProviderMessage]:
        if len(v) > 256:
            raise ValueError(f"Message count exceeds maximum allowed limit (256, got {len(v)})")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int | None) -> int | None:
        if v is not None and (v < 1 or v > 128000):
            raise ValueError(f"max_tokens must be between 1 and 128000, got {v}")
        return v


class UsageEstimate(BaseModel):
    """Token and cost estimation for a request."""

    estimated_prompt_tokens: int = 0
    estimated_completion_tokens: int = 0
    estimated_total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class ProviderCapabilities(BaseModel):
    """Metadata describing provider features and limitations."""

    provider_name: str
    supported_models: list[str]
    supports_streaming: bool = True
    supports_structured_output: bool = True
    supports_system_instructions: bool = True
    supports_tool_calling: bool = True
    max_context_window: int = 128000
    rate_limits: dict[str, Any] = Field(default_factory=dict)


class ProviderResponse(BaseModel):
    """Standardized response from an LLM provider."""

    content: str = Field(description="Generated text content")
    model: str = Field(description="Model identifier that produced the response")
    provider: str = Field(description="Provider name")
    prompt_tokens: int | None = 0
    completion_tokens: int | None = 0
    total_tokens: int | None = 0
    latency_seconds: float = 0.0
    finish_reason: str | None = "stop"
    raw_response: dict[str, Any] | None = None


class BaseLLMProvider(ABC):
    """Abstract base class for cloud LLM provider adapters."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g. gemini, groq, mistral, openrouter, nvidia)."""

    @abstractmethod
    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate a complete response for the given standardized request."""

    @abstractmethod
    def stream(self, request: ProviderRequest) -> AsyncIterator[str]:
        """Stream chunks of generated text for the given request."""

    @abstractmethod
    def estimate_usage(self, request: ProviderRequest) -> UsageEstimate:
        """Estimate token consumption and compute cost before making the API call."""

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Return provider features, supported models, and limits."""

    @abstractmethod
    async def health(self) -> bool:
        """Check provider connectivity, authentication, and service availability."""
