"""Provider factory and registry for Inference (7 Verified Free Cloud Providers)."""

from app.providers.base import BaseLLMProvider
from app.providers.cohere import CohereProvider
from app.providers.gemini import GeminiProvider
from app.providers.groq import GroqProvider
from app.providers.huggingface import HuggingFaceProvider
from app.providers.litellm import LiteLLMTransport
from app.providers.litellm_adapter import LiteLLMProvider
from app.providers.llamacpp import LlamaCppTransport
from app.providers.local import LocalProvider
from app.providers.mistral import MistralProvider
from app.providers.nvidia import NvidiaProvider
from app.providers.openai_compatible import OpenAICompatibleTransport
from app.providers.openrouter import OpenRouterProvider
from app.providers.sglang import SGLangTransport
from app.providers.vllm import VLLMTransport

_PROVIDER_MAP: dict[str, type[BaseLLMProvider]] = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "mistral": MistralProvider,
    "openrouter": OpenRouterProvider,
    "cohere": CohereProvider,
    "huggingface": HuggingFaceProvider,
    "nvidia": NvidiaProvider,
    "litellm": LiteLLMProvider,
    # Self-hosted OpenAI-compatible endpoint (llama.cpp / vLLM / SGLang / Ollama).
    # Registered so that "local inference" is reachable from ModelGateway instead of
    # existing only as the unregistered transports in llamacpp.py / vllm.py / sglang.py.
    "local": LocalProvider,
}

_PROVIDER_CACHE: dict[str, BaseLLMProvider] = {}


def get_provider(name: str = "gemini", **kwargs) -> BaseLLMProvider:
    """Returns a singleton or configured instance of the requested LLM provider."""
    normalized_name = name.lower().strip()
    if normalized_name not in _PROVIDER_MAP:
        raise ValueError(f"Unsupported provider '{name}'. Available: {list(_PROVIDER_MAP.keys())}")

    if normalized_name not in _PROVIDER_CACHE or kwargs:
        provider_cls = _PROVIDER_MAP[normalized_name]
        instance = provider_cls(**kwargs)
        if not kwargs:
            _PROVIDER_CACHE[normalized_name] = instance
        return instance

    return _PROVIDER_CACHE[normalized_name]


__all__ = [
    "BaseLLMProvider",
    "CohereProvider",
    "GeminiProvider",
    "GroqProvider",
    "HuggingFaceProvider",
    "LiteLLMProvider",
    "LiteLLMTransport",
    "LlamaCppTransport",
    "LocalProvider",
    "MistralProvider",
    "NvidiaProvider",
    "OpenAICompatibleTransport",
    "OpenRouterProvider",
    "SGLangTransport",
    "VLLMTransport",
    "get_provider",
]
