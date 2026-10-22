from __future__ import annotations

from .openai_compatible import OpenAICompatibleTransport


class VLLMTransport(OpenAICompatibleTransport):
    """vLLM OpenAI-compatible backend. Keep authentication and endpoint policy outside the model runtime."""

    pass
