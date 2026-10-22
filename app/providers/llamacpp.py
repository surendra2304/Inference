from __future__ import annotations

from .openai_compatible import OpenAICompatibleTransport


class LlamaCppTransport(OpenAICompatibleTransport):
    """llama.cpp server transport. Use per-model endpoint entries for slot/adapter isolation."""

    pass
