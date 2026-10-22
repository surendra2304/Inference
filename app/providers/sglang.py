from __future__ import annotations

from .openai_compatible import OpenAICompatibleTransport


class SGLangTransport(OpenAICompatibleTransport):
    """SGLang OpenAI-compatible backend; structured-output support is capability-gated by endpoint metadata."""

    pass
