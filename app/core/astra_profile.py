"""ASTRA Reasoning Profile and Persona Configuration for Inference.

Exposes ASTRA as the central multi-model reasoning council of the FRIDAY Universe.
In accordance with Rule 16, ASTRA is defined as a multi-model reasoning identity,
never falsely claimed as a proprietary 'GPT-6' model.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class AstraProfile(BaseModel):
    """Canonical ASTRA reasoning persona and runtime profile for FRIDAY Universe."""

    name: str = "ASTRA"
    role: str = "Central Deliberative Reasoning Council & Consensus Synthesizer"
    identity_description: str = (
        "ASTRA is the multi-model deliberative reasoning identity of the FRIDAY Universe. "
        "It orchestrates multi-agent debate, evidence verification, and consensus synthesis "
        "across verified free and open-source model providers."
    )
    allowed_providers: list[str] = Field(
        default_factory=lambda: ["groq", "gemini", "mistral", "openrouter", "nvidia", "cohere", "huggingface"]
    )
    model_routing: dict[str, str] = Field(
        default_factory=lambda: {
            "fast": "gemini-2.0-flash",
            "reasoning": "llama-3.3-70b-versatile",
            "synthesis": "gemini-2.0-flash",
            "coding": "mistral-large-latest",
            "fallback": "openrouter/free",
        }
    )
    temperature: float = 0.3
    maximum_deliberation_depth: int = 3
    evidence_policy: str = "STRICT_EMPIRICAL"
    response_style: str = "structured_analytical"
    # Strict Invariant: Do not claim GPT-6 unless an actual provider exposes that identifier
    claim_gpt6: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


# Global canonical profile instance
astra_profile = AstraProfile()
