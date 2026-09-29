"""Secret-free inventory of configured provider key counts.

This reports configuration presence only. It never prints key values and does
not claim that a credential is valid, unique across accounts, or has quota.
"""

from __future__ import annotations

from app.core.config import settings

PROVIDERS = (
    "GEMINI",
    "GROQ",
    "MISTRAL",
    "OPENROUTER",
    "COHERE",
    "HUGGINGFACE",
    "NVIDIA",
)


def provider_key_counts(config=None) -> dict[str, int]:
    """Return the deduplicated number of configured keys for each provider."""
    if config is None:
        config = settings
    return {provider: len(config.get_provider_keys(provider)) for provider in PROVIDERS}


def main() -> None:
    """Print counts only, suitable for a local environment audit."""
    counts = provider_key_counts()
    for provider, count in counts.items():
        print(f"{provider}: {count}")
    print(f"TOTAL_CONFIGURED_PROVIDER_KEYS: {sum(counts.values())}")
    print("Credential validity, account uniqueness, and provider quota were not checked.")


if __name__ == "__main__":
    main()
