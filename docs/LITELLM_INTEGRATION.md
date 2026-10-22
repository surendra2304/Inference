# LiteLLM Integration for Inference

## What this patch actually does

It adds LiteLLM as a compatibility/transport layer behind Inference's existing provider contracts.

Inference remains responsible for:
- task routing
- agent selection
- debate/collaboration
- health policy
- budgets
- memory
- telemetry/provenance
- final synthesis

LiteLLM provides normalized provider calls for models identified by LiteLLM model strings.

## Install

Add to `pyproject.toml`:

```toml
"litellm>=1.101.0,<2.0.0",
```

Then:

```bash
pip install -e ".[dev]"
```

## Enable

```env
INFERENCE_LITELLM_ENABLED=true
INFERENCE_LITELLM_FALLBACK_ENABLED=true
LITELLM_DEFAULT_TIMEOUT=60
LITELLM_DROP_PARAMS=true
```

Provider credentials should remain in environment variables supported by LiteLLM/provider SDKs.

## Model examples

```text
openai/gpt-4o-mini
anthropic/claude-sonnet-4-20250514
gemini/gemini-2.5-flash
openrouter/deepseek/deepseek-r1
groq/llama-3.3-70b-versatile
```

Exact model availability must be verified against the provider/LiteLLM version actually installed.

## Required gateway behavior

Do not bypass `ModelGateway` policy. A LiteLLM call must still be subject to:
- Inference timeout
- provider health
- retry/fallback policy
- request budget
- redacted telemetry
- cancellation
- provenance recording

## Rollout

Use a feature flag and route only a subset of requests through LiteLLM first. Keep existing native provider adapters as the fallback path.

## Security

Never hard-code API keys. Never log request headers, API keys, or full provider raw responses containing secrets.
