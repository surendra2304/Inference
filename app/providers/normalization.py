from __future__ import annotations

from typing import Any

from app.inference_runtime.contracts import CompletionRequest, CompletionResult, ProviderEndpoint


def usage(response: Any) -> tuple[int, int, int]:
    u = getattr(response, "usage", None)
    if u is None and isinstance(response, dict):
        u = response.get("usage")
    if u is None:
        return 0, 0, 0

    def get(name):
        return int(getattr(u, name, u.get(name, 0) if isinstance(u, dict) else 0) or 0)

    p, c, t = get("prompt_tokens"), get("completion_tokens"), get("total_tokens")
    return p, c, t or p + c


def text(response: Any) -> str:
    if isinstance(response, dict):
        try:
            return str(response["choices"][0]["message"].get("content") or "")
        except (KeyError, IndexError, TypeError):
            return str(response.get("output") or response.get("text") or "")
    try:
        return str(response.choices[0].message.content or "")
    except (AttributeError, IndexError):
        return ""


def normalize(
    response: Any, request: CompletionRequest, ep: ProviderEndpoint, latency: float, raw: dict | None = None
) -> CompletionResult:
    p, c, t = usage(response)
    return CompletionResult(
        text=response if isinstance(response, str) else text(response),
        provider=ep.provider,
        model=ep.model,
        latency_seconds=latency,
        prompt_tokens=p,
        completion_tokens=c,
        total_tokens=t,
        cost_usd=(p / 1000) * ep.cost_per_1k_input + (c / 1000) * ep.cost_per_1k_output,
        raw=raw if raw is not None else (response if isinstance(response, dict) else None),
    )
