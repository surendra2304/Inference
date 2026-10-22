from __future__ import annotations

from .contracts import CompletionRequest


def fallback_models(request: CompletionRequest, failed: str) -> list[str]:
    requested = request.model if request.model not in {"", "auto"} else "auto"
    candidates = [x for x in ("auto", requested) if x not in {failed, ""}]
    if requested == "auto":
        candidates = ["auto"]
    return candidates
