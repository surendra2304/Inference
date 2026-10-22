from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_messages(messages) -> list[dict[str, Any]]:
    return [
        {"role": m.role, "content": m.content, **({"name": m.name} if getattr(m, "name", None) else {})}
        for m in messages
    ]


def make_cache_key(tenant: str, request) -> str:
    payload = {
        "tenant": tenant,
        "model": request.model,
        "messages": canonical_messages(request.messages),
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "schema": request.response_schema,
        "extra": dict(sorted(request.extra.items())),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()
