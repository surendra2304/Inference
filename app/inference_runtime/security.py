from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from typing import Any

SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"AIza[A-Za-z0-9_-]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{16,}", re.I),
)


@dataclass(frozen=True)
class SanitizedPayload:
    text: str
    findings: tuple[str, ...] = ()


def redact(text: str) -> SanitizedPayload:
    findings = []
    out = text
    for pat in SECRET_PATTERNS:
        if pat.search(out):
            findings.append(pat.pattern)
            out = pat.sub("[REDACTED]", out)
    return SanitizedPayload(out, tuple(findings))


def safe_log_value(value: Any) -> str:
    raw = repr(value)
    return redact(raw).text[:2000]


def constant_time_equal(provided: str | None, expected: str | None) -> bool:
    if provided is None or expected is None:
        return False
    return hmac.compare_digest(hashlib.sha256(provided.encode()).digest(), hashlib.sha256(expected.encode()).digest())


def is_safe_cache_content(text: str) -> bool:
    return not bool(redact(text).findings)
