from __future__ import annotations

import re
from dataclasses import dataclass

from app.security.prompt_isolation import (
    scrub_credentials,
    strip_zero_width_and_invisible,
)

_PATTERNS = [
    re.compile(r"ignore (?:all|any) previous instructions", re.I),
    re.compile(r"reveal (?:the )?(?:system|developer) prompt", re.I),
    re.compile(r"exfiltrat|steal.{0,40}(?:key|token|secret)", re.I),
    re.compile(r"system prompt override", re.I),
    re.compile(r"<\s*/?\s*im_start\s*>", re.I),
]


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    sanitized: str
    findings: tuple[str, ...]


def inspect(text: str) -> GuardResult:
    # 1. Strip zero-width and invisible control code points
    cleaned = strip_zero_width_and_invisible(text)
    findings = []
    for p in _PATTERNS:
        if p.search(cleaned):
            findings.append(p.pattern)
    # 2. Scrub credentials thoroughly
    sanitized = scrub_credentials(cleaned)
    return GuardResult(not findings, sanitized, tuple(findings))


def guard_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            result = inspect(c)
            out.append({**m, "content": result.sanitized})
        else:
            out.append(m)
    return out
