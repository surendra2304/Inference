"""Ingress Prompt Injection Barrier, Homoglyph Normalization, and Content Defense."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from app.security.prompt_isolation import (
    scrub_credentials,
    strip_zero_width_and_invisible,
)

# Common homoglyph mappings (Cyrillic, Greek, lookalikes -> ASCII Latin)
_HOMOGLYPH_MAP: dict[str, str] = {
    "а": "a", "А": "A", "с": "c", "С": "C", "е": "e", "Е": "E",
    "о": "o", "О": "O", "р": "p", "Р": "P", "х": "x", "Х": "X",
    "у": "y", "У": "Y", "і": "i", "І": "I", "ј": "j", "Ј": "J",
    "ѕ": "s", "Ѕ": "S", "ԁ": "d", "Ԃ": "D", "ԛ": "q", "Ԝ": "w",
    "ԡ": "l", "ν": "v", "ο": "o", "ρ": "p", "τ": "t", "υ": "u",
    "ω": "w", "α": "a", "β": "b", "γ": "y", "κ": "k",
}

_HOMOGLYPH_TRANS = str.maketrans(_HOMOGLYPH_MAP)

# Strict prompt injection and delimiter breakout patterns
_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+|any\s+)?(?:previous\s+|prior\s+)?(?:instructions|prompts|directives|rules)", re.I),
    re.compile(r"disregard\s+(?:all\s+|any\s+)?(?:previous\s+|prior\s+)?(?:instructions|prompts|directives|rules)", re.I),
    re.compile(r"forget\s+(?:all|everything)\s+you\s+(?:were\s+told|learned|know)", re.I),
    re.compile(r"reveal\s+(?:the\s+)?(?:system|developer|hidden|secret)\s+prompt", re.I),
    re.compile(r"(?:print|output|display|show|leak)\s+(?:the\s+)?(?:system|developer)\s+(?:prompt|instructions)", re.I),
    re.compile(r"exfiltrat|steal.{0,40}(?:key|token|secret|password|credential)", re.I),
    re.compile(r"system\s+prompt\s+override", re.I),
    re.compile(r"<\s*/?\s*im_start\s*>", re.I),
    re.compile(r"<\s*/?\s*im_end\s*>", re.I),
    re.compile(r"<\s*/?\s*system\s*>", re.I),
    re.compile(r"\[\s*/?\s*INST\s*\]", re.I),
    re.compile(r"<<\s*/?\s*SYS\s*>>", re.I),
    re.compile(r"<\s*\|(?:im_start|im_end|endoftext|startoftext)\|\s*>", re.I),
    re.compile(r"<\s*/?\s*turn_(?:start|end)\s*>", re.I),
    re.compile(r"you\s+are\s+now\s+(?:in\s+developer\s+mode|dan|jailbreak)", re.I),
]

# Markdown exfiltration image link pattern: ![alt](url?data=...)
_MARKDOWN_EXFILTRATION_PATTERN = re.compile(
    r"!\[([^\]]*)\]\((https?://[^\s)]+?(?:\?[^\s)]*|\&[^\s)]*))\)",
    re.IGNORECASE,
)


def normalize_homoglyphs(text: str) -> str:
    """Normalizes Unicode text and maps common Cyrillic/Greek lookalikes to Latin."""
    # Decompose compatibility characters
    decomposed = unicodedata.normalize("NFKD", text)
    # Map explicit homoglyphs
    return decomposed.translate(_HOMOGLYPH_TRANS)


def defang_markdown_exfiltration(text: str) -> str:
    """Defangs markdown images that may attempt to exfiltrate context via query parameters."""
    def _defang(match: re.Match) -> str:
        alt = match.group(1)
        url = match.group(2)
        return f"[defanged-img: {alt}]({url})"

    return _MARKDOWN_EXFILTRATION_PATTERN.sub(_defang, text)


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    sanitized: str
    findings: tuple[str, ...]
    risk_level: Literal["SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"] = "SAFE"
    defanged_count: int = 0


def inspect(text: str) -> GuardResult:
    """Performs deep security inspection on untrusted ingress prompts."""
    # 1. Strip zero-width and invisible control code points
    cleaned = strip_zero_width_and_invisible(text)

    # 2. Defang markdown image exfiltration links
    defanged = defang_markdown_exfiltration(cleaned)
    defanged_count = 1 if defanged != cleaned else 0

    # 3. Check for injection patterns on normalized text (homoglyph-immune)
    normalized = normalize_homoglyphs(defanged)
    findings = []
    for p in _PATTERNS:
        if p.search(normalized) or p.search(cleaned):
            findings.append(p.pattern)

    # 4. Determine risk level
    if any("exfiltrat" in f or "jailbreak" in f or "im_start" in f for f in findings):
        risk = "CRITICAL"
    elif any("turn_" in f or "SYS" in f or "INST" in f or "system" in f for f in findings):
        risk = "HIGH"
    elif len(findings) >= 2:
        risk = "HIGH"
    elif findings:
        risk = "MEDIUM"
    elif defanged_count > 0:
        risk = "LOW"
    else:
        risk = "SAFE"

    # 5. Scrub credentials thoroughly
    sanitized = scrub_credentials(defanged)
    return GuardResult(
        allowed=not findings,
        sanitized=sanitized,
        findings=tuple(findings),
        risk_level=risk,
        defanged_count=defanged_count,
    )


def guard_messages(messages: list[dict]) -> list[dict]:
    """Guards and sanitizes all messages in a conversation payload."""
    out = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            result = inspect(c)
            out.append({**m, "content": result.sanitized})
        else:
            out.append(m)
    return out
