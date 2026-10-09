"""Output quality checks, and the only quality numbers this process reports.

Until this module was changed, ``get_quality_report()`` returned constants: an
"overall_output_quality_score" of 96.4, a "syntactic_validity_rate_pct" of 99.2, a calibration
table saying "WELL_CALIBRATED", and agent rankings with scores to one decimal place. None of
those were computed from anything. ``GET /v1/admin/analytics/quality`` served them as if they
were measurements, and that is the kind of number an operator makes decisions from.

The rule now: a figure is reported only when this process has measured it, and the report says
how many observations back it. Anything that has not been measured is ``null`` with an explicit
``not_measured`` status, so a missing measurement can never be mistaken for a good one.

Syntactic validity is the measurement that exists today. The code-generation service calls
:meth:`QualityAssuranceService.check_output` for every generation it validates, and that call
is the only thing that moves the counters.
"""

from __future__ import annotations

import ast
import json
import threading
from dataclasses import dataclass
from typing import Any

#: The languages this module can actually verify. Anything else is "not verified", which is a
#: different claim from "valid", and the caller must not treat it as one.
VERIFIABLE_LANGUAGES = frozenset({"python", "json"})


@dataclass(frozen=True)
class SyntaxVerdict:
    language: str
    verified: bool
    is_valid: bool | None
    error: str | None


class QualityAssuranceService:
    """Checks outputs and keeps counts of what it checked."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._checked: dict[str, int] = {}
        self._valid: dict[str, int] = {}

    # -- checks ------------------------------------------------------------------------

    def check_output(self, text: str, language: str) -> SyntaxVerdict:
        """Verify ``text`` as ``language`` and record the result if it was verifiable."""
        lang = language.lower().strip()
        if lang == "python":
            try:
                ast.parse(text)
                verdict = SyntaxVerdict(lang, True, True, None)
            except SyntaxError as exc:
                verdict = SyntaxVerdict(lang, True, False, f"SyntaxError at line {exc.lineno}: {exc.msg}")
        elif lang == "json":
            try:
                json.loads(text)
                verdict = SyntaxVerdict(lang, True, True, None)
            except ValueError as exc:
                verdict = SyntaxVerdict(lang, True, False, f"JSONDecodeError: {exc}")
        else:
            verdict = SyntaxVerdict(lang, False, None, None)

        if verdict.verified:
            self.record(lang, bool(verdict.is_valid))
        return verdict

    @staticmethod
    def would_parse(text: str, language: str) -> bool | None:
        """Whether ``text`` parses as ``language``, or ``None`` if this module cannot tell.

        Unlike :meth:`check_output` this does not touch the counters. Use it to inspect input
        (for example, the code a caller asked about), where a recorded check would be counted
        as an output the agent produced.
        """
        lang = language.lower().strip()
        if lang == "python":
            try:
                ast.parse(text)
                return True
            except SyntaxError:
                return False
        if lang == "json":
            try:
                json.loads(text)
                return True
            except ValueError:
                return False
        return None

    def evaluate_code_syntax(self, code: str, language: str = "python") -> dict[str, Any]:
        """Backwards-compatible shape for existing callers: ``{"is_valid", "error"}``.

        Unlike the earlier version, an unverifiable language now reports ``is_valid: None``
        rather than ``True``: "we did not check" must not read as "it is valid".
        """
        verdict = self.check_output(code, language)
        return {"is_valid": verdict.is_valid, "error": verdict.error}

    def record(self, language: str, valid: bool) -> None:
        with self._lock:
            self._checked[language] = self._checked.get(language, 0) + 1
            if valid:
                self._valid[language] = self._valid.get(language, 0) + 1

    # -- report ------------------------------------------------------------------------

    def get_quality_report(self) -> dict[str, Any]:
        """Only measured figures. Everything else is ``None`` with a stated status."""
        with self._lock:
            checked = dict(self._checked)
            valid = dict(self._valid)
        total = sum(checked.values())
        by_language = {
            lang: {
                "checked": count,
                "valid": valid.get(lang, 0),
                "syntactic_validity_rate_pct": round(100.0 * valid.get(lang, 0) / count, 1),
            }
            for lang, count in sorted(checked.items())
        }
        return {
            "evidence_class": "process_observed_only",
            "evidence_scope": (
                "Counts of outputs this process verified since it started. Nothing here is a "
                "benchmark, and nothing is inferred for outputs that were not checked."
            ),
            "verifiable_languages": sorted(VERIFIABLE_LANGUAGES),
            "syntactic_validity": {
                "status": "measured" if total else "not_measured",
                "sample_size": total,
                "rate_pct": round(100.0 * sum(valid.values()) / total, 1) if total else None,
                "by_language": by_language,
            },
            "overall_output_quality_score": {"status": "not_measured", "value": None},
            "confidence_calibration": {
                "status": "not_measured",
                "reason": "no outcome labels are recorded, so stated confidence cannot be compared with accuracy",
                "bins": None,
            },
            "agent_quality_rankings": {
                "status": "not_measured",
                "reason": "no per-agent outcome labels are recorded",
                "rankings": None,
            },
        }


quality_assurance_service = QualityAssuranceService()
