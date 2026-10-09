"""Debugging Intelligence Service for FORGE's automated recovery engine.

What this replaced. The earlier service sent the error to a model and returned a fixed template
as the diagnosis: ``root_cause`` was ``"Exception '<error>' identified in code execution path."``,
``fix_strategy`` was the same sentence every time, ``confidence`` was a constant 0.89 even when
no provider answered, and ``patch_code`` was the model's raw reply whatever it said. [FACT]
measured on ``POST /v1/forge/debug`` with the model returning prose: the response was
``diagnosis`` = the error echoed back, confidence 0.89, and a patch that was prose.

The rules now:

* The model is asked for JSON with ``root_cause``, ``fix_strategy`` and ``patch_code``. A reply
  that does not validate is **not a diagnosis**: ``diagnosis_status`` is ``unparsed`` or
  ``unavailable``, ``root_cause`` says so and repeats the reported error as *unexplained*,
  ``patch_code`` is ``None``, and confidence is 0.0.
* A patch is returned only when the model supplied one, and it is syntax-checked whenever the
  code under debug parses as Python, which is the only language this service can check.
  A patch that fails the check is withheld, and ``patch_check`` says why.
* ``confidence`` is a prior for a diagnosis the model supplied (0.6), not a calibrated measure.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from app.providers.unified_manager import (
    UnifiedExecutionRequest,
    unified_provider_manager,
)
from app.services.quality_assurance import quality_assurance_service
from app.utils.model_json import extract_json_object

DIAGNOSED_CONFIDENCE = 0.6
PROMPT_TRACEBACK_CHARS = 1500
PROMPT_CONTEXT_CHARS = 2000


class DebugRequest(BaseModel):
    error: str = Field(..., description="Error message or exception string")
    traceback: str = Field(default="", description="Full stack trace")
    code_context: str = Field(..., description="Relevant code snippet where error occurred")
    attempted_fixes: list[str] = Field(default_factory=list, description="Prior failed fix attempts to avoid repeats")
    verification_failure: str | None = Field(default="", description="Test output or assertion error")


class DebugResponse(BaseModel):
    root_cause: str
    fix_strategy: str
    patch_code: str | None = None
    confidence: float
    latency_ms: float
    diagnosis_status: Literal["diagnosed", "unparsed", "unavailable"]
    patch_check: Literal["not_provided", "parses", "not_checked", "withheld"] = "not_provided"


class _ModelDiagnosis(BaseModel):
    """What the model must return. Anything that does not validate is not a diagnosis."""

    root_cause: str = Field(min_length=1)
    fix_strategy: str = Field(min_length=1)
    patch_code: str | None = None


class DebuggingIntelligenceService:
    """Asks a model for a structured diagnosis and reports only what it actually obtained."""

    async def diagnose_and_fix(self, req: DebugRequest) -> DebugResponse:
        start_time = time.perf_counter()

        prompt = (
            "Diagnose and solve this error in FORGE.\n"
            f"Error: {req.error}\n"
            f"Traceback:\n{req.traceback[:PROMPT_TRACEBACK_CHARS]}\n\n"
            f"Code Context:\n```\n{req.code_context[:PROMPT_CONTEXT_CHARS]}\n```\n\n"
            f"Attempted Fixes: {', '.join(req.attempted_fixes) if req.attempted_fixes else 'None'}\n\n"
            "Respond with ONLY a JSON object, no prose, in this shape:\n"
            '{"root_cause": "the exact cause, naming the variable, key or line involved", '
            '"fix_strategy": "what to change and why", '
            '"patch_code": "the corrected code, or null if no code change is needed"}'
        )

        exec_res = await unified_provider_manager.execute(UnifiedExecutionRequest(
            provider="auto",
            agent_role="coder",
            prompt=prompt,
            max_tokens=2500,
            temperature=0.2,
        ))
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        diagnosis: _ModelDiagnosis | None = None
        if exec_res is None or getattr(exec_res, "degraded", False):
            status: Literal["diagnosed", "unparsed", "unavailable"] = "unavailable"
        else:
            diagnosis = self._parse(exec_res.content or "")
            status = "diagnosed" if diagnosis is not None else "unparsed"

        if diagnosis is None:
            return DebugResponse(
                root_cause=(
                    f"Not diagnosed ({'no model answer was available' if status == 'unavailable' else 'the model answer could not be parsed'}). "
                    f"The reported error is unexplained: {req.error}"
                ),
                fix_strategy="None: no model diagnosis was obtained, so no fix is proposed.",
                patch_code=None,
                confidence=0.0,
                latency_ms=elapsed_ms,
                diagnosis_status=status,
                patch_check="not_provided",
            )

        patch, patch_check = self._check_patch(diagnosis.patch_code, req.code_context)
        return DebugResponse(
            root_cause=diagnosis.root_cause,
            fix_strategy=diagnosis.fix_strategy,
            patch_code=patch,
            confidence=DIAGNOSED_CONFIDENCE,
            latency_ms=elapsed_ms,
            diagnosis_status="diagnosed",
            patch_check=patch_check,
        )

    @staticmethod
    def _parse(text: str) -> _ModelDiagnosis | None:
        if not text.strip():
            return None
        try:
            return _ModelDiagnosis.model_validate(extract_json_object(text))
        except (ValueError, ValidationError):
            return None

    @staticmethod
    def _check_patch(
        patch: str | None, code_context: str,
    ) -> tuple[str | None, Literal["not_provided", "parses", "not_checked", "withheld"]]:
        if not patch or not patch.strip():
            return None, "not_provided"
        # Only a Python context gives us a parser for the patch. Without one, the patch is passed
        # through labelled "not_checked", never presented as verified.
        if quality_assurance_service.would_parse(code_context, "python") is not True:
            return patch, "not_checked"
        verdict = quality_assurance_service.check_output(patch, "python")
        if verdict.is_valid:
            return patch, "parses"
        return None, "withheld"


debugging_service = DebuggingIntelligenceService()
