"""Universal Task Execution and Deliberative Reasoning Routes for FRIDAY Universe."""

from __future__ import annotations

import re
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.analytics.outcome_learning import DetailedOutcomeReport, outcome_learning_engine
from app.inference_runtime.errors import RateLimitExceeded
from app.inference_runtime.quotas import agent_quota_manager
from app.providers.unified_manager import UnifiedExecutionRequest, unified_provider_manager
from app.security.prompt_isolation import (
    sanitize_incoming_payload,
    sanitize_incoming_prompt,
    scrub_credentials,
)
from app.utils.confidence import DEGRADED_CONFIDENCE, UNVERIFIED_MODEL_CONFIDENCE
from app.utils.errors import internal_error
from app.utils.logger import logger

task_router = APIRouter(tags=["FRIDAY Universe Universal Task Protocol"])


class TaskEnvelopeModel(BaseModel):
    """Universal Task Envelope accepted from any agent in the FRIDAY Universe."""

    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    source_agent: str = Field(default="friday")
    target_agent: str = Field(default="inference")
    action: str = Field(default="reason")
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: str = Field(default="normal")
    created_at: str | None = None
    trace_id: str | None = None


class TaskResultModel(BaseModel):
    """Universal Task Result returned to calling agent."""

    task_id: str
    target_agent: str = "inference"
    status: str = "SUCCESS"
    result: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    error: str | None = None
    execution_time_ms: int = 0
    receipt: dict[str, Any] | None = None


class AstraReasonRequest(BaseModel):
    """Deliberative reasoning consultation request."""

    prompt: str = Field(..., description="The query or problem requiring multi-perspective reasoning")
    context: dict[str, Any] = Field(default_factory=dict)
    mode: str = Field(default="deliberative", description="deliberative | consensus | fast")
    caller: str = Field(default="friday")


class AstraReasonResponse(BaseModel):
    """Deliberative reasoning synthesis."""

    task_id: str
    synthesis: str
    confidence: float
    model_used: str
    provider_used: str
    latency_ms: int
    perspective: str


def _extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Extracts (language, code) tuples from markdown code blocks."""
    pattern = re.compile(r"```([a-zA-Z0-9_\-\.\+]*)\n(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    return matches


def _format_forge_result(content: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Structures code generation outputs for Forge."""
    blocks = _extract_code_blocks(content)
    primary_code = blocks[0][1].strip() if blocks else content
    lang = blocks[0][0].strip().lower() if blocks and blocks[0][0].strip() else payload.get("language", "python")
    tests = ""
    for _, block_code in blocks[1:]:
        if "test" in block_code.lower() or "assert " in block_code:
            tests = block_code.strip()
            break
    files = payload.get("files") or []
    if not files and blocks:
        default_file = payload.get("target_file") or f"generated_code.{'py' if lang == 'python' else 'ts'}"
        files = [{"path": default_file, "content": primary_code, "action": "create"}]
    return {
        "code": primary_code,
        "files": files,
        "language": lang,
        "tests": tests,
        "architecture_notes": content[:300].strip(),
    }


def _format_vyralx_result(content: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Structures viral scriptwriting outputs for VyralX."""
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    hook = lines[0] if lines else "Attention Hook"
    cta = lines[-1] if len(lines) > 1 else "Subscribe & Follow"
    beats = []
    for line in lines[1:-1]:
        if any(marker in line.lower() for marker in ("beat", "scene", "cue", "hook", "payoff", ":")):
            beats.append(line)
        if len(beats) >= 5:
            break
    if not beats:
        beats = ["Introduction Hook", "Core Tension", "Solution Reveal", "Call to Action"]
    return {
        "hook": hook,
        "narrative_arc": payload.get("topic") or "High-retention video narrative",
        "retention_beats": beats,
        "call_to_action": cta,
        "script_text": content,
        "target_audience": payload.get("target_audience") or "Tech and developer audience",
    }


def _format_intelx_result(content: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Structures intelligence research and claim extraction outputs for IntelX."""
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    claims = []
    idx = 1
    for line in lines:
        cleaned_line = line.lstrip("-*1234567890. ")
        if cleaned_line.endswith(":") or any(cleaned_line.lower().startswith(h) for h in ("executive briefing", "briefing", "summary", "key findings", "findings")):
            continue
        if len(cleaned_line) > 15 and not cleaned_line.startswith("#"):
            claims.append({
                "claim_id": f"claim_{idx}",
                "statement": cleaned_line,
                # Extracting a sentence from model output does not independently verify it.
                "confidence": None,
                "verification_status": "unverified",
                "evidence_markers": [],
                "source": payload.get("source"),
            })
            idx += 1
        if len(claims) >= 10:
            break
    if not claims and content.strip():
        # Preserve a non-empty response as an extracted candidate, never as a verified claim.
        claims.append({
            "claim_id": "claim_1",
            "statement": content[:200].strip(),
            "confidence": None,
            "verification_status": "unverified",
            "evidence_markers": [],
            "source": payload.get("source"),
        })
    return {
        "claims": claims,
        "total_claims": len(claims),
        "summary": content[:300].strip(),
        "contradictions": [],
        # Uncertainty has not been calibrated from independent evidence here.
        "uncertainty_score": None,
    }


@task_router.post("/v1/task/execute", response_model=TaskResultModel, status_code=status.HTTP_200_OK)
async def execute_task(
    envelope: TaskEnvelopeModel,
) -> TaskResultModel:
    """Execute a structured TaskEnvelope from any peer agent in the FRIDAY Universe.

    Enforces:
    1. Safety Invariant 3: Calling agent token & request quotas per minute.
    2. Objective 4: Ingress prompt injection barrier & zero-width unicode stripping.
    3. Objective 2: Specialist task reasoning for Forge (code), VyralX (script), and IntelX (claims).
    4. Objective 3: Direct telemetry persistence to runs & strategies.
    """
    t0 = time.time()
    source = envelope.source_agent.lower().strip()
    action = envelope.action.lower().strip()
    logger.info(f"[TASK_EXECUTE] Received action '{action}' from '{source}' (Task ID: {envelope.task_id})")

    # 1. Enforce Token Quotas per minute (Safety Invariant 3)
    estimated_tokens = max(100, len(str(envelope.payload)) // 4)
    try:
        agent_quota_manager.admit(source, estimated_tokens=estimated_tokens)
    except RateLimitExceeded as rle:
        logger.warning(f"[TASK_EXECUTE] Token quota exceeded for '{source}': {rle}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(rle),
        )

    # 2. Ingress Prompt Injection Barrier & Zero-Width Stripping (Objective 4)
    raw_prompt = (
        envelope.payload.get("prompt")
        or envelope.payload.get("question")
        or envelope.payload.get("query")
        or envelope.payload.get("task")
        or str(envelope.payload)
    )
    clean_prompt = sanitize_incoming_prompt(raw_prompt)
    clean_payload = sanitize_incoming_payload(envelope.payload)

    # 3. Detect specialist agent requirements
    is_forge = source in ("forge", "code_generator") or action in ("code_generation", "generate_code", "refactor")
    is_vyralx = source in ("vyralx", "scriptwriter") or action in ("scriptwriting", "script", "generate_script")
    is_intelx = source in ("intelx", "claim_extractor") or action in ("claim_extraction", "extract_claims", "research", "verify")

    if is_forge:
        role = "software_architect"
        augmented_prompt = f"[FORGE CODE GENERATION PROTOCOL]\nProduce robust, modular production code and unit tests for:\n{clean_prompt}"
    elif is_vyralx:
        role = "viral_scriptwriter"
        augmented_prompt = f"[VYRALX SCRIPTWRITING PROTOCOL]\nProduce a high-retention, engaging script with hook, retention beats, and CTA for:\n{clean_prompt}"
    elif is_intelx:
        role = "intelligence_analyst"
        augmented_prompt = f"[INTELX CLAIM EXTRACTION PROTOCOL]\nExtract atomic factual claims, verification status, and evidence markers for:\n{clean_prompt}"
    else:
        role = "system_architect"
        augmented_prompt = clean_prompt

    try:
        req = UnifiedExecutionRequest(
            provider="auto",
            agent_role=role,
            prompt=augmented_prompt,
            context=clean_payload.get("context", {}) if isinstance(clean_payload, dict) else {},
            fast_lane=True,
            max_tokens=2048,
        )
        resp = await unified_provider_manager.execute(req)
        lat = int((time.time() - t0) * 1000)

        # 4. Handle degradation honestly
        if resp.degraded:
            clean_error = scrub_credentials(resp.error or "all provider calls failed")
            return TaskResultModel(
                task_id=envelope.task_id,
                target_agent="inference",
                status="DEGRADED",
                error=clean_error,
                result={
                    "response": "",
                    "model_used": resp.model_used,
                    "provider_used": resp.provider_used,
                    "token_usage": {},
                },
                summary="No model provider produced output; no answer fabricated.",
                execution_time_ms=lat,
            )

        # 5. Format Specialist Task Deliverables (Objective 2)
        if is_forge:
            structured_data = _format_forge_result(resp.content, clean_payload if isinstance(clean_payload, dict) else {})
            task_type = "code_generation"
        elif is_vyralx:
            structured_data = _format_vyralx_result(resp.content, clean_payload if isinstance(clean_payload, dict) else {})
            task_type = "scriptwriting"
        elif is_intelx:
            structured_data = _format_intelx_result(resp.content, clean_payload if isinstance(clean_payload, dict) else {})
            task_type = "claim_extraction"
        else:
            structured_data = {"response": resp.content}
            task_type = "general_reasoning"

        structured_data["model_used"] = resp.model_used
        structured_data["provider_used"] = resp.provider_used
        structured_data["token_usage"] = resp.token_usage

        summary_text = (
            structured_data.get("summary")
            or structured_data.get("hook")
            or structured_data.get("architecture_notes")
            or resp.content[:250]
        )

        # 6. Outcome Learning & Database Persistence (Objective 3)
        outcome_report = DetailedOutcomeReport(
            consumer=source if source in ("trading_bot", "forge", "nexus", "sentinel", "intelx", "futuris", "friday", "human") else "friday",
            request_id=envelope.task_id,
            outcome="success",
            detail=f"{task_type}_completed",
            measured_metrics={
                "latency_ms": lat,
                "latency_s": lat / 1000.0,
                "prompt_tokens": resp.token_usage.get("prompt_tokens", estimated_tokens),
                "completion_tokens": resp.token_usage.get("completion_tokens", len(resp.content) // 4),
                "model": resp.model_used,
            },
            task_type=task_type,
            provider_used=resp.provider_used,
        )
        outcome_learning_engine.record_outcome(outcome_report)

        return TaskResultModel(
            task_id=envelope.task_id,
            target_agent="inference",
            status="SUCCESS",
            result=structured_data,
            summary=summary_text,
            execution_time_ms=lat,
        )
    except HTTPException:
        raise
    except Exception as e:
        lat = int((time.time() - t0) * 1000)
        detail, _reference = internal_error(logger, e, doing_what="task execution", prefix="task")
        return TaskResultModel(
            task_id=envelope.task_id,
            target_agent="inference",
            status="ERROR",
            error=scrub_credentials(detail),
            execution_time_ms=lat,
        )


@task_router.post("/v1/astra/reason", response_model=AstraReasonResponse, status_code=status.HTTP_200_OK)
async def astra_reason(request: AstraReasonRequest) -> AstraReasonResponse:
    """Deliberative high-level reasoning gateway using optimal free model pool."""
    t0 = time.time()
    clean_prompt = sanitize_incoming_prompt(request.prompt)
    logger.info(f"[ASTRA_REASON] Reasoning query from '{request.caller}': {clean_prompt[:80]}")

    try:
        req = UnifiedExecutionRequest(
            provider="auto",
            agent_role="system_architect",
            prompt=f"[ASTRA DELIBERATIVE REASONING]\nEvaluate context and provide structured resolution:\n{clean_prompt}",
            context=sanitize_incoming_payload(request.context),
            fast_lane=request.mode == "fast",
            max_tokens=2000,
        )
        resp = await unified_provider_manager.execute(req)
        lat = int((time.time() - t0) * 1000)

        if resp.degraded:
            return AstraReasonResponse(
                task_id=f"astra_deg_{uuid.uuid4().hex[:8]}",
                synthesis=(
                    "No model provider produced output for this reasoning request; "
                    "ASTRA returns no synthesis instead of fabricating one."
                ),
                confidence=DEGRADED_CONFIDENCE,
                model_used=resp.model_used,
                provider_used=resp.provider_used,
                latency_ms=lat,
                perspective="Central Reasoning Council (ASTRA) — degraded",
            )

        return AstraReasonResponse(
            task_id=f"astra_{uuid.uuid4().hex[:8]}",
            synthesis=resp.content,
            confidence=UNVERIFIED_MODEL_CONFIDENCE,
            model_used=resp.model_used,
            provider_used=resp.provider_used,
            latency_ms=lat,
            perspective="Central Reasoning Council (ASTRA)",
        )
    except Exception as e:
        logger.error(f"[ASTRA_REASON] Error: {e}")
        lat = int((time.time() - t0) * 1000)
        return AstraReasonResponse(
            task_id=f"astra_err_{uuid.uuid4().hex[:8]}",
            synthesis=f"Reasoning failure: {scrub_credentials(str(e))}",
            confidence=DEGRADED_CONFIDENCE,
            model_used="none",
            provider_used="none",
            latency_ms=lat,
            perspective="Error Handler",
        )
