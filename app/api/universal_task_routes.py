"""Universal Task Execution and Deliberative Reasoning Routes for FRIDAY Universe."""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from typing import Any, cast

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.analytics.outcome_learning import ConsumerName, DetailedOutcomeReport, outcome_learning_engine
from app.inference_runtime.errors import RateLimitExceeded
from app.inference_runtime.prompt_guard import inspect as inspect_prompt
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


class BatchTaskEnvelopeModel(BaseModel):
    """Batch execution payload for multiple Universal Task envelopes."""

    batch_id: str = Field(default_factory=lambda: f"batch_{uuid.uuid4().hex[:12]}")
    source_agent: str = Field(default="friday")
    tasks: list[TaskEnvelopeModel] = Field(..., min_length=1, max_length=50, description="Tasks to execute")
    concurrency_limit: int = Field(default=5, ge=1, le=20, description="Max concurrent executions")
    fail_fast: bool = Field(default=False, description="Abort remaining tasks upon first failure")
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchTaskResultModel(BaseModel):
    """Aggregated batch execution result."""

    batch_id: str
    target_agent: str = "inference"
    total_tasks: int
    succeeded_tasks: int
    failed_tasks: int
    degraded_tasks: int
    execution_time_ms: int
    results: list[TaskResultModel]
    summary: dict[str, Any] = Field(default_factory=dict)


class TaskValidateRequest(BaseModel):
    """Pre-flight validation request for Universal Task Envelope."""

    source_agent: str = Field(default="friday")
    action: str = Field(default="reason")
    payload: dict[str, Any] = Field(default_factory=dict)


class TaskValidateResponse(BaseModel):
    """Validation report without executing LLM inference."""

    valid: bool
    source_agent: str
    action: str
    estimated_tokens: int
    quota_available: bool
    prompt_safe: bool
    findings: list[str] = Field(default_factory=list)
    risk_level: str = "SAFE"
    recommended_model: str = "auto"


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
    claims: list[dict[str, Any]] = []
    idx = 1
    for line in lines:
        cleaned_line = line.lstrip("-*1234567890. ")
        if cleaned_line.endswith(":") or any(
            cleaned_line.lower().startswith(h)
            for h in ("executive briefing", "briefing", "summary", "key findings", "findings")
        ):
            continue
        if len(cleaned_line) > 15 and not cleaned_line.startswith("#"):
            claims.append({
                "claim_id": f"claim_{idx}",
                "statement": cleaned_line,
                "confidence": None,
                "verification_status": "unverified",
                "evidence_markers": [],
                "source": payload.get("source"),
            })
            idx += 1
        if len(claims) >= 10:
            break
    if not claims and content.strip():
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
    is_intelx = source in ("intelx", "claim_extractor") or action in (
        "claim_extraction", "extract_claims", "research", "verify"
    )

    if is_forge:
        role = "software_architect"
        augmented_prompt = (
            f"[FORGE CODE GENERATION PROTOCOL]\nProduce robust, modular production code and unit tests for:\n{clean_prompt}"
        )
    elif is_vyralx:
        role = "viral_scriptwriter"
        augmented_prompt = (
            f"[VYRALX SCRIPTWRITING PROTOCOL]\nProduce a high-retention, engaging script with hook, retention beats, and CTA for:\n{clean_prompt}"
        )
    elif is_intelx:
        role = "intelligence_analyst"
        augmented_prompt = (
            f"[INTELX CLAIM EXTRACTION PROTOCOL]\nExtract atomic factual claims, verification status, and evidence markers for:\n{clean_prompt}"
        )
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
                    # Usage is empty when no provider completed; never synthesize token counts.
                    "token_usage": resp.token_usage or {},
                    "confidence": DEGRADED_CONFIDENCE,
                    "degraded": True,
                },
                summary=f"Task degraded: {clean_error}",
                execution_time_ms=lat,
            )

        content = resp.content

        # 5. Format structured response per consumer contract
        if is_forge:
            structured_data = _format_forge_result(content, envelope.payload)
            summary_text = f"Forge code generated ({len(structured_data['code'])} chars, {len(structured_data['files'])} files)"
        elif is_vyralx:
            structured_data = _format_vyralx_result(content, envelope.payload)
            summary_text = f"VyralX script generated ({len(structured_data['retention_beats'])} beats)"
        elif is_intelx:
            structured_data = _format_intelx_result(content, envelope.payload)
            summary_text = f"IntelX research extracted {structured_data['total_claims']} factual claims"
        else:
            structured_data = {
                "response": content,
                "confidence": UNVERIFIED_MODEL_CONFIDENCE,
                "model_used": resp.model_used,
                "provider_used": resp.provider_used,
            }
            summary_text = f"Execution completed via {resp.provider_used}/{resp.model_used}"

        # 6. Record outcome and persist to runs & strategies
        try:
            tokens_dict = resp.token_usage or {}
            outcome_report = DetailedOutcomeReport(
                consumer=cast(ConsumerName, source) if source in ("trading_bot", "forge", "nexus", "sentinel", "intelx", "futuris", "friday", "human") else "friday",
                request_id=envelope.task_id,
                outcome="success",
                detail=summary_text,
                task_type=action,
                provider_used=resp.provider_used,
                agent_composition=[source, "inference"],
                stated_confidence=UNVERIFIED_MODEL_CONFIDENCE,
                measured_metrics={
                    "latency_ms": lat,
                    "latency_seconds": lat / 1000.0,
                    "prompt_tokens": tokens_dict.get("prompt_tokens", estimated_tokens),
                    "completion_tokens": tokens_dict.get("completion_tokens", len(content) // 4),
                    "total_tokens": tokens_dict.get("total_tokens", estimated_tokens + len(content) // 4),
                    "model": resp.model_used,
                },
            )
            outcome_learning_engine.record_outcome(outcome_report)
        except Exception as o_err:
            logger.debug(f"[TASK_EXECUTE] Telemetry recording skipped: {o_err}")

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


@task_router.post("/v1/task/batch", response_model=BatchTaskResultModel, status_code=status.HTTP_200_OK)
async def execute_task_batch(batch_envelope: BatchTaskEnvelopeModel) -> BatchTaskResultModel:
    """Executes a batch of Universal Tasks in parallel with bounded concurrency and per-agent quota enforcement."""
    t0 = time.time()
    source = batch_envelope.source_agent.lower().strip()
    tasks = batch_envelope.tasks
    logger.info(
        f"[TASK_BATCH] Starting batch '{batch_envelope.batch_id}' with {len(tasks)} tasks from '{source}' "
        f"(concurrency: {batch_envelope.concurrency_limit}, fail_fast: {batch_envelope.fail_fast})"
    )

    sem = asyncio.Semaphore(batch_envelope.concurrency_limit)
    aborted = False

    async def _run_single(task_item: TaskEnvelopeModel) -> TaskResultModel:
        nonlocal aborted
        if aborted and batch_envelope.fail_fast:
            return TaskResultModel(
                task_id=task_item.task_id,
                target_agent="inference",
                status="ABORTED",
                error="Batch aborted due to fail_fast policy on prior failure",
                execution_time_ms=0,
            )

        async with sem:
            if aborted and batch_envelope.fail_fast:
                return TaskResultModel(
                    task_id=task_item.task_id,
                    target_agent="inference",
                    status="ABORTED",
                    error="Batch aborted due to fail_fast policy on prior failure",
                    execution_time_ms=0,
                )
            try:
                if not task_item.source_agent or task_item.source_agent == "friday":
                    task_item.source_agent = source
                res = await execute_task(task_item)
                if batch_envelope.fail_fast and res.status in ("ERROR", "DEGRADED"):
                    aborted = True
                return res
            except Exception as exc:
                if batch_envelope.fail_fast:
                    aborted = True
                return TaskResultModel(
                    task_id=task_item.task_id,
                    target_agent="inference",
                    status="ERROR",
                    error=scrub_credentials(str(exc)),
                    execution_time_ms=0,
                )

    results = await asyncio.gather(*[_run_single(t) for t in tasks])

    lat = int((time.time() - t0) * 1000)
    succeeded = sum(1 for r in results if r.status == "SUCCESS")
    failed = sum(1 for r in results if r.status in ("ERROR", "ABORTED"))
    degraded = sum(1 for r in results if r.status == "DEGRADED")

    summary = {
        "success_rate": round(succeeded / max(1, len(tasks)), 3),
        "avg_task_latency_ms": round(lat / max(1, len(tasks)), 1),
        "aborted_count": sum(1 for r in results if r.status == "ABORTED"),
    }

    return BatchTaskResultModel(
        batch_id=batch_envelope.batch_id,
        target_agent="inference",
        total_tasks=len(tasks),
        succeeded_tasks=succeeded,
        failed_tasks=failed,
        degraded_tasks=degraded,
        execution_time_ms=lat,
        results=results,
        summary=summary,
    )


@task_router.post("/v1/task/validate", response_model=TaskValidateResponse, status_code=status.HTTP_200_OK)
async def validate_task(req: TaskValidateRequest) -> TaskValidateResponse:
    """Pre-flight check for a task payload without executing model generation."""
    source = req.source_agent.lower().strip()
    raw_prompt = (
        req.payload.get("prompt")
        or req.payload.get("question")
        or req.payload.get("query")
        or req.payload.get("task")
        or str(req.payload)
    )
    guard = inspect_prompt(raw_prompt)
    est_tokens = max(50, len(raw_prompt) // 4)

    quota_info = agent_quota_manager.get_status(source)
    quota_available = (quota_info["tokens_used"] + est_tokens <= quota_info["tpm_limit"]) and (
        quota_info["requests_used"] + 1 <= quota_info["rpm_limit"]
    )

    is_valid = guard.allowed and (guard.risk_level in ("SAFE", "LOW")) and quota_available

    return TaskValidateResponse(
        valid=is_valid,
        source_agent=source,
        action=req.action,
        estimated_tokens=est_tokens,
        quota_available=quota_available,
        prompt_safe=guard.allowed,
        findings=list(guard.findings),
        risk_level=guard.risk_level,
        recommended_model="openai/gpt-oss-120b" if source in ("forge", "cortex") else "auto",
    )


@task_router.get("/v1/task/quotas/{agent_id}", status_code=status.HTTP_200_OK)
async def get_agent_quotas(agent_id: str) -> dict[str, Any]:
    """Retrieves current quota consumption, bucket capacity, and time to reset for an agent."""
    clean_id = agent_id.lower().strip()
    status_data = agent_quota_manager.get_status(clean_id)
    tokens_remaining = max(0, status_data["tpm_limit"] - status_data["tokens_used"])
    requests_remaining = max(0, status_data["rpm_limit"] - status_data["requests_used"])
    return {
        "agent_id": clean_id,
        "token_capacity": status_data["tpm_limit"],
        "tokens_used": status_data["tokens_used"],
        "tokens_remaining": tokens_remaining,
        "request_capacity": status_data["rpm_limit"],
        "requests_used": status_data["requests_used"],
        "requests_remaining": requests_remaining,
        "window_remaining_seconds": status_data["window_remaining_seconds"],
        "status": "HEALTHY" if tokens_remaining > 5000 else "CONGESTED",
    }


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
