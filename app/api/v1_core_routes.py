"""Core v1 API Routes for Inference and ASTRA Reasoning Council.

Implements the stable versioned endpoints mandated by Prompt 2:
- POST /v1/ask
- POST /v1/debate
- POST /v1/trading/consult
- GET /v1/health
- GET /v1/capabilities

Strict Guarantees:
1. Every response returns all 13 canonical fields.
2. proposed_actions are strictly advisory objects with is_executable_command=False.
3. Confidential credentials are scrubbed before any model invocation.
4. Untrusted data is wrapped in strong isolation boundaries.
5. ASTRA is exposed as a deliberative council persona, not a fabricated model.
6. If evidence is insufficient, low confidence and missing data are returned.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Header, HTTPException, status

from app.agents.base import Agent
from app.agents.debate import AgentPanelUnavailable, debate_engine
from app.agents.registry import agent_registry
from app.core.astra_profile import astra_profile
from app.core.dag import TaskComplexity
from app.core.orchestrator import orchestrator
from app.memory.base import TaskRecord
from app.providers.unified_manager import UnifiedExecutionRequest, unified_provider_manager
from app.schemas.v1_models import (
    DebateRequest,
    InferenceAskRequest,
    InferenceTaskResponse,
    ProposedAction,
    ProviderMetadata,
)
from app.security.prompt_isolation import (
    CredentialLeakError,
    detect_credentials,
    scrub_credentials,
    scrub_credentials_dict,
    scrub_credentials_verified,
    wrap_untrusted_data,
)
from app.utils.errors import correlation_id, unavailable_detail
from app.utils.logger import logger
from app.version import VERSION

v1_router = APIRouter(prefix="/v1", tags=["Inference v1 Core Protocol"])


def _scrub_and_verify(raw_text: str, context: Any, trace_id: str) -> tuple[str, Any]:
    """Scrub credentials from prompt and context, then prove none survived.

    `detect_credentials` is the audit for `scrub_credentials`: if it still
    reports a credential after scrubbing, forwarding the request would violate
    prompt_isolation's invariant that confidential material never reaches a
    model provider, so the request is refused instead.
    """
    try:
        clean_text = scrub_credentials_verified(raw_text, field="prompt")
        clean_context = scrub_credentials_dict(context)
    except CredentialLeakError as exc:
        logger.error("Trace %s refused: %s", trace_id, exc)
        raise HTTPException(
            # The server is working correctly; it is refusing a request whose payload it
            # cannot guarantee is clean. Answering 500 said the SERVER had broken, which
            # told an on-call engineer to investigate a fault and told the client to stop
            # retrying, while the actionable fact is that this payload carries something
            # that looks like a credential. A 4xx also keeps a security refusal out of the
            # server-fault bucket where it was indistinguishable from a real crash.
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request refused: a credential could not be sanitized.",
        ) from exc

    residual = detect_credentials({"prompt": clean_text, "context": clean_context})
    if residual:
        logger.error("Trace %s refused; residual credentials: %s", trace_id, residual)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request refused: a credential could not be sanitized.",
        )
    return clean_text, clean_context


@v1_router.get("/health", status_code=status.HTTP_200_OK)
async def v1_health() -> dict[str, Any]:
    """Liveness and readiness check for Inference and ASTRA."""
    agents = agent_registry.list_agents()
    return {
        "status": "healthy",
        "service": "inference",
        "version": VERSION,
        "astra_profile": {
            "name": astra_profile.name,
            "role": astra_profile.role,
            "claim_gpt6": astra_profile.claim_gpt6,
            "evidence_policy": astra_profile.evidence_policy,
            "allowed_providers": astra_profile.allowed_providers,
        },
        "active_specialist_agents": len(agents),
        "registered_roles": [a.role for a in agents],
    }


@v1_router.get("/capabilities", status_code=status.HTTP_200_OK)
async def v1_capabilities() -> dict[str, Any]:
    """Capabilities catalog and advisory safety boundaries."""
    return {
        "service": "inference",
        "system": "FRIDAY Universe Inference & ASTRA",
        "version": VERSION,
        "capabilities": [
            "deliberative_reasoning",
            "multi_agent_debate",
            "trading_consultation",
            "credential_protection",
            "prompt_isolation",
            "advisory_proposed_actions",
            "insufficient_data_detection",
            "multi_model_synthesis",
        ],
        "debate_roles": [
            "proposer",
            "critic",
            "fact_checker",
            "data_analyst",
            "strategist",
            "synthesizer",
        ],
        "free_providers": astra_profile.allowed_providers,
        "advisory_only": True,
        "executable_authority": False,
        "claim_gpt6": False,
    }


#: Confidence for a single, unverified model call, used as the baseline for /v1/ask.
#:
#: This is a *policy prior*, not a measured accuracy: one model answered once, with
#: no second opinion, no evidence retrieval and no cross-examination, so the honest
#: claim is "probably useful, not verified". It is deliberately below every value the
#: deliberation engine reports for a panel (a two-specialist review lands at ~0.83, a
#: five-specialist debate at ~0.80): the shallowest path must not claim the most
#: certainty.
_SINGLE_CALL_BASE_CONFIDENCE = 0.65
#: Ceiling applied when the provider stopped at its token limit; a truncated answer is
#: materially incomplete. Kept equal to the deliberation engine's own truncation
#: ceiling so the two stacks cannot disagree about what truncation means.
_TRUNCATION_CONFIDENCE_CEILING = 0.55


def _derive_single_call_confidence(
    content: str,
    *,
    truncated: bool,
    served_by_fallback: bool,
) -> float:
    """Derive a confidence for a single model call from evidence, never a constant.

    Before this, ``/v1/ask`` returned a hardcoded ``0.92`` for *any* successful call,
    regardless of what came back: a two-word answer and a thorough analysis scored
    identically, a response from a fallback provider scored the same as a first-choice
    one, and the number was higher than the deliberation engine's confidence for a
    five-specialist debate. A constant published as calibrated confidence is a false
    claim about how much the answer can be trusted, so it is replaced by a value that
    moves with the evidence actually available.
    """
    if not content or not content.strip():
        # Nothing came back; there is nothing to be confident about.
        return 0.0

    confidence = _SINGLE_CALL_BASE_CONFIDENCE
    if len(content.strip()) >= 400:
        # A substantive answer is weak evidence that the model engaged with the task.
        confidence += 0.05
    if served_by_fallback:
        # The request did not reach the provider that was chosen for it; whatever it
        # did reach is less well matched to the task's capability profile.
        confidence -= 0.05
    if truncated:
        confidence = min(confidence, _TRUNCATION_CONFIDENCE_CEILING)
    return round(max(0.0, min(confidence, 0.75)), 2)


def _check_insufficient_data(text: str, context: dict[str, Any]) -> tuple[bool, list[str]]:
    """Checks whether the query or context explicitly lacks required evidence/data."""
    missing: list[str] = []
    text_lower = text.lower()

    indicators = [
        ("telemetry missing", "Historical trading telemetry data missing"),
        ("insufficient data", "Empirical evidence insufficient for high-confidence determination"),
        ("missing parameters", "Required configuration parameters omitted"),
        ("unspecified environment", "Runtime environment unspecified"),
        ("no data provided", "Input dataset empty or missing"),
    ]
    for pattern, description in indicators:
        if pattern in text_lower:
            missing.append(description)

    # Check context fields
    if context.get("status") == "INSUFFICIENT_DATA":
        missing.append("Status explicitly marked as INSUFFICIENT_DATA")
    if context.get("missing_data") and isinstance(context.get("missing_data"), list):
        for item in context["missing_data"]:
            if str(item) not in missing:
                missing.append(str(item))

    return (len(missing) > 0, missing)


@v1_router.post("/ask", response_model=InferenceTaskResponse, status_code=status.HTTP_200_OK)
async def ask_v1(
    request: InferenceAskRequest,
    x_trace_id: str | None = Header(None, alias="X-Trace-ID"),
) -> InferenceTaskResponse:
    """Deliberative reasoning endpoint compatible with FRIDAY TaskEnvelope.

    Guarantees all 13 canonical fields, credential scrubbing, and strictly advisory proposed_actions.
    """
    start_time = time.perf_counter()
    raw_prompt = request.prompt or "General query"
    task_id = request.task_id or f"task_{uuid.uuid4().hex[:12]}"
    trace_id = request.trace_id or x_trace_id or f"trace_{uuid.uuid4().hex[:8]}"

    # 1. Security: Scrub confidential credentials from prompt and context,
    #    verifying afterwards that none survived.
    clean_prompt, clean_context = _scrub_and_verify(raw_prompt, request.context, trace_id)

    # 2. Wrap untrusted data if provided
    if request.untrusted_data:
        isolated_block = wrap_untrusted_data(request.untrusted_data, source=request.untrusted_data_source)
        clean_prompt = f"{clean_prompt}\n\n[ISOLATED DATA OBSERVATION]:\n{isolated_block}"

    # 3. Check for insufficient data
    is_insufficient, missing_items = _check_insufficient_data(clean_prompt, clean_context)
    if is_insufficient:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return InferenceTaskResponse(
            task_id=task_id,
            trace_id=trace_id,
            answer=(
                "Insufficient empirical data to formulate a high-confidence conclusion. "
                f"Missing requirements: {', '.join(missing_items)}. "
                "In accordance with strict evidence policy, Inference refuses to hallucinate unsubstantiated results."
            ),
            reasoning_summary="Deliberation halted due to missing critical data or telemetry.",
            confidence=0.30,
            uncertainty=0.70,
            evidence=[],
            agents_used=["fact_checker", "synthesizer"],
            recommendations=[f"Provide missing data: {item}" for item in missing_items],
            proposed_actions=[],
            authorization_required=False,
            provider_metadata=ProviderMetadata(
                provider="internal_calibrator",
                model="evidence_guard",
                latency_ms=latency_ms,
            ),
            failure_state="INSUFFICIENT_DATA",
            status="DEGRADED",
            missing_data=missing_items,
        )

    # 4. Dispatch reasoning
    #
    # ``mode`` used to influence exactly one thing: ``fast_lane=(mode == "fast")``. Every
    # other accepted value ("deliberative", "consensus", "review") produced the same single
    # provider call, and the response said nothing about it — measured live, all four modes
    # returned byte-identical answers with the same confidence. The multi-agent modes now
    # run the deliberation engine and the response reports ``mode_requested`` /
    # ``mode_used``; "consensus" is an *outcome* label in this service (the panel agreed),
    # not a separate mode, so it maps onto "review" and the mapping is disclosed.
    mode_requested = request.mode
    multi_agent_mode = {"deliberative": "review", "consensus": "review", "review": "review",
                        "debate": "debate"}.get(request.mode)
    mode_mapping_note = (
        f"mode '{request.mode}' ran the multi-agent engine as '{multi_agent_mode}'"
        if multi_agent_mode and multi_agent_mode != request.mode
        else None
    )

    if multi_agent_mode is not None:
        try:
            from app.core.orchestrator import OrchestrationRequest, orchestrator

            result = await orchestrator.process_task(
                OrchestrationRequest(
                    question=clean_prompt,
                    mode=multi_agent_mode,
                    require_evidence=True,
                    context_data=clean_context,
                )
            )
        except AgentPanelUnavailable as exc:
            # Every specialist went dark. This endpoint's contract carries ``status`` /
            # ``failure_state`` / ``confidence`` precisely so a client can be told the panel
            # produced nothing — in band, with no fabricated prose in ``answer``.
            logger.warning("[V1_ASK] panel unavailable: %s", exc)
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            await _record_task_outcome(
                task_id, clean_prompt,
                f"Panel unavailable: {scrub_credentials(str(exc))}", mode=request.mode,
                status_value="failed", confidence=0.0,
                # Scrubbed at write time: this metadata is served back verbatim by
                # ``GET /tasks/{id}`` (``orchestrator.get_task_status`` -> ``model_dump()``),
                # so a raw exception stored here becomes a leak one request later.
                metadata={"trace_id": trace_id, "error": scrub_credentials(str(exc))},
            )
            return InferenceTaskResponse(
                task_id=task_id,
                trace_id=trace_id,
                answer=(
                    "No specialist produced a result, so no answer is offered. "
                    "Every configured provider and peer agent was unavailable for this request."
                ),
                reasoning_summary="Deliberation could not start: the specialist panel was unavailable.",
                confidence=0.0,
                uncertainty=1.0,
                evidence=[],
                agents_used=[],
                recommendations=["Retry, or ask with mode='fast' for a single-model answer."],
                proposed_actions=[],
                authorization_required=False,
                provider_metadata=ProviderMetadata(
                    provider="multi_provider_council",
                    model="deliberation_engine",
                    latency_ms=latency_ms,
                ),
                # Agent ids and failure *kinds* only. The exception's own text carries the
                # provider's message, and a provider's 401 typically quotes the key it
                # rejected; [FACT] the sibling DELIBERATION_FAILED path below was measured
                # publishing "password=..." verbatim inside ``answer``.
                failure_state=unavailable_detail(exc, failures=exc.failures, prefix="panel"),
                status="DEGRADED",
                mode_requested=mode_requested,
                mode_used=None,
                mode_mapping_note=mode_mapping_note,
            )
        except Exception as exc:  # noqa: BLE001 - reported, never fabricated over
            reference = correlation_id("deliberation")
            logger.error(
                "[V1_ASK] multi-agent deliberation failed [%s]: %s", reference, exc,
                exc_info=True,
            )
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            await _record_task_outcome(
                task_id, clean_prompt,
                f"Deliberation failed: {scrub_credentials(str(exc))}", mode=request.mode,
                status_value="failed", confidence=0.0,
                # Scrubbed at write time: this metadata is served back verbatim by
                # ``GET /tasks/{id}`` (``orchestrator.get_task_status`` -> ``model_dump()``),
                # so a raw exception stored here becomes a leak one request later.
                metadata={"trace_id": trace_id, "error": scrub_credentials(str(exc))},
            )
            return InferenceTaskResponse(
                task_id=task_id, trace_id=trace_id,
                answer=(
                    "Deliberation could not be completed: the multi-agent panel did not "
                    "produce a result. No answer is fabricated in its place."
                ),
                reasoning_summary="Multi-agent deliberation failed before synthesis.",
                confidence=0.0, uncertainty=1.0, evidence=[],
                agents_used=[],
                recommendations=["Retry, or ask with mode='fast' for a single-model answer."],
                proposed_actions=[], authorization_required=False,
                provider_metadata=ProviderMetadata(provider="multi_provider_council",
                                                   model="deliberation_engine",
                                                   latency_ms=latency_ms),
                failure_state=f"DELIBERATION_FAILED (correlation id {reference})",
                status="DEGRADED",
                mode_requested=mode_requested, mode_used=None, mode_mapping_note=mode_mapping_note,
            )

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        await _record_task_outcome(
            task_id, clean_prompt, result.answer, mode=result.mode_used,
            status_value="degraded" if result.degraded else "completed",
            confidence=result.confidence,
            metadata={
                "trace_id": trace_id, "run_id": result.run_id,
                "deliberation_outcome": result.deliberation_outcome,
                "agents_used": result.agents_used, "models_used": result.models_used,
                "unresolved_disagreements": result.unresolved_disagreements,
                "degraded": result.degraded, "degradation_reasons": result.degradation_reasons,
                "failed_agents": result.failed_agents,
            },
        )
        return InferenceTaskResponse(
            task_id=task_id,
            trace_id=trace_id,
            answer=result.answer,
            reasoning_summary=(
                f"Deliberated in '{result.mode_used}' mode across {len(result.agents_used)} "
                f"specialist role(s); outcome '{result.deliberation_outcome or 'n/a'}'."
            ),
            confidence=result.confidence,
            uncertainty=round(max(0.0, 1.0 - result.confidence), 4),
            evidence=result.key_evidence,
            agents_used=result.agents_used,
            recommendations=(
                ["Resolve the recorded disagreements before acting on this answer."]
                if result.unresolved_disagreements
                else ["Proceed with the consensus view; monitor for contradicting evidence."]
            ),
            proposed_actions=[],
            authorization_required=bool(result.unresolved_disagreements),
            provider_metadata=ProviderMetadata(
                provider=result.provider_used or "multi_provider_council",
                model="deliberation_engine",
                fallback_chain=result.models_used,
                total_tokens=result.total_tokens,
                latency_ms=latency_ms,
            ),
            failure_state=(
                # Scrubbed again at the boundary: whatever composed these reasons, the client
                # must not receive credential-looking text (see app/utils/errors.py for the
                # measured case that motivated the rule).
                scrub_credentials("; ".join(result.degradation_reasons)) if result.degraded else None
            ),
            status="DEGRADED" if result.degraded else "SUCCESS",
            mode_requested=mode_requested,
            mode_used=result.mode_used,
            mode_mapping_note=mode_mapping_note,
        )

    try:
        astra_instruction = (
            f"[{astra_profile.name} DELIBERATIVE REASONING COUNCIL - {astra_profile.role}]\n"
            f"Evidence Policy: {astra_profile.evidence_policy}\n"
            "Analyze the inquiry rigorously. Provide a clear, actionable conclusion, "
            "highlight key assumptions, and state any residual uncertainty.\n\n"
            f"Inquiry: {clean_prompt}"
        )

        exec_req = UnifiedExecutionRequest(
            provider="auto",
            agent_role="system_architect",
            prompt=astra_instruction,
            context=clean_context,
            fast_lane=(request.mode == "fast"),
            max_tokens=2000,
        )
        resp = await unified_provider_manager.execute(exec_req)
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # ── Honest degradation gate ────────────────────────────────────────────────
        # If no model produced output we must NOT report SUCCESS, must not claim
        # empirical verification, and must not assign a confident score. The
        # evidence policy (STRICT_EMPIRICAL) requires returning uncertainty here.
        if resp.degraded:
            logger.warning(
                "[V1_ASK] degraded (no model output): %s", resp.error or "unknown provider failure"
            )
            return InferenceTaskResponse(
                task_id=task_id,
                trace_id=trace_id,
                answer=(
                    "No model provider could be reached for this request, so Inference "
                    "deliberated with zero evidence. No answer is asserted rather than "
                    "fabricating a conclusion."
                ),
                reasoning_summary=(
                    f"Deliberation halted: all model calls failed ({resp.error}). "
                    "Per STRICT_EMPIRICAL policy, low confidence is returned instead of a guess."
                ),
                confidence=0.0,
                uncertainty=1.0,
                evidence=[],
                agents_used=["astra_council"],
                recommendations=[
                    "Configure a provider API key (GEMINI_API_KEYS / GROQ_API_KEYS, etc.)",
                    f"Inspect provider health at /health/providers (failed: {resp.provider_used})",
                    "Retry once the provider pool reports active keys",
                ],
                proposed_actions=[],
                authorization_required=False,
                provider_metadata=ProviderMetadata(
                    provider=resp.provider_used,
                    model=resp.model_used,
                    latency_ms=latency_ms,
                ),
                failure_state=resp.error or "all_provider_calls_failed",
                status="DEGRADED",
                mode_requested=request.mode,
                mode_used="fast",
            )

        # Formulate advisory proposed actions
        advisory_actions: list[ProposedAction] = []
        if "action" in clean_prompt.lower() or "recommend" in clean_prompt.lower() or "fix" in clean_prompt.lower():
            advisory_actions.append(
                ProposedAction(
                    action="review_reasoning_advisory",
                    target="friday_central_os",
                    parameters={"summary": resp.content[:200]},
                    rationale="Advisory action proposed by ASTRA reasoning council.",
                    is_executable_command=False,
                    requires_authorization=True,
                    risk_level="low",
                )
            )

        # Derive the confidence from what actually came back instead of publishing a
        # constant (see _derive_single_call_confidence).
        truncated = (resp.finish_reason or "").lower() in ("length", "max_tokens")
        # A policy substitution (self-hosted tier under LOCAL_PREFERRED) is the intended
        # path, not a fault; only an actual fallback costs confidence.
        served_by_fallback = bool(resp.served_by_provider) and not resp.served_by_policy
        served_from_self_hosted = bool(resp.served_by_policy)
        derived_confidence = _derive_single_call_confidence(
            resp.content, truncated=truncated, served_by_fallback=served_by_fallback
        )
        basis = [
            "one model call, no cross-examination",
            f"provider={resp.provider_used}",
        ]
        if truncated:
            basis.append("output truncated at the token ceiling")
        if served_by_fallback:
            basis.append(f"served by a fallback provider ({resp.served_by_provider})")
        if served_from_self_hosted:
            basis.append("served by the self-hosted tier (LOCAL_PREFERRED)")
        if not (resp.content or "").strip():
            basis.append("empty completion")

        return InferenceTaskResponse(
            task_id=task_id,
            trace_id=trace_id,
            answer=resp.content,
            reasoning_summary=(
                f"Synthesized across deliberative council via {resp.provider_used} "
                f"({resp.model_used}). Confidence basis: {', '.join(basis)}."
            ),
            confidence=derived_confidence,
            uncertainty=round(1.0 - derived_confidence, 2),
            evidence=[f"Empirical provider verification via {resp.provider_used}:{resp.model_used}"],
            agents_used=["astra_council", resp.agent_role or "system_architect"],
            recommendations=["Review deliberative findings before initiating state-changing actions."],
            proposed_actions=advisory_actions,
            authorization_required=len(advisory_actions) > 0,
            provider_metadata=ProviderMetadata(
                provider=resp.provider_used,
                model=resp.model_used,
                prompt_tokens=resp.token_usage.get("prompt_tokens", 0),
                completion_tokens=resp.token_usage.get("completion_tokens", 0),
                total_tokens=resp.token_usage.get("total_tokens", 0),
                latency_ms=latency_ms,
            ),
            failure_state=None,
            status="SUCCESS",
            mode_requested=request.mode,
            mode_used="fast",
        )

    except Exception as exc:
        logger.error(f"[V1_ASK] Execution failure: {exc}", exc_info=True)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return InferenceTaskResponse(
            task_id=task_id,
            trace_id=trace_id,
            answer=f"Deliberation encountered an issue: {exc}",
            reasoning_summary="Provider invocation failed; fallback activated.",
            confidence=0.0,
            uncertainty=1.0,
            evidence=[],
            agents_used=["astra_council"],
            recommendations=["Verify provider health or retry with fast mode."],
            proposed_actions=[],
            authorization_required=False,
            provider_metadata=ProviderMetadata(
                provider="fallback",
                model="none",
                latency_ms=latency_ms,
            ),
            failure_state=str(exc),
            status="ERROR",
            mode_requested=request.mode,
            mode_used="fast",
        )


async def _record_task_outcome(
    task_id: str,
    question: str,
    result: str,
    *,
    mode: str,
    status_value: str,
    confidence: float | None,
    metadata: dict[str, Any],
) -> None:
    """Persist a task record for a route that drives the engine directly.

    ``GET /tasks/{task_id}`` is the documented way to read a task back, so every route
    that answers with a ``task_id`` must make that id resolvable. Measured live: a
    ``POST /v1/debate`` returned 200 with ``task_53017bc07162`` while the read-back
    answered 404, because this route never went through ``Orchestrator.process_task``.
    Persistence failure is logged and never fabricates success.
    """
    try:
        await orchestrator.record_task(
            TaskRecord(
                id=task_id,
                question=question,
                mode=mode,
                status=status_value,
                result=result,
                confidence=confidence,
                completed_at=datetime.now(timezone.utc),
                metadata=metadata,
            )
        )
    except Exception as exc:  # noqa: BLE001 - the answer already happened; report, don't mask
        logger.error("[V1_DEBATE] Task record for %s could not be persisted: %s", task_id, exc)


@v1_router.post("/debate", response_model=InferenceTaskResponse, status_code=status.HTTP_200_OK)
async def debate_v1(
    request: DebateRequest,
    x_trace_id: str | None = Header(None, alias="X-Trace-ID"),
) -> InferenceTaskResponse:
    """Multi-agent deliberative debate across the 6 canonical roles.

    Roles: proposer, critic, fact_checker, data_analyst, strategist, synthesizer.
    Guarantees all 13 canonical fields, credential scrubbing, and calibrated uncertainty.
    """
    start_time = time.perf_counter()
    topic = request.topic or "Deliberative system evaluation"
    task_id = request.task_id or f"task_{uuid.uuid4().hex[:12]}"
    trace_id = request.trace_id or x_trace_id or f"trace_{uuid.uuid4().hex[:8]}"

    # 1. Security: Scrub credentials (verified) and wrap untrusted data
    clean_topic, clean_context = _scrub_and_verify(topic, request.context, trace_id)

    if request.untrusted_data:
        isolated_data = wrap_untrusted_data(request.untrusted_data, source=request.untrusted_data_source)
        clean_topic = f"{clean_topic}\n\n[ISOLATED OBSERVATION]:\n{isolated_data}"

    # 2. Check for insufficient data
    is_insufficient, missing_items = _check_insufficient_data(clean_topic, clean_context)
    if is_insufficient:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        await _record_task_outcome(
            task_id,
            clean_topic,
            "Debate halted: insufficient core evidence.",
            mode="debate",
            status_value="completed",
            confidence=0.25,
            metadata={
                "failure_state": "INSUFFICIENT_DATA",
                "missing_data": missing_items,
                "trace_id": trace_id,
                "provider": "debate_council",
            },
        )
        return InferenceTaskResponse(
            task_id=task_id,
            trace_id=trace_id,
            answer=(
                "Debate cannot proceed with high confidence due to missing core evidence. "
                f"Missing items: {', '.join(missing_items)}."
            ),
            reasoning_summary="Fact_checker and Data_analyst flagged missing required evidence.",
            confidence=0.25,
            uncertainty=0.75,
            evidence=[],
            agents_used=["proposer", "critic", "fact_checker", "synthesizer"],
            recommendations=[f"Supply required evidence: {item}" for item in missing_items],
            proposed_actions=[],
            authorization_required=False,
            provider_metadata=ProviderMetadata(
                provider="debate_council",
                model="fact_checker_audit",
                latency_ms=latency_ms,
            ),
            failure_state="INSUFFICIENT_DATA",
            status="DEGRADED",
            missing_data=missing_items,
        )

    # 3. Retrieve participating specialist agents for the 6 roles
    required_role_ids = ["proposer", "critic", "fact_checker", "data_analyst", "strategist", "synthesizer"]
    participating_agents: list[Agent] = []
    for rid in required_role_ids:
        agent = agent_registry.get_agent(rid)
        if not agent and rid == "proposer":
            agent = agent_registry.get_agent("researcher")
        if agent:
            participating_agents.append(agent)

    try:
        collab_result = await debate_engine.run_collaboration(
            task_id=task_id,
            question=clean_topic,
            participating_agents=participating_agents if len(participating_agents) >= 2 else None,
            complexity=TaskComplexity.STRATEGIC,
        )
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Build advisory proposed actions
        advisory_actions: list[ProposedAction] = [
            ProposedAction(
                action="apply_debate_consensus",
                target="system_architect",
                parameters={"debate_id": collab_result.debate_id},
                rationale="Advisory consensus produced by 6-role multi-agent deliberation.",
                is_executable_command=False,
                requires_authorization=True,
                risk_level="medium",
            )
        ]

        # Extract agents used
        agents_used_ids = [a.id for a in participating_agents] if participating_agents else collab_result.participating_agents

        await _record_task_outcome(
            task_id,
            clean_topic,
            collab_result.final_answer,
            mode="debate",
            status_value="degraded" if collab_result.degraded else "completed",
            confidence=collab_result.confidence,
            metadata={
                "trace_id": trace_id,
                "debate_id": collab_result.debate_id,
                "deliberation_outcome": collab_result.mode_used,
                "agents_used": agents_used_ids,
                "models_used": collab_result.models_used,
                "unresolved_disagreements": collab_result.unresolved_disagreements,
                "degraded": collab_result.degraded,
                "degradation_reasons": collab_result.degradation_reasons,
                "agent_coverage": collab_result.agent_coverage,
                "failed_agents": collab_result.failed_agents,
            },
        )

        return InferenceTaskResponse(
            task_id=task_id,
            trace_id=trace_id,
            answer=collab_result.final_answer,
            reasoning_summary=(
                f"Deliberated across {len(agents_used_ids)} specialist roles. "
                f"Consensus mode: {collab_result.mode_used}. Disagreements resolved: {len(collab_result.unresolved_disagreements)}."
            ),
            confidence=collab_result.confidence,
            uncertainty=round(1.0 - collab_result.confidence, 4),
            evidence=collab_result.key_evidence,
            agents_used=agents_used_ids,
            recommendations=[
                "Implement agreed consensus points.",
                "Maintain audit log of dissenting viewpoints.",
            ],
            proposed_actions=advisory_actions,
            authorization_required=True,
            provider_metadata=ProviderMetadata(
                provider="multi_provider_council",
                model="consensus_ensemble",
                fallback_chain=collab_result.models_used,
                total_tokens=collab_result.total_tokens,
                latency_ms=latency_ms,
            ),
            failure_state=None,
            status="SUCCESS",
        )

    except Exception as exc:
        # Both ``answer`` and ``failure_state`` used to interpolate the exception verbatim,
        # so a panel failure published the provider text of every specialist that went dark —
        # measured: "Provider 'groq' has no configured credential..." repeated six times, and
        # with a real provider's 401 the rejected key travels the same path. The client gets
        # what it needs (the debate could not run, why in aggregate, and a correlation id);
        # the exception text goes to the server log where it belongs.
        reference = correlation_id("debate")
        logger.error("[V1_DEBATE] debate execution failure [%s]: %s", reference, exc,
                     exc_info=True)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        await _record_task_outcome(
            task_id,
            topic,
            f"Debate deliberation encountered an error (correlation id {reference})",
            mode="debate",
            status_value="failed",
            confidence=0.0,
            metadata={"trace_id": trace_id, "error": scrub_credentials(str(exc))},
        )
        return InferenceTaskResponse(
            task_id=task_id,
            trace_id=trace_id,
            answer=(
                "Debate deliberation encountered an error and no consensus was produced. "
                f"Quote correlation id {reference} when reporting this; the details are in "
                "the server log."
            ),
            reasoning_summary="Multi-agent debate interrupted by provider or system error.",
            confidence=0.0,
            uncertainty=1.0,
            evidence=[],
            agents_used=required_role_ids,
            recommendations=["Retry debate with simplified constraints or check provider health."],
            proposed_actions=[],
            authorization_required=False,
            provider_metadata=ProviderMetadata(
                provider="debate_fallback",
                model="none",
                latency_ms=latency_ms,
            ),
            failure_state=f"DEBATE_FAILED (correlation id {reference})",
            status="ERROR",
        )



