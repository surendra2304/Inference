"""FastAPI API routes for Inference."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.agents.debate import AgentPanelUnavailable
from app.core.orchestrator import OrchestrationRequest, orchestrator
from app.core.security import require_inference_api_key
from app.utils.logger import logger

router = APIRouter()


#: Canonical execution modes. ``auto`` lets the router classify the question;
#: the other three pin the depth explicitly.
VALID_MODES: tuple[str, ...] = ("auto", "fast", "review", "debate")


class AskRequest(BaseModel):
    """Payload for submitting a question to Inference."""
    question: str = Field(description="The user or system inquiry to analyze and answer")
    mode: str = Field(default="auto", description="auto, fast, review, debate")

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        """Reject unknown modes instead of silently reclassifying them.

        ``DebateOrchestrator.classify_mode`` falls through to keyword-based
        classification for *any* unrecognised string, which means the mode is treated
        as ``auto``. A caller who mistypes ``"reviews"`` therefore receives a 200 and a
        plausible-looking answer produced by a different amount of work than they asked
        for — the difference between ``fast`` and ``debate`` is one model call versus a
        five-specialist panel plus synthesis, i.e. several times the latency and cost.
        Silence is the wrong default when the caller's intent is unambiguous and the
        consequence is a change in depth and spend, so this fails loudly and names the
        accepted values. Case and surrounding whitespace stay tolerated.
        """
        normalized = value.strip().lower()
        if normalized not in VALID_MODES:
            raise ValueError(
                f"Unknown mode {value!r}. Accepted values: {', '.join(VALID_MODES)}. "
                "Use 'auto' to let the router classify the question."
            )
        return normalized
    max_agents: int = Field(default=5, ge=1, le=10)
    require_evidence: bool = Field(default=True)
    max_budget: float | None = Field(default=None, description="Max budget in USD for this task")
    max_latency: float | None = Field(default=None, description="Max desired latency in seconds")
    context_data: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    """Structured response for the /ask endpoint."""
    task_id: str
    run_id: str
    answer: str
    #: Execution mode that ran: ``fast``, ``review`` or ``debate``.
    mode_used: str
    #: How the collaboration concluded (``consensus``, ``debate`` or ``fast``). Kept
    #: separate from ``mode_used``: the outcome of the panel is not the mode the client
    #: asked for, and reporting one as the other breaks callers that branch on the mode.
    deliberation_outcome: str = ""
    provider: str
    models_used: list[str]
    agents_used: list[str]
    confidence: float
    latency_seconds: float
    total_tokens: int
    unresolved_disagreements: list[str] = Field(default_factory=list)
    key_evidence: list[str] = Field(default_factory=list)
    # Self-healing / honest-degradation metadata
    degraded: bool = False
    degradation_reasons: list[str] = Field(default_factory=list)
    agent_coverage: dict[str, str] = Field(
        default_factory=dict,
        description="original_agent_id -> covering peer agent id, when a specialist had to be covered",
    )
    failed_agents: list[str] = Field(default_factory=list)


class DebateRequest(BaseModel):
    """Payload for explicitly requesting a multi-agent structured debate."""
    question: str = Field(description="The question or proposal to debate")
    max_agents: int = Field(default=5, ge=2, le=10)
    require_evidence: bool = Field(default=True)
    max_budget: float | None = Field(default=None, description="Max budget in USD for this debate")
    max_latency: float | None = Field(default=None, description="Max desired latency in seconds")
    context_data: dict[str, Any] = Field(default_factory=dict)


class DebateResponse(BaseModel):
    """Structured response for the /debate endpoint."""
    task_id: str
    run_id: str
    answer: str
    mode_used: str = "debate"
    agents_used: list[str]
    models_used: list[str]
    confidence: float
    unresolved_disagreements: list[str] = Field(default_factory=list)
    key_evidence: list[str] = Field(default_factory=list)
    total_tokens: int = 0
    latency_seconds: float = 0.0
    degraded: bool = False
    degradation_reasons: list[str] = Field(default_factory=list)
    agent_coverage: dict[str, str] = Field(default_factory=dict)
    failed_agents: list[str] = Field(default_factory=list)


@router.post("/ask", response_model=AskResponse, status_code=status.HTTP_200_OK, dependencies=[Depends(require_inference_api_key)])
async def ask_question(request: AskRequest) -> AskResponse:
    """Submit a question to the Inference orchestrator (auto-routes to Fast, Review, or Debate)."""
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty."
        )

    orch_request = OrchestrationRequest(
        question=request.question,
        mode=request.mode,
        max_agents=request.max_agents,
        require_evidence=request.require_evidence,
        max_budget=request.max_budget,
        max_latency=request.max_latency,
        context_data=request.context_data
    )

    try:
        result = await orchestrator.process_task(orch_request)
        return AskResponse(
            task_id=result.task_id,
            run_id=result.run_id,
            answer=result.answer,
            mode_used=result.mode_used,
            deliberation_outcome=getattr(result, "deliberation_outcome", ""),
            provider=result.provider_used,
            models_used=result.models_used,
            agents_used=result.agents_used,
            confidence=result.confidence,
            latency_seconds=result.total_latency_seconds,
            total_tokens=result.total_tokens,
            unresolved_disagreements=result.unresolved_disagreements,
            key_evidence=result.key_evidence,
            degraded=result.degraded,
            degradation_reasons=result.degradation_reasons,
            agent_coverage=result.agent_coverage,
            failed_agents=result.failed_agents,
        )
    except AgentPanelUnavailable as exc:
        # Every specialist (and every peer that tried to cover) went dark.
        # 503 Service Unavailable is the truthful status: the service could not
        # perform the requested inference, and we refuse to fabricate an answer.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task orchestration failed: {exc!s}"
        )


@router.post("/debate", response_model=DebateResponse, status_code=status.HTTP_200_OK, dependencies=[Depends(require_inference_api_key)])
async def trigger_debate(request: DebateRequest) -> DebateResponse:
    """Explicitly trigger the 6-Round Structured Multi-Agent Debate Engine."""
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty."
        )

    orch_request = OrchestrationRequest(
        question=request.question,
        mode="debate",
        max_agents=request.max_agents,
        require_evidence=request.require_evidence,
        max_budget=request.max_budget,
        max_latency=request.max_latency,
        context_data=request.context_data
    )

    try:
        result = await orchestrator.process_task(orch_request)
        return DebateResponse(
            task_id=result.task_id,
            run_id=result.run_id,
            answer=result.answer,
            mode_used="debate",
            agents_used=result.agents_used,
            models_used=result.models_used,
            confidence=result.confidence,
            unresolved_disagreements=result.unresolved_disagreements,
            key_evidence=result.key_evidence,
            total_tokens=result.total_tokens,
            latency_seconds=result.total_latency_seconds,
            degraded=result.degraded,
            degradation_reasons=result.degradation_reasons,
            agent_coverage=result.agent_coverage,
            failed_agents=result.failed_agents,
        )
    except AgentPanelUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Debate orchestration failed: {exc!s}"
        )


@router.get("/tasks/{task_id}", dependencies=[Depends(require_inference_api_key)])
async def get_task(task_id: str):
    """Retrieve details and state of a task by ID."""
    status_data = await orchestrator.get_task_status(task_id)
    if not status_data:
        # A bounded recent-task cache makes "never recorded" and "recorded, then dropped
        # from the window" different facts; answer with the one that is true.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=orchestrator.recent_task_cache_miss_detail(task_id),
        )
    return status_data


@router.get("/agents")
async def list_public_agents():
    """Retrieve list of all active specialist agents in Inference."""
    agents = orchestrator.registry.list_agents()
    return [
        {
            "id": a.id,
            "name": a.name,
            "role": a.role,
            "purpose": a.purpose,
            "provider": a.model_provider,
            "model": a.model_name,
            "strengths": a.strengths,
            "status": a.status
        }
        for a in agents
    ]


class ExperimentTriggerRequest(BaseModel):
    """Payload to trigger an experiment run via API."""
    experiment_type: str = Field(description="benchmark_suite, baseline_vs_debate, model_comparison")
    question: str | None = None
    target_benchmarks: list[str] | None = None
    providers_to_test: list[str] | None = None


@router.post("/experiments", status_code=status.HTTP_200_OK, dependencies=[Depends(require_inference_api_key)])
async def trigger_experiment(request: ExperimentTriggerRequest):
    """Trigger an automated benchmark or comparison experiment."""
    from app.experiments.harness import benchmark_harness
    benchmark_harness.memory = orchestrator.memory
    benchmark_harness.orchestrator.memory = orchestrator.memory

    try:
        if request.experiment_type == "benchmark_suite":
            record = await benchmark_harness.run_benchmark_suite(
                benchmark_ids=request.target_benchmarks
            )
        elif request.experiment_type == "baseline_vs_debate":
            q = request.question or "What are the trade-offs of microservices vs monoliths?"
            record = await benchmark_harness.run_baseline_vs_debate_comparison(question=q)
        elif request.experiment_type == "model_comparison":
            q = request.question or "Design a fault-tolerant caching layer with Redis and SQLite."
            record = await benchmark_harness.run_model_comparison_matrix(
                prompt=q,
                providers_to_test=request.providers_to_test
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown experiment_type: {request.experiment_type}"
            )
        return record.model_dump()
    except HTTPException:
        # A deliberate client error (an unsupported experiment_type is a 400) must reach
        # the caller as that 4xx. Catching it with the blanket handler below reported it
        # as "500 Experiment execution failed", telling the client the SERVER had broken
        # when in fact the client had asked for something that does not exist — an
        # unsupported experiment_type could not be distinguished from a real crash.
        raise
    except Exception as exc:
        logger.exception("Experiment %s failed", request.experiment_type)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Experiment execution failed: {type(exc).__name__}"
        ) from exc


@router.get("/experiments/{experiment_id}", dependencies=[Depends(require_inference_api_key)])
async def get_experiment(experiment_id: str):
    """Retrieve details and results of an experiment by ID."""
    record = await orchestrator.memory.get_experiment(experiment_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' not found."
        )
    return record.model_dump()
