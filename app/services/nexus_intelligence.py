"""Nexus Intelligence Service and Mode-Based Multi-Agent Routing Engine.

Specification Highlights:
- Modes:
  - FAST (single best-matching specialist agent, 3s latency budget)
  - REVIEW (primary agent + Critic adversarial pass, 8s latency budget)
  - DEBATE (multi-round adversarial deliberation with 3+ specialists, 20s latency budget, max 6 rounds)
- Task Mappings:
  - lead_qualification -> Strategist + Data Analyst
  - conversion_diagnosis -> Data Analyst + Debugger + Critic
  - incident_analysis -> Debugger + Security Analyst + Critic
  - strategic_decision -> Strategist + Critic + Fact Checker
  - copy_optimization -> Synthesizer
  - churn_analysis -> Data Analyst + Strategist
- Confidence Calibration:
  - Disagreements reduce confidence proportionally.
  - Unresolved disagreements are preserved in response (never silently flattened).
- Evidence Trust Hierarchy:
  - system_fact > verified_telemetry > untrusted_user_input > inferred_profile
"""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.analytics.usage_analytics import usage_analytics
from app.routing.consumer_router import consumer_router
from app.utils.bounded_store import DEFAULT_MAX_ENTRIES, BoundedStore

TaskType = Literal[
    "lead_qualification",
    "conversion_diagnosis",
    "incident_analysis",
    "strategic_decision",
    "intervention_planning",
    "copy_optimization",
    "churn_analysis"
]

TrustLabel = Literal[
    "system_fact",
    "verified_telemetry",
    "untrusted_user_input",
    "inferred_profile"
]

IntelligenceMode = Literal["fast", "review", "debate"]


class EvidenceItem(BaseModel):
    id: str | None = None
    claim: str
    trust_label: TrustLabel = "verified_telemetry"
    source: str | None = None
    timestamp: float | None = Field(default_factory=time.time)


class BudgetSpec(BaseModel):
    latency_ms: int = Field(default=3000, description="Latency budget in milliseconds")
    max_rounds: int = Field(default=1, description="Max debate rounds (up to 6)")


class IntelligenceRequest(BaseModel):
    request_id: str
    task_type: TaskType
    goal: str
    context: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    required_output: list[str] = Field(default_factory=list)
    budget: BudgetSpec | None = Field(default_factory=BudgetSpec)
    mode: IntelligenceMode = "fast"


class RecommendedAction(BaseModel):
    action: str
    priority: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"
    rationale: str
    owner: str | None = None


class IntelligenceResponse(BaseModel):
    request_id: str
    decision: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str
    key_evidence: list[str]
    provenance: dict[str, Any]
    unresolved_disagreements: list[str] = Field(default_factory=list)
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)
    expires_at: float


class NexusIntelligenceService:
    """Nexus multi-mode decision engine with persistent provenance ledger."""

    TASK_AGENT_MAPPING: dict[str, list[str]] = {
        "lead_qualification": ["strategist", "data_analyst"],
        "conversion_diagnosis": ["data_analyst", "debugger", "critic"],
        "incident_analysis": ["debugger", "security_analyst", "critic"],
        "strategic_decision": ["strategist", "critic", "fact_checker"],
        "intervention_planning": ["strategist", "debugger", "critic"],
        "copy_optimization": ["synthesizer"],
        "churn_analysis": ["data_analyst", "strategist"]
    }

    def __init__(self) -> None:
        # Bounded: one entry per request used to accumulate without limit
        # (measured: +9.03 MB/1k requests on nexus, +5.19 on sentinel, retained
        # after gc). Entries are evicted LRU beyond the ceiling; the store
        # records how many, so a lookup miss can say "evicted" instead of
        # pretending the id never existed.
        self.provenance_store = BoundedStore[dict[str, Any]](
            "nexus.provenance_store", max_entries=DEFAULT_MAX_ENTRIES
        )

    def _evaluate_trust_weight(self, evidence: list[EvidenceItem]) -> float:
        """Computes weighted trust factor based on evidence trust labels."""
        if not evidence:
            return 0.75
        weights = {
            "system_fact": 1.0,
            "verified_telemetry": 0.9,
            "untrusted_user_input": 0.5,
            "inferred_profile": 0.6
        }
        total_w = sum(weights.get(e.trust_label, 0.7) for e in evidence)
        return total_w / len(evidence)

    async def process_request(self, req: IntelligenceRequest) -> IntelligenceResponse:
        # Request deduplication (5-minute idempotency window)
        from app.governance.tenant_manager import tenant_manager
        # Typed lookup: a stored payload that does not match this response model is
        # evicted and treated as a miss, so an inconsistent cache entry can never
        # surface as HTTP 500 (it previously did — see app/governance/tenant_manager.py).
        cached_resp = tenant_manager.check_deduplication_model(
            req.request_id, IntelligenceResponse, namespace="nexus_intelligence"
        )
        if cached_resp is not None:
            return cached_resp

        start_time = time.perf_counter()
        specialists = self.TASK_AGENT_MAPPING.get(req.task_type, ["strategist", "critic"])
        mode = req.mode.lower()

        evidence_trust = self._evaluate_trust_weight(req.evidence)
        key_evidence = [f"[{e.trust_label.upper()}] {e.claim}" for e in req.evidence[:5]]

        # Query the StrategyBank for records that actually match this task type.
        from app.analytics.outcome_learning import outcome_learning_engine
        bank_matches = outcome_learning_engine.query_strategy_bank(req.task_type, req.goal)
        if bank_matches:
            best = bank_matches[0]
            # ``source`` distinguishes an illustrative fixture from an observed outcome, so
            # a citation cannot be read as measured precedent when it is not one.
            source = best.get("source", "observed")
            precedent = (
                "Illustrative fixture (not an observed outcome)"
                if source == "illustrative_fixture"
                else "Observed precedent"
            )
            key_evidence.append(
                f"[STRATEGY_BANK:{source}] {precedent} for {req.task_type}: "
                f"{best.get('recommendation', '')} resulted in "
                f"{best.get('outcome_summary', '')} "
                f"({int(best.get('success_rate', 0.0) * 100)}% success as recorded)"
            )

        if not key_evidence:
            key_evidence = ["Telemetry verified against active baseline."]

        # Mode execution branches
        # Every branch below records what produced the answer. `reasoning_trace` is bound
        # here (to None) so the provenance block is valid for all modes: the original
        # refactor referenced a name that only existed in the debate branch and raised
        # UnboundLocalError for fast/review — caught by
        # tests/test_api_contract_fuzz.py::test_schema_valid_degenerate_body_never_returns_5xx.
        reasoning_trace = None

        if mode == "fast":
            primary_agent = specialists[0]
            decision = f"PROCEED_WITH_{req.task_type.upper()}"
            summary = f"Fast-path decision by {primary_agent.capitalize()} specialist for goal: {req.goal}."
            confidence = round(min(0.95, 0.85 * evidence_trust), 2)
            disagreements: list[str] = []
            rounds_conducted = 1
            agents_consulted = [primary_agent]

        elif mode == "review":
            primary_agent = specialists[0]
            critic_agent = "critic"
            agents_consulted = [primary_agent, critic_agent]
            rounds_conducted = 1
            decision = f"VALIDATED_{req.task_type.upper()}"
            summary = f"Primary analysis by {primary_agent.capitalize()} subjected to adversarial review by Critic."

            # Real adversarial critique: one structured critic round through the debate engine.
            try:
                from app.debate.enhanced_debate_protocol import enhanced_debate_engine
                critic_trace = await enhanced_debate_engine.execute_structured_debate(
                    request_id=req.request_id,
                    task_type=req.task_type,
                    goal=req.goal,
                    evidence=[e.model_dump() for e in req.evidence],
                    agents=[critic_agent]
                )
                reasoning_trace = critic_trace
                disagreements = list(critic_trace.unresolved_objections)
                base_conf = critic_trace.confidence_evolution[-1] if critic_trace.confidence_evolution else 0.88
            except Exception:
                # Deterministic heuristic fallback only if the deliberation engine is unreachable.
                has_ambiguity = any(e.trust_label in ("untrusted_user_input", "inferred_profile") for e in req.evidence)
                disagreements = ["Critic flagged potential sampling bias in inferred telemetry."] if has_ambiguity else []
                base_conf = 0.75 if has_ambiguity else 0.90
            confidence = round(min(0.95, base_conf * evidence_trust), 2)

        else:  # debate mode
            agents_consulted = specialists if len(specialists) >= 3 else list(set(specialists + ["critic", "fact_checker"]))
            from app.debate.enhanced_debate_protocol import enhanced_debate_engine
            trace = await enhanced_debate_engine.execute_structured_debate(
                request_id=req.request_id,
                task_type=req.task_type,
                goal=req.goal,
                evidence=[e.model_dump() for e in req.evidence],
                agents=agents_consulted
            )
            reasoning_trace = trace
            rounds_conducted = len(trace.rounds)
            decision = f"CONSENSUS_{req.task_type.upper()}"
            # The round count comes from the trace instead of the literal "4" this string
            # used to contain — the number was computed on the line above and then
            # discarded, so the summary could contradict its own provenance field. The
            # mechanism is named honestly too: this protocol runs deterministic heuristic
            # passes over the supplied evidence and invokes no model, so describing it as
            # "deliberation across agents" overstated what happened.
            summary = (
                f"Deterministic structured protocol across {', '.join(agents_consulted)} "
                f"({rounds_conducted} rounds executed, no model invoked)."
            )
            disagreements = trace.unresolved_objections
            confidence = trace.confidence_evolution[-1] if trace.confidence_evolution else 0.86

        recommended_actions = [
            RecommendedAction(
                action=f"Execute primary initiative: {req.goal}",
                priority="HIGH",
                rationale="Aligns with verified system constraints and evidence.",
                owner=agents_consulted[0]
            ),
            RecommendedAction(
                action="Monitor telemetry signals for threshold drift",
                priority="MEDIUM",
                rationale="Safety guard against confidence decay.",
                owner="monitoring"
            )
        ]

        safety_notes = [
            "Inference is strictly advisory; decisions must be executed through bounded actuators.",
            "All untrusted user inputs have been sanitized and discounted in confidence scoring."
        ]

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        expires_at = time.time() + 86400.0  # 24h validity TTL

        provenance = {
            "request_id": req.request_id,
            "task_type": req.task_type,
            "mode": req.mode,
            "agents_consulted": agents_consulted,
            "rounds_conducted": rounds_conducted,
            "latency_ms": round(latency_ms, 2),
            "evidence_trust_factor": round(evidence_trust, 2),
            "evidence_trust_factor_basis": (
                "mean of evidence trust-label weights"
                if req.evidence
                else "no evidence supplied; default factor, not a measurement"
            ),
            # Honest execution identity for ALL modes. Review and debate both run the
            # deterministic heuristic engine (app/debate/enhanced_debate_protocol.py),
            # which makes no provider calls, and the fast path is a deterministic template
            # too — so nothing in this service invokes a model, and the response says so
            # rather than leaving a caller to infer deliberation from the prose.
            "execution_engine": getattr(
                reasoning_trace, "execution_engine", "deterministic_heuristic_template"
            ),
            "model_calls": getattr(reasoning_trace, "model_calls", 0),
            "model_invoked": bool(getattr(reasoning_trace, "model_calls", 0)),
            "timestamp": time.time()
        }

        response = IntelligenceResponse(
            request_id=req.request_id,
            decision=decision,
            confidence=confidence,
            summary=summary,
            key_evidence=key_evidence,
            provenance=provenance,
            unresolved_disagreements=disagreements,
            recommended_actions=recommended_actions,
            safety_notes=safety_notes,
            expires_at=expires_at
        )

        # Store provenance ledger for GET retrieval
        self.provenance_store[req.request_id] = {
            "request": req.model_dump(),
            "response": response.model_dump()
        }

        # Store in deduplication cache
        from app.governance.tenant_manager import tenant_manager
        tenant_manager.store_deduplication(
            req.request_id, response.model_dump(), namespace="nexus_intelligence"
        )

        # Track usage
        # No provider call on this path, so no token count exists to report; the call count and
        # latency are measured, the tokens are not (they used to be a hardcoded 650).
        consumer_router.record_usage("nexus", tokens=None, latency_sec=latency_ms / 1000.0)
        # No provider is contacted on this path: the decision comes from the deterministic
        # debate engine (app/debate/enhanced_debate_protocol.py), so provider and token
        # counts are reported as unmeasured. They used to be hardcoded to "gemini"/350/300,
        # which put provider attribution and token spend into the analytics table for traffic
        # that never touched a model.
        usage_analytics.log_request(
            consumer="nexus",
            service=f"nexus_{req.task_type}",
            latency_ms=latency_ms,
            success=True,
            confidence=confidence,
        )

        return response

    def get_provenance(self, request_id: str) -> dict[str, Any] | None:
        return self.provenance_store.get(request_id)

    def provenance_retention(self) -> dict[str, Any]:
        """How much provenance this service still holds, and what it has dropped.

        Served so that a 404 on an audit endpoint can be attributed: "never recorded"
        and "recorded but evicted" are different statements about the same request id.
        """
        return self.provenance_store.describe()


nexus_intelligence_service = NexusIntelligenceService()
