"""Enhanced Debate Protocol, Reasoning Chains, Evidence Scoring, Assumption Tracking & Multi-Model Routing."""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.utils.bounded_store import DEFAULT_MAX_ENTRIES, BoundedStore


class EvidenceScore(BaseModel):
    evidence_id: str
    claim: str
    trust_label: str
    #: None: no relevance measure is computed for evidence. (Was a default of 0.90.)
    relevance_score: float | None = None
    reliability_weight: float = 0.90  # 0.3 for untrusted_user_input, 1.0 for system_fact
    is_contradictory: bool = False
    flag_notes: str | None = None


class StatedAssumption(BaseModel):
    agent: str
    hypothesis: str
    #: None: an assumption is not measured. The former fixed 0.5 / 0.3 were stated as confidence.
    confidence: float | None
    validation_status: Literal["PENDING", "VALIDATED", "INVALIDATED"] = "PENDING"


class DebateRound(BaseModel):
    round_number: int
    stage: Literal["INDEPENDENT_ANALYSIS", "CROSS_EXAMINATION", "SYNTHESIS_ATTEMPT", "FINAL_OBJECTIONS"]
    agent_outputs: dict[str, str]
    challenges: list[dict[str, str]] = Field(default_factory=list)
    defenses: list[dict[str, str]] = Field(default_factory=list)


class ReasoningChainTrace(BaseModel):
    request_id: str
    task_type: str
    mode: str
    # A *planned* routing assignment: which provider each agent would be routed to if
    # this protocol dispatched model calls. It is not evidence that any provider ran.
    # Measured before this was documented: a trace reported
    # ``provider_allocation={'debugger': 'gemini', 'security_analyst': 'groq', ...}``
    # while the engine made zero provider calls, and round-1 "agent outputs" were
    # prefixed ``[GEMINI]`` / ``[GROQ]`` — attributing text to vendors that never ran
    # (the same defect class as the speculative-race winner, see app/providers/gateway.py).
    provider_allocation: dict[str, str]
    # Provenance of the reasoning: what actually produced this trace. Consumers must read
    # these two fields before treating the rounds as model deliberation.
    execution_engine: str = "deterministic_heuristic_template"
    model_calls: int = 0
    provider_allocation_exercised: bool = False
    rounds: list[DebateRound]
    synthesis_logic: str
    unresolved_objections: list[str] = Field(default_factory=list)
    stated_assumptions: list[StatedAssumption] = Field(default_factory=list)
    evidence_scores: list[EvidenceScore] = Field(default_factory=list)
    confidence_evolution: list[float] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)


class EnhancedDebateEngine:
    """Orchestrates structured 4-round multi-agent adversarial debate with multi-model diversity."""

    def __init__(self) -> None:
        # Bounded: one trace per debate, previously unbounded (nothing ever removed an
        # entry, so a long-lived process accumulated every trace it had ever produced).
        self.reasoning_traces = BoundedStore[ReasoningChainTrace](
            "debate.reasoning_traces", max_entries=DEFAULT_MAX_ENTRIES
        )
        # Counters start at zero and only ever move because a debate actually ran. They
        # used to be seeded with a fiction — ``total_debates = 24`` before any request,
        # ``provider_diversity_success_rate = 93.8`` and
        # ``single_provider_success_rate = 81.2`` — which /v1/debate/statistics then
        # published as measured performance, including a derived "diversity lift" of
        # 12.6 percentage points. Nothing had been measured at all: the protocol is a
        # deterministic template that never calls a model, so it cannot observe a success
        # rate for a provider, diverse or otherwise.
        self.total_debates = 0
        self._rounds_total = 0
        self._objections_total = 0
        self._debates_with_objections = 0
        self._composition_counts: dict[str, int] = {}

    def score_evidence(self, evidence_list: list[dict[str, Any]]) -> list[EvidenceScore]:
        """Calculates relevance and reliability weights for all evidence."""
        scored: list[EvidenceScore] = []
        for idx, item in enumerate(evidence_list):
            label = item.get("trust_label", "verified_telemetry")
            claim = item.get("claim", "")

            # Low-trust penalty (0.3x)
            if label == "untrusted_user_input":
                rel_weight = 0.30
            elif label == "inferred_profile":
                rel_weight = 0.60
            elif label == "verified_telemetry":
                rel_weight = 0.90
            else:  # system_fact
                rel_weight = 1.00

            # Detect potential contradiction flags
            is_contra = "disconnect" in claim.lower() or "spike" in claim.lower() or "fail" in claim.lower()

            # Keep the caller's own identifier when they supplied one. Renumbering every
            # item as EVD-001..EVD-nnn discarded the only handle the caller had: nexus
            # passes evidence through with its own ids, and a trace that challenges
            # "EVD-001" cannot be mapped back to the item the caller sent. A generated id
            # is used only when none was given.
            supplied_id = item.get("evidence_id") or item.get("id")
            scored.append(
                EvidenceScore(
                    evidence_id=str(supplied_id) if supplied_id else f"EVD-{idx+1:03d}",
                    claim=claim,
                    trust_label=label,
                    relevance_score=None,
                    reliability_weight=rel_weight,
                    is_contradictory=is_contra,
                    flag_notes="Contradictory signal flagged for debate cross-examination" if is_contra else None
                )
            )
        return scored

    async def execute_structured_debate(
        self,
        request_id: str,
        task_type: str,
        goal: str,
        evidence: list[dict[str, Any]],
        agents: list[str]
    ) -> ReasoningChainTrace:
        """Executes the 4-round structured adversarial debate protocol."""
        scored_evidence = self.score_evidence(evidence)

        # Planned multi-provider routing. This is a *plan*: nothing here dispatches a
        # call, so the allocation is recorded as planned and flagged unexercised on the
        # trace rather than being presented as the origin of the round text.
        providers = ["gemini", "groq", "mistral", "nvidia", "openrouter"]
        provider_allocation = {agent: providers[idx % len(providers)] for idx, agent in enumerate(agents)}

        # Round 1: Independent Analysis (deterministic). The text states the mechanism
        # instead of wearing a vendor tag: the previous form produced
        # "[GEMINI] Debugger independent assessment for goal: ..." for every goal, which
        # reads as a model transcript and was emitted even when no provider was reachable.
        round1_outputs = {
            agent: (
                f"{agent} independent pass over {len(scored_evidence)} scored evidence item(s) "
                f"for goal '{goal}'; weight derived from trust labels, no model invoked."
            )
            for agent in agents
        }
        r1 = DebateRound(
            round_number=1,
            stage="INDEPENDENT_ANALYSIS",
            agent_outputs=round1_outputs
        )

        # Round 2: Cross-Examination, derived from the evidence actually supplied.
        #
        # These were four hardcoded strings: every debate — a database incident, a
        # compliance question, anything — was cross-examined for "constant market
        # liquidity" and defended with "dynamic stop boundaries", and the round reported
        # "completed across 2 adversarial exchanges" whether or not there was anything to
        # examine. Cross-examination now names the weakest evidence item (or its absence),
        # so the round says something specific to the request, and the exchange count is
        # the length of the lists it actually produced.
        weakest = min(scored_evidence, key=lambda e: e.reliability_weight) if scored_evidence else None
        challenger = "critic"
        second_challenger = "fact_checker" if "fact_checker" in agents else agents[-1]

        if weakest is None:
            challenges = [{
                "challenger": challenger,
                "target": agents[0],
                "challenge": (
                    "No evidence was supplied, so this assessment rests on the goal statement "
                    "alone and cannot be corroborated. Objection registered."
                ),
            }]
            defenses = [{
                "defender": agents[0],
                "defense": (
                    "Acknowledged: with no evidence the protocol can only restate the goal; "
                    "the confidence reflects that."
                ),
            }]
        else:
            challenges = [{
                "challenger": challenger,
                "target": agents[0],
                "challenge": (
                    f"Weakest evidence item is {weakest.evidence_id!r} "
                    f"(trust={weakest.trust_label}, weight={weakest.reliability_weight:.2f}); "
                    f"the assessment inherits that weakness."
                ),
            }]
            defenses = [{
                "defender": agents[0],
                "defense": (
                    f"{weakest.evidence_id!r} is weighted down to "
                    f"{weakest.reliability_weight:.2f} by its trust label, so its influence on "
                    f"the synthesis is bounded by that factor."
                ),
            }]
            # Only when a second, distinct agent is available: with a three-agent panel
            # ``agents[-1]`` is often "critic" itself, which produced a challenge whose
            # challenger and target were the same agent.
            if len(scored_evidence) > 1 and second_challenger != challenger:
                challenges.append({
                    "challenger": second_challenger,
                    "target": challenger,
                    "challenge": (
                        "The critique addresses a single item; the remaining "
                        f"{len(scored_evidence) - 1} item(s) are unchallenged."
                    ),
                })
                defenses.append({
                    "defender": challenger,
                    "defense": (
                        "Only the lowest-reliability item can change the outcome materially; "
                        "the stronger items already dominate the weighted mean."
                    ),
                })

        r2 = DebateRound(
            round_number=2,
            stage="CROSS_EXAMINATION",
            agent_outputs={
                "status": (
                    f"Cross-examination completed across {len(challenges)} adversarial "
                    f"exchange(s)."
                )
            },
            challenges=challenges,
            defenses=defenses
        )

        # Round 3: Synthesis Attempt, describing the weighting it applied.
        synth_agent = "synthesizer"
        if scored_evidence:
            by_label: dict[str, int] = {}
            for item in scored_evidence:
                by_label[item.trust_label] = by_label.get(item.trust_label, 0) + 1
            breakdown = ", ".join(f"{label}={count}" for label, count in sorted(by_label.items()))
            synthesis_logic = (
                f"Synthesizer combined {len(scored_evidence)} evidence item(s) "
                f"({breakdown}) into a reliability-weighted mean of "
                f"{self._derive_confidence_evolution(scored_evidence)[0]:.2f}; no item was "
                f"discarded and no model was invoked."
            )
        else:
            synthesis_logic = (
                "Synthesizer had no evidence to combine; the assessment is derived from the "
                "goal statement only."
            )
        r3 = DebateRound(
            round_number=3,
            stage="SYNTHESIS_ATTEMPT",
            agent_outputs={synth_agent: synthesis_logic}
        )

        # Round 4: Final Objections, registered only when the evidence warrants one.
        #
        # The previous list held a single hardcoded string — "Telemetry must be re-sampled
        # within 6 hours to confirm absence of drift" — returned as the debate's
        # unresolved_objections for every request, including one with no telemetry at all.
        # It was also the source of the fabricated "objection_rate_pct" (every debate
        # registered exactly this objection, so a real rate could never have been 24.5%).
        objections: list[str] = []
        weak_items = [e for e in scored_evidence if e.reliability_weight < 0.7]
        if not scored_evidence:
            objections.append(
                "Critic note: no evidence was supplied, so the conclusion cannot be "
                "corroborated and must not be relied on without independent data."
            )
        elif weak_items:
            weakest_ids = ", ".join(e.evidence_id for e in weak_items)
            objections.append(
                f"Critic note: {len(weak_items)} evidence item(s) carry reduced trust "
                f"({weakest_ids}); corroborate them before acting on this assessment."
            )
        elif any(e.is_contradictory for e in scored_evidence):
            contradictions = ", ".join(
                e.evidence_id for e in scored_evidence if e.is_contradictory
            )
            objections.append(
                f"Critic note: contradictory evidence detected ({contradictions}); the "
                f"synthesis resolved it by weighting, not by verification."
            )
        # Otherwise the panel has no unresolved objection, and it now says so by returning
        # an empty list rather than inventing one.
        r4 = DebateRound(
            round_number=4,
            stage="FINAL_OBJECTIONS",
            agent_outputs={"objections_registered": str(len(objections))},
            challenges=[],
            defenses=[]
        )

        # Assumptions named after what they actually rest on, instead of two canned
        # market-specific statements ("constant market liquidity", "adverse slippage risk
        # ... if external API latency exceeds 500ms") that were attached to every task type.
        if scored_evidence:
            strongest = max(scored_evidence, key=lambda e: e.reliability_weight)
            assumptions = [
                StatedAssumption(
                    agent=agents[0],
                    hypothesis=(
                        f"Evidence {strongest.evidence_id!r} (trust="
                        f"{strongest.trust_label}) is representative of the situation and has "
                        f"not changed since it was captured."
                    ),
                    confidence=round(strongest.reliability_weight, 2),
                ),
                StatedAssumption(
                    agent="critic",
                    hypothesis=(
                        "Items marked untrusted or inferred are assumed to be incomplete "
                        "rather than false; the weighting reflects uncertainty, not dishonesty."
                    ),
                    confidence=None,  # an assumption carries no measured confidence
                ),
            ]
        else:
            assumptions = [
                StatedAssumption(
                    agent=agents[0],
                    hypothesis="The goal statement is a complete description of the problem.",
                    confidence=None,
                )
            ]

        trace = ReasoningChainTrace(
            request_id=request_id,
            task_type=task_type,
            mode="debate",
            provider_allocation=provider_allocation,
            provider_allocation_exercised=False,
            execution_engine="deterministic_heuristic_template",
            model_calls=0,
            rounds=[r1, r2, r3, r4],
            synthesis_logic=synthesis_logic,
            unresolved_objections=objections,
            stated_assumptions=assumptions,
            evidence_scores=scored_evidence,
            confidence_evolution=self._derive_confidence_evolution(scored_evidence),
        )

        self.reasoning_traces[request_id] = trace
        self._record_debate(trace)
        return trace

    @staticmethod
    def _derive_confidence_evolution(scored_evidence: list[EvidenceScore]) -> list[float]:
        """Compute the per-round confidence path from the supplied evidence.

        The previous value was the literal ``[0.82, 0.76, 0.88, 0.86]`` — a fixed
        sequence published as a debate's confidence trajectory and consumed by callers as
        ``confidence = confidence_evolution[-1]`` (0.86 forever, regardless of input). A
        constant cannot be a trajectory. This computes the path from the evidence the
        caller actually supplied: the base is the mean reliability weight, cross
        examination discounts it (a challenge is raised against every goal), synthesis
        recovers part of the discount only when the evidence is trustworthy, and final
        objections apply the residual discount. The result is still deterministic — the
        protocol makes no model calls — but it now varies with the input, which is the
        difference between a measurement and a decoration.
        """
        if not scored_evidence:
            # Derived from the trust scale, not chosen by feel. The floor of that scale is
            # 0.30 — see score_evidence: an ``untrusted_user_input`` item weighs 0.30, the
            # lowest weight any evidence can carry. "No evidence at all" must therefore sit
            # strictly *below* 0.30, or a system with no basis for an answer would claim
            # more confidence than one holding weak grounds for it. Two earlier values were
            # wrong in exactly that way and were caught by
            # tests/test_no_fabricated_measurements.py::
            # test_confidence_evolution_is_derived_from_the_supplied_evidence — first an
            # arbitrary 0.55 (above the whole scale), then 0.30 (equal to the floor, so
            # "nothing" and "distrusted something" scored identically).
            base = 0.15
        else:
            base = sum(e.reliability_weight for e in scored_evidence) / len(scored_evidence)
            base = max(0.05, min(0.95, base))
        challenged = round(max(0.05, base - 0.06), 3)
        # Synthesis can only recover ground when the evidence is strong enough to support
        # it; weak evidence stays weak rather than being talked up by a later round.
        synthesised = round(min(0.95, challenged + 0.12 * base), 3)
        final = round(max(0.05, synthesised - 0.02), 3)
        return [round(base, 3), challenged, synthesised, final]

    def _record_debate(self, trace: ReasoningChainTrace) -> None:
        """Accumulate real observations so the statistics endpoint can report them."""
        self.total_debates += 1
        self._rounds_total += len(trace.rounds)
        self._objections_total += len(trace.unresolved_objections)
        if trace.unresolved_objections:
            self._debates_with_objections += 1
        composition_key = ", ".join(sorted(trace.provider_allocation)) or "none"
        self._composition_counts[composition_key] = self._composition_counts.get(composition_key, 0) + 1


    def get_trace(self, request_id: str) -> ReasoningChainTrace | None:
        return self.reasoning_traces.get(request_id)

    def trace_was_evicted(self, request_id: str) -> bool:
        """True when this request id had a trace that was dropped for capacity.

        Lets the trace endpoint answer "expired from the retention window" instead of
        "not found", which are different statements about the same request.
        """
        return self.reasoning_traces.was_evicted(request_id)

    def trace_retention(self) -> dict[str, Any]:
        return self.reasoning_traces.describe()

    #: Below this many observations a rate is not reported. A percentage over a handful of
    #: runs invites a decision it cannot support, so the endpoint says so instead.
    MIN_SAMPLES_FOR_RATES = 20

    def get_debate_statistics(self) -> dict[str, Any]:
        """Return debate metrics that were actually observed, with their sample sizes.

        This endpoint previously published a fabricated performance study:

        * ``total_structured_debates: 24`` before a single debate had been requested.
        * ``objection_rate_pct: 24.5`` — a constant; the engine registers exactly one
          objection per debate by construction, so no run could have produced 24.5%.
        * ``provider_diversity_impact`` reporting 93.8% for multi-model versus 81.2% for
          single-model debates with a derived "diversity_lift_pct" of 12.6 — presented as
          empirical evidence for a design decision, from two hardcoded floats.
        * ``composition_success_distribution`` with three invented compositions and
          invented success rates (94.2 / 92.8 / 91.5).

        None of that was measured, and this protocol cannot measure it: it makes no model
        calls, so it has no provider to attribute a success to. What it *can* report is
        what it did — how many traces it produced, how many rounds and objections they
        contained, and which agent sets were used — so that is what it reports now. Rates
        appear only once ``MIN_SAMPLES_FOR_RATES`` observations exist, and carry
        ``sample_size`` next to them.
        """
        debates = self.total_debates
        has_sample = debates >= self.MIN_SAMPLES_FOR_RATES
        stats: dict[str, Any] = {
            # What this engine is, stated where a consumer will see it next to the numbers.
            "execution_engine": "deterministic_heuristic_template",
            "model_calls": 0,
            "measurement_note": (
                "Rounds are deterministic heuristic passes over caller-supplied evidence; "
                "no model provider is invoked, so no provider performance can be derived "
                "from this engine."
            ),
            "total_structured_debates": debates,
            "total_rounds_conducted": self._rounds_total,
            "total_objections_registered": self._objections_total,
            "average_rounds_conducted": (
                round(self._rounds_total / debates, 2) if debates else None
            ),
            "objection_rate_pct": (
                round(100.0 * self._debates_with_objections / debates, 1) if debates else None
            ),
            "composition_usage": dict(sorted(self._composition_counts.items())),
        }
        if not has_sample:
            # Explicit absence beats a confident fabricated number.
            stats["insufficient_data"] = (
                f"{debates} debate(s) observed; at least {self.MIN_SAMPLES_FOR_RATES} are "
                "required before any rate is reported."
            )
            stats["provider_diversity_impact"] = None
            stats["composition_success_distribution"] = None
            return stats

        stats["sample_size"] = debates
        # Provider diversity cannot be assessed here: the engine records a *planned*
        # allocation and never dispatches, so there is no diverse-vs-single outcome to
        # compare. Saying so is the honest answer, not a number.
        stats["provider_diversity_impact"] = {
            "available": False,
            "reason": (
                "This protocol does not dispatch to providers, so multi-model versus "
                "single-model outcomes are not observable from it. Use provider telemetry "
                "from app/monitoring.py instead."
            ),
        }
        stats["composition_success_distribution"] = {
            "available": False,
            "reason": (
                "Success requires an outcome signal; this engine records composition usage "
                "only. See composition_usage."
            ),
        }
        return stats


enhanced_debate_engine = EnhancedDebateEngine()
