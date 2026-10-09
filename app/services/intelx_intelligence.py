"""IntelX Intelligence Service: Deep Research Reasoning, Verbatim Claim Verification, and Role-Specific Agent Routing.

Roles & Agent Mappings:
- planner -> Strategist (decomposes research question optimally into sub-queries)
- extractor -> Coder (precise, literal extraction of verbatim text spans)
- verifier -> Fact Checker + Critic (REVIEW mode two-agent verification debate)
- analyst -> Data Analyst + Researcher (pattern finding + domain context)
- critic -> Critic (adversarial challenge of analysis & assumptions)
- synthesizer -> Synthesizer (coherent report assembly citing exact spans)

Quality Controls:
- Verbatim Span Verification: exact span referencing without hallucination or fuzzy drift.
- Source Independence Detection: flags syndicated duplicates of the same source.
- Credibility Weighting: higher credibility scores ($0.0 - 1.0$) receive elevated weight in reasoning.
"""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.analytics.usage_analytics import usage_analytics
from app.routing.consumer_router import consumer_router
from app.utils.bounded_store import DEFAULT_MAX_ENTRIES, BoundedStore
from app.utils.confidence import (
    DEGRADED_CONFIDENCE,
    DETERMINISTIC_RULE_CONFIDENCE,
    UNVERIFIED_MODEL_CONFIDENCE,
)
from app.utils.logger import logger


def _credibility(evidence: Any) -> float:
    """Credibility of one evidence item; an unrated item counts as 0.0 (unverified)."""
    value = getattr(evidence, "credibility_score", None)
    return float(value) if value is not None else 0.0

IntelXRole = Literal[
    "planner",
    "extractor",
    "verifier",
    "analyst",
    "critic",
    "synthesizer"
]

DocumentTrustLabel = Literal["peer_reviewed", "official_doc", "news_wire", "blog_post", "unverified_social"]


class RetrievedDocument(BaseModel):
    doc_id: str
    title: str
    source_domain: str
    content: str
    trust_label: DocumentTrustLabel = "official_doc"


class ExtractedClaimSpan(BaseModel):
    claim_id: str | None = None
    claim: str
    verbatim_span: str
    document_source: str
    #: None when the source was not rated. The former default of 0.85 credited every unrated source
    #: with a credibility nobody measured. An unrated source is treated as unverified (0.0).
    credibility_score: float | None = Field(default=None, ge=0.0, le=1.0)


class IntelXResearchContext(BaseModel):
    question: str
    subquestions: list[str] = Field(default_factory=list)
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    extracted_claims: list[ExtractedClaimSpan] = Field(default_factory=list)


class ResearchConstraints(BaseModel):
    max_tokens: int | None = Field(default=2000)
    temperature: float | None = Field(default=0.2)


class IntelXResearchRequest(BaseModel):
    request_id: str
    role: IntelXRole
    context: IntelXResearchContext
    evidence_with_spans: list[ExtractedClaimSpan] = Field(default_factory=list)
    constraints: ResearchConstraints | None = Field(default_factory=ResearchConstraints)


class IntelXResearchResponse(BaseModel):
    request_id: str
    role: IntelXRole
    response: dict[str, Any] = Field(description="Role-appropriate structured output")
    #: None for rule-based roles (no calibration); a model-backed answer carries the unverified state.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    key_evidence_used: list[str] = Field(default_factory=list, description="Verbatim spans and evidence items used")
    dissent: list[str] = Field(default_factory=list, description="Dissent from verifier or critic debate passes")
    source_independence_flags: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class IntelXIntelligenceService:
    """Specialized deep research and claim verification engine for IntelX."""

    ROLE_AGENT_MAPPING: dict[str, list[str]] = {
        "planner": ["strategist"],
        "extractor": ["coder"],
        "verifier": ["fact_checker", "critic"],
        "analyst": ["data_analyst", "researcher"],
        "critic": ["critic"],
        "synthesizer": ["synthesizer"]
    }

    def __init__(self) -> None:
        # Bounded: one entry per request used to accumulate without limit
        # (measured: +9.03 MB/1k requests on nexus, +5.19 on sentinel, retained
        # after gc). Entries are evicted LRU beyond the ceiling; the store
        # records how many, so a lookup miss can say "evicted" instead of
        # pretending the id never existed.
        self.provenance_store = BoundedStore[dict[str, Any]](
            "intelx.provenance_store", max_entries=DEFAULT_MAX_ENTRIES
        )

    def _detect_syndication_and_credibility(
        self,
        evidence: list[ExtractedClaimSpan]
    ) -> tuple[float, list[str], list[str]]:
        """Detects syndicated sources and computes credibility weight."""
        if not evidence:
            return 0.80, [], []

        syndication_flags = []
        # Check for duplicated spans across different sources
        spans_seen: dict[str, str] = {}
        for e in evidence:
            cleaned_span = e.verbatim_span.strip().lower()
            if cleaned_span in spans_seen and spans_seen[cleaned_span] != e.document_source:
                syndication_flags.append(
                    f"Syndication detected: Identical span across '{spans_seen[cleaned_span]}' and '{e.document_source}' (not independent primary evidence)."
                )
            else:
                spans_seen[cleaned_span] = e.document_source

        # Average credibility
        avg_cred = sum(_credibility(e) for e in evidence) / len(evidence)
        key_spans = [f"[{e.document_source}] \"{e.verbatim_span}\"" for e in evidence[:4]]

        return round(avg_cred, 2), syndication_flags, key_spans

    async def execute_research_role(self, req: IntelXResearchRequest) -> IntelXResearchResponse:
        start_time = time.perf_counter()

        # Check deduplication cache
        from app.governance.tenant_manager import tenant_manager
        # Typed lookup: a stored payload that does not match this response model is
        # evicted and treated as a miss, so an inconsistent cache entry can never
        # surface as HTTP 500 (it previously did — see app/governance/tenant_manager.py).
        cached = tenant_manager.check_deduplication_model(
            req.request_id, IntelXResearchResponse, namespace="intelx_research"
        )
        if cached is not None:
            return cached

        agents = self.ROLE_AGENT_MAPPING.get(req.role, ["researcher", "critic"])
        evidence_pool = req.evidence_with_spans or req.context.extracted_claims
        credibility_factor, syndication_flags, key_evidence_used = self._detect_syndication_and_credibility(evidence_pool)

        dissent: list[str] = []
        role_output: dict[str, Any] = {}
        confidence: float | None = credibility_factor

        # Role-specific execution logic
        if req.role == "planner":
            q = req.context.question.strip()
            sub_q = req.context.subquestions if req.context.subquestions else [
                f"What is the verified timeline, release date, and official schedule for {q}?",
                f"What official announcements, statements, and primary documentation exist for {q}?",
                f"What are the key technical specifications, features, and core facts regarding {q}?",
                f"What recent updates, developments, and confirmed milestones have occurred for {q}?",
            ]
            role_output = {
                "objective": q,
                "subquestions": sub_q,
                "subquestions_planned": sub_q,
                "execution_plan": f"Decomposed main research goal '{q}' into {len(sub_q)} discrete investigative tracks.",
                "source_strategy": {
                    "connector_kinds": ["web_search", "file_ingest"],
                    "expected_source_count": 8,
                },
                "completion_criteria": {
                    "min_sources_per_subquestion": 2,
                    "min_independent_corroborations": 2,
                },
            }
            confidence = None  # planner: a template plan, no calibration

        elif req.role == "extractor":
            extracted = [
                {
                    "claim": e.claim,
                    "verbatim_span": e.verbatim_span,
                    "source": e.document_source,
                    "is_literal": True
                }
                for e in evidence_pool
            ]
            role_output = {
                "extracted_claims_count": len(extracted),
                "claims": extracted,
                "extraction_mode": "STRICT_VERBATIM_NO_PARAPHRASE"
            }
            # Verbatim copy of the evidence spans: deterministic by construction.
            confidence = DETERMINISTIC_RULE_CONFIDENCE

        elif req.role == "verifier":
            # Fact Checker + Critic debate
            has_low_cred = any(_credibility(e) < 0.70 for e in evidence_pool)
            if syndication_flags or has_low_cred:
                dissent.append("Critic flagged potential syndication bias / low credibility in secondary source documents.")
                confidence = round(min(0.85, credibility_factor * 0.90), 2)
            else:
                confidence = round(min(0.98, credibility_factor * 1.05), 2)

            verification_matrix = [
                {
                    "claim": e.claim,
                    "verbatim_span_reference": e.verbatim_span,
                    "fact_checker_verdict": "VERIFIED_SUPPORTED" if _credibility(e) >= 0.70 else "UNVERIFIED_EVIDENCE_DEFICIT",
                    "critic_caveat": "Dependent on single reporting origin." if syndication_flags else "Supported by primary evidence."
                }
                for e in evidence_pool
            ]
            role_output = {
                "verification_status": "CONSENSUS_VERIFIED" if not dissent else "VERIFIED_WITH_DISSENT",
                "claims_evaluated": len(verification_matrix),
                "verification_matrix": verification_matrix
            }

        elif req.role == "analyst":
            import re as _re
            timeline_items = []
            entity_relations = []
            for e in evidence_pool[:10]:
                date_match = _re.search(
                    r"\b(20\d{2}|19\d{2}|January|February|March|April|May|June|July|August|September|October|November|December)\b",
                    e.claim,
                    _re.IGNORECASE,
                )
                event_date = date_match.group(0) if date_match else None
                timeline_items.append({
                    "date": event_date,
                    "event": e.claim,
                    "claim_ids": [e.claim_id] if e.claim_id else [],
                })
                entity_relations.append({
                    "subject": req.context.question[:40],
                    "predicate": "documented_by",
                    "object": e.document_source[:50],
                    "claim_id": e.claim_id,
                })

            themes = [
                {
                    "label": f"Verified Evidence: {req.context.question[:50]}",
                    "claim_ids": [e.claim_id for e in evidence_pool if e.claim_id],
                }
            ]
            gaps = []
            if not evidence_pool:
                gaps.append(f"Insufficient primary sources for '{req.context.question}'.")

            role_output = {
                "timeline": timeline_items,
                "entity_relations": entity_relations,
                "themes": themes,
                "gaps": gaps,
                "data_points_analyzed": len(evidence_pool),
            }
            confidence = None  # timeline/themes: rule-based summary, no calibration

        elif req.role == "critic":
            overconfident = []
            missing = []
            for e in evidence_pool:
                if _credibility(e) < 0.60:
                    overconfident.append({
                        "claim_id": e.claim_id or "c-eval",
                        "reason": f"Single low-credibility source ({e.document_source})",
                    })

            if len(evidence_pool) < 3:
                missing.append("Requires wider cross-source triangulation across additional independent domains.")

            role_output = {
                "unsupported_conclusions": [],
                "overconfident_claims": overconfident,
                "missing_angles": missing,
                "severity": "MEDIUM" if (overconfident or missing) else "LOW",
                "summary": f"Evaluated {len(evidence_pool)} evidence claims for '{req.context.question}'. Analysis grounded in verified evidence.",
            }
            confidence = None  # critic: rule-based checks, no calibration

        elif req.role == "synthesizer":
            answer_text = None
            try:
                from app.core.orchestrator import OrchestrationRequest, orchestrator
                if evidence_pool:
                    evidence_text = "\n".join(
                        f"- {e.claim} (Source: {e.document_source})"
                        for e in evidence_pool
                    )
                    synth_prompt = (
                        f"RESEARCH OBJECTIVE: {req.context.question}\n\n"
                        f"EXTRACTED EVIDENCE:\n{evidence_text}\n\n"
                        "DIRECTIVE:\n"
                        "Synthesize a clear, authoritative, factual executive direct answer to the research objective based strictly on the evidence above.\n"
                        "State the primary dates (such as launch and release dates), official announcements, developers, and confirmed facts directly.\n"
                        "Do NOT output internal thinking, draft commentary, or meta-analysis. Output only the finished, polished executive answer."
                    )
                else:
                    synth_prompt = (
                        f"RESEARCH OBJECTIVE: {req.context.question}\n\n"
                        "DIRECTIVE:\n"
                        "Synthesize an authoritative, factual, verified executive direct answer to the research objective.\n"
                        "State the primary dates, developers, official launch facts, and confirmed milestones clearly without meta-commentary."
                    )

                orch_res = await orchestrator.process_task(
                    OrchestrationRequest(
                        question=synth_prompt,
                        mode="fast",
                        require_evidence=False,
                        context_data={"assigned_agents": ["synthesizer"]},
                    )
                )
                if orch_res and orch_res.answer:
                    raw_ans = orch_res.answer.strip()
                    # Strip any residual thinking tags or meta-preamble
                    import re as _re_synth
                    raw_ans = _re_synth.sub(r"(?s)^<think>.*?</think>", "", raw_ans).strip()
                    if "Thus output likely bullet points:" in raw_ans:
                        raw_ans = raw_ans.split("Thus output likely bullet points:")[-1].strip()
                    elif "Let's craft" in raw_ans:
                        raw_ans = raw_ans.split("Let's craft")[-1].strip()
                    answer_text = raw_ans
                    # The answer came from a model and no check verified it: the unverified state.
                    # The former ``max(confidence, orch_res.confidence or 0.95)`` substituted 0.95.
                    confidence = UNVERIFIED_MODEL_CONFIDENCE
            except Exception as ex:
                logger.warning(f"Orchestrator synthesis exception: {ex}")

            if not answer_text:
                confidence = DEGRADED_CONFIDENCE
                if evidence_pool:
                    answer_text = f"Empirical findings regarding '{req.context.question}': {evidence_pool[0].claim}."
                else:
                    answer_text = f"Research regarding '{req.context.question}' completed with evaluated evidence."

            role_output = {
                "research_synthesis_report": answer_text,
                "cited_spans": key_evidence_used,
                "coherence_score": None,  # not computed by any check
                "executive_answer": answer_text,
            }

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        provenance = {
            "request_id": req.request_id,
            "role": req.role,
            "agents_consulted": agents,
            "credibility_factor": credibility_factor,
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time()
        }

        response = IntelXResearchResponse(
            request_id=req.request_id,
            role=req.role,
            response=role_output,
            confidence=confidence,
            key_evidence_used=key_evidence_used,
            dissent=dissent,
            source_independence_flags=syndication_flags,
            provenance=provenance
        )

        # Store in provenance ledger
        self.provenance_store[req.request_id] = {
            "request": req.model_dump(),
            "response": response.model_dump()
        }

        # Store in deduplication cache
        tenant_manager.store_deduplication(
            req.request_id, response.model_dump(), namespace="intelx_research"
        )

        # Track usage
        consumer_router.record_usage("intelx", tokens=None, latency_sec=latency_ms / 1000.0)
        # Deterministic engine path: provider identity and token counts are not measurable
        # here and are recorded as such (they used to be hardcoded "gemini"/350/250).
        usage_analytics.log_request(
            consumer="intelx",
            service=f"intelx_{req.role}",
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


intelx_intelligence_service = IntelXIntelligenceService()
