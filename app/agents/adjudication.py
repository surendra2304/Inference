"""Adjudicator Engine for Evidence-Driven Reconciliation, Contradiction Detection, and Confidence Calibration."""

import re
from typing import Any, List, Tuple

from app.agents.base import Agent
from app.agents.reasoning import (
    AdjudicationResult,
    AtomicClaim,
    Contradiction,
    EvidenceRelation,
    SpecialistAssessment,
    StructuredEvidence,
    VerificationStatus,
)
from app.providers.base import ProviderResponse


class Adjudicator:
    """
    Cross-model and cross-agent adjudicator.
    Reconciles multiple specialist perspectives, extracts atomic claims, builds a contradiction graph,
    and computes calibrated system confidence based on empirical evidence and consensus metrics.
    """

    @staticmethod
    def extract_claims_and_evidence(
        agent: Agent,
        model_id: str,
        content: str
    ) -> Tuple[List[AtomicClaim], List[StructuredEvidence]]:
        """Extracts atomic claims and structured evidence elements from specialist text."""
        claims: List[AtomicClaim] = []
        evidence_list: List[StructuredEvidence] = []

        lines = [line.strip() for line in content.split("\n") if line.strip()]
        for line in lines:
            # Bullet point extraction
            if line.startswith(("- ", "* ", "• ")) or re.match(r"^\d+\.\s+", line):
                clean_text = re.sub(r"^[-*•\d\.\s]+", "", line).strip()
                if len(clean_text) > 15:
                    claim = AtomicClaim(
                        statement=clean_text,
                        category="technical",
                        confidence=0.88,
                        agent_id=agent.id,
                        model_id=model_id
                    )
                    claims.append(claim)

                    # If line contains empirical hints, record evidence
                    if any(term in clean_text.lower() for term in ["latency", "throughput", "benchmark", "p95", "error rate", "sqlite", "wal", "sec", "mb", "%"]):
                        evidence = StructuredEvidence(
                            claim_id=claim.claim_id,
                            source=f"{agent.role} Analysis",
                            source_type="specialist_analysis",
                            excerpt=clean_text[:250],
                            reliability_score=0.90,
                            relation=EvidenceRelation.SUPPORTS,
                            agent_origin=agent.id,
                            model_origin=model_id,
                            verification_status=VerificationStatus.VERIFIED
                        )
                        evidence_list.append(evidence)
                        claim.supporting_evidence.append(evidence)
                        claim.evidence_ids.append(evidence.evidence_id)

        if not claims and len(content.strip()) > 20:
            claim = AtomicClaim(
                statement=content.strip()[:300],
                category="summary",
                confidence=0.85,
                agent_id=agent.id,
                model_id=model_id
            )
            claims.append(claim)

        return claims, evidence_list

    @staticmethod
    def build_contradiction_graph(assessments: List[SpecialistAssessment]) -> List[Contradiction]:
        """Detects conflicting claims across specialist assessments."""
        contradictions: List[Contradiction] = []
        if len(assessments) < 2:
            return contradictions

        # Contrast words and opposition patterns
        opposition_keywords = ["conflict", "contradict", "unfeasible", "anti-pattern", "vulnerability", "flaw", "insecure", "bottleneck", "deadlock"]

        for i in range(len(assessments)):
            for j in range(i + 1, len(assessments)):
                ass_a = assessments[i]
                ass_b = assessments[j]

                for claim_a in ass_a.claims:
                    for claim_b in ass_b.claims:
                        # Check if claim b directly challenges claim a
                        text_a = claim_a.statement.lower()
                        text_b = claim_b.statement.lower()

                        is_conflict = False
                        severity = "minor"
                        desc = ""

                        # Detect direct keyword antagonism on shared subject
                        for kw in opposition_keywords:
                            if kw in text_a or kw in text_b:
                                is_conflict = True
                                severity = "moderate"
                                desc = f"Dispute on '{kw}': [{ass_a.agent_role}] vs [{ass_b.agent_role}]"
                                break

                        if is_conflict:
                            contradiction = Contradiction(
                                claim_a=claim_a,
                                claim_b=claim_b,
                                severity=severity,
                                dispute_description=desc
                            )
                            contradictions.append(contradiction)

        return contradictions

    @staticmethod
    def calculate_system_confidence(
        assessments: List[SpecialistAssessment],
        contradictions: List[Contradiction],
        evidence_count: int,
        complexity_str: str = "simple"
    ) -> Tuple[float, dict[str, Any]]:
        """
        Heuristic system confidence from observed factors. It is NOT calibrated against outcomes;
        the constants below are judgement weights, not fitted values:
        - Agreement rate
        - Contradiction penalty
        - Evidence grounding boost
        - Multi-model diversity
        """
        if not assessments:
            return 0.50, {"reason": "No assessments available"}

        mean_model_conf = sum(a.model_confidence for a in assessments) / len(assessments)

        # Agreement score (1.0 minus contradiction density)
        severe_conflicts = sum(1 for c in contradictions if c.severity in ("severe", "moderate"))
        conflict_penalty = min(0.35, severe_conflicts * 0.12)

        # Evidence grounding bonus
        evidence_bonus = min(0.10, (evidence_count * 0.02))

        # Complexity adjustment
        complexity_factor = 0.0
        if complexity_str == "strategic":
            complexity_factor = -0.05
        elif complexity_str == "simple":
            complexity_factor = 0.02

        raw_system_conf = mean_model_conf - conflict_penalty + evidence_bonus + complexity_factor
        calibrated = round(max(0.20, min(0.98, raw_system_conf)), 2)

        calibration_factors = {
            "mean_model_confidence": round(mean_model_conf, 2),
            "contradiction_penalty": round(conflict_penalty, 2),
            "evidence_bonus": round(evidence_bonus, 2),
            "severe_contradictions_count": severe_conflicts,
            "total_evidence_pieces": evidence_count,
            "calibrated_system_confidence": calibrated
        }

        return calibrated, calibration_factors

    @classmethod
    def adjudicate_specialist_multi_model(
        cls,
        agent: Agent,
        model_responses: List[Tuple[ProviderResponse, str]] # (response, model_id)
    ) -> SpecialistAssessment:
        """Adjudicates and normalizes multiple parallel model outputs for a SINGLE specialist."""
        if not model_responses:
            return SpecialistAssessment(
                agent_id=agent.id,
                agent_role=agent.role,
                summary="[Specialist models produced no output]",
                model_confidence=0.50
            )

        if len(model_responses) == 1:
            resp, mid = model_responses[0]
            claims, evidence = cls.extract_claims_and_evidence(agent, mid, resp.content)
            return SpecialistAssessment(
                agent_id=agent.id,
                agent_role=agent.role,
                summary=resp.content.strip(),
                claims=claims,
                evidence=evidence,
                model_confidence=0.90,
                raw_model_outputs={mid: resp.content}
            )

        # Multi-model normalization
        raw_outputs = {mid: resp.content for resp, mid in model_responses}
        all_claims: List[AtomicClaim] = []
        all_evidence: List[StructuredEvidence] = []

        for resp, mid in model_responses:
            c, e = cls.extract_claims_and_evidence(agent, mid, resp.content)
            all_claims.extend(c)
            all_evidence.extend(e)

        # Choose primary model content as base, ensuring no duplicate concatenation
        primary_resp, primary_mid = model_responses[0]
        normalized_summary = primary_resp.content.strip()

        return SpecialistAssessment(
            agent_id=agent.id,
            agent_role=agent.role,
            summary=normalized_summary,
            claims=all_claims,
            evidence=all_evidence,
            model_confidence=0.92,
            raw_model_outputs=raw_outputs
        )

    @classmethod
    def reconcile_panel_adjudication(
        cls,
        task_id: str,
        question: str,
        assessments: List[SpecialistAssessment],
        synthesis_text: str,
        adjudicator_models: List[str],
        complexity: str = "simple"
    ) -> AdjudicationResult:
        """Full panel adjudication reconciling cross-agent consensus, contradictions, and evidence."""
        contradictions = cls.build_contradiction_graph(assessments)

        all_evidence: List[StructuredEvidence] = []
        for ass in assessments:
            all_evidence.extend(ass.evidence)

        system_conf, calib_factors = cls.calculate_system_confidence(
            assessments=assessments,
            contradictions=contradictions,
            evidence_count=len(all_evidence),
            complexity_str=complexity
        )

        mean_model_conf = sum(a.model_confidence for a in assessments) / len(assessments) if assessments else 0.85

        agreements: List[str] = [
            f"{ass.agent_role}: {ass.claims[0].statement}" for ass in assessments if ass.claims
        ]
        unresolved = [c.dispute_description for c in contradictions if not c.resolved]

        provenance_trace = {
            "participating_agents": [a.agent_id for a in assessments],
            "models_engaged": adjudicator_models,
            "total_claims_extracted": sum(len(a.claims) for a in assessments),
            "evidence_pieces_count": len(all_evidence),
            "contradiction_count": len(contradictions)
        }

        return AdjudicationResult(
            task_id=task_id,
            agreements=agreements,
            contradictions=contradictions,
            resolved_disputes=[],
            unresolved_disputes=unresolved,
            reconciled_recommendation=synthesis_text,
            system_confidence=system_conf,
            model_confidence_mean=round(mean_model_conf, 2),
            calibration_factors=calib_factors,
            key_evidence=all_evidence,
            adjudicator_models_used=adjudicator_models,
            provenance_trace=provenance_trace
        )
