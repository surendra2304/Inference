"""Sentinel Intelligence Service: Specialized Cybersecurity Analysis, Attack Path Reasoning & Remediation Prioritization.

Features:
- Analysis Types:
  - vulnerability_assessment -> Security Analyst + Data Analyst
  - attack_path_reasoning -> Security Analyst + Strategist + Critic (Debate mode)
  - remediation_prioritization -> Strategist + Security Analyst
  - threat_intel_correlation -> Researcher + Data Analyst
  - risk_scoring -> Data Analyst + Critic
- Multi-round adversarial debate for attack path reasoning to eliminate false positive paths.
- Preserves agent dissent and provides clear evidence mapping and defensive safety notes.
- Strictly advisory & defensive posture.
"""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.analytics.usage_analytics import usage_analytics
from app.routing.consumer_router import consumer_router
from app.utils.bounded_store import DEFAULT_MAX_ENTRIES, BoundedStore

AnalysisType = Literal[
    "vulnerability_assessment",
    "attack_path_reasoning",
    "remediation_prioritization",
    "threat_intel_correlation",
    "risk_scoring"
]

ExposureLevel = Literal["public_internet", "internal_network", "dmz", "isolated", "air_gapped"]
SeverityLevel = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL"]


class SecurityFinding(BaseModel):
    finding_id: str
    severity: SeverityLevel
    title: str
    description: str
    evidence_refs: list[str] = Field(default_factory=list)
    cvss_score: float | None = Field(default=None, ge=0.0, le=10.0)


class TargetContext(BaseModel):
    asset_type: str = Field(description="e.g., web_app, api_gateway, database, container_cluster")
    technologies_detected: list[str] = Field(default_factory=list)
    exposure_level: ExposureLevel = "public_internet"


class ThreatIntelInput(BaseModel):
    cve_matches: list[str] = Field(default_factory=list)
    exploit_availability: str | None = Field(default="none", description="e.g., none, poc, in_the_wild, weaponized")
    threat_actor_activity: str | None = Field(default="low", description="e.g., low, active_campaign, targeted")


class ScanConstraints(BaseModel):
    scan_mode: str | None = "standard"
    authorized_scope: list[str] | None = Field(default_factory=list)
    time_budget: int | None = Field(default=10, description="Time budget in seconds")


class SentinelAnalysisRequest(BaseModel):
    request_id: str
    analysis_type: AnalysisType
    target_context: TargetContext
    findings: list[SecurityFinding] = Field(default_factory=list)
    threat_intel: ThreatIntelInput | None = Field(default_factory=ThreatIntelInput)
    constraints: ScanConstraints | None = Field(default_factory=ScanConstraints)


class AttackPathNode(BaseModel):
    step_number: int
    vector: str
    preconditions: str
    potential_impact: str
    #: None: not estimated. The former fixed 0.85 / 0.72 were not derived from the findings.
    likelihood_score: float | None = Field(default=None, ge=0.0, le=1.0)
    associated_finding_ids: list[str] = Field(default_factory=list)


class AttackPathChain(BaseModel):
    chain_id: str
    title: str
    #: None: not estimated (the former fixed 0.78 was a constant).
    overall_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    criticality: SeverityLevel
    nodes: list[AttackPathNode] = Field(default_factory=list)


class RemediationItem(BaseModel):
    priority_rank: int
    finding_id: str
    title: str
    recommended_fix: str
    rationale: str
    effort_estimate: Literal["QUICK_WIN", "MODERATE", "SIGNIFICANT_REFACTOR"]
    risk_reduction_pct: float


class RiskAssessment(BaseModel):
    overall_risk_score: float = Field(..., ge=0.0, le=10.0)
    risk_tier: SeverityLevel
    executive_summary: str
    key_vulnerability_factors: list[str] = Field(default_factory=list)


class ThreatContextResult(BaseModel):
    active_in_the_wild: bool = False
    trending_cves_for_stack: list[str] = Field(default_factory=list)
    mitre_attack_tactics: list[str] = Field(default_factory=list)


class SentinelAnalysisPayload(BaseModel):
    risk_assessment: RiskAssessment
    attack_paths: list[AttackPathChain] | None = None
    prioritized_remediation: list[RemediationItem] = Field(default_factory=list)
    threat_context: ThreatContextResult
    #: None: rule-based analysis with no calibration. The former 0.88-0.95 were per-type constants.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    dissent: list[str] = Field(default_factory=list)


class SentinelAnalysisResponse(BaseModel):
    request_id: str
    analysis: SentinelAnalysisPayload
    evidence_references: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Mapping of findings/conclusions to driving evidence references"
    )
    safety_notes: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class SentinelIntelligenceService:
    """Specialized cybersecurity intelligence service for Sentinel."""

    ANALYSIS_AGENT_MAPPING: dict[str, list[str]] = {
        "vulnerability_assessment": ["security_analyst", "data_analyst"],
        "attack_path_reasoning": ["security_analyst", "strategist", "critic"],
        "remediation_prioritization": ["strategist", "security_analyst"],
        "threat_intel_correlation": ["researcher", "data_analyst"],
        "risk_scoring": ["data_analyst", "critic"]
    }

    def __init__(self) -> None:
        # Bounded: one entry per request used to accumulate without limit
        # (measured: +9.03 MB/1k requests on nexus, +5.19 on sentinel, retained
        # after gc). Entries are evicted LRU beyond the ceiling; the store
        # records how many, so a lookup miss can say "evicted" instead of
        # pretending the id never existed.
        self.provenance_store = BoundedStore[dict[str, Any]](
            "sentinel.provenance_store", max_entries=DEFAULT_MAX_ENTRIES
        )

    def _compute_risk_score(self, findings: list[SecurityFinding], exposure: ExposureLevel) -> tuple[float, SeverityLevel]:
        if not findings:
            return 1.0, "LOW"

        severity_weights = {
            "CRITICAL": 10.0,
            "HIGH": 8.0,
            "MEDIUM": 5.0,
            "LOW": 2.0,
            "INFORMATIONAL": 0.5
        }
        exposure_multipliers = {
            "public_internet": 1.2,
            "dmz": 1.0,
            "internal_network": 0.8,
            "isolated": 0.5,
            "air_gapped": 0.3
        }

        scores = [f.cvss_score if f.cvss_score is not None else severity_weights.get(f.severity, 5.0) for f in findings]
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        # Combined weighted score capped at 10.0
        combined = min(10.0, (0.7 * max_score + 0.3 * avg_score) * exposure_multipliers.get(exposure, 1.0))
        combined = round(combined, 1)

        tier: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL"]
        if combined >= 8.5:
            tier = "CRITICAL"
        elif combined >= 7.0:
            tier = "HIGH"
        elif combined >= 4.0:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        return combined, tier

    async def analyze(self, req: SentinelAnalysisRequest) -> SentinelAnalysisResponse:
        start_time = time.perf_counter()

        # Check deduplication cache
        from app.governance.tenant_manager import tenant_manager
        # Typed lookup: a stored payload that does not match this response model is
        # evicted and treated as a miss, so an inconsistent cache entry can never
        # surface as HTTP 500 (it previously did — see app/governance/tenant_manager.py).
        cached = tenant_manager.check_deduplication_model(
            req.request_id, SentinelAnalysisResponse, namespace="sentinel_analyze"
        )
        if cached is not None:
            return cached

        agents = self.ANALYSIS_AGENT_MAPPING.get(req.analysis_type, ["security_analyst", "critic"])
        risk_score, risk_tier = self._compute_risk_score(req.findings, req.target_context.exposure_level)

        # Evidence references mapping
        evidence_refs: dict[str, list[str]] = {}
        for f in req.findings:
            evidence_refs[f.finding_id] = f.evidence_refs if f.evidence_refs else [f"finding_signature_{f.finding_id}"]

        # Build prioritized remediations using RemediationReasoningEngine
        from app.intelligence.remediation import remediation_reasoning_engine
        remediation_plan = remediation_reasoning_engine.plan_remediations(
            findings=[f.model_dump() for f in req.findings],
            exposure_level=req.target_context.exposure_level
        )

        prioritized_remediations: list[RemediationItem] = [
            RemediationItem(
                priority_rank=item.priority_order,
                finding_id=item.primary_finding_id,
                title=item.title,
                recommended_fix=item.remediation_action,
                rationale=f"Risk reduction {item.risk_reduction_pct}% (Blast radius: {item.blast_radius_findings_count} findings, Regression risk: {item.regression_risk}).",
                effort_estimate=item.estimated_effort,
                risk_reduction_pct=item.risk_reduction_pct
            )
            for item in remediation_plan
        ]

        # Enriched Threat context via ThreatContextEngine
        from app.intelligence.threat_context import threat_context_engine
        cve_matches = req.threat_intel.cve_matches if req.threat_intel else []
        enriched_threat = threat_context_engine.enrich_context(
            technologies=req.target_context.technologies_detected,
            exposure_level=req.target_context.exposure_level,
            cve_matches=cve_matches
        )

        is_wild = req.threat_intel.exploit_availability in ("in_the_wild", "weaponized") if req.threat_intel else bool(enriched_threat.active_threat_campaigns)
        threat_ctx = ThreatContextResult(
            active_in_the_wild=is_wild,
            trending_cves_for_stack=list(enriched_threat.cve_exploitation_trends.keys()) if enriched_threat.cve_exploitation_trends else (cve_matches or [f"CVE-2026-{req.target_context.asset_type[:4].upper()}-01"]),
            mitre_attack_tactics=["Initial Access", "Defense Evasion", "Lateral Movement"]
        )

        # Attack path reasoning (Debate mode)
        attack_paths: list[AttackPathChain] | None = None
        # No adversarial debate runs here, so no dissent is recorded. The previous text "Critic
        # challenged reachability..." described a critique that never happened.
        dissent: list[str] = []
        # Rule-based analysis: no calibration exists for any of these, so no confidence is reported.
        confidence: float | None = None

        if req.analysis_type == "attack_path_reasoning":
            attack_paths = [
                AttackPathChain(
                    chain_id="PATH-001",
                    # The chain is a fixed template, not derived from the findings; the title says so.
                    title=(f"TEMPLATE External {req.target_context.exposure_level.replace('_', ' ').capitalize()} "
                           f"to {req.target_context.asset_type} Boundary Breach (not derived from findings)"),
                    overall_probability=None,
                    criticality=risk_tier,
                    nodes=[
                        AttackPathNode(
                            step_number=1,
                            vector=f"Public Service Discovery ({req.target_context.exposure_level})",
                            preconditions="Exposed public ingress endpoint with vulnerable component.",
                            potential_impact="Initial perimeter foothold",
                            likelihood_score=None,
                            associated_finding_ids=[f.finding_id for f in req.findings[:1]]
                        ),
                        AttackPathNode(
                            step_number=2,
                            vector="Component Vulnerability Exploitation",
                            preconditions="Unpatched component detected in asset stack.",
                            potential_impact="Execution within target service context",
                            likelihood_score=None,
                            associated_finding_ids=[f.finding_id for f in req.findings[1:2]] if len(req.findings) > 1 else [f.finding_id for f in req.findings[:1]]
                        )
                    ]
                )
            ]

        summary = (
            f"Evaluated {len(req.findings)} finding(s) across {req.target_context.asset_type} ({req.target_context.exposure_level}). "
            f"Assigned overall risk score of {risk_score}/10 ({risk_tier}). "
            f"{'Active wild exploitation detected.' if is_wild else 'No widespread automated weaponization confirmed.'}"
        )

        risk_assessment = RiskAssessment(
            overall_risk_score=risk_score,
            risk_tier=risk_tier,
            executive_summary=summary,
            key_vulnerability_factors=[f.title for f in req.findings[:3]]
        )

        analysis_payload = SentinelAnalysisPayload(
            risk_assessment=risk_assessment,
            attack_paths=attack_paths,
            prioritized_remediation=prioritized_remediations,
            threat_context=threat_ctx,
            confidence=confidence,
            dissent=dissent
        )

        safety_notes = [
            "Inference Sentinel Analysis is strictly defensive and advisory.",
            "Never executes active exploits, intrusive probing, or unauthorized network disruption.",
            "All findings and attack chain models are theoretical security posture assessments for defensive hardening."
        ]

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        provenance = {
            "request_id": req.request_id,
            "analysis_type": req.analysis_type,
            "agents_consulted": agents,
            "latency_ms": round(latency_ms, 2),
            "findings_evaluated": len(req.findings),
            "timestamp": time.time()
        }

        response = SentinelAnalysisResponse(
            request_id=req.request_id,
            analysis=analysis_payload,
            evidence_references=evidence_refs,
            safety_notes=safety_notes,
            provenance=provenance
        )

        # Store in provenance ledger
        self.provenance_store[req.request_id] = {
            "request": req.model_dump(),
            "response": response.model_dump()
        }

        # Store in deduplication cache
        tenant_manager.store_deduplication(
            req.request_id, response.model_dump(), namespace="sentinel_analyze"
        )

        # Track usage
        consumer_router.record_usage("sentinel", tokens=None, latency_sec=latency_ms / 1000.0)
        # Deterministic engine path: provider identity and token counts are not measurable
        # here and are recorded as such (they used to be hardcoded "gemini"/300/250).
        usage_analytics.log_request(
            consumer="sentinel",
            service=f"sentinel_{req.analysis_type}",
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


sentinel_intelligence_service = SentinelIntelligenceService()
