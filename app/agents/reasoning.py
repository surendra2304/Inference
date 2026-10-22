"""First-Class Structured Reasoning Domain Models: Claims, Evidence, Contradictions, and Adjudication."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from app.utils.confidence import UNVERIFIED_MODEL_CONFIDENCE
from app.utils.ids import generate_id


class EvidenceRelation(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    DISPUTED = "disputed"
    REFUTED = "refuted"


class StructuredEvidence(BaseModel):
    """Traceable, verifiable empirical evidence piece with strict provenance."""
    evidence_id: str = Field(default_factory=lambda: generate_id("evi"))
    claim_id: Optional[str] = None
    source: str = Field(description="Document source, URL, paper title, or telemetry metric")
    source_type: str = Field(default="specialist_analysis", description="specialist_analysis, telemetry, documentation, empirical_test")
    source_locator: Optional[str] = Field(default=None, description="Section, line range, or metric path")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    excerpt: str = Field(description="Direct verbatim text span or extracted quantitative data")
    reliability_score: float = Field(default=UNVERIFIED_MODEL_CONFIDENCE, ge=0.0, le=1.0)
    relation: EvidenceRelation = EvidenceRelation.SUPPORTS
    agent_origin: str = Field(description="Agent ID that extracted this evidence")
    model_origin: str = Field(description="Model/Provider that generated this evidence")
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED


class AtomicClaim(BaseModel):
    """An individual atomic technical assertion made by an agent."""
    claim_id: str = Field(default_factory=lambda: generate_id("clm"))
    statement: str = Field(description="Atomic declarative factual or strategic proposition")
    category: str = Field(default="technical", description="technical, architectural, risk, performance, security")
    confidence: float = Field(default=UNVERIFIED_MODEL_CONFIDENCE, ge=0.0, le=1.0)
    agent_id: str
    model_id: str
    evidence_ids: List[str] = Field(default_factory=list)
    supporting_evidence: List[StructuredEvidence] = Field(default_factory=list)


class Contradiction(BaseModel):
    """Identified conflict or dispute between two opposing claims/specialists."""
    dispute_id: str = Field(default_factory=lambda: generate_id("dsp"))
    claim_a: AtomicClaim
    claim_b: AtomicClaim
    severity: str = Field(default="moderate", description="minor, moderate, severe")
    dispute_description: str
    resolved: bool = False
    resolution_rationale: Optional[str] = None
    winning_claim_id: Optional[str] = None


class SpecialistAssessment(BaseModel):
    """Normalized structured assessment from a specialist (reconciling parallel models)."""
    agent_id: str
    agent_role: str
    summary: str
    claims: List[AtomicClaim] = Field(default_factory=list)
    evidence: List[StructuredEvidence] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    trade_offs: List[str] = Field(default_factory=list)
    model_confidence: float = Field(default=UNVERIFIED_MODEL_CONFIDENCE, ge=0.0, le=1.0, description="Self-reported model confidence (unverified prior when absent)")
    raw_model_outputs: dict[str, str] = Field(default_factory=dict, description="Provenance audit trail of candidate model outputs")


class AdjudicationResult(BaseModel):
    """Comprehensive cross-agent reconciliation outcome produced by the Adjudicator."""
    adjudication_id: str = Field(default_factory=lambda: generate_id("adj"))
    task_id: str
    agreements: List[str] = Field(default_factory=list, description="Points of technical consensus across specialists")
    contradictions: List[Contradiction] = Field(default_factory=list, description="Identified technical disputes")
    resolved_disputes: List[str] = Field(default_factory=list)
    unresolved_disputes: List[str] = Field(default_factory=list)
    reconciled_recommendation: str
    #: Heuristic: a mean of model self-reports adjusted by fixed penalties and bonuses. It is NOT
    #: empirically calibrated (no outcome data fits those constants); see calibration_factors.
    system_confidence: float = Field(ge=0.0, le=1.0, description="Heuristic system confidence from self-reports (uncalibrated)")
    model_confidence_mean: float = Field(ge=0.0, le=1.0, description="Mean of self-reported agent confidences")
    calibration_factors: dict[str, Any] = Field(default_factory=dict)
    key_evidence: List[StructuredEvidence] = Field(default_factory=list)
    adjudicator_models_used: List[str] = Field(default_factory=list)
    provenance_trace: dict[str, Any] = Field(default_factory=dict)
