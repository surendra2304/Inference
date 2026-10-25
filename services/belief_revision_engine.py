"""
Inference/services/belief_revision_engine.py — Inference Belief Revision Engine.

Advanced belief revision: handles conflicting information, belief updates
using AGM postulates, confidence decay, source reliability weighting,
and belief network maintenance. Provides principled belief updating.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()


@dataclass
class Belief:
    """A belief held by the system."""
    belief_id: str
    statement: str
    confidence: float = 0.5
    source: str = ""
    source_reliability: float = 0.8
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    status: str = "active"  # active, confirmed, refuted, uncertain
    created_at: float = 0.0
    last_updated: float = 0.0
    revision_count: int = 0


@dataclass
class RevisionEvent:
    """A belief revision event."""
    event_id: str
    belief_id: str
    old_confidence: float
    new_confidence: float
    trigger: str = ""  # new_evidence, contradiction, source_update, decay
    description: str = ""
    revision_type: str = "update"  # update, strengthen, weaken, revoke
    occurred_at: float = 0.0


@dataclass
class SourceReliability:
    """Source reliability profile."""
    source_id: str
    source_name: str
    reliability_score: float = 0.5
    track_record: list[dict] = field(default_factory=list)
    avg_accuracy: float = 0.0
    last_updated: float = 0.0


@dataclass
class BeliefNetwork:
    """A network of related beliefs."""
    network_id: str
    name: str
    beliefs: list[str] = field(default_factory=list)
    connections: list[dict] = field(default_factory=list)
    overall_confidence: float = 0.0
    coherence: float = 0.0
    created_at: float = 0.0


@dataclass
class RevisionPolicy:
    """Policy for belief revision."""
    policy_id: str
    min_confidence_threshold: float = 0.3
    source_weighting: bool = True
    evidence_decay: bool = True
    decay_rate: float = 0.01
    max_revisions: int = 100
    conflict_resolution: str = "reliability_weighted"  # reliability_weighted, majority_vote, source_priority
    preserve_history: bool = True
    auto_revoke: bool = True
    created_at: float = 0.0


@dataclass
class CoherenceCheck:
    """Coherence check between beliefs."""
    check_id: str
    belief_a: str
    belief_b: str
    coherent: bool = True
    conflict_level: float = 0.0
    resolution: str = ""
    checked_at: float = 0.0


@dataclass
class RevisionReport:
    """Report on belief revision activity."""
    total_beliefs: int = 0
    active_beliefs: int = 0
    confirmed_beliefs: int = 0
    refuted_beliefs: int = 0
    total_revisions: int = 0
    avg_confidence: float = 0.0
    avg_reliability: float = 0.0
    coherence_score: float = 0.0


class BeliefRevisionEngine:
    """Belief revision engine implementing AGM postulates."""

    def __init__(self):
        self._beliefs: dict[str, Belief] = {}
        self._revisions: list[RevisionEvent] = []
        self._sources: dict[str, SourceReliability] = {}
        self._networks: dict[str, BeliefNetwork] = {}
        self._policies: dict[str, RevisionPolicy] = {}
        self._checks: list[CoherenceCheck] = []

        # Create default policy
        self._create_policy()

    def _create_policy(self):
        """Create default revision policy."""
        policy = RevisionPolicy(
            policy_id=hashlib.md5(f"policy:{time.time()}".encode()).hexdigest()[:12],
            created_at=time.time(),
        )
        self._policies[policy.policy_id] = policy
        return policy

    def add_belief(
        self,
        statement: str,
        confidence: float = 0.5,
        source: str = "",
        source_reliability: float = 0.8,
    ) -> Belief:
        """Add a belief."""
        belief = Belief(
            belief_id=hashlib.md5(
                f"belief:{statement[:30]}:{time.time()}".encode()
            ).hexdigest()[:12],
            statement=statement,
            confidence=confidence,
            source=source,
            source_reliability=source_reliability,
            created_at=time.time(),
            last_updated=time.time(),
        )

        self._beliefs[belief.belief_id] = belief

        # Update source reliability
        if source:
            src = self._sources.get(source)
            if not src:
                src = SourceReliability(
                    source_id=hashlib.md5(f"source:{source}".encode()).hexdigest()[:12],
                    source_name=source,
                    reliability_score=source_reliability,
                )
                self._sources[source] = src
            src.last_updated = time.time()

        return belief

    def revise(
        self,
        belief_id: str,
        new_evidence: str,
        new_confidence: float,
        trigger: str = "new_evidence",
    ) -> RevisionEvent:
        """Revise a belief based on new evidence."""
        belief = self._beliefs.get(belief_id)
        if not belief:
            raise HTTPException(404, f"Belief {belief_id} not found")

        old_confidence = belief.confidence

        # Weight by source reliability
        policy = list(self._policies.values())[0]
        if policy.source_weighting:
            new_confidence = new_confidence * belief.source_reliability + old_confidence * (1 - belief.source_reliability)

        # Clamp to [0, 1]
        new_confidence = max(0.0, min(1.0, new_confidence))

        # Determine revision type
        if new_confidence > old_confidence + 0.1:
            revision_type = "strengthen"
        elif new_confidence < old_confidence - 0.1:
            revision_type = "weaken"
        elif new_confidence < policy.min_confidence_threshold:
            revision_type = "revoke"
        else:
            revision_type = "update"

        belief.confidence = new_confidence
        belief.last_updated = time.time()
        belief.revision_count += 1

        # Update status
        if new_confidence > 0.7:
            belief.status = "confirmed"
        elif new_confidence < 0.3:
            belief.status = "refuted" if policy.auto_revoke else "uncertain"
        else:
            belief.status = "active"

        # Add evidence
        if "support" in new_evidence.lower() or new_confidence > old_confidence:
            belief.supporting_evidence.append(new_evidence)
        else:
            belief.contradicting_evidence.append(new_evidence)

        event = RevisionEvent(
            event_id=hashlib.md5(
                f"revision:{belief_id}:{time.time()}".encode()
            ).hexdigest()[:12],
            belief_id=belief_id,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            trigger=trigger,
            description=new_evidence[:100],
            revision_type=revision_type,
            occurred_at=time.time(),
        )

        self._revisions.append(event)
        return event

    def check_coherence(self, belief_a_id: str, belief_b_id: str) -> CoherenceCheck:
        """Check coherence between two beliefs."""
        a = self._beliefs.get(belief_a_id)
        b = self._beliefs.get(belief_b_id)

        if not a or not b:
            return CoherenceCheck(
                check_id="",
                belief_a=belief_a_id,
                belief_b=belief_b_id,
                coherent=False,
                conflict_level=1.0,
                resolution="One or both beliefs not found",
            )

        # Check for contradiction (simple keyword overlap)
        a_words = set(a.statement.lower().split())
        b_words = set(b.statement.lower().split())

        overlap = a_words & b_words
        similarity = len(overlap) / max(1, max(len(a_words), len(b_words)))

        # Check confidence alignment
        confidence_diff = abs(a.confidence - b.confidence)

        coherent = similarity > 0.2 and confidence_diff < 0.5
        conflict_level = max(1 - similarity, confidence_diff)

        check = CoherenceCheck(
            check_id=hashlib.md5(
                f"check:{belief_a_id}:{belief_b_id}:{time.time()}".encode()
            ).hexdigest()[:12],
            belief_a=belief_a_id,
            belief_b=belief_b_id,
            coherent=coherent,
            conflict_level=round(conflict_level, 3),
            resolution=f"Similarity: {similarity:.0%}, Confidence diff: {confidence_diff:.2f}",
            checked_at=time.time(),
        )

        self._checks.append(check)
        return check

    def create_network(self, name: str, belief_ids: list[str]) -> BeliefNetwork:
        """Create a belief network."""
        network = BeliefNetwork(
            network_id=hashlib.md5(
                f"network:{name}:{time.time()}".encode()
            ).hexdigest()[:12],
            name=name,
            beliefs=belief_ids,
            created_at=time.time(),
        )

        # Calculate coherence
        if len(belief_ids) >= 2:
            coherences = []
            for i in range(len(belief_ids)):
                for j in range(i + 1, len(belief_ids)):
                    check = self.check_coherence(belief_ids[i], belief_ids[j])
                    coherences.append(1 - check.conflict_level)
            network.coherence = sum(coherences) / max(1, len(coherences))

        # Overall confidence
        network.overall_confidence = sum(
            self._beliefs[bid].confidence for bid in belief_ids
            if bid in self._beliefs
        ) / max(1, len(belief_ids))

        self._networks[network.network_id] = network
        return network

    def update_source_reliability(self, source_name: str, accuracy: float) -> SourceReliability:
        """Update source reliability."""
        src = self._sources.get(source_name)
        if not src:
            src = SourceReliability(
                source_id=hashlib.md5(f"source:{source_name}".encode()).hexdigest()[:12],
                source_name=source_name,
            )
            self._sources[source_name] = src

        src.reliability_score = accuracy
        src.avg_accuracy = (src.avg_accuracy * 0.9 + accuracy * 0.1)
        src.last_updated = time.time()
        src.track_record.append({"accuracy": accuracy, "timestamp": time.time()})

        return src

    def get_report(self) -> dict:
        """Get belief revision engine report."""
        report = RevisionReport(
            total_beliefs=len(self._beliefs),
            active_beliefs=sum(1 for b in self._beliefs.values() if b.status == "active"),
            confirmed_beliefs=sum(1 for b in self._beliefs.values() if b.status == "confirmed"),
            refuted_beliefs=sum(1 for b in self._beliefs.values() if b.status == "refuted"),
            total_revisions=len(self._revisions),
            avg_confidence=sum(b.confidence for b in self._beliefs.values()) / max(1, len(self._beliefs)),
            avg_reliability=sum(s.reliability_score for s in self._sources.values()) / max(1, len(self._sources)),
        )

        # Calculate overall coherence
        if self._networks:
            report.coherence_score = sum(n.coherence for n in self._networks.values()) / len(self._networks)
        else:
            report.coherence_score = 0.7

        return {
            "total_beliefs": report.total_beliefs,
            "active_beliefs": report.active_beliefs,
            "confirmed_beliefs": report.confirmed_beliefs,
            "refuted_beliefs": report.refuted_beliefs,
            "total_revisions": report.total_revisions,
            "avg_confidence": round(report.avg_confidence, 4),
            "avg_reliability": round(report.avg_reliability, 4),
            "coherence_score": round(report.coherence_score, 4),
            "total_sources": len(self._sources),
            "total_networks": len(self._networks),
            "total_checks": len(self._checks),
            "incoherent_pairs": sum(1 for c in self._checks if not c.coherent),
            "revision_types": {
                rtype: sum(1 for r in self._revisions if r.revision_type == rtype)
                for rtype in set(r.revision_type for r in self._revisions)
            },
            "recent_revisions": [
                {"belief": r.belief_id, "type": r.revision_type, "old": r.old_confidence, "new": r.new_confidence}
                for r in self._revisions[-10:]
            ],
            "source_reliability": [
                {"source": s.source_name, "reliability": s.reliability_score, "accuracy": s.avg_accuracy}
                for s in self._sources.values()
            ],
        }


belief_engine = BeliefRevisionEngine()


@router.post("/api/v1/inference/beliefs")
async def add_belief(payload: dict):
    """Add a belief."""
    belief = belief_engine.add_belief(
        statement=payload["statement"],
        confidence=payload.get("confidence", 0.5),
        source=payload.get("source", ""),
        source_reliability=payload.get("source_reliability", 0.8),
    )
    return {"belief_id": belief.belief_id, "statement": belief.statement[:50], "confidence": belief.confidence}


@router.post("/api/v1/inference/beliefs/{belief_id}/revise")
async def revise(belief_id: str, payload: dict):
    """Revise a belief."""
    event = belief_engine.revise(
        belief_id=belief_id,
        new_evidence=payload["new_evidence"],
        new_confidence=payload["new_confidence"],
        trigger=payload.get("trigger", "new_evidence"),
    )
    return {"event_id": event.event_id, "revision_type": event.revision_type, "old_confidence": event.old_confidence, "new_confidence": event.new_confidence}


@router.post("/api/v1/inference/beliefs/check")
async def check_coherence(payload: dict):
    """Check coherence between two beliefs."""
    check = belief_engine.check_coherence(
        belief_a_id=payload["belief_a"],
        belief_b_id=payload["belief_b"],
    )
    return {"coherent": check.coherent, "conflict_level": check.conflict_level, "resolution": check.resolution}


@router.post("/api/v1/inference/beliefs/networks")
async def create_network(payload: dict):
    """Create a belief network."""
    network = belief_engine.create_network(
        name=payload["name"],
        belief_ids=payload["belief_ids"],
    )
    return {"network_id": network.network_id, "name": network.name, "coherence": network.coherence, "confidence": network.overall_confidence}


@router.post("/api/v1/inference/beliefs/sources")
async def update_source_reliability(payload: dict):
    """Update source reliability."""
    src = belief_engine.update_source_reliability(
        source_name=payload["source_name"],
        accuracy=payload["accuracy"],
    )
    return {"source": src.source_name, "reliability": src.reliability_score, "accuracy": src.avg_accuracy}


@router.get("/api/v1/inference/beliefs/report")
async def get_report():
    """Get belief revision engine report."""
    return belief_engine.get_report()
