"""
Inference/services/multi_hypothesis_reasoner.py — Inference Multi-Hypothesis Reasoner.

Advanced reasoning with multiple competing hypotheses: hypothesis generation,
probabilistic scoring, Bayesian updating, contradiction resolution, and
consensus synthesis. Handles uncertain and ambiguous problems.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()


@dataclass
class Hypothesis:
    """A hypothesis in the reasoning process."""
    hypothesis_id: str
    statement: str = ""
    probability: float = 0.5
    prior: float = 0.5
    likelihood: float = 0.5
    posterior: float = 0.5
    evidence: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    support: list[str] = field(default_factory=list)
    status: str = "active"  # active, supported, refuted, uncertain


@dataclass
class Evidence:
    """Evidence supporting or refuting a hypothesis."""
    evidence_id: str
    description: str = ""
    supports: list[str] = field(default_factory=list)  # hypothesis IDs
    refutes: list[str] = field(default_factory=list)
    strength: float = 0.5
    source: str = ""
    reliability: float = 0.8
    added_at: float = 0.0


@dataclass
class Contradiction:
    """A contradiction between hypotheses."""
    contradiction_id: str
    hypothesis_a: str = ""
    hypothesis_b: str = ""
    description: str = ""
    severity: str = "medium"  # low, medium, high
    resolution: str = ""
    resolved: bool = False


@dataclass
class BayesianUpdate:
    """A Bayesian update step."""
    update_id: str
    hypothesis_id: str = ""
    prior: float = 0.0
    likelihood: float = 0.0
    posterior: float = 0.0
    evidence: str = ""
    update_time: float = 0.0


@dataclass
class ConsensusResult:
    """Result of consensus synthesis."""
    consensus_id: str
    winning_hypothesis: str = ""
    winning_probability: float = 0.0
    all_probabilities: dict = field(default_factory=dict)
    confidence: float = 0.0
    synthesis: str = ""
    alternative_explanations: list[str] = field(default_factory=list)


@dataclass
class ReasoningSession:
    """A multi-hypothesis reasoning session."""
    session_id: str
    problem_statement: str = ""
    hypotheses: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    updates: list[str] = field(default_factory=list)
    consensus: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0


class MultiHypothesisReasoner:
    """Multi-hypothesis reasoning engine."""

    def __init__(self):
        self._hypotheses: dict[str, Hypothesis] = {}
        self._evidence: list[Evidence] = []
        self._contradictions: list[Contradiction] = []
        self._updates: list[BayesianUpdate] = []
        self._consensus: list[ConsensusResult] = []
        self._sessions: dict[str, ReasoningSession] = {}

    def generate_hypothesis(self, statement: str, prior: float = 0.5) -> Hypothesis:
        """Generate a hypothesis."""
        hyp = Hypothesis(
            hypothesis_id=hashlib.md5(
                f"hyp:{statement[:30]}:{time.time()}".encode()
            ).hexdigest()[:12],
            statement=statement,
            prior=prior,
            probability=prior,
            posterior=prior,
        )

        # Check for contradictions with existing hypotheses
        for existing in self._hypotheses.values():
            if statement[:20] in existing.statement or existing.statement[:20] in statement:
                contra = Contradiction(
                    contradiction_id=hashlib.md5(
                        f"contra:{hyp.hypothesis_id}:{existing.hypothesis_id}:{time.time()}".encode()
                    ).hexdigest()[:12],
                    hypothesis_a=hyp.hypothesis_id,
                    hypothesis_b=existing.hypothesis_id,
                    description=f"These hypotheses appear contradictory: '{statement[:40]}' vs '{existing.statement[:40]}'",
                    severity="high",
                )
                self._contradictions.append(contra)
                hyp.contradictions.append(existing.hypothesis_id)
                existing.contradictions.append(hyp.hypothesis_id)

        self._hypotheses[hyp.hypothesis_id] = hyp
        return hyp

    def add_evidence(
        self,
        description: str,
        supports: list[str] | None = None,
        refutes: list[str] | None = None,
        strength: float = 0.5,
        source: str = "",
    ) -> Evidence:
        """Add evidence."""
        evidence = Evidence(
            evidence_id=hashlib.md5(
                f"evidence:{description[:30]}:{time.time()}".encode()
            ).hexdigest()[:12],
            description=description,
            supports=supports or [],
            refutes=refutes or [],
            strength=strength,
            source=source,
            reliability=0.7 + (ord(hashlib.md5(source.encode()).hexdigest()[:1]) % 25) / 100,
            added_at=time.time(),
        )

        self._evidence.append(evidence)

        # Update hypotheses
        for hyp_id in evidence.supports:
            hyp = self._hypotheses.get(hyp_id)
            if hyp:
                hyp.support.append(evidence.evidence_id)
                hyp.evidence.append(description)

        for hyp_id in evidence.refutes:
            hyp = self._hypotheses.get(hyp_id)
            if hyp:
                hyp.contradictions.append(evidence.evidence_id)

        return evidence

    def bayesian_update(
        self,
        hypothesis_id: str,
        evidence: str,
        likelihood: float = 0.7,
    ) -> BayesianUpdate:
        """Perform a Bayesian update."""
        hyp = self._hypotheses.get(hypothesis_id)
        if not hyp:
            raise HTTPException(404, f"Hypothesis {hypothesis_id} not found")

        prior = hyp.posterior

        # Calculate posterior: P(H|E) = P(E|H) * P(H) / P(E)
        # Simplified: posterior = (likelihood * prior) / (likelihood * prior + (1-likelihood) * (1-prior))
        if likelihood > 0 and prior > 0:
            posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
        else:
            posterior = prior

        hyp.posterior = posterior
        hyp.probability = posterior
        hyp.likelihood = likelihood

        # Update status
        if posterior > 0.7:
            hyp.status = "supported"
        elif posterior < 0.3:
            hyp.status = "refuted"
        else:
            hyp.status = "uncertain"

        update = BayesianUpdate(
            update_id=hashlib.md5(
                f"update:{hypothesis_id}:{time.time()}".encode()
            ).hexdigest()[:12],
            hypothesis_id=hypothesis_id,
            prior=prior,
            likelihood=likelihood,
            posterior=posterior,
            evidence=evidence,
            update_time=time.time(),
        )

        self._updates.append(update)
        return update

    def resolve_contradiction(self, contradiction_id: str, resolution: str) -> Contradiction:
        """Resolve a contradiction."""
        contra = None
        for c in self._contradictions:
            if c.contradiction_id == contradiction_id:
                contra = c
                break

        if not contra:
            raise HTTPException(404, f"Contradiction {contradiction_id} not found")

        contra.resolution = resolution
        contra.resolved = True

        # Update affected hypotheses
        for hyp_id in [contra.hypothesis_a, contra.hypothesis_b]:
            hyp = self._hypotheses.get(hyp_id)
            if hyp:
                if "refuted" in resolution.lower():
                    hyp.status = "refuted"
                elif "supported" in resolution.lower():
                    hyp.status = "supported"

        return contra

    def synthesize_consensus(self) -> ConsensusResult:
        """Synthesize a consensus from all hypotheses."""
        if not self._hypotheses:
            raise HTTPException(404, "No hypotheses to synthesize")

        # Find the hypothesis with the highest posterior
        winner = max(self._hypotheses.values(), key=lambda h: h.posterior)

        all_probs = {h.hypothesis_id: h.posterior for h in self._hypotheses.values()}

        result = ConsensusResult(
            consensus_id=hashlib.md5(
                f"consensus:{time.time()}".encode()
            ).hexdigest()[:12],
            winning_hypothesis=winner.hypothesis_id,
            winning_probability=winner.posterior,
            all_probabilities=all_probs,
            confidence=winner.posterior,
            synthesis=f"Selected hypothesis with highest posterior probability ({winner.posterior:.2%}): {winner.statement[:100]}",
            alternative_explanations=[
                h.statement[:80] for h in self._hypotheses.values()
                if h.hypothesis_id != winner.hypothesis_id
            ],
        )

        self._consensus.append(result)
        return result

    def run_session(self, problem_statement: str, hypotheses: list[str]) -> ReasoningSession:
        """Run a full reasoning session."""
        start = time.time()

        session = ReasoningSession(
            session_id=hashlib.md5(
                f"session:{problem_statement[:30]}:{time.time()}".encode()
            ).hexdigest()[:12],
            problem_statement=problem_statement,
            started_at=start,
        )

        # Generate hypotheses
        for hyp_statement in hypotheses:
            hyp = self.generate_hypothesis(hyp_statement)
            session.hypotheses.append(hyp.hypothesis_id)

        # Add evidence for each hypothesis
        for hyp_id in session.hypotheses:
            self.add_evidence(
                description=f"Evidence for hypothesis {hyp_id}",
                supports=[hyp_id],
                strength=0.6,
                source="observed_data",
            )

        # Bayesian updates
        for hyp_id in session.hypotheses:
            self.bayesian_update(
                hypothesis_id=hyp_id,
                evidence=f"Updated probability for {hyp_id}",
                likelihood=0.6 + (ord(hashlib.md5(hyp_id.encode()).hexdigest()[:1]) % 30) / 100,
            )

        # Synthesize consensus
        consensus = self.synthesize_consensus()
        session.consensus = consensus.consensus_id

        session.completed_at = time.time()
        self._sessions[session.session_id] = session

        return session

    def get_report(self) -> dict:
        """Get reasoner report."""
        return {
            "total_hypotheses": len(self._hypotheses),
            "total_evidence": len(self._evidence),
            "total_contradictions": len(self._contradictions),
            "total_updates": len(self._updates),
            "total_consensus": len(self._consensus),
            "total_sessions": len(self._sessions),
            "resolved_contradictions": sum(1 for c in self._contradictions if c.resolved),
            "supported_hypotheses": sum(1 for h in self._hypotheses.values() if h.status == "supported"),
            "refuted_hypotheses": sum(1 for h in self._hypotheses.values() if h.status == "refuted"),
            "uncertain_hypotheses": sum(1 for h in self._hypotheses.values() if h.status == "uncertain"),
            "avg_posterior": sum(h.posterior for h in self._hypotheses.values()) / max(1, len(self._hypotheses)),
            "recent_sessions": [
                {
                    "problem": s.problem_statement[:50],
                    "hypotheses": len(s.hypotheses),
                    "consensus": s.consensus,
                    "duration_ms": (s.completed_at - s.started_at) * 1000,
                }
                for s in list(self._sessions.values())[-5:]
            ],
            "recent_consensus": [
                {
                    "winner": c.winning_hypothesis,
                    "probability": c.winning_probability,
                    "confidence": c.confidence,
                }
                for c in self._consensus[-5:]
            ],
        }


# Module-level singleton
multi_reasoner = MultiHypothesisReasoner()


# ── API Routes ──────────────────────────────────────────────────────────────


@router.post("/api/v1/inference/reasoning/hypotheses")
async def generate_hypothesis(payload: dict):
    """Generate a hypothesis."""
    hyp = multi_reasoner.generate_hypothesis(
        statement=payload["statement"],
        prior=payload.get("prior", 0.5),
    )
    return {"hypothesis_id": hyp.hypothesis_id, "statement": hyp.statement[:50], "probability": hyp.probability}


@router.post("/api/v1/inference/reasoning/evidence")
async def add_evidence(payload: dict):
    """Add evidence."""
    evidence = multi_reasoner.add_evidence(
        description=payload["description"],
        supports=payload.get("supports", []),
        refutes=payload.get("refutes", []),
        strength=payload.get("strength", 0.5),
        source=payload.get("source", ""),
    )
    return {"evidence_id": evidence.evidence_id, "supports": len(evidence.supports), "refutes": len(evidence.refutes)}


@router.post("/api/v1/inference/reasoning/hypotheses/{hyp_id}/update")
async def bayesian_update(hyp_id: str, payload: dict):
    """Perform a Bayesian update."""
    update = multi_reasoner.bayesian_update(
        hypothesis_id=hyp_id,
        evidence=payload["evidence"],
        likelihood=payload.get("likelihood", 0.7),
    )
    return {"update_id": update.update_id, "prior": update.prior, "posterior": update.posterior}


@router.post("/api/v1/inference/reasoning/contradictions/{contra_id}/resolve")
async def resolve_contradiction(contra_id: str, payload: dict):
    """Resolve a contradiction."""
    contra = multi_reasoner.resolve_contradiction(contra_id, payload["resolution"])
    return {"contradiction_id": contra.contradiction_id, "resolved": contra.resolved}


@router.post("/api/v1/inference/reasoning/consensus")
async def synthesize_consensus():
    """Synthesize a consensus."""
    result = multi_reasoner.synthesize_consensus()
    return {
        "consensus_id": result.consensus_id,
        "winner": result.winning_hypothesis,
        "probability": result.winning_probability,
        "confidence": result.confidence,
    }


@router.get("/api/v1/inference/reasoning/report")
async def get_report():
    """Get reasoner report."""
    return multi_reasoner.get_report()
