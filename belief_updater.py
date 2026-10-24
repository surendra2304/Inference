"""
Inference/belief_updater.py — Inference Belief Updating Engine.

Extends Inference's capabilities with:
1. Bayesian Belief Updating: update beliefs based on new evidence
2. Belief Networks: maintain and update belief networks
3. Confidence Calibration: calibrate confidence in beliefs
4. Belief Conflict Resolution: resolve conflicting beliefs
5. Belief Erosion: detect and erode outdated beliefs
6. Posterior Sampling: sample from posterior distributions
7. Belief Propagation: propagate beliefs through networks
8. Epistemic Uncertainty Tracking: track and quantify uncertainty
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()


@dataclass
class Belief:
    """A belief with uncertainty."""
    belief_id: str
    proposition: str
    probability: float  # 0-1
    confidence: float  # 0-1 (how sure we are in the probability)
    evidence_count: int = 0
    last_updated: float = 0.0
    source: str = ""
    prior: float = 0.5
    category: str = "general"


@dataclass
class Evidence:
    """Evidence that updates a belief."""
    evidence_id: str
    proposition: str
    likelihood: float  # P(evidence | proposition)
    evidence_text: str = ""
    source: str = ""
    timestamp: float = 0.0
    weight: float = 1.0


@dataclass
class BeliefNetwork:
    """A network of related beliefs."""
    network_id: str
    beliefs: dict[str, Belief] = field(default_factory=dict)
    edges: list[tuple[str, str, float]] = field(default_factory=list)  # (belief_a, belief_b, correlation)
    update_count: int = 0


@dataclass
class PosteriorSample:
    """A sample from a posterior distribution."""
    proposition: str
    probability: float
    sample_weight: float
    is_outlier: bool = False


@dataclass
class BeliefConflict:
    """Conflicting beliefs."""
    conflict_id: str
    belief_a_id: str
    belief_b_id: str
    proposition_a: str
    proposition_b: str
    conflict_type: str  # logical, factual, probabilistic
    resolution: str
    resolution_reason: str


@dataclass
class UncertaintyReport:
    """Epistemic uncertainty report."""
    proposition: str
    expected_uncertainty: float
    decomposition: dict[str, float]
    recommended_actions: list[str]


@dataclass
class BeliefUpdate:
    """Result of a belief update."""
    belief_id: str
    old_probability: float
    new_probability: float
    change: float
    confidence: float
    evidence_used: list[str]
    interpretation: str


class BeliefUpdater:
    """Bayesian belief updating engine."""

    def __init__(self):
        self._beliefs: dict[str, Belief] = {}
        self._evidence: dict[str, Evidence] = {}
        self._networks: dict[str, BeliefNetwork] = {}
        self._conflicts: list[BeliefConflict] = []
        self._posterior_samples: dict[str, list[PosteriorSample]] = {}

    # ── Bayesian Belief Updating ──────────────────────────────────────────

    def create_belief(
        self,
        proposition: str,
        prior: float = 0.5,
        confidence: float = 0.5,
        source: str = "",
        category: str = "general",
    ) -> Belief:
        """Create a new belief."""
        belief = Belief(
            belief_id=hashlib.md5(
                f"belief:{proposition}:{time.time()}".encode()
            ).hexdigest()[:12],
            proposition=proposition,
            probability=prior,
            confidence=confidence,
            evidence_count=0,
            last_updated=time.time(),
            source=source,
            prior=prior,
            category=category,
        )

        self._beliefs[belief.belief_id] = belief
        return belief

    def add_evidence(
        self,
        proposition: str,
        likelihood: float,
        evidence_text: str = "",
        source: str = "",
        weight: float = 1.0,
    ) -> Evidence:
        """Add evidence to update a belief."""
        evidence = Evidence(
            evidence_id=hashlib.md5(
                f"evidence:{proposition}:{time.time()}".encode()
            ).hexdigest()[:12],
            proposition=proposition,
            likelihood=likelihood,
            evidence_text=evidence_text,
            source=source,
            timestamp=time.time(),
            weight=weight,
        )

        self._evidence[evidence.evidence_id] = evidence
        return evidence

    def update_belief(
        self,
        belief_id: str,
        evidence_id: str,
    ) -> BeliefUpdate:
        """Update a belief using evidence via Bayesian inference."""
        belief = self._beliefs.get(belief_id)
        if not belief:
            raise HTTPException(404, f"Belief {belief_id} not found")

        evidence = self._evidence.get(evidence_id)
        if not evidence:
            raise HTTPException(404, f"Evidence {evidence_id} not found")

        # Bayesian update: P(A|E) = P(E|A) * P(A) / P(E)
        # Simplified: posterior = likelihood * prior / (likelihood * prior + (1-likelihood) * (1-prior))
        prior = belief.probability
        likelihood = evidence.likelihood

        # Normalize
        posterior = (likelihood * prior) / (
            likelihood * prior + (1 - likelihood) * (1 - prior)
        )

        # Apply evidence weight
        posterior = prior + (posterior - prior) * evidence.weight

        # Clamp to [0, 1]
        posterior = max(0.0, min(1.0, posterior))

        old_prob = belief.probability
        belief.probability = posterior
        belief.evidence_count += 1
        belief.last_updated = time.time()

        # Increase confidence as more evidence accumulates
        belief.confidence = min(1.0, belief.confidence + 0.05)

        # Propagate to network
        self._propagate_belief(belief_id, posterior)

        update = BeliefUpdate(
            belief_id=belief_id,
            old_probability=old_prob,
            new_probability=posterior,
            change=posterior - old_prob,
            confidence=belief.confidence,
            evidence_used=[evidence_id],
            interpretation=(
                f"Evidence shifted belief from {old_prob:.2f} to {posterior:.2f} "
                f"(change: {posterior - old_prob:+.2f})"
            ),
        )

        return update

    def _propagate_belief(self, belief_id: str, new_prob: float):
        """Propagate belief update through the network."""
        # Find networks containing this belief
        for network in self._networks.values():
            if belief_id in network.beliefs:
                for edge_a, edge_b, corr in network.edges:
                    if edge_a == belief_id:
                        other = network.beliefs.get(edge_b)
                        if other:
                            # Adjust other belief based on correlation
                            delta = (new_prob - self._beliefs[belief_id].prior) * corr
                            other.probability = max(0.0, min(1.0, other.probability + delta * 0.1))
                    elif edge_b == belief_id:
                        other = network.beliefs.get(edge_a)
                        if other:
                            delta = (new_prob - self._beliefs[belief_id].prior) * corr
                            other.probability = max(0.0, min(1.0, other.probability + delta * 0.1))

    # ── Belief Networks ───────────────────────────────────────────────────

    def create_network(
        self,
        belief_ids: list[str],
        edges: list[tuple[str, str, float]],
    ) -> BeliefNetwork:
        """Create a belief network."""
        network = BeliefNetwork(
            network_id=hashlib.md5(
                f"network:{time.time()}".encode()
            ).hexdigest()[:12],
            beliefs={bid: self._beliefs.get(bid) for bid in belief_ids},
            edges=edges,
        )

        self._networks[network.network_id] = network
        return network

    def get_network(self, network_id: str) -> BeliefNetwork | None:
        return self._networks.get(network_id)

    def get_all_networks(self) -> list[BeliefNetwork]:
        return list(self._networks.values())

    # ── Confidence Calibration ───────────────────────────────────────────

    def calibrate_confidence(
        self,
        belief_id: str,
        historical_accuracy: float = 0.5,
    ) -> float:
        """Calibrate confidence in a belief."""
        belief = self._beliefs.get(belief_id)
        if not belief:
            raise HTTPException(404, f"Belief {belief_id} not found")

        # Brier score based calibration
        if belief.evidence_count > 5:
            # If we've been accurate, increase confidence
            calibrated = belief.confidence * (1 - abs(belief.probability - historical_accuracy))
        else:
            # Not enough data, keep prior confidence
            calibrated = belief.confidence * 0.5

        belief.confidence = max(0.0, min(1.0, calibrated))
        return belief.confidence

    def calibrate_all(self) -> dict:
        """Calibrate confidence for all beliefs."""
        calibrated = {}
        for bid, belief in self._beliefs.items():
            # Use evidence count as proxy for historical accuracy
            accuracy = min(1.0, belief.evidence_count / 10.0) if belief.evidence_count > 0 else 0.5
            calibrated[bid] = self.calibrate_confidence(bid, accuracy)
        return calibrated

    # ── Belief Conflict Resolution ───────────────────────────────────────

    def detect_conflicts(
        self,
        belief_ids: list[str],
    ) -> list[BeliefConflict]:
        """Detect conflicting beliefs."""
        beliefs = [self._beliefs[bid] for bid in belief_ids if bid in self._beliefs]
        conflicts = []

        for i, b1 in enumerate(beliefs):
            for b2 in beliefs[i+1:]:
                # Check for logical conflict (contradictory propositions)
                if self._are_contradictory(b1.proposition, b2.proposition):
                    # Resolve by preferring higher confidence
                    if b1.confidence > b2.confidence:
                        winner = b1
                        loser = b2
                    else:
                        winner = b2
                        loser = b1

                    conflict = BeliefConflict(
                        conflict_id=hashlib.md5(
                            f"conflict:{b1.belief_id}:{b2.belief_id}:{time.time()}".encode()
                        ).hexdigest()[:12],
                        belief_a_id=b1.belief_id,
                        belief_b_id=b2.belief_id,
                        proposition_a=b1.proposition,
                        proposition_b=b2.proposition,
                        conflict_type="logical",
                        resolution=winner.proposition,
                        resolution_reason=f"Higher confidence ({winner.confidence:.2f} vs {loser.confidence:.2f})",
                    )
                    conflicts.append(conflict)

                # Check for probabilistic conflict (both high probability but contradictory)
                if b1.probability > 0.7 and b2.probability > 0.7 and self._are_contradictory(b1.proposition, b2.proposition):
                    # Average the probabilities
                    avg_prob = (b1.probability + b2.probability) / 2
                    b1.probability = avg_prob
                    b2.probability = avg_prob

        self._conflicts.extend(conflicts)
        return conflicts

    def _are_contradictory(self, prop_a: str, prop_b: str) -> bool:
        """Check if two propositions are contradictory."""
        a = prop_a.lower()
        b = prop_b.lower()

        contradictions = [
            ("is true", "is false"),
            ("yes", "no"),
            ("always", "never"),
            ("increases", "decreases"),
            ("positive", "negative"),
        ]

        for pos, neg in contradictions:
            if pos in a and neg in b:
                return True
            if neg in a and pos in b:
                return True

        return False

    # ── Belief Erosion ───────────────────────────────────────────────────

    def erode_beliefs(
        self,
        max_age_days: float = 30.0,
        min_confidence: float = 0.1,
    ) -> list[str]:
        """Erode outdated or low-confidence beliefs."""
        to_erode = []
        now = time.time()

        for bid, belief in self._beliefs.items():
            age = now - belief.last_updated

            # Erode old beliefs
            if age > max_age_days * 86400 and belief.evidence_count < 3:
                belief.confidence *= 0.8  # 20% confidence reduction
                to_erode.append(f"{bid}: aged ({age / 86400:.0f} days), confidence reduced to {belief.confidence:.2f}")

            # Erode low-confidence beliefs
            if belief.confidence < min_confidence:
                belief.probability = belief.prior  # Reset to prior
                belief.confidence = 0.0
                to_erode.append(f"{bid}: confidence below threshold, reset to prior")

        return to_erode

    # ── Posterior Sampling ───────────────────────────────────────────────

    def sample_posterior(
        self,
        belief_id: str,
        n_samples: int = 100,
    ) -> list[PosteriorSample]:
        """Sample from posterior distribution."""
        belief = self._beliefs.get(belief_id)
        if not belief:
            raise HTTPException(404, f"Belief {belief_id} not found")

        samples = []
        mean = belief.probability
        std = belief.confidence * 0.1  # uncertainty around the belief

        for i in range(n_samples):
            # Normal sample using Box-Muller
            u1 = max(0.0001, (hash(f"sample_{i}_a_{time.time()}") % 10000) / 10000)
            u2 = max(0.0001, (hash(f"sample_{i}_b_{time.time()}") % 10000) / 10000)
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

            prob = max(0.0, min(1.0, mean + std * z))

            is_outlier = abs(prob - mean) > 2 * std

            samples.append(PosteriorSample(
                proposition=belief.proposition,
                probability=prob,
                sample_weight=1.0 / n_samples,
                is_outlier=is_outlier,
            ))

        self._posterior_samples[belief_id] = samples
        return samples

    # ── Belief Propagation ───────────────────────────────────────────────

    def propagate_beliefs(
        self,
        network_id: str,
        source_belief_id: str,
    ) -> dict:
        """Propagate beliefs through a network."""
        network = self._networks.get(network_id)
        if not network:
            raise HTTPException(404, f"Network {network_id} not found")

        source = self._beliefs.get(source_belief_id)
        if not source:
            raise HTTPException(404, f"Source belief {source_belief_id} not found")

        propagated = {}

        # Simple propagation along edges
        for edge_a, edge_b, corr in network.edges:
            if edge_a == source_belief_id:
                other = network.beliefs.get(edge_b)
                if other:
                    delta = (source.probability - source.prior) * corr
                    other.probability = max(0.0, min(1.0, other.probability + delta * 0.1))
                    other.last_updated = time.time()
                    propagated[edge_b] = {
                        "old": other.probability - delta * 0.1,
                        "new": other.probability,
                        "correlation": corr,
                    }
            elif edge_b == source_belief_id:
                other = network.beliefs.get(edge_a)
                if other:
                    delta = (source.probability - source.prior) * corr
                    other.probability = max(0.0, min(1.0, other.probability + delta * 0.1))
                    other.last_updated = time.time()
                    propagated[edge_a] = {
                        "old": other.probability - delta * 0.1,
                        "new": other.probability,
                        "correlation": corr,
                    }

        network.update_count += 1
        return {
            "network_id": network_id,
            "source": source_belief_id,
            "propagated_to": propagated,
        }

    # ── Epistemic Uncertainty Tracking ───────────────────────────────────

    def assess_uncertainty(
        self,
        belief_id: str,
    ) -> UncertaintyReport:
        """Assess epistemic uncertainty of a belief."""
        belief = self._beliefs.get(belief_id)
        if not belief:
            raise HTTPException(404, f"Belief {belief_id} not found")

        # Decompose uncertainty
        total_uncertainty = 1.0 - belief.confidence

        decomposition = {
            "data_uncertainty": 1.0 - min(1.0, belief.evidence_count / 10.0),
            "model_uncertainty": 0.2 if belief.evidence_count < 5 else 0.05,
            "aleatoric_uncertainty": 0.1,  # inherent randomness
            "representation_uncertainty": 0.1,
        }

        # Normalize
        total_decomp = sum(decomposition.values())
        if total_decomp > 0:
            for key in decomposition:
                decomposition[key] = decomposition[key] / total_decomp * total_uncertainty

        # Recommended actions
        actions = []
        if decomposition.get("data_uncertainty", 0) > 0.3:
            actions.append("Collect more evidence")
        if decomposition.get("model_uncertainty", 0) > 0.2:
            actions.append("Validate with alternative models")
        if belief.confidence < 0.3:
            actions.append("Seek expert review")

        return UncertaintyReport(
            proposition=belief.proposition,
            expected_uncertainty=total_uncertainty,
            decomposition=decomposition,
            recommended_actions=actions,
        )

    # ── Reporting ────────────────────────────────────────────────────────

    def get_belief_report(self) -> dict:
        """Get comprehensive belief report."""
        return {
            "total_beliefs": len(self._beliefs),
            "total_evidence": len(self._evidence),
            "total_networks": len(self._networks),
            "total_conflicts": len(self._conflicts),
            "avg_confidence": sum(b.confidence for b in self._beliefs.values()) / max(1, len(self._beliefs)),
            "beliefs_by_category": {
                cat: sum(1 for b in self._beliefs.values() if b.category == cat)
                for cat in set(b.category for b in self._beliefs.values())
            },
        }


# ── API Routes ──────────────────────────────────────────────────────────────

belief_updater = BeliefUpdater()


@router.post("/api/v1/inference/beliefs/create")
async def create_belief(payload: dict):
    """Create a belief."""
    belief = belief_updater.create_belief(
        proposition=payload["proposition"],
        prior=payload.get("prior", 0.5),
        confidence=payload.get("confidence", 0.5),
        source=payload.get("source", ""),
        category=payload.get("category", "general"),
    )
    return belief.__dict__


@router.post("/api/v1/inference/beliefs/evidence")
async def add_evidence(payload: dict):
    """Add evidence."""
    evidence = belief_updater.add_evidence(
        proposition=payload["proposition"],
        likelihood=payload["likelihood"],
        evidence_text=payload.get("evidence_text", ""),
        source=payload.get("source", ""),
        weight=payload.get("weight", 1.0),
    )
    return evidence.__dict__


@router.post("/api/v1/inference/beliefs/{belief_id}/update")
async def update_belief(belief_id: str, payload: dict):
    """Update a belief with evidence."""
    return belief_updater.update_belief(belief_id, payload["evidence_id"]).__dict__


@router.post("/api/v1/inference/beliefs/networks")
async def create_network(payload: dict):
    """Create a belief network."""
    network = belief_updater.create_network(
        payload["belief_ids"],
        payload["edges"],
    )
    return {
        "network_id": network.network_id,
        "beliefs": list(network.beliefs.keys()),
        "edges": [list(e) for e in network.edges],
    }


@router.get("/api/v1/inference/beliefs/networks")
async def get_all_networks():
    """Get all networks."""
    return [
        {
            "network_id": n.network_id,
            "beliefs": list(n.beliefs.keys()),
            "edges": [list(e) for e in n.edges],
        }
        for n in belief_updater.get_all_networks()
    ]


@router.post("/api/v1/inference/beliefs/calibrate")
async def calibrate_all():
    """Calibrate all beliefs."""
    return belief_updater.calibrate_all()


@router.post("/api/v1/inference/beliefs/conflicts")
async def detect_conflicts(payload: dict):
    """Detect conflicts."""
    conflicts = belief_updater.detect_conflicts(payload["belief_ids"])
    return [c.__dict__ for c in conflicts]


@router.post("/api/v1/inference/beliefs/erode")
async def erode_beliefs(payload: dict):
    """Erode outdated beliefs."""
    result = belief_updater.erode_beliefs(
        max_age_days=payload.get("max_age_days", 30.0),
        min_confidence=payload.get("min_confidence", 0.1),
    )
    return {"eroded": result}


@router.post("/api/v1/inference/beliefs/{belief_id}/sample")
async def sample_posterior(belief_id: str, n_samples: int = 100):
    """Sample from posterior."""
    samples = belief_updater.sample_posterior(belief_id, n_samples)
    return [s.__dict__ for s in samples]


@router.post("/api/v1/inference/beliefs/networks/{network_id}/propagate")
async def propagate_beliefs(network_id: str, payload: dict):
    """Propagate beliefs through network."""
    return belief_updater.propagate_beliefs(
        network_id,
        payload["source_belief_id"],
    )


@router.get("/api/v1/inference/beliefs/{belief_id}/uncertainty")
async def assess_uncertainty(belief_id: str):
    """Assess uncertainty."""
    return belief_updater.assess_uncertainty(belief_id).__dict__


@router.get("/api/v1/inference/beliefs/report")
async def get_belief_report():
    """Get belief report."""
    return belief_updater.get_belief_report()
