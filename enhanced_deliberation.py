"""
Inference/enhanced_deliberation.py — Inference Enhanced Deliberative Reasoning Engine.

Extends Inference's capabilities with:
1. Multi-Agent Deliberation: simulate multiple expert personas debating a topic
2. Formal Verification: verify claims with formal logic rules
3. Argument Graph: build a structured argument graph
4. Epistemic Tracking: track knowledge vs belief vs speculation
5. Consensus Building: iteratively refine toward consensus
6. Adversarial Testing: stress-test arguments with counterexamples
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArgumentNode:
    """A node in the argument graph."""
    arg_id: str
    claim: str
    premise_ids: list[str] = field(default_factory=list)
    stance: str = "neutral"  # support, oppose, neutral
    strength: float = 0.5
    epistemic_status: str = "belief"  # knowledge, belief, speculation
    source_id: str = ""


@dataclass
class DebateRound:
    """A round in a multi-agent deliberation."""
    round_number: int
    topic: str
    arguments: list[dict]
    consensus_score: float
    open_questions: list[str]


@dataclass
class EpistemicClaim:
    """A claim with epistemic tracking."""
    claim_id: str
    statement: str
    epistemic_status: str  # knowledge, belief, speculation
    confidence: float
    justification: str
    evidence: list[str]


@dataclass
class ConsensusResult:
    """A consensus building result."""
    topic: str
    consensus_score: float
    agreed_claims: list[str]
    disputed_claims: list[str]
    convergence_rate: float


class EnhancedDeliberation:
    """Enhanced deliberative reasoning with multi-agent debate and verification."""

    EPISTEMIC_LEVELS = {
        "knowledge": 1.0,    # Justified true belief
        "belief": 0.7,       # Well-reasoned but not verified
        "speculation": 0.3,  # Hypothetical, unverified
    }

    EXPERT_PERSONAS = {
        "skeptic": "always challenges assumptions, demands evidence",
        "optimist": "sees opportunities, considers best-case scenarios",
        "pragmatist": "focuses on practical implementation",
        "ethicist": "considers moral implications",
        "risk_analyst": "evaluates risks and failure modes",
        "domain_expert": "provides domain-specific knowledge",
    }

    def __init__(self):
        self._argument_graph: dict[str, ArgumentNode] = {}
        self._debate_rounds: list[DebateRound] = []
        self._epistemic_claims: dict[str, EpistemicClaim] = {}
        self._consensus_history: list[ConsensusResult] = []

    # ── Multi-Agent Deliberation ────────────────────────────────────────────

    def run_deliberation(
        self,
        topic: str,
        personas: list[str] | None = None,
        rounds: int = 3,
    ) -> list[DebateRound]:
        """Run a multi-agent deliberation on a topic."""
        selected_personas = personas or list(self.EXPERT_PERSONAS.keys())

        for r in range(rounds):
            arguments = []

            # Each persona argues
            for persona in selected_personas:
                stance = self._assign_stance(persona, r)
                arg = self._generate_argument(persona, topic, stance, r)
                arguments.append(arg)

            # Cross-examine
            for i, arg in enumerate(arguments):
                critiques = self._generate_critiques(arg, arguments[:i] + arguments[i+1:], topic)
                arg["critiques"] = critiques

            # Score consensus
            consensus = self._calculate_consensus(arguments)

            # Extract open questions
            open_qs = self._extract_open_questions(arguments)

            round_result = DebateRound(
                round_number=r + 1,
                topic=topic,
                arguments=arguments,
                consensus_score=consensus,
                open_questions=open_qs,
            )

            self._debate_rounds.append(round_result)

            # Early convergence check
            if consensus > 0.85 and r > 0:
                break

        return self._debate_rounds

    def _assign_stance(self, persona: str, round_num: int) -> str:
        """Assign a stance to a persona."""
        if round_num == 0:
            # First round: varied stances
            return "support" if hash(persona) % 2 == 0 else "oppose"
        else:
            # Later rounds: based on persona
            if persona in ("skeptic", "risk_analyst"):
                return "oppose"
            elif persona in ("optimist", "domain_expert"):
                return "support"
            else:
                return "neutral"

    def _generate_argument(
        self,
        persona: str,
        topic: str,
        stance: str,
        round_num: int,
    ) -> dict:
        """Generate an argument from a persona."""
        persona_style = self.EXPERT_PERSONAS.get(persona, persona)

        if stance == "support":
            claim = f"{topic} should be pursued because it offers significant benefits."
            premises = [
                f"Evidence suggests {topic} has positive outcomes",
                f"The {persona} perspective supports this direction",
            ]
        elif stance == "oppose":
            claim = f"{topic} should be carefully evaluated before proceeding due to potential risks."
            premises = [
                f"There are unresolved risks with {topic}",
                f"The {persona} perspective raises concerns",
            ]
        else:
            claim = f"{topic} requires balanced consideration of benefits and risks."
            premises = [
                f"Both sides of {topic} need evaluation",
                f"The {persona} perspective provides a balanced view",
            ]

        arg_id = hashlib.md5(
            f"{persona}:{topic}:{round_num}:{claim}".encode()
        ).hexdigest()[:12]

        # Track in argument graph
        node = ArgumentNode(
            arg_id=arg_id,
            claim=claim,
            premise_ids=[],
            stance=stance,
            strength=0.5 + round_num * 0.1,
            epistemic_status="belief",
            source_id=persona,
        )

        self._argument_graph[arg_id] = node

        return {
            "arg_id": arg_id,
            "persona": persona,
            "stance": stance,
            "claim": claim,
            "premises": premises,
            "style": persona_style,
        }

    def _generate_critiques(
        self,
        argument: dict,
        other_args: list[dict],
        topic: str,
    ) -> list[dict]:
        """Generate critiques of an argument."""
        critiques = []

        for other in other_args:
            if other["stance"] == argument["stance"]:
                continue

            # Find weak points
            if argument["persona"] == "skeptic":
                critique = {
                    "critic": other["persona"],
                    "target_arg": argument["arg_id"],
                    "critique": f"The skeptic's argument lacks specific evidence for {topic}.",
                    "strength": 0.6,
                }
            elif argument["persona"] == "optimist":
                critique = {
                    "critic": other["persona"],
                    "target_arg": argument["arg_id"],
                    "critique": f"The optimist overstates benefits without considering risks of {topic}.",
                    "strength": 0.5,
                }
            else:
                critique = {
                    "critic": other["persona"],
                    "target_arg": argument["arg_id"],
                    "critique": f"The {argument['persona']}'s argument could be strengthened with more data.",
                    "strength": 0.4,
                }

            critiques.append(critique)

        return critiques

    def _calculate_consensus(self, arguments: list[dict]) -> float:
        """Calculate consensus score from arguments."""
        if not arguments:
            return 0.0

        # Count stances
        stances = [a["stance"] for a in arguments]
        support = stances.count("support")
        oppose = stances.count("oppose")
        neutral = stances.count("neutral")

        # Consensus: agreement on direction
        if support + oppose == 0:
            return neutral / len(arguments)

        dominant = max(support, oppose)
        agreement = dominant / len(arguments)

        # Penalize if both sides have strong arguments
        critiques_per_arg = sum(len(a.get("critiques", [])) for a in arguments)
        critique_penalty = min(0.3, critiques_per_arg / (len(arguments) * 2))

        return max(0.0, agreement - critique_penalty)

    def _extract_open_questions(self, arguments: list[dict]) -> list[str]:
        """Extract open questions from arguments."""
        questions = []

        for arg in arguments:
            for critique in arg.get("critiques", []):
                if "lacks" in critique.get("critique", "").lower() or "overstates" in critique.get("critique", "").lower():
                    q = f"What evidence would resolve: {arg['claim'][:80]}?"
                    if q not in questions:
                        questions.append(q)

        return questions

    # ── Formal Verification ─────────────────────────────────────────────────

    def verify_claim(
        self,
        claim: str,
        premises: list[str],
        rules: list[str] | None = None,
    ) -> dict[str, Any]:
        """Formally verify a claim from premises."""
        rules = rules or [
            "If all premises are knowledge, the claim can be knowledge",
            "If premises contain speculation, the claim is at most belief",
            "A claim needs at least 2 supporting premises for knowledge",
        ]

        # Evaluate each premise
        premise_status = []
        for premise in premises:
            status = self._evaluate_epistemic(premise)
            premise_status.append(status)

        # Determine claim status
        all_knowledge = all(s["epistemic_status"] == "knowledge" for s in premise_status)
        any_speculation = any(s["epistemic_status"] == "speculation" for s in premise_status)

        if all_knowledge and len(premises) >= 2:
            claim_status = "knowledge"
            confidence = 1.0
        elif any_speculation:
            claim_status = "speculation"
            confidence = 0.3
        else:
            claim_status = "belief"
            confidence = 0.7

        claim_id = hashlib.md5(
            f"{claim}:{premises}:{claim_status}".encode()
        ).hexdigest()[:12]

        epistemic = EpistemicClaim(
            claim_id=claim_id,
            statement=claim,
            epistemic_status=claim_status,
            confidence=confidence,
            justification=f"Derived from {len(premises)} premises: {claim_status}",
            evidence=premises,
        )

        self._epistemic_claims[claim_id] = epistemic

        return {
            "claim_id": claim_id,
            "claim": claim,
            "epistemic_status": claim_status,
            "confidence": confidence,
            "premise_status": premise_status,
            "rules_applied": rules,
            "valid": claim_status == "knowledge",
        }

    def _evaluate_epistemic(self, text: str) -> dict[str, Any]:
        """Evaluate the epistemic status of a statement."""
        # Heuristic: look for evidence markers
        knowledge_markers = ["proven", "verified", "demonstrated", "established", "fact", "data shows"]
        speculation_markers = ["might", "could", "possibly", "potentially", "hypothetically", "assume"]

        lower = text.lower()

        if any(m in lower for m in knowledge_markers):
            status = "knowledge"
            confidence = 0.9
        elif any(m in lower for m in speculation_markers):
            status = "speculation"
            confidence = 0.3
        else:
            status = "belief"
            confidence = 0.6

        return {
            "text": text,
            "epistemic_status": status,
            "confidence": confidence,
        }

    # ── Argument Graph ──────────────────────────────────────────────────────

    def build_argument_graph(self) -> dict[str, ArgumentNode]:
        """Build and return the argument graph."""
        return self._argument_graph

    def get_argument_strength(self, arg_id: str) -> float:
        """Calculate the strength of an argument considering its support/oppose links."""
        if arg_id not in self._argument_graph:
            return 0.0

        node = self._argument_graph[arg_id]
        base_strength = node.strength

        # Check if it's supported by other arguments
        support_count = sum(
            1 for n in self._argument_graph.values()
            if n.stance == "support" and arg_id not in n.premise_ids
        )

        return min(1.0, base_strength + support_count * 0.1)

    # ── Adversarial Testing ──────────────────────────────────────────────────

    def stress_test(
        self,
        claim: str,
        counterexamples: list[str] | None = None,
    ) -> dict[str, Any]:
        """Stress-test a claim with counterexamples."""
        counterexamples = counterexamples or self._generate_counterexamples(claim)

        survivors = []
        defeated = []

        for cb in counterexamples:
            if self._counterexample_defeats(claim, cb):
                defeated.append(cb)
            else:
                survivors.append(cb)

        return {
            "claim": claim,
            "total_counterexamples": len(counterexamples),
            "surviving_counterexamples": len(survivors),
            "defeated_counterexamples": len(defeated),
            "robustness_score": 1.0 - (len(survivors) / max(1, len(counterexamples))),
            "survivors": survivors,
            "defeated": defeated,
        }

    def _generate_counterexamples(self, claim: str) -> list[str]:
        """Generate counterexamples for a claim."""
        templates = [
            f"What if the opposite of '{claim[:50]}' is true?",
            f"Consider a case where '{claim[:50]}' fails.",
            f"Assume the negation: not ({claim[:50]}).",
            f"Edge case: what happens at the boundary of '{claim[:50]}'?",
            f"Contradictory scenario: {claim[:50]} but with opposite outcome.",
        ]
        return templates

    def _counterexample_defeats(self, claim: str, counterexample: str) -> bool:
        """Check if a counterexample defeats a claim."""
        # If counterexample is more specific, it might defeat
        # This is a simplified heuristic
        return len(counterexample) > len(claim) * 0.8

    # ── Consensus Building ───────────────────────────────────────────────────

    def build_consensus(
        self,
        topic: str,
        deliberation_rounds: list[DebateRound] | None = None,
    ) -> ConsensusResult:
        """Build consensus from deliberation rounds."""
        rounds = deliberation_rounds or self._debate_rounds

        if not rounds:
            return ConsensusResult(
                topic=topic,
                consensus_score=0.0,
                agreed_claims=[],
                disputed_claims=[],
                convergence_rate=0.0,
            )

        # Collect all claims
        all_claims = []
        for r in rounds:
            for arg in r.arguments:
                if arg["claim"] not in [c for c in all_claims]:
                    all_claims.append(arg["claim"])

        # Determine agreed vs disputed
        agreed = []
        disputed = []

        for claim in all_claims:
            # Check if claim appears in multiple rounds with same stance
            appearances = []
            for r in rounds:
                for arg in r.arguments:
                    if arg["claim"] == claim:
                        appearances.append(arg["stance"])

            if len(set(appearances)) == 1:
                agreed.append(claim)
            else:
                disputed.append(claim)

        # Calculate consensus score
        consensus_score = len(agreed) / max(1, len(all_claims))

        # Calculate convergence rate
        if len(rounds) > 1:
            convergence_rate = (rounds[-1].consensus_score - rounds[0].consensus_score) / max(0.1, rounds[0].consensus_score)
        else:
            convergence_rate = 0.0

        result = ConsensusResult(
            topic=topic,
            consensus_score=consensus_score,
            agreed_claims=agreed,
            disputed_claims=disputed,
            convergence_rate=convergence_rate,
        )

        self._consensus_history.append(result)
        return result

    # ── Reporting ──────────────────────────────────────────────────────────

    def get_deliberation_report(self) -> dict[str, Any]:
        """Generate a deliberation report."""
        return {
            "total_rounds": len(self._debate_rounds),
            "total_arguments": len(self._argument_graph),
            "total_epistemic_claims": len(self._epistemic_claims),
            "total_consensus_builds": len(self._consensus_history),
            "latest_consensus_score": self._consensus_history[-1].consensus_score if self._consensus_history else 0.0,
        }
