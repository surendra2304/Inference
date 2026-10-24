"""
Inference/argument_evaluator.py — Inference Argument Evaluation Engine.

Extends Inference's capabilities with:
1. Argument Structure Analysis: analyze the structure of arguments
2. Claim Evaluation: evaluate the strength of claims
3. Evidence Assessment: assess the quality of evidence
4. Reasoning Chain Validation: validate chains of reasoning
5. Fallacy Detection: detect logical fallacies
6. Argument Strength Scoring: score overall argument strength
7. Counterargument Analysis: analyze and evaluate counterarguments
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArgumentClaim:
    """A claim within an argument."""
    claim_id: str
    statement: str
    type: str  # factual, normative, predictive
    confidence: float
    supporting_evidence: list[str] = field(default_factory=list)
    is_conclusion: bool = False
    is_premise: bool = False


@dataclass
class Argument:
    """A complete argument."""
    argument_id: str
    topic: str
    claims: list[ArgumentClaim] = field(default_factory=list)
    structure: str = ""  # deductive, inductive, abductive
    strength: float = 0.0
    validity: bool = False
    soundness: bool = False
    fallacies: list[str] = field(default_factory=list)


@dataclass
class Fallacy:
    """A logical fallacy."""
    fallacy_id: str
    name: str
    description: str
    location: str
    severity: str  # minor, major, critical
    impact: str


@dataclass
class EvidenceQuality:
    """Quality assessment of evidence."""
    source: str
    reliability: float
    relevance: float
    recency: float
    sufficiency: float
    overall: float
    weaknesses: list[str] = field(default_factory=list)


@dataclass
class Counterargument:
    """A counterargument."""
    counterargument_id: str
    statement: str
    addresses_claim: str
    strength: float
    weaknesses: list[str] = field(default_factory=list)
    rebuttal: str = ""


class ArgumentEvaluator:
    """Argument evaluation engine."""

    FALLACIES = {
        "ad_hominem": {
            "pattern": r"(you|they|he|she)\s+(are|is|is just|is only)",
            "description": "Attacking the person rather than the argument",
        },
        "straw_man": {
            "pattern": r"(you say|you claim|you think).*but actually",
            "description": "Misrepresenting someone's argument",
        },
        "false_dilemma": {
            "pattern": r"(either|either.*or)\s+(and|but)",
            "description": "Presenting only two options when more exist",
        },
        "hasty_generalization": {
            "pattern": r"(all|every|always|never|none)\s+\w+",
            "description": "Making generalizations from limited examples",
        },
        "circular_reasoning": {
            "pattern": r"(\w+).*because.*\1",
            "description": "Using the conclusion as a premise",
        },
        "appeal_to_authority": {
            "pattern": r"(experts say|scientists say|studies show|research proves)",
            "description": "Relying on authority without evidence",
        },
        "slippery_slope": {
            "pattern": r"(if.*then.*will.*will.*will)",
            "description": "Assuming a chain of events without evidence",
        },
        "begging_the_question": {
            "pattern": r"(obviously|clearly|self-evident)",
            "description": "Assuming the conclusion in the premises",
        },
    }

    def __init__(self):
        self._arguments: dict[str, Argument] = {}
        self._fallacies: list[Fallacy] = []
        self._counterarguments: dict[str, Counterargument] = {}

    # ── Argument Structure Analysis ─────────────────────────────────────────

    def analyze_structure(
        self,
        topic: str,
        text: str,
    ) -> Argument:
        """Analyze the structure of an argument."""
        argument_id = f"arg_{int(time.time())}"

        # Extract claims
        claims = self._extract_claims(text)

        # Determine structure
        structure = self._determine_structure(text)

        # Find conclusion and premises
        conclusion_idx = self._find_conclusion(claims)
        premises = [c for i, c in enumerate(claims) if i != conclusion_idx]

        argument = Argument(
            argument_id=argument_id,
            topic=topic,
            claims=claims,
            structure=structure,
        )

        # Evaluate argument
        self._evaluate_argument(argument)

        self._arguments[argument_id] = argument
        return argument

    def _extract_claims(self, text: str) -> list[ArgumentClaim]:
        """Extract claims from text."""
        claims = []

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            # Determine claim type
            claim_type = "factual"
            if any(w in sentence.lower() for w in ["should", "must", "ought", "good", "bad"]):
                claim_type = "normative"
            elif any(w in sentence.lower() for w in ["will", "would", "might", "could", "predict"]):
                claim_type = "predictive"

            claim = ArgumentClaim(
                claim_id=f"claim_{i}",
                statement=sentence,
                type=claim_type,
                confidence=0.5,
            )

            claims.append(claim)

        return claims

    def _determine_structure(self, text: str) -> str:
        """Determine argument structure."""
        text_lower = text.lower()

        if any(w in text_lower for w in ["therefore", "thus", "so", "conclusion"]):
            return "deductive"
        elif any(w in text_lower for w in ["likely", "probably", "suggests", "indicates"]):
            return "inductive"
        elif any(w in text_lower for w in ["best explanation", "must be", "only explanation"]):
            return "abductive"
        else:
            return "unclear"

    def _find_conclusion(self, claims: list[ArgumentClaim]) -> int:
        """Find the conclusion in the claims."""
        # Look for conclusion indicators
        for i, claim in enumerate(claims):
            text = claim.statement.lower()
            if any(w in text for w in ["therefore", "thus", "so", "in conclusion", "finally"]):
                claim.is_conclusion = True
                return i

        # If no explicit conclusion, assume the last claim
        if claims:
            claims[-1].is_conclusion = True
            return len(claims) - 1

        return -1

    # ── Claim Evaluation ─────────────────────────────────────────────────────

    def evaluate_claim(self, claim: ArgumentClaim) -> dict[str, Any]:
        """Evaluate a single claim."""
        # Check for evidence
        has_evidence = len(claim.supporting_evidence) > 0

        # Check for quantifiers
        text = claim.statement.lower()
        has_quantifier = any(w in text for w in ["all", "some", "most", "many", "few"])

        # Check for hedges
        has_hedge = any(w in text for w in ["might", "could", "possibly", "likely", "suggests"])

        # Calculate confidence
        confidence = 0.5
        if has_evidence:
            confidence += 0.2
        if has_quantifier:
            confidence += 0.1
        if has_hedge:
            confidence -= 0.1

        confidence = max(0.0, min(1.0, confidence))
        claim.confidence = confidence

        return {
            "claim_id": claim.claim_id,
            "confidence": confidence,
            "has_evidence": has_evidence,
            "has_quantifier": has_quantifier,
            "has_hedge": has_hedge,
            "type": claim.type,
        }

    # ── Evidence Assessment ──────────────────────────────────────────────────

    def assess_evidence(
        self,
        evidence_list: list[dict],
    ) -> EvidenceQuality:
        """Assess the quality of evidence."""
        if not evidence_list:
            return EvidenceQuality(
                source="none",
                reliability=0.0,
                relevance=0.0,
                recency=0.0,
                sufficiency=0.0,
                overall=0.0,
                weaknesses=["No evidence provided"],
            )

        # Calculate individual scores
        reliability = sum(e.get("reliability", 0.5) for e in evidence_list) / len(evidence_list)
        relevance = sum(e.get("relevance", 0.5) for e in evidence_list) / len(evidence_list)

        # Recency (simplified: newer is better)
        now = time.time()
        recencies = []
        for e in evidence_list:
            timestamp = e.get("timestamp", now)
            recency = max(0.0, 1.0 - (now - timestamp) / (365 * 86400 * 5))  # 5-year decay
            recencies.append(recency)
        recency = sum(recencies) / len(recencies)

        # Sufficiency
        sufficiency = min(1.0, len(evidence_list) / 5.0)

        # Overall
        overall = (reliability * 0.4 + relevance * 0.3 + recency * 0.2 + sufficiency * 0.1)

        # Identify weaknesses
        weaknesses = []
        if reliability < 0.5:
            weaknesses.append("Low source reliability")
        if relevance < 0.5:
            weaknesses.append("Low relevance to claim")
        if recency < 0.3:
            weaknesses.append("Outdated evidence")
        if sufficiency < 0.4:
            weaknesses.append("Insufficient evidence")

        return EvidenceQuality(
            source="multiple",
            reliability=reliability,
            relevance=relevance,
            recency=recency,
            sufficiency=sufficiency,
            overall=overall,
            weaknesses=weaknesses,
        )

    # ── Fallacy Detection ────────────────────────────────────────────────────

    def detect_fallacies(self, text: str) -> list[Fallacy]:
        """Detect logical fallacies in text."""
        fallacies = []

        for name, info in self.FALLACIES.items():
            matches = re.findall(info["pattern"], text, re.IGNORECASE)

            for match in matches:
                fallacy = Fallacy(
                    fallacy_id=f"fallacy_{name}_{int(time.time())}",
                    name=name,
                    description=info["description"],
                    location=str(match)[:100],
                    severity="major" if "critical" in name.lower() else "minor",
                    impact=f"Reduces argument credibility by {name}",
                )
                fallacies.append(fallacy)
                self._fallacies.append(fallacy)

        return fallacies

    # ── Reasoning Chain Validation ──────────────────────────────────────────

    def validate_reasoning_chain(
        self,
        argument: Argument,
    ) -> dict[str, Any]:
        """Validate the reasoning chain of an argument."""
        if not argument.claims:
            return {"valid": False, "reason": "No claims found"}

        # Check if claims are in logical order
        conclusion_idx = next(
            (i for i, c in enumerate(argument.claims) if c.is_conclusion),
            -1
        )

        # Check if all non-conclusion claims are premises
        premises = [c for c in argument.claims if not c.is_conclusion]

        if not premises:
            return {"valid": False, "reason": "No premises found"}

        # Check if conclusion follows from premises
        # Simplified: check if conclusion shares keywords with premises
        conclusion = argument.claims[conclusion_idx].statement.lower()
        conclusion_words = set(conclusion.split())

        premise_words = set()
        for p in premises:
            premise_words.update(p.statement.lower().split())

        overlap = len(conclusion_words & premise_words) / max(1, len(conclusion_words))

        return {
            "valid": overlap > 0.1,
            "reason": f"Conclusion shares {overlap:.1%} keywords with premises",
            "overlap": overlap,
            "conclusion": argument.claims[conclusion_idx].statement,
            "premise_count": len(premises),
        }

    # ── Argument Strength Scoring ─────────────────────────────────────────────

    def score_argument(self, argument: Argument) -> dict[str, Any]:
        """Score the overall strength of an argument."""
        # Evaluate all claims
        claim_scores = [
            self.evaluate_claim(c) for c in argument.claims
        ]

        # Average claim confidence
        avg_claim_conf = sum(s["confidence"] for s in claim_scores) / max(1, len(claim_scores))

        # Fallacy penalty
        fallacy_penalty = len(argument.fallacies) * 0.1

        # Evidence bonus
        evidence_bonus = 0.0
        for claim in argument.claims:
            if claim.supporting_evidence:
                evidence_bonus += 0.05

        # Structure bonus
        structure_bonus = 0.0
        if argument.structure in ("deductive", "inductive"):
            structure_bonus = 0.05

        # Overall strength
        strength = avg_claim_conf - fallacy_penalty + evidence_bonus + structure_bonus
        strength = max(0.0, min(1.0, strength))

        # Validity and soundness
        validity = self.validate_reasoning_chain(argument)["valid"]
        soundness = validity and avg_claim_conf > 0.5

        argument.strength = strength
        argument.validity = validity
        argument.soundness = soundness

        return {
            "strength": strength,
            "validity": validity,
            "soundness": soundness,
            "avg_claim_confidence": avg_claim_conf,
            "fallacy_penalty": fallacy_penalty,
            "evidence_bonus": evidence_bonus,
            "structure_bonus": structure_bonus,
        }

    def _evaluate_argument(self, argument: Argument):
        """Evaluate an argument."""
        # Detect fallacies in each claim
        for claim in argument.claims:
            fallacies = self.detect_fallacies(claim.statement)
            argument.fallacies.extend(fallacies)

        # Score the argument
        self.score_argument(argument)

    # ── Counterargument Analysis ─────────────────────────────────────────────

    def analyze_counterargument(
        self,
        statement: str,
        addresses_claim: str,
        rebuttal: str = "",
    ) -> Counterargument:
        """Analyze a counterargument."""
        counterargument_id = f"ca_{int(time.time())}"

        # Assess strength
        strength = 0.5

        # Check for evidence
        if any(w in statement.lower() for w in ["evidence", "data", "study", "research"]):
            strength += 0.2

        # Check for fallacies
        fallacies = self.detect_fallacies(statement)
        strength -= len(fallacies) * 0.1

        strength = max(0.0, min(1.0, strength))

        counterargument = Counterargument(
            counterargument_id=counterargument_id,
            statement=statement,
            addresses_claim=addresses_claim,
            strength=strength,
            weaknesses=[f.description for f in fallacies],
            rebuttal=rebuttal,
        )

        self._counterarguments[counterargument_id] = counterargument
        return counterargument

    def generate_rebuttal(
        self,
        argument: Argument,
        counterargument: Counterargument,
    ) -> str:
        """Generate a rebuttal to a counterargument."""
        if counterargument.strength < 0.3:
            return (
                f"The counterargument has limited strength ({counterargument.strength:.1f}). "
                f"Key weaknesses: {', '.join(counterargument.weaknesses) or 'none identified'}."
            )

        # Find the claim that the counterargument addresses
        claim = next(
            (c for c in argument.claims if c.statement == counterargument.addresses_claim),
            None
        )

        if claim:
            return (
                f"While the counterargument raises valid concerns, "
                f"the original claim is supported by "
                f"{len(claim.supporting_evidence)} pieces of evidence "
                f"and maintains a confidence of {claim.confidence:.1f}. "
                f"The counterargument can be addressed by "
                f"providing additional context or clarifying assumptions."
            )

        return "The counterargument does not directly address the main claims of the argument."

    # ── Reporting ──────────────────────────────────────────────────────────

    def get_evaluation_report(self) -> dict[str, Any]:
        """Generate an argument evaluation report."""
        return {
            "total_arguments": len(self._arguments),
            "total_fallacies": len(self._fallacies),
            "total_counterarguments": len(self._counterarguments),
            "avg_argument_strength": sum(
                a.strength for a in self._arguments.values()
            ) / max(1, len(self._arguments)),
            "valid_arguments": sum(1 for a in self._arguments.values() if a.validity),
            "sound_arguments": sum(1 for a in self._arguments.values() if a.soundness),
        }
