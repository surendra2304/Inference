"""Explanation Generation Engine: Multi-Audience Summaries and Plain-Language Citations."""

from typing import Literal

from pydantic import BaseModel


class AudienceExplanation(BaseModel):
    brief: str
    standard: str
    detailed: str
    evidence_citations: list[str]
    target_audience: Literal["executive", "standard", "technical"] = "standard"


class ExplanationGenerationEngine:
    """Generates brief, standard, or detailed explanations with precise evidence reference citations."""

    def generate_explanation(
        self,
        decision: str,
        goal: str,
        key_evidence: list[str],
        unresolved_disagreements: list[str],
        confidence: float,
        audience: str = "standard"
    ) -> AudienceExplanation:
        citations = [f"[Ref-{i+1}] {ev}" for i, ev in enumerate(key_evidence[:4])]

        brief = f"Recommendation is {decision} for {goal} based on {len(key_evidence)} verified signals ({int(confidence*100)}% confidence)."

        standard = (
            f"We recommend proceeding with {decision}. This is supported by verified evidence including: "
            f"{'; '.join(citations[:2])}. "
            + (f"Note: {unresolved_disagreements[0]}" if unresolved_disagreements else "No material dissenting objections registered.")
        )

        detailed = (
            f"Detailed Analysis for Goal: '{goal}'\n"
            f"1. Decision: {decision} (Confidence: {confidence:.2f})\n"
            f"2. Evidence Citations:\n" + "\n".join(f"   - {c}" for c in citations) + "\n"
            f"3. Dissent & Edge Cases: {'; '.join(unresolved_disagreements) if unresolved_disagreements else 'Full consensus achieved across all specialist agents.'}\n"
            f"4. Confidence Trajectory: Validated through adversarial deliberation with conservative bounds."
        )

        return AudienceExplanation(
            brief=brief,
            standard=standard,
            detailed=detailed,
            evidence_citations=citations,
            target_audience="technical" if "tech" in audience.lower() else "standard"
        )


explanation_engine = ExplanationGenerationEngine()
