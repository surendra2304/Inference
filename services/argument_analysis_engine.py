"""
Inference/services/argument_analysis_engine.py — Inference Argument Analysis Engine.

Argument structure analysis, claim extraction, evidence evaluation,
and argument quality assessment.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()


@dataclass
class ArgumentClaim:
    """A claim extracted from an argument."""
    claim_id: str
    text: str
    type: str = "assertion"  # assertion, premise, conclusion, evidence
    confidence: float = 0.5
    supporting_evidence: list[str] = field(default_factory=list)


@dataclass
class ArgumentStructure:
    """The structure of an argument."""
    argument_id: str
    premises: list[ArgumentClaim] = field(default_factory=list)
    conclusion: ArgumentClaim | None = None
    evidence: list[ArgumentClaim] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    overall_quality: float = 0.0


@dataclass
class ArgumentAnalysis:
    """Result of analyzing an argument."""
    analysis_id: str
    original_text: str
    structure: ArgumentStructure | None = None
    extracted_claims: list[ArgumentClaim] = field(default_factory=list)
    logical_fallacies: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class ArgumentAnalysisEngine:
    """Engine for analyzing argument structure and quality."""

    def __init__(self) -> None:
        self._analyses: dict[str, ArgumentAnalysis] = {}

    def analyze_argument(self, text: str) -> ArgumentAnalysis:
        """Analyze an argument's structure and quality."""
        analysis_id = f"arg_{uuid.uuid4().hex[:12]}"
        analysis = ArgumentAnalysis(analysis_id=analysis_id, original_text=text)

        # Extract claims (simplified)
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
        claims = []
        for i, sentence in enumerate(sentences):
            claim = ArgumentClaim(
                claim_id=f"claim_{i}",
                text=sentence[:200],
                type="premise" if i < len(sentences) - 1 else "conclusion",
                confidence=0.7 if len(sentence) > 30 else 0.5,
            )
            claims.append(claim)
            analysis.extracted_claims.append(claim)

        # Build structure
        structure = ArgumentStructure(argument_id=analysis_id)
        structure.premises = [c for c in claims if c.type == "premise"]
        structure.conclusion = claims[-1] if claims else None

        # Detect fallacies (simplified)
        fallacies = []
        lower_text = text.lower()
        if "ad hominem" in lower_text or "you are" in lower_text and "because" in lower_text:
            fallacies.append("ad_hominem")
        if "all " in lower_text and "always" in lower_text:
            fallacies.append("hasty_generalization")
        if "because everyone" in lower_text:
            fallacies.append("bandwagon")
        analysis.logical_fallacies = fallacies

        # Assess quality
        quality = 50.0
        if structure.premises:
            quality += len(structure.premises) * 10.0
        if structure.conclusion:
            quality += 15.0
        if fallacies:
            quality -= len(fallacies) * 15.0
        structure.overall_quality = max(0.0, min(100.0, quality))

        if len(structure.premises) > 2:
            structure.strengths.append("Multiple premises supporting conclusion")
        if structure.conclusion and structure.premises:
            structure.strengths.append("Clear conclusion with supporting premises")
        if fallacies:
            structure.weaknesses.append("Contains logical fallacies")
        if not structure.premises:
            structure.weaknesses.append("Missing supporting premises")

        analysis.structure = structure
        self._analyses[analysis_id] = analysis
        return analysis

    def get_analysis(self, analysis_id: str) -> dict[str, Any]:
        """Get an analysis by ID."""
        analysis = self._analyses.get(analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return {
            "analysis_id": analysis.analysis_id,
            "claim_count": len(analysis.extracted_claims),
            "fallacies": analysis.logical_fallacies,
            "quality": analysis.structure.overall_quality if analysis.structure else 0,
            "strengths": analysis.structure.strengths if analysis.structure else [],
            "weaknesses": analysis.structure.weaknesses if analysis.structure else [],
        }

    def get_report(self) -> dict[str, Any]:
        """Get engine report."""
        avg_quality = sum(
            a.structure.overall_quality for a in self._analyses.values() if a.structure
        ) / max(1, len(self._analyses))
        return {
            "total_analyses": len(self._analyses),
            "average_quality": round(avg_quality, 1),
            "engine_status": "operational",
        }


engine = ArgumentAnalysisEngine()


@router.post("/inference/arguments/analyze")
async def analyze_argument(text: str):
    """Analyze an argument."""
    analysis = engine.analyze_argument(text)
    return {
        "analysis_id": analysis.analysis_id,
        "claim_count": len(analysis.extracted_claims),
        "fallacies": analysis.logical_fallacies,
        "quality": analysis.structure.overall_quality if analysis.structure else 0,
        "strengths": analysis.structure.strengths if analysis.structure else [],
        "weaknesses": analysis.structure.weaknesses if analysis.structure else [],
    }


@router.get("/inference/arguments/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get an argument analysis."""
    return engine.get_analysis(analysis_id)


@router.get("/inference/arguments/report")
async def argument_report():
    """Get argument analysis report."""
    return engine.get_report()
