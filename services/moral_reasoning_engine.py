"""
Inference/services/moral_reasoning_engine.py — Inference Moral Reasoning Engine.

Ethical reasoning, moral principle application, deontic logic,
utilitarian calculation, and virtue ethics evaluation.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()


@dataclass
class MoralPrinciple:
    """A moral principle."""
    principle_id: str
    name: str
    framework: str = "deontology"  # deontology, utilitarianism, virtue, rights
    description: str = ""
    weight: float = 1.0


@dataclass
class EthicalDilemma:
    """An ethical dilemma."""
    dilemma_id: str
    description: str
    options: list[str] = field(default_factory=list)
    stakeholders: list[str] = field(default_factory=list)
    context: str = ""


@dataclass
class MoralEvaluation:
    """A moral evaluation of a dilemma."""
    evaluation_id: str
    dilemma_id: str
    recommended_option: str = ""
    rationale: str = ""
    principle_scores: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5
    evaluated_at: float = field(default_factory=time.time)


class MoralReasoningEngine:
    """Engine for moral and ethical reasoning."""

    def __init__(self) -> None:
        self._principles: dict[str, MoralPrinciple] = {}
        self._dilemmas: dict[str, EthicalDilemma] = {}
        self._evaluations: list[MoralEvaluation] = []

    def add_principle(self, name: str, framework: str = "deontology",
                      description: str = "", weight: float = 1.0) -> MoralPrinciple:
        """Add a moral principle."""
        principle_id = f"prin_{uuid.uuid4().hex[:12]}"
        principle = MoralPrinciple(
            principle_id=principle_id, name=name, framework=framework,
            description=description, weight=weight,
        )
        self._principles[principle_id] = principle
        return principle

    def create_dilemma(self, description: str, options: list[str],
                       stakeholders: list[str] | None = None,
                       context: str = "") -> EthicalDilemma:
        """Create an ethical dilemma."""
        dilemma_id = f"dil_{uuid.uuid4().hex[:12]}"
        dilemma = EthicalDilemma(
            dilemma_id=dilemma_id, description=description,
            options=options, stakeholders=stakeholders or [], context=context,
        )
        self._dilemmas[dilemma_id] = dilemma
        return dilemma

    def evaluate(self, dilemma_id: str) -> MoralEvaluation:
        """Evaluate an ethical dilemma."""
        dilemma = self._dilemmas.get(dilemma_id)
        if not dilemma:
            raise HTTPException(status_code=404, detail="Dilemma not found")

        evaluation_id = f"eval_{uuid.uuid4().hex[:12]}"
        evaluation = MoralEvaluation(evaluation_id=evaluation_id, dilemma_id=dilemma_id)

        # Score each option based on principles
        for option in dilemma.options:
            score = 0.0
            for principle in self._principles.values():
                if principle.framework == "utilitarianism":
                    score += 0.3 * principle.weight * len(dilemma.stakeholders)
                elif principle.framework == "deontology":
                    score += 0.3 * principle.weight
                elif principle.framework == "virtue":
                    score += 0.2 * principle.weight
                elif principle.framework == "rights":
                    score += 0.2 * principle.weight
            evaluation.principle_scores[option] = score

        best_option = max(evaluation.principle_scores, key=evaluation.principle_scores.get)
        evaluation.recommended_option = best_option
        evaluation.rationale = f"Selected '{best_option}' based on principle analysis across {len(self._principles)} frameworks"
        evaluation.confidence = min(0.9, len(evaluation.principle_scores) / 10)

        self._evaluations.append(evaluation)
        return evaluation

    def get_dilemmas(self) -> list[dict[str, Any]]:
        """Get ethical dilemmas."""
        return [
            {"description": d.description[:80], "option_count": len(d.options),
             "stakeholder_count": len(d.stakeholders)}
            for d in self._dilemmas.values()
        ]

    def get_evaluations(self, dilemma_id: str | None = None) -> list[dict[str, Any]]:
        """Get moral evaluations."""
        evaluations = self._evaluations
        if dilemma_id:
            evaluations = [e for e in evaluations if e.dilemma_id == dilemma_id]
        return [
            {"dilemma_id": e.dilemma_id, "recommended": e.recommended_option,
             "confidence": e.confidence, "principle_count": len(e.principle_scores)}
            for e in evaluations
        ]

    def get_report(self) -> dict[str, Any]:
        """Get engine report."""
        return {
            "total_principles": len(self._principles),
            "total_dilemmas": len(self._dilemmas),
            "total_evaluations": len(self._evaluations),
            "engine_status": "operational",
        }


engine = MoralReasoningEngine()


@router.post("/inference/moral/principles")
async def add_principle(name: str, framework: str = "deontology",
                        description: str = "", weight: float = 1.0):
    """Add a moral principle."""
    principle = engine.add_principle(name, framework, description, weight)
    return {"principle_id": principle.principle_id, "framework": principle.framework}


@router.post("/inference/moral/dilemmas")
async def create_dilemma(description: str, options: list[str],
                         stakeholders: list[str] | None = None, context: str = ""):
    """Create an ethical dilemma."""
    dilemma = engine.create_dilemma(description, options, stakeholders, context)
    return {"dilemma_id": dilemma.dilemma_id, "option_count": len(dilemma.options)}


@router.post("/inference/moral/evaluate/{dilemma_id}")
async def evaluate(dilemma_id: str):
    """Evaluate an ethical dilemma."""
    evaluation = engine.evaluate(dilemma_id)
    return {
        "evaluation_id": evaluation.evaluation_id,
        "recommended_option": evaluation.recommended_option,
        "confidence": evaluation.confidence,
        "rationale": evaluation.rationale,
    }


@router.get("/inference/moral/dilemmas")
async def get_dilemmas():
    """Get ethical dilemmas."""
    return {"dilemmas": engine.get_dilemmas()}


@router.get("/inference/moral/evaluations")
async def get_evaluations(dilemma_id: str | None = None):
    """Get moral evaluations."""
    return {"evaluations": engine.get_evaluations(dilemma_id)}


@router.get("/inference/moral/report")
async def moral_report():
    """Get moral reasoning engine report."""
    return engine.get_report()
