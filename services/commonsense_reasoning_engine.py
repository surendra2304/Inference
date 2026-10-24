"""
Inference/services/commonsense_reasoning_engine.py — Inference Commonsense Reasoning Engine.

Commonsense knowledge application, everyday reasoning, causal
inference from experience, and implicit knowledge filling.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()


@dataclass
class CommonsenseFact:
    """A commonsense fact."""
    fact_id: str
    subject: str
    predicate: str
    object: str
    confidence: float = 0.5
    source: str = "general"  # general, experience, learned
    created_at: float = field(default_factory=time.time)


@dataclass
class ReasoningStep:
    """A step in commonsense reasoning."""
    step_id: int
    premise: str
    conclusion: str
    justification: str = ""
    confidence: float = 0.5


@dataclass
class ReasoningResult:
    """Result of commonsense reasoning."""
    result_id: str
    question: str
    answer: str = ""
    steps: list[ReasoningStep] = field(default_factory=list)
    confidence: float = 0.5
    reasoning_type: str = "commonsense"


class CommonsenseReasoningEngine:
    """Engine for commonsense reasoning and implicit knowledge."""

    def __init__(self) -> None:
        self._facts: dict[str, CommonsenseFact] = {}
        self._results: list[ReasoningResult] = []

    def add_fact(self, subject: str, predicate: str, object: str,
                confidence: float = 0.5, source: str = "general") -> str:
        """Add a commonsense fact."""
        fact_id = f"fact_{uuid.uuid4().hex[:12]}"
        fact = CommonsenseFact(
            fact_id=fact_id, subject=subject, predicate=predicate,
            object=object, confidence=confidence, source=source,
        )
        self._facts[fact_id] = fact
        return fact_id

    def reason(self, question: str) -> ReasoningResult:
        """Apply commonsense reasoning to a question."""
        result_id = f"reason_{uuid.uuid4().hex[:12]}"
        result = ReasoningResult(result_id=result_id, question=question)

        # Find relevant facts
        words = question.lower().split()
        relevant_facts = []
        for fact in self._facts.values():
            if fact.subject.lower() in words or fact.object.lower() in words:
                relevant_facts.append(fact)

        # Generate reasoning steps
        for i, fact in enumerate(relevant_facts[:3]):
            step = ReasoningStep(
                step_id=i + 1,
                premise=f"{fact.subject} {fact.predicate} {fact.object}",
                conclusion=f"Related to question: {question[:50]}",
                justification=f"Fact from {fact.source} with confidence {fact.confidence}",
                confidence=fact.confidence,
            )
            result.steps.append(step)

        # Generate answer
        if relevant_facts:
            top_fact = max(relevant_facts, key=lambda f: f.confidence)
            result.answer = f"Based on commonsense knowledge, {top_fact.subject} {top_fact.predicate} {top_fact.object}"
            result.confidence = top_fact.confidence
        else:
            result.answer = "No relevant commonsense knowledge found"
            result.confidence = 0.2

        self._results.append(result)
        return result

    def get_facts(self, subject: str | None = None) -> list[dict[str, Any]]:
        """Get commonsense facts."""
        facts = list(self._facts.values())
        if subject:
            facts = [f for f in facts if f.subject.lower() == subject.lower()]
        return [
            {"subject": f.subject, "predicate": f.predicate, "object": f.object,
             "confidence": f.confidence, "source": f.source}
            for f in facts
        ]

    def get_report(self) -> dict[str, Any]:
        """Get engine report."""
        return {
            "total_facts": len(self._facts),
            "total_reasonings": len(self._results),
            "engine_status": "operational",
        }


engine = CommonsenseReasoningEngine()


@router.post("/inference/commonsense/facts")
async def add_fact(subject: str, predicate: str, object: str, confidence: float = 0.5, source: str = "general"):
    """Add a commonsense fact."""
    fact_id = engine.add_fact(subject, predicate, object, confidence, source)
    return {"fact_id": fact_id, "status": "added"}


@router.post("/inference/commonsense/reason")
async def reason(question: str):
    """Apply commonsense reasoning."""
    result = engine.reason(question)
    return {
        "result_id": result.result_id,
        "answer": result.answer,
        "confidence": result.confidence,
        "steps": [{"premise": s.premise, "conclusion": s.conclusion} for s in result.steps],
    }


@router.get("/inference/commonsense/facts")
async def get_facts(subject: str | None = None):
    """Get commonsense facts."""
    return {"facts": engine.get_facts(subject)}


@router.get("/inference/commonsense/report")
async def commonsense_report():
    """Get commonsense reasoning report."""
    return engine.get_report()
