"""
Inference/services/multi_model_deliberation_engine.py — Inference Multi-Model Deliberation Engine.

Multi-model reasoning with consensus building, model comparison, and
confidence aggregation.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()


@dataclass
class ModelReasoning:
    """Reasoning output from a single model."""
    model_name: str
    reasoning_id: str
    reasoning: str
    confidence: float = 0.5
    tokens_used: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConsensusResult:
    """Result of consensus building across models."""
    consensus_id: str
    topic: str
    reasonings: list[ModelReasoning] = field(default_factory=list)
    consensus_answer: str = ""
    agreement_score: float = 0.0
    dissenting_models: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class DeliberationSession:
    """A deliberation session with multiple models."""
    session_id: str
    topic: str
    models: list[str] = field(default_factory=list)
    status: str = "pending"
    consensus: ConsensusResult | None = None
    started_at: float = field(default_factory=time.time)


class MultiModelDeliberationEngine:
    """Engine for multi-model deliberation and consensus."""

    def __init__(self) -> None:
        self._sessions: dict[str, DeliberationSession] = {}
        self._reasonings: dict[str, ModelReasoning] = {}

    def start_deliberation(self, topic: str, models: list[str]) -> str:
        """Start a deliberation session."""
        session_id = f"delim_{uuid.uuid4().hex[:12]}"
        session = DeliberationSession(session_id=session_id, topic=topic, models=models)
        self._sessions[session_id] = session
        return session_id

    def submit_reasoning(self, session_id: str, model_name: str,
                         reasoning: str, confidence: float = 0.5) -> str:
        """Submit reasoning from a model."""
        session = self._sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        reasoning_id = f"reason_{uuid.uuid4().hex[:12]}"
        model_reasoning = ModelReasoning(
            model_name=model_name,
            reasoning_id=reasoning_id,
            reasoning=reasoning,
            confidence=confidence,
            tokens_used=len(reasoning) // 4,
        )
        self._reasonings[reasoning_id] = model_reasoning
        session.status = "deliberating"
        return reasoning_id

    def build_consensus(self, session_id: str) -> ConsensusResult:
        """Build consensus from all submitted reasonings."""
        session = self._sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        session_reasonings = [
            r for r in self._reasonings.values()
            if r.model_name in session.models
        ]
        if not session_reasonings:
            raise HTTPException(status_code=400, detail="No reasonings submitted")

        consensus_id = f"cons_{uuid.uuid4().hex[:12]}"
        consensus = ConsensusResult(consensus_id=consensus_id, topic=session.topic)
        consensus.reasonings = session_reasonings

        # Calculate agreement score
        avg_confidence = sum(r.confidence for r in session_reasonings) / len(session_reasonings)
        consensus.agreement_score = avg_confidence

        # Find dissenting models
        mean_reasoning = session_reasonings[len(session_reasonings) // 2].reasoning
        consensus.dissenting_models = [
            r.model_name for r in session_reasonings
            if r.confidence < avg_confidence - 0.2
        ]

        # Build consensus answer
        consensus.consensus_answer = f"Based on {len(session_reasonings)} model reasonings, the consensus is: {mean_reasoning[:200]}"

        session.consensus = consensus
        session.status = "completed"
        return consensus

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Get a deliberation session."""
        session = self._sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "session_id": session.session_id,
            "topic": session.topic,
            "models": session.models,
            "status": session.status,
            "consensus": {
                "answer": session.consensus.consensus_answer[:100],
                "agreement_score": session.consensus.agreement_score,
                "dissenting_models": session.consensus.dissenting_models,
            } if session.consensus else None,
        }

    def get_report(self) -> dict[str, Any]:
        """Get engine report."""
        return {
            "total_sessions": len(self._sessions),
            "total_reasonings": len(self._reasonings),
            "completed_sessions": sum(1 for s in self._sessions.values() if s.status == "completed"),
            "engine_status": "operational",
        }


engine = MultiModelDeliberationEngine()


@router.post("/inference/deliberation/sessions")
async def start_deliberation(topic: str, models: list[str]):
    """Start a multi-model deliberation session."""
    session_id = engine.start_deliberation(topic, models)
    return {"session_id": session_id, "status": "started"}


@router.post("/inference/deliberation/sessions/{session_id}/reasoning")
async def submit_reasoning(session_id: str, model_name: str, reasoning: str, confidence: float = 0.5):
    """Submit model reasoning."""
    rid = engine.submit_reasoning(session_id, model_name, reasoning, confidence)
    return {"reasoning_id": rid, "status": "submitted"}


@router.post("/inference/deliberation/sessions/{session_id}/consensus")
async def build_consensus(session_id: str):
    """Build consensus from reasonings."""
    consensus = engine.build_consensus(session_id)
    return {
        "consensus_id": consensus.consensus_id,
        "answer": consensus.consensus_answer[:200],
        "agreement_score": consensus.agreement_score,
        "dissenting_models": consensus.dissenting_models,
    }


@router.get("/inference/deliberation/sessions/{session_id}")
async def get_session(session_id: str):
    """Get deliberation session."""
    return engine.get_session(session_id)


@router.get("/inference/deliberation/report")
async def deliberation_report():
    """Get deliberation report."""
    return engine.get_report()
