"""
Inference/services/multi_agent_argumentation_engine.py — Inference Multi-Agent Argumentation Engine.

Multi-agent argumentation, debate simulation, argument evaluation,
and consensus formation through structured argumentation.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()


@dataclass
class Argument:
    """A formal argument."""
    argument_id: str
    claim: str
    premises: list[str] = field(default_factory=list)
    conclusion: str = ""
    agent_id: str = ""
    strength: float = 0.5
    created_at: float = field(default_factory=time.time)


@dataclass
class DebateTurn:
    """A turn in a debate."""
    turn_id: int
    agent_id: str
    argument: str
    response_to: str = ""
    attack_type: str = ""  # rebuttal, challenge, support, question
    created_at: float = field(default_factory=time.time)


@dataclass
class DebateSession:
    """A debate session."""
    session_id: str
    topic: str
    agent_ids: list[str] = field(default_factory=list)
    turns: list[DebateTurn] = field(default_factory=list)
    consensus: str = ""
    consensus_confidence: float = 0.0
    status: str = "active"  # active, concluded
    created_at: float = field(default_factory=time.time)


class MultiAgentArgumentationEngine:
    """Engine for multi-agent argumentation and debate."""

    def __init__(self) -> None:
        self._arguments: dict[str, Argument] = {}
        self._debates: dict[str, DebateSession] = {}

    def register_agent(self, agent_id: str) -> str:
        """Register an agent for argumentation."""
        return agent_id

    def submit_argument(self, claim: str, premises: list[str] = None,
                       conclusion: str = "", agent_id: str = "",
                       strength: float = 0.5) -> str:
        """Submit a formal argument."""
        argument_id = f"arg_{uuid.uuid4().hex[:12]}"
        argument = Argument(
            argument_id=argument_id, claim=claim,
            premises=premises or [], conclusion=conclusion,
            agent_id=agent_id, strength=strength,
        )
        self._arguments[argument_id] = argument
        return argument_id

    def start_debate(self, topic: str, agent_ids: list[str]) -> DebateSession:
        """Start a debate session."""
        session_id = f"deb_{uuid.uuid4().hex[:12]}"
        session = DebateSession(session_id=session_id, topic=topic, agent_ids=agent_ids)
        self._debates[session_id] = session
        return session

    def add_turn(self, session_id: str, agent_id: str, argument: str,
                response_to: str = "", attack_type: str = "") -> DebateTurn:
        """Add a turn to a debate."""
        session = self._debates.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Debate not found")

        turn = DebateTurn(
            turn_id=len(session.turns) + 1,
            agent_id=agent_id, argument=argument,
            response_to=response_to, attack_type=attack_type,
        )
        session.turns.append(turn)
        return turn

    def conclude_debate(self, session_id: str, consensus: str,
                       confidence: float = 0.5) -> DebateSession:
        """Conclude a debate with a consensus."""
        session = self._debates.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Debate not found")
        session.consensus = consensus
        session.consensus_confidence = confidence
        session.status = "concluded"
        return session

    def evaluate_argument(self, argument_id: str) -> dict[str, Any]:
        """Evaluate the strength of an argument."""
        argument = self._arguments.get(argument_id)
        if not argument:
            raise HTTPException(status_code=404, detail="Argument not found")
        return {
            "argument_id": argument.argument_id,
            "claim": argument.claim[:80],
            "premise_count": len(argument.premises),
            "strength": argument.strength,
            "evaluation": f"Argument has {len(argument.premises)} premises with strength {argument.strength}",
        }

    def get_report(self) -> dict[str, Any]:
        """Get engine report."""
        return {
            "total_arguments": len(self._arguments),
            "total_debates": len(self._debates),
            "active_debates": sum(1 for d in self._debates.values() if d.status == "active"),
            "engine_status": "operational",
        }


engine = MultiAgentArgumentationEngine()


@router.post("/inference/argumentation/arguments")
async def submit_argument(claim: str, premises: list[str] | None = None,
                          conclusion: str = "", agent_id: str = "", strength: float = 0.5):
    """Submit a formal argument."""
    argument_id = engine.submit_argument(claim, premises, conclusion, agent_id, strength)
    return {"argument_id": argument_id, "strength": strength}


@router.post("/inference/argumentation/debates")
async def start_debate(topic: str, agent_ids: list[str]):
    """Start a debate session."""
    session = engine.start_debate(topic, agent_ids)
    return {"session_id": session.session_id, "topic": session.topic, "agent_count": len(agent_ids)}


@router.post("/inference/argumentation/debates/{session_id}/turns")
async def add_turn(session_id: str, agent_id: str, argument: str,
                   response_to: str = "", attack_type: str = ""):
    """Add a turn to a debate."""
    turn = engine.add_turn(session_id, agent_id, argument, response_to, attack_type)
    return {"turn_id": turn.turn_id, "agent_id": turn.agent_id}


@router.post("/inference/argumentation/debates/{session_id}/conclude")
async def conclude_debate(session_id: str, consensus: str, confidence: float = 0.5):
    """Conclude a debate."""
    session = engine.conclude_debate(session_id, consensus, confidence)
    return {"session_id": session.session_id, "consensus": session.consensus, "confidence": confidence}


@router.get("/inference/argumentation/arguments/{argument_id}/evaluate")
async def evaluate_argument(argument_id: str):
    """Evaluate an argument."""
    return engine.evaluate_argument(argument_id)


@router.get("/inference/argumentation/report")
async def argumentation_report():
    """Get argumentation engine report."""
    return engine.get_report()
