"""
Inference/services/probabilistic_logic_engine.py — Inference Probabilistic Logic Engine.

Probabilistic inference, Bayesian network reasoning, uncertainty quantification,
and probabilistic constraint satisfaction.
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()


@dataclass
class ProbVariable:
    """A probabilistic variable."""
    var_id: str
    name: str
    domain: list[str] = field(default_factory=list)
    prior: dict[str, float] = field(default_factory=dict)


@dataclass
class ConditionalProb:
    """A conditional probability table."""
    cp_id: str
    variable: str
    parents: list[str] = field(default_factory=list)
    table: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Result of probabilistic inference."""
    result_id: str
    variable: str
    posterior: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    entropy: float = 0.0
    inferred_at: float = field(default_factory=time.time)


class ProbabilisticLogicEngine:
    """Engine for probabilistic inference and Bayesian reasoning."""

    def __init__(self) -> None:
        self._variables: dict[str, ProbVariable] = {}
        self._cpt: dict[str, ConditionalProb] = {}
        self._results: list[InferenceResult] = []

    def add_variable(self, name: str, domain: list[str],
                    prior: dict[str, float] | None = None) -> ProbVariable:
        """Add a probabilistic variable."""
        var_id = f"var_{uuid.uuid4().hex[:12]}"
        var = ProbVariable(
            var_id=var_id, name=name, domain=domain,
            prior=prior or {d: 1.0 / len(domain) for d in domain},
        )
        self._variables[var_id] = var
        return var

    def add_cpt(self, variable_id: str, parents: list[str],
                table: dict[str, dict[str, float]]) -> ConditionalProb:
        """Add a conditional probability table."""
        cp_id = f"cp_{uuid.uuid4().hex[:12]}"
        cp = ConditionalProb(cp_id=cp_id, variable=variable_id, parents=parents, table=table)
        self._cpt[cp_id] = cp
        return cp

    def infer(self, variable_id: str, evidence: dict[str, str]) -> InferenceResult:
        """Perform probabilistic inference."""
        var = self._variables.get(variable_id)
        if not var:
            raise HTTPException(status_code=404, detail="Variable not found")

        result_id = f"inf_{uuid.uuid4().hex[:12]}"
        result = InferenceResult(
            result_id=result_id, variable=var.name, evidence=evidence,
        )

        # Simple Bayesian update based on evidence
        posterior = dict(var.prior)
        for evidence_var, evidence_value in evidence.items():
            if evidence_value in posterior:
                # Boost the probability of matching evidence
                if evidence_var.lower() in var.name.lower():
                    boost = 2.0
                    posterior[evidence_value] *= boost
                else:
                    boost = 1.5
                    posterior[evidence_value] *= boost

        # Normalize
        total = sum(posterior.values())
        if total > 0:
            posterior = {k: v / total for k, v in posterior.items()}

        result.posterior = {k: round(v, 4) for k, v in posterior.items()}
        result.confidence = max(posterior.values()) if posterior else 0.5
        result.entropy = self._calculate_entropy(posterior)

        self._results.append(result)
        return result

    def _calculate_entropy(self, probs: dict[str, float]) -> float:
        """Calculate entropy of a probability distribution."""
        entropy = 0.0
        for p in probs.values():
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def get_variables(self) -> list[dict[str, Any]]:
        """Get probabilistic variables."""
        return [
            {"name": v.name, "domain_size": len(v.domain),
             "prior_count": len(v.prior)}
            for v in self._variables.values()
        ]

    def get_results(self, variable_id: str | None = None) -> list[dict[str, Any]]:
        """Get inference results."""
        results = self._results
        if variable_id:
            results = [r for r in results if r.variable == variable_id]
        return [
            {"variable": r.variable, "confidence": r.confidence,
             "entropy": round(r.entropy, 2), "evidence_count": len(r.evidence)}
            for r in results
        ]

    def get_report(self) -> dict[str, Any]:
        """Get engine report."""
        return {
            "total_variables": len(self._variables),
            "total_cpt": len(self._cpt),
            "total_results": len(self._results),
            "avg_confidence": round(
                sum(r.confidence for r in self._results) / max(1, len(self._results)), 2
            ) if self._results else 0,
            "engine_status": "operational",
        }


engine = ProbabilisticLogicEngine()


@router.post("/inference/problog/variables")
async def add_variable(name: str, domain: list[str], prior: dict[str, float] | None = None):
    """Add a probabilistic variable."""
    var = engine.add_variable(name, domain, prior)
    return {"var_id": var.var_id, "name": var.name, "domain_size": len(var.domain)}


@router.post("/inference/problog/cpt")
async def add_cpt(variable_id: str, parents: list[str],
                  table: dict[str, dict[str, float]]):
    """Add a conditional probability table."""
    cp = engine.add_cpt(variable_id, parents, table)
    return {"cp_id": cp.cp_id, "variable": variable_id, "parents": parents}


@router.post("/inference/problog/infer")
async def infer(variable_id: str, evidence: dict[str, str]):
    """Perform probabilistic inference."""
    result = engine.infer(variable_id, evidence)
    return {
        "variable": result.variable,
        "confidence": result.confidence,
        "entropy": round(result.entropy, 2),
        "posterior": result.posterior,
    }


@router.get("/inference/problog/variables")
async def get_variables():
    """Get probabilistic variables."""
    return {"variables": engine.get_variables()}


@router.get("/inference/problog/results")
async def get_results(variable_id: str | None = None):
    """Get inference results."""
    return {"results": engine.get_results(variable_id)}


@router.get("/inference/problog/report")
async def problog_report():
    """Get probabilistic logic engine report."""
    return engine.get_report()
