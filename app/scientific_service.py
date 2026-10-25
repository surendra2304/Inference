"""
Inference/app/services/scientific_service.py — Inference Research Capability.

Inference is the reasoning agent; research questions reach it through
``/v1/agent/assist`` and through the reasoning routers. This service turns the
scientific-skills engine into an actual Inference capability:

* ``ScientificResearchService.plan``      — which of the 51 research skills apply,
  in what order, and which databases they hit.
* ``.analyse``                            — run the right statistical test on the
  supplied groups and report effect sizes.
* ``.power``                              — sample-size planning for a study.
* ``.hypotheses``                          — evidence-weighted hypotheses with
  falsification tests.
* ``.audit`` / ``.render``                 — evidence-traceable writing gate.
* ``.capabilities``                        — machine-readable capability card.

``router`` is mounted by ``Inference/app/main.py`` and ``Inference``'s
inter-agent gateway exposes ``/v1/agent/research-plan`` so FRIDAY, IntelX and
Cortex can ask for a research plan before calling a model.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Iterable

_MODULE_PATH = Path(__file__).resolve().parent / "agents" / "real_scientific_skills.py"

if "inference_scientific_skills" in sys.modules:
    sci = sys.modules["inference_scientific_skills"]
else:
    _spec = importlib.util.spec_from_file_location("inference_scientific_skills", _MODULE_PATH)
    sci = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
    sys.modules["inference_scientific_skills"] = sci
    _spec.loader.exec_module(sci)  # type: ignore[union-attr]

try:  # pragma: no cover - only when FastAPI is installed
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field

    _ROUTER_AVAILABLE = True
except Exception:  # pragma: no cover
    APIRouter = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment]

    def Field(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[misc]
        return None

    _ROUTER_AVAILABLE = False


class ScientificResearchService:
    """Research skill routing, statistics, design and evidence auditing."""

    def __init__(self, engine: Any | None = None) -> None:
        self.engine = engine or sci.ScientificAgentEngine()

    # -- skills -----------------------------------------------------------
    def skills(self, query: str = "", limit: int = 10) -> dict[str, Any]:
        if query.strip():
            matches = self.engine.library.search(query, limit=limit)
            return {
                "query": query,
                "matches": [
                    {"skill": match.skill.skill_id, "title": match.skill.title,
                     "category": match.skill.category, "score": match.score,
                     "databases": list(match.skill.databases)}
                    for match in matches
                ],
            }
        report = self.engine.library.coverage_report()
        report["skills"] = len(self.engine.library)
        return report

    def plan(self, task: str, limit: int = 3) -> dict[str, Any]:
        if not task.strip():
            raise ValueError("task must not be empty")
        plan = self.engine.plan(task, limit=limit)
        library = self.engine.library.skills
        plan["databases"] = sorted({
            database
            for skill in plan["skills"]
            if skill["skill"] in library
            for database in library[skill["skill"]].databases
        })
        return plan

    def gaps(self, task: str, threshold: float = 0.15) -> list[str]:
        return self.engine.matcher.gaps(task, threshold)

    # -- analysis ---------------------------------------------------------
    def analyse(self, groups: Iterable[Iterable[float]], alpha: float = 0.05) -> dict[str, Any]:
        clean = [[float(value) for value in group] for group in groups]
        if len(clean) < 2:
            raise ValueError("at least two groups are required")
        if any(len(group) < 2 for group in clean):
            raise ValueError("each group needs at least two observations")
        result = self.engine.analyse_groups(clean, alpha)
        return result

    def power(self, effect_size: float, alpha: float = 0.05, power: float = 0.8,
              sample_size: int | None = None) -> dict[str, Any]:
        if effect_size <= 0:
            raise ValueError("effect_size must be positive")
        if not 0 < alpha < 1 or not 0 < power < 1:
            raise ValueError("alpha and power must be in (0, 1)")
        plan = self.engine.designer.power_plan(effect_size, alpha, power)
        if sample_size is not None and sample_size >= 2:
            n_per_group = max(2, sample_size // 2)
            achieved = self.engine.designer.power.power_two_sample(effect_size, n_per_group, alpha)
            plan["actual_sample_size"] = sample_size
            plan["n_per_group"] = n_per_group
            plan["total_units"] = sample_size
            plan["power"] = round(achieved, 4)
            plan["achieved_power"] = round(achieved, 4)
        return plan

    def hypotheses(self, measurement: str, mechanism: str,
                   observations: Iterable[dict[str, Any]],
                   prior: float = 0.3) -> list[dict[str, Any]]:
        if not measurement.strip() or not mechanism.strip():
            raise ValueError("measurement and mechanism are required")
        prepared = [
            sci.Observation(
                statement=str(item.get("description") or item.get("statement") or f"observation {index}"),
                likelihood_ratio=(float(item.get("likelihood_ratio", item.get("weight", 2.0)))
                                  if item.get("supports", True)
                                  else 1.0 / max(1e-6, float(item.get("likelihood_ratio", item.get("weight", 2.0))))),
                source=str(item.get("source", "")),
            )
            for index, item in enumerate(observations, start=1)
        ]
        if not prepared:
            raise ValueError("at least one observation is required")
        hypotheses = self.engine.hypotheses.generate(
            measurement, mechanism, prepared, prior=prior)
        return [hypothesis.as_dict() for hypothesis in hypotheses]

    # -- writing ----------------------------------------------------------
    def audit(self, claims: Iterable[dict[str, Any]], evidence: Iterable[dict[str, Any]] = ()
              ) -> dict[str, Any]:
        writer = sci.ScientificWriter()
        for record in evidence:
            writer.add_evidence(sci.EvidenceRecord(
                evidence_id=str(record.get("id") or record.get("evidence_id") or ""),
                title=str(record.get("title", "")),
                source=str(record.get("source", "")),
                value=str(record.get("value", "")),
            ))
        prepared = [
            sci.Claim(text=str(claim["text"]),
                      evidence_ids=[str(item) for item in claim.get("evidence", [])])
            for claim in claims
        ]
        report = writer.audit(prepared)
        report["rendered"] = writer.render("Evidence audit", prepared)
        unsupported = report.get("unsupported") or []
        unsupported_count = len(unsupported) if isinstance(unsupported, list) else int(unsupported)
        report["verdict"] = "pass" if unsupported_count == 0 else "warn"
        warnings = []
        if unsupported_count > 0:
            warnings.append(f"{unsupported_count} claim(s) lack citation evidence")
        report["warnings"] = warnings
        return report

    # -- capability card --------------------------------------------------
    def capabilities(self) -> dict[str, Any]:
        coverage = self.engine.library.coverage_report()
        return {
            "agent": "Inference",
            "capabilities": [
                "research_skill_routing", "statistical_analysis", "power_analysis",
                "hypothesis_generation", "evidence_audit", "experiment_design",
            ],
            "skills": len(self.engine.library),
            "categories": coverage.get("categories", len(coverage.get("per_category", {}))),
            "databases": coverage.get("databases", []),
        }


_service: ScientificResearchService | None = None


def get_service() -> ScientificResearchService:
    global _service
    if _service is None:
        _service = ScientificResearchService()
    return _service


def reset_service(service: ScientificResearchService | None = None) -> ScientificResearchService:
    global _service
    _service = service or ScientificResearchService()
    return _service


# ── REST surface ──────────────────────────────────────────────────────────

if _ROUTER_AVAILABLE:

    class SkillQuery(BaseModel):  # type: ignore[misc]
        query: str = ""
        limit: int = Field(default=10, ge=1, le=51)

    class PlanRequest(BaseModel):  # type: ignore[misc]
        task: str = Field(min_length=1)
        limit: int = Field(default=3, ge=1, le=8)

    class AnalyseRequest(BaseModel):  # type: ignore[misc]
        groups: list[list[float]] = Field(min_length=2)
        alpha: float = Field(default=0.05, gt=0.0, lt=1.0)

    class PowerRequest(BaseModel):  # type: ignore[misc]
        effect_size: float = Field(gt=0.0)
        alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
        power: float = Field(default=0.8, gt=0.0, lt=1.0)
        sample_size: int | None = Field(default=None, ge=2)

    class ObservationModel(BaseModel):  # type: ignore[misc]
        id: str = "o1"
        description: str = ""
        supports: bool = True
        weight: float = 1.0

    class HypothesisRequest(BaseModel):  # type: ignore[misc]
        measurement: str = Field(min_length=1)
        mechanism: str = Field(min_length=1)
        observations: list[ObservationModel] = Field(min_length=1)
        prior: float = Field(default=0.3, gt=0.0, lt=1.0)

    class ClaimModel(BaseModel):  # type: ignore[misc]
        text: str = Field(min_length=1)
        evidence: list[str] = Field(default_factory=list)

    class EvidenceModel(BaseModel):  # type: ignore[misc]
        id: str = Field(min_length=1)
        title: str = ""
        source: str = ""

    class AuditRequest(BaseModel):  # type: ignore[misc]
        claims: list[ClaimModel] = Field(min_length=1)
        evidence: list[EvidenceModel] = Field(default_factory=list)

    router = APIRouter(prefix="/v1/scientific", tags=["Inference Scientific Skills"])

    @router.get("/capabilities")
    def capabilities() -> dict[str, Any]:
        return get_service().capabilities()

    @router.get("/skills")
    def skills(query: str = "", limit: int = 10) -> dict[str, Any]:
        return get_service().skills(query, limit)

    @router.post("/plan")
    def plan(request: PlanRequest) -> dict[str, Any]:
        try:
            return get_service().plan(request.task, request.limit)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/analyse")
    def analyse(request: AnalyseRequest) -> dict[str, Any]:
        try:
            return get_service().analyse(request.groups, request.alpha)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/power")
    def power(request: PowerRequest) -> dict[str, Any]:
        try:
            return get_service().power(request.effect_size, request.alpha, request.power, request.sample_size)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/hypotheses")
    def hypotheses(request: HypothesisRequest) -> dict[str, Any]:
        try:
            return {"hypotheses": get_service().hypotheses(
                request.measurement, request.mechanism,
                [item.model_dump() for item in request.observations], request.prior)}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/audit")
    def audit(request: AuditRequest) -> dict[str, Any]:
        try:
            return get_service().audit(
                [claim.model_dump() for claim in request.claims],
                [record.model_dump() for record in request.evidence],
            )
        except Exception as exc:
            return {
                "verdict": "fail",
                "warnings": [f"Evidence audit exception: {exc}"],
                "claims": len(request.claims),
                "unsupported": len(request.claims),
                "supported": 0,
            }

else:  # pragma: no cover
    router = None


__all__ = [
    "ScientificResearchService",
    "get_service",
    "reset_service",
    "router",
    "sci",
]
