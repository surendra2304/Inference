"""
Inference/services/analogical_reasoning_engine.py — Inference Analogical Reasoning Engine.

Analogical mapping, structural similarity detection, cross-domain analogy,
and abstract pattern transfer.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()


@dataclass
class Concept:
    """A concept for analogical reasoning."""
    concept_id: str
    name: str
    source_domain: str = ""
    attributes: dict = field(default_factory=dict)
    relationships: list[str] = field(default_factory=list)


@dataclass
class Analogy:
    """An analogy between two concepts."""
    analogy_id: str
    source_concept: str
    target_concept: str
    mapping: dict[str, str] = field(default_factory=dict)
    similarity_score: float = 0.0
    new_insights: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class AbstractPattern:
    """An abstract pattern extracted from analogies."""
    pattern_id: str
    description: str
    source_analogies: list[str] = field(default_factory=list)
    generality: float = 0.0
    applications: list[str] = field(default_factory=list)


class AnalogicalReasoningEngine:
    """Engine for analogical reasoning and cross-domain transfer."""

    def __init__(self) -> None:
        self._concepts: dict[str, Concept] = {}
        self._analogies: list[Analogy] = []
        self._patterns: list[AbstractPattern] = []

    def add_concept(self, name: str, source_domain: str = "",
                    attributes: dict | None = None,
                    relationships: list[str] | None = None) -> Concept:
        """Add a concept."""
        concept_id = f"con_{uuid.uuid4().hex[:12]}"
        concept = Concept(
            concept_id=concept_id, name=name, source_domain=source_domain,
            attributes=attributes or {}, relationships=relationships or [],
        )
        self._concepts[concept_id] = concept
        return concept

    def create_analogy(self, source_id: str, target_id: str) -> Analogy:
        """Create an analogy between two concepts."""
        source = self._concepts.get(source_id)
        target = self._concepts.get(target_id)
        if not source or not target:
            raise HTTPException(status_code=404, detail="Concept not found")

        analogy_id = f"ana_{uuid.uuid4().hex[:12]}"
        analogy = Analogy(
            analogy_id=analogy_id, source_concept=source.name,
            target_concept=target.name,
        )

        # Map attributes
        for attr_name, attr_value in source.attributes.items():
            if attr_name in target.attributes:
                analogy.mapping[attr_name] = target.attributes[attr_name]

        # Calculate similarity
        common_attrs = len(analogy.mapping)
        total_attrs = max(len(source.attributes), len(target.attributes), 1)
        analogy.similarity_score = common_attrs / total_attrs

        # Generate insights
        for attr_name in source.attributes:
            if attr_name not in target.attributes and analogy.similarity_score > 0.3:
                analogy.new_insights.append(
                    f"Target may have attribute '{attr_name}' similar to source"
                )

        self._analogies.append(analogy)
        return analogy

    def extract_pattern(self, description: str, analogy_ids: list[str] | None = None) -> AbstractPattern:
        """Extract an abstract pattern from analogies."""
        pattern_id = f"pat_{uuid.uuid4().hex[:12]}"
        pattern = AbstractPattern(pattern_id=pattern_id, description=description)

        analogies = [a for a in self._analogies if a.analogy_id in (analogy_ids or [])]
        if not analogies:
            analogies = self._analogies[:5]

        pattern.source_analogies = [a.analogy_id for a in analogies]
        pattern.generality = (
            sum(a.similarity_score for a in analogies) / max(1, len(analogies))
        )
        pattern.applications = [
            f"Can apply to {a.target_concept}" for a in analogies[:3]
        ]

        self._patterns.append(pattern)
        return pattern

    def transfer_knowledge(self, source_id: str, target_domain: str) -> dict[str, Any]:
        """Transfer knowledge via analogy to a new domain."""
        source = self._concepts.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source concept not found")

        transfers = []
        for analogy in self._analogies:
            if analogy.source_concept == source.name and analogy.similarity_score > 0.4:
                transfers.append({
                    "concept": analogy.target_concept,
                    "similarity": analogy.similarity_score,
                    "mapping": analogy.mapping,
                })

        return {
            "source": source.name,
            "target_domain": target_domain,
            "transfer_count": len(transfers),
            "transfers": transfers[:3],
        }

    def get_concepts(self, domain: str | None = None) -> list[dict[str, Any]]:
        """Get concepts."""
        concepts = list(self._concepts.values())
        if domain:
            concepts = [c for c in concepts if c.source_domain == domain]
        return [
            {"name": c.name, "domain": c.source_domain,
             "attribute_count": len(c.attributes), "relations": len(c.relationships)}
            for c in concepts
        ]

    def get_report(self) -> dict[str, Any]:
        """Get engine report."""
        return {
            "total_concepts": len(self._concepts),
            "total_analogies": len(self._analogies),
            "total_patterns": len(self._patterns),
            "avg_similarity": round(
                sum(a.similarity_score for a in self._analogies) / max(1, len(self._analogies)), 2
            ) if self._analogies else 0,
            "engine_status": "operational",
        }


engine = AnalogicalReasoningEngine()


@router.post("/inference/analogy/concepts")
async def add_concept(name: str, source_domain: str = "", attributes: dict | None = None,
                      relationships: list[str] | None = None):
    """Add a concept."""
    concept = engine.add_concept(name, source_domain, attributes, relationships)
    return {"concept_id": concept.concept_id, "name": concept.name}


@router.post("/inference/analogy/create")
async def create_analogy(source_id: str, target_id: str):
    """Create an analogy."""
    analogy = engine.create_analogy(source_id, target_id)
    return {
        "analogy_id": analogy.analogy_id,
        "similarity": analogy.similarity_score,
        "mapping_count": len(analogy.mapping),
        "insights": analogy.new_insights[:3],
    }


@router.post("/inference/analogy/patterns")
async def extract_pattern(description: str, analogy_ids: list[str] | None = None):
    """Extract an abstract pattern."""
    pattern = engine.extract_pattern(description, analogy_ids)
    return {
        "pattern_id": pattern.pattern_id,
        "generality": pattern.generality,
        "applications": pattern.applications,
    }


@router.post("/inference/analogy/transfer")
async def transfer_knowledge(source_id: str, target_domain: str):
    """Transfer knowledge via analogy."""
    return engine.transfer_knowledge(source_id, target_domain)


@router.get("/inference/analogy/concepts")
async def get_concepts(domain: str | None = None):
    """Get concepts."""
    return {"concepts": engine.get_concepts(domain)}


@router.get("/inference/analogy/report")
async def analogy_report():
    """Get analogical reasoning engine report."""
    return engine.get_report()
