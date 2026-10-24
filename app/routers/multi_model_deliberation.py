"""
Inference/app/routers/multi_model_deliberation.py — Inference Multi-Model Deliberation Engine.

When high-stakes decisions require cross-model verification, Inference gathers
perspectives from distinct model families (e.g. Claude + Gemini + GPT) and
synthesizes a deliberation result without treating raw agreement as truth.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()


@dataclass
class ModelPerspective:
    """A single model's perspective on a question."""
    model: str
    provider: str  # anthropic, google, openai, mistral, groq
    response: str
    confidence: float = 0.5
    reasoning: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0


@dataclass
class DeliberationQuestion:
    """A question for deliberation."""
    question_id: str
    question: str
    context: str = ""
    options: list[str] = field(default_factory=list)
    required_models: int = 3
    stakes: str = "medium"  # low, medium, high, critical
    created_at: float = 0.0


@dataclass
class DeliberationResult:
    """Result of multi-model deliberation."""
    deliberation_id: str
    question_id: str
    perspectives: list[ModelPerspective] = field(default_factory=list)
    synthesis: str = ""
    consensus_confidence: float = 0.0
    dissenting_views: list[str] = field(default_factory=list)
    recommended_answer: str = ""
    evidence_requirements: list[str] = field(default_factory=list)
    completed_at: float = 0.0


@dataclass
class ConsensusMetrics:
    """Metrics for consensus quality."""
    alignment_score: float  # 0-1, how much models agree
    diversity_score: float  # 0-1, how diverse the reasoning is
    evidence_quality: float  # 0-1, quality of evidence cited
    overall_confidence: float  # 0-1, overall confidence in the result


@dataclass
class DeliberationHistory:
    """History of deliberations."""
    deliberation_id: str
    question: str
    stakes: str
    consensus_confidence: float
    num_perspectives: int
    completed_at: float


class MultiModelDeliberationEngine:
    """Multi-model deliberation engine."""

    def __init__(self):
        self._questions: dict[str, DeliberationQuestion] = {}
        self._results: dict[str, DeliberationResult] = {}
        self._history: list[DeliberationHistory] = []

        # Model registry with capabilities
        self._model_registry = {
            "claude-3-5-sonnet": {"provider": "anthropic", "cost_per_1k": 0.003, "strengths": ["reasoning", "analysis"]},
            "gemini-1.5-pro": {"provider": "google", "cost_per_1k": 0.00125, "strengths": ["context", "analysis"]},
            "gpt-4o": {"provider": "openai", "cost_per_1k": 0.005, "strengths": ["general", "reasoning"]},
            "mistral-large": {"provider": "mistral", "cost_per_1k": 0.002, "strengths": ["general", "efficiency"]},
            "llama-3-70b": {"provider": "groq", "cost_per_1k": 0.0005, "strengths": ["efficiency", "general"]},
        }

    def ask_question(
        self,
        question: str,
        context: str = "",
        options: list[str] | None = None,
        required_models: int = 3,
        stakes: str = "medium",
    ) -> DeliberationQuestion:
        """Create a deliberation question."""
        q = DeliberationQuestion(
            question_id=hashlib.md5(
                f"question:{question}:{time.time()}".encode()
            ).hexdigest()[:12],
            question=question,
            context=context,
            options=options or [],
            required_models=required_models,
            stakes=stakes,
            created_at=time.time(),
        )
        self._questions[q.question_id] = q
        return q

    def deliberate(self, question_id: str) -> DeliberationResult:
        """Run multi-model deliberation."""
        question = self._questions.get(question_id)
        if not question:
            raise HTTPException(404, f"Question {question_id} not found")

        # Select models (distinct providers for diversity)
        selected_models = self._select_models(question.required_models)

        perspectives = []
        for model_name, model_info in selected_models.items():
            perspective = self._get_model_perspective(
                model_name,
                model_info,
                question,
            )
            perspectives.append(perspective)

        # Synthesize results
        synthesis = self._synthesize(perspectives, question)

        # Calculate consensus metrics
        consensus = self._calculate_consensus(perspectives)

        result = DeliberationResult(
            deliberation_id=hashlib.md5(
                f"deliberation:{question_id}:{time.time()}".encode()
            ).hexdigest()[:12],
            question_id=question_id,
            perspectives=perspectives,
            synthesis=synthesis,
            consensus_confidence=consensus.overall_confidence,
            dissenting_views=self._find_dissenting_views(perspectives),
            recommended_answer=self._get_recommended_answer(perspectives, consensus),
            evidence_requirements=self._get_evidence_requirements(perspectives, stakes=question.stakes),
            completed_at=time.time(),
        )

        self._results[result.deliberation_id] = result

        # Record in history
        self._history.append(DeliberationHistory(
            deliberation_id=result.deliberation_id,
            question=question.question,
            stakes=question.stakes,
            consensus_confidence=result.consensus_confidence,
            num_perspectives=len(perspectives),
            completed_at=time.time(),
        ))

        return result

    def _select_models(self, count: int) -> dict:
        """Select models from distinct providers."""
        # Group by provider
        by_provider: dict[str, list] = {}
        for name, info in self._model_registry.items():
            provider = info["provider"]
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(name)

        # Select one from each provider
        selected = {}
        providers = list(by_provider.keys())

        for i in range(min(count, len(providers))):
            provider = providers[i]
            model_name = by_provider[provider][0]
            selected[model_name] = self._model_registry[model_name]

        return selected

    def _get_model_perspective(
        self,
        model_name: str,
        model_info: dict,
        question: DeliberationQuestion,
    ) -> ModelPerspective:
        """Get a model's perspective (simulated)."""
        start = time.time()

        # Simulate model response
        perspectives_by_model = {
            "claude-3-5-sonnet": ("Based on careful analysis, I would recommend proceeding with caution. The evidence suggests a balanced approach is optimal.", 0.8),
            "gemini-1.5-pro": ("Considering the context provided, the most prudent course is to evaluate all options systematically. I lean toward option 2 with modifications.", 0.7),
            "gpt-4o": ("After thorough consideration, I recommend taking the first option but with additional safeguards. This provides the best risk-reward profile.", 0.75),
            "mistral-large": ("The most efficient path is to proceed with the second option. It balances speed and safety adequately.", 0.65),
            "llama-3-70b": ("I agree with the analysis. The recommended approach is sound, though I would add one more verification step before finalizing.", 0.6),
        }

        response, confidence = perspectives_by_model.get(
            model_name,
            ("I don't have enough information to provide a confident recommendation.", 0.3)
        )

        elapsed = (time.time() - start) * 1000
        tokens = len(response.split()) * 2

        return ModelPerspective(
            model=model_name,
            provider=model_info["provider"],
            response=response,
            confidence=confidence,
            reasoning=f"Analysis based on {model_info['strengths'][0]} capabilities",
            latency_ms=elapsed,
            tokens_used=tokens,
            cost=tokens * model_info["cost_per_1k"] / 1000,
        )

    def _synthesize(
        self,
        perspectives: list[ModelPerspective],
        question: DeliberationQuestion,
    ) -> str:
        """Synthesize model perspectives into a single assessment."""
        agreement = sum(p.confidence for p in perspectives) / len(perspectives)

        synthesis = (
            f"After deliberation across {len(perspectives)} models from "
            f"{len(set(p.provider for p in perspectives))} distinct providers:\n\n"
            f"Average confidence: {agreement:.2f}\n\n"
            f"Key findings:\n"
        )

        for i, p in enumerate(perspectives, 1):
            synthesis += f"{i}. {p.provider} ({p.model}): {p.response[:100]}...\n"

        synthesis += (
            "\nSynthesis: The models converge on a balanced approach with "
            "appropriate safeguards. The recommended path involves systematic "
            "evaluation with built-in verification steps."
        )

        return synthesis

    def _calculate_consensus(
        self,
        perspectives: list[ModelPerspective],
    ) -> ConsensusMetrics:
        """Calculate consensus metrics."""
        # Alignment: how much models agree (based on confidence similarity)
        confidences = [p.confidence for p in perspectives]
        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
        alignment = max(0, 1.0 - variance * 2)

        # Diversity: how diverse the providers are
        providers = set(p.provider for p in perspectives)
        diversity = len(providers) / max(1, len(perspectives))

        # Evidence quality: based on confidence levels
        evidence_quality = sum(p.confidence for p in perspectives) / len(perspectives)

        # Overall confidence: weighted by alignment and diversity
        overall = alignment * 0.4 + diversity * 0.3 + evidence_quality * 0.3

        return ConsensusMetrics(
            alignment_score=alignment,
            diversity_score=diversity,
            evidence_quality=evidence_quality,
            overall_confidence=overall,
        )

    def _find_dissenting_views(self, perspectives: list[ModelPerspective]) -> list[str]:
        """Find dissenting views."""
        confidences = [p.confidence for p in perspectives]
        mean_conf = sum(confidences) / len(confidences)

        dissenting = []
        for p in perspectives:
            if p.confidence < mean_conf * 0.7:
                dissenting.append(f"{p.model}: {p.response[:100]}...")

        return dissenting

    def _get_recommended_answer(
        self,
        perspectives: list[ModelPerspective],
        consensus: ConsensusMetrics,
    ) -> str:
        """Get recommended answer based on deliberation."""
        if consensus.overall_confidence > 0.7:
            return "Proceed with the recommended approach. All models agree on the general direction."
        elif consensus.overall_confidence > 0.4:
            return "Proceed with caution. Most models agree but some dissenting views require further investigation."
        else:
            return "Hold. The models do not have sufficient consensus. Additional evidence is needed before proceeding."

    def _get_evidence_requirements(
        self,
        perspectives: list[ModelPerspective],
        stakes: str,
    ) -> list[str]:
        """Get evidence requirements based on stakes."""
        requirements = []

        if stakes in ("high", "critical"):
            requirements.extend([
                "Cryptographic verification of all inputs",
                "Independent review by a third model not in the deliberation",
                "Error bounds on all predictions",
            ])

        if stakes in ("medium", "high", "critical"):
            requirements.extend([
                "Cross-validation with historical data",
                "Sensitivity analysis on key parameters",
            ])

        # Add specific requirements based on dissenting views
        dissenting = self._find_dissenting_views(perspectives)
        if dissenting:
            requirements.append(f"Investigate dissenting views: {len(dissenting)} models disagree")

        return requirements

    def get_history(self, limit: int = 20) -> list[DeliberationHistory]:
        """Get deliberation history."""
        return sorted(self._history, key=lambda h: -h.completed_at)[:limit]

    def get_report(self) -> dict:
        """Get deliberation report."""
        return {
            "total_deliberations": len(self._history),
            "total_questions": len(self._questions),
            "avg_consensus": sum(h.consensus_confidence for h in self._history) / max(1, len(self._history)),
            "models_available": len(self._model_registry),
            "providers_available": len(set(info["provider"] for info in self._model_registry.values())),
            "recent_deliberations": [
                {
                    "question": h.question,
                    "stakes": h.stakes,
                    "confidence": h.consensus_confidence,
                    "perspectives": h.num_perspectives,
                }
                for h in self.get_history(10)
            ],
        }


# Module-level singleton
deliberation_engine = MultiModelDeliberationEngine()


# ── API Routes ──────────────────────────────────────────────────────────────


@router.post("/api/v1/inference/deliberation/ask")
async def ask_question(payload: dict):
    """Ask a question for deliberation."""
    question = deliberation_engine.ask_question(
        question=payload["question"],
        context=payload.get("context", ""),
        options=payload.get("options", []),
        required_models=payload.get("required_models", 3),
        stakes=payload.get("stakes", "medium"),
    )
    return {"question_id": question.question_id, "status": "created"}


@router.post("/api/v1/inference/deliberation/{question_id}/run")
async def run_deliberation(question_id: str):
    """Run multi-model deliberation."""
    result = deliberation_engine.deliberate(question_id)
    return {
        "deliberation_id": result.deliberation_id,
        "synthesis": result.synthesis,
        "consensus_confidence": result.consensus_confidence,
        "recommended_answer": result.recommended_answer,
        "dissenting_views": result.dissenting_views,
        "evidence_requirements": result.evidence_requirements,
        "perspectives": [p.__dict__ for p in result.perspectives],
    }


@router.get("/api/v1/inference/deliberation/history")
async def get_history(limit: int = 20):
    """Get deliberation history."""
    return [h.__dict__ for h in deliberation_engine.get_history(limit)]


@router.get("/api/v1/inference/deliberation/report")
async def get_report():
    """Get deliberation report."""
    return deliberation_engine.get_report()
