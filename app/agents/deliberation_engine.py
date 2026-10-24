"""
app/agents/deliberation_engine.py — Inference Multi-Model Deliberation Engine.

Enhances Inference with:
1. Multi-Model Consensus: Gathers perspectives from distinct model families
2. Evidence-Based Reasoning: Requires verifiable evidence, not just model agreement
3. Confidence Scoring: Rates each model's contribution based on track record
4. Conflict Resolution: Handles disagreements between models
5. Cost-Aware Routing: Routes to cheapest/fastest model for simple tasks
6. Quality Gates: Enforces minimum quality thresholds for high-stakes decisions
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.security import require_inference_api_key
from app.providers.gateway import model_gateway

logger = logging.getLogger(__name__)


class DeliberationStrategy(str, Enum):
    SINGLE_MODEL = "single_model"
    MULTI_MODEL_CONSENSUS = "multi_model_consensus"
    EVIDENCE_REQUIRED = "evidence_required"
    HUMAN_REVIEW = "human_review"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ModelOpinion:
    """An opinion from a single model."""
    model_name: str
    model_family: str  # e.g., "gpt", "gemini", "claude", "llama"
    response: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    reasoning: str = ""
    latency_ms: float = 0.0
    token_cost: float = 0.0


@dataclass
class DeliberationResult:
    """Result of multi-model deliberation."""
    decision: str
    consensus_reached: bool
    agreeing_models: list[str] = field(default_factory=list)
    dissenting_models: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    model_opinions: list[ModelOpinion] = field(default_factory=list)
    strategy: DeliberationStrategy = DeliberationStrategy.SINGLE_MODEL
    risk_level: RiskLevel = RiskLevel.LOW
    requires_human_review: bool = False


class DeliberationEngine:
    """Multi-model deliberation engine for high-stakes decisions."""

    MODEL_FAMILIES = {
        "gpt": {"models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"], "cost": 1.0},
        "gemini": {"models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-flash-lite"], "cost": 0.8},
        "claude": {"models": ["claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus"], "cost": 1.2},
        "llama": {"models": ["llama-3-70b", "llama-3-8b"], "cost": 0.5},
    }

    def __init__(self):
        self.model_track_record: dict[str, dict[str, float]] = {}
        self.deliberation_history: list[DeliberationResult] = []

    async def deliberate(
        self,
        question: str,
        strategy: DeliberationStrategy = DeliberationStrategy.MULTI_MODEL_CONSENSUS,
        risk_level: RiskLevel = RiskLevel.LOW,
        require_evidence: bool = False,
        context: dict[str, Any] | None = None,
    ) -> DeliberationResult:
        """
        Deliberate on a question using multiple models.

        Args:
            question: The question to deliberate on
            strategy: Deliberation strategy to use
            risk_level: Risk level of the decision
            require_evidence: Whether to require verifiable evidence
            context: Additional context for the models

        Returns:
            DeliberationResult with consensus decision
        """
        logger.info(f"Deliberating: {question[:80]} (strategy={strategy}, risk={risk_level})")

        start_time = time.time()

        # Select strategy based on risk level
        if risk_level == RiskLevel.CRITICAL:
            strategy = DeliberationStrategy.HUMAN_REVIEW
        elif risk_level == RiskLevel.HIGH:
            strategy = DeliberationStrategy.EVIDENCE_REQUIRED
            require_evidence = True

        # Single model for simple/low-risk tasks
        if strategy == DeliberationStrategy.SINGLE_MODEL:
            return await self._single_model_deliberation(question, context)

        # Multi-model deliberation
        opinions: list[ModelOpinion] = []

        # Gather opinions from distinct model families
        for family, config in self.MODEL_FAMILIES.items():
            try:
                model_name = config["models"][0]  # Use best model from each family
                opinion = await self._get_model_opinion(
                    model_name, family, question, context, require_evidence
                )
                if opinion:
                    opinions.append(opinion)
            except Exception as e:
                logger.warning(f"Failed to get opinion from {family}: {e}")

        if not opinions:
            return DeliberationResult(
                decision="UNAVAILABLE",
                consensus_reached=False,
                confidence=0.0,
                strategy=strategy,
                risk_level=risk_level,
                requires_human_review=True,
            )

        # Analyze opinions for consensus
        result = self._analyze_consensus(opinions, strategy, risk_level, require_evidence)

        # Record history
        result.model_opinions = opinions
        self.deliberation_history.append(result)

        # Update track records
        self._update_track_records(opinions, result)

        logger.info(
            f"Deliberation complete: {result.decision} "
            f"(consensus={result.consensus_reached}, confidence={result.confidence:.2f})"
        )

        return result

    async def _single_model_deliberation(
        self,
        question: str,
        context: dict[str, Any] | None,
    ) -> DeliberationResult:
        """Deliberate using a single model."""
        opinion = await self._get_model_opinion(
            "gpt-4o-mini", "gpt", question, context, False
        )

        if not opinion:
            return DeliberationResult(
                decision="UNAVAILABLE",
                consensus_reached=False,
                strategy=DeliberationStrategy.SINGLE_MODEL,
                requires_human_review=True,
            )

        return DeliberationResult(
            decision=opinion.response[:500],
            consensus_reached=True,
            agreeing_models=[opinion.model_name],
            confidence=opinion.confidence,
            evidence=opinion.evidence,
            model_opinions=[opinion],
            strategy=DeliberationStrategy.SINGLE_MODEL,
        )

    async def _get_model_opinion(
        self,
        model_name: str,
        model_family: str,
        question: str,
        context: dict[str, Any] | None,
        require_evidence: bool,
    ) -> ModelOpinion | None:
        """Get an opinion from a specific model."""
        start_time = time.time()

        prompt = f"""You are a highly qualified expert. Please provide a thorough, evidence-based response.

Question: {question}

Context: {json.dumps(context or {}, indent=2)}

Instructions:
- Provide a clear, well-reasoned answer
- Cite evidence when possible
- If uncertain, say so explicitly
- Provide a confidence score (0-1)
"""

        if require_evidence:
            prompt += "\nIMPORTANT: You must cite specific evidence or sources for your claims."

        try:
            response = await model_gateway.generate(
                prompt=prompt,
                model=model_name,
                task="deliberation",
            )

            latency = (time.time() - start_time) * 1000

            # Extract confidence and evidence from response
            confidence = 0.5
            evidence = []

            # Simple parsing (in production, use structured output)
            if "confidence" in response.text.lower():
                conf_match = re.search(r'confidence[:\s]*(\d+\.?\d*)', response.text, re.IGNORECASE)
                if conf_match:
                    confidence = float(conf_match.group(1)) / 100.0

            # Get track record bonus
            track_record = self.model_track_record.get(model_name, {})
            track_bonus = track_record.get("accuracy", 0.5) * 0.2
            confidence = min(1.0, confidence + track_bonus)

            return ModelOpinion(
                model_name=model_name,
                model_family=model_family,
                response=response.text,
                confidence=confidence,
                evidence=evidence,
                reasoning=response.text[:200],
                latency_ms=latency,
                token_cost=response.usage.get("total_tokens", 0) * 0.0001,
            )
        except Exception as e:
            logger.error(f"Model {model_name} failed: {e}")
            return None

    def _analyze_consensus(
        self,
        opinions: list[ModelOpinion],
        strategy: DeliberationStrategy,
        risk_level: RiskLevel,
        require_evidence: bool,
    ) -> DeliberationResult:
        """Analyze opinions to determine consensus."""
        # Extract key points from each opinion
        key_points: dict[str, list[ModelOpinion]] = {}

        for opinion in opinions:
            # Simple key point extraction (first sentence)
            first_sentence = opinion.response.split('.')[0] if opinion.response else ""
            key_points.setdefault(first_sentence[:100], []).append(opinion)

        # Find the most agreed-upon point
        best_point = max(key_points, key=lambda p: len(key_points[p]))
        agreeing = key_points[best_point]

        # Check for dissent
        dissenting = [o for o in opinions if o not in agreeing]

        # Calculate confidence
        avg_confidence = sum(o.confidence for o in agreeing) / len(agreeing)
        agreement_ratio = len(agreeing) / len(opinions)
        consensus_confidence = avg_confidence * agreement_ratio

        # Evidence check
        all_evidence = []
        for o in opinions:
            all_evidence.extend(o.evidence)

        # Human review required if:
        # 1. Risk level is critical
        # 2. Consensus confidence is low
        # 3. Strong dissent exists
        requires_review = (
            risk_level == RiskLevel.CRITICAL or
            consensus_confidence < 0.5 or
            len(dissenting) >= len(opinions) // 2
        )

        if require_evidence and not all_evidence:
            requires_review = True
            consensus_confidence *= 0.5

        return DeliberationResult(
            decision=best_point,
            consensus_reached=len(dissenting) < len(opinions) // 2,
            agreeing_models=[o.model_name for o in agreeing],
            dissenting_models=[o.model_name for o in dissenting],
            confidence=consensus_confidence,
            evidence=all_evidence,
            strategy=strategy,
            risk_level=risk_level,
            requires_human_review=requires_review,
        )

    def _update_track_records(
        self,
        opinions: list[ModelOpinion],
        result: DeliberationResult,
    ) -> None:
        """Update model track records based on deliberation outcome."""
        for opinion in opinions:
            if opinion.model_name not in self.model_track_record:
                self.model_track_record[opinion.model_name] = {
                    "accuracy": 0.5,
                    "total_deliberations": 0,
                    "correct_predictions": 0,
                }

            record = self.model_track_record[opinion.model_name]
            record["total_deliberations"] += 1

            # If model was in agreeing group and consensus was correct, increment
            if opinion.model_name in result.agreeing_models and result.consensus_reached:
                record["correct_predictions"] += 1

            # Recalculate accuracy
            if record["total_deliberations"] > 0:
                record["accuracy"] = record["correct_predictions"] / record["total_deliberations"]

    def get_model_rankings(self) -> list[tuple[str, float]]:
        """Get model rankings by accuracy."""
        rankings = []
        for model, record in self.model_track_record.items():
            if record["total_deliberations"] > 0:
                rankings.append((model, record["accuracy"]))

        return sorted(rankings, key=lambda x: x[1], reverse=True)


# Module-level singleton
deliberation_engine = DeliberationEngine()
