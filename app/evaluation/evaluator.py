"""Evaluator engine and contracts for multi-dimensional quality control and LLM-as-a-judge scoring."""

import json
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from app.evaluation.rubrics import EVALUATION_RUBRICS, RUBRIC_DIMENSION_NAMES
from app.providers.base import ProviderMessage, ProviderRequest
from app.providers.gateway import model_gateway
from app.utils.logger import logger


class EvaluationScore(BaseModel):
    """Score breakdown for an individual criterion."""
    criterion: str = Field(description="correctness, relevance, completeness, reasoning_quality, evidence_quality, safety, latency, efficiency")
    score: float = Field(ge=0.0, le=1.0, description="Normalized score between 0.0 and 1.0")
    reasoning: str = Field(description="Explanation or evidence justifying the score")


class EvaluationReport(BaseModel):
    """Consolidated evaluation assessment for a synthesized answer or debate round.

    ``judge_ran`` and ``judge_error`` exist because a failed judge used to be
    indistinguishable from a passing one: when the judge call or its JSON failed, this
    module substituted six hardcoded scores (correctness 0.85, relevance 0.90, completeness
    0.80, reasoning_quality 0.85, evidence_quality 0.80, safety 1.0, confidence 0.80),
    averaged them together with the deterministic latency/efficiency scores, and returned
    the result under ``evaluator_model="gemini:gemini-2.5-pro"`` — a model that had not
    run, scoring an answer it had not seen. A consumer reading ``overall_score`` could not
    tell the difference. Now the semantic dimensions are simply absent when the judge did
    not run, ``confidence`` is ``None``, ``judge_ran`` is ``False`` and ``judge_error``
    says why.
    """
    run_id: str
    overall_score: float = Field(ge=0.0, le=1.0)
    scores: list[EvaluationScore]
    strengths: list[str] = Field(default_factory=list)
    flaws_identified: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evaluator_model: str | None = None
    #: True only when a model actually produced the semantic scores.
    judge_ran: bool = False
    #: Which dimensions were scored by the judge, and which by deterministic rules.
    judge_dimensions: list[str] = Field(default_factory=list)
    deterministic_dimensions: list[str] = Field(default_factory=list)
    judge_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseEvaluator(ABC):
    """Abstract base class for evaluation engines."""

    @abstractmethod
    async def evaluate_answer(
        self,
        question: str,
        answer: str,
        context: dict[str, Any] | None = None,
        criteria: list[str] | None = None
    ) -> EvaluationReport:
        """Evaluate and score a generated answer across dimensions."""

    @abstractmethod
    async def evaluate_debate_round(
        self,
        question: str,
        round_number: int,
        round_messages: list[dict[str, Any]]
    ) -> EvaluationReport:
        """Evaluate intermediate arguments, critiques, and rebuttals in a debate round."""

    @abstractmethod
    def get_supported_criteria(self) -> list[str]:
        """Return list of supported evaluation criteria."""


class Evaluator(BaseEvaluator):
    """
    Evaluates generated answers and debate rounds against the 8-dimension rubric.
    Enforces the core principle: 'Confidence is not correctness'.
    Uses hybrid evaluation: deterministic scoring for latency/tokens + LLM-as-a-judge for semantic quality.
    """

    #: Deadline for one judge call (seconds).
    judge_timeout_seconds: float = 45.0

    def __init__(
        self,
        judge_provider_name: str = "gemini",
        judge_model_name: str = "gemini-2.5-pro"
    ) -> None:
        self.judge_provider_name = judge_provider_name
        self.judge_model_name = judge_model_name

    def get_supported_criteria(self) -> list[str]:
        return list(RUBRIC_DIMENSION_NAMES)

    def _score_deterministic_dimensions(
        self,
        latency_seconds: float,
        total_tokens: int,
        mode: str = "fast"
    ) -> list[EvaluationScore]:
        """Calculates exact scores for latency and usage efficiency."""
        scores: list[EvaluationScore] = []

        # Latency scoring
        if mode == "fast":
            if latency_seconds <= 1.5:
                lat_score = 1.0
            elif latency_seconds <= 3.0:
                lat_score = 0.8
            elif latency_seconds <= 6.0:
                lat_score = 0.6
            else:
                lat_score = max(0.2, 1.0 - (latency_seconds / 20.0))
        else:  # debate / review
            if latency_seconds <= 5.0:
                lat_score = 1.0
            elif latency_seconds <= 15.0:
                lat_score = 0.85
            elif latency_seconds <= 30.0:
                lat_score = 0.7
            else:
                lat_score = max(0.2, 1.0 - (latency_seconds / 60.0))

        scores.append(EvaluationScore(
            criterion="latency",
            score=round(lat_score, 2),
            reasoning=f"Wall-clock execution took {latency_seconds:.2f}s in mode '{mode}'."
        ))

        # Usage efficiency scoring
        if mode == "fast":
            if total_tokens <= 500:
                eff_score = 1.0
            elif total_tokens <= 1500:
                eff_score = 0.8
            else:
                eff_score = max(0.3, 1.0 - (total_tokens / 5000.0))
        else:
            if total_tokens <= 3000:
                eff_score = 1.0
            elif total_tokens <= 6000:
                eff_score = 0.85
            elif total_tokens <= 10000:
                eff_score = 0.7
            else:
                eff_score = max(0.2, 1.0 - (total_tokens / 20000.0))

        scores.append(EvaluationScore(
            criterion="usage_efficiency",
            score=round(eff_score, 2),
            reasoning=f"Consumed {total_tokens} total prompt and completion tokens."
        ))

        return scores

    def _build_judge_prompt(
        self,
        question: str,
        answer: str,
        context: dict[str, Any] | None = None
    ) -> str:
        """Constructs LLM-as-a-judge evaluation prompt."""
        rubrics_desc = "\n".join([
            f"- {k.upper()}: {v.description}\n  High score criteria: {v.high_score_criteria}\n  Low score criteria: {v.min_score_criteria}"
            for k, v in EVALUATION_RUBRICS.items() if not v.is_deterministic
        ])

        return f"""You are an expert impartial judge in the Inference Quality Evaluation System.
Evaluate the following generated answer against the given question and context.

QUESTION:
{question}

ANSWER TO EVALUATE:
{answer}

ADDITIONAL CONTEXT / METADATA:
{json.dumps(context or {}, indent=2)}

SCORING CRITERIA (Score each from 0.0 to 1.0):
{rubrics_desc}

PRINCIPLE: Confidence is not correctness. Do not award high scores to assertive or authoritative prose if the underlying technical reasoning is flawed, speculative, or unverified.

Return ONLY a valid JSON object matching this schema:
{{
  "scores": [
    {{"criterion": "correctness", "score": 0.95, "reasoning": "..."}},
    {{"criterion": "relevance", "score": 1.0, "reasoning": "..."}},
    {{"criterion": "completeness", "score": 0.90, "reasoning": "..."}},
    {{"criterion": "reasoning_quality", "score": 0.92, "reasoning": "..."}},
    {{"criterion": "evidence_quality", "score": 0.88, "reasoning": "..."}},
    {{"criterion": "safety", "score": 1.0, "reasoning": "..."}}
  ],
  "strengths": ["...", "..."],
  "flaws_identified": ["...", "..."],
  "calibrated_confidence": 0.90
}}
"""

    async def evaluate_answer(
        self,
        question: str,
        answer: str,
        context: dict[str, Any] | None = None,
        criteria: list[str] | None = None
    ) -> EvaluationReport:
        """Evaluate an answer using LLM-as-a-judge and deterministic metrics."""
        ctx = context or {}
        latency = float(ctx.get("latency_seconds", 1.0))
        tokens = int(ctx.get("total_tokens", 500))
        mode = str(ctx.get("mode_used", "fast"))
        run_id = str(ctx.get("run_id", "eval_run"))

        # 1. Deterministic scores
        deterministic_scores = self._score_deterministic_dimensions(latency, tokens, mode)

        # 2. LLM-as-a-judge for semantic dimensions
        judge_prompt = self._build_judge_prompt(question, answer, context)
        judge_req = ProviderRequest(
            messages=[ProviderMessage(role="user", content=judge_prompt)],
            system_instruction="You are an uncompromising, objective technical benchmark judge.",
            model=self.judge_model_name,
            temperature=0.1,
            # An evaluation must not hang behind a stalled provider; the gateway enforces
            # this as an overall deadline across retries and fallbacks.
            extra_params={"timeout": float(self.judge_timeout_seconds)},
        )

        def _extract_json_object(text: str) -> dict[str, Any]:
            """First balanced JSON object in ``text``, tolerating prose around it.

            Models routinely answer "Here is my assessment:\n{...}" despite an
            instruction to return only JSON. Failing the whole evaluation over a leading
            sentence throws away a usable judgement; ``raw_decode`` from each ``{`` finds
            the object without assuming the response is *only* the object.
            """
            decoder = json.JSONDecoder()
            for index, char in enumerate(text):
                if char != "{":
                    continue
                try:
                    candidate, _ = decoder.raw_decode(text[index:])
                except ValueError:
                    continue
                if isinstance(candidate, dict):
                    return candidate
            return {}

        def _validate_judge_payload(parsed: dict[str, Any]) -> list[EvaluationScore]:
            """Rebuild judge scores, rejecting anything outside the declared range.

            ``float(s["score"])`` was applied without bounds or key checks, so a judge
            that returned ``score: 1.4`` raised a ``ValidationError`` from deep inside
            (surfaced as a generic partial evaluation) and a judge that omitted
            ``criterion`` raised ``KeyError``. Both are reported as judge errors with the
            offending value named.
            """
            if not isinstance(parsed.get("scores"), list):
                raise ValueError("judge payload has no 'scores' list")
            built: list[EvaluationScore] = []
            for index, row in enumerate(parsed["scores"]):
                if not isinstance(row, dict):
                    raise ValueError(f"judge score entry {index} is not an object")
                criterion = row.get("criterion")
                if not isinstance(criterion, str) or not criterion.strip():
                    raise ValueError(f"judge score entry {index} has no 'criterion'")
                raw_value = row.get("score")
                if not isinstance(raw_value, (int, float, str)):
                    raise ValueError(
                        f"judge score for {criterion!r} is not a number: {raw_value!r}"
                    )
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"judge score for {criterion!r} is not a number: {row.get('score')!r}"
                    ) from exc
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"judge score for {criterion!r} is outside 0.0-1.0: {value}"
                    )
                built.append(EvaluationScore(
                    criterion=criterion,
                    score=value,
                    reasoning=str(row.get("reasoning", "")),
                ))
            return built

        try:
            # Through the gateway, not ``get_provider(...).generate(...)`` directly: the
            # direct call skipped per-provider rate limiting (so a benchmark loop could
            # trip external 429s), key rotation and quarantine, the deadline budget, and
            # health recording — meaning a judge that kept failing never marked its
            # provider unhealthy and never fell back.
            resp = await model_gateway.execute(
                self.judge_provider_name, judge_req,
                capability="evaluation", stage_name="llm_judge",
            )
            raw_text = resp.content.strip()

            # Clean possible markdown code fences
            raw_text = raw_text.removeprefix("```json")
            raw_text = raw_text.removeprefix("```")
            raw_text = raw_text.removesuffix("```")

            parsed = json.loads(raw_text.strip())
            llm_scores = _validate_judge_payload(parsed)
            strengths = list(parsed.get("strengths") or [])
            flaws = list(parsed.get("flaws_identified") or [])
            raw_confidence = parsed.get("calibrated_confidence")
            # No default: a judge that did not state a calibrated confidence must not have
            # 0.85 invented for it.
            confidence = (
                float(raw_confidence)
                if isinstance(raw_confidence, (int, float)) and 0.0 <= float(raw_confidence) <= 1.0
                else None
            )

            judge_ran = True
            judge_error = None

        except json.JSONDecodeError as exc:
            # The judge replied, but not with JSON. Try to find an object inside the prose
            # before giving up; if there is none, report the partial evaluation honestly.
            recovered = _extract_json_object(raw_text)
            if recovered.get("scores"):
                try:
                    llm_scores = _validate_judge_payload(recovered)
                    strengths = list(recovered.get("strengths") or [])
                    flaws = list(recovered.get("flaws_identified") or [])
                    raw_confidence = recovered.get("calibrated_confidence")
                    confidence = (
                        float(raw_confidence)
                        if isinstance(raw_confidence, (int, float)) and 0.0 <= float(raw_confidence) <= 1.0
                        else None
                    )
                    judge_ran = True
                    judge_error = (
                        "recovered JSON object from a non-JSON judge reply "
                        f"({type(exc).__name__})"
                    )
                except Exception as inner:
                    llm_scores, strengths, flaws, confidence = [], [], [], None
                    judge_ran = False
                    judge_error = f"{type(inner).__name__}: {str(inner)[:200]}"
            else:
                llm_scores, strengths, flaws, confidence = [], [], [], None
                judge_ran = False
                judge_error = f"{type(exc).__name__}: {str(exc)[:200]}"

        except Exception as exc:
            # The judge is unavailable or unusable. Say so, and report only what was
            # actually measured: the deterministic latency/efficiency scores. Producing
            # ~0.85 semantic scores here (what this branch used to do) is a passing grade
            # for an answer nobody evaluated.
            logger.warning("LLM judge unavailable, reporting deterministic dimensions only: %s", str(exc))
            llm_scores = []
            strengths = []
            flaws = []
            confidence = None
            judge_ran = False
            judge_error = f"{type(exc).__name__}: {str(exc)[:200]}"

        all_scores = llm_scores + deterministic_scores
        overall_score = round(sum(s.score for s in all_scores) / len(all_scores), 3)
        judge_dimensions = [s.criterion for s in llm_scores]
        deterministic_dimensions = [s.criterion for s in deterministic_scores]

        return EvaluationReport(
            run_id=run_id,
            overall_score=overall_score,
            scores=all_scores,
            strengths=strengths,
            flaws_identified=flaws,
            confidence=confidence,
            # Named only when the judge ran: the string was previously emitted even on the
            # failure path, attributing invented scores to a model.
            evaluator_model=(
                f"{self.judge_provider_name}:{self.judge_model_name}" if judge_ran else None
            ),
            judge_ran=judge_ran,
            judge_dimensions=judge_dimensions,
            deterministic_dimensions=deterministic_dimensions,
            judge_error=judge_error,
            metadata={
                "mode": mode,
                "latency": latency,
                "tokens": tokens,
                "partial_evaluation": not judge_ran,
                "overall_score_basis": (
                    "judge + deterministic dimensions" if judge_ran
                    else "deterministic dimensions only (no semantic evaluation was performed)"
                ),
            }
        )

    async def evaluate_debate_round(
        self,
        question: str,
        round_number: int,
        round_messages: list[dict[str, Any]]
    ) -> EvaluationReport:
        """Evaluate the quality of arguments and critiques in a specific debate round."""
        combined_text = "\n".join([f"{m.get('agent_role', 'Agent')}: {m.get('content', '')}" for m in round_messages])
        return await self.evaluate_answer(
            question=f"Debate Round {round_number}: {question}",
            answer=combined_text,
            context={"round_number": round_number, "agent_count": len(round_messages)}
        )


# Global default evaluator
evaluator = Evaluator()
