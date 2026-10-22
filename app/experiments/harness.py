"""Benchmark harness and automated experiment runner for Inference."""

import time
from typing import Any

from pydantic import BaseModel, Field

from app.core.orchestrator import OrchestrationRequest, Orchestrator
from app.evaluation.benchmarks import GOLDEN_BENCHMARK_SUITE
from app.evaluation.evaluator import Evaluator
from app.memory.base import BaseMemory, ExperimentRecord
from app.memory.sqlite import SQLiteMemory
from app.providers.base import ProviderMessage, ProviderRequest
from app.providers.gateway import model_gateway
from app.utils.ids import generate_id


class ExperimentRunRequest(BaseModel):
    """Payload to trigger an automated experiment."""
    experiment_type: str = Field(description="benchmark_suite, baseline_vs_debate, model_comparison")
    hypothesis: str
    target_benchmarks: list[str] | None = Field(default=None, description="Subset of benchmark IDs to run")
    models_to_test: list[str] | None = Field(default=None, description="List of providers/models for model_comparison")
    custom_question: str | None = None


class BenchmarkHarness:
    """Automated harness for running benchmarks, comparisons, and saving experiments."""

    def __init__(
        self,
        memory: BaseMemory | None = None,
        orchestrator: Orchestrator | None = None,
        evaluator: Evaluator | None = None
    ) -> None:
        self.memory = memory or SQLiteMemory()
        self.orchestrator = orchestrator or Orchestrator(memory=self.memory)
        self.evaluator = evaluator or Evaluator()

    async def run_benchmark_suite(
        self,
        benchmark_ids: list[str] | None = None
    ) -> ExperimentRecord:
        """Runs the Golden Benchmark Suite and evaluates outputs."""
        exp_id = generate_id("exp_bench")
        start_time = time.perf_counter()

        benchmarks = GOLDEN_BENCHMARK_SUITE
        if benchmark_ids:
            benchmarks = [b for b in benchmarks if b.id in benchmark_ids]
        cases = benchmarks

        results: list[dict[str, Any]] = []
        for case in cases:
            # 1. Run via Orchestrator in debate mode
            orch_res = await self.orchestrator.process_task(OrchestrationRequest(
                question=case.question,
                mode="debate",
                require_evidence=True
            ))

            # 2. Evaluate with Evaluator
            eval_report = await self.evaluator.evaluate_answer(
                question=case.question,
                answer=orch_res.answer,
                context={
                    "mode_used": orch_res.mode_used,
                    "latency_seconds": orch_res.total_latency_seconds,
                    "total_tokens": orch_res.total_tokens,
                    "benchmark_id": case.id
                }
            )

            # 3. Check concepts
            answer_lower = orch_res.answer.lower()
            present_concepts = [c for c in case.expected_key_concepts if c.lower() in answer_lower]
            concept_coverage = len(present_concepts) / len(case.expected_key_concepts) if case.expected_key_concepts else 1.0

            results.append({
                "benchmark_id": case.id,
                "domain": case.domain,
                "mode_used": orch_res.mode_used,
                "overall_score": eval_report.overall_score,
                "confidence": orch_res.confidence,
                "concept_coverage": round(concept_coverage, 2),
                "key_concepts_found": present_concepts,
                "latency_seconds": orch_res.total_latency_seconds,
                "total_tokens": orch_res.total_tokens
            })

        total_score: float = sum(float(r["overall_score"]) for r in results) if results else 0.0
        avg_score: float = round(total_score / len(results), 3) if results else 0.0
        duration = round(time.perf_counter() - start_time, 2)

        exp_record = ExperimentRecord(
            id=exp_id,
            hypothesis="Evaluate Inference multi-agent debate performance across the Golden Benchmark Suite",
            configuration={"benchmark_count": len(results), "selected_ids": benchmark_ids},
            status="completed",
            result={
                "average_score": avg_score,
                "duration_seconds": duration,
                "detailed_cases": results
            }
        )
        await self.memory.save_experiment(exp_record)
        return exp_record

    async def run_baseline_vs_debate_comparison(
        self,
        question: str
    ) -> ExperimentRecord:
        """Compares single-agent Fast mode against 6-round Multi-Agent Debate mode."""
        exp_id = generate_id("exp_comp")

        # 1. Single-agent Fast baseline
        fast_req = OrchestrationRequest(question=question, mode="fast", max_agents=1)
        fast_res = await self.orchestrator.process_task(fast_req)
        fast_eval = await self.evaluator.evaluate_answer(
            question=question,
            answer=fast_res.answer,
            context={"mode_used": "fast", "latency_seconds": fast_res.total_latency_seconds, "total_tokens": fast_res.total_tokens}
        )

        # 2. 6-Round Multi-agent Debate
        debate_req = OrchestrationRequest(question=question, mode="debate", max_agents=5)
        debate_res = await self.orchestrator.process_task(debate_req)
        debate_eval = await self.evaluator.evaluate_answer(
            question=question,
            answer=debate_res.answer,
            context={"mode_used": "debate", "latency_seconds": debate_res.total_latency_seconds, "total_tokens": debate_res.total_tokens}
        )

        score_diff = round(debate_eval.overall_score - fast_eval.overall_score, 3)
        winner = "debate" if score_diff > 0 else ("fast" if score_diff < 0 else "tie")

        # What the score actually measures must travel with it. When the semantic judge did
        # not run, ``overall_score`` is a composite of the deterministic latency/efficiency
        # dimensions only — measured live, a run with no judge produced ``winner: "fast"``,
        # ``score_difference: -0.15`` and ``score: 1.0`` vs ``0.85``, which reads as
        # "single-agent beat debate on reasoning quality". It had not been compared on
        # reasoning quality at all; the only difference measured was speed. The payload now
        # says which dimensions produced the score and withholds the quality claim.
        judge_ran = fast_eval.judge_ran and debate_eval.judge_ran
        reason_quality_compared = judge_ran
        exp_record = ExperimentRecord(
            id=exp_id,
            hypothesis="Does multi-agent structured debate outperform single-agent baseline on reasoning quality?",
            configuration={"question": question},
            status="completed",
            result={
                "winner": winner,
                "score_difference": score_diff,
                "verdict_basis": (
                    "semantic judge + deterministic dimensions" if judge_ran
                    else "deterministic dimensions only (latency/efficiency); the semantic "
                         "judge did not run, so no comparison of reasoning quality was made"
                ),
                "quality_verdict_withheld": not reason_quality_compared,
                # False means: do not read this comparison as evidence about answer quality.
                "hypothesis_tested": reason_quality_compared,
                "fast_baseline": {
                    "score": fast_eval.overall_score,
                    "confidence": fast_res.confidence,
                    "latency": fast_res.total_latency_seconds,
                    "tokens": fast_res.total_tokens,
                    "judge_ran": fast_eval.judge_ran,
                    "scored_dimensions": [s.criterion for s in fast_eval.scores],
                    "judge_error": fast_eval.judge_error,
                },
                "multi_agent_debate": {
                    "score": debate_eval.overall_score,
                    "confidence": debate_res.confidence,
                    "latency": debate_res.total_latency_seconds,
                    "tokens": debate_res.total_tokens,
                    "unresolved_disagreements": debate_res.unresolved_disagreements,
                    "judge_ran": debate_eval.judge_ran,
                    "scored_dimensions": [s.criterion for s in debate_eval.scores],
                    "judge_error": debate_eval.judge_error,
                }
            }
        )
        await self.memory.save_experiment(exp_record)
        return exp_record

    async def run_model_comparison_matrix(
        self,
        prompt: str,
        providers_to_test: list[str] | None = None
    ) -> ExperimentRecord:
        """Tests the same prompt across different cloud providers."""
        exp_id = generate_id("exp_models")
        providers = providers_to_test or ["gemini", "groq", "mistral", "openrouter", "nvidia"]

        matrix_results = []
        for p_name in providers:
            try:
                start_p = time.perf_counter()
                req = ProviderRequest(
                    messages=[ProviderMessage(role="user", content=prompt)],
                    system_instruction="You are a specialist benchmark evaluation agent."
                )
                # Via the gateway so the benchmark inherits per-provider rate limiting,
                # key quarantine, the deadline budget and health recording; a direct
                # provider call bypassed all four.
                resp = await model_gateway.execute(
                    p_name, req, capability="general", stage_name="provider_matrix",
                )
                lat = round(time.perf_counter() - start_p, 3)

                eval_rep = await self.evaluator.evaluate_answer(
                    question=prompt,
                    answer=resp.content,
                    context={"provider": p_name, "latency_seconds": lat, "total_tokens": resp.total_tokens}
                )

                matrix_results.append({
                    "provider": p_name,
                    "model": resp.model,
                    "score": eval_rep.overall_score,
                    "judge_ran": eval_rep.judge_ran,
                    "scored_dimensions": [s.criterion for s in eval_rep.scores],
                    "latency_seconds": lat,
                    "total_tokens": resp.total_tokens,
                    "status": "success"
                })
            except Exception as e:
                matrix_results.append({
                    "provider": p_name,
                    "status": "failed",
                    "error": str(e)
                })

        exp_record = ExperimentRecord(
            id=exp_id,
            hypothesis="Benchmark provider matrix across quality, latency, and token consumption",
            configuration={"prompt": prompt, "providers": providers},
            status="completed",
            result={"comparisons": matrix_results}
        )
        await self.memory.save_experiment(exp_record)
        return exp_record


# Global default harness
benchmark_harness = BenchmarkHarness()
