"""
Inference/services/reasoning_chain_optimizer.py — Inference Reasoning Chain Optimizer.

Optimizes reasoning chains for multi-step problems: finds the most efficient
reasoning path, eliminates redundant steps, suggests better intermediate
representations, and provides cost-optimized reasoning strategies.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""
    step_id: str
    step_number: int
    description: str
    input_representation: str = ""
    operation: str = ""  # derive, infer, transform, compute, verify
    output_representation: str = ""
    cost: float = 1.0  # computational cost
    confidence: float = 0.9
    is_redundant: bool = False


@dataclass
class ReasoningChain:
    """A chain of reasoning steps."""
    chain_id: str
    problem_statement: str
    steps: list[ReasoningStep] = field(default_factory=list)
    total_cost: float = 0.0
    total_confidence: float = 0.0
    estimated_length: int = 0
    optimality_score: float = 0.0


@dataclass
class OptimizationSuggestion:
    """A suggestion for optimizing a reasoning chain."""
    suggestion_id: str
    chain_id: str
    suggestion_type: str  # remove_step, merge_steps, reorder, alternative_representation, skip_verification
    target_step: int = 0
    description: str = ""
    potential_savings: float = 0.0
    confidence: float = 0.0


@dataclass
class AlternativeRepresentation:
    """An alternative representation for a reasoning step."""
    representation_id: str
    original_representation: str
    alternative: str
    representation_type: str  # symbolic, numeric, graph, textual, matrix
    reason: str = ""
    estimated_efficiency: float = 0.0


@dataclass
class ReasoningStrategy:
    """A reasoning strategy profile."""
    strategy_id: str
    strategy_name: str  # forward_chaining, backward_chaining, divide_conquer, analogical, hybrid
    description: str = ""
    estimated_steps: int = 0
    estimated_cost: float = 0.0
    success_probability: float = 0.0
    applicable_problem_types: list[str] = field(default_factory=list)


@dataclass
class OptimizedChain:
    """An optimized reasoning chain."""
    original_chain_id: str
    optimized_chain_id: str
    steps_before: int = 0
    steps_after: int = 0
    cost_before: float = 0.0
    cost_after: float = 0.0
    efficiency_improvement: float = 0.0
    optimized_steps: list[dict] = field(default_factory=list)


class ReasoningChainOptimizer:
    """Reasoning chain optimization engine."""

    def __init__(self):
        self._chains: dict[str, ReasoningChain] = {}
        self._suggestions: list[OptimizationSuggestion] = []
        self._representations: list[AlternativeRepresentation] = []
        self._strategies: dict[str, ReasoningStrategy] = {}
        self._optimized: list[OptimizedChain] = []

        # Register default strategies
        self._register_strategies()

    def _register_strategies(self):
        """Register reasoning strategies."""
        strategies = [
            ("forward_chaining", "Forward Chaining", "Start from premises and derive conclusions", "deductive, mathematical"),
            ("backward_chaining", "Backward Chaining", "Start from goal and work backwards", "diagnostic, goal-oriented"),
            ("divide_conquer", "Divide and Conquer", "Break problem into subproblems", "complex, multi-part"),
            ("analogical", "Analogical Reasoning", "Use analogies from similar problems", "creative, unfamiliar"),
            ("hybrid", "Hybrid Strategy", "Combine multiple strategies", "complex, multi-domain"),
        ]
        for sid, name, desc, problem_types in strategies:
            self._strategies[sid] = ReasoningStrategy(
                strategy_id=hashlib.md5(f"strategy:{sid}").hexdigest()[:12],
                strategy_name=name,
                description=desc,
                estimated_steps=5 + len(problem_types) * 3,
                estimated_cost=10.0 + len(problem_types) * 5,
                success_probability=0.7 + (ord(hashlib.md5(sid.encode()).hexdigest()[:1]) % 25) / 100,
                applicable_problem_types=problem_types,
            )

    def create_chain(self, problem_statement: str, steps: list[dict]) -> ReasoningChain:
        """Create a reasoning chain."""
        chain = ReasoningChain(
            chain_id=hashlib.md5(
                f"chain:{problem_statement[:50]}:{time.time()}".encode()
            ).hexdigest()[:12],
            problem_statement=problem_statement,
        )

        total_cost = 0.0
        total_confidence = 0.0

        for i, step_data in enumerate(steps):
            step = ReasoningStep(
                step_id=hashlib.md5(
                    f"step:{chain.chain_id}:{i}:{time.time()}".encode()
                ).hexdigest()[:12],
                step_number=i + 1,
                description=step_data.get("description", ""),
                input_representation=step_data.get("input", ""),
                operation=step_data.get("operation", "derive"),
                output_representation=step_data.get("output", ""),
                cost=step_data.get("cost", 1.0),
                confidence=step_data.get("confidence", 0.9),
            )
            chain.steps.append(step)
            total_cost += step.cost
            total_confidence += step.confidence

        chain.total_cost = total_cost
        chain.total_confidence = total_confidence / max(1, len(chain.steps))
        chain.estimated_length = len(chain.steps)
        chain.optimality_score = 1.0 - (total_cost / max(1, len(chain.steps) * 5))

        self._chains[chain.chain_id] = chain
        return chain

    def analyze_chain(self, chain_id: str) -> list[OptimizationSuggestion]:
        """Analyze a reasoning chain for optimization opportunities."""
        chain = self._chains.get(chain_id)
        if not chain:
            raise HTTPException(404, f"Chain {chain_id} not found")

        suggestions = []

        # Check for redundant steps
        seen_outputs = set()
        for i, step in enumerate(chain.steps):
            if step.output_representation in seen_outputs:
                suggestion = OptimizationSuggestion(
                    suggestion_id=hashlib.md5(
                        f"suggestion:remove:{chain_id}:{i}:{time.time()}".encode()
                    ).hexdigest()[:12],
                    chain_id=chain_id,
                    suggestion_type="remove_step",
                    target_step=i + 1,
                    description=f"Step {i+1} produces duplicate output",
                    potential_savings=step.cost,
                    confidence=0.9,
                )
                suggestions.append(suggestion)

            seen_outputs.add(step.output_representation)

        # Check for mergeable steps
        for i in range(len(chain.steps) - 1):
            if chain.steps[i].operation == chain.steps[i + 1].operation:
                suggestion = OptimizationSuggestion(
                    suggestion_id=hashlib.md5(
                        f"suggestion:merge:{chain_id}:{i}:{time.time()}".encode()
                    ).hexdigest()[:12],
                    chain_id=chain_id,
                    suggestion_type="merge_steps",
                    target_step=i + 1,
                    description=f"Steps {i+1} and {i+2} can be merged",
                    potential_savings=chain.steps[i].cost,
                    confidence=0.75,
                )
                suggestions.append(suggestion)

        # Check for verification steps that could be skipped
        for i, step in enumerate(chain.steps):
            if step.operation == "verify" and step.confidence > 0.95:
                suggestion = OptimizationSuggestion(
                    suggestion_id=hashlib.md5(
                        f"suggestion:skip:{chain_id}:{i}:{time.time()}".encode()
                    ).hexdigest()[:12],
                    chain_id=chain_id,
                    suggestion_type="skip_verification",
                    target_step=i + 1,
                    description=f"Verification step {i+1} has high confidence and can be skipped",
                    potential_savings=step.cost,
                    confidence=0.6,
                )
                suggestions.append(suggestion)

        # Suggest alternative representations
        for step in chain.steps:
            if step.cost > 2.0:
                suggestion = OptimizationSuggestion(
                    suggestion_id=hashlib.md5(
                        f"suggestion:repr:{chain_id}:{step.step_id}:{time.time()}".encode()
                    ).hexdigest()[:12],
                    chain_id=chain_id,
                    suggestion_type="alternative_representation",
                    target_step=step.step_number,
                    description="Consider using a more efficient representation",
                    potential_savings=step.cost * 0.3,
                    confidence=0.5,
                )
                suggestions.append(suggestion)

        self._suggestions.extend(suggestions)
        return suggestions

    def suggest_representation(
        self,
        original: str,
        operation: str = "",
    ) -> AlternativeRepresentation:
        """Suggest an alternative representation."""
        repr_types = ["symbolic", "numeric", "graph", "textual", "matrix"]
        chosen_type = repr_types[ord(hashlib.md5(original.encode()).hexdigest()[:1]) % len(repr_types)]

        representation = AlternativeRepresentation(
            representation_id=hashlib.md5(
                f"repr:{original[:30]}:{time.time()}".encode()
            ).hexdigest()[:12],
            original_representation=original,
            alternative=f"[{chosen_type}] representation of: {original[:50]}",
            representation_type=chosen_type,
            reason=f"Switching to {chosen_type} representation may reduce computational cost",
            estimated_efficiency=0.6 + (ord(hashlib.md5(original.encode()).hexdigest()[:1]) % 30) / 100,
        )

        self._representations.append(representation)
        return representation

    def recommend_strategy(self, problem_type: str) -> ReasoningStrategy:
        """Recommend a reasoning strategy for a problem type."""
        # Find the best matching strategy
        best_match = None
        best_score = 0.0

        for strategy in self._strategies.values():
            if problem_type in strategy.applicable_problem_types:
                score = strategy.success_probability
            else:
                score = strategy.success_probability * 0.3

            if score > best_score:
                best_score = score
                best_match = strategy

        if not best_match:
            best_match = self._strategies.get("hybrid", list(self._strategies.values())[0])

        return best_match

    def optimize_chain(self, chain_id: str) -> OptimizedChain:
        """Optimize a reasoning chain."""
        chain = self._chains.get(chain_id)
        if not chain:
            raise HTTPException(404, f"Chain {chain_id} not found")

        # Get suggestions
        suggestions = self.analyze_chain(chain_id)

        # Build optimized chain
        optimized_steps = []
        cost_after = 0.0

        remove_indices = set()
        merge_indices = set()
        skip_indices = set()

        for s in suggestions:
            if s.suggestion_type == "remove_step":
                remove_indices.add(s.target_step - 1)
            elif s.suggestion_type == "merge_steps":
                merge_indices.add(s.target_step - 1)
            elif s.suggestion_type == "skip_verification":
                skip_indices.add(s.target_step - 1)

        for i, step in enumerate(chain.steps):
            if i in remove_indices or i in skip_indices:
                continue
            optimized_steps.append({
                "step": i + 1,
                "description": step.description,
                "operation": step.operation,
                "cost": step.cost,
            })
            cost_after += step.cost

        # Apply merge savings
        for i in merge_indices:
            if i + 1 < len(optimized_steps):
                cost_after -= chain.steps[i].cost

        optimized = OptimizedChain(
            original_chain_id=chain_id,
            optimized_chain_id=hashlib.md5(
                f"optimized:{chain_id}:{time.time()}".encode()
            ).hexdigest()[:12],
            steps_before=len(chain.steps),
            steps_after=len(optimized_steps),
            cost_before=chain.total_cost,
            cost_after=max(0.1, cost_after),
            efficiency_improvement=(chain.total_cost - cost_after) / max(0.1, chain.total_cost),
            optimized_steps=optimized_steps,
        )

        self._optimized.append(optimized)
        return optimized

    def get_report(self) -> dict:
        """Get reasoning chain optimizer report."""
        return {
            "total_chains": len(self._chains),
            "total_suggestions": len(self._suggestions),
            "total_representations": len(self._representations),
            "total_strategies": len(self._strategies),
            "total_optimizations": len(self._optimized),
            "avg_efficiency_improvement": sum(o.efficiency_improvement for o in self._optimized) / max(1, len(self._optimized)),
            "available_strategies": [
                {
                    "name": s.strategy_name,
                    "estimated_steps": s.estimated_steps,
                    "success_probability": s.success_probability,
                    "applicable_types": s.applicable_problem_types,
                }
                for s in self._strategies.values()
            ],
            "recent_optimizations": [
                {
                    "steps_before": o.steps_before,
                    "steps_after": o.steps_after,
                    "cost_before": o.cost_before,
                    "cost_after": o.cost_after,
                    "improvement": o.efficiency_improvement,
                }
                for o in self._optimized[-5:]
            ],
        }


# Module-level singleton
reasoning_optimizer = ReasoningChainOptimizer()


# ── API Routes ──────────────────────────────────────────────────────────────


@router.post("/api/v1/inference/reasoning/chains")
async def create_chain(payload: dict):
    """Create a reasoning chain."""
    chain = reasoning_optimizer.create_chain(
        problem_statement=payload["problem_statement"],
        steps=payload.get("steps", []),
    )
    return {"chain_id": chain.chain_id, "steps": len(chain.steps), "cost": chain.total_cost}


@router.get("/api/v1/inference/reasoning/chains/{chain_id}/analyze")
async def analyze_chain(chain_id: str):
    """Analyze a reasoning chain for optimization."""
    suggestions = reasoning_optimizer.analyze_chain(chain_id)
    return {
        "chain_id": chain_id,
        "suggestions": [
            {
                "type": s.suggestion_type,
                "step": s.target_step,
                "description": s.description,
                "savings": s.potential_savings,
                "confidence": s.confidence,
            }
            for s in suggestions
        ],
    }


@router.post("/api/v1/inference/reasoning/chains/{chain_id}/optimize")
async def optimize_chain(chain_id: str):
    """Optimize a reasoning chain."""
    result = reasoning_optimizer.optimize_chain(chain_id)
    return {
        "optimized_chain_id": result.optimized_chain_id,
        "steps_before": result.steps_before,
        "steps_after": result.steps_after,
        "cost_before": result.cost_before,
        "cost_after": result.cost_after,
        "efficiency_improvement": result.efficiency_improvement,
    }


@router.post("/api/v1/inference/reasoning/representations")
async def suggest_representation(payload: dict):
    """Suggest an alternative representation."""
    rep = reasoning_optimizer.suggest_representation(
        original=payload["original"],
        operation=payload.get("operation", ""),
    )
    return {
        "alternative": rep.alternative,
        "type": rep.representation_type,
        "efficiency": rep.estimated_efficiency,
    }


@router.get("/api/v1/inference/reasoning/strategy/{problem_type}")
async def recommend_strategy(problem_type: str):
    """Recommend a reasoning strategy."""
    strategy = reasoning_optimizer.recommend_strategy(problem_type)
    return {
        "strategy": strategy.strategy_name,
        "description": strategy.description,
        "estimated_steps": strategy.estimated_steps,
        "success_probability": strategy.success_probability,
    }


@router.get("/api/v1/inference/reasoning/report")
async def get_report():
    """Get reasoning chain optimizer report."""
    return reasoning_optimizer.get_report()
