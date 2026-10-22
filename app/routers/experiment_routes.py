"""FastAPI Router for Research, Experimentation, Strategy Evolution, and Knowledge Distillation."""

from fastapi import APIRouter, Query, status
from pydantic import BaseModel

from app.experiments.runner import experiment_runner
from app.learning.distillation import knowledge_distillation_engine
from app.learning.strategy_evolution import strategy_evolution_engine

experiment_router = APIRouter(prefix="/v1/experiments", tags=["Research, A/B Testing & Evolution"])


class DistillRuleRequest(BaseModel):
    consumer: str
    task_type: str
    condition_trigger: str
    prescribed_strategy: str
    #: The caller's measured confidence. None when the caller measured none (was a default 0.90).
    empirical_confidence: float | None = None


@experiment_router.get("", status_code=status.HTTP_200_OK)
async def list_active_and_concluded_experiments():
    """Returns all active, paused, and concluded A/B intelligence experiments and statistical evaluations."""
    return experiment_runner.get_experiments()


@experiment_router.get("/strategies", status_code=status.HTTP_200_OK)
async def get_strategy_evolution_population():
    """Returns the population of 20 strategy variants, elitism status, and mutation history."""
    return strategy_evolution_engine.get_population_dashboard()


@experiment_router.post("/strategies/evolve", status_code=status.HTTP_200_OK)
async def trigger_strategy_evolution():
    """Mutates bottom 25% underperforming strategy variants and preserves top elites."""
    return strategy_evolution_engine.evolve_population()


@experiment_router.get("/distilled-rules", status_code=status.HTTP_200_OK)
async def get_distilled_knowledge_rules(consumer: str = Query(default="nexus"), task_type: str = Query(default="lead_qualification")):
    """Queries distilled empirical rules for injection into debates and decision routing."""
    return knowledge_distillation_engine.query_distilled_rules(consumer, task_type)


@experiment_router.post("/distilled-rules", status_code=status.HTTP_201_CREATED)
async def create_distilled_rule(req: DistillRuleRequest):
    """Distills a newly validated intelligence pattern into the persistent knowledge bank."""
    rule = knowledge_distillation_engine.distill_new_rule(
        consumer=req.consumer,
        task_type=req.task_type,
        condition=req.condition_trigger,
        strategy=req.prescribed_strategy,
        confidence=req.empirical_confidence
    )
    return rule.model_dump()
