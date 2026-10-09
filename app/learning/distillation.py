"""Knowledge Distillation Engine: Distilling Learned Patterns into StrategyBank Rules."""

import time
from typing import Any

from pydantic import BaseModel, Field


class DistilledKnowledgeRule(BaseModel):
    rule_id: str
    consumer: str
    task_type: str
    condition_trigger: str
    prescribed_strategy: str
    #: None when no measurement was supplied.
    empirical_confidence: float | None
    created_at: float = Field(default_factory=time.time)


class KnowledgeDistillationEngine:
    """Extracts empirical rules from multi-consumer outcomes and injects them into future debates."""

    def __init__(self) -> None:
        # Previously seeded with RULE-001..003, each carrying an "empirical_confidence" of
        # 0.89-0.95 that no outcome data produced. Rules now enter only through distill_new_rule,
        # and each carries the confidence its caller measured (None when unmeasured).
        self.rules: list[DistilledKnowledgeRule] = []

    def distill_new_rule(self, consumer: str, task_type: str, condition: str, strategy: str, confidence: float | None) -> DistilledKnowledgeRule:
        rule = DistilledKnowledgeRule(
            rule_id=f"RULE-{len(self.rules)+1:03d}",
            consumer=consumer,
            task_type=task_type,
            condition_trigger=condition,
            prescribed_strategy=strategy,
            empirical_confidence=confidence
        )
        self.rules.append(rule)
        return rule

    def query_distilled_rules(self, consumer: str, task_type: str) -> list[dict[str, Any]]:
        return [r.model_dump() for r in self.rules if r.task_type == task_type or r.consumer == consumer]


knowledge_distillation_engine = KnowledgeDistillationEngine()
