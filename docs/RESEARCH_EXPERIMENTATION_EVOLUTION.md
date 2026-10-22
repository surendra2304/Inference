# Research, Experimentation & Strategy Evolution Platform

Inference implements automated hypothesis definition, controlled A/B testing with statistical significance ($p < 0.05$), genetic strategy variant evolution with elitism, and knowledge distillation into persistent rule memory.

---

## 1. Experiment Runner & Statistical Significance

- **A/B Testing Infrastructure** ([`app/experiments/runner.py`](file:///d:/AI%20Universe/app/experiments/runner.py)):
  - Defines formal hypotheses across agent compositions, providers, prompt templates, and modes.
  - Deterministically partitions traffic into control and treatment arms.
  - Computes chi-square proxy $p$-values and automatically triggers configuration rollout recommendations upon reaching target statistical significance ($p < 0.05$).

---

## 2. Strategy Evolution Engine & Genetic Adaptation

- **20-Variant Population** ([`app/learning/strategy_evolution.py`](file:///d:/AI%20Universe/app/learning/strategy_evolution.py)):
  - Explores and exploits across `{agent_composition, provider_selection, prompt_template, mode, confidence_threshold}`.
  - **80/20 Exploitation/Exploration Ratio**: Routes $80\%$ of traffic to elite performers and $20\%$ to explorer variants.
  - **Genetic Elitism & Mutation**: Top strategies are permanently preserved, while the bottom $25\%$ underperforming variants are mutated periodically.

---

## 3. Knowledge Distillation Engine

- **Rule Extraction** ([`app/learning/distillation.py`](file:///d:/AI%20Universe/app/learning/distillation.py)):
  - Distills successful recommendation patterns into structured trigger-action rules (e.g. *when `intent_score > 0.70`, prioritize behavioral telemetry over firmographics*).
  - Automatically queries and injects distilled empirical heuristics into active debate deliberations.

---

## 4. Endpoints Reference

- `GET /v1/experiments`: Active and concluded experiments with $p$-values and recommendations.
- `GET /v1/experiments/strategies`: 20-variant strategy population dashboard and elitism stats.
- `POST /v1/experiments/strategies/evolve`: Mutates bottom variants and updates elite designations.
- `GET /v1/experiments/distilled-rules`: Queries empirical rules for a consumer and task type.
- `POST /v1/experiments/distilled-rules`: Ingests and distills a newly validated intelligence rule.
