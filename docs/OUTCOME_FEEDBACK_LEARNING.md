# Outcome Feedback Loop & Cross-Consumer Learning Architecture

Inference closes the feedback loop between upstream multi-agent intelligence and downstream execution outcomes across **FORGE**, **Trading Bot**, **Nexus**, and **FRIDAY**.

---

## 1. Downstream Outcome Learning & Strategy Bank

```
┌─────────────────────────────────┐
│     Inference Generation      │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Downstream Consumer Execution   │
│ (Builds, Trades, Conversions)   │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  POST /v1/analytics/outcome     │
│  • consumer & task_type         │
│  • measured_metrics             │
│  • agent_composition            │
└────────────────┬────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌──────────────┐   ┌──────────────┐
│ StrategyBank │   │ Weight Tuning│
│ (90d memory) │   │ (Router Auto)│
└──────────────┘   └──────────────┘
```

---

## 2. Cross-Consumer Insights & Intelligence Quality

- **Exponential Decay Weighting**: Recent outcomes ($\le 24\text{h}$) receive $3\times$ statistical weight over legacy data.
- **Composition Analysis**: Empirically validates that multi-agent debate (e.g. Strategist + Critic: 92.5%) outperforms solo agents (78.4%) by $+13\%$.
- **Automated Alerts**: Triggers warnings if any consumer's downstream rolling success rate drops below $60\%$ over a 24-hour window.

---

## 3. Confidence Calibration & Honesty Curves

- Automatically groups predictions into confidence bins ($0.90-1.00$, $0.80-0.89$, $0.70-0.79$, $<0.70$) and compares stated confidence against empirical ground-truth success.
- Automatically discounts confidence multipliers by $0.85\times$ if historical success rates fall $>15\%$ below stated confidence.

---

## 4. Endpoints Reference

- `POST /v1/analytics/outcome`: Universal downstream outcome ingestion.
- `GET /v1/analytics/calibration`: Calibration curves and honesty metrics.
- `GET /v1/analytics/insights`: Cross-consumer learning patterns and composition accuracy.
- `GET /v1/analytics/strategy-bank?task_type={type}`: Query historical situation patterns and proven recommendations.
