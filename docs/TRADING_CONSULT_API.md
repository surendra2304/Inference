# Trading Consultation, A/B Testing & Testnet API Reference

The **Trading Consultation API** exposes Inference as an asynchronous advisory intelligence layer for autonomous trading bots. It provides multi-agent deliberation over quantitative performance metrics, parameter adjustments, drawdown risk assessments, **A/B experimental arm evaluations (Control vs. Treatment)**, and **Testnet Live Environment Safety Safeguards**.

> [!IMPORTANT]
> **Safety Rule & Invariant**: Inference is strictly an **advisory system**. It never interacts with exchange endpoints, never holds or signs private keys, and never executes trades. Trading bots independently validate all returned parameter recommendations against hard deterministic safety bounds.

---

## Architecture Overview

When a trading bot submits performance telemetry, Inference convenes a dedicated multi-agent specialist panel:
1. **Trading Analyst (`trading_analyst`)**: Analyzes win rate, profit factor, drawdown curves, and consecutive loss streaks to propose initial parameter calibrations.
2. **Strategist (`strategist`)**: Evaluates trade-offs between capital preservation and trade frequency across active strategies and ensures A/B treatment arms do not diverge excessively from control baselines.
3. **Adversarial Critic (`critic`)**: Challenges curve-fitting risks, small-sample overfitting, and regime shifts.
4. **Data Analyst (`data_analyst`)**: Quantitatively verifies empirical distributions, mathematical significance thresholds, and cross-arm delta statistics.
5. **Synthesizer (`synthesizer`)**: Emits a bounded `AIUniverseDecision` respecting hard constraints.

### Synthesis Decision Rules
- **Maximum 2 parameter adjustments** per decision (prefers 1).
- **Mandatory Quantitative Evidence**: Every adjustment must cite specific telemetry metrics (e.g. *"consecutive_losses=6 justifies tightening stop_loss_pct by 15%"*).
- **Sample Size Gate**: If total closed trades $N < 20$, the system strictly returns `status = "INSUFFICIENT_DATA"`.
- **Healthy Gate**: If win rate $\ge 50\%$, profit factor $\ge 1.25$, and max drawdown $\le 5\%$, the system returns `status = "NO_CHANGE"`.
- **A/B Testing Control Bound**: When consulting for a `TREATMENT` arm with `control_metrics`, adjustments are bounded conservatively ($\le \pm 15\%$) to preserve comparative experimental validity.
- **Testnet Conservative Mode**: When `trading_mode == "TESTNET"`, stop losses are tightened by 10% more than paper simulations (e.g., 20% tightening), position sizing is constrained to $\le 0.8\times$, and dedicated `testnet_risk_assessment` metrics are emitted.

---

## Testnet Endpoints & Tracking

### 1. Testnet Performance Overview
`GET /v1/trading/testnet/performance`

Aggregates historical testnet consultations and compares performance distributions against paper trading.

#### Response Payload (`TestnetPerformanceResponse`)
```json
{
  "total_consultations": 42,
  "testnet_consultations": 18,
  "paper_consultations": 24,
  "testnet_metrics": {
    "count": 18,
    "avg_win_rate": 0.512,
    "avg_profit_factor": 1.34,
    "avg_drawdown_pct": 4.12,
    "total_trades_analyzed": 680
  },
  "paper_metrics": {
    "count": 24,
    "avg_win_rate": 0.548,
    "avg_profit_factor": 1.48,
    "avg_drawdown_pct": 3.85,
    "total_trades_analyzed": 1120
  },
  "drawdown_distribution": {
    "testnet_low_dd_pct": 14,
    "testnet_high_dd_pct": 4,
    "paper_low_dd_pct": 21,
    "paper_high_dd_pct": 3
  }
}
```

---

### 2. Testnet vs. Paper Comparison
`GET /v1/trading/testnet/comparison`

Compares testnet execution dynamics against paper simulations and highlights strategy divergences.

#### Response Payload (`TestnetComparisonResponse`)
```json
{
  "comparison_timestamp": "2026-08-27T17:00:00.000000",
  "testnet_summary": {
    "count": 18,
    "avg_win_rate": 0.512,
    "avg_profit_factor": 1.34,
    "avg_drawdown_pct": 4.12
  },
  "paper_summary": {
    "count": 24,
    "avg_win_rate": 0.548,
    "avg_profit_factor": 1.48,
    "avg_drawdown_pct": 3.85
  },
  "strategy_divergence": [
    {
      "strategy": "Supertrend_Breakout",
      "testnet_pf": 1.34,
      "paper_pf": 1.48,
      "slippage_impact": "Medium (Spread friction detected on testnet fills)",
      "recommended_action": "Tighter stop loss and wider take profit multiple on testnet"
    }
  ],
  "recommendations_summary": "Testnet executions encounter realistic orderbook depth and latency, justifying a 10% tighter stop loss and 0.8x position sizing relative to paper simulations."
}
```

---

## Consultation with Testnet Telemetry

`POST /v1/trading/consult`

#### Request Payload (`TradingConsultRequest` with `testnet_specific`)
```json
{
  "bot_id": "testnet_scalper_01",
  "trading_mode": "TESTNET",
  "testnet_specific": {
    "testnet_equity": 9800.0,
    "testnet_drawdown_pct": 7.2,
    "testnet_daily_loss": 200.0,
    "testnet_open_positions": 2,
    "testnet_margin_level": 140.0
  },
  "telemetry": {
    "equity": 9800.0,
    "unrealized_pnl": -40.0,
    "realized_pnl": -200.0,
    "win_rate": 0.38,
    "profit_factor": 0.76,
    "max_drawdown_pct": 7.2,
    "consecutive_losses": 4,
    "total_trades": 36
  },
  "current_parameters": {
    "Supertrend_5m": {
      "stop_loss_pct": 0.02,
      "take_profit_pct": 0.03,
      "position_size_usdt": 100.0
    }
  },
  "consultation_reason": "DRAWDOWN_EVENT"
}
```

#### Response Payload (`AIUniverseDecision`)
```json
{
  "decision_id": "b6a82741-f761-486a-bca9-813c01826ca4",
  "timestamp": "2026-08-27T17:00:00.000000",
  "status": "RECOMMENDATION",
  "confidence": 0.88,
  "parameter_changes": [
    {
      "strategy": "Supertrend_5m",
      "parameter": "stop_loss_pct",
      "current_value": 0.02,
      "recommended_value": 0.016,
      "change_pct": -20.0,
      "rationale": "Consecutive losses (4) or Max Drawdown (7.20%) on Supertrend_5m justifies tightening stop_loss_pct by 20.0% for capital preservation [Testnet Conservative Safety]."
    }
  ],
  "risk_assessment": "ELEVATED RISK: Account max drawdown reached 7.20% with 4 consecutive losses. Capital preservation protocol activated.",
  "regime_analysis": "Market regime indicates chop or unfavorable volatility for current breakout/trend parameters.",
  "dissent_notes": "Critic noted that wider stop losses in this regime increase tail-risk exposure; tightening stop loss is strictly indicated.",
  "debate_summary": "Multi-Agent Deliberation:\n- TradingAnalyst: Quantitative Analysis: Evaluated win rate...\n- Strategist: Strategic Assessment: Prioritizing capital preservation...\n- Critic: Adversarial Review: Scrutinized sample size reliability...",
  "valid_until": "2026-08-28T17:00:00.000000",
  "testnet_risk_assessment": "TESTNET RISK ASSESSMENT: Operating on live testnet infrastructure with real market depth & orderbook fills. Testnet Equity: $9,800.00 | Current Drawdown: 7.20% | Margin Level: 140.0% | Open Positions: 2. WARNING: Margin level approaching critical buffer (<150%). Sizing reductions strictly enforced. CRITICAL: Testnet drawdown exceeds 6.0%. Capital preservation protocol engaged. Recommended Sizing: Maintain position sizing at <= 0.8x paper trading standard. Tighten stop loss by 10%."
}
```

---

## A/B Testing Endpoints

- `POST /v1/trading/experiment/start`: Register A/B experiments.
- `GET /v1/trading/experiment/{id}/status`: Track experiment status.
- `GET /v1/trading/experiment/{id}/results`: Aggregate comparative results.
