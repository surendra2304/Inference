# Strategy Evolution Intelligence & Overfitting Diagnostics

Inference provides advanced quantitative evaluation, overfitting detection (Deflated Sharpe Ratio, Probability of Backtest Overfitting), cross-regime robustness analysis, and genetic population health tracking for algorithmic trading systems.

> [!IMPORTANT]
> **Advisory Invariant**: Inference evaluates evolved strategies and outputs structured recommendations. It NEVER activates or modifies trading strategies directly on the exchange.

---

## 1. 5-Specialist Strategy Evaluation Panel

Evolved strategy candidates undergo structured evaluation by 5 domain specialists:

1. **Quantitative Analyst**: Validates sample sizes, win rates, and raw Sharpe ratios.
2. **Risk Analyst**: Assesses tail-risk drawdown and profit factor durability.
3. **Overfitting Detective**: Calculates Deflated Sharpe Ratio (DSR), Probability of Backtest Overfitting (PBO), and curve-fitting penalties.
4. **Market Regime Expert**: Evaluates performance across Bull, Bear, High Volatility, and Sideways Chop regimes.
5. **Contrarian (with Veto Power)**: Probes sensitivity to execution slippage, spread widening, and hidden distribution tails.

---

## 2. Statistical Overfitting Diagnostics

- **Deflated Sharpe Ratio (DSR)**: Adjusts the observed Sharpe ratio for selection bias and the number of backtest trials tested.
- **Probability of Backtest Overfitting (PBO)**: Quantifies the likelihood that in-sample performance was produced by data mining.
- **Minimum Backtest Length (MinBTL)**: Computes the required sample window in days to statistically reject the null hypothesis of spurious alpha.

---

## 3. Endpoints Reference

### 3.1 Candidate Strategy Evaluation
`POST /v1/evolution/evaluate`
- Conducts full 5-specialist evaluation debate and generates composite scoring.

### 3.2 Overfitting Check
`POST /v1/evolution/overfitting-check`
- Computes DSR, PBO, MinBTL, and emits `ACCEPT_ROBUST`, `TEST_LONGER`, or `REJECT_OVERFITTED` verdicts.

### 3.3 Regime Robustness Test
`POST /v1/evolution/regime-test`
- Evaluates strategy drawdown across Bull, Bear, and Chop regimes and identifies whipsaw transition vulnerabilities.

### 3.4 Genetic Population Health & Trends
`GET /v1/evolution/trends`
- Analyzes population diversity and recommends mutation rate adjustments to avoid premature convergence.
