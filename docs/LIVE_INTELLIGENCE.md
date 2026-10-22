# Live Capital Intelligence & Crisis Defense Architecture

Inference provides conservative, capital-preservation-first intelligence when interacting with live deployed trading capital.

> [!IMPORTANT]
> **Advisory Invariant**: Inference remains strictly an advisory intelligence layer. It NEVER signs transactions, NEVER touches private keys, and NEVER executes live market orders.

---

## 1. Live Capital Consultation Profile

When `trading_mode="LIVE"`, the advisory engine automatically enforces strict safety constraints:

| Metric / Rule | Testnet Mode | Live Capital Mode |
| :--- | :--- | :--- |
| **Max Parameter Changes** | 2 | **1 per decision** |
| **Max Change Magnitude** | $\pm 20\%$ | **$\pm 10\%$ max** |
| **Min Confidence Threshold** | $0.60$ | **$0.75$** |
| **Min Sample Size** | 20 trades | **50 trades** |
| **Critic Agent Power** | Advisory Weight | **Veto Authority ($>0.8$ opposition $\rightarrow$ OBSERVATION_ONLY)** |
| **Consensus Target** | $3/5$ Agents | **$4/5$ Agents (Supermajority)** |

---

## 2. Multi-Tier Crisis Protocols

```
[Normal Market Execution]
           │
   DD ≥ 4% │ 3+ Loss Streak
           ▼
┌───────────────────────────────────────┐
│ LEVEL 1: WATCH                        │
│ • Reduce position sizing by 25%       │
│ • Increase signal entry selectivity   │
└──────────────────┬────────────────────┘
                   │ DD ≥ 7.5% | 5+ Loss Streak
                   ▼
┌───────────────────────────────────────┐
│ LEVEL 2: ALERT                        │
│ • Engage defensive posture            │
│ • Reduce sizing by 50%                │
│ • Freeze all optimization suggestions │
└──────────────────┬────────────────────┘
                   │ DD ≥ 12.0% | Daily Loss ≥ 5%
                   ▼
┌───────────────────────────────────────┐
│ LEVEL 3: CRISIS                       │
│ • Recommend immediate entry halt      │
│ • Flatten correlated risk exposure    │
│ • Capital preservation override       │
└───────────────────────────────────────┘
```

---

## 3. Endpoints Reference

- `GET /v1/trading/live/intelligence`: Full live capital status, stress indices, and conservative guidance.
- `GET /v1/trading/live/crisis-status`: Real-time crisis tier evaluation.
- `POST /v1/trading/live/stress-test`: Historical crisis shock simulation (COVID March 2020, FTX collapse, May 2021 flash wick).
- `GET /v1/trading/live/attribution`: Live execution slippage and strategy reliability scoring.
