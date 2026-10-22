# Cross-Market & Cross-Asset Intelligence Architecture

Inference delivers unified multi-exchange market data, cross-asset correlation modeling, macro regime intelligence, and venue liquidity scoring.

> [!IMPORTANT]
> **Advisory Invariant**: Inference remains purely an advisory intelligence layer. It NEVER signs transactions, NEVER touches private keys, and NEVER executes live market orders.

---

## 1. Unified Architecture

```
                    +------------------------------------+
                    | Consolidated Multi-Exchange Engine |
                    | (Binance, Bybit, Coinbase, OKX)    |
                    +-----------------+------------------+
                                      |
         +----------------------------+----------------------------+
         |                            |                            |
         v                            v                            v
+------------------+         +------------------+         +------------------+
| Cross-Asset      |         | Market Regime    |         | Venue Liquidity  |
| Correlation      |         | Intelligence     |         | Depth & Slippage |
| (Crypto & Macro) |         | (Transition 48h) |         | Impact Models    |
+--------+---------+         +--------+---------+         +--------+---------+
         |                            |                            |
         +----------------------------+----------------------------+
                                      |
                                      v
                    +------------------------------------+
                    | Cross-Market Multi-Agent Debate    |
                    | (Macro, Liquidity, Correlation)    |
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    | Portfolio-Level Allocation Decision|
                    +------------------------------------+
```

---

## 2. API Endpoints

### 2.1 Consolidated Cross-Exchange Book
`GET /v1/market/cross-exchange/{asset}`
- Returns consolidated orderbook depth, best bids/asks per exchange, and price divergence arbitrage indicators.

### 2.2 Cross-Asset Correlation Matrix
`GET /v1/market/correlations`
- Computes rolling correlations across BTC, ETH, SOL, BNB, S&P 500, Gold, and DXY.

### 2.3 Market Regime Intelligence
`GET /v1/market/regime`
- Macro regime classification (`RISK_ON` / `RISK_OFF`), leading indicators (stablecoin netflows, BTC dominance trend), and 48-hour transition probabilities.

### 2.4 Venue Liquidity & Price Impact
`GET /v1/market/liquidity/{asset}`
- Evaluates liquidity depth and models slippage on \$10k, \$50k, and \$250k order sizes across Binance, Bybit, and Coinbase.

### 2.5 Portfolio-Level Cross-Market Deliberation
`POST /v1/market/portfolio-analysis`
- Evaluates active portfolio holdings against concentration risk, BTC beta exposure, and multi-market opportunities.
