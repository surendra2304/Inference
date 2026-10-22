"""Trading Analyst Specialist Agent for Inference.

Analyzes quantitative trading performance data (win rate, profit factor, consecutive losses,
drawdown, risk:reward) supplied by FRIDAY and advises on strategy adjustments.

STRICT SAFETY CONSTRAINTS:
- NEVER execute trades or call exchange/bot APIs directly.
- Only analyze data, engage in debate/consultation, and return structured recommendations.
"""

from typing import Any

from pydantic import BaseModel, Field

from app.agents.base import Agent, AgentModelConfig


class AIUniverseDecision(BaseModel):
    """Structured decision output containing trading strategy recommendations."""
    recommendation: str = Field(description="Actionable parameter recommendation (e.g. 'Tighten Stop Loss to 0.4%')")
    #: None: the rule that fired has no calibrated accuracy. The former default 0.85 and the fixed
    #: 0.88 / 0.92 / 0.95 per rule were not measured.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="Confidence in this advice (None = not calibrated)")
    evidence: list[str] = Field(default_factory=list, description="Empirical metrics and calculations supporting advice")
    risk_assessment: str = Field(default="", description="Evaluation of current drawdown, risk, and downside exposure")
    suggested_parameters: dict[str, Any] = Field(default_factory=dict, description="Recommended key-value parameter changes")
    dissent_or_alternatives: list[str] = Field(default_factory=list, description="Alternative viewpoints or counter-risks")


class TradingAnalyst:
    """Quantitative trading analysis specialist."""

    def __init__(self, agent_id: str = "trading_analyst") -> None:
        self.agent_id = agent_id

    def evaluate_performance(self, metrics: dict[str, Any]) -> AIUniverseDecision:
        """
        Evaluates trading metrics (win rate, profit factor, consecutive losses, max drawdown)
        and derives algorithmic parameter adjustments with clear evidence.
        """
        win_rate = float(metrics.get("win_rate", metrics.get("win_rate_pct", 50.0)))
        profit_factor = float(metrics.get("profit_factor", 1.0))
        consecutive_losses = int(metrics.get("consecutive_losses", 0))
        max_drawdown = float(metrics.get("max_drawdown", metrics.get("drawdown_pct", 0.0)))
        total_trades = int(metrics.get("total_trades", metrics.get("closed_trades", 0)))
        unrealized_pnl = float(metrics.get("unrealized_pnl", 0.0))

        evidence: list[str] = []
        evidence.append(f"Win Rate: {win_rate:.1f}% across {total_trades} closed trades")
        evidence.append(f"Profit Factor: {profit_factor:.2f}")
        evidence.append(f"Max Drawdown: {max_drawdown:.2f}%")
        evidence.append(f"Consecutive Losses: {consecutive_losses}")
        if unrealized_pnl != 0.0:
            evidence.append(f"Active Unrealized PnL: ${unrealized_pnl:,.2f} USDT")

        suggested_params: dict[str, Any] = {}
        alternatives: list[str] = []

        # Decision Logic:
        # Case 1: High Drawdown or excessive consecutive losses -> Capital Preservation
        if max_drawdown > 5.0 or consecutive_losses >= 4:
            recommendation = "Tighten Stop Loss to 0.4% and reduce leverage ceiling to 5x"
            suggested_params = {"stop_loss_pct": 0.004, "max_leverage": 5, "position_size_pct": 0.01}
            risk_assessment = "HIGH RISK: Recent consecutive drawdowns threaten capital preservation thresholds."
            alternatives.append("Temporarily pause new scalper entries until volatility regime normalizes.")
            confidence = None  # rule-based: no calibrated accuracy

        # Case 2: Sub-optimal Profit Factor (PF < 1.1) with low win rate -> Calibrate SL/TP Geometry
        elif profit_factor < 1.1 and win_rate < 45.0:
            recommendation = "Tighten Stop Loss to 0.4% and expand Take Profit to 0.6% (1:1.5 R:R)"
            suggested_params = {"stop_loss_pct": 0.004, "take_profit_pct": 0.006}
            risk_assessment = "MODERATE RISK: Expectancy is dragged down by asymmetric friction and early exits."
            alternatives.append("Switch to trend-following filters on 15m timeframe to avoid chop stop-outs.")
            confidence = None  # rule-based: no calibrated accuracy

        # Case 3: High Win Rate (WR > 65%) and Healthy Profit Factor (PF > 1.5) -> Safe Scaling
        elif win_rate >= 60.0 and profit_factor >= 1.5:
            recommendation = "Maintain 0.5% SL / 0.3% TP scalper baseline; enable trailing stop after +0.2% gain"
            suggested_params = {"stop_loss_pct": 0.005, "take_profit_pct": 0.003, "trailing_stop_activation": 0.002}
            risk_assessment = "LOW RISK: System exhibits statistically robust edge in current market regime."
            alternatives.append("Increase position size allocation by 0.5% on A+ setups.")
            confidence = None  # rule-based: no calibrated accuracy

        # Case 4: Balanced / Baseline
        else:
            recommendation = "Maintain standard risk parameters (0.5% SL / 0.3% TP); monitor 5m ATR regime"
            suggested_params = {"stop_loss_pct": 0.005, "take_profit_pct": 0.003}
            risk_assessment = "STABLE: Bot is operating within expected statistical boundaries."
            alternatives.append("Tighten cooldown window between consecutive signals on the same symbol.")
            confidence = None  # rule-based: no calibrated accuracy

        return AIUniverseDecision(
            recommendation=recommendation,
            confidence=confidence,
            evidence=evidence,
            risk_assessment=risk_assessment,
            suggested_parameters=suggested_params,
            dissent_or_alternatives=alternatives,
        )


def create_trading_analyst_agent() -> Agent:
    """Factory creating the Trading Analyst specialist Agent definition."""
    return Agent(
        id="trading_analyst",
        name="Quantitative Trading Analyst",
        role="Trading Analyst",
        purpose="Analyze quantitative trading metrics, risk-reward ratios, drawdown curves, and advise on strategy parameter calibration.",
        system_instructions=(
            "You are the Quantitative Trading Analyst in Inference. Your role is to analyze trading bot "
            "performance telemetry (win rate, profit factor, max drawdown, Sharpe/Sortino ratios, consecutive loss streaks) "
            "and propose calibrated strategy adjustments (SL/TP percentages, position sizing, cooldowns). "
            "Strict Invariant: You NEVER execute trades or call exchange APIs directly; you only analyze and advise FRIDAY."
        ),
        model_provider="openrouter",
        model_name="deepseek/deepseek-v4-flash:free",
        models=[
            AgentModelConfig(provider="openrouter", model="deepseek/deepseek-v4-flash:free", capability="reasoning"),
            AgentModelConfig(provider="gemini", model="gemini-3.6-flash", capability="reasoning"),
            AgentModelConfig(provider="nvidia", model="nvidia/nemotron-3-ultra-550b-a55b", capability="reasoning"),
        ],
        strengths=["quantitative trading analysis", "risk-adjusted return modeling", "drawdown mitigation", "statistical expectancy"],
        weaknesses=["direct execution authority (strictly disallowed)"],
        metadata={"domain": "algorithmic_trading", "safety_constraint": "ADVISORY_ONLY"}
    )
