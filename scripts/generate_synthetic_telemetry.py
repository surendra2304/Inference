"""Synthetic Telemetry Generator for Inference Trading Consultation Testing."""

import json
import os
from typing import Any

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures")


def generate_healthy_telemetry() -> dict[str, Any]:
    """Generates synthetic telemetry for a healthy, profitable trading bot."""
    return {
        "bot_id": "bot_healthy_alpha",
        "trading_mode": "PAPER",
        "experiment_id": "exp_healthy_baseline_01",
        "telemetry": {
            "equity": 12500.0,
            "unrealized_pnl": 150.0,
            "realized_pnl": 2500.0,
            "win_rate": 0.68,
            "profit_factor": 1.85,
            "max_drawdown_pct": 2.1,
            "consecutive_losses": 1,
            "total_trades": 85,
            "sharpe_ratio": 2.35
        },
        "strategy_performance": [
            {
                "strategy_name": "Supertrend_5m",
                "trade_count": 55,
                "win_rate": 0.70,
                "profit_factor": 1.95,
                "net_pnl": 1800.0,
                "avg_win": 45.0,
                "avg_loss": 20.0,
                "consecutive_losses": 1
            },
            {
                "strategy_name": "EMA_Cross_15m",
                "trade_count": 30,
                "win_rate": 0.63,
                "profit_factor": 1.65,
                "net_pnl": 700.0,
                "avg_win": 55.0,
                "avg_loss": 35.0,
                "consecutive_losses": 0
            }
        ],
        "current_parameters": {
            "Supertrend_5m": {
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.035,
                "atr_multiplier": 2.5
            },
            "EMA_Cross_15m": {
                "fast_period": 9,
                "slow_period": 21,
                "stop_loss_pct": 0.025
            }
        },
        "regime_data": {
            "volatility": "normal",
            "trend": "bullish_continuation",
            "adx_14": 28.5
        },
        "recent_trades": [
            {"id": "t_h1", "strategy": "Supertrend_5m", "side": "BUY", "pnl": 42.0, "duration_sec": 360},
            {"id": "t_h2", "strategy": "EMA_Cross_15m", "side": "BUY", "pnl": 58.0, "duration_sec": 720},
            {"id": "t_h3", "strategy": "Supertrend_5m", "side": "SELL", "pnl": -18.0, "duration_sec": 180},
            {"id": "t_h4", "strategy": "Supertrend_5m", "side": "BUY", "pnl": 49.0, "duration_sec": 410}
        ],
        "consultation_reason": "SCHEDULED"
    }


def generate_struggling_telemetry() -> dict[str, Any]:
    """Generates synthetic telemetry for a bot experiencing severe drawdown and loss streaks."""
    return {
        "bot_id": "bot_struggling_beta",
        "trading_mode": "TESTNET",
        "experiment_id": "exp_struggling_drawdown_02",
        "telemetry": {
            "equity": 8800.0,
            "unrealized_pnl": -220.0,
            "realized_pnl": -1200.0,
            "win_rate": 0.35,
            "profit_factor": 0.72,
            "max_drawdown_pct": 9.8,
            "consecutive_losses": 6,
            "total_trades": 64,
            "sharpe_ratio": -0.45
        },
        "strategy_performance": [
            {
                "strategy_name": "Breakout_1m",
                "trade_count": 44,
                "win_rate": 0.31,
                "profit_factor": 0.62,
                "net_pnl": -950.0,
                "avg_win": 22.0,
                "avg_loss": 38.0,
                "consecutive_losses": 5
            },
            {
                "strategy_name": "MeanReversion_5m",
                "trade_count": 20,
                "win_rate": 0.45,
                "profit_factor": 0.90,
                "net_pnl": -250.0,
                "avg_win": 30.0,
                "avg_loss": 33.0,
                "consecutive_losses": 2
            }
        ],
        "current_parameters": {
            "Breakout_1m": {
                "stop_loss_pct": 0.03,
                "take_profit_pct": 0.02,
                "cooldown_seconds": 60
            },
            "MeanReversion_5m": {
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "stop_loss_pct": 0.025
            }
        },
        "regime_data": {
            "volatility": "high_chop",
            "trend": "range_bound_whipsaw",
            "adx_14": 14.2
        },
        "recent_trades": [
            {"id": "t_s1", "strategy": "Breakout_1m", "side": "BUY", "pnl": -38.0, "duration_sec": 120},
            {"id": "t_s2", "strategy": "Breakout_1m", "side": "SELL", "pnl": -40.0, "duration_sec": 95},
            {"id": "t_s3", "strategy": "Breakout_1m", "side": "BUY", "pnl": -35.0, "duration_sec": 140},
            {"id": "t_s4", "strategy": "MeanReversion_5m", "side": "BUY", "pnl": -30.0, "duration_sec": 450}
        ],
        "consultation_reason": "DRAWDOWN_EVENT"
    }


def generate_insufficient_data_telemetry() -> dict[str, Any]:
    """Generates synthetic telemetry for a newly launched bot with <20 total trades."""
    return {
        "bot_id": "bot_newborn_gamma",
        "trading_mode": "PAPER",
        "experiment_id": "exp_newborn_initial_03",
        "telemetry": {
            "equity": 10050.0,
            "unrealized_pnl": 10.0,
            "realized_pnl": 50.0,
            "win_rate": 0.42,
            "profit_factor": 1.10,
            "max_drawdown_pct": 1.2,
            "consecutive_losses": 2,
            "total_trades": 12,
            "sharpe_ratio": 0.80
        },
        "strategy_performance": [
            {
                "strategy_name": "Scalper_3m",
                "trade_count": 12,
                "win_rate": 0.42,
                "profit_factor": 1.10,
                "net_pnl": 50.0,
                "avg_win": 25.0,
                "avg_loss": 18.0,
                "consecutive_losses": 2
            }
        ],
        "current_parameters": {
            "Scalper_3m": {
                "stop_loss_pct": 0.015,
                "take_profit_pct": 0.02
            }
        },
        "regime_data": {
            "volatility": "low",
            "trend": "neutral"
        },
        "recent_trades": [
            {"id": "t_i1", "strategy": "Scalper_3m", "side": "BUY", "pnl": 25.0, "duration_sec": 180},
            {"id": "t_i2", "strategy": "Scalper_3m", "side": "SELL", "pnl": -18.0, "duration_sec": 110}
        ],
        "consultation_reason": "SCHEDULED"
    }


def generate_mixed_strategies_telemetry() -> dict[str, Any]:
    """Generates synthetic telemetry with 3 mixed strategies (1 profitable, 1 failing, 1 neutral)."""
    return {
        "bot_id": "bot_portfolio_delta",
        "trading_mode": "TESTNET",
        "experiment_id": "exp_portfolio_mixed_04",
        "telemetry": {
            "equity": 9800.0,
            "unrealized_pnl": -60.0,
            "realized_pnl": -200.0,
            "win_rate": 0.48,
            "profit_factor": 0.94,
            "max_drawdown_pct": 6.2,
            "consecutive_losses": 4,
            "total_trades": 90,
            "sharpe_ratio": 0.25
        },
        "strategy_performance": [
            {
                "strategy_name": "TrendFollower_1h",
                "trade_count": 30,
                "win_rate": 0.65,
                "profit_factor": 1.80,
                "net_pnl": 650.0,
                "avg_win": 70.0,
                "avg_loss": 40.0,
                "consecutive_losses": 0
            },
            {
                "strategy_name": "ChopScalper_1m",
                "trade_count": 45,
                "win_rate": 0.33,
                "profit_factor": 0.58,
                "net_pnl": -900.0,
                "avg_win": 15.0,
                "avg_loss": 26.0,
                "consecutive_losses": 5
            },
            {
                "strategy_name": "GridNeutral_15m",
                "trade_count": 15,
                "win_rate": 0.53,
                "profit_factor": 1.05,
                "net_pnl": 50.0,
                "avg_win": 20.0,
                "avg_loss": 19.0,
                "consecutive_losses": 1
            }
        ],
        "current_parameters": {
            "ChopScalper_1m": {
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.015,
                "cooldown_seconds": 30
            },
            "TrendFollower_1h": {
                "stop_loss_pct": 0.035,
                "take_profit_pct": 0.08
            },
            "GridNeutral_15m": {
                "grid_spacing_pct": 0.005,
                "num_levels": 10
            }
        },
        "regime_data": {
            "volatility": "expanding",
            "trend": "directional_breakout"
        },
        "recent_trades": [
            {"id": "t_m1", "strategy": "ChopScalper_1m", "side": "BUY", "pnl": -26.0, "duration_sec": 45},
            {"id": "t_m2", "strategy": "TrendFollower_1h", "side": "BUY", "pnl": 95.0, "duration_sec": 3600},
            {"id": "t_m3", "strategy": "ChopScalper_1m", "side": "SELL", "pnl": -28.0, "duration_sec": 60}
        ],
        "consultation_reason": "LOSS_STREAK"
    }


def main():
    """Generates all synthetic fixture files in tests/fixtures/."""
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    scenarios = {
        "telemetry_healthy.json": generate_healthy_telemetry(),
        "telemetry_struggling.json": generate_struggling_telemetry(),
        "telemetry_insufficient_data.json": generate_insufficient_data_telemetry(),
        "telemetry_mixed_strategies.json": generate_mixed_strategies_telemetry()
    }

    for filename, payload in scenarios.items():
        path = os.path.join(FIXTURES_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[+] Generated fixture: {path} ({payload['bot_id']} - {payload['consultation_reason']})")

    print(f"\nSuccessfully generated {len(scenarios)} synthetic telemetry scenarios in {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
