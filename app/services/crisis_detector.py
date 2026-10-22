"""Crisis Detection and Automated Multi-Tier Defensive Protocols."""

from enum import Enum
from typing import Any


class CrisisLevel(str, Enum):
    LEVEL_0_NORMAL = "NORMAL"
    LEVEL_1_WATCH = "WATCH"
    LEVEL_2_ALERT = "ALERT"
    LEVEL_3_CRISIS = "CRISIS"


class CrisisDetector:
    """Monitors live telemetry for rapid drawdowns, loss streaks, and regime shocks."""

    def evaluate_crisis_level(
        self,
        current_drawdown_pct: float,
        consecutive_losses: int,
        daily_loss_pct: float = 0.0,
        volatility_shock: bool = False
    ) -> dict[str, Any]:
        """Classifies crisis level and generates defensive recommendations."""
        level = CrisisLevel.LEVEL_0_NORMAL
        actions = []
        is_defense_only = False

        # LEVEL 3 (Crisis): Drawdown >= 12% or Daily Loss >= 5% or 7+ loss streak
        if current_drawdown_pct >= 12.0 or daily_loss_pct >= 5.0 or consecutive_losses >= 7:
            level = CrisisLevel.LEVEL_3_CRISIS
            is_defense_only = True
            actions = [
                "HALT_NEW_ENTRIES",
                "CAPITAL_PRESERVATION_ENGAGED",
                "FLATTEN_CORRELATED_EXPOSURE",
                "DISABLE_OPTIMIZATION_RECOMMENDATIONS"
            ]

        # LEVEL 2 (Alert): Drawdown >= 7.5% or 5+ loss streak or volatility shock
        elif current_drawdown_pct >= 7.5 or consecutive_losses >= 5 or volatility_shock:
            level = CrisisLevel.LEVEL_2_ALERT
            is_defense_only = True
            actions = [
                "DEFENSIVE_POSTURE_ENGAGED",
                "REDUCE_POSITION_SIZING_50PCT",
                "TIGHTEN_STOP_LOSS_BOUNDS"
            ]

        # LEVEL 1 (Watch): Drawdown >= 4.0% or 3+ loss streak
        elif current_drawdown_pct >= 4.0 or consecutive_losses >= 3:
            level = CrisisLevel.LEVEL_1_WATCH
            actions = [
                "REDUCE_POSITION_SIZING_25PCT",
                "INCREASE_SIGNAL_SELECTIVITY"
            ]

        return {
            "crisis_level": level.value,
            "is_defense_only_mode": is_defense_only,
            "recommended_defensive_actions": actions,
            "telemetry_evaluated": {
                "drawdown_pct": current_drawdown_pct,
                "consecutive_losses": consecutive_losses,
                "daily_loss_pct": daily_loss_pct,
                "volatility_shock": volatility_shock
            }
        }


crisis_detector = CrisisDetector()
