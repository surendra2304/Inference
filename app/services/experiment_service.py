"""A/B Experiment Management Service for Algorithmic Trading Bots."""

import time
from datetime import datetime, timezone
from typing import Any, Literal

from app.memory.base import BaseMemory
from app.memory.sqlite import SQLiteMemory
from app.schemas.trading_consult import (
    ExperimentConfigResponse,
    ExperimentResultsResponse,
    ExperimentStartRequest,
    ExperimentStatusResponse,
)
from app.utils.logger import logger


class ExperimentService:
    """
    Manages A/B trading experiments tracking CONTROL vs TREATMENT arms.
    Stores experiment metadata, arm telemetry snapshots, and computes comparative results.
    """

    def __init__(self, memory: BaseMemory | None = None) -> None:
        self.memory = memory or SQLiteMemory()
        # In-memory fast cache of active experiments
        self._experiments: dict[str, dict[str, Any]] = {}

    def start_experiment(self, req: ExperimentStartRequest) -> ExperimentConfigResponse:
        """Initializes and registers a new A/B trading experiment."""
        now_iso = datetime.now(timezone.utc).isoformat()
        exp_data: dict[str, Any] = {
            "experiment_id": req.experiment_id,
            "hypothesis": req.hypothesis,
            "duration_hours": req.duration_hours,
            "start_time": now_iso,
            "start_timestamp": time.time(),
            "status": "ACTIVE",
            "success_metrics": req.success_metrics,
            "control_bot_id": req.control_bot_id,
            "treatment_bot_id": req.treatment_bot_id,
            "initial_parameters": req.initial_parameters or {},
            "consultations": {"CONTROL": 0, "TREATMENT": 0},
            "telemetry_history": {"CONTROL": [], "TREATMENT": []},
            "latest_telemetry": {"CONTROL": None, "TREATMENT": None}
        }
        self._experiments[req.experiment_id] = exp_data

        logger.info(
            "A/B Experiment '%s' initialized: Control=%s, Treatment=%s, Duration=%.1fh",
            req.experiment_id, req.control_bot_id, req.treatment_bot_id, req.duration_hours
        )

        return ExperimentConfigResponse(
            experiment_id=req.experiment_id,
            status="ACTIVE",
            start_time=now_iso,
            duration_hours=req.duration_hours,
            control_config={
                "bot_id": req.control_bot_id,
                "arm": "CONTROL",
                "parameters": req.initial_parameters
            },
            treatment_config={
                "bot_id": req.treatment_bot_id,
                "arm": "TREATMENT",
                "parameters": req.initial_parameters
            },
            success_metrics=req.success_metrics,
            message="A/B Experiment successfully initialized."
        )

    def record_consultation(
        self,
        experiment_id: str,
        arm: str,
        telemetry: dict[str, Any],
        decision_id: str
    ) -> None:
        """Records a consultation event and telemetry snapshot for an arm."""
        if experiment_id not in self._experiments:
            return
        exp = self._experiments[experiment_id]
        arm_upper = arm.upper()
        if arm_upper in exp["consultations"]:
            exp["consultations"][arm_upper] += 1
            exp["latest_telemetry"][arm_upper] = telemetry
            exp["telemetry_history"][arm_upper].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "decision_id": decision_id,
                "telemetry": telemetry
            })

    def get_status(self, experiment_id: str) -> ExperimentStatusResponse | None:
        """Returns the current status and telemetry summary of an experiment."""
        if experiment_id not in self._experiments:
            return None
        exp = self._experiments[experiment_id]
        elapsed = (time.time() - exp["start_timestamp"]) / 3600.0
        is_completed = elapsed >= exp["duration_hours"]

        return ExperimentStatusResponse(
            experiment_id=experiment_id,
            status="COMPLETED" if is_completed else exp["status"],
            start_time=exp["start_time"],
            elapsed_hours=round(elapsed, 2),
            duration_hours=exp["duration_hours"],
            active_arms=[exp["control_bot_id"], exp["treatment_bot_id"]],
            consultations_count=exp["consultations"],
            latest_telemetry=exp["latest_telemetry"]
        )

    def get_results(self, experiment_id: str) -> ExperimentResultsResponse | None:
        """Computes comparative analysis and aggregates results across arms."""
        if experiment_id not in self._experiments:
            return None
        exp = self._experiments[experiment_id]
        elapsed = (time.time() - exp["start_timestamp"]) / 3600.0

        ctrl_telem = exp["latest_telemetry"].get("CONTROL") or {}
        treat_telem = exp["latest_telemetry"].get("TREATMENT") or {}

        # Default metric fallbacks if no telemetry recorded yet
        c_pf = ctrl_telem.get("profit_factor", 1.0)
        t_pf = treat_telem.get("profit_factor", 1.0)
        c_wr = ctrl_telem.get("win_rate", 0.5)
        t_wr = treat_telem.get("win_rate", 0.5)
        c_dd = ctrl_telem.get("max_drawdown_pct", 5.0)
        t_dd = treat_telem.get("max_drawdown_pct", 5.0)
        c_trades = ctrl_telem.get("total_trades", 0)
        t_trades = treat_telem.get("total_trades", 0)

        # Comparative metrics
        pf_diff = round(t_pf - c_pf, 2)
        wr_diff_pct = round((t_wr - c_wr) * 100.0, 2)
        dd_diff_pct = round(t_dd - c_dd, 2)

        # Winner scoring based on success metrics
        treatment_score = 0
        control_score = 0

        if t_pf > c_pf:
            treatment_score += 1
        elif c_pf > t_pf:
            control_score += 1

        if t_wr > c_wr:
            treatment_score += 1
        elif c_wr > t_wr:
            control_score += 1

        if t_dd < c_dd:
            treatment_score += 1
        elif c_dd < t_dd:
            control_score += 1

        winner: Literal["CONTROL", "TREATMENT", "INCONCLUSIVE"]
        if t_pf > c_pf * 1.05 and t_wr >= c_wr * 0.98 and t_trades >= 10:
            winner = "TREATMENT"
            conclusion = (
                f"Treatment demonstrated superior profit factor ({t_pf:.2f} vs {c_pf:.2f}) "
                f"while maintaining comparable win rate ({t_wr*100:.1f}% vs {c_wr*100:.1f}%). "
                f"Calibrated parameter changes generated higher statistical expectancy."
            )
        elif c_pf > t_pf * 1.05 and c_trades >= 10:
            winner = "CONTROL"
            conclusion = (
                f"Control outperformed Treatment (PF: {c_pf:.2f} vs {t_pf:.2f} | "
                f"MaxDD: {c_dd:.2f}% vs {t_dd:.2f}%). Treatment modifications did not demonstrate superior expectancy."
            )
        else:
            winner = "INCONCLUSIVE"
            conclusion = (
                f"Sample size or delta between arms is insufficient for statistical significance. "
                f"Both arms operating at comparable expectancy levels (Delta PF: {pf_diff:+.2f})."
            )

        status_str: Literal["ACTIVE", "COMPLETED", "TERMINATED"] = (
            "COMPLETED" if elapsed >= exp["duration_hours"] else ("ACTIVE" if exp.get("status") != "TERMINATED" else "TERMINATED")
        )

        return ExperimentResultsResponse(
            experiment_id=experiment_id,
            status=status_str,
            winner=winner,
            duration_hours=exp["duration_hours"],
            control_summary={
                "bot_id": exp["control_bot_id"],
                "total_trades": c_trades,
                "win_rate": c_wr,
                "profit_factor": c_pf,
                "max_drawdown_pct": c_dd,
                "consultations_count": exp["consultations"]["CONTROL"]
            },
            treatment_summary={
                "bot_id": exp["treatment_bot_id"],
                "total_trades": t_trades,
                "win_rate": t_wr,
                "profit_factor": t_pf,
                "max_drawdown_pct": t_dd,
                "consultations_count": exp["consultations"]["TREATMENT"]
            },
            comparison_analysis={
                "profit_factor_delta": pf_diff,
                "win_rate_delta_pct": wr_diff_pct,
                "drawdown_delta_pct": dd_diff_pct,
                "sample_significance": (t_trades >= 20 and c_trades >= 20)
            },
            conclusion=conclusion
        )


# Global singleton experiment service
experiment_service = ExperimentService()
