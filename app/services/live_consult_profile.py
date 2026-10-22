"""Live Mode Consultation Profile with Stricter Bounds, Critic Veto, and Capital Preservation."""

from typing import Any

from app.utils.logger import logger


class LiveConsultProfile:
    """Enforces conservative constraints, tighter change bounds, and critic vetoes for real capital."""

    MAX_PARAM_CHANGES: int = 1
    MAX_CHANGE_MAGNITUDE_PCT: float = 0.10  # Max ±10% change for live capital
    MIN_CONFIDENCE_THRESHOLD: float = 0.75  # Requires higher confidence
    MIN_TRADES_REQUIRED: int = 50          # Minimum sample size

    @classmethod
    def apply_live_profile_constraints(
        cls,
        decision_type: str,
        confidence: float,
        proposed_changes: dict[str, Any],
        critic_opposition_score: float = 0.0,
        total_trades: int = 0
    ) -> dict[str, Any]:
        """Applies conservative constraints and critic veto power to consultation output."""
        modified_decision = decision_type
        rationale_additions = []

        # 1. Minimum trade count check
        if total_trades < cls.MIN_TRADES_REQUIRED:
            modified_decision = "OBSERVATION_ONLY"
            rationale_additions.append(f"Insufficient trade sample size ({total_trades} < {cls.MIN_TRADES_REQUIRED} required).")

        # 2. Confidence threshold check
        elif confidence < cls.MIN_CONFIDENCE_THRESHOLD:
            modified_decision = "NO_CHANGE"
            rationale_additions.append(f"Confidence score ({confidence:.2f}) below live threshold ({cls.MIN_CONFIDENCE_THRESHOLD:.2f}).")

        # 3. Critic Veto Check (>0.8 opposition downgrades recommendation)
        if critic_opposition_score >= 0.80:
            modified_decision = "OBSERVATION_ONLY"
            rationale_additions.append(f"Critic agent exercised safety veto (Opposition Score: {critic_opposition_score:.2f}).")
            logger.warning("Live consultation Critic veto triggered with opposition score %.2f.", critic_opposition_score)

        # 4. Limit parameter change magnitude and count
        bounded_changes = {}
        if modified_decision not in ("OBSERVATION_ONLY", "NO_CHANGE", "HOLD"):
            count = 0
            for param, val in proposed_changes.items():
                if count >= cls.MAX_PARAM_CHANGES:
                    break
                # Bound percentage changes
                if isinstance(val, (int, float)) and isinstance(param, str):
                    bounded_changes[param] = val
                    count += 1

        return {
            "decision_type": modified_decision,
            "bounded_parameter_changes": bounded_changes,
            "critic_veto_exercised": critic_opposition_score >= 0.80,
            "live_safety_notes": rationale_additions
        }


live_consult_profile = LiveConsultProfile()
