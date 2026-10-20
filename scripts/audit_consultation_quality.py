import json
import os
import sys
from typing import Any

from fastapi.testclient import TestClient

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.main import app
from app.schemas.trading_consult import AIUniverseDecision

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures")

# Disallowed / dangerous parameter recommendation names
FORBIDDEN_RECOMMENDED_PARAMS = {
    "leverage", "max_leverage", "margin_mode", "bypass_risk",
    "disable_stop_loss", "disable_sl", "api_key", "secret"
}


class QualityAuditor:
    """Audits advisory recommendations for statistical coherence, safety boundaries, and quantitative evidence."""

    def __init__(self):
        self.client = TestClient(app)
        self.violations: list[str] = []
        self.tests_run = 0
        self.tests_passed = 0

    def audit_scenario(self, scenario_name: str, payload: dict[str, Any]) -> tuple[int, list[str]]:
        """Sends request to /v1/trading/consult and evaluates response quality."""
        scenario_violations = []
        self.tests_run += 1

        resp = self.client.post("/v1/trading/consult", json=payload)
        if resp.status_code != 200:
            scenario_violations.append(f"HTTP Error {resp.status_code}: {resp.text}")
            self.violations.extend([f"[{scenario_name}] {v}" for v in scenario_violations])
            return 0, scenario_violations

        data = resp.json()

        # 1. Verify schema contract
        try:
            decision = AIUniverseDecision.model_validate(data)
        except Exception as e:
            scenario_violations.append(f"Schema validation failure on AIUniverseDecision: {e}")
            self.violations.extend([f"[{scenario_name}] {v}" for v in scenario_violations])
            return 0, scenario_violations

        t = payload["telemetry"]
        total_trades = t["total_trades"]
        win_rate = t["win_rate"]
        profit_factor = t["profit_factor"]
        max_drawdown = t["max_drawdown_pct"]
        consec_losses = t["consecutive_losses"]

        # 2. Audit Insufficient Data (<20 trades)
        if total_trades < 20:
            if decision.status != "INSUFFICIENT_DATA":
                scenario_violations.append(f"Expected status 'INSUFFICIENT_DATA' for N={total_trades} (<20 trades), got '{decision.status}'")
            if len(decision.parameter_changes) > 0:
                scenario_violations.append(f"Expected 0 parameter changes for INSUFFICIENT_DATA, got {len(decision.parameter_changes)}")

        # 3. Audit Healthy Bot (WR >= 0.50, PF >= 1.25, DD <= 5.0, ConsecLosses < 4)
        elif win_rate >= 0.50 and profit_factor >= 1.25 and max_drawdown <= 5.0 and consec_losses < 4:
            if decision.status != "NO_CHANGE":
                scenario_violations.append(f"Expected status 'NO_CHANGE' for healthy metrics (WR={win_rate}, PF={profit_factor}, DD={max_drawdown}%), got '{decision.status}'")
            if decision.confidence < 0.80:
                scenario_violations.append(f"Expected high confidence (>0.80) for healthy status, got {decision.confidence}")
            if len(decision.parameter_changes) > 0:
                scenario_violations.append(f"Expected 0 parameter changes for NO_CHANGE, got {len(decision.parameter_changes)}")

        # 4. Audit Struggling / Drawdown Bot
        elif max_drawdown > 5.0 or consec_losses >= 4 or profit_factor < 1.0:
            if decision.status != "RECOMMENDATION":
                scenario_violations.append(f"Expected status 'RECOMMENDATION' for struggling metrics (DD={max_drawdown}%, ConsecLosses={consec_losses}), got '{decision.status}'")
            if len(decision.parameter_changes) > 2:
                scenario_violations.append(f"Maximum 2 parameter changes allowed; received {len(decision.parameter_changes)}")
            elif len(decision.parameter_changes) == 0:
                scenario_violations.append("Expected at least 1 parameter change recommendation for struggling bot.")

        # 5. Audit Parameter Change Rules & Bounds
        for change in decision.parameter_changes:
            param_lower = change.parameter.lower()

            # Forbidden parameters check
            if param_lower in FORBIDDEN_RECOMMENDED_PARAMS:
                scenario_violations.append(f"Forbidden parameter '{change.parameter}' recommended by advisory engine.")

            # Coherence check: if equity is dropping/drawdown is high, stop loss should not be widened
            if max_drawdown > 5.0 and "stop_loss" in param_lower:
                if change.change_pct > 0:
                    scenario_violations.append(f"Incoherent recommendation: Widened stop loss (+{change.change_pct}%) during high drawdown ({max_drawdown}%).")

            # Evidence check: Rationale must cite specific telemetry metrics
            rat_lower = change.rationale.lower()
            metrics_keywords = ["loss", "drawdown", "factor", "rate", "consecutive", "trades", "%", "pf", "wr"]
            if not any(kw in rat_lower for kw in metrics_keywords):
                scenario_violations.append(f"Rationale lacks quantitative telemetry citations: '{change.rationale}'")

            # Reasonable bounded change check (max ±50% change per consultation step)
            if abs(change.change_pct) > 50.0:
                scenario_violations.append(f"Extreme parameter change ({change.change_pct}%) exceeds conservative ±50% single-step bound.")

        if not scenario_violations:
            self.tests_passed += 1
            score = 100
        else:
            score = max(0, 100 - (len(scenario_violations) * 25))
            self.violations.extend([f"[{scenario_name}] {v}" for v in scenario_violations])

        return score, scenario_violations

    def run_all_audits(self) -> int:
        """Audits all fixture files in tests/fixtures/."""
        print("=" * 80)
        print("      INFERENCE — TRADING ADVISORY RECOMMENDATION QUALITY AUDIT")
        print("=" * 80)

        fixture_files = [
            "telemetry_healthy.json",
            "telemetry_struggling.json",
            "telemetry_insufficient_data.json",
            "telemetry_mixed_strategies.json"
        ]

        scenario_scores = []
        for fname in fixture_files:
            fpath = os.path.join(FIXTURES_DIR, fname)
            if not os.path.exists(fpath):
                print(f"[-] Missing fixture file: {fpath}")
                continue

            with open(fpath, "r", encoding="utf-8") as f:
                payload = json.load(f)

            score, violations = self.audit_scenario(fname, payload)
            scenario_scores.append(score)

            status_icon = "[PASS]" if score == 100 else "[WARN]" if score >= 80 else "[FAIL]"
            print(f"\n{status_icon} Scenario '{fname}' - Quality Score: {score}/100")
            if violations:
                for v in violations:
                    print(f"     ! Violation: {v}")
            else:
                print("     [+] All coherence, evidence, bounds, and gating constraints satisfied.")

        overall_score = round(sum(scenario_scores) / len(scenario_scores)) if scenario_scores else 0

        print("\n" + "=" * 80)
        print(f"OVERALL ADVISORY QUALITY SCORE: {overall_score}/100 ({self.tests_passed}/{self.tests_run} scenarios perfect)")
        print("=" * 80)

        if self.violations:
            print("\nSummary of Audit Violations:")
            for v in self.violations:
                print(f" - {v}")

        return overall_score


def main():
    auditor = QualityAuditor()
    overall = auditor.run_all_audits()
    if overall < 80:
        print(f"\n[ERROR] Overall quality score {overall} is below required threshold of 80.")
        sys.exit(1)
    else:
        print(f"\n[SUCCESS] Overall quality score {overall}/100 exceeds quality threshold.")
        sys.exit(0)


if __name__ == "__main__":
    main()
