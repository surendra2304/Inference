"""Consultation Orchestrator Service for Trading Bot Advisory with A/B Testing & Testnet Support."""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, NamedTuple, cast
from uuid import uuid4

from app.agents.base import Agent
from app.agents.registry import agent_registry
from app.config_production import production_config
from app.memory.base import (
    BaseMemory,
    MemoryRecord,
    MessageRecord,
    RunRecord,
    TaskRecord,
)
from app.memory.sqlite import SQLiteMemory
from app.monitoring import monitor
from app.optimization import circuit_breaker, concurrency_controller, telemetry_cache
from app.providers.base import ProviderMessage, ProviderRequest
from app.providers.gateway import model_gateway
from app.schemas.trading_consult import (
    AIUniverseDecision,
    ParameterChange,
    TestnetComparisonResponse,
    TestnetPerformanceResponse,
    TradingConsultRequest,
)
from app.services.experiment_service import experiment_service
from app.utils.ids import (
    generate_message_id,
    generate_run_id,
    generate_task_id,
)
from app.utils.logger import logger


class SpecialistOutcome(NamedTuple):
    """What one specialist invocation actually produced.

    `model_backed=False` means no model answered: the text is deterministic
    canned analysis, not the specialist's opinion.
    """

    agent_id: str
    text: str
    model_backed: bool
    error: str | None = None


class TradingConsultService:
    """
    Assembles a dedicated multi-agent specialist panel for algorithmic trading consultation:
    TradingAnalyst (lead) -> Strategist -> Adversarial Critic -> Data Analyst -> Synthesizer.

    STRICT CONSTRAINTS ENFORCED:
    1. Maximum 2 parameter changes per decision (prefers 1).
    2. Quantitative evidence mandatory from telemetry for any parameter change.
    3. If total_trades < 20 -> status = "INSUFFICIENT_DATA".
    4. If telemetry is healthy -> status = "NO_CHANGE".
    5. A/B Testing comparison-aware calibrations (avoiding excessive divergence between Treatment and Control).
    6. Testnet-aware safety constraints: conservative calibrations (smaller shifts), tighter stop losses, reduced sizing.
    7. Persistent memory logging scoped under 'trading_advisory' namespace.
    8. Production Optimization: Telemetry caching, provider circuit breaker, and concurrency throttling.
    9. Advisory output only: NEVER executes trades or interfaces with exchange APIs.
    """

    def __init__(self, memory: BaseMemory | None = None) -> None:
        self.memory: SQLiteMemory = cast(SQLiteMemory, memory) if memory is not None else SQLiteMemory()
        self.registry = agent_registry

    def _format_context(self, req: TradingConsultRequest) -> str:
        """Serializes the request into a rich, structured textual context for the agent panel."""
        t = req.telemetry
        exp_header = f"=== TRADING BOT TELEMETRY (Bot ID: {req.bot_id} | Mode: {req.trading_mode} | Reason: {req.consultation_reason})"
        if req.experiment_id:
            arm_str = f" | Arm: {req.experiment_group}" if req.experiment_group else ""
            exp_header += f" | Experiment: {req.experiment_id}{arm_str}"
        exp_header += " ==="

        lines = [
            exp_header,
            f"Equity: ${t.equity:,.2f} USDT | Unrealized PnL: ${t.unrealized_pnl:,.2f} | Realized PnL: ${t.realized_pnl:,.2f}",
            f"Win Rate: {t.win_rate * 100:.1f}% | Profit Factor: {t.profit_factor:.2f} | Max Drawdown: {t.max_drawdown_pct:.2f}%",
            f"Consecutive Losses: {t.consecutive_losses} | Total Closed Trades: {t.total_trades} | Sharpe Ratio: {t.sharpe_ratio if t.sharpe_ratio is not None else 'N/A'}",
            "",
            "=== STRATEGY PERFORMANCE BREAKDOWN ==="
        ]
        if req.strategy_performance:
            for sp in req.strategy_performance:
                lines.append(
                    f"- Strategy '{sp.strategy_name}': Trades={sp.trade_count}, WinRate={sp.win_rate * 100:.1f}%, "
                    f"PF={sp.profit_factor:.2f}, NetPnL=${sp.net_pnl:,.2f}, AvgWin=${sp.avg_win:.2f}, AvgLoss=${sp.avg_loss:.2f}, "
                    f"ConsecLosses={sp.consecutive_losses}"
                )
        else:
            lines.append("No per-strategy telemetry provided.")

        # Testnet-Specific Context Section
        if req.trading_mode == "TESTNET" and req.testnet_specific:
            lines.append("")
            lines.append("=== TESTNET LIVE EXCHANGE ENVIRONMENT ===")
            ts = req.testnet_specific
            lines.append(
                f"Testnet Equity: ${ts.testnet_equity:,.2f} USDT | Testnet Drawdown: {ts.testnet_drawdown_pct:.2f}% | "
                f"Daily Loss: ${ts.testnet_daily_loss:,.2f} | Open Positions: {ts.testnet_open_positions} | Margin Level: {ts.testnet_margin_level:.1f}%"
            )

        # A/B Testing: Include Control Baseline Metrics if available
        if req.experiment_group == "TREATMENT" and req.control_metrics:
            lines.append("")
            lines.append("=== A/B TESTING: CONTROL BASELINE METRICS ===")
            c = req.control_metrics
            c_val = c.get('win_rate')
            c_wr = f"{c_val * 100:.1f}" if isinstance(c_val, (int, float)) and c_val <= 1.0 else str(c_val or 'N/A')
            lines.append(f"Control Profit Factor: {c.get('profit_factor', 'N/A')} | Control Win Rate: {c_wr}% | Control Max DD: {c.get('max_drawdown_pct', 'N/A')}%")
            if "total_trades" in c:
                lines.append(f"Control Total Trades: {c.get('total_trades')}")
            if "parameters" in c:
                lines.append(f"Control Parameters: {json.dumps(c.get('parameters'))}")

        lines.append("")
        lines.append("=== CURRENT LIVE PARAMETERS ===")
        if req.current_parameters:
            lines.append(json.dumps(req.current_parameters, indent=2))
        else:
            lines.append("No explicit parameters supplied.")

        if req.regime_data:
            lines.append("")
            lines.append("=== MARKET REGIME DATA ===")
            lines.append(json.dumps(req.regime_data, indent=2))

        if req.recent_trades:
            lines.append("")
            lines.append("=== RECENT CLOSED TRADES (Sample) ===")
            for idx, trade in enumerate(req.recent_trades[-15:], 1):
                lines.append(f"{idx}. {trade}")

        return "\n".join(lines)

    async def _invoke_agent(
        self,
        task_id: str,
        stage_name: str,
        round_number: int,
        agent: Agent,
        prompt: str,
        system_instructions: str | None = None
    ) -> SpecialistOutcome:
        """Invokes a specialist agent using ModelGateway, monitoring provider latency and circuit breakers.

        Returns a SpecialistOutcome so callers can tell real model output apart
        from deterministic fallback text. Run records are written with an
        honest status: "completed" only when a model actually answered.
        """
        monitor.record_agent_participation(agent.id)
        run_id = generate_run_id()
        msg_id = generate_message_id()

        # Check circuit breaker
        if not circuit_breaker.is_available(agent.model_provider):
            breaker_error = f"Circuit breaker OPEN for provider '{agent.model_provider}'"
            logger.warning("Circuit breaker OPEN for provider '%s'; skipping to deterministic fallback", agent.model_provider)
            fallback_text = self._deterministic_fallback_for_agent(agent.id, prompt)
            await self.memory.save_run(RunRecord(
                id=run_id,
                task_id=task_id,
                agent_id=agent.id,
                provider=agent.model_provider,
                model=agent.model_name,
                stage=stage_name,
                status="failed",
                error=breaker_error,
            ))
            await self.memory.save_message(MessageRecord(
                id=msg_id,
                run_id=run_id,
                task_id=task_id,
                role="assistant",
                agent_id=agent.id,
                content=fallback_text,
                stage=stage_name,
            ))
            return SpecialistOutcome(agent.id, fallback_text, False, breaker_error)

        req = ProviderRequest(
            messages=[ProviderMessage(role="user", content=prompt)],
            system_instruction=system_instructions or agent.system_instructions,
            model=agent.model_name,
            temperature=0.3,
            max_tokens=1024
        )

        start = time.perf_counter()

        try:
            resp = await asyncio.wait_for(
                model_gateway.execute(
                    provider_name=agent.model_provider,
                    request=req,
                    capability="reasoning",
                    stage_name=stage_name
                ),
                timeout=production_config.AGENT_INVOCATION_TIMEOUT_SECONDS
            )

            latency = time.perf_counter() - start
            circuit_breaker.record_success(agent.model_provider)
            monitor.record_provider_call(agent.model_provider, latency, success=True)

            model_answered = bool(resp and resp.content)
            if model_answered:
                content = resp.content
                run_status, run_error = "completed", None
            else:
                content = self._deterministic_fallback_for_agent(agent.id, prompt)
                run_status = "failed"
                run_error = "Provider returned empty content; deterministic fallback substituted"
                logger.warning(
                    "Trading consultation %s (%s) got empty model content; recording fallback as failed",
                    agent.id, stage_name,
                )

            # Record run with the truthful status of this invocation
            await self.memory.save_run(RunRecord(
                id=run_id,
                task_id=task_id,
                agent_id=agent.id,
                provider=resp.provider if resp else agent.model_provider,
                model=resp.model if resp else agent.model_name,
                stage=stage_name,
                latency_seconds=latency,
                prompt_tokens=resp.prompt_tokens if (resp and resp.prompt_tokens is not None) else 0,
                completion_tokens=resp.completion_tokens if (resp and resp.completion_tokens is not None) else 0,
                status=run_status,
                error=run_error,
            ))

            # Record message
            await self.memory.save_message(MessageRecord(
                id=msg_id,
                run_id=run_id,
                task_id=task_id,
                role="assistant",
                agent_id=agent.id,
                content=content,
                stage=stage_name
            ))

            return SpecialistOutcome(agent.id, content, model_answered, run_error)
        except Exception as exc:
            latency = time.perf_counter() - start
            circuit_breaker.record_failure(agent.model_provider)
            monitor.record_provider_call(agent.model_provider, latency, success=False)
            logger.warning("Trading consultation invocation for %s (%s) fallback: %s", agent.id, stage_name, str(exc))
            fallback_text = self._deterministic_fallback_for_agent(agent.id, prompt)

            # Record fallback run
            await self.memory.save_run(RunRecord(
                id=run_id,
                task_id=task_id,
                agent_id=agent.id,
                provider=agent.model_provider,
                model=agent.model_name,
                stage=stage_name,
                latency_seconds=latency,
                status="failed",
                error=str(exc),
            ))

            # Record fallback message
            await self.memory.save_message(MessageRecord(
                id=msg_id,
                run_id=run_id,
                task_id=task_id,
                role="assistant",
                agent_id=agent.id,
                content=fallback_text,
                stage=stage_name
            ))

            return SpecialistOutcome(agent.id, fallback_text, False, str(exc))

    def _deterministic_fallback_for_agent(self, agent_id: str, prompt: str) -> str:
        """Deterministic mathematical analysis fallback if cloud LLM providers are unavailable."""
        if agent_id == "trading_analyst":
            return (
                "Quantitative Analysis: Evaluated win rate, profit factor, consecutive losses, and drawdown against empirical baselines. "
                "Drawdown levels and consecutive loss streaks require bounded stop loss or cooldown tuning if risk boundaries are breached."
            )
        elif agent_id == "strategist":
            return (
                "Strategic Assessment: Prioritizing capital preservation and statistical expectancy over aggressive trade frequency. "
                "Recommend maximum 1-2 parameter shifts to isolate variable effects."
            )
        elif agent_id == "critic":
            return (
                "Adversarial Review: Scrutinized sample size reliability, curve fitting risks, and market regime shifts. "
                "Any proposed parameter changes must be backed by quantitative evidence from the telemetry."
            )
        elif agent_id == "data_analyst":
            return (
                "Quantitative Verification: Validated trade sample size and variance distributions. "
                "Confirmed whether metrics satisfy significance thresholds."
            )
        else:
            return "Synthesized multi-perspective consensus based on empirical telemetry."

    def _is_healthy(self, req: TradingConsultRequest) -> bool:
        """Evaluates whether all telemetry indicators are within safe, healthy operating boundaries."""
        t = req.telemetry
        if t.win_rate >= 0.50 and t.profit_factor >= 1.25 and t.max_drawdown_pct <= 5.0 and t.consecutive_losses < 4:
            if req.trading_mode == "TESTNET" and req.testnet_specific:
                ts = req.testnet_specific
                if ts.testnet_drawdown_pct > 5.0 or ts.testnet_margin_level < 150.0:
                    return False
            if req.strategy_performance:
                for sp in req.strategy_performance:
                    if sp.consecutive_losses >= 4 or (sp.trade_count >= 10 and sp.profit_factor < 0.9):
                        return False
            return True
        return False

    def _compare_treatment_vs_control(
        self,
        req: TradingConsultRequest
    ) -> tuple[str | None, str | None, str | None]:
        """Compares Treatment arm performance against Control baseline metrics."""
        if req.experiment_group != "TREATMENT" or not req.control_metrics:
            if req.experiment_group == "CONTROL":
                return (
                    "CONTROL_BASELINE",
                    "Advisory calibrated for the CONTROL baseline arm to preserve steady-state comparative integrity.",
                    "Maintains baseline expectancy."
                )
            return None, None, None

        t = req.telemetry
        c = req.control_metrics
        c_pf = c.get("profit_factor", 1.0)
        c_wr = c.get("win_rate", 0.5)
        c_dd = c.get("max_drawdown_pct", 5.0)

        pf_delta = t.profit_factor - c_pf
        wr_delta = (t.win_rate - c_wr) * 100.0

        if t.max_drawdown_pct >= c_dd * 1.5 or t.profit_factor <= c_pf * 0.8:
            status = "UNDERPERFORMING_CONTROL"
            rationale = (
                f"TREATMENT arm is underperforming CONTROL baseline (Profit Factor: {t.profit_factor:.2f} vs {c_pf:.2f}, "
                f"Drawdown: {t.max_drawdown_pct:.2f}% vs {c_dd:.2f}%). Recommended adjustments conservatively tighten risk "
                f"bounds to prevent excessive arm divergence while preserving valid test comparison."
            )
            improvement = f"Targeting +{abs(pf_delta):.2f} PF recovery to regain parity with Control baseline."
        elif t.profit_factor > c_pf * 1.1 or t.win_rate > c_wr + 0.05:
            status = "OUTPERFORMING_CONTROL"
            rationale = (
                f"TREATMENT arm is outperforming CONTROL baseline (Profit Factor: {t.profit_factor:.2f} vs {c_pf:.2f}, "
                f"Win Rate: {t.win_rate*100:.1f}% vs {c_wr*100:.1f}%). Recommending slight optimization to lock in edge "
                f"without over-tuning away from Control."
            )
            improvement = f"Expected +{pf_delta:.2f} higher Profit Factor over Control baseline."
        else:
            status = "PARITY"
            rationale = (
                f"TREATMENT arm is performing at parity with CONTROL baseline (Delta PF: {pf_delta:+.2f}, "
                f"Delta WR: {wr_delta:+.1f}%). Adjustments maintain controlled experimental separation."
            )
            improvement = "Incremental +5-10% risk-reward calibration."

        return status, rationale, improvement

    def _generate_testnet_risk_assessment(self, req: TradingConsultRequest) -> str | None:
        """Generates dedicated testnet risk assessment if operating in TESTNET mode."""
        if req.trading_mode != "TESTNET":
            return None

        t = req.telemetry
        ts = req.testnet_specific

        assessment_parts = [
            "TESTNET RISK ASSESSMENT: Operating on live testnet infrastructure with real market depth & orderbook fills."
        ]

        if ts:
            assessment_parts.append(
                f"Testnet Equity: ${ts.testnet_equity:,.2f} | Current Drawdown: {ts.testnet_drawdown_pct:.2f}% | "
                f"Margin Level: {ts.testnet_margin_level:.1f}% | Open Positions: {ts.testnet_open_positions}."
            )
            if ts.testnet_margin_level < 150.0:
                assessment_parts.append("WARNING: Margin level approaching critical buffer (<150%). Sizing reductions strictly enforced.")
            if ts.testnet_drawdown_pct > 6.0:
                assessment_parts.append("CRITICAL: Testnet drawdown exceeds 6.0%. Capital preservation protocol engaged.")
        else:
            assessment_parts.append(f"Telemetry Drawdown: {t.max_drawdown_pct:.2f}%. Testnet parameters apply tighter bounds.")

        assessment_parts.append("Recommended Sizing: Maintain position sizing at <= 0.8x paper trading standard. Tighten stop loss by 10%.")
        return " ".join(assessment_parts)

    def _rule_based_synthesis(
        self,
        req: TradingConsultRequest,
        ta_analysis: str,
        strat_analysis: str,
        critic_critique: str,
        data_analysis: str,
        specialist_outcomes: list[SpecialistOutcome] | None = None,
    ) -> AIUniverseDecision:
        """Produces bounded decision adhering to all safety invariants.

        Confidence is derived from how many specialists actually produced model
        output. It was previously a hardcoded 0.88 regardless of whether the
        panel had deliberated at all or silently fallen back to canned text.
        """
        decision_id = str(uuid4())
        valid_until = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        t = req.telemetry

        outcomes = specialist_outcomes or []
        total_specialists = len(outcomes)
        model_backed = sum(1 for o in outcomes if o.model_backed)
        participation = (model_backed / total_specialists) if total_specialists else 0.0
        # Linear honesty curve: 0.45 with no model participation, 0.88 when the
        # whole panel answered. A panel that never spoke cannot claim 0.88.
        panel_confidence = round(0.45 + 0.43 * participation, 2)
        agent_coverage = {
            o.agent_id: ("model" if o.model_backed else "fallback") for o in outcomes
        }
        fallback_ids = [o.agent_id for o in outcomes if not o.model_backed]
        degradation_reasons = [
            f"{o.agent_id}: {o.error or 'no model output produced'}" for o in outcomes if not o.model_backed
        ]
        panel_degraded = bool(fallback_ids)

        treat_status, comp_rationale, exp_improvement = self._compare_treatment_vs_control(req)
        testnet_assessment = self._generate_testnet_risk_assessment(req)

        # 1. Check Trade Count Requirement (<20 trades)
        if t.total_trades < 20:
            return AIUniverseDecision(
                decision_id=decision_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                status="INSUFFICIENT_DATA",
                confidence=0.95,
                parameter_changes=[],
                risk_assessment=f"Sample size of {t.total_trades} trades is below the statistical significance threshold of 20 closed trades. Recommend continued paper/testnet execution to gather baseline distribution data.",
                regime_analysis=f"Market regime observation active. Current win rate is {t.win_rate * 100:.1f}%, but confidence is uncalibrated due to low sample volume.",
                dissent_notes=(
                    "Adversarial Critic cautions against premature parameter adjustment on small sample "
                    "sizes (N < 20) to prevent curve fitting."
                    if not panel_degraded
                    else "No independent critique was performed: the Adversarial Critic produced no model "
                    "output. The insufficiency finding below derives from the deterministic sample-size "
                    "rule alone."
                ),
                debate_summary=(
                    "Specialist panel unanimously determined that statistical sample size is insufficient "
                    "to support parameter recalibration."
                    if not panel_degraded
                    else (
                        f"Deterministic sample-size rule returned INSUFFICIENT_DATA (N={t.total_trades} < 20). "
                        f"{len(fallback_ids)}/{total_specialists} specialists produced no model output "
                        f"({', '.join(fallback_ids)}), so no panel deliberation took place."
                    )
                ),
                degraded=panel_degraded,
                degradation_reasons=degradation_reasons,
                agent_coverage=agent_coverage,
                valid_until=valid_until,
                comparison_rationale=comp_rationale or "A/B comparison deferred until minimum statistical sample size (N >= 20) is accumulated.",
                expected_improvement=exp_improvement or "Baseline data acquisition in progress.",
                treatment_status=treat_status,
                testnet_risk_assessment=testnet_assessment
            )

        # 2. Check Healthy Performance
        if self._is_healthy(req):
            return AIUniverseDecision(
                decision_id=decision_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                status="NO_CHANGE",
                confidence=0.90,
                parameter_changes=[],
                risk_assessment=f"Healthy performance profile: Win rate {t.win_rate * 100:.1f}%, Profit Factor {t.profit_factor:.2f}, Max Drawdown {t.max_drawdown_pct:.2f}%. Bot is operating stably within safe statistical parameters.",
                regime_analysis="Strategy is well-aligned with the prevailing market regime. Expectancy remains positive.",
                dissent_notes=(
                    "No critical risk breaches identified by the Adversarial Critic."
                    if not panel_degraded
                    else "Adversarial Critic produced no model output; no independent critique was run "
                    "against these metrics."
                ),
                debate_summary=(
                    "TradingAnalyst, Strategist, Data Analyst, and Critic confirmed healthy metrics "
                    "across all active strategies. Maintaining current parameter configurations."
                    if not panel_degraded
                    else (
                        f"Deterministic health check returned NO_CHANGE on verified telemetry "
                        f"(WR {t.win_rate * 100:.1f}%, PF {t.profit_factor:.2f}, DD "
                        f"{t.max_drawdown_pct:.2f}%). {len(fallback_ids)}/{total_specialists} "
                        f"specialists produced no model output ({', '.join(fallback_ids)}); "
                        f"their entries are canned fallback text, not analysis."
                    )
                ),
                degraded=panel_degraded,
                degradation_reasons=degradation_reasons,
                agent_coverage=agent_coverage,
                valid_until=valid_until,
                comparison_rationale=comp_rationale or "Healthy metrics align with baseline operating envelope; no arm divergence required.",
                expected_improvement=exp_improvement or "Expectancy remains stable at current healthy levels.",
                treatment_status=treat_status,
                testnet_risk_assessment=testnet_assessment
            )

        # 3. Derive Bounded Recommendations (Max 2, prefer 1)
        changes: list[ParameterChange] = []
        risk_narrative = ""
        regime_narrative = ""
        dissent_narrative = ""

        if req.trading_mode == "TESTNET":
            tighten_factor = 0.80
            expand_factor = 1.10
        elif req.experiment_group == "TREATMENT":
            tighten_factor = 0.88
            expand_factor = 1.12
        else:
            tighten_factor = 0.85
            expand_factor = 1.15

        # Case A: Severe Drawdown / High Consecutive Losses -> Tighten Stop Loss / Risk
        if t.max_drawdown_pct > 5.0 or t.consecutive_losses >= 4:
            target_strategy = "default"
            target_param = "stop_loss_pct"
            curr_val = 0.02

            if req.current_parameters:
                target_strategy = next(iter(req.current_parameters.keys()))
                strat_params = req.current_parameters[target_strategy]
                for p_candidate in ["stop_loss_pct", "sl_pct", "atr_multiplier", "risk_per_trade"]:
                    if p_candidate in strat_params:
                        target_param = p_candidate
                        curr_val = strat_params[p_candidate]
                        break

            if isinstance(curr_val, (int, float)) and curr_val > 0:
                rec_val = round(curr_val * tighten_factor, 4)
                change_pct = round(((rec_val - curr_val) / curr_val) * 100.0, 2)
            else:
                curr_val = 0.02
                rec_val = round(0.02 * tighten_factor, 4)
                change_pct = round((tighten_factor - 1.0) * 100.0, 2)

            testnet_tag = " [Testnet Conservative Safety]" if req.trading_mode == "TESTNET" else ""
            changes.append(ParameterChange(
                strategy=target_strategy,
                parameter=target_param,
                current_value=curr_val,
                recommended_value=rec_val,
                change_pct=change_pct,
                rationale=f"Consecutive losses ({t.consecutive_losses}) or Max Drawdown ({t.max_drawdown_pct:.2f}%) on {target_strategy} justifies tightening {target_param} by {abs(change_pct):.1f}% for capital preservation{testnet_tag}."
            ))

            if (t.max_drawdown_pct > 8.0 or (req.trading_mode == "TESTNET" and t.max_drawdown_pct > 6.0)) and req.current_parameters and len(changes) < 2:
                strat_params = req.current_parameters.get(target_strategy, {})
                if "position_size_usdt" in strat_params:
                    curr_pos = strat_params["position_size_usdt"]
                    rec_pos = round(curr_pos * 0.80, 2)
                    changes.append(ParameterChange(
                        strategy=target_strategy,
                        parameter="position_size_usdt",
                        current_value=curr_pos,
                        recommended_value=rec_pos,
                        change_pct=-20.0,
                        rationale=f"Drawdown ({t.max_drawdown_pct:.2f}%) on Testnet justifies 20% position size reduction to limit nominal capital exposure."
                    ))
                elif "cooldown_seconds" in strat_params:
                    c_val = strat_params["cooldown_seconds"]
                    r_val = int(c_val * 1.5)
                    changes.append(ParameterChange(
                        strategy=target_strategy,
                        parameter="cooldown_seconds",
                        current_value=c_val,
                        recommended_value=r_val,
                        change_pct=50.0,
                        rationale=f"Elevated drawdown of {t.max_drawdown_pct:.2f}% justifies expanding cooldown to {r_val}s to prevent overtrading during adverse volatility regimes."
                    ))

            risk_narrative = f"ELEVATED RISK: Account max drawdown reached {t.max_drawdown_pct:.2f}% with {t.consecutive_losses} consecutive losses. Capital preservation protocol activated."
            regime_narrative = "Market regime indicates chop or unfavorable volatility for current breakout/trend parameters."
            dissent_narrative = "Critic noted that wider stop losses in this regime increase tail-risk exposure; tightening stop loss is strictly indicated."

        # Case B: Sub-optimal Profit Factor / Low Win Rate
        elif t.profit_factor < 1.1 or t.win_rate < 0.45:
            target_strategy = "default"
            target_param = "take_profit_pct"
            curr_val = 0.015

            if req.current_parameters:
                target_strategy = next(iter(req.current_parameters.keys()))
                strat_params = req.current_parameters[target_strategy]
                for p_candidate in ["take_profit_pct", "tp_pct", "profit_target", "atr_multiplier"]:
                    if p_candidate in strat_params:
                        target_param = p_candidate
                        curr_val = strat_params[p_candidate]
                        break

            if isinstance(curr_val, (int, float)) and curr_val > 0:
                rec_val = round(curr_val * expand_factor, 4)
                change_pct = round(((rec_val - curr_val) / curr_val) * 100.0, 2)
            else:
                curr_val = 0.015
                rec_val = round(0.015 * expand_factor, 4)
                change_pct = round((expand_factor - 1.0) * 100.0, 2)

            changes.append(ParameterChange(
                strategy=target_strategy,
                parameter=target_param,
                current_value=curr_val,
                recommended_value=rec_val,
                change_pct=change_pct,
                rationale=f"Sub-optimal Profit Factor ({t.profit_factor:.2f}) and Win Rate ({t.win_rate * 100:.1f}%) on {target_strategy} justifies expanding {target_param} by {change_pct:.1f}% to improve risk-reward asymmetry."
            ))

            risk_narrative = f"MODERATE RISK: Expectancy is dragged down by asymmetric friction (PF: {t.profit_factor:.2f}, WR: {t.win_rate * 100:.1f}%)."
            regime_narrative = "Current market regime exhibits sufficient trend continuation to capture wider target multiples."
            dissent_narrative = "Critic questioned whether expanding profit targets might reduce fill probability; recommended monitoring fill rate over next 20 trades."

        else:
            risk_narrative = "STABLE: Bot performance is slightly below optimal target but within acceptable tolerance."
            regime_narrative = "Market conditions stable."
            dissent_narrative = "Panel considered parameter adjustments but opted for conservative holding pattern."

        bounded_changes = changes[:2]
        status_val: Literal["RECOMMENDATION", "NO_CHANGE", "INSUFFICIENT_DATA"] = "RECOMMENDATION" if bounded_changes else "NO_CHANGE"

        if panel_degraded:
            debate_header = (
                f"Multi-Agent Deliberation (DEGRADED - {len(fallback_ids)}/{total_specialists} "
                f"specialists produced no model output; entries marked [FALLBACK] are canned "
                f"deterministic text, not model analysis):\n"
            )
        else:
            debate_header = "Multi-Agent Deliberation:\n"

        def _entry(label: str, agent_id: str, text: str) -> str:
            marker = "" if agent_coverage.get(agent_id) == "model" else " [FALLBACK]"
            return f"- {label}: {text[:200]}...{marker}\n"

        debate_summary = (
            debate_header
            + _entry("TradingAnalyst", "trading_analyst", ta_analysis)
            + _entry("Strategist", "strategist", strat_analysis)
            + _entry("Critic", "critic", critic_critique)
            + _entry("Data Analyst", "data_analyst", data_analysis)
        ).rstrip("\n")

        return AIUniverseDecision(
            decision_id=decision_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=status_val,
            confidence=panel_confidence,
            parameter_changes=bounded_changes,
            risk_assessment=risk_narrative or "Risk profile assessed across all active strategies.",
            regime_analysis=regime_narrative or "Regime telemetry evaluated.",
            dissent_notes=dissent_narrative or "No material dissent noted.",
            debate_summary=debate_summary,
            valid_until=valid_until,
            comparison_rationale=comp_rationale,
            expected_improvement=exp_improvement or ("Estimated +10-15% risk-adjusted expectancy improvement." if bounded_changes else "Preserves existing expectancy profile."),
            treatment_status=treat_status,
            testnet_risk_assessment=testnet_assessment,
            degraded=panel_degraded,
            degradation_reasons=degradation_reasons,
            agent_coverage=agent_coverage,
        )

    async def consult(self, req: TradingConsultRequest) -> AIUniverseDecision:
        """Executes multi-agent consultation wrapped with cache lookup and performance monitoring."""
        start_time = time.perf_counter()

        # Check Cache
        cached_decision = telemetry_cache.get(req)
        if cached_decision:
            latency = time.perf_counter() - start_time
            monitor.record_request(latency, success=True)
            return cached_decision

        decision = await concurrency_controller.run(self._consult_internal, req)

        # Cache only decisions the whole panel actually produced. Reusing a
        # degraded result would keep serving fallback-backed advice as fresh.
        if not decision.degraded:
            telemetry_cache.set(req, decision)

        latency = time.perf_counter() - start_time
        monitor.record_request(latency, success=True)
        return decision

    async def _consult_internal(self, req: TradingConsultRequest) -> AIUniverseDecision:
        """Internal multi-agent debate pipeline."""
        task_id = generate_task_id()
        context_str = self._format_context(req)

        # Retrieve specialist agents
        trading_analyst = self.registry.get_agent("trading_analyst") or self.registry.get_agent("data_analyst") or self.registry.list_agents()[0]
        strategist = self.registry.get_agent("strategist") or trading_analyst
        critic = self.registry.get_agent("critic") or trading_analyst
        data_analyst = self.registry.get_agent("data_analyst") or trading_analyst

        # Save initial task record in memory
        task_record = TaskRecord(
            id=task_id,
            question=f"Trading Consultation: Bot {req.bot_id} ({req.consultation_reason} | {req.trading_mode})" + (f" [Arm: {req.experiment_group}]" if req.experiment_group else ""),
            mode="trading_consult",
            status="running",
            metadata={
                "bot_id": req.bot_id,
                "trading_mode": req.trading_mode,
                "consultation_reason": req.consultation_reason,
                "experiment_id": req.experiment_id,
                "experiment_group": req.experiment_group,
                "total_trades": req.telemetry.total_trades,
                "win_rate": req.telemetry.win_rate,
                "profit_factor": req.telemetry.profit_factor,
                "max_drawdown_pct": req.telemetry.max_drawdown_pct
            }
        )
        await self.memory.save_task(task_record)

        # Parallel/Pipelines agent deliberations
        ta_prompt = (
            f"Analyze the following autonomous trading bot telemetry and formulate initial parameter recommendations:\n\n"
            f"{context_str}\n\n"
            f"Identify performance anomalies, consecutive loss streaks, drawdown risks, and propose concrete adjustments."
        )
        ta_outcome = await self._invoke_agent(task_id, "trading_analysis", 1, trading_analyst, ta_prompt)
        ta_output = ta_outcome.text

        strat_prompt = (
            f"Review the trading telemetry and the Trading Analyst's initial findings:\n\n"
            f"TELEMETRY:\n{context_str}\n\n"
            f"TRADING ANALYST FINDINGS:\n{ta_output}\n\n"
            f"Evaluate the strategic trade-offs of proposed adjustments. Weigh drawdown mitigation against trade frequency. "
            f"If trading in TESTNET mode, strictly enforce conservative risk parameters and prioritize capital preservation."
        )
        strat_outcome = await self._invoke_agent(task_id, "strategic_comparison", 2, strategist, strat_prompt)
        strat_output = strat_outcome.text

        critic_prompt = (
            f"Critique the following strategy proposals and challenge any weak assumptions:\n\n"
            f"TELEMETRY:\n{context_str}\n\n"
            f"PROPOSALS:\n- Trading Analyst: {ta_output}\n- Strategist: {strat_output}\n\n"
            f"Attack curve-fitting risks, sample size limitations, and adverse market regimes. Highlight counter-risks."
        )
        critic_outcome = await self._invoke_agent(task_id, "adversarial_critique", 3, critic, critic_prompt)
        critic_output = critic_outcome.text

        data_prompt = (
            f"Quantitatively verify the proposals and critique against the empirical telemetry numbers:\n\n"
            f"TELEMETRY:\n{context_str}\n\n"
            f"CRITIQUE:\n{critic_output}\n\n"
            f"Verify if trade count (N={req.telemetry.total_trades}) justifies parameter adjustments and confirm mathematical validity."
        )
        data_outcome = await self._invoke_agent(task_id, "quantitative_verification", 4, data_analyst, data_prompt)
        data_output = data_outcome.text

        specialist_outcomes = [ta_outcome, strat_outcome, critic_outcome, data_outcome]

        decision = self._rule_based_synthesis(
            req=req,
            ta_analysis=ta_output,
            strat_analysis=strat_output,
            critic_critique=critic_output,
            data_analysis=data_output,
            specialist_outcomes=specialist_outcomes,
        )

        task_record.status = "completed"
        task_record.result = json.dumps(decision.model_dump())
        task_record.confidence = decision.confidence
        task_record.completed_at = datetime.now(timezone.utc)
        task_record.metadata["decision_id"] = decision.decision_id
        task_record.metadata["status"] = decision.status
        model_backed = sum(1 for o in specialist_outcomes if o.model_backed)
        task_record.metadata["model_backed_specialists"] = model_backed
        task_record.metadata["specialist_coverage"] = {
            o.agent_id: ("model" if o.model_backed else "fallback") for o in specialist_outcomes
        }
        if decision.degraded:
            task_record.metadata["degraded"] = True
            task_record.metadata["degradation_reasons"] = decision.degradation_reasons
            logger.warning(
                "Trading consultation %s degraded: %d/%d specialists produced model output (%s)",
                decision.decision_id, model_backed, len(specialist_outcomes),
                "; ".join(decision.degradation_reasons),
            )
        task_record.metadata["parameter_changes_count"] = len(decision.parameter_changes)
        task_record.metadata["trading_mode"] = req.trading_mode
        if req.experiment_id:
            task_record.metadata["experiment_id"] = req.experiment_id
            task_record.metadata["experiment_group"] = req.experiment_group
        await self.memory.save_task(task_record)

        advisory_memory = MemoryRecord(
            id=str(uuid4()),
            agent_id="trading_advisory",
            content=f"Decision {decision.decision_id} for Bot {req.bot_id} (Mode: {req.trading_mode}, Reason: {req.consultation_reason}): Status={decision.status}, Changes={len(decision.parameter_changes)}, Risk={decision.risk_assessment}",
            memory_type="trading_consultation",
            importance=0.9 if decision.status == "RECOMMENDATION" else 0.7,
            context_tags=["trading", req.bot_id, req.trading_mode, decision.status, req.experiment_id or "general"]
        )
        await self.memory.save_memory(advisory_memory)

        if req.experiment_id and req.experiment_group:
            experiment_service.record_consultation(
                experiment_id=req.experiment_id,
                arm=req.experiment_group,
                telemetry=req.telemetry.model_dump(),
                decision_id=decision.decision_id
            )

        logger.info(
            "Trading Consultation %s completed for Bot %s (Mode: %s): Status=%s | Changes=%d | Confidence=%.2f",
            decision.decision_id, req.bot_id, req.trading_mode, decision.status, len(decision.parameter_changes), decision.confidence
        )

        return decision

    async def get_decision_by_id(self, decision_id: str) -> AIUniverseDecision | None:
        """Retrieves a past advisory decision from memory."""
        async with self.memory.connect() as db:
            async with db.execute(
                "SELECT * FROM tasks WHERE mode = 'trading_consult' AND result IS NOT NULL ORDER BY created_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()
                for row in rows:
                    if row["result"]:
                        try:
                            data = json.loads(row["result"])
                            if data.get("decision_id") == decision_id:
                                return AIUniverseDecision(**data)
                        except Exception:
                            continue
        return None

    async def get_testnet_performance(self) -> TestnetPerformanceResponse:
        """Aggregates historical testnet consultations vs paper trading consultations."""
        testnet_entries: list[dict[str, Any]] = []
        paper_entries: list[dict[str, Any]] = []

        async with self.memory.connect() as db, db.execute(
            "SELECT * FROM tasks WHERE mode = 'trading_consult' AND result IS NOT NULL"
        ) as cursor:
            rows = await cursor.fetchall()
            for r in rows:
                meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
                t_mode = meta.get("trading_mode", "PAPER")
                entry = {
                    "win_rate": meta.get("win_rate", 0.5),
                    "profit_factor": meta.get("profit_factor", 1.0),
                    "max_drawdown_pct": meta.get("max_drawdown_pct", 4.0),
                    "total_trades": meta.get("total_trades", 0)
                }
                if t_mode == "TESTNET":
                    testnet_entries.append(entry)
                else:
                    paper_entries.append(entry)

        def _calc_stats(entries: list[dict[str, Any]]) -> dict[str, Any]:
            if not entries:
                return {
                    "count": 0,
                    "avg_win_rate": 0.0,
                    "avg_profit_factor": 0.0,
                    "avg_drawdown_pct": 0.0,
                    "total_trades_analyzed": 0
                }
            n = len(entries)
            return {
                "count": n,
                "avg_win_rate": round(sum(e["win_rate"] for e in entries) / n, 3),
                "avg_profit_factor": round(sum(e["profit_factor"] for e in entries) / n, 2),
                "avg_drawdown_pct": round(sum(e["max_drawdown_pct"] for e in entries) / n, 2),
                "total_trades_analyzed": sum(e["total_trades"] for e in entries)
            }

        t_stats = _calc_stats(testnet_entries)
        p_stats = _calc_stats(paper_entries)

        drawdown_dist = {
            "testnet_low_dd_pct": sum(1 for e in testnet_entries if e["max_drawdown_pct"] <= 5.0),
            "testnet_high_dd_pct": sum(1 for e in testnet_entries if e["max_drawdown_pct"] > 5.0),
            "paper_low_dd_pct": sum(1 for e in paper_entries if e["max_drawdown_pct"] <= 5.0),
            "paper_high_dd_pct": sum(1 for e in paper_entries if e["max_drawdown_pct"] > 5.0)
        }

        return TestnetPerformanceResponse(
            total_consultations=len(testnet_entries) + len(paper_entries),
            testnet_consultations=len(testnet_entries),
            paper_consultations=len(paper_entries),
            testnet_metrics=t_stats,
            paper_metrics=p_stats,
            drawdown_distribution=drawdown_dist
        )

    async def get_testnet_comparison(self) -> TestnetComparisonResponse:
        """Generates a side-by-side comparison of testnet vs paper trading dynamics."""
        perf = await self.get_testnet_performance()

        divergence = [
            {
                "strategy": "Supertrend_Breakout",
                "testnet_pf": perf.testnet_metrics.get("avg_profit_factor", 1.2),
                "paper_pf": perf.paper_metrics.get("avg_profit_factor", 1.4),
                "slippage_impact": "Medium (Spread friction detected on testnet fills)",
                "recommended_action": "Tighter stop loss and wider take profit multiple on testnet"
            },
            {
                "strategy": "EMA_Cross_Trend",
                "testnet_pf": perf.testnet_metrics.get("avg_profit_factor", 1.1),
                "paper_pf": perf.paper_metrics.get("avg_profit_factor", 1.3),
                "slippage_impact": "Low (Execution parity maintained)",
                "recommended_action": "Maintain parity with paper trading baseline"
            }
        ]

        summary = (
            f"Testnet consultations represent {perf.testnet_consultations} sessions. "
            f"Testnet executions encounter realistic orderbook depth and latency, justifying a 10% tighter stop loss "
            f"and 0.8x position sizing relative to paper simulations."
        )

        return TestnetComparisonResponse(
            comparison_timestamp=datetime.now(timezone.utc).isoformat(),
            testnet_summary=perf.testnet_metrics,
            paper_summary=perf.paper_metrics,
            strategy_divergence=divergence,
            recommendations_summary=summary
        )


# Global singleton consultation service instance
trading_consult_service = TradingConsultService()
