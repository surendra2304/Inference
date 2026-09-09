"""Task router module for classifying problem complexity, budget constraints, and selecting specialist agents."""

import re
from typing import Any

from pydantic import BaseModel, Field

from app.core.policies import SystemPolicies
from app.utils.logger import logger

DEBATE_TRIGGERS: list[str] = [
    "compare", "tradeoff", "trade-off", "vs", "versus", "architecture",
    "design", "debate", "should i", "which is better", "evaluate", "critique",
    "pros and cons", "alternatives"
]

DOMAIN_KEYWORD_MAP = {
    "debugger": ["debug", "debugger", "debugging", "error", "errors", "traceback", "tracebacks", "crash", "crashes", "deadlock", "deadlocks", "exception", "exceptions", "bug", "bugs", "failure", "failures", "fail", "broken", "stacktrace"],
    "security_analyst": ["security", "vulnerability", "vulnerabilities", "auth", "authentication", "authorization", "authorize", "threat", "threats", "injection", "injections", "permission", "permissions", "leak", "leaks", "secret", "secrets", "attack", "attacks", "audit"],
    "data_analyst": ["data", "sql", "metric", "metrics", "statistic", "statistics", "table", "tables", "dataset", "datasets", "dataframe", "dataframes", "chart", "charts", "distribution", "distributions"],
    "coder": ["code", "coding", "implement", "implementation", "function", "functions", "refactor", "refactoring", "algorithm", "algorithms", "class", "classes", "syntax", "python", "fastapi", "write a", "script"],
    "architect": ["architecture", "architectures", "system design", "modular", "modularity", "scalability", "scalable", "pipeline", "pipelines", "schema design", "microservice", "microservices", "schema"],
    "fact_checker": ["verify", "verification", "is it true", "fact", "facts", "check claim", "verifiable", "source", "sources", "accuracy", "accurate"],
    "strategist": ["strategy", "strategies", "strategic", "priority", "priorities", "roadmap", "decision", "decisions", "trade-off", "tradeoffs", "trade-offs", "cost-benefit", "plan", "planning"],
    "critic": ["critique", "critiques", "red team", "weakness", "weaknesses", "fallacy", "fallacies", "counterexample", "counterexamples", "challenge"],
    "trading_analyst": ["trading", "trade", "trades", "trader", "pnl", "drawdown", "drawdowns", "win rate", "profit factor", "stop loss", "take profit", "bot", "bots", "futures", "scalper", "scalping", "sharpe", "sortino", "orderbook"],
}


class RoutingDecision(BaseModel):
    """Result of task complexity, budget evaluation, and agent allocation analysis."""
    mode: str = Field(description="fast, review, or debate")
    reason: str = Field(description="Explanation of routing rationale")
    selected_agent_ids: list[str] = Field(description="Ordered list of agent IDs assigned to the task")
    degraded: bool = Field(default=False, description="Whether degradation was applied due to budget/latency")
    telemetry: dict[str, Any] = Field(default_factory=dict, description="Telemetry metrics for future learning")


class TaskRouter:
    """Classifies task domain, evaluates budget/latency constraints, and selects specialist agent panels."""

    def detect_domain_specialist(self, question: str) -> str:
        """Detect the single most relevant specialist for a given query."""
        q_lower = question.lower()
        for agent_id, keywords in DOMAIN_KEYWORD_MAP.items():
            if any(re.search(rf"\b{re.escape(kw)}\b", q_lower) for kw in keywords):
                return agent_id
        return "researcher"

    def select_review_pair(self, question: str) -> list[str]:
        """Select two complementary agents for Review mode."""
        primary = self.detect_domain_specialist(question)
        if primary == "coder":
            return ["coder", "critic"]
        elif primary == "architect":
            return ["architect", "security_analyst"]
        elif primary == "security_analyst":
            return ["security_analyst", "architect"]
        elif primary == "debugger":
            return ["debugger", "coder"]
        elif primary == "data_analyst":
            return ["data_analyst", "fact_checker"]
        elif primary == "trading_analyst":
            return ["trading_analyst", "strategist"]
        else:
            return [primary, "critic"]

    def select_debate_panel(self, question: str, max_agents: int = 5) -> list[str]:
        """Assemble a diverse panel of 3-5 specialists for structured debate."""
        primary = self.detect_domain_specialist(question)
        panel: list[str] = []

        # Always include the primary domain specialist
        panel.append(primary)

        # Candidate specialists for debate
        candidates = ["trading_analyst", "strategist", "critic", "data_analyst", "architect", "security_analyst", "coder", "fact_checker"]
        for cand in candidates:
            if cand not in panel and len(panel) < max_agents:
                panel.append(cand)

        # Ensure Critic is included if panel size > 1
        if "critic" not in panel and len(panel) > 1:
            panel[-1] = "critic"

        return panel[:max_agents]

    def classify_mode(self, question: str, requested_mode: str = "auto") -> tuple[str, str]:
        """
        Determines the candidate execution mode (fast, review, debate).
        Returns a tuple of (selected_mode, reason).
        """
        normalized_mode = requested_mode.lower().strip()
        if normalized_mode in ("fast", "review", "debate"):
            return normalized_mode, f"Explicitly requested by client: {normalized_mode}"

        q_lower = question.lower()
        words = question.split()

        # Check for complex debate keywords
        matched_triggers = [t for t in DEBATE_TRIGGERS if re.search(rf"\b{re.escape(t)}\b", q_lower)]
        if matched_triggers:
            return "debate", f"Matched complex reasoning triggers: {', '.join(matched_triggers)}"

        # If question is concise (< 30 words) -> Fast mode
        if len(words) < 30:
            return "fast", "Concise factual or exploratory inquiry"

        return "review", "Moderate complexity requiring validation"

    def apply_budget_and_latency_guardrails(
        self,
        candidate_mode: str,
        reason: str,
        question: str,
        max_budget: float | None = None,
        max_latency: float | None = None
    ) -> tuple[str, str, bool]:
        """
        Applies budget, latency, and triviality guardrails to prevent excessive API costs.
        Gracefully degrades Debate -> Review -> Fast when limits are tight.
        """
        degraded = False
        final_mode = candidate_mode
        updated_reason = reason

        words = question.strip().split()
        is_trivial = len(words) <= 4 and not any(
            re.search(rf"\b{re.escape(t)}\b", question.lower()) for t in DEBATE_TRIGGERS
        )

        # Triviality guardrail: Avoid 5-agent debate on trivial/short greetings or simple words
        if final_mode == "debate" and is_trivial:
            final_mode = "fast"
            degraded = True
            updated_reason = f"Trivial query guardrail: {len(words)} words inquiry degraded from debate to fast mode"
            logger.info(updated_reason)
            return final_mode, updated_reason, degraded

        # Latency & budget threshold checks
        budget_limit = max_budget if max_budget is not None else float("inf")
        latency_limit = max_latency if max_latency is not None else float("inf")

        if final_mode == "debate":
            if (
                latency_limit <= SystemPolicies.FAST_MODE_LATENCY_THRESHOLD_SECONDS
                or budget_limit <= SystemPolicies.FAST_MODE_BUDGET_THRESHOLD_USD
            ):
                final_mode = "fast"
                degraded = True
                updated_reason = (
                    f"Tight constraints (budget=${budget_limit:.4f}, max_latency={latency_limit:.1f}s) "
                    "forced degradation from debate to fast mode."
                )
            elif (
                latency_limit <= SystemPolicies.REVIEW_MODE_LATENCY_THRESHOLD_SECONDS
                or budget_limit <= SystemPolicies.REVIEW_MODE_BUDGET_THRESHOLD_USD
            ):
                final_mode = "review"
                degraded = True
                updated_reason = (
                    f"Moderate constraints (budget=${budget_limit:.4f}, max_latency={latency_limit:.1f}s) "
                    "forced degradation from debate to review mode."
                )

        elif final_mode == "review":
            if (
                latency_limit <= SystemPolicies.FAST_MODE_LATENCY_THRESHOLD_SECONDS
                or budget_limit <= SystemPolicies.FAST_MODE_BUDGET_THRESHOLD_USD
            ):
                final_mode = "fast"
                degraded = True
                updated_reason = (
                    f"Tight constraints (budget=${budget_limit:.4f}, max_latency={latency_limit:.1f}s) "
                    "forced degradation from review to fast mode."
                )

        if degraded:
            logger.info("Guardrails applied: %s", updated_reason)

        return final_mode, updated_reason, degraded

    def route_task(
        self,
        question: str,
        requested_mode: str = "auto",
        max_agents: int = 5,
        max_budget: float | None = None,
        max_latency: float | None = None
    ) -> RoutingDecision:
        """
        Full routing decision returning mode, reason, selected agents, and telemetry.
        Evaluates problem domain and enforces budget/latency guardrails.
        """
        candidate_mode, initial_reason = self.classify_mode(question, requested_mode)

        final_mode, final_reason, degraded = self.apply_budget_and_latency_guardrails(
            candidate_mode=candidate_mode,
            reason=initial_reason,
            question=question,
            max_budget=max_budget,
            max_latency=max_latency
        )

        if final_mode == "fast":
            agent_id = self.detect_domain_specialist(question)
            selected = [agent_id]
        elif final_mode == "review":
            selected = self.select_review_pair(question)
        else:  # debate
            selected = self.select_debate_panel(question, max_agents=max_agents)

        # Estimate metrics for telemetry
        estimated_latency = 1.0 if final_mode == "fast" else (4.0 if final_mode == "review" else 12.0)
        estimated_cost = 0.0005 if final_mode == "fast" else (0.002 if final_mode == "review" else 0.015)

        telemetry = {
            "requested_mode": requested_mode,
            "candidate_mode": candidate_mode,
            "final_mode": final_mode,
            "degraded": degraded,
            "max_budget": max_budget,
            "max_latency": max_latency,
            "agent_count": len(selected),
            "estimated_latency_s": estimated_latency,
            "estimated_cost_usd": estimated_cost,
            "reason": final_reason
        }

        logger.info(
            "TELEMETRY: Task routed to mode '%s' (degraded=%s) with agents %s. Reason: %s",
            final_mode, degraded, selected, final_reason
        )

        return RoutingDecision(
            mode=final_mode,
            reason=final_reason,
            selected_agent_ids=selected,
            degraded=degraded,
            telemetry=telemetry
        )


router = TaskRouter()
