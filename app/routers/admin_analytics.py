"""FastAPI Router for Usage Analytics, Provider Intelligence, Self-Optimization, and Admin Dashboards."""

from fastapi import APIRouter, Path, Query, status

from app.alerts import alert_system
from app.analytics.outcome_learning import (
    DetailedOutcomeReport,
    outcome_learning_engine,
)
from app.analytics.outcomes import OutcomeReportRequest, consumer_outcome_tracker
from app.analytics.predictive import predictive_provider_manager
from app.analytics.provider_intel import provider_intel
from app.analytics.usage_analytics import usage_analytics
from app.routing.self_optimizer import self_optimizing_router
from app.services.quality_assurance import quality_assurance_service

analytics_router = APIRouter(prefix="/v1", tags=["Analytics, Optimization & Admin"])


# Analytics endpoints
@analytics_router.get("/analytics/overview", status_code=status.HTTP_200_OK)
async def get_analytics_overview():
    """Returns total calls, tokens, latency, cost, and budget ceiling status."""
    return usage_analytics.get_overview()


@analytics_router.get("/analytics/consumer/{id}", status_code=status.HTTP_200_OK)
async def get_consumer_analytics(id: str = Path(..., description="Consumer identifier, e.g. forge, trading_bot")):
    """Returns analytics breakdown for a specific consumer."""
    return usage_analytics.get_consumer_breakdown(id)


@analytics_router.get("/analytics/service/{name}", status_code=status.HTTP_200_OK)
async def get_service_analytics(name: str = Path(..., description="Service name, e.g. generate-code, trading_consult")):
    """Returns performance and token usage for a specific service."""
    return usage_analytics.get_service_breakdown(name)


@analytics_router.get("/analytics/providers", status_code=status.HTTP_200_OK)
async def get_providers_comparison():
    """Returns token and latency comparison across all cloud providers."""
    return usage_analytics.get_providers_comparison()


@analytics_router.get("/analytics/provider-intel", status_code=status.HTTP_200_OK)
async def get_provider_intelligence():
    """Returns deep performance matrices, failure patterns, and routing recommendations."""
    return provider_intel.get_performance_matrix()


@analytics_router.get("/analytics/quality", status_code=status.HTTP_200_OK)
async def get_quality_report():
    """Returns response quality trends, confidence calibration curves, and agent rankings."""
    return quality_assurance_service.get_quality_report()


@analytics_router.post("/analytics/outcome", status_code=status.HTTP_200_OK)
async def report_consumer_outcome(req: DetailedOutcomeReport):
    """Universal outcome reporting endpoint for all consumers (trading_bot, forge, nexus, friday)."""
    # Record in learning engine
    res = outcome_learning_engine.record_outcome(req)
    # Also log to the legacy tracker, for the consumers it supports. Two fields used to be
    # rewritten on the way in: every consumer outside its four-name enum was relabelled
    # "forge" (so nexus and sentinel outcomes were counted as forge outcomes), and a missing
    # provider/service was filled with "gemini"/"code_generation" rather than left unstated.
    # Both corrupt the pass-rate summary the router adapts on.
    if req.consumer in ("forge", "trading_bot", "friday", "human"):
        legacy_req = OutcomeReportRequest(
            consumer=req.consumer,
            request_id=req.request_id,
            outcome=req.outcome,
            detail=req.detail,
            provider_used=req.provider_used,
            service=req.task_type,
        )
        consumer_outcome_tracker.record_outcome(legacy_req)
    self_optimizing_router.adapt_weights_from_outcomes()
    return res


@analytics_router.get("/analytics/calibration", status_code=status.HTTP_200_OK)
async def get_confidence_calibration_curve():
    """Returns honest calibration curve comparing stated confidence vs empirical outcomes."""
    return outcome_learning_engine.get_confidence_calibration()


@analytics_router.get("/analytics/insights", status_code=status.HTTP_200_OK)
async def get_cross_consumer_insights():
    """Returns cross-consumer learning patterns, weekly quality trends, and agent composition stats."""
    return outcome_learning_engine.get_cross_consumer_insights()


@analytics_router.get("/analytics/strategy-bank", status_code=status.HTTP_200_OK)
async def query_strategy_bank_patterns(task_type: str = Query(default="lead_qualification"), query: str = Query(default="")):
    """Queries persistent StrategyBank for relevant past successful situation patterns."""
    return outcome_learning_engine.query_strategy_bank(task_type, query)


# Admin endpoints
@analytics_router.get("/admin/dashboard", status_code=status.HTTP_200_OK)
async def get_admin_dashboard():
    """Consolidated admin overview spanning costs, usage, providers, predictive alerts, and health."""
    return {
        "usage_overview": usage_analytics.get_overview(),
        "provider_comparison": usage_analytics.get_providers_comparison(),
        "predictive_forecast": predictive_provider_manager.generate_forecasts(),
        "downstream_outcomes": consumer_outcome_tracker.get_outcome_summary(),
        "active_alerts": [a.model_dump() for a in alert_system.get_alerts(unacknowledged_only=True)]
    }


@analytics_router.get("/admin/optimization/status", status_code=status.HTTP_200_OK)
async def get_optimization_status():
    """Returns current self-optimizing router weights and rebalance audit logs."""
    return self_optimizing_router.get_routing_status()


@analytics_router.get("/admin/costs", status_code=status.HTTP_200_OK)
async def get_admin_costs():
    """Returns detailed cost breakdowns, cost per successful recommendation, and provider efficiency leaderboards."""
    from app.analytics.cost_tracking import provider_cost_tracker
    from app.routing.cost_router import cost_aware_router
    overview = usage_analytics.get_overview()
    prov_comp = usage_analytics.get_providers_comparison()
    cost_report = provider_cost_tracker.get_cost_report()
    return {
        "daily_budget_usd": overview["daily_budget_usd"],
        "total_cost_usd": overview["total_cost_usd"],
        "budget_used_pct": overview["budget_used_pct"],
        "ceiling_alert_active": overview["ceiling_alert_active"],
        "cost_per_successful_outcome_usd": cost_report["cost_per_successful_outcome_usd"],
        "projected_monthly_spend_usd": cost_report["projected_monthly_spend_usd"],
        "provider_leaderboard": cost_report["provider_leaderboard"],
        "consumer_budgets": cost_aware_router.get_budget_dashboard(),
        "provider_costs": {p: data["cost_usd"] for p, data in prov_comp.items()}
    }


@analytics_router.get("/admin/budgets", status_code=status.HTTP_200_OK)
async def get_consumer_budgets_dashboard():
    """Returns per-consumer monthly budget allocations, soft/hard limits, and spend trends."""
    from app.routing.cost_router import cost_aware_router
    return cost_aware_router.get_budget_dashboard()


@analytics_router.get("/admin/reliability", status_code=status.HTTP_200_OK)
async def get_admin_reliability():
    """Returns reliability metrics and provider failure rates."""
    outcomes = consumer_outcome_tracker.get_outcome_summary()
    perf = provider_intel.get_performance_matrix()
    return {
        "downstream_pass_rates": outcomes["provider_verification_pass_rates"],
        "overall_success_rate_pct": outcomes["overall_downstream_success_rate_pct"],
        "failure_patterns": perf["failure_pattern_analysis"]
    }


@analytics_router.get("/admin/alerts", status_code=status.HTTP_200_OK)
async def get_admin_alerts(unacknowledged_only: bool = Query(default=False)):
    """Returns list of active system alerts."""
    return [a.model_dump() for a in alert_system.get_alerts(unacknowledged_only=unacknowledged_only)]


@analytics_router.get("/admin/export", status_code=status.HTTP_200_OK)
async def export_analytics_data(format: str = Query(default="json", description="json or csv")):
    """Exports all telemetry and usage analytics in JSON or CSV format."""
    overview = usage_analytics.get_overview()
    provs = usage_analytics.get_providers_comparison()
    if format.lower() == "csv":
        csv_lines = ["provider,calls,tokens,cost_usd,avg_latency_ms"]
        for p, d in provs.items():
            csv_lines.append(f"{p},{d['calls']},{d['tokens']},{d['cost_usd']},{d['avg_latency_ms']}")
        return {"content_type": "text/csv", "data": "\n".join(csv_lines)}
    return {"overview": overview, "providers": provs}
