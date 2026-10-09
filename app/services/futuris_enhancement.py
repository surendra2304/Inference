"""Futuris Enhancement & Statistical Grounding Engine.

Features:
- Enhances raw statistical forecasts from Futuris with qualitative intelligence, risks, and drivers.
- Agent Panel:
  - Data Analyst -> Interprets statistical distributions, margins, and confidence intervals.
  - Strategist -> Assesses macro implications, policy shifts, and structural drivers.
  - Critic -> Challenges modeling assumptions, sample biases, and regime shifts.
- Statistical Grounding:
  - Injects relevant Futuris statistical forecasts into Trading Bot, Sentinel, Nexus, and FORGE recommendations.
"""

import json
import time
from typing import Any

from pydantic import BaseModel, Field

from app.analytics.usage_analytics import usage_analytics
from app.routing.consumer_router import consumer_router
from app.utils.bounded_store import DEFAULT_MAX_ENTRIES, BoundedStore


class StatisticalForecastInput(BaseModel):
    metric_name: str | None = "target_metric"
    point_estimate: float
    confidence_interval: list[float] = Field(..., min_length=2, max_length=2, description="[lower_bound, upper_bound]")
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    model_used: str = Field(description="e.g. ARIMA, Prophet, GARCH, MonteCarlo")


class FuturisEnhanceRequest(BaseModel):
    request_id: str
    statistical_forecast: StatisticalForecastInput
    target_context: dict[str, Any] = Field(default_factory=dict)
    contextual_factors: list[str] = Field(default_factory=list)
    question: str | None = "Given this forecast and context, what risks or drivers should be considered?"


class EnhancedAssessmentPayload(BaseModel):
    key_risks: list[str]
    contextual_drivers: list[str]
    uncertainty_factors: list[str]
    qualitative_adjustments: list[str]


class FuturisEnhanceResponse(BaseModel):
    request_id: str
    enhanced_assessment: EnhancedAssessmentPayload
    confidence_adjustment: float | None = Field(
        default=None,
        description="Not measured by this service; always null. Kept for response-shape compatibility.",
    )
    dissent: list[str] = Field(default_factory=list)
    grounded_forecast_summary: str
    provenance: dict[str, Any] = Field(default_factory=dict)


class StatisticalGroundingEngine:
    """Provides statistical grounding context to other consumers (Trading, Sentinel, FORGE, Nexus)."""

    def __init__(self) -> None:
        # No measured forecasts exist in this service. Before this, three hard-coded series
        # (for example volatility_btc 0.045 with an 88% probability) were served as grounding.
        self.cached_forecasts: dict[str, StatisticalForecastInput] = {}

    def get_grounding_context(self, metric_key: str) -> dict[str, Any] | None:
        """Retrieves active statistical forecast grounding for a specific consumer metric."""
        forecast = self.cached_forecasts.get(metric_key)
        if not forecast:
            return None
        return {
            "grounding_available": True,
            "metric": forecast.metric_name,
            "point_estimate": forecast.point_estimate,
            "ci_95": forecast.confidence_interval,
            "forecast_model": forecast.model_used,
            "statistical_confidence": forecast.probability
        }


class FuturisEnhancementService:
    """Specialized qualitative enhancement service for Futuris statistical models."""

    def __init__(self) -> None:
        # Bounded: one entry per request used to accumulate without limit
        # (measured: +9.03 MB/1k requests on nexus, +5.19 on sentinel, retained
        # after gc). Entries are evicted LRU beyond the ceiling; the store
        # records how many, so a lookup miss can say "evicted" instead of
        # pretending the id never existed.
        self.provenance_store = BoundedStore[dict[str, Any]](
            "futuris.provenance_store", max_entries=DEFAULT_MAX_ENTRIES
        )
        self.grounding_engine = StatisticalGroundingEngine()

    async def _qualitative_review(self, req: FuturisEnhanceRequest) -> dict[str, Any]:
        """Optional qualitative review by a model. Returns status ``parsed`` only when the model
        answered and its answer is a JSON object with the expected lists; otherwise the review is
        reported as ``not_performed`` or ``unparsed`` with the reason, and no risks are invented."""
        from app.providers.unified_manager import UnifiedExecutionRequest, unified_provider_manager
        from app.utils.model_json import extract_json_object

        forecast = req.statistical_forecast
        prompt = (
            "You review a statistical forecast. Use ONLY the numbers and context given here; do not "
            "invent events or facts. Return a single JSON object with the keys key_risks, "
            "contextual_drivers, uncertainty_factors, qualitative_adjustments, dissent. Each is a list "
            "of short strings.\n"
            f"Model: {forecast.model_used}\n"
            f"Point estimate: {forecast.point_estimate}\n"
            f"Interval: {forecast.confidence_interval}\n"
            f"Probability: {forecast.probability}\n"
            f"Context: {json.dumps(req.target_context, default=str)[:1500]}\n"
            f"Factors: {json.dumps(req.contextual_factors, default=str)[:1500]}\n"
            f"Question: {req.question}"
        )
        try:
            resp = await unified_provider_manager.execute(
                UnifiedExecutionRequest(
                    provider="auto",
                    agent_role="data_analyst",
                    prompt=prompt,
                    max_tokens=800,
                    temperature=0.2,
                )
            )
        except Exception:  # noqa: BLE001 - a failed review is reported, never fabricated
            return {"status": "not_performed", "reason": "qualitative model call failed"}
        if resp.degraded or not resp.content:
            return {"status": "not_performed", "reason": "no model produced output"}

        parsed = extract_json_object(resp.content)
        keys = ("key_risks", "contextual_drivers", "uncertainty_factors", "qualitative_adjustments", "dissent")
        if not isinstance(parsed, dict) or not all(isinstance(parsed.get(k), list) for k in keys):
            return {"status": "unparsed", "reason": "model answer was not the expected JSON object"}

        def clean(name: str) -> list[str]:
            return [str(x)[:300] for x in parsed[name] if isinstance(x, (str, int, float))][:10]

        return {"status": "parsed", **{k: clean(k) for k in keys}}

    async def enhance_forecast(self, req: FuturisEnhanceRequest) -> FuturisEnhanceResponse:
        start_time = time.perf_counter()

        # Check deduplication cache
        from app.governance.tenant_manager import tenant_manager
        # Typed lookup: a stored payload that does not match this response model is
        # evicted and treated as a miss, so an inconsistent cache entry can never
        # surface as HTTP 500 (it previously did — see app/governance/tenant_manager.py).
        cached = tenant_manager.check_deduplication_model(
            req.request_id, FuturisEnhanceResponse, namespace="futuris_enhancement"
        )
        if cached is not None:
            return cached

        forecast = req.statistical_forecast
        context = req.target_context
        factors = req.contextual_factors
        lo, hi = forecast.confidence_interval[0], forecast.confidence_interval[1]
        ci_width = hi - lo

        # Everything in this block is either measured from the caller's own numbers or is
        # echoed from the caller. Before this, the risks and drivers were fixed sentences (one
        # asserted a "macro policy announcement"), the dissent was a fixed note, the response
        # claimed three agents had been consulted, and confidence_adjustment was a constant 0.08.
        width_note = f"{ci_width:.4g}"
        if forecast.point_estimate:
            width_note += f" ({abs(ci_width / forecast.point_estimate) * 100:.1f}% of the point estimate {forecast.point_estimate:.4g})"
        key_risks: list[str] = [f"Interval width reported for {forecast.model_used}: {width_note}."]
        key_risks += [f"Caller-supplied factor: {str(f)[:200]}" for f in factors[:10]]

        contextual_drivers: list[str] = [f"Caller context {k}: {str(v)[:200]}" for k, v in list(context.items())[:10]]

        uncertainty_factors: list[str] = [
            f"Interval [{lo:.4g}, {hi:.4g}] is taken as supplied; its assumptions were not checked by this service.",
        ]
        if forecast.probability is None:
            uncertainty_factors.append("No probability was supplied, so none is reported.")

        review = await self._qualitative_review(req)
        if review["status"] == "parsed":
            key_risks += review["key_risks"]
            contextual_drivers += review["contextual_drivers"]
            uncertainty_factors += review["uncertainty_factors"]
            qualitative_adjustments: list[str] = review["qualitative_adjustments"]
            dissent: list[str] = review["dissent"]
            agents = ["qualitative_model_review"]
        else:
            qualitative_adjustments = []
            dissent = []
            agents = []

        enhanced_assessment = EnhancedAssessmentPayload(
            key_risks=key_risks,
            contextual_drivers=contextual_drivers,
            uncertainty_factors=uncertainty_factors,
            qualitative_adjustments=qualitative_adjustments,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        provenance = {
            "request_id": req.request_id,
            "agents_consulted": agents,
            "qualitative_review_status": review["status"],
            "qualitative_review_reason": review.get("reason"),
            "model_evaluated": forecast.model_used,
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time(),
        }

        summary = (
            f"Forecast {forecast.model_used}: point {forecast.point_estimate:.4g}, "
            f"interval [{lo:.4g}, {hi:.4g}]. Qualitative review: {review['status']}."
        )

        response = FuturisEnhanceResponse(
            request_id=req.request_id,
            enhanced_assessment=enhanced_assessment,
            confidence_adjustment=None,
            dissent=dissent,
            grounded_forecast_summary=summary,
            provenance=provenance,
        )

        # Store in provenance ledger
        self.provenance_store[req.request_id] = {
            "request": req.model_dump(),
            "response": response.model_dump()
        }

        # Store in deduplication cache
        tenant_manager.store_deduplication(
            req.request_id, response.model_dump(), namespace="futuris_enhancement"
        )

        # Track usage
        consumer_router.record_usage("futuris", tokens=None, latency_sec=latency_ms / 1000.0)
        # ``forecast.probability or 0.85`` replaced an absent probability with a
        # measured-looking 0.85; an unmeasured confidence is now recorded as None.
        usage_analytics.log_request(
            consumer="futuris",
            service="futuris_enhance",
            latency_ms=latency_ms,
            success=True,
            confidence=forecast.probability,
        )

        return response

    def get_provenance(self, request_id: str) -> dict[str, Any] | None:
        return self.provenance_store.get(request_id)

    def provenance_retention(self) -> dict[str, Any]:
        """How much provenance this service still holds, and what it has dropped.

        Served so that a 404 on an audit endpoint can be attributed: "never recorded"
        and "recorded but evicted" are different statements about the same request id.
        """
        return self.provenance_store.describe()


futuris_enhancement_service = FuturisEnhancementService()
