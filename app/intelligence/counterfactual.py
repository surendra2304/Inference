"""Counterfactual Reasoning Engine: What-If Scenario Analysis with Confidence Intervals.

No effect is estimated here. The previous version returned +12.0% for any variable whose name
contained the letter "b" and -8.5% for every other, with a fixed 72% confidence and a 95% interval
of plus or minus 7 points. Those were constants keyed on spelling, not estimates. An estimate needs
observed outcomes for both arms; this engine has none, so it reports ``not_estimated``.
"""

from pydantic import BaseModel, Field


class CounterfactualScenario(BaseModel):
    scenario_name: str
    proposed_intervention: str
    baseline_variable: str
    counterfactual_variable: str


class CounterfactualResult(BaseModel):
    scenario_name: str
    #: None: no estimate was produced.
    estimated_outcome_delta_pct: float | None
    confidence_interval_95: dict[str, float] | None = Field(description="Lower and upper bound of 95% CI, or None")
    counterfactual_confidence: float | None = None
    estimate_status: str = "not_estimated"
    is_counterfactual: bool = True
    reasoning_basis: str
    caveats: list[str] = Field(default_factory=list)


class CounterfactualReasoningEngine:
    """Scenario framing for what-if questions; effect estimates are withheld until data supports them."""

    def evaluate_what_if(self, scenario: CounterfactualScenario) -> CounterfactualResult:
        caveats = [
            "No historical outcome data for the baseline or the counterfactual arm was available; no effect was estimated.",
            "Any counterfactual figure must be labelled as counterfactual wherever it is shown.",
        ]
        return CounterfactualResult(
            scenario_name=scenario.scenario_name,
            estimated_outcome_delta_pct=None,
            confidence_interval_95=None,
            counterfactual_confidence=None,
            estimate_status="not_estimated: no observed outcomes for the baseline or counterfactual arm",
            is_counterfactual=True,
            reasoning_basis=(
                f"Intervention '{scenario.proposed_intervention}' would substitute "
                f"'{scenario.baseline_variable}' with '{scenario.counterfactual_variable}'; "
                "an effect needs observed outcomes for both arms."
            ),
            caveats=caveats,
        )


counterfactual_engine = CounterfactualReasoningEngine()
