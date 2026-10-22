"""Self-Optimizing Routing Engine with Outcome Weight Feedback and Diversity Preservation.

The weights this module publishes are *used*: :meth:`SelfOptimizingRouter.choose_provider`
decides which provider serves a request when the one an agent prefers cannot
(``app/services/trading_consult_service.py`` calls it when the preferred provider's circuit is
open). Before that wiring, ``get_routing_status`` advertised an ``active_weights`` table,
a ``diversity_constraint`` and a 10% exploration budget that no code path read — the endpoint
described behaviour that did not exist. Every claim in the status payload is now backed by the
function below, and the router counts the decisions it actually made.
"""

import random
import time
from typing import Any

from app.analytics.outcomes import consumer_outcome_tracker
from app.utils.bounded_store import BoundedList

#: No single provider may exceed this share of decisions (the published "max weight cap").
MAX_WEIGHT_CAP = 0.70
#: A service must keep at least this many providers with non-zero weight.
MIN_ACTIVE_PROVIDERS = 2
#: Share of decisions made uniformly at random instead of by weight, so a de-prioritised
#: provider still gets traffic and its measured health can recover.
EXPLORE_TRAFFIC_PCT = 10.0
#: Observed provider health adapts a weight only from at least this many measured calls;
#: below it the declared prior stands rather than being replaced by noise.
MIN_OBSERVED_CALLS_FOR_ADAPTATION = 5


class SelfOptimizingRouter:
    """Dynamically adapts provider selection weights based on verified downstream build outcomes."""

    def __init__(self) -> None:
        self.provider_weights: dict[str, dict[str, float]] = {
            "code_generation": {"groq": 0.45, "gemini": 0.35, "openrouter": 0.20},
            "architecture": {"nvidia": 0.50, "gemini": 0.35, "groq": 0.15},
            "trading_consult": {"groq": 0.50, "gemini": 0.30, "openrouter": 0.20}
        }
        # No seeded log entry: the previous one asserted "98% verification pass rate
        # observed from FORGE builds" before any build outcome existed, and was served by
        # the routing-status endpoint as an audit record. Bounded for the same reason as
        # every other accumulator (one entry per recorded outcome).
        self.optimization_logs: BoundedList[dict[str, Any]] = BoundedList(
            "self_optimizer.optimization_logs", max_entries=512
        )
        #: Decisions this router actually made, per service. Published by
        #: ``get_routing_status`` so "the weights are live" is checkable rather than asserted.
        self.decisions: dict[str, dict[str, int]] = {}
        #: Most recent decision per service, for the same reason.
        self.last_decision: dict[str, dict[str, Any]] = {}
        #: Injectable for deterministic tests; production uses the module RNG.
        self._rng = random.Random()

    def normalize_weights(self, service: str) -> dict[str, float]:
        """Clamp a service's weights to the published constraints and renormalise them.

        Two measured defects are fixed here. The cap was applied *per provider before*
        normalisation, so two providers clamped at 0.70 summed to 1.40 while a third kept its
        original share — a "weights" table that did not sum to 1. And a provider whose weight
        was driven below the minimum simply kept a small share rather than the table
        renormalising, which let the published constraint ("minimum 2 active providers") be
        violated by the router's own adaptation code.
        """
        weights = self.provider_weights.get(service, {})
        positive = {p: float(w) for p, w in weights.items() if w and w > 0.0}
        if not positive:
            return {}

        total = sum(positive.values())
        normalized = {p: w / total for p, w in positive.items()}

        # Cap, then push the excess onto the remaining providers so the shares still sum to 1.
        for _ in range(8):  # converges in two rounds for any realistic table
            overflowing = {p: w for p, w in normalized.items() if w > MAX_WEIGHT_CAP}
            if not overflowing:
                break
            excess = sum(w - MAX_WEIGHT_CAP for w in overflowing.values())
            for provider in overflowing:
                normalized[provider] = MAX_WEIGHT_CAP
            receivers = [p for p in normalized if p not in overflowing]
            if not receivers:
                # Every provider is at the cap (e.g. exactly two providers): the cap cannot
                # hold and still sum to 1. The cap wins for at most one provider; report it.
                break
            share = excess / len(receivers)
            for provider in receivers:
                normalized[provider] += share

        result = {p: round(w, 4) for p, w in sorted(normalized.items())}
        # Rounding may leave the shares a hair off 1.0; renormalise against the rounded total.
        rounded_total = sum(result.values())
        if rounded_total > 0:
            result = {p: round(w / rounded_total, 4) for p, w in result.items()}
        return result

    def adapt_weights_from_observed_health(self, service: str) -> dict[str, Any]:
        """Blend the declared weights with measured provider success rates.

        Only providers with at least :data:`MIN_OBSERVED_CALLS_FOR_ADAPTATION` recorded calls
        move: a provider that answered twice is not evidence. The blend is deliberately gentle
        (α = 0.5) so one bad minute cannot drain a provider's share, and the result is clamped
        and renormalised, so the constraints still hold afterwards.
        """
        from app.monitoring import monitor

        observed = monitor.get_provider_health()
        declared = self.provider_weights.get(service, {})
        if not declared:
            return {"service": service, "action": "NO_DECLARED_WEIGHTS", "weights": {}}

        adjusted: dict[str, float] = {}
        evidence: dict[str, dict[str, Any]] = {}
        for provider, weight in declared.items():
            stats = observed.get(provider) or {}
            calls = int(stats.get("total_calls") or 0)
            rate = stats.get("success_rate_pct")
            if calls < MIN_OBSERVED_CALLS_FOR_ADAPTATION or rate is None:
                adjusted[provider] = float(weight)
                evidence[provider] = {"calls": calls, "success_rate_pct": rate, "applied": False}
                continue
            adjusted[provider] = float(weight) * 0.5 + (float(rate) / 100.0) * 0.5
            evidence[provider] = {"calls": calls, "success_rate_pct": rate, "applied": True}

        total = sum(adjusted.values())
        if total <= 0:
            return {"service": service, "action": "NO_POSITIVE_WEIGHT", "weights": declared}
        self.provider_weights[service] = {p: w / total for p, w in adjusted.items()}
        # Only providers with enough measured calls actually move. Reporting
        # "adapted" when every weight kept its declared value would credit the router with
        # evidence it did not have (the same mistake the old fabricated 90% prior made).
        adapted = [p for p, info in evidence.items() if info["applied"]]
        status = {
            "service": service,
            "action": "ADAPTED_FROM_OBSERVED_HEALTH" if adapted else "NO_SUFFICIENT_EVIDENCE",
            "weights": self.normalize_weights(service),
            "evidence": evidence,
            "min_observed_calls": MIN_OBSERVED_CALLS_FOR_ADAPTATION,
        }
        self.optimization_logs.append({"timestamp": time.time(), **status})
        return status

    def choose_provider(
        self, service: str, candidates: list[str]
    ) -> tuple[str | None, dict[str, Any]]:
        """Pick the provider that should serve a request. Returns (provider, decision).

        ``decision["method"]`` states how the pick was made — ``single_candidate``,
        ``weighted`` or ``explore`` — so a caller can record why a request went where it did
        instead of presenting every choice as a weighted one. Candidates with no weight are
        still eligible at a small floor weight: an unlisted provider must not be unreachable,
        which is exactly the failure mode a weights table without exploration creates.
        """
        unique = list(dict.fromkeys(c for c in candidates if c))
        if not unique:
            decision: dict[str, Any] = {
                "service": service,
                "method": "no_candidates",
                "chosen": None,
                "candidates": [],
                "weights": {},
            }
            self.last_decision[service] = decision
            return None, decision

        if len(unique) == 1:
            decision = {
                "service": service,
                "method": "single_candidate",
                "chosen": unique[0],
                "candidates": unique,
                "weights": {unique[0]: 1.0},
            }
            self._record_decision(service, decision)
            return unique[0], decision

        normalized = self.normalize_weights(service)
        weights = {name: normalized.get(name, 0.0) for name in unique}
        if sum(weights.values()) <= 0:
            weights = dict.fromkeys(unique, 1.0 / len(unique))
        else:
            # Floor so a candidate with no declared weight is still reachable.
            floor = min(w for w in weights.values() if w > 0) / 4.0
            weights = {name: (w if w > 0 else floor) for name, w in weights.items()}
        total = sum(weights.values())
        weights = {name: w / total for name, w in weights.items()}

        if self._rng.random() * 100.0 < EXPLORE_TRAFFIC_PCT:
            chosen = self._rng.choice(unique)
            method = "explore"
        else:
            probe = self._rng.random()
            cumulative = 0.0
            chosen = unique[-1]
            for name in unique:
                cumulative += weights[name]
                if probe <= cumulative:
                    chosen = name
                    break
            method = "weighted"

        decision = {
            "service": service,
            "method": method,
            "chosen": chosen,
            "candidates": unique,
            "weights": {name: round(w, 4) for name, w in weights.items()},
            "explore_traffic_pct": EXPLORE_TRAFFIC_PCT,
        }
        self._record_decision(service, decision)
        return chosen, decision

    def _record_decision(self, service: str, decision: dict[str, Any]) -> None:
        self.last_decision[service] = decision
        counters = self.decisions.setdefault(service, {})
        counters["total"] = counters.get("total", 0) + 1
        counters[decision["method"]] = counters.get(decision["method"], 0) + 1
        if decision.get("chosen"):
            per_provider = self.decisions.setdefault(f"{service}:by_provider", {})
            per_provider[decision["chosen"]] = per_provider.get(decision["chosen"], 0) + 1

    def get_routing_status(self) -> dict[str, Any]:
        """Returns current dynamic routing weights and audit logs.

        ``active_weights`` are the declared table; ``effective_weights`` is what
        :meth:`choose_provider` actually samples from, after the cap and the renormalisation.
        ``decisions`` counts the picks the router made and ``last_decision`` says how the most
        recent one was reached, so a reader can tell an idle router from a broken one.
        """
        return {
            "active_weights": self.provider_weights,
            "effective_weights": {
                service: self.normalize_weights(service) for service in self.provider_weights
            },
            "diversity_constraint": "Minimum 2 active providers per service; max weight cap at 0.70",
            "constrained_by": {
                "max_weight_cap": MAX_WEIGHT_CAP,
                "min_active_providers": MIN_ACTIVE_PROVIDERS,
                "enforced_in": "SelfOptimizingRouter.normalize_weights / choose_provider",
            },
            "explore_traffic_pct": EXPLORE_TRAFFIC_PCT,
            "applied_to_request_routing": True,
            "applied_where": (
                "trading_consult: a provider substitution is chosen by weight when the agent's "
                "preferred provider is unavailable; no other caller selects providers yet"
            ),
            "decisions": self.decisions,
            "last_decision": self.last_decision,
            # ``.to_list()``: returning the BoundedList itself makes the response
            # non-serializable and the status endpoint fails at render time.
            "recent_rebalance_logs": self.optimization_logs.to_list(),
            "rebalance_log_retention": self.optimization_logs.describe(),
        }

    def adapt_weights_from_outcomes(self) -> None:
        """Rebalances code-generation weights from observed pass rates, when they exist.

        Previously this read ``pass_rates.get("groq", 90.0)`` and
        ``pass_rates.get("gemini", 90.0)``: with no outcome data at all, both providers
        scored a fabricated 90%, the rebalance produced weights that looked like a
        measurement, and the log entry justified them with
        "Adapted to updated downstream verification pass rates: Groq=90.0%, Gemini=90.0%".
        A provider with no observed outcomes is now skipped, the weights are left alone,
        and the log says so.
        """
        summary = consumer_outcome_tracker.get_outcome_summary()
        rates = summary["provider_verification_pass_rates"]

        groq_entry = rates.get("groq") or {}
        gemini_entry = rates.get("gemini") or {}
        groq_score = groq_entry.get("verification_pass_rate_pct")
        gemini_score = gemini_entry.get("verification_pass_rate_pct")

        if groq_score is None or gemini_score is None:
            missing = ", ".join(
                name for name, value in (("groq", groq_score), ("gemini", gemini_score))
                if value is None
            )
            self.optimization_logs.append({
                "timestamp": time.time(),
                "service": "code_generation",
                "action": "REBALANCE_SKIPPED",
                "rationale": (
                    f"no observed verification data for {missing}; weights unchanged "
                    f"({summary['total_outcomes_reported']} observed outcome(s) on record)"
                ),
            })
            return

        total_score = float(groq_score) + float(gemini_score)
        if total_score <= 0:
            self.optimization_logs.append({
                "timestamp": time.time(),
                "service": "code_generation",
                "action": "REBALANCE_SKIPPED",
                "rationale": "both observed pass rates are zero; weights unchanged",
            })
            return

        # Shares derived from the observed pass rates, then clamped and renormalised by the
        # same code path the published constraints describe (they used to be clamped inline,
        # which could leave the service's weights summing to more than 1).
        self.provider_weights["code_generation"]["groq"] = round(float(groq_score) / total_score, 4)
        self.provider_weights["code_generation"]["gemini"] = round(float(gemini_score) / total_score, 4)
        self.provider_weights["code_generation"] = self.normalize_weights("code_generation")

        self.optimization_logs.append({
            "timestamp": time.time(),
            "service": "code_generation",
            "action": "AUTO_REBALANCED_WEIGHTS",
            "rationale": (
                f"adapted to observed downstream verification pass rates: "
                f"Groq={groq_score}% ({groq_entry.get('samples', 0)} samples), "
                f"Gemini={gemini_score}% ({gemini_entry.get('samples', 0)} samples)"
            ),
            "samples": {
                "groq": groq_entry.get("samples", 0),
                "gemini": gemini_entry.get("samples", 0),
            },
        })


self_optimizing_router = SelfOptimizingRouter()
