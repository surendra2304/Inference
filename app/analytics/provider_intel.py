"""Provider Performance Intelligence and Failure Matrix Engine."""

from typing import Any

from app.analytics.usage_analytics import usage_analytics

#: Observed calls a (provider, service) cell needs before its figures are reported as a rate.
MIN_CALLS_PER_CELL = 5


class ProviderPerformanceIntelligence:
    """Provider performance computed from observed calls only.

    The previous version returned a fixed matrix (for example groq/code_generation at 98.2% success,
    35 ms and "confidence 0.93"), a fixed failure table and routing advice such as "Groq is 40%
    faster". None of that came from a call. The matrix is now derived from the observed request
    records in ``usage_analytics``; a cell with fewer than ``MIN_CALLS_PER_CELL`` calls is labelled
    ``insufficient_calls`` instead of being given a rate.
    """

    def get_performance_matrix(self) -> dict[str, Any]:
        observed = [r for r in usage_analytics.records if r.source == "observed" and r.provider]
        if not observed:
            return {
                "status": "no_observed_calls",
                "provider_service_matrix": {},
                "failure_pattern_analysis": [],
                "routing_recommendations": [],
            }

        cells: dict[tuple[str, str], list[Any]] = {}
        for record in observed:
            cells.setdefault((record.provider or "unattributed", record.service), []).append(record)

        matrix: dict[str, dict[str, Any]] = {}
        for (provider, service), records in sorted(cells.items()):
            calls = len(records)
            latencies = [r.latency_ms for r in records if r.latency_ms is not None]
            reported = [r.confidence for r in records if r.confidence is not None]
            cell: dict[str, Any] = {"calls": calls, "status": "measured" if calls >= MIN_CALLS_PER_CELL else "insufficient_calls"}
            if calls >= MIN_CALLS_PER_CELL:
                cell["success_rate_pct"] = round(sum(1 for r in records if r.success) / calls * 100.0, 1)
                cell["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1) if latencies else None
                cell["mean_caller_reported_confidence"] = round(sum(reported) / len(reported), 3) if reported else None
            matrix.setdefault(provider, {})[service] = cell

        failures: list[dict[str, Any]] = []
        by_provider: dict[str, list[Any]] = {}
        for record in observed:
            by_provider.setdefault(record.provider or "unattributed", []).append(record)
        for provider, records in sorted(by_provider.items()):
            failed = sum(1 for r in records if not r.success)
            if failed:
                failures.append({
                    "provider": provider,
                    "failed_calls": failed,
                    "frequency_pct": round(failed / len(records) * 100.0, 2),
                    "dominant_error": "not_recorded: request records carry no error class",
                })

        recommendations: list[str] = []
        for service in sorted({r.service for r in observed}):
            ranked = [
                (provider, cell)
                for provider, services in matrix.items()
                for svc, cell in services.items()
                if svc == service and cell.get("avg_latency_ms") is not None
            ]
            if len(ranked) >= 2:
                ranked.sort(key=lambda item: item[1]["avg_latency_ms"])
                (fast, fc), (slow, sc) = ranked[0], ranked[1]
                recommendations.append(
                    f"{service}: {fast} has the lower mean latency ({fc['avg_latency_ms']} ms over "
                    f"{fc['calls']} calls) than {slow} ({sc['avg_latency_ms']} ms over {sc['calls']} calls)."
                )

        return {
            "status": "measured",
            "provider_service_matrix": matrix,
            "failure_pattern_analysis": failures,
            "routing_recommendations": recommendations,
        }


provider_intel = ProviderPerformanceIntelligence()
