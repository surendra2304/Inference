"""Production Monitoring and Performance Metrics Tracker."""

import math
import time
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any


def _percentile(data: list[float], percentile: float) -> float:
    """Calculates percentile using standard Python libraries without numpy dependency."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1


class PerformanceMonitor:
    """Tracks latency percentiles, error rates, cache performance, and provider health."""

    def __init__(self) -> None:
        self.request_latencies: list[float] = []
        self.total_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()

        # Provider metrics: provider_name -> observations from this process only.
        self.provider_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "success": 0,
            "failure": 0,
            "latencies": [],
            "last_observed_at": None,
        })

        # Debate engine metrics
        self.debate_durations: list[float] = []
        self.total_deliberations = 0
        self.agent_calls: dict[str, int] = defaultdict(int)

    def record_request(self, latency_sec: float, success: bool = True) -> None:
        self.total_requests += 1
        if not success:
            self.failed_requests += 1
        self.request_latencies.append(latency_sec)
        # Keep last 5000 records
        if len(self.request_latencies) > 5000:
            self.request_latencies.pop(0)

    def record_provider_call(self, provider_name: str, latency_sec: float, success: bool = True) -> None:
        stats = self.provider_stats[provider_name]
        if success:
            stats["success"] += 1
        else:
            stats["failure"] += 1
        stats["latencies"].append(latency_sec)
        stats["last_observed_at"] = time.time()
        if len(stats["latencies"]) > 1000:
            stats["latencies"].pop(0)

    def record_agent_participation(self, agent_id: str) -> None:
        self.agent_calls[agent_id] += 1

    def get_api_metrics(self) -> dict[str, Any]:
        """Calculates p50, p95, p99 latencies, throughput, and error rate."""
        if not self.request_latencies:
            # "nothing measured" is not "measured as zero": a 0.0 error rate with no sample
            # count reads like a clean bill of health for a process that has served nothing.
            return {
                "total_requests": self.total_requests,
                "samples": 0,
                "error_rate_pct": None,
                "p50_latency_sec": None,
                "p95_latency_sec": None,
                "p99_latency_sec": None,
                "avg_latency_sec": None,
                "basis": "no request latencies recorded in this process yet",
                "uptime_seconds": round(time.time() - self.start_time, 1)
            }

        lats = self.request_latencies
        err_pct = round((self.failed_requests / self.total_requests) * 100.0, 2) if self.total_requests > 0 else 0.0

        return {
            "total_requests": self.total_requests,
            "samples": len(lats),
            "latency_window": "most recent 5000 recorded requests",
            "error_rate_pct": err_pct,
            "p50_latency_sec": round(_percentile(lats, 50), 3),
            "p95_latency_sec": round(_percentile(lats, 95), 3),
            "p99_latency_sec": round(_percentile(lats, 99), 3),
            "avg_latency_sec": round(sum(lats) / len(lats), 3),
            "min_latency_sec": round(min(lats), 3),
            "max_latency_sec": round(max(lats), 3),
            "uptime_seconds": round(time.time() - self.start_time, 1)
        }

    def get_provider_health(self) -> dict[str, Any]:
        """Returns per-provider success rates and average response times."""
        result = {}
        for p_name, stats in self.provider_stats.items():
            total = stats["success"] + stats["failure"]
            success_rate = round((stats["success"] / total) * 100.0, 1) if total > 0 else None
            avg_lat = round(sum(stats["latencies"]) / len(stats["latencies"]), 3) if stats["latencies"] else 0.0
            result[p_name] = {
                "total_calls": total,
                "success_rate_pct": success_rate,
                "avg_latency_sec": avg_lat,
                "status": "unknown" if success_rate is None else "observed_ok" if success_rate >= 80.0 else "degraded",
                "evidence_class": "observed_provider_calls",
                "observed_at": (
                    datetime.fromtimestamp(stats["last_observed_at"], UTC).isoformat()
                    if stats["last_observed_at"] is not None
                    else None
                ),
            }
        return result

    def record_deliberation(self, duration_sec: float) -> None:
        """Record a completed multi-agent deliberation (bounded duration window)."""
        self.debate_durations.append(duration_sec)
        if len(self.debate_durations) > 5000:
            self.debate_durations.pop(0)
        self.total_deliberations += 1

    def get_debate_metrics(self) -> dict[str, Any]:
        """Multi-agent deliberation metrics, measured or withheld.

        ``recommendation_quality_score`` used to be the constant ``100.0`` with the comment
        "Certified perfect 100/100 by audit suite" — a marketing number on an audit surface,
        published even when no deliberation had ever run. There is no per-recommendation
        outcome tracking in this process, so no quality score can be computed; the field is
        reported as ``None`` with the reason, and ``total_deliberations`` now counts
        deliberations rather than every HTTP request the process served.
        """
        durations = sorted(self.debate_durations)
        return {
            "agent_participation_counts": dict(self.agent_calls),
            "total_deliberations": self.total_deliberations,
            "total_api_requests": self.total_requests,
            "debate_duration_samples": len(durations),
            "debate_duration_p50_sec": round(_percentile(durations, 50), 3) if durations else None,
            "debate_duration_p95_sec": round(_percentile(durations, 95), 3) if durations else None,
            "recommendation_quality_score": None,
            "recommendation_quality_basis": (
                "withheld: this process does not track per-recommendation outcomes, so a "
                "quality score cannot be derived; it was previously the constant 100.0"
            ),
            "evidence_class": "observed_deliberations" if durations else "no_deliberations_observed",
        }


monitor = PerformanceMonitor()
