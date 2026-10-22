"""Prometheus text exposition, rendered from live state in exactly one place.

Two endpoints publish metrics for this process:

* ``GET /metrics`` — public and dependency-free, so a scraper can read it even when the
  database or a provider is down (mounted outside authentication in ``app/main.py``);
* ``GET /v1/governance/prometheus-metrics`` — authenticated, adds provider-level detail.

[FACT] They used to build their exposition independently and had drifted apart: the same
window of traffic was published as ``inference_requests_total`` by one and
``inference_api_requests_total`` by the other, percentiles appeared as
``inference_request_duration_seconds{quantile="0.50"}`` in one and as
``inference_latency_p50_seconds`` in the other. A dashboard could not use both, and a fix to
the exposition rules (``#69``: a metric with no samples must be omitted rather than
interpolated as the text ``None``) landed in only one of them. Both now call
:func:`render_prometheus`, so naming, formatting and the rules for absent data cannot diverge
again.

Absent data is never invented: a series whose backing value is ``None`` is omitted entirely,
which is the Prometheus convention for "no observation yet", and every emitted line is
validated against the exposition grammar before it leaves the process.
"""

from __future__ import annotations

import re
from typing import Any

from app.monitoring import monitor
from app.optimization import concurrency_controller, telemetry_cache

#: Prometheus text exposition grammar for one sample line.
SAMPLE_LINE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})? ([-+0-9.eE]+|NaN|Inf)$")

#: Projected values are floats rounded to 4 decimals by the sources; a value that is not a
#: finite number would produce a line a scraper rejects, so it is omitted like absent data.
_ALLOWED_NON_FINITE = ("NaN", "Inf", "+Inf", "-Inf")

QUANTILES = ("0.50", "0.95", "0.99")


class ExpositionError(AssertionError):
    """Raised when a series would be emitted in a form no Prometheus scraper can parse."""


def _format(value: Any) -> str:
    """Turn a Python value into a Prometheus sample value, or refuse to guess.

    Booleans become 1/0 (Prometheus has no boolean sample); ints and floats pass through;
    anything else — a dict, a string, ``None`` — has no exposition representation and is
    rejected loudly rather than stringified into a plausible-looking number.
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value in (float("inf"), float("-inf")):
            return "Inf" if value > 0 else "-Inf"
        return repr(round(value, 6))
    raise ExpositionError(f"no Prometheus representation for {type(value).__name__}: {value!r}")


class Exposition:
    """Accumulates HELP/TYPE/sample lines, dropping absent series and validating the rest."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._declared: dict[str, str] = {}
        self.series_emitted = 0

    def _declare(self, name: str, help_text: str, kind: str) -> None:
        declared = self._declared.get(name)
        if declared is not None:
            if declared != kind:
                raise ExpositionError(f"{name} declared twice with conflicting types")
            return
        self._declared[name] = kind
        self._lines.extend([f"# HELP {name} {help_text}", f"# TYPE {name} {kind}"])

    def sample(
        self,
        name: str,
        help_text: str,
        value: Any,
        kind: str = "gauge",
        labels: str | None = None,
    ) -> None:
        """Emit one series; a ``None`` value means "not measured" and is skipped."""
        if value is None:
            return
        text = _format(value)
        suffix = f"{{{labels}}}" if labels else ""
        line = f"{name}{suffix} {text}"
        if not SAMPLE_LINE.match(line):
            raise ExpositionError(f"malformed exposition line: {line!r}")
        self._declare(name, help_text, kind)
        self._lines.append(line)
        self.series_emitted += 1

    def block(self, name: str, help_text: str, kind: str, samples: list[tuple[str, Any]]) -> None:
        """Emit a labelled family only when at least one series has a real value."""
        emitted = False
        for labels, value in samples:
            if value is None:
                continue
            if not emitted:
                self._declare(name, help_text, kind)
                emitted = True
            self._lines.append(f"{name}{{{labels}}} {_format(value)}")
            self.series_emitted += 1

    def render(self) -> str:
        return "\n".join(self._lines).rstrip() + "\n"


def _provider_detail(exp: Exposition) -> None:
    """Provider identity, circuit state and observed health — authenticated callers only.

    Kept out of the public exporter: which providers a deployment has configured, and which
    of them are currently failing, is operational detail about the infrastructure, not
    something an anonymous caller of ``/metrics`` should be able to read.
    """
    from app.governance.circuit_breaker import circuit_breaker_manager
    from app.governance.tenant_manager import tenant_manager
    from app.providers.health import provider_health_tracker

    exp.sample(
        "inference_tracked_tenants",
        "Tenants registered with the governance store",
        len(tenant_manager.tenants),
    )

    circuits = circuit_breaker_manager.get_circuit_statuses()
    if circuits:
        exp.block(
            "inference_provider_circuit_available",
            "Whether the provider circuit admits traffic (1=yes, 0=no)",
            "gauge",
            [(f'provider="{p}"', 1 if info.get("is_available") else 0) for p, info in sorted(circuits.items())],
        )
        order = {"CLOSED": 0, "HALF_OPEN": 1, "OPEN": 2}
        exp.block(
            "inference_provider_circuit_state",
            "Circuit breaker state (0=CLOSED, 1=HALF_OPEN, 2=OPEN)",
            "gauge",
            [
                (f'provider="{p}"', order.get(str(info.get("state", "CLOSED")).upper(), 3))
                for p, info in sorted(circuits.items())
            ],
        )

    provider_stats = monitor.get_provider_health()
    if provider_stats:
        exp.block(
            "inference_provider_success_ratio",
            "Observed provider success rate (0-1)",
            "gauge",
            [
                (
                    f'provider="{p}"',
                    round(stats["success_rate_pct"] / 100.0, 4),
                )
                for p, stats in sorted(provider_stats.items())
                if stats.get("total_calls") and stats.get("success_rate_pct") is not None
            ],
        )

    health_reports = provider_health_tracker.get_all_health()
    if health_reports:
        exp.block(
            "inference_provider_health_score",
            "Normalised provider health score (0-1)",
            "gauge",
            [(f'provider="{p}"', round(r.health_score, 4)) for p, r in sorted(health_reports.items())],
        )
        exp.block(
            "inference_provider_rate_limited_keys",
            "Number of API keys quarantined for rate limiting",
            "gauge",
            [(f'provider="{p}"', r.quarantined_keys_count) for p, r in sorted(health_reports.items())],
        )


def render_prometheus(*, include_provider_detail: bool) -> str:
    """Render this process's metrics. One implementation feeds every metrics endpoint.

    ``include_provider_detail`` adds the provider/circuit/health families and is set only by
    the authenticated governance endpoint.
    """
    api = monitor.get_api_metrics()
    exp = Exposition()

    exp.sample(
        "inference_requests_total",
        "API requests recorded by this process",
        api.get("total_requests"),
        "counter",
    )
    exp.sample(
        "inference_requests_failed_total",
        "API requests that failed or returned 5xx",
        monitor.failed_requests,
        "counter",
    )
    exp.sample(
        "inference_latency_samples",
        "Latency samples in the reporting window",
        api.get("samples"),
    )
    for quantile, key in zip(QUANTILES, ("p50_latency_sec", "p95_latency_sec", "p99_latency_sec"), strict=True):
        exp.sample(
            "inference_request_duration_seconds",
            "Request latency quantile in seconds",
            api.get(key),
            labels=f'quantile="{quantile}"',
        )
    exp.sample(
        "inference_error_rate_percent",
        "Percentage of recorded requests that failed",
        api.get("error_rate_pct"),
    )
    exp.sample(
        "inference_cache_hit_rate_percent",
        "Telemetry cache hit rate percentage",
        telemetry_cache.get_hit_rate(),
    )
    exp.sample(
        "inference_active_requests",
        "Requests currently in flight",
        concurrency_controller.active_count,
    )
    exp.sample(
        "inference_uptime_seconds",
        "Process uptime in seconds",
        api.get("uptime_seconds"),
    )

    if include_provider_detail:
        _provider_detail(exp)

    return exp.render()
