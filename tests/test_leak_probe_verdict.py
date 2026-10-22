"""The leak probe's verdict must survive the case that fooled it: a *decaying* leak.

[FACT] ``scripts/leak_probe.py`` classifies per-round RSS growth. Its first version judged the
trend alone (last round vs first round), and the leak actually measured in this repo produced
``[5.9, 11.1, 2.4, 1.5, 1.5, 1.3]`` MB per 1000 requests: a decaying series. A trend-only rule
calls that a plateau — while the live process was retaining ~1.4 KB per request, ~36 GB/day at
300 rps, forever. The verdict now also enforces an absolute per-request budget, and these tests
pin the measured series to the correct answer so the probe cannot go quiet again.
"""

import importlib.util
import pathlib

import pytest

PROBE = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "leak_probe.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("leak_probe_under_test", PROBE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


def test_measured_leak_series_is_called_a_leak():
    """The real series, measured live with 6x1000 unique-id POSTs at concurrency 8."""
    kind, reason = probe.verdict([5.9, 11.1, 2.4, 1.5, 1.5, 1.3], per_round_requests=1000)
    assert kind == "leak", reason
    assert "KB/request" in reason


def test_genuine_plateau_is_not_called_a_leak():
    """Caches filling then flat: growth decays to noise for three rounds."""
    kind, _ = probe.verdict([8.0, 6.0, 3.0, 0.4, 0.2, 0.1], per_round_requests=1000)
    assert kind == "plateau"


def test_constant_growth_is_a_leak():
    kind, reason = probe.verdict([2.0, 1.9, 2.1, 2.0, 2.0, 1.9], per_round_requests=1000)
    assert kind == "leak", reason


def test_per_request_budget_is_what_decides_a_decaying_series():
    """Same shape, different request size: 1.5 MB per 1000 requests is over budget, per 100000
    requests it is noise. The verdict must follow the per-request rate, not the shape."""
    decaying = [5.9, 11.1, 2.4, 1.5, 1.5, 1.3]
    assert probe.verdict(decaying, per_round_requests=1000)[0] == "leak"
    assert probe.verdict(decaying, per_round_requests=100_000)[0] == "plateau"


def test_short_runs_are_inconclusive_rather_than_guessed():
    kind, reason = probe.verdict([1.0, 1.0], per_round_requests=1000)
    assert kind == "inconclusive"
    assert "4 rounds" in reason


@pytest.mark.parametrize("size", [1000, 1])
def test_budget_boundary_is_strict(size: int):
    """Exactly at the budget (0.5 KB/request) is accepted; just over is a leak."""
    at_budget = [0.5 * size / 1024.0] * 6
    over_budget = [0.6 * size / 1024.0] * 6
    assert probe.verdict(at_budget, per_round_requests=size)[0] == "plateau"
    assert probe.verdict(over_budget, per_round_requests=size)[0] == "leak"


def test_growth_while_a_store_is_still_filling_is_not_a_leak():
    """The measurement that settled the leak investigation.

    [FACT] A 12,000-request run reported per-1000 growth
    ``[15.8, 14.1, 9.4, 2.5, 6.4, 2.5, 5.1, 4.7, -0.1, 10.9, 2.6]`` MB. The traced (Python-level)
    growth was ~1.5 MB per 1000 requests while the 10,000-record usage-analytics store was
    filling, and collapsed to 40 KiB and 75 KiB per 1000 in the last two rounds once it
    saturated. A rate rule cannot tell that apart from a leak, so the verdict must refuse to
    judge while capacity remains — the failure mode being a false "leak" on a healthy
    process, followed by a pointless hunt.
    """
    series = [15.8, 14.1, 9.4, 2.5, 6.4, 2.5, 5.1, 4.7, -0.1, 10.9, 2.6]
    kind, reason = probe.verdict(series, per_round_requests=1000, remaining_capacity_entries=2167)
    assert kind == "filling", reason
    assert "2167" in reason


def test_the_same_series_after_saturation_is_judged_by_rate():
    """The same grower, with every ceiling reached, is judged on what it retains per request."""
    # Once saturated, the measured growth fell to ~0.04 MB per 1000 requests.
    kind, _ = probe.verdict(
        [1.5, 1.4, 0.9, 0.2, 0.1, 0.04], per_round_requests=1000, remaining_capacity_entries=0
    )
    assert kind == "plateau"

    kind, reason = probe.verdict(
        [15.8, 14.1, 9.4, 2.5, 6.4, 2.5], per_round_requests=1000, remaining_capacity_entries=0
    )
    assert kind == "leak", reason
