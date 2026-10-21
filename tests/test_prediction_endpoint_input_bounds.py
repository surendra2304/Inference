"""Input contract tests for the prediction endpoint.

The endpoint uses caller-provided market data when both price and return observations
are present. Invalid or incomplete provided values must not silently become fixtures.
"""

import pytest
from fastapi.testclient import TestClient


def _client():
    return TestClient(__import__("app.main", fromlist=["app"]).app)


@pytest.mark.parametrize("price", ["nan", "inf", "-inf", "0", "-1"])
def test_prediction_endpoint_rejects_non_finite_or_non_positive_price(auth, price):
    response = _client().get(
        "/v1/predict/btc",
        headers=auth,
        params={"current_price": price, "recent_returns": "0.01,0.02"},
    )

    assert response.status_code == 422, response.text
    assert "current_price" in response.json()["detail"]


@pytest.mark.parametrize(
    "returns",
    [
        "",
        " ",
        ",",
        "0.01,,0.02",
        "0.01, ",
        "nan,0.02",
        "inf,-0.01",
        "-inf,0.01",
        "1e999,0.01",
        "0.01,not-a-number",
    ],
)
def test_prediction_endpoint_rejects_empty_malformed_or_non_finite_returns(auth, returns):
    response = _client().get(
        "/v1/predict/btc",
        headers=auth,
        params={"current_price": "65000", "recent_returns": returns},
    )

    assert response.status_code == 422, response.text
    assert "recent_returns" in response.json()["detail"]


def test_prediction_endpoint_rejects_oversized_raw_return_string(auth):
    oversized = "0.001," * 1400
    response = _client().get(
        "/v1/predict/btc",
        headers=auth,
        params={"current_price": "65000", "recent_returns": oversized},
    )

    assert response.status_code == 422, response.text
    assert "8192 characters" in response.json()["detail"]


def test_prediction_endpoint_bounds_return_observation_count(auth):
    too_many_returns = ",".join("0.001" for _ in range(257))
    response = _client().get(
        "/v1/predict/btc",
        headers=auth,
        params={"current_price": "65000", "recent_returns": too_many_returns},
    )

    assert response.status_code == 422, response.text
    assert "256" in response.json()["detail"]


def test_prediction_endpoint_accepts_maximum_return_observation_count(auth):
    returns = ",".join("0.001" for _ in range(256))
    response = _client().get(
        "/v1/predict/btc",
        headers=auth,
        params={"current_price": "65000", "recent_returns": returns},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["inputs_simulated"] is False
    assert body["evidence_class"] == "caller_supplied_price_and_returns_with_synthetic_alt_data"
    assert body["unified_confidence"] is None


@pytest.mark.parametrize(
    ("params", "input_source", "synthetic_leg", "observed_leg"),
    [
        (
            {"current_price": "65000"},
            "caller_supplied_price_with_placeholder_returns",
            "return_series (no recent_returns supplied)",
            "price_series (no current_price supplied)",
        ),
        (
            {"recent_returns": "0.01,0.02"},
            "placeholder_price_with_caller_supplied_returns",
            "price_series (no current_price supplied)",
            "return_series (no recent_returns supplied)",
        ),
    ],
)
def test_prediction_endpoint_labels_partial_inputs_as_simulated(
    auth, params, input_source, synthetic_leg, observed_leg
):
    response = _client().get("/v1/predict/btc", headers=auth, params=params)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["inputs_simulated"] is True
    assert body["unified_confidence"] is None
    assert body["evidence_class"] == "partially_synthetic_inputs"
    assert body["input_source"] == input_source
    assert synthetic_leg in body["synthetic_legs"]
    assert observed_leg not in body["synthetic_legs"]
