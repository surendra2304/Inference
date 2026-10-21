"""Tests for SLA Predictor and Speculative Draft Token Verifier."""

from app.inference_runtime.contracts import Capability, ProviderEndpoint
from app.inference_runtime.sla_predictor import SlaPredictor
from app.inference_runtime.speculative_verifier import SpeculativeVerifier


def test_sla_predictor_estimates_latency():
    pred = SlaPredictor()
    # Predict for Groq gpt-oss-120b: base ~60ms, ~8.5ms/token
    lat_50 = pred.predict_latency_ms("groq", "openai/gpt-oss-120b", prompt_tokens=10, completion_tokens=50)
    lat_100 = pred.predict_latency_ms("groq", "openai/gpt-oss-120b", prompt_tokens=10, completion_tokens=100)

    assert lat_50 > 60.0
    assert lat_100 > lat_50
    assert lat_100 < 2000.0


def test_sla_predictor_recommends_fastest_endpoint():
    pred = SlaPredictor()
    endpoints = [
        ProviderEndpoint(
            provider="groq",
            model="openai/gpt-oss-120b",
            capabilities=frozenset({Capability.CHAT, Capability.STREAM}),
        ),
        ProviderEndpoint(
            provider="openrouter",
            model="meta-llama/llama-3.3-70b-instruct",
            capabilities=frozenset({Capability.CHAT, Capability.STREAM}),
        ),
        ProviderEndpoint(
            provider="gemini",
            model="gemini-flash-latest",
            capabilities=frozenset({Capability.CHAT, Capability.STREAM, Capability.VISION}),
        ),
    ]

    recommendations = pred.recommend_routing(
        target_sla_ms=600.0,
        prompt_tokens=50,
        completion_tokens=50,
        required_capabilities=frozenset({Capability.CHAT}),
        candidates=endpoints,
    )

    assert len(recommendations) == 3
    # Fastest should be ranked first
    assert recommendations[0][0].provider == "groq"
    assert recommendations[0][2] is True  # meets SLA


def test_sla_predictor_records_observations():
    pred = SlaPredictor()
    # Record fast completions
    for _ in range(5):
        pred.record_observation("groq", "test_fast_model", prompt_tokens=20, completion_tokens=50, latency_ms=150.0)

    summary = pred.get_profile_summary()
    assert "groq/test_fast_model" in summary
    assert summary["groq/test_fast_model"]["samples"] == 5
    assert summary["groq/test_fast_model"]["success_rate"] == 1.0


def test_speculative_verifier_perfect_match():
    verifier = SpeculativeVerifier()
    text = "The quick brown fox jumps over the lazy dog"
    res = verifier.verify_completion(
        draft_text=text,
        target_text=text,
        draft_latency_ms=25.0,
        target_latency_ms=120.0,
        draft_model="small",
        target_model="large",
    )

    assert res.acceptance_rate == 1.0
    assert res.accepted_tokens_count == res.draft_tokens_count
    assert res.mismatch_index is None
    assert res.effective_speedup > 1.0
    assert res.is_accelerated is True


def test_speculative_verifier_partial_mismatch():
    verifier = SpeculativeVerifier()
    draft = "The quick brown fox leaps into the pond"
    target = "The quick brown fox jumps over the lazy dog"

    res = verifier.verify_completion(
        draft_text=draft,
        target_text=target,
        draft_latency_ms=30.0,
        target_latency_ms=100.0,
        draft_model="small",
        target_model="large",
    )

    # First 4 tokens match ("The", "quick", "brown", "fox")
    assert res.accepted_tokens_count == 4
    assert res.mismatch_index == 4
    assert 0.0 < res.acceptance_rate < 1.0


def test_speculative_verifier_empty_draft():
    verifier = SpeculativeVerifier()
    res = verifier.verify_completion(
        draft_text="",
        target_text="Some target text",
        draft_latency_ms=5.0,
        target_latency_ms=50.0,
    )
    assert res.acceptance_rate == 0.0
    assert res.draft_tokens_count == 0
    assert res.is_accelerated is False


def test_speculative_verifier_telemetry():
    verifier = SpeculativeVerifier()
    verifier.verify_completion("Hello world", "Hello world", 10.0, 50.0, "m1", "m2")
    verifier.verify_completion("Hello world", "Hello there", 10.0, 50.0, "m1", "m2")

    telem = verifier.get_telemetry()
    assert "m1->m2" in telem
    assert telem["m1->m2"]["runs"] == 2
    assert telem["m1->m2"]["avg_acceptance_rate"] > 0.0
