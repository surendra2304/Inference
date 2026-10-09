"""FORGE must not present unverified text as code, and must not serve invented quality numbers.

Found by the operator tour (``scripts/operator_tour.py``), which asked for a Python function and
read what came back. [FACT] measured before the fix, on ``POST /v1/forge/generate-code``:

* the ``code`` field held the model's *prose* ("[specialist] round analysis ... Generate ...")
  with ``confidence: 0.92`` and ``generation_path: "agent"``. The confidence was a constant
  (``app/services/code_generation.py``: ``0.92 if gen_path == "agent" else 0.55``); nothing
  checked the output;
* the syntax validator the module already had (``QualityAssuranceService.evaluate_code_syntax``)
  was never called on generation output;
* ``GET /v1/admin/analytics/quality`` returned ``overall_output_quality_score: 96.4``,
  ``syntactic_validity_rate_pct: 99.2`` and a calibration table saying ``WELL_CALIBRATED``.
  Those are literals in the source, not measurements;
* the fence extraction returned the *prose* when a fenced block followed a sentence, so a
  correct answer in the usual shape was reported as the wrong text.

Each test below fails against the pre-fix code; the docstrings say how.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services import code_generation as cg
from app.services.code_generation import CodeGenerationRequest, code_generation_service
from app.services.quality_assurance import QualityAssuranceService, quality_assurance_service


def extract_code(text: str) -> str:
    """Looked up at call time so a missing helper fails one test, not the whole module."""
    return cg.extract_code(text)


def _provider_reply(content: str, status: str = "success") -> SimpleNamespace:
    return SimpleNamespace(
        content=content, status=status, degraded=False, error=None,
        token_usage={"total_tokens": 42},
    )


@pytest.fixture
def scripted_provider(monkeypatch):
    """Replace the provider call with a scripted reply; the cache is cleared per test."""
    replies: list[str] = []
    calls: list[str] = []

    async def execute(exec_req, *args, **kwargs):  # noqa: ARG001 - signature of the real method
        calls.append(exec_req.prompt)
        return _provider_reply(replies.pop(0) if replies else "")

    monkeypatch.setattr(cg.unified_provider_manager, "execute", execute)
    monkeypatch.setattr(code_generation_service, "_cache", {})
    return SimpleNamespace(replies=replies, calls=calls)


# -- extraction -------------------------------------------------------------------------


def test_fenced_block_after_prose_is_the_code():
    """Pre-fix: returned 'Here it is:' (the prose), because the closing fence was taken as the
    end of the block that the opening fence started."""
    assert extract_code("Here it is:\n```python\nx = 1\n```\nDone.") == "x = 1"


def test_fence_on_the_first_line_still_works():
    assert extract_code("```python\ny = 2\n```") == "y = 2"


def test_unfenced_text_is_returned_as_is():
    assert extract_code("z = 3") == "z = 3"


def test_unclosed_fence_from_a_token_ceiling_is_body_not_the_fence_marker():
    """A reply cut off mid-block should fail the syntax check honestly, not pass with a stray
    fence line in front of it."""
    assert extract_code("```python\nw = 4\nv = (") == "w = 4\nv = ("


# -- the verdict decides what is returned ------------------------------------------------


@pytest.mark.asyncio
async def test_prose_is_never_returned_as_python_code(scripted_provider):
    """Pre-fix: code == prose, generation_path == 'agent', confidence == 0.92."""
    scripted_provider.replies.append("Here is a plan: first parse the date, then convert it to UTC.")
    response = await code_generation_service.generate_code(
        CodeGenerationRequest(file_type="python", filename="prose_only.py"))

    assert response.code == ""
    assert response.generation_path == "invalid_output"
    assert response.confidence == 0.0
    assert response.error and "did not parse as python" in response.error


@pytest.mark.asyncio
async def test_invalid_output_is_not_cached_so_a_retry_can_succeed(scripted_provider):
    scripted_provider.replies.extend(["this is not code (", "x = 5\n"])
    req = CodeGenerationRequest(file_type="python", filename="retry_me.py")

    first = await code_generation_service.generate_code(req)
    second = await code_generation_service.generate_code(req)

    assert first.generation_path == "invalid_output"
    assert len(scripted_provider.calls) == 2, "the invalid result must not be served from cache"
    assert second.generation_path == "agent"
    assert second.code == "x = 5"


@pytest.mark.asyncio
async def test_valid_python_gets_the_parser_verified_confidence(scripted_provider):
    scripted_provider.replies.append("```python\ndef parse(s: str) -> int:\n    return int(s)\n```")
    response = await code_generation_service.generate_code(
        CodeGenerationRequest(file_type="python", filename="verified.py"))

    assert response.generation_path == "agent"
    assert response.confidence == 0.92
    assert "def parse" in response.code


@pytest.mark.asyncio
async def test_unverifiable_type_is_reported_as_unverified_not_as_verified(scripted_provider):
    """Markdown has no validator here. It must not carry the verified confidence."""
    scripted_provider.replies.append("# Title\n\nSome notes.")
    response = await code_generation_service.generate_code(
        CodeGenerationRequest(file_type="markdown", filename="notes.md"))

    assert response.generation_path == "agent"
    assert response.confidence == 0.55


# -- the quality report carries only measurements ---------------------------------------


def test_quality_report_reports_nothing_it_did_not_measure():
    """Pre-fix: returned 96.4 / 99.2 / WELL_CALIBRATED with no observations behind them."""
    service = QualityAssuranceService()
    report = service.get_quality_report()

    assert report["syntactic_validity"]["status"] == "not_measured"
    assert report["syntactic_validity"]["rate_pct"] is None
    assert report["overall_output_quality_score"]["value"] is None
    assert report["confidence_calibration"]["status"] == "not_measured"
    assert report["agent_quality_rankings"]["rankings"] is None
    assert "WELL_CALIBRATED" not in str(report)


def test_quality_report_counts_real_checks_and_says_how_many():
    service = QualityAssuranceService()
    service.check_output("x = 1", "python")
    service.check_output("x = (", "python")
    service.check_output("{}", "json")
    service.check_output("# not verifiable here", "markdown")

    report = service.get_quality_report()
    syntax = report["syntactic_validity"]
    assert syntax["status"] == "measured"
    assert syntax["sample_size"] == 3, "the unverifiable markdown check must not be counted"
    assert syntax["rate_pct"] == pytest.approx(66.7, abs=0.1)
    assert syntax["by_language"]["python"] == {"checked": 2, "valid": 1, "syntactic_validity_rate_pct": 50.0}


def test_unverifiable_language_does_not_read_as_valid():
    """Pre-fix: evaluate_code_syntax returned is_valid=True for every non-Python language."""
    assert QualityAssuranceService().evaluate_code_syntax("<div>", "html")["is_valid"] is None


# -- over HTTP, the way an operator sees it ---------------------------------------------


@pytest.mark.asyncio
async def test_generate_code_route_does_not_return_prose_as_code(scripted_provider, auth):
    scripted_provider.replies.append("Sure! The module would parse dates with datetime.fromisoformat.")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver",
                           headers=auth, timeout=60) as client:
        response = await client.post("/v1/forge/generate-code", json={
            "file_type": "python", "filename": "http_prose.py",
        })

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == ""
    assert body["generation_path"] == "invalid_output"
    assert body["confidence"] == 0.0


def test_quality_counters_are_shared_with_the_report():
    before = quality_assurance_service.get_quality_report()["syntactic_validity"]["sample_size"]
    quality_assurance_service.check_output("y = 9", "python")
    after = quality_assurance_service.get_quality_report()["syntactic_validity"]["sample_size"]
    assert after == before + 1
