"""FORGE's debugger must not invent a diagnosis, a fix strategy, or a confidence.

Before the fix (``app/services/debugging.py``), the response's ``root_cause`` was
``"Exception '<error>' identified in code execution path."``, ``fix_strategy`` was the fixed
sentence "Apply type validation, null-safety check, or exception handling patch.", ``confidence``
was 0.89 regardless of what the model returned, and ``patch_code`` was the model's raw reply.
Each test names the behaviour it guards.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

import app.services.debugging as debugging_module
from app.main import app
from app.providers import unified_manager
from app.services.debugging import DebugRequest, debugging_service

TEMPLATE_ROOT = "identified in code execution path"
TEMPLATE_FIX = "Apply type validation, null-safety check, or exception handling patch"
CONTEXT = "rate = plans[plan_code]['discount']\n"
REQ = DebugRequest(error="KeyError: 'enterprise'", traceback="KeyError: 'enterprise'", code_context=CONTEXT)


@pytest.fixture
def model_reply(monkeypatch):
    state = {"content": "Some prose about the error.", "degraded": False}

    async def execute(exec_req, *args, **kwargs):  # noqa: ARG001
        return SimpleNamespace(content=state["content"], degraded=state["degraded"], error=None, token_usage={})

    monkeypatch.setattr(unified_manager.unified_provider_manager, "execute", execute)
    return state


def _json(root: str = "plans has no 'enterprise' key", fix: str = "use plans.get with a default",
          patch: str | None = None) -> str:
    import json
    return json.dumps({"root_cause": root, "fix_strategy": fix, "patch_code": patch})


@pytest.mark.asyncio
async def test_prose_reply_is_reported_as_not_diagnosed_not_as_a_template(model_reply):
    model_reply["content"] = "The error is probably a missing key."
    response = await debugging_service.diagnose_and_fix(REQ)

    assert response.diagnosis_status == "unparsed"
    assert response.confidence == 0.0
    assert response.patch_code is None
    assert response.patch_check == "not_provided"
    assert "Not diagnosed" in response.root_cause
    assert TEMPLATE_ROOT not in response.root_cause
    assert TEMPLATE_FIX not in response.fix_strategy
    assert response.fix_strategy.startswith("None:")


@pytest.mark.asyncio
async def test_unavailable_model_is_reported_as_unavailable(model_reply):
    model_reply["degraded"] = True
    response = await debugging_service.diagnose_and_fix(REQ)

    assert response.diagnosis_status == "unavailable"
    assert response.confidence == 0.0
    assert "no model answer was available" in response.root_cause


@pytest.mark.asyncio
async def test_validated_model_diagnosis_is_returned_as_the_model_wrote_it(model_reply):
    model_reply["content"] = _json()
    response = await debugging_service.diagnose_and_fix(REQ)

    assert response.diagnosis_status == "diagnosed"
    assert response.root_cause == "plans has no 'enterprise' key"
    assert response.fix_strategy == "use plans.get with a default"
    assert response.confidence == debugging_module.DIAGNOSED_CONFIDENCE


@pytest.mark.asyncio
async def test_fenced_json_diagnosis_is_accepted(model_reply):
    model_reply["content"] = "```json\n" + _json() + "\n```"
    response = await debugging_service.diagnose_and_fix(REQ)
    assert response.diagnosis_status == "diagnosed"


@pytest.mark.asyncio
async def test_json_missing_a_required_field_is_not_a_diagnosis(model_reply):
    model_reply["content"] = '{"root_cause": "x"}'
    response = await debugging_service.diagnose_and_fix(REQ)
    assert response.diagnosis_status == "unparsed"


@pytest.mark.asyncio
async def test_python_patch_that_parses_is_returned_and_marked_parses(model_reply):
    model_reply["content"] = _json(patch="rate = plans.get(plan_code, {}).get('discount', 0)\n")
    response = await debugging_service.diagnose_and_fix(REQ)

    assert response.patch_check == "parses"
    assert response.patch_code and "plans.get" in response.patch_code


@pytest.mark.asyncio
async def test_python_patch_that_does_not_parse_is_withheld(model_reply):
    model_reply["content"] = _json(patch="rate = plans[plan_code\n")
    response = await debugging_service.diagnose_and_fix(REQ)

    assert response.diagnosis_status == "diagnosed"
    assert response.patch_check == "withheld"
    assert response.patch_code is None


@pytest.mark.asyncio
async def test_patch_for_non_python_context_is_passed_through_but_marked_not_checked(model_reply, monkeypatch):
    js_req = DebugRequest(error="TypeError", code_context="const r = plans[k].discount;")
    model_reply["content"] = _json(patch="const r = (plans[k] || {}).discount || 0;")
    response = await debugging_service.diagnose_and_fix(js_req)

    assert response.patch_check == "not_checked"
    assert response.patch_code == "const r = (plans[k] || {}).discount || 0;"


@pytest.mark.asyncio
async def test_null_patch_means_no_change_is_proposed(model_reply):
    model_reply["content"] = _json(patch=None)
    response = await debugging_service.diagnose_and_fix(REQ)

    assert response.diagnosis_status == "diagnosed"
    assert response.patch_check == "not_provided"
    assert response.patch_code is None


@pytest.mark.asyncio
async def test_route_never_returns_the_fixed_template(model_reply, auth):
    model_reply["content"] = "No idea."
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver",
                           headers=auth, timeout=60) as client:
        response = await client.post("/v1/forge/debug", json={
            "error": "KeyError: 'enterprise'", "traceback": "", "code_context": CONTEXT,
        })

    assert response.status_code == 200
    body = response.json()
    assert TEMPLATE_ROOT not in body["root_cause"]
    assert TEMPLATE_FIX not in body["fix_strategy"]
    assert body["confidence"] == 0.0
    assert body["diagnosis_status"] == "unparsed"
