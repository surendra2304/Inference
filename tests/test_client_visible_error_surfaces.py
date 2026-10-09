"""Every error a client can read must be one we chose to publish.

``tests/test_internal_errors_never_leak.py`` covers the routes that were fixed first. This file
covers the rest of the surface, found by driving each endpoint the way a real operator does
(``scripts/real_life_drive.py``) and reading the bodies that came back:

* ``POST /v1/friday/debate`` with no provider credentials answered **503** with the internal
  provider chain embedded in ``detail``: attempted provider/model pairs and the phrase
  ``has no configured credential``.
* ``POST /v1/forge/stream-code`` streamed ``{"error": "GEMINI_API_KEY is not configured."}``
  to the client, naming a configuration variable.
* ``POST /v1/forge/batch-generate`` returned per-item ``"error": "groq: Provider..."`` text
  produced by ``app/providers/unified_manager.py``.
* ``POST /v1/ask`` (the catch-all in ``ask_v1``) put ``f"...: {exc}"`` into ``answer`` and
  ``str(exc)`` into ``failure_state``, so a raised ``RuntimeError`` reached the caller verbatim.
* ``POST /v1/task/execute`` returned ``str(e)`` in ``error``.

The first three are reproduced with **no mocks** (the real unconfigured path). The last two
inject a marked secret, because that is the failure shape a provider library actually produces.

Each test asserts the *shape* of an honest answer (status class, a correlation id a support
engineer can grep for) and the *absence* of internal text. None of them pins a specific
sentence, so the wording can improve without breaking the contract.
"""

from __future__ import annotations

import json
import re

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

#: Deliberately full of what a leak would expose: a path, a credential, a table name.
MARKER = "s3cr3t-marker-7d4e"
SECRET_TEXT = f"upstream rejected request: api_key=sk-{MARKER} at /srv/private/db.sqlite"

#: Fragments that only appear when internal provider/configuration text leaks.
INTERNAL_FRAGMENTS = (
    "has no configured credential",
    "GEMINI_API_KEY",
    "attempted gemini",
    "attempted groq",
    "AgentCallUnavailable",
    "skipping without an attempt",
    # The provider-name prefix (``groq: ProviderUnconfiguredError``) is the contract now;
    # the raw message after it is the leak.
    "Provider 'groq' has no configured credential",
    "skipping without an attempt",
)

CORRELATION = re.compile(r"\b[a-z]+_[0-9a-f]{12}\b")  # any helper prefix: err_, provider_, ...


class Boom(RuntimeError):
    """A failure of the kind a provider library raises."""


@pytest.fixture
def client(auth):
    return AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver", headers=auth, timeout=120
    )


def _assert_no_internal_text(body: str) -> None:
    for fragment in INTERNAL_FRAGMENTS:
        assert fragment not in body, f"internal text {fragment!r} reached the client: {body[:300]}"


# -- real path, no mocks: the unconfigured provider chain -----------------------------------


async def test_debate_503_does_not_publish_the_provider_chain(client):
    """Measured before the fix: 503 detail contained 'Provider gemini has no configured credential'
    and the list of attempted provider/model pairs."""
    async with client as c:
        response = await c.post("/v1/friday/debate", json={"question": "Is the sky blue?", "no_cache": True})
    assert response.status_code == 503, "honest unavailability is 503 (see the FRIDAY outage contract)"
    detail = response.json()["detail"]
    _assert_no_internal_text(detail)
    assert CORRELATION.search(detail), f"a client must get something to quote: {detail!r}"


async def test_stream_code_error_event_does_not_name_configuration(client):
    """Measured before the fix: the SSE stream carried ``GEMINI_API_KEY is not configured.``"""
    async with client as c:
        response = await c.post("/v1/forge/stream-code", json={"filename": "a.py", "file_type": "python"})
    assert response.status_code == 200
    events = [json.loads(line[len("data: "):]) for line in response.text.splitlines() if line.startswith("data: ")]
    error_events = [event for event in events if "error" in event]
    assert error_events, "the failure must still be reported in-band"
    _assert_no_internal_text(json.dumps(error_events))
    assert error_events[-1]["done"] is True
    assert CORRELATION.search(error_events[-1]["error"])


async def test_batch_items_do_not_publish_provider_error_text(client):
    """Measured before the fix: per-item ``error`` was ``groq: Provider ... has no configured credential``.
    The batch envelope itself must still be honest about the failure count."""
    async with client as c:
        response = await c.post(
            "/v1/forge/batch-generate",
            json={"requests": [{"filename": "a.py", "file_type": "python"}, {"filename": "b.py", "file_type": "python"}]},
        )
    assert response.status_code == 200
    body = response.text
    _assert_no_internal_text(body)
    failing = [item for item in response.json()["results"] if item.get("error")]
    assert failing, "with no credentials configured, the items must say they failed"
    for item in failing:
        assert CORRELATION.search(item["error"]), f"each failed item needs a correlation id: {item}"


# -- injected: a provider library raises text containing a secret --------------------------


async def test_ask_catch_all_publishes_neither_answer_nor_failure_state_text(client, monkeypatch):
    """The catch-all in ``ask_v1`` used ``answer=f"...: {exc}"`` and ``failure_state=str(exc)``.
    Measured before the fix: both carried ``MARKER`` verbatim."""
    import app.api.v1_core_routes as v1

    async def explode(*_args, **_kwargs):
        raise Boom(SECRET_TEXT)

    monkeypatch.setattr(v1.unified_provider_manager, "execute", explode)
    async with client as c:
        response = await c.post("/v1/ask", json={"prompt": "Summarise the quarterly risk.", "mode": "fast"})
    assert response.status_code == 200, "/v1/ask answers in band (status/failure_state), by contract"
    body = response.json()
    assert MARKER not in response.text, f"secret reached the client: {response.text[:300]}"
    assert "/srv/private" not in response.text
    assert body["status"] in {"ERROR", "DEGRADED"}
    assert body["failure_state"] is None or CORRELATION.search(body["failure_state"]) or "internal" in body["failure_state"]


async def test_task_execute_error_field_does_not_echo_exception(client, monkeypatch):
    """``POST /v1/task/execute`` returned ``error=str(e)``. Measured before the fix: MARKER in body."""
    import app.api.universal_task_routes as tasks

    async def explode(*_args, **_kwargs):
        raise Boom(SECRET_TEXT)

    monkeypatch.setattr(tasks.unified_provider_manager, "execute", explode)
    async with client as c:
        response = await c.post(
            "/v1/task/execute",
            json={"action": "reason", "payload": {"prompt": "Check the disk layout."}},
        )
    assert response.status_code == 200
    assert MARKER not in response.text, response.text[:300]
    assert "/srv/private" not in response.text


async def test_failed_ask_leaves_no_secret_for_get_tasks_to_serve(client, monkeypatch):
    """The task record written by a failed ask is served back by ``GET /tasks/{id}``. Whatever
    the write path stored must already be safe, not merely scrubbed on the way out."""
    import app.api.v1_core_routes as v1

    async def explode(*_args, **_kwargs):
        raise Boom(SECRET_TEXT)

    monkeypatch.setattr(v1.unified_provider_manager, "execute", explode)
    async with client as c:
        response = await c.post("/v1/ask", json={"prompt": "Record this failed inquiry.", "mode": "fast"})
        task_id = response.json()["task_id"]
        served = await c.get(f"/tasks/{task_id}")
    assert MARKER not in served.text, served.text[:300]
    assert "/srv/private" not in served.text
