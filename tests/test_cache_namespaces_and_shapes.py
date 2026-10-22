"""The shared response cache must not mix producers, contexts, or shapes.

Found by the sensitivity probe (``scripts/sensitivity_probe.py``): ``POST /v1/friday/ask`` and
``/v1/friday/debate`` returned HTTP 500 on both of two different inputs. [FACT] measured in the
agent's log: ``FridayResponse.model_validate`` was given a tuple ``(content, {provider, model,
tokens})``, which the provider layer had written under the *global* cache key that friday also
reads. Reading the code (``app/performance_cache.py``) explained the rest:

* the global and normalized keys ignored the mode, the caller and the producing endpoint, so any
  reader could be handed any writer's value (a 500, or a cross-caller answer);
* friday's cache key ignored ``context_data``, so the same question with a different private
  context was answered from the first context's result.

Each test names the behaviour it guards and the old behaviour it would have caught.
"""

from __future__ import annotations

import pytest

import app.performance_cache as performance_cache_module
from app.api import friday_routes
from app.performance_cache import perf_cache


def as_answer_pair(value):
    """Looked up at call time so a missing helper fails one test, not the module at import."""
    return performance_cache_module.as_answer_pair(value)

Q = "Compare Postgres and SQLite for the reporting database"


@pytest.fixture(autouse=True)
def _fresh_cache():
    """Each test starts from an empty cache so one test's writes cannot serve another."""
    perf_cache.clear()
    yield
    perf_cache.clear()


def test_a_value_from_another_producer_is_never_returned():
    """Pre-fix: friday read a provider tuple from the shared key. The provider wrote under
    namespace provider.unified; friday asks under friday.ask and must get nothing."""
    perf_cache.set_query(Q, mode="unified_x", value=("provider content", {"provider": "groq"}),
                         caller_id="role", namespace="provider.unified")

    assert perf_cache.get_query(Q, mode="auto", caller_id="op", namespace="friday.ask") is None
    assert perf_cache.get_query(Q, mode="auto", caller_id="op", namespace="instant.ask") is None


def test_the_same_namespace_still_serves_its_own_value():
    """Namespacing must not disable caching: a value is served to its own producer's readers."""
    value = {"answer": "x"}
    perf_cache.set_query(Q, mode="auto", value=value, caller_id="op", namespace="instant.ask")

    assert perf_cache.get_query(Q, mode="auto", caller_id="op", namespace="instant.ask") == value


def test_the_normalized_and_shared_keys_are_namespaced():
    """Pre-fix: the normalized key (punctuation-stripped) and the global key were unscoped, so a
    punctuation-only variant of the question found another endpoint's value."""
    perf_cache.set_query(Q, mode="auto", value=("a", {}), caller_id="op", namespace="agent.assist")

    variant = Q.replace(" ", "  ") + "?"
    assert perf_cache.get_query(variant, mode="auto", caller_id="op", namespace="friday.ask") is None
    assert perf_cache.get_query(variant, mode="auto", caller_id="op", namespace="agent.assist") == ("a", {})


def test_different_private_context_is_a_different_question():
    """Pre-fix: friday's key ignored context_data, so context A's answer served context B."""
    ctx_a = friday_routes._context_scope({"tenant_plan": "enterprise"})
    ctx_b = friday_routes._context_scope({"tenant_plan": "free"})
    perf_cache.set_query(Q, mode="auto", value={"answer": "for enterprise"}, caller_id="op",
                         extra=ctx_a, namespace=friday_routes.FRIDAY_ASK_NS)

    assert perf_cache.get_query(Q, mode="auto", caller_id="op", extra=ctx_b,
                                namespace=friday_routes.FRIDAY_ASK_NS) is None
    assert perf_cache.get_query(Q, mode="auto", caller_id="op", extra=ctx_a,
                                namespace=friday_routes.FRIDAY_ASK_NS) == {"answer": "for enterprise"}


def test_no_context_and_empty_context_share_a_key():
    """An absent context is the same as an empty one; they must not fragment the cache."""
    assert friday_routes._context_scope(None) == friday_routes._context_scope({}) == {}


def test_namespace_is_required_and_cannot_be_blank():
    with pytest.raises(TypeError):
        perf_cache.get_query(Q)  # type: ignore[call-arg]  # namespace is keyword-only and required
    with pytest.raises(ValueError):
        perf_cache.set_query(Q, mode="auto", value=1, namespace="   ")


@pytest.mark.parametrize("value,expected", [
    (("content", {"provider": "x"}), ("content", {"provider": "x"})),
    ({"answer": "dict"}, None),
    ("a string", None),
    (("only one",), None),
    ((1, {}), None),
    (("content", "not a dict"), None),
    (None, None),
])
def test_only_a_content_and_meta_pair_is_an_answer(value, expected):
    """Pre-fix: readers unpacked whatever was stored, so a dict or a 1-tuple raised in the request."""
    assert as_answer_pair(value) == expected


def test_friday_treats_a_foreign_shape_as_a_miss_not_an_error():
    """Defence in depth: even if a wrong shape is stored, the cache hit path must not 500."""
    assert friday_routes._as_friday_response(("content", {"provider": "groq"})) is None
    assert friday_routes._as_friday_response({"answer": "dict from instant"}) is None
    assert friday_routes._as_friday_response(None) is None


@pytest.mark.asyncio
async def test_friday_ask_is_not_500_when_another_endpoint_wrote_the_question(auth):
    """The end-to-end form of the defect: another producer has cached this exact question. Friday
    must answer (from the model, or degraded), never 500, and never return the foreign value."""
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    perf_cache.set_query(Q, mode="unified_general", value=("PROVIDER-TUPLE-CONTENT", {"provider": "groq"}),
                         caller_id="general", namespace="provider.unified")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver",
                           headers=auth, timeout=60) as client:
        response = await client.post("/v1/friday/ask", json={"question": Q, "caller_id": "op"})

    assert response.status_code != 500, response.text
    assert "PROVIDER-TUPLE-CONTENT" not in response.text
