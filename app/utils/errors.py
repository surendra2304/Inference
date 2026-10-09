"""One place that decides what an internal failure is allowed to tell a client.

Three routes published the raw exception text of whatever broke inside them::

    detail=f"Task orchestration failed: {exc!s}"
    detail=f"Debate orchestration failed: {exc!s}"
    detail=f"FRIDAY task orchestration failed: {exc!s}"

Measured impact, on the trading route: a raised ``RuntimeError`` carrying
``"secret path /srv/private/db.sqlite: password=hunter2"`` was returned verbatim in the HTTP
response body of a 500. Provider libraries are worse than that — a 401 from an upstream API
commonly quotes the key it rejected, and ``AgentPanelUnavailable`` builds its message from
per-agent provider failures (``app/agents/debate.py:739``), so a provider that echoes a key
would publish it in a 503 to every caller.

The rule here is simple and is applied by construction rather than by discipline:

* the client receives a *correlation id* and a sentence saying what the service was doing;
* the server log receives the correlation id, the exception type and the full text, with
  credential-looking substrings scrubbed (defence in depth: the log is the safe place for
  detail, not a place to dump a key either);
* nothing else is passed through.

The helpers return ``(detail, correlation_id)`` so callers cannot accidentally publish the
exception while intending to log it.
"""

from __future__ import annotations

import logging
import re
from uuid import uuid4

from app.security.prompt_isolation import scrub_credentials


def correlation_id(prefix: str = "err") -> str:
    """A short id an operator can grep for in the logs and a client can quote."""
    return f"{prefix}_{uuid4().hex[:12]}"


def internal_error(
    logger: logging.Logger,
    exc: BaseException,
    *,
    doing_what: str,
    prefix: str = "err",
    extra: str | None = None,
) -> tuple[str, str]:
    """Log an internal failure safely and return ``(client_detail, correlation_id)``.

    ``doing_what`` is a short, honest description of the operation ("trading consultation",
    "debate orchestration"); it is the only part of the failure the client sees, because it
    describes *our* work rather than the internals that broke.
    """
    reference = correlation_id(prefix)
    scrubbed = scrub_credentials(f"{exc!s}")
    logger.error(
        "%s failed [%s]%s: %s: %s",
        doing_what, reference, f" ({extra})" if extra else "",
        type(exc).__name__, scrubbed, exc_info=True,
    )
    detail = (
        f"{doing_what.capitalize()} could not be completed due to an internal error. "
        f"Quote correlation id {reference} when reporting this; the details are in the "
        "server log."
    )
    return detail, reference


def unavailable_detail(exc: BaseException, *, failures: list[str] | None = None,
                       prefix: str = "unavailable") -> str:
    """Build a 503 detail that names *which* work is unavailable without provider internals.

    ``AgentPanelUnavailable`` carries one string per failed specialist that includes the raw
    provider exception (``app/agents/debate.py:739`` — ``f"{agent.id}: {outcome}"``). This
    trims each line to the specialist id and the exception *type*, keeping the reasons
    meaningful (an operator learns which agents were dark and how they failed) while dropping
    anything a provider chose to quote back.
    """
    trimmed: list[str] = []
    for line in failures or []:
        agent, _, detail = line.partition(": ")
        agent = agent.strip() or "specialist"
        # The useful, publishable signal is *what kind* of failure it was. Prefer the
        # exception class name when the text carries one; otherwise say only that the call
        # failed, because anything else is provider-supplied free text.
        # ``AgentCallUnavailable`` and ``RuntimeError`` are both failure *kinds* a caller can
        # act on; matching only ``*Error`` missed the former and reported "call failed".
        match = re.search(
            r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception|Unavailable|Timeout|Refused)\b", detail
        )
        if match:
            trimmed.append(f"{agent} ({match.group(0)})")
        else:
            trimmed.append(f"{agent} (call failed)")
    if not trimmed:
        return (
            "no specialist produced output, and no per-specialist reason was recorded "
            f"(correlation id {correlation_id(prefix)})"
        )
    return (
        "no specialist produced output; per-specialist failure kinds: "
        + "; ".join(trimmed[:12])
        + f" (correlation id {correlation_id(prefix)})"
    )


def provider_failure_reason(logger: logging.Logger, provider: str, exc: BaseException) -> str:
    """Short, publishable reason for one provider call that failed.

    ``UnifiedExecutionResponse.error`` used to be ``f"{provider}: {type(exc).__name__}: {exc}"``.
    That string is copied into batch items, task results and code-generation responses, so
    whatever a provider library put in its exception message (a rejected key, a quoted
    request, a configuration variable name) reached every client. [FACT] measured on
    ``POST /v1/forge/batch-generate`` with no credentials configured: each item carried
    ``groq: ProviderUnconfiguredError: Provider 'groq' has no configured credential; ...``.

    The client keeps what it can act on, the provider and the failure kind, plus a correlation
    id. The full scrubbed exception goes to the server log under the same id.
    """
    reference = correlation_id("provider")
    logger.warning(
        "provider %s call failed [%s]: %s: %s",
        provider, reference, type(exc).__name__, scrub_credentials(f"{exc!s}"),
    )
    return f"{provider}: {type(exc).__name__} (correlation id {reference})"
