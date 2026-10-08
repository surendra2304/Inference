"""FastAPI Router for Trading Bot Advisory Consultation, A/B Testing, and Testnet Tracking."""

import asyncio
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status

from app.agents.registry import agent_registry
from app.schemas.trading_consult import (
    AIUniverseDecision,
    ExperimentConfigResponse,
    ExperimentResultsResponse,
    ExperimentStartRequest,
    ExperimentStatusResponse,
    TestnetComparisonResponse,
    TestnetPerformanceResponse,
    TradingConsultRequest,
)
from app.services.experiment_service import experiment_service
from app.services.trading_consult_service import trading_consult_service
from app.utils.errors import internal_error
from app.utils.logger import logger

router = APIRouter(prefix="/v1/trading", tags=["Trading Consultation"])

# In-memory sliding-window rate limiter: max 20 requests per bot_id per hour (3600s)
_bot_request_timestamps: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_WINDOW_SECONDS = 3600.0
RATE_LIMIT_MAX_REQUESTS = 20
# bot_id is supplied in the request body, so every distinct value would
# otherwise create a permanent dict entry - a trivial memory-exhaustion vector.
MAX_TRACKED_BOT_IDS = 10_000

# Payload size threshold: 1MB (1,048,576 bytes)
MAX_PAYLOAD_BYTES = 1024 * 1024

# Disallowed credential keywords to strictly enforce security boundaries
FORBIDDEN_CREDENTIAL_KEYS = {
    "api_key", "secret", "credential", "private_key", "api_secret",
    "password", "auth_token", "access_token", "passphrase", "secret_key"
}


def _evict_stale_bot_windows(now: float) -> None:
    """Drop idle bot windows and cap how many distinct bot_ids we track."""
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    for bot_id in [b for b, ts in _bot_request_timestamps.items() if not ts or ts[-1] <= cutoff]:
        del _bot_request_timestamps[bot_id]

    if len(_bot_request_timestamps) >= MAX_TRACKED_BOT_IDS:
        # Evict the least-recently active windows first.
        oldest_first = sorted(
            _bot_request_timestamps.items(),
            key=lambda kv: kv[1][-1] if kv[1] else 0.0,
        )
        for bot_id, _ in oldest_first[: max(1, MAX_TRACKED_BOT_IDS // 10)]:
            del _bot_request_timestamps[bot_id]


def _check_rate_limit(bot_id: str) -> None:
    """Enforces max 20 consultations per bot_id per hour."""
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    _evict_stale_bot_windows(now)
    # Clean expired timestamps
    _bot_request_timestamps[bot_id] = [ts for ts in _bot_request_timestamps[bot_id] if ts > cutoff]
    if len(_bot_request_timestamps[bot_id]) >= RATE_LIMIT_MAX_REQUESTS:
        logger.warning("Rate limit exceeded for bot_id '%s' (%d requests in 1 hour)", bot_id, len(_bot_request_timestamps[bot_id]))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for bot '{bot_id}': Maximum {RATE_LIMIT_MAX_REQUESTS} consultations per hour allowed."
        )
    _bot_request_timestamps[bot_id].append(now)


def _scan_for_forbidden_keys(obj: Any, path: str = "") -> None:
    """Recursively validates that incoming payloads contain NO exchange credentials."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            k_lower = str(k).lower()
            if any(forbidden in k_lower for forbidden in FORBIDDEN_CREDENTIAL_KEYS):
                logger.error("Security violation: Forbidden credential key '%s' found at path '%s'", k, path)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Security violation: Payload contains forbidden credential field '{k}'. Inference never accepts exchange credentials."
                )
            _scan_for_forbidden_keys(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            _scan_for_forbidden_keys(item, f"{path}[{idx}]")


def _inline_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``schema`` with every ``#/$defs/<Name>`` reference inlined.

    Pydantic hoists nested models into ``$defs`` and points at them with ``#/$defs/<Name>``.
    The walk is depth-guarded rather than trusted: a cyclic model would otherwise expand
    forever and hang module import, and a missing definition raises instead of silently
    emitting an unresolvable reference.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any, depth: int = 0) -> Any:
        if depth > 25:
            raise ValueError("schema $defs are cyclic; refusing to inline")
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.rsplit("/", 1)[-1]
                if name not in defs:
                    raise KeyError(f"unresolvable schema reference: {ref}")
                resolved = resolve(defs[name], depth + 1)
                siblings = {k: resolve(v, depth + 1) for k, v in node.items() if k != "$ref"}
                if siblings and isinstance(resolved, dict):
                    return {**resolved, **siblings}
                return resolved
            return {k: resolve(v, depth + 1) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v, depth + 1) for v in node]
        return node

    return resolve({k: v for k, v in schema.items() if k != "$defs"})


def _consult_request_body_schema() -> dict[str, Any]:
    """Publish the real request body for ``POST /consult`` in the OpenAPI document.

    The handler deliberately takes a raw ``Request`` so it can enforce a 1 MB size limit
    (413) and scan for smuggled credentials *before* validation — the security ordering
    is the point. The cost of that choice was that FastAPI generated **no requestBody at
    all** for this route: ``/openapi.json`` advertised an empty body while the service
    required ``bot_id``, ``trading_mode``, ``telemetry`` and ``consultation_reason``, so
    every spec-generated client and every reader of ``/docs`` sent a body that could not
    work. The published contract contradicted the code. Declaring the schema explicitly
    here restores it without weakening the runtime checks.
    """
    schema = TradingConsultRequest.model_json_schema()
    # Pydantic hoists nested models into ``$defs`` and points at them with
    # ``#/$defs/<Name>``. FastAPI only rewrites those pointers into
    # ``#/components/schemas/<Name>`` for models it generated itself; because this route
    # is declared with a raw ``Request``, nothing registers them in ``components``. The
    # naive ``ref_template`` override therefore produced three *dangling* ``$ref``s
    # (StrategyPerformance, TestnetContext, TradingTelemetry) which break Swagger UI and
    # any generated client. Inlining the definitions keeps the document self-contained.
    schema = _inline_defs(schema)
    return {"required": True, "content": {"application/json": {"schema": schema}}}


@router.post(
    "/consult",
    response_model=AIUniverseDecision,
    status_code=status.HTTP_200_OK,
    openapi_extra={"requestBody": _consult_request_body_schema()},
)
async def consult_trading_bot(request: Request) -> AIUniverseDecision:
    """
    Submits performance telemetry for multi-agent trading consultation.
    Enforces rate limiting, payload size validation, credential scanning, and a 180s server timeout.
    Supports A/B testing with experiment context and testnet-specific risk evaluation.
    """
    # 1. Payload size validation (max 1MB)
    body_bytes = await request.body()
    if len(body_bytes) > MAX_PAYLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Payload size ({len(body_bytes)} bytes) exceeds the 1MB limit."
        )

    # 2. Raw JSON parsing and credential inspection
    try:
        raw_json = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
    except Exception as parse_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON payload: {parse_err!s}"
        )

    _scan_for_forbidden_keys(raw_json)

    # 3. Pydantic schema validation
    try:
        req = TradingConsultRequest.model_validate(raw_json)
    except Exception as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Schema validation error: {val_err!s}"
        )

    # 4. Rate limiting per bot_id
    _check_rate_limit(req.bot_id)

    # 5. Execute consultation with 180-second server-side timeout
    try:
        decision = await asyncio.wait_for(
            trading_consult_service.consult(req),
            timeout=180.0
        )
        return decision
    except asyncio.TimeoutError:
        logger.error("Trading consultation timed out after 180.0s for bot '%s'", req.bot_id)
        # On timeout, return a NO_CHANGE decision with explanatory note
        return AIUniverseDecision(
            decision_id=str(uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="NO_CHANGE",
            confidence=0.50,
            parameter_changes=[],
            risk_assessment="Server-side consultation timeout (180s) reached during multi-agent deliberation. Existing parameters maintained safely.",
            regime_analysis="Analysis incomplete due to deliberation timeout.",
            dissent_notes="Timeout encountered during multi-agent debate stages.",
            debate_summary="Consultation orchestration reached the 180-second timeout threshold. Defaulting to NO_CHANGE safe holding pattern.",
            valid_until=(datetime.now(timezone.utc)).isoformat(),
            comparison_rationale="Consultation timed out before completing A/B comparative synthesis."
        )
    except Exception as exc:
        # The caller gets a correlation id; the details stay in the server log. Handing the
        # raw exception to the client leaked whatever the failure happened to contain —
        # measured: a raised RuntimeError carrying "password=hunter2" and an absolute path
        # came straight back in the HTTP response body. An operator can join the two through
        # the correlation id, and a client can retry safely without learning our internals.
        detail, _ = internal_error(
            logger, exc, doing_what="trading consultation", prefix="consult",
            extra=f"bot={req.bot_id}",
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail) from exc


@router.get("/consult/health", status_code=status.HTTP_200_OK)
async def trading_consult_health():
    """Returns the operational health and available specialist agents for trading consultation."""
    registered = agent_registry.list_agents()
    available_agent_roles = [a.role for a in registered]
    return {
        "status": "ok",
        "service": "trading_consultation",
        "agents_available": available_agent_roles,
        "advisory_only": True,
        "exchange_execution": False,
        "ab_testing_supported": True,
        "testnet_supported": True
    }


@router.get("/decisions/{decision_id}", response_model=AIUniverseDecision, status_code=status.HTTP_200_OK)
async def get_trading_decision(decision_id: str) -> AIUniverseDecision:
    """Retrieves a historical trading consultation decision by its UUID."""
    decision = await trading_consult_service.get_decision_by_id(decision_id)
    if not decision:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trading consultation decision '{decision_id}' not found in persistent memory."
        )
    return decision


# --- A/B Experiment Tracking Endpoints ---

@router.post("/experiment/start", response_model=ExperimentConfigResponse, status_code=status.HTTP_201_CREATED)
async def start_ab_experiment(req: ExperimentStartRequest) -> ExperimentConfigResponse:
    """
    Registers and launches a new A/B trading experiment with CONTROL and TREATMENT arms.
    """
    return experiment_service.start_experiment(req)


@router.get("/experiment/{experiment_id}/status", response_model=ExperimentStatusResponse, status_code=status.HTTP_200_OK)
async def get_ab_experiment_status(experiment_id: str) -> ExperimentStatusResponse:
    """
    Returns current experiment status, duration, active arms, and consultation counts.
    """
    exp_status = experiment_service.get_status(experiment_id)
    if not exp_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"A/B Experiment '{experiment_id}' not found."
        )
    return exp_status


@router.get("/experiment/{experiment_id}/results", response_model=ExperimentResultsResponse, status_code=status.HTTP_200_OK)
async def get_ab_experiment_results(experiment_id: str) -> ExperimentResultsResponse:
    """
    Returns aggregated comparative results and conclusion between CONTROL and TREATMENT arms.
    """
    exp_results = experiment_service.get_results(experiment_id)
    if not exp_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"A/B Experiment '{experiment_id}' not found."
        )
    return exp_results


# --- Testnet Tracking & Comparison Endpoints ---

@router.get("/testnet/performance", response_model=TestnetPerformanceResponse, status_code=status.HTTP_200_OK)
async def get_testnet_performance() -> TestnetPerformanceResponse:
    """
    Returns aggregated testnet performance metrics across historical consultations,
    comparing average win rate, profit factor, and drawdown distributions between testnet and paper modes.
    """
    return await trading_consult_service.get_testnet_performance()


@router.get("/testnet/comparison", response_model=TestnetComparisonResponse, status_code=status.HTTP_200_OK)
async def get_testnet_comparison() -> TestnetComparisonResponse:
    """
    Compares testnet execution dynamics against paper simulations and highlights strategy divergences.
    """
    return await trading_consult_service.get_testnet_comparison()
