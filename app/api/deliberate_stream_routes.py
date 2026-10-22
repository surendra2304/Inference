"""Real-Time Duplex Deliberation & Streaming Gateway for FRIDAY Universe.

Provides:
1. WebSocket `/v1/deliberate/ws`: Bi-directional multi-agent deliberation sessions
   between FRIDAY (orchestrator), Stratex (strategy), Cortex (context), and peer agents.
2. WebSocket `/v1/stream`: Fast bi-directional token streaming.
3. HTTP POST `/v1/stream`: Standardized SSE streaming endpoint.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.security import _configured_keys, require_inference_api_key
from app.inference_runtime.prompt_guard import inspect as inspect_prompt
from app.memory.base import DeliberationRecord
from app.memory.postgres import postgres_memory
from app.providers.base import ProviderMessage, ProviderRequest
from app.providers.gateway import model_gateway
from app.security.prompt_isolation import scrub_credentials
from app.utils.logger import logger

deliberate_stream_router = APIRouter(tags=["Real-Time Duplex Deliberation & Streaming Gateway"])


def _authenticate_ws_key(key: str | None) -> bool:
    """Validates presented WebSocket API key in constant time."""
    if settings.INSECURE_DEV_AUTH:
        return True
    valid = _configured_keys()
    if not valid or not key:
        return False
    return any(hmac.compare_digest(key.strip(), k.strip()) for k in valid)


class DeliberationSessionManager:
    """In-memory coordinator for real-time multi-agent deliberation sessions."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # session_id -> dict of websocket: agent_id
        self._sessions: dict[str, dict[WebSocket, str]] = {}
        # session_id -> list of turn records
        self._history: dict[str, list[dict[str, Any]]] = {}

    async def connect(self, session_id: str, agent_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {}
                self._history[session_id] = []
            self._sessions[session_id][websocket] = agent_id
        logger.info(f"[WS_DELIBERATE] Agent '{agent_id}' joined deliberation session '{session_id}'")

    async def disconnect(self, session_id: str, websocket: WebSocket) -> str | None:
        agent_id = None
        async with self._lock:
            if session_id in self._sessions and websocket in self._sessions[session_id]:
                agent_id = self._sessions[session_id].pop(websocket, None)
                if not self._sessions[session_id]:
                    # Clean up empty sessions
                    del self._sessions[session_id]
        if agent_id:
            logger.info(f"[WS_DELIBERATE] Agent '{agent_id}' disconnected from session '{session_id}'")
        return agent_id

    async def broadcast(self, session_id: str, message: dict[str, Any], exclude: WebSocket | None = None) -> None:
        payload = json.dumps(message)
        async with self._lock:
            recipients = list(self._sessions.get(session_id, {}).keys())

        for ws in recipients:
            if ws != exclude:
                try:
                    await ws.send_text(payload)
                except Exception as exc:
                    logger.debug(f"[WS_DELIBERATE] Broadcast send failed: {exc}")

    async def record_turn(self, session_id: str, turn_data: dict[str, Any]) -> None:
        async with self._lock:
            if session_id not in self._history:
                self._history[session_id] = []
            self._history[session_id].append(turn_data)
            # Bound session history to prevent unbounded memory growth
            if len(self._history[session_id]) > 50:
                self._history[session_id].pop(0)

    async def get_history(self, session_id: str) -> list[dict[str, Any]]:
        async with self._lock:
            return list(self._history.get(session_id, []))


deliberation_manager = DeliberationSessionManager()


@deliberate_stream_router.websocket("/v1/deliberate/ws")
async def websocket_deliberation_endpoint(websocket: WebSocket) -> None:
    """Bi-directional multi-agent deliberation WebSocket gateway."""
    await websocket.accept()

    # 1. Authentication check
    auth_header = (
        websocket.headers.get("x-inference-api-key")
        or websocket.headers.get("x-api-key")
        or websocket.query_params.get("api_key")
    )

    current_session: str | None = None
    current_agent: str = "unknown_agent"

    try:
        # Await initial handshake / connect frame if not authenticated via headers
        first_raw = await asyncio.wait_for(websocket.receive_text(), timeout=15.0)
        first_msg = json.loads(first_raw)

        if not auth_header:
            auth_header = first_msg.get("api_key")

        if not _authenticate_ws_key(auth_header):
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": "UNAUTHORIZED",
                "message": "Invalid or missing integration key for deliberation gateway",
            }))
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        current_session = str(first_msg.get("session_id") or f"delib_{uuid.uuid4().hex[:8]}")
        current_agent = str(first_msg.get("agent_id") or first_msg.get("caller") or "friday").lower()
        role = str(first_msg.get("role") or "participant")

        await deliberation_manager.connect(current_session, current_agent, websocket)
        await websocket.send_text(json.dumps({
            "type": "connected",
            "session_id": current_session,
            "agent_id": current_agent,
            "role": role,
            "status": "READY",
            "message": f"Connected to deliberation session {current_session}",
        }))

        # Notify peers
        await deliberation_manager.broadcast(
            current_session,
            {
                "type": "peer_joined",
                "session_id": current_session,
                "agent_id": current_agent,
                "role": role,
                "timestamp": time.time(),
            },
            exclude=websocket,
        )

        # 2. Main event loop
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "deliberate_turn").lower()

            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))
                continue

            elif msg_type == "get_history":
                history = await deliberation_manager.get_history(current_session)
                await websocket.send_text(json.dumps({
                    "type": "history_report",
                    "session_id": current_session,
                    "turns": history,
                    "count": len(history),
                }))
                continue

            elif msg_type == "deliberate_turn":
                prompt = msg.get("content") or msg.get("prompt") or ""
                turn_id = msg.get("turn_id") or f"turn_{uuid.uuid4().hex[:8]}"

                # Guard incoming content
                guard = inspect_prompt(prompt)
                if not guard.allowed:
                    await websocket.send_text(json.dumps({
                        "type": "turn_rejected",
                        "turn_id": turn_id,
                        "reason": f"Prompt flagged by security barrier ({guard.risk_level})",
                    }))
                    continue

                clean_text = guard.sanitized
                # Stream synthesis back to sender and broadcast chunks
                prov_req = ProviderRequest(
                    messages=[ProviderMessage(role="user", content=clean_text)],
                    system_instruction=(
                        f"You are the central deliberation facilitator synthesizing input from '{current_agent}' "
                        f"within a multi-agent council (FRIDAY, Stratex, Cortex). Provide sharp, structured deliberation points."
                    ),
                    model="openai/gpt-oss-120b",
                    temperature=0.3,
                    max_tokens=min(int(msg.get("max_tokens") or 1024), 4096),
                )

                t0 = time.time()
                collected_chunks = []
                # Streaming token loop with fallback
                try:
                    async for chunk in model_gateway.stream("groq", prov_req, stage_name="deliberate_turn"):
                        collected_chunks.append(chunk)
                        chunk_payload = {
                            "type": "chunk",
                            "turn_id": turn_id,
                            "sender": current_agent,
                            "token": chunk,
                        }
                        await websocket.send_text(json.dumps(chunk_payload))
                except Exception as stream_err:
                    logger.warning(f"[WS_DELIBERATE] Groq stream failed, falling back to openrouter: {stream_err}")
                    prov_req.model = "meta-llama/llama-3.3-70b-instruct"
                    try:
                        async for chunk in model_gateway.stream("openrouter", prov_req, stage_name="deliberate_fallback"):
                            collected_chunks.append(chunk)
                            chunk_payload = {
                                "type": "chunk",
                                "turn_id": turn_id,
                                "sender": current_agent,
                                "token": chunk,
                            }
                            await websocket.send_text(json.dumps(chunk_payload))
                    except Exception as fallback_err:
                        error_msg = scrub_credentials(str(fallback_err))
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "turn_id": turn_id,
                            "error": error_msg,
                        }))
                        continue

                full_synthesis = "".join(collected_chunks)
                lat_ms = int((time.time() - t0) * 1000)

                turn_record = {
                    "turn_id": turn_id,
                    "session_id": current_session,
                    "agent_id": current_agent,
                    "input_prompt": clean_text[:200],
                    "synthesis": full_synthesis,
                    "latency_ms": lat_ms,
                    "timestamp": time.time(),
                }
                await deliberation_manager.record_turn(current_session, turn_record)

                complete_payload = {
                    "type": "turn_complete",
                    "turn_id": turn_id,
                    "session_id": current_session,
                    "sender": current_agent,
                    "synthesis": full_synthesis,
                    "latency_ms": lat_ms,
                }
                await websocket.send_text(json.dumps(complete_payload))
                # Broadcast completed turn to all peers
                await deliberation_manager.broadcast(current_session, complete_payload, exclude=websocket)

            elif msg_type == "consensus_request":
                history = await deliberation_manager.get_history(current_session)
                summary_context = "\n".join([
                    f"- {t['agent_id']}: {t['synthesis'][:250]}" for t in history[-6:]
                ]) or "No prior deliberation turns recorded."

                consensus_prompt = (
                    f"Synthesize final consensus for session '{current_session}' based on all perspectives:\n"
                    f"{summary_context}\n\n"
                    "Provide: 1) Executive Resolution, 2) Convergence Score (0.0 to 1.0), 3) Immediate Action Items."
                )

                consensus_req = ProviderRequest(
                    messages=[ProviderMessage(role="user", content=consensus_prompt)],
                    model="openai/gpt-oss-120b",
                    temperature=0.2,
                    max_tokens=1500,
                )

                t0 = time.time()
                collected = []
                try:
                    async for chunk in model_gateway.stream("groq", consensus_req, stage_name="consensus_stream"):
                        collected.append(chunk)
                except Exception:
                    consensus_req.model = "deepseek/deepseek-r1"
                    try:
                        async for chunk in model_gateway.stream("openrouter", consensus_req, stage_name="consensus_stream"):
                            collected.append(chunk)
                    except Exception:
                        consensus_req.model = "meta-llama/llama-3.3-70b-instruct"
                        async for chunk in model_gateway.stream("openrouter", consensus_req, stage_name="consensus_fallback"):
                            collected.append(chunk)

                verdict_text = "".join(collected)
                lat_ms = int((time.time() - t0) * 1000)
                convergence_score = 0.92 if "consensus" in verdict_text.lower() else 0.85
                delib_id = f"delib_{uuid.uuid4().hex[:12]}"
                participating_agents = list({t.get("agent_id", "") for t in history if t.get("agent_id")} | {current_agent})

                # Dual-persist deliberation audit to database
                delib_rec = DeliberationRecord(
                    id=delib_id,
                    session_id=current_session,
                    topic=str(msg.get("topic") or (history[0]["input_prompt"] if history else "Multi-agent Deliberation")),
                    participating_agents=participating_agents,
                    consensus_strategy=str(msg.get("strategy") or "adjudicator"),
                    final_decision=verdict_text,
                    confidence=convergence_score,
                    rounds=len(history),
                    metadata={"latency_ms": lat_ms, "caller": current_agent},
                )
                try:
                    await postgres_memory.save_deliberation(delib_rec)
                except Exception as db_exc:
                    logger.debug("[WS_DELIBERATE] Deliberation persistence notice: %s", db_exc)

                verdict_payload = {
                    "type": "consensus_verdict",
                    "session_id": current_session,
                    "deliberation_id": delib_id,
                    "verdict": verdict_text,
                    "convergence_score": convergence_score,
                    "participating_agents": participating_agents,
                    "turns_evaluated": len(history),
                    "latency_ms": lat_ms,
                }
                # Broadcast verdict to all participants in session
                await websocket.send_text(json.dumps(verdict_payload))
                await deliberation_manager.broadcast(current_session, verdict_payload, exclude=websocket)

    except WebSocketDisconnect:
        if current_session:
            departed = await deliberation_manager.disconnect(current_session, websocket)
            if departed:
                await deliberation_manager.broadcast(
                    current_session,
                    {
                        "type": "peer_left",
                        "session_id": current_session,
                        "agent_id": departed,
                        "timestamp": time.time(),
                    },
                )
    except Exception as exc:
        logger.warning(f"[WS_DELIBERATE] Exception in session loop: {exc}")
        if current_session:
            await deliberation_manager.disconnect(current_session, websocket)


@deliberate_stream_router.websocket("/v1/stream")
async def websocket_direct_stream_endpoint(websocket: WebSocket) -> None:
    """Duplex direct token streaming over WebSocket."""
    await websocket.accept()

    auth_key = (
        websocket.headers.get("x-inference-api-key")
        or websocket.headers.get("x-api-key")
        or websocket.query_params.get("api_key")
    )

    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=15.0)
        req = json.loads(raw)

        if not auth_key:
            auth_key = req.get("api_key")

        if not _authenticate_ws_key(auth_key):
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": "UNAUTHORIZED",
                "message": "Invalid API key",
            }))
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        prompt = req.get("prompt") or req.get("content") or "Hello"
        provider = str(req.get("provider") or "groq").lower()
        model = req.get("model") or ("openai/gpt-oss-120b" if provider == "groq" else "meta-llama/llama-3.3-70b-instruct")
        max_tokens = min(int(req.get("max_tokens") or 1024), 8192)

        prov_req = ProviderRequest(
            messages=[ProviderMessage(role="user", content=prompt)],
            model=model,
            temperature=float(req.get("temperature", 0.3)),
            max_tokens=max_tokens,
        )

        t0 = time.time()
        try:
            async for token in model_gateway.stream(provider, prov_req, stage_name="ws_direct_stream"):
                await websocket.send_text(json.dumps({"type": "token", "token": token, "done": False}))
        except Exception as e:
            # Automatic failover to OpenRouter
            logger.warning(f"[WS_STREAM] Primary stream failed ({e}), failing over to OpenRouter")
            prov_req.model = "meta-llama/llama-3.3-70b-instruct"
            async for token in model_gateway.stream("openrouter", prov_req, stage_name="ws_direct_fallback"):
                await websocket.send_text(json.dumps({"type": "token", "token": token, "done": False}))

        lat_ms = int((time.time() - t0) * 1000)
        await websocket.send_text(json.dumps({"type": "done", "done": True, "latency_ms": lat_ms}))
        await websocket.close()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug(f"[WS_STREAM] Stream closed with error: {exc}")


class V1StreamRequest(BaseModel):
    """Payload for HTTP POST /v1/stream."""

    prompt: str = Field(default="Hello", description="Prompt or query to stream")
    model: str = Field(default="auto", description="Target model or 'auto'")
    provider: str = Field(default="auto", description="Target provider or 'auto'")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=128000)


@deliberate_stream_router.post("/v1/stream", dependencies=[Depends(require_inference_api_key)])
async def v1_http_stream(req: V1StreamRequest) -> StreamingResponse:
    """Standardized SSE token streaming endpoint for mesh clients."""
    tokens = max(1, min(req.max_tokens, 128000))
    prov = "groq" if req.provider == "auto" else req.provider
    mod = "openai/gpt-oss-120b" if req.model == "auto" else req.model

    prov_req = ProviderRequest(
        messages=[ProviderMessage(role="user", content=req.prompt)],
        system_instruction="You are Inference, the centralized LLM gateway for the FRIDAY Universe.",
        model=mod,
        temperature=req.temperature,
        max_tokens=tokens,
    )

    async def _token_generator():
        try:
            async for chunk in model_gateway.stream(provider=prov, request=prov_req, stage_name="v1_stream"):
                yield f"data: {json.dumps({'token': chunk, 'done': False})}\n\n"
        except Exception as exc:
            logger.warning(f"[V1_STREAM] Primary stream failed ({exc}), attempting openrouter fallback")
            try:
                prov_req.model = "meta-llama/llama-3.3-70b-instruct"
                async for chunk in model_gateway.stream(provider="openrouter", request=prov_req, stage_name="v1_stream_fallback"):
                    yield f"data: {json.dumps({'token': chunk, 'done': False})}\n\n"
            except Exception as f_err:
                yield f"data: {json.dumps({'error': scrub_credentials(str(f_err)), 'done': True})}\n\n"
                return
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        _token_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
