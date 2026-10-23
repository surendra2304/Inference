"""FRIDAY Universe Peer Mesh Discovery, Status, and Probe Gateway.



Provides centralized visibility into all 9 peer agents in the FRIDAY Universe,

their mutual authentication posture, assigned burst quotas, and active model affinities.

"""



from __future__ import annotations

import os
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.security import require_inference_api_key
from app.ha.high_availability import HighAvailabilityManager
from app.inference_runtime.quotas import agent_quota_manager
from app.providers.http_client import get_shared_client
from app.utils.logger import logger

mesh_router = APIRouter(prefix="/v1/mesh", tags=["FRIDAY Universe Peer Mesh"])



PEER_METADATA: dict[str, dict[str, str]] = {

    "friday": {

        "name": "FRIDAY",

        "role": "Executive Central Orchestrator & Autonomous Decision Gateway",

        "agent_number": "1",

        "primary_model": "openai/gpt-oss-120b",

        "fallback_model": "llama-3.3-70b-versatile",

    },

    "memora": {

        "name": "Memora",

        "role": "Long-Term Episodic & Semantic Memory Intelligence Engine",

        "agent_number": "3",

        "primary_model": "meta-llama/llama-3.3-70b-instruct",

        "fallback_model": "gemini-2.5-flash",

    },

    "forge": {

        "name": "Forge",

        "role": "Autonomous Software Engineering, AST Synthesis, & Code Review",

        "agent_number": "4",

        "primary_model": "qwen/qwen-2.5-coder-32b-instruct",

        "fallback_model": "llama-3.3-70b-versatile",

    },

    "cortex": {

        "name": "Cortex",

        "role": "Deep Analytical Reasoning, Chain-of-Thought, & Council Deliberation",

        "agent_number": "5",

        "primary_model": "deepseek/deepseek-r1",

        "fallback_model": "meta-llama/llama-3.3-70b-instruct",

    },

    "sentinel": {

        "name": "Sentinel",

        "role": "Security Auditing, Threat Detection, & Prompt Injection Barrier",

        "agent_number": "6",

        "primary_model": "meta-llama/llama-3.3-70b-instruct",

        "fallback_model": "gemini-2.5-flash",

    },

    "stratex": {

        "name": "Stratex",

        "role": "Strategic Planning, Market Regime Analysis, & Scenario Adjudication",

        "agent_number": "7",

        "primary_model": "deepseek/deepseek-r1",

        "fallback_model": "openai/gpt-oss-120b",

    },

    "intelx": {

        "name": "IntelX",

        "role": "Deep Research Synthesis, Information Extraction, & Verification",

        "agent_number": "8",

        "primary_model": "meta-llama/llama-3.3-70b-instruct",

        "fallback_model": "gemini-2.5-flash",

    },

    "futuris": {

        "name": "Futuris",

        "role": "Predictive Forecasting, Trend Projection, & Probabilistic Risk",

        "agent_number": "9",

        "primary_model": "meta-llama/llama-3.3-70b-instruct",

        "fallback_model": "openai/gpt-oss-120b",

    },

    "vyralx": {

        "name": "VyralX",

        "role": "Viral Media Optimization, Content Generation, & Audience Resonance",

        "agent_number": "10",

        "primary_model": "meta-llama/llama-3.3-70b-instruct",

        "fallback_model": "gemini-2.5-flash",

    },

}



_ha_mgr = HighAvailabilityManager()





class PeerAgentInfo(BaseModel):

    agent_id: str

    name: str

    agent_number: str

    role: str

    url: str

    authenticated: bool

    primary_model: str

    fallback_model: str

    quota: dict[str, Any]





class MeshPeersResponse(BaseModel):

    total_peers: int

    peers: list[PeerAgentInfo]

    timestamp: float = Field(default_factory=time.time)





class MeshStatusResponse(BaseModel):

    mesh_network: str = "friday_mesh"

    node_id: str = "inference"

    node_role: str = "Model Broker & Deliberation Gateway"

    agent_number: str = "2"

    status: str = "OPTIMAL"

    ha_status: dict[str, Any]

    provider_pools: dict[str, int]

    total_keys_managed: int

    active_reasoning_model: str

    active_coding_model: str

    persistence_authority: dict[str, Any]

    timestamp: float = Field(default_factory=time.time)





@mesh_router.get("/peers", response_model=MeshPeersResponse, dependencies=[Depends(require_inference_api_key)])

async def get_mesh_peers() -> MeshPeersResponse:

    """Returns status, connection details, quotas, and model affinities for all 9 peer agents."""

    peer_list: list[PeerAgentInfo] = []

    for agent_id, meta in PEER_METADATA.items():

        url, key = settings.get_peer_config(agent_id)

        quota_status = agent_quota_manager.get_status(agent_id)

        peer_list.append(

            PeerAgentInfo(

                agent_id=agent_id,

                name=meta["name"],

                agent_number=meta["agent_number"],

                role=meta["role"],

                url=url,

                authenticated=bool(key),

                primary_model=meta["primary_model"],

                fallback_model=meta["fallback_model"],

                quota=quota_status,

            )

        )



    return MeshPeersResponse(

        total_peers=len(peer_list),

        peers=peer_list,

    )





@mesh_router.get("/status", response_model=MeshStatusResponse, dependencies=[Depends(require_inference_api_key)])

async def get_mesh_status() -> MeshStatusResponse:

    """Returns aggregated mesh health, active HA chain, key pool inventory, and persistence state."""

    groq_keys = len(settings.get_provider_keys("groq"))

    openrouter_keys = len(settings.get_provider_keys("openrouter"))

    gemini_keys = len(settings.get_provider_keys("gemini"))

    nvidia_keys = len(settings.get_provider_keys("nvidia"))



    pools = {

        "groq": groq_keys,

        "openrouter": openrouter_keys,

        "gemini": gemini_keys,

        "nvidia": nvidia_keys,

    }

    total_keys = sum(pools.values())

    ha_telemetry = _ha_mgr.get_ha_status()



    return MeshStatusResponse(

        ha_status=ha_telemetry,

        provider_pools=pools,

        total_keys_managed=total_keys,

        active_reasoning_model="deepseek/deepseek-r1",

        active_coding_model="qwen/qwen-2.5-coder-32b-instruct",

        persistence_authority={

            "primary": f"PostgreSQL 18 ({settings.POSTGRES_DB} on {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT})",

            "oracle_23ai": f"Oracle ATP 23ai (friday_universe_db on {settings.ORACLE_HOST}:{settings.ORACLE_PORT})",

            "fallback": f"SQLite ({settings.DATABASE_URL} with WAL mode)",

            "ephemeral_shm": "/dev/shm/inference.db" if os.path.exists("/dev/shm") else ":memory:",

            "zero_disk_compliant": True,

            "schema_synced": True,

        },

    )





@mesh_router.post("/ping/{agent_id}", dependencies=[Depends(require_inference_api_key)])

async def ping_mesh_peer(agent_id: str) -> dict[str, Any]:

    """Pings a peer agent's health endpoint with a fast 2.0s deadline."""

    norm_id = agent_id.lower().strip()

    if norm_id not in PEER_METADATA:

        raise HTTPException(

            status_code=status.HTTP_404_NOT_FOUND,

            detail=f"Agent '{agent_id}' is not a recognized FRIDAY Universe peer agent.",

        )



    url, key = settings.get_peer_config(norm_id)

    target_health_url = f"{url.rstrip('/')}/health"

    correlation_id = str(uuid.uuid4())

    headers = {

        "X-Agent-Name": "Inference",

        "X-Correlation-ID": correlation_id,

    }

    if key:

        headers["Authorization"] = f"Bearer {key}"

        headers["X-API-KEY"] = key



    client = await get_shared_client()

    t0 = time.perf_counter()

    try:

        resp = await client.get(target_health_url, headers=headers, timeout=2.0)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {

            "agent_id": norm_id,

            "target_url": target_health_url,

            "reachable": resp.status_code == 200,

            "status_code": resp.status_code,

            "latency_ms": elapsed_ms,

            "authenticated_ping": bool(key),

        }

    except Exception as exc:

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        logger.debug("Peer ping failed for %s: %s", norm_id, exc)

        return {

            "agent_id": norm_id,

            "target_url": target_health_url,

            "reachable": False,

            "status_code": None,

            "error": "PEER_UNREACHABLE",

            "latency_ms": elapsed_ms,

            "authenticated_ping": bool(key),

        }

