"""Python SDK Client for Inference Multi-Agent Intelligence Platform."""

from typing import Any

import httpx
from pydantic import BaseModel, Field


class IntelligenceRequest(BaseModel):
    request_id: str
    task_type: str
    goal: str
    context: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    mode: str = "fast"


class AIUniverseClient:
    """Typed client for interacting with Inference intelligence endpoints."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(base_url=self.base_url, headers=self.headers, timeout=timeout)

    def query_nexus_intelligence(self, request: IntelligenceRequest) -> dict[str, Any]:
        """Queries Nexus decision intelligence endpoint."""
        resp = self.client.post("/v1/nexus/intelligence", json=request.model_dump())
        resp.raise_for_status()
        return resp.json()

    def generate_code(self, file_type: str, filename: str, context: dict[str, Any], requirements: list[str] | None = None) -> dict[str, Any]:
        """Queries FORGE code generation endpoint."""
        payload = {
            "file_type": file_type,
            "filename": filename,
            "context": context,
            "requirements": requirements or []
        }
        resp = self.client.post("/v1/forge/generate-code", json=payload)
        resp.raise_for_status()
        return resp.json()

    def enhance_statistical_forecast(self, request_id: str, statistical_forecast: dict[str, Any], target_context: dict[str, Any] | None = None, contextual_factors: list[str] | None = None, question: str | None = None) -> dict[str, Any]:
        """Queries Futuris statistical forecast enhancement endpoint."""
        payload = {
            "request_id": request_id,
            "statistical_forecast": statistical_forecast,
            "target_context": target_context or {},
            "contextual_factors": contextual_factors or [],
            "question": question or "Given this forecast and context, what risks or drivers should be considered?"
        }
        resp = self.client.post("/v1/futuris/enhance", json=payload)
        resp.raise_for_status()
        return resp.json()

    def query_intelx_research(self, request_id: str, role: str, context: dict[str, Any], evidence_with_spans: list[dict[str, Any]] | None = None, constraints: dict[str, Any] | None = None) -> dict[str, Any]:
        """Queries IntelX deep research intelligence endpoint."""
        payload = {
            "request_id": request_id,
            "role": role,
            "context": context,
            "evidence_with_spans": evidence_with_spans or [],
            "constraints": constraints or {}
        }
        resp = self.client.post("/v1/intelx/research", json=payload)
        resp.raise_for_status()
        return resp.json()

    def query_sentinel_analysis(self, request_id: str, analysis_type: str, target_context: dict[str, Any], findings: list[dict[str, Any]] | None = None, threat_intel: dict[str, Any] | None = None) -> dict[str, Any]:
        """Queries Sentinel cybersecurity intelligence endpoint."""
        payload = {
            "request_id": request_id,
            "analysis_type": analysis_type,
            "target_context": target_context,
            "findings": findings or [],
            "threat_intel": threat_intel or {}
        }
        resp = self.client.post("/v1/sentinel/analyze", json=payload)
        resp.raise_for_status()
        return resp.json()

    def report_outcome(self, consumer: str, request_id: str, outcome: str, detail: str | None = None, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reports downstream verification or execution outcome for self-optimization."""
        payload = {
            "consumer": consumer,
            "request_id": request_id,
            "outcome": outcome,
            "detail": detail or "",
            "measured_metrics": metrics or {}
        }
        resp = self.client.post("/v1/analytics/outcome", json=payload)
        resp.raise_for_status()
        return resp.json()

    def agent_assist(
        self,
        caller_agent: str,
        task_type: str,
        prompt: str,
        context: dict[str, Any] | None = None,
        fast_lane: bool = True,
        speculative: bool = False,
        no_cache: bool = False,
        max_tokens: int = 1500,
    ) -> dict[str, Any]:
        """Universal fast-path inter-agent assist endpoint with L1 caching and speculative racing."""
        payload = {
            "caller_agent": caller_agent,
            "task_type": task_type,
            "prompt": prompt,
            "context": context or {},
            "fast_lane": fast_lane,
            "speculative": speculative,
            "no_cache": no_cache,
            "max_tokens": max_tokens,
        }
        resp = self.client.post("/v1/agent/assist", json=payload)
        resp.raise_for_status()
        return resp.json()

    def stream_agent_assist(
        self,
        caller_agent: str,
        task_type: str,
        prompt: str,
        context: dict[str, Any] | None = None,
        max_tokens: int = 1500,
    ):
        """Streams inter-agent intelligence tokens in real-time as a Python generator."""
        import json
        payload = {
            "caller_agent": caller_agent,
            "task_type": task_type,
            "prompt": prompt,
            "context": context or {},
            "max_tokens": max_tokens,
        }
        with self.client.stream("POST", "/v1/agent/stream", json=payload) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if data.get("done"):
                            break
                        tok = data.get("token", "")
                        if tok:
                            yield tok
                    except Exception:
                        pass

    def stream_code(
        self,
        file_type: str,
        filename: str,
        context: dict[str, Any],
        requirements: list[str] | None = None,
    ):
        """Streams generated code chunks from FORGE stream-code endpoint."""
        import json
        payload = {
            "file_type": file_type,
            "filename": filename,
            "context": context,
            "requirements": requirements or [],
        }
        with self.client.stream("POST", "/v1/forge/stream-code", json=payload) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if data.get("done"):
                            break
                        chunk = data.get("chunk", "")
                        if chunk:
                            yield chunk
                    except Exception:
                        pass

    def ask_friday(
        self,
        question: str,
        mode: str = "auto",
        fast_lane: bool = True,
        no_cache: bool = False,
    ) -> dict[str, Any]:
        """Queries FRIDAY reasoning endpoint."""
        payload = {
            "question": question,
            "mode": mode,
            "fast_lane": fast_lane,
            "no_cache": no_cache,
        }
        resp = self.client.post("/v1/friday/ask", json=payload)
        resp.raise_for_status()
        return resp.json()

    def instant_ask(
        self,
        prompt: str,
        caller_id: str = "human",
        no_cache: bool = False,
    ) -> dict[str, Any]:
        """Queries instant ultra-low latency endpoint with L0 grounding and speculative execution."""
        payload = {
            "prompt": prompt,
            "caller_id": caller_id,
            "no_cache": no_cache,
        }
        resp = self.client.post("/v1/instant/ask", json=payload)
        resp.raise_for_status()
        return resp.json()

    def stream_instant(
        self,
        prompt: str,
        caller_id: str = "human",
    ):
        """Streams instant answer tokens in real-time."""
        import json

        payload = {
            "prompt": prompt,
            "caller_id": caller_id,
        }
        with self.client.stream("POST", "/v1/instant/stream", json=payload) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if data.get("done"):
                            break
                        tok = data.get("token", "")
                        if tok:
                            yield tok
                    except Exception:
                        pass
