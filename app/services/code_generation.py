"""Code Generation Service optimized for FORGE autonomous software engineering engine."""

import hashlib
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.providers.base import ProviderMessage, ProviderRequest
from app.providers.gateway import model_gateway
from app.providers.unified_manager import (
    UnifiedExecutionRequest,
    unified_provider_manager,
)
from app.utils.logger import logger


class CodeGenerationRequest(BaseModel):
    file_type: Literal["python", "html", "css", "js", "json", "markdown", "sql"] = Field(
        default="python", description="Target file language/format"
    )
    filename: str = Field(..., description="Target file name or relative path")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Project context: project_goal, architecture_spec, file_manifest, related_files"
    )
    requirements: list[str] = Field(default_factory=list, description="Specific functional requirements")
    language_features: list[str] = Field(default_factory=list, description="Target syntax/framework features")


class CodeGenerationResponse(BaseModel):
    code: str
    confidence: float
    generation_path: Literal["agent", "template_fallback"]
    token_usage: int
    latency_ms: float
    filename: str


class CodeGenerationService:
    """Specialized code generator with per-language prompt engineering, context pruning, and caching."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, CodeGenerationResponse]] = {}
        self._cache_ttl = 3600.0  # 1 hour TTL

    def _get_cache_key(self, req: CodeGenerationRequest) -> str:
        goal = str(req.context.get("project_goal", ""))
        raw = f"{goal}:{req.filename}:{req.file_type}:{req.requirements}:{req.language_features}"
        return hashlib.sha256(raw.encode()).hexdigest()

    async def generate_code(self, req: CodeGenerationRequest) -> CodeGenerationResponse:
        start_time = time.perf_counter()
        cache_key = self._get_cache_key(req)
        now = time.time()

        # Check in-memory cache
        if cache_key in self._cache:
            ts, cached_resp = self._cache[cache_key]
            if now - ts < self._cache_ttl:
                logger.info("Code generation cache hit for %s", req.filename)
                return cached_resp

        # Format specialized prompt
        prompt = self._build_prompt(req)

        # Execute through unified provider manager
        exec_req = UnifiedExecutionRequest(
            provider="auto",
            agent_role="code_generator",
            prompt=prompt,
            context=self._prune_context(req.context),
            max_tokens=8000,
            temperature=0.2
        )

        exec_res = await unified_provider_manager.execute(exec_req)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        code_text = exec_res.content
        if "```" in code_text:
            lines = code_text.splitlines()
            start_idx = 0
            end_idx = len(lines)
            for i, line in enumerate(lines):
                if line.startswith("```") and i == 0:
                    start_idx = 1
                elif line.startswith("```") and i > 0:
                    end_idx = i
                    break
            code_text = "\n".join(lines[start_idx:end_idx]).strip()

        gen_path: Literal["agent", "template_fallback"] = "template_fallback" if exec_res.status == "fallback_success" else "agent"
        confidence = 0.92 if gen_path == "agent" else 0.55

        response = CodeGenerationResponse(
            code=code_text,
            confidence=confidence,
            generation_path=gen_path,
            token_usage=exec_res.token_usage.get("total_tokens", 350),
            latency_ms=elapsed_ms,
            filename=req.filename
        )

        self._cache[cache_key] = (now, response)
        return response

    async def stream_code(self, req: CodeGenerationRequest) -> AsyncIterator[str]:
        """Streams generated code tokens in real-time for FORGE."""
        prompt = self._build_prompt(req)
        system_prompt = (
            "You are an expert autonomous software engineer in Inference. "
            "Write clean, production-ready source code. Return ONLY valid runnable code."
        )
        prov_req = ProviderRequest(
            messages=[ProviderMessage(role="user", content=prompt)],
            system_instruction=system_prompt,
            model="openai/gpt-oss-120b",
            temperature=0.2,
            max_tokens=4096,
        )

        try:
            async for chunk in model_gateway.stream("groq", prov_req, stage_name="forge_stream_code"):
                yield chunk
        except Exception as exc:
            logger.warning("Groq stream for code generation failed, falling back to gemini: %s", exc)
            prov_req.model = "gemini-3.6-flash"
            async for chunk in model_gateway.stream("gemini", prov_req, stage_name="forge_stream_code_fallback"):
                yield chunk

    def _build_prompt(self, req: CodeGenerationRequest) -> str:
        lang_guides = {
            "python": "Adhere strictly to PEP 8, full type annotations, docstrings, and robust error handling.",
            "html": "Generate semantic HTML5, valid ARIA tags, meta tags, and clean hierarchy.",
            "css": "Use modern CSS3 (Flexbox/Grid), CSS custom properties, and responsive media queries.",
            "js": "Use ES6+, async/await, no global scope pollution, and try/catch blocks.",
            "json": "Generate strictly valid JSON syntax.",
            "markdown": "Produce GitHub Flavored Markdown with clean tables and code blocks.",
            "sql": "Write ANSI SQL standard queries with parameterized syntax."
        }
        guide = lang_guides.get(req.file_type, "Write clean, idiomatic code.")

        req_list = "\n".join([f"- {r}" for r in req.requirements]) if req.requirements else "None specified."
        feat_list = ", ".join(req.language_features) if req.language_features else "Standard"

        return (
            f"Generate complete, production-ready, fully closed code for file: `{req.filename}` ({req.file_type}).\n"
            f"Standards: {guide}\n"
            f"Language Features: {feat_list}\n"
            f"Specific Requirements:\n{req_list}\n\n"
            "Keep the implementation modular, focused, and under 400 lines with zero truncated blocks.\n"
            "Return ONLY the runnable source code."
        )

    def _prune_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """Prunes and limits context size (max 3 related files)."""
        pruned: dict[str, Any] = {}
        if "project_goal" in context:
            pruned["project_goal"] = str(context["project_goal"])[:500]
        if "architecture_spec" in context:
            pruned["architecture_spec"] = str(context["architecture_spec"])[:1000]
        if "related_files" in context:
            rel = context["related_files"]
            if isinstance(rel, dict):
                pruned["related_files"] = {k: str(v)[:400] for i, (k, v) in enumerate(rel.items()) if i < 3}
            elif isinstance(rel, list):
                pruned["related_files"] = [str(x)[:400] for x in rel[:3]]
        return pruned


code_generation_service = CodeGenerationService()
