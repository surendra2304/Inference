"""Architecture Planning Service for FORGE."""

import time
from typing import Literal

from pydantic import BaseModel, Field

from app.providers.unified_manager import (
    UnifiedExecutionRequest,
    unified_provider_manager,
)
from app.utils.confidence import DEGRADED_CONFIDENCE, UNVERIFIED_MODEL_CONFIDENCE


class ArchitecturePlanRequest(BaseModel):
    goal: str = Field(..., description="Project goal or high-level application description")
    project_type: Literal["cli", "web", "api", "script", "dashboard"] = Field(
        default="api", description="Category of software artifact"
    )
    constraints: list[str] = Field(default_factory=list, description="Performance, security, or deployment constraints")
    preferences: list[str] = Field(default_factory=list, description="Preferred frameworks or libraries")


class FileManifestEntry(BaseModel):
    filename: str
    purpose: str
    dependencies: list[str] = Field(default_factory=list)


class ArchitecturePlanResponse(BaseModel):
    architecture_spec: str
    file_manifest: list[FileManifestEntry]
    tech_stack: list[str]
    confidence: float
    latency_ms: float


class ArchitecturePlanningService:
    """Deconstructs goals into modular architectures, tech stacks, and manifest dependencies."""

    async def plan_architecture(self, req: ArchitecturePlanRequest) -> ArchitecturePlanResponse:
        start_time = time.perf_counter()

        prompt = (
            f"Design a modular software architecture for goal: {req.goal}\n"
            f"Project Type: {req.project_type}\n"
            f"Constraints: {', '.join(req.constraints) if req.constraints else 'None'}\n"
            f"Preferences: {', '.join(req.preferences) if req.preferences else 'Standard production best practices'}\n\n"
            "Provide: (1) Architecture Overview, (2) Recommended Tech Stack, (3) File Manifest with inter-file dependencies."
        )

        exec_req = UnifiedExecutionRequest(
            provider="auto",
            agent_role="system_architect",
            prompt=prompt,
            max_tokens=3000,
            temperature=0.3
        )

        exec_res = await unified_provider_manager.execute(exec_req)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Default structured manifest based on project type
        manifest = self._generate_default_manifest(req.project_type)
        tech_stack = self._determine_tech_stack(req.project_type, req.preferences)

        # The spec is unverified model text and the manifest is a fixed template for the project
        # type, not derived from the spec. A degraded call reports no spec. Before this, every
        # call returned confidence 0.94 (see app/utils/confidence.py).
        if exec_res.degraded or not exec_res.content:
            return ArchitecturePlanResponse(
                architecture_spec="",
                file_manifest=manifest,
                tech_stack=tech_stack,
                confidence=DEGRADED_CONFIDENCE,
                latency_ms=elapsed_ms,
            )
        return ArchitecturePlanResponse(
            architecture_spec=exec_res.content,
            file_manifest=manifest,
            tech_stack=tech_stack,
            confidence=UNVERIFIED_MODEL_CONFIDENCE,
            latency_ms=elapsed_ms
        )

    def _determine_tech_stack(self, ptype: str, prefs: list[str]) -> list[str]:
        if prefs:
            return prefs
        defaults = {
            "api": ["FastAPI", "Pydantic v2", "Uvicorn", "Pytest", "AsyncIO"],
            "web": ["HTML5", "Modern CSS", "ES6 Vanilla JS", "FastAPI"],
            "cli": ["Click/Typer", "Rich", "Pytest"],
            "script": ["Python 3.11+", "Standard Library"],
            "dashboard": ["Streamlit", "Plotly", "Pandas"]
        }
        return defaults.get(ptype, ["Python", "FastAPI"])

    def _generate_default_manifest(self, ptype: str) -> list[FileManifestEntry]:
        if ptype == "api":
            return [
                FileManifestEntry(filename="app/main.py", purpose="FastAPI application entrypoint", dependencies=["app/routes.py", "app/config.py"]),
                FileManifestEntry(filename="app/routes.py", purpose="API endpoint controllers", dependencies=["app/schemas.py", "app/services.py"]),
                FileManifestEntry(filename="app/schemas.py", purpose="Pydantic request/response models", dependencies=[]),
                FileManifestEntry(filename="app/services.py", purpose="Core business logic domain", dependencies=["app/schemas.py"]),
                FileManifestEntry(filename="tests/test_api.py", purpose="Pytest unit and integration test suite", dependencies=["app/main.py"])
            ]
        elif ptype == "cli":
            return [
                FileManifestEntry(filename="cli.py", purpose="Command line entrypoint and arg parsing", dependencies=["core.py"]),
                FileManifestEntry(filename="core.py", purpose="CLI operational command logic", dependencies=[]),
                FileManifestEntry(filename="tests/test_cli.py", purpose="CLI test runner", dependencies=["cli.py"])
            ]
        return [
            FileManifestEntry(filename="main.py", purpose="Application entrypoint", dependencies=["utils.py"]),
            FileManifestEntry(filename="utils.py", purpose="Helper functions and utilities", dependencies=[]),
            FileManifestEntry(filename="tests/test_main.py", purpose="Test suite", dependencies=["main.py"])
        ]


architecture_planning_service = ArchitecturePlanningService()
