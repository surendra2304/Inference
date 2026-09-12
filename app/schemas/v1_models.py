"""Pydantic Request and Response Models for Inference v1 Core Protocol.

Strictly compatible with the FRIDAY Task Envelope and compliant with Prompt 2:
1. Every response includes all 13 canonical fields:
   task_id, trace_id, answer, reasoning_summary, confidence, uncertainty,
   evidence, agents_used, recommendations, proposed_actions,
   authorization_required, provider_metadata, failure_state.
2. proposed_actions are strictly advisory objects with is_executable_command=False.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ProposedAction(BaseModel):
    """Advisory action proposed by deliberative reasoning.

    STRICT INVARIANT:
    These are purely deliberative recommendations for FRIDAY and the user.
    They NEVER hold direct execution authority (is_executable_command MUST BE False).
    """

    action: str = Field(description="Action name or intent (e.g. adjust_stop_loss, update_config, trigger_backup)")
    target: str = Field(description="Target subsystem, service, or asset (e.g. stratex, sentinel, config)")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Recommended parameters")
    rationale: str = Field(default="", description="Advisory rationale from reasoning council")
    is_executable_command: bool = Field(
        default=False,
        description="Strict safety invariant: Always False. Model outputs are never executable commands.",
    )
    requires_authorization: bool = Field(
        default=True,
        description="Whether this proposed action requires human-in-the-loop confirmation",
    )
    risk_level: str = Field(
        default="low",
        description="Risk classification: low | medium | high | critical",
    )

    @field_validator("is_executable_command")
    @classmethod
    def enforce_non_executable(cls, v: bool) -> bool:
        if v is True:
            raise ValueError(
                "Security Invariant Violation: is_executable_command cannot be True. "
                "Inference proposed_actions are strictly advisory and never executable commands."
            )
        return False


class ProviderMetadata(BaseModel):
    """Telemetry and routing metadata for the inference provider."""

    provider: str = Field(default="auto")
    model: str = Field(default="deliberative-council")
    fallback_chain: list[str] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    budget_consumed_usd: float = 0.0
    budget_ceiling_usd: float = 0.0


class InferenceTaskResponse(BaseModel):
    """Universal Deliberative Reasoning Response containing all 13 canonical fields."""

    # 1. task_id
    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    # 2. trace_id
    trace_id: str = Field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:8]}")
    # 3. answer
    answer: str = Field(description="The primary synthesized resolution or response")
    # 4. reasoning_summary
    reasoning_summary: str = Field(default="", description="High-level deliberation summary")
    # 5. confidence
    confidence: float = Field(default=0.90, ge=0.0, le=1.0, description="Calibrated confidence (0.0 to 1.0)")
    # 6. uncertainty
    uncertainty: float = Field(default=0.10, ge=0.0, le=1.0, description="Calibrated residual uncertainty (0.0 to 1.0)")
    # 7. evidence
    evidence: list[str] = Field(default_factory=list, description="Verified factual evidence and citations")
    # 8. agents_used
    agents_used: list[str] = Field(default_factory=list, description="Specialist agents or debate roles involved")
    # 9. recommendations
    recommendations: list[str] = Field(default_factory=list, description="Actionable advisory guidance")
    # 10. proposed_actions
    proposed_actions: list[ProposedAction] = Field(default_factory=list, description="Strictly advisory non-executable actions")
    # 11. authorization_required
    authorization_required: bool = Field(default=False, description="Whether human authorization is recommended")
    # 12. provider_metadata
    provider_metadata: ProviderMetadata = Field(default_factory=ProviderMetadata)
    # 13. failure_state
    failure_state: str | None = Field(default=None, description="Failure or error description if degraded")

    # FRIDAY TaskEnvelope compatibility and audit extensions
    status: str = Field(default="SUCCESS")
    target_agent: str = Field(default="inference")
    missing_data: list[str] = Field(default_factory=list, description="Explicitly identifies missing data if evidence is insufficient")
    completed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class InferenceAskRequest(BaseModel):
    """Request schema for /v1/ask compatible with direct queries and FRIDAY TaskEnvelope."""

    prompt: str | None = Field(default=None, description="Primary prompt or question")
    question: str | None = Field(default=None, description="Alias for prompt")
    context: dict[str, Any] = Field(default_factory=dict, description="Structured query context")
    mode: str = Field(default="deliberative", description="fast | deliberative | consensus | review")
    persona: str = Field(default="ASTRA", description="Reasoning persona/council profile")
    task_id: str | None = Field(default=None)
    trace_id: str | None = Field(default=None)
    budget_ceiling: float | None = Field(default=None, description="Max spend limit in USD")
    timeout_seconds: float = Field(default=60.0, description="Timeout budget in seconds")
    untrusted_data: str | None = Field(default=None, description="Raw untrusted input (OCR, scraped text, etc.)")
    untrusted_data_source: str = Field(default="external_tool", description="Source label for untrusted data")

    # TaskEnvelope backward compatibility fields
    inputs: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None
    objective: str | None = None
    action: str | None = None
    source_agent: str = "friday"
    target_agent: str = "inference"
    priority: str = "normal"
    trust_level: str = "operator_confirmed"
    authorization_context: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def resolve_prompt_and_envelope(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Extract prompt from multiple possible locations
            p = (
                data.get("prompt")
                or data.get("question")
                or data.get("objective")
                or data.get("action")
            )
            if not p and "inputs" in data and isinstance(data["inputs"], dict):
                p = (
                    data["inputs"].get("prompt")
                    or data["inputs"].get("question")
                    or data["inputs"].get("query")
                    or data["inputs"].get("objective")
                )
            if not p and "payload" in data and isinstance(data["payload"], dict):
                p = (
                    data["payload"].get("prompt")
                    or data["payload"].get("question")
                    or data["payload"].get("query")
                    or data["payload"].get("objective")
                )
            data["prompt"] = p or "General reasoning inquiry"
            if not data.get("task_id"):
                data["task_id"] = f"task_{uuid.uuid4().hex[:12]}"
            if not data.get("trace_id"):
                data["trace_id"] = f"trace_{uuid.uuid4().hex[:8]}"
        return data


class DebateRequest(BaseModel):
    """Request schema for /v1/debate multi-agent deliberation."""

    topic: str | None = Field(default=None, description="Primary debate topic or query")
    question: str | None = Field(default=None, description="Alias for topic")
    context: dict[str, Any] = Field(default_factory=dict, description="Context data for debate")
    roles: list[str] = Field(
        default_factory=lambda: ["proposer", "critic", "fact_checker", "data_analyst", "strategist", "synthesizer"],
        description="The 6 standard debate roles",
    )
    rounds: int = Field(default=2, ge=1, le=5, description="Number of deliberation rounds")
    budget_ceiling: float | None = Field(default=None, description="Budget limit in USD")
    timeout_seconds: float = Field(default=90.0, description="Timeout budget in seconds")
    task_id: str | None = Field(default=None)
    trace_id: str | None = Field(default=None)
    untrusted_data: str | None = Field(default=None, description="Untrusted input data to isolate")
    untrusted_data_source: str = Field(default="external", description="Source label")

    # Envelope compatibility fields
    inputs: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None
    objective: str | None = None
    action: str | None = None
    source_agent: str = "friday"
    target_agent: str = "inference"

    @model_validator(mode="before")
    @classmethod
    def resolve_topic_and_envelope(cls, data: Any) -> Any:
        if isinstance(data, dict):
            t = (
                data.get("topic")
                or data.get("question")
                or data.get("objective")
                or data.get("action")
            )
            if not t and "inputs" in data and isinstance(data["inputs"], dict):
                t = data["inputs"].get("topic") or data["inputs"].get("question") or data["inputs"].get("query")
            if not t and "payload" in data and isinstance(data["payload"], dict):
                t = data["payload"].get("topic") or data["payload"].get("question") or data["payload"].get("query")
            data["topic"] = t or "Multi-perspective system analysis"
            if not data.get("task_id"):
                data["task_id"] = f"task_{uuid.uuid4().hex[:12]}"
            if not data.get("trace_id"):
                data["trace_id"] = f"trace_{uuid.uuid4().hex[:8]}"
        return data
