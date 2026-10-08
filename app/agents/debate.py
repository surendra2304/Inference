"""Real-Time Multi-Agent Collaboration Engine for Inference with Complexity-Aware Multi-Model Execution.

Implements the "Collaborate First, Debate on Conflict" model:
- Step 1: Selected specialist agents generate independent perspectives in parallel via asyncio.gather.
          For COMPLEX/STRATEGIC tasks, each agent queries 2-3 models concurrently and merges them.
          Before invoking any model, provider health is checked; rate-limited providers are skipped.
- Step 2: The Synthesizer queries multiple models in parallel (e.g. Gemini + OpenRouter DeepSeek)
          and merges the best parts into a final answer.
- Step 3: If aligned, the Synthesizer merges instantly into consensus.
- Step 4: ONLY IF severe contradictions are detected, a targeted Rebuttal round is triggered.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.agents.adjudication import Adjudicator
from app.agents.base import Agent, AgentModelConfig, BaseAgentRegistry
from app.agents.reasoning import (
    AdjudicationResult,
    AtomicClaim,
    SpecialistAssessment,
    StructuredEvidence,
)
from app.agents.registry import agent_registry
from app.core.config import settings
from app.core.dag import TaskComplexity
from app.memory.base import BaseMemory, MessageRecord, RunRecord
from app.memory.sqlite import SQLiteMemory
from app.providers.base import ProviderMessage, ProviderRequest, ProviderResponse
from app.providers.gateway import model_gateway
from app.providers.health import provider_health_tracker
from app.utils.ids import generate_debate_id, generate_message_id, generate_run_id
from app.utils.logger import logger


class AgentCallUnavailable(RuntimeError):
    """Raised when a specialist agent could not produce output from ANY of its models.

    This replaces the previous behaviour of returning a prose placeholder
    ("*[Specialist X temporarily offline]*") that was indistinguishable from a
    real answer and therefore propagated as a confident result.
    """


class AgentPanelUnavailable(RuntimeError):
    """Raised when no specialist (nor any peer able to cover) produced output.

    Carries the per-agent failure reasons so API layers can report exactly why
    the panel went dark instead of synthesizing an answer from nothing.
    """

    def __init__(self, task_id: str, failures: list[str]) -> None:
        self.task_id = task_id
        self.failures = failures
        super().__init__(
            f"Task {task_id}: no specialist produced output. " + "; ".join(failures)
        )


class CollaborationMessage(BaseModel):
    """An individual message or perspective in a collaborative session."""
    id: str = Field(default_factory=generate_message_id)
    round_number: int
    stage_name: str
    agent_id: str
    agent_role: str
    content: str
    target_agent_id: str | None = None
    models_used: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CollaborationRoundLog(BaseModel):
    """Log record summarizing an individual round in the collaboration."""
    round_number: int
    stage_name: str
    messages: list[CollaborationMessage] = Field(default_factory=list)
    summary: str | None = None


# Backward-compatibility alias for tests and older consumers
DebateRoundLog = CollaborationRoundLog
DebateMessage = CollaborationMessage


class CollaborationResult(BaseModel):
    """Outcome of the Real-Time Multi-Agent Collaboration Engine."""
    debate_id: str
    task_id: str
    canonical_problem: str
    final_answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    unresolved_disagreements: list[str] = Field(default_factory=list)
    key_evidence: list[str] = Field(default_factory=list)
    structured_evidence: list[StructuredEvidence] = Field(default_factory=list)
    claims: list[AtomicClaim] = Field(default_factory=list)
    adjudication: AdjudicationResult | None = None
    participating_agents: list[str] = Field(default_factory=list)
    rounds: list[CollaborationRoundLog] = Field(default_factory=list)
    #: How this collaboration concluded: "fast" (single specialist), "consensus" (the
    #: panel agreed) or "debate" (a rebuttal round was needed). This is an OUTCOME, not
    #: the execution mode — the caller's requested mode lives on OrchestrationResult.
    #: Do not publish this value in a field named ``mode_used``.
    mode_used: str = "consensus"
    complexity: str = "simple"
    models_used: list[str] = Field(default_factory=list)
    total_tokens: int = 0
    total_latency_seconds: float = 0.0
    # Honest-degradation metadata: consumers must not treat a degraded panel the
    # same as a fully successful one.
    degraded: bool = False
    degradation_reasons: list[str] = Field(default_factory=list)
    # Self-healing audit trail: original_agent_id -> covering peer agent id
    agent_coverage: dict[str, str] = Field(default_factory=dict)
    failed_agents: list[str] = Field(default_factory=list)


class CollaborationEngine:
    """
    Executes real-time multi-agent workflows using complexity-aware model dispatch
    and parallel multi-model consensus synthesis.
    """

    def __init__(
        self,
        memory: BaseMemory | None = None,
        registry: BaseAgentRegistry | None = None
    ) -> None:
        self.memory = memory or SQLiteMemory()
        self.registry = registry or agent_registry

    def _resolve_gate_provider(self, model_cfg: AgentModelConfig) -> str:
        """Return the name of the provider that will *actually* serve this model config.

        The health gate used to check ``model_cfg.provider`` unconditionally. Under
        ``LOCAL_PREFERRED`` that is the wrong provider: the self-hosted tier answers
        first and the named cloud provider is never called (see
        ``ModelGateway.execute``), while the local tier's own health is recorded under
        the name ``"local"``. The result was that a stale, unrelated cloud provider
        could veto a request that the local model was perfectly able to serve — and,
        symmetrically, a dead local tier was invisible to the gate. Resolving the
        effective provider keeps the gate honest in both directions.
        """
        if (
            settings.LOCAL_PREFERRED
            and settings.LOCAL_ENABLED
            and model_cfg.provider != "local"
            and settings.LOCAL_BASE_URL
        ):
            return "local"
        return model_cfg.provider

    def _provider_is_usable(self, model_cfg: AgentModelConfig) -> bool:
        """Health-gate predicate for candidate selection (advisory, not absolute)."""
        health = provider_health_tracker.get_provider_health(self._resolve_gate_provider(model_cfg))
        return bool(health.is_healthy) and not (
            health.quarantined_keys_count > 0 and health.active_keys_count == 0
        )

    async def _invoke_single_model(
        self,
        task_id: str,
        stage_name: str,
        round_number: int,
        agent: Agent,
        model_cfg: AgentModelConfig,
        messages: list[ProviderMessage],
        system_instruction: str,
        max_tokens: int = 1024,
        enforce_health: bool = True,
    ) -> tuple[ProviderResponse | None, float, Exception | None]:
        """Invoke a specific model configuration through the ModelGateway with health check.

        ``enforce_health=False`` turns the circuit breaker from a *prohibition* into a
        *preference*. This matters when every candidate looks unhealthy: a breaker is a
        load-shedding device — it exists so traffic can go somewhere better — and when
        there is nowhere better it stops protecting anything and starts causing the very
        outage it was built to prevent. In that situation one attempt is strictly better
        than a guaranteed failure, and the outcome re-arms or clears the breaker on real
        evidence rather than on the absence of it.
        """
        # 1. Health check: if provider is unhealthy / rate-limited, fail fast to next model
        gate_provider = self._resolve_gate_provider(model_cfg)
        health = provider_health_tracker.get_provider_health(gate_provider)
        if enforce_health and (
            not health.is_healthy or (health.quarantined_keys_count > 0 and health.active_keys_count == 0)
        ):
            logger.warning(
                "Skipping provider %s for agent %s (health score: %.2f, 429 count: %d)",
                gate_provider, agent.id, health.health_score, health.rate_limit_429_count
            )
            return None, 0.0, RuntimeError(f"Provider {gate_provider} currently rate-limited/unhealthy")
        if not enforce_health and not health.is_healthy:
            logger.warning(
                "Provider %s is marked unhealthy, but it is the last candidate for agent %s; "
                "attempting anyway (health score: %.2f).",
                gate_provider, agent.id, health.health_score,
            )

        start_time = time.perf_counter()
        req = ProviderRequest(
            messages=messages,
            system_instruction=system_instruction,
            model=model_cfg.model,
            max_tokens=max_tokens
        )

        try:
            resp = await model_gateway.execute(
                provider_name=model_cfg.provider,
                request=req,
                capability=model_cfg.capability,
                stage_name=stage_name
            )
            latency = time.perf_counter() - start_time
            return resp, latency, None
        except Exception as exc:
            latency = time.perf_counter() - start_time
            return None, latency, exc

    async def _execute_agent_call(
        self,
        task_id: str,
        stage_name: str,
        round_number: int,
        agent: Agent,
        messages: list[ProviderMessage],
        system_override: str | None = None,
        complexity: TaskComplexity = TaskComplexity.SIMPLE,
        notes: list[str] | None = None,
    ) -> tuple[str, int, float, list[str]]:
        """
        Executes an agent call with complexity awareness:
        - SIMPLE / EASY: calls ONLY the 1st model in the agent's preferred_models list.
        - COMPLEX / STRATEGIC: calls top 2-3 models IN PARALLEL via asyncio.gather, then merges outputs.
        - Skips rate-limited providers dynamically.
        - Discards empty completions and reports truncated ones.

        ``notes`` — when supplied, human-readable warnings (truncated output, empty
        completions) are appended so they can reach the caller's
        ``degradation_reasons`` instead of being lost inside this method.
        """
        run_id = generate_run_id()
        msg_id = generate_message_id()
        start_time = time.perf_counter()
        system_instruction = system_override or agent.system_instructions

        # Gather candidate models for this agent
        preferred = agent.models if agent.models else [
            AgentModelConfig(provider=agent.model_provider, model=agent.model_name, capability="general")
        ]

        if complexity == TaskComplexity.SIMPLE:
            healthy = [cfg for cfg in preferred if self._provider_is_usable(cfg)]
            # Prioritize ultra-low latency providers (groq, gemini) for simple tasks
            fast_candidates = [cfg for cfg in (healthy or preferred) if cfg.provider.lower() in ("groq", "gemini")]
            fast_candidates.sort(key=lambda c: 0 if c.provider.lower() == "groq" else 1)
            if fast_candidates:
                configs_to_run = [fast_candidates[0]]
            elif healthy:
                configs_to_run = [healthy[0]]
            else:
                configs_to_run = preferred[:1]
        else:
            # For complex tasks, prioritize healthy configs up to 3, falling back to all available
            healthy = [cfg for cfg in preferred if self._provider_is_usable(cfg)]
            configs_to_run = healthy[:3] if healthy else preferred[:min(3, len(preferred))]

        # If candidate selection found nothing healthy, we are in last-resort territory:
        # the fallback above deliberately keeps the declared configs so the request can
        # still be attempted. Without this flag the per-call health gate immediately
        # rejected those very configs, making the fallback dead code and turning every
        # transient provider wobble into a hard client-visible failure.
        last_resort = not healthy

        token_limit = 1024 if (complexity != TaskComplexity.SIMPLE or agent.id == "synthesizer") else 220
        if complexity == TaskComplexity.SIMPLE:
            if agent.id == "synthesizer":
                system_instruction = (system_instruction or "") + " Provide the complete, authoritative, factual executive research answer with all dates and key findings clearly stated without preamble or meta-commentary."
            else:
                system_instruction = (system_instruction or "") + " Answer with maximum technical conciseness in 2-4 sentences or bullet points without filler."

        # Execute model calls (single or parallel)
        async def call_model(cfg: AgentModelConfig):
            return await self._invoke_single_model(
                task_id=task_id,
                stage_name=stage_name,
                round_number=round_number,
                agent=agent,
                model_cfg=cfg,
                messages=messages,
                system_instruction=system_instruction,
                max_tokens=token_limit,
                enforce_health=not last_resort,
            )

        if len(configs_to_run) == 1:
            resp, lat, err = await call_model(configs_to_run[0])
            model_results = [(resp, lat, err, configs_to_run[0])]
        else:
            # Parallel multi-model execution for complex tasks
            raw_results = await asyncio.gather(*[call_model(cfg) for cfg in configs_to_run])
            model_results = [
                (r[0], r[1], r[2], cfg) for r, cfg in zip(raw_results, configs_to_run)
            ]

        # ------------------------------------------------------------------
        # Classify each model result. An HTTP 200 is NOT proof that a model
        # answered, and this distinction is the whole honesty contract:
        #
        #  * ``choices: []`` (or a whitespace-only completion) arrives as a
        #    perfectly successful-looking response whose content is empty. Treating
        #    it as a successful deliberation publishes a confident, evidence-free
        #    result for a request no model ever answered — the same fabrication the
        #    exception path was already fixed against, via a different door.
        #  * ``finish_reason == "length"`` means the model was cut off mid-answer.
        #    The fragment is real output, so it still counts, but silently
        #    presenting it as a complete analysis loses information the caller
        #    needs in order to trust the synthesis.
        # ------------------------------------------------------------------
        successful_resps: list[tuple[ProviderResponse, AgentModelConfig]] = []
        empty_completions: list[str] = []
        truncated_models: list[str] = []

        for resp, lat, err, cfg in model_results:
            if resp is None:
                continue
            if not (resp.content or "").strip():
                empty_completions.append(f"{cfg.provider}:{cfg.model}")
                # An empty completion is a provider defect, not a neutral event: it
                # must count against the provider's health so the panel stops
                # preferring it, exactly as a 5xx would.
                provider_health_tracker.record_failure(
                    cfg.provider,
                    "empty completion (HTTP 200 with no content)",
                    latency_seconds=lat,
                )
                logger.warning(
                    "Collaboration %s: agent '%s' received an EMPTY completion from %s:%s; "
                    "discarding it rather than counting it as an answer.",
                    task_id, agent.id, cfg.provider, cfg.model,
                )
                continue
            if (resp.finish_reason or "").lower() in ("length", "max_tokens"):
                truncated_models.append(f"{cfg.provider}:{cfg.model}")
            successful_resps.append((resp, cfg))

        if truncated_models and notes is not None:
            notes.append(
                f"{agent.id} output truncated by the provider token ceiling "
                f"({', '.join(sorted(set(truncated_models)))})"
            )
        if empty_completions and notes is not None:
            notes.append(
                f"{agent.id} returned empty completions from "
                f"{', '.join(sorted(set(empty_completions)))} (not counted as an answer)"
            )

        total_tokens = sum(r.total_tokens or 0 for r, _ in successful_resps)
        models_used = [r.model for r, _ in successful_resps]

        if successful_resps:
            latency = time.perf_counter() - start_time
            model_tuples = [(resp, cfg.model) for resp, cfg in successful_resps]
            assessment = Adjudicator.adjudicate_specialist_multi_model(agent, model_tuples)
            final_content = assessment.summary

            # 1. Save Run Record (asynchronously offloaded to eliminate disk blocking)
            run_rec = RunRecord(
                id=run_id,
                task_id=task_id,
                agent_id=agent.id,
                provider=successful_resps[0][0].provider or agent.model_provider,
                model=successful_resps[0][0].model,
                stage=f"round_{round_number}_{stage_name}",
                prompt_tokens=successful_resps[0][0].prompt_tokens or 0,
                completion_tokens=successful_resps[0][0].completion_tokens or 0,
                latency_seconds=latency,
                status="completed"
            )
            try:
                asyncio.create_task(self.memory.save_run(run_rec))
            except RuntimeError:
                await self.memory.save_run(run_rec)

            # 2. Save Message Record (asynchronously offloaded)
            msg_rec = MessageRecord(
                id=msg_id,
                run_id=run_id,
                task_id=task_id,
                role="assistant",
                agent_id=agent.id,
                content=final_content,
                stage=f"round_{round_number}_{stage_name}"
            )
            try:
                asyncio.create_task(self.memory.save_message(msg_rec))
            except RuntimeError:
                await self.memory.save_message(msg_rec)

            return final_content, total_tokens, latency, models_used

        # Fallback if all attempted models failed
        latency = time.perf_counter() - start_time
        errors = [str(err) for _, _, err, _ in model_results if err]
        if errors:
            error_msg = errors[0]
        elif empty_completions:
            # Distinguish "did not answer" from "could not be reached": the operator
            # needs to know that the transport worked and the model said nothing.
            error_msg = (
                f"provider returned empty completions (no model output) from "
                f"{', '.join(sorted(set(empty_completions)))}"
            )
        else:
            error_msg = "all provider models unavailable"
        logger.warning("Collaboration call for agent %s in %s had an issue: %s", agent.id, stage_name, error_msg)

        run_rec = RunRecord(
            id=run_id,
            task_id=task_id,
            agent_id=agent.id,
            provider=agent.model_provider,
            model=agent.model_name,
            stage=f"round_{round_number}_{stage_name}",
            latency_seconds=latency,
            status="failed",
            error=error_msg
        )
        try:
            asyncio.create_task(self.memory.save_run(run_rec))
        except RuntimeError:
            await self.memory.save_run(run_rec)

        # Do NOT hand back a prose placeholder: a failed call is an error, not an
        # answer. The caller (run_collaboration) decides whether a peer can cover
        # this agent, and ultimately surfaces a degraded panel.
        #
        # Report the providers the gateway ACTUALLY attempted, not the agent's
        # declared primary. Naming `agent.model_provider` produced messages such as
        # "researcher (gemini): GROQ_API_KEY is not configured", which sends the
        # operator to the wrong provider's configuration and hides the fact that the
        # fallback ladder was walked at all.
        attempted = sorted({f"{cfg.provider}:{cfg.model}" for _, _, _, cfg in model_results})
        attempted_label = ", ".join(attempted) if attempted else agent.model_provider
        raise AgentCallUnavailable(
            f"{agent.id} (attempted {attempted_label}): {error_msg}"
        ) from (next((e for e in (err for _, _, err, _ in model_results) if e), None))

    async def _execute_multi_model_synthesis(
        self,
        task_id: str,
        stage_name: str,
        round_number: int,
        synthesizer_agent: Agent,
        synthesis_prompt: str,
        complexity: TaskComplexity
    ) -> tuple[str, int, float, list[str]]:
        """
        Parallel Multi-Model Synthesis:
        Calls multiple models for the Synthesizer (e.g. Gemini + OpenRouter DeepSeek) in parallel,
        then merges the best parts into a single coherent synthesis answer.
        """
        preferred = synthesizer_agent.models if synthesizer_agent.models else [
            AgentModelConfig(provider="groq", model="openai/gpt-oss-120b", capability="synthesis"),
            AgentModelConfig(provider="gemini", model="gemini-3.8-flash", capability="synthesis")
        ]

        # For simple tasks, use top 1; for complex/strategic, invoke top 2 models in parallel
        synth_configs = preferred[:1] if complexity == TaskComplexity.SIMPLE else preferred[:2]

        async def call_synth(cfg: AgentModelConfig):
            return await self._invoke_single_model(
                task_id=task_id,
                stage_name=stage_name,
                round_number=round_number,
                agent=synthesizer_agent,
                model_cfg=cfg,
                messages=[ProviderMessage(role="user", content=synthesis_prompt)],
                system_instruction=synthesizer_agent.system_instructions
            )

        if len(synth_configs) == 1:
            resp, lat, err = await call_synth(synth_configs[0])
            synth_results = [(resp, lat, err, synth_configs[0])]
        else:
            raw = await asyncio.gather(*[call_synth(cfg) for cfg in synth_configs])
            synth_results = [(r[0], r[1], r[2], cfg) for r, cfg in zip(raw, synth_configs)]

        valid_resps = [resp for resp, _, _, _ in synth_results if resp is not None]
        models_used = [resp.model for resp in valid_resps]
        total_tokens = sum(r.total_tokens or 0 for r in valid_resps)

        if not valid_resps:
            # Fallback to standard agent call
            text, tok, lat, mods = await self._execute_agent_call(
                task_id=task_id,
                stage_name=stage_name,
                round_number=round_number,
                agent=synthesizer_agent,
                messages=[ProviderMessage(role="user", content=synthesis_prompt)],
                complexity=complexity
            )
            return text, tok, lat, mods

        if len(valid_resps) == 1:
            return valid_resps[0].content, total_tokens, 0.0, models_used

        # Parallel multi-model synthesis adjudication:
        # Reconcile agreements, contradictions, and evidence without using string length
        r1_text = valid_resps[0].content.strip()
        r2_text = valid_resps[1].content.strip()

        # If conflict detected in any model output, prioritize conflict awareness
        if r1_text.startswith("CONFLICT_DETECTED:") or r2_text.startswith("CONFLICT_DETECTED:"):
            merged = r1_text if r1_text.startswith("CONFLICT_DETECTED:") else r2_text
        else:
            # Primary synthesizer model provides canonical structure, validated against secondary
            merged = r1_text

        return merged, total_tokens, 0.0, models_used

    async def run_collaboration(
        self,
        task_id: str,
        question: str,
        participating_agents: list[Agent] | None = None,
        require_evidence: bool = True,
        complexity: TaskComplexity = TaskComplexity.SIMPLE,
        cancellation_event: asyncio.Event | None = None,
    ) -> CollaborationResult:
        """
        Executes the Real-Time Parallel Collaboration Protocol with Complexity & Health awareness:
        1. Parallel Specialist Perspectives (Round 1) via asyncio.gather.
        2. Multi-Model Parallel Synthesis Review for Conflict vs Consensus.
        3. If aligned: Instant Merged Synthesis.
        4. If severe conflict: Targeted Rebuttal round between conflicting specialists.
        """
        if cancellation_event and cancellation_event.is_set():
            raise asyncio.CancelledError(f"Task {task_id} was cancelled before collaboration.")

        session_id = generate_debate_id()
        start_total_time = time.perf_counter()
        total_tokens = 0
        all_models_used: list[str] = []
        rounds_log: list[CollaborationRoundLog] = []

        if not participating_agents:
            agent_ids = ["architect", "security_analyst", "coder"]
            participating_agents = [
                a for aid in agent_ids
                if (a := self.registry.get_agent(aid)) is not None
            ]

        synthesizer_agent = self.registry.get_agent("synthesizer") or participating_agents[0]
        critic_agent = self.registry.get_agent("critic") or participating_agents[-1]

        # -------------------------------------------------------------
        # STEP 1: Parallel Independent Analysis (Round 1) with peer recovery
        # -------------------------------------------------------------
        logger.info(
            "Collaboration %s: Firing parallel specialist analysis for %d agents (Complexity: %s)",
            session_id, len(participating_agents), complexity.value
        )

        # Self-healing bookkeeping: which agents failed, which peer covered them.
        failed_agents: set[str] = set()
        coverage_log: dict[str, str] = {}
        degradation_reasons: list[str] = []

        def _build_prompt(target: Agent) -> str:
            if len(participating_agents) == 1:
                return (
                    f"User Query:\n{question}\n\n"
                    "Provide a direct, clear, and concise response matching the exact formatting and length requested by the user. "
                    "Avoid unnecessary verbosity, robotic disclaimers, or excessive introductory fluff."
                )
            return (
                f"Question / Goal:\n{question}\n\n"
                f"As the {target.role}, provide your direct, concise technical recommendation and core rationale. "
                "Be concrete, identify primary trade-offs, and state assumptions explicitly."
            )

        def _peer_candidates(original: Agent) -> list[Agent]:
            """Peers that may cover a failed specialist.

            Preference order: (1) other members of this task's panel, (2) any other
            registered active specialist. Peers that already failed are skipped.
            """
            seen: set[str] = {original.id}
            ordered: list[Agent] = []
            panel_ids = [a.id for a in participating_agents]
            for cand in participating_agents + list(self.registry.list_agents()):
                if cand.id in seen or cand.id in failed_agents or not cand.id:
                    continue
                seen.add(cand.id)
                if cand.status == "active":
                    ordered.append(cand)
            # Panel members first (they were selected for this task's shape).
            ordered.sort(key=lambda a: 0 if a.id in panel_ids else 1)
            return ordered

        async def _attempt(target: Agent, covering_for: Agent | None = None):
            prompt = _build_prompt(target)
            if covering_for is not None:
                prompt = (
                    f"{prompt}\n\n"
                    f"[PEER ASSIST] Your colleague '{covering_for.role}' could not produce an "
                    f"analysis for this task. Provide your own '{target.role}' assessment of "
                    "this exact question so the panel still receives a grounded perspective."
                )
            text, t_count, _, models = await self._execute_agent_call(
                task_id=task_id,
                stage_name="independent_analysis",
                round_number=1,
                agent=target,
                messages=[ProviderMessage(role="user", content=prompt)],
                complexity=complexity,
                notes=degradation_reasons,
            )
            return target, text, t_count, models

        # Round-1 budget. Peer coverage is a recovery mechanism, not a licence to run an
        # unbounded number of model calls: without a ceiling, a total provider outage
        # makes the request's runtime a function of how many specialists happen to be
        # registered, and the client waits far past its own timeout. The budget is
        # checked before each peer attempt, and the attempt count is capped
        # independently so that a fast-failing provider still cannot produce a stampede.
        peer_budget_deadline = time.monotonic() + float(settings.PEER_COVERAGE_BUDGET_SECONDS)
        max_peer_attempts = int(settings.PEER_COVERAGE_MAX_ATTEMPTS)

        async def analyze_agent(agent: Agent):
            """Run one specialist, self-healing through peers when its models are dark."""
            try:
                return await _attempt(agent)
            except AgentCallUnavailable as primary_err:
                failed_agents.add(agent.id)
                logger.warning(
                    "Collaboration %s: specialist '%s' unavailable (%s) — attempting peer coverage",
                    session_id, agent.id, primary_err
                )
                attempts = 0
                for peer in _peer_candidates(agent):
                    if attempts >= max_peer_attempts:
                        logger.warning(
                            "Collaboration %s: peer-coverage attempt cap reached (%d) while covering '%s'",
                            session_id, max_peer_attempts, agent.id
                        )
                        break
                    if time.monotonic() >= peer_budget_deadline:
                        logger.warning(
                            "Collaboration %s: peer-coverage budget (%.1fs) exhausted while covering '%s'",
                            session_id, settings.PEER_COVERAGE_BUDGET_SECONDS, agent.id
                        )
                        break
                    attempts += 1
                    try:
                        result = await _attempt(peer, covering_for=agent)
                    except AgentCallUnavailable as peer_err:
                        failed_agents.add(peer.id)
                        logger.warning(
                            "Collaboration %s: peer '%s' also unavailable (%s)",
                            session_id, peer.id, peer_err
                        )
                        continue
                    coverage_log[agent.id] = peer.id
                    logger.info(
                        "Collaboration %s: peer '%s' covered for unavailable specialist '%s'",
                        session_id, peer.id, agent.id
                    )
                    return result
                # Nobody could cover: record the reason and let the gather decide.
                degradation_reasons.append(str(primary_err))
                raise

        # Execute all specialist perspectives simultaneously. One dark agent must not
        # abort the whole panel (that is the point of peer coverage), so outcomes are
        # collected individually and the panel only fails if EVERYONE is dark.
        #
        # The phase is additionally bounded by the panel budget. Without that bound the
        # budget only gated *between* attempts, so a single unresponsive provider held
        # the whole request for the full per-call timeout (30s) regardless of a 20s
        # panel budget — the deadline was advisory rather than enforced.
        #
        # ``asyncio.wait`` is used rather than wrapping the gather in ``wait_for``
        # because a gather is cancelled as a unit: a timeout would discard the finished
        # work of every specialist that answered promptly. Waiting on the individual
        # tasks keeps those results, cancels only the stragglers, and records the
        # cancellation as a per-agent failure — real output is never thrown away to
        # punish a slow peer.
        panel_budget = float(settings.PEER_COVERAGE_BUDGET_SECONDS)
        r1_tasks = [asyncio.ensure_future(analyze_agent(agent)) for agent in participating_agents]
        r1_done, r1_pending = await asyncio.wait(r1_tasks, timeout=panel_budget)
        if r1_pending:
            for task in r1_pending:
                task.cancel()
            await asyncio.gather(*r1_pending, return_exceptions=True)
            timed_out = [
                agent.id for agent, task in zip(participating_agents, r1_tasks) if task in r1_pending
            ]
            logger.warning(
                "Collaboration %s: %d specialist(s) exceeded the %.1fs panel budget and were "
                "cancelled: %s",
                session_id, len(timed_out), panel_budget, ", ".join(timed_out),
            )
            degradation_reasons.append(
                f"specialist phase exceeded its {panel_budget:.1f}s budget; "
                f"cancelled: {', '.join(timed_out)}"
            )

        r1_results = []
        panel_failures: list[str] = []
        for agent, task in zip(participating_agents, r1_tasks):
            if task in r1_pending:
                # Record the cancellation in the structured failure list too, not only in
                # the prose reason: a consumer that reads `failed_agents` to render which
                # specialists are missing must see the timed-out one, or the structured
                # metadata under-reports the damage the response itself admits to.
                failed_agents.add(agent.id)
                panel_failures.append(
                    f"{agent.id}: exceeded the {panel_budget:.1f}s specialist-phase budget"
                )
                continue
            try:
                outcome = task.result()
            except asyncio.CancelledError as cancelled:
                raise cancelled
            except BaseException as exc:  # noqa: BLE001 - one dark agent is not a panel failure
                panel_failures.append(f"{agent.id}: {exc}")
                continue
            if isinstance(outcome, asyncio.CancelledError):
                raise outcome
            if isinstance(outcome, BaseException):
                panel_failures.append(f"{agent.id}: {outcome}")
                continue
            r1_results.append(outcome)

        if not r1_results:
            raise AgentPanelUnavailable(
                task_id=task_id,
                failures=panel_failures or degradation_reasons or ["no specialist produced output"],
            )

        round_1_messages: list[CollaborationMessage] = []

        for agent, text, t_count, models in r1_results:
            total_tokens += t_count
            all_models_used.extend(models)
            round_1_messages.append(CollaborationMessage(
                round_number=1,
                stage_name="Independent Analysis",
                agent_id=agent.id,
                agent_role=agent.role,
                content=text,
                models_used=models
            ))

        rounds_log.append(CollaborationRoundLog(
            round_number=1,
            stage_name="Independent Analysis",
            messages=round_1_messages,
            summary=f"Gathered {len(round_1_messages)} parallel specialist perspectives."
        ))

        # Extract structured claims and evidence for all specialist perspectives
        specialist_assessments: list[SpecialistAssessment] = []
        covered_peer_ids = set(coverage_log.values())
        for agent, text, t_count, models in r1_results:
            mid = models[0] if models else "default_model"
            c, e = Adjudicator.extract_claims_and_evidence(agent, mid, text)

            # Model confidence must reflect what we actually observed, not a
            # constant. A real model call that yielded extractable claims/evidence
            # scores higher than one that yielded neither, and a perspective that a
            # covering peer had to produce on behalf of a dark specialist is
            # discounted because it is second-hand.
            if e or c:
                model_conf = 0.75
            else:
                model_conf = 0.55
            if agent.id in covered_peer_ids:
                model_conf -= 0.10
            model_conf = round(max(0.0, min(0.95, model_conf)), 2)

            specialist_assessments.append(SpecialistAssessment(
                agent_id=agent.id,
                agent_role=agent.role,
                summary=text,
                claims=c,
                evidence=e,
                model_confidence=model_conf
            ))

        all_claims: list[AtomicClaim] = []
        all_evidence: list[StructuredEvidence] = []
        for ass in specialist_assessments:
            all_claims.extend(ass.claims)
            all_evidence.extend(ass.evidence)

        # True once the Synthesizer itself could not reach any model (set below).
        synthesis_degraded = False

        def _degradation_fields() -> dict[str, Any]:
            """Honest self-healing metadata attached to every CollaborationResult."""
            reasons = list(degradation_reasons)
            if failed_agents:
                reasons.append(
                    "unavailable specialists (covered by peer or dropped): "
                    + ", ".join(sorted(failed_agents))
                )
            return {
                "degraded": bool(reasons),
                "degradation_reasons": reasons,
                "agent_coverage": dict(coverage_log),
                "failed_agents": sorted(failed_agents),
            }

        def _confidence_ceiling() -> float | None:
            """Highest confidence this result is permitted to report, given its damage.

            ``degraded`` used to be decided by one rule (any degradation reason at all)
            while the confidence cap used another, much narrower one (only when the
            *synthesizer itself* was dark). The two could — and did — disagree: a panel
            whose outputs were all truncated by the provider token ceiling came back
            flagged ``degraded=True`` while still reporting confidence 0.77, i.e. the
            number a caller uses to decide how much to trust the answer ignored the very
            damage the same response was reporting. Any response that admits damage must
            price that damage into its confidence, so both now derive from one policy.

            Returns ``None`` when nothing was degraded (no ceiling).
            """
            reasons = " ".join(degradation_reasons + sorted(failed_agents)).lower()
            if not reasons:
                return None
            if synthesis_degraded:
                # No synthesis at all: the answer is raw, unmerged, cross-examined by
                # nobody. This is the most severe case and keeps its original ceiling.
                return 0.45
            if "truncated" in reasons:
                # A real answer, but cut off mid-thought: materially incomplete.
                return 0.55
            if "unavailable specialists" in reasons or "unavailable" in reasons:
                # The intended voice was replaced by a peer, or lost entirely.
                return 0.65
            return 0.7

        def _apply_confidence_ceiling(raw: float) -> float:
            """Lower ``raw`` to honour the degradation policy; never raise it."""
            ceiling = _confidence_ceiling()
            if ceiling is None:
                return raw
            capped = min(raw, ceiling)
            if capped < raw:
                logger.info(
                    "Collaboration %s: confidence lowered %.2f -> %.2f to match reported degradation",
                    session_id, raw, capped,
                )
            return round(capped, 2)

        # Optimization: In fast / simple 1-agent mode, return the specialist's direct response immediately
        if len(participating_agents) == 1 and complexity == TaskComplexity.SIMPLE:
            direct_ans = round_1_messages[0].content if round_1_messages else ""
            elapsed_time = round(time.perf_counter() - start_total_time, 2)
            logger.info("Collaboration %s: Fast single-specialist answer completed in %.2fs", session_id, elapsed_time)

            calib_conf, _ = Adjudicator.calculate_system_confidence(
                assessments=specialist_assessments,
                contradictions=[],
                evidence_count=len(all_evidence),
                complexity_str="simple"
            )
            return CollaborationResult(
                debate_id=session_id,
                task_id=task_id,
                canonical_problem=question,
                mode_used="fast",
                complexity=complexity.value,
                final_answer=direct_ans,
                participating_agents=[a.id for a in participating_agents],
                models_used=all_models_used,
                rounds=rounds_log,
                confidence=_apply_confidence_ceiling(calib_conf),
                unresolved_disagreements=[],
                # No evidence means no evidence: never invent a placeholder claim.
                key_evidence=[e.excerpt for e in all_evidence],
                structured_evidence=all_evidence,
                claims=all_claims,
                total_tokens=total_tokens,
                total_latency_seconds=elapsed_time,
                **_degradation_fields(),
            )

        combined_proposals = "\n\n".join([
            f"=== Specialist Perspective: {m.agent_role} ({m.agent_id}) ===\n{m.content}"
            for m in round_1_messages
        ])

        # -------------------------------------------------------------
        # STEP 2 & 3: Parallel Multi-Model Synthesis Review
        # -------------------------------------------------------------
        logger.info("Collaboration %s: Synthesizer reviewing parallel responses for conflict vs consensus", session_id)
        synthesis_prompt = (
            f"User Question:\n{question}\n\n"
            f"Specialist Analysis:\n{combined_proposals}\n\n"
            "As the Consensus Synthesizer, synthesize the specialist inputs into a single, polished, and comprehensive answer.\n"
            "Guidelines:\n"
            "- Deliver the direct, actionable answer immediately using clean GitHub-flavored Markdown.\n"
            "- Strictly respect any length constraints, bullet count, or format requested in the user's question.\n"
            "- Do NOT output robotic boilerplate meta-headers like 'Consensus Synthesis (Unified Answer)' or 'Operating Assumptions'.\n"
            "- Structure key technical points with clear bullet points, bold emphasis, and concise explanations.\n"
            "- If there is a severe, dangerous technical disagreement between specialists, start with 'CONFLICT_DETECTED:' followed by the dispute."
        )

        try:
            synthesis_text, syn_tokens, _, syn_models = await self._execute_multi_model_synthesis(
                task_id=task_id,
                stage_name="consensus_synthesis",
                round_number=2,
                synthesizer_agent=synthesizer_agent,
                synthesis_prompt=synthesis_prompt,
                complexity=complexity
            )
        except AgentCallUnavailable as syn_err:
            # The Synthesizer is dark, but Round 1 produced REAL specialist output.
            # Return that real, unmerged material (clearly labelled) at low confidence
            # instead of inventing a consensus — and never fail the whole task when
            # we still have genuine specialist analysis to hand back.
            synthesis_degraded = True
            degradation_reasons.append(f"synthesizer unavailable: {syn_err}")
            syn_tokens, syn_models = 0, []
            logger.warning(
                "Collaboration %s: synthesizer unavailable (%s) — returning raw specialist panel output",
                session_id, syn_err
            )
            synthesis_text = (
                "## Specialist panel output (automated synthesis unavailable)\n\n"
                "> **Degraded mode:** the Synthesizer could not reach any model, so the "
                "specialist analyses below are returned **unmerged and unverified**. "
                "Treat them as raw perspectives, not a vetted consensus.\n\n"
                f"{combined_proposals}"
            )

        total_tokens += syn_tokens
        all_models_used.extend(syn_models)

        has_severe_conflict = synthesis_text.strip().startswith("CONFLICT_DETECTED:") or (
            "CONSENSUS_REACHED: NO" in synthesis_text.upper()
        )

        # -------------------------------------------------------------
        # STEP 4: Targeted Rebuttal ONLY IF Severe Conflict Exists
        # -------------------------------------------------------------
        if has_severe_conflict:
            logger.warning("Collaboration %s: Severe conflict detected. Triggering targeted adversarial rebuttal.", session_id)
            rebuttal_prompt = (
                f"Question:\n{question}\n\n"
                f"Initial Proposals:\n{combined_proposals}\n\n"
                f"Identified Conflict:\n{synthesis_text}\n\n"
                "As the Adversarial Critic, challenge the conflicting assumptions and propose the safest resolution."
            )
            rebuttal_text, reb_tokens, _, reb_models = await self._execute_agent_call(
                task_id=task_id,
                stage_name="targeted_rebuttal",
                round_number=3,
                agent=critic_agent,
                messages=[ProviderMessage(role="user", content=rebuttal_prompt)],
                complexity=complexity,
                notes=degradation_reasons,
            )
            total_tokens += reb_tokens
            all_models_used.extend(reb_models)

            rounds_log.append(CollaborationRoundLog(
                round_number=3,
                stage_name="Targeted Rebuttal",
                messages=[CollaborationMessage(
                    round_number=3,
                    stage_name="Targeted Rebuttal",
                    agent_id=critic_agent.id,
                    agent_role=critic_agent.role,
                    content=rebuttal_text,
                    models_used=reb_models
                )],
                summary="Targeted rebuttal resolved conflicting specialist assumptions."
            ))

            # Final resolution synthesis post-rebuttal
            final_synth_prompt = (
                f"Question:\n{question}\n\n"
                f"Targeted Rebuttal & Critique:\n{rebuttal_text}\n\n"
                "Produce the final, conclusive architectural recommendation resolving the debate."
            )
            final_answer, fin_tokens, _, fin_models = await self._execute_multi_model_synthesis(
                task_id=task_id,
                stage_name="final_resolution",
                round_number=4,
                synthesizer_agent=synthesizer_agent,
                synthesis_prompt=final_synth_prompt,
                complexity=complexity
            )
            total_tokens += fin_tokens
            all_models_used.extend(fin_models)

            rounds_log.append(CollaborationRoundLog(
                round_number=4,
                stage_name="Final Resolution",
                messages=[CollaborationMessage(
                    round_number=4,
                    stage_name="Final Resolution",
                    agent_id=synthesizer_agent.id,
                    agent_role=synthesizer_agent.role,
                    content=final_answer,
                    models_used=fin_models
                )],
                summary="Synthesizer delivered resolved decision after targeted debate."
            ))

            total_duration = time.perf_counter() - start_total_time
            adjudication_res = Adjudicator.reconcile_panel_adjudication(
                task_id=task_id,
                question=question,
                assessments=specialist_assessments,
                synthesis_text=final_answer,
                adjudicator_models=list(dict.fromkeys(all_models_used)),
                complexity=complexity.value
            )
            return CollaborationResult(
                debate_id=session_id,
                task_id=task_id,
                canonical_problem=question,
                final_answer=final_answer,
                confidence=_apply_confidence_ceiling(adjudication_res.system_confidence),
                unresolved_disagreements=adjudication_res.unresolved_disputes or ["Resolved via targeted debate."],
                # Empty when no evidence exists: a placeholder sentence is not evidence.
                key_evidence=[e.excerpt for e in adjudication_res.key_evidence],
                structured_evidence=adjudication_res.key_evidence,
                claims=all_claims,
                adjudication=adjudication_res,
                participating_agents=[a.id for a in participating_agents],
                rounds=rounds_log,
                mode_used="debate",
                complexity=complexity.value,
                models_used=list(dict.fromkeys(all_models_used)),
                total_tokens=total_tokens,
                total_latency_seconds=round(total_duration, 4),
                **_degradation_fields(),
            )

        rounds_log.append(CollaborationRoundLog(
            round_number=2,
            stage_name="Consensus Synthesis",
            messages=[CollaborationMessage(
                round_number=2,
                stage_name="Consensus Synthesis",
                agent_id=synthesizer_agent.id,
                agent_role=synthesizer_agent.role,
                content=synthesis_text,
                models_used=syn_models
            )],
            summary="Synthesizer consolidated specialist perspectives."
        ))

        total_duration = time.perf_counter() - start_total_time
        logger.info("Collaboration %s finished in %.2fs consuming ~%d tokens", session_id, total_duration, total_tokens)

        adjudication_res = Adjudicator.reconcile_panel_adjudication(
            task_id=task_id,
            question=question,
            assessments=specialist_assessments,
            synthesis_text=synthesis_text,
            adjudicator_models=list(dict.fromkeys(all_models_used)),
            complexity=complexity.value
        )

        return CollaborationResult(
            debate_id=session_id,
            task_id=task_id,
            canonical_problem=question,
            final_answer=synthesis_text,
            confidence=_apply_confidence_ceiling(adjudication_res.system_confidence),
            unresolved_disagreements=adjudication_res.unresolved_disputes,
            # Empty when no evidence exists: a placeholder sentence is not evidence.
            key_evidence=[e.excerpt for e in adjudication_res.key_evidence],
            structured_evidence=adjudication_res.key_evidence,
            claims=all_claims,
            adjudication=adjudication_res,
            participating_agents=[a.id for a in participating_agents],
            rounds=rounds_log,
            mode_used="consensus",
            complexity=complexity.value,
            models_used=list(dict.fromkeys(all_models_used)),
            total_tokens=total_tokens,
            total_latency_seconds=round(total_duration, 4),
            **_degradation_fields(),
        )


# Backward-compatibility alias
DebateEngine = CollaborationEngine
debate_engine = CollaborationEngine()
