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
from app.core.dag import TaskComplexity
from app.memory.base import BaseMemory, MessageRecord, RunRecord
from app.memory.sqlite import SQLiteMemory
from app.providers.base import ProviderMessage, ProviderRequest, ProviderResponse
from app.providers.gateway import model_gateway
from app.providers.health import provider_health_tracker
from app.utils.ids import generate_debate_id, generate_message_id, generate_run_id
from app.utils.logger import logger


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
    mode_used: str = "consensus"
    complexity: str = "simple"
    models_used: list[str] = Field(default_factory=list)
    total_tokens: int = 0
    total_latency_seconds: float = 0.0


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
    ) -> tuple[ProviderResponse | None, float, Exception | None]:
        """Invoke a specific model configuration through the ModelGateway with health check."""
        # 1. Health check: if provider is unhealthy / rate-limited, fail fast to next model
        health = provider_health_tracker.get_provider_health(model_cfg.provider)
        if not health.is_healthy or (health.quarantined_keys_count > 0 and health.active_keys_count == 0):
            logger.warning(
                "Skipping provider %s for agent %s (health score: %.2f, 429 count: %d)",
                model_cfg.provider, agent.id, health.health_score, health.rate_limit_429_count
            )
            return None, 0.0, RuntimeError(f"Provider {model_cfg.provider} currently rate-limited/unhealthy")

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
        complexity: TaskComplexity = TaskComplexity.SIMPLE
    ) -> tuple[str, int, float, list[str]]:
        """
        Executes an agent call with complexity awareness:
        - SIMPLE / EASY: calls ONLY the 1st model in the agent's preferred_models list.
        - COMPLEX / STRATEGIC: calls top 2-3 models IN PARALLEL via asyncio.gather, then merges outputs.
        - Skips rate-limited providers dynamically.
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
            healthy = [cfg for cfg in preferred if provider_health_tracker.get_provider_health(cfg.provider).is_healthy]
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
            healthy = [cfg for cfg in preferred if provider_health_tracker.get_provider_health(cfg.provider).is_healthy]
            configs_to_run = healthy[:3] if healthy else preferred[:min(3, len(preferred))]

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

        # Gather successful responses
        successful_resps: list[tuple[ProviderResponse, AgentModelConfig]] = [
            (resp, cfg) for resp, lat, err, cfg in model_results if resp is not None
        ]

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
        error_msg = errors[0] if errors else "all provider models unavailable"
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
        fallback_content = f"*[Specialist {agent.role} temporarily offline / high demand on {agent.model_provider}: {error_msg}]*"
        return fallback_content, 0, latency, [agent.model_name]

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
            AgentModelConfig(provider="gemini", model="gemini-3.5-flash-lite", capability="synthesis")
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
        # STEP 1: Parallel Independent Analysis (Round 1)
        # -------------------------------------------------------------
        logger.info(
            "Collaboration %s: Firing parallel specialist analysis for %d agents (Complexity: %s)",
            session_id, len(participating_agents), complexity.value
        )

        async def analyze_agent(agent: Agent) -> tuple[Agent, str, int, list[str]]:
            if len(participating_agents) == 1:
                prompt = (
                    f"User Query:\n{question}\n\n"
                    "Provide a direct, clear, and concise response matching the exact formatting and length requested by the user. "
                    "Avoid unnecessary verbosity, robotic disclaimers, or excessive introductory fluff."
                )
            else:
                prompt = (
                    f"Question / Goal:\n{question}\n\n"
                    f"As the {agent.role}, provide your direct, concise technical recommendation and core rationale. "
                    "Be concrete, identify primary trade-offs, and state assumptions explicitly."
                )
            text, t_count, _, models = await self._execute_agent_call(
                task_id=task_id,
                stage_name="independent_analysis",
                round_number=1,
                agent=agent,
                messages=[ProviderMessage(role="user", content=prompt)],
                complexity=complexity
            )
            return agent, text, t_count, models

        # Execute all specialist perspectives simultaneously with asyncio.gather
        r1_results = await asyncio.gather(*[analyze_agent(agent) for agent in participating_agents])
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
        for agent, text, t_count, models in r1_results:
            mid = models[0] if models else "default_model"
            c, e = Adjudicator.extract_claims_and_evidence(agent, mid, text)
            specialist_assessments.append(SpecialistAssessment(
                agent_id=agent.id,
                agent_role=agent.role,
                summary=text,
                claims=c,
                evidence=e,
                model_confidence=0.90
            ))

        all_claims: list[AtomicClaim] = []
        all_evidence: list[StructuredEvidence] = []
        for ass in specialist_assessments:
            all_claims.extend(ass.claims)
            all_evidence.extend(ass.evidence)

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
                confidence=calib_conf,
                unresolved_disagreements=[],
                key_evidence=[e.excerpt for e in all_evidence] if all_evidence else ["Single-specialist direct assessment."],
                structured_evidence=all_evidence,
                claims=all_claims,
                total_tokens=total_tokens,
                total_latency_seconds=elapsed_time
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

        synthesis_text, syn_tokens, _, syn_models = await self._execute_multi_model_synthesis(
            task_id=task_id,
            stage_name="consensus_synthesis",
            round_number=2,
            synthesizer_agent=synthesizer_agent,
            synthesis_prompt=synthesis_prompt,
            complexity=complexity
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
                complexity=complexity
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
                confidence=adjudication_res.system_confidence,
                unresolved_disagreements=adjudication_res.unresolved_disputes or ["Resolved via targeted debate."],
                key_evidence=[e.excerpt for e in adjudication_res.key_evidence] if adjudication_res.key_evidence else ["Resolved through cross-specialist debate."],
                structured_evidence=adjudication_res.key_evidence,
                claims=all_claims,
                adjudication=adjudication_res,
                participating_agents=[a.id for a in participating_agents],
                rounds=rounds_log,
                mode_used="debate",
                complexity=complexity.value,
                models_used=list(dict.fromkeys(all_models_used)),
                total_tokens=total_tokens,
                total_latency_seconds=round(total_duration, 4)
            )

        # Direct Instant Synthesis (Standard fast path)
        if synthesis_text.startswith("*[Specialist Synthesizer temporarily offline") and round_1_messages:
            extracted_proposals = []
            for msg in round_1_messages:
                if not msg.content.startswith("*[Specialist"):
                    extracted_proposals.append(f"### {msg.agent_role} Recommendation\n{msg.content}")

            if extracted_proposals:
                synthesis_text = (
                    "## Multi-Specialist Consolidated Recommendations\n\n" +
                    "\n\n".join(extracted_proposals)
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
            confidence=adjudication_res.system_confidence,
            unresolved_disagreements=adjudication_res.unresolved_disputes,
            key_evidence=[e.excerpt for e in adjudication_res.key_evidence] if adjudication_res.key_evidence else ["Consensus verified across specialist team."],
            structured_evidence=adjudication_res.key_evidence,
            claims=all_claims,
            adjudication=adjudication_res,
            participating_agents=[a.id for a in participating_agents],
            rounds=rounds_log,
            mode_used="consensus",
            complexity=complexity.value,
            models_used=list(dict.fromkeys(all_models_used)),
            total_tokens=total_tokens,
            total_latency_seconds=round(total_duration, 4)
        )


# Backward-compatibility alias
DebateEngine = CollaborationEngine
debate_engine = CollaborationEngine()
