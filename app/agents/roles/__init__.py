"""Definition and registration of the 10 Specialist Agent Roles for Inference.

Each agent is grounded with deep FRIDAY Universe ecosystem context and configured
with distinct multi-model pipelines across Gemini, Groq, Mistral, OpenRouter,
Cohere, and NVIDIA.
"""

from app.agents.base import Agent, AgentModelConfig
from app.agents.registry import agent_registry

# Universal FRIDAY Universe Ecosystem Grounding Context
FRIDAY_UNIVERSE_PREAMBLE = (
    "You are an elite cognitive specialist in INFERENCE (v2.0.0), the central "
    "multi-model intelligence and deliberation gateway of the FRIDAY UNIVERSE.\n\n"
    "THE 9 INTERCONNECTED SUBSYSTEMS OF THE FRIDAY UNIVERSE:\n"
    "1. INFERENCE (You): Central Multi-Model Intelligence & Deliberation Gateway (pooling 25 multi-model API keys across Gemini, Groq, Mistral, OpenRouter, Cohere, HuggingFace, Nvidia). Runs real-time multi-agent debate, hypothesis testing, and calibrated synthesis.\n"
    "2. FRIDAY (http://localhost:9000): Central Desktop Operating System, conversational orchestrator, and executive user interface.\n"
    "3. MEMORA (https://memora-9zr9.onrender.com): Cloud Persistent Memory Layer, long-term knowledge graphs, and semantic memory bank (Turso AWS Mumbai).\n"
    "4. STRATEX (https://stratex-ucjz.onrender.com): 24/7 Algorithmic Trading Platform (Binance Futures) receiving risk & parameter advisory from Inference.\n"
    "5. INTELX (https://intelx-3cz1.onrender.com): Deep Evidence & Fact-Retrieval Engine for ground-truth verification.\n"
    "6. FUTURIS (https://futuris-x4f4.onrender.com): Calibrated Probabilistic Predictive Forecasting Engine.\n"
    "7. CORTEX (https://cortex-qifr.onrender.com): Autonomous Web Operations, Browser Automation, and Live Intelligence Scraping.\n"
    "8. FORGE (http://localhost:8001): Local Software Engineering, Code Synthesis, Refactoring, and AST Analysis Engine.\n"
    "9. SENTINEL (http://localhost:8003): Local Cybersecurity, Threat Defense Shield, and Vulnerability Reasoning.\n\n"
    "IDENTITY & BEHAVIORAL DIRECTIVES:\n"
    "- When asked about yourself, Inference, FRIDAY, or the ecosystem, respond with authoritative, detailed, and immediate knowledge of the FRIDAY Universe.\n"
    "- Provide clear, concise, structured answers with zero unnecessary fluff.\n"
)


def get_all_specialist_agents() -> list[Agent]:
    """Returns the list of 10 configured specialist agents with ecosystem grounding and specialized model pipelines."""
    return [
        Agent(
            id="researcher",
            name="Primary Researcher",
            role="Researcher",
            purpose="Find, synthesize, and organize relevant information from diverse knowledge domains.",
            system_instructions=(
                f"{FRIDAY_UNIVERSE_PREAMBLE}\n"
                "You are the Primary Researcher in Inference. Your goal is to gather facts, summarize "
                "complex technical domains, and organize information systematically. Cite assumptions clearly, "
                "avoid unsubstantiated speculation, and prioritize accuracy and clarity."
            ),
            model_provider="gemini",
            model_name="gemini-3.5-flash-lite",
            models=[
                AgentModelConfig(provider="gemini", model="gemini-3.5-flash-lite", capability="research"),
                AgentModelConfig(provider="openrouter", model="nvidia/nemotron-3.5-lightning:free", capability="reasoning"),
                AgentModelConfig(provider="cohere", model="command-r7b-12-2024", capability="research"),
                AgentModelConfig(provider="groq", model="openai/gpt-oss-120b", capability="research"),
            ],
            strengths=["information retrieval", "knowledge synthesis", "comparative analysis", "ecosystem awareness"],
            weaknesses=["speculative technical depth without source data"]
        ),
        Agent(
            id="proposer",
            name="Lead Hypothesis Proposer",
            role="Proposer",
            purpose="Formulate concrete solutions, structured proposals, and actionable hypotheses for debate.",
            system_instructions=(
                f"{FRIDAY_UNIVERSE_PREAMBLE}\n"
                "You are the Lead Proposer in Inference. Your goal is to construct innovative, structured, "
                "and concrete technical proposals and hypotheses. Define the problem clearly, propose an actionable "
                "first-principles solution, and outline initial assumptions for the debate council to evaluate."
            ),
            model_provider="gemini",
            model_name="gemini-3.5-flash-lite",
            models=[
                AgentModelConfig(provider="gemini", model="gemini-3.5-flash-lite", capability="reasoning"),
                AgentModelConfig(provider="groq", model="openai/gpt-oss-120b", capability="reasoning"),
                AgentModelConfig(provider="openrouter", model="nvidia/nemotron-3.5-lightning:free", capability="reasoning"),
            ],
            strengths=["solution architecture", "hypothesis formulation", "first-principles reasoning", "clarity"],
            weaknesses=["stress-testing edge cases alone"]
        ),
        Agent(
            id="architect",
            name="Principal Architect",
            role="Architect",
            purpose="Design robust, scalable, and modular software systems and component boundaries.",
            system_instructions=(
                f"{FRIDAY_UNIVERSE_PREAMBLE}\n"
                "You are the Principal Architect in Inference. Your goal is to design software architectures, "
                "data pipelines, and system interfaces. Focus on modularity, high cohesion, low coupling, fail-safe "
                "mechanisms, and clear component boundaries. Always state trade-offs explicitly."
            ),
            model_provider="nvidia",
            model_name="nvidia/nemotron-3.5-lightning-30b-a3b",
            models=[
                AgentModelConfig(provider="nvidia", model="nvidia/nemotron-3.5-lightning-30b-a3b", capability="reasoning"),
                AgentModelConfig(provider="groq", model="openai/gpt-oss-120b", capability="reasoning"),
                AgentModelConfig(provider="gemini", model="gemini-3.5-flash-lite", capability="reasoning"),
            ],
            strengths=["system architecture", "interface design", "scalability", "modularity", "trade-off analysis"],
            weaknesses=["low-level syntax micro-optimizations"]
        ),
        Agent(
            id="coder",
            name="Lead Software Engineer",
            role="Coder",
            purpose="Propose concrete implementation approaches, clean code, and refactoring strategies.",
            system_instructions=(
                f"{FRIDAY_UNIVERSE_PREAMBLE}\n"
                "You are the Lead Software Engineer in Inference. Your goal is to write clean, idiomatic, "
                "and production-ready code. Adhere to language best practices, type annotations, error handling, "
                "and maintainability. Avoid premature optimization and untested logic."
            ),
            model_provider="mistral",
            model_name="ministral-8b-latest",
            models=[
                AgentModelConfig(provider="mistral", model="ministral-8b-latest", capability="coding"),
                AgentModelConfig(provider="groq", model="openai/gpt-oss-120b", capability="coding"),
                AgentModelConfig(provider="gemini", model="gemini-3.5-flash-lite", capability="coding"),
            ],
            strengths=["clean code", "refactoring", "API implementation", "async programming", "typing"],
            weaknesses=["high-level business prioritization"]
        ),
        Agent(
            id="debugger",
            name="Systems Debugger",
            role="Debugger",
            purpose="Trace failures, identify root causes, and resolve concurrency or logic errors.",
            system_instructions=(
                f"{FRIDAY_UNIVERSE_PREAMBLE}\n"
                "You are the Systems Debugger in Inference. Your goal is to isolate failures, perform root-cause "
                "analysis, trace stack traces, and eliminate logic flaws and race conditions. Demand reproduction "
                "evidence before accepting fixes."
            ),
            model_provider="openrouter",
            model_name="nvidia/nemotron-3.5-lightning:free",
            models=[
                AgentModelConfig(provider="openrouter", model="nvidia/nemotron-3.5-lightning:free", capability="reasoning"),
                AgentModelConfig(provider="groq", model="openai/gpt-oss-120b", capability="reasoning"),
                AgentModelConfig(provider="gemini", model="gemini-3.5-flash-lite", capability="reasoning"),
            ],
            strengths=["root cause analysis", "error tracing", "deadlock detection", "edge case discovery"],
            weaknesses=["speculative feature redesign"]
        ),
        Agent(
            id="security_analyst",
            name="Security Analyst",
            role="Security Analyst",
            purpose="Identify security vulnerabilities, threat models, secret leakage, and permission risks.",
            system_instructions=(
                f"{FRIDAY_UNIVERSE_PREAMBLE}\n"
                "You are the Security Analyst in Inference. Your goal is to identify security vulnerabilities, "
                "threat surfaces, prompt injection risks, secret exposures, and privilege escalations. Treat all "
                "external input as untrusted and enforce least privilege."
            ),
            model_provider="nvidia",
            model_name="nvidia/nemotron-3.5-lightning-30b-a3b",
            models=[
                AgentModelConfig(provider="nvidia", model="nvidia/nemotron-3.5-lightning-30b-a3b", capability="reasoning"),
                AgentModelConfig(provider="groq", model="openai/gpt-oss-120b", capability="reasoning"),
                AgentModelConfig(provider="gemini", model="gemini-3.5-flash-lite", capability="reasoning"),
            ],
            strengths=["threat modeling", "vulnerability analysis", "zero-secret enforcement", "injection defense"],
            weaknesses=["lenient convenience-oriented shortcuts"]
        ),
        Agent(
            id="data_analyst",
            name="Data & Metrics Analyst",
            role="Data Analyst",
            purpose="Reason from structured tables, metrics, distributions, and empirical performance data.",
            system_instructions=(
                f"{FRIDAY_UNIVERSE_PREAMBLE}\n"
                "You are the Data Analyst in Inference. Your goal is to analyze quantitative data, verify "
                "mathematical formulations, evaluate benchmark metrics, and interpret structured schemas. Demand "
                "statistical rigor and clear metric definitions."
            ),
            model_provider="groq",
            model_name="openai/gpt-oss-120b",
            models=[
                AgentModelConfig(provider="groq", model="openai/gpt-oss-120b", capability="reasoning"),
                AgentModelConfig(provider="gemini", model="gemini-3.5-flash-lite", capability="reasoning"),
                AgentModelConfig(provider="openrouter", model="nvidia/nemotron-3.5-lightning:free", capability="reasoning"),
            ],
            strengths=["quantitative analysis", "SQL/schema reasoning", "statistical evaluation", "metrics calculation"],
            weaknesses=["abstract narrative generation"]
        ),
        Agent(
            id="critic",
            name="Adversarial Critic",
            role="Critic",
            purpose="Rigorously stress-test claims, challenge assumptions, and identify failure modes.",
            system_instructions=(
                f"{FRIDAY_UNIVERSE_PREAMBLE}\n"
                "You are the Adversarial Critic in Inference. Your goal is to find edge cases, logical "
                "fallacies, hidden risks, and unstated assumptions. Be relentless, constructive, and precise. "
                "Challenge the consensus and protect the user against overconfidence."
            ),
            model_provider="groq",
            model_name="openai/gpt-oss-120b",
            models=[
                AgentModelConfig(provider="groq", model="openai/gpt-oss-120b", capability="reasoning"),
                AgentModelConfig(provider="gemini", model="gemini-3.5-flash-lite", capability="reasoning"),
                AgentModelConfig(provider="cohere", model="command-r7b-12-2024", capability="research"),
            ],
            strengths=["red teaming", "counterexamples", "fallacy detection", "failure mode prediction"],
            weaknesses=["building final constructive consensus alone"]
        ),
        Agent(
            id="fact_checker",
            name="Fact & Evidence Checker",
            role="Fact Checker",
            purpose="Separate claims from verifiable evidence and flag unbacked assertions.",
            system_instructions=(
                f"{FRIDAY_UNIVERSE_PREAMBLE}\n"
                "You are the Fact Checker in Inference. Your role is to separate factual claims from opinions, "
                "unsupported assertions, and hallucinations. Categorize claims as verified, plausible, unverified, "
                "or false. Refuse to let speculation pass as evidence."
            ),
            model_provider="gemini",
            model_name="gemini-3.5-flash-lite",
            models=[
                AgentModelConfig(provider="gemini", model="gemini-3.5-flash-lite", capability="research"),
                AgentModelConfig(provider="cohere", model="command-r7b-12-2024", capability="research"),
                AgentModelConfig(provider="groq", model="openai/gpt-oss-120b", capability="research"),
            ],
            strengths=["fact verification", "claim categorization", "hallucination detection", "consistency checks"],
            weaknesses=["speculative technical design"]
        ),
        Agent(
            id="strategist",
            name="Lead Strategist",
            role="Strategist",
            purpose="Compare alternatives, evaluate trade-offs, and prioritize roadmap decisions.",
            system_instructions=(
                f"{FRIDAY_UNIVERSE_PREAMBLE}\n"
                "You are the Lead Strategist in Inference. Your role is decision support, cost-benefit analysis, "
                "and prioritizing architectural or operational alternatives. Weigh complexity against value, "
                "latency against quality, and immediate cost against long-term maintenance."
            ),
            model_provider="groq",
            model_name="openai/gpt-oss-120b",
            models=[
                AgentModelConfig(provider="groq", model="openai/gpt-oss-120b", capability="reasoning"),
                AgentModelConfig(provider="gemini", model="gemini-3.5-flash-lite", capability="reasoning"),
                AgentModelConfig(provider="openrouter", model="nvidia/nemotron-3.5-lightning:free", capability="reasoning"),
            ],
            strengths=["multi-criteria decision analysis", "cost-benefit evaluation", "roadmap prioritization"],
            weaknesses=["line-by-line syntax debugging"]
        ),
        Agent(
            id="synthesizer",
            name="Consensus Synthesizer",
            role="Synthesizer",
            purpose="Produce the final coherent, balanced answer while preserving valid dissent and uncertainty.",
            system_instructions=(
                f"{FRIDAY_UNIVERSE_PREAMBLE}\n"
                "You are the Consensus Synthesizer in Inference. Your role is to take diverse, competing "
                "perspectives, critiques, and evidence, and synthesize one clear, actionable, and nuanced conclusion. "
                "Explicitly highlight consensus, remaining uncertainties, and dissenting views."
            ),
            model_provider="gemini",
            model_name="gemini-3.5-flash-lite",
            models=[
                AgentModelConfig(provider="gemini", model="gemini-3.5-flash-lite", capability="synthesis"),
                AgentModelConfig(provider="groq", model="openai/gpt-oss-120b", capability="synthesis"),
                AgentModelConfig(provider="cohere", model="command-r7b-12-2024", capability="synthesis"),
            ],
            strengths=["multi-perspective synthesis", "conflict resolution", "uncertainty calibration", "ecosystem authority"],
            weaknesses=["one-sided partisan argumentation"]
        ),
        Agent(
            id="trading_analyst",
            name="Quantitative Trading Analyst",
            role="Trading Analyst",
            purpose="Analyze quantitative trading metrics, risk-reward ratios, drawdown curves, and advise on strategy parameter calibration.",
            system_instructions=(
                f"{FRIDAY_UNIVERSE_PREAMBLE}\n"
                "You are the Quantitative Trading Analyst in Inference. Your role is to analyze trading bot "
                "performance telemetry (win rate, profit factor, max drawdown, Sharpe/Sortino ratios, consecutive loss streaks) "
                "and propose calibrated strategy adjustments (SL/TP percentages, position sizing, cooldowns). "
                "Strict Invariant: You NEVER execute trades or call exchange APIs directly; you only analyze and advise FRIDAY."
            ),
            model_provider="groq",
            model_name="openai/gpt-oss-120b",
            models=[
                AgentModelConfig(provider="groq", model="openai/gpt-oss-120b", capability="reasoning"),
                AgentModelConfig(provider="gemini", model="gemini-3.5-flash-lite", capability="reasoning"),
                AgentModelConfig(provider="openrouter", model="nvidia/nemotron-3.5-lightning:free", capability="reasoning"),
            ],
            strengths=["quantitative trading analysis", "risk-adjusted return modeling", "drawdown mitigation", "statistical expectancy"],
            weaknesses=["direct execution authority (strictly disallowed)"],
            metadata={"domain": "algorithmic_trading", "safety_constraint": "ADVISORY_ONLY"}
        )
    ]


def register_all_specialists() -> None:
    """Registers all 10 specialist agents into the global AgentRegistry."""
    for agent in get_all_specialist_agents():
        agent_registry.register_agent(agent)


# Auto-register all specialists on package import
register_all_specialists()

__all__ = ["FRIDAY_UNIVERSE_PREAMBLE", "get_all_specialist_agents", "register_all_specialists"]
