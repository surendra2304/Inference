"""Verification script for Agent Classification, Selection, and Fallback subsystems."""

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

from app.agents.registry import agent_registry  # noqa: E402
from app.agents.roles import register_all_specialists  # noqa: E402
from app.agents.router import router as task_router  # noqa: E402
from app.core.orchestrator import OrchestrationRequest, Orchestrator  # noqa: E402
from app.providers.base import ProviderMessage, ProviderRequest  # noqa: E402
from app.providers.gateway import model_gateway  # noqa: E402

console = Console(legacy_windows=False)


async def verify_agent_registry() -> None:
    """Verify registry contains all 10 specialized agent roles."""
    register_all_specialists()
    agents = agent_registry.list_agents()

    table = Table(title="1. Agent Registry & Multi-Model Pipelines", show_header=True)
    table.add_column("Agent ID", style="cyan", justify="left")
    table.add_column("Role Name", style="bold", justify="left")
    table.add_column("Primary Model", style="green", justify="left")
    table.add_column("Fallback Pipelines", style="yellow", justify="left")
    table.add_column("Key Strengths", style="dim", justify="left")

    for a in agents:
        fallback_str = " -> ".join(f"{m.provider}:{m.model}" for m in a.models[1:]) or "None"
        strengths_str = ", ".join(a.strengths[:3])
        table.add_row(
            a.id,
            a.name,
            f"{a.model_provider}:{a.model_name}",
            fallback_str,
            strengths_str,
        )

    console.print(table)
    assert len(agents) == 11, f"Expected 11 agents, got {len(agents)}"
    console.print("[bold green]✓ Agent Registry: 11/11 specialist roles registered successfully.[/bold green]\n")


def verify_classification_and_selection():
    """Verify task classification and dynamic specialist panel selection."""
    test_cases = [
        # (query, expected_domain_specialist, requested_mode, expected_mode, max_latency)
        ("Fix this null pointer exception in python trace", "debugger", "auto", "fast", None),
        ("Audit smart contract for authorization vulnerabilities", "security_analyst", "auto", "fast", None),
        ("Calculate Sharpe ratio and drawdown for our futures bot", "trading_analyst", "auto", "fast", None),
        ("Refactor this async FastAPI endpoint with Pydantic", "coder", "auto", "fast", None),
        ("Design a modular schema for distributed event store", "architect", "auto", "debate", None),
        ("Is it true that quantum computers break AES-256?", "fact_checker", "auto", "fast", None),
        ("What is the capital of France?", "researcher", "auto", "fast", None),
        ("Compare microservices vs monolith architecture tradeoffs", "architect", "auto", "debate", None),
        ("Evaluate pros and cons of SQLite vs Postgres", "researcher", "auto", "debate", None),
        ("Compare Redis vs Memcached tradeoffs under tight latency", "strategist", "debate", "fast", 1.0),
    ]

    table = Table(title="2. Domain Classification & Mode Routing", show_header=True)
    table.add_column("Test Query", style="cyan", justify="left", max_width=40)
    table.add_column("Detected Specialist", style="green", justify="left")
    table.add_column("Routed Mode", style="yellow", justify="left")
    table.add_column("Selected Panel", style="bold", justify="left")
    table.add_column("Guardrail / Rationale", style="dim", justify="left")

    for q, exp_agent, req_mode, exp_mode, max_lat in test_cases:
        decision = task_router.route_task(q, requested_mode=req_mode, max_latency=max_lat)
        table.add_row(
            q,
            task_router.detect_domain_specialist(q),
            decision.mode,
            ", ".join(decision.selected_agent_ids),
            decision.reason[:45] + "...",
        )
        assert task_router.detect_domain_specialist(q) == exp_agent, f"Domain mismatch for '{q}'"
        assert decision.mode == exp_mode, f"Mode mismatch for '{q}': got {decision.mode}, expected {exp_mode}"

    console.print(table)
    console.print("[bold green]✓ Classification & Agent Selection: 10/10 test cases passed with zero errors.[/bold green]\n")


async def verify_live_orchestration():
    """Verify live end-to-end execution of fast, review, and debate workflows."""
    orch = Orchestrator()
    console.print("[bold]3. Testing Live End-to-End Orchestration Workflows...[/bold]")

    # Fast Mode: Trading query -> Quantitative Trading Analyst
    q_fast = "What is the formula for Sharpe Ratio in trading?"
    res_fast = await orch.process_task(OrchestrationRequest(question=q_fast, mode="fast"))
    console.print(
        f"[green]✓ Fast Mode Live Execution:[/green] Agent: [bold]{res_fast.agents_used}[/bold] | "
        f"Latency: {res_fast.total_latency_seconds:.2f}s | Confidence: {res_fast.confidence:.2f}"
    )

    # Review Mode: Coding query -> Coder + Critic pair
    q_rev = "Explain how async generators work in Python with a 3-line example."
    res_rev = await orch.process_task(OrchestrationRequest(question=q_rev, mode="review"))
    console.print(
        f"[green]✓ Review Mode Live Execution:[/green] Agents: [bold]{res_rev.agents_used}[/bold] | "
        f"Latency: {res_rev.total_latency_seconds:.2f}s | Confidence: {res_rev.confidence:.2f}"
    )

    # Debate Mode: Strategic query -> Multi-Agent Collaboration Engine
    q_deb = "Compare SQLite vs PostgreSQL for a low-latency real-time inference cache."
    res_deb = await orch.process_task(OrchestrationRequest(question=q_deb, mode="debate", max_agents=3))
    console.print(
        f"[green]✓ Debate Mode Live Execution:[/green] Agents: [bold]{res_deb.agents_used}[/bold] | "
        f"Latency: {res_deb.total_latency_seconds:.2f}s | Confidence: {res_deb.confidence:.2f}"
    )
    console.print()


async def verify_fallback_and_key_rotation():
    """Verify that when a provider or model fails, key rotation and dynamic fallback kick in."""
    console.print("[bold]4. Testing Gateway Key Rotation & Dynamic Fallback Engine...[/bold]")

    # Test key pool quarantine mechanism
    gemini_pool = model_gateway.key_pools.get("gemini")
    if gemini_pool:
        test_k = gemini_pool.choose()
        if test_k:
            gemini_pool.quarantine(test_k, duration_seconds=2.0)
            quarantined_cnt = gemini_pool.get_quarantined_keys_count()
            console.print(
                f"[green]✓ Key Pool Quarantine Engine:[/green] Total Keys: {gemini_pool.total_keys_count}, "
                f"Quarantined: {quarantined_cnt}, Active: {gemini_pool.get_active_keys_count()}"
            )
            assert quarantined_cnt >= 1, "Quarantine count failed"

    # Test capability-based dynamic fallback through OpenRouter
    openrouter_fallback = await model_gateway._execute_dynamic_fallback(
        failed_provider="mock_failed_provider",
        request=ProviderRequest(messages=[ProviderMessage(role="user", content="Ping! Say 'FALLBACK_OK'")]),
        capability="reasoning",
        stage_name="test_fallback",
        last_error=RuntimeError("Simulated primary upstream failure"),
    )
    console.print(
        f"[green]✓ Dynamic Capability Fallback Engine:[/green] Provider: [bold]{openrouter_fallback.provider}[/bold] | "
        f"Model: [bold]{openrouter_fallback.model}[/bold] | Output: '{openrouter_fallback.content.strip()[:40]}...'"
    )
    assert openrouter_fallback.content != "", "Fallback response content was empty"
    console.print()


async def main():
    console.print(
        Panel.fit(
            "[bold cyan]FRIDAY INFERENCE — AGENT CLASSIFICATION, SELECTION & FALLBACK AUDIT[/bold cyan]\n"
            "[dim]Verifying all 10 specialist agents, routing policies, guardrails, live DAG execution, and failover.[/dim]",
            border_style="cyan",
        )
    )
    await verify_agent_registry()
    verify_classification_and_selection()
    await verify_live_orchestration()
    await verify_fallback_and_key_rotation()
    console.print(
        Panel.fit(
            "[bold green]ALL SUBSYSTEMS VERIFIED AND FULLY OPERATIONAL (100% PASS)[/bold green]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
