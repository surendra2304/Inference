"""Live visual runner for Inference's 6-Round Structured Multi-Agent Debate Engine."""

import asyncio
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from app.agents.roles import get_all_specialist_agents
from app.core.orchestrator import orchestrator

# Configure stdout
sys.stdout.reconfigure(encoding="utf-8")
console = Console()

ROLE_COLORS = {
    "Researcher": "cyan",
    "Architect": "blue",
    "Coder": "green",
    "Debugger": "magenta",
    "Security Analyst": "red",
    "Data Analyst": "yellow",
    "Critic": "bold red",
    "Fact Checker": "bold yellow",
    "Strategist": "bold blue",
    "Synthesizer": "bold green",
}


async def run_live_debate(question: str, use_all_agents: bool = True):
    console.print(Panel.fit(
        f"[bold white]{question}[/bold white]",
        title="🌌 [bold cyan]INFERENCE — LIVE MULTI-AGENT ADVERSARIAL DEBATE[/bold cyan]",
        border_style="cyan"
    ))

    # Initialize memory
    await orchestrator.memory.initialize()

    # Retrieve all 10 specialist agents
    all_agents = get_all_specialist_agents()
    panel_agents = all_agents if use_all_agents else all_agents[:5]

    console.print(f"\n[bold yellow]Full Panel of {len(panel_agents)} Specialists Assembled across All Active Cloud Providers:[/bold yellow]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", justify="center")
    table.add_column("Specialist Role", style="bold")
    table.add_column("Agent ID")
    table.add_column("Cloud Provider", style="cyan")
    table.add_column("Assigned Model", style="green")

    for idx, a in enumerate(panel_agents, 1):
        table.add_row(str(idx), a.role, a.id, a.model_provider.upper(), a.model_name)
    console.print(table)
    console.print("\n" + "="*80 + "\n")

    # Run the debate engine
    engine = orchestrator.debate_engine
    engine.memory = orchestrator.memory

    with console.status(f"[bold green]Executing 6-Round Adversarial Protocol across {len(panel_agents)} Cloud Models...", spinner="dots"):
        result = await engine.run_debate(
            task_id="live_demo_task",
            question=question,
            participating_agents=panel_agents,
            require_evidence=True
        )

    # Display each round of debate, critique, rebuttal, and synthesis
    for r in result.rounds:
        round_title = f"ROUND {r.round_number}: {r.stage_name.replace('_', ' ').upper()}"
        console.print(f"\n[bold yellow]{'━'*30} {round_title} {'━'*30}[/bold yellow]\n")

        for msg in r.messages:
            color = ROLE_COLORS.get(msg.agent_role, "white")
            target = f" [dim](Targeting: {msg.target_agent_id})[/dim]" if msg.target_agent_id else ""

            console.print(Panel(
                Markdown(msg.content),
                title=f"[{color}]{msg.agent_role} ({msg.agent_id}){target}[/{color}]",
                border_style=color.replace("bold ", "")
            ))

    # Final Synthesized Answer
    console.print("\n" + "="*80)
    console.print(Panel(
        Markdown(result.final_answer),
        title=f"🏆 [bold green]FINAL SYNTHESIZED CONSENSUS (Confidence: {result.confidence * 100:.1f}%)[/bold green]",
        border_style="green"
    ))

    if result.unresolved_disagreements:
        console.print("\n[bold red]⚡ Unresolved Disagreements / Preserved Dissent (No Fake Harmony):[/bold red]")
        for d in result.unresolved_disagreements:
            console.print(f" • [red]{d}[/red]")

    if result.key_evidence:
        console.print("\n[bold cyan]📌 Surviving Empirical Claims & Verified Evidence:[/bold cyan]")
        for e in result.key_evidence:
            console.print(f" • [cyan]{e}[/cyan]")

    console.print(f"\n[dim]Total Tokens: {result.total_tokens} | Deliberation Latency: {result.total_latency_seconds:.2f}s[/dim]\n")


if __name__ == "__main__":
    test_question = (
        sys.argv[1] if len(sys.argv) > 1 else
        "Should high-frequency trading platforms use a distributed event-driven microservices architecture, "
        "or a shared-memory modular monolith with hardware lock-free queues?"
    )
    asyncio.run(run_live_debate(test_question, use_all_agents=True))
