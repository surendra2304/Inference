"""Interactive CLI for Inference — Fast, Review, and Debate query execution."""

import asyncio
import sys
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from app.core.orchestrator import OrchestrationRequest, orchestrator

# Configure stdout
sys.stdout.reconfigure(encoding="utf-8")
console = Console()


async def ask_question(question: str, mode: str = "auto", max_agents: int = 5):
    console.print(Panel.fit(
        f"[bold white]{question}[/bold white]",
        title=f"🌌 [bold cyan]INFERENCE — QUERY RUNNER (Mode: {mode.upper()})[/bold cyan]",
        border_style="cyan"
    ))

    t0 = time.perf_counter()
    with console.status(f"[bold green]Evaluating query via Inference Orchestrator (Mode: {mode})...", spinner="dots"):
        req = OrchestrationRequest(
            question=question,
            mode=mode,
            max_agents=max_agents
        )
        result = await orchestrator.process_task(req)

    latency = time.perf_counter() - t0

    # Display Answer
    console.print(Panel(
        Markdown(result.answer),
        title=f"🏆 [bold green]ANSWER ({result.mode_used.upper()} MODE • Confidence: {result.confidence * 100:.1f}%)[/bold green]",
        border_style="green"
    ))

    # Meta table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Execution Mode", result.mode_used.upper())
    table.add_row("Primary Provider", result.provider_used.upper())
    table.add_row("Agents Used", ", ".join(result.agents_used))
    table.add_row("Models Active", ", ".join(result.models_used))
    table.add_row("Total Latency", f"{latency:.2f}s")
    table.add_row("Total Tokens", str(result.total_tokens))

    console.print(table)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[yellow]Usage: python scripts/ask.py \"Your question here\" [fast|review|debate|auto][/yellow]")
        sys.exit(1)

    user_query = sys.argv[1]
    exec_mode = sys.argv[2] if len(sys.argv) > 2 else "auto"
    asyncio.run(ask_question(user_query, mode=exec_mode))
