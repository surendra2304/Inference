"""Interactive Command-Line Interface (CLI) for Inference.

Executes directly in-process using the Orchestrator and SQLiteMemory without
requiring a running uvicorn background server.
"""

import asyncio
import json
import logging
import sys

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from app.core.orchestrator import OrchestrationRequest, Orchestrator
from app.experiments.harness import BenchmarkHarness
from app.memory.sqlite import SQLiteMemory

# Configure UTF-8 safe console for Windows terminal
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

cli_app = typer.Typer(help="Inference — Multi-Agent Intelligence CLI")
console = Console(legacy_windows=False)


def _silence_internal_logs() -> None:
    """Mute noisy backend logs so CLI output is clean and uncluttered."""
    for log_name in ["inference", "httpx", "aiosqlite", "uvicorn"]:
        lg = logging.getLogger(log_name)
        lg.setLevel(logging.WARNING)


async def _run_in_process_ask(
    question: str,
    mode: str = "auto",
    max_agents: int = 5,
    budget: float | None = None,
    latency: float | None = None,
    verbose: bool = False
) -> None:
    """Instantiate orchestrator and run task directly in-process."""
    if not verbose:
        _silence_internal_logs()

    memory = SQLiteMemory()
    await memory.initialize()
    orch = Orchestrator(memory=memory)

    req = OrchestrationRequest(
        question=question,
        mode=mode,
        max_agents=max_agents,
        max_budget=budget,
        max_latency=latency
    )

    try:
        if verbose:
            console.print(f"[bold cyan]Processing query in-process:[/bold cyan] {question}")

        res = await orch.process_task(req)

        if console:
            console.print()
            console.print(Markdown(res.answer))
            console.print()
            if verbose:
                console.print(
                    f"[dim]Mode: {res.mode_used} | Agents: {', '.join(res.agents_used)} | "
                    f"Latency: {res.total_latency_seconds:.2f}s | Tokens: {res.total_tokens}[/dim]"
                )
        else:
            print(f"\n{res.answer}\n")
    except Exception as exc:
        if console:
            console.print(f"[bold red]Execution error:[/bold red] {exc}")
        else:
            print(f"Execution error: {exc}")


async def _run_in_process_debate(
    question: str,
    max_agents: int = 5
) -> None:
    """Instantiate orchestrator and run multi-agent debate directly in-process."""
    memory = SQLiteMemory()
    await memory.initialize()
    orch = Orchestrator(memory=memory)

    req = OrchestrationRequest(
        question=question,
        mode="debate",
        max_agents=max_agents
    )

    if console:
        console.print(f"[bold magenta]Initiating Dynamic Multi-Agent Debate Protocol:[/bold magenta] {question}")
    else:
        print(f"Initiating Dynamic Multi-Agent Debate Protocol: {question}")

    try:
        res = await orch.process_task(req)
        if console:
            console.print(Panel(
                f"[bold green]Debate Consensus & Synthesis:[/bold green]\n\n{res.answer}",
                title=f"Debate: {res.run_id} (Confidence: {res.confidence:.2f} | Complexity: {res.complexity.upper()})"
            ))
            console.print(
                f"[dim]Agents: {', '.join(res.agents_used)} | "
                f"Models: {', '.join(res.models_used)} | "
                f"Latency: {res.total_latency_seconds:.2f}s | "
                f"Tokens: {res.total_tokens}[/dim]"
            )
            if res.unresolved_disagreements:
                console.print("[bold yellow]Unresolved Disagreements / Preserved Dissent:[/bold yellow]")
                for d in res.unresolved_disagreements:
                    console.print(f" - {d}")
        else:
            print(f"\n--- Debate Consensus (Complexity: {res.complexity.upper()}) ---")
            print(res.answer)
            print(f"Agents: {res.agents_used} | Latency: {res.total_latency_seconds:.2f}s | Tokens: {res.total_tokens}")
    except Exception as exc:
        if console:
            console.print(f"[bold red]Execution error:[/bold red] {exc}")
        else:
            print(f"Execution error: {exc}")


async def _run_in_process_experiment(
    exp_type: str = "benchmark_suite",
    question: str | None = None
) -> None:
    """Run an automated experiment in-process."""
    memory = SQLiteMemory()
    await memory.initialize()
    orch = Orchestrator(memory=memory)
    harness = BenchmarkHarness(memory=memory, orchestrator=orch)

    try:
        if exp_type == "benchmark_suite":
            rec = await harness.run_benchmark_suite()
        elif exp_type == "baseline_vs_debate":
            q = question or "Should Inference use microservices or monolithic architecture?"
            rec = await harness.run_baseline_vs_debate_comparison(q)
        else:
            rec = await harness.run_benchmark_suite()

        if console:
            console.print(Panel(
                json.dumps(rec.model_dump(), indent=2, default=str),
                title=f"Experiment: {rec.id} ({rec.status})"
            ))
        else:
            print(json.dumps(rec.model_dump(), indent=2, default=str))
    except Exception as exc:
        if console:
            console.print(f"[bold red]Experiment error:[/bold red] {exc}")
        else:
            print(f"Experiment error: {exc}")


@cli_app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask Inference"),
    mode: str = typer.Option("auto", "--mode", "-m", help="Mode: auto, fast, review, debate"),
    max_agents: int = typer.Option(5, "--agents", "-a", help="Max agents to allocate"),
    budget: float | None = typer.Option(None, "--budget", "-b", help="Max budget in USD"),
    latency: float | None = typer.Option(None, "--latency", "-l", help="Max latency in seconds"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug logs & execution metadata")
):
    """Submit a question to the orchestrator directly in-process."""
    asyncio.run(_run_in_process_ask(question, mode=mode, max_agents=max_agents, budget=budget, latency=latency, verbose=verbose))


@cli_app.command()
def debate(
    question: str = typer.Argument(..., help="Complex question to debate"),
    max_agents: int = typer.Option(5, "--agents", "-a", help="Max specialists in panel")
):
    """Trigger the Multi-Agent Collaboration & Debate Engine directly in-process."""
    asyncio.run(_run_in_process_debate(question, max_agents=max_agents))


@cli_app.command()
def experiment(
    exp_type: str = typer.Option("benchmark_suite", "--type", "-t", help="benchmark_suite, baseline_vs_debate, model_comparison"),
    question: str | None = typer.Option(None, "--question", "-q", help="Optional test question")
):
    """Trigger an automated benchmark or comparison experiment directly in-process."""
    asyncio.run(_run_in_process_experiment(exp_type=exp_type, question=question))


@cli_app.command("test-keys")
def test_keys_cmd(
    mode: str = typer.Option("concurrent", "--mode", "-m", help="concurrent or sequential"),
    timeout: float = typer.Option(20.0, "--timeout", "-t", help="Per-key request timeout in seconds"),
    concurrency: int = typer.Option(6, "--concurrency", "-c", help="Concurrent worker count"),
    provider: str | None = typer.Option(None, "--provider", "-p", help="Filter to a specific provider"),
):
    """Test and verify all configured inference provider API keys in real-time."""
    from scripts.test_all_keys import run_all_key_tests

    asyncio.run(
        run_all_key_tests(
            mode=mode,
            timeout=timeout,
            specific_provider=provider,
            concurrency=concurrency,
        )
    )


def main():
    cli_app()


if __name__ == "__main__":
    main()
