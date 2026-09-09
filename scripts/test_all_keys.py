"""Real-time LLM API Key Verification & Latency Benchmark Script.

Tests all configured provider keys individually with upstream endpoints,
measuring live status, latency, and response viability.
"""

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.providers.base import ProviderMessage, ProviderRequest  # noqa: E402
from app.providers.cohere import CohereProvider  # noqa: E402
from app.providers.gemini import GeminiProvider  # noqa: E402
from app.providers.groq import GroqProvider  # noqa: E402
from app.providers.huggingface import HuggingFaceProvider  # noqa: E402
from app.providers.mistral import MistralProvider  # noqa: E402
from app.providers.nvidia import NvidiaProvider  # noqa: E402
from app.providers.openrouter import OpenRouterProvider  # noqa: E402


@dataclass
class KeyTestResult:
    provider: str
    key_name: str
    masked_key: str
    model: str
    status: str
    status_emoji: str
    latency_ms: float
    response_preview: str
    error_detail: str = ""


def mask_key(key: str) -> str:
    """Masks an API key preserving only leading and trailing identifiers."""
    if not key:
        return "<EMPTY>"
    key_str = key.strip()
    if len(key_str) <= 10:
        return f"{key_str[:3]}...{key_str[-2:]}"
    return f"{key_str[:7]}...{key_str[-4:]}"


def get_provider_instance(provider_name: str, key: str, timeout: float = 12.0):
    """Creates an isolated provider instance for a single key."""
    prov = provider_name.lower().strip()
    if prov == "gemini":
        return GeminiProvider(api_key=key, default_model="gemini-3.5-flash-lite", timeout=timeout)
    elif prov == "groq":
        return GroqProvider(api_key=key, default_model="openai/gpt-oss-120b", timeout=timeout)
    elif prov == "mistral":
        return MistralProvider(api_key=key, default_model="ministral-8b-latest", timeout=timeout)
    elif prov == "openrouter":
        return OpenRouterProvider(api_key=key, default_model="nvidia/nemotron-3.5-lightning:free", timeout=timeout)
    elif prov == "cohere":
        return CohereProvider(api_key=key, default_model="command-r7b-12-2024", timeout=timeout)
    elif prov == "huggingface":
        return HuggingFaceProvider(api_key=key, default_model="meta-llama/llama-3.1-8b-instruct", timeout=timeout)
    elif prov == "nvidia":
        return NvidiaProvider(api_key=key, default_model="nvidia/nemotron-3.5-lightning-30b-a3b", timeout=timeout)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


async def test_single_key(
    provider_name: str,
    key_name: str,
    key: str,
    timeout: float = 15.0,
    console: Console | None = None,
) -> KeyTestResult:
    """Tests a single key against its provider endpoint."""
    masked = mask_key(key)
    provider_instance = get_provider_instance(provider_name, key, timeout=timeout)
    model = getattr(provider_instance, "default_model", "default")

    req = ProviderRequest(
        messages=[ProviderMessage(role="user", content="Ping! Reply with 'PONG: OK' in 3 words or less.")],
        max_tokens=150,
        temperature=0.1,
    )

    start_t = time.perf_counter()
    try:
        response = await asyncio.wait_for(provider_instance.generate(req), timeout=timeout)
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        content = response.content.strip().replace("\n", " ")
        if len(content) > 50:
            content = content[:47] + "..."

        result = KeyTestResult(
            provider=provider_name.upper(),
            key_name=key_name,
            masked_key=masked,
            model=model,
            status="ACTIVE (200 OK)",
            status_emoji="[PASS]",
            latency_ms=round(latency_ms, 1),
            response_preview=content or "<empty>",
        )
    except asyncio.TimeoutError:
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        result = KeyTestResult(
            provider=provider_name.upper(),
            key_name=key_name,
            masked_key=masked,
            model=model,
            status="TIMEOUT",
            status_emoji="[TIMEOUT]",
            latency_ms=round(latency_ms, 1),
            response_preview="",
            error_detail=f"Request timed out after {timeout:.1f}s",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        err_msg = str(exc)
        status = "ERROR"
        status_emoji = "[FAIL]"

        if "429" in err_msg or "rate limit" in err_msg.lower() or "quota" in err_msg.lower():
            status = "RATE_LIMITED (429)"
            status_emoji = "[RATE]"
        elif "401" in err_msg or "unauthorized" in err_msg.lower() or "invalid api key" in err_msg.lower():
            status = "INVALID_KEY (401)"
            status_emoji = "[AUTH]"
        elif "403" in err_msg or "forbidden" in err_msg.lower():
            status = "FORBIDDEN (403)"
            status_emoji = "[AUTH]"
        elif "404" in err_msg or "not found" in err_msg.lower():
            status = "MODEL_NOT_FOUND (404)"
            status_emoji = "[NOTFOUND]"

        clean_err = err_msg.replace("\n", " ")
        if len(clean_err) > 70:
            clean_err = clean_err[:67] + "..."

        result = KeyTestResult(
            provider=provider_name.upper(),
            key_name=key_name,
            masked_key=masked,
            model=model,
            status=status,
            status_emoji=status_emoji,
            latency_ms=round(latency_ms, 1),
            response_preview="",
            error_detail=clean_err,
        )

    if console:
        status_style = "green" if "ACTIVE" in result.status else ("yellow" if "RATE" in result.status else "red")
        detail = result.response_preview if result.response_preview else result.error_detail
        console.print(
            f"[{status_style}]{result.status_emoji} [{result.provider}] {result.key_name} ({result.masked_key}) -> "
            f"{result.status} | {result.latency_ms:.0f}ms | {detail}[/{status_style}]"
        )

    return result


async def run_all_key_tests(
    mode: str = "concurrent",
    timeout: float = 20.0,
    specific_provider: str | None = None,
    concurrency: int = 6,
) -> list[KeyTestResult]:
    """Discovers all keys and executes real-time tests."""
    console = Console()
    providers = ["gemini", "groq", "mistral", "openrouter", "cohere", "huggingface", "nvidia"]
    if specific_provider:
        providers = [specific_provider.lower().strip()]

    all_keys_to_test: list[tuple[str, str, str]] = []
    for prov in providers:
        keys = settings.get_provider_keys(prov)
        for idx, k in enumerate(keys, start=1):
            key_name = f"{prov.upper()}_KEY_{idx}" if len(keys) > 1 else f"{prov.upper()}_KEY"
            all_keys_to_test.append((prov, key_name, k))

    console.print(
        Panel.fit(
            f"[bold cyan]FRIDAY INFERENCE -- REAL-TIME API KEY VERIFICATION[/bold cyan]\n"
            f"[dim]Total Keys Configured:[/dim] [bold yellow]{len(all_keys_to_test)}[/bold yellow] across [bold]{len(providers)}[/bold] providers\n"
            f"[dim]Mode:[/dim] [bold]{mode.upper()}[/bold] | [dim]Concurrency:[/dim] [bold]{concurrency}[/bold] | [dim]Per-key Timeout:[/dim] [bold]{timeout}s[/bold]",
            border_style="cyan",
        )
    )

    if not all_keys_to_test:
        console.print("[bold red]No API keys discovered in environment configuration![/bold red]")
        return []

    console.print("\n[bold]Testing Keys Live in Realtime...[/bold]\n")
    results: list[KeyTestResult] = []

    if mode == "sequential":
        for prov, key_name, key in all_keys_to_test:
            res = await test_single_key(prov, key_name, key, timeout=timeout, console=console)
            results.append(res)
    else:
        sem = asyncio.Semaphore(concurrency)

        async def sem_worker(prov: str, k_name: str, k_val: str) -> KeyTestResult:
            async with sem:
                return await test_single_key(prov, k_name, k_val, timeout=timeout, console=console)

        tasks = [
            asyncio.create_task(sem_worker(prov, key_name, key))
            for prov, key_name, key in all_keys_to_test
        ]
        results = await asyncio.gather(*tasks)

    table = Table(title="Inference API Keys Realtime Test Summary", show_header=True, header_style="bold magenta")
    table.add_column("Provider", style="cyan", justify="left")
    table.add_column("Key Name", style="bold", justify="left")
    table.add_column("Masked Key", style="dim", justify="left")
    table.add_column("Target Model", style="blue", justify="left")
    table.add_column("Status", justify="left")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Response / Error Snippet", justify="left")

    active_count = 0
    total_active_latency = 0.0

    for r in results:
        is_active = "ACTIVE" in r.status
        if is_active:
            active_count += 1
            total_active_latency += r.latency_ms
            status_cell = Text(f"{r.status_emoji} {r.status}", style="bold green")
            preview_cell = Text(r.response_preview, style="green")
        elif "RATE" in r.status:
            status_cell = Text(f"{r.status_emoji} {r.status}", style="bold yellow")
            preview_cell = Text(r.error_detail, style="yellow")
        else:
            status_cell = Text(f"{r.status_emoji} {r.status}", style="bold red")
            preview_cell = Text(r.error_detail, style="red")

        table.add_row(
            r.provider,
            r.key_name,
            r.masked_key,
            r.model,
            status_cell,
            f"{r.latency_ms:.0f} ms",
            preview_cell,
        )

    console.print("\n")
    console.print(table)

    avg_latency = (total_active_latency / active_count) if active_count > 0 else 0.0
    summary_text = (
        f"Summary Results:\n"
        f"  * Total Keys Tested: {len(results)}\n"
        f"  * Active & Operational: {active_count} / {len(results)} ({(active_count / len(results) * 100):.1f}%)\n"
        f"  * Degraded / Failed / Inactive: {len(results) - active_count}\n"
        f"  * Average Latency (Active Keys): {avg_latency:.1f} ms"
    )
    console.print(Panel(summary_text, border_style="green" if active_count == len(results) else "yellow"))

    return results


def main():
    parser = argparse.ArgumentParser(description="Test all Inference API keys in real-time.")
    parser.add_argument(
        "--mode",
        choices=["concurrent", "sequential"],
        default="concurrent",
        help="Test execution mode (concurrent or sequential)",
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="Per-key request timeout in seconds")
    parser.add_argument("--concurrency", type=int, default=6, help="Concurrent worker count")
    parser.add_argument("--provider", type=str, default=None, help="Filter to a specific provider")
    args = parser.parse_args()

    asyncio.run(
        run_all_key_tests(
            mode=args.mode,
            timeout=args.timeout,
            specific_provider=args.provider,
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
