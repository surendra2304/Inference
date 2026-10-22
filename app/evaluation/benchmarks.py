"""Golden Benchmark Dataset for Inference Quality Control."""


from pydantic import BaseModel, Field


class BenchmarkTestCase(BaseModel):
    """An individual curated benchmark question with expected criteria."""
    id: str
    domain: str = Field(description="architecture, coding, debugging, security, reasoning, fact_checking")
    question: str
    expected_key_concepts: list[str] = Field(description="Core concepts or keywords that MUST be present")
    required_dissent_or_tradeoffs: list[str] = Field(description="Key trade-offs or failure modes that must be identified")
    ideal_mode: str = Field(default="debate", description="Expected ideal execution mode")
    minimum_expected_score: float = Field(default=0.80, ge=0.0, le=1.0)


GOLDEN_BENCHMARK_SUITE: list[BenchmarkTestCase] = [
    BenchmarkTestCase(
        id="bench_001_arch",
        domain="architecture",
        question="What architecture should I use for a local-first multi-agent AI system?",
        expected_key_concepts=[
            "modular orchestration",
            "provider-agnostic gateway",
            "sqlite",
            "cloud inference",
            "decoupled agent identity"
        ],
        required_dissent_or_tradeoffs=[
            "local compute vs cloud API dependency",
            "multi-agent orchestration latency vs single-model speed"
        ],
        ideal_mode="debate",
        minimum_expected_score=0.85
    ),
    BenchmarkTestCase(
        id="bench_002_code",
        domain="coding",
        question="Implement an asynchronous token bucket rate limiter with sliding window burst protection in Python.",
        expected_key_concepts=[
            "asyncio.Lock",
            "token refill rate",
            "capacity",
            "monotonic time",
            "time.perf_counter"
        ],
        required_dissent_or_tradeoffs=[
            "lock contention under high concurrency",
            "memory consumption of sliding timestamp logs vs pure token bucket"
        ],
        ideal_mode="review",
        minimum_expected_score=0.85
    ),
    BenchmarkTestCase(
        id="bench_003_debug",
        domain="debugging",
        question="Diagnose why concurrent writes to an SQLite database cause 'database is locked' errors in an async Python worker pool.",
        expected_key_concepts=[
            "write concurrency limitation",
            "WAL mode (Write-Ahead Logging)",
            "busy_timeout",
            "connection serialization",
            "single writer constraint"
        ],
        required_dissent_or_tradeoffs=[
            "WAL mode checkpoint overhead vs standard rollback journal",
            "thread pool connection pooling vs dedicated write queue"
        ],
        ideal_mode="review",
        minimum_expected_score=0.80
    ),
    BenchmarkTestCase(
        id="bench_004_sec",
        domain="security",
        question="Perform a comprehensive threat model for an AI agent authorized to execute read-only shell commands on a developer's machine.",
        expected_key_concepts=[
            "prompt injection via file contents",
            "command argument injection",
            "indirect prompt injection",
            "least privilege allowlists",
            "secret exfiltration via DNS/HTTP"
        ],
        required_dissent_or_tradeoffs=[
            "read-only command execution can still read private SSH keys and .env files",
            "sandboxing/containerization overhead vs native speed"
        ],
        ideal_mode="debate",
        minimum_expected_score=0.90
    ),
    BenchmarkTestCase(
        id="bench_005_reasoning",
        domain="reasoning",
        question="Compare dense 70B parameter models vs Mixture of Experts (MoE) 8x7B for multi-agent adversarial debate.",
        expected_key_concepts=[
            "active parameter count",
            "routing latency",
            "expert specialization",
            "reasoning depth",
            "memory footprint"
        ],
        required_dissent_or_tradeoffs=[
            "MoE inference speed vs coherence in long debate contexts",
            "dense model reasoning stability vs compute cost"
        ],
        ideal_mode="debate",
        minimum_expected_score=0.80
    )
]


def get_benchmark_by_id(bench_id: str) -> BenchmarkTestCase | None:
    """Retrieve a golden benchmark case by its ID."""
    for b in GOLDEN_BENCHMARK_SUITE:
        if b.id == bench_id:
            return b
    return None
