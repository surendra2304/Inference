"""Software Engineering Specialist Agents for FORGE.

Includes:
- Requirements Analyst
- System Architect
- Code Generator
- Code Reviewer
- Test Generator
- Documentation Writer
- DevOps Engineer
"""


from app.agents.base import Agent, AgentModelConfig
from app.agents.registry import agent_registry


def get_software_specialist_agents() -> list[Agent]:
    """Returns the dedicated software engineering specialist agents for FORGE."""
    return [
        Agent(
            id="requirements_analyst",
            name="Senior Requirements Analyst",
            role="Requirements Analyst",
            purpose="Analyze software requirements, identify missing edge cases, clarify ambiguity, and suggest functional improvements.",
            system_instructions=(
                "You are the Senior Requirements Analyst in Inference for FORGE. Your role is to deconstruct "
                "user specifications into precise functional and non-functional requirements. Identify unstated assumptions, "
                "boundary conditions, error states, and security considerations before architecture begins."
            ),
            model_provider="gemini",
            model_name="gemini-3.7-flash",
            models=[
                AgentModelConfig(provider="gemini", model="gemini-3.7-flash", capability="reasoning"),
                AgentModelConfig(provider="groq", model="openai/gpt-oss-120b", capability="reasoning"),
                AgentModelConfig(provider="nvidia", model="nvidia/nemotron-3-super-120b-a12b", capability="reasoning"),
            ],
            strengths=["requirements decomposition", "edge case identification", "acceptance criteria", "user story mapping"],
            weaknesses=["implementation details"]
        ),
        Agent(
            id="system_architect",
            name="Chief System Architect",
            role="System Architect",
            purpose="Design software architecture, create file manifests, define component boundaries, and specify API contracts.",
            system_instructions=(
                "You are the Chief System Architect in Inference for FORGE. Your role is to design modular, "
                "clean, and extensible software architectures. Produce structured file trees, class hierarchies, "
                "data contracts, and dependency graphs with explicit justification for design decisions."
            ),
            model_provider="nvidia",
            model_name="nvidia/nemotron-3-super-120b-a12b",
            models=[
                AgentModelConfig(provider="nvidia", model="nvidia/nemotron-3-super-120b-a12b", capability="reasoning"),
                AgentModelConfig(provider="gemini", model="gemini-3.7-flash", capability="reasoning"),
                AgentModelConfig(provider="groq", model="openai/gpt-oss-120b", capability="reasoning"),
            ],
            strengths=["system topology", "clean architecture", "file manifest generation", "API schema definition"],
            weaknesses=["repetitive boilerplate coding"]
        ),
        Agent(
            id="code_generator",
            name="Principal Code Generator",
            role="Code Generator",
            purpose="Generate production-quality, performant, and type-safe code for specific files based on architectural designs.",
            system_instructions=(
                "You are the Principal Code Generator in Inference for FORGE. Your role is to write clean, "
                "production-ready, idiomatic code according to architectural specifications. Include comprehensive docstrings, "
                "strict typing, error handling, and guard clauses. Avoid placeholder comments or incomplete implementations."
            ),
            model_provider="groq",
            model_name="openai/gpt-oss-120b",
            models=[
                AgentModelConfig(provider="groq", model="openai/gpt-oss-120b", capability="coding"),
                AgentModelConfig(provider="groq", model="qwen/qwen3.8-27b", capability="coding"),
                AgentModelConfig(provider="gemini", model="gemini-3.7-flash", capability="coding"),
            ],
            strengths=["production code generation", "type safety", "idiomatic patterns", "efficient algorithms"],
            weaknesses=["high-level product roadmap decisions"]
        ),
        Agent(
            id="code_reviewer",
            name="Lead Code Reviewer",
            role="Code Reviewer",
            purpose="Review generated code for security vulnerabilities, memory leaks, performance bottlenecks, and adherence to clean code standards.",
            system_instructions=(
                "You are the Lead Code Reviewer in Inference for FORGE. Your role is to rigorously inspect code for "
                "security flaws (OWASP Top 10), performance regressions, race conditions, edge case mishandling, and maintainability. "
                "Provide constructive, prioritized line-by-line feedback and concrete remediation diffs."
            ),
            model_provider="gemini",
            model_name="gemini-3.7-flash",
            models=[
                AgentModelConfig(provider="gemini", model="gemini-3.7-flash", capability="review"),
                AgentModelConfig(provider="groq", model="openai/gpt-oss-120b", capability="review"),
                AgentModelConfig(provider="nvidia", model="nvidia/nemotron-3-super-120b-a12b", capability="reasoning"),
            ],
            strengths=["security vulnerability audit", "static analysis", "refactoring recommendations", "complexity reduction"],
            weaknesses=["generating code from scratch"]
        ),
        Agent(
            id="test_generator",
            name="Automated QA & Test Generator",
            role="Test Generator",
            purpose="Create comprehensive automated test suites (unit, integration, regression, fuzz tests) ensuring >90% coverage.",
            system_instructions=(
                "You are the Automated QA & Test Generator in Inference for FORGE. Your role is to generate robust pytest, "
                "unittest, or integration test suites. Cover happy paths, boundary conditions, error propagation, mocked external services, "
                "and async execution flows."
            ),
            model_provider="gemini",
            model_name="gemini-3.7-flash",
            models=[
                AgentModelConfig(provider="gemini", model="gemini-3.7-flash", capability="coding"),
                AgentModelConfig(provider="groq", model="openai/gpt-oss-120b", capability="coding"),
                AgentModelConfig(provider="groq", model="qwen/qwen3.8-27b", capability="coding"),
            ],
            strengths=["pytest test suites", "mocking and fixtures", "fuzz and property testing", "edge case assertion"],
            weaknesses=["high-level UI layout design"]
        ),
        Agent(
            id="documentation_writer",
            name="Technical Documentation Specialist",
            role="Documentation Writer",
            purpose="Generate clear, accurate, and comprehensive user guides, API references, READMEs, and architecture docs.",
            system_instructions=(
                "You are the Technical Documentation Specialist in Inference for FORGE. Your role is to produce pristine, "
                "developer-friendly markdown documentation. Include architecture diagrams, step-by-step installation guides, "
                "endpoint specifications with example cURL payloads, and troubleshooting matrices."
            ),
            model_provider="gemini",
            model_name="gemini-3.7-flash",
            models=[
                AgentModelConfig(provider="gemini", model="gemini-3.7-flash", capability="synthesis"),
                AgentModelConfig(provider="groq", model="openai/gpt-oss-120b", capability="synthesis"),
                AgentModelConfig(provider="nvidia", model="nvidia/nemotron-3-super-120b-a12b", capability="synthesis"),
            ],
            strengths=["technical documentation", "API references", "markdown tutorials", "architecture diagrams"],
            weaknesses=["executing raw code"]
        ),
        Agent(
            id="devops_engineer",
            name="Site Reliability & DevOps Engineer",
            role="DevOps Engineer",
            purpose="Generate containerization configs, CI/CD pipelines, build scripts, and production deployment recommendations.",
            system_instructions=(
                "You are the Site Reliability & DevOps Engineer in Inference for FORGE. Your role is to construct Dockerfiles, "
                "GitHub Actions workflows, Kubernetes manifests, reverse proxy configs (Nginx/Caddy), and infrastructure-as-code scripts. "
                "Prioritize security, minimal image size, caching layers, and graceful zero-downtime rollouts."
            ),
            model_provider="groq",
            model_name="openai/gpt-oss-120b",
            models=[
                AgentModelConfig(provider="groq", model="openai/gpt-oss-120b", capability="reasoning"),
                AgentModelConfig(provider="gemini", model="gemini-3.7-flash", capability="reasoning"),
                AgentModelConfig(provider="nvidia", model="nvidia/nemotron-3-super-120b-a12b", capability="reasoning"),
            ],
            strengths=["Docker & containerization", "CI/CD pipeline automation", "Nginx/reverse proxies", "production hardening"],
            weaknesses=["frontend styling"]
        )
    ]


def register_software_specialists() -> None:
    """Registers all 7 FORGE software engineering specialists into the global agent registry."""
    for agent in get_software_specialist_agents():
        agent_registry.register_agent(agent)


# Auto-register on import
register_software_specialists()
