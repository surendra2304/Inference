# Unified API Provider & Software Engineering Specialists Architecture

Inference provides unified execution routing across 7 verified cloud providers (Gemini, Groq, Mistral, OpenRouter, NVIDIA, Cohere, HuggingFace) to simultaneously power both quantitative trading bots and the **FORGE** autonomous software engineering engine.

> [!IMPORTANT]
> **Advisory Invariant**: Model executions provide intelligence, code generation, and advisory analysis. Inference NEVER executes live trades, touches private keys, or applies unverified code directly.

---

## 1. Specialized FORGE Software Engineering Agents

| Agent ID | Display Name & Role | Primary Provider & Model | Core Strengths |
| :--- | :--- | :--- | :--- |
| `requirements_analyst` | Senior Requirements Analyst | Gemini (`gemini-3.7-flash`) | Requirements decomposition, edge cases, acceptance criteria |
| `system_architect` | Chief System Architect | NVIDIA (`nemotron-3-ultra-550b-a55b`) | File trees, clean architecture, class contracts |
| `code_generator` | Principal Code Generator | Groq (`openai/gpt-oss-120b`) | Idiomatic production code, type safety, algorithms |
| `code_reviewer` | Lead Code Reviewer | OpenRouter (`deepseek-v4-flash:free`) | OWASP security audit, static analysis, refactoring diffs |
| `test_generator` | Automated QA & Test Generator | Gemini (`gemini-3.7-flash`) | Pytest test suites, mocking, edge case coverage |
| `documentation_writer` | Technical Documentation Specialist | Cohere (`command-a-plus-05-2026`) | Architecture docs, tutorials, markdown specs |
| `devops_engineer` | Site Reliability & DevOps Engineer | Mistral (`mistral-large-2411`) | Docker, CI/CD workflows, reverse proxies |

---

## 2. Unified Provider Execution

`POST /v1/providers/execute`
- **Universal Payload**:
```json
{
  "provider": "auto",
  "agent_role": "system_architect",
  "prompt": "Design high-throughput microservices architecture",
  "context": {"rps": 5000},
  "max_tokens": 2000,
  "temperature": 0.7
}
```
- **Response**:
```json
{
  "provider_used": "nvidia",
  "model_used": "nvidia/nemotron-3-ultra-550b-a55b",
  "agent_role": "system_architect",
  "content": "...",
  "latency_ms": 28.4,
  "timestamp": 1787742000.0,
  "token_usage": {"total_tokens": 350},
  "status": "success"
}
```
