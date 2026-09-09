"""Unit tests for the 10 Specialist Agent Roles and Registry."""

from app.agents.registry import agent_registry
from app.agents.roles import get_all_specialist_agents, register_all_specialists

EXPECTED_ROLES = [
    "researcher",
    "architect",
    "coder",
    "debugger",
    "security_analyst",
    "data_analyst",
    "critic",
    "fact_checker",
    "strategist",
    "synthesizer",
    "trading_analyst"
]


def test_specialist_agents_definition():
    agents = get_all_specialist_agents()
    assert len(agents) == 11
    agent_ids = [a.id for a in agents]
    for expected_id in EXPECTED_ROLES:
        assert expected_id in agent_ids


def test_agent_registry_contains_all_10_specialists():
    register_all_specialists()
    registered = agent_registry.list_agents()
    assert len(registered) >= 11

    for expected_id in EXPECTED_ROLES:
        agent = agent_registry.get_agent(expected_id)
        assert agent is not None, f"Agent {expected_id} not found in registry."
        assert agent.id == expected_id
        assert len(agent.name) > 0
        assert len(agent.role) > 0
        assert len(agent.purpose) > 0
        assert len(agent.system_instructions) > 20
        assert len(agent.model_provider) > 0
        assert len(agent.model_name) > 0
        assert len(agent.strengths) > 0
        assert agent.status == "active"

    # Verify agent primary model provider assignments
    provider_map = {a.id: a.model_provider for a in get_all_specialist_agents()}
    expected_providers = {
        "researcher": "gemini",
        "architect": "nvidia",
        "coder": "mistral",
        "debugger": "openrouter",
        "security_analyst": "nvidia",
        "data_analyst": "groq",
        "critic": "groq",
        "fact_checker": "gemini",
        "strategist": "groq",
        "synthesizer": "gemini",
        "trading_analyst": "groq"
    }
    assert provider_map == expected_providers

    # Verify Researcher specialized model list
    researcher = agent_registry.get_agent("researcher")
    assert researcher.model_name in ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.7-flash"]
    assert len(researcher.models) >= 3
    assert researcher.models[0].provider == "gemini"
    assert researcher.models[0].model in ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.7-flash"]
    assert researcher.models[0].capability == "research"
    assert researcher.models[1].provider == "openrouter"
    assert researcher.models[1].capability == "reasoning"
    assert researcher.models[2].provider == "cohere"
    assert researcher.models[2].capability == "research"


def test_agent_capability_lookup():
    register_all_specialists()

    threat_agents = agent_registry.get_agents_by_capability("threat")
    assert any(a.id == "security_analyst" for a in threat_agents)

    debug_agents = agent_registry.get_agents_by_capability("deadlock")
    assert any(a.id == "debugger" for a in debug_agents)

    code_agents = agent_registry.get_agents_by_capability("refactoring")
    assert any(a.id == "coder" for a in code_agents)


def test_agent_structured_response_validation():
    from app.agents.base import AgentResponse

    # 1. Valid JSON payload
    json_data = (
        '{"summary": "Use Redis for caching", "rationale": "High throughput sub-millisecond latency", '
        '"confidence": 0.95, "trade_offs": ["memory cost", "durability"], "assumptions": ["in-memory datasets"]}'
    )
    resp = AgentResponse.parse_raw_or_json(json_data)
    assert resp.summary == "Use Redis for caching"
    assert resp.confidence == 0.95
    assert "memory cost" in resp.trade_offs

    # 2. Markdown wrapped JSON payload
    fenced_json = (
        "```json\n"
        '{"summary": "Refactor to async generator", "confidence": 0.92, "dissent": "May increase complexity"}\n'
        "```"
    )
    resp_fenced = AgentResponse.parse_raw_or_json(fenced_json)
    assert resp_fenced.summary == "Refactor to async generator"
    assert resp_fenced.confidence == 0.92
    assert resp_fenced.dissent == "May increase complexity"

    # 3. Plain text fallback
    plain_text = "Direct analysis: modular monolith is better for initial speed."
    resp_plain = AgentResponse.parse_raw_or_json(plain_text)
    assert resp_plain.summary == plain_text
    assert resp_plain.confidence == 0.85


def test_all_10_specialist_model_lists_configured():
    register_all_specialists()
    agents = get_all_specialist_agents()
    assert len(agents) == 11

    for agent in agents:
        assert len(agent.models) >= 3, f"Agent {agent.id} does not have at least 3 configured models."
        for m in agent.models:
            assert m.provider in ["gemini", "nvidia", "mistral", "openrouter", "cohere", "huggingface", "groq"]
            assert len(m.model) > 0
            assert m.capability in ["research", "reasoning", "coding", "safety", "synthesis"]
