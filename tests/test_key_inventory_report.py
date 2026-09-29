from app.core import key_inventory


class ExampleSettings:
    def get_provider_keys(self, provider):
        return {
            "GEMINI": ["gemini-secret-a", "gemini-secret-b"],
            "GROQ": ["groq-secret"],
        }.get(provider, [])


def test_provider_key_counts_report_only_provider_counts():
    assert key_inventory.provider_key_counts(ExampleSettings()) == {
        "GEMINI": 2,
        "GROQ": 1,
        "MISTRAL": 0,
        "OPENROUTER": 0,
        "COHERE": 0,
        "HUGGINGFACE": 0,
        "NVIDIA": 0,
    }


def test_command_never_prints_credential_values(monkeypatch, capsys):
    monkeypatch.setattr(key_inventory, "settings", ExampleSettings())

    key_inventory.main()

    output = capsys.readouterr().out
    assert "GEMINI: 2" in output
    assert "TOTAL_CONFIGURED_PROVIDER_KEYS: 3" in output
    assert "gemini-secret-a" not in output
    assert "groq-secret" not in output
    assert "quota were not checked" in output
