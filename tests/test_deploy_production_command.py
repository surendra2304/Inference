"""Regression tests for safe deployment command execution."""

from __future__ import annotations

import subprocess

import pytest

import deploy_production


def test_run_command_uses_argument_array_without_shell(monkeypatch):
    observed = {}

    def fake_run(args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(deploy_production.subprocess, "run", fake_run)

    assert deploy_production.run_command(["docker", "compose", "ps"]) is True
    assert observed["args"] == ["docker", "compose", "ps"]
    assert observed["kwargs"]["shell"] is False
    assert observed["kwargs"]["check"] is False


def test_run_command_string_is_split_without_shell_interpretation(monkeypatch):
    observed = {}

    def fake_run(args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(deploy_production.subprocess, "run", fake_run)

    assert deploy_production.run_command("docker compose -f 'prod stack.yml' ps") is True
    assert observed["args"] == ["docker", "compose", "-f", "prod stack.yml", "ps"]
    assert observed["kwargs"]["shell"] is False


def test_shell_metacharacters_remain_plain_arguments(monkeypatch):
    observed = {}

    def fake_run(args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(deploy_production.subprocess, "run", fake_run)

    assert deploy_production.run_command("echo safe; whoami") is True
    assert observed["args"] == ["echo", "safe;", "whoami"]
    assert observed["kwargs"]["shell"] is False


@pytest.mark.parametrize("command", ["", "   ", [], (), ["docker", ""], ["docker", None]])
def test_run_command_rejects_empty_arguments(command, capsys):
    assert deploy_production.run_command(command) is False
    assert "[ERROR]" in capsys.readouterr().out


def test_run_command_rejects_malformed_quoted_string(capsys):
    assert deploy_production.run_command("docker compose 'unterminated") is False
    assert "Invalid command syntax" in capsys.readouterr().out


def test_run_command_returns_false_when_subprocess_fails(monkeypatch, capsys):
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 17)

    monkeypatch.setattr(deploy_production.subprocess, "run", fake_run)

    assert deploy_production.run_command(["docker", "info"]) is False
    assert "return code 17" in capsys.readouterr().out


def test_run_command_returns_false_when_executable_cannot_start(monkeypatch, capsys):
    def fake_run(args, **kwargs):
        raise FileNotFoundError("binary not found")

    monkeypatch.setattr(deploy_production.subprocess, "run", fake_run)

    assert deploy_production.run_command(["missing-tool"]) is False
    output = capsys.readouterr().out
    assert "Command could not be started" in output
    assert "FileNotFoundError" in output
