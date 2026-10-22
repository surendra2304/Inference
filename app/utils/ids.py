"""Identifier generator utilities for tasks, runs, and debates."""

import uuid


def generate_id(prefix: str = "id") -> str:
    """Generate a unique hex UUID with a semantic prefix."""
    unique_suffix = uuid.uuid4().hex[:12]
    return f"{prefix}_{unique_suffix}"


def generate_task_id() -> str:
    """Generate a unique identifier for a user/system task."""
    return generate_id("task")


def generate_run_id() -> str:
    """Generate a unique identifier for an execution run or sub-agent call."""
    return generate_id("run")


def generate_debate_id() -> str:
    """Generate a unique identifier for a multi-agent debate session."""
    return generate_id("deb")


def generate_message_id() -> str:
    """Generate a unique identifier for a conversation or debate message."""
    return generate_id("msg")


def generate_deterministic_id(namespace: str, name: str) -> str:
    """Generate a reproducible deterministic UUID based on a namespace and input name."""
    generated_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"{namespace}:{name}")
    return f"{namespace}_{generated_uuid.hex[:12]}"
