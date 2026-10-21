from __future__ import annotations

import asyncio

import pytest

from app.core.dag import DAGNode, ExecutionDAG
from app.core.dag_executor import DAGExecutor


def node(node_id: str, *dependencies: str) -> DAGNode:
    return DAGNode(
        node_id=node_id,
        agent_id=f"agent-{node_id}",
        agent_role=node_id,
        dependencies=list(dependencies),
    )


def test_topological_layers_preserve_dependencies_and_parallelize_independent_nodes():
    dag = ExecutionDAG()
    dag.add_node(node("a"))
    dag.add_node(node("b"))
    dag.add_node(node("c", "a"))
    dag.add_node(node("d", "b", "c"))

    assert dag.build_layers() == [["a", "b"], ["c"], ["d"]]


def test_build_layers_rejects_missing_dependencies_instead_of_running_unblocked():
    dag = ExecutionDAG()
    dag.add_node(node("child", "not-registered"))

    with pytest.raises(ValueError, match="missing dependencies"):
        dag.build_layers()

    assert dag.layers == []


@pytest.mark.parametrize(
    "nodes",
    [
        [node("a", "b"), node("b", "a")],
        [node("a", "a")],
    ],
)
def test_build_layers_rejects_cycles(nodes):
    dag = ExecutionDAG()
    for item in nodes:
        dag.add_node(item)

    with pytest.raises(ValueError, match="Cycle detected"):
        dag.build_layers()
    assert dag.layers == []


def test_build_layers_rejects_duplicate_dependency_edges():
    dag = ExecutionDAG()
    dag.add_node(node("root"))
    dag.add_node(node("child", "root", "root"))

    with pytest.raises(ValueError, match="duplicate dependencies"):
        dag.build_layers()


def test_dag_rejects_duplicate_node_ids():
    dag = ExecutionDAG()
    dag.add_node(node("same"))

    with pytest.raises(ValueError, match="Duplicate DAG node id"):
        dag.add_node(node("same"))

    assert list(dag.nodes) == ["same"]


def test_adding_node_invalidates_previously_computed_layers():
    dag = ExecutionDAG()
    dag.add_node(node("a"))
    assert dag.build_layers() == [["a"]]
    dag.add_node(node("b", "a"))

    assert dag.layers == []
    assert dag.build_layers() == [["a"], ["b"]]


@pytest.mark.asyncio
async def test_executor_passes_only_completed_dependency_outputs():
    dag = ExecutionDAG()
    dag.add_node(node("source"))
    dag.add_node(node("dependent", "source"))
    execution_order = []

    async def run(item, dependency_outputs):
        execution_order.append(item.node_id)
        if item.node_id == "source":
            assert dependency_outputs == {}
            return {"answer": 42}
        assert dependency_outputs == {"source": {"answer": 42}}
        return dependency_outputs["source"]["answer"] + 1

    results = await DAGExecutor(dag).execute(run)

    assert execution_order == ["source", "dependent"]
    assert results["source"].output == {"answer": 42}
    assert results["dependent"].output == 43
    assert all(result.status == "completed" for result in results.values())


@pytest.mark.asyncio
async def test_executor_skips_dependents_when_required_node_fails():
    dag = ExecutionDAG()
    dag.add_node(node("broken"))
    dag.add_node(node("dependent", "broken"))

    async def run(item, _dependency_outputs):
        if item.node_id == "broken":
            raise RuntimeError("provider unavailable")
        pytest.fail("dependent node must never run after upstream failure")

    results = await DAGExecutor(dag).execute(run, overall_timeout_seconds=3.0)

    assert results["broken"].status == "failed"
    assert results["broken"].attempt_count == 2
    assert results["dependent"].status == "skipped"


@pytest.mark.asyncio
async def test_node_retry_succeeds_and_reports_attempt_count():
    dag = ExecutionDAG()
    dag.add_node(node("retry"))
    attempts = 0

    async def run(_item, _dependency_outputs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient")
        return "recovered"

    results = await DAGExecutor(dag).execute(run)

    assert results["retry"].status == "completed"
    assert results["retry"].output == "recovered"
    assert results["retry"].attempt_count == 2


@pytest.mark.asyncio
async def test_cooperative_cancel_event_interrupts_active_node_runner():
    dag = ExecutionDAG()
    dag.add_node(node("long-running"))
    cancellation = asyncio.Event()
    started = asyncio.Event()
    runner_cancelled = asyncio.Event()

    async def run(_item, _dependency_outputs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            runner_cancelled.set()
            raise

    executor = DAGExecutor(dag, cancellation_event=cancellation, default_node_timeout_seconds=10.0)
    task = asyncio.create_task(executor.execute(run))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    cancellation.set()

    results = await asyncio.wait_for(task, timeout=1.0)

    assert runner_cancelled.is_set()
    assert results["long-running"].status == "cancelled"
    assert "cooperative" in results["long-running"].error


@pytest.mark.asyncio
async def test_external_cancellation_is_recorded_and_propagated():
    dag = ExecutionDAG()
    dag.add_node(node("long-running"))
    started = asyncio.Event()

    async def run(_item, _dependency_outputs):
        started.set()
        await asyncio.Event().wait()

    executor = DAGExecutor(dag)
    task = asyncio.create_task(executor.execute(run))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert executor.results["long-running"].status == "cancelled"


@pytest.mark.asyncio
async def test_overall_timeout_returns_explicit_deadline_failure():
    dag = ExecutionDAG()
    dag.add_node(node("long-running"))
    runner_cancelled = asyncio.Event()

    async def run(_item, _dependency_outputs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            runner_cancelled.set()
            raise

    results = await DAGExecutor(dag, default_node_timeout_seconds=10.0).execute(
        run,
        overall_timeout_seconds=0.05,
    )

    assert runner_cancelled.is_set()
    assert results["long-running"].status == "failed"
    assert results["long-running"].error == "Overall DAG timeout exceeded"


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_executor_rejects_invalid_node_timeout(timeout):
    dag = ExecutionDAG()
    with pytest.raises(ValueError, match="default_node_timeout_seconds"):
        DAGExecutor(dag, default_node_timeout_seconds=timeout)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
@pytest.mark.asyncio
async def test_executor_rejects_invalid_overall_timeout(timeout):
    dag = ExecutionDAG()

    async def unused_runner(_item, _dependency_outputs):
        return None

    with pytest.raises(ValueError, match="overall_timeout_seconds"):
        await DAGExecutor(dag).execute(unused_runner, overall_timeout_seconds=timeout)
