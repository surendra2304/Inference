"""Generic asynchronous DAG executor with dependency ordering and resource cleanup."""

import asyncio
import math
import time
from collections.abc import Callable, Coroutine
from typing import Any

from pydantic import BaseModel

from app.core.dag import DAGNode, ExecutionDAG
from app.utils.logger import logger


class NodeExecutionResult(BaseModel):
    """Execution outcome and metrics for an individual DAG node."""

    node_id: str
    status: str = "completed"  # completed, failed, cancelled, skipped
    output: Any = None
    latency_seconds: float = 0.0
    error: str | None = None
    attempt_count: int = 1


class _CooperativeNodeCancellation(Exception):
    """Internal signal: cancellation_event was set while a node was running."""


class DAGExecutor:
    """Executes topologically ordered layers with bounded retries and cancellation.

    Both global deadlines and the cooperative cancellation event actively interrupt
    in-flight node runners. Caller cancellation is still re-raised after recording
    node state, allowing outer task cancellation and deadline enforcement to work.
    """

    def __init__(
        self,
        dag: ExecutionDAG,
        cancellation_event: asyncio.Event | None = None,
        default_node_timeout_seconds: float = 30.0,
    ) -> None:
        if (
            isinstance(default_node_timeout_seconds, bool)
            or not isinstance(default_node_timeout_seconds, (int, float))
            or not math.isfinite(default_node_timeout_seconds)
            or default_node_timeout_seconds <= 0
        ):
            raise ValueError("default_node_timeout_seconds must be finite and positive")
        self.dag = dag
        self.cancellation_event = cancellation_event or asyncio.Event()
        self.default_node_timeout_seconds = float(default_node_timeout_seconds)
        self.results: dict[str, NodeExecutionResult] = {}

    async def execute(
        self,
        node_runner: Callable[[DAGNode, dict[str, Any]], Coroutine[Any, Any, Any]],
        overall_timeout_seconds: float | None = None,
    ) -> dict[str, NodeExecutionResult]:
        """Execute all dependency layers; every invocation starts with fresh results."""
        if overall_timeout_seconds is not None and (
            isinstance(overall_timeout_seconds, bool)
            or not isinstance(overall_timeout_seconds, (int, float))
            or not math.isfinite(overall_timeout_seconds)
            or overall_timeout_seconds <= 0
        ):
            raise ValueError("overall_timeout_seconds must be finite and positive")

        self.results = {}
        # Always rebuild: callers may mutate a graph after an earlier execution.
        layers = self.dag.build_layers()

        async def run_layers() -> dict[str, NodeExecutionResult]:
            for layer_idx, layer_nodes in enumerate(layers):
                if self.cancellation_event.is_set():
                    logger.warning("DAG execution cancelled before layer %d", layer_idx)
                    for node_id in layer_nodes:
                        self.results[node_id] = NodeExecutionResult(
                            node_id=node_id,
                            status="cancelled",
                            error="Task was cancelled",
                        )
                    continue

                # Independent nodes in the same layer may run concurrently.
                await asyncio.gather(
                    *(self._execute_single_node(node_id, node_runner) for node_id in layer_nodes)
                )

            return self.results

        if overall_timeout_seconds is None:
            return await run_layers()

        try:
            return await asyncio.wait_for(run_layers(), timeout=float(overall_timeout_seconds))
        except asyncio.TimeoutError:
            logger.error("DAG execution timed out after %.2fs", overall_timeout_seconds)
            # Active node tasks record cancellation when the deadline interrupts them.
            # Present a consistent deadline failure to callers instead of mixing
            # unfinished nodes marked "cancelled" with absent result entries.
            for node_id in self.dag.nodes:
                current = self.results.get(node_id)
                if current is None or current.status == "cancelled":
                    self.results[node_id] = NodeExecutionResult(
                        node_id=node_id,
                        status="failed",
                        error="Overall DAG timeout exceeded",
                        latency_seconds=current.latency_seconds if current else 0.0,
                        attempt_count=current.attempt_count if current else 1,
                    )
            return self.results

    async def _run_node_attempt(
        self,
        node: DAGNode,
        node_runner: Callable[[DAGNode, dict[str, Any]], Coroutine[Any, Any, Any]],
        dependency_outputs: dict[str, Any],
    ) -> Any:
        """Run one node while racing completion, cooperative cancellation, and timeout."""
        operation = asyncio.create_task(node_runner(node, dependency_outputs))
        cancellation_waiter = asyncio.create_task(self.cancellation_event.wait())
        try:
            done, _ = await asyncio.wait(
                {operation, cancellation_waiter},
                timeout=self.default_node_timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            # If both signal in the same loop turn, prefer a completed operation.
            if operation in done:
                return await operation
            if cancellation_waiter in done and self.cancellation_event.is_set():
                raise _CooperativeNodeCancellation("Task cancellation event was set")
            raise asyncio.TimeoutError(
                f"Node {node.node_id} timed out after {self.default_node_timeout_seconds:.2f}s"
            )
        finally:
            for task in (operation, cancellation_waiter):
                if not task.done():
                    task.cancel()
            # Retrieve any exceptions from tasks cancelled during timeout/cancellation.
            await asyncio.gather(operation, cancellation_waiter, return_exceptions=True)

    async def _execute_single_node(
        self,
        node_id: str,
        node_runner: Callable[[DAGNode, dict[str, Any]], Coroutine[Any, Any, Any]],
        max_attempts: int = 2,
        max_retries: int | None = None,
    ) -> NodeExecutionResult:
        """Execute one node, skipping failed dependencies and recording each outcome."""
        if max_retries is not None:
            if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
                raise ValueError("max_retries must be a non-negative integer")
            max_attempts = max_retries + 1
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")

        node = self.dag.nodes[node_id]
        node_start = time.perf_counter()

        if self.cancellation_event.is_set():
            result = NodeExecutionResult(node_id=node_id, status="cancelled", error="Cancelled before start")
            self.results[node_id] = result
            return result

        dependency_outputs = {
            dependency_id: self.results[dependency_id].output
            for dependency_id in node.dependencies
            if dependency_id in self.results and self.results[dependency_id].status == "completed"
        }
        failed_dependencies = [
            dependency_id
            for dependency_id in node.dependencies
            if dependency_id in self.results
            and self.results[dependency_id].status in ("failed", "cancelled", "skipped")
        ]
        if failed_dependencies:
            result = NodeExecutionResult(
                node_id=node_id,
                status="skipped",
                error=f"Skipped due to dependency failure: {failed_dependencies}",
            )
            self.results[node_id] = result
            return result

        for attempt in range(1, max_attempts + 1):
            if self.cancellation_event.is_set():
                result = NodeExecutionResult(
                    node_id=node_id,
                    status="cancelled",
                    latency_seconds=time.perf_counter() - node_start,
                    error="Node cancelled before attempt",
                    attempt_count=attempt,
                )
                self.results[node_id] = result
                return result

            try:
                output = await self._run_node_attempt(node, node_runner, dependency_outputs)
                result = NodeExecutionResult(
                    node_id=node_id,
                    status="completed",
                    output=output,
                    latency_seconds=time.perf_counter() - node_start,
                    attempt_count=attempt,
                )
                self.results[node_id] = result
                return result
            except _CooperativeNodeCancellation:
                result = NodeExecutionResult(
                    node_id=node_id,
                    status="cancelled",
                    latency_seconds=time.perf_counter() - node_start,
                    error="Node cancelled by cooperative cancellation event",
                    attempt_count=attempt,
                )
                self.results[node_id] = result
                return result
            except asyncio.CancelledError:
                result = NodeExecutionResult(
                    node_id=node_id,
                    status="cancelled",
                    latency_seconds=time.perf_counter() - node_start,
                    error="Node task cancelled",
                    attempt_count=attempt,
                )
                self.results[node_id] = result
                # Do not swallow external cancellation: wait_for() must observe the
                # cancellation and raise TimeoutError when its deadline expires.
                raise
            except Exception as exc:
                if attempt >= max_attempts:
                    result = NodeExecutionResult(
                        node_id=node_id,
                        status="failed",
                        latency_seconds=time.perf_counter() - node_start,
                        error=str(exc),
                        attempt_count=attempt,
                    )
                    self.results[node_id] = result
                    return result
                # Avoid retrying after cancellation while waiting between attempts.
                await asyncio.sleep(0.5)

        result = NodeExecutionResult(node_id=node_id, status="failed", error="Unknown node execution error")
        self.results[node_id] = result
        return result
