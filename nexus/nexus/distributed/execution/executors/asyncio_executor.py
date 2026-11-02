from __future__ import annotations

import asyncio
import contextlib
import threading

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar

from nexus.distributed.execution.executors.base import (
    ExecutionDriver,
    ExecutionHandle,
    ExecutionInfoDriver,
    ExecutionReporter,
    Executor,
    ExecutorCapacitySnapshot,
    ExecutorHealthSnapshot,
    RuntimeEndpointDriver,
    SseEventStreamingDriver,
    run_driver_with_failure_reporting,
)
from nexus.distributed.observability.event_loop import AiomonitorHandle, LoopAiomonitorConfig, start_aiomonitor_for_loop
from nexus.distributed.utils.threadsafe import await_threadsafe


T = TypeVar('T')


@dataclass(slots=True)
class _LocalExecutionInstance:
    assignment_id: str
    execution_id: str
    driver: ExecutionDriver
    handle: ExecutionHandle
    task: asyncio.Task[None]
    slot_cost: int


class AsyncioExecutor(Executor):
    """Runs execution drivers on a dedicated child-thread event loop."""

    instance_mode = 'asyncio'

    def __init__(
        self,
        *,
        capacity_total: int = 1,
        aiomonitor_config: LoopAiomonitorConfig | None = None,
        aiomonitor_name: str = 'asyncio-executor',
        aiomonitor_port_offset: int = 1,
        bridge_timeout: float = 5.0,
    ) -> None:
        if bridge_timeout <= 0:
            raise ValueError('bridge_timeout must be positive')
        self.capacity_total = capacity_total
        self.aiomonitor_config = aiomonitor_config
        self.aiomonitor_name = aiomonitor_name
        self.aiomonitor_port_offset = aiomonitor_port_offset
        self.bridge_timeout = bridge_timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._aiomonitor_handle: AiomonitorHandle | None = None
        self._ready = threading.Event()
        self._closed = False
        self._assignments: dict[str, _LocalExecutionInstance] = {}

    async def start(self) -> None:
        if self._loop is not None:
            return

        self._closed = False
        self._ready.clear()
        self._thread = threading.Thread(target=self._run_loop, name='nexus-distributed-executor', daemon=True)
        self._thread.start()
        await asyncio.to_thread(self._ready.wait)

    async def close(self) -> None:
        if self._loop is None:
            return

        assignment_ids = await self._run(self._list_assignment_ids)
        for assignment_id in assignment_ids:
            with contextlib.suppress(Exception):
                await self.stop_assignment(assignment_id, timeout=1.0)

        loop = self._loop
        self._closed = True
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join, 2.0)
        self._loop = None
        self._thread = None

    async def has_assignment(self, assignment_id: str) -> bool:
        if self._loop is None or self._closed:
            return False
        return await self._run(lambda: self._has_assignment(assignment_id))

    async def capacity_snapshot(self) -> ExecutorCapacitySnapshot:
        if self._loop is None or self._closed:
            return ExecutorCapacitySnapshot(
                capacity_total=self.capacity_total,
                running_slot_cost=0,
                available_slots=self.capacity_total,
            )
        return await self._run(self._capacity_snapshot)

    async def probe_health(self, *, timeout: float, hang_timeout: float, now: float) -> ExecutorHealthSnapshot:
        del hang_timeout, now
        try:
            await self._run(self._ping, timeout=timeout)
        except (RuntimeError, TimeoutError):
            return ExecutorHealthSnapshot(responsive=False, effective_capacity=0, unresponsive_workers=1)
        return ExecutorHealthSnapshot(responsive=True, effective_capacity=self.capacity_total)

    async def start_assignment(
        self,
        assignment_id: str,
        execution_id: str,
        driver: ExecutionDriver,
        spec: dict[str, Any],
        config: dict[str, Any],
        reporter: ExecutionReporter,
        *,
        slot_cost: int = 1,
    ) -> ExecutionHandle:
        if slot_cost < 1:
            raise ValueError('slot_cost must be at least 1')
        return await self._run(
            lambda: self._spawn_assignment(assignment_id, execution_id, driver, spec, config, reporter, slot_cost)
        )

    async def stop_assignment(self, assignment_id: str, timeout: float = 10.0) -> None:
        await self._run(
            lambda: self._stop_assignment(assignment_id, timeout),
            timeout=max(0.0, timeout) + 1.0,
        )

    async def submit_command(self, assignment_id: str, command_type: str, payload: dict[str, Any]) -> None:
        await self._run(lambda: self._submit_command(assignment_id, command_type, payload))

    async def collect_events_after(
        self, assignment_id: str, cursor: int, *, timeout: float = 15.0
    ) -> list[tuple[int, bytes]]:
        return await self._call_sse_driver(assignment_id, cursor, timeout=timeout)

    async def get_runtime_base_url(self, assignment_id: str, runtime_name: str) -> str:
        return await self._call_runtime_driver(assignment_id, runtime_name)

    async def get_assignment_info(self, assignment_id: str) -> dict[str, Any]:
        return await self._call_info_driver(assignment_id)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        if self.aiomonitor_config is not None:
            self._aiomonitor_handle = start_aiomonitor_for_loop(
                loop,
                config=self.aiomonitor_config,
                name=self.aiomonitor_name,
                port_offset=self.aiomonitor_port_offset,
            )
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            if self._aiomonitor_handle is not None:
                self._aiomonitor_handle.close()
                self._aiomonitor_handle = None
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._closed:
            raise RuntimeError('AsyncioExecutor is not running')
        return self._loop

    async def _run(self, call: Callable[[], Coroutine[Any, Any, T]], *, timeout: float | None = None) -> T:
        loop = self._require_loop()
        deadline = self.bridge_timeout if timeout is None else timeout
        if loop is asyncio.get_running_loop():
            return await asyncio.wait_for(call(), timeout=deadline)
        future = asyncio.run_coroutine_threadsafe(call(), loop)
        return await await_threadsafe(future, timeout=deadline)

    async def _list_assignment_ids(self) -> list[str]:
        return list(self._assignments)

    async def _ping(self) -> None:
        await asyncio.sleep(0)

    async def _has_assignment(self, assignment_id: str) -> bool:
        return assignment_id in self._assignments

    async def _capacity_snapshot(self) -> ExecutorCapacitySnapshot:
        running_slot_cost = sum(item.slot_cost for item in self._assignments.values())
        return ExecutorCapacitySnapshot(
            capacity_total=self.capacity_total,
            running_slot_cost=running_slot_cost,
            available_slots=max(0, self.capacity_total - running_slot_cost),
        )

    def _available_slots(self) -> int:
        return max(0, self.capacity_total - sum(item.slot_cost for item in self._assignments.values()))

    async def _spawn_assignment(
        self,
        assignment_id: str,
        execution_id: str,
        driver: ExecutionDriver,
        spec: dict[str, Any],
        config: dict[str, Any],
        reporter: ExecutionReporter,
        slot_cost: int,
    ) -> ExecutionHandle:
        handle = ExecutionHandle(execution_id=execution_id, assignment_id=assignment_id)

        async def run_driver() -> None:
            await run_driver_with_failure_reporting(driver, handle, spec, config, reporter)

        if assignment_id in self._assignments:
            raise ValueError(f'assignment {assignment_id} is already running')
        if self._available_slots() < slot_cost:
            raise RuntimeError('executor has no available slots')

        task = asyncio.create_task(run_driver(), name=f'nexus-execution-{assignment_id}')
        assignment = _LocalExecutionInstance(
            assignment_id=assignment_id,
            execution_id=execution_id,
            driver=driver,
            handle=handle,
            task=task,
            slot_cost=slot_cost,
        )

        self._assignments[assignment_id] = assignment

        def on_done(done_task: asyncio.Task[None]) -> None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                done_task.result()
            self._remove_assignment(assignment_id)

        task.add_done_callback(on_done)
        return handle

    async def _stop_assignment(self, assignment_id: str, timeout: float) -> None:
        assignment = self._assignments.get(assignment_id)
        if assignment is None:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(timeout, 0.0)

        remaining = deadline - loop.time()
        if remaining > 0:
            with contextlib.suppress(asyncio.TimeoutError, Exception):
                await asyncio.wait_for(assignment.driver.stop(assignment.handle), timeout=remaining)

        if not assignment.task.done():
            remaining = deadline - loop.time()
            if remaining > 0:
                with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    await asyncio.wait_for(asyncio.shield(assignment.task), timeout=remaining)

        if not assignment.task.done():
            assignment.task.cancel()
            remaining = deadline - loop.time()
            if remaining > 0:
                with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                    await asyncio.wait_for(assignment.task, timeout=remaining)

        if assignment.task.done():
            self._remove_assignment(assignment.assignment_id)

    async def _submit_command(self, assignment_id: str, command_type: str, payload: dict[str, Any]) -> None:
        assignment = self._assignments.get(assignment_id)
        if assignment is None:
            raise KeyError(assignment_id)
        await assignment.driver.submit_command(assignment.handle, command_type, payload)

    def _remove_assignment(self, assignment_id: str) -> _LocalExecutionInstance | None:
        return self._assignments.pop(assignment_id, None)

    async def _call_sse_driver(self, assignment_id: str, cursor: int, *, timeout: float) -> list[tuple[int, bytes]]:
        return await self._run(
            lambda: self._collect_events_after(assignment_id, cursor, timeout=timeout),
            timeout=max(0.0, timeout) + 2.0,
        )

    async def _call_runtime_driver(self, assignment_id: str, runtime_name: str) -> str:
        return await self._run(lambda: self._get_runtime_base_url(assignment_id, runtime_name))

    async def _call_info_driver(self, assignment_id: str) -> dict[str, Any]:
        return await self._run(lambda: self._get_assignment_info(assignment_id))

    async def _collect_events_after(
        self, assignment_id: str, cursor: int, *, timeout: float
    ) -> list[tuple[int, bytes]]:
        assignment = self._assignments.get(assignment_id)
        if assignment is None:
            raise KeyError(assignment_id)
        if not isinstance(assignment.driver, SseEventStreamingDriver):
            raise ValueError('events_not_supported')
        return await assignment.driver.collect_sse_events_after(cursor, timeout=timeout)

    async def _get_runtime_base_url(self, assignment_id: str, runtime_name: str) -> str:
        assignment = self._assignments.get(assignment_id)
        if assignment is None:
            raise KeyError(assignment_id)
        if not isinstance(assignment.driver, RuntimeEndpointDriver):
            raise ValueError('runtime_proxy_not_supported')
        return await assignment.driver.get_runtime_base_url(runtime_name)

    async def _get_assignment_info(self, assignment_id: str) -> dict[str, Any]:
        assignment = self._assignments.get(assignment_id)
        if assignment is None:
            raise KeyError(assignment_id)
        if not isinstance(assignment.driver, ExecutionInfoDriver):
            raise ValueError('info_not_supported')
        return await assignment.driver.get_execution_info(assignment.handle)
