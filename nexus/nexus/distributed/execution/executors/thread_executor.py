from __future__ import annotations

import asyncio
import contextlib
import threading

from dataclasses import dataclass
from typing import Any, Literal

from nexus.distributed.execution.executors.base import (
    ExecutionDriver,
    ExecutionHandle,
    ExecutionInfoDriver,
    ExecutionReporter,
    Executor,
    ExecutorCapacitySnapshot,
    ExecutorHealthSnapshot,
    QuarantinedAssignment,
    RuntimeEndpointDriver,
    SseEventStreamingDriver,
    run_driver_with_failure_reporting,
)
from nexus.distributed.observability.event_loop import AiomonitorHandle, LoopAiomonitorConfig, start_aiomonitor_for_loop
from nexus.distributed.utils.threadsafe import await_threadsafe


ThreadStrategy = Literal['pooled', 'per_execution']


@dataclass(slots=True)
class _LocalExecutionInstance:
    assignment_id: str
    execution_id: str
    driver: ExecutionDriver
    handle: ExecutionHandle
    task: asyncio.Task[None]
    worker: _ThreadWorker
    slot_cost: int
    stop_waiter_active: bool = False


@dataclass(slots=True)
class _StartRequest:
    cancelled: bool = False


class _ThreadWorker:
    def __init__(self, worker_id: str, *, aiomonitor_config: LoopAiomonitorConfig | None = None) -> None:
        self.worker_id = worker_id
        self.aiomonitor_config = aiomonitor_config
        self.thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.aiomonitor_handle: AiomonitorHandle | None = None
        self.ready = threading.Event()
        self.tasks_completed = 0
        self.current_assignment_id: str | None = None
        self.state = 'starting'
        self.health_state = 'healthy'
        self.unresponsive_since: float | None = None

    async def start(self) -> None:
        self.ready.clear()
        self.thread = threading.Thread(target=self._run_loop, name=f'nexus-dist-worker-{self.worker_id}', daemon=True)
        self.thread.start()
        await asyncio.to_thread(self.ready.wait)

    async def stop(self, *, join_timeout: float = 2.0) -> None:
        loop = self.loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if self.thread is not None and self.thread is not threading.current_thread():
            await asyncio.to_thread(self.thread.join, join_timeout)
        if self.thread is None or not self.thread.is_alive():
            self.state = 'closed'
        else:
            self.state = 'closing'

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loop = loop
        if self.aiomonitor_config is not None:
            self.aiomonitor_handle = start_aiomonitor_for_loop(
                loop,
                config=self.aiomonitor_config,
                name=f'thread-worker-{self.worker_id}',
                port_offset=10 + int(self.worker_id),
            )
        self.ready.set()
        try:
            loop.run_forever()
        finally:
            if self.aiomonitor_handle is not None:
                self.aiomonitor_handle.close()
                self.aiomonitor_handle = None
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            self.state = 'closed'


class ThreadExecutor(Executor):
    """Runs each assignment on a worker thread with its own event loop."""

    instance_mode = 'thread'

    def __init__(
        self,
        *,
        strategy: ThreadStrategy = 'per_execution',
        max_threads: int = 1,
        max_tasks_per_thread: int = 100,
        thread_join_timeout: float = 2.0,
        aiomonitor_config: LoopAiomonitorConfig | None = None,
        bridge_timeout: float = 5.0,
    ) -> None:
        if strategy not in {'pooled', 'per_execution'}:
            raise ValueError(f'unsupported thread executor strategy: {strategy}')
        if max_threads < 1:
            raise ValueError('max_threads must be at least 1')
        if max_tasks_per_thread < 1:
            raise ValueError('max_tasks_per_thread must be at least 1')
        if bridge_timeout <= 0:
            raise ValueError('bridge_timeout must be positive')
        self.strategy = strategy
        self.max_threads = max_threads
        self.max_tasks_per_thread = max_tasks_per_thread
        self.thread_join_timeout = thread_join_timeout
        self.aiomonitor_config = aiomonitor_config
        self.bridge_timeout = bridge_timeout
        self._started = False
        self._closed = False
        self._next_worker_id = 1
        self._workers: dict[str, _ThreadWorker] = {}
        self._idle_workers: list[_ThreadWorker] = []
        self._retiring_workers: list[_ThreadWorker] = []
        self._assignments: dict[str, _LocalExecutionInstance] = {}
        self._quarantined_events: dict[str, QuarantinedAssignment] = {}
        self._lock = threading.RLock()
        self._stuck_slot_cost = 0

    async def start(self) -> None:
        self._started = True
        self._closed = False

    async def close(self) -> None:
        if not self._started:
            return

        assignment_ids = list(self._assignments)
        for assignment_id in assignment_ids:
            with contextlib.suppress(Exception):
                await self.stop_assignment(assignment_id, timeout=1.0)

        workers = list(self._workers.values()) + list(self._retiring_workers)
        for worker in workers:
            with contextlib.suppress(Exception):
                await worker.stop(join_timeout=self.thread_join_timeout)
        with self._lock:
            self._assignments.clear()
            self._idle_workers.clear()
            self._retiring_workers.clear()
            self._workers.clear()
            self._quarantined_events.clear()
            self._stuck_slot_cost = 0
            self._closed = True
            self._started = False

    async def has_assignment(self, assignment_id: str) -> bool:
        with self._lock:
            return assignment_id in self._assignments

    async def capacity_snapshot(self) -> ExecutorCapacitySnapshot:
        with self._lock:
            capacity_total = self._effective_capacity_locked()
            running_slot_cost = self._running_slot_cost_locked()
            return ExecutorCapacitySnapshot(
                capacity_total=capacity_total,
                running_slot_cost=running_slot_cost,
                available_slots=max(0, capacity_total - running_slot_cost),
            )

    async def probe_health(self, *, timeout: float, hang_timeout: float, now: float) -> ExecutorHealthSnapshot:
        with self._lock:
            self._reap_retiring_workers_locked()
            for worker_id, worker in list(self._workers.items()):
                if worker.state == 'closed':
                    if worker.current_assignment_id is not None and worker.health_state != 'quarantined':
                        self._quarantine_worker_locked(worker)
                    else:
                        self._workers.pop(worker_id, None)
            workers = [
                worker
                for worker in self._workers.values()
                if worker.state != 'closed' and worker.health_state != 'quarantined'
            ]
        await asyncio.gather(
            *(self._probe_worker(worker, timeout=timeout, hang_timeout=hang_timeout, now=now) for worker in workers)
        )
        with self._lock:
            unresponsive_workers = sum(
                worker.state != 'closed' and worker.health_state in {'suspect', 'quarantined'}
                for worker in self._workers.values()
            )
            suspect_idle_slot_cost = sum(
                1 for worker in self._workers.values() if worker.health_state == 'suspect' and worker.state == 'idle'
            )
            quarantined_assignments = tuple(self._quarantined_events.values())
            effective_capacity = max(0, self._effective_capacity_locked() - suspect_idle_slot_cost)
        return ExecutorHealthSnapshot(
            responsive=True,
            effective_capacity=effective_capacity,
            unresponsive_workers=unresponsive_workers,
            quarantined_assignments=quarantined_assignments,
            recycle_requested=bool(quarantined_assignments),
        )

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
        self._require_started()
        if slot_cost < 1:
            raise ValueError('slot_cost must be at least 1')
        with self._lock:
            if assignment_id in self._assignments:
                raise ValueError(f'assignment {assignment_id} is already running')
            if self._available_slots_locked() < slot_cost:
                raise RuntimeError('executor has no available slots')

        worker = await self._acquire_worker()
        loop = self._require_worker_loop(worker)
        request = _StartRequest()
        future = asyncio.run_coroutine_threadsafe(
            self._spawn_assignment(
                assignment_id,
                execution_id,
                driver,
                spec,
                config,
                reporter,
                worker,
                slot_cost,
                request,
            ),
            loop,
        )
        try:
            return await await_threadsafe(future, timeout=self.bridge_timeout)
        except (TimeoutError, asyncio.CancelledError):
            with self._lock:
                assignment_started = assignment_id in self._assignments
                if not assignment_started:
                    request.cancelled = True
            if not assignment_started:
                self._release_worker_after_start_failure(worker, suspect=True)
            raise
        except Exception:
            self._release_worker_after_start_failure(worker)
            raise

    async def stop_assignment(self, assignment_id: str, timeout: float = 10.0) -> None:
        assignment = self._get_assignment(assignment_id)
        if assignment is None:
            return
        loop = self._require_worker_loop(assignment.worker)
        future = asyncio.run_coroutine_threadsafe(self._stop_assignment(assignment, timeout), loop)
        await await_threadsafe(future, timeout=max(0.0, timeout) + 1.0)
        if assignment.worker.state in {'closed', 'closing'}:
            await assignment.worker.stop(join_timeout=self.thread_join_timeout)

    async def submit_command(self, assignment_id: str, command_type: str, payload: dict[str, Any]) -> None:
        assignment = self._require_assignment(assignment_id)
        loop = self._require_worker_loop(assignment.worker)
        future = asyncio.run_coroutine_threadsafe(
            assignment.driver.submit_command(assignment.handle, command_type, payload),
            loop,
        )
        await await_threadsafe(future, timeout=self.bridge_timeout)

    async def collect_events_after(
        self, assignment_id: str, cursor: int, *, timeout: float = 15.0
    ) -> list[tuple[int, bytes]]:
        return await self._call_sse_driver(assignment_id, cursor, timeout=timeout)

    async def get_runtime_base_url(self, assignment_id: str, runtime_name: str) -> str:
        return await self._call_runtime_driver(assignment_id, runtime_name)

    async def get_assignment_info(self, assignment_id: str) -> dict[str, Any]:
        return await self._call_info_driver(assignment_id)

    def _require_started(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError('ThreadExecutor is not running')

    def _get_assignment(self, assignment_id: str) -> _LocalExecutionInstance | None:
        with self._lock:
            return self._assignments.get(assignment_id)

    def _require_assignment(self, assignment_id: str) -> _LocalExecutionInstance:
        assignment = self._get_assignment(assignment_id)
        if assignment is None:
            raise KeyError(assignment_id)
        return assignment

    def _require_worker_loop(self, worker: _ThreadWorker) -> asyncio.AbstractEventLoop:
        loop = worker.loop
        if loop is None or worker.state == 'closed':
            raise RuntimeError('thread worker is not running')
        return loop

    def _running_slot_cost_locked(self) -> int:
        return sum(item.slot_cost for item in self._assignments.values())

    def _effective_capacity_locked(self) -> int:
        return max(0, self.max_threads - self._stuck_slot_cost)

    def _available_slots_locked(self) -> int:
        return max(0, self._effective_capacity_locked() - self._running_slot_cost_locked())

    async def _probe_worker(
        self,
        worker: _ThreadWorker,
        *,
        timeout: float,
        hang_timeout: float,
        now: float,
    ) -> QuarantinedAssignment | None:
        loop = worker.loop
        responsive = False
        if loop is not None and loop.is_running():

            async def ping() -> None:
                await asyncio.sleep(0)

            future = asyncio.run_coroutine_threadsafe(ping(), loop)
            try:
                await await_threadsafe(future, timeout=timeout)
                responsive = True
            except (RuntimeError, TimeoutError):
                pass

        with self._lock:
            if worker.state == 'closed' or worker.health_state == 'quarantined':
                return None
            if responsive:
                worker.health_state = 'healthy'
                worker.unresponsive_since = None
                if worker.state == 'idle' and worker not in self._idle_workers:
                    self._idle_workers.append(worker)
                return None
            if worker.unresponsive_since is None:
                worker.unresponsive_since = now
                worker.health_state = 'suspect'
                if worker in self._idle_workers:
                    self._idle_workers.remove(worker)
                return None
            if now - worker.unresponsive_since < hang_timeout:
                return None
            return self._quarantine_worker_locked(worker)

    def _quarantine_worker_locked(self, worker: _ThreadWorker) -> QuarantinedAssignment:
        worker.health_state = 'quarantined'
        if worker in self._idle_workers:
            self._idle_workers.remove(worker)
        assignment_id = worker.current_assignment_id
        assignment = self._assignments.pop(assignment_id, None) if assignment_id is not None else None
        slot_cost = assignment.slot_cost if assignment is not None else 1
        self._stuck_slot_cost += slot_cost
        event = QuarantinedAssignment(
            assignment_id=assignment.assignment_id if assignment is not None else None,
            execution_id=assignment.execution_id if assignment is not None else None,
            worker_id=worker.worker_id,
            slot_cost=slot_cost,
        )
        self._quarantined_events[worker.worker_id] = event
        return event

    async def _acquire_worker(self) -> _ThreadWorker:
        with self._lock:
            if self.strategy == 'pooled':
                while self._idle_workers:
                    worker = self._idle_workers.pop(0)
                    if worker.state == 'idle' and worker.health_state == 'healthy':
                        worker.state = 'running'
                        return worker

            self._reap_retiring_workers_locked()
            for worker_id, worker in list(self._workers.items()):
                if worker.state == 'closed':
                    self._workers.pop(worker_id, None)
            active_workers = [worker for worker in self._workers.values() if worker.state != 'closed']
            if len(active_workers) >= self.max_threads:
                raise RuntimeError('thread executor has no available workers')

            worker = _ThreadWorker(worker_id=str(self._next_worker_id), aiomonitor_config=self.aiomonitor_config)
            self._next_worker_id += 1
            self._workers[worker.worker_id] = worker

        await worker.start()
        with self._lock:
            if self._closed:
                await worker.stop(join_timeout=self.thread_join_timeout)
                raise RuntimeError('ThreadExecutor is not running')
            worker.state = 'running'
        return worker

    def _retire_worker_locked(self, worker: _ThreadWorker) -> None:
        """Free the worker's ``max_threads`` slot as soon as it is asked to shut down.

        A worker retired after its assignment completed no longer holds an assignment, so
        capacity accounting already reports the slot as free. Keeping it in ``_workers``
        until its thread happens to die makes ``_acquire_worker`` reject the next
        assignment with ``no available workers`` even though ``capacity_snapshot`` just
        advertised a free slot -- the pooled ``max_tasks_per_thread`` rollover hit this on
        every recycle. Workers abandoned by a failed or hung start are deliberately *not*
        retired here: their loop may be wedged, so they keep reserving the slot until a
        health probe rules on them.
        """
        self._workers.pop(worker.worker_id, None)
        if worker in self._idle_workers:
            self._idle_workers.remove(worker)
        if worker not in self._retiring_workers:
            self._retiring_workers.append(worker)

    def _reap_retiring_workers_locked(self) -> None:
        for worker in list(self._retiring_workers):
            if worker.thread is None or not worker.thread.is_alive():
                worker.state = 'closed'
                self._retiring_workers.remove(worker)

    def _release_worker_after_start_failure(self, worker: _ThreadWorker, *, suspect: bool = False) -> None:
        with self._lock:
            if suspect:
                worker.health_state = 'suspect'
                worker.unresponsive_since = None
            if self.strategy == 'pooled' and not self._closed:
                worker.state = 'idle'
                if worker.health_state == 'healthy' and worker not in self._idle_workers:
                    self._idle_workers.append(worker)
                return
            worker.state = 'closing'
        loop = worker.loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)

    async def _spawn_assignment(
        self,
        assignment_id: str,
        execution_id: str,
        driver: ExecutionDriver,
        spec: dict[str, Any],
        config: dict[str, Any],
        reporter: ExecutionReporter,
        worker: _ThreadWorker,
        slot_cost: int,
        request: _StartRequest,
    ) -> ExecutionHandle:
        handle = ExecutionHandle(execution_id=execution_id, assignment_id=assignment_id)

        async def run_driver() -> None:
            await run_driver_with_failure_reporting(driver, handle, spec, config, reporter)

        with self._lock:
            if request.cancelled:
                raise TimeoutError('assignment start was cancelled after bridge timeout')
            if assignment_id in self._assignments:
                raise ValueError(f'assignment {assignment_id} is already running')
            task = asyncio.create_task(run_driver(), name=f'nexus-thread-execution-{assignment_id}')
            assignment = _LocalExecutionInstance(
                assignment_id=assignment_id,
                execution_id=execution_id,
                driver=driver,
                handle=handle,
                task=task,
                worker=worker,
                slot_cost=slot_cost,
            )
            worker.current_assignment_id = assignment_id
            self._assignments[assignment_id] = assignment

        def on_done(done_task: asyncio.Task[None]) -> None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                done_task.result()
            if assignment.stop_waiter_active:
                return
            self._finish_assignment(assignment)

        task.add_done_callback(on_done)
        return handle

    async def _stop_assignment(self, assignment: _LocalExecutionInstance, timeout: float) -> None:
        assignment.worker.state = 'stopping'
        assignment.stop_waiter_active = True
        try:
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
        finally:
            assignment.stop_waiter_active = False
            if assignment.task.done():
                self._finish_assignment(assignment)

    def _finish_assignment(self, assignment: _LocalExecutionInstance) -> None:
        should_stop_worker = False
        with self._lock:
            removed = self._assignments.pop(assignment.assignment_id, None)
            worker = assignment.worker
            was_current = worker.current_assignment_id == assignment.assignment_id
            if worker.health_state == 'quarantined':
                if was_current:
                    worker.current_assignment_id = None
                return
            if was_current:
                worker.current_assignment_id = None
            if removed is None and not was_current:
                return
            worker.tasks_completed += 1

            if self._closed:
                should_stop_worker = True
            elif (
                self.strategy == 'pooled'
                and worker.health_state == 'healthy'
                and worker.tasks_completed < self.max_tasks_per_thread
            ):
                worker.state = 'idle'
                if worker not in self._idle_workers:
                    self._idle_workers.append(worker)
            elif self.strategy == 'pooled' and worker.health_state == 'suspect':
                worker.state = 'idle'
            else:
                should_stop_worker = True
                worker.state = 'closing'
                self._retire_worker_locked(worker)

        if should_stop_worker:
            loop = worker.loop
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(loop.stop)

    async def _call_sse_driver(self, assignment_id: str, cursor: int, *, timeout: float) -> list[tuple[int, bytes]]:
        assignment = self._require_assignment(assignment_id)
        loop = self._require_worker_loop(assignment.worker)

        async def call() -> list[tuple[int, bytes]]:
            if not isinstance(assignment.driver, SseEventStreamingDriver):
                raise ValueError('events_not_supported')
            return await assignment.driver.collect_sse_events_after(cursor, timeout=timeout)

        future = asyncio.run_coroutine_threadsafe(call(), loop)
        return await await_threadsafe(future, timeout=max(0.0, timeout) + 2.0)

    async def _call_runtime_driver(self, assignment_id: str, runtime_name: str) -> str:
        assignment = self._require_assignment(assignment_id)
        loop = self._require_worker_loop(assignment.worker)

        async def call() -> str:
            if not isinstance(assignment.driver, RuntimeEndpointDriver):
                raise ValueError('runtime_proxy_not_supported')
            return await assignment.driver.get_runtime_base_url(runtime_name)

        future = asyncio.run_coroutine_threadsafe(call(), loop)
        return await await_threadsafe(future, timeout=self.bridge_timeout)

    async def _call_info_driver(self, assignment_id: str) -> dict[str, Any]:
        assignment = self._require_assignment(assignment_id)
        loop = self._require_worker_loop(assignment.worker)

        async def call() -> dict[str, Any]:
            if not isinstance(assignment.driver, ExecutionInfoDriver):
                raise ValueError('info_not_supported')
            return await assignment.driver.get_execution_info(assignment.handle)

        future = asyncio.run_coroutine_threadsafe(call(), loop)
        return await await_threadsafe(future, timeout=self.bridge_timeout)
