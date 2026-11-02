from __future__ import annotations

import asyncio
import contextvars
import threading

import pytest

from nexus.distributed.execution.executors.thread_executor import ThreadExecutor


_TRACE_CONTEXT = contextvars.ContextVar[str]('thread_executor_trace_context', default='missing')


class ThreadProbeDriver:
    spec_type = 'probe'

    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancelled = threading.Event()
        self.stopped = threading.Event()
        self.commands: list[tuple[str, dict]] = []
        self.thread_ids: list[int] = []
        self.thread_names: list[str] = []

    async def start(self, handle, spec, config, reporter) -> None:
        self.thread_ids.append(threading.get_ident())
        self.thread_names.append(threading.current_thread().name)
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    async def submit_command(self, handle, command_type, payload) -> None:
        self.commands.append((command_type, payload))

    async def stop(self, handle) -> None:
        self.stopped.set()

    async def collect_sse_events_after(self, cursor: int, *, timeout: float):
        return [(cursor + 1, b'event: probe\n\n')]

    async def get_runtime_base_url(self, runtime_name: str) -> str:
        return f'http://runtime/{runtime_name}'

    async def get_execution_info(self, handle):
        return {
            'assignment_id': handle.assignment_id,
            'execution_id': handle.execution_id,
            'driver_thread_id': threading.get_ident(),
        }


class ThreadContextProbeDriver(ThreadProbeDriver):
    def __init__(self) -> None:
        super().__init__()
        self.context_value = 'missing'

    async def start(self, handle, spec, config, reporter) -> None:  # noqa: ARG002
        self.context_value = _TRACE_CONTEXT.get()
        self.started.set()


async def test_thread_executor_driver_inherits_contextvars_from_start_assignment() -> None:
    driver = ThreadContextProbeDriver()
    executor = ThreadExecutor(strategy='per_execution', max_threads=1)
    await executor.start()
    try:
        token = _TRACE_CONTEXT.set('assignment-trace-context')
        try:
            await executor.start_assignment(
                assignment_id='assignment-trace-context',
                execution_id='execution-trace-context',
                driver=driver,
                spec={},
                config={},
                reporter=object(),
            )
        finally:
            _TRACE_CONTEXT.reset(token)

        assert await asyncio.to_thread(driver.started.wait, 2)
        assert driver.context_value == 'assignment-trace-context'
    finally:
        await executor.close()


class ThreadGracefulStopDriver(ThreadProbeDriver):
    def __init__(self) -> None:
        super().__init__()
        self.finished = threading.Event()
        self._finish: asyncio.Event | None = None

    async def start(self, handle, spec, config, reporter) -> None:
        self.thread_ids.append(threading.get_ident())
        self.thread_names.append(threading.current_thread().name)
        self._finish = asyncio.Event()
        self.started.set()
        await self._finish.wait()
        self.finished.set()

    async def stop(self, handle) -> None:
        self.stopped.set()
        assert self._finish is not None
        self._finish.set()


class ThreadBlockingStopDriver(ThreadProbeDriver):
    async def stop(self, handle) -> None:
        self.stopped.set()
        await asyncio.Event().wait()


async def test_thread_executor_per_execution_runs_driver_on_dedicated_thread_and_cancels_task():
    driver = ThreadProbeDriver()
    executor = ThreadExecutor(strategy='per_execution', max_threads=1)
    await executor.start()
    try:
        await executor.start_assignment(
            assignment_id='assignment-1',
            execution_id='execution-1',
            driver=driver,
            spec={},
            config={},
            reporter=object(),
        )

        assert await asyncio.to_thread(driver.started.wait, 2)
        assert driver.thread_ids[0] != threading.get_ident()
        capacity = await executor.capacity_snapshot()
        assert capacity.running_slot_cost == 1
        assert capacity.available_slots == 0

        await executor.stop_assignment('assignment-1', timeout=0.01)

        assert await asyncio.to_thread(driver.stopped.wait, 2)
        assert await asyncio.to_thread(driver.cancelled.wait, 2)
        assert not await executor.has_assignment('assignment-1')
        capacity = await executor.capacity_snapshot()
        assert capacity.available_slots == 1
    finally:
        await executor.close()


async def test_thread_executor_waits_for_driver_task_after_stop():
    driver = ThreadGracefulStopDriver()
    executor = ThreadExecutor(strategy='per_execution', max_threads=1)
    await executor.start()
    try:
        await executor.start_assignment(
            assignment_id='assignment-1',
            execution_id='execution-1',
            driver=driver,
            spec={},
            config={},
            reporter=object(),
        )

        assert await asyncio.to_thread(driver.started.wait, 2)
        await executor.stop_assignment('assignment-1', timeout=1)

        assert driver.stopped.is_set()
        assert driver.finished.is_set()
        assert not driver.cancelled.is_set()
        assert not await executor.has_assignment('assignment-1')
    finally:
        await executor.close()


async def test_thread_executor_uses_one_deadline_for_stop_and_task_cancellation():
    driver = ThreadBlockingStopDriver()
    executor = ThreadExecutor(strategy='per_execution', max_threads=1)
    await executor.start()
    try:
        await executor.start_assignment(
            assignment_id='assignment-deadline',
            execution_id='execution-deadline',
            driver=driver,
            spec={},
            config={},
            reporter=object(),
        )
        assert await asyncio.to_thread(driver.started.wait, 1)
        started_at = asyncio.get_running_loop().time()

        await executor.stop_assignment('assignment-deadline', timeout=0.02)

        assert asyncio.get_running_loop().time() - started_at < 0.1
        assert await asyncio.to_thread(driver.cancelled.wait, 1)
    finally:
        await executor.close()


async def test_thread_executor_pooled_reuses_worker_until_max_tasks_per_thread():
    executor = ThreadExecutor(strategy='pooled', max_threads=1, max_tasks_per_thread=2)
    await executor.start()
    try:
        first = ThreadProbeDriver()
        await executor.start_assignment(
            assignment_id='assignment-1',
            execution_id='execution-1',
            driver=first,
            spec={},
            config={},
            reporter=object(),
        )
        assert await asyncio.to_thread(first.started.wait, 2)
        await executor.stop_assignment('assignment-1', timeout=0.01)
        assert await asyncio.to_thread(first.cancelled.wait, 2)

        second = ThreadProbeDriver()
        await executor.start_assignment(
            assignment_id='assignment-2',
            execution_id='execution-2',
            driver=second,
            spec={},
            config={},
            reporter=object(),
        )
        assert await asyncio.to_thread(second.started.wait, 2)
        await executor.stop_assignment('assignment-2', timeout=0.01)
        assert await asyncio.to_thread(second.cancelled.wait, 2)

        third = ThreadProbeDriver()
        await executor.start_assignment(
            assignment_id='assignment-3',
            execution_id='execution-3',
            driver=third,
            spec={},
            config={},
            reporter=object(),
        )
        assert await asyncio.to_thread(third.started.wait, 2)

        assert first.thread_names == second.thread_names
        assert third.thread_names[0] != first.thread_names[0]
    finally:
        await executor.close()


async def test_thread_executor_dispatches_command_and_live_helpers_to_owner_worker():
    driver = ThreadProbeDriver()
    executor = ThreadExecutor(strategy='per_execution', max_threads=1)
    await executor.start()
    try:
        await executor.start_assignment(
            assignment_id='assignment-1',
            execution_id='execution-1',
            driver=driver,
            spec={},
            config={},
            reporter=object(),
        )
        assert await asyncio.to_thread(driver.started.wait, 2)

        await executor.submit_command('assignment-1', 'input.submit', {'text': 'Alpha'})
        assert driver.commands == [('input.submit', {'text': 'Alpha'})]
        assert await executor.collect_events_after('assignment-1', 4, timeout=0.01) == [(5, b'event: probe\n\n')]
        assert await executor.get_runtime_base_url('assignment-1', 'main') == 'http://runtime/main'
        assert await executor.get_assignment_info('assignment-1') == {
            'assignment_id': 'assignment-1',
            'execution_id': 'execution-1',
            'driver_thread_id': driver.thread_ids[0],
        }

        with pytest.raises(KeyError):
            await executor.collect_events_after('missing', 0, timeout=0.01)
        with pytest.raises(KeyError):
            await executor.get_assignment_info('missing')
    finally:
        await executor.close()


async def test_thread_executor_bridge_times_out_when_worker_loop_is_blocked() -> None:
    release = threading.Event()
    entered = threading.Event()
    driver = ThreadProbeDriver()
    executor = ThreadExecutor(strategy='per_execution', max_threads=1, bridge_timeout=0.02)
    await executor.start()
    try:
        await executor.start_assignment(
            assignment_id='assignment-blocked-loop',
            execution_id='execution-blocked-loop',
            driver=driver,
            spec={},
            config={},
            reporter=object(),
        )
        assert await asyncio.to_thread(driver.started.wait, 1)
        worker = next(iter(executor._workers.values()))
        loop = worker.loop
        assert loop is not None

        def block_loop() -> None:
            entered.set()
            release.wait()

        loop.call_soon_threadsafe(block_loop)
        assert await asyncio.to_thread(entered.wait, 1)

        with pytest.raises(TimeoutError):
            await executor.get_assignment_info('assignment-blocked-loop')
    finally:
        release.set()
        await executor.close()


async def test_thread_executor_rejects_late_start_after_bridge_timeout(monkeypatch) -> None:
    release = threading.Event()
    entered = threading.Event()
    driver = ThreadProbeDriver()
    executor = ThreadExecutor(strategy='pooled', max_threads=1, bridge_timeout=0.02)
    await executor.start()
    original_spawn = executor._spawn_assignment

    async def delayed_spawn(*args, **kwargs):
        entered.set()
        try:
            await asyncio.to_thread(release.wait)
        except asyncio.CancelledError:
            await asyncio.to_thread(release.wait)
        try:
            return await original_spawn(*args, **kwargs)
        except TimeoutError:
            return None

    monkeypatch.setattr(executor, '_spawn_assignment', delayed_spawn)
    try:
        start = asyncio.create_task(
            executor.start_assignment(
                assignment_id='assignment-late-start',
                execution_id='execution-late-start',
                driver=driver,
                spec={},
                config={},
                reporter=object(),
            )
        )
        assert await asyncio.to_thread(entered.wait, 1)

        with pytest.raises(TimeoutError):
            await start

        release.set()
        await asyncio.sleep(0.05)
        assert not await executor.has_assignment('assignment-late-start')
        assert not driver.started.is_set()
    finally:
        release.set()
        await executor.close()


async def test_thread_executor_counts_live_closing_worker_against_per_execution_limit(monkeypatch) -> None:
    release = threading.Event()
    entered = threading.Event()
    executor = ThreadExecutor(strategy='per_execution', max_threads=1, bridge_timeout=0.02)
    await executor.start()
    original_spawn = executor._spawn_assignment

    async def blocked_spawn(*args, **kwargs):
        entered.set()
        release.wait()
        return await original_spawn(*args, **kwargs)

    monkeypatch.setattr(executor, '_spawn_assignment', blocked_spawn)
    try:
        first = asyncio.create_task(
            executor.start_assignment(
                assignment_id='assignment-hung-start-1',
                execution_id='execution-hung-start-1',
                driver=ThreadProbeDriver(),
                spec={},
                config={},
                reporter=object(),
            )
        )
        assert await asyncio.to_thread(entered.wait, 1)
        with pytest.raises(TimeoutError):
            await first

        with pytest.raises(RuntimeError, match='no available workers'):
            await executor.start_assignment(
                assignment_id='assignment-hung-start-2',
                execution_id='execution-hung-start-2',
                driver=ThreadProbeDriver(),
                spec={},
                config={},
                reporter=object(),
            )
        assert len(executor._workers) == 1
    finally:
        release.set()
        await asyncio.sleep(0.05)
        await executor.close()


async def test_thread_executor_withholds_pooled_worker_after_start_timeout_until_probe(monkeypatch) -> None:
    release = threading.Event()
    entered = threading.Event()
    executor = ThreadExecutor(strategy='pooled', max_threads=1, bridge_timeout=0.02)
    await executor.start()
    original_spawn = executor._spawn_assignment

    async def blocked_spawn(*args, **kwargs):
        entered.set()
        release.wait()
        return await original_spawn(*args, **kwargs)

    monkeypatch.setattr(executor, '_spawn_assignment', blocked_spawn)
    try:
        first = asyncio.create_task(
            executor.start_assignment(
                assignment_id='assignment-pooled-hung-start-1',
                execution_id='execution-pooled-hung-start-1',
                driver=ThreadProbeDriver(),
                spec={},
                config={},
                reporter=object(),
            )
        )
        assert await asyncio.to_thread(entered.wait, 1)
        with pytest.raises(TimeoutError):
            await first

        with pytest.raises(RuntimeError, match='no available workers'):
            await executor.start_assignment(
                assignment_id='assignment-pooled-hung-start-2',
                execution_id='execution-pooled-hung-start-2',
                driver=ThreadProbeDriver(),
                spec={},
                config={},
                reporter=object(),
            )
    finally:
        release.set()
        await asyncio.sleep(0.05)
        await executor.close()


async def test_thread_executor_reuses_assignment_id_after_cancelled_start_never_ran() -> None:
    release = threading.Event()
    entered = threading.Event()
    executor = ThreadExecutor(strategy='pooled', max_threads=1, bridge_timeout=0.02)
    await executor.start()
    try:
        seed = ThreadProbeDriver()
        await executor.start_assignment(
            assignment_id='assignment-seed-worker',
            execution_id='execution-seed-worker',
            driver=seed,
            spec={},
            config={},
            reporter=object(),
        )
        assert await asyncio.to_thread(seed.started.wait, 1)
        await executor.stop_assignment('assignment-seed-worker', timeout=0.1)

        worker = next(iter(executor._workers.values()))
        loop = worker.loop
        assert loop is not None

        def block_loop() -> None:
            entered.set()
            release.wait()

        loop.call_soon_threadsafe(block_loop)
        assert await asyncio.to_thread(entered.wait, 1)
        with pytest.raises(TimeoutError):
            await executor.start_assignment(
                assignment_id='assignment-retry-after-timeout',
                execution_id='execution-retry-after-timeout',
                driver=ThreadProbeDriver(),
                spec={},
                config={},
                reporter=object(),
            )

        release.set()
        recovered = await executor.probe_health(timeout=1.0, hang_timeout=3.0, now=11.0)
        assert recovered.unresponsive_workers == 0

        retry = ThreadProbeDriver()
        await executor.start_assignment(
            assignment_id='assignment-retry-after-timeout',
            execution_id='execution-retry-after-timeout',
            driver=retry,
            spec={},
            config={},
            reporter=object(),
        )
        assert await asyncio.to_thread(retry.started.wait, 1)
    finally:
        release.set()
        await executor.close()


async def test_thread_executor_probes_workers_concurrently(monkeypatch) -> None:
    executor = ThreadExecutor(strategy='per_execution', max_threads=2)
    await executor.start()
    drivers = [ThreadProbeDriver(), ThreadProbeDriver()]
    try:
        for index, driver in enumerate(drivers):
            await executor.start_assignment(
                assignment_id=f'assignment-probe-{index}',
                execution_id=f'execution-probe-{index}',
                driver=driver,
                spec={},
                config={},
                reporter=object(),
            )
            assert await asyncio.to_thread(driver.started.wait, 1)

        both_started = asyncio.Event()
        call_count = 0

        async def synchronized_probe(worker, *, timeout, hang_timeout, now):  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                both_started.set()
            await both_started.wait()

        monkeypatch.setattr(executor, '_probe_worker', synchronized_probe)

        await asyncio.wait_for(executor.probe_health(timeout=1.0, hang_timeout=3.0, now=10.0), timeout=0.2)
        assert call_count == 2
    finally:
        await executor.close()


async def test_thread_executor_health_probe_recovers_suspect_worker_before_threshold() -> None:
    release = threading.Event()
    entered = threading.Event()
    driver = ThreadProbeDriver()
    executor = ThreadExecutor(strategy='per_execution', max_threads=1)
    await executor.start()
    try:
        await executor.start_assignment(
            assignment_id='assignment-recovers',
            execution_id='execution-recovers',
            driver=driver,
            spec={},
            config={},
            reporter=object(),
        )
        assert await asyncio.to_thread(driver.started.wait, 1)
        assignment = executor._get_assignment('assignment-recovers')
        assert assignment is not None
        loop = assignment.worker.loop
        assert loop is not None

        def block_loop() -> None:
            entered.set()
            release.wait()

        loop.call_soon_threadsafe(block_loop)
        assert await asyncio.to_thread(entered.wait, 1)

        suspect = await executor.probe_health(timeout=0.02, hang_timeout=3.0, now=10.0)
        assert suspect.unresponsive_workers == 1
        assert suspect.quarantined_assignments == ()

        release.set()
        recovered = await executor.probe_health(timeout=1.0, hang_timeout=3.0, now=11.0)
        assert recovered.unresponsive_workers == 0
        assert await executor.has_assignment('assignment-recovers')
    finally:
        release.set()
        await executor.close()


async def test_thread_executor_withholds_completed_suspect_pooled_worker_until_probe() -> None:
    release = threading.Event()
    entered = threading.Event()
    driver = ThreadGracefulStopDriver()
    executor = ThreadExecutor(strategy='pooled', max_threads=1)
    await executor.start()
    try:
        await executor.start_assignment(
            assignment_id='assignment-suspect-completes',
            execution_id='execution-suspect-completes',
            driver=driver,
            spec={},
            config={},
            reporter=object(),
        )
        assert await asyncio.to_thread(driver.started.wait, 1)
        assignment = executor._get_assignment('assignment-suspect-completes')
        assert assignment is not None
        loop = assignment.worker.loop
        assert loop is not None

        def block_loop() -> None:
            entered.set()
            release.wait()

        loop.call_soon_threadsafe(block_loop)
        assert await asyncio.to_thread(entered.wait, 1)
        suspect = await executor.probe_health(timeout=0.02, hang_timeout=3.0, now=10.0)
        assert suspect.unresponsive_workers == 1

        assert driver._finish is not None
        loop.call_soon_threadsafe(driver._finish.set)
        release.set()
        assert await asyncio.to_thread(driver.finished.wait, 1)
        for _ in range(100):
            if not await executor.has_assignment('assignment-suspect-completes'):
                break
            await asyncio.sleep(0.01)
        assert assignment.worker not in executor._idle_workers

        with pytest.raises(RuntimeError, match='no available workers'):
            await executor.start_assignment(
                assignment_id='assignment-before-recovery-probe',
                execution_id='execution-before-recovery-probe',
                driver=ThreadProbeDriver(),
                spec={},
                config={},
                reporter=object(),
            )
    finally:
        release.set()
        await executor.close()


async def test_thread_executor_blocked_worker_does_not_delay_healthy_worker() -> None:
    release = threading.Event()
    entered = threading.Event()
    blocked_driver = ThreadProbeDriver()
    healthy_driver = ThreadProbeDriver()
    executor = ThreadExecutor(strategy='per_execution', max_threads=2)
    await executor.start()
    try:
        for assignment_id, execution_id, driver in (
            ('assignment-blocked', 'execution-blocked', blocked_driver),
            ('assignment-healthy', 'execution-healthy', healthy_driver),
        ):
            await executor.start_assignment(
                assignment_id=assignment_id,
                execution_id=execution_id,
                driver=driver,
                spec={},
                config={},
                reporter=object(),
            )
            assert await asyncio.to_thread(driver.started.wait, 1)

        blocked = executor._get_assignment('assignment-blocked')
        assert blocked is not None
        loop = blocked.worker.loop
        assert loop is not None

        def block_loop() -> None:
            entered.set()
            release.wait()

        loop.call_soon_threadsafe(block_loop)
        assert await asyncio.to_thread(entered.wait, 1)

        await asyncio.wait_for(
            executor.submit_command('assignment-healthy', 'input.submit', {'text': 'still-alive'}),
            timeout=0.2,
        )
        health = await executor.probe_health(timeout=0.02, hang_timeout=3.0, now=10.0)

        assert healthy_driver.commands == [('input.submit', {'text': 'still-alive'})]
        assert health.responsive is True
        assert health.unresponsive_workers == 1
    finally:
        release.set()
        await executor.close()


async def test_thread_executor_quarantines_worker_after_continuous_hang() -> None:
    release = threading.Event()
    entered = threading.Event()
    driver = ThreadProbeDriver()
    executor = ThreadExecutor(strategy='per_execution', max_threads=1)
    await executor.start()
    try:
        await executor.start_assignment(
            assignment_id='assignment-quarantined',
            execution_id='execution-quarantined',
            driver=driver,
            spec={},
            config={},
            reporter=object(),
            slot_cost=1,
        )
        assert await asyncio.to_thread(driver.started.wait, 1)
        assignment = executor._get_assignment('assignment-quarantined')
        assert assignment is not None
        loop = assignment.worker.loop
        assert loop is not None

        def block_loop() -> None:
            entered.set()
            release.wait()

        loop.call_soon_threadsafe(block_loop)
        assert await asyncio.to_thread(entered.wait, 1)

        first = await executor.probe_health(timeout=0.02, hang_timeout=3.0, now=10.0)
        assert first.effective_capacity == 1
        assert first.recycle_requested is False
        assert first.quarantined_assignments == ()

        quarantined = await executor.probe_health(timeout=0.02, hang_timeout=3.0, now=13.0)
        assert quarantined.effective_capacity == 0
        assert quarantined.recycle_requested is True
        assert [
            (item.assignment_id, item.execution_id, item.worker_id, item.slot_cost)
            for item in quarantined.quarantined_assignments
        ] == [('assignment-quarantined', 'execution-quarantined', assignment.worker.worker_id, 1)]
        assert not await executor.has_assignment('assignment-quarantined')
        capacity = await executor.capacity_snapshot()
        assert capacity.capacity_total == 0
        assert capacity.running_slot_cost == 0
        assert capacity.available_slots == 0
    finally:
        release.set()
        await executor.close()


async def test_thread_executor_reports_assignment_when_worker_closes_unexpectedly() -> None:
    driver = ThreadProbeDriver()
    executor = ThreadExecutor(strategy='per_execution', max_threads=1)
    await executor.start()
    worker = None
    try:
        await executor.start_assignment(
            assignment_id='assignment-worker-closed',
            execution_id='execution-worker-closed',
            driver=driver,
            spec={},
            config={},
            reporter=object(),
        )
        assert await asyncio.to_thread(driver.started.wait, 1)
        assignment = executor._get_assignment('assignment-worker-closed')
        assert assignment is not None
        worker = assignment.worker
        worker.state = 'closed'

        health = await executor.probe_health(timeout=0.02, hang_timeout=3.0, now=10.0)

        assert health.effective_capacity == 0
        assert health.recycle_requested is True
        assert [item.assignment_id for item in health.quarantined_assignments] == ['assignment-worker-closed']
        assert not await executor.has_assignment('assignment-worker-closed')
    finally:
        if worker is not None:
            await worker.stop()
        await executor.close()
