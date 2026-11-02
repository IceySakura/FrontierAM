from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import threading

import pytest

from nexus.distributed.execution.executors.asyncio_executor import AsyncioExecutor
from nexus.distributed.observability import logging as distributed_logs
from nexus.distributed.observability.context import distributed_execution_context


_TRACE_CONTEXT = contextvars.ContextVar[str]('asyncio_executor_trace_context', default='missing')


class BlockingDriver:
    spec_type = 'blocking'

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.stopped = asyncio.Event()

    async def start(self, handle, spec, config, callbacks) -> None:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    async def submit_command(self, handle, command_type, payload) -> None:
        raise AssertionError('not used')

    async def stop(self, handle) -> None:
        self.stopped.set()


class GracefulStopDriver(BlockingDriver):
    def __init__(self) -> None:
        super().__init__()
        self.finished = asyncio.Event()

    async def start(self, handle, spec, config, callbacks) -> None:
        self.started.set()
        await self.stopped.wait()
        self.finished.set()


class BlockingStopDriver(BlockingDriver):
    async def stop(self, handle) -> None:
        self.stopped.set()
        await asyncio.Event().wait()


class LoggingDriver:
    spec_type = 'logging'

    def __init__(self) -> None:
        self.logged = threading.Event()

    async def start(self, handle, spec, config, callbacks) -> None:  # noqa: ARG002
        logging.getLogger('nexus.test.executor').info('driver started')
        self.logged.set()

    async def submit_command(self, handle, command_type, payload) -> None:
        raise AssertionError('not used')

    async def stop(self, handle) -> None:
        return None


class ContextProbeDriver(LoggingDriver):
    def __init__(self) -> None:
        super().__init__()
        self.context_value = 'missing'

    async def start(self, handle, spec, config, callbacks) -> None:  # noqa: ARG002
        self.context_value = _TRACE_CONTEXT.get()
        self.logged.set()


async def test_asyncio_executor_driver_inherits_contextvars_from_start_assignment() -> None:
    driver = ContextProbeDriver()
    executor = AsyncioExecutor()
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

        assert await asyncio.to_thread(driver.logged.wait, 2)
        assert driver.context_value == 'assignment-trace-context'
    finally:
        await executor.close()


async def test_asyncio_executor_driver_logs_inherit_distributed_execution_context(tmp_path):
    distributed_logs.setup_distributed_logging(
        log_dir=str(tmp_path),
        log_level='INFO',
        component='instance-manager',
        identity='im-executor',
    )
    driver = LoggingDriver()
    executor = AsyncioExecutor()
    await executor.start()
    try:
        with distributed_execution_context(execution_id='execution-ctx', assignment_id='assignment-ctx'):
            await executor.start_assignment(
                assignment_id='assignment-ctx',
                execution_id='execution-ctx',
                driver=driver,
                spec={},
                config={},
                reporter=object(),
            )

        assert await asyncio.to_thread(driver.logged.wait, 2)
        for handler in logging.getLogger().handlers:
            handler.flush()
        payloads = [
            json.loads(line)
            for line in (tmp_path / 'instance-manager-im-executor.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        payload = next(item for item in payloads if item['message'] == 'driver started')
        assert payload['execution_id'] == 'execution-ctx'
        assert payload['assignment_id'] == 'assignment-ctx'
    finally:
        await executor.close()


async def test_asyncio_executor_runs_driver_on_child_loop_and_cancels_task():
    driver = BlockingDriver()
    callbacks = object()
    executor = AsyncioExecutor()
    await executor.start()
    try:
        await executor.start_assignment(
            assignment_id='assignment-1',
            execution_id='execution-1',
            driver=driver,
            spec={},
            config={},
            reporter=callbacks,
        )

        await asyncio.wait_for(driver.started.wait(), timeout=2)
        await executor.stop_assignment('assignment-1', timeout=0.01)

        await asyncio.wait_for(driver.stopped.wait(), timeout=2)
        await asyncio.wait_for(driver.cancelled.wait(), timeout=2)
        assert not await executor.has_assignment('assignment-1')
    finally:
        await executor.close()


async def test_asyncio_executor_waits_for_driver_task_after_stop():
    driver = GracefulStopDriver()
    executor = AsyncioExecutor()
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

        await asyncio.wait_for(driver.started.wait(), timeout=2)
        await executor.stop_assignment('assignment-1', timeout=1)

        assert driver.stopped.is_set()
        assert driver.finished.is_set()
        assert not driver.cancelled.is_set()
        assert not await executor.has_assignment('assignment-1')
    finally:
        await executor.close()


async def test_asyncio_executor_uses_one_deadline_for_stop_and_task_cancellation():
    driver = BlockingStopDriver()
    executor = AsyncioExecutor()
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
        await asyncio.wait_for(driver.started.wait(), timeout=1)
        started_at = asyncio.get_running_loop().time()

        await executor.stop_assignment('assignment-deadline', timeout=0.02)

        assert asyncio.get_running_loop().time() - started_at < 0.1
        await asyncio.wait_for(driver.cancelled.wait(), timeout=1)
    finally:
        await executor.close()


async def test_asyncio_executor_rejects_duplicate_assignment():
    driver = BlockingDriver()
    executor = AsyncioExecutor()
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
        with pytest.raises(ValueError, match='already running'):
            await executor.start_assignment(
                assignment_id='assignment-1',
                execution_id='execution-1',
                driver=driver,
                spec={},
                config={},
                reporter=object(),
            )
    finally:
        await executor.close()


async def test_asyncio_executor_rejects_non_positive_slot_cost():
    executor = AsyncioExecutor()
    await executor.start()
    try:
        for slot_cost in (0, -1):
            with pytest.raises(ValueError, match='slot_cost must be at least 1'):
                await executor.start_assignment(
                    assignment_id=f'assignment-{slot_cost}',
                    execution_id=f'execution-{slot_cost}',
                    driver=BlockingDriver(),
                    spec={},
                    config={},
                    reporter=object(),
                    slot_cost=slot_cost,
                )
    finally:
        await executor.close()


class CommandDriver(BlockingDriver):
    def __init__(self) -> None:
        super().__init__()
        self.commands: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()

    async def submit_command(self, handle, command_type, payload) -> None:
        await self.commands.put((command_type, payload))


class LoopOwnedAssignments(dict):
    def __init__(self, owner_thread_id: int) -> None:
        super().__init__()
        self.owner_thread_id = owner_thread_id

    def _check_owner(self) -> None:
        if threading.get_ident() != self.owner_thread_id:
            raise AssertionError('assignment registry must be accessed on executor loop thread')

    def __contains__(self, key) -> bool:
        self._check_owner()
        return super().__contains__(key)

    def __iter__(self):
        self._check_owner()
        return super().__iter__()

    def __len__(self) -> int:
        self._check_owner()
        return super().__len__()

    def __setitem__(self, key, value) -> None:
        self._check_owner()
        return super().__setitem__(key, value)

    def clear(self) -> None:
        self._check_owner()
        return super().clear()

    def get(self, key, default=None):
        self._check_owner()
        return super().get(key, default)

    def pop(self, key, default=None):
        self._check_owner()
        return super().pop(key, default)

    def values(self):
        self._check_owner()
        return super().values()


class CommandRuntimeDriver(CommandDriver):
    async def get_runtime_base_url(self, name):
        return f'http://runtime/{name}'

    async def collect_sse_events_after(self, cursor, *, timeout=15.0):
        return [(cursor + 1, b'event: ping\n\n')]


async def test_asyncio_executor_owns_assignment_registry_on_child_loop():
    driver = CommandRuntimeDriver()
    executor = AsyncioExecutor()
    await executor.start()
    assert executor._thread is not None
    assert executor._thread.ident is not None
    executor._assignments = LoopOwnedAssignments(executor._thread.ident)
    try:
        await executor.start_assignment(
            assignment_id='assignment-1',
            execution_id='execution-1',
            driver=driver,
            spec={},
            config={},
            reporter=object(),
        )
        await asyncio.wait_for(driver.started.wait(), timeout=2)

        assert await executor.has_assignment('assignment-1')
        capacity = await executor.capacity_snapshot()
        assert capacity.running_slot_cost == 1
        assert capacity.available_slots == 0
        await executor.submit_command('assignment-1', 'input.submit', {'text': 'Alpha'})
        assert await asyncio.wait_for(driver.commands.get(), timeout=2) == ('input.submit', {'text': 'Alpha'})
        assert await executor.get_runtime_base_url('assignment-1', 'main') == 'http://runtime/main'
        assert await executor.collect_events_after('assignment-1', 0, timeout=0.01) == [(1, b'event: ping\n\n')]
        await executor.stop_assignment('assignment-1', timeout=0.01)
        assert not await executor.has_assignment('assignment-1')
        capacity = await executor.capacity_snapshot()
        assert capacity.running_slot_cost == 0
        assert capacity.available_slots == 1
    finally:
        await executor.close()


async def test_asyncio_executor_dispatches_command_on_child_loop():
    driver = CommandDriver()
    executor = AsyncioExecutor()
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
        await asyncio.wait_for(driver.started.wait(), timeout=2)

        await executor.submit_command('assignment-1', 'input.submit', {'text': 'Alpha'})

        assert await asyncio.wait_for(driver.commands.get(), timeout=2) == ('input.submit', {'text': 'Alpha'})
        with pytest.raises(KeyError):
            await executor.submit_command('missing', 'input.submit', {})
    finally:
        await executor.close()


class HangingStopDriver(BlockingDriver):
    async def stop(self, handle) -> None:
        self.stopped.set()
        await asyncio.Event().wait()


async def test_asyncio_executor_cancels_task_when_driver_stop_hangs():
    driver = HangingStopDriver()
    executor = AsyncioExecutor()
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
        await asyncio.wait_for(driver.started.wait(), timeout=2)

        await executor.stop_assignment('assignment-1', timeout=0.01)

        await asyncio.wait_for(driver.stopped.wait(), timeout=2)
        await asyncio.wait_for(driver.cancelled.wait(), timeout=2)
        assert not await executor.has_assignment('assignment-1')
    finally:
        await executor.close()


async def test_asyncio_executor_rejects_operations_before_start_and_after_close():
    executor = AsyncioExecutor()
    with pytest.raises(RuntimeError, match='not running'):
        await executor.submit_command('missing', 'input.submit', {})

    await executor.start()
    await executor.close()
    with pytest.raises(RuntimeError, match='not running'):
        await executor.start_assignment(
            assignment_id='assignment-closed',
            execution_id='execution-closed',
            driver=BlockingDriver(),
            spec={},
            config={},
            reporter=object(),
        )


async def test_asyncio_executor_close_cleans_running_assignments():
    driver = BlockingDriver()
    executor = AsyncioExecutor()
    await executor.start()
    await executor.start_assignment(
        assignment_id='assignment-1',
        execution_id='execution-1',
        driver=driver,
        spec={},
        config={},
        reporter=object(),
    )
    await asyncio.wait_for(driver.started.wait(), timeout=2)

    await executor.close()

    await asyncio.wait_for(driver.stopped.wait(), timeout=2)
    await asyncio.wait_for(driver.cancelled.wait(), timeout=2)
    assert not await executor.has_assignment('assignment-1')


class FailingStartDriver(BlockingDriver):
    def __init__(self) -> None:
        super().__init__()
        self.failed = asyncio.Event()

    async def start(self, handle, spec, config, callbacks) -> None:
        self.failed.set()
        raise RuntimeError('start failed')


async def test_asyncio_executor_cleans_assignment_after_driver_start_failure():
    driver = FailingStartDriver()
    executor = AsyncioExecutor()
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
        await asyncio.wait_for(driver.failed.wait(), timeout=2)
        deadline = asyncio.get_running_loop().time() + 2
        while await executor.has_assignment('assignment-1') and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        assert not await executor.has_assignment('assignment-1')
    finally:
        await executor.close()


class RuntimeInfoDriver(BlockingDriver):
    async def get_runtime_base_url(self, name):
        return f'http://runtime/{name}'


async def test_asyncio_executor_live_helpers_run_on_child_loop():
    driver = RuntimeInfoDriver()
    executor = AsyncioExecutor()
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
        await asyncio.wait_for(driver.started.wait(), timeout=2)
        assert await executor.get_runtime_base_url('assignment-1', 'main') == 'http://runtime/main'
        with pytest.raises(KeyError):
            await executor.get_runtime_base_url('missing', 'main')
    finally:
        await executor.close()


async def test_asyncio_executor_bridge_times_out_when_child_loop_is_blocked() -> None:
    release = threading.Event()
    entered = threading.Event()
    executor = AsyncioExecutor(bridge_timeout=0.02)
    await executor.start()
    try:
        loop = executor._loop
        assert loop is not None

        def block_loop() -> None:
            entered.set()
            release.wait()

        loop.call_soon_threadsafe(block_loop)
        assert await asyncio.to_thread(entered.wait, 1)

        with pytest.raises(TimeoutError):
            await executor.capacity_snapshot()
    finally:
        release.set()
        await executor.close()


async def test_asyncio_executor_health_probe_reports_responsive_loop() -> None:
    executor = AsyncioExecutor()
    await executor.start()
    try:
        health = await executor.probe_health(timeout=0.1, hang_timeout=1.0, now=1.0)

        assert health.responsive is True
        assert health.effective_capacity == 1
        assert health.unresponsive_workers == 0
    finally:
        await executor.close()


async def test_asyncio_executor_health_probe_bounds_blocked_loop() -> None:
    release = threading.Event()
    entered = threading.Event()
    executor = AsyncioExecutor()
    await executor.start()
    try:
        loop = executor._loop
        assert loop is not None

        def block_loop() -> None:
            entered.set()
            release.wait()

        loop.call_soon_threadsafe(block_loop)
        assert await asyncio.to_thread(entered.wait, 1)

        health = await executor.probe_health(timeout=0.02, hang_timeout=1.0, now=1.0)

        assert health.responsive is False
        assert health.effective_capacity == 0
        assert health.unresponsive_workers == 1
    finally:
        release.set()
        await executor.close()
