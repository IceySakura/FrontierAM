from __future__ import annotations

import asyncio
import threading

import pytest

from nexus.distributed.execution.executors.asyncio_executor import AsyncioExecutor
from nexus.distributed.execution.executors.base import StopDisposition
from nexus.distributed.execution.executors.thread_executor import ThreadExecutor


class UnhandledFailureDriver:
    spec_type = 'unhandled_failure'
    stop_disposition = StopDisposition.IMMEDIATE

    async def start(self, handle, spec, config, reporter) -> None:
        raise RuntimeError('driver escaped its exception boundary')

    async def submit_command(self, handle, command_type, payload) -> None:
        raise AssertionError('not used')

    async def stop(self, handle) -> None:
        return None


class RecordingReporter:
    def __init__(self) -> None:
        self.failed = threading.Event()
        self.error: dict | None = None

    async def fail(self, error: dict, *, result: dict | None = None) -> None:
        assert result is None
        self.error = error
        self.failed.set()


@pytest.mark.parametrize(
    'executor',
    [
        pytest.param(AsyncioExecutor(), id='asyncio'),
        pytest.param(ThreadExecutor(strategy='per_execution', max_threads=1), id='thread'),
    ],
)
async def test_executor_reports_unhandled_driver_failure_and_releases_assignment(executor):
    reporter = RecordingReporter()
    await executor.start()
    try:
        await executor.start_assignment(
            assignment_id='assignment-unhandled',
            execution_id='execution-unhandled',
            driver=UnhandledFailureDriver(),
            spec={},
            config={},
            reporter=reporter,
        )

        assert await asyncio.to_thread(reporter.failed.wait, 2)
        assert reporter.error is not None
        assert {key: reporter.error[key] for key in ('type', 'cause_type', 'message')} == {
            'type': 'UnhandledDriverError',
            'cause_type': 'RuntimeError',
            'message': 'driver escaped its exception boundary',
        }
        assert 'RuntimeError: driver escaped its exception boundary' in reporter.error['traceback']
        deadline = asyncio.get_running_loop().time() + 2
        while await executor.has_assignment('assignment-unhandled') and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        assert not await executor.has_assignment('assignment-unhandled')
    finally:
        await executor.close()
