from __future__ import annotations

import asyncio

from typing import get_type_hints

import pytest

from nexus.distributed.execution.drivers.nexus_batch import NexusBatchDriver
from nexus.distributed.execution.drivers.nexus_live import NexusLiveDriver
from nexus.distributed.execution.drivers.nexus_single import NexusSingleDriver
from nexus.distributed.execution.executors.asyncio_executor import AsyncioExecutor
from nexus.distributed.execution.executors.base import (
    ExecutionDriver,
    ExecutionHandle,
    ExecutionReporter,
    Executor,
    LiveExecutionDriver,
    RuntimeEndpointDriver,
    SseEventStreamingDriver,
    StopDisposition,
)
from nexus.distributed.execution.executors.thread_executor import ThreadExecutor
from nexus.distributed.execution.trace_backends import RunTraceBackendRegistry


class HandleProbeDriver:
    spec_type = 'handle_probe'
    stop_disposition = StopDisposition.IMMEDIATE

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.handle: ExecutionHandle | None = None

    async def start(
        self,
        handle: ExecutionHandle,
        spec: dict,
        config: dict,
        reporter: ExecutionReporter,
    ) -> None:
        self.handle = handle
        self.started.set()
        await asyncio.Event().wait()

    async def submit_command(self, handle: ExecutionHandle, command_type: str, payload: dict) -> None:
        raise RuntimeError('not used')

    async def stop(self, handle: ExecutionHandle) -> None:
        return None


class FailureReporter:
    def __init__(self) -> None:
        self.error: dict | None = None

    async def fail(self, error: dict, *, result: dict | None = None) -> None:
        assert result is None
        self.error = error


def test_stop_disposition_accepts_only_declared_values():
    assert StopDisposition('immediate') is StopDisposition.IMMEDIATE
    assert StopDisposition('deferred') is StopDisposition.DEFERRED
    with pytest.raises(ValueError, match='is not a valid StopDisposition'):
        StopDisposition('unknown')


def test_execution_driver_protocol_declares_stop_disposition():
    assert get_type_hints(ExecutionDriver)['stop_disposition'] is StopDisposition


@pytest.mark.parametrize(
    ('driver_type', 'expected'),
    [
        (NexusLiveDriver, StopDisposition.IMMEDIATE),
        (NexusSingleDriver, StopDisposition.IMMEDIATE),
        (NexusBatchDriver, StopDisposition.DEFERRED),
    ],
)
def test_distributed_drivers_declare_stop_disposition(driver_type, expected):
    assert driver_type.stop_disposition is expected


async def test_executor_creates_execution_handle_and_passes_it_to_driver_start():
    driver = HandleProbeDriver()
    executor = AsyncioExecutor()
    await executor.start()
    try:
        handle = await executor.start_assignment(
            assignment_id='assignment-1',
            execution_id='execution-1',
            driver=driver,
            spec={},
            config={},
            reporter=object(),
        )
        await asyncio.wait_for(driver.started.wait(), timeout=2)

        assert isinstance(handle, ExecutionHandle)
        assert driver.handle is handle
        assert handle.execution_id == 'execution-1'
        assert handle.assignment_id == 'assignment-1'
    finally:
        await executor.close()


async def test_executor_rejects_unsupported_sse_and_runtime_capabilities():
    driver = HandleProbeDriver()
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

        with pytest.raises(ValueError, match='events_not_supported'):
            await executor.collect_events_after('assignment-1', 0, timeout=0.01)
        with pytest.raises(ValueError, match='runtime_proxy_not_supported'):
            await executor.get_runtime_base_url('assignment-1', 'main')
    finally:
        await executor.close()


def test_live_driver_declares_live_capability_protocols():
    driver = NexusLiveDriver(
        execution_id='execution-1',
        assignment_id='assignment-1',
        internal_base_url='http://manager/v1/internal/executions/assignment-1',
        trace_registry=RunTraceBackendRegistry(default_sink_names=()),
    )

    assert isinstance(driver, LiveExecutionDriver)
    assert isinstance(driver, SseEventStreamingDriver)
    assert isinstance(driver, RuntimeEndpointDriver)


def test_executor_implementations_inherit_executor_protocol():
    assert Executor in AsyncioExecutor.__mro__
    assert Executor in ThreadExecutor.__mro__


@pytest.mark.parametrize('driver_kind', ['single', 'live'])
async def test_driver_request_construction_failure_is_reported(driver_kind):
    registry = RunTraceBackendRegistry(default_sink_names=())
    if driver_kind == 'single':
        driver = NexusSingleDriver(
            execution_id='execution-invalid',
            assignment_id='assignment-invalid',
            trace_registry=registry,
        )
    else:
        driver = NexusLiveDriver(
            execution_id='execution-invalid',
            assignment_id='assignment-invalid',
            internal_base_url='http://manager/v1/internal/executions/assignment-invalid',
            trace_registry=registry,
        )
    reporter = FailureReporter()

    await driver.start(
        ExecutionHandle(execution_id='execution-invalid', assignment_id='assignment-invalid'),
        {'run_input': {}},
        {},
        reporter,
    )

    assert reporter.error is not None
    assert reporter.error['type'] == 'InvalidExecutionSpec'
    assert 'task_name' in reporter.error['message']
