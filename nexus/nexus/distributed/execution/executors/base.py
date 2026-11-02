from __future__ import annotations

import asyncio
import logging
import traceback

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExecutionHandle:
    execution_id: str
    assignment_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutorCapacitySnapshot:
    capacity_total: int
    running_slot_cost: int
    available_slots: int


@dataclass(frozen=True, slots=True)
class QuarantinedAssignment:
    assignment_id: str | None
    execution_id: str | None
    worker_id: str
    slot_cost: int


@dataclass(frozen=True, slots=True)
class ExecutorHealthSnapshot:
    responsive: bool
    effective_capacity: int
    unresponsive_workers: int = 0
    quarantined_assignments: tuple[QuarantinedAssignment, ...] = ()
    recycle_requested: bool = False


class StopDisposition(str, Enum):
    IMMEDIATE = 'immediate'
    DEFERRED = 'deferred'


class ExecutionReporter(Protocol):
    async def mark_running(
        self,
        *,
        endpoint_url: str | None = None,
        endpoint_ready: bool = False,
        runtime_info: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    async def mark_waiting_input(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    async def update_runtime_info(self, runtime_info: dict[str, Any]) -> None:
        raise NotImplementedError

    async def complete(
        self,
        result: dict[str, Any],
        *,
        runtime_info: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    async def fail(
        self,
        error: dict[str, Any],
        *,
        result: dict[str, Any] | None = None,
        runtime_info: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    async def mark_killed(self) -> None:
        raise NotImplementedError


class ExecutionDriver(Protocol):
    spec_type: str
    stop_disposition: StopDisposition

    async def start(
        self,
        handle: ExecutionHandle,
        spec: dict[str, Any],
        config: dict[str, Any],
        reporter: ExecutionReporter,
    ) -> None:
        raise NotImplementedError

    async def submit_command(self, handle: ExecutionHandle, command_type: str, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    async def stop(self, handle: ExecutionHandle) -> None:
        raise NotImplementedError


async def run_driver_with_failure_reporting(
    driver: ExecutionDriver,
    handle: ExecutionHandle,
    spec: dict[str, Any],
    config: dict[str, Any],
    reporter: ExecutionReporter,
) -> None:
    try:
        await driver.start(handle, spec, config, reporter)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        error = {
            'type': 'UnhandledDriverError',
            'cause_type': type(exc).__name__,
            'message': str(exc),
            'traceback': traceback.format_exc(),
        }
        try:
            await reporter.fail(error)
        except Exception:
            logger.exception('Failed to report unhandled distributed driver error')


@runtime_checkable
class SseEventStreamingDriver(Protocol):
    async def collect_sse_events_after(self, cursor: int, *, timeout: float = 15.0) -> list[tuple[int, bytes]]:
        raise NotImplementedError


@runtime_checkable
class RuntimeEndpointDriver(Protocol):
    async def get_runtime_base_url(self, runtime_name: str) -> str:
        raise NotImplementedError


@runtime_checkable
class ExecutionInfoDriver(Protocol):
    async def get_execution_info(self, handle: ExecutionHandle) -> dict[str, Any]:
        raise NotImplementedError


@runtime_checkable
class LiveExecutionDriver(
    ExecutionDriver,
    SseEventStreamingDriver,
    RuntimeEndpointDriver,
    ExecutionInfoDriver,
    Protocol,
):
    pass


class Executor(Protocol):
    instance_mode: str

    async def start(self) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError

    async def has_assignment(self, assignment_id: str) -> bool:
        raise NotImplementedError

    async def capacity_snapshot(self) -> ExecutorCapacitySnapshot:
        raise NotImplementedError

    async def probe_health(self, *, timeout: float, hang_timeout: float, now: float) -> ExecutorHealthSnapshot:
        raise NotImplementedError

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
        raise NotImplementedError

    async def stop_assignment(self, assignment_id: str, timeout: float = 10.0) -> None:
        raise NotImplementedError

    async def submit_command(self, assignment_id: str, command_type: str, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    async def collect_events_after(
        self, assignment_id: str, cursor: int, *, timeout: float = 15.0
    ) -> list[tuple[int, bytes]]:
        raise NotImplementedError

    async def get_runtime_base_url(self, assignment_id: str, runtime_name: str) -> str:
        raise NotImplementedError

    async def get_assignment_info(self, assignment_id: str) -> dict[str, Any]:
        raise NotImplementedError
