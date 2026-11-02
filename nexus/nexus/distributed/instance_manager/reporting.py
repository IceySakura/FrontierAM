from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

from sqlalchemy.exc import DBAPIError

from nexus.distributed.batch.ports import (
    BatchControl,
    BatchControlServicePort,
    BatchControlStorageError,
    BatchControlTransientError,
)
from nexus.distributed.execution.executors.base import ExecutionReporter, StopDisposition
from nexus.distributed.execution.models import AssignmentRecord
from nexus.distributed.persistence.db import is_retryable_database_error
from nexus.utils.platform_metrics import measured


logger = logging.getLogger(__name__)

_T = TypeVar('_T')
_TRANSACTION_RETRY_LIMIT = 8
_DEFAULT_BATCH_GENERATION_BRIDGE_TIMEOUT_SECONDS = 3602.0


@dataclass(slots=True)
class RunningAssignment:
    execution_id: str
    assignment_id: str
    slot_cost: int
    stop_timeout_seconds: int | None
    stop_disposition: StopDisposition
    attempt_id: str | None = None
    temporary_task_files: TemporaryTaskFiles | None = None


@dataclass(slots=True)
class TemporaryTaskFiles:
    directory: Path
    zip_path: Path
    _cleaned: bool = field(default=False, init=False, repr=False)

    def cleanup(self) -> None:
        if self._cleaned:
            return
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(self.directory)
        self._cleaned = True


@dataclass(frozen=True, slots=True)
class ResolvedAssignmentSpec:
    spec: dict[str, Any]
    temporary_task_files: TemporaryTaskFiles | None = None


class ManagerServicePort(Protocol):
    batch_control: BatchControlServicePort

    async def call_manager_loop(self, coroutine: Awaitable[_T], *, timeout: float = 30.0) -> _T: ...

    async def _mark_running(
        self,
        assignment_id: str,
        *,
        endpoint_url: str | None,
        endpoint_ready: bool,
        runtime_info: dict[str, Any],
    ) -> None: ...

    async def _mark_waiting_input(self, assignment_id: str, *, payload: dict[str, Any]) -> None: ...

    async def _update_runtime_info(self, assignment_id: str, *, runtime_info: dict[str, Any]) -> None: ...

    async def _complete(
        self,
        assignment_id: str,
        *,
        result: dict[str, Any],
        runtime_info: dict[str, Any] | None = None,
    ) -> None: ...

    async def _fail(
        self,
        assignment_id: str,
        *,
        error: dict[str, Any],
        result: dict[str, Any] | None = None,
        runtime_info: dict[str, Any] | None = None,
    ) -> None: ...

    async def _mark_killed(self, assignment_id: str) -> None: ...


class ManagerExecutionReporter(ExecutionReporter):
    def __init__(self, service: ManagerServicePort, assignment: AssignmentRecord) -> None:
        self._service = service
        self._assignment = assignment

    async def mark_running(
        self,
        *,
        endpoint_url: str | None = None,
        endpoint_ready: bool = False,
        runtime_info: dict[str, Any] | None = None,
    ) -> None:
        await self._service.call_manager_loop(
            retry_transient_db_transaction(
                lambda: self._service._mark_running(
                    self._assignment.assignment_id,
                    endpoint_url=endpoint_url,
                    endpoint_ready=endpoint_ready,
                    runtime_info=runtime_info or {},
                )
            )
        )

    async def mark_waiting_input(self, payload: dict[str, Any]) -> None:
        await self._service.call_manager_loop(
            retry_transient_db_transaction(
                lambda: self._service._mark_waiting_input(self._assignment.assignment_id, payload=payload)
            )
        )

    async def update_runtime_info(self, runtime_info: dict[str, Any]) -> None:
        await self._service.call_manager_loop(
            retry_transient_db_transaction(
                lambda: self._service._update_runtime_info(self._assignment.assignment_id, runtime_info=runtime_info)
            )
        )

    @measured('terminal_report', family='terminal', operation='complete')
    async def complete(
        self,
        result: dict[str, Any],
        *,
        runtime_info: dict[str, Any] | None = None,
    ) -> None:
        await self._service.call_manager_loop(
            retry_transient_db_transaction(
                lambda: self._service._complete(
                    self._assignment.assignment_id,
                    result=result,
                    runtime_info=runtime_info,
                )
            )
        )

    @measured('terminal_report', family='terminal', operation='fail')
    async def fail(
        self,
        error: dict[str, Any],
        *,
        result: dict[str, Any] | None = None,
        runtime_info: dict[str, Any] | None = None,
    ) -> None:
        await self._service.call_manager_loop(
            retry_transient_db_transaction(
                lambda: self._service._fail(
                    self._assignment.assignment_id,
                    error=error,
                    result=result,
                    runtime_info=runtime_info,
                )
            )
        )

    @measured('terminal_report', family='terminal', operation='mark_killed')
    async def mark_killed(self) -> None:
        await self._service.call_manager_loop(
            retry_transient_db_transaction(lambda: self._service._mark_killed(self._assignment.assignment_id))
        )


class ManagerBatchControl(BatchControl):
    def __init__(self, service: ManagerServicePort) -> None:
        self._service = service

    async def _call(self, operation: Callable[[], Awaitable[_T]], *, timeout: float = 30.0) -> _T:
        return await self._service.call_manager_loop(
            retry_batch_control_operation(operation),
            timeout=timeout,
        )

    async def prepare(self, parent_execution_id: str, *, parent_attempt_id: str | None = None):
        return await self._call(
            lambda: self._service.batch_control.prepare(parent_execution_id, parent_attempt_id=parent_attempt_id)
        )

    async def ensure_prepared(
        self,
        parent_execution_id: str,
        *,
        parent_attempt_id: str | None = None,
        bridge_timeout: float | None = None,
    ):
        return await self._call(
            lambda: self._service.batch_control.ensure_prepared(
                parent_execution_id,
                parent_attempt_id=parent_attempt_id,
            ),
            timeout=bridge_timeout if bridge_timeout is not None else _DEFAULT_BATCH_GENERATION_BRIDGE_TIMEOUT_SECONDS,
        )

    async def reconcile_once(
        self,
        parent_execution_id: str,
        *,
        max_inflight: int,
        submit_batch_size: int,
        parent_attempt_id: str | None = None,
    ):
        return await self._call(
            lambda: self._service.batch_control.reconcile_once(
                parent_execution_id,
                max_inflight=max_inflight,
                submit_batch_size=submit_batch_size,
                parent_attempt_id=parent_attempt_id,
            )
        )

    async def stop_children(self, parent_execution_id: str, *, parent_attempt_id: str | None = None) -> None:
        await self._call(
            lambda: self._service.batch_control.stop_children(
                parent_execution_id,
                parent_attempt_id=parent_attempt_id,
            )
        )

    async def release_parent(self, parent_execution_id: str) -> None:
        await self._call(lambda: self._service.batch_control.release_parent(parent_execution_id))


async def retry_batch_control_operation(operation: Callable[[], Awaitable[_T]]) -> _T:
    for attempt in range(_TRANSACTION_RETRY_LIMIT + 1):
        try:
            return await operation()
        except DBAPIError as exc:
            if not is_retryable_database_error(exc):
                raise BatchControlStorageError('batch_control_storage_failed') from exc
            if attempt >= _TRANSACTION_RETRY_LIMIT:
                raise BatchControlTransientError('batch_control_storage_retry_exhausted') from exc
            delay = min(0.05 * (2**attempt), 1.0)
            logger.warning('Retrying BatchControl storage operation', exc_info=True)
            await asyncio.sleep(delay)
    raise AssertionError('unreachable')


async def retry_transient_db_transaction(operation: Callable[[], Awaitable[_T]]) -> _T:
    for attempt in range(_TRANSACTION_RETRY_LIMIT + 1):
        try:
            return await operation()
        except DBAPIError as exc:
            if not is_retryable_database_error(exc) or attempt >= _TRANSACTION_RETRY_LIMIT:
                raise
            delay = min(0.05 * (2**attempt), 1.0)
            logger.warning('Retrying manager transaction after database conflict', exc_info=True)
            await asyncio.sleep(delay)
    raise AssertionError('unreachable')


__all__ = [
    'BatchControlServicePort',
    'ManagerBatchControl',
    'ManagerExecutionReporter',
    'ManagerServicePort',
    'ResolvedAssignmentSpec',
    'RunningAssignment',
    'TemporaryTaskFiles',
    'retry_batch_control_operation',
    'retry_transient_db_transaction',
]
