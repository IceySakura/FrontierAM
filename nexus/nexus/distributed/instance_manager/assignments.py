from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import logging
import os
import shutil
import traceback

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from nexus.distributed.batch.task_store import BatchTaskStore, decode_task_record
from nexus.distributed.config.views import InstanceManagerConfigView
from nexus.distributed.execution.drivers import NexusBatchDriver, NexusLiveDriver, NexusSingleDriver
from nexus.distributed.execution.executors.base import ExecutionDriver, Executor, StopDisposition
from nexus.distributed.execution.models import (
    AssignmentRecord,
    AssignmentStatus,
    CommandStatus,
    ExecutionRecord,
)
from nexus.distributed.execution.trace_backends import (
    RunTraceBackendRegistry,
    live_trace_metadata,
    single_trace_metadata,
)
from nexus.distributed.instance_manager.reporting import (
    ManagerBatchControl,
    ManagerExecutionReporter,
    ManagerServicePort,
    ResolvedAssignmentSpec,
    RunningAssignment,
    TemporaryTaskFiles,
)
from nexus.distributed.instance_manager.request_budget import RequestBudgetControl
from nexus.distributed.observability.context import distributed_execution_context
from nexus.distributed.persistence.orm import AssignmentRow, ExecutionRow
from nexus.distributed.policies.models import RequestPolicySelection
from nexus.distributed.repositories.manager import ManagerRepository
from nexus.distributed.utils.clock import utcnow
from nexus.runner.trace import RunTraceStreamEvent
from nexus.utils.otel.propagation import use_remote_context
from nexus.utils.otel.transport import transport_span
from nexus.utils.platform_metrics import metric_scope


logger = logging.getLogger(__name__)


class InstanceManagerAssignmentsMixin:
    config: InstanceManagerConfigView
    driver_factories: dict[str, Callable[..., ExecutionDriver]]
    endpoint_url: str
    executor: Executor
    manager_id: str
    session_factory: async_sessionmaker
    session_id: str
    trace_registry: RunTraceBackendRegistry
    request_budget_control: RequestBudgetControl | None
    batch_task_store: BatchTaskStore | None
    _command_lock: asyncio.Lock
    _poll_lock: asyncio.Lock
    _running: dict[str, RunningAssignment]
    _assignment_files_root: Path

    def _local_capacity_snapshot(self): ...

    def _mark_executor_unavailable(self) -> None: ...

    async def poll_assignments_once(self) -> int:
        async with self._poll_lock:
            return await self._poll_assignments_once_unlocked()

    async def claim_commands_once(self) -> int:
        async with self._command_lock:
            return await self._claim_commands_once_unlocked()

    async def reconcile_local_assignments_once(self) -> int:
        running = list(self._running.values())
        if not running:
            return 0

        stale_assignment_ids: set[str] = set()
        async with self.session_factory() as session:
            assignment_ids = [item.assignment_id for item in running]
            execution_ids = [item.execution_id for item in running]
            assignments = (
                await session.execute(
                    select(
                        AssignmentRow.assignment_id,
                        AssignmentRow.execution_id,
                        AssignmentRow.attempt_id,
                        AssignmentRow.manager_id,
                        AssignmentRow.manager_session_id,
                        AssignmentRow.status,
                    ).where(AssignmentRow.assignment_id.in_(assignment_ids))
                )
            ).all()
            executions = (
                await session.execute(
                    select(
                        ExecutionRow.execution_id,
                        ExecutionRow.current_assignment_id,
                        ExecutionRow.current_attempt_id,
                    ).where(ExecutionRow.execution_id.in_(execution_ids))
                )
            ).all()
            assignments_by_id = {assignment.assignment_id: assignment for assignment in assignments}
            executions_by_id = {execution.execution_id: execution for execution in executions}
            for item in running:
                assignment = assignments_by_id.get(item.assignment_id)
                execution = executions_by_id.get(item.execution_id)
                if (
                    assignment is None
                    or execution is None
                    or assignment.manager_id != self.manager_id
                    or assignment.manager_session_id != self.session_id
                    or assignment.status != AssignmentStatus.ACTIVE
                    or execution.current_assignment_id != item.assignment_id
                    or execution.current_attempt_id != assignment.attempt_id
                ):
                    stale_assignment_ids.add(item.assignment_id)

        removed = 0
        for assignment_id in stale_assignment_ids:
            try:
                await self.executor.stop_assignment(assignment_id, timeout=1.0)
            except Exception:
                try:
                    still_present = await self.executor.has_assignment(assignment_id)
                except Exception:
                    continue
                if still_present:
                    continue
            await self._release_running_assignment(assignment_id)
            removed += 1
        return removed

    async def read_artifact_result(self, assignment_id: str) -> dict[str, Any]:
        metadata = await self._run_trace_metadata_for_assignment(assignment_id)
        return await self.trace_registry.read_result(metadata)

    async def read_live_trace_events_after(self, assignment_id: str, cursor: int) -> list[RunTraceStreamEvent]:
        metadata = await self._live_trace_metadata_for_assignment(assignment_id)
        return await self.trace_registry.read_live_events_after(metadata, cursor)

    async def get_assignment_info(self, assignment_id: str) -> dict[str, Any]:
        return await self.executor.get_assignment_info(assignment_id)

    async def _live_trace_metadata_for_assignment(self, assignment_id: str) -> dict[str, Any]:
        metadata = await self._run_trace_metadata_for_assignment(assignment_id, live_only=True)
        if metadata is None:
            raise FileNotFoundError('live artifact metadata is not available')
        return metadata

    async def _run_trace_metadata_for_assignment(
        self,
        assignment_id: str,
        *,
        live_only: bool = False,
    ) -> dict[str, Any]:
        async with self.session_factory() as session:
            assignment = await session.get(AssignmentRow, assignment_id)
            if assignment is None:
                raise KeyError(assignment_id)
            metadata = live_trace_metadata(assignment.runtime_info_json)
            if metadata is None and not live_only:
                metadata = single_trace_metadata(assignment.runtime_info_json)
        if metadata is None:
            raise FileNotFoundError('run artifact metadata is not available')
        return metadata

    async def _poll_assignments_once_unlocked(self) -> int:
        capacity = self._local_capacity_snapshot()
        available_slots = min(capacity.available_slots, self._start_batch_size(capacity.available_slots))
        if available_slots <= 0:
            return 0
        async with self.session_factory() as session:
            repo = ManagerRepository(session, healthy_ttl=self.config.manager_healthy_ttl)
            pending = await repo.list_pending_assignments(self.manager_id, self.session_id, limit=available_slots)
            eligible = [
                (execution, assignment, trace_carrier)
                for execution, assignment, trace_carrier in pending
                if assignment is not None and execution.current_assignment is not None
            ]
            active_assignments = await repo.ack_assignments(
                [assignment.assignment_id for _, assignment, _ in eligible],
                self.manager_id,
                self.session_id,
                now=utcnow(),
            )
            active_assignment_ids = {assignment.assignment_id for assignment in active_assignments}
            acknowledged = [
                (execution, assignment, trace_carrier)
                for execution, assignment, trace_carrier in eligible
                if assignment.assignment_id in active_assignment_ids
            ]
            await session.commit()

        started = 0
        for execution, assignment, trace_carrier in acknowledged:
            if await self._start_assignment(execution, assignment, trace_carrier=trace_carrier):
                started += 1
        return started

    async def _claim_commands_once_unlocked(self) -> int:
        async with self.session_factory() as session:
            repo = ManagerRepository(session, healthy_ttl=self.config.manager_healthy_ttl)
            commands = await repo.claim_commands(self.manager_id, self.session_id, now=utcnow())
            await session.commit()

        for command in commands:
            with distributed_execution_context(
                execution_id=command.execution_id,
                assignment_id=command.assignment_id,
                attempt_id=command.attempt_id,
                command_id=command.command_id,
                manager_id=self.manager_id,
                manager_session_id=self.session_id,
            ):
                try:
                    logger.info('Dispatching distributed command', extra={'command_type': command.type})
                    if command.type == 'stop.request' and command.assignment_id:
                        running = self._running.get(command.assignment_id)
                        stop_disposition = (
                            running.stop_disposition if running is not None else StopDisposition.IMMEDIATE
                        )
                        try:
                            await self.executor.stop_assignment(
                                command.assignment_id, timeout=self._stop_timeout(command.execution_id)
                            )
                        except TimeoutError:
                            if await self.executor.has_assignment(command.assignment_id):
                                raise
                            logger.warning('Executor stop timed out after assignment was removed')
                        if stop_disposition == StopDisposition.DEFERRED:
                            await self._release_deferred_stop_if_active(command.assignment_id)
                        else:
                            await self._mark_killed_if_active(command.assignment_id)
                    elif command.assignment_id:
                        await self.executor.submit_command(command.assignment_id, command.type, command.payload)
                    await self._complete_command(command.command_id)
                except Exception as exc:
                    await self._complete_command(
                        command.command_id,
                        status=CommandStatus.FAILED,
                        error={'type': type(exc).__name__, 'message': str(exc)},
                    )
        return len(commands)

    async def _start_assignment(
        self,
        execution: ExecutionRecord,
        assignment: AssignmentRecord,
        *,
        trace_carrier: dict[str, str] | None = None,
    ) -> bool:
        with (
            metric_scope(environment=execution.environment_id or 'all'),
            use_remote_context(trace_carrier, fallback_only=False),
        ):
            with distributed_execution_context(
                execution_id=execution.execution_id,
                assignment_id=assignment.assignment_id,
                attempt_id=assignment.attempt_id,
                manager_id=self.manager_id,
                manager_session_id=self.session_id,
            ):
                with transport_span(
                    'distributed.assignment.start',
                    kind='consumer',
                    attributes={
                        'nexus.execution.id': execution.execution_id,
                        'nexus.assignment.id': assignment.assignment_id,
                        'nexus.attempt.id': assignment.attempt_id,
                        'nexus.manager.id': self.manager_id,
                    },
                ):
                    reporter = ManagerExecutionReporter(cast(ManagerServicePort, self), assignment)
                    untransferred_task_files: TemporaryTaskFiles | None = None
                    try:
                        driver = self._create_driver(execution, assignment)
                        if isinstance(driver, NexusSingleDriver):
                            await driver.prepare_data_credentials()
                        resolved = await self._resolve_execution_spec(
                            execution,
                            assignment_id=assignment.assignment_id,
                        )
                        untransferred_task_files = resolved.temporary_task_files
                        self._running[assignment.assignment_id] = RunningAssignment(
                            execution_id=execution.execution_id,
                            assignment_id=assignment.assignment_id,
                            attempt_id=assignment.attempt_id,
                            slot_cost=assignment.slot_cost,
                            stop_timeout_seconds=execution.stop_timeout_seconds,
                            stop_disposition=driver.stop_disposition,
                            temporary_task_files=resolved.temporary_task_files,
                        )
                        untransferred_task_files = None
                        logger.info(
                            'Starting distributed assignment',
                            extra={'execution_mode': execution.mode, 'spec_type': execution.spec_type},
                        )
                        await self.executor.start_assignment(
                            assignment_id=assignment.assignment_id,
                            execution_id=execution.execution_id,
                            driver=driver,
                            spec=resolved.spec,
                            config=execution.driver_config,
                            reporter=reporter,
                            slot_cost=assignment.slot_cost,
                        )
                    except TimeoutError as exc:
                        try:
                            present = await self.executor.has_assignment(assignment.assignment_id)
                        except (RuntimeError, TimeoutError):
                            self._mark_executor_unavailable()
                            logger.warning(
                                'Executor start timed out and assignment state is unknown; retaining reservation'
                            )
                            return True
                        if present:
                            logger.warning('Executor start timed out after assignment became present')
                            return True
                        await self._release_running_assignment(assignment.assignment_id)
                        logger.exception('Executor start timed out before assignment became present')
                        await self._mark_start_failed(
                            assignment.assignment_id,
                            error={
                                'type': 'AssignmentStartFailed',
                                'cause_type': type(exc).__name__,
                                'message': str(exc),
                                'traceback': traceback.format_exc(),
                            },
                        )
                        return False
                    except Exception as exc:
                        await self._release_running_assignment(assignment.assignment_id)
                        logger.exception('Failed to start distributed assignment')
                        await self._mark_start_failed(
                            assignment.assignment_id,
                            error={
                                'type': 'AssignmentStartFailed',
                                'cause_type': type(exc).__name__,
                                'message': str(exc),
                                'traceback': traceback.format_exc(),
                            },
                        )
                        return False
                    finally:
                        if untransferred_task_files is not None:
                            try:
                                await asyncio.to_thread(untransferred_task_files.cleanup)
                            except Exception:
                                logger.warning(
                                    'Failed to clean untransferred assignment task files',
                                    extra={'assignment_id': assignment.assignment_id},
                                    exc_info=True,
                                )
                    logger.info('Distributed assignment started')
        return True

    async def _resolve_execution_spec(
        self,
        execution: ExecutionRecord,
        *,
        assignment_id: str | None = None,
    ) -> ResolvedAssignmentSpec:
        spec = copy.deepcopy(execution.spec)
        raw_ref = spec.get('run_input_ref')
        if raw_ref is None:
            return ResolvedAssignmentSpec(spec=spec)
        if 'run_input' in spec or not isinstance(raw_ref, dict):
            raise ValueError('batch_run_input_reference_invalid')
        parent_execution_id = raw_ref.get('parent_execution_id')
        item_id = raw_ref.get('item_id')
        record_sha256 = raw_ref.get('record_sha256')
        if (
            raw_ref.get('version') != 1
            or not isinstance(parent_execution_id, str)
            or not isinstance(item_id, str)
            or not isinstance(record_sha256, str)
            or parent_execution_id != execution.parent_execution_id
        ):
            raise ValueError('batch_run_input_reference_invalid')
        task_store = self.batch_task_store
        if task_store is None or raw_ref.get('store') != task_store.backend_name:
            raise RuntimeError('batch_task_store_unavailable')
        record = await task_store.read_record(
            parent_execution_id=parent_execution_id,
            item_id=item_id,
        )
        if record.parent_execution_id != parent_execution_id or record.item_id != item_id:
            raise ValueError('batch_task_record_identity_mismatch')
        decoded = await asyncio.to_thread(
            decode_task_record,
            record,
            expected_sha256=record_sha256,
            max_record_bytes=self.config.batch_task_record_max_size_bytes,
            max_frame_bytes=self.config.batch_data_frame_max_size_bytes,
        )
        run_input = decoded.run_input
        temporary_task_files = None
        try:
            if decoded.frame is not None:
                if assignment_id is None or decoded.frame.zip_bytes is None:
                    raise ValueError('batch_task_record_invalid')
                temporary_task_files = await asyncio.to_thread(
                    _write_assignment_task_files,
                    self._assignment_files_root,
                    assignment_id,
                    decoded.frame.zip_bytes,
                    expected_sha256=decoded.frame.zip_sha256,
                    expected_size=decoded.frame.zip_size,
                )
                run_input['task_files'] = {
                    'type': 'zip',
                    'path': str(temporary_task_files.zip_path),
                    'sha256': decoded.frame.zip_sha256,
                    'size': decoded.frame.zip_size,
                }
                if root_name := decoded.frame.metadata.get('task_files', {}).get('root_name'):
                    run_input['task_files']['root_name'] = root_name
            run_input['run_id'] = execution.execution_id
            spec.pop('run_input_ref')
            spec['run_input'] = run_input
            return ResolvedAssignmentSpec(
                spec=spec,
                temporary_task_files=temporary_task_files,
            )
        except BaseException:
            if temporary_task_files is not None:
                try:
                    await asyncio.to_thread(temporary_task_files.cleanup)
                except Exception:
                    logger.warning('Failed to clean unresolved assignment task files', exc_info=True)
            raise

    async def _release_running_assignment(self, assignment_id: str) -> RunningAssignment | None:
        running = self._running.pop(assignment_id, None)
        if running is not None and self.request_budget_control is not None:
            self.request_budget_control.cancel_execution(running.execution_id, attempt_id=running.attempt_id)
        if running is None or running.temporary_task_files is None:
            return running
        try:
            await asyncio.to_thread(running.temporary_task_files.cleanup)
        except Exception:
            logger.warning(
                'Failed to clean assignment task files',
                extra={'assignment_id': assignment_id},
                exc_info=True,
            )
        return running

    def _create_driver(self, execution: ExecutionRecord, assignment: AssignmentRecord) -> ExecutionDriver:
        factory = self.driver_factories[execution.spec_type]
        return factory(
            execution=execution, assignment=assignment, internal_base_url=self._assignment_internal_base(assignment)
        )

    def _create_single_driver(
        self, *, execution: ExecutionRecord, assignment: AssignmentRecord, internal_base_url: str
    ):
        async def load_credentials() -> dict[str, Any]:
            from nexus.distributed.repositories.execution_credentials import ExecutionCredentialRepository

            async with self.session_factory() as session:
                repository = ExecutionCredentialRepository(session)
                result = {}
                for kind in ('wedata', 'cos'):
                    if payload := await repository.get_payload(execution.execution_id, kind, now=utcnow()):
                        result[kind] = payload
                return result

        return NexusSingleDriver(
            execution_id=execution.execution_id,
            assignment_id=assignment.assignment_id,
            attempt_id=assignment.attempt_id,
            owner_principal_id=execution.owner_principal_id,
            parent_execution_id=execution.parent_execution_id,
            internal_base_url=internal_base_url,
            trace_registry=self.trace_registry,
            artifact_scope_available=self._artifact_scope_available(execution, assignment),
            result_publish_deadline_at=self._result_publish_deadline_at(execution),
            execution_created_at=execution.created_at,
            data_credentials_loader=load_credentials,
            request_admission=self._resolve_execution_request_admission(execution),
        )

    def _create_live_driver(self, *, execution: ExecutionRecord, assignment: AssignmentRecord, internal_base_url: str):
        return NexusLiveDriver(
            execution_id=execution.execution_id,
            assignment_id=assignment.assignment_id,
            attempt_id=assignment.attempt_id,
            owner_principal_id=execution.owner_principal_id,
            internal_base_url=internal_base_url,
            trace_registry=self.trace_registry,
            artifact_scope_available=self._artifact_scope_available(execution, assignment),
            result_publish_deadline_at=self._result_publish_deadline_at(execution),
            execution_created_at=execution.created_at,
            request_admission=self._resolve_execution_request_admission(execution),
        )

    def _resolve_execution_request_admission(self, execution: ExecutionRecord) -> RequestBudgetControl | None:
        placement = execution.placement if isinstance(execution.placement, dict) else {}
        raw_expected = placement.get('request_policy')
        expected = RequestPolicySelection.model_validate(raw_expected) if raw_expected is not None else None
        actual = self.request_budget_control.selection if self.request_budget_control is not None else None
        if expected != actual:
            raise RuntimeError('execution_request_policy_mismatch')
        return self.request_budget_control

    def _create_batch_driver(self, *, execution: ExecutionRecord, assignment: AssignmentRecord, internal_base_url: str):
        del internal_base_url
        return NexusBatchDriver(
            execution_id=execution.execution_id,
            attempt_id=assignment.attempt_id,
            owner_principal_id=execution.owner_principal_id or '',
            batch_control=ManagerBatchControl(cast(ManagerServicePort, self)),
            trace_registry=self.trace_registry,
            reconcile_poll_interval=self.config.batch_reconcile_poll_interval,
        )

    @staticmethod
    def _artifact_scope_available(execution: ExecutionRecord, assignment: AssignmentRecord) -> bool:
        return bool(
            execution.owner_principal_id
            and execution.environment_id
            and execution.allocation_id
            and assignment.owner_principal_id == execution.owner_principal_id
            and assignment.environment_id == execution.environment_id
            and assignment.allocation_id == execution.allocation_id
        )

    @staticmethod
    def _result_publish_deadline_at(execution: ExecutionRecord) -> datetime | None:
        deadlines = [
            deadline
            for deadline in (execution.execution_expires_at, execution.stop_deadline_at)
            if deadline is not None
        ]
        return min(deadlines) if deadlines else None

    async def _mark_running(
        self,
        assignment_id: str,
        *,
        endpoint_url: str | None,
        endpoint_ready: bool,
        runtime_info: dict[str, Any],
    ) -> None:
        async with self.session_factory() as session:
            repo = ManagerRepository(session, healthy_ttl=self.config.manager_healthy_ttl)
            await repo.mark_running(
                assignment_id,
                self.manager_id,
                self.session_id,
                endpoint_url=endpoint_url,
                endpoint_ready=endpoint_ready,
                runtime_info=runtime_info,
                now=utcnow(),
            )
            await session.commit()

    async def _mark_waiting_input(self, assignment_id: str, *, payload: dict[str, Any]) -> None:
        async with self.session_factory() as session:
            repo = ManagerRepository(session, healthy_ttl=self.config.manager_healthy_ttl)
            await repo.mark_waiting_input(
                assignment_id, self.manager_id, self.session_id, payload=payload, now=utcnow()
            )
            await session.commit()

    async def _update_runtime_info(self, assignment_id: str, *, runtime_info: dict[str, Any]) -> None:
        async with self.session_factory() as session:
            repo = ManagerRepository(session, healthy_ttl=self.config.manager_healthy_ttl)
            await repo.update_runtime_info(
                assignment_id,
                self.manager_id,
                self.session_id,
                runtime_info=runtime_info,
                now=utcnow(),
            )
            await session.commit()

    async def _complete(
        self,
        assignment_id: str,
        *,
        result: dict[str, Any],
        runtime_info: dict[str, Any] | None = None,
    ) -> None:
        async with self.session_factory() as session:
            repo = ManagerRepository(session, healthy_ttl=self.config.manager_healthy_ttl)
            await repo.complete_execution(
                assignment_id,
                self.manager_id,
                self.session_id,
                result=result,
                runtime_info=runtime_info,
                now=utcnow(),
            )
            await session.commit()
        await self._release_running_assignment(assignment_id)

    async def _fail(
        self,
        assignment_id: str,
        *,
        error: dict[str, Any],
        result: dict[str, Any] | None = None,
        runtime_info: dict[str, Any] | None = None,
    ) -> None:
        async with self.session_factory() as session:
            repo = ManagerRepository(session, healthy_ttl=self.config.manager_healthy_ttl)
            await repo.fail_execution(
                assignment_id,
                self.manager_id,
                self.session_id,
                error=error,
                result=result,
                runtime_info=runtime_info,
                now=utcnow(),
            )
            await session.commit()
        await self._release_running_assignment(assignment_id)

    async def _mark_killed(self, assignment_id: str) -> None:
        async with self.session_factory() as session:
            repo = ManagerRepository(session, healthy_ttl=self.config.manager_healthy_ttl)
            await repo.mark_killed(assignment_id, self.manager_id, self.session_id, now=utcnow())
            await session.commit()
        await self._release_running_assignment(assignment_id)

    async def _mark_killed_if_active(self, assignment_id: str) -> None:
        async with self.session_factory() as session:
            repo = ManagerRepository(session, healthy_ttl=self.config.manager_healthy_ttl)
            marked = await repo.mark_killed_if_active(assignment_id, self.manager_id, self.session_id, now=utcnow())
            await session.commit()
        if marked:
            await self._release_running_assignment(assignment_id)

    async def _release_deferred_stop_if_active(self, assignment_id: str) -> None:
        async with self.session_factory() as session:
            repo = ManagerRepository(session, healthy_ttl=self.config.manager_healthy_ttl)
            released = await repo.release_deferred_stop_if_active(
                assignment_id,
                self.manager_id,
                self.session_id,
                now=utcnow(),
            )
            await session.commit()
        if released:
            await self._release_running_assignment(assignment_id)

    async def _mark_start_failed(self, assignment_id: str, *, error: dict[str, Any]) -> None:
        async with self.session_factory() as session:
            repo = ManagerRepository(session, healthy_ttl=self.config.manager_healthy_ttl)
            await repo.mark_start_failed(assignment_id, self.manager_id, self.session_id, error=error, now=utcnow())
            await session.commit()

    async def _complete_command(
        self,
        command_id: str,
        *,
        status: str = CommandStatus.DONE,
        error: dict[str, Any] | None = None,
    ) -> None:
        async with self.session_factory() as session:
            repo = ManagerRepository(session, healthy_ttl=self.config.manager_healthy_ttl)
            await repo.complete_command(command_id, status=status, error=error, now=utcnow())
            await session.commit()

    def _start_batch_size(self, available_slots: int) -> int:
        batch_size = self.config.manager_start_batch_size
        if batch_size <= 0:
            return available_slots
        return batch_size

    def _assignment_internal_base(self, assignment: AssignmentRecord) -> str:
        return f'{self.endpoint_url}/v1/internal/executions/{assignment.assignment_id}'

    def _stop_timeout(self, execution_id: str) -> float:
        for item in self._running.values():
            if item.execution_id == execution_id and item.stop_timeout_seconds is not None:
                return float(item.stop_timeout_seconds)
        return 10.0


def _write_assignment_task_files(
    root: Path,
    assignment_id: str,
    zip_bytes: bytes,
    *,
    expected_sha256: str,
    expected_size: int,
) -> TemporaryTaskFiles:
    if (
        not assignment_id
        or assignment_id in {'.', '..'}
        or Path(assignment_id).name != assignment_id
        or '/' in assignment_id
        or '\\' in assignment_id
    ):
        raise ValueError('batch_task_record_invalid')
    if len(zip_bytes) != expected_size or hashlib.sha256(zip_bytes).hexdigest() != expected_sha256:
        raise ValueError('batch_task_record_invalid')

    directory = root / assignment_id
    archive_path = directory / 'files.zip'
    temporary_path = directory / '.files.zip.tmp'
    try:
        directory.mkdir(mode=0o700)
        os.chmod(directory, 0o700)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
        descriptor = os.open(temporary_path, flags, 0o600)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(zip_bytes)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(descriptor)
            raise
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, archive_path)
        directory_descriptor = os.open(directory, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return TemporaryTaskFiles(directory=directory, zip_path=archive_path)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise


__all__ = ['InstanceManagerAssignmentsMixin']
