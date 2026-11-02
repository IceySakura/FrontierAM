from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import load_only

from nexus.distributed.environment.models import (
    DEFAULT_ENVIRONMENT_ID,
    DEFAULT_POOL_ID,
)
from nexus.distributed.errors import (
    DomainConflictError,
    DomainValidationError,
    FencingError,
    ResourceNotFoundError,
    TemporaryUnavailableError,
)
from nexus.distributed.execution.models import (
    AssignmentRecord,
    AssignmentStatus,
    CommandStatus,
    ExecutionStatus,
    RouteStatus,
)
from nexus.distributed.execution.runtime_info import merge_runtime_info
from nexus.distributed.instance_manager.liveness import effective_manager_liveness
from nexus.distributed.observability.database import queue_committed_metric
from nexus.distributed.persistence.orm import (
    ArtifactRow,
    AssignmentRow,
    ExecutionCommandRow,
    ExecutionEventRow,
    ExecutionRow,
    InstanceManagerLivenessRow,
    InstanceManagerRow,
)
from nexus.distributed.repositories._records import assignment_record, command_record, execution_record
from nexus.distributed.repositories.events import append_control_event_row
from nexus.distributed.trace.bindings import source_binding, validate_source_binding_update
from nexus.distributed.utils.clock import deadline_after, utcnow
from nexus.distributed.utils.identity import validate_environment_id, validate_pool_id


_ACK_EXECUTION_COLUMNS = (
    ExecutionRow.execution_id,
    ExecutionRow.status,
    ExecutionRow.current_assignment_id,
    ExecutionRow.version,
    ExecutionRow.next_event_seq,
    ExecutionRow.updated_at,
)

_ACK_ASSIGNMENT_COLUMNS = (
    AssignmentRow.assignment_id,
    AssignmentRow.execution_id,
    AssignmentRow.owner_principal_id,
    AssignmentRow.environment_id,
    AssignmentRow.pool_id,
    AssignmentRow.allocation_id,
    AssignmentRow.workload,
    AssignmentRow.allocation_spec_hash,
    AssignmentRow.attempt_id,
    AssignmentRow.attempt_seq,
    AssignmentRow.manager_id,
    AssignmentRow.manager_session_id,
    AssignmentRow.status,
    AssignmentRow.slot_cost,
    AssignmentRow.endpoint_url,
    AssignmentRow.endpoint_ready,
    AssignmentRow.runtime_info_json,
    AssignmentRow.created_at,
    AssignmentRow.updated_at,
)

_COMMAND_COLUMNS = (
    ExecutionCommandRow.command_id,
    ExecutionCommandRow.execution_id,
    ExecutionCommandRow.assignment_id,
    ExecutionCommandRow.attempt_id,
    ExecutionCommandRow.type,
    ExecutionCommandRow.status,
    ExecutionCommandRow.payload_json,
    ExecutionCommandRow.idempotency_key,
    ExecutionCommandRow.claimed_by_manager_id,
    ExecutionCommandRow.claimed_at,
    ExecutionCommandRow.completed_at,
    ExecutionCommandRow.created_at,
    ExecutionCommandRow.updated_at,
)

_COMMAND_ASSIGNMENT_COLUMNS = (
    AssignmentRow.assignment_id,
    AssignmentRow.execution_id,
    AssignmentRow.attempt_id,
    AssignmentRow.manager_id,
    AssignmentRow.manager_session_id,
    AssignmentRow.status,
)

_COMMAND_EXECUTION_COLUMNS = (
    ExecutionRow.execution_id,
    ExecutionRow.current_assignment_id,
    ExecutionRow.current_attempt_id,
    ExecutionRow.next_event_seq,
)


class ManagerRepository:
    def __init__(self, session: AsyncSession, *, healthy_ttl: timedelta = timedelta(seconds=15)) -> None:
        self.session = session
        self.healthy_ttl = healthy_ttl

    def _is_healthy(
        self,
        row: InstanceManagerRow,
        liveness: InstanceManagerLivenessRow | None,
        now: datetime,
    ) -> bool:
        effective = effective_manager_liveness(row, liveness)
        return effective.active and now - effective.last_heartbeat_at <= self.healthy_ttl

    async def register_session(
        self,
        *,
        manager_id: str,
        session_id: str,
        endpoint_url: str,
        supported_modes: list[str],
        supported_spec_types: list[str],
        supported_instance_modes: list[str],
        capacity_total: int,
        force_replace: bool,
        now: datetime,
        environment_id: str = DEFAULT_ENVIRONMENT_ID,
        pool_id: str = DEFAULT_POOL_ID,
        workload: str = 'execution',
        labels: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        draining: bool = False,
        bootstrap_version: int | None = None,
        artifact_ref_version: int | None = None,
        supported_spec_versions: dict[str, list[int]] | None = None,
        owner_principal_id: str | None = None,
        allocation_id: str | None = None,
        allocation_spec_hash: str | None = None,
        request_policy_binding: str | None = None,
        request_policy_scope: str | None = None,
        capability_snapshot: dict[str, Any] | None = None,
    ) -> bool:
        validate_environment_id(environment_id)
        validate_pool_id(pool_id)
        if workload not in {'execution', 'orchestration'}:
            raise DomainValidationError('invalid_manager_workload')
        row = await self.session.get(InstanceManagerRow, manager_id)
        liveness = await self.session.get(InstanceManagerLivenessRow, manager_id) if row is not None else None
        if (
            row is not None
            and self._is_healthy(row, liveness, now)
            and row.session_id != session_id
            and not force_replace
        ):
            return False
        if (
            row is not None
            and (row.request_policy_binding, row.request_policy_scope) != (request_policy_binding, request_policy_scope)
            and (row.session_id == session_id or not row.draining or (row.observed_capacity_used or 0) > 0)
        ):
            raise DomainValidationError('manager_request_policy_immutable')
        if row is None:
            row = InstanceManagerRow(
                manager_id=manager_id,
                session_id=session_id,
                endpoint_url=endpoint_url,
                labels_json=dict(labels or {}),
                metadata_json=dict(metadata or {}),
                draining=draining,
                draining_at=now if draining else None,
                owner_principal_id=owner_principal_id,
                environment_id=environment_id,
                pool_id=pool_id,
                allocation_id=allocation_id,
                workload=workload,
                allocation_spec_hash=allocation_spec_hash,
                request_policy_binding=request_policy_binding,
                request_policy_scope=request_policy_scope,
                capability_snapshot_json=dict(capability_snapshot or {}),
                bootstrap_version=bootstrap_version,
                artifact_ref_version=artifact_ref_version,
                supported_spec_versions_json=dict(supported_spec_versions or {}),
                supported_modes_json=supported_modes,
                supported_spec_types_json=supported_spec_types,
                supported_instance_modes_json=supported_instance_modes,
                capacity_total=capacity_total,
                observed_capacity_used=0,
                local_instances_json={},
                last_heartbeat_at=now,
                created_at=now,
                updated_at=now,
            )
            self.session.add(row)
        else:
            row.session_id = session_id
            row.endpoint_url = endpoint_url
            row.labels_json = dict(labels or {})
            row.metadata_json = dict(metadata or {})
            row.draining = draining
            row.draining_at = row.draining_at or now if draining else None
            row.owner_principal_id = owner_principal_id
            row.environment_id = environment_id
            row.pool_id = pool_id
            row.allocation_id = allocation_id
            row.workload = workload
            row.allocation_spec_hash = allocation_spec_hash
            row.request_policy_binding = request_policy_binding
            row.request_policy_scope = request_policy_scope
            row.capability_snapshot_json = dict(capability_snapshot or {})
            row.bootstrap_version = bootstrap_version
            row.artifact_ref_version = artifact_ref_version
            row.supported_spec_versions_json = dict(supported_spec_versions or {})
            row.supported_modes_json = supported_modes
            row.supported_spec_types_json = supported_spec_types
            row.supported_instance_modes_json = supported_instance_modes
            row.capacity_total = capacity_total
            row.last_heartbeat_at = now
            row.updated_at = now
        await self.session.flush()
        if liveness is None:
            liveness = InstanceManagerLivenessRow(
                manager_id=manager_id,
                session_id=session_id,
                capacity_total=capacity_total,
                observed_capacity_used=0,
                last_heartbeat_at=now,
                fenced_at=None,
                updated_at=now,
            )
            self.session.add(liveness)
        else:
            liveness.session_id = session_id
            liveness.capacity_total = capacity_total
            liveness.observed_capacity_used = 0
            liveness.last_heartbeat_at = now
            liveness.fenced_at = None
            liveness.updated_at = now
        await self.session.flush()
        return True

    async def heartbeat(
        self,
        *,
        manager_id: str,
        session_id: str,
        capacity_total: int,
        observed_capacity_used: int,
        local_instances: dict | None = None,
        now: datetime,
    ) -> bool:
        del local_instances
        current_registration = (
            select(InstanceManagerRow.manager_id)
            .where(
                InstanceManagerRow.manager_id == manager_id,
                InstanceManagerRow.session_id == session_id,
            )
            .exists()
        )
        result = await self.session.execute(
            update(InstanceManagerLivenessRow)
            .where(
                InstanceManagerLivenessRow.manager_id == manager_id,
                InstanceManagerLivenessRow.session_id == session_id,
                InstanceManagerLivenessRow.fenced_at.is_(None),
                current_registration,
            )
            .values(
                capacity_total=capacity_total,
                observed_capacity_used=observed_capacity_used,
                last_heartbeat_at=now,
                updated_at=now,
            )
        )
        if int(getattr(result, 'rowcount', 0) or 0) == 1:
            return True
        liveness = await self.session.get(InstanceManagerLivenessRow, manager_id)
        if liveness is not None:
            return False
        manager = await self.session.get(InstanceManagerRow, manager_id)
        if manager is None or manager.session_id != session_id:
            return False
        self.session.add(
            InstanceManagerLivenessRow(
                manager_id=manager_id,
                session_id=session_id,
                capacity_total=capacity_total,
                observed_capacity_used=observed_capacity_used,
                last_heartbeat_at=now,
                fenced_at=None,
                updated_at=now,
            )
        )
        await self.session.flush()
        return True

    async def update_local_snapshot(
        self,
        *,
        manager_id: str,
        session_id: str,
        local_instances: dict,
        now: datetime,
    ) -> bool:
        result = await self.session.execute(
            update(InstanceManagerRow)
            .where(
                InstanceManagerRow.manager_id == manager_id,
                InstanceManagerRow.session_id == session_id,
            )
            .values(local_instances_json=local_instances, updated_at=now)
        )
        return int(getattr(result, 'rowcount', 0) or 0) == 1

    async def set_draining(
        self,
        *,
        manager_id: str,
        session_id: str,
        draining: bool,
        now: datetime,
    ) -> bool:
        result = await self.session.execute(
            update(InstanceManagerRow)
            .where(
                InstanceManagerRow.manager_id == manager_id,
                InstanceManagerRow.session_id == session_id,
            )
            .values(draining=draining, updated_at=now)
        )
        return int(getattr(result, 'rowcount', 0) or 0) == 1

    async def ack_assignments(
        self,
        assignment_ids: list[str],
        manager_id: str,
        session_id: str,
        *,
        now: datetime,
    ) -> list[AssignmentRecord]:
        requested = list(dict.fromkeys(assignment_ids))
        if not requested:
            return []

        identities = (
            await self.session.execute(
                select(AssignmentRow.assignment_id, AssignmentRow.execution_id).where(
                    AssignmentRow.assignment_id.in_(requested)
                )
            )
        ).all()
        execution_ids = sorted({execution_id for _, execution_id in identities})
        executions = list(
            (
                await self.session.scalars(
                    select(ExecutionRow)
                    .options(load_only(*_ACK_EXECUTION_COLUMNS))
                    .where(ExecutionRow.execution_id.in_(execution_ids))
                    .order_by(ExecutionRow.execution_id)
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
            ).all()
        )
        assignments = list(
            (
                await self.session.scalars(
                    select(AssignmentRow)
                    .options(load_only(*_ACK_ASSIGNMENT_COLUMNS))
                    .where(AssignmentRow.assignment_id.in_(requested))
                    .order_by(AssignmentRow.execution_id, AssignmentRow.assignment_id)
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
            ).all()
        )
        executions_by_id = {execution.execution_id: execution for execution in executions}
        assignments_by_id = {assignment.assignment_id: assignment for assignment in assignments}
        acknowledged = []
        event_values: list[dict[str, Any]] = []
        for assignment_id in requested:
            assignment = assignments_by_id.get(assignment_id)
            if assignment is None:
                continue
            execution = executions_by_id.get(assignment.execution_id)
            if execution is None:
                continue
            if assignment.manager_id != manager_id or assignment.manager_session_id != session_id:
                continue
            if assignment.status != AssignmentStatus.PENDING:
                continue
            if execution.current_assignment_id != assignment.assignment_id:
                continue

            old_assignment_status = assignment.status
            old_execution_status = execution.status
            assignment.status = AssignmentStatus.ACTIVE
            assignment.accepted_at = now
            queue_committed_metric(
                self.session,
                'assignment_ack_distribution',
                max(0, (now - assignment.created_at).total_seconds()),
                environment=assignment.environment_id or 'all',
                outcome='success',
            )
            assignment.updated_at = now
            execution.status = ExecutionStatus.STARTING
            execution.version += 1
            execution.updated_at = now
            first_seq = execution.next_event_seq
            execution.next_event_seq += 2
            event_values.extend([
                {
                    'execution_id': execution.execution_id,
                    'seq': first_seq,
                    'assignment_id': assignment.assignment_id,
                    'attempt_id': assignment.attempt_id,
                    'type': 'assignment.status.changed',
                    'source': 'manager',
                    'visibility': 'public',
                    'payload_json': {'from': old_assignment_status, 'to': assignment.status},
                    'event_key': None,
                    'created_at': now,
                },
                {
                    'execution_id': execution.execution_id,
                    'seq': first_seq + 1,
                    'assignment_id': assignment.assignment_id,
                    'attempt_id': assignment.attempt_id,
                    'type': 'execution.status.changed',
                    'source': 'manager',
                    'visibility': 'public',
                    'payload_json': {'from': old_execution_status, 'to': execution.status},
                    'event_key': None,
                    'created_at': now,
                },
            ])
            record = assignment_record(assignment)
            if record is not None:
                acknowledged.append(record)

        if event_values:
            await self.session.execute(insert(ExecutionEventRow), event_values)
        await self.session.flush()
        return acknowledged

    async def ack_assignment(self, assignment_id: str, manager_id: str, session_id: str, *, now: datetime):
        assignment, execution = await self._locked_execution_assignment(assignment_id)
        if assignment.manager_id != manager_id or assignment.manager_session_id != session_id:
            raise FencingError('assignment_session_mismatch')
        if assignment.status != AssignmentStatus.PENDING:
            raise DomainConflictError('assignment_not_pending')
        if execution.current_assignment_id != assignment.assignment_id:
            raise FencingError('execution_assignment_mismatch')
        old_assignment_status = assignment.status
        old_execution_status = execution.status
        result = await self.session.execute(
            update(AssignmentRow)
            .where(
                AssignmentRow.assignment_id == assignment_id,
                AssignmentRow.manager_id == manager_id,
                AssignmentRow.manager_session_id == session_id,
                AssignmentRow.status == AssignmentStatus.PENDING,
            )
            .values(
                status=AssignmentStatus.ACTIVE,
                accepted_at=now,
                updated_at=now,
            )
        )
        if int(getattr(result, 'rowcount', 0) or 0) != 1:
            raise DomainConflictError('assignment_not_pending')
        assignment.status = AssignmentStatus.ACTIVE
        assignment.accepted_at = now
        queue_committed_metric(
            self.session,
            'assignment_ack_distribution',
            max(0, (now - assignment.created_at).total_seconds()),
            environment=assignment.environment_id or 'all',
            outcome='success',
        )
        assignment.updated_at = now
        execution.status = ExecutionStatus.STARTING
        execution.version += 1
        execution.updated_at = now
        await append_control_event_row(
            self.session,
            execution,
            event_type='assignment.status.changed',
            source='manager',
            payload={'from': old_assignment_status, 'to': assignment.status},
            assignment_id=assignment.assignment_id,
            attempt_id=assignment.attempt_id,
            now=now,
        )
        await append_control_event_row(
            self.session,
            execution,
            event_type='execution.status.changed',
            source='manager',
            payload={'from': old_execution_status, 'to': execution.status},
            assignment_id=assignment.assignment_id,
            attempt_id=assignment.attempt_id,
            now=now,
        )
        await self.session.flush()
        return assignment_record(assignment)

    async def mark_start_failed(
        self,
        assignment_id: str,
        manager_id: str,
        session_id: str,
        *,
        error: dict,
        now: datetime,
    ) -> None:
        marked = await self._mark_assignment_lost(
            assignment_id,
            manager_id,
            session_id,
            error=error,
            event_type='manager.start_failed',
            event_payload={'error': error},
            strict=True,
            now=now,
        )
        assert marked

    async def mark_assignment_lost(
        self,
        assignment_id: str,
        manager_id: str,
        session_id: str,
        *,
        reason: str,
        error: dict,
        now: datetime,
    ) -> bool:
        return await self._mark_assignment_lost(
            assignment_id,
            manager_id,
            session_id,
            error=error,
            event_type='manager.assignment_lost',
            event_payload={'error': error, 'reason': reason},
            strict=False,
            now=now,
        )

    async def _mark_assignment_lost(
        self,
        assignment_id: str,
        manager_id: str,
        session_id: str,
        *,
        error: dict,
        event_type: str,
        event_payload: dict,
        strict: bool,
        now: datetime,
    ) -> bool:
        assignment, execution = await self._locked_execution_assignment(assignment_id)
        if assignment.manager_id != manager_id or assignment.manager_session_id != session_id:
            raise FencingError('assignment_session_mismatch')
        if assignment.status != AssignmentStatus.ACTIVE:
            if strict:
                raise DomainConflictError('assignment_not_active')
            return False
        if execution.current_assignment_id != assignment.assignment_id:
            if strict:
                raise FencingError('execution_assignment_mismatch')
            return False

        old_assignment_status = assignment.status
        old_execution_status = execution.status
        old_route_status = execution.route_status
        assignment.status = AssignmentStatus.LOST
        assignment.endpoint_ready = False
        assignment.finished_at = now
        assignment.updated_at = now
        await self._cancel_unfinished_commands_for_assignment(assignment, execution, source='manager', now=now)

        retrying = execution.attempt_count < execution.max_attempts
        if retrying:
            execution.status = ExecutionStatus.QUEUED
            execution.queue_entered_at = now
            execution.version += 1
            execution.next_schedule_at = now
        else:
            execution.status = ExecutionStatus.LOST
            execution.version += 1
            execution.finished_at = now
            if execution.mode == 'live':
                execution.route_status = RouteStatus.CLOSED
        execution.error_json = error
        execution.updated_at = now

        await append_control_event_row(
            self.session,
            execution,
            event_type='assignment.status.changed',
            source='manager',
            payload={'from': old_assignment_status, 'to': assignment.status},
            assignment_id=assignment.assignment_id,
            attempt_id=assignment.attempt_id,
            now=now,
        )
        if old_execution_status != execution.status:
            await append_control_event_row(
                self.session,
                execution,
                event_type='execution.status.changed',
                source='manager',
                payload={'from': old_execution_status, 'to': execution.status},
                assignment_id=assignment.assignment_id,
                attempt_id=assignment.attempt_id,
                now=now,
            )
        if old_route_status != execution.route_status:
            await append_control_event_row(
                self.session,
                execution,
                event_type='route.status.changed',
                source='manager',
                payload={'from': old_route_status, 'to': execution.route_status},
                assignment_id=assignment.assignment_id,
                attempt_id=assignment.attempt_id,
                now=now,
            )
        await append_control_event_row(
            self.session,
            execution,
            event_type=event_type,
            source='manager',
            payload={**event_payload, 'retrying': retrying},
            assignment_id=assignment.assignment_id,
            attempt_id=assignment.attempt_id,
            now=now,
        )
        await self.session.flush()
        return True

    async def list_pending_assignments(self, manager_id: str, session_id: str, *, limit: int = 20):
        candidate_ids = list(
            await self.session.scalars(
                select(AssignmentRow.assignment_id)
                .join(ExecutionRow, ExecutionRow.execution_id == AssignmentRow.execution_id)
                .where(
                    AssignmentRow.manager_id == manager_id,
                    AssignmentRow.manager_session_id == session_id,
                    AssignmentRow.status == AssignmentStatus.PENDING,
                    ExecutionRow.current_assignment_id == AssignmentRow.assignment_id,
                    ExecutionRow.current_attempt_id == AssignmentRow.attempt_id,
                )
                .order_by(AssignmentRow.created_at, AssignmentRow.assignment_id)
                .limit(limit)
            )
        )
        if not candidate_ids:
            return []
        candidate_order = {assignment_id: index for index, assignment_id in enumerate(candidate_ids)}
        rows = (
            await self.session.execute(
                select(AssignmentRow, ExecutionRow)
                .join(ExecutionRow, ExecutionRow.execution_id == AssignmentRow.execution_id)
                .where(
                    AssignmentRow.assignment_id.in_(candidate_ids),
                    AssignmentRow.manager_id == manager_id,
                    AssignmentRow.manager_session_id == session_id,
                    AssignmentRow.status == AssignmentStatus.PENDING,
                    ExecutionRow.current_assignment_id == AssignmentRow.assignment_id,
                    ExecutionRow.current_attempt_id == AssignmentRow.attempt_id,
                )
            )
        ).all()
        rows = list(rows)
        rows.sort(key=lambda item: candidate_order[item[0].assignment_id])
        records = []
        for assignment, execution in rows:
            execution_value = execution_record(execution, assignment)
            assignment_value = execution_value.current_assignment
            records.append((
                execution_value,
                assignment_value,
                dict(assignment.trace_carrier_json) if assignment.trace_carrier_json else None,
            ))
        return records

    async def mark_running(
        self,
        assignment_id: str,
        manager_id: str,
        session_id: str,
        *,
        endpoint_url: str | None,
        endpoint_ready: bool,
        runtime_info: dict,
        now: datetime,
    ) -> None:
        assignment, execution = await self._active_assignment(assignment_id, manager_id, session_id)
        old_execution_status = execution.status
        assignment.endpoint_url = endpoint_url
        assignment.endpoint_ready = endpoint_ready
        if runtime_info:
            assignment.runtime_info_json = await self._merge_assignment_runtime_info(
                assignment, execution, runtime_info
            )
        assignment.endpoint_updated_at = now
        assignment.last_instance_update_at = now
        assignment.started_at = assignment.started_at or now
        assignment.updated_at = now
        execution.status = ExecutionStatus.RUNNING
        execution.version += 1
        execution.started_at = execution.started_at or now
        execution.execution_expires_at = execution.execution_expires_at or deadline_after(
            execution.started_at,
            execution.execution_ttl_seconds,
        )
        execution.updated_at = now
        if old_execution_status != execution.status:
            await append_control_event_row(
                self.session,
                execution,
                event_type='execution.status.changed',
                source='manager',
                payload={'from': old_execution_status, 'to': execution.status},
                assignment_id=assignment.assignment_id,
                attempt_id=assignment.attempt_id,
                now=now,
            )
        await self.session.flush()

    async def mark_waiting_input(
        self,
        assignment_id: str,
        manager_id: str,
        session_id: str,
        *,
        payload: dict,
        now: datetime,
    ) -> None:
        assignment, execution = await self._active_assignment(assignment_id, manager_id, session_id)
        old_execution_status = execution.status
        assignment.last_instance_update_at = now
        assignment.runtime_info_json = merge_runtime_info(assignment.runtime_info_json, {'input': payload})
        assignment.updated_at = now
        execution.status = ExecutionStatus.WAITING_INPUT
        execution.version += 1
        execution.updated_at = now
        if old_execution_status != execution.status:
            await append_control_event_row(
                self.session,
                execution,
                event_type='execution.status.changed',
                source='manager',
                payload={'from': old_execution_status, 'to': execution.status},
                assignment_id=assignment.assignment_id,
                attempt_id=assignment.attempt_id,
                now=now,
            )
        await self.session.flush()

    async def update_runtime_info(
        self,
        assignment_id: str,
        manager_id: str,
        session_id: str,
        *,
        runtime_info: dict,
        now: datetime,
    ) -> None:
        assignment, execution = await self._active_assignment(
            assignment_id, manager_id, session_id, allow_stopping=True
        )
        if runtime_info:
            assignment.runtime_info_json = await self._merge_assignment_runtime_info(
                assignment, execution, runtime_info
            )
        assignment.last_instance_update_at = now
        assignment.updated_at = now
        await self.session.flush()

    async def complete_execution(
        self,
        assignment_id: str,
        manager_id: str,
        session_id: str,
        *,
        result: dict,
        runtime_info: dict | None = None,
        now: datetime,
    ) -> None:
        assignment, execution = await self._active_assignment(assignment_id, manager_id, session_id)
        await self._require_ready_result_artifact(execution, assignment)
        old_assignment_status = assignment.status
        old_execution_status = execution.status
        old_route_status = execution.route_status
        assignment.status = AssignmentStatus.RELEASED
        assignment.finished_at = now
        assignment.updated_at = now
        assignment.endpoint_ready = False
        if runtime_info:
            assignment.runtime_info_json = await self._merge_assignment_runtime_info(
                assignment, execution, runtime_info
            )
            assignment.last_instance_update_at = now
        execution.status = ExecutionStatus.SUCCEEDED
        execution.version += 1
        execution.result_json = result
        execution.finished_at = now
        execution.updated_at = now
        if execution.mode == 'live':
            execution.route_status = 'closed'
        await self._cancel_unfinished_commands_for_assignment(assignment, execution, source='manager', now=now)
        await self._append_terminal_events(
            execution,
            assignment,
            old_assignment_status=old_assignment_status,
            old_execution_status=old_execution_status,
            old_route_status=old_route_status,
            now=now,
        )
        await self.session.flush()

    async def _require_ready_result_artifact(
        self,
        execution: ExecutionRow,
        assignment: AssignmentRow,
    ) -> None:
        if execution.environment_id is None or execution.spec_type not in {'nexus_single', 'nexus_live'}:
            return
        artifact = (
            await self.session.scalar(
                select(ArtifactRow).where(ArtifactRow.artifact_id == execution.result_artifact_id).with_for_update()
            )
            if execution.result_artifact_id
            else None
        )
        expected_producer_key = f'run-result:{execution.execution_id}:{assignment.attempt_id}'
        if (
            artifact is None
            or artifact.status != 'ready'
            or artifact.kind != 'run_result'
            or artifact.owner_principal_id != execution.owner_principal_id
            or artifact.environment_id != execution.environment_id
            or artifact.producer_key != expected_producer_key
        ):
            raise TemporaryUnavailableError('execution_result_artifact_not_ready')

    async def fail_execution(
        self,
        assignment_id: str,
        manager_id: str,
        session_id: str,
        *,
        error: dict,
        result: dict | None = None,
        runtime_info: dict | None = None,
        now: datetime,
    ) -> None:
        assignment, execution = await self._active_assignment(assignment_id, manager_id, session_id)
        old_assignment_status = assignment.status
        old_execution_status = execution.status
        old_route_status = execution.route_status
        assignment.status = AssignmentStatus.RELEASED
        assignment.finished_at = now
        assignment.updated_at = now
        assignment.endpoint_ready = False
        if runtime_info:
            assignment.runtime_info_json = await self._merge_assignment_runtime_info(
                assignment, execution, runtime_info
            )
            assignment.last_instance_update_at = now
        execution.status = ExecutionStatus.FAILED
        execution.version += 1
        execution.error_json = error
        execution.result_json = result
        execution.finished_at = now
        execution.updated_at = now
        if execution.mode == 'live':
            execution.route_status = 'closed'
        await self._cancel_unfinished_commands_for_assignment(assignment, execution, source='manager', now=now)
        await self._append_terminal_events(
            execution,
            assignment,
            old_assignment_status=old_assignment_status,
            old_execution_status=old_execution_status,
            old_route_status=old_route_status,
            now=now,
        )
        await self.session.flush()

    async def mark_killed(self, assignment_id: str, manager_id: str, session_id: str, *, now: datetime) -> None:
        assignment, execution = await self._active_assignment(
            assignment_id, manager_id, session_id, allow_stopping=True
        )
        old_assignment_status = assignment.status
        old_execution_status = execution.status
        old_route_status = execution.route_status
        assignment.status = AssignmentStatus.RELEASED
        assignment.finished_at = now
        assignment.updated_at = now
        assignment.endpoint_ready = False
        execution.status = (
            ExecutionStatus.EXPIRED
            if execution.stop_reason in {'execution_ttl', 'parent_ttl'}
            else ExecutionStatus.KILLED
        )
        execution.version += 1
        execution.finished_at = now
        execution.updated_at = now
        if execution.mode == 'live':
            execution.route_status = 'closed'
        await self._cancel_unfinished_commands_for_assignment(assignment, execution, source='manager', now=now)
        await self._append_terminal_events(
            execution,
            assignment,
            old_assignment_status=old_assignment_status,
            old_execution_status=old_execution_status,
            old_route_status=old_route_status,
            now=now,
        )
        await self.session.flush()

    async def mark_killed_if_active(
        self, assignment_id: str, manager_id: str, session_id: str, *, now: datetime
    ) -> bool:
        assignment = await self.session.get(AssignmentRow, assignment_id)
        if assignment is None:
            raise ResourceNotFoundError('assignment_not_found')
        if assignment.manager_id != manager_id or assignment.manager_session_id != session_id:
            raise FencingError('assignment_session_mismatch')
        if assignment.status != AssignmentStatus.ACTIVE:
            return False
        await self.mark_killed(assignment_id, manager_id, session_id, now=now)
        return True

    async def release_deferred_stop_if_active(
        self, assignment_id: str, manager_id: str, session_id: str, *, now: datetime
    ) -> bool:
        assignment = await self.session.get(AssignmentRow, assignment_id)
        if assignment is None:
            raise ResourceNotFoundError('assignment_not_found')
        if assignment.manager_id != manager_id or assignment.manager_session_id != session_id:
            raise FencingError('assignment_session_mismatch')
        if assignment.status != AssignmentStatus.ACTIVE:
            return False

        assignment, execution = await self._active_assignment(
            assignment_id,
            manager_id,
            session_id,
            allow_stopping=True,
        )
        if execution.status != ExecutionStatus.STOPPING:
            return False

        old_assignment_status = assignment.status
        old_execution_status = execution.status
        old_route_status = execution.route_status
        assignment.status = AssignmentStatus.RELEASED
        assignment.finished_at = now
        assignment.updated_at = now
        assignment.endpoint_ready = False
        execution.version += 1
        execution.updated_at = now
        await self._append_terminal_events(
            execution,
            assignment,
            old_assignment_status=old_assignment_status,
            old_execution_status=old_execution_status,
            old_route_status=old_route_status,
            now=now,
        )
        await self.session.flush()
        return True

    async def claim_commands(self, manager_id: str, session_id: str, *, limit: int = 20, now: datetime | None = None):
        candidate_ids = list(
            await self.session.scalars(
                select(ExecutionCommandRow.command_id)
                .join(AssignmentRow, ExecutionCommandRow.assignment_id == AssignmentRow.assignment_id)
                .join(ExecutionRow, ExecutionCommandRow.execution_id == ExecutionRow.execution_id)
                .where(
                    AssignmentRow.manager_id == manager_id,
                    AssignmentRow.manager_session_id == session_id,
                    AssignmentRow.status == AssignmentStatus.ACTIVE,
                    ExecutionCommandRow.status == CommandStatus.PENDING,
                )
                .order_by(ExecutionCommandRow.created_at, ExecutionCommandRow.command_id)
                .limit(limit)
            )
        )
        if not candidate_ids:
            return []
        candidate_order = {command_id: index for index, command_id in enumerate(candidate_ids)}
        rows = (
            await self.session.execute(
                select(ExecutionCommandRow, AssignmentRow, ExecutionRow)
                .options(
                    load_only(*_COMMAND_COLUMNS),
                    load_only(*_COMMAND_ASSIGNMENT_COLUMNS),
                    load_only(*_COMMAND_EXECUTION_COLUMNS),
                )
                .join(AssignmentRow, ExecutionCommandRow.assignment_id == AssignmentRow.assignment_id)
                .join(ExecutionRow, ExecutionCommandRow.execution_id == ExecutionRow.execution_id)
                .where(
                    ExecutionCommandRow.command_id.in_(candidate_ids),
                    AssignmentRow.manager_id == manager_id,
                    AssignmentRow.manager_session_id == session_id,
                    AssignmentRow.status == AssignmentStatus.ACTIVE,
                    ExecutionCommandRow.status == CommandStatus.PENDING,
                )
            )
        ).all()
        rows = list(rows)
        rows.sort(key=lambda item: candidate_order[item[0].command_id])
        claimed_at = now or utcnow()
        claimed = []
        for row, assignment, execution in rows:
            if not self._command_matches_joined_current_assignment(row, assignment, execution):
                old_status = row.status
                result = await self.session.execute(
                    update(ExecutionCommandRow)
                    .where(
                        ExecutionCommandRow.command_id == row.command_id,
                        ExecutionCommandRow.status == CommandStatus.PENDING,
                    )
                    .values(
                        status=CommandStatus.CANCELLED,
                        completed_at=claimed_at,
                        updated_at=claimed_at,
                    )
                )
                if int(getattr(result, 'rowcount', 0) or 0) != 1:
                    continue
                row.status = CommandStatus.CANCELLED
                row.completed_at = claimed_at
                row.updated_at = claimed_at
                await append_control_event_row(
                    self.session,
                    execution,
                    event_type='command.status.changed',
                    source='manager',
                    payload={'command_id': row.command_id, 'from': old_status, 'to': row.status},
                    assignment_id=row.assignment_id,
                    attempt_id=row.attempt_id,
                    now=claimed_at,
                )
                continue
            old_status = row.status
            result = await self.session.execute(
                update(ExecutionCommandRow)
                .where(
                    ExecutionCommandRow.command_id == row.command_id,
                    ExecutionCommandRow.status == CommandStatus.PENDING,
                )
                .values(
                    status=CommandStatus.CLAIMED,
                    claimed_by_manager_id=manager_id,
                    claimed_at=claimed_at,
                    updated_at=claimed_at,
                )
            )
            if int(getattr(result, 'rowcount', 0) or 0) != 1:
                continue
            row.status = CommandStatus.CLAIMED
            row.claimed_by_manager_id = manager_id
            row.claimed_at = claimed_at
            row.updated_at = claimed_at
            await append_control_event_row(
                self.session,
                execution,
                event_type='command.status.changed',
                source='manager',
                payload={'command_id': row.command_id, 'from': old_status, 'to': row.status},
                assignment_id=row.assignment_id,
                attempt_id=row.attempt_id,
                now=claimed_at,
            )
            claimed.append(row)
        await self.session.flush()
        return [command_record(row) for row in claimed]

    async def complete_command(
        self, command_id: str, *, status: str = CommandStatus.DONE, error: dict | None = None, now: datetime
    ) -> None:
        row = await self.session.get(ExecutionCommandRow, command_id)
        if row is None:
            raise ResourceNotFoundError('command_not_found')
        old_status = row.status
        row.status = status
        row.error_json = error
        row.completed_at = now
        row.updated_at = now
        execution = await self.session.get(ExecutionRow, row.execution_id)
        if execution is not None and old_status != row.status:
            await append_control_event_row(
                self.session,
                execution,
                event_type='command.status.changed',
                source='manager',
                payload={'command_id': row.command_id, 'from': old_status, 'to': row.status},
                assignment_id=row.assignment_id,
                attempt_id=row.attempt_id,
                now=now,
            )
        await self.session.flush()

    async def _merge_assignment_runtime_info(
        self,
        assignment: AssignmentRow,
        execution: ExecutionRow,
        runtime_info: dict,
    ) -> dict:
        current = source_binding(assignment.runtime_info_json)
        incoming = source_binding(runtime_info)
        if current is not None or incoming is not None:
            if execution.spec_type != 'nexus_single':
                raise DomainConflictError('trace_source_binding_type_mismatch')
            if execution.current_attempt_id != assignment.attempt_id:
                raise FencingError('execution_attempt_mismatch')
            manager = await self.session.scalar(
                select(InstanceManagerRow)
                .where(InstanceManagerRow.manager_id == assignment.manager_id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            if manager is None or manager.session_id != assignment.manager_session_id:
                raise FencingError('manager_session_mismatch')
            liveness = await self.session.get(InstanceManagerLivenessRow, assignment.manager_id)
            if liveness is not None and (
                liveness.session_id != assignment.manager_session_id or liveness.fenced_at is not None
            ):
                raise FencingError('manager_session_fenced')
            validate_source_binding_update(
                assignment.runtime_info_json,
                runtime_info,
                execution_id=execution.execution_id,
                assignment_id=assignment.assignment_id,
                attempt_id=assignment.attempt_id,
                owner_principal_id=execution.owner_principal_id,
            )
        return merge_runtime_info(assignment.runtime_info_json, runtime_info)

    async def _active_assignment(
        self,
        assignment_id: str,
        manager_id: str,
        session_id: str,
        *,
        allow_stopping: bool = False,
    ) -> tuple[AssignmentRow, ExecutionRow]:
        assignment, execution = await self._locked_execution_assignment(assignment_id)
        if assignment.manager_id != manager_id or assignment.manager_session_id != session_id:
            raise FencingError('assignment_session_mismatch')
        if assignment.status != AssignmentStatus.ACTIVE:
            raise DomainConflictError('assignment_not_active')
        if execution.current_assignment_id != assignment.assignment_id:
            raise FencingError('execution_assignment_mismatch')
        if not allow_stopping and execution.status not in (
            ExecutionStatus.STARTING,
            ExecutionStatus.RUNNING,
            ExecutionStatus.WAITING_INPUT,
        ):
            raise DomainConflictError('execution_not_active', detail=str(execution.status))
        return assignment, execution

    async def _locked_execution_assignment(self, assignment_id: str) -> tuple[AssignmentRow, ExecutionRow]:
        assignment_identity = await self.session.get(AssignmentRow, assignment_id)
        if assignment_identity is None:
            raise ResourceNotFoundError('assignment_not_found')
        execution = await self.session.scalar(
            select(ExecutionRow)
            .where(ExecutionRow.execution_id == assignment_identity.execution_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if execution is None:
            raise FencingError('execution_assignment_mismatch')
        assignment = await self.session.scalar(
            select(AssignmentRow)
            .where(AssignmentRow.assignment_id == assignment_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if assignment is None:
            raise ResourceNotFoundError('assignment_not_found')
        if assignment.execution_id != execution.execution_id:
            raise FencingError('execution_assignment_mismatch')
        return assignment, execution

    async def _command_matches_current_assignment(
        self,
        command: ExecutionCommandRow,
        manager_id: str,
        session_id: str,
    ) -> bool:
        if command.assignment_id is None or command.attempt_id is None:
            return False
        assignment = await self.session.get(AssignmentRow, command.assignment_id)
        if assignment is None:
            return False
        if assignment.manager_id != manager_id or assignment.manager_session_id != session_id:
            return False
        if assignment.status != AssignmentStatus.ACTIVE:
            return False
        execution = await self.session.get(ExecutionRow, command.execution_id)
        if execution is None:
            return False
        return (
            execution.current_assignment_id == assignment.assignment_id
            and execution.current_attempt_id == assignment.attempt_id
            and command.attempt_id == assignment.attempt_id
        )

    @staticmethod
    def _command_matches_joined_current_assignment(
        command: ExecutionCommandRow,
        assignment: AssignmentRow,
        execution: ExecutionRow,
    ) -> bool:
        return (
            command.assignment_id == assignment.assignment_id
            and assignment.status == AssignmentStatus.ACTIVE
            and execution.current_assignment_id == assignment.assignment_id
            and execution.current_attempt_id == assignment.attempt_id
            and command.attempt_id == assignment.attempt_id
        )

    async def _append_terminal_events(
        self,
        execution: ExecutionRow,
        assignment: AssignmentRow,
        *,
        old_assignment_status: str,
        old_execution_status: str,
        old_route_status: str,
        now: datetime,
    ) -> None:
        if old_assignment_status != assignment.status:
            await append_control_event_row(
                self.session,
                execution,
                event_type='assignment.status.changed',
                source='manager',
                payload={'from': old_assignment_status, 'to': assignment.status},
                assignment_id=assignment.assignment_id,
                attempt_id=assignment.attempt_id,
                now=now,
            )
        if old_execution_status != execution.status:
            await append_control_event_row(
                self.session,
                execution,
                event_type='execution.status.changed',
                source='manager',
                payload={'from': old_execution_status, 'to': execution.status},
                assignment_id=assignment.assignment_id,
                attempt_id=assignment.attempt_id,
                now=now,
            )
        if old_route_status != execution.route_status:
            await append_control_event_row(
                self.session,
                execution,
                event_type='route.status.changed',
                source='manager',
                payload={'from': old_route_status, 'to': execution.route_status},
                assignment_id=assignment.assignment_id,
                attempt_id=assignment.attempt_id,
                now=now,
            )

    async def _cancel_unfinished_commands_for_assignment(
        self,
        assignment: AssignmentRow,
        execution: ExecutionRow,
        *,
        source: str,
        now: datetime,
    ) -> int:
        rows = (
            await self.session.scalars(
                select(ExecutionCommandRow).where(
                    ExecutionCommandRow.assignment_id == assignment.assignment_id,
                    ExecutionCommandRow.status.in_([CommandStatus.PENDING, CommandStatus.CLAIMED]),
                )
            )
        ).all()
        changed = 0
        for row in rows:
            old_status = row.status
            row.status = CommandStatus.CANCELLED
            row.completed_at = now
            row.updated_at = now
            await append_control_event_row(
                self.session,
                execution,
                event_type='command.status.changed',
                source=source,
                payload={'command_id': row.command_id, 'from': old_status, 'to': row.status},
                assignment_id=row.assignment_id,
                attempt_id=row.attempt_id,
                now=now,
            )
            changed += 1
        return changed
