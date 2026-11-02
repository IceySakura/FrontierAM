from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.distributed.errors import (
    DomainConflictError,
    ResourceGoneError,
    ResourceNotFoundError,
    TemporaryUnavailableError,
)
from nexus.distributed.execution.models import (
    TERMINAL_EXECUTION_STATUSES,
    AssignmentStatus,
    CommandStatus,
    ExecutionStatus,
    RouteStatus,
)
from nexus.distributed.instance_manager.liveness import effective_manager_liveness
from nexus.distributed.persistence.orm import (
    AssignmentRow,
    ExecutionCommandRow,
    ExecutionRow,
    InstanceManagerLivenessRow,
    InstanceManagerRow,
)
from nexus.distributed.repositories._records import command_record
from nexus.distributed.repositories.events import append_control_event_row
from nexus.distributed.utils.identity import new_id


class GatewayRepository:
    def __init__(self, session: AsyncSession, *, manager_healthy_ttl: timedelta) -> None:
        self.session = session
        self.manager_healthy_ttl = manager_healthy_ttl

    async def submit_command(
        self,
        execution_id: str,
        *,
        command_type: str,
        payload: dict,
        idempotency_key: str | None,
        now: datetime,
        created_by: str | None = None,
    ):
        execution = await self.session.get(ExecutionRow, execution_id)
        if execution is None:
            raise ResourceNotFoundError('execution_not_found')
        assignment = await self._active_assignment_for_command(execution, now)
        if idempotency_key:
            existing = await self.session.scalar(
                select(ExecutionCommandRow).where(
                    ExecutionCommandRow.execution_id == execution_id,
                    ExecutionCommandRow.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                if existing.assignment_id != assignment.assignment_id or existing.attempt_id != assignment.attempt_id:
                    raise DomainConflictError('idempotency_key_conflict')
                return command_record(existing)
        command = ExecutionCommandRow(
            command_id=new_id('cmd'),
            execution_id=execution.execution_id,
            assignment_id=assignment.assignment_id,
            attempt_id=assignment.attempt_id,
            type=command_type,
            status=CommandStatus.PENDING,
            payload_json=payload,
            idempotency_key=idempotency_key,
            created_by=created_by,
            created_at=now,
            updated_at=now,
        )
        self.session.add(command)
        await self.session.flush()
        await append_control_event_row(
            self.session,
            execution,
            event_type='command.status.changed',
            source='gateway',
            payload={'command_id': command.command_id, 'from': None, 'to': CommandStatus.PENDING},
            assignment_id=assignment.assignment_id,
            attempt_id=assignment.attempt_id,
            now=now,
        )
        return command_record(command)

    async def resolve_route(
        self, execution_id: str, *, now: datetime
    ) -> tuple[ExecutionRow, AssignmentRow, InstanceManagerRow]:
        execution = await self.session.get(ExecutionRow, execution_id)
        if execution is None:
            raise ResourceNotFoundError('execution_not_found')
        assignment = await self._active_assignment_for_route(execution, now)
        manager = await self.session.get(InstanceManagerRow, assignment.manager_id)
        if manager is None:
            raise TemporaryUnavailableError('assignment_unavailable')
        return execution, assignment, manager

    async def _active_assignment_for_command(self, execution: ExecutionRow, now: datetime) -> AssignmentRow:
        if execution.status in TERMINAL_EXECUTION_STATUSES or execution.route_status == RouteStatus.CLOSED:
            raise ResourceGoneError('execution_closed')
        if execution.route_status != RouteStatus.ACTIVE:
            raise DomainConflictError('route_not_active')
        if execution.status != ExecutionStatus.WAITING_INPUT:
            raise DomainConflictError('execution_not_ready')
        return await self._active_assignment_for_route(execution, now)

    async def _active_assignment_for_route(self, execution: ExecutionRow, now: datetime) -> AssignmentRow:
        if execution.status in TERMINAL_EXECUTION_STATUSES or execution.route_status == RouteStatus.CLOSED:
            raise ResourceGoneError('execution_closed')
        if execution.route_status != RouteStatus.ACTIVE:
            raise DomainConflictError('route_not_active')
        if execution.current_assignment_id is None:
            raise TemporaryUnavailableError('assignment_unavailable')
        assignment = await self.session.get(AssignmentRow, execution.current_assignment_id)
        if assignment is None or assignment.status != AssignmentStatus.ACTIVE:
            raise TemporaryUnavailableError('assignment_unavailable')
        manager = await self.session.get(InstanceManagerRow, assignment.manager_id)
        if manager is None or manager.session_id != assignment.manager_session_id:
            raise TemporaryUnavailableError('assignment_unavailable')
        liveness = await self.session.get(InstanceManagerLivenessRow, manager.manager_id)
        effective = effective_manager_liveness(manager, liveness)
        if not effective.active or now - effective.last_heartbeat_at > self.manager_healthy_ttl:
            raise TemporaryUnavailableError('executor_unavailable')
        if not assignment.endpoint_ready or not assignment.endpoint_url:
            raise TemporaryUnavailableError('endpoint_not_ready')
        return assignment
