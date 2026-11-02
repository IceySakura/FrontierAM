from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import inspect, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.distributed.errors import ResourceNotFoundError
from nexus.distributed.observability.database import queue_committed_metric
from nexus.distributed.persistence.orm import ExecutionEventRow, ExecutionRow


async def append_control_event_row(
    session: AsyncSession,
    execution: ExecutionRow,
    *,
    event_type: str,
    source: str,
    payload: dict[str, Any],
    now: datetime,
    visibility: str = 'public',
    assignment_id: str | None = None,
    attempt_id: str | None = None,
    event_key: str | None = None,
) -> ExecutionEventRow:
    if event_key is not None:
        existing = await session.scalar(
            select(ExecutionEventRow).where(
                ExecutionEventRow.execution_id == execution.execution_id,
                ExecutionEventRow.event_key == event_key,
            )
        )
        if existing is not None:
            return existing
    await session.execute(
        update(ExecutionRow)
        .where(ExecutionRow.execution_id == execution.execution_id)
        .values(next_event_seq=ExecutionRow.next_event_seq + 1)
    )
    next_event_seq = await session.scalar(
        select(ExecutionRow.next_event_seq).where(ExecutionRow.execution_id == execution.execution_id)
    )
    if next_event_seq is None:
        raise ResourceNotFoundError('execution_not_found')
    seq = int(next_event_seq) - 1
    execution.next_event_seq = int(next_event_seq)

    event = ExecutionEventRow(
        execution_id=execution.execution_id,
        seq=seq,
        assignment_id=assignment_id,
        attempt_id=attempt_id,
        type=event_type,
        source=source,
        visibility=visibility,
        payload_json=payload,
        event_key=event_key,
        created_at=now,
    )
    session.add(event)
    await session.flush()
    queue_terminal_event_metrics(session, execution, event_type=event_type, payload=payload)
    return event


def queue_terminal_event_metrics(
    session: AsyncSession, execution: ExecutionRow, *, event_type: str, payload: dict
) -> None:
    status = payload.get('to')
    if (
        event_type == 'execution.status.changed'
        and payload.get('from') != status
        and status in {'succeeded', 'failed', 'killed', 'lost', 'expired', 'cancelled'}
    ):
        # Only use already loaded state: event creation must not add a lazy SQL read.
        state = inspect(execution).dict
        error = state.get('error_json') or {}
        error_type = error.get('type') if isinstance(error, dict) else None
        category = 'unknown'
        if status == 'succeeded':
            category = 'success'
        elif status in {'killed', 'cancelled'}:
            category = 'cancelled'
        elif status == 'lost' or error_type in {
            'AssignmentStartFailed',
            'ResultPersistenceError',
            'ResultPersistenceFailed',
            'RuntimeUnhealthyError',
            'TraceArtifactUnavailable',
            'RunTraceSinkError',
            'UnhandledDriverError',
            'EnvironmentProvisioningTimeoutError',
            'EnvironmentProvisioningFailed',
        }:
            category = 'platform'
        elif error_type in {'InvalidExecutionSpec', 'InvalidDriverConfig'}:
            category = 'task'
        queue_committed_metric(
            session,
            'platform_terminal_results',
            status=status,
            category=category,
            environment=state.get('environment_id') or 'all',
        )
        labels = {'status': status, 'category': category, 'environment': state.get('environment_id') or 'all'}
        queue_committed_metric(session, 'platform_terminal_failures', int(category == 'platform'), **labels)
        queue_committed_metric(session, 'platform_terminal_unknown', int(category == 'unknown'), **labels)
