from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import and_, false, func, insert, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import load_only

from nexus.distributed.analytics.fields import task_name_from_spec
from nexus.distributed.environment.models import (
    DEFAULT_ENVIRONMENT_ID,
    DEFAULT_POOL_ID,
)
from nexus.distributed.errors import (
    DomainConflictError,
    DomainValidationError,
    ResourceGoneError,
    ResourceNotFoundError,
    TemporaryUnavailableError,
)
from nexus.distributed.execution.models import (
    TERMINAL_EXECUTION_STATUSES,
    AssignmentRecord,
    AssignmentStatus,
    CommandStatus,
    ExecutionMode,
    ExecutionRecord,
    ExecutionStatus,
    RouteStatus,
)
from nexus.distributed.persistence.db import TransactionRetryError
from nexus.distributed.persistence.orm import AssignmentRow, ExecutionCommandRow, ExecutionEventRow, ExecutionRow
from nexus.distributed.repositories._records import command_record, event_record, execution_record
from nexus.distributed.repositories.events import append_control_event_row, queue_terminal_event_metrics
from nexus.distributed.repositories.pagination import decode_cursor, encode_cursor
from nexus.distributed.utils.clock import deadline_after, stop_deadline_after
from nexus.distributed.utils.identity import (
    canonical_sha256,
    new_id,
    stable_shard,
    validate_environment_id,
    validate_pool_id,
)
from nexus.utils.otel.propagation import inject_carrier


def stop_terminal_status(stop_reason: str | None) -> str:
    return ExecutionStatus.EXPIRED if stop_reason == 'parent_ttl' else ExecutionStatus.KILLED


_STOPPABLE_EXECUTION_STATUSES = frozenset({
    ExecutionStatus.QUEUED,
    ExecutionStatus.ASSIGNED,
    ExecutionStatus.STARTING,
    ExecutionStatus.RUNNING,
    ExecutionStatus.WAITING_INPUT,
    ExecutionStatus.STOPPING,
    ExecutionStatus.KILLED,
    ExecutionStatus.EXPIRED,
})

_STOP_EXECUTION_COLUMNS = (
    ExecutionRow.execution_id,
    ExecutionRow.environment_id,
    ExecutionRow.mode,
    ExecutionRow.status,
    ExecutionRow.route_status,
    ExecutionRow.current_assignment_id,
    ExecutionRow.current_attempt_id,
    ExecutionRow.next_event_seq,
    ExecutionRow.stop_reason,
    ExecutionRow.stop_timeout_seconds,
    ExecutionRow.stop_deadline_at,
    ExecutionRow.version,
    ExecutionRow.updated_at,
    ExecutionRow.finished_at,
)

_STOP_ASSIGNMENT_COLUMNS = (
    AssignmentRow.assignment_id,
    AssignmentRow.execution_id,
    AssignmentRow.attempt_id,
    AssignmentRow.status,
    AssignmentRow.endpoint_ready,
    AssignmentRow.updated_at,
    AssignmentRow.finished_at,
)

_STOP_COMMAND_COLUMNS = (
    ExecutionCommandRow.command_id,
    ExecutionCommandRow.execution_id,
    ExecutionCommandRow.assignment_id,
    ExecutionCommandRow.attempt_id,
    ExecutionCommandRow.type,
    ExecutionCommandRow.status,
    ExecutionCommandRow.completed_at,
    ExecutionCommandRow.created_at,
    ExecutionCommandRow.updated_at,
)
_COMMAND_NOT_LOADED = object()

_SUMMARY_EXECUTION_FIELDS = (
    'execution_id',
    'shard_id',
    'mode',
    'spec_type',
    'instance_mode',
    'status',
    'route_status',
    'slot_cost',
    'current_assignment_id',
    'current_attempt_id',
    'attempt_count',
    'max_attempts',
    'next_schedule_at',
    'created_at',
    'updated_at',
    'owner_principal_id',
    'owner_tenant',
    'created_by',
    'parent_execution_id',
    'environment_id',
    'pool_id',
    'allocation_id',
    'workload',
    'result_artifact_id',
    'first_schedulable_at',
    'stop_reason',
    'queue_expires_at',
    'execution_expires_at',
    'stop_deadline_at',
    'cleanup_done_at',
    'stop_timeout_seconds',
    'version',
    'spec_sha256',
    'submission_sha256',
    'allocation_spec_hash',
)
_SUMMARY_ASSIGNMENT_FIELDS = (
    'assignment_id',
    'execution_id',
    'attempt_id',
    'attempt_seq',
    'manager_id',
    'manager_session_id',
    'status',
    'slot_cost',
    'created_at',
    'updated_at',
    'endpoint_url',
    'endpoint_ready',
    'owner_principal_id',
    'environment_id',
    'pool_id',
    'allocation_id',
    'workload',
    'allocation_spec_hash',
)


def _validate_stop_reason(stop_reason: str) -> None:
    if (
        not stop_reason
        or len(stop_reason) > 32
        or not stop_reason.isascii()
        or any(character != '_' and not character.isalnum() for character in stop_reason)
    ):
        raise DomainValidationError('invalid_stop_reason')


def _decode_execution_cursor(value: str) -> tuple[datetime, datetime, str]:
    payload = decode_cursor(value, fields=frozenset({'updated_at', 'created_at', 'execution_id'}))
    try:
        updated_at = datetime.fromisoformat(payload['updated_at'])
        created_at = datetime.fromisoformat(payload['created_at'])
    except ValueError as exc:
        raise DomainValidationError('invalid_cursor') from exc
    if updated_at.tzinfo is not None or created_at.tzinfo is not None:
        raise DomainValidationError('invalid_cursor')
    return updated_at, created_at, payload['execution_id']


@dataclass(slots=True)
class CreateExecution:
    execution_id: str
    mode: str
    spec_type: str
    instance_mode: str
    spec: dict[str, Any]
    driver_config: dict[str, Any] = field(default_factory=dict)
    placement: dict[str, Any] = field(default_factory=dict)
    route_status: str = RouteStatus.NONE
    slot_cost: int = 1
    max_attempts: int = 1
    priority: int = 0
    queue_ttl_seconds: int | None = None
    execution_ttl_seconds: int | None = None
    stop_timeout_seconds: int | None = 60
    trace_carrier: dict[str, str] | None = field(default_factory=lambda: inject_carrier() or None)
    idempotency_scope: str | None = None
    idempotency_key: str | None = None
    owner_principal_id: str = 'legacy'
    owner_tenant: str | None = None
    created_by: str = 'system:legacy'
    parent_execution_id: str | None = None
    environment_id: str | None = None
    pool_id: str | None = None
    allocation_id: str | None = None
    workload: str | None = None
    spec_sha256: str | None = None
    submission_sha256: str | None = None
    allocation_spec_hash: str | None = None
    first_schedulable_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ExecutionLogContext:
    execution: ExecutionRecord
    started_at: datetime | None
    finished_at: datetime | None


class ExecutionRepository:
    def __init__(self, session: AsyncSession, *, shard_count: int) -> None:
        self.session = session
        self.shard_count = shard_count

    async def create_execution(
        self,
        request: CreateExecution,
        *,
        now: datetime,
        reuse_existing: bool = True,
    ) -> ExecutionRecord:
        _validate_environment_scope(request)
        placement = self._normalize_placement(request.placement)
        spec_sha256 = request.spec_sha256 or canonical_sha256(request.spec)
        if request.idempotency_scope is not None and request.idempotency_key is not None:
            existing = await self.session.scalar(
                select(ExecutionRow).where(
                    ExecutionRow.idempotency_scope == request.idempotency_scope,
                    ExecutionRow.idempotency_key == request.idempotency_key,
                )
            )
            if existing is not None:
                if not reuse_existing:
                    raise TransactionRetryError('execution_idempotency_winner_changed')
                if existing.spec_sha256 != spec_sha256 or existing.owner_principal_id != request.owner_principal_id:
                    raise DomainConflictError('execution_idempotency_conflict')
                assignment = (
                    await self.session.get(AssignmentRow, existing.current_assignment_id)
                    if existing.current_assignment_id
                    else None
                )
                return execution_record(existing, assignment)
        row = self._new_execution_row(request, placement=placement, spec_sha256=spec_sha256, now=now)
        self.session.add(row)
        await self.session.flush()
        await self._append_control_event_row(
            row,
            event_type='execution.created',
            source='control',
            payload={'status': row.status, 'mode': row.mode, 'spec_type': row.spec_type},
            event_key='execution.created',
            now=now,
        )
        return execution_record(row)

    async def create_executions_batch(
        self,
        requests: list[CreateExecution],
        *,
        now: datetime,
    ) -> list[ExecutionRecord]:
        if not requests:
            return []
        identities: list[tuple[str, str]] = []
        seen_identities: set[tuple[str, str]] = set()
        by_scope: dict[str, set[str]] = {}
        for request in requests:
            _validate_environment_scope(request)
            if request.idempotency_scope is None or request.idempotency_key is None:
                raise DomainValidationError('batch_execution_idempotency_required')
            identity = (request.idempotency_scope, request.idempotency_key)
            if identity in seen_identities:
                raise DomainValidationError('duplicate_execution_idempotency_key')
            identities.append(identity)
            seen_identities.add(identity)
            by_scope.setdefault(identity[0], set()).add(identity[1])

        existing_rows = (
            await self.session.scalars(
                select(ExecutionRow).where(
                    or_(
                        *(
                            and_(
                                ExecutionRow.idempotency_scope == scope,
                                ExecutionRow.idempotency_key.in_(sorted(keys)),
                            )
                            for scope, keys in sorted(by_scope.items())
                        )
                    )
                )
            )
        ).all()
        existing_by_identity = {
            (row.idempotency_scope, row.idempotency_key): row
            for row in existing_rows
            if row.idempotency_scope is not None and row.idempotency_key is not None
        }

        rows: list[ExecutionRow] = []
        new_rows: list[ExecutionRow] = []
        events: list[dict[str, Any]] = []
        for request, identity in zip(requests, identities, strict=True):
            placement = self._normalize_placement(request.placement)
            spec_sha256 = request.spec_sha256 or canonical_sha256(request.spec)
            row = existing_by_identity.get(identity)
            if row is not None:
                if row.spec_sha256 != spec_sha256 or row.owner_principal_id != request.owner_principal_id:
                    raise DomainConflictError('execution_idempotency_conflict')
                rows.append(row)
                continue

            row = self._new_execution_row(request, placement=placement, spec_sha256=spec_sha256, now=now)
            row.next_event_seq = 2
            rows.append(row)
            new_rows.append(row)
            events.append({
                'execution_id': row.execution_id,
                'seq': 1,
                'assignment_id': None,
                'attempt_id': None,
                'type': 'execution.created',
                'source': 'control',
                'visibility': 'public',
                'payload_json': {'status': row.status, 'mode': row.mode, 'spec_type': row.spec_type},
                'event_key': 'execution.created',
                'created_at': now,
            })

        self.session.add_all(new_rows)
        await self.session.flush()
        if events:
            await self.session.execute(insert(ExecutionEventRow), events)
        return [execution_record(row) for row in rows]

    def _new_execution_row(
        self,
        request: CreateExecution,
        *,
        placement: dict[str, Any],
        spec_sha256: str,
        now: datetime,
    ) -> ExecutionRow:
        queue_started_at = request.first_schedulable_at
        if queue_started_at is None and request.allocation_id is None:
            queue_started_at = now
        return ExecutionRow(
            execution_id=request.execution_id,
            idempotency_scope=request.idempotency_scope,
            idempotency_key=request.idempotency_key,
            owner_principal_id=request.owner_principal_id,
            owner_tenant=request.owner_tenant,
            created_by=request.created_by,
            parent_execution_id=request.parent_execution_id,
            spec_sha256=spec_sha256,
            environment_id=request.environment_id,
            pool_id=request.pool_id,
            allocation_id=request.allocation_id,
            workload=request.workload,
            submission_sha256=request.submission_sha256,
            allocation_spec_hash=request.allocation_spec_hash,
            first_schedulable_at=request.first_schedulable_at,
            version=1,
            shard_id=stable_shard(request.execution_id, self.shard_count),
            mode=request.mode,
            spec_type=request.spec_type,
            instance_mode=request.instance_mode,
            spec_json=request.spec,
            task_name=task_name_from_spec(request.spec_type, request.spec),
            driver_config_json=request.driver_config,
            placement_json=placement,
            trace_carrier_json=dict(request.trace_carrier) if request.trace_carrier else None,
            status=ExecutionStatus.QUEUED,
            route_status=request.route_status,
            slot_cost=request.slot_cost,
            current_assignment_id=None,
            current_attempt_id=None,
            next_attempt_seq=1,
            attempt_count=0,
            max_attempts=request.max_attempts,
            next_event_seq=1,
            priority=request.priority,
            next_schedule_at=now,
            queue_entered_at=now,
            queue_ttl_seconds=request.queue_ttl_seconds,
            execution_ttl_seconds=request.execution_ttl_seconds,
            stop_timeout_seconds=request.stop_timeout_seconds,
            queue_expires_at=deadline_after(queue_started_at, request.queue_ttl_seconds),
            created_at=now,
            updated_at=now,
        )

    @staticmethod
    def _normalize_placement(placement: dict[str, Any]) -> dict[str, Any]:
        if placement:
            return dict(placement)
        return {'environment': DEFAULT_ENVIRONMENT_ID, 'pool': DEFAULT_POOL_ID}

    async def get_execution(
        self,
        execution_id: str,
        *,
        owner_principal_id: str | None = None,
        owner_principal_ids: Collection[str] | None = None,
        environment_id: str | None = None,
    ) -> ExecutionRecord:
        row = await self.session.get(ExecutionRow, execution_id)
        if (
            row is None
            or (owner_principal_id is not None and row.owner_principal_id != owner_principal_id)
            or (owner_principal_ids is not None and row.owner_principal_id not in owner_principal_ids)
            or (environment_id is not None and row.environment_id != environment_id)
        ):
            raise ResourceNotFoundError('execution_not_found')
        assignment = None
        if row.current_assignment_id is not None:
            assignment = await self.session.get(AssignmentRow, row.current_assignment_id)
        return execution_record(row, assignment)

    async def get_summary(
        self,
        execution_id: str,
        *,
        owner_principal_id: str | None = None,
        owner_principal_ids: Collection[str] | None = None,
        environment_id: str | None = None,
    ) -> ExecutionRecord:
        names = _SUMMARY_EXECUTION_FIELDS
        assignment_names = _SUMMARY_ASSIGNMENT_FIELDS
        query = (
            select(
                *(getattr(ExecutionRow, name) for name in names),
                *(getattr(AssignmentRow, name).label(f'assignment_{name}') for name in assignment_names),
            )
            .outerjoin(AssignmentRow, AssignmentRow.assignment_id == ExecutionRow.current_assignment_id)
            .where(ExecutionRow.execution_id == execution_id)
        )
        if owner_principal_id is not None:
            query = query.where(ExecutionRow.owner_principal_id == owner_principal_id)
        if owner_principal_ids is not None:
            query = query.where(ExecutionRow.owner_principal_id.in_(owner_principal_ids))
        if environment_id is not None:
            query = query.where(ExecutionRow.environment_id == environment_id)
        row = (await self.session.execute(query)).mappings().first()
        if row is None:
            raise ResourceNotFoundError('execution_not_found')
        assignment = None
        if row['assignment_assignment_id'] is not None:
            assignment = AssignmentRecord(**{name: row[f'assignment_{name}'] for name in assignment_names})
        return ExecutionRecord(
            **{name: row[name] for name in names},
            spec={},
            driver_config={},
            placement={},
            current_assignment=assignment,
        )

    async def get_log_context(
        self,
        execution_id: str,
        *,
        owner_principal_id: str | None = None,
        owner_principal_ids: Collection[str] | None = None,
        environment_id: str | None = None,
    ) -> ExecutionLogContext:
        row = await self.session.get(ExecutionRow, execution_id)
        if (
            row is None
            or (owner_principal_id is not None and row.owner_principal_id != owner_principal_id)
            or (owner_principal_ids is not None and row.owner_principal_id not in owner_principal_ids)
            or (environment_id is not None and row.environment_id != environment_id)
        ):
            raise ResourceNotFoundError('execution_not_found')
        assignment = (
            await self.session.get(AssignmentRow, row.current_assignment_id)
            if row.current_assignment_id is not None
            else None
        )
        return ExecutionLogContext(
            execution=execution_record(row, assignment),
            started_at=row.started_at,
            finished_at=row.finished_at,
        )

    async def list_executions(
        self,
        *,
        summary_only: bool = False,
        status: str | None = None,
        mode: str | None = None,
        spec_type: str | None = None,
        q: str | None = None,
        limit: int = 100,
        offset: int = 0,
        cursor: str | None = None,
        top_level_only: bool = False,
        owner_principal_id: str | None = None,
        owner_principal_ids: Collection[str] | None = None,
        owner_environment_pairs: Collection[tuple[str, str]] | None = None,
        environment_id: str | None = None,
        environment_ids: Collection[str] | None = None,
    ) -> list[ExecutionRecord]:
        query = (
            select(*(getattr(ExecutionRow, name) for name in _SUMMARY_EXECUTION_FIELDS))
            if summary_only
            else select(ExecutionRow)
        )
        filters = self._execution_list_filters(
            status=status,
            mode=mode,
            spec_type=spec_type,
            q=q,
            top_level_only=top_level_only,
            owner_principal_id=owner_principal_id,
            owner_principal_ids=owner_principal_ids,
            owner_environment_pairs=owner_environment_pairs,
            environment_id=environment_id,
            environment_ids=environment_ids,
        )
        if filters:
            query = query.where(*filters)
        if cursor is not None:
            updated_at, created_at, execution_id = _decode_execution_cursor(cursor)
            query = query.where(
                or_(
                    ExecutionRow.updated_at < updated_at,
                    and_(
                        ExecutionRow.updated_at == updated_at,
                        ExecutionRow.created_at < created_at,
                    ),
                    and_(
                        ExecutionRow.updated_at == updated_at,
                        ExecutionRow.created_at == created_at,
                        ExecutionRow.execution_id < execution_id,
                    ),
                )
            )
        query = (
            query
            .order_by(
                ExecutionRow.updated_at.desc(),
                ExecutionRow.created_at.desc(),
                ExecutionRow.execution_id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )
        if summary_only:
            rows = list((await self.session.execute(query)).mappings())
            assignment_ids = [row['current_assignment_id'] for row in rows if row['current_assignment_id']]
            assignments = (
                list(
                    (
                        await self.session.execute(
                            select(*(getattr(AssignmentRow, name) for name in _SUMMARY_ASSIGNMENT_FIELDS)).where(
                                AssignmentRow.assignment_id.in_(assignment_ids)
                            )
                        )
                    ).mappings()
                )
                if assignment_ids
                else []
            )
            by_id = {row['assignment_id']: AssignmentRecord(**row) for row in assignments}
            return [
                ExecutionRecord(
                    **row,
                    spec={},
                    driver_config={},
                    placement={},
                    current_assignment=by_id.get(row['current_assignment_id']),
                )
                for row in rows
            ]
        rows = list((await self.session.scalars(query)).all())
        assignment_ids = [row.current_assignment_id for row in rows if row.current_assignment_id]
        assignments = (
            list(
                await self.session.scalars(select(AssignmentRow).where(AssignmentRow.assignment_id.in_(assignment_ids)))
            )
            if assignment_ids
            else []
        )
        assignments_by_id = {assignment.assignment_id: assignment for assignment in assignments}
        return [execution_record(row, assignments_by_id.get(row.current_assignment_id)) for row in rows]

    async def count_executions(
        self,
        *,
        status: str | None = None,
        mode: str | None = None,
        spec_type: str | None = None,
        q: str | None = None,
        top_level_only: bool = False,
        owner_principal_id: str | None = None,
        owner_principal_ids: Collection[str] | None = None,
        owner_environment_pairs: Collection[tuple[str, str]] | None = None,
        environment_id: str | None = None,
        environment_ids: Collection[str] | None = None,
    ) -> int:
        query = select(func.count()).select_from(ExecutionRow)
        filters = self._execution_list_filters(
            status=status,
            mode=mode,
            spec_type=spec_type,
            q=q,
            top_level_only=top_level_only,
            owner_principal_id=owner_principal_id,
            owner_principal_ids=owner_principal_ids,
            owner_environment_pairs=owner_environment_pairs,
            environment_id=environment_id,
            environment_ids=environment_ids,
        )
        if filters:
            query = query.where(*filters)
        return int(await self.session.scalar(query) or 0)

    @staticmethod
    def _execution_list_filters(
        *,
        status: str | None,
        mode: str | None,
        spec_type: str | None,
        q: str | None,
        top_level_only: bool,
        owner_principal_id: str | None = None,
        owner_principal_ids: Collection[str] | None = None,
        owner_environment_pairs: Collection[tuple[str, str]] | None = None,
        environment_id: str | None = None,
        environment_ids: Collection[str] | None = None,
    ):
        filters = []
        if owner_principal_id is not None:
            filters.append(ExecutionRow.owner_principal_id == owner_principal_id)
        if owner_principal_ids is not None:
            filters.append(ExecutionRow.owner_principal_id.in_(owner_principal_ids))
        if owner_environment_pairs is not None:
            pairs = sorted(set(owner_environment_pairs))
            filters.append(
                or_(
                    *(
                        and_(
                            ExecutionRow.owner_principal_id == owner_principal_id,
                            ExecutionRow.environment_id == environment_id,
                        )
                        for owner_principal_id, environment_id in pairs
                    )
                )
                if pairs
                else false()
            )
        if environment_id is not None:
            filters.append(ExecutionRow.environment_id == environment_id)
        if environment_ids is not None:
            filters.append(ExecutionRow.environment_id.in_(environment_ids))
        if status:
            filters.append(ExecutionRow.status == status)
        if mode:
            filters.append(ExecutionRow.mode == mode)
        if spec_type:
            filters.append(ExecutionRow.spec_type == spec_type)
        if top_level_only:
            filters.append(ExecutionRow.parent_execution_id.is_(None))
        if q and q.strip():
            pattern = f'%{q.strip()}%'
            manager_match = (
                select(AssignmentRow.assignment_id)
                .where(
                    AssignmentRow.assignment_id == ExecutionRow.current_assignment_id,
                    AssignmentRow.manager_id.ilike(pattern),
                )
                .exists()
            )
            filters.append(
                or_(
                    ExecutionRow.execution_id.ilike(pattern),
                    ExecutionRow.mode.ilike(pattern),
                    ExecutionRow.spec_type.ilike(pattern),
                    ExecutionRow.status.ilike(pattern),
                    ExecutionRow.route_status.ilike(pattern),
                    manager_match,
                )
            )
        return filters

    @staticmethod
    def page_cursor(execution: ExecutionRecord) -> str:
        return encode_cursor({
            'updated_at': execution.updated_at.isoformat(),
            'created_at': execution.created_at.isoformat(),
            'execution_id': execution.execution_id,
        })

    async def activate_route(self, execution_id: str, *, now: datetime) -> ExecutionRecord:
        row = await self.session.get(ExecutionRow, execution_id)
        if row is None:
            raise ResourceNotFoundError('execution_not_found')
        if row.mode != ExecutionMode.LIVE:
            raise DomainConflictError('route_not_supported')
        if row.status in TERMINAL_EXECUTION_STATUSES:
            raise ResourceGoneError('execution_terminal')
        if row.route_status == RouteStatus.CLOSED:
            raise ResourceGoneError('route_closed')
        if row.route_status == RouteStatus.NONE:
            raise DomainConflictError('route_not_cached')
        if row.route_status == RouteStatus.CACHED:
            old_route_status = row.route_status
            row.route_status = RouteStatus.ACTIVE
            row.version += 1
            row.updated_at = now
            await self._append_control_event_row(
                row,
                event_type='route.status.changed',
                source='control',
                payload={'from': old_route_status, 'to': row.route_status},
                now=now,
            )
            await self.session.flush()
        assignment = (
            await self.session.get(AssignmentRow, row.current_assignment_id) if row.current_assignment_id else None
        )
        return execution_record(row, assignment)

    async def list_commands(self, execution_id: str):
        rows = (
            await self.session.scalars(
                select(ExecutionCommandRow)
                .where(ExecutionCommandRow.execution_id == execution_id)
                .order_by(ExecutionCommandRow.created_at)
            )
        ).all()
        return [command_record(row) for row in rows]

    async def append_control_event(
        self,
        execution_id: str,
        *,
        event_type: str,
        source: str,
        payload: dict[str, Any],
        now: datetime,
        visibility: str = 'public',
        assignment_id: str | None = None,
        attempt_id: str | None = None,
        event_key: str | None = None,
    ):
        row = await self.session.get(ExecutionRow, execution_id)
        if row is None:
            raise ResourceNotFoundError('execution_not_found')
        event = await self._append_control_event_row(
            row,
            event_type=event_type,
            source=source,
            payload=payload,
            visibility=visibility,
            assignment_id=assignment_id,
            attempt_id=attempt_id,
            event_key=event_key,
            now=now,
        )
        return event_record(event)

    async def list_control_events(self, execution_id: str, *, after_seq: int = 0, limit: int = 100):
        rows = (
            await self.session.scalars(
                select(ExecutionEventRow)
                .where(
                    ExecutionEventRow.execution_id == execution_id,
                    ExecutionEventRow.seq > after_seq,
                )
                .order_by(ExecutionEventRow.seq)
                .limit(limit)
            )
        ).all()
        return [event_record(row) for row in rows]

    async def request_stop(
        self,
        execution_id: str,
        *,
        now: datetime,
        stop_reason: str = 'user_stop',
    ) -> ExecutionRecord:
        _validate_stop_reason(stop_reason)
        row = await self.session.scalar(
            select(ExecutionRow)
            .where(ExecutionRow.execution_id == execution_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if row is None:
            raise ResourceNotFoundError('execution_not_found')
        assignment = None
        if row.current_assignment_id is not None:
            assignment = await self.session.scalar(
                select(AssignmentRow)
                .where(AssignmentRow.assignment_id == row.current_assignment_id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )

        await self._apply_stop_locked(
            row,
            assignment,
            stop_reason=stop_reason,
            now=now,
        )
        await self.session.flush()
        assignment = (
            await self.session.get(AssignmentRow, row.current_assignment_id) if row.current_assignment_id else None
        )
        return execution_record(row, assignment)

    async def request_stop_many(
        self,
        execution_ids: list[str] | tuple[str, ...],
        *,
        now: datetime,
        stop_reason: str = 'user_stop',
    ) -> int:
        _validate_stop_reason(stop_reason)
        unique_ids = tuple(sorted(set(execution_ids)))
        if not unique_ids:
            return 0
        if len(unique_ids) > 10000:
            raise DomainValidationError('stop_batch_too_large')

        rows = list(
            await self.session.scalars(
                select(ExecutionRow)
                .options(load_only(*_STOP_EXECUTION_COLUMNS))
                .where(
                    ExecutionRow.execution_id.in_(unique_ids),
                    ExecutionRow.status.in_(_STOPPABLE_EXECUTION_STATUSES),
                )
                .order_by(ExecutionRow.execution_id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
        )
        assignment_ids = tuple(sorted({row.current_assignment_id for row in rows if row.current_assignment_id}))
        assignments = {}
        commands_by_assignment: dict[str, list[ExecutionCommandRow]] = {}
        if assignment_ids:
            assignment_rows = list(
                await self.session.scalars(
                    select(AssignmentRow)
                    .options(load_only(*_STOP_ASSIGNMENT_COLUMNS))
                    .where(AssignmentRow.assignment_id.in_(assignment_ids))
                    .order_by(AssignmentRow.assignment_id)
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
            )
            assignments = {assignment.assignment_id: assignment for assignment in assignment_rows}
            command_rows = list(
                await self.session.scalars(
                    select(ExecutionCommandRow)
                    .options(load_only(*_STOP_COMMAND_COLUMNS))
                    .where(
                        ExecutionCommandRow.assignment_id.in_(assignment_ids),
                        ExecutionCommandRow.status.in_([CommandStatus.PENDING, CommandStatus.CLAIMED]),
                    )
                    .order_by(ExecutionCommandRow.assignment_id, ExecutionCommandRow.created_at)
                    .with_for_update()
                )
            )
            for command in command_rows:
                commands_by_assignment.setdefault(command.assignment_id, []).append(command)

        changed = 0
        for row in rows:
            assignment = assignments.get(row.current_assignment_id) if row.current_assignment_id else None
            unfinished_commands = commands_by_assignment.get(row.current_assignment_id, [])
            existing_stop_request = next(
                (
                    command
                    for command in unfinished_commands
                    if command.type == 'stop.request' and command.status == CommandStatus.PENDING
                ),
                None,
            )
            changed += int(
                await self._apply_stop_locked(
                    row,
                    assignment,
                    stop_reason=stop_reason,
                    now=now,
                    unfinished_commands=unfinished_commands,
                    existing_stop_request=existing_stop_request,
                    events_locked=True,
                )
            )
        await self.session.flush()
        return changed

    async def _apply_stop_locked(
        self,
        row: ExecutionRow,
        assignment: AssignmentRow | None,
        *,
        stop_reason: str,
        now: datetime,
        unfinished_commands: list[ExecutionCommandRow] | None = None,
        existing_stop_request: ExecutionCommandRow | None | object = _COMMAND_NOT_LOADED,
        events_locked: bool = False,
    ) -> bool:
        if row.status not in _STOPPABLE_EXECUTION_STATUSES:
            raise DomainConflictError(
                'execution_not_stoppable',
                detail=f'cannot stop execution in status {row.status}',
            )
        if row.status == ExecutionStatus.ASSIGNED and (
            assignment is None or assignment.status != AssignmentStatus.PENDING
        ):
            raise TemporaryUnavailableError('assignment_unavailable')
        old_status = row.status
        old_route_status = row.route_status
        old_stop_reason = row.stop_reason
        if row.stop_reason is None:
            row.stop_reason = stop_reason
        effective_stop_reason = row.stop_reason
        terminal_status = stop_terminal_status(effective_stop_reason)

        if row.status == ExecutionStatus.QUEUED:
            row.status = terminal_status
            row.finished_at = now
        elif (
            row.status == ExecutionStatus.ASSIGNED
            and assignment is not None
            and assignment.status == AssignmentStatus.PENDING
        ):
            assignment.status = AssignmentStatus.REVOKED
            assignment.updated_at = now
            assignment.finished_at = now
            await self._cancel_unfinished_commands_for_assignment(
                assignment,
                row,
                source='control',
                now=now,
                rows=unfinished_commands,
                events_locked=events_locked,
            )
            row.status = terminal_status
            row.finished_at = now
        elif row.status in (ExecutionStatus.STARTING, ExecutionStatus.RUNNING, ExecutionStatus.WAITING_INPUT):
            if assignment is None or assignment.status != AssignmentStatus.ACTIVE:
                if assignment is not None:
                    assignment.endpoint_ready = False
                    assignment.updated_at = now
                    await self._cancel_unfinished_commands_for_assignment(
                        assignment,
                        row,
                        source='control',
                        now=now,
                        rows=unfinished_commands,
                        events_locked=events_locked,
                    )
                row.status = terminal_status
                row.finished_at = now
            else:
                row.status = ExecutionStatus.STOPPING
                row.stop_deadline_at = row.stop_deadline_at or stop_deadline_after(now, row.stop_timeout_seconds)
                existing = existing_stop_request
                if existing is _COMMAND_NOT_LOADED and row.current_assignment_id is not None:
                    existing = await self.session.scalar(
                        select(ExecutionCommandRow).where(
                            ExecutionCommandRow.execution_id == row.execution_id,
                            ExecutionCommandRow.assignment_id == row.current_assignment_id,
                            ExecutionCommandRow.type == 'stop.request',
                            ExecutionCommandRow.status == CommandStatus.PENDING,
                        )
                    )
                elif existing is _COMMAND_NOT_LOADED:
                    existing = None
                if existing is None:
                    self.session.add(
                        ExecutionCommandRow(
                            command_id=new_id('cmd'),
                            execution_id=row.execution_id,
                            assignment_id=assignment.assignment_id,
                            attempt_id=assignment.attempt_id,
                            type='stop.request',
                            status=CommandStatus.PENDING,
                            payload_json={'reason': effective_stop_reason},
                            created_at=now,
                            updated_at=now,
                        )
                    )
        elif row.status in (ExecutionStatus.STOPPING, ExecutionStatus.KILLED, ExecutionStatus.EXPIRED):
            pass

        if row.mode == ExecutionMode.LIVE:
            row.route_status = RouteStatus.CLOSED

        if row.status != old_status:
            if events_locked:
                self._append_control_event_row_locked(
                    row,
                    event_type='execution.status.changed',
                    source='control',
                    payload={'from': old_status, 'to': row.status},
                    now=now,
                )
            else:
                await self._append_control_event_row(
                    row,
                    event_type='execution.status.changed',
                    source='control',
                    payload={'from': old_status, 'to': row.status},
                    now=now,
                )
        if row.route_status != old_route_status:
            if events_locked:
                self._append_control_event_row_locked(
                    row,
                    event_type='route.status.changed',
                    source='control',
                    payload={'from': old_route_status, 'to': row.route_status},
                    now=now,
                )
            else:
                await self._append_control_event_row(
                    row,
                    event_type='route.status.changed',
                    source='control',
                    payload={'from': old_route_status, 'to': row.route_status},
                    now=now,
                )
        changed = row.status != old_status or row.route_status != old_route_status or row.stop_reason != old_stop_reason
        if changed:
            row.version += 1
        row.updated_at = now
        return changed

    def _append_control_event_row_locked(
        self,
        row: ExecutionRow,
        *,
        event_type: str,
        source: str,
        payload: dict[str, Any],
        now: datetime,
        assignment_id: str | None = None,
        attempt_id: str | None = None,
    ) -> ExecutionEventRow:
        seq = int(row.next_event_seq)
        row.next_event_seq = seq + 1
        event = ExecutionEventRow(
            execution_id=row.execution_id,
            seq=seq,
            assignment_id=assignment_id,
            attempt_id=attempt_id,
            type=event_type,
            source=source,
            visibility='public',
            payload_json=payload,
            event_key=None,
            created_at=now,
        )
        self.session.add(event)
        queue_terminal_event_metrics(self.session, row, event_type=event_type, payload=payload)
        return event

    async def _append_control_event_row(
        self,
        row: ExecutionRow,
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
            existing = await self.session.scalar(
                select(ExecutionEventRow).where(
                    ExecutionEventRow.execution_id == row.execution_id,
                    ExecutionEventRow.event_key == event_key,
                )
            )
            if existing is not None:
                return existing
        return await append_control_event_row(
            self.session,
            row,
            event_type=event_type,
            source=source,
            payload=payload,
            visibility=visibility,
            assignment_id=assignment_id,
            attempt_id=attempt_id,
            event_key=event_key,
            now=now,
        )

    async def _cancel_unfinished_commands_for_assignment(
        self,
        assignment: AssignmentRow,
        execution: ExecutionRow,
        *,
        source: str,
        now: datetime,
        rows: list[ExecutionCommandRow] | None = None,
        events_locked: bool = False,
    ) -> int:
        commands = rows
        if commands is None:
            commands = list(
                await self.session.scalars(
                    select(ExecutionCommandRow).where(
                        ExecutionCommandRow.assignment_id == assignment.assignment_id,
                        ExecutionCommandRow.status.in_([CommandStatus.PENDING, CommandStatus.CLAIMED]),
                    )
                )
            )
        changed = 0
        for command in commands:
            if command.status not in {CommandStatus.PENDING, CommandStatus.CLAIMED}:
                continue
            old_status = command.status
            command.status = CommandStatus.CANCELLED
            command.completed_at = now
            command.updated_at = now
            if events_locked:
                self._append_control_event_row_locked(
                    execution,
                    event_type='command.status.changed',
                    source=source,
                    payload={'command_id': command.command_id, 'from': old_status, 'to': command.status},
                    assignment_id=command.assignment_id,
                    attempt_id=command.attempt_id,
                    now=now,
                )
            else:
                await append_control_event_row(
                    self.session,
                    execution,
                    event_type='command.status.changed',
                    source=source,
                    payload={'command_id': command.command_id, 'from': old_status, 'to': command.status},
                    assignment_id=command.assignment_id,
                    attempt_id=command.attempt_id,
                    now=now,
                )
            changed += 1
        return changed


def _validate_environment_scope(request: CreateExecution) -> None:
    if request.environment_id is not None:
        validate_environment_id(request.environment_id)
    if request.pool_id is not None:
        validate_pool_id(request.pool_id)
