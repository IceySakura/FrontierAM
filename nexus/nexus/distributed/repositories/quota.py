from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import exists, func, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.distributed.errors import AdmissionRejectedError, DomainConflictError, ResourceNotFoundError
from nexus.distributed.execution.models import (
    AssignmentStatus,
    ExecutionStatus,
)
from nexus.distributed.persistence.orm import (
    AssignmentRow,
    BatchItemAttemptRow,
    EnvironmentAllocationRow,
    EnvironmentGrantRow,
    EnvironmentResourceUsageRow,
    EnvironmentRow,
    ExecutionRow,
    QuotaUsageRow,
    UserQuotaOverrideRow,
)
from nexus.distributed.submission.resources import ResourceNormalizationError, validate_resource_totals
from nexus.utils.platform_metrics import record


_OWNER_SCOPE = ('*', '*', 'owner')
_RUNNING_STATUSES = {
    ExecutionStatus.ASSIGNED,
    ExecutionStatus.STARTING,
    ExecutionStatus.RUNNING,
    ExecutionStatus.WAITING_INPUT,
    ExecutionStatus.STOPPING,
}


def _mysql_ensure_owner_usage(values: dict[str, Any]) -> Any:
    return (
        mysql_insert(QuotaUsageRow)
        .values(values)
        .on_duplicate_key_update(owner_principal_id=values['owner_principal_id'])
    )


def _mysql_ensure_owner_usage_many(values: list[dict[str, Any]]) -> Any:
    statement = mysql_insert(QuotaUsageRow).values(values)
    return statement.on_duplicate_key_update(owner_principal_id=statement.inserted.owner_principal_id)


class QuotaExceededError(AdmissionRejectedError):
    def __init__(self, code: str | None = None, **kwargs):
        super().__init__(code, **kwargs)
        record('capacity_rejections', operation='quota', outcome='rejected', reason=self.code)


@dataclass(frozen=True, slots=True)
class QuotaSnapshot:
    queued_executions: int
    running_executions: int
    reserved_slots: int


@dataclass(frozen=True, slots=True)
class ComputeUsage:
    manager_count: int
    cpu_millicores: int
    memory_bytes: int
    slots: int
    active_batches: int


@dataclass(slots=True)
class _OwnerAssignmentReservation:
    row: QuotaUsageRow
    snapshot: QuotaSnapshot
    limits: dict[str, Any]


class AssignmentQuotaBatch:
    def __init__(self, reservations: dict[str, _OwnerAssignmentReservation]) -> None:
        self._reservations = reservations

    def reserve(self, execution: ExecutionRow, *, now: datetime) -> None:
        owner_principal_id = execution.owner_principal_id
        if owner_principal_id is None:
            return
        reservation = self._reservations.get(owner_principal_id)
        if reservation is None:
            raise RuntimeError('assignment_quota_owner_not_prepared')
        snapshot = reservation.snapshot
        limits = reservation.limits
        if snapshot.running_executions + 1 > int(limits.get('running_executions', 100)):
            raise QuotaExceededError('running_execution_quota_exceeded')
        if snapshot.reserved_slots + execution.slot_cost > int(limits.get('reserved_slots', 100)):
            raise QuotaExceededError('reserved_slot_quota_exceeded')
        next_snapshot = QuotaSnapshot(
            queued_executions=max(0, snapshot.queued_executions - 1),
            running_executions=snapshot.running_executions + 1,
            reserved_slots=snapshot.reserved_slots + execution.slot_cost,
        )
        QuotaRepository._write_snapshot(reservation.row, next_snapshot, now=now)
        reservation.snapshot = next_snapshot


class QuotaRepository:
    """Serialize owner quota decisions through one owner-scoped usage row.

    The counters are a repairable projection. Admission always takes the owner
    row lock and recomputes the authoritative counts in the same transaction,
    so stale counters cannot oversell quota.
    """

    def __init__(self, session: AsyncSession, *, default_limits: dict[str, Any]) -> None:
        self.session = session
        self.default_limits = dict(default_limits)

    async def reserve_allocation(
        self,
        allocation_id: str,
        *,
        now: datetime,
    ) -> tuple[EnvironmentResourceUsageRow, EnvironmentResourceUsageRow] | None:
        allocation = await self.session.scalar(
            select(EnvironmentAllocationRow)
            .where(EnvironmentAllocationRow.allocation_id == allocation_id)
            .with_for_update()
        )
        if allocation is None:
            raise ResourceNotFoundError('allocation_not_found')
        if allocation.resources_reserved_at is not None:
            return None
        environment_usage, grant_usage = await self._lock_compute_usage_pair(
            environment_id=allocation.environment_id,
            principal_id=allocation.owner_principal_id,
        )
        if allocation.status == 'released':
            raise DomainConflictError('allocation_already_released')

        environment = await self.session.scalar(
            select(EnvironmentRow)
            .where(EnvironmentRow.environment_id == allocation.environment_id)
            .execution_options(populate_existing=True)
        )
        grant = await self.session.scalar(
            select(EnvironmentGrantRow)
            .where(
                EnvironmentGrantRow.principal_id == allocation.owner_principal_id,
                EnvironmentGrantRow.environment_id == allocation.environment_id,
            )
            .execution_options(populate_existing=True)
        )
        if environment is None or grant is None:
            raise RuntimeError('allocation_access_scope_missing')
        if environment.status != 'enabled':
            raise DomainConflictError('environment_disabled')
        if not grant.enabled:
            raise DomainConflictError('environment_grant_disabled')

        requested = _allocation_usage(allocation)
        next_environment = _plus(_compute_usage(environment_usage), requested)
        next_grant = _plus(_compute_usage(grant_usage), requested)
        _enforce_compute_limits(next_environment, environment.resource_limits_json or {}, scope='environment')
        _enforce_compute_limits(next_grant, grant.limits_json or {}, scope='grant')

        _write_compute_usage(environment_usage, next_environment, now=now)
        _write_compute_usage(grant_usage, next_grant, now=now)
        allocation.resources_reserved_at = now
        allocation.resources_released_at = None
        allocation.updated_at = now
        await self.session.flush()
        return environment_usage, grant_usage

    async def release_allocation(
        self,
        allocation_id: str,
        *,
        now: datetime,
    ) -> tuple[EnvironmentResourceUsageRow, EnvironmentResourceUsageRow] | None:
        allocation = await self.session.scalar(
            select(EnvironmentAllocationRow)
            .where(EnvironmentAllocationRow.allocation_id == allocation_id)
            .with_for_update()
        )
        if allocation is None:
            raise ResourceNotFoundError('allocation_not_found')
        if allocation.resources_released_at is not None or allocation.resources_reserved_at is None:
            return None
        environment_usage, grant_usage = await self._lock_compute_usage_pair(
            environment_id=allocation.environment_id,
            principal_id=allocation.owner_principal_id,
        )
        if allocation.status != 'released':
            raise DomainConflictError('allocation_not_released')

        requested = _allocation_usage(allocation)
        next_environment = _minus(_compute_usage(environment_usage), requested)
        next_grant = _minus(_compute_usage(grant_usage), requested)
        _write_compute_usage(environment_usage, next_environment, now=now)
        _write_compute_usage(grant_usage, next_grant, now=now)
        allocation.resources_released_at = now
        allocation.updated_at = now
        await self.session.flush()
        return environment_usage, grant_usage

    async def get_compute_usage(
        self,
        *,
        environment_id: str,
        principal_id: str | None = None,
    ) -> ComputeUsage:
        scope_key = _compute_scope_key(environment_id=environment_id, principal_id=principal_id)
        row = await self.session.get(EnvironmentResourceUsageRow, scope_key)
        return _compute_usage(row) if row is not None else ComputeUsage(0, 0, 0, 0, 0)

    async def _lock_compute_usage_pair(
        self,
        *,
        environment_id: str,
        principal_id: str,
    ) -> tuple[EnvironmentResourceUsageRow, EnvironmentResourceUsageRow]:
        environment_key = _compute_scope_key(environment_id=environment_id, principal_id=None)
        grant_key = _compute_scope_key(environment_id=environment_id, principal_id=principal_id)
        rows = list(
            await self.session.scalars(
                select(EnvironmentResourceUsageRow)
                .where(EnvironmentResourceUsageRow.scope_key.in_((environment_key, grant_key)))
                .order_by(EnvironmentResourceUsageRow.scope_key)
                .with_for_update()
            )
        )
        by_key = {row.scope_key: row for row in rows}
        if set(by_key) != {environment_key, grant_key}:
            raise RuntimeError('environment_resource_usage_unavailable')
        return by_key[environment_key], by_key[grant_key]

    async def limits_for(self, owner_principal_id: str) -> dict[str, Any]:
        limits = dict(self.default_limits)
        override = await self.session.get(UserQuotaOverrideRow, owner_principal_id)
        if override is not None:
            limits.update(override.limits_json or {})
        return limits

    async def admit_submission(self, owner_principal_id: str, *, now: datetime) -> dict[str, Any]:
        usage = await self._lock_owner_usage(owner_principal_id, now=now)
        snapshot = await self._measure(owner_principal_id)
        limits = await self.limits_for(owner_principal_id)
        if snapshot.queued_executions + 1 > int(limits.get('queued_executions', 1000)):
            raise QuotaExceededError('queued_execution_quota_exceeded')
        self._write_snapshot(
            usage,
            QuotaSnapshot(
                queued_executions=snapshot.queued_executions + 1,
                running_executions=snapshot.running_executions,
                reserved_slots=snapshot.reserved_slots,
            ),
            now=now,
        )
        return limits

    async def reserve_assignment(self, execution: ExecutionRow, *, now: datetime) -> None:
        reservation = await self.prepare_assignment_batch([execution], now=now)
        reservation.reserve(execution, now=now)

    async def prepare_assignment_batch(
        self,
        executions: list[ExecutionRow],
        *,
        now: datetime,
    ) -> AssignmentQuotaBatch:
        owners = sorted({
            execution.owner_principal_id for execution in executions if execution.owner_principal_id is not None
        })
        if not owners:
            return AssignmentQuotaBatch({})
        usage_rows = await self._lock_owner_usage_many(owners, now=now)
        snapshots = await self._measure_many(owners)
        overrides = {
            row.owner_principal_id: dict(row.limits_json or {})
            for row in await self.session.scalars(
                select(UserQuotaOverrideRow).where(UserQuotaOverrideRow.owner_principal_id.in_(owners))
            )
        }
        reservations = {}
        for owner in owners:
            limits = dict(self.default_limits)
            limits.update(overrides.get(owner, {}))
            reservations[owner] = _OwnerAssignmentReservation(
                row=usage_rows[owner],
                snapshot=snapshots[owner],
                limits=limits,
            )
        return AssignmentQuotaBatch(reservations)

    async def refresh_owner(self, owner_principal_id: str, *, now: datetime) -> QuotaSnapshot:
        usage = await self._lock_owner_usage(owner_principal_id, now=now)
        snapshot = await self._measure(owner_principal_id)
        self._write_snapshot(usage, snapshot, now=now)
        return snapshot

    async def repair_all(self, *, now: datetime, limit: int = 500) -> int:
        owners = set(
            await self.session.scalars(
                select(ExecutionRow.owner_principal_id)
                .distinct()
                .order_by(ExecutionRow.owner_principal_id)
                .limit(limit)
            )
        )
        owners.update(
            await self.session.scalars(
                select(QuotaUsageRow.owner_principal_id)
                .where(
                    QuotaUsageRow.environment == _OWNER_SCOPE[0],
                    QuotaUsageRow.pool == _OWNER_SCOPE[1],
                    QuotaUsageRow.quota_class == _OWNER_SCOPE[2],
                )
                .order_by(QuotaUsageRow.owner_principal_id)
                .limit(limit)
            )
        )
        selected = sorted(owner for owner in owners if owner is not None)[:limit]
        for owner in selected:
            await self.refresh_owner(owner, now=now)
        return len(selected)

    async def _measure(self, owner_principal_id: str) -> QuotaSnapshot:
        queued = int(
            await self.session.scalar(
                select(func.count(ExecutionRow.execution_id)).where(
                    ExecutionRow.owner_principal_id == owner_principal_id,
                    ExecutionRow.status == ExecutionStatus.QUEUED,
                    ~exists().where(BatchItemAttemptRow.child_execution_id == ExecutionRow.execution_id),
                )
            )
            or 0
        )
        running = int(
            await self.session.scalar(
                select(func.count(ExecutionRow.execution_id)).where(
                    ExecutionRow.owner_principal_id == owner_principal_id,
                    ExecutionRow.status.in_(_RUNNING_STATUSES),
                )
            )
            or 0
        )
        slots = int(
            await self.session.scalar(
                select(func.coalesce(func.sum(AssignmentRow.slot_cost), 0))
                .join(ExecutionRow, ExecutionRow.execution_id == AssignmentRow.execution_id)
                .where(
                    ExecutionRow.owner_principal_id == owner_principal_id,
                    AssignmentRow.status.in_([AssignmentStatus.PENDING, AssignmentStatus.ACTIVE]),
                )
            )
            or 0
        )
        return QuotaSnapshot(
            queued_executions=queued,
            running_executions=running,
            reserved_slots=slots,
        )

    async def _measure_many(self, owner_principal_ids: list[str]) -> dict[str, QuotaSnapshot]:
        snapshots = {
            owner: QuotaSnapshot(queued_executions=0, running_executions=0, reserved_slots=0)
            for owner in owner_principal_ids
        }
        queued_rows = (
            await self.session.execute(
                select(ExecutionRow.owner_principal_id, func.count(ExecutionRow.execution_id))
                .where(
                    ExecutionRow.owner_principal_id.in_(owner_principal_ids),
                    ExecutionRow.status == ExecutionStatus.QUEUED,
                    ~exists().where(BatchItemAttemptRow.child_execution_id == ExecutionRow.execution_id),
                )
                .group_by(ExecutionRow.owner_principal_id)
            )
        ).all()
        running_rows = (
            await self.session.execute(
                select(ExecutionRow.owner_principal_id, func.count(ExecutionRow.execution_id))
                .where(
                    ExecutionRow.owner_principal_id.in_(owner_principal_ids),
                    ExecutionRow.status.in_(_RUNNING_STATUSES),
                )
                .group_by(ExecutionRow.owner_principal_id)
            )
        ).all()
        slot_rows = (
            await self.session.execute(
                select(ExecutionRow.owner_principal_id, func.coalesce(func.sum(AssignmentRow.slot_cost), 0))
                .join(ExecutionRow, ExecutionRow.execution_id == AssignmentRow.execution_id)
                .where(
                    ExecutionRow.owner_principal_id.in_(owner_principal_ids),
                    AssignmentRow.status.in_([AssignmentStatus.PENDING, AssignmentStatus.ACTIVE]),
                )
                .group_by(ExecutionRow.owner_principal_id)
            )
        ).all()
        queued = {owner: int(value or 0) for owner, value in queued_rows}
        running = {owner: int(value or 0) for owner, value in running_rows}
        slots = {owner: int(value or 0) for owner, value in slot_rows}
        return {
            owner: QuotaSnapshot(
                queued_executions=queued.get(owner, 0),
                running_executions=running.get(owner, 0),
                reserved_slots=slots.get(owner, 0),
            )
            for owner in snapshots
        }

    async def _lock_owner_usage(self, owner_principal_id: str, *, now: datetime) -> QuotaUsageRow:
        values = {
            'owner_principal_id': owner_principal_id,
            'environment': _OWNER_SCOPE[0],
            'pool': _OWNER_SCOPE[1],
            'quota_class': _OWNER_SCOPE[2],
            'queued_executions': 0,
            'running_executions': 0,
            'reserved_slots': 0,
            'version': 1,
            'updated_at': now,
        }
        dialect = self.session.get_bind().dialect.name
        if dialect == 'mysql':
            await self.session.execute(_mysql_ensure_owner_usage(values))
        elif dialect == 'sqlite':
            await self.session.execute(
                sqlite_insert(QuotaUsageRow)
                .values(values)
                .on_conflict_do_nothing(index_elements=['owner_principal_id', 'environment', 'pool', 'quota_class'])
            )
        elif await self.session.get(QuotaUsageRow, (owner_principal_id, *_OWNER_SCOPE)) is None:
            self.session.add(QuotaUsageRow(**values))
        await self.session.flush()
        row = await self.session.scalar(
            select(QuotaUsageRow)
            .where(
                QuotaUsageRow.owner_principal_id == owner_principal_id,
                QuotaUsageRow.environment == _OWNER_SCOPE[0],
                QuotaUsageRow.pool == _OWNER_SCOPE[1],
                QuotaUsageRow.quota_class == _OWNER_SCOPE[2],
            )
            .with_for_update()
        )
        if row is None:
            raise RuntimeError('quota_usage_row_unavailable')
        return row

    async def _lock_owner_usage_many(
        self,
        owner_principal_ids: list[str],
        *,
        now: datetime,
    ) -> dict[str, QuotaUsageRow]:
        owner_principal_ids = sorted(set(owner_principal_ids))
        values = [
            {
                'owner_principal_id': owner,
                'environment': _OWNER_SCOPE[0],
                'pool': _OWNER_SCOPE[1],
                'quota_class': _OWNER_SCOPE[2],
                'queued_executions': 0,
                'running_executions': 0,
                'reserved_slots': 0,
                'version': 1,
                'updated_at': now,
            }
            for owner in owner_principal_ids
        ]
        dialect = self.session.get_bind().dialect.name
        if dialect == 'mysql':
            await self.session.execute(_mysql_ensure_owner_usage_many(values))
        elif dialect == 'sqlite':
            await self.session.execute(
                sqlite_insert(QuotaUsageRow)
                .values(values)
                .on_conflict_do_nothing(index_elements=['owner_principal_id', 'environment', 'pool', 'quota_class'])
            )
        else:
            for value in values:
                key = (value['owner_principal_id'], *_OWNER_SCOPE)
                if await self.session.get(QuotaUsageRow, key) is None:
                    self.session.add(QuotaUsageRow(**value))
        await self.session.flush()
        query = (
            select(QuotaUsageRow)
            .where(
                QuotaUsageRow.owner_principal_id.in_(owner_principal_ids),
                QuotaUsageRow.environment == _OWNER_SCOPE[0],
                QuotaUsageRow.pool == _OWNER_SCOPE[1],
                QuotaUsageRow.quota_class == _OWNER_SCOPE[2],
            )
            .order_by(QuotaUsageRow.owner_principal_id)
        )
        # MySQL's duplicate-key upsert already holds X locks on every target
        # row. A second range-style FOR UPDATE can acquire neighboring rows in
        # a different order and deadlock overlapping Scheduler batches.
        if dialect != 'mysql':
            query = query.with_for_update()
        rows = list(await self.session.scalars(query))
        result = {row.owner_principal_id: row for row in rows}
        if set(result) != set(owner_principal_ids):
            raise RuntimeError('quota_usage_rows_unavailable')
        return result

    @staticmethod
    def _write_snapshot(row: QuotaUsageRow, snapshot: QuotaSnapshot, *, now: datetime) -> None:
        row.queued_executions = snapshot.queued_executions
        row.running_executions = snapshot.running_executions
        row.reserved_slots = snapshot.reserved_slots
        row.version += 1
        row.updated_at = now


def _compute_scope_key(*, environment_id: str, principal_id: str | None) -> str:
    return f'environment:{environment_id}' if principal_id is None else f'grant:{environment_id}:{principal_id}'


def _allocation_usage(row: EnvironmentAllocationRow) -> ComputeUsage:
    return ComputeUsage(
        manager_count=row.manager_count,
        cpu_millicores=row.cpu_millicores,
        memory_bytes=row.memory_bytes,
        slots=row.slots,
        active_batches=row.active_batches,
    )


def _compute_usage(row: EnvironmentResourceUsageRow) -> ComputeUsage:
    return ComputeUsage(
        manager_count=row.manager_count,
        cpu_millicores=row.cpu_millicores,
        memory_bytes=row.memory_bytes,
        slots=row.slots,
        active_batches=row.active_batches,
    )


def _plus(left: ComputeUsage, right: ComputeUsage) -> ComputeUsage:
    values = ComputeUsage(
        manager_count=left.manager_count + right.manager_count,
        cpu_millicores=left.cpu_millicores + right.cpu_millicores,
        memory_bytes=left.memory_bytes + right.memory_bytes,
        slots=left.slots + right.slots,
        active_batches=left.active_batches + right.active_batches,
    )
    try:
        validate_resource_totals(
            manager_count=values.manager_count,
            cpu_millicores=values.cpu_millicores,
            memory_bytes=values.memory_bytes,
            slots=values.slots,
            active_batches=values.active_batches,
            field='compute_usage',
        )
    except ResourceNormalizationError as exc:
        raise QuotaExceededError('compute_usage_storage_range_exceeded') from exc
    return values


def _minus(left: ComputeUsage, right: ComputeUsage) -> ComputeUsage:
    values = ComputeUsage(
        manager_count=left.manager_count - right.manager_count,
        cpu_millicores=left.cpu_millicores - right.cpu_millicores,
        memory_bytes=left.memory_bytes - right.memory_bytes,
        slots=left.slots - right.slots,
        active_batches=left.active_batches - right.active_batches,
    )
    if (
        min(
            values.manager_count,
            values.cpu_millicores,
            values.memory_bytes,
            values.slots,
            values.active_batches,
        )
        < 0
    ):
        raise RuntimeError('environment_resource_usage_underflow')
    return values


def _write_compute_usage(row: EnvironmentResourceUsageRow, usage: ComputeUsage, *, now: datetime) -> None:
    row.manager_count = usage.manager_count
    row.cpu_millicores = usage.cpu_millicores
    row.memory_bytes = usage.memory_bytes
    row.slots = usage.slots
    row.active_batches = usage.active_batches
    row.version += 1
    row.updated_at = now


def _enforce_compute_limits(usage: ComputeUsage, limits: dict[str, Any], *, scope: str) -> None:
    dimensions = {
        'manager_count': usage.manager_count,
        'cpu_millicores': usage.cpu_millicores,
        'memory_bytes': usage.memory_bytes,
        'slots': usage.slots,
        'active_batches': usage.active_batches,
    }
    for dimension, value in dimensions.items():
        limit = _compute_limit(limits, dimension)
        if limit is not None and value > limit:
            raise QuotaExceededError(f'{scope}_{dimension}_limit_exceeded')


def _compute_limit(limits: dict[str, Any], dimension: str) -> int | None:
    aliases = {
        'manager_count': ('manager_count', 'managers'),
        'cpu_millicores': ('cpu_millicores',),
        'memory_bytes': ('memory_bytes',),
        'slots': ('slots',),
        'active_batches': ('active_batches',),
    }
    for key in aliases[dimension]:
        if key in limits:
            return int(limits[key])
    if dimension == 'cpu_millicores' and 'cpu_cores' in limits:
        return int(float(limits['cpu_cores']) * 1000)
    if dimension == 'memory_bytes' and 'memory_gib' in limits:
        return int(float(limits['memory_gib']) * 1024**3)
    return None
