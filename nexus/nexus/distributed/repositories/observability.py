"""Bounded scalar queries for live inventory and persisted status groups."""

from __future__ import annotations

import asyncio

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, case, func, or_, select, text, union
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.distributed.config import DistributedConfig
from nexus.distributed.instance_manager.liveness import manager_is_healthy_clause
from nexus.distributed.persistence.orm import (
    ArtifactUploadRow,
    AssignmentRow,
    BatchRunRow,
    DependencyBuildRow,
    EnvironmentAllocationRow,
    EnvironmentResourceUsageRow,
    EnvironmentRow,
    ExecutionCommandRow,
    ExecutionRow,
    InstanceManagerLivenessRow,
    InstanceManagerRow,
    ObservabilitySnapshotRow,
    ShardLeaseRow,
)


STATUS_GROUPS = ('inventory', 'queues', 'allocations', 'builds', 'batches', 'commands', 'uploads', 'leases')
COMMAND_OPERATIONS = ('stop.request', 'input.submit', 'other')


class ObservabilityRepository:
    def __init__(self, session: AsyncSession, config: DistributedConfig):
        self.session = session
        self.config = config

    async def query(self, statement):
        timeout = self.config.status_sampling.query_timeout_seconds
        statement = statement.prefix_with(f'/*+ MAX_EXECUTION_TIME({max(1, int(timeout * 1000))}) */', dialect='mysql')
        return await asyncio.wait_for(self.session.execute(statement), timeout=timeout)

    def live_managers(self, *, now: datetime, environment_id: str | None = None, pool_id: str | None = None):
        cutoff = now - self.config.manager_healthy_ttl
        manager = InstanceManagerRow
        liveness = InstanceManagerLivenessRow
        # Both recent-heartbeat indexes are bounded by live activity, including during a mixed-version rollout.
        recent = union(
            select(manager.manager_id).where(manager.last_heartbeat_at >= cutoff),
            select(liveness.manager_id).where(liveness.last_heartbeat_at >= cutoff),
        ).subquery()
        live_query = (
            select(
                manager.manager_id,
                manager.session_id,
                manager.environment_id,
                manager.pool_id,
                manager.allocation_id,
                manager.owner_principal_id,
                manager.allocation_spec_hash,
                manager.workload,
                case(
                    (liveness.last_heartbeat_at > manager.last_heartbeat_at, liveness.last_heartbeat_at),
                    else_=manager.last_heartbeat_at,
                ).label('heartbeat_at'),
                manager.draining,
                case(
                    (
                        and_(liveness.manager_id.is_not(None), liveness.last_heartbeat_at >= manager.last_heartbeat_at),
                        liveness.capacity_total,
                    ),
                    else_=manager.capacity_total,
                ).label('capacity'),
            )
            .select_from(manager)
            .join(recent, recent.c.manager_id == manager.manager_id)
            .outerjoin(
                liveness, and_(liveness.manager_id == manager.manager_id, liveness.session_id == manager.session_id)
            )
            .where(manager_is_healthy_clause(cutoff))
        )
        if environment_id is not None:
            live_query = live_query.where(manager.environment_id == environment_id)
        if pool_id is not None:
            live_query = live_query.where(manager.pool_id == pool_id)
        return live_query

    async def live_inventory(
        self, *, now: datetime, environment_id: str | None = None, pool_id: str | None = None
    ) -> dict[str, Any]:
        assignment = AssignmentRow
        live_query = self.live_managers(now=now, environment_id=environment_id, pool_id=pool_id)
        live = live_query.subquery()
        reserved = (
            select(assignment.manager_id, func.sum(assignment.slot_cost).label('slots'))
            .join(
                live,
                and_(live.c.manager_id == assignment.manager_id, live.c.session_id == assignment.manager_session_id),
            )
            .where(assignment.status.in_(('pending', 'active')))
            .group_by(assignment.manager_id)
            .subquery()
        )
        used = func.coalesce(reserved.c.slots, 0)
        row = (
            await self.query(
                select(
                    func.count(live.c.manager_id),
                    func.coalesce(func.sum(case((live.c.draining.is_(True), 1), else_=0)), 0),
                    func.coalesce(func.sum(live.c.capacity), 0),
                    func.coalesce(func.sum(used), 0),
                    func.coalesce(
                        func.sum(
                            case(
                                (and_(live.c.draining.is_(False), live.c.capacity > used), live.c.capacity - used),
                                else_=0,
                            )
                        ),
                        0,
                    ),
                    func.max(live.c.heartbeat_at),
                )
                .select_from(live)
                .outerjoin(reserved, reserved.c.manager_id == live.c.manager_id)
            )
        ).one()
        healthy, draining, capacity, reserved_slots, available = map(int, row[:5])
        usage = await self.resource_usage(environment_id=environment_id)
        return {
            'schema_version': 2,
            'source': 'live',
            'queried_at': now,
            'data_status': 'ready' if usage['complete'] else 'partial',
            'scope': {'environment_id': environment_id, 'pool_id': pool_id},
            'managers': {
                'healthy': healthy,
                'draining': draining,
                'capacity_total': capacity,
                'reserved_slots': reserved_slots,
                'available_slots': available,
            },
            'resource_usage': usage,
            'liveness': {
                'last_heartbeat_at': row[5],
                'heartbeat_age_seconds': _age(now, row[5]),
                'heartbeat_interval_seconds': self.config.manager_heartbeat_interval,
                'healthy_ttl_seconds': self.config.manager_healthy_ttl.total_seconds(),
            },
        }

    async def resource_usage(self, *, environment_id: str | None) -> dict[str, Any]:
        usage = EnvironmentResourceUsageRow
        fields = ('manager_count', 'cpu_millicores', 'memory_bytes', 'slots', 'active_batches')
        if environment_id is not None:
            # Scope keys are the same primary keys maintained by QuotaRepository.
            statement = select(*(getattr(usage, field) for field in fields)).where(
                usage.scope_key == f'environment:{environment_id}'
            )
            row = (await self.query(statement)).first()
            values = dict(zip(fields, map(int, row), strict=True)) if row is not None else dict.fromkeys(fields)
            return {
                'scope_type': 'environment',
                'environment_id': environment_id,
                'complete': row is not None,
                **values,
            }
        statement = (
            select(
                func.count(EnvironmentRow.environment_id),
                func.count(usage.scope_key),
                *(func.coalesce(func.sum(getattr(usage, field)), 0) for field in fields),
            )
            .select_from(EnvironmentRow)
            .outerjoin(
                usage, and_(usage.environment_id == EnvironmentRow.environment_id, usage.scope_type == 'environment')
            )
        )
        row = (await self.query(statement)).one()
        complete = row[0] == row[1]
        return {
            'scope_type': 'environment',
            'environment_id': None,
            'complete': complete,
            'covered_environments': int(row[1]),
            'total_environments': int(row[0]),
            **(dict(zip(fields, map(int, row[2:]), strict=True)) if complete else dict.fromkeys(fields)),
        }

    async def database_server(self) -> dict[str, Any]:
        if self.session.get_bind().dialect.name != 'mysql':
            return {'data_status': 'not_supported'}

        async def read():
            rows = (
                await self.session.execute(
                    text("SHOW GLOBAL STATUS WHERE Variable_name IN ('Threads_connected', 'Threads_running')")
                )
            ).all()
            counters = {str(name).lower(): int(value) for name, value in rows}
            limit = (await self.session.execute(text('SELECT @@GLOBAL.max_connections'))).scalar_one()
            return {
                'data_status': 'ready',
                'connected': counters['threads_connected'],
                'running': counters['threads_running'],
                'limit': int(limit),
            }

        try:
            return await asyncio.wait_for(read(), timeout=self.config.status_sampling.query_timeout_seconds)
        except Exception as exc:
            await self.session.rollback()
            return {'data_status': 'error', 'error_code': type(exc).__name__}

    async def collect_group(self, group: str, *, now: datetime, environment_id: str | None) -> dict[str, Any]:
        if group == 'inventory':
            values = await self.live_inventory(now=now, environment_id=environment_id)
            if environment_id is None:
                # Server counters belong to the database as a whole and are sampled only in the background.
                values['database_server'] = await self.database_server()
            return values
        if group == 'queues':
            return await self.queues(now=now, environment_id=environment_id)
        if group == 'commands':
            command = ExecutionCommandRow
            operation = case((command.type.in_(COMMAND_OPERATIONS[:-1]), command.type), else_='other')
            query = (
                select(operation, command.status, func.count(), func.min(command.created_at))
                .where(command.status.in_(('pending', 'claimed')))
                .group_by(operation, command.status)
            )
            if environment_id is not None:
                query = query.join(ExecutionRow, ExecutionRow.execution_id == command.execution_id).where(
                    ExecutionRow.environment_id == environment_id
                )
            rows = (await self.query(query)).all()
            return {
                'items': [
                    {'operation': kind, 'state': state, 'count': int(count), 'oldest_age_seconds': _age(now, oldest)}
                    for kind, state, count, oldest in rows
                ]
            }
        if group == 'leases':
            rows = (
                await self.query(
                    select(
                        ShardLeaseRow.shard_group,
                        func.count(),
                        func.sum(case((ShardLeaseRow.lease_expires_at > now, 1), else_=0)),
                    )
                    .where(ShardLeaseRow.shard_group.in_(('execution-scheduler', 'execution-reconciler')))
                    .where(ShardLeaseRow.shard_id >= 0, ShardLeaseRow.shard_id < self.config.shard_count)
                    .group_by(ShardLeaseRow.shard_group)
                )
            ).all()
            active_by_group = {kind: int(active or 0) for kind, _total, active in rows}
            return {
                'items': [
                    {
                        'component': kind,
                        'total': self.config.shard_count,
                        'active': active_by_group.get(kind, 0),
                        'unowned': self.config.shard_count - active_by_group.get(kind, 0),
                    }
                    for kind in ('execution-scheduler', 'execution-reconciler')
                ]
            }
        if group == 'allocations':
            allocation = EnvironmentAllocationRow
            parent = ExecutionRow
            live = (
                self
                .live_managers(now=now, environment_id=environment_id)
                .where(InstanceManagerRow.draining.is_(False))
                .subquery()
            )
            ready = (
                select(live.c.allocation_id, func.count().label('count'))
                .join(
                    allocation,
                    and_(
                        allocation.allocation_id == live.c.allocation_id,
                        allocation.environment_id == live.c.environment_id,
                        allocation.owner_principal_id == live.c.owner_principal_id,
                        allocation.pool_id == live.c.pool_id,
                        allocation.workload == live.c.workload,
                        allocation.allocation_spec_hash == live.c.allocation_spec_hash,
                    ),
                )
                .group_by(live.c.allocation_id)
                .subquery()
            )
            pooled = (
                select(live.c.environment_id, live.c.pool_id, func.count().label('count'))
                .group_by(live.c.environment_id, live.c.pool_id)
                .subquery()
            )
            effective_ready = case(
                (EnvironmentRow.allocation_type == 'predefined', func.coalesce(pooled.c.count, 0)),
                else_=func.coalesce(ready.c.count, 0),
            )
            terminal = parent.status.in_(('succeeded', 'failed', 'lost', 'expired', 'killed', 'cancelled'))
            releasing = or_(
                parent.execution_id.is_(None),
                allocation.status.in_(('draining', 'deleting')),
                and_(
                    terminal, or_(BatchRunRow.parent_execution_id.is_(None), BatchRunRow.control_state == 'finalized')
                ),
            )
            release_started = func.coalesce(allocation.release_requested_at, parent.finished_at)
            cutoff = now - timedelta(seconds=self.config.status_sampling.release_grace_seconds)
            provisioning = allocation.status.in_(('requested', 'provisioning'))
            query = (
                select(
                    allocation.status,
                    func.count(),
                    func.min(allocation.created_at),
                    func.sum(case((~releasing, allocation.desired_manager_count), else_=0)),
                    func.sum(case((~releasing, effective_ready), else_=0)),
                    func.sum(case((allocation.operation_lease_until < now, 1), else_=0)),
                    func.sum(case((and_(provisioning, allocation.provisioning_deadline_at < now), 1), else_=0)),
                    func.sum(case((and_(provisioning, allocation.provisioning_deadline_at.is_(None)), 1), else_=0)),
                    func.sum(case((and_(releasing, release_started < cutoff), 1), else_=0)),
                    func.sum(case((and_(releasing, release_started.is_(None)), 1), else_=0)),
                    func.sum(
                        case(
                            (
                                and_(~releasing, allocation.desired_manager_count > effective_ready),
                                allocation.desired_manager_count - effective_ready,
                            ),
                            else_=0,
                        )
                    ),
                )
                .select_from(allocation)
                .outerjoin(ready, ready.c.allocation_id == allocation.allocation_id)
                .outerjoin(
                    pooled,
                    and_(pooled.c.environment_id == allocation.environment_id, pooled.c.pool_id == allocation.pool_id),
                )
                .outerjoin(EnvironmentRow, EnvironmentRow.environment_id == allocation.environment_id)
                .outerjoin(parent, parent.execution_id == allocation.top_level_execution_id)
                .outerjoin(BatchRunRow, BatchRunRow.parent_execution_id == allocation.top_level_execution_id)
                .where(allocation.status.in_(('requested', 'provisioning', 'ready', 'draining', 'deleting')))
                .group_by(allocation.status)
            )
            if environment_id is not None:
                query = query.where(allocation.environment_id == environment_id)
            rows = (await self.query(query)).all()
            return {
                'release_grace_seconds': self.config.status_sampling.release_grace_seconds,
                'release_source': 'control_plane_confirmation',
                'items': [
                    {
                        'state': state,
                        'count': int(count),
                        'oldest_age_seconds': _age(now, oldest),
                        'desired': int(desired or 0),
                        'ready': int(ready_count or 0),
                        'operation_lease_expired': int(expired or 0),
                        'provisioning_overdue': int(overdue or 0),
                        'unknown_provisioning_budget': int(unknown or 0),
                        'release_overdue': int(release or 0),
                        'unknown_release_start': int(unknown_release or 0),
                        'manager_gap': int(gap or 0),
                    }
                    for state, count, oldest, desired, ready_count, expired, overdue, unknown, release, unknown_release, gap in rows
                ],
            }
        if group == 'builds':
            build = DependencyBuildRow
            building = build.status == 'building'
            cleanup = and_(build.status == 'ready', build.next_retry_at.is_not(None))
            cutoff = now - timedelta(seconds=self.config.status_sampling.release_grace_seconds)
            query = (
                select(
                    build.status,
                    func.count(),
                    func.min(
                        case(
                            (building, build.started_at),
                            (build.status.in_(('failed', 'ready')), build.finished_at),
                            else_=build.created_at,
                        )
                    ),
                    func.sum(case((and_(building, build.deadline_at < now), 1), else_=0)),
                    func.sum(case((and_(building, build.deadline_at.is_(None)), 1), else_=0)),
                    func.sum(case((and_(cleanup, build.finished_at < cutoff), 1), else_=0)),
                )
                .where(
                    or_(
                        build.status.in_(('requested', 'building')),
                        and_(build.status.in_(('failed', 'ready')), build.next_retry_at.is_not(None)),
                    )
                )
                .group_by(build.status)
            )
            if environment_id is not None:
                query = query.where(build.environment_id == environment_id)
            rows = (await self.query(query)).all()
            return {
                'release_grace_seconds': self.config.status_sampling.release_grace_seconds,
                'release_source': 'control_plane_confirmation',
                'items': [
                    {
                        'state': state,
                        'count': int(count),
                        'oldest_age_seconds': _age(now, oldest),
                        'overdue': int(overdue or 0),
                        'unknown_budget': int(unknown or 0),
                        'release_overdue': int(release or 0),
                    }
                    for state, count, oldest, overdue, unknown, release in rows
                ],
            }
        if group == 'batches':
            batch = BatchRunRow
            parent = ExecutionRow
            active = parent.status.in_(('queued', 'assigned', 'starting', 'running', 'stopping'))
            preparing = and_(active, batch.generation_status.in_(('pending', 'publishing', 'materializing')))
            finalizing = and_(active, or_(batch.control_state == 'stopping', batch.submission_closed.is_(True)))
            deadline = case(
                (parent.stop_deadline_at.is_(None), parent.execution_expires_at),
                (parent.execution_expires_at.is_(None), parent.stop_deadline_at),
                (parent.stop_deadline_at < parent.execution_expires_at, parent.stop_deadline_at),
                else_=parent.execution_expires_at,
            )
            query = (
                select(
                    batch.control_state,
                    batch.generation_status,
                    func.count(),
                    func.min(batch.last_progress_at),
                    func.sum(batch.active_items),
                    func.sum(batch.terminal_items),
                    func.sum(case((and_(preparing, batch.preparation_deadline_at < now), 1), else_=0)),
                    func.sum(case((and_(preparing, batch.preparation_deadline_at.is_(None)), 1), else_=0)),
                    func.sum(case((and_(finalizing, deadline < now), 1), else_=0)),
                    func.sum(case((and_(finalizing, deadline.is_(None)), 1), else_=0)),
                )
                .join(parent, parent.execution_id == batch.parent_execution_id)
                .where(batch.control_state.in_(('active', 'stopping')))
                .group_by(batch.control_state, batch.generation_status)
            )
            if environment_id is not None:
                query = query.where(parent.environment_id == environment_id)
            rows = (await self.query(query)).all()
            return {
                'items': [
                    {
                        'state': state,
                        'stage': stage,
                        'count': int(count),
                        'oldest_progress_age_seconds': _age(now, progress),
                        'active_items': int(active_items or 0),
                        'terminal_items': int(terminal or 0),
                        'preparation_overdue': int(preparation_overdue or 0),
                        'unknown_preparation_budget': int(unknown_preparation or 0),
                        'finalization_overdue': int(finalization_overdue or 0),
                        'unbounded_finalization': int(unbounded or 0),
                    }
                    for state, stage, count, progress, active_items, terminal, preparation_overdue, unknown_preparation, finalization_overdue, unbounded in rows
                ]
            }
        if group == 'uploads':
            upload = ArtifactUploadRow
            query = select(
                upload.status,
                func.count(),
                func.min(upload.created_at),
                func.sum(case((upload.expires_at < now, 1), else_=0)),
            )
            query = query.where(upload.status.in_(('pending', 'receiving', 'maintenance'))).group_by(upload.status)
            if environment_id is not None:
                query = query.where(upload.environment_id == environment_id)
            rows = (await self.query(query)).all()
            return {
                'items': [
                    {
                        'state': state,
                        'count': int(count),
                        'oldest_age_seconds': _age(now, created),
                        'overdue': int(overdue or 0),
                    }
                    for state, count, created, overdue in rows
                ]
            }
        raise ValueError('unknown status group')

    async def queues(self, *, now: datetime, environment_id: str | None) -> dict[str, Any]:
        execution = ExecutionRow
        allocation = EnvironmentAllocationRow
        allocation_ready = or_(execution.allocation_id.is_(None), allocation.status == 'ready')
        ready = and_(execution.status == 'queued', allocation_ready, execution.next_schedule_at <= now)
        entered = func.coalesce(execution.queue_entered_at, case((execution.attempt_count == 0, execution.created_at)))
        eligible_since = case((allocation.ready_at > entered, allocation.ready_at), else_=entered)
        query = (
            select(
                execution.status,
                func.count(),
                func.sum(case((ready, 1), else_=0)),
                func.sum(case((and_(execution.status == 'queued', ~allocation_ready), 1), else_=0)),
                func.sum(
                    case(
                        (and_(execution.status == 'queued', allocation_ready, execution.next_schedule_at > now), 1),
                        else_=0,
                    )
                ),
                func.min(case((ready, eligible_since))),
                func.sum(case((and_(ready, eligible_since.is_(None)), 1), else_=0)),
                func.sum(
                    case(
                        (
                            or_(
                                and_(execution.status == 'queued', execution.queue_expires_at < now),
                                and_(
                                    execution.status == 'assigned',
                                    AssignmentRow.status == 'pending',
                                    AssignmentRow.created_at <= now - self.config.assign_ack_timeout,
                                ),
                                and_(
                                    execution.status == 'starting',
                                    AssignmentRow.status == 'active',
                                    AssignmentRow.started_at.is_(None),
                                    AssignmentRow.accepted_at <= now - self.config.start_timeout,
                                ),
                                and_(
                                    execution.status.in_(('starting', 'running', 'waiting_input')),
                                    execution.execution_expires_at < now,
                                ),
                                and_(execution.status == 'stopping', execution.stop_deadline_at < now),
                            ),
                            1,
                        ),
                        else_=0,
                    )
                ),
            )
            .select_from(execution)
            .outerjoin(allocation, allocation.allocation_id == execution.allocation_id)
            .outerjoin(AssignmentRow, AssignmentRow.assignment_id == execution.current_assignment_id)
            .where(execution.status.in_(('queued', 'assigned', 'starting', 'running', 'waiting_input', 'stopping')))
            .group_by(execution.status)
        )
        if environment_id is not None:
            query = query.where(execution.environment_id == environment_id)
        rows = (await self.query(query)).all()
        return {
            'items': [
                {
                    'state': state,
                    'count': int(count),
                    'schedulable': int(schedulable or 0),
                    'preparing': int(preparing or 0),
                    'retry_wait': int(retry or 0),
                    'oldest_wait_seconds': None if missing else _age(now, oldest),
                    'unknown_wait_start': int(missing or 0),
                    'overdue': int(overdue or 0),
                }
                for state, count, schedulable, preparing, retry, oldest, missing, overdue in rows
            ]
        }

    async def snapshots(self, *, now: datetime, environment_id: str | None) -> dict[str, Any]:
        scope_type, scope_id = ('environment', environment_id) if environment_id is not None else ('global', '*')
        query = select(ObservabilitySnapshotRow).where(
            ObservabilitySnapshotRow.scope_type == scope_type,
            ObservabilitySnapshotRow.scope_id == scope_id,
            ObservabilitySnapshotRow.metric_group.in_(STATUS_GROUPS),
        )
        rows = (await self.query(query)).scalars().all()
        expected = (
            STATUS_GROUPS if environment_id is None else tuple(group for group in STATUS_GROUPS if group != 'leases')
        )
        groups = {
            group: {
                'values': None,
                'sampled_at': None,
                'last_attempt_at': None,
                'age_seconds': None,
                'data_status': 'unavailable',
                'error_code': 'not_collected',
                'version': None,
            }
            for group in expected
        }
        for row in rows:
            if row.metric_group not in groups:
                continue
            age = _age(now, row.sampled_at)
            status = (
                'unavailable'
                if row.sampled_at is None or row.sampled_at > now
                else (
                    'stale'
                    if age is None or age > self.config.status_sampling.stale_after_seconds
                    else row.collection_status
                )
            )
            groups[row.metric_group] = {
                'values': row.values_json if status != 'unavailable' else None,
                'sampled_at': row.sampled_at,
                'last_attempt_at': row.last_attempt_at,
                'age_seconds': age,
                'data_status': status,
                'error_code': row.last_error_code,
                'version': row.version,
            }
        return {
            'schema_version': 2,
            'source': 'snapshot',
            'queried_at': now,
            'scope': {'environment_id': environment_id},
            'groups': groups,
            'data_status': 'unavailable'
            if all(v['data_status'] == 'unavailable' for v in groups.values())
            else ('ready' if all(v['data_status'] == 'ready' for v in groups.values()) else 'partial'),
        }


def _age(now: datetime, started: datetime | None) -> float | None:
    return max(0, (now - started).total_seconds()) if started is not None else None
