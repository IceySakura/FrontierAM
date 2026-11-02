"""A leased, failure-isolated producer of complete status snapshots."""

from __future__ import annotations

import asyncio
import logging
import time

from collections import deque
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from nexus.distributed.config import DistributedConfig
from nexus.distributed.control_plane.leases import ShardLeaseCoordinator
from nexus.distributed.gateways.common.serde import to_jsonable
from nexus.distributed.observability.database import install_database_metrics
from nexus.distributed.observability.telemetry import PlatformMetrics
from nexus.distributed.persistence.db import create_engine, create_session_factory
from nexus.distributed.persistence.orm import EnvironmentRow, ObservabilitySnapshotRow
from nexus.distributed.repositories.observability import COMMAND_OPERATIONS, STATUS_GROUPS, ObservabilityRepository
from nexus.distributed.repositories.shard_lease import ShardLeaseFenceStatus, ShardLeaseRepository
from nexus.distributed.utils.clock import utcnow
from nexus.utils.platform_metrics import metric_scope


logger = logging.getLogger(__name__)
LEASE_GROUP = 'observability-snapshot'
SNAPSHOT_GAUGES = (
    'snapshot_age',
    'observer_success_timestamp',
    'active_executions',
    'queue_depth',
    'queue_oldest_age',
    'slots',
    'objects_by_state',
    'object_oldest_age',
    'stage_overdue',
    'stage_unknown_budget',
    'batch_progress_lag',
    'pending_commands',
    'command_oldest_age',
    'resource_managers',
    'resource_cpu_millicores',
    'resource_memory_bytes',
    'resource_reserved_slots',
    'resource_active_batches',
    'manager_gap',
    'manager_heartbeat_age',
    'unowned_shards',
    'database_server',
)


class StatusSampler:
    def __init__(
        self,
        config: DistributedConfig,
        *,
        session_factory: async_sessionmaker | None = None,
        metrics: PlatformMetrics | None = None,
        clock: Callable[[], datetime] = utcnow,
        owner_id: str = 'status-sampler',
    ) -> None:
        self.config = config
        self.clock = clock
        self._engine = None
        if session_factory is None:
            pool_config = config.model_copy(update={'db_pool_size': 1, 'db_max_overflow': 0, 'db_pool_timeout': 1})
            self._engine = create_engine(pool_config)
            session_factory = create_session_factory(self._engine)
        self.session_factory = session_factory
        self.metrics = metrics or PlatformMetrics(
            config.metrics, service='status-sampler', instance_id='nexus-cloud-snapshot'
        )
        install_database_metrics(self.session_factory, self.metrics, source='snapshot_db')
        self.lease = ShardLeaseCoordinator(
            session_factory=session_factory,
            shard_count=1,
            owner_id=owner_id,
            shard_group=LEASE_GROUP,
            lease_ttl=timedelta(seconds=60),
            renew_interval=10,
            discover_interval=5,
            ownership_limit=1,
        )
        self._expires_at: datetime | None = None
        self._stop = asyncio.Event()
        self._pending: deque[tuple[str | None, str]] = deque()
        self._scope_cursor: str | None = None
        self._samples: dict[tuple[str | None, str], tuple[datetime, dict[str, Any]]] = {}
        self._closed = False
        self._tick_lock = asyncio.Lock()
        for name in SNAPSHOT_GAUGES:
            self.metrics.set_gauge_callback(name, lambda metric=name: self.gauge_values(metric))

    def owns_lease(self) -> bool:
        return not self._stop.is_set() and self._expires_at is not None and self.clock() < self._expires_at

    async def refresh_lease(self) -> bool:
        try:
            owned = await self.lease.refresh(self.clock())
            self._expires_at = self.lease.valid_until if owned else None
            if not owned:
                self._samples = {}
            return bool(owned)
        except Exception as exc:
            self._expires_at = None
            self.lease.invalidate()
            logger.warning('Status sampler lease unavailable: error_type=%s', type(exc).__name__)
            return False

    async def _plan(self) -> None:
        await self._cleanup_deleted_scopes()
        async with self.session_factory() as session:
            query = select(EnvironmentRow.environment_id).order_by(EnvironmentRow.environment_id).limit(64)
            if self._scope_cursor is not None:
                query = query.where(EnvironmentRow.environment_id > self._scope_cursor)
            environments = list((await ObservabilityRepository(session, self.config).query(query)).scalars())
        self._scope_cursor = environments[-1] if len(environments) == 64 else None
        for group in STATUS_GROUPS:
            self._pending.append((None, group))
            if group != 'leases':
                self._pending.extend((environment, group) for environment in environments)

    async def _cleanup_deleted_scopes(self) -> None:
        async with self.session_factory() as session:
            missing = (
                select(ObservabilitySnapshotRow.scope_id)
                .outerjoin(
                    EnvironmentRow,
                    EnvironmentRow.environment_id == ObservabilitySnapshotRow.scope_id,
                )
                .where(
                    ObservabilitySnapshotRow.scope_type == 'environment',
                    EnvironmentRow.environment_id.is_(None),
                )
                .distinct()
                .limit(64)
            )
            scopes = list((await ObservabilityRepository(session, self.config).query(missing)).scalars())
        if not scopes:
            return
        async with self.session_factory() as session:
            fence = await ShardLeaseRepository(session, shard_count=1).fence(
                owner_id=self.lease.owner_id,
                shard_group=LEASE_GROUP,
                shard_ids=[0],
                now=self.clock(),
            )
            if fence != ShardLeaseFenceStatus.VALID:
                return
            await session.execute(
                delete(ObservabilitySnapshotRow).where(
                    ObservabilitySnapshotRow.scope_type == 'environment',
                    ObservabilitySnapshotRow.scope_id.in_(scopes),
                    ~select(EnvironmentRow.environment_id)
                    .where(
                        EnvironmentRow.environment_id == ObservabilitySnapshotRow.scope_id,
                    )
                    .exists(),
                )
            )
            await session.commit()
        self._samples = {key: value for key, value in self._samples.items() if key[0] not in scopes}

    async def tick(self) -> int:
        with metric_scope(self.metrics):
            return await self._tick()

    async def _tick(self) -> int:
        if self._tick_lock.locked() or not self.config.status_sampling.enabled:
            return 0
        async with self._tick_lock:
            if not await self.refresh_lease():
                return 0
            deadline = time.monotonic() + self.config.status_sampling.round_timeout_seconds
            if not self._pending:
                await self._plan()
            completed = 0
            while self._pending and self.owns_lease() and time.monotonic() < deadline:
                environment, group = self._pending.popleft()
                started = self.clock()
                timer = time.monotonic()
                values = None
                error_code = None
                try:
                    async with self.session_factory() as session:
                        values = await asyncio.wait_for(
                            ObservabilityRepository(session, self.config).collect_group(
                                group, now=started, environment_id=environment
                            ),
                            timeout=min(
                                self.config.status_sampling.group_timeout_seconds,
                                max(0.001, deadline - time.monotonic()),
                            ),
                        )
                except Exception as exc:
                    error_code = (
                        'query_timeout'
                        if isinstance(exc, (TimeoutError, asyncio.TimeoutError))
                        else 'collection_failed'
                    )
                    logger.warning('Status collection failed: group=%s error_type=%s', group, type(exc).__name__)
                if not self.owns_lease():
                    break
                try:
                    published = await asyncio.wait_for(
                        self.publish(environment, group, started, values, error_code),
                        timeout=min(
                            self.config.status_sampling.query_timeout_seconds, max(0.001, deadline - time.monotonic())
                        ),
                    )
                except Exception as exc:
                    published = False
                    logger.warning('Status snapshot publish failed: group=%s error_type=%s', group, type(exc).__name__)
                if published and values is not None:
                    cutoff = self.clock() - timedelta(seconds=self.config.status_sampling.stale_after_seconds * 2)
                    self._samples = {key: value for key, value in self._samples.items() if value[0] >= cutoff}
                    self._samples[(environment, group)] = (started, values)
                    while len(self._samples) > 1024:
                        self._samples.pop(next(iter(self._samples)))
                    completed += 1
                labels = {
                    'environment': environment or 'all',
                    'category': group,
                    'outcome': 'success' if published and values is not None else 'error',
                }
                self.metrics.record('observer_runs', attributes=labels)
                self.metrics.record('observer_duration_distribution', time.monotonic() - timer, labels)
            return completed

    async def publish(
        self, environment: str | None, group: str, sampled_at: datetime, values: dict | None, error_code: str | None
    ) -> bool:
        if not self.owns_lease():
            return False
        now = self.clock()
        scope_type, scope_id = ('environment', environment) if environment is not None else ('global', '*')
        async with self.session_factory() as session:
            fence = await ShardLeaseRepository(session, shard_count=1).fence(
                owner_id=self.lease.owner_id,
                shard_group=LEASE_GROUP,
                shard_ids=[0],
                now=now,
            )
            if fence != ShardLeaseFenceStatus.VALID:
                self._expires_at = None
                self.lease.invalidate()
                self._samples = {}
                return False
            if environment is not None and await session.get(EnvironmentRow, environment) is None:
                await session.execute(
                    delete(ObservabilitySnapshotRow).where(
                        ObservabilitySnapshotRow.scope_type == 'environment',
                        ObservabilitySnapshotRow.scope_id == environment,
                    )
                )
                await session.commit()
                self._samples.pop((environment, group), None)
                return False
            row = await session.get(ObservabilitySnapshotRow, (scope_type, scope_id, group))
            if row is not None and row.last_attempt_at > sampled_at:
                return False
            if row is None:
                row = ObservabilitySnapshotRow(scope_type=scope_type, scope_id=scope_id, metric_group=group, version=0)
                session.add(row)
            row.last_attempt_at = sampled_at
            row.publisher_id = self.lease.owner_id
            row.version += 1
            row.collection_status = 'ready' if values is not None else 'error'
            row.last_error_code = error_code
            if values is not None:
                row.values_json = to_jsonable(values)
                row.sampled_at = sampled_at
            await session.commit()
        return True

    def gauge_values(self, metric: str):
        now = self.clock()
        if not self.owns_lease():
            return []
        result = []
        for (environment, group), (sampled, values) in tuple(self._samples.items()):
            age = (now - sampled).total_seconds()
            labels = {
                'environment': environment or 'all',
                'category': group,
                'source': 'control_db',
                'instance_id': 'nexus-cloud-snapshot',
            }
            if metric == 'snapshot_age':
                result.append((max(0, age), labels))
            elif metric == 'observer_success_timestamp':
                result.append(((sampled - datetime(1970, 1, 1)).total_seconds(), labels))
            elif 0 <= age <= self.config.status_sampling.stale_after_seconds:
                result.extend(_group_gauges(metric, group, values, labels))
        return result

    async def _heartbeat(self) -> None:
        while not self._stop.is_set():
            with metric_scope(self.metrics):
                await self.refresh_lease()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=5)

    async def run_forever(self) -> None:
        heartbeat = asyncio.create_task(self._heartbeat(), name='status-sampler-lease')
        try:
            while not self._stop.is_set():
                try:
                    await self.tick()
                except Exception as exc:
                    logger.warning('Status sampler failed: error_type=%s', type(exc).__name__)
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=self.config.status_sampling.interval_seconds)
        finally:
            self._stop.set()
            self._expires_at = None
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            await self.close()

    def stop(self) -> None:
        self._stop.set()
        self._expires_at = None

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop()
        self._samples = {}
        with suppress(Exception):
            await asyncio.wait_for(self.lease.release(), timeout=self.config.status_sampling.group_timeout_seconds)
        await self.metrics.close()
        if self._engine is not None:
            await self._engine.dispose()


def _group_gauges(metric, group, values, labels):
    if group == 'inventory':
        if metric == 'database_server':
            server = values.get('database_server', {})
            if server.get('data_status') == 'ready':
                return [(server[kind], {**labels, 'kind': kind}) for kind in ('connected', 'running', 'limit')]
            return []
        if metric == 'manager_heartbeat_age':
            age = values['liveness'].get('heartbeat_age_seconds')
            return [(age, {**labels, 'kind': 'freshest_healthy'})] if age is not None else []
        if metric == 'slots':
            return [
                (values['managers'][key], {**labels, 'kind': kind})
                for key, kind in [
                    ('capacity_total', 'total'),
                    ('reserved_slots', 'reserved'),
                    ('available_slots', 'available'),
                ]
            ]
        if metric == 'objects_by_state':
            return [
                (values['managers'][key], {**labels, 'kind': 'manager', 'state': key})
                for key in ('healthy', 'draining')
            ]
        resource_fields = {
            'resource_managers': 'manager_count',
            'resource_cpu_millicores': 'cpu_millicores',
            'resource_memory_bytes': 'memory_bytes',
            'resource_reserved_slots': 'slots',
            'resource_active_batches': 'active_batches',
        }
        if metric in resource_fields and values['resource_usage']['complete']:
            return [(values['resource_usage'][resource_fields[metric]], labels)]
    rows = values.get('items', [])
    states = {
        'allocations': ('requested', 'provisioning', 'ready', 'draining', 'deleting'),
        'builds': ('requested', 'building', 'failed', 'ready'),
        'uploads': ('pending', 'receiving', 'maintenance'),
    }
    if group in states and metric in {'objects_by_state', 'object_oldest_age'}:
        by_state = {row['state']: row for row in rows}
        key = 'count' if metric == 'objects_by_state' else 'oldest_age_seconds'
        return [
            (by_state.get(state, {}).get(key, 0), {**labels, 'state': state})
            for state in states[group]
            if by_state.get(state, {}).get(key, 0) is not None
        ]
    if group == 'queues':
        if metric == 'active_executions':
            counts = {row['state']: row['count'] for row in rows}
            return [
                (counts.get(state, 0), {**labels, 'state': state})
                for state in ('queued', 'assigned', 'starting', 'running', 'waiting_input', 'stopping')
            ]
        if metric == 'queue_depth':
            return [
                (sum(row[key] for row in rows), {**labels, 'kind': key})
                for key in ('schedulable', 'preparing', 'retry_wait')
            ]
        if metric == 'queue_oldest_age':
            if any(row['unknown_wait_start'] for row in rows):
                return []
            return [(max((row['oldest_wait_seconds'] or 0 for row in rows), default=0), labels)]
    if metric == 'stage_overdue' and group == 'queues':
        counts = {row['state']: row['overdue'] for row in rows}
        return [
            (counts.get(state, 0), {**labels, 'stage': state})
            for state in ('queued', 'assigned', 'starting', 'running', 'waiting_input', 'stopping')
        ]
    stage_fields = {
        'allocations': [
            ('provisioning_overdue', 'unknown_provisioning_budget', 'provisioning'),
            ('release_overdue', 'unknown_release_start', 'allocation_release'),
        ],
        'builds': [('overdue', 'unknown_budget', 'dependency_build'), ('release_overdue', None, 'builder_release')],
        'batches': [
            ('preparation_overdue', 'unknown_preparation_budget', 'batch_preparation'),
            ('finalization_overdue', None, 'batch_finalization'),
        ],
        'uploads': [('overdue', None, 'artifact_upload')],
    }
    if metric in {'stage_overdue', 'stage_unknown_budget'} and group in stage_fields:
        result = []
        for field, unknown_field, stage in stage_fields[group]:
            unknown = sum(row.get(unknown_field, 0) for row in rows) if unknown_field else 0
            if metric == 'stage_unknown_budget':
                result.append((unknown, {**labels, 'stage': stage}))
            elif not unknown:
                result.append((sum(row[field] for row in rows), {**labels, 'stage': stage}))
        return result
    if group == 'leases' and metric == 'unowned_shards':
        return [(row['unowned'], {**labels, 'component': row['component']}) for row in rows]
    if group == 'allocations' and metric == 'manager_gap':
        return [
            (
                sum(row['manager_gap'] for row in rows),
                labels,
            )
        ]
    if group == 'commands':
        if metric == 'pending_commands':
            counts = {(row['operation'], row['state']): row['count'] for row in rows}
            return [
                (counts.get((operation, state), 0), {**labels, 'operation': operation, 'state': state})
                for operation in COMMAND_OPERATIONS
                for state in ('pending', 'claimed')
            ]
        if metric == 'command_oldest_age':
            return [
                (
                    max((row['oldest_age_seconds'] or 0 for row in rows if row['operation'] == operation), default=0),
                    {**labels, 'operation': operation},
                )
                for operation in COMMAND_OPERATIONS
            ]
    if group == 'batches' and metric == 'batch_progress_lag':
        return [(max((row['oldest_progress_age_seconds'] or 0 for row in rows), default=0), labels)]
    return []
