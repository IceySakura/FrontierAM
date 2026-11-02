"""Measure Scheduler capacity with independent processes and logical Managers.

The benchmark exercises the real repository, transaction, and shard-lease paths.
It never starts an InstanceManager process, Pod, or sandbox.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback

from datetime import timedelta
from pathlib import Path
from typing import Any, cast

from sqlalchemy import and_, case, func, insert, or_, select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import async_sessionmaker
from tests.stress.distributed.capacity_support import (
    DATABASE_GAUGES,
    LogicalManagerHeartbeatTarget,
    ManagerHeartbeatLoad,
    ProcessSampler,
    database_identity,
    database_status,
    database_status_report,
    join_processes,
    retry_tick,
    source_identity,
)

from nexus.distributed.config import DistributedConfig
from nexus.distributed.config.views import scheduler_config_view
from nexus.distributed.persistence.db import create_engine, create_session_factory, is_retryable_database_error
from nexus.distributed.environment.models import (
    EnvironmentAllocationStatus,
    EnvironmentAllocationType,
    EnvironmentStatus,
)
from nexus.distributed.execution.models import (
    ExecutionMode,
    ExecutionStatus,
    RouteStatus,
)
from nexus.distributed.persistence.orm import (
    AssignmentRow,
    EnvironmentAllocationRow,
    EnvironmentGrantRow,
    EnvironmentResourceUsageRow,
    EnvironmentRow,
    ExecutionEventRow,
    ExecutionRow,
    InstanceManagerLivenessRow,
    InstanceManagerRow,
    QuotaUsageRow,
    ShardLeaseRow,
    create_schema,
)
from nexus.distributed.repositories.allocation import (
    AllocationRepository,
    AllocationResources,
    CreateEnvironmentAllocation,
)
from nexus.distributed.repositories.manager import ManagerRepository
from nexus.distributed.repositories.quota import QuotaRepository
from nexus.distributed.control_plane.scheduler import SchedulerService
from nexus.distributed.utils.clock import utcnow
from nexus.distributed.utils.identity import canonical_sha256, stable_shard


SHARD_COUNT = 64
TOPOLOGIES = ((1, 64), (4, 16), (16, 4), (32, 2))
_SOURCE_FILES = (
    'nexus/distributed/manager_liveness.py',
    'nexus.distributed.persistence.orm.py',
    'nexus/distributed/repositories/manager.py',
    'nexus/distributed/repositories/quota.py',
    'nexus/distributed/repositories/scheduler.py',
    'nexus/distributed/repositories/shard_lease.py',
    'nexus.distributed.control_plane.scheduler.py',
    'nexus/distributed/shard_leases.py',
    'tests/stress/distributed/capacity_support.py',
    'tests/stress/distributed/run_control_plane_capacity.py',
)


def validate_topology(*, replicas: int, ownership_limit: int) -> None:
    if replicas < 1 or ownership_limit < 1 or replicas * ownership_limit != SHARD_COUNT:
        raise ValueError('replicas * ownership_limit must exactly cover all shards')


def validate_recovery_config(*, replicas: int, worker_index: int, restart_after_seconds: float) -> None:
    if restart_after_seconds < 0:
        raise ValueError('restart_after_seconds must be non-negative')
    if restart_after_seconds > 0 and not 0 <= worker_index < replicas:
        raise ValueError('restart worker index must identify a configured replica')


def _config(
    database_url: str,
    *,
    ownership_limit: int,
    task_count: int,
    batch_size: int,
    replicas: int,
    manager_heartbeat_interval: float = 5.0,
) -> DistributedConfig:
    quota_limit = max(task_count * 2, 1000)
    return DistributedConfig(
        database_url=database_url,
        db_pool_size=max(8, replicas * 2),
        db_max_overflow=max(8, replicas * 2),
        shard_count=SHARD_COUNT,
        manager_heartbeat_interval=manager_heartbeat_interval,
        manager_healthy_ttl=timedelta(seconds=15),
        manager_self_fence_timeout=timedelta(seconds=30),
        manager_dead_ttl=timedelta(seconds=45),
        default_user_quota={
            'queued_executions': quota_limit,
            'running_executions': quota_limit,
            'reserved_slots': quota_limit,
        },
        scheduler_scan_limit=batch_size,
        scheduler_max_assignments=batch_size,
        scheduler_manager_lock_limit=64,
        scheduler_shard_ownership_limit=ownership_limit,
        scheduler_shard_discover_interval=0.01,
        scheduler_shard_discover_jitter=0,
        scheduler_lease_ttl=timedelta(seconds=60),
        scheduler_lease_renew_interval=10,
        reconciler_batch_size=batch_size,
        reconciler_shard_ownership_limit=ownership_limit,
        reconciler_shard_discover_interval=0.01,
        reconciler_shard_discover_jitter=0,
        reconciler_lease_ttl=timedelta(seconds=60),
        reconciler_lease_renew_interval=10,
        reconciler_core_max_batches_per_tick=4,
        reconciler_core_time_budget=60,
        reconciler_core_timeout=120,
    )


async def _assert_empty_database(session_factory: async_sessionmaker) -> None:
    async with session_factory() as session:
        counts = [
            int(await session.scalar(select(func.count()).select_from(table)) or 0)
            for table in (ExecutionRow, InstanceManagerRow, EnvironmentAllocationRow, ShardLeaseRow)
        ]
    if any(counts):
        raise RuntimeError('capacity benchmark requires an empty dedicated database')


async def _register_logical_managers(
    session_factory: async_sessionmaker,
    *,
    manager_count: int,
    task_count: int,
    prefix: str,
) -> list[LogicalManagerHeartbeatTarget]:
    capacity = max(1, math.ceil(task_count * 2 / manager_count))
    total_capacity = capacity * manager_count
    identity = _allocation_identity(prefix)
    allocation_spec = _allocation_spec(prefix)
    now = utcnow()
    heartbeat_targets: list[LogicalManagerHeartbeatTarget] = []
    async with session_factory() as session:
        resource_limits = {
            'manager_count': manager_count * 2,
            'cpu_millicores': manager_count * 2000,
            'memory_bytes': manager_count * 2 * 1024**3,
            'slots': total_capacity * 2,
            'active_batches': 2,
        }
        session.add(
            EnvironmentRow(
                environment_id=identity['environment_id'],
                name=identity['environment_id'],
                allocation_type=EnvironmentAllocationType.DYNAMIC,
                provider_type='kubernetes',
                provider_config_ref='kubernetes/capacity-benchmark',
                resource_limits_json=resource_limits,
                default_timeouts_json={},
                config_version=1,
                status=EnvironmentStatus.ENABLED,
                created_by='capacity-benchmark',
                updated_by='capacity-benchmark',
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            EnvironmentGrantRow(
                principal_id=identity['owner_principal_id'],
                environment_id=identity['environment_id'],
                limits_json=resource_limits,
                enabled=True,
                version=1,
                created_by='capacity-benchmark',
                updated_by='capacity-benchmark',
                created_at=now,
                updated_at=now,
            )
        )
        session.add_all(
            [
                EnvironmentResourceUsageRow(
                    scope_key=f'environment:{identity["environment_id"]}',
                    scope_type='environment',
                    environment_id=identity['environment_id'],
                    principal_id=None,
                    manager_count=0,
                    cpu_millicores=0,
                    memory_bytes=0,
                    slots=0,
                    active_batches=0,
                    version=1,
                    updated_at=now,
                ),
                EnvironmentResourceUsageRow(
                    scope_key=(f'grant:{identity["environment_id"]}:{identity["owner_principal_id"]}'),
                    scope_type='grant',
                    environment_id=identity['environment_id'],
                    principal_id=identity['owner_principal_id'],
                    manager_count=0,
                    cpu_millicores=0,
                    memory_bytes=0,
                    slots=0,
                    active_batches=0,
                    version=1,
                    updated_at=now,
                ),
            ]
        )
        await session.flush()
        allocation_repository = AllocationRepository(session)
        await allocation_repository.create_allocation(
            CreateEnvironmentAllocation(
                allocation_id=identity['allocation_id'],
                top_level_execution_id=f'{prefix}-parent',
                owner_principal_id=identity['owner_principal_id'],
                environment_id=identity['environment_id'],
                pool_id=identity['pool_id'],
                workload='execution',
                desired_manager_count=manager_count,
                resources=AllocationResources(
                    manager_count=manager_count,
                    cpu_millicores=manager_count * 1000,
                    memory_bytes=manager_count * 1024**3,
                    slots=total_capacity,
                ),
                allocation_spec=allocation_spec,
                allocation_spec_hash=identity['allocation_spec_hash'],
            ),
            now=now,
        )
        await QuotaRepository(session, default_limits={}).reserve_allocation(
            identity['allocation_id'],
            now=now,
        )
        repository = ManagerRepository(session)
        for index in range(manager_count):
            manager_id = f'{prefix}-manager-{index:04d}'
            session_id = f'{prefix}-session-{index:04d}'
            registered = await repository.register_session(
                manager_id=manager_id,
                session_id=session_id,
                endpoint_url=f'http://127.0.0.1:1/{prefix}/{index}',
                supported_modes=[ExecutionMode.JOB],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                capacity_total=capacity,
                force_replace=False,
                now=now,
                environment_id=identity['environment_id'],
                pool_id=identity['pool_id'],
                workload='execution',
                labels={'workload': 'execution'},
                owner_principal_id=identity['owner_principal_id'],
                allocation_id=identity['allocation_id'],
                allocation_spec_hash=identity['allocation_spec_hash'],
            )
            if not registered:
                raise RuntimeError('logical Manager registration was fenced')
            heartbeat_targets.append(
                LogicalManagerHeartbeatTarget(
                    manager_id=manager_id,
                    session_id=session_id,
                    capacity_total=capacity,
                )
            )
        claim = await allocation_repository.claim_allocation(
            identity['allocation_id'],
            operation_owner='capacity-benchmark',
            operation_ttl=timedelta(seconds=30),
            now=now,
        )
        transitioned = await allocation_repository.transition(
            claim,
            status=EnvironmentAllocationStatus.READY,
            ready_manager_count=manager_count,
            provider_resource_ref={'kind': 'logical-capacity-benchmark'},
            now=now,
        )
        if not transitioned:
            raise RuntimeError('logical Allocation ready transition was fenced')
        await session.commit()
    return heartbeat_targets


def _allocation_spec(prefix: str) -> dict[str, Any]:
    return {
        'allocation_type': EnvironmentAllocationType.DYNAMIC,
        'benchmark_prefix': prefix,
    }


def _allocation_identity(prefix: str) -> dict[str, str]:
    return {
        'owner_principal_id': f'{prefix}-owner',
        'environment_id': f'{prefix}-environment',
        'pool_id': f'{prefix}-execution',
        'allocation_id': f'{prefix}-allocation',
        'allocation_spec_hash': canonical_sha256(_allocation_spec(prefix)),
    }


def _execution_values(
    *,
    prefix: str,
    task_count: int,
    expired: bool,
    allocation_backed: bool = True,
) -> list[dict[str, Any]]:
    now = utcnow()
    queue_expires_at = now - timedelta(seconds=1) if expired else None
    values = [
        {
            'execution_id': execution_id,
            'owner_principal_id': f'{prefix}-owner-{index % 16:02d}',
            'created_by': 'capacity-benchmark',
            'shard_id': stable_shard(execution_id, SHARD_COUNT),
            'mode': ExecutionMode.JOB,
            'spec_type': 'nexus_single',
            'instance_mode': 'asyncio',
            'spec_json': {'run_input': {'task_name': 'echo', 'task_config': {'index': index}}},
            'driver_config_json': {},
            'placement_json': {},
            'status': ExecutionStatus.QUEUED,
            'route_status': RouteStatus.NONE,
            'slot_cost': 1,
            'next_attempt_seq': 1,
            'attempt_count': 0,
            'max_attempts': 1,
            'next_event_seq': 1,
            'priority': 0,
            'next_schedule_at': now,
            'queue_expires_at': queue_expires_at,
            'version': 1,
            'created_at': now,
            'updated_at': now,
        }
        for index in range(task_count)
        for execution_id in (f'{prefix}-execution-{index:08d}',)
    ]
    if not allocation_backed:
        return values
    identity = _allocation_identity(prefix)
    placement = {
        'environment': identity['environment_id'],
        'pool': identity['pool_id'],
        'allocation_id': identity['allocation_id'],
        'labels': {'workload': 'execution'},
    }
    for value in values:
        value.update(
            owner_principal_id=identity['owner_principal_id'],
            environment_id=identity['environment_id'],
            pool_id=identity['pool_id'],
            allocation_id=identity['allocation_id'],
            workload='execution',
            allocation_spec_hash=identity['allocation_spec_hash'],
            placement_json=placement,
            first_schedulable_at=now,
        )
    return values


async def _insert_executions(
    session_factory: async_sessionmaker,
    *,
    prefix: str,
    task_count: int,
    expired: bool,
    allocation_backed: bool = True,
) -> None:
    values = _execution_values(
        prefix=prefix,
        task_count=task_count,
        expired=expired,
        allocation_backed=allocation_backed,
    )
    async with session_factory() as session:
        for offset in range(0, len(values), 1000):
            await session.execute(insert(ExecutionRow), values[offset : offset + 1000])
        await session.commit()


def _require_mysql(dialect_name: str) -> None:
    if dialect_name != 'mysql':
        raise ValueError(f'control-plane capacity benchmark requires MySQL, got {dialect_name}')


def _scheduler_config(args: argparse.Namespace) -> DistributedConfig:
    config = _config(
        args.database_url,
        ownership_limit=args.ownership_limit,
        task_count=args.task_count,
        batch_size=args.batch_size,
        replicas=args.replicas,
        manager_heartbeat_interval=args.manager_heartbeat_interval,
    )
    return config.model_copy(
        update={
            'db_pool_size': args.scheduler_pool_size,
            'db_max_overflow': args.scheduler_max_overflow,
            'scheduler_poll_interval': args.scheduler_poll_interval,
            'scheduler_lease_ttl': timedelta(seconds=args.shard_lease_ttl),
            'scheduler_lease_renew_interval': args.shard_lease_renew_interval,
        }
    )


def _heartbeat_config(args: argparse.Namespace) -> DistributedConfig:
    dead_ttl = max(
        args.manager_healthy_ttl * 2,
        args.manager_healthy_ttl + 60,
        args.manager_self_fence_timeout + 60,
    )
    return DistributedConfig(
        database_url=args.database_url,
        db_pool_size=args.heartbeat_pool_size,
        db_max_overflow=args.heartbeat_max_overflow,
        manager_healthy_ttl=timedelta(seconds=args.manager_healthy_ttl),
        manager_dead_ttl=timedelta(seconds=dead_ttl),
        manager_self_fence_timeout=timedelta(seconds=args.manager_self_fence_timeout),
    )


async def _setup_fixture(
    args: argparse.Namespace,
    *,
    prefix: str,
) -> list[LogicalManagerHeartbeatTarget]:
    config = DistributedConfig(database_url=args.database_url, db_pool_size=4, db_max_overflow=4)
    engine = create_engine(config)
    _require_mysql(engine.dialect.name)
    session_factory = create_session_factory(engine)
    try:
        await create_schema(engine)
        await _assert_empty_database(session_factory)
        targets = await _register_logical_managers(
            session_factory,
            manager_count=args.manager_count,
            task_count=args.task_count,
            prefix=prefix,
        )
        await _insert_executions(
            session_factory,
            prefix=prefix,
            task_count=args.task_count,
            expired=False,
        )
        return targets
    finally:
        await engine.dispose()


def _scheduler_process(
    index: int,
    args: argparse.Namespace,
    prefix: str,
    ready_queue: Any,
    result_queue: Any,
    start_event: Any,
    stop_event: Any,
) -> None:
    try:
        asyncio.run(
            _run_scheduler_process(
                index=index,
                args=args,
                prefix=prefix,
                ready_queue=ready_queue,
                result_queue=result_queue,
                start_event=start_event,
                stop_event=stop_event,
            )
        )
    except BaseException as error:
        result_queue.put(
            {
                'kind': 'scheduler_error',
                'index': index,
                'pid': os.getpid(),
                'error': f'{type(error).__name__}: {error}',
                'traceback': traceback.format_exc(),
            }
        )
        raise


def _create_scheduler_process(
    context: Any,
    *,
    index: int,
    args: argparse.Namespace,
    prefix: str,
    ready_queue: Any,
    result_queue: Any,
    start_event: Any,
    stop_event: Any,
) -> mp.Process:
    return context.Process(
        target=_scheduler_process,
        args=(index, args, prefix, ready_queue, result_queue, start_event, stop_event),
        name=f'nexus-scheduler-capacity-{index:02d}',
    )


async def _run_scheduler_process(
    *,
    index: int,
    args: argparse.Namespace,
    prefix: str,
    ready_queue: Any,
    result_queue: Any,
    start_event: Any,
    stop_event: Any,
) -> None:
    logging.basicConfig(level=logging.WARNING)
    config = _scheduler_config(args)
    engine = create_engine(config)
    session_factory = create_session_factory(engine)
    owner_id = f'{prefix}-s-{index:02d}-p{os.getpid()}'
    service = SchedulerService(
        session_factory=session_factory,
        config=scheduler_config_view(config),
        owner_id=owner_id,
    )
    cast(Any, service)._schedule_manager_wakeup = lambda _manager_id, _endpoint_url: None
    lease_task: asyncio.Task[None] | None = None
    closed = False
    try:
        owned: list[int] = []
        startup_deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < startup_deadline:

            async def refresh_leases() -> int:
                nonlocal owned
                owned = await service._shard_leases.refresh(utcnow())
                return len(owned)

            try:
                await retry_tick(refresh_leases)
            except DBAPIError as error:
                if not is_retryable_database_error(error):
                    raise
                await asyncio.sleep(0.05)
                continue
            if len(owned) == args.ownership_limit:
                break
            await asyncio.sleep(0.05)
        else:
            raise RuntimeError(f'Scheduler {index} did not acquire {args.ownership_limit} shards: {owned}')

        lease_task = asyncio.create_task(
            service._run_lease_heartbeat(),
            name=f'scheduler-capacity-lease:{index}',
        )
        ready_queue.put(
            {
                'kind': 'scheduler_ready',
                'index': index,
                'pid': os.getpid(),
                'owned_shards': owned,
            }
        )
        await asyncio.to_thread(start_event.wait)

        stats = {
            'cycles': 0,
            'idle_cycles': 0,
            'assigned': 0,
            'scanned': 0,
            'claimed': 0,
            'lock_contention': 0,
            'no_capacity': 0,
            'candidate_queries': 0,
        }
        while not stop_event.is_set():
            if lease_task.done():
                await lease_task
            changed = await retry_tick(service.tick_once)
            claim_stats = service._last_assignment_stats
            stats['cycles'] += 1
            stats['idle_cycles'] += int(changed == 0)
            stats['assigned'] += changed
            stats['scanned'] += claim_stats.scanned
            stats['claimed'] += claim_stats.claimed
            stats['lock_contention'] += claim_stats.lock_contention
            stats['no_capacity'] += claim_stats.no_capacity
            stats['candidate_queries'] += claim_stats.candidate_queries
            await asyncio.sleep(config.scheduler_poll_interval)

        service.stop()
        if lease_task is not None:
            await lease_task
        await service.close()
        closed = True
        result_queue.put(
            {
                'kind': 'scheduler_done',
                'index': index,
                'pid': os.getpid(),
                'stats': stats,
            }
        )
    finally:
        if not closed:
            service.stop()
            if lease_task is not None:
                await asyncio.gather(lease_task, return_exceptions=True)
            await service.close()
        await engine.dispose()


def _heartbeat_process(
    args: argparse.Namespace,
    targets: list[LogicalManagerHeartbeatTarget],
    ready_queue: Any,
    result_queue: Any,
    stop_event: Any,
) -> None:
    try:
        asyncio.run(_run_heartbeat_process(args, targets, ready_queue, result_queue, stop_event))
    except BaseException as error:
        result_queue.put(
            {
                'kind': 'heartbeat_error',
                'pid': os.getpid(),
                'error': f'{type(error).__name__}: {error}',
                'traceback': traceback.format_exc(),
            }
        )
        raise


async def _run_heartbeat_process(
    args: argparse.Namespace,
    targets: list[LogicalManagerHeartbeatTarget],
    ready_queue: Any,
    result_queue: Any,
    stop_event: Any,
) -> None:
    config = _heartbeat_config(args)
    engine = create_engine(config)
    session_factory = create_session_factory(engine)
    load = ManagerHeartbeatLoad(
        session_factory=session_factory,
        targets=targets,
        interval_seconds=args.manager_heartbeat_interval,
        healthy_ttl=config.manager_healthy_ttl,
        self_fence_timeout=config.manager_self_fence_timeout,
    )
    try:
        await load.start()
        ready_queue.put({'kind': 'heartbeat_ready', 'pid': os.getpid()})
        await asyncio.to_thread(stop_event.wait)
        result_queue.put({'kind': 'heartbeat_done', 'pid': os.getpid(), 'report': await load.stop()})
    finally:
        await engine.dispose()


async def _assigned_count(session_factory: async_sessionmaker, *, prefix: str) -> int:
    async with session_factory() as session:
        return int(
            await session.scalar(
                select(func.count())
                .select_from(ExecutionRow)
                .where(
                    ExecutionRow.execution_id.like(f'{prefix}%'),
                    ExecutionRow.status == ExecutionStatus.ASSIGNED,
                )
            )
            or 0
        )


async def _validate(args: argparse.Namespace, *, prefix: str) -> dict[str, Any]:
    config = DistributedConfig(database_url=args.database_url, db_pool_size=2, db_max_overflow=0)
    engine = create_engine(config)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            assigned_rows = await _assigned_count(session_factory, prefix=prefix)
            assignment_rows = int(
                await session.scalar(
                    select(func.count())
                    .select_from(AssignmentRow)
                    .join(ExecutionRow, ExecutionRow.execution_id == AssignmentRow.execution_id)
                    .where(ExecutionRow.execution_id.like(f'{prefix}%'))
                )
                or 0
            )
            unique_assignments = int(
                await session.scalar(
                    select(func.count(func.distinct(AssignmentRow.execution_id)))
                    .join(ExecutionRow, ExecutionRow.execution_id == AssignmentRow.execution_id)
                    .where(ExecutionRow.execution_id.like(f'{prefix}%'))
                )
                or 0
            )
            active_reserved_slots = int(
                await session.scalar(
                    select(func.coalesce(func.sum(AssignmentRow.slot_cost), 0))
                    .join(ExecutionRow, ExecutionRow.execution_id == AssignmentRow.execution_id)
                    .where(
                        ExecutionRow.execution_id.like(f'{prefix}%'),
                        AssignmentRow.status.in_(['pending', 'active']),
                    )
                )
                or 0
            )
            capacity_rows = (
                await session.execute(
                    select(
                        InstanceManagerLivenessRow.manager_id,
                        InstanceManagerLivenessRow.capacity_total,
                        func.coalesce(func.sum(AssignmentRow.slot_cost), 0),
                    )
                    .outerjoin(
                        AssignmentRow,
                        and_(
                            AssignmentRow.manager_id == InstanceManagerLivenessRow.manager_id,
                            AssignmentRow.status.in_(['pending', 'active']),
                        ),
                    )
                    .group_by(
                        InstanceManagerLivenessRow.manager_id,
                        InstanceManagerLivenessRow.capacity_total,
                    )
                )
            ).all()
            quota_usage_rows = int(await session.scalar(select(func.count()).select_from(QuotaUsageRow)) or 0)
            schedule_error_rows = int(
                await session.scalar(
                    select(func.count())
                    .select_from(ExecutionRow)
                    .where(
                        ExecutionRow.execution_id.like(f'{prefix}%'),
                        ExecutionRow.last_schedule_error.is_not(None),
                    )
                )
                or 0
            )
            execution_event_rows = int(
                await session.scalar(
                    select(func.count())
                    .select_from(ExecutionEventRow)
                    .where(ExecutionEventRow.execution_id.like(f'{prefix}%'))
                )
                or 0
            )
            event_type_counts = {
                str(event_type): int(count)
                for event_type, count in (
                    await session.execute(
                        select(ExecutionEventRow.type, func.count())
                        .where(ExecutionEventRow.execution_id.like(f'{prefix}%'))
                        .group_by(ExecutionEventRow.type)
                    )
                ).all()
            }
            invalid_event_executions = int(
                await session.scalar(
                    select(func.count()).select_from(
                        select(
                            ExecutionEventRow.execution_id,
                        )
                        .where(ExecutionEventRow.execution_id.like(f'{prefix}%'))
                        .group_by(ExecutionEventRow.execution_id)
                        .having(
                            or_(
                                func.count() != 2,
                                func.sum(case((ExecutionEventRow.type == 'assignment.status.changed', 1), else_=0))
                                != 1,
                                func.sum(case((ExecutionEventRow.type == 'scheduler.assigned', 1), else_=0)) != 1,
                            )
                        )
                        .subquery()
                    )
                )
                or 0
            )
            remaining_leases = int(
                await session.scalar(
                    select(func.count())
                    .select_from(ShardLeaseRow)
                    .where(
                        ShardLeaseRow.shard_group == 'execution-scheduler',
                        ShardLeaseRow.lease_expires_at > ShardLeaseRow.updated_at,
                    )
                )
                or 0
            )
            shard_histogram = {
                str(shard_id): int(count)
                for shard_id, count in (
                    await session.execute(
                        select(ExecutionRow.shard_id, func.count())
                        .where(ExecutionRow.execution_id.like(f'{prefix}%'))
                        .group_by(ExecutionRow.shard_id)
                        .order_by(ExecutionRow.shard_id)
                    )
                ).all()
            }
            allocation = await session.get(EnvironmentAllocationRow, _allocation_identity(prefix)['allocation_id'])
    finally:
        await engine.dispose()

    oversold = [
        {'manager_id': manager_id, 'reserved': int(reserved or 0), 'capacity': int(capacity)}
        for manager_id, capacity, reserved in capacity_rows
        if int(reserved or 0) > int(capacity)
    ]
    if (assigned_rows, assignment_rows, unique_assignments, active_reserved_slots) != (args.task_count,) * 4:
        raise RuntimeError(
            'Scheduler result mismatch: '
            f'assigned={assigned_rows} assignments={assignment_rows} unique={unique_assignments} '
            f'active_reserved_slots={active_reserved_slots}'
        )
    if oversold:
        raise RuntimeError(f'Scheduler oversold logical Managers: {oversold[:5]}')
    if quota_usage_rows:
        raise RuntimeError(f'Allocation-backed Scheduler unexpectedly touched owner quota: rows={quota_usage_rows}')
    if schedule_error_rows:
        raise RuntimeError(f'Assigned executions retained stale scheduling errors: rows={schedule_error_rows}')
    expected_event_types = {
        'assignment.status.changed': args.task_count,
        'scheduler.assigned': args.task_count,
    }
    if (
        execution_event_rows != args.task_count * 2
        or event_type_counts != expected_event_types
        or invalid_event_executions
    ):
        raise RuntimeError(
            'Scheduler event mismatch: '
            f'rows={execution_event_rows}/{args.task_count * 2} types={event_type_counts} '
            f'invalid_executions={invalid_event_executions}'
        )
    if remaining_leases:
        raise RuntimeError(f'Scheduler did not release shard leases: rows={remaining_leases}')
    if len(shard_histogram) != SHARD_COUNT or sum(shard_histogram.values()) != args.task_count:
        raise RuntimeError(f'Scheduler shard fixture mismatch: {shard_histogram}')
    if (
        allocation is None
        or allocation.status != EnvironmentAllocationStatus.READY
        or allocation.resources_reserved_at is None
    ):
        raise RuntimeError('Allocation-backed Scheduler fixture is not ready and resource-reserved')
    return {
        'assigned': assigned_rows,
        'assignments': assignment_rows,
        'unique_assignments': unique_assignments,
        'active_reserved_slots': active_reserved_slots,
        'oversold_managers': len(oversold),
        'quota_usage_rows': quota_usage_rows,
        'schedule_error_rows': schedule_error_rows,
        'execution_event_rows': execution_event_rows,
        'event_type_counts': event_type_counts,
        'invalid_event_executions': invalid_event_executions,
        'remaining_shard_leases': remaining_leases,
        'shard_histogram': shard_histogram,
        'manager_capacity': max((int(row[1]) for row in capacity_rows), default=0),
        'max_manager_reserved_slots': max((int(row[2] or 0) for row in capacity_rows), default=0),
    }


def _drain_messages(result_queue: Any) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    while True:
        try:
            message = result_queue.get_nowait()
        except queue.Empty:
            return messages
        if str(message.get('kind', '')).endswith('_error'):
            raise RuntimeError(json.dumps(message, indent=2))
        messages.append(message)


def _wait_for_ready(
    *,
    args: argparse.Namespace,
    processes: list[mp.Process],
    ready_queue: Any,
    result_queue: Any,
) -> tuple[list[dict[str, Any]], int]:
    schedulers: list[dict[str, Any]] = []
    heartbeat_pid: int | None = None
    deadline = time.monotonic() + args.startup_timeout
    while len(schedulers) < args.replicas or heartbeat_pid is None:
        if time.monotonic() >= deadline:
            states = [(process.name, process.pid, process.exitcode) for process in processes]
            raise TimeoutError(f'Scheduler process startup timed out: {states}')
        dead = [(process.name, process.pid, process.exitcode) for process in processes if process.exitcode is not None]
        if dead:
            for message in _drain_messages(result_queue):
                if str(message.get('kind', '')).endswith('_done'):
                    raise RuntimeError(f'Scheduler process exited during startup: {message}')
            raise RuntimeError(f'Scheduler process exited during startup: {dead}')
        try:
            message = ready_queue.get(timeout=0.5)
        except queue.Empty:
            _drain_messages(result_queue)
            continue
        if message['kind'] == 'scheduler_ready':
            schedulers.append(message)
        elif message['kind'] == 'heartbeat_ready':
            heartbeat_pid = int(message['pid'])
        else:
            raise RuntimeError(f'Unexpected Scheduler readiness message: {message}')

    schedulers.sort(key=lambda item: int(item['index']))
    shard_ids = [int(shard_id) for scheduler in schedulers for shard_id in scheduler['owned_shards']]
    if len(shard_ids) != SHARD_COUNT or len(set(shard_ids)) != SHARD_COUNT:
        raise RuntimeError(f'Scheduler shard ownership did not converge uniquely: {schedulers}')
    return schedulers, heartbeat_pid


def _wait_for_replacement_ready(
    *,
    args: argparse.Namespace,
    process: mp.Process,
    worker_index: int,
    ready_queue: Any,
    result_queue: Any,
) -> dict[str, Any]:
    deadline = time.monotonic() + args.startup_timeout
    while time.monotonic() < deadline:
        if process.exitcode is not None:
            raise RuntimeError(
                f'Replacement Scheduler exited before ready: index={worker_index} exitcode={process.exitcode}'
            )
        try:
            message = ready_queue.get(timeout=0.5)
        except queue.Empty:
            for result in _drain_messages(result_queue):
                if str(result.get('kind', '')).endswith('_done'):
                    raise RuntimeError(
                        f'Unexpected process completion while waiting for replacement: {result}'
                    ) from None
            continue
        if message['kind'] == 'scheduler_ready' and int(message['index']) == worker_index:
            return message
        raise RuntimeError(f'Unexpected readiness message while waiting for replacement: {message}')
    raise TimeoutError(f'Replacement Scheduler startup timed out: index={worker_index} pid={process.pid}')


def _signal_live_processes(processes: list[mp.Process], stop_events: list[Any]) -> None:
    for process, stop_event in zip(processes, stop_events, strict=True):
        if process.is_alive():
            stop_event.set()


def run(args: argparse.Namespace) -> dict[str, Any]:
    return asyncio.run(_run_capacity(args))


async def _run_capacity(args: argparse.Namespace) -> dict[str, Any]:
    validate_topology(replicas=args.replicas, ownership_limit=args.ownership_limit)
    validate_recovery_config(
        replicas=args.replicas,
        worker_index=args.restart_worker_index,
        restart_after_seconds=args.restart_worker_after_seconds,
    )
    prefix = 'scheduler-capacity'
    heartbeat_targets = await _setup_fixture(args, prefix=prefix)

    context = mp.get_context('fork')
    ready_queue = context.Queue()
    result_queue = context.Queue()
    start_event = context.Event()
    scheduler_stop_events = [context.Event() for _ in range(args.replicas)]
    heartbeat_stop_event = context.Event()
    schedulers = cast(
        list[mp.Process],
        [
            _create_scheduler_process(
                context,
                index=index,
                args=args,
                prefix=prefix,
                ready_queue=ready_queue,
                result_queue=result_queue,
                start_event=start_event,
                stop_event=scheduler_stop_events[index],
            )
            for index in range(args.replicas)
        ],
    )
    heartbeat = cast(
        mp.Process,
        context.Process(
            target=_heartbeat_process,
            args=(args, heartbeat_targets, ready_queue, result_queue, heartbeat_stop_event),
            name='nexus-scheduler-capacity-heartbeat',
        ),
    )
    processes = [*schedulers, heartbeat]
    stop_events = [*scheduler_stop_events, heartbeat_stop_event]
    for process in processes:
        process.start()

    monitor_config = DistributedConfig(database_url=args.database_url, db_pool_size=2, db_max_overflow=0)
    monitor_engine = create_engine(monitor_config)
    monitor_factory = create_session_factory(monitor_engine)
    worker_reports: dict[int, dict[str, Any]] = {}
    heartbeat_report: dict[str, Any] | None = None
    recovery_report: dict[str, Any] | None = None
    try:
        ready, heartbeat_pid = await asyncio.to_thread(
            _wait_for_ready,
            args=args,
            processes=processes,
            ready_queue=ready_queue,
            result_queue=result_queue,
        )
        scheduler_pids = [int(item['pid']) for item in ready]
        process_sampler = ProcessSampler(scheduler_pids)
        all_process_sampler = ProcessSampler([*scheduler_pids, heartbeat_pid])
        database_process_sampler = ProcessSampler([args.database_process_pid]) if args.database_process_pid else None
        database_before = await database_status(monitor_factory)
        database_details = await database_identity(monitor_factory)
        gauge_peaks = {name: database_before.get(name, 0) for name in DATABASE_GAUGES}
        process_sampler.sample()
        all_process_sampler.sample()
        if database_process_sampler is not None:
            database_process_sampler.sample()
        started_epoch = time.time()
        started = time.perf_counter()
        start_event.set()
        deadline = time.monotonic() + args.progress_timeout
        last_log_at = 0.0
        assigned = 0
        while assigned < args.task_count:
            for message in _drain_messages(result_queue):
                if message['kind'] == 'scheduler_done':
                    raise RuntimeError(f'Scheduler exited before benchmark completed: {message}')
                if message['kind'] == 'heartbeat_done':
                    raise RuntimeError(f'Manager heartbeat exited before benchmark completed: {message}')
            if (
                args.restart_worker_after_seconds > 0
                and recovery_report is None
                and time.perf_counter() - started >= args.restart_worker_after_seconds
            ):
                worker_index = args.restart_worker_index
                old_process = schedulers[worker_index]
                old_pid = int(old_process.pid or 0)
                old_shards = [int(shard_id) for shard_id in ready[worker_index]['owned_shards']]
                process_sampler.sample()
                all_process_sampler.sample()
                killed_at = time.perf_counter()
                old_process.kill()
                old_process.join(5)
                if old_process.is_alive():
                    raise TimeoutError(f'Scheduler worker did not stop after kill: pid={old_pid}')

                replacement_stop_event = context.Event()
                replacement = _create_scheduler_process(
                    context,
                    index=worker_index,
                    args=args,
                    prefix=prefix,
                    ready_queue=ready_queue,
                    result_queue=result_queue,
                    start_event=start_event,
                    stop_event=replacement_stop_event,
                )
                replacement.start()
                schedulers[worker_index] = replacement
                scheduler_stop_events[worker_index] = replacement_stop_event
                processes[processes.index(old_process)] = replacement
                stop_events[worker_index] = replacement_stop_event
                replacement_ready = await asyncio.to_thread(
                    _wait_for_replacement_ready,
                    args=args,
                    process=replacement,
                    worker_index=worker_index,
                    ready_queue=ready_queue,
                    result_queue=result_queue,
                )
                new_pid = int(replacement_ready['pid'])
                replacement_shards = [int(shard_id) for shard_id in replacement_ready['owned_shards']]
                if set(replacement_shards) != set(old_shards):
                    raise RuntimeError(
                        f'Replacement Scheduler recovered different shards: {replacement_shards}/{old_shards}'
                    )
                ready[worker_index] = replacement_ready
                scheduler_pids[worker_index] = new_pid
                process_sampler.replace_pid(old_pid, new_pid)
                all_process_sampler.replace_pid(old_pid, new_pid)
                recovered_at = time.perf_counter()
                recovery_report = {
                    'worker_index': worker_index,
                    'old_pid': old_pid,
                    'new_pid': new_pid,
                    'owned_shards': replacement_shards,
                    'killed_at_seconds': round(killed_at - started, 6),
                    'replacement_ready_at_seconds': round(recovered_at - started, 6),
                    'outage_seconds': round(recovered_at - killed_at, 6),
                    'shard_lease_ttl_seconds': args.shard_lease_ttl,
                }
            dead = [
                (process.name, process.pid, process.exitcode) for process in processes if process.exitcode is not None
            ]
            if dead:
                raise RuntimeError(f'Scheduler capacity process exited early: {dead}')
            assigned = await _assigned_count(monitor_factory, prefix=prefix)
            if assigned > args.task_count:
                raise RuntimeError(f'Scheduler assigned too many tasks: {assigned}/{args.task_count}')
            process_sampler.sample()
            all_process_sampler.sample()
            if database_process_sampler is not None:
                database_process_sampler.sample()
            database_current = await database_status(monitor_factory)
            for name in DATABASE_GAUGES:
                gauge_peaks[name] = max(gauge_peaks.get(name, 0), database_current.get(name, 0))
            if assigned == args.task_count:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Scheduler made no progress before timeout: assigned={assigned}/{args.task_count}')
            if time.monotonic() - last_log_at >= args.progress_log_interval:
                print(
                    f'SCHEDULER_PROGRESS assigned={assigned}/{args.task_count}',
                    file=sys.stderr,
                    flush=True,
                )
                last_log_at = time.monotonic()
            await asyncio.sleep(args.progress_poll_interval)

        elapsed = time.perf_counter() - started
        finished_epoch = time.time()
        if args.restart_worker_after_seconds > 0 and recovery_report is None:
            raise RuntimeError('Scheduler recovery was configured but the fixture completed before restart')
        _signal_live_processes(processes, stop_events)
        shutdown_deadline = time.monotonic() + args.shutdown_timeout
        while len(worker_reports) < args.replicas or heartbeat_report is None:
            if time.monotonic() >= shutdown_deadline:
                raise TimeoutError(
                    f'Scheduler process reports timed out: workers={len(worker_reports)}/{args.replicas} '
                    f'heartbeat={heartbeat_report is not None}'
                )
            try:
                message = result_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if str(message.get('kind', '')).endswith('_error'):
                raise RuntimeError(json.dumps(message, indent=2))
            if message['kind'] == 'scheduler_done':
                worker_reports[int(message['index'])] = message
            elif message['kind'] == 'heartbeat_done':
                heartbeat_report = message['report']
            else:
                raise RuntimeError(f'Unexpected Scheduler result message: {message}')

        join_processes(processes, timeout=args.shutdown_timeout)
        bad_exits = [(process.name, process.pid, process.exitcode) for process in processes if process.exitcode != 0]
        if bad_exits:
            raise RuntimeError(f'Scheduler worker process failures: {bad_exits}')
        validation = await _validate(args, prefix=prefix)
        database_after_cleanup = await database_status(monitor_factory)
        heartbeat_failures = {
            key: heartbeat_report[key]
            for key in ('fatal_failures', 'fenced_managers', 'self_fenced_managers', 'stale_managers')
            if heartbeat_report[key]
        }
        if heartbeat_failures:
            raise RuntimeError(f'Manager heartbeat load was unhealthy: {heartbeat_failures}')

        worker_totals = {
            key: sum(int(report['stats'][key]) for report in worker_reports.values())
            for key in (
                'cycles',
                'idle_cycles',
                'assigned',
                'scanned',
                'claimed',
                'lock_contention',
                'no_capacity',
                'candidate_queries',
            )
        }
        worker_reports_cover_full_run = recovery_report is None
        if worker_reports_cover_full_run and worker_totals['assigned'] != args.task_count:
            raise RuntimeError(
                f'Scheduler worker reports do not match database: {worker_totals["assigned"]}/{args.task_count}'
            )
        return {
            'source_identity': source_identity(Path(__file__).resolve().parents[3], _SOURCE_FILES),
            'database_identity': database_details,
            'replicas': args.replicas,
            'ownership_limit': args.ownership_limit,
            'shard_count': SHARD_COUNT,
            'task_count': args.task_count,
            'manager_count': args.manager_count,
            'batch_size': args.batch_size,
            'scheduler_poll_interval_seconds': args.scheduler_poll_interval,
            'scheduler_pool_size': args.scheduler_pool_size,
            'scheduler_max_overflow': args.scheduler_max_overflow,
            'manager_liveness_profile': {
                'heartbeat_targets': len(heartbeat_targets),
                'heartbeat_interval_seconds': args.manager_heartbeat_interval,
                'healthy_ttl_seconds': args.manager_healthy_ttl,
                'self_fence_timeout_seconds': args.manager_self_fence_timeout,
            },
            'shard_lease_profile': {
                'ttl_seconds': args.shard_lease_ttl,
                'renew_interval_seconds': args.shard_lease_renew_interval,
            },
            'elapsed_seconds': round(elapsed, 6),
            'items_per_second': round(args.task_count / elapsed, 2),
            'start_epoch': started_epoch,
            'end_epoch': finished_epoch,
            'scheduler_pids': scheduler_pids,
            'heartbeat_pid': heartbeat_pid,
            'recovery': recovery_report,
            'worker_reports_cover_full_run': worker_reports_cover_full_run,
            'worker_totals': worker_totals,
            'process_resources': process_sampler.report(),
            'all_process_resources': all_process_sampler.report(),
            'database_process_resources': (
                database_process_sampler.report() if database_process_sampler is not None else None
            ),
            'database': database_status_report(
                before=database_before,
                after=database_current,
                gauge_peaks=gauge_peaks,
            ),
            'database_after_cleanup': database_after_cleanup,
            'manager_heartbeats': heartbeat_report,
            'validation': validation,
        }
    finally:
        _signal_live_processes(processes, stop_events)
        start_event.set()
        with contextlib.suppress(TimeoutError):
            join_processes(processes, timeout=10)
        for channel in (ready_queue, result_queue):
            with contextlib.suppress(Exception):
                channel.close()
                channel.join_thread()
        await monitor_engine.dispose()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database-url', required=True, help='Empty dedicated async SQLAlchemy MySQL URL.')
    parser.add_argument('--task-count', type=int, default=50_000)
    parser.add_argument('--manager-count', type=int, default=2_000)
    parser.add_argument('--replicas', type=int, default=16)
    parser.add_argument('--ownership-limit', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=250)
    parser.add_argument('--scheduler-poll-interval', type=float, default=0.1)
    parser.add_argument('--scheduler-pool-size', type=int, default=2)
    parser.add_argument('--scheduler-max-overflow', type=int, default=2)
    parser.add_argument('--heartbeat-pool-size', type=int, default=8)
    parser.add_argument('--heartbeat-max-overflow', type=int, default=8)
    parser.add_argument('--database-process-pid', type=int, default=0)
    parser.add_argument('--manager-heartbeat-interval', type=float, default=5)
    parser.add_argument('--manager-healthy-ttl', type=float, default=15)
    parser.add_argument('--manager-self-fence-timeout', type=float, default=30)
    parser.add_argument('--shard-lease-ttl', type=float, default=20)
    parser.add_argument('--shard-lease-renew-interval', type=float, default=5)
    parser.add_argument('--restart-worker-after-seconds', type=float, default=0)
    parser.add_argument('--restart-worker-index', type=int, default=0)
    parser.add_argument('--progress-poll-interval', type=float, default=0.25)
    parser.add_argument('--progress-log-interval', type=float, default=5)
    parser.add_argument('--startup-timeout', type=float, default=180)
    parser.add_argument('--progress-timeout', type=float, default=1_800)
    parser.add_argument('--shutdown-timeout', type=float, default=60)
    args = parser.parse_args()
    validate_topology(replicas=args.replicas, ownership_limit=args.ownership_limit)
    validate_recovery_config(
        replicas=args.replicas,
        worker_index=args.restart_worker_index,
        restart_after_seconds=args.restart_worker_after_seconds,
    )
    if args.task_count < SHARD_COUNT:
        parser.error(f'--task-count must be at least {SHARD_COUNT}')
    if args.manager_count < 1:
        parser.error('--manager-count must be positive')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    for name in ('scheduler_pool_size', 'heartbeat_pool_size'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    for name in ('scheduler_max_overflow', 'heartbeat_max_overflow'):
        if getattr(args, name) < 0:
            parser.error(f'--{name.replace("_", "-")} must be non-negative')
    if args.database_process_pid < 0:
        parser.error('--database-process-pid must be non-negative')
    for name in (
        'scheduler_poll_interval',
        'manager_heartbeat_interval',
        'manager_healthy_ttl',
        'manager_self_fence_timeout',
        'shard_lease_ttl',
        'shard_lease_renew_interval',
        'progress_poll_interval',
        'progress_log_interval',
        'startup_timeout',
        'progress_timeout',
        'shutdown_timeout',
    ):
        if getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.manager_heartbeat_interval >= args.manager_healthy_ttl:
        parser.error('--manager-heartbeat-interval must be shorter than --manager-healthy-ttl')
    if args.manager_self_fence_timeout <= args.manager_healthy_ttl:
        parser.error('--manager-self-fence-timeout must exceed --manager-healthy-ttl')
    if args.shard_lease_renew_interval >= args.shard_lease_ttl:
        parser.error('--shard-lease-renew-interval must be shorter than --shard-lease-ttl')
    return args


def main() -> None:
    print(json.dumps(run(_parse_args()), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
