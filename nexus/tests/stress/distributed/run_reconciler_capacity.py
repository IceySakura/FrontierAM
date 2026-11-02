"""Measure Reconciler capacity with independent worker processes and MySQL.

The benchmark replaces only external provider and Artifact I/O. Reconciler
repositories, transactions, leases, quota accounting, Manager liveness, and
Batch lifecycle state transitions all use production code.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, cast

from sqlalchemy import case, func, insert, or_, select, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
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

from nexus.distributed.artifacts.run_result import RunResultPublisher
from nexus.distributed.batch.lifecycle import BatchLifecycleReconciler
from nexus.distributed.config import DistributedConfig
from nexus.distributed.config.views import (
    batch_lifecycle_config_view,
    environment_config_view,
    reconciler_config_view,
)
from nexus.distributed.persistence.db import create_engine, create_session_factory, is_retryable_database_error
from nexus.distributed.environment.controller import EnvironmentController
from nexus.distributed.environment.ports import AllocatorObservation
from nexus.distributed.environment.models import (
    EnvironmentAllocationStatus,
    EnvironmentAllocationType,
    EnvironmentProviderType,
    EnvironmentStatus,
)
from nexus.distributed.execution.models import (
    AssignmentStatus,
    ExecutionMode,
    ExecutionStatus,
    RouteStatus,
    TERMINAL_EXECUTION_STATUSES,
)
from nexus.distributed.observability import ControlLoopWork
from nexus.distributed.persistence.orm import (
    ArtifactRow,
    AssignmentRow,
    BatchItemAttemptRow,
    BatchItemRow,
    BatchRunRow,
    EnvironmentAllocationRow,
    EnvironmentGrantRow,
    EnvironmentResourceUsageRow,
    EnvironmentRow,
    ExecutionCommandRow,
    ExecutionEventRow,
    ExecutionRow,
    InstanceManagerLivenessRow,
    InstanceManagerRow,
    ShardLeaseRow,
    create_schema,
)
from nexus.distributed.control_plane.reconciler import ReconcilerService
from nexus.distributed.repositories.artifact import ArtifactRepository
from nexus.distributed.utils.clock import utcnow
from nexus.distributed.utils.identity import canonical_sha256, stable_shard


SHARD_COUNT = 64
TOPOLOGIES = ((1, 64), (4, 16), (16, 4), (32, 2))
ScenarioName = Literal['core', 'allocation', 'batch', 'mixed']
_SOURCE_FILES = (
    'nexus/distributed/batch/lifecycle.py',
    'nexus/distributed/environment/controller.py',
    'nexus/distributed/manager_liveness.py',
    'nexus.distributed.persistence.orm.py',
    'nexus.distributed.control_plane.reconciler.py',
    'nexus/distributed/repositories/manager.py',
    'nexus/distributed/repositories/environment.py',
    'nexus/distributed/repositories/quota.py',
    'nexus/distributed/repositories/reconciler.py',
    'nexus/distributed/repositories/shard_lease.py',
    'nexus/distributed/shard_leases.py',
    'tests/stress/distributed/capacity_support.py',
    'tests/stress/distributed/run_reconciler_capacity.py',
)
CORE_OPERATIONS = (
    'reconcile_manager_dead',
    'reconcile_session_mismatch',
    'reconcile_pending_timeouts',
    'reconcile_starting_timeouts',
    'expire_queued',
    'request_ttl_stops',
    'repair_active_assignments',
    'reconcile_stopping_timeouts',
    'cleanup_terminal',
)
_CORE_HEARTBEAT_MANAGER_KINDS = ('sm', 'ok')

_CORE_CODES = {
    'reconcile_manager_dead': 'md',
    'reconcile_session_mismatch': 'sm',
    'reconcile_pending_timeouts': 'pa',
    'reconcile_starting_timeouts': 'st',
    'expire_queued': 'eq',
    'request_ttl_stops': 'tt',
    'repair_active_assignments': 'ra',
    'reconcile_stopping_timeouts': 'so',
    'cleanup_terminal': 'ct',
}
_CORE_EXPECTED_STATUSES = {
    'md': ExecutionStatus.LOST,
    'sm': ExecutionStatus.LOST,
    'pa': ExecutionStatus.LOST,
    'st': ExecutionStatus.LOST,
    'eq': ExecutionStatus.EXPIRED,
    'tt': ExecutionStatus.EXPIRED,
    'ra': ExecutionStatus.LOST,
    'so': ExecutionStatus.KILLED,
    'ct': ExecutionStatus.SUCCEEDED,
}
_CORE_EXPECTED_EVENT_MULTIPLIERS = {
    'md': {'assignment.status.changed': 1, 'execution.status.changed': 1, 'reconciler.terminal': 1},
    'sm': {'assignment.status.changed': 1, 'execution.status.changed': 1, 'reconciler.terminal': 1},
    'pa': {'assignment.status.changed': 1, 'execution.status.changed': 1, 'reconciler.terminal': 1},
    'st': {'assignment.status.changed': 1, 'execution.status.changed': 1, 'reconciler.terminal': 1},
    'eq': {'execution.status.changed': 1},
    'tt': {
        'assignment.status.changed': 1,
        'command.status.changed': 2,
        'execution.status.changed': 2,
        'reconciler.terminal': 1,
    },
    'ra': {'execution.status.changed': 1, 'reconciler.terminal': 1},
    'so': {'assignment.status.changed': 1, 'execution.status.changed': 1, 'reconciler.terminal': 1},
    'ct': {},
}


@dataclass(frozen=True, slots=True)
class _FixtureSummary:
    logical_items: int
    executions: int = 0
    core_executions: int = 0
    allocations: int = 0
    managers: int = 0
    heartbeat_eligible_managers: int = 0
    batch_parents: int = 0
    batch_children: int = 0


class _NoopController:
    last_stats = ControlLoopWork()
    last_environment_stats = ControlLoopWork()

    def set_shard_ids(self, _shard_ids: list[int]) -> None:
        return None

    async def reconcile_once(self) -> int:
        return 0

    async def close(self) -> None:
        return None


class _DeterministicEnvironmentAllocator:
    """Stateless idempotent provider boundary for multiprocess tests."""

    def __init__(self, *, delay_seconds: float = 0) -> None:
        self.delay_seconds = delay_seconds
        self.create_calls = 0
        self.observe_calls = 0
        self.delete_calls = 0
        self.observe_deletion_calls = 0

    async def create(self, allocation: Any) -> AllocatorObservation:
        self.create_calls += 1
        await self._delay()
        return self._present(allocation)

    async def observe(self, allocation: Any) -> AllocatorObservation:
        self.observe_calls += 1
        await self._delay()
        return self._present(allocation)

    async def delete(self, _allocation: Any) -> None:
        self.delete_calls += 1
        await self._delay()

    async def observe_deletion(self, _allocation: Any) -> AllocatorObservation:
        self.observe_deletion_calls += 1
        await self._delay()
        return AllocatorObservation(exists=False, provider_ready_replicas=0, provider_resource_ref=None)

    async def close(self) -> None:
        return None

    async def _delay(self) -> None:
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

    @staticmethod
    def _present(allocation: Any) -> AllocatorObservation:
        return AllocatorObservation(
            exists=True,
            provider_ready_replicas=allocation.desired_manager_count,
            provider_resource_ref={
                'kind': 'benchmark-workload',
                'name': allocation.allocation_id,
            },
        )

    def report(self) -> dict[str, int]:
        return {
            'create': self.create_calls,
            'observe': self.observe_calls,
            'delete': self.delete_calls,
            'observe_deletion': self.observe_deletion_calls,
        }


class _DatabaseRunResultPublisher(RunResultPublisher):
    """Persist a deterministic ready Artifact exactly like Application Gateway."""

    def __init__(self, session_factory: async_sessionmaker) -> None:
        self.session_factory = session_factory
        self.publish_calls = 0

    async def publish_run_result(
        self,
        *,
        execution_id: str,
        assignment_id: str,
        attempt_id: str,
        producer_key: str,
        content: bytes,
    ) -> Mapping[str, Any]:
        self.publish_calls += 1
        digest = hashlib.sha256(content).hexdigest()
        artifact_id = f'rr-{hashlib.sha256(producer_key.encode()).hexdigest()[:48]}'
        now = utcnow()
        async with self.session_factory() as session:
            execution = await session.get(ExecutionRow, execution_id)
            if execution is None:
                raise RuntimeError(f'benchmark parent execution missing: {execution_id}')
            artifact = await session.get(ArtifactRow, artifact_id)
            if artifact is None:
                session.add(
                    ArtifactRow(
                        artifact_id=artifact_id,
                        owner_principal_id=execution.owner_principal_id or 'benchmark-owner',
                        environment_id=execution.environment_id or 'benchmark-environment',
                        owner_tenant=execution.owner_tenant,
                        kind='run_result',
                        producer_key=producer_key,
                        uri=f'benchmark://{artifact_id}',
                        sha256=digest,
                        size_bytes=len(content),
                        provider_etag=None,
                        status='ready',
                        operation_id=None,
                        operation_owner=None,
                        operation_expires_at=None,
                        failure_count=0,
                        created_by='reconciler-capacity',
                        expires_at=None,
                        created_at=now,
                        updated_at=now,
                    )
                )
                await session.flush()
            elif (
                artifact.producer_key != producer_key
                or artifact.sha256 != digest
                or artifact.size_bytes != len(content)
            ):
                raise RuntimeError('benchmark run_result idempotency conflict')
            await ArtifactRepository(session).link_run_result(
                artifact_id,
                execution_id=execution_id,
                assignment_id=assignment_id,
                attempt_id=attempt_id,
                now=now,
            )
            await session.commit()
        return {
            'artifact_id': artifact_id,
            'kind': 'run_result',
            'sha256': digest,
            'size_bytes': len(content),
        }


def scenario_loop_names(scenario: ScenarioName) -> tuple[str, ...]:
    return {
        'core': ('core',),
        'allocation': ('environment',),
        'batch': ('batch',),
        'mixed': ('core', 'environment', 'batch'),
    }[scenario]


def allocation_fixture_counts(
    *,
    ordinary_count: int,
    batch_count: int,
    batch_execution_managers: int,
) -> dict[str, int]:
    return {
        'top_level_executions': ordinary_count + batch_count,
        'allocations': ordinary_count + 2 * batch_count,
        'managers': ordinary_count + batch_count * (1 + batch_execution_managers),
        'ordinary_allocations': ordinary_count,
        'batch_orchestration_allocations': batch_count,
        'batch_execution_allocations': batch_count,
    }


def core_fixture_counts(*, items_per_operation: int) -> dict[str, int]:
    if items_per_operation < 1:
        raise ValueError('items_per_operation must be positive')
    return dict.fromkeys(CORE_OPERATIONS, items_per_operation)


def core_heartbeat_target_count() -> int:
    return len(_CORE_HEARTBEAT_MANAGER_KINDS) * SHARD_COUNT


def heartbeat_fixture_target_count(
    *,
    scenario: ScenarioName,
    ordinary_count: int,
    allocation_batch_count: int,
    batch_execution_managers: int,
    heartbeat_manager_limit: int,
    batch_parent_count: int,
) -> int:
    count = core_heartbeat_target_count() if scenario in {'core', 'mixed'} else 0
    if scenario in {'allocation', 'mixed'}:
        allocation_managers = allocation_fixture_counts(
            ordinary_count=ordinary_count,
            batch_count=allocation_batch_count,
            batch_execution_managers=batch_execution_managers,
        )['managers']
        count += (
            allocation_managers if heartbeat_manager_limit == 0 else min(allocation_managers, heartbeat_manager_limit)
        )
    if scenario in {'batch', 'mixed'}:
        count += batch_parent_count
    return count


def validate_heartbeat_coverage(*, eligible_managers: int, target_count: int, allow_partial: bool) -> None:
    if not 0 <= target_count <= eligible_managers:
        raise ValueError(f'invalid Manager heartbeat coverage: {target_count}/{eligible_managers}')
    if target_count < eligible_managers and not allow_partial:
        raise ValueError(
            'partial Manager heartbeat coverage requires --allow-partial-heartbeat-coverage: '
            f'{target_count}/{eligible_managers}'
        )


def capacity_fixture_prefix(scenario: ScenarioName) -> str:
    return f'reconciler-capacity-{scenario}'


def batch_fixture_counts(*, parent_count: int, children_per_parent: int) -> dict[str, int]:
    if parent_count < 0 or children_per_parent < 0:
        raise ValueError('Batch fixture sizes must be non-negative')
    return {
        'parents': parent_count,
        'children': parent_count * children_per_parent,
        'managers': parent_count,
    }


def validate_topology(*, replicas: int, ownership_limit: int) -> None:
    if replicas < 1 or ownership_limit < 1 or replicas * ownership_limit != SHARD_COUNT:
        raise ValueError('replicas * ownership_limit must exactly cover all shards')


def validate_recovery_config(*, replicas: int, worker_index: int, restart_after_seconds: float) -> None:
    if restart_after_seconds < 0:
        raise ValueError('restart_after_seconds must be non-negative')
    if restart_after_seconds > 0 and not 0 <= worker_index < replicas:
        raise ValueError('restart worker index must identify a configured replica')


def validate_timing_config(
    *,
    manager_heartbeat_interval: float,
    manager_healthy_ttl: float,
    manager_self_fence_timeout: float,
    shard_lease_ttl: float,
    shard_lease_renew_interval: float,
    environment_operation_ttl: float,
    environment_lease_renew_interval: float,
    batch_claim_ttl: float,
    batch_claim_renew_interval: float,
) -> None:
    values = {
        'manager heartbeat interval': manager_heartbeat_interval,
        'manager healthy TTL': manager_healthy_ttl,
        'manager self-fence timeout': manager_self_fence_timeout,
        'shard lease TTL': shard_lease_ttl,
        'shard lease renew interval': shard_lease_renew_interval,
        'environment operation TTL': environment_operation_ttl,
        'environment lease renew interval': environment_lease_renew_interval,
        'batch claim TTL': batch_claim_ttl,
        'batch claim renew interval': batch_claim_renew_interval,
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f'{name} must be positive')
    if manager_heartbeat_interval >= manager_healthy_ttl:
        raise ValueError('Manager heartbeat interval must be shorter than the healthy TTL')
    if manager_healthy_ttl >= manager_self_fence_timeout:
        raise ValueError('Manager healthy TTL must be shorter than the self-fence timeout')
    if shard_lease_renew_interval >= shard_lease_ttl:
        raise ValueError('Shard lease renew interval must be shorter than the lease TTL')
    if environment_lease_renew_interval >= environment_operation_ttl:
        raise ValueError('Environment lease renew interval must be shorter than the operation TTL')
    if batch_claim_renew_interval >= batch_claim_ttl:
        raise ValueError('Batch claim renew interval must be shorter than the claim TTL')


def _base_execution(
    execution_id: str,
    *,
    owner_principal_id: str,
    status: str,
    now: Any,
    spec_type: str = 'nexus_single',
    environment_id: str | None = None,
    pool_id: str | None = None,
    allocation_id: str | None = None,
    workload: str | None = None,
    allocation_spec_hash: str | None = None,
) -> dict[str, Any]:
    return {
        'execution_id': execution_id,
        'owner_principal_id': owner_principal_id,
        'owner_tenant': 'capacity',
        'created_by': 'reconciler-capacity',
        'environment_id': environment_id,
        'pool_id': pool_id,
        'allocation_id': allocation_id,
        'workload': workload,
        'allocation_spec_hash': allocation_spec_hash,
        'version': 1,
        'shard_id': stable_shard(execution_id, SHARD_COUNT),
        'mode': ExecutionMode.JOB,
        'spec_type': spec_type,
        'instance_mode': 'asyncio',
        'spec_json': {'batch': {'retry_failed': 0, 'max_inflight': 2000}} if spec_type == 'nexus_batch' else {},
        'driver_config_json': {},
        'placement_json': {},
        'status': status,
        'route_status': RouteStatus.NONE,
        'slot_cost': 1,
        'next_attempt_seq': 2,
        'attempt_count': 1,
        'max_attempts': 1,
        'next_event_seq': 1,
        'priority': 0,
        'next_schedule_at': now,
        'created_at': now,
        'updated_at': now,
    }


def _assignment_values(
    *,
    execution_id: str,
    assignment_id: str,
    attempt_id: str,
    manager_id: str,
    manager_session_id: str,
    status: str,
    now: Any,
    accepted_at: Any = None,
    started_at: Any = None,
) -> dict[str, Any]:
    return {
        'assignment_id': assignment_id,
        'execution_id': execution_id,
        'owner_principal_id': 'core-owner',
        'attempt_id': attempt_id,
        'attempt_seq': 1,
        'manager_id': manager_id,
        'manager_session_id': manager_session_id,
        'status': status,
        'slot_cost': 1,
        'endpoint_url': f'http://{manager_id}:8090',
        'endpoint_ready': True,
        'accepted_at': accepted_at,
        'started_at': started_at,
        'created_at': now,
        'updated_at': now,
    }


def _manager_values(
    *,
    manager_id: str,
    session_id: str,
    now: Any,
    heartbeat_at: Any,
    environment_id: str = 'core-environment',
    pool_id: str = 'core-pool',
    allocation_id: str | None = None,
    workload: str = 'execution',
    owner_principal_id: str = 'core-owner',
    allocation_spec_hash: str | None = None,
    capacity_total: int = 100_000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manager = {
        'manager_id': manager_id,
        'session_id': session_id,
        'endpoint_url': f'http://{manager_id}:8090',
        'labels_json': {'workload': workload},
        'draining': False,
        'owner_principal_id': owner_principal_id,
        'environment_id': environment_id,
        'pool_id': pool_id,
        'allocation_id': allocation_id,
        'workload': workload,
        'allocation_spec_hash': allocation_spec_hash,
        'supported_spec_versions_json': {'nexus_single': [1], 'nexus_batch': [1]},
        'supported_modes_json': ['job'],
        'supported_spec_types_json': ['nexus_single', 'nexus_batch'],
        'supported_instance_modes_json': ['asyncio'],
        'capacity_total': capacity_total,
        'observed_capacity_used': 0,
        'last_heartbeat_at': heartbeat_at,
        'created_at': now,
        'updated_at': heartbeat_at,
    }
    liveness = {
        'manager_id': manager_id,
        'session_id': session_id,
        'capacity_total': capacity_total,
        'observed_capacity_used': 0,
        'last_heartbeat_at': heartbeat_at,
        'fenced_at': None,
        'updated_at': heartbeat_at,
    }
    return manager, liveness


async def _insert_rows(
    session_factory: async_sessionmaker,
    model: Any,
    rows: list[dict[str, Any]],
    *,
    chunk_size: int = 1000,
) -> None:
    if not rows:
        return
    async with session_factory() as session:
        for offset in range(0, len(rows), chunk_size):
            await session.execute(insert(model), rows[offset : offset + chunk_size])
            await session.commit()


async def _seed_core_fixture(
    session_factory: async_sessionmaker,
    *,
    prefix: str,
    items_per_operation: int,
) -> tuple[int, list[LogicalManagerHeartbeatTarget]]:
    now = utcnow()
    stale = now - timedelta(hours=2)
    overdue = now - timedelta(minutes=5)
    managers: list[dict[str, Any]] = []
    liveness_rows: list[dict[str, Any]] = []
    heartbeat_targets: list[LogicalManagerHeartbeatTarget] = []
    manager_sessions: dict[tuple[str, int], str] = {}
    for kind, heartbeat_at in (('md', stale), ('sm', now), ('ok', now)):
        for index in range(SHARD_COUNT):
            manager_id = f'{prefix}-cm-{kind}-{index:02d}'
            session_id = f'{kind}-current-{index:02d}'
            manager, liveness = _manager_values(
                manager_id=manager_id,
                session_id=session_id,
                now=now,
                heartbeat_at=heartbeat_at,
            )
            managers.append(manager)
            liveness_rows.append(liveness)
            manager_sessions[(kind, index)] = session_id
            if kind in _CORE_HEARTBEAT_MANAGER_KINDS:
                heartbeat_targets.append(
                    LogicalManagerHeartbeatTarget(
                        manager_id=manager_id,
                        session_id=session_id,
                        capacity_total=100_000,
                    )
                )

    executions: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for operation in CORE_OPERATIONS:
        code = _CORE_CODES[operation]
        for index in range(items_per_operation):
            execution_id = f'{prefix}-c-{code}-{index:08d}'
            execution = _base_execution(
                execution_id,
                owner_principal_id='core-owner',
                status=ExecutionStatus.QUEUED,
                now=now,
            )
            assignment_status: str | None = None
            manager_kind = 'ok'
            manager_session_override: str | None = None
            accepted_at = None
            started_at = None

            if operation == 'reconcile_manager_dead':
                execution['status'] = ExecutionStatus.RUNNING
                execution['started_at'] = overdue
                assignment_status = AssignmentStatus.ACTIVE
                manager_kind = 'md'
                accepted_at = overdue
                started_at = overdue
            elif operation == 'reconcile_session_mismatch':
                execution['status'] = ExecutionStatus.RUNNING
                execution['started_at'] = overdue
                assignment_status = AssignmentStatus.ACTIVE
                manager_kind = 'sm'
                manager_session_override = f'sm-old-{index % SHARD_COUNT:02d}'
                accepted_at = overdue
                started_at = overdue
            elif operation == 'reconcile_pending_timeouts':
                execution['status'] = ExecutionStatus.ASSIGNED
                assignment_status = AssignmentStatus.PENDING
            elif operation == 'reconcile_starting_timeouts':
                execution['status'] = ExecutionStatus.STARTING
                assignment_status = AssignmentStatus.ACTIVE
                accepted_at = overdue
            elif operation == 'expire_queued':
                execution['queue_expires_at'] = overdue
            elif operation == 'request_ttl_stops':
                execution['status'] = ExecutionStatus.RUNNING
                execution['started_at'] = overdue
                execution['execution_expires_at'] = overdue
                execution['stop_deadline_at'] = overdue
                assignment_status = AssignmentStatus.ACTIVE
                accepted_at = overdue
                started_at = overdue
            elif operation == 'repair_active_assignments':
                execution['status'] = ExecutionStatus.RUNNING
                execution['started_at'] = overdue
            elif operation == 'reconcile_stopping_timeouts':
                execution['status'] = ExecutionStatus.STOPPING
                execution['started_at'] = overdue
                execution['stop_reason'] = 'user_stop'
                execution['stop_deadline_at'] = overdue
                assignment_status = AssignmentStatus.ACTIVE
                accepted_at = overdue
                started_at = overdue
            elif operation == 'cleanup_terminal':
                execution['status'] = ExecutionStatus.SUCCEEDED
                execution['finished_at'] = overdue
            else:
                raise AssertionError(operation)

            if assignment_status is not None:
                ordinal = index % SHARD_COUNT
                manager_id = f'{prefix}-cm-{manager_kind}-{ordinal:02d}'
                assignment_id = f'{prefix}-ca-{code}-{index:08d}'
                attempt_id = f'{prefix}-ct-{code}-{index:08d}'
                manager_session_id = manager_session_override or manager_sessions[(manager_kind, ordinal)]
                assignment = _assignment_values(
                    execution_id=execution_id,
                    assignment_id=assignment_id,
                    attempt_id=attempt_id,
                    manager_id=manager_id,
                    manager_session_id=manager_session_id,
                    status=assignment_status,
                    now=overdue if operation == 'reconcile_pending_timeouts' else now,
                    accepted_at=accepted_at,
                    started_at=started_at,
                )
                execution['current_assignment_id'] = assignment_id
                execution['current_attempt_id'] = attempt_id
                assignments.append(assignment)
            executions.append(execution)

    await _insert_rows(session_factory, InstanceManagerRow, managers)
    await _insert_rows(session_factory, InstanceManagerLivenessRow, liveness_rows)
    await _insert_rows(session_factory, ExecutionRow, executions)
    await _insert_rows(session_factory, AssignmentRow, assignments)
    return len(executions), heartbeat_targets


def _allocation_spec(*, workload: str) -> dict[str, Any]:
    return {
        'version': 1,
        'allocation_type': EnvironmentAllocationType.DYNAMIC,
        'provider_type': EnvironmentProviderType.KUBERNETES,
        'provider_config_ref': 'benchmark/provider',
        'provider_config': {
            'provider_type': EnvironmentProviderType.KUBERNETES,
            'namespace': 'reconciler-capacity',
        },
        'code': {'mode': 'config_only'},
        'workload': workload,
    }


async def _seed_allocation_fixture(
    session_factory: async_sessionmaker,
    *,
    prefix: str,
    ordinary_count: int,
    batch_count: int,
    batch_execution_managers: int,
    scope_count: int,
    heartbeat_manager_limit: int,
) -> tuple[_FixtureSummary, list[LogicalManagerHeartbeatTarget]]:
    counts = allocation_fixture_counts(
        ordinary_count=ordinary_count,
        batch_count=batch_count,
        batch_execution_managers=batch_execution_managers,
    )
    top_level_count = counts['top_level_executions']
    if top_level_count == 0:
        return _FixtureSummary(logical_items=0), []
    scope_count = max(1, min(scope_count, top_level_count))
    now = utcnow()
    limits = {
        'manager_count': max(1, counts['managers'] * 2),
        'cpu_millicores': max(1, counts['managers'] * 4000),
        'memory_bytes': max(1, counts['managers'] * 4 * 1024**3),
        'slots': max(1, counts['managers'] * 40),
        'active_batches': max(1, batch_count * 2),
    }
    environments: list[dict[str, Any]] = []
    grants: list[dict[str, Any]] = []
    resource_usage: list[dict[str, Any]] = []
    for scope in range(scope_count):
        environment_id = f'{prefix}-e-{scope:02d}'
        owner_id = f'{prefix}-u-{scope:02d}'
        environments.append(
            {
                'environment_id': environment_id,
                'name': environment_id,
                'allocation_type': EnvironmentAllocationType.DYNAMIC,
                'provider_type': EnvironmentProviderType.KUBERNETES,
                'provider_config_ref': 'benchmark/provider',
                'resource_limits_json': limits,
                'default_timeouts_json': {},
                'config_version': 1,
                'status': EnvironmentStatus.ENABLED,
                'created_by': 'reconciler-capacity',
                'updated_by': 'reconciler-capacity',
                'created_at': now,
                'updated_at': now,
            }
        )
        grants.append(
            {
                'principal_id': owner_id,
                'environment_id': environment_id,
                'limits_json': limits,
                'enabled': True,
                'version': 1,
                'created_by': 'reconciler-capacity',
                'updated_by': 'reconciler-capacity',
                'created_at': now,
                'updated_at': now,
            }
        )
        resource_usage.extend(
            (
                {
                    'scope_key': f'environment:{environment_id}',
                    'scope_type': 'environment',
                    'environment_id': environment_id,
                    'principal_id': None,
                    'manager_count': 0,
                    'cpu_millicores': 0,
                    'memory_bytes': 0,
                    'slots': 0,
                    'active_batches': 0,
                    'version': 1,
                    'updated_at': now,
                },
                {
                    'scope_key': f'grant:{environment_id}:{owner_id}',
                    'scope_type': 'grant',
                    'environment_id': environment_id,
                    'principal_id': owner_id,
                    'manager_count': 0,
                    'cpu_millicores': 0,
                    'memory_bytes': 0,
                    'slots': 0,
                    'active_batches': 0,
                    'version': 1,
                    'updated_at': now,
                },
            )
        )

    executions: list[dict[str, Any]] = []
    allocations: list[dict[str, Any]] = []
    managers: list[dict[str, Any]] = []
    liveness_rows: list[dict[str, Any]] = []
    heartbeat_targets: list[LogicalManagerHeartbeatTarget] = []

    def add_allocation(
        *,
        execution_id: str,
        allocation_id: str,
        owner_id: str,
        environment_id: str,
        pool_id: str,
        workload: str,
        desired_manager_count: int,
    ) -> None:
        spec = _allocation_spec(workload=workload)
        spec_hash = canonical_sha256(spec)
        allocations.append(
            {
                'allocation_id': allocation_id,
                'top_level_execution_id': execution_id,
                'owner_principal_id': owner_id,
                'environment_id': environment_id,
                'pool_id': pool_id,
                'workload': workload,
                'status': EnvironmentAllocationStatus.REQUESTED,
                'desired_manager_count': desired_manager_count,
                'ready_manager_count': 0,
                'manager_count': desired_manager_count,
                'cpu_millicores': desired_manager_count * 1000,
                'memory_bytes': desired_manager_count * 1024**3,
                'slots': desired_manager_count * 20,
                'active_batches': int(workload == 'orchestration'),
                'allocation_spec_json': spec,
                'allocation_spec_hash': spec_hash,
                'config_version': 1,
                'status_version': 1,
                'last_progress_at': now,
                'created_at': now,
                'updated_at': now,
            }
        )
        for ordinal in range(desired_manager_count):
            manager_id = f'{allocation_id}-{ordinal}'
            session_id = f's-{hashlib.sha256(manager_id.encode()).hexdigest()[:24]}'
            manager, liveness = _manager_values(
                manager_id=manager_id,
                session_id=session_id,
                now=now,
                heartbeat_at=now,
                environment_id=environment_id,
                pool_id=pool_id,
                allocation_id=allocation_id,
                workload=workload,
                owner_principal_id=owner_id,
                allocation_spec_hash=spec_hash,
                capacity_total=20,
            )
            managers.append(manager)
            liveness_rows.append(liveness)
            if heartbeat_manager_limit == 0 or len(heartbeat_targets) < heartbeat_manager_limit:
                heartbeat_targets.append(
                    LogicalManagerHeartbeatTarget(
                        manager_id=manager_id,
                        session_id=session_id,
                        capacity_total=20,
                    )
                )

    for index in range(ordinary_count):
        scope = index % scope_count
        owner_id = f'{prefix}-u-{scope:02d}'
        environment_id = f'{prefix}-e-{scope:02d}'
        execution_id = f'{prefix}-ao-{index:08d}'
        allocation_id = f'{prefix}-aox-{index:08d}'
        spec_hash = canonical_sha256(_allocation_spec(workload='execution'))
        execution = _base_execution(
            execution_id,
            owner_principal_id=owner_id,
            status=ExecutionStatus.QUEUED,
            now=now,
            environment_id=environment_id,
            pool_id='execution',
            allocation_id=allocation_id,
            workload='execution',
            allocation_spec_hash=spec_hash,
        )
        execution['placement_json'] = {
            'environment': environment_id,
            'pool': 'execution',
            'allocation_id': allocation_id,
            'labels': {'workload': 'execution'},
        }
        executions.append(execution)
        add_allocation(
            execution_id=execution_id,
            allocation_id=allocation_id,
            owner_id=owner_id,
            environment_id=environment_id,
            pool_id='execution',
            workload='execution',
            desired_manager_count=1,
        )

    for index in range(batch_count):
        scope = (ordinary_count + index) % scope_count
        owner_id = f'{prefix}-u-{scope:02d}'
        environment_id = f'{prefix}-e-{scope:02d}'
        execution_id = f'{prefix}-ab-{index:08d}'
        orchestration_id = f'{prefix}-abo-{index:08d}'
        orchestration_hash = canonical_sha256(_allocation_spec(workload='orchestration'))
        execution = _base_execution(
            execution_id,
            owner_principal_id=owner_id,
            status=ExecutionStatus.QUEUED,
            now=now,
            spec_type='nexus_batch',
            environment_id=environment_id,
            pool_id='driver',
            allocation_id=orchestration_id,
            workload='orchestration',
            allocation_spec_hash=orchestration_hash,
        )
        execution['placement_json'] = {
            'environment': environment_id,
            'pool': 'driver',
            'allocation_id': orchestration_id,
            'labels': {'workload': 'orchestration'},
        }
        executions.append(execution)
        add_allocation(
            execution_id=execution_id,
            allocation_id=orchestration_id,
            owner_id=owner_id,
            environment_id=environment_id,
            pool_id='driver',
            workload='orchestration',
            desired_manager_count=1,
        )
        add_allocation(
            execution_id=execution_id,
            allocation_id=f'{prefix}-abe-{index:08d}',
            owner_id=owner_id,
            environment_id=environment_id,
            pool_id='execution',
            workload='execution',
            desired_manager_count=batch_execution_managers,
        )

    await _insert_rows(session_factory, EnvironmentRow, environments)
    await _insert_rows(session_factory, EnvironmentGrantRow, grants)
    await _insert_rows(session_factory, EnvironmentResourceUsageRow, resource_usage)
    await _insert_rows(session_factory, ExecutionRow, executions)
    await _insert_rows(session_factory, EnvironmentAllocationRow, allocations)
    await _insert_rows(session_factory, InstanceManagerRow, managers)
    await _insert_rows(session_factory, InstanceManagerLivenessRow, liveness_rows)
    return (
        _FixtureSummary(
            logical_items=len(allocations),
            executions=len(executions),
            allocations=len(allocations),
            managers=len(managers),
            heartbeat_eligible_managers=len(managers),
        ),
        heartbeat_targets,
    )


async def _seed_batch_fixture(
    session_factory: async_sessionmaker,
    *,
    prefix: str,
    parent_count: int,
    children_per_parent: int,
) -> tuple[_FixtureSummary, list[LogicalManagerHeartbeatTarget]]:
    counts = batch_fixture_counts(parent_count=parent_count, children_per_parent=children_per_parent)
    if parent_count == 0 or children_per_parent == 0:
        return _FixtureSummary(logical_items=0), []
    now = utcnow()
    finished_at = now - timedelta(seconds=1)
    parents: list[dict[str, Any]] = []
    parent_assignments: list[dict[str, Any]] = []
    batch_runs: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    managers: list[dict[str, Any]] = []
    liveness_rows: list[dict[str, Any]] = []
    heartbeat_targets: list[LogicalManagerHeartbeatTarget] = []
    for parent_index in range(parent_count):
        parent_execution_id = f'{prefix}-bp-{parent_index:06d}'
        assignment_id = f'{prefix}-ba-{parent_index:06d}'
        attempt_id = f'{prefix}-bt-{parent_index:06d}'
        owner_id = f'{prefix}-bu-{parent_index % 64:02d}'
        environment_id = f'{prefix}-be-{parent_index % 64:02d}'
        parent = _base_execution(
            parent_execution_id,
            owner_principal_id=owner_id,
            status=ExecutionStatus.RUNNING,
            now=now,
            spec_type='nexus_batch',
            environment_id=environment_id,
            pool_id='driver',
            workload='orchestration',
        )
        parent.update(
            current_assignment_id=assignment_id,
            current_attempt_id=attempt_id,
            started_at=now,
        )
        parents.append(parent)
        parent_assignments.append(
            {
                'assignment_id': assignment_id,
                'execution_id': parent_execution_id,
                'owner_principal_id': owner_id,
                'environment_id': environment_id,
                'pool_id': 'driver',
                'workload': 'orchestration',
                'attempt_id': attempt_id,
                'attempt_seq': 1,
                'manager_id': f'{prefix}-bm-{parent_index:06d}',
                'manager_session_id': f'{prefix}-bs-{parent_index:06d}',
                'status': AssignmentStatus.ACTIVE,
                'slot_cost': 1,
                'endpoint_url': 'http://batch-manager.invalid:8090',
                'endpoint_ready': True,
                'accepted_at': now,
                'started_at': now,
                'created_at': now,
                'updated_at': now,
            }
        )
        manager_id = f'{prefix}-bm-{parent_index:06d}'
        manager_session_id = f'{prefix}-bs-{parent_index:06d}'
        manager, liveness = _manager_values(
            manager_id=manager_id,
            session_id=manager_session_id,
            now=now,
            heartbeat_at=now,
            environment_id=environment_id,
            pool_id='driver',
            workload='orchestration',
            owner_principal_id=owner_id,
            capacity_total=1,
        )
        managers.append(manager)
        liveness_rows.append(liveness)
        heartbeat_targets.append(
            LogicalManagerHeartbeatTarget(
                manager_id=manager_id,
                session_id=manager_session_id,
                capacity_total=1,
            )
        )
        batch_runs.append(
            {
                'parent_execution_id': parent_execution_id,
                'shard_id': stable_shard(parent_execution_id, SHARD_COUNT),
                'generation_mode': 'local',
                'generation_status': 'complete',
                'generation_epoch': 1,
                'materialize_cursor_bytes': 0,
                'materialized_count': children_per_parent,
                'owner_principal_id': owner_id,
                'owner_tenant': 'capacity',
                'total_items': children_per_parent,
                'submitted_items': children_per_parent,
                'active_items': children_per_parent,
                'terminal_items': 0,
                'submission_closed': True,
                'control_state': 'active',
                'active_parent_attempt_id': attempt_id,
                'summary_version': 0,
                'last_progress_at': now,
                'created_at': now,
                'updated_at': now,
            }
        )
        for child_index in range(children_per_parent):
            item_id = f'{child_index:016d}'
            child_execution_id = f'{prefix}-bc-{parent_index:06d}-{child_index:08d}'
            child = _base_execution(
                child_execution_id,
                owner_principal_id=owner_id,
                status=ExecutionStatus.SUCCEEDED,
                now=now,
                environment_id=environment_id,
                pool_id='execution',
                workload='execution',
            )
            child.update(
                version=2,
                result_json={'success': True, 'index': child_index},
                finished_at=finished_at,
                cleanup_done_at=finished_at,
            )
            children.append(child)
            items.append(
                {
                    'parent_execution_id': parent_execution_id,
                    'item_id': item_id,
                    'task_ordinal': child_index,
                    'task_offset': child_index,
                    'task_length': 1,
                    'current_attempt_no': 0,
                    'current_child_execution_id': child_execution_id,
                    'spec_sha256': 'a' * 64,
                    'status': 'running',
                    'status_version': 1,
                    'child_execution_status': ExecutionStatus.RUNNING,
                    'child_execution_version': 1,
                    'metadata_json': {},
                    'created_at': now,
                    'updated_at': now,
                }
            )
            attempts.append(
                {
                    'parent_execution_id': parent_execution_id,
                    'item_id': item_id,
                    'attempt_no': 0,
                    'child_execution_id': child_execution_id,
                    'spec_sha256': 'a' * 64,
                    'status': ExecutionStatus.RUNNING,
                    'status_version': 1,
                    'created_at': now,
                    'updated_at': now,
                }
            )

    await _insert_rows(session_factory, InstanceManagerRow, managers)
    await _insert_rows(session_factory, InstanceManagerLivenessRow, liveness_rows)
    await _insert_rows(session_factory, ExecutionRow, parents)
    await _insert_rows(session_factory, AssignmentRow, parent_assignments)
    await _insert_rows(session_factory, BatchRunRow, batch_runs)
    await _insert_rows(session_factory, ExecutionRow, children)
    await _insert_rows(session_factory, BatchItemRow, items)
    await _insert_rows(session_factory, BatchItemAttemptRow, attempts)
    return _FixtureSummary(
        logical_items=counts['children'],
        executions=parent_count + counts['children'],
        managers=counts['managers'],
        heartbeat_eligible_managers=counts['managers'],
        batch_parents=parent_count,
        batch_children=counts['children'],
    ), heartbeat_targets


def _merge_fixture_summaries(*summaries: _FixtureSummary) -> _FixtureSummary:
    return _FixtureSummary(
        logical_items=sum(summary.logical_items for summary in summaries),
        executions=sum(summary.executions for summary in summaries),
        core_executions=sum(summary.core_executions for summary in summaries),
        allocations=sum(summary.allocations for summary in summaries),
        managers=sum(summary.managers for summary in summaries),
        heartbeat_eligible_managers=sum(summary.heartbeat_eligible_managers for summary in summaries),
        batch_parents=sum(summary.batch_parents for summary in summaries),
        batch_children=sum(summary.batch_children for summary in summaries),
    )


async def _assert_empty_database(session_factory: async_sessionmaker) -> None:
    async with session_factory() as session:
        counts = {
            model.__tablename__: int(await session.scalar(select(func.count()).select_from(model)) or 0)
            for model in (
                ExecutionRow,
                EnvironmentAllocationRow,
                BatchRunRow,
                ArtifactRow,
                ShardLeaseRow,
            )
        }
    nonempty = {name: count for name, count in counts.items() if count}
    if nonempty:
        raise RuntimeError(f'Reconciler capacity benchmark requires an empty dedicated database: {nonempty}')


async def _setup_fixture(
    args: argparse.Namespace,
    *,
    prefix: str,
) -> tuple[_FixtureSummary, list[LogicalManagerHeartbeatTarget]]:
    config = DistributedConfig(database_url=args.database_url, db_pool_size=4, db_max_overflow=4)
    engine = create_engine(config)
    if engine.dialect.name != 'mysql':
        await engine.dispose()
        raise ValueError(f'Reconciler capacity benchmark requires MySQL, got {engine.dialect.name}')
    session_factory = create_session_factory(engine)
    try:
        await create_schema(engine)
        await _assert_empty_database(session_factory)
        summaries: list[_FixtureSummary] = []
        heartbeat_targets: list[LogicalManagerHeartbeatTarget] = []
        if args.scenario in {'core', 'mixed'}:
            core_count, core_heartbeat_targets = await _seed_core_fixture(
                session_factory,
                prefix=prefix,
                items_per_operation=args.core_items_per_operation,
            )
            summaries.append(
                _FixtureSummary(
                    logical_items=core_count,
                    executions=core_count,
                    core_executions=core_count,
                    managers=3 * SHARD_COUNT,
                    heartbeat_eligible_managers=len(core_heartbeat_targets),
                )
            )
            heartbeat_targets.extend(core_heartbeat_targets)
        if args.scenario in {'allocation', 'mixed'}:
            allocation_summary, allocation_heartbeat_targets = await _seed_allocation_fixture(
                session_factory,
                prefix=prefix,
                ordinary_count=args.ordinary_count,
                batch_count=args.allocation_batch_count,
                batch_execution_managers=args.batch_execution_managers,
                scope_count=args.allocation_scope_count,
                heartbeat_manager_limit=args.heartbeat_manager_limit,
            )
            summaries.append(allocation_summary)
            heartbeat_targets.extend(allocation_heartbeat_targets)
        if args.scenario in {'batch', 'mixed'}:
            batch_summary, batch_heartbeat_targets = await _seed_batch_fixture(
                session_factory,
                prefix=prefix,
                parent_count=args.batch_parent_count,
                children_per_parent=args.batch_children_per_parent,
            )
            summaries.append(batch_summary)
            heartbeat_targets.extend(batch_heartbeat_targets)
        expected_heartbeat_targets = heartbeat_fixture_target_count(
            scenario=args.scenario,
            ordinary_count=args.ordinary_count,
            allocation_batch_count=args.allocation_batch_count,
            batch_execution_managers=args.batch_execution_managers,
            heartbeat_manager_limit=args.heartbeat_manager_limit,
            batch_parent_count=args.batch_parent_count,
        )
        if len(heartbeat_targets) != expected_heartbeat_targets:
            raise RuntimeError(
                f'Heartbeat target fixture mismatch: {len(heartbeat_targets)}/{expected_heartbeat_targets}'
            )
        summary = _merge_fixture_summaries(*summaries)
        validate_heartbeat_coverage(
            eligible_managers=summary.heartbeat_eligible_managers,
            target_count=len(heartbeat_targets),
            allow_partial=args.allow_partial_heartbeat_coverage,
        )
        return summary, heartbeat_targets
    finally:
        await engine.dispose()


def _service_config(args: argparse.Namespace) -> DistributedConfig:
    return DistributedConfig(
        database_url=args.database_url,
        db_pool_size=args.worker_pool_size,
        db_max_overflow=args.worker_max_overflow,
        shard_count=SHARD_COUNT,
        manager_healthy_ttl=timedelta(seconds=args.manager_healthy_ttl),
        manager_dead_ttl=timedelta(hours=1),
        assign_ack_timeout=timedelta(seconds=30),
        start_timeout=timedelta(seconds=120),
        reconciler_batch_size=args.reconciler_batch_size,
        reconciler_poll_interval=args.idle_poll_interval,
        reconciler_lease_ttl=timedelta(seconds=args.shard_lease_ttl),
        reconciler_lease_renew_interval=args.shard_lease_renew_interval,
        reconciler_shard_ownership_limit=args.ownership_limit,
        reconciler_shard_discover_interval=0.01,
        reconciler_shard_discover_jitter=0,
        reconciler_core_max_batches_per_tick=args.core_max_batches_per_tick,
        reconciler_core_time_budget=args.core_time_budget,
        reconciler_core_timeout=max(args.core_time_budget * 2, 30),
        environment_controller_batch_size=args.environment_batch_size,
        environment_operation_ttl=timedelta(seconds=args.environment_operation_ttl),
        environment_operation_lease_renew_interval=timedelta(seconds=args.environment_lease_renew_interval),
        environment_reconcile_timeout=max(30, args.environment_operation_ttl * 2),
        batch_lifecycle_page_size=args.batch_page_size,
        batch_lifecycle_parent_limit=args.batch_parent_limit,
        batch_lifecycle_max_pages_per_parent=args.batch_max_pages_per_parent,
        batch_lifecycle_scan_limit=args.batch_scan_limit,
        batch_lifecycle_claim_ttl=timedelta(seconds=args.batch_claim_ttl),
        batch_lifecycle_claim_renew_interval=args.batch_claim_renew_interval,
        batch_lifecycle_poll_interval=args.idle_poll_interval,
        batch_lifecycle_reconcile_timeout=max(30, args.batch_claim_ttl * 2),
        artifact_maintenance_enabled=False,
    )


def _heartbeat_config(
    *,
    database_url: str,
    pool_size: int,
    max_overflow: int,
    healthy_ttl_seconds: float,
    self_fence_timeout_seconds: float,
) -> DistributedConfig:
    dead_ttl_seconds = max(
        healthy_ttl_seconds * 2,
        healthy_ttl_seconds + 60,
        self_fence_timeout_seconds + 60,
    )
    return DistributedConfig(
        database_url=database_url,
        db_pool_size=pool_size,
        db_max_overflow=max_overflow,
        manager_healthy_ttl=timedelta(seconds=healthy_ttl_seconds),
        manager_dead_ttl=timedelta(seconds=dead_ttl_seconds),
        manager_self_fence_timeout=timedelta(seconds=self_fence_timeout_seconds),
    )


def _create_service(
    *,
    args: argparse.Namespace,
    session_factory: async_sessionmaker,
    owner_id: str,
) -> tuple[ReconcilerService, _DeterministicEnvironmentAllocator, _DatabaseRunResultPublisher]:
    config = _service_config(args)
    noop_resolver = _NoopController()
    allocator = _DeterministicEnvironmentAllocator(delay_seconds=args.provider_delay_ms / 1000)
    artifact_publisher = _DatabaseRunResultPublisher(session_factory)
    service = ReconcilerService(
        session_factory=session_factory,
        config=reconciler_config_view(config),
        owner_id=owner_id,
        environment_controller=EnvironmentController(
            session_factory=session_factory,
            config=environment_config_view(config),
            allocators={EnvironmentProviderType.KUBERNETES: cast(Any, allocator)},
            client_resolver=cast(Any, noop_resolver),
            owner_id=f'{owner_id}:environment',
        ),
        dependency_build_controller=_NoopController(),
        workload_client_resolver=cast(Any, noop_resolver),
        batch_lifecycle=BatchLifecycleReconciler(
            session_factory=session_factory,
            config=batch_lifecycle_config_view(config),
            owner_id=f'{owner_id}:batch',
            artifact_client=artifact_publisher,
        ),
    )
    return service, allocator, artifact_publisher


async def _tick_loop(
    *,
    loop_name: str,
    tick: Callable[[], Awaitable[int]],
    stop_event: asyncio.Event,
    idle_poll_interval: float,
    stats: dict[str, dict[str, int]],
) -> None:
    loop_stats = stats.setdefault(loop_name, {'ticks': 0, 'changed': 0, 'idle_ticks': 0})
    while not stop_event.is_set():
        changed = await retry_tick(tick)
        loop_stats['ticks'] += 1
        loop_stats['changed'] += changed
        if changed == 0:
            loop_stats['idle_ticks'] += 1
            await asyncio.sleep(idle_poll_interval)


def _worker_process(
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
            _run_worker_process(
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
                'kind': 'worker_error',
                'index': index,
                'pid': os.getpid(),
                'error': f'{type(error).__name__}: {error}',
                'traceback': traceback.format_exc(),
            }
        )
        raise


def _create_worker_process(
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
        target=_worker_process,
        args=(index, args, prefix, ready_queue, result_queue, start_event, stop_event),
        name=f'nexus-reconciler-capacity-{index:02d}',
    )


async def _run_worker_process(
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
    config = _service_config(args)
    engine = create_engine(config)
    session_factory = create_session_factory(engine)
    owner_id = f'{prefix}-r-{index:02d}-p{os.getpid()}'
    service, allocator, artifact_publisher = _create_service(
        args=args,
        session_factory=session_factory,
        owner_id=owner_id,
    )
    lease_task: asyncio.Task[None] | None = None
    closed = False
    try:
        owned: list[int] = []
        local_stop_event = asyncio.Event()
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
            raise RuntimeError(f'Reconciler {index} did not acquire {args.ownership_limit} shards: {owned}')
        lease_task = asyncio.create_task(
            service._run_lease_heartbeat(),
            name=f'reconciler-capacity-lease:{index}',
        )
        ready_queue.put(
            {
                'kind': 'worker_ready',
                'index': index,
                'pid': os.getpid(),
                'owned_shards': owned,
            }
        )
        await asyncio.to_thread(start_event.wait)
        stats: dict[str, dict[str, int]] = {}
        tasks: list[asyncio.Task[None]] = []
        for loop_name in scenario_loop_names(args.scenario):
            tick: Callable[[], Awaitable[int]]
            worker_count = args.controller_workers if loop_name in {'environment', 'batch'} else 1
            if loop_name == 'core':
                tick = service._core_tick_once
            elif loop_name == 'environment':
                tick = service._environment_tick_once
            elif loop_name == 'batch':
                tick = service._batch_tick_once
            else:
                raise AssertionError(loop_name)
            for worker_index in range(worker_count):
                task_name = loop_name if worker_index == 0 else f'{loop_name}:{worker_index}'
                tasks.append(
                    asyncio.create_task(
                        _tick_loop(
                            loop_name=loop_name,
                            tick=tick,
                            stop_event=local_stop_event,
                            idle_poll_interval=args.idle_poll_interval,
                            stats=stats,
                        ),
                        name=f'reconciler-capacity:{task_name}',
                    )
                )
        await asyncio.to_thread(stop_event.wait)
        local_stop_event.set()
        await asyncio.gather(*tasks)
        service.stop()
        if lease_task is not None:
            await lease_task
        await service.close()
        closed = True
        result_queue.put(
            {
                'kind': 'worker_done',
                'index': index,
                'pid': os.getpid(),
                'stats': stats,
                'provider_calls': allocator.report(),
                'artifact_publish_calls': artifact_publisher.publish_calls,
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
    config = _heartbeat_config(
        database_url=args.database_url,
        pool_size=args.heartbeat_pool_size,
        max_overflow=args.heartbeat_max_overflow,
        healthy_ttl_seconds=args.manager_healthy_ttl,
        self_fence_timeout_seconds=args.manager_self_fence_timeout,
    )
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


async def _request_allocation_release(session_factory: async_sessionmaker, *, prefix: str) -> int:
    now = utcnow()
    async with session_factory() as session:
        result = await session.execute(
            update(ExecutionRow)
            .where(
                ExecutionRow.execution_id.like(f'{prefix}-a%'),
                ExecutionRow.status == ExecutionStatus.QUEUED,
            )
            .values(
                status=ExecutionStatus.SUCCEEDED,
                version=ExecutionRow.version + 1,
                result_json={'success': True, 'source': 'capacity-coordinator'},
                finished_at=now,
                updated_at=now,
            )
        )
        await session.commit()
        return int(getattr(result, 'rowcount', 0) or 0)


async def _read_progress(
    session_factory: async_sessionmaker,
    *,
    args: argparse.Namespace,
    prefix: str,
    summary: _FixtureSummary,
    release_requested: bool,
) -> tuple[dict[str, int], bool, bool]:
    progress: dict[str, int] = {}
    complete = True
    should_release = False
    async with session_factory() as session:
        if args.scenario in {'core', 'mixed'}:
            core_done = int(
                await session.scalar(
                    select(func.count())
                    .select_from(ExecutionRow)
                    .where(
                        ExecutionRow.execution_id.like(f'{prefix}-c-%'),
                        ExecutionRow.status.in_(TERMINAL_EXECUTION_STATUSES),
                        ExecutionRow.cleanup_done_at.is_not(None),
                    )
                )
                or 0
            )
            progress['core_done'] = core_done
            progress['core_total'] = summary.core_executions
            complete = complete and core_done == summary.core_executions

        if args.scenario in {'allocation', 'mixed'}:
            allocation_ready = int(
                await session.scalar(
                    select(func.count())
                    .select_from(EnvironmentAllocationRow)
                    .where(
                        EnvironmentAllocationRow.allocation_id.like(f'{prefix}-a%'),
                        EnvironmentAllocationRow.status == EnvironmentAllocationStatus.READY,
                    )
                )
                or 0
            )
            allocation_released = int(
                await session.scalar(
                    select(func.count())
                    .select_from(EnvironmentAllocationRow)
                    .where(
                        EnvironmentAllocationRow.allocation_id.like(f'{prefix}-a%'),
                        EnvironmentAllocationRow.status == EnvironmentAllocationStatus.RELEASED,
                    )
                )
                or 0
            )
            progress['allocations_ready'] = allocation_ready
            progress['allocations_released'] = allocation_released
            progress['allocations_total'] = summary.allocations
            should_release = not release_requested and allocation_ready == summary.allocations
            complete = complete and release_requested and allocation_released == summary.allocations

        if args.scenario in {'batch', 'mixed'}:
            batch_finalized = int(
                await session.scalar(
                    select(func.count())
                    .select_from(BatchRunRow)
                    .where(
                        BatchRunRow.parent_execution_id.like(f'{prefix}-bp-%'),
                        BatchRunRow.control_state == 'finalized',
                    )
                )
                or 0
            )
            child_synchronized = int(
                await session.scalar(
                    select(func.count())
                    .select_from(BatchItemRow)
                    .where(
                        BatchItemRow.parent_execution_id.like(f'{prefix}-bp-%'),
                        BatchItemRow.status == ExecutionStatus.SUCCEEDED,
                    )
                )
                or 0
            )
            progress['batch_finalized'] = batch_finalized
            progress['batch_parents'] = summary.batch_parents
            progress['batch_children_synchronized'] = child_synchronized
            progress['batch_children'] = summary.batch_children
            complete = (
                complete and batch_finalized == summary.batch_parents and child_synchronized == summary.batch_children
            )
    return progress, complete, should_release


async def _validate_core(
    session: AsyncSession,
    *,
    prefix: str,
    items_per_operation: int,
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    event_profiles: dict[str, Any] = {}
    for code, expected_status in _CORE_EXPECTED_STATUSES.items():
        count = int(
            await session.scalar(
                select(func.count())
                .select_from(ExecutionRow)
                .where(
                    ExecutionRow.execution_id.like(f'{prefix}-c-{code}-%'),
                    ExecutionRow.status == expected_status,
                    ExecutionRow.cleanup_done_at.is_not(None),
                )
            )
            or 0
        )
        status_counts[code] = count
        if count != items_per_operation:
            raise RuntimeError(
                f'Core validation mismatch: operation={code} status={expected_status} '
                f'actual={count} expected={items_per_operation}'
            )
        event_profiles[code] = await _validate_execution_event_profile(
            session,
            execution_pattern=f'{prefix}-c-{code}-%',
            expected_executions=items_per_operation,
            expected_multipliers=_CORE_EXPECTED_EVENT_MULTIPLIERS[code],
        )
    active_assignments = int(
        await session.scalar(
            select(func.count())
            .select_from(AssignmentRow)
            .join(ExecutionRow, ExecutionRow.execution_id == AssignmentRow.execution_id)
            .where(
                ExecutionRow.execution_id.like(f'{prefix}-c-%'),
                AssignmentRow.status.in_([AssignmentStatus.PENDING, AssignmentStatus.ACTIVE]),
            )
        )
        or 0
    )
    unfinished_commands = int(
        await session.scalar(
            select(func.count())
            .select_from(ExecutionCommandRow)
            .where(
                ExecutionCommandRow.execution_id.like(f'{prefix}-c-%'),
                ExecutionCommandRow.status.in_(['pending', 'claimed']),
            )
        )
        or 0
    )
    if active_assignments or unfinished_commands:
        raise RuntimeError(
            f'Core validation left active work: assignments={active_assignments} commands={unfinished_commands}'
        )
    return {
        'status_counts': status_counts,
        'event_profiles': event_profiles,
        'active_assignments': active_assignments,
        'unfinished_commands': unfinished_commands,
    }


async def _validate_allocations(
    session: AsyncSession,
    *,
    prefix: str,
    expected: int,
) -> dict[str, Any]:
    released = int(
        await session.scalar(
            select(func.count())
            .select_from(EnvironmentAllocationRow)
            .where(
                EnvironmentAllocationRow.allocation_id.like(f'{prefix}-a%'),
                EnvironmentAllocationRow.status == EnvironmentAllocationStatus.RELEASED,
                EnvironmentAllocationRow.resources_reserved_at.is_not(None),
                EnvironmentAllocationRow.resources_released_at.is_not(None),
                EnvironmentAllocationRow.operation_id.is_(None),
                EnvironmentAllocationRow.operation_owner.is_(None),
                EnvironmentAllocationRow.operation_lease_until.is_(None),
                EnvironmentAllocationRow.last_error_code.is_(None),
            )
        )
        or 0
    )
    nonzero_usage = int(
        await session.scalar(
            select(func.count())
            .select_from(EnvironmentResourceUsageRow)
            .where(
                EnvironmentResourceUsageRow.environment_id.like(f'{prefix}-e-%'),
                (
                    (EnvironmentResourceUsageRow.manager_count != 0)
                    | (EnvironmentResourceUsageRow.cpu_millicores != 0)
                    | (EnvironmentResourceUsageRow.memory_bytes != 0)
                    | (EnvironmentResourceUsageRow.slots != 0)
                    | (EnvironmentResourceUsageRow.active_batches != 0)
                ),
            )
        )
        or 0
    )
    if released != expected or nonzero_usage:
        raise RuntimeError(
            f'Allocation validation mismatch: released={released}/{expected} nonzero_usage={nonzero_usage}'
        )
    return {
        'released': released,
        'nonzero_resource_usage_rows': nonzero_usage,
    }


async def _validate_batches(
    session: AsyncSession,
    *,
    prefix: str,
    expected_parents: int,
    expected_children: int,
) -> dict[str, Any]:
    finalized_parents = int(
        await session.scalar(
            select(func.count())
            .select_from(BatchRunRow)
            .join(ExecutionRow, ExecutionRow.execution_id == BatchRunRow.parent_execution_id)
            .where(
                BatchRunRow.parent_execution_id.like(f'{prefix}-bp-%'),
                BatchRunRow.control_state == 'finalized',
                BatchRunRow.active_items == 0,
                BatchRunRow.terminal_items == BatchRunRow.total_items,
                BatchRunRow.lifecycle_owner_id.is_(None),
                BatchRunRow.lifecycle_operation_id.is_(None),
                BatchRunRow.lifecycle_lease_expires_at.is_(None),
                ExecutionRow.status == ExecutionStatus.SUCCEEDED,
                ExecutionRow.result_artifact_id.is_not(None),
            )
        )
        or 0
    )
    synchronized_items = int(
        await session.scalar(
            select(func.count())
            .select_from(BatchItemRow)
            .where(
                BatchItemRow.parent_execution_id.like(f'{prefix}-bp-%'),
                BatchItemRow.status == ExecutionStatus.SUCCEEDED,
                BatchItemRow.child_execution_status == ExecutionStatus.SUCCEEDED,
                BatchItemRow.child_execution_version == 2,
            )
        )
        or 0
    )
    synchronized_attempts = int(
        await session.scalar(
            select(func.count())
            .select_from(BatchItemAttemptRow)
            .where(
                BatchItemAttemptRow.parent_execution_id.like(f'{prefix}-bp-%'),
                BatchItemAttemptRow.status == ExecutionStatus.SUCCEEDED,
            )
        )
        or 0
    )
    released_parent_assignments = int(
        await session.scalar(
            select(func.count())
            .select_from(AssignmentRow)
            .where(
                AssignmentRow.execution_id.like(f'{prefix}-bp-%'),
                AssignmentRow.status == AssignmentStatus.RELEASED,
            )
        )
        or 0
    )
    if (
        finalized_parents != expected_parents
        or synchronized_items != expected_children
        or synchronized_attempts != expected_children
        or released_parent_assignments != expected_parents
    ):
        raise RuntimeError(
            'Batch validation mismatch: '
            f'parents={finalized_parents}/{expected_parents} '
            f'items={synchronized_items}/{expected_children} '
            f'attempts={synchronized_attempts}/{expected_children} '
            f'assignments={released_parent_assignments}/{expected_parents}'
        )
    event_profile = await _validate_execution_event_profile(
        session,
        execution_pattern=f'{prefix}-bp-%',
        expected_executions=expected_parents,
        expected_multipliers={
            'assignment.status.changed': 1,
            'execution.status.changed': 1,
        },
    )
    artifact_rows, unique_artifacts, unique_producers = (
        await session.execute(
            select(
                func.count(ArtifactRow.artifact_id),
                func.count(func.distinct(ArtifactRow.artifact_id)),
                func.count(func.distinct(ArtifactRow.producer_key)),
            )
            .select_from(ExecutionRow)
            .join(ArtifactRow, ArtifactRow.artifact_id == ExecutionRow.result_artifact_id)
            .where(
                ExecutionRow.execution_id.like(f'{prefix}-bp-%'),
                ArtifactRow.kind == 'run_result',
                ArtifactRow.status == 'ready',
            )
        )
    ).one()
    artifact_identity = {
        'ready_rows': int(artifact_rows or 0),
        'unique_artifacts': int(unique_artifacts or 0),
        'unique_producers': int(unique_producers or 0),
    }
    if set(artifact_identity.values()) != {expected_parents}:
        raise RuntimeError(f'Batch run_result Artifact identity mismatch: {artifact_identity}/{expected_parents}')
    return {
        'finalized_parents': finalized_parents,
        'synchronized_items': synchronized_items,
        'synchronized_attempts': synchronized_attempts,
        'released_parent_assignments': released_parent_assignments,
        'event_profile': event_profile,
        'run_result_artifacts': artifact_identity,
    }


async def _validate_execution_event_profile(
    session: AsyncSession,
    *,
    execution_pattern: str,
    expected_executions: int,
    expected_multipliers: Mapping[str, int],
) -> dict[str, Any]:
    expected_total = sum(expected_multipliers.values())
    event_types = sorted(expected_multipliers)
    aggregate_columns = [
        func.sum(case((ExecutionEventRow.type == event_type, 1), else_=0)).label(f'event_{index}')
        for index, event_type in enumerate(event_types)
    ]
    profiles = (
        select(
            ExecutionRow.execution_id.label('execution_id'),
            func.count(ExecutionEventRow.event_id).label('event_total'),
            *aggregate_columns,
        )
        .outerjoin(ExecutionEventRow, ExecutionEventRow.execution_id == ExecutionRow.execution_id)
        .where(ExecutionRow.execution_id.like(execution_pattern))
        .group_by(ExecutionRow.execution_id)
        .subquery()
    )
    invalid_conditions = [profiles.c.event_total != expected_total]
    invalid_conditions.extend(
        profiles.c[f'event_{index}'] != expected_multipliers[event_type] for index, event_type in enumerate(event_types)
    )
    execution_count, invalid_executions = (
        await session.execute(
            select(
                func.count(),
                func.sum(case((or_(*invalid_conditions), 1), else_=0)),
            ).select_from(profiles)
        )
    ).one()
    actual_type_counts = {
        str(event_type): int(count)
        for event_type, count in (
            await session.execute(
                select(ExecutionEventRow.type, func.count())
                .where(ExecutionEventRow.execution_id.like(execution_pattern))
                .group_by(ExecutionEventRow.type)
                .order_by(ExecutionEventRow.type)
            )
        ).all()
    }
    expected_type_counts = {
        event_type: multiplier * expected_executions for event_type, multiplier in expected_multipliers.items()
    }
    result = {
        'executions': int(execution_count or 0),
        'invalid_executions': int(invalid_executions or 0),
        'event_type_counts': actual_type_counts,
    }
    if (
        result['executions'] != expected_executions
        or result['invalid_executions']
        or actual_type_counts != expected_type_counts
    ):
        raise RuntimeError(
            'Execution event profile mismatch: '
            f'pattern={execution_pattern} actual={result} expected_executions={expected_executions} '
            f'expected_types={expected_type_counts}'
        )
    return result


async def _validate(
    args: argparse.Namespace,
    *,
    prefix: str,
    summary: _FixtureSummary,
) -> dict[str, Any]:
    config = DistributedConfig(database_url=args.database_url, db_pool_size=2, db_max_overflow=0)
    engine = create_engine(config)
    session_factory = create_session_factory(engine)
    result: dict[str, Any] = {}
    try:
        async with session_factory() as session:
            if args.scenario in {'core', 'mixed'}:
                result['core'] = await _validate_core(
                    session,
                    prefix=prefix,
                    items_per_operation=args.core_items_per_operation,
                )
            if args.scenario in {'allocation', 'mixed'}:
                result['allocation'] = await _validate_allocations(
                    session,
                    prefix=prefix,
                    expected=summary.allocations,
                )
            if args.scenario in {'batch', 'mixed'}:
                result['batch'] = await _validate_batches(
                    session,
                    prefix=prefix,
                    expected_parents=summary.batch_parents,
                    expected_children=summary.batch_children,
                )
            shard_histogram = {
                str(shard_id): int(count)
                for shard_id, count in (
                    await session.execute(
                        select(ExecutionRow.shard_id, func.count())
                        .where(ExecutionRow.execution_id.like(f'{prefix}-%'))
                        .group_by(ExecutionRow.shard_id)
                        .order_by(ExecutionRow.shard_id)
                    )
                ).all()
            }
            if sum(shard_histogram.values()) != summary.executions:
                raise RuntimeError(
                    f'Reconciler execution shard fixture mismatch: {sum(shard_histogram.values())}/{summary.executions}'
                )
            result['execution_shard_histogram'] = shard_histogram
    finally:
        await engine.dispose()
    return result


def _wait_for_ready(
    *,
    args: argparse.Namespace,
    processes: list[Any],
    ready_queue: Any,
    result_queue: Any,
    expect_heartbeat: bool,
) -> tuple[list[dict[str, Any]], int | None]:
    workers: list[dict[str, Any]] = []
    heartbeat_pid: int | None = None
    deadline = time.monotonic() + args.startup_timeout
    while len(workers) < args.replicas or (expect_heartbeat and heartbeat_pid is None):
        if time.monotonic() >= deadline:
            states = [(process.name, process.pid, process.exitcode) for process in processes]
            raise TimeoutError(f'Reconciler process startup timed out: {states}')
        try:
            message = ready_queue.get(timeout=0.5)
        except queue.Empty:
            while True:
                try:
                    failure = result_queue.get_nowait()
                except queue.Empty:
                    break
                if str(failure.get('kind', '')).endswith('_error'):
                    raise RuntimeError(json.dumps(failure, indent=2)) from None
            continue
        if message['kind'] == 'worker_ready':
            workers.append(message)
        elif message['kind'] == 'heartbeat_ready':
            heartbeat_pid = int(message['pid'])
    workers.sort(key=lambda item: int(item['index']))
    shard_ids = [int(shard_id) for worker in workers for shard_id in worker['owned_shards']]
    if len(shard_ids) != SHARD_COUNT or len(set(shard_ids)) != SHARD_COUNT:
        raise RuntimeError(f'Reconciler shard ownership did not converge uniquely: {workers}')
    return workers, heartbeat_pid


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
                f'Replacement Reconciler exited before ready: index={worker_index} exitcode={process.exitcode}'
            )
        try:
            message = ready_queue.get(timeout=0.5)
        except queue.Empty:
            for result in _drain_failures(result_queue):
                if result['kind'] in {'worker_done', 'heartbeat_done'}:
                    raise RuntimeError(
                        f'Unexpected process completion while waiting for replacement: {result}'
                    ) from None
            continue
        if message['kind'] == 'worker_ready' and int(message['index']) == worker_index:
            return message
        raise RuntimeError(f'Unexpected readiness message while waiting for replacement: {message}')
    raise TimeoutError(f'Replacement Reconciler startup timed out: index={worker_index} pid={process.pid}')


def _signal_live_processes(processes: list[Any], stop_events: list[Any]) -> None:
    for process, stop_event in zip(processes, stop_events, strict=True):
        if process.is_alive():
            stop_event.set()


def _drain_failures(result_queue: Any) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    while True:
        try:
            message = result_queue.get_nowait()
        except queue.Empty:
            return messages
        if str(message.get('kind', '')).endswith('_error'):
            raise RuntimeError(json.dumps(message, indent=2))
        messages.append(message)


def _completed_processes(processes: list[Any]) -> list[tuple[str, int | None, int]]:
    return [(process.name, process.pid, process.exitcode) for process in processes if process.exitcode is not None]


def run(args: argparse.Namespace) -> dict[str, Any]:
    return asyncio.run(_run_capacity(args))


async def _run_capacity(args: argparse.Namespace) -> dict[str, Any]:
    validate_topology(replicas=args.replicas, ownership_limit=args.ownership_limit)
    validate_recovery_config(
        replicas=args.replicas,
        worker_index=args.restart_worker_index,
        restart_after_seconds=args.restart_worker_after_seconds,
    )
    validate_timing_config(
        manager_heartbeat_interval=args.manager_heartbeat_interval,
        manager_healthy_ttl=args.manager_healthy_ttl,
        manager_self_fence_timeout=args.manager_self_fence_timeout,
        shard_lease_ttl=args.shard_lease_ttl,
        shard_lease_renew_interval=args.shard_lease_renew_interval,
        environment_operation_ttl=args.environment_operation_ttl,
        environment_lease_renew_interval=args.environment_lease_renew_interval,
        batch_claim_ttl=args.batch_claim_ttl,
        batch_claim_renew_interval=args.batch_claim_renew_interval,
    )
    prefix = capacity_fixture_prefix(args.scenario)
    summary, heartbeat_targets = await _setup_fixture(args, prefix=prefix)

    context = mp.get_context('fork')
    ready_queue = context.Queue()
    result_queue = context.Queue()
    start_event = context.Event()
    worker_stop_events = [context.Event() for _ in range(args.replicas)]
    heartbeat_stop_event = context.Event()
    workers = [
        _create_worker_process(
            context,
            index=index,
            args=args,
            prefix=prefix,
            ready_queue=ready_queue,
            result_queue=result_queue,
            start_event=start_event,
            stop_event=worker_stop_events[index],
        )
        for index in range(args.replicas)
    ]
    heartbeat = (
        context.Process(
            target=_heartbeat_process,
            args=(args, heartbeat_targets, ready_queue, result_queue, heartbeat_stop_event),
            name='nexus-reconciler-capacity-heartbeat',
        )
        if heartbeat_targets
        else None
    )
    processes = [*workers, *([heartbeat] if heartbeat is not None else [])]
    for process in processes:
        process.start()

    monitor_config = DistributedConfig(database_url=args.database_url, db_pool_size=2, db_max_overflow=0)
    monitor_engine = create_engine(monitor_config)
    monitor_factory = create_session_factory(monitor_engine)
    worker_reports: dict[int, dict[str, Any]] = {}
    heartbeat_report: dict[str, Any] | None = None
    recovery_report: dict[str, Any] | None = None
    release_requested = False
    final_progress: dict[str, int] = {}
    try:
        ready, heartbeat_pid = _wait_for_ready(
            args=args,
            processes=processes,
            ready_queue=ready_queue,
            result_queue=result_queue,
            expect_heartbeat=heartbeat is not None,
        )
        initial_worker_pids = [int(item['pid']) for item in ready]
        worker_pids = list(initial_worker_pids)
        process_sampler = ProcessSampler(worker_pids)
        all_process_sampler = ProcessSampler([*worker_pids, *([heartbeat_pid] if heartbeat_pid is not None else [])])
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
        while True:
            for message in _drain_failures(result_queue):
                if message['kind'] == 'worker_done':
                    worker_reports[int(message['index'])] = message
                elif message['kind'] == 'heartbeat_done':
                    heartbeat_report = message['report']
            if (
                args.restart_worker_after_seconds > 0
                and recovery_report is None
                and time.perf_counter() - started >= args.restart_worker_after_seconds
            ):
                worker_index = args.restart_worker_index
                old_process = workers[worker_index]
                old_pid = int(old_process.pid or 0)
                old_shards = [int(shard_id) for shard_id in ready[worker_index]['owned_shards']]
                process_sampler.sample()
                all_process_sampler.sample()
                killed_at = time.perf_counter()
                old_process.kill()
                old_process.join(5)
                if old_process.is_alive():
                    raise TimeoutError(f'Reconciler worker did not stop after kill: pid={old_pid}')

                replacement_stop_event = context.Event()
                replacement = _create_worker_process(
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
                workers[worker_index] = replacement
                worker_stop_events[worker_index] = replacement_stop_event
                processes[processes.index(old_process)] = replacement
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
                        f'Replacement Reconciler recovered different shards: {replacement_shards}/{old_shards}'
                    )
                ready[worker_index] = replacement_ready
                worker_pids[worker_index] = new_pid
                process_sampler.replace_pid(old_pid, new_pid)
                all_process_sampler.replace_pid(old_pid, new_pid)
                recovered_at = time.perf_counter()
                recovery_report = {
                    'worker_index': worker_index,
                    'old_pid': old_pid,
                    'new_pid': new_pid,
                    'old_owned_shards': old_shards,
                    'replacement_owned_shards': replacement_shards,
                    'overlapping_shards': sorted(set(old_shards) & set(replacement_shards)),
                    'killed_at_seconds': round(killed_at - started, 6),
                    'replacement_ready_at_seconds': round(recovered_at - started, 6),
                    'outage_seconds': round(recovered_at - killed_at, 6),
                    'shard_lease_ttl_seconds': args.shard_lease_ttl,
                    'environment_operation_ttl_seconds': args.environment_operation_ttl,
                    'batch_claim_ttl_seconds': args.batch_claim_ttl,
                }
            completed = _completed_processes(processes)
            if completed:
                raise RuntimeError(f'Reconciler capacity process exited before completion: {completed}')
            final_progress, complete, should_release = await _read_progress(
                monitor_factory,
                args=args,
                prefix=prefix,
                summary=summary,
                release_requested=release_requested,
            )
            if should_release:
                released_executions = await _request_allocation_release(monitor_factory, prefix=prefix)
                expected_top_levels = args.ordinary_count + args.allocation_batch_count
                if released_executions != expected_top_levels:
                    raise RuntimeError(
                        f'Allocation release trigger mismatch: {released_executions}/{expected_top_levels}'
                    )
                release_requested = True
                continue
            process_sampler.sample()
            all_process_sampler.sample()
            if database_process_sampler is not None:
                database_process_sampler.sample()
            database_current = await database_status(monitor_factory)
            for name in DATABASE_GAUGES:
                gauge_peaks[name] = max(gauge_peaks.get(name, 0), database_current.get(name, 0))
            if complete:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Reconciler made no progress before timeout: {final_progress}')
            if time.monotonic() - last_log_at >= args.progress_log_interval:
                print(f'RECONCILER_PROGRESS {json.dumps(final_progress, sort_keys=True)}', file=sys.stderr, flush=True)
                last_log_at = time.monotonic()
            await asyncio.sleep(args.progress_poll_interval)

        elapsed = time.perf_counter() - started
        finished_epoch = time.time()
        if args.restart_worker_after_seconds > 0 and recovery_report is None:
            raise RuntimeError('Reconciler recovery was configured but the fixture completed before restart')
        stop_processes = [*workers, *([heartbeat] if heartbeat is not None else [])]
        stop_events = [*worker_stop_events, *([heartbeat_stop_event] if heartbeat is not None else [])]
        _signal_live_processes(stop_processes, stop_events)
        deadline = time.monotonic() + args.shutdown_timeout
        while len(worker_reports) < args.replicas or (heartbeat is not None and heartbeat_report is None):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f'Reconciler worker reports timed out: workers={len(worker_reports)}/{args.replicas} '
                    f'heartbeat={heartbeat_report is not None}'
                )
            try:
                message = result_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if str(message.get('kind', '')).endswith('_error'):
                raise RuntimeError(json.dumps(message, indent=2))
            if message['kind'] == 'worker_done':
                worker_reports[int(message['index'])] = message
            elif message['kind'] == 'heartbeat_done':
                heartbeat_report = message['report']
        join_processes(processes, timeout=args.shutdown_timeout)
        bad_exits = [(process.name, process.pid, process.exitcode) for process in processes if process.exitcode != 0]
        if bad_exits:
            raise RuntimeError(f'Reconciler worker process failures: {bad_exits}')
        validation = await _validate(args, prefix=prefix, summary=summary)
        database_after_cleanup = await database_status(monitor_factory)
        if heartbeat_report is not None:
            heartbeat_failures = {
                key: heartbeat_report[key]
                for key in ('fatal_failures', 'fenced_managers', 'self_fenced_managers', 'stale_managers')
                if heartbeat_report[key]
            }
            if heartbeat_failures:
                raise RuntimeError(f'Manager heartbeat load was unhealthy: {heartbeat_failures}')
        loop_totals: dict[str, dict[str, int]] = {}
        provider_calls = {'create': 0, 'observe': 0, 'delete': 0, 'observe_deletion': 0}
        artifact_publish_calls = 0
        for report in worker_reports.values():
            for loop_name, stats in report['stats'].items():
                totals = loop_totals.setdefault(loop_name, {'ticks': 0, 'changed': 0, 'idle_ticks': 0})
                for key in totals:
                    totals[key] += int(stats[key])
            for operation, count in report['provider_calls'].items():
                provider_calls[operation] += int(count)
            artifact_publish_calls += int(report['artifact_publish_calls'])
        external_call_validation = {
            'worker_reports_complete': recovery_report is None,
            'allocation_minimum_calls_met': None,
            'batch_publish_minimum_met': None,
        }
        if recovery_report is None and summary.allocations:
            allocation_minimum_calls_met = all(
                provider_calls[operation] >= summary.allocations
                for operation in ('create', 'delete', 'observe_deletion')
            )
            if not allocation_minimum_calls_met:
                raise RuntimeError(
                    f'Environment provider calls did not cover every allocation: {provider_calls}/{summary.allocations}'
                )
            external_call_validation['allocation_minimum_calls_met'] = True
        if recovery_report is None and summary.batch_parents:
            batch_publish_minimum_met = artifact_publish_calls >= summary.batch_parents
            if not batch_publish_minimum_met:
                raise RuntimeError(
                    f'Batch result publish calls did not cover every parent: {artifact_publish_calls}/'
                    f'{summary.batch_parents}'
                )
            external_call_validation['batch_publish_minimum_met'] = True
        return {
            'source_identity': source_identity(Path(__file__).resolve().parents[3], _SOURCE_FILES),
            'database_identity': database_details,
            'scenario': args.scenario,
            'replicas': args.replicas,
            'ownership_limit': args.ownership_limit,
            'shard_count': SHARD_COUNT,
            'controller_workers_per_process': args.controller_workers,
            'fixture': {
                'logical_items': summary.logical_items,
                'executions': summary.executions,
                'core_executions': summary.core_executions,
                'allocations': summary.allocations,
                'managers': summary.managers,
                'heartbeat_eligible_managers': summary.heartbeat_eligible_managers,
                'batch_parents': summary.batch_parents,
                'batch_children': summary.batch_children,
            },
            'manager_liveness_profile': {
                'heartbeat_targets': len(heartbeat_targets),
                'heartbeat_eligible_managers': summary.heartbeat_eligible_managers,
                'coverage_ratio': (
                    round(len(heartbeat_targets) / summary.heartbeat_eligible_managers, 6)
                    if summary.heartbeat_eligible_managers
                    else 1.0
                ),
                'partial_coverage': len(heartbeat_targets) < summary.heartbeat_eligible_managers,
                'heartbeat_interval_seconds': args.manager_heartbeat_interval,
                'healthy_ttl_seconds': args.manager_healthy_ttl,
                'self_fence_timeout_seconds': args.manager_self_fence_timeout,
            },
            'elapsed_seconds': round(elapsed, 6),
            'logical_items_per_second': round(summary.logical_items / elapsed, 2),
            'start_epoch': started_epoch,
            'end_epoch': finished_epoch,
            'progress': final_progress,
            'initial_worker_pids': initial_worker_pids,
            'worker_pids': worker_pids,
            'recovery': recovery_report,
            'heartbeat_pid': heartbeat_pid,
            'worker_loops': loop_totals,
            'provider_calls': provider_calls,
            'artifact_publish_calls': artifact_publish_calls,
            'external_call_validation': external_call_validation,
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
        stop_processes = [*workers, *([heartbeat] if heartbeat is not None else [])]
        stop_events = [*worker_stop_events, *([heartbeat_stop_event] if heartbeat is not None else [])]
        _signal_live_processes(stop_processes, stop_events)
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
    parser.add_argument('--scenario', choices=('core', 'allocation', 'batch', 'mixed'), required=True)
    parser.add_argument('--replicas', type=int, default=1)
    parser.add_argument('--ownership-limit', type=int, default=64)
    parser.add_argument('--controller-workers', type=int, default=1)
    parser.add_argument('--core-items-per-operation', type=int, default=10_000)
    parser.add_argument('--ordinary-count', type=int, default=10_000)
    parser.add_argument('--allocation-batch-count', type=int, default=200)
    parser.add_argument('--batch-execution-managers', type=int, default=50)
    parser.add_argument('--allocation-scope-count', type=int, default=64)
    parser.add_argument(
        '--heartbeat-manager-limit',
        type=int,
        default=0,
        help='Maximum eligible Managers to heartbeat; 0 means all.',
    )
    parser.add_argument('--allow-partial-heartbeat-coverage', action='store_true')
    parser.add_argument('--batch-parent-count', type=int, default=1_000)
    parser.add_argument('--batch-children-per-parent', type=int, default=100)
    parser.add_argument('--reconciler-batch-size', type=int, default=500)
    parser.add_argument('--environment-batch-size', type=int, default=20)
    parser.add_argument('--batch-page-size', type=int, default=500)
    parser.add_argument('--batch-parent-limit', type=int, default=50)
    parser.add_argument('--batch-max-pages-per-parent', type=int, default=2)
    parser.add_argument('--batch-scan-limit', type=int, default=5_000)
    parser.add_argument('--core-max-batches-per-tick', type=int, default=4)
    parser.add_argument('--core-time-budget', type=float, default=60)
    parser.add_argument('--provider-delay-ms', type=float, default=0)
    parser.add_argument('--worker-pool-size', type=int, default=2)
    parser.add_argument('--worker-max-overflow', type=int, default=2)
    parser.add_argument('--heartbeat-pool-size', type=int, default=8)
    parser.add_argument('--heartbeat-max-overflow', type=int, default=8)
    parser.add_argument('--database-process-pid', type=int, default=0)
    parser.add_argument('--manager-heartbeat-interval', type=float, default=5)
    parser.add_argument('--manager-healthy-ttl', type=float, default=15)
    parser.add_argument('--manager-self-fence-timeout', type=float, default=30)
    parser.add_argument('--shard-lease-ttl', type=float, default=30)
    parser.add_argument('--shard-lease-renew-interval', type=float, default=5)
    parser.add_argument('--environment-operation-ttl', type=float, default=30)
    parser.add_argument('--environment-lease-renew-interval', type=float, default=5)
    parser.add_argument('--batch-claim-ttl', type=float, default=60)
    parser.add_argument('--batch-claim-renew-interval', type=float, default=10)
    parser.add_argument('--restart-worker-after-seconds', type=float, default=0)
    parser.add_argument('--restart-worker-index', type=int, default=0)
    parser.add_argument('--idle-poll-interval', type=float, default=0.01)
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
    for name in (
        'controller_workers',
        'core_items_per_operation',
        'reconciler_batch_size',
        'environment_batch_size',
        'batch_page_size',
        'batch_parent_limit',
        'batch_max_pages_per_parent',
        'batch_scan_limit',
        'worker_pool_size',
        'heartbeat_pool_size',
    ):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    for name in (
        'ordinary_count',
        'allocation_batch_count',
        'batch_execution_managers',
        'heartbeat_manager_limit',
        'batch_parent_count',
        'batch_children_per_parent',
    ):
        if getattr(args, name) < 0:
            parser.error(f'--{name.replace("_", "-")} must be non-negative')
    if args.database_process_pid < 0:
        parser.error('--database-process-pid must be non-negative')
    try:
        validate_timing_config(
            manager_heartbeat_interval=args.manager_heartbeat_interval,
            manager_healthy_ttl=args.manager_healthy_ttl,
            manager_self_fence_timeout=args.manager_self_fence_timeout,
            shard_lease_ttl=args.shard_lease_ttl,
            shard_lease_renew_interval=args.shard_lease_renew_interval,
            environment_operation_ttl=args.environment_operation_ttl,
            environment_lease_renew_interval=args.environment_lease_renew_interval,
            batch_claim_ttl=args.batch_claim_ttl,
            batch_claim_renew_interval=args.batch_claim_renew_interval,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


__all__ = [
    'SHARD_COUNT',
    'TOPOLOGIES',
    'CORE_OPERATIONS',
    'allocation_fixture_counts',
    'batch_fixture_counts',
    'capacity_fixture_prefix',
    'core_heartbeat_target_count',
    'core_fixture_counts',
    'heartbeat_fixture_target_count',
    'scenario_loop_names',
    'validate_heartbeat_coverage',
    'validate_recovery_config',
    'validate_timing_config',
    'validate_topology',
]


if __name__ == '__main__':
    print(json.dumps(run(_parse_args()), indent=2, sort_keys=True))
