from __future__ import annotations

import asyncio
import time

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.batch.lifecycle import BatchLifecycleReconciler
from nexus.distributed.config import DistributedConfig
from nexus.distributed.control_plane.reconciler import ReconcilerService
from nexus.distributed.control_plane.scheduler import SchedulerService
from nexus.distributed.dependency_build.controller import DependencyBuildController
from nexus.distributed.environment.allocator import AllocationSpecConflictError
from nexus.distributed.environment.controller import EnvironmentController
from nexus.distributed.environment.kubernetes import WorkloadPermanentError
from nexus.distributed.environment.models import EnvironmentAllocationStatus
from nexus.distributed.environment.ports import AllocationSnapshot, AllocatorObservation
from nexus.distributed.environment.providers import WorkloadClientResolver
from nexus.distributed.execution.models import (
    ExecutionMode,
    ExecutionStatus,
)
from nexus.distributed.persistence.orm import (
    BatchRunRow,
    EnvironmentAllocationRow,
    EnvironmentGrantRow,
    EnvironmentResourceUsageRow,
    EnvironmentRow,
    ExecutionRow,
    ShardLeaseRow,
    create_schema,
)
from nexus.distributed.repositories.allocation import (
    AllocationRepository,
    AllocationResources,
    CreateEnvironmentAllocation,
    EnvironmentAllocationClaim,
)
from nexus.distributed.repositories.batch import BatchLifecycleClaim, BatchRepository
from nexus.distributed.repositories.execution import CreateExecution, ExecutionRepository
from nexus.distributed.repositories.manager import ManagerRepository
from nexus.distributed.repositories.quota import QuotaRepository
from nexus.distributed.repositories.scheduler import SchedulerRepository
from nexus.distributed.repositories.shard_lease import ShardLeaseRepository
from nexus.distributed.utils.clock import utcnow


async def _session_factory():
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    await create_schema(engine)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


def _add_resource_usage_rows(session, *, environment_id: str, principal_id: str, now) -> None:
    session.add_all(
        [
            EnvironmentResourceUsageRow(
                scope_key=f'environment:{environment_id}',
                scope_type='environment',
                environment_id=environment_id,
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
                scope_key=f'grant:{environment_id}:{principal_id}',
                scope_type='grant',
                environment_id=environment_id,
                principal_id=principal_id,
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


class _NoopController:
    async def reconcile_once(self) -> int:
        return 0

    async def close(self) -> None:
        pass


class _NoopWorkloadClientResolver:
    async def close(self) -> None:
        pass


async def test_reconciler_composes_independent_controllers_with_one_workload_client_resolver(tmp_path) -> None:
    engine, session_factory = await _session_factory()

    class Client:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1

    class BatchLifecycle:
        async def reconcile_once(self):
            return 0

        async def close(self):
            pass

    token_file = tmp_path / 'token'
    ca_file = tmp_path / 'ca.crt'
    token_file.write_text('token')
    ca_file.write_text('ca')
    config = DistributedConfig(
        artifact_maintenance_enabled=False,
        environment_namespace='workloads',
        kubernetes_token_file=str(token_file),
        kubernetes_ca_file=str(ca_file),
    )
    client = Client()
    resolver = WorkloadClientResolver(
        config=config,
        clients={('kubernetes', 'kubernetes/main'): client},
    )
    service = ReconcilerService(
        session_factory=session_factory,
        config=config,
        batch_lifecycle=BatchLifecycle(),
        workload_client_resolver=resolver,
        environment_controller=EnvironmentController(
            session_factory=session_factory,
            config=config,
            client_resolver=resolver,
        ),
        dependency_build_controller=DependencyBuildController(
            session_factory=session_factory,
            config=config,
            client_resolver=resolver,
        ),
    )
    try:
        assert service.environment_controller.client_resolver is resolver
        assert service.dependency_build_controller.client_resolver is resolver
        assert not hasattr(service.environment_controller, '_dependency_build_controller')

        await service.close()

        assert client.close_calls == 1
    finally:
        await engine.dispose()


class _FakeEnvironmentAllocator:
    def __init__(self) -> None:
        self.exists = False
        self.create_calls = 0
        self.delete_calls = 0

    async def observe(self, allocation):
        return AllocatorObservation(
            exists=self.exists,
            provider_ready_replicas=allocation.desired_manager_count if self.exists else 0,
            provider_resource_ref={'workload': allocation.allocation_id} if self.exists else None,
        )

    async def observe_deletion(self, allocation):
        return await self.observe(allocation)

    async def create(self, allocation):
        self.create_calls += 1
        self.exists = True
        return await self.observe(allocation)

    async def delete(self, allocation):
        del allocation
        self.delete_calls += 1
        self.exists = False

    async def close(self):
        pass


class _PermanentFailureAllocator(_FakeEnvironmentAllocator):
    async def create(self, allocation):
        del allocation
        self.create_calls += 1
        raise WorkloadPermanentError('kubernetes_api_403: forbidden')


class _PermanentDeleteFailureAllocator(_FakeEnvironmentAllocator):
    async def delete(self, allocation):
        del allocation
        self.delete_calls += 1
        raise WorkloadPermanentError('kubernetes_api_403: delete forbidden')


class _SlowCreateAllocator(_FakeEnvironmentAllocator):
    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay

    async def create(self, allocation):
        await asyncio.sleep(self.delay)
        return await super().create(allocation)


class _BlockingCreateAllocator(_FakeEnvironmentAllocator):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def create(self, allocation):
        self.started.set()
        await self.release.wait()
        return await super().create(allocation)


async def _seed_dynamic_controller_allocations(
    session,
    *,
    allocation_ids: list[str],
    now,
) -> None:
    session.add(
        EnvironmentRow(
            environment_id='environment-slow',
            name='environment-slow',
            allocation_type='dynamic',
            provider_type='kubernetes',
            provider_config_ref='cluster-a',
            resource_limits_json={'manager_count': len(allocation_ids), 'slots': len(allocation_ids)},
            default_timeouts_json={},
            config_version=1,
            status='enabled',
            created_by='admin',
            updated_by='admin',
            created_at=now,
            updated_at=now,
        )
    )
    session.add(
        EnvironmentGrantRow(
            principal_id='principal-slow',
            environment_id='environment-slow',
            limits_json={'manager_count': len(allocation_ids), 'slots': len(allocation_ids)},
            enabled=True,
            version=1,
            created_by='admin',
            updated_by='admin',
            created_at=now,
            updated_at=now,
        )
    )
    _add_resource_usage_rows(
        session,
        environment_id='environment-slow',
        principal_id='principal-slow',
        now=now,
    )
    for ordinal, allocation_id in enumerate(allocation_ids):
        execution_id = f'execution-slow-{ordinal}'
        session.add(
            ExecutionRow(
                execution_id=execution_id,
                owner_principal_id='principal-slow',
                created_by='principal-slow',
                environment_id='environment-slow',
                pool_id='execution',
                allocation_id=allocation_id,
                workload='execution',
                allocation_spec_hash=str(ordinal) * 64,
                version=1,
                shard_id=ordinal,
                mode=ExecutionMode.JOB,
                spec_type='nexus_single',
                instance_mode='asyncio',
                spec_json={},
                driver_config_json={},
                placement_json={},
                status=ExecutionStatus.QUEUED,
                route_status='none',
                slot_cost=1,
                next_attempt_seq=1,
                attempt_count=0,
                max_attempts=1,
                next_event_seq=1,
                priority=0,
                next_schedule_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        await AllocationRepository(session).create_allocation(
            CreateEnvironmentAllocation(
                allocation_id=allocation_id,
                top_level_execution_id=execution_id,
                owner_principal_id='principal-slow',
                environment_id='environment-slow',
                pool_id='execution',
                workload='execution',
                desired_manager_count=1,
                resources=AllocationResources(
                    manager_count=1,
                    cpu_millicores=1000,
                    memory_bytes=1024,
                    slots=1,
                ),
                allocation_spec={
                    'version': 1,
                    'allocation_type': 'dynamic',
                    'provider_type': 'kubernetes',
                    'provider_config_ref': 'cluster-a',
                    'code': {'mode': 'config_only'},
                },
                allocation_spec_hash=str(ordinal) * 64,
            ),
            now=now,
        )
    await session.commit()


async def test_environment_controller_rejects_changed_environment_for_legacy_allocation_snapshot() -> None:
    engine, session_factory = await _session_factory()
    now = utcnow()
    controller = EnvironmentController(
        session_factory=session_factory,
        config=DistributedConfig(),
        client_resolver=MagicMock(),
    )
    try:
        async with session_factory() as session:
            session.add(
                EnvironmentRow(
                    environment_id='environment-changed',
                    name='environment-changed',
                    allocation_type='dynamic',
                    provider_type='kubernetes',
                    provider_config_ref='cluster-new',
                    resource_limits_json={},
                    default_timeouts_json={},
                    config_version=2,
                    status='enabled',
                    created_by='admin',
                    updated_by='admin',
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                EnvironmentAllocationRow(
                    allocation_id='allocation-legacy',
                    top_level_execution_id='execution-legacy',
                    owner_principal_id='principal-a',
                    environment_id='environment-changed',
                    pool_id='execution',
                    workload='execution',
                    status='requested',
                    desired_manager_count=1,
                    ready_manager_count=0,
                    manager_count=1,
                    cpu_millicores=1000,
                    memory_bytes=1024,
                    slots=1,
                    active_batches=0,
                    allocation_spec_json={'version': 1, 'code': {'mode': 'config_only'}},
                    allocation_spec_hash='a' * 64,
                    config_version=1,
                    status_version=1,
                    last_progress_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

        with pytest.raises(AllocationSpecConflictError, match='allocation_provider_snapshot_missing'):
            await controller._load_snapshot('allocation-legacy')
    finally:
        await controller.close()
        await engine.dispose()


async def test_environment_controller_reconciles_dynamic_allocation_to_ready_and_released() -> None:
    engine, session_factory = await _session_factory()
    now = utcnow()
    allocator = _FakeEnvironmentAllocator()
    controller = EnvironmentController(
        session_factory=session_factory,
        config=DistributedConfig(environment_operation_ttl=timedelta(seconds=30)),
        allocators={'kubernetes': allocator},
        client_resolver=MagicMock(),
        owner_id='reconciler-test',
    )
    try:
        async with session_factory() as session:
            session.add(
                EnvironmentRow(
                    environment_id='environment-a',
                    name='environment-a',
                    allocation_type='dynamic',
                    provider_type='kubernetes',
                    provider_config_ref='cluster-a',
                    resource_limits_json={'manager_count': 2, 'slots': 4},
                    default_timeouts_json={},
                    config_version=1,
                    status='enabled',
                    created_by='admin',
                    updated_by='admin',
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                EnvironmentGrantRow(
                    principal_id='principal-a',
                    environment_id='environment-a',
                    limits_json={'manager_count': 2, 'slots': 4},
                    enabled=True,
                    version=1,
                    created_by='admin',
                    updated_by='admin',
                    created_at=now,
                    updated_at=now,
                )
            )
            _add_resource_usage_rows(
                session,
                environment_id='environment-a',
                principal_id='principal-a',
                now=now,
            )
            session.add(
                ExecutionRow(
                    execution_id='execution-a',
                    owner_principal_id='principal-a',
                    created_by='principal-a',
                    environment_id='environment-a',
                    pool_id='execution',
                    allocation_id='allocation-a',
                    workload='execution',
                    allocation_spec_hash='a' * 64,
                    version=1,
                    shard_id=0,
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec_json={},
                    driver_config_json={},
                    placement_json={},
                    status=ExecutionStatus.QUEUED,
                    route_status='none',
                    slot_cost=1,
                    next_attempt_seq=1,
                    attempt_count=0,
                    max_attempts=1,
                    next_event_seq=1,
                    priority=0,
                    next_schedule_at=now,
                    queue_ttl_seconds=30,
                    created_at=now,
                    updated_at=now,
                )
            )
            await AllocationRepository(session).create_allocation(
                CreateEnvironmentAllocation(
                    allocation_id='allocation-a',
                    top_level_execution_id='execution-a',
                    owner_principal_id='principal-a',
                    environment_id='environment-a',
                    pool_id='execution',
                    workload='execution',
                    desired_manager_count=2,
                    resources=AllocationResources(
                        manager_count=2,
                        cpu_millicores=2000,
                        memory_bytes=2048,
                        slots=4,
                    ),
                    allocation_spec={
                        'version': 1,
                        'code': {'mode': 'config_only'},
                        'artifact_read_credential': {'secret_name': 'allocation-a-read', 'token_key': 'token'},
                    },
                    allocation_spec_hash='a' * 64,
                ),
                now=now,
            )
            await session.commit()

        assert await controller.reconcile_once() == 1
        async with session_factory() as session:
            allocation = await session.get(EnvironmentAllocationRow, 'allocation-a')
            environment_usage = await session.get(EnvironmentResourceUsageRow, 'environment:environment-a')
        assert allocation.status == EnvironmentAllocationStatus.PROVISIONING
        assert environment_usage.manager_count == 2
        assert allocator.create_calls == 1

        async with session_factory() as session:
            await ManagerRepository(session).register_session(
                manager_id='allocation-a-0',
                session_id='session-0',
                endpoint_url='http://allocation-a-0',
                supported_modes=[ExecutionMode.JOB],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                capacity_total=2,
                force_replace=True,
                owner_principal_id='principal-a',
                environment_id='environment-a',
                pool_id='execution',
                allocation_id='allocation-a',
                workload='execution',
                allocation_spec_hash='wrong',
                now=utcnow(),
            )
            await session.commit()

        await controller.reconcile_once()
        async with session_factory() as session:
            allocation = await session.get(EnvironmentAllocationRow, 'allocation-a')
        assert allocation.status == EnvironmentAllocationStatus.PROVISIONING
        assert allocation.ready_manager_count == 0

        async with session_factory() as session:
            await ManagerRepository(session).register_session(
                manager_id='allocation-a-0',
                session_id='session-0-replaced',
                endpoint_url='http://allocation-a-0',
                supported_modes=[ExecutionMode.JOB],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                capacity_total=2,
                force_replace=True,
                owner_principal_id='principal-a',
                environment_id='environment-a',
                pool_id='execution',
                allocation_id='allocation-a',
                workload='execution',
                allocation_spec_hash='a' * 64,
                now=utcnow(),
            )
            await session.commit()

        await controller.reconcile_once()
        async with session_factory() as session:
            allocation = await session.get(EnvironmentAllocationRow, 'allocation-a')
        assert allocation.ready_manager_count == 1

        async with session_factory() as session:
            await ManagerRepository(session).register_session(
                manager_id='allocation-a-1',
                session_id='session-1',
                endpoint_url='http://allocation-a-1',
                supported_modes=[ExecutionMode.JOB],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                capacity_total=2,
                force_replace=True,
                owner_principal_id='principal-a',
                environment_id='environment-a',
                pool_id='execution',
                allocation_id='allocation-a',
                workload='execution',
                allocation_spec_hash='a' * 64,
                now=utcnow(),
            )
            await session.commit()

        await controller.reconcile_once()
        async with session_factory() as session:
            allocation = await session.get(EnvironmentAllocationRow, 'allocation-a')
            execution = await session.get(ExecutionRow, 'execution-a')
        assert allocation.status == EnvironmentAllocationStatus.READY
        assert allocation.ready_manager_count == 2
        assert execution.first_schedulable_at is None
        assert execution.queue_expires_at is None

        async with session_factory() as session:
            execution = await session.get(ExecutionRow, 'execution-a')
            execution.status = ExecutionStatus.SUCCEEDED
            execution.finished_at = utcnow()
            execution.updated_at = utcnow()
            await session.commit()

        await controller.reconcile_once()
        await controller.reconcile_once()
        await controller.reconcile_once()
        async with session_factory() as session:
            allocation = await session.get(EnvironmentAllocationRow, 'allocation-a')
            environment_usage = await session.get(EnvironmentResourceUsageRow, 'environment:environment-a')
        assert allocation.status == EnvironmentAllocationStatus.RELEASED
        assert environment_usage.manager_count == 0
        assert allocator.delete_calls >= 1
    finally:
        await controller.close()
        await engine.dispose()


async def test_environment_controller_claims_and_renews_one_slow_allocation_at_a_time() -> None:
    engine, session_factory = await _session_factory()
    now = utcnow()
    allocator = _SlowCreateAllocator(delay=0.15)
    controller = EnvironmentController(
        session_factory=session_factory,
        config=DistributedConfig(
            environment_controller_batch_size=2,
            environment_operation_ttl=timedelta(milliseconds=100),
            environment_operation_lease_renew_interval=timedelta(milliseconds=20),
        ),
        allocators={'kubernetes': allocator},
        client_resolver=MagicMock(),
        owner_id='slow-controller',
    )
    try:
        async with session_factory() as session:
            await _seed_dynamic_controller_allocations(
                session,
                allocation_ids=['allocation-slow-0', 'allocation-slow-1'],
                now=now,
            )

        assert await controller.reconcile_once() == 2
        async with session_factory() as session:
            rows = [
                await session.get(EnvironmentAllocationRow, allocation_id)
                for allocation_id in ('allocation-slow-0', 'allocation-slow-1')
            ]
        assert [row.status for row in rows] == [
            EnvironmentAllocationStatus.PROVISIONING,
            EnvironmentAllocationStatus.PROVISIONING,
        ]
        assert allocator.create_calls == 2
    finally:
        await controller.close()
        await engine.dispose()


async def test_environment_controller_stops_writing_after_allocation_lease_loss(monkeypatch) -> None:
    engine, session_factory = await _session_factory()
    now = utcnow()
    allocator = _BlockingCreateAllocator()
    lease_lost = asyncio.Event()
    original_renew = AllocationRepository.renew_claim

    async def lose_after_provider_starts(self, claim, **kwargs):
        if allocator.started.is_set():
            lease_lost.set()
            return False
        return await original_renew(self, claim, **kwargs)

    monkeypatch.setattr(AllocationRepository, 'renew_claim', lose_after_provider_starts)
    controller = EnvironmentController(
        session_factory=session_factory,
        config=DistributedConfig(
            environment_controller_batch_size=1,
            environment_operation_ttl=timedelta(milliseconds=100),
            environment_operation_lease_renew_interval=timedelta(milliseconds=20),
        ),
        allocators={'kubernetes': allocator},
        client_resolver=MagicMock(),
        owner_id='lease-losing-controller',
    )
    try:
        async with session_factory() as session:
            await _seed_dynamic_controller_allocations(
                session,
                allocation_ids=['allocation-lease-lost'],
                now=now,
            )

        reconcile = asyncio.create_task(controller.reconcile_once())
        await asyncio.wait_for(allocator.started.wait(), timeout=1)
        await asyncio.wait_for(lease_lost.wait(), timeout=1)
        allocator.release.set()
        assert await reconcile == 1

        async with session_factory() as session:
            allocation = await session.get(EnvironmentAllocationRow, 'allocation-lease-lost')
            execution = await session.get(ExecutionRow, 'execution-slow-0')
        assert allocation.status == EnvironmentAllocationStatus.REQUESTED
        assert allocation.provider_resource_ref_json is None
        assert execution.status == ExecutionStatus.QUEUED
        assert lease_lost.is_set()
    finally:
        allocator.release.set()
        await controller.close()
        await engine.dispose()


async def test_stale_environment_owner_cannot_fail_execution_after_allocation_takeover() -> None:
    engine, session_factory = await _session_factory()
    now = utcnow()
    controller = EnvironmentController(
        session_factory=session_factory,
        config=DistributedConfig(),
        allocators={'kubernetes': _FakeEnvironmentAllocator()},
        client_resolver=MagicMock(),
        owner_id='controller-a',
    )
    try:
        async with session_factory() as session:
            await _seed_dynamic_controller_allocations(
                session,
                allocation_ids=['allocation-takeover'],
                now=now,
            )
            first_claim = await AllocationRepository(session).claim_allocation(
                'allocation-takeover',
                operation_owner='controller-a',
                operation_ttl=timedelta(seconds=1),
                now=now,
            )
            await session.commit()

        snapshot = await controller._load_snapshot('allocation-takeover')
        async with session_factory() as session:
            replacement_claim = await AllocationRepository(session).claim_allocation(
                'allocation-takeover',
                operation_owner='controller-b',
                operation_ttl=timedelta(seconds=30),
                now=now + timedelta(seconds=2),
            )
            await session.commit()

        await controller._fail_provisioning(
            first_claim,
            snapshot,
            WorkloadPermanentError('old owner failed late'),
        )

        async with session_factory() as session:
            allocation = await session.get(EnvironmentAllocationRow, 'allocation-takeover')
            execution = await session.get(ExecutionRow, 'execution-slow-0')
        assert allocation.status == EnvironmentAllocationStatus.REQUESTED
        assert allocation.operation_id == replacement_claim.operation_id
        assert execution.status == ExecutionStatus.QUEUED
        assert execution.error_json is None
    finally:
        await controller.close()
        await engine.dispose()


async def test_environment_controller_releases_uncreated_draining_allocation_with_failed_dependency_build(
    monkeypatch,
) -> None:
    allocator = _FakeEnvironmentAllocator()
    allocator.exists = True
    controller = EnvironmentController(
        session_factory=None,  # type: ignore[arg-type]
        config=DistributedConfig(),
        allocators={'kubernetes': allocator},
        client_resolver=MagicMock(),
    )
    transitions: list[str] = []

    async def record_transition(_claim, status, **_kwargs):
        transitions.append(status)

    monkeypatch.setattr(controller, '_transition', record_transition)
    now = utcnow()
    claim = EnvironmentAllocationClaim(
        allocation_id='allocation-failed-build',
        operation_id='operation-failed-build',
        operation_owner='reconciler-test',
        operation_lease_until=now + timedelta(seconds=30),
        status=EnvironmentAllocationStatus.DRAINING,
        status_version=1,
    )
    allocation = AllocationSnapshot(
        allocation_id=claim.allocation_id,
        top_level_execution_id='execution-failed-build',
        owner_principal_id='principal-a',
        environment_id='environment-a',
        allocation_type='dynamic',
        provider_type='kubernetes',
        provider_config_ref='kubernetes/default',
        pool_id='execution',
        workload='execution',
        status=EnvironmentAllocationStatus.DRAINING,
        desired_manager_count=1,
        ready_manager_count=0,
        cpu_millicores=1000,
        memory_bytes=1024,
        slots=1,
        allocation_spec={'version': 1, 'code': {'mode': 'source_package'}},
        allocation_spec_hash='a' * 64,
        provider_resource_ref=None,
        execution_status=ExecutionStatus.FAILED,
        dependency_build_id='build-failed',
        dependency_build_status='failed',
    )

    try:
        await controller._reconcile_claim(claim, allocation)
    finally:
        await controller.close()

    assert allocator.delete_calls == 0
    assert transitions == [EnvironmentAllocationStatus.RELEASED]


async def test_environment_controller_waits_for_retryable_dependency_build_failure(monkeypatch) -> None:
    allocator = _FakeEnvironmentAllocator()
    controller = EnvironmentController(
        session_factory=None,  # type: ignore[arg-type]
        config=DistributedConfig(),
        allocators={'kubernetes': allocator},
        client_resolver=MagicMock(),
    )
    observations = 0

    async def record_observation(*_args, **_kwargs):
        nonlocal observations
        observations += 1

    monkeypatch.setattr(controller, '_complete_observation', record_observation)
    now = utcnow()
    claim = EnvironmentAllocationClaim(
        allocation_id='allocation-retrying-build',
        operation_id='operation-retrying-build',
        operation_owner='reconciler-test',
        operation_lease_until=now + timedelta(seconds=30),
        status=EnvironmentAllocationStatus.REQUESTED,
        status_version=1,
    )
    allocation = AllocationSnapshot(
        allocation_id=claim.allocation_id,
        top_level_execution_id='execution-retrying-build',
        owner_principal_id='principal-a',
        environment_id='environment-a',
        allocation_type='dynamic',
        provider_type='kubernetes',
        provider_config_ref='kubernetes/default',
        pool_id='execution',
        workload='execution',
        status=EnvironmentAllocationStatus.REQUESTED,
        desired_manager_count=1,
        ready_manager_count=0,
        cpu_millicores=1000,
        memory_bytes=1024,
        slots=1,
        allocation_spec={'version': 1, 'code': {'mode': 'source_package'}},
        allocation_spec_hash='a' * 64,
        provider_resource_ref=None,
        execution_status=ExecutionStatus.QUEUED,
        dependency_build_id='build-retrying',
        dependency_build_status='failed',
        dependency_build_next_retry_at=now + timedelta(seconds=15),
    )

    try:
        await controller._reconcile_claim(claim, allocation)
    finally:
        await controller.close()

    assert observations == 1
    assert allocator.create_calls == 0


@pytest.mark.parametrize('dependency_status', ['pending', 'building', 'skipped'])
async def test_environment_controller_gates_batch_child_until_module_dependencies_are_ready(
    monkeypatch,
    dependency_status,
) -> None:
    allocator = _FakeEnvironmentAllocator()
    controller = EnvironmentController(
        session_factory=None,  # type: ignore[arg-type]
        config=DistributedConfig(),
        allocators={'kubernetes': allocator},
        client_resolver=MagicMock(),
    )
    observations = 0

    async def record_observation(*_args, **_kwargs):
        nonlocal observations
        observations += 1

    monkeypatch.setattr(controller, '_complete_observation', record_observation)
    now = utcnow()
    claim = EnvironmentAllocationClaim(
        allocation_id='allocation-batch-child',
        operation_id='operation-batch-child',
        operation_owner='reconciler-test',
        operation_lease_until=now + timedelta(seconds=30),
        status=EnvironmentAllocationStatus.REQUESTED,
        status_version=1,
    )
    allocation = AllocationSnapshot(
        allocation_id=claim.allocation_id,
        top_level_execution_id='batch-parent',
        owner_principal_id='principal-a',
        environment_id='environment-a',
        allocation_type='dynamic',
        provider_type='kubernetes',
        provider_config_ref='kubernetes/default',
        pool_id='execution',
        workload='execution',
        status=EnvironmentAllocationStatus.REQUESTED,
        desired_manager_count=1,
        ready_manager_count=0,
        cpu_millicores=1000,
        memory_bytes=1024,
        slots=1,
        allocation_spec={'version': 1, 'code': {'mode': 'source_package'}},
        allocation_spec_hash='a' * 64,
        provider_resource_ref=None,
        execution_status=ExecutionStatus.RUNNING,
        batch_control_state='active',
        batch_dependency_status=dependency_status,
        dependency_build_id='build-ready',
        dependency_build_status='ready',
    )

    try:
        await controller._reconcile_claim(claim, allocation)
    finally:
        await controller.close()

    assert allocator.create_calls == 0
    assert observations == 1


async def test_environment_controller_fails_execution_on_permanent_provider_error() -> None:
    engine, session_factory = await _session_factory()
    now = utcnow()
    allocator = _PermanentFailureAllocator()
    controller = EnvironmentController(
        session_factory=session_factory,
        config=DistributedConfig(environment_operation_ttl=timedelta(seconds=30)),
        allocators={'kubernetes': allocator},
        client_resolver=MagicMock(),
        owner_id='permanent-error-test',
    )
    try:
        async with session_factory() as session:
            session.add(
                EnvironmentRow(
                    environment_id='environment-permanent-error',
                    name='environment-permanent-error',
                    allocation_type='dynamic',
                    provider_type='kubernetes',
                    provider_config_ref='cluster-a',
                    resource_limits_json={'manager_count': 1, 'slots': 1},
                    default_timeouts_json={},
                    config_version=1,
                    status='enabled',
                    created_by='admin',
                    updated_by='admin',
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                EnvironmentGrantRow(
                    principal_id='principal-permanent-error',
                    environment_id='environment-permanent-error',
                    limits_json={'manager_count': 1, 'slots': 1},
                    enabled=True,
                    version=1,
                    created_by='admin',
                    updated_by='admin',
                    created_at=now,
                    updated_at=now,
                )
            )
            _add_resource_usage_rows(
                session,
                environment_id='environment-permanent-error',
                principal_id='principal-permanent-error',
                now=now,
            )
            session.add(
                ExecutionRow(
                    execution_id='execution-permanent-error',
                    owner_principal_id='principal-permanent-error',
                    created_by='principal-permanent-error',
                    environment_id='environment-permanent-error',
                    pool_id='execution',
                    allocation_id='allocation-permanent-error',
                    workload='execution',
                    allocation_spec_hash='f' * 64,
                    version=1,
                    shard_id=0,
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec_json={},
                    driver_config_json={},
                    placement_json={},
                    status=ExecutionStatus.QUEUED,
                    route_status='none',
                    slot_cost=1,
                    next_attempt_seq=1,
                    attempt_count=0,
                    max_attempts=1,
                    next_event_seq=1,
                    priority=0,
                    next_schedule_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            await AllocationRepository(session).create_allocation(
                CreateEnvironmentAllocation(
                    allocation_id='allocation-permanent-error',
                    top_level_execution_id='execution-permanent-error',
                    owner_principal_id='principal-permanent-error',
                    environment_id='environment-permanent-error',
                    pool_id='execution',
                    workload='execution',
                    desired_manager_count=1,
                    resources=AllocationResources(
                        manager_count=1,
                        cpu_millicores=1000,
                        memory_bytes=1024,
                        slots=1,
                    ),
                    allocation_spec={'version': 1, 'code': {'mode': 'config_only'}},
                    allocation_spec_hash='f' * 64,
                ),
                now=now,
            )
            await session.commit()

        assert await controller.reconcile_once() == 1
        async with session_factory() as session:
            allocation = await session.get(EnvironmentAllocationRow, 'allocation-permanent-error')
            execution = await session.get(ExecutionRow, 'execution-permanent-error')
            environment_usage = await session.get(
                EnvironmentResourceUsageRow,
                'environment:environment-permanent-error',
            )
            assert allocation.status == EnvironmentAllocationStatus.RELEASED
            assert allocation.last_error_code == 'WorkloadPermanentError'
            assert allocation.resources_released_at is not None
            assert environment_usage.manager_count == 0
            assert execution.status == ExecutionStatus.FAILED
            assert execution.error_json == {
                'type': 'EnvironmentProvisioningFailed',
                'message': 'kubernetes_api_403: forbidden',
            }
        assert allocator.create_calls == 1
    finally:
        await controller.close()
        await engine.dispose()


async def test_environment_controller_keeps_permanent_cleanup_error_in_deleting() -> None:
    engine, session_factory = await _session_factory()
    now = utcnow()
    allocator = _PermanentDeleteFailureAllocator()
    controller = EnvironmentController(
        session_factory=session_factory,
        config=DistributedConfig(),
        allocators={'kubernetes': allocator},
        client_resolver=MagicMock(),
        owner_id='cleanup-error-test',
    )
    try:
        async with session_factory() as session:
            await _seed_dynamic_controller_allocations(
                session,
                allocation_ids=['allocation-cleanup-error'],
                now=now,
            )
            allocation = await session.get(EnvironmentAllocationRow, 'allocation-cleanup-error')
            execution = await session.get(ExecutionRow, 'execution-slow-0')
            allocation.status = EnvironmentAllocationStatus.DELETING
            allocation.provider_resource_ref_json = {
                'kind': 'StatefulSet',
                'name': 'allocation-cleanup-error',
            }
            execution.status = ExecutionStatus.SUCCEEDED
            execution.finished_at = now
            await session.commit()

        assert await controller.reconcile_once() == 1
        async with session_factory() as session:
            allocation = await session.get(EnvironmentAllocationRow, 'allocation-cleanup-error')
            execution = await session.get(ExecutionRow, 'execution-slow-0')
        assert allocation.status == EnvironmentAllocationStatus.DELETING
        assert allocation.last_error_code == 'WorkloadPermanentError'
        assert allocation.last_error_message == 'kubernetes_api_403: delete forbidden'
        assert allocation.operation_id is None
        assert execution.status == ExecutionStatus.SUCCEEDED
        assert allocator.delete_calls == 1
    finally:
        await controller.close()
        await engine.dispose()


async def test_environment_controller_releases_draining_allocation_without_provider_resource_or_config() -> None:
    engine, session_factory = await _session_factory()
    now = utcnow()
    controller = EnvironmentController(
        session_factory=session_factory,
        config=DistributedConfig(),
        client_resolver=MagicMock(),
        owner_id='missing-provider-config-test',
    )
    try:
        async with session_factory() as session:
            await _seed_dynamic_controller_allocations(
                session,
                allocation_ids=['allocation-no-resource'],
                now=now,
            )
            await QuotaRepository(session, default_limits=controller.config.default_user_quota).reserve_allocation(
                'allocation-no-resource',
                now=now,
            )
            allocation = await session.get(EnvironmentAllocationRow, 'allocation-no-resource')
            execution = await session.get(ExecutionRow, 'execution-slow-0')
            allocation.status = EnvironmentAllocationStatus.DRAINING
            execution.status = ExecutionStatus.FAILED
            execution.finished_at = now
            await session.commit()

        assert await controller.reconcile_once() == 1
        async with session_factory() as session:
            allocation = await session.get(EnvironmentAllocationRow, 'allocation-no-resource')
            usage = await session.get(EnvironmentResourceUsageRow, 'environment:environment-slow')
        assert allocation.status == EnvironmentAllocationStatus.RELEASED
        assert allocation.resources_released_at is not None
        assert usage.manager_count == 0
    finally:
        await controller.close()
        await engine.dispose()


async def test_environment_controller_waits_for_batch_finalization_before_draining_allocations() -> None:
    engine, session_factory = await _session_factory()
    now = utcnow()
    allocator = _FakeEnvironmentAllocator()
    allocator.exists = True
    controller = EnvironmentController(
        session_factory=session_factory,
        config=DistributedConfig(environment_operation_ttl=timedelta(seconds=30)),
        allocators={'kubernetes': allocator},
        client_resolver=MagicMock(),
        owner_id='reconciler-test',
    )
    try:
        async with session_factory() as session:
            session.add(
                EnvironmentRow(
                    environment_id='batch-environment',
                    name='batch-environment',
                    allocation_type='dynamic',
                    provider_type='kubernetes',
                    provider_config_ref='cluster-a',
                    resource_limits_json={},
                    default_timeouts_json={},
                    config_version=1,
                    status='enabled',
                    created_by='admin',
                    updated_by='admin',
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                ExecutionRow(
                    execution_id='batch-failed-parent',
                    owner_principal_id='principal-a',
                    created_by='principal-a',
                    environment_id='batch-environment',
                    pool_id='orchestration',
                    allocation_id='batch-parent-allocation',
                    workload='orchestration',
                    allocation_spec_hash='b' * 64,
                    version=1,
                    shard_id=0,
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_batch',
                    instance_mode='asyncio',
                    spec_json={'batch': {}},
                    driver_config_json={},
                    placement_json={},
                    status=ExecutionStatus.FAILED,
                    route_status='none',
                    slot_cost=1,
                    next_attempt_seq=1,
                    attempt_count=1,
                    max_attempts=2,
                    next_event_seq=1,
                    priority=0,
                    next_schedule_at=now,
                    finished_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                BatchRunRow(
                    parent_execution_id='batch-failed-parent',
                    shard_id=0,
                    generation_mode='remote_materialize',
                    generation_status='pending',
                    generation_epoch=1,
                    materialize_cursor_bytes=0,
                    materialized_count=0,
                    owner_principal_id='principal-a',
                    total_items=0,
                    submitted_items=0,
                    active_items=0,
                    terminal_items=0,
                    submission_closed=False,
                    control_state='active',
                    summary_version=0,
                    last_progress_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                EnvironmentAllocationRow(
                    allocation_id='batch-parent-allocation',
                    top_level_execution_id='batch-failed-parent',
                    owner_principal_id='principal-a',
                    environment_id='batch-environment',
                    pool_id='orchestration',
                    workload='orchestration',
                    status=EnvironmentAllocationStatus.READY,
                    desired_manager_count=0,
                    ready_manager_count=0,
                    manager_count=0,
                    cpu_millicores=0,
                    memory_bytes=0,
                    slots=1,
                    active_batches=1,
                    provider_resource_ref_json={'workload': 'batch-parent-allocation'},
                    allocation_spec_json={'version': 1, 'code': {'mode': 'config_only'}},
                    allocation_spec_hash='b' * 64,
                    config_version=1,
                    status_version=1,
                    last_progress_at=now,
                    created_at=now,
                    ready_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

        await controller.reconcile_once()
        async with session_factory() as session:
            allocation = await session.get(EnvironmentAllocationRow, 'batch-parent-allocation')
            assert allocation.status == EnvironmentAllocationStatus.READY
            batch = await session.get(BatchRunRow, 'batch-failed-parent')
            batch.control_state = 'finalized'
            await session.commit()

        await controller.reconcile_once()
        async with session_factory() as session:
            allocation = await session.get(EnvironmentAllocationRow, 'batch-parent-allocation')
            assert allocation.status == EnvironmentAllocationStatus.DRAINING
    finally:
        await controller.close()
        await engine.dispose()


async def test_reconciler_runs_batch_lifecycle_before_allocation_controller() -> None:
    engine, session_factory = await _session_factory()
    calls = []

    class BatchHook:
        async def reconcile_once(self):
            calls.append('batch')
            return 2

        async def close(self):
            pass

    class AllocationController:
        async def reconcile_once(self):
            calls.append('allocation')
            return 3

        async def close(self):
            pass

    service = ReconcilerService(
        session_factory=session_factory,
        config=DistributedConfig(artifact_maintenance_enabled=False),
        batch_lifecycle=BatchHook(),
        environment_controller=AllocationController(),
        dependency_build_controller=_NoopController(),
        workload_client_resolver=_NoopWorkloadClientResolver(),  # type: ignore[arg-type]
    )
    try:
        assert await service.tick_once() == 5
    finally:
        await service.close()
        await engine.dispose()

    assert calls == ['batch', 'allocation']


async def test_reconciler_run_loop_keeps_batch_lifecycle_independent_of_blocked_system_maintenance() -> None:
    engine, session_factory = await _session_factory()
    environment_started = asyncio.Event()
    environment_finished = asyncio.Event()
    release_environment = asyncio.Event()
    batch_reconciled_twice = asyncio.Event()
    batch_calls = 0

    class BatchHook:
        async def reconcile_once(self) -> int:
            nonlocal batch_calls
            await environment_started.wait()
            batch_calls += 1
            if batch_calls >= 2:
                batch_reconciled_twice.set()
            return 0

        async def close(self) -> None:
            pass

    class BlockingAllocationController:
        async def reconcile_once(self) -> int:
            environment_started.set()
            try:
                await release_environment.wait()
            finally:
                environment_finished.set()
            return 0

        async def close(self) -> None:
            release_environment.set()

    service = ReconcilerService(
        session_factory=session_factory,
        config=DistributedConfig(
            reconciler_poll_interval=10,
            batch_lifecycle_poll_interval=0.01,
            batch_lifecycle_worker_count=1,
            environment_reconcile_worker_count=1,
            artifact_maintenance_enabled=False,
        ),
        batch_lifecycle=BatchHook(),
        environment_controller=BlockingAllocationController(),
        dependency_build_controller=_NoopController(),
        workload_client_resolver=_NoopWorkloadClientResolver(),  # type: ignore[arg-type]
    )
    run_task = asyncio.create_task(service.run_forever())
    try:
        # NOTE(jerryhrzhou): Test independent progress, not sub-second scheduling latency on a loaded CI worker.
        await asyncio.wait_for(environment_started.wait(), timeout=5)
        await asyncio.wait_for(batch_reconciled_twice.wait(), timeout=5)
        assert not environment_finished.is_set(), 'Batch must make progress while maintenance is still blocked'
    finally:
        service.stop()
        release_environment.set()
        try:
            await asyncio.wait_for(run_task, timeout=5)
        finally:
            await engine.dispose()


async def test_reconciler_runs_five_named_loops_without_cross_domain_blocking(monkeypatch) -> None:
    engine, session_factory = await _session_factory()
    release_environment = asyncio.Event()
    environment_started = asyncio.Event()
    progressed = asyncio.Event()
    calls = {'core': 0, 'dependency-build': 0, 'artifact-maintenance': 0, 'batch': 0}

    def record(name: str) -> None:
        calls[name] += 1
        if all(count >= 2 for count in calls.values()):
            progressed.set()

    class BatchLifecycle:
        async def reconcile_once(self):
            record('batch')
            return 0

        async def close(self):
            pass

    class EnvironmentReconciler:
        async def reconcile_once(self):
            environment_started.set()
            await release_environment.wait()
            return 0

        async def close(self):
            release_environment.set()

    class DependencyBuildReconciler:
        async def reconcile_once(self):
            record('dependency-build')
            return 0

        async def close(self):
            pass

    class ArtifactMaintenance:
        async def reconcile_once(self):
            record('artifact-maintenance')
            return 0

        async def close(self):
            pass

    service = ReconcilerService(
        session_factory=session_factory,
        config=DistributedConfig(
            reconciler_poll_interval=0.01,
            batch_lifecycle_poll_interval=0.01,
            dependency_build_poll_interval=timedelta(seconds=0.01),
            reconciler_core_timeout=1,
            environment_reconcile_timeout=1,
            dependency_build_reconcile_timeout=1,
            artifact_maintenance_reconcile_timeout=1,
            batch_lifecycle_reconcile_timeout=1,
        ),
        batch_lifecycle=BatchLifecycle(),
        environment_controller=EnvironmentReconciler(),
        dependency_build_controller=DependencyBuildReconciler(),
        workload_client_resolver=_NoopWorkloadClientResolver(),  # type: ignore[arg-type]
        artifact_maintenance=ArtifactMaintenance(),
    )

    async def core_tick_once():
        record('core')
        return 0

    monkeypatch.setattr(service, '_core_tick_once', core_tick_once)
    run_task = asyncio.create_task(service.run_forever())
    try:
        await asyncio.wait_for(environment_started.wait(), timeout=1)
        expected_names = {
            'dist-reconciler:core',
            'dist-reconciler:environment',
            'dist-reconciler:dependency-build',
            'dist-reconciler:artifact-maintenance',
            'dist-reconciler:batch',
        }
        assert expected_names <= {task.get_name() for task in asyncio.all_tasks()}
        # Allow scheduler tasks enough wall-clock headroom when the full suite loads the event loop.
        await asyncio.wait_for(progressed.wait(), timeout=5)
    finally:
        service.stop()
        release_environment.set()
        await asyncio.wait_for(run_task, timeout=1)
        await engine.dispose()


async def test_reconciler_retries_transient_loop_failure_without_restarting_other_loops(monkeypatch) -> None:
    from nexus.distributed.artifacts.store import ArtifactStoreTransientError

    engine, session_factory = await _session_factory()
    maintenance_recovered = asyncio.Event()
    batch_progressed = asyncio.Event()
    maintenance_calls = 0
    batch_calls = 0

    class EnvironmentReconciler:
        async def reconcile_once(self):
            return 0

        async def close(self):
            pass

    class DependencyBuildReconciler:
        async def reconcile_once(self):
            return 0

        async def close(self):
            pass

    class ArtifactMaintenance:
        async def reconcile_once(self):
            nonlocal maintenance_calls
            maintenance_calls += 1
            if maintenance_calls == 1:
                raise ArtifactStoreTransientError(detail='temporary')
            maintenance_recovered.set()
            return 0

        async def close(self):
            pass

    class BatchLifecycle:
        async def reconcile_once(self):
            nonlocal batch_calls
            batch_calls += 1
            if batch_calls >= 2:
                batch_progressed.set()
            return 0

        async def close(self):
            pass

    service = ReconcilerService(
        session_factory=session_factory,
        config=DistributedConfig(
            reconciler_poll_interval=0.01,
            batch_lifecycle_poll_interval=0.01,
            dependency_build_poll_interval=timedelta(seconds=0.01),
            reconciler_failure_backoff=0.01,
        ),
        batch_lifecycle=BatchLifecycle(),
        environment_controller=EnvironmentReconciler(),
        dependency_build_controller=DependencyBuildReconciler(),
        workload_client_resolver=_NoopWorkloadClientResolver(),  # type: ignore[arg-type]
        artifact_maintenance=ArtifactMaintenance(),
    )

    async def core_tick_once():
        return 0

    monkeypatch.setattr(service, '_core_tick_once', core_tick_once)
    run_task = asyncio.create_task(service.run_forever())
    try:
        await asyncio.wait_for(maintenance_recovered.wait(), timeout=1)
        await asyncio.wait_for(batch_progressed.wait(), timeout=1)
    finally:
        service.stop()
        await asyncio.wait_for(run_task, timeout=1)
        await engine.dispose()


async def test_reconciler_propagates_programming_errors_and_stops_all_loops() -> None:
    engine, session_factory = await _session_factory()

    class EnvironmentReconciler:
        async def reconcile_once(self):
            return 0

        async def close(self):
            pass

    class DependencyBuildReconciler:
        async def reconcile_once(self):
            return 0

        async def close(self):
            pass

    class BrokenBatchLifecycle:
        async def reconcile_once(self):
            raise AssertionError('broken batch invariant')

        async def close(self):
            pass

    service = ReconcilerService(
        session_factory=session_factory,
        config=DistributedConfig(artifact_maintenance_enabled=False),
        batch_lifecycle=BrokenBatchLifecycle(),
        environment_controller=EnvironmentReconciler(),
        dependency_build_controller=DependencyBuildReconciler(),
        workload_client_resolver=_NoopWorkloadClientResolver(),  # type: ignore[arg-type]
    )
    try:
        with pytest.raises(AssertionError, match='broken batch invariant'):
            await service.run_forever()
    finally:
        await engine.dispose()


async def test_repository_batch_lifecycle_closes_owned_artifact_transport() -> None:
    engine, session_factory = await _session_factory()
    try:
        hook = BatchLifecycleReconciler(
            session_factory=session_factory,
            config=DistributedConfig(secrets={'internal_token': 'internal-token'}),
        )
        client = hook._result_artifact_client()
        transport = client.transport
        await transport._http_runtime.run(lambda _session: asyncio.sleep(0))

        await hook.close()
        await hook.close()

        assert transport.session is not None
        assert transport.session.closed
    finally:
        await engine.dispose()


async def test_repository_batch_lifecycle_renews_and_releases_parent_claim(monkeypatch) -> None:
    engine, session_factory = await _session_factory()
    now = utcnow()
    try:
        async with session_factory() as session:
            session.add(
                BatchRunRow(
                    parent_execution_id='batch-parent-slow-page',
                    shard_id=0,
                    generation_mode='local_task_list',
                    generation_status='complete',
                    generation_epoch=1,
                    materialize_cursor_bytes=0,
                    materialized_count=0,
                    owner_principal_id='principal-a',
                    total_items=0,
                    submitted_items=0,
                    active_items=0,
                    terminal_items=0,
                    submission_closed=False,
                    control_state='active',
                    summary_version=0,
                    last_progress_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

        renew_calls = 0
        original_renew = BatchRepository.renew_lifecycle_claim

        async def recording_renew(self, claim, *, lease_ttl, now):
            nonlocal renew_calls
            renew_calls += 1
            return await original_renew(self, claim, lease_ttl=lease_ttl, now=now)

        async def slow_reconcile(claim, *, lease, page_budget, scan_budget):
            del claim, page_budget, scan_budget
            await asyncio.sleep(0.04)
            await lease.ensure_owned()
            return SimpleNamespace(changed=0, scanned=0)

        monkeypatch.setattr(BatchRepository, 'renew_lifecycle_claim', recording_renew)
        hook = BatchLifecycleReconciler(
            session_factory=session_factory,
            config=DistributedConfig(
                batch_lifecycle_claim_ttl=timedelta(seconds=1),
                batch_lifecycle_claim_renew_interval=0.01,
            ),
            owner_id='reconciler-slow-page',
        )
        monkeypatch.setattr(hook, '_reconcile_parent_with_retry', slow_reconcile)

        assert await hook.reconcile_once() == 0
        assert renew_calls >= 2
        async with session_factory() as session:
            batch = await session.get(BatchRunRow, 'batch-parent-slow-page')
        assert batch is not None
        assert batch.lifecycle_owner_id is None
        assert batch.lifecycle_operation_id is None
        assert batch.lifecycle_lease_expires_at is None
    finally:
        await engine.dispose()


async def test_repository_batch_lifecycle_applies_parent_page_and_scan_budgets(monkeypatch) -> None:
    engine, session_factory = await _session_factory()
    now = utcnow()
    try:
        async with session_factory() as session:
            for index in range(3):
                session.add(
                    BatchRunRow(
                        parent_execution_id=f'batch-budget-{index}',
                        shard_id=0,
                        generation_mode='local_task_list',
                        generation_status='complete',
                        generation_epoch=1,
                        materialize_cursor_bytes=0,
                        materialized_count=0,
                        owner_principal_id='principal-a',
                        total_items=100,
                        submitted_items=100,
                        active_items=100,
                        terminal_items=0,
                        submission_closed=True,
                        control_state='active',
                        summary_version=0,
                        last_progress_at=now,
                        created_at=now,
                        updated_at=now,
                    )
                )
            await session.commit()

        observed = []

        async def bounded_reconcile(claim, *, lease, page_budget, scan_budget):
            del lease
            observed.append((claim.parent_execution_id, page_budget, scan_budget))
            return SimpleNamespace(changed=0, scanned=scan_budget)

        hook = BatchLifecycleReconciler(
            session_factory=session_factory,
            config=DistributedConfig(
                batch_lifecycle_parent_limit=2,
                batch_lifecycle_max_pages_per_parent=1,
                batch_lifecycle_scan_limit=5,
            ),
            owner_id='reconciler-budget',
        )
        monkeypatch.setattr(hook, '_reconcile_parent_with_retry', bounded_reconcile)

        assert await hook.reconcile_once() == 0
        assert observed == [('batch-budget-0', 1, 5)]
        async with session_factory() as session:
            batches = list(await session.scalars(select(BatchRunRow).order_by(BatchRunRow.parent_execution_id)))
        assert all(batch.lifecycle_owner_id is None for batch in batches)
    finally:
        await engine.dispose()


async def test_repository_batch_lifecycle_finishes_ttl_stopped_empty_parent() -> None:
    engine, session_factory = await _session_factory()
    now = utcnow()
    try:
        async with session_factory() as session:
            parent = await ExecutionRepository(session, shard_count=8).create_execution(
                CreateExecution(
                    execution_id='batch-parent-ttl',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_batch',
                    instance_mode='asyncio',
                    spec={'batch': {'retry_failed': 0}},
                ),
                now=now,
            )
            parent_row = await session.get(ExecutionRow, parent.execution_id)
            parent_row.status = ExecutionStatus.STOPPING
            parent_row.stop_reason = 'execution_ttl'
            session.add(
                BatchRunRow(
                    parent_execution_id=parent.execution_id,
                    shard_id=parent.shard_id,
                    generation_mode='local_task_list',
                    generation_status='complete',
                    generation_epoch=1,
                    materialize_cursor_bytes=0,
                    materialized_count=0,
                    owner_principal_id='principal-a',
                    total_items=0,
                    submitted_items=0,
                    active_items=0,
                    terminal_items=0,
                    submission_closed=False,
                    control_state='active',
                    summary_version=0,
                    last_progress_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

        metrics = MagicMock()
        hook = BatchLifecycleReconciler(
            session_factory=session_factory,
            config=DistributedConfig(reconciler_batch_size=50),
            metrics=metrics,
        )
        assert await hook.reconcile_once() > 0
        metrics.record_batch_full_scan.assert_called_once()

        async with session_factory() as session:
            parent = await session.get(ExecutionRow, 'batch-parent-ttl')
            batch = await session.get(BatchRunRow, 'batch-parent-ttl')
        assert parent.status == ExecutionStatus.EXPIRED
        assert batch.control_state == 'finalized'
        assert batch.stop_reason == 'parent_ttl'
    finally:
        await engine.dispose()


async def test_scheduler_service_uses_configured_scan_and_assignment_limits(monkeypatch):
    engine, session_factory = await _session_factory()
    observed = []

    async def fake_claim(self, *, shard_ids, now):
        del shard_ids, now
        observed.append((self.scan_limit, self.max_assignments, self.manager_lock_contention_backoff))
        return []

    monkeypatch.setattr(SchedulerRepository, 'claim_and_assign_batch', fake_claim)
    try:
        service = SchedulerService(
            session_factory=session_factory,
            config=DistributedConfig(
                shard_count=4,
                scheduler_scan_limit=250,
                scheduler_max_assignments=125,
                scheduler_manager_lock_contention_backoff=timedelta(milliseconds=400),
            ),
            owner_id='scheduler-configured-limits',
        )

        assert await service.tick_once() == 0
    finally:
        await engine.dispose()

    assert observed == [(250, 125, timedelta(milliseconds=400))]


async def test_scheduler_service_commits_lease_refresh_before_assignment_batch(monkeypatch):
    events = []

    class FakeSession:
        def __init__(self, ordinal):
            self.ordinal = ordinal

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def commit(self):
            events.append(('commit', self.ordinal))

    class FakeSessionFactory:
        def __init__(self):
            self.created = 0

        def __call__(self):
            session = FakeSession(self.created)
            self.created += 1
            return session

    async def fake_renew(self, *, owner_id, shard_group, shard_ids, lease_ttl, now):
        del owner_id, shard_group, shard_ids, lease_ttl, now
        events.append(('renew', self.session.ordinal))
        return [0]

    async def fake_claim(self, *, shard_ids, now):
        del shard_ids, now
        events.append(('claim', self.session.ordinal))
        return []

    async def fake_endpoints(self, manager_ids):
        del manager_ids
        return {}

    async def fake_validate(self, *, shard_group):
        del self, shard_group

    monkeypatch.setattr(SchedulerRepository, 'renew_shard_leases', fake_renew)
    monkeypatch.setattr(ShardLeaseRepository, 'validate_monotonic_shard_count', fake_validate)
    monkeypatch.setattr(ShardLeaseRepository, 'lock_discovery', fake_validate)
    monkeypatch.setattr(SchedulerRepository, 'claim_and_assign_batch', fake_claim)
    monkeypatch.setattr(SchedulerRepository, 'manager_endpoints_by_id', fake_endpoints)

    service = SchedulerService(
        session_factory=FakeSessionFactory(),  # type: ignore[arg-type]
        config=DistributedConfig(shard_count=4),
        owner_id='scheduler-separate-lease-transaction',
    )

    assert await service.tick_once() == 0
    assert events == [('renew', 0), ('commit', 0), ('claim', 1), ('commit', 1)]


async def test_scheduler_service_idle_tick_does_not_lock_shard_lease_fence(monkeypatch):
    engine, session_factory = await _session_factory()
    fence_calls = 0

    async def unexpected_fence(self, **kwargs):
        nonlocal fence_calls
        del self, kwargs
        fence_calls += 1
        raise AssertionError('idle Scheduler tick must not fence a shard lease')

    monkeypatch.setattr(ShardLeaseRepository, 'fence', unexpected_fence)
    service = SchedulerService(
        session_factory=session_factory,
        config=DistributedConfig(shard_count=4, scheduler_shard_ownership_limit=1),
        owner_id='scheduler-idle',
    )
    try:
        assert await service.tick_once() == 0
        assert fence_calls == 0
        assert len(service._shard_leases._owned_shards) == 1
    finally:
        await service.close()
        await engine.dispose()


@pytest.mark.parametrize('failure_kind', ['deadlock', 'connection_lost', 'unknown_outcome'])
async def test_scheduler_service_does_not_publish_lease_state_when_refresh_commit_fails(monkeypatch, failure_kind):
    original_cursor = 2

    if failure_kind == 'unknown_outcome':
        commit_error = RuntimeError('unknown lease commit outcome')
    else:
        errno = 1213 if failure_kind == 'deadlock' else 2013
        commit_error = OperationalError(
            'COMMIT',
            {},
            RuntimeError(errno, f'lease commit {failure_kind}'),
        )

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def commit(self):
            raise commit_error

    async def fake_renew(self, *, owner_id, shard_group, shard_ids, lease_ttl, now):
        del owner_id, shard_group, shard_ids, lease_ttl, now
        return [0, 3]

    async def fake_validate(self, *, shard_group):
        del self, shard_group

    monkeypatch.setattr(SchedulerRepository, 'renew_shard_leases', fake_renew)
    monkeypatch.setattr(ShardLeaseRepository, 'validate_monotonic_shard_count', fake_validate)
    monkeypatch.setattr(ShardLeaseRepository, 'lock_discovery', fake_validate)
    service = SchedulerService(
        session_factory=lambda: FakeSession(),  # type: ignore[arg-type]
        config=DistributedConfig(shard_count=4),
        owner_id='scheduler-lease-commit-failure',
    )
    service._shard_leases._owned_shards = {3}
    service._shard_leases._last_renew_at = utcnow()
    service._shard_leases._last_discover_at = utcnow()
    service._shard_leases._next_discover_at = None
    service._shard_leases._discover_cursor = original_cursor

    with pytest.raises((OperationalError, RuntimeError)):
        await service.tick_once()

    assert service._shard_leases._owned_shards == set()
    assert service._shard_leases._last_renew_at is None
    assert service._shard_leases._last_discover_at is None
    assert service._shard_leases._next_discover_at is None
    assert service._shard_leases._discover_cursor == original_cursor


async def test_scheduler_service_rolls_back_assignment_when_lease_expires_before_commit(monkeypatch):
    engine, session_factory = await _session_factory()
    try:
        async with session_factory() as session:
            await ManagerRepository(session).register_session(
                manager_id='lease-expiry-manager',
                session_id='session-1',
                endpoint_url='http://lease-expiry-manager.internal',
                supported_modes=[ExecutionMode.JOB],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                capacity_total=1,
                force_replace=False,
                now=utcnow(),
            )
            execution = await ExecutionRepository(session, shard_count=8).create_execution(
                CreateExecution(
                    execution_id='lease-expiry-execution',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                ),
                now=utcnow(),
            )
            started_at = utcnow()
            session.add(
                ShardLeaseRow(
                    shard_group='default',
                    shard_id=execution.shard_id,
                    shard_count=8,
                    owner_id='scheduler-expired-owner',
                    lease_expires_at=started_at + timedelta(seconds=1),
                    updated_at=started_at,
                )
            )
            await session.commit()

        observed_times = iter((started_at, started_at + timedelta(seconds=2)))
        monkeypatch.setattr('nexus.distributed.control_plane.scheduler.utcnow', lambda: next(observed_times))
        service = SchedulerService(
            session_factory=session_factory,
            config=DistributedConfig(
                shard_count=8,
                scheduler_lease_ttl=timedelta(seconds=120),
                scheduler_lease_renew_interval=60,
                scheduler_shard_discover_interval=60,
            ),
            owner_id='scheduler-expired-owner',
        )
        service._shard_leases._owned_shards = {execution.shard_id}
        service._shard_leases._last_renew_at = started_at
        service._shard_leases._last_discover_at = started_at
        service._shard_leases._next_discover_at = started_at + timedelta(minutes=1)

        assert await service.tick_once() == 0
        assert service._shard_leases._owned_shards == set()

        async with session_factory() as session:
            current = await session.get(ExecutionRow, execution.execution_id)
        assert current is not None
        assert current.status == ExecutionStatus.QUEUED
    finally:
        await engine.dispose()


async def test_scheduler_service_rolls_back_no_capacity_backoff_when_lease_expires_before_commit(monkeypatch):
    engine, session_factory = await _session_factory()
    try:
        async with session_factory() as session:
            execution = await ExecutionRepository(session, shard_count=8).create_execution(
                CreateExecution(
                    execution_id='lease-expiry-no-capacity',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                ),
                now=utcnow(),
            )
            started_at = utcnow()
            session.add(
                ShardLeaseRow(
                    shard_group='default',
                    shard_id=execution.shard_id,
                    shard_count=8,
                    owner_id='scheduler-expired-no-capacity',
                    lease_expires_at=started_at + timedelta(seconds=1),
                    updated_at=started_at,
                )
            )
            await session.commit()

        observed_times = iter((started_at, started_at + timedelta(seconds=2)))
        monkeypatch.setattr('nexus.distributed.control_plane.scheduler.utcnow', lambda: next(observed_times))
        service = SchedulerService(
            session_factory=session_factory,
            config=DistributedConfig(
                shard_count=8,
                scheduler_lease_ttl=timedelta(seconds=120),
                scheduler_lease_renew_interval=60,
                scheduler_shard_discover_interval=60,
            ),
            owner_id='scheduler-expired-no-capacity',
        )
        service._shard_leases._owned_shards = {execution.shard_id}
        service._shard_leases._last_renew_at = started_at
        service._shard_leases._last_discover_at = started_at
        service._shard_leases._next_discover_at = started_at + timedelta(minutes=1)

        assert await service.tick_once() == 0
        assert service._shard_leases._owned_shards == set()

        async with session_factory() as session:
            current = await session.get(ExecutionRow, execution.execution_id)
        assert current is not None
        assert current.next_schedule_at == execution.next_schedule_at
        assert current.last_schedule_error is None
    finally:
        await engine.dispose()


async def test_two_scheduler_services_only_current_lease_owner_assigns():
    engine, session_factory = await _session_factory()
    first = None
    second = None
    try:
        config = DistributedConfig(
            shard_count=8,
            scheduler_lease_ttl=timedelta(seconds=120),
            scheduler_lease_renew_interval=60,
            scheduler_shard_discover_interval=60,
        )
        first = SchedulerService(
            session_factory=session_factory,
            config=config,
            owner_id='scheduler-stale',
        )
        second = SchedulerService(
            session_factory=session_factory,
            config=config,
            owner_id='scheduler-current',
        )
        async with session_factory() as session:
            now = utcnow()
            await ManagerRepository(session).register_session(
                manager_id='two-scheduler-manager',
                session_id='session-1',
                endpoint_url='http://127.0.0.1:9',
                supported_modes=[ExecutionMode.JOB],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                capacity_total=1,
                force_replace=False,
                now=now,
            )
            execution = await ExecutionRepository(session, shard_count=8).create_execution(
                CreateExecution(
                    execution_id='two-scheduler-execution',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                ),
                now=now,
            )
            session.add(
                ShardLeaseRow(
                    shard_group=second.shard_group,
                    shard_id=execution.shard_id,
                    shard_count=8,
                    owner_id=second.lease_owner_id,
                    lease_expires_at=now + timedelta(minutes=1),
                    updated_at=now,
                )
            )
            await session.commit()

        cached_at = utcnow()
        for service in (first, second):
            service._shard_leases._owned_shards = {execution.shard_id}
            service._shard_leases._last_renew_at = cached_at
            service._shard_leases._last_discover_at = cached_at
            service._shard_leases._next_discover_at = cached_at + timedelta(minutes=1)

        results = await asyncio.gather(first.tick_once(), second.tick_once())

        assert sorted(results) == [0, 1]
        assert first._shard_leases._owned_shards == set()
        async with session_factory() as session:
            current = await session.get(ExecutionRow, execution.execution_id)
        assert current is not None
        assert current.status == ExecutionStatus.ASSIGNED
        assert current.attempt_count == 1
    finally:
        if first is not None:
            await first.close()
        if second is not None:
            await second.close()
        await engine.dispose()


async def test_scheduler_service_renews_leases_while_assignment_batch_is_blocked(monkeypatch):
    engine, session_factory = await _session_factory()
    assignment_started = asyncio.Event()
    release_assignment = asyncio.Event()
    renewed_while_blocked = asyncio.Event()
    renew_calls = 0

    async def fake_renew(self, *, owner_id, shard_group, shard_ids, lease_ttl, now):
        nonlocal renew_calls
        del owner_id, shard_group, shard_ids, lease_ttl, now
        renew_calls += 1
        if assignment_started.is_set() and renew_calls >= 2:
            renewed_while_blocked.set()
        return [0]

    async def blocked_claim(self, *, shard_ids, now):
        del shard_ids, now
        assignment_started.set()
        await release_assignment.wait()
        return []

    async def fake_endpoints(self, manager_ids):
        del manager_ids
        return {}

    monkeypatch.setattr(SchedulerRepository, 'renew_shard_leases', fake_renew)
    monkeypatch.setattr(SchedulerRepository, 'claim_and_assign_batch', blocked_claim)
    monkeypatch.setattr(SchedulerRepository, 'manager_endpoints_by_id', fake_endpoints)
    service = SchedulerService(
        session_factory=session_factory,
        config=DistributedConfig(
            shard_count=4,
            scheduler_poll_interval=0.01,
            scheduler_lease_renew_interval=0.01,
        ),
        owner_id='scheduler-independent-heartbeat',
    )
    run_task = asyncio.create_task(service.run_forever())
    try:
        await asyncio.wait_for(assignment_started.wait(), timeout=1)
        await asyncio.wait_for(renewed_while_blocked.wait(), timeout=0.2)
    finally:
        service.stop()
        release_assignment.set()
        await asyncio.wait_for(run_task, timeout=1)
        await engine.dispose()

    assert renew_calls >= 2


async def test_scheduler_service_wakes_manager_with_internal_token_after_assignment():
    engine, session_factory = await _session_factory()
    wakeups = []

    async def wakeup(request):
        wakeups.append(request.headers.get('Authorization'))
        return web.json_response({'status': 'ok'})

    manager_app = web.Application()
    manager_app.router.add_post('/v1/internal/wakeup', wakeup)
    manager_client = TestClient(TestServer(manager_app))
    await manager_client.start_server()
    try:
        async with session_factory() as session:
            manager = ManagerRepository(session)
            await manager.register_session(
                manager_id='im-1',
                session_id='session-1',
                endpoint_url=str(manager_client.make_url('/')).rstrip('/'),
                supported_modes=[ExecutionMode.JOB],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                capacity_total=1,
                force_replace=False,
                now=utcnow(),
            )
            repo = ExecutionRepository(session, shard_count=8)
            execution = await repo.create_execution(
                CreateExecution(
                    execution_id='wake-job',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                ),
                now=utcnow(),
            )
            await session.commit()

        service = SchedulerService(
            session_factory=session_factory,
            config=DistributedConfig(shard_count=8, secrets={'internal_token': 'secret'}),
            owner_id='scheduler-1',
        )
        try:
            assert await service.tick_once() == 1
            await service.drain_wakeups(timeout=2)
            assert wakeups == ['Bearer secret']
        finally:
            await service.close()

        async with session_factory() as session:
            current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
        assert current.status == ExecutionStatus.ASSIGNED
    finally:
        await manager_client.close()
        await engine.dispose()


async def test_scheduler_service_wakes_manager_once_for_batch_assignments():
    engine, session_factory = await _session_factory()
    wakeups = []

    async def wakeup(request):
        wakeups.append(request.headers.get('Authorization'))
        return web.json_response({'status': 'ok'})

    manager_app = web.Application()
    manager_app.router.add_post('/v1/internal/wakeup', wakeup)
    manager_client = TestClient(TestServer(manager_app))
    await manager_client.start_server()
    try:
        async with session_factory() as session:
            manager = ManagerRepository(session)
            await manager.register_session(
                manager_id='im-1',
                session_id='session-1',
                endpoint_url=str(manager_client.make_url('/')).rstrip('/'),
                supported_modes=[ExecutionMode.JOB],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                capacity_total=3,
                force_replace=False,
                now=utcnow(),
            )
            repo = ExecutionRepository(session, shard_count=8)
            for idx in range(3):
                await repo.create_execution(
                    CreateExecution(
                        execution_id=f'wakeup-batch-job-{idx}',
                        mode=ExecutionMode.JOB,
                        spec_type='nexus_single',
                        instance_mode='asyncio',
                        spec={},
                    ),
                    now=utcnow(),
                )
            await session.commit()

        service = SchedulerService(
            session_factory=session_factory,
            config=DistributedConfig(shard_count=8, secrets={'internal_token': 'secret'}),
            owner_id='scheduler-1',
        )
        try:
            assert await service.tick_once() == 3
            await service.drain_wakeups(timeout=2)
            assert wakeups == ['Bearer secret']
        finally:
            await service.close()
    finally:
        await manager_client.close()
        await engine.dispose()


async def test_scheduler_service_keeps_assignment_when_manager_wakeup_fails():
    engine, session_factory = await _session_factory()
    try:
        async with session_factory() as session:
            manager = ManagerRepository(session)
            await manager.register_session(
                manager_id='im-1',
                session_id='session-1',
                endpoint_url='http://127.0.0.1:9',
                supported_modes=[ExecutionMode.JOB],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                capacity_total=1,
                force_replace=False,
                now=utcnow(),
            )
            repo = ExecutionRepository(session, shard_count=8)
            execution = await repo.create_execution(
                CreateExecution(
                    execution_id='wakeup-fail-job',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                ),
                now=utcnow(),
            )
            await session.commit()

        service = SchedulerService(
            session_factory=session_factory,
            config=DistributedConfig(shard_count=8),
            owner_id='scheduler-1',
        )
        try:
            assert await service.tick_once() == 1
            await service.drain_wakeups(timeout=2)
        finally:
            await service.close()
        async with session_factory() as session:
            current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
        assert current.status == ExecutionStatus.ASSIGNED
    finally:
        await engine.dispose()


async def test_scheduler_service_does_not_block_tick_on_slow_manager_wakeup():
    engine, session_factory = await _session_factory()
    wakeup_started = asyncio.Event()
    wakeup_release = asyncio.Event()

    async def wakeup(_request):
        wakeup_started.set()
        await wakeup_release.wait()
        return web.json_response({'status': 'ok'})

    manager_app = web.Application()
    manager_app.router.add_post('/v1/internal/wakeup', wakeup)
    manager_client = TestClient(TestServer(manager_app))
    await manager_client.start_server()
    try:
        async with session_factory() as session:
            manager = ManagerRepository(session)
            await manager.register_session(
                manager_id='im-slow-wakeup',
                session_id='session-1',
                endpoint_url=str(manager_client.make_url('/')).rstrip('/'),
                supported_modes=[ExecutionMode.JOB],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                capacity_total=1,
                force_replace=False,
                now=utcnow(),
            )
            repo = ExecutionRepository(session, shard_count=8)
            await repo.create_execution(
                CreateExecution(
                    execution_id='slow-wakeup-job',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                ),
                now=utcnow(),
            )
            await session.commit()

        service = SchedulerService(
            session_factory=session_factory,
            config=DistributedConfig(shard_count=8, secrets={'internal_token': 'secret'}),
            owner_id='scheduler-1',
        )
        try:
            started_at = time.monotonic()
            assert await service.tick_once() == 1
            elapsed = time.monotonic() - started_at

            assert elapsed < 0.1
            await asyncio.wait_for(wakeup_started.wait(), timeout=2)
            wakeup_release.set()
            await service.drain_wakeups(timeout=2)
        finally:
            await service.close()
    finally:
        await manager_client.close()
        await engine.dispose()


async def test_scheduler_service_reuses_cached_owned_shards_between_lease_refreshes(monkeypatch):
    engine, session_factory = await _session_factory()
    renew_calls = []
    claim_calls = []

    async def fake_renew(
        self,
        *,
        owner_id,
        shard_group,
        shard_ids,
        lease_ttl,
        now,
    ):
        renew_calls.append(list(shard_ids))
        return [0, 2]

    async def fake_claim(self, *, shard_ids, now):
        claim_calls.append(list(shard_ids))
        return []

    monkeypatch.setattr(SchedulerRepository, 'renew_shard_leases', fake_renew)
    monkeypatch.setattr(SchedulerRepository, 'claim_and_assign_batch', fake_claim)
    clock = iter(utcnow() + timedelta(seconds=index) for index in range(4))
    monkeypatch.setattr('nexus.distributed.control_plane.scheduler.utcnow', lambda: next(clock))
    try:
        service = SchedulerService(
            session_factory=session_factory,
            config=DistributedConfig(
                shard_count=4,
                scheduler_lease_ttl=timedelta(seconds=120),
                scheduler_lease_renew_interval=60.0,
                scheduler_shard_discover_interval=60.0,
            ),
            owner_id='scheduler-1',
        )

        assert await service.tick_once() == 0
        assert await service.tick_once() == 0
    finally:
        await engine.dispose()

    assert renew_calls == [[0, 1, 2, 3]]
    assert claim_calls == [[0, 2], [0, 2]]


async def test_scheduler_service_backs_off_discovery_when_no_shards_owned(monkeypatch):
    engine, session_factory = await _session_factory()
    renew_calls = []
    claim_calls = []

    async def fake_renew(
        self,
        *,
        owner_id,
        shard_group,
        shard_ids,
        lease_ttl,
        now,
    ):
        renew_calls.append(list(shard_ids))
        return []

    async def fake_claim(self, *, shard_ids, now):
        claim_calls.append(list(shard_ids))
        return []

    monkeypatch.setattr(SchedulerRepository, 'renew_shard_leases', fake_renew)
    monkeypatch.setattr(SchedulerRepository, 'claim_and_assign_batch', fake_claim)
    clock = iter(utcnow() + timedelta(seconds=index) for index in range(3))
    monkeypatch.setattr('nexus.distributed.control_plane.scheduler.utcnow', lambda: next(clock))
    try:
        service = SchedulerService(
            session_factory=session_factory,
            config=DistributedConfig(
                shard_count=4,
                scheduler_lease_ttl=timedelta(seconds=120),
                scheduler_lease_renew_interval=60.0,
                scheduler_shard_discover_interval=60.0,
            ),
            owner_id='scheduler-1',
        )

        assert await service.tick_once() == 0
        assert await service.tick_once() == 0
    finally:
        await engine.dispose()

    assert renew_calls == [[0, 1, 2, 3]]
    assert claim_calls == []


async def test_scheduler_service_discovers_owner_hash_windows(monkeypatch):
    engine, session_factory = await _session_factory()
    renew_calls = []

    async def fake_renew(
        self,
        *,
        owner_id,
        shard_group,
        shard_ids,
        lease_ttl,
        now,
    ):
        renew_calls.append(list(shard_ids))
        return []

    async def fake_claim(self, *, shard_ids, now):
        return []

    monkeypatch.setattr(SchedulerRepository, 'renew_shard_leases', fake_renew)
    monkeypatch.setattr(SchedulerRepository, 'claim_and_assign_batch', fake_claim)
    clock = iter(utcnow() + timedelta(seconds=index) for index in range(3))
    monkeypatch.setattr('nexus.distributed.control_plane.scheduler.utcnow', lambda: next(clock))
    try:
        config = DistributedConfig(
            shard_count=8,
            scheduler_shard_ownership_limit=3,
            scheduler_shard_discover_interval=0.001,
            scheduler_shard_discover_jitter=0.0,
        )
        first = SchedulerService(session_factory=session_factory, config=config, owner_id='scheduler-a')
        second = SchedulerService(session_factory=session_factory, config=config, owner_id='scheduler-b')

        assert await first.tick_once() == 0
        assert await first.tick_once() == 0
        assert await second.tick_once() == 0
    finally:
        await engine.dispose()

    assert len(renew_calls) == 9
    for offset in (0, 3, 6):
        assert {shard_id for call in renew_calls[offset : offset + 3] for shard_id in call} == set(range(8))
        assert sorted(len(call) for call in renew_calls[offset : offset + 3]) == [2, 3, 3]
    assert renew_calls[0] != [0, 1, 2]
    assert renew_calls[0] != renew_calls[3]
    assert renew_calls[0] != renew_calls[6]


async def test_scheduler_service_keeps_owned_shards_within_discover_limit(monkeypatch):
    engine, session_factory = await _session_factory()
    renew_calls = []
    claim_calls = []

    async def fake_renew(
        self,
        *,
        owner_id,
        shard_group,
        shard_ids,
        lease_ttl,
        now,
    ):
        renew_calls.append(list(shard_ids))
        return list(shard_ids)

    async def fake_claim(self, *, shard_ids, now):
        claim_calls.append(list(shard_ids))
        return []

    monkeypatch.setattr(SchedulerRepository, 'renew_shard_leases', fake_renew)
    monkeypatch.setattr(SchedulerRepository, 'claim_and_assign_batch', fake_claim)
    clock = iter(utcnow() + timedelta(seconds=index) for index in range(4))
    monkeypatch.setattr('nexus.distributed.control_plane.scheduler.utcnow', lambda: next(clock))
    try:
        service = SchedulerService(
            session_factory=session_factory,
            config=DistributedConfig(
                shard_count=8,
                scheduler_shard_ownership_limit=3,
                scheduler_shard_discover_interval=0.001,
                scheduler_shard_discover_jitter=0.0,
            ),
            owner_id='scheduler-sticky',
        )
        ordered_shards = service._shard_leases.owner_hash_ordered_shards()
        first_window = ordered_shards[:3]

        assert await service.tick_once() == 0
        assert await service.tick_once() == 0
    finally:
        await engine.dispose()

    assert set(renew_calls[0]) == set(first_window)
    assert set(renew_calls[1]) == set(first_window)
    assert set(claim_calls[0]) == set(first_window)
    assert set(claim_calls[1]) == set(first_window)


async def test_scheduler_service_discovers_only_until_owned_shard_limit(monkeypatch):
    engine, session_factory = await _session_factory()
    renew_calls = []
    claim_calls = []

    async def fake_renew(
        self,
        *,
        owner_id,
        shard_group,
        shard_ids,
        lease_ttl,
        now,
    ):
        renew_calls.append(list(shard_ids))
        if len(renew_calls) == 1:
            return list(shard_ids[:2])
        return list(shard_ids)

    async def fake_claim(self, *, shard_ids, now):
        claim_calls.append(list(shard_ids))
        return []

    monkeypatch.setattr(SchedulerRepository, 'renew_shard_leases', fake_renew)
    monkeypatch.setattr(SchedulerRepository, 'claim_and_assign_batch', fake_claim)
    clock = iter(utcnow() + timedelta(seconds=index) for index in range(4))
    monkeypatch.setattr('nexus.distributed.control_plane.scheduler.utcnow', lambda: next(clock))
    try:
        service = SchedulerService(
            session_factory=session_factory,
            config=DistributedConfig(
                shard_count=8,
                scheduler_shard_ownership_limit=3,
                scheduler_shard_discover_interval=0.001,
                scheduler_shard_discover_jitter=0.0,
            ),
            owner_id='scheduler-partial',
        )
        ordered_shards = service._shard_leases.owner_hash_ordered_shards()
        first_window = ordered_shards[:3]
        second_window = ordered_shards[3:6]

        assert await service.tick_once() == 0
        assert await service.tick_once() == 0
    finally:
        await engine.dispose()

    assert set(renew_calls[0]) == set(first_window)
    assert renew_calls[1] == [second_window[0]]
    assert set(renew_calls[2]) == set(first_window[:2]) | {second_window[0]}
    assert set(claim_calls[0]) == set(first_window[:2]) | {second_window[0]}
    assert set(claim_calls[1]) == set(first_window[:2]) | {second_window[0]}


async def test_scheduler_service_eventually_scans_every_shard_with_partial_ownership(monkeypatch):
    engine, session_factory = await _session_factory()
    renew_calls = []
    owned_shards: set[int] = set()

    async def fake_renew(
        self,
        *,
        owner_id,
        shard_group,
        shard_ids,
        lease_ttl,
        now,
    ):
        renew_calls.append(list(shard_ids))
        if not owned_shards:
            owned_shards.add(shard_ids[0])
        if stranded_shard in shard_ids:
            owned_shards.add(stranded_shard)
        return [shard_id for shard_id in shard_ids if shard_id in owned_shards]

    async def fake_claim(self, *, shard_ids, now):
        return []

    monkeypatch.setattr(SchedulerRepository, 'renew_shard_leases', fake_renew)
    monkeypatch.setattr(SchedulerRepository, 'claim_and_assign_batch', fake_claim)
    clock = iter(utcnow() + timedelta(seconds=index) for index in range(8))
    monkeypatch.setattr('nexus.distributed.control_plane.scheduler.utcnow', lambda: next(clock))
    try:
        service = SchedulerService(
            session_factory=session_factory,
            config=DistributedConfig(
                shard_count=8,
                scheduler_shard_ownership_limit=2,
                scheduler_shard_discover_interval=0.001,
                scheduler_shard_discover_jitter=0.0,
            ),
            owner_id='scheduler-partial-scan',
        )
        ordered_shards = service._shard_leases.owner_hash_ordered_shards()
        stranded_shard = ordered_shards[3]

        for _ in range(4):
            assert await service.tick_once() == 0
    finally:
        await engine.dispose()

    assert stranded_shard in owned_shards
    assert any(stranded_shard in call for call in renew_calls)


async def test_scheduler_service_limits_cached_owned_shards_between_refreshes(monkeypatch):
    engine, session_factory = await _session_factory()
    claim_calls = []

    async def fake_renew(
        self,
        *,
        owner_id,
        shard_group,
        shard_ids,
        lease_ttl,
        now,
    ):
        raise AssertionError('lease should not renew before the refresh interval')

    async def fake_claim(self, *, shard_ids, now):
        claim_calls.append(list(shard_ids))
        return []

    monkeypatch.setattr(SchedulerRepository, 'renew_shard_leases', fake_renew)
    monkeypatch.setattr(SchedulerRepository, 'claim_and_assign_batch', fake_claim)
    try:
        service = SchedulerService(
            session_factory=session_factory,
            config=DistributedConfig(
                shard_count=8,
                scheduler_lease_ttl=timedelta(seconds=120),
                scheduler_lease_renew_interval=60.0,
                scheduler_shard_ownership_limit=3,
                scheduler_shard_discover_interval=60.0,
            ),
            owner_id='scheduler-limited',
        )
        ordered_shards = service._shard_leases.owner_hash_ordered_shards()
        service._shard_leases._owned_shards = set(ordered_shards[:5])
        service._shard_leases._last_renew_at = utcnow()
        service._shard_leases._next_discover_at = service._shard_leases._last_renew_at + timedelta(seconds=60)

        assert await service.tick_once() == 0
    finally:
        await engine.dispose()

    assert set(claim_calls[0]) == set(ordered_shards[:3])
    assert service._shard_leases._owned_shards == set(ordered_shards[:3])


async def test_scheduler_service_applies_discover_jitter(monkeypatch):
    engine, session_factory = await _session_factory()
    renew_calls = []

    async def fake_renew(
        self,
        *,
        owner_id,
        shard_group,
        shard_ids,
        lease_ttl,
        now,
    ):
        renew_calls.append(list(shard_ids))
        return []

    async def fake_claim(self, *, shard_ids, now):
        return []

    monkeypatch.setattr(SchedulerRepository, 'renew_shard_leases', fake_renew)
    monkeypatch.setattr(SchedulerRepository, 'claim_and_assign_batch', fake_claim)
    try:
        service = SchedulerService(
            session_factory=session_factory,
            config=DistributedConfig(
                shard_count=4,
                scheduler_shard_discover_interval=60.0,
                scheduler_shard_discover_jitter=0.5,
            ),
            owner_id='scheduler-jitter',
        )
        service._shard_leases._random.uniform = lambda low, high: 0.25

        assert await service.tick_once() == 0
    finally:
        await engine.dispose()

    assert renew_calls == [[0, 1, 2, 3]]
    assert service._shard_leases._next_discover_at is not None
    elapsed = (service._shard_leases._next_discover_at - service._shard_leases._last_discover_at).total_seconds()
    assert elapsed == 75.0


@pytest.mark.parametrize('errno', [1205, 1213])
async def test_scheduler_service_run_loop_continues_after_mysql_lock_error(monkeypatch, errno):
    engine, session_factory = await _session_factory()

    class MysqlLockError(Exception):
        def __init__(self, errno: int) -> None:
            super().__init__(errno, 'Deadlock found when trying to get lock; try restarting transaction')

    try:
        service = SchedulerService(
            session_factory=session_factory,
            config=DistributedConfig(shard_count=4, scheduler_poll_interval=0.001),
            owner_id='scheduler-lock',
        )
        calls = 0

        async def fake_tick_once():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OperationalError('update shard_leases', {}, MysqlLockError(errno))
            service.stop()
            return 0

        monkeypatch.setattr(service, 'tick_once', fake_tick_once)

        await asyncio.wait_for(service.run_forever(), timeout=1)
    finally:
        await engine.dispose()

    assert calls == 2


async def test_scheduler_service_run_loop_reraises_non_lock_database_error(monkeypatch):
    engine, session_factory = await _session_factory()

    class MysqlDatabaseError(Exception):
        def __init__(self) -> None:
            super().__init__(1049, "Unknown database 'nexus_dist'")

    try:
        service = SchedulerService(
            session_factory=session_factory,
            config=DistributedConfig(shard_count=4, scheduler_poll_interval=0.001),
            owner_id='scheduler-db-error',
        )

        async def fake_tick_once():
            raise OperationalError('select 1', {}, MysqlDatabaseError())

        monkeypatch.setattr(service, 'tick_once', fake_tick_once)

        with pytest.raises(OperationalError):
            await asyncio.wait_for(service.run_forever(), timeout=1)
    finally:
        await engine.dispose()


@pytest.mark.parametrize('errno', [1205, 1213])
async def test_batch_lifecycle_retries_mysql_lock_error_without_process_local_cursor(monkeypatch, errno):
    engine, session_factory = await _session_factory()

    class MysqlLockError(Exception):
        def __init__(self, errno: int) -> None:
            super().__init__(errno, 'Deadlock found when trying to get lock; try restarting transaction')

    try:
        hook = BatchLifecycleReconciler(
            session_factory=session_factory,
            config=DistributedConfig(reconciler_poll_interval=0.001),
        )
        parent_id = 'batch-parent-lock-retry'
        claim = BatchLifecycleClaim(
            parent_execution_id=parent_id,
            owner_id='reconciler-a',
            operation_id='operation-a',
            lease_expires_at=utcnow() + timedelta(minutes=1),
        )
        lease = SimpleNamespace()
        calls = 0

        async def fake_reconcile_parent(current_claim, *, lease, page_budget, scan_budget):
            nonlocal calls
            del lease, page_budget, scan_budget
            calls += 1
            assert current_claim == claim
            if calls == 1:
                raise OperationalError('select batch for update', {}, MysqlLockError(errno))
            return 7

        monkeypatch.setattr(hook, '_reconcile_parent', fake_reconcile_parent)

        assert (
            await hook._reconcile_parent_with_retry(
                claim,
                lease=lease,  # type: ignore[arg-type]
                page_budget=2,
                scan_budget=100,
            )
            == 7
        )
        assert calls == 2
    finally:
        await engine.dispose()


async def test_batch_lifecycle_does_not_retry_non_lock_database_error(monkeypatch):
    engine, session_factory = await _session_factory()

    class MysqlDatabaseError(Exception):
        def __init__(self) -> None:
            super().__init__(1049, "Unknown database 'nexus_dist'")

    try:
        hook = BatchLifecycleReconciler(
            session_factory=session_factory,
            config=DistributedConfig(reconciler_poll_interval=0.001),
        )
        calls = 0

        claim = BatchLifecycleClaim(
            parent_execution_id='batch-parent-db-error',
            owner_id='reconciler-a',
            operation_id='operation-a',
            lease_expires_at=utcnow() + timedelta(minutes=1),
        )

        async def fake_reconcile_parent(current_claim, *, lease, page_budget, scan_budget):
            nonlocal calls
            del current_claim, lease, page_budget, scan_budget
            calls += 1
            raise OperationalError('select 1', {}, MysqlDatabaseError())

        monkeypatch.setattr(hook, '_reconcile_parent', fake_reconcile_parent)

        with pytest.raises(OperationalError):
            await hook._reconcile_parent_with_retry(
                claim,
                lease=SimpleNamespace(),  # type: ignore[arg-type]
                page_budget=2,
                scan_budget=100,
            )
        assert calls == 1
    finally:
        await engine.dispose()


async def test_reconciler_service_tick_runs_ttl_and_cleanup_reconcilers():
    engine, session_factory = await _session_factory()
    now = utcnow()
    try:
        async with session_factory() as session:
            repo = ExecutionRepository(session, shard_count=8)
            execution = await repo.create_execution(
                CreateExecution(
                    execution_id='queued-expired-by-service',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                    queue_ttl_seconds=1,
                ),
                now=now - timedelta(seconds=5),
            )
            await session.commit()

        service = ReconcilerService(
            session_factory=session_factory,
            config=DistributedConfig(shard_count=8),
            environment_controller=_NoopController(),
            dependency_build_controller=_NoopController(),
            workload_client_resolver=_NoopWorkloadClientResolver(),  # type: ignore[arg-type]
            batch_lifecycle=_NoopController(),  # type: ignore[arg-type]
        )
        assert await service.tick_once() == 2

        async with session_factory() as session:
            current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
        assert current.status == ExecutionStatus.EXPIRED
        assert current.cleanup_done_at is not None
    finally:
        await engine.dispose()
