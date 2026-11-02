from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.config import DistributedConfig
from nexus.distributed.observability.sampler import StatusSampler
from nexus.distributed.persistence.orm import (
    AssignmentRow,
    Base,
    EnvironmentAllocationRow,
    EnvironmentResourceUsageRow,
    EnvironmentRow,
    ExecutionRow,
    InstanceManagerLivenessRow,
    InstanceManagerRow,
    ObservabilitySnapshotRow,
)
from nexus.distributed.repositories.observability import STATUS_GROUPS, ObservabilityRepository


NOW = datetime(2026, 9, 14, 0, 0)


@pytest.fixture
async def database(tmp_path):
    config = DistributedConfig(database_url=f'sqlite+aiosqlite:///{tmp_path}/status.db', shard_count=1)
    engine = create_async_engine(config.read_database_url())
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield config, factory, engine
    await engine.dispose()


def environment(name):
    return EnvironmentRow(
        environment_id=name,
        name=name,
        allocation_type='predefined',
        provider_type='predefined',
        status='enabled',
        created_by='test',
        updated_by='test',
        created_at=NOW,
        updated_at=NOW,
    )


def manager(name, *, environment_id='one', age=0, draining=False, session='s1'):
    return InstanceManagerRow(
        manager_id=name,
        session_id=session,
        endpoint_url='http://manager',
        environment_id=environment_id,
        pool_id='batch',
        workload='execution',
        capacity_total=10,
        supported_spec_versions_json={},
        supported_modes_json=[],
        supported_spec_types_json=[],
        supported_instance_modes_json=[],
        last_heartbeat_at=NOW - timedelta(seconds=age),
        created_at=NOW,
        updated_at=NOW,
        draining=draining,
        metadata_json={'large': 'x' * 10000},
    )


def assignment(name, manager_id, *, session='s1', cost=3, status='active'):
    return AssignmentRow(
        assignment_id=name,
        execution_id='execution-' + name,
        attempt_id='attempt-' + name,
        attempt_seq=1,
        manager_id=manager_id,
        manager_session_id=session,
        status=status,
        slot_cost=cost,
        created_at=NOW,
        updated_at=NOW,
    )


def execution(name, **kwargs):
    return ExecutionRow(
        execution_id=name,
        idempotency_scope='test',
        idempotency_key=name,
        shard_id=0,
        mode='job',
        spec_type='nexus_single',
        instance_mode='default',
        spec_json={'large': 'x' * 10000},
        status=kwargs.pop('status', 'queued'),
        route_status='none',
        environment_id='one',
        next_schedule_at=kwargs.pop('next_schedule_at', NOW),
        created_at=NOW - timedelta(hours=1),
        updated_at=NOW,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_live_inventory_uses_effective_sessions_and_not_historical_rows(database):
    config, factory, engine = database
    async with factory() as session:
        session.add_all([environment('one'), environment('two')])
        session.add_all(
            [
                EnvironmentResourceUsageRow(
                    scope_key='environment:one',
                    scope_type='environment',
                    environment_id='one',
                    cpu_millicores=1000,
                    memory_bytes=1024,
                    slots=30,
                    manager_count=3,
                    active_batches=1,
                    updated_at=NOW,
                ),
                EnvironmentResourceUsageRow(
                    scope_key='grant:one:user',
                    scope_type='grant',
                    environment_id='one',
                    principal_id='user',
                    cpu_millicores=1000,
                    memory_bytes=1024,
                    slots=30,
                    manager_count=3,
                    active_batches=1,
                    updated_at=NOW,
                ),
                EnvironmentResourceUsageRow(
                    scope_key='environment:two',
                    scope_type='environment',
                    environment_id='two',
                    cpu_millicores=500,
                    memory_bytes=512,
                    slots=20,
                    manager_count=2,
                    active_batches=0,
                    updated_at=NOW,
                ),
                manager('healthy'),
                manager('draining', draining=True),
                manager('fenced'),
                manager('new-session', session='new'),
                manager('fresh-liveness', age=500),
                manager('old', age=500),
                manager('other', environment_id='two'),
                InstanceManagerLivenessRow(
                    manager_id='fenced',
                    session_id='s1',
                    capacity_total=10,
                    last_heartbeat_at=NOW,
                    fenced_at=NOW,
                    updated_at=NOW,
                ),
                InstanceManagerLivenessRow(
                    manager_id='new-session',
                    session_id='old',
                    capacity_total=99,
                    last_heartbeat_at=NOW,
                    fenced_at=NOW,
                    updated_at=NOW,
                ),
                InstanceManagerLivenessRow(
                    manager_id='fresh-liveness',
                    session_id='s1',
                    capacity_total=15,
                    last_heartbeat_at=NOW,
                    updated_at=NOW,
                ),
                assignment('a', 'healthy'),
                assignment('old-session', 'new-session', session='old', cost=9),
                assignment('released', 'healthy', status='released', cost=7),
                assignment('b', 'draining', cost=2),
            ]
        )
        await session.commit()
    statements = []
    event.listen(engine.sync_engine, 'before_cursor_execute', lambda _c, _u, s, *_args: statements.append(s))
    async with factory() as session:
        value = await ObservabilityRepository(session, config).live_inventory(now=NOW, environment_id='one')
        assert value['managers'] == {
            'healthy': 4,
            'draining': 1,
            'capacity_total': 45,
            'reserved_slots': 5,
            'available_slots': 32,
        }
        assert value['resource_usage']['cpu_millicores'] == 1000
        assert len(session.identity_map) == 0
        global_value = await ObservabilityRepository(session, config).resource_usage(environment_id=None)
        assert global_value['cpu_millicores'] == 1500
        assert global_value['manager_count'] == 5
    assert not any(
        'metadata_json' in statement or 'SELECT instance_managers.*' in statement for statement in statements
    )
    assert len(statements) == 3


@pytest.mark.asyncio
async def test_missing_resource_ledger_is_unknown_not_zero(database):
    config, factory, _ = database
    async with factory() as session:
        session.add(environment('one'))
        await session.commit()
        value = await ObservabilityRepository(session, config).live_inventory(now=NOW, environment_id='one')
        assert value['data_status'] == 'partial'
        assert value['resource_usage']['slots'] is None


@pytest.mark.asyncio
async def test_queue_wait_excludes_preparation_and_retry_backoff(database):
    config, factory, _ = database
    async with factory() as session:
        session.add(
            EnvironmentAllocationRow(
                allocation_id='preparing',
                top_level_execution_id='p',
                owner_principal_id='user',
                environment_id='one',
                pool_id='batch',
                workload='execution',
                status='requested',
                allocation_spec_hash='hash',
                config_version=1,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add_all(
            [
                execution('ready', queue_entered_at=NOW - timedelta(seconds=20)),
                execution('retry', next_schedule_at=NOW + timedelta(seconds=5), queue_entered_at=NOW),
                execution('prepare', allocation_id='preparing', queue_entered_at=NOW - timedelta(seconds=500)),
                execution('normal-live', status='waiting_input'),
                execution('expired-live', status='waiting_input', execution_expires_at=NOW - timedelta(seconds=1)),
            ]
        )
        await session.commit()
        value = await ObservabilityRepository(session, config).queues(now=NOW, environment_id='one')
    rows = {row['state']: row for row in value['items']}
    assert rows['queued']['schedulable'] == 1
    assert rows['queued']['preparing'] == 1
    assert rows['queued']['retry_wait'] == 1
    assert rows['queued']['oldest_wait_seconds'] == 20
    assert rows['waiting_input']['overdue'] == 1


@pytest.mark.asyncio
async def test_failed_group_keeps_last_complete_sample_and_missing_groups(database):
    config, factory, _ = database
    clock = [NOW]
    sampler = StatusSampler(config, session_factory=factory, clock=lambda: clock[0])
    try:
        assert await sampler.refresh_lease()
        assert await sampler.publish(None, 'queues', NOW, {'items': [{'count': 7}]}, None)
        clock[0] += timedelta(seconds=5)
        assert await sampler.publish(None, 'queues', clock[0], None, 'query_timeout')
        async with factory() as session:
            value = await ObservabilityRepository(session, config).snapshots(now=clock[0], environment_id=None)
        assert set(value['groups']) == set(STATUS_GROUPS)
        assert value['groups']['queues']['values'] == {'items': [{'count': 7}]}
        assert value['groups']['queues']['sampled_at'] == NOW
        assert value['groups']['queues']['error_code'] == 'query_timeout'
        assert value['data_status'] == 'partial'
        assert value['groups']['inventory']['data_status'] == 'unavailable'
    finally:
        await sampler.close()


@pytest.mark.asyncio
async def test_new_owner_fences_old_publisher_even_with_stale_local_ownership(database):
    config, factory, _ = database
    clock = [NOW]
    first = StatusSampler(config, session_factory=factory, clock=lambda: clock[0], owner_id='first')
    second = StatusSampler(config, session_factory=factory, clock=lambda: clock[0], owner_id='second')
    try:
        assert await first.refresh_lease()
        assert not await second.refresh_lease()
        clock[0] += timedelta(seconds=61)
        assert await second.refresh_lease()
        assert await second.publish(None, 'queues', clock[0], {'count': 2}, None)
        first._expires_at = clock[0] + timedelta(seconds=60)
        assert not await first.publish(None, 'queues', clock[0], {'count': 1}, None)
        async with factory() as session:
            row = await session.get(ObservabilitySnapshotRow, ('global', '*', 'queues'))
            assert row.values_json == {'count': 2}
            assert row.publisher_id == second.lease.owner_id
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_sampler_collects_groups_without_any_http_reader(database):
    config, factory, _ = database
    sampler = StatusSampler(config, session_factory=factory, clock=lambda: NOW)
    try:
        assert await sampler.tick() == len(STATUS_GROUPS)
        async with factory() as session:
            value = await ObservabilityRepository(session, config).snapshots(now=NOW, environment_id=None)
        assert value['data_status'] == 'ready'
        assert sampler.gauge_values('queue_depth')
        assert all(value == 0 for value, _ in sampler.gauge_values('queue_depth'))
        sampler.stop()
        assert sampler.gauge_values('queue_depth') == []
    finally:
        await sampler.close()
