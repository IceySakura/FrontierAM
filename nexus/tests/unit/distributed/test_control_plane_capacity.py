from __future__ import annotations

import asyncio

from datetime import timedelta
from typing import Any, cast

import pytest

from tests.stress.distributed.capacity_support import (
    LogicalManagerHeartbeatTarget,
    ManagerHeartbeatLoad,
    source_identity,
)
from tests.stress.distributed.run_control_plane_capacity import (
    SHARD_COUNT,
    TOPOLOGIES,
    _config,
    _execution_values,
    _register_logical_managers,
    _require_mysql,
    validate_recovery_config,
    validate_topology,
)

from nexus.distributed.config import DistributedConfig
from nexus.distributed.persistence.db import create_engine, create_session_factory
from nexus.distributed.persistence.orm import create_schema


def test_capacity_benchmark_topologies_cover_every_shard_once() -> None:
    assert TOPOLOGIES == ((1, 64), (4, 16), (16, 4), (32, 2))
    assert all(replicas * ownership_limit == SHARD_COUNT for replicas, ownership_limit in TOPOLOGIES)


def test_capacity_benchmark_rejects_incomplete_shard_topology() -> None:
    validate_topology(replicas=16, ownership_limit=4)

    with pytest.raises(ValueError, match='exactly cover'):
        validate_topology(replicas=16, ownership_limit=3)


def test_capacity_benchmark_validates_scheduler_restart_target() -> None:
    validate_recovery_config(replicas=4, worker_index=3, restart_after_seconds=1)
    validate_recovery_config(replicas=4, worker_index=0, restart_after_seconds=0)

    with pytest.raises(ValueError, match='worker index'):
        validate_recovery_config(replicas=4, worker_index=4, restart_after_seconds=1)
    with pytest.raises(ValueError, match='non-negative'):
        validate_recovery_config(replicas=4, worker_index=0, restart_after_seconds=-1)


def test_capacity_source_identity_is_order_independent_and_content_sensitive(tmp_path) -> None:
    (tmp_path / 'a.py').write_text('a = 1\n')
    (tmp_path / 'b.py').write_text('b = 2\n')

    first = source_identity(tmp_path, ['b.py', 'a.py'])
    second = source_identity(tmp_path, ['a.py', 'b.py'])
    (tmp_path / 'b.py').write_text('b = 3\n')
    changed = source_identity(tmp_path, ['a.py', 'b.py'])

    assert first == second
    assert first['files'] == ['a.py', 'b.py']
    assert first['sha256'] != changed['sha256']


def test_capacity_benchmark_uses_production_manager_liveness_intervals() -> None:
    config = _config(
        'sqlite+aiosqlite:///:memory:',
        ownership_limit=64,
        task_count=50_000,
        batch_size=250,
        replicas=1,
    )

    assert config.manager_heartbeat_interval == 5
    assert config.manager_healthy_ttl == timedelta(seconds=15)
    assert config.manager_self_fence_timeout == timedelta(seconds=30)


def test_capacity_benchmark_rejects_non_mysql_database() -> None:
    with pytest.raises(ValueError, match='requires MySQL'):
        _require_mysql('sqlite')

    _require_mysql('mysql')


async def test_manager_heartbeat_load_updates_every_logical_manager(tmp_path) -> None:
    database_url = f'sqlite+aiosqlite:///{tmp_path / "manager-heartbeats.db"}'
    engine = create_engine(DistributedConfig(database_url=database_url))
    session_factory = create_session_factory(engine)
    await create_schema(engine)
    try:
        targets = await _register_logical_managers(
            session_factory,
            manager_count=2,
            task_count=64,
            prefix='heartbeat-load',
        )
        load = ManagerHeartbeatLoad(
            session_factory=session_factory,
            targets=targets,
            interval_seconds=0.01,
            healthy_ttl=timedelta(seconds=0.05),
            self_fence_timeout=timedelta(seconds=0.1),
        )

        await load.start()
        await asyncio.sleep(0.03)
        report = await load.stop()
    finally:
        await engine.dispose()

    assert report['manager_count'] == 2
    assert report['successes'] >= 4
    assert report['transient_failures'] == 0
    assert report['fatal_failures'] == 0
    assert report['fenced_managers'] == 0
    assert report['self_fenced_managers'] == 0
    assert report['stale_managers'] == 0
    assert report['latency_samples'] == report['successes']


async def test_manager_heartbeat_load_self_fences_when_transaction_never_returns(monkeypatch, tmp_path) -> None:
    database_url = f'sqlite+aiosqlite:///{tmp_path / "hung-manager-heartbeat.db"}'
    engine = create_engine(DistributedConfig(database_url=database_url))
    session_factory = create_session_factory(engine)
    await create_schema(engine)
    target = LogicalManagerHeartbeatTarget(manager_id='manager-hung', session_id='session-hung', capacity_total=1)
    load = ManagerHeartbeatLoad(
        session_factory=session_factory,
        targets=[target],
        interval_seconds=0.01,
        healthy_ttl=timedelta(seconds=0.02),
        self_fence_timeout=timedelta(seconds=0.03),
    )

    async def heartbeat_never_returns(_target):
        await asyncio.Event().wait()
        return True

    monkeypatch.setattr(load, '_heartbeat_once', heartbeat_never_returns)

    try:
        with pytest.raises(RuntimeError, match='failed to start'):
            await asyncio.wait_for(load.start(), timeout=0.2)
    finally:
        await engine.dispose()


async def test_manager_heartbeat_load_stops_after_inflight_transaction_finishes(monkeypatch) -> None:
    target = LogicalManagerHeartbeatTarget(
        manager_id='manager-stopping', session_id='session-stopping', capacity_total=1
    )
    load = ManagerHeartbeatLoad(
        session_factory=cast(Any, None),
        targets=[target],
        interval_seconds=0.1,
        healthy_ttl=timedelta(seconds=1),
        self_fence_timeout=timedelta(seconds=2),
    )
    started = asyncio.Event()
    release = asyncio.Event()
    cancelled = False

    async def heartbeat_once(_target):
        nonlocal cancelled
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled = True
            raise
        return True

    async def report():
        return {}

    monkeypatch.setattr(load, '_heartbeat_once', heartbeat_once)
    monkeypatch.setattr(load, '_report', report)

    start_task = asyncio.create_task(load.start())
    await started.wait()
    stop_task = asyncio.create_task(load.stop())
    await asyncio.sleep(0)
    stopped_before_release = stop_task.done()
    release.set()
    await asyncio.gather(start_task, stop_task)

    assert not stopped_before_release
    assert not cancelled


def test_capacity_benchmark_generates_allocation_backed_batch_children() -> None:
    rows = _execution_values(prefix='scheduler-r4', task_count=3, expired=False)

    assert {row['owner_principal_id'] for row in rows} == {'scheduler-r4-owner'}
    assert {row.get('environment_id') for row in rows} == {'scheduler-r4-environment'}
    assert {row.get('pool_id') for row in rows} == {'scheduler-r4-execution'}
    assert {row.get('allocation_id') for row in rows} == {'scheduler-r4-allocation'}
    assert {row.get('workload') for row in rows} == {'execution'}
    assert all(row.get('allocation_spec_hash') for row in rows)
    assert all(
        row['placement_json']
        == {
            'environment': 'scheduler-r4-environment',
            'pool': 'scheduler-r4-execution',
            'allocation_id': 'scheduler-r4-allocation',
            'labels': {'workload': 'execution'},
        }
        for row in rows
    )
