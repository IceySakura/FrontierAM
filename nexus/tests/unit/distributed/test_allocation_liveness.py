from __future__ import annotations

import argparse
import json
import os

from datetime import datetime, timedelta

import pytest

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.cli.commands import distributed as distributed_cli
from nexus.distributed.config import DistributedConfig
from nexus.distributed.environment.liveness import ManagerLivenessPolicy, allocation_liveness, startup_liveness
from nexus.distributed.instance_manager.liveness import allocation_heartbeat_cutoff
from nexus.distributed.persistence.orm import (
    EnvironmentAllocationRow,
    InstanceManagerLivenessRow,
    InstanceManagerRow,
    create_schema,
)
from nexus.distributed.repositories.reconciler import ReconcilerRepository
from nexus.distributed.repositories.scheduler import SchedulerRepository
from nexus.distributed.repositories.submission import _submission_sha256
from nexus.distributed.submission.specs import SingleManagerResources
from nexus.distributed.utils.clock import utcnow
from nexus.distributed.utils.identity import canonical_sha256
from tests.unit.distributed.submission.test_typed_submission_repository import _dynamic_single
from tests.unit.distributed.test_repositories import _create_execution, _register_manager


POLICY = {'self_fence_timeout_seconds': 90, 'executor_hang_timeout_seconds': 180, 'dead_timeout_seconds': 120}
SNAPSHOT = {'version': 1, **POLICY}
INVALID = [
    None,
    {},
    {**SNAPSHOT, 'version': True},
    {**SNAPSHOT, 'version': 2},
    {**SNAPSHOT, 'dead_timeout_seconds': '120'},
    {**SNAPSHOT, 'dead_timeout_seconds': 120.5},
    {**SNAPSHOT, 'dead_timeout_seconds': True},
    {**SNAPSHOT, 'dead_timeout_seconds': 601},
    {**SNAPSHOT, 'self_fence_timeout_seconds': 0},
    {**SNAPSHOT, 'executor_hang_timeout_seconds': 89},
    {**SNAPSHOT, 'dead_timeout_seconds': 90},
    {**SNAPSHOT, 'unexpected': 1},
]


@pytest.fixture
async def session_factory():
    # An explicitly isolated database may be supplied for the DevCloud MySQL acceptance job.
    engine = create_async_engine(os.getenv('NEXUS_LIVENESS_TEST_DATABASE_URL', 'sqlite+aiosqlite:///:memory:'))
    await create_schema(engine)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        if os.getenv('NEXUS_LIVENESS_TEST_DATABASE_URL'):
            from nexus.distributed.persistence.orm import Base

            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


@pytest.mark.parametrize('value', INVALID)
def test_invalid_snapshot_fails_closed(value):
    with pytest.raises(ValueError, match='liveness|validation'):
        allocation_liveness({'manager_liveness': value})


def test_optional_resource_roundtrip_and_legacy_hash():
    resources = SingleManagerResources(cpu_cores=1, memory_gib=1)
    assert 'liveness' not in resources.model_dump()
    assert SingleManagerResources.model_validate(resources.model_dump()) == resources
    with pytest.raises(ValueError, match='manager_liveness'):
        SingleManagerResources(cpu_cores=1, memory_gib=1, liveness=None)
    submission = _dynamic_single()
    legacy = submission.model_dump(mode='json')
    legacy.pop('run_trace_backends')
    assert _submission_sha256(submission) == canonical_sha256(legacy)
    submission.resources.managers.liveness = ManagerLivenessPolicy(**POLICY)
    assert _submission_sha256(submission) != canonical_sha256(legacy)


def test_platform_boundaries():
    policy = ManagerLivenessPolicy(**POLICY)
    policy.validate_platform(DistributedConfig())
    for overrides in (
        {'manager_healthy_ttl': timedelta(seconds=90)},
        {'manager_hard_exit_timeout': timedelta(seconds=30)},
    ):
        with pytest.raises(ValueError, match='manager_liveness'):
            policy.validate_platform(DistributedConfig().model_copy(update=overrides))


def test_startup_identity_and_capability_guard():
    spec = {
        'allocation_id': 'a',
        'environment_id': 'e',
        'pool_id': 'p',
        'workload': 'execution',
        'allocation_type': 'dynamic',
        'manager_liveness': SNAPSHOT,
    }
    kwargs = {'required': True, 'allocation_id': 'a', 'environment_id': 'e', 'pool_id': 'p', 'workload': 'execution'}
    assert startup_liveness(spec, **kwargs).snapshot() == SNAPSHOT
    for changes in ({'required': False}, {'allocation_id': 'other'}, {'workload': 'batch'}, {'pool_id': 'other'}):
        with pytest.raises(ValueError, match='allocation_liveness'):
            startup_liveness(spec, **{**kwargs, **changes})
    with pytest.raises(ValueError, match='allocation_liveness'):
        startup_liveness({}, **kwargs)


def test_manager_policy_overrides_environment(monkeypatch):
    spec = {
        'allocation_id': 'a',
        'environment_id': 'e',
        'pool_id': 'p',
        'workload': 'execution',
        'allocation_type': 'dynamic',
        'manager_liveness': SNAPSHOT,
    }
    monkeypatch.setenv('NEXUS_DIST_ALLOCATION_SPEC_JSON', json.dumps(spec))
    monkeypatch.setenv('NEXUS_DIST_MANAGER_ALLOCATION_ID', 'a')
    args = argparse.Namespace(
        distributed_command='instance-manager',
        require_allocation_liveness_v1=True,
        environment_id='e',
        pool_id='p',
        workload='execution',
    )
    config = distributed_cli._config_from_args(args)
    assert config.manager_self_fence_timeout.total_seconds() == 90
    assert config.manager_executor_hang_timeout.total_seconds() == 180
    assert config.manager_dead_ttl.total_seconds() == 120


def _allocation(allocation_id, spec, now):
    return EnvironmentAllocationRow(
        allocation_id=allocation_id,
        top_level_execution_id='unused',
        owner_principal_id='principal-a',
        environment_id='predefined-default',
        pool_id='default',
        workload='execution',
        status='ready',
        desired_manager_count=1,
        ready_manager_count=1,
        manager_count=1,
        cpu_millicores=1000,
        memory_bytes=1024,
        slots=1,
        active_batches=0,
        allocation_spec_json=spec,
        allocation_spec_hash='a' * 64,
        config_version=1,
        status_version=1,
        last_progress_at=now,
        created_at=now,
        updated_at=now,
    )


@pytest.mark.parametrize('value', [*INVALID, SNAPSHOT, 'absent'])
async def test_sql_and_python_policy_parity(session_factory, value):
    now = datetime(2026, 9, 22, 12, 0, 0, 123456)
    await _register_manager(session_factory, manager_id='policy-manager')
    async with session_factory() as session:
        spec = {} if value == 'absent' else {'manager_liveness': value}
        session.add(_allocation('a', spec, now))
        manager = await session.get(InstanceManagerRow, 'policy-manager')
        manager.allocation_id = 'a'
        await session.flush()
        cutoff = allocation_heartbeat_cutoff(
            now=now, default_ttl=timedelta(seconds=45.5), dialect=session.get_bind().dialect.name
        )
        actual = (
            await session.execute(
                select(cutoff)
                .select_from(InstanceManagerRow)
                .outerjoin(
                    EnvironmentAllocationRow, EnvironmentAllocationRow.allocation_id == InstanceManagerRow.allocation_id
                )
            )
        ).scalar_one()
        if value == 'absent':
            assert actual == now - timedelta(seconds=45.5)
        elif value is SNAPSHOT:
            assert actual == now - timedelta(seconds=120)
        else:
            assert actual is None


async def _assigned(session_factory, name, *, policy, now):
    await _register_manager(session_factory, manager_id=name)
    execution = await _create_execution(session_factory, execution_id=f'exe-{name}')
    async with session_factory() as session:
        assignment = await SchedulerRepository(
            session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15)
        ).claim_and_assign(shard_ids=[execution.shard_id], now=utcnow() + timedelta(seconds=2))
        assert assignment is not None
        manager = await session.get(InstanceManagerRow, assignment.manager_id)
        live = await session.get(InstanceManagerLivenessRow, assignment.manager_id)
        manager.last_heartbeat_at = live.last_heartbeat_at = now
        if policy != 'legacy':
            manager.allocation_id = f'alloc-{name}'
            if policy != 'missing':
                session.add(_allocation(manager.allocation_id, {'manager_liveness': policy}, now))
        await session.commit()
    return assignment.manager_id


async def test_mixed_timeouts_do_not_starve_limited_scan_and_expire_exactly(session_factory):
    now = utcnow().replace(microsecond=0)
    await _assigned(session_factory, 'long', policy=SNAPSHOT, now=now)
    await _assigned(session_factory, 'legacy', policy='legacy', now=now)
    async with session_factory() as session:
        repo = ReconcilerRepository(session, batch_size=1)
        assert await repo.reconcile_manager_dead(now=now + timedelta(seconds=45)) == 1
        assert repo.last_stats.scanned == 1
        assert await repo.reconcile_manager_dead(now=now + timedelta(seconds=119, microseconds=999999)) == 0
        assert await repo.reconcile_manager_dead(now=now + timedelta(seconds=120)) == 1
        await session.commit()


@pytest.mark.parametrize('policy', [None, 'missing', SNAPSHOT])
async def test_explicit_fence_is_immediate_even_for_invalid_policy(session_factory, policy):
    now = utcnow().replace(microsecond=0)
    manager_id = await _assigned(session_factory, 'fenced', policy=policy, now=now)
    async with session_factory() as session:
        live = await session.get(InstanceManagerLivenessRow, manager_id)
        live.fenced_at = now
        await session.flush()
        assert await ReconcilerRepository(session).reconcile_manager_dead(now=now) == 1


@pytest.mark.parametrize('policy', [None, 'missing'])
async def test_invalid_or_missing_allocation_never_uses_default_expiry(session_factory, policy):
    now = utcnow().replace(microsecond=0)
    await _assigned(session_factory, 'invalid', policy=policy, now=now)
    async with session_factory() as session:
        assert await ReconcilerRepository(session).reconcile_manager_dead(now=now + timedelta(hours=1)) == 0


async def test_custom_policy_rechecks_fresh_heartbeat_after_scan(session_factory, monkeypatch):
    now = utcnow().replace(microsecond=0)
    manager_id = await _assigned(session_factory, 'race', policy=SNAPSHOT, now=now)
    async with session_factory() as session:
        repo = ReconcilerRepository(session)
        original = repo._lock_assignment_candidates

        async def refresh(candidates):
            live = await session.get(InstanceManagerLivenessRow, manager_id)
            live.last_heartbeat_at = now + timedelta(seconds=120)
            await session.flush()
            return await original(candidates)

        monkeypatch.setattr(repo, '_lock_assignment_candidates', refresh)
        assert await repo.reconcile_manager_dead(now=now + timedelta(seconds=120)) == 0
        assert repo.last_stats.scanned == 1


@pytest.mark.parametrize('kind', ['single', 'live', 'batch'])
async def test_submission_persists_only_execution_policy(session_factory, kind):
    from nexus.distributed.repositories.submission import SubmissionRepository
    from nexus.distributed.submission.specs import BatchSubmission, LiveSubmission, SingleSubmission
    from tests.unit.distributed.submission.test_typed_submission_repository import _config, _seed_environment

    async with session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-dynamic',
            allocation_type='dynamic',
            provider_type='kubernetes',
            environment_limits={},
            grant_limits={},
        )
        payload = _dynamic_single().model_dump(mode='json', exclude_none=True)
        payload['resources']['managers']['liveness'] = POLICY
        if kind == 'batch':
            payload.pop('run_input')
            payload['resources']['batch_manager'] = {'cpu_cores': 1, 'memory_gib': 1}
            payload['generation'] = {
                'mode': 'remote_materialize',
                'source': {
                    'type': 'compose',
                    'compose': {'parts': [{'kind': 'set', 'value': 'task_name=identity', 'source': 'test'}]},
                },
            }
            payload['batch'] = {'max_inflight': 1}
        model = {'single': SingleSubmission, 'live': LiveSubmission, 'batch': BatchSubmission}[kind]
        submission = model.model_validate(payload)
        repo = SubmissionRepository(session, config=_config())
        kwargs = {
            'actor_principal_id': 'principal-submit',
            'owner_principal_id': 'principal-submit',
            'owner_tenant': 'tenant',
            'now': utcnow(),
        }
        result = await repo.create(submission, **kwargs)
        await session.commit()
        assert (await repo.create(submission, **kwargs)).reused
        rows = (
            await session.scalars(
                select(EnvironmentAllocationRow).where(
                    EnvironmentAllocationRow.top_level_execution_id == result.execution.execution_id
                )
            )
        ).all()
        assert len(rows) == (2 if kind == 'batch' else 1)
        for row in rows:
            if row.workload == 'execution':
                assert row.allocation_spec_json['manager_liveness'] == SNAPSHOT
                assert row.allocation_spec_hash == canonical_sha256(row.allocation_spec_json)
                from nexus.distributed.repositories.allocation import _allocation_record

                assert _allocation_record(row).configured_manager_liveness == SNAPSHOT
            else:
                assert 'manager_liveness' not in row.allocation_spec_json
        from nexus.distributed.errors import DomainConflictError

        submission.resources.managers.liveness = ManagerLivenessPolicy(**{**POLICY, 'dead_timeout_seconds': 121})
        with pytest.raises(DomainConflictError):
            await repo.create(submission, **kwargs)


def test_cli_requires_all_three_timeouts():
    from nexus.cli.commands.submit import _manager_liveness_args
    from tests.unit.distributed.submission.test_submit_cli import _parser

    base = ['submit', 'single', '--environment', 'env-dynamic']
    # The argument parser owns flag spelling; the helper owns the all-or-none contract.
    args = _parser().parse_args([
        *base,
        '--manager-self-fence-timeout-seconds',
        '90',
        '--manager-executor-hang-timeout-seconds',
        '180',
        '--manager-dead-timeout-seconds',
        '120',
    ])
    assert _manager_liveness_args(args) == POLICY
    args.manager_dead_timeout_seconds = None
    with pytest.raises(ValueError, match='all three'):
        _manager_liveness_args(args)
    args.manager_dead_timeout_seconds = 120
    args.pool = 'shared'
    with pytest.raises(ValueError, match='dynamic'):
        _manager_liveness_args(args)


async def test_custom_policy_fence_cas_preserves_concurrent_heartbeat(session_factory, monkeypatch):
    from sqlalchemy import update
    from sqlalchemy.sql.dml import Update

    now = utcnow().replace(microsecond=0)
    manager_id = await _assigned(session_factory, 'cas-race', policy=SNAPSHOT, now=now)
    async with session_factory() as session:
        original = session.execute
        injected = False

        async def heartbeat_before_cas(statement, *args, **kwargs):
            nonlocal injected
            if isinstance(statement, Update) and statement.table.name == InstanceManagerLivenessRow.__tablename__:
                assert not injected
                injected = True
                async with session_factory() as writer:
                    await writer.execute(
                        update(InstanceManagerLivenessRow)
                        .where(InstanceManagerLivenessRow.manager_id == manager_id)
                        .values(last_heartbeat_at=now + timedelta(seconds=120))
                    )
                    await writer.commit()
            return await original(statement, *args, **kwargs)

        monkeypatch.setattr(session, 'execute', heartbeat_before_cas)
        assert await ReconcilerRepository(session).reconcile_manager_dead(now=now + timedelta(seconds=120)) == 0
        assert injected


async def test_policy_beats_database_config_scope(session_factory):
    from nexus.distributed.config.store import load_distributed_config
    from nexus.distributed.persistence.orm import DistributedConfigOverrideRow

    async with session_factory() as session:
        now = utcnow().replace(microsecond=0)
        session.add(
            DistributedConfigOverrideRow(
                role='instance-manager',
                identity='*',
                version=1,
                config_json={
                    'manager_self_fence_timeout': 30,
                    'manager_executor_hang_timeout': 180,
                    'manager_dead_ttl': 45,
                },
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()
    config = await load_distributed_config(
        session_factory,
        database_url='sqlite+aiosqlite:///:memory:',
        role='instance-manager',
        overrides=ManagerLivenessPolicy(**POLICY).config_overrides(),
    )
    assert config.manager_self_fence_timeout.total_seconds() == 90
    assert config.manager_dead_ttl.total_seconds() == 120


@pytest.mark.parametrize('custom', [False, True])
def test_renderer_adds_capability_flag_only_for_explicit_policy(custom):
    from nexus.distributed.config.views import environment_config_view
    from nexus.distributed.environment.ports import AllocationSnapshot
    from nexus.distributed.environment.rendering import AllocationWorkloadRenderer

    spec = {'code': {'mode': 'config_only'}}
    if custom:
        spec['manager_liveness'] = SNAPSHOT
    snapshot = AllocationSnapshot(
        allocation_id='allocation-a',
        top_level_execution_id='execution-a',
        owner_principal_id='principal-a',
        environment_id='environment-a',
        allocation_type='dynamic',
        provider_type='kubernetes',
        pool_id='execution',
        workload='execution',
        status='requested',
        desired_manager_count=1,
        ready_manager_count=0,
        cpu_millicores=1000,
        memory_bytes=1024**3,
        slots=1,
        allocation_spec=spec,
        allocation_spec_hash='a' * 64,
        provider_resource_ref=None,
    )
    config = DistributedConfig(environment_database_url_secret_name='test-db')
    rendered = AllocationWorkloadRenderer(config=environment_config_view(config)).render(
        snapshot, resource_kind='StatefulSet'
    )
    container = rendered.workload['spec']['template']['spec']['containers'][0]
    assert ('--require-allocation-liveness-v1' in container['args']) is custom
    value = next(item['value'] for item in container['env'] if item['name'] == 'NEXUS_DIST_ALLOCATION_SPEC_JSON')
    assert ('manager_liveness' in json.loads(value)) is custom


def test_bootstrap_forwards_capability_flag(tmp_path, monkeypatch):
    from nexus.distributed.instance_manager.bootstrap import PreparedEnvironment, exec_instance_manager

    captured = {}
    monkeypatch.setenv('POD_NAME', 'allocation-a-0')
    monkeypatch.setattr(os, 'chdir', lambda _: None)
    monkeypatch.setattr(
        os, 'execve', lambda executable, argv, environment: captured.update(argv=argv, environment=environment)
    )
    prepared = PreparedEnvironment(
        workspace=tmp_path,
        venv=tmp_path,
        environment_id='env-a',
        pool_id='pool-a',
        workload='execution',
        manager_labels={},
    )
    exec_instance_manager(
        prepared=prepared, port=8082, capacity=1, allocation_id='allocation-a', require_allocation_liveness_v1=True
    )
    assert '--require-allocation-liveness-v1' in captured['argv']
    assert captured['environment']['NEXUS_DIST_MANAGER_ALLOCATION_ID'] == 'allocation-a'


@pytest.mark.parametrize('spec', [None, [], 'corrupt'])
async def test_non_object_spec_is_excluded_before_limit(session_factory, spec):
    now = utcnow().replace(microsecond=0)
    manager_id = await _assigned(session_factory, 'corrupt', policy=SNAPSHOT, now=now)
    async with session_factory() as session:
        manager = await session.get(InstanceManagerRow, manager_id)
        allocation = await session.get(EnvironmentAllocationRow, manager.allocation_id)
        allocation.allocation_spec_json = spec
        await session.flush()
        repo = ReconcilerRepository(session, batch_size=1)
        assert await repo.reconcile_manager_dead(now=now + timedelta(hours=1)) == 0
        assert repo.last_stats.scanned == 0
        with pytest.raises(ValueError, match='invalid_allocation_liveness_spec'):
            allocation_liveness(spec)
