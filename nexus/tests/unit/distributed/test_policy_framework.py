from __future__ import annotations

import asyncio
import importlib

from datetime import timedelta

import pytest

from aiohttp import web
from aiohttp.test_utils import TestServer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.errors import DomainValidationError
from nexus.distributed.persistence.orm import (
    AssignmentRow,
    ExecutionEventRow,
    ExecutionRow,
    PolicyScopeRow,
    create_schema,
)
from nexus.distributed.policies.collections.infer_proxy import InferProxyCapacitySource, InferProxyTidePolicy
from nexus.distributed.policies.executor import PolicyExecutor
from nexus.distributed.policies.models import PolicyExecutorConfig, PolicyInput, PolicyResult, SignalObservation
from nexus.distributed.policies.scheduler import resolve_selection
from nexus.distributed.policies.workers import PolicyWorkerPool
from nexus.distributed.repositories.policy import POLICY_SHARD_GROUP, PolicyRepository, policy_scope_key
from nexus.distributed.repositories.scheduler import SchedulerRepository
from nexus.distributed.repositories.shard_lease import ShardLeaseRepository
from nexus.distributed.utils.clock import utcnow
from tests.unit.distributed.test_repositories import _create_execution, _register_manager


ENV = 'predefined-default'
SCOPE = 'shared-model-key'
IMPORTS = ['tests.unit.distributed.policy_plugins']


def policy_config(**overrides):
    binding = {
        'environments': [ENV],
        'mode': 'enforce',
        'scopes': {SCOPE: {'source_config': {'allow': True}}},
        'policy': 'test_gate',
        'source': 'test_signal',
        'refresh_seconds': 1,
        'max_age_seconds': 30,
        'max_defer_seconds': 1,
        **overrides,
    }
    return PolicyExecutorConfig.model_validate({'imports': IMPORTS, 'poll_seconds': 0.1, 'bindings': {'tide': binding}})


@pytest.fixture
async def sessions(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path}/policy.db')
    await create_schema(engine)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


def tide_input(values, *, params=None, now=None, age=120):
    now = now or utcnow()
    return PolicyInput(
        scope_id=SCOPE,
        now=now,
        params=params or {},
        observation=SignalObservation(
            version='sample',
            observed_at=now,
            valid_until=now + timedelta(seconds=age),
            values=values,
        ),
    )


@pytest.mark.parametrize(
    ('values', 'allow', 'reason'),
    [
        ({'remaining': 10, 'factor': 0.01, 'hard_brake': False}, True, 'capacity_available'),
        ({'remaining': 10, 'factor': 2, 'hard_brake': True}, False, 'hard_brake'),
        ({'remaining': 0, 'hard_brake': False}, False, 'quota_exhausted'),
        ({'remaining': 10}, False, 'signal_incomplete'),
        ({'remaining': True, 'hard_brake': False}, False, 'signal_incomplete'),
        ({'hard_brake': False}, False, 'signal_incomplete'),
    ],
)
def test_tide_admission_uses_current_quota_and_hard_brake(values, allow, reason):
    policy = InferProxyTidePolicy()
    # Previous hysteresis state must not affect the new rule or fill a missing signal.
    result = policy.evaluate(tide_input(values), {'congestion_blocked': True, 'quota_blocked': False})
    assert result.decision['allow'] is allow
    assert result.reason_code == reason


def test_tide_rejects_expired_signals():
    result = InferProxyTidePolicy().evaluate(tide_input({'remaining': 10, 'hard_brake': False}, age=0), {})
    assert result.decision['allow'] is False
    assert result.reason_code == 'signal_expired'


async def test_capacity_source_does_not_persist_credentials_or_invalid_fields(tmp_path):
    calls = []

    async def capacity(request):
        calls.append(await request.json())
        return web.json_response({
            'key_remaining_raw_qpm': True,
            'scaling_advice': {'factor': False},
            'echo': 'private-test-key',
        })

    app = web.Application()
    app.router.add_post('/capacity', capacity)
    async with TestServer(app) as server:
        observation = await InferProxyCapacitySource().read(
            {'model': 'test-model', 'api_key': 'private-test-key', 'url': str(server.make_url('/capacity'))},
            max_age_seconds=10,
        )
    assert observation.values == {}
    assert 'private-test-key' not in observation.model_dump_json()
    assert calls == [{'model': 'test-model', 'api_key': 'private-test-key'}]


async def test_worker_timeout_reaps_process_and_other_workers_continue():
    pool = PolicyWorkerPool(size=2)
    request = {
        'imports': IMPORTS,
        'kind': 'scheduler',
        'policy': 'test_gate',
        'source': 'test_signal',
        'source_config': {'allow': True},
        'scope_id': SCOPE,
        'params': {'delay': 10},
        'previous_state': {},
        'max_age_seconds': 30,
    }
    try:
        slow = asyncio.create_task(pool.run(request, timeout=2))
        fast = asyncio.create_task(pool.run({**request, 'params': {}}, timeout=10))
        ticks = 0
        while not slow.done():
            await asyncio.sleep(0.05)
            ticks += 1
        with pytest.raises(TimeoutError):
            await slow
        assert (await fast)['evaluation']['decision']['allow'] is True
        assert ticks >= 5
        assert any(worker.process is None for worker in pool.workers)
        # A timed out worker can be reused with a fresh process.
        assert (await pool.run({**request, 'params': {}}, timeout=10))['evaluation']['next_state']['calls'] == 1
    finally:
        await pool.close()
    assert all(worker.process is None for worker in pool.workers)


async def test_same_executor_supports_another_decision_type():
    pool = PolicyWorkerPool(size=1)
    try:
        result = await pool.run(
            {
                'imports': IMPORTS,
                'kind': 'request_rate',
                'policy': 'test_rate',
                'source': 'test_signal',
                'source_config': {'qpm': 42},
                'scope_id': SCOPE,
                'params': {},
                'previous_state': {},
                'max_age_seconds': 30,
            },
            timeout=10,
        )
        assert result['evaluation']['decision'] == {'qpm': 42}
    finally:
        await pool.close()


async def seed_policy(sessions, config, *, allow, preferred=None):
    now = utcnow()
    async with sessions() as session:
        repo = PolicyRepository(session, shard_count=8)
        await repo.ensure_scopes(config, now=now)
        key = policy_scope_key('scheduler', 'tide', SCOPE)
        await repo.demand([key], now=now)
        await ShardLeaseRepository(session, shard_count=8).renew(
            owner_id='publisher',
            shard_group=POLICY_SHARD_GROUP,
            shard_ids=list(range(8)),
            lease_ttl=timedelta(seconds=60),
            now=now,
        )
        claims = await repo.due(list(range(8)), now=now + timedelta(seconds=1), idle_seconds=60, limit=10)
        if not claims:
            row = await session.get(PolicyScopeRow, key)
            row.next_evaluate_at = now
            await session.flush()
            claims = await repo.due(list(range(8)), now=now, idle_seconds=60, limit=10)
        claim = claims[0]
        signal = SignalObservation(version='one', observed_at=now, valid_until=now + timedelta(seconds=30), values={})
        result = PolicyResult(
            decision={'allow': allow, 'preferred_managers': preferred or []},
            valid_until=signal.valid_until,
            reason_code='capacity_available' if allow else 'quota_exhausted',
            next_state={'quota_blocked': not allow},
        )
        assert await repo.publish(
            claim,
            owner_id='publisher',
            binding=config.bindings['tide'],
            result=result,
            observation=signal,
            error=None,
            evaluation_id='evaluation',
            now=now,
            delay=1,
        )
        await session.commit()
        return key, claim


async def test_shared_snapshot_defers_without_assignments_then_resumes_once(sessions):
    config = policy_config()
    await _register_manager(sessions, manager_id='manager', capacity_total=4)
    for index in range(2):
        await _create_execution(
            sessions,
            execution_id=f'exe-{index}',
            placement={
                'environment': ENV,
                'pool': 'default',
                'scheduler_policy': {'binding': 'tide', 'scopes': [SCOPE]},
            },
        )
    await seed_policy(sessions, config, allow=False)
    async with sessions() as session:
        repo = SchedulerRepository(
            session, shard_count=8, manager_healthy_ttl=timedelta(seconds=60), policy_config=config
        )
        assert await repo.claim_and_assign_batch(shard_ids=list(range(8)), now=utcnow()) == []
        assert len(repo.policy_demands) == 1
        await session.commit()
        executions = (await session.scalars(select(ExecutionRow))).all()
        assert all(e.status == 'queued' and e.attempt_count == 0 for e in executions)
        assert all(e.last_schedule_error == 'policy_deferred:quota_exhausted' for e in executions)
        assert not list(await session.scalars(select(AssignmentRow)))
    await seed_policy(sessions, config, allow=True)
    async with sessions() as session:
        repo = SchedulerRepository(
            session, shard_count=8, manager_healthy_ttl=timedelta(seconds=60), policy_config=config
        )
        assigned = await repo.claim_and_assign_batch(shard_ids=list(range(8)), now=utcnow() + timedelta(seconds=2))
        assert len(assigned) == 2
        await session.commit()
        assert await repo.claim_and_assign_batch(shard_ids=list(range(8)), now=utcnow() + timedelta(seconds=3)) == []
        events = list(
            await session.scalars(select(ExecutionEventRow).where(ExecutionEventRow.type == 'scheduler.policy_applied'))
        )
        assert len(events) == 2


async def test_policy_preference_participates_in_lock_candidates(sessions):
    config = policy_config()
    for manager in ['a', 'b']:
        await _register_manager(sessions, manager_id=manager)
    await _create_execution(
        sessions,
        execution_id='choose-b',
        placement={
            'environment': ENV,
            'pool': 'default',
            'scheduler_policy': {'binding': 'tide', 'scopes': [SCOPE]},
        },
    )
    await seed_policy(sessions, config, allow=True, preferred=['b', 'unknown-manager'])
    async with sessions() as session:
        repo = SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=60),
            policy_config=config,
            manager_lock_limit=1,
        )
        assigned = await repo.claim_and_assign_batch(shard_ids=list(range(8)), now=utcnow())
        assert assigned[0].manager_id == 'b'


async def test_old_revision_cannot_replace_new_policy_and_demand_does_not_invalidate_computation(sessions):
    config = policy_config()
    key, claim = await seed_policy(sessions, config, allow=False)
    async with sessions() as session:
        repo = PolicyRepository(session, shard_count=8)
        row = await session.get(PolicyScopeRow, key)
        version = row.version
        await repo.demand([key], now=utcnow())
        await session.refresh(row)
        assert row.version == version
        newer = policy_config(revision=2)
        await repo.ensure_scopes(newer, now=utcnow())
        await repo.ensure_scopes(config, now=utcnow())
        await session.refresh(row)
        assert row.config_revision == 2
        assert row.snapshot_json is None
        assert not await repo.publish(
            claim,
            owner_id='publisher',
            binding=config.bindings['tide'],
            result=None,
            observation=None,
            error='policy_timeout',
            evaluation_id='late',
            now=utcnow(),
            delay=1,
        )


async def test_executor_demand_computes_once_and_reuses_persisted_state(sessions):
    config = policy_config()
    executor = PolicyExecutor(session_factory=sessions, config=config, shard_count=8, owner_id='test-executor')
    try:
        await executor.initialize()
        async with sessions() as session:
            await PolicyRepository(session, shard_count=8).demand(
                [policy_scope_key('scheduler', 'tide', SCOPE)], now=utcnow()
            )
            await session.commit()
        for _ in range(150):
            await executor.tick_once()
            async with sessions() as session:
                row = await session.get(PolicyScopeRow, policy_scope_key('scheduler', 'tide', SCOPE))
                if row.snapshot_json:
                    break
            await asyncio.sleep(0.05)
        assert row.snapshot_json['payload']['allow'] is True
        assert row.state_json == {'calls': 1}
    finally:
        await executor.close()


@pytest.mark.parametrize(('field', 'value'), [('binding', 'unknown'), ('scopes', ['unknown'])])
def test_submission_cannot_select_unregistered_policy_scope(field, value):
    with pytest.raises(DomainValidationError):
        resolve_selection(
            policy_config(),
            {'binding': 'tide', 'scopes': [SCOPE], field: value},
            environment_id=ENV,
            owner_principal_id='legacy',
            spec_type='nexus_single',
        )


def test_default_policy_cannot_be_bypassed_and_orchestration_is_exempt():
    config = policy_config()
    values = config.model_dump()
    values['defaults'][ENV] = {'binding': 'tide', 'scopes': [SCOPE]}
    config = PolicyExecutorConfig.model_validate(values)
    assert (
        resolve_selection(
            config, None, environment_id=ENV, owner_principal_id='legacy', spec_type='nexus_single'
        ).binding
        == 'tide'
    )
    assert (
        resolve_selection(config, None, environment_id=ENV, owner_principal_id='legacy', spec_type='nexus_batch')
        is None
    )


@pytest.mark.parametrize(
    ('mode', 'available', 'expected'), [('enforce', False, False), ('shadow', False, True), ('disabled', False, True)]
)
async def test_unavailable_shadow_and_disabled_semantics(sessions, mode, available, expected):
    from nexus.distributed.policies.models import PolicySelection
    from nexus.distributed.policies.scheduler import consume_snapshot

    config = policy_config(mode=mode)
    effect = consume_snapshot(PolicySelection(binding='tide', scopes=[SCOPE]), config, {}, now=utcnow())
    assert effect.allow is expected


async def test_failed_refresh_does_not_extend_snapshot_or_clear_known_block(sessions):
    from nexus.distributed.policies.models import PolicySelection
    from nexus.distributed.policies.scheduler import consume_snapshot

    config = policy_config(on_unavailable='default')
    key, _ = await seed_policy(sessions, config, allow=False)
    async with sessions() as session:
        repo = PolicyRepository(session, shard_count=8)
        row = await session.get(PolicyScopeRow, key)
        original_expiry = row.valid_until
        claim = (await repo.due(list(range(8)), now=utcnow() + timedelta(seconds=2), idle_seconds=60, limit=10))[0]
        assert await repo.publish(
            claim,
            owner_id='publisher',
            binding=config.bindings['tide'],
            result=None,
            observation=None,
            error='policy_timeout',
            evaluation_id='failed-refresh',
            now=utcnow(),
            delay=1,
        )
        await session.refresh(row)
        assert row.valid_until == original_expiry
        assert row.snapshot_json['payload']['allow'] is False
        effect = consume_snapshot(
            PolicySelection(binding='tide', scopes=[SCOPE]),
            config,
            {key: row},
            now=original_expiry + timedelta(seconds=1),
        )
        assert effect.allow is False


async def test_lost_scope_lease_rejects_late_worker_result(sessions):
    config = policy_config()
    key, _ = await seed_policy(sessions, config, allow=True)
    async with sessions() as session:
        repo = PolicyRepository(session, shard_count=8)
        claim = (await repo.due(list(range(8)), now=utcnow() + timedelta(seconds=2), idle_seconds=60, limit=10))[0]
        await ShardLeaseRepository(session, shard_count=8).release(
            owner_id='publisher',
            shard_group=POLICY_SHARD_GROUP,
            shard_ids=list(range(8)),
            now=utcnow(),
        )
        assert not await repo.publish(
            claim,
            owner_id='publisher',
            binding=config.bindings['tide'],
            result=None,
            observation=None,
            error='late',
            evaluation_id='late-refresh',
            now=utcnow(),
            delay=1,
        )
        assert (await session.get(PolicyScopeRow, key)).last_error is None


def test_policy_migration_preserves_legacy_data_and_is_restartable(tmp_path):
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import create_engine, inspect

    from nexus.distributed.persistence.orm import Base

    migration = importlib.import_module('nexus.distributed.persistence.migrations.versions.0035_add_policy_scopes')
    engine = create_engine(f'sqlite:///{tmp_path}/migration.db')
    with engine.begin() as connection:
        Base.metadata.create_all(connection)
        op = Operations(MigrationContext.configure(connection))
        migration.downgrade(op)
        migration.upgrade(op)
        migration.upgrade(op)
        assert 'distributed_policy_scopes' in inspect(connection).get_table_names()
        assert 'executions' in inspect(connection).get_table_names()
    engine.dispose()


async def test_policy_explanation_is_scoped_to_authorized_execution(sessions):
    from aiohttp.test_utils import TestClient

    from nexus.distributed.config import DistributedConfig
    from nexus.distributed.gateways.control.app import create_control_app

    policy = policy_config()
    await _create_execution(
        sessions,
        execution_id='explained',
        placement={
            'environment': ENV,
            'pool': 'default',
            'scheduler_policy': {'binding': 'tide', 'scopes': [SCOPE]},
        },
    )
    await seed_policy(sessions, policy, allow=False)
    app = create_control_app(
        session_factory=sessions, config=DistributedConfig(auth_mode='disabled', policy_executor=policy)
    )
    async with TestClient(TestServer(app)) as client:
        response = await client.get('/v1/executions/explained/scheduler-policy')
        assert response.status == 200
        payload = await response.json()
        assert payload['allow'] is False
        assert payload['reason_code'] == 'quota_exhausted'
        assert payload['scopes'][0]['snapshot']['payload']['allow'] is False
        missing = await client.get('/v1/executions/not-owned/scheduler-policy')
        assert missing.status == 404


@pytest.mark.parametrize('resume_via', ['allow', 'shadow'])
async def test_policy_release_clears_policy_deferred_for_unassigned_tail_and_projects_queued(sessions, resume_via):
    from nexus.distributed.repositories.execution import ExecutionRepository
    from nexus.distributed.repositories.execution_phase import ExecutionPhaseRepository

    config = policy_config()
    await _register_manager(sessions, manager_id='mgr-1')
    for index in range(2):
        await _create_execution(
            sessions,
            execution_id=f'deferred-{index}',
            placement={
                'environment': ENV,
                'pool': 'default',
                'scheduler_policy': {'binding': 'tide', 'scopes': [SCOPE]},
            },
        )
    await seed_policy(sessions, config, allow=False)
    t0 = utcnow()
    async with sessions() as session:
        scheduler = SchedulerRepository(
            session, shard_count=8, manager_healthy_ttl=timedelta(seconds=60), policy_config=config
        )
        assert await scheduler.claim_and_assign_batch(shard_ids=list(range(8)), now=t0) == []
        await session.commit()
        for index in range(2):
            row = await session.get(ExecutionRow, f'deferred-{index}')
            assert row is not None
            assert row.last_schedule_error == 'policy_deferred:quota_exhausted'
            record = await ExecutionRepository(session, shard_count=8).get_execution(f'deferred-{index}')
            assert record is not None
            assert await ExecutionPhaseRepository(session).project(record) == 'waiting_policy'

    if resume_via == 'allow':
        await seed_policy(sessions, config, allow=True)
        next_config = config
    else:
        next_config = policy_config(mode='shadow')
        await seed_policy(sessions, next_config, allow=False)

    async with sessions() as session:
        scheduler = SchedulerRepository(
            session, shard_count=8, manager_healthy_ttl=timedelta(seconds=60), policy_config=next_config
        )
        assigned = await scheduler.claim_and_assign_batch(
            shard_ids=list(range(8)), now=t0 + timedelta(seconds=2), max_assignments=1
        )
        assert len(assigned) == 1
        await session.commit()
        unassigned_id = 'deferred-1' if assigned[0].execution_id == 'deferred-0' else 'deferred-0'
        row = await session.get(ExecutionRow, unassigned_id)
        assert row is not None
        assert row.status == 'queued'
        assert row.last_schedule_error is None
        record = await ExecutionRepository(session, shard_count=8).get_execution(unassigned_id)
        assert record is not None
        assert await ExecutionPhaseRepository(session).project(record) == 'queued'


async def test_batch_submission_inherits_policy_only_to_children(sessions):
    from nexus.distributed.config import DistributedConfig
    from nexus.distributed.repositories.submission import SubmissionRepository
    from nexus.distributed.submission.specs import BatchSubmission
    from tests.unit.distributed.submission.test_typed_submission_repository import _seed_environment

    policy = policy_config()
    async with sessions() as session:
        await _seed_environment(
            session,
            environment_id=ENV,
            allocation_type='dynamic',
            provider_type='kubernetes',
            environment_limits={},
            grant_limits={},
        )
        await session.commit()
        config = DistributedConfig(policy_executor=policy, environment_enabled_providers=['kubernetes'])
        submission = BatchSubmission.model_validate({
            'idempotency_key': 'policy-child-inheritance',
            'environment_id': ENV,
            'code': {'mode': 'config_only'},
            'scheduler_policy': {'binding': 'tide', 'scopes': [SCOPE]},
            'resources': {
                'batch_manager': {'cpu_cores': 1, 'memory_gib': 2},
                'managers': {'cpu_cores': 1, 'memory_gib': 2, 'count': 1, 'slots': 2},
            },
            'generation': {
                'mode': 'remote_materialize',
                'source': {
                    'type': 'compose',
                    'compose': {
                        'parts': [{'kind': 'set', 'value': 'task_name=identity', 'source': 'test'}],
                    },
                },
            },
        })
        result = await SubmissionRepository(session, config=config).create(
            submission,
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=utcnow(),
        )
        parent = await session.get(ExecutionRow, result.execution_id)
        assert 'scheduler_policy' not in parent.placement_json
        assert parent.spec_json['batch']['child_template']['scheduler_policy'] == {'binding': 'tide', 'scopes': [SCOPE]}
        await session.commit()

    await seed_policy(sessions, policy, allow=False)
    from aiohttp.test_utils import TestClient

    from nexus.distributed.gateways.control.app import create_control_app

    app = create_control_app(
        session_factory=sessions,
        config=DistributedConfig(auth_mode='disabled', policy_executor=policy),
    )
    async with TestClient(TestServer(app)) as client:
        response = await client.get(f'/v1/executions/{result.execution_id}/scheduler-policy')
        assert response.status == 200
        payload = await response.json()
        assert payload['enabled'] is True
        assert payload['allow'] is False
        assert payload['reason_code'] == 'quota_exhausted'
        assert payload['binding'] == 'tide'
        assert payload['next_schedule_at'] is None


async def test_scheduler_shutdown_stops_policy_loop_and_reaps_workers(sessions):
    from nexus.distributed.config import DistributedConfig
    from nexus.distributed.control_plane.scheduler import SchedulerService

    service = SchedulerService(
        session_factory=sessions,
        config=DistributedConfig(shard_count=8, policy_executor=policy_config()),
        owner_id='policy-shutdown',
    )
    running = asyncio.create_task(service.run_forever())
    try:
        await asyncio.sleep(0.2)
        service.stop()
        await asyncio.wait_for(running, timeout=10)
        assert all(worker.process is None for worker in service._policy_executor.pool.workers)
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)


async def test_slow_plugin_initialization_does_not_stop_unbound_scheduling(sessions):
    from nexus.distributed.config import DistributedConfig
    from nexus.distributed.control_plane.scheduler import SchedulerService

    config = policy_config(policy='test_slow_constructor', timeout_seconds=0.3)
    await _register_manager(sessions, manager_id='unbound-manager', capacity_total=2)
    await _create_execution(
        sessions,
        execution_id='slow-gated',
        placement={
            'environment': ENV,
            'pool': 'default',
            'scheduler_policy': {'binding': 'tide', 'scopes': [SCOPE]},
        },
    )
    await _create_execution(sessions, execution_id='unbound-independent')
    service = SchedulerService(
        session_factory=sessions,
        config=DistributedConfig(shard_count=8, policy_executor=config),
        owner_id='slow-plugin',
    )
    loop = asyncio.create_task(service.run_forever())
    try:
        for _ in range(200):
            async with sessions() as session:
                row = await session.get(PolicyScopeRow, policy_scope_key('scheduler', 'tide', SCOPE))
                if row is not None and row.last_error == 'policy_timeout':
                    break
            if loop.done():
                await loop
                pytest.fail('scheduler stopped unexpectedly')
            await asyncio.sleep(0.05)
        assert row.last_error == 'policy_timeout'
        async with sessions() as session:
            assert (await session.get(ExecutionRow, 'unbound-independent')).status == 'assigned'
            assert (await session.get(ExecutionRow, 'slow-gated')).status == 'queued'
        assert not loop.done()
    finally:
        service.stop()
        await asyncio.wait_for(loop, timeout=10)


async def test_capacity_worker_can_query_http_with_multithreaded_parent(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '32')

    async def capacity(request):
        assert await request.json() == {'model': 'hy3-offline', 'api_key': 'test-capacity-key'}
        return web.json_response({'key_remaining_raw_qpm': 10, 'scaling_advice': {'factor': 0.5, 'hard_brake': False}})

    app = web.Application()
    app.router.add_post('/capacity', capacity)
    pool = PolicyWorkerPool(size=1)
    try:
        async with TestServer(app) as server:
            result = await pool.run(
                {
                    'imports': [],
                    'kind': 'scheduler',
                    'policy': 'infer_proxy_tide',
                    'source': 'infer_proxy_capacity',
                    'source_config': {
                        'model': 'hy3-offline',
                        'api_key': 'test-capacity-key',
                        'url': str(server.make_url('/capacity')),
                    },
                    'scope_id': 'real-capacity-adapter',
                    'params': {},
                    'previous_state': {},
                    'max_age_seconds': 30,
                },
                timeout=5,
            )
            assert result['observation']['values'] == {'remaining': 10, 'factor': 0.5, 'hard_brake': False}
            assert result['evaluation']['decision']['allow'] is True
    finally:
        await pool.close()
