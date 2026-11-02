"""Unit tests for migration 0036 and RequestBudgetRepository transactional lease rules."""

from __future__ import annotations

import importlib
import math

from datetime import datetime, timedelta

import pytest

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.errors import DomainValidationError
from nexus.distributed.execution.models import AssignmentStatus
from nexus.distributed.persistence.orm import (
    AssignmentRow,
    Base,
    ExecutionRow,
    InstanceManagerRow,
    PolicyScopeRow,
    RequestBudgetLeaseRow,
    create_schema,
)
from nexus.distributed.policies.collections.infer_proxy_request_rate import (
    REQUEST_CAPACITY_SOURCE,
    REQUEST_RATE_POLICY,
)
from nexus.distributed.policies.models import (
    PolicyExecutorConfig,
    RequestBudgetConfig,
    RequestPolicySelection,
)
from nexus.distributed.policies.request_rate import request_budget_key
from nexus.distributed.repositories.request_budget import (
    BudgetDemand,
    RequestBudgetRepository,
    compute_target_shares,
)


NOW = datetime(2026, 10, 19, 12, 0, 0)
MIGRATION_MODULE = 'nexus.distributed.persistence.migrations.versions.0036_add_request_budget_leases'


def _policy_config(*, mode: str = 'enforce', revision: int = 1) -> PolicyExecutorConfig:
    return PolicyExecutorConfig.model_validate({
        'bindings': {
            'team-a-hy3-rate': {
                'kind': 'request_rate',
                'policy': REQUEST_RATE_POLICY,
                'source': REQUEST_CAPACITY_SOURCE,
                'revision': revision,
                'mode': mode,
                'on_unavailable': 'defer',
                'environments': ['env-a'],
                'principals': ['alice'],
                'scopes': {
                    'capacity': {
                        'source_config': {
                            'model': 'hy3-offline',
                            'api_key': 'test-key',
                        }
                    }
                },
                'params': {'quota_qpm': 600.0},
            }
        },
        'request_defaults': {
            'env-a': {'binding': 'team-a-hy3-rate', 'scope': 'capacity'},
        },
    })


def _register_manager(
    manager_id: str,
    session_id: str,
    *,
    binding: str | None = 'team-a-hy3-rate',
    scope: str | None = 'capacity',
    environment_id: str = 'env-a',
    owner_principal_id: str = 'alice',
) -> InstanceManagerRow:
    return InstanceManagerRow(
        manager_id=manager_id,
        session_id=session_id,
        endpoint_url=f'http://127.0.0.1/{manager_id}',
        labels_json={},
        draining=False,
        owner_principal_id=owner_principal_id,
        environment_id=environment_id,
        pool_id='default',
        allocation_id=None,
        workload='execution',
        allocation_spec_hash=None,
        request_policy_binding=binding,
        request_policy_scope=scope,
        supported_spec_versions_json={'nexus_single': [1]},
        supported_modes_json=['job'],
        supported_spec_types_json=['nexus_single'],
        supported_instance_modes_json=['asyncio'],
        capacity_total=4,
        observed_capacity_used=0,
        last_heartbeat_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )


def _scope_snapshot_row(
    config: PolicyExecutorConfig,
    selection: RequestPolicySelection,
    *,
    allow: bool = True,
    rate_per_minute: float = 600.0,
    valid_until: datetime = NOW + timedelta(seconds=120),
    snapshot_version: int = 1,
    reason_code: str = 'adopted',
) -> PolicyScopeRow:
    binding = config.bindings[selection.binding]
    digest = binding.fingerprint(config.imports)
    return PolicyScopeRow(
        scope_key=request_budget_key(selection),
        kind='request_rate',
        binding_id=selection.binding,
        scope_id=selection.scope,
        shard_id=0,
        config_revision=binding.revision,
        config_hash=digest,
        state_json={},
        snapshot_json={
            'snapshot_version': snapshot_version,
            'reason_code': reason_code,
            'payload': {'allow': allow, 'rate_per_minute': rate_per_minute},
        },
        valid_until=valid_until,
        next_evaluate_at=NOW + timedelta(seconds=60),
        last_demand_at=NOW,
        version=snapshot_version,
        failures=0,
        last_error=None,
        updated_at=NOW,
    )


def test_migration_0036_is_idempotent_and_reversible(tmp_path):
    module = importlib.import_module(MIGRATION_MODULE)
    engine = create_engine(f'sqlite:///{tmp_path / "m0036.db"}')
    try:
        with engine.begin() as conn:
            Base.metadata.create_all(conn)
            op = Operations(MigrationContext.configure(conn))
            module.downgrade(op)
            module.downgrade(op)
            tables = set(inspect(conn).get_table_names())
            assert 'distributed_request_budgets' not in tables
            assert 'distributed_request_budget_leases' not in tables
            mgr_cols = {c['name'] for c in inspect(conn).get_columns('instance_managers')}
            assert 'request_policy_binding' not in mgr_cols

            module.upgrade(op)
            module.upgrade(op)
            tables = set(inspect(conn).get_table_names())
            assert 'distributed_request_budgets' in tables
            assert 'distributed_request_budget_leases' in tables
            mgr_cols = {c['name'] for c in inspect(conn).get_columns('instance_managers')}
            assert {'request_policy_binding', 'request_policy_scope'} <= mgr_cols
    finally:
        engine.dispose()


def test_compute_target_shares_water_filling():
    # Three unbounded demanders share 600 equally -> 200 each.
    shares = compute_target_shares(
        600.0,
        {('m1', 's1'): None, ('m2', 's2'): None, ('m3', 's3'): None},
    )
    assert all(math.isclose(v, 200.0) for v in shares.values())

    # m1 asks for at most 100, m2 and m3 unbounded -> m1=100, m2=250, m3=250.
    bounded = compute_target_shares(
        600.0,
        {('m1', 's1'): 100.0, ('m2', 's2'): None, ('m3', 's3'): None},
    )
    assert math.isclose(bounded[('m1', 's1')], 100.0)
    assert math.isclose(bounded[('m2', 's2')], 250.0)
    assert math.isclose(bounded[('m3', 's3')], 250.0)


async def test_coordinate_lease_renewal_idempotency_and_multi_manager_rebalance(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "budget.db"}')
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    config = _policy_config()
    selection = RequestPolicySelection(binding='team-a-hy3-rate', scope='capacity')
    budget_key = request_budget_key(selection)
    budget_cfg = RequestBudgetConfig(lease_seconds=10, renew_before_seconds=4)

    try:
        async with sessions() as session:
            session.add(_register_manager('mgr-1', 'sess-1'))
            session.add(_register_manager('mgr-2', 'sess-2'))
            session.add(_scope_snapshot_row(config, selection, rate_per_minute=600.0))
            await session.commit()

        # 1. Manager 1 requests initial lease at t=0 -> gets full 600 QPM until t=10s.
        async with sessions() as session:
            repo = RequestBudgetRepository(session)
            r1 = await repo.coordinate_lease(
                BudgetDemand(budget_key=budget_key, manager_id='mgr-1', manager_session_id='sess-1', request_seq=1),
                selection=selection,
                config=config,
                budget_config=budget_cfg,
                now=NOW,
            )
            await session.commit()
        assert r1.status == 'granted'
        assert r1.lease is not None
        assert math.isclose(r1.lease.granted_qpm, 600.0)
        assert r1.lease.lease_version == 1
        assert r1.lease.valid_until == NOW + timedelta(seconds=10)
        lease_1_id = r1.lease.lease_id

        # 2. Idempotent replay of request_seq=1 at t=2s returns same lease and absolute valid_until,
        #    but updates server_now to t=2s.
        t2 = NOW + timedelta(seconds=2)
        async with sessions() as session:
            repo = RequestBudgetRepository(session)
            replay = await repo.coordinate_lease(
                BudgetDemand(budget_key=budget_key, manager_id='mgr-1', manager_session_id='sess-1', request_seq=1),
                selection=selection,
                config=config,
                budget_config=budget_cfg,
                now=t2,
            )
            await session.commit()
        assert replay.status == 'granted'
        assert replay.lease == r1.lease
        assert replay.server_now == t2

        # Same request_seq with different requested_qpm is rejected; stale seq is rejected.
        async with sessions() as session:
            repo = RequestBudgetRepository(session)
            with pytest.raises(DomainValidationError, match='request_budget_seq_conflict'):
                await repo.coordinate_lease(
                    BudgetDemand(
                        budget_key=budget_key,
                        manager_id='mgr-1',
                        manager_session_id='sess-1',
                        request_seq=1,
                        requested_qpm=100.0,
                    ),
                    selection=selection,
                    config=config,
                    budget_config=budget_cfg,
                    now=t2,
                )

        # 3. Same-share renewal at t=6s (no other demanders) keeps lease_id, increments lease_version to 2,
        #    and extends valid_until to t=16s.
        t6 = NOW + timedelta(seconds=6)
        async with sessions() as session:
            repo = RequestBudgetRepository(session)
            r1_renew = await repo.coordinate_lease(
                BudgetDemand(budget_key=budget_key, manager_id='mgr-1', manager_session_id='sess-1', request_seq=2),
                selection=selection,
                config=config,
                budget_config=budget_cfg,
                now=t6,
            )
            await session.commit()
        assert r1_renew.status == 'granted'
        assert r1_renew.lease is not None
        assert r1_renew.lease.lease_id == lease_1_id
        assert r1_renew.lease.lease_version == 2
        assert r1_renew.lease.valid_until == t6 + timedelta(seconds=10)  # NOW + 16s

        # 4. Manager 2 arrives at t=8s: registers demand, sees all 600 QPM committed to Manager 1 until NOW+16s,
        #    so gets status='wait' (no over-allocation).
        t8 = NOW + timedelta(seconds=8)
        async with sessions() as session:
            repo = RequestBudgetRepository(session)
            r2_wait = await repo.coordinate_lease(
                BudgetDemand(budget_key=budget_key, manager_id='mgr-2', manager_session_id='sess-2', request_seq=1),
                selection=selection,
                config=config,
                budget_config=budget_cfg,
                now=t8,
            )
            await session.commit()
        assert r2_wait.status == 'wait'
        assert r2_wait.lease is None
        assert r2_wait.reason_code == 'budget_occupied'

        # 5. At t=12s, Manager 1 tries to renew early, but target share is now 300 QPM (< 600 QPM).
        #    Manager 1 must NOT extend its 600 QPM lease; it gets 'lease_rebalance_wait'.
        t12 = NOW + timedelta(seconds=12)
        async with sessions() as session:
            repo = RequestBudgetRepository(session)
            r1_rebalance = await repo.coordinate_lease(
                BudgetDemand(budget_key=budget_key, manager_id='mgr-1', manager_session_id='sess-1', request_seq=3),
                selection=selection,
                config=config,
                budget_config=budget_cfg,
                now=t12,
            )
            await session.commit()
        assert r1_rebalance.status == 'wait'
        assert r1_rebalance.reason_code == 'lease_rebalance_wait'

        # 6. At t=16s (Manager 1's old 600 QPM lease has expired), Manager 1 and Manager 2 each get 300 QPM!
        t16 = NOW + timedelta(seconds=16)
        async with sessions() as session:
            repo = RequestBudgetRepository(session)
            r1_new = await repo.coordinate_lease(
                BudgetDemand(budget_key=budget_key, manager_id='mgr-1', manager_session_id='sess-1', request_seq=4),
                selection=selection,
                config=config,
                budget_config=budget_cfg,
                now=t16,
            )
            r2_new = await repo.coordinate_lease(
                BudgetDemand(budget_key=budget_key, manager_id='mgr-2', manager_session_id='sess-2', request_seq=2),
                selection=selection,
                config=config,
                budget_config=budget_cfg,
                now=t16,
            )
            await session.commit()
        assert r1_new.status == 'granted'
        assert r1_new.lease is not None
        assert r2_new.status == 'granted'
        assert r2_new.lease is not None
        assert r1_new.lease.lease_id != lease_1_id
        assert math.isclose(r1_new.lease.granted_qpm, 300.0)
        assert math.isclose(r2_new.lease.granted_qpm, 300.0)
        assert math.isclose(r1_new.lease.granted_qpm + r2_new.lease.granted_qpm, 600.0)
    finally:
        await engine.dispose()


async def test_coordinate_lease_respects_restarted_session_and_budget_drop_and_modes(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "modes.db"}')
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    config = _policy_config()
    selection = RequestPolicySelection(binding='team-a-hy3-rate', scope='capacity')
    budget_key = request_budget_key(selection)
    budget_cfg = RequestBudgetConfig(lease_seconds=10, renew_before_seconds=4)

    try:
        async with sessions() as session:
            session.add(_register_manager('mgr-1', 'sess-1'))
            session.add(_scope_snapshot_row(config, selection, rate_per_minute=600.0))
            await session.commit()

        # Grant 600 QPM to (mgr-1, sess-1) until NOW+10s.
        async with sessions() as session:
            repo = RequestBudgetRepository(session)
            first = await repo.coordinate_lease(
                BudgetDemand(budget_key=budget_key, manager_id='mgr-1', manager_session_id='sess-1', request_seq=1),
                selection=selection,
                config=config,
                budget_config=budget_cfg,
                now=NOW,
            )
            await session.commit()
        assert first.status == 'granted'

        # Manager restarts with sess-2 at t=3s: old sess-1 lease is still unexpired until NOW+10s,
        # so sess-2 cannot double-spend the 600 QPM before NOW+10s!
        t3 = NOW + timedelta(seconds=3)
        async with sessions() as session:
            mgr = await session.get(InstanceManagerRow, 'mgr-1')
            assert mgr is not None
            mgr.session_id = 'sess-2'
            repo = RequestBudgetRepository(session)
            restarted = await repo.coordinate_lease(
                BudgetDemand(budget_key=budget_key, manager_id='mgr-1', manager_session_id='sess-2', request_seq=1),
                selection=selection,
                config=config,
                budget_config=budget_cfg,
                now=t3,
            )
            await session.commit()
        assert restarted.status == 'wait'
        assert restarted.reason_code == 'budget_occupied'

        # At t=10s, sess-1 lease expires, so sess-2 gets the 600 QPM lease.
        t10 = NOW + timedelta(seconds=10)
        async with sessions() as session:
            repo = RequestBudgetRepository(session)
            restarted_granted = await repo.coordinate_lease(
                BudgetDemand(budget_key=budget_key, manager_id='mgr-1', manager_session_id='sess-2', request_seq=2),
                selection=selection,
                config=config,
                budget_config=budget_cfg,
                now=t10,
            )
            await session.commit()
        assert restarted_granted.status == 'granted'
        assert restarted_granted.lease is not None
        assert math.isclose(restarted_granted.lease.granted_qpm, 600.0)

        # Switching to shadow mode returns status='inactive' while preserving old lease row until expiry.
        shadow_cfg = _policy_config(mode='shadow', revision=2)
        t12 = NOW + timedelta(seconds=12)
        async with sessions() as session:
            repo = RequestBudgetRepository(session)
            inactive = await repo.coordinate_lease(
                BudgetDemand(budget_key=budget_key, manager_id='mgr-1', manager_session_id='sess-2', request_seq=3),
                selection=selection,
                config=shadow_cfg,
                budget_config=budget_cfg,
                now=t12,
            )
            lease_row = await session.scalar(
                select(RequestBudgetLeaseRow).where(
                    RequestBudgetLeaseRow.manager_id == 'mgr-1',
                    RequestBudgetLeaseRow.manager_session_id == 'sess-2',
                )
            )
            await session.commit()
        assert inactive.status == 'inactive'
        assert inactive.mode == 'shadow'
        assert inactive.lease is None
        assert lease_row is not None
        assert lease_row.valid_until == t10 + timedelta(seconds=10)
    finally:
        await engine.dispose()


async def test_coordinate_lease_rejects_unauthorized_assigned_execution(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "authz.db"}')
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    config = _policy_config()
    selection = RequestPolicySelection(binding='team-a-hy3-rate', scope='capacity')
    budget_key = request_budget_key(selection)

    try:
        async with sessions() as session:
            session.add(_register_manager('mgr-1', 'sess-1'))
            session.add(_scope_snapshot_row(config, selection, rate_per_minute=600.0))
            # Add an active execution owned by 'bob' (not in principals=['alice']) assigned to mgr-1.
            session.add(
                ExecutionRow(
                    execution_id='exe-bob',
                    shard_id=0,
                    mode='job',
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    status='running',
                    route_status='none',
                    spec_json={'spec_version': 1},
                    placement_json={'request_policy': {'binding': 'team-a-hy3-rate', 'scope': 'capacity'}},
                    owner_principal_id='bob',
                    environment_id='env-a',
                    pool_id='default',
                    workload='execution',
                    slot_cost=1,
                    max_attempts=1,
                    next_attempt_seq=2,
                    next_event_seq=1,
                    priority=0,
                    next_schedule_at=NOW,
                    queue_entered_at=NOW,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
            session.add(
                AssignmentRow(
                    assignment_id='asn-bob',
                    execution_id='exe-bob',
                    owner_principal_id='bob',
                    environment_id='env-a',
                    pool_id='default',
                    workload='execution',
                    attempt_id='exe-bob-attempt-1',
                    attempt_seq=1,
                    manager_id='mgr-1',
                    manager_session_id='sess-1',
                    status=AssignmentStatus.ACTIVE,
                    slot_cost=1,
                    endpoint_ready=True,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
            await session.commit()

        async with sessions() as session:
            repo = RequestBudgetRepository(session)
            with pytest.raises(DomainValidationError, match='request_policy_not_permitted'):
                await repo.coordinate_lease(
                    BudgetDemand(budget_key=budget_key, manager_id='mgr-1', manager_session_id='sess-1', request_seq=1),
                    selection=selection,
                    config=config,
                    now=NOW,
                )
    finally:
        await engine.dispose()


async def test_under_allocated_manager_renews_existing_partial_share_while_peer_holds_oversized_lease(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "partial_renew.db"}')
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    config = _policy_config()
    selection = RequestPolicySelection(binding='team-a-hy3-rate', scope='capacity')
    budget_key = request_budget_key(selection)
    budget_cfg = RequestBudgetConfig(lease_seconds=10, renew_before_seconds=4)

    try:
        async with sessions() as session:
            session.add(_register_manager('mgr-1', 'sess-1'))
            session.add(_register_manager('mgr-2', 'sess-2'))
            session.add(_scope_snapshot_row(config, selection, rate_per_minute=600.0))
            await session.commit()

        # 1. At t=0, mgr-1 asks for bounded 400 QPM -> gets 400 QPM until t=10s (200 QPM remains uncommitted).
        async with sessions() as session:
            repo = RequestBudgetRepository(session)
            r1 = await repo.coordinate_lease(
                BudgetDemand(
                    budget_key=budget_key,
                    manager_id='mgr-1',
                    manager_session_id='sess-1',
                    request_seq=1,
                    requested_qpm=400.0,
                ),
                selection=selection,
                config=config,
                budget_config=budget_cfg,
                now=NOW,
            )
            await session.commit()
        assert r1.status == 'granted'
        assert r1.lease is not None
        assert math.isclose(r1.lease.granted_qpm, 400.0)

        # 2. At t=1s, mgr-2 arrives unbounded (target_qpm = 300 QPM, since mgr-1+mgr-2 equally share 600 QPM -> 300 each,
        #    while mgr-1 still holds 400 QPM until t=10s). mgr-2 gets the remaining 200 QPM (< target 300 QPM)!
        t1 = NOW + timedelta(seconds=1)
        async with sessions() as session:
            repo = RequestBudgetRepository(session)
            r2 = await repo.coordinate_lease(
                BudgetDemand(
                    budget_key=budget_key,
                    manager_id='mgr-2',
                    manager_session_id='sess-2',
                    request_seq=1,
                ),
                selection=selection,
                config=config,
                budget_config=budget_cfg,
                now=t1,
            )
            await session.commit()
        assert r2.status == 'granted'
        assert r2.lease is not None
        assert math.isclose(r2.lease.granted_qpm, 200.0)
        lease_2_id = r2.lease.lease_id

        # 3. At t=7s (mgr-1's 400 QPM lease is still unexpired until t=10s, so available_qpm == 0):
        #    mgr-2 (holding 200 < target 300) renews its existing 200 QPM share without dropping into wait!
        t7 = NOW + timedelta(seconds=7)
        async with sessions() as session:
            repo = RequestBudgetRepository(session)
            r2_renew = await repo.coordinate_lease(
                BudgetDemand(
                    budget_key=budget_key,
                    manager_id='mgr-2',
                    manager_session_id='sess-2',
                    request_seq=2,
                ),
                selection=selection,
                config=config,
                budget_config=budget_cfg,
                now=t7,
            )
            await session.commit()
        assert r2_renew.status == 'granted'
        assert r2_renew.lease is not None
        assert r2_renew.lease.lease_id == lease_2_id
        assert r2_renew.lease.lease_version == 2
        assert math.isclose(r2_renew.lease.granted_qpm, 200.0)
    finally:
        await engine.dispose()
