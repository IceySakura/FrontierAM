"""Unit tests for RequestBudgetControl under controlled clocks and failure injection."""

from __future__ import annotations

import asyncio

from datetime import datetime, timedelta

import pytest

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.core.request_admission import RequestAttempt
from nexus.distributed.instance_manager.request_budget import RequestBudgetControl
from nexus.distributed.persistence.orm import (
    DistributedConfigOverrideRow,
    InstanceManagerRow,
    PolicyScopeRow,
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


BASE_WALL = datetime(2026, 10, 19, 14, 0, 0)


class ControlledClock:
    def __init__(self, *, mono: float = 1000.0, wall: datetime = BASE_WALL) -> None:
        self.mono = mono
        self.wall = wall

    def monotonic(self) -> float:
        return self.mono

    def utcnow(self) -> datetime:
        return self.wall

    def advance(self, seconds: float) -> None:
        self.mono += seconds
        self.wall = self.wall + timedelta(seconds=seconds)


def _make_config(*, mode: str = 'enforce', revision: int = 1, quota_qpm: float = 600.0) -> PolicyExecutorConfig:
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
                'params': {'quota_qpm': quota_qpm},
            }
        },
        'request_defaults': {
            'env-a': {'binding': 'team-a-hy3-rate', 'scope': 'capacity'},
        },
    })


def _manager_row(now: datetime) -> InstanceManagerRow:
    return InstanceManagerRow(
        manager_id='mgr-1',
        session_id='sess-1',
        endpoint_url='http://127.0.0.1:8091',
        labels_json={},
        draining=False,
        owner_principal_id='alice',
        environment_id='env-a',
        pool_id='default',
        allocation_id=None,
        workload='execution',
        allocation_spec_hash=None,
        request_policy_binding='team-a-hy3-rate',
        request_policy_scope='capacity',
        supported_spec_versions_json={'nexus_single': [1]},
        supported_modes_json=['job'],
        supported_spec_types_json=['nexus_single'],
        supported_instance_modes_json=['asyncio'],
        capacity_total=4,
        observed_capacity_used=0,
        last_heartbeat_at=now,
        created_at=now,
        updated_at=now,
    )


def _scope_row(
    config: PolicyExecutorConfig,
    selection: RequestPolicySelection,
    *,
    now: datetime,
    allow: bool = True,
    rate_per_minute: float = 60.0,
    valid_seconds: float = 120.0,
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
        valid_until=now + timedelta(seconds=valid_seconds),
        next_evaluate_at=now + timedelta(seconds=60),
        last_demand_at=datetime(1970, 1, 1),
        version=snapshot_version,
        failures=0,
        last_error=None,
        updated_at=now,
    )


async def test_enforce_waits_for_initial_lease_and_registers_policy_demand(tmp_path):
    clock = ControlledClock()
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "ctrl1.db"}')
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    config = _make_config(mode='enforce')
    selection = RequestPolicySelection(binding='team-a-hy3-rate', scope='capacity')
    key = request_budget_key(selection)

    try:
        async with sessions() as session:
            session.add(_manager_row(clock.utcnow()))
            # Scope exists without snapshot yet (simulating cold start before PolicyExecutor evaluates).
            binding = config.bindings[selection.binding]
            session.add(
                PolicyScopeRow(
                    scope_key=key,
                    kind='request_rate',
                    binding_id=selection.binding,
                    scope_id=selection.scope,
                    shard_id=0,
                    config_revision=1,
                    config_hash=binding.fingerprint(config.imports),
                    state_json={},
                    snapshot_json=None,
                    valid_until=None,
                    next_evaluate_at=clock.utcnow(),
                    last_demand_at=datetime(1970, 1, 1),
                    version=1,
                    failures=0,
                    last_error=None,
                    updated_at=clock.utcnow(),
                )
            )
            await session.commit()

        sleep_gate = asyncio.Event()

        async def fake_sleep(seconds: float) -> None:
            await sleep_gate.wait()

        ctrl = RequestBudgetControl(
            session_factory=sessions,
            manager_id='mgr-1',
            manager_session_id='sess-1',
            environment_id='env-a',
            owner_principal_id='alice',
            selection=selection,
            policy_config=config,
            budget_config=RequestBudgetConfig(lease_seconds=10, renew_before_seconds=4, safety_margin_seconds=0.2),
            monotonic_clock=clock.monotonic,
            wall_clock=clock.utcnow,
            sleep_func=fake_sleep,
        )

        # Start an acquire() call in background: must wait (no full-speed window before first lease).
        acquire_task = asyncio.create_task(
            ctrl.acquire(RequestAttempt(execution_id='exe-1', attempt_id='att-1', client_id='c-1'))
        )
        await asyncio.sleep(0.02)
        assert not acquire_task.done()

        # Coordinate while snapshot is still missing -> returns unavailable, but registers Policy demand!
        res1 = await ctrl.coordinate_once()
        assert res1 is not None
        assert res1.status == 'unavailable'
        async with sessions() as session:
            scope = await session.get(PolicyScopeRow, key)
            assert scope is not None
            assert scope.last_demand_at == clock.utcnow()

        assert not acquire_task.done()

        # Publish a 60 QPM snapshot (1 req / 1.0s) and coordinate again at t=0.
        async with sessions() as session:
            scope = await session.get(PolicyScopeRow, key)
            assert scope is not None
            scope.snapshot_json = {
                'snapshot_version': 2,
                'reason_code': 'adopted',
                'payload': {'allow': True, 'rate_per_minute': 60.0},
            }
            scope.valid_until = clock.utcnow() + timedelta(seconds=120)
            await session.commit()

        res2 = await ctrl.coordinate_once()
        assert res2 is not None
        assert res2.status == 'granted'
        await asyncio.sleep(0.02)
        # Initial free permit is 0, so at t=0 (0s elapsed since lease start) acquire_task is still waiting 1.0s.
        assert not acquire_task.done()

        # Advance 1.0s -> first permit is issued!
        clock.advance(1.0)
        sleep_gate.set()
        permit = await asyncio.wait_for(acquire_task, timeout=1.0)
        assert permit.session_id == 'sess-1'
        assert permit.lease_id is not None
        assert permit.is_valid(now=clock.monotonic())
        await ctrl.close()
    finally:
        await engine.dispose()


async def test_low_qpm_continuous_renewal_preserves_pacing_and_expired_gap_does_not_backfill(tmp_path):
    clock = ControlledClock()
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "ctrl_low_qpm.db"}')
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    config = _make_config(mode='enforce', quota_qpm=3.0)
    selection = RequestPolicySelection(binding='team-a-hy3-rate', scope='capacity')

    try:
        async with sessions() as session:
            session.add(_manager_row(clock.utcnow()))
            # 3.0 QPM -> 1 request every 20 seconds, across 10-second leases!
            session.add(_scope_row(config, selection, now=clock.utcnow(), rate_per_minute=3.0, valid_seconds=300.0))
            await session.commit()

        step_event = asyncio.Event()

        async def stepped_sleep(seconds: float) -> None:
            await step_event.wait()

        ctrl = RequestBudgetControl(
            session_factory=sessions,
            manager_id='mgr-1',
            manager_session_id='sess-1',
            environment_id='env-a',
            owner_principal_id='alice',
            selection=selection,
            policy_config=config,
            budget_config=RequestBudgetConfig(lease_seconds=10, renew_before_seconds=4, safety_margin_seconds=0.2),
            monotonic_clock=clock.monotonic,
            wall_clock=clock.utcnow,
            sleep_func=stepped_sleep,
        )

        # Start first acquire() so there is local demand, then grant initial 10s lease at t=0.
        acquire_task = asyncio.create_task(
            ctrl.acquire(RequestAttempt(execution_id='exe-1', attempt_id='att-1', client_id='c-1'))
        )
        await asyncio.sleep(0.02)
        assert not acquire_task.done()

        r1 = await ctrl.coordinate_once()
        assert r1 is not None
        assert r1.status == 'granted'
        assert r1.lease is not None
        lease_id = r1.lease.lease_id
        await asyncio.sleep(0.02)
        assert not acquire_task.done()

        # Renew continuously at t=6s, t=12s, t=18s (same lease_id, never expires locally).
        for step in (6.0, 6.0, 6.0):
            clock.advance(step)
            renewed = await ctrl.coordinate_once()
            assert renewed is not None
            assert renewed.status == 'granted'
            assert renewed.lease is not None
            assert renewed.lease.lease_id == lease_id
            step_event.set()
            await asyncio.sleep(0.02)
            step_event.clear()

        # At t=18s (< 20s), still waiting!
        assert not acquire_task.done()

        # Advance 2.0s to t=20.0s -> 20s accumulated across three renewals -> permit issued!
        clock.advance(2.0)
        step_event.set()
        permit = await asyncio.wait_for(acquire_task, timeout=1.0)
        assert permit.lease_id == lease_id

        # Now let the lease expire completely (advance 40 seconds without renewal to t=60s).
        step_event.clear()
        clock.advance(40.0)
        second_task = asyncio.create_task(
            ctrl.acquire(RequestAttempt(execution_id='exe-1', attempt_id='att-1', client_id='c-1'))
        )
        await asyncio.sleep(0.02)
        assert not second_task.done()

        # Late renewal at t=60s gets a new lease and resets pacer at t=60s (no backfilled permit for 20..60s!).
        late = await ctrl.coordinate_once()
        assert late is not None
        assert late.status == 'granted'
        await asyncio.sleep(0.02)
        assert not second_task.done()
        await ctrl.close()
        with pytest.raises(RuntimeError, match='request_budget_control_closed'):
            await second_task
    finally:
        await engine.dispose()


async def test_single_task_cancellation_and_mode_transitions(tmp_path):
    clock = ControlledClock()
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "ctrl_cancel.db"}')
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    current_config = _make_config(mode='shadow', revision=1)
    selection = RequestPolicySelection(binding='team-a-hy3-rate', scope='capacity')

    async def load_config() -> PolicyExecutorConfig:
        return current_config

    try:
        async with sessions() as session:
            session.add(_manager_row(clock.utcnow()))
            session.add(_scope_row(current_config, selection, now=clock.utcnow(), allow=False, rate_per_minute=0.0))
            await session.commit()

        ctrl = RequestBudgetControl(
            session_factory=sessions,
            manager_id='mgr-1',
            manager_session_id='sess-1',
            environment_id='env-a',
            owner_principal_id='alice',
            selection=selection,
            policy_config=current_config,
            monotonic_clock=clock.monotonic,
            wall_clock=clock.utcnow,
            config_loader=load_config,
        )

        # In shadow mode, acquire() passes immediately without lease even when allow=False in snapshot.
        p_shadow = await ctrl.acquire(RequestAttempt(execution_id='exe-1', attempt_id='att-1', client_id='c-1'))
        assert p_shadow.lease_id is None
        assert p_shadow.is_valid()

        # Switch to enforce (revision 2) with allow=False (quota exhausted).
        current_config = _make_config(mode='enforce', revision=2)
        async with sessions() as session:
            scope = await session.get(PolicyScopeRow, request_budget_key(selection))
            assert scope is not None
            scope.config_revision = 2
            scope.config_hash = current_config.bindings['team-a-hy3-rate'].fingerprint(current_config.imports)
            scope.snapshot_json = {
                'snapshot_version': 2,
                'reason_code': 'quota_exhausted',
                'payload': {'allow': False, 'rate_per_minute': 0.0},
            }
            await session.commit()

        await ctrl.coordinate_once()
        assert ctrl.mode == 'enforce'

        # Two tasks wait under enforce while quota is 0.
        t_exe1 = asyncio.create_task(
            ctrl.acquire(RequestAttempt(execution_id='exe-1', attempt_id='att-1', client_id='c-1'))
        )
        t_exe2 = asyncio.create_task(
            ctrl.acquire(RequestAttempt(execution_id='exe-2', attempt_id='att-1', client_id='c-2'))
        )
        await asyncio.sleep(0.02)
        assert not t_exe1.done()
        assert not t_exe2.done()

        # Cancel only exe-1: exe-1 raises CancelledError while exe-2 continues waiting.
        ctrl.cancel_execution('exe-1', attempt_id='att-1')
        with pytest.raises(asyncio.CancelledError):
            await t_exe1
        assert not t_exe2.done()

        # Switch to disabled (revision 3): exe-2 immediately wakes up and passes through!
        current_config = _make_config(mode='disabled', revision=3)
        await ctrl.coordinate_once()
        assert ctrl.mode == 'disabled'
        p_disabled = await asyncio.wait_for(t_exe2, timeout=1.0)
        assert p_disabled.lease_id is None

        # A stale config response with revision=1 cannot roll back revision=3.
        current_config = _make_config(mode='enforce', revision=1)
        await ctrl.coordinate_once()
        assert ctrl.mode == 'disabled'
        assert ctrl.status_snapshot().observed_config_revision == 3

        await ctrl.close()
    finally:
        await engine.dispose()


async def test_enforce_db_outage_does_not_bypass_and_shadow_db_outage_does_not_block(tmp_path):
    clock = ControlledClock()
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "ctrl_outage.db"}')
    await create_schema(engine)
    real_sessions = async_sessionmaker(engine, expire_on_commit=False)

    db_healthy = True

    def fault_injecting_sessions():
        if not db_healthy:
            raise RuntimeError('database_connection_refused')
        return real_sessions()

    async def fault_injecting_config_loader():
        if not db_healthy:
            raise RuntimeError('database_connection_refused')

    config = _make_config(mode='enforce', quota_qpm=60.0)
    selection = RequestPolicySelection(binding='team-a-hy3-rate', scope='capacity')

    try:
        async with real_sessions() as session:
            session.add(_manager_row(clock.utcnow()))
            session.add(_scope_row(config, selection, now=clock.utcnow(), rate_per_minute=60.0, valid_seconds=120.0))
            await session.commit()

        step_event = asyncio.Event()

        async def stepped_sleep(seconds: float) -> None:
            await step_event.wait()

        ctrl = RequestBudgetControl(
            session_factory=fault_injecting_sessions,  # type: ignore[arg-type]
            manager_id='mgr-1',
            manager_session_id='sess-1',
            environment_id='env-a',
            owner_principal_id='alice',
            selection=selection,
            policy_config=config,
            budget_config=RequestBudgetConfig(lease_seconds=10, renew_before_seconds=4, safety_margin_seconds=0.2),
            monotonic_clock=clock.monotonic,
            wall_clock=clock.utcnow,
            sleep_func=stepped_sleep,
            config_loader=fault_injecting_config_loader,
        )

        # 1. Start a request, coordinate a 10s lease at t=0 (valid locally until t=9.8s).
        req1 = asyncio.create_task(
            ctrl.acquire(RequestAttempt(execution_id='exe-1', attempt_id='att-1', client_id='c-1'))
        )
        await asyncio.sleep(0.02)
        res = await ctrl.coordinate_once()
        assert res is not None
        assert res.status == 'granted'

        # 2. Break the DB at t=1.0s: while the existing lease is still valid (t=1.0s < 9.8s),
        #    req1 gets its permit from local memory without touching the DB.
        db_healthy = False
        clock.advance(1.0)
        step_event.set()
        permit1 = await asyncio.wait_for(req1, timeout=1.0)
        assert permit1.lease_id is not None

        # Failed renewal during outage increments coordination_failures_total and does NOT switch mode to bypass.
        failed_res = await ctrl.coordinate_once()
        assert failed_res is None
        assert ctrl.mode == 'enforce'
        assert ctrl.status_snapshot().coordination_failures_total == 1

        # 3. Advance past local lease deadline (to t=11.0s > 9.8s) while DB is still down:
        #    new acquire() must block and wait, never bypassing enforce!
        step_event.clear()
        clock.advance(10.0)
        req2 = asyncio.create_task(
            ctrl.acquire(RequestAttempt(execution_id='exe-1', attempt_id='att-1', client_id='c-1'))
        )
        await asyncio.sleep(0.02)
        assert not req2.done()
        assert await ctrl.coordinate_once() is None
        assert not req2.done()
        await ctrl.close()
        with pytest.raises(RuntimeError, match='request_budget_control_closed'):
            await req2

        # 4. Under shadow mode, even when DB is down and snapshot is expired, acquire() never blocks.
        shadow_ctrl = RequestBudgetControl(
            session_factory=fault_injecting_sessions,  # type: ignore[arg-type]
            manager_id='mgr-1',
            manager_session_id='sess-1',
            environment_id='env-a',
            owner_principal_id='alice',
            selection=selection,
            policy_config=_make_config(mode='shadow', quota_qpm=60.0),
            monotonic_clock=clock.monotonic,
            wall_clock=clock.utcnow,
            config_loader=fault_injecting_config_loader,
        )
        assert await shadow_ctrl.coordinate_once() is None
        p_shadow = await shadow_ctrl.acquire(RequestAttempt(execution_id='exe-s', attempt_id='att-s', client_id='c-s'))
        assert p_shadow.lease_id is None
        await shadow_ctrl.close()
    finally:
        await engine.dispose()


async def test_request_volume_scaling_does_not_cause_per_request_db_access_and_tracks_utilization(tmp_path):
    clock = ControlledClock()
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "ctrl_scale.db"}')
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    # 600 QPM -> 10 req/s (0.1s per request). A 10s lease holds capacity for ~98 requests.
    config = _make_config(mode='enforce', quota_qpm=600.0)
    selection = RequestPolicySelection(binding='team-a-hy3-rate', scope='capacity')

    try:
        async with sessions() as session:
            session.add(_manager_row(clock.utcnow()))
            session.add(_scope_row(config, selection, now=clock.utcnow(), rate_per_minute=600.0, valid_seconds=120.0))
            await session.commit()

        async def auto_advance_sleep(seconds: float) -> None:
            clock.advance(seconds)

        ctrl = RequestBudgetControl(
            session_factory=sessions,
            manager_id='mgr-1',
            manager_session_id='sess-1',
            environment_id='env-a',
            owner_principal_id='alice',
            selection=selection,
            policy_config=config,
            budget_config=RequestBudgetConfig(lease_seconds=10, renew_before_seconds=2, safety_margin_seconds=0.2),
            monotonic_clock=clock.monotonic,
            wall_clock=clock.utcnow,
            sleep_func=auto_advance_sleep,
        )

        # Coordinate once at t=0 after registering demand via first waiter.
        first_task = asyncio.create_task(
            ctrl.acquire(RequestAttempt(execution_id='exe-scale', attempt_id='att-1', client_id='c-1'))
        )
        await asyncio.sleep(0.01)
        res = await ctrl.coordinate_once()
        assert res is not None
        assert res.status == 'granted'
        await first_task

        # Issue 49 more requests (50 total, 10x more than a 5-request batch) within the same 10s lease:
        # zero additional DB coordination calls are made on the acquire() path!
        for i in range(49):
            await ctrl.acquire(RequestAttempt(execution_id='exe-scale', attempt_id='att-1', client_id=f'c-{i + 2}'))

        snap = ctrl.status_snapshot()
        assert snap.permitted_requests_total == 50
        assert snap.enforced_permitted_requests_total == 50
        assert snap.coordination_requests_total == 1
        assert snap.leased_permit_capacity_total > 0
        assert 0.45 <= snap.lease_utilization_ratio <= 0.55
        assert snap.coordination_seconds_total >= 0.0
        await ctrl.close()
    finally:
        await engine.dispose()


async def test_background_coordination_loop_does_not_spin_under_waiting_clients(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "ctrl_nospin.db"}')
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    selection = RequestPolicySelection(binding='team-a-hy3-rate', scope='capacity')

    try:
        # Case 1: 6 QPM (10s interval), 1 waiting client, 10s lease / 4s renew_before, real start() loop over 0.2s.
        config_6 = _make_config(mode='enforce', quota_qpm=6.0)
        from nexus.distributed.utils.clock import utcnow

        now = utcnow()
        async with sessions() as session:
            session.add(_manager_row(now))
            session.add(_scope_row(config_6, selection, now=now, rate_per_minute=6.0, valid_seconds=120.0))
            await session.commit()

        ctrl_6 = RequestBudgetControl(
            session_factory=sessions,
            manager_id='mgr-1',
            manager_session_id='sess-1',
            environment_id='env-a',
            owner_principal_id='alice',
            selection=selection,
            policy_config=config_6,
            budget_config=RequestBudgetConfig(lease_seconds=10, renew_before_seconds=4, safety_margin_seconds=0.2),
        )
        ctrl_6.start()
        waiter_1 = asyncio.create_task(
            ctrl_6.acquire(RequestAttempt(execution_id='exe-spin-1', attempt_id='att-1', client_id='c-1'))
        )
        await asyncio.sleep(0.2)
        snap_6 = ctrl_6.status_snapshot()
        assert snap_6.permitted_requests_total == 0
        assert snap_6.coordination_requests_total == 1
        await ctrl_6.close()
        with pytest.raises(RuntimeError, match='request_budget_control_closed'):
            await waiter_1

        # Case 2: 60 QPM (1s interval), 2 waiting clients, 10s lease / 4s renew_before, real start() loop over 0.2s.
        config_60 = _make_config(mode='enforce', quota_qpm=60.0)
        now2 = utcnow()
        async with sessions() as session:
            mgr = await session.get(InstanceManagerRow, 'mgr-1')
            assert mgr is not None
            mgr.session_id = 'sess-2'
            mgr.last_heartbeat_at = now2
            scope = await session.get(PolicyScopeRow, request_budget_key(selection))
            assert scope is not None
            scope.config_hash = config_60.bindings['team-a-hy3-rate'].fingerprint(config_60.imports)
            scope.snapshot_json = {
                'snapshot_version': 2,
                'reason_code': 'adopted',
                'payload': {'allow': True, 'rate_per_minute': 60.0},
            }
            scope.valid_until = now2 + timedelta(seconds=120)
            await session.commit()

        ctrl_60 = RequestBudgetControl(
            session_factory=sessions,
            manager_id='mgr-1',
            manager_session_id='sess-2',
            environment_id='env-a',
            owner_principal_id='alice',
            selection=selection,
            policy_config=config_60,
            budget_config=RequestBudgetConfig(lease_seconds=10, renew_before_seconds=4, safety_margin_seconds=0.2),
        )
        ctrl_60.start()
        w1 = asyncio.create_task(
            ctrl_60.acquire(RequestAttempt(execution_id='exe-spin-2', attempt_id='att-1', client_id='c-1'))
        )
        w2 = asyncio.create_task(
            ctrl_60.acquire(RequestAttempt(execution_id='exe-spin-2', attempt_id='att-1', client_id='c-2'))
        )
        await asyncio.sleep(0.2)
        snap_60 = ctrl_60.status_snapshot()
        assert snap_60.permitted_requests_total == 0
        assert snap_60.coordination_requests_total == 1
        await ctrl_60.close()
        for w in (w1, w2):
            with pytest.raises(RuntimeError, match='request_budget_control_closed'):
                await w
    finally:
        await engine.dispose()


async def test_background_config_refresh_preserves_role_and_identity_override_precedence(tmp_path):
    clock = ControlledClock()
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "ctrl_override.db"}')
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    global_disabled_cfg = _make_config(mode='disabled', revision=1, quota_qpm=60.0)
    manager_enforce_cfg = _make_config(mode='enforce', revision=1, quota_qpm=60.0)
    selection = RequestPolicySelection(binding='team-a-hy3-rate', scope='capacity')

    try:
        async with sessions() as session:
            session.add(_manager_row(clock.utcnow()))
            session.add(
                _scope_row(
                    manager_enforce_cfg, selection, now=clock.utcnow(), rate_per_minute=60.0, valid_seconds=120.0
                )
            )
            # Global (*, *) is disabled at revision=1, while ('instance-manager', 'mgr-1') overrides to enforce at revision=1!
            session.add(
                DistributedConfigOverrideRow(
                    role='*',
                    identity='*',
                    config_json={'policy_executor': global_disabled_cfg.model_dump(mode='json')},
                    version=1,
                    created_at=clock.utcnow(),
                    updated_at=clock.utcnow(),
                )
            )
            session.add(
                DistributedConfigOverrideRow(
                    role='instance-manager',
                    identity='mgr-1',
                    config_json={'policy_executor': manager_enforce_cfg.model_dump(mode='json')},
                    version=1,
                    created_at=clock.utcnow(),
                    updated_at=clock.utcnow(),
                )
            )
            await session.commit()

        ctrl = RequestBudgetControl(
            session_factory=sessions,
            manager_id='mgr-1',
            manager_session_id='sess-1',
            environment_id='env-a',
            owner_principal_id='alice',
            selection=selection,
            policy_config=manager_enforce_cfg,
            monotonic_clock=clock.monotonic,
            wall_clock=clock.utcnow,
        )
        assert ctrl.mode == 'enforce'

        # Refreshing from DB must respect ('instance-manager', 'mgr-1') over ('*', '*') and keep enforce!
        await ctrl._refresh_policy_config_if_available()
        assert ctrl.mode == 'enforce'

        # Waiter arrives and coordinates under enforce (must NOT bypass as disabled!).
        req = asyncio.create_task(
            ctrl.acquire(RequestAttempt(execution_id='exe-ov', attempt_id='att-1', client_id='c-1'))
        )
        await asyncio.sleep(0.02)
        assert not req.done()
        res = await ctrl.coordinate_once()
        assert res is not None
        assert res.status == 'granted'
        assert ctrl.mode == 'enforce'
        await ctrl.close()
        with pytest.raises(RuntimeError, match='request_budget_control_closed'):
            await req
    finally:
        await engine.dispose()
