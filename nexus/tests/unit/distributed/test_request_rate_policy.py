"""Unit tests for request_rate contracts, consumer validation, and Tide request_rate plugin."""

from __future__ import annotations

import asyncio
import math

from datetime import datetime, timedelta

import pytest

from aiohttp import web
from aiohttp.test_utils import TestServer
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.core.request_admission import (
    RequestAttempt,
    RequestPermit,
    RequestPermitExpiredError,
    SmoothRatePacer,
    bind_request_admission,
    current_request_admission,
)
from nexus.distributed.errors import DomainValidationError
from nexus.distributed.persistence.orm import PolicyScopeRow, create_schema
from nexus.distributed.policies.collections.infer_proxy_request_rate import (
    REQUEST_CAPACITY_SOURCE,
    REQUEST_RATE_POLICY,
    InferProxyRequestCapacitySource,
    InferProxyRequestRatePolicy,
)
from nexus.distributed.policies.executor import PolicyExecutor
from nexus.distributed.policies.models import (
    PolicyExecutorConfig,
    PolicyInput,
    RequestBudgetConfig,
    RequestPolicySelection,
    RequestRateDecision,
    SignalObservation,
)
from nexus.distributed.policies.request_rate import (
    configured_rate_per_minute,
    consume_request_rate_snapshot,
    request_budget_key,
    resolve_request_selection,
    validate_manager_request_policy,
)
from nexus.distributed.repositories.policy import PolicyRepository
from nexus.distributed.utils.clock import utcnow


NOW = datetime(2026, 10, 19, 10, 0, 0)


def test_smooth_rate_pacer_enforces_zero_initial_free_permit_and_single_burst_cap():
    pacer = SmoothRatePacer(now=100.0)
    # 6 QPM -> 10 seconds per request. At t=100.0 (initial), no free permit is available.
    wait = pacer.try_acquire(qpm=6.0, now=100.0)
    assert math.isclose(wait, 10.0)

    # At t=104.0 (still within first 10s), 6s remain; continuous renewal does not reset progress.
    assert math.isclose(pacer.try_acquire(qpm=6.0, now=104.0), 6.0)

    # At t=110.0, exactly 1 permit is issued.
    assert pacer.try_acquire(qpm=6.0, now=110.0) == 0.0

    # Idle for 60 seconds (until t=170.0): at most 1 burst permit is retained, so the first request
    # at t=170.0 succeeds immediately and the second must wait a full 10s interval.
    assert pacer.try_acquire(qpm=6.0, now=170.0) == 0.0
    assert math.isclose(pacer.try_acquire(qpm=6.0, now=170.0), 10.0)

    # After an expired gap, reset(now=200.0) discards any accumulated idle time.
    pacer.reset(200.0)
    assert math.isclose(pacer.try_acquire(qpm=6.0, now=200.0), 10.0)


async def test_request_admission_context_binding_and_permit_deadline():
    calls: list[RequestAttempt] = []
    exited: list[bool] = []

    class DummyAdmission:
        async def acquire(self, request: RequestAttempt) -> RequestPermit:
            calls.append(request)
            return RequestPermit(
                execution_id=request.execution_id,
                attempt_id=request.attempt_id,
                session_id='sess-1',
                lease_id='lease-1',
                deadline_at=50.0,
            )

    assert current_request_admission() is None
    with bind_request_admission(
        DummyAdmission(),
        execution_id='exe-1',
        attempt_id='exe-1-attempt-1',
        on_exit=lambda: exited.append(True),
    ) as bound:
        assert current_request_admission() is bound
        permit = await bound.acquire(client_id='client-1')
        assert permit.is_valid(now=49.9)
        permit.ensure_valid(now=49.9)
        assert not permit.is_valid(now=50.0)
        with pytest.raises(RequestPermitExpiredError):
            permit.ensure_valid(now=50.0)
    assert current_request_admission() is None
    assert len(calls) == 1
    assert exited == [True]


def test_request_rate_decision_and_binding_config_validation():
    allowed = RequestRateDecision(allow=True, rate_per_minute=300.0)
    assert allowed.rate_per_minute == 300.0
    denied = RequestRateDecision(allow=False, rate_per_minute=0.0)
    assert denied.rate_per_minute == 0.0

    with pytest.raises(ValueError, match='allowed request rate must be positive'):
        RequestRateDecision(allow=True, rate_per_minute=0.0)
    with pytest.raises(ValueError, match='denied request rate must be zero'):
        RequestRateDecision(allow=False, rate_per_minute=10.0)

    multi_scope_cfg = PolicyExecutorConfig.model_validate({
        'bindings': {
            'rate-a': {
                'kind': 'request_rate',
                'policy': REQUEST_RATE_POLICY,
                'source': REQUEST_CAPACITY_SOURCE,
                'environments': ['env-a'],
                'params': {'quota_qpm': 100.0, 'min_quota_ratio': 0.01},
                'scopes': {
                    's1': {'source_config': {'model': 'hy4-preview-offline', 'api_key': 'k1'}},
                    's2': {'source_config': {'model': 'hy4-preview-offline', 'api_key': 'k2', 'quota_qpm': 6600.0}},
                },
            }
        }
    })
    binding_a = multi_scope_cfg.bindings['rate-a']
    assert configured_rate_per_minute(binding_a, 's1') == 100.0
    assert configured_rate_per_minute(binding_a, 's2') == 6600.0

    with pytest.raises(ValueError, match='requires on_unavailable=defer'):
        PolicyExecutorConfig.model_validate({
            'bindings': {
                'rate-a': {
                    'kind': 'request_rate',
                    'policy': REQUEST_RATE_POLICY,
                    'source': REQUEST_CAPACITY_SOURCE,
                    'mode': 'enforce',
                    'on_unavailable': 'default',
                    'environments': ['env-a'],
                    'scopes': {'capacity': {}},
                }
            }
        })

    with pytest.raises(ValueError, match='renew_before_seconds must be less than lease_seconds'):
        RequestBudgetConfig(lease_seconds=10, renew_before_seconds=10)


def test_resolve_request_selection_and_snapshot_consumption():
    config = PolicyExecutorConfig.model_validate({
        'bindings': {
            'team-a-hy3-rate': {
                'kind': 'request_rate',
                'policy': REQUEST_RATE_POLICY,
                'source': REQUEST_CAPACITY_SOURCE,
                'mode': 'enforce',
                'on_unavailable': 'defer',
                'environments': ['env-a'],
                'principals': ['alice'],
                'scopes': {'capacity': {'source_config': {'model': 'hy3-offline', 'api_key': 'test-key'}}},
            }
        },
        'request_defaults': {
            'env-a': {'binding': 'team-a-hy3-rate', 'scope': 'capacity'},
        },
    })

    # Batch parent orchestration never binds a model request_policy.
    assert (
        resolve_request_selection(
            config,
            None,
            environment_id='env-a',
            owner_principal_id='alice',
            spec_type='nexus_batch',
        )
        is None
    )

    # Business execution inherits environment request_defaults and cannot bypass it.
    selection = resolve_request_selection(
        config,
        None,
        environment_id='env-a',
        owner_principal_id='alice',
        spec_type='nexus_single',
    )
    assert selection == RequestPolicySelection(binding='team-a-hy3-rate', scope='capacity')

    with pytest.raises(DomainValidationError, match='request_policy_required'):
        resolve_request_selection(
            config,
            {'binding': 'other', 'scope': 'capacity'},
            environment_id='env-a',
            owner_principal_id='alice',
            spec_type='nexus_single',
        )

    with pytest.raises(DomainValidationError, match='request_policy_not_permitted'):
        resolve_request_selection(
            config,
            None,
            environment_id='env-a',
            owner_principal_id='bob',
            spec_type='nexus_single',
        )

    assert validate_manager_request_policy(config, selection, environment_id='env-a') == selection
    with pytest.raises(ValueError, match='invalid manager request_policy binding'):
        validate_manager_request_policy(config, {'binding': 'missing', 'scope': 'capacity'})

    binding = config.bindings['team-a-hy3-rate']
    digest = binding.fingerprint(config.imports)
    key = request_budget_key(selection)
    row = PolicyScopeRow(
        scope_key=key,
        kind='request_rate',
        binding_id='team-a-hy3-rate',
        scope_id='capacity',
        shard_id=0,
        config_revision=1,
        config_hash=digest,
        # Consumer must ignore private state_json and read only snapshot_json.
        state_json={'quota_blocked': True, 'quota_ratio': 0.1},
        snapshot_json={
            'snapshot_version': 4,
            'reason_code': 'adopted',
            'payload': {'allow': True, 'rate_per_minute': 300.0},
        },
        valid_until=NOW + timedelta(seconds=60),
        next_evaluate_at=NOW + timedelta(seconds=30),
        last_demand_at=NOW,
        version=4,
        failures=0,
        last_error=None,
        updated_at=NOW,
    )
    reading = consume_request_rate_snapshot(selection, config, row, now=NOW)
    assert reading.ready is True
    assert reading.allow is True
    assert reading.rate_per_minute == 300.0
    assert reading.snapshot_version == 4

    # Expired snapshot under enforce -> unavailable (allow=False, rate=0.0).
    expired = consume_request_rate_snapshot(selection, config, row, now=NOW + timedelta(seconds=61))
    assert expired.ready is False
    assert expired.allow is False
    assert expired.rate_per_minute == 0.0
    assert expired.reason_code == 'policy_unavailable'


def test_request_rate_pauses_below_minimum_and_requires_growth_to_resume():
    policy = InferProxyRequestRatePolicy()
    state = {}

    def step(values, expected_rate, reason, *, age=180):
        nonlocal state
        result = policy.evaluate(
            PolicyInput(
                scope_id='capacity',
                now=NOW,
                params={'quota_qpm': 600.0, 'min_quota_ratio': 0.01},
                observation=SignalObservation(
                    version='sample',
                    observed_at=NOW,
                    valid_until=NOW + timedelta(seconds=age),
                    values=values,
                ),
            ),
            state,
        )
        state = result.next_state
        assert result.decision == {'allow': expected_rate > 0, 'rate_per_minute': expected_rate}
        assert result.reason_code == reason

    step({'remaining': 500, 'factor': 1.0, 'hard_brake': True}, 600, 'unchanged')
    step({'remaining': 500, 'factor': 0.5, 'hard_brake': True}, 300, 'adopted')
    # No cooldown or step clamp: a second observation immediately halves the quota again.
    step({'remaining': 500, 'factor': 0.5, 'hard_brake': False}, 150, 'adopted')
    # Equality with the minimum still permits requests; falling below it pauses.
    step({'remaining': 500, 'factor': 0.04, 'hard_brake': True}, 6, 'adopted')
    step({'remaining': 500, 'factor': 1.0}, 6, 'unchanged')
    step({'remaining': 500, 'factor': 0.5}, 0, 'below_minimum')
    assert state == {'quota_ratio': 0.01, 'paused_below_minimum': True}
    step({'remaining': 500, 'factor': 0.5, 'hard_brake': False}, 0, 'below_minimum')
    step({'remaining': 500, 'factor': 1.0, 'hard_brake': False}, 0, 'below_minimum')
    # Quota loss and incomplete observations do not erase the below-minimum pause.
    step({'remaining': 0, 'factor': 2}, 0, 'quota_exhausted')
    step({'remaining': 500}, 0, 'signal_incomplete')
    step({'remaining': 500, 'factor': 1.0}, 0, 'below_minimum')
    step({'remaining': 500, 'factor': 2, 'hard_brake': True}, 12, 'resumed')
    step({'remaining': 0, 'factor': 2, 'hard_brake': False}, 0, 'quota_exhausted')
    step({'remaining': 500, 'factor': 1.0, 'hard_brake': True}, 12, 'unchanged')
    step({'remaining': 500, 'factor': 100, 'hard_brake': False}, 600, 'adopted')
    step({'remaining': 500, 'factor': 1, 'hard_brake': False}, 0, 'signal_expired', age=0)


@pytest.mark.parametrize('saved_ratio', [0.015625, 0.25, 1.0])
def test_request_rate_recovers_from_saved_rate_instead_of_zero_or_minimum(saved_ratio):
    policy = InferProxyRequestRatePolicy()

    def evaluate(factor, state):
        return policy.evaluate(
            PolicyInput(
                scope_id='capacity',
                now=NOW,
                params={'quota_qpm': 600, 'min_quota_ratio': 0.01},
                observation=SignalObservation(
                    version='sample',
                    observed_at=NOW,
                    valid_until=NOW + timedelta(seconds=180),
                    values={'remaining': 500, 'factor': factor, 'hard_brake': True},
                ),
            ),
            state,
        )

    paused = evaluate(0.001, {'quota_ratio': saved_ratio})
    assert paused.decision == {'allow': False, 'rate_per_minute': 0.0}
    assert paused.next_state['quota_ratio'] == saved_ratio
    held = evaluate(0.5, paused.next_state)
    assert held.next_state == paused.next_state
    resumed = evaluate(1.1, held.next_state)
    assert resumed.decision['allow'] is True
    assert resumed.decision['rate_per_minute'] == pytest.approx(600 * min(1, saved_ratio * 1.1))
    assert resumed.reason_code == 'resumed'
    assert resumed.next_state['paused_below_minimum'] is False


@pytest.mark.parametrize('hard_brake', [True, False, None, 'invalid'])
def test_request_rate_does_not_require_or_use_hard_brake(hard_brake):
    values = {'remaining': 500, 'factor': 0.5}
    if hard_brake is not None:
        values['hard_brake'] = hard_brake
    result = InferProxyRequestRatePolicy().evaluate(
        PolicyInput(
            scope_id='capacity',
            now=NOW,
            params={'quota_qpm': 600},
            observation=SignalObservation(
                version='sample',
                observed_at=NOW,
                valid_until=NOW + timedelta(seconds=180),
                values=values,
            ),
        ),
        {},
    )
    assert result.decision == {'allow': True, 'rate_per_minute': 300.0}


@pytest.mark.parametrize(
    'values',
    [
        {'factor': 1.0, 'hard_brake': False},
        {'remaining': 500},
        {'remaining': True, 'factor': 1.0, 'hard_brake': False},
        {'remaining': 500, 'factor': True, 'hard_brake': False},
        {'remaining': 500, 'factor': float('nan'), 'hard_brake': False},
        {'remaining': 500, 'factor': -1, 'hard_brake': False},
        {'remaining': 500, 'factor': 0},
        {'remaining': 500, 'factor': float('inf')},
    ],
)
def test_request_rate_requires_a_complete_current_signal(values):
    result = InferProxyRequestRatePolicy().evaluate(
        PolicyInput(
            scope_id='capacity',
            now=NOW,
            params={'quota_qpm': 600},
            observation=SignalObservation(
                version='sample',
                observed_at=NOW,
                valid_until=NOW + timedelta(seconds=180),
                values=values,
            ),
        ),
        {'quota_ratio': 0.5, 'quota_blocked': False, 'quota_valid_until': (NOW + timedelta(seconds=180)).isoformat()},
    )
    assert result.decision == {'allow': False, 'rate_per_minute': 0.0}
    assert result.reason_code == 'signal_incomplete'


def test_unkeyed_request_rate_uses_model_capacity_without_requiring_key_quota():
    result = InferProxyRequestRatePolicy().evaluate(
        PolicyInput(
            scope_id='capacity',
            now=NOW,
            params={'quota_qpm': 600},
            observation=SignalObservation(
                version='sample',
                observed_at=NOW,
                valid_until=NOW + timedelta(seconds=180),
                values={'keyed': False, 'factor': 0.5, 'hard_brake': False},
            ),
        ),
        {},
    )
    assert result.decision == {'allow': True, 'rate_per_minute': 300.0}


async def test_policy_executor_evaluates_and_publishes_request_rate_snapshot(tmp_path):
    async def handle_capacity(request: web.Request) -> web.Response:
        body = await request.json()
        assert body == {'model': 'hy3-offline', 'api_key': 'secret-taiji-key'}
        return web.json_response({
            'success': True,
            'key_remaining_raw_qpm': 320,
            'scaling_advice': {'factor': 0.5, 'hard_brake': True},
        })

    app = web.Application()
    app.router.add_post('/capacity', handle_capacity)

    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "policy.db"}')
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with TestServer(app) as server:
            source = InferProxyRequestCapacitySource()
            obs = await source.read(
                {
                    'model': 'hy3-offline',
                    'api_key': 'secret-taiji-key',
                    'url': str(server.make_url('/capacity')),
                },
                max_age_seconds=120,
            )
            assert 'secret-taiji-key' not in str(obs.model_dump())
            assert obs.values == {'keyed': True, 'remaining': 320, 'factor': 0.5, 'hard_brake': True}

            config = PolicyExecutorConfig.model_validate({
                'imports': ['nexus.distributed.policies.collections.infer_proxy_request_rate'],
                'bindings': {
                    'team-a-hy3-rate': {
                        'kind': 'request_rate',
                        'policy': REQUEST_RATE_POLICY,
                        'source': REQUEST_CAPACITY_SOURCE,
                        'mode': 'enforce',
                        'on_unavailable': 'defer',
                        'environments': ['env-a'],
                        'scopes': {
                            'capacity': {
                                'source_config': {
                                    'model': 'hy3-offline',
                                    'api_key': 'secret-taiji-key',
                                    'url': str(server.make_url('/capacity')),
                                }
                            },
                            'capacity-high': {
                                'source_config': {
                                    'model': 'hy3-offline',
                                    'api_key': 'secret-taiji-key',
                                    'url': str(server.make_url('/capacity')),
                                    'quota_qpm': 6600.0,
                                }
                            },
                        },
                        'params': {'quota_qpm': 600.0},
                    }
                },
            })
            selection = RequestPolicySelection(binding='team-a-hy3-rate', scope='capacity')
            selection_high = RequestPolicySelection(binding='team-a-hy3-rate', scope='capacity-high')
            executor = PolicyExecutor(
                session_factory=sessions,
                config=config,
                shard_count=8,
                owner_id='sched-1',
            )
            await executor.initialize()
            try:
                async with sessions() as session:
                    await PolicyRepository(session, shard_count=8).demand(
                        [request_budget_key(selection), request_budget_key(selection_high)],
                        now=utcnow(),
                    )
                    await session.commit()
                await executor.tick_once()
                await asyncio.gather(*(task for _, task in executor.active.values()))
                await executor.tick_once()
                await asyncio.gather(*(task for _, task in executor.active.values()))
                async with sessions() as session:
                    rows = await PolicyRepository(session, shard_count=8).read_scopes([
                        request_budget_key(selection),
                        request_budget_key(selection_high),
                    ])
                    row = rows[request_budget_key(selection)]
                    reading = consume_request_rate_snapshot(selection, config, row, now=utcnow())
                    assert reading.ready is True
                    assert reading.allow is True
                    assert reading.rate_per_minute == 300
                    assert reading.reason_code == 'adopted'

                    row_high = rows[request_budget_key(selection_high)]
                    reading_high = consume_request_rate_snapshot(selection_high, config, row_high, now=utcnow())
                    assert reading_high.ready is True
                    assert reading_high.allow is True
                    assert reading_high.rate_per_minute == 3300
                    assert reading_high.reason_code == 'adopted'
            finally:
                await executor.close()
    finally:
        await engine.dispose()
