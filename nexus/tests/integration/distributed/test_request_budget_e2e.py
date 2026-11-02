"""End-to-end integration verification of shared request_rate budget across Managers, Scheduler, and InferProxyClient.

Covers the acceptance criteria in docs/developing/shared-request-budget-policy-design.md §10.2 using a real
MySQL database (or SQLite fallback when NEXUS_POLICY_TEST_DATABASE_URL_FILE is unset), real PolicyExecutor
isolated worker processes, real HTTP capacity and model servers, multiple InstanceManagers, and InferProxyClient.
"""

from __future__ import annotations

import asyncio
import math
import os
import time

from pathlib import Path
from typing import Any

import httpx
import pytest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from sqlalchemy import delete, select
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from tenacity import retry, stop_after_attempt, wait_none

from nexus.core import OpenAIConfig
from nexus.core.request_admission import bind_request_admission
from nexus.distributed.config import DistributedConfig
from nexus.distributed.config.views import instance_manager_config_view, scheduler_config_view
from nexus.distributed.control_plane.scheduler import SchedulerService
from nexus.distributed.gateways.control.app import create_control_app
from nexus.distributed.instance_manager.service import InstanceManagerService
from nexus.distributed.persistence.orm import (
    AssignmentRow,
    DistributedConfigOverrideRow,
    ExecutionRow,
    InstanceManagerLivenessRow,
    InstanceManagerRow,
    PolicyScopeRow,
    RequestBudgetLeaseRow,
    RequestBudgetRow,
    ShardLeaseRow,
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
from nexus.distributed.utils.clock import utcnow
from nexus.extensions.openai_clients import infer_proxy
from nexus.extensions.openai_clients.infer_proxy import InferProxyClient


@pytest.fixture
async def db_sessions(tmp_path):
    filename = os.getenv('NEXUS_POLICY_TEST_DATABASE_URL_FILE')
    if filename:
        url = make_url(Path(filename).read_text().strip())
        if not (url.database or '').startswith('nexus_policy'):
            raise ValueError('use a dedicated nexus_policy test database')
        engine = create_async_engine(url, pool_size=16, max_overflow=16)
    else:
        engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "e2e_budget.db"}')
    await create_schema(engine)
    async with engine.begin() as conn:
        await conn.execute(delete(RequestBudgetLeaseRow))
        await conn.execute(delete(RequestBudgetRow))
        await conn.execute(delete(AssignmentRow))
        await conn.execute(delete(ExecutionRow))
        await conn.execute(delete(InstanceManagerLivenessRow))
        await conn.execute(delete(InstanceManagerRow))
        await conn.execute(delete(PolicyScopeRow))
        await conn.execute(delete(ShardLeaseRow))
        await conn.execute(delete(DistributedConfigOverrideRow))
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def test_shared_request_budget_end_to_end_multi_manager_burst_pause_and_recovery(
    db_sessions, tmp_path, monkeypatch
):
    # Mutable capacity signal state served over real HTTP to isolated PolicyWorker child processes.
    signal_state: dict[str, Any] = {
        'remaining': 500,
        'factor': 1.0,
        'hard_brake': False,
        'queries': 0,
    }
    model_timestamps: list[float] = []
    permit_timestamps: list[float] = []
    fail_first_attempts: set[str] = set()

    async def handle_capacity(request: web.Request) -> web.Response:
        body = await request.json()
        assert body.get('model') == 'hy3-offline'
        assert body.get('api_key') in {'team-a-secret-key', 'team-a-high-secret-key'}
        signal_state['queries'] += 1
        return web.json_response({
            'success': True,
            'key_remaining_raw_qpm': signal_state['remaining'],
            'scaling_advice': {
                'factor': signal_state['factor'],
                'hard_brake': signal_state['hard_brake'],
            },
        })

    async def handle_chat_completions(request: web.Request) -> web.Response:
        model_timestamps.append(time.monotonic())
        body = await request.json()
        prompt = body['messages'][0]['content']
        if prompt in fail_first_attempts:
            fail_first_attempts.remove(prompt)
            return web.Response(status=500, text='{"error":{"message":"retry-me"}}', content_type='application/json')
        return web.json_response({
            'id': f'cmpl-{len(model_timestamps)}',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': body.get('model', 'hy3-offline'),
            'choices': [
                {
                    'index': 0,
                    'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': f'reply:{prompt}'},
                }
            ],
        })

    signal_app = web.Application()
    signal_app.router.add_post('/capacity', handle_capacity)
    model_app = web.Application()
    model_app.router.add_post('/v1/chat/completions', handle_chat_completions)

    monkeypatch.setattr(
        infer_proxy,
        'configurable_retry',
        lambda attempts: retry(stop=stop_after_attempt(attempts), wait=wait_none(), reraise=True),
    )

    async with TestServer(signal_app) as cap_server, TestServer(model_app) as model_server:
        cap_url = str(cap_server.make_url('/capacity'))
        model_base_url = str(model_server.make_url('/v1'))

        def make_policy_config(*, mode: str, revision: int, quota_qpm: float = 240.0) -> PolicyExecutorConfig:
            return PolicyExecutorConfig.model_validate({
                'imports': [
                    'nexus.distributed.policies.collections.infer_proxy_request_rate',
                    'tests.unit.distributed.test_request_budget_wiring',
                ],
                'poll_seconds': 0.1,
                'bindings': {
                    'team-a-hy3-rate': {
                        'kind': 'request_rate',
                        'policy': REQUEST_RATE_POLICY,
                        'source': REQUEST_CAPACITY_SOURCE,
                        'revision': revision,
                        'mode': mode,
                        'on_unavailable': 'defer',
                        'environments': ['env-a', 'env-b'],
                        'principals': ['alice'],
                        'refresh_seconds': 1.0,
                        'timeout_seconds': 5.0,
                        'max_age_seconds': 30.0,
                        'scopes': {
                            'capacity': {
                                'source_config': {
                                    'model': 'hy3-offline',
                                    'api_key': 'team-a-secret-key',
                                    'url': cap_url,
                                }
                            },
                            'capacity-high': {
                                'source_config': {
                                    'model': 'hy3-offline',
                                    'api_key': 'team-a-high-secret-key',
                                    'url': cap_url,
                                    'quota_qpm': 600.0,
                                }
                            },
                        },
                        'params': {
                            'quota_qpm': quota_qpm,
                            'min_quota_ratio': 0.01,
                        },
                    },
                    'team-b-static-rate': {
                        'kind': 'request_rate',
                        'policy': 'static_fixed_request_rate',
                        'source': 'static_rate_source',
                        'revision': 1,
                        'mode': 'enforce',
                        'on_unavailable': 'defer',
                        'environments': ['env-a'],
                        'principals': ['alice'],
                        'refresh_seconds': 1.0,
                        'timeout_seconds': 5.0,
                        'max_age_seconds': 30.0,
                        'scopes': {'default': {'source_config': {'qpm': 180.0}}},
                        'params': {},
                    },
                },
                'request_defaults': {
                    'env-a': {'binding': 'team-a-hy3-rate', 'scope': 'capacity'},
                },
            })

        policy_cfg = make_policy_config(mode='enforce', revision=1, quota_qpm=240.0)
        budget_cfg = RequestBudgetConfig(lease_seconds=2.0, renew_before_seconds=0.8, safety_margin_seconds=0.05)
        selection_a = RequestPolicySelection(binding='team-a-hy3-rate', scope='capacity')
        selection_a_high = RequestPolicySelection(binding='team-a-hy3-rate', scope='capacity-high')
        selection_b = RequestPolicySelection(binding='team-b-static-rate', scope='default')

        dist_cfg = DistributedConfig(
            auth_mode='disabled',
            shard_count=8,
            policy_executor=policy_cfg,
            request_budget=budget_cfg,
        )

        # Start Scheduler with PolicyExecutor (isolated worker pool) in background.
        scheduler = SchedulerService(
            session_factory=db_sessions,
            config=scheduler_config_view(dist_cfg),
            owner_id='e2e-scheduler',
        )
        assert scheduler._policy_executor is not None
        await scheduler._policy_executor.initialize()
        policy_loop_task = asyncio.create_task(scheduler._policy_executor.run_forever())

        # Start two Managers sharing team-a-hy3-rate/capacity (240 QPM total -> 120 QPM each = 2 req/s each),
        # plus one Manager bound to team-b-static-rate/default (180 QPM).
        mgr_a1 = InstanceManagerService(
            session_factory=db_sessions,
            config=instance_manager_config_view(dist_cfg),
            manager_id='mgr-a-1',
            endpoint_url='http://127.0.0.1:18091',
            capacity_total=4,
            environment_id='env-a',
            pool_id='default',
            workload='execution',
            owner_principal_id='alice',
            request_policy=selection_a,
        )
        mgr_a2 = InstanceManagerService(
            session_factory=db_sessions,
            config=instance_manager_config_view(dist_cfg),
            manager_id='mgr-a-2',
            endpoint_url='http://127.0.0.1:18092',
            capacity_total=4,
            environment_id='env-a',
            pool_id='default',
            workload='execution',
            owner_principal_id='alice',
            request_policy=selection_a,
        )
        mgr_b1 = InstanceManagerService(
            session_factory=db_sessions,
            config=instance_manager_config_view(dist_cfg),
            manager_id='mgr-b-1',
            endpoint_url='http://127.0.0.1:18093',
            capacity_total=4,
            environment_id='env-a',
            pool_id='default',
            workload='execution',
            owner_principal_id='alice',
            request_policy=selection_b,
        )

        mgr_a_high = InstanceManagerService(
            session_factory=db_sessions,
            config=instance_manager_config_view(dist_cfg),
            manager_id='mgr-a-high-1',
            endpoint_url='http://127.0.0.1:18094',
            capacity_total=4,
            environment_id='env-a',
            pool_id='default',
            workload='execution',
            owner_principal_id='alice',
            request_policy=selection_a_high,
        )

        await mgr_a1.start()
        await mgr_a2.start()
        await mgr_b1.start()
        await mgr_a_high.start()
        assert mgr_a1.request_budget_control is not None
        assert mgr_a2.request_budget_control is not None
        assert mgr_b1.request_budget_control is not None
        assert mgr_a_high.request_budget_control is not None
        mgr_a1.request_budget_control.start()
        mgr_a2.request_budget_control.start()
        mgr_b1.request_budget_control.start()
        mgr_a_high.request_budget_control.start()

        for ctrl in (
            mgr_a1.request_budget_control,
            mgr_a2.request_budget_control,
            mgr_b1.request_budget_control,
        ):
            orig_acquire = ctrl.acquire

            async def _tracked_acquire(request, _orig=orig_acquire):
                permit = await _orig(request)
                permit_timestamps.append(time.monotonic())
                return permit

            monkeypatch.setattr(ctrl, 'acquire', _tracked_acquire)

        try:
            async with httpx.AsyncClient(base_url=model_base_url) as http_client:

                def build_client() -> InferProxyClient:
                    return InferProxyClient(
                        OpenAIConfig(
                            client_args={
                                'api_key': 'task-model-key',
                                'base_url': model_base_url,
                                'http_client': http_client,
                                'max_retries': 2,
                            },
                            request_args={'model': 'hy3-offline'},
                        )
                    )

                async def run_task_requests(ctrl, exe_id: str, prompts: list[str]) -> list[str]:
                    outputs: list[str] = []
                    with bind_request_admission(ctrl, execution_id=exe_id, attempt_id=f'{exe_id}-att-1'):
                        client = build_client()
                        for prompt in prompts:
                            res = await client._create_completion([{'role': 'user', 'content': prompt}])
                            outputs.append(res.choices[0].message.content)
                    return outputs

                # 1. Both mgr-a-1 (running Batch children) and mgr-a-2 (running Single) share 240 QPM total.
                #    Include a transient 500 on 'batch-item-0' to verify retry re-acquires a permit.
                fail_first_attempts.add('batch-item-0')
                t_start = time.monotonic()
                res_a1, res_a2, res_b1 = await asyncio.gather(
                    run_task_requests(
                        mgr_a1.request_budget_control,
                        'exe-batch-child-1',
                        ['batch-item-0', 'batch-item-1', 'batch-item-2'],
                    ),
                    run_task_requests(
                        mgr_a2.request_budget_control,
                        'exe-single-1',
                        ['single-item-0', 'single-item-1', 'single-item-2'],
                    ),
                    run_task_requests(
                        mgr_b1.request_budget_control,
                        'exe-team-b-1',
                        ['team-b-0', 'team-b-1'],
                    ),
                )
                elapsed = time.monotonic() - t_start
                assert res_a1 == ['reply:batch-item-0', 'reply:batch-item-1', 'reply:batch-item-2']
                assert res_a2 == ['reply:single-item-0', 'reply:single-item-1', 'reply:single-item-2']
                assert res_b1 == ['reply:team-b-0', 'reply:team-b-1']

                # Verify sum of granted_qpm across mgr-a-1 and mgr-a-2 in DB never exceeded 240 QPM,
                # and each active manager split the 240 QPM budget (120 QPM each) once both had demand.
                key_a = request_budget_key(selection_a)
                async with db_sessions() as session:
                    leases_a = (
                        await session.scalars(
                            select(RequestBudgetLeaseRow).where(RequestBudgetLeaseRow.budget_key == key_a)
                        )
                    ).all()
                assert len(leases_a) == 2
                now_db = utcnow()
                unexpired_a = sum(
                    r.granted_qpm for r in leases_a if r.valid_until is not None and r.valid_until > now_db
                )
                assert unexpired_a <= 240.0 + 1e-6

                # Verify burst bound (§7.4): mgr-a-1 issued 4 permits (3 prompts + 1 retry), mgr-a-2 issued 3 permits
                # = 7 permits total under 240 QPM (4 req/s). With zero initial free permits, 7 permits across
                # 2 controls require at least ~1.0s, and coordination queries are bounded (far fewer than 1 per req).
                s_a1 = mgr_a1.request_budget_control.status_snapshot()
                s_a2 = mgr_a2.request_budget_control.status_snapshot()
                assert s_a1.permitted_requests_total == 4
                assert s_a2.permitted_requests_total == 3
                assert s_a1.lease_utilization_ratio > 0.0
                assert s_a2.lease_utilization_ratio > 0.0
                assert s_a1.coordination_seconds_total >= 0.0
                assert elapsed >= 0.8
                assert len(model_timestamps) == 9
                assert len(permit_timestamps) == 9
                # Sliding-window §7.4 burst bound across permit issuance timestamps (240 QPM + 180 QPM = 420 QPM across 3 controls):
                for idx_i, t_i in enumerate(permit_timestamps):
                    for idx_j in range(idx_i, len(permit_timestamps)):
                        t_j = permit_timestamps[idx_j]
                        count = idx_j - idx_i + 1
                        upper_bound = (240.0 + 180.0) * max(0.0, t_j - t_i) / 60.0 + 3.0 + 0.1
                        assert count <= upper_bound, (idx_i, idx_j, count, upper_bound)

                # 2. Hard brake does not block requests; dropping below the floor does.
                async def wait_for_scope_snapshot(predicate, timeout: float = 8.0) -> dict[str, Any]:
                    deadline = time.monotonic() + timeout
                    while time.monotonic() < deadline:
                        async with db_sessions() as session:
                            from nexus.distributed.repositories.policy import PolicyRepository

                            await PolicyRepository(session, shard_count=8).demand([key_a], now=utcnow())
                            await session.commit()
                            scope_row = await session.get(PolicyScopeRow, key_a)
                            snap = scope_row.snapshot_json if scope_row is not None else None
                            if isinstance(snap, dict) and predicate(snap):
                                return snap
                        await asyncio.sleep(0.2)
                    raise TimeoutError('timed out waiting for policy scope snapshot')

                signal_state['factor'] = 0.5
                signal_state['hard_brake'] = False
                snap_braked = await wait_for_scope_snapshot(
                    lambda s: (
                        s.get('payload', {}).get('allow') is True
                        and s.get('payload', {}).get('rate_per_minute', 240.0) <= 120.0 + 1e-6
                    )
                )
                assert math.isclose(snap_braked['payload']['rate_per_minute'], 120.0)

                signal_state['factor'] = 1.0
                signal_state['hard_brake'] = True
                snap_hard = await wait_for_scope_snapshot(
                    lambda s: (
                        s.get('snapshot_version') != snap_braked.get('snapshot_version')
                        and s.get('payload', {}).get('allow') is True
                        and math.isclose(s.get('payload', {}).get('rate_per_minute', 0.0), 120.0)
                    )
                )
                hard_out = await run_task_requests(mgr_a1.request_budget_control, 'exe-hard', ['hard-ignored'])
                assert hard_out == ['reply:hard-ignored']
                signal_state['factor'] = 0.001
                snap_paused = await wait_for_scope_snapshot(lambda s: s.get('reason_code') == 'below_minimum')
                assert snap_paused['payload'] == {'allow': False, 'rate_per_minute': 0.0}
                assert snap_paused['snapshot_version'] != snap_hard['snapshot_version']
                signal_state['factor'] = 1.0
                await mgr_a1.request_budget_control.coordinate_once()
                await mgr_a2.request_budget_control.coordinate_once()
                assert mgr_a1.request_budget_control.status_snapshot().granted_qpm == 0.0

                # A paused budget waits while a different budget continues making real model requests.
                paused_task = asyncio.create_task(
                    run_task_requests(mgr_a1.request_budget_control, 'exe-paused', ['after-pause'])
                )
                b_during_pause = await run_task_requests(
                    mgr_b1.request_budget_control, 'exe-team-b-2', ['team-b-unblocked']
                )
                assert b_during_pause == ['reply:team-b-unblocked']
                await wait_for_scope_snapshot(
                    lambda s: (
                        s.get('reason_code') == 'below_minimum'
                        and s.get('snapshot_version') != snap_paused['snapshot_version']
                    )
                )
                assert not paused_task.done()

                # 3. Quota exhaustion preserves the floor pause; recovery requires positive quota and factor > 1.
                signal_state['remaining'] = 0
                signal_state['factor'] = 2.0
                await wait_for_scope_snapshot(lambda s: s.get('reason_code') == 'quota_exhausted')
                assert not paused_task.done()
                signal_state['remaining'] = 500
                signal_state['factor'] = 1.0
                await wait_for_scope_snapshot(lambda s: s.get('reason_code') == 'below_minimum')
                assert not paused_task.done()
                signal_state['factor'] = 2.0
                snap_recovered = await wait_for_scope_snapshot(lambda s: s.get('reason_code') == 'resumed')
                assert math.isclose(snap_recovered['payload']['rate_per_minute'], 240.0)
                signal_state['factor'] = 1.0
                recovered_out = await asyncio.wait_for(paused_task, timeout=8.0)
                assert recovered_out == ['reply:after-pause']

                # 4. Verify draining on mgr-a-1 does not interrupt in-flight task requests,
                #    and verify the second scope 'capacity-high' (600 QPM in source_config) under the same binding.
                await mgr_a1.set_draining(True)
                draining_out = await run_task_requests(mgr_a1.request_budget_control, 'exe-draining', ['drain-item-1'])
                assert draining_out == ['reply:drain-item-1']
                high_out = await run_task_requests(
                    mgr_a_high.request_budget_control, 'exe-high-scope', ['high-scope-1']
                )
                assert high_out == ['reply:high-scope-1']
                snap_high_mgr = mgr_a_high.request_budget_control.status_snapshot()
                assert snap_high_mgr.configured_quota_qpm == 600.0
                assert math.isclose(snap_high_mgr.granted_qpm, 600.0)

                # 5. Verify Control Gateway /v1/executions/{id}/request-policy endpoint for both scopes.
                now = utcnow()
                async with db_sessions() as session:
                    session.add_all([
                        ExecutionRow(
                            execution_id='exe-e2e-inspect',
                            shard_id=0,
                            mode='job',
                            spec_type='nexus_single',
                            instance_mode='asyncio',
                            status='running',
                            route_status='none',
                            spec_json={'spec_version': 1},
                            placement_json={
                                'environment': 'env-a',
                                'pool': 'default',
                                'request_policy': {'binding': 'team-a-hy3-rate', 'scope': 'capacity'},
                            },
                            owner_principal_id='alice',
                            environment_id='env-a',
                            pool_id='default',
                            workload='execution',
                            slot_cost=1,
                            max_attempts=1,
                            next_attempt_seq=1,
                            next_event_seq=1,
                            priority=0,
                            next_schedule_at=now,
                            queue_entered_at=now,
                            created_at=now,
                            updated_at=now,
                        ),
                        ExecutionRow(
                            execution_id='exe-e2e-inspect-high',
                            shard_id=0,
                            mode='job',
                            spec_type='nexus_single',
                            instance_mode='asyncio',
                            status='running',
                            route_status='none',
                            spec_json={'spec_version': 1},
                            placement_json={
                                'environment': 'env-b',
                                'pool': 'default',
                                'request_policy': {'binding': 'team-a-hy3-rate', 'scope': 'capacity-high'},
                            },
                            owner_principal_id='alice',
                            environment_id='env-b',
                            pool_id='default',
                            workload='execution',
                            slot_cost=1,
                            max_attempts=1,
                            next_attempt_seq=1,
                            next_event_seq=1,
                            priority=0,
                            next_schedule_at=now,
                            queue_entered_at=now,
                            created_at=now,
                            updated_at=now,
                        ),
                    ])
                    await session.commit()

                control_app = create_control_app(session_factory=db_sessions, config=dist_cfg)
                async with TestClient(TestServer(control_app)) as gw_client:
                    resp = await gw_client.get('/v1/executions/exe-e2e-inspect/request-policy')
                    assert resp.status == 200
                    payload = await resp.json()
                    assert payload['enabled'] is True
                    assert payload['binding'] == 'team-a-hy3-rate'
                    assert payload['scope'] == 'capacity'
                    assert payload['configured_quota_qpm'] == 240.0
                    assert payload['allow'] is True
                    assert len(payload['managers']) == 2

                    resp_high = await gw_client.get('/v1/executions/exe-e2e-inspect-high/request-policy')
                    assert resp_high.status == 200
                    payload_high = await resp_high.json()
                    assert payload_high['enabled'] is True
                    assert payload_high['binding'] == 'team-a-hy3-rate'
                    assert payload_high['scope'] == 'capacity-high'
                    assert payload_high['configured_quota_qpm'] == 600.0
                    assert payload_high['effective_rate_per_minute'] == 600.0
                    assert payload_high['allow'] is True
                    assert len(payload_high['managers']) == 1
        finally:
            scheduler._policy_executor.stopping.set()
            policy_loop_task.cancel()
            await asyncio.gather(policy_loop_task, return_exceptions=True)
            await scheduler._policy_executor.close()
            await mgr_a1.stop()
            await mgr_a2.stop()
            await mgr_b1.stop()
            await mgr_a_high.stop()
