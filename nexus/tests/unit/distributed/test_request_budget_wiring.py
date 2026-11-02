"""Unit tests for Phase 4 end-to-end wiring: Scheduler matching, Driver, InferProxyClient, and Gateway query."""

from __future__ import annotations

import math

from datetime import datetime, timedelta
from typing import Any

import httpx
import pytest

from aiohttp.test_utils import TestClient, TestServer
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from tenacity import retry, stop_after_attempt, wait_none

from nexus.core import OpenAIConfig
from nexus.core.request_admission import bind_request_admission
from nexus.distributed.config import DistributedConfig
from nexus.distributed.config.views import instance_manager_config_view
from nexus.distributed.errors import DomainValidationError
from nexus.distributed.execution.drivers.nexus_live import NexusLiveDriver
from nexus.distributed.gateways.control.app import create_control_app
from nexus.distributed.instance_manager.request_budget import RequestBudgetControl
from nexus.distributed.instance_manager.service import InstanceManagerService
from nexus.distributed.persistence.orm import (
    ExecutionRow,
    PolicyScopeRow,
    create_schema,
)
from nexus.distributed.policies.collections.infer_proxy_request_rate import (
    REQUEST_CAPACITY_SOURCE,
    REQUEST_RATE_POLICY,
)
from nexus.distributed.policies.models import (
    REQUEST_RATE_KIND,
    PolicyExecutorConfig,
    PolicyInput,
    PolicyResult,
    RequestBudgetConfig,
    RequestPolicySelection,
    RequestRateDecision,
    SignalObservation,
)
from nexus.distributed.policies.registry import policy_registry, signal_source_registry
from nexus.distributed.policies.request_rate import request_budget_key
from nexus.distributed.repositories.manager import ManagerRepository
from nexus.distributed.repositories.scheduler import SchedulerRepository
from nexus.distributed.utils.clock import utcnow
from nexus.extensions.openai_clients import infer_proxy
from nexus.extensions.openai_clients.infer_proxy import InferProxyClient


NOW = datetime(2026, 10, 19, 15, 0, 0)


@signal_source_registry.register('static_rate_source')
class StaticRateSource:
    async def read(self, config: dict[str, Any], *, max_age_seconds: float) -> SignalObservation:
        now = utcnow()
        return SignalObservation(
            version=now.isoformat(),
            observed_at=now,
            valid_until=now + timedelta(seconds=max_age_seconds),
            values={'qpm': float(config.get('qpm', 120.0))},
        )


@policy_registry.register('static_fixed_request_rate')
class StaticFixedRequestRatePolicy:
    kind = REQUEST_RATE_KIND
    decision_model = RequestRateDecision

    def evaluate(self, inputs: PolicyInput, previous_state: dict[str, Any]) -> PolicyResult:
        qpm = float(inputs.observation.values.get('qpm', 120.0))
        return PolicyResult(
            decision={'allow': qpm > 0, 'rate_per_minute': max(0.0, qpm)},
            next_state=dict(previous_state),
            valid_until=inputs.observation.valid_until,
            reason_code='static_rate',
        )


def _policy_config() -> PolicyExecutorConfig:
    return PolicyExecutorConfig.model_validate({
        'bindings': {
            'team-a-hy3-rate': {
                'kind': 'request_rate',
                'policy': REQUEST_RATE_POLICY,
                'source': REQUEST_CAPACITY_SOURCE,
                'revision': 1,
                'mode': 'enforce',
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
                'scopes': {'default': {'source_config': {'qpm': 120.0}}},
                'params': {},
            },
        }
    })


async def test_scheduler_enforces_request_policy_consistency_and_rejects_session_rebind(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "sched_match.db"}')
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with sessions() as session:
            mgr_repo = ManagerRepository(session)
            # Manager 1: bound to team-a-hy3-rate / capacity
            await mgr_repo.register_session(
                manager_id='mgr-team-a',
                session_id='sess-a',
                endpoint_url='http://127.0.0.1:8091',
                supported_modes=['job'],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                supported_spec_versions={'nexus_single': [1]},
                capacity_total=4,
                force_replace=True,
                environment_id='env-a',
                pool_id='default',
                workload='execution',
                request_policy_binding='team-a-hy3-rate',
                request_policy_scope='capacity',
                now=NOW,
            )
            # Attempting to switch request_policy on the same session_id is rejected!
            with pytest.raises(DomainValidationError, match='manager_request_policy_immutable'):
                await mgr_repo.register_session(
                    manager_id='mgr-team-a',
                    session_id='sess-a',
                    endpoint_url='http://127.0.0.1:8091',
                    supported_modes=['job'],
                    supported_spec_types=['nexus_single'],
                    supported_instance_modes=['asyncio'],
                    supported_spec_versions={'nexus_single': [1]},
                    capacity_total=4,
                    force_replace=True,
                    environment_id='env-a',
                    pool_id='default',
                    workload='execution',
                    request_policy_binding='team-b-static-rate',
                    request_policy_scope='default',
                    now=NOW,
                )

            # Attempting to switch request_policy on a new session_id with force_replace while not drained is also rejected!
            with pytest.raises(DomainValidationError, match='manager_request_policy_immutable'):
                await mgr_repo.register_session(
                    manager_id='mgr-team-a',
                    session_id='sess-a-new',
                    endpoint_url='http://127.0.0.1:8091',
                    supported_modes=['job'],
                    supported_spec_types=['nexus_single'],
                    supported_instance_modes=['asyncio'],
                    supported_spec_versions={'nexus_single': [1]},
                    capacity_total=4,
                    force_replace=True,
                    environment_id='env-a',
                    pool_id='default',
                    workload='execution',
                    request_policy_binding='team-b-static-rate',
                    request_policy_scope='default',
                    now=NOW,
                )

            # Manager 2: unbound
            await mgr_repo.register_session(
                manager_id='mgr-unbound',
                session_id='sess-u',
                endpoint_url='http://127.0.0.1:8092',
                supported_modes=['job'],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                supported_spec_versions={'nexus_single': [1]},
                capacity_total=4,
                force_replace=True,
                environment_id='env-a',
                pool_id='default',
                workload='execution',
                request_policy_binding=None,
                request_policy_scope=None,
                now=NOW,
            )

            # Execution 1: requires team-a-hy3-rate/capacity -> must go to mgr-team-a
            # Execution 2: requires team-b-static-rate/default -> has no matching Manager, stays queued!
            # Execution 3: unbound -> must go to mgr-unbound
            for exe_id, req_policy in (
                ('exe-a', {'binding': 'team-a-hy3-rate', 'scope': 'capacity'}),
                ('exe-b', {'binding': 'team-b-static-rate', 'scope': 'default'}),
                ('exe-unbound', None),
            ):
                placement: dict[str, Any] = {'environment': 'env-a', 'pool': 'default'}
                if req_policy is not None:
                    placement['request_policy'] = req_policy
                session.add(
                    ExecutionRow(
                        execution_id=exe_id,
                        shard_id=0,
                        mode='job',
                        spec_type='nexus_single',
                        instance_mode='asyncio',
                        status='queued',
                        route_status='none',
                        spec_json={'spec_version': 1},
                        placement_json=placement,
                        owner_principal_id='alice',
                        environment_id='env-a',
                        pool_id='default',
                        workload='execution',
                        slot_cost=1,
                        max_attempts=1,
                        next_attempt_seq=1,
                        next_event_seq=1,
                        priority=0,
                        next_schedule_at=NOW,
                        queue_entered_at=NOW,
                        created_at=NOW,
                        updated_at=NOW,
                    )
                )
            await session.commit()

        async with sessions() as session:
            sched_repo = SchedulerRepository(
                session,
                shard_count=8,
                manager_healthy_ttl=timedelta(seconds=30),
                default_user_quota={
                    'max_running_executions': 10,
                    'max_queued_executions': 10,
                    'max_reserved_slots': 10,
                },
                policy_config=_policy_config(),
            )
            assigned = await sched_repo.claim_and_assign_batch(shard_ids=list(range(8)), now=NOW)
            await session.commit()

        by_exe = {record.execution_id: record.manager_id for record in assigned}
        assert by_exe == {
            'exe-a': 'mgr-team-a',
            'exe-unbound': 'mgr-unbound',
        }
        # exe-b had no matching Manager for team-b-static-rate and was not assigned to mgr-team-a or mgr-unbound!
        assert 'exe-b' not in by_exe
    finally:
        await engine.dispose()


async def test_infer_proxy_client_uses_request_budget_control_and_rejects_static_tide_qpm_conflict(
    tmp_path, monkeypatch
):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "client_wire.db"}')
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    policy_cfg = _policy_config()
    selection = RequestPolicySelection(binding='team-b-static-rate', scope='default')
    key = request_budget_key(selection)
    binding = policy_cfg.bindings['team-b-static-rate']

    try:
        async with sessions() as session:
            mgr_repo = ManagerRepository(session)
            await mgr_repo.register_session(
                manager_id='mgr-b',
                session_id='sess-b',
                endpoint_url='http://127.0.0.1:8093',
                supported_modes=['job'],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                supported_spec_versions={'nexus_single': [1]},
                capacity_total=4,
                force_replace=True,
                environment_id='env-a',
                pool_id='default',
                workload='execution',
                request_policy_binding='team-b-static-rate',
                request_policy_scope='default',
                now=NOW,
            )
            # High-rate snapshot (60000 QPM -> 1ms interval) for fast test execution.
            session.add(
                PolicyScopeRow(
                    scope_key=key,
                    kind='request_rate',
                    binding_id=selection.binding,
                    scope_id=selection.scope,
                    shard_id=0,
                    config_revision=1,
                    config_hash=binding.fingerprint(policy_cfg.imports),
                    state_json={},
                    snapshot_json={
                        'snapshot_version': 1,
                        'reason_code': 'static_rate',
                        'payload': {'allow': True, 'rate_per_minute': 60000.0},
                    },
                    valid_until=NOW + timedelta(seconds=120),
                    next_evaluate_at=NOW + timedelta(seconds=60),
                    last_demand_at=NOW,
                    version=1,
                    failures=0,
                    last_error=None,
                    updated_at=NOW,
                )
            )
            await session.commit()

        ctrl = RequestBudgetControl(
            session_factory=sessions,
            manager_id='mgr-b',
            manager_session_id='sess-b',
            environment_id='env-a',
            owner_principal_id='alice',
            selection=selection,
            policy_config=policy_cfg,
            budget_config=RequestBudgetConfig(lease_seconds=10, renew_before_seconds=4),
            wall_clock=lambda: NOW,
        )
        ctrl.start()

        send_attempts = 0

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal send_attempts
            send_attempts += 1
            if send_attempts == 1:
                return httpx.Response(500, json={'error': {'message': 'transient'}})
            return httpx.Response(
                200,
                json={
                    'id': 'cmpl-1',
                    'object': 'chat.completion',
                    'created': 1,
                    'model': 'hy3-offline',
                    'choices': [
                        {
                            'index': 0,
                            'finish_reason': 'stop',
                            'message': {'role': 'assistant', 'content': 'ok'},
                        }
                    ],
                },
            )

        monkeypatch.setattr(
            infer_proxy,
            'configurable_retry',
            lambda attempts: retry(stop=stop_after_attempt(attempts), wait=wait_none(), reraise=True),
        )

        async with httpx.AsyncClient(transport=httpx.MockTransport(mock_handler)) as http_client:
            with bind_request_admission(ctrl, execution_id='exe-1', attempt_id='exe-1-attempt-1'):
                # 1. Conflicting static tide_qpm config under platform request_policy raises ValueError!
                with pytest.raises(ValueError, match='tide_qpm cannot be combined with platform request_policy'):
                    InferProxyClient(
                        OpenAIConfig(
                            client_args={
                                'api_key': 'k',
                                'base_url': 'https://provider.test/v1',
                                'http_client': http_client,
                                'tide_qpm': {'quota_qpm': 600, 'task_concurrency': 10},
                            },
                            request_args={'model': 'hy3-offline'},
                        )
                    )

                # 2. Normal InferProxyClient acquires a permit on initial send AND on retry (2 permits total).
                client = InferProxyClient(
                    OpenAIConfig(
                        client_args={
                            'api_key': 'k',
                            'base_url': 'https://provider.test/v1',
                            'http_client': http_client,
                            'max_retries': 2,
                        },
                        request_args={'model': 'hy3-offline'},
                    )
                )
                assert client.client.max_retries == 0
                resp = await client._create_completion([{'role': 'user', 'content': 'hi'}])
                assert resp.choices[0].message.content == 'ok'
                assert send_attempts == 2
                status = ctrl.status_snapshot()
                assert status.permitted_requests_total == 2
                assert math.isclose(status.granted_qpm, 60000.0)

        await ctrl.close()
    finally:
        await engine.dispose()


async def test_driver_verifies_request_policy_and_gateway_exposes_request_policy_status(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "gw_req_policy.db"}')
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    policy_cfg = _policy_config()
    selection = RequestPolicySelection(binding='team-a-hy3-rate', scope='capacity')
    dist_cfg = DistributedConfig(
        auth_mode='disabled',
        shard_count=8,
        policy_executor=policy_cfg,
        request_policy=selection,
    )

    try:
        mgr_service = InstanceManagerService(
            session_factory=sessions,
            config=instance_manager_config_view(dist_cfg),
            manager_id='mgr-a',
            endpoint_url='http://127.0.0.1:8091',
            capacity_total=2,
            environment_id='env-a',
            pool_id='default',
            workload='execution',
            owner_principal_id='alice',
        )
        assert mgr_service.request_policy == selection
        assert mgr_service.request_budget_control is not None

        # Manager assignment resolution rejects mismatched request_policy and injects RequestAdmission into Single/Live drivers.
        from types import SimpleNamespace

        bad_exe = SimpleNamespace(
            execution_id='exe-bad',
            placement={'request_policy': {'binding': 'team-b-static-rate', 'scope': 'default'}},
        )
        with pytest.raises(RuntimeError, match='execution_request_policy_mismatch'):
            mgr_service._resolve_execution_request_admission(bad_exe)  # type: ignore[arg-type]

        good_exe = SimpleNamespace(
            execution_id='exe-ok',
            placement={'request_policy': {'binding': 'team-a-hy3-rate', 'scope': 'capacity'}},
        )
        resolved_admission = mgr_service._resolve_execution_request_admission(good_exe)  # type: ignore[arg-type]
        assert resolved_admission is mgr_service.request_budget_control

        good_live_driver = NexusLiveDriver(
            execution_id='exe-live-ok',
            assignment_id='asn-live-ok',
            internal_base_url='http://127.0.0.1:8091',
            trace_registry=mgr_service.trace_registry,
            request_admission=resolved_admission,
        )
        from nexus.core.request_admission import bind_request_admission, current_request_admission

        with bind_request_admission(
            good_live_driver.request_admission,
            execution_id=good_live_driver.execution_id,
            attempt_id=good_live_driver.attempt_id,
        ):
            bound = current_request_admission()
            assert bound is not None
            assert bound.admission is mgr_service.request_budget_control

        await mgr_service.start()
        try:
            key = request_budget_key(selection)
            binding = policy_cfg.bindings['team-a-hy3-rate']
            now = utcnow()
            async with sessions() as session:
                session.add(
                    PolicyScopeRow(
                        scope_key=key,
                        kind='request_rate',
                        binding_id=selection.binding,
                        scope_id=selection.scope,
                        shard_id=0,
                        config_revision=1,
                        config_hash=binding.fingerprint(policy_cfg.imports),
                        state_json={},
                        snapshot_json={
                            'generated_at': now.isoformat(),
                            'snapshot_version': 3,
                            'reason_code': 'adopted',
                            'payload': {'allow': True, 'rate_per_minute': 480.0},
                        },
                        valid_until=now + timedelta(seconds=120),
                        next_evaluate_at=now + timedelta(seconds=60),
                        last_demand_at=now,
                        version=3,
                        failures=0,
                        last_error=None,
                        updated_at=now,
                    )
                )
                session.add(
                    ExecutionRow(
                        execution_id='exe-query-1',
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
                    )
                )
                await session.commit()

            # Trigger demand and one coordination step on mgr_service's RequestBudgetControl.
            import time as _time

            mgr_service.request_budget_control._last_activity_mono = _time.monotonic()
            await mgr_service.request_budget_control.coordinate_once()

            app = create_control_app(session_factory=sessions, config=dist_cfg)
            async with TestClient(TestServer(app)) as client:
                resp = await client.get('/v1/executions/exe-query-1/request-policy')
                assert resp.status == 200
                data = await resp.json()
                assert data['enabled'] is True
                assert data['binding'] == 'team-a-hy3-rate'
                assert data['scope'] == 'capacity'
                assert data['mode'] == 'enforce'
                assert data['configured_quota_qpm'] == 600.0
                assert data['effective_rate_per_minute'] == 480.0
                assert data['reason_code'] == 'adopted'
                assert len(data['managers']) == 1
                assert data['managers'][0]['manager_id'] == 'mgr-a'
                assert data['managers'][0]['observed_mode'] == 'enforce'
                assert data['managers'][0]['granted_qpm'] == 480.0
                assert isinstance(data['managers'][0]['metrics'], dict)
                assert 'lease_utilization_ratio' in data['managers'][0]['metrics']
        finally:
            await mgr_service.stop()
    finally:
        await engine.dispose()
