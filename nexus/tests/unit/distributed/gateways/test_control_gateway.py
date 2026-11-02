from __future__ import annotations

import re

from datetime import timedelta
from types import SimpleNamespace

import pytest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.config import DistributedConfig
from nexus.distributed.config.store import get_config_override, set_config_override
from nexus.distributed.execution.models import (
    AssignmentStatus,
    ExecutionMode,
    ExecutionStatus,
    RouteStatus,
)
from nexus.distributed.gateways.common.platform_metrics import LIVE_CACHE_KEY
from nexus.distributed.gateways.control.app import (
    CONSOLE_INDEX,
    _allocation_payload,
    _database_runtime_metrics,
    create_control_app,
)
from nexus.distributed.observability.cache import LiveStatusCache
from nexus.distributed.persistence.orm import (
    AssignmentRow,
    BatchItemAttemptRow,
    BatchItemRow,
    BatchRunRow,
    DependencyBuildRow,
    DistributedConfigOverrideRow,
    EnvironmentAllocationRow,
    ExecutionCommandRow,
    ExecutionRow,
    InstanceManagerLivenessRow,
    InstanceManagerRow,
    ShardLeaseRow,
    create_schema,
)
from nexus.distributed.repositories.environment import EnvironmentInventoryRepository, EnvironmentRepository
from nexus.distributed.repositories.execution import CreateExecution, ExecutionRepository
from nexus.distributed.repositories.manager import ManagerRepository
from nexus.distributed.repositories.scheduler import SchedulerRepository
from nexus.distributed.utils.clock import utcnow


@pytest.fixture
async def session_factory():
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    await create_schema(engine)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


def test_allocation_payload_includes_dependency_build_progress_without_user_internal_details():
    now = utcnow()
    allocation = {
        'allocation_id': 'allocation-a',
        'status': 'requested',
        'ready_manager_count': 0,
        'desired_manager_count': 1,
    }
    build = SimpleNamespace(
        dependency_build_id='build-a',
        status='building',
        attempt=1,
        provider_job_name='builder-job-a',
        error_code='pip_wheel_failed',
        error_message='private builder diagnostics',
        next_retry_at=None,
        started_at=now,
        finished_at=None,
        updated_at=now,
    )

    user_payload = _allocation_payload(allocation, admin=False, dependency_build=build)
    admin_payload = _allocation_payload(allocation, admin=True, dependency_build=build)

    assert user_payload['dependency_build'] == {
        'dependency_build_id': 'build-a',
        'status': 'building',
        'attempt': 1,
        'next_retry_at': None,
        'started_at': now.isoformat(),
        'finished_at': None,
        'updated_at': now.isoformat(),
    }
    assert admin_payload['dependency_build']['provider_job_name'] == 'builder-job-a'
    assert admin_payload['dependency_build']['error_code'] == 'pip_wheel_failed'
    assert admin_payload['dependency_build']['error_message'] == 'private builder diagnostics'


async def test_capabilities_exposes_fixed_workload_image_repository(session_factory):
    base_image = f'registry.example:5000/team/workload@sha256:{"a" * 64}'
    app = create_control_app(
        session_factory=session_factory,
        config=DistributedConfig(environment_base_image=base_image),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/capabilities')
        assert response.status == 200
        capabilities = await response.json()

        assert capabilities['workload_image'] == {
            'enabled': True,
            'repository': 'registry.example:5000/team/workload',
            'base_image': base_image,
        }
    finally:
        await client.close()


async def test_capabilities_disables_workload_image_for_unpinned_production_base(session_factory):
    base_image = 'registry.example/team/workload:platform-v1'
    app = create_control_app(
        session_factory=session_factory,
        config=DistributedConfig(environment_base_image=base_image, production_mode=True),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/capabilities')
        assert response.status == 200
        capabilities = await response.json()

        assert capabilities['workload_image'] == {
            'enabled': False,
            'repository': None,
            'base_image': base_image,
        }
    finally:
        await client.close()


async def test_control_gateway_creates_activates_and_stops_cached_live(session_factory):
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    app = create_control_app(session_factory=session_factory, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            '/v1/executions',
            json={
                'execution_id': 'live-cached-1',
                'mode': ExecutionMode.LIVE,
                'spec_type': 'nexus_live',
                'instance_mode': 'asyncio',
                'spec': {'task_name': 'identity'},
                'route_status': RouteStatus.CACHED,
            },
        )
        assert response.status == 201
        created = await response.json()
        assert created['execution_id'] == 'live-cached-1'
        assert created['route_status'] == RouteStatus.CACHED
        assert created['owner_principal_id'] == 'local-dev'

        response = await client.post('/v1/executions/live-cached-1/activate')
        assert response.status == 200
        activated = await response.json()
        assert activated['route_status'] == RouteStatus.ACTIVE

        response = await client.get('/v1/access/audit')
        assert response.status == 200
        activation_audit = next(
            item
            for item in (await response.json())['items']
            if item['operation'] == 'execution.activate' and item['resource_id'] == 'live-cached-1'
        )
        assert activation_audit['actor_principal_id'] == 'local-dev'
        assert activation_audit['target_owner_principal_id'] == 'local-dev'

        response = await client.post('/v1/executions/live-cached-1/stop')
        assert response.status == 200
        stopped = await response.json()
        assert stopped['status'] == ExecutionStatus.KILLED
        assert stopped['route_status'] == RouteStatus.CLOSED

        response = await client.get('/v1/executions/live-cached-1')
        assert response.status == 200
        current = await response.json()
        assert current['status'] == ExecutionStatus.KILLED
        assert current['phase'] == ExecutionStatus.KILLED

        response = await client.get('/v1/executions/live-cached-1/control-events')
        assert response.status == 200
        events = (await response.json())['items']
        assert [event['type'] for event in events] == [
            'execution.created',
            'route.status.changed',
            'execution.status.changed',
            'route.status.changed',
        ]
        assert events[1]['payload'] == {'from': RouteStatus.CACHED, 'to': RouteStatus.ACTIVE}
        assert events[2]['payload'] == {'from': ExecutionStatus.QUEUED, 'to': ExecutionStatus.KILLED}
    finally:
        await client.close()


async def test_batch_summary_status_counts_require_explicit_diagnostic_query(session_factory):
    now = utcnow()
    async with session_factory() as session:
        await ExecutionRepository(session, shard_count=8).create_execution(
            CreateExecution(
                execution_id='batch-summary-diagnostic',
                mode=ExecutionMode.JOB,
                spec_type='nexus_batch',
                instance_mode='asyncio',
                spec={},
                owner_principal_id='local-dev',
            ),
            now=now,
        )
        session.add(
            BatchRunRow(
                parent_execution_id='batch-summary-diagnostic',
                shard_id=0,
                generation_mode='local_task_list',
                generation_status='complete',
                generation_epoch=1,
                materialized_cursor='0',
                materialize_cursor_bytes=0,
                materialized_count=0,
                task_list_ref_json=None,
                task_list_manifest_json=None,
                task_list_sha256=None,
                owner_principal_id='local-dev',
                owner_tenant=None,
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

    app = create_control_app(session_factory=session_factory, config=DistributedConfig(shard_count=8))
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/executions/batch-summary-diagnostic/batch')
        assert response.status == 200
        assert 'child_execution_status_counts' not in await response.json()

        response = await client.get('/v1/executions/batch-summary-diagnostic/batch?include_status_counts=true')
        assert response.status == 200
        payload = await response.json()
        assert payload['child_execution_status_counts'] == {}
        assert payload['child_waiting_policy_count'] == 0
        assert payload['item_status_counts'] == {}
    finally:
        await client.close()


async def test_batch_outcome_is_exposed_on_summary_and_parent_execution_views(session_factory):
    now = utcnow()
    cases = {
        'batch-active': (ExecutionStatus.RUNNING, None),
        'batch-succeeded': (
            ExecutionStatus.SUCCEEDED,
            {'total': 2, 'succeeded': 2, 'failed': 0, 'killed': 0, 'lost': 0, 'expired': 0, 'cancelled': 0},
        ),
        'batch-partial': (
            ExecutionStatus.FAILED,
            {'total': 2, 'succeeded': 1, 'failed': 1, 'killed': 0, 'lost': 0, 'expired': 0, 'cancelled': 0},
        ),
        'batch-failed': (
            ExecutionStatus.FAILED,
            {'total': 2, 'succeeded': 0, 'failed': 2, 'killed': 0, 'lost': 0, 'expired': 0, 'cancelled': 0},
        ),
    }
    async with session_factory() as session:
        repo = ExecutionRepository(session, shard_count=8)
        for execution_id, (status, summary) in cases.items():
            await repo.create_execution(
                CreateExecution(
                    execution_id=execution_id,
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_batch',
                    instance_mode='asyncio',
                    spec={},
                    owner_principal_id='local-dev',
                ),
                now=now,
            )
            execution = await session.get(ExecutionRow, execution_id)
            assert execution is not None
            execution.status = status
            total = summary['total'] if summary is not None else 2
            session.add(
                BatchRunRow(
                    parent_execution_id=execution_id,
                    shard_id=execution.shard_id,
                    generation_mode='local_task_list',
                    generation_status='complete',
                    generation_epoch=1,
                    materialized_cursor=str(total),
                    materialize_cursor_bytes=0,
                    materialized_count=total,
                    task_list_ref_json=None,
                    task_list_manifest_json=None,
                    task_list_sha256=None,
                    owner_principal_id='local-dev',
                    owner_tenant=None,
                    total_items=total,
                    submitted_items=total,
                    active_items=0 if summary is not None else total,
                    terminal_items=total if summary is not None else 0,
                    submission_closed=summary is not None,
                    control_state='finalized' if summary is not None else 'active',
                    summary_version=1 if summary is not None else 0,
                    summary_json=summary,
                    last_progress_at=now,
                    finalized_at=now if summary is not None else None,
                    created_at=now,
                    updated_at=now,
                )
            )
        await session.commit()

    app = create_control_app(session_factory=session_factory, config=DistributedConfig(shard_count=8))
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/executions?spec_type=nexus_batch&top_level=1&summary=1&limit=10')
        assert response.status == 200
        listed = {item['execution_id']: item['outcome'] for item in (await response.json())['items']}
        assert listed == {
            'batch-active': None,
            'batch-succeeded': 'succeeded',
            'batch-partial': 'partial_success',
            'batch-failed': 'failed',
        }

        response = await client.get('/v1/executions/batch-partial')
        assert response.status == 200
        assert (await response.json())['outcome'] == 'partial_success'

        response = await client.get('/v1/executions/batch-partial/batch')
        assert response.status == 200
        assert (await response.json())['outcome'] == 'partial_success'
    finally:
        await client.close()


async def test_batch_items_reject_noncanonical_cursor(session_factory):
    async with session_factory() as session:
        await ExecutionRepository(session, shard_count=8).create_execution(
            CreateExecution(
                execution_id='batch-invalid-cursor',
                mode=ExecutionMode.JOB,
                spec_type='nexus_batch',
                instance_mode='asyncio',
                spec={},
                owner_principal_id='local-dev',
            ),
            now=utcnow(),
        )
        await session.commit()

    app = create_control_app(session_factory=session_factory, config=DistributedConfig(shard_count=8))
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/executions/batch-invalid-cursor/batch/items?cursor=Item')

        assert response.status == 422
        assert await response.json() == {'error': 'invalid_batch_item_id', 'retryable': False}
    finally:
        await client.close()


async def test_batch_items_support_indexed_status_identity_and_error_filters(session_factory):
    now = utcnow()
    parent_execution_id = 'batch-filter-items'
    async with session_factory() as session:
        await ExecutionRepository(session, shard_count=8).create_execution(
            CreateExecution(
                execution_id=parent_execution_id,
                mode=ExecutionMode.JOB,
                spec_type='nexus_batch',
                instance_mode='asyncio',
                spec={},
                owner_principal_id='local-dev',
            ),
            now=now,
        )
        session.add(
            BatchRunRow(
                parent_execution_id=parent_execution_id,
                shard_id=0,
                generation_mode='local_task_list',
                generation_status='complete',
                generation_epoch=1,
                materialized_cursor='3',
                materialize_cursor_bytes=3,
                materialized_count=3,
                owner_principal_id='local-dev',
                total_items=3,
                submitted_items=3,
                active_items=0,
                terminal_items=3,
                submission_closed=True,
                control_state='active',
                summary_version=1,
                last_progress_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        for ordinal, item_id, status, child_execution_id, error_code in (
            (0, 'item-001', 'failed', 'child-alpha', 'ModelError'),
            (1, 'item-002', 'succeeded', 'child-beta', None),
            (2, 'item-003', 'failed', 'child-gamma', 'RuntimeError'),
        ):
            session.add(
                BatchItemRow(
                    parent_execution_id=parent_execution_id,
                    item_id=item_id,
                    task_ordinal=ordinal,
                    task_offset=ordinal,
                    task_length=1,
                    current_attempt_no=1,
                    current_child_execution_id=child_execution_id,
                    spec_sha256=str(ordinal) * 64,
                    status=status,
                    status_version=1,
                    child_execution_status=status,
                    metadata_json={},
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                BatchItemAttemptRow(
                    parent_execution_id=parent_execution_id,
                    item_id=item_id,
                    attempt_no=1,
                    child_execution_id=child_execution_id,
                    spec_sha256=str(ordinal) * 64,
                    status=status,
                    status_version=1,
                    error_code=error_code,
                    created_at=now,
                    updated_at=now,
                )
            )
        await session.commit()

    app = create_control_app(session_factory=session_factory, config=DistributedConfig(shard_count=8))
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get(
            f'/v1/executions/{parent_execution_id}/batch/items',
            params={
                'status': 'failed',
                'item_id_prefix': 'item-00',
                'child_execution_id': 'child-alpha',
                'error_code': 'ModelError',
            },
        )
        assert response.status == 200
        payload = await response.json()
        assert [item['item_id'] for item in payload['items']] == ['item-001']
        assert payload['next_cursor'] is None
    finally:
        await client.close()


async def test_control_gateway_persists_trace_carrier_without_exposing_it(
    session_factory, monkeypatch, otel_hyway_instrumentor
):
    trace_carrier = {
        'traceparent': '00-11111111111111111111111111111111-2222222222222222-01',
        'tracestate': 'vendor=value',
    }
    monkeypatch.setattr('nexus.distributed.repositories.execution.inject_carrier', lambda: trace_carrier)
    app = create_control_app(session_factory=session_factory, config=DistributedConfig(shard_count=8))
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            '/v1/executions',
            json={
                'execution_id': 'trace-carrier-create',
                'mode': ExecutionMode.JOB,
                'spec_type': 'nexus_single',
                'instance_mode': 'asyncio',
                'spec': {'task_name': 'identity'},
            },
        )

        assert response.status == 201
        body = await response.json()
        assert 'trace_carrier' not in body
        assert 'trace_carrier_json' not in body
        async with session_factory() as session:
            row = await session.get(ExecutionRow, 'trace-carrier-create')
        assert row is not None
        assert row.trace_carrier_json == trace_carrier
    finally:
        await client.close()


async def test_control_gateway_rejects_non_positive_slot_cost(session_factory):
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    app = create_control_app(session_factory=session_factory, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        for slot_cost in (0, -1):
            response = await client.post(
                '/v1/executions',
                json={
                    'execution_id': f'invalid-slot-{slot_cost}',
                    'mode': ExecutionMode.JOB,
                    'spec_type': 'nexus_single',
                    'instance_mode': 'asyncio',
                    'spec': {'task_name': 'identity'},
                    'slot_cost': slot_cost,
                },
            )
            assert response.status == 422
            assert 'slot_cost' in await response.text()
    finally:
        await client.close()


async def test_disabled_auth_admin_can_update_local_dev_roles(session_factory):
    app = create_control_app(session_factory=session_factory, config=DistributedConfig(shard_count=8))
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.put(
            '/v1/access/users/local-dev/roles',
            json={
                'issuer': 'local',
                'tenant': 'local',
                'subject': 'local-dev',
                'roles': ['admin', 'user'],
            },
        )

        assert response.status == 200, await response.text()
        payload = await response.json()
        assert payload['principal_id'] == 'local-dev'
        assert payload['roles'] == ['admin', 'user']

        response = await client.put(
            '/v1/access/users/local-dev/roles',
            json={
                'issuer': 'local',
                'tenant': 'local',
                'subject': 'local-dev',
                'roles': ['submit_all', 'user'],
            },
        )
        assert response.status == 422
        assert (await response.json())['error'] == 'invalid_roles'
    finally:
        await client.close()


async def test_control_gateway_rejects_unsafe_execution_id(session_factory):
    app = create_control_app(session_factory=session_factory, config=DistributedConfig(shard_count=8))
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        for execution_id in ('', '.', '..', 'a/b', 'a\\b', 'a\x00b', 'a' * 65):
            response = await client.post(
                '/v1/executions',
                json={
                    'execution_id': execution_id,
                    'mode': 'job',
                    'spec_type': 'nexus_single',
                    'spec': {'task_name': 'identity'},
                },
            )
            assert response.status == 422
            assert 'execution_id' in await response.text()
    finally:
        await client.close()


async def test_control_gateway_rejects_non_positive_max_attempts(session_factory):
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    app = create_control_app(session_factory=session_factory, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        for max_attempts in (0, -1):
            response = await client.post(
                '/v1/executions',
                json={
                    'execution_id': f'invalid-attempts-{max_attempts}',
                    'mode': ExecutionMode.JOB,
                    'spec_type': 'nexus_single',
                    'instance_mode': 'asyncio',
                    'spec': {'task_name': 'identity'},
                    'max_attempts': max_attempts,
                },
            )
            assert response.status == 422
            assert 'max_attempts' in await response.text()
    finally:
        await client.close()


async def test_control_gateway_public_routes_are_under_v1(session_factory):
    app = create_control_app(
        session_factory=session_factory,
        config=DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15)),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/health')
        assert response.status == 200
        assert await response.json() == {'status': 'ok'}

        response = await client.get('/health')
        assert response.status == 404

        response = await client.get('/v1/console?applicationBaseUrl=http://127.0.0.1:8082')
        assert response.status == 200
        assert response.content_type == 'text/html'
        text = await response.text()
        assert 'window.__NEXUS_DISTRIBUTED_CONSOLE__' in text

        response = await client.get('/v1/console/batches/batch-1/items/item-1')
        assert response.status == 200
        assert response.content_type == 'text/html'
        assert '<div id="root"></div>' in await response.text()

        response = await client.get(
            '/v1/console',
            params={'applicationBaseUrl': '</script><script>alert(1)</script>'},
        )
        text = await response.text()
        assert '</script><script>alert(1)</script>' not in text
        assert '\\u003c/script\\u003e' in text

        asset_match = re.search(r'src="(/v1/console/assets/[^"]+\.js)"', CONSOLE_INDEX.read_text(encoding='utf-8'))
        assert asset_match is not None
        response = await client.get(asset_match.group(1))
        assert response.status == 200
        assert response.content_type in {'application/javascript', 'text/javascript'}

        response = await client.get('/console')
        assert response.status == 404
    finally:
        await client.close()


async def test_control_gateway_control_events_cursor(session_factory):
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    app = create_control_app(session_factory=session_factory, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            '/v1/executions',
            json={
                'execution_id': 'event-cursor',
                'mode': ExecutionMode.JOB,
                'spec_type': 'nexus_single',
                'instance_mode': 'asyncio',
                'spec': {'task_name': 'identity'},
            },
        )
        assert response.status == 201

        response = await client.get('/v1/executions/event-cursor/control-events?cursor=0')
        assert response.status == 200
        first_page = await response.json()
        assert first_page['items'][0]['seq'] == 1

        response = await client.get('/v1/executions/event-cursor/control-events?cursor=1')
        assert response.status == 200
        assert (await response.json())['items'] == []
    finally:
        await client.close()


async def test_control_gateway_control_events_query_bounds(session_factory):
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    app = create_control_app(session_factory=session_factory, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            '/v1/executions',
            json={
                'execution_id': 'event-bounds',
                'mode': ExecutionMode.LIVE,
                'spec_type': 'nexus_live',
                'instance_mode': 'asyncio',
                'spec': {'task_name': 'identity'},
                'route_status': RouteStatus.CACHED,
            },
        )
        assert response.status == 201

        response = await client.post('/v1/executions/event-bounds/activate')
        assert response.status == 200

        response = await client.get('/v1/executions/event-bounds/control-events?cursor=-10&limit=0')
        assert response.status == 200
        payload = await response.json()
        assert [item['seq'] for item in payload['items']] == [1]

        response = await client.get('/v1/executions/event-bounds/control-events?cursor=not-int')
        assert response.status == 400

        response = await client.get('/v1/executions/event-bounds/control-events?limit=not-int')
        assert response.status == 400
    finally:
        await client.close()


async def test_control_gateway_lists_executions_by_status(session_factory):
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    app = create_control_app(session_factory=session_factory, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        for execution_id in ('job-list-1', 'job-list-2'):
            response = await client.post(
                '/v1/executions',
                json={
                    'execution_id': execution_id,
                    'mode': ExecutionMode.JOB,
                    'spec_type': 'nexus_single',
                    'instance_mode': 'asyncio',
                    'spec': {'task_name': 'identity'},
                },
            )
            assert response.status == 201

        response = await client.get('/v1/executions?status=queued')
        assert response.status == 200
        payload = await response.json()
        assert [item['execution_id'] for item in payload['items']] == ['job-list-2', 'job-list-1']
        assert [item['phase'] for item in payload['items']] == ['queued', 'queued']
    finally:
        await client.close()


async def test_control_gateway_lists_executions_with_offset_total_and_query(session_factory):
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    app = create_control_app(session_factory=session_factory, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        for execution_id in ('page-alpha-1', 'page-alpha-2', 'page-alpha-3', 'page-beta-1'):
            response = await client.post(
                '/v1/executions',
                json={
                    'execution_id': execution_id,
                    'mode': ExecutionMode.JOB,
                    'spec_type': 'nexus_single',
                    'instance_mode': 'asyncio',
                    'spec': {'task_name': 'identity'},
                },
            )
            assert response.status == 201

        now = utcnow()
        async with session_factory() as session:
            for execution_id, seconds_ago in (
                ('page-alpha-1', 10),
                ('page-alpha-2', 30),
                ('page-alpha-3', 20),
            ):
                row = await session.get(ExecutionRow, execution_id)
                assert row is not None
                row.updated_at = now - timedelta(seconds=seconds_ago)
            await session.commit()

        response = await client.get('/v1/executions?summary=1&q=page-alpha&limit=2&offset=1')
        assert response.status == 200
        payload = await response.json()

        assert payload['total'] == 3
        assert payload['limit'] == 2
        assert payload['offset'] == 1
        assert [item['execution_id'] for item in payload['items']] == ['page-alpha-3', 'page-alpha-2']
    finally:
        await client.close()


async def test_control_gateway_pages_top_level_executions_with_stable_cursor(session_factory):
    now = utcnow()
    async with session_factory() as session:
        repository = ExecutionRepository(session, shard_count=8)
        for execution_id, spec_type, parent_execution_id in (
            ('run-cursor-c', 'nexus_single', None),
            ('run-cursor-b', 'nexus_batch', None),
            ('run-cursor-a', 'nexus_single', None),
            ('run-cursor-child', 'nexus_single', 'run-cursor-b'),
        ):
            await repository.create_execution(
                CreateExecution(
                    execution_id=execution_id,
                    mode=ExecutionMode.JOB,
                    spec_type=spec_type,
                    instance_mode='asyncio',
                    spec={'task_name': 'identity'},
                    owner_principal_id='local-dev',
                    created_by='system:batch-parent' if parent_execution_id else 'local-dev',
                    parent_execution_id=parent_execution_id,
                ),
                now=now,
            )
        await session.commit()

    app = create_control_app(
        session_factory=session_factory,
        config=DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15)),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/executions?summary=1&top_level=1&limit=2')
        assert response.status == 200
        first = await response.json()

        assert [item['execution_id'] for item in first['items']] == ['run-cursor-c', 'run-cursor-b']
        assert first['next_cursor']
        assert 'total' not in first
        assert all(item['parent_execution_id'] is None for item in first['items'])

        response = await client.get(
            '/v1/executions',
            params={'summary': '1', 'top_level': '1', 'limit': '2', 'cursor': first['next_cursor']},
        )
        assert response.status == 200
        second = await response.json()
        assert [item['execution_id'] for item in second['items']] == ['run-cursor-a']
        assert second['next_cursor'] is None

        response = await client.get('/v1/executions?summary=1&top_level=1&spec_type=nexus_batch')
        assert response.status == 200
        batches = await response.json()
        assert [item['execution_id'] for item in batches['items']] == ['run-cursor-b']
    finally:
        await client.close()


async def test_control_gateway_lists_execution_summaries_without_large_payloads(session_factory):
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    large_text = 'x' * 50_000
    async with session_factory() as session:
        manager = ManagerRepository(session, healthy_ttl=timedelta(seconds=15))
        await manager.register_session(
            manager_id='im-summary',
            session_id='session-summary',
            endpoint_url='http://im-summary.internal:9001',
            supported_modes=[ExecutionMode.LIVE],
            supported_spec_types=['nexus_live'],
            supported_instance_modes=['asyncio'],
            capacity_total=1,
            force_replace=False,
            now=utcnow(),
        )
        execution = await ExecutionRepository(session, shard_count=8).create_execution(
            CreateExecution(
                execution_id='summary-live',
                mode=ExecutionMode.LIVE,
                spec_type='nexus_live',
                instance_mode='asyncio',
                spec={'task_config': {'prompt': large_text}},
                route_status=RouteStatus.ACTIVE,
            ),
            now=utcnow(),
        )
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        assert assignment is not None
        await manager.ack_assignment(assignment.assignment_id, 'im-summary', 'session-summary', now=utcnow())
        await manager.mark_running(
            assignment.assignment_id,
            'im-summary',
            'session-summary',
            endpoint_url='http://im-summary.internal:9001/v1/internal/executions/summary-live',
            endpoint_ready=True,
            runtime_info={'input': {'request': {'hint': large_text}}},
            now=utcnow(),
        )
        await session.commit()

    statements = []
    engine = session_factory.kw['bind']

    def capture(_conn, _cursor, statement, _params, _context, _many):
        statements.append(statement.lower())

    event.listen(engine.sync_engine, 'before_cursor_execute', capture)
    app = create_control_app(session_factory=session_factory, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/executions?summary=1')
        assert response.status == 200
        item = (await response.json())['items'][0]

        assert item['execution_id'] == 'summary-live'
        assert item['phase'] == 'running'
        assert item['current_assignment']['manager_id'] == 'im-summary'
        assert re.match(r'\d{4}-\d{2}-\d{2}T', item['created_at'])
        assert re.match(r'\d{4}-\d{2}-\d{2}T', item['updated_at'])
        assert re.match(r'\d{4}-\d{2}-\d{2}T', item['current_assignment']['created_at'])
        assert re.match(r'\d{4}-\d{2}-\d{2}T', item['current_assignment']['updated_at'])
        assert 'spec' not in item
        assert 'driver_config' not in item
        assert 'result_json' not in item
        assert 'error_json' not in item
        assert 'runtime_info' not in item['current_assignment']

        single_response = await client.get('/v1/executions/summary-live?summary=1')
        assert single_response.status == 200
        single = await single_response.json()
        assert single['status'] == 'running'
        assert single['phase'] == 'running'
        assert 'spec' not in single
        assert 'placement' not in single
        for statement in statements:
            if statement.lstrip().startswith('select'):
                projection = statement.split('from', 1)[0]
                for heavy in (
                    'spec_json',
                    'driver_config_json',
                    'result_json',
                    'error_json',
                    'runtime_info_json',
                    'placement_json',
                ):
                    assert heavy not in projection
        event.remove(engine.sync_engine, 'before_cursor_execute', capture)

        detail_response = await client.get('/v1/executions/summary-live')
        assert detail_response.status == 200
        detail = await detail_response.json()
        assert detail['phase'] == 'running'
        assert detail['spec']['task_config']['prompt'] == large_text
        assert re.match(r'\d{4}-\d{2}-\d{2}T', detail['created_at'])
        assert re.match(r'\d{4}-\d{2}-\d{2}T', detail['updated_at'])
        assert re.match(r'\d{4}-\d{2}-\d{2}T', detail['current_assignment']['created_at'])
        assert re.match(r'\d{4}-\d{2}-\d{2}T', detail['current_assignment']['updated_at'])
    finally:
        await client.close()


async def test_control_gateway_normalizes_legacy_assignment_runtime_info(session_factory):
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    runtime = {
        'name': 'main',
        'runtime_id': 'rt-main',
        'proxy_url': '/v1/executions/legacy-runtime/runtimes/main/proxy',
        'debug_gui_url': '/v1/executions/legacy-runtime/runtimes/main/proxy/debug/gui',
    }
    input_payload = {'turn_id': 'turn-1', 'request': {'hint': 'choose'}}
    artifact = {'events_path': '/data/events.jsonl'}
    async with session_factory() as session:
        manager = ManagerRepository(session, healthy_ttl=timedelta(seconds=15))
        await manager.register_session(
            manager_id='im-legacy-runtime',
            session_id='session-legacy-runtime',
            endpoint_url='http://im-legacy-runtime.internal:9001',
            supported_modes=[ExecutionMode.LIVE],
            supported_spec_types=['nexus_live'],
            supported_instance_modes=['asyncio'],
            capacity_total=1,
            force_replace=False,
            now=utcnow(),
        )
        execution = await ExecutionRepository(session, shard_count=8).create_execution(
            CreateExecution(
                execution_id='legacy-runtime',
                mode=ExecutionMode.LIVE,
                spec_type='nexus_live',
                instance_mode='asyncio',
                spec={'task_name': 'identity'},
                route_status=RouteStatus.ACTIVE,
            ),
            now=utcnow(),
        )
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        assert assignment is not None
        row = await session.get(AssignmentRow, assignment.assignment_id)
        assert row is not None
        row.runtime_info_json = {
            'main': runtime,
            'input': input_payload,
            '_artifacts': {'live': artifact},
        }
        await session.commit()

    app = create_control_app(session_factory=session_factory, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/executions/legacy-runtime')
        assert response.status == 200
        detail = await response.json()
    finally:
        await client.close()

    assert detail['current_assignment']['runtime_info'] == {
        'runtimes': {'main': runtime},
        'input': {'turn_id': 'turn-1'},
        '_artifacts': {'live': artifact},
    }


async def test_control_gateway_system_status_reads_collected_groups(session_factory, monkeypatch):
    now = utcnow()
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    async with session_factory() as session:
        manager = ManagerRepository(session, healthy_ttl=timedelta(seconds=15))
        await manager.register_session(
            manager_id='im-system-a',
            session_id='session-system-a',
            endpoint_url='http://im-system-a.internal:9001',
            supported_modes=[ExecutionMode.LIVE],
            supported_spec_types=['nexus_live'],
            supported_instance_modes=['asyncio'],
            capacity_total=3,
            environment_id='predefined-system',
            pool_id='interactive',
            force_replace=False,
            now=now,
        )
        await manager.register_session(
            manager_id='im-system-stale',
            session_id='session-system-stale',
            endpoint_url='http://im-system-stale.internal:9001',
            supported_modes=[ExecutionMode.LIVE],
            supported_spec_types=['nexus_live'],
            supported_instance_modes=['asyncio'],
            capacity_total=2,
            environment_id='predefined-system',
            pool_id='interactive',
            force_replace=False,
            now=now - timedelta(minutes=5),
        )
        await ExecutionRepository(session, shard_count=8).create_execution(
            CreateExecution(
                execution_id='system-queued',
                mode=ExecutionMode.LIVE,
                spec_type='nexus_live',
                instance_mode='asyncio',
                spec={},
            ),
            now=now,
        )
        await ExecutionRepository(session, shard_count=8).create_execution(
            CreateExecution(
                execution_id='system-active',
                mode=ExecutionMode.JOB,
                spec_type='nexus_single',
                instance_mode='asyncio',
                spec={},
            ),
            now=now,
        )
        active = await session.get(ExecutionRow, 'system-active')
        active.status = 'running'
        session.add(
            ShardLeaseRow(
                shard_group='execution-scheduler',
                shard_id=3,
                shard_count=8,
                owner_id='scheduler-a',
                lease_expires_at=now + timedelta(seconds=20),
                updated_at=now,
            )
        )
        session.add_all(
            [
                DependencyBuildRow(
                    dependency_build_id='build-system-building',
                    build_key='c' * 64,
                    owner_principal_id='local-dev',
                    environment_id='dynamic-system',
                    status='building',
                    build_spec_json={},
                    attempt=1,
                    created_at=now,
                    updated_at=now,
                ),
                EnvironmentAllocationRow(
                    allocation_id='allocation-system-active',
                    top_level_execution_id='system-active',
                    owner_principal_id='local-dev',
                    environment_id='dynamic-system',
                    pool_id='execution',
                    workload='execution',
                    status='ready',
                    desired_manager_count=2,
                    ready_manager_count=1,
                    manager_count=2,
                    cpu_millicores=2000,
                    memory_bytes=2048,
                    slots=2,
                    active_batches=0,
                    allocation_spec_json={},
                    allocation_spec_hash='a' * 64,
                    config_version=1,
                    status_version=1,
                    last_progress_at=now,
                    created_at=now,
                    updated_at=now,
                ),
                EnvironmentAllocationRow(
                    allocation_id='allocation-system-released',
                    top_level_execution_id='system-released',
                    owner_principal_id='local-dev',
                    environment_id='dynamic-system',
                    pool_id='execution',
                    workload='execution',
                    status='released',
                    desired_manager_count=50,
                    ready_manager_count=0,
                    manager_count=50,
                    cpu_millicores=50000,
                    memory_bytes=51200,
                    slots=50,
                    active_batches=0,
                    allocation_spec_json={},
                    allocation_spec_hash='b' * 64,
                    config_version=1,
                    status_version=2,
                    last_progress_at=now,
                    resources_released_at=now,
                    created_at=now,
                    released_at=now,
                    updated_at=now,
                ),
            ]
        )
        await session.commit()

    from nexus.distributed.observability.sampler import StatusSampler
    from nexus.distributed.repositories.observability import ObservabilityRepository

    sampler = StatusSampler(config, session_factory=session_factory, clock=lambda: now)
    assert await sampler.tick() == 8
    await sampler.close()

    async def no_collection_in_request(*args, **kwargs):
        raise AssertionError('snapshot requests must not recompute business state')

    monkeypatch.setattr(ObservabilityRepository, 'collect_group', no_collection_in_request)
    app = create_control_app(session_factory=session_factory, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/system/status')
        assert response.status == 200
        status = await response.json()

        assert status['source'] == 'snapshot'
        assert status['data_status'] == 'ready'
        groups = status['groups']
        assert groups['queues']['values']['items'][0]['schedulable'] == 1
        managers = groups['inventory']['values']['managers']
        assert managers['healthy'] == 1
        assert managers['capacity_total'] == 3
        assert managers['available_slots'] == 3
        leases = groups['leases']['values']['items']
        assert leases == [
            {'component': 'execution-reconciler', 'total': 8, 'active': 0, 'unowned': 8},
            {'component': 'execution-scheduler', 'total': 8, 'active': 1, 'unowned': 7},
        ] or leases == [
            {'component': 'execution-scheduler', 'total': 8, 'active': 1, 'unowned': 7},
            {'component': 'execution-reconciler', 'total': 8, 'active': 0, 'unowned': 8},
        ]
        allocations = groups['allocations']['values']['items']
        assert len(allocations) == 1
        assert allocations[0]['state'] == 'ready'
        assert allocations[0]['desired'] == 2
        assert allocations[0]['ready'] == 0
        assert allocations[0]['manager_gap'] == 2
        assert groups['builds']['values']['items'][0]['count'] == 1
        assert all(group['sampled_at'] == now.isoformat() for group in groups.values())
    finally:
        await client.close()


async def test_database_runtime_metrics_normalizes_mysql_status_without_connection_details():
    class FakeResult:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return self._rows

    class FakeSession:
        def __init__(self):
            self.statements = []

        def get_bind(self):
            return SimpleNamespace(dialect=SimpleNamespace(name='mysql'))

        async def execute(self, statement):
            sql = str(statement)
            self.statements.append(sql)
            if 'SHOW VARIABLES' in sql:
                return FakeResult([('max_connections', '2000')])
            return FakeResult(
                [
                    ('Threads_connected', '120'),
                    ('Threads_running', '7'),
                    ('Max_used_connections', '321'),
                    ('Slow_queries', '4'),
                    ('Aborted_connects', '9'),
                ]
            )

    session = FakeSession()
    metrics = await _database_runtime_metrics(session)

    assert metrics == {
        'available': True,
        'dialect': 'mysql',
        'max_connections': 2000,
        'threads_connected': 120,
        'threads_running': 7,
        'max_used_connections': 321,
        'slow_queries': 4,
        'aborted_connects': 9,
    }
    assert len(session.statements) == 2
    assert all('11.142.' not in statement for statement in session.statements)


async def test_control_gateway_system_summary_does_not_load_manager_inventory(session_factory, monkeypatch):
    now = utcnow()
    async with session_factory() as session:
        manager = ManagerRepository(session, healthy_ttl=timedelta(seconds=15))
        await manager.register_session(
            manager_id='summary-manager-healthy',
            session_id='session-healthy',
            endpoint_url='http://manager.internal:9001',
            supported_modes=[ExecutionMode.JOB],
            supported_spec_types=['nexus_single'],
            supported_instance_modes=['asyncio'],
            capacity_total=20,
            force_replace=False,
            now=now,
        )
        await manager.register_session(
            manager_id='summary-manager-lost',
            session_id='session-lost',
            endpoint_url='http://manager.internal:9001',
            supported_modes=[ExecutionMode.JOB],
            supported_spec_types=['nexus_single'],
            supported_instance_modes=['asyncio'],
            capacity_total=20,
            force_replace=False,
            now=now - timedelta(minutes=5),
        )
        await session.commit()

    async def fail_if_loaded(*_args, **_kwargs):
        raise AssertionError('system summary must not load every Manager row')

    monkeypatch.setattr(EnvironmentInventoryRepository, 'list_managers', fail_if_loaded)
    app = create_control_app(
        session_factory=session_factory,
        config=DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15)),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/system/summary')
        assert response.status == 200
        payload = await response.json()
        assert payload['source'] == 'live'
        assert payload['managers'] == {
            'healthy': 1,
            'draining': 0,
            'capacity_total': 20,
            'reserved_slots': 0,
            'available_slots': 20,
        }
        assert payload['queried_at']
        assert payload['cache_age_seconds'] <= 1
        assert 'scheduler' not in payload
        assert 'total' not in payload['managers']
    finally:
        await client.close()


async def test_control_gateway_pages_allocations_without_exact_total(session_factory):
    now = utcnow()
    async with session_factory() as session:
        for index, allocation_id in enumerate(('allocation-cursor-a', 'allocation-cursor-b')):
            session.add(
                EnvironmentAllocationRow(
                    allocation_id=allocation_id,
                    top_level_execution_id=f'execution-{index}',
                    owner_principal_id='local-dev',
                    environment_id='environment-cursor',
                    pool_id='execution',
                    workload='execution',
                    status='ready',
                    desired_manager_count=1,
                    ready_manager_count=1,
                    manager_count=1,
                    cpu_millicores=1000,
                    memory_bytes=1024,
                    slots=1,
                    active_batches=0,
                    allocation_spec_json={},
                    allocation_spec_hash=str(index) * 64,
                    config_version=1,
                    status_version=1,
                    last_progress_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
        await session.commit()

    app = create_control_app(session_factory=session_factory, config=DistributedConfig(shard_count=8))
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/allocations?limit=1')
        assert response.status == 200
        first = await response.json()
        assert [item['allocation_id'] for item in first['items']] == ['allocation-cursor-b']
        assert first['next_cursor']
        assert 'total' not in first

        response = await client.get('/v1/allocations', params={'limit': '1', 'cursor': first['next_cursor']})
        assert response.status == 200
        second = await response.json()
        assert [item['allocation_id'] for item in second['items']] == ['allocation-cursor-a']
        assert second['next_cursor'] is None
    finally:
        await client.close()


async def test_control_gateway_filters_allocations_by_environment_allocation_type(session_factory):
    now = utcnow()
    async with session_factory() as session:
        environments = EnvironmentRepository(session)
        await environments.create_environment(
            environment_id='environment-dynamic-filter',
            name='Dynamic filter',
            allocation_type='dynamic',
            provider_type='kubernetes',
            provider_config_ref='kubernetes/filter',
            resource_limits={},
            default_timeouts={},
            status='disabled',
            actor='admin',
            now=now,
        )
        await environments.create_environment(
            environment_id='environment-predefined-filter',
            name='Predefined filter',
            allocation_type='predefined',
            provider_type='predefined',
            provider_config_ref=None,
            resource_limits={},
            default_timeouts={},
            status='enabled',
            actor='admin',
            now=now,
        )
        for allocation_id, environment_id, allocation_type in (
            ('allocation-dynamic-filter', 'environment-dynamic-filter', 'dynamic'),
            ('allocation-predefined-filter', 'environment-predefined-filter', 'predefined'),
        ):
            session.add(
                EnvironmentAllocationRow(
                    allocation_id=allocation_id,
                    top_level_execution_id=f'execution-{allocation_type}',
                    owner_principal_id='local-dev',
                    environment_id=environment_id,
                    pool_id='execution',
                    workload='execution',
                    status='ready',
                    desired_manager_count=1,
                    ready_manager_count=1,
                    manager_count=1,
                    cpu_millicores=1000,
                    memory_bytes=1024,
                    slots=1,
                    active_batches=0,
                    allocation_spec_json={'allocation_type': allocation_type},
                    allocation_spec_hash=allocation_type[0] * 64,
                    config_version=1,
                    status_version=1,
                    last_progress_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
        await session.commit()

    app = create_control_app(session_factory=session_factory, config=DistributedConfig(shard_count=8))
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/allocations', params={'allocation_type': 'dynamic'})
        assert response.status == 200
        assert [item['allocation_id'] for item in (await response.json())['items']] == ['allocation-dynamic-filter']

        response = await client.get('/v1/allocations', params={'allocation_type': 'predefined'})
        assert response.status == 200
        assert [item['allocation_id'] for item in (await response.json())['items']] == ['allocation-predefined-filter']

        response = await client.get('/v1/allocations', params={'allocation_type': 'unsupported'})
        assert response.status == 400
    finally:
        await client.close()


async def test_control_gateway_lists_predefined_environment_inventory(session_factory):
    # Live stats run every SQL under a 1s budget by default; a loaded CI loop can stall past it.
    config = DistributedConfig(
        shard_count=8,
        manager_healthy_ttl=timedelta(seconds=15),
        status_sampling={'query_timeout_seconds': 5, 'group_timeout_seconds': 5},
    )
    now = utcnow()
    async with session_factory() as session:
        await EnvironmentRepository(session).create_environment(
            environment_id='environment-east',
            name='Predefined East',
            allocation_type='predefined',
            provider_type='predefined',
            provider_config_ref=None,
            resource_limits={},
            default_timeouts={},
            status='enabled',
            actor='admin',
            now=now,
        )
        manager = ManagerRepository(session)
        await manager.register_session(
            manager_id='im-env-a',
            session_id='session-a',
            endpoint_url='http://im-env-a.internal:9001',
            supported_modes=[ExecutionMode.JOB],
            supported_spec_types=['nexus_single'],
            supported_instance_modes=['asyncio'],
            capacity_total=2,
            environment_id='environment-east',
            pool_id='interactive',
            workload='execution',
            labels={'region': 'east'},
            metadata={'endpoint_kind': 'pod_ip', 'cluster_name': 'cls-east'},
            draining=False,
            force_replace=False,
            now=utcnow(),
        )
        await manager.register_session(
            manager_id='im-env-b',
            session_id='session-b',
            endpoint_url='http://im-env-b.internal:9001',
            supported_modes=[ExecutionMode.JOB],
            supported_spec_types=['nexus_single'],
            supported_instance_modes=['asyncio'],
            capacity_total=1,
            environment_id='environment-east',
            pool_id='batch',
            workload='execution',
            labels={'region': 'east'},
            draining=True,
            force_replace=False,
            now=utcnow(),
        )
        execution = await ExecutionRepository(session, shard_count=8).create_execution(
            CreateExecution(
                execution_id='env-reserved',
                mode=ExecutionMode.JOB,
                spec_type='nexus_single',
                instance_mode='asyncio',
                spec={},
                placement={'environment': 'environment-east', 'pool': 'interactive'},
            ),
            now=utcnow(),
        )
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        manager_row = await session.get(InstanceManagerRow, 'im-env-a')
        assert manager_row is not None
        manager_row.bootstrap_version = 1
        manager_row.artifact_ref_version = 1
        manager_row.supported_spec_versions_json = {'nexus_single': [1]}
        manager_row.allocation_id = 'allocation-env-a'
        manager_row.allocation_spec_hash = 'a' * 64
        manager_row.capability_snapshot_json = {'source_package': True}
        await session.commit()

    assert assignment is not None

    app = create_control_app(session_factory=session_factory, config=config)
    # The cache's own 0.45s budget wraps the whole query, so it is lifted too.
    app[LIVE_CACHE_KEY] = LiveStatusCache(ttl=app[LIVE_CACHE_KEY].ttl, query_timeout=30)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/environments/environment-east?pool_id=interactive')
        assert response.status == 200
        environment = await response.json()
        assert environment['pool_stats'] == {
            'environment_id': 'environment-east',
            'pool_id': 'interactive',
            'manager_count': 1,
            'healthy_manager_count': 1,
            'draining_manager_count': 0,
            'capacity_total': 2,
            'reserved_slots': 1,
            'available_slots': 1,
        }

        response = await client.get('/v1/environments/environment-east/stats?pool_id=interactive')
        assert response.status == 200
        stats = await response.json()
        assert stats['source'] == 'live'
        assert stats['scope'] == {'environment_id': 'environment-east', 'pool_id': 'interactive'}
        assert stats['managers'] == {
            'healthy': 1,
            'draining': 0,
            'capacity_total': 2,
            'reserved_slots': 1,
            'available_slots': 1,
        }
        assert stats['resource_usage']['scope_type'] == 'environment'
        assert stats['queried_at']

        response = await client.get(
            '/v1/environments/environment-east/managers?pool_id=interactive&label_key=region&label_value=east'
        )
        assert response.status == 200
        payload = await response.json()
        assert payload['items'] == [
            {
                'manager_id': 'im-env-a',
                'session_id': 'session-a',
                'endpoint_url': 'http://im-env-a.internal:9001',
                'environment_id': 'environment-east',
                'pool_id': 'interactive',
                'allocation_id': 'allocation-env-a',
                'workload': 'execution',
                'allocation_spec_hash': 'a' * 64,
                'owner_principal_id': None,
                'labels': {'region': 'east'},
                'metadata': {'endpoint_kind': 'pod_ip', 'cluster_name': 'cls-east'},
                'draining': False,
                'bootstrap_version': 1,
                'artifact_ref_version': 1,
                'supported_spec_versions': {'nexus_single': [1]},
                'capability_snapshot': {'source_package': True},
                'supported_modes': [ExecutionMode.JOB],
                'supported_spec_types': ['nexus_single'],
                'supported_instance_modes': ['asyncio'],
                'capacity_total': 2,
                'reserved_slots': 1,
                'available_slots': 1,
                'healthy': True,
                'observed_capacity_used': 0,
                'last_heartbeat_at': payload['items'][0]['last_heartbeat_at'],
                'created_at': payload['items'][0]['created_at'],
                'updated_at': payload['items'][0]['updated_at'],
            }
        ]
        assert re.match(r'\d{4}-\d{2}-\d{2}T', payload['items'][0]['created_at'])
        assert re.match(r'\d{4}-\d{2}-\d{2}T', payload['items'][0]['updated_at'])

        response = await client.get('/v1/environment-inventory')
        assert response.status == 200
        environments = await response.json()
        assert environments == {
            'items': [
                {
                    'environment_id': 'environment-east',
                    'name': 'Predefined East',
                    'allocation_type': 'predefined',
                    'provider_type': 'predefined',
                    'status': 'enabled',
                    'manager_count': 2,
                    'healthy_manager_count': 2,
                    'draining_manager_count': 1,
                    'lost_manager_count': 0,
                    'capacity_total': 3,
                    'reserved_slots': 1,
                    'available_slots': 1,
                    'pools': ['batch', 'interactive'],
                    'allocations': [],
                }
            ]
        }

        response = await client.get('/v1/managers?environment_id=environment-east')
        assert response.status == 200
        managers = await response.json()
        assert [item['manager_id'] for item in managers['items']] == ['im-env-b', 'im-env-a']
    finally:
        await client.close()


async def test_control_gateway_pages_managers_and_filters_health_on_server(session_factory):
    now = utcnow()
    async with session_factory() as session:
        repository = ManagerRepository(session, healthy_ttl=timedelta(seconds=15))
        for manager_id, heartbeat_at in (
            ('manager-a', now),
            ('manager-b', now),
            ('manager-lost', now - timedelta(minutes=5)),
        ):
            await repository.register_session(
                manager_id=manager_id,
                session_id=f'session-{manager_id}',
                endpoint_url=f'http://{manager_id}.internal:9001',
                supported_modes=[ExecutionMode.JOB],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                capacity_total=20,
                force_replace=False,
                now=heartbeat_at,
            )
        await session.commit()

    app = create_control_app(
        session_factory=session_factory,
        config=DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15)),
    )
    # The summary is served through a cache that abandons a query after 450ms and answers
    # 503 rather than waiting -- a deliberate degradation, and not what this test is about:
    # it asserts the counts. The first request cannot fall back on a cached value, so under
    # the load of a full test run that budget is reachable, and the 503 then reads as a
    # gateway fault rather than as a slow machine.
    #
    # Only the cache's own budget is lifted. The handler still gives up after 3s, so the
    # request cannot hang here, and the endpoint's timeout behaviour stays exercised
    # everywhere it is not the machine being measured. The real ttl is kept, so caching is
    # unchanged and only the timing is relaxed.
    app[LIVE_CACHE_KEY] = LiveStatusCache(ttl=app[LIVE_CACHE_KEY].ttl, query_timeout=30)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/managers?health=healthy&limit=1')
        assert response.status == 200
        first = await response.json()
        assert [item['manager_id'] for item in first['items']] == ['manager-a']
        assert first['next_cursor'] == 'manager-a'

        response = await client.get(
            '/v1/managers', params={'health': 'healthy', 'limit': '1', 'cursor': first['next_cursor']}
        )
        assert response.status == 200
        second = await response.json()
        assert [item['manager_id'] for item in second['items']] == ['manager-b']
        assert second['next_cursor'] is None

        response = await client.get('/v1/managers?health=lost&limit=100')
        assert response.status == 200
        lost = await response.json()
        assert [item['manager_id'] for item in lost['items']] == ['manager-lost']
        response = await client.get('/v1/managers/summary')
        assert response.status == 200
        totals = await response.json()
        assert totals['total'] == 3
        assert totals['healthy'] == 2
        assert totals['lost'] == 1
        assert totals['reserved_slots'] == 0
        detail = await client.get('/v1/managers/manager-a')
        assert detail.status == 200
        assert (await detail.json())['session_id'] == 'session-manager-a'
        assert (await client.get('/v1/managers/manager-a/logs?session_id=old')).status == 409
    finally:
        await client.close()


async def test_control_gateway_projects_dynamic_allocation_status_into_environment_inventory(session_factory):
    now = utcnow()
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    async with session_factory() as session:
        await EnvironmentRepository(session).create_environment(
            environment_id='environment-dynamic',
            name='Dynamic East',
            allocation_type='dynamic',
            provider_type='kubernetes',
            provider_config_ref='kubernetes/east',
            resource_limits={},
            default_timeouts={},
            status='enabled',
            actor='admin',
            now=now,
        )
        session.add(
            EnvironmentAllocationRow(
                allocation_id='allocation-dynamic',
                top_level_execution_id='execution-dynamic',
                owner_principal_id='local-dev',
                environment_id='environment-dynamic',
                pool_id='execution',
                workload='execution',
                status='provisioning',
                desired_manager_count=3,
                ready_manager_count=2,
                manager_count=3,
                cpu_millicores=3000,
                memory_bytes=3 * 1024**3,
                slots=60,
                active_batches=0,
                provider_resource_ref_json={'kind': 'StatefulSet', 'name': 'nexus-im-dynamic'},
                allocation_spec_json={},
                allocation_spec_hash='a' * 64,
                config_version=1,
                status_version=2,
                last_error_code='manager_not_ready',
                last_error_message='one manager is pending',
                last_progress_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()

    app = create_control_app(session_factory=session_factory, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/environment-inventory')
        assert response.status == 200
        payload = await response.json()
        assert payload == {
            'items': [
                {
                    'environment_id': 'environment-dynamic',
                    'name': 'Dynamic East',
                    'allocation_type': 'dynamic',
                    'provider_type': 'kubernetes',
                    'status': 'enabled',
                    'manager_count': 0,
                    'healthy_manager_count': 0,
                    'draining_manager_count': 0,
                    'lost_manager_count': 0,
                    'capacity_total': 0,
                    'reserved_slots': 0,
                    'available_slots': 0,
                    'pools': ['execution'],
                    'allocations': [
                        {
                            'allocation_id': 'allocation-dynamic',
                            'pool_id': 'execution',
                            'workload': 'execution',
                            'status': 'provisioning',
                            'desired_manager_count': 3,
                            'ready_manager_count': 2,
                            'manager_count': 3,
                            'provider_resource_ref': {'kind': 'StatefulSet', 'name': 'nexus-im-dynamic'},
                            'last_error_code': 'manager_not_ready',
                            'last_error_message': 'one manager is pending',
                            'updated_at': now.isoformat(),
                        }
                    ],
                }
            ]
        }
    finally:
        await client.close()


async def test_control_gateway_validates_provider_registry_and_preserves_patch_unset_semantics(
    session_factory,
):
    config = DistributedConfig(
        environment_namespace='nexus-workloads',
        environment_enabled_providers=['kubernetes', 'tkex'],
    )
    app = create_control_app(session_factory=session_factory, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    base_payload = {
        'name': 'Dynamic East',
        'allocation_type': 'dynamic',
        'provider_type': 'kubernetes',
        'resource_limits': {},
        'default_timeouts': {},
        'status': 'disabled',
    }
    try:
        for provider_config_ref, provider_config in (
            (
                'kubernetes/east',
                {
                    'provider_type': 'kubernetes',
                    'namespace': 'nexus-workloads',
                    'auth_type': 'in_cluster',
                },
            ),
            (
                'tkex/east',
                {
                    'provider_type': 'tkex',
                    'namespace': 'nexus-workloads',
                    'base_url': 'https://tkex-east.example',
                    'api_account': 'account',
                    'credential_ref': 'tkex-east',
                    'cluster_name': 'cluster-east',
                    'project_name': 'project-east',
                },
            ),
            (
                'kubernetes/wrong-namespace',
                {
                    'provider_type': 'kubernetes',
                    'namespace': 'other-workloads',
                    'auth_type': 'in_cluster',
                },
            ),
        ):
            request_payload = {
                'provider_config_ref': provider_config_ref,
                'config': provider_config,
                'status': 'enabled',
            }
            if provider_config['provider_type'] == 'tkex':
                request_payload['credential'] = {
                    'provider_type': 'tkex',
                    'token': 'tkex-east-test-token',
                }
            response = await client.post(
                '/v1/provider-configs',
                json=request_payload,
            )
            assert response.status == 201

        response = await client.post(
            '/v1/environments',
            json={
                **base_payload,
                'environment_id': 'env-unknown-limit',
                'provider_config_ref': 'kubernetes/east',
                'resource_limits': {'unknown_limit': 1},
            },
        )
        assert response.status == 422
        assert 'unknown_limit' in await response.text()

        response = await client.post(
            '/v1/environments',
            json={
                **base_payload,
                'environment_id': 'env-timeout-conflict',
                'provider_config_ref': 'kubernetes/east',
                'resource_limits': {'max_queue_ttl_seconds': 10},
                'default_timeouts': {'queue_ttl_seconds': 11},
            },
        )
        assert response.status == 422
        assert 'queue_ttl_seconds_exceeds_limit' in await response.text()

        for environment_id, provider_ref, expected_error in (
            ('env-typo', 'kubernetes/typo', 'environment_provider_config_not_found'),
            ('env-type', 'tkex/east', 'environment_provider_config_type_mismatch'),
            ('env-namespace', 'kubernetes/wrong-namespace', 'environment_provider_config_namespace_mismatch'),
        ):
            response = await client.post(
                '/v1/environments',
                json={
                    **base_payload,
                    'environment_id': environment_id,
                    'provider_config_ref': provider_ref,
                },
            )
            assert response.status == 409
            assert expected_error in (await response.json())['error']

        response = await client.post(
            '/v1/environments',
            json={
                **base_payload,
                'environment_id': 'env-valid',
                'provider_config_ref': 'kubernetes/east',
            },
        )
        assert response.status == 201

        response = await client.patch('/v1/environments/env-valid', json={'name': 'Renamed Environment'})
        assert response.status == 200
        assert (await response.json())['name'] == 'Renamed Environment'

        response = await client.patch('/v1/environments/env-valid', json={'status': 'enabled'})
        assert response.status == 200
        assert (await response.json())['provider_config_ref'] == 'kubernetes/east'

        response = await client.patch(
            '/v1/environments/env-valid',
            json={
                'allocation_type': 'predefined',
                'provider_type': 'predefined',
                'provider_config_ref': None,
            },
        )
        assert response.status == 200
        payload = await response.json()
        assert payload['allocation_type'] == 'predefined'
        assert payload['provider_type'] == 'predefined'
        assert payload['provider_config_ref'] is None
    finally:
        await client.close()


async def test_control_gateway_manages_database_provider_configs_and_environment_refs(session_factory):
    app = create_control_app(
        session_factory=session_factory,
        config=DistributedConfig(
            environment_namespace='nexus-workloads',
            environment_enabled_providers=['kubernetes'],
        ),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    provider_payload = {
        'provider_config_ref': 'kubernetes/devcloud',
        'config': {
            'provider_type': 'kubernetes',
            'namespace': 'nexus-workloads',
            'auth_type': 'in_cluster',
            'manager_resource_request_ratio': 1.0,
        },
        'status': 'enabled',
    }
    try:
        response = await client.post('/v1/provider-configs', json=provider_payload)
        assert response.status == 201
        created = await response.json()
        assert created['provider_config_ref'] == 'kubernetes/devcloud'
        assert created['config']['auth_type'] == 'in_cluster'
        assert created['version'] == 1

        response = await client.get('/v1/provider-configs')
        assert response.status == 200
        assert [item['provider_config_ref'] for item in (await response.json())['items']] == ['kubernetes/devcloud']

        response = await client.post(
            '/v1/environments',
            json={
                'environment_id': 'devcloud',
                'name': 'DevCloud',
                'allocation_type': 'dynamic',
                'provider_type': 'kubernetes',
                'provider_config_ref': 'kubernetes/devcloud',
                'resource_limits': {},
                'default_timeouts': {},
                'status': 'enabled',
            },
        )
        assert response.status == 201

        response = await client.patch(
            '/v1/provider-configs/kubernetes%2Fdevcloud',
            json={'status': 'disabled'},
        )
        assert response.status == 200
        assert (await response.json())['status'] == 'disabled'

        response = await client.patch('/v1/environments/devcloud', json={'status': 'enabled'})
        assert response.status == 409
        assert (await response.json())['error'] == 'provider_config_disabled'
    finally:
        await client.close()


async def test_control_gateway_stores_and_redacts_provider_credentials(session_factory):
    async with session_factory() as session:
        await set_config_override(
            session,
            role='*',
            identity='*',
            values={'internal_token': 'existing-internal-token'},
            expected_version=0,
            actor='bootstrap',
            secret=True,
        )
        await session.commit()

    app = create_control_app(
        session_factory=session_factory,
        config=DistributedConfig(
            environment_namespace='nexus-workloads',
            environment_enabled_providers=['tkex'],
        ),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    payload = {
        'provider_config_ref': 'tkex/shanghai',
        'config': {
            'provider_type': 'tkex',
            'namespace': 'nexus-workloads',
            'base_url': 'https://tkex.example',
            'api_account': 'account',
            'credential_ref': 'tkex-shanghai',
            'cluster_name': 'cluster',
            'project_name': 'project',
        },
        'credential': {
            'provider_type': 'tkex',
            'token': 'provider-token-v1',
        },
        'status': 'enabled',
    }
    try:
        response = await client.post('/v1/provider-configs', json=payload)
        assert response.status == 201
        raw = await response.text()
        assert 'provider-token-v1' not in raw
        created = await response.json()
        assert created['credential_configured'] is True
        assert created['credential_restart_required'] is True

        response = await client.get('/v1/provider-configs')
        assert response.status == 200
        raw = await response.text()
        assert 'provider-token-v1' not in raw
        assert (await response.json())['items'][0]['credential_configured'] is True

        response = await client.patch(
            '/v1/provider-configs/tkex%2Fshanghai',
            json={
                'credential': {
                    'provider_type': 'tkex',
                    'token': 'provider-token-v2',
                }
            },
        )
        assert response.status == 200
        raw = await response.text()
        assert 'provider-token-v2' not in raw
        assert (await response.json())['credential_restart_required'] is True

        async with session_factory() as session:
            stored = await get_config_override(
                session,
                role='*',
                identity='*',
                secret=True,
                include_secret_values=True,
            )
        assert stored is not None
        assert stored.values['internal_token'] == 'existing-internal-token'
        assert stored.values['provider_credentials'] == {
            'tkex-shanghai': {
                'provider_type': 'tkex',
                'token': 'provider-token-v2',
            }
        }
    finally:
        await client.close()


async def test_control_gateway_requires_matching_stored_credential_before_enable(session_factory):
    app = create_control_app(
        session_factory=session_factory,
        config=DistributedConfig(
            environment_namespace='nexus-workloads',
            environment_enabled_providers=['kubernetes', 'tkex'],
        ),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    config = {
        'provider_type': 'tkex',
        'namespace': 'nexus-workloads',
        'base_url': 'https://tkex.example',
        'api_account': 'account',
        'credential_ref': 'tkex-missing',
        'cluster_name': 'cluster',
        'project_name': 'project',
    }
    try:
        response = await client.post(
            '/v1/provider-configs',
            json={
                'provider_config_ref': 'tkex/missing',
                'config': config,
                'status': 'disabled',
            },
        )
        assert response.status == 201
        assert (await response.json())['credential_configured'] is False

        response = await client.patch('/v1/provider-configs/tkex%2Fmissing', json={'status': 'enabled'})
        assert response.status == 409
        assert (await response.json())['error'] == 'provider_credential_unavailable'

        response = await client.patch(
            '/v1/provider-configs/tkex%2Fmissing',
            json={
                'credential': {
                    'provider_type': 'kubernetes',
                    'token': 'wrong-provider-token',
                    'ca_certificate': 'wrong-provider-ca',
                }
            },
        )
        assert response.status == 409
        assert (await response.json())['error'] == 'provider_credential_type_mismatch'
        assert 'wrong-provider-token' not in await response.text()
    finally:
        await client.close()


async def test_control_gateway_only_enables_deployment_supported_provider_types(session_factory):
    app = create_control_app(
        session_factory=session_factory,
        config=DistributedConfig(
            environment_namespace='nexus-workloads',
            environment_enabled_providers=['kubernetes'],
        ),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    payload = {
        'provider_config_ref': 'tkex/future',
        'config': {
            'provider_type': 'tkex',
            'namespace': 'nexus-workloads',
            'base_url': 'https://tkex.example',
            'api_account': 'account',
            'credential_ref': 'tkex-future',
            'cluster_name': 'cluster',
            'project_name': 'project',
        },
    }
    try:
        response = await client.post('/v1/provider-configs', json={**payload, 'status': 'disabled'})
        assert response.status == 201

        response = await client.patch('/v1/provider-configs/tkex%2Ffuture', json={'status': 'enabled'})
        assert response.status == 409
        assert (await response.json())['error'] == 'provider_type_not_enabled'
    finally:
        await client.close()


async def test_control_gateway_empty_provider_allowlist_denies_dynamic_provider(session_factory):
    app = create_control_app(
        session_factory=session_factory,
        config=DistributedConfig(environment_namespace='nexus-workloads'),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            '/v1/provider-configs',
            json={
                'provider_config_ref': 'kubernetes/disabled',
                'config': {
                    'provider_type': 'kubernetes',
                    'namespace': 'nexus-workloads',
                    'auth_type': 'in_cluster',
                },
                'status': 'enabled',
            },
        )

        assert response.status == 409
        assert (await response.json())['error'] == 'provider_type_not_enabled'
    finally:
        await client.close()


async def test_control_gateway_deletes_stale_manager_without_reserved_slots(session_factory):
    now = utcnow()
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    async with session_factory() as session:
        manager = ManagerRepository(session)
        await manager.register_session(
            manager_id='im-stale-clean',
            session_id='session-stale-clean',
            endpoint_url='http://im-stale-clean.internal:9001',
            supported_modes=[ExecutionMode.JOB],
            supported_spec_types=['nexus_single'],
            supported_instance_modes=['asyncio'],
            capacity_total=2,
            environment_id='predefined-east',
            pool_id='interactive',
            force_replace=False,
            now=now - timedelta(minutes=5),
        )
        await session.commit()

    app = create_control_app(session_factory=session_factory, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.delete('/v1/managers/im-stale-clean')
        assert response.status == 200
        assert await response.json() == {'deleted': True, 'manager_id': 'im-stale-clean'}

        response = await client.get('/v1/managers')
        assert response.status == 200
        assert (await response.json()) == {'items': []}
    finally:
        await client.close()


async def test_control_gateway_rejects_deleting_healthy_manager(session_factory):
    now = utcnow()
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    async with session_factory() as session:
        manager = ManagerRepository(session)
        await manager.register_session(
            manager_id='im-healthy-keep',
            session_id='session-healthy-keep',
            endpoint_url='http://im-healthy-keep.internal:9001',
            supported_modes=[ExecutionMode.JOB],
            supported_spec_types=['nexus_single'],
            supported_instance_modes=['asyncio'],
            capacity_total=2,
            environment_id='predefined-east',
            pool_id='interactive',
            force_replace=False,
            now=now,
        )
        await session.commit()

    app = create_control_app(session_factory=session_factory, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.delete('/v1/managers/im-healthy-keep')
        assert response.status == 409
        error = await response.json()
        assert error == {
            'error': 'manager_healthy',
            'retryable': False,
            'detail': 'manager im-healthy-keep is healthy',
        }

        response = await client.get('/v1/managers')
        assert response.status == 200
        assert [item['manager_id'] for item in (await response.json())['items']] == ['im-healthy-keep']
    finally:
        await client.close()


async def test_control_gateway_rejects_deleting_stale_manager_with_reserved_slots(session_factory):
    now = utcnow()
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    async with session_factory() as session:
        manager = ManagerRepository(session, healthy_ttl=timedelta(seconds=15))
        await manager.register_session(
            manager_id='im-stale-busy',
            session_id='session-stale-busy',
            endpoint_url='http://im-stale-busy.internal:9001',
            supported_modes=[ExecutionMode.JOB],
            supported_spec_types=['nexus_single'],
            supported_instance_modes=['asyncio'],
            capacity_total=2,
            environment_id='predefined-east',
            pool_id='interactive',
            force_replace=False,
            now=now,
        )
        execution = await ExecutionRepository(session, shard_count=8).create_execution(
            CreateExecution(
                execution_id='stale-busy-execution',
                mode=ExecutionMode.JOB,
                spec_type='nexus_single',
                instance_mode='asyncio',
                spec={},
                placement={'environment': 'predefined-east', 'pool': 'interactive'},
            ),
            now=now,
        )
        assignment = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        ).claim_and_assign(shard_ids=[execution.shard_id], now=now)
        assert assignment is not None
        manager_row = await session.get(InstanceManagerRow, 'im-stale-busy')
        liveness = await session.get(InstanceManagerLivenessRow, 'im-stale-busy')
        assert manager_row is not None
        assert liveness is not None
        manager_row.last_heartbeat_at = now - timedelta(minutes=5)
        liveness.last_heartbeat_at = now - timedelta(minutes=5)
        await session.commit()

    app = create_control_app(session_factory=session_factory, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.delete('/v1/managers/im-stale-busy')
        assert response.status == 409
        error = await response.json()
        assert error == {
            'error': 'manager_has_reserved_slots',
            'retryable': False,
            'detail': 'manager im-stale-busy has reserved slots',
        }

        response = await client.get('/v1/managers')
        assert response.status == 200
        assert [item['manager_id'] for item in (await response.json())['items']] == ['im-stale-busy']
    finally:
        await client.close()


async def test_control_gateway_serves_distributed_console(session_factory):
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    app = create_control_app(session_factory=session_factory, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/console?applicationBaseUrl=http://127.0.0.1:8082')
        assert response.status == 200
        assert response.content_type == 'text/html'
        html = await response.text()
        assert 'Nexus Distributed Console' in html
        assert '<div id="root"></div>' in html
        assert '/v1/console/assets/' in html
        assert 'window.__NEXUS_DISTRIBUTED_CONSOLE__' in html
        assert 'http://127.0.0.1:8082' in html
    finally:
        await client.close()


async def test_admin_lists_database_config_overrides_without_secret_values(session_factory):
    async with session_factory() as session:
        await set_config_override(
            session,
            role='*',
            identity='*',
            values={'scheduler_poll_interval': 0.25},
            expected_version=0,
            actor='admin-test',
        )
        await set_config_override(
            session,
            role='control-gateway',
            identity='*',
            values={'internal_token': 'never-expose-this-secret'},
            expected_version=0,
            actor='admin-test',
            secret=True,
        )
        await session.commit()

    app = create_control_app(session_factory=session_factory, config=DistributedConfig())
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/system/config-overrides')

        assert response.status == 200, await response.text()
        assert response.headers['Cache-Control'] == 'no-store'
        response_text = await response.text()
        assert 'never-expose-this-secret' not in response_text
        assert await response.json() == {
            'config_overrides': [
                {
                    'role': '*',
                    'identity': '*',
                    'version': 1,
                    'fields': ['scheduler_poll_interval'],
                    'values': {'scheduler_poll_interval': 0.25},
                }
            ],
            'secret_overrides': [
                {
                    'role': 'control-gateway',
                    'identity': '*',
                    'version': 1,
                    'fields': ['internal_token'],
                }
            ],
        }
    finally:
        await client.close()


async def test_admin_puts_database_config_override_and_returns_ordinary_values(session_factory):
    app = create_control_app(session_factory=session_factory, config=DistributedConfig())
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.put(
            '/v1/system/config-overrides',
            json={
                'role': 'scheduler',
                'identity': '*',
                'values': {'scheduler_poll_interval': 0.25},
                'expected_version': 0,
            },
        )

        assert response.status == 200, await response.text()
        assert response.headers['Cache-Control'] == 'no-store'
        assert await response.json() == {
            'role': 'scheduler',
            'identity': '*',
            'version': 1,
            'fields': ['scheduler_poll_interval'],
            'values': {'scheduler_poll_interval': 0.25},
        }
    finally:
        await client.close()


async def test_admin_puts_secret_config_override_without_returning_secret_values(session_factory):
    app = create_control_app(session_factory=session_factory, config=DistributedConfig())
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.put(
            '/v1/system/config-overrides',
            json={
                'role': 'control-gateway',
                'identity': '*',
                'values': {'internal_token': 'never-return-this-secret'},
                'expected_version': 0,
                'secret': True,
            },
        )

        assert response.status == 200, await response.text()
        response_text = await response.text()
        assert 'never-return-this-secret' not in response_text
        assert await response.json() == {
            'role': 'control-gateway',
            'identity': '*',
            'version': 1,
            'fields': ['internal_token'],
        }
    finally:
        await client.close()


async def test_admin_config_override_put_maps_version_conflict_to_409(session_factory):
    app = create_control_app(session_factory=session_factory, config=DistributedConfig())
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        payload = {
            'role': 'scheduler',
            'identity': '*',
            'values': {'scheduler_poll_interval': 0.25},
            'expected_version': 0,
        }
        response = await client.put('/v1/system/config-overrides', json=payload)
        assert response.status == 200

        response = await client.put('/v1/system/config-overrides', json=payload)
        assert response.status == 409
        assert await response.json() == {'error': 'config_version_conflict', 'retryable': False}
    finally:
        await client.close()


async def test_admin_deletes_database_config_override(session_factory):
    app = create_control_app(session_factory=session_factory, config=DistributedConfig())
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.put(
            '/v1/system/config-overrides',
            json={
                'role': 'scheduler',
                'identity': '*',
                'values': {'scheduler_poll_interval': 0.25},
                'expected_version': 0,
            },
        )
        assert response.status == 200

        response = await client.delete(
            '/v1/system/config-overrides',
            json={'role': 'scheduler', 'identity': '*', 'expected_version': 1},
        )
        assert response.status == 200, await response.text()
        assert await response.json() == {'deleted': True, 'role': 'scheduler', 'identity': '*'}
    finally:
        await client.close()


async def test_admin_config_override_put_rejects_non_strict_and_extra_fields(session_factory):
    app = create_control_app(session_factory=session_factory, config=DistributedConfig())
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.put(
            '/v1/system/config-overrides',
            json={
                'role': 'scheduler',
                'identity': '*',
                'values': {'scheduler_poll_interval': 0.25},
                'expected_version': True,
            },
        )
        assert response.status == 422

        response = await client.put(
            '/v1/system/config-overrides',
            json={
                'role': 'scheduler',
                'identity': '*',
                'values': {'scheduler_poll_interval': 0.25},
                'expected_version': 0,
                'unexpected': 'rejected',
            },
        )
        assert response.status == 422
    finally:
        await client.close()


async def test_admin_config_override_put_returns_validation_error_for_invalid_values(session_factory):
    app = create_control_app(session_factory=session_factory, config=DistributedConfig())
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.put(
            '/v1/system/config-overrides',
            json={
                'role': 'scheduler',
                'identity': '*',
                'values': {'not_a_distributed_config_field': True},
                'expected_version': 0,
            },
        )
        assert response.status == 422
        assert (await response.json())['error'] == 'validation_error'
    finally:
        await client.close()


async def test_admin_config_overrides_rejects_invalid_ordinary_secret_without_exposing_it(session_factory):
    async with session_factory() as session:
        session.add(
            DistributedConfigOverrideRow(
                role='*',
                identity='*',
                version=1,
                config_json={'internal_token': 'never-expose-this-secret'},
            )
        )
        await session.commit()

    app = create_control_app(session_factory=session_factory, config=DistributedConfig())
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/system/config-overrides')

        assert response.status == 500
        assert response.headers['Cache-Control'] == 'no-store'
        assert 'never-expose-this-secret' not in await response.text()
    finally:
        await client.close()


async def test_console_application_base_url_uses_deployment_config_and_only_allows_local_override(
    session_factory,
):
    disabled = create_control_app(
        session_factory=session_factory,
        config=DistributedConfig(
            auth_mode='disabled',
            application_public_base_url='http://application.public',
            console_application_base_url='http://application.configured',
        ),
    )
    disabled_client = TestClient(TestServer(disabled))
    await disabled_client.start_server()
    try:
        response = await disabled_client.get('/v1/console')
        assert 'http://application.configured' in await response.text()

        response = await disabled_client.get(
            '/v1/console',
            params={'applicationBaseUrl': 'http://application.local-override'},
        )
        html = await response.text()
        assert 'http://application.local-override' in html
        assert 'http://application.configured' not in html
    finally:
        await disabled_client.close()

    required = create_control_app(
        session_factory=session_factory,
        config=DistributedConfig(
            auth_mode='required',
            application_public_base_url='https://application-mcp.authenticated',
            console_application_base_url='https://application.authenticated',
        ),
    )
    required_client = TestClient(TestServer(required))
    await required_client.start_server()
    try:
        response = await required_client.get(
            '/v1/console',
            params={'applicationBaseUrl': 'https://attacker.invalid'},
        )
        html = await response.text()
        assert 'https://application.authenticated' in html
        assert 'https://application-mcp.authenticated' not in html
        assert 'https://attacker.invalid' not in html
    finally:
        await required_client.close()


async def test_control_gateway_sets_manager_draining_through_current_session(session_factory):
    received: list[dict] = []

    async def update_draining(request):
        received.append(
            {
                'internal_token': request.headers.get('X-Nexus-Internal-Token'),
                'payload': await request.json(),
            }
        )
        return web.json_response(
            {
                'manager_id': 'im-drain',
                'session_id': 'session-drain',
                'draining': True,
            }
        )

    manager_app = web.Application()
    manager_app.router.add_patch('/v1/internal/draining', update_draining)
    manager_client = TestClient(TestServer(manager_app))
    await manager_client.start_server()
    try:
        async with session_factory() as session:
            manager = ManagerRepository(session)
            await manager.register_session(
                manager_id='im-drain',
                session_id='session-drain',
                endpoint_url=str(manager_client.make_url('/')).rstrip('/'),
                supported_modes=[ExecutionMode.LIVE],
                supported_spec_types=['nexus_live'],
                supported_instance_modes=['asyncio'],
                capacity_total=1,
                force_replace=False,
                now=utcnow(),
            )
            await session.commit()

        app = create_control_app(
            session_factory=session_factory,
            config=DistributedConfig(
                secrets={'internal_token': 'internal-secret'}, manager_healthy_ttl=timedelta(seconds=15)
            ),
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.patch('/v1/managers/im-drain', json={'draining': True})

            assert response.status == 200
            assert await response.json() == {
                'manager_id': 'im-drain',
                'session_id': 'session-drain',
                'draining': True,
            }
            assert received == [{'internal_token': 'internal-secret', 'payload': {'draining': True}}]

            response = await client.patch('/v1/managers/im-drain', json={'draining': 'true'})
            assert response.status == 422
            assert received == [{'internal_token': 'internal-secret', 'payload': {'draining': True}}]
        finally:
            await client.close()
    finally:
        await manager_client.close()


def test_allocation_payload_marks_ready_capacity_gap_as_degraded():
    from nexus.distributed.gateways.control.app import _allocation_payload

    degraded = _allocation_payload(
        {'status': 'ready', 'desired_manager_count': 2, 'ready_manager_count': 1},
        admin=False,
    )
    restored = _allocation_payload(
        {'status': 'ready', 'desired_manager_count': 2, 'ready_manager_count': 2},
        admin=False,
    )

    assert degraded['degraded'] is True
    assert restored['degraded'] is False


async def test_control_gateway_proxies_manager_aiomonitor_webui(session_factory):
    received: list[dict] = []

    async def aiomonitor_proxy(request):
        received.append(
            {
                'loop': request.match_info['loop'],
                'tail': request.match_info['tail'],
                'query': dict(request.query),
                'authorization': request.headers.get('Authorization'),
            }
        )
        return web.Response(text='aiomonitor page', content_type='text/html')

    manager_app = web.Application()
    manager_app.router.add_get('/v1/internal/debug/aiomonitor/{loop}/{tail:.*}', aiomonitor_proxy)
    manager_client = TestClient(TestServer(manager_app))
    await manager_client.start_server()
    try:
        async with session_factory() as session:
            manager = ManagerRepository(session)
            await manager.register_session(
                manager_id='im-debug',
                session_id='session-debug',
                endpoint_url=str(manager_client.make_url('/')).rstrip('/'),
                supported_modes=[ExecutionMode.LIVE],
                supported_spec_types=['nexus_live'],
                supported_instance_modes=['asyncio'],
                capacity_total=1,
                force_replace=False,
                now=utcnow(),
            )
            await session.commit()

        app = create_control_app(
            session_factory=session_factory,
            config=DistributedConfig(
                secrets={'internal_token': 'internal-secret'}, manager_healthy_ttl=timedelta(seconds=15)
            ),
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.get('/v1/managers/im-debug/debug/aiomonitor/main/static/app.js?theme=dark')
            assert response.status == 200
            assert await response.text() == 'aiomonitor page'
            assert received == [
                {
                    'loop': 'main',
                    'tail': 'static/app.js',
                    'query': {'theme': 'dark'},
                    'authorization': 'Bearer internal-secret',
                }
            ]
        finally:
            await client.close()
    finally:
        await manager_client.close()


async def test_control_gateway_proxies_manager_py_spy_debug(session_factory):
    received: list[dict] = []

    async def py_spy_proxy(request):
        received.append(
            {
                'action': request.match_info['action'],
                'query': dict(request.query),
                'authorization': request.headers.get('Authorization'),
            }
        )
        return web.Response(text='py-spy profile', content_type='text/plain')

    manager_app = web.Application()
    manager_app.router.add_get('/v1/internal/debug/py-spy/{action}', py_spy_proxy)
    manager_client = TestClient(TestServer(manager_app))
    await manager_client.start_server()
    try:
        async with session_factory() as session:
            manager = ManagerRepository(session)
            await manager.register_session(
                manager_id='im-debug',
                session_id='session-debug',
                endpoint_url=str(manager_client.make_url('/')).rstrip('/'),
                supported_modes=[ExecutionMode.LIVE],
                supported_spec_types=['nexus_live'],
                supported_instance_modes=['asyncio'],
                capacity_total=1,
                force_replace=False,
                now=utcnow(),
            )
            await session.commit()

        app = create_control_app(
            session_factory=session_factory,
            config=DistributedConfig(
                secrets={'internal_token': 'internal-secret'}, manager_healthy_ttl=timedelta(seconds=15)
            ),
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.get('/v1/managers/im-debug/debug/py-spy/record?duration=3&format=svg')
            assert response.status == 200
            assert await response.text() == 'py-spy profile'
            assert received == [
                {
                    'action': 'record',
                    'query': {'duration': '3', 'format': 'svg'},
                    'authorization': 'Bearer internal-secret',
                }
            ]
        finally:
            await client.close()
    finally:
        await manager_client.close()


async def test_request_stop_persists_reason_and_parent_ttl_expires_early_states(session_factory):
    now = utcnow()
    async with session_factory() as session:
        repository = ExecutionRepository(session, shard_count=8)
        for execution_id, reason, expected in (
            ('queued-default-stop', 'user_stop', ExecutionStatus.KILLED),
            ('queued-fail-fast-stop', 'fail_fast', ExecutionStatus.KILLED),
            ('queued-parent-ttl-stop', 'parent_ttl', ExecutionStatus.EXPIRED),
        ):
            await repository.create_execution(
                CreateExecution(
                    execution_id=execution_id,
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                ),
                now=now,
            )
            stopped = await repository.request_stop(execution_id, stop_reason=reason, now=now)
            assert stopped.status == expected
            assert stopped.stop_reason == reason
        await session.commit()

    async with session_factory() as session:
        manager = ManagerRepository(session)
        await manager.register_session(
            manager_id='im-stop-reason',
            session_id='session-stop-reason',
            endpoint_url='http://im-stop-reason.internal:9001',
            supported_modes=[ExecutionMode.JOB],
            supported_spec_types=['nexus_single'],
            supported_instance_modes=['asyncio'],
            capacity_total=2,
            force_replace=False,
            now=now,
        )
        repository = ExecutionRepository(session, shard_count=8)
        assigned = await repository.create_execution(
            CreateExecution(
                execution_id='assigned-parent-ttl-stop',
                mode=ExecutionMode.JOB,
                spec_type='nexus_single',
                instance_mode='asyncio',
                spec={},
            ),
            now=now,
        )
        running = await repository.create_execution(
            CreateExecution(
                execution_id='running-parent-ttl-stop',
                mode=ExecutionMode.JOB,
                spec_type='nexus_single',
                instance_mode='asyncio',
                spec={},
            ),
            now=now,
        )
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assigned_claim = await scheduler.claim_and_assign(shard_ids=[assigned.shard_id], now=now)
        running_claim = await scheduler.claim_and_assign(shard_ids=[running.shard_id], now=now)
        assert assigned_claim is not None
        assert running_claim is not None

        assigned_stop = await repository.request_stop(
            assigned.execution_id,
            stop_reason='parent_ttl',
            now=now,
        )
        assert assigned_stop.status == ExecutionStatus.EXPIRED
        assert assigned_stop.current_assignment is not None
        assert assigned_stop.current_assignment.status == AssignmentStatus.REVOKED

        await manager.ack_assignment(
            running_claim.assignment_id,
            'im-stop-reason',
            running_claim.manager_session_id,
            now=now,
        )
        await manager.mark_running(
            running_claim.assignment_id,
            'im-stop-reason',
            running_claim.manager_session_id,
            endpoint_url=None,
            endpoint_ready=False,
            runtime_info={},
            now=now,
        )
        running_stop = await repository.request_stop(
            running.execution_id,
            stop_reason='parent_ttl',
            now=now,
        )
        command = await session.scalar(
            select(ExecutionCommandRow).where(
                ExecutionCommandRow.execution_id == running.execution_id,
                ExecutionCommandRow.type == 'stop.request',
            )
        )
        await session.commit()

    assert running_stop.status == ExecutionStatus.STOPPING
    assert running_stop.stop_reason == 'parent_ttl'
    assert command.payload_json == {'reason': 'parent_ttl'}


async def test_child_detail_resolves_parent_item_attempt_context(session_factory):
    now = utcnow()
    async with session_factory() as session:
        await ExecutionRepository(session, shard_count=8).create_execution(
            CreateExecution(
                execution_id='child-context',
                mode=ExecutionMode.JOB,
                spec_type='nexus_single',
                instance_mode='asyncio',
                spec={},
                owner_principal_id='local-dev',
                parent_execution_id='parent-context',
            ),
            now=now,
        )
        session.add(
            BatchItemAttemptRow(
                parent_execution_id='parent-context',
                item_id='task-7',
                attempt_no=2,
                child_execution_id='child-context',
                spec_sha256='a' * 64,
                status='submitted',
                status_version=1,
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()
    app = create_control_app(session_factory=session_factory, config=DistributedConfig(shard_count=8))
    async with TestClient(TestServer(app)) as client:
        response = await client.get('/v1/executions/child-context')
        assert response.status == 200
        assert (await response.json())['batch_context'] == {
            'parent_execution_id': 'parent-context',
            'item_id': 'task-7',
            'attempt_no': 2,
        }
        response = await client.get('/v1/executions/child-context?summary=1')
        assert response.status == 200
        assert 'batch_context' not in await response.json()
