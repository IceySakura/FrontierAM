from types import SimpleNamespace

import pytest

from aiohttp.test_utils import TestClient, TestServer
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.config import DistributedConfig
from nexus.distributed.execution.models import ExecutionMode
from nexus.distributed.gateways.common.keys import CONFIG_KEY
from nexus.distributed.gateways.control.app import create_control_app
from nexus.distributed.persistence.orm import BatchRunRow, DependencyBuildRow, create_schema
from nexus.distributed.repositories.execution import CreateExecution, ExecutionRepository
from nexus.distributed.utils.clock import utcnow


@pytest.fixture
async def client():
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    await create_schema(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    now = utcnow()
    async with factory() as session:
        await ExecutionRepository(session, shard_count=8).create_execution(
            CreateExecution(
                execution_id='batch',
                mode=ExecutionMode.JOB,
                spec_type='nexus_batch',
                instance_mode='asyncio',
                spec={},
                environment_id='env',
                pool_id='predefined',
                owner_principal_id='local-dev',
            ),
            now=now,
        )
        session.add(
            BatchRunRow(
                parent_execution_id='batch',
                shard_id=0,
                generation_mode='local_task_list',
                generation_status='complete',
                owner_principal_id='local-dev',
                child_dependency_build_id='build',
                last_progress_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            DependencyBuildRow(
                dependency_build_id='build',
                build_key='a' * 64,
                owner_principal_id='local-dev',
                environment_id='env',
                status='building',
                build_spec_json={},
                provider_type='kubernetes',
                provider_config_ref='test',
                provider_config_snapshot_json={},
                provider_job_name='builder-job',
                attempt=1,
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()
    app = create_control_app(session_factory=factory, config=DistributedConfig())
    async with TestClient(TestServer(app)) as http:
        yield http
    await engine.dispose()


@pytest.mark.parametrize(('field', 'value'), [('owner_principal_id', 'other'), ('environment_id', 'other-env')])
async def test_build_logs_do_not_follow_foreign_build_references(client, field, value):
    from nexus.distributed.gateways.common.keys import SESSION_FACTORY_KEY

    async with client.app[SESSION_FACTORY_KEY]() as session:
        build = await session.get(DependencyBuildRow, 'build')
        setattr(build, field, value)
        await session.commit()
    response = await client.get('/v1/executions/batch/build-logs')
    assert response.status == 200
    assert (await response.json())['items'] == []


async def test_build_logs_find_allocations_without_batch_metadata(client):
    from nexus.distributed.gateways.common.keys import SESSION_FACTORY_KEY
    from nexus.distributed.persistence.orm import EnvironmentAllocationRow

    async with client.app[SESSION_FACTORY_KEY]() as session:
        batch = await session.get(BatchRunRow, 'batch')
        await session.delete(batch)
        session.add(
            EnvironmentAllocationRow(
                allocation_id='allocation',
                top_level_execution_id='batch',
                owner_principal_id='local-dev',
                environment_id='env',
                pool_id='dynamic',
                workload='execution',
                status='requested',
                dependency_build_id='build',
                allocation_spec_hash='a' * 64,
                config_version=1,
                created_at=utcnow(),
                updated_at=utcnow(),
            )
        )
        build = await session.get(DependencyBuildRow, 'build')
        build.provider_job_name = None
        await session.commit()
    response = await client.get('/v1/executions/batch/build-logs')
    data = await response.json()
    assert data['items'][0]['dependency_build_id'] == 'build'
    assert data['items'][0]['availability'] == 'not_started'


@pytest.mark.parametrize('status', ['ready', 'failed'])
async def test_finished_build_logs_are_unavailable_without_history_lookup(client, monkeypatch, status):
    from nexus.distributed.gateways.common.keys import HTTP_CLIENT_KEY, SESSION_FACTORY_KEY

    def unexpected_query(*args, **kwargs):
        pytest.fail('Finished build logs must not query an upstream service')

    monkeypatch.setattr(client.app[HTTP_CLIENT_KEY], 'post', unexpected_query)
    monkeypatch.setattr(client.app[HTTP_CLIENT_KEY], 'get', unexpected_query)
    async with client.app[SESSION_FACTORY_KEY]() as session:
        build = await session.get(DependencyBuildRow, 'build')
        build.status = status
        build.finished_at = utcnow()
        build.provider_job_name = None
        await session.commit()
    response = await client.get('/v1/executions/batch/build-logs')
    assert response.status == 200
    item = (await response.json())['items'][0]
    assert item['source'] == 'builder_file'
    assert item['availability'] == 'unavailable'
    assert item['error_code'] == 'build_log_expired'
    assert item['lines'] == []


@pytest.mark.parametrize(
    ('case', 'availability'),
    [('live', 'ready'), ('many', 'ready'), ('missing', 'not_started'), ('auth_failure', 'unavailable')],
)
@pytest.mark.parametrize('route', ['/v1/executions/batch/build-logs', '/v1/builds/build/logs'])
async def test_live_file_api_reports_real_availability(client, tmp_path, monkeypatch, case, availability, route):
    from nexus.distributed.dependency_build.logs import (
        build_log_token,
        configure_build_log,
        create_build_log_app,
        output_logger,
    )
    from nexus.distributed.gateways.control.routes import _build_logs

    config = client.app[CONFIG_KEY]
    kid = config.artifact_upload_ticket_current_key_id
    token = build_log_token(config.artifact_ticket_keyring()[kid], 'build', 1)
    path = tmp_path / 'build.jsonl'
    handler = configure_build_log(path, build_id='build', attempt='1')
    try:
        output_logger.info('Collecting package https://user:secret@example.org/?token=private')
        handler.close()
        if case == 'many':
            import json

            path.write_text(
                ''.join(
                    json.dumps({'timestamp': '2026-09-10T00:00:00Z', 'level': 'INFO', 'message': f'line {i}'}) + '\n'
                    for i in range(1200)
                )
            )
        async with TestServer(create_build_log_app(path, token=token if case != 'auth_failure' else 'wrong')) as server:
            monkeypatch.setattr(_build_logs, 'BUILD_LOG_PORT', server.port)

            class Provider:
                async def list_job_pods(self, job):
                    assert job == 'builder-job'
                    if case == 'missing':
                        return []
                    return [
                        {
                            'metadata': {'annotations': {'nexus.io/build-log-key-id': kid}},
                            'status': {'podIP': '127.0.0.1'},
                        }
                    ]

            class Resolver:
                def __init__(self, **kwargs):
                    pass

                def resolve(self, **kwargs):
                    return SimpleNamespace(client=Provider())

                async def close(self):
                    pass

            monkeypatch.setattr(_build_logs, 'WorkloadClientResolver', Resolver)
            response = await client.get(f'{route}?lines=1000')
            assert response.status == 200
            payload = await response.json()
            item = payload['items'][0]
            assert item['availability'] == availability
            assert item['source'] == 'builder_file'
            assert 'secret' not in str(payload)
            assert 'private' not in str(payload)
            if case == 'live':
                assert 'Collecting package' in item['lines'][0]['message']
            elif case == 'many':
                assert item['line_count'] == 1000
                assert item['lines'][0]['message'] == 'line 200'
                assert item['lines'][-1]['message'] == 'line 1199'
            else:
                assert item['lines'] == []
            capped = await client.get(f'{route}?lines=1001')
            assert capped.status == 200
            assert (await capped.json())['items'][0]['line_count'] == item['line_count']
            assert (await client.get('/v1/executions/missing/build-logs')).status == 404
    finally:
        output_logger.removeHandler(handler)
        handler.close()


def test_response_budget_keeps_latest_records_and_handles_large_unicode_lines():
    import json

    from nexus.distributed.gateways.control.routes._build_logs import _apply_log_byte_budget

    item = {'lines': [{'message': 'old'}, {'message': '日志' * 8000}], 'partial': False}
    _apply_log_byte_budget(item, 2048)
    assert len(json.dumps(item['lines']).encode()) <= 2048
    assert item['partial']
    assert item['line_count'] >= 1
    assert '[truncated]' in item['lines'][-1]['message']


@pytest.fixture
async def allocation_client(client):
    from nexus.distributed.gateways.common.keys import SESSION_FACTORY_KEY
    from nexus.distributed.persistence.orm import EnvironmentAllocationRow

    async with client.app[SESSION_FACTORY_KEY]() as session:
        for name, workload, build_id in [('orchestration', 'orchestration', 'build'), ('workers', 'execution', None)]:
            session.add(
                EnvironmentAllocationRow(
                    allocation_id=name,
                    top_level_execution_id='batch',
                    owner_principal_id='local-dev',
                    environment_id='env',
                    pool_id=name,
                    workload=workload,
                    status='requested',
                    dependency_build_id=build_id,
                    allocation_spec_hash='a' * 64,
                    config_version=1,
                    created_at=utcnow(),
                    updated_at=utcnow(),
                )
            )
        build = await session.get(DependencyBuildRow, 'build')
        build.provider_job_name = None
        await session.commit()
    return client


@pytest.mark.parametrize(
    ('pod_status', 'expected'),
    [
        ({}, 'preparing'),
        (
            {
                'conditions': [
                    {
                        'type': 'PodScheduled',
                        'status': 'False',
                        'reason': 'Unschedulable',
                        'message': 'Insufficient cpu',
                    }
                ]
            },
            'waiting_for_scheduling',
        ),
        (
            {
                'containerStatuses': [
                    {'name': 'builder', 'state': {'waiting': {'reason': 'ErrImagePull', 'message': 'token=private'}}}
                ]
            },
            'image_pull_failed',
        ),
    ],
)
async def test_build_startup_status_is_visible_without_contacting_builder(client, monkeypatch, pod_status, expected):
    from nexus.distributed.gateways.common.keys import HTTP_CLIENT_KEY
    from nexus.distributed.gateways.control.routes import _build_logs

    class Resolver:
        def __init__(self, **kwargs):
            pass

        def resolve(self, **kwargs):
            return SimpleNamespace(client=self)

        async def list_job_pods(self, job):
            return [{'status': pod_status}]

        async def close(self):
            pass

    def unexpected_request(*args, **kwargs):
        pytest.fail('Do not contact Builder before it has started')

    monkeypatch.setattr(_build_logs, 'WorkloadClientResolver', Resolver)
    monkeypatch.setattr(client.app[HTTP_CLIENT_KEY], 'get', unexpected_request)
    response = await client.get('/v1/executions/batch/build-logs')
    item = (await response.json())['items'][0]
    assert item['availability'] == 'not_started'
    assert item['phase'] == expected
    assert 'startup_message' not in item
    assert 'private' not in str(item)


@pytest.mark.parametrize('has_records', [True, False])
async def test_finished_build_queries_zhiyan_with_build_and_attempt(client, has_records):
    from aiohttp import web
    from pydantic import SecretStr

    from nexus.distributed.config import ZhiyanLogQueryConfig
    from nexus.distributed.gateways.common.keys import SESSION_FACTORY_KEY

    captured = []

    async def query(request):
        captured.append(await request.json())
        records = (
            [{'timestamp': utcnow().isoformat(), 'level': 'INFO', 'message': 'Build complete'}] if has_records else []
        )
        return web.json_response({'code': 0, 'data': {'records': records}})

    app = web.Application()
    app.router.add_post('/log/v2/search/query', query)
    async with TestServer(app) as server:
        config = client.app[CONFIG_KEY]
        config.zhiyan_log_query = ZhiyanLogQueryConfig(
            enabled=True,
            endpoint=str(server.make_url('')).rstrip('/'),
            project_name='project',
            staff_name='staff',
            credential_ref='logs',
        )
        config.secrets.zhiyan_log_credentials['logs'] = SecretStr('test-token')
        async with client.app[SESSION_FACTORY_KEY]() as session:
            build = await session.get(DependencyBuildRow, 'build')
            build.status = 'failed'
            build.finished_at = utcnow()
            build.provider_type = 'tkex'
            build.provider_config_snapshot_json = {'builder_zhiyan_log': {'log_name': 'build-flow'}}
            await session.commit()
        response = await client.get('/v1/executions/batch/build-logs')
        item = (await response.json())['items'][0]
        assert item['source'] == 'zhiyan'
        assert item['availability'] == ('ready' if has_records else 'ingest_pending')
        assert captured[0]['filter'] == [
            {'key': 'dependency_build_id', 'operator': 'eq', 'value': 'build'},
            {'key': 'build_attempt', 'operator': 'eq', 'value': '1'},
        ]
        assert captured[0]['dataflowName'] == 'build-flow'


async def test_builds_can_be_browsed_without_an_allocation(client):
    response = await client.get('/v1/builds?limit=1')
    assert response.status == 200
    payload = await response.json()
    assert [item['dependency_build_id'] for item in payload['items']] == ['build']
    assert 'build_spec_json' not in payload['items'][0]
    response = await client.get('/v1/builds/missing/logs')
    assert response.status == 404


async def test_build_log_endpoint_reads_the_selected_build(client):
    from nexus.distributed.gateways.common.keys import SESSION_FACTORY_KEY

    async with client.app[SESSION_FACTORY_KEY]() as session:
        build = await session.get(DependencyBuildRow, 'build')
        build.status = 'ready'
        await session.commit()
    response = await client.get('/v1/builds/build/logs')
    assert response.status == 200
    assert (await response.json())['items'][0]['dependency_build_id'] == 'build'


async def test_build_logs_enforce_owner(client, monkeypatch):
    from nexus.distributed.access.auth import AuthContext
    from nexus.distributed.gateways.control.routes import operations

    context = AuthContext(
        principal_id='stranger',
        issuer='test',
        tenant='',
        subject='stranger',
        roles=frozenset(),
        scopes=frozenset({'execution:read'}),
        auth_type='pat',
    )
    monkeypatch.setattr(operations, 'auth_context', lambda _: context)
    response = await client.get('/v1/builds/build/logs')
    assert response.status == 404


async def test_build_list_filters_access_before_pagination(client, monkeypatch):

    async def allowed(*args):
        return None, None, frozenset({('someone-else', 'env')})

    from nexus.distributed.gateways.control.routes import executions

    monkeypatch.setattr(executions, '_execution_list_environment_access', allowed)
    response = await client.get('/v1/builds?limit=1')
    assert response.status == 200
    assert (await response.json())['items'] == []


async def test_build_list_pages_tied_timestamps_without_duplicates(client):
    from nexus.distributed.gateways.common.keys import SESSION_FACTORY_KEY

    async with client.app[SESSION_FACTORY_KEY]() as session:
        build = await session.get(DependencyBuildRow, 'build')
        session.add(
            DependencyBuildRow(
                dependency_build_id='build-2',
                build_key='b' * 64,
                owner_principal_id='local-dev',
                environment_id='env',
                status='ready',
                build_spec_json={},
                attempt=1,
                created_at=build.created_at,
                updated_at=build.updated_at,
            )
        )
        await session.commit()
    first = await (await client.get('/v1/builds?limit=1')).json()
    second = await (await client.get('/v1/builds', params={'limit': '1', 'cursor': first['next_cursor']})).json()
    assert [item['dependency_build_id'] for item in first['items'] + second['items']] == ['build-2', 'build']
    assert second['next_cursor'] is None
    assert (await client.get('/v1/builds?cursor=invalid')).status == 422


async def test_existing_allocation_detail_remains_available(allocation_client):
    response = await allocation_client.get('/v1/allocations/orchestration')
    assert response.status == 200
    payload = await response.json()
    assert payload['allocation_id'] == 'orchestration'
    assert payload['dependency_build']['dependency_build_id'] == 'build'
    assert 'shared_build_allocations' not in payload


@pytest.mark.parametrize(('path', 'expected'), [('/v1/allocations', 403), ('/v1/allocations/orchestration', 404)])
async def test_allocation_list_requires_admin_and_detail_requires_environment_grant(
    allocation_client, monkeypatch, path, expected
):
    from nexus.distributed.access.auth import AuthContext
    from nexus.distributed.gateways.control.routes import operations

    context = AuthContext(
        principal_id='local-dev',
        issuer='test',
        tenant='',
        subject='local-dev',
        roles=frozenset(),
        scopes=frozenset({'execution:read'}),
        auth_type='pat',
    )
    monkeypatch.setattr(operations, 'auth_context', lambda _: context)
    response = await allocation_client.get(path)
    assert response.status == expected


@pytest.mark.parametrize('exit_code', [0, 1])
async def test_exited_builder_shows_finalizing_before_control_plane_completion(client, monkeypatch, exit_code):
    from nexus.distributed.gateways.control.routes._build_logs import _read_live_build_logs

    async def pods(_):
        return [
            {
                'status': {
                    'containerStatuses': [
                        {'name': 'builder', 'state': {'terminated': {'exitCode': exit_code}}},
                        {'name': 'zhiyan-agent', 'state': {'running': {}}},
                    ]
                }
            }
        ]

    request = SimpleNamespace(match_info={'build_id': 'build'})
    build = SimpleNamespace(provider_job_name='job', status='building', dependency_build_id='build', attempt=1)
    result = await _read_live_build_logs(request, build, SimpleNamespace(list_job_pods=pods), lines=20)
    assert result['phase'] == 'finalizing'
    assert result['status'] == 'building'
    assert result['availability'] == 'unavailable'
    assert result['lines'] == []


@pytest.mark.parametrize(
    ('upstream_status', 'has_records', 'age', 'expected_status', 'availability'),
    [
        (200, True, 10, 200, 'ready'),
        (200, False, 10, 200, 'ingest_pending'),
        (200, False, 400, 200, 'unavailable'),
        (429, False, 10, 429, 'unavailable'),
        (500, False, 10, 502, 'unavailable'),
    ],
)
async def test_existing_execution_history_route_after_log_refactor(
    client,
    monkeypatch,
    upstream_status,
    has_records,
    age,
    expected_status,
    availability,
):
    from datetime import timedelta

    from aiohttp import web
    from pydantic import SecretStr

    from nexus.distributed.config import ZhiyanLogQueryConfig
    from nexus.distributed.gateways.control.routes import execution_logs
    from nexus.distributed.repositories.allocation import AllocationRepository

    now = utcnow()
    execution = SimpleNamespace(
        execution_id='execution',
        status='succeeded',
        allocation_id='allocation',
        created_at=now - timedelta(hours=8),
        updated_at=now - timedelta(seconds=age),
    )

    async def context(*args):
        return SimpleNamespace(execution=execution, finished_at=execution.updated_at)

    async def binding(*args, predefined_log_name=None):
        assert predefined_log_name is None
        return SimpleNamespace(log_name='manager-flow')

    monkeypatch.setattr(execution_logs, '_get_authorized_execution_log_context', context)
    monkeypatch.setattr(AllocationRepository, 'zhiyan_log_binding', binding)
    monkeypatch.setattr(execution_logs, 'utcnow', lambda: now)

    async def upstream(request):
        data = await request.json()
        assert data['filter'] == [{'key': 'execution_id', 'operator': 'eq', 'value': 'execution'}]
        assert data['dataflowName'] == 'manager-flow'
        assert request.headers['token'] == 'test-token'
        assert data['endTime'] - data['startTime'] == 21600 * 1000
        return web.json_response(
            {
                'code': 0,
                'data': {
                    'records': [{'timestamp': now.isoformat(), 'message': 'manager history'}] if has_records else []
                },
            },
            status=upstream_status,
        )

    app = web.Application()
    app.router.add_post('/log/v2/search/query', upstream)
    async with TestServer(app) as server:
        config = client.app[CONFIG_KEY]
        config.zhiyan_log_query = ZhiyanLogQueryConfig(
            enabled=True,
            endpoint=str(server.make_url('')).rstrip('/'),
            project_name='project',
            staff_name='staff',
            credential_ref='logs',
        )
        config.secrets.zhiyan_log_credentials['logs'] = SecretStr('test-token')
        response = await client.get('/v1/executions/execution/logs')
        assert response.status == expected_status
        result = await response.json()
        assert result['source'] == 'zhiyan'
        assert result['availability'] == availability
        assert result['partial'] is (upstream_status == 200 and has_records)
        assert result['retry_after_ms'] == (2000 if availability == 'ingest_pending' else None)
        if has_records:
            assert result['lines'][0]['message'] == 'manager history'
        elif upstream_status == 200:
            assert result['error_code'] == 'zhiyan_log_ingest_pending'


@pytest.mark.parametrize('upstream_status', [200, 403, 500])
async def test_existing_execution_live_route_after_log_refactor(client, monkeypatch, upstream_status):
    from aiohttp import web
    from pydantic import SecretStr

    from nexus.distributed.gateways.common.keys import SESSION_FACTORY_KEY
    from nexus.distributed.gateways.control.routes import execution_logs
    from nexus.distributed.persistence.orm import InstanceManagerRow

    async def context(*args):
        return SimpleNamespace(
            execution=SimpleNamespace(
                execution_id='execution', status='running', current_assignment=SimpleNamespace(manager_id='manager')
            )
        )

    monkeypatch.setattr(execution_logs, '_get_authorized_execution_log_context', context)
    client.app[CONFIG_KEY].secrets.internal_token = SecretStr('internal-test')

    async def upstream(request):
        assert request.query['format'] == 'jsonl'
        assert request.query['lines'] == '200'
        assert request.headers['X-Nexus-Internal-Token'] == 'internal-test'
        return web.Response(
            text='{"timestamp":"2026-01-01T00:00:00Z","message":"manager live"}\n', status=upstream_status
        )

    app = web.Application()
    app.router.add_get('/v1/internal/logs', upstream)
    async with TestServer(app) as server:
        async with client.app[SESSION_FACTORY_KEY]() as session:
            session.add(
                InstanceManagerRow(
                    manager_id='manager',
                    session_id='session',
                    endpoint_url=str(server.make_url('')).rstrip('/'),
                    environment_id='env',
                    pool_id='pool',
                    workload='execution',
                    supported_modes_json=[],
                    supported_spec_types_json=[],
                    supported_instance_modes_json=[],
                    capacity_total=1,
                    last_heartbeat_at=utcnow(),
                    created_at=utcnow(),
                    updated_at=utcnow(),
                )
            )
            await session.commit()
        response = await client.get('/v1/executions/execution/logs')
        result = await response.json()
        assert response.status == (200 if upstream_status == 200 else 502)
        assert result['source'] == 'instance_manager'
        if upstream_status == 200:
            assert result['lines'][0]['message'] == 'manager live'
        else:
            assert result['error_code'] == 'manager_log_unavailable'


@pytest.mark.parametrize('status', ['failed', 'ready'])
async def test_build_metadata_distinguishes_retry_from_cleanup_without_logs(client, status):
    from datetime import timedelta

    from nexus.distributed.gateways.common.keys import SESSION_FACTORY_KEY

    async with client.app[SESSION_FACTORY_KEY]() as session:
        build = await session.get(DependencyBuildRow, 'build')
        build.status = status
        build.next_retry_at = utcnow() + timedelta(minutes=1)
        build.error_code = 'install_failed' if status == 'failed' else None
        await session.commit()
    response = await client.get('/v1/builds/build')
    assert response.status == 200
    data = await response.json()
    assert data['status'] == status
    assert (data['retry_scheduled_at'] is not None) == (status == 'failed')
    assert (data['cleanup_scheduled_at'] is not None) == (status == 'ready')
    assert data['can_retry_automatically'] == (status == 'failed')
    assert 'provider_config_snapshot_json' not in data


async def test_build_usage_pages_shared_allocations_and_filters_access_before_limit(allocation_client, monkeypatch):
    from nexus.distributed.gateways.common.keys import SESSION_FACTORY_KEY
    from nexus.distributed.gateways.control.routes import executions
    from nexus.distributed.persistence.orm import EnvironmentAllocationRow

    async with allocation_client.app[SESSION_FACTORY_KEY]() as session:
        row = await session.get(EnvironmentAllocationRow, 'workers')
        row.dependency_build_id = 'build'
        await session.commit()
    first = await (await allocation_client.get('/v1/builds/build/usage?limit=1')).json()
    assert first['next_cursor']
    second = await (
        await allocation_client.get('/v1/builds/build/usage', params={'limit': 1, 'cursor': first['next_cursor']})
    ).json()
    assert {row['allocation_id'] for row in first['items'] + second['items']} == {'workers', 'orchestration'}
    assert second['next_cursor'] is None

    async def allowed(*args):
        return None, None, frozenset({('different-owner', 'env')})

    monkeypatch.setattr(executions, '_execution_list_environment_access', allowed)
    restricted = await (await allocation_client.get('/v1/builds/build/usage?limit=1')).json()
    assert restricted['items'] == []
    assert restricted['next_cursor'] is None


@pytest.mark.parametrize('path', ['/v1/builds/build', '/v1/builds/build/usage', '/v1/allocations/orchestration'])
async def test_resource_details_do_not_disclose_foreign_owner(allocation_client, monkeypatch, path):
    from nexus.distributed.access.auth import AuthContext
    from nexus.distributed.gateways.control.routes import console_resources, operations

    context = AuthContext(
        principal_id='stranger',
        issuer='test',
        tenant='',
        subject='stranger',
        roles=frozenset(),
        scopes=frozenset({'execution:read'}),
        auth_type='pat',
    )
    monkeypatch.setattr(operations, 'auth_context', lambda _: context)
    monkeypatch.setattr(console_resources, 'auth_context', lambda _: context)
    assert (await allocation_client.get(path)).status == 404


async def test_allocation_owner_can_read_sanitized_details_and_cleanup_is_version_guarded(
    allocation_client, monkeypatch
):
    from nexus.distributed.access.auth import AuthContext
    from nexus.distributed.gateways.control.routes import operations
    from nexus.distributed.repositories.environment import EnvironmentRepository

    context = AuthContext(
        principal_id='local-dev',
        issuer='test',
        tenant='',
        subject='local-dev',
        roles=frozenset(),
        scopes=frozenset({'execution:read'}),
        auth_type='pat',
    )
    original = operations.auth_context
    monkeypatch.setattr(operations, 'auth_context', lambda _: context)
    checked = []

    async def grant(self, owner, environment, **kwargs):
        checked.append((owner, environment, kwargs['required_scope']))

    monkeypatch.setattr(EnvironmentRepository, 'require_grant_scope', grant)
    response = await allocation_client.get('/v1/allocations/orchestration')
    assert response.status == 200
    data = await response.json()
    assert data['active_assignments'] == 0
    assert 'provider_resource_ref' not in data
    assert 'last_error_message' not in data
    assert checked == [('local-dev', 'env', 'execution:read')]
    assert (await allocation_client.post('/v1/allocations/orchestration/force-cleanup')).status == 403
    monkeypatch.setattr(operations, 'auth_context', original)
    version = data['status_version']
    assert (
        await allocation_client.post(f'/v1/allocations/orchestration/force-cleanup?expected_version={version + 1}')
    ).status == 409
    response = await allocation_client.post(f'/v1/allocations/orchestration/force-cleanup?expected_version={version}')
    assert response.status == 200
    assert (await response.json())['status'] == 'draining'
