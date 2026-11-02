from __future__ import annotations

import dataclasses
import hashlib
import ssl

from datetime import datetime, timedelta

import pytest

from aiohttp import web
from aiohttp.test_utils import TestServer

from nexus.distributed.config import DistributedConfig
from nexus.distributed.config.views import environment_config_view
from nexus.distributed.environment.allocator import (
    AllocationSpecConflictError,
    KubernetesEnvironmentAllocator,
)
from nexus.distributed.environment.kubernetes import (
    KubernetesWorkloadClient,
    WorkloadConflictError,
    WorkloadNotFoundError,
    WorkloadPermanentError,
)
from nexus.distributed.environment.ports import AllocationSnapshot
from nexus.distributed.environment.provider_config import TkexZhiyanLogConfig
from nexus.distributed.environment.providers import WorkloadClientResolver
from nexus.distributed.environment.rendering import AllocationWorkloadRenderer, TkexAllocationWorkloadRenderer
from nexus.distributed.environment.tkex import TkexWorkloadClient


def test_allocation_snapshot_rejects_identity_longer_than_provider_contract() -> None:
    with pytest.raises(ValueError, match='invalid_pool_id'):
        AllocationSnapshot(
            allocation_id='allocation-a',
            top_level_execution_id='execution-a',
            owner_principal_id='principal-a',
            environment_id='environment-a',
            allocation_type='dynamic',
            provider_type='kubernetes',
            pool_id='p' * 65,
            workload='execution',
            status='requested',
            desired_manager_count=1,
            ready_manager_count=0,
            cpu_millicores=1_000,
            memory_bytes=1024,
            slots=1,
            allocation_spec={},
            allocation_spec_hash='a' * 64,
            provider_resource_ref=None,
        )


async def test_tkex_workload_client_request_contract(tmp_path, monkeypatch) -> None:
    requests = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.json() if request.can_read_body else None
        requests.append(
            {
                'method': request.method,
                'path': request.path,
                'api_account': request.headers.get('TKEX-Api-Account'),
                'timestamp': request.headers.get('TKEX-Api-TimeStamp'),
                'signature': request.headers.get('TKEX-Api-Sig'),
                'cluster': request.headers.get('X-TKE-ClusterName'),
                'project': request.headers.get('X-Request-ProjectName'),
                'content_type': request.headers.get('Content-Type'),
                'body': body,
            }
        )
        return web.json_response({'method': request.method})

    app = web.Application()
    app.router.add_route('*', '/{tail:.*}', handler)
    server = TestServer(app)
    await server.start_server()
    token_file = tmp_path / 'tkex-token'
    token_file.write_text('tkex-secret\n')
    monkeypatch.setattr('nexus.distributed.environment.tkex.time.time', lambda: 1_700_000_000)
    client = TkexWorkloadClient(
        base_url=str(server.make_url('/')).rstrip('/'),
        namespace='nexus-system',
        api_account='nexus-account',
        api_token=token_file.read_text().strip(),
        cluster_name='cluster-a',
        project_name='project-a',
    )
    try:
        assert await client.get_stateful_workload('runner-a') == {'method': 'GET'}
        assert await client.create_stateful_workload({'kind': 'StatefulSetPlus'}) == {'method': 'POST'}
        assert await client.patch_stateful_workload('runner-a', {'spec': {'replicas': 2}}) == {'method': 'PATCH'}
        assert await client.delete_stateful_workload('runner-a') == {'method': 'DELETE'}
        assert await client.list_pods() == {'method': 'GET'}
        assert await client.get_pod('runner-a-0') == {'method': 'GET'}
        assert await client.delete_pod('runner-a-0', uid='pod-uid-a') == {'method': 'DELETE'}
    finally:
        await client.close()
        await server.close()

    collection_path = '/v2/forward/stke/apis/platform.stke/v1alpha1/namespaces/nexus-system/statefulsetpluses'
    pods_path = '/v2/forward/stke/api/v1/namespaces/nexus-system/pods'
    assert [(item['method'], item['path']) for item in requests] == [
        ('GET', f'{collection_path}/runner-a'),
        ('POST', collection_path),
        ('PATCH', f'{collection_path}/runner-a'),
        ('DELETE', f'{collection_path}/runner-a'),
        ('GET', pods_path),
        ('GET', f'{pods_path}/runner-a-0'),
        ('DELETE', f'{pods_path}/runner-a-0'),
    ]
    for item in requests:
        expected_signature = hashlib.sha256(f'tkex-secret1700000000{item["path"]}'.encode()).hexdigest()
        assert item['api_account'] == 'nexus-account'
        assert item['timestamp'] == '1700000000'
        assert item['signature'] == expected_signature
        assert item['cluster'] == 'cluster-a'
        assert item['project'] == 'project-a'
    assert requests[1]['content_type'] == 'application/json'
    assert requests[1]['body'] == {'kind': 'StatefulSetPlus'}
    assert requests[2]['content_type'] == 'application/merge-patch+json'
    assert requests[2]['body'] == {'spec': {'replicas': 2}}
    assert requests[-1]['content_type'] == 'application/json'
    assert requests[-1]['body'] == {
        'apiVersion': 'v1',
        'kind': 'DeleteOptions',
        'gracePeriodSeconds': 0,
        'propagationPolicy': 'Background',
        'preconditions': {'uid': 'pod-uid-a'},
    }


async def test_tkex_workload_client_maps_api_errors(tmp_path) -> None:
    async def handler(request: web.Request) -> web.Response:
        name = request.path.rsplit('/', 1)[-1]
        if name == 'missing':
            return web.json_response({'message': 'missing'}, status=404)
        if name == 'conflict':
            return web.json_response({'message': 'conflict'}, status=409)
        return web.Response(text='upstream unavailable', status=502)

    app = web.Application()
    app.router.add_get('/{tail:.*}', handler)
    server = TestServer(app)
    await server.start_server()
    token_file = tmp_path / 'tkex-token'
    token_file.write_text('tkex-secret')
    client = TkexWorkloadClient(
        base_url=str(server.make_url('/')).rstrip('/'),
        namespace='nexus-system',
        api_account='nexus-account',
        api_token=token_file.read_text(),
        cluster_name='cluster-a',
        project_name='project-a',
    )
    try:
        with pytest.raises(WorkloadNotFoundError):
            await client.get_stateful_workload('missing')
        with pytest.raises(WorkloadConflictError):
            await client.get_stateful_workload('conflict')
        with pytest.raises(RuntimeError, match='tkex_api_502'):
            await client.get_stateful_workload('failed')
    finally:
        await client.close()
        await server.close()


async def test_tkex_workload_client_rejects_body_level_error_on_http_200(tmp_path) -> None:
    async def handler(request: web.Request) -> web.Response:
        del request
        return web.json_response({'code': 10042, 'reason': 'CreateRejected', 'message': 'quota exceeded'})

    app = web.Application()
    app.router.add_get('/{tail:.*}', handler)
    server = TestServer(app)
    await server.start_server()
    token_file = tmp_path / 'tkex-token'
    token_file.write_text('tkex-secret')
    client = TkexWorkloadClient(
        base_url=str(server.make_url('/')).rstrip('/'),
        namespace='nexus-system',
        api_account='nexus-account',
        api_token=token_file.read_text(),
        cluster_name='cluster-a',
        project_name='project-a',
    )
    try:
        with pytest.raises(WorkloadPermanentError, match='tkex_api_body_error.*10042.*CreateRejected.*quota exceeded'):
            await client.get_stateful_workload('rejected')
    finally:
        await client.close()
        await server.close()


async def test_tkex_workload_client_maps_body_level_not_found_on_http_200(tmp_path) -> None:
    async def handler(request: web.Request) -> web.Response:
        del request
        message = '[Failure]secrets "nexus-build-artifacts" not found'
        return web.json_response({'code': 404, 'reason': message, 'message': message})

    app = web.Application()
    app.router.add_get('/{tail:.*}', handler)
    server = TestServer(app)
    await server.start_server()
    token_file = tmp_path / 'tkex-token'
    token_file.write_text('tkex-secret')
    client = TkexWorkloadClient(
        base_url=str(server.make_url('/')).rstrip('/'),
        namespace='nexus-system',
        api_account='nexus-account',
        api_token=token_file.read_text(),
        cluster_name='cluster-a',
        project_name='project-a',
    )
    try:
        with pytest.raises(WorkloadNotFoundError):
            await client.get_secret('nexus-build-artifacts')
    finally:
        await client.close()
        await server.close()


async def test_tkex_workload_client_unwraps_documented_success_envelopes(tmp_path) -> None:
    stateful = {
        'apiVersion': 'platform.stke/v1alpha1',
        'kind': 'StatefulSetPlus',
        'metadata': {'name': 'runner-a', 'uid': 'stateful-uid'},
        'status': {'readyReplicas': 2},
    }
    job = {
        'apiVersion': 'batch/v1',
        'kind': 'Job',
        'metadata': {'name': 'builder-a', 'uid': 'job-uid'},
        'status': {'succeeded': 1},
    }
    secret_status = {
        'apiVersion': 'v1',
        'kind': 'Status',
        'status': 'Success',
        'details': {'name': 'builder-secret'},
    }

    async def handler(request: web.Request) -> web.Response:
        if 'statefulsetpluses' in request.path:
            return web.json_response({'code': 0, 'data': stateful})
        if '/jobs' in request.path:
            return web.json_response({'code': '0', 'data': job})
        return web.json_response({'code': 0, 'data': secret_status})

    app = web.Application()
    app.router.add_route('*', '/{tail:.*}', handler)
    server = TestServer(app)
    await server.start_server()
    token_file = tmp_path / 'tkex-token'
    token_file.write_text('tkex-secret')
    client = TkexWorkloadClient(
        base_url=str(server.make_url('/')).rstrip('/'),
        namespace='nexus-system',
        api_account='nexus-account',
        api_token=token_file.read_text(),
        cluster_name='cluster-a',
        project_name='project-a',
    )
    try:
        assert await client.get_stateful_workload('runner-a') == stateful
        assert await client.create_job({'kind': 'Job'}) == job
        assert await client.delete_secret('builder-secret') == secret_status
    finally:
        await client.close()
        await server.close()


@pytest.mark.parametrize(
    'payload',
    [
        {'code': 0},
        {'code': 0, 'data': None},
        {'code': 0, 'data': ['not', 'a', 'resource']},
    ],
)
async def test_tkex_workload_client_rejects_malformed_success_envelope(tmp_path, payload) -> None:
    async def handler(request: web.Request) -> web.Response:
        del request
        return web.json_response(payload)

    app = web.Application()
    app.router.add_get('/{tail:.*}', handler)
    server = TestServer(app)
    await server.start_server()
    token_file = tmp_path / 'tkex-token'
    token_file.write_text('tkex-secret')
    client = TkexWorkloadClient(
        base_url=str(server.make_url('/')).rstrip('/'),
        namespace='nexus-system',
        api_account='nexus-account',
        api_token=token_file.read_text(),
        cluster_name='cluster-a',
        project_name='project-a',
    )
    try:
        with pytest.raises(WorkloadPermanentError, match='tkex_api_invalid_success_envelope'):
            await client.get_stateful_workload('runner-a')
    finally:
        await client.close()
        await server.close()


async def test_kubernetes_workload_client_request_contract(tmp_path) -> None:
    requests = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.json() if request.can_read_body else None
        requests.append(
            {
                'method': request.method,
                'path': request.path,
                'authorization': request.headers.get('Authorization'),
                'content_type': request.headers.get('Content-Type'),
                'body': body,
            }
        )
        return web.json_response({'method': request.method})

    app = web.Application()
    app.router.add_route('*', '/{tail:.*}', handler)
    server = TestServer(app)
    await server.start_server()
    token_file = tmp_path / 'service-account-token'
    token_file.write_text('kubernetes-secret\n')
    client = KubernetesWorkloadClient(
        namespace='nexus-system',
        api_url=str(server.make_url('/')).rstrip('/'),
        token_file=str(token_file),
        ca_file=str(tmp_path / 'missing-ca.crt'),
    )
    try:
        assert await client.get_stateful_workload('runner-a') == {'method': 'GET'}
        assert await client.create_stateful_workload({'kind': 'StatefulSet'}) == {'method': 'POST'}
        assert await client.patch_stateful_workload('runner-a', {'spec': {'replicas': 2}}) == {'method': 'PATCH'}
        assert await client.delete_stateful_workload('runner-a') == {'method': 'DELETE'}
    finally:
        await client.close()
        await server.close()

    collection_path = '/apis/apps/v1/namespaces/nexus-system/statefulsets'
    assert [(item['method'], item['path']) for item in requests] == [
        ('GET', f'{collection_path}/runner-a'),
        ('POST', collection_path),
        ('PATCH', f'{collection_path}/runner-a'),
        ('DELETE', f'{collection_path}/runner-a'),
    ]
    assert all(item['authorization'] == 'Bearer kubernetes-secret' for item in requests)
    assert requests[1]['content_type'] == 'application/json'
    assert requests[1]['body'] == {'kind': 'StatefulSet'}
    assert requests[2]['content_type'] == 'application/merge-patch+json'
    assert requests[2]['body'] == {'spec': {'replicas': 2}}
    assert requests[3]['content_type'] == 'application/json'
    assert requests[3]['body'] == {
        'apiVersion': 'v1',
        'kind': 'DeleteOptions',
        'propagationPolicy': 'Background',
    }


async def test_kubernetes_workload_client_classifies_permanent_and_transient_errors(tmp_path) -> None:
    async def handler(request: web.Request) -> web.Response:
        status = 403 if request.path.endswith('/forbidden') else 429
        return web.json_response({'message': 'rejected'}, status=status)

    app = web.Application()
    app.router.add_get('/{tail:.*}', handler)
    server = TestServer(app)
    await server.start_server()
    token_file = tmp_path / 'service-account-token'
    token_file.write_text('token')
    client = KubernetesWorkloadClient(
        namespace='nexus-system',
        api_url=str(server.make_url('/')).rstrip('/'),
        token_file=str(token_file),
        ca_file=str(tmp_path / 'missing-ca'),
    )
    try:
        with pytest.raises(WorkloadPermanentError, match='kubernetes_api_403'):
            await client.get_stateful_workload('forbidden')
        with pytest.raises(RuntimeError, match='kubernetes_api_429'):
            await client.get_stateful_workload('rate-limited')
    finally:
        await client.close()
        await server.close()


async def test_kubernetes_workload_client_can_disable_strict_x509_validation(tmp_path, monkeypatch) -> None:
    token_file = tmp_path / 'service-account-token'
    ca_file = tmp_path / 'ca.crt'
    token_file.write_text('token')
    ca_file.write_text('ca')
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.verify_flags |= ssl.VERIFY_X509_STRICT
    monkeypatch.setattr(ssl, 'create_default_context', lambda *, cafile: context)
    client = KubernetesWorkloadClient(
        namespace='nexus-system',
        api_url='https://kubernetes.example',
        token_file=str(token_file),
        ca_file=str(ca_file),
        tls_strict=False,
    )

    try:
        await client._client()
        assert context.verify_flags & ssl.VERIFY_X509_STRICT == 0
    finally:
        await client.close()


async def test_kubernetes_workload_client_rebuilds_owned_session_after_ca_rotation(tmp_path, monkeypatch) -> None:
    token_file = tmp_path / 'service-account-token'
    ca_file = tmp_path / 'ca.crt'
    token_file.write_text('token')
    ca_file.write_text('ca-one')
    sessions = []
    contexts = []

    class Session:
        def __init__(self, **_kwargs):
            self.closed = False
            sessions.append(self)

        async def close(self):
            self.closed = True

    monkeypatch.setattr(ssl, 'create_default_context', lambda *, cafile: contexts.append(cafile) or True)
    monkeypatch.setattr('nexus.distributed.environment.kubernetes.aiohttp.TCPConnector', lambda **_kwargs: object())
    monkeypatch.setattr('nexus.distributed.environment.kubernetes.aiohttp.ClientSession', Session)
    client = KubernetesWorkloadClient(
        namespace='nexus-system',
        api_url='https://kubernetes.example',
        token_file=str(token_file),
        ca_file=str(ca_file),
    )

    try:
        first = await client._client()
        ca_file.write_text('ca-two-longer')
        second = await client._client()

        assert first is not second
        assert first.closed is True
        assert second.closed is False
        assert contexts == [str(ca_file), str(ca_file)]
    finally:
        await client.close()


def test_environment_controller_selects_provider_configuration_by_ref(tmp_path) -> None:
    from nexus.distributed.environment.controller import EnvironmentController

    del tmp_path
    config = DistributedConfig(
        database_url='sqlite+aiosqlite:///nexus.db',
        environment_namespace='nexus-workloads',
        environment_database_url_secret_name='nexus-database',
        secrets={
            'provider_credentials': {
                'cluster-a': {'provider_type': 'kubernetes', 'token': 'token-a', 'ca_certificate': 'ca'},
                'cluster-b': {'provider_type': 'kubernetes', 'token': 'token-b', 'ca_certificate': 'ca'},
                'tkex-a': {'provider_type': 'tkex', 'token': 'tkex-token'},
            }
        },
    )
    controller = EnvironmentController(
        session_factory=None,  # type: ignore[arg-type]
        config=config,
        client_resolver=WorkloadClientResolver(config=config),
    )

    allocator_a = controller._allocator_for(
        'kubernetes',
        'cluster-a',
        provider_config_snapshot={
            'provider_type': 'kubernetes',
            'namespace': 'nexus-workloads',
            'auth_type': 'secret',
            'api_url': 'https://cluster-a.example',
            'credential_ref': 'cluster-a',
        },
    )
    allocator_b = controller._allocator_for(
        'kubernetes',
        'cluster-b',
        provider_config_snapshot={
            'provider_type': 'kubernetes',
            'namespace': 'nexus-workloads',
            'auth_type': 'secret',
            'api_url': 'https://cluster-b.example',
            'credential_ref': 'cluster-b',
            'tls_strict': False,
        },
    )
    tkex_allocator = controller._allocator_for(
        'tkex',
        'tkex-a',
        provider_config_snapshot={
            'provider_type': 'tkex',
            'namespace': 'nexus-workloads',
            'base_url': 'https://tkex.example',
            'api_account': 'account-a',
            'credential_ref': 'tkex-a',
            'cluster_name': 'cluster-name-a',
            'project_name': 'project-a',
            'zhiyan_log': {
                'log_name': 'dev_nexus_dynamic_manager_log',
                'agent_image': 'mirrors.tencent.com/zhiyan-agent/agent-integration:0.0.12',
                'volume_size_bytes': 4 * 1024**3,
            },
        },
    )

    assert allocator_a is not allocator_b
    assert allocator_a.client.api_url == 'https://cluster-a.example'
    assert allocator_a.client.namespace == 'nexus-workloads'
    assert allocator_a.client.token == 'token-a'
    assert allocator_a.renderer.config.environment_namespace == 'nexus-workloads'
    assert allocator_b.client.api_url == 'https://cluster-b.example'
    assert allocator_b.client.namespace == 'nexus-workloads'
    assert allocator_b.client.token == 'token-b'
    assert allocator_b.client.tls_strict is False
    assert tkex_allocator.client.base_url == 'https://tkex.example'
    assert tkex_allocator.client.namespace == 'nexus-workloads'
    assert tkex_allocator.client.cluster_name == 'cluster-name-a'
    assert tkex_allocator.renderer.config.environment_namespace == 'nexus-workloads'
    assert tkex_allocator.renderer.zhiyan_log.log_name == 'dev_nexus_dynamic_manager_log'
    with pytest.raises(ValueError, match='allocation_provider_config_snapshot_required'):
        controller._allocator_for('kubernetes', 'cluster-typo')
    with pytest.raises(ValueError, match='environment_provider_config_type_mismatch'):
        controller._allocator_for(
            'kubernetes',
            'tkex-a-as-kubernetes',
            provider_config_snapshot={
                'provider_type': 'tkex',
                'namespace': 'nexus-workloads',
                'base_url': 'https://tkex.example',
                'api_account': 'account-a',
                'credential_ref': 'tkex-a',
                'cluster_name': 'cluster-name-a',
                'project_name': 'project-a',
            },
        )


def test_environment_controller_uses_immutable_provider_snapshot_after_registry_change(tmp_path) -> None:
    from nexus.distributed.environment.controller import EnvironmentController

    del tmp_path
    config = DistributedConfig(
        database_url='sqlite+aiosqlite:///nexus.db',
        environment_namespace='new-workloads',
        environment_database_url_secret_name='nexus-database',
        secrets={
            'provider_credentials': {
                'old-cluster': {'provider_type': 'kubernetes', 'token': 'token', 'ca_certificate': 'ca'}
            }
        },
    )
    controller = EnvironmentController(
        session_factory=None,  # type: ignore[arg-type]
        config=config,
        client_resolver=WorkloadClientResolver(config=config),
    )

    allocator = controller._allocator_for(
        'kubernetes',
        'kubernetes/removed',
        provider_config_snapshot={
            'provider_type': 'kubernetes',
            'namespace': 'old-workloads',
            'auth_type': 'secret',
            'api_url': 'https://old-cluster.example',
            'credential_ref': 'old-cluster',
        },
    )

    assert allocator.client.namespace == 'old-workloads'
    assert allocator.client.api_url == 'https://old-cluster.example'
    assert allocator.client.token == 'token'


@pytest.mark.parametrize(
    ('client_type', 'kwargs', 'statefulset_path', 'job_path'),
    [
        (
            KubernetesWorkloadClient,
            {'api_url': 'server', 'token_file': 'token', 'ca_file': 'ca'},
            '/apis/apps/v1/namespaces/nexus-system/statefulsets',
            '/apis/batch/v1/namespaces/nexus-system/jobs',
        ),
        (
            TkexWorkloadClient,
            {
                'base_url': 'server',
                'api_account': 'account',
                'api_token': 'token',
                'cluster_name': 'cluster',
                'project_name': 'project',
            },
            '/v2/forward/stke/apis/platform.stke/v1alpha1/namespaces/nexus-system/statefulsetpluses',
            '/v2/forward/stke/apis/batch/v1/namespaces/nexus-system/jobs',
        ),
    ],
)
async def test_workload_clients_manage_allocation_service_and_stateful_workload(
    tmp_path,
    monkeypatch,
    client_type,
    kwargs,
    statefulset_path,
    job_path,
) -> None:
    requests = []
    delete_bodies = {}

    async def handler(request: web.Request) -> web.Response:
        requests.append((request.method, request.path))
        if request.method == 'DELETE':
            delete_bodies[request.path] = await request.json() if request.can_read_body else None
        return web.json_response({'metadata': {'name': request.path.rsplit('/', 1)[-1]}})

    app = web.Application()
    app.router.add_route('*', '/{tail:.*}', handler)
    server = TestServer(app)
    await server.start_server()
    token_file = tmp_path / 'token'
    token_file.write_text('secret')
    resolved = {
        key: (
            str(server.make_url('/')).rstrip('/')
            if value == 'server'
            else token_file.read_text()
            if key == 'api_token'
            else str(token_file)
            if value == 'token'
            else str(tmp_path / 'missing-ca')
            if value == 'ca'
            else value
        )
        for key, value in kwargs.items()
    }
    if client_type is TkexWorkloadClient:
        monkeypatch.setattr('nexus.distributed.environment.tkex.time.time', lambda: 1_700_000_000)
    client = client_type(namespace='nexus-system', **resolved)
    try:
        await client.get_secret('allocation-a-artifacts')
        await client.create_secret({'kind': 'Secret'})
        await client.patch_secret('allocation-a-artifacts', {'data': {'token': 'rotated'}})
        await client.get_job('build-a')
        await client.create_job({'kind': 'Job'})
        await client.delete_job('build-a')
        await client.get_service('allocation-a-headless')
        await client.create_service({'kind': 'Service'})
        await client.get_stateful_workload('allocation-a')
        await client.create_stateful_workload({'kind': 'StatefulSet'})
        await client.delete_stateful_workload('allocation-a')
        await client.delete_service('allocation-a-headless')
        await client.delete_secret('allocation-a-artifacts')
    finally:
        await client.close()
        await server.close()

    service_path = (
        '/api/v1/namespaces/nexus-system/services'
        if client_type is KubernetesWorkloadClient
        else '/v2/forward/stke/api/v1/namespaces/nexus-system/services'
    )
    secret_path = (
        '/api/v1/namespaces/nexus-system/secrets'
        if client_type is KubernetesWorkloadClient
        else '/v2/forward/stke/api/v1/namespaces/nexus-system/secrets'
    )
    assert requests == [
        ('GET', f'{secret_path}/allocation-a-artifacts'),
        ('POST', secret_path),
        ('PATCH', f'{secret_path}/allocation-a-artifacts'),
        ('GET', f'{job_path}/build-a'),
        ('POST', job_path),
        ('DELETE', f'{job_path}/build-a'),
        ('GET', f'{service_path}/allocation-a-headless'),
        ('POST', service_path),
        ('GET', f'{statefulset_path}/allocation-a'),
        ('POST', statefulset_path),
        ('DELETE', f'{statefulset_path}/allocation-a'),
        ('DELETE', f'{service_path}/allocation-a-headless'),
        ('DELETE', f'{secret_path}/allocation-a-artifacts'),
    ]
    delete_options = {
        'apiVersion': 'v1',
        'kind': 'DeleteOptions',
        'propagationPolicy': 'Background',
    }
    assert delete_bodies[f'{job_path}/build-a'] == delete_options
    assert delete_bodies[f'{statefulset_path}/allocation-a'] == delete_options
    assert delete_bodies[f'{service_path}/allocation-a-headless'] is None
    assert delete_bodies[f'{secret_path}/allocation-a-artifacts'] is None


@pytest.mark.parametrize(
    ('client_type', 'kwargs', 'api_prefix'),
    [
        (
            KubernetesWorkloadClient,
            {'api_url': 'server', 'token_file': 'token', 'ca_file': 'ca'},
            '',
        ),
        (
            TkexWorkloadClient,
            {
                'base_url': 'server',
                'api_account': 'account',
                'api_token': 'token',
                'cluster_name': 'cluster',
                'project_name': 'project',
            },
            '/v2/forward/stke',
        ),
    ],
)
async def test_workload_clients_read_job_failure_diagnostics(
    tmp_path,
    monkeypatch,
    client_type,
    kwargs,
    api_prefix,
) -> None:
    requests = []
    signatures = []

    async def handler(request: web.Request) -> web.Response:
        requests.append((request.path, dict(request.query)))
        signatures.append((request.path, request.headers.get('TKEX-Api-Sig')))
        if request.path.endswith('/log'):
            return web.Response(text='builder failed\nlast line')
        if request.path.endswith('/events'):
            return web.json_response({'items': [{'reason': 'FailedMount'}]})
        return web.json_response({'items': [{'metadata': {'name': 'build-a-pod'}}]})

    app = web.Application()
    app.router.add_route('*', '/{tail:.*}', handler)
    server = TestServer(app)
    await server.start_server()
    token_file = tmp_path / 'token'
    token_file.write_text('secret')
    resolved = {
        key: (
            str(server.make_url('/')).rstrip('/')
            if value == 'server'
            else token_file.read_text()
            if key == 'api_token'
            else str(token_file)
            if value == 'token'
            else str(tmp_path / 'missing-ca')
            if value == 'ca'
            else value
        )
        for key, value in kwargs.items()
    }
    if client_type is TkexWorkloadClient:
        monkeypatch.setattr('nexus.distributed.environment.tkex.time.time', lambda: 1_700_000_000)
    client = client_type(namespace='nexus-system', **resolved)
    try:
        assert await client.list_job_pods('build-a') == [{'metadata': {'name': 'build-a-pod'}}]
        assert await client.list_resource_events('build-a-pod') == [{'reason': 'FailedMount'}]
        assert (
            await client.read_pod_log('build-a-pod', container='builder', tail_lines=50, limit_bytes=256 * 1024)
            == 'builder failed\nlast line'
        )
    finally:
        await client.close()
        await server.close()

    resource_prefix = f'{api_prefix}/api/v1/namespaces/nexus-system'
    assert requests == [
        (f'{resource_prefix}/pods', {'labelSelector': 'job-name=build-a'}),
        (f'{resource_prefix}/events', {'fieldSelector': 'involvedObject.name=build-a-pod'}),
        (
            f'{resource_prefix}/pods/build-a-pod/log',
            {'container': 'builder', 'tailLines': '50', 'timestamps': 'true', 'limitBytes': str(256 * 1024)},
        ),
    ]
    if client_type is TkexWorkloadClient:
        assert signatures == [
            (path, hashlib.sha256(f'secret1700000000{path}'.encode()).hexdigest()) for path, _query in requests
        ]


@pytest.mark.parametrize(
    ('client_type', 'kwargs'),
    [
        (KubernetesWorkloadClient, {'api_url': 'server', 'token_file': 'token', 'ca_file': 'ca'}),
        (
            TkexWorkloadClient,
            {
                'base_url': 'server',
                'api_account': 'account',
                'api_token': 'token',
                'cluster_name': 'cluster',
                'project_name': 'project',
            },
        ),
    ],
)
async def test_workload_clients_bound_pod_log_response_bytes(tmp_path, client_type, kwargs) -> None:
    observed_query: dict[str, str] = {}

    async def handler(request: web.Request) -> web.Response:
        observed_query.update(request.query)
        return web.Response(body=b'x' * 4096)

    app = web.Application()
    app.router.add_get('/{tail:.*}', handler)
    server = TestServer(app)
    await server.start_server()
    token_file = tmp_path / 'token'
    token_file.write_text('secret')
    resolved = {
        key: (
            str(server.make_url('/')).rstrip('/')
            if value == 'server'
            else token_file.read_text()
            if key == 'api_token'
            else str(token_file)
            if value == 'token'
            else str(tmp_path / 'missing-ca')
            if value == 'ca'
            else value
        )
        for key, value in kwargs.items()
    }
    client = client_type(namespace='nexus-system', **resolved)
    try:
        log = await client.read_pod_log('build-a-pod', container='builder', tail_lines=50, limit_bytes=128)
    finally:
        await client.close()
        await server.close()

    assert log == 'x' * 128
    assert observed_query['limitBytes'] == '128'


def test_allocation_renderer_uses_pod_identity_and_scoped_application_gateway_credential() -> None:
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
        desired_manager_count=2,
        ready_manager_count=0,
        cpu_millicores=4000,
        memory_bytes=8 * 1024**3,
        slots=40,
        allocation_spec={
            'version': 1,
            'code': {'mode': 'config_only'},
            'base_image': 'nexus-workload:test',
            'workspace_size_bytes': 4 * 1024**3,
            'host_env': [
                {'name': 'HOST_ENV_MVP_PROBE', 'value': 'ok'},
                {'name': 'HYTUNER', 'value': '1'},
            ],
            'artifact_read_credential': {'secret_name': 'allocation-a-read', 'token_key': 'token'},
        },
        allocation_spec_hash='a' * 64,
        provider_resource_ref=None,
        provider_config_ref='tkex/shanghai',
        created_at=datetime(2026, 8, 3, 12, 0, 0),
        provisioning_timeout_seconds=300,
    )
    config = DistributedConfig(
        config_store_required=False,
        environment_namespace='nexus-system',
        environment_database_url_secret_name='nexus-database',
        environment_image_pull_secrets=['csighub-yunzhiluo', 'csighub-taolin'],
        application_internal_base_url='https://nexus-application.internal.example',
        batch_submit_batch_size=321,
        manager_heartbeat_interval=7,
        manager_healthy_ttl=timedelta(seconds=21),
        manager_self_fence_timeout=timedelta(seconds=42),
        manager_dead_ttl=timedelta(seconds=84),
    )
    renderer = AllocationWorkloadRenderer(
        config=environment_config_view(config),
        cluster_name='cls-g84u227v',
        resource_request_ratio=0.25,
    )

    rendered = renderer.render(snapshot, resource_kind='StatefulSet')

    workload = rendered.workload
    assert workload['kind'] == 'StatefulSet'
    assert workload['spec']['serviceName'] == workload['metadata']['name']
    assert workload['spec']['podManagementPolicy'] == 'Parallel'
    assert workload['spec']['replicas'] == 2
    container = workload['spec']['template']['spec']['containers'][0]
    pod_spec = workload['spec']['template']['spec']
    assert len(pod_spec['containers']) == 1
    assert pod_spec['automountServiceAccountToken'] is False
    assert pod_spec['enableServiceLinks'] is False
    assert pod_spec['imagePullSecrets'] == [
        {'name': 'csighub-yunzhiluo'},
        {'name': 'csighub-taolin'},
    ]
    assert pod_spec['securityContext'] == {
        'runAsNonRoot': True,
        'runAsUser': 1000,
        'runAsGroup': 1000,
        'fsGroup': 1000,
        'seccompProfile': {'type': 'RuntimeDefault'},
    }
    assert container['securityContext'] == {
        'allowPrivilegeEscalation': False,
        'capabilities': {'drop': ['ALL']},
        'readOnlyRootFilesystem': True,
    }
    env = {item['name']: item.get('value') for item in container['env']}
    assert env['TMPDIR'] == '/tmp'
    assert env['TMP'] == '/tmp'
    assert env['TEMP'] == '/tmp'
    assert env['HOST_ENV_MVP_PROBE'] == 'ok'
    assert env['HYTUNER'] == '1'
    assert {'name': 'data', 'mountPath': '/data'} in container['volumeMounts']
    assert {'name': 'tmp', 'mountPath': '/tmp'} in container['volumeMounts']
    assert {
        'name': 'nexus-secrets',
        'mountPath': '/var/run/secrets/nexus',
        'readOnly': True,
    } in container['volumeMounts']
    assert container['resources'] == {
        'limits': {'cpu': '2000m', 'memory': str(4 * 1024**3)},
        'requests': {'cpu': '500m', 'memory': str(1024**3)},
    }
    env_names = {item['name'] for item in container['env']}
    assert 'NEXUS_DIST_APPLICATION_INTERNAL_BASE_URL' in env_names
    assert 'NEXUS_DIST_ARTIFACT_READ_TOKEN_FILE' in env_names
    assert 'NEXUS_DIST_INTERNAL_TOKEN_FILE' not in env_names
    assert 'NEXUS_DIST_INTERNAL_TOKEN' not in env_names
    env_by_name = {item['name']: item for item in container['env']}
    assert env_by_name['POD_IP']['valueFrom']['fieldRef']['fieldPath'] == 'status.podIP'
    assert env_by_name['POD_UID']['valueFrom']['fieldRef']['fieldPath'] == 'metadata.uid'
    assert env_by_name['NEXUS_DIST_PROVIDER_CONFIG_REF']['value'] == 'tkex/shanghai'
    assert env_by_name['NEXUS_DIST_CLUSTER_NAME']['value'] == 'cls-g84u227v'
    assert env_by_name['NEXUS_DIST_APPLICATION_INTERNAL_BASE_URL']['value'] == (
        'https://nexus-application.internal.example'
    )
    assert env_by_name['NEXUS_DIST_RUN_TRACE_ARTIFACT_ROOT']['value'] == '/data/nexus-environment/run-trace'
    assert env_by_name['NEXUS_DIST_DATABASE_URL_FILE']['value'] == '/var/run/secrets/nexus/database/url'
    assert env_by_name['NEXUS_DIST_CONFIG_STORE_REQUIRED']['value'] == '0'
    assert env_by_name['NEXUS_DIST_ARTIFACT_READ_TOKEN_FILE']['value'] == (
        '/var/run/secrets/nexus/artifact/reader/token'
    )
    assert env_by_name['NEXUS_DIST_ARTIFACT_DOWNLOAD_DEADLINE_AT']['value'] == '2026-08-03T12:05:00'
    assert 'WEDATA_KEY_FILE' not in env_by_name
    assert env_by_name['NEXUS_DIST_DB_POOL_SIZE']['value'] == '1'
    assert env_by_name['NEXUS_DIST_DB_MAX_OVERFLOW']['value'] == '0'
    assert env_by_name['NEXUS_DIST_ARTIFACT_RUN_RESULT_MAX_SIZE_BYTES']['value'] == '16777216'
    assert env_by_name['NEXUS_DIST_ARTIFACT_RUN_RESULT_PUBLISH_DEADLINE']['value'] == '1800.0'
    assert env_by_name['NEXUS_DIST_BATCH_SUBMIT_BATCH_SIZE']['value'] == '321'
    assert env_by_name['NEXUS_DIST_MANAGER_HEARTBEAT_INTERVAL']['value'] == '7.0'
    assert env_by_name['NEXUS_DIST_MANAGER_HEALTHY_TTL_SECONDS']['value'] == '21.0'
    assert env_by_name['NEXUS_DIST_MANAGER_SELF_FENCE_TIMEOUT_SECONDS']['value'] == '42.0'
    assert env_by_name['NEXUS_DIST_MANAGER_DEAD_TTL_SECONDS']['value'] == '84.0'
    assert '--service-name' not in container['args']
    assert '--endpoint-url' not in container['args']
    assert 'NEXUS_DIST_ARTIFACT_GENERIC_BASE_URL' not in env_names
    assert 'NEXUS_DIST_ARTIFACT_READER_TOKEN_FILE' not in env_names
    projected = next(
        volume for volume in workload['spec']['template']['spec']['volumes'] if volume['name'] == 'nexus-secrets'
    )
    secret_names = {source['secret']['name'] for source in projected['projected']['sources']}
    assert secret_names == {'allocation-a-read', 'nexus-database'}
    projected_paths = {
        item['path'] for source in projected['projected']['sources'] for item in source['secret']['items']
    }
    assert projected_paths == {'artifact/reader/token', 'database/url'}
    data_volume = next(volume for volume in pod_spec['volumes'] if volume['name'] == 'data')
    assert data_volume['emptyDir']['sizeLimit'] == str(4 * 1024**3)
    tmp_volume = next(volume for volume in pod_spec['volumes'] if volume['name'] == 'tmp')
    assert tmp_volume['emptyDir']['sizeLimit'] == str(1024**3)
    annotations = workload['metadata']['annotations']
    assert annotations['nexus.io/allocation-id'] == 'allocation-a'
    assert annotations['nexus.io/allocation-spec-hash'] == 'a' * 64

    strict_renderer = AllocationWorkloadRenderer(
        config=environment_config_view(config.model_copy(update={'config_store_required': True}))
    )
    strict_container = strict_renderer.render(snapshot, resource_kind='StatefulSet').workload['spec']['template'][
        'spec'
    ]['containers'][0]
    strict_env = {item['name']: item.get('value') for item in strict_container['env']}
    assert strict_env['NEXUS_DIST_CONFIG_STORE_REQUIRED'] == '1'

    orchestration = dataclasses.replace(
        snapshot,
        allocation_id='allocation-b',
        pool_id='orchestration',
        workload='orchestration',
        allocation_spec_hash='b' * 64,
    )
    orchestration_container = renderer.render(orchestration, resource_kind='StatefulSet').workload['spec']['template'][
        'spec'
    ]['containers'][0]
    orchestration_env = {item['name']: item.get('value') for item in orchestration_container['env']}
    assert orchestration_env['NEXUS_DIST_DB_POOL_SIZE'] == '4'
    assert orchestration_env['NEXUS_DIST_DB_MAX_OVERFLOW'] == '2'
    assert 'WEDATA_KEY_FILE' not in orchestration_env

    override_renderer = AllocationWorkloadRenderer(
        config=DistributedConfig(
            environment_database_url_secret_name='nexus-database',
            environment_extra_env=[
                {
                    'name': 'WEDATA_KEY_FILE',
                    'value': '/tmp/untrusted-wedata-key',
                }
            ],
        )
    )
    with pytest.raises(ValueError, match='invalid_dynamic_environment_variable:WEDATA_KEY_FILE'):
        override_renderer.render(snapshot, resource_kind='StatefulSet')


def test_tkex_allocation_renderer_adds_zhiyan_log_sidecar() -> None:
    snapshot = AllocationSnapshot(
        allocation_id='allocation-zhiyan',
        top_level_execution_id='execution-a',
        owner_principal_id='principal-a',
        environment_id='environment-a',
        allocation_type='dynamic',
        provider_type='tkex',
        pool_id='execution',
        workload='execution',
        status='requested',
        desired_manager_count=1,
        ready_manager_count=0,
        cpu_millicores=2_000,
        memory_bytes=4 * 1024**3,
        slots=20,
        allocation_spec={
            'version': 1,
            'code': {'mode': 'config_only'},
            'artifact_read_credential': {'secret_name': 'allocation-zhiyan-read', 'token_key': 'token'},
        },
        allocation_spec_hash='a' * 64,
        provider_resource_ref=None,
        provider_config_ref='tkex/shanghai',
    )
    renderer = TkexAllocationWorkloadRenderer(
        config=environment_config_view(
            DistributedConfig(
                environment_database_url_secret_name='nexus-database',
                environment_cache_root='/data/test-nexus-environment',
                environment_extra_env=[{'name': 'NEXUS_DIST_LOG_DIR', 'value': '/data/test-nexus-environment/logs'}],
            )
        ),
        cluster_name='cls-shanghai',
        workload_namespace='nexus-workloads',
        zhiyan_log=TkexZhiyanLogConfig(
            log_name='dev_nexus_dynamic_manager_log',
            agent_image='mirrors.tencent.com/zhiyan-agent/agent-integration:0.0.12',
            volume_size_bytes=4 * 1024**3,
        ),
    )

    workload = renderer.render(snapshot, resource_kind='StatefulSetPlus').workload

    assert workload['kind'] == 'StatefulSetPlus'
    pod_spec = workload['spec']['template']['spec']
    containers = {container['name']: container for container in pod_spec['containers']}
    assert set(containers) == {'instance-manager', 'zhiyan-agent'}
    manager = containers['instance-manager']
    manager_env = {item['name']: item.get('value') for item in manager['env']}
    manager_log_dir = '/data/test-nexus-environment/logs'
    assert manager_env['NEXUS_DIST_LOG_DIR'] == manager_log_dir
    assert [item['name'] for item in manager['env']].count('NEXUS_DIST_LOG_DIR') == 1
    assert {'name': 'log', 'mountPath': manager_log_dir} in manager['volumeMounts']
    assert {'name': 'cgroup', 'mountPath': '/sys/fs/cgroup', 'readOnly': True} in manager['volumeMounts']
    assert {'name': 'shm', 'mountPath': '/dev/shm'} in manager['volumeMounts']
    assert pod_spec['initContainers'] == [
        {
            'name': 'prepare-manager-cache-root',
            'image': manager['image'],
            'imagePullPolicy': 'IfNotPresent',
            'command': ['/opt/nexus-venv/bin/python'],
            'args': [
                '-c',
                'from pathlib import Path; import sys; Path(sys.argv[1]).mkdir(parents=True, exist_ok=True)',
                '/data/test-nexus-environment',
            ],
            'volumeMounts': [{'name': 'data', 'mountPath': '/data'}],
            'securityContext': {
                'allowPrivilegeEscalation': False,
                'capabilities': {'drop': ['ALL']},
                'readOnlyRootFilesystem': True,
            },
        }
    ]
    agent = containers['zhiyan-agent']
    assert agent == {
        'name': 'zhiyan-agent',
        'image': 'mirrors.tencent.com/zhiyan-agent/agent-integration:0.0.12',
        'imagePullPolicy': 'Always',
        'stdin': True,
        'env': [
            {'name': 'ZHIYAN_LOG_NAME', 'value': 'dev_nexus_dynamic_manager_log'},
            {
                'name': 'POD_NAME',
                'valueFrom': {'fieldRef': {'apiVersion': 'v1', 'fieldPath': 'metadata.name'}},
            },
        ],
        'resources': {
            'limits': {'cpu': '0', 'memory': '0'},
            'requests': {'cpu': '0', 'memory': '0'},
        },
        'securityContext': {'runAsNonRoot': False, 'runAsUser': 0},
        'volumeMounts': [
            {'name': 'cgroup', 'mountPath': '/sys/fs/cgroup', 'readOnly': True},
            {'name': 'shm', 'mountPath': '/dev/shm'},
            {'name': 'log', 'mountPath': '/data/logs'},
        ],
    }
    volumes = {volume['name']: volume for volume in pod_spec['volumes']}
    assert volumes['log'] == {'name': 'log', 'emptyDir': {'sizeLimit': str(4 * 1024**3)}}
    assert volumes['cgroup'] == {'name': 'cgroup', 'hostPath': {'path': '/sys/fs/cgroup', 'type': ''}}
    assert volumes['shm'] == {'name': 'shm', 'emptyDir': {'medium': 'Memory'}}

    for invalid_allocation in (
        dataclasses.replace(snapshot, provider_type='kubernetes'),
        dataclasses.replace(snapshot, allocation_type='predefined'),
    ):
        with pytest.raises(ValueError, match='tkex_renderer_requires_dynamic_tkex_allocation'):
            renderer.render(invalid_allocation, resource_kind='StatefulSetPlus')


@pytest.mark.parametrize(
    'existing_env',
    [
        {'name': 'NEXUS_DIST_LOG_DIR', 'value': '/data/other/logs'},
        {
            'name': 'NEXUS_DIST_LOG_DIR',
            'valueFrom': {'secretKeyRef': {'name': 'dist-log-dir', 'key': 'value'}},
        },
    ],
)
def test_tkex_allocation_renderer_rejects_conflicting_zhiyan_log_dir(existing_env: dict[str, object]) -> None:
    renderer = TkexAllocationWorkloadRenderer(
        config=environment_config_view(
            DistributedConfig(
                environment_database_url_secret_name='nexus-database',
                environment_cache_root='/data/test-nexus-environment',
            )
        ),
        cluster_name='cls-shanghai',
        workload_namespace='nexus-workloads',
        zhiyan_log=TkexZhiyanLogConfig(
            log_name='dev_nexus_dynamic_manager_log',
            agent_image='mirrors.tencent.com/zhiyan-agent/agent-integration:0.0.12',
            volume_size_bytes=4 * 1024**3,
        ),
    )
    workload = {
        'spec': {
            'template': {
                'spec': {
                    'containers': [
                        {
                            'name': 'instance-manager',
                            'env': [existing_env],
                        }
                    ]
                }
            }
        }
    }

    with pytest.raises(ValueError, match='tkex_zhiyan_log_dir_conflicts_with_environment_extra_env'):
        renderer._add_zhiyan_log_sidecar(workload)


def test_allocation_renderer_rejects_non_normalized_per_manager_resources() -> None:
    snapshot = AllocationSnapshot(
        allocation_id='allocation-invalid',
        top_level_execution_id='execution-invalid',
        owner_principal_id='principal-a',
        environment_id='environment-a',
        allocation_type='dynamic',
        provider_type='kubernetes',
        pool_id='execution',
        workload='execution',
        status='requested',
        desired_manager_count=2,
        ready_manager_count=0,
        cpu_millicores=2001,
        memory_bytes=2048,
        slots=2,
        allocation_spec={'version': 1, 'code': {'mode': 'config_only'}},
        allocation_spec_hash='a' * 64,
        provider_resource_ref=None,
    )
    renderer = AllocationWorkloadRenderer(
        config=DistributedConfig(
            environment_database_url_secret_name='nexus-database',
        )
    )

    with pytest.raises(ValueError, match='cpu_millicores_not_divisible_by_manager_count'):
        renderer.render(snapshot, resource_kind='StatefulSet')


async def test_kubernetes_allocator_reuses_matching_create_conflict_and_rejects_spec_mismatch() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.secrets = {}
            self.service = None
            self.workload = None
            self.service_creates = 0
            self.workload_creates = 0

        async def get_secret(self, name):
            if name not in self.secrets:
                raise WorkloadNotFoundError(name)
            return self.secrets[name]

        async def create_secret(self, body):
            name = body['metadata']['name']
            if name in self.secrets:
                raise WorkloadConflictError(name)
            self.secrets[name] = body
            return body

        async def patch_secret(self, name, body):
            self.secrets[name] = body
            return body

        async def delete_secret(self, name):
            self.secrets.pop(name, None)
            return {}

        async def get_service(self, name):
            if self.service is None:
                raise WorkloadNotFoundError(name)
            return self.service

        async def create_service(self, body):
            self.service_creates += 1
            if self.service is not None:
                raise WorkloadConflictError(body['metadata']['name'])
            self.service = body
            return body

        async def delete_service(self, name):
            del name
            self.service = None
            return {}

        async def get_stateful_workload(self, name):
            if self.workload is None:
                raise WorkloadNotFoundError(name)
            return self.workload

        async def create_stateful_workload(self, body):
            self.workload_creates += 1
            if self.workload is not None:
                raise WorkloadConflictError(body['metadata']['name'])
            self.workload = {**body, 'status': {'readyReplicas': 1}}
            return self.workload

        async def delete_stateful_workload(self, name):
            del name
            self.workload = None
            return {}

        async def close(self):
            pass

    allocation = AllocationSnapshot(
        allocation_id='allocation-idempotent',
        top_level_execution_id='execution-idempotent',
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
        memory_bytes=1024,
        slots=1,
        allocation_spec={'code': {'mode': 'config_only'}},
        allocation_spec_hash='a' * 64,
        provider_resource_ref=None,
        artifact_credential_secret_name='allocation-idempotent-artifacts',
        artifact_credential_token='credential-token-v1',
    )
    client = FakeClient()
    from nexus.distributed.environment.database_secret import EnvironmentDatabaseSecretSynchronizer

    allocator = KubernetesEnvironmentAllocator(
        client=client,
        renderer=AllocationWorkloadRenderer(
            config=DistributedConfig(
                environment_database_url_secret_name='nexus-database',
            )
        ),
        database_secret_synchronizer=EnvironmentDatabaseSecretSynchronizer(
            namespace='default',
            app_name='nexus-distributed',
            release_name='nexus-dist',
            secret_name='nexus-database',
            secret_key='database-url',
            database_url='sqlite+aiosqlite:///nexus.db',
        ),
    )

    assert (await allocator.create(allocation)).exists
    assert (await allocator.create(allocation)).exists
    assert client.secrets['allocation-idempotent-artifacts']['metadata']['annotations']['nexus.io/credential-sha256']
    assert client.secrets['nexus-database']['metadata']['annotations']['nexus.io/shared-secret'] == 'database-url'
    assert client.service_creates == 0
    assert client.workload_creates == 1

    client.service = {
        'metadata': {
            'name': 'allocation-idempotent-headless',
            'annotations': dict(client.workload['metadata']['annotations']),
        }
    }
    rotated = dataclasses.replace(allocation, artifact_credential_token='credential-token-v2')
    assert (await allocator.observe(rotated)).exists
    assert client.secrets['allocation-idempotent-artifacts']['data']['token'] != 'credential-token-v1'
    assert client.service is not None

    client.workload['metadata']['annotations']['nexus.io/allocation-spec-hash'] = 'b' * 64
    with pytest.raises(AllocationSpecConflictError, match='spec_hash_conflict'):
        await allocator.observe(allocation)
    with pytest.raises(AllocationSpecConflictError, match='spec_hash_conflict'):
        await allocator.delete(allocation)
    assert client.workload is not None

    client.workload['metadata']['annotations']['nexus.io/allocation-spec-hash'] = 'a' * 64
    await allocator.delete(rotated)
    assert 'allocation-idempotent-artifacts' not in client.secrets
    assert 'nexus-database' in client.secrets
    assert client.service is None
    assert not (await allocator.observe_deletion(rotated)).exists
    assert 'allocation-idempotent-artifacts' not in client.secrets
    assert 'nexus-database' in client.secrets
