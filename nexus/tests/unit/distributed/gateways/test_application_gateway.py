from __future__ import annotations

import asyncio
import hashlib
import json

from datetime import timedelta
from types import SimpleNamespace

import aiohttp
import pytest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.config import DistributedConfig
from nexus.distributed.execution.models import (
    ExecutionMode,
    ExecutionStatus,
    RouteStatus,
)
from nexus.distributed.gateways.application.app import create_application_app
from nexus.distributed.gateways.common.proxy import validate_proxy_path as validate_application_proxy_path
from nexus.distributed.instance_manager.internal import _validate_proxy_path as validate_manager_proxy_path
from nexus.distributed.persistence.orm import (
    ArtifactRow,
    ArtifactUploadRow,
    AssignmentRow,
    BatchRunRow,
    DependencyBuildRow,
    EnvironmentAllocationRow,
    EnvironmentRow,
    ExecutionRow,
    InstanceManagerLivenessRow,
    InstanceManagerRow,
    create_schema,
)
from nexus.distributed.repositories.execution import CreateExecution, ExecutionRepository
from nexus.distributed.repositories.manager import ManagerRepository
from nexus.distributed.repositories.scheduler import SchedulerRepository
from nexus.distributed.utils.clock import utcnow


@pytest.mark.parametrize(
    ('top_level_execution_id', 'linked', 'allowed'),
    [
        pytest.param('batch-parent', True, True, id='allocation-linked'),
        pytest.param('batch-parent', False, False, id='allocation-unlinked'),
        pytest.param(None, True, False, id='legacy-ticket-requires-static-id'),
    ],
)
async def test_artifact_ticket_read_scope(top_level_execution_id, linked, allowed):
    from nexus.distributed.artifacts.tickets import ARTIFACT_CONTENT_TICKET_HEADER, ArtifactContentTicketSigner
    from nexus.distributed.gateways.application.app import ARTIFACT_CONTENT_TICKET_SIGNER_KEY
    from nexus.distributed.gateways.application.routes.artifact_shared import _resolve_internal_artifact_scope
    from nexus.distributed.persistence.orm import ArtifactRefRow

    signer = ArtifactContentTicketSigner(
        keys={'current': 'test-secret'},
        current_key_id='current',
        ttl=timedelta(minutes=5),
    )
    ticket = signer.issue(
        execution_id=top_level_execution_id or 'batch-child',
        top_level_execution_id=top_level_execution_id,
        environment_id='env-1',
        allocation_id='alloc-child',
        operations=['artifact:read'],
        artifact_ids=[],
        now=utcnow(),
    )
    execution = SimpleNamespace(
        execution_id='batch-child',
        owner_principal_id='user-1',
        environment_id='env-1',
        allocation_id='alloc-child',
    )
    allocation = SimpleNamespace(
        allocation_id='alloc-child',
        top_level_execution_id='batch-parent',
        owner_principal_id='user-1',
        environment_id='env-1',
        status='ready',
    )
    linked_ref_key = {
        'artifact_id': 'artifact-after-ticket',
        'resource_type': 'execution',
        'resource_id': execution.execution_id,
    }

    class FakeSession:
        async def get(self, model, identity):
            if model is ExecutionRow:
                return execution
            if model is EnvironmentAllocationRow:
                return allocation
            if model is ArtifactRefRow and linked and identity == linked_ref_key:
                return SimpleNamespace(**linked_ref_key)
            return None

    request = SimpleNamespace(
        headers={
            ARTIFACT_CONTENT_TICKET_HEADER: ticket,
            'X-Nexus-Execution-Id': execution.execution_id,
        },
        app={ARTIFACT_CONTENT_TICKET_SIGNER_KEY: signer},
    )

    if allowed:
        scope = await _resolve_internal_artifact_scope(
            request,
            FakeSession(),
            operation='artifact:read',
            artifact_id=linked_ref_key['artifact_id'],
        )
        assert scope.execution_id == execution.execution_id
        assert scope.credential_allocation_id == allocation.allocation_id
    else:
        with pytest.raises(web.HTTPForbidden, match='Forbidden') as exc_info:
            await _resolve_internal_artifact_scope(
                request,
                FakeSession(),
                operation='artifact:read',
                artifact_id=linked_ref_key['artifact_id'],
            )
        assert exc_info.value.text == 'artifact_content_scope_mismatch'


async def test_dependency_build_ticket_reads_only_linked_input_artifacts():
    from nexus.distributed.artifacts.tickets import ARTIFACT_CONTENT_TICKET_HEADER, ArtifactContentTicketSigner
    from nexus.distributed.gateways.application.app import ARTIFACT_CONTENT_TICKET_SIGNER_KEY
    from nexus.distributed.gateways.application.routes.artifact_shared import _resolve_dependency_build_artifact_scope
    from nexus.distributed.persistence.orm import ArtifactRefRow

    signer = ArtifactContentTicketSigner(
        keys={'current': 'test-secret'},
        current_key_id='current',
        ttl=timedelta(minutes=5),
    )
    ticket = signer.issue_dependency_build(
        dependency_build_id='build-1',
        attempt=1,
        environment_id='env-1',
        operations=['artifact:read'],
        artifact_ids=['artifact-linked', 'artifact-unlinked'],
        now=utcnow(),
    )
    build = SimpleNamespace(
        dependency_build_id='build-1',
        environment_id='env-1',
        owner_principal_id='user-1',
        status='building',
        attempt=1,
    )

    class FakeSession:
        async def get(self, model, identity):
            if model is DependencyBuildRow:
                return build
            if model is ArtifactRefRow and identity['artifact_id'] == 'artifact-linked':
                return SimpleNamespace(**identity)
            return None

    request = SimpleNamespace(
        headers={ARTIFACT_CONTENT_TICKET_HEADER: ticket},
        app={ARTIFACT_CONTENT_TICKET_SIGNER_KEY: signer},
    )
    scope = await _resolve_dependency_build_artifact_scope(
        request,
        FakeSession(),
        operation='artifact:read',
        artifact_id='artifact-linked',
    )
    assert scope.owner_principal_id == 'user-1'

    with pytest.raises(web.HTTPNotFound) as exc_info:
        await _resolve_dependency_build_artifact_scope(
            request,
            FakeSession(),
            operation='artifact:read',
            artifact_id='artifact-unlinked',
        )
    assert exc_info.value.text == 'artifact_not_found'

    stale_ticket = signer.issue_dependency_build(
        dependency_build_id='build-1',
        attempt=2,
        environment_id='env-1',
        operations=['artifact:read'],
        artifact_ids=['artifact-linked'],
        now=utcnow(),
    )
    request.headers = {ARTIFACT_CONTENT_TICKET_HEADER: stale_ticket}
    with pytest.raises(web.HTTPForbidden) as exc_info:
        await _resolve_dependency_build_artifact_scope(
            request,
            FakeSession(),
            operation='artifact:read',
            artifact_id='artifact-linked',
        )
    assert exc_info.value.text == 'artifact_content_scope_mismatch'


@pytest.fixture
async def session_factory():
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    await create_schema(engine)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def test_application_gateway_has_no_global_upload_semaphore(session_factory):
    from nexus.distributed.gateways.application.routes._result_pipeline import RESULT_PIPELINE_KEY

    app = create_application_app(
        session_factory=session_factory,
        config=DistributedConfig(),
    )

    assert not any(isinstance(value, asyncio.Semaphore) for value in app.values())
    assert app[RESULT_PIPELINE_KEY]._cpu_concurrency == 1


async def test_result_download_overlaps_cpu_without_exceeding_cpu_limit(session_factory, tmp_path, monkeypatch):
    import threading

    from contextlib import asynccontextmanager

    from nexus.distributed.artifacts.models import ArtifactRef
    from nexus.distributed.artifacts.run_result import encode_run_result_content
    from nexus.distributed.gateways.application.routes import _result_pipeline, executions

    entered = asyncio.Event()
    downloaded = asyncio.Event()
    release = threading.Event()
    active = 0
    peak = 0
    downloads = 0
    loop = asyncio.get_running_loop()
    path = tmp_path / 'result.json'
    path.write_bytes(encode_run_result_content(status='succeeded', output={'success': True}, error=None))
    ref = ArtifactRef(artifact_id='result', kind='run_result', size_bytes=path.stat().st_size, sha256='0' * 64)
    original = _result_pipeline.ResultEncoder.prepare

    @asynccontextmanager
    async def download_result(*args, **kwargs):
        nonlocal downloads
        downloads += 1
        if downloads == 2:
            downloaded.set()
        yield path

    def encode_result(*args, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        loop.call_soon_threadsafe(entered.set)
        try:
            assert release.wait(5)
            return original(*args, **kwargs)
        finally:
            active -= 1

    async def send_result(request, chunks):
        return web.Response(body=b''.join([chunk async for chunk in chunks]))

    app = create_application_app(session_factory=session_factory, config=DistributedConfig())
    request = SimpleNamespace(app=app)
    monkeypatch.setattr(executions, '_download_result_file', download_result)
    monkeypatch.setattr(_result_pipeline.ResultEncoder, 'prepare', encode_result)
    monkeypatch.setattr(executions, '_send_result_response', send_result)
    first = asyncio.create_task(executions._read_result_response(request, stored_object=None, ref=ref))
    await entered.wait()
    second = asyncio.create_task(executions._read_result_response(request, stored_object=None, ref=ref))
    try:
        await asyncio.wait_for(downloaded.wait(), timeout=2)
        assert active == 1
    finally:
        release.set()
        responses = await asyncio.gather(first, second)
    assert peak == 1
    assert all(json.loads(response.body) == {'success': True} for response in responses)


@pytest.mark.parametrize(
    ('stage', 'interruption'),
    [('decode', 'cancel'), ('transform', 'cancel'), ('decode', 'deadline')],
)
@pytest.mark.parametrize('worker_fails', [False, True])
async def test_result_cancellation_keeps_slot_until_thread_finishes(
    session_factory, tmp_path, monkeypatch, stage, interruption, worker_fails
):
    import threading

    from nexus.distributed.artifacts.models import ArtifactRef
    from nexus.distributed.artifacts.run_result import encode_run_result_content
    from nexus.distributed.gateways.application.routes import _result_pipeline, executions
    from nexus.distributed.gateways.common.keys import ARTIFACT_SPOOL_KEY, ARTIFACT_STORE_KEY

    content = encode_run_result_content(status='succeeded', output={'success': True}, error=None, compression='zstd')
    ref = ArtifactRef(
        artifact_id='result', kind='run_result', size_bytes=len(content), sha256=hashlib.sha256(content).hexdigest()
    )

    class Store:
        async def get(self, stored_object, destination, **kwargs):
            destination.write_bytes(content)

    app = create_application_app(
        session_factory=session_factory,
        config=DistributedConfig(
            artifact_upload_spool_dir=str(tmp_path),
            artifact_download_write_idle_timeout=0.05,
            artifact_download_deadline=timedelta(seconds=0.1 if interruption == 'deadline' else 60),
        ),
    )
    app[ARTIFACT_STORE_KEY] = Store()
    request = SimpleNamespace(app=app)
    owner, name = {
        'decode': (_result_pipeline.ResultEncoder, 'next_chunk'),
        'transform': (_result_pipeline.ResultEncoder, 'prepare'),
    }[stage]
    original = getattr(owner, name)
    entered = asyncio.Event()
    release = threading.Event()
    lock = threading.Lock()
    loop = asyncio.get_running_loop()
    active = peak = calls = 0

    def blocked(*args, **kwargs):
        nonlocal active, peak, calls
        with lock:
            index = calls
            calls += 1
            active += 1
            peak = max(peak, active)
        try:
            if index == 0:
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(5)
                if worker_fails:
                    raise ValueError('worker failed during cancellation')
            return original(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(owner, name, blocked)

    async def send_result(request, chunks):
        return web.Response(body=b''.join([chunk async for chunk in chunks]))

    monkeypatch.setattr(executions, '_send_result_response', send_result)
    spool = app[ARTIFACT_SPOOL_KEY]
    await spool.start()
    first = asyncio.create_task(executions._read_result_response(request, stored_object=None, ref=ref))
    tasks = [first]
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        if interruption == 'cancel':
            for _ in range(2):
                first.cancel()
                await asyncio.sleep(0)
        else:
            await asyncio.sleep(0.15)
        tasks.append(asyncio.create_task(executions._read_result_response(request, stored_object=None, ref=ref)))
        await asyncio.sleep(0)
        assert not first.done()
        assert app[_result_pipeline.RESULT_PIPELINE_KEY].active >= 1
        assert list((tmp_path / 'cache').iterdir())
    finally:
        release.set()
        try:
            outcomes = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5)
        finally:
            await spool.close()

    expected_error = asyncio.CancelledError if interruption == 'cancel' else web.HTTPGatewayTimeout
    assert isinstance(outcomes[0], expected_error)
    assert json.loads(outcomes[1].body) == {'success': True}
    assert calls >= 2
    assert peak == 1
    assert active == 0
    assert spool.reserved_bytes == 0
    assert not list(tmp_path.glob('*.result'))


async def test_application_gateway_purges_orphaned_transfer_spool_before_serving(session_factory, tmp_path):
    spool = tmp_path / 'upload-spool'
    spool.mkdir()
    (spool / 'stale.download').write_bytes(b'partial download')
    stale_directory = spool / 'stale-work'
    stale_directory.mkdir()
    (stale_directory / 'partial').write_bytes(b'partial upload')
    app = create_application_app(
        session_factory=session_factory,
        config=DistributedConfig(
            artifact_upload_spool_dir=str(spool),
            artifact_local_root=str(tmp_path / 'artifacts'),
        ),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        assert list(spool.iterdir()) == [spool / 'cache']
    finally:
        await client.close()


async def test_artifact_download_deadline_cancels_stalled_backend_work() -> None:
    from nexus.distributed.gateways.application.routes.artifact_shared import _artifact_download_deadline

    config = DistributedConfig(
        artifact_download_write_idle_timeout=0.01,
        artifact_download_deadline=timedelta(seconds=0.05),
    )
    with pytest.raises(web.HTTPGatewayTimeout) as exc_info:
        async with _artifact_download_deadline(config):
            await asyncio.sleep(1)

    assert exc_info.value.text == 'artifact_download_deadline_exceeded'


@pytest.mark.parametrize(
    ('metadata_protocol', 'with_selectors', 'expected_status'),
    [
        pytest.param('staged-v2', False, 201, id='staged-v2'),
        pytest.param('header-v2', False, 201, id='header-v2-rolling-compatibility'),
        pytest.param('legacy-v1', False, 201, id='legacy-v1-rolling-compatibility'),
        pytest.param('legacy-v1', True, 400, id='legacy-v1-rejects-selectors'),
    ],
)
async def test_dependency_builder_uploads_validated_wheelhouse_with_build_scope(
    session_factory,
    tmp_path,
    metadata_protocol,
    with_selectors,
    expected_status,
):
    from nexus.core.module_dependencies import ModuleSelector
    from nexus.distributed.artifacts.models import ArtifactRef
    from nexus.distributed.artifacts.tickets import ArtifactContentTicketSigner
    from nexus.distributed.dependency_build.wheelhouse import (
        WheelhouseBuildSpec,
        WheelhouseMetadata,
        write_wheelhouse_archive,
    )
    from nexus.distributed.repositories.dependency_build import DependencyBuildRepository
    from nexus.distributed.utils.identity import canonical_sha256

    now = utcnow()

    def ref(kind: str, marker: str) -> ArtifactRef:
        return ArtifactRef(
            artifact_id=f'art-{kind}',
            kind=kind,
            sha256=marker * 64,
            size_bytes=100,
        )

    spec = WheelhouseBuildSpec(
        owner_principal_id='user-1',
        environment_id='env-1',
        source_package=ref('source_package', 'a'),
        nexus_wheel=ref('nexus_wheel', 'b'),
        uv_lock=ref('uv_lock', 'c'),
        extras=['runtime'],
        module_selectors=(
            [ModuleSelector(kind='object', object_type='openai_clients', value='litellm')] if with_selectors else []
        ),
        workload_image=f'registry.example/workload@sha256:{"d" * 64}',
        builder_image=f'registry.example/builder@sha256:{"e" * 64}',
    )
    async with session_factory() as session:
        session.add(
            EnvironmentRow(
                environment_id='env-1',
                name='env-1',
                allocation_type='dynamic',
                provider_type='kubernetes',
                provider_config_ref='kubernetes/default',
                resource_limits_json={'max_artifact_size_bytes': 1024 * 1024},
                default_timeouts_json={},
                config_version=1,
                status='enabled',
                created_by='admin',
                updated_by='admin',
                created_at=now,
                updated_at=now,
            )
        )
        repository = DependencyBuildRepository(session)
        build, _ = await repository.get_or_create(
            spec,
            provider_type='kubernetes',
            provider_config_ref='kubernetes/default',
            provider_config_snapshot={
                'provider_type': 'kubernetes',
                'namespace': 'default',
                'api_url': 'https://kubernetes.example',
                'token_file': '/var/run/secrets/nexus/provider/kubernetes/token',
                'ca_file': '/var/run/secrets/nexus/provider/kubernetes/ca.crt',
            },
            now=now,
        )
        claim = await repository.claim_next(
            operation_owner='controller',
            lease_ttl=timedelta(minutes=2),
            now=now,
        )
        assert claim is not None
        build = await repository.begin_attempt(
            build.dependency_build_id,
            operation_id=claim.operation_id,
            attempt=1,
            provider_job_name='build-job-1',
            now=now,
        )
        await session.commit()

    wheels = tmp_path / 'wheels'
    wheels.mkdir()
    (wheels / 'nexus_agent-1.5.0-py3-none-any.whl').write_bytes(b'wheel')
    archive = tmp_path / 'wheelhouse.tar.zst'
    nexus_wheel_path = 'wheels/nexus_agent-1.5.0-py3-none-any.whl'
    built_metadata = write_wheelhouse_archive(
        wheels,
        spec,
        archive,
        nexus_wheel_path=nexus_wheel_path,
        module_names=['identity'],
    )
    if metadata_protocol in {'staged-v2', 'header-v2'}:
        expected_metadata = built_metadata
        if metadata_protocol == 'staged-v2':
            metadata_headers = {
                'X-Nexus-Wheelhouse-Metadata-Version': str(expected_metadata.schema_version),
                'X-Nexus-Wheelhouse-Metadata-Sha256': canonical_sha256(expected_metadata.model_dump(mode='json')),
            }
        else:
            metadata_headers = {'X-Nexus-Wheelhouse-Metadata': expected_metadata.model_dump_json()}
    else:
        expected_metadata = WheelhouseMetadata(
            build_key=built_metadata.build_key,
            nexus_wheel_path=nexus_wheel_path,
        )
        metadata_headers = {'X-Nexus-Wheelhouse-Nexus-Wheel-Path': nexus_wheel_path}
    content = archive.read_bytes()
    config = DistributedConfig(
        artifact_local_root=str(tmp_path / 'artifacts'),
        artifact_upload_spool_dir=str(tmp_path / 'spool'),
    )
    signer = ArtifactContentTicketSigner(
        keys=config.artifact_ticket_keyring(),
        current_key_id=config.artifact_upload_ticket_current_key_id,
        ttl=config.artifact_upload_ticket_ttl,
    )
    ticket = signer.issue_dependency_build(
        dependency_build_id=build.dependency_build_id,
        attempt=build.attempt,
        environment_id='env-1',
        operations=['wheelhouse:write'],
        artifact_ids=[],
        now=now,
    )
    client = TestClient(TestServer(create_application_app(session_factory=session_factory, config=config)))
    await client.start_server()
    try:
        missing_metadata = await client.put(
            '/v1/internal/artifacts/wheelhouse',
            data=content,
            headers={
                'X-Nexus-Artifact-Content-Ticket': ticket,
                'X-Nexus-Sha256': hashlib.sha256(content).hexdigest(),
                'Content-Length': str(len(content)),
            },
        )
        assert missing_metadata.status == 400
        assert (await missing_metadata.json())['error'] == 'dependency_build_metadata_invalid'
        if metadata_protocol == 'staged-v2':
            metadata_response = await client.post(
                '/v1/internal/dependency-build/wheelhouse-metadata',
                json=expected_metadata.model_dump(mode='json'),
                headers={'X-Nexus-Artifact-Content-Ticket': ticket},
            )
            assert metadata_response.status == 204, await metadata_response.text()
        response = await client.put(
            '/v1/internal/artifacts/wheelhouse',
            data=content,
            headers={
                'X-Nexus-Artifact-Content-Ticket': ticket,
                'X-Nexus-Sha256': hashlib.sha256(content).hexdigest(),
                'Content-Length': str(len(content)),
                **metadata_headers,
            },
        )
        assert response.status == expected_status, await response.text()
        if expected_status != 201:
            assert (await response.json())['error'] == 'dependency_build_metadata_invalid'
            return
        output_ref = (await response.json())['ref']
    finally:
        await client.close()

    async with session_factory() as session:
        persisted = await session.get(DependencyBuildRow, build.dependency_build_id)
        assert persisted.status == 'building'
        assert persisted.wheelhouse_ref_json == output_ref
        assert persisted.manifest_json == expected_metadata.model_dump(mode='json')


async def test_dependency_builder_reports_attempt_scoped_bounded_failure(session_factory):
    from nexus.distributed.artifacts.models import ArtifactRef
    from nexus.distributed.artifacts.tickets import ArtifactContentTicketSigner
    from nexus.distributed.dependency_build.wheelhouse import WheelhouseBuildSpec
    from nexus.distributed.repositories.dependency_build import DependencyBuildRepository

    now = utcnow()

    def ref(kind: str, marker: str) -> ArtifactRef:
        return ArtifactRef(
            artifact_id=f'art-{kind}',
            kind=kind,
            sha256=marker * 64,
            size_bytes=100,
        )

    spec = WheelhouseBuildSpec(
        owner_principal_id='user-1',
        environment_id='env-1',
        source_package=ref('source_package', 'a'),
        nexus_wheel=ref('nexus_wheel', 'b'),
        uv_lock=ref('uv_lock', 'c'),
        extras=['runtime'],
        workload_image=f'registry.example/workload@sha256:{"d" * 64}',
        builder_image=f'registry.example/builder@sha256:{"e" * 64}',
    )
    async with session_factory() as session:
        session.add(
            EnvironmentRow(
                environment_id='env-1',
                name='env-1',
                allocation_type='dynamic',
                provider_type='kubernetes',
                provider_config_ref='kubernetes/default',
                resource_limits_json={},
                default_timeouts_json={},
                config_version=1,
                status='enabled',
                created_by='admin',
                updated_by='admin',
                created_at=now,
                updated_at=now,
            )
        )
        repository = DependencyBuildRepository(session)
        build, _ = await repository.get_or_create(
            spec,
            provider_type='kubernetes',
            provider_config_ref='kubernetes/default',
            provider_config_snapshot={
                'provider_type': 'kubernetes',
                'namespace': 'default',
                'api_url': 'https://kubernetes.example',
                'token_file': '/var/run/secrets/nexus/provider/kubernetes/token',
                'ca_file': '/var/run/secrets/nexus/provider/kubernetes/ca.crt',
            },
            now=now,
        )
        claim = await repository.claim_next(
            operation_owner='controller',
            lease_ttl=timedelta(minutes=2),
            now=now,
        )
        assert claim is not None
        build = await repository.begin_attempt(
            build.dependency_build_id,
            operation_id=claim.operation_id,
            attempt=1,
            provider_job_name='build-job-1',
            now=now,
        )
        await session.commit()

    config = DistributedConfig()
    signer = ArtifactContentTicketSigner(
        keys=config.artifact_ticket_keyring(),
        current_key_id=config.artifact_upload_ticket_current_key_id,
        ttl=config.artifact_upload_ticket_ttl,
    )
    ticket = signer.issue_dependency_build(
        dependency_build_id=build.dependency_build_id,
        attempt=build.attempt,
        environment_id=build.environment_id,
        operations=['dependency_build:report_failure'],
        artifact_ids=[],
        now=now,
    )
    client = TestClient(TestServer(create_application_app(session_factory=session_factory, config=config)))
    await client.start_server()
    try:
        response = await client.post(
            '/v1/internal/dependency-build/failure',
            json={
                'error_code': 'dependency_offline_install_failed',
                'error_message': 'missing dependency wheel',
            },
            headers={'X-Nexus-Artifact-Content-Ticket': ticket},
        )
        assert response.status == 204, await response.text()
    finally:
        await client.close()

    async with session_factory() as session:
        persisted = await DependencyBuildRepository(session).get(build.dependency_build_id)
    assert persisted.status == 'building'
    assert persisted.error_code == 'dependency_offline_install_failed'
    assert persisted.error_message == 'missing dependency wheel'


async def test_application_gateway_rejects_run_result_above_dedicated_limit(session_factory, tmp_path):
    content = b'12345'
    app = create_application_app(
        session_factory=session_factory,
        config=DistributedConfig(
            secrets={'internal_token': 'internal-secret'},
            artifact_local_root=str(tmp_path / 'result-limit-artifacts'),
            artifact_upload_spool_dir=str(tmp_path / 'result-limit-spool'),
            artifact_run_result_max_size_bytes=4,
        ),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.put(
            '/v1/internal/artifacts/run-result',
            data=content,
            headers={
                'X-Nexus-Sha256': hashlib.sha256(content).hexdigest(),
                'X-Nexus-Internal-Token': 'internal-secret',
            },
        )
        assert response.status == 413
    finally:
        await client.close()


async def test_local_dev_application_cors_echoes_origin_with_credentials(session_factory):
    app = create_application_app(
        session_factory=session_factory,
        config=DistributedConfig(auth_mode='disabled'),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        # Cross-origin request from the control-gateway-hosted console with credentials: 'include'.
        cross_origin = await client.options(
            '/v1/executions/example/result',
            headers={'Origin': 'http://21.160.200.50'},
        )
        # Non-browser clients (curl, server-to-server) omit Origin and must still get a permissive
        # wildcard without a credentials header.
        no_origin = await client.get('/v1/health')
    finally:
        await client.close()

    assert cross_origin.status == 204
    assert cross_origin.headers['Access-Control-Allow-Origin'] == 'http://21.160.200.50'
    assert cross_origin.headers['Access-Control-Allow-Credentials'] == 'true'
    assert no_origin.headers['Access-Control-Allow-Origin'] == '*'
    assert 'Access-Control-Allow-Credentials' not in no_origin.headers


async def test_authenticated_application_cors_allows_credentials_only_for_configured_origin(session_factory):
    app = create_application_app(
        session_factory=session_factory,
        config=DistributedConfig(
            auth_mode='required',
            cors_origins=['https://console.example'],
        ),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        allowed = await client.options(
            '/v1/executions/example/result',
            headers={'Origin': 'https://console.example'},
        )
        denied = await client.options(
            '/v1/executions/example/result',
            headers={'Origin': 'https://untrusted.example'},
        )
    finally:
        await client.close()

    assert allowed.headers['Access-Control-Allow-Origin'] == 'https://console.example'
    assert allowed.headers['Access-Control-Allow-Credentials'] == 'true'
    assert 'Access-Control-Allow-Origin' not in denied.headers
    assert 'Access-Control-Allow-Credentials' not in denied.headers


async def _prepare_live_execution(session_factory, endpoint_url: str):
    async with session_factory() as session:
        manager = ManagerRepository(session)
        await manager.register_session(
            manager_id='im-live',
            session_id='session-1',
            endpoint_url='http://manager.internal',
            supported_modes=[ExecutionMode.LIVE],
            supported_spec_types=['nexus_live'],
            supported_instance_modes=['asyncio'],
            capacity_total=1,
            force_replace=False,
            now=utcnow(),
        )
        execution_repo = ExecutionRepository(session, shard_count=8)
        execution = await execution_repo.create_execution(
            CreateExecution(
                execution_id='live-1',
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
        await manager.ack_assignment(
            assignment.assignment_id, assignment.manager_id, assignment.manager_session_id, now=utcnow()
        )
        await manager.mark_running(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            endpoint_url=endpoint_url,
            endpoint_ready=True,
            runtime_info={},
            now=utcnow(),
        )
        await manager.mark_waiting_input(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            payload={'prompt': 'choose'},
            now=utcnow(),
        )
        await session.commit()
        return assignment


async def _bind_execution_artifact_scope(session_factory, execution_id: str, assignment_id: str) -> None:
    async with session_factory() as session:
        execution = await session.get(ExecutionRow, execution_id)
        assignment = await session.get(AssignmentRow, assignment_id)
        assert execution is not None
        assert assignment is not None
        execution.owner_principal_id = 'local-dev'
        execution.environment_id = 'env-1'
        execution.allocation_id = 'alloc-1'
        assignment.owner_principal_id = 'local-dev'
        assignment.environment_id = 'env-1'
        assignment.allocation_id = 'alloc-1'
        now = utcnow()
        session.add(
            EnvironmentAllocationRow(
                allocation_id='alloc-1',
                top_level_execution_id=execution_id,
                owner_principal_id='local-dev',
                environment_id='env-1',
                pool_id='default',
                workload='execution',
                status='ready',
                desired_manager_count=1,
                ready_manager_count=1,
                manager_count=1,
                cpu_millicores=1000,
                memory_bytes=1024**3,
                slots=1,
                active_batches=0,
                allocation_spec_json={},
                allocation_spec_hash='a' * 64,
                config_version=1,
                status_version=1,
                last_progress_at=now,
                resources_reserved_at=now,
                created_at=now,
                ready_at=now,
                updated_at=now,
            )
        )
        await session.commit()


@pytest.mark.parametrize('compression', ['none', 'zstd'])
@pytest.mark.parametrize('without_result_pointer', [False, True])
async def test_internal_run_result_upload_derives_scope_and_links_execution(
    session_factory, tmp_path, monkeypatch, compression, without_result_pointer
):
    from nexus.distributed.artifacts.run_result import encode_run_result_content
    from nexus.distributed.artifacts.store import ArtifactUploadCoordinator

    assignment = await _prepare_live_execution(session_factory, 'http://manager.internal')
    await _bind_execution_artifact_scope(session_factory, assignment.execution_id, assignment.assignment_id)
    async with session_factory() as session:
        allocation = await session.get(EnvironmentAllocationRow, 'alloc-1')
        assert allocation is not None
        allocation.top_level_execution_id = 'batch-parent'
        await session.commit()
    content = encode_run_result_content(
        status='succeeded', output={'success': True, 'value': 7}, error=None, compression=compression
    )
    async with session_factory() as session:
        now = utcnow()
        session.add(
            EnvironmentRow(
                environment_id='env-1',
                name='env-1',
                allocation_type='predefined',
                provider_type='predefined',
                provider_config_ref=None,
                resource_limits_json={'max_artifact_size_bytes': len(content) + 1},
                default_timeouts_json={},
                config_version=1,
                status='enabled',
                created_by='admin',
                updated_by='admin',
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()
    config = DistributedConfig(
        secrets={'internal_token': 'internal-secret'},
        artifact_local_root=str(tmp_path / 'artifacts'),
        artifact_upload_spool_dir=str(tmp_path / 'spool'),
        artifact_download_total_concurrency=1,
        artifact_download_input_concurrency=1,
        artifact_download_result_concurrency=1,
        artifact_download_max_waiters=1,
    )
    bound_objects = []
    original_put_reserved = ArtifactUploadCoordinator.put_reserved

    async def record_bound_object(self, *args, **kwargs):
        bound_objects.append(kwargs.get('stored_object'))
        return await original_put_reserved(self, *args, **kwargs)

    monkeypatch.setattr(ArtifactUploadCoordinator, 'put_reserved', record_bound_object)
    app = create_application_app(session_factory=session_factory, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    from nexus.distributed.artifacts.tickets import ArtifactContentTicketSigner

    run_result_ticket = ArtifactContentTicketSigner(
        keys=config.artifact_ticket_keyring(),
        current_key_id=config.artifact_upload_ticket_current_key_id,
        ttl=config.artifact_upload_ticket_ttl,
    ).issue(
        execution_id='batch-parent',
        top_level_execution_id='batch-parent',
        environment_id='env-1',
        allocation_id='alloc-1',
        operations=['run_result:write'],
        artifact_ids=[],
        now=utcnow(),
    )
    headers = {
        'X-Nexus-Artifact-Content-Ticket': run_result_ticket,
        'X-Nexus-Execution-Id': assignment.execution_id,
        'X-Nexus-Assignment-Id': assignment.assignment_id,
        'X-Nexus-Attempt-Id': assignment.attempt_id,
        'X-Nexus-Sha256': hashlib.sha256(content).hexdigest(),
        'X-Nexus-Owner': 'forged-owner',
        'X-Nexus-Producer-Key': 'forged-producer',
        'Content-Length': str(len(content)),
    }
    try:
        response = await client.put('/v1/internal/artifacts/run-result', data=content, headers=headers)
        assert response.status == 201, await response.text()
        ref = (await response.json())['ref']

        repeated = await client.put('/v1/internal/artifacts/run-result', data=content, headers=headers)
        assert repeated.status == 200, await repeated.text()
        assert (await repeated.json())['ref'] == ref
        assert len(bound_objects) == 1
        assert bound_objects[0] is not None
        assert bound_objects[0].uri.startswith('file://')

        from nexus.distributed.artifacts.gateway import (
            ApplicationGatewayArtifactTransport,
            ArtifactGatewayResponseError,
        )

        content_ticket = ArtifactContentTicketSigner(
            keys=config.artifact_ticket_keyring(),
            current_key_id=config.artifact_upload_ticket_current_key_id,
            ttl=config.artifact_upload_ticket_ttl,
        ).issue(
            execution_id='batch-parent',
            top_level_execution_id='batch-parent',
            environment_id='env-1',
            allocation_id='alloc-1',
            operations=['artifact:read'],
            artifact_ids=[],
            now=utcnow(),
        )
        content_ticket_file = tmp_path / 'artifact-read-token'
        content_ticket_file.write_text(content_ticket)
        from nexus.distributed.gateways.application.app import ARTIFACT_STORE_KEY

        store = app[ARTIFACT_STORE_KEY]
        original_get = store.get
        get_calls = 0
        get_started = asyncio.Event()
        release_get = asyncio.Event()

        async def blocking_get(*args, **kwargs):
            nonlocal get_calls
            get_calls += 1
            get_started.set()
            await release_get.wait()
            return await original_get(*args, **kwargs)

        monkeypatch.setattr(store, 'get', blocking_get)
        download_headers = {
            'X-Nexus-Artifact-Content-Ticket': content_ticket,
            'X-Nexus-Execution-Id': assignment.execution_id,
            'X-Nexus-Assignment-Id': assignment.assignment_id,
            'X-Nexus-Attempt-Id': assignment.attempt_id,
            'X-Nexus-Artifact-Kind': ref['kind'],
            'X-Nexus-Sha256': ref['sha256'],
            'X-Nexus-Artifact-Size': str(ref['size_bytes']),
        }
        first_download = asyncio.create_task(
            client.get(f'/v1/internal/artifacts/{ref["artifact_id"]}/content', headers=download_headers)
        )
        second_download = None
        try:
            await asyncio.wait_for(get_started.wait(), timeout=1)
            second_download = asyncio.create_task(
                client.get(f'/v1/internal/artifacts/{ref["artifact_id"]}/content', headers=download_headers)
            )
            await asyncio.sleep(0.05)
            overloaded = await client.get(
                f'/v1/internal/artifacts/{ref["artifact_id"]}/content',
                headers=download_headers,
            )
            assert overloaded.status == 503
            assert await overloaded.json() == {
                'error': 'artifact_download_busy:queue_full',
                'retryable': True,
            }
            assert overloaded.headers['Retry-After'] == '1'
            release_get.set()
            first_response = await asyncio.wait_for(first_download, timeout=1)
            second_response = await asyncio.wait_for(second_download, timeout=1)
            assert await first_response.read() == content
            assert await second_response.read() == content
        finally:
            release_get.set()
            for task in (first_download, second_download):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (first_download, second_download) if task is not None),
                return_exceptions=True,
            )
        artifact_transport = ApplicationGatewayArtifactTransport(
            base_url=str(client.make_url('/')).rstrip('/'),
            content_ticket_file=content_ticket_file,
        )
        downloaded = await artifact_transport.download_bytes(
            execution_id=assignment.execution_id,
            assignment_id=assignment.assignment_id,
            attempt_id=assignment.attempt_id,
            artifact_ref=ref,
        )
        assert downloaded == content
        assert get_calls == 3
        content_ticket_file.write_text('rotated-invalid-ticket')
        with pytest.raises(ArtifactGatewayResponseError) as caught:
            await artifact_transport.download_bytes(
                execution_id=assignment.execution_id,
                assignment_id=assignment.assignment_id,
                attempt_id=assignment.attempt_id,
                artifact_ref=ref,
            )
        assert caught.value.status == 401
        assert caught.value.detail == 'invalid_artifact_content_ticket'
        content_ticket_file.write_text(content_ticket)
        await artifact_transport.close()

        async with session_factory() as session:
            execution = await session.get(ExecutionRow, assignment.execution_id)
            assignment_row = await session.get(AssignmentRow, assignment.assignment_id)
            manager = await session.get(InstanceManagerRow, assignment.manager_id)
            liveness = await session.get(InstanceManagerLivenessRow, assignment.manager_id)
            assert execution is not None
            assert assignment_row is not None
            assert manager is not None
            assert liveness is not None
            execution.status = ExecutionStatus.SUCCEEDED
            execution.route_status = RouteStatus.CLOSED
            execution.result_json = {
                '_external': True,
                'kind': 'nexus_live_result',
                'version': 2,
                'metadata': {},
            }
            if without_result_pointer:
                execution.result_json = None
            assignment_row.runtime_info_json = {
                'run_trace': {
                    'live': {
                        'version': 3,
                        'run_kind': 'live',
                        'sinks': [],
                    }
                }
            }
            stale_at = utcnow() - timedelta(minutes=1)
            manager.last_heartbeat_at = stale_at
            liveness.last_heartbeat_at = stale_at
            await session.commit()

        result_response = await client.get(f'/v1/executions/{assignment.execution_id}/live/result')
        assert result_response.status == 200, await result_response.text()
        assert await result_response.json() == {'success': True, 'value': 7}
        standard_result = await client.get(f'/v1/executions/{assignment.execution_id}/result')
        assert standard_result.status == 200
        assert await standard_result.json() == {'success': True, 'value': 7}

        async with session_factory() as session:
            execution = await session.get(ExecutionRow, assignment.execution_id)
            execution.current_attempt_id = 'different-attempt'
            await session.commit()
        stale_result = await client.get(f'/v1/executions/{assignment.execution_id}/result')
        assert stale_result.status == 404
        async with session_factory() as session:
            execution = await session.get(ExecutionRow, assignment.execution_id)
            execution.current_attempt_id = assignment.attempt_id
            await session.commit()

        events_response = await client.get(f'/v1/executions/{assignment.execution_id}/live/events?cursor=0')
        assert events_response.status == 410
        assert (await events_response.json())['error'] == 'trace_events_not_retained'

        oversized_content = content + b'\n\n'
        oversized = await client.put(
            '/v1/internal/artifacts/run-result',
            data=oversized_content,
            headers={
                **headers,
                'X-Nexus-Sha256': hashlib.sha256(oversized_content).hexdigest(),
                'Content-Length': str(len(oversized_content)),
            },
        )
        assert oversized.status == 413

        conflicting_content = content + b'\n'
        conflicting_headers = {
            **headers,
            'X-Nexus-Sha256': hashlib.sha256(conflicting_content).hexdigest(),
            'Content-Length': str(len(conflicting_content)),
        }
        conflict = await client.put(
            '/v1/internal/artifacts/run-result',
            data=conflicting_content,
            headers=conflicting_headers,
        )
        assert conflict.status == 409
        assert (await conflict.json())['error'] == 'artifact_producer_conflict'
    finally:
        await client.close()

    async with session_factory() as session:
        execution = await session.get(ExecutionRow, assignment.execution_id)
        artifact = await session.get(ArtifactRow, ref['artifact_id'])
        assert execution is not None
        assert artifact is not None
        assert execution.result_artifact_id == artifact.artifact_id
        assert artifact.owner_principal_id == 'local-dev'
        assert artifact.environment_id == 'env-1'
        assert artifact.kind == 'run_result'
        assert artifact.producer_key == f'run-result:{assignment.execution_id}:{assignment.attempt_id}'


async def test_internal_result_upload_waits_for_lane_before_opening_database_session(
    session_factory,
    tmp_path,
    monkeypatch,
):
    from nexus.distributed.artifacts.tickets import ArtifactContentTicketSigner
    from nexus.distributed.gateways.application.app import ARTIFACT_STORE_KEY

    assignment = await _prepare_live_execution(session_factory, 'http://manager.internal')
    await _bind_execution_artifact_scope(session_factory, assignment.execution_id, assignment.assignment_id)
    content = json.dumps(
        {'version': 1, 'status': 'succeeded', 'output': {'success': True}, 'error': None},
        sort_keys=True,
        separators=(',', ':'),
    ).encode()
    async with session_factory() as session:
        now = utcnow()
        session.add(
            EnvironmentRow(
                environment_id='env-1',
                name='env-1',
                allocation_type='predefined',
                provider_type='predefined',
                provider_config_ref=None,
                resource_limits_json={'max_artifact_size_bytes': len(content) + 1},
                default_timeouts_json={},
                config_version=1,
                status='enabled',
                created_by='admin',
                updated_by='admin',
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()

    session_calls = 0

    def counting_session_factory():
        nonlocal session_calls
        session_calls += 1
        return session_factory()

    config = DistributedConfig(
        secrets={'internal_token': 'internal-secret'},
        artifact_local_root=str(tmp_path / 'artifacts'),
        artifact_upload_spool_dir=str(tmp_path / 'spool'),
        artifact_upload_total_concurrency=1,
        artifact_upload_input_concurrency=1,
        artifact_upload_result_concurrency=1,
        artifact_upload_max_waiters=1,
    )
    app = create_application_app(session_factory=counting_session_factory, config=config)
    store = app[ARTIFACT_STORE_KEY]
    original_put = store.put
    put_started = asyncio.Event()
    release_put = asyncio.Event()

    async def blocking_put(*args, **kwargs):
        put_started.set()
        await release_put.wait()
        return await original_put(*args, **kwargs)

    monkeypatch.setattr(store, 'put', blocking_put)
    signer = ArtifactContentTicketSigner(
        keys=config.artifact_ticket_keyring(),
        current_key_id=config.artifact_upload_ticket_current_key_id,
        ttl=config.artifact_upload_ticket_ttl,
    )
    ticket = signer.issue(
        execution_id=assignment.execution_id,
        environment_id='env-1',
        allocation_id='alloc-1',
        assignment_id=assignment.assignment_id,
        attempt_id=assignment.attempt_id,
        operations=['run_result:write'],
        artifact_ids=[],
        now=utcnow(),
    )
    headers = {
        'X-Nexus-Artifact-Content-Ticket': ticket,
        'X-Nexus-Execution-Id': assignment.execution_id,
        'X-Nexus-Assignment-Id': assignment.assignment_id,
        'X-Nexus-Attempt-Id': assignment.attempt_id,
        'X-Nexus-Sha256': hashlib.sha256(content).hexdigest(),
        'Content-Length': str(len(content)),
    }
    client = TestClient(TestServer(app))
    await client.start_server()
    first = asyncio.create_task(client.put('/v1/internal/artifacts/run-result', data=content, headers=headers))
    second = None
    third = None
    try:
        await asyncio.wait_for(put_started.wait(), timeout=1)
        sessions_before_second = session_calls
        second = asyncio.create_task(client.put('/v1/internal/artifacts/run-result', data=content, headers=headers))
        await asyncio.sleep(0.05)

        assert session_calls == sessions_before_second
        assert not second.done()
        third = asyncio.create_task(client.put('/v1/internal/artifacts/run-result', data=content, headers=headers))
        third_response = await asyncio.wait_for(third, timeout=1)
        assert third_response.status == 503
        assert await third_response.json() == {
            'error': 'artifact_upload_busy:queue_full',
            'retryable': True,
        }
        assert third_response.headers['Retry-After'] == '1'

        release_put.set()
        first_response = await asyncio.wait_for(first, timeout=1)
        second_response = await asyncio.wait_for(second, timeout=1)
        assert first_response.status == 201, await first_response.text()
        assert second_response.status == 200, await second_response.text()
    finally:
        release_put.set()
        for task in (first, second, third):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (first, second, third) if task is not None), return_exceptions=True)
        await client.close()


async def test_internal_result_upload_renews_receiving_lease_during_slow_store(
    session_factory,
    tmp_path,
    monkeypatch,
):
    from nexus.distributed.artifacts.tickets import ArtifactContentTicketSigner
    from nexus.distributed.gateways.application.app import ARTIFACT_STORE_KEY

    assignment = await _prepare_live_execution(session_factory, 'http://manager.internal')
    await _bind_execution_artifact_scope(session_factory, assignment.execution_id, assignment.assignment_id)
    content = json.dumps(
        {'version': 1, 'status': 'succeeded', 'output': {'success': True}, 'error': None},
        sort_keys=True,
        separators=(',', ':'),
    ).encode()
    async with session_factory() as session:
        now = utcnow()
        session.add(
            EnvironmentRow(
                environment_id='env-1',
                name='env-1',
                allocation_type='predefined',
                provider_type='predefined',
                provider_config_ref=None,
                resource_limits_json={'max_artifact_size_bytes': len(content) + 1},
                default_timeouts_json={},
                config_version=1,
                status='enabled',
                created_by='admin',
                updated_by='admin',
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()

    config = DistributedConfig(
        secrets={'internal_token': 'internal-secret'},
        artifact_local_root=str(tmp_path / 'lease-artifacts'),
        artifact_upload_spool_dir=str(tmp_path / 'lease-spool'),
        artifact_upload_lease_ttl=timedelta(milliseconds=150),
        artifact_upload_lease_renew_interval=timedelta(milliseconds=20),
    )
    app = create_application_app(session_factory=session_factory, config=config)
    store = app[ARTIFACT_STORE_KEY]
    original_put = store.put
    put_started = asyncio.Event()
    release_put = asyncio.Event()

    async def blocking_put(*args, **kwargs):
        put_started.set()
        await release_put.wait()
        return await original_put(*args, **kwargs)

    monkeypatch.setattr(store, 'put', blocking_put)
    ticket = ArtifactContentTicketSigner(
        keys=config.artifact_ticket_keyring(),
        current_key_id=config.artifact_upload_ticket_current_key_id,
        ttl=config.artifact_upload_ticket_ttl,
    ).issue(
        execution_id=assignment.execution_id,
        environment_id='env-1',
        allocation_id='alloc-1',
        assignment_id=assignment.assignment_id,
        attempt_id=assignment.attempt_id,
        operations=['run_result:write'],
        artifact_ids=[],
        now=utcnow(),
    )
    headers = {
        'X-Nexus-Artifact-Content-Ticket': ticket,
        'X-Nexus-Execution-Id': assignment.execution_id,
        'X-Nexus-Assignment-Id': assignment.assignment_id,
        'X-Nexus-Attempt-Id': assignment.attempt_id,
        'X-Nexus-Sha256': hashlib.sha256(content).hexdigest(),
        'Content-Length': str(len(content)),
    }
    client = TestClient(TestServer(app))
    await client.start_server()
    request_task = asyncio.create_task(client.put('/v1/internal/artifacts/run-result', data=content, headers=headers))
    try:
        await asyncio.wait_for(put_started.wait(), timeout=1)
        await asyncio.sleep(0.25)
        async with session_factory() as session:
            active_upload = await session.scalar(
                select(ArtifactUploadRow).where(
                    ArtifactUploadRow.expected_sha256 == hashlib.sha256(content).hexdigest()
                )
            )
            assert active_upload is not None
            assert active_upload.status == 'receiving'
            assert active_upload.operation_expires_at is not None
            assert active_upload.operation_expires_at > utcnow()
        release_put.set()
        response = await asyncio.wait_for(request_task, timeout=1)
        assert response.status == 201, await response.text()
    finally:
        release_put.set()
        if not request_task.done():
            request_task.cancel()
        await asyncio.gather(request_task, return_exceptions=True)
        await client.close()


async def test_internal_artifact_upload_retries_concurrent_producer_insert(
    session_factory,
    tmp_path,
    monkeypatch,
):
    from sqlalchemy.exc import IntegrityError

    from nexus.distributed.artifacts.store import LocalArtifactStore
    from nexus.distributed.artifacts.tickets import ArtifactContentTicketSigner
    from nexus.distributed.gateways.application.app import ARTIFACT_STORE_KEY
    from nexus.distributed.gateways.application.routes.artifact_shared import _internal_artifact_idempotency_scope
    from nexus.distributed.repositories.artifact import ArtifactRepository, CreateArtifactUpload

    assignment = await _prepare_live_execution(session_factory, 'http://manager.internal')
    await _bind_execution_artifact_scope(session_factory, assignment.execution_id, assignment.assignment_id)
    content = json.dumps(
        {'version': 1, 'status': 'succeeded', 'output': {'value': 7}, 'error': None},
        sort_keys=True,
        separators=(',', ':'),
    ).encode()
    content_sha256 = hashlib.sha256(content).hexdigest()
    config = DistributedConfig(
        secrets={'internal_token': 'internal-secret'},
        artifact_local_root=str(tmp_path / 'artifacts'),
        artifact_upload_spool_dir=str(tmp_path / 'spool'),
    )
    app = create_application_app(session_factory=session_factory, config=config)
    assert isinstance(app[ARTIFACT_STORE_KEY], LocalArtifactStore)
    producer_key = f'run-result:{assignment.execution_id}:{assignment.attempt_id}'
    idempotency_scope = _internal_artifact_idempotency_scope(
        'env-1',
        assignment.execution_id,
        assignment.attempt_id,
        'run_result',
    )
    idempotency_key = hashlib.sha256(producer_key.encode()).hexdigest()
    async with session_factory() as session:
        await ArtifactRepository(session).create_upload(
            CreateArtifactUpload(
                owner_principal_id='local-dev',
                environment_id='env-1',
                owner_tenant=None,
                created_by='system:artifact-content',
                kind='run_result',
                sha256=content_sha256,
                size_bytes=len(content),
                idempotency_scope=idempotency_scope,
                idempotency_key=idempotency_key,
                session_ttl=config.artifact_upload_session_ttl,
                artifact_ttl=config.artifact_default_ttl,
                producer_key=producer_key,
            ),
            store=app[ARTIFACT_STORE_KEY],
            now=utcnow(),
        )
        await session.commit()

    original_create_upload = ArtifactRepository.create_upload
    calls = 0

    async def raise_unique_conflict_once(self, request, *, store, now):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise IntegrityError(
                'INSERT INTO distributed_artifacts',
                {},
                RuntimeError(
                    'UNIQUE constraint failed: distributed_artifacts.owner_principal_id, '
                    'distributed_artifacts.kind, distributed_artifacts.producer_key'
                ),
            )
        return await original_create_upload(self, request, store=store, now=now)

    monkeypatch.setattr(ArtifactRepository, 'create_upload', raise_unique_conflict_once)
    ticket = ArtifactContentTicketSigner(
        keys=config.artifact_ticket_keyring(),
        current_key_id=config.artifact_upload_ticket_current_key_id,
        ttl=config.artifact_upload_ticket_ttl,
    ).issue(
        execution_id=assignment.execution_id,
        environment_id='env-1',
        allocation_id='alloc-1',
        assignment_id=assignment.assignment_id,
        attempt_id=assignment.attempt_id,
        operations=['run_result:write'],
        artifact_ids=[],
        now=utcnow(),
    )
    headers = {
        'X-Nexus-Artifact-Content-Ticket': ticket,
        'X-Nexus-Execution-Id': assignment.execution_id,
        'X-Nexus-Assignment-Id': assignment.assignment_id,
        'X-Nexus-Attempt-Id': assignment.attempt_id,
        'X-Nexus-Sha256': content_sha256,
        'Content-Length': str(len(content)),
    }
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.put('/v1/internal/artifacts/run-result', data=content, headers=headers)
        assert response.status == 201, await response.text()
    finally:
        await client.close()

    assert calls == 2


async def test_batch_control_client_publishes_task_list_with_approved_producer_key(
    session_factory,
    tmp_path,
    monkeypatch,
):
    from nexus.distributed.artifacts.gateway import (
        ApplicationGatewayArtifactTransport,
        ArtifactGatewayResponseError,
        TaskListArtifactGateway,
    )
    from nexus.distributed.artifacts.tickets import ArtifactContentTicketSigner

    assignment = await _prepare_live_execution(session_factory, 'http://manager.internal')
    await _bind_execution_artifact_scope(session_factory, assignment.execution_id, assignment.assignment_id)
    content = b'{"version":1,"item_id":"item-00000001","run_input":{}}\n'
    async with session_factory() as session:
        execution = await session.get(ExecutionRow, assignment.execution_id)
        assert execution is not None
        execution.spec_type = 'nexus_batch'
        now = utcnow()
        session.add(
            BatchRunRow(
                parent_execution_id=execution.execution_id,
                shard_id=execution.shard_id,
                generation_mode='remote_materialize',
                generation_status='publishing',
                generation_epoch=3,
                materialized_count=0,
                task_list_manifest_json={
                    'version': 1,
                    'item_count': 1,
                    'size_bytes': len(content),
                    'sha256': hashlib.sha256(content).hexdigest(),
                    'source': {'mode': 'remote_materialize'},
                    'datasets': [],
                },
                task_list_sha256=hashlib.sha256(content).hexdigest(),
                total_items=1,
                owner_principal_id='local-dev',
                submitted_items=0,
                submission_closed=False,
                summary_version=0,
                last_progress_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()

    task_list_path = tmp_path / 'tasks.jsonl'
    task_list_path.write_bytes(content)
    config = DistributedConfig(
        secrets={'internal_token': 'internal-secret'},
        artifact_local_root=str(tmp_path / 'artifacts'),
        artifact_upload_spool_dir=str(tmp_path / 'spool'),
    )
    app = create_application_app(session_factory=session_factory, config=config)
    from nexus.distributed.gateways.application.app import ARTIFACT_STORE_KEY

    store = app[ARTIFACT_STORE_KEY]
    original_get = store.get
    get_calls = 0
    get_started = asyncio.Event()
    release_get = asyncio.Event()

    async def counted_get(*args, **kwargs):
        nonlocal get_calls
        get_calls += 1
        get_started.set()
        await release_get.wait()
        return await original_get(*args, **kwargs)

    monkeypatch.setattr(store, 'get', counted_get)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        signer = ArtifactContentTicketSigner(
            keys=config.artifact_ticket_keyring(),
            current_key_id=config.artifact_upload_ticket_current_key_id,
            ttl=config.artifact_upload_ticket_ttl,
        )
        content_ticket_file = tmp_path / 'batch-artifact-token'
        content_ticket_file.write_text(
            signer.issue(
                execution_id=assignment.execution_id,
                environment_id='env-1',
                allocation_id='alloc-1',
                assignment_id=assignment.assignment_id,
                attempt_id=assignment.attempt_id,
                operations=['task_list:write'],
                artifact_ids=[],
                now=utcnow(),
            )
        )
        artifact_transport = ApplicationGatewayArtifactTransport(
            base_url=str(client.make_url('/')).rstrip('/'),
            content_ticket_file=content_ticket_file,
        )
        artifact_client = TaskListArtifactGateway(artifact_transport)
        ref = await artifact_client.publish_task_list(
            parent_execution_id=assignment.execution_id,
            assignment_id=assignment.assignment_id,
            attempt_id=assignment.attempt_id,
            path=task_list_path,
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
        )
        assert ref['kind'] == 'task_list'
        read_ticket = signer.issue(
            execution_id=assignment.execution_id,
            environment_id='env-1',
            allocation_id='alloc-1',
            assignment_id=assignment.assignment_id,
            attempt_id=assignment.attempt_id,
            operations=['artifact:read'],
            artifact_ids=[ref['artifact_id']],
            now=utcnow(),
        )
        content_ticket_file.write_text(read_ticket)
        downloaded_path = tmp_path / 'downloaded-tasks.jsonl'
        concurrent_path = tmp_path / 'downloaded-tasks-concurrent.jsonl'
        first_download = asyncio.create_task(
            artifact_client.download_artifact_to_path(
                execution_id=assignment.execution_id,
                assignment_id=assignment.assignment_id,
                attempt_id=assignment.attempt_id,
                artifact_ref=ref,
                destination=downloaded_path,
            )
        )
        second_download = None
        try:
            await asyncio.wait_for(get_started.wait(), timeout=1)
            second_download = asyncio.create_task(
                artifact_client.download_artifact_to_path(
                    execution_id=assignment.execution_id,
                    assignment_id=assignment.assignment_id,
                    attempt_id=assignment.attempt_id,
                    artifact_ref=ref,
                    destination=concurrent_path,
                )
            )
            await asyncio.sleep(0)
            release_get.set()
            await asyncio.gather(first_download, second_download)
        finally:
            release_get.set()
            for task in (first_download, second_download):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (first_download, second_download) if task is not None),
                return_exceptions=True,
            )
        assert downloaded_path.read_bytes() == content
        assert concurrent_path.read_bytes() == content
        assert get_calls == 1
        content_ticket_file.write_text('invalid-ticket')
        with pytest.raises(ArtifactGatewayResponseError) as caught:
            await artifact_client.download_artifact_to_path(
                execution_id=assignment.execution_id,
                assignment_id=assignment.assignment_id,
                attempt_id=assignment.attempt_id,
                artifact_ref=ref,
                destination=tmp_path / 'unauthorized-tasks.jsonl',
            )
        assert caught.value.status == 401
        assert get_calls == 1
        content_ticket_file.write_text(read_ticket)
        second_downloaded_path = tmp_path / 'downloaded-tasks-again.jsonl'
        await artifact_client.download_artifact_to_path(
            execution_id=assignment.execution_id,
            assignment_id=assignment.assignment_id,
            attempt_id=assignment.attempt_id,
            artifact_ref=ref,
            destination=second_downloaded_path,
        )
        assert second_downloaded_path.read_bytes() == content
        assert get_calls == 1
        await artifact_transport.close()
    finally:
        await client.close()

    async with session_factory() as session:
        artifact = await session.get(ArtifactRow, ref['artifact_id'])
        batch = await session.get(BatchRunRow, assignment.execution_id)
        assert artifact is not None
        assert artifact.producer_key == f'batch-task-list:{assignment.execution_id}:3'
        assert batch is not None
        assert batch.task_list_ref_json == ref
        assert batch.generation_status == 'materializing'


@pytest.mark.parametrize('sort_keys', [False, True])
async def test_result_api_preserves_stored_business_key_order(session_factory, tmp_path, sort_keys):
    from nexus.distributed.artifacts.gateway import ApplicationGatewayArtifactTransport, RunResultArtifactGateway
    from nexus.distributed.artifacts.run_result import encode_run_result_content, run_result_producer_key

    assignment = await _prepare_live_execution(session_factory, 'http://manager.internal')
    await _bind_execution_artifact_scope(session_factory, assignment.execution_id, assignment.assignment_id)
    config = DistributedConfig(
        secrets={'internal_token': 'internal-secret'},
        artifact_local_root=str(tmp_path / 'artifacts'),
        artifact_upload_spool_dir=str(tmp_path / 'spool'),
    )
    output = {
        'success': True,
        'agent_output': {'messages': [{'role': 'assistant', 'content': 'first'}, {'z': 2, 'a': 1}]},
        'task_output': {'z': 1, 'a': {'y': 2, 'b': 3}},
    }
    content = encode_run_result_content(status='succeeded', output=output, error=None, sort_keys=sort_keys)
    client = TestClient(TestServer(create_application_app(session_factory=session_factory, config=config)))
    await client.start_server()
    transport = ApplicationGatewayArtifactTransport(
        base_url=str(client.make_url('/')).rstrip('/'), internal_token=config.read_internal_token()
    )
    try:
        await RunResultArtifactGateway(transport).publish_run_result(
            execution_id=assignment.execution_id,
            assignment_id=assignment.assignment_id,
            attempt_id=assignment.attempt_id,
            producer_key=run_result_producer_key(assignment.execution_id, assignment.attempt_id),
            content=content,
        )
        async with session_factory() as session:
            execution = await session.get(ExecutionRow, assignment.execution_id)
            assert execution is not None
            execution.status = ExecutionStatus.SUCCEEDED
            execution.result_json = {'_external': True, 'kind': 'nexus_live_result', 'version': 2, 'metadata': {}}
            await session.commit()
        expected = json.loads(json.dumps(output, sort_keys=sort_keys))
        for endpoint in ('result', 'live/result'):
            response = await client.get(f'/v1/executions/{assignment.execution_id}/{endpoint}')
            assert response.status == 200, await response.text()
            assert json.dumps(await response.json()) == json.dumps(expected)
    finally:
        await transport.close()
        await client.close()


async def test_reconciler_can_publish_batch_parent_run_result_with_internal_token(session_factory, tmp_path):
    from nexus.distributed.artifacts.gateway import ApplicationGatewayArtifactTransport, RunResultArtifactGateway
    from nexus.distributed.artifacts.run_result import (
        encode_run_result_content,
        run_result_producer_key,
    )

    assignment = await _prepare_live_execution(session_factory, 'http://manager.internal')
    await _bind_execution_artifact_scope(session_factory, assignment.execution_id, assignment.assignment_id)
    async with session_factory() as session:
        execution = await session.get(ExecutionRow, assignment.execution_id)
        assert execution is not None
        execution.spec_type = 'nexus_batch'
        await session.commit()

    config = DistributedConfig(
        secrets={'internal_token': 'internal-secret'},
        artifact_local_root=str(tmp_path / 'artifacts'),
        artifact_upload_spool_dir=str(tmp_path / 'spool'),
    )
    client = TestClient(TestServer(create_application_app(session_factory=session_factory, config=config)))
    await client.start_server()
    content = encode_run_result_content(
        status='succeeded',
        output={'batch': {'total': 1, 'succeeded': 1}},
        error=None,
    )
    try:
        artifact_transport = ApplicationGatewayArtifactTransport(
            base_url=str(client.make_url('/')).rstrip('/'),
            internal_token=config.read_internal_token(),
        )
        artifact_client = RunResultArtifactGateway(artifact_transport)
        ref = await artifact_client.publish_run_result(
            execution_id=assignment.execution_id,
            assignment_id=assignment.assignment_id,
            attempt_id=assignment.attempt_id,
            producer_key=run_result_producer_key(assignment.execution_id, assignment.attempt_id),
            content=content,
        )
        repeated = await artifact_client.publish_run_result(
            execution_id=assignment.execution_id,
            assignment_id=assignment.assignment_id,
            attempt_id=assignment.attempt_id,
            producer_key=run_result_producer_key(assignment.execution_id, assignment.attempt_id),
            content=encode_run_result_content(
                status='succeeded', output={'batch': {'succeeded': 1, 'total': 1}}, error=None
            ),
        )
        await artifact_transport.close()
    finally:
        await client.close()

    assert ref['kind'] == 'run_result'
    assert repeated == ref
    async with session_factory() as session:
        execution = await session.get(ExecutionRow, assignment.execution_id)
        artifact = await session.get(ArtifactRow, ref['artifact_id'])
    assert execution is not None
    assert artifact is not None
    assert execution.result_artifact_id == ref['artifact_id']
    assert artifact.producer_key == f'run-result:{assignment.execution_id}:{assignment.attempt_id}'


@pytest.mark.parametrize('value', ['../health', 'safe/../health', r'safe\..\health', 'safe\x00health'])
@pytest.mark.parametrize(
    'validator',
    [validate_application_proxy_path, validate_manager_proxy_path],
)
def test_runtime_proxy_rejects_path_traversal(validator, value):
    with pytest.raises(web.HTTPBadRequest) as exc_info:
        validator(value)
    assert exc_info.value.text == 'invalid_proxy_path'


async def _prepare_job_execution(session_factory, endpoint_url: str):
    async with session_factory() as session:
        manager = ManagerRepository(session)
        await manager.register_session(
            manager_id='im-job',
            session_id='session-job',
            endpoint_url='http://manager.internal',
            supported_modes=[ExecutionMode.JOB],
            supported_spec_types=['nexus_single'],
            supported_instance_modes=['asyncio'],
            capacity_total=1,
            force_replace=False,
            now=utcnow(),
        )
        execution_repo = ExecutionRepository(session, shard_count=8)
        execution = await execution_repo.create_execution(
            CreateExecution(
                execution_id='job-1',
                mode=ExecutionMode.JOB,
                spec_type='nexus_single',
                instance_mode='asyncio',
                spec={'task_name': 'identity'},
                route_status=RouteStatus.ACTIVE,
            ),
            now=utcnow(),
        )
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        assert assignment is not None
        await manager.ack_assignment(
            assignment.assignment_id, assignment.manager_id, assignment.manager_session_id, now=utcnow()
        )
        sink_metadata = {
            'version': 2,
            'format': 'span_event_record.jsonl',
            'run_kind': 'single',
            'run_id': 'job-1',
            'run_span_id': 'run-span-1',
            'execution_id': 'job-1',
            'assignment_id': assignment.assignment_id,
            'attempt_id': assignment.attempt_id,
            'trace_events_path': '/data/nexus-distributed/live-artifacts/job-1/attempt/trace_events.jsonl',
            'trace_event_cursor': 5,
            'status': 'succeeded',
        }
        metadata = {
            'version': 4,
            'run_id': 'job-1',
            'run_span_id': 'run-span-1',
            'run_kind': 'single',
            'sinks': [
                {
                    'name': 'local',
                    'metadata': sink_metadata,
                }
            ],
        }
        await manager.mark_running(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            endpoint_url=endpoint_url.replace('{assignment_id}', assignment.assignment_id).replace(
                '%7Bassignment_id%7D', assignment.assignment_id
            ),
            endpoint_ready=False,
            runtime_info={'_artifacts': {'single': metadata}},
            now=utcnow(),
        )
        await manager.complete_execution(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            result={
                '_external': True,
                'kind': 'nexus_run_trace_result',
                'version': 1,
                'metadata': metadata,
            },
            now=utcnow(),
        )
        await session.commit()
        return assignment


async def test_application_gateway_submits_command_and_proxies_runtime(session_factory):
    wakeups = []

    async def proxy_ping(request):
        return web.json_response(
            {
                'path': request.path,
                'method': request.method,
                'query': dict(request.query),
                'body': await request.text(),
                'authorization': request.headers.get('Authorization'),
                'custom': request.headers.get('X-Custom'),
                'request_id': request.headers.get('X-Request-Id'),
            }
        )

    async def wakeup(request):
        wakeups.append(request.headers.get('Authorization'))
        return web.json_response({'status': 'ok'})

    manager_app = web.Application()
    manager_app.router.add_post('/v1/internal/wakeup', wakeup)
    manager_app.router.add_route(
        '*',
        '/v1/internal/executions/{assignment_id}/runtimes/{runtime_name}/proxy/{tail:.*}',
        proxy_ping,
    )
    manager_client = TestClient(TestServer(manager_app))
    await manager_client.start_server()
    try:
        assignment = await _prepare_live_execution(
            session_factory,
            str(manager_client.make_url('/v1/internal/executions/asn-placeholder')).replace(
                'asn-placeholder',
                '{assignment_id}',
            ),
        )
        # Update the endpoint now that the actual assignment id is known.
        endpoint_url = str(manager_client.make_url(f'/v1/internal/executions/{assignment.assignment_id}'))
        async with session_factory() as session:
            manager_row = await session.get(InstanceManagerRow, 'im-live')
            assert manager_row is not None
            manager_row.endpoint_url = str(manager_client.make_url('/')).rstrip('/')
            manager = ManagerRepository(session)
            await manager.mark_running(
                assignment.assignment_id,
                assignment.manager_id,
                assignment.manager_session_id,
                endpoint_url=endpoint_url,
                endpoint_ready=True,
                runtime_info={},
                now=utcnow(),
            )
            await manager.mark_waiting_input(
                assignment.assignment_id,
                assignment.manager_id,
                assignment.manager_session_id,
                payload={'prompt': 'choose'},
                now=utcnow(),
            )
            await session.commit()

        config = DistributedConfig(
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
            secrets={'internal_token': 'internal-secret'},
            gateway_json_body_max_size_bytes=256,
            gateway_proxy_body_max_size_bytes=16,
        )
        app = create_application_app(session_factory=session_factory, config=config)
        app_client = TestClient(TestServer(app))
        await app_client.start_server()
        try:
            preflight = await app_client.options(
                '/v1/executions/live-1/commands',
                headers={
                    'Origin': 'http://console.local',
                    'Access-Control-Request-Headers': 'content-type',
                },
            )
            assert preflight.status == 204
            assert preflight.headers['Access-Control-Allow-Origin'] == 'http://console.local'
            assert preflight.headers['Access-Control-Allow-Headers'] == 'content-type'

            response = await app_client.post(
                '/v1/executions/live-1/commands',
                json={'type': 'input.submit', 'payload': {'text': 'Alpha'}, 'idempotency_key': 'turn-1'},
                headers={'Origin': 'http://console.local'},
            )
            assert response.status == 202
            assert response.headers['Access-Control-Allow-Origin'] == 'http://console.local'
            command = await response.json()
            assert command['assignment_id'] == assignment.assignment_id
            assert command['attempt_id'] == assignment.attempt_id
            assert 'T' in command['created_at']
            assert 'T' in command['updated_at']
            assert wakeups == ['Bearer internal-secret']

            response = await app_client.get(
                '/v1/executions/live-1/runtimes/main/proxy/ping',
                headers={'Origin': 'http://console.local'},
            )
            assert response.status == 200
            assert response.headers['Access-Control-Allow-Origin'] == 'http://console.local'
            proxied = await response.json()
            assert proxied['path'].endswith(
                f'/v1/internal/executions/{assignment.assignment_id}/runtimes/main/proxy/ping'
            )

            response = await app_client.post(
                '/v1/executions/live-1/runtimes/main/proxy/submit?debug=1',
                data='payload-body',
                headers={'X-Custom': 'drop-me', 'X-Request-Id': 'request-1'},
            )
            assert response.status == 200
            proxied = await response.json()
            assert proxied['method'] == 'POST'
            assert proxied['query'] == {'debug': '1'}
            assert proxied['body'] == 'payload-body'
            assert proxied['authorization'] == 'Bearer internal-secret'
            assert proxied['custom'] is None
            assert proxied['request_id'] == 'request-1'

            response = await app_client.post(
                '/v1/executions/live-1/runtimes/main/proxy/submit',
                data='x' * 17,
            )
            assert response.status == 413

            response = await app_client.post(
                '/v1/executions/live-1/commands',
                json={'type': 'input.submit', 'payload': {'text': 'x' * 512}},
            )
            assert response.status == 413

            duplicate = await app_client.post(
                '/v1/executions/live-1/commands',
                json={'type': 'input.submit', 'payload': {'text': 'Ignored'}, 'idempotency_key': 'turn-1'},
            )
            assert duplicate.status == 202
            duplicate_payload = await duplicate.json()
            assert duplicate_payload['command_id'] == command['command_id']
            assert duplicate_payload['payload'] == {'text': 'Alpha'}
        finally:
            await app_client.close()
    finally:
        await manager_client.close()


async def test_application_gateway_forwards_runtime_aiomonitor_public_base_path(session_factory):
    received: list[dict[str, str | None]] = []

    async def aiomonitor_proxy(request):
        received.append(
            {
                'path': request.path,
                'debug_base_path': request.headers.get('X-Nexus-Debug-Base-Path'),
                'authorization': request.headers.get('Authorization'),
            }
        )
        return web.Response(text='runtime aiomonitor')

    manager_app = web.Application()
    manager_app.router.add_route(
        '*',
        '/v1/internal/executions/{assignment_id}/runtimes/{runtime_name}/proxy/debug/aiomonitor/{loop}/{tail:.*}',
        aiomonitor_proxy,
    )
    manager_client = TestClient(TestServer(manager_app))
    await manager_client.start_server()
    try:
        assignment = await _prepare_live_execution(session_factory, 'http://placeholder')
        endpoint_url = str(manager_client.make_url(f'/v1/internal/executions/{assignment.assignment_id}'))
        async with session_factory() as session:
            manager_row = await session.get(InstanceManagerRow, 'im-live')
            assert manager_row is not None
            manager_row.endpoint_url = str(manager_client.make_url('/')).rstrip('/')
            manager = ManagerRepository(session)
            await manager.mark_running(
                assignment.assignment_id,
                assignment.manager_id,
                assignment.manager_session_id,
                endpoint_url=endpoint_url,
                endpoint_ready=True,
                runtime_info={},
                now=utcnow(),
            )
            await session.commit()

        app = create_application_app(
            session_factory=session_factory,
            config=DistributedConfig(
                shard_count=8,
                manager_healthy_ttl=timedelta(seconds=15),
                secrets={'internal_token': 'internal-secret'},
            ),
        )
        app_client = TestClient(TestServer(app))
        await app_client.start_server()
        try:
            response = await app_client.get(
                '/v1/executions/live-1/runtimes/main/proxy/debug/aiomonitor/main/static/app.js?theme=dark'
            )

            assert response.status == 200
            assert await response.text() == 'runtime aiomonitor'
            assert received == [
                {
                    'path': (
                        f'/v1/internal/executions/{assignment.assignment_id}'
                        '/runtimes/main/proxy/debug/aiomonitor/main/static/app.js'
                    ),
                    'debug_base_path': '/v1/executions/live-1/runtimes/main/proxy/debug/aiomonitor/main',
                    'authorization': 'Bearer internal-secret',
                }
            ]
        finally:
            await app_client.close()
    finally:
        await manager_client.close()


async def test_application_gateway_returns_active_execution_info(session_factory):
    info_requests = []

    async def execution_info(request):
        info_requests.append(request.headers.get('Authorization'))
        return web.json_response(
            {
                'assignment_id': request.match_info['assignment_id'],
                'execution_id': 'live-1',
                'attempt_id': 'live-1-attempt-1',
                'driver': {'spec_type': 'nexus_live', 'instance_mode': 'asyncio'},
                'runtime_info': {'runtimes': {'main': {'base_url': 'http://runtime'}}},
                'live': {'status': 'idle', 'events_cursor': 12, 'current_turn_id': 'turn-1'},
            }
        )

    manager_app = web.Application()
    manager_app.router.add_get('/v1/internal/executions/{assignment_id}/info', execution_info)
    manager_client = TestClient(TestServer(manager_app))
    await manager_client.start_server()
    try:
        assignment = await _prepare_live_execution(session_factory, 'http://placeholder')
        endpoint_url = str(manager_client.make_url(f'/v1/internal/executions/{assignment.assignment_id}'))
        async with session_factory() as session:
            manager_row = await session.get(InstanceManagerRow, 'im-live')
            assert manager_row is not None
            manager_row.endpoint_url = str(manager_client.make_url('/')).rstrip('/')
            manager = ManagerRepository(session)
            await manager.mark_running(
                assignment.assignment_id,
                assignment.manager_id,
                assignment.manager_session_id,
                endpoint_url=endpoint_url,
                endpoint_ready=True,
                runtime_info={},
                now=utcnow(),
            )
            await manager.mark_waiting_input(
                assignment.assignment_id,
                assignment.manager_id,
                assignment.manager_session_id,
                payload={'turn_id': 'turn-1', 'request': {'hint': 'choose'}},
                now=utcnow(),
            )
            await session.commit()

        app = create_application_app(
            session_factory=session_factory,
            config=DistributedConfig(
                shard_count=8,
                manager_healthy_ttl=timedelta(seconds=15),
                secrets={'internal_token': 'internal-secret'},
            ),
        )
        app_client = TestClient(TestServer(app))
        await app_client.start_server()
        try:
            response = await app_client.get('/v1/executions/live-1/info')
            assert response.status == 200
            payload = await response.json()
            assert payload['assignment_id'] == assignment.assignment_id
            assert payload['live'] == {'status': 'idle', 'events_cursor': 12, 'current_turn_id': 'turn-1'}
            assert info_requests == ['Bearer internal-secret']
        finally:
            await app_client.close()
    finally:
        await manager_client.close()


async def test_application_gateway_public_routes_are_under_v1(session_factory):
    app = create_application_app(
        session_factory=session_factory,
        config=DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15)),
    )
    app_client = TestClient(TestServer(app))
    await app_client.start_server()
    try:
        response = await app_client.get('/v1/health')
        assert response.status == 200
        assert await response.json() == {'status': 'ok'}

        response = await app_client.get('/health')
        assert response.status == 404
    finally:
        await app_client.close()


async def test_application_gateway_does_not_mount_live_compatibility_routes(session_factory):
    app = create_application_app(
        session_factory=session_factory,
        config=DistributedConfig(auth_mode='disabled'),
    )
    app_client = TestClient(TestServer(app))
    await app_client.start_server()
    try:
        assert (await app_client.get('/compat/live/health')).status == 404
        assert (await app_client.post('/compat/live/v1/runs', json={})).status == 404
    finally:
        await app_client.close()


async def test_application_gateway_http_proxy_streams_without_buffering_full_response(session_factory):
    async def slow_stream(_request):
        response = web.StreamResponse(status=200, headers={'Content-Type': 'text/plain'})
        await response.prepare(_request)
        await response.write(b'first\n')
        await asyncio.sleep(1.0)
        await response.write(b'second\n')
        await response.write_eof()
        return response

    manager_app = web.Application()
    manager_app.router.add_get(
        '/v1/internal/executions/{assignment_id}/runtimes/{runtime_name}/proxy/slow', slow_stream
    )
    manager_client = TestClient(TestServer(manager_app))
    await manager_client.start_server()
    try:
        assignment = await _prepare_live_execution(session_factory, 'http://placeholder')
        endpoint_url = str(manager_client.make_url(f'/v1/internal/executions/{assignment.assignment_id}'))
        async with session_factory() as session:
            manager_row = await session.get(InstanceManagerRow, 'im-live')
            assert manager_row is not None
            manager_row.endpoint_url = str(manager_client.make_url('/')).rstrip('/')
            manager = ManagerRepository(session)
            await manager.mark_running(
                assignment.assignment_id,
                assignment.manager_id,
                assignment.manager_session_id,
                endpoint_url=endpoint_url,
                endpoint_ready=True,
                runtime_info={},
                now=utcnow(),
            )
            await session.commit()

        app = create_application_app(
            session_factory=session_factory,
            config=DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15)),
        )
        app_client = TestClient(TestServer(app))
        await app_client.start_server()
        try:
            response = await asyncio.wait_for(
                app_client.get('/v1/executions/live-1/runtimes/main/proxy/slow'),
                timeout=0.2,
            )
            assert response.status == 200
            first = await asyncio.wait_for(response.content.readline(), timeout=0.2)
            assert first == b'first\n'
            response.close()
        finally:
            await app_client.close()
    finally:
        await manager_client.close()


async def test_application_gateway_live_events_path_proxies_driver_events(session_factory):
    async def stream_events(request):
        response = web.StreamResponse(
            status=200,
            headers={'Content-Type': 'text/event-stream'},
        )
        await response.prepare(request)
        await response.write(b'event: input.required\n')
        await response.write(b'data: {"id": 1, "prompt": "choose"}\n\n')
        await response.write_eof()
        return response

    manager_app = web.Application()
    manager_app.router.add_get('/v1/internal/executions/{assignment_id}/events', stream_events)
    manager_client = TestClient(TestServer(manager_app))
    await manager_client.start_server()
    try:
        assignment = await _prepare_live_execution(session_factory, 'http://placeholder')
        endpoint_url = str(manager_client.make_url(f'/v1/internal/executions/{assignment.assignment_id}'))
        async with session_factory() as session:
            manager_row = await session.get(InstanceManagerRow, 'im-live')
            assert manager_row is not None
            manager_row.endpoint_url = str(manager_client.make_url('/')).rstrip('/')
            manager = ManagerRepository(session)
            await manager.mark_running(
                assignment.assignment_id,
                assignment.manager_id,
                assignment.manager_session_id,
                endpoint_url=endpoint_url,
                endpoint_ready=True,
                runtime_info={},
                now=utcnow(),
            )
            await session.commit()

        app = create_application_app(
            session_factory=session_factory,
            config=DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15)),
        )
        app_client = TestClient(TestServer(app))
        await app_client.start_server()
        try:
            response = await app_client.get('/v1/executions/live-1/live/events?cursor=0')
            assert response.status == 200
            assert await response.text() == 'event: input.required\ndata: {"id": 1, "prompt": "choose"}\n\n'
        finally:
            await app_client.close()
    finally:
        await manager_client.close()


async def test_application_gateway_reads_terminal_live_artifacts(session_factory):
    artifact_event_requests = []

    async def artifact_events(request):
        artifact_event_requests.append(
            {
                'cursor': request.query.get('cursor'),
                'last_event_id': request.headers.get('Last-Event-ID'),
            }
        )
        response = web.StreamResponse(status=200, headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        await response.write(b'id: 5\nevent: run.output\ndata: {"success": true}\n\n')
        await response.write_eof()
        return response

    async def artifact_result(request):
        return web.json_response({'success': True, 'large': 'payload'})

    manager_app = web.Application()
    manager_app.router.add_get('/v1/internal/executions/{assignment_id}/artifacts/events', artifact_events)
    manager_app.router.add_get('/v1/internal/executions/{assignment_id}/artifacts/result', artifact_result)
    manager_client = TestClient(TestServer(manager_app))
    await manager_client.start_server()
    try:
        assignment = await _prepare_live_execution(session_factory, 'http://placeholder')
        sink_metadata = {
            'version': 2,
            'format': 'span_event_record.jsonl',
            'run_kind': 'live',
            'run_id': 'live-1',
            'run_span_id': 'run-span-1',
            'execution_id': 'live-1',
            'assignment_id': assignment.assignment_id,
            'attempt_id': assignment.attempt_id,
            'trace_events_path': '/data/nexus-distributed/live-artifacts/live-1/attempt/trace_events.jsonl',
            'trace_event_cursor': 5,
            'status': 'succeeded',
        }
        artifact_meta = {
            'version': 4,
            'run_id': 'live-1',
            'run_span_id': 'run-span-1',
            'run_kind': 'live',
            'sinks': [
                {
                    'name': 'local',
                    'metadata': sink_metadata,
                }
            ],
        }
        pointer = {
            '_external': True,
            'kind': 'nexus_live_result',
            'version': 2,
            'metadata': artifact_meta,
        }
        async with session_factory() as session:
            manager_row = await session.get(InstanceManagerRow, 'im-live')
            assert manager_row is not None
            manager_row.endpoint_url = str(manager_client.make_url('/')).rstrip('/')
            manager = ManagerRepository(session)
            await manager.mark_running(
                assignment.assignment_id,
                assignment.manager_id,
                assignment.manager_session_id,
                endpoint_url=None,
                endpoint_ready=False,
                runtime_info={'_artifacts': {'live': artifact_meta}},
                now=utcnow(),
            )
            await manager.complete_execution(
                assignment.assignment_id,
                assignment.manager_id,
                assignment.manager_session_id,
                result=pointer,
                now=utcnow(),
            )
            await session.commit()

        app = create_application_app(
            session_factory=session_factory,
            config=DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15)),
        )
        app_client = TestClient(TestServer(app))
        await app_client.start_server()
        try:
            response = await app_client.get('/v1/executions/live-1/live/events?cursor=4')
            assert response.status == 200
            assert await response.text() == 'id: 5\nevent: run.output\ndata: {"success": true}\n\n'
            assert artifact_event_requests[-1] == {'cursor': '4', 'last_event_id': None}

            response = await app_client.get(
                '/v1/executions/live-1/live/events?cursor=0',
                headers={'Last-Event-ID': '4'},
            )
            assert response.status == 200
            assert await response.text() == 'id: 5\nevent: run.output\ndata: {"success": true}\n\n'
            assert artifact_event_requests[-1] == {'cursor': '0', 'last_event_id': '4'}

            response = await app_client.get('/v1/executions/live-1/live/result')
            assert response.status == 404
            assert (await response.json())['error'] == 'artifact_not_found'
        finally:
            await app_client.close()
    finally:
        await manager_client.close()


async def test_application_gateway_does_not_read_terminal_single_result_from_manager(session_factory):
    artifact_result_requests = 0

    async def artifact_result(request):
        nonlocal artifact_result_requests
        artifact_result_requests += 1
        return web.json_response(
            {
                'success': True,
                'task_output': {'message': 'done'},
                'processor_results': {'trace': {'tokens': 1}},
            }
        )

    manager_app = web.Application()
    manager_app.router.add_get('/v1/internal/executions/{assignment_id}/artifacts/result', artifact_result)
    manager_client = TestClient(TestServer(manager_app))
    await manager_client.start_server()
    try:
        await _prepare_job_execution(
            session_factory,
            str(manager_client.make_url('/v1/internal/executions/{assignment_id}')),
        )
        async with session_factory() as session:
            manager_row = await session.get(InstanceManagerRow, 'im-job')
            assert manager_row is not None
            manager_row.endpoint_url = str(manager_client.make_url('/')).rstrip('/')
            await session.commit()

        app = create_application_app(
            session_factory=session_factory,
            config=DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15)),
        )
        app_client = TestClient(TestServer(app))
        await app_client.start_server()
        try:
            response = await app_client.get('/v1/executions/job-1/result')
            assert response.status == 404
            assert (await response.json())['error'] == 'artifact_not_found'
            assert artifact_result_requests == 0
        finally:
            await app_client.close()
    finally:
        await manager_client.close()


async def test_application_gateway_maps_route_failures_to_precise_status_codes(session_factory):
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    app = create_application_app(session_factory=session_factory, config=config)
    app_client = TestClient(TestServer(app))
    await app_client.start_server()
    try:
        cached = await _prepare_live_execution(session_factory, 'http://manager.internal/executions/cached')
        async with session_factory() as session:
            execution = await session.get(ExecutionRow, cached.execution_id)
            assert execution is not None
            execution.route_status = RouteStatus.CACHED
            execution.updated_at = utcnow()
            await session.commit()

        response = await app_client.get('/v1/executions/live-1/runtimes/main/proxy/ping')
        assert response.status == 409
        assert (await response.json())['error'] == 'route_not_active'

        async with session_factory() as session:
            execution = await session.get(ExecutionRow, cached.execution_id)
            assert execution is not None
            execution.route_status = RouteStatus.CLOSED
            execution.status = ExecutionStatus.KILLED
            execution.updated_at = utcnow()
            await session.commit()

        response = await app_client.get('/v1/executions/live-1/runtimes/main/proxy/ping')
        assert response.status == 410
        assert (await response.json())['error'] == 'execution_closed'
    finally:
        await app_client.close()


async def test_application_gateway_live_events_reject_job_and_cached_live(session_factory):
    async with session_factory() as session:
        repo = ExecutionRepository(session, shard_count=8)
        await repo.create_execution(
            CreateExecution(
                execution_id='job-events',
                mode=ExecutionMode.JOB,
                spec_type='nexus_single',
                instance_mode='asyncio',
                spec={},
            ),
            now=utcnow(),
        )
        await session.commit()

    cached = await _prepare_live_execution(session_factory, 'http://manager.internal/executions/cached')
    async with session_factory() as session:
        execution = await session.get(ExecutionRow, cached.execution_id)
        assert execution is not None
        execution.route_status = RouteStatus.CACHED
        await session.commit()

    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    app = create_application_app(session_factory=session_factory, config=config)
    app_client = TestClient(TestServer(app))
    await app_client.start_server()
    try:
        response = await app_client.get('/v1/executions/job-events/live/events')
        assert response.status == 404
        assert (await response.json())['error'] == 'events_not_supported'

        response = await app_client.get('/v1/executions/live-1/live/events')
        assert response.status == 409
        assert (await response.json())['error'] == 'route_not_active'

        response = await app_client.get('/v1/executions/live-1/events')
        assert response.status == 404
    finally:
        await app_client.close()


async def test_application_gateway_returns_unavailable_for_stale_manager(session_factory):
    await _prepare_live_execution(session_factory, 'http://manager.internal/executions/live-1')
    async with session_factory() as session:
        manager = await session.get(InstanceManagerRow, 'im-live')
        liveness = await session.get(InstanceManagerLivenessRow, 'im-live')
        assert manager is not None
        assert liveness is not None
        stale_at = utcnow() - timedelta(seconds=30)
        manager.last_heartbeat_at = stale_at
        liveness.last_heartbeat_at = stale_at
        await session.commit()
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    app = create_application_app(session_factory=session_factory, config=config)
    app_client = TestClient(TestServer(app))
    await app_client.start_server()
    try:
        response = await app_client.get('/v1/executions/live-1/runtimes/main/proxy/ping')
        assert response.status == 503
        assert (await response.json())['error'] == 'executor_unavailable'
    finally:
        await app_client.close()


async def test_application_gateway_returns_unavailable_when_upstream_proxy_is_down(session_factory):
    await _prepare_live_execution(session_factory, 'http://127.0.0.1:9/v1/internal/executions/live-1')
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    app = create_application_app(session_factory=session_factory, config=config)
    app_client = TestClient(TestServer(app))
    await app_client.start_server()
    try:
        response = await app_client.get('/v1/executions/live-1/runtimes/main/proxy/ping')
        assert response.status == 503
        assert (await response.json())['error'] == 'executor_unavailable'
    finally:
        await app_client.close()


async def test_application_gateway_proxies_websocket_runtime(session_factory):
    upstream_protocols = []

    async def websocket_handler(request):
        upstream_protocols.append(request.headers.get('Sec-WebSocket-Protocol'))
        ws = web.WebSocketResponse(protocols=('runtime.v1',))
        await ws.prepare(request)
        async for message in ws:
            if message.type == aiohttp.WSMsgType.TEXT:
                await ws.send_str(f'echo:{message.data}')
            elif message.type == aiohttp.WSMsgType.BINARY:
                await ws.send_bytes(b'echo:' + message.data)
        return ws

    manager_app = web.Application()
    manager_app.router.add_get(
        '/v1/internal/executions/{assignment_id}/runtimes/{runtime_name}/proxy/ws',
        websocket_handler,
    )
    manager_client = TestClient(TestServer(manager_app))
    await manager_client.start_server()
    try:
        assignment = await _prepare_live_execution(session_factory, 'http://placeholder')
        endpoint_url = str(manager_client.make_url(f'/v1/internal/executions/{assignment.assignment_id}'))
        async with session_factory() as session:
            manager = ManagerRepository(session)
            await manager.mark_running(
                assignment.assignment_id,
                assignment.manager_id,
                assignment.manager_session_id,
                endpoint_url=endpoint_url,
                endpoint_ready=True,
                runtime_info={},
                now=utcnow(),
            )
            await manager.mark_waiting_input(
                assignment.assignment_id,
                assignment.manager_id,
                assignment.manager_session_id,
                payload={'prompt': 'choose'},
                now=utcnow(),
            )
            await session.commit()

        config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        app_client = TestClient(TestServer(create_application_app(session_factory=session_factory, config=config)))
        await app_client.start_server()
        try:
            ws = await app_client.ws_connect(
                '/v1/executions/live-1/runtimes/main/proxy/ws',
                protocols=('nexus-csrf.' + '0' * 64, 'runtime.v1'),
            )
            assert ws.protocol == 'runtime.v1'
            await ws.send_str('hello')
            message = await ws.receive(timeout=2)
            assert message.type == aiohttp.WSMsgType.TEXT
            assert message.data == 'echo:hello'
            await ws.close()
            assert upstream_protocols == ['runtime.v1']
        finally:
            await app_client.close()
    finally:
        await manager_client.close()
