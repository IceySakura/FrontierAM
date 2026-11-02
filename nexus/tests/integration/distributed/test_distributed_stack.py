from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import aiohttp
import pytest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import func, insert, inspect, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.artifacts.store import LocalArtifactStore
from nexus.distributed.config import DistributedConfig
from nexus.distributed.errors import DomainConflictError
from nexus.distributed.execution.models import ExecutionMode
from nexus.distributed.persistence import db as distributed_db
from nexus.distributed.persistence.db import init_database, run_idempotent_transaction, upgrade_database
from nexus.distributed.persistence.migrations import HEAD_REVISION, MIGRATIONS
from nexus.distributed.persistence.orm import (
    ArtifactRefRow,
    ArtifactRow,
    ArtifactUploadRow,
    AuditRow,
    Base,
    BatchItemRow,
    DependencyBuildRow,
    EnvironmentAllocationRow,
    EnvironmentResourceUsageRow,
    ExecutionRow,
    QuotaUsageRow,
    ShardLeaseRow,
)
from nexus.distributed.repositories.access import AccessRepository
from nexus.distributed.repositories.allocation import (
    AllocationRepository,
    AllocationResources,
    CreateEnvironmentAllocation,
)
from nexus.distributed.repositories.artifact import ArtifactRepository, CreateArtifactUpload
from nexus.distributed.repositories.audit import AuditRepository
from nexus.distributed.repositories.environment import EnvironmentRepository
from nexus.distributed.repositories.execution import CreateExecution, ExecutionRepository
from nexus.distributed.repositories.manager import ManagerRepository
from nexus.distributed.repositories.provider_config import ProviderConfigRepository
from nexus.distributed.repositories.quota import QuotaExceededError, QuotaRepository
from nexus.distributed.repositories.scheduler import SchedulerRepository
from nexus.distributed.repositories.submission import SubmissionRepository
from nexus.distributed.submission.specs import SingleSubmission
from nexus.distributed.utils.clock import utcnow
from nexus.distributed.batch.task_list import is_valid_batch_item_id


pytestmark = pytest.mark.integration


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


async def _reset_schema(database_url: str) -> None:
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.execute(text('DROP TABLE IF EXISTS alembic_version'))
            await conn.execute(text('DROP TABLE IF EXISTS schema_version'))
    finally:
        await engine.dispose()
    await init_database(DistributedConfig(database_url=database_url))


async def _wait_database(database_url: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        engine = create_async_engine(database_url)
        try:
            async with engine.connect() as conn:
                await conn.execute(text('SELECT 1'))
            await engine.dispose()
            return
        except Exception as exc:
            last_error = exc
            await engine.dispose()
            await asyncio.sleep(1)
    raise TimeoutError(f'database did not become ready: {last_error}')


async def _seed_resource_usage_lock_fixture(session_factory: async_sessionmaker) -> str:
    now = utcnow()
    async with session_factory() as session:
        await AccessRepository(session).upsert_user(
            principal_id='resource-owner',
            issuer='tai',
            tenant='integration',
            subject='resource-owner',
            roles={'user'},
            display_name=None,
            actor='admin',
            now=now,
        )
        environments = EnvironmentRepository(session)
        await environments.create_environment(
            environment_id='resource-environment',
            name='resource-environment',
            allocation_type='dynamic',
            provider_type='kubernetes',
            provider_config_ref='kubernetes/integration',
            resource_limits={'manager_count': 5},
            default_timeouts={},
            status='enabled',
            actor='admin',
            now=now,
        )
        await environments.upsert_grant(
            principal_id='resource-owner',
            environment_id='resource-environment',
            limits={'manager_count': 5},
            enabled=True,
            actor='admin',
            now=now,
        )
        allocation = await AllocationRepository(session).create_allocation(
            CreateEnvironmentAllocation(
                allocation_id='resource-allocation',
                top_level_execution_id='resource-execution',
                owner_principal_id='resource-owner',
                environment_id='resource-environment',
                pool_id='execution',
                workload='execution',
                desired_manager_count=2,
                resources=AllocationResources(
                    manager_count=2,
                    cpu_millicores=2000,
                    memory_bytes=2048,
                    slots=2,
                    active_batches=0,
                ),
                allocation_spec={'image': 'nexus:test', 'manager_count': 2},
                allocation_spec_hash='a' * 64,
            ),
            now=now,
        )
        await session.commit()
    return allocation.allocation_id


@pytest.fixture(scope='session')
def distributed_database_url() -> Iterator[str]:
    configured = os.getenv('NEXUS_DIST_TEST_DATABASE_URL')
    if configured:
        yield configured
        return

    if os.getenv('NEXUS_DIST_TEST_DOCKER_MYSQL') != '1':
        pytest.skip('set NEXUS_DIST_TEST_DATABASE_URL or NEXUS_DIST_TEST_DOCKER_MYSQL=1 to run distributed integration')
    if not shutil.which('docker'):
        pytest.skip('docker is required for distributed MySQL integration')

    name = f'nexus-dist-test-{uuid.uuid4().hex[:8]}'
    password = 'nexus_dist_test'
    subprocess.run(
        [
            'docker',
            'run',
            '-d',
            '--rm',
            '--name',
            name,
            '-e',
            f'MYSQL_ROOT_PASSWORD={password}',
            '-e',
            'MYSQL_DATABASE=nexus_dist',
            '-p',
            '127.0.0.1::3306',
            'mysql:8.0',
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    try:
        port_output = subprocess.check_output(['docker', 'port', name, '3306/tcp'], text=True).strip()
        port = port_output.rsplit(':', 1)[1]
        url = f'mysql+asyncmy://root:{password}@127.0.0.1:{port}/nexus_dist?charset=utf8mb4'
        asyncio.run(_wait_database(url))
        yield url
    finally:
        subprocess.run(['docker', 'rm', '-f', name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@contextlib.contextmanager
def _start_process(tmp_path: Path, name: str, args: list[str], env: dict[str, str]) -> Iterator[subprocess.Popen]:
    log_path = tmp_path / f'{name}.log'
    with log_path.open('w') as log:
        process = subprocess.Popen(
            [sys.executable, '-m', 'nexus.cli.main', *args],
            cwd=Path(__file__).parents[3],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            yield process
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


async def _wait_http_ok(url: str, *, headers: dict[str, str] | None = None, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        return
            except Exception:
                pass
            await asyncio.sleep(0.2)
    raise TimeoutError(f'HTTP endpoint did not become ready: {url}')


async def _poll_execution(control_url: str, execution_id: str, predicate, timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            async with session.get(f'{control_url}/v1/executions/{execution_id}') as response:
                payload = await response.json()
            if predicate(payload):
                return payload
            await asyncio.sleep(0.2)
    raise TimeoutError(f'execution {execution_id} did not reach expected state; last={payload}')


async def _read_sse_until(url: str, event_type: str, timeout: float = 10.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout + 1)) as response:
            assert response.status == 200
            current_event = None
            current_data = None
            while time.monotonic() < deadline:
                line = await response.content.readline()
                if not line:
                    await asyncio.sleep(0.05)
                    continue
                text_line = line.decode().strip()
                if text_line.startswith('event:'):
                    current_event = text_line.split(':', 1)[1].strip()
                elif text_line.startswith('data:'):
                    current_data = json.loads(text_line.split(':', 1)[1].strip())
                elif text_line == '' and current_event == event_type and current_data is not None:
                    return current_data
    raise TimeoutError(f'SSE event {event_type} not received from {url}')


async def _resolved_execution_result(app_url: str, item: dict[str, Any]) -> dict[str, Any]:
    result = item.get('result_json') or {}
    if isinstance(result, dict) and result.get('_external') is True:
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{app_url}/v1/executions/{item["execution_id"]}/result') as response:
                assert response.status == 200
                return await response.json()
    return result


def _distributed_processes(
    control_port: int,
    app_port: int,
    manager_port: int,
    *,
    executor: str = 'asyncio',
    environment_id: str | None = None,
    pool_id: str | None = None,
    labels: dict[str, str] | None = None,
) -> list[tuple[str, list[str]]]:
    manager_args = [
        'distributed',
        'instance-manager',
        '--host',
        '127.0.0.1',
        '--port',
        str(manager_port),
        '--instance-manager-id',
        'im-1',
        '--capacity',
        '4',
        '--executor',
        executor,
    ]
    if environment_id is not None:
        manager_args.extend(['--environment-id', environment_id])
    if pool_id is not None:
        manager_args.extend(['--pool-id', pool_id])
    for key, value in (labels or {}).items():
        manager_args.extend(['--label', f'{key}={value}'])
    manager_args.append('--force-replace')
    return [
        (
            'control',
            [
                'distributed',
                'control-gateway',
                '--host',
                '127.0.0.1',
                '--port',
                str(control_port),
            ],
        ),
        (
            'application',
            [
                'distributed',
                'application-gateway',
                '--host',
                '127.0.0.1',
                '--port',
                str(app_port),
            ],
        ),
        ('manager', manager_args),
        ('scheduler', ['distributed', 'scheduler', '--owner-id', 'scheduler-1']),
        ('reconciler', ['distributed', 'reconciler']),
    ]


def _base_process_env(database_url: str, token: str, *, application_internal_base_url: str) -> dict[str, str]:
    env = os.environ.copy()
    env['NEXUS_DIST_DATABASE_URL'] = database_url
    env['NEXUS_DIST_INTERNAL_TOKEN'] = token
    env['NEXUS_DIST_APPLICATION_INTERNAL_BASE_URL'] = application_internal_base_url
    env['NEXUS_DIST_AIOMONITOR_ENABLED'] = 'false'
    env['PYTHONPATH'] = str(Path(__file__).parents[3])
    return env


def _real_model_openai_clients() -> tuple[list[str], dict[str, Any], str]:
    client_name = os.getenv('NEXUS_DIST_TEST_MODEL_CLIENT', 'openai')
    client_import = os.getenv('NEXUS_DIST_TEST_MODEL_CLIENT_IMPORT', f'nexus.extensions.openai_clients.{client_name}')
    model = os.environ['NEXUS_DIST_TEST_MODEL_NAME']
    expected = os.getenv('NEXUS_DIST_TEST_MODEL_EXPECTED', 'NEXUS_DISTRIBUTED_REAL_MODEL_OK')
    client_args: dict[str, Any] = {}
    if api_key := os.getenv('NEXUS_DIST_TEST_MODEL_API_KEY'):
        client_args['api_key'] = api_key
    if base_url := os.getenv('NEXUS_DIST_TEST_MODEL_BASE_URL'):
        client_args['base_url'] = base_url
    request_args: dict[str, Any] = {'model': model, 'temperature': 0}
    if extra_request_args := os.getenv('NEXUS_DIST_TEST_MODEL_REQUEST_ARGS_JSON'):
        request_args.update(json.loads(extra_request_args))
    return (
        [client_import],
        {
            'main': {
                'name': client_name,
                'config': {
                    'client_args': client_args,
                    'request_args': request_args,
                },
            }
        },
        expected,
    )


async def test_distributed_stack_job_and_live_paths(distributed_database_url: str, tmp_path: Path):
    await _reset_schema(distributed_database_url)

    async def fake_runtime_ping(_request):
        return web.json_response({'pong': True})

    async def fake_runtime_gui(_request):
        return web.Response(
            text='<html><body><h1>Nexus Distributed Fake Runtime</h1></body></html>',
            content_type='text/html',
        )

    fake_runtime_app = web.Application()
    fake_runtime_app.router.add_get('/ping', fake_runtime_ping)
    fake_runtime_app.router.add_get('/debug/gui', fake_runtime_gui)
    fake_runtime_client = TestClient(TestServer(fake_runtime_app))
    await fake_runtime_client.start_server()

    control_port = _free_port()
    app_port = _free_port()
    manager_port = _free_port()
    control_url = f'http://127.0.0.1:{control_port}'
    app_url = f'http://127.0.0.1:{app_port}'
    manager_url = f'http://127.0.0.1:{manager_port}'
    token = 'dist-test-token'
    env = _base_process_env(distributed_database_url, token, application_internal_base_url=app_url)
    env['NEXUS_DIST_FAKE_RUNTIME_BASE_URL'] = str(fake_runtime_client.make_url('/')).rstrip('/')
    processes = _distributed_processes(
        control_port,
        app_port,
        manager_port,
        environment_id='predefined-it',
        pool_id='interactive',
        labels={'region': 'it'},
    )

    with contextlib.ExitStack() as stack:
        for name, args in processes:
            stack.enter_context(_start_process(tmp_path, name, args, env))
        await _wait_http_ok(f'{control_url}/v1/health')
        await _wait_http_ok(f'{app_url}/v1/health')
        await _wait_http_ok(f'{manager_url}/v1/internal/health', headers={'Authorization': f'Bearer {token}'})

        async with aiohttp.ClientSession() as session:
            async with session.get(f'{control_url}/v1/console?applicationBaseUrl={app_url}') as response:
                assert response.status == 200
                console = await response.text()
                assert 'Nexus Distributed Console' in console
                assert app_url in console

            async with session.get(
                f'{control_url}/v1/environments/predefined-it/stats?pool_id=interactive'
            ) as response:
                assert response.status == 200
                stats = await response.json()
                assert stats['manager_count'] == 1
                assert stats['capacity_total'] == 4
            async with session.get(f'{control_url}/v1/managers?environment=predefined-it') as response:
                assert response.status == 200
                managers = await response.json()
                assert [item['manager_id'] for item in managers['items']] == ['im-1']

            async with session.post(
                f'{control_url}/v1/executions',
                json={
                    'execution_id': 'job-normal',
                    'mode': 'job',
                    'spec_type': 'nexus_single',
                    'instance_mode': 'asyncio',
                    'placement': {
                        'environment': 'predefined-it',
                        'pool': 'interactive',
                        'labels': {'region': 'it'},
                        'target_manager_id': 'im-1',
                    },
                    'spec': {
                        'imports': ['tests.integration.distributed.fixtures'],
                        'task_name': 'distributed_success',
                        'task_config': {'message': 'done'},
                    },
                },
            ) as response:
                assert response.status == 201

        job = await _poll_execution(control_url, 'job-normal', lambda item: item['status'] == 'succeeded')
        job_result = await _resolved_execution_result(app_url, job)
        assert job['result_json'].get('_external') is not True
        assert job_result['task_output']['message'] == 'done'

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f'{control_url}/v1/executions',
                json={
                    'execution_id': 'live-normal',
                    'mode': 'live',
                    'spec_type': 'nexus_live',
                    'instance_mode': 'asyncio',
                    'route_status': 'cached',
                    'placement': {
                        'environment': 'predefined-it',
                        'pool': 'interactive',
                        'labels': {'region': 'it'},
                    },
                    'spec': {
                        'imports': [
                            'nexus.extensions.tasks.identity',
                            'tests.integration.distributed.fixtures',
                        ],
                        'runtime_provider_name': 'distributed_fake_runtime',
                        'runtime_configs': {'main': {}},
                        'task_name': 'identity',
                        'agent_name': 'distributed_interactive_agent',
                        'agent_config': {'prompt': 'choose'},
                    },
                },
            ) as response:
                assert response.status == 201

        await _poll_execution(control_url, 'live-normal', lambda item: item['status'] == 'waiting_input')
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{control_url}/v1/executions/live-normal/activate') as response:
                assert response.status == 200

        event = await _read_sse_until(f'{app_url}/v1/executions/live-normal/live/events?cursor=0', 'input.required')
        assert event['type'] == 'input.required'

        async with aiohttp.ClientSession() as session:
            async with session.get(f'{app_url}/v1/executions/live-normal/runtimes/main/proxy/ping') as response:
                assert response.status == 200
                assert await response.json() == {'pong': True}
            async with session.get(f'{app_url}/v1/executions/live-normal/runtimes/main/proxy/debug/gui') as response:
                assert response.status == 200
                assert 'Nexus Distributed Fake Runtime' in await response.text()
            async with session.post(
                f'{app_url}/v1/executions/live-normal/commands',
                json={'type': 'input.submit', 'payload': {'text': 'Alpha'}, 'idempotency_key': 'normal-input'},
            ) as response:
                assert response.status == 202

        live = await _poll_execution(control_url, 'live-normal', lambda item: item['status'] == 'succeeded')
        live_result = await _resolved_execution_result(app_url, live)
        assert live_result['success'] is True

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f'{control_url}/v1/executions',
                json={
                    'execution_id': 'live-stop',
                    'mode': 'live',
                    'spec_type': 'nexus_live',
                    'instance_mode': 'asyncio',
                    'route_status': 'active',
                    'placement': {
                        'environment': 'predefined-it',
                        'pool': 'interactive',
                        'labels': {'region': 'it'},
                    },
                    'spec': {
                        'imports': [
                            'nexus.extensions.tasks.identity',
                            'tests.integration.distributed.fixtures',
                        ],
                        'task_name': 'identity',
                        'agent_name': 'distributed_interactive_agent',
                        'agent_config': {'prompt': 'stop'},
                    },
                },
            ) as response:
                assert response.status == 201

        await _poll_execution(control_url, 'live-stop', lambda item: item['status'] == 'waiting_input')
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{control_url}/v1/executions/live-stop/stop') as response:
                assert response.status == 200
        stopped = await _poll_execution(control_url, 'live-stop', lambda item: item['status'] == 'killed')
        assert stopped['route_status'] == 'closed'

    await fake_runtime_client.close()


async def test_distributed_stack_thread_executor_job_and_live_paths(distributed_database_url: str, tmp_path: Path):
    await _reset_schema(distributed_database_url)

    async def fake_runtime_ping(_request):
        return web.json_response({'pong': True})

    fake_runtime_app = web.Application()
    fake_runtime_app.router.add_get('/ping', fake_runtime_ping)
    fake_runtime_client = TestClient(TestServer(fake_runtime_app))
    await fake_runtime_client.start_server()

    control_port = _free_port()
    app_port = _free_port()
    manager_port = _free_port()
    control_url = f'http://127.0.0.1:{control_port}'
    app_url = f'http://127.0.0.1:{app_port}'
    manager_url = f'http://127.0.0.1:{manager_port}'
    token = 'dist-thread-token'
    env = _base_process_env(distributed_database_url, token, application_internal_base_url=app_url)
    env['NEXUS_DIST_FAKE_RUNTIME_BASE_URL'] = str(fake_runtime_client.make_url('/')).rstrip('/')
    processes = _distributed_processes(control_port, app_port, manager_port, executor='thread')

    try:
        with contextlib.ExitStack() as stack:
            for name, args in processes:
                stack.enter_context(_start_process(tmp_path, name, args, env))
            await _wait_http_ok(f'{control_url}/v1/health')
            await _wait_http_ok(f'{app_url}/v1/health')
            await _wait_http_ok(f'{manager_url}/v1/internal/health', headers={'Authorization': f'Bearer {token}'})

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'{control_url}/v1/executions',
                    json={
                        'execution_id': 'job-thread',
                        'mode': 'job',
                        'spec_type': 'nexus_single',
                        'instance_mode': 'thread',
                        'spec': {
                            'imports': ['tests.integration.distributed.fixtures'],
                            'task_name': 'distributed_success',
                            'task_config': {'message': 'thread-done'},
                        },
                    },
                ) as response:
                    assert response.status == 201

            job = await _poll_execution(control_url, 'job-thread', lambda item: item['status'] == 'succeeded')
            job_result = await _resolved_execution_result(app_url, job)
            assert job['result_json'].get('_external') is not True
            assert job_result['task_output']['message'] == 'thread-done'

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'{control_url}/v1/executions',
                    json={
                        'execution_id': 'live-thread',
                        'mode': 'live',
                        'spec_type': 'nexus_live',
                        'instance_mode': 'thread',
                        'route_status': 'cached',
                        'spec': {
                            'imports': [
                                'nexus.extensions.tasks.identity',
                                'tests.integration.distributed.fixtures',
                            ],
                            'runtime_provider_name': 'distributed_fake_runtime',
                            'runtime_configs': {'main': {}},
                            'task_name': 'identity',
                            'agent_name': 'distributed_interactive_agent',
                            'agent_config': {'prompt': 'choose'},
                        },
                    },
                ) as response:
                    assert response.status == 201

            await _poll_execution(control_url, 'live-thread', lambda item: item['status'] == 'waiting_input')
            async with aiohttp.ClientSession() as session:
                async with session.post(f'{control_url}/v1/executions/live-thread/activate') as response:
                    assert response.status == 200

            event = await _read_sse_until(f'{app_url}/v1/executions/live-thread/live/events?cursor=0', 'input.required')
            assert event['type'] == 'input.required'

            async with aiohttp.ClientSession() as session:
                async with session.get(f'{app_url}/v1/executions/live-thread/runtimes/main/proxy/ping') as response:
                    assert response.status == 200
                    assert await response.json() == {'pong': True}
                async with session.post(
                    f'{app_url}/v1/executions/live-thread/commands',
                    json={'type': 'input.submit', 'payload': {'text': 'Alpha'}, 'idempotency_key': 'thread-input'},
                ) as response:
                    assert response.status == 202

            live = await _poll_execution(control_url, 'live-thread', lambda item: item['status'] == 'succeeded')
            live_result = await _resolved_execution_result(app_url, live)
            assert live_result['success'] is True
    finally:
        await fake_runtime_client.close()


async def test_environment_limit_update_waits_for_reservation_usage_lock_on_mysql(
    distributed_database_url: str,
):
    if 'mysql' not in distributed_database_url:
        pytest.skip('resource usage locking is validated against MySQL/InnoDB')
    await _reset_schema(distributed_database_url)
    engine = create_async_engine(distributed_database_url, pool_size=4, max_overflow=0)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    allocation_id = await _seed_resource_usage_lock_fixture(session_factory)
    update_started = asyncio.Event()

    async def lower_limit() -> None:
        async with session_factory() as session:
            update_started.set()
            try:
                await EnvironmentRepository(session).update_environment(
                    'resource-environment',
                    resource_limits={'manager_count': 1},
                    actor='admin',
                    now=utcnow(),
                )
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    update_task: asyncio.Task[None] | None = None
    try:
        async with session_factory() as reservation_session:
            await QuotaRepository(reservation_session, default_limits={}).reserve_allocation(
                allocation_id,
                now=utcnow(),
            )
            update_task = asyncio.create_task(lower_limit())
            await asyncio.wait_for(update_started.wait(), timeout=1)
            await asyncio.sleep(0.2)
            assert not update_task.done()
            await reservation_session.commit()

        with pytest.raises(DomainConflictError, match='environment_manager_count_limit_below_usage'):
            await asyncio.wait_for(update_task, timeout=5)
    finally:
        if update_task is not None and not update_task.done():
            update_task.cancel()
            await asyncio.gather(update_task, return_exceptions=True)
        await engine.dispose()


async def test_artifact_release_receiving_cannot_overwrite_maintenance_fence_on_mysql(
    distributed_database_url: str,
    tmp_path: Path,
):
    if 'mysql' not in distributed_database_url:
        pytest.skip('artifact upload fencing is validated against MySQL/InnoDB')
    await _reset_schema(distributed_database_url)
    engine = create_async_engine(distributed_database_url, pool_size=2, max_overflow=0)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    store = LocalArtifactStore(tmp_path / 'artifact-store')
    claimed_at = utcnow() - timedelta(minutes=2)

    try:
        async with session_factory() as stale_session:
            stale_repo = ArtifactRepository(stale_session)
            upload, _ = await stale_repo.create_upload(
                CreateArtifactUpload(
                    owner_principal_id='artifact-owner',
                    environment_id='artifact-environment',
                    owner_tenant=None,
                    created_by='artifact-owner',
                    kind='source_package',
                    sha256='a' * 64,
                    size_bytes=1,
                    idempotency_scope='artifact-fencing',
                    idempotency_key='upload',
                    session_ttl=timedelta(hours=1),
                    artifact_ttl=timedelta(days=1),
                ),
                store=store,
                now=claimed_at,
            )
            _, stale_operation_id = await stale_repo.claim_receiving(
                upload.upload_id,
                owner_principal_id='artifact-owner',
                environment_id='artifact-environment',
                operation_owner='stale-uploader',
                lease_ttl=timedelta(minutes=1),
                now=claimed_at,
            )
            assert stale_operation_id is not None
            await stale_session.commit()
            stale_row = await stale_session.get(ArtifactUploadRow, upload.upload_id)
            assert stale_row is not None
            assert stale_row.operation_id == stale_operation_id

            async with session_factory() as maintenance_session:
                claims = await ArtifactRepository(maintenance_session).claim_upload_maintenance(
                    operation_owner='maintenance',
                    lease_ttl=timedelta(minutes=5),
                    limit=1,
                    now=utcnow(),
                )
                assert len(claims) == 1
                maintenance_operation_id = claims[0].operation_id
                await maintenance_session.commit()

            released = await stale_repo.release_receiving(
                upload.upload_id,
                operation_id=stale_operation_id,
                failure_code='stale_uploader_failed',
                now=utcnow(),
            )
            await stale_session.commit()
            assert released is False

        async with session_factory() as verify_session:
            current = await verify_session.get(ArtifactUploadRow, upload.upload_id)
            assert current is not None
            assert current.status == 'maintenance'
            assert current.operation_id == maintenance_operation_id
            assert current.failure_count == 0
    finally:
        await store.close()
        await engine.dispose()


async def test_resource_usage_migration_backfills_mysql_without_overwriting(
    distributed_database_url: str,
):
    if 'mysql' not in distributed_database_url:
        pytest.skip('resource usage migration is validated against MySQL/InnoDB')
    await _reset_schema(distributed_database_url)
    engine = create_async_engine(distributed_database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    await _seed_resource_usage_lock_fixture(session_factory)
    migration = importlib.import_module(
        'nexus.distributed.persistence.migrations.versions.0018_precreate_resource_usage'
    )
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    'DELETE FROM distributed_environment_resource_usage WHERE scope_key = '
                    "'grant:resource-environment:resource-owner'"
                )
            )
            await connection.execute(
                text(
                    'UPDATE distributed_environment_resource_usage SET manager_count = 3 '
                    "WHERE scope_key = 'environment:resource-environment'"
                )
            )
            await connection.run_sync(
                lambda sync_connection: migration.upgrade(Operations(MigrationContext.configure(sync_connection)))
            )
        async with engine.connect() as connection:
            rows = {
                row.scope_key: row.manager_count
                for row in (
                    await connection.execute(
                        text(
                            'SELECT scope_key, manager_count '
                            'FROM distributed_environment_resource_usage ORDER BY scope_key'
                        )
                    )
                )
            }
    finally:
        await engine.dispose()

    assert rows == {
        'environment:resource-environment': 3,
        'grant:resource-environment:resource-owner': 0,
    }


async def test_offline_upgrade_from_0003_clears_legacy_managers_on_mysql(
    distributed_database_url: str,
):
    if 'mysql' not in distributed_database_url:
        pytest.skip('offline migration is validated against MySQL/InnoDB')
    await _reset_schema(distributed_database_url)
    engine = create_async_engine(distributed_database_url)

    def migrate_to_0003(sync_connection) -> None:
        for migration in MIGRATIONS[:3]:
            module = importlib.import_module(migration.module)
            module.upgrade(Operations(MigrationContext.configure(sync_connection)))
            distributed_db._set_alembic_revision(sync_connection, migration.revision)

    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
            await connection.execute(text('DROP TABLE IF EXISTS alembic_version'))
            await connection.execute(text('DROP TABLE IF EXISTS schema_version'))
            await connection.run_sync(migrate_to_0003)
            await connection.execute(
                text(
                    """
                    INSERT INTO instance_managers (
                        manager_id, session_id, endpoint_url,
                        environment_type, environment_name, pool_name,
                        labels_json, draining,
                        supported_modes_json, supported_spec_types_json,
                        supported_instance_modes_json,
                        capacity_total, observed_capacity_used,
                        local_instances_json, version, metadata_json,
                        last_heartbeat_at, created_at, updated_at
                    ) VALUES (
                        'legacy-manager', 'legacy-session', 'http://127.0.0.1:9001',
                        'predefined', 'legacy-environment', 'legacy-pool',
                        '{}', 0, '[\"job\"]', '[\"nexus_single\"]', '[\"asyncio\"]',
                        1, 0, NULL, 'legacy', '{}',
                        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                    )
                    """
                )
            )
    finally:
        await engine.dispose()

    await upgrade_database(DistributedConfig(database_url=distributed_database_url))

    engine = create_async_engine(distributed_database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.connect() as connection:
            legacy_manager_count = await connection.scalar(text('SELECT COUNT(*) FROM instance_managers'))
            columns = await connection.run_sync(
                lambda sync_connection: {
                    column['name']: column for column in inspect(sync_connection).get_columns('instance_managers')
                }
            )
            revision = await connection.scalar(text('SELECT version_num FROM alembic_version'))

        async with session_factory() as session:
            registered = await ManagerRepository(session).register_session(
                manager_id='current-manager',
                session_id='current-session',
                endpoint_url='http://127.0.0.1:9002',
                supported_modes=['job'],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                capacity_total=1,
                force_replace=False,
                now=utcnow(),
            )
            await session.commit()
    finally:
        await engine.dispose()

    assert legacy_manager_count == 0
    assert revision == HEAD_REVISION
    assert registered is True
    assert {'environment_id', 'pool_id', 'workload'} <= columns.keys()
    assert all(columns[name]['nullable'] is False for name in ('environment_id', 'pool_id', 'workload'))
    assert {
        'environment_type',
        'environment_name',
        'pool_name',
        'environment_key',
        'workload_class',
    }.isdisjoint(columns)


async def test_allocation_reservation_waits_for_environment_limit_update_on_mysql(
    distributed_database_url: str,
):
    if 'mysql' not in distributed_database_url:
        pytest.skip('resource usage locking is validated against MySQL/InnoDB')
    await _reset_schema(distributed_database_url)
    engine = create_async_engine(distributed_database_url, pool_size=4, max_overflow=0)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    allocation_id = await _seed_resource_usage_lock_fixture(session_factory)
    reservation_started = asyncio.Event()

    async def reserve() -> None:
        async with session_factory() as session:
            reservation_started.set()
            try:
                await QuotaRepository(session, default_limits={}).reserve_allocation(
                    allocation_id,
                    now=utcnow(),
                )
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    reservation_task: asyncio.Task[None] | None = None
    try:
        async with session_factory() as admin_session:
            await EnvironmentRepository(admin_session).update_environment(
                'resource-environment',
                resource_limits={'manager_count': 1},
                actor='admin',
                now=utcnow(),
            )
            reservation_task = asyncio.create_task(reserve())
            await asyncio.wait_for(reservation_started.wait(), timeout=1)
            await asyncio.sleep(0.2)
            assert not reservation_task.done()
            await admin_session.commit()

        with pytest.raises(QuotaExceededError, match='environment_manager_count_limit_exceeded'):
            await asyncio.wait_for(reservation_task, timeout=5)
    finally:
        if reservation_task is not None and not reservation_task.done():
            reservation_task.cancel()
            await asyncio.gather(reservation_task, return_exceptions=True)
        await engine.dispose()


async def test_scheduler_concurrent_claim_creates_one_assignment_on_mysql(distributed_database_url: str):
    if 'mysql' not in distributed_database_url:
        pytest.skip('concurrent claim locking is validated against MySQL/InnoDB')
    await _reset_schema(distributed_database_url)
    engine = create_async_engine(distributed_database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with session_factory() as session:
            manager = ManagerRepository(session)
            await manager.register_session(
                manager_id='im-concurrent',
                session_id='session-1',
                endpoint_url='http://127.0.0.1:9',
                supported_modes=[ExecutionMode.JOB],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                capacity_total=10,
                force_replace=False,
                now=utcnow(),
            )
            repo = ExecutionRepository(session, shard_count=8)
            execution = await repo.create_execution(
                CreateExecution(
                    execution_id='concurrent-claim',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                ),
                now=utcnow(),
            )
            await session.commit()

        claim_now = utcnow() + timedelta(seconds=1)

        async def claim_once():
            async with session_factory() as session:
                scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
                assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=claim_now)
                await session.commit()
                return assignment

        results = await asyncio.gather(claim_once(), claim_once())
        claimed = [assignment for assignment in results if assignment is not None]
        assert len(claimed) == 1
    finally:
        await engine.dispose()


async def test_concurrent_typed_submission_creates_one_complete_aggregate_on_mysql(
    distributed_database_url: str,
):
    if 'mysql' not in distributed_database_url:
        pytest.skip('submission aggregate concurrency is validated against MySQL/InnoDB')
    await _reset_schema(distributed_database_url)
    engine = create_async_engine(distributed_database_url, pool_size=5, max_overflow=0)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    now = utcnow()
    config = DistributedConfig(
        database_url=distributed_database_url,
        environment_enabled_providers=['kubernetes'],
        environment_base_image=f'registry.example/workload@sha256:{"d" * 64}',
        environment_builder_image=f'registry.example/builder@sha256:{"e" * 64}',
        default_user_quota={
            'queued_executions': 10,
            'running_executions': 10,
            'reserved_slots': 10,
        },
    )
    try:
        async with session_factory() as session:
            await ProviderConfigRepository(session).create(
                provider_config_ref='kubernetes/default',
                config={
                    'provider_type': 'kubernetes',
                    'namespace': 'default',
                    'auth_type': 'in_cluster',
                },
                status='enabled',
                actor='admin',
                now=now,
            )
            await AccessRepository(session).upsert_user(
                principal_id='concurrent-owner',
                issuer='tai',
                tenant='tenant',
                subject='concurrent-owner',
                roles={'user'},
                display_name=None,
                actor='admin',
                now=now,
            )
            environments = EnvironmentRepository(session)
            await environments.create_environment(
                environment_id='concurrent-environment',
                name='concurrent-environment',
                allocation_type='dynamic',
                provider_type='kubernetes',
                provider_config_ref='kubernetes/default',
                resource_limits={
                    'manager_count': 2,
                    'cpu_millicores': 4_000,
                    'memory_bytes': 4 * 1024**3,
                    'slots': 2,
                },
                default_timeouts={},
                status='enabled',
                actor='admin',
                now=now,
            )
            await environments.upsert_grant(
                principal_id='concurrent-owner',
                environment_id='concurrent-environment',
                limits={
                    'manager_count': 2,
                    'cpu_millicores': 4_000,
                    'memory_bytes': 4 * 1024**3,
                    'slots': 2,
                },
                enabled=True,
                actor='admin',
                now=now,
            )
            for artifact_id, kind, marker, size_bytes, filename in (
                ('concurrent-source', 'source_package', 'a', 101, 'source.tar.zst'),
                ('concurrent-wheel', 'nexus_wheel', 'b', 102, 'nexus-agent.whl'),
                ('concurrent-lock', 'uv_lock', 'c', 103, 'uv.lock'),
            ):
                session.add(
                    ArtifactRow(
                        artifact_id=artifact_id,
                        owner_principal_id='concurrent-owner',
                        environment_id='concurrent-environment',
                        owner_tenant='tenant',
                        kind=kind,
                        producer_key=f'test:{kind}',
                        uri=f'file:///tmp/{filename}',
                        sha256=marker * 64,
                        size_bytes=size_bytes,
                        provider_etag=None,
                        status='ready',
                        created_by='concurrent-owner',
                        expires_at=None,
                        created_at=now,
                        updated_at=now,
                    )
                )
            await session.commit()

        submission = SingleSubmission.model_validate(
            {
                'idempotency_key': 'concurrent-submission',
                'environment_id': 'concurrent-environment',
                'code': {
                    'mode': 'source_package',
                    'source_package': {
                        'artifact_id': 'concurrent-source',
                        'kind': 'source_package',
                        'sha256': 'a' * 64,
                        'size_bytes': 101,
                    },
                    'nexus_wheel': {
                        'artifact_id': 'concurrent-wheel',
                        'kind': 'nexus_wheel',
                        'sha256': 'b' * 64,
                        'size_bytes': 102,
                    },
                    'uv_lock': {
                        'artifact_id': 'concurrent-lock',
                        'kind': 'uv_lock',
                        'sha256': 'c' * 64,
                        'size_bytes': 103,
                    },
                },
                'resources': {
                    'managers': {'count': 1, 'cpu_cores': 1, 'memory_gib': 1, 'slots': 1},
                },
                'run_input': {'task_name': 'identity'},
            }
        )
        start = asyncio.Event()

        async def submit_once():
            await start.wait()

            async def create(session):
                result = await SubmissionRepository(session, config=config).create(
                    submission,
                    actor_principal_id='concurrent-owner',
                    owner_principal_id='concurrent-owner',
                    owner_tenant='tenant',
                    now=utcnow(),
                )
                if not result.reused:
                    await AuditRepository(session).append(
                        actor_principal_id='concurrent-owner',
                        target_owner_principal_id='concurrent-owner',
                        operation='submission.single.create',
                        resource_type='execution',
                        resource_id=result.execution.execution_id,
                        now=utcnow(),
                    )
                return result

            return await run_idempotent_transaction(
                session_factory,
                create,
                conflict_markers=('uq_executions_idempotency', 'executions.idempotency_scope'),
            )

        tasks = [asyncio.create_task(submit_once()) for _ in range(2)]
        start.set()
        results = await asyncio.gather(*tasks)

        assert len({result.execution.execution_id for result in results}) == 1
        assert sorted(result.reused for result in results) == [False, True]
        async with session_factory() as session:
            assert await session.scalar(select(func.count(ExecutionRow.execution_id))) == 1
            assert await session.scalar(select(func.count(DependencyBuildRow.dependency_build_id))) == 1
            assert await session.scalar(select(func.count(EnvironmentAllocationRow.allocation_id))) == 1
            assert await session.scalar(select(func.count(ArtifactRefRow.artifact_id))) == 9
            assert await session.scalar(select(func.count(AuditRow.audit_id))) == 1
            owner_usage = await session.get(QuotaUsageRow, ('concurrent-owner', '*', '*', 'owner'))
            environment_usage = await session.get(
                EnvironmentResourceUsageRow,
                'environment:concurrent-environment',
            )
            grant_usage = await session.get(
                EnvironmentResourceUsageRow,
                'grant:concurrent-environment:concurrent-owner',
            )
            assert owner_usage is not None
            assert owner_usage.queued_executions == 1
            assert environment_usage is not None
            assert grant_usage is not None
            assert environment_usage.manager_count == grant_usage.manager_count == 1
            ref_counts = dict(
                (
                    await session.execute(
                        select(ArtifactRefRow.resource_type, func.count())
                        .group_by(ArtifactRefRow.resource_type)
                        .order_by(ArtifactRefRow.resource_type)
                    )
                ).all()
            )
            assert ref_counts == {'allocation': 3, 'dependency_build': 3, 'execution': 3}
    finally:
        await engine.dispose()


async def test_scheduler_concurrent_claim_respects_manager_capacity_on_mysql(distributed_database_url: str):
    if 'mysql' not in distributed_database_url:
        pytest.skip('concurrent capacity locking is validated against MySQL/InnoDB')
    await _reset_schema(distributed_database_url)
    engine = create_async_engine(distributed_database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with session_factory() as session:
            manager = ManagerRepository(session)
            await manager.register_session(
                manager_id='im-capacity',
                session_id='session-1',
                endpoint_url='http://127.0.0.1:9',
                supported_modes=[ExecutionMode.JOB],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                capacity_total=1,
                force_replace=False,
                now=utcnow(),
            )
            repo = ExecutionRepository(session, shard_count=8)
            first = await repo.create_execution(
                CreateExecution(
                    execution_id='capacity-claim-1',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                ),
                now=utcnow(),
            )
            second = await repo.create_execution(
                CreateExecution(
                    execution_id='capacity-claim-2',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                ),
                now=utcnow(),
            )
            await session.commit()

        claim_now = utcnow() + timedelta(seconds=1)

        async def claim_once(shard_id: int):
            async with session_factory() as session:
                scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
                assignment = await scheduler.claim_and_assign(shard_ids=[shard_id], now=claim_now)
                await session.commit()
                return assignment

        results = await asyncio.gather(claim_once(first.shard_id), claim_once(second.shard_id))
        claimed = [assignment for assignment in results if assignment is not None]
        assert len(claimed) == 1
    finally:
        await engine.dispose()


async def test_scheduler_concurrent_expired_shard_lease_claim_has_single_owner_on_mysql(distributed_database_url: str):
    if 'mysql' not in distributed_database_url:
        pytest.skip('concurrent shard lease CAS is validated against MySQL/InnoDB')
    await _reset_schema(distributed_database_url)
    engine = create_async_engine(distributed_database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        expired_now = utcnow()
        async with session_factory() as session:
            scheduler = SchedulerRepository(session, shard_count=4, manager_healthy_ttl=timedelta(seconds=15))
            owned = await scheduler.renew_shard_leases(
                owner_id='expired-owner',
                shard_group='default',
                shard_ids=[0],
                lease_ttl=timedelta(seconds=1),
                now=expired_now,
            )
            await session.commit()
        assert owned == [0]

        claim_now = expired_now + timedelta(seconds=5)

        async def claim_once(owner_id: str) -> list[int]:
            async with session_factory() as session:
                scheduler = SchedulerRepository(session, shard_count=4, manager_healthy_ttl=timedelta(seconds=15))
                owned = await scheduler.renew_shard_leases(
                    owner_id=owner_id,
                    shard_group='default',
                    shard_ids=[0],
                    lease_ttl=timedelta(seconds=20),
                    now=claim_now,
                )
                await session.commit()
                return owned

        results = await asyncio.gather(claim_once('scheduler-a'), claim_once('scheduler-b'))
        assert sorted(len(result) for result in results) == [0, 1]
    finally:
        await engine.dispose()


async def test_scheduler_assignment_skips_lease_locked_by_heartbeat_on_mysql(distributed_database_url: str):
    if 'mysql' not in distributed_database_url:
        pytest.skip('scheduler heartbeat and assignment lease locking is validated against MySQL/InnoDB')
    await _reset_schema(distributed_database_url)
    engine = create_async_engine(distributed_database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    owner_id = 'scheduler-heartbeat-owner'
    try:
        async with session_factory() as session:
            now = utcnow()
            claim_now = now + timedelta(seconds=1)
            await ManagerRepository(session).register_session(
                manager_id='im-heartbeat-fence',
                session_id='session-1',
                endpoint_url='http://127.0.0.1:9',
                supported_modes=[ExecutionMode.JOB],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                capacity_total=1,
                force_replace=False,
                now=now,
            )
            execution = await ExecutionRepository(session, shard_count=8).create_execution(
                CreateExecution(
                    execution_id='heartbeat-fence-execution',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                ),
                now=now,
            )
            session.add(
                ShardLeaseRow(
                    shard_group='default',
                    shard_id=execution.shard_id,
                    shard_count=8,
                    owner_id=owner_id,
                    lease_expires_at=now + timedelta(minutes=1),
                    updated_at=now,
                )
            )
            await session.commit()

        async with session_factory() as heartbeat_session:
            heartbeat = SchedulerRepository(
                heartbeat_session,
                shard_count=8,
                manager_healthy_ttl=timedelta(minutes=5),
            )
            assert await heartbeat.renew_shard_leases(
                owner_id=owner_id,
                shard_group='default',
                shard_ids=[execution.shard_id],
                lease_ttl=timedelta(minutes=1),
                now=claim_now,
            ) == [execution.shard_id]

            async with session_factory() as assignment_session:
                scheduler = SchedulerRepository(
                    assignment_session,
                    shard_count=8,
                    manager_healthy_ttl=timedelta(minutes=5),
                    lease_owner_id=owner_id,
                )
                assignments = await asyncio.wait_for(
                    scheduler.claim_and_assign_batch(
                        shard_ids=[execution.shard_id],
                        now=claim_now,
                    ),
                    timeout=2,
                )
                assert assignments == []
                assert scheduler.last_claim_stats.scanned == 1
                assert scheduler.last_claim_stats.claimed == 1
                assert scheduler.last_claim_stats.candidate_queries == 2
                assert scheduler.lease_fence_lost is False
                assert await scheduler.revalidate_shard_lease_fence(now=claim_now) is True
                await assignment_session.commit()

            await heartbeat_session.commit()

        async with session_factory() as session:
            lease = await session.get(ShardLeaseRow, ('default', execution.shard_id))
            assert lease is not None
            assert lease.owner_id == owner_id
            assert lease.lease_expires_at > utcnow()
    finally:
        await engine.dispose()


async def test_batch_item_ids_use_canonical_validation_and_binary_mysql_pagination(
    distributed_database_url: str,
):
    if 'mysql' not in distributed_database_url:
        pytest.skip('batch item collation is validated against MySQL/InnoDB')
    await _reset_schema(distributed_database_url)
    engine = create_async_engine(distributed_database_url)
    parent_execution_id = 'batch-collation'
    canonical_ids = ['a', 'a-', 'a.', 'a0', 'a_', 'aa', 'z']
    now = utcnow()
    try:
        assert all(is_valid_batch_item_id(item_id) for item_id in canonical_ids)
        assert is_valid_batch_item_id('a' * 64)
        assert not is_valid_batch_item_id('a' * 65)
        assert not is_valid_batch_item_id('A')
        assert not is_valid_batch_item_id('a ')
        assert not is_valid_batch_item_id('café')
        assert not is_valid_batch_item_id('cafe\u0301')

        async with engine.begin() as connection:
            columns = (
                await connection.execute(
                    text(
                        """
                        SELECT TABLE_NAME, COLUMN_NAME, COLLATION_NAME
                        FROM information_schema.COLUMNS
                        WHERE TABLE_SCHEMA = DATABASE()
                          AND (
                            (TABLE_NAME = 'distributed_batch_items' AND COLUMN_NAME = 'item_id')
                            OR (TABLE_NAME = 'distributed_batch_item_attempts' AND COLUMN_NAME = 'item_id')
                            OR (TABLE_NAME = 'distributed_batch_runs' AND COLUMN_NAME IN ('stop_cursor', 'child_sync_cursor'))
                          )
                        """
                    )
                )
            ).mappings()
            assert {
                (str(row['TABLE_NAME']), str(row['COLUMN_NAME'])): str(row['COLLATION_NAME']) for row in columns
            } == {
                ('distributed_batch_item_attempts', 'item_id'): 'utf8mb4_bin',
                ('distributed_batch_items', 'item_id'): 'utf8mb4_bin',
                ('distributed_batch_runs', 'child_sync_cursor'): 'utf8mb4_bin',
                ('distributed_batch_runs', 'stop_cursor'): 'utf8mb4_bin',
            }
            case_sensitive = await connection.scalar(text("SELECT _utf8mb4'A' = _utf8mb4'a' COLLATE utf8mb4_bin"))
            assert case_sensitive == 0
            await connection.execute(
                insert(BatchItemRow),
                [
                    {
                        'parent_execution_id': parent_execution_id,
                        'item_id': item_id,
                        'task_ordinal': ordinal,
                        'task_offset': ordinal,
                        'task_length': 1,
                        'current_attempt_no': 0,
                        'current_child_execution_id': None,
                        'spec_sha256': 'a' * 64,
                        'status': 'pending',
                        'status_version': 1,
                        'child_execution_status': None,
                        'child_execution_version': None,
                        'metadata_json': {},
                        'created_at': now,
                        'updated_at': now,
                    }
                    for ordinal, item_id in enumerate(reversed(canonical_ids))
                ],
            )

        async with engine.connect() as connection:
            page = (
                await connection.execute(
                    select(BatchItemRow.item_id)
                    .where(
                        BatchItemRow.parent_execution_id == parent_execution_id,
                        BatchItemRow.item_id > 'a.',
                    )
                    .order_by(BatchItemRow.item_id)
                )
            ).scalars()
            assert list(page) == ['a0', 'a_', 'aa', 'z']
    finally:
        await engine.dispose()


async def test_batch_pending_hot_path_uses_covering_index_with_one_hundred_thousand_items(
    distributed_database_url: str,
):
    if 'mysql' not in distributed_database_url:
        pytest.skip('batch hot-path query plan is validated against MySQL/InnoDB')
    await _reset_schema(distributed_database_url)
    engine = create_async_engine(distributed_database_url)
    parent_execution_id = 'batch-hot-path-100k'
    item_count = 100_000
    insert_chunk_size = 5_000
    now = utcnow()
    try:
        async with engine.begin() as connection:
            for start in range(0, item_count, insert_chunk_size):
                await connection.execute(
                    insert(BatchItemRow),
                    [
                        {
                            'parent_execution_id': parent_execution_id,
                            'item_id': f'item-{index:012d}',
                            'task_ordinal': index,
                            'task_offset': index,
                            'task_length': 1,
                            'current_attempt_no': 0,
                            'current_child_execution_id': None,
                            'spec_sha256': 'a' * 64,
                            'status': 'pending' if index % 100 == 0 else 'succeeded',
                            'status_version': 1,
                            'child_execution_status': None,
                            'child_execution_version': None,
                            'metadata_json': {},
                            'created_at': now,
                            'updated_at': now,
                        }
                        for index in range(start, min(start + insert_chunk_size, item_count))
                    ],
                )

        async with engine.connect() as connection:
            assert (
                await connection.scalar(
                    select(func.count(BatchItemRow.item_id)).where(
                        BatchItemRow.parent_execution_id == parent_execution_id
                    )
                )
                == item_count
            )
            assert (
                await connection.scalar(
                    select(func.count(BatchItemRow.item_id)).where(
                        BatchItemRow.parent_execution_id == parent_execution_id,
                        BatchItemRow.status == 'pending',
                    )
                )
                == item_count // 100
            )
            plan = (
                (
                    await connection.execute(
                        text(
                            """
                        EXPLAIN
                        SELECT item_id, task_offset, task_length
                        FROM distributed_batch_items
                        WHERE parent_execution_id = :parent_execution_id
                          AND status = 'pending'
                        ORDER BY task_ordinal
                        LIMIT 500
                        """
                        ),
                        {'parent_execution_id': parent_execution_id},
                    )
                )
                .mappings()
                .one()
            )
            assert plan['key'] == 'idx_batch_items_parent_status_ordinal'
            assert 'Using filesort' not in str(plan['Extra'])

            page = (
                await connection.execute(
                    select(BatchItemRow.item_id)
                    .where(
                        BatchItemRow.parent_execution_id == parent_execution_id,
                        BatchItemRow.status == 'pending',
                    )
                    .order_by(BatchItemRow.task_ordinal)
                    .limit(500)
                )
            ).scalars()
            item_ids = list(page)
            assert item_ids[0] == 'item-000000000000'
            assert item_ids[-1] == 'item-000000049900'
    finally:
        await engine.dispose()


async def test_distributed_live_sends_http_request_to_openai_compatible_model_service(
    distributed_database_url: str,
    tmp_path: Path,
):
    await _reset_schema(distributed_database_url)
    expected = 'NEXUS_DISTRIBUTED_LOCAL_MODEL_OK'
    model_requests: list[dict[str, Any]] = []

    async def chat_completions(request):
        payload = await request.json()
        model_requests.append(payload)
        return web.json_response(
            {
                'id': 'chatcmpl-dist-test',
                'object': 'chat.completion',
                'created': 0,
                'model': payload.get('model', 'dist-test-model'),
                'choices': [
                    {
                        'index': 0,
                        'message': {'role': 'assistant', 'content': expected},
                        'finish_reason': 'stop',
                    }
                ],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
            }
        )

    model_app = web.Application()
    model_app.router.add_post('/chat/completions', chat_completions)
    model_app.router.add_post('/v1/chat/completions', chat_completions)
    model_client = TestClient(TestServer(model_app))
    await model_client.start_server()

    control_port = _free_port()
    app_port = _free_port()
    manager_port = _free_port()
    control_url = f'http://127.0.0.1:{control_port}'
    app_url = f'http://127.0.0.1:{app_port}'
    manager_url = f'http://127.0.0.1:{manager_port}'
    token = 'dist-local-model-token'
    env = _base_process_env(distributed_database_url, token, application_internal_base_url=app_url)
    processes = _distributed_processes(control_port, app_port, manager_port)

    try:
        with contextlib.ExitStack() as stack:
            for name, args in processes:
                stack.enter_context(_start_process(tmp_path, name, args, env))
            await _wait_http_ok(f'{control_url}/v1/health')
            await _wait_http_ok(f'{app_url}/v1/health')
            await _wait_http_ok(f'{manager_url}/v1/internal/health', headers={'Authorization': f'Bearer {token}'})

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'{control_url}/v1/executions',
                    json={
                        'execution_id': 'live-local-model',
                        'mode': 'live',
                        'spec_type': 'nexus_live',
                        'instance_mode': 'asyncio',
                        'route_status': 'active',
                        'spec': {
                            'imports': [
                                'nexus.extensions.tasks.identity',
                                'nexus.extensions.agents.single_turn',
                                'nexus.extensions.openai_clients.openai',
                            ],
                            'task_name': 'identity',
                            'agent_name': 'single_turn',
                            'agent_config': {
                                'prompt': [
                                    {
                                        'role': 'user',
                                        'content': f'Reply with exactly: {expected}',
                                    }
                                ]
                            },
                            'openai_clients': {
                                'main': {
                                    'name': 'openai',
                                    'config': {
                                        'client_args': {
                                            'api_key': 'test-key',
                                            'base_url': str(model_client.make_url('/')).rstrip('/'),
                                            'max_retries': 0,
                                        },
                                        'request_args': {'model': 'dist-test-model', 'temperature': 0},
                                    },
                                }
                            },
                        },
                    },
                ) as response:
                    assert response.status == 201

            live = await _poll_execution(control_url, 'live-local-model', lambda item: item['status'] == 'succeeded')
            live_result = await _resolved_execution_result(app_url, live)
            assert live_result['success'] is True
            assert live_result['agent_output']['response'] == expected
            assert len(model_requests) == 1
            assert model_requests[0]['model'] == 'dist-test-model'
            assert model_requests[0]['messages'][0]['content'] == f'Reply with exactly: {expected}'
    finally:
        await model_client.close()


@pytest.mark.skipif(
    os.getenv('NEXUS_DIST_TEST_REAL_MODEL') != '1',
    reason='set NEXUS_DIST_TEST_REAL_MODEL=1 plus model env vars to run real model live verification',
)
async def test_distributed_live_single_turn_real_model(distributed_database_url: str, tmp_path: Path):
    missing = [name for name in ('NEXUS_DIST_TEST_MODEL_NAME',) if not os.getenv(name)]
    if missing:
        pytest.skip(f'missing required real-model env vars: {", ".join(missing)}')

    await _reset_schema(distributed_database_url)
    control_port = _free_port()
    app_port = _free_port()
    manager_port = _free_port()
    control_url = f'http://127.0.0.1:{control_port}'
    app_url = f'http://127.0.0.1:{app_port}'
    manager_url = f'http://127.0.0.1:{manager_port}'
    token = 'dist-real-model-token'
    env = _base_process_env(distributed_database_url, token, application_internal_base_url=app_url)
    processes = _distributed_processes(control_port, app_port, manager_port)
    client_imports, openai_clients, expected = _real_model_openai_clients()
    prompt = os.getenv(
        'NEXUS_DIST_TEST_MODEL_PROMPT',
        f'Reply with exactly this token and no extra text: {expected}',
    )

    with contextlib.ExitStack() as stack:
        for name, args in processes:
            stack.enter_context(_start_process(tmp_path, name, args, env))
        await _wait_http_ok(f'{control_url}/v1/health')
        await _wait_http_ok(f'{app_url}/v1/health')
        await _wait_http_ok(f'{manager_url}/v1/internal/health', headers={'Authorization': f'Bearer {token}'})

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f'{control_url}/v1/executions',
                json={
                    'execution_id': 'live-real-model',
                    'mode': 'live',
                    'spec_type': 'nexus_live',
                    'instance_mode': 'asyncio',
                    'route_status': 'active',
                    'max_attempts': 1,
                    'spec': {
                        'imports': [
                            'nexus.extensions.tasks.identity',
                            'nexus.extensions.agents.single_turn',
                            *client_imports,
                        ],
                        'task_name': 'identity',
                        'agent_name': 'single_turn',
                        'agent_config': {'prompt': [{'role': 'user', 'content': prompt}]},
                        'openai_clients': openai_clients,
                    },
                },
            ) as response:
                assert response.status == 201

        live = await _poll_execution(
            control_url,
            'live-real-model',
            lambda item: item['status'] in {'succeeded', 'failed'},
            timeout=float(os.getenv('NEXUS_DIST_TEST_REAL_MODEL_TIMEOUT', '90')),
        )
        assert live['status'] == 'succeeded', live.get('error_json')
        live_result = await _resolved_execution_result(app_url, live)
        assert live_result['success'] is True
        response_text = live_result['agent_output']['response']
        assert response_text
        if os.getenv('NEXUS_DIST_TEST_MODEL_ASSERT_EXPECTED', '1') == '1':
            assert expected in response_text
