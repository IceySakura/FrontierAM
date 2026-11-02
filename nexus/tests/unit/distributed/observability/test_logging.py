from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import threading

from datetime import timedelta

import pytest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.cli.commands.distributed import register_distributed_parser
from nexus.distributed.config import DistributedConfig
from nexus.distributed.execution.models import AssignmentStatus, ExecutionMode, ExecutionStatus
from nexus.distributed.gateways.control.app import create_control_app
from nexus.distributed.instance_manager.internal import create_instance_manager_internal_app
from nexus.distributed.observability import logging as distributed_logs
from nexus.distributed.observability.context import distributed_execution_context
from nexus.distributed.observability.logging import tail_log_file
from nexus.distributed.persistence.orm import AssignmentRow, EnvironmentAllocationRow, ExecutionRow, create_schema
from nexus.distributed.repositories.execution import CreateExecution, ExecutionRepository
from nexus.distributed.repositories.manager import ManagerRepository
from nexus.distributed.utils.clock import utcnow
from nexus.utils.log.formatters import TrimFormatter


@pytest.fixture
async def session_factory():
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    await create_schema(engine)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


class FakeService:
    manager_id = 'im-logs'
    session_id = 'session-logs'

    def __init__(self, log_path: str | None) -> None:
        self.log_path = log_path

    async def poll_assignments_once(self):
        return 0

    async def claim_commands_once(self):
        return 0


async def _attach_assignment(session, *, execution_id: str, manager_id: str, manager_session_id: str) -> None:
    now = utcnow()
    execution = await session.get(ExecutionRow, execution_id)
    assert execution is not None
    assignment_id = f'asn-{execution_id}'
    attempt_id = f'{execution_id}-attempt-1'
    session.add(
        AssignmentRow(
            assignment_id=assignment_id,
            execution_id=execution_id,
            owner_principal_id=execution.owner_principal_id,
            environment_id=execution.environment_id,
            pool_id=execution.pool_id,
            allocation_id=execution.allocation_id,
            workload=execution.workload,
            allocation_spec_hash=execution.allocation_spec_hash,
            attempt_id=attempt_id,
            attempt_seq=1,
            manager_id=manager_id,
            manager_session_id=manager_session_id,
            status=AssignmentStatus.PENDING,
            slot_cost=1,
            endpoint_ready=False,
            created_at=now,
            updated_at=now,
        )
    )
    execution.status = ExecutionStatus.ASSIGNED
    execution.current_assignment_id = assignment_id
    execution.current_attempt_id = attempt_id
    execution.attempt_count = 1
    execution.next_attempt_seq = 2
    execution.updated_at = now


def test_distributed_config_reads_log_environment(monkeypatch, tmp_path):
    monkeypatch.setenv('NEXUS_DIST_LOG_DIR', str(tmp_path))
    monkeypatch.setenv('NEXUS_DIST_LOG_LEVEL', 'debug')

    config = DistributedConfig.from_env()

    assert config.log_dir == str(tmp_path)
    assert config.log_level == 'DEBUG'


def test_tail_log_file_returns_last_lines(tmp_path):
    path = tmp_path / 'instance-manager-im-1.log'
    path.write_text('one\n二\nthree\nfour\n', encoding='utf-8')

    assert tail_log_file(path, lines=2) == 'three\nfour\n'
    assert tail_log_file(path, lines=0) == ''


def test_tail_log_file_filters_by_time_range_and_keeps_continuation_lines(tmp_path):
    path = tmp_path / 'instance-manager-im-1.log'
    path.write_text(
        '\n'.join([
            '2026-07-09 10:00:00 | INFO | nexus.before | before',
            '2026-07-09 11:00:00 | INFO | nexus.inside | inside one',
            'Traceback line for inside one',
            '2026-07-09 12:00:00 | INFO | nexus.inside | inside two',
            '2026-07-09 13:00:00 | INFO | nexus.after | after',
            '',
        ]),
        encoding='utf-8',
    )

    assert tail_log_file(
        path,
        lines=10,
        start_time='2026-07-09 11:00:00',
        end_time='2026-07-09T12:00:00',
    ) == (
        '2026-07-09 11:00:00 | INFO | nexus.inside | inside one\n'
        'Traceback line for inside one\n'
        '2026-07-09 12:00:00 | INFO | nexus.inside | inside two\n'
    )
    assert tail_log_file(path, lines=1, start_time='2026-07-09') == (
        '2026-07-09 13:00:00 | INFO | nexus.after | after\n'
    )


def _flush_root_handlers() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()


def _json_log_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def test_setup_distributed_logging_writes_json_lines_file(tmp_path):
    text_path = distributed_logs.setup_distributed_logging(
        log_dir=str(tmp_path),
        log_level='INFO',
        component='instance-manager',
        identity='im-json',
    )

    logging.getLogger('nexus.test.distributed').info(
        'collect me',
        extra={
            'execution_id': 'exec-1',
            'assignment_id': 'assign-1',
            'batch_id': 'batch-1',
            'allocation_id': 'allocation-1',
            'environment_id': 'environment-1',
            'artifact_id': 'artifact-1',
        },
    )
    _flush_root_handlers()

    assert text_path == tmp_path / 'instance-manager-im-json.log'
    json_path = tmp_path / 'instance-manager-im-json.jsonl'
    payload = _json_log_lines(json_path)[-1]
    assert re.match(r'\d{4}-\d{2}-\d{2}T.*Z', payload['timestamp'])
    assert payload['level'] == 'INFO'
    assert payload['logger'] == 'nexus.test.distributed'
    assert payload['message'] == 'collect me'
    assert payload['component'] == 'instance-manager'
    assert payload['identity'] == 'im-json'
    assert payload['execution_id'] == 'exec-1'
    assert payload['assignment_id'] == 'assign-1'
    assert payload['batch_id'] == 'batch-1'
    assert payload['allocation_id'] == 'allocation-1'
    assert payload['environment_id'] == 'environment-1'
    assert payload['artifact_id'] == 'artifact-1'
    assert isinstance(payload['lineno'], int)
    assert payload['func_name'] == 'test_setup_distributed_logging_writes_json_lines_file'
    assert isinstance(payload['process'], int)
    assert isinstance(payload['thread'], int)


def test_setup_distributed_logging_registers_json_handler_first(tmp_path):
    distributed_logs.setup_distributed_logging(
        log_dir=str(tmp_path),
        log_level='INFO',
        component='instance-manager',
        identity='im-order',
    )

    first_handler = logging.getLogger().handlers[0]
    assert str(getattr(first_handler, 'baseFilename', '')).endswith('instance-manager-im-order.jsonl')


def test_json_logging_keeps_full_message_when_text_handlers_trim(tmp_path):
    distributed_logs.setup_distributed_logging(
        log_dir=str(tmp_path),
        log_level='INFO',
        component='instance-manager',
        identity='im-full',
        trim_threshold=(20, 20),
    )

    message = 'plain ' + ('x' * 100) + ' tail'
    logging.getLogger('nexus.test.long').info(message)
    _flush_root_handlers()

    json_payload = _json_log_lines(tmp_path / 'instance-manager-im-full.jsonl')[-1]
    text_log = (tmp_path / 'instance-manager-im-full.log').read_text(encoding='utf-8')
    assert json_payload['message'] == message
    assert 'truncated log' in text_log


def test_trim_formatter_does_not_mutate_log_record():
    record = logging.LogRecord(
        name='nexus.test.trim',
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='plain ' + ('x' * 100) + ' tail',
        args=(),
        exc_info=None,
    )

    formatted = TrimFormatter((20, 20)).format(record)

    assert 'truncated log' in formatted
    assert 'truncated log' not in record.msg


def test_json_logging_includes_distributed_execution_context_and_resets(tmp_path):
    distributed_logs.setup_distributed_logging(
        log_dir=str(tmp_path),
        log_level='INFO',
        component='instance-manager',
        identity='im-context',
    )

    logger = logging.getLogger('nexus.test.context')
    with distributed_execution_context(
        execution_id='exec-context',
        assignment_id='assign-context',
        attempt_id='attempt-context',
        manager_id='manager-1',
        manager_session_id='session-1',
    ):
        logger.info('inside context')
    logger.info('outside context')
    _flush_root_handlers()

    payloads = _json_log_lines(tmp_path / 'instance-manager-im-context.jsonl')
    inside = next(payload for payload in payloads if payload['message'] == 'inside context')
    outside = next(payload for payload in payloads if payload['message'] == 'outside context')
    assert inside['execution_id'] == 'exec-context'
    assert inside['assignment_id'] == 'assign-context'
    assert inside['attempt_id'] == 'attempt-context'
    assert inside['manager_id'] == 'manager-1'
    assert inside['manager_session_id'] == 'session-1'
    assert 'execution_id' not in outside
    assert 'assignment_id' not in outside


def test_json_logging_extra_overrides_context_and_keeps_unknown_extra(tmp_path):
    distributed_logs.setup_distributed_logging(
        log_dir=str(tmp_path),
        log_level='INFO',
        component='scheduler',
        identity='owner-1',
    )

    with distributed_execution_context(execution_id='exec-context', scheduler_owner_id='owner-context'):
        logging.getLogger('nexus.test.extra').info(
            'extra fields',
            extra={
                'execution_id': 'exec-extra',
                'command_id': 'command-1',
                'custom_payload': {'nested': True},
                'not_json': {1, 2},
            },
        )
    _flush_root_handlers()

    payload = _json_log_lines(tmp_path / 'scheduler-owner-1.jsonl')[-1]
    assert payload['execution_id'] == 'exec-extra'
    assert payload['scheduler_owner_id'] == 'owner-context'
    assert payload['command_id'] == 'command-1'
    assert payload['extra']['custom_payload'] == {'nested': True}
    assert payload['extra']['not_json'] == '{1, 2}'


def test_json_logging_records_exception_details(tmp_path):
    distributed_logs.setup_distributed_logging(
        log_dir=str(tmp_path),
        log_level='INFO',
        component='reconciler',
    )

    try:
        raise RuntimeError('boom')
    except RuntimeError:
        logging.getLogger('nexus.test.exception').exception('failed operation')
    _flush_root_handlers()

    payload = _json_log_lines(tmp_path / 'reconciler.jsonl')[-1]
    assert payload['message'] == 'failed operation'
    assert payload['exception']['type'] == 'RuntimeError'
    assert payload['exception']['message'] == 'boom'
    assert 'Traceback' in payload['exception']['traceback']


def test_setup_distributed_logging_without_log_dir_does_not_add_json_file_handler():
    assert (
        distributed_logs.setup_distributed_logging(log_dir='', log_level='INFO', component='instance-manager') is None
    )

    assert all(
        not str(getattr(handler, 'baseFilename', '')).endswith('.jsonl') for handler in logging.getLogger().handlers
    )


async def test_async_tail_log_file_runs_sync_tail_in_worker_thread(monkeypatch):
    main_thread_id = threading.get_ident()
    observed_thread_ids = []

    def fake_tail(
        path,
        *,
        lines=200,
        max_bytes=distributed_logs.DEFAULT_TAIL_MAX_BYTES,
        start_time=None,
        end_time=None,
    ):
        assert start_time is None
        assert end_time is None
        observed_thread_ids.append(threading.get_ident())
        return f'{path}:{lines}:{max_bytes}'

    monkeypatch.setattr(distributed_logs, 'tail_log_file', fake_tail)

    assert await distributed_logs.async_tail_log_file('/tmp/example.log', lines=7, max_bytes=123) == (
        '/tmp/example.log:7:123'
    )
    assert observed_thread_ids
    assert observed_thread_ids[0] != main_thread_id


async def test_manager_internal_serves_local_log_tail(tmp_path):
    log_path = tmp_path / 'instance-manager-im-logs.log'
    log_path.write_text('first\nsecond\nthird\n', encoding='utf-8')
    app = create_instance_manager_internal_app(
        service=FakeService(str(log_path)),
        config=DistributedConfig(),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/internal/logs?lines=2')
        assert response.status == 200
        assert response.content_type == 'text/plain'
        assert await response.text() == 'second\nthird\n'
    finally:
        await client.close()


async def test_manager_internal_serves_structured_json_log_tail(tmp_path):
    text_path = tmp_path / 'instance-manager-im-logs.log'
    text_path.write_text('text fallback\n', encoding='utf-8')
    json_path = tmp_path / 'instance-manager-im-logs.jsonl'
    json_path.write_text(
        '\n'.join([
            json.dumps({'timestamp': '2026-08-28T10:00:00Z', 'message': 'first'}),
            json.dumps({'timestamp': '2026-08-28T10:00:01Z', 'message': 'second'}),
            '',
        ]),
        encoding='utf-8',
    )
    app = create_instance_manager_internal_app(
        service=FakeService(str(text_path)),
        config=DistributedConfig(),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/internal/logs?format=jsonl&lines=1')
        assert response.status == 200
        assert response.content_type == 'application/x-ndjson'
        assert json.loads(await response.text()) == {
            'timestamp': '2026-08-28T10:00:01Z',
            'message': 'second',
        }
    finally:
        await client.close()


async def test_manager_internal_serves_local_log_tail_with_time_range(tmp_path):
    log_path = tmp_path / 'instance-manager-im-logs.log'
    log_path.write_text(
        '2026-07-09 10:00:00 | INFO | nexus | before\n'
        '2026-07-09 11:00:00 | INFO | nexus | selected\n'
        '2026-07-09 12:00:00 | INFO | nexus | after\n',
        encoding='utf-8',
    )
    app = create_instance_manager_internal_app(
        service=FakeService(str(log_path)),
        config=DistributedConfig(),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get(
            '/v1/internal/logs?lines=20&start_time=2026-07-09+11:00:00&end_time=2026-07-09+11:00:00'
        )
        assert response.status == 200
        assert await response.text() == '2026-07-09 11:00:00 | INFO | nexus | selected\n'
    finally:
        await client.close()


async def test_control_gateway_proxies_manager_log_tail(session_factory):
    async def manager_logs(request):
        assert request.headers['Authorization'] == 'Bearer internal-secret'
        assert request.query['lines'] == '2'
        return web.Response(text='alpha\nbeta\n', content_type='text/plain')

    manager_app = web.Application()
    manager_app.router.add_get('/v1/internal/logs', manager_logs)
    manager_client = TestClient(TestServer(manager_app))
    await manager_client.start_server()
    try:
        async with session_factory() as session:
            repo = ManagerRepository(session, healthy_ttl=timedelta(seconds=15))
            await repo.register_session(
                manager_id='im-logs',
                session_id='session-logs',
                endpoint_url=str(manager_client.make_url('/')).rstrip('/'),
                supported_modes=['live'],
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
        control_client = TestClient(TestServer(app))
        await control_client.start_server()
        try:
            response = await control_client.get('/v1/managers/im-logs/logs?lines=2')
            assert response.status == 200
            assert response.content_type == 'text/plain'
            assert await response.text() == 'alpha\nbeta\n'
        finally:
            await control_client.close()
    finally:
        await manager_client.close()


async def test_control_gateway_proxies_manager_log_tail_time_range(session_factory):
    async def manager_logs(request):
        assert request.headers['Authorization'] == 'Bearer internal-secret'
        assert request.query['lines'] == '3'
        assert request.query['start_time'] == '2026-07-09 11:00:00'
        assert request.query['end_time'] == '2026-07-09 12:00:00'
        return web.Response(text='selected\n', content_type='text/plain')

    manager_app = web.Application()
    manager_app.router.add_get('/v1/internal/logs', manager_logs)
    manager_client = TestClient(TestServer(manager_app))
    await manager_client.start_server()
    try:
        async with session_factory() as session:
            repo = ManagerRepository(session, healthy_ttl=timedelta(seconds=15))
            await repo.register_session(
                manager_id='im-logs',
                session_id='session-logs',
                endpoint_url=str(manager_client.make_url('/')).rstrip('/'),
                supported_modes=['live'],
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
        control_client = TestClient(TestServer(app))
        await control_client.start_server()
        try:
            response = await control_client.get(
                '/v1/managers/im-logs/logs?lines=3&start_time=2026-07-09+11:00:00&end_time=2026-07-09+12:00:00'
            )
            assert response.status == 200
            assert response.content_type == 'text/plain'
            assert await response.text() == 'selected\n'
        finally:
            await control_client.close()
    finally:
        await manager_client.close()


async def test_control_gateway_proxies_execution_manager_log_tail(session_factory):
    async def manager_logs(request):
        assert request.headers['Authorization'] == 'Bearer internal-secret'
        assert request.query['lines'] == '500'
        return web.Response(text='manager alpha\nmanager beta\n', content_type='text/plain')

    manager_app = web.Application()
    manager_app.router.add_get('/v1/internal/logs', manager_logs)
    manager_client = TestClient(TestServer(manager_app))
    await manager_client.start_server()
    try:
        async with session_factory() as session:
            await ManagerRepository(session, healthy_ttl=timedelta(seconds=15)).register_session(
                manager_id='im-execution-logs',
                session_id='session-execution-logs',
                endpoint_url=str(manager_client.make_url('/')).rstrip('/'),
                supported_modes=[ExecutionMode.JOB],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                capacity_total=1,
                environment_id='env-execution-logs',
                pool_id='predefined',
                force_replace=False,
                now=utcnow(),
            )
            execution = await ExecutionRepository(session, shard_count=8).create_execution(
                CreateExecution(
                    execution_id='execution-manager-logs',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                    environment_id='env-execution-logs',
                    pool_id='predefined',
                    owner_principal_id='local-dev',
                ),
                now=utcnow(),
            )
            await _attach_assignment(
                session,
                execution_id=execution.execution_id,
                manager_id='im-execution-logs',
                manager_session_id='session-execution-logs',
            )
            await session.commit()

        app = create_control_app(
            session_factory=session_factory,
            config=DistributedConfig(
                secrets={'internal_token': 'internal-secret'},
                manager_healthy_ttl=timedelta(seconds=15),
            ),
        )
        control_client = TestClient(TestServer(app))
        await control_client.start_server()
        try:
            response = await control_client.get('/v1/executions/execution-manager-logs/manager-logs?lines=500')

            assert response.status == 200
            assert response.content_type == 'text/plain'
            assert await response.text() == 'manager alpha\nmanager beta\n'
        finally:
            await control_client.close()
    finally:
        await manager_client.close()


@pytest.mark.parametrize('chunked', [False, True], ids=['complete-body', 'split-jsonl-record'])
async def test_control_gateway_unified_execution_logs_reads_instance_manager_for_active_execution(
    session_factory, chunked
):
    async def manager_logs(request):
        assert request.headers['Authorization'] == 'Bearer internal-secret'
        assert request.query['format'] == 'jsonl'
        body = (
            json.dumps({
                'timestamp': '2026-08-25T10:00:00Z',
                'level': 'INFO',
                'logger': 'nexus.worker',
                'message': 'active line',
                'execution_id': 'execution-unified-active',
                'exception': {
                    'type': 'RuntimeError',
                    'message': 'active failure',
                    'traceback': 'Traceback: active failure',
                },
            })
            + '\n'
        )
        if chunked:
            response = web.StreamResponse(headers={'Content-Type': 'application/x-ndjson'})
            await response.prepare(request)
            midpoint = len(body) // 2
            await response.write(body[:midpoint].encode())
            await asyncio.sleep(0.05)
            await response.write(body[midpoint:].encode())
            await response.write_eof()
            return response
        return web.Response(text=body, content_type='application/x-ndjson')

    manager_app = web.Application()
    manager_app.router.add_get('/v1/internal/logs', manager_logs)
    manager_client = TestClient(TestServer(manager_app))
    await manager_client.start_server()
    try:
        async with session_factory() as session:
            await ManagerRepository(session, healthy_ttl=timedelta(seconds=15)).register_session(
                manager_id='im-unified-log',
                session_id='session-unified-log',
                endpoint_url=str(manager_client.make_url('/')).rstrip('/'),
                supported_modes=[ExecutionMode.JOB],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                capacity_total=1,
                force_replace=False,
                now=utcnow(),
            )
            execution = await ExecutionRepository(session, shard_count=8).create_execution(
                CreateExecution(
                    execution_id='execution-unified-active',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                    owner_principal_id='local-dev',
                ),
                now=utcnow(),
            )
            await _attach_assignment(
                session,
                execution_id=execution.execution_id,
                manager_id='im-unified-log',
                manager_session_id='session-unified-log',
            )
            await session.commit()

        app = create_control_app(
            session_factory=session_factory,
            config=DistributedConfig(
                auth_mode='disabled',
                secrets={'internal_token': 'internal-secret'},
            ),
        )
        control_client = TestClient(TestServer(app))
        await control_client.start_server()
        try:
            response = await control_client.get('/v1/executions/execution-unified-active/logs?lines=100')
            assert response.status == 200
            payload = await response.json()
            assert payload['source'] == 'instance_manager'
            assert payload['availability'] == 'ready'
            assert payload['lines'][0] == {
                'timestamp': '2026-08-25T10:00:00+00:00',
                'level': 'INFO',
                'logger': 'nexus.worker',
                'message': 'active line',
                'fields': {'execution_id': 'execution-unified-active'},
                'exception': {
                    'type': 'RuntimeError',
                    'message': 'active failure',
                    'traceback': 'Traceback: active failure',
                },
            }
        finally:
            await control_client.close()
    finally:
        await manager_client.close()


async def test_control_gateway_unified_execution_logs_uses_terminal_zhiyan_without_manager_fallback(session_factory):
    async with session_factory() as session:
        execution = await ExecutionRepository(session, shard_count=8).create_execution(
            CreateExecution(
                execution_id='execution-unified-terminal',
                mode=ExecutionMode.JOB,
                spec_type='nexus_single',
                instance_mode='asyncio',
                spec={},
                owner_principal_id='local-dev',
                allocation_id='missing-allocation',
            ),
            now=utcnow(),
        )
        row = await session.get(ExecutionRow, execution.execution_id)
        assert row is not None
        row.status = ExecutionStatus.SUCCEEDED
        row.finished_at = utcnow()
        await session.commit()

    app = create_control_app(
        session_factory=session_factory,
        config=DistributedConfig(
            auth_mode='disabled',
            zhiyan_log_query={
                'enabled': True,
                'project_name': 'Nexus-Agent',
                'staff_name': 'service-account',
                'credential_ref': 'zhiyan',
            },
            secrets={'zhiyan_log_credentials': {'zhiyan': 'TOKEN_SENTINEL'}},
        ),
    )
    control_client = TestClient(TestServer(app))
    await control_client.start_server()
    try:
        response = await control_client.get('/v1/executions/execution-unified-terminal/logs')
        assert response.status == 200
        payload = await response.json()
        assert payload['source'] == 'zhiyan'
        assert payload['error_code'] == 'zhiyan_log_not_configured'
        assert 'TOKEN_SENTINEL' not in await response.text()
    finally:
        await control_client.close()


async def test_control_gateway_unified_execution_logs_reads_terminal_zhiyan_records(session_factory):
    captured = {}

    async def zhiyan_query(request):
        captured['headers'] = dict(request.headers)
        captured['body'] = await request.json()
        return web.json_response({
            'code': 0,
            'data': {
                'records': [
                    {
                        'timestamp': '2026-08-25T10:00:02Z',
                        'level': 'INFO',
                        'logger': 'nexus.worker',
                        'message': 'terminal late',
                    },
                    {
                        'timestamp': '2026-08-25T10:00:01Z',
                        'level': 'ERROR',
                        'logger': 'nexus.worker',
                        'message': 'terminal early',
                        'exception': {
                            'type': 'RuntimeError',
                            'message': 'terminal failure',
                            'traceback': 'Traceback: terminal failure',
                        },
                    },
                ]
            },
        })

    zhiyan_app = web.Application()
    zhiyan_app.router.add_post('/log/v2/search/query', zhiyan_query)
    zhiyan_server = TestServer(zhiyan_app)
    await zhiyan_server.start_server()
    try:
        now = utcnow()
        async with session_factory() as session:
            execution = await ExecutionRepository(session, shard_count=8).create_execution(
                CreateExecution(
                    execution_id='execution-unified-terminal-real',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                    owner_principal_id='local-dev',
                    allocation_id='allocation-unified-terminal-real',
                ),
                now=now,
            )
            row = await session.get(ExecutionRow, execution.execution_id)
            assert row is not None
            row.status = ExecutionStatus.SUCCEEDED
            row.finished_at = now
            row.updated_at = now
            session.add(
                EnvironmentAllocationRow(
                    allocation_id='allocation-unified-terminal-real',
                    top_level_execution_id=execution.execution_id,
                    owner_principal_id='local-dev',
                    environment_id='predefined-default',
                    pool_id='default',
                    workload='execution',
                    status='ready',
                    desired_manager_count=1,
                    ready_manager_count=1,
                    manager_count=1,
                    cpu_millicores=1,
                    memory_bytes=1,
                    slots=1,
                    active_batches=0,
                    allocation_spec_json={
                        'provider_type': 'tkex',
                        'provider_config': {'zhiyan_log': {'log_name': 'immutable-manager-flow'}},
                    },
                    allocation_spec_hash='a' * 64,
                    config_version=1,
                    status_version=1,
                    last_progress_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

        app = create_control_app(
            session_factory=session_factory,
            config=DistributedConfig(
                auth_mode='disabled',
                zhiyan_log_query={
                    'enabled': True,
                    'endpoint': str(zhiyan_server.make_url('/')).rstrip('/'),
                    'project_name': 'Nexus-Agent',
                    'staff_name': 'service-account',
                    'credential_ref': 'zhiyan',
                },
                secrets={'zhiyan_log_credentials': {'zhiyan': 'TOKEN_SENTINEL'}},
            ),
        )
        control_client = TestClient(TestServer(app))
        await control_client.start_server()
        try:
            response = await control_client.get('/v1/executions/execution-unified-terminal-real/logs?lines=100')
            assert response.status == 200
            payload = await response.json()
            assert payload['source'] == 'zhiyan'
            assert payload['availability'] == 'ready'
            assert [line['message'] for line in payload['lines']] == ['terminal early', 'terminal late']
            assert payload['lines'][0]['logger'] == 'nexus.worker'
            assert payload['lines'][0]['exception'] == {
                'type': 'RuntimeError',
                'message': 'terminal failure',
                'traceback': 'Traceback: terminal failure',
            }
            assert captured['headers']['token'] == 'TOKEN_SENTINEL'
            assert captured['headers']['projectname'] == 'Nexus-Agent'
            assert captured['body']['dataflowName'] == 'immutable-manager-flow'
            assert captured['body']['filter'] == [
                {'key': 'execution_id', 'operator': 'eq', 'value': execution.execution_id}
            ]
            assert 'TOKEN_SENTINEL' not in await response.text()
        finally:
            await control_client.close()
    finally:
        await zhiyan_server.close()


async def test_control_gateway_terminal_logs_bind_predefined_pool_via_deployment_dataflow(session_factory):
    captured = {}

    async def zhiyan_query(request):
        captured['body'] = await request.json()
        return web.json_response({
            'code': 0,
            'data': {
                'records': [
                    {
                        'timestamp': '2026-09-12T10:00:01Z',
                        'level': 'INFO',
                        'logger': 'nexus.worker',
                        'message': 'predefined terminal line',
                    },
                ]
            },
        })

    zhiyan_app = web.Application()
    zhiyan_app.router.add_post('/log/v2/search/query', zhiyan_query)
    zhiyan_server = TestServer(zhiyan_app)
    await zhiyan_server.start_server()
    try:
        now = utcnow()
        async with session_factory() as session:
            execution = await ExecutionRepository(session, shard_count=8).create_execution(
                CreateExecution(
                    execution_id='execution-predefined-terminal',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                    owner_principal_id='local-dev',
                    allocation_id='allocation-predefined-terminal',
                ),
                now=now,
            )
            row = await session.get(ExecutionRow, execution.execution_id)
            assert row is not None
            row.status = ExecutionStatus.SUCCEEDED
            row.finished_at = now
            row.updated_at = now
            session.add(
                EnvironmentAllocationRow(
                    allocation_id='allocation-predefined-terminal',
                    top_level_execution_id=execution.execution_id,
                    owner_principal_id='local-dev',
                    environment_id='predefined-local',
                    pool_id='interactive',
                    workload='execution',
                    status='ready',
                    desired_manager_count=1,
                    ready_manager_count=1,
                    manager_count=1,
                    cpu_millicores=1,
                    memory_bytes=1,
                    slots=1,
                    active_batches=0,
                    allocation_spec_json={
                        'allocation_type': 'predefined',
                        'provider_type': 'predefined',
                        'provider_config_ref': None,
                    },
                    allocation_spec_hash='b' * 64,
                    config_version=1,
                    status_version=1,
                    last_progress_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

        app = create_control_app(
            session_factory=session_factory,
            config=DistributedConfig(
                auth_mode='disabled',
                predefined_manager_zhiyan_log_name='dev_nexus_manager_log',
                zhiyan_log_query={
                    'enabled': True,
                    'endpoint': str(zhiyan_server.make_url('/')).rstrip('/'),
                    'project_name': 'Nexus-Agent',
                    'staff_name': 'service-account',
                    'credential_ref': 'zhiyan',
                },
                secrets={'zhiyan_log_credentials': {'zhiyan': 'TOKEN_SENTINEL'}},
            ),
        )
        control_client = TestClient(TestServer(app))
        await control_client.start_server()
        try:
            response = await control_client.get('/v1/executions/execution-predefined-terminal/logs?lines=100')
            assert response.status == 200
            payload = await response.json()
            assert payload['source'] == 'zhiyan'
            assert payload['availability'] == 'ready'
            assert [line['message'] for line in payload['lines']] == ['predefined terminal line']
            assert captured['body']['dataflowName'] == 'dev_nexus_manager_log'
            assert captured['body']['filter'] == [
                {'key': 'execution_id', 'operator': 'eq', 'value': execution.execution_id}
            ]
        finally:
            await control_client.close()
    finally:
        await zhiyan_server.close()


async def test_control_gateway_terminal_logs_predefined_without_dataflow_not_configured(session_factory):
    now = utcnow()
    async with session_factory() as session:
        execution = await ExecutionRepository(session, shard_count=8).create_execution(
            CreateExecution(
                execution_id='execution-predefined-unconfigured',
                mode=ExecutionMode.JOB,
                spec_type='nexus_single',
                instance_mode='asyncio',
                spec={},
                owner_principal_id='local-dev',
                allocation_id='allocation-predefined-unconfigured',
            ),
            now=now,
        )
        row = await session.get(ExecutionRow, execution.execution_id)
        assert row is not None
        row.status = ExecutionStatus.SUCCEEDED
        row.finished_at = now
        row.updated_at = now
        session.add(
            EnvironmentAllocationRow(
                allocation_id='allocation-predefined-unconfigured',
                top_level_execution_id=execution.execution_id,
                owner_principal_id='local-dev',
                environment_id='predefined-local',
                pool_id='interactive',
                workload='execution',
                status='ready',
                desired_manager_count=1,
                ready_manager_count=1,
                manager_count=1,
                cpu_millicores=1,
                memory_bytes=1,
                slots=1,
                active_batches=0,
                allocation_spec_json={
                    'allocation_type': 'predefined',
                    'provider_type': 'predefined',
                    'provider_config_ref': None,
                },
                allocation_spec_hash='c' * 64,
                config_version=1,
                status_version=1,
                last_progress_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()

    app = create_control_app(
        session_factory=session_factory,
        config=DistributedConfig(
            auth_mode='disabled',
            zhiyan_log_query={
                'enabled': True,
                'project_name': 'Nexus-Agent',
                'staff_name': 'service-account',
                'credential_ref': 'zhiyan',
            },
            secrets={'zhiyan_log_credentials': {'zhiyan': 'TOKEN_SENTINEL'}},
        ),
    )
    control_client = TestClient(TestServer(app))
    await control_client.start_server()
    try:
        response = await control_client.get('/v1/executions/execution-predefined-unconfigured/logs')
        assert response.status == 200
        payload = await response.json()
        assert payload['source'] == 'zhiyan'
        assert payload['error_code'] == 'zhiyan_log_not_configured'
    finally:
        await control_client.close()


async def test_control_gateway_streams_execution_manager_logs(session_factory):
    async def manager_logs_stream(request):
        assert request.headers['Authorization'] == 'Bearer internal-secret'
        assert request.query['lines'] == '200'
        response = web.StreamResponse(headers={'Content-Type': 'text/plain; charset=utf-8'})
        await response.prepare(request)
        await response.write(b'first\n')
        await response.write(b'second\n')
        await response.write_eof()
        return response

    manager_app = web.Application()
    manager_app.router.add_get('/v1/internal/logs/stream', manager_logs_stream)
    manager_client = TestClient(TestServer(manager_app))
    await manager_client.start_server()
    try:
        async with session_factory() as session:
            await ManagerRepository(session, healthy_ttl=timedelta(seconds=15)).register_session(
                manager_id='im-execution-log-stream',
                session_id='session-execution-log-stream',
                endpoint_url=str(manager_client.make_url('/')).rstrip('/'),
                supported_modes=[ExecutionMode.JOB],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                capacity_total=1,
                environment_id='env-execution-log-stream',
                pool_id='predefined',
                force_replace=False,
                now=utcnow(),
            )
            execution = await ExecutionRepository(session, shard_count=8).create_execution(
                CreateExecution(
                    execution_id='execution-manager-log-stream',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                    environment_id='env-execution-log-stream',
                    pool_id='predefined',
                    owner_principal_id='local-dev',
                ),
                now=utcnow(),
            )
            await _attach_assignment(
                session,
                execution_id=execution.execution_id,
                manager_id='im-execution-log-stream',
                manager_session_id='session-execution-log-stream',
            )
            await session.commit()

        app = create_control_app(
            session_factory=session_factory,
            config=DistributedConfig(
                secrets={'internal_token': 'internal-secret'},
                manager_healthy_ttl=timedelta(seconds=15),
            ),
        )
        control_client = TestClient(TestServer(app))
        await control_client.start_server()
        try:
            response = await control_client.get(
                '/v1/executions/execution-manager-log-stream/manager-logs/stream?lines=200'
            )

            assert response.status == 200
            assert response.headers['Cache-Control'] == 'no-cache'
            assert await response.text() == 'first\nsecond\n'
        finally:
            await control_client.close()
    finally:
        await manager_client.close()


def test_distributed_cli_registers_logs_tail_command():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    register_distributed_parser(subparsers)

    args = parser.parse_args([
        'distributed',
        'logs',
        'tail',
        '--control-base-url',
        'http://control.local',
        '--manager-id',
        'im-1',
        '--lines',
        '50',
        '--start-time',
        '2026-07-09 11:00:00',
        '--end-time',
        '2026-07-09 12:00:00',
        '--follow',
    ])

    assert args.distributed_command == 'logs'
    assert args.logs_command == 'tail'
    assert args.control_base_url == 'http://control.local'
    assert args.manager_id == 'im-1'
    assert args.lines == 50
    assert args.start_time == '2026-07-09 11:00:00'
    assert args.end_time == '2026-07-09 12:00:00'
    assert args.follow is True
