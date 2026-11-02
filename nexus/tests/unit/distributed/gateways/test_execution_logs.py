from __future__ import annotations

import json

from datetime import datetime, timezone

import pytest

from nexus.distributed.config import ZhiyanLogQueryConfig
from nexus.distributed.gateways.control.execution_logs import (
    ZhiyanLogClient,
    normalize_zhiyan_records,
    parse_manager_log_text,
)


def test_manager_text_is_normalized_and_truncated_at_byte_boundary() -> None:
    lines, partial = parse_manager_log_text(
        '2026-08-25T10:00:02Z WARN second\nplain first\n',
        max_bytes=40,
        max_lines=10,
    )

    assert partial is True
    assert [line.message for line in lines] == ['second']
    assert lines[0].level == 'WARN'


def test_manager_json_lines_use_structured_execution_log_shape() -> None:
    text = json.dumps({
        'timestamp': '2026-08-25T10:00:00Z',
        'level': 'ERROR',
        'logger': 'nexus.worker',
        'message': 'structured line',
        'execution_id': 'execution-1',
        'exception': {
            'type': 'RuntimeError',
            'message': 'boom',
            'traceback': 'Traceback: boom',
        },
    })

    lines, partial = parse_manager_log_text(text)

    assert partial is False
    assert len(lines) == 1
    assert lines[0].timestamp == datetime(2026, 8, 25, 10, 0, tzinfo=timezone.utc)
    assert lines[0].level == 'ERROR'
    assert lines[0].logger == 'nexus.worker'
    assert lines[0].message == 'structured line'
    assert lines[0].fields == {'execution_id': 'execution-1'}
    assert lines[0].exception == {
        'type': 'RuntimeError',
        'message': 'boom',
        'traceback': 'Traceback: boom',
    }


def test_manager_pipe_text_lines_use_structured_execution_log_shape() -> None:
    lines, partial = parse_manager_log_text('2026-08-25 10:00:00 | WARN | nexus.worker | text line\n')

    assert partial is False
    assert len(lines) == 1
    assert lines[0].timestamp == datetime(2026, 8, 25, 10, 0, tzinfo=timezone.utc)
    assert lines[0].level == 'WARN'
    assert lines[0].logger == 'nexus.worker'
    assert lines[0].message == 'text line'


def test_zhiyan_records_are_filtered_to_safe_fields_and_sorted_ascending() -> None:
    records = [
        {'timestamp': '2026-08-25T10:00:02Z', 'level': 'ERROR', 'message': 'late', 'token': 'LEAK'},
        {
            'timestamp': '2026-08-25T10:00:01Z',
            'msg': 'early',
            'fields': {'manager_id': 'manager-1', 'credential': 'LEAK'},
        },
        {'message': 'malformed timestamp'},
    ]

    normalized, partial = normalize_zhiyan_records(records, max_lines=2)

    assert partial is False
    assert [line.message for line in normalized] == ['early', 'late']
    assert normalized[0].fields == {'manager_id': 'manager-1'}
    assert all('LEAK' not in repr(line) for line in normalized)


class _FakeResponse:
    status = 200
    headers = {'x-request-id': 'request-1'}

    async def json(self):
        return {'code': 0, 'data': {'records': [{'timestamp': '2026-08-25T10:00:00Z', 'message': 'ok'}]}}


class _FakeRequest:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *_args):
        return None


class _FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _FakeRequest(_FakeResponse())


@pytest.mark.asyncio
async def test_zhiyan_client_uses_fixed_filter_and_redacts_token_from_failures() -> None:
    session = _FakeSession()
    config = ZhiyanLogQueryConfig(
        enabled=True,
        endpoint='https://zhiyan.example.test',
        project_name='Nexus-Agent',
        staff_name='service-account',
        credential_ref='credential',
    )
    client = ZhiyanLogClient(session=session, config=config, token='TOKEN_SENTINEL')

    records = await client.query(
        log_name='manager-log-flow',
        filters=[{'key': 'execution_id', 'operator': 'eq', 'value': 'execution-1'}],
        start_time=datetime(2026, 8, 25, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 8, 25, 11, 0, tzinfo=timezone.utc),
        limit=200,
    )

    assert records[0]['message'] == 'ok'
    args, kwargs = session.calls[0]
    assert args[0] == 'https://zhiyan.example.test/log/v2/search/query'
    assert kwargs['headers']['token'] == 'TOKEN_SENTINEL'
    assert kwargs['headers']['x-api-version'] == 'v2'
    assert kwargs['auto_decompress'] is True
    assert kwargs['json']['dataflowName'] == 'manager-log-flow'
    assert kwargs['json']['filter'] == [{'key': 'execution_id', 'operator': 'eq', 'value': 'execution-1'}]
    assert kwargs['json']['orderBy'] == 'desc'
    assert kwargs['json']['startTime'] == 1787652000000
    assert kwargs['json']['endTime'] == 1787655600000


@pytest.mark.parametrize('max_bytes', [12, 1024])
async def test_remote_log_reads_all_chunks_with_a_byte_limit(max_bytes):
    import asyncio

    import aiohttp

    from aiohttp import web
    from aiohttp.test_utils import TestServer

    from nexus.distributed.gateways.control.execution_logs import read_remote_log

    async def stream(request):
        assert request.headers['Authorization'] == 'Bearer test-read'
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b'first\n')
        await asyncio.sleep(0.01)
        await response.write(b'second\nthird\n')
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get('/logs', stream)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        lines, partial = await read_remote_log(
            session,
            str(server.make_url('/logs')),
            headers={'Authorization': 'Bearer test-read'},
            lines=20,
            max_bytes=max_bytes,
        )
    assert partial is (max_bytes == 12)
    assert [line.message for line in lines] == (['first'] if max_bytes == 12 else ['first', 'second', 'third'])


async def test_remote_builder_log_does_not_follow_redirects():
    import aiohttp

    from aiohttp import web
    from aiohttp.test_utils import TestServer

    from nexus.distributed.gateways.control.execution_logs import read_remote_log

    async def redirect(request):
        raise web.HTTPFound('/target')

    async def target(request):
        pytest.fail('Builder read token must not be forwarded to a redirect target')

    app = web.Application()
    app.router.add_get('/logs', redirect)
    app.router.add_get('/target', target)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        with pytest.raises(aiohttp.ClientError):
            await read_remote_log(session, str(server.make_url('/logs')), headers={}, lines=20, allow_redirects=False)


async def test_zhiyan_query_uses_exact_requested_interval_and_limit():
    session = _FakeSession()
    config = ZhiyanLogQueryConfig(query_window_seconds=60, max_lines=10)
    client = ZhiyanLogClient(session=session, config=config, token='test')
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 2, tzinfo=timezone.utc)
    await client.query(
        log_name='test',
        filters=[{'key': 'execution_id', 'operator': 'eq', 'value': 'execution'}],
        start_time=start,
        end_time=end,
        limit=200,
    )
    body = session.calls[0][1]['json']
    assert body['startTime'] == int(start.timestamp() * 1000)
    assert body['endTime'] == int(end.timestamp() * 1000)
    assert body['limit'] == 200
