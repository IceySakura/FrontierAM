"""Real MySQL + authenticated HTTP acceptance; uses a disposable local database."""

from __future__ import annotations

import json

from pathlib import Path

import pytest

from tests.integration.distributed.test_cloud_event_export import database as database
from tests.unit.distributed.gateways.test_trace_event_routes import LIST_URL, OWNER
from tests.unit.distributed.gateways.test_trace_event_routes import gateway as gateway
from tests.unit.distributed.test_cloud_submit_auth_integration import _identity_headers
from tests.unit.distributed.test_repositories import session_factory as session_factory
from tests.unit.distributed.test_trace_read_service import event

from nexus.distributed.config.secrets import MySQLTraceDatabaseCredential
from nexus.distributed.gateways.application.routes.trace_events import TRACE_READ_SERVICE_KEY
from nexus.distributed.persistence.orm import ExecutionRow
from nexus.distributed.trace.sources import TraceSourceResolver
from nexus.runner.trace import principal_trace_scope_id
from nexus.runner.trace.tdsql.schema import principal_trace_table_name, principal_trace_table_statement


pytestmark = pytest.mark.integration
SCOPE = principal_trace_scope_id(OWNER)
TABLE = principal_trace_table_name(SCOPE)


def headers():
    return _identity_headers(1001)


@pytest.fixture
async def real_api(gateway, database, monkeypatch):
    client, _, _, sessions = gateway
    config, connect = database
    service = client.app[TRACE_READ_SERVICE_KEY]
    settings = service.config.model_copy(
        update={
            'readers': tuple(
                reader.model_copy(
                    update={'host': config['host'], 'port': config['port'], 'database': config['database']}
                )
                for reader in service.config.readers
            )
        }
    )
    service.config = settings
    service.resolver.settings = settings
    service.resolver.config.trace_read = settings
    service.resolver.config.secrets.trace_read_credentials['reader'] = MySQLTraceDatabaseCredential(
        username=config['reader_user'],
        password=Path(config['reader_password_file']).read_text(),
    )
    monkeypatch.setattr(service.resolver, 'create_source', TraceSourceResolver.create_source.__get__(service.resolver))
    connection = await connect()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(principal_trace_table_statement(SCOPE).replace(' shardkey=run_id', ''))
            await cursor.execute(f'DELETE FROM {TABLE} WHERE run_id = %s', ('exec',))
    finally:
        connection.close()
    return client, connect, sessions


async def insert(connection, records):
    columns = tuple(event(1))
    async with connection.cursor() as cursor:
        await cursor.executemany(
            f'INSERT INTO {TABLE} ({", ".join(columns)}) VALUES ({", ".join(["%s"] * len(columns))})',
            [tuple(row.values()) for row in records],
        )


async def discover(client):
    response = await client.get(LIST_URL, headers=headers())
    assert response.status == 200, await response.text()
    return await response.json()


async def test_real_http_reads_only_committed_rows_and_late_terminal_tail(real_api):
    client, connect, sessions = real_api
    writer = await connect(autocommit=False)
    try:
        await insert(writer, [event(1)])
        assert (await discover(client))['items'] == []
        await writer.commit()
        selected = (await discover(client))['items'][0]['stream_id']
        path = f'{LIST_URL}/{selected}/events'
        response = await client.get(path, headers=headers())
        assert response.status == 200
        first = await response.json()
        assert first['items'][0]['id'] == '9007199254740993'
        assert first['items'][0]['event_data'] == event(1)['event_data']
        await insert(writer, [event(3)])
        pending = await (await client.get(path, params={'cursor': first['next_cursor']}, headers=headers())).json()
        assert pending['items'] == []
        async with sessions() as session:
            execution = await session.get(ExecutionRow, 'exec')
            execution.status = 'succeeded'
            await session.commit()
        await writer.commit()
        late = await (await client.get(path, params={'cursor': pending['next_cursor']}, headers=headers())).json()
        assert [row['record_id'] for row in late['items']] == ['3']
        assert late['execution_status'] == 'succeeded'
        assert late['has_more'] is False
    finally:
        writer.close()


async def test_real_http_multi_stream_600_rows_equal_all_database_columns(real_api):
    client, connect, _ = real_api
    writer = await connect()
    try:
        records = [event(i * 2 + 1, span=('B' if i < 300 else 'Z')) for i in range(600)]
        await insert(writer, records)
        first = await (await client.get(LIST_URL, params={'limit': 1}, headers=headers())).json()
        assert first['items'][0]['run_span_id'] == 'B'
        await insert(writer, [event(1203, span='a')])
        rest = await (
            await client.get(LIST_URL, params={'limit': 1, 'cursor': first['next_cursor']}, headers=headers())
        ).json()
        assert [row['run_span_id'] for row in rest['items']] == ['Z']
        discovery = await discover(client)
        assert [row['run_span_id'] for row in discovery['items']] == ['a', 'B', 'Z']
        all_rows = []
        for selected in discovery['items']:
            cursor = None
            while True:
                params = {'limit': 77}
                if cursor:
                    params['cursor'] = cursor
                response = await client.get(
                    f'{LIST_URL}/{selected["stream_id"]}/events', params=params, headers=headers()
                )
                assert response.status == 200, await response.text()
                page = await response.json()
                for item in page['items']:
                    all_rows.append({**item, **{key: int(item[key]) for key in ('id', 'record_id', 'time_unix_nano')}})
                cursor = page['next_cursor']
                if not page['has_more']:
                    break
        async with writer.cursor() as cursor:
            columns = tuple(event(1))
            await cursor.execute(
                f'SELECT {", ".join(columns)} FROM {TABLE} WHERE run_id=%s ORDER BY run_span_id,record_id', ('exec',)
            )
            oracle = [dict(zip(columns, row, strict=True)) for row in await cursor.fetchall()]
        for row in oracle:
            row['created_at'] = row['created_at'].isoformat()
        assert all_rows == oracle
        assert len(all_rows) == 601
    finally:
        writer.close()


async def test_real_http_large_singleton_and_raw_body_guard(real_api):
    client, connect, _ = real_api
    writer = await connect()
    try:
        payload = json.dumps({'data': '轨迹' * (400 * 1024)}, ensure_ascii=False)
        await insert(writer, [event(1, payload=payload), event(2, payload='x' * (9 * 1024 * 1024)), event(3)])
        selected = (await discover(client))['items'][0]['stream_id']
        path = f'{LIST_URL}/{selected}/events'
        response = await client.get(path, headers=headers())
        body = await response.read()
        assert response.status == 200, body[:200]
        assert len(body) > 2 * 1024 * 1024
        page = json.loads(body)
        assert len(page['items']) == 1
        assert page['items'][0]['event_data'] == payload
        assert page['has_more'] is True
        for _ in range(2):
            response = await client.get(path, params={'cursor': page['next_cursor']}, headers=headers())
            error = await response.json()
            assert response.status == 413
            assert error['error'] == 'trace_event_too_large'
            assert error['detail'] == f'record_id=2 size_bytes={9 * 1024 * 1024}'
    finally:
        writer.close()


async def test_real_http_preview_suppresses_twenty_mib_in_sql_and_preserves_cursor(real_api, monkeypatch):
    from nexus.runner.trace.tdsql import sink_source

    client, connect, _ = real_api
    writer = await connect()
    large = json.dumps({'attributes': {'large': 'x' * (20 * 1024 * 1024)}}, ensure_ascii=False)
    observed = []
    fetch = sink_source._RawCursor.fetchone

    async def capture(cursor):
        row = await fetch(cursor)
        if row is not None and len(row) == 13 and row[4] == 2:
            observed.append((row[10] is None, row[12]))
        return row

    monkeypatch.setattr(sink_source._RawCursor, 'fetchone', capture)
    try:
        await insert(writer, [event(1), event(2, payload=large), event(3)])
        selected = (await discover(client))['items'][0]['stream_id']
        path = f'{LIST_URL}/{selected}/events'
        first = await (await client.get(path, params={'limit': 1}, headers=headers())).json()
        failed = await client.get(path, params={'cursor': first['next_cursor']}, headers=headers())
        assert failed.status == 413
        response = await client.get(
            path, params={'cursor': first['next_cursor'], 'mode': 'preview', 'limit': 1}, headers=headers()
        )
        assert response.status == 200
        body = await response.read()
        assert len(body) < 4096
        preview = json.loads(body)
        row = preview['items'][0]
        assert row['event_data'] is None
        assert row['event_data_omitted'] is True
        assert row['event_data_bytes'] == len(large.encode())
        assert observed
        assert all(null and size == len(large.encode()) for null, size in observed)
        tail_response = await client.get(path, params={'cursor': preview['next_cursor']}, headers=headers())
        assert tail_response.status == 200
        tail = await tail_response.json()
        assert tail['mode'] == 'raw'
        assert [row['record_id'] for row in tail['items']] == ['3']
        async with writer.cursor() as cursor:
            # Match the raw reader's identity-first column order for a multi-packet row.
            await cursor.execute(
                f'SELECT record_id,event_data FROM {TABLE} WHERE run_id=%s AND record_id=%s', ('exec', 2)
            )
            assert (await cursor.fetchone())[1] == large
    finally:
        writer.close()


async def test_mysql_discovery_and_small_page_use_indexes(real_api, tmp_path):
    _, connect, _ = real_api
    writer = await connect()
    try:
        await insert(writer, [event(i + 1) for i in range(2000)])
        queries = [
            (f'SELECT DISTINCT run_span_id FROM {TABLE} WHERE run_id=%s ORDER BY run_span_id LIMIT %s', ('exec', 101)),
            (
                f'SELECT id,record_id,event_data FROM {TABLE} WHERE run_id=%s AND run_span_id=%s AND record_id>%s ORDER BY record_id LIMIT %s',
                ('exec', 'span', 1900, 6),
            ),
        ]
        evidence = []
        async with writer.cursor() as cursor:
            for sql, params in queries:
                await cursor.execute('EXPLAIN ' + sql, params)
                columns = [column[0] for column in cursor.description]
                plan = [dict(zip(columns, row, strict=True)) for row in await cursor.fetchall()]
                assert all(row['type'] != 'ALL' and row['key'] is not None for row in plan)
                await cursor.execute('EXPLAIN ANALYZE ' + sql, params)
                actual = [row[0] for row in await cursor.fetchall()]
                evidence.append({'plan': plan, 'analyze': actual})
        (tmp_path / 'query-plans.json').write_text(json.dumps(evidence, indent=2, default=str))
    finally:
        writer.close()


async def test_real_mysql_cancel_terminates_server_query_before_releasing_budget(real_api, database):
    import asyncio
    import time

    from nexus.runner.trace.tdsql.sink_source import TDSQLTraceSource

    client, connect, _ = real_api
    config, _ = database
    source = TDSQLTraceSource(
        host=config['host'],
        port=config['port'],
        db=config['database'],
        user=config['reader_user'],
        password=Path(config['reader_password_file']).read_text(),
    )
    entered = asyncio.Event()
    ids = []

    async def slow():
        async with source._raw_cursor() as cursor:
            await cursor.execute('SELECT CONNECTION_ID()')
            ids.append((await cursor.fetchone())[0])
            await cursor.fetchone()
            entered.set()
            await cursor.execute('SELECT SLEEP(20)')

    service = client.app[TRACE_READ_SERVICE_KEY]
    from tests.unit.distributed.test_trace_read_service import target

    task = asyncio.create_task(service._query(target(), slow))
    await entered.wait()
    await asyncio.sleep(0.05)
    started = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - started < 2
    assert service.admission._active == 0
    oracle = await connect()
    try:
        async with oracle.cursor() as cursor:
            await cursor.execute('SHOW PROCESSLIST')
            assert not any(row[0] == ids[0] for row in await cursor.fetchall())
    finally:
        oracle.close()
