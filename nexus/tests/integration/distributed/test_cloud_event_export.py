"""Real MySQL and CLI process checks; opt in with NEXUS_EXPORT_TEST_MYSQL_CONFIG.

The JSON config names host, port, database, password_file (local fixture root),
and optionally reader_user/reader_password_file for the SELECT-only exporter.
Use a disposable local MySQL database: this fixture creates its trace table.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
import uuid

from datetime import datetime
from pathlib import Path

import pytest

from aiohttp import web
from aiohttp.test_utils import TestServer

from nexus.distributed.export_output import JsonlEventWriter
from nexus.distributed.export_types import TraceStreamKey
from nexus.runner.trace.tdsql.schema import principal_trace_table_name, principal_trace_table_statement
from nexus.runner.trace.tdsql.sink_source import TDSQLTraceSource


pytestmark = pytest.mark.integration
SCOPE = '0123456789abcdef0123456789abcdef'
TABLE = principal_trace_table_name(SCOPE)
ROOT = Path(__file__).parents[3]
COLUMNS = (
    'id',
    'run_id',
    'run_span_id',
    'parent_execution_id',
    'record_id',
    'event_type',
    'time_unix_nano',
    'trace_id',
    'span_id',
    'parent_span_id',
    'event_data',
    'created_at',
)


@pytest.fixture
async def database():
    config_path = os.environ.get('NEXUS_EXPORT_TEST_MYSQL_CONFIG')
    if not config_path:
        pytest.skip('set NEXUS_EXPORT_TEST_MYSQL_CONFIG to a disposable local MySQL fixture')
    assert config_path is not None
    asyncmy = pytest.importorskip('asyncmy')
    config = json.loads(Path(config_path).read_text())
    assert config['host'] in {'localhost', '127.0.0.1', '::1'}, 'fixture writes require a local database'

    async def connect(*, autocommit=True):
        return await asyncmy.connect(
            host=config['host'],
            port=config['port'],
            user='root',
            password=Path(config['password_file']).read_text(),
            db=config['database'],
            charset='utf8mb4',
            autocommit=autocommit,
        )

    connection = await connect()
    try:
        async with connection.cursor() as cursor:
            ddl = principal_trace_table_statement(SCOPE)
            await cursor.execute(ddl.replace(' shardkey=run_id', ''))
    finally:
        connection.close()
    return config, connect


async def _insert(connection, *, parent, run, record, timestamp=1789084800000000000, payload=None):
    async with connection.cursor() as cursor:
        await cursor.execute(
            f'INSERT INTO {TABLE} '
            '(run_id, run_span_id, parent_execution_id, record_id, event_type, time_unix_nano, '
            'trace_id, span_id, parent_span_id, event_data, created_at) '
            'VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
            (
                run,
                f'span-{run}',
                parent,
                record,
                'update',
                timestamp,
                'trace-fixture',
                'step-span',
                None,
                payload if payload is not None else '{ "attributes": {"text": "实时 ✓", "step": 42} }',
                datetime(2026, 9, 11, 0, 0, 0, 123456),
            ),
        )


async def _wait_for(check, process, *, timeout=20):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if process.returncode is not None:
            stdout, stderr = await process.communicate()
            raise AssertionError(f'export exited {process.returncode}: {stdout.decode()} {stderr.decode()}')
        result = check()
        if result:
            return result
        await asyncio.sleep(0.05)
    raise AssertionError('export did not make expected progress before timeout')


def _checkpoint(output):
    path = output.with_name(f'{output.name}.checkpoint.json')
    return json.loads(path.read_text()) if path.exists() else {}


async def _launch(
    config, server, token_file, output, parent, *, follow=True, resume=False, poll_interval=0.05, slow_writes=False
):
    command = [
        str(Path(sys.executable).with_name('nexus')),
        'export',
        '--execution-id',
        parent,
        '--content',
        'traces',
        '--control-url',
        str(server.make_url('')).rstrip('/'),
        '--taihu-token-file',
        str(token_file),
        '--db-host',
        config['host'],
        '--db-port',
        str(config['port']),
        '--db-name',
        config['database'],
        '--db-user',
        config.get('reader_user', 'root'),
        '--db-password-file',
        config.get('reader_password_file', config['password_file']),
        '--trace-table',
        TABLE,
        '-o',
        str(output),
    ]
    if follow:
        command += ['--follow']
        if poll_interval is not None:
            command += ['--poll-interval', str(poll_interval)]
    if resume:
        command += ['--resume']
    if slow_writes:
        script = """import time
from nexus.distributed.export import writer
original = writer._write_all
def slow(stream, chunk):
    if len(chunk) >= 16384:
        time.sleep(0.002)
    original(stream, chunk)
writer._write_all = slow
from nexus.cli.main import main
main()
"""
        command = [sys.executable, '-c', script, *command[1:]]
    return await asyncio.create_subprocess_exec(
        *command,
        cwd=ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


def _control_app(parent, state):
    async def execution(request):
        assert request.headers.get('Authorization') == 'Bearer local-fixture-token'
        assert request.match_info['execution_id'] == parent
        return web.json_response({
            'execution_id': parent,
            'spec_type': 'nexus_batch',
            'status': state['status'],
            'attempt_count': 1,
        })

    app = web.Application()
    app.router.add_get('/v1/executions/{execution_id}', execution)
    return app


async def test_real_cli_twenty_mib_records_resume_after_sigkill_during_write(database, tmp_path):
    config, connect = database
    nonce = uuid.uuid4().hex[:12]
    parent, child = f'large-{nonce}', f'child-{nonce}'
    token = tmp_path / 'token'
    token.write_text('local-fixture-token')
    token.chmod(0o600)
    output = tmp_path / 'events.jsonl'
    large = json.dumps({'attributes': {'large': 'x' * (20 * 1024 * 1024)}})
    connection = await connect()
    process = None
    try:
        await _insert(connection, parent=parent, run=child, record=1)
        async with TestServer(_control_app(parent, {'status': 'running'})) as server:
            process = await _launch(config, server, token, output, parent, slow_writes=True)
            await _wait_for(lambda: _checkpoint(output).get('committed_records') == 1, process)
            saved = _checkpoint(output)
            prefix = output.read_bytes()
            await _insert(connection, parent=parent, run=child, record=2, payload=large)
            await _wait_for(lambda: output.stat().st_size > saved['committed_bytes'] + 128 * 1024, process)
            assert _checkpoint(output)['committed_records'] == 1
            process.kill()
            assert await process.wait() == -signal.SIGKILL
            await _insert(connection, parent=parent, run=child, record=3)
            await _insert(connection, parent=parent, run=child, record=4, payload=large)
            await _insert(connection, parent=parent, run=child, record=5)
            process = await _launch(config, server, token, output, parent, resume=True)
            await _wait_for(lambda: _checkpoint(output).get('committed_records') == 5, process, timeout=45)
            process.send_signal(signal.SIGINT)
            await asyncio.wait_for(process.wait(), timeout=20)
            stdout, stderr = await process.communicate()
            assert process.returncode == 130, (stdout.decode(), stderr.decode())
        assert output.read_bytes().startswith(prefix)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert [row['record_id'] for row in rows] == [1, 2, 3, 4, 5]
        assert rows[1]['event_data'] == rows[3]['event_data'] == large
        assert all(len(row) == 12 for row in rows)
        assert _checkpoint(output)['version'] == 1
        async with connection.cursor() as cursor:
            await cursor.execute(
                f'SELECT {", ".join(COLUMNS)} FROM {TABLE} WHERE parent_execution_id=%s ORDER BY record_id', (parent,)
            )
            oracle = [dict(zip(COLUMNS, row, strict=True)) for row in await cursor.fetchall()]
        for row in oracle:
            row['created_at'] = row['created_at'].isoformat()
        assert rows == oracle
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        async with connection.cursor() as cursor:
            await cursor.execute(f'DELETE FROM {TABLE} WHERE parent_execution_id=%s', (parent,))
        connection.close()


async def test_real_cli_follows_transactions_discovers_late_stream_and_recovers_after_sigkill(database, tmp_path):
    config, connect = database
    nonce = uuid.uuid4().hex[:12]
    parent, early, late = f'batch-{nonce}', f'z-{nonce}', f'a-{nonce}'
    state = {'status': 'running'}
    token_file = tmp_path / 'token'
    token_file.write_text('local-fixture-token')
    token_file.chmod(0o600)
    output = tmp_path / 'events.jsonl'
    committed = await connect()
    pending = await connect(autocommit=False)
    process = None
    try:
        async with TestServer(_control_app(parent, state)) as server:
            process = await _launch(config, server, token_file, output, parent)
            await _wait_for(lambda: bool(_checkpoint(output)), process)
            await _insert(committed, parent=parent, run=early, record=1)
            await _insert(committed, parent=parent, run=early, record=5)
            await _wait_for(lambda: _checkpoint(output).get('committed_records') == 2, process)
            assert state['status'] == 'running'

            # Another stream commits while this transaction remains invisible.
            await _insert(pending, parent=parent, run=early, record=9)
            await _insert(committed, parent=parent, run=late, record=1, timestamp=1789084700000000000)
            await _wait_for(lambda: _checkpoint(output).get('committed_records') == 3, process)
            assert [json.loads(line)['record_id'] for line in output.read_text().splitlines()].count(9) == 0
            state['status'] = 'succeeded'
            await pending.commit()
            await _wait_for(lambda: _checkpoint(output).get('committed_records') == 4, process)
            await asyncio.sleep(0.15)
            assert process.returncode is None, 'terminal status must not terminate follow'

            saved = _checkpoint(output)
            prefix = output.read_bytes()[: saved['committed_bytes']]
            process.kill()
            assert await process.wait() == -signal.SIGKILL
            with output.open('ab') as stream:
                stream.write(b'{"interrupted_tail":')
            process = await _launch(config, server, token_file, output, parent, resume=True)
            await _wait_for(lambda: output.stat().st_size == saved['committed_bytes'], process)
            assert output.read_bytes() == prefix
            await _insert(committed, parent=parent, run=late, record=6)
            await _wait_for(lambda: _checkpoint(output).get('committed_records') == 5, process)
            process.send_signal(signal.SIGINT)
            await asyncio.wait_for(process.wait(), timeout=15)
            stdout, stderr = await process.communicate()
            assert process.returncode == 130, (stdout.decode(), stderr.decode())

        rows = [json.loads(line) for line in output.read_text().splitlines()]
        async with committed.cursor() as cursor:
            await cursor.execute(f'SELECT {", ".join(COLUMNS)} FROM {TABLE} WHERE parent_execution_id=%s', (parent,))
            expected = [dict(zip(COLUMNS, row, strict=True)) for row in await cursor.fetchall()]
        for row in expected:
            row['created_at'] = row['created_at'].isoformat()

        def key(row):
            return row['run_id'], row['run_span_id'], row['record_id']

        assert sorted(rows, key=key) == sorted(expected, key=key)
        assert len({key(row) for row in rows}) == 5
        assert output.read_bytes().startswith(prefix)
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        await pending.rollback()
        pending.close()
        async with committed.cursor() as cursor:
            await cursor.execute(f'DELETE FROM {TABLE} WHERE parent_execution_id=%s', (parent,))
        committed.close()


async def test_real_cli_exports_running_snapshot_without_resumable_checkpoint(database, tmp_path):
    config, connect = database
    nonce = uuid.uuid4().hex[:12]
    parent = f'snapshot-{nonce}'
    connection = await connect()
    token_file = tmp_path / 'token'
    token_file.write_text('local-fixture-token')
    token_file.chmod(0o600)
    output = tmp_path / 'events.jsonl'
    process = None
    try:
        await _insert(connection, parent=parent, run=f'child-{nonce}', record=1)
        await _insert(connection, parent=parent, run=f'child-{nonce}', record=7)
        async with TestServer(_control_app(parent, {'status': 'running'})) as server:
            process = await _launch(config, server, token_file, output, parent, follow=False)
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=20)
            assert process.returncode == 0, (stdout.decode(), stderr.decode())
        assert [json.loads(line)['record_id'] for line in output.read_text().splitlines()] == [1, 7]
        assert not output.with_name(f'{output.name}.checkpoint.json').exists()
        manifest = json.loads(output.with_name(f'{output.name}.manifest.json').read_text())
        assert manifest['status'] == 'snapshot'
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        async with connection.cursor() as cursor:
            await cursor.execute(f'DELETE FROM {TABLE} WHERE parent_execution_id=%s', (parent,))
        connection.close()


async def test_raw_discovery_preserves_all_mixed_case_streams_with_one_row_pages(database):
    config, connect = database
    parent = f'case-{uuid.uuid4().hex[:12]}'
    connection = await connect()
    source = TDSQLTraceSource(
        host=config['host'],
        port=config['port'],
        user=config.get('reader_user', 'root'),
        password=Path(config.get('reader_password_file', config['password_file'])).read_text(),
        db=config['database'],
    )
    try:
        async with connection.cursor() as cursor:
            for run_span in ('a', 'B', 'Z', 'c'):
                await cursor.execute(
                    f'INSERT INTO {TABLE} '
                    '(run_id, run_span_id, parent_execution_id, record_id, event_type, event_data) '
                    'VALUES (%s,%s,%s,1,%s,%s)',
                    (parent, run_span, parent, 'start', '{}'),
                )
        after = None
        seen = []
        for _ in range(5):
            rows = await source.discover_raw_streams(
                {'storage_scope': 'principal_v1', 'principal_scope_id': SCOPE, 'parent_execution_id': parent},
                after=after,
                limit=1,
            )
            if not rows:
                break
            assert len(rows) == 1
            seen.append(rows[0]['run_span_id'])
            after = rows[-1]['run_id'], rows[-1]['run_span_id']
        assert sorted(seen) == ['B', 'Z', 'a', 'c']
    finally:
        async with connection.cursor() as cursor:
            await cursor.execute(f'DELETE FROM {TABLE} WHERE parent_execution_id=%s', (parent,))
        connection.close()


@pytest.mark.parametrize('crash_after_replace', [False, True])
async def test_writer_recovers_when_process_dies_at_checkpoint_replace(tmp_path, crash_after_replace):
    output = tmp_path / 'events.jsonl'
    marker = tmp_path / 'paused'
    source = {
        'control_url': 'http://127.0.0.1',
        'execution_id': 'batch-probe',
        'spec_type': 'nexus_batch',
        'db_host': '127.0.0.1',
        'db_port': 3306,
        'db_name': 'trace_fixture',
        'table': TABLE,
        'storage_scope': 'principal_v1',
        'content': 'traces',
        'schema_version': 1,
    }
    first = {
        'id': 1,
        'run_id': 'child-probe',
        'run_span_id': 'span-probe',
        'parent_execution_id': 'batch-probe',
        'record_id': 1,
        'event_type': 'update',
        'time_unix_nano': 1789084800000000001,
        'trace_id': 'trace-probe',
        'span_id': 'span-probe',
        'parent_span_id': None,
        'event_data': '{ "attributes": {"text": "实时 ✓"} }',
        'created_at': '2026-09-11T00:00:00.123456',
    }
    script = """
import asyncio, json, os, sys, threading
from pathlib import Path
from nexus.distributed.export_output import JsonlEventWriter
from nexus.distributed.export_types import TraceStreamKey
output, marker = map(Path, sys.argv[1:3])
source, first, after = json.loads(sys.argv[3]), json.loads(sys.argv[4]), json.loads(sys.argv[5])
replace = os.replace
checkpoint = output.with_name(output.name + '.checkpoint.json')
def crashing_replace(src, dst):
    if Path(dst) == checkpoint and output.exists() and output.stat().st_size:
        if after:
            replace(src, dst)
        marker.write_text('checkpoint boundary reached')
        threading.Event().wait(30)
        raise AssertionError('parent did not kill process at checkpoint boundary')
    return replace(src, dst)
os.replace = crashing_replace
async def main():
    writer = JsonlEventWriter(output, source=source)
    await writer.open()
    key = TraceStreamKey('principal_v1', source['table'], first['run_id'], first['run_span_id'])
    await writer.commit_page(key, expected_cursor=0, rows=[first])
    await writer.close(reason='interrupted')
asyncio.run(main())
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        '-c',
        script,
        str(output),
        str(marker),
        json.dumps(source),
        json.dumps(first),
        json.dumps(crash_after_replace),
        cwd=ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        await _wait_for(marker.exists, process)
        process.kill()
        assert await process.wait() == -signal.SIGKILL
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()

    state = _checkpoint(output)
    assert state['committed_records'] == (1 if crash_after_replace else 0)
    writer = JsonlEventWriter(output, source=source, resume=True)
    await writer.open()
    key = TraceStreamKey('principal_v1', TABLE, 'child-probe', 'span-probe')
    try:
        if not crash_after_replace:
            assert writer.cursor_for(key) == 0
            assert output.read_bytes() == b''
            await writer.commit_page(key, expected_cursor=0, rows=[first])
        else:
            assert writer.cursor_for(key) == 1
        second = {**first, 'id': 2, 'record_id': 7}
        await writer.commit_page(key, expected_cursor=1, rows=[second])
    finally:
        await writer.close(reason='interrupted')
    assert [json.loads(line) for line in output.read_text().splitlines()] == [first, second]
    assert _checkpoint(output)['committed_records'] == 2


async def test_real_raw_source_and_snapshot_writer_freeze_upper_bound_and_preserve_payload(database, tmp_path):
    config, connect = database
    parent = f'upper-{uuid.uuid4().hex[:12]}'
    run = f'child-{parent}'
    connection = await connect()
    source = TDSQLTraceSource(
        host=config['host'],
        port=config['port'],
        user=config.get('reader_user', 'root'),
        password=Path(config.get('reader_password_file', config['password_file'])).read_text(),
        db=config['database'],
    )
    metadata = {'storage_scope': 'batch_principal_v1', 'principal_scope_id': SCOPE, 'parent_execution_id': parent}
    writer = JsonlEventWriter(
        tmp_path / 'snapshot.jsonl',
        follow=False,
        source={
            'control_url': 'http://127.0.0.1',
            'execution_id': parent,
            'spec_type': 'nexus_batch',
            'db_host': config['host'],
            'db_port': config['port'],
            'db_name': config['database'],
            'table': TABLE,
            'storage_scope': 'principal_v1',
            'content': 'traces',
            'schema_version': 1,
        },
    )
    try:
        await _insert(connection, parent=parent, run=run, record=1)
        await _insert(connection, parent=parent, run=run, record=5)
        streams = await source.discover_raw_streams(metadata, limit=1)
        assert streams[0]['max_record_id'] == 5
        await _insert(connection, parent=parent, run=run, record=9)
        rows = await source.read_raw_events_after(
            {**metadata, 'run_id': run, 'run_span_id': f'span-{run}'},
            0,
            limit=500,
            through_record_id=streams[0]['max_record_id'],
        )
        assert [row['record_id'] for row in rows] == [1, 5]
        assert all(row['event_data'] == '{ "attributes": {"text": "实时 ✓", "step": 42} }' for row in rows)
        assert all(row['time_unix_nano'] == 1789084800000000000 for row in rows)
        assert all(row['created_at'] == '2026-09-11T00:00:00.123456' for row in rows)
        assert all(set(row) == set(COLUMNS) for row in rows)
        await writer.open()
        key = TraceStreamKey('batch_principal_v1', TABLE, run, f'span-{run}')
        await writer.commit_page(key, expected_cursor=0, rows=rows)
        await writer.close(reason='snapshot')
        assert [json.loads(line) for line in writer.output.read_text().splitlines()] == rows
        assert not writer.checkpoint.exists()
    finally:
        await writer.close(reason='error')
        async with connection.cursor() as cursor:
            await cursor.execute(f'DELETE FROM {TABLE} WHERE parent_execution_id=%s', (parent,))
        connection.close()


async def test_known_stream_commit_ack_to_checkpoint_latency_at_default_poll_interval(database, tmp_path):
    config, connect = database
    parent = f'latency-{uuid.uuid4().hex[:12]}'
    run = f'child-{parent}'
    connection = await connect(autocommit=False)
    process = None
    token_file = tmp_path / 'token'
    token_file.write_text('local-fixture-token')
    token_file.chmod(0o600)
    output = tmp_path / 'latency.jsonl'
    source = TDSQLTraceSource(
        host=config['host'],
        port=config['port'],
        user=config.get('reader_user', 'root'),
        password=Path(config.get('reader_password_file', config['password_file'])).read_text(),
        db=config['database'],
    )
    identity = {
        'control_url': 'http://127.0.0.1',
        'execution_id': parent,
        'spec_type': 'nexus_batch',
        'db_host': config['host'],
        'db_port': config['port'],
        'db_name': config['database'],
        'table': TABLE,
        'storage_scope': 'principal_v1',
        'content': 'traces',
        'schema_version': 1,
    }
    metadata = {
        'storage_scope': 'batch_principal_v1',
        'principal_scope_id': SCOPE,
        'parent_execution_id': parent,
        'run_id': run,
        'run_span_id': f'span-{run}',
    }
    try:
        await _insert(connection, parent=parent, run=run, record=1)
        await connection.commit()
        baseline_writer = JsonlEventWriter(tmp_path / 'baseline.jsonl', source=identity)
        await baseline_writer.open()
        try:
            baseline_started = time.monotonic()
            baseline_rows = await source.read_raw_events_after(metadata, 0, limit=500)
            await baseline_writer.commit_page(
                TraceStreamKey('batch_principal_v1', TABLE, run, f'span-{run}'),
                expected_cursor=0,
                rows=baseline_rows,
            )
            io_baseline_seconds = time.monotonic() - baseline_started
        finally:
            await baseline_writer.close(reason='interrupted')
        async with TestServer(_control_app(parent, {'status': 'running'})) as server:
            # Omit the option to exercise the CLI's actual default (one second).
            process = await _launch(config, server, token_file, output, parent, poll_interval=None)
            await _wait_for(lambda: _checkpoint(output).get('committed_records') == 1, process)
            await _insert(connection, parent=parent, run=run, record=7)
            await connection.commit()
            commit_acknowledged = time.monotonic()
            await _wait_for(lambda: _checkpoint(output).get('committed_records') == 2, process)
            checkpoint_observed = time.monotonic()
            latency_seconds = checkpoint_observed - commit_acknowledged
            bound_seconds = 3 * 1.0 + io_baseline_seconds
            measurement = {
                'scope': 'local MySQL known single stream; not a Cloud SLA',
                'poll_interval_seconds': 1.0,
                'commit_ack_to_checkpoint_seconds': latency_seconds,
                'independent_read_and_file_fsync_seconds': io_baseline_seconds,
                'bound_seconds': bound_seconds,
            }
            (tmp_path / 'latency-measurement.json').write_text(json.dumps(measurement, indent=2))
            print('EXPORT_LATENCY_MEASUREMENT ' + json.dumps(measurement), flush=True)
            assert latency_seconds <= bound_seconds
            assert [json.loads(line)['record_id'] for line in output.read_text().splitlines()] == [1, 7]
            process.send_signal(signal.SIGINT)
            await asyncio.wait_for(process.wait(), 15)
            assert process.returncode == 130
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        await connection.rollback()
        async with connection.cursor() as cursor:
            await cursor.execute(f'DELETE FROM {TABLE} WHERE parent_execution_id=%s', (parent,))
        await connection.commit()
        connection.close()
