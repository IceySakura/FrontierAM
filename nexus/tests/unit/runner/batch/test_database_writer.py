from __future__ import annotations

import asyncio
import sqlite3

import pytest

from nexus.runner.batch import database as database_module
from nexus.runner.batch.database import SCHEMA, SCHEMA_VERSION, BatchDatabase


def _count_rows(db_path, table: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
    finally:
        conn.close()


async def test_open_migrated_database_does_not_take_writer_lock(tmp_path, monkeypatch):
    monkeypatch.delenv('NEXUS_SQLITE_NO_WAL', raising=False)
    monkeypatch.setattr(database_module, '_SQLITE_BUSY_TIMEOUT_MS', 50)
    path = tmp_path / 'busy.db'
    async with BatchDatabase(path) as initial:
        await initial.insert_run('existing', {'run_id': 'existing'}, {})

    writer = sqlite3.connect(path)
    reader = BatchDatabase(path)
    try:
        writer.execute('BEGIN IMMEDIATE')
        writer.execute("UPDATE runs SET status='running' WHERE run_id='existing'")
        await asyncio.wait_for(reader.connect(), timeout=2)
        assert await reader.get_run('existing') is not None
    finally:
        writer.rollback()
        writer.close()
        await reader.close()


async def test_concurrent_old_database_migrations_preserve_records(tmp_path, monkeypatch):
    monkeypatch.delenv('NEXUS_SQLITE_NO_WAL', raising=False)
    path = tmp_path / 'concurrent-old.db'
    with sqlite3.connect(path) as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.executescript('\n'.join(line for line in SCHEMA.splitlines() if 'judge_output' not in line))
        conn.execute('INSERT INTO schema_version VALUES (?)', (SCHEMA_VERSION,))
        conn.execute("INSERT INTO runs (run_id, status) VALUES ('old', 'success')")
    readers = [BatchDatabase(path) for _ in range(4)]
    try:
        await asyncio.gather(*(reader.connect() for reader in readers))
        for reader in readers:
            assert (await reader.get_run('old'))['judge_output'] is None
    finally:
        await asyncio.gather(*(reader.close() for reader in readers))
    with sqlite3.connect(path) as conn:
        columns = [row[1] for row in conn.execute('PRAGMA table_info(runs)')]
        assert columns.count('judge_output') == 1
        assert conn.execute('SELECT run_id, status FROM runs').fetchall() == [('old', 'success')]


@pytest.mark.parametrize('cancel', [False, True])
async def test_failed_connect_closes_connection_and_clears_state(tmp_path, monkeypatch, cancel):
    db = BatchDatabase(tmp_path / 'failed-connect.db')
    entered = asyncio.Event()
    connections = []

    async def failed_schema():
        connections.append(db.conn)
        entered.set()
        if cancel:
            await asyncio.Event().wait()
        raise RuntimeError('schema initialization failed')

    monkeypatch.setattr(db, '_init_schema', failed_schema)
    task = asyncio.create_task(db.connect())
    await entered.wait()
    if cancel:
        task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
            await task
        assert db.conn is None
        assert db._writer_task is None
        assert db._loop is None
        with pytest.raises(ValueError, match='no active connection'):
            await connections[0].execute('SELECT 1')
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_existing_v2_database_gets_nullable_judge_column_without_changing_old_results(tmp_path):
    db_path = tmp_path / 'old.db'
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript('\n'.join(line for line in SCHEMA.splitlines() if 'judge_output' not in line))
        conn.execute('INSERT INTO schema_version VALUES (?)', (SCHEMA_VERSION,))
        conn.execute(
            'INSERT INTO runs (run_id, status, task_output, agent_output) VALUES (?, ?, ?, ?)',
            ('old', 'success', '{"score":1}', '{"exit_status":"finished"}'),
        )
        original = dict(conn.execute('SELECT * FROM runs').fetchone())

    # Reopening a resumed DB is idempotent; absence of old judge data stays NULL.
    for _ in range(2):
        async with BatchDatabase(db_path) as db:
            assert (await db.get_run('old'))['judge_output'] is None
            assert (await db.get_all_runs())[0]['task_output'] == {'score': 1}
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        migrated = dict(conn.execute('SELECT * FROM runs').fetchone())
    assert migrated.pop('judge_output') is None
    assert migrated == original

    async with BatchDatabase(db_path) as db:
        await db.update_run_status('old', 'success', judge_output={'exit_status': 'finished', 'exit_code': 0})
        assert (await db.get_all_runs())[0]['judge_output'] == {'exit_status': 'finished', 'exit_code': 0}


@pytest.mark.asyncio
@pytest.mark.parametrize('no_wal', [False, True])
async def test_low_priority_trace_and_log_writes_flush_on_close(tmp_path, monkeypatch, no_wal):
    db_path = tmp_path / 'batch.db'
    if no_wal:
        monkeypatch.setenv('NEXUS_SQLITE_NO_WAL', '1')
    else:
        monkeypatch.delenv('NEXUS_SQLITE_NO_WAL', raising=False)

    async with BatchDatabase(db_path) as db:
        await db.insert_run('run-1', {'run_id': 'run-1'}, {'case': 'flush'})

        cursor = await db.conn.execute('PRAGMA journal_mode')
        assert (await cursor.fetchone())[0].lower() == ('delete' if no_wal else 'wal')
        if no_wal:
            assert not db_path.with_name(db_path.name + '-shm').exists()

        assert db.enqueue_trace_record(
            run_id='run-1',
            event_type='start',
            time_unix_nano=1,
            trace_id='trace-1',
            span_id='span-1',
            parent_span_id=None,
            event_data='{"event": "start"}',
        )
        assert db.enqueue_log_chunk(
            run_id='run-1',
            chunk_index=0,
            stream_type='log',
            start_time='2026-06-11T00:00:00+00:00',
            end_time='2026-06-11T00:00:01+00:00',
            content='hello\n',
        )

    assert _count_rows(db_path, 'run_trace_events') == 1
    assert _count_rows(db_path, 'run_logs') == 1


@pytest.mark.asyncio
async def test_async_low_priority_insert_waits_until_record_is_queryable(tmp_path):
    db_path = tmp_path / 'batch.db'

    class SpanStartEvent:
        time_unix_nano = 1
        trace_id = 'trace-1'
        span_id = 'span-1'
        parent_span_id = None

        def model_dump(self):
            return {'name': 'root'}

    async with BatchDatabase(db_path) as db:
        await db.insert_run('run-1', {'run_id': 'run-1'}, {})
        await db.insert_trace_event('run-1', SpanStartEvent())

        events = await db.get_trace_events('run-1')

    assert len(events) == 1
    assert events[0]['event_type'] == 'start'


@pytest.mark.asyncio
async def test_batch_database_sets_sqlite_write_pragmas(tmp_path, monkeypatch):
    monkeypatch.delenv('NEXUS_SQLITE_NO_WAL', raising=False)
    db_path = tmp_path / 'batch.db'

    async with BatchDatabase(db_path) as db:
        assert db.conn is not None

        cursor = await db.conn.execute('PRAGMA busy_timeout')
        assert (await cursor.fetchone())[0] == 30_000

        cursor = await db.conn.execute('PRAGMA journal_mode')
        assert (await cursor.fetchone())[0].lower() == 'wal'

        cursor = await db.conn.execute('PRAGMA synchronous')
        assert (await cursor.fetchone())[0] == 1

        cursor = await db.conn.execute('PRAGMA wal_autocheckpoint')
        assert (await cursor.fetchone())[0] == 1000
