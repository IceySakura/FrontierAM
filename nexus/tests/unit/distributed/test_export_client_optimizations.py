from __future__ import annotations

import asyncio
import json
import time

from pathlib import Path
from typing import Any

import pytest

from nexus.distributed.export import results
from nexus.distributed.export.follow import AmbiguousTraceBackendError, resolve_trace_read_scope
from nexus.distributed.export.types import (
    ExportDeadlineExceeded,
    ExportError,
    TraceDatabaseAuth,
    TraceDatabaseTimeout,
    TraceDatabaseUnavailable,
    TraceProcessingTimeout,
)


TABLE = 'trajectory_span_events_pr_' + 'a' * 32
PARTITION = '2026-09-28 00:00:00.000000'


class ProbeReader:
    """A plain reader double: no TdsqlTraceReader cache, so the export must cache."""

    def __init__(self, *, boundless: bool = True, output: Any = None) -> None:
        self.table = TABLE
        self.parent_created_at: str | None = None
        self.trace_backend: str | None = None
        self.boundless = boundless
        self.probes = 0
        self.partitions: list[str | None] = []
        self.output = output if output is not None else {'success': True, 'task_output': {'value': 1}}

    async def has_partition_column(self) -> bool:
        self.probes += 1
        return self.boundless

    async def read_many(self, ids):
        if not ids:
            return {}
        self.partitions.append(self.parent_created_at)
        return {run_id: results.TraceResult(output=self.output) for run_id in ids}


class BatchApi:
    def __init__(self, count: int) -> None:
        self.count = count
        self.parent = {
            'execution_id': 'batch',
            'spec_type': 'nexus_batch',
            'status': 'succeeded',
            'attempt_count': 1,
            'created_at': '2026-09-28T00:00:00',
        }
        self.result_calls = 0
        self.execution_calls = 0

    async def get_execution(self, execution_id: str) -> dict[str, Any]:
        self.execution_calls += 1
        return dict(self.parent)

    async def get_batch_summary(self, execution_id: str) -> dict[str, Any]:
        return {
            'generation_status': 'complete',
            'submission_closed': True,
            'summary_version': 1,
            'materialized_count': self.count,
            'total_items': self.count,
            'terminal_items': self.count,
            'item_status_counts': {'succeeded': self.count},
        }

    async def get_batch_items(self, execution_id: str, cursor: str | None) -> dict[str, Any]:
        return {
            'items': [
                {
                    'item_id': f'item-{index}',
                    'status': 'succeeded',
                    'current_attempt_no': 0,
                    'current_child_execution_id': f'child-{index}',
                }
                for index in range(self.count)
            ],
            'next_cursor': None,
        }

    async def get_control_events(self, execution_id: str) -> list[Any]:
        return []

    async def get_result(self, execution_id: str) -> Any:
        self.result_calls += 1
        return None


async def test_reverse_scan_keeps_the_max_key_from_the_newest_row() -> None:
    records = [
        (1, 100, _event({'success': True, 'task_output': {'value': 'old'}})),
        (2, 200, _event({'success': True, 'task_output': {'value': 'middle'}})),
        (3, 300, _event({'success': True, 'task_output': {'value': 'new'}})),
    ]
    database = RootDatabase(records)
    reader = _root_reader(database)
    result = (await reader.read_many(['run'], include_timestamps=False))['run']
    assert result.output == {'success': True, 'task_output': {'value': 'new'}}
    body_queries = [sql for sql, _ in database.queries if sql.startswith('SELECT event_data')]
    assert len(body_queries) == 1


async def test_reverse_scan_returns_none_for_the_newest_null_output() -> None:
    records = [
        (1, 100, _event({'success': True, 'task_output': {'old': True}})),
        (2, 200, _event(None)),
    ]
    database = RootDatabase(records)
    reader = _root_reader(database)
    result = (await reader.read_many(['run'], include_timestamps=False))['run']
    assert result.output is None
    assert not any('ORDER BY record_id LIMIT' in sql for sql, _ in database.queries)


async def test_reverse_window_then_older_scan_reads_each_body_once(monkeypatch) -> None:
    monkeypatch.setenv('NEXUS_EXPORT_ROOT_SCAN_N', '2')
    records = [
        (1, 100, _event({'success': True, 'task_output': {'old': True}})),
        (2, 200, _no_output_event()),
        (3, 300, _no_output_event()),
        (4, 400, _no_output_event()),
        (5, 500, _no_output_event()),
    ]
    database = RootDatabase(records)
    reader = _root_reader(database)
    result = (await reader.read_many(['run'], include_timestamps=False))['run']
    assert result.output == {'success': True, 'task_output': {'old': True}}
    body_queries = [sql for sql, _ in database.queries if sql.startswith('SELECT event_data')]
    # Window (records 4-5) plus older records (1-3); never window twice.
    assert len(body_queries) == 5


async def test_reverse_scan_stops_after_first_output_when_older_records_cannot_win(monkeypatch) -> None:
    monkeypatch.setenv('NEXUS_EXPORT_ROOT_SCAN_N', '2')
    records = [
        (1, 100, _event({'success': True, 'task_output': {'value': 'old'}})),
        (2, 200, _no_output_event()),
        (3, 300, _event({'success': True, 'task_output': {'value': 'middle'}})),
        (4, 400, _no_output_event()),
    ]
    database = RootDatabase(records)
    reader = _root_reader(database)
    result = (await reader.read_many(['run'], include_timestamps=False))['run']
    assert result.output == {'success': True, 'task_output': {'value': 'middle'}}
    body_queries = [sql for sql, _ in database.queries if sql.startswith('SELECT event_data')]
    assert len(body_queries) == 2  # records 4 and 3 only
    assert any(sql.startswith('SELECT MAX(time_unix_nano)') for sql, _ in database.queries)


async def test_reverse_scan_continues_older_when_max_time_can_win(monkeypatch) -> None:
    monkeypatch.setenv('NEXUS_EXPORT_ROOT_SCAN_N', '2')
    records = [
        (1, 500, _event({'success': True, 'task_output': {'value': 'future'}})),
        (2, 200, _no_output_event()),
        (3, 300, _event({'success': True, 'task_output': {'value': 'middle'}})),
        (4, 400, _no_output_event()),
    ]
    database = RootDatabase(records)
    reader = _root_reader(database)
    result = (await reader.read_many(['run'], include_timestamps=False))['run']
    assert result.output == {'success': True, 'task_output': {'value': 'future'}}


async def test_reverse_scan_falls_back_on_non_monotonic_rows() -> None:
    records = [
        (1, 200, _event({'success': True, 'task_output': {'old': True}})),
        (2, 100, _event({'success': True, 'task_output': {'new': True}})),
    ]
    database = RootDatabase(records)
    reader = _root_reader(database)
    result = (await reader.read_many(['run'], include_timestamps=False))['run']
    # Full walk keeps the legacy (timestamp, span, record) ordering.
    assert result.output == {'success': True, 'task_output': {'old': True}}
    assert any('ORDER BY record_id LIMIT' in sql for sql, _ in database.queries)


async def test_full_root_scan_mode_never_issues_the_reverse_query(monkeypatch) -> None:
    monkeypatch.setenv('NEXUS_EXPORT_ROOT_SCAN', 'full')
    records = [(1, 100, _event({'success': True, 'task_output': {'value': 1}}))]
    database = RootDatabase(records)
    reader = _root_reader(database)
    result = (await reader.read_many(['run'], include_timestamps=False))['run']
    assert result.output == {'success': True, 'task_output': {'value': 1}}
    assert not any('ORDER BY record_id DESC' in sql for sql, _ in database.queries)


async def test_reverse_window_size_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv('NEXUS_EXPORT_ROOT_SCAN_N', '1')
    records = [
        (1, 100, _event({'success': True, 'task_output': {'old': True}})),
        (2, 200, _event({'success': True, 'task_output': {'new': True}})),
    ]
    database = RootDatabase(records)
    reader = _root_reader(database)
    result = (await reader.read_many(['run'], include_timestamps=False))['run']
    assert result.output == {'success': True, 'task_output': {'new': True}}
    body_queries = [sql for sql, _ in database.queries if sql.startswith('SELECT event_data')]
    assert len(body_queries) == 1


async def test_batch_probe_is_shared_by_all_items(tmp_path: Path) -> None:
    reader = ProbeReader()
    api = BatchApi(10)
    await results.export_execution(
        api=api, execution_id='batch', output=tmp_path / 'out.jsonl', force=False, trace_reader=reader
    )
    assert reader.probes == 1
    assert reader.partitions == [PARTITION] * 10
    assert api.result_calls == 0
    # Expansion plus the final snapshot recheck; no per-item child query.
    assert api.execution_calls == 2


async def test_batch_child_must_be_single(tmp_path: Path) -> None:
    class Api(BatchApi):
        async def get_execution(self, execution_id: str) -> dict[str, Any]:
            if execution_id == 'child-0':
                return {
                    'execution_id': 'child-0',
                    'spec_type': 'nexus_live',
                    'status': 'succeeded',
                    'attempt_count': 1,
                    'parent_execution_id': 'batch',
                    'created_at': '2026-09-28T00:00:00',
                }
            return await super().get_execution(execution_id)

    with pytest.raises(ExportError, match='Single execution'):
        await results.export_execution(
            api=Api(1),
            execution_id='batch',
            output=tmp_path / 'out.jsonl',
            force=False,
            allow_partial=True,
            result_status='success',
        )


async def test_probe_uses_a_short_timeout_and_caches_failure(monkeypatch) -> None:
    monkeypatch.setattr(results, '_TDSQL_PROBE_TIMEOUT_SECONDS', 0.01)
    attempts = 0

    class Cursor:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def execute(self, sql: str, params: tuple[Any, ...]) -> None:
            await asyncio.sleep(1)

        def fetchall(self) -> list[tuple[Any, ...]]:
            return []

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor()

        def close(self) -> None:
            return None

    async def connect(**_kwargs: Any) -> Connection:
        nonlocal attempts
        attempts += 1
        return Connection()

    reader = results.TdsqlTraceReader(
        host='db.test',
        port=3306,
        user='reader',
        password='unused',
        database='trace',
        table=TABLE,
        connect=connect,
        transient_errors=(),
    )
    assert await reader.has_partition_column_cached() is None
    assert await reader.has_partition_column_cached() is None
    assert attempts == 1


async def test_probe_failure_warns_and_keeps_the_unpruned_read(caplog) -> None:
    reader = ProbeReader()

    async def fail() -> bool:
        raise OSError('probe unavailable')

    reader.has_partition_column = fail  # type: ignore[method-assign]
    execution = {
        'execution_id': 'exe',
        'spec_type': 'nexus_single',
        'created_at': '2026-09-29T01:00:00',
    }
    with caplog.at_level('WARNING'):
        scoped = await results._execution_trace_reader(object(), reader, execution)
    assert scoped is reader
    assert 'partition pruning is disabled' in caplog.text


async def test_probe_failure_with_explicit_key_is_fail_closed() -> None:
    reader = ProbeReader()
    reader.parent_created_at = PARTITION

    async def fail() -> bool:
        raise OSError('probe unavailable')

    reader.has_partition_column = fail  # type: ignore[method-assign]
    execution = {
        'execution_id': 'exe',
        'spec_type': 'nexus_single',
        'created_at': '2026-09-29T01:00:00',
    }
    with pytest.raises(ExportError, match='Cannot determine'):
        await results._execution_trace_reader(object(), reader, execution)


async def test_transient_db_failure_aggregates_api_fallback_error() -> None:
    reader = ProbeReader()
    reader.table = None

    async def read_many(ids):
        raise TraceDatabaseTimeout('db timed out')

    reader.read_many = read_many  # type: ignore[method-assign]

    class Api:
        async def get_control_events(self, execution_id: str) -> list[Any]:
            return []

        async def get_result(self, execution_id: str) -> Any:
            raise RuntimeError('api down')

    single = results._SingleExport(
        execution_id='exe',
        run_id='exe',
        status='succeeded',
        retry_count=0,
        context={'execution_id': 'exe', 'spec_type': 'nexus_single'},
    )
    with pytest.raises(ExportError, match='db timed out.*api down'):
        await results._single_record(api=Api(), single=single, trace_reader=reader, trace_db_fallback='api')


async def test_explicit_partition_on_classic_table_is_rejected(tmp_path: Path) -> None:
    reader = ProbeReader(boundless=False)
    reader.parent_created_at = PARTITION

    class Api:
        async def get_execution(self, execution_id: str) -> dict[str, Any]:
            return {
                'execution_id': execution_id,
                'spec_type': 'nexus_single',
                'status': 'succeeded',
                'attempt_count': 1,
                'created_at': '2026-09-28T00:00:00',
            }

        async def get_control_events(self, execution_id: str) -> list[Any]:
            return []

        async def get_result(self, execution_id: str) -> Any:
            pytest.fail('no result should be read')

    with pytest.raises(ExportError, match='non-partitioned'):
        await results.export_execution(
            api=Api(), execution_id='exe', output=tmp_path / 'out.jsonl', force=False, trace_reader=reader
        )


async def test_live_uses_its_own_created_at_not_the_parent() -> None:
    class Api:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def get_execution(self, execution_id: str) -> dict[str, Any]:
            self.calls.append(execution_id)
            return {
                'execution_id': execution_id,
                'spec_type': 'nexus_batch',
                'created_at': '2026-01-01T00:00:00',
            }

    api = Api()
    reader = ProbeReader()
    execution = {
        'execution_id': 'live',
        'spec_type': 'nexus_live',
        'parent_execution_id': 'batch',
        'created_at': '2026-09-29T01:00:00',
    }
    scoped = await results._execution_trace_reader(api, reader, execution)
    assert scoped is not reader
    assert scoped.parent_created_at == '2026-09-29 01:00:00.000000'
    assert api.calls == []


async def test_partition_errors_distinguish_parent_lookup_from_missing_created_at() -> None:
    async def probe() -> bool:
        return True

    execution = {
        'execution_id': 'child',
        'spec_type': 'nexus_single',
        'parent_execution_id': 'parent',
    }
    with pytest.raises(ExportError, match='parent execution lookup'):
        await resolve_trace_read_scope(
            execution,
            'nexus_single',
            TABLE,
            probe=probe,
        )

    async def fetch_parent(execution_id: str) -> dict[str, Any]:
        return {'execution_id': execution_id}

    with pytest.raises(ExportError, match='created_at for the parent execution'):
        await resolve_trace_read_scope(
            execution,
            'nexus_single',
            TABLE,
            probe=probe,
            fetch_execution=fetch_parent,
        )

    with pytest.raises(ExportError, match='created_at for the execution'):
        await resolve_trace_read_scope(
            {'execution_id': 'live', 'spec_type': 'nexus_live'},
            'nexus_live',
            TABLE,
            probe=probe,
        )


async def test_ambiguous_backend_requires_trace_backend() -> None:
    execution = {
        'execution_id': 'exe',
        'spec_type': 'nexus_single',
        'runtime_info': {
            '_artifacts': {
                'single': {
                    'sinks': [
                        {
                            'name': 'classic',
                            'metadata': {
                                'storage_scope': 'principal_v1',
                                'principal_scope_id': 'a' * 32,
                            },
                        },
                        {
                            'name': 'boundless',
                            'metadata': {
                                'storage_scope': 'principal_v1',
                                'principal_scope_id': 'a' * 32,
                                'parent_created_at': '2026-09-28T00:00:00',
                            },
                        },
                    ]
                }
            }
        },
    }
    reader = ProbeReader()
    with pytest.raises(AmbiguousTraceBackendError, match='several backends'):
        await results._execution_trace_reader(object(), reader, execution)

    reader.trace_backend = 'classic'
    assert await results._execution_trace_reader(object(), reader, execution) is reader
    assert reader.probes == 0

    reader.trace_backend = 'not-published'
    with pytest.raises(ExportError, match='no trace sink named'):
        await results._execution_trace_reader(object(), reader, execution)


class _Rows:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows


async def test_statement_timeout_does_not_cover_local_fetchall() -> None:
    class Cursor:
        def __init__(self) -> None:
            self.rows: list[tuple[Any, ...]] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def execute(self, sql: str, params: tuple[Any, ...]) -> None:
            self.rows = [(1,)]

        def fetchall(self) -> list[tuple[Any, ...]]:
            time.sleep(0.02)
            return self.rows

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor()

    rows = await results._fetch_rows(Connection(), 'SELECT 1', (), timeout=0.001)
    assert rows == [(1,)]


async def test_statement_timeout_is_raised_only_by_execute() -> None:
    class Cursor:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def execute(self, sql: str, params: tuple[Any, ...]) -> None:
            await asyncio.sleep(0.1)

        def fetchall(self) -> list[tuple[Any, ...]]:
            return []

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor()

    with pytest.raises(TraceDatabaseTimeout):
        await results._fetch_rows(Connection(), 'SELECT 1', (), timeout=0.001)


async def test_execute_timeout_is_retried_and_classified_for_fallback(monkeypatch) -> None:
    monkeypatch.setattr(results, '_TDSQL_TIMEOUT_SECONDS', 0.005)
    attempts = 0

    class Cursor:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def execute(self, sql: str, params: tuple[Any, ...]) -> None:
            await asyncio.sleep(0.1)

        def fetchall(self) -> list[tuple[Any, ...]]:
            return []

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor()

        def close(self) -> None:
            return None

    async def connect(**_kwargs: Any) -> Connection:
        nonlocal attempts
        attempts += 1
        return Connection()

    reader = results.TdsqlTraceReader(
        host='db.test',
        port=3306,
        user='reader',
        password='unused',
        database='trace',
        table=TABLE,
        connect=connect,
        transient_errors=(),
        sleep=lambda delay: asyncio.sleep(0),
    )
    with pytest.raises(TraceDatabaseTimeout):
        await reader.read_many(['run'], include_timestamps=False)
    assert attempts == 3


async def test_db_retry_reconnects_and_classifies_unavailable() -> None:
    class Lost(Exception):  # noqa: N818 - test double
        pass

    attempts = 0
    closed = 0

    class Connection:
        def cursor(self) -> Any:
            raise Lost(2013, 'connection lost')

        def close(self) -> None:
            nonlocal closed
            closed += 1

    async def connect(**_kwargs: Any) -> Connection:
        nonlocal attempts
        attempts += 1
        return Connection()

    reader = results.TdsqlTraceReader(
        host='db.test',
        port=3306,
        user='reader',
        password='unused',
        database='trace',
        table=TABLE,
        connect=connect,
        transient_errors=(Lost,),
        sleep=lambda delay: asyncio.sleep(0),
    )
    with pytest.raises(TraceDatabaseUnavailable, match='3 attempts'):
        await reader.read_many(['run'])
    assert attempts == 3
    assert closed == 3


async def test_auth_failure_is_not_retried_or_fallen_back() -> None:
    class Auth(Exception):  # noqa: N818 - test double
        pass

    reader = ProbeReader()
    reader.table = None

    async def read_many(ids):
        raise TraceDatabaseAuth('access denied')

    reader.read_many = read_many  # type: ignore[method-assign]

    class Api:
        def __init__(self) -> None:
            self.results = 0

        async def get_control_events(self, execution_id: str) -> list[Any]:
            return []

        async def get_result(self, execution_id: str) -> Any:
            self.results += 1
            return {'success': True}

    api = Api()
    single = results._SingleExport(
        execution_id='exe',
        run_id='exe',
        status='succeeded',
        retry_count=0,
        context={'execution_id': 'exe', 'spec_type': 'nexus_single'},
    )
    with pytest.raises(ExportError, match='access denied'):
        await results._single_record(api=api, single=single, trace_reader=reader, trace_db_fallback='api')
    assert api.results == 0


async def test_transient_db_failure_falls_back_to_api_once() -> None:
    reader = ProbeReader()
    reader.table = None

    async def read_many(ids):
        raise TraceDatabaseTimeout('statement timed out')

    reader.read_many = read_many  # type: ignore[method-assign]

    class Api:
        def __init__(self) -> None:
            self.control_calls = 0
            self.result_calls = 0

        async def get_control_events(self, execution_id: str) -> list[Any]:
            self.control_calls += 1
            return []

        async def get_result(self, execution_id: str) -> Any:
            self.result_calls += 1
            return {'success': True, 'task_output': {'source': 'api'}}

    api = Api()
    single = results._SingleExport(
        execution_id='exe',
        run_id='exe',
        status='succeeded',
        retry_count=0,
        context={'execution_id': 'exe', 'spec_type': 'nexus_single'},
    )
    record = await results._single_record(api=api, single=single, trace_reader=reader, trace_db_fallback='api')
    assert isinstance(record, dict)
    assert record['task_output'] == {'source': 'api'}
    assert api.control_calls == 1
    assert api.result_calls == 1


async def test_transient_failure_falls_back_to_streaming_api_body(tmp_path: Path) -> None:
    reader = ProbeReader()
    reader.table = None

    async def read_many(ids):
        raise TraceDatabaseTimeout('statement timed out')

    reader.read_many = read_many  # type: ignore[method-assign]

    payload = json.dumps({'success': True, 'task_output': {'streamed': True}}).encode()

    class Api:
        async def get_control_events(self, execution_id: str) -> list[Any]:
            return []

        async def get_result_to_file(self, execution_id: str, fileobj: Any, scan: Any = None) -> int:
            fileobj.write(payload)
            if scan is not None:
                scan.feed(payload)
            return len(payload)

    single = results._SingleExport(
        execution_id='exe',
        run_id='exe',
        status='succeeded',
        retry_count=0,
        context={'execution_id': 'exe', 'spec_type': 'nexus_single'},
    )
    pool = results.ResultSpillPool(tmp_path / 'out.jsonl')
    row = await results._single_record(
        api=Api(),
        single=single,
        trace_reader=reader,
        fallback_spill=pool,
        scan_factory=results._ScanSlot,
        trace_db_fallback='api',
    )
    assert isinstance(row, results._StreamingRow)
    try:
        chunks = b''.join(row.iter_chunks())
        assert json.loads(chunks)['task_output'] == {'streamed': True}
    finally:
        row.discard()


async def test_transient_db_failure_with_fail_mode_does_not_call_api() -> None:
    reader = ProbeReader()
    reader.table = None

    async def read_many(ids):
        raise TraceDatabaseTimeout('statement timed out')

    reader.read_many = read_many  # type: ignore[method-assign]

    class Api:
        def __init__(self) -> None:
            self.result_calls = 0

        async def get_control_events(self, execution_id: str) -> list[Any]:
            return []

        async def get_result(self, execution_id: str) -> Any:
            self.result_calls += 1
            return {'success': True}

    api = Api()
    single = results._SingleExport(
        execution_id='exe',
        run_id='exe',
        status='succeeded',
        retry_count=0,
        context={'execution_id': 'exe', 'spec_type': 'nexus_single'},
    )
    with pytest.raises(ExportError, match='statement timed out'):
        await results._single_record(api=api, single=single, trace_reader=reader, trace_db_fallback='fail')
    assert api.result_calls == 0


async def test_item_deadline_aborts_processing() -> None:
    reader = ProbeReader()
    reader.table = None

    async def read_many(ids):
        if not ids:
            return {}
        await asyncio.sleep(0.1)
        return {run_id: results.TraceResult(output={'success': True}) for run_id in ids}

    reader.read_many = read_many  # type: ignore[method-assign]

    class Api:
        async def get_control_events(self, execution_id: str) -> list[Any]:
            return []

        async def get_result(self, execution_id: str) -> Any:
            return {'success': True}

    single = results._SingleExport(
        execution_id='exe',
        run_id='exe',
        status='succeeded',
        retry_count=0,
        context={'execution_id': 'exe', 'spec_type': 'nexus_single'},
    )
    with pytest.raises(TraceProcessingTimeout):
        await results._single_record(api=Api(), single=single, trace_reader=reader, deadline_seconds=0.005)


async def test_disabled_item_deadline_allows_slow_healthy_read() -> None:
    reader = ProbeReader()
    reader.table = None

    async def read_many(ids):
        if not ids:
            return {}
        await asyncio.sleep(0.05)
        return {run_id: results.TraceResult(output={'success': True}) for run_id in ids}

    reader.read_many = read_many  # type: ignore[method-assign]

    class Api:
        async def get_control_events(self, execution_id: str) -> list[Any]:
            return []

        async def get_result(self, execution_id: str) -> Any:
            return {'success': True}

    single = results._SingleExport(
        execution_id='exe',
        run_id='exe',
        status='succeeded',
        retry_count=0,
        context={'execution_id': 'exe', 'spec_type': 'nexus_single'},
    )
    record = await results._single_record(api=Api(), single=single, trace_reader=reader, deadline_seconds=None)
    assert record['run_id'] == 'exe'


async def test_disabled_item_deadline_does_not_mislabel_inner_timeout() -> None:
    reader = ProbeReader()
    reader.table = None

    async def read_many(ids):
        raise TimeoutError('api stream idle')

    reader.read_many = read_many  # type: ignore[method-assign]

    class Api:
        async def get_control_events(self, execution_id: str) -> list[Any]:
            return []

        async def get_result(self, execution_id: str) -> Any:
            return {'success': True}

    single = results._SingleExport(
        execution_id='exe',
        run_id='exe',
        status='succeeded',
        retry_count=0,
        context={'execution_id': 'exe', 'spec_type': 'nexus_single'},
    )
    with pytest.raises(results.TraceProcessingError) as exc_info:
        await results._single_record(api=Api(), single=single, trace_reader=reader, deadline_seconds=None)
    assert not isinstance(exc_info.value, results.TraceProcessingTimeout)
    assert 'phase=process' in str(exc_info.value)
    assert 'None' not in str(exc_info.value)


async def test_export_deadline_keeps_partial_and_does_not_publish(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv('NEXUS_EXPORT_ITEM_DEADLINE_SECONDS', '0.01')
    reader = ProbeReader()
    reader.table = None

    async def read_many(ids):
        if not ids:
            return {}
        await asyncio.sleep(0.1)
        return {run_id: results.TraceResult(output={'success': True}) for run_id in ids}

    reader.read_many = read_many  # type: ignore[method-assign]

    class Api:
        async def get_execution(self, execution_id: str) -> dict[str, Any]:
            return {
                'execution_id': execution_id,
                'spec_type': 'nexus_single',
                'status': 'succeeded',
                'attempt_count': 1,
                'created_at': '2026-09-29T01:00:00',
            }

        async def get_control_events(self, execution_id: str) -> list[Any]:
            return []

        async def get_result(self, execution_id: str) -> Any:
            return {'success': True}

    output = tmp_path / 'out.jsonl'
    with pytest.raises(TraceProcessingTimeout, match='phase=process'):
        await results.export_execution(api=Api(), execution_id='exe', output=output, force=False, trace_reader=reader)
    assert not output.exists()
    assert output.with_name('out.jsonl.partial').exists()


async def test_export_deadline_stops_before_reading() -> None:
    single = results._SingleExport(
        execution_id='exe',
        run_id='exe',
        status='succeeded',
        retry_count=0,
        context={},
    )
    plans = [results._RecordPlan(single=single)]
    with pytest.raises(ExportDeadlineExceeded):
        await results._write_records(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            plans,
            export_deadline=time.monotonic() - 1,
        )


def _event(value: Any) -> str:
    return json.dumps({
        'events': [
            {
                'name': 'run.event',
                'attributes': {
                    'event_type': 'run.output',
                    'event_data': {'output': value},
                },
            }
        ]
    })


def _no_output_event() -> str:
    return json.dumps({
        'events': [
            {
                'name': 'run.event',
                'attributes': {'event_type': 'run.started', 'event_data': {}},
            }
        ]
    })


class RootDatabase:
    """One root span with metadata/body rows, enough for the root scanner."""

    def __init__(self, records: list[tuple[int, int, str]]) -> None:
        self.records = records
        self.queries: list[tuple[str, tuple[Any, ...]]] = []

    async def connect(self, **kwargs: Any) -> RootConnection:
        return RootConnection(self)

    async def close(self) -> None:
        return None


class RootConnection:
    def __init__(self, database: RootDatabase) -> None:
        self.database = database

    def cursor(self) -> RootCursor:
        return RootCursor(self.database)

    def close(self) -> None:
        return None


class RootCursor(_Rows):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def __init__(self, database: RootDatabase) -> None:
        self.database = database
        self.rows: list[tuple[Any, ...]] = []

    async def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        self.database.queries.append((sql, params))
        if sql.startswith('SELECT DISTINCT run_span_id'):
            self.rows = [('span',)]
        elif sql.startswith('SELECT MAX(time_unix_nano)'):
            before = params[-1]
            times = [timestamp for record_id, timestamp, _ in self.database.records if record_id < before]
            self.rows = [(max(times) if times else None,)]
        elif sql.startswith('SELECT record_id, time_unix_nano'):
            ordered = sorted(self.database.records)
            if 'ORDER BY record_id DESC' in sql:
                self.rows = [(record_id, timestamp) for record_id, timestamp, _ in reversed(ordered[-params[-1] :])]
            else:
                selected = ordered
                if 'record_id < %s' in sql:
                    before = params[4]
                    selected = [row for row in selected if row[0] < before]
                    if 'time_unix_nano >= %s' in sql:
                        minimum = params[5]
                        selected = [row for row in selected if row[1] >= minimum]
                    if 'record_id > %s' in sql:
                        after = params[6]
                        selected = [row for row in selected if row[0] > after]
                elif 'record_id > %s' in sql:
                    after = params[4]
                    selected = [row for row in selected if row[0] > after]
                self.rows = [(record_id, timestamp) for record_id, timestamp, _ in selected[: params[-1]]]
        elif sql.startswith('SELECT event_data'):
            record_id = params[-1]
            self.rows = [(body,) for rid, _, body in self.database.records if rid == record_id]


def _root_reader(database: RootDatabase) -> results.TdsqlTraceReader:
    return results.TdsqlTraceReader(
        host='db.test',
        port=3306,
        user='reader',
        password='unused',
        database='trace',
        table=TABLE,
        connect=database.connect,
        transient_errors=(),
    )


async def test_reverse_scan_accepts_newest_output_without_full_walk(monkeypatch) -> None:
    monkeypatch.setenv('NEXUS_EXPORT_ROOT_SCAN', 'reverse')
    records = [(1, 100, _event({'success': True, 'task_output': {'old': True}}))]
    records += [(record_id, record_id * 100, _event(None)) for record_id in range(2, 131)]
    records[-1] = (130, 13000, _event({'success': True, 'task_output': {'new': True}}))
    database = RootDatabase(records)
    reader = _root_reader(database)
    result = (await reader.read_many(['run'], include_timestamps=False))['run']
    assert result.output == {'success': True, 'task_output': {'new': True}}
    assert any(
        'FORCE INDEX (idx_traj_span_events_span_id)' in sql and 'ORDER BY record_id DESC' in sql
        for sql, _ in database.queries
    )
    assert not any('ORDER BY record_id LIMIT' in sql for sql, _ in database.queries)
    body_queries = [sql for sql, _ in database.queries if sql.startswith('SELECT event_data')]
    # The newest row has run.output, so the other 127 window rows are not read.
    assert len(body_queries) == 1


async def test_reverse_scan_reads_only_older_records_when_newest_has_no_output() -> None:
    records = [
        (1, 100, _event({'success': True, 'task_output': {'old': True}})),
        (2, 200, _no_output_event()),
    ]
    database = RootDatabase(records)
    reader = _root_reader(database)
    result = (await reader.read_many(['run'], include_timestamps=False))['run']
    assert result.output == {'success': True, 'task_output': {'old': True}}
    # The window covered the whole span, so it was not replayed by the full
    # ascending walk.
    assert not any('ORDER BY record_id LIMIT' in sql for sql, _ in database.queries)
    body_queries = [sql for sql, _ in database.queries if sql.startswith('SELECT event_data')]
    assert len(body_queries) == 2


def test_hard_deadlines_are_disabled_by_default_and_env_can_enable(monkeypatch) -> None:
    monkeypatch.delenv('NEXUS_EXPORT_ITEM_DEADLINE_SECONDS', raising=False)
    monkeypatch.delenv('NEXUS_EXPORT_DEADLINE_SECONDS', raising=False)
    assert results._item_deadline_seconds() is None
    assert results._export_deadline_seconds() is None

    monkeypatch.setenv('NEXUS_EXPORT_ITEM_DEADLINE_SECONDS', 'off')
    monkeypatch.setenv('NEXUS_EXPORT_DEADLINE_SECONDS', '0')
    assert results._item_deadline_seconds() is None
    assert results._export_deadline_seconds() is None

    monkeypatch.setenv('NEXUS_EXPORT_ITEM_DEADLINE_SECONDS', '120')
    monkeypatch.setenv('NEXUS_EXPORT_DEADLINE_SECONDS', '7200')
    assert results._item_deadline_seconds() == 120
    assert results._export_deadline_seconds() == 7200


def test_tdsql_timeout_is_configurable(monkeypatch) -> None:
    monkeypatch.delenv('NEXUS_EXPORT_TDSQL_TIMEOUT_SECONDS', raising=False)
    assert results._tdsql_timeout_seconds() == 900
    monkeypatch.setenv('NEXUS_EXPORT_TDSQL_TIMEOUT_SECONDS', '1800')
    assert results._tdsql_timeout_seconds() == 1800
    monkeypatch.setenv('NEXUS_EXPORT_TDSQL_TIMEOUT_SECONDS', 'off')
    assert results._tdsql_timeout_seconds() == results._TDSQL_TIMEOUT_SECONDS


def test_results_fallback_defaults_to_api_and_can_be_disabled(monkeypatch) -> None:
    monkeypatch.delenv('NEXUS_EXPORT_TRACE_DB_FALLBACK', raising=False)
    assert results._db_fallback_mode() == 'api'
    monkeypatch.setenv('NEXUS_EXPORT_TRACE_DB_FALLBACK', 'fail')
    assert results._db_fallback_mode() == 'fail'


def test_results_cli_accepts_the_fallback_option(tmp_path: Path) -> None:
    from nexus.distributed.export import cli as export_cli

    args = export_cli.parse_args([
        '--execution-id',
        'exe',
        '--output',
        str(tmp_path / 'out.jsonl'),
        '--trace-db-fallback',
        'fail',
    ])
    assert args.trace_db_fallback == 'fail'


def test_trace_export_rejects_the_results_only_fallback_option() -> None:
    from argparse import Namespace

    from nexus.distributed.export import cli as export_cli

    with pytest.raises(ExportError, match='results exports'):
        export_cli.run_trace_export(Namespace(trace_db_fallback='fail'))


def test_integrated_export_parser_accepts_the_fallback_option(tmp_path: Path) -> None:
    import argparse

    from nexus.cli.commands.export import register_export_parser

    parser = argparse.ArgumentParser()
    register_export_parser(parser.add_subparsers())
    args = parser.parse_args([
        'export',
        '--execution-id',
        'exe',
        '--output',
        str(tmp_path / 'out.jsonl'),
        '--trace-db-fallback',
        'fail',
    ])
    assert args.trace_db_fallback == 'fail'
