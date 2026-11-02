"""TDSQL Boundless (TDStore) dialect: partitioned schema and parent_created_at plumbing.

The tdsql dialect stays byte-for-byte unchanged so the three existing DCDB backends keep
working; every assertion below that names ``dialect='tdsql'`` is a regression guard.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytest

from tests.unit.runner.trace._tdsql_fakes import AsyncCursorMixin, ConnectionMixin, install_connections

from nexus.runner.trace import RunTraceContext, RunTraceScope
from nexus.runner.trace.tdsql import schema as tdsql_schema
from nexus.runner.trace.tdsql import writer as tdsql_writer
from nexus.runner.trace.tdsql.schema import (
    BOUNDLESS_DIALECT,
    TDSQL_DIALECT,
    principal_trace_interval_statement,
    principal_trace_table_statement,
    schema_statements,
    span_event_insert_sql,
)
from nexus.runner.trace.tdsql.sink_source import (
    TDSQLTraceSink,
    TDSQLTraceSource,
    _build_query,
    normalize_parent_created_at,
)
from nexus.trace import SpanEventRecord


PRINCIPAL_SCOPE_ID = '3190e1c7731fff7995861c3221cec9a6'
PARENT_EXECUTION_ID = 'exe_0123456789abcdef0123456789abcdef'
PARENT_CREATED_AT_ISO = '2026-09-20T10:11:12.131415'
PARENT_CREATED_AT_SQL = '2026-09-20 10:11:12.131415'


def _record(record_id: int = 1) -> SpanEventRecord:
    return SpanEventRecord(
        record_id=record_id,
        event_type='update',
        time_unix_nano=record_id,
        trace_id='trace-1',
        span_id='run-span-1',
        parent_span_id='',
        event_data={'events': []},
    )


def _sink(**overrides: Any) -> TDSQLTraceSink:
    kwargs: dict[str, Any] = {
        'host': 'boundless-host',
        'user': 'nexus_dev',
        'password': 'secret',
        'db': 'runner',
        'dialect': 'boundless',
    }
    kwargs.update(overrides)
    return TDSQLTraceSink(**kwargs)


def _context(**metadata: Any) -> RunTraceContext:
    payload: dict[str, Any] = {
        'storage_scope': 'principal_v1',
        'principal_scope_id': PRINCIPAL_SCOPE_ID,
    }
    payload.update(metadata)
    return RunTraceContext(run_id='run-1', run_span_id='run-span-1', run_kind='single', metadata=payload)


# --- schema -----------------------------------------------------------------


def test_boundless_principal_table_uses_partitions_without_shardkey() -> None:
    statement = principal_trace_table_statement(
        PRINCIPAL_SCOPE_ID,
        dialect='boundless',
        subpartitions=12,
        initial_partition_date=date(2026, 9, 20),
    )

    assert 'shardkey' not in statement
    # The instance decides the engine (verified ROCKSDB on tdsql3-49qrr9xo6y).
    assert 'ENGINE=' not in statement
    assert 'parent_created_at DATETIME(6) NOT NULL' in statement
    assert 'PRIMARY KEY (id, run_id, parent_created_at)' in statement
    assert 'UNIQUE KEY uq_traj_span_events_record (run_id, run_span_id, record_id, parent_created_at)' in statement
    assert 'PARTITION BY RANGE COLUMNS (parent_created_at)' in statement
    assert 'SUBPARTITION BY KEY (run_id) SUBPARTITIONS 12' in statement
    assert "PARTITION p_before_20260920 VALUES LESS THAN ('2026-09-20 00:00:00')" in statement
    assert "PARTITION p20260920 VALUES LESS THAN ('2026-09-21 00:00:00')" in statement
    assert 'INDEX idx_traj_span_events_time_unix_nano (time_unix_nano)' in statement
    assert 'COLLATE=utf8mb4_0900_ai_ci' in statement


def test_tdsql_principal_table_is_unchanged() -> None:
    statement = principal_trace_table_statement(PRINCIPAL_SCOPE_ID)

    assert 'shardkey=run_id' in statement
    assert 'ENGINE=InnoDB' in statement
    assert 'parent_created_at' not in statement
    assert 'PRIMARY KEY (id, run_id)' in statement
    assert 'PARTITION BY' not in statement


def test_boundless_subpartitions_are_configurable() -> None:
    statement = principal_trace_table_statement(
        PRINCIPAL_SCOPE_ID,
        dialect='boundless',
        subpartitions=6,
        initial_partition_date=date(2026, 1, 2),
    )

    assert 'SUBPARTITIONS 6' in statement
    assert "PARTITION p20260102 VALUES LESS THAN ('2026-01-03 00:00:00')" in statement


def test_interval_statement_enables_daily_auto_partitioning() -> None:
    statement = principal_trace_interval_statement(PRINCIPAL_SCOPE_ID)

    assert statement.endswith('SET INTERVAL(DAY, 1)')
    assert f'trajectory_span_events_pr_{PRINCIPAL_SCOPE_ID}' in statement


@pytest.mark.parametrize('value', ['DAY', 'DAY, 0', 'CENTURY, 1', '', 'DROP TABLE'])
def test_interval_statement_rejects_invalid_intervals(value: str) -> None:
    with pytest.raises(ValueError):
        principal_trace_interval_statement(PRINCIPAL_SCOPE_ID, partition_interval=value)


@pytest.mark.parametrize('subpartitions', [0, -1, 5000, True])
def test_boundless_rejects_invalid_subpartitions(subpartitions: Any) -> None:
    with pytest.raises(ValueError):
        principal_trace_table_statement(PRINCIPAL_SCOPE_ID, dialect='boundless', subpartitions=subpartitions)


def test_unknown_dialect_is_rejected() -> None:
    with pytest.raises(ValueError):
        principal_trace_table_statement(PRINCIPAL_SCOPE_ID, dialect='tdsql4')


def test_boundless_insert_adds_partition_key_column() -> None:
    statement = span_event_insert_sql(PRINCIPAL_SCOPE_ID, dialect='boundless')

    assert 'parent_execution_id, parent_created_at, record_id' in statement
    assert statement.count('%s') == 11
    assert statement.endswith('ON DUPLICATE KEY UPDATE id = id')


def test_tdsql_insert_column_count_is_unchanged() -> None:
    assert span_event_insert_sql(PRINCIPAL_SCOPE_ID).count('%s') == 10
    assert span_event_insert_sql(None).count('%s') == 10


def test_boundless_rejects_shared_scope_insert() -> None:
    with pytest.raises(ValueError):
        span_event_insert_sql(None, dialect='boundless')


def test_boundless_skips_shared_bootstrap_table() -> None:
    # The shared table has no parent_created_at, so it is never created on Boundless.
    assert schema_statements('boundless') == []
    assert schema_statements() == [tdsql_schema.SHARED_TRACE_TABLE_STATEMENT]


# --- parent_created_at normalization ----------------------------------------


@pytest.mark.parametrize(
    'value',
    [
        PARENT_CREATED_AT_ISO,
        '2026-09-20 10:11:12.131415',
        datetime(2026, 9, 20, 10, 11, 12, 131415),
    ],
)
def test_parent_created_at_normalizes_to_mysql_datetime(value: Any) -> None:
    # isoformat() emits a "T" separator that MySQL DATETIME does not accept.
    assert normalize_parent_created_at(value) == PARENT_CREATED_AT_SQL


@pytest.mark.parametrize(
    'value',
    [
        '2026-09-20T10:11:12.131415+00:00',
        '2026-09-20T10:11:12.131415Z',
        '2026-09-20T18:11:12.131415+08:00',
        '2026-09-20T05:11:12.131415-05:00',
        datetime(2026, 9, 20, 10, 11, 12, 131415, tzinfo=timezone.utc),
        datetime(2026, 9, 20, 18, 11, 12, 131415, tzinfo=timezone(timedelta(hours=8))),
        datetime(2026, 9, 20, 5, 11, 12, 131415, tzinfo=timezone(timedelta(hours=-5))),
    ],
)
def test_parent_created_at_converts_offsets_to_utc(value: Any) -> None:
    # Rows carry Execution.created_at, which is naive UTC (distributed.utils.clock.utcnow).
    # Stripping the offset instead of converting would key "+08:00" reads eight hours away
    # from the partition the matching rows actually live in.
    assert normalize_parent_created_at(value) == PARENT_CREATED_AT_SQL


def test_parent_created_at_is_stable_across_representations_of_one_instant() -> None:
    utc = normalize_parent_created_at('2026-09-22T01:02:03.000000Z')
    shanghai = normalize_parent_created_at('2026-09-22T09:02:03.000000+08:00')

    assert utc == shanghai == '2026-09-22 01:02:03.000000'


def test_parent_created_at_offset_conversion_can_cross_the_day_partition() -> None:
    # Boundless partitions are daily, so the conversion decides which partition is read.
    assert normalize_parent_created_at('2026-09-21T02:30:00+08:00') == '2026-09-20 18:30:00.000000'


@pytest.mark.parametrize('value', [None, '', '   ', 'not-a-date', 42, 'Z', '2026-09-20T10:11:12+99:00'])
def test_parent_created_at_rejects_invalid_values(value: Any) -> None:
    with pytest.raises(ValueError):
        normalize_parent_created_at(value)


# --- the one entry point for a recorded partition key --------------------------------
# Callers used to re-check the type and emptiness before delegating, restating the checks
# below and leaving each free to disagree about which malformed values were worth reporting.


def test_an_absent_partition_key_is_the_classic_schema() -> None:
    from nexus.runner.trace.tdsql.sink_source import parse_partition_key

    assert parse_partition_key(None) is None


def test_an_absent_partition_key_is_a_failure_where_one_is_required() -> None:
    """On Boundless a missing filter scans every day in the retention window."""
    from nexus.runner.trace.tdsql.sink_source import InvalidPartitionKeyError, parse_partition_key

    with pytest.raises(InvalidPartitionKeyError, match='requires parent_created_at'):
        parse_partition_key(None, required=True)


@pytest.mark.parametrize('value', ['', '   ', 'not-a-date', 42, PARENT_EXECUTION_ID])
def test_an_unusable_partition_key_is_reported_as_such(value: Any) -> None:
    from nexus.runner.trace.tdsql.sink_source import InvalidPartitionKeyError, parse_partition_key

    with pytest.raises(InvalidPartitionKeyError):
        parse_partition_key(value)


def test_the_partition_key_entry_point_normalizes_like_the_writer() -> None:
    from nexus.runner.trace.tdsql.sink_source import parse_partition_key

    assert parse_partition_key(PARENT_CREATED_AT_ISO) == PARENT_CREATED_AT_SQL
    assert parse_partition_key(PARENT_CREATED_AT_ISO, required=True) == PARENT_CREATED_AT_SQL


def test_an_unusable_partition_key_stays_a_value_error() -> None:
    """Callers that already catch ValueError must keep working."""
    from nexus.runner.trace.tdsql.sink_source import InvalidPartitionKeyError

    assert issubclass(InvalidPartitionKeyError, ValueError)


# --- the principal table name and its inverse ----------------------------------------


def test_a_principal_table_name_round_trips_through_the_schema_module() -> None:
    from nexus.runner.trace.tdsql.schema import principal_scope_id_from_table, principal_trace_table_name

    table = principal_trace_table_name(PRINCIPAL_SCOPE_ID)

    assert principal_scope_id_from_table(table) == PRINCIPAL_SCOPE_ID


@pytest.mark.parametrize(
    'table',
    ['trajectory_span_events', 'some_other_table', 'trajectory_span_events_pr_deadbeef', '', 'x' * 64],
)
def test_a_table_no_writer_produced_has_no_principal_scope(table: str) -> None:
    """``removeprefix`` returned an unrelated name whole, as though it were a scope id."""
    from nexus.runner.trace.tdsql.schema import principal_scope_id_from_table

    with pytest.raises(ValueError, match='principal trace table'):
        principal_scope_id_from_table(table)


# --- sink -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_boundless_sink_writes_partition_key_column() -> None:
    sink = _sink()
    await sink.start(_context(parent_execution_id=PARENT_EXECUTION_ID, parent_created_at=PARENT_CREATED_AT_ISO))
    queued: list[Any] = []

    class Writer:
        is_ready = True

        def enqueue_span_event(self, params: tuple[Any, ...], **_kwargs: Any) -> int:
            queued.append(params)
            return 1

    sink._writer = Writer()  # type: ignore[assignment]
    sink.enqueue_record(_record())

    row = queued[0]
    assert row[0] == 'run-1'
    assert row[2] == PARENT_EXECUTION_ID
    assert row[3] == PARENT_CREATED_AT_SQL
    # Row width must match span_event_insert_sql(dialect='boundless').
    assert len(row) == 11


@pytest.mark.asyncio
async def test_boundless_sink_requires_parent_created_at() -> None:
    sink = _sink()

    with pytest.raises(ValueError):
        await sink.start(_context(parent_execution_id=PARENT_EXECUTION_ID))


@pytest.mark.asyncio
async def test_boundless_sink_rejects_shared_scope() -> None:
    sink = _sink()
    context = RunTraceContext(run_id='run-1', run_span_id='run-span-1', run_kind='single', metadata={})

    with pytest.raises(ValueError):
        await sink.start(context)


@pytest.mark.asyncio
async def test_boundless_sink_publishes_partition_key_in_envelope() -> None:
    sink = _sink()
    await sink.start(_context(parent_execution_id=PARENT_EXECUTION_ID, parent_created_at=PARENT_CREATED_AT_ISO))

    # Readers rely on the envelope to rebuild the pruning predicate.
    assert sink.metadata()['parent_created_at'] == PARENT_CREATED_AT_SQL


@pytest.mark.asyncio
async def test_tdsql_sink_does_not_emit_partition_key() -> None:
    sink = _sink(dialect='tdsql')
    await sink.start(_context(parent_execution_id=PARENT_EXECUTION_ID, parent_created_at=PARENT_CREATED_AT_ISO))

    assert 'parent_created_at' not in sink.metadata()


@pytest.mark.asyncio
async def test_boundless_prepare_scope_creates_partitioned_table_then_sets_interval() -> None:
    executed: list[str] = []

    class Cursor:
        async def __aenter__(self) -> Cursor:
            return self

        async def __aexit__(self, *_exc: Any) -> None:
            return None

        async def execute(self, sql: str, _params: Any = None) -> None:
            executed.append(sql)

        async def fetchone(self) -> Any:
            return None  # table does not exist yet

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor()

        async def commit(self) -> None:
            executed.append('COMMIT')

        def close(self) -> None:
            return None

    async def connect(**_kwargs: Any) -> Connection:
        return Connection()

    import nexus.runner.trace.tdsql._driver as driver

    original = driver.connect
    driver.connect = connect  # type: ignore[assignment]
    try:
        await _sink().prepare_scope(RunTraceScope(run_kind='single', scope_id=PRINCIPAL_SCOPE_ID))
    finally:
        driver.connect = original  # type: ignore[assignment]

    create = next(sql for sql in executed if sql.startswith('\nCREATE TABLE'))
    assert 'PARTITION BY RANGE COLUMNS (parent_created_at)' in create
    assert 'shardkey' not in create
    # SET INTERVAL only extends an existing RANGE layout, so it must follow the CREATE.
    assert executed.index('COMMIT') > executed.index(create)
    assert any('SET INTERVAL(DAY, 1)' in sql for sql in executed)
    # The table did not exist, so its layout needs no inspection.
    assert not any(sql.startswith('SHOW CREATE TABLE') for sql in executed)


_PARTITIONED_DDL = 'CREATE TABLE t (\n) /*!50500 PARTITION BY RANGE COLUMNS(parent_created_at) */'
# Boundless echoes the auto-extension back inside its versioned comment.
_INTERVAL_DDL = f'{_PARTITIONED_DDL}\n/*B!210604 INTERVAL(DAY, 1) */'
_NO_INTERVAL_DDL = _PARTITIONED_DDL


class _PrepareScopeConnection(AsyncCursorMixin, ConnectionMixin):
    """Scriptable stand-in serving both the connection and the cursor protocol."""

    def __init__(self, *, column_row: Any, table_ddl: str = _NO_INTERVAL_DDL, fail_once_on: str | None = None) -> None:
        self.column_row = column_row
        self.table_ddl = table_ddl
        self.fail_once_on = fail_once_on
        self.executed: list[str] = []
        self._last = ''

    async def execute(self, sql: str, _params: Any = None) -> None:
        self._last = sql
        self.executed.append(sql)
        if self.fail_once_on is not None and self.fail_once_on in sql:
            self.fail_once_on = None
            raise RuntimeError('DDL interrupted')

    async def fetchone(self) -> Any:
        if self._last.startswith('SHOW CREATE TABLE'):
            return ('table', self.table_ddl)
        return self.column_row

    async def commit(self) -> None:
        self.executed.append('COMMIT')


_install_connections = install_connections


def _scope() -> RunTraceScope:
    return RunTraceScope(run_kind='single', scope_id=PRINCIPAL_SCOPE_ID)


@pytest.mark.asyncio
async def test_boundless_prepare_scope_applies_a_missing_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """A table whose INTERVAL never landed must be repaired, not skipped.

    The column probe cannot tell the two apart: CREATE and SET INTERVAL auto-commit
    separately, so a partially initialised scope looks exactly like a finished one.
    """
    connection = _PrepareScopeConnection(column_row=('YES',), table_ddl=_NO_INTERVAL_DDL)
    _install_connections(monkeypatch, connection)

    await _sink().prepare_scope(_scope())

    assert not any(sql.startswith('\nCREATE TABLE') for sql in connection.executed)
    assert any(sql.startswith('SHOW CREATE TABLE') for sql in connection.executed)
    assert any('SET INTERVAL(DAY, 1)' in sql for sql in connection.executed)


@pytest.mark.asyncio
async def test_boundless_prepare_scope_leaves_a_configured_interval_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _PrepareScopeConnection(column_row=('YES',), table_ddl=_INTERVAL_DDL)
    _install_connections(monkeypatch, connection)

    await _sink().prepare_scope(_scope())

    assert any(sql.startswith('SHOW CREATE TABLE') for sql in connection.executed)
    assert not any('SET INTERVAL' in sql for sql in connection.executed)
    assert not any(sql.startswith('\nCREATE TABLE') for sql in connection.executed)


@pytest.mark.asyncio
async def test_boundless_prepare_scope_recovers_from_a_failed_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fault injection: SET INTERVAL fails after the CREATE committed; a retry must fix it.

    Granting again is the only path that reaches this code, so if the retry reports success
    without applying the INTERVAL, the scope writes fine until the seeded day runs out.
    """
    first = _PrepareScopeConnection(column_row=None, fail_once_on='SET INTERVAL')
    second = _PrepareScopeConnection(column_row=('YES',), table_ddl=_NO_INTERVAL_DDL)
    _install_connections(monkeypatch, first, second)

    with pytest.raises(RuntimeError):
        await _sink().prepare_scope(_scope())
    assert any(sql.startswith('\nCREATE TABLE') for sql in first.executed)
    assert 'COMMIT' not in first.executed

    await _sink().prepare_scope(_scope())

    assert any('SET INTERVAL(DAY, 1)' in sql for sql in second.executed)
    assert 'COMMIT' in second.executed


@pytest.mark.asyncio
async def test_boundless_prepare_scope_applies_interval_when_layout_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An undeterminable layout is treated as unconfigured: fail loudly now, not tomorrow."""
    connection = _PrepareScopeConnection(column_row=('YES',), table_ddl='')
    _install_connections(monkeypatch, connection)

    await _sink().prepare_scope(_scope())

    assert any('SET INTERVAL(DAY, 1)' in sql for sql in connection.executed)


@pytest.mark.asyncio
async def test_tdsql_prepare_scope_never_inspects_the_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard: the classic dialect has no INTERVAL and must issue no extra query."""
    connection = _PrepareScopeConnection(column_row=('YES',))
    _install_connections(monkeypatch, connection)

    await _sink(dialect='tdsql').prepare_scope(_scope())

    assert not any(sql.startswith('SHOW CREATE TABLE') for sql in connection.executed)
    assert not any('SET INTERVAL' in sql for sql in connection.executed)


# --- source -----------------------------------------------------------------


def test_boundless_query_prunes_by_partition_key() -> None:
    metadata = {
        'storage_scope': 'principal_v1',
        'principal_scope_id': PRINCIPAL_SCOPE_ID,
        'parent_execution_id': PARENT_EXECUTION_ID,
        'parent_created_at': PARENT_CREATED_AT_ISO,
        'run_id': 'run-1',
        'run_span_id': 'run-span-1',
    }

    sql, params = _build_query(metadata, 0, dialect='boundless')

    assert 'parent_created_at = %s' in sql
    assert PARENT_CREATED_AT_SQL in params


def test_tdsql_query_has_no_partition_predicate() -> None:
    metadata = {
        'storage_scope': 'principal_v1',
        'principal_scope_id': PRINCIPAL_SCOPE_ID,
        'parent_execution_id': PARENT_EXECUTION_ID,
        'run_id': 'run-1',
        'run_span_id': 'run-span-1',
    }

    sql, params = _build_query(metadata, 0)

    assert 'parent_created_at' not in sql
    assert PARENT_CREATED_AT_SQL not in params


def test_boundless_query_requires_partition_key() -> None:
    metadata = {
        'storage_scope': 'principal_v1',
        'principal_scope_id': PRINCIPAL_SCOPE_ID,
        'run_id': 'run-1',
    }

    with pytest.raises(ValueError):
        _build_query(metadata, 0, dialect='boundless')


def test_source_rejects_unknown_dialect() -> None:
    with pytest.raises(ValueError):
        TDSQLTraceSource(host='h', user='u', db='runner', dialect='tdstore')


# --- writer -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_writer_pool_separates_dialects(monkeypatch: pytest.MonkeyPatch) -> None:
    class Writer:
        def __init__(self, *, dialect: str, **_kwargs) -> None:
            self.dialect = dialect
            self.is_ready = True

        async def ensure_started(self, _db_config: dict[str, Any] | None) -> bool:
            return True

    monkeypatch.setattr(tdsql_writer, 'TDSQLTraceWriter', Writer)
    monkeypatch.setattr(tdsql_writer, '_writers', {}, raising=False)
    config = {'host': 'h', 'port': 3306, 'user': 'u', 'password': 'p', 'db': 'runner'}

    tdsql_one = await tdsql_writer.get_writer(dict(config))
    tdsql_two = await tdsql_writer.get_writer(dict(config))
    boundless = await tdsql_writer.get_writer(dict(config), dialect='boundless')

    assert tdsql_one is tdsql_two
    # A shared writer would emit the wrong INSERT column list for one of the dialects.
    assert boundless is not tdsql_one
    assert boundless.dialect == 'boundless'  # type: ignore[union-attr]


# --- driver plumbing --------------------------------------------------------


def _single_driver(**overrides: Any) -> Any:
    from nexus.distributed.execution.drivers.nexus_single import NexusSingleDriver

    kwargs: dict[str, Any] = {
        'execution_id': 'exe_child',
        'assignment_id': 'asg-1',
        'trace_registry': object(),
        'owner_principal_id': 'principal-owner-a',
        'execution_created_at': datetime(2026, 9, 21, 1, 2, 3, 400000),
    }
    kwargs.update(overrides)
    return NexusSingleDriver(**kwargs)


def test_single_driver_passes_parent_created_at_for_batch_child() -> None:
    driver = _single_driver(parent_execution_id=PARENT_EXECUTION_ID)

    metadata = driver._trace_context_metadata(
        {
            'batch': {
                'parent_execution_id': PARENT_EXECUTION_ID,
                'parent_created_at': PARENT_CREATED_AT_ISO,
                'item_id': 'item-1',
                'attempt_no': 0,
            }
        }
    )

    # The parent's timestamp wins over the child's own created_at.
    assert metadata['parent_created_at'] == PARENT_CREATED_AT_ISO
    assert metadata['parent_execution_id'] == PARENT_EXECUTION_ID


def test_single_driver_falls_back_to_own_created_at_without_batch() -> None:
    driver = _single_driver()

    metadata = driver._trace_context_metadata({})

    assert metadata['storage_scope'] == 'principal_v1'
    assert metadata['parent_created_at'] == '2026-09-21T01:02:03.400000'


def test_single_driver_omits_partition_key_when_unavailable() -> None:
    # Older call sites that do not pass execution_created_at keep the legacy metadata.
    driver = _single_driver(execution_created_at=None)

    assert 'parent_created_at' not in driver._trace_context_metadata({})


def test_single_driver_rejects_non_string_parent_created_at() -> None:
    driver = _single_driver(parent_execution_id=PARENT_EXECUTION_ID)

    with pytest.raises(ValueError):
        driver._trace_context_metadata(
            {
                'batch': {
                    'parent_execution_id': PARENT_EXECUTION_ID,
                    'parent_created_at': 1758000000,
                    'item_id': 'item-1',
                    'attempt_no': 0,
                }
            }
        )


def test_live_driver_uses_execution_created_at() -> None:
    from nexus.distributed.execution.drivers.nexus_live import NexusLiveDriver

    driver = NexusLiveDriver(
        execution_id='exe_live',
        assignment_id='asg-1',
        internal_base_url='http://internal',
        trace_registry=object(),
        owner_principal_id='principal-owner-a',
        execution_created_at=datetime(2026, 9, 21, 5, 6, 7, 800000),
    )

    # Live runs have no batch parent, so the run's own timestamp is the partition key.
    assert driver._trace_context_metadata()['parent_created_at'] == '2026-09-21T05:06:07.800000'


# --- export -----------------------------------------------------------------


def _export_reader(**overrides: Any) -> tuple[Any, list[tuple[str, tuple[Any, ...]]]]:
    from nexus.distributed.export import results as export_results

    seen: list[tuple[str, tuple[Any, ...]]] = []

    class Cursor(AsyncCursorMixin):
        async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
            seen.append((sql, params))

        async def fetchall(self) -> list[tuple[Any, ...]]:
            return []

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor()

        def close(self) -> None:
            return None

    async def connect(**_kwargs: Any) -> Connection:
        return Connection()

    kwargs: dict[str, Any] = {
        'host': 'db.test',
        'port': 3306,
        'user': 'readonly',
        'password': 'secret',
        'database': 'runner',
        'table': f'trajectory_span_events_pr_{PRINCIPAL_SCOPE_ID}',
        'connect': connect,
        'transient_errors': (),
    }
    kwargs.update(overrides)
    return export_results.TdsqlTraceReader(**kwargs), seen


@pytest.mark.asyncio
async def test_export_reader_prunes_partitions_when_key_supplied() -> None:
    reader, seen = _export_reader(parent_created_at=PARENT_CREATED_AT_ISO)

    await reader.read_many(['run-1', 'run-2'])

    from nexus.distributed.export import results as export_results

    assert seen, 'expected the reader to issue queries'
    for sql, params in seen:
        assert 'parent_created_at = %s' in sql
        assert PARENT_CREATED_AT_SQL in params
    # Root discovery uses only the run/span index; even an empty discovery
    # must stay inside the supplied partition.
    output_sql, output_params = seen[-1]
    assert output_sql.startswith('SELECT DISTINCT run_span_id')
    assert 'LIKE' not in output_sql
    assert 'span_id = run_span_id' not in output_sql
    assert 'event_data' not in output_sql
    assert output_params[1] == PARENT_CREATED_AT_SQL
    assert output_params[-1] == export_results._TDSQL_ROOT_PAGE_SIZE


@pytest.mark.asyncio
async def test_export_reader_unchanged_without_partition_key() -> None:
    reader, seen = _export_reader()

    await reader.read_many(['run-1'])

    assert seen
    for sql, params in seen:
        assert 'parent_created_at' not in sql
        assert params[0] == 'run-1'


# --- raw cursor reads (console paging) --------------------------------------
#
# These three queries drive the console; an unpruned variant sweeps every day partition
# on the Boundless table, so each needs its own guard plus a tdsql regression twin.


def _raw_metadata(**overrides: Any) -> dict[str, Any]:
    # run_id/run_span_id must match the rows produced by ``raw_row``: the reader rejects
    # any page whose identity columns disagree with the requested stream.
    payload: dict[str, Any] = {
        'storage_scope': 'principal_v1',
        'principal_scope_id': PRINCIPAL_SCOPE_ID,
        'run_id': 'exec',
        'run_span_id': 'span',
        'parent_created_at': PARENT_CREATED_AT_ISO,
    }
    payload.update(overrides)
    return payload


def _raw_source(
    monkeypatch: pytest.MonkeyPatch, rows: list[Any], *, dialect: str = 'boundless'
) -> tuple[Any, list[tuple[str, tuple[Any, ...]]]]:
    from tests.unit.runner.trace.test_tdsql_raw_page import install

    source, connection = install(monkeypatch, rows, dialect=dialect)
    return source, connection.instance.calls


@pytest.mark.asyncio
async def test_boundless_stream_listing_prunes_by_partition_key(monkeypatch: pytest.MonkeyPatch) -> None:
    source, calls = _raw_source(monkeypatch, [('run-span-1',)])

    assert await source.list_raw_streams(_raw_metadata(), limit=2) == ['run-span-1']

    sql, params = calls[0]
    assert 'parent_created_at = %s' in sql
    # The partition key must precede the keyset bound so LIMIT stays the last placeholder.
    assert params == ('exec', PARENT_CREATED_AT_SQL, 2)


@pytest.mark.asyncio
async def test_boundless_stream_listing_keeps_partition_key_before_keyset(monkeypatch: pytest.MonkeyPatch) -> None:
    source, calls = _raw_source(monkeypatch, [('run-span-2',)])

    await source.list_raw_streams(_raw_metadata(), after='run-span-1', limit=2)

    sql, params = calls[0]
    assert sql.index('parent_created_at') < sql.index('run_span_id > %s')
    assert params == ('exec', PARENT_CREATED_AT_SQL, 'run-span-1', 2)


@pytest.mark.asyncio
async def test_tdsql_stream_listing_has_no_partition_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    source, calls = _raw_source(monkeypatch, [('run-span-1',)], dialect='tdsql')

    await source.list_raw_streams(_raw_metadata(), after='run-span-0', limit=2)

    sql, params = calls[0]
    assert 'parent_created_at' not in sql
    assert params == ('exec', 'run-span-0', 2)


@pytest.mark.asyncio
async def test_boundless_stream_existence_prunes_by_partition_key(monkeypatch: pytest.MonkeyPatch) -> None:
    source, calls = _raw_source(monkeypatch, [(1,), None])

    assert await source.raw_stream_exists(_raw_metadata()) is True

    sql, params = calls[0]
    assert 'parent_created_at = %s' in sql
    assert params == ('exec', 'span', PARENT_CREATED_AT_SQL)


@pytest.mark.asyncio
async def test_tdsql_stream_existence_has_no_partition_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    source, calls = _raw_source(monkeypatch, [], dialect='tdsql')

    assert await source.raw_stream_exists(_raw_metadata()) is False

    sql, params = calls[0]
    assert 'parent_created_at' not in sql
    assert params == ('exec', 'span')


@pytest.mark.asyncio
async def test_boundless_event_page_prunes_by_partition_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.runner.trace.test_tdsql_raw_page import raw_row

    source, calls = _raw_source(monkeypatch, [raw_row(1)])

    page = await source.read_raw_event_page(_raw_metadata(), 0, limit=2, accept_record=lambda _row: True)

    assert [row['record_id'] for row in page.items] == [1]
    sql, params = calls[0]
    assert 'parent_created_at = %s' in sql
    # The projection placeholder binds first, then the WHERE clause in written order.
    assert params == (8 * 1024 * 1024, 'exec', 'span', PARENT_CREATED_AT_SQL, 0, 3)
    assert sql.index('parent_created_at') < sql.index('record_id > %s')


@pytest.mark.asyncio
async def test_tdsql_event_page_has_no_partition_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.runner.trace.test_tdsql_raw_page import raw_row

    source, calls = _raw_source(monkeypatch, [raw_row(1)], dialect='tdsql')

    await source.read_raw_event_page(_raw_metadata(), 0, limit=2, accept_record=lambda _row: True)

    sql, params = calls[0]
    assert 'parent_created_at' not in sql
    assert params == (8 * 1024 * 1024, 'exec', 'span', 0, 3)


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['list_raw_streams', 'raw_stream_exists', 'read_raw_event_page'])
async def test_boundless_raw_reads_require_partition_key(monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    source, _calls = _raw_source(monkeypatch, [])
    metadata = _raw_metadata()
    del metadata['parent_created_at']
    arguments: dict[str, tuple[Any, ...]] = {
        'list_raw_streams': (metadata,),
        'raw_stream_exists': (metadata,),
        'read_raw_event_page': (metadata, 0),
    }

    with pytest.raises(ValueError):
        call = getattr(source, operation)
        if operation == 'read_raw_event_page':
            await call(*arguments[operation], accept_record=lambda _row: True)
        else:
            await call(*arguments[operation])


# --- console read plumbing --------------------------------------------------
#
# The reader is built from trace_read config rather than run_trace_sinks, so the dialect
# and the partition key have to travel through resolution and through paging cursors.


def _console_config(*, dialect: str = 'boundless') -> Any:
    from nexus.distributed.config import DistributedConfig

    return DistributedConfig(
        trace_read={
            'enabled': True,
            'backend_allowlist': ['tdsql'],
            'readers': [
                {
                    'backend': 'tdsql',
                    'source_id': 'dataset-tdsql',
                    'source_revision': '1',
                    'environment_ids': ['env-1'],
                    'host': 'reader.invalid',
                    'database': 'traces',
                    'credential_ref': 'reader',
                    'dialect': dialect,
                }
            ],
        },
        run_trace_sinks={'tdsql': {'kind': 'tdsql'}},
        secrets={
            'trace_cursor_keys': {'default': 'k' * 32},
            'trace_read_credentials': {'reader': {'username': 'only-select', 'password': 'secret'}},
        },
    )


def _console_target() -> Any:
    from nexus.distributed.trace.models import AuthorizedTraceTarget

    return AuthorizedTraceTarget('owner', 'exec', 'owner', 'env-1', 'running', 'asn', {})


def _console_assignment(
    *,
    parent_created_at: str | None,
    binding_partition_key: str | None = None,
    publish_envelope: bool = True,
) -> Any:
    """One assignment as the Console sees it.

    ``publish_envelope=False`` is the running state: the Driver has reported its
    ``source_binding`` but ``_artifacts`` is only published when the run closes.
    """
    from nexus.distributed.trace.bindings import make_source_binding
    from nexus.distributed.trace.models import AssignmentTraceMetadata, TraceSourceIdentity

    binding = make_source_binding(
        run_id='exec',
        run_span_id='span',
        assignment_id='asn',
        attempt_id='attempt-1',
        owner_principal_id='owner',
        selected_sinks=('tdsql',),
        source_identities={'tdsql': TraceSourceIdentity(source_id='dataset-tdsql', source_revision='1')},
        parent_created_at=binding_partition_key,
    )
    runtime_info: dict[str, Any] = {'run_trace': {'source_binding': binding}}
    if publish_envelope:
        sink_metadata: dict[str, Any] = {'storage_scope': 'principal_v1'}
        if parent_created_at is not None:
            sink_metadata['parent_created_at'] = parent_created_at
        runtime_info['_artifacts'] = {
            'single': {
                'run_id': 'exec',
                'run_kind': 'single',
                'sinks': [{'name': 'tdsql', 'metadata': sink_metadata}],
            }
        }
    return AssignmentTraceMetadata('asn', 'attempt-1', 1, 'active', datetime.now(timezone.utc), runtime_info)


def test_console_resolution_carries_partition_key_into_read_metadata() -> None:
    from nexus.distributed.trace.sources import TraceSourceResolver

    resolver = TraceSourceResolver(_console_config())

    resolution = resolver.resolve(_console_target(), [_console_assignment(parent_created_at=PARENT_CREATED_AT_SQL)])

    source = resolution.sources[0]
    assert source.parent_created_at == PARENT_CREATED_AT_SQL
    assert source.scope('exec', 'span')['parent_created_at'] == PARENT_CREATED_AT_SQL


def test_console_resolution_rejects_boundless_reader_without_partition_key() -> None:
    from nexus.distributed.trace.models import TraceReadError
    from nexus.distributed.trace.sources import TraceSourceResolver

    resolver = TraceSourceResolver(_console_config())

    # Resolving anyway would silently downgrade every later query to a full-table scan.
    with pytest.raises(TraceReadError, match='trace_source_unresolved'):
        resolver.resolve(_console_target(), [_console_assignment(parent_created_at=None)])


def test_console_resolution_ignores_partition_key_for_tdsql_reader() -> None:
    from nexus.distributed.trace.sources import TraceSourceResolver

    resolver = TraceSourceResolver(_console_config(dialect='tdsql'))

    resolution = resolver.resolve(_console_target(), [_console_assignment(parent_created_at=PARENT_CREATED_AT_SQL)])

    source = resolution.sources[0]
    assert source.parent_created_at is None
    assert 'parent_created_at' not in source.scope('exec', 'span')


@pytest.mark.parametrize(('dialect', 'expected'), [('boundless', 'boundless'), ('tdsql', 'tdsql')])
def test_console_reader_is_created_with_the_configured_dialect(dialect: str, expected: str) -> None:
    from nexus.distributed.trace.sources import TraceSourceResolver

    resolver = TraceSourceResolver(_console_config(dialect=dialect))
    resolution = resolver.resolve(_console_target(), [_console_assignment(parent_created_at=PARENT_CREATED_AT_SQL)])

    reader = resolver.create_source(resolution.sources[0])

    assert reader.dialect == expected


def test_paging_cursor_round_trips_the_partition_key() -> None:
    from nexus.distributed.trace.cursor import TraceCursorCodec
    from nexus.distributed.trace.sources import TraceSourceResolver

    config = _console_config()
    resolver = TraceSourceResolver(config)
    resolved = resolver.resolve(
        _console_target(), [_console_assignment(parent_created_at=PARENT_CREATED_AT_SQL)]
    ).sources[0]
    codec = TraceCursorCodec({'default': 'k' * 32}, current_key_id='default')
    scope = {
        'principal': 'owner',
        'owner': 'owner',
        'environment': 'env-1',
        'execution': 'exec',
        'backend': 'tdsql',
        'source_id': 'dataset-tdsql',
        'source_revision': '1',
        'parent_created_at': resolved.parent_created_at,
        'span': 'span',
    }

    token = codec.issue(scope, purpose='events', position='0')
    decoded = codec.decode(token, purpose='events', target=_console_target())

    assert decoded['parent_created_at'] == PARENT_CREATED_AT_SQL
    # Paging never re-resolves, so the cursor alone must restore the pruning key.
    assert resolver.from_cursor(_console_target(), decoded).parent_created_at == PARENT_CREATED_AT_SQL


def test_paging_cursor_without_partition_key_still_decodes_for_tdsql() -> None:
    from nexus.distributed.trace.cursor import TraceCursorCodec
    from nexus.distributed.trace.sources import TraceSourceResolver

    codec = TraceCursorCodec({'default': 'k' * 32}, current_key_id='default')
    scope = {
        'principal': 'owner',
        'owner': 'owner',
        'environment': 'env-1',
        'execution': 'exec',
        'backend': 'tdsql',
        'source_id': 'dataset-tdsql',
        'source_revision': '1',
        'span': 'span',
    }
    decoded = codec.decode(
        codec.issue(scope, purpose='events', position='0'), purpose='events', target=_console_target()
    )

    assert 'parent_created_at' not in decoded
    assert (
        TraceSourceResolver(_console_config(dialect='tdsql')).from_cursor(_console_target(), decoded).parent_created_at
        is None
    )


def test_cursor_minted_for_another_dialect_is_rejected() -> None:
    from nexus.distributed.trace.models import TraceReadError
    from nexus.distributed.trace.sources import TraceSourceResolver

    resolver = TraceSourceResolver(_console_config())
    stale = {'backend': 'tdsql', 'source_id': 'dataset-tdsql', 'source_revision': '1'}

    # A cursor issued before the dataset moved to Boundless carries no partition key.
    with pytest.raises(TraceReadError, match='invalid_trace_cursor'):
        resolver.from_cursor(_console_target(), stale)


# --- running reads: the binding carries the partition key -------------------
#
# ``_artifacts`` is published when the run closes, so a reader that depends on it cannot
# serve a running Execution. The source binding is reported with the running state and is
# therefore the only carrier available before close.


def test_console_resolution_reads_the_partition_key_from_the_binding_while_running() -> None:
    from nexus.distributed.trace.sources import TraceSourceResolver

    resolver = TraceSourceResolver(_console_config())

    resolution = resolver.resolve(
        _console_target(),
        [
            _console_assignment(
                parent_created_at=None,
                binding_partition_key=PARENT_CREATED_AT_SQL,
                publish_envelope=False,
            )
        ],
    )

    source = resolution.sources[0]
    assert source.parent_created_at == PARENT_CREATED_AT_SQL
    assert source.scope('exec', 'span')['parent_created_at'] == PARENT_CREATED_AT_SQL


def test_console_resolution_fails_when_neither_binding_nor_envelope_has_the_key() -> None:
    from nexus.distributed.trace.models import TraceReadError
    from nexus.distributed.trace.sources import TraceSourceResolver

    resolver = TraceSourceResolver(_console_config())

    with pytest.raises(TraceReadError, match='trace_source_unresolved'):
        resolver.resolve(
            _console_target(),
            [_console_assignment(parent_created_at=None, publish_envelope=False)],
        )


def test_console_resolution_rejects_a_binding_the_envelope_contradicts() -> None:
    from nexus.distributed.trace.models import TraceReadError
    from nexus.distributed.trace.sources import TraceSourceResolver

    resolver = TraceSourceResolver(_console_config())

    # Two different keys mean one of them prunes to a partition holding none of the rows.
    with pytest.raises(TraceReadError, match='trace_source_unresolved'):
        resolver.resolve(
            _console_target(),
            [
                _console_assignment(
                    parent_created_at=PARENT_CREATED_AT_SQL,
                    binding_partition_key='2026-09-19 00:00:00.000000',
                )
            ],
        )


def test_console_resolution_prefers_the_envelope_once_it_agrees() -> None:
    from nexus.distributed.trace.sources import TraceSourceResolver

    resolver = TraceSourceResolver(_console_config())

    resolution = resolver.resolve(
        _console_target(),
        [
            _console_assignment(
                parent_created_at=PARENT_CREATED_AT_SQL,
                binding_partition_key=PARENT_CREATED_AT_ISO,
            )
        ],
    )

    # The binding stores the normalized form, so the ISO input above must not read as a
    # conflict with the envelope's SQL literal.
    assert resolution.sources[0].parent_created_at == PARENT_CREATED_AT_SQL


def test_console_resolution_ignores_a_binding_key_for_a_classic_reader() -> None:
    from nexus.distributed.trace.sources import TraceSourceResolver

    resolver = TraceSourceResolver(_console_config(dialect='tdsql'))

    resolution = resolver.resolve(
        _console_target(),
        [
            _console_assignment(
                parent_created_at=None,
                binding_partition_key=PARENT_CREATED_AT_SQL,
                publish_envelope=False,
            )
        ],
    )

    assert resolution.sources[0].parent_created_at is None


def test_source_binding_normalizes_the_partition_key_like_the_sink_does() -> None:
    from nexus.distributed.trace.bindings import make_source_binding, source_binding
    from nexus.distributed.trace.models import TraceSourceIdentity

    # The Driver holds an ISO string with an offset; the sink envelope holds a MySQL
    # DATETIME literal. Storing the raw ISO form would make every cross-check a conflict.
    binding = make_source_binding(
        run_id='exec',
        run_span_id='span',
        assignment_id='asn',
        attempt_id='attempt-1',
        owner_principal_id='owner',
        selected_sinks=('tdsql',),
        source_identities={'tdsql': TraceSourceIdentity(source_id='dataset-tdsql', source_revision='1')},
        parent_created_at='2026-09-20T18:11:12.131415+08:00',
    )

    assert binding['parent_created_at'] == PARENT_CREATED_AT_SQL
    assert binding['parent_created_at'] == normalize_parent_created_at('2026-09-20T18:11:12.131415+08:00')
    parsed = source_binding({'run_trace': {'source_binding': binding}})
    assert parsed is not None
    assert parsed.parent_created_at == PARENT_CREATED_AT_SQL


def test_source_binding_omits_the_partition_key_when_there_is_none() -> None:
    from nexus.distributed.trace.bindings import make_source_binding
    from nexus.distributed.trace.models import TraceSourceIdentity

    # Builds that predate the field reject unknown keys, so a classic run must keep
    # publishing the exact shape they already accept.
    binding = make_source_binding(
        run_id='exec',
        run_span_id='span',
        assignment_id='asn',
        attempt_id='attempt-1',
        owner_principal_id='owner',
        selected_sinks=('tdsql',),
        source_identities={'tdsql': TraceSourceIdentity(source_id='dataset-tdsql', source_revision='1')},
    )

    assert 'parent_created_at' not in binding


def test_binding_partition_key_is_cross_checked_against_the_final_envelope() -> None:
    from nexus.distributed.errors import DomainConflictError
    from nexus.distributed.trace.bindings import make_source_binding, validate_source_binding_update
    from nexus.distributed.trace.models import TraceSourceIdentity

    binding = make_source_binding(
        run_id='exec',
        run_span_id='span',
        assignment_id='asn',
        attempt_id='attempt-1',
        owner_principal_id='owner',
        selected_sinks=('tdsql',),
        source_identities={'tdsql': TraceSourceIdentity(source_id='dataset-tdsql', source_revision='1')},
        parent_created_at=PARENT_CREATED_AT_SQL,
    )
    final = {
        '_artifacts': {
            'single': {
                'run_id': 'exec',
                'run_span_id': 'span',
                'run_kind': 'single',
                'sinks': [
                    {
                        'name': 'tdsql',
                        'metadata': {
                            'storage_scope': 'principal_v1',
                            'parent_created_at': '2026-01-01 00:00:00.000000',
                        },
                    }
                ],
            }
        }
    }

    with pytest.raises(DomainConflictError, match='artifact_mismatch'):
        validate_source_binding_update(
            {'run_trace': {'source_binding': binding}},
            final,
            execution_id='exec',
            assignment_id='asn',
            attempt_id='attempt-1',
            owner_principal_id='owner',
        )


# --- trace export cli -------------------------------------------------------


def _run_trace_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, **overrides: Any) -> tuple[int, dict[str, Any]]:
    import argparse

    from nexus.distributed.export import cli as export_cli

    captured: dict[str, Any] = {}

    class Api:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        async def get_execution(self, _execution_id: str) -> dict[str, Any]:
            return {'spec_type': 'nexus_single'}

    async def fake_export(**kwargs: Any) -> None:
        captured['source'] = kwargs['source']
        captured['scope'] = kwargs['scope_metadata']

    monkeypatch.setattr(export_cli, 'CloudApi', Api)
    monkeypatch.setattr(export_cli, 'export_trace_events', fake_export)
    monkeypatch.setattr(export_cli, '_validate_target', lambda *_a, **_k: 'nexus_single')
    monkeypatch.setattr(export_cli, 'JsonlEventWriter', lambda *_a, **_k: object())

    arguments: dict[str, Any] = {
        'control_url': 'https://control.test',
        'execution_id': 'exe-1',
        'taihu_token': 'token',
        'taihu_token_file': None,
        'db_host': 'db.test',
        'db_port': 3306,
        'db_name': 'runner',
        'db_user': 'readonly',
        'db_password': 'secret',
        'db_password_file': None,
        'trace_table': f'trajectory_span_events_pr_{PRINCIPAL_SCOPE_ID}',
        'parent_created_at': None,
        'output': tmp_path / 'events.jsonl',
        'resume': False,
        'follow': False,
        'poll_interval': 1.0,
    }
    arguments.update(overrides)
    return export_cli.run_trace_export(argparse.Namespace(**arguments)), captured


def test_trace_export_cli_enables_boundless_when_partition_key_supplied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    code, captured = _run_trace_export(monkeypatch, tmp_path, parent_created_at=PARENT_CREATED_AT_ISO)

    assert code == 0
    assert captured['source'].dialect == 'boundless'
    # follow.py derives every read's metadata from this scope.
    assert captured['scope']['parent_created_at'] == PARENT_CREATED_AT_SQL


def test_trace_export_cli_stays_on_tdsql_without_partition_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    code, captured = _run_trace_export(monkeypatch, tmp_path)

    assert code == 0
    assert captured['source'].dialect == 'tdsql'
    assert 'parent_created_at' not in captured['scope']


def test_trace_export_cli_rejects_malformed_partition_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    code, captured = _run_trace_export(monkeypatch, tmp_path, parent_created_at='not-a-datetime')

    assert code == 1
    assert captured == {}


# --- extend_run_trace_sinks -------------------------------------------------
#
# A Boundless entry is configured here rather than in run_trace_sinks so that builds
# predating the field never attempt it. The outage this guards against came from a key
# placed one level too deep: inside a sink entry's `config`, which is dict[str, Any] and
# therefore reaches the constructor untouched.


def _tdsql_entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        'kind': 'tdsql',
        'config': {'host': 'db.test', 'port': 3306, 'user': 'u', 'db': 'runner'},
        'secret_config': {'password': 'pwd-ref'},
        'sink_class': 'nexus.runner.trace.tdsql::TDSQLTraceSink',
        'source_class': 'nexus.runner.trace.tdsql::TDSQLTraceSource',
    }
    entry.update(overrides)
    return entry


def _boundless_entry() -> dict[str, Any]:
    return _tdsql_entry(
        sink_class='nexus.runner.trace.tdsql::TDSQLBoundlessTraceSink',
        source_class='nexus.runner.trace.tdsql::TDSQLBoundlessTraceSource',
    )


def test_extension_entries_reach_every_reader_of_run_trace_sinks() -> None:
    from nexus.distributed.config.model import DistributedConfig

    config = DistributedConfig.model_validate(
        {
            'run_trace_sinks': {'tdsql': _tdsql_entry()},
            'extend_run_trace_sinks': {'tdsql-boundless': _boundless_entry()},
        }
    )

    # Merged into the canonical field so the existing readers need no change; leaving them
    # to merge individually would make the backend exist for writes but not for reads.
    assert sorted(config.run_trace_sinks) == ['tdsql', 'tdsql-boundless']


def test_canonical_entry_wins_over_the_extension_copy() -> None:
    from nexus.distributed.config.model import DistributedConfig

    config = DistributedConfig.model_validate(
        {
            'run_trace_sinks': {'shared': _tdsql_entry(display_name='canonical')},
            'extend_run_trace_sinks': {'shared': _tdsql_entry(display_name='extension')},
        }
    )

    # Lets an entry be promoted out of the extension without a flag day: add it to
    # run_trace_sinks first, delete it from the extension once every build has caught up.
    assert config.run_trace_sinks['shared'].display_name == 'canonical'


def test_extension_backend_names_are_validated() -> None:
    from nexus.distributed.config.model import DistributedConfig

    with pytest.raises(ValueError):
        DistributedConfig.model_validate({'extend_run_trace_sinks': {'bad name!': _tdsql_entry()}})


def test_older_build_ignores_the_extension_instead_of_failing() -> None:
    """The whole point: a build without the field must still load the config."""
    from nexus.distributed.config.store import _validate_distributed_config_for_db_read

    stored = {
        'run_trace_sinks': {'tdsql': _tdsql_entry()},
        # Stands in for the field such a build lacks; model_validate rejects it as
        # extra_forbidden exactly as an older DistributedConfig would.
        'unreleased_future_field': {'tdsql-boundless': _boundless_entry()},
    }

    config = _validate_distributed_config_for_db_read(
        stored,
        stored_field_scopes={('unreleased_future_field',): ('*', '*')},
    )

    # Stripped with a warning, and the backend is simply absent rather than half-built.
    assert sorted(config.run_trace_sinks) == ['tdsql']


def test_unknown_key_inside_a_sink_entry_config_still_reaches_the_constructor() -> None:
    """Regression guard for the outage: `config` bypasses validation entirely."""
    from dataclasses import dataclass, field

    from nexus.distributed.config.model import RunTraceSinkConfig

    @dataclass(slots=True)
    class SourceBeforeTheDialectParameter:
        """The signature shipped in the wheels that crashed; kept as the reference point."""

        host: str
        user: str
        password: str = field(default='', repr=False)
        db: str = ''
        port: int = 3306

    entry = _tdsql_entry()
    entry['config'] = {**entry['config'], 'dialect': 'boundless'}
    model = RunTraceSinkConfig.model_validate(entry)

    # `config` is dict[str, Any]: the key survives validation untouched and is then
    # expanded into the sink constructor, so adding one here is equivalent to editing that
    # signature. This is why the Boundless flavour is a separate class behind
    # extend_run_trace_sinks rather than a flag on the shared entry.
    assert model.config['dialect'] == 'boundless'
    with pytest.raises(TypeError, match='dialect'):
        SourceBeforeTheDialectParameter(**{**model.config, 'password': 'pwd'})

    # Same value one level up is rejected as extra_forbidden, which the DB read path
    # strips -- the difference between a warning and a crash loop.
    with pytest.raises(ValueError):
        RunTraceSinkConfig.model_validate({**_tdsql_entry(), 'dialect': 'boundless'})


def test_boundless_classes_bind_the_dialect_without_a_config_flag() -> None:
    from nexus.runner.trace.tdsql import TDSQLBoundlessTraceSink, TDSQLBoundlessTraceSource

    kwargs = {'host': 'db.test', 'user': 'u', 'password': 'pwd', 'db': 'runner', 'port': 3306}

    assert TDSQLBoundlessTraceSink(**kwargs).dialect == BOUNDLESS_DIALECT
    assert TDSQLBoundlessTraceSource(**kwargs).dialect == BOUNDLESS_DIALECT
    # The shared classes must stay on the classic dialect for the three DCDB backends.
    assert TDSQLTraceSink(**kwargs).dialect == TDSQL_DIALECT
    assert TDSQLTraceSource(**kwargs).dialect == TDSQL_DIALECT


def test_boundless_class_paths_resolve_from_configuration() -> None:
    """A bad class path would only surface at run time, so pin it in configuration form."""
    from nexus.core.registry import import_from_path

    entry = _boundless_entry()
    assert import_from_path(entry['sink_class']).__name__ == 'TDSQLBoundlessTraceSink'
    assert import_from_path(entry['source_class']).__name__ == 'TDSQLBoundlessTraceSource'


# --- extend_trace_database_access --------------------------------------------------
# The same escape hatch for the maintenance side of a backend. It guards a sharper edge
# than the sink one: an unrecognised provider inside trace_database_access is not an
# unknown *key* but an unknown discriminator value, which the read path does not strip.


def _access_entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        'read_host': 'db.test',
        'read_port': 3306,
        'database': 'runner',
        'maintenance_credential_ref': 'tdsql-maintenance',
        'account_provider': {
            'provider_type': 'tencent_cloud_tdmysql',
            'region': 'ap-nanjing',
            'instance_id': 'tdsql-49qrr9xo6y',
            'api_endpoint': 'tdmysql.internal.tencentcloudapi.com',
            'account_host': '%',
        },
    }
    entry.update(overrides)
    return entry


def _secrets() -> dict[str, Any]:
    return {
        'internal_token': 'x' * 32,
        'trace_database_credentials': {
            'tdsql-maintenance': {'provider_type': 'tencent_cloud_dcdb', 'secret_id': 'AKID', 'secret_key': 'k'},
        },
    }


def test_trace_database_access_extension_reaches_every_reader() -> None:
    from nexus.distributed.config.model import DistributedConfig

    config = DistributedConfig.model_validate(
        {
            'extend_run_trace_sinks': {'tdsql-boundless': _boundless_entry()},
            'extend_trace_database_access': {'tdsql-boundless': _access_entry()},
        }
    )

    assert sorted(config.trace_database_access) == ['tdsql-boundless']


def test_canonical_trace_database_access_wins_over_the_extension_copy() -> None:
    from nexus.distributed.config.model import DistributedConfig

    config = DistributedConfig.model_validate(
        {
            'run_trace_sinks': {'shared': _tdsql_entry()},
            'trace_database_access': {'shared': _access_entry(read_host='canonical.test')},
            'extend_trace_database_access': {'shared': _access_entry(read_host='extension.test')},
        }
    )

    # Promotion out of the extension needs no flag day, matching run_trace_sinks.
    assert config.trace_database_access['shared'].read_host == 'canonical.test'


def test_extension_entry_is_validated_like_a_canonical_one() -> None:
    """Merging before the cross-field checks, not after, is what makes this hold."""
    from nexus.distributed.config.model import DistributedConfig

    # Every access entry must name a configured backend. Were the merge to run after that
    # check, a typo here would reach the maintenance path unvalidated.
    with pytest.raises(ValueError, match='not configured'):
        DistributedConfig.model_validate(
            {
                'run_trace_sinks': {'tdsql': _tdsql_entry()},
                'extend_trace_database_access': {'typo-boundless': _access_entry()},
            }
        )


def test_extension_entry_is_checked_against_the_credentials() -> None:
    from nexus.distributed.config.model import DistributedConfig

    config = DistributedConfig.model_validate(
        {
            'extend_run_trace_sinks': {'tdsql-boundless': _boundless_entry()},
            'extend_trace_database_access': {'tdsql-boundless': _access_entry()},
            'secrets': _secrets(),
            'database_url': 'mysql://u:p@h/d',
        }
    )

    # The control gateway resolves maintenance credentials for every enabled entry, so an
    # extension entry referencing a missing credential must fail here rather than at use.
    config.validate_for_service('control-gateway')


def test_unknown_provider_in_the_canonical_field_is_not_stripped() -> None:
    """Why this field exists at all -- the read path cannot rescue the canonical one."""
    import pydantic

    from nexus.distributed.config.store import _validate_distributed_config_for_db_read

    stored = {
        'run_trace_sinks': {'tdsql': _tdsql_entry()},
        'trace_database_access': {
            'tdsql': _access_entry(
                account_provider={
                    'provider_type': 'a_product_this_build_lacks',
                    'region': 'ap-nanjing',
                    'instance_id': 'tdsql-49qrr9xo6y',
                }
            )
        },
    }

    with pytest.raises(pydantic.ValidationError) as excinfo:
        _validate_distributed_config_for_db_read(
            stored,
            stored_field_scopes={('trace_database_access',): ('*', '*')},
        )

    # union_tag_invalid, not extra_forbidden: the strip only removes the latter, so the
    # whole config is rejected and every service stops, not just those using the backend.
    assert [error['type'] for error in excinfo.value.errors()] == ['union_tag_invalid']


def test_older_build_ignores_the_trace_database_access_extension() -> None:
    """The same entry under a key the build never heard of is merely dropped."""
    from nexus.distributed.config.store import _validate_distributed_config_for_db_read

    # Stands in for extend_trace_database_access as an older DistributedConfig sees it:
    # an unknown top-level key, hence extra_forbidden, hence strippable.
    field = 'unreleased_access_extension_field'
    stored = {'run_trace_sinks': {'tdsql': _tdsql_entry()}, field: {'tdsql-boundless': _access_entry()}}

    config = _validate_distributed_config_for_db_read(stored, stored_field_scopes={(field,): ('*', '*')})

    assert sorted(config.run_trace_sinks) == ['tdsql']
    assert config.trace_database_access == {}


# --- saving a config that carries both fields ----------------------------------------
# The merges are a runtime view. Persisting that view would copy the extension entries
# into the ordinary fields, where an older build meets them head-on -- so a plain save of
# a working config would be enough to lock those builds out.


def _mixed_override_values() -> dict[str, Any]:
    return {
        'run_trace_sinks': {'tdsql': _tdsql_entry()},
        'extend_run_trace_sinks': {'tdsql-boundless': _boundless_entry()},
        'trace_database_access': {'tdsql': _access_entry(account_provider={'provider_type': 'mysql'})},
        'extend_trace_database_access': {'tdsql-boundless': _access_entry()},
    }


def test_saving_leaves_extension_entries_in_the_extension_fields() -> None:
    from nexus.distributed.config.store import _normalize_override_values

    saved = _normalize_override_values(_mixed_override_values(), secret=False, role='*', identity='*')

    assert sorted(saved['run_trace_sinks']) == ['tdsql']
    assert sorted(saved['extend_run_trace_sinks']) == ['tdsql-boundless']
    assert sorted(saved['trace_database_access']) == ['tdsql']
    assert sorted(saved['extend_trace_database_access']) == ['tdsql-boundless']


def test_an_older_build_can_still_read_a_saved_config() -> None:
    """The compatibility this guards is round-trip, not just first write."""
    from nexus.distributed.config.store import _normalize_override_values, _validate_distributed_config_for_db_read

    saved = _normalize_override_values(_mixed_override_values(), secret=False, role='*', identity='*')

    # A build predating the two extension fields sees them as unknown top-level keys, which
    # the read path strips; anything the save promoted into the ordinary fields it cannot.
    sink_field = 'unreleased_future_field'
    access_field = 'unreleased_access_extension_field'
    stored = {
        'run_trace_sinks': saved['run_trace_sinks'],
        'trace_database_access': saved['trace_database_access'],
        sink_field: saved['extend_run_trace_sinks'],
        access_field: saved['extend_trace_database_access'],
    }

    config = _validate_distributed_config_for_db_read(
        stored,
        stored_field_scopes={(sink_field,): ('*', '*'), (access_field,): ('*', '*')},
    )

    assert sorted(config.run_trace_sinks) == ['tdsql']
    assert sorted(config.trace_database_access) == ['tdsql']


def test_saving_keeps_a_promoted_entry_in_the_ordinary_field() -> None:
    """Promotion out of the extension must survive a save, or it would be undone."""
    from nexus.distributed.config.store import _normalize_override_values

    values = {
        'run_trace_sinks': {'shared': _tdsql_entry(display_name='canonical')},
        'extend_run_trace_sinks': {'shared': _tdsql_entry(display_name='extension')},
        'trace_database_access': {'shared': _access_entry(read_host='canonical.test')},
        'extend_trace_database_access': {'shared': _access_entry(read_host='extension.test')},
    }

    saved = _normalize_override_values(values, secret=False, role='*', identity='*')

    assert saved['run_trace_sinks']['shared']['display_name'] == 'canonical'
    assert saved['trace_database_access']['shared']['read_host'] == 'canonical.test'
    assert saved['extend_run_trace_sinks']['shared']['display_name'] == 'extension'
    assert saved['extend_trace_database_access']['shared']['read_host'] == 'extension.test'


def test_saving_only_the_extension_field_does_not_materialise_the_ordinary_one() -> None:
    from nexus.distributed.config.store import _normalize_override_values

    values = {
        'run_trace_sinks': {'tdsql': _tdsql_entry()},
        'extend_run_trace_sinks': {'tdsql-boundless': _boundless_entry()},
        'extend_trace_database_access': {'tdsql-boundless': _access_entry()},
    }

    saved = _normalize_override_values(values, secret=False, role='*', identity='*')

    # Only the submitted fields are written, so the ordinary access field keeps whatever a
    # wider scope set rather than gaining an empty or extension-filled value.
    assert sorted(saved) == ['extend_run_trace_sinks', 'extend_trace_database_access', 'run_trace_sinks']
    assert sorted(saved['run_trace_sinks']) == ['tdsql']


# --- midnight partition race ---------------------------------------------------------
# The Boundless table ships with no partition for tomorrow; SET INTERVAL adds one when the
# first out-of-range row arrives. Several nodes reach that point together after midnight,
# so the losers see 1517 -- once a day, on the day's first writes, not rarely.


class _RacingCursor(AsyncCursorMixin):
    """Rejects the first ``failures`` writes with 1517, as a losing node is told."""

    def __init__(self, *, failures: int) -> None:
        self.remaining = failures
        self.attempts = 0

    async def _maybe_fail(self) -> None:
        self.attempts += 1
        if self.remaining <= 0:
            return
        self.remaining -= 1
        from asyncmy.errors import OperationalError

        raise OperationalError(1517, 'Duplicate partition name _p20260923000000')

    async def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        await self._maybe_fail()

    async def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        await self._maybe_fail()


class _RacingConnection:
    def __init__(self, *, failures: int) -> None:
        self.cursor_instance = _RacingCursor(failures=failures)
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _RacingCursor:
        return self.cursor_instance

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1


@pytest.mark.asyncio
async def test_partition_race_is_retried_rather_than_dropping_the_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tdsql_writer, '_PARTITION_RACE_BACKOFF_SECONDS', 0.0)
    writer = tdsql_writer.TDSQLTraceWriter(dialect=BOUNDLESS_DIALECT)
    connection = _RacingConnection(failures=1)
    writer._conn = connection

    async def do_write(conn: Any) -> None:
        async with conn.cursor() as cur:
            await cur.executemany('INSERT INTO t VALUES (%s)', [(1,)])

    await writer._execute_with_reconnect(do_write)

    # The winner has published the partition by now, so the retry lands. Dropping instead
    # would lose a run's trace every day the first write crosses midnight.
    assert connection.cursor_instance.attempts == 2
    assert connection.commits == 1
    assert connection.rollbacks == 1


@pytest.mark.asyncio
async def test_partition_race_retry_does_not_reconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    """The connection is healthy -- only the DDL lost. Reconnecting would be wasted work."""
    monkeypatch.setattr(tdsql_writer, '_PARTITION_RACE_BACKOFF_SECONDS', 0.0)
    writer = tdsql_writer.TDSQLTraceWriter(dialect=BOUNDLESS_DIALECT)
    writer._conn = _RacingConnection(failures=1)

    async def fail_reconnect() -> None:
        raise AssertionError('a partition race must not reconnect')

    monkeypatch.setattr(writer, '_reconnect', fail_reconnect)

    async def do_write(conn: Any) -> None:
        async with conn.cursor() as cur:
            await cur.executemany('INSERT INTO t VALUES (%s)', [(1,)])

    await writer._execute_with_reconnect(do_write)


@pytest.mark.asyncio
async def test_relentless_partition_error_still_surfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrying is bounded: a genuine layout problem must not spin forever."""
    from asyncmy.errors import OperationalError

    monkeypatch.setattr(tdsql_writer, '_PARTITION_RACE_BACKOFF_SECONDS', 0.0)
    writer = tdsql_writer.TDSQLTraceWriter(dialect=BOUNDLESS_DIALECT)
    connection = _RacingConnection(failures=99)
    writer._conn = connection

    async def do_write(conn: Any) -> None:
        async with conn.cursor() as cur:
            await cur.executemany('INSERT INTO t VALUES (%s)', [(1,)])

    with pytest.raises(OperationalError):
        await writer._execute_with_reconnect(do_write)

    assert connection.cursor_instance.attempts == tdsql_writer._MAX_PARTITION_RACE_ATTEMPTS + 1


def test_partition_race_code_is_not_treated_as_a_connection_fault() -> None:
    """1517 must not reach the reconnect path, which would churn a healthy connection."""
    from asyncmy.errors import OperationalError

    exc = OperationalError(1517, 'Duplicate partition name _p20260923000000')

    assert tdsql_writer._is_partition_race_error(exc)
    assert not tdsql_writer._is_recoverable_error(exc)

    transient = OperationalError(8502, 'Query failed for write fence error 341654(EC_TDS_SCHEMA_CHECK_VERSION_MISMATCH)')
    assert tdsql_writer._is_partition_race_error(transient)
    assert not tdsql_writer._is_recoverable_error(transient)
