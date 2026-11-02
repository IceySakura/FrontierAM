from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from nexus.runner.trace.tdsql.partition_maintenance import (
    TDSQLPartitionMaintenance,
    TDSQLPartitionMaintenanceError,
    TDSQLPartitionTarget,
    _add_partition_sql,
    _is_partition_race,
    _is_transient_database_error,
    _missing_partitions,
    _partition_boundaries,
)


class _MySQLError(Exception):
    def __init__(self, errno, message):
        super().__init__(errno, message)
        self.errno = errno


class _Cursor:
    def __init__(self, rows, execute_error=None, alter_errors=None, refresh_rows=None):
        self.rows = rows
        self.execute_error = execute_error
        self.alter_errors = list(alter_errors or [])
        self.refresh_rows = list(refresh_rows or [])
        self.executed = []
        self._pending = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if sql.startswith('ALTER'):
            error = self.alter_errors.pop(0) if self.alter_errors else self.execute_error
            if error is not None:
                raise error
            return
        if 'TABLE_NAME = %s' in sql and self.refresh_rows:
            self._pending = self.refresh_rows.pop(0)
            return
        self._pending = self.rows

    async def fetchall(self):
        return self._pending


class _Connection:
    def __init__(self, cursor):
        self.cursor_obj = cursor
        self.commits = 0

    def cursor(self):
        return self.cursor_obj

    async def commit(self):
        self.commits += 1

    def close(self):
        pass


def test_missing_partitions_materializes_a_full_future_window() -> None:
    now = datetime(2026, 9, 24, 12)

    additions = _missing_partitions(
        [datetime(2026, 9, 26)],
        now=now,
        horizon=timedelta(days=7),
    )

    assert [name for name, _ in additions] == [
        'p20260926',
        'p20260927',
        'p20260928',
        'p20260929',
        'p20260930',
        'p20261001',
    ]
    assert [boundary for _, boundary in additions] == [
        datetime(2026, 9, 27),
        datetime(2026, 9, 28),
        datetime(2026, 9, 29),
        datetime(2026, 9, 30),
        datetime(2026, 10, 1),
        datetime(2026, 10, 2),
    ]


def test_partition_rows_ignore_non_principal_tables_and_maxvalue() -> None:
    rows = [
        ('trajectory_span_events_pr_' + 'a' * 32, 'p20260924', '2026-09-25 00:00:00'),
        ('trajectory_span_events_pr_' + 'a' * 32, 'p20260925', '2026-09-26 00:00:00'),
        ('other_table', 'p0', '2026-09-25 00:00:00'),
        ('trajectory_span_events_pr_' + 'b' * 32, 'pmax', 'MAXVALUE'),
    ]

    result = _partition_boundaries(rows)

    assert sorted(result) == ['trajectory_span_events_pr_' + 'a' * 32]
    assert result['trajectory_span_events_pr_' + 'a' * 32][-1] == datetime(2026, 9, 26)


def test_partition_rows_collapse_repeated_subpartition_rows() -> None:
    table = 'trajectory_span_events_pr_' + 'a' * 32
    rows = [(table, 'p20260924', '2026-09-25 00:00:00') for _ in range(12)]

    result = _partition_boundaries(rows)

    assert result[table] == [datetime(2026, 9, 25)]


def test_partition_boundaries_accept_parenthesized_column_bounds() -> None:
    table = 'trajectory_span_events_pr_' + 'a' * 32

    result = _partition_boundaries([(table, 'p20260925', "('2026-09-26 00:00:00')")])

    assert result[table] == [datetime(2026, 9, 26)]


def test_add_partition_sql_quotes_generated_identifiers() -> None:
    table = 'trajectory_span_events_pr_' + 'a' * 32

    sql = _add_partition_sql(table, [('p20260926', datetime(2026, 9, 27))])

    assert sql == (
        f"ALTER TABLE `{table}` ADD PARTITION (PARTITION `p20260926` VALUES LESS THAN ('2026-09-27 00:00:00'))"
    )


@pytest.mark.parametrize('database', ['runner', 'runner_prod'])
@pytest.mark.asyncio
async def test_reconcile_scans_and_adds_partitions_for_configured_database(database: str) -> None:
    table = 'trajectory_span_events_pr_' + 'a' * 32
    # PARTITIONS exposes one row per subpartition for a RANGE + KEY table. The
    # reconciler must use the repeated outer partition boundary from those rows.
    cursor = _Cursor(
        [
            (table, 'p20260925', '2026-09-26 00:00:00'),
            (table, 'p20260925', '2026-09-26 00:00:00'),
        ]
    )
    connection = _Connection(cursor)

    connection_config = {}

    async def connect(**kwargs):
        connection_config.update(kwargs)
        return connection

    maintenance = TDSQLPartitionMaintenance(
        (
            TDSQLPartitionTarget(
                backend_name='boundless',
                host='db',
                user='writer',
                password='secret',
                database=database,
            ),
        ),
        connector=connect,
    )

    changed = await maintenance.reconcile_once(now=datetime(2026, 9, 24, 12))

    assert changed == 6
    assert maintenance.last_stats.scanned == 1
    assert connection.commits == 1
    assert len(cursor.executed) == 2
    assert cursor.executed[0][1] == (database,)
    assert connection_config['db'] == database
    # Boundless trace tables always carry subpartitions, so a bare
    # SUBPARTITION_NAME IS NULL filter would match no row at all.
    assert 'SUBPARTITION_ORDINAL_POSITION = 1' in cursor.executed[0][0]
    assert 'AND SUBPARTITION_NAME IS NULL' not in cursor.executed[0][0]


@pytest.mark.asyncio
async def test_duplicate_partition_race_is_idempotent() -> None:
    table = 'trajectory_span_events_pr_' + 'a' * 32
    cursor = _Cursor(
        [(table, 'p20260925', '2026-09-26 00:00:00')],
        alter_errors=[RuntimeError('1517 Duplicate partition name')],
        # A peer reconciler already materialised the whole window.
        refresh_rows=[[(table, 'p20261001', '2026-10-02 00:00:00')]],
    )
    connection = _Connection(cursor)

    async def connect(**_kwargs):
        return connection

    maintenance = TDSQLPartitionMaintenance(
        (TDSQLPartitionTarget(backend_name='boundless', host='db', user='writer', database='runner'),),
        connector=connect,
    )

    assert await maintenance.reconcile_once(now=datetime(2026, 9, 24, 12)) == 0
    assert connection.commits == 1
    assert [sql.startswith('ALTER') for sql, _ in cursor.executed] == [False, True, False]


@pytest.mark.asyncio
async def test_interval_auto_partition_race_replans_from_refreshed_boundaries() -> None:
    table = 'trajectory_span_events_pr_' + 'a' * 32
    # INTERVAL auto-partitioning created 2026-09-26 under its own name, so the planned
    # bound is no longer strictly increasing and MySQL answers 1493 instead of 1517.
    cursor = _Cursor(
        [(table, 'p20260925', '2026-09-26 00:00:00')],
        alter_errors=[_MySQLError(1493, 'VALUES LESS THAN value must be strictly increasing for each partition')],
        refresh_rows=[
            [
                (table, 'p20260925', '2026-09-26 00:00:00'),
                (table, '_p20260927', '2026-09-27 00:00:00'),
            ]
        ],
    )
    connection = _Connection(cursor)

    async def connect(**_kwargs):
        return connection

    maintenance = TDSQLPartitionMaintenance(
        (TDSQLPartitionTarget(backend_name='boundless', host='db', user='writer', database='runner'),),
        connector=connect,
    )

    changed = await maintenance.reconcile_once(now=datetime(2026, 9, 24, 12))

    assert changed == 5
    assert connection.commits == 1
    refresh_sql, refresh_params = cursor.executed[2]
    assert 'TABLE_NAME = %s' in refresh_sql
    assert refresh_params == ('runner', table)
    retried_sql = cursor.executed[3][0]
    assert '`p20260926`' not in retried_sql
    assert retried_sql.startswith(f'ALTER TABLE `{table}` ADD PARTITION (PARTITION `p20260927`')


@pytest.mark.asyncio
async def test_unresolved_partition_race_stays_transient_instead_of_fatal() -> None:
    table = 'trajectory_span_events_pr_' + 'a' * 32
    cursor = _Cursor(
        [(table, 'p20260925', '2026-09-26 00:00:00')],
        execute_error=_MySQLError(1493, 'VALUES LESS THAN value must be strictly increasing for each partition'),
    )
    connection = _Connection(cursor)

    async def connect(**_kwargs):
        return connection

    maintenance = TDSQLPartitionMaintenance(
        (TDSQLPartitionTarget(backend_name='boundless', host='db', user='writer', database='runner'),),
        race_retries=2,
        connector=connect,
    )

    with pytest.raises(TDSQLPartitionMaintenanceError):
        await maintenance.reconcile_once(now=datetime(2026, 9, 24, 12))

    assert sum(1 for sql, _ in cursor.executed if sql.startswith('ALTER')) == 3


def test_partition_race_error_is_transient_for_the_reconciler_loop() -> None:
    # A fatal classification here would cancel the core loop and the lease heartbeat.
    from nexus.distributed.control_plane.reconciler import _is_transient_loop_error

    assert _is_transient_loop_error(TDSQLPartitionMaintenanceError('lost race'))


def test_ddl_conflict_error_is_transient_for_partition_maintenance() -> None:
    from nexus.runner.trace.tdsql._driver import OperationalError

    error = OperationalError(8506, 'A conflict DDL job is ongoing; please wait and retry later')

    assert _is_transient_database_error(error)


@pytest.mark.parametrize(
    ('error', 'expected'),
    [
        (_MySQLError(1493, 'VALUES LESS THAN value must be strictly increasing'), True),
        (_MySQLError(1517, 'Duplicate partition name p20260926'), True),
        (_MySQLError(1050, "Table 'x' already exists"), False),
        (RuntimeError('1517 Duplicate partition name'), True),
        (RuntimeError('unknown column parent_created_at'), False),
        # Race detection reads the wording, so a code quoted in an unrelated failure -- or a
        # partition whose name happens to contain one -- must not trigger a retry.
        (RuntimeError('syntax error near PARTITION p1493 VALUES LESS THAN'), False),
        (RuntimeError('cannot parse statement, error 1493 reported by proxy'), False),
    ],
)
def test_partition_race_detection_covers_both_ddl_codes(error: Exception, expected: bool) -> None:
    assert _is_partition_race(error) is expected


@pytest.mark.asyncio
async def test_unexpected_database_error_is_not_hidden_as_transient() -> None:
    table = 'trajectory_span_events_pr_' + 'a' * 32
    cursor = _Cursor(
        [(table, 'p20260925', '2026-09-26 00:00:00')],
        execute_error=ValueError('invalid partition definition'),
    )
    connection = _Connection(cursor)

    async def connect(**_kwargs):
        return connection

    maintenance = TDSQLPartitionMaintenance(
        (TDSQLPartitionTarget(backend_name='boundless', host='db', user='writer', database='runner'),),
        connector=connect,
    )

    with pytest.raises(ValueError, match='invalid partition definition'):
        await maintenance.reconcile_once(now=datetime(2026, 9, 24, 12))
