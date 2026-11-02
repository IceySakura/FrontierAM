from __future__ import annotations

import logging
import re

from datetime import date, datetime, timedelta

import pytest

from nexus.distributed.config import DistributedConfig
from nexus.distributed.control_plane import terminal_purge
from nexus.runner.trace import principal_trace_scope_id
from nexus.runner.trace.tdsql.schema import principal_trace_table_name


_NOW = datetime(2026, 9, 27, 3, 0)


@pytest.fixture(autouse=True)
def _frozen_clock(monkeypatch) -> None:
    # cutoff_day is relative to now, so the partition fixtures would rot against a real clock.
    monkeypatch.setattr(terminal_purge, 'utcnow', lambda: _NOW)


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class _Session:
    """Stand in for the control plane: the rows are the (owner, day) pairs with live work."""

    def __init__(self, blocking_rows=()):
        self.blocking_rows = blocking_rows
        self.params: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def execute(self, statement, params=None):
        params = dict(params or {})
        self.params.append(params)
        rows = list(self.blocking_rows)
        # Honour a bound batch the way a SQL LIMIT would, so reintroducing one truncates the
        # veto set here too and the coverage below fails instead of passing on a lenient fake.
        if (limit := params.get('partition_batch')) is not None:
            rows = rows[:limit]
        return _Result(rows)


class _SessionFactory:
    def __init__(self, blocking_rows=()):
        self.blocking_rows = blocking_rows
        self.sessions: list[_Session] = []

    def __call__(self):
        session = _Session(self.blocking_rows)
        self.sessions.append(session)
        return session

    @property
    def candidate_params(self) -> list[dict]:
        return [params for session in self.sessions for params in session.params]


_DROP_STATEMENT = re.compile(r'^ALTER TABLE `(?P<table>[^`]+)` DROP PARTITION `(?P<partition>[^`]+)`$')
_NAME_LIKE = re.compile(r"PARTITION_NAME LIKE '(?P<prefix>[^%']*)%")


def _name_predicate(statement: str):
    """Mirror whatever PARTITION_NAME predicate the listing carries, or accept everything."""
    match = _NAME_LIKE.search(statement)
    if match is None:
        return lambda name: True
    prefix = match.group('prefix').replace('\\_', '_')
    return lambda name: name.startswith(prefix)


class _Cursor:
    def __init__(self, existing_rows=()):
        self.statements: list[str] = []
        self.existing_rows = list(existing_rows)
        self.listed_rows: list[tuple] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def execute(self, statement, params=None):
        statement = str(statement)
        self.statements.append(statement)
        if 'information_schema.PARTITIONS' in statement:
            # Apply the listing predicates the same way the server would, so restoring a
            # name bound fails these tests instead of passing on a fake that ignores it.
            self.listed_rows = [row for row in self.existing_rows if _name_predicate(statement)(str(row[2]))]
        # A dropped partition leaves information_schema while its executions stay, which is
        # the asymmetry the multi-pass test below depends on.
        if (match := _DROP_STATEMENT.match(statement)) is not None:
            gone = (match.group('table'), match.group('partition'))
            self.existing_rows = [row for row in self.existing_rows if (row[0], row[2]) != gone]

    async def fetchone(self):
        return (1,)

    async def fetchall(self):
        return list(self.listed_rows)


_OWNER_1_TABLE = 'trajectory_span_events_pr_e6eec6f579711e71ab80b46ea723ecde'
_OWNER_2_TABLE = 'trajectory_span_events_pr_76a914091d1147064f738ce2972f15d2'


def _table_for(owner: str) -> str:
    return principal_trace_table_name(principal_trace_scope_id(owner))


def _bound(value: datetime | date) -> str:
    # information_schema quotes the VALUES LESS THAN literal for RANGE COLUMNS.
    moment = value if isinstance(value, datetime) else datetime.combine(value, datetime.min.time())
    return f"'{moment:%Y-%m-%d %H:%M:%S}'"


def _rows(table_name: str, *partitions: tuple[str, str | None]):
    """Turn (name, upper bound) pairs into listing rows, numbered in RANGE order."""
    return [(table_name, ordinal, name, bound) for ordinal, (name, bound) in enumerate(partitions, start=1)]


def _day_rows(table_name: str, first_day: date, days: int = 1, *, interval_named: bool = False):
    """The catch-all CREATE TABLE seeds, plus ``days`` daily partitions from ``first_day``.

    ``interval_named`` switches to the _pYYYYMMDDhhmmss form Boundless INTERVAL generates,
    whose digits are the exclusive upper bound and so sit one day after the rows it holds.
    """
    partitions: list[tuple[str, str | None]] = [(f'p_before_{first_day:%Y%m%d}', _bound(first_day))]
    for offset in range(days):
        day = first_day + timedelta(days=offset)
        upper = datetime.combine(day + timedelta(days=1), datetime.min.time())
        name = f'_p{upper:%Y%m%d%H%M%S}' if interval_named else f'p{day:%Y%m%d}'
        partitions.append((name, _bound(upper)))
    return _rows(table_name, *partitions)


class _Connection:
    def __init__(self, existing_rows=None):
        if existing_rows is None:
            existing_rows = _day_rows(_OWNER_1_TABLE, date(2026, 9, 1))
        self.cursor_instance = _Cursor(existing_rows)
        self.committed = False
        self.closed = False

    def cursor(self):
        return self.cursor_instance

    async def commit(self):
        self.committed = True

    async def ensure_closed(self):
        self.closed = True

    @property
    def drop_statements(self) -> list[str]:
        return [s for s in self.cursor_instance.statements if _DROP_STATEMENT.match(s) is not None]


def test_blocking_sql_uses_root_executions_only() -> None:
    sql = terminal_purge._BLOCKING_SQL

    # Batch parent status gates its children, so only root executions are scanned: no child
    # join, no parent status term. The Boundless floor bound must survive that simplification.
    assert 'e.parent_execution_id IS NULL' in sql
    assert 'UNION ALL' not in sql
    assert 'JOIN executions parent' not in sql
    assert 'e.status NOT IN' in sql
    assert 'parent.status' not in sql
    assert ':floor_datetime' in sql


def test_blocking_sql_filters_status_instead_of_grouping_every_day() -> None:
    # A HAVING form returns one row per (owner, day) and says nothing about a day with no
    # rows at all, which is exactly the empty-partition case.
    sql = terminal_purge._BLOCKING_SQL

    assert 'HAVING' not in sql
    assert 'AND e.status NOT IN' in sql


def test_blocking_sql_is_not_limited() -> None:
    # A LIMIT here would silently shrink the veto set and drop partitions that are in use;
    # terminal_purge_partition_batch caps the DDL instead.
    assert 'LIMIT' not in terminal_purge._BLOCKING_SQL
    assert ':partition_batch' not in terminal_purge._BLOCKING_SQL


def test_partition_listing_selects_boundaries_not_names() -> None:
    sql = terminal_purge._EXISTING_PARTITIONS_SQL

    # A name bound would hide every INTERVAL partition, and the ordering is what lets a
    # partition inherit its lower bound from the predecessor's PARTITION_DESCRIPTION.
    assert "PARTITION_NAME LIKE 'p" not in sql
    assert 'PARTITION_DESCRIPTION' in sql
    assert 'PARTITION_ORDINAL_POSITION' in sql
    assert 'ORDER BY TABLE_NAME, PARTITION_ORDINAL_POSITION' in sql


@pytest.mark.parametrize(
    ('description', 'expected'),
    [
        ("'2026-09-02 00:00:00'", datetime(2026, 9, 2)),
        ('2026-09-02 00:00:00', datetime(2026, 9, 2)),
        ("'2026-09-02 00:00:00.000000'", datetime(2026, 9, 2)),
        ("'2026-09-02 08:00:00+08:00'", datetime(2026, 9, 2)),
        ('MAXVALUE', None),
        ('maxvalue', None),
        (None, None),
        ("'not a datetime'", None),
        ("''", None),
    ],
)
def test_partition_bound_parsing(description, expected) -> None:
    assert terminal_purge._partition_bound(description) == expected


def _config() -> DistributedConfig:
    return DistributedConfig(
        terminal_data_purge_enabled=True,
        terminal_purge_trace_backend='boundless',
        run_trace_sinks={
            'boundless': {
                'sinkClass': 'nexus.runner.trace.tdsql::TDSQLBoundlessTraceSink',
                'sourceClass': 'nexus.runner.trace.tdsql::TDSQLBoundlessTraceSource',
                'config': {'host': 'trace.example', 'port': 3306, 'user': 'writer', 'db': 'trace'},
                'secretConfig': {'password': 'trace-password'},
            }
        },
        trace_database_access={
            'boundless': {
                'read_host': 'trace.example',
                'database': 'trace',
                'maintenance_credential_ref': 'trace-maintenance',
            }
        },
        secrets={
            'trace_database_credentials': {
                'trace-maintenance': {'username': 'maintenance', 'password': 'maintenance-password'}
            }
        },
    )


@pytest.mark.asyncio
async def test_reconcile_drops_only_existing_generated_partition(monkeypatch, caplog) -> None:
    caplog.set_level(logging.INFO, logger='nexus.distributed.control_plane.terminal_purge')
    connection = _Connection()

    async def connect(**kwargs):
        assert kwargs['user'] == 'maintenance'
        assert kwargs['db'] == 'trace'
        return connection

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', connect)
    reconciler = terminal_purge.BoundlessPartitionPurgeReconciler(
        session_factory=_SessionFactory(),
        config=_config(),
    )

    assert await reconciler.reconcile_once() == 1
    assert connection.committed is True
    assert connection.closed is True
    assert connection.drop_statements == [f'ALTER TABLE `{_OWNER_1_TABLE}` DROP PARTITION `p20260901`']
    assert 'sql=blocking_query elapsed_ms=' in caplog.text
    assert 'sql=blocking_recheck elapsed_ms=' in caplog.text
    assert 'sql=existing_partitions elapsed_ms=' in caplog.text
    assert 'sql=drop_partition elapsed_ms=' in caplog.text


def _terminal_owners(count: int, day: date = date(2026, 9, 1)):
    """One table per owner with a single expired partition, all terminal on ``day``."""
    owners = [f'owner-{index:02d}' for index in range(count)]
    rows = [row for owner in owners for row in _day_rows(_table_for(owner), day)]
    return _Connection(existing_rows=rows), [(owner, day) for owner in owners]


def _reconciler_for(connection, monkeypatch, *, blocking_rows=()):
    async def connect(**kwargs):
        return connection

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', connect)
    return terminal_purge.BoundlessPartitionPurgeReconciler(
        session_factory=_SessionFactory(blocking_rows=blocking_rows),
        config=_config(),
    )


@pytest.mark.asyncio
async def test_reconcile_drops_interval_generated_partitions(monkeypatch) -> None:
    """Every partition after the first is created by INTERVAL and named _pYYYYMMDDhhmmss.

    Bounding the listing with LIKE 'p%' hid all of them, so a real deployment could only
    ever purge the single partition CREATE TABLE seeds -- once, and never again.
    """
    connection = _Connection(existing_rows=_day_rows(_OWNER_1_TABLE, date(2026, 9, 1), days=3, interval_named=True))
    reconciler = _reconciler_for(connection, monkeypatch)

    assert await reconciler.reconcile_once() == 3
    # Named after the exclusive upper bound, so each name is a day ahead of what it holds.
    assert connection.drop_statements == [
        f'ALTER TABLE `{_OWNER_1_TABLE}` DROP PARTITION `_p2026090{day}000000`' for day in (2, 3, 4)
    ]


@pytest.mark.asyncio
async def test_reconcile_maps_a_blocked_day_to_the_partition_that_holds_it(monkeypatch) -> None:
    """An INTERVAL name is one day ahead of its rows, so the name is never read as a date.

    Live 09-01 lives in _p20260902000000, so that one must stay while _p20260903000000
    holding 09-02 still goes: parsing the digits would invert both decisions and delete a
    day whose executions are still running.
    """
    connection = _Connection(existing_rows=_day_rows(_OWNER_1_TABLE, date(2026, 9, 1), days=2, interval_named=True))
    reconciler = _reconciler_for(connection, monkeypatch, blocking_rows=[('owner-1', date(2026, 9, 1))])

    assert await reconciler.reconcile_once() == 1
    assert connection.drop_statements == [
        f'ALTER TABLE `{_OWNER_1_TABLE}` DROP PARTITION `_p20260903000000`',
    ]


@pytest.mark.asyncio
async def test_reconcile_reclaims_a_partition_that_absorbed_its_neighbours(monkeypatch) -> None:
    """Dropping a partition widens the next one, so refusing wide ones leaked them forever.

    A lower bound is the previous upper bound, so each pass left its successor spanning the
    gap; every day inside is checked instead, which reclaims the partition and self-heals.
    """
    connection = _Connection(
        existing_rows=_rows(
            _OWNER_1_TABLE,
            ('p_before_20260801', _bound(date(2026, 8, 1))),
            ('_p20260901000000', _bound(date(2026, 9, 1))),
        )
    )
    reconciler = _reconciler_for(connection, monkeypatch)

    assert await reconciler.reconcile_once() == 1
    assert connection.drop_statements == [f'ALTER TABLE `{_OWNER_1_TABLE}` DROP PARTITION `_p20260901000000`']


@pytest.mark.asyncio
async def test_reconcile_keeps_a_wide_partition_when_any_day_inside_is_live(monkeypatch) -> None:
    """One live day anywhere inside vetoes the whole partition, not just its first day.

    Checking the first day alone would delete a month-wide partition on the strength of a
    single day, taking every other day in it unchecked.
    """
    connection = _Connection(
        existing_rows=_rows(
            _OWNER_1_TABLE,
            ('p_before_20260801', _bound(date(2026, 8, 1))),
            ('_p20260901000000', _bound(date(2026, 9, 1))),
        )
    )
    # 08-15 sits in the middle of [08-01, 09-01); neither bound would notice it.
    reconciler = _reconciler_for(connection, monkeypatch, blocking_rows=[('owner-1', date(2026, 8, 15))])

    assert await reconciler.reconcile_once() == 0
    assert connection.drop_statements == []


@pytest.mark.asyncio
async def test_reconcile_keeps_a_wide_partition_reaching_into_retention(monkeypatch) -> None:
    """Retention is read off the newest day inside, so no day still in it can be dropped.

    This partition starts before the cutoff and ends after it: bounding on the first day
    would delete the days the retention window still covers along with it.
    """
    cutoff_day = _NOW.date() - timedelta(days=_config().terminal_purge_retention_days)
    connection = _Connection(
        existing_rows=_rows(
            _OWNER_1_TABLE,
            ('p_before', _bound(cutoff_day - timedelta(days=2))),
            # Spans [cutoff - 2, cutoff + 2): the first two days are expired, the rest are not.
            ('p_wide', _bound(cutoff_day + timedelta(days=2))),
        )
    )
    reconciler = _reconciler_for(connection, monkeypatch)

    assert await reconciler.reconcile_once() == 0
    assert connection.drop_statements == []


@pytest.mark.asyncio
async def test_reconcile_never_drops_a_maxvalue_partition(monkeypatch) -> None:
    connection = _Connection(
        existing_rows=_rows(
            _OWNER_1_TABLE,
            ('p_before_20260901', _bound(date(2026, 9, 1))),
            ('p20260901', _bound(date(2026, 9, 2))),
            ('p_max', 'MAXVALUE'),
        )
    )
    reconciler = _reconciler_for(connection, monkeypatch)

    assert await reconciler.reconcile_once() == 1
    assert connection.drop_statements == [f'ALTER TABLE `{_OWNER_1_TABLE}` DROP PARTITION `p20260901`']


@pytest.mark.asyncio
async def test_reconcile_keeps_partitions_inside_the_retention_window(monkeypatch) -> None:
    """Retention is applied to the real upper bound, so a recent partition never lists."""
    connection = _Connection(existing_rows=_day_rows(_OWNER_1_TABLE, date(2026, 9, 25), days=2, interval_named=True))
    session_factory = _SessionFactory()

    async def connect(**kwargs):
        return connection

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', connect)
    reconciler = terminal_purge.BoundlessPartitionPurgeReconciler(
        session_factory=session_factory,
        config=_config(),
    )

    assert await reconciler.reconcile_once() == 0
    assert session_factory.candidate_params == [], 'an empty listing must not reach the control plane'


@pytest.mark.asyncio
async def test_reconcile_advances_past_partitions_dropped_by_an_earlier_pass(monkeypatch) -> None:
    """One day with batch + 1 terminal owners has to finish, not stall after the first pass.

    Candidate rows stay fixed because DROP PARTITION leaves the executions behind, so a
    bounded candidate query handed the window to the rows dropped first: 20, 0, 0.
    """
    batch = _config().terminal_purge_partition_batch
    connection, _ = _terminal_owners(batch + 1)
    reconciler = _reconciler_for(connection, monkeypatch)

    passes = [await reconciler.reconcile_once() for _ in range(3)]

    assert passes == [batch, 1, 0]
    assert len(set(connection.drop_statements)) == batch + 1


@pytest.mark.asyncio
async def test_reconcile_caps_dropped_partitions_at_the_configured_batch(monkeypatch) -> None:
    batch = _config().terminal_purge_partition_batch
    connection, _ = _terminal_owners(batch + 5)
    reconciler = _reconciler_for(connection, monkeypatch)

    assert await reconciler.reconcile_once() == batch
    assert len(connection.drop_statements) == batch


@pytest.mark.asyncio
async def test_reconcile_reclaims_a_partition_with_no_executions_at_all(monkeypatch) -> None:
    """A day nobody ran on has no execution row, so no terminal check can ever clear it.

    Preallocation creates one partition per day regardless, so asking the control plane for
    terminal days left these stranded forever; a day with nothing to protect is droppable.
    """
    connection = _Connection(existing_rows=_day_rows(_OWNER_1_TABLE, date(2026, 9, 1)))
    reconciler = _reconciler_for(connection, monkeypatch, blocking_rows=())

    assert await reconciler.reconcile_once() == 1
    assert connection.drop_statements == [f'ALTER TABLE `{_OWNER_1_TABLE}` DROP PARTITION `p20260901`']


@pytest.mark.asyncio
async def test_reconcile_keeps_a_partition_with_a_live_execution(monkeypatch) -> None:
    """The veto has to outrank the drop: this is the data a running execution still writes."""
    connection = _Connection(existing_rows=_day_rows(_OWNER_1_TABLE, date(2026, 9, 1)))
    reconciler = _reconciler_for(connection, monkeypatch, blocking_rows=[('owner-1', date(2026, 9, 1))])

    assert await reconciler.reconcile_once() == 0
    assert connection.drop_statements == []


@pytest.mark.asyncio
async def test_reconcile_unions_the_recheck_so_a_day_going_live_survives(monkeypatch) -> None:
    """An execution starting between the two reads must still veto the drop.

    The reads are combined with a union, not an intersection: taking the intersection would
    let a day that only the second read reports as live be dropped anyway.
    """
    connection = _Connection(existing_rows=_day_rows(_OWNER_1_TABLE, date(2026, 9, 1)))

    class _LateLiveFactory(_SessionFactory):
        def __call__(self):
            # Empty on the first read, live on the recheck.
            rows = [] if not self.sessions else [('owner-1', date(2026, 9, 1))]
            session = _Session(rows)
            self.sessions.append(session)
            return session

    async def connect(**kwargs):
        return connection

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', connect)
    reconciler = terminal_purge.BoundlessPartitionPurgeReconciler(
        session_factory=_LateLiveFactory(),
        config=_config(),
    )

    assert await reconciler.reconcile_once() == 0
    assert connection.drop_statements == []


@pytest.mark.asyncio
async def test_reconcile_drops_the_oldest_partitions_when_capped(monkeypatch) -> None:
    """A capped pass must spend its budget on the oldest backlog, not an arbitrary slice.

    The listing groups partitions per table, so its order says nothing about how the days of
    two tables interleave.  Here the table listed first holds the newer days, so taking the
    listing order would fill the whole batch with them and starve the older table.
    """
    batch = _config().terminal_purge_partition_batch
    # Far enough back that the cap is what limits the pass, not the retention cutoff.
    older_first_day = date(2026, 8, 1)
    connection = _Connection(
        existing_rows=[
            *_day_rows(_OWNER_1_TABLE, date(2026, 8, 10), days=batch),
            *_day_rows(_OWNER_2_TABLE, older_first_day, days=3),
        ]
    )
    reconciler = _reconciler_for(connection, monkeypatch)

    assert await reconciler.reconcile_once() == batch
    dropped = {
        (match.group('table'), match.group('partition'))
        for statement in connection.drop_statements
        if (match := _DROP_STATEMENT.match(statement)) is not None
    }
    # The three oldest days belong to the table listed last, so they must go first.
    assert {(_OWNER_2_TABLE, f'p{older_first_day + timedelta(days=offset):%Y%m%d}') for offset in range(3)} <= dropped


@pytest.mark.asyncio
async def test_reconcile_skips_executions_without_a_day_partition(monkeypatch) -> None:
    """Executions predating the Boundless rollout must not reach the control plane at all.

    Their traces were never written to these tables, so no day partition exists and probing
    them per candidate used to fill the window forever, starving the droppable partitions.
    """
    connection = _Connection(existing_rows=())

    async def connect(**kwargs):
        return connection

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', connect)
    session_factory = _SessionFactory()
    reconciler = terminal_purge.BoundlessPartitionPurgeReconciler(
        session_factory=session_factory,
        config=_config(),
    )

    assert await reconciler.reconcile_once() == 0
    assert session_factory.candidate_params == []
    assert not any('ALTER TABLE' in statement for statement in connection.cursor_instance.statements)
    assert connection.closed is True


@pytest.mark.asyncio
async def test_reconcile_bounds_candidates_by_the_oldest_existing_partition(monkeypatch) -> None:
    connection = _Connection(existing_rows=_day_rows(_OWNER_1_TABLE, date(2026, 9, 1), days=3))

    async def connect(**kwargs):
        return connection

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', connect)
    session_factory = _SessionFactory()
    reconciler = terminal_purge.BoundlessPartitionPurgeReconciler(
        session_factory=session_factory,
        config=_config(),
    )

    await reconciler.reconcile_once()

    assert session_factory.candidate_params, 'the control plane must be queried once partitions exist'
    assert all(params['floor_datetime'] == datetime(2026, 9, 1) for params in session_factory.candidate_params)
    listing = [s for s in connection.cursor_instance.statements if 'information_schema.PARTITIONS' in s]
    assert len(listing) == 1, 'partition existence must cost one listing, not one probe per candidate'


@pytest.mark.asyncio
async def test_reconcile_never_drops_the_catch_all_partition(monkeypatch) -> None:
    connection = _Connection(existing_rows=_day_rows(_OWNER_1_TABLE, date(2026, 9, 1)))

    async def connect(**kwargs):
        return connection

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', connect)
    reconciler = terminal_purge.BoundlessPartitionPurgeReconciler(
        session_factory=_SessionFactory(),
        config=_config(),
    )

    assert await reconciler.reconcile_once() == 1
    assert not any('p_before_' in statement for statement in connection.cursor_instance.statements)


@pytest.mark.asyncio
async def test_reconcile_skips_owners_whose_table_lacks_the_partition(monkeypatch) -> None:
    """The day bound is global, so a table created later still needs a per-table check."""
    connection = _Connection(existing_rows=_day_rows(_OWNER_1_TABLE, date(2026, 9, 1)))

    async def connect(**kwargs):
        return connection

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', connect)
    session_factory = _SessionFactory()
    reconciler = terminal_purge.BoundlessPartitionPurgeReconciler(
        session_factory=session_factory,
        config=_config(),
    )

    assert await reconciler.reconcile_once() == 1
    assert not any(_OWNER_2_TABLE in statement for statement in connection.cursor_instance.statements)


def _operational_error(code: int, message: str) -> Exception:
    from nexus.runner.trace.tdsql._driver import OperationalError

    return OperationalError(code, message)


class _FailingCursor(_Cursor):
    """Raise on the statement whose name appears in ``fail_on``."""

    def __init__(self, existing_rows, *, fail_on: str, error: Exception):
        super().__init__(existing_rows)
        self.fail_on = fail_on
        self.error = error

    async def execute(self, statement, params=None):
        await super().execute(statement, params)
        if self.fail_on in str(statement):
            raise self.error

    async def fetchall(self):
        if self.fail_on == 'information_schema':
            raise self.error
        return await super().fetchall()


class _FailingConnection(_Connection):
    def __init__(self, *, fail_on: str, error: Exception, existing_rows=None):
        super().__init__(existing_rows)
        self.error = error
        self.fail_on = fail_on
        self.close_error: Exception | None = None
        if fail_on != 'commit':
            self.cursor_instance = _FailingCursor(self.cursor_instance.existing_rows, fail_on=fail_on, error=error)

    async def commit(self):
        if self.fail_on == 'commit':
            raise self.error
        await super().commit()

    async def ensure_closed(self):
        await super().ensure_closed()
        if self.close_error is not None:
            raise self.close_error


@pytest.mark.parametrize('fail_on', ['information_schema', 'DROP PARTITION', 'commit'])
@pytest.mark.asyncio
async def test_reconcile_wraps_mid_query_disconnects_as_transient(fail_on, monkeypatch) -> None:
    # The raw driver never produces a SQLAlchemy DBAPIError, so an unwrapped 2013 would be
    # classified fatal and cancel the sibling loops, lease heartbeat included.
    error = _operational_error(2013, 'Lost connection to MySQL server during query')
    connection = _FailingConnection(fail_on=fail_on, error=error)
    reconciler = _reconciler_for(connection, monkeypatch)

    with pytest.raises(terminal_purge.TerminalPurgeTransientError) as raised:
        await reconciler.reconcile_once()

    assert raised.value.__cause__ is error
    assert connection.closed is True


@pytest.mark.asyncio
async def test_transient_purge_error_keeps_the_reconciler_loop_alive() -> None:
    from nexus.distributed.control_plane.reconciler import _is_transient_loop_error

    assert _is_transient_loop_error(terminal_purge.TerminalPurgeTransientError('boom')) is True


@pytest.mark.parametrize(
    'error',
    [
        _operational_error(1045, "Access denied for user 'maintenance'"),
        _operational_error(1064, 'You have an error in your SQL syntax'),
    ],
)
@pytest.mark.asyncio
async def test_reconcile_still_raises_errors_that_need_a_human(error, monkeypatch) -> None:
    connection = _FailingConnection(fail_on='information_schema', error=error)
    reconciler = _reconciler_for(connection, monkeypatch)

    with pytest.raises(type(error)):
        await reconciler.reconcile_once()


@pytest.mark.asyncio
async def test_reconcile_keeps_going_when_one_table_is_unusable(monkeypatch, caplog) -> None:
    """A rejected statement for one owner is skipped; only the connection ends the pass."""
    caplog.set_level(logging.WARNING, logger='nexus.distributed.control_plane.terminal_purge')
    error = _operational_error(1142, 'ALTER command denied')
    connection = _FailingConnection(
        fail_on=_OWNER_2_TABLE,
        error=error,
        existing_rows=_day_rows(_OWNER_1_TABLE, date(2026, 9, 1)) + _day_rows(_OWNER_2_TABLE, date(2026, 9, 1)),
    )
    reconciler = _reconciler_for(
        connection,
        monkeypatch,
    )

    assert await reconciler.reconcile_once() == 1
    assert connection.committed is True
    assert 'Failed to drop terminal Boundless trace partition' in caplog.text


@pytest.mark.asyncio
async def test_close_failure_does_not_mask_the_original_error(monkeypatch) -> None:
    """A dropped connection usually fails to close too; the real cause must survive."""
    error = _operational_error(2013, 'Lost connection to MySQL server during query')
    connection = _FailingConnection(fail_on='commit', error=error)
    connection.close_error = RuntimeError('connection already gone')
    reconciler = _reconciler_for(connection, monkeypatch)

    with pytest.raises(terminal_purge.TerminalPurgeTransientError) as raised:
        await reconciler.reconcile_once()

    assert raised.value.__cause__ is error


@pytest.mark.asyncio
async def test_connection_failures_still_skip_the_pass(monkeypatch, caplog) -> None:
    """Connect already had its own guard; widening the scope must not change it."""
    caplog.set_level(logging.WARNING, logger='nexus.distributed.control_plane.terminal_purge')

    async def connect(**kwargs):
        raise _operational_error(2003, "Can't connect to MySQL server")

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', connect)
    reconciler = terminal_purge.BoundlessPartitionPurgeReconciler(
        session_factory=_SessionFactory(),
        config=_config(),
    )

    assert await reconciler.reconcile_once() == 0
    assert 'terminal data purge database connection failed' in caplog.text


@pytest.mark.parametrize(
    ('error', 'transient'),
    [
        (_operational_error(2013, 'Lost connection'), True),
        (_operational_error(1213, 'Deadlock found'), True),
        (_operational_error(1045, 'Access denied'), False),
        (TimeoutError('ddl timed out'), True),
        (OSError('broken pipe'), True),
        (ValueError('bad config'), False),
    ],
)
def test_transient_classification_covers_recoverable_driver_errors(error, transient) -> None:
    assert terminal_purge._is_transient_database_error(error) is transient


def test_transient_classification_ignores_boolean_error_args() -> None:
    # bool is a subclass of int, so a driver arg of True must not be read as an error code.
    assert terminal_purge._mysql_error_code(_operational_error(True, 'not a code')) is None
