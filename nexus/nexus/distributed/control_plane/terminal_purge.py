from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time

from collections.abc import Iterable, Iterator
from datetime import date, datetime, timedelta, timezone
from datetime import time as datetime_time
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from nexus.distributed.config import DistributedConfig, MySQLTraceDatabaseCredential
from nexus.distributed.execution.models import TERMINAL_EXECUTION_STATUSES
from nexus.distributed.utils.clock import utcnow
from nexus.runner.trace import principal_trace_scope_id
from nexus.runner.trace.tdsql.config import build_db_config
from nexus.runner.trace.tdsql.schema import principal_trace_table_name


logger = logging.getLogger(__name__)

_SCOPE_TABLE = re.compile(r'^trajectory_span_events_pr_[0-9a-f]{32}$')

# Partition names are read from the server and only quoted into DDL, never parsed as a date.
_PARTITION_IDENT = re.compile(r'^[A-Za-z_][A-Za-z0-9_]{0,63}$')

# Steps back off an exclusive upper bound to land on the last instant the partition holds.
_BOUND_EPSILON = timedelta(microseconds=1)

# 2006/2013 are mid-query disconnects, 2003/2055 a failed connection, 1205/1213 lock waits
# and deadlocks.  All clear on their own, unlike a rejected credential or statement.
_RETRYABLE_MYSQL_ERROR_CODES = frozenset({1205, 1213, 2003, 2006, 2013, 2055})

_TERMINAL_STATUSES = tuple(sorted(TERMINAL_EXECUTION_STATUSES))
_TERMINAL_STATUS_PARAMS = ', '.join(f':terminal_status_{index}' for index in range(len(_TERMINAL_STATUSES)))

# The days that must survive: asking for terminal days says nothing about a day with no rows,
# stranding its preallocated partition.  Status in WHERE, not HAVING, keeps the result small.
_BLOCKING_SQL = f"""
SELECT
    e.owner_principal_id AS owner_principal_id,
    DATE(e.created_at) AS partition_day
FROM executions e
WHERE e.parent_execution_id IS NULL
  AND e.owner_principal_id IS NOT NULL
  AND e.created_at >= :floor_datetime
  AND e.created_at < :cutoff_datetime
  AND e.status NOT IN ({_TERMINAL_STATUS_PARAMS})
GROUP BY owner_principal_id, partition_day
"""

# PARTITIONS has a row per subpartition, hence the GROUP BY.  Unfiltered by name because
# INTERVAL names them after their upper bound: the day and order come from the boundaries.
_EXISTING_PARTITIONS_SQL = """
SELECT TABLE_NAME, PARTITION_ORDINAL_POSITION, PARTITION_NAME, PARTITION_DESCRIPTION
FROM information_schema.PARTITIONS
WHERE TABLE_SCHEMA = %s
  AND TABLE_NAME LIKE 'trajectory\\_span\\_events\\_pr\\_%%'
  AND PARTITION_NAME IS NOT NULL
GROUP BY TABLE_NAME, PARTITION_ORDINAL_POSITION, PARTITION_NAME, PARTITION_DESCRIPTION
ORDER BY TABLE_NAME, PARTITION_ORDINAL_POSITION
"""


class TerminalPurgeTransientError(RuntimeError):
    """A recoverable trace-database failure that should retry instead of ending the loop."""


def _mysql_error_code(error: BaseException) -> int | None:
    code = getattr(error, 'errno', None)
    if isinstance(code, int) and not isinstance(code, bool):
        return code
    args = getattr(error, 'args', ())
    return next((value for value in args if isinstance(value, int) and not isinstance(value, bool)), None)


def _is_transient_database_error(error: BaseException) -> bool:
    # The trace database is reached through the raw driver, so its failures never become the
    # SQLAlchemy DBAPIError the reconciler loop retries on; they have to be classified here.
    if isinstance(error, (TimeoutError, OSError)):
        return True
    try:
        from nexus.runner.trace.tdsql._driver import InterfaceError, OperationalError
    except ImportError:  # pragma: no cover - the production path has a driver installed
        return False
    if isinstance(error, InterfaceError):
        return True
    if not isinstance(error, OperationalError):
        return False
    return _mysql_error_code(error) in _RETRYABLE_MYSQL_ERROR_CODES


def _partition_bound(value: Any) -> datetime | None:
    # PARTITION_DESCRIPTION is the VALUES LESS THAN literal, quoted for RANGE COLUMNS.
    # MAXVALUE and anything unparseable yield None so the partition is left alone.
    if value is None:
        return None
    literal = str(value).strip().strip('\'"').strip()
    if not literal or literal.upper() == 'MAXVALUE':
        return None
    try:
        parsed = datetime.fromisoformat(literal)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed
    # Bounds are naive literals, so an offset has to be converted rather than dropped.
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _partition_day(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _days_of(table_name: str, first_day: date, last_day: date) -> Iterator[tuple[str, date]]:
    """Yield one veto key per day the partition holds, both ends included."""
    for offset in range((last_day - first_day).days + 1):
        yield table_name, first_day + timedelta(days=offset)


class BoundlessPartitionPurgeReconciler:
    """Drop old Boundless day partitions after a control-plane terminal-state check."""

    def __init__(self, *, session_factory: async_sessionmaker, config: DistributedConfig) -> None:
        self.session_factory = session_factory
        self.config = config
        self._lock = asyncio.Lock()

    async def reconcile_once(self) -> int:
        if not self.config.terminal_data_purge_enabled:
            return 0
        async with self._lock:
            return await self._reconcile_locked()

    async def _reconcile_locked(self) -> int:
        backend_name = self.config.terminal_purge_trace_backend
        if not backend_name:
            raise ValueError('terminal_purge_trace_backend is required when terminal data purge is enabled')
        self._validate_backend(backend_name)

        cutoff_day = utcnow().date() - timedelta(days=self.config.terminal_purge_retention_days)
        cutoff_datetime = datetime.combine(cutoff_day, datetime_time.min)

        try:
            db_config = self._trace_db_config(backend_name)
            from nexus.runner.trace.tdsql._driver import connect as mysql_connect

            conn = await mysql_connect(**db_config)
        except Exception:
            logger.warning('terminal data purge database connection failed', exc_info=True)
            return 0

        # Listing partitions and the final commit run on the raw driver too, so a disconnect
        # there has to be classified here rather than escaping as a fatal loop error.
        try:
            return await self._purge_with_connection(
                conn,
                backend_name=backend_name,
                cutoff_day=cutoff_day,
                cutoff_datetime=cutoff_datetime,
            )
        except Exception as exc:
            if not _is_transient_database_error(exc):
                raise
            raise TerminalPurgeTransientError('terminal data purge trace database failed') from exc
        finally:
            await self._close_connection(conn)

    @staticmethod
    async def _close_connection(conn: Any) -> None:
        # A connection that just dropped usually fails to close as well, and an exception
        # raised here would replace the failure that actually ended the purge.
        with contextlib.suppress(Exception):
            close = getattr(conn, 'ensure_closed', None)
            if close is not None:
                await close()
            else:
                conn.close()

    async def _purge_with_connection(
        self,
        conn: Any,
        *,
        backend_name: str,
        cutoff_day: date,
        cutoff_datetime: datetime,
    ) -> int:
        # Drive from the trace database: list the partitions that exist and are old enough,
        # then let the control plane veto the ones it still needs.
        async with conn.cursor() as cursor:
            existing = await self._existing_partitions(cursor, backend_name=backend_name, cutoff_day=cutoff_day)
        if not existing:
            return 0

        floor_day = min(day for _, day in existing)
        params = {
            'floor_datetime': datetime.combine(floor_day, datetime_time.min),
            'cutoff_datetime': cutoff_datetime,
        }
        params.update({f'terminal_status_{index}': value for index, value in enumerate(_TERMINAL_STATUSES)})

        blocked = await self._blocked_days(params, sql_name='blocking_query')
        # Re-read immediately before DDL and take the union: a day that either read calls
        # live stays, which closes the status-change race without a query per partition.
        blocked |= await self._blocked_days(params, sql_name='blocking_recheck')

        # Oldest first, so a capped pass always makes progress on the oldest backlog.
        batch = self.config.terminal_purge_partition_batch
        dropped = 0
        async with conn.cursor() as cursor:
            for (table_name, first_day), (partition_name, last_day) in sorted(
                existing.items(), key=lambda item: (item[0][1], item[0][0])
            ):
                if dropped >= batch:
                    break
                # Every day inside has to clear, so a partition that absorbed its neighbours
                # is vetoed by any one of them rather than being kept for its width alone.
                if any(day in blocked for day in _days_of(table_name, first_day, last_day)):
                    continue
                try:
                    statement = f'ALTER TABLE `{table_name}` DROP PARTITION `{partition_name}`'
                    await self._execute_trace_query(
                        cursor,
                        statement,
                        sql_name='drop_partition',
                        timeout=self.config.terminal_purge_ddl_timeout,
                    )
                    dropped += 1
                    logger.info(
                        'Dropped terminal Boundless trace partition: table=%s partition=%s days=%s..%s',
                        table_name,
                        partition_name,
                        first_day,
                        last_day,
                    )
                except Exception as exc:
                    # One unusable table must not stop the pass, but a dropped connection
                    # makes every remaining statement fail for the same reason.
                    if _is_transient_database_error(exc):
                        raise
                    logger.warning(
                        'Failed to drop terminal Boundless trace partition: table=%s days=%s..%s',
                        table_name,
                        first_day,
                        last_day,
                        exc_info=True,
                    )
        await conn.commit()
        return dropped

    async def _existing_partitions(
        self, cursor: Any, *, backend_name: str, cutoff_day: date
    ) -> dict[tuple[str, date], tuple[str, date]]:
        rows = await self._execute_trace_query(
            cursor,
            _EXISTING_PARTITIONS_SQL,
            (self._trace_database_name(backend_name),),
            sql_name='existing_partitions',
            fetch_all=True,
        )
        by_table: dict[str, list[tuple[int, str, datetime | None]]] = {}
        for row in rows or ():
            table_name = str(row[0])
            partition_name = str(row[2])
            if _SCOPE_TABLE.fullmatch(table_name) is None or _PARTITION_IDENT.fullmatch(partition_name) is None:
                continue
            try:
                ordinal = int(row[1])
            except (TypeError, ValueError):
                continue
            by_table.setdefault(table_name, []).append((ordinal, partition_name, _partition_bound(row[3])))

        partitions: dict[tuple[str, date], tuple[str, date]] = {}
        for table_name, entries in by_table.items():
            entries.sort(key=lambda entry: entry[0])
            previous: datetime | None = None
            for _, partition_name, upper_bound in entries:
                lower_bound, previous = previous, upper_bound
                # A None lower bound is the leading catch-all, which owns every day before
                # the table existed; a None upper bound is MAXVALUE or an unreadable literal.
                if lower_bound is None or upper_bound is None:
                    continue
                # The bound is exclusive, so the last day it holds is a moment earlier.  Any
                # width is kept: dropping one clears the gap that widened its successor.
                last_day = (upper_bound - _BOUND_EPSILON).date()
                # Retention reads off the newest day inside, or a wide partition would take
                # days still within it.  Keyed by day: the name is a day ahead of its rows.
                if last_day < cutoff_day:
                    partitions[(table_name, lower_bound.date())] = (partition_name, last_day)
        return partitions

    async def _execute_control_query(
        self,
        session: Any,
        statement: Any,
        params: dict[str, Any] | None,
        *,
        sql_name: str,
    ) -> list[Any]:
        started = time.perf_counter()
        row_count = 0
        try:
            result = await session.execute(statement, params)
            rows = list(result.all())
            row_count = len(rows)
            return rows
        finally:
            logger.info(
                'terminal purge SQL completed: sql=%s elapsed_ms=%.3f rows=%d',
                sql_name,
                (time.perf_counter() - started) * 1000,
                row_count,
            )

    async def _execute_trace_query(
        self,
        cursor: Any,
        statement: str,
        params: tuple[Any, ...] | None = None,
        *,
        sql_name: str,
        fetch_one: bool = False,
        fetch_all: bool = False,
        timeout: float | None = None,
    ) -> Any:
        started = time.perf_counter()
        result = None
        try:
            execution = cursor.execute(statement, params) if params is not None else cursor.execute(statement)
            if timeout is not None:
                await asyncio.wait_for(execution, timeout=timeout)
            else:
                await execution
            if fetch_all:
                result = await cursor.fetchall()
            elif fetch_one:
                result = await cursor.fetchone()
            return result
        finally:
            logger.info(
                'terminal purge SQL completed: sql=%s elapsed_ms=%.3f statement=%s',
                sql_name,
                (time.perf_counter() - started) * 1000,
                ' '.join(statement.split()),
            )

    async def _blocked_days(self, params: dict[str, Any], *, sql_name: str) -> set[tuple[str, date]]:
        """Read the (table, day) pairs a live root execution still needs."""
        async with self.session_factory() as session:
            rows = await self._execute_control_query(session, text(_BLOCKING_SQL), params, sql_name=sql_name)
        return self._blocked_from_rows(rows)

    @staticmethod
    def _blocked_from_rows(rows: Iterable[Any]) -> set[tuple[str, date]]:
        # Keyed by table name to match `existing`; the owner hash is one-way, so the
        # mapping only runs in this direction.
        blocked: set[tuple[str, date]] = set()
        for row in rows:
            if not row[0] or (day := _partition_day(row[1])) is None:
                continue
            table_name = principal_trace_table_name(principal_trace_scope_id(str(row[0])))
            if _SCOPE_TABLE.fullmatch(table_name) is None:
                continue
            blocked.add((table_name, day))
        return blocked

    def _validate_backend(self, backend_name: str) -> None:
        backend = self.config.run_trace_sinks.get(backend_name)
        if backend is None:
            raise ValueError(f'unknown terminal purge trace backend: {backend_name!r}')
        if 'TDSQLBoundlessTraceSink' not in (backend.sink_class or ''):
            raise ValueError(f'terminal purge backend {backend_name!r} is not a TDSQL Boundless sink')

    def _trace_database_name(self, backend_name: str) -> str:
        access = self.config.trace_database_access.get(backend_name)
        if access is None:
            raise ValueError(f'trace database access is not configured for {backend_name!r}')
        return access.database

    def _trace_db_config(self, backend_name: str) -> dict[str, Any]:
        access = self.config.trace_database_access.get(backend_name)
        backend = self.config.run_trace_sinks.get(backend_name)
        if access is None or backend is None:
            raise ValueError(f'trace database access is not configured for {backend_name!r}')

        credential = self.config.resolve_trace_database_credential(access.maintenance_credential_ref)
        if isinstance(credential, MySQLTraceDatabaseCredential):
            username, password = credential.read()
            return build_db_config(
                host=access.read_host,
                port=access.read_port,
                user=username,
                password=password,
                db=access.database,
            )

        # Cloud account credentials are API keys, not MySQL passwords, so fall back to the
        # Boundless writer account, which the deployment must grant ALTER/DROP PARTITION.
        sink_values = backend.config
        host = sink_values.get('host')
        user = sink_values.get('user')
        port = sink_values.get('port', 3306)
        password_ref = backend.secret_config.get('password')
        if not isinstance(host, str) or not isinstance(user, str) or not password_ref:
            raise ValueError(f'no SQL credential is available for terminal purge backend {backend_name!r}')
        return build_db_config(
            host=host,
            port=port,
            user=user,
            password=self.config.read_run_trace_credential(password_ref),
            db=access.database,
        )


__all__ = ['BoundlessPartitionPurgeReconciler', 'TerminalPurgeTransientError']
