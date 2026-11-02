from __future__ import annotations

import asyncio
import contextlib
import logging

from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, cast

from nexus.distributed.observability.metrics import ControlLoopWork
from nexus.distributed.utils.clock import utcnow
from nexus.runner.trace.tdsql.config import build_db_config
from nexus.runner.trace.tdsql.schema import (
    PRINCIPAL_TRACE_TABLE_PATTERN,
    PRINCIPAL_TRACE_TABLE_PREFIX,
    principal_trace_table_name,
)


logger = logging.getLogger(__name__)

DEFAULT_PARTITION_HORIZON = timedelta(days=7)
DEFAULT_PARTITION_RACE_RETRIES = 3
# 1517 is a name collision with a partition this loop already added; 1493 is a non-increasing
# bound, which INTERVAL auto-partitioning produces under its own naming scheme.
_PARTITION_RACE_ERROR_CODES = frozenset({1493, 1517})
_PARTITION_RACE_MARKERS = ('duplicate partition', 'must be strictly increasing')
_RETRYABLE_MYSQL_ERROR_CODES = frozenset({1205, 1213, 2003, 2006, 2013, 2055, 8506})
# Subpartitioned tables emit one row per subpartition and no summary row, so the boundary
# has to be read from the first subpartition of each partition.
_PARTITION_QUERY = (
    'SELECT TABLE_NAME, PARTITION_NAME, PARTITION_DESCRIPTION '
    'FROM information_schema.PARTITIONS '
    'WHERE TABLE_SCHEMA = %s AND PARTITION_NAME IS NOT NULL '
    'AND (SUBPARTITION_NAME IS NULL OR SUBPARTITION_ORDINAL_POSITION = 1)'
)


class TDSQLPartitionMaintenanceError(RuntimeError):
    """A database failure that should make the maintenance loop retry."""


@dataclass(frozen=True, slots=True)
class TDSQLPartitionTarget:
    """Credentials for one Boundless trace database."""

    backend_name: str
    host: str
    user: str
    database: str
    password: str = ''
    port: int = 3306


class TDSQLPartitionMaintenance:
    """Keep Boundless principal tables ahead of the write frontier and absorb DDL races."""

    def __init__(
        self,
        targets: tuple[TDSQLPartitionTarget, ...] = (),
        *,
        horizon: timedelta = DEFAULT_PARTITION_HORIZON,
        race_retries: int = DEFAULT_PARTITION_RACE_RETRIES,
        connector: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        if horizon <= timedelta(0):
            raise ValueError('TDSQL partition horizon must be positive')
        if isinstance(race_retries, bool) or not isinstance(race_retries, int) or race_retries < 0:
            raise ValueError('TDSQL partition race retries must be a non-negative integer')
        self.targets = targets
        self.horizon = horizon
        self.race_retries = race_retries
        self._connector = connector
        self.last_stats = ControlLoopWork()

    @classmethod
    def from_config(cls, config: Any, *, horizon_days: int = 7) -> TDSQLPartitionMaintenance:
        targets: list[TDSQLPartitionTarget] = []
        for backend_name, backend in sorted(config.run_trace_sinks.items()):
            if backend.sink_class != 'nexus.runner.trace.tdsql::TDSQLBoundlessTraceSink':
                continue
            values = dict(backend.config)
            values.update(backend.sink_config)
            for field_name, credential_ref in backend.secret_config.items():
                values[field_name] = config.read_run_trace_credential(credential_ref)
            host = values.get('host')
            user = values.get('user')
            database = values.get('db')
            password = values.get('password', '')
            port = values.get('port', 3306)
            if not all(isinstance(value, str) and value.strip() for value in (host, user, database)):
                raise ValueError(f'Boundless trace backend connection is invalid: {backend_name}')
            if (
                not isinstance(password, str)
                or isinstance(port, bool)
                or not isinstance(port, int)
                or not 1 <= port <= 65535
            ):
                raise ValueError(f'Boundless trace backend connection is invalid: {backend_name}')
            targets.append(
                TDSQLPartitionTarget(
                    backend_name=backend_name,
                    host=cast(str, host),
                    user=cast(str, user),
                    database=cast(str, database),
                    password=password,
                    port=port,
                )
            )
        return cls(tuple(targets), horizon=timedelta(days=horizon_days))

    async def reconcile_once(self, *, now: datetime | None = None) -> int:
        if not self.targets:
            self.last_stats = ControlLoopWork()
            return 0
        current = now or utcnow()
        scanned = 0
        changed = 0
        try:
            for target in self.targets:
                target_scanned, target_changed = await self._reconcile_target(target, current)
                scanned += target_scanned
                changed += target_changed
        except Exception as exc:
            # An unresolved partition race already arrives as the loop's retry type, and
            # _is_transient_database_error rejects it, so it re-raises here untouched.
            if not _is_transient_database_error(exc):
                raise
            raise TDSQLPartitionMaintenanceError('TDSQL partition maintenance failed') from exc
        self.last_stats = ControlLoopWork(scanned=scanned, changed=changed)
        return changed

    async def _reconcile_target(self, target: TDSQLPartitionTarget, now: datetime) -> tuple[int, int]:
        connector = self._connector or _connect
        connection = await connector(
            **build_db_config(
                host=target.host,
                port=target.port,
                user=target.user,
                password=target.password,
                db=target.database,
            )
        )
        try:
            async with connection.cursor() as cursor:
                await cursor.execute(_PARTITION_QUERY, (target.database,))
                rows = await cursor.fetchall()
                partitions = _partition_boundaries(rows)
                changed = 0
                for table_name, boundaries in partitions.items():
                    changed += await self._extend_table(
                        cursor,
                        target=target,
                        table_name=table_name,
                        boundaries=boundaries,
                        now=now,
                    )
            await connection.commit()
            return len(partitions), changed
        finally:
            with contextlib.suppress(Exception):
                connection.close()

    async def _extend_table(
        self,
        cursor: Any,
        *,
        target: TDSQLPartitionTarget,
        table_name: str,
        boundaries: list[datetime],
        now: datetime,
    ) -> int:
        # A concurrent INSERT can move the INTERVAL frontier between the read and the ALTER,
        # so a lost race re-reads the boundaries and replans instead of ending the loop.
        attempts = self.race_retries + 1
        lost_race: Exception | None = None
        for attempt in range(1, attempts + 1):
            additions = _missing_partitions(boundaries, now=now, horizon=self.horizon)
            if not additions:
                return 0
            try:
                await cursor.execute(_add_partition_sql(table_name, additions))
                return len(additions)
            except Exception as exc:
                if not _is_partition_race(exc):
                    raise
                lost_race = exc
                logger.info(
                    'TDSQL partition changed concurrently: backend=%s table=%s attempt=%s/%s reason=%s',
                    target.backend_name,
                    table_name,
                    attempt,
                    attempts,
                    exc,
                )
                boundaries = await self._read_boundaries(cursor, database=target.database, table_name=table_name)
        raise TDSQLPartitionMaintenanceError(
            f'TDSQL partition race unresolved: backend={target.backend_name} table={table_name}'
        ) from lost_race

    async def _read_boundaries(self, cursor: Any, *, database: str, table_name: str) -> list[datetime]:
        await cursor.execute(f'{_PARTITION_QUERY} AND TABLE_NAME = %s', (database, table_name))
        rows = await cursor.fetchall()
        return _partition_boundaries(rows).get(table_name, [])


async def _connect(**kwargs: Any) -> Any:
    from nexus.runner.trace.tdsql._driver import connect

    return await connect(**kwargs)


def _is_transient_database_error(error: BaseException) -> bool:
    if isinstance(error, (asyncio.TimeoutError, TimeoutError, OSError)):
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


def _mysql_error_code(error: BaseException) -> int | None:
    code = getattr(error, 'errno', None)
    if isinstance(code, int) and not isinstance(code, bool):
        return code
    args = getattr(error, 'args', ())
    return next((value for value in args if isinstance(value, int) and not isinstance(value, bool)), None)


def _partition_boundaries(rows: Any) -> dict[str, list[datetime]]:
    result: dict[str, set[datetime]] = defaultdict(set)
    unbounded: set[str] = set()
    seen: set[tuple[str, Any]] = set()
    for row in rows:
        table_name, partition_name, description = row[:3]
        if (table_name, partition_name) in seen:
            continue
        seen.add((table_name, partition_name))
        if not isinstance(table_name, str) or PRINCIPAL_TRACE_TABLE_PATTERN.fullmatch(table_name) is None:
            continue
        if isinstance(description, bytes):
            description = description.decode('utf-8', errors='ignore')
        if isinstance(description, str) and description.strip().strip("'").strip('"').upper() == 'MAXVALUE':
            unbounded.add(table_name)
            continue
        boundary = _parse_partition_boundary(description)
        if boundary is not None:
            result[table_name].add(boundary)
    for table_name in unbounded:
        result.pop(table_name, None)
    return {table_name: sorted(boundaries) for table_name, boundaries in result.items()}


def _parse_partition_boundary(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='ignore')
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value.startswith('(') and value.endswith(')'):
        value = value[1:-1].strip()
    value = value.strip("'").strip('"')
    if value.upper() == 'MAXVALUE':
        return None
    try:
        return datetime.fromisoformat(value.replace('T', ' ')).replace(tzinfo=None)
    except ValueError:
        return None


def _missing_partitions(
    boundaries: list[datetime],
    *,
    now: datetime,
    horizon: timedelta,
) -> list[tuple[str, datetime]]:
    if not boundaries:
        return []
    next_lower_bound = max(boundaries)
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)
    additions: list[tuple[str, datetime]] = []
    target = now + horizon
    while next_lower_bound < target:
        upper_bound = next_lower_bound + timedelta(days=1)
        additions.append((f'p{next_lower_bound:%Y%m%d}', upper_bound))
        next_lower_bound = upper_bound
    return additions


def _add_partition_sql(table_name: str, additions: list[tuple[str, datetime]]) -> str:
    # Table and partition names are generated/validated identifiers, never user values.
    principal_trace_table_name(table_name[len(PRINCIPAL_TRACE_TABLE_PREFIX) :])
    definitions = ', '.join(
        f"PARTITION `{name}` VALUES LESS THAN ('{boundary:%Y-%m-%d %H:%M:%S}')" for name, boundary in additions
    )
    return f'ALTER TABLE `{table_name}` ADD PARTITION ({definitions})'


def _is_partition_race(error: BaseException) -> bool:
    if _mysql_error_code(error) in _PARTITION_RACE_ERROR_CODES:
        return True
    # Matched on wording, not on the code: a driver that loses the errno keeps the message,
    # while a bare '1493' can just as easily be a partition name inside a failing statement.
    text = f'{error} {getattr(error, "args", "")} {getattr(error, "code", "")}'.lower()
    return any(marker in text for marker in _PARTITION_RACE_MARKERS)


__all__ = [
    'DEFAULT_PARTITION_HORIZON',
    'DEFAULT_PARTITION_RACE_RETRIES',
    'TDSQLPartitionMaintenanceError',
    'TDSQLPartitionMaintenance',
    'TDSQLPartitionTarget',
]
