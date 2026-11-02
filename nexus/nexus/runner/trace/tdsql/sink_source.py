from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import math
import os
import re
import secrets
import time

from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from nexus.runner.trace.tdsql.config import (
    build_db_config,
    resolve_max_queue_bytes,
    resolve_max_queue_items,
    resolve_writer_shards,
)
from nexus.runner.trace.tdsql.schema import (
    BOUNDLESS_DIALECT,
    DEFAULT_BOUNDLESS_PARTITION_INTERVAL,
    DEFAULT_BOUNDLESS_SUBPARTITIONS,
    SHARED_TRACE_TABLE,
    TDSQL_DIALECT,
    principal_trace_interval_statement,
    principal_trace_table_name,
    principal_trace_table_statement,
    principal_trace_table_upgrade_statement,
    validate_dialect,
    validate_partition_interval,
    validate_subpartitions,
)
from nexus.runner.trace.trace import (
    RunTraceContext,
    RunTraceScope,
    RunTraceScopePreparer,
    RunTraceSink,
    RunTraceSource,
    RunTraceSourceUnavailable,
)
from nexus.trace.records import SpanEventRecord
from nexus.utils.platform_metrics import measure_database


if TYPE_CHECKING:
    from nexus.runner.trace.tdsql.writer import TDSQLTraceWriter


try:
    import orjson

    def _json_loads(data: str | bytes) -> Any:
        try:
            return orjson.loads(data)
        except orjson.JSONDecodeError:
            return json.loads(data)

except ImportError:  # orjson is an optional speedup; stdlib json is the fallback.

    def _json_loads(data: str | bytes) -> Any:
        return json.loads(data)


SLOW_OPERATION_WARN_SECONDS = 5.0
_DEFAULT_CLOSE_FLUSH_TIMEOUT_SECONDS = 1200.0
_CLOSE_FLUSH_TIMEOUT_ENV = 'NEXUS_TRAJ_TDSQL_CLOSE_FLUSH_TIMEOUT_SECONDS'
_READ_RESULT_MAX_UPDATES = 200
_SELECT_COLUMNS = 'record_id, event_type, time_unix_nano, trace_id, span_id, parent_span_id, event_data'
_RAW_SELECT_COLUMNS = (
    'id, run_id, run_span_id, parent_execution_id, record_id, event_type, '
    'time_unix_nano, trace_id, span_id, parent_span_id, event_data, created_at'
)
_RAW_COLUMN_NAMES = (
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


def _default_close_flush_timeout_seconds() -> float:
    raw = os.getenv(_CLOSE_FLUSH_TIMEOUT_ENV)
    if raw is None:
        return _DEFAULT_CLOSE_FLUSH_TIMEOUT_SECONDS
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f'{_CLOSE_FLUSH_TIMEOUT_ENV} must be a positive finite number of seconds') from None


class RawTraceRecordInvalidError(ValueError):
    """A malformed source record; its payload is intentionally omitted."""


class RawTraceEventTooLargeError(ValueError):
    def __init__(self, record_id: int, size_bytes: int) -> None:
        self.record_id = record_id
        self.size_bytes = size_bytes
        super().__init__('raw trace event exceeds the configured size limit')


class RawTraceCleanupError(RuntimeError):
    """Termination of this reader's server query could not be confirmed."""


@dataclass(frozen=True)
class RawTraceReadPlan:
    """A bounded immutable prefix whose bodies have not been downloaded yet."""

    record_sizes: tuple[tuple[int, int], ...]
    has_more: bool

    @property
    def size_bytes(self) -> int:
        return sum(size for _, size in self.record_sizes)

    @property
    def through_record_id(self) -> int | None:
        return self.record_sizes[-1][0] if self.record_sizes else None


class _RawCursor:
    def __init__(self, cursor: Any) -> None:
        self.cursor = cursor
        self.query_marker = 'nexus_trace_read_' + secrets.token_hex(16)
        self.exhausted = False
        self.pending: asyncio.Task[Any] | None = None

    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        self.exhausted = False
        # Shield query initialization: asyncmy only cleans its temporary result
        # on Exception, not CancelledError. Socket shutdown below wakes this task
        # with a driver error, and cleanup joins it before releasing the reader.
        # Keep the marker inside SELECT: TDSQL preserves it in the backend
        # process list even when proxy and backend connection IDs differ.
        sql = sql.replace('SELECT ', f'SELECT /*{self.query_marker}*/ ', 1)
        self.pending = asyncio.create_task(self.cursor.execute(sql, params))
        try:
            await asyncio.shield(self.pending)
        finally:
            if self.pending.done():
                self.pending = None

    async def fetchone(self) -> Any:
        row = await self.cursor.fetchone()
        if row is None:
            self.exhausted = True
        return row


@dataclass(frozen=True, slots=True)
class RawTraceEventPage:
    items: list[dict[str, Any]]
    has_more: bool


_SHARED_SCOPE = 'shared'
_PRINCIPAL_SCOPE = 'principal_v1'
_LEGACY_BATCH_SCOPE = 'batch_principal_v1'
_BASE_QUERYABLE_COLUMNS = (
    'run_id',
    'run_span_id',
    'event_type',
    'trace_id',
    'span_id',
    'parent_span_id',
)
_TRANSIENT_STREAM_EVENT_NAMES = frozenset({
    'agent.stream',
    'chat.completion.delta',
    'chat.completion.reasoning.delta',
    'chat.completion.tool_call.delta',
})
_TERMINAL_AGENT_STREAM_EVENT_TYPES = frozenset({'assistant', 'result'})


def _read_timeout_seconds() -> float:
    raw = os.environ.get('NEXUS_TRAJ_DB_READ_TIMEOUT')
    if raw is None:
        return 90.0
    try:
        value = float(raw)
    except ValueError:
        return 90.0
    return value if value > 0 else 90.0


_READ_TIMEOUT_SECONDS = _read_timeout_seconds()


def _utc_naive(value: datetime) -> datetime:
    """Move the key onto the UTC wall clock the rows were written with.

    ``Execution.created_at`` comes from ``distributed.utils.clock.utcnow()``, which is naive
    UTC, so a naive value already sits in the stored frame. An offset-aware one has to be
    converted rather than stripped: dropping ``+08:00`` would place the equality predicate
    eight hours away from the partition key of the very rows it is meant to find.
    """
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def normalize_parent_created_at(value: Any) -> str:
    """Render the Boundless partition key as a MySQL DATETIME(6) literal.

    ``datetime.isoformat()`` emits a ``T`` separator and may carry an offset; MySQL wants
    ``YYYY-MM-DD HH:MM:SS[.ffffff]``. Values cross the wire as strings (execution specs are
    JSON), so both shapes are accepted and normalized here.
    """
    if isinstance(value, datetime):
        return _utc_naive(value).strftime('%Y-%m-%d %H:%M:%S.%f')
    if not isinstance(value, str) or not value.strip():
        raise ValueError('invalid TDSQL parent_created_at')
    text = value.strip()
    if text.endswith(('Z', 'z')) and len(text) > 1:
        # fromisoformat only learned the military suffix in 3.11, and this package supports
        # 3.10; --parent-created-at documents plain ISO 8601 and must not shift meaning
        # with the interpreter running the export.
        text = f'{text[:-1]}+00:00'
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError('invalid TDSQL parent_created_at') from exc
    return _utc_naive(parsed).strftime('%Y-%m-%d %H:%M:%S.%f')


class InvalidPartitionKeyError(ValueError):
    """A Boundless partition key that is absent where required, or cannot be normalized.

    A ``ValueError`` subclass so callers that already catch the broader type keep working.
    """


def parse_partition_key(value: Any, *, required: bool = False) -> str | None:
    """The single entry point for turning a recorded partition key into a query literal.

    ``None`` means the envelope carries no key: correct for classic TDSQL, and a resolution
    failure on Boundless, where a missing filter turns every read into a full scan -- hence
    ``required``. Callers used to re-check the type and emptiness before delegating, which
    restated the checks in :func:`normalize_parent_created_at` and let each of them
    disagree about which malformed values were worth reporting.
    """
    if value is None:
        if required:
            raise InvalidPartitionKeyError('boundless TDSQL trace metadata requires parent_created_at')
        return None
    try:
        return normalize_parent_created_at(value)
    except ValueError as exc:
        raise InvalidPartitionKeyError(str(exc)) from exc


def _metadata_parent_created_at(metadata: Mapping[str, Any], *, dialect: str) -> str | None:
    """Boundless prunes partitions on this column, so every query must carry it."""
    if dialect != BOUNDLESS_DIALECT:
        return None
    return parse_partition_key(metadata.get('parent_created_at'), required=True)


def _query_group_id(metadata: Mapping[str, Any]) -> str | None:
    group_id = metadata.get('trajectory_group_id')
    if isinstance(group_id, str) and group_id:
        return group_id
    run_id = metadata.get('run_id')
    if isinstance(run_id, str) and run_id:
        return run_id
    return None


def _storage_table(metadata: Mapping[str, Any]) -> tuple[str, str | None, str | None]:
    scope = metadata.get('storage_scope', _SHARED_SCOPE)
    if scope == _SHARED_SCOPE:
        return SHARED_TRACE_TABLE, None, None
    if scope not in {_PRINCIPAL_SCOPE, _LEGACY_BATCH_SCOPE}:
        raise ValueError('invalid TDSQL trace storage_scope')
    principal_scope_id = metadata.get('principal_scope_id')
    if not isinstance(principal_scope_id, str):
        raise ValueError('invalid TDSQL principal_scope_id')
    parent_execution_id = metadata.get('parent_execution_id')
    if parent_execution_id is not None and (not isinstance(parent_execution_id, str) or not parent_execution_id):
        raise ValueError('invalid TDSQL parent_execution_id')
    if scope == _LEGACY_BATCH_SCOPE and parent_execution_id is None:
        raise ValueError('invalid TDSQL batch parent_execution_id')
    return principal_trace_table_name(principal_scope_id), principal_scope_id, parent_execution_id


def _raw_principal_scope(metadata: Mapping[str, Any]) -> tuple[str, str, str]:
    scope = metadata.get('storage_scope')
    if scope not in {_PRINCIPAL_SCOPE, _LEGACY_BATCH_SCOPE}:
        raise ValueError('invalid raw TDSQL trace storage_scope')
    principal_scope_id = metadata.get('principal_scope_id')
    if not isinstance(principal_scope_id, str):
        raise ValueError('invalid TDSQL principal_scope_id')
    table_name = principal_trace_table_name(principal_scope_id)
    parent_execution_id = metadata.get('parent_execution_id')
    if parent_execution_id is not None and (not isinstance(parent_execution_id, str) or not parent_execution_id):
        raise ValueError('invalid TDSQL parent_execution_id')
    run_id = metadata.get('run_id')
    if run_id is not None and (not isinstance(run_id, str) or not run_id):
        raise ValueError('invalid TDSQL run_id')
    if scope == _LEGACY_BATCH_SCOPE and parent_execution_id is None:
        raise ValueError('invalid TDSQL batch parent_execution_id')
    if parent_execution_id is not None:
        return table_name, 'parent_execution_id', parent_execution_id
    if isinstance(run_id, str) and run_id:
        return table_name, 'run_id', run_id
    raise ValueError('raw TDSQL trace metadata requires run_id or parent_execution_id')


def _single_raw_scope(metadata: Mapping[str, Any]) -> tuple[str, str]:
    table_name, scope_column, run_id = _raw_principal_scope(metadata)
    if scope_column != 'run_id' or len(run_id) > 128:
        raise ValueError('bounded raw reads require one run_id')
    return table_name, run_id


def _raw_span_id(metadata: Mapping[str, Any]) -> str:
    span = metadata.get('run_span_id')
    if not isinstance(span, str) or not 1 <= len(span) <= 128:
        raise ValueError('bounded raw reads require one run_span_id')
    return span


def _require_positive_limit(limit: int) -> None:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError('limit must be a positive integer')


def _require_cursor(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f'{field_name} must be a non-negative integer')


def _raw_event_read_query(
    metadata: Mapping[str, Any],
    cursor: int,
    limit: int,
    through_record_id: int | None,
    columns: str,
    *,
    dialect: str = TDSQL_DIALECT,
) -> tuple[str, tuple[Any, ...]]:
    table_name, scope_column, scope_value = _raw_principal_scope(metadata)
    parent_created_at = _metadata_parent_created_at(metadata, dialect=dialect)
    run_id = metadata.get('run_id')
    if not isinstance(run_id, str) or not run_id:
        raise ValueError('raw TDSQL trace metadata requires run_id')
    run_span_id = metadata.get('run_span_id')
    if not isinstance(run_span_id, str) or not run_span_id:
        raise ValueError('raw TDSQL trace metadata requires run_span_id')
    _require_cursor(cursor, 'cursor')
    _require_positive_limit(limit)
    if through_record_id is not None:
        _require_cursor(through_record_id, 'through_record_id')
        if through_record_id < cursor:
            raise ValueError('through_record_id must be greater than or equal to cursor')
    conditions = ['run_id = %s', 'run_span_id = %s']
    params: list[Any] = [run_id, run_span_id]
    if scope_column == 'parent_execution_id':
        conditions.append('parent_execution_id = %s')
        params.append(scope_value)
    if parent_created_at is not None:
        conditions.append('parent_created_at = %s')
        params.append(parent_created_at)
    conditions.append('record_id > %s')
    params.append(cursor)
    if through_record_id is not None:
        conditions.append('record_id <= %s')
        params.append(through_record_id)
    params.append(limit)
    return (
        f'SELECT {columns}\nFROM {table_name}\nWHERE {" AND ".join(conditions)}\nORDER BY record_id\nLIMIT %s',
        tuple(params),
    )


def _build_query(
    metadata: Mapping[str, Any],
    cursor: int,
    limit: int | None = None,
    *,
    descending: bool = False,
    dialect: str = TDSQL_DIALECT,
) -> tuple[str, tuple[Any, ...]]:
    table_name, principal_scope_id, parent_execution_id = _storage_table(metadata)
    parent_created_at = _metadata_parent_created_at(metadata, dialect=dialect)
    conditions: list[str] = []
    params: list[Any] = []
    for column in _BASE_QUERYABLE_COLUMNS:
        value = metadata.get(column)
        if isinstance(value, str) and value:
            conditions.append(f'{column} = %s')
            params.append(value)
    if principal_scope_id is None:
        group_id = _query_group_id(metadata)
        if group_id:
            conditions.insert(2 if len(conditions) >= 2 else len(conditions), 'trajectory_group_id = %s')
            params.insert(2 if len(params) >= 2 else len(params), group_id)
    elif parent_execution_id is not None:
        conditions.append('parent_execution_id = %s')
        params.append(parent_execution_id)
    if parent_created_at is not None:
        conditions.append('parent_created_at = %s')
        params.append(parent_created_at)
    if not conditions:
        raise FileNotFoundError('TDSQL trace metadata has no queryable fields')
    order = 'DESC' if descending else 'ASC'
    sql = (
        f'SELECT {_SELECT_COLUMNS}\n'
        f'FROM {table_name}\n'
        f'WHERE {" AND ".join(conditions)} AND record_id > %s\n'
        f'ORDER BY record_id {order}'
    )
    params.append(cursor)
    if limit is not None:
        sql = f'{sql}\nLIMIT %s'
        params.append(limit)
    return sql, tuple(params)


def _read_report_fields(metadata: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in (
        *_BASE_QUERYABLE_COLUMNS,
        'trajectory_group_id',
        'principal_scope_id',
        'parent_execution_id',
        'storage_scope',
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            fields[key] = value
    return fields


def _span_event_row(
    run_id: str,
    run_span_id: str,
    group_id: str,
    record: SpanEventRecord,
    *,
    principal_scope_id: str | None,
    parent_execution_id: str | None,
    event_data_json: str | None = None,
    parent_created_at: str | None = None,
) -> tuple[Any, ...]:
    common = (
        record.record_id,
        record.event_type,
        record.time_unix_nano,
        record.trace_id,
        record.span_id,
        record.parent_span_id,
        event_data_json if event_data_json is not None else json.dumps(record.event_data, default=str),
    )
    if principal_scope_id is not None:
        # Column order mirrors span_event_insert_sql(): the Boundless dialect inserts the
        # partition key right after parent_execution_id.
        if parent_created_at is not None:
            return (run_id, run_span_id, parent_execution_id, parent_created_at, *common)
        return (run_id, run_span_id, parent_execution_id, *common)
    return (run_id, run_span_id, group_id, *common)


def _run_output_from_record(record: SpanEventRecord) -> dict[str, Any] | None:
    if record.event_type != 'update':
        return None
    for item in reversed(record.event_data.get('events') or []):
        if item.get('name') != 'run.event':
            continue
        attributes = item.get('attributes') or {}
        if attributes.get('event_type') != 'run.output':
            continue
        event_data = attributes.get('event_data') or {}
        if not isinstance(event_data, dict):
            continue
        output = event_data.get('output')
        return output if isinstance(output, dict) else None
    return None


def _is_transient_stream_update(record: SpanEventRecord) -> bool:
    if record.event_type != 'update':
        return False
    event_data = record.event_data
    if event_data.get('attributes') or event_data.get('status'):
        return False
    events = event_data.get('events')
    return bool(events) and all(_is_transient_stream_event(event) for event in events)


def _is_transient_stream_event(event: Any) -> bool:
    if not isinstance(event, Mapping):
        return False
    name = event.get('name')
    if name not in _TRANSIENT_STREAM_EVENT_NAMES:
        return False
    if name != 'agent.stream':
        return True
    attributes = event.get('attributes')
    return not (isinstance(attributes, Mapping) and attributes.get('event_type') in _TERMINAL_AGENT_STREAM_EVENT_TYPES)


# Boundless reports the auto-extension inside a versioned comment, as in
# ``/*B!210604 INTERVAL(DAY, 1) */``; no information_schema view exposes it.
_INTERVAL_PARTITIONING_PATTERN = re.compile(r'INTERVAL\s*\(', re.IGNORECASE)


async def _boundless_interval_configured(cur: Any, table_name: str) -> bool:
    """Report whether ``table_name`` already auto-extends its RANGE layout.

    An undeterminable answer counts as "not configured" on purpose. Re-issuing SET INTERVAL
    fails loudly during a grant, with an operator watching; skipping it leaves a table that
    writes fine today and rejects every row tomorrow, long after anyone connects the two.
    """
    await cur.execute(f'SHOW CREATE TABLE {table_name}')
    row = await cur.fetchone()
    if not row or len(row) < 2 or not isinstance(row[1], str):
        return False
    return _INTERVAL_PARTITIONING_PATTERN.search(row[1]) is not None


@dataclass(slots=True)
class TDSQLTraceSink(RunTraceSink, RunTraceScopePreparer):
    host: str
    user: str
    password: str = field(default='', repr=False)
    db: str = ''
    port: int = 3306
    execution_id: str = ''
    assignment_id: str = ''
    attempt_id: str = ''
    max_run_output_bytes: int = 0
    dialect: str = TDSQL_DIALECT
    subpartitions: int = DEFAULT_BOUNDLESS_SUBPARTITIONS
    partition_interval: str = DEFAULT_BOUNDLESS_PARTITION_INTERVAL
    close_flush_timeout_seconds: float = field(default_factory=_default_close_flush_timeout_seconds)
    max_queue_items: int = field(default_factory=resolve_max_queue_items)
    max_queue_bytes: int = field(default_factory=resolve_max_queue_bytes)
    writer_shards: int = field(default_factory=resolve_writer_shards)
    _writer: TDSQLTraceWriter | None = field(default=None, init=False)
    _context: RunTraceContext | None = field(default=None, init=False)
    _run_id: str = field(default='', init=False)
    _run_span_id: str = field(default='', init=False)
    _trajectory_group_id: str = field(default='', init=False)
    _principal_scope_id: str | None = field(default=None, init=False)
    _parent_execution_id: str | None = field(default=None, init=False)
    _parent_created_at: str | None = field(default=None, init=False)
    _cursor: int = field(default=0, init=False)
    _last_seq: int = field(default=0, init=False)
    _closed: bool = field(default=False, init=False)
    _status: str = field(default='running', init=False)

    def __post_init__(self) -> None:
        self.max_queue_items = resolve_max_queue_items(self.max_queue_items)
        self.max_queue_bytes = resolve_max_queue_bytes(self.max_queue_bytes)
        self.writer_shards = resolve_writer_shards(self.writer_shards)
        if (
            isinstance(self.close_flush_timeout_seconds, bool)
            or not isinstance(self.close_flush_timeout_seconds, (int, float))
            or self.close_flush_timeout_seconds <= 0
            or not math.isfinite(self.close_flush_timeout_seconds)
        ):
            raise ValueError('close_flush_timeout_seconds must be a positive finite number of seconds')
        if (
            isinstance(self.max_run_output_bytes, bool)
            or not isinstance(self.max_run_output_bytes, int)
            or self.max_run_output_bytes < 0
        ):
            raise ValueError('max_run_output_bytes must be nonnegative')
        validate_dialect(self.dialect)
        if self.dialect == BOUNDLESS_DIALECT:
            validate_subpartitions(self.subpartitions)
            validate_partition_interval(self.partition_interval)
        self._db_config()

    def _db_config(self) -> dict[str, Any]:
        return build_db_config(
            host=self.host,
            user=self.user,
            password=self.password,
            db=self.db,
            port=self.port,
        )

    async def prepare_scope(self, scope: RunTraceScope) -> None:
        table_name = principal_trace_table_name(scope.scope_id)
        from nexus.runner.trace.tdsql._driver import connect as mysql_connect

        conn = None

        async def prepare() -> None:
            nonlocal conn
            conn = await mysql_connect(**self._db_config())
            async with conn.cursor() as cur:
                await cur.execute(
                    'SELECT IS_NULLABLE FROM information_schema.COLUMNS '
                    'WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s LIMIT 1',
                    (self.db, table_name, 'parent_execution_id'),
                )
                column = await cur.fetchone()
                if column is None:
                    await cur.execute(
                        principal_trace_table_statement(
                            scope.scope_id,
                            dialect=self.dialect,
                            subpartitions=self.subpartitions,
                        )
                    )
                    if self.dialect == BOUNDLESS_DIALECT:
                        # Must follow the CREATE: SET INTERVAL only extends an existing
                        # RANGE layout, and without it out-of-range inserts would fail.
                        await cur.execute(
                            principal_trace_interval_statement(
                                scope.scope_id,
                                partition_interval=self.partition_interval,
                            )
                        )
                elif self.dialect == BOUNDLESS_DIALECT:
                    # The table alone does not mean the scope is ready. CREATE and SET INTERVAL
                    # are two auto-committed statements, so a failure between them leaves a
                    # table that passes the column probe above and still cannot extend its
                    # partitions -- writes succeed until the seeded day runs out, then every
                    # row is rejected. Converge on the desired state instead of assuming the
                    # first attempt got that far.
                    if await _boundless_interval_configured(cur, table_name):
                        return
                    await cur.execute(
                        principal_trace_interval_statement(
                            scope.scope_id,
                            partition_interval=self.partition_interval,
                        )
                    )
                elif self.dialect == TDSQL_DIALECT and str(column[0]).upper() != 'YES':
                    await cur.execute(principal_trace_table_upgrade_statement(scope.scope_id))
                else:
                    return
            await conn.commit()

        try:
            await asyncio.wait_for(prepare(), timeout=_READ_TIMEOUT_SECONDS)
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()

    async def start(self, context: RunTraceContext) -> None:
        self._context = context
        self._run_id = context.run_id
        self._run_span_id = context.run_span_id
        metadata = context.metadata or {}
        storage_scope = metadata.get('storage_scope', _SHARED_SCOPE)
        if storage_scope in {_PRINCIPAL_SCOPE, _LEGACY_BATCH_SCOPE}:
            principal_scope_id = metadata.get('principal_scope_id')
            parent_execution_id = metadata.get('parent_execution_id')
            if not isinstance(principal_scope_id, str):
                raise ValueError('invalid TDSQL principal_scope_id')
            if parent_execution_id is not None and (
                not isinstance(parent_execution_id, str) or not parent_execution_id
            ):
                raise ValueError('invalid TDSQL parent_execution_id')
            if storage_scope == _LEGACY_BATCH_SCOPE and parent_execution_id is None:
                raise ValueError('invalid TDSQL batch parent_execution_id')
            principal_trace_table_name(principal_scope_id)
            self._principal_scope_id = principal_scope_id
            self._parent_execution_id = parent_execution_id
            if self.dialect == BOUNDLESS_DIALECT:
                # NOT NULL partition key: refuse to start rather than fail every insert later.
                self._parent_created_at = normalize_parent_created_at(metadata.get('parent_created_at'))
        elif storage_scope != _SHARED_SCOPE:
            raise ValueError('invalid TDSQL trace storage_scope')
        elif metadata.get('principal_scope_id') is not None or metadata.get('parent_execution_id') is not None:
            raise ValueError('TDSQL principal trace metadata requires principal_v1 storage_scope')
        elif self.dialect == BOUNDLESS_DIALECT:
            # Boundless tables are provisioned per principal; the shared table has no
            # partition key and is never created there (see schema.schema_statements).
            raise ValueError('boundless TDSQL trace backend requires principal_v1 storage_scope')
        group_id = metadata.get('trajectory_group_id')
        self._trajectory_group_id = str(group_id or context.run_id)

    async def on_event(self, context: RunTraceContext, event: SpanEventRecord) -> None:
        self._context = context
        if _is_transient_stream_update(event):
            self._cursor = max(self._cursor, event.record_id)
            return
        from nexus.runner.trace.tdsql.writer import get_writer

        self._writer = await get_writer(
            self._db_config(),
            dialect=self.dialect,
            max_queue_items=self.max_queue_items,
            max_queue_bytes=self.max_queue_bytes,
            writer_shards=self.writer_shards,
            run_span_id=self._run_span_id,
        )
        await asyncio.to_thread(self.enqueue_record, event)

    def _event_data_json(self, record: SpanEventRecord) -> str:
        from nexus.utils.json_stream import bounded_json_text

        if (
            not self.max_run_output_bytes
            or self._context is None
            or not self._context.result_artifact_available
            or record.span_id != self._run_span_id
            or _run_output_from_record(record) is None
        ):
            return json.dumps(record.event_data, default=str)
        encoded, size = bounded_json_text(record.event_data, self.max_run_output_bytes)
        if encoded is not None:
            return encoded
        # Only change the TDSQL projection; the result sink retains the complete output.
        projected = dict(record.event_data)
        projected['events'] = []
        for event in record.event_data.get('events') or []:
            attributes = event.get('attributes') or {}
            if event.get('name') == 'run.event' and attributes.get('event_type') == 'run.output':
                data = {
                    **(attributes.get('event_data') or {}),
                    'output': None,
                    'output_omitted': True,
                    'output_omitted_reason': 'size_limit',
                    'original_event_bytes': size,
                }
                event = {**event, 'attributes': {**attributes, 'event_data': data}}
            projected['events'].append(event)
        return json.dumps(projected, default=str)

    def enqueue_record(self, record: SpanEventRecord) -> None:
        if not isinstance(record, SpanEventRecord):
            return
        self._cursor = max(self._cursor, record.record_id)
        row = _span_event_row(
            self._run_id,
            self._run_span_id,
            self._trajectory_group_id,
            record,
            principal_scope_id=self._principal_scope_id,
            parent_execution_id=self._parent_execution_id,
            event_data_json=self._event_data_json(record),
            parent_created_at=self._parent_created_at,
        )
        from nexus.runner.trace.tdsql.writer import log_dropped_write

        if self._writer is None:
            self._status = 'upload_unavailable'
            log_dropped_write('upload_unavailable', row, **self._drop_fields())
            raise RuntimeError(f'TDSQL trace writer unavailable for run {self._run_id}')
        try:
            seq = self._writer.enqueue_span_event(
                row,
                run_span_id=self._run_span_id,
                principal_scope_id=self._principal_scope_id,
                parent_execution_id=self._parent_execution_id,
            )
        except Exception:
            self._status = 'artifact_failed'
            log_dropped_write('enqueue_failed', row, exc_info=True, **self._drop_fields())
            raise
        if seq == 0:
            self._status = 'artifact_failed'
            log_dropped_write('enqueue_rejected', row, **self._drop_fields())
            raise RuntimeError(f'TDSQL trace upload rejected (backpressure) for run {self._run_id}')
        if seq > self._last_seq:
            self._last_seq = seq

    def _drop_fields(self) -> dict[str, Any]:
        fields = {
            'execution_id': self.execution_id,
            'assignment_id': self.assignment_id,
            'attempt_id': self.attempt_id,
            'run_id': self._run_id,
            'run_span_id': self._run_span_id,
            'storage_scope': _PRINCIPAL_SCOPE if self._principal_scope_id else _SHARED_SCOPE,
        }
        if self._principal_scope_id:
            fields['principal_scope_id'] = self._principal_scope_id
            if self._parent_execution_id is not None:
                fields['parent_execution_id'] = self._parent_execution_id
        else:
            fields['trajectory_group_id'] = self._trajectory_group_id
        return fields

    async def close(self, context: RunTraceContext | None = None) -> None:
        if context is not None:
            self._context = context
        self._closed = True
        writer = self._writer
        if writer is None or self._last_seq <= 0:
            from nexus.runner.trace.tdsql.writer import log_dropped_write

            reason = 'upload_unavailable' if writer is None else 'no_events'
            log_dropped_write(
                'close_no_flush',
                (f'TDSQL trace close skipped flush ({reason}): last_seq={self._last_seq}',),
                **self._drop_fields(),
            )
            return
        try:
            await writer.wait_until_settled(self._last_seq, timeout=self.close_flush_timeout_seconds)
        except TimeoutError:
            self._status = 'upload_timeout'
            from nexus.runner.trace.tdsql.writer import log_dropped_write

            log_dropped_write(
                'close_flush_timeout',
                (
                    f'TDSQL trace close flush timed out after {self.close_flush_timeout_seconds:g}s: '
                    f'last_seq={self._last_seq} settled_through={writer.settled_through}',
                ),
                **self._drop_fields(),
            )
            raise RuntimeError(
                f'TDSQL trace upload did not settle within {self.close_flush_timeout_seconds:g}s for run {self._run_id}'
            ) from None
        failed_seq = writer.consume_run_failure(self._run_span_id)
        if failed_seq is not None:
            self._status = 'artifact_failed'
            from nexus.runner.trace.tdsql.writer import log_dropped_write

            log_dropped_write(
                'close_upload_dropped',
                (f'TDSQL trace upload dropped rows: last_seq={self._last_seq} dropped_seq={failed_seq}',),
                **self._drop_fields(),
            )
            raise RuntimeError(
                f'TDSQL trace upload dropped rows for run {self._run_id} (run_span_id {self._run_span_id})'
            )
        self._status = 'succeeded'

    def metadata(self) -> dict[str, Any]:
        context = self._context
        payload: dict[str, Any] = {
            'run_id': self._run_id,
            'run_span_id': self._run_span_id,
            'execution_id': self.execution_id,
            'assignment_id': self.assignment_id,
            'attempt_id': self.attempt_id,
            'trace_event_cursor': self._cursor,
            'status': self._status,
            'storage_scope': _PRINCIPAL_SCOPE if self._principal_scope_id else _SHARED_SCOPE,
        }
        if self._principal_scope_id:
            payload['principal_scope_id'] = self._principal_scope_id
            if self._parent_execution_id is not None:
                payload['parent_execution_id'] = self._parent_execution_id
            if self._parent_created_at is not None:
                # Readers need the partition key to prune; the envelope is their only source.
                payload['parent_created_at'] = self._parent_created_at
        else:
            payload['trajectory_group_id'] = self._trajectory_group_id
        if context is not None:
            payload['run_span_id'] = context.run_span_id
            payload['run_kind'] = context.run_kind
            if context.metadata:
                payload['metadata'] = dict(context.metadata)
        return payload


@dataclass(slots=True)
class TDSQLTraceSource(RunTraceSource):
    host: str
    user: str
    password: str = field(default='', repr=False)
    db: str = ''
    port: int = 3306
    slave_read_hint: str = '/*slave*/ '
    dialect: str = TDSQL_DIALECT
    # Accepted so a single backend config block can feed both sink and source kwargs.
    subpartitions: int = DEFAULT_BOUNDLESS_SUBPARTITIONS
    partition_interval: str = DEFAULT_BOUNDLESS_PARTITION_INTERVAL
    read_driver: Literal['asyncmy', 'aiomysql'] = 'asyncmy'

    def __post_init__(self) -> None:
        validate_dialect(self.dialect)
        if self.read_driver not in {'asyncmy', 'aiomysql'}:
            raise ValueError('unsupported TDSQL read driver')
        self._db_config()

    def _read_driver_module(self) -> Any:
        if self.read_driver == 'aiomysql':
            import aiomysql

            return aiomysql
        from nexus.runner.trace.tdsql import _driver

        return _driver

    def _db_config(self) -> dict[str, Any]:
        return build_db_config(
            host=self.host,
            user=self.user,
            password=self.password,
            db=self.db,
            port=self.port,
        )

    async def read_events_after(
        self,
        metadata: Mapping[str, Any],
        cursor: int,
        *,
        limit: int | None = None,
        descending: bool = False,
    ) -> list[SpanEventRecord]:
        return await self._read_span_records(metadata, cursor, limit=limit, descending=descending, use_slave=False)

    async def discover_raw_streams(
        self,
        metadata: Mapping[str, Any],
        *,
        after: tuple[str, str] | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Discover principal-table streams for Batch-parent or Single-run metadata.

        ``after`` is the opaque key from the last database-returned row; its order is defined by the database
        collation and must not be recomputed in Python. Each call uses one connection and closes it after the query,
        ending the read transaction so a later discovery sweep cannot retain a stale snapshot.
        """
        table_name, scope_column, scope_value = _raw_principal_scope(metadata)
        parent_created_at = _metadata_parent_created_at(metadata, dialect=self.dialect)
        _require_positive_limit(limit)
        if after is not None and (
            not isinstance(after, tuple)
            or len(after) != 2
            or any(not isinstance(value, str) or not value for value in after)
        ):
            raise ValueError('after must be a pair of non-empty stream identifiers')

        conditions = [f'{scope_column} = %s']
        params: list[Any] = [scope_value]
        # Without the partition key this GROUP BY would sweep every day partition.
        if parent_created_at is not None:
            conditions.append('parent_created_at = %s')
            params.append(parent_created_at)
        if after is not None:
            conditions.append('(run_id > %s OR (run_id = %s AND run_span_id > %s))')
            params.extend((after[0], after[0], after[1]))
        params.append(limit)
        sql = (
            'SELECT run_id, run_span_id, MAX(record_id) AS max_record_id\n'
            f'FROM {table_name}\n'
            f'WHERE {" AND ".join(conditions)}\n'
            'GROUP BY run_id, run_span_id\n'
            'ORDER BY run_id, run_span_id\n'
            'LIMIT %s'
        )
        rows = await self._select_span_events(
            sql,
            tuple(params),
            report_fields=_read_report_fields(metadata),
            payload_column=None,
            log_exception_details=False,
        )
        if scope_column == 'run_id' and any(row[0] != scope_value for row in rows):
            raise ValueError('raw TDSQL discovery returned a mismatched stream identity')
        return [{'run_id': row[0], 'run_span_id': row[1], 'max_record_id': row[2]} for row in rows]

    async def plan_raw_event_read(
        self,
        metadata: Mapping[str, Any],
        cursor: int,
        *,
        limit: int = 500,
        target_bytes: int = 8 * 1024 * 1024,
        through_record_id: int | None = None,
    ) -> RawTraceReadPlan:
        """Measure a small prefix before reserving memory for complete raw bodies.

        A first row larger than the target remains in the plan on its own. The
        caller must read it exclusively rather than truncate it or skip ahead.
        """
        _require_positive_limit(limit)
        _require_positive_limit(target_bytes)
        if limit > 500:
            raise ValueError('raw event plan limit exceeds 500')
        sql, params = _raw_event_read_query(
            metadata,
            cursor,
            limit + 1,
            through_record_id,
            'record_id, OCTET_LENGTH(event_data)',
            dialect=self.dialect,
        )
        if through_record_id == cursor:
            return RawTraceReadPlan((), False)
        rows = await self._select_span_events(
            sql, params, report_fields=_read_report_fields(metadata), payload_column=None, log_exception_details=False
        )
        selected: list[tuple[int, int]] = []
        total = 0
        previous = cursor
        for record_id, size in rows:
            if type(record_id) is not int or record_id <= previous or type(size) is not int or size < 0:
                raise RawTraceRecordInvalidError('raw event read plan invalid')
            if through_record_id is not None and record_id > through_record_id:
                raise RawTraceRecordInvalidError('raw event read plan exceeds snapshot')
            if len(selected) == limit or (selected and total + size > target_bytes):
                return RawTraceReadPlan(tuple(selected), True)
            selected.append((record_id, size))
            previous = record_id
            total += size
        return RawTraceReadPlan(tuple(selected), False)

    async def read_raw_events_after(
        self,
        metadata: Mapping[str, Any],
        cursor: int,
        *,
        limit: int = 500,
        through_record_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Read exact raw principal-table rows for one run and run span.

        ``cursor`` is exclusive and ``through_record_id`` is an optional inclusive snapshot bound. The result keeps
        all 12 principal columns, leaves ``event_data`` unchanged, and ISO-formats database datetime values. Each
        query uses one connection that is closed in ``finally`` to end its read transaction and avoid stale snapshots.
        """
        sql, params = _raw_event_read_query(
            metadata, cursor, limit, through_record_id, _RAW_SELECT_COLUMNS, dialect=self.dialect
        )
        if through_record_id == cursor:
            return []
        run_id, run_span_id = metadata['run_id'], metadata['run_span_id']
        expected_parent = metadata.get('parent_execution_id')
        rows = await self._select_span_events(
            sql,
            tuple(params),
            report_fields=_read_report_fields(metadata),
            payload_column=10,
            log_exception_details=False,
        )
        for row in rows:
            if row[1] != run_id or row[2] != run_span_id or (expected_parent is not None and row[3] != expected_parent):
                raise ValueError('raw TDSQL read returned a mismatched stream identity')

        records: list[dict[str, Any]] = []
        for row in rows:
            record = dict(zip(_RAW_COLUMN_NAMES, row, strict=True))
            created_at = record['created_at']
            if created_at is not None and not isinstance(created_at, str):
                record['created_at'] = created_at.isoformat()
            records.append(record)
        return records

    async def _terminate_raw_query(self, thread_id: int, query_marker: str) -> None:
        connect = self._read_driver_module().connect

        controller = None
        try:
            controller = await connect(**{**self._db_config(), 'autocommit': True, 'connect_timeout': 1})
            async with controller.cursor() as cursor:
                try:
                    # Use SQL, not COM_PROCESS_KILL, which TDSQL rejects. The
                    # connection ID comes from our driver, never the request.
                    await cursor.execute(f'KILL CONNECTION {int(thread_id)}')
                except Exception as error:
                    # TDSQL reports 704 when the frontend connection is already
                    # gone. Neither code proves that its backend query stopped;
                    # the marker check below remains mandatory.
                    if not error.args or error.args[0] not in (1094, 704):
                        raise
                while True:
                    await cursor.execute('SHOW FULL PROCESSLIST')
                    info_index = next(
                        index for index, column in enumerate(cursor.description) if column[0].lower() == 'info'
                    )
                    rows = await cursor.fetchall()
                    if not any(query_marker in (row[info_index] or '') for row in rows):
                        return
                    await asyncio.sleep(0.02)
        finally:
            if controller is not None:
                controller.close()

    async def _close_raw_cursor(
        self,
        connection: Any,
        cursor: Any,
        *,
        exhausted: bool,
        thread_id: int,
        query_marker: str,
        pending: asyncio.Task[Any] | None = None,
    ) -> None:
        connection.close()
        failed = False
        try:
            if not exhausted:
                try:
                    await asyncio.wait_for(self._terminate_raw_query(thread_id, query_marker), timeout=2)
                except Exception:
                    failed = True
            if pending is not None:
                try:
                    await asyncio.wait_for(pending, timeout=1)
                except asyncio.TimeoutError:
                    failed = True
                except Exception:
                    pass  # Socket shutdown completes the shielded driver await.
        finally:
            # asyncmy's result destructor calls an async drain without awaiting
            # it. Once the dedicated socket is closed there is nothing to drain;
            # disarm that destructor before invoking the public cursor cleanup.
            result = getattr(cursor, '_result', None)
            if result is not None:
                result.unbuffered_active = False
            with contextlib.suppress(Exception):
                await cursor.close()
        if failed:
            raise RawTraceCleanupError('raw trace query cleanup could not be confirmed')

    @contextlib.asynccontextmanager
    async def _raw_cursor(self) -> AsyncIterator[Any]:
        driver = self._read_driver_module()

        connection = None
        cursor = None
        wrapped = None
        thread_id = 0
        try:
            connection = await driver.connect(**{**self._db_config(), 'autocommit': True})
            thread_id = connection.thread_id()
            cursor = connection.cursor(driver.SSCursor)
            if inspect.isawaitable(cursor):
                cursor = await cursor
            wrapped = _RawCursor(cursor)
            yield wrapped
        finally:
            if connection is not None and cursor is not None:
                cleanup = asyncio.create_task(
                    self._close_raw_cursor(
                        connection,
                        cursor,
                        exhausted=wrapped is not None and wrapped.exhausted,
                        thread_id=thread_id,
                        query_marker=wrapped.query_marker if wrapped is not None else '',
                        pending=wrapped.pending if wrapped is not None else None,
                    )
                )
                cancelled = False
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        cancelled = True
                cleanup.result()
                if cancelled:
                    raise asyncio.CancelledError
            elif connection is not None:
                connection.close()

    async def list_raw_streams(
        self,
        metadata: Mapping[str, Any],
        *,
        after: str | None = None,
        limit: int = 100,
    ) -> list[str]:
        """List only database-ordered stream keys, without scanning MAX(record_id)."""
        table_name, run_id = _single_raw_scope(metadata)
        # DISTINCT plus ORDER BY is the costliest read here: without the partition key it
        # deduplicates and sorts across every day partition of the table.
        parent_created_at = _metadata_parent_created_at(metadata, dialect=self.dialect)
        _require_positive_limit(limit)
        if limit > 201 or (after is not None and (not isinstance(after, str) or not 1 <= len(after) <= 128)):
            raise ValueError('invalid raw stream discovery bounds')
        conditions = ['run_id = %s']
        params: list[Any] = [run_id]
        if parent_created_at is not None:
            conditions.append('parent_created_at = %s')
            params.append(parent_created_at)
        if after is not None:
            conditions.append('run_span_id > %s')
            params.append(after)
        params.append(limit)
        sql = (
            f'SELECT DISTINCT run_span_id FROM {table_name} '
            f'WHERE {" AND ".join(conditions)} ORDER BY run_span_id LIMIT %s'
        )
        result: list[str] = []
        async with self._raw_cursor() as cursor:
            await cursor.execute(sql, tuple(params))
            while (row := await cursor.fetchone()) is not None:
                if len(row) != 1 or not isinstance(row[0], str) or not 1 <= len(row[0]) <= 128:
                    raise RawTraceRecordInvalidError('invalid raw stream identity')
                result.append(row[0])
        return result

    async def raw_stream_exists(self, metadata: Mapping[str, Any]) -> bool:
        table_name, run_id = _single_raw_scope(metadata)
        span = _raw_span_id(metadata)
        # Only reached when a page came back empty, so LIMIT 1 finds nothing and the scan
        # runs to completion; the partition key is what keeps that scan to one partition.
        parent_created_at = _metadata_parent_created_at(metadata, dialect=self.dialect)
        conditions = ['run_id = %s', 'run_span_id = %s']
        params: list[Any] = [run_id, span]
        if parent_created_at is not None:
            conditions.append('parent_created_at = %s')
            params.append(parent_created_at)
        async with self._raw_cursor() as cursor:
            await cursor.execute(f'SELECT 1 FROM {table_name} WHERE {" AND ".join(conditions)} LIMIT 1', tuple(params))
            exists = await cursor.fetchone() is not None
            if exists:
                await cursor.fetchone()
            return exists

    async def read_raw_event_page(
        self,
        metadata: Mapping[str, Any],
        after_record_id: int,
        *,
        limit: int = 200,
        max_event_bytes: int = 8 * 1024 * 1024,
        omit_oversized_event_data: bool = False,
        accept_record: Callable[[dict[str, Any]], bool],
    ) -> RawTraceEventPage:
        """Stream a bounded prefix, never transferring an oversized event body.

        The caller's synchronous accept_record callback accounts for its transport
        encoding budget. Rejected and malformed rows never advance the prefix.
        SQL failures fail the whole page; record failures after a valid prefix are
        exposed on the next read at the same position.
        """
        table_name, run_id = _single_raw_scope(metadata)
        span = _raw_span_id(metadata)
        # The console pages through this query, so an unpruned variant would sweep every
        # day partition on every page rather than once.
        parent_created_at = _metadata_parent_created_at(metadata, dialect=self.dialect)
        _require_cursor(after_record_id, 'after_record_id')
        _require_positive_limit(limit)
        _require_positive_limit(max_event_bytes)
        if limit > 500:
            raise ValueError('raw event page limit exceeds 500')
        columns = _RAW_SELECT_COLUMNS.replace(
            'event_data,', 'CASE WHEN OCTET_LENGTH(event_data) <= %s THEN event_data ELSE NULL END,'
        )
        conditions = ['run_id = %s', 'run_span_id = %s']
        # The projection placeholder above is bound before any WHERE parameter.
        params: list[Any] = [max_event_bytes, run_id, span]
        if parent_created_at is not None:
            conditions.append('parent_created_at = %s')
            params.append(parent_created_at)
        conditions.append('record_id > %s')
        params.extend((after_record_id, limit + 1))
        sql = (
            f'SELECT {columns}, OCTET_LENGTH(event_data) FROM {table_name} '
            f'WHERE {" AND ".join(conditions)} ORDER BY record_id LIMIT %s'
        )
        items: list[dict[str, Any]] = []
        previous = after_record_id
        async with self._raw_cursor() as cursor:
            await cursor.execute(sql, tuple(params))
            while (row := await cursor.fetchone()) is not None:
                if len(items) == limit:
                    await cursor.fetchone()  # Consume EOF after the SQL limit+1 lookahead.
                    return RawTraceEventPage(items, True)
                try:
                    if len(row) != 13 or row[1] != run_id or row[2] != span:
                        raise RawTraceRecordInvalidError('raw event identity mismatch')
                    record_id = row[4]
                    if isinstance(record_id, bool) or not isinstance(record_id, int) or record_id <= previous:
                        raise RawTraceRecordInvalidError('raw event ordering invalid')
                    raw_bytes = row[12]
                    if type(raw_bytes) is not int or raw_bytes < 0:
                        raise RawTraceRecordInvalidError('raw event size invalid')
                    omitted = raw_bytes > max_event_bytes
                    if omitted and not omit_oversized_event_data:
                        raise RawTraceEventTooLargeError(record_id, raw_bytes)
                    record = dict(zip(_RAW_COLUMN_NAMES, row[:12], strict=True))
                    if (omitted and record['event_data'] is not None) or (
                        not omitted and not isinstance(record['event_data'], str)
                    ):
                        raise RawTraceRecordInvalidError('raw event body projection invalid')
                    if omit_oversized_event_data:
                        record['event_data_bytes'] = raw_bytes
                        record['event_data_omitted'] = omitted
                    created_at = record['created_at']
                    if created_at is not None and not isinstance(created_at, str):
                        record['created_at'] = created_at.isoformat()
                    if not accept_record(record):
                        return RawTraceEventPage(items, True)
                except (RawTraceRecordInvalidError, RawTraceEventTooLargeError):
                    if items:
                        return RawTraceEventPage(items, True)
                    raise
                items.append(record)
                previous = record_id
        return RawTraceEventPage(items, False)

    async def _read_span_records(
        self,
        metadata: Mapping[str, Any],
        cursor: int,
        *,
        limit: int | None = None,
        descending: bool = False,
        use_slave: bool = True,
    ) -> list[SpanEventRecord]:
        if cursor < 0:
            raise ValueError('cursor must not be negative')
        if limit is not None and limit < 1:
            raise ValueError('limit must be at least 1')
        sql, params = _build_query(metadata, cursor, limit, descending=descending, dialect=self.dialect)
        if use_slave and self.slave_read_hint:
            sql = f'{self.slave_read_hint}{sql}'
        report_fields = _read_report_fields(metadata)
        from nexus.runner.trace.tdsql.writer import report_monitor

        started_at = time.monotonic()
        rows = await self._select_span_events(sql, params, report_fields=report_fields)
        elapsed_ms = (time.monotonic() - started_at) * 1000
        report_monitor(
            'read_events_after',
            f'trajectory_tdsql: read_events_after elapsed_ms={elapsed_ms:.1f} rows={len(rows)} '
            f'sql={sql!r} params={params!r}',
            level=logging.DEBUG,
            elapsed_ms=round(elapsed_ms, 3),
            rows=len(rows),
            **report_fields,
        )
        return [
            SpanEventRecord(
                record_id=row[0],
                event_type=row[1],
                time_unix_nano=row[2],
                trace_id=row[3],
                span_id=row[4],
                parent_span_id=row[5],
                event_data=_json_loads(row[6]) if row[6] else {},
            )
            for row in rows
        ]

    async def read_result(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        run_span_id = metadata.get('run_span_id')
        if not isinstance(run_span_id, str) or not run_span_id:
            raise FileNotFoundError('TDSQL trace result metadata is missing run_span_id')
        query_metadata = {**metadata, 'event_type': 'update', 'span_id': run_span_id}
        for record in await self._read_span_records(
            query_metadata,
            0,
            limit=_READ_RESULT_MAX_UPDATES,
            descending=True,
            use_slave=False,
        ):
            output = _run_output_from_record(record)
            if output is not None:
                return output
            if any(
                item.get('name') == 'run.event' and (item.get('attributes') or {}).get('event_type') == 'run.output'
                for item in record.event_data.get('events') or []
            ):
                break
        raise FileNotFoundError('TDSQL trace result is not available')

    async def _select_span_events(
        self,
        sql: str,
        params: tuple[Any, ...],
        *,
        report_fields: Mapping[str, Any] | None = None,
        payload_column: int | None = 6,
        log_exception_details: bool = True,
    ) -> list[tuple[Any, ...]]:
        from nexus.runner.trace.tdsql.writer import report_monitor

        fields = dict(report_fields or {})
        run_id = fields.get('run_id', '')
        config = self._db_config()
        try:
            driver = self._read_driver_module()
        except Exception as exc:  # pragma: no cover - import guard
            raise FileNotFoundError('no async mysql driver is available') from exc

        started_at = time.monotonic()
        conn = None

        async def _run() -> list[tuple[Any, ...]]:
            nonlocal conn
            with measure_database('connect', source='trace_db', acquire=True):
                conn = await driver.connect(**config)
            with measure_database('select', source='trace_db'):
                async with conn.cursor() as cur:
                    await cur.execute(sql, params)
                    fetched = cur.fetchall()
                    if inspect.isawaitable(fetched):
                        fetched = await fetched
                    return list(fetched)

        try:
            rows = await asyncio.wait_for(_run(), timeout=_READ_TIMEOUT_SECONDS)
        except Exception as exc:
            elapsed_ms = (time.monotonic() - started_at) * 1000
            report_monitor(
                'read_failure',
                f'trajectory_tdsql: db read_failure elapsed_ms={elapsed_ms:.1f} run_id={run_id}',
                level=logging.WARNING,
                exc_info=log_exception_details,
                elapsed_ms=round(elapsed_ms, 3),
                **fields,
            )
            if isinstance(exc, (asyncio.TimeoutError, TimeoutError, driver.OperationalError, driver.InterfaceError)):
                raise RunTraceSourceUnavailable(f'TDSQL trace source unavailable: {exc}') from exc
            raise
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()
        elapsed = time.monotonic() - started_at
        if elapsed > SLOW_OPERATION_WARN_SECONDS:
            payload_bytes = 0
            if payload_column is not None:
                payload_bytes = sum(
                    len(row[payload_column])
                    for row in rows
                    if len(row) > payload_column and isinstance(row[payload_column], str)
                )
            report_monitor(
                'read_slow',
                f'trajectory_tdsql: slow db read {elapsed:.1f}s rows={len(rows)} '
                f'bytes={payload_bytes} elapsed_ms={elapsed * 1000:.1f}',
                level=logging.WARNING,
                rows=len(rows),
                payload_bytes=payload_bytes,
                elapsed_ms=round(elapsed * 1000, 3),
                **fields,
            )
        return rows


@dataclass(slots=True)
class TDSQLBoundlessTraceSink(TDSQLTraceSink):
    """Boundless-flavoured sink, selected by class rather than by a config flag.

    Boundless entries are configured under ``extend_run_trace_sinks``, which builds that
    predate the field never see. Naming a class they also lack therefore costs nothing and
    buys a second guarantee: such a build cannot drive this backend even by accident, which
    a flag on the shared class could not prevent -- it would construct and then write rows
    the partitioned table rejects.
    """

    dialect: str = BOUNDLESS_DIALECT


@dataclass(slots=True)
class TDSQLBoundlessTraceSource(TDSQLTraceSource):
    """Reader counterpart of :class:`TDSQLBoundlessTraceSink`; adds the partition predicate."""

    dialect: str = BOUNDLESS_DIALECT


__all__ = [
    'InvalidPartitionKeyError',
    'RawTraceCleanupError',
    'RawTraceEventPage',
    'RawTraceEventTooLargeError',
    'RawTraceReadPlan',
    'RawTraceRecordInvalidError',
    'TDSQLBoundlessTraceSink',
    'TDSQLBoundlessTraceSource',
    'TDSQLTraceSink',
    'TDSQLTraceSource',
    'normalize_parent_created_at',
    'parse_partition_key',
]
