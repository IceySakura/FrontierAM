"""Cloud export results."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
import json
import logging
import os
import queue
import random
import threading
import time

from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nexus.distributed.export.client import CloudApi, _loads_json_bytes
from nexus.distributed.export.follow import (
    _retryable,
    resolve_trace_read_scope,
)
from nexus.distributed.export.types import (
    _STATUS_MAP,
    ExportDeadlineExceeded,
    ExportError,
    ResultStatus,
    TraceDataInvalid,
    TraceDatabaseAuth,
    TraceDatabaseError,
    TraceDatabaseInterrupted,
    TraceDatabaseSchema,
    TraceDatabaseThrottled,
    TraceDatabaseTimeout,
    TraceDatabaseUnavailable,
    TraceProcessingError,
    TraceProcessingTimeout,
    TraceResult,
    resolve_result_status,
)
from nexus.distributed.export.writer import (
    AtomicJsonOutput,
    AtomicJsonlOutput,
    ResultSpillFile,
    ResultSpillPool,
    _has_artifacts,  # noqa: F401  (re-exported for backwards compatibility)
    _unlink_and_fsync,
)
from nexus.runner.trace.tdsql.schema import PRINCIPAL_TRACE_TABLE_PATTERN
from nexus.runner.trace.tdsql.sink_source import normalize_parent_created_at
from nexus.utils.json_scan import JsonFallbackNeeded, JsonFieldScanner, iter_array_element_spans
from nexus.utils.json_transcode import iter_spaced_json


logger = logging.getLogger(__name__)


_TERMINAL_EXECUTION_STATUSES = frozenset({'succeeded', 'failed', 'killed', 'lost', 'expired'})


_TERMINAL_ITEM_STATUSES = _TERMINAL_EXECUTION_STATUSES | {'cancelled'}


_SUPPORTED_SPEC_TYPES = frozenset({'nexus_batch', 'nexus_single', 'nexus_live'})


_RESULT_CONCURRENCY_ENV = 'NEXUS_EXPORT_RESULT_CONCURRENCY'
_DEFAULT_RESULT_CONCURRENCY = 4
_MAX_RESULT_CONCURRENCY = 8
# The in-RAM pipeline (Trace-Database / embedder fallback without a spill pool)
# keeps ~N records' payload in flight: the DB hands the whole event_data column
# to the reader as one string, and the fallback paths (scanner-incompatible
# spans, orjson fallback, TRANSCODE=0) parse transient full object graphs. Its
# depth must NOT ride the spill-pipeline concurrency knob; it stays at the
# legacy value by default. The escape hatch exists for ops: dropping it to 1
# serializes the pipeline so no download ever overlaps another record's
# processing window.
_IN_RAM_CONCURRENCY_ENV = 'NEXUS_EXPORT_IN_RAM_CONCURRENCY'
_DEFAULT_IN_RAM_CONCURRENCY = 2
_MAX_IN_RAM_CONCURRENCY = 4


def _in_ram_concurrency() -> int:
    """Depth of the no-spill (Trace-Database / embedder) pipeline.

    Every slot holds one full body plus its object graph in RAM, so the
    default stays at the legacy value of 2 and the upper bound is tight:
    each extra slot is a potentially GB-scale resident object graph.
    """
    raw = os.environ.get(_IN_RAM_CONCURRENCY_ENV)
    try:
        value = int(raw) if raw else _DEFAULT_IN_RAM_CONCURRENCY
    except ValueError:
        value = _DEFAULT_IN_RAM_CONCURRENCY
    return max(1, min(value, _MAX_IN_RAM_CONCURRENCY))


def _result_concurrency() -> int:
    """Pipeline depth: N-1 spill downloads overlap the single consuming slot.

    Default 4 matches the measured rates (download ~25 MiB/s per stream,
    consume ~70 MiB/s single-slot → 3 prefetch streams keep the consumer fed)
    and the server's admission quota (artifact_run_result_max_inflight=4).
    Upper bound also caps transient disk usage at ~N × largest single result.
    """
    raw = os.environ.get(_RESULT_CONCURRENCY_ENV)
    try:
        value = int(raw) if raw else _DEFAULT_RESULT_CONCURRENCY
    except ValueError:
        value = _DEFAULT_RESULT_CONCURRENCY
    return max(1, min(value, _MAX_RESULT_CONCURRENCY))


# Missing-value sentinel for _single_record(prefetched=...): distinguishes "no
# prefetched body" from a prefetched 404 (None = missing result).
_MISSING: Any = object()


_TDSQL_ID_CHUNK_SIZE = 1


_TDSQL_ROOT_PAGE_SIZE = 100


_TDSQL_CONCURRENCY = 4


_TDSQL_ATTEMPTS = 3


_TDSQL_TIMEOUT_SECONDS = 900.0


# The schema probe is a primary-key lookup in information_schema, not a scan: it
# must be cheap enough that a slow or unavailable probe cannot delay the first
# record read. The data-query budget below stays reserved for actual reads.
_TDSQL_PROBE_TIMEOUT_SECONDS = 5.0


_PARTITION_PROBE_ENV = 'NEXUS_EXPORT_TRACE_PARTITION_PROBE'
_DB_FALLBACK_ENV = 'NEXUS_EXPORT_TRACE_DB_FALLBACK'
_TDSQL_TIMEOUT_ENV = 'NEXUS_EXPORT_TDSQL_TIMEOUT_SECONDS'
_ITEM_DEADLINE_ENV = 'NEXUS_EXPORT_ITEM_DEADLINE_SECONDS'
_EXPORT_DEADLINE_ENV = 'NEXUS_EXPORT_DEADLINE_SECONDS'
_ROOT_SCAN_ENV = 'NEXUS_EXPORT_ROOT_SCAN'
_ROOT_SCAN_N_ENV = 'NEXUS_EXPORT_ROOT_SCAN_N'
_DEFAULT_ITEM_DEADLINE_SECONDS: float | None = None
_DEFAULT_EXPORT_DEADLINE_SECONDS: float | None = None
_DEFAULT_ROOT_SCAN_N = 128
_MAX_ROOT_SCAN_N = 512


def _optional_float_env(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {'0', 'off', 'none', 'disabled'}:
        return None
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _tdsql_timeout_seconds() -> float:
    value = _optional_float_env(_TDSQL_TIMEOUT_ENV, _TDSQL_TIMEOUT_SECONDS)
    return value if value is not None else _TDSQL_TIMEOUT_SECONDS


def _partition_probe_mode() -> str:
    value = os.environ.get(_PARTITION_PROBE_ENV, 'fail-open').strip().lower()
    return value if value in {'fail-open', 'fail-closed'} else 'fail-open'


def _db_fallback_mode() -> str:
    value = os.environ.get(_DB_FALLBACK_ENV, 'api').strip().lower()
    return value if value in {'api', 'fail'} else 'api'


def _item_deadline_seconds() -> float | None:
    return _optional_float_env(_ITEM_DEADLINE_ENV, _DEFAULT_ITEM_DEADLINE_SECONDS)


def _export_deadline_seconds() -> float | None:
    return _optional_float_env(_EXPORT_DEADLINE_ENV, _DEFAULT_EXPORT_DEADLINE_SECONDS)


def _root_scan_mode() -> str:
    value = os.environ.get(_ROOT_SCAN_ENV, 'reverse').strip().lower()
    return value if value in {'reverse', 'full'} else 'reverse'


def _root_scan_n() -> int:
    raw = os.environ.get(_ROOT_SCAN_N_ENV)
    try:
        value = int(raw) if raw else _DEFAULT_ROOT_SCAN_N
    except ValueError:
        value = _DEFAULT_ROOT_SCAN_N
    return max(1, min(value, _MAX_ROOT_SCAN_N))


_ROOT_SCAN_FALLBACK: Any = object()


def _database_code(error: BaseException | None) -> int | None:
    seen: set[int] = set()
    cause = error
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if cause.args and type(cause.args[0]) is int:
            return cause.args[0]
        cause = cause.__cause__
    return None


def _classify_database_error(
    error: BaseException | None,
    *,
    run_id: str,
    attempts: int,
    sql_kind: str = 'unknown',
) -> TraceDatabaseError:
    """Map a driver failure onto the fallback contract.

    The original exception is preserved as ``__cause__`` so ``_retryable`` can
    still inspect it, while the new type tells the per-item fallback whether an
    Application API read is a safe substitute.
    """
    code = _database_code(error)
    detail = type(error).__name__ if error is not None else 'unknown error'
    if code is not None:
        detail += f' (DB code {code})'
    message = (
        f'Trace Database query failed for {run_id} after {attempts} attempts: '
        f'{detail}; phase=execute, sql_kind={sql_kind}'
    )
    if isinstance(error, TraceDatabaseTimeout):
        cls = TraceDatabaseTimeout
    elif code in {1205, 1213, 1158, 1159, 1160, 1161}:
        cls: type[TraceDatabaseError] = TraceDatabaseThrottled
    elif code in {1317, 3024}:
        cls = TraceDatabaseInterrupted
    elif code in {1044, 1045}:
        cls = TraceDatabaseAuth
    elif code in {1054, 1146, 1176}:
        cls = TraceDatabaseSchema
    elif code in {2002, 2003, 2006, 2013, 1040, 1042, 1047}:
        cls = TraceDatabaseUnavailable
    elif isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        cls = TraceDatabaseTimeout
    elif isinstance(error, (ConnectionError, OSError)):
        cls = TraceDatabaseUnavailable
    else:
        cls = TraceDatabaseError
    return cls(message)


def _sql_kind(sql: str) -> str:
    head = sql.lstrip().split(None, 1)
    return head[0].upper() if head else 'unknown'


def normalize_status(value: str) -> str:
    try:
        return _STATUS_MAP[value]
    except KeyError as exc:
        raise ExportError(f'unsupported terminal status: {value}') from exc


def validate_trace_table(value: str) -> str:
    if PRINCIPAL_TRACE_TABLE_PATTERN.fullmatch(value) is None:
        raise ExportError(f'invalid Trace Database table: {value!r}')
    return value


def validate_batch_summary(summary: Mapping[str, Any]) -> int:
    if summary.get('generation_status') != 'complete' or summary.get('submission_closed') is not True:
        raise ExportError('Batch generation or submission is not complete')
    counts: dict[str, int] = {}
    for name in ('materialized_count', 'total_items', 'terminal_items'):
        value = summary.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ExportError(f'invalid Batch {name}')
        counts[name] = value
    if len(set(counts.values())) != 1:
        raise ExportError('Batch is not fully terminal')
    status_counts = summary.get('item_status_counts')
    if not isinstance(status_counts, Mapping):
        raise ExportError('Batch summary is missing item status counts')
    total_from_statuses = 0
    for status_name, raw_count in status_counts.items():
        if status_name not in _TERMINAL_ITEM_STATUSES:
            raise ExportError(f'Batch has non-terminal item status: {status_name}')
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
            raise ExportError(f'invalid Batch status count: {status_name}')
        total_from_statuses += raw_count
    if total_from_statuses != counts['total_items']:
        raise ExportError('Batch item status counts do not match total_items')
    return counts['total_items']


def validate_execution(execution: Mapping[str, Any], expected_execution_id: str, *, allow_partial: bool = False) -> str:
    if execution.get('execution_id') != expected_execution_id:
        raise ExportError(f'Execution identity mismatch: expected {expected_execution_id}')
    spec_type = execution.get('spec_type')
    if spec_type not in _SUPPORTED_SPEC_TYPES:
        raise ExportError(f'unsupported Execution spec_type: {spec_type}')
    status = execution.get('status')
    if not allow_partial and status not in _TERMINAL_EXECUTION_STATUSES:
        raise ExportError(f'Execution is not terminal: {status}')
    attempt_count = execution.get('attempt_count')
    if isinstance(attempt_count, bool) or not isinstance(attempt_count, int) or attempt_count < 0:
        raise ExportError('invalid Execution attempt_count')
    return str(spec_type)


def execution_snapshot(
    execution: Mapping[str, Any], *, expected_execution_id: str | None = None, allow_partial: bool = False
) -> dict[str, Any]:
    execution_id = execution.get('execution_id')
    if not isinstance(execution_id, str) or not execution_id:
        raise ExportError('invalid Execution execution_id')
    validate_execution(
        execution, execution_id if expected_execution_id is None else expected_execution_id, allow_partial=allow_partial
    )
    identity = {'execution_id': execution_id, 'spec_type': execution['spec_type']}
    if allow_partial:
        return identity
    return {
        **identity,
        'status': execution['status'],
        'attempt_count': execution['attempt_count'],
        'current_attempt_id': execution.get('current_attempt_id'),
        'result_artifact_id': execution.get('result_artifact_id'),
        'error_json': execution.get('error_json'),
    }


def batch_snapshot(summary: Mapping[str, Any]) -> dict[str, Any]:
    validate_batch_summary(summary)
    summary_version = summary.get('summary_version')
    if isinstance(summary_version, bool) or not isinstance(summary_version, int) or summary_version < 0:
        raise ExportError('invalid Batch summary_version')
    snapshot = {
        'generation_status': summary['generation_status'],
        'materialized_count': summary['materialized_count'],
        'total_items': summary['total_items'],
        'submission_closed': summary['submission_closed'],
    }
    raw_status_counts = summary['item_status_counts']
    assert isinstance(raw_status_counts, Mapping)
    return {
        **snapshot,
        'summary_version': summary_version,
        'terminal_items': summary['terminal_items'],
        'item_status_counts': dict(sorted(raw_status_counts.items())),
    }


def batch_metadata(summary: Mapping[str, Any]) -> dict[str, Any]:
    manifest = summary.get('task_list_manifest')
    if manifest is None:
        return {}
    if not isinstance(manifest, Mapping):
        raise ExportError('Batch summary has invalid task_list_manifest')
    value = manifest.get('batch_metadata')
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ExportError('Batch summary has invalid batch_metadata')
    return copy.deepcopy(dict(value))


def extract_run_output(event_data: Any) -> dict[str, Any] | None:
    if not isinstance(event_data, Mapping):
        return None
    result: dict[str, Any] | None = None
    events = event_data.get('events')
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes, bytearray)):
        return None
    for item in events:
        if not isinstance(item, Mapping) or item.get('name') != 'run.event':
            continue
        attributes = item.get('attributes')
        if not isinstance(attributes, Mapping) or attributes.get('event_type') != 'run.output':
            continue
        raw_event_data = attributes.get('event_data')
        if not isinstance(raw_event_data, Mapping):
            continue
        output = raw_event_data.get('output')
        result = dict(output) if isinstance(output, Mapping) else None
    return result


def _has_run_output(event_data: Mapping[str, Any]) -> bool:
    return any(
        isinstance(item, Mapping)
        and item.get('name') == 'run.event'
        and isinstance(item.get('attributes'), Mapping)
        and item['attributes'].get('event_type') == 'run.output'
        for item in event_data.get('events') or []
    )


def project_lifecycle_times(events: Sequence[Mapping[str, Any]]) -> tuple[str | None, str | None]:
    start_time: str | None = None
    end_time: str | None = None
    for event in events:
        if event.get('type') != 'execution.status.changed':
            continue
        payload = event.get('payload')
        created_at = event.get('created_at')
        if not isinstance(payload, Mapping) or not isinstance(created_at, str) or not created_at:
            continue
        status = payload.get('to')
        if start_time is None and status in {'starting', 'running'}:
            start_time = created_at
        if status in _TERMINAL_EXECUTION_STATUSES:
            end_time = created_at
    return start_time, end_time


@dataclass
class _RootScanState:
    """The latest ``run.output`` discovered while walking one run's root spans."""

    latest: Any = None
    latest_key: tuple[int, str, int] | None = None
    candidate_path: Path | None = None
    candidate_prepared: Any = None
    candidate_none_output: bool = False
    candidate_key: tuple[int, str, int] | None = None


class TdsqlTraceReader:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        table: str,
        parent_created_at: str | None = None,
        trace_backend: str | None = None,
        connect: Callable[..., Awaitable[Any]] | None = None,
        transient_errors: tuple[type[BaseException], ...] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        from nexus.runner.trace.tdsql.config import build_db_config

        self.config = build_db_config(host=host, port=port, user=user, password=password, db=database)
        self.table = validate_trace_table(table)
        # Names which backend this connection belongs to. The table name cannot: it is
        # derived from the owner principal, so two backends holding the same principal
        # publish the same name under different schemas.
        self.trace_backend = trace_backend
        # Required against a Boundless partitioned table: without it both queries below
        # scan every day partition. Left unset for classic TDSQL tables, which have no
        # such column. All run_ids exported together belong to one execution and
        # therefore share a single parent_created_at value.
        self.parent_created_at = (
            normalize_parent_created_at(parent_created_at) if parent_created_at is not None else None
        )
        explicit_transient_errors = transient_errors is not None
        if connect is None:
            from aiomysql import InterfaceError, OperationalError
            from aiomysql import connect as mysql_connect

            connect = mysql_connect
            transient_errors = (OperationalError, InterfaceError)
        self.connect = connect
        self.transient_errors = (asyncio.TimeoutError, TimeoutError, *(transient_errors or ()))
        # Default driver errors include authentication/schema failures; only a
        # code-less one is retried without a DB-code check. A caller that names
        # its own transient class opts into retrying it.
        self._explicit_transient_errors = explicit_transient_errors
        self.sleep = sleep
        self.semaphore = asyncio.Semaphore(_TDSQL_CONCURRENCY)
        # The schema is stable for the life of an export, but the resolver runs
        # once per record. Probe once and share the answer: a Batch with 10k
        # items must not issue 10k information_schema queries.
        self._partition_probe: bool | None = None
        self._partition_probe_failed = False
        self._partition_probe_lock = asyncio.Lock()

    async def has_partition_column_cached(self) -> bool | None:
        """Probe once; ``None`` means the probe itself failed.

        Callers decide whether an unknown dialect is fail-open (skip pruning,
        correctness is unchanged) or fail-closed (an explicit partition key was
        supplied, so silently dropping it would run a different query).
        """
        if self._partition_probe is not None:
            return self._partition_probe
        if self._partition_probe_failed:
            return None
        async with self._partition_probe_lock:
            if self._partition_probe is not None:
                return self._partition_probe
            if self._partition_probe_failed:
                return None
            try:
                value = await self.has_partition_column()
            except Exception as exc:
                logger.warning(
                    'Trace partition probe failed for %s (backend=%s): %s; partition pruning is disabled',
                    self.table,
                    self.trace_backend or '-',
                    type(exc).__name__,
                )
                self._partition_probe_failed = True
                return None
            self._partition_probe = bool(value)
            return self._partition_probe

    async def has_partition_column(self) -> bool:
        """Whether the target table is Boundless, i.e. carries the partition key column.

        The reader is given a table name, never a dialect, so the schema is the only
        authority on which one it is. The distinction matters because the predicate is not
        merely useless on a classic table -- the column does not exist there, so adding it
        fails the query with 1054 instead of widening the scan.
        """
        sql = (
            'SELECT 1\n'
            'FROM information_schema.COLUMNS\n'
            'WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s\n'
            'LIMIT 1'
        )

        async def perform() -> bool:
            connection = None
            try:
                connection = await self.connect(**self.config)
                rows = await _fetch_rows(
                    connection,
                    sql,
                    (self.config['db'], self.table, 'parent_created_at'),
                    timeout=_TDSQL_PROBE_TIMEOUT_SECONDS,
                )
                return bool(rows)
            finally:
                if connection is not None:
                    closed = connection.close()
                    if inspect.isawaitable(closed):
                        await closed

        # Bound the connect as well as the statement: a slow/unreachable
        # database must not delay the first record by the 90s data budget.
        return await asyncio.wait_for(perform(), timeout=_TDSQL_PROBE_TIMEOUT_SECONDS)

    async def read_many(
        self,
        run_ids: Sequence[str],
        *,
        parent_execution_id: str | None = None,
        include_timestamps: bool = True,
        scan_out: dict[str, Any] | None = None,
        spill_pool: ResultSpillPool | None = None,
    ) -> dict[str, TraceResult]:
        unique_ids = tuple(dict.fromkeys(run_ids))
        if any(not isinstance(run_id, str) or not run_id for run_id in unique_ids):
            raise ExportError('TDSQL run IDs must be non-empty strings')
        chunks = [
            unique_ids[index : index + _TDSQL_ID_CHUNK_SIZE]
            for index in range(0, len(unique_ids), _TDSQL_ID_CHUNK_SIZE)
        ]
        if not chunks:
            return {}
        chunk_results = await asyncio.gather(
            *(
                self._read_chunk_with_retry(
                    chunk,
                    parent_execution_id=parent_execution_id,
                    include_timestamps=include_timestamps,
                    scan_out=scan_out,
                    spill_pool=spill_pool,
                )
                for chunk in chunks
            )
        )
        results: dict[str, TraceResult] = {}
        for chunk_result in chunk_results:
            results.update(chunk_result)
        return results

    async def _read_chunk_with_retry(
        self,
        run_ids: tuple[str, ...],
        *,
        parent_execution_id: str | None,
        include_timestamps: bool,
        scan_out: dict[str, Any] | None = None,
        spill_pool: ResultSpillPool | None = None,
    ) -> dict[str, TraceResult]:
        last_error: BaseException | None = None
        async with self.semaphore:
            for attempt in range(_TDSQL_ATTEMPTS):
                try:
                    # The statement deadline belongs to execute(), not to the
                    # whole chunk: transferring and scanning a GB-scale body is
                    # client work and must not be cancelled as a DB timeout.
                    # A cancellation still closes the connection, so every
                    # attempt below opens a fresh one.
                    return await self._read_chunk(
                        run_ids,
                        parent_execution_id=parent_execution_id,
                        include_timestamps=include_timestamps,
                        scan_out=scan_out,
                        spill_pool=spill_pool,
                    )
                except Exception as exc:
                    retryable = _retryable(exc)
                    if not retryable and isinstance(exc, self.transient_errors):
                        retryable = self._explicit_transient_errors or _database_code(exc) is None
                    if not retryable:
                        if isinstance(exc, self.transient_errors) or _database_code(exc) is not None:
                            raise _classify_database_error(
                                exc,
                                run_id=run_ids[0],
                                attempts=attempt + 1,
                                sql_kind='SELECT',
                            ) from exc
                        raise
                    last_error = exc
                    if attempt + 1 == _TDSQL_ATTEMPTS:
                        break
                    await self.sleep((0.25 * (2**attempt)) + random.uniform(0, 0.1))
        raise _classify_database_error(
            last_error,
            run_id=run_ids[0],
            attempts=_TDSQL_ATTEMPTS,
            sql_kind='SELECT',
        ) from last_error

    async def _read_chunk(
        self,
        run_ids: tuple[str, ...],
        *,
        parent_execution_id: str | None,
        include_timestamps: bool,
        scan_out: dict[str, Any] | None = None,
        spill_pool: ResultSpillPool | None = None,
    ) -> dict[str, TraceResult]:
        placeholders = ', '.join(['%s'] * len(run_ids))
        partition_predicate = ' AND parent_created_at = %s' if self.parent_created_at is not None else ''
        partition_params: tuple[Any, ...] = (self.parent_created_at,) if self.parent_created_at is not None else ()
        # The run/record index cannot cover time_unix_nano. Batch queries can
        # use the existing parent cursor index without fetching every data row.
        index = ' FORCE INDEX (idx_traj_span_events_parent_cursor)' if parent_execution_id is not None else ''
        parent_predicate = 'parent_execution_id = %s AND ' if parent_execution_id is not None else ''
        parent_params = (parent_execution_id,) if parent_execution_id is not None else ()
        timestamp_sql = (
            'SELECT run_id, MIN(time_unix_nano), MAX(time_unix_nano)\n'
            f'FROM {self.table}{index}\n'
            f'WHERE {parent_predicate}run_id IN ({placeholders}){partition_predicate}\n'
            'GROUP BY run_id'
        )
        connection = None
        try:
            connection = await self.connect(**self.config)
            timestamp_rows = (
                await _fetch_rows(connection, timestamp_sql, (*parent_params, *run_ids, *partition_params))
                if include_timestamps
                else []
            )
            if include_timestamps and parent_execution_id is not None and not timestamp_rows:
                # Old rows can predate parent attribution. Preserve their timestamp
                # fallback without adding a parent predicate to result-body reads.
                timestamp_rows = await _fetch_rows(
                    connection,
                    timestamp_sql.replace(index, '').replace(parent_predicate, ''),
                    (*run_ids, *partition_params),
                )
            output = await self._read_root_output(connection, run_ids[0], scan_out=scan_out, spill_pool=spill_pool)
        finally:
            if connection is not None:
                closed = connection.close()
                if inspect.isawaitable(closed):
                    await closed

        requested = set(run_ids)
        values: dict[str, TraceResult] = {run_id: TraceResult() for run_id in run_ids}
        for row in timestamp_rows:
            if not isinstance(row, (tuple, list)) or len(row) != 3:
                raise ExportError('Trace Database returned an invalid timestamp row')
            run_id, raw_start, raw_end = row
            if run_id not in requested:
                raise ExportError('Trace Database returned an unexpected run_id')
            values[run_id] = TraceResult(
                start_time=_nanos_to_iso(raw_start),
                end_time=_nanos_to_iso(raw_end),
            )
        current = values[run_ids[0]]
        values[run_ids[0]] = TraceResult(output=output, start_time=current.start_time, end_time=current.end_time)
        return values

    async def _read_root_output(
        self,
        connection: Any,
        run_id: str,
        *,
        scan_out: dict[str, Any] | None = None,
        spill_pool: ResultSpillPool | None = None,
    ) -> Any:
        """Read the newest run.output, preferring a bounded reverse probe.

        Reverse mode reads only the newest window and then continues with the
        older records when the window's newest row has no output. That keeps
        the body count at most the full walk's count: the window is never read
        twice merely because a later status update carries no run.output. A
        non-monotonic window still falls back to the full walk.
        """
        if _root_scan_mode() == 'reverse':
            reversed_result = await self._read_root_output_reverse(
                connection, run_id, scan_out=scan_out, spill_pool=spill_pool
            )
            if reversed_result is not _ROOT_SCAN_FALLBACK:
                return reversed_result
        return await self._read_root_output_full(connection, run_id, scan_out=scan_out, spill_pool=spill_pool)

    async def _read_root_output_reverse(
        self,
        connection: Any,
        run_id: str,
        *,
        scan_out: dict[str, Any] | None = None,
        spill_pool: ResultSpillPool | None = None,
    ) -> Any:
        partition_predicate = ' AND parent_created_at = %s' if self.parent_created_at is not None else ''
        partition_params: tuple[Any, ...] = (self.parent_created_at,) if self.parent_created_at is not None else ()
        state = _RootScanState()
        after_span: str | None = None
        while True:
            conditions = f'run_id = %s{partition_predicate}'
            params: tuple[Any, ...] = (run_id, *partition_params)
            if after_span is not None:
                conditions += ' AND run_span_id > %s'
                params += (after_span,)
            spans = await _fetch_rows(
                connection,
                f'SELECT DISTINCT run_span_id FROM {self.table} WHERE {conditions} ORDER BY run_span_id LIMIT %s',
                (*params, _TDSQL_ROOT_PAGE_SIZE),
            )
            for span_row in spans:
                if len(span_row) != 1 or not isinstance(span_row[0], str) or not span_row[0]:
                    raise ExportError(f'Trace Database returned invalid root span for {run_id}')
                span_id = span_row[0]
                accepted = await self._reverse_span_output(
                    connection,
                    run_id,
                    span_id,
                    state,
                    scan_out=scan_out,
                    spill_pool=spill_pool,
                )
                if not accepted:
                    self._discard_root_state(state)
                    return _ROOT_SCAN_FALLBACK
            if len(spans) < _TDSQL_ROOT_PAGE_SIZE:
                return self._root_state_result(state, scan_out)
            cursor_span = spans[-1][0]
            if cursor_span == after_span:
                raise ExportError(f'Trace Database span cursor did not advance for {run_id}')
            after_span = cursor_span

    async def _reverse_span_output(
        self,
        connection: Any,
        run_id: str,
        span_id: str,
        state: _RootScanState,
        *,
        scan_out: dict[str, Any] | None,
        spill_pool: ResultSpillPool | None,
    ) -> bool:
        """Process the newest N root records of one span.

        Returns False only when the window is not monotonically ordered (so
        the max key cannot be assumed to be the last row). Otherwise it walks
        the records older than the window in ascending order, reusing the
        state from the window; this avoids reading the same bodies twice when
        the newest record is a status update without run.output.
        """
        limit = _root_scan_n()
        partition_predicate = ' AND parent_created_at = %s' if self.parent_created_at is not None else ''
        partition_params: tuple[Any, ...] = (self.parent_created_at,) if self.parent_created_at is not None else ()
        conditions = f'run_id = %s{partition_predicate} AND run_span_id = %s AND span_id = %s AND event_type = %s'
        rows = await _fetch_rows(
            connection,
            f'SELECT record_id, time_unix_nano FROM {self.table} '
            f'FORCE INDEX (idx_traj_span_events_span_id) WHERE {conditions} '
            'ORDER BY record_id DESC LIMIT %s',
            (run_id, *partition_params, span_id, span_id, 'update', limit),
        )
        if not rows:
            return True
        metadata: list[tuple[int, int]] = []
        for row in rows:
            if len(row) != 2 or type(row[0]) is not int or type(row[1]) is not int:
                raise TraceDataInvalid(f'Trace Database returned invalid root metadata for {run_id}')
            metadata.append((row[0], row[1]))
        # ORDER BY record_id is the SQL contract; sorting here also keeps the
        # monotonic check meaningful for test doubles and for a backend that
        # returns the window in an unexpected order.
        metadata.sort(key=lambda value: value[0])
        previous_record: int | None = None
        previous_time: int | None = None
        for record_id, timestamp in metadata:
            if previous_record is not None and record_id <= previous_record:
                return False
            if previous_time is not None and timestamp < previous_time:
                return False
            previous_record = record_id
            previous_time = timestamp
        # The newest row is the common case: a successful run's final root
        # update normally carries run.output. Read it first and stop; only a
        # status-only tail forces the older window rows to be read.
        last: bool | None = await self._process_root_row(
            connection,
            run_id,
            span_id,
            metadata[-1][0],
            metadata[-1][1],
            state,
            scan_out=scan_out,
            spill_pool=spill_pool,
        )
        if last is not False:
            # The newest row settles this span. Do not run a MAX aggregate here:
            # for a long history it would scan all older metadata even though
            # the newest run.output already answered the question.
            return True
        # Descend only until the first run.output. The window was checked to be
        # monotonic above, so rows below that key cannot supersede it.
        for record_id, timestamp in reversed(metadata[:-1]):
            result = await self._process_root_row(
                connection,
                run_id,
                span_id,
                record_id,
                timestamp,
                state,
                scan_out=scan_out,
                spill_pool=spill_pool,
            )
            if result is not False:
                break
        if len(metadata) < limit:
            return True
        await self._finish_reverse_span(
            connection,
            run_id,
            span_id,
            metadata[0][0],
            state,
            scan_out=scan_out,
            spill_pool=spill_pool,
        )
        return True

    async def _finish_reverse_span(
        self,
        connection: Any,
        run_id: str,
        span_id: str,
        before_record: int,
        state: _RootScanState,
        *,
        scan_out: dict[str, Any] | None,
        spill_pool: ResultSpillPool | None,
    ) -> None:
        candidate_key = self._output_key(state)
        if candidate_key is None:
            await self._scan_older_root_rows(
                connection,
                run_id,
                span_id,
                before_record,
                state,
                scan_out=scan_out,
                spill_pool=spill_pool,
            )
            return
        older_max_time = await self._older_max_time(connection, run_id, span_id, before_record)
        if older_max_time is None or older_max_time < candidate_key[0]:
            return
        await self._scan_older_root_rows(
            connection,
            run_id,
            span_id,
            before_record,
            state,
            scan_out=scan_out,
            spill_pool=spill_pool,
            minimum_time=candidate_key[0],
        )

    @staticmethod
    def _output_key(state: _RootScanState) -> tuple[int, str, int] | None:
        keys = [key for key in (state.latest_key, state.candidate_key) if key is not None]
        return max(keys) if keys else None

    async def _older_max_time(
        self,
        connection: Any,
        run_id: str,
        span_id: str,
        before_record: int,
    ) -> int | None:
        partition_predicate = ' AND parent_created_at = %s' if self.parent_created_at is not None else ''
        partition_params: tuple[Any, ...] = (self.parent_created_at,) if self.parent_created_at is not None else ()
        rows = await _fetch_rows(
            connection,
            f'SELECT MAX(time_unix_nano) FROM {self.table} '
            f'FORCE INDEX (idx_traj_span_events_span_id) '
            f'WHERE run_id = %s{partition_predicate} AND run_span_id = %s AND span_id = %s '
            "AND event_type = 'update' AND record_id < %s",
            (run_id, *partition_params, span_id, span_id, before_record),
        )
        if len(rows) != 1 or len(rows[0]) != 1:
            raise TraceDataInvalid(f'Trace Database returned invalid root aggregate for {run_id}')
        value = rows[0][0]
        if value is None:
            return None
        if type(value) is not int:
            raise TraceDataInvalid(f'Trace Database returned invalid root aggregate for {run_id}')
        return value

    async def _scan_older_root_rows(
        self,
        connection: Any,
        run_id: str,
        span_id: str,
        before_record: int,
        state: _RootScanState,
        *,
        scan_out: dict[str, Any] | None,
        spill_pool: ResultSpillPool | None,
        minimum_time: int | None = None,
    ) -> None:
        """Walk the records older than the reverse window in ascending order."""
        partition_predicate = ' AND parent_created_at = %s' if self.parent_created_at is not None else ''
        partition_params: tuple[Any, ...] = (self.parent_created_at,) if self.parent_created_at is not None else ()
        after_record: int | None = None
        while True:
            conditions = (
                f'run_id = %s{partition_predicate} AND run_span_id = %s AND span_id = %s AND event_type = %s '
                'AND record_id < %s'
            )
            params: tuple[Any, ...] = (run_id, *partition_params, span_id, span_id, 'update', before_record)
            if minimum_time is not None:
                conditions += ' AND time_unix_nano >= %s'
                params += (minimum_time,)
            if after_record is not None:
                conditions += ' AND record_id > %s'
                params += (after_record,)
            rows = await _fetch_rows(
                connection,
                f'SELECT record_id, time_unix_nano FROM {self.table} '
                f'FORCE INDEX (idx_traj_span_events_span_id) WHERE {conditions} '
                'ORDER BY record_id LIMIT %s',
                (*params, _TDSQL_ROOT_PAGE_SIZE),
            )
            for row in rows:
                if len(row) != 2 or type(row[0]) is not int or type(row[1]) is not int:
                    raise TraceDataInvalid(f'Trace Database returned invalid root metadata for {run_id}')
                record_id, timestamp = row
                await self._process_root_row(
                    connection,
                    run_id,
                    span_id,
                    record_id,
                    timestamp,
                    state,
                    scan_out=scan_out,
                    spill_pool=spill_pool,
                )
            if len(rows) < _TDSQL_ROOT_PAGE_SIZE:
                return
            cursor = rows[-1][0]
            if after_record is not None and cursor <= after_record:
                raise ExportError(f'Trace Database root cursor did not advance for {run_id}')
            after_record = cursor

    async def _process_root_row(
        self,
        connection: Any,
        run_id: str,
        span_id: str,
        record_id: int,
        timestamp: int,
        state: _RootScanState,
        *,
        scan_out: dict[str, Any] | None,
        spill_pool: ResultSpillPool | None,
    ) -> bool | None:
        """Read one root body and update the latest-output state.

        Returns True when the row carried run.output, False when it did not,
        and None when the row's key is older than the current candidate.
        """
        key = (timestamp, span_id, record_id)
        if state.latest_key is not None and key <= state.latest_key:
            return None
        if state.candidate_key is not None and key <= state.candidate_key:
            return None
        partition_predicate = ' AND parent_created_at = %s' if self.parent_created_at is not None else ''
        partition_params: tuple[Any, ...] = (self.parent_created_at,) if self.parent_created_at is not None else ()
        bodies = await _fetch_rows(
            connection,
            f'SELECT event_data FROM {self.table} '
            f'WHERE run_id = %s{partition_predicate} AND run_span_id = %s AND record_id = %s',
            (run_id, *partition_params, span_id, record_id),
        )
        if len(bodies) != 1:
            raise TraceDataInvalid(f'Trace Database root record disappeared for {run_id}')
        if scan_out is not None and spill_pool is not None:
            event_file = spill_pool.new_file()
            legacy_decode = False
            try:
                try:
                    found, output_span = await asyncio.to_thread(
                        self._locate_run_output_span, bodies[0][0], event_file.path
                    )
                except (JsonFallbackNeeded, ValueError):
                    legacy_decode = True
                except ExportError as exc:
                    raise TraceProcessingError(str(exc)) from exc
            except BaseException:
                event_file.discard()
                raise
            if legacy_decode:
                event_file.discard()
                try:
                    event_data = await asyncio.to_thread(_decode_event_data, bodies[0][0])
                except ExportError as exc:
                    raise TraceDataInvalid(str(exc)) from exc
                if _has_run_output(event_data):
                    if state.candidate_path is not None:
                        with contextlib.suppress(OSError):
                            Path(state.candidate_path).unlink(missing_ok=True)
                    state.candidate_path = None
                    state.candidate_prepared = None
                    state.candidate_none_output = False
                    state.latest = extract_run_output(event_data)
                    state.latest_key = key
                    return True
                return False
            if not found:
                event_file.discard()
                return False
            if output_span is None:
                event_file.discard()
                if state.candidate_path is not None:
                    with contextlib.suppress(OSError):
                        Path(state.candidate_path).unlink(missing_ok=True)
                state.candidate_path = None
                state.candidate_prepared = None
                state.candidate_none_output = True
                state.candidate_key = key
                return True
            try:
                out_file, prepared, legacy_dict = await asyncio.to_thread(
                    self._copy_span_with_scan, event_file.path, output_span, spill_pool
                )
            except ExportError as exc:
                event_file.discard()
                raise TraceProcessingError(str(exc)) from exc
            event_file.discard()
            if state.candidate_path is not None:
                with contextlib.suppress(OSError):
                    Path(state.candidate_path).unlink(missing_ok=True)
            if legacy_dict:
                state.candidate_path = None
                state.candidate_prepared = None

                def _load_span(path: Path) -> Any:
                    with path.open('rb') as f:
                        return json.load(f)

                try:
                    state.latest = await asyncio.to_thread(_load_span, out_file.path)
                except ValueError as exc:
                    out_file.discard()
                    raise TraceDataInvalid(f'malformed TDSQL run.output for {run_id}') from exc
                out_file.discard()
                state.candidate_none_output = False
                state.candidate_key = key
            else:
                state.candidate_path = out_file.path
                state.candidate_prepared = prepared
                state.candidate_none_output = False
                state.candidate_key = key
            return True
        try:
            event_data = await asyncio.to_thread(_decode_event_data, bodies[0][0])
        except ExportError as exc:
            raise TraceDataInvalid(str(exc)) from exc
        if _has_run_output(event_data):
            state.latest = extract_run_output(event_data)
            state.latest_key = key
            return True
        return False

    @staticmethod
    def _root_state_result(state: _RootScanState, scan_out: dict[str, Any] | None) -> Any:
        if state.candidate_none_output:
            return None
        if state.candidate_path is not None and scan_out is not None:
            scan_out['prepared'] = state.candidate_prepared
            scan_out['path'] = state.candidate_path
            scan_out['normalize'] = True  # column text is spaced + ensure_ascii
            return _STREAMING_PREPARED
        return state.latest

    @staticmethod
    def _discard_root_state(state: _RootScanState) -> None:
        if state.candidate_path is not None:
            with contextlib.suppress(OSError):
                Path(state.candidate_path).unlink(missing_ok=True)
        state.latest = None
        state.latest_key = None
        state.candidate_path = None
        state.candidate_prepared = None
        state.candidate_none_output = False
        state.candidate_key = None

    async def _read_root_output_full(
        self,
        connection: Any,
        run_id: str,
        *,
        scan_out: dict[str, Any] | None = None,
        spill_pool: ResultSpillPool | None = None,
    ) -> Any:
        # Discover attempts from the covering run/record index. Comparing the two
        # span columns in SQL forces a data-row lookup for every nested event.
        # A literal root span instead uses the span_id index and reads only roots.
        partition_predicate = ' AND parent_created_at = %s' if self.parent_created_at is not None else ''
        partition_params: tuple[Any, ...] = (self.parent_created_at,) if self.parent_created_at is not None else ()
        after_span: str | None = None
        latest: dict[str, Any] | None = None
        latest_key: tuple[int, str, int] | None = None
        # Byte-pipeline candidates: the newest record whose event_data carries
        # a run.output event, kept as a spill file holding the raw output JSON
        # plus the scan tee result. At most one candidate lives on disk.
        candidate_path: Path | None = None
        candidate_prepared: Any = None
        candidate_none_output = False
        candidate_key: tuple[int, str, int] | None = None
        while True:
            conditions = f'run_id = %s{partition_predicate}'
            params: tuple[Any, ...] = (run_id, *partition_params)
            if after_span is not None:
                conditions += ' AND run_span_id > %s'
                params += (after_span,)
            spans = await _fetch_rows(
                connection,
                f'SELECT DISTINCT run_span_id FROM {self.table} WHERE {conditions} ORDER BY run_span_id LIMIT %s',
                (*params, _TDSQL_ROOT_PAGE_SIZE),
            )
            for span_row in spans:
                if len(span_row) != 1 or not isinstance(span_row[0], str) or not span_row[0]:
                    raise ExportError(f'Trace Database returned invalid root span for {run_id}')
                span_id = span_row[0]
                after_record: int | None = None
                while True:
                    conditions = (
                        f'run_id = %s{partition_predicate} AND run_span_id = %s AND span_id = %s AND event_type = %s'
                    )
                    params = (run_id, *partition_params, span_id, span_id, 'update')
                    if after_record is not None:
                        conditions += ' AND record_id > %s'
                        params += (after_record,)
                    rows = await _fetch_rows(
                        connection,
                        f'SELECT record_id, time_unix_nano FROM {self.table} '
                        f'FORCE INDEX (idx_traj_span_events_span_id) WHERE {conditions} '
                        'ORDER BY record_id LIMIT %s',
                        (*params, _TDSQL_ROOT_PAGE_SIZE),
                    )
                    for row in rows:
                        if len(row) != 2 or type(row[0]) is not int or type(row[1]) is not int:
                            raise ExportError(f'Trace Database returned invalid root metadata for {run_id}')
                        record_id, timestamp = row
                        key = (timestamp, span_id, record_id)
                        if latest_key is not None and key <= latest_key:
                            continue
                        if candidate_key is not None and key <= candidate_key:
                            continue
                        bodies = await _fetch_rows(
                            connection,
                            f'SELECT event_data FROM {self.table} '
                            f'WHERE run_id = %s{partition_predicate} AND run_span_id = %s AND record_id = %s',
                            (run_id, *partition_params, span_id, record_id),
                        )
                        if len(bodies) != 1:
                            raise ExportError(f'Trace Database root record disappeared for {run_id}')
                        if scan_out is not None and spill_pool is not None:
                            event_file = spill_pool.new_file()
                            legacy_decode = False
                            try:
                                try:
                                    # Sync-heavy work (GB-scale column write +
                                    # scan) runs in a worker thread: blocking
                                    # the loop here starves a concurrent
                                    # record's streaming download (the server
                                    # kills readers that stall — the part43
                                    # failure). If this task is cancelled
                                    # mid-thread the thread runs to completion
                                    # on its own; a late-recreated orphan file
                                    # is removed by ResultSpillPool.sweep on
                                    # the next export.
                                    found, output_span = await asyncio.to_thread(
                                        self._locate_run_output_span, bodies[0][0], event_file.path
                                    )
                                except (JsonFallbackNeeded, ValueError):
                                    # Scanner-incompatible content (duplicate keys,
                                    # nesting depth, NaN): decode exactly like the
                                    # legacy path instead of failing the export.
                                    legacy_decode = True
                                except ExportError as exc:
                                    raise TraceProcessingError(str(exc)) from exc
                            except BaseException:
                                event_file.discard()
                                raise
                            if legacy_decode:
                                event_file.discard()
                                try:
                                    event_data = await asyncio.to_thread(_decode_event_data, bodies[0][0])
                                except ExportError as exc:
                                    raise TraceProcessingError(str(exc)) from exc
                                if _has_run_output(event_data):
                                    if candidate_path is not None:
                                        with contextlib.suppress(OSError):
                                            Path(candidate_path).unlink(missing_ok=True)
                                    candidate_path = None
                                    candidate_prepared = None
                                    candidate_none_output = False
                                    latest = extract_run_output(event_data)
                                    latest_key = key
                                del bodies, event_data
                            elif not found:
                                # No run.output event: the watermark does not
                                # advance (legacy latest_key only moves when a
                                # record carries run.output).
                                event_file.discard()
                            elif output_span is None:
                                # run.output present but its output is not an
                                # object: legacy records None at this key — drop
                                # any earlier candidate and let the API fallback in.
                                event_file.discard()
                                if candidate_path is not None:
                                    with contextlib.suppress(OSError):
                                        Path(candidate_path).unlink(missing_ok=True)
                                candidate_path = None
                                candidate_prepared = None
                                candidate_none_output = True
                                candidate_key = key
                            else:
                                try:
                                    out_file, prepared, legacy_dict = await asyncio.to_thread(
                                        self._copy_span_with_scan, event_file.path, output_span, spill_pool
                                    )
                                except ExportError as exc:
                                    event_file.discard()
                                    raise TraceProcessingError(str(exc)) from exc
                                event_file.discard()
                                if candidate_path is not None:
                                    with contextlib.suppress(OSError):
                                        Path(candidate_path).unlink(missing_ok=True)
                                if legacy_dict:
                                    # Scanner-incompatible output (duplicate keys
                                    # / nesting / NaN): parse the copied span with
                                    # the legacy json.loads semantics.
                                    candidate_path = None
                                    candidate_prepared = None

                                    def _load_span(path: Path) -> Any:
                                        with path.open('rb') as f:
                                            return json.load(f)

                                    try:
                                        latest = await asyncio.to_thread(_load_span, out_file.path)
                                    except ValueError as exc:
                                        out_file.discard()
                                        raise TraceDataInvalid(f'malformed TDSQL run.output for {run_id}') from exc
                                    out_file.discard()
                                    candidate_none_output = False
                                    candidate_key = key
                                else:
                                    candidate_path = out_file.path
                                    candidate_prepared = prepared
                                    candidate_none_output = False
                                    candidate_key = key
                        else:
                            try:
                                event_data = await asyncio.to_thread(_decode_event_data, bodies[0][0])
                            except ExportError as exc:
                                raise TraceProcessingError(str(exc)) from exc
                            if _has_run_output(event_data):
                                latest = extract_run_output(event_data)
                                latest_key = key
                            del bodies, event_data
                    if len(rows) < _TDSQL_ROOT_PAGE_SIZE:
                        break
                    cursor = rows[-1][0]
                    if after_record is not None and cursor <= after_record:
                        raise ExportError(f'Trace Database root cursor did not advance for {run_id}')
                    after_record = cursor
            if len(spans) < _TDSQL_ROOT_PAGE_SIZE:
                if candidate_none_output:
                    return None
                if candidate_path is not None and scan_out is not None:
                    scan_out['prepared'] = candidate_prepared
                    scan_out['path'] = candidate_path
                    scan_out['normalize'] = True  # column text is spaced + ensure_ascii
                    return _STREAMING_PREPARED
                return latest
            cursor_span = spans[-1][0]
            if cursor_span == after_span:
                raise ExportError(f'Trace Database span cursor did not advance for {run_id}')
            after_span = cursor_span

    def _scan_event_window(self, path: Path, start: int, end: int) -> JsonFieldScanner:
        """Validate and index a JSON object stored as a byte window in a spill file."""
        scanner = JsonFieldScanner(root_string_length_only=True, metadata_limit=_RESULT_MEMBER_PARSE_LIMIT)
        remaining = end - start
        with path.open('rb') as stream:
            stream.seek(start)
            while remaining > 0:
                block = stream.read(min(remaining, _RESULT_SCAN_CHUNK_BYTES))
                if not block:
                    raise ExportError('Trace Database event_data ended unexpectedly')
                remaining -= len(block)
                scanner.feed(block)
        scanner.finish()
        return scanner

    @staticmethod
    def _span_holds_object(path: Path, span: tuple[int, int]) -> bool:
        with path.open('rb') as stream:
            stream.seek(span[0])
            return stream.read(1) == b'{'

    def _locate_run_output_span(self, event_data_text: str, event_path: Path) -> tuple[bool, tuple[int, int] | None]:
        """Stream event_data text to ``event_path`` and locate the last
        run.output event's ``output`` span, mirroring ``extract_run_output``:
        the last array element with name 'run.event', attributes.event_type
        'run.output' and an object attributes.event_data wins; its output
        must be an object, otherwise the caller records a None output."""
        found = False
        output_span: tuple[int, int] | None = None
        with event_path.open('wb') as dst:
            for i in range(0, len(event_data_text), _RESULT_SCAN_CHUNK_BYTES):
                dst.write(event_data_text[i : i + _RESULT_SCAN_CHUNK_BYTES].encode('utf-8'))
        size = event_path.stat().st_size
        root = self._scan_event_window(event_path, 0, size)
        events_span = root.spans.get('events')
        if events_span is None:
            return found, output_span
        first = events_span[0] + 1
        remaining = events_span[1] - first

        def chunks() -> Iterator[bytes]:
            nonlocal remaining
            with event_path.open('rb') as stream:
                stream.seek(first)
                while remaining > 0:
                    block = stream.read(min(remaining, _RESULT_SCAN_CHUNK_BYTES))
                    if not block:
                        break
                    remaining -= len(block)
                    yield block

        for elem_start, elem_end in iter_array_element_spans(chunks(), first):
            element = self._scan_event_window(event_path, elem_start, elem_end)
            if element.scalars.get('name') != 'run.event':
                continue
            # Nested scanner spans are relative to their window; translate each
            # level back to absolute event_path offsets.
            attributes_span = element.spans.get('attributes')
            if attributes_span is None:
                continue
            attributes_abs = (
                elem_start + attributes_span[0],
                elem_start + attributes_span[1],
            )
            if not self._span_holds_object(event_path, attributes_abs):
                continue
            attributes = self._scan_event_window(event_path, *attributes_abs)
            if attributes.scalars.get('event_type') != 'run.output':
                continue
            found = True
            event_data_span = attributes.spans.get('event_data')
            if event_data_span is None:
                output_span = None
                continue
            event_data_abs = (
                attributes_abs[0] + event_data_span[0],
                attributes_abs[0] + event_data_span[1],
            )
            if not self._span_holds_object(event_path, event_data_abs):
                output_span = None
                continue
            event_data = self._scan_event_window(event_path, *event_data_abs)
            span = event_data.spans.get('output')
            if span is not None:
                output_abs = (event_data_abs[0] + span[0], event_data_abs[0] + span[1])
                output_span = output_abs if self._span_holds_object(event_path, output_abs) else None
            else:
                output_span = None
        return found, output_span

    def _copy_span_with_scan(
        self, source: Path, span: tuple[int, int], pool: ResultSpillPool
    ) -> tuple[ResultSpillFile, _PreparedBody | None, bool]:
        """Copy a byte span into a fresh spill file while scanning it (tee).

        Returns ``(out_file, prepared, legacy_dict)``. ``legacy_dict`` is True
        when the span must go through the legacy full parse (scanner fallback
        for duplicate keys/nesting, or a hard scanner error such as NaN);
        ``out_file`` then holds the raw copied bytes for that parse (None when
        the copy itself failed and nothing usable was written).
        """
        out_file = pool.new_file()
        scan = _ScanSlot()
        try:
            with source.open('rb') as src, out_file.path.open('wb') as dst:
                src.seek(span[0])
                remaining = span[1] - span[0]
                while remaining > 0:
                    block = src.read(min(remaining, _RESULT_SCAN_CHUNK_BYTES))
                    if not block:
                        raise _SpanTruncated('Trace Database event_data ended unexpectedly')
                    dst.write(block)
                    scan.feed(block)
                    remaining -= len(block)
            prepared = scan.finish()
        except (JsonFallbackNeeded, ValueError, ExportError) as exc:
            # Scanner fallback (duplicate keys / nesting) or a hard scanner
            # error (NaN): hand the raw bytes to the legacy json.loads parse
            # instead of failing the export or emitting divergent bytes.
            # Truncated input stays a hard error (legacy json.loads fails on
            # it too).
            with contextlib.suppress(Exception):
                scan.abort()
            if isinstance(exc, _SpanTruncated):
                out_file.discard()
                raise ExportError(str(exc)) from exc
            # Semantic errors keep the raw bytes: the legacy json.loads parse
            # below still needs them.
            return out_file, None, True
        if scan.fallback:
            return out_file, None, True
        return out_file, prepared, False


async def _fetch_rows(
    connection: Any,
    sql: str,
    params: tuple[Any, ...],
    *,
    timeout: float | None = None,
) -> list[tuple[Any, ...]]:
    effective_timeout = _tdsql_timeout_seconds() if timeout is None else timeout
    async with connection.cursor() as cursor:
        try:
            await asyncio.wait_for(cursor.execute(sql, params), timeout=effective_timeout)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise TraceDatabaseTimeout(
                f'Trace Database {_sql_kind(sql)} did not finish within {effective_timeout}s; phase=execute'
            ) from exc
        rows = cursor.fetchall()
        if inspect.isawaitable(rows):
            rows = await rows
        return list(rows)


def _nanos_to_iso(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    try:
        return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc).isoformat()
    except (OSError, OverflowError, ValueError) as exc:
        raise ExportError('Trace Database returned an invalid timestamp') from exc


def _project_cache_task_ids(
    *,
    output: Mapping[str, Any] | None,
    metadata: Any,
    run_id: str,
) -> Any:
    """Copy recorded IDs without inferring client identity or applying client transforms."""
    if output is None:
        return metadata
    fields = {
        name: output[name]
        for name in ('input_cache_task_id', 'input_cache_task_ids', 'cache_task_id', 'cache_task_ids')
        if output.get(name) not in (None, '', {})
    }
    if not fields:
        return metadata
    if metadata is not None and not isinstance(metadata, Mapping):
        logger.warning('Cache task ID export: run=%s reason=invalid_metadata', run_id)
        return metadata
    projected = dict(metadata) if metadata is not None else {}
    raw_nexus = projected.get('_nexus')
    if raw_nexus is not None and not isinstance(raw_nexus, Mapping):
        logger.warning('Cache task ID export: run=%s reason=invalid_nexus_metadata', run_id)
        return metadata
    nexus = dict(raw_nexus) if raw_nexus is not None else {}
    execution_id = nexus.get('child_execution_id') or run_id
    for name, value in fields.items():
        if name.endswith('_ids'):
            valid = isinstance(value, Mapping) and all(
                isinstance(key, str) and key and isinstance(item, str) and item for key, item in value.items()
            )
        else:
            valid = isinstance(value, str)
        if not valid:
            logger.warning(
                'Cache task ID export: execution=%s field=%s reason=invalid_cache_task_id', execution_id, name
            )
            continue
        if name not in nexus:
            nexus[name] = copy.deepcopy(value)
        elif isinstance(value, Mapping) and isinstance(nexus[name], Mapping):
            merged = dict(nexus[name])
            for key, item in value.items():
                if key not in merged:
                    merged[key] = item
                elif merged[key] != item:
                    logger.warning(
                        'Cache task ID export: execution=%s field=%s client=%s reason=metadata_conflict',
                        execution_id,
                        name,
                        key,
                    )
            nexus[name] = merged
        elif nexus[name] != value:
            logger.warning('Cache task ID export: execution=%s field=%s reason=metadata_conflict', execution_id, name)
    if nexus:
        projected['_nexus'] = nexus
        return projected
    return metadata


def make_jsonl_record(
    *,
    run_id: str,
    distributed_status: str,
    start_time: str | None,
    end_time: str | None,
    retry_count: int,
    fallback_error: Any,
    output: Mapping[str, Any] | None,
    metadata: Any,
) -> dict[str, Any]:
    status = normalize_status(distributed_status)
    if not isinstance(run_id, str) or not run_id:
        raise ExportError('run_id must be a non-empty string')
    if isinstance(retry_count, bool) or not isinstance(retry_count, int) or retry_count < 0:
        raise ExportError(f'invalid retry_count for {run_id}')
    if output is None:
        if distributed_status == 'succeeded':
            raise ExportError(f'successful execution has no RunOutput: {run_id}')
        error = fallback_error
        task_output = None
        agent_output = None
        processor_results = None
    else:
        success = output.get('success')
        if not isinstance(success, bool):
            raise ExportError(f'invalid RunOutput success field: {run_id}')
        if distributed_status == 'succeeded' and not success:
            raise ExportError(f'successful execution has failed RunOutput: {run_id}')
        output_error = output.get('error')
        error = output_error if distributed_status == 'succeeded' or output_error is not None else fallback_error
        task_output = output.get('task_output')
        agent_output = output.get('agent_output')
        processor_results = output.get('processor_results')
    record = {
        'run_id': run_id,
        'status': status,
        'start_time': start_time,
        'end_time': end_time,
        'retry_count': retry_count,
        'error': error,
        'task_output': task_output,
        'agent_output': agent_output,
        'processor_results': processor_results,
        'metadata': _project_cache_task_ids(output=output, metadata=metadata, run_id=run_id),
    }
    if output is not None and output.get('extra'):
        record['extra'] = output['extra']
    return record


async def _probe_partitioned(trace_reader: Any, cache: dict[Any, Any] | None = None) -> bool | None:
    cached = getattr(trace_reader, 'has_partition_column_cached', None)
    if callable(cached):
        return await cached()
    probe = getattr(trace_reader, 'has_partition_column', None)
    if not callable(probe):
        return None
    cache_key = ('probe', id(trace_reader))
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    lock: asyncio.Lock | None = None
    if cache is not None:
        lock_key = ('probe-lock', id(trace_reader))
        lock = cache.get(lock_key)
        if lock is None:
            lock = asyncio.Lock()
            cache[lock_key] = lock

    async def run_probe() -> bool | None:
        try:
            return bool(await probe())
        except Exception as exc:
            logger.warning(
                'Trace partition probe failed for %s (backend=%s): %s; partition pruning is disabled',
                getattr(trace_reader, 'table', '?'),
                getattr(trace_reader, 'trace_backend', None) or '-',
                type(exc).__name__,
            )
            return None

    if lock is not None:
        async with lock:
            if cache is not None and cache_key in cache:
                return cache[cache_key]
            value = await run_probe()
            if cache is not None:
                cache[cache_key] = value
            return value
    return await run_probe()


def _matches_result_status(status: str, result_status: ResultStatus) -> bool:
    return status in _TERMINAL_ITEM_STATUSES and (
        result_status == 'all' or (status == 'succeeded') == (result_status == 'success')
    )


@dataclass
class _SingleExport:
    execution_id: str | None
    run_id: str
    status: str
    retry_count: int
    context: Mapping[str, Any]
    parent_execution_id: str | None = None
    item_id: str | None = None
    metadata: Any = None
    error: Any = None


@dataclass(frozen=True)
class _RecordPlan:
    """Everything needed to materialize one output row later.

    The depth-2 pipeline splits each record into a download phase (possibly
    spilled to disk by the prefetch slot) and an assembly phase; the plan is
    the token handed between them.
    """

    single: _SingleExport
    trace_reader: Any | None = None


@dataclass
class _ExportPlan:
    singles: dict[tuple[str, ...], _SingleExport] = field(default_factory=dict)
    snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    batches: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)


async def _expand_executions(
    api: CloudApi,
    ids: Sequence[str],
    *,
    allow_partial: bool,
    result_status: str | None = None,
    item_ids: set[str] | None = None,
) -> _ExportPlan:
    """Select and deduplicate records without reading result bodies or touching the DB."""
    selection = resolve_result_status(allow_partial=allow_partial, result_status=result_status)
    plan = _ExportPlan()
    contexts = {}
    for target_id in dict.fromkeys(ids):
        execution = await api.get_execution(target_id)
        is_batch = execution.get('spec_type') == 'nexus_batch'
        if item_ids is not None and not is_batch:
            raise ExportError('--item-ids requires a Batch --execution-id')
        plan.snapshots[target_id] = execution_snapshot(
            execution, expected_execution_id=target_id, allow_partial=allow_partial and is_batch
        )
        contexts[target_id] = {
            key: execution.get(key)
            for key in (
                'execution_id',
                'spec_type',
                'parent_execution_id',
                'created_at',
                'runtime_info',
                'current_assignment',
            )
        }
        if is_batch:
            summary = await api.get_batch_summary(target_id)
            plan.batches[target_id] = None if allow_partial else batch_snapshot(summary)
            plan.metadata[target_id] = batch_metadata(summary)
    if allow_partial and not plan.batches:
        raise ExportError('--allow-partial is supported only for Batch results')

    # Preserve Batch metadata when the same child is also explicitly selected.
    for parent_id, snapshot in plan.batches.items():
        async for item in _batch_items(
            api,
            parent_id,
            allow_partial=allow_partial,
            expected_count=snapshot['total_items'] if snapshot is not None else None,
            result_status=selection,
            item_ids=item_ids,
        ):
            single = _batch_export(contexts[parent_id], item)
            key = (
                ('execution', single.execution_id) if single.execution_id else ('item', parent_id, str(single.item_id))
            )
            plan.singles.setdefault(key, single)
            if single.execution_id is not None and selection != 'all':
                child_id = single.execution_id
                if child_id not in plan.snapshots:
                    execution = await api.get_execution(child_id)
                    try:
                        plan.snapshots[child_id] = execution_snapshot(execution, expected_execution_id=child_id)
                    except ExportError as exc:
                        raise ExportError(f'Batch item {single.item_id}, child {child_id}: {exc}') from exc
                    contexts[child_id] = {
                        key: execution.get(key)
                        for key in (
                            'execution_id',
                            'spec_type',
                            'parent_execution_id',
                            'created_at',
                            'runtime_info',
                            'current_assignment',
                        )
                    }
                child = plan.snapshots[child_id]
                if child['spec_type'] != 'nexus_single':
                    # Batch children are created from a Single template; a
                    # non-Single child would carry a different scope and the
                    # parent-created_at partition below would not describe it.
                    raise ExportError(f'Batch child must be a Single execution: {child_id}')
                if not _matches_result_status(child['status'], selection):
                    raise ExportError(
                        f'Batch item {single.item_id}, child {child_id}: Execution status {child["status"]} '
                        f'does not match --result-status {selection}'
                    )
                if contexts[child_id].get('parent_execution_id') not in {None, parent_id}:
                    raise ExportError(f'child execution belongs to another Batch: {child_id}')
                single.error = child.get('error_json')
    for target_id in dict.fromkeys(ids):
        snapshot = plan.snapshots[target_id]
        if target_id not in plan.batches:
            plan.singles.setdefault(
                ('execution', target_id),
                _SingleExport(
                    execution_id=target_id,
                    run_id=target_id,
                    status=snapshot['status'],
                    retry_count=max(snapshot['attempt_count'] - 1, 0),
                    context=contexts[target_id],
                    parent_execution_id=contexts[target_id]['parent_execution_id'],
                    error=snapshot.get('error_json'),
                ),
            )
    return plan


async def export_execution(
    *,
    api: CloudApi,
    execution_id: str | Sequence[str],
    output: Path,
    force: bool,
    trace_reader: Any | None = None,
    allow_partial: bool = False,
    result_status: str | None = None,
    item_ids: Sequence[str] | None = None,
    strict_validate: bool = False,
    trace_db_fallback: str | None = None,
) -> None:
    """Expand, read and publish all results through the same Single pipeline."""
    selection = resolve_result_status(allow_partial=allow_partial, result_status=result_status)
    ids = [execution_id] if isinstance(execution_id, str) else execution_id
    if (
        isinstance(ids, bytes)
        or not isinstance(ids, Sequence)
        or not ids
        or any(not isinstance(value, str) or not value.strip() for value in ids)
    ):
        raise ExportError('execution_id must contain non-empty execution IDs')
    selected_items = None
    if item_ids is not None:
        if len(set(ids)) != 1:
            raise ExportError('--item-ids requires exactly one Batch --execution-id')
        if (
            isinstance(item_ids, (str, bytes))
            or not isinstance(item_ids, Sequence)
            or any(not isinstance(value, str) for value in item_ids)
        ):
            raise ExportError('--item-ids must contain non-empty item IDs')
        selected_items = {item.strip() for value in item_ids for item in value.split(',')}
        if not selected_items or '' in selected_items:
            raise ExportError('--item-ids must contain non-empty item IDs')
    plan = await _expand_executions(
        api, ids, allow_partial=allow_partial, result_status=result_status, item_ids=selected_items
    )

    metadata = {}
    if any(plan.metadata.values()):
        metadata = next(iter(plan.metadata.values())) if len(plan.metadata) == 1 else {'batches': plan.metadata}
    writer = AtomicJsonlOutput(output, force=force)
    await writer.open()
    metadata_output = writer.output.with_name(f'{writer.output.name}.batch_metadata.json')
    metadata_writer = AtomicJsonOutput(metadata_output, force=force) if metadata else None
    # The public pipeline accepts CloudApi-compatible test doubles and legacy
    # embedders that only implement get_result(). Keep those on the established
    # path; disk prefetch is enabled for the concrete client that guarantees the
    # streaming-download contract.
    supports_result_stream = isinstance(api, CloudApi) and callable(getattr(api, 'get_result_to_file', None))
    spill_pool = ResultSpillPool(writer.output) if supports_result_stream else None
    disk_prefetch = spill_pool if trace_reader is None else None
    fallback_spill = spill_pool if trace_reader is not None else None
    record_plans = [_RecordPlan(single=single, trace_reader=trace_reader) for single in plan.singles.values()]
    partition_cache: dict[Any, Any] = {}
    export_deadline_seconds = _export_deadline_seconds()
    export_deadline = time.monotonic() + export_deadline_seconds if export_deadline_seconds is not None else None
    try:
        if metadata_writer is not None:
            await metadata_writer.prepare(metadata)
        written = await _write_records(
            writer,
            api,
            record_plans,
            spill=disk_prefetch,
            fallback_spill=fallback_spill,
            partition_cache=partition_cache,
            export_deadline=export_deadline,
            trace_db_fallback=trace_db_fallback,
        )
        await writer.close_and_validate(expected_count=written, require_sorted=False, strict=strict_validate)
        if allow_partial and selection in {'failed', 'all'}:
            # The write-time index is dropped for very large exports (and never used
            # in strict mode); both fall back to the parse-per-line filter.
            if strict_validate or not writer.index_available:
                written = await writer.filter_records(lambda row: row['status'] == 'success' or _has_artifacts(row))
            else:
                written = await writer.filter_records_by_index(
                    lambda entry: entry.status == 'success' or entry.has_artifacts
                )
        for target_id, snapshot in plan.snapshots.items():
            execution = await api.get_execution(target_id)
            if (
                execution_snapshot(
                    execution,
                    expected_execution_id=target_id,
                    allow_partial=allow_partial and target_id in plan.batches,
                )
                != snapshot
            ):
                raise ExportError(f'Execution changed during export: {target_id}')
        for parent_id, snapshot in plan.batches.items():
            summary = await api.get_batch_summary(parent_id)
            if snapshot is not None and batch_snapshot(summary) != snapshot:
                raise ExportError(f'Batch changed during export: {parent_id}')
            if batch_metadata(summary) != plan.metadata[parent_id]:
                raise ExportError(f'Batch metadata changed during export: {parent_id}')
        if metadata_writer is not None:
            await metadata_writer.publish()
        elif force:
            await asyncio.to_thread(_unlink_and_fsync, metadata_output)
        await writer.publish()
        logger.info('Results export: written_records=%s', written)
    except BaseException:
        with contextlib.suppress(Exception):
            await writer.abort()
        if metadata_writer is not None:
            with contextlib.suppress(Exception):
                await metadata_writer.abort()
        raise


async def _execution_trace_reader(
    api: CloudApi,
    reader: Any | None,
    execution: Mapping[str, Any],
    *,
    partition_cache: dict[Any, Any] | None = None,
) -> Any | None:
    """Share DB settings while keeping each target's partition value independent.

    The dialect is resolved once per Execution and shared by all of its records.
    Batch parents have no sink envelope, so their schema is probed (cached on the
    reader); Single/Live targets prefer the author's envelope and only probe
    when the envelope does not name the selected table. An explicit
    ``--parent-created-at`` is validated against the schema before use: a key on
    a classic table would make every query fail with 1054.
    """
    if reader is None or getattr(reader, 'table', None) is None:
        return reader
    cache = partition_cache if partition_cache is not None else {}
    spec_type = execution.get('spec_type')
    table = reader.table
    cache_key = ('partition', execution.get('execution_id'), spec_type)
    if cache_key in cache:
        partition = cache[cache_key]
        if partition is None:
            return reader
        scoped_reader = copy.copy(reader)
        scoped_reader.parent_created_at = partition
        return scoped_reader

    explicit = getattr(reader, 'parent_created_at', None)
    backend = getattr(reader, 'trace_backend', None)

    async def fetch_execution(execution_id: str) -> Mapping[str, Any]:
        fetch_key = ('execution', execution_id)
        if fetch_key in cache:
            return cache[fetch_key]
        value = await api.get_execution(execution_id)
        cache[fetch_key] = value
        return value

    resolved = await resolve_trace_read_scope(
        execution,
        str(spec_type),
        table,
        backend=backend,
        override_partition_key=explicit,
        probe=lambda: _probe_partitioned(reader, cache),
        fetch_execution=fetch_execution,
        fail_closed_on_unknown=_partition_probe_mode() == 'fail-closed' or explicit is not None,
    )
    partition = resolved.partition_key
    cache[cache_key] = partition
    if partition is None:
        return reader
    scoped_reader = copy.copy(reader)
    scoped_reader.parent_created_at = partition
    return scoped_reader


async def _single_record(
    *,
    api: CloudApi,
    single: _SingleExport,
    trace_reader: Any | None,
    prefetched: Any = _MISSING,
    fallback_spill: ResultSpillPool | None = None,
    scan_factory: Any | None = None,
    partition_cache: dict[Any, Any] | None = None,
    deadline_seconds: float | None = None,
    trace_db_fallback: str | None = None,
) -> dict[str, Any] | _StreamingRow:
    trace = TraceResult()
    error = single.error
    scan_out: dict[str, Any] | None = (
        {} if (prefetched is _MISSING and fallback_spill is not None and scan_factory is not None) else None
    )
    target = (
        f'item {single.item_id}, child {single.execution_id}' if single.item_id else f'execution {single.execution_id}'
    )
    timeout = asyncio.timeout(deadline_seconds) if deadline_seconds is not None else contextlib.nullcontext()
    try:
        async with timeout:
            if single.execution_id is not None:
                try:
                    lifecycle_times = project_lifecycle_times(await api.get_control_events(single.execution_id))
                    reader = await _execution_trace_reader(
                        api,
                        trace_reader,
                        single.context,
                        partition_cache=partition_cache,
                    )
                    try:
                        trace = await _read_result(
                            api=api,
                            execution_id=single.execution_id,
                            prefetched=prefetched,
                            trace_reader=reader,
                            parent_execution_id=single.parent_execution_id,
                            fallback_spill=fallback_spill,
                            scan_factory=scan_factory,
                            scan_out=scan_out,
                            lifecycle_times=lifecycle_times,
                        )
                    except TraceDatabaseError as db_exc:
                        fallback = trace_db_fallback if trace_db_fallback is not None else _db_fallback_mode()
                        transient = isinstance(
                            db_exc,
                            (
                                TraceDatabaseTimeout,
                                TraceDatabaseUnavailable,
                                TraceDatabaseThrottled,
                                TraceDatabaseInterrupted,
                            ),
                        )
                        if fallback != 'api' or not transient:
                            raise
                        logger.warning(
                            'Trace Database read failed for %s (%s); source=api-fallback',
                            target,
                            type(db_exc).__name__,
                        )
                        if scan_out is not None:
                            stale = scan_out.get('path')
                            if stale is not None:
                                with contextlib.suppress(OSError):
                                    Path(stale).unlink(missing_ok=True)
                            scan_out.clear()
                        try:
                            trace = await _read_result(
                                api=api,
                                execution_id=single.execution_id,
                                prefetched=_MISSING,
                                trace_reader=None,
                                parent_execution_id=single.parent_execution_id,
                                fallback_spill=fallback_spill,
                                scan_factory=scan_factory,
                                scan_out=scan_out,
                                lifecycle_times=lifecycle_times,
                            )
                        except Exception as api_exc:
                            raise ExportError(f'{db_exc}; API fallback failed: {api_exc}') from api_exc
                    if scan_out is not None and scan_out.get('prepared') is not None:
                        # Streaming: TDSQL had no output, so the API body was scanned
                        # during download; assemble the row without building the
                        # object graph. Enrichment (error fallback / control events)
                        # is handled inside _streaming_record.
                        return await _streaming_record(
                            api=api,
                            single=single,
                            body_path=scan_out['path'],
                            prepared=scan_out['prepared'],
                            times=(trace.start_time, trace.end_time),
                            source_normalized=bool(scan_out.get('normalize')),
                        )
                    if (
                        single.item_id is not None
                        and error is None
                        and (
                            trace.output is None or (single.status != 'succeeded' and trace.output.get('error') is None)
                        )
                    ):
                        execution = await api.get_execution(single.execution_id)
                        if execution.get('parent_execution_id') not in {None, single.parent_execution_id}:
                            raise ExportError(f'child execution belongs to another Batch: {single.execution_id}')
                        error = execution.get('error_json')
                except ExportError as exc:
                    raise ExportError(f'Result export failed for {target}: {exc}') from exc
    except TimeoutError as exc:
        if deadline_seconds is None:
            raise TraceProcessingError(f'Result export timed out for {target}; phase=process') from exc
        raise TraceProcessingTimeout(
            f'Result export deadline exceeded for {target} after {deadline_seconds}s; phase=process'
        ) from exc
    return make_jsonl_record(
        run_id=single.run_id,
        distributed_status=single.status,
        start_time=trace.start_time,
        end_time=trace.end_time,
        retry_count=single.retry_count,
        fallback_error=error,
        output=trace.output,
        metadata=single.metadata,
    )


def _prefetched_result(
    execution_id: str,
    raw: bytes | bytearray | memoryview | None,
) -> dict[str, Any] | None:
    """Validate a spilled response body exactly like CloudApi.get_result does."""
    if raw is None:
        return None
    try:
        payload = _loads_json_bytes(raw)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ExportError(f'API returned invalid JSON for result of {execution_id}') from exc
    if not isinstance(payload, dict):
        raise ExportError(f'API returned a non-object response for result of {execution_id}')
    if not payload:
        return None
    if not isinstance(payload.get('success'), bool):
        raise ExportError(f'invalid RunOutput returned for {execution_id}')
    return payload


# ---------------------------------------------------------------------------
# Streaming transform (方案 D): rows assembled by transcoding member spans of
# the spilled body instead of parsing it into a dict. On by default;
# NEXUS_EXPORT_RESULT_TRANSCODE=0 falls back to the legacy orjson path (both
# produce byte-identical rows).
# ---------------------------------------------------------------------------

_RESULT_TRANSCODE_ENV = 'NEXUS_EXPORT_RESULT_TRANSCODE'
_RESULT_SCAN_CHUNK_BYTES = 1 << 20
_RESULT_MEMBER_PARSE_LIMIT = 1 << 20
_RESULT_SCAN_MAX_DEPTH = 1000
_UNSET: Any = object()
_STREAMING_PREPARED: Any = object()


class _SpanTruncated(ExportError):  # noqa: N818 - reads as a sentence
    """The copied span was shorter than the located range (IO truncation)."""


_JSONL_BODY_FIELDS = ('task_output', 'agent_output', 'processor_results')


def _result_transcode_enabled() -> bool:
    """Streaming transform is on by default; set the env var to 0 to disable."""
    return os.environ.get(_RESULT_TRANSCODE_ENV) != '0'


class _ScanSlot:
    """Pass-1 scan running in a worker thread while the body downloads (tee).

    Download chunks are fed through a bounded queue (~8 MiB, backpressure keeps
    memory flat), so validation/spans are ready when the download completes and
    the separate re-read pass disappears. Scanner/`json.loads` semantic
    divergences set ``fallback`` (the record goes to the legacy orjson path)
    instead of failing; NaN/Infinity and grammar errors are hard errors surfaced
    at the next ``feed`` call.
    """

    def __init__(self) -> None:
        self.scanner = JsonFieldScanner(
            reject_non_finite=True,
            root_string_length_only=True,
            metadata_limit=_RESULT_MEMBER_PARSE_LIMIT,
            max_depth=_RESULT_SCAN_MAX_DEPTH,
        )
        self.fallback = False
        self._queue: queue.Queue = queue.Queue(maxsize=8)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name='nexus-result-scan')
        self._thread.start()

    def _run(self) -> None:
        while True:
            chunk = self._queue.get()
            try:
                if chunk is None:
                    return
                self.scanner.feed(chunk)
            except JsonFallbackNeeded:
                self.fallback = True
                self._drain()
                return
            except BaseException as exc:
                self._error = exc
                self._drain()
                return
            finally:
                self._queue.task_done()

    def _drain(self) -> None:
        while True:
            if self._queue.get() is None:
                return

    def reset(self) -> None:
        """Fresh scanner for a download retry; the old thread drains its leftovers."""
        self._queue.put(None)
        self._thread.join()
        self.scanner = JsonFieldScanner(
            reject_non_finite=True,
            root_string_length_only=True,
            metadata_limit=_RESULT_MEMBER_PARSE_LIMIT,
            max_depth=_RESULT_SCAN_MAX_DEPTH,
        )
        self.fallback = False
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True, name='nexus-result-scan')
        self._thread.start()

    def feed(self, chunk: bytes) -> None:
        if self._error is not None:
            raise ExportError(f'invalid RunOutput body: {self._error}') from self._error
        self._queue.put(chunk)

    def finish(self) -> _PreparedBody:
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise ExportError(f'invalid RunOutput body: {self._error}') from self._error
        return _PreparedBody(
            spans=dict(self.scanner.spans),
            kinds=dict(self.scanner.kinds),
            scalars=dict(self.scanner.scalars),
            empty=set(self.scanner.empty_members),
        )

    def abort(self) -> None:
        """Stop the scan thread without a result (404/missing body)."""
        self._queue.put(None)
        self._thread.join()


@dataclass
class _PreparedBody:
    """Pass-1 scan outcome for one spilled RunOutput body."""

    spans: dict[str, tuple[int, int]]
    kinds: dict[str, str]
    scalars: dict[str, Any]
    empty: set[str]

    def span(self, name: str) -> tuple[int, int] | None:
        return self.spans.get(name)

    def kind(self, name: str) -> str | None:
        return self.kinds.get(name)

    def scalar(self, name: str) -> Any:
        # Long (truncated) strings have no captured value; callers only rely on
        # null-ness/emptiness for those, which span/kind/empty already encode.
        return self.scalars.get(name, _UNSET)

    def value_is_none(self, name: str) -> bool:
        if name not in self.spans:
            return True
        if self.kind(name) != 'scalar':
            return False
        return self.scalar(name) is None

    def has_artifacts_member(self, name: str) -> bool:
        if name not in self.spans:
            return False
        if self.kind(name) != 'scalar':
            return name not in self.empty
        value = self.scalar(name)
        if value is _UNSET:  # long string → non-empty artifact
            return True
        return value is not None and value != ''


@dataclass
class _StreamingRow:
    """A row assembled by transcoding member spans of a spilled body."""

    run_id: str
    status: str
    retry_count: int
    start_time: str | None
    end_time: str | None
    error_from_body: bool
    error_span: tuple[int, int] | None
    error_fallback: Any
    metadata: Any
    extra_span: tuple[int, int] | None
    body_path: Path
    prepared: _PreparedBody
    # True when the spilled body was written in the stdlib-default column
    # format (spaced separators + ensure_ascii), e.g. extracted TDSQL
    # event_data: members are transcoded with normalize=True so escapes
    # decode and spacing is canonical instead of doubled.
    source_normalized: bool = False

    def discard(self) -> None:
        with contextlib.suppress(OSError):
            self.body_path.unlink(missing_ok=True)

    def iter_chunks(self) -> Iterator[bytes]:
        from nexus.utils.json_stream import iter_json_bytes

        prepared = self.prepared

        def literal(value: Any) -> Iterator[bytes]:
            yield from iter_json_bytes(value, sort_keys=False, allow_nan=False, separators=(', ', ': '))

        def member(name: str, span: tuple[int, int] | None) -> Iterator[bytes]:
            if span is None:
                yield from literal(None)
                return

            def blocks() -> Iterator[bytes]:
                with self.body_path.open('rb') as stream:
                    stream.seek(span[0])
                    remaining = span[1] - span[0]
                    while remaining > 0:
                        block = stream.read(min(remaining, _RESULT_SCAN_CHUNK_BYTES))
                        if not block:
                            raise ExportError(f'truncated result body member {name}')
                        remaining -= len(block)
                        yield block

            # One transcoder for the whole member: its state machine must
            # survive chunk boundaries. A fresh transcoder per block would
            # resume in the outside-string state whenever a boundary falls
            # inside a string and insert spaces after its ':' / ',' bytes.
            yield from iter_spaced_json(blocks(), normalize=self.source_normalized)

        def error_chunks() -> Iterator[bytes]:
            if self.error_from_body:
                yield from member('error', self.error_span)
            else:
                yield from literal(self.error_fallback)

        fields: list[tuple[bytes, Iterator[bytes]]] = [
            (b'"run_id"', literal(self.run_id)),
            (b'"status"', literal(self.status)),
            (b'"start_time"', literal(self.start_time)),
            (b'"end_time"', literal(self.end_time)),
            (b'"retry_count"', literal(self.retry_count)),
            (b'"error"', error_chunks()),
            (b'"task_output"', member('task_output', prepared.span('task_output'))),
            (b'"agent_output"', member('agent_output', prepared.span('agent_output'))),
            (b'"processor_results"', member('processor_results', prepared.span('processor_results'))),
            (b'"metadata"', literal(self.metadata)),
        ]
        yield b'{'
        first = True
        for key, value_iter in fields:
            if first:
                first = False
            else:
                yield b', '
            yield key + b': '
            yield from value_iter
        if self.extra_span is not None:
            yield b', "extra": '
            span = self.extra_span

            def extra_blocks() -> Iterator[bytes]:
                with self.body_path.open('rb') as stream:
                    stream.seek(span[0])
                    remaining = span[1] - span[0]
                    while remaining > 0:
                        block = stream.read(min(remaining, _RESULT_SCAN_CHUNK_BYTES))
                        if not block:
                            raise ExportError('truncated result body member extra')
                        remaining -= len(block)
                        yield block

            yield from iter_spaced_json(extra_blocks(), normalize=self.source_normalized)
        yield b'}'


async def _streaming_record(
    *,
    api: CloudApi,
    single: _SingleExport,
    body_path: Path,
    prepared: _PreparedBody,
    times: tuple[str | None, str | None] | None = None,
    source_normalized: bool = False,
) -> _StreamingRow:
    """Assemble the row for a spilled body without materializing its values.

    ``prepared`` comes from the scan that ran during the download (tee); the
    legacy orjson path handled any body the scanner diverged on.
    """
    execution_id = single.execution_id
    assert execution_id is not None
    if not isinstance(single.run_id, str) or not single.run_id:
        raise ExportError('run_id must be a non-empty string')
    if isinstance(single.retry_count, bool) or not isinstance(single.retry_count, int) or single.retry_count < 0:
        raise ExportError(f'invalid retry_count for {single.run_id}')
    status = normalize_status(single.status)
    if times is not None:
        start_time, end_time = times
    else:
        start_time, end_time = project_lifecycle_times(await api.get_control_events(execution_id))

    success = prepared.scalar('success')
    if not isinstance(success, bool):
        raise ExportError(f'invalid RunOutput success field: {single.run_id}')
    if single.status == 'succeeded' and not success:
        raise ExportError(f'successful execution has failed RunOutput: {single.run_id}')

    body_error_is_none = prepared.value_is_none('error')
    error_from_body = single.status == 'succeeded' or not body_error_is_none

    if single.item_id is not None and single.error is None and (single.status != 'succeeded' and body_error_is_none):
        execution = await api.get_execution(execution_id)
        if execution.get('parent_execution_id') not in {None, single.parent_execution_id}:
            raise ExportError(f'child execution belongs to another Batch: {execution_id}')
        fallback_error = execution.get('error_json')
    else:
        fallback_error = single.error

    extra_span = None
    extra = prepared.span('extra')
    if extra is not None:
        if prepared.kind('extra') == 'scalar':
            value = prepared.scalar('extra')
            if value is _UNSET or value:  # truncated long string → truthy
                extra_span = extra
        elif 'extra' not in prepared.empty:  # non-empty container → truthy
            extra_span = extra

    return _StreamingRow(
        run_id=single.run_id,
        status=status,
        retry_count=single.retry_count,
        start_time=start_time,
        end_time=end_time,
        error_from_body=error_from_body,
        error_span=prepared.span('error') if error_from_body else None,
        error_fallback=fallback_error,
        metadata=_project_cache_task_ids_streaming(prepared, body_path, metadata=single.metadata, run_id=single.run_id),
        extra_span=extra_span,
        body_path=body_path,
        prepared=prepared,
        source_normalized=source_normalized,
    )


def _project_cache_task_ids_streaming(
    prepared: _PreparedBody,
    body_path: Path,
    *,
    metadata: Any,
    run_id: str,
) -> Any:
    names = ('input_cache_task_id', 'input_cache_task_ids', 'cache_task_id', 'cache_task_ids')
    if all(name not in prepared.spans for name in names):
        return metadata
    values: dict[str, Any] = {}
    for name in names:
        span = prepared.span(name)
        if span is None:
            continue
        if prepared.kind(name) == 'scalar':
            value = prepared.scalar(name)
            if value is _UNSET:
                # Oversized string member: parse from its exact span. Memory is
                # bounded by the member itself — never worse than the legacy
                # full-parse path this replaces, and byte-identical output.
                with body_path.open('rb') as stream:
                    stream.seek(span[0])
                    value = json.loads(stream.read(span[1] - span[0]))
            values[name] = value
            continue
        with body_path.open('rb') as stream:
            stream.seek(span[0])
            values[name] = json.loads(stream.read(span[1] - span[0]))
    return _project_cache_task_ids(output=values, metadata=metadata, run_id=run_id)


async def _read_result(
    *,
    api: CloudApi,
    execution_id: str,
    trace_reader: Any | None,
    parent_execution_id: str | None,
    prefetched: Any = _MISSING,
    fallback_spill: ResultSpillPool | None = None,
    scan_factory: Any | None = None,
    scan_out: dict[str, Any] | None = None,
    lifecycle_times: tuple[str | None, str | None] | None = None,
) -> TraceResult:
    if lifecycle_times is None:
        start_time, end_time = project_lifecycle_times(await api.get_control_events(execution_id))
    else:
        start_time, end_time = lifecycle_times
    result = None
    if trace_reader is not None:
        if isinstance(trace_reader, TdsqlTraceReader):
            traces = await trace_reader.read_many(
                [execution_id],
                parent_execution_id=parent_execution_id,
                include_timestamps=start_time is None or end_time is None,
                scan_out=scan_out,
                spill_pool=fallback_spill,
            )
        else:
            traces = await trace_reader.read_many([execution_id])
        if not isinstance(traces, Mapping) or any(key != execution_id for key in traces):
            raise ExportError('Trace Database returned unexpected child results')
        trace = traces.get(execution_id, TraceResult())
        if not isinstance(trace, TraceResult):
            raise ExportError('Trace Database returned an invalid result')
        result = trace.output
        start_time = start_time or trace.start_time
        end_time = end_time or trace.end_time
    if result is None:
        if prefetched is _MISSING:
            if fallback_spill is None:
                result = await api.get_result(execution_id)
            else:
                spill_file = fallback_spill.new_file()
                try:
                    await spill_file.download(api, execution_id, scan_factory=scan_factory)
                    prepared = spill_file.prepared
                    if spill_file.scan_fallback:
                        prepared = None
                    if prepared is not None and scan_out is not None:
                        # Streaming: hand the spill file to the caller (discarded
                        # after the row is written) and skip the orjson parse.
                        result = _STREAMING_PREPARED
                        scan_out['prepared'] = prepared
                        scan_out['path'] = spill_file.path
                    else:
                        with spill_file.open_view() as body:
                            result = _prefetched_result(execution_id, body)
                        spill_file.discard()
                except BaseException:
                    spill_file.discard()
                    raise
        else:
            result = _prefetched_result(execution_id, prefetched)
    return TraceResult(output=result, start_time=start_time, end_time=end_time)


async def _write_records(
    writer: AtomicJsonlOutput,
    api: CloudApi,
    plans: Sequence[_RecordPlan] | Iterator[_RecordPlan],
    *,
    spill: ResultSpillPool | None = None,
    fallback_spill: ResultSpillPool | None = None,
    partition_cache: dict[Any, Any] | None = None,
    export_deadline: float | None = None,
    trace_db_fallback: str | None = None,
) -> int:
    """Materialize every plan and write the rows in plan order.

    With a spill pool, at most one record is materialized in RAM at a time:
    the head slot fetches/parses/writes while the next responses stream to
    disk (the configured concurrency worth of entries ahead), so fetch latency overlaps the
    CPU-bound write without a second body ever sitting in memory. Without a
    pool (Trace-Database path), falls back to the in-RAM pipeline: the bodies
    come from the DB driver and spilling does not apply.
    """
    plan_iter = iter(plans)
    transcode = _result_transcode_enabled()
    scan_factory = _ScanSlot if transcode else None

    def next_plan() -> _RecordPlan | None:
        return next(plan_iter, None)

    def check_export_deadline() -> None:
        if export_deadline is not None and time.monotonic() >= export_deadline:
            configured = _export_deadline_seconds()
            detail = f' after {configured:g}s' if configured is not None else ''
            raise ExportDeadlineExceeded(f'Export deadline exceeded{detail}')

    def single_kwargs(plan: _RecordPlan) -> dict[str, Any]:
        try:
            check_export_deadline()
        except ExportDeadlineExceeded as exc:
            target = (
                f'item {plan.single.item_id}, child {plan.single.execution_id}'
                if plan.single.item_id
                else f'execution {plan.single.execution_id}'
            )
            raise ExportDeadlineExceeded(f'{exc}; item={target}, phase=queue') from None
        deadline_seconds = _item_deadline_seconds()
        if export_deadline is not None:
            remaining = max(0.001, export_deadline - time.monotonic())
            deadline_seconds = remaining if deadline_seconds is None else min(deadline_seconds, remaining)
        kwargs: dict[str, Any] = {
            'api': api,
            'single': plan.single,
            'trace_reader': plan.trace_reader,
            'scan_factory': scan_factory,
            'deadline_seconds': deadline_seconds,
        }
        if partition_cache is not None:
            kwargs['partition_cache'] = partition_cache
        if trace_db_fallback is not None:
            kwargs['trace_db_fallback'] = trace_db_fallback
        if fallback_spill is not None:
            kwargs['fallback_spill'] = fallback_spill
        return kwargs

    def start_in_ram(plan: _RecordPlan) -> asyncio.Task[dict[str, Any] | _StreamingRow]:
        return asyncio.create_task(_single_record(**single_kwargs(plan)))

    if spill is None:
        written = 0
        pending: deque[asyncio.Task[dict[str, Any] | _StreamingRow]] = deque()

        def enqueue() -> None:
            plan = next_plan()
            if plan is not None:
                pending.append(start_in_ram(plan))

        try:
            for _ in range(_in_ram_concurrency()):
                enqueue()
            while pending:
                task = pending.popleft()
                record = await task
                check_export_deadline()
                if isinstance(record, _StreamingRow):
                    try:
                        await writer.write_row_chunks(
                            record.iter_chunks(),
                            run_id=record.run_id,
                            status=record.status,
                            retry_count=record.retry_count,
                            has_artifacts=any(
                                record.prepared.has_artifacts_member(name) for name in _JSONL_BODY_FIELDS
                            ),
                        )
                    finally:
                        record.discard()
                else:
                    await writer.write_records([record])
                written += 1
                del record, task
                enqueue()
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if fallback_spill is not None:
                fallback_spill.cleanup()
        return written

    # Spill pipeline: download every record to disk first, then materialize the
    # head record from its mapped file. This keeps even the first record out of
    # the old get_result/read-bytes path while retaining one response download
    # ahead of the record currently being parsed and written.
    concurrency = _result_concurrency()
    spill_ahead = max(concurrency - 1, 1)
    entries: deque[tuple[_RecordPlan, ResultSpillFile | None, asyncio.Task]] = deque()

    def fill() -> None:
        while len(entries) < spill_ahead:
            check_export_deadline()
            plan = next_plan()
            if plan is None:
                return
            if plan.single.execution_id is None:
                # No result body to download (known-failure rows): tiny record,
                # safe to assemble in RAM inside the prefetch queue.
                entries.append((plan, None, start_in_ram(plan)))
            else:
                file = spill.new_file()
                entries.append((
                    plan,
                    file,
                    asyncio.create_task(file.download(api, plan.single.execution_id, scan_factory=scan_factory)),
                ))

    def hydrate(
        entry: tuple[_RecordPlan, ResultSpillFile | None, asyncio.Task],
    ) -> asyncio.Task[dict[str, Any] | _StreamingRow]:
        plan, file, task = entry
        if file is None:
            return task

        async def run() -> dict[str, Any] | _StreamingRow:
            try:
                await task
            except ExportError as exc:
                target = (
                    f'item {plan.single.item_id}, child {plan.single.execution_id}'
                    if plan.single.item_id
                    else f'execution {plan.single.execution_id}'
                )
                raise ExportError(f'Result export failed for {target}: {exc}') from exc
            if (
                transcode
                and plan.trace_reader is None
                and not file.missing
                and file.prepared is not None
                and not file.scan_fallback
            ):
                try:
                    row = await _streaming_record(
                        api=api, single=plan.single, body_path=file.path, prepared=file.prepared
                    )
                except BaseException:
                    file.discard()
                    raise
                else:
                    # Discard is deferred until the row has been written: pass 2
                    # reads member spans from the spill file.
                    return row
                # Record-level fallback: scanner/json.loads divergence — the
                # orjson dict path below produces identical bytes.
            try:
                with file.open_view() as body:
                    kwargs = single_kwargs(plan)
                    kwargs['prefetched'] = body
                    return await _single_record(**kwargs)
            finally:
                file.discard()

        return asyncio.create_task(run())

    fill()
    if not entries:
        return 0
    written = 0
    head: asyncio.Task[dict[str, Any] | _StreamingRow] | None = None
    try:
        while entries:
            head = hydrate(entries.popleft())
            row = await head
            check_export_deadline()
            if isinstance(row, _StreamingRow):
                try:
                    await writer.write_row_chunks(
                        row.iter_chunks(),
                        run_id=row.run_id,
                        status=row.status,
                        retry_count=row.retry_count,
                        has_artifacts=any(row.prepared.has_artifacts_member(name) for name in _JSONL_BODY_FIELDS),
                    )
                finally:
                    row.discard()
            else:
                await writer.write_records([row])
            written += 1
            del row
            fill()
    finally:
        for _, _, task in entries:
            task.cancel()
        if head is not None:
            head.cancel()
        tasks = [task for *_, task in entries]
        if head is not None:
            tasks.append(head)
        await asyncio.gather(*tasks, return_exceptions=True)
        spill.cleanup()
        if fallback_spill is not None:
            fallback_spill.cleanup()
    return written


async def _batch_items(
    api: CloudApi,
    parent_id: str,
    *,
    allow_partial: bool,
    expected_count: int | None,
    result_status: ResultStatus = 'all',
    item_ids: set[str] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Read only item metadata; partial selects terminal items at page-read time."""
    cursor = previous = None
    seen = skipped = 0
    missing = set(item_ids or ())
    while True:
        page = await api.get_batch_items(parent_id, cursor)
        items = page.get('items')
        if not isinstance(items, list):
            raise ExportError('invalid Batch items response')
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get('item_id'), str) or not item['item_id']:
                raise ExportError('Batch item has invalid item_id')
            item_id = item['item_id']
            if previous is not None and item_id <= previous:
                raise ExportError('Batch item order is not strictly increasing')
            previous = item_id
            seen += 1
            missing.discard(item_id)
            if item.get('status') not in _TERMINAL_ITEM_STATUSES:
                if not allow_partial:
                    raise ExportError(f'Batch item is not terminal: {item_id}')
                skipped += 1
                continue
            if item_ids is not None and item_id not in item_ids:
                continue
            if not _matches_result_status(item['status'], result_status):
                continue
            yield item
        cursor = page.get('next_cursor')
        if cursor is None:
            break
        if not items or not isinstance(cursor, str) or cursor != previous:
            raise ExportError('Batch item cursor did not advance')
    if expected_count is not None and seen != expected_count:
        raise ExportError(f'Batch item count mismatch: expected {expected_count}, got {seen}')
    if missing:
        raise ExportError(f'Item IDs not found in Batch {parent_id}: {", ".join(sorted(missing))}')
    if allow_partial:
        logger.warning(
            'Partial Batch export: read %s terminal items; skipped %s non-terminal items',
            seen - skipped,
            skipped,
        )


def _batch_export(parent: Mapping[str, Any], item: Mapping[str, Any]) -> _SingleExport:
    """Keep Batch projection data with the child while sharing Single result reads."""
    item_id = str(item['item_id'])
    child_id = item.get('current_child_execution_id')
    if child_id is not None and (not isinstance(child_id, str) or not child_id):
        raise ExportError(f'Batch item has invalid current_child_execution_id: {item_id}')
    retry_count = item.get('current_attempt_no')
    if isinstance(retry_count, bool) or not isinstance(retry_count, int) or retry_count < 0:
        raise ExportError(f'Batch item has invalid current_attempt_no: {item_id}')
    parent_id = str(parent['execution_id'])
    run_id, metadata = _project_batch_identity(
        item_id=item_id,
        metadata=item.get('metadata'),
        parent_execution_id=parent_id,
        child_execution_id=child_id,
    )
    status = str(item['status'])
    return _SingleExport(
        execution_id=child_id,
        run_id=run_id,
        status=status,
        retry_count=retry_count,
        context=parent,
        parent_execution_id=parent_id,
        item_id=item_id,
        metadata=metadata,
        error={
            'type': 'BatchItemNoChildExecution',
            'message': f'Batch item reached {status} before a child execution was created',
        }
        if child_id is None
        else None,
    )


def _project_batch_identity(
    *,
    item_id: str,
    metadata: Any,
    parent_execution_id: str,
    child_execution_id: str | None,
) -> tuple[str, dict[str, Any]]:
    if metadata is None:
        projected: dict[str, Any] = {}
    elif isinstance(metadata, Mapping):
        projected = copy.deepcopy(dict(metadata))
    else:
        raise ExportError(f'Batch item has invalid metadata: {item_id}')

    raw_nexus = projected.get('_nexus')
    if raw_nexus is None:
        nexus: dict[str, Any] = {}
    elif isinstance(raw_nexus, Mapping):
        nexus = dict(raw_nexus)
    else:
        raise ExportError(f'Batch item has invalid metadata _nexus: {item_id}')

    original_run_id = nexus.get('original_run_id')
    run_id = original_run_id if isinstance(original_run_id, str) and original_run_id else item_id
    nexus.update({
        'original_run_id': run_id,
        'parent_execution_id': parent_execution_id,
        'child_execution_id': child_execution_id,
    })
    projected['_nexus'] = nexus
    return run_id, projected


def _decode_event_data(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (str, bytes, bytearray)):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ExportError('malformed TDSQL event_data') from exc
        if isinstance(decoded, dict):
            return decoded
    raise ExportError('malformed TDSQL event_data')
