from __future__ import annotations

import asyncio
import atexit
import contextlib
import hashlib
import inspect
import logging
import threading
import time

from collections import deque
from dataclasses import dataclass
from typing import Any

from nexus.runner.trace.tdsql._driver import DriverWarning, InterfaceError, OperationalError
from nexus.runner.trace.tdsql._driver import connect as mysql_connect
from nexus.runner.trace.tdsql.config import resolve_max_queue_bytes, resolve_max_queue_items, resolve_writer_shards
from nexus.runner.trace.tdsql.schema import TDSQL_DIALECT, schema_statements, span_event_insert_sql, validate_dialect
from nexus.utils.platform_metrics import measure_database, measure_operation, record


logger = logging.getLogger(__name__)

# A DB write taking at least this long is logged as slow.
SLOW_OPERATION_WARN_SECONDS = 5.0

_BATCH_SIZE = 100
_BATCH_WINDOW_SECONDS = 0.1


_STANDALONE_PAYLOAD_BYTES = 256 * 1024


_RECONNECT_ERROR_CODES = frozenset({
    2003,  # CR_CONN_HOST_ERROR
    2006,  # CR_SERVER_GONE_ERROR
    2013,  # CR_SERVER_LOST
    2055,  # CR_SERVER_LOST_EXTENDED
})
_MAX_RECONNECT_ATTEMPTS = 8
_RECONNECT_BACKOFF_BASE_SECONDS = 0.5
_RECONNECT_BACKOFF_MAX_SECONDS = 8.0

# The Boundless layout carries no partition for tomorrow: SET INTERVAL extends the RANGE
# on demand, so the first rows written after midnight make several nodes add the same
# partition at once and the losers get 1517. The write is sound -- the partition exists by
# the time we look again -- and the statement is an upsert, so retrying in place is safe.
# Reconnecting is not: the connection is healthy, and dropping the batch loses a run's
# trace every single day, which is how this surfaced rather than as a rare race.
_PARTITION_RACE_ERROR_CODES = frozenset({
    1517,  # ER_SAME_NAME_PARTITION
    8502,  # DROP PARTITION DDL refreshes metadata for 30-60s; connection stays healthy, so retry in place.
})
_MAX_PARTITION_RACE_ATTEMPTS = 8
_PARTITION_RACE_BACKOFF_SECONDS = 60.0 / (2**_MAX_PARTITION_RACE_ATTEMPTS - 1)

_HEALTH_RECOVER_INTERVAL_SECONDS = 5.0

_WRITER_START_TIMEOUT_SECONDS = 30.0
_START_WAIT_TIMEOUT_SECONDS = _WRITER_START_TIMEOUT_SECONDS + 5.0

# wait_until_settled() polls this often while a close() waits for its rows.
_SETTLE_POLL_INTERVAL_SECONDS = 0.05


_BACKPRESSURE_REPORT_EVERY = 500


@dataclass(slots=True)
class _Write:
    sql: str
    params: tuple[Any, ...]
    standalone: bool = False
    seq: int = 0
    run_span_id: str = ''
    principal_scope_id: str | None = None
    parent_execution_id: str | None = None
    nbytes: int = 0


def _write_bytes(params: tuple[Any, ...]) -> int:
    """Approximate serialized payload size of a row (sum of its string params)."""
    return sum(len(value) for value in params if isinstance(value, str))


def _is_oversized(params: tuple[Any, ...]) -> bool:
    """True when the largest string param is big enough to warrant a solo flush."""
    return any(isinstance(value, str) and len(value) >= _STANDALONE_PAYLOAD_BYTES for value in params)


# Max chars of a truncated content sample included in a dropped-write log line.
_DROPPED_SAMPLE_CHARS = 120

_MONITOR_FIELD_DEFAULTS: dict[str, Any] = {
    'rows': 0,  # rows involved in the operation (written for writes, returned for reads)
    'payload_bytes': 0,  # serialized size of the batch/row
    'elapsed_ms': 0.0,  # DB round-trip latency
    'run_id': '',
    'run_span_id': '',
    'trajectory_group_id': '',
    'principal_scope_id': '',
    'parent_execution_id': '',
    'storage_scope': '',
    'execution_id': '',
    'assignment_id': '',
    'attempt_id': '',
}


def report_monitor(
    monitor_type: str, message: str, *, level: int = logging.INFO, exc_info: bool = False, **fields: Any
) -> None:
    extra = {'monitor_type': monitor_type, **_MONITOR_FIELD_DEFAULTS, **fields}
    logger.log(level, '%s', message, exc_info=exc_info, extra=extra)


_DROP_LOG_MIN_INTERVAL_SECONDS = 5.0
_drop_log_lock = threading.Lock()
_drop_log_last_at: dict[str, float] = {}
_drop_log_suppressed: dict[str, int] = {}


def log_dropped_write(monitor_type: str, params: tuple[Any, ...], *, exc_info: bool = False, **fields: Any) -> None:

    record('data_integrity_errors', source='trace_db', reason='event_dropped')
    now = time.monotonic()
    with _drop_log_lock:
        last = _drop_log_last_at.get(monitor_type)
        if last is not None and now - last < _DROP_LOG_MIN_INTERVAL_SECONDS:
            _drop_log_suppressed[monitor_type] = _drop_log_suppressed.get(monitor_type, 0) + 1
            return
        _drop_log_last_at[monitor_type] = now
        suppressed = _drop_log_suppressed.pop(monitor_type, 0)

    # Only the emitted line pays the hashing cost (suppressed rows skip it).
    payload_bytes = sum(len(value) for value in params if isinstance(value, str))
    digest = hashlib.sha256()
    for value in params:
        digest.update(str(value).encode('utf-8', 'replace'))
    sample = next((value[:_DROPPED_SAMPLE_CHARS] for value in params if isinstance(value, str) and value), '')
    message = (
        f'trajectory_tdsql: {monitor_type} payload_bytes={payload_bytes} '
        f'sha256={digest.hexdigest()[:16]} sample={sample!r}'
    )
    if suppressed:
        message = f'{message} suppressed={suppressed}'
    report_monitor(
        monitor_type,
        message,
        level=logging.WARNING,
        exc_info=exc_info,
        payload_bytes=payload_bytes,
        **fields,
    )


def _mysql_error_code(exc: BaseException) -> int | None:
    if isinstance(exc, OperationalError):
        code = exc.args[0] if exc.args else None
        return code if isinstance(code, int) else None
    return None


def _is_recoverable_error(exc: BaseException) -> bool:
    if isinstance(exc, InterfaceError):
        return True
    return _mysql_error_code(exc) in _RECONNECT_ERROR_CODES


def _is_partition_race_error(exc: BaseException) -> bool:
    return _mysql_error_code(exc) in _PARTITION_RACE_ERROR_CODES


class TDSQLTraceWriter:
    def __init__(
        self,
        dialect: str = TDSQL_DIALECT,
        *,
        max_queue_items: int | None = None,
        max_queue_bytes: int | None = None,
    ) -> None:
        self.dialect = validate_dialect(dialect)
        self.max_queue_items = resolve_max_queue_items(max_queue_items)
        self.max_queue_bytes = resolve_max_queue_bytes(max_queue_bytes)
        self._db_config: dict[str, Any] | None = None
        self._conn: Any | None = None
        self._io_loop: asyncio.AbstractEventLoop | None = None
        self._io_thread: threading.Thread | None = None
        self._task: asyncio.Task[None] | None = None
        self._event: asyncio.Event | None = None
        self._queue_lock = threading.Lock()
        self._queue: deque[_Write] = deque()
        self._queued_bytes = 0
        self._dropped_since_report = 0
        self._started = False
        self._start_lock = threading.Lock()
        self._starting = False
        self._ready_event = threading.Event()
        self._closing = False
        self._healthy = True
        self._writer_dead = False

        self._state_lock = threading.Lock()
        self._seq = 0
        self._settled_through = 0
        self._failed_run_spans: dict[str, int] = {}

    @property
    def is_ready(self) -> bool:
        return self._started and self._conn is not None and self._healthy and not self._writer_dead

    async def ensure_started(self, db_config: dict[str, Any] | None) -> bool:
        if db_config is None:
            logger.debug('trajectory_tdsql: upload disabled, db_config missing')
            return False
        config = dict(db_config)

        with self._start_lock:
            if self._db_config is not None and self._db_config != config:
                raise ValueError('TDSQL writer is already bound to a different database config')
            if self._started:
                return True
            if not self._starting:
                self._starting = True
                self._db_config = config
                self._ready_event = threading.Event()
                self._launch_io_thread()
            ready_event = self._ready_event

        await asyncio.to_thread(ready_event.wait, _START_WAIT_TIMEOUT_SECONDS)
        return self._started

    def _launch_io_thread(self) -> None:
        thread = threading.Thread(target=self._run_io_loop, name='TDSQLTraceWriterIO', daemon=True)
        self._io_thread = thread
        thread.start()

    def _run_io_loop(self) -> None:
        """Own event loop for the writer, run on a dedicated daemon thread."""
        current = threading.current_thread()
        loop = asyncio.new_event_loop()
        self._io_loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(asyncio.wait_for(self._startup(), timeout=_WRITER_START_TIMEOUT_SECONDS))
        except Exception:
            logger.warning('trajectory_tdsql: writer failed to start, disabling upload', exc_info=True)
            with contextlib.suppress(Exception):
                loop.run_until_complete(self._safe_close_conn())
            with contextlib.suppress(Exception):
                loop.close()
            with self._start_lock:
                self._started = False
                # Only clear our own handles: a concurrent ensure_started() may
                # have already relaunched and overwritten them.
                if self._io_thread is current:
                    self._io_thread = None
                if self._io_loop is loop:
                    self._io_loop = None
                # Signal the event *these* waiters hold, then reset _starting so a
                # relaunch cannot swap in a new _ready_event before we set this one.
                ready_event = self._ready_event
                self._starting = False
            ready_event.set()
            return

        self._event = asyncio.Event()
        self._closing = False
        self._task = loop.create_task(self._writer_loop(), name='TDSQLTraceWriter')
        with self._start_lock:
            self._started = True
            self._starting = False
        self._ready_event.set()
        logger.debug('trajectory_tdsql: writer started')
        try:
            loop.run_forever()
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(self._safe_close_conn())
            with contextlib.suppress(Exception):
                loop.close()
            with self._start_lock:
                self._started = False
                if self._io_thread is current:
                    self._io_thread = None
                if self._io_loop is loop:
                    self._io_loop = None

    async def _startup(self) -> None:
        assert self._db_config is not None
        with measure_database('connect', source='trace_db', acquire=True):
            self._conn = await mysql_connect(**self._db_config)
        await self._init_schema()

    async def _init_schema(self) -> None:
        import warnings

        assert self._conn is not None
        async with self._conn.cursor() as cur:
            # Boundless yields no bootstrap statements: its principal tables are created
            # lazily by prepare_scope with a partition layout.
            for statement in schema_statements(self.dialect):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    with contextlib.suppress(DriverWarning):
                        await cur.execute(statement)
        await self._conn.commit()

    # -- enqueue (thread-safe, fire-and-forget) ------------------------------

    def _enqueue(self, write: _Write) -> int:

        if not self._started or self._closing:
            return 0
        report_drops = 0
        with self._queue_lock:
            if self._writer_dead:
                return 0
            if len(self._queue) >= self.max_queue_items or self._queued_bytes + write.nbytes > self.max_queue_bytes:
                self._dropped_since_report += 1
                if self._dropped_since_report >= _BACKPRESSURE_REPORT_EVERY:
                    report_drops = self._dropped_since_report
                    self._dropped_since_report = 0
                seq = 0
            else:
                self._seq += 1
                write.seq = self._seq
                self._queue.append(write)
                self._queued_bytes += write.nbytes
                seq = self._seq
        if report_drops:
            report_monitor(
                'backpressure_drop',
                f'trajectory_tdsql: dropped {report_drops} rows under backpressure (queue full)',
                level=logging.WARNING,
                rows=report_drops,
            )
        if seq:
            self._notify()
        return seq

    def _notify(self) -> None:
        loop = self._io_loop
        if loop is None or loop.is_closed():
            return

        def wake() -> None:
            if self._event is not None:
                self._event.set()

        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(wake)

    def enqueue_span_event(
        self,
        params: tuple[Any, ...],
        *,
        run_span_id: str = '',
        principal_scope_id: str | None = None,
        parent_execution_id: str | None = None,
    ) -> int:
        """Queue one span-event row; returns its monotonic seq (0 if dropped)."""
        return self._enqueue(
            _Write(
                span_event_insert_sql(principal_scope_id, dialect=self.dialect),
                params,
                standalone=_is_oversized(params),
                run_span_id=run_span_id,
                principal_scope_id=principal_scope_id,
                parent_execution_id=parent_execution_id,
                nbytes=_write_bytes(params),
            )
        )

    # -- settle watermark + per-run failure tracking -------------------------

    @property
    def settled_through(self) -> int:
        """Highest seq with a final outcome (committed or dropped)."""
        with self._state_lock:
            return self._settled_through

    def _advance_settled(self, batch_max_seq: int) -> None:
        """Advance the settle watermark after a batch reaches a final outcome."""
        with self._state_lock:
            if batch_max_seq > self._settled_through:
                self._settled_through = batch_max_seq

    def _record_run_failures(self, batch: list[_Write]) -> None:
        with self._state_lock:
            for write in batch:
                if not write.run_span_id:
                    continue
                # Sink close removes this marker; eviction before then would turn a dropped row into a false success.
                self._failed_run_spans[write.run_span_id] = write.seq

    def consume_run_failure(self, run_span_id: str) -> int | None:
        with self._state_lock:
            return self._failed_run_spans.pop(run_span_id, None)

    async def wait_until_settled(self, seq: int, *, timeout: float) -> None:

        if seq <= 0:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while self.settled_through < seq:
            if loop.time() >= deadline:
                raise TimeoutError
            await asyncio.sleep(_SETTLE_POLL_INTERVAL_SECONDS)

    # -- writer loop ---------------------------------------------------------

    def _pop_batch(self) -> list[_Write]:
        batch: list[_Write] = []
        with self._queue_lock:
            while self._queue and len(batch) < _BATCH_SIZE:
                nxt = self._queue[0]
                if nxt.standalone:
                    if not batch:
                        popped = self._queue.popleft()
                        self._queued_bytes -= popped.nbytes
                        batch.append(popped)
                    break
                popped = self._queue.popleft()
                self._queued_bytes -= popped.nbytes
                batch.append(popped)
            if not self._queue:
                # Guard against drift from any accounting edge case.
                self._queued_bytes = 0
        return batch

    def _queue_len(self) -> int:
        with self._queue_lock:
            return len(self._queue)

    async def _writer_loop(self) -> None:
        assert self._event is not None
        try:
            while True:
                if self._queue_len() == 0:
                    if self._closing:
                        break
                    self._event.clear()
                    if self._queue_len() == 0 and not self._closing:
                        if self._healthy:
                            await self._event.wait()
                        else:
                            with contextlib.suppress(asyncio.TimeoutError):
                                await asyncio.wait_for(self._event.wait(), timeout=_HEALTH_RECOVER_INTERVAL_SECONDS)
                            if self._queue_len() == 0 and not self._closing and not self._healthy:
                                await self._try_recover()
                    continue

                # Small window to accumulate a fuller batch before flushing.
                if self._queue_len() < _BATCH_SIZE and not self._closing:
                    await asyncio.sleep(_BATCH_WINDOW_SECONDS)

                batch = self._pop_batch()
                if batch:
                    await self._flush_batch(batch)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning('trajectory_tdsql: writer loop stopped unexpectedly', exc_info=True)
            self._fail_all_pending()

    def _fail_all_pending(self) -> None:
        """Unblock close() waiters when the writer loop dies with rows still queued.

        Without this the io loop keeps running but nothing drains the queue, so every
        wait_until_settled() would block until its configured timeout. Instead we mark the
        writer dead (refusing new rows), record the queued run-spans as failed, and
        advance the settle watermark so waiters return promptly with a drop outcome.
        """
        self._healthy = False
        with self._queue_lock:
            self._writer_dead = True
            pending = list(self._queue)
            self._queue.clear()
            self._queued_bytes = 0
            max_seq = self._seq
        if pending:
            self._record_run_failures(pending)
        self._advance_settled(max_seq)

    async def _flush_batch(self, batch: list[_Write]) -> None:
        grouped: dict[str, list[_Write]] = {}
        for write in batch:
            grouped.setdefault(write.sql, []).append(write)

        batch_max_seq = max((write.seq for write in batch), default=0)
        all_succeeded = True
        for sql, writes in grouped.items():
            started_at = time.monotonic()
            payload_bytes = sum(write.nbytes or _write_bytes(write.params) for write in writes)

            async def do_write(conn: Any, *, sql: str = sql, writes: list[_Write] = writes) -> None:
                rows = [write.params for write in writes]
                async with conn.cursor() as cur:
                    if len(rows) == 1:
                        await cur.execute(sql, rows[0])
                    else:
                        await cur.executemany(sql, rows)

            try:
                await self._execute_with_reconnect(do_write)
            except Exception:
                all_succeeded = False
                self._record_run_failures(writes)
                self._report_write_failure(writes, payload_bytes=payload_bytes, started_at=started_at)
                continue

            elapsed = time.monotonic() - started_at
            report_monitor(
                'write_success',
                f'trajectory_tdsql: db write_success writes={len(writes)} bytes={payload_bytes} '
                f'elapsed_ms={elapsed * 1000:.1f}',
                level=logging.DEBUG,
                rows=len(writes),
                payload_bytes=payload_bytes,
                elapsed_ms=round(elapsed * 1000, 3),
            )
            if elapsed > SLOW_OPERATION_WARN_SECONDS:
                logger.warning('trajectory_tdsql: slow DB write %.1fs for %d write(s)', elapsed, len(writes))

        self._healthy = all_succeeded
        self._advance_settled(batch_max_seq)

    def _report_write_failure(self, writes: list[_Write], *, payload_bytes: int, started_at: float) -> None:
        elapsed_ms = (time.monotonic() - started_at) * 1000
        report_monitor(
            'write_failure',
            f'trajectory_tdsql: db write_failure writes={len(writes)} bytes={payload_bytes} '
            f'elapsed_ms={elapsed_ms:.1f}',
            level=logging.WARNING,
            exc_info=True,
            rows=len(writes),
            payload_bytes=payload_bytes,
            elapsed_ms=round(elapsed_ms, 3),
        )
        for write in writes:
            params = write.params
            scope_fields: dict[str, Any]
            if write.principal_scope_id is not None:
                scope_fields = {
                    'principal_scope_id': write.principal_scope_id,
                    'parent_execution_id': write.parent_execution_id,
                    'storage_scope': 'principal_v1',
                }
            else:
                scope_fields = {
                    'trajectory_group_id': params[2],
                    'storage_scope': 'shared',
                }
            log_dropped_write(
                'batch_dropped',
                params,
                run_id=params[0],
                run_span_id=params[1],
                **scope_fields,
            )

    async def _execute_with_reconnect(self, do_write: Any) -> None:
        attempt = 0
        race_attempt = 0
        while True:
            assert self._conn is not None
            try:
                with (
                    measure_operation('trace_write', family='persistence', source='trace_db'),
                    measure_database('write', source='trace_db'),
                ):
                    await do_write(self._conn)
                    await self._conn.commit()
                return
            except Exception as exc:
                with contextlib.suppress(Exception):
                    await self._conn.rollback()
                if _is_partition_race_error(exc):
                    race_attempt += 1
                    if race_attempt > _MAX_PARTITION_RACE_ATTEMPTS:
                        raise
                    # Whoever won has already published the partition, so this only waits
                    # out the DDL rather than backing off an overloaded server.
                    backoff = _PARTITION_RACE_BACKOFF_SECONDS * (2 ** (race_attempt - 1))
                    logger.warning(
                        'trajectory_tdsql: partition race (attempt %d/%d), retrying in %.1fs: %s',
                        race_attempt,
                        _MAX_PARTITION_RACE_ATTEMPTS,
                        backoff,
                        exc,
                    )
                    await asyncio.sleep(backoff)
                    continue
                if not _is_recoverable_error(exc):
                    raise
                attempt += 1
                if attempt > _MAX_RECONNECT_ATTEMPTS:
                    raise
                backoff = min(
                    _RECONNECT_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)),
                    _RECONNECT_BACKOFF_MAX_SECONDS,
                )
                logger.warning(
                    'trajectory_tdsql: recoverable DB error (attempt %d/%d), reconnecting in %.1fs: %s',
                    attempt,
                    _MAX_RECONNECT_ATTEMPTS,
                    backoff,
                    exc,
                )
                await asyncio.sleep(backoff)
                with contextlib.suppress(Exception):
                    await self._reconnect()

    async def _reconnect(self) -> None:
        assert self._db_config is not None

        with measure_database('connect', source='trace_db', acquire=True):
            new_conn = await asyncio.wait_for(mysql_connect(**self._db_config), timeout=_WRITER_START_TIMEOUT_SECONDS)
        old = self._conn
        self._conn = new_conn
        if old is not None:
            with contextlib.suppress(Exception):
                old.close()

    async def _try_recover(self) -> None:
        try:
            await self._reconnect()
            conn = self._conn
            if conn is None:
                raise RuntimeError('TDSQL trace writer reconnect returned no connection')
            async with conn.cursor() as cur:
                await cur.execute('SELECT @@read_only')
                fetched = cur.fetchone()
                if inspect.isawaitable(fetched):
                    fetched = await fetched
        except Exception:
            logger.warning('trajectory_tdsql: writer recovery attempt failed', exc_info=True)
            return
        if fetched and int(fetched[0] or 0):
            logger.warning('trajectory_tdsql: target DB still read-only, staying unhealthy')
            return
        self._healthy = True
        logger.info('trajectory_tdsql: writer connection recovered')

    async def _safe_close_conn(self) -> None:
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._conn = None

    async def _shutdown(self) -> None:
        """Drain and tear down on the io loop, then stop it."""
        if self._task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._task, timeout=10)
        await self._safe_close_conn()
        loop = self._io_loop
        if loop is not None:
            loop.stop()

    async def aclose(self) -> None:
        """Flush and shut down (best-effort; for process teardown)."""
        if not self._started:
            return
        self._closing = True
        self._notify()
        loop = self._io_loop
        thread = self._io_thread
        if loop is not None and not loop.is_closed():
            # Run the drain/teardown on the io loop, then wait for it from here.
            with contextlib.suppress(Exception):
                future = asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
                await asyncio.wait_for(asyncio.wrap_future(future), timeout=15)
        if thread is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(thread.join, 5)
        self._started = False

    def close_sync(self) -> None:
        if not self._started:
            return
        self._closing = True
        self._notify()
        loop = self._io_loop
        thread = self._io_thread
        if loop is not None and not loop.is_closed():
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(self._shutdown(), loop).result(timeout=15)
        if thread is not None:
            with contextlib.suppress(Exception):
                thread.join(timeout=5)
        self._started = False


# Only sinks with identical connection settings, dialect and queue limits share a
# writer. Different limits must not silently overwrite another run's queue policy.
_DbConfigKey = tuple[tuple[tuple[str, Any], ...], str, int, int]


def _db_config_key(
    db_config: dict[str, Any],
    dialect: str = TDSQL_DIALECT,
    *,
    max_queue_items: int | None = None,
    max_queue_bytes: int | None = None,
) -> _DbConfigKey:
    key = tuple(sorted(db_config.items()))
    try:
        hash(key)
    except TypeError as exc:
        raise ValueError('TDSQL writer database config values must be scalar') from exc
    return (
        key,
        validate_dialect(dialect),
        resolve_max_queue_items(max_queue_items),
        resolve_max_queue_bytes(max_queue_bytes),
    )


_WriterKey = tuple[_DbConfigKey, int, int]
_writers: dict[_WriterKey, TDSQLTraceWriter] = {}
_writer_lock = threading.Lock()

_DISABLED_REPORT_EVERY = 500
_disabled_counts: dict[_WriterKey, int] = {}


async def get_writer(
    db_config: dict[str, Any] | None,
    *,
    dialect: str = TDSQL_DIALECT,
    max_queue_items: int | None = None,
    max_queue_bytes: int | None = None,
    writer_shards: int | None = None,
    run_span_id: str = '',
) -> TDSQLTraceWriter | None:
    if db_config is None:
        logger.debug('trajectory_tdsql: upload disabled, db_config missing')
        return None
    items = resolve_max_queue_items(max_queue_items)
    size = resolve_max_queue_bytes(max_queue_bytes)
    shards = resolve_writer_shards(writer_shards)
    if not isinstance(run_span_id, str):
        raise ValueError('TDSQL writer run_span_id must be a string')
    # A run stays on one serial writer, so sequence watermarks remain local and
    # close() cannot mistake another shard's progress for its own completion.
    shard = int.from_bytes(hashlib.sha256(run_span_id.encode()).digest()[:8], 'big') % shards if run_span_id else 0
    key = (_db_config_key(db_config, dialect, max_queue_items=items, max_queue_bytes=size), shards, shard)
    with _writer_lock:
        writer = _writers.get(key)
        if writer is None:
            writer = TDSQLTraceWriter(dialect=dialect, max_queue_items=items, max_queue_bytes=size)
            _writers[key] = writer
    started = await writer.ensure_started(db_config)
    if started and writer.is_ready:
        with _writer_lock:
            _disabled_counts.pop(key, None)
        return writer

    with _writer_lock:
        disabled_count = _disabled_counts.get(key, 0) + 1
        report_disabled = disabled_count >= _DISABLED_REPORT_EVERY
        _disabled_counts[key] = 0 if report_disabled else disabled_count
    if report_disabled:
        logger.warning('trajectory_tdsql: writer unavailable; upload skipped %d times', disabled_count)
    return None


__all__ = ['TDSQLTraceWriter', 'get_writer', 'log_dropped_write', 'report_monitor']


def _close_writer_at_exit() -> None:
    with _writer_lock:
        writers = list(_writers.values())
    for writer in writers:
        with contextlib.suppress(Exception):
            writer.close_sync()


atexit.register(_close_writer_at_exit)
