"""Database storage for batch runner.

All batch runs are stored in SQLite database, supporting:
- Unified storage backend for all output modes
- Async operations with aiosqlite
- Flexible export to different formats
- In-memory mode support
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
import time

from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import aiosqlite


logger = logging.getLogger(__name__)


# SQLite schema version
SCHEMA_VERSION = 2

_SQLITE_BUSY_TIMEOUT_MS = 30_000
_LOW_PRIORITY_BATCH_SIZE = 100
_LOW_PRIORITY_BATCH_WINDOW_SECONDS = 0.25
_HIGH_PRIORITY_WRITE_WARN_SECONDS = 5.0
_LOW_PRIORITY_WRITE_WARN_SECONDS = 30.0

# aiosqlite's default rows-per-thread-hop for ``async for`` over a cursor.
_DEFAULT_ROW_PREFETCH = 64

_TRACE_INSERT_SQL = """
INSERT INTO run_trace_events (
    run_id, event_type, time_unix_nano, trace_id, span_id,
    parent_span_id, event_data
) VALUES (?, ?, ?, ?, ?, ?, ?)
"""

_LOG_INSERT_SQL = """
INSERT INTO run_logs (
    run_id, chunk_index, stream_type, start_time, end_time,
    size_bytes, content
) VALUES (?, ?, ?, ?, ?, ?, ?)
"""


@dataclass
class _HighPriorityWrite:
    name: str
    write: Callable[[aiosqlite.Connection], Awaitable[None]]
    future: asyncio.Future[None]
    enqueued_at: float
    commit: bool = True


@dataclass
class _TraceWrite:
    run_id: str
    event_type: str
    time_unix_nano: int
    trace_id: str
    span_id: str
    parent_span_id: str | None
    event_data: str
    enqueued_at: float
    future: asyncio.Future[None] | None = None


@dataclass
class _LogWrite:
    run_id: str
    chunk_index: int
    stream_type: str
    start_time: str
    end_time: str
    size_bytes: int
    content: str
    enqueued_at: float
    future: asyncio.Future[None] | None = None


_LowPriorityWrite: TypeAlias = _TraceWrite | _LogWrite

# Database schema
SCHEMA = """
-- Schema version
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

-- Batch metadata
CREATE TABLE IF NOT EXISTS batch_info (
    key TEXT PRIMARY KEY,
    value TEXT  -- JSON
);

-- Run records (main table)
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK(status IN ('pending','running','success','failed','timeout','cancelled')),
    start_time TEXT,
    end_time TEXT,
    error TEXT,
    retry_count INTEGER DEFAULT 0,

    -- Configuration and metadata
    config TEXT,      -- JSON: RunInput
    metadata TEXT,    -- JSON: BatchItem.metadata

    -- Outputs
    task_output TEXT,      -- JSON: TaskRunOutput
    agent_output TEXT,     -- JSON: AgentRunOutput (nullable)
    judge_output TEXT,     -- JSON: Judge output (nullable)
    processor_results TEXT, -- JSON: {key: result}

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_start_time ON runs(start_time);
CREATE INDEX IF NOT EXISTS idx_created_at ON runs(created_at);

-- Trace events table (new in v2)
CREATE TABLE IF NOT EXISTS run_trace_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK(event_type IN ('start', 'update', 'end')),
    time_unix_nano INTEGER NOT NULL,
    trace_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    parent_span_id TEXT,

    -- Event data (JSON)
    event_data TEXT NOT NULL,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_trace_run_id ON run_trace_events(run_id);
CREATE INDEX IF NOT EXISTS idx_trace_timestamp ON run_trace_events(run_id, time_unix_nano);
CREATE INDEX IF NOT EXISTS idx_trace_span ON run_trace_events(trace_id, span_id);

-- Run logs table (new in v2)
CREATE TABLE IF NOT EXISTS run_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    stream_type TEXT NOT NULL CHECK(stream_type IN ('stdout', 'stderr', 'log')),

    -- Chunk metadata
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,

    -- Log content
    content TEXT NOT NULL,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_logs_run_id ON run_logs(run_id);
CREATE INDEX IF NOT EXISTS idx_logs_time ON run_logs(run_id, start_time);
CREATE INDEX IF NOT EXISTS idx_logs_chunk ON run_logs(run_id, chunk_index);
"""


class BatchDatabase:
    """Async database manager for batch runs."""

    def __init__(self, db_path: str | Path):
        """Initialize database manager.

        Args:
            db_path: Path to SQLite database file, or ':memory:' for in-memory
        """
        self.db_path = str(db_path)
        self.conn: aiosqlite.Connection | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._writer_event: asyncio.Event | None = None
        self._high_priority_event: asyncio.Event | None = None
        self._queue_lock = threading.Lock()
        self._high_priority_writes: deque[_HighPriorityWrite] = deque()
        self._low_priority_writes: deque[_LowPriorityWrite] = deque()
        self._writer_closing = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        del exc_type, exc_val, exc_tb  # Unused
        await self.close()

    async def connect(self) -> None:
        """Connect to database and initialize schema."""
        self._loop = asyncio.get_running_loop()
        try:
            self.conn = await aiosqlite.connect(self.db_path)

            # Tune SQLite for long-running batch writes. The application-level writer
            # still serializes mutations; these pragmas reduce external lock friction.
            await self.conn.execute(f'PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}')
            if self.db_path != ':memory:':
                # Allow DELETE journaling on filesystems that cannot reliably support WAL shared memory.
                # Keep WAL unless the environment explicitly selects the alternative.
                if os.environ.get('NEXUS_SQLITE_NO_WAL', '').strip() in ('1', 'true', 'True'):
                    await self.conn.execute('PRAGMA journal_mode=DELETE')
                    await self.conn.execute('PRAGMA synchronous=NORMAL')
                else:
                    await self.conn.execute('PRAGMA journal_mode=WAL')
                    await self.conn.execute('PRAGMA synchronous=NORMAL')
                    await self.conn.execute('PRAGMA wal_autocheckpoint=1000')

            # Enable foreign keys
            await self.conn.execute('PRAGMA foreign_keys=ON')

            # Initialize schema
            await self._init_schema()

            with self._queue_lock:
                self._high_priority_writes.clear()
                self._low_priority_writes.clear()
                self._writer_closing = False

            self._writer_event = asyncio.Event()
            self._high_priority_event = asyncio.Event()
            self._writer_task = asyncio.create_task(self._writer_loop(), name='BatchDatabaseWriter')

        except BaseException:
            try:
                await self.close()
            finally:
                self._loop = None
                self._writer_task = None
                self._writer_event = None
                self._high_priority_event = None
            raise

        logger.debug(f'Connected to database: {self.db_path}')

    async def close(self) -> None:
        """Close database connection."""
        # Close the database connection even if writer shutdown raises or is cancelled.
        try:
            if self._writer_task is not None:
                await self._close_writer()
        except asyncio.CancelledError:
            pass
        except BaseException:
            logger.warning('Writer shutdown failed during database close', exc_info=True)

        if self.conn:
            await self.conn.close()
            self.conn = None
            logger.debug('Database connection closed')

    def _notify_writer(self, *, high_priority: bool = False) -> None:
        """Wake the writer task from any thread."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        def wake() -> None:
            if self._writer_event is not None:
                self._writer_event.set()
            if high_priority and self._high_priority_event is not None:
                self._high_priority_event.set()

        loop.call_soon_threadsafe(wake)

    @staticmethod
    def _set_future_result(future: asyncio.Future[None]) -> None:
        def set_result() -> None:
            if not future.done():
                future.set_result(None)

        loop = future.get_loop()
        if loop.is_closed():
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            set_result()
        else:
            loop.call_soon_threadsafe(set_result)

    @staticmethod
    def _set_future_exception(future: asyncio.Future[None], exc: BaseException) -> None:
        def set_exception() -> None:
            if not future.done():
                future.set_exception(exc)

        loop = future.get_loop()
        if loop.is_closed():
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            set_exception()
        else:
            loop.call_soon_threadsafe(set_exception)

    async def _enqueue_high_priority_write(
        self,
        name: str,
        write: Callable[[aiosqlite.Connection], Awaitable[None]],
        *,
        commit: bool = True,
    ) -> None:
        if self.conn is None:
            raise RuntimeError('Database connection is not open')

        future = asyncio.get_running_loop().create_future()
        item = _HighPriorityWrite(
            name=name,
            write=write,
            future=future,
            enqueued_at=time.monotonic(),
            commit=commit,
        )

        with self._queue_lock:
            if self._writer_task is None or self._writer_closing:
                raise RuntimeError('Database writer is not running')
            self._high_priority_writes.append(item)

        self._notify_writer(high_priority=True)
        await future

    def _enqueue_low_priority_write(self, item: _LowPriorityWrite) -> bool:
        if self.conn is None:
            return False

        with self._queue_lock:
            if self._writer_task is None or self._writer_closing:
                return False
            self._low_priority_writes.append(item)

        self._notify_writer()
        return True

    async def _enqueue_low_priority_write_and_wait(self, name: str, item: _LowPriorityWrite) -> None:
        future = asyncio.get_running_loop().create_future()
        item.future = future
        if not self._enqueue_low_priority_write(item):
            raise RuntimeError('Database writer is not running')

        await future
        elapsed = time.monotonic() - item.enqueued_at
        if elapsed > _LOW_PRIORITY_WRITE_WARN_SECONDS:
            logger.warning('Slow low-priority batch DB write: %.1fs total for %s', elapsed, name)

    async def _close_writer(self) -> None:
        assert self._writer_task is not None

        with self._queue_lock:
            self._writer_closing = True

        self._notify_writer(high_priority=True)

        try:
            await self._writer_task
        finally:
            self._writer_task = None
            self._writer_event = None
            self._high_priority_event = None

    def _pop_high_priority_write(self) -> _HighPriorityWrite | None:
        with self._queue_lock:
            if self._high_priority_writes:
                return self._high_priority_writes.popleft()
        return None

    def _low_priority_count(self) -> int:
        with self._queue_lock:
            return len(self._low_priority_writes)

    def _has_high_priority_writes_or_closing(self) -> bool:
        with self._queue_lock:
            return bool(self._high_priority_writes) or self._writer_closing

    def _has_pending_writes_or_closing(self) -> bool:
        with self._queue_lock:
            return bool(self._high_priority_writes or self._low_priority_writes) or self._writer_closing

    def _pop_low_priority_batch(self) -> list[_LowPriorityWrite]:
        batch = []
        with self._queue_lock:
            while self._low_priority_writes and len(batch) < _LOW_PRIORITY_BATCH_SIZE:
                batch.append(self._low_priority_writes.popleft())
        return batch

    async def _writer_loop(self) -> None:
        try:
            while True:
                high_priority_write = self._pop_high_priority_write()
                if high_priority_write is not None:
                    await self._run_high_priority_write(high_priority_write)
                    continue

                low_priority_count = self._low_priority_count()
                if low_priority_count:
                    if low_priority_count < _LOW_PRIORITY_BATCH_SIZE and not self._writer_closing:
                        await self._wait_for_high_priority_or_batch_window()
                        if self._has_high_priority_writes_or_closing() and self._low_priority_count():
                            continue

                    batch = self._pop_low_priority_batch()
                    if batch:
                        await self._run_low_priority_batch(batch)
                    continue

                if self._writer_closing:
                    break

                assert self._writer_event is not None
                self._writer_event.clear()
                if self._has_pending_writes_or_closing():
                    continue
                await self._writer_event.wait()
        except Exception as exc:
            logger.exception('Batch database writer stopped unexpectedly')
            self._fail_queued_writes(exc)
            raise

    async def _wait_for_high_priority_or_batch_window(self) -> None:
        assert self._high_priority_event is not None
        self._high_priority_event.clear()
        if self._has_high_priority_writes_or_closing():
            return

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._high_priority_event.wait(), timeout=_LOW_PRIORITY_BATCH_WINDOW_SECONDS)

    async def _run_high_priority_write(self, item: _HighPriorityWrite) -> None:
        assert self.conn is not None

        started_at = time.monotonic()
        try:
            await item.write(self.conn)
            if item.commit:
                await self.conn.commit()
        except Exception as exc:
            logger.exception('Failed high-priority batch DB write: operation=%s', item.name)
            try:
                await self.conn.rollback()
            except Exception:
                logger.exception('Failed to roll back batch DB write after %s failed', item.name)
            self._set_future_exception(item.future, exc)
            return

        write_elapsed = time.monotonic() - started_at
        total_elapsed = time.monotonic() - item.enqueued_at
        if total_elapsed > _HIGH_PRIORITY_WRITE_WARN_SECONDS:
            logger.warning(
                'Slow high-priority batch DB write: %.1fs total, %.1fs executing, operation=%s',
                total_elapsed,
                write_elapsed,
                item.name,
            )
        self._set_future_result(item.future)

    async def _run_low_priority_batch(self, batch: list[_LowPriorityWrite]) -> None:
        assert self.conn is not None

        trace_rows = [
            (
                item.run_id,
                item.event_type,
                item.time_unix_nano,
                item.trace_id,
                item.span_id,
                item.parent_span_id,
                item.event_data,
            )
            for item in batch
            if isinstance(item, _TraceWrite)
        ]
        log_rows = [
            (
                item.run_id,
                item.chunk_index,
                item.stream_type,
                item.start_time,
                item.end_time,
                item.size_bytes,
                item.content,
            )
            for item in batch
            if isinstance(item, _LogWrite)
        ]

        started_at = time.monotonic()
        try:
            if trace_rows:
                await self.conn.executemany(_TRACE_INSERT_SQL, trace_rows)
            if log_rows:
                await self.conn.executemany(_LOG_INSERT_SQL, log_rows)
            await self.conn.commit()
        except Exception as exc:
            try:
                await self.conn.rollback()
            except Exception:
                logger.exception('Failed to roll back low-priority batch DB write')
            logger.exception(
                'Failed to write low-priority batch DB records: trace_events=%d log_chunks=%d',
                len(trace_rows),
                len(log_rows),
            )
            for item in batch:
                if item.future is not None:
                    self._set_future_exception(item.future, exc)
            return

        write_elapsed = time.monotonic() - started_at
        oldest_age = time.monotonic() - min(item.enqueued_at for item in batch)
        if write_elapsed > _LOW_PRIORITY_WRITE_WARN_SECONDS or oldest_age > _LOW_PRIORITY_WRITE_WARN_SECONDS:
            logger.warning(
                'Slow low-priority batch DB write: %.1fs executing, %.1fs oldest queued, trace_events=%d log_chunks=%d',
                write_elapsed,
                oldest_age,
                len(trace_rows),
                len(log_rows),
            )

        for item in batch:
            if item.future is not None:
                self._set_future_result(item.future)

    def _fail_queued_writes(self, exc: BaseException) -> None:
        with self._queue_lock:
            high_priority_writes = list(self._high_priority_writes)
            low_priority_writes = list(self._low_priority_writes)
            self._high_priority_writes.clear()
            self._low_priority_writes.clear()

        for item in high_priority_writes:
            self._set_future_exception(item.future, exc)
        for item in low_priority_writes:
            if item.future is not None:
                self._set_future_exception(item.future, exc)

    async def _init_schema(self) -> None:
        """Initialize database schema."""
        assert self.conn is not None
        # Create tables
        await self.conn.executescript(SCHEMA)

        # Check/set schema version
        async with self.conn.execute('SELECT version FROM schema_version') as cursor:
            row = await cursor.fetchone()

        if row is None:
            await self.conn.execute(
                'INSERT INTO schema_version (version) VALUES (?)',
                (SCHEMA_VERSION,),
            )
            await self.conn.commit()
            logger.debug(f'Initialized database schema version {SCHEMA_VERSION}')
        else:
            version = row[0]
            if version != SCHEMA_VERSION:
                raise RuntimeError(
                    f'Database schema version mismatch: database has version {version}, '
                    f'but this version of nexus requires version {SCHEMA_VERSION}. '
                    f'Please use a compatible version of nexus to run this batch.'
                )

        # Additive, nullable extension of v2: old runs retain NULL and all
        # existing columns/records remain readable by older batch consumers.
        # Readers of an already migrated DB must not contend for the writer lock.
        async with self.conn.execute('PRAGMA table_info(runs)') as cursor:
            columns = {column[1] for column in await cursor.fetchall()}
        if 'judge_output' in columns:
            return

        # Recheck under the lock: another connection may have just migrated it.
        await self.conn.execute('BEGIN IMMEDIATE')
        try:
            async with self.conn.execute('PRAGMA table_info(runs)') as cursor:
                columns = {column[1] for column in await cursor.fetchall()}
            if 'judge_output' not in columns:
                await self.conn.execute('ALTER TABLE runs ADD COLUMN judge_output TEXT')
            await self.conn.commit()
        except BaseException:
            await self.conn.rollback()
            raise

    async def set_batch_info(self, key: str, value: Any) -> None:
        """Set batch metadata.

        Args:
            key: Metadata key
            value: Metadata value (will be JSON serialized)
        """
        serialized_value = json.dumps(value, ensure_ascii=False)

        async def write(conn: aiosqlite.Connection) -> None:
            await conn.execute(
                'INSERT OR REPLACE INTO batch_info (key, value) VALUES (?, ?)',
                (key, serialized_value),
            )

        await self._enqueue_high_priority_write('set_batch_info', write)

    async def get_batch_info(self, key: str) -> Any | None:
        """Get batch metadata.

        Args:
            key: Metadata key

        Returns:
            Metadata value or None
        """
        assert self.conn is not None
        cursor = await self.conn.execute('SELECT value FROM batch_info WHERE key = ?', (key,))
        row = await cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    async def is_generation_complete(self) -> bool:
        """Check if config generation phase is complete.

        Returns:
            True if generation is complete and configs are locked
        """
        result = await self.get_batch_info('generation_complete')
        return result is True

    async def mark_generation_complete(self) -> None:
        """Mark config generation phase as complete.

        After this, run configs are locked and cannot be modified.
        """
        await self.set_batch_info('generation_complete', True)
        logger.info('Config generation phase completed, configs are now locked')

    async def get_total_runs(self) -> int:
        """Get total number of runs in the database.

        Returns:
            Total number of runs
        """
        assert self.conn is not None
        cursor = await self.conn.execute('SELECT COUNT(*) FROM runs')
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def insert_run(
        self,
        run_id: str,
        config: dict,
        metadata: dict,
    ) -> None:
        """Insert a new run record.

        Args:
            run_id: Unique run identifier
            config: RunInput configuration (as dict)
            metadata: BatchItem metadata
        """
        config_json = json.dumps(config, ensure_ascii=False)
        metadata_json = json.dumps(metadata, ensure_ascii=False)

        async def write(conn: aiosqlite.Connection) -> None:
            await conn.execute(
                """
                INSERT INTO runs (run_id, status, config, metadata)
                VALUES (?, 'pending', ?, ?)
                """,
                (run_id, config_json, metadata_json),
            )

        await self._enqueue_high_priority_write('insert_run', write)

    async def update_run_status(
        self,
        run_id: str,
        status: str,
        **kwargs: Any,
    ) -> None:
        """Update run status and optional fields.

        Args:
            run_id: Run identifier
            status: New status
            **kwargs: Additional fields to update (start_time, end_time, error, task_output, etc.)
        """
        # Build UPDATE statement dynamically
        fields = ['status = ?', 'updated_at = CURRENT_TIMESTAMP']
        values = [status]

        for key, value in kwargs.items():
            if key in [
                'start_time',
                'end_time',
                'error',
                'retry_count',
                'task_output',
                'agent_output',
                'judge_output',
                'processor_results',
            ]:
                fields.append(f'{key} = ?')
                # JSON serialize dict/list values
                if isinstance(value, (dict, list)):
                    values.append(json.dumps(value, ensure_ascii=False))
                else:
                    values.append(value)

        values.append(run_id)

        query = f'UPDATE runs SET {", ".join(fields)} WHERE run_id = ?'

        async def write(conn: aiosqlite.Connection) -> None:
            await conn.execute(query, values)

        await self._enqueue_high_priority_write('update_run_status', write)

    async def get_run_status(self, run_id: str) -> str | None:
        """Get run status.

        Args:
            run_id: Run identifier

        Returns:
            Status string or None
        """
        assert self.conn is not None
        cursor = await self.conn.execute('SELECT status FROM runs WHERE run_id = ?', (run_id,))
        row = await cursor.fetchone()
        return row[0] if row else None

    async def run_exists(self, run_id: str) -> bool:
        """Check if a run exists in the database.

        Args:
            run_id: Run identifier

        Returns:
            True if run exists, False otherwise
        """
        assert self.conn is not None
        cursor = await self.conn.execute('SELECT 1 FROM runs WHERE run_id = ? LIMIT 1', (run_id,))
        row = await cursor.fetchone()
        return row is not None

    async def get_retry_count(self, run_id: str) -> int:
        """Get current retry count for a run.

        Args:
            run_id: Run identifier

        Returns:
            Current retry count, or 0 if run doesn't exist
        """
        assert self.conn is not None
        cursor = await self.conn.execute('SELECT retry_count FROM runs WHERE run_id = ?', (run_id,))
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_run(self, run_id: str) -> dict | None:
        """Get complete run record.

        Args:
            run_id: Run identifier

        Returns:
            Run record as dict or None
        """
        assert self.conn is not None
        cursor = await self.conn.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,))
        row = await cursor.fetchone()
        if not row:
            return None

        # Convert row to dict
        columns = [desc[0] for desc in cursor.description]
        record = dict(zip(columns, row, strict=True))

        # Deserialize JSON fields
        for field in ['config', 'metadata', 'task_output', 'agent_output', 'judge_output', 'processor_results']:
            if record.get(field):
                record[field] = json.loads(record[field])

        return record

    _JSON_COLUMNS = frozenset({
        'config',
        'metadata',
        'task_output',
        'agent_output',
        'judge_output',
        'processor_results',
    })
    _ALL_COLUMNS = (
        'run_id',
        'status',
        'start_time',
        'end_time',
        'error',
        'retry_count',
        'config',
        'metadata',
        'task_output',
        'agent_output',
        'judge_output',
        'processor_results',
        'created_at',
        'updated_at',
    )

    async def iter_runs(
        self,
        *,
        status: str | None = None,
        fields: Sequence[str] | None = None,
        prefetch: int | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream run records in ``created_at`` order.

        aiosqlite materializes ``prefetch`` raw rows per thread hop. When any JSON
        payload column is selected the default is 1, so peak memory is one raw row
        plus its deserialized form rather than 64 raw rows; summary-only selections
        keep the aiosqlite default because one hop per small row is ~17x slower.

        Do not issue other queries on this connection until the iterator is
        exhausted (single shared aiosqlite connection). Callers that may stop
        early should wrap it in ``contextlib.aclosing`` so the cursor is closed
        before the connection.
        """
        assert self.conn is not None
        selected = tuple(fields) if fields is not None else self._ALL_COLUMNS
        unknown = set(selected) - set(self._ALL_COLUMNS)
        if unknown:
            raise ValueError(f'unknown run columns: {sorted(unknown)}')
        if prefetch is not None and prefetch < 1:
            raise ValueError('prefetch must be >= 1')
        sql = (
            f'SELECT {", ".join(selected)} FROM runs' + (' WHERE status = ?' if status else '') + ' ORDER BY created_at'
        )
        json_columns = self._JSON_COLUMNS.intersection(selected)
        async with self.conn.execute(sql, (status,) if status else ()) as cursor:
            cursor.iter_chunk_size = prefetch or (1 if json_columns else _DEFAULT_ROW_PREFETCH)
            columns = [desc[0] for desc in cursor.description]
            async for row in cursor:
                record = dict(zip(columns, row, strict=True))
                for field_name in json_columns:
                    if record.get(field_name):
                        try:
                            record[field_name] = json.loads(record[field_name])
                        except (json.JSONDecodeError, TypeError):
                            record[field_name] = None
                yield record

    async def get_all_runs(self, status: str | None = None) -> list[dict]:
        """Get all run records, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of run records
        """
        # Everything is materialized anyway, so keep aiosqlite's batched fetch.
        return [record async for record in self.iter_runs(status=status, prefetch=_DEFAULT_ROW_PREFETCH)]

    async def get_summary(self) -> dict[str, int]:
        """Get summary statistics including score breakdown.

        Returns:
            Dict with counts by status and score statistics
        """
        assert self.conn is not None

        # Get status counts
        cursor = await self.conn.execute('SELECT status, COUNT(*) as count FROM runs GROUP BY status')

        summary = {
            'total': 0,
            'pending': 0,
            'running': 0,
            'success': 0,
            'failed': 0,
            'timeout': 0,
            'cancelled': 0,
            'score_correct': 0,
            'score_incorrect': 0,
            'score_unknown': 0,
        }

        async for row in cursor:
            status, count = row
            summary[status] = count
            summary['total'] += count

        # Get score statistics using JSON functions
        # Count successful runs with score=1 (correct)
        cursor = await self.conn.execute(
            """
            SELECT COUNT(*) FROM runs
            WHERE status = 'success'
            AND task_output IS NOT NULL
            AND json_extract(task_output, '$.score') = 1
            """
        )
        row = await cursor.fetchone()
        summary['score_correct'] = row[0] if row else 0

        # Count successful runs with score=0 (incorrect)
        cursor = await self.conn.execute(
            """
            SELECT COUNT(*) FROM runs
            WHERE status = 'success'
            AND task_output IS NOT NULL
            AND json_extract(task_output, '$.score') = 0
            """
        )
        row = await cursor.fetchone()
        summary['score_incorrect'] = row[0] if row else 0

        # Count successful runs with other score values or no score (unknown)
        cursor = await self.conn.execute(
            """
            SELECT COUNT(*) FROM runs
            WHERE status = 'success'
            AND task_output IS NOT NULL
            AND (
                json_extract(task_output, '$.score') IS NULL
                OR (json_extract(task_output, '$.score') != 0 AND json_extract(task_output, '$.score') != 1)
            )
            """
        )
        row = await cursor.fetchone()
        summary['score_unknown'] = row[0] if row else 0

        return summary

    async def vacuum(self) -> None:
        """Optimize database (reclaim space after deletions)."""
        if self.db_path != ':memory:':

            async def write(conn: aiosqlite.Connection) -> None:
                await conn.execute('VACUUM')

            await self._enqueue_high_priority_write('vacuum', write, commit=False)
            logger.debug('Database vacuumed')

    # Trace events methods (new in v2)

    def enqueue_trace_record(
        self,
        *,
        run_id: str,
        event_type: str,
        time_unix_nano: int,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        event_data: str,
    ) -> bool:
        """Queue a trace event for low-priority batched insertion.

        This method is thread-safe and intentionally fire-and-forget. It returns
        False if the database is already closing or unavailable.
        """
        return self._enqueue_low_priority_write(
            _TraceWrite(
                run_id=run_id,
                event_type=event_type,
                time_unix_nano=time_unix_nano,
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                event_data=event_data,
                enqueued_at=time.monotonic(),
            )
        )

    async def insert_trace_event(
        self,
        run_id: str,
        event: Any,  # SpanStartEvent | SpanUpdateEvent | SpanEndEvent
    ) -> None:
        """Insert a single trace event.

        Args:
            run_id: Run identifier
            event: Trace event (SpanStartEvent/SpanUpdateEvent/SpanEndEvent)
        """
        # Determine event type
        event_type_name = type(event).__name__
        if event_type_name == 'SpanStartEvent':
            event_type = 'start'
        elif event_type_name == 'SpanUpdateEvent':
            event_type = 'update'
        elif event_type_name == 'SpanEndEvent':
            event_type = 'end'
        else:
            raise ValueError(f'Unknown event type: {event_type_name}')

        # Extract timestamp (use time_unix_nano field)
        time_unix_nano = event.time_unix_nano

        # Serialize event data
        event_data = json.dumps(event.model_dump(), default=str, ensure_ascii=False)

        await self._enqueue_low_priority_write_and_wait(
            'insert_trace_event',
            _TraceWrite(
                run_id,
                event_type,
                time_unix_nano,
                event.trace_id,
                event.span_id,
                event.parent_span_id,
                event_data,
                time.monotonic(),
            ),
        )

    async def get_trace_events(
        self,
        run_id: str,
        start_time_ns: int | None = None,
        end_time_ns: int | None = None,
    ) -> list[dict]:
        """Query trace events for a run with optional time range filter.

        Args:
            run_id: Run identifier
            start_time_ns: Optional start time filter (unix nano)
            end_time_ns: Optional end time filter (unix nano)

        Returns:
            List of trace event records
        """
        assert self.conn is not None

        query = 'SELECT * FROM run_trace_events WHERE run_id = ?'
        params: list[Any] = [run_id]

        if start_time_ns is not None:
            query += ' AND time_unix_nano >= ?'
            params.append(start_time_ns)
        if end_time_ns is not None:
            query += ' AND time_unix_nano <= ?'
            params.append(end_time_ns)

        query += ' ORDER BY time_unix_nano ASC'

        cursor = await self.conn.execute(query, tuple(params))
        rows = await cursor.fetchall()

        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row, strict=False)) for row in rows]

    # Run logs methods (new in v2)

    def enqueue_log_chunk(
        self,
        *,
        run_id: str,
        chunk_index: int,
        stream_type: str,
        start_time: str,
        end_time: str,
        content: str,
    ) -> bool:
        """Queue a log chunk for low-priority batched insertion.

        This method is thread-safe and intentionally fire-and-forget. It returns
        False if the database is already closing or unavailable.
        """
        return self._enqueue_low_priority_write(
            _LogWrite(
                run_id=run_id,
                chunk_index=chunk_index,
                stream_type=stream_type,
                start_time=start_time,
                end_time=end_time,
                size_bytes=len(content.encode('utf-8')),
                content=content,
                enqueued_at=time.monotonic(),
            )
        )

    async def insert_log_chunk(
        self,
        run_id: str,
        chunk_index: int,
        stream_type: str,
        start_time: str,
        end_time: str,
        content: str,
    ) -> None:
        """Insert a log chunk.

        Args:
            run_id: Run identifier
            chunk_index: Chunk index (0-based)
            stream_type: 'stdout', 'stderr', or 'log'
            start_time: ISO timestamp when chunk started
            end_time: ISO timestamp when chunk was flushed
            content: Log content
        """
        await self._enqueue_low_priority_write_and_wait(
            'insert_log_chunk',
            _LogWrite(
                run_id,
                chunk_index,
                stream_type,
                start_time,
                end_time,
                len(content.encode('utf-8')),
                content,
                time.monotonic(),
            ),
        )

    async def get_log_chunks(
        self,
        run_id: str,
        stream_type: str | None = None,
    ) -> list[dict]:
        """Query log chunks for a run, optionally filtered by stream type.

        Args:
            run_id: Run identifier
            stream_type: Optional stream type filter

        Returns:
            List of log chunk records
        """
        assert self.conn is not None

        if stream_type:
            cursor = await self.conn.execute(
                'SELECT * FROM run_logs WHERE run_id = ? AND stream_type = ? ORDER BY chunk_index',
                (run_id, stream_type),
            )
        else:
            cursor = await self.conn.execute(
                'SELECT * FROM run_logs WHERE run_id = ? ORDER BY chunk_index',
                (run_id,),
            )

        rows = await cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row, strict=False)) for row in rows]


__all__ = [
    'BatchDatabase',
    'SCHEMA_VERSION',
]
