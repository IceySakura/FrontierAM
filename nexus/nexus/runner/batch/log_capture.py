"""Logging handler for capturing logs to database with chunking.

This module provides a logging handler that captures application logs and writes
them to the batch runner's database in chunks (4MB or 1 minute intervals).

Logs are queued to the global database writer thread shared with trace events.
A single global handler routes logs to different buffers based on context variable.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time

from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
from typing import TYPE_CHECKING

from nexus.runner.batch.trace_processor import enqueue_log_chunk


if TYPE_CHECKING:
    from nexus.runner.batch.database import BatchDatabase

logger = logging.getLogger(__name__)

# ContextVar for log capture context
_log_capture_context: ContextVar[LogCaptureContext | None] = ContextVar('log_capture_context', default=None)

# Global handler and flusher
_global_handler: ContextAwareLogHandler | None = None
_global_flusher_thread: threading.Thread | None = None
_flusher_lock = threading.Lock()
_flusher_running = False
_handler_lock = threading.Lock()

# Registry of active contexts for time-based flushing
_active_contexts: set[LogCaptureContext] = set()
_active_contexts_lock = threading.Lock()


@dataclass(unsafe_hash=True)
class LogCaptureContext:
    """Context for capturing logs for a specific run."""

    database: BatchDatabase = field(hash=False, compare=False)
    run_id: str
    buffer: StringIO = field(hash=False, compare=False)
    buffer_lock: threading.Lock = field(hash=False, compare=False)
    chunk_index: int = field(hash=False, compare=False)
    chunk_start_time: datetime = field(hash=False, compare=False)
    chunk_size_bytes: int = field(hash=False, compare=False)
    chunk_time_seconds: float = field(hash=False, compare=False)
    max_record_size: int = field(hash=False, compare=False)
    # Separate size limit for records containing grading markers.
    max_record_size_exempt: int = field(default=10_000, hash=False, compare=False)
    buffer_size_bytes: int = field(default=0, hash=False, compare=False)


# Apply the grading-record limit to reports, test statuses, and restoration diagnostics.
_NO_TRUNCATE_MARKERS = (
    'report_json',
    'OMNIGRIL_EXIT_CODE',
    '[ANTI-TAMPER]',
    'tests_status',
)


def _resolve_exempt_size(base: int) -> int:
    """Resolve the grading-record limit from NEXUS_LOG_EXEMPT_SIZE or its default.

    A non-positive configured value restores the ordinary record limit.
    """
    raw = os.environ.get('NEXUS_LOG_EXEMPT_SIZE')
    if raw is None or raw.strip() == '':
        return base
    try:
        v = int(raw)
    except ValueError:
        logger.warning('NEXUS_LOG_EXEMPT_SIZE=%r 无法解析为整数，沿用普通记录上限 %s', raw, base)
        return base
    # A non-positive value disables the separate grading-record limit.
    return v if v > 0 else base


def _ensure_global_handler():
    """Ensure global handler is installed on root logger."""
    global _global_handler

    with _handler_lock:
        if _global_handler is None:
            _global_handler = ContextAwareLogHandler()
            # Set formatter
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
            )
            _global_handler.setFormatter(formatter)

            # Add to root logger
            root_logger = logging.getLogger()
            root_logger.addHandler(_global_handler)

            logger.debug('Installed global context-aware log handler')


def _ensure_global_flusher():
    """Ensure global flusher thread is started."""
    global _global_flusher_thread, _flusher_running

    with _flusher_lock:
        if not _flusher_running:
            _flusher_running = True
            _global_flusher_thread = threading.Thread(target=_global_flusher_loop, daemon=True, name='GlobalLogFlusher')
            _global_flusher_thread.start()


def _global_flusher_loop():
    """Global flusher loop - periodically flushes all active contexts."""
    while _flusher_running:
        time.sleep(10)  # Check every 10 seconds

        # Get snapshot of active contexts
        with _active_contexts_lock:
            contexts_to_check = list(_active_contexts)

        # Check each active context for time-based flush
        for ctx in contexts_to_check:
            with contextlib.suppress(Exception):
                _check_time_based_flush(ctx)


def _check_time_based_flush(ctx: LogCaptureContext) -> None:
    """Check if time threshold reached and flush if needed.

    Called by global flusher thread periodically.

    Args:
        ctx: LogCaptureContext to check and potentially flush
    """
    elapsed = (datetime.now(timezone.utc) - ctx.chunk_start_time).total_seconds()
    if elapsed >= ctx.chunk_time_seconds:
        _flush_context(ctx)


def _flush_context(ctx: LogCaptureContext) -> None:
    """Flush context's buffer to database via global queue.

    This method can be called from any thread.

    Args:
        ctx: LogCaptureContext to flush
    """
    with ctx.buffer_lock:
        content = ctx.buffer.getvalue()
        if not content:
            return

        end_time = datetime.now(timezone.utc)
        chunk_index = ctx.chunk_index
        start_time = ctx.chunk_start_time.isoformat()

        # Reset buffer before enqueueing so any logging triggered during enqueue
        # writes into the new buffer instead of recursively re-entering this flush.
        ctx.buffer = StringIO()
        ctx.buffer_size_bytes = 0
        ctx.chunk_index += 1
        ctx.chunk_start_time = end_time

    # Enqueue outside the buffer lock to avoid self-deadlocks if enqueue/setup logs.
    with contextlib.suppress(Exception):
        enqueue_log_chunk(
            database=ctx.database,
            run_id=ctx.run_id,
            chunk_index=chunk_index,
            start_time=start_time,
            end_time=end_time.isoformat(),
            content=content,
        )


class ContextAwareLogHandler(logging.Handler):
    """Global logging handler that routes logs to per-run buffers based on context variable.

    This handler is installed once on the root logger and routes logs to different buffers
    based on the current LogCaptureContext in the context variable.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the appropriate context buffer.

        Args:
            record: Log record to emit
        """
        # Get the current context
        ctx = _log_capture_context.get()
        if ctx is None:
            # No context set, ignore this log
            return

        try:
            # Format the record
            msg = self.format(record)

            # Use the configured grading-record limit for matching records; apply the ordinary limit to other logs.
            limit = ctx.max_record_size
            if limit and any(marker in msg for marker in _NO_TRUNCATE_MARKERS):
                limit = max(limit, ctx.max_record_size_exempt)
            if limit and len(msg) > limit:
                # Preserve the start and end, where grading summaries may appear.
                head = limit * 2 // 3
                tail = limit - head
                msg = (
                    msg[:head]
                    + f' ... (log record truncated: {len(msg)} -> {limit} chars, middle elided) ... '
                    + msg[-tail:]
                )

            # Check if we need to flush (size threshold)
            should_flush = False
            with ctx.buffer_lock:
                # Write to context's buffer
                line = msg + '\n'
                ctx.buffer.write(line)

                # Incrementally track buffer size (use char length as approximation)
                ctx.buffer_size_bytes += len(line)
                if ctx.buffer_size_bytes >= ctx.chunk_size_bytes:
                    should_flush = True

            # Flush outside the lock to avoid blocking the handler
            if should_flush:
                _flush_context(ctx)

        except Exception:
            self.handleError(record)


class log_capture_context:  # noqa: N801
    """Context manager for log capture.

    This sets up a LogCaptureContext in the context variable, which the global
    handler uses to route logs to the appropriate buffer.

    Example:
        async with log_capture_context(db, run_id):
            # All logs from this block will be captured for this run_id
            logger.info('This will be captured')
    """

    def __init__(
        self,
        database: BatchDatabase,
        run_id: str,
        dual_write: bool = True,  # noqa: ARG002
        chunk_size_mb: float = 4.0,
        chunk_time_seconds: float = 60.0,
        max_record_size: int = 10_000,
        # Marker-based expansion is opt-in; an unset or nonpositive environment value keeps the ordinary limit.
        max_record_size_exempt: int | None = None,
    ):
        """Initialize log capture context.

        Args:
            database: BatchDatabase instance
            run_id: Run identifier
            dual_write: Ignored (kept for backward compatibility)
            chunk_size_mb: Max chunk size in MB
            chunk_time_seconds: Max time before flush in seconds
            max_record_size: Max size per log record in chars
        """
        self.ctx = LogCaptureContext(
            database=database,
            run_id=run_id,
            buffer=StringIO(),
            buffer_lock=threading.Lock(),
            chunk_index=0,
            chunk_start_time=datetime.now(timezone.utc),
            chunk_size_bytes=int(chunk_size_mb * 1024 * 1024),
            chunk_time_seconds=chunk_time_seconds,
            max_record_size=max_record_size,
            max_record_size_exempt=(
                max_record_size_exempt if max_record_size_exempt is not None else _resolve_exempt_size(max_record_size)
            ),
        )
        self.token = None

    async def __aenter__(self):
        """Enter the context (start log capture)."""
        # Ensure global handler and flusher are installed
        _ensure_global_handler()
        _ensure_global_flusher()

        # Register context for time-based flushing
        with _active_contexts_lock:
            _active_contexts.add(self.ctx)

        # Set context variable
        self.token = _log_capture_context.set(self.ctx)

        return self.ctx

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ARG002
        """Exit the context (stop log capture and flush)."""
        try:
            # Reset context variable
            if self.token is not None:
                _log_capture_context.reset(self.token)

            # Final flush after resetting context so flush-internal logs are not
            # captured back into the same per-run buffer.
            _flush_context(self.ctx)

            # Unregister context
            with _active_contexts_lock:
                _active_contexts.discard(self.ctx)
        except Exception:
            # Silently ignore all errors during cleanup
            pass


__all__ = [
    'LogCaptureContext',
    'ContextAwareLogHandler',
    'log_capture_context',
]
