"""Database-backed span processor for batch runner.

This module provides a SpanProcessor that writes trace events to the batch
runner's SQLite database through BatchDatabase's prioritized writer. A small
global queue/thread is kept only to serialize trace payloads off the main event
loop before enqueueing low-priority database writes.
"""

from __future__ import annotations

import json
import logging
import queue
import threading

from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nexus.trace.events import SpanEndEvent, SpanStartEvent, SpanUpdateEvent
from nexus.trace.processors.base import SpanProcessor, span_processor_registry
from nexus.trace.types import Span


if TYPE_CHECKING:
    from nexus.runner.batch.database import BatchDatabase

logger = logging.getLogger(__name__)


# ContextVar for batch database context
_batch_db_context: ContextVar[BatchDatabaseContext | None] = ContextVar('batch_db_context', default=None)


# Global trace serialization queue and worker thread (shared by all processor instances)
_global_queue: queue.Queue | None = None
_global_writer_thread: threading.Thread | None = None
_writer_lock = threading.Lock()
_shutdown_requested = False


@dataclass
class BatchDatabaseContext:
    """Context for batch runner database access."""

    database: BatchDatabase
    run_id: str


def _ensure_global_writer():
    """Ensure global trace serialization thread is started."""
    global _global_queue, _global_writer_thread

    with _writer_lock:
        if _global_queue is None:
            _global_queue = queue.Queue()
            logger.debug('Created global trace event queue')

        if _global_writer_thread is None or not _global_writer_thread.is_alive():
            _global_writer_thread = threading.Thread(target=_writer_loop, daemon=True, name='TraceWriter')
            _global_writer_thread.start()
            logger.debug('Started global trace serialization thread')


def _writer_loop():
    """Global trace serialization thread loop.

    Reads trace events from the global queue, serializes the payload, and
    delegates the actual SQLite write to BatchDatabase's low-priority writer.
    """
    while not _shutdown_requested:
        if _global_queue is None:
            break
        try:
            item = _global_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if item is None:  # Sentinel value
            _global_queue.task_done()
            break

        try:
            item_type = item[0]
            if item_type != 'trace':
                continue

            # Trace event: (type, ctx, span, event)
            _, ctx, span, event = item
            if ctx.database.conn is None:
                continue

            _enqueue_trace_event(ctx, span, event)
        except Exception:
            logger.exception('Failed to enqueue batch trace event')
        finally:
            if _global_queue is not None:
                _global_queue.task_done()


def _enqueue_trace_event(ctx: BatchDatabaseContext, span: Span, event) -> None:
    """Serialize and queue a trace event using span information."""
    # Create a modified event dict with trace_id and parent_span_id from span
    event_dict = event.model_dump()
    event_dict['trace_id'] = span.trace_id
    event_dict['parent_span_id'] = span.parent_span_id

    # Reconstruct the event with complete information
    event_type_name = type(event).__name__
    if event_type_name == 'SpanStartEvent':
        event_type = 'start'
    elif event_type_name == 'SpanUpdateEvent':
        event_type = 'update'
    elif event_type_name == 'SpanEndEvent':
        event_type = 'end'
    else:
        raise ValueError(f'Unknown event type: {event_type_name}')

    ctx.database.enqueue_trace_record(
        run_id=ctx.run_id,
        event_type=event_type,
        time_unix_nano=event.time_unix_nano,
        trace_id=span.trace_id,
        span_id=span.span_id,
        parent_span_id=span.parent_span_id,
        event_data=json.dumps(event_dict, default=str),
    )


@span_processor_registry.register('batch_trace')
class DatabaseSpanProcessor(SpanProcessor):
    """SpanProcessor that writes trace events to batch runner's database.

    Uses ContextVar to access database instance (set by batch runner).
    All instances share a single global queue and writer thread for efficiency.
    TODO: migrate this writer to the standard RunTraceSink/TraceSink path
    after batch adopts the unified run trace session model.
    """

    def __init__(self):
        """Initialize processor. Database context retrieved from ContextVar."""
        self._ctx = _batch_db_context.get()
        if self._ctx is None:
            raise RuntimeError(
                'DatabaseSpanProcessor requires batch database context. Use batch_database_context() context manager.'
            )

        # Ensure global writer thread is running
        _ensure_global_writer()

        logger.debug(f'Created DatabaseSpanProcessor for run_id: {self._ctx.run_id}')

    def on_start(self, span: Span, event: SpanStartEvent) -> None:
        """Called when a span is started."""
        assert _global_queue is not None
        _global_queue.put(('trace', self._ctx, span, event))

    def on_update(self, span: Span, event: SpanUpdateEvent) -> None:
        """Called when a span is updated."""
        assert _global_queue is not None
        _global_queue.put(('trace', self._ctx, span, event))

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        """Called when a span is ended."""
        assert _global_queue is not None
        _global_queue.put(('trace', self._ctx, span, event))

    def finalize(self) -> None:
        """Flush remaining events (wait for queue to be processed)."""
        if _global_queue is not None:
            # Wait for all tasks to complete (with short timeout to avoid hanging)
            try:
                # Use a thread to wait with timeout
                import time

                start_time = time.time()
                while not _global_queue.empty() and time.time() - start_time < 5.0:
                    time.sleep(0.1)
            except Exception:
                pass  # Ignore errors during finalize


def flush_trace_queue(timeout: float = 5.0) -> bool:
    """Wait until queued trace events have been handed to the database writer."""
    queue_ref = _global_queue
    if queue_ref is None:
        return True

    done = threading.Event()

    def wait_for_join() -> None:
        queue_ref.join()
        done.set()

    waiter = threading.Thread(target=wait_for_join, daemon=True, name='TraceQueueFlush')
    waiter.start()
    flushed = done.wait(timeout)
    if not flushed:
        logger.warning('Timed out waiting %.1fs for batch trace queue to flush', timeout)
    return flushed


class batch_database_context:  # noqa: N801
    """Context manager to set batch database context.

    This makes the database instance available to DatabaseSpanProcessor via ContextVar.

    Example:
        with batch_database_context(db, run_id):
            # DatabaseSpanProcessor can now access db
            processor = DatabaseSpanProcessor()
    """

    def __init__(self, database: BatchDatabase, run_id: str):
        self._ctx = BatchDatabaseContext(database, run_id)
        self._token = None

    def __enter__(self):
        self._token = _batch_db_context.set(self._ctx)
        return self._ctx

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token is not None:
            _batch_db_context.reset(self._token)


def enqueue_log_chunk(database, run_id: str, chunk_index: int, start_time: str, end_time: str, content: str):
    """Enqueue a log chunk to be written to the database.

    This function can be called from any thread. The BatchDatabase writer handles
    the actual low-priority batched database write.

    Args:
        database: BatchDatabase instance
        run_id: Run identifier
        chunk_index: Chunk index
        start_time: ISO timestamp when chunk started
        end_time: ISO timestamp when chunk ended
        content: Log content
    """
    database.enqueue_log_chunk(
        run_id=run_id,
        chunk_index=chunk_index,
        stream_type='log',
        start_time=start_time,
        end_time=end_time,
        content=content,
    )


__all__ = [
    'DatabaseSpanProcessor',
    'BatchDatabaseContext',
    'batch_database_context',
    'enqueue_log_chunk',
    'flush_trace_queue',
]
