"""Custom logging handlers for the application."""

from __future__ import annotations

import contextvars
import logging

from typing import TYPE_CHECKING, Any

from nexus.utils.socketio import AsyncSocketIOEmitter
from nexus.utils.truncate import truncate_string


if TYPE_CHECKING:
    import socketio


# Context variable to prevent recursive log emission
_emitting_log: contextvars.ContextVar[bool] = contextvars.ContextVar('_emitting_log', default=False)


class TqdmStreamHandler(logging.StreamHandler):
    """Stream handler that plays nicely with tqdm progress bars."""

    def emit(self, record):
        from tqdm import tqdm

        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


class _LogEmitter(AsyncSocketIOEmitter):
    """Socket.IO emitter with recursion prevention for logging."""

    async def _emit_batch(self, batch: list[dict[str, Any]], *, barrier: bool = False) -> None:
        """Override to set recursion prevention flag during emit."""
        token = _emitting_log.set(True)
        try:
            await super()._emit_batch(batch, barrier=barrier)
        finally:
            _emitting_log.reset(token)


class SocketIOHandler(logging.Handler):
    """Logging handler that broadcasts log records to Socket.IO clients.

    This handler sends log events to connected clients via Socket.IO,
    enabling real-time log monitoring from remote clients.

    Features:
    - Batched sending to reduce network overhead
    - Bounded queue to prevent memory exhaustion (backpressure)
    - Contextvar to prevent recursive logging

    Client-side usage:
        socket.on('log', (data) => {
            renderLogs(data.items);
        });
    """

    def __init__(
        self,
        sio: socketio.AsyncServer,
        event_name: str = 'log',
        max_queue_size: int = 1000,
        batch_size: int = 50,
        flush_interval: float = 0.1,
    ) -> None:
        """Initialize the Socket.IO logging handler.

        Args:
            sio: The Socket.IO server instance to use for broadcasting.
            event_name: The event name to emit (default: 'log').
            max_queue_size: Maximum number of log records to buffer (default: 1000).
            batch_size: Maximum number of logs to send in one batch (default: 50).
            flush_interval: Maximum seconds to wait before flushing a partial batch (default: 0.1).
        """
        super().__init__()
        self._emitter = _LogEmitter(
            sio=sio,
            event_name=event_name,
            max_queue_size=max_queue_size,
            batch_size=batch_size,
            flush_interval=flush_interval,
            drop_unbound_items=True,
        )

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to all connected Socket.IO clients.

        Args:
            record: The log record to emit.
        """
        if _emitting_log.get():
            return

        # Records marked skip_sio should print locally but not broadcast to clients.
        # Used by Base server to relay worker logs to stdout without duplicating to Host.
        if getattr(record, 'skip_sio', False):
            return

        try:
            message = truncate_string(record.getMessage())
            log_data = {
                'level': record.levelname,
                'name': record.name,
                'message': message,
                'timestamp': record.created,
                'pathname': record.pathname,
                'lineno': record.lineno,
                'funcName': record.funcName,
            }
            self._emitter.enqueue_nowait(log_data)
        except Exception:
            self.handleError(record)

    async def close_async(self) -> None:
        """Gracefully shutdown the handler, flushing remaining logs."""
        await self._emitter.close()

    async def start_async(self) -> None:
        """Bind the log emitter to the application event loop."""
        await self._emitter.start()
