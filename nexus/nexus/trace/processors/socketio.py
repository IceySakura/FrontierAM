"""Socket.IO Span Broadcaster for distributed tracing.

This processor broadcasts span events to all connected Socket.IO clients,
enabling real-time trace visualization and distributed span propagation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nexus.trace.processors.base import SpanProcessor
from nexus.trace.types import Span
from nexus.utils.truncate import truncate_recursive


if TYPE_CHECKING:
    import socketio

    from nexus.trace.events import SpanEndEvent, SpanStartEvent, SpanUpdateEvent


# 256KB max size for truncatable fields
_MAX_FIELD_SIZE = 256 * 1024
DEFAULT_SPAN_QUEUE_SIZE = 2000


def truncate_span_event_data(data: dict[str, Any]) -> dict[str, Any]:
    """Truncate large fields (attributes) in span event data.

    Uses recursive truncation to preserve data structure while truncating
    large string values.
    """
    result = data.copy()

    # Truncate attributes recursively
    if 'attributes' in result and result['attributes']:
        result['attributes'] = truncate_recursive(result['attributes'], max_size=_MAX_FIELD_SIZE)

    return result


class SocketIOSpanBroadcaster(SpanProcessor):
    """Broadcasts span events to all connected Socket.IO clients.

    This processor is used on the runtime server side to forward span events
    back to the client, enabling unified distributed tracing across client
    and server.

    Features:
    - Batched sending to reduce network overhead
    - Bounded queue to prevent memory exhaustion (backpressure)
    - Automatic truncation of large attributes/events (256KB limit)

    Example:
        sio = socketio.AsyncServer(...)
        broadcaster = SocketIOSpanBroadcaster(sio)
        add_span_processor(broadcaster)

        # All span events will now be broadcast to connected clients
        # Clients can listen for 'span' events to receive them

    Client-side usage:
        socket.on('span', (data) => {
            processSpans(data.items);
        });
    """

    def __init__(
        self,
        sio: socketio.AsyncServer,
        event_name: str = 'span',
        max_queue_size: int = DEFAULT_SPAN_QUEUE_SIZE,
        batch_size: int = 50,
        flush_interval: float = 0.1,
    ):
        """Initialize the broadcaster.

        Args:
            sio: The Socket.IO async server instance
            event_name: The event name to emit (default: 'span')
            max_queue_size: Maximum number of span events to buffer (default: 2000).
            batch_size: Maximum events to send in one batch (default: 50).
            flush_interval: Max seconds to wait before flushing a partial batch (default: 0.1).
        """
        from nexus.utils.socketio import AsyncSocketIOEmitter

        self._emitter = AsyncSocketIOEmitter(
            sio=sio,
            event_name=event_name,
            max_queue_size=max_queue_size,
            batch_size=batch_size,
            flush_interval=flush_interval,
        )

    def enqueue_event(self, data: dict[str, Any]) -> bool:
        """Truncate and enqueue one serialized span event."""
        return self._emitter.enqueue(truncate_span_event_data(data))

    def on_start(self, span: Span, event: SpanStartEvent) -> None:  # noqa: ARG002
        """Broadcast span start event."""
        self.enqueue_event(event.model_dump())

    def on_update(self, span: Span, event: SpanUpdateEvent) -> None:  # noqa: ARG002
        """Broadcast span update event."""
        self.enqueue_event(event.model_dump())

    def on_end(self, span: Span, event: SpanEndEvent) -> None:  # noqa: ARG002
        """Broadcast span end event."""
        self.enqueue_event(event.model_dump())

    async def start(self) -> None:
        """Bind the broadcaster to the runtime server event loop."""
        await self._emitter.start()

    async def flush(self) -> None:
        """Wait for all accepted span events to be emitted."""
        await self._emitter.flush()

    async def barrier(self) -> int:
        """Emit a sequenced marker after all accepted span events."""
        return await self._emitter.barrier()

    async def close(self) -> None:
        """Gracefully shutdown the broadcaster, flushing remaining events."""
        await self._emitter.close()
