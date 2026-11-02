"""Socket.IO utilities with batching support."""

from __future__ import annotations

import asyncio
import contextlib
import logging

from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from typing import Any

import socketio


class SocketIOEventLoggerAdapter(logging.LoggerAdapter):
    """Keep Socket.IO event traffic available at DEBUG without flooding INFO logs."""

    _DEBUG_PREFIXES = ('received event ', 'emitting event ', 'received ack ')

    def log(self, level: int, msg: object, *args: Any, **kwargs: Any) -> None:
        if level == logging.INFO and isinstance(msg, str) and msg.startswith(self._DEBUG_PREFIXES):
            level = logging.DEBUG
        super().log(level, msg, *args, **kwargs)


class SafeAsyncServer(socketio.AsyncServer):
    """Thread-safe subclass of socketio.AsyncServer.

    Overrides emit() and call() to serialize concurrent calls with a lock,
    preventing message interleaving when multiple tasks emit concurrently.

    Example:
        sio = SafeAsyncServer(async_mode='asgi')

        await sio.emit('event', data)  # Thread-safe
        sio.on('connect', handler)     # Works as normal
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._emit_lock = asyncio.Lock()

    async def emit(
        self,
        event: str,
        data: Any = None,
        to: str | None = None,
        room: str | None = None,
        skip_sid: str | list[str] | None = None,
        namespace: str | None = None,
        callback: Callable[..., Any] | None = None,
        ignore_queue: bool = False,
    ) -> None:
        """Thread-safe emit that serializes concurrent calls."""
        async with self._emit_lock:
            await super().emit(
                event,
                data=data,
                to=to,
                room=room,
                skip_sid=skip_sid,
                namespace=namespace,
                callback=callback,
                ignore_queue=ignore_queue,
            )

    async def call(
        self,
        event: str,
        data: Any = None,
        to: str | None = None,
        sid: str | None = None,
        namespace: str | None = None,
        timeout: float = 60,
        ignore_queue: bool = False,
    ) -> Any:
        """Thread-safe call — the internal emit is serialized by our emit() override.

        We must NOT acquire ``_emit_lock`` here because ``super().call()``
        internally calls ``self.emit()`` (our override), which acquires the
        same lock.  ``asyncio.Lock`` is **not** reentrant, so wrapping
        ``call()`` with the lock would deadlock.
        """
        return await super().call(
            event,
            data=data,
            to=to,
            sid=sid,
            namespace=namespace,
            timeout=timeout,
            ignore_queue=ignore_queue,
        )


class AsyncSocketIOEmitter:
    """Async Socket.IO emitter with batching support for broadcast scenarios.

    This class provides a reliable way to emit events to Socket.IO clients with:
    - Bounded queue to prevent memory exhaustion (backpressure via queue size)
    - Batched sending to reduce network overhead

    Note: This emitter uses broadcast (emit to all clients), so ack-based
    backpressure is not available. Queue overflow and emit failures are
    retained and surfaced by ``flush()``/``close()``.

    For thread-safe emit, pass a SafeAsyncServer instead of raw AsyncServer.

    Client-side usage:
        socket.on('event_name', (data) => {
            processData(data.items);
        });

    Example:
        sio = SafeAsyncServer(socketio.AsyncServer(...))
        emitter = AsyncSocketIOEmitter(sio, 'logs')

        # From sync context
        emitter.enqueue({'message': 'hello'})

        # Graceful shutdown
        await emitter.close()
    """

    def __init__(
        self,
        sio: socketio.AsyncServer | socketio.AsyncClient | SafeAsyncServer,
        event_name: str,
        max_queue_size: int = 1000,
        batch_size: int = 50,
        flush_interval: float = 0.1,
        emit_func: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
        drop_unbound_items: bool = False,
    ) -> None:
        """Initialize the emitter.

        Args:
            sio: The Socket.IO server instance (or SafeAsyncServer wrapper).
            event_name: The event name to emit.
            max_queue_size: Maximum number of items to buffer (default: 1000).
            batch_size: Maximum items to send in one batch (default: 50).
            flush_interval: Max seconds to wait before flushing a partial batch (default: 0.1).
            emit_func: Optional transport override for acknowledged or namespaced emission.
            drop_unbound_items: Drop pre-start items instead of making the emitter fail permanently.
        """
        self.event_name = event_name
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._emit_func = emit_func or sio.emit
        self._drop_unbound_items = drop_unbound_items

        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=max_queue_size)
        self._worker_task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._batch_lock = asyncio.Lock()
        self._last_emitted_batch_id = 0
        self._shutdown = False
        self._error: Exception | None = None

    async def start(self) -> None:
        """Bind the emitter to its owner loop before callbacks can arrive from threads."""
        loop = asyncio.get_running_loop()
        self._bind_loop(loop)
        self._shutdown = False
        self._ensure_worker_started(loop)

    def enqueue(self, data: dict[str, Any]) -> bool:
        """Add data to the emit queue.

        This method is safe to call from sync context when an event loop is running.

        Args:
            data: The data to emit.

        Returns:
            True if data was queued, False if queue is full or no event loop.
        """
        return self._enqueue(data, wait_for_enqueue=True)

    def enqueue_nowait(self, data: dict[str, Any]) -> bool:
        """Schedule data from any thread without waiting for the owner loop.

        Foreign-thread callers return once the owner-loop callback is scheduled.
        Any later queue overflow is retained and surfaced by ``flush()``/``close()``.
        """
        return self._enqueue(data, wait_for_enqueue=False)

    def _enqueue(self, data: dict[str, Any], *, wait_for_enqueue: bool) -> bool:
        if self._shutdown or self._error is not None:
            return False
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._loop is None:
            if current_loop is None:
                if not self._drop_unbound_items:
                    self._record_error(RuntimeError(f'{self.event_name} emitter requires a running event loop'))
                return False
            self._bind_loop(current_loop)

        if current_loop is self._loop:
            return self._enqueue_on_owner_loop(data)
        return self._enqueue_threadsafe(data, wait_for_enqueue=wait_for_enqueue)

    def _enqueue_on_owner_loop(self, data: dict[str, Any]) -> bool:
        if self._shutdown or self._error is not None:
            return False
        if self._loop is None:
            self._record_error(RuntimeError(f'{self.event_name} emitter has no owner event loop'))
            return False
        self._ensure_worker_started(self._loop)
        try:
            self._queue.put_nowait(data)
            return True
        except asyncio.QueueFull:
            self._record_error(RuntimeError(f'{self.event_name} emitter queue is full'))
            return False

    def _enqueue_threadsafe(self, data: dict[str, Any], *, wait_for_enqueue: bool) -> bool:
        loop = self._loop
        if loop is None or not loop.is_running() or loop.is_closed():
            self._record_error(RuntimeError(f'{self.event_name} emitter owner event loop is not running'))
            return False

        if not wait_for_enqueue:
            try:
                loop.call_soon_threadsafe(self._enqueue_on_owner_loop, data)
            except RuntimeError as exc:
                self._record_error(exc)
                return False
            return True

        result: Future[bool] = Future()

        def enqueue() -> None:
            try:
                result.set_result(self._enqueue_on_owner_loop(data))
            except BaseException as exc:
                result.set_exception(exc)

        try:
            loop.call_soon_threadsafe(enqueue)
            return result.result()
        except Exception as exc:
            self._record_error(exc)
            return False

    def _bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError(f'{self.event_name} emitter cannot move to another event loop')
        self._loop = loop

    def _ensure_worker_started(self, loop: asyncio.AbstractEventLoop) -> None:
        """Ensure the background worker task is running."""
        self._bind_loop(loop)
        if self._worker_task is None or self._worker_task.done():
            self._shutdown = False
            self._worker_task = loop.create_task(self._process_queue())

    async def _process_queue(self) -> None:
        """Background worker that batches and sends data with backpressure."""
        while not self._shutdown:
            batch: list[dict[str, Any]] = []

            # Wait for the first item
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=self._flush_interval)
                batch.append(first)
            except asyncio.TimeoutError:
                continue

            # Collect more items without blocking (up to batch_size)
            while len(batch) < self._batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            try:
                await self._emit_batch(batch)
            except asyncio.CancelledError:
                # Never swallow cancellation: otherwise this worker keeps looping forever and
                # asyncio's shutdown (gather over cancelled tasks) never completes.
                raise
            except Exception as exc:
                self._record_error(exc)
            finally:
                for _ in batch:
                    self._queue.task_done()

    async def _emit_batch(self, batch: list[dict[str, Any]], *, barrier: bool = False) -> None:
        """Broadcast a batch of data to all connected clients.

        Args:
            batch: List of data items to send.
        """
        async with self._batch_lock:
            batch_id = self._last_emitted_batch_id + 1
            payload: dict[str, Any] = {'batch_id': batch_id, 'items': batch}
            if barrier:
                payload['barrier'] = True
            emit_task = asyncio.ensure_future(self._emit_func(self.event_name, payload))
            try:
                await asyncio.shield(emit_task)
            except asyncio.CancelledError:
                if emit_task.cancelled():
                    # The transport task was cancelled without us asking (e.g. the event loop is
                    # shutting down and cancelling every task). Record it so flush()/close() still
                    # surface the failure, but keep propagating CancelledError: converting it into a
                    # regular exception makes this coroutine uncancellable and hangs loop teardown.
                    self._record_error(RuntimeError(f'{self.event_name} emitter transport was unexpectedly cancelled'))
                    raise
                if not emit_task.done():
                    emit_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await emit_task
                else:
                    # emit_task finished (success or exception) while *we* were cancelled.
                    try:
                        emit_task.result()
                        # Transport succeeded — honour the batch_id so subsequent
                        # batches do not reuse the same id and confuse the client.
                        self._last_emitted_batch_id = batch_id
                    except Exception as _transport_err:
                        self._record_error(_transport_err)
                raise
            self._last_emitted_batch_id = batch_id

    async def flush(self) -> None:
        """Wait for all accepted items and surface any transport failure."""
        loop = asyncio.get_running_loop()
        self._bind_loop(loop)
        self._raise_error()
        if not self._queue.empty() and (self._worker_task is None or self._worker_task.done()):
            self._ensure_worker_started(loop)
        await self._queue.join()
        self._raise_error()

    async def barrier(self) -> int:
        """Flush queued items and emit an ordered marker for downstream replay."""
        await self.flush()
        try:
            await self._emit_batch([], barrier=True)
        except Exception as exc:
            self._record_error(exc)
            raise
        return self._last_emitted_batch_id

    async def close(self) -> None:
        """Gracefully shutdown the emitter, flushing remaining data."""
        error: Exception | None = None
        try:
            await self.flush()
        except Exception as exc:
            error = exc
        finally:
            self._shutdown = True
            if self._worker_task is not None and not self._worker_task.done():
                self._worker_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._worker_task
        if error is not None:
            raise error

    def _record_error(self, error: Exception) -> None:
        if self._error is None:
            self._error = error

    def _raise_error(self) -> None:
        if self._error is not None:
            raise self._error
