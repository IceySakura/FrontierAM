from __future__ import annotations

import queue
import threading

from dataclasses import dataclass
from logging import getLogger
from typing import Literal, Protocol, runtime_checkable

from nexus.trace.events import SpanEvent
from nexus.trace.processors import SpanProcessor
from nexus.trace.types import Span


logger = getLogger(__name__)


@dataclass
class HookItem:
    """A resolved hook invocation to run through a span dispatcher."""

    hook: Literal['on_start', 'on_update', 'on_end']
    span: Span  # deep-copied snapshot taken at enqueue time
    event: SpanEvent
    processors: list[SpanProcessor]  # processor list resolved at enqueue time


@dataclass
class BarrierItem:
    """A sentinel used to drain the queue up to this point."""

    done: threading.Event


@runtime_checkable
class SpanDispatcherProtocol(Protocol):
    """Common dispatch interface for span processor hook execution."""

    def submit(self, item: HookItem) -> None: ...

    def barrier(self, timeout: float = 5.0) -> bool: ...


class BackgroundSpanDispatcher(SpanDispatcherProtocol):
    """Single dedicated thread that invokes span processor hooks off the caller's thread.

    Items are processed strictly FIFO. A single processor raising an exception is
    logged and isolated; it neither skips sibling processors in the same item nor
    kills the worker thread.
    """

    def __init__(self, max_queue_size: int = 10000):
        self._queue: queue.Queue[HookItem | BarrierItem] = queue.Queue(maxsize=max_queue_size)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def submit(self, item: HookItem) -> None:
        """Enqueue a hook item, lazily starting the worker thread. Blocks if the queue is full."""
        self._ensure_thread()
        self._queue.put(item)

    def barrier(self, timeout: float = 5.0) -> bool:
        """Enqueue a sentinel and block until the worker has processed everything before it.

        Returns True if the sentinel was reached within ``timeout``, False otherwise.
        If the worker thread was never started, the queue is empty and this returns
        True immediately without starting it. If the queue is full and the sentinel
        cannot be enqueued within ``timeout``, returns False rather than blocking
        indefinitely.
        """
        if self._thread is None:
            return True
        ev = threading.Event()
        try:
            self._queue.put(BarrierItem(ev), timeout=timeout)
        except queue.Full:
            logger.warning('BackgroundSpanDispatcher.barrier: queue full, could not enqueue barrier sentinel')
            return False
        return ev.wait(timeout)

    def _ensure_thread(self) -> None:
        if self._thread is not None:
            return
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name='span-dispatcher', daemon=True)
                self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if isinstance(item, BarrierItem):
                item.done.set()
                continue
            for processor in item.processors:
                try:
                    getattr(processor, item.hook)(item.span, item.event)
                except Exception:
                    logger.exception('Error in processor %s.%s', type(processor).__name__, item.hook)


class InlineSpanDispatcher(SpanDispatcherProtocol):
    """Invoke span processor hooks immediately on the caller's thread."""

    def submit(self, item: HookItem) -> None:
        for processor in item.processors:
            try:
                getattr(processor, item.hook)(item.span, item.event)
            except Exception:
                logger.exception('Error in processor %s.%s', type(processor).__name__, item.hook)

    def barrier(self, timeout: float = 5.0) -> bool:  # noqa: ARG002
        return True
