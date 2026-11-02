from __future__ import annotations

import threading
import time

from nexus.trace.dispatcher import BackgroundSpanDispatcher, HookItem
from nexus.trace.events import SpanStartEvent
from nexus.trace.processors.base import SpanProcessor
from nexus.trace.types import Span


def _make_span(span_id: str, attributes: dict | None = None) -> Span:
    return Span(
        trace_id='trace',
        span_id=span_id,
        parent_span_id='',
        name='op',
        start_time_unix_nano=0,
        attributes=attributes or {},
    )


def _start_event() -> SpanStartEvent:
    return SpanStartEvent(parent_span_id='', name='op')


def test_thread_not_started_until_submit() -> None:
    dispatcher = BackgroundSpanDispatcher()
    assert dispatcher._thread is None


def test_submit_invokes_hooks_in_fifo_order() -> None:
    dispatcher = BackgroundSpanDispatcher()
    calls: list[str] = []

    class P(SpanProcessor):
        def on_start(self, span: Span, event) -> None:  # noqa: ANN001
            calls.append(span.span_id)

    p = P()
    for i in range(3):
        dispatcher.submit(HookItem('on_start', _make_span(str(i)), _start_event(), [p]))

    assert dispatcher.barrier(timeout=2.0)
    assert calls == ['0', '1', '2']


def test_processor_exception_does_not_block_others_or_kill_thread() -> None:
    dispatcher = BackgroundSpanDispatcher()
    calls: list[str] = []

    class Bad(SpanProcessor):
        def on_start(self, span: Span, event) -> None:  # noqa: ANN001
            raise RuntimeError('boom')

    class Good(SpanProcessor):
        def on_start(self, span: Span, event) -> None:  # noqa: ANN001
            calls.append(span.span_id)

    dispatcher.submit(HookItem('on_start', _make_span('x'), _start_event(), [Bad(), Good()]))
    assert dispatcher.barrier(timeout=2.0)
    assert calls == ['x']

    # Thread survived the exception and keeps processing.
    dispatcher.submit(HookItem('on_start', _make_span('y'), _start_event(), [Good()]))
    assert dispatcher.barrier(timeout=2.0)
    assert calls == ['x', 'y']


def test_barrier_waits_for_prior_events_to_finish() -> None:
    dispatcher = BackgroundSpanDispatcher()
    order: list[str] = []
    release = threading.Event()

    class Slow(SpanProcessor):
        def on_start(self, span: Span, event) -> None:  # noqa: ANN001
            release.wait(3.0)
            order.append('hook')

    dispatcher.submit(HookItem('on_start', _make_span('s'), _start_event(), [Slow()]))

    def releaser() -> None:
        time.sleep(0.2)
        order.append('release')
        release.set()

    t = threading.Thread(target=releaser)
    t.start()

    assert dispatcher.barrier(timeout=3.0)
    order.append('after_barrier')
    t.join()

    assert order == ['release', 'hook', 'after_barrier']


def test_full_queue_blocks_submit_until_drained() -> None:
    dispatcher = BackgroundSpanDispatcher(max_queue_size=1)
    gate = threading.Event()

    class Block(SpanProcessor):
        def on_start(self, span: Span, event) -> None:  # noqa: ANN001
            gate.wait(3.0)

    p = Block()

    # Item 1 is picked up by the worker, which blocks inside the hook.
    dispatcher.submit(HookItem('on_start', _make_span('1'), _start_event(), [p]))
    time.sleep(0.1)
    # Item 2 fills the bounded queue (maxsize=1).
    dispatcher.submit(HookItem('on_start', _make_span('2'), _start_event(), [p]))

    # Item 3 must block because the queue is full and the worker is busy.
    unblocked = threading.Event()

    def third() -> None:
        dispatcher.submit(HookItem('on_start', _make_span('3'), _start_event(), [p]))
        unblocked.set()

    t = threading.Thread(target=third)
    t.start()

    assert not unblocked.wait(0.5)  # still blocked on a full queue
    gate.set()  # release the worker -> queue drains -> submit unblocks
    assert unblocked.wait(3.0)
    t.join()


def test_barrier_times_out_when_queue_full_and_worker_blocked() -> None:
    dispatcher = BackgroundSpanDispatcher(max_queue_size=1)
    gate = threading.Event()

    class Block(SpanProcessor):
        def on_start(self, span: Span, event) -> None:  # noqa: ANN001
            gate.wait(5.0)

    p = Block()
    # Item 1 is picked up by the worker, which blocks inside the hook.
    dispatcher.submit(HookItem('on_start', _make_span('1'), _start_event(), [p]))
    time.sleep(0.1)
    # Item 2 fills the bounded queue (maxsize=1); the worker is still blocked.
    dispatcher.submit(HookItem('on_start', _make_span('2'), _start_event(), [p]))

    # The barrier sentinel cannot be enqueued (queue full, worker blocked).
    # It must return False within ~timeout rather than hang forever.
    start = time.monotonic()
    result = dispatcher.barrier(timeout=0.5)
    elapsed = time.monotonic() - start

    gate.set()  # release the worker so the daemon thread can finish cleanly
    assert result is False
    assert elapsed < 2.0  # honored the timeout, did not block indefinitely
