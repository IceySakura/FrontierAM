from __future__ import annotations

import threading
import time

from nexus.trace.controller import SpanController
from nexus.trace.dispatcher import BackgroundSpanDispatcher, InlineSpanDispatcher, SpanDispatcherProtocol
from nexus.trace.events import SpanEndEvent, SpanStartEvent, SpanUpdateEvent
from nexus.trace.processors.base import SpanProcessor
from nexus.trace.types import Span


def test_dispatchers_share_protocol_contract() -> None:
    assert isinstance(BackgroundSpanDispatcher(), SpanDispatcherProtocol)
    assert isinstance(InlineSpanDispatcher(), SpanDispatcherProtocol)


def test_add_event_does_not_block_on_slow_hook() -> None:
    controller = SpanController()
    gate = threading.Event()

    class Slow(SpanProcessor):
        dispatch_mode = 'background'

        def on_start(self, span: Span, event) -> None:  # noqa: ANN001
            gate.wait(5.0)

    controller.add_processor(Slow())

    ev = SpanStartEvent(parent_span_id='', name='op')
    start = time.monotonic()
    controller.add_event(ev)
    elapsed = time.monotonic() - start

    gate.set()
    assert elapsed < 0.5


def test_active_span_dict_updated_synchronously() -> None:
    controller = SpanController()
    ev = SpanStartEvent(parent_span_id='', name='op')
    controller.add_event(ev)
    assert ev.span_id in controller.active_span_dict

    controller.add_event(SpanEndEvent(span_id=ev.span_id))
    assert ev.span_id not in controller.active_span_dict


def test_processor_sees_span_snapshot_at_event_time() -> None:
    controller = SpanController()
    seen: dict[str, str] = {}
    release = threading.Event()

    class Capture(SpanProcessor):
        dispatch_mode = 'background'

        def on_start(self, span: Span, event) -> None:  # noqa: ANN001
            release.wait(2.0)
            seen['k'] = span.attributes.get('k', '')

    controller.add_processor(Capture())

    ev = SpanStartEvent(parent_span_id='', name='op', attributes={'k': 'v0'})
    controller.add_event(ev)
    # Mutate the live span before the (still-blocked) on_start hook runs.
    controller.add_event(SpanUpdateEvent(span_id=ev.span_id, attributes={'k': 'v1'}))
    release.set()

    assert controller.background_dispatcher.barrier(timeout=2.0)
    assert seen['k'] == 'v0'  # snapshot, not the later-mutated value


def test_event_order_preserved_per_span() -> None:
    controller = SpanController()
    hooks: list[str] = []

    class Rec(SpanProcessor):
        def on_start(self, span: Span, event) -> None:  # noqa: ANN001
            hooks.append('start')

        def on_update(self, span: Span, event) -> None:  # noqa: ANN001
            hooks.append('update')

        def on_end(self, span: Span, event) -> None:  # noqa: ANN001
            hooks.append('end')

    controller.add_processor(Rec())

    ev = SpanStartEvent(parent_span_id='', name='op')
    controller.add_event(ev)
    controller.add_event(SpanUpdateEvent(span_id=ev.span_id, attributes={'a': 1}))
    controller.add_event(SpanEndEvent(span_id=ev.span_id))

    assert controller.background_dispatcher.barrier(timeout=2.0)
    assert hooks == ['start', 'update', 'end']


def test_no_processors_does_not_start_thread() -> None:
    controller = SpanController()
    controller.add_event(SpanStartEvent(parent_span_id='', name='op'))
    assert controller.background_dispatcher._thread is None


def test_inline_processor_runs_on_calling_thread() -> None:
    controller = SpanController()
    caller_thread_id = threading.get_ident()
    seen_thread_ids: list[int] = []

    class DefaultInline(SpanProcessor):
        def on_start(self, span: Span, event) -> None:  # noqa: ANN001
            seen_thread_ids.append(threading.get_ident())

    controller.add_processor(DefaultInline())
    controller.add_event(SpanStartEvent(parent_span_id='', name='op'))

    assert seen_thread_ids == [caller_thread_id]
    assert controller.background_dispatcher._thread is None


def test_background_processor_still_uses_dispatcher_thread() -> None:
    controller = SpanController()
    caller_thread_id = threading.get_ident()
    seen_thread_ids: list[int] = []

    class Background(SpanProcessor):
        dispatch_mode = 'background'

        def on_start(self, span: Span, event) -> None:  # noqa: ANN001
            seen_thread_ids.append(threading.get_ident())

    controller.add_processor(Background())
    controller.add_event(SpanStartEvent(parent_span_id='', name='op'))

    assert controller.background_dispatcher.barrier(timeout=2.0)
    assert seen_thread_ids
    assert seen_thread_ids != [caller_thread_id]


def test_processor_context_drains_before_finalize() -> None:
    from nexus.trace.tracer import processor_context, span_context

    events: list[str] = []

    class Slow(SpanProcessor):
        dispatch_mode = 'background'

        def on_end(self, span: Span, event) -> None:  # noqa: ANN001
            time.sleep(0.3)
            events.append('on_end')

        def finalize(self) -> None:
            events.append('finalize')

    with processor_context([Slow()]):
        with span_context('op'):
            pass

    # finalize must run only AFTER the queued on_end has been processed.
    assert 'on_end' in events
    assert events.index('on_end') < events.index('finalize')
