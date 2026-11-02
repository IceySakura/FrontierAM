from __future__ import annotations

from nexus.trace.controller import SpanController
from nexus.trace.processors.base import SpanProcessor
from nexus.trace.tracer import processor_context, span_context
from nexus.trace.types import Span


def test_run_trace_context_adds_run_attributes_to_normal_spans() -> None:
    SpanController._instance = None
    seen: list[dict] = []

    class Capture(SpanProcessor):
        def on_start(self, span: Span, event) -> None:  # noqa: ANN001, ARG002
            seen.append(dict(span.attributes))

    from nexus.trace import run_trace_context

    with processor_context([Capture()]):
        with run_trace_context(run_id='run-1', run_span_id='run-span-1', run_kind='live'):
            with span_context('worker.op'):
                pass

    assert seen[0]['_run_id'] == 'run-1'
    assert seen[0]['_run_span_id'] == 'run-span-1'
    assert seen[0]['_run_kind'] == 'live'
