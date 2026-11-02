# /// script
# dependencies = []
#
# [project]
# name = "print"
# authors = [
#     { name = "tarschen" },
# ]
#
# [tool.nexus]
# span_processors = { print = "nexus.trace.processors.print" }
# ///

from __future__ import annotations

from nexus.trace.events import SpanEndEvent, SpanStartEvent, SpanUpdateEvent
from nexus.trace.processors.base import SpanProcessor, span_processor_registry
from nexus.trace.types import Span


@span_processor_registry.register('print')
class PrintSpanProcessor(SpanProcessor):
    """A processor that logs span events."""

    def on_start(self, span: Span, event: SpanStartEvent) -> None:
        print(f'Span started: {span.name} (id={span.span_id}): {event}')

    def on_update(self, span: Span, event: SpanUpdateEvent) -> None:
        print(f'Span updated: {span.name} (id={span.span_id}): {event}')

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        if span.end_time_unix_nano is None:
            raise ValueError(f'Span {span.span_id} has no end_time_unix_nano')
        duration_ms = (span.end_time_unix_nano - span.start_time_unix_nano) / 1e6
        print(f'Span ended: {span.name} (id={span.span_id}, duration={duration_ms:.2f}ms): {event}')
