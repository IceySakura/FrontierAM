from __future__ import annotations

import asyncio
import threading

from contextvars import ContextVar
from dataclasses import dataclass
from logging import getLogger
from typing import Any

from nexus.trace.events import SpanEndEvent, SpanUpdateEvent
from nexus.trace.types import Span
from nexus.utils.otel.auto import is_enabled
from nexus.utils.otel.nexus_semconv import apply_status, map_event_attributes, map_span_attributes, map_span_kind


logger = getLogger(__name__)

_BRIDGE_TRACER_SCOPE = 'nexus.utils.otel.nexus_bridge'


@dataclass(frozen=True, slots=True)
class _ActivationRecord:
    nexus_span_id: str
    parent_context: Any
    owner_thread_id: int
    owner_task_id: int | None


_activation_stack: ContextVar[tuple[_ActivationRecord, ...]] = ContextVar(
    'nexus_otel_mirror_activation_stack',
    default=(),
)


def _current_task_id() -> int | None:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        return None
    return id(task) if task is not None else None


class NexusOTelBridge:
    """Mirror local Nexus span lifecycle events into one process-local OTel pipeline."""

    def __init__(self, *, tracer_provider: Any | None = None) -> None:
        self._tracer_provider = tracer_provider
        self._lock = threading.RLock()
        self._mirrors: dict[str, Any] = {}

    def _tracer(self) -> Any:
        if self._tracer_provider is not None:
            return self._tracer_provider.get_tracer(_BRIDGE_TRACER_SCOPE)
        from opentelemetry import trace

        return trace.get_tracer(_BRIDGE_TRACER_SCOPE)

    def start_mirror(self, span: Span) -> Any | None:
        if not is_enabled():
            return None
        from opentelemetry import context, trace

        with self._lock:
            existing = self._mirrors.get(span.span_id)
            if existing is not None:
                logger.warning('duplicate OTel mirror START for Nexus span %s; keeping the first mirror', span.span_id)
                return existing

            parent_mirror = self._mirrors.get(span.parent_span_id)
            parent_context = context.get_current()
            if parent_mirror is not None:
                parent_context = trace.set_span_in_context(parent_mirror, parent_context)
            otel_span = self._tracer().start_span(
                span.name,
                context=parent_context,
                kind=map_span_kind(span),
                attributes=map_span_attributes(span),
                start_time=span.start_time_unix_nano,
            )
            for event in span.events:
                otel_span.add_event(
                    event.name,
                    attributes=map_event_attributes(event),
                    timestamp=event.time_unix_nano,
                )
            self._mirrors[span.span_id] = otel_span
            return otel_span

    def update_mirror(self, span: Span, event: SpanUpdateEvent) -> None:
        if not is_enabled():
            return
        with self._lock:
            otel_span = self._mirrors.get(span.span_id)
            if otel_span is None:
                logger.warning('OTel mirror for Nexus span %s is missing; skipping UPDATE', span.span_id)
                return
            otel_span.set_attributes(map_span_attributes(span))
            for nexus_event in event.events or ():
                otel_span.add_event(
                    nexus_event.name,
                    attributes=map_event_attributes(nexus_event),
                    timestamp=nexus_event.time_unix_nano,
                )

    def end_mirror(self, span: Span, event: SpanEndEvent) -> None:
        if not is_enabled():
            return
        with self._lock:
            otel_span = self._mirrors.pop(span.span_id, None)
        if otel_span is None:
            logger.warning('OTel mirror for Nexus span %s is missing; skipping END', span.span_id)
            return
        otel_span.set_attributes(map_span_attributes(span))
        apply_status(otel_span, span.status)
        otel_span.end(end_time=event.time_unix_nano)

    def activate_mirror(self, nexus_span_id: str) -> bool:
        if not is_enabled():
            return False
        from opentelemetry import context, trace

        with self._lock:
            otel_span = self._mirrors.get(nexus_span_id)
        if otel_span is None:
            logger.warning('OTel mirror for Nexus span %s is missing; cannot activate it', nexus_span_id)
            return False

        parent_context = context.get_current()
        context.attach(trace.set_span_in_context(otel_span, parent_context))
        record = _ActivationRecord(
            nexus_span_id=nexus_span_id,
            parent_context=parent_context,
            owner_thread_id=threading.get_ident(),
            owner_task_id=_current_task_id(),
        )
        _activation_stack.set((*_activation_stack.get(), record))
        return True

    def deactivate_mirror(self, nexus_span_id: str) -> bool:
        if not is_enabled():
            return False
        stack = _activation_stack.get()
        if not stack:
            logger.warning('OTel mirror activation stack is empty; cannot deactivate Nexus span %s', nexus_span_id)
            return False
        record = stack[-1]
        if record.nexus_span_id != nexus_span_id:
            logger.warning(
                'non-current OTel mirror deactivation ignored: requested Nexus span %s, current is %s',
                nexus_span_id,
                record.nexus_span_id,
            )
            return False
        current_thread_id = threading.get_ident()
        current_task_id = _current_task_id()
        if record.owner_thread_id != current_thread_id or record.owner_task_id != current_task_id:
            logger.warning(
                'inherited OTel mirror activation for Nexus span %s belongs to another thread/task; ignoring',
                nexus_span_id,
            )
            return False

        from opentelemetry import context

        # Restoring the saved parent with a fresh attach is safe even inside a copied
        # Python Context. Reusing/detaching the original token there is invalid and
        # could consume a token still owned by the originating context.
        context.attach(record.parent_context)
        _activation_stack.set(stack[:-1])
        return True

    def try_activate_mirror(self, nexus_span_id: str) -> bool:
        """Activate a mirror without allowing optional telemetry to break Nexus tracing."""
        try:
            return self.activate_mirror(nexus_span_id)
        except Exception:
            logger.warning('Failed to activate OTel mirror for Nexus span %s', nexus_span_id, exc_info=True)
            return False

    def try_deactivate_mirror(self, nexus_span_id: str) -> bool:
        """Deactivate a mirror without allowing optional telemetry to break Nexus tracing."""
        try:
            return self.deactivate_mirror(nexus_span_id)
        except Exception:
            logger.warning('Failed to deactivate OTel mirror for Nexus span %s', nexus_span_id, exc_info=True)
            return False


_PROCESS_BRIDGE = NexusOTelBridge()


def get_nexus_bridge() -> NexusOTelBridge:
    return _PROCESS_BRIDGE
