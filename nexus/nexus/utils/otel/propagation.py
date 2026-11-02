from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from logging import getLogger
from typing import Any, Literal

from nexus.utils.otel.auto import is_enabled


ALLOWED_CARRIER_KEYS = frozenset({'traceparent', 'tracestate'})
logger = getLogger(__name__)


def _normalize_trace_carrier(carrier: Mapping[str, Any] | None) -> dict[str, str]:
    """Normalize casing and keep only non-empty W3C Trace Context string fields."""
    normalized_carrier: dict[str, str] = {}
    for key, value in (carrier or {}).items():
        normalized = key.lower() if isinstance(key, str) else ''
        if normalized in ALLOWED_CARRIER_KEYS and isinstance(value, str) and value:
            normalized_carrier[normalized] = value
    return normalized_carrier


def inject_carrier() -> dict[str, str]:
    if not is_enabled():
        return {}

    try:
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

        carrier: dict[str, str] = {}
        TraceContextTextMapPropagator().inject(carrier)
    except Exception:
        return {}
    return _normalize_trace_carrier(carrier)


def add_trace_state_entries(
    carrier: Mapping[str, Any] | None,
    entries: Iterable[tuple[str, str | None]],
) -> dict[str, str]:
    """Add missing W3C tracestate members while preserving upstream vendor state."""
    normalized_carrier = _normalize_trace_carrier(carrier)
    if 'traceparent' not in normalized_carrier:
        return normalized_carrier
    try:
        from opentelemetry.trace import TraceState

        header = normalized_carrier.get('tracestate')
        trace_state = TraceState.from_header([header]) if header else TraceState()
        for key, value in entries:
            if value and key not in trace_state:
                trace_state = trace_state.add(key, value)
        rendered = trace_state.to_header()
    except Exception:
        return normalized_carrier
    if rendered:
        normalized_carrier['tracestate'] = rendered
    return normalized_carrier


@contextmanager
def use_trace_state_entries(entries: Iterable[tuple[str, str | None]]) -> Iterator[None]:
    """Temporarily enrich the current W3C trace state for downstream propagation."""
    carrier = add_trace_state_entries(inject_carrier(), entries)
    with use_remote_context(carrier, fallback_only=False):
        yield


@contextmanager
def use_remote_context(
    carrier: Mapping[str, Any] | None,
    *,
    fallback_only: bool = True,
    fallback_policy: Literal['valid_context', 'same_trace_id'] = 'valid_context',
) -> Iterator[None]:
    """Activate a valid remote carrier unless the selected current-context policy keeps precedence.

    When ``fallback_only`` is enabled, ``valid_context`` preserves any valid current context, while
    ``same_trace_id`` preserves it only when the current and remote contexts belong to the same trace.
    """
    if not is_enabled():
        yield
        return

    extracted = None
    try:
        from opentelemetry import context, trace
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

        current_span_context = trace.get_current_span().get_span_context()
        keep_valid_current = fallback_only and fallback_policy == 'valid_context' and current_span_context.is_valid
        if not keep_valid_current:
            normalized_carrier = _normalize_trace_carrier(carrier)
            if 'traceparent' in normalized_carrier:
                extracted = TraceContextTextMapPropagator().extract(normalized_carrier)
                extracted_span_context = trace.get_current_span(extracted).get_span_context()
                if not extracted_span_context.is_valid or (
                    fallback_only
                    and fallback_policy == 'same_trace_id'
                    and current_span_context.is_valid
                    and extracted_span_context.trace_id == current_span_context.trace_id
                ):
                    extracted = None
    except Exception:
        extracted = None

    if extracted is None:
        yield
        return

    try:
        token = context.attach(extracted)
    except Exception:
        yield
        return

    try:
        yield
    finally:
        try:
            context.detach(token)
        except Exception:
            logger.warning('Failed to detach remote OTel context', exc_info=True)
