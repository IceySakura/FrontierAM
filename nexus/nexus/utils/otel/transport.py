"""Explicit spans for Nexus RPC boundaries that library auto-instrumentation cannot infer.

Callers should use this only at transport client/server boundaries (runtime and MCP), not around
ordinary business-layer methods such as ``ToolExecutor`` dispatch.
"""

from __future__ import annotations

import sys

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from logging import getLogger
from typing import Any, cast

from nexus.utils.otel.auto import is_enabled


logger = getLogger(__name__)


def _span_kind(kind: Any) -> Any:
    from opentelemetry.trace import SpanKind

    if isinstance(kind, SpanKind):
        return kind
    kinds = {
        'client': SpanKind.CLIENT,
        'consumer': SpanKind.CONSUMER,
        'internal': SpanKind.INTERNAL,
        'producer': SpanKind.PRODUCER,
        'server': SpanKind.SERVER,
    }
    try:
        return kinds[str(kind).strip().lower()]
    except KeyError as exc:
        raise ValueError(f'Unsupported OpenTelemetry span kind: {kind!r}') from exc


@contextmanager
def transport_span(
    name: str,
    *,
    kind: Any = 'internal',
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[Any | None]:
    if not is_enabled():
        yield None
        return

    try:
        from opentelemetry import trace

        from nexus.utils.otel.nexus_semconv import map_event_attributes

        scope = trace.get_tracer('nexus.utils.otel.transport').start_as_current_span(
            name,
            kind=_span_kind(kind),
            attributes=cast(Any, map_event_attributes(attributes or {})),
        )
        span = scope.__enter__()
    except Exception:
        logger.warning('Failed to start OTel transport span %s', name, exc_info=True)
        yield None
        return

    try:
        yield span
    except BaseException:
        try:
            scope.__exit__(*sys.exc_info())
        except Exception:
            logger.warning('Failed to end OTel transport span %s', name, exc_info=True)
        raise
    else:
        try:
            scope.__exit__(None, None, None)
        except Exception:
            logger.warning('Failed to end OTel transport span %s', name, exc_info=True)
