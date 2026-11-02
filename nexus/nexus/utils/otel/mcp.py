"""OpenTelemetry helpers for MCP message-level propagation."""

from __future__ import annotations

import re

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from nexus.utils.otel.auto import is_enabled as is_otel_enabled
from nexus.utils.otel.propagation import inject_carrier, use_remote_context
from nexus.utils.otel.transport import transport_span


_RESERVED_META_KEYS = frozenset({'traceparent', 'tracestate', 'baggage'})
_MCP_SERVICE_PREFIX = 'nexus-mcp-'


def _normalize_meta(meta: Any | None) -> dict[str, Any]:
    if meta is None:
        return {}
    if isinstance(meta, Mapping):
        return dict(meta)
    model_dump = getattr(meta, 'model_dump', None)
    if callable(model_dump):
        dumped = model_dump(by_alias=True, mode='json', exclude_unset=True)
        if isinstance(dumped, Mapping):
            return dict(dumped)
    raise TypeError('MCP metadata must be a mapping or Pydantic model')


def merge_trace_meta(meta: Any | None) -> dict[str, Any] | None:
    """Copy caller metadata and inject fresh direct SEP-414 W3C keys."""

    if not is_otel_enabled():
        return None if meta is None else _normalize_meta(meta)

    merged = custom_mcp_meta(meta)
    merged.update(inject_carrier())
    return merged


def custom_mcp_meta(meta: Any | None) -> dict[str, Any]:
    """Copy metadata while removing reserved W3C propagation keys."""

    return {
        key: value
        for key, value in _normalize_meta(meta).items()
        if not isinstance(key, str) or key.lower() not in _RESERVED_META_KEYS
    }


def mcp_service_name(server_name: str) -> str:
    """Build a bounded ASCII OTel service name for an MCP child."""

    suffix = re.sub(r'[^a-z0-9._-]+', '-', server_name.strip().lower()).strip('._-') or 'mcp'
    suffix = suffix[: 255 - len(_MCP_SERVICE_PREFIX)].rstrip('._-') or 'mcp'
    return f'{_MCP_SERVICE_PREFIX}{suffix}'


def extract_trace_carrier(meta: Any | None) -> dict[str, str]:
    """Return a validated direct W3C carrier from MCP metadata."""

    if not is_otel_enabled():
        return {}

    try:
        from opentelemetry import trace
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

        normalized: dict[str, str] = {}
        for key, value in _normalize_meta(meta).items():
            normalized_key = key.lower() if isinstance(key, str) else ''
            if normalized_key in {'traceparent', 'tracestate'} and isinstance(value, str) and value:
                normalized[normalized_key] = value
        if 'traceparent' not in normalized:
            return {}
        extracted = TraceContextTextMapPropagator().extract(normalized)
        span_context = trace.get_current_span(extracted).get_span_context()
        if not span_context.is_valid:
            return {}
        carrier = {'traceparent': normalized['traceparent']}
        trace_state = span_context.trace_state.to_header()
        if trace_state:
            carrier['tracestate'] = trace_state
        return carrier
    except Exception:
        return {}


def _span_attributes(server_name: str, tool_name: str) -> dict[str, str]:
    return {
        'rpc.system': 'mcp',
        'rpc.method': 'tools/call',
        'nexus.mcp.server': server_name,
        'nexus.tool.name': tool_name,
    }


@contextmanager
def mcp_client_span(server_name: str, tool_name: str) -> Iterator[Any | None]:
    with transport_span(
        'mcp.call',
        kind='client',
        attributes=_span_attributes(server_name, tool_name),
    ) as span:
        yield span


@contextmanager
def mcp_server_span(meta: Any | None, server_name: str, tool_name: str) -> Iterator[Any | None]:
    carrier = extract_trace_carrier(meta)
    if not carrier:
        with transport_span(
            'mcp.call',
            kind='server',
            attributes=_span_attributes(server_name, tool_name),
        ) as span:
            yield span
        return

    with use_remote_context(carrier, fallback_only=False):
        with transport_span(
            'mcp.call',
            kind='server',
            attributes=_span_attributes(server_name, tool_name),
        ) as span:
            yield span


__all__ = [
    'custom_mcp_meta',
    'extract_trace_carrier',
    'mcp_client_span',
    'mcp_service_name',
    'mcp_server_span',
    'merge_trace_meta',
]
