"""Semantic mapping for Nexus mirror spans."""

from __future__ import annotations

import json
import os

from collections.abc import Mapping
from typing import Any

from nexus.trace.types import RUN_ID_ATTRIBUTE, RUN_KIND_ATTRIBUTE, RUN_SPAN_ID_ATTRIBUTE, Event, Span, Status


Scalar = str | bool | int | float
AttributeValue = Scalar | tuple[Scalar, ...]

_SPAN_ATTRIBUTE_MAP = {
    'type': 'nexus.span.type',
    'name': 'nexus.component.name',
    'hostname': 'host.name',
    'process_id': 'process.pid',
    'thread_id': 'thread.id',
    'task_id': 'nexus.task.id',
    'project_id': 'nexus.project.id',
    RUN_ID_ATTRIBUTE: 'nexus.run.id',
    RUN_SPAN_ID_ATTRIBUTE: 'nexus.run.span_id',
    RUN_KIND_ATTRIBUTE: 'nexus.run.kind',
    '_client_key': 'nexus.llm.client_key',
    'agent.name': 'nexus.agent.name',
    'tool.name': 'nexus.tool.name',
    'runtime.id': 'nexus.runtime.id',
    'worker.id': 'nexus.worker.id',
    'mcp.server': 'nexus.mcp.server',
    'nexus.runtime.id': 'nexus.runtime.id',
    'nexus.worker.id': 'nexus.worker.id',
    'nexus.mcp.server': 'nexus.mcp.server',
    'rpc.system': 'rpc.system',
    'rpc.method': 'rpc.method',
    'rpc.request.id': 'rpc.request.id',
    'server.address': 'server.address',
    'server.port': 'server.port',
    'error.type': 'error.type',
}

_CLIENT_SPAN_TYPES = frozenset({'client', 'llm', 'runtime.client'})
_SERVER_SPAN_TYPES = frozenset({'runtime.server', 'server'})
_INTERNAL_SPAN_TYPES = frozenset({'agent', 'agent.inference', 'task', 'task.phase', 'tool'})


def _normalize_attribute(value: Any) -> AttributeValue:
    if type(value) in {str, bool, int, float}:
        return value
    if isinstance(value, (list, tuple)):
        if not value:
            return ()
        item_type = type(value[0])
        if item_type in {str, bool, int, float} and all(type(item) is item_type for item in value):
            return tuple(value)
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), default=str)


def _set_attribute(target: dict[str, AttributeValue], key: str, value: Any) -> None:
    target[key] = _normalize_attribute(value)


def map_span_kind(span: Span) -> Any:
    from opentelemetry.trace import SpanKind

    raw_type = span.attributes.get('type')
    span_type = str(raw_type).strip().lower().replace('_', '.') if raw_type is not None else ''
    if span_type in _CLIENT_SPAN_TYPES:
        return SpanKind.CLIENT
    if span_type in _SERVER_SPAN_TYPES:
        return SpanKind.SERVER
    if span_type in _INTERNAL_SPAN_TYPES:
        return SpanKind.INTERNAL
    if span.name == 'openai_completion':
        return SpanKind.CLIENT
    return SpanKind.INTERNAL


def map_span_attributes(span: Span, process_role: str | None = None) -> dict[str, AttributeValue]:
    keep_traditional_completion = span.name == 'openai_completion'
    mapped = {
        str(key): _normalize_attribute(value)
        for key, value in span.attributes.items()
        if not (keep_traditional_completion and str(key).startswith('gen_ai.'))
    }
    mapped['nexus.trace_id'] = span.trace_id
    mapped['nexus.span_id'] = span.span_id
    if span.parent_span_id:
        mapped['nexus.parent_span_id'] = span.parent_span_id
    mapped['nexus.status.code'] = span.status.code
    if span.status.message is not None:
        mapped['nexus.status.message'] = span.status.message
    resolved_process_role = process_role or os.getenv('NEXUS_OTEL_PROCESS_ROLE')
    if resolved_process_role:
        mapped['nexus.process.role'] = resolved_process_role

    for source_key, target_key in _SPAN_ATTRIBUTE_MAP.items():
        if keep_traditional_completion and target_key.startswith('gen_ai.'):
            continue
        if source_key in span.attributes:
            _set_attribute(mapped, target_key, span.attributes[source_key])
    return mapped


def map_event_attributes(event: Event | Mapping[str, Any]) -> dict[str, AttributeValue]:
    attributes = event.attributes if isinstance(event, Event) else event
    return {str(key): _normalize_attribute(value) for key, value in attributes.items()}


def apply_status(otel_span: Any, status: Status) -> None:
    if status.code == 'UNSET':
        return
    from opentelemetry.trace import Status as OTelStatus
    from opentelemetry.trace import StatusCode

    status_code = StatusCode.OK if status.code == 'OK' else StatusCode.ERROR
    description = status.message if status_code is StatusCode.ERROR else None
    otel_span.set_status(OTelStatus(status_code, description=description))
