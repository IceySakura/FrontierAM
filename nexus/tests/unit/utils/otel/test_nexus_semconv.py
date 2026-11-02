from __future__ import annotations

import json

from typing import Any

from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.trace import Status as OTelStatus

from nexus.trace.types import Event, Span, Status
from nexus.utils.otel.nexus_semconv import apply_status, map_event_attributes, map_span_attributes, map_span_kind


def _sample_span(**attribute_overrides: Any) -> Span:
    attributes = {
        'type': 'llm',
        'name': 'primary',
        'hostname': 'worker.example',
        'process_id': 42,
        '_run_id': 'run-1',
        '_run_span_id': 'run-span-1',
        '_run_kind': 'single',
        '_client_key': 'main',
        'server.port': 4317,
        'inputs': {
            'model': 'gpt-test',
            'messages': [{'role': 'user', 'content': 'secret prompt'}],
        },
        'outputs': {
            'model': 'gpt-test-2026-07-15',
            'usage': {'prompt_tokens': 12, 'completion_tokens': 4},
            'choices': [{'message': {'content': 'secret response'}}],
        },
        'authorization': 'Bearer secret',
        'unlisted': 'must not escape',
        'tags': ['safe', 1],
    }
    attributes.update(attribute_overrides)
    return Span(
        trace_id='trace-1',
        span_id='span-1',
        parent_span_id='parent-1',
        name='openai_completion',
        start_time_unix_nano=1,
        attributes=attributes,
    )


def test_semconv_keeps_all_original_attributes_and_status() -> None:
    span = _sample_span()
    span.status = Status(code='ERROR', message='failed with detail')

    attrs = map_span_attributes(span, process_role='host')

    assert attrs['nexus.span_id'] == span.span_id
    assert attrs['nexus.trace_id'] == span.trace_id
    assert attrs['nexus.process.role'] == 'host'
    assert attrs['nexus.status.code'] == 'ERROR'
    assert attrs['nexus.status.message'] == 'failed with detail'
    assert json.loads(attrs['inputs']) == span.attributes['inputs']
    assert json.loads(attrs['outputs']) == span.attributes['outputs']
    assert attrs['authorization'] == 'Bearer secret'
    assert attrs['unlisted'] == 'must not escape'
    assert json.loads(attrs['tags']) == ['safe', 1]
    assert all(not isinstance(value, dict) for value in attrs.values())


def test_semconv_keeps_openai_completion_content_without_genai_mapping() -> None:
    span = _sample_span(
        **{
            'gen_ai.provider.name': 'legacy-provider',
            'gen_ai.operation.name': 'chat',
        }
    )

    attrs = map_span_attributes(span, process_role='runtime-worker')

    assert not any(key.startswith('gen_ai.') for key in attrs)
    assert attrs['nexus.llm.client_key'] == 'main'
    assert 'secret prompt' in attrs['inputs']
    assert 'secret response' in attrs['outputs']


def test_semconv_keeps_homogeneous_sequences_and_json_encodes_mixed_sequences() -> None:
    span = _sample_span(**{'rpc.system': ['mcp', 'jsonrpc'], 'rpc.method': ['tools/call', 1]})

    attrs = map_span_attributes(span)

    assert attrs['rpc.system'] == ('mcp', 'jsonrpc')
    assert attrs['rpc.method'] == '["tools/call",1]'


def test_span_kind_mapping_uses_name_and_explicit_runtime_types() -> None:
    assert map_span_kind(_sample_span()) is SpanKind.CLIENT
    assert map_span_kind(_sample_span(type='runtime.client')) is SpanKind.CLIENT
    assert map_span_kind(_sample_span(type='runtime.server')) is SpanKind.SERVER
    assert map_span_kind(_sample_span(type='tool')) is SpanKind.INTERNAL


def test_event_mapping_keeps_all_original_attributes() -> None:
    event = Event(
        time_unix_nano=1,
        name='chat.completion.delta',
        attributes={
            'llm.sequence': 2,
            'llm.model': 'gpt-test',
            'choice.index': 0,
            'delta': 'secret response fragment',
            'authorization': 'Bearer secret',
            'rpc.system': ['mcp', 'jsonrpc'],
            'rpc.method': ['tools/call', 1],
        },
    )

    attrs = map_event_attributes(event)

    assert attrs == {
        'llm.sequence': 2,
        'llm.model': 'gpt-test',
        'choice.index': 0,
        'delta': 'secret response fragment',
        'authorization': 'Bearer secret',
        'rpc.system': ('mcp', 'jsonrpc'),
        'rpc.method': '["tools/call",1]',
    }


class RecordingSpan:
    def __init__(self) -> None:
        self.statuses: list[OTelStatus] = []

    def set_status(self, status: OTelStatus) -> None:
        self.statuses.append(status)


def test_apply_status_maps_code_and_error_message() -> None:
    recording_span = RecordingSpan()

    apply_status(recording_span, Status(code='ERROR', message='secret error payload'))

    assert len(recording_span.statuses) == 1
    otel_status = recording_span.statuses[0]
    assert otel_status.status_code is StatusCode.ERROR
    assert otel_status.description == 'secret error payload'


def test_apply_status_leaves_unset_status_untouched() -> None:
    recording_span = RecordingSpan()

    apply_status(recording_span, Status())

    assert recording_span.statuses == []
