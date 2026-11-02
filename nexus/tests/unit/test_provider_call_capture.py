from __future__ import annotations

from typing import Any

from tests.provider_call_capture import ProviderCallCaptureProcessor

from nexus.trace.events import SpanEndEvent
from nexus.trace.types import Span, Status


def _span(*, name: str = 'openai_completion', provider_call: dict | None = None) -> Span:
    attributes = {'_client_key': 'main', '_chat_phase': 'main'}
    if provider_call is not None:
        attributes['provider_call'] = provider_call
    return Span(
        trace_id='trace',
        span_id='span',
        parent_span_id='',
        name=name,
        start_time_unix_nano=1_000_000,
        end_time_unix_nano=4_000_000,
        attributes=attributes,
        status=Status(code='ERROR', message='provider failed'),
    )


def test_capture_ignores_non_provider_completion_spans() -> None:
    capture = ProviderCallCaptureProcessor()

    capture.on_end(_span(name='agent.inference', provider_call={'provider': 'ignored'}), SpanEndEvent())
    capture.on_end(_span(), SpanEndEvent())

    assert capture.get_result() == []


def test_capture_returns_compact_defensive_provider_call_records() -> None:
    provider_call: dict[str, Any] = {
        'provider': 'openai',
        'protocol': 'responses',
        'model': 'gpt-test',
        'endpoint': 'https://provider.test/v1/responses',
        'request': {'input': [{'role': 'user', 'content': 'hello'}]},
        'response': {'output': [{'type': 'message'}]},
        'attempts': [{'attempt': 1, 'status': 'error'}],
        'error': {'type': 'upstream_error'},
    }
    span = _span(provider_call=provider_call)
    capture = ProviderCallCaptureProcessor()

    capture.on_end(span, SpanEndEvent())
    provider_call['request']['input'][0]['content'] = 'mutated'

    assert capture.get_result() == [
        {
            'span_id': 'span',
            'client_key': 'main',
            'phase': 'main',
            'model': 'gpt-test',
            'provider': 'openai',
            'protocol': 'responses',
            'provider_io': {
                'provider': 'openai',
                'protocol': 'responses',
                'model': 'gpt-test',
                'endpoint': 'https://provider.test/v1/responses',
                'request': {'input': [{'role': 'user', 'content': 'hello'}]},
                'response': {'output': [{'type': 'message'}]},
                'attempts': [{'attempt': 1, 'status': 'error'}],
                'error': {'type': 'upstream_error'},
            },
            'start_time_unix_nano': 1_000_000,
            'end_time_unix_nano': 4_000_000,
            'duration_ms': 3.0,
            'status': {'code': 'ERROR', 'message': 'provider failed'},
            'error': {'type': 'upstream_error', 'message': 'provider failed'},
        }
    ]
