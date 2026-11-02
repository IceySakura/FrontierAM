from __future__ import annotations

import pytest

from opentelemetry import context as otel_context
from opentelemetry import propagate, trace
from opentelemetry.propagators.textmap import DefaultGetter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import SpanKind
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from nexus.utils.otel.propagation import ALLOWED_CARRIER_KEYS, inject_carrier, use_remote_context
from nexus.utils.otel.transport import transport_span


class B3OnlyPropagator:
    def inject(self, carrier, context=None, setter=None) -> None:
        carrier['x-b3-traceid'] = 'global-propagator-was-used'

    def extract(self, carrier, context=None, getter=None):
        return context or otel_context.Context()

    @property
    def fields(self) -> set[str]:
        return {'x-b3-traceid'}


@pytest.fixture(autouse=True)
def _enabled_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')


@pytest.fixture
def in_memory_tracer():
    previous_propagator = propagate.get_global_textmap()
    propagate.set_global_textmap(TraceContextTextMapPropagator())
    provider = TracerProvider(shutdown_on_exit=False)
    try:
        yield provider.get_tracer(__name__)
    finally:
        provider.shutdown()
        propagate.set_global_textmap(previous_propagator)


def test_carrier_round_trip_uses_only_tracecontext(in_memory_tracer) -> None:
    with in_memory_tracer.start_as_current_span('parent') as parent:
        carrier = inject_carrier()

    with use_remote_context(carrier):
        with in_memory_tracer.start_as_current_span('child') as child:
            assert child.parent is not None
            assert child.parent.span_id == parent.get_span_context().span_id

    assert set(carrier) <= ALLOWED_CARRIER_KEYS
    assert set(carrier) <= {'traceparent', 'tracestate'}


def test_carrier_contract_ignores_the_global_propagator(in_memory_tracer) -> None:
    propagate.set_global_textmap(B3OnlyPropagator())

    with in_memory_tracer.start_as_current_span('parent') as parent:
        carrier = inject_carrier()

    assert set(carrier) == {'traceparent'}
    assert 'x-b3-traceid' not in carrier
    with use_remote_context(carrier):
        with in_memory_tracer.start_as_current_span('child') as child:
            assert child.parent is not None
            assert child.parent.span_id == parent.get_span_context().span_id


def test_remote_context_is_fallback_when_current_span_is_valid(in_memory_tracer) -> None:
    with in_memory_tracer.start_as_current_span('remote'):
        remote_carrier = inject_carrier()

    with in_memory_tracer.start_as_current_span('current') as current:
        with use_remote_context(remote_carrier):
            with in_memory_tracer.start_as_current_span('child') as child:
                assert child.parent is not None
                assert child.parent.span_id == current.get_span_context().span_id


def test_remote_context_can_explicitly_replace_current_span(in_memory_tracer) -> None:
    with in_memory_tracer.start_as_current_span('remote') as remote:
        remote_carrier = inject_carrier()

    with in_memory_tracer.start_as_current_span('current'):
        with use_remote_context(remote_carrier, fallback_only=False):
            with in_memory_tracer.start_as_current_span('child') as child:
                assert child.parent is not None
                assert child.parent.span_id == remote.get_span_context().span_id


def test_invalid_and_non_tracecontext_carrier_is_ignored(in_memory_tracer) -> None:
    carrier = {
        'traceparent': 'invalid',
        'baggage': 'authorization=secret',
        'authorization': 'secret',
    }

    with use_remote_context(carrier):
        with in_memory_tracer.start_as_current_span('root') as root:
            assert root.parent is None


def test_disabled_propagation_is_a_noop(monkeypatch: pytest.MonkeyPatch, in_memory_tracer) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'false')

    with in_memory_tracer.start_as_current_span('parent'):
        assert inject_carrier() == {}
        with use_remote_context({'traceparent': 'invalid'}):
            with transport_span('mcp.call', kind='client') as span:
                assert span is None


def test_transport_span_maps_string_kind_and_attributes(
    monkeypatch: pytest.MonkeyPatch,
    in_memory_tracer,
) -> None:
    monkeypatch.setattr(trace, 'get_tracer', lambda *_args, **_kwargs: in_memory_tracer)

    with transport_span('mcp.call', kind='client', attributes={'rpc.method': 'tools/call'}) as span:
        assert span is not None
        assert span.kind is SpanKind.CLIENT
        assert span.attributes['rpc.method'] == 'tools/call'


def test_transport_span_keeps_all_attributes(
    monkeypatch: pytest.MonkeyPatch,
    in_memory_tracer,
) -> None:
    monkeypatch.setattr(trace, 'get_tracer', lambda *_args, **_kwargs: in_memory_tracer)

    with transport_span(
        'mcp.call',
        attributes={
            'rpc.system': ['mcp', 'jsonrpc'],
            'rpc.method': ['tools/call', 1],
            'authorization': 'secret',
            'arguments': {'query': 'secret'},
        },
    ) as span:
        assert span is not None
    assert span.attributes == {
        'rpc.system': ('mcp', 'jsonrpc'),
        'rpc.method': '["tools/call",1]',
        'authorization': 'secret',
        'arguments': '{"query":"secret"}',
    }


def test_transport_span_start_failure_does_not_block_business(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenTracer:
        def start_as_current_span(self, *args, **kwargs):
            raise RuntimeError('start failed')

    monkeypatch.setattr(trace, 'get_tracer', lambda *_args, **_kwargs: BrokenTracer())

    with transport_span('mcp.call') as span:
        result = 'business-result'

    assert span is None
    assert result == 'business-result'


def test_transport_span_end_failure_does_not_replace_business_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenScope:
        def __enter__(self):
            return object()

        def __exit__(self, *args):
            raise RuntimeError('end failed')

    class BrokenTracer:
        def start_as_current_span(self, *args, **kwargs):
            return BrokenScope()

    monkeypatch.setattr(trace, 'get_tracer', lambda *_args, **_kwargs: BrokenTracer())

    with transport_span('mcp.call'):
        result = 'business-result'

    assert result == 'business-result'


def test_transport_span_end_failure_does_not_replace_business_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenScope:
        def __enter__(self):
            return object()

        def __exit__(self, *args):
            raise RuntimeError('otel end failed')

    class BrokenTracer:
        def start_as_current_span(self, *args, **kwargs):
            return BrokenScope()

    monkeypatch.setattr(trace, 'get_tracer', lambda *_args, **_kwargs: BrokenTracer())

    with pytest.raises(ValueError, match='business failed'):
        with transport_span('mcp.call'):
            raise ValueError('business failed')


def test_transport_span_missing_optional_dependency_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == 'opentelemetry' or name.startswith('opentelemetry.'):
            raise ModuleNotFoundError("No module named 'opentelemetry'", name='opentelemetry')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr('builtins.__import__', guarded_import)

    with transport_span('mcp.call') as span:
        result = 'business-result'

    assert span is None
    assert result == 'business-result'


def test_carrier_filter_is_case_insensitive(in_memory_tracer) -> None:
    with in_memory_tracer.start_as_current_span('parent'):
        carrier = inject_carrier()

    mixed_case = {'TraceParent': carrier['traceparent'], 'Baggage': 'secret=value'}
    extracted = propagate.extract(
        {'traceparent': mixed_case['TraceParent']},
        getter=DefaultGetter(),
    )
    expected_parent = trace.get_current_span(extracted).get_span_context()

    with use_remote_context(mixed_case):
        assert trace.get_current_span().get_span_context() == expected_parent
