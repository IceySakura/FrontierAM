from __future__ import annotations

import importlib

import pytest

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


def _load_propagation():
    spec = importlib.util.find_spec('nexus.utils.otel.propagation')
    assert spec is not None, 'nexus.utils.otel.propagation is not implemented'
    return importlib.import_module('nexus.utils.otel.propagation')


@pytest.fixture
def tracer():
    provider = TracerProvider(shutdown_on_exit=False)
    try:
        yield provider.get_tracer(__name__)
    finally:
        provider.shutdown()


def test_carrier_round_trip_uses_only_tracecontext(monkeypatch: pytest.MonkeyPatch, tracer) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    propagation = _load_propagation()

    with tracer.start_as_current_span('parent') as parent:
        carrier = propagation.inject_carrier()

    with propagation.use_remote_context(carrier):
        with tracer.start_as_current_span('child') as child:
            assert child.parent is not None
            assert child.parent.span_id == parent.get_span_context().span_id

    assert set(carrier) <= {'traceparent', 'tracestate'}


def test_normalize_trace_carrier_accepts_case_and_rejects_baggage_and_non_strings() -> None:
    propagation = _load_propagation()

    assert propagation._normalize_trace_carrier({
        'TraceParent': 'parent-value',
        'TRACESTATE': 'vendor=value',
        'baggage': 'authorization=secret',
        'traceparent-numeric': 1,
        42: 'ignored',
    }) == {
        'traceparent': 'parent-value',
        'tracestate': 'vendor=value',
    }


def test_add_trace_state_entries_preserves_vendor_and_existing_nexus_members() -> None:
    propagation = _load_propagation()
    carrier = {
        'traceparent': '00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01',
        'tracestate': 'vendor=value,idempotency-key=parent-request',
    }

    enriched = propagation.add_trace_state_entries(
        carrier,
        (
            ('hyway_slice_id', 'child-retry-key'),
            ('hyway_instance_id', 'item-1'),
        ),
    )

    assert enriched == {
        'traceparent': carrier['traceparent'],
        'tracestate': 'hyway_instance_id=item-1,hyway_slice_id=child-retry-key,vendor=value,idempotency-key=parent-request',
    }
    assert propagation.add_trace_state_entries({}, (('hyway_instance_id', 'item-1'),)) == {}


def test_use_trace_state_entries_updates_current_carrier_and_restores_parent(
    monkeypatch: pytest.MonkeyPatch,
    tracer,
) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    propagation = _load_propagation()

    with tracer.start_as_current_span('submit'):
        original = propagation.add_trace_state_entries(propagation.inject_carrier(), (('vendor', 'value'),))
        with propagation.use_remote_context(original, fallback_only=False):
            with propagation.use_trace_state_entries((('hyway_slice_id', 'request-1'),)):
                enriched = propagation.inject_carrier()
            restored = propagation.inject_carrier()

    assert enriched['tracestate'] == 'hyway_slice_id=request-1,vendor=value'
    assert restored['tracestate'] == 'vendor=value'


def test_valid_current_context_wins_over_fallback(monkeypatch: pytest.MonkeyPatch, tracer) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    propagation = _load_propagation()
    with tracer.start_as_current_span('remote'):
        remote_carrier = propagation.inject_carrier()

    with tracer.start_as_current_span('current') as current:
        with propagation.use_remote_context(remote_carrier, fallback_only=True):
            assert trace.get_current_span().get_span_context().span_id == current.get_span_context().span_id


def test_same_trace_fallback_policy_keeps_current_context(monkeypatch: pytest.MonkeyPatch, tracer) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    propagation = _load_propagation()

    with tracer.start_as_current_span('shared-trace'):
        with tracer.start_as_current_span('remote-parent'):
            remote_carrier = propagation.inject_carrier()

        with tracer.start_as_current_span('current') as current:
            with propagation.use_remote_context(
                remote_carrier,
                fallback_only=True,
                fallback_policy='same_trace_id',
            ):
                assert trace.get_current_span().get_span_context().span_id == current.get_span_context().span_id


def test_disabled_propagation_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('NEXUS_OTEL_ENABLED', raising=False)
    propagation = _load_propagation()

    assert propagation.inject_carrier() == {}
    with propagation.use_remote_context({'traceparent': 'invalid'}):
        assert not trace.get_current_span().get_span_context().is_valid


def test_inject_failure_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    propagation = _load_propagation()

    def fail_inject(*_args, **_kwargs) -> None:
        raise RuntimeError('inject failed')

    monkeypatch.setattr(TraceContextTextMapPropagator, 'inject', fail_inject)

    assert propagation.inject_carrier() == {}


def test_missing_optional_dependency_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    propagation = _load_propagation()
    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == 'opentelemetry' or name.startswith('opentelemetry.'):
            raise ModuleNotFoundError("No module named 'opentelemetry'", name='opentelemetry')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr('builtins.__import__', guarded_import)

    assert propagation.inject_carrier() == {}
    with propagation.use_remote_context({'traceparent': 'valid-looking'}):
        pass


def test_noop_remote_context_does_not_swallow_business_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')

    with pytest.raises(ValueError, match='business failed'):
        with _load_propagation().use_remote_context(None):
            raise ValueError('business failed')


def test_detach_failure_does_not_change_business_result(
    monkeypatch: pytest.MonkeyPatch,
    tracer,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    propagation = _load_propagation()
    with tracer.start_as_current_span('remote'):
        carrier = propagation.inject_carrier()

    from opentelemetry import context

    real_detach = context.detach

    def detach_then_fail(token) -> None:  # noqa: ANN001
        real_detach(token)
        raise RuntimeError('detach failed')

    monkeypatch.setattr(context, 'detach', detach_then_fail)

    with propagation.use_remote_context(carrier, fallback_only=False):
        result = 'business-result'

    assert result == 'business-result'
    assert 'Failed to detach remote OTel context' in caplog.text
