from __future__ import annotations

import asyncio
import importlib

import pytest

from mcp.types import RequestParams
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind


@pytest.fixture(autouse=True)
def _enabled_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')


@pytest.fixture
def otel_harness(monkeypatch: pytest.MonkeyPatch):
    provider = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, 'get_tracer', provider.get_tracer)
    try:
        yield provider.get_tracer('tests.mcp.telemetry'), exporter
    finally:
        provider.shutdown()


def _telemetry():
    return importlib.import_module('nexus.utils.otel.mcp')


def test_merge_trace_meta_preserves_custom_overwrites_stale_and_drops_baggage(otel_harness) -> None:
    tracer, _exporter = otel_harness
    caller_meta = {
        'custom': {'request': 'keep'},
        'TraceParent': '00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01',
        'TRACESTATE': 'stale=value',
        'Baggage': 'authorization=secret',
    }

    with tracer.start_as_current_span('caller') as caller:
        merged = _telemetry().merge_trace_meta(caller_meta)

    assert caller_meta['TraceParent'].endswith('bbbbbbbbbbbbbbbb-01')
    assert merged['custom'] == {'request': 'keep'}
    assert set(merged) == {'custom', 'traceparent'}
    assert merged['traceparent'].split('-')[1] == f'{caller.get_span_context().trace_id:032x}'
    assert merged['traceparent'].split('-')[2] == f'{caller.get_span_context().span_id:016x}'


def test_merge_trace_meta_accepts_sdk_meta_and_does_not_mutate(otel_harness) -> None:
    tracer, _exporter = otel_harness
    meta = RequestParams.Meta(
        custom='keep',
        explicit_null=None,
        TraceParent='00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01',
        baggage='secret=value',
    )
    original = meta.model_dump()

    with tracer.start_as_current_span('caller'):
        merged = _telemetry().merge_trace_meta(meta)

    assert meta.model_dump() == original
    assert merged['custom'] == 'keep'
    assert 'explicit_null' in merged
    assert merged['explicit_null'] is None
    assert 'TraceParent' not in merged
    assert 'baggage' not in merged
    assert set(merged) == {'custom', 'explicit_null', 'traceparent'}


def test_disabled_sdk_meta_preserves_explicit_null(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'false')
    meta = RequestParams.Meta(custom=None)

    merged = _telemetry().merge_trace_meta(meta)

    assert merged == {'custom': None}


def test_disabled_none_is_noop(monkeypatch: pytest.MonkeyPatch, otel_harness) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'false')

    assert _telemetry().merge_trace_meta(None) is None
    with _telemetry().mcp_client_span('server', 'tool') as span:
        assert span is None
    with _telemetry().mcp_server_span(None, 'server', 'tool') as span:
        assert span is None


def test_extract_invalid_direct_carrier_is_noop() -> None:
    assert (
        _telemetry().extract_trace_carrier(
            {
                'traceparent': 'invalid',
                'tracestate': 'vendor=value',
                'baggage': 'authorization=secret',
                'authorization': 'secret',
            }
        )
        == {}
    )


def test_extract_metadata_failure_is_fail_open() -> None:
    class BrokenMeta:
        @staticmethod
        def model_dump(**_kwargs):
            raise RuntimeError('metadata unavailable')

    assert _telemetry().extract_trace_carrier(BrokenMeta()) == {}


def test_missing_optional_dependency_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == 'opentelemetry' or name.startswith('opentelemetry.'):
            raise ModuleNotFoundError("No module named 'opentelemetry'", name='opentelemetry')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr('builtins.__import__', guarded_import)

    assert _telemetry().extract_trace_carrier({'traceparent': 'invalid'}) == {}
    with _telemetry().mcp_client_span('server', 'tool') as client:
        assert client is None
    with _telemetry().mcp_server_span(None, 'server', 'tool') as server:
        assert server is None


def test_server_second_extract_failure_is_fail_open(monkeypatch: pytest.MonkeyPatch, otel_harness) -> None:
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    telemetry = _telemetry()
    monkeypatch.setattr(
        telemetry,
        'extract_trace_carrier',
        lambda _meta: {'traceparent': '00-11111111111111111111111111111111-2222222222222222-01'},
    )
    monkeypatch.setattr(TraceContextTextMapPropagator, 'extract', lambda *_args, **_kwargs: 1 / 0)

    with telemetry.mcp_server_span({}, 'server', 'tool') as server:
        assert server is not None


def test_mcp_service_name_is_safe_stable_and_bounded() -> None:
    telemetry = _telemetry()

    assert telemetry.mcp_service_name(' Catalog / PROD ') == 'nexus-mcp-catalog-prod'
    assert telemetry.mcp_service_name('中文 / !!!') == 'nexus-mcp-mcp'
    assert telemetry.mcp_service_name('A' * 400).startswith('nexus-mcp-')
    assert len(telemetry.mcp_service_name('A' * 400)) <= 255


def test_mcp_client_span_injects_its_current_context(otel_harness) -> None:
    tracer, exporter = otel_harness

    with tracer.start_as_current_span('caller') as caller:
        with _telemetry().mcp_client_span('catalog', 'search') as client:
            carrier = _telemetry().merge_trace_meta({'custom': 'keep'})

    assert client.kind is SpanKind.CLIENT
    assert client.parent is not None
    assert client.parent.span_id == caller.get_span_context().span_id
    assert carrier['traceparent'].split('-')[1] == f'{client.get_span_context().trace_id:032x}'
    assert carrier['traceparent'].split('-')[2] == f'{client.get_span_context().span_id:016x}'
    [exported] = [span for span in exporter.get_finished_spans() if span.name == 'mcp.call']
    assert exported.attributes == {
        'rpc.system': 'mcp',
        'rpc.method': 'tools/call',
        'nexus.mcp.server': 'catalog',
        'nexus.tool.name': 'search',
    }
    assert exported.events == ()
    assert not ({'arguments', 'results', 'content', 'command'} & set(exported.attributes))


def test_server_valid_meta_replaces_same_trace_transport_current(otel_harness) -> None:
    tracer, exporter = otel_harness

    with tracer.start_as_current_span('remote-parent') as remote_parent:
        inbound_meta = _telemetry().merge_trace_meta(None)
        with tracer.start_as_current_span('http.server'):
            with _telemetry().mcp_server_span(inbound_meta, 'proxy', 'search') as server:
                pass

    assert server.kind is SpanKind.SERVER
    assert server.parent is not None
    assert server.parent.span_id == remote_parent.get_span_context().span_id
    [exported] = [span for span in exporter.get_finished_spans() if span.name == 'mcp.call']
    assert exported.context.trace_id == remote_parent.get_span_context().trace_id


def test_server_valid_meta_replaces_unrelated_transport_current(otel_harness) -> None:
    tracer, _exporter = otel_harness

    with tracer.start_as_current_span('remote-parent') as remote_parent:
        inbound_meta = _telemetry().merge_trace_meta(None)
    with tracer.start_as_current_span('http.server'):
        with _telemetry().mcp_server_span(inbound_meta, 'proxy', 'search') as server:
            pass

    assert server.parent is not None
    assert server.parent.span_id == remote_parent.get_span_context().span_id
    assert server.get_span_context().trace_id == remote_parent.get_span_context().trace_id


@pytest.mark.asyncio
async def test_concurrent_server_calls_do_not_cross_context(otel_harness) -> None:
    tracer, exporter = otel_harness
    expected_parents: dict[str, tuple[int, int]] = {}
    inbound_meta: dict[str, dict] = {}

    for name in ('alpha', 'beta'):
        with tracer.start_as_current_span(f'{name}-parent') as parent:
            expected_parents[name] = (parent.get_span_context().trace_id, parent.get_span_context().span_id)
            inbound_meta[name] = _telemetry().merge_trace_meta({'call': name})

    async def invoke(name: str) -> None:
        with _telemetry().mcp_server_span(inbound_meta[name], name, 'run'):
            await asyncio.sleep(0)

    await asyncio.gather(invoke('alpha'), invoke('beta'))

    spans = {
        span.attributes['nexus.mcp.server']: span for span in exporter.get_finished_spans() if span.name == 'mcp.call'
    }
    for name in ('alpha', 'beta'):
        span = spans[name]
        assert span.context.trace_id == expected_parents[name][0]
        assert span.parent is not None
        assert span.parent.span_id == expected_parents[name][1]
