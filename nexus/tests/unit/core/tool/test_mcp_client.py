import builtins
import sys

from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace

import pytest

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from nexus.core.tool.mcp.client import ConnectedMCPSession, _prepare_client_config, call_mcp_tool, call_mcp_tool_raw
from nexus.core.tool.mcp.models import MCPClientConfig, MCPTransportKind


EXCEPTION_GROUP = getattr(builtins, 'ExceptionGroup', None)


@pytest.fixture
def otel_harness(monkeypatch):
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    provider = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, 'get_tracer', provider.get_tracer)
    try:
        yield provider.get_tracer('tests.mcp.client'), exporter
    finally:
        provider.shutdown()


class _RecordingToolSession:
    def __init__(self, result=None, error: BaseException | None = None):
        self.calls = []
        self.result = result if result is not None else SimpleNamespace(content=[], isError=False)
        self.error = error

    async def call_tool(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


def _dummy_stdio_config() -> MCPClientConfig:
    return MCPClientConfig(
        transport=MCPTransportKind.STDIO,
        command='dummy-mcp-server',
    )


def test_prepare_stdio_client_preserves_runtime_library_path_for_current_python():
    config = MCPClientConfig(
        transport=MCPTransportKind.STDIO,
        command=sys.executable,
        env={'CUSTOM': 'keep'},
    )

    prepared = _prepare_client_config(config)

    assert prepared.env == {'CUSTOM': 'keep'}
    assert config.env == {'CUSTOM': 'keep'}


def test_prepare_stdio_client_clears_runtime_library_path_for_image_server():
    config = MCPClientConfig(
        transport=MCPTransportKind.STDIO,
        command='image-mcp-server',
        env={'CUSTOM': 'keep'},
    )

    prepared = _prepare_client_config(config)

    assert prepared.env == {'CUSTOM': 'keep', 'LD_LIBRARY_PATH': ''}


class _FailingTransportContext:
    async def __aenter__(self):
        raise RuntimeError('transport boom')

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _FailingClientSession:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def initialize(self):
        raise ValueError('session boom')


class _FailingClientSessionWithExceptionGroup(_FailingClientSession):
    async def initialize(self):
        if EXCEPTION_GROUP is None:
            raise RuntimeError('ExceptionGroup is unavailable on this Python version')
        raise EXCEPTION_GROUP('unhandled errors in a TaskGroup', [TimeoutError('session timed out')])


@pytest.mark.asyncio
async def test_start_transport_preserves_original_error(monkeypatch):
    from nexus.core.tool.mcp import client as client_module

    session = ConnectedMCPSession(_dummy_stdio_config())
    monkeypatch.setattr(client_module, 'create_mcp_transport', lambda *args, **kwargs: _FailingTransportContext())

    with pytest.raises(RuntimeError, match='Failed to start MCP transport: RuntimeError: transport boom') as exc_info:
        await session._start_transport()

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == 'transport boom'


@pytest.mark.asyncio
async def test_start_session_preserves_original_error(monkeypatch):
    from nexus.core.tool.mcp import client as client_module

    session = ConnectedMCPSession(_dummy_stdio_config())
    monkeypatch.setattr(client_module, 'ClientSession', _FailingClientSession)

    with pytest.raises(RuntimeError, match='Failed to initialize MCP session: ValueError: session boom') as exc_info:
        await session._start_session((object(), object()))

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__) == 'session boom'


@pytest.mark.asyncio
@pytest.mark.skipif(EXCEPTION_GROUP is None, reason='ExceptionGroup requires Python 3.11+')
async def test_start_session_formats_exception_group_details(monkeypatch):
    from nexus.core.tool.mcp import client as client_module

    session = ConnectedMCPSession(_dummy_stdio_config())
    monkeypatch.setattr(client_module, 'ClientSession', _FailingClientSessionWithExceptionGroup)

    with pytest.raises(
        RuntimeError,
        match=r'Failed to initialize MCP session: ExceptionGroup: unhandled errors in a TaskGroup \(1 sub-exception\)\n\[0\] TimeoutError: session timed out',
    ) as exc_info:
        await session._start_session((object(), object()))

    assert EXCEPTION_GROUP is not None
    assert isinstance(exc_info.value.__cause__, EXCEPTION_GROUP)


@pytest.mark.asyncio
async def test_call_mcp_tool_raw_injects_fresh_direct_meta_and_client_span(otel_harness):
    tracer, exporter = otel_harness
    session = _RecordingToolSession()
    caller_meta = {
        'custom': 'keep',
        'TraceParent': '00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01',
        'baggage': 'authorization=secret',
    }

    with tracer.start_as_current_span('caller') as caller:
        result = await call_mcp_tool_raw(
            session,
            'search',
            {'query': 'secret'},
            timeout_s=10,
            meta=caller_meta,
            server_name='catalog',
        )

    assert result is session.result
    assert caller_meta['TraceParent'].endswith('bbbbbbbbbbbbbbbb-01')
    [(args, kwargs)] = session.calls
    assert args == ('search', {'query': 'secret'})
    assert kwargs['read_timeout_seconds'] == timedelta(seconds=9)
    assert kwargs['meta']['custom'] == 'keep'
    assert set(kwargs['meta']) == {'custom', 'traceparent'}
    [span] = [span for span in exporter.get_finished_spans() if span.name == 'mcp.call']
    assert span.kind is SpanKind.CLIENT
    assert span.parent is not None
    assert span.parent.span_id == caller.get_span_context().span_id
    assert kwargs['meta']['traceparent'].split('-')[2] == f'{span.context.span_id:016x}'
    assert span.status.status_code is StatusCode.UNSET
    assert span.attributes == {
        'rpc.system': 'mcp',
        'rpc.method': 'tools/call',
        'nexus.mcp.server': 'catalog',
        'nexus.tool.name': 'search',
    }
    assert 'query' not in repr(span.attributes)


@pytest.mark.asyncio
async def test_call_mcp_tool_raw_preserves_exception_identity_and_marks_error(otel_harness):
    _tracer, exporter = otel_harness
    error = RuntimeError('backend unavailable')
    session = _RecordingToolSession(error=error)

    with pytest.raises(RuntimeError) as exc_info:
        await call_mcp_tool_raw(session, 'explode', {'secret': 'value'}, server_name='catalog')

    assert exc_info.value is error
    [span] = [span for span in exporter.get_finished_spans() if span.name == 'mcp.call']
    assert span.kind is SpanKind.CLIENT
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes['nexus.tool.name'] == 'explode'
    assert 'secret' not in repr(span.attributes)


@pytest.mark.asyncio
async def test_call_mcp_tool_raw_timeout_preserves_outer_timeout(otel_harness):
    _tracer, exporter = otel_harness

    class SlowSession(_RecordingToolSession):
        async def call_tool(self, *args, **kwargs):
            import asyncio

            self.calls.append((args, kwargs))
            await asyncio.sleep(60)

    session = SlowSession()
    with pytest.raises(TimeoutError):
        await call_mcp_tool_raw(session, 'slow', timeout_s=0.01, server_name='catalog')

    [(args, kwargs)] = session.calls
    assert args == ('slow', {})
    assert kwargs['read_timeout_seconds'] == timedelta(seconds=0.009)
    [span] = [span for span in exporter.get_finished_spans() if span.name == 'mcp.call']
    assert span.status.status_code is StatusCode.ERROR


@pytest.mark.asyncio
async def test_call_mcp_tool_raw_disabled_none_keeps_old_sdk_call_shape(monkeypatch):
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'false')
    session = _RecordingToolSession()

    await call_mcp_tool_raw(session, 'search', {'query': 'value'})

    assert session.calls == [(('search', {'query': 'value'}), {'read_timeout_seconds': None})]


@pytest.mark.asyncio
async def test_call_mcp_tool_raw_disabled_explicit_meta_passes_through(monkeypatch):
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'false')
    session = _RecordingToolSession()
    caller_meta = {'custom': 'keep', 'TraceParent': 'stale', 'baggage': 'keep-when-disabled'}

    await call_mcp_tool_raw(session, 'search', meta=caller_meta)

    assert session.calls[0][1]['meta'] == caller_meta


@pytest.mark.asyncio
async def test_call_mcp_tool_uses_source_name_as_server_identity(otel_harness):
    _tracer, exporter = otel_harness
    session = _RecordingToolSession()

    result = await call_mcp_tool(session, 'search', source_name='inventory')

    assert result.is_error is False
    [span] = [span for span in exporter.get_finished_spans() if span.name == 'mcp.call']
    assert span.attributes['nexus.mcp.server'] == 'inventory'


@pytest.mark.asyncio
async def test_connected_session_threads_child_identity_to_transport(monkeypatch):
    from nexus.core.tool.mcp import client as client_module

    captured: dict = {}

    @asynccontextmanager
    async def transport(config, **kwargs):
        captured.update(config=config, kwargs=kwargs)
        yield object(), object(), lambda: None

    monkeypatch.setattr(client_module, 'create_mcp_transport', transport)
    connected = ConnectedMCPSession(
        _dummy_stdio_config(),
        server_name='catalog',
        nexus_managed=True,
    )

    await connected._start_transport()
    await connected._stop_transport()

    assert captured['kwargs']['server_name'] == 'catalog'
    assert captured['kwargs']['nexus_managed'] is True
