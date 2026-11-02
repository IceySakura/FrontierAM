from __future__ import annotations

import asyncio

from types import SimpleNamespace

import pytest

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from nexus.core.tool.mcp.models import MCPClientConfig, MCPServerSpec, MCPStartProxyRequest, MCPTransportKind
from nexus.runtime.client.base import ExecutorMixin
from nexus.runtime.executors.mcp import MCPProxySession
from nexus.utils.otel.mcp import merge_trace_meta


@pytest.fixture
def otel_harness(monkeypatch):
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    provider = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, 'get_tracer', provider.get_tracer)
    try:
        yield provider.get_tracer('tests.mcp.proxy'), exporter
    finally:
        provider.shutdown()


def _client_config() -> MCPClientConfig:
    return MCPClientConfig(transport=MCPTransportKind.STDIO, command='third-party-server')


class _RecordingSession:
    def __init__(self, *, error: BaseException | None = None):
        self.calls: list[tuple[tuple, dict]] = []
        self.error = error

    async def call_tool(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        await asyncio.sleep(0)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=[], isError=False)


def _proxy(session: _RecordingSession, *, server_name: str = 'catalog') -> MCPProxySession:
    proxy = MCPProxySession('proxy-1', _client_config(), server_name=server_name)
    proxy.backend = SimpleNamespace(session=session)
    return proxy


def test_mcp_wire_models_default_nexus_managed_false_and_accept_legacy_shape() -> None:
    spec = MCPServerSpec.model_validate(
        {
            'name': 'legacy',
            'transport': 'stdio',
            'command': 'legacy-server',
        }
    )
    request = MCPStartProxyRequest(client=spec.client)
    nested = MCPServerSpec.model_validate(
        {
            'name': 'managed',
            'nexus_managed': True,
            'client': {'transport': 'stdio', 'command': 'managed-server'},
        }
    )
    old_request = MCPStartProxyRequest.model_validate({'client': spec.client.model_dump()})

    assert spec.nexus_managed is False
    assert nested.nexus_managed is True
    assert request.server_name == 'mcp'
    assert request.nexus_managed is False
    assert old_request.server_name == 'mcp'
    assert old_request.nexus_managed is False


@pytest.mark.asyncio
async def test_proxy_chain_prefers_message_carrier_then_reinjects_fresh_client_carrier(otel_harness) -> None:
    tracer, exporter = otel_harness
    backend = _RecordingSession()
    proxy = _proxy(backend)

    with tracer.start_as_current_span('incoming-parent') as incoming_parent:
        incoming_meta = merge_trace_meta(
            {
                'custom': 'keep',
                'baggage': 'authorization=secret',
                'TraceParent': '00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01',
            }
        )
    inbound_span_id = incoming_meta['traceparent'].split('-')[2]

    with tracer.start_as_current_span('http.server'):
        result = await proxy._call_tool('search', {'query': 'secret'}, incoming_meta)

    assert result.content == []
    [(args, kwargs)] = backend.calls
    assert args == ('search', {'query': 'secret'})
    assert kwargs['meta']['custom'] == 'keep'
    assert set(kwargs['meta']) == {'custom', 'traceparent'}
    assert kwargs['meta']['traceparent'].split('-')[2] != inbound_span_id

    mcp_spans = [span for span in exporter.get_finished_spans() if span.name == 'mcp.call']
    server = next(span for span in mcp_spans if span.kind is SpanKind.SERVER)
    client = next(span for span in mcp_spans if span.kind is SpanKind.CLIENT)
    assert server.parent is not None
    assert server.parent.span_id == incoming_parent.get_span_context().span_id
    assert client.parent is not None
    assert client.parent.span_id == server.context.span_id
    assert client.context.trace_id == server.context.trace_id == incoming_parent.get_span_context().trace_id
    assert kwargs['meta']['traceparent'].split('-')[2] == f'{client.context.span_id:016x}'
    assert (
        server.attributes
        == client.attributes
        == {
            'rpc.system': 'mcp',
            'rpc.method': 'tools/call',
            'nexus.mcp.server': 'catalog',
            'nexus.tool.name': 'search',
        }
    )
    assert 'secret' not in repr(server.attributes)
    assert 'secret' not in repr(client.attributes)


@pytest.mark.asyncio
async def test_actual_proxy_handler_reads_sdk_meta_from_request_context(monkeypatch, otel_harness) -> None:
    from contextlib import asynccontextmanager

    from mcp.server import lowlevel as lowlevel_module
    from mcp.server import streamable_http_manager as manager_module
    from mcp.types import RequestParams

    from nexus.runtime.executors import mcp as executor_module

    tracer, exporter = otel_harness
    backend_session = _RecordingSession()
    captured: dict = {}

    class Connection:
        session = backend_session

        async def close(self):
            return None

    class Server:
        def __init__(self, name):
            captured['server'] = self
            self.name = name
            self.request_context = SimpleNamespace(meta=None)

        def list_tools(self):
            return lambda func: func

        def call_tool(self):
            def register(func):
                captured['handler'] = func
                return func

            return register

    class SessionManager:
        def __init__(self, **kwargs):
            captured['manager_kwargs'] = kwargs

        @asynccontextmanager
        async def run(self):
            yield

    async def connect(*args, **kwargs):
        captured['connect'] = (args, kwargs)
        return Connection()

    async def list_tools(*args, **kwargs):
        return []

    monkeypatch.setattr(lowlevel_module, 'Server', Server)
    monkeypatch.setattr(manager_module, 'StreamableHTTPSessionManager', SessionManager)
    monkeypatch.setattr(executor_module, 'connect_mcp_session', connect)
    monkeypatch.setattr(executor_module, 'list_mcp_tools', list_tools)
    client_config = _client_config()
    proxy = MCPProxySession(
        'actual-handler',
        client_config,
        server_name='catalog',
        nexus_managed=True,
    )
    await proxy.start()
    try:
        with tracer.start_as_current_span('incoming-parent') as incoming_parent:
            inbound = merge_trace_meta({'custom': 'from-request-context'})
        captured['server'].request_context.meta = RequestParams.Meta(**inbound)

        await captured['handler']('search', {})
    finally:
        await proxy.stop()

    assert captured['connect'][0][0] is client_config
    assert captured['connect'][1]['nexus_managed'] is True
    assert backend_session.calls[0][1]['meta']['custom'] == 'from-request-context'
    spans = [span for span in exporter.get_finished_spans() if span.name == 'mcp.call']
    server_span = next(span for span in spans if span.kind is SpanKind.SERVER)
    client_span = next(span for span in spans if span.kind is SpanKind.CLIENT)
    assert server_span.parent is not None
    assert server_span.parent.span_id == incoming_parent.get_span_context().span_id
    assert client_span.parent is not None
    assert client_span.parent.span_id == server_span.context.span_id


@pytest.mark.asyncio
async def test_proxy_invalid_meta_falls_back_to_http_current_and_preserves_custom(otel_harness) -> None:
    tracer, exporter = otel_harness
    backend = _RecordingSession()
    proxy = _proxy(backend)

    with tracer.start_as_current_span('http.server') as http_current:
        await proxy._call_tool(
            'search',
            {},
            {'traceparent': 'invalid', 'tracestate': 'invalid', 'custom': 'keep'},
        )

    server = next(
        span for span in exporter.get_finished_spans() if span.name == 'mcp.call' and span.kind is SpanKind.SERVER
    )
    assert server.parent is not None
    assert server.parent.span_id == http_current.get_span_context().span_id
    assert backend.calls[0][1]['meta']['custom'] == 'keep'
    assert backend.calls[0][1]['meta']['traceparent'] != 'invalid'


@pytest.mark.asyncio
async def test_proxy_preserves_valid_tracestate_across_server_and_fresh_client(otel_harness) -> None:
    _tracer, exporter = otel_harness
    backend = _RecordingSession()
    proxy = _proxy(backend)
    trace_id = '11111111111111111111111111111111'
    parent_id = '2222222222222222'
    inbound = {
        'traceparent': f'00-{trace_id}-{parent_id}-01',
        'tracestate': 'vendor=opaque',
    }

    await proxy._call_tool('search', {}, inbound)

    outgoing = backend.calls[0][1]['meta']
    assert outgoing['tracestate'] == 'vendor=opaque'
    assert outgoing['traceparent'].split('-')[1] == trace_id
    assert outgoing['traceparent'].split('-')[2] != parent_id
    spans = [span for span in exporter.get_finished_spans() if span.name == 'mcp.call']
    server = next(span for span in spans if span.kind is SpanKind.SERVER)
    client = next(span for span in spans if span.kind is SpanKind.CLIENT)
    assert server.parent is not None
    assert server.parent.span_id == int(parent_id, 16)
    assert client.parent is not None
    assert client.parent.span_id == server.context.span_id
    assert outgoing['traceparent'].split('-')[2] == f'{client.context.span_id:016x}'


@pytest.mark.asyncio
async def test_proxy_errors_mark_both_semantic_spans_and_preserve_identity(otel_harness) -> None:
    _tracer, exporter = otel_harness
    error = RuntimeError('backend unavailable')
    proxy = _proxy(_RecordingSession(error=error))

    with pytest.raises(RuntimeError) as exc_info:
        await proxy._call_tool('explode', {'secret': 'value'}, None)

    assert exc_info.value is error
    spans = [span for span in exporter.get_finished_spans() if span.name == 'mcp.call']
    assert {span.kind for span in spans} == {SpanKind.CLIENT, SpanKind.SERVER}
    assert all(span.status.status_code is StatusCode.ERROR for span in spans)
    assert all('secret' not in repr(span.attributes) for span in spans)


@pytest.mark.asyncio
async def test_concurrent_proxy_calls_keep_independent_parent_chains(otel_harness) -> None:
    tracer, exporter = otel_harness
    backend = _RecordingSession()
    proxy = _proxy(backend)
    expected: dict[str, tuple[int, int, dict]] = {}
    for name in ('alpha', 'beta'):
        with tracer.start_as_current_span(f'{name}-parent') as parent:
            expected[name] = (
                parent.get_span_context().trace_id,
                parent.get_span_context().span_id,
                merge_trace_meta({'call': name}),
            )

    await asyncio.gather(
        *(proxy._call_tool(name, {}, expected[name][2]) for name in ('alpha', 'beta')),
    )

    spans = [span for span in exporter.get_finished_spans() if span.name == 'mcp.call']
    servers = {span.attributes['nexus.tool.name']: span for span in spans if span.kind is SpanKind.SERVER}
    clients = {span.attributes['nexus.tool.name']: span for span in spans if span.kind is SpanKind.CLIENT}
    for name in ('alpha', 'beta'):
        server = servers[name]
        client = clients[name]
        assert server.context.trace_id == expected[name][0]
        assert server.parent is not None
        assert server.parent.span_id == expected[name][1]
        assert client.parent is not None
        assert client.parent.span_id == server.context.span_id


@pytest.mark.asyncio
async def test_proxy_disabled_keeps_backend_call_shape_without_meta(monkeypatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'false')
    backend = _RecordingSession()
    proxy = _proxy(backend)

    await proxy._call_tool('search', {}, {'custom': 'ignored', 'traceparent': 'stale'})

    assert backend.calls == [(('search', {}), {'read_timeout_seconds': None})]


@pytest.mark.asyncio
async def test_runtime_client_threads_server_identity_and_managed_flag() -> None:
    captured: dict = {}

    class Client(ExecutorMixin):
        async def call_executor(self, name, payload):
            captured.update(name=name, payload=payload)
            return {'proxy_id': 'proxy-1', 'endpoint_path': '/mcp/proxy/proxy-1', 'tools': []}

    result = await Client().mcp_start_proxy(_client_config(), server_name='catalog', nexus_managed=True)

    assert result.proxy_id == 'proxy-1'
    assert captured['name'] == 'mcp_start_proxy'
    assert captured['payload']['server_name'] == 'catalog'
    assert captured['payload']['nexus_managed'] is True


@pytest.mark.asyncio
async def test_executor_threads_request_identity_into_base_proxy(monkeypatch) -> None:
    from nexus.runtime.executors import mcp as executor_module

    captured: dict = {}

    class Proxy:
        def __init__(self, proxy_id, client, *, server_name, nexus_managed):
            captured.update(
                proxy_id=proxy_id,
                client=client,
                server_name=server_name,
                nexus_managed=nexus_managed,
            )

        async def start(self):
            return []

    monkeypatch.setattr(executor_module, '_is_in_worker', lambda: False)
    monkeypatch.setattr(executor_module, 'MCPProxySession', Proxy)
    request = MCPStartProxyRequest(client=_client_config(), server_name='catalog', nexus_managed=True)

    response = await executor_module.mcp_start_proxy(request)

    assert response.endpoint_path.startswith('/mcp/proxy/')
    assert captured['server_name'] == 'catalog'
    assert captured['nexus_managed'] is True
    executor_module.MCP_PROXY_SESSIONS.pop(response.proxy_id, None)
