from __future__ import annotations

import pytest

from aiohttp import web
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind


@pytest.mark.asyncio
async def test_gongfeng_http_auto_instrumentation_injects_client_span_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aiohttp_client_instrumentation = pytest.importorskip(
        'opentelemetry.instrumentation.aiohttp_client',
        reason='requires the otel extra',
    )

    from nexus.runtime.providers import gongfeng

    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    received_headers: dict[str, str] = {}

    async def handle(request: web.Request) -> web.Response:
        received_headers.update(request.headers)
        return web.json_response({'code': 0, 'data': {}})

    app = web.Application()
    app.router.add_post('/gongfeng', handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    assert site._server is not None  # noqa: SLF001
    port = site._server.sockets[0].getsockname()[1]  # noqa: SLF001

    provider = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = aiohttp_client_instrumentation.AioHttpClientInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    instrumentor.instrument(tracer_provider=provider)

    await gongfeng.http_client.close()
    runtime_provider = gongfeng.GongfengRuntimeProvider(
        api_base_url=f'http://127.0.0.1:{port}',
        token='control-plane-token',
    )
    tracer = provider.get_tracer('tests.gongfeng.http')
    try:
        with tracer.start_as_current_span('parent') as parent:
            response = await runtime_provider._call_api('/gongfeng', {})
        assert response == {'code': 0, 'data': {}}
    finally:
        await gongfeng.http_client.close()
        instrumentor.uninstrument()
        await runner.cleanup()
        provider.shutdown()

    client_spans = [span for span in exporter.get_finished_spans() if span.kind is SpanKind.CLIENT]
    assert len(client_spans) == 1
    [http_span] = client_spans
    assert http_span.parent is not None
    assert http_span.parent.span_id == parent.get_span_context().span_id
    assert received_headers['traceparent'].split('-')[2] == f'{http_span.context.span_id:016x}'
    assert received_headers['token'] == 'control-plane-token'
