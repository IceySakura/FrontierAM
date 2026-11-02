"""Deterministic local and real-Collector validation for unified OTel traces."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue


MIRROR_NAME = 'agent.inference.mvp'
NATIVE_NAME = 'model.request'
MCP_NAME = 'mcp.call'
SUBPROCESS_NAME = 'managed.subprocess'
COLLECTOR_HEADER = 'collector-header-sentinel'
REAL_ENDPOINT_ENV = 'NEXUS_OTEL_E2E_ENDPOINT'
REAL_RESOURCE_ATTRIBUTES_ENV = 'NEXUS_OTEL_E2E_RESOURCE_ATTRIBUTES'
REAL_TENANT_ID_ENV = 'NEXUS_OTEL_E2E_TENANT_ID'

_CONTENT = {
    'prompt': 'Return exactly OK.',
    'output': {'text': 'OK'},
    'tool.arguments': {'command': 'printf OK'},
    'mcp.result': {'content': [{'type': 'text', 'text': 'OK'}], 'is_error': False},
    'authorization': 'Bearer fixture-token',
}

MCP_SERVER_FIXTURE = Path(__file__).parent / 'fixtures' / 'otel_mcp_server.py'

_PARENT_SCRIPT = f"""
import asyncio
import json
import sys

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult

export_results = []
original_export = OTLPSpanExporter.export

def recording_export(self, spans):
    result = original_export(self, spans)
    export_results.append(result)
    return result

OTLPSpanExporter.export = recording_export

import nexus

from opentelemetry import trace
from opentelemetry.trace import SpanKind

from nexus.core.tool.mcp import MCPClientConfig, MCPTransportKind, call_mcp_tool_raw, connect_mcp_session
from nexus.trace import span_context
from nexus.utils.otel.lifecycle import force_flush, shutdown

async def main():
    content = json.loads({json.dumps(_CONTENT)!r})
    tracer = trace.get_tracer('tests.e2e.otel')
    with span_context({MIRROR_NAME!r}, attributes=content):
        with tracer.start_as_current_span({NATIVE_NAME!r}, kind=SpanKind.CLIENT):
            trace_id = f'{{trace.get_current_span().get_span_context().trace_id:032x}}'
            config = MCPClientConfig(
                transport=MCPTransportKind.STDIO,
                command=sys.executable,
                args=[{str(MCP_SERVER_FIXTURE)!r}],
            )
            connection = await connect_mcp_session(
                config,
                server_name='otel-e2e',
                nexus_managed=True,
            )
            try:
                result = await call_mcp_tool_raw(
                    connection.session,
                    'spawn_child',
                    {{}},
                    timeout_s=30,
                    server_name='otel-e2e',
                )
                assert result.content[0].text == 'OK'
            finally:
                await connection.close()
    assert force_flush(10_000)
    shutdown()
    assert export_results, 'OTLP exporter was not invoked'
    assert all(result is SpanExportResult.SUCCESS for result in export_results), 'OTLP export did not succeed'
    print(json.dumps({{'trace_id': trace_id}}))

asyncio.run(main())
"""


@dataclass(frozen=True, slots=True)
class SpanView:
    name: str
    kind: int
    trace_id: str
    span_id: str
    parent_span_id: str
    attributes: dict[str, Any]


def _decode_any_value(value: AnyValue) -> Any:
    kind = value.WhichOneof('value')
    if kind == 'array_value':
        return [_decode_any_value(item) for item in value.array_value.values]
    if kind == 'kvlist_value':
        return {item.key: _decode_any_value(item.value) for item in value.kvlist_value.values}
    if kind == 'bytes_value':
        return bytes(value.bytes_value)
    return getattr(value, kind) if kind is not None else None


def _decode_spans(payloads: list[bytes]) -> list[SpanView]:
    spans: list[SpanView] = []
    for payload in payloads:
        request = ExportTraceServiceRequest.FromString(payload)
        for resource_spans in request.resource_spans:
            for scope_spans in resource_spans.scope_spans:
                for span in scope_spans.spans:
                    spans.append(
                        SpanView(
                            name=span.name,
                            kind=span.kind,
                            trace_id=span.trace_id.hex(),
                            span_id=span.span_id.hex(),
                            parent_span_id=span.parent_span_id.hex(),
                            attributes={item.key: _decode_any_value(item.value) for item in span.attributes},
                        )
                    )
    return spans


def _resource_attribute_values(payloads: list[bytes], key: str) -> set[Any]:
    values: set[Any] = set()
    for payload in payloads:
        request = ExportTraceServiceRequest.FromString(payload)
        for resource_spans in request.resource_spans:
            for attribute in resource_spans.resource.attributes:
                if attribute.key == key:
                    values.add(_decode_any_value(attribute.value))
    return values


def _one_span(spans: list[SpanView], name: str) -> SpanView:
    matches = [span for span in spans if span.name == name]
    assert len(matches) == 1, f'expected one {name!r} span, got {len(matches)}'
    return matches[0]


def _assert_trace(spans: list[SpanView], trace_id: str) -> None:
    mirror = _one_span(spans, MIRROR_NAME)
    native = _one_span(spans, NATIVE_NAME)
    mcp_spans = [span for span in spans if span.name == MCP_NAME]
    assert len(mcp_spans) == 2
    mcp_client = next(span for span in mcp_spans if span.kind == 3)
    mcp_server = next(span for span in mcp_spans if span.kind == 2)
    subprocess_span = _one_span(spans, SUBPROCESS_NAME)

    assert mirror.attributes['prompt'] == _CONTENT['prompt']
    assert mirror.attributes['authorization'] == _CONTENT['authorization']
    for key in ('output', 'tool.arguments', 'mcp.result'):
        assert json.loads(mirror.attributes[key]) == _CONTENT[key]
    assert mirror.attributes['nexus.status.code'] == 'OK'

    assert native.parent_span_id == mirror.span_id
    assert mcp_client.parent_span_id == native.span_id
    assert mcp_server.parent_span_id == mcp_client.span_id
    assert subprocess_span.parent_span_id == mcp_server.span_id
    assert {mirror.trace_id, native.trace_id, mcp_client.trace_id, mcp_server.trace_id, subprocess_span.trace_id} == {
        trace_id
    }


class _CollectorServer(ThreadingHTTPServer):
    daemon_threads = True


class OTLPCollector:
    def __init__(self, *, response_status: int = 200) -> None:
        self.payloads: list[bytes] = []
        self.authorization_headers: list[str | None] = []
        self._lock = threading.Lock()
        self._response_status = response_status
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def do_POST(self) -> None:  # noqa: N802
                if self.path != '/v1/traces':
                    self.send_error(404)
                    return
                payload = self.rfile.read(int(self.headers.get('Content-Length', '0')))
                ExportTraceServiceRequest.FromString(payload)
                with owner._lock:
                    owner.payloads.append(payload)
                    owner.authorization_headers.append(self.headers.get('authorization'))
                if owner._response_status == 200:
                    response = ExportTraceServiceResponse().SerializeToString()
                    content_type = 'application/x-protobuf'
                else:
                    response = b'rejected'
                    content_type = 'text/plain'
                self.send_response(owner._response_status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, _format: str, *args: Any) -> None:
                pass

        self._server = _CollectorServer(('127.0.0.1', 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def endpoint(self) -> str:
        host, port = self._server.server_address[:2]
        return f'http://{host}:{port}'

    def __enter__(self) -> OTLPCollector:
        self._thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _run_trace(
    endpoint: str,
    headers: str | None,
    *,
    resource_attributes: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if '://' not in endpoint:
        endpoint = f'http://{endpoint}'
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith(('OTEL_', 'NEXUS_OTEL_'))}
    env.update(
        {
            'NEXUS_OTEL_ENABLED': 'true',
            'OTEL_EXPORTER_OTLP_ENDPOINT': endpoint,
            'OTEL_EXPORTER_OTLP_PROTOCOL': 'http/protobuf',
            'OTEL_TRACES_EXPORTER': 'otlp_proto_http',
            'OTEL_METRICS_EXPORTER': 'none',
            'OTEL_LOGS_EXPORTER': 'none',
            'OTEL_PROPAGATORS': 'tracecontext',
            'OTEL_SERVICE_NAME': 'nexus-otel-e2e',
        }
    )
    if headers:
        env['OTEL_EXPORTER_OTLP_HEADERS'] = headers
    if resource_attributes:
        env['OTEL_RESOURCE_ATTRIBUTES'] = resource_attributes
    return subprocess.run(
        [sys.executable, '-c', _PARENT_SCRIPT],
        cwd=Path(__file__).parents[2],
        env=env,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )


def _trace_id(completed: subprocess.CompletedProcess[str]) -> str:
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.splitlines()[-1])['trace_id']


@pytest.mark.e2e
def test_official_distro_exports_full_mirror_native_and_cross_process_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_TRACES_ENDPOINT', 'http://127.0.0.1:1')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_TRACES_HEADERS', 'authorization=ambient-secret')
    monkeypatch.setenv('OTEL_TRACES_SAMPLER', 'always_off')
    with OTLPCollector() as collector:
        completed = _run_trace(
            collector.endpoint,
            f'authorization={COLLECTOR_HEADER}',
        )

    trace_id = _trace_id(completed)
    _assert_trace(_decode_spans(collector.payloads), trace_id)
    assert collector.authorization_headers
    assert set(collector.authorization_headers) == {COLLECTOR_HEADER}
    assert COLLECTOR_HEADER.encode() not in b''.join(collector.payloads)


@pytest.mark.e2e
def test_official_distro_exports_tenant_resource_without_headers() -> None:
    tenant_id = 'tenant#fixture'
    with OTLPCollector() as collector:
        completed = _run_trace(
            collector.endpoint,
            None,
            resource_attributes=f'tps.tenant.id={quote(tenant_id, safe="")}',
        )

    _trace_id(completed)
    assert _resource_attribute_values(collector.payloads, 'tps.tenant.id') == {tenant_id}
    assert collector.authorization_headers
    assert set(collector.authorization_headers) == {None}


@pytest.mark.e2e
def test_export_failure_is_not_reported_as_success() -> None:
    with OTLPCollector(response_status=400) as collector:
        completed = _run_trace(collector.endpoint, None)

    assert completed.returncode != 0
    assert 'OTLP export did not succeed' in completed.stderr


@pytest.mark.e2e
def test_real_collector_accepts_unified_trace() -> None:
    endpoint = os.getenv(REAL_ENDPOINT_ENV)
    tenant_id = os.getenv(REAL_TENANT_ID_ENV)
    if not endpoint or not tenant_id:
        pytest.skip(f'set {REAL_ENDPOINT_ENV} and {REAL_TENANT_ID_ENV}')

    resource_attributes = os.getenv(REAL_RESOURCE_ATTRIBUTES_ENV, '').strip().strip(',')
    tenant_attribute = f'tps.tenant.id={quote(tenant_id, safe="")}'
    resource_attributes = ','.join(value for value in (tenant_attribute, resource_attributes) if value)

    completed = _run_trace(
        endpoint,
        None,
        resource_attributes=resource_attributes,
    )

    trace_id = _trace_id(completed)
    print(f'trace_id={trace_id}')
    assert 'Failed to export traces' not in completed.stderr
    assert 'Transient error' not in completed.stderr
