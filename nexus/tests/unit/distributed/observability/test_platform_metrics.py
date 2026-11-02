from __future__ import annotations

import asyncio
import signal
import sys
import threading

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from opentelemetry import metrics
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
from opentelemetry.sdk.metrics import Counter, Histogram
from opentelemetry.sdk.metrics.export import AggregationTemporality, InMemoryMetricReader
from pydantic import ValidationError

from nexus.distributed.config import DistributedConfig
from nexus.distributed.config.observability import PlatformMetricsConfig
from nexus.distributed.observability.telemetry import PlatformMetrics
from nexus.utils.platform_metrics import measure_operation, metric_scope


def config(**overrides):
    return PlatformMetricsConfig.model_validate(
        {
            'enabled': True,
            'endpoint': 'http://127.0.0.1:4318/v1/metrics',
            'app_mark': 'test-app',
            'environment': 'test',
            **overrides,
        }
    )


def reader():
    return InMemoryMetricReader(
        preferred_temporality={
            Counter: AggregationTemporality.DELTA,
            Histogram: AggregationTemporality.DELTA,
        }
    )


def metric_map(value):
    return {
        metric.name: metric
        for resource in value.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }


@pytest.mark.asyncio
async def test_private_provider_exports_delta_histogram_and_zero_gauge(monkeypatch):
    monkeypatch.setenv('OTEL_METRICS_EXPORTER', 'none')
    global_provider = metrics.get_meter_provider()
    sink = reader()
    runtime = PlatformMetrics(config(), service='control-gateway', readers=[sink])
    try:
        assert runtime.available
        runtime.record('gateway_requests', 2, {'route': '/v1/executions/{execution_id}', 'token': 'must-not-appear'})
        runtime.record('gateway_duration_distribution', 0.12)
        runtime.record('queue_depth', 7, {'environment': 'test'})
        first = sink.get_metrics_data()
        values = metric_map(first)
        assert values['gateway_requests'].data.data_points[0].value == 2
        assert 'token' not in values['gateway_requests'].data.data_points[0].attributes
        assert values['gateway_requests'].data.aggregation_temporality == AggregationTemporality.DELTA
        assert values['gateway_duration'].data.data_points[0].sum == 0.12
        assert 0.2 in values['gateway_duration'].data.data_points[0].explicit_bounds
        assert values['queue_depth'].data.data_points[0].value == 7
        assert first.resource_metrics[0].resource.attributes['__zhiyan_app_mark__'] == 'test-app'
        assert first.resource_metrics[0].resource.attributes['__zhiyan_env__'] == 'test'
        assert first.resource_metrics[0].scope_metrics[0].scope.name == 'nexus_cloud'
        runtime.record('gateway_requests', 1, {'route': '/v1/executions/{execution_id}'})
        runtime.record('queue_depth', 0, {'environment': 'test'})
        second = metric_map(sink.get_metrics_data())
        assert second['gateway_requests'].data.data_points[0].value == 1
        assert second['queue_depth'].data.data_points[0].value == 0
        assert metrics.get_meter_provider() is global_provider
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_operation_errors_and_cancellation_keep_original_behavior():
    sink = reader()
    runtime = PlatformMetrics(config(), service='instance-manager', readers=[sink])
    try:
        with metric_scope(runtime, environment='sandbox'):
            with pytest.raises(TimeoutError), measure_operation('worker_start'):
                raise TimeoutError('secret failure body')
        item = metric_map(sink.get_metrics_data())['stage_operations'].data.data_points[0]
        assert item.attributes['outcome'] == 'timeout'
        assert item.attributes['environment'] == 'sandbox'
        assert 'secret failure body' not in str(item.attributes)
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_disabled_metrics_do_not_create_provider():
    runtime = PlatformMetrics(PlatformMetricsConfig(), service='control-gateway')
    runtime.record('gateway_requests')
    assert runtime.provider is None
    assert runtime.state() == {
        'enabled': False,
        'available': False,
        'error_code': None,
        'last_export_attempt_at': None,
        'last_export_success_at': None,
        'last_export_error_code': None,
    }
    await runtime.close()


@pytest.mark.asyncio
async def test_http_export_headers_are_isolated_and_shutdown_flushes(monkeypatch):
    captured = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured.append((dict(self.headers), self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-protobuf')
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_HEADERS', 'Authorization=Bearer%20unrelated-query-token')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_METRICS_HEADERS', 'token=unrelated-metrics-token')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE', '/does/not/exist')
    runtime = PlatformMetrics(
        config(endpoint=f'http://127.0.0.1:{server.server_port}/v1/metrics'), service='control-gateway'
    )
    try:
        runtime.record('gateway_requests')
        runtime.record('queue_depth', 0)
        await runtime.close()
        assert captured
        headers, data = captured[-1]
        assert all(key.lower() not in {'authorization', 'token', 'staffname', 'projectname'} for key in headers)
        assert b'unrelated-query-token' not in data
        request = ExportMetricsServiceRequest.FromString(data)
        names = {metric.name for rm in request.resource_metrics for sm in rm.scope_metrics for metric in sm.metrics}
        assert {'gateway_requests', 'queue_depth'} <= names
    finally:
        await runtime.close()
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX service termination')
async def test_sigterm_exports_short_lived_service_metrics_before_exit():
    captured = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured.append(self.rfile.read(int(self.headers['Content-Length'])))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    script = f"""
import asyncio
from nexus.cli.commands.distributed import _run_until_terminated
from nexus.distributed.config.observability import PlatformMetricsConfig
from nexus.distributed.observability.telemetry import PlatformMetrics
async def service():
    metrics = PlatformMetrics(PlatformMetricsConfig(enabled=True, app_mark='test', environment='test',
        endpoint='http://127.0.0.1:{server.server_port}/v1/metrics'), service='instance-manager')
    metrics.record('platform_terminal_results', 1, {{'status': 'failed', 'environment': 'test'}})
    print('ready', flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        await metrics.close()
asyncio.run(_run_until_terminated(service()))
"""
    child = await asyncio.create_subprocess_exec(
        sys.executable, '-c', script, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        assert await asyncio.wait_for(child.stdout.readline(), 15) == b'ready\n'
        child.send_signal(signal.SIGTERM)
        _, error = await asyncio.wait_for(child.communicate(), 15)
        assert child.returncode == 0, error.decode()
        amounts = [
            point.as_int
            for data in captured
            for resource in ExportMetricsServiceRequest.FromString(data).resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
            if metric.name == 'platform_terminal_results'
            for point in metric.sum.data_points
        ]
        assert amounts == [1]
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    'endpoint',
    ['https://user:secret@example.com/v1/metrics', 'https://example.com/v1/metrics?token=secret', 'file:///v1/metrics'],
)
def test_metrics_endpoint_rejects_embedded_credentials(endpoint):
    with pytest.raises(ValidationError):
        config(endpoint=endpoint)


def test_metrics_environment_json_config(monkeypatch):
    monkeypatch.setenv('NEXUS_DIST_METRICS', config().model_dump_json())
    value = DistributedConfig.from_env()
    assert value.metrics.app_mark == 'test-app'
