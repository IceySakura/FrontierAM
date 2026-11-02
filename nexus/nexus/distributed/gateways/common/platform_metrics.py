from __future__ import annotations

import asyncio
import time

from dataclasses import dataclass, field

from aiohttp import web

from nexus.distributed.config import DistributedConfig
from nexus.distributed.observability.cache import LiveStatusCache
from nexus.distributed.observability.database import install_database_metrics
from nexus.distributed.observability.event_loop import start_event_loop_monitor
from nexus.distributed.observability.telemetry import PlatformMetrics
from nexus.utils.platform_metrics import bind_metric_attributes, current_metric_attributes, metric_scope


METRICS_KEY = web.AppKey('platform_metrics', PlatformMetrics)
LIVE_CACHE_KEY = web.AppKey('live_status_cache', LiveStatusCache)
MANAGER_SUMMARY_CACHE_KEY = web.AppKey('manager_summary_cache', LiveStatusCache)


@dataclass
class GatewayMeasurements:
    metrics: PlatformMetrics
    streams: dict[str, int] = field(default_factory=dict)


STATE_KEY = web.AppKey('gateway_measurements', GatewayMeasurements)


class MeasuredFileResponse(web.FileResponse):
    async def prepare(self, request):
        # FileResponse sends the file during prepare; the server calls prepare again
        # after our middleware has measured the completed transfer.
        if self.prepared:
            return await web.StreamResponse.prepare(self, request)
        return await super().prepare(request)


@dataclass
class RequestMeasurements:
    gateway: GatewayMeasurements
    route: str
    started: float
    environment: str = 'all'
    stream_started: bool = False


def bind_metric_environment(request: web.Request, environment: str | None) -> None:
    measurement = request.get('platform_measurement')
    if measurement is not None and environment:
        measurement.environment = environment
        bind_metric_attributes(environment=environment)


def _route(request: web.Request) -> str:
    resource = request.match_info.route.resource
    return getattr(resource, 'canonical', None) or 'unmatched'


@web.middleware
async def platform_metrics_middleware(request: web.Request, handler):
    state = request.app[STATE_KEY]
    runtime = state.metrics
    started = time.perf_counter()
    request['platform_request_started'] = started
    measurement = RequestMeasurements(state, _route(request), started)
    # Compatibility proxies clone requests; this shared object preserves stream ownership across clones.
    request['platform_measurement'] = measurement
    response_status = 500
    outcome = 'error'
    environment = 'all'
    try:
        with metric_scope(runtime, environment=environment):
            response = await handler(request)
            bound_environment = current_metric_attributes().get('environment')
            if bound_environment and bound_environment != 'all':
                measurement.environment = bound_environment
        response_status = response.status
        outcome = 'success' if response_status < 400 else 'rejected' if response_status < 500 else 'error'
        if not isinstance(response, web.WebSocketResponse):
            await response.prepare(request)
            await response.write_eof()
        return response
    except (asyncio.CancelledError, ConnectionResetError):
        response_status = 499
        outcome = 'client_aborted'
        runtime.record('gateway_client_aborted', attributes={'route': _route(request), 'operation': request.method})
        raise
    finally:
        dimensions = {
            'route': measurement.route,
            'operation': request.method,
            'status': str(response_status),
            'outcome': outcome,
            'environment': request.get('platform_metric_environment', measurement.environment),
        }
        runtime.record('gateway_requests', attributes=dimensions)
        runtime.record('gateway_server_errors', int(response_status >= 500), dimensions)
        runtime.record('gateway_duration_distribution', time.perf_counter() - started, dimensions)
        if measurement.stream_started:
            route = measurement.route
            state.streams[route] = max(0, state.streams.get(route, 0) - 1)
            runtime.record('gateway_stream_active', state.streams[route], {'route': route})
            runtime.record('gateway_stream_closed', attributes=dimensions)


async def _response_prepare(request, response):
    if response.content_type != 'text/event-stream' and not isinstance(response, web.WebSocketResponse):
        return
    measurement = request.get('platform_measurement')
    if measurement is None or measurement.stream_started:
        return
    measurement.stream_started = True
    state = measurement.gateway
    route = measurement.route
    state.streams[route] = state.streams.get(route, 0) + 1
    state.metrics.record('gateway_stream_connections', attributes={'route': route})
    state.metrics.record('gateway_stream_active', state.streams[route], {'route': route})
    state.metrics.record(
        'gateway_stream_handshake_distribution',
        time.perf_counter() - measurement.started,
        {'route': route},
    )


def install_platform_metrics(app: web.Application, *, session_factory, config: DistributedConfig, service: str) -> None:
    runtime = PlatformMetrics(config.metrics, service=service)
    app[METRICS_KEY] = runtime
    app[STATE_KEY] = GatewayMeasurements(runtime)
    app[LIVE_CACHE_KEY] = LiveStatusCache(ttl=config.status_sampling.live_cache_seconds)
    summary_cache = None
    if service == 'control-gateway':
        # Inventory aggregates need a larger budget without occupying live-status query slots.
        summary_cache = LiveStatusCache(
            ttl=config.status_sampling.manager_summary_cache_seconds,
            query_timeout=config.status_sampling.manager_summary_timeout_seconds,
            capacity=64,
            max_concurrent=2,
        )
        app[MANAGER_SUMMARY_CACHE_KEY] = summary_cache
    app.middlewares.insert(0, platform_metrics_middleware)
    app.on_response_prepare.append(_response_prepare)
    install_database_metrics(session_factory, runtime, slow_seconds=config.slow_sql_log_threshold_ms / 1000)

    async def lifecycle(_app):
        with metric_scope(runtime):
            monitor = start_event_loop_monitor(service)
            try:
                yield
            finally:
                if monitor is not None:
                    monitor.stop()
                await app[LIVE_CACHE_KEY].close()
                if summary_cache is not None:
                    await summary_cache.close()
                await runtime.close()

    app.cleanup_ctx.append(lifecycle)
