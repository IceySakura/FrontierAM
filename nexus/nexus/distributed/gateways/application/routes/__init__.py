from __future__ import annotations

from aiohttp import web

from nexus.distributed.gateways.application.routes.artifacts import register_artifact_routes
from nexus.distributed.gateways.application.routes.executions import register_execution_routes
from nexus.distributed.gateways.application.routes.proxy import register_proxy_routes
from nexus.distributed.gateways.application.routes.trace_events import register_trace_event_routes


def register_application_routes(app: web.Application) -> None:
    from nexus.distributed.gateways.application.app import health

    app.router.add_get('/v1/health', health)
    register_artifact_routes(app)
    register_execution_routes(app)
    register_trace_event_routes(app)
    register_proxy_routes(app)


__all__ = ['register_application_routes']
