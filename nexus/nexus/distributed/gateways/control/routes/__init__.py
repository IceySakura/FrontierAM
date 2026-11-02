from __future__ import annotations

from aiohttp import web

from nexus.distributed.gateways.control.routes.access import register_access_routes
from nexus.distributed.gateways.control.routes.artifacts import register_artifact_routes
from nexus.distributed.gateways.control.routes.environments import register_environment_routes
from nexus.distributed.gateways.control.routes.execution_logs import register_execution_log_routes
from nexus.distributed.gateways.control.routes.executions import register_execution_routes
from nexus.distributed.gateways.control.routes.operations import register_operation_routes
from nexus.distributed.gateways.control.routes.provider_configs import register_provider_config_routes
from nexus.distributed.gateways.control.routes.run_trace_backends import register_run_trace_backend_routes
from nexus.distributed.gateways.control.routes.submissions import register_submission_routes
from nexus.distributed.gateways.control.routes.trace_database import register_trace_database_routes


def register_control_routes(app: web.Application) -> None:
    register_submission_routes(app)
    register_artifact_routes(app)
    register_access_routes(app)
    register_run_trace_backend_routes(app)
    register_trace_database_routes(app)
    register_provider_config_routes(app)
    register_environment_routes(app)
    register_execution_routes(app)
    register_execution_log_routes(app)
    register_operation_routes(app)


__all__ = ['register_control_routes']
