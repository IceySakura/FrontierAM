from __future__ import annotations

from aiohttp import web

from nexus.distributed.config import DistributedConfig
from nexus.distributed.gateways.application import app as application
from nexus.distributed.gateways.application.routes import proxy as application_proxy
from nexus.distributed.gateways.application.routes import register_application_routes
from nexus.distributed.gateways.application.routes.artifacts import (
    register_artifact_routes as register_application_artifacts,
)
from nexus.distributed.gateways.application.routes.executions import (
    register_execution_routes as register_application_executions,
)
from nexus.distributed.gateways.application.routes.proxy import register_proxy_routes
from nexus.distributed.gateways.application.routes.trace_events import register_trace_event_routes
from nexus.distributed.gateways.common import keys as _keys
from nexus.distributed.gateways.common import proxy as _proxy
from nexus.distributed.gateways.control import app as control
from nexus.distributed.gateways.control.routes import register_control_routes
from nexus.distributed.gateways.control.routes.access import register_access_routes
from nexus.distributed.gateways.control.routes.artifacts import register_artifact_routes as register_control_artifacts
from nexus.distributed.gateways.control.routes.environments import register_environment_routes
from nexus.distributed.gateways.control.routes.execution_logs import register_execution_log_routes
from nexus.distributed.gateways.control.routes.executions import (
    register_execution_routes as register_control_executions,
)
from nexus.distributed.gateways.control.routes.operations import register_operation_routes
from nexus.distributed.gateways.control.routes.provider_configs import register_provider_config_routes
from nexus.distributed.gateways.control.routes.run_trace_backends import register_run_trace_backend_routes
from nexus.distributed.gateways.control.routes.submissions import register_submission_routes
from nexus.distributed.gateways.control.routes.trace_database import register_trace_database_routes


def _inventory(app: web.Application) -> set[tuple[str, str]]:
    return {(route.method, route.resource.canonical) for route in app.router.routes() if route.method != 'HEAD'}


def _app() -> web.Application:
    app = web.Application()
    app[_keys.CONFIG_KEY] = DistributedConfig()
    return app


def _registered(register) -> set[tuple[str, str]]:
    app = _app()
    register(app)
    return _inventory(app)


def test_control_routes_are_registered_once_by_domain():
    groups = [
        _registered(register_access_routes),
        _registered(register_control_artifacts),
        _registered(register_environment_routes),
        _registered(register_control_executions),
        _registered(register_execution_log_routes),
        _registered(register_operation_routes),
        _registered(register_provider_config_routes),
        _registered(register_run_trace_backend_routes),
        _registered(register_submission_routes),
        _registered(register_trace_database_routes),
    ]

    assert all(left.isdisjoint(right) for index, left in enumerate(groups) for right in groups[index + 1 :])

    app = _app()
    register_control_routes(app)
    assert _inventory(app) == set().union(*groups)


def test_application_routes_are_registered_once_by_domain():
    groups = [
        _registered(register_application_artifacts),
        _registered(register_application_executions),
        _registered(register_trace_event_routes),
        _registered(register_proxy_routes),
    ]

    assert all(left.isdisjoint(right) for index, left in enumerate(groups) for right in groups[index + 1 :])

    app = _app()
    register_application_routes(app)
    assert _inventory(app) == {('GET', '/v1/health')} | set().union(*groups)


def test_gateways_share_common_app_keys():
    assert control.SESSION_FACTORY_KEY is application.SESSION_FACTORY_KEY is _keys.SESSION_FACTORY_KEY
    assert control.CONFIG_KEY is application.CONFIG_KEY is _keys.CONFIG_KEY
    assert control.AUTHORIZATION_KEY is application.AUTHORIZATION_KEY is _keys.AUTHORIZATION_KEY
    assert control.UPLOAD_TICKET_SIGNER_KEY is application.UPLOAD_TICKET_SIGNER_KEY is _keys.UPLOAD_TICKET_SIGNER_KEY


def test_gateways_share_proxy_primitives():
    assert control._proxy_headers is application_proxy._proxy_headers is _proxy.proxy_request_headers
    assert control._is_websocket_request is application_proxy._is_websocket_request is _proxy.is_websocket_request
    assert application_proxy._validate_proxy_path is _proxy.validate_proxy_path
    assert callable(_proxy.proxy_response_headers)
    assert callable(_proxy.run_websocket_bridge)
