from __future__ import annotations

import logging

from typing import Any

import aiohttp

from aiohttp import web
from sqlalchemy.ext.asyncio import async_sessionmaker

from nexus.distributed.access.auth import (
    AuthorizationPolicy,
    create_auth_middleware,
)
from nexus.distributed.access.taihu import parse_tai_identity
from nexus.distributed.access.trace_database import TraceDatabaseAccountManager
from nexus.distributed.artifacts.tickets import UploadTicketSigner
from nexus.distributed.config import DistributedConfig
from nexus.distributed.gateways.common.errors import handle_gateway_errors
from nexus.distributed.gateways.common.keys import (
    AUTHORIZATION_KEY,
    CONFIG_KEY,
    EXECUTION_LOG_METRICS_KEY,
    HTTP_CLIENT_KEY,
    SESSION_FACTORY_KEY,
    TRACE_DATABASE_ACCOUNT_MANAGER_KEY,
    UPLOAD_TICKET_SIGNER_KEY,
)
from nexus.distributed.gateways.common.proxy import is_websocket_request as _shared_is_websocket_request
from nexus.distributed.gateways.common.proxy import proxy_request_headers as _shared_proxy_headers
from nexus.distributed.gateways.control.routes._allocations import allocation_payload as _route_allocation_payload
from nexus.distributed.gateways.control.routes.operations import CONSOLE_INDEX as _CONTROL_CONSOLE_INDEX
from nexus.distributed.gateways.control.routes.operations import (
    _database_runtime_metrics as _operation_database_runtime_metrics,
)
from nexus.distributed.observability.metrics import OpenTelemetryExecutionLogMetrics
from nexus.distributed.persistence.db import run_idempotent_transaction as _db_run_idempotent_transaction
from nexus.distributed.repositories.access import AccessRepository
from nexus.distributed.utils.clock import utcnow
from nexus.utils.otel.propagation import inject_carrier as _otel_inject_carrier


logger = logging.getLogger(__name__)
CONSOLE_INDEX = _CONTROL_CONSOLE_INDEX
_allocation_payload = _route_allocation_payload
_database_runtime_metrics = _operation_database_runtime_metrics
_run_idempotent_transaction = _db_run_idempotent_transaction
_is_websocket_request = _shared_is_websocket_request
_proxy_headers = _shared_proxy_headers


def inject_carrier() -> dict[str, str]:
    """Compatibility hook for callers that patch trace propagation at the app entrypoint."""

    return _otel_inject_carrier()


@web.middleware
async def error_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    response = await handle_gateway_errors(
        request,
        handler,
        logger=logger,
        component='control gateway',
    )
    if request.path == '/v1/system/config-overrides':
        response.headers['Cache-Control'] = 'no-store'
    return response


def create_control_app(*, session_factory: async_sessionmaker, config: DistributedConfig) -> web.Application:
    from nexus.distributed.gateways.common.platform_metrics import install_platform_metrics
    from nexus.distributed.gateways.control.business_statistics import install_business_statistics
    from nexus.distributed.gateways.control.routes import register_control_routes
    from nexus.distributed.gateways.control.status import install_metric_history

    app = web.Application(
        middlewares=[error_middleware, create_auth_middleware(session_factory=session_factory, config=config)],
        client_max_size=config.gateway_json_body_max_size_bytes,
    )
    app[SESSION_FACTORY_KEY] = session_factory
    app[CONFIG_KEY] = config
    install_platform_metrics(app, session_factory=session_factory, config=config, service='control-gateway')
    install_metric_history(app, config)
    app[EXECUTION_LOG_METRICS_KEY] = OpenTelemetryExecutionLogMetrics()
    app.cleanup_ctx.append(_http_client_context)
    app[AUTHORIZATION_KEY] = AuthorizationPolicy()
    app[TRACE_DATABASE_ACCOUNT_MANAGER_KEY] = TraceDatabaseAccountManager(
        session_factory=session_factory,
        config=config,
    )
    app[UPLOAD_TICKET_SIGNER_KEY] = UploadTicketSigner(
        keys=config.artifact_ticket_keyring(),
        current_key_id=config.artifact_upload_ticket_current_key_id,
        ttl=config.artifact_upload_ticket_ttl,
    )
    app.cleanup_ctx.append(_access_bootstrap_context)
    register_control_routes(app)
    install_business_statistics(app, config)
    return app


async def _access_bootstrap_context(app: web.Application):
    config = app[CONFIG_KEY]
    identities = list(config.admin_users)
    if config.auth_mode == 'disabled':
        identities.append('local/local/local-dev')
    async with app[SESSION_FACTORY_KEY]() as session:
        repo = AccessRepository(session)
        for raw in identities:
            identity = parse_tai_identity(raw)
            bootstrap_principal = 'local-dev' if raw == 'local/local/local-dev' else identity.principal_id
            existing = await repo.get_user(bootstrap_principal)
            display_name = identity.display_name
            if display_name is None and existing is not None:
                display_name = existing.display_name
            await repo.upsert_user(
                principal_id=bootstrap_principal,
                issuer=identity.issuer,
                tenant=identity.tenant,
                subject=identity.subject,
                roles={'user', 'admin'},
                display_name=display_name,
                actor='system:bootstrap',
                now=utcnow(),
            )
        await session.commit()
    yield


async def _http_client_context(app: web.Application):
    app[HTTP_CLIENT_KEY] = aiohttp.ClientSession(
        auto_decompress=False,
        connector=aiohttp.TCPConnector(limit=256, ttl_dns_cache=300),
    )
    try:
        yield
    finally:
        await app[HTTP_CLIENT_KEY].close()


__all__ = ['create_control_app', 'error_middleware']
