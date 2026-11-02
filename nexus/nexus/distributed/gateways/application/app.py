from __future__ import annotations

import logging

from pathlib import Path
from typing import Any

import aiohttp

from aiohttp import web
from sqlalchemy.ext.asyncio import async_sessionmaker

from nexus.distributed.access.auth import (
    AuthorizationPolicy,
    create_auth_middleware,
)
from nexus.distributed.artifacts.admission import TransferAdmission
from nexus.distributed.artifacts.spool import ArtifactSpool
from nexus.distributed.artifacts.store import ArtifactUploadCoordinator, create_artifact_store
from nexus.distributed.artifacts.tickets import (
    ArtifactContentTicketSigner,
    UploadTicketSigner,
)
from nexus.distributed.config import DistributedConfig
from nexus.distributed.gateways.application.routes._cors import add_cors_headers as _add_cors_headers
from nexus.distributed.gateways.application.routes.live_compatibility import create_live_compatibility_app
from nexus.distributed.gateways.application.routes.trace_events import TRACE_READ_SERVICE_KEY, trace_response_headers
from nexus.distributed.gateways.common.errors import handle_gateway_errors
from nexus.distributed.gateways.common.keys import (
    ARTIFACT_CONTENT_TICKET_SIGNER_KEY,
    ARTIFACT_DOWNLOAD_ADMISSION_KEY,
    ARTIFACT_SPOOL_KEY,
    ARTIFACT_STORE_KEY,
    ARTIFACT_UPLOAD_COORDINATOR_KEY,
    AUTHORIZATION_KEY,
    CONFIG_KEY,
    HTTP_CLIENT_KEY,
    SESSION_FACTORY_KEY,
    UPLOAD_TICKET_SIGNER_KEY,
)
from nexus.distributed.trace.service import TraceReadService


logger = logging.getLogger(__name__)


@web.middleware
async def cors_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    if request.method == 'OPTIONS':
        return _add_cors_headers(web.Response(status=204), request)
    response = await handler(request)
    if not response.prepared:
        _add_cors_headers(response, request)
    return response


@web.middleware
async def error_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    return await handle_gateway_errors(
        request,
        handler,
        logger=logger,
        component='application gateway',
    )


async def health(_request: web.Request) -> web.Response:
    return web.json_response({'status': 'ok'})


def create_application_app(
    *,
    session_factory: async_sessionmaker,
    config: DistributedConfig,
    live_compatibility_control_base_url: str | None = None,
    live_compatibility_application_base_url: str | None = None,
    live_compatibility_default_placement: dict[str, Any] | None = None,
) -> web.Application:
    from nexus.distributed.gateways.application.routes import register_application_routes
    from nexus.distributed.gateways.common.platform_metrics import install_platform_metrics

    app = web.Application(
        middlewares=[
            trace_response_headers,
            cors_middleware,
            error_middleware,
            create_auth_middleware(session_factory=session_factory, config=config),
        ],
        client_max_size=config.gateway_json_body_max_size_bytes,
    )
    app[SESSION_FACTORY_KEY] = session_factory
    app[CONFIG_KEY] = config
    install_platform_metrics(app, session_factory=session_factory, config=config, service='application-gateway')
    if config.trace_read.enabled:
        app[TRACE_READ_SERVICE_KEY] = TraceReadService(config)
    app[AUTHORIZATION_KEY] = AuthorizationPolicy()
    app[UPLOAD_TICKET_SIGNER_KEY] = UploadTicketSigner(
        keys=config.artifact_ticket_keyring(),
        current_key_id=config.artifact_upload_ticket_current_key_id,
        ttl=config.artifact_upload_ticket_ttl,
    )
    app[ARTIFACT_CONTENT_TICKET_SIGNER_KEY] = ArtifactContentTicketSigner(
        keys=config.artifact_ticket_keyring(),
        current_key_id=config.artifact_upload_ticket_current_key_id,
        ttl=config.artifact_upload_ticket_ttl,
    )
    app[ARTIFACT_STORE_KEY] = create_artifact_store(config, access='writer')
    artifact_spool = ArtifactSpool(
        Path(config.artifact_upload_spool_dir),
        capacity_bytes=config.artifact_transfer_spool_capacity_bytes,
    )
    app[ARTIFACT_SPOOL_KEY] = artifact_spool
    app[ARTIFACT_UPLOAD_COORDINATOR_KEY] = ArtifactUploadCoordinator(
        total_concurrency=config.artifact_upload_total_concurrency,
        input_concurrency=config.artifact_upload_input_concurrency,
        result_concurrency=config.artifact_upload_result_concurrency,
        max_waiters=config.artifact_upload_max_waiters,
        acquire_timeout=config.artifact_upload_admission_timeout,
        spool_capacity=artifact_spool,
        retry_attempts=config.artifact_upload_retry_attempts,
    )
    app[ARTIFACT_DOWNLOAD_ADMISSION_KEY] = TransferAdmission(
        total_concurrency=config.artifact_download_total_concurrency,
        input_concurrency=config.artifact_download_input_concurrency,
        result_concurrency=config.artifact_download_result_concurrency,
        max_waiters=config.artifact_download_max_waiters,
        acquire_timeout=config.artifact_download_admission_timeout,
        spool_capacity=artifact_spool,
    )
    app.cleanup_ctx.append(_artifact_spool_context)
    app.cleanup_ctx.append(_http_client_context)
    app.cleanup_ctx.append(_artifact_store_context)
    register_application_routes(app)
    if live_compatibility_control_base_url is not None:
        app.add_subapp(
            '/compat/live',
            create_live_compatibility_app(
                control_base_url=live_compatibility_control_base_url,
                application_base_url=live_compatibility_application_base_url or '',
                default_placement=live_compatibility_default_placement,
                route_prefix='/v1/runs',
                public_runs_base_path='/compat/live/v1/runs',
                health_path='/health',
                instrument_http_server=False,
                json_body_max_size_bytes=config.gateway_json_body_max_size_bytes,
                proxy_body_max_size_bytes=config.gateway_proxy_body_max_size_bytes,
                internal_token=config.read_internal_token(),
            ),
        )
    return app


async def _artifact_spool_context(app: web.Application):
    spool = app[ARTIFACT_SPOOL_KEY]
    await spool.start()
    try:
        yield
    finally:
        await spool.close()


async def _http_client_context(app: web.Application):
    app[HTTP_CLIENT_KEY] = aiohttp.ClientSession(
        auto_decompress=False,
        connector=aiohttp.TCPConnector(limit=512, ttl_dns_cache=300),
    )
    try:
        yield
    finally:
        await app[HTTP_CLIENT_KEY].close()


async def _artifact_store_context(app: web.Application):
    try:
        yield
    finally:
        store = app[ARTIFACT_STORE_KEY]
        await store.close()
