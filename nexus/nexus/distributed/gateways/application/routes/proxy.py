from __future__ import annotations

import logging

import aiohttp

from aiohttp import web

from nexus.distributed.access.auth import (
    websocket_application_protocols,
)
from nexus.distributed.errors import (
    TemporaryUnavailableError,
)
from nexus.distributed.gateways.application.routes._cors import add_cors_headers as _add_cors_headers
from nexus.distributed.gateways.application.routes.executions import _authorize_execution
from nexus.distributed.gateways.common.errors import gateway_error_response
from nexus.distributed.gateways.common.keys import (
    CONFIG_KEY,
    HTTP_CLIENT_KEY,
    SESSION_FACTORY_KEY,
)
from nexus.distributed.gateways.common.proxy import (
    is_websocket_request as _is_websocket_request,
)
from nexus.distributed.gateways.common.proxy import (
    proxy_http as _forward_http_proxy,
)
from nexus.distributed.gateways.common.proxy import (
    proxy_request_headers as _proxy_headers,
)
from nexus.distributed.gateways.common.proxy import (
    proxy_websocket as _forward_websocket_proxy,
)
from nexus.distributed.gateways.common.proxy import (
    validate_proxy_path as _validate_proxy_path,
)
from nexus.distributed.repositories.gateway import GatewayRepository
from nexus.distributed.utils.clock import utcnow


logger = logging.getLogger(__name__)


def _runtime_aiomonitor_public_base_path(request: web.Request) -> str | None:
    tail = request.match_info.get('tail') or request.match_info.get('proxy_path') or ''
    parts = tail.split('/')
    if len(parts) >= 3 and parts[0] == 'debug' and parts[1] == 'aiomonitor' and parts[2]:
        prefix = request.path.rsplit('/debug/aiomonitor/', 1)[0]
        return f'{prefix}/debug/aiomonitor/{parts[2]}'
    return None


async def proxy_runtime(request: web.Request) -> web.StreamResponse:
    config = request.app[CONFIG_KEY]
    execution_id = request.match_info['execution_id']
    scope = (
        'execution:command'
        if _is_websocket_request(request) or request.method not in {'GET', 'HEAD', 'OPTIONS'}
        else 'execution:read'
    )
    async with request.app[SESSION_FACTORY_KEY]() as session:
        await _authorize_execution(request, session, scope=scope)
        repo = GatewayRepository(session, manager_healthy_ttl=config.manager_healthy_ttl)
        _, assignment, _ = await repo.resolve_route(execution_id, now=utcnow())
    if assignment.endpoint_url is None:
        raise TemporaryUnavailableError('endpoint_not_ready')

    runtime_name = request.match_info.get('runtime_name')
    if runtime_name:
        tail = request.match_info.get('tail') or ''
        _validate_proxy_path(runtime_name)
        _validate_proxy_path(tail)
        upstream_path = f'/runtimes/{runtime_name}/proxy'
        if tail:
            upstream_path = f'{upstream_path}/{tail}'
    else:
        tail = request.match_info.get('proxy_path') or ''
        _validate_proxy_path(tail)
        upstream_path = f'/{tail}' if tail else '/'

    upstream_url = f'{assignment.endpoint_url.rstrip("/")}{upstream_path}'
    if request.query_string:
        upstream_url = f'{upstream_url}?{request.query_string}'
    debug_base_path = _runtime_aiomonitor_public_base_path(request)
    if _is_websocket_request(request):
        return await _proxy_websocket(request, upstream_url, debug_base_path=debug_base_path)
    return await _proxy_http(request, upstream_url, debug_base_path=debug_base_path)


async def _proxy_http(
    request: web.Request,
    upstream_url: str,
    *,
    debug_base_path: str | None = None,
) -> web.StreamResponse:
    config = request.app[CONFIG_KEY]
    session = request.app[HTTP_CLIENT_KEY]
    try:
        return await _forward_http_proxy(
            request,
            upstream_url,
            session=session,
            headers=_proxy_headers(request.headers, config, debug_base_path=debug_base_path),
            max_body_size_bytes=config.gateway_proxy_body_max_size_bytes,
            response_hook=_add_cors_headers,
        )
    except aiohttp.ClientError as exc:
        logger.warning('Runtime HTTP proxy unavailable: %s', type(exc).__name__)
        return gateway_error_response(TemporaryUnavailableError('executor_unavailable'))


async def _proxy_websocket(
    request: web.Request,
    upstream_url: str,
    *,
    debug_base_path: str | None = None,
) -> web.StreamResponse:
    config = request.app[CONFIG_KEY]
    session = request.app[HTTP_CLIENT_KEY]
    application_protocols = websocket_application_protocols(request)
    try:
        return await _forward_websocket_proxy(
            request,
            upstream_url,
            session=session,
            headers=_proxy_headers(request.headers, config, debug_base_path=debug_base_path),
            protocols=application_protocols,
        )
    except aiohttp.ClientError as exc:
        raise TemporaryUnavailableError('executor_unavailable') from exc


def register_proxy_routes(app: web.Application) -> None:
    app.router.add_route(
        '*',
        '/v1/executions/{execution_id}/runtimes/{runtime_name}/proxy',
        proxy_runtime,
    )
    app.router.add_route(
        '*',
        '/v1/executions/{execution_id}/runtimes/{runtime_name}/proxy/{tail:.*}',
        proxy_runtime,
    )
    app.router.add_route(
        '*',
        '/v1/executions/{execution_id}/proxy/{proxy_path:.*}',
        proxy_runtime,
    )


__all__ = ['register_proxy_routes']
