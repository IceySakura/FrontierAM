from __future__ import annotations

import asyncio
import urllib.parse

from contextlib import asynccontextmanager

import aiohttp

from aiohttp import web

from nexus.distributed.gateways.common.keys import CONFIG_KEY, HTTP_CLIENT_KEY
from nexus.distributed.gateways.common.proxy import internal_request_headers


def manager_logs_query_string(request: web.Request, *, lines: int) -> str:
    params = {'lines': str(lines)}
    for name in ('start_time', 'end_time'):
        value = request.query.get(name)
        if value:
            params[name] = value
    return urllib.parse.urlencode(params)


async def proxy_manager_log_tail(request: web.Request, *, endpoint_url: str, lines: int) -> web.Response:
    url = f'{endpoint_url}/v1/internal/logs?{manager_logs_query_string(request, lines=lines)}'
    config = request.app[CONFIG_KEY]
    async with _request_session(request) as session:
        try:
            async with session.get(url, headers=internal_request_headers(config)) as upstream:
                body = await upstream.read()
                return web.Response(body=body, status=upstream.status, content_type=upstream.content_type)
        except aiohttp.ClientError as exc:
            return _unavailable_response(exc)


async def proxy_manager_log_stream(request: web.Request, *, endpoint_url: str, lines: int) -> web.StreamResponse:
    url = f'{endpoint_url}/v1/internal/logs/stream?{manager_logs_query_string(request, lines=lines)}'
    config = request.app[CONFIG_KEY]
    async with _request_session(request) as session:
        try:
            async with session.get(url, headers=internal_request_headers(config)) as upstream:
                if upstream.status >= 400:
                    return web.Response(
                        body=await upstream.read(),
                        status=upstream.status,
                        content_type=upstream.content_type,
                    )
                response = web.StreamResponse(
                    status=200,
                    headers={
                        'Content-Type': 'text/plain; charset=utf-8',
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                        'X-Accel-Buffering': 'no',
                    },
                )
                await response.prepare(request)
                try:
                    async for chunk in upstream.content.iter_chunked(8192):
                        await response.write(chunk)
                except (ConnectionResetError, asyncio.CancelledError):
                    pass
                return response
        except aiohttp.ClientError as exc:
            return _unavailable_response(exc)


def _unavailable_response(exc: aiohttp.ClientError) -> web.Response:
    return web.json_response(
        {'error': 'manager_logs_unavailable', 'detail': f'{type(exc).__name__}: {exc}'},
        status=502,
    )


@asynccontextmanager
async def _request_session(request: web.Request):
    session = request.app.get(HTTP_CLIENT_KEY)
    if session is not None:
        yield session
        return
    async with aiohttp.ClientSession() as fallback:
        yield fallback


__all__ = ['proxy_manager_log_stream', 'proxy_manager_log_tail']
