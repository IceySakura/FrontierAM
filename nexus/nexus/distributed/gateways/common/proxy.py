from __future__ import annotations

import asyncio
import contextlib

from collections.abc import Callable, Coroutine, Mapping, Sequence
from typing import Any

import aiohttp

from aiohttp import web

from nexus.distributed.config import DistributedConfig


_REQUEST_HEADER_ALLOWLIST = frozenset({
    'accept',
    'accept-language',
    'baggage',
    'cache-control',
    'content-encoding',
    'content-type',
    'if-match',
    'if-modified-since',
    'if-none-match',
    'if-unmodified-since',
    'last-event-id',
    'pragma',
    'prefer',
    'range',
    'traceparent',
    'tracestate',
    'user-agent',
    'x-request-id',
})
_HOP_BY_HOP_RESPONSE_HEADERS = frozenset({
    'connection',
    'content-length',
    'keep-alive',
    'te',
    'trailer',
    'transfer-encoding',
    'upgrade',
})
ResponseHook = Callable[[web.StreamResponse, web.Request], object]


def forward_request_headers(headers: Any) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() in _REQUEST_HEADER_ALLOWLIST or key.lower().startswith('hx-')
    }


def internal_request_headers(config: DistributedConfig) -> dict[str, str]:
    internal_token = config.read_internal_token()
    if not internal_token:
        return {}
    headers = {'X-Nexus-Internal-Token': internal_token}
    if config.auth_mode == 'disabled':
        headers['Authorization'] = f'Bearer {internal_token}'
    return headers


def proxy_request_headers(
    headers: Any,
    config: DistributedConfig,
    *,
    debug_base_path: str | None = None,
) -> dict[str, str]:
    result = forward_request_headers(headers)
    result.update(internal_request_headers(config))
    if debug_base_path is not None:
        result['X-Nexus-Debug-Base-Path'] = debug_base_path
    return result


def proxy_response_headers(headers: Any) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() not in _HOP_BY_HOP_RESPONSE_HEADERS}


def is_websocket_request(request: web.Request) -> bool:
    return request.headers.get('Upgrade', '').lower() == 'websocket'


def validate_proxy_path(value: str) -> None:
    if '\x00' in value or '\\' in value or any(part in {'.', '..'} for part in value.split('/')):
        raise web.HTTPBadRequest(text='invalid_proxy_path')


async def read_request_body_limited(request: web.Request, *, max_size_bytes: int) -> bytes:
    if request.content_length is not None and request.content_length > max_size_bytes:
        raise web.HTTPRequestEntityTooLarge(
            max_size=max_size_bytes,
            actual_size=request.content_length,
        )
    body = bytearray()
    async for chunk in request.content.iter_chunked(64 * 1024):
        body.extend(chunk)
        if len(body) > max_size_bytes:
            raise web.HTTPRequestEntityTooLarge(
                max_size=max_size_bytes,
                actual_size=len(body),
            )
    return bytes(body)


async def proxy_http(
    request: web.Request,
    upstream_url: str,
    *,
    session: aiohttp.ClientSession,
    headers: Mapping[str, str],
    max_body_size_bytes: int,
    response_hook: ResponseHook | None = None,
) -> web.StreamResponse:
    body = await read_request_body_limited(request, max_size_bytes=max_body_size_bytes)
    async with session.request(
        request.method,
        upstream_url,
        data=body or None,
        headers=dict(headers),
        allow_redirects=False,
    ) as upstream:
        return await _stream_upstream_response(request, upstream, response_hook=response_hook)


async def proxy_stream(
    request: web.Request,
    upstream_url: str,
    *,
    session: aiohttp.ClientSession,
    headers: Mapping[str, str],
    timeout: aiohttp.ClientTimeout,
    response_hook: ResponseHook | None = None,
) -> web.StreamResponse:
    async with session.request(
        request.method,
        upstream_url,
        headers=dict(headers),
        allow_redirects=False,
        timeout=timeout,
    ) as upstream:
        return await _stream_upstream_response(request, upstream, response_hook=response_hook)


async def _stream_upstream_response(
    request: web.Request,
    upstream: aiohttp.ClientResponse,
    *,
    response_hook: ResponseHook | None,
) -> web.StreamResponse:
    response = web.StreamResponse(status=upstream.status, headers=proxy_response_headers(upstream.headers))
    if response_hook is not None:
        response_hook(response, request)
    await response.prepare(request)
    async for chunk in upstream.content.iter_chunked(8192):
        await response.write(chunk)
    return response


async def proxy_websocket(
    request: web.Request,
    upstream_url: str,
    *,
    session: aiohttp.ClientSession,
    headers: Mapping[str, str],
    protocols: Sequence[str] = (),
    prepare_client_before_upstream: bool = False,
    connect_error_close_code: int | None = None,
) -> web.WebSocketResponse:
    client_ws: web.WebSocketResponse | None = None
    if prepare_client_before_upstream:
        client_ws = web.WebSocketResponse(protocols=protocols)
        await client_ws.prepare(request)
    try:
        upstream_context = session.ws_connect(
            upstream_url,
            headers=dict(headers),
            protocols=protocols,
        )
        upstream_ws = await upstream_context.__aenter__()
    except aiohttp.ClientError:
        if client_ws is not None and connect_error_close_code is not None:
            await client_ws.close(code=connect_error_close_code)
            return client_ws
        raise

    if client_ws is None:
        selected_protocols = (upstream_ws.protocol,) if upstream_ws.protocol else ()
        client_ws = web.WebSocketResponse(protocols=selected_protocols)
        try:
            await client_ws.prepare(request)
        except Exception:
            await upstream_context.__aexit__(None, None, None)
            raise
    try:

        async def client_to_upstream() -> None:
            async for message in client_ws:
                if message.type == aiohttp.WSMsgType.TEXT:
                    await upstream_ws.send_str(message.data)
                elif message.type == aiohttp.WSMsgType.BINARY:
                    await upstream_ws.send_bytes(message.data)
                elif message.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED}:
                    await upstream_ws.close()
                    break

        async def upstream_to_client() -> None:
            async for message in upstream_ws:
                if message.type == aiohttp.WSMsgType.TEXT:
                    await client_ws.send_str(message.data)
                elif message.type == aiohttp.WSMsgType.BINARY:
                    await client_ws.send_bytes(message.data)
                elif message.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED}:
                    await client_ws.close()
                    break

        await run_websocket_bridge(client_to_upstream(), upstream_to_client())
    finally:
        await upstream_context.__aexit__(None, None, None)
    return client_ws


async def run_websocket_bridge(*coroutines: Coroutine[Any, Any, Any]) -> None:
    tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    error: BaseException | None = None
    for task in done:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if not _is_websocket_close_error(exc):
                error = exc
    for task in pending:
        task.cancel()
    for task in pending:
        with contextlib.suppress(asyncio.CancelledError):
            try:
                await task
            except Exception as exc:
                if not _is_websocket_close_error(exc):
                    error = error or exc
    if error is not None:
        raise error


def _is_websocket_close_error(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, aiohttp.ClientError)):
        return True
    return isinstance(exc, RuntimeError) and any(
        text in str(exc).lower() for text in ('closing transport', 'closed', 'disconnect')
    )


__all__ = [
    'forward_request_headers',
    'internal_request_headers',
    'is_websocket_request',
    'proxy_http',
    'proxy_request_headers',
    'proxy_response_headers',
    'proxy_stream',
    'proxy_websocket',
    'read_request_body_limited',
    'run_websocket_bridge',
    'validate_proxy_path',
]
