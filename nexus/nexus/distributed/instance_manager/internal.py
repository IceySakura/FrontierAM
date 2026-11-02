from __future__ import annotations

import asyncio
import hmac
import logging
import os
import tempfile

from contextlib import suppress
from pathlib import Path
from typing import Any

import aiohttp

from aiohttp import web

from nexus.distributed.config import DistributedConfig
from nexus.distributed.instance_manager.service import InstanceManagerService
from nexus.distributed.observability.logging import async_tail_log_file, follow_log_file
from nexus.runner.live.sse import sse_bytes


SERVICE_KEY = web.AppKey('service', InstanceManagerService)
CONFIG_KEY = web.AppKey('config', DistributedConfig)
HTTP_CLIENT_KEY = web.AppKey('http_client', aiohttp.ClientSession)
logger = logging.getLogger(__name__)
PY_SPY_RECORD_FORMAT_CONTENT_TYPES = {
    'svg': 'image/svg+xml',
    'flamegraph': 'image/svg+xml',
    'speedscope': 'application/json',
    'chrometrace': 'application/json',
    'raw': 'text/plain',
}
PY_SPY_RECORD_FORMAT_ARGS = {
    'svg': 'flamegraph',
    'flamegraph': 'flamegraph',
    'speedscope': 'speedscope',
    'chrometrace': 'chrometrace',
    'raw': 'raw',
}


@web.middleware
async def auth_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    config = request.app[CONFIG_KEY]
    try:
        token = config.read_internal_token()
    except ValueError:
        return web.json_response({'error': 'internal_auth_unavailable'}, status=503)
    if not token and config.production_mode:
        return web.json_response({'error': 'internal_auth_unavailable'}, status=503)
    if token:
        expected = f'Bearer {token}'
        dedicated = request.headers.get('X-Nexus-Internal-Token')
        legacy = request.headers.get('Authorization') if config.auth_mode == 'disabled' else None
        dedicated_valid = dedicated is not None and hmac.compare_digest(dedicated, token)
        legacy_valid = legacy is not None and hmac.compare_digest(legacy, expected)
        if not dedicated_valid and not legacy_valid:
            return web.json_response({'error': 'unauthorized'}, status=401)
    return await handler(request)


@web.middleware
async def error_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    try:
        return await handler(request)
    except KeyError as exc:
        return web.json_response({'error': 'assignment_not_found', 'detail': str(exc)}, status=404)
    except FileNotFoundError as exc:
        return web.json_response({'error': 'artifact_not_found', 'detail': str(exc)}, status=404)
    except ValueError as exc:
        return web.json_response({'error': str(exc)}, status=409)
    except web.HTTPException:
        raise
    except Exception:
        logger.exception('Unhandled instance manager internal API error')
        return web.json_response({'error': 'internal_error'}, status=500)


def _proxy_headers(headers: Any, *, forward_debug_base_path: bool = False) -> dict[str, str]:
    skip = {
        'connection',
        'authorization',
        'cookie',
        'content-length',
        'host',
        'keep-alive',
        'proxy-authenticate',
        'proxy-authorization',
        'sec-websocket-extensions',
        'sec-websocket-key',
        'sec-websocket-protocol',
        'sec-websocket-version',
        'te',
        'trailer',
        'transfer-encoding',
        'upgrade',
        'x-csrf-token',
        'x-nexus-internal-token',
        'x-nexus-upload-ticket',
        'x-tai-identity',
    }
    if not forward_debug_base_path:
        skip.add('x-nexus-debug-base-path')
    return {key: value for key, value in headers.items() if key.lower() not in skip}


def _response_headers(headers: Any) -> dict[str, str]:
    skip = {'connection', 'content-length', 'keep-alive', 'te', 'trailer', 'transfer-encoding', 'upgrade'}
    return {key: value for key, value in headers.items() if key.lower() not in skip}


def _is_websocket_request(request: web.Request) -> bool:
    return request.headers.get('Upgrade', '').lower() == 'websocket'


def _validate_proxy_path(value: str) -> None:
    if '\x00' in value or '\\' in value or any(part in {'.', '..'} for part in value.split('/')):
        raise web.HTTPBadRequest(text='invalid_proxy_path')


def _aiomonitor_webui_port(loop_name: str, *, base_port: int) -> int:
    offsets = {'main': 0, 'executor': 1}
    try:
        return base_port + offsets[loop_name]
    except KeyError as exc:
        raise web.HTTPNotFound(
            text='{"error": "aiomonitor_loop_not_found"}',
            content_type='application/json',
        ) from exc


def _debug_base_path(request: web.Request, *, loop_name: str) -> str:
    header = request.headers.get('X-Nexus-Debug-Base-Path')
    if header:
        return header.rstrip('/')
    return f'/v1/internal/debug/aiomonitor/{loop_name}'


def _rewrite_aiomonitor_html(html: str, *, base_path: str) -> str:
    base_path = base_path.rstrip('/')
    replacements = {
        'src="/': f'src="{base_path}/',
        'href="/': f'href="{base_path}/',
        'hx-get="/': f'hx-get="{base_path}/',
        'hx-post="/': f'hx-post="{base_path}/',
        'hx-delete="/': f'hx-delete="{base_path}/',
        'hx-put="/': f'hx-put="{base_path}/',
        'hx-patch="/': f'hx-patch="{base_path}/',
    }
    for old, new in replacements.items():
        html = html.replace(old, new)
    return html


def _query_float(request: web.Request, name: str, default: float, *, min_value: float, max_value: float) -> float:
    raw = request.query.get(name)
    if raw is None or raw == '':
        value = default
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise web.HTTPBadRequest(text=f'invalid {name}') from exc
    return float(min(max(value, min_value), max_value))


async def _run_py_spy(*args: str) -> tuple[bytes, bytes, int]:
    process = await asyncio.create_subprocess_exec(
        'py-spy',
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return stdout, stderr, int(process.returncode or 0)


async def health(request: web.Request) -> web.Response:
    service = request.app[SERVICE_KEY]
    return web.json_response({
        'status': 'ok',
        'manager_id': service.manager_id,
        'session_id': service.session_id,
    })


async def py_spy_dump(request: web.Request) -> web.Response:
    stdout, stderr, returncode = await _run_py_spy('dump', '--pid', str(os.getpid()))
    if returncode != 0:
        return web.json_response(
            {
                'error': 'py_spy_failed',
                'returncode': returncode,
                'stderr': stderr.decode('utf-8', errors='replace'),
            },
            status=500,
        )
    return web.Response(body=stdout, content_type='text/plain')


async def py_spy_record(request: web.Request) -> web.Response:
    output_format = request.query.get('format') or 'svg'
    if output_format not in PY_SPY_RECORD_FORMAT_CONTENT_TYPES:
        raise web.HTTPBadRequest(text='invalid format')
    duration = float(_query_float(request, 'duration', 5.0, min_value=1.0, max_value=60.0))
    duration_arg = str(int(duration)) if duration.is_integer() else str(duration)
    py_spy_format = PY_SPY_RECORD_FORMAT_ARGS[output_format]
    suffix = {
        'svg': '.svg',
        'flamegraph': '.svg',
        'speedscope': '.speedscope.json',
        'chrometrace': '.chrometrace.json',
        'raw': '.txt',
    }[output_format]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as output:
        output_path = Path(output.name)
    try:
        _stdout, stderr, returncode = await _run_py_spy(
            'record',
            '--pid',
            str(os.getpid()),
            '--duration',
            duration_arg,
            '--format',
            py_spy_format,
            '--output',
            str(output_path),
        )
        if returncode != 0:
            return web.json_response(
                {
                    'error': 'py_spy_failed',
                    'returncode': returncode,
                    'stderr': stderr.decode('utf-8', errors='replace'),
                },
                status=500,
            )
        return web.Response(
            body=output_path.read_bytes(),
            content_type=PY_SPY_RECORD_FORMAT_CONTENT_TYPES[output_format],
        )
    finally:
        with suppress(FileNotFoundError):
            output_path.unlink()


async def wakeup(request: web.Request) -> web.Response:
    service = request.app[SERVICE_KEY]
    service.request_wakeup()
    return web.json_response({'status': 'ok'})


async def update_draining(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except (ValueError, web.HTTPException):
        return web.json_response({'error': 'invalid_json_body'}, status=400)
    draining = payload.get('draining') if isinstance(payload, dict) else None
    if not isinstance(draining, bool):
        return web.json_response({'error': 'draining_must_be_boolean'}, status=400)

    service = request.app[SERVICE_KEY]
    if not await service.set_draining(draining):
        return web.json_response({'error': 'manager_session_lost'}, status=409)
    return web.json_response({
        'manager_id': service.manager_id,
        'session_id': service.session_id,
        'draining': service.draining,
    })


def _service_log_path(service: InstanceManagerService) -> str:
    log_path = getattr(service, 'log_path', None)
    if not log_path:
        raise ValueError('logs_not_configured')
    return str(log_path)


def _service_json_log_path(service: InstanceManagerService) -> str:
    return str(Path(_service_log_path(service)).with_suffix('.jsonl'))


def _log_lines(request: web.Request) -> int:
    try:
        return min(max(int(request.query.get('lines', '200') or '200'), 1), 5000)
    except ValueError as exc:
        raise web.HTTPBadRequest(text='invalid lines') from exc


def _log_time_range(request: web.Request) -> tuple[str | None, str | None]:
    return request.query.get('start_time'), request.query.get('end_time')


async def read_logs(request: web.Request) -> web.Response:
    service = request.app[SERVICE_KEY]
    output_format = request.query.get('format', 'text')
    if output_format not in {'jsonl', 'text'}:
        raise web.HTTPBadRequest(text='invalid format')
    start_time, end_time = _log_time_range(request)
    try:
        text = await async_tail_log_file(
            _service_json_log_path(service) if output_format == 'jsonl' else _service_log_path(service),
            lines=_log_lines(request),
            start_time=None if output_format == 'jsonl' else start_time,
            end_time=None if output_format == 'jsonl' else end_time,
        )
    except FileNotFoundError as exc:
        raise web.HTTPNotFound(text='log_not_found') from exc
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc
    return web.Response(
        text=text,
        content_type='application/x-ndjson' if output_format == 'jsonl' else 'text/plain',
    )


async def stream_logs(request: web.Request) -> web.StreamResponse:
    service = request.app[SERVICE_KEY]
    path = _service_log_path(service)
    start_time, end_time = _log_time_range(request)
    try:
        await async_tail_log_file(path, lines=1, start_time=start_time, end_time=end_time)
    except FileNotFoundError as exc:
        raise web.HTTPNotFound(text='log_not_found') from exc
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc
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
        async for chunk in follow_log_file(path, lines=_log_lines(request), start_time=start_time, end_time=end_time):
            await response.write(chunk)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return response


async def stream_events(request: web.Request) -> web.StreamResponse:
    service = request.app[SERVICE_KEY]
    assignment_id = request.match_info['assignment_id']
    cursor = _event_cursor(request)
    response = web.StreamResponse(
        status=200,
        headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        },
    )
    await response.prepare(request)
    try:
        while True:
            try:
                events = await service.executor.collect_events_after(assignment_id, cursor, timeout=15.0)
            except TimeoutError:
                # The executor loop was momentarily too busy to answer within the
                # bridge deadline (e.g. serialising a large batch). This is not a
                # fatal error: keep the SSE connection alive and retry the same
                # cursor instead of letting the TimeoutError bubble into a 500 that
                # aborts an already-prepared stream.
                await response.write(b': keep-alive\n\n')
                continue
            if not events:
                await response.write(b': keep-alive\n\n')
                continue
            for event_id, payload in events:
                cursor = event_id
                await response.write(payload)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return response


async def read_execution_info(request: web.Request) -> web.Response:
    service = request.app[SERVICE_KEY]
    assignment_id = request.match_info['assignment_id']
    return web.json_response(await service.get_assignment_info(assignment_id))


async def read_artifact_result(request: web.Request) -> web.Response:
    service = request.app[SERVICE_KEY]
    assignment_id = request.match_info['assignment_id']
    return web.json_response(await service.read_artifact_result(assignment_id))


async def stream_artifact_events(request: web.Request) -> web.StreamResponse:
    service = request.app[SERVICE_KEY]
    assignment_id = request.match_info['assignment_id']
    cursor = _event_cursor(request)
    events = await service.read_live_trace_events_after(assignment_id, cursor)
    response = web.StreamResponse(
        status=200,
        headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        },
    )
    await response.prepare(request)
    for event in events:
        await response.write(sse_bytes(event))
    return response


def _event_cursor(request: web.Request) -> int:
    cursor = int(request.query.get('cursor', '0') or '0')
    if cursor <= 0:
        with suppress(ValueError, TypeError):
            cursor = max(cursor, int(request.headers.get('Last-Event-ID') or 0))
    return cursor


async def proxy_runtime(request: web.Request) -> web.StreamResponse:
    service = request.app[SERVICE_KEY]
    assignment_id = request.match_info['assignment_id']
    runtime_name = request.match_info['runtime_name']
    tail = request.match_info.get('tail') or ''
    _validate_proxy_path(runtime_name)
    _validate_proxy_path(tail)
    base_url = await service.executor.get_runtime_base_url(assignment_id, runtime_name)
    upstream_url = f'{base_url.rstrip("/")}/{tail}' if tail else base_url.rstrip('/')
    if request.query_string:
        upstream_url = f'{upstream_url}?{request.query_string}'
    if _is_websocket_request(request):
        return await _proxy_websocket(request, upstream_url, forward_debug_base_path=True)
    return await _proxy_http(request, upstream_url, forward_debug_base_path=True)


async def proxy_aiomonitor(request: web.Request) -> web.StreamResponse:
    config = request.app[CONFIG_KEY]
    loop_name = request.match_info['loop']
    tail = request.match_info.get('tail') or ''
    port = _aiomonitor_webui_port(loop_name, base_port=config.aiomonitor_webui_base_port)
    upstream_url = f'http://127.0.0.1:{port}/{tail}' if tail else f'http://127.0.0.1:{port}/'
    if request.query_string:
        upstream_url = f'{upstream_url}?{request.query_string}'
    base_path = _debug_base_path(request, loop_name=loop_name)
    if _is_websocket_request(request):
        return await _proxy_websocket(request, upstream_url)
    return await _proxy_http(request, upstream_url, aiomonitor_base_path=base_path)


async def _proxy_http(
    request: web.Request,
    upstream_url: str,
    *,
    aiomonitor_base_path: str | None = None,
    forward_debug_base_path: bool = False,
) -> web.StreamResponse:
    body = await request.read()
    session = request.app[HTTP_CLIENT_KEY]
    try:
        async with session.request(
            request.method,
            upstream_url,
            data=body or None,
            headers=_proxy_headers(request.headers, forward_debug_base_path=forward_debug_base_path),
            allow_redirects=False,
        ) as upstream:
            response_headers = _response_headers(upstream.headers)
            if aiomonitor_base_path and upstream.headers.get('Location', '').startswith('/'):
                response_headers['Location'] = f'{aiomonitor_base_path}{upstream.headers["Location"]}'
            content_type = upstream.headers.get('Content-Type', '')
            if aiomonitor_base_path and 'text/html' in content_type.lower():
                text = await upstream.text()
                response_headers.pop('Content-Type', None)
                response_headers.pop('content-type', None)
                return web.Response(
                    text=_rewrite_aiomonitor_html(text, base_path=aiomonitor_base_path),
                    status=upstream.status,
                    headers=response_headers,
                    content_type='text/html',
                )
            response = web.StreamResponse(status=upstream.status, headers=response_headers)
            await response.prepare(request)
            async for chunk in upstream.content.iter_chunked(8192):
                await response.write(chunk)
            return response
    except aiohttp.ClientError as exc:
        logger.warning('Runtime proxy unavailable: %s', type(exc).__name__)
        return web.json_response({'error': 'runtime_proxy_unavailable'}, status=502)


async def _proxy_websocket(
    request: web.Request,
    upstream_url: str,
    *,
    forward_debug_base_path: bool = False,
) -> web.WebSocketResponse:
    client_ws = web.WebSocketResponse()
    await client_ws.prepare(request)
    session = request.app[HTTP_CLIENT_KEY]
    try:
        upstream_context = session.ws_connect(
            upstream_url,
            headers=_proxy_headers(request.headers, forward_debug_base_path=forward_debug_base_path),
        )
        upstream_ws = await upstream_context.__aenter__()
    except aiohttp.ClientError:
        await client_ws.close(code=1011)
        return client_ws
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

        tasks = [asyncio.create_task(client_to_upstream()), asyncio.create_task(upstream_to_client())]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
    finally:
        await upstream_context.__aexit__(None, None, None)
    return client_ws


def create_instance_manager_internal_app(
    *, service: InstanceManagerService, config: DistributedConfig
) -> web.Application:
    app = web.Application(middlewares=[error_middleware, auth_middleware])
    app[SERVICE_KEY] = service
    app[CONFIG_KEY] = config
    app.cleanup_ctx.append(_http_client_context)
    app.router.add_get('/v1/internal/health', health)
    app.router.add_post('/v1/internal/wakeup', wakeup)
    app.router.add_patch('/v1/internal/draining', update_draining)
    app.router.add_get('/v1/internal/logs', read_logs)
    app.router.add_get('/v1/internal/logs/stream', stream_logs)
    app.router.add_route('*', '/v1/internal/debug/aiomonitor/{loop}', proxy_aiomonitor)
    app.router.add_route('*', '/v1/internal/debug/aiomonitor/{loop}/{tail:.*}', proxy_aiomonitor)
    app.router.add_get('/v1/internal/debug/py-spy/dump', py_spy_dump)
    app.router.add_get('/v1/internal/debug/py-spy/record', py_spy_record)
    app.router.add_get('/v1/internal/executions/{assignment_id}/info', read_execution_info)
    app.router.add_get('/v1/internal/executions/{assignment_id}/events', stream_events)
    app.router.add_get('/v1/internal/executions/{assignment_id}/artifacts/result', read_artifact_result)
    app.router.add_get('/v1/internal/executions/{assignment_id}/artifacts/events', stream_artifact_events)
    app.router.add_get('/v1/internal/executions/{assignment_id}/run-trace/result', read_artifact_result)
    app.router.add_get('/v1/internal/executions/{assignment_id}/run-trace/events', stream_artifact_events)
    app.router.add_route(
        '*',
        '/v1/internal/executions/{assignment_id}/runtimes/{runtime_name}/proxy',
        proxy_runtime,
    )
    app.router.add_route(
        '*',
        '/v1/internal/executions/{assignment_id}/runtimes/{runtime_name}/proxy/{tail:.*}',
        proxy_runtime,
    )
    return app


async def _http_client_context(app: web.Application):
    app[HTTP_CLIENT_KEY] = aiohttp.ClientSession(
        auto_decompress=False,
        connector=aiohttp.TCPConnector(limit=512, ttl_dns_cache=300),
    )
    try:
        yield
    finally:
        await app[HTTP_CLIENT_KEY].close()
