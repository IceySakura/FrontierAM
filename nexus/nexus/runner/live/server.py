"""aiohttp server for live interactive runs."""

from __future__ import annotations

import asyncio
import json
import logging

from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any, TypeVar, get_args
from urllib.parse import urlsplit

import aiohttp

from aiohttp import web
from pydantic import BaseModel, ValidationError

from nexus.runner.live.manager import LiveRun, LiveRunError, LiveRunManager
from nexus.runner.live.models import (
    CreateRunFromYamlRequest,
    CreateRunRequest,
    RunConfigResponse,
    RunListResponse,
    RunStatus,
    RunSummaryListResponse,
    RuntimeListResponse,
    SubmitTurnRequest,
)
from nexus.runner.live.sse import prefix_runtime_urls, sse_bytes
from nexus.runner.single import RunInput


logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).with_name('static')
INDEX_HTML = STATIC_DIR / 'index.html'
MANAGER_KEY = web.AppKey('manager', LiveRunManager)
BASE_PATH_KEY = web.AppKey('base_path', str)
ModelT = TypeVar('ModelT', bound=BaseModel)
_VALID_RUN_STATUSES = set(get_args(RunStatus))


def _normalize_base_path(base_path: str | None) -> str:
    if not base_path or base_path == '/':
        return ''
    normalized = base_path.strip()
    if not normalized:
        return ''
    if not normalized.startswith('/'):
        normalized = f'/{normalized}'
    return normalized.rstrip('/')


def _route(base_path: str, path: str) -> str:
    if not base_path:
        return path
    if path == '/':
        return base_path
    return f'{base_path}{path}'


def _query_bool(query: Any, key: str, *, default: bool = False) -> bool:
    value = query.get(key) if hasattr(query, 'get') else None
    if value is None:
        return default
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


_SENSITIVE_KEY_MARKERS = (
    'api_key',
    'apikey',
    'app_key',
    'authorization',
    'auth_token',
    'credential',
    'password',
    'secret',
    'token',
)


def _redact_sensitive_config(value: Any, *, parent_key: str = '') -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key).lower()
            if any(marker in key_text for marker in _SENSITIVE_KEY_MARKERS):
                redacted[key] = '<redacted>'
            else:
                redacted[key] = _redact_sensitive_config(child, parent_key=key_text)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive_config(item, parent_key=parent_key) for item in value]
    return value


def _json_response(
    model_or_data: BaseModel | dict[str, Any],
    *,
    status: int = 200,
    request: web.Request | None = None,
) -> web.Response:
    data = model_or_data.model_dump(mode='json') if isinstance(model_or_data, BaseModel) else model_or_data
    if request is not None:
        prefix_runtime_urls(data, request.app[BASE_PATH_KEY])
    return web.json_response(data, status=status)


async def _read_model(request: web.Request, model: type[ModelT]) -> ModelT:
    try:
        data = await request.json()
    except json.JSONDecodeError as exc:
        raise LiveRunError(f'invalid JSON body: {exc}', status=400) from exc
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise LiveRunError(exc.json(), status=422) from exc


@web.middleware
async def error_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    try:
        return await handler(request)
    except LiveRunError as exc:
        return web.json_response({'error': str(exc)}, status=exc.status)
    except web.HTTPException:
        raise
    except Exception as exc:
        logger.exception('Unhandled live runner API error')
        return web.json_response({'error': f'{type(exc).__name__}: {exc}'}, status=500)


async def create_run(request: web.Request) -> web.Response:
    manager = request.app[MANAGER_KEY]
    payload = await _read_model(request, CreateRunRequest)
    response = await manager.create_run(payload)
    return _json_response(response, status=201, request=request)


async def create_cache_run(request: web.Request) -> web.Response:
    manager = request.app[MANAGER_KEY]
    payload = await _read_model(request, CreateRunRequest)
    response = await manager.create_cache_run(payload)
    return _json_response(response, status=201, request=request)


async def create_run_from_yaml(request: web.Request) -> web.Response:
    manager = request.app[MANAGER_KEY]
    payload = await _read_model(request, CreateRunFromYamlRequest)
    try:
        import yaml

        raw = yaml.safe_load(payload.run_input_yaml)
    except Exception as exc:
        raise LiveRunError(f'invalid RunInput YAML: {exc}', status=400) from exc
    if not isinstance(raw, dict):
        raise LiveRunError('RunInput YAML must resolve to a mapping', status=400)
    try:
        run_input = RunInput.from_dict(raw)
    except Exception as exc:
        raise LiveRunError(f'invalid RunInput YAML: {exc}', status=422) from exc
    response = await manager.create_run(LiveRun.create_request_from_run_input(run_input, live=payload.live))
    return _json_response(response, status=201, request=request)


async def list_runs(request: web.Request) -> web.Response:
    manager = request.app[MANAGER_KEY]
    limit = int(request.query.get('limit', '100') or '100')
    statuses = _parse_status_filters(request.query)
    cache_key = request.query.get('cache_key') or None
    query = request.query.get('q') or None
    if _query_bool(request.query, 'summary'):
        return _json_response(
            RunSummaryListResponse(
                runs=await manager.list_run_summaries(limit=limit, statuses=statuses, cache_key=cache_key, query=query)
            ),
            request=request,
        )
    return _json_response(
        RunListResponse(
            runs=await manager.list_run_infos(limit=limit, statuses=statuses, cache_key=cache_key, query=query)
        ),
        request=request,
    )


def _query_values(query: Any, key: str) -> Iterable[str]:
    if hasattr(query, 'getall'):
        return query.getall(key, [])
    value = query.get(key) if hasattr(query, 'get') else None
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _parse_status_filters(query: Any) -> list[str] | None:
    statuses: list[str] = []
    for raw_value in _query_values(query, 'status'):
        statuses.extend(item.strip() for item in str(raw_value).split(',') if item.strip())
    invalid = sorted(set(statuses) - _VALID_RUN_STATUSES)
    if invalid:
        raise LiveRunError(f'invalid run status filter: {", ".join(invalid)}', status=400)
    return statuses or None


async def get_run(request: web.Request) -> web.Response:
    manager = request.app[MANAGER_KEY]
    return _json_response(await manager.get_run_info(request.match_info['run_id']), request=request)


async def get_run_config(request: web.Request) -> web.Response:
    manager = request.app[MANAGER_KEY]
    run_id = request.match_info['run_id']
    config = _redact_sensitive_config(await manager.get_run_request(run_id))
    return _json_response(RunConfigResponse(run_id=run_id, request=config), request=request)


async def submit_turn(request: web.Request) -> web.Response:
    manager = request.app[MANAGER_KEY]
    run = manager.get_run(request.match_info['run_id'])
    payload = await _read_model(request, SubmitTurnRequest)
    response = await run.submit_turn(payload)
    return _json_response(response, status=202, request=request)


async def kill_run(request: web.Request) -> web.Response:
    manager = request.app[MANAGER_KEY]
    response = await manager.kill_run(request.match_info['run_id'])
    return _json_response(response, request=request)


async def list_runtimes(request: web.Request) -> web.Response:
    manager = request.app[MANAGER_KEY]
    run = manager.get_run(request.match_info['run_id'])
    return _json_response(RuntimeListResponse(runtimes=run.get_info().runtimes), request=request)


def _runtime_base_url(run: LiveRun, runtime_name: str) -> str:
    runtime = run.get_runtime(runtime_name)
    base_url = runtime.base_url
    if not base_url:
        raise LiveRunError(f'run {run.run_id} runtime {runtime_name!r} is not ready: {run.status}', status=409)
    return base_url.rstrip('/')


def _proxy_headers(headers: Any) -> dict[str, str]:
    skip = {
        'connection',
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
    }
    return {key: value for key, value in headers.items() if key.lower() not in skip}


def _response_headers(headers: Any) -> dict[str, str]:
    skip = {'connection', 'content-length', 'keep-alive', 'te', 'trailer', 'transfer-encoding', 'upgrade'}
    return {key: value for key, value in headers.items() if key.lower() not in skip}


def _runtime_proxy_external_base(request: web.Request) -> str:
    return _route(
        request.app[BASE_PATH_KEY],
        (f'/v1/runs/{request.match_info["run_id"]}/runtimes/{request.match_info["runtime_name"]}/proxy'),
    )


def _rewrite_upstream_location(location: str | None, *, request: web.Request, upstream_url: str) -> str | None:
    if not location:
        return location
    proxy_base = _runtime_proxy_external_base(request)
    if location == proxy_base or location.startswith(f'{proxy_base}/'):
        return location
    if location.startswith('/'):
        return f'{proxy_base}{location}'

    upstream = urlsplit(upstream_url)
    target = urlsplit(location)
    if target.scheme and target.netloc and target.scheme == upstream.scheme and target.netloc == upstream.netloc:
        path = target.path or '/'
        rewritten = f'{proxy_base}{path}'
        if target.query:
            rewritten = f'{rewritten}?{target.query}'
        if target.fragment:
            rewritten = f'{rewritten}#{target.fragment}'
        return rewritten
    return location


def _is_websocket_request(request: web.Request) -> bool:
    return request.headers.get('Upgrade', '').lower() == 'websocket'


def _is_websocket_close_error(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, aiohttp.ClientError)):
        return True
    return isinstance(exc, RuntimeError) and any(
        text in str(exc).lower() for text in ('closing transport', 'closed', 'disconnect')
    )


async def _run_websocket_bridge(*coroutines: Any) -> None:
    tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    error: BaseException | None = None
    for task in done:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if _is_websocket_close_error(exc):
                pass
            else:
                error = exc
    for task in pending:
        task.cancel()
    for task in pending:
        with suppress(asyncio.CancelledError):
            try:
                await task
            except Exception as exc:
                if not _is_websocket_close_error(exc):
                    error = error or exc
    if error is not None:
        raise error


def _runtime_proxy_upstream_url(request: web.Request, base_url: str) -> str:
    runtime_path = request.match_info.get('runtime_path') or ''
    path = f'/{runtime_path}' if runtime_path else '/'
    upstream_url = f'{base_url}{path}'
    if request.query_string:
        upstream_url = f'{upstream_url}?{request.query_string}'
    return upstream_url


async def proxy_runtime(request: web.Request) -> web.StreamResponse:
    manager = request.app[MANAGER_KEY]
    run = manager.get_run(request.match_info['run_id'])
    runtime_name = request.match_info['runtime_name']
    upstream_url = _runtime_proxy_upstream_url(request, _runtime_base_url(run, runtime_name))

    if _is_websocket_request(request):
        return await _proxy_runtime_websocket(request, upstream_url)
    return await _proxy_runtime_http(request, upstream_url)


async def _proxy_runtime_http(request: web.Request, upstream_url: str) -> web.Response:
    body = await request.read()
    async with aiohttp.ClientSession(auto_decompress=False) as session:
        try:
            async with session.request(
                request.method,
                upstream_url,
                data=body or None,
                headers=_proxy_headers(request.headers),
                allow_redirects=False,
            ) as upstream:
                headers = _response_headers(upstream.headers)
                location = _rewrite_upstream_location(
                    headers.get('Location'), request=request, upstream_url=upstream_url
                )
                if location is not None:
                    headers['Location'] = location
                return web.Response(
                    body=await upstream.read(),
                    status=upstream.status,
                    headers=headers,
                )
        except aiohttp.ClientError as exc:
            return web.json_response(
                {
                    'error': 'Cannot connect to runtime proxy',
                    'detail': f'{type(exc).__name__}: {exc}',
                },
                status=502,
            )


async def _proxy_runtime_websocket(request: web.Request, upstream_url: str) -> web.WebSocketResponse:
    client_ws = web.WebSocketResponse()
    await client_ws.prepare(request)

    session = aiohttp.ClientSession()
    try:
        try:
            upstream_context = session.ws_connect(upstream_url, headers=_proxy_headers(request.headers))
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

            await _run_websocket_bridge(client_to_upstream(), upstream_to_client())
        finally:
            await upstream_context.__aexit__(None, None, None)
    finally:
        await session.close()
    return client_ws


async def stream_events(request: web.Request) -> web.StreamResponse:
    manager = request.app[MANAGER_KEY]
    run_id = request.match_info['run_id']
    cursor = int(request.query.get('cursor', '0') or '0')
    historical_events = None
    try:
        run = manager.get_run(run_id)
    except LiveRunError:
        run = None
        historical_events = await manager.get_run_events_after(run_id, cursor)

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

    if run is None:
        for event in historical_events or []:
            await response.write(sse_bytes(event, base_path=request.app[BASE_PATH_KEY]))
        return response

    try:
        while True:
            events = await run.events.wait_after(cursor, timeout=15.0)
            if not events:
                await response.write(b': keep-alive\n\n')
                continue
            for event in events:
                await response.write(sse_bytes(event, base_path=request.app[BASE_PATH_KEY]))
                cursor = event.id
    except (ConnectionResetError, asyncio.CancelledError):
        logger.debug('SSE client disconnected: run_id=%s', run.run_id)
    return response


async def health(_request: web.Request) -> web.Response:
    return web.json_response({'status': 'ok'})


def _render_live_ui_html(base_path: str) -> str:
    html = INDEX_HTML.read_text(encoding='utf-8')
    marker = '<script>\n'
    script = f'<script>\n    window.__NEXUS_LIVE_BASE_PATH__ = {json.dumps(base_path)};\n'
    return html.replace(marker, script, 1)


async def live_ui(request: web.Request) -> web.Response:
    """Serve the built-in live runner frontend."""
    return web.Response(
        text=_render_live_ui_html(request.app[BASE_PATH_KEY]),
        content_type='text/html',
        headers={'Cache-Control': 'no-store'},
    )


async def _cleanup(app: web.Application) -> None:
    manager = app[MANAGER_KEY]
    await manager.close()


async def _startup(app: web.Application) -> None:
    manager = app[MANAGER_KEY]
    await manager.start()


def create_app(
    *,
    manager: LiveRunManager | None = None,
    base_path: str | None = None,
    storage_path: str | Path | None = None,
    terminal_run_retention_seconds: float | None = 60.0,
) -> web.Application:
    """Create the live runner aiohttp app."""
    normalized_base_path = _normalize_base_path(base_path)
    app = web.Application(middlewares=[error_middleware])
    app[MANAGER_KEY] = manager or LiveRunManager(
        storage_path=storage_path,
        terminal_run_retention_seconds=terminal_run_retention_seconds,
    )
    app[BASE_PATH_KEY] = normalized_base_path
    app.router.add_get(_route(normalized_base_path, '/'), live_ui)
    if normalized_base_path:
        app.router.add_get(f'{normalized_base_path}/', live_ui)
    app.router.add_get(_route(normalized_base_path, '/ui'), live_ui)
    app.router.add_get(_route(normalized_base_path, '/health'), health)
    app.router.add_get(_route(normalized_base_path, '/v1/runs'), list_runs)
    app.router.add_post(_route(normalized_base_path, '/v1/runs'), create_run)
    app.router.add_post(_route(normalized_base_path, '/v1/runs/cache'), create_cache_run)
    app.router.add_post(_route(normalized_base_path, '/v1/runs/yaml'), create_run_from_yaml)
    app.router.add_get(_route(normalized_base_path, '/v1/runs/{run_id}'), get_run)
    app.router.add_get(_route(normalized_base_path, '/v1/runs/{run_id}/config'), get_run_config)
    app.router.add_post(_route(normalized_base_path, '/v1/runs/{run_id}/kill'), kill_run)
    app.router.add_delete(_route(normalized_base_path, '/v1/runs/{run_id}'), kill_run)
    app.router.add_get(_route(normalized_base_path, '/v1/runs/{run_id}/events'), stream_events)
    app.router.add_post(_route(normalized_base_path, '/v1/runs/{run_id}/turns'), submit_turn)
    app.router.add_get(_route(normalized_base_path, '/v1/runs/{run_id}/runtimes'), list_runtimes)
    app.router.add_route(
        '*',
        _route(normalized_base_path, '/v1/runs/{run_id}/runtimes/{runtime_name}/proxy'),
        proxy_runtime,
    )
    app.router.add_route(
        '*',
        _route(normalized_base_path, '/v1/runs/{run_id}/runtimes/{runtime_name}/proxy/{runtime_path:.*}'),
        proxy_runtime,
    )
    app.on_startup.append(_startup)
    app.on_cleanup.append(_cleanup)
    return app
