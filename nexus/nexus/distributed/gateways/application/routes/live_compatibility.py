from __future__ import annotations

import asyncio
import json
import logging

from collections.abc import Iterable
from contextlib import suppress
from typing import Any, TypeVar
from urllib.parse import urlsplit

import aiohttp

from aiohttp import web
from aiohttp.web_app import Application as AiohttpApplication
from pydantic import BaseModel, ValidationError

from nexus.distributed.access.auth import (
    AUTH_CONTEXT_KEY,
    DELEGATED_APP_ID_HEADER,
    DELEGATED_PRINCIPAL_ID_HEADER,
    INTERNAL_TOKEN_HEADER,
    AuthContext,
    websocket_requested_protocols,
)
from nexus.distributed.access.taihu import SGW_AUTH_HEADERS, TAI_IDENTITY_HEADER
from nexus.distributed.compatibility.live import (
    cache_key as _cache_key,
)
from nexus.distributed.compatibility.live import (
    create_submission_payload,
    run_info_from_execution,
    run_summary_from_execution,
    runtime_info_from_execution,
    runtime_info_from_payload,
    valid_run_statuses,
)
from nexus.distributed.compatibility.live import (
    filter_executions as _filter_executions,
)
from nexus.distributed.compatibility.live import (
    int_or_none as _int_or_none,
)
from nexus.distributed.compatibility.live import (
    request_from_execution as _request_from_execution,
)
from nexus.distributed.compatibility.live import (
    status_from_execution as _status_from_execution,
)
from nexus.distributed.execution.models import ExecutionMode, ExecutionStatus, RouteStatus
from nexus.distributed.execution.trace_backends import is_live_result_pointer
from nexus.distributed.gateways.common.proxy import read_request_body_limited
from nexus.distributed.utils.identity import new_id
from nexus.runner.live.manager import LiveRun
from nexus.runner.live.models import (
    CreateRunFromYamlRequest,
    CreateRunRequest,
    CreateRunResponse,
    RunConfigResponse,
    RunListResponse,
    RunSummaryListResponse,
    RuntimeListResponse,
    SubmitTurnRequest,
    SubmitTurnResponse,
    request_cache_key,
)
from nexus.runner.live.server import _redact_sensitive_config
from nexus.runner.single import RunInput


logger = logging.getLogger(__name__)
CONTROL_BASE_URL_KEY = web.AppKey('control_base_url', str)
APPLICATION_BASE_URL_KEY = web.AppKey('application_base_url', str)
DEFAULT_PLACEMENT_KEY = web.AppKey('default_placement', dict)
HTTP_CLIENT_KEY = web.AppKey('http_client', aiohttp.ClientSession)
INTERNAL_TOKEN_KEY = web.AppKey('internal_token', str)
STREAM_CLIENT_TIMEOUT = aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=300)
PUBLIC_RUNS_BASE_PATH_KEY = web.AppKey('public_runs_base_path', str)
PROXY_BODY_MAX_SIZE_KEY = web.AppKey('proxy_body_max_size', int)
ModelT = TypeVar('ModelT', bound=BaseModel)
_VALID_RUN_STATUSES = set(valid_run_statuses())
_IDENTITY_HEADER_NAMES = frozenset({
    TAI_IDENTITY_HEADER,
    INTERNAL_TOKEN_HEADER.lower(),
    DELEGATED_APP_ID_HEADER.lower(),
    DELEGATED_PRINCIPAL_ID_HEADER.lower(),
    'x-nexus-identity-proxy-token',
    *SGW_AUTH_HEADERS,
})


class CompatibilityError(Exception):
    def __init__(self, error: str, *, status: int = 500, detail: Any = None) -> None:
        super().__init__(error)
        self.error = error
        self.status = status
        self.detail = detail


@web.middleware
async def error_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    try:
        return await handler(request)
    except CompatibilityError as exc:
        payload: dict[str, Any] = {'error': exc.error}
        if exc.detail is not None:
            payload['detail'] = exc.detail
        return web.json_response(payload, status=exc.status)
    except web.HTTPException:
        raise
    except Exception:
        logger.exception('Unhandled live compatibility API error')
        return web.json_response({'error': 'internal_error'}, status=500)


async def _read_model(request: web.Request, model: type[ModelT]) -> ModelT:
    data = await _read_json(request)
    return _model_from_data(data, model)


async def _read_json(request: web.Request) -> dict[str, Any]:
    try:
        data = await request.json()
    except json.JSONDecodeError as exc:
        raise CompatibilityError(f'invalid JSON body: {exc}', status=400) from exc
    if not isinstance(data, dict):
        raise CompatibilityError('JSON body must be an object', status=400)
    return data


def _model_from_data(data: dict[str, Any], model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise CompatibilityError(exc.json(), status=422) from exc


async def _upstream_json(
    request: web.Request,
    method: str,
    base_url_key: web.AppKey[str],
    path: str,
    *,
    body: dict[str, Any] | None = None,
    query: dict[str, str] | None = None,
    expected_statuses: set[int] | None = None,
) -> tuple[dict[str, Any], int]:
    app = request.app
    session = app[HTTP_CLIENT_KEY]
    url = f'{app[base_url_key].rstrip("/")}{path}'
    headers = _upstream_auth_headers(request)
    async with session.request(
        method,
        url,
        json=body,
        params=query,
        headers=headers,
    ) as response:
        text = await response.text()
        try:
            payload = json.loads(text) if text else {}
        except json.JSONDecodeError:
            payload = {'body': text}
        if expected_statuses is not None and response.status not in expected_statuses:
            raise CompatibilityError(
                'upstream_error',
                status=response.status,
                detail=_redact_upstream_detail(payload, secrets=headers.values()),
            )
        if expected_statuses is None and response.status >= 400:
            error = payload.get('error') if isinstance(payload, dict) else None
            raise CompatibilityError(
                str(error or 'upstream_error'),
                status=response.status,
                detail=_redact_upstream_detail(payload, secrets=headers.values()),
            )
        if not isinstance(payload, dict):
            raise CompatibilityError('invalid_upstream_response', detail=payload)
        return payload, response.status


def _create_submission_payload(
    payload: CreateRunRequest,
    *,
    app: web.Application,
    idempotency_key: str,
    clear_run_id: bool = False,
) -> dict[str, Any]:
    return create_submission_payload(
        payload,
        placement=app[DEFAULT_PLACEMENT_KEY],
        idempotency_key=idempotency_key,
        clear_run_id=clear_run_id,
    )


def _runtime_base_path(app: web.Application, run_id: str, runtime_name: str) -> str:
    return f'{app[PUBLIC_RUNS_BASE_PATH_KEY]}/{run_id}/runtimes/{runtime_name}/proxy'


def _runtime_info_from_payload(
    app: web.Application,
    run_id: str,
    runtime_info: dict[str, Any] | None,
):
    return runtime_info_from_payload(app[PUBLIC_RUNS_BASE_PATH_KEY], run_id, runtime_info)


def _runtime_info_from_execution(app: web.Application, execution: dict[str, Any]):
    return runtime_info_from_execution(app[PUBLIC_RUNS_BASE_PATH_KEY], execution)


def _run_info_from_execution(
    app: web.Application,
    execution: dict[str, Any],
    active_info: dict[str, Any] | None = None,
):
    return run_info_from_execution(app[PUBLIC_RUNS_BASE_PATH_KEY], execution, active_info)


def _run_summary_from_execution(app: web.Application, execution: dict[str, Any]):
    return run_summary_from_execution(app[PUBLIC_RUNS_BASE_PATH_KEY], execution)


def _json_response(model_or_data: BaseModel | dict[str, Any], *, status: int = 200) -> web.Response:
    data = model_or_data.model_dump(mode='json') if isinstance(model_or_data, BaseModel) else model_or_data
    return web.json_response(data, status=status)


async def create_run(request: web.Request) -> web.Response:
    data = await _read_json(request)
    payload = _model_from_data(data, CreateRunRequest)
    response, _ = await _upstream_json(
        request,
        'POST',
        CONTROL_BASE_URL_KEY,
        '/v1/submissions/live',
        body=_create_submission_payload(
            payload,
            app=request.app,
            idempotency_key=f'legacy-live:{new_id("req")}',
        ),
        expected_statuses={202},
    )
    return _json_response(
        CreateRunResponse(
            run_id=response['execution_id'],
            status=_status_from_execution(response),
            events_cursor=0,
            reused_cache=bool(response.get('reused')),
            cache_key=request_cache_key(payload),
        ),
        status=201,
    )


async def create_cache_run(request: web.Request) -> web.Response:
    data = await _read_json(request)
    payload = _model_from_data(data, CreateRunRequest)
    cache_key = _cache_key(payload)
    response, _ = await _upstream_json(
        request,
        'POST',
        CONTROL_BASE_URL_KEY,
        '/v1/submissions/live',
        body=_create_submission_payload(
            payload,
            app=request.app,
            idempotency_key=f'legacy-live-cache:{cache_key}',
            clear_run_id=True,
        ),
        expected_statuses={202},
    )
    return _json_response(
        CreateRunResponse(
            run_id=response['execution_id'],
            status=_status_from_execution(response),
            events_cursor=0,
            reused_cache=bool(response.get('reused')),
            cache_key=cache_key,
        ),
        status=201,
    )


async def create_run_from_yaml(request: web.Request) -> web.Response:
    data = await _read_json(request)
    payload = _model_from_data(data, CreateRunFromYamlRequest)
    try:
        import yaml

        raw = yaml.safe_load(payload.run_input_yaml)
    except Exception as exc:
        raise CompatibilityError(f'invalid RunInput YAML: {exc}', status=400) from exc
    if not isinstance(raw, dict):
        raise CompatibilityError('RunInput YAML must resolve to a mapping', status=400)
    try:
        run_input = RunInput.from_dict(raw)
    except Exception as exc:
        raise CompatibilityError(f'invalid RunInput YAML: {exc}', status=422) from exc
    create_payload = LiveRun.create_request_from_run_input(run_input, live=payload.live)
    response, _ = await _upstream_json(
        request,
        'POST',
        CONTROL_BASE_URL_KEY,
        '/v1/submissions/live',
        body=_create_submission_payload(
            create_payload,
            app=request.app,
            idempotency_key=f'legacy-live:{new_id("req")}',
        ),
        expected_statuses={202},
    )
    return _json_response(
        CreateRunResponse(
            run_id=response['execution_id'],
            status=_status_from_execution(response),
            events_cursor=0,
            reused_cache=bool(response.get('reused')),
            cache_key=request_cache_key(create_payload),
        ),
        status=201,
    )


async def list_runs(request: web.Request) -> web.Response:
    statuses = _parse_status_filters(request.query)
    cache_key = request.query.get('cache_key') or None
    text_query = request.query.get('q') or None
    limit = _query_limit(request.query)
    executions = await _list_filtered_executions(
        request,
        limit=limit,
        statuses=statuses,
        cache_key=cache_key,
        text_query=text_query,
    )
    if _query_bool(request.query, 'summary'):
        return _json_response(
            RunSummaryListResponse(runs=[_run_summary_from_execution(request.app, item) for item in executions])
        )
    return _json_response(RunListResponse(runs=[_run_info_from_execution(request.app, item) for item in executions]))


async def get_run(request: web.Request) -> web.Response:
    execution = await _get_execution(request)
    active_info = await _get_active_execution_info(request, execution)
    await _hydrate_external_result(request, execution)
    return _json_response(_run_info_from_execution(request.app, execution, active_info=active_info))


async def get_run_config(request: web.Request) -> web.Response:
    execution = await _get_execution(request)
    config = _redact_sensitive_config(_request_from_execution(execution))
    return _json_response(RunConfigResponse(run_id=execution['execution_id'], request=config))


async def submit_turn(request: web.Request) -> web.Response:
    payload = await _read_model(request, SubmitTurnRequest)
    run_id = request.match_info['run_id']
    command, _ = await _upstream_json(
        request,
        'POST',
        APPLICATION_BASE_URL_KEY,
        f'/v1/executions/{run_id}/commands',
        body={'type': 'input.submit', 'payload': {'text': payload.text}, 'idempotency_key': None},
        expected_statuses={202},
    )
    execution = await _get_execution(request)
    return _json_response(
        SubmitTurnResponse(
            run_id=run_id,
            turn_id=str(command.get('command_id') or ''),
            status=_status_from_execution(execution),
        ),
        status=202,
    )


async def kill_run(request: web.Request) -> web.Response:
    run_id = request.match_info['run_id']
    execution, _ = await _upstream_json(
        request,
        'POST',
        CONTROL_BASE_URL_KEY,
        f'/v1/executions/{run_id}/stop',
    )
    return _json_response(_run_info_from_execution(request.app, execution))


async def list_runtimes(request: web.Request) -> web.Response:
    execution = await _get_execution(request)
    active_info = await _get_active_execution_info(request, execution)
    if isinstance(active_info, dict) and isinstance(active_info.get('runtime_info'), dict):
        runtimes = _runtime_info_from_payload(
            request.app,
            execution['execution_id'],
            active_info['runtime_info'],
        )
    else:
        runtimes = _runtime_info_from_execution(request.app, execution)
    return _json_response(RuntimeListResponse(runtimes=runtimes))


async def stream_events(request: web.Request) -> web.StreamResponse:
    run_id = request.match_info['run_id']
    query = f'?{request.query_string}' if request.query_string else ''
    upstream_url = f'{request.app[APPLICATION_BASE_URL_KEY].rstrip("/")}/v1/executions/{run_id}/live/events{query}'
    session = request.app[HTTP_CLIENT_KEY]
    try:
        async with session.get(
            upstream_url,
            headers=_proxy_headers(request),
            allow_redirects=False,
            timeout=STREAM_CLIENT_TIMEOUT,
        ) as upstream:
            if upstream.status >= 400:
                return web.Response(
                    body=await upstream.read(),
                    status=upstream.status,
                    headers=_response_headers(upstream.headers),
                )
            response = web.StreamResponse(status=upstream.status, headers=_response_headers(upstream.headers))
            await response.prepare(request)
            marker = f'/v1/executions/{run_id}/runtimes/'.encode()
            replacement = f'{request.app[PUBLIC_RUNS_BASE_PATH_KEY]}/{run_id}/runtimes/'.encode()
            async for chunk in upstream.content.iter_chunked(8192):
                await response.write(chunk.replace(marker, replacement))
            return response
    except aiohttp.ClientError as exc:
        return web.json_response(
            {'error': 'events_unavailable', 'detail': f'{type(exc).__name__}: {exc}'},
            status=503,
        )


async def proxy_runtime(request: web.Request) -> web.StreamResponse:
    run_id = request.match_info['run_id']
    runtime_name = request.match_info['runtime_name']
    tail = request.match_info.get('runtime_path') or ''
    _validate_proxy_path(runtime_name)
    _validate_proxy_path(tail)
    upstream_path = f'/v1/executions/{run_id}/runtimes/{runtime_name}/proxy'
    if tail:
        upstream_path = f'{upstream_path}/{tail}'
    upstream_url = f'{request.app[APPLICATION_BASE_URL_KEY].rstrip("/")}{upstream_path}'
    if request.query_string:
        upstream_url = f'{upstream_url}?{request.query_string}'
    if _is_websocket_request(request):
        return await _proxy_websocket(request, upstream_url)
    return await _proxy_http(request, upstream_url)


async def health(_request: web.Request) -> web.Response:
    return web.json_response({'status': 'ok'})


def _query_bool(query: Any, key: str, *, default: bool = False) -> bool:
    value = query.get(key) if hasattr(query, 'get') else None
    if value is None:
        return default
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


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
        raise CompatibilityError(f'invalid run status filter: {", ".join(invalid)}', status=400)
    return statuses or None


def _query_limit(query: Any) -> int:
    try:
        return max(1, int(query.get('limit', '100') or '100'))
    except (TypeError, ValueError) as exc:
        raise CompatibilityError('invalid limit', status=400) from exc


async def _list_filtered_executions(
    request: web.Request,
    *,
    limit: int,
    statuses: list[str] | None,
    cache_key: str | None,
    text_query: str | None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    offset = 0
    pages = 0
    while len(result) < limit and pages < 1000:
        query = {'mode': ExecutionMode.LIVE, 'limit': str(limit), 'offset': str(offset)}
        payload, _ = await _upstream_json(request, 'GET', CONTROL_BASE_URL_KEY, '/v1/executions', query=query)
        items = payload.get('items') or []
        if not isinstance(items, list) or not items:
            break
        result.extend(
            _filter_executions(
                items,
                statuses=statuses,
                cache_key=cache_key,
                text_query=text_query,
            )
        )
        offset += len(items)
        pages += 1
        total = _int_or_none(payload.get('total'))
        upstream_limit = _int_or_none(payload.get('limit')) or limit
        if len(items) < upstream_limit or (total is not None and offset >= total):
            break
    return result[:limit]


async def _get_execution(request: web.Request) -> dict[str, Any]:
    payload, _ = await _upstream_json(
        request,
        'GET',
        CONTROL_BASE_URL_KEY,
        f'/v1/executions/{request.match_info["run_id"]}',
    )
    return payload


async def _get_active_execution_info(request: web.Request, execution: dict[str, Any]) -> dict[str, Any] | None:
    if execution.get('route_status') != RouteStatus.ACTIVE:
        return None
    if execution.get('status') in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.KILLED}:
        return None
    try:
        payload, _ = await _upstream_json(
            request,
            'GET',
            APPLICATION_BASE_URL_KEY,
            f'/v1/executions/{execution["execution_id"]}/info',
        )
        return payload
    except Exception:
        logger.debug('Failed to fetch active distributed execution info', exc_info=True)
        return None


async def _hydrate_external_result(request: web.Request, execution: dict[str, Any]) -> None:
    if not is_live_result_pointer(execution.get('result_json')):
        return
    payload, _ = await _upstream_json(
        request,
        'GET',
        APPLICATION_BASE_URL_KEY,
        f'/v1/executions/{execution["execution_id"]}/live/result',
    )
    execution['result_json'] = payload


def _proxy_headers(request: web.Request) -> dict[str, str]:
    skip = {
        'authorization',
        'connection',
        'content-length',
        'cookie',
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
        *_IDENTITY_HEADER_NAMES,
    }
    forwarded = {key: value for key, value in request.headers.items() if key.lower() not in skip}
    forwarded.update(_upstream_auth_headers(request))
    return forwarded


def _upstream_auth_headers(request: web.Request) -> dict[str, str]:
    context = request.get(AUTH_CONTEXT_KEY)
    if not isinstance(context, AuthContext):
        if any(key.lower() in _IDENTITY_HEADER_NAMES for key in request.headers):
            raise web.HTTPUnauthorized(text='authentication_required')
        return {}
    if context.auth_type == 'disabled':
        return {}
    internal_token = request.app[INTERNAL_TOKEN_KEY]
    if not internal_token:
        raise RuntimeError('internal identity delegation is unavailable')
    if context.auth_type == 'tai_app' and context.actor_app_id:
        return {
            INTERNAL_TOKEN_HEADER: internal_token,
            DELEGATED_APP_ID_HEADER: context.actor_app_id,
        }
    if context.auth_type == 'tai_identity':
        return {
            INTERNAL_TOKEN_HEADER: internal_token,
            DELEGATED_PRINCIPAL_ID_HEADER: context.principal_id,
        }
    raise web.HTTPUnauthorized(text='authentication_required')


def _redact_upstream_detail(value: Any, *, secrets: Iterable[str]) -> Any:
    secret_values = tuple(secret for secret in secrets if secret)
    if isinstance(value, dict):
        return {
            key: '[REDACTED]'
            if key.lower()
            in {
                'authorization',
                'token',
                'submit_token',
                *_IDENTITY_HEADER_NAMES,
            }
            else _redact_upstream_detail(item, secrets=secret_values)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_upstream_detail(item, secrets=secret_values) for item in value]
    if isinstance(value, str):
        for secret in secret_values:
            value = value.replace(secret, '[REDACTED]')
    return value


def _response_headers(headers: Any) -> dict[str, str]:
    skip = {'connection', 'content-length', 'keep-alive', 'te', 'trailer', 'transfer-encoding', 'upgrade'}
    return {key: value for key, value in headers.items() if key.lower() not in skip}


def _is_websocket_request(request: web.Request) -> bool:
    return request.headers.get('Upgrade', '').lower() == 'websocket'


def _validate_proxy_path(value: str) -> None:
    if '\x00' in value or '\\' in value or any(part in {'.', '..'} for part in value.split('/')):
        raise web.HTTPBadRequest(text='invalid_proxy_path')


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
            if not _is_websocket_close_error(exc):
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


async def _proxy_http(request: web.Request, upstream_url: str) -> web.StreamResponse:
    body = await read_request_body_limited(
        request,
        max_size_bytes=request.app[PROXY_BODY_MAX_SIZE_KEY],
    )
    session = request.app[HTTP_CLIENT_KEY]
    try:
        async with session.request(
            request.method,
            upstream_url,
            data=body or None,
            headers=_proxy_headers(request),
            allow_redirects=False,
        ) as upstream:
            headers = _response_headers(upstream.headers)
            location = _rewrite_upstream_location(headers.get('Location'), request=request, upstream_url=upstream_url)
            if location is not None:
                headers['Location'] = location
            response = web.StreamResponse(status=upstream.status, headers=headers)
            await response.prepare(request)
            async for chunk in upstream.content.iter_chunked(8192):
                await response.write(chunk)
            return response
    except aiohttp.ClientError as exc:
        logger.warning('Compatibility runtime proxy unavailable: %s', type(exc).__name__)
        return web.json_response({'error': 'runtime_proxy_unavailable'}, status=502)


async def _proxy_websocket(request: web.Request, upstream_url: str) -> web.StreamResponse:
    session = request.app[HTTP_CLIENT_KEY]
    try:
        upstream_context = session.ws_connect(
            upstream_url,
            headers=_proxy_headers(request),
            protocols=websocket_requested_protocols(request),
        )
        upstream_ws = await upstream_context.__aenter__()
    except aiohttp.ClientError as exc:
        raise CompatibilityError('runtime_proxy_unavailable', status=502) from exc
    selected_protocols = (upstream_ws.protocol,) if upstream_ws.protocol else ()
    client_ws = web.WebSocketResponse(protocols=selected_protocols)
    try:
        await client_ws.prepare(request)

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
    return client_ws


def _rewrite_upstream_location(location: str | None, *, request: web.Request, upstream_url: str) -> str | None:
    if not location:
        return location
    run_id = request.match_info['run_id']
    runtime_name = request.match_info['runtime_name']
    proxy_base = _runtime_base_path(request.app, run_id, runtime_name)
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


async def _http_client_context(app: web.Application):
    app[HTTP_CLIENT_KEY] = aiohttp.ClientSession(
        auto_decompress=False,
        connector=aiohttp.TCPConnector(limit=512, ttl_dns_cache=300),
    )
    try:
        yield
    finally:
        await app[HTTP_CLIENT_KEY].close()


def create_live_compatibility_app(
    *,
    control_base_url: str,
    application_base_url: str,
    default_placement: dict[str, Any] | None = None,
    route_prefix: str = '/v1/runs',
    public_runs_base_path: str | None = None,
    health_path: str | None = '/v1/health',
    instrument_http_server: bool = True,
    json_body_max_size_bytes: int = 1024**2,
    proxy_body_max_size_bytes: int = 8 * 1024**2,
    internal_token: str = '',
) -> web.Application:
    placement = dict(default_placement or {})
    if not isinstance(placement.get('environment'), str) or not placement['environment']:
        raise ValueError('live compatibility requires a default environment')
    if not isinstance(placement.get('pool'), str) or not placement['pool']:
        raise ValueError('live compatibility requires a default pool')

    # The process-wide aiohttp server instrumentor replaces ``web.Application``.
    # A mounted sub-application is already covered by its parent's middleware;
    # instrumenting both creates two sibling SERVER spans for one request.
    application_type = web.Application if instrument_http_server else AiohttpApplication
    app = application_type(
        middlewares=[error_middleware],
        client_max_size=json_body_max_size_bytes,
    )
    app[CONTROL_BASE_URL_KEY] = control_base_url.rstrip('/')
    app[APPLICATION_BASE_URL_KEY] = application_base_url.rstrip('/')
    app[DEFAULT_PLACEMENT_KEY] = placement
    app[INTERNAL_TOKEN_KEY] = internal_token
    app[PROXY_BODY_MAX_SIZE_KEY] = proxy_body_max_size_bytes
    route_prefix = '/' + route_prefix.strip('/')
    app[PUBLIC_RUNS_BASE_PATH_KEY] = (public_runs_base_path or route_prefix).rstrip('/')
    app.cleanup_ctx.append(_http_client_context)
    if health_path is not None:
        app.router.add_get(health_path, health)
    app.router.add_get(route_prefix, list_runs)
    app.router.add_post(route_prefix, create_run)
    app.router.add_post(f'{route_prefix}/cache', create_cache_run)
    app.router.add_post(f'{route_prefix}/yaml', create_run_from_yaml)
    app.router.add_get(f'{route_prefix}/{{run_id}}', get_run)
    app.router.add_get(f'{route_prefix}/{{run_id}}/config', get_run_config)
    app.router.add_post(f'{route_prefix}/{{run_id}}/kill', kill_run)
    app.router.add_delete(f'{route_prefix}/{{run_id}}', kill_run)
    app.router.add_get(f'{route_prefix}/{{run_id}}/events', stream_events)
    app.router.add_post(f'{route_prefix}/{{run_id}}/turns', submit_turn)
    app.router.add_get(f'{route_prefix}/{{run_id}}/runtimes', list_runtimes)
    app.router.add_route('*', f'{route_prefix}/{{run_id}}/runtimes/{{runtime_name}}/proxy', proxy_runtime)
    app.router.add_route(
        '*',
        f'{route_prefix}/{{run_id}}/runtimes/{{runtime_name}}/proxy/{{runtime_path:.*}}',
        proxy_runtime,
    )
    return app
