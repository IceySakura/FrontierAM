from __future__ import annotations

import json
import time

from asyncio import TimeoutError as AsyncTimeoutError
from datetime import timedelta
from typing import Any

import aiohttp

from aiohttp import web

from nexus.distributed.access.auth import AuthContext, auth_context
from nexus.distributed.execution.models import TERMINAL_EXECUTION_STATUSES
from nexus.distributed.gateways.common.keys import (
    AUTHORIZATION_KEY,
    CONFIG_KEY,
    EXECUTION_LOG_METRICS_KEY,
    HTTP_CLIENT_KEY,
    SESSION_FACTORY_KEY,
)
from nexus.distributed.gateways.common.proxy import internal_request_headers
from nexus.distributed.gateways.common.serde import to_jsonable
from nexus.distributed.gateways.control.execution_logs import (
    AVAILABILITY_INGEST_PENDING,
    AVAILABILITY_NOT_STARTED,
    AVAILABILITY_READY,
    AVAILABILITY_UNAVAILABLE,
    ERROR_MANAGER_NOT_STARTED,
    ERROR_MANAGER_UNAVAILABLE,
    ERROR_ZHIYAN_INGEST_PENDING,
    ERROR_ZHIYAN_NOT_CONFIGURED,
    ERROR_ZHIYAN_QUERY_LIMITED,
    SOURCE_INSTANCE_MANAGER,
    SOURCE_NONE,
    SOURCE_ZHIYAN,
    ExecutionLogLine,
    ExecutionLogPayload,
    ZhiyanLogClient,
    ZhiyanLogError,
    normalize_zhiyan_records,
    read_remote_log,
)
from nexus.distributed.gateways.control.routes._manager_logs import proxy_manager_log_stream, proxy_manager_log_tail
from nexus.distributed.gateways.control.routes._shared import query_int as _query_int
from nexus.distributed.persistence.orm import InstanceManagerRow
from nexus.distributed.repositories.allocation import AllocationRepository
from nexus.distributed.repositories.environment import EnvironmentRepository
from nexus.distributed.repositories.execution import ExecutionRepository
from nexus.distributed.utils.clock import utcnow


async def _get_authorized_execution_log_context(request: web.Request, session: Any, context: AuthContext):
    config = request.app[CONFIG_KEY]
    execution_context = await ExecutionRepository(session, shard_count=config.shard_count).get_log_context(
        request.match_info['execution_id'],
        owner_principal_ids=request.app[AUTHORIZATION_KEY].accessible_owner_ids(context),
        environment_id=None,
    )
    execution = execution_context.execution
    if execution.owner_principal_id is None:
        raise web.HTTPNotFound(text='resource_not_found')
    request.app[AUTHORIZATION_KEY].require_execution_access(
        context,
        owner_principal_id=execution.owner_principal_id,
        environment_id=execution.environment_id,
    )
    if not context.is_admin and context.auth_type != 'disabled':
        if execution.environment_id is None:
            raise web.HTTPNotFound(text='resource_not_found')
        await EnvironmentRepository(session).require_grant_scope(
            execution.owner_principal_id,
            execution.environment_id,
            required_scope='execution:read',
        )
    return execution_context


async def _authorized_execution_manager_endpoint(request: web.Request) -> str:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_scope(context, 'execution:read')
    async with request.app[SESSION_FACTORY_KEY]() as session:
        log_context = await _get_authorized_execution_log_context(request, session, context)
        execution = log_context.execution
        assignment = execution.current_assignment
        if assignment is None:
            raise web.HTTPNotFound(text='manager_logs_not_available')
        manager = await session.get(InstanceManagerRow, assignment.manager_id)
        if manager is None:
            raise web.HTTPNotFound(text='manager_logs_not_available')
        return manager.endpoint_url.rstrip('/')


async def get_execution_manager_logs(request: web.Request) -> web.Response:
    lines = _query_int(request, 'lines', 200, min_value=1, max_value=5000)
    endpoint_url = await _authorized_execution_manager_endpoint(request)
    return await proxy_manager_log_tail(request, endpoint_url=endpoint_url, lines=lines)


async def stream_execution_manager_logs(request: web.Request) -> web.StreamResponse:
    lines = _query_int(request, 'lines', 200, min_value=1, max_value=5000)
    endpoint_url = await _authorized_execution_manager_endpoint(request)
    return await proxy_manager_log_stream(request, endpoint_url=endpoint_url, lines=lines)


def _execution_log_json(payload: ExecutionLogPayload) -> dict[str, Any]:
    result = to_jsonable(payload)
    result['line_count'] = len(payload.lines)
    return result


def _log_payload(
    *,
    execution_id: str,
    status: str,
    source: str,
    availability: str,
    lines: list[ExecutionLogLine] | None = None,
    partial: bool = False,
    retry_after_ms: int | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    return _execution_log_json(
        ExecutionLogPayload(
            execution_id=execution_id,
            execution_status=status,
            source=source,
            availability=availability,
            partial=partial,
            lines=lines or [],
            fetched_at=utcnow(),
            retry_after_ms=retry_after_ms,
            error_code=error_code,
        )
    )


async def get_execution_logs(request: web.Request) -> web.Response:
    started = time.monotonic()
    config = request.app[CONFIG_KEY]
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_scope(context, 'execution:read')
    lines_limit = min(1000, config.zhiyan_log_query.max_lines)
    lines = _query_int(request, 'lines', 200, min_value=1, max_value=lines_limit)
    async with request.app[SESSION_FACTORY_KEY]() as session:
        log_context = await _get_authorized_execution_log_context(request, session, context)
        execution = log_context.execution
        if execution.status not in TERMINAL_EXECUTION_STATUSES:
            response = await _get_active_execution_logs(request, execution, lines)
        else:
            response = await _get_terminal_execution_logs(request, session, log_context, lines)
    _record_execution_log_metrics(request, response, latency_seconds=time.monotonic() - started)
    return response


def _record_execution_log_metrics(request: web.Request, response: web.Response, *, latency_seconds: float) -> None:
    body = response.body
    if not isinstance(body, (bytes, bytearray)):
        return
    try:
        payload = json.loads(body)
        request.app[EXECUTION_LOG_METRICS_KEY].record_query(
            source=str(payload.get('source', 'unknown')),
            result=str(payload.get('availability', 'unknown')),
            latency_seconds=max(0.0, latency_seconds),
            records=int(payload.get('line_count', 0)),
            response_bytes=len(body),
        )
    except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return


async def _get_active_execution_logs(request: web.Request, execution: Any, lines: int) -> web.Response:
    execution_id = execution.execution_id
    assignment = execution.current_assignment
    if assignment is None:
        return web.json_response(
            _log_payload(
                execution_id=execution_id,
                status=execution.status,
                source=SOURCE_NONE,
                availability=AVAILABILITY_NOT_STARTED,
                error_code=ERROR_MANAGER_NOT_STARTED,
            )
        )
    manager_id = assignment.manager_id
    async with request.app[SESSION_FACTORY_KEY]() as session:
        manager = await session.get(InstanceManagerRow, manager_id)
    if manager is None:
        return web.json_response(
            _log_payload(
                execution_id=execution_id,
                status=execution.status,
                source=SOURCE_INSTANCE_MANAGER,
                availability=AVAILABILITY_UNAVAILABLE,
                error_code=ERROR_MANAGER_UNAVAILABLE,
            )
        )
    url = f'{manager.endpoint_url.rstrip("/")}/v1/internal/logs?lines={lines}&format=jsonl'
    config = request.app[CONFIG_KEY]
    try:
        parsed, partial = await read_remote_log(
            request.app[HTTP_CLIENT_KEY],
            url,
            headers=internal_request_headers(config),
            lines=lines,
        )
    except (aiohttp.ClientError, AsyncTimeoutError):
        return web.json_response(
            _log_payload(
                execution_id=execution_id,
                status=execution.status,
                source=SOURCE_INSTANCE_MANAGER,
                availability=AVAILABILITY_UNAVAILABLE,
                error_code=ERROR_MANAGER_UNAVAILABLE,
            ),
            status=502,
        )
    return web.json_response(
        _log_payload(
            execution_id=execution_id,
            status=execution.status,
            source=SOURCE_INSTANCE_MANAGER,
            availability=AVAILABILITY_READY,
            lines=parsed,
            partial=partial,
        )
    )


async def _get_terminal_execution_logs(
    request: web.Request, session: Any, log_context: Any, lines: int
) -> web.Response:
    config = request.app[CONFIG_KEY]
    execution = log_context.execution

    def empty_response(availability: str, error_code: str, retry_after_ms: int | None = None) -> web.Response:
        return web.json_response(
            _log_payload(
                execution_id=execution.execution_id,
                status=execution.status,
                source=SOURCE_ZHIYAN,
                availability=availability,
                error_code=error_code,
                retry_after_ms=retry_after_ms,
            )
        )

    if not config.zhiyan_log_query.enabled or execution.allocation_id is None:
        return empty_response(AVAILABILITY_UNAVAILABLE, ERROR_ZHIYAN_NOT_CONFIGURED)
    binding = await AllocationRepository(session).zhiyan_log_binding(
        execution.allocation_id,
        predefined_log_name=config.predefined_manager_zhiyan_log_name or None,
    )
    if binding is None:
        return empty_response(AVAILABILITY_UNAVAILABLE, ERROR_ZHIYAN_NOT_CONFIGURED)
    try:
        token = config.read_zhiyan_log_credential(config.zhiyan_log_query.credential_ref)
    except ValueError:
        return empty_response(AVAILABILITY_UNAVAILABLE, ERROR_ZHIYAN_NOT_CONFIGURED)
    terminal_at = log_context.finished_at or execution.updated_at
    now = utcnow()
    end_time = min(now, terminal_at + timedelta(seconds=config.zhiyan_log_query.ingest_grace_seconds))
    start_time = max(
        execution.created_at - timedelta(seconds=60),
        end_time - timedelta(seconds=config.zhiyan_log_query.query_window_seconds),
    )
    try:
        records = await ZhiyanLogClient(
            session=request.app[HTTP_CLIENT_KEY],
            config=config.zhiyan_log_query,
            token=token,
        ).query(
            log_name=binding.log_name,
            filters=[{'key': 'execution_id', 'operator': 'eq', 'value': execution.execution_id}],
            start_time=start_time,
            end_time=end_time,
            limit=lines,
        )
    except ZhiyanLogError as exc:
        error_code = exc.error_code
        return web.json_response(
            _log_payload(
                execution_id=execution.execution_id,
                status=execution.status,
                source=SOURCE_ZHIYAN,
                availability=AVAILABILITY_UNAVAILABLE,
                error_code=error_code,
            ),
            status=429 if error_code == ERROR_ZHIYAN_QUERY_LIMITED else 502,
        )
    normalized, partial = normalize_zhiyan_records(records, max_lines=lines)
    if not normalized:
        pending = (now - terminal_at).total_seconds() < config.zhiyan_log_query.ingest_grace_seconds
        return web.json_response(
            _log_payload(
                execution_id=execution.execution_id,
                status=execution.status,
                source=SOURCE_ZHIYAN,
                availability=AVAILABILITY_INGEST_PENDING if pending else AVAILABILITY_UNAVAILABLE,
                retry_after_ms=2000 if pending else None,
                error_code=ERROR_ZHIYAN_INGEST_PENDING,
            )
        )
    return web.json_response(
        _log_payload(
            execution_id=execution.execution_id,
            status=execution.status,
            source=SOURCE_ZHIYAN,
            availability=AVAILABILITY_READY,
            lines=normalized,
            partial=partial or start_time > execution.created_at - timedelta(seconds=60),
        )
    )


def register_execution_log_routes(app: web.Application) -> None:
    app.router.add_get('/v1/executions/{execution_id}/manager-logs', get_execution_manager_logs)
    app.router.add_get('/v1/executions/{execution_id}/manager-logs/stream', stream_execution_manager_logs)
    app.router.add_get('/v1/executions/{execution_id}/logs', get_execution_logs)


__all__ = ['register_execution_log_routes']
