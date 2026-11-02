"""Application routes for stateless Single trace discovery and event pages."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid

from typing import Any

from aiohttp import web
from sqlalchemy import select

from nexus.distributed.access.auth import auth_context
from nexus.distributed.errors import DistributedDomainError
from nexus.distributed.gateways.application.routes.executions import _authorize_execution
from nexus.distributed.gateways.common.errors import gateway_error_response
from nexus.distributed.gateways.common.keys import CONFIG_KEY, SESSION_FACTORY_KEY
from nexus.distributed.persistence.orm import AssignmentRow
from nexus.distributed.trace.models import (
    AssignmentTraceMetadata,
    AuthorizedTraceTarget,
    TraceReadDiagnostics,
    TraceReadError,
    json_bytes,
)
from nexus.distributed.trace.service import TraceReadService


TRACE_READ_SERVICE_KEY = web.AppKey('trace_read_service', TraceReadService)
logger = logging.getLogger(__name__)
_BACKEND = re.compile(r'^[A-Za-z0-9_-]{1,64}$')


def _parameters(request: web.Request, *, events: bool) -> dict[str, Any]:
    allowed = {'limit', 'cursor', 'mode'} if events else {'limit', 'cursor', 'backend'}
    if set(request.query) - allowed or any(len(request.query.getall(key)) != 1 for key in request.query):
        raise TraceReadError('invalid_request', 400)
    raw_limit = request.query.get('limit', '200' if events else '100')
    if not raw_limit.isascii() or not raw_limit.isdigit() or len(raw_limit) > 3:
        raise TraceReadError('invalid_request', 400)
    limit = int(raw_limit)
    if not 1 <= limit <= (500 if events else 200):
        raise TraceReadError('invalid_request', 400)
    result: dict[str, Any] = {'limit': limit, 'cursor': request.query.get('cursor')}
    if events:
        mode = request.query.get('mode', 'raw')
        if mode not in ('raw', 'preview'):
            raise TraceReadError('invalid_request', 400)
        result['mode'] = mode
    if result['cursor'] is not None and not 1 <= len(result['cursor']) <= 4096:
        raise TraceReadError('invalid_trace_cursor', 400)
    if not events:
        backend = request.query.get('backend')
        if backend is not None and _BACKEND.fullmatch(backend) is None:
            raise TraceReadError('invalid_request', 400)
        result['backend'] = backend
    return result


async def _read(request: web.Request, *, events: bool, diagnostics: TraceReadDiagnostics) -> dict[str, Any]:
    service = request.app.get(TRACE_READ_SERVICE_KEY)
    if service is None:
        raise TraceReadError('resource_not_found', 404)
    execution_id = request.match_info['execution_id']
    if not 1 <= len(execution_id) <= 64:
        raise TraceReadError('invalid_request', 400)
    async with request.app[SESSION_FACTORY_KEY]() as session:
        execution = await _authorize_execution(request, session, scope='execution:read')
        if execution.spec_type != 'nexus_single':
            raise TraceReadError('execution_type_not_supported', 400)
        if not execution.owner_principal_id or not execution.environment_id:
            raise TraceReadError('trace_source_unresolved', 409)
        target = AuthorizedTraceTarget(
            principal_id=auth_context(request).principal_id,
            execution_id=execution.execution_id,
            owner_principal_id=execution.owner_principal_id,
            environment_id=execution.environment_id,
            execution_status=execution.status,
            current_assignment_id=execution.current_assignment_id,
            driver_config=execution.driver_config_json,
        )
    diagnostics.execution_id = target.execution_id
    diagnostics.principal_id = target.principal_id
    diagnostics.owner_principal_id = target.owner_principal_id
    params = _parameters(request, events=events)

    async def load_metadata() -> list[AssignmentTraceMetadata]:
        result: list[AssignmentTraceMetadata] = []
        after = None
        async with request.app[SESSION_FACTORY_KEY]() as session:
            while True:
                count = min(service.config.assignment_page_size, service.config.max_assignments - len(result) + 1)
                statement = select(
                    AssignmentRow.assignment_id,
                    AssignmentRow.attempt_id,
                    AssignmentRow.attempt_seq,
                    AssignmentRow.status,
                    AssignmentRow.created_at,
                    AssignmentRow.runtime_info_json,
                ).where(AssignmentRow.execution_id == target.execution_id)
                if after is not None:
                    statement = statement.where(AssignmentRow.assignment_id > after)
                rows = (await session.execute(statement.order_by(AssignmentRow.assignment_id).limit(count))).all()
                if len(result) + len(rows) > service.config.max_assignments:
                    raise TraceReadError('trace_metadata_scan_incomplete', 503, retryable=True)
                result.extend(AssignmentTraceMetadata(*row) for row in rows)
                if len(rows) < count:
                    return result
                after = rows[-1].assignment_id

    if events:
        return await service.read_events(
            target, request.match_info['stream_id'], load_metadata=load_metadata, diagnostics=diagnostics, **params
        )
    return await service.list_streams(target, load_metadata=load_metadata, diagnostics=diagnostics, **params)


async def _handle(request: web.Request, *, events: bool) -> web.Response:
    started = time.monotonic()
    request_id = uuid.uuid4().hex
    endpoint = 'events' if events else 'streams'
    records = 0
    diagnostics = TraceReadDiagnostics()
    error_code = None
    try:
        try:
            value = await asyncio.wait_for(
                _read(request, events=events, diagnostics=diagnostics),
                timeout=request.app[CONFIG_KEY].trace_read.request_timeout_seconds - 3,
            )
        except asyncio.TimeoutError:
            raise TraceReadError('trace_source_unavailable', 503, retryable=True) from None
        encode_started = time.monotonic()
        body = json_bytes(value)
        diagnostics.encoding_seconds += time.monotonic() - encode_started
        records = len(value['items'])
        response = web.Response(body=body, content_type='application/json')
    except DistributedDomainError as error:
        error_code = error.code
        response = gateway_error_response(error)
        if error.retryable:
            response.headers['Retry-After'] = '1'
    except web.HTTPException as error:
        error.headers['Cache-Control'] = 'no-store'
        raise
    except Exception:
        error_code = 'trace_source_unavailable'
        # Source and metadata exceptions must never reach the generic traceback
        # logger, which may include SQL parameters or credential-bearing errors.
        response = gateway_error_response(TraceReadError('trace_source_unavailable', 503, retryable=True))
        response.headers['Retry-After'] = '1'
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Request-ID'] = request_id
    logger.info(
        'trace_read endpoint=%s backend=%s execution=%s principal=%s owner=%s status=%s error=%s '
        'records=%s bytes=%s queries=%s query_ms=%.2f encoding_ms=%.2f elapsed_ms=%.2f limited=%s errors=%s '
        'mode=%s omitted_records=%s omitted_bytes=%s request_id=%s',
        endpoint,
        diagnostics.backend,
        diagnostics.execution_id,
        diagnostics.principal_id,
        diagnostics.owner_principal_id,
        response.status,
        error_code,
        records,
        response.content_length or 0,
        diagnostics.queries,
        diagnostics.query_seconds * 1000,
        diagnostics.encoding_seconds * 1000,
        (time.monotonic() - started) * 1000,
        int(response.status == 429),
        int(response.status >= 400),
        diagnostics.mode,
        diagnostics.omitted_records,
        diagnostics.omitted_bytes,
        request_id,
    )
    return response


async def list_trace_streams(request: web.Request) -> web.Response:
    return await _handle(request, events=False)


async def read_trace_events(request: web.Request) -> web.Response:
    return await _handle(request, events=True)


@web.middleware
async def trace_response_headers(request: web.Request, handler: Any) -> web.StreamResponse:
    response = await handler(request)
    if request.match_info.route.name in {'single-trace-streams', 'single-trace-events'}:
        response.headers['Cache-Control'] = 'no-store'
        if 'X-Request-ID' not in response.headers:
            request_id = uuid.uuid4().hex
            response.headers['X-Request-ID'] = request_id
            logger.info(
                'trace_read endpoint=%s status=%s errors=%s request_id=%s',
                request.match_info.route.name,
                response.status,
                int(response.status >= 400),
                request_id,
            )
    return response


def register_trace_event_routes(app: web.Application) -> None:
    app.router.add_get('/v1/executions/{execution_id}/trace-streams', list_trace_streams, name='single-trace-streams')
    app.router.add_get(
        '/v1/executions/{execution_id}/trace-streams/{stream_id}/events', read_trace_events, name='single-trace-events'
    )
