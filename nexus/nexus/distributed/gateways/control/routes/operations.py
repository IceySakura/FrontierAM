from __future__ import annotations

import json
import logging

from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

from aiohttp import web
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import and_, or_, select, text, tuple_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.distributed.access.auth import auth_context
from nexus.distributed.config.store import (
    ConfigVersionConflictError,
    delete_config_override,
    list_config_overrides,
    list_secret_override_summaries,
    set_config_override,
)
from nexus.distributed.errors import DomainValidationError, ResourceNotFoundError
from nexus.distributed.gateways.common.keys import AUTHORIZATION_KEY, CONFIG_KEY, SESSION_FACTORY_KEY
from nexus.distributed.gateways.common.platform_metrics import MeasuredFileResponse
from nexus.distributed.gateways.common.proxy import internal_request_headers as _internal_headers
from nexus.distributed.gateways.common.proxy import is_websocket_request as _is_websocket_request
from nexus.distributed.gateways.common.proxy import proxy_http as _forward_http_proxy
from nexus.distributed.gateways.common.proxy import proxy_request_headers as _proxy_headers
from nexus.distributed.gateways.common.proxy import proxy_response_headers as _response_headers
from nexus.distributed.gateways.common.proxy import proxy_websocket as _forward_websocket_proxy
from nexus.distributed.gateways.common.serde import to_jsonable
from nexus.distributed.gateways.control.routes._allocations import allocation_payload as _allocation_payload
from nexus.distributed.gateways.control.routes._allocations import dependency_builds_by_id as _dependency_builds_by_id
from nexus.distributed.gateways.control.routes._build_logs import build_logs_response
from nexus.distributed.gateways.control.routes._manager_logs import proxy_manager_log_stream, proxy_manager_log_tail
from nexus.distributed.gateways.control.routes._shared import is_truthy_query as _is_truthy_query
from nexus.distributed.gateways.control.routes._shared import query_int as _query_int
from nexus.distributed.gateways.control.routes._shared import read_model as _read_model
from nexus.distributed.persistence.orm import (
    DependencyBuildRow,
    EnvironmentAllocationRow,
    InstanceManagerRow,
)
from nexus.distributed.repositories.allocation import AllocationRepository
from nexus.distributed.repositories.audit import AuditRepository
from nexus.distributed.repositories.environment import EnvironmentInventoryRepository, EnvironmentRepository
from nexus.distributed.repositories.pagination import decode_cursor, encode_cursor
from nexus.distributed.utils.clock import utcnow


CONSOLE_STATIC_DIR = Path(__file__).parents[3] / 'static' / 'console'
CONSOLE_INDEX = CONSOLE_STATIC_DIR / 'index.html'
CONSOLE_ASSETS_DIR = CONSOLE_STATIC_DIR / 'assets'
logger = logging.getLogger(__name__)


class UpdateManagerPayload(BaseModel):
    draining: bool = Field(strict=True)
    expected_session_id: str | None = Field(default=None, max_length=64)


class ConfigOverridePutPayload(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)

    role: str = Field(min_length=1, max_length=64, strict=True)
    identity: str = Field(min_length=1, max_length=128, strict=True)
    values: dict[str, Any] = Field(strict=True)
    expected_version: int = Field(ge=0, strict=True)
    secret: bool = Field(default=False, strict=True)


class ConfigOverrideDeletePayload(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)

    role: str = Field(min_length=1, max_length=64, strict=True)
    identity: str = Field(min_length=1, max_length=128, strict=True)
    expected_version: int = Field(ge=0, strict=True)
    secret: bool = Field(default=False, strict=True)


async def list_allocations(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    limit = _query_int(request, 'limit', 100, min_value=1, max_value=500)
    legacy_offset = 'offset' in request.query
    offset = _query_int(request, 'offset', 0, min_value=0) if legacy_offset else 0
    cursor = request.query.get('cursor')
    if legacy_offset and cursor is not None:
        raise web.HTTPBadRequest(text='cursor_and_offset_are_mutually_exclusive')
    allocation_type = request.query.get('allocation_type')
    if allocation_type not in {None, 'dynamic', 'predefined'}:
        raise web.HTTPBadRequest(text='invalid_allocation_type')
    filters = {
        'owner_principal_id': request.query.get('owner'),
        'environment_id': request.query.get('environment_id'),
        'status': request.query.get('status'),
        'top_level_execution_id': request.query.get('execution_id'),
        'allocation_type': allocation_type,
    }
    async with request.app[SESSION_FACTORY_KEY]() as session:
        repository = AllocationRepository(session)
        total = await repository.count_allocations(**filters) if legacy_offset else None
        allocations = await repository.list_allocations(
            **filters,
            limit=limit if legacy_offset else limit + 1,
            offset=offset,
            cursor=cursor,
        )
        has_more = not legacy_offset and len(allocations) > limit
        allocations = allocations[:limit]
        dependency_builds = await _dependency_builds_by_id(session, allocations)
    payload = {
        'items': [
            _allocation_payload(
                allocation,
                admin=True,
                dependency_build=(
                    dependency_builds.get(allocation.dependency_build_id)
                    if allocation.dependency_build_id is not None
                    else None
                ),
            )
            for allocation in allocations
        ],
        'limit': limit,
    }
    if legacy_offset:
        payload.update({'total': total, 'offset': offset})
    else:
        payload['next_cursor'] = repository.page_cursor(allocations[-1]) if has_more and allocations else None
    return web.json_response(payload)


_BUILD_LIST_FIELDS = (
    'dependency_build_id',
    'environment_id',
    'status',
    'attempt',
    'created_at',
    'finished_at',
    'started_at',
    'updated_at',
    'next_retry_at',
    'error_code',
    'owner_principal_id',
)


async def _authorized_build(request, session):
    context = auth_context(request)
    authorization = request.app[AUTHORIZATION_KEY]
    authorization.require_scope(context, 'execution:read')
    build = await session.get(DependencyBuildRow, request.match_info['build_id'])
    if build is None:
        raise web.HTTPNotFound(text='build_not_found')
    authorization.require_execution_access(
        context, owner_principal_id=build.owner_principal_id, environment_id=build.environment_id
    )
    if not context.is_admin and context.auth_type != 'disabled':
        await EnvironmentRepository(session).require_grant_scope(
            build.owner_principal_id, build.environment_id, required_scope='execution:read'
        )
    return build


async def list_builds(request):
    from nexus.distributed.gateways.control.routes.executions import _execution_list_environment_access

    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_scope(context, 'execution:read')
    limit = _query_int(request, 'limit', 100, min_value=1, max_value=500)
    statement = select(*(getattr(DependencyBuildRow, field) for field in _BUILD_LIST_FIELDS))
    if status := request.query.get('status'):
        if status not in {'requested', 'building', 'ready', 'failed'}:
            raise web.HTTPBadRequest(text='invalid_build_status')
        statement = statement.where(DependencyBuildRow.status == status)
    if prefix := request.query.get('q'):
        if len(prefix) > 128:
            raise web.HTTPBadRequest(text='invalid_build_filter')
        statement = statement.where(DependencyBuildRow.dependency_build_id.startswith(prefix, autoescape=True))
    if owner := request.query.get('owner'):
        statement = statement.where(DependencyBuildRow.owner_principal_id == owner)
    if execution_id := request.query.get('execution_id'):
        statement = statement.where(
            DependencyBuildRow.dependency_build_id.in_(
                select(EnvironmentAllocationRow.dependency_build_id).where(
                    EnvironmentAllocationRow.top_level_execution_id == execution_id
                )
            )
        )

    if cursor := request.query.get('cursor'):
        values = decode_cursor(cursor, fields=frozenset({'created_at', 'build_id'}))
        try:
            created_at = datetime.fromisoformat(values['created_at'])
        except ValueError as exc:
            raise DomainValidationError('invalid_cursor') from exc
        if created_at.tzinfo is not None:
            raise DomainValidationError('invalid_cursor')
        statement = statement.where(
            or_(
                DependencyBuildRow.created_at < created_at,
                and_(
                    DependencyBuildRow.created_at == created_at,
                    DependencyBuildRow.dependency_build_id < values['build_id'],
                ),
            )
        )
    async with request.app[SESSION_FACTORY_KEY]() as session:
        environment, _, pairs = await _execution_list_environment_access(request, session, context)
        if pairs is not None:
            statement = statement.where(
                tuple_(DependencyBuildRow.owner_principal_id, DependencyBuildRow.environment_id).in_(pairs)
            )
        if environment:
            statement = statement.where(DependencyBuildRow.environment_id == environment)
        builds = list(
            await session.execute(
                statement.order_by(
                    DependencyBuildRow.created_at.desc(), DependencyBuildRow.dependency_build_id.desc()
                ).limit(limit + 1)
            )
        )
    next_cursor = None
    if len(builds) > limit:
        builds = builds[:limit]
        next_cursor = encode_cursor({
            'created_at': builds[-1].created_at.isoformat(),
            'build_id': builds[-1].dependency_build_id,
        })
    return web.json_response({
        'items': [to_jsonable(dict(build._mapping)) for build in builds],
        'next_cursor': next_cursor,
    })


async def get_build_logs(request):
    lines = _query_int(request, 'lines', 200, min_value=1, max_value=1000)
    async with request.app[SESSION_FACTORY_KEY]() as session:
        build = await _authorized_build(request, session)
    return await build_logs_response(request, [build], lines=lines)


async def get_allocation(request: web.Request) -> web.Response:
    from nexus.distributed.gateways.control.routes.console_resources import allocation_activity

    context = auth_context(request)
    authorization = request.app[AUTHORIZATION_KEY]
    authorization.require_scope(context, 'execution:read')
    async with request.app[SESSION_FACTORY_KEY]() as session:
        allocation = await AllocationRepository(session).get_allocation(request.match_info['allocation_id'])
        authorization.require_execution_access(
            context, owner_principal_id=allocation.owner_principal_id, environment_id=allocation.environment_id
        )
        if not context.is_admin and context.auth_type != 'disabled':
            await EnvironmentRepository(session).require_grant_scope(
                allocation.owner_principal_id, allocation.environment_id, required_scope='execution:read'
            )
        dependency_build = (
            await session.get(DependencyBuildRow, allocation.dependency_build_id)
            if allocation.dependency_build_id
            else None
        )
        payload = _allocation_payload(allocation, admin=context.is_admin, dependency_build=dependency_build)
        payload.update(await allocation_activity(session, allocation.allocation_id))
    return web.json_response(payload)


async def force_cleanup_allocation(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    now = utcnow()
    async with request.app[SESSION_FACTORY_KEY]() as session:
        repository = AllocationRepository(session)
        previous = await repository.get_allocation(request.match_info['allocation_id'])
        allocation = await repository.force_cleanup(
            previous.allocation_id,
            now=now,
            expected_version=(
                _query_int(request, 'expected_version', 0, min_value=0) if 'expected_version' in request.query else None
            ),
        )
        dependency_build = (
            await session.get(DependencyBuildRow, allocation.dependency_build_id)
            if allocation.dependency_build_id
            else None
        )
        await AuditRepository(session).append(
            actor_principal_id=context.principal_id,
            target_owner_principal_id=allocation.owner_principal_id,
            operation='allocation.force_cleanup',
            resource_type='environment_allocation',
            resource_id=allocation.allocation_id,
            metadata={
                'environment_id': allocation.environment_id,
                'top_level_execution_id': allocation.top_level_execution_id,
                'from_status': previous.status,
                'to_status': allocation.status,
            },
            now=now,
        )
        await session.commit()
    return web.json_response(_allocation_payload(allocation, admin=True, dependency_build=dependency_build))


async def list_environment_inventory(request: web.Request) -> web.Response:
    request.app[AUTHORIZATION_KEY].require_admin(auth_context(request))
    config = request.app[CONFIG_KEY]
    async with request.app[SESSION_FACTORY_KEY]() as session:
        repo = EnvironmentInventoryRepository(session, manager_healthy_ttl=config.manager_healthy_ttl)
        environments = await repo.list_environments(now=utcnow())
    return web.json_response({'items': to_jsonable(environments)})


async def list_managers(request: web.Request) -> web.Response:
    request.app[AUTHORIZATION_KEY].require_admin(auth_context(request))
    config = request.app[CONFIG_KEY]
    async with request.app[SESSION_FACTORY_KEY]() as session:
        repo = EnvironmentInventoryRepository(session, manager_healthy_ttl=config.manager_healthy_ttl)
        paged = any(
            name in request.query for name in ('limit', 'cursor', 'health', 'draining', 'workload', 'allocation_id')
        )
        if paged:
            if request.query.get('label_key'):
                raise web.HTTPBadRequest(text='label_filter_not_supported_with_cursor')
            limit = _query_int(request, 'limit', 100, min_value=1, max_value=500)
            draining = None
            if 'draining' in request.query:
                draining = _is_truthy_query(request.query.get('draining'))
            managers = await repo.list_managers_page(
                environment_id=request.query.get('environment_id'),
                pool_id=request.query.get('pool_id'),
                workload=request.query.get('workload') or None,
                allocation_id=request.query.get('allocation_id') or None,
                health=request.query.get('health') or 'all',
                draining=draining,
                after_manager_id=request.query.get('cursor'),
                limit=limit + 1,
                now=utcnow(),
            )
            has_more = len(managers) > limit
            managers = managers[:limit]
            return web.json_response({
                'items': to_jsonable(managers),
                'next_cursor': managers[-1].manager_id if has_more and managers else None,
                'limit': limit,
            })
        managers = await repo.list_managers(
            environment_id=request.query.get('environment_id'),
            pool_id=request.query.get('pool_id'),
            label_key=request.query.get('label_key'),
            label_value=request.query.get('label_value'),
            now=utcnow(),
        )
    return web.json_response({'items': to_jsonable(managers)})


async def delete_manager(request: web.Request) -> web.Response:
    request.app[AUTHORIZATION_KEY].require_admin(auth_context(request))
    config = request.app[CONFIG_KEY]
    manager_id = request.match_info['manager_id']
    async with request.app[SESSION_FACTORY_KEY]() as session:
        repo = EnvironmentInventoryRepository(session, manager_healthy_ttl=config.manager_healthy_ttl)
        await repo.delete_stale_manager(manager_id, now=utcnow(), expected_session_id=request.query.get('session_id'))
        await session.commit()
    return web.json_response({'deleted': True, 'manager_id': manager_id})


async def _manager_endpoint(request: web.Request, manager_id: str, *, expected_session_id: str | None = None) -> str:
    request.app[AUTHORIZATION_KEY].require_admin(auth_context(request))
    async with request.app[SESSION_FACTORY_KEY]() as session:
        manager = await session.get(InstanceManagerRow, manager_id)
    if manager is None:
        raise ResourceNotFoundError('manager_not_found')
    expected_session_id = expected_session_id or request.query.get('session_id')
    if expected_session_id is not None and manager.session_id != expected_session_id:
        raise web.HTTPConflict(text='manager_session_changed')
    return manager.endpoint_url.rstrip('/')


async def update_manager(request: web.Request) -> web.Response:
    config = request.app[CONFIG_KEY]
    manager_id = request.match_info['manager_id']
    payload = await _read_model(request, UpdateManagerPayload)
    endpoint_url = await _manager_endpoint(request, manager_id, expected_session_id=payload.expected_session_id)
    url = f'{endpoint_url}/v1/internal/draining'
    async with aiohttp.ClientSession() as session:
        try:
            async with session.patch(
                url, json=payload.model_dump(exclude={'expected_session_id'}), headers=_internal_headers(config)
            ) as upstream:
                return web.Response(
                    body=await upstream.read(),
                    status=upstream.status,
                    headers=_response_headers(upstream.headers),
                )
        except aiohttp.ClientError as exc:
            return web.json_response(
                {'error': 'manager_update_unavailable', 'detail': f'{type(exc).__name__}: {exc}'},
                status=502,
            )


async def get_manager_logs(request: web.Request) -> web.Response:
    manager_id = request.match_info['manager_id']
    lines = _query_int(request, 'lines', 200, min_value=1, max_value=5000)
    endpoint_url = await _manager_endpoint(request, manager_id)
    return await proxy_manager_log_tail(request, endpoint_url=endpoint_url, lines=lines)


async def stream_manager_logs(request: web.Request) -> web.StreamResponse:
    manager_id = request.match_info['manager_id']
    lines = _query_int(request, 'lines', 200, min_value=1, max_value=5000)
    endpoint_url = await _manager_endpoint(request, manager_id)
    return await proxy_manager_log_stream(request, endpoint_url=endpoint_url, lines=lines)


async def proxy_manager_aiomonitor(request: web.Request) -> web.StreamResponse:
    manager_id = request.match_info['manager_id']
    loop_name = request.match_info['loop']
    tail = request.match_info.get('tail') or ''
    endpoint_url = await _manager_endpoint(request, manager_id)
    upstream_path = f'/v1/internal/debug/aiomonitor/{loop_name}'
    if tail:
        upstream_path = f'{upstream_path}/{tail}'
    upstream_url = f'{endpoint_url}{upstream_path}'
    if request.query_string:
        upstream_url = f'{upstream_url}?{request.query_string}'
    debug_base_path = f'/v1/managers/{manager_id}/debug/aiomonitor/{loop_name}'
    if _is_websocket_request(request):
        return await _proxy_manager_websocket(request, upstream_url, debug_base_path=debug_base_path)
    return await _proxy_manager_http(request, upstream_url, debug_base_path=debug_base_path)


async def proxy_manager_py_spy(request: web.Request) -> web.StreamResponse:
    manager_id = request.match_info['manager_id']
    action = request.match_info['action']
    endpoint_url = await _manager_endpoint(request, manager_id)
    upstream_url = f'{endpoint_url}/v1/internal/debug/py-spy/{action}'
    if request.query_string:
        upstream_url = f'{upstream_url}?{request.query_string}'
    debug_base_path = f'/v1/managers/{manager_id}/debug/py-spy/{action}'
    return await _proxy_manager_http(request, upstream_url, debug_base_path=debug_base_path)


async def _proxy_manager_http(
    request: web.Request,
    upstream_url: str,
    *,
    debug_base_path: str,
) -> web.StreamResponse:
    config = request.app[CONFIG_KEY]
    async with aiohttp.ClientSession() as session:
        try:
            return await _forward_http_proxy(
                request,
                upstream_url,
                session=session,
                headers=_proxy_headers(request.headers, config, debug_base_path=debug_base_path),
                max_body_size_bytes=config.gateway_proxy_body_max_size_bytes,
            )
        except aiohttp.ClientError as exc:
            return web.json_response(
                {'error': 'manager_debug_unavailable', 'detail': f'{type(exc).__name__}: {exc}'},
                status=502,
            )


async def _proxy_manager_websocket(
    request: web.Request,
    upstream_url: str,
    *,
    debug_base_path: str,
) -> web.WebSocketResponse:
    config = request.app[CONFIG_KEY]
    async with aiohttp.ClientSession() as session:
        return await _forward_websocket_proxy(
            request,
            upstream_url,
            session=session,
            headers=_proxy_headers(request.headers, config, debug_base_path=debug_base_path),
            prepare_client_before_upstream=True,
            connect_error_close_code=1011,
        )


async def _database_runtime_metrics(session: AsyncSession) -> dict[str, Any]:
    dialect = session.get_bind().dialect.name
    unavailable = {'available': False, 'dialect': dialect}
    if dialect not in {'mysql', 'mariadb'}:
        return unavailable
    try:
        max_connection_rows = (await session.execute(text("SHOW VARIABLES LIKE 'max_connections'"))).all()
        status_rows = (
            await session.execute(
                text(
                    'SHOW GLOBAL STATUS WHERE Variable_name IN '
                    "('Threads_connected', 'Threads_running', 'Max_used_connections', "
                    "'Slow_queries', 'Aborted_connects')"
                )
            )
        ).all()
        if len(max_connection_rows) != 1:
            return unavailable
        status = {str(name): int(value) for name, value in status_rows}
        required = {
            'Threads_connected',
            'Threads_running',
            'Max_used_connections',
            'Slow_queries',
            'Aborted_connects',
        }
        if not required.issubset(status):
            return unavailable
        return {
            'available': True,
            'dialect': dialect,
            'max_connections': int(max_connection_rows[0][1]),
            'threads_connected': status['Threads_connected'],
            'threads_running': status['Threads_running'],
            'max_used_connections': status['Max_used_connections'],
            'slow_queries': status['Slow_queries'],
            'aborted_connects': status['Aborted_connects'],
        }
    except (SQLAlchemyError, TypeError, ValueError, IndexError):
        logger.warning('database runtime metrics are unavailable', exc_info=True)
        return unavailable


async def get_system_summary(request: web.Request) -> web.Response:
    from nexus.distributed.gateways.control.status import read_live_status

    return web.json_response(to_jsonable(await read_live_status(request)))


async def get_system_status(request: web.Request) -> web.Response:
    from nexus.distributed.gateways.control.status import read_snapshot_status

    return web.json_response(to_jsonable(await read_snapshot_status(request)))


def _config_override_payload(record, *, include_values: bool) -> dict[str, Any]:
    payload = {
        'role': record.role,
        'identity': record.identity,
        'version': record.version,
        'fields': list(record.fields),
    }
    if include_values:
        payload['values'] = record.values
    return payload


async def get_system_config_overrides(request: web.Request) -> web.Response:
    request.app[AUTHORIZATION_KEY].require_admin(auth_context(request))
    async with request.app[SESSION_FACTORY_KEY]() as session:
        config_overrides = await list_config_overrides(session)
        secret_overrides = await list_secret_override_summaries(session)
    return web.json_response(
        {
            'config_overrides': [_config_override_payload(record, include_values=True) for record in config_overrides],
            'secret_overrides': [_config_override_payload(record, include_values=False) for record in secret_overrides],
        },
        headers={'Cache-Control': 'no-store'},
    )


async def put_system_config_override(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    payload = await _read_model(request, ConfigOverridePutPayload)
    async with request.app[SESSION_FACTORY_KEY]() as session:
        try:
            record = await set_config_override(
                session,
                role=payload.role,
                identity=payload.identity,
                values=payload.values,
                expected_version=payload.expected_version,
                actor=context.principal_id,
                secret=payload.secret,
            )
        except ConfigVersionConflictError as exc:
            raise web.HTTPConflict(text='config_version_conflict') from exc
        except (TypeError, ValueError) as exc:
            raise web.HTTPUnprocessableEntity(text='validation_error') from exc
        await session.commit()
    return web.json_response(
        _config_override_payload(record, include_values=not payload.secret),
        headers={'Cache-Control': 'no-store'},
    )


async def delete_system_config_override(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    payload = await _read_model(request, ConfigOverrideDeletePayload)
    async with request.app[SESSION_FACTORY_KEY]() as session:
        try:
            deleted = await delete_config_override(
                session,
                role=payload.role,
                identity=payload.identity,
                expected_version=payload.expected_version,
                actor=context.principal_id,
                secret=payload.secret,
            )
        except ConfigVersionConflictError as exc:
            raise web.HTTPConflict(text='config_version_conflict') from exc
        except (TypeError, ValueError) as exc:
            raise web.HTTPUnprocessableEntity(text='validation_error') from exc
        await session.commit()
    return web.json_response(
        {'deleted': deleted, 'role': payload.role, 'identity': payload.identity},
        headers={'Cache-Control': 'no-store'},
    )


async def console_ui(request: web.Request) -> web.Response:
    html = CONSOLE_INDEX.read_text(encoding='utf-8')
    config = request.app[CONFIG_KEY]
    application_base_url = config.console_application_base_url or config.application_public_base_url
    if config.auth_mode == 'disabled':
        application_base_url = request.query.get('applicationBaseUrl') or application_base_url
    console_config = json.dumps({
        'applicationBaseUrl': application_base_url,
        'welcomeAnimationEnabled': config.cloud_console_welcome_animation_enabled,
        'guideUrl': config.cloud_console_guide_url,
    }).replace('&', '\\u0026')
    console_config = console_config.replace('<', '\\u003c').replace('>', '\\u003e')
    config_script = f'<script>\nwindow.__NEXUS_DISTRIBUTED_CONSOLE__ = {console_config};\n</script>'
    html = html.replace('</head>', f'{config_script}\n</head>', 1)
    return web.Response(text=html, content_type='text/html', headers={'Cache-Control': 'no-store'})


async def console_asset(request: web.Request) -> web.FileResponse:
    asset_path = (CONSOLE_ASSETS_DIR / request.match_info['path']).resolve()
    assets_dir = CONSOLE_ASSETS_DIR.resolve()
    try:
        asset_path.relative_to(assets_dir)
    except ValueError as exc:
        raise web.HTTPNotFound() from exc
    if not asset_path.is_file():
        raise web.HTTPNotFound()
    return MeasuredFileResponse(asset_path, headers={'Cache-Control': 'public, max-age=31536000, immutable'})


async def health(_request: web.Request) -> web.Response:
    return web.json_response({'status': 'ok'})


def register_operation_routes(app: web.Application) -> None:
    from nexus.distributed.gateways.control.routes.console_resources import (
        get_build,
        get_build_usage,
        get_manager,
        get_manager_summary,
        list_manager_pools,
    )
    from nexus.distributed.gateways.control.status import read_metric_history

    app.router.add_get('/v1/builds', list_builds)
    app.router.add_get('/v1/builds/{build_id}', get_build)
    app.router.add_get('/v1/builds/{build_id}/usage', get_build_usage)
    app.router.add_get('/v1/builds/{build_id}/logs', get_build_logs)
    app.router.add_get('/v1/health', health)
    app.router.add_get('/v1/system/summary', get_system_summary)
    app.router.add_get('/v1/system/status', get_system_status)
    app.router.add_get('/v1/system/metrics', read_metric_history)
    app.router.add_get('/v1/system/config-overrides', get_system_config_overrides)
    app.router.add_put('/v1/system/config-overrides', put_system_config_override)
    app.router.add_delete('/v1/system/config-overrides', delete_system_config_override)
    app.router.add_get('/v1/allocations', list_allocations)
    app.router.add_get('/v1/allocations/{allocation_id}', get_allocation)
    app.router.add_post('/v1/allocations/{allocation_id}/force-cleanup', force_cleanup_allocation)
    app.router.add_get('/v1/console/assets/{path:.*}', console_asset)
    app.router.add_get('/v1/console', console_ui)
    app.router.add_get('/v1/console/{path:.*}', console_ui)
    app.router.add_get('/v1/environment-inventory', list_environment_inventory)
    app.router.add_get('/v1/managers', list_managers)
    app.router.add_get('/v1/managers/summary', get_manager_summary)
    app.router.add_get('/v1/manager-pools', list_manager_pools)
    app.router.add_get('/v1/managers/{manager_id}', get_manager)
    app.router.add_get('/v1/managers/{manager_id}/logs', get_manager_logs)
    app.router.add_get('/v1/managers/{manager_id}/logs/stream', stream_manager_logs)
    app.router.add_route(
        '*',
        '/v1/managers/{manager_id}/debug/aiomonitor/{loop}',
        proxy_manager_aiomonitor,
    )
    app.router.add_route(
        '*',
        '/v1/managers/{manager_id}/debug/aiomonitor/{loop}/{tail:.*}',
        proxy_manager_aiomonitor,
    )
    app.router.add_get('/v1/managers/{manager_id}/debug/py-spy/{action}', proxy_manager_py_spy)
    app.router.add_patch('/v1/managers/{manager_id}', update_manager)
    app.router.add_delete('/v1/managers/{manager_id}', delete_manager)


__all__ = ['register_operation_routes']
