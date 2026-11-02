"""Scoped inventory summaries and resource metadata for Console navigation."""

from __future__ import annotations

from aiohttp import web
from sqlalchemy import and_, func, or_, select, tuple_

from nexus.distributed.access.auth import auth_context
from nexus.distributed.gateways.common.keys import AUTHORIZATION_KEY, CONFIG_KEY, SESSION_FACTORY_KEY
from nexus.distributed.gateways.common.platform_metrics import MANAGER_SUMMARY_CACHE_KEY
from nexus.distributed.gateways.common.serde import to_jsonable
from nexus.distributed.gateways.control.routes._shared import query_int
from nexus.distributed.persistence.orm import AssignmentRow, EnvironmentAllocationRow, ExecutionRow, InstanceManagerRow
from nexus.distributed.repositories.environment import EnvironmentInventoryRepository
from nexus.distributed.repositories.pagination import decode_cursor, encode_cursor
from nexus.distributed.utils.clock import utcnow


BUILD_FIELDS = (
    'dependency_build_id',
    'environment_id',
    'owner_principal_id',
    'status',
    'attempt',
    'created_at',
    'started_at',
    'finished_at',
    'updated_at',
    'next_retry_at',
    'error_code',
    'error_message',
)


def build_metadata(build, *, admin=False):
    value = {field: getattr(build, field) for field in BUILD_FIELDS}
    if not admin:
        value.pop('error_message', None)
    value['retry_scheduled_at'] = build.next_retry_at if build.status == 'failed' else None
    value['cleanup_scheduled_at'] = build.next_retry_at if build.status == 'ready' else None
    value['can_retry_automatically'] = value['retry_scheduled_at'] is not None
    return to_jsonable(value)


def manager_scope(request):
    scope = {key: request.query.get(key) or None for key in ('environment_id', 'pool_id', 'workload', 'allocation_id')}
    if any(value is not None and (len(value) > 128 or any(ord(c) < 32 for c in value)) for value in scope.values()):
        raise web.HTTPBadRequest(text='invalid_manager_scope')
    return scope


async def get_manager_summary(request):
    request.app[AUTHORIZATION_KEY].require_admin(auth_context(request))
    scope = manager_scope(request)

    async def load():
        async with request.app[SESSION_FACTORY_KEY]() as session:
            repo = EnvironmentInventoryRepository(
                session, manager_healthy_ttl=request.app[CONFIG_KEY].manager_healthy_ttl
            )
            return await repo.manager_summary(now=utcnow(), **scope)

    try:
        value = await request.app[MANAGER_SUMMARY_CACHE_KEY].get(('manager-summary', *scope.values()), load)
    except Exception:
        return web.json_response(
            {'data_status': 'unavailable', 'scope': scope, 'error_code': 'manager_summary_unavailable'}, status=503
        )
    return web.json_response(to_jsonable(value))


async def list_manager_pools(request):
    request.app[AUTHORIZATION_KEY].require_admin(auth_context(request))
    scope = manager_scope(request)
    scope.pop('pool_id')
    limit = query_int(request, 'limit', 50, min_value=1, max_value=100)
    async with request.app[SESSION_FACTORY_KEY]() as session:
        repo = EnvironmentInventoryRepository(session, manager_healthy_ttl=request.app[CONFIG_KEY].manager_healthy_ttl)
        values = await repo.manager_pools(
            **scope, after=request.query.get('cursor'), prefix=request.query.get('q'), limit=limit + 1
        )
    return web.json_response({
        'items': values[:limit],
        'next_cursor': values[limit - 1] if len(values) > limit else None,
    })


async def get_manager(request):
    request.app[AUTHORIZATION_KEY].require_admin(auth_context(request))
    async with request.app[SESSION_FACTORY_KEY]() as session:
        row = await session.get(InstanceManagerRow, request.match_info['manager_id'])
        if row is None:
            raise web.HTTPNotFound(text='manager_not_found')
        repo = EnvironmentInventoryRepository(session, manager_healthy_ttl=request.app[CONFIG_KEY].manager_healthy_ttl)
        records = await repo._manager_records([row], now=utcnow())
    return web.json_response(to_jsonable(records[0]))


async def get_build(request):
    from nexus.distributed.gateways.control.routes.operations import _authorized_build

    async with request.app[SESSION_FACTORY_KEY]() as session:
        build = await _authorized_build(request, session)
        value = build_metadata(build, admin=auth_context(request).is_admin)
    return web.json_response(value)


async def get_build_usage(request):
    from nexus.distributed.gateways.control.routes.executions import _execution_list_environment_access
    from nexus.distributed.gateways.control.routes.operations import _authorized_build

    limit = query_int(request, 'limit', 50, min_value=1, max_value=100)
    allocation, execution = EnvironmentAllocationRow, ExecutionRow
    statement = (
        select(
            allocation.allocation_id,
            allocation.top_level_execution_id,
            allocation.environment_id,
            allocation.workload,
            allocation.status.label('allocation_status'),
            allocation.created_at,
            execution.spec_type,
            execution.status.label('execution_status'),
        )
        .join(execution, execution.execution_id == allocation.top_level_execution_id)
        .where(allocation.dependency_build_id == request.match_info['build_id'])
    )
    if cursor := request.query.get('cursor'):
        from datetime import datetime

        values = decode_cursor(cursor, fields=frozenset({'created_at', 'allocation_id'}))
        try:
            stamp = datetime.fromisoformat(values['created_at'])
            if stamp.tzinfo is not None:
                raise ValueError
        except (ValueError, TypeError):
            raise web.HTTPBadRequest(text='invalid_cursor') from None
        statement = statement.where(
            or_(
                allocation.created_at < stamp,
                and_(allocation.created_at == stamp, allocation.allocation_id < values['allocation_id']),
            )
        )
    async with request.app[SESSION_FACTORY_KEY]() as session:
        await _authorized_build(request, session)
        environment, _, pairs = await _execution_list_environment_access(request, session, auth_context(request))
        if pairs is not None:
            statement = statement.where(tuple_(execution.owner_principal_id, execution.environment_id).in_(pairs))
        if environment:
            statement = statement.where(execution.environment_id == environment)
        rows = (
            (
                await session.execute(
                    statement.order_by(allocation.created_at.desc(), allocation.allocation_id.desc()).limit(limit + 1)
                )
            )
            .mappings()
            .all()
        )
    items = [dict(row) for row in rows[:limit]]
    cursor = (
        encode_cursor({'created_at': items[-1]['created_at'].isoformat(), 'allocation_id': items[-1]['allocation_id']})
        if len(rows) > limit
        else None
    )
    return web.json_response({'items': to_jsonable(items), 'next_cursor': cursor, 'limit': limit})


async def allocation_activity(session, allocation_id):
    count = await session.scalar(
        select(func.count())
        .select_from(AssignmentRow)
        .where(AssignmentRow.allocation_id == allocation_id, AssignmentRow.status.in_(('pending', 'active')))
    )
    return {'active_assignments': int(count or 0)}
