from __future__ import annotations

from aiohttp import web

from nexus.distributed.access.auth import auth_context
from nexus.distributed.gateways.common.keys import AUTHORIZATION_KEY, CONFIG_KEY, SESSION_FACTORY_KEY
from nexus.distributed.gateways.common.platform_metrics import LIVE_CACHE_KEY, METRICS_KEY
from nexus.distributed.gateways.common.reads import read_while_connected
from nexus.distributed.observability.history import HISTORY_CATALOG, HISTORY_PANELS, MetricHistoryReader
from nexus.distributed.repositories.observability import ObservabilityRepository
from nexus.distributed.utils.clock import utcnow


HISTORY_KEY = web.AppKey('metric_history', MetricHistoryReader)


def install_metric_history(app: web.Application, config) -> None:
    app[HISTORY_KEY] = MetricHistoryReader(config)

    async def cleanup(_app):
        await _app[HISTORY_KEY].close()

    app.on_cleanup.append(cleanup)


async def read_metric_history(request: web.Request):
    request.app[AUTHORIZATION_KEY].require_admin(auth_context(request))
    if request.query.get('catalog') == '1' and len(request.query) == 1:
        return web.json_response({
            'items': list(HISTORY_CATALOG.values()),
            'sources': {'trace_db': request.app[HISTORY_KEY].trace_db_enabled},
        })
    if set(request.query) - {
        'panel',
        'hours',
        'environment_id',
        'component',
        'stage',
        'outcome',
        'category',
        'source',
        'operation',
        'route',
        'view',
    }:
        raise web.HTTPBadRequest(text='invalid_history_query')
    panel = request.query.get('panel', 'gateway')
    hours = request.query.get('hours', '1')
    if panel not in HISTORY_PANELS or hours not in {'1', '24', '168', '720'}:
        raise web.HTTPBadRequest(text='invalid_history_query')
    environment, _ = status_scope(request)
    filters = {
        key: value
        for key, value in {
            'environment': environment,
            'service': request.query.get('component'),
            'stage': request.query.get('stage'),
            'outcome': request.query.get('outcome'),
            'category': request.query.get('category'),
            'source': request.query.get('source'),
            'operation': request.query.get('operation'),
            'route': request.query.get('route'),
        }.items()
        if value
    }
    try:
        value = await read_while_connected(
            request,
            request.app[HISTORY_KEY].get(
                panel=panel, hours=int(hours), filters=filters, view=request.query.get('view')
            ),
        )
    except ValueError as exc:
        raise web.HTTPBadRequest(text='invalid_history_filter') from exc
    except Exception:
        return web.json_response(
            {'source': 'zhiyan', 'data_status': 'unavailable', 'error_code': 'history_query_failed'}, status=503
        )
    status = (
        503 if value.get('data_status') == 'unavailable' and value.get('error_code') == 'history_query_failed' else 200
    )
    return web.json_response(value, status=status)


def status_scope(request: web.Request, *, environment_id: str | None = None) -> tuple[str | None, str | None]:
    if request.query.get('label_key') or request.query.get('label_value'):
        raise web.HTTPBadRequest(text='labels_require_inventory_query')
    environment_id = environment_id or request.query.get('environment_id') or None
    pool_id = request.query.get('pool_id') or None
    if any(
        value is not None and (len(value) > 64 or any(ord(char) < 32 for char in value))
        for value in (environment_id, pool_id)
    ):
        raise web.HTTPBadRequest(text='invalid_status_scope')
    return environment_id, pool_id


async def read_live_status(request: web.Request, *, environment_id: str | None = None):
    request.app[AUTHORIZATION_KEY].require_admin(auth_context(request))
    environment_id, pool_id = status_scope(request, environment_id=environment_id)
    config = request.app[CONFIG_KEY]

    async def load():
        async with request.app[SESSION_FACTORY_KEY]() as session:
            return await ObservabilityRepository(session, config).live_inventory(
                now=utcnow(), environment_id=environment_id, pool_id=pool_id
            )

    try:
        value = await request.app[LIVE_CACHE_KEY].get(('admin', environment_id, pool_id), load)
    except Exception as exc:
        raise web.HTTPServiceUnavailable(text='live_status_unavailable') from exc
    value['metrics'] = request.app[METRICS_KEY].state()
    if environment_id and value['resource_usage']['complete']:
        request['platform_metric_environment'] = environment_id
    return value


async def read_snapshot_status(request: web.Request):
    request.app[AUTHORIZATION_KEY].require_admin(auth_context(request))
    environment_id, pool_id = status_scope(request)
    if pool_id is not None:
        raise web.HTTPBadRequest(text='snapshot_scope_is_environment')

    async def load():
        async with request.app[SESSION_FACTORY_KEY]() as session:
            return await ObservabilityRepository(session, request.app[CONFIG_KEY]).snapshots(
                now=utcnow(), environment_id=environment_id
            )

    try:
        return await request.app[LIVE_CACHE_KEY].get(('snapshot', environment_id), load)
    except Exception as exc:
        raise web.HTTPServiceUnavailable(text='status_snapshot_unavailable') from exc
