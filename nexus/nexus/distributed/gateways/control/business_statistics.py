from __future__ import annotations

from aiohttp import web
from pydantic import ValidationError

from nexus.distributed.access.auth import auth_context
from nexus.distributed.analytics.query import StatisticsLimitError, StatisticsQuery
from nexus.distributed.analytics.reader import BusinessStatisticsReader, StatisticsBusyError
from nexus.distributed.gateways.common.keys import AUTHORIZATION_KEY, SESSION_FACTORY_KEY
from nexus.distributed.gateways.common.reads import read_while_connected


BUSINESS_STATISTICS_KEY = web.AppKey('business_statistics', BusinessStatisticsReader)


def install_business_statistics(app: web.Application, config) -> None:
    reader = BusinessStatisticsReader(config, session_factory=app[SESSION_FACTORY_KEY])
    app[BUSINESS_STATISTICS_KEY] = reader

    async def cleanup(_app):
        await reader.close()

    app.on_cleanup.append(cleanup)
    app.router.add_get('/v1/system/business-statistics', read_business_statistics)


async def read_business_statistics(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    reader = request.app[BUSINESS_STATISTICS_KEY]
    headers = {'Cache-Control': 'no-store'}
    if not reader.config.enabled:
        return web.json_response({'error_code': 'statistics_disabled'}, status=503, headers=headers)
    try:
        if len(request.query) != len(dict(request.query)):
            raise ValueError('duplicate_statistics_parameters')
        query = StatisticsQuery.model_validate(dict(request.query))
    except (ValidationError, ValueError) as exc:
        raise web.HTTPBadRequest(text='invalid_statistics_query') from exc
    try:
        value = await read_while_connected(request, reader.get(query, principal_id=context.principal_id))
    except StatisticsBusyError:
        return web.json_response({'error_code': 'statistics_busy'}, status=429, headers=headers)
    except StatisticsLimitError as exc:
        return web.json_response({'error_code': str(exc)}, status=422, headers=headers)
    except TimeoutError:
        return web.json_response({'error_code': 'statistics_timeout'}, status=504, headers=headers)
    except Exception:
        return web.json_response({'error_code': 'statistics_unavailable'}, status=503, headers=headers)
    return web.json_response(value, headers=headers)
