from __future__ import annotations

from typing import Literal

from aiohttp import web
from pydantic import BaseModel, ConfigDict

from nexus.distributed.access.auth import auth_context
from nexus.distributed.gateways.common.keys import AUTHORIZATION_KEY, TRACE_DATABASE_ACCOUNT_MANAGER_KEY
from nexus.distributed.gateways.common.serde import to_jsonable
from nexus.distributed.gateways.control.routes._shared import read_model
from nexus.distributed.utils.clock import utcnow


class TraceDatabaseCredentialPayload(BaseModel):
    model_config = ConfigDict(extra='forbid')

    backend_name: str
    operation: Literal['create', 'reset']


def _credential_response(account, password: str) -> web.Response:
    return web.json_response(
        {**to_jsonable(account), 'password': password},
        headers={'Cache-Control': 'no-store'},
    )


async def issue_my_trace_database_credential(request: web.Request) -> web.Response:
    context = auth_context(request)
    if context.auth_type == 'tai_app':
        raise web.HTTPForbidden(text='app_identity_cannot_manage_trace_database_credentials')
    payload = await read_model(request, TraceDatabaseCredentialPayload)
    issued = await request.app[TRACE_DATABASE_ACCOUNT_MANAGER_KEY].create_or_reset(
        owner_principal_id=context.principal_id,
        environment_id=request.match_info['environment_id'],
        backend_name=payload.backend_name,
        operation=payload.operation,
        actor_principal_id=context.principal_id,
        now=utcnow(),
    )
    return _credential_response(issued.account, issued.password)


async def list_trace_database_accounts(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    accounts = await request.app[TRACE_DATABASE_ACCOUNT_MANAGER_KEY].list_all()
    return web.json_response({'items': to_jsonable(accounts)})


async def manage_trace_database_account(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    owner_principal_id = request.match_info['principal_id']
    backend_name = request.match_info['backend_name']
    operation = request.match_info['operation']
    manager = request.app[TRACE_DATABASE_ACCOUNT_MANAGER_KEY]
    if operation == 'reset':
        issued = await manager.admin_reset(
            owner_principal_id=owner_principal_id,
            backend_name=backend_name,
            actor_principal_id=context.principal_id,
            now=utcnow(),
        )
        return _credential_response(issued.account, issued.password)
    account = await manager.admin_operation(
        owner_principal_id=owner_principal_id,
        backend_name=backend_name,
        operation=operation,
        actor_principal_id=context.principal_id,
        now=utcnow(),
    )
    return web.json_response(to_jsonable(account))


def register_trace_database_routes(app: web.Application) -> None:
    app.router.add_post(
        '/v1/my-environments/{environment_id}/trace-database-credentials',
        issue_my_trace_database_credential,
    )
    app.router.add_get('/v1/access/trace-database-accounts', list_trace_database_accounts)
    app.router.add_post(
        '/v1/access/trace-database-accounts/{principal_id}/{backend_name}/{operation:disable|restore|reset|reconcile}',
        manage_trace_database_account,
    )


__all__ = ['register_trace_database_routes']
