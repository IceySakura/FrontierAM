from __future__ import annotations

from collections.abc import Sequence

from aiohttp import web
from pydantic import BaseModel, ConfigDict, Field, field_validator

from nexus.distributed.access.auth import auth_context
from nexus.distributed.access.run_trace_backends import RunTraceBackendAccessPolicy, RunTraceBackendView
from nexus.distributed.config import RunTraceBackendMetadata, run_trace_backend_catalog
from nexus.distributed.errors import ResourceNotFoundError
from nexus.distributed.gateways.common.keys import AUTHORIZATION_KEY, CONFIG_KEY, SESSION_FACTORY_KEY
from nexus.distributed.gateways.control.routes._shared import read_model as _read_model
from nexus.distributed.repositories.access import AccessRepository
from nexus.distributed.repositories.audit import AuditRepository
from nexus.distributed.repositories.run_trace_backend import (
    RunTraceBackendGrantInput,
    RunTraceBackendGrantRecord,
    RunTraceBackendGrantRepository,
)
from nexus.distributed.utils.clock import utcnow
from nexus.distributed.utils.identity import validate_run_trace_backend_name


class RunTraceBackendGrantPayload(BaseModel):
    model_config = ConfigDict(extra='forbid')

    backend_name: str = Field(min_length=1, max_length=64)
    enabled: bool = True
    default_order: int | None = Field(default=None, ge=0)

    @field_validator('backend_name')
    @classmethod
    def validate_backend_name(cls, value: str) -> str:
        return validate_run_trace_backend_name(value)


class ReplaceRunTraceBackendGrantsPayload(BaseModel):
    model_config = ConfigDict(extra='forbid')

    grants: list[RunTraceBackendGrantPayload] = Field(max_length=256)


async def list_run_trace_backend_catalog(request: web.Request) -> web.Response:
    request.app[AUTHORIZATION_KEY].require_admin(auth_context(request))
    items = [_catalog_document(backend) for backend in run_trace_backend_catalog(request.app[CONFIG_KEY])]
    return web.json_response({'items': items})


async def list_my_run_trace_backends(request: web.Request) -> web.Response:
    context = auth_context(request)
    async with request.app[SESSION_FACTORY_KEY]() as session:
        access = await RunTraceBackendAccessPolicy(session, config=request.app[CONFIG_KEY]).list_access_for_principal(
            context.principal_id
        )
    return web.json_response({
        'items': [_view_document(item) for item in access.items],
        'claimable_sinks': [_claimable_document(item) for item in access.claimable_sinks],
        'required_sinks': [
            _required_document(backend)
            for backend in run_trace_backend_catalog(request.app[CONFIG_KEY])
            if backend.system_managed and backend.required
        ],
    })


async def self_grant_run_trace_backend(request: web.Request) -> web.Response:
    context = auth_context(request)
    backend_name = request.match_info['backend_name']
    now = utcnow()
    async with request.app[SESSION_FACTORY_KEY]() as session:
        mutation = await RunTraceBackendAccessPolicy(session, config=request.app[CONFIG_KEY]).self_grant(
            context.principal_id,
            backend_name,
            actor=context.principal_id,
            now=now,
        )
        if mutation.changed:
            await AuditRepository(session).append(
                actor_principal_id=context.principal_id,
                target_owner_principal_id=context.principal_id,
                operation='access.run_trace_backends.self_grant',
                resource_type='run_trace_backend_grant',
                resource_id=backend_name,
                metadata={'backend_name': backend_name},
                now=now,
            )
        await session.commit()
    return web.json_response({'item': _grant_document(mutation.grant)})


async def list_principal_run_trace_backend_grants(request: web.Request) -> web.Response:
    request.app[AUTHORIZATION_KEY].require_admin(auth_context(request))
    principal_id = request.match_info['principal_id']
    async with request.app[SESSION_FACTORY_KEY]() as session:
        if await AccessRepository(session).get_user(principal_id) is None:
            raise ResourceNotFoundError('user_not_found')
        items = await RunTraceBackendGrantRepository(session).list_for_principal(principal_id)
    return web.json_response({'principal_id': principal_id, 'items': [_grant_document(item) for item in items]})


async def replace_principal_run_trace_backend_grants(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    principal_id = request.match_info['principal_id']
    payload = await _read_model(request, ReplaceRunTraceBackendGrantsPayload)
    requested = [
        RunTraceBackendGrantInput(
            backend_name=item.backend_name,
            enabled=item.enabled,
            default_order=item.default_order,
        )
        for item in payload.grants
    ]
    now = utcnow()
    async with request.app[SESSION_FACTORY_KEY]() as session:
        replacement = await RunTraceBackendAccessPolicy(session, config=request.app[CONFIG_KEY]).replace_grants(
            principal_id,
            requested,
            actor=context.principal_id,
            now=now,
        )
        added, removed, updated = _grant_changes(replacement.before, replacement.after)
        await AuditRepository(session).append(
            actor_principal_id=context.principal_id,
            target_owner_principal_id=principal_id,
            operation='access.run_trace_backends.replace',
            resource_type='run_trace_backend_grants',
            resource_id=principal_id,
            metadata={'added': added, 'removed': removed, 'updated': updated},
            now=now,
        )
        await session.commit()
    return web.json_response({
        'principal_id': principal_id,
        'items': [_grant_document(item) for item in replacement.after],
    })


def _view_document(item: RunTraceBackendView) -> dict[str, object]:
    return {
        'name': item.name,
        'kind': item.kind,
        'display_name': item.display_name,
        'description': item.description,
        'default_order': item.default_order,
    }


def _catalog_document(item: RunTraceBackendMetadata) -> dict[str, object]:
    return {
        'name': item.name,
        'kind': item.kind,
        'display_name': item.display_name,
        'description': item.description,
        'user_selectable': item.user_selectable,
        'self_grantable': item.self_grantable,
        'system_managed': item.system_managed,
        'required': item.required,
    }


def _claimable_document(item: RunTraceBackendMetadata) -> dict[str, object]:
    return {
        'name': item.name,
        'kind': item.kind,
        'display_name': item.display_name,
        'description': item.description,
    }


def _required_document(item: RunTraceBackendMetadata) -> dict[str, object]:
    return {
        'name': item.name,
        'kind': item.kind,
        'display_name': item.display_name,
        'description': item.description,
        'system_managed': item.system_managed,
        'required': item.required,
    }


def _grant_document(item: RunTraceBackendGrantRecord) -> dict[str, object]:
    return {
        'backend_name': item.backend_name,
        'enabled': item.enabled,
        'default_order': item.default_order,
        'version': item.version,
        'created_by': item.created_by,
        'updated_by': item.updated_by,
        'created_at': item.created_at.isoformat(),
        'updated_at': item.updated_at.isoformat(),
    }


def _grant_changes(
    before: Sequence[RunTraceBackendGrantRecord],
    after: Sequence[RunTraceBackendGrantRecord],
) -> tuple[list[str], list[str], list[str]]:
    previous = {item.backend_name: item for item in before}
    current = {item.backend_name: item for item in after}
    added = sorted(current.keys() - previous.keys())
    removed = sorted(previous.keys() - current.keys())
    updated = sorted(
        name
        for name in current.keys() & previous.keys()
        if (current[name].enabled, current[name].default_order)
        != (previous[name].enabled, previous[name].default_order)
    )
    return added, removed, updated


def register_run_trace_backend_routes(app: web.Application) -> None:
    app.router.add_get('/v1/access/run-trace-backends', list_my_run_trace_backends)
    app.router.add_put('/v1/access/run-trace-backends/{backend_name}', self_grant_run_trace_backend)
    app.router.add_get('/v1/admin/run-trace-backends', list_run_trace_backend_catalog)
    app.router.add_get(
        '/v1/admin/principals/{principal_id}/run-trace-backends',
        list_principal_run_trace_backend_grants,
    )
    app.router.add_put(
        '/v1/admin/principals/{principal_id}/run-trace-backends',
        replace_principal_run_trace_backend_grants,
    )


__all__ = ['register_run_trace_backend_routes']
