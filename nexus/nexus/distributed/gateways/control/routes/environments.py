from __future__ import annotations

from typing import Literal, cast

from aiohttp import web
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nexus.distributed.access.auth import auth_context
from nexus.distributed.environment.limits import (
    EnvironmentDefaultQuota,
    EnvironmentDefaultTimeouts,
    EnvironmentResourceLimits,
    validate_timeout_defaults,
)
from nexus.distributed.environment.provider_config import require_provider_type_enabled
from nexus.distributed.gateways.common.keys import (
    AUTHORIZATION_KEY,
    CONFIG_KEY,
    SESSION_FACTORY_KEY,
    TRACE_DATABASE_ACCOUNT_MANAGER_KEY,
)
from nexus.distributed.gateways.common.serde import to_jsonable
from nexus.distributed.gateways.control.routes._shared import is_truthy_query as _is_truthy_query
from nexus.distributed.gateways.control.routes._shared import read_model as _read_model
from nexus.distributed.gateways.control.routes.access import (
    list_environment_grants,
    upsert_environment_grant,
)
from nexus.distributed.repositories.audit import AuditRepository
from nexus.distributed.repositories.environment import (
    ENVIRONMENT_FIELD_UNSET,
    EnvironmentInventoryRepository,
    EnvironmentRepository,
)
from nexus.distributed.repositories.provider_config import ProviderConfigRepository
from nexus.distributed.utils.clock import utcnow
from nexus.distributed.utils.identity import ENVIRONMENT_ID_MAX_LENGTH, validate_pool_id


class CreateEnvironmentPayload(BaseModel):
    model_config = ConfigDict(extra='forbid')

    environment_id: str = Field(min_length=1, max_length=ENVIRONMENT_ID_MAX_LENGTH)
    name: str = Field(min_length=1, max_length=128)
    allocation_type: Literal['predefined', 'dynamic']
    provider_type: Literal['predefined', 'kubernetes', 'tkex']
    provider_config_ref: str | None = Field(default=None, max_length=512)
    resource_limits: EnvironmentResourceLimits = Field(default_factory=EnvironmentResourceLimits)
    default_timeouts: EnvironmentDefaultTimeouts = Field(default_factory=EnvironmentDefaultTimeouts)
    default_quota: EnvironmentDefaultQuota | None = None
    status: Literal['enabled', 'disabled'] = 'disabled'

    @model_validator(mode='after')
    def validate_limit_defaults(self) -> CreateEnvironmentPayload:
        validate_timeout_defaults(
            self.resource_limits.model_dump(exclude_none=True),
            self.default_timeouts.model_dump(exclude_none=True),
        )
        return self


class UpdateEnvironmentPayload(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str | None = Field(default=None, min_length=1, max_length=128)
    allocation_type: Literal['predefined', 'dynamic'] | None = None
    provider_type: Literal['predefined', 'kubernetes', 'tkex'] | None = None
    provider_config_ref: str | None = Field(default=None, max_length=512)
    resource_limits: EnvironmentResourceLimits | None = None
    default_timeouts: EnvironmentDefaultTimeouts | None = None
    default_quota: EnvironmentDefaultQuota | None = None
    status: Literal['enabled', 'disabled'] | None = None

    @model_validator(mode='after')
    def reject_null_limit_objects(self) -> UpdateEnvironmentPayload:
        for field_name in ('resource_limits', 'default_timeouts'):
            if field_name in self.model_fields_set and getattr(self, field_name) is None:
                raise ValueError(f'{field_name}_must_be_an_object')
        return self


async def create_environment(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    payload = await _read_model(request, CreateEnvironmentPayload)
    now = utcnow()
    require_provider_type_enabled(payload.provider_type, request.app[CONFIG_KEY].environment_enabled_providers)
    async with request.app[SESSION_FACTORY_KEY]() as session:
        await ProviderConfigRepository(session).validate_environment(
            allocation_type=payload.allocation_type,
            provider_type=payload.provider_type,
            provider_config_ref=payload.provider_config_ref,
            workload_namespace=request.app[CONFIG_KEY].environment_namespace,
        )
        environment = await EnvironmentRepository(session).create_environment(
            environment_id=payload.environment_id,
            name=payload.name,
            allocation_type=payload.allocation_type,
            provider_type=payload.provider_type,
            provider_config_ref=payload.provider_config_ref,
            resource_limits=payload.resource_limits.model_dump(exclude_none=True),
            default_timeouts=payload.default_timeouts.model_dump(exclude_none=True),
            default_quota=(payload.default_quota.model_dump() if payload.default_quota is not None else None),
            status=payload.status,
            actor=context.principal_id,
            now=now,
        )
        await AuditRepository(session).append(
            actor_principal_id=context.principal_id,
            operation='environment.create',
            resource_type='environment',
            resource_id=environment.environment_id,
            metadata={
                'allocation_type': environment.allocation_type,
                'provider_type': environment.provider_type,
                'status': environment.status,
                'config_version': environment.config_version,
            },
            now=now,
        )
        await session.commit()
    return web.json_response(to_jsonable(environment), status=201)


async def list_environment_definitions(request: web.Request) -> web.Response:
    request.app[AUTHORIZATION_KEY].require_admin(auth_context(request))
    include_disabled = not _is_truthy_query(request.query.get('enabled_only'))
    async with request.app[SESSION_FACTORY_KEY]() as session:
        environments = await EnvironmentRepository(session).list_environment_definitions(
            include_disabled=include_disabled
        )
    return web.json_response({'items': to_jsonable(environments)})


async def get_environment_definition(request: web.Request) -> web.Response:
    context = auth_context(request)
    environment_id = request.match_info['environment_id']
    async with request.app[SESSION_FACTORY_KEY]() as session:
        repository = EnvironmentRepository(session)
        environment = await repository.get_environment(environment_id)
        grant = None
        if not context.is_admin:
            try:
                grant = await repository.get_grant(context.principal_id, environment_id)
            except KeyError as exc:
                raise web.HTTPNotFound(text='resource_not_found') from exc
    payload = to_jsonable(environment)
    if grant is not None:
        payload['grant'] = to_jsonable(grant)
    pool_id = request.query.get('pool_id')
    if pool_id is not None:
        validate_pool_id(pool_id)
        async with request.app[SESSION_FACTORY_KEY]() as session:
            stats = await EnvironmentInventoryRepository(
                session,
                manager_healthy_ttl=request.app[CONFIG_KEY].manager_healthy_ttl,
            ).get_environment_stats(
                environment_id,
                pool_id=pool_id,
                now=utcnow(),
            )
        payload['pool_stats'] = to_jsonable(stats)
    return web.json_response(payload)


async def update_environment(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    payload = await _read_model(request, UpdateEnvironmentPayload)
    if not payload.model_fields_set:
        raise web.HTTPBadRequest(text='environment_update_required')
    now = utcnow()
    async with request.app[SESSION_FACTORY_KEY]() as session:
        repository = EnvironmentRepository(session)
        current = await repository.get_environment(request.match_info['environment_id'])
        fields = payload.model_fields_set
        next_provider_type = cast(
            str,
            payload.provider_type if 'provider_type' in fields else current.provider_type,
        )
        require_provider_type_enabled(next_provider_type, request.app[CONFIG_KEY].environment_enabled_providers)
        await ProviderConfigRepository(session).validate_environment(
            allocation_type=cast(
                str,
                payload.allocation_type if 'allocation_type' in fields else current.allocation_type,
            ),
            provider_type=next_provider_type,
            provider_config_ref=(
                payload.provider_config_ref if 'provider_config_ref' in fields else current.provider_config_ref
            ),
            workload_namespace=request.app[CONFIG_KEY].environment_namespace,
        )
        if payload.status == 'disabled' and current.status != 'disabled':
            await request.app[TRACE_DATABASE_ACCOUNT_MANAGER_KEY].disable_for_environment_revocation(
                environment_id=current.environment_id,
                actor_principal_id=context.principal_id,
                now=now,
            )
        environment = await repository.update_environment(
            request.match_info['environment_id'],
            name=payload.name,
            allocation_type=payload.allocation_type,
            provider_type=payload.provider_type,
            provider_config_ref=(
                payload.provider_config_ref
                if 'provider_config_ref' in payload.model_fields_set
                else ENVIRONMENT_FIELD_UNSET
            ),
            resource_limits=(
                payload.resource_limits.model_dump(exclude_none=True) if payload.resource_limits is not None else None
            ),
            default_timeouts=(
                payload.default_timeouts.model_dump(exclude_none=True) if payload.default_timeouts is not None else None
            ),
            default_quota=(
                payload.default_quota.model_dump()
                if payload.default_quota is not None
                else None
                if 'default_quota' in payload.model_fields_set
                else ENVIRONMENT_FIELD_UNSET
            ),
            status=payload.status,
            actor=context.principal_id,
            now=now,
        )
        await AuditRepository(session).append(
            actor_principal_id=context.principal_id,
            operation='environment.update',
            resource_type='environment',
            resource_id=environment.environment_id,
            metadata={'config_version': environment.config_version, 'fields': sorted(payload.model_fields_set)},
            now=now,
        )
        await session.commit()
    return web.json_response(to_jsonable(environment))


async def disable_environment(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    now = utcnow()
    async with request.app[SESSION_FACTORY_KEY]() as session:
        repository = EnvironmentRepository(session)
        current = await repository.get_environment(request.match_info['environment_id'])
        if current.status != 'disabled':
            await request.app[TRACE_DATABASE_ACCOUNT_MANAGER_KEY].disable_for_environment_revocation(
                environment_id=current.environment_id,
                actor_principal_id=context.principal_id,
                now=now,
            )
        environment = await repository.set_environment_status(
            request.match_info['environment_id'],
            status='disabled',
            actor=context.principal_id,
            now=now,
        )
        await AuditRepository(session).append(
            actor_principal_id=context.principal_id,
            operation='environment.disable',
            resource_type='environment',
            resource_id=environment.environment_id,
            metadata={'config_version': environment.config_version},
            now=now,
        )
        await session.commit()
    return web.json_response(to_jsonable(environment))


async def get_environment_stats(request: web.Request) -> web.Response:
    from nexus.distributed.gateways.control.status import read_live_status

    value = await read_live_status(request, environment_id=request.match_info['environment_id'])
    return web.json_response(to_jsonable(value))


async def list_environment_managers(request: web.Request) -> web.Response:
    request.app[AUTHORIZATION_KEY].require_admin(auth_context(request))
    config = request.app[CONFIG_KEY]
    async with request.app[SESSION_FACTORY_KEY]() as session:
        repo = EnvironmentInventoryRepository(session, manager_healthy_ttl=config.manager_healthy_ttl)
        managers = await repo.list_environment_managers(
            request.match_info['environment_id'],
            pool_id=request.query.get('pool_id'),
            label_key=request.query.get('label_key'),
            label_value=request.query.get('label_value'),
            now=utcnow(),
        )
    return web.json_response({'items': to_jsonable(managers)})


def register_environment_routes(app: web.Application) -> None:
    app.router.add_get('/v1/environments', list_environment_definitions)
    app.router.add_post('/v1/environments', create_environment)
    app.router.add_get('/v1/environments/{environment_id}', get_environment_definition)
    app.router.add_patch('/v1/environments/{environment_id}', update_environment)
    app.router.add_put('/v1/environments/{environment_id}', update_environment)
    app.router.add_delete('/v1/environments/{environment_id}', disable_environment)
    app.router.add_post('/v1/environments/{environment_id}/disable', disable_environment)
    app.router.add_get('/v1/environments/{environment_id}/grants', list_environment_grants)
    app.router.add_put(
        '/v1/environments/{environment_id}/grants/{principal_id}',
        upsert_environment_grant,
    )
    app.router.add_get('/v1/environments/{environment_id}/stats', get_environment_stats)
    app.router.add_get('/v1/environments/{environment_id}/managers', list_environment_managers)


__all__ = ['register_environment_routes']
