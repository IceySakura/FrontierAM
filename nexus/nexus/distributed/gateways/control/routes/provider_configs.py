from __future__ import annotations

from typing import Any, Literal

from aiohttp import web
from pydantic import BaseModel, ConfigDict, Field

from nexus.distributed.access.auth import auth_context
from nexus.distributed.config import ProviderCredential
from nexus.distributed.environment.models import ProviderConfigRecord
from nexus.distributed.environment.provider_config import (
    ProviderConfig,
    parse_provider_config,
    require_provider_type_enabled,
)
from nexus.distributed.errors import DomainConflictError
from nexus.distributed.gateways.common.keys import AUTHORIZATION_KEY, CONFIG_KEY, SESSION_FACTORY_KEY
from nexus.distributed.gateways.common.serde import to_jsonable
from nexus.distributed.gateways.control.routes._shared import is_truthy_query, read_model
from nexus.distributed.repositories.audit import AuditRepository
from nexus.distributed.repositories.provider_config import ProviderConfigRepository
from nexus.distributed.repositories.provider_credential import ProviderCredentialRepository
from nexus.distributed.utils.clock import utcnow


class CreateProviderConfigPayload(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)

    provider_config_ref: str = Field(min_length=1, max_length=512)
    config: ProviderConfig
    credential: ProviderCredential | None = None
    status: Literal['enabled', 'disabled'] = 'disabled'


class UpdateProviderConfigPayload(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)

    config: ProviderConfig | None = None
    credential: ProviderCredential | None = None
    status: Literal['enabled', 'disabled'] | None = None


async def create_provider_config(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    payload = await read_model(request, CreateProviderConfigPayload)
    if 'credential' in payload.model_fields_set and payload.credential is None:
        raise web.HTTPBadRequest(text='provider_credential_must_be_an_object')
    _require_enabled_provider_type(request, provider_type=payload.config.provider_type, status=payload.status)
    now = utcnow()
    async with request.app[SESSION_FACTORY_KEY]() as session:
        credential_repo = ProviderCredentialRepository(session)
        await _save_and_validate_credential(
            credential_repo,
            config=payload.config,
            credential=payload.credential,
            status=payload.status,
            actor=context.principal_id,
        )
        record = await ProviderConfigRepository(session).create(
            provider_config_ref=payload.provider_config_ref,
            config=payload.config.model_dump(mode='json'),
            status=payload.status,
            actor=context.principal_id,
            now=now,
        )
        await AuditRepository(session).append(
            actor_principal_id=context.principal_id,
            operation='provider_config.create',
            resource_type='provider_config',
            resource_id=record.provider_config_ref,
            metadata={'provider_type': record.provider_type, 'status': record.status, 'version': record.version},
            now=now,
        )
        response_payload = await _provider_config_payload(
            record,
            credential_repo,
            credential_restart_required=payload.credential is not None,
        )
        await session.commit()
    return web.json_response(response_payload, status=201)


async def list_provider_configs(request: web.Request) -> web.Response:
    request.app[AUTHORIZATION_KEY].require_admin(auth_context(request))
    include_disabled = not is_truthy_query(request.query.get('enabled_only'))
    async with request.app[SESSION_FACTORY_KEY]() as session:
        records = await ProviderConfigRepository(session).list(include_disabled=include_disabled)
        configured_types = await ProviderCredentialRepository(session).configured_types()
    return web.json_response({
        'items': [_provider_config_payload_from_types(record, configured_types) for record in records]
    })


async def get_provider_config(request: web.Request) -> web.Response:
    request.app[AUTHORIZATION_KEY].require_admin(auth_context(request))
    async with request.app[SESSION_FACTORY_KEY]() as session:
        record = await ProviderConfigRepository(session).get(request.match_info['provider_config_ref'])
        payload = await _provider_config_payload(record, ProviderCredentialRepository(session))
    return web.json_response(payload)


async def update_provider_config(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    payload = await read_model(request, UpdateProviderConfigPayload)
    if not payload.model_fields_set:
        raise web.HTTPBadRequest(text='provider_config_update_required')
    if 'config' in payload.model_fields_set and payload.config is None:
        raise web.HTTPBadRequest(text='provider_config_must_be_an_object')
    if 'credential' in payload.model_fields_set and payload.credential is None:
        raise web.HTTPBadRequest(text='provider_credential_must_be_an_object')
    now = utcnow()
    async with request.app[SESSION_FACTORY_KEY]() as session:
        repo = ProviderConfigRepository(session)
        current = await repo.get(request.match_info['provider_config_ref'])
        next_config = payload.config if payload.config is not None else parse_provider_config(current.config)
        next_status = payload.status if payload.status is not None else current.status
        _require_enabled_provider_type(
            request,
            provider_type=payload.config.provider_type if payload.config is not None else current.provider_type,
            status=next_status,
        )
        credential_repo = ProviderCredentialRepository(session)
        await _save_and_validate_credential(
            credential_repo,
            config=next_config,
            credential=payload.credential,
            status=next_status,
            actor=context.principal_id,
        )
        record = await repo.update(
            request.match_info['provider_config_ref'],
            config=payload.config.model_dump(mode='json') if payload.config is not None else None,
            status=payload.status,
            actor=context.principal_id,
            now=now,
        )
        await AuditRepository(session).append(
            actor_principal_id=context.principal_id,
            operation='provider_config.update',
            resource_type='provider_config',
            resource_id=record.provider_config_ref,
            metadata={'fields': sorted(payload.model_fields_set), 'status': record.status, 'version': record.version},
            now=now,
        )
        response_payload = await _provider_config_payload(
            record,
            credential_repo,
            credential_restart_required=payload.credential is not None,
        )
        await session.commit()
    return web.json_response(response_payload)


def _require_enabled_provider_type(request: web.Request, *, provider_type: str, status: str) -> None:
    if status == 'enabled':
        require_provider_type_enabled(provider_type, request.app[CONFIG_KEY].environment_enabled_providers)


async def _save_and_validate_credential(
    repository: ProviderCredentialRepository,
    *,
    config: ProviderConfig,
    credential: ProviderCredential | None,
    status: str,
    actor: str,
) -> None:
    provider_type = config.provider_type
    credential_ref = config.credential_ref
    if credential_ref is None:
        if credential is not None:
            raise DomainConflictError('provider_credential_not_allowed')
        return
    if credential is not None:
        if credential.provider_type != provider_type:
            raise DomainConflictError('provider_credential_type_mismatch')
        await repository.upsert(credential_ref, credential, actor=actor)
    stored = await repository.get(credential_ref)
    if stored is not None and stored.provider_type != provider_type:
        raise DomainConflictError('provider_credential_type_mismatch')
    if status == 'enabled' and stored is None:
        raise DomainConflictError('provider_credential_unavailable')


async def _provider_config_payload(
    record: ProviderConfigRecord,
    repository: ProviderCredentialRepository,
    *,
    credential_restart_required: bool = False,
) -> dict[str, Any]:
    return _provider_config_payload_from_types(
        record,
        await repository.configured_types(),
        credential_restart_required=credential_restart_required,
    )


def _provider_config_payload_from_types(
    record: ProviderConfigRecord,
    configured_types: dict[str, str],
    *,
    credential_restart_required: bool = False,
) -> dict[str, Any]:
    payload = to_jsonable(record)
    credential_ref = record.config.get('credential_ref')
    payload['credential_configured'] = (
        None if credential_ref is None else configured_types.get(str(credential_ref)) == record.provider_type
    )
    payload['credential_restart_required'] = credential_restart_required
    return payload


async def disable_provider_config(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    now = utcnow()
    async with request.app[SESSION_FACTORY_KEY]() as session:
        credential_repo = ProviderCredentialRepository(session)
        record = await ProviderConfigRepository(session).update(
            request.match_info['provider_config_ref'],
            status='disabled',
            actor=context.principal_id,
            now=now,
        )
        await AuditRepository(session).append(
            actor_principal_id=context.principal_id,
            operation='provider_config.disable',
            resource_type='provider_config',
            resource_id=record.provider_config_ref,
            metadata={'status': record.status, 'version': record.version},
            now=now,
        )
        response_payload = await _provider_config_payload(record, credential_repo)
        await session.commit()
    return web.json_response(response_payload)


def register_provider_config_routes(app: web.Application) -> None:
    app.router.add_get('/v1/provider-configs', list_provider_configs)
    app.router.add_post('/v1/provider-configs', create_provider_config)
    app.router.add_get('/v1/provider-configs/{provider_config_ref:.+}', get_provider_config)
    app.router.add_patch('/v1/provider-configs/{provider_config_ref:.+}', update_provider_config)
    app.router.add_put('/v1/provider-configs/{provider_config_ref:.+}', update_provider_config)
    app.router.add_delete('/v1/provider-configs/{provider_config_ref:.+}', disable_provider_config)


__all__ = ['register_provider_config_routes']
