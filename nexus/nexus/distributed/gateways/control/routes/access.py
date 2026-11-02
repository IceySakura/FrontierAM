from __future__ import annotations

import hashlib
import json

from datetime import datetime

from aiohttp import web
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.distributed.access.auth import auth_context, csrf_token
from nexus.distributed.access.taihu import parse_tai_identity
from nexus.distributed.environment.limits import EnvironmentResourceLimits
from nexus.distributed.errors import DomainValidationError, ResourceNotFoundError
from nexus.distributed.gateways.common.keys import (
    AUTHORIZATION_KEY,
    CONFIG_KEY,
    SESSION_FACTORY_KEY,
    TRACE_DATABASE_ACCOUNT_MANAGER_KEY,
)
from nexus.distributed.gateways.common.serde import to_jsonable
from nexus.distributed.gateways.control.routes._shared import query_int as _query_int
from nexus.distributed.gateways.control.routes._shared import read_model as _read_model
from nexus.distributed.persistence.orm import SubmitTokenRow
from nexus.distributed.repositories.access import CANONICAL_USER_SCOPES, AccessRepository
from nexus.distributed.repositories.audit import AuditRepository
from nexus.distributed.repositories.environment import EnvironmentRepository
from nexus.distributed.utils.clock import utcnow
from nexus.distributed.utils.identity import (
    IDENTITY_DISPLAY_NAME_MAX_LENGTH,
    IDENTITY_ISSUER_MAX_LENGTH,
    IDENTITY_SUBJECT_MAX_LENGTH,
    IDENTITY_TENANT_MAX_LENGTH,
)


class UserRolesPayload(BaseModel):
    issuer: str = Field(min_length=1, max_length=IDENTITY_ISSUER_MAX_LENGTH)
    tenant: str = Field(min_length=1, max_length=IDENTITY_TENANT_MAX_LENGTH)
    subject: str = Field(min_length=1, max_length=IDENTITY_SUBJECT_MAX_LENGTH)
    roles: list[str]
    display_name: str | None = Field(default=None, max_length=IDENTITY_DISPLAY_NAME_MAX_LENGTH)


class EnvironmentGrantPayload(BaseModel):
    model_config = ConfigDict(extra='forbid')

    limits: EnvironmentResourceLimits = Field(default_factory=EnvironmentResourceLimits)
    scopes: list[str] = Field(default_factory=lambda: sorted(CANONICAL_USER_SCOPES))
    enabled: bool = True

    @field_validator('scopes')
    @classmethod
    def validate_scopes(cls, value: list[str]) -> list[str]:
        if not set(value) <= CANONICAL_USER_SCOPES:
            raise ValueError('environment grant contains unsupported scopes')
        return sorted(set(value))


class QuotaPayload(BaseModel):
    limits: dict[str, int]

    @field_validator('limits')
    @classmethod
    def validate_limits(cls, value: dict[str, int]) -> dict[str, int]:
        if any(limit < 0 for limit in value.values()):
            raise ValueError('quota limits must be non-negative')
        return value


class AppBindingPayload(BaseModel):
    model_config = ConfigDict(extra='forbid')

    app_id: str = Field(min_length=1, max_length=128)


def _principal_access_grant_resource_id(owner_principal_id: str, grantee_principal_id: str) -> str:
    digest = hashlib.sha256(f'{owner_principal_id}\0{grantee_principal_id}'.encode()).hexdigest()
    return f'pag_{digest}'


async def _update_token_app_binding(
    session: AsyncSession,
    *,
    actor_principal_id: str,
    token_id: str,
    app_id: str | None,
    now: datetime,
) -> SubmitTokenRow:
    repository = AccessRepository(session)
    if app_id is None:
        row = await repository.unbind_token_app_id(token_id)
        operation = 'access.token.app_unbind'
        metadata = {'token_kind': row.token_kind}
    else:
        row = await repository.bind_token_app_id(token_id, app_id=app_id)
        operation = 'access.token.app_bind'
        metadata = {'app_id': row.bound_app_id, 'token_kind': row.token_kind}
    await AuditRepository(session).append(
        actor_principal_id=actor_principal_id,
        target_owner_principal_id=row.owner_principal_id,
        operation=operation,
        resource_type='submit_token',
        resource_id=row.token_id,
        metadata=metadata,
        now=now,
    )
    return row


def _app_binding_document(row: SubmitTokenRow) -> dict[str, object]:
    return {
        'token_id': row.token_id,
        'token_kind': row.token_kind,
        'owner_principal_id': row.owner_principal_id,
        'environment_id': row.environment_id,
        'app_id': row.bound_app_id,
    }


async def get_me(request: web.Request) -> web.Response:
    context = auth_context(request)
    payload = {
        'principal_id': context.principal_id,
        'issuer': context.issuer,
        'tenant': context.tenant,
        'subject': context.subject,
        'display_name': context.display_name,
        'roles': sorted(context.roles),
        'scopes': sorted(context.scopes),
        'auth_type': context.auth_type,
        'system_token': {'label': '默认'} if context.system_token_id is not None else None,
        'csrf_token': csrf_token(context, request.app[CONFIG_KEY]),
    }
    if context.actor_app_id is not None:
        payload['app_id'] = context.actor_app_id
    return web.json_response(payload)


async def put_token_app_binding(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    payload = await _read_model(request, AppBindingPayload)
    token_id = request.match_info['token_id']
    now = utcnow()
    async with request.app[SESSION_FACTORY_KEY]() as session:
        row = await _update_token_app_binding(
            session,
            actor_principal_id=context.principal_id,
            token_id=token_id,
            app_id=payload.app_id,
            now=now,
        )
        await session.commit()
    return web.json_response(_app_binding_document(row))


async def delete_token_app_binding(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    token_id = request.match_info['token_id']
    now = utcnow()
    async with request.app[SESSION_FACTORY_KEY]() as session:
        row = await _update_token_app_binding(
            session,
            actor_principal_id=context.principal_id,
            token_id=token_id,
            app_id=None,
            now=now,
        )
        await session.commit()
    return web.json_response(_app_binding_document(row))


async def put_user_app_binding(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    payload = await _read_model(request, AppBindingPayload)
    principal_id = request.match_info['principal_id']
    now = utcnow()
    async with request.app[SESSION_FACTORY_KEY]() as session:
        token = await AccessRepository(session).ensure_system_token(
            owner_principal_id=principal_id,
            created_by=context.principal_id,
            now=now,
        )
        row = await _update_token_app_binding(
            session,
            actor_principal_id=context.principal_id,
            token_id=token.token_id,
            app_id=payload.app_id,
            now=now,
        )
        await session.commit()
    return web.json_response(_app_binding_document(row))


async def delete_user_app_binding(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    principal_id = request.match_info['principal_id']
    now = utcnow()
    async with request.app[SESSION_FACTORY_KEY]() as session:
        token = await AccessRepository(session).ensure_system_token(
            owner_principal_id=principal_id,
            created_by=context.principal_id,
            now=now,
        )
        row = await _update_token_app_binding(
            session,
            actor_principal_id=context.principal_id,
            token_id=token.token_id,
            app_id=None,
            now=now,
        )
        await session.commit()
    return web.json_response(_app_binding_document(row))


async def upsert_environment_grant(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    payload = await _read_model(request, EnvironmentGrantPayload)
    environment_id = request.match_info['environment_id']
    principal_id = request.match_info['principal_id']
    now = utcnow()
    if not payload.enabled:
        await request.app[TRACE_DATABASE_ACCOUNT_MANAGER_KEY].disable_for_access_revocation(
            owner_principal_id=principal_id,
            revoked_environment_id=environment_id,
            actor_principal_id=context.principal_id,
            now=now,
        )
    async with request.app[SESSION_FACTORY_KEY]() as session:
        grant = await EnvironmentRepository(session).upsert_grant(
            principal_id=principal_id,
            environment_id=environment_id,
            limits=payload.limits.model_dump(exclude_none=True),
            scopes=set(payload.scopes),
            enabled=payload.enabled,
            actor=context.principal_id,
            now=now,
        )
        await AuditRepository(session).append(
            actor_principal_id=context.principal_id,
            target_owner_principal_id=principal_id,
            operation='environment.grant.upsert',
            resource_type='environment_grant',
            resource_id=f'{environment_id}:{principal_id}',
            metadata={
                'environment_id': environment_id,
                'enabled': grant.enabled,
                'scopes': sorted(grant.scopes),
                'version': grant.version,
            },
            now=now,
        )
        await session.commit()
    return web.json_response(to_jsonable(grant))


async def list_environment_grants(request: web.Request) -> web.Response:
    request.app[AUTHORIZATION_KEY].require_admin(auth_context(request))
    async with request.app[SESSION_FACTORY_KEY]() as session:
        grants = await EnvironmentRepository(session).list_grants(
            principal_id=request.query.get('principal_id'),
            environment_id=request.match_info.get('environment_id') or request.query.get('environment_id'),
        )
    return web.json_response({'items': to_jsonable(grants)})


async def list_my_environments(request: web.Request) -> web.Response:
    context = auth_context(request)
    async with request.app[SESSION_FACTORY_KEY]() as session:
        pairs = await EnvironmentRepository(session).list_granted_environments(
            context.principal_id,
            include_disabled=True,
        )
    if context.credential_environment_id is not None:
        pairs = [pair for pair in pairs if pair[0].environment_id == context.credential_environment_id]
    items = [{**to_jsonable(environment), 'grant': to_jsonable(grant)} for environment, grant in pairs]
    trace_database_accounts = []
    if context.auth_type != 'tai_app':
        trace_database_accounts = await request.app[TRACE_DATABASE_ACCOUNT_MANAGER_KEY].list_for_principal(
            context.principal_id
        )
    return web.json_response({
        'items': items,
        'trace_database_accounts': to_jsonable(trace_database_accounts),
    })


async def list_environment_catalog(request: web.Request) -> web.Response:
    context = auth_context(request)
    async with request.app[SESSION_FACTORY_KEY]() as session:
        pairs = await EnvironmentRepository(session).list_environment_catalog(context.principal_id)
    if context.credential_environment_id is not None:
        pairs = [pair for pair in pairs if pair[0].environment_id == context.credential_environment_id]
    items = [
        {
            **to_jsonable(environment),
            'claimable': grant is None and environment.default_quota is not None,
            'grant': to_jsonable(grant) if grant is not None else None,
        }
        for environment, grant in pairs
    ]
    return web.json_response({
        'system_token': {'label': '默认'},
        'items': items,
    })


async def claim_environment_default_quota(request: web.Request) -> web.Response:
    context = auth_context(request)
    if context.auth_type == 'tai_app':
        raise web.HTTPForbidden(text='app_identity_cannot_claim_quota')
    now = utcnow()
    async with request.app[SESSION_FACTORY_KEY]() as session:
        system_token_id = context.system_token_id
        if system_token_id is None:
            access = AccessRepository(session)
            if await access.get_user(context.principal_id) is None:
                await access.ensure_user(
                    principal_id=context.principal_id,
                    issuer=context.issuer,
                    tenant=context.tenant,
                    subject=context.subject,
                    roles=set(context.roles),
                    display_name=context.display_name,
                    actor='system:default-token',
                    now=now,
                )
            system_token = await access.ensure_system_token(
                owner_principal_id=context.principal_id,
                created_by='system:default-token',
                now=now,
            )
            system_token_id = system_token.token_id
        repository = EnvironmentRepository(session)
        grant, created = await repository.claim_default_quota(
            principal_id=context.principal_id,
            environment_id=request.match_info['environment_id'],
            actor=context.principal_id,
            now=now,
        )
        if created:
            environment = await repository.get_environment(grant.environment_id)
            await AuditRepository(session).append(
                actor_principal_id=context.principal_id,
                target_owner_principal_id=context.principal_id,
                operation='environment.quota.claim',
                resource_type='environment_grant',
                resource_id=f'{grant.environment_id}:{context.principal_id}',
                metadata={
                    'environment_id': grant.environment_id,
                    'system_token_id': system_token_id,
                    'limits': grant.limits,
                    'config_version': environment.config_version,
                },
                now=now,
            )
        await session.commit()
    return web.json_response({
        'system_token': {'label': '默认'},
        'grant': to_jsonable(grant),
    })


async def list_access_users(request: web.Request) -> web.Response:
    context = auth_context(request)
    async with request.app[SESSION_FACTORY_KEY]() as session:
        repository = AccessRepository(session)
        users = await repository.list_users()
        if context.is_admin:
            app_bindings = await repository.list_system_token_app_bindings()
            items = [{**to_jsonable(user), 'app_id': app_bindings.get(user.principal_id)} for user in users]
        else:
            if context.auth_type != 'tai_identity':
                raise web.HTTPForbidden(text='personal_identity_required')
            items = [
                {
                    'principal_id': user.principal_id,
                    'issuer': user.issuer,
                    'subject': user.subject,
                    'display_name': user.display_name,
                }
                for user in users
            ]
    return web.json_response({'items': items})


def _require_principal_grant_manager(context, owner_principal_id: str) -> None:
    if context.auth_type not in {'tai_identity', 'disabled'}:
        raise web.HTTPForbidden(text='personal_identity_required')
    if not context.is_admin and context.principal_id != owner_principal_id:
        raise web.HTTPNotFound(text='resource_not_found')


async def list_principal_access_grants(request: web.Request) -> web.Response:
    context = auth_context(request)
    async with request.app[SESSION_FACTORY_KEY]() as session:
        repository = AccessRepository(session)
        if context.is_admin:
            grants = await repository.list_principal_access_grants(
                owner_principal_id=request.query.get('owner_principal_id'),
                grantee_principal_id=request.query.get('grantee_principal_id'),
            )
        else:
            if context.auth_type != 'tai_identity':
                raise web.HTTPForbidden(text='personal_identity_required')
            grants = await repository.list_principal_access_grants(
                owner_principal_id=context.principal_id,
            )
    return web.json_response({'items': to_jsonable(grants)})


async def put_principal_access_grant(request: web.Request) -> web.Response:
    context = auth_context(request)
    owner_principal_id = request.match_info['owner_principal_id']
    grantee_principal_id = request.match_info['grantee_principal_id']
    _require_principal_grant_manager(context, owner_principal_id)
    now = utcnow()
    async with request.app[SESSION_FACTORY_KEY]() as session:
        grant = await AccessRepository(session).put_principal_access_grant(
            owner_principal_id=owner_principal_id,
            grantee_principal_id=grantee_principal_id,
            actor=context.principal_id,
            now=now,
        )
        await AuditRepository(session).append(
            actor_principal_id=context.principal_id,
            target_owner_principal_id=owner_principal_id,
            operation='principal_access.grant',
            resource_type='principal_access_grant',
            resource_id=_principal_access_grant_resource_id(owner_principal_id, grantee_principal_id),
            metadata={'grantee_principal_id': grantee_principal_id},
            now=now,
        )
        await session.commit()
    return web.json_response(to_jsonable(grant))


async def delete_principal_access_grant(request: web.Request) -> web.Response:
    context = auth_context(request)
    owner_principal_id = request.match_info['owner_principal_id']
    grantee_principal_id = request.match_info['grantee_principal_id']
    _require_principal_grant_manager(context, owner_principal_id)
    now = utcnow()
    async with request.app[SESSION_FACTORY_KEY]() as session:
        deleted = await AccessRepository(session).delete_principal_access_grant(
            owner_principal_id=owner_principal_id,
            grantee_principal_id=grantee_principal_id,
        )
        if deleted:
            await AuditRepository(session).append(
                actor_principal_id=context.principal_id,
                target_owner_principal_id=owner_principal_id,
                operation='principal_access.revoke',
                resource_type='principal_access_grant',
                resource_id=_principal_access_grant_resource_id(owner_principal_id, grantee_principal_id),
                metadata={'grantee_principal_id': grantee_principal_id},
                now=now,
            )
        await session.commit()
    return web.Response(status=204)


async def put_access_user_roles(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    payload = await _read_model(request, UserRolesPayload)
    principal_id = request.match_info['principal_id']
    identity = parse_tai_identity(
        json.dumps({
            'issuer': payload.issuer,
            'tenant': payload.tenant,
            'subject': payload.subject,
            'display_name': payload.display_name,
        })
    )
    now = utcnow()
    async with request.app[SESSION_FACTORY_KEY]() as session:
        repo = AccessRepository(session)
        existing = await repo.get_user(principal_id)
        if existing is None:
            if identity.principal_id != principal_id:
                raise DomainValidationError('principal_identity_mismatch')
        elif (existing.issuer, existing.tenant, existing.subject) != (
            identity.issuer,
            identity.tenant,
            identity.subject,
        ):
            raise DomainValidationError('principal_identity_mismatch')
        user = await repo.upsert_user(
            principal_id=principal_id,
            issuer=identity.issuer,
            tenant=identity.tenant,
            subject=identity.subject,
            roles=set(payload.roles),
            display_name=identity.display_name,
            actor=context.principal_id,
            now=now,
        )
        await AuditRepository(session).append(
            actor_principal_id=context.principal_id,
            target_owner_principal_id=user.principal_id,
            operation='access.roles.update',
            resource_type='user',
            resource_id=user.principal_id,
            metadata={'roles': sorted(user.roles)},
            now=now,
        )
        await session.commit()
    return web.json_response(to_jsonable(user))


async def get_access_quota(request: web.Request) -> web.Response:
    context = auth_context(request)
    owner = request.match_info['principal_id']
    request.app[AUTHORIZATION_KEY].require_owner(context, owner)
    async with request.app[SESSION_FACTORY_KEY]() as session:
        row = await AccessRepository(session).get_quota(owner)
    overrides = dict(row.limits_json or {}) if row is not None else {}
    limits = {**request.app[CONFIG_KEY].default_user_quota, **overrides}
    return web.json_response({
        'owner_principal_id': owner,
        'limits': limits,
        'overrides': overrides,
        'version': row.version if row is not None else 0,
        'override': row is not None,
    })


async def put_access_quota(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    payload = await _read_model(request, QuotaPayload)
    unknown_limits = set(payload.limits).difference(request.app[CONFIG_KEY].default_user_quota)
    if unknown_limits:
        raise DomainValidationError(
            'unknown_quota_limits',
            detail=','.join(sorted(unknown_limits)),
        )
    owner = request.match_info['principal_id']
    now = utcnow()
    async with request.app[SESSION_FACTORY_KEY]() as session:
        if await AccessRepository(session).get_user(owner) is None:
            raise ResourceNotFoundError('user_not_found')
        row = await AccessRepository(session).set_quota(
            owner,
            limits=payload.limits,
            actor=context.principal_id,
            now=now,
        )
        await AuditRepository(session).append(
            actor_principal_id=context.principal_id,
            target_owner_principal_id=owner,
            operation='access.quota.update',
            resource_type='user',
            resource_id=owner,
            metadata={'version': row.version},
            now=now,
        )
        await session.commit()
    limits = {**request.app[CONFIG_KEY].default_user_quota, **(row.limits_json or {})}
    return web.json_response({
        'owner_principal_id': owner,
        'limits': limits,
        'overrides': row.limits_json,
        'version': row.version,
        'override': True,
    })


async def list_access_audit(request: web.Request) -> web.Response:
    request.app[AUTHORIZATION_KEY].require_admin(auth_context(request))
    limit = _query_int(request, 'limit', 100, min_value=1, max_value=500)
    offset = _query_int(request, 'offset', 0, min_value=0)
    async with request.app[SESSION_FACTORY_KEY]() as session:
        rows = await AuditRepository(session).list(
            actor=request.query.get('actor'),
            target=request.query.get('target'),
            limit=limit,
            offset=offset,
        )
    return web.json_response({
        'items': [
            {
                'audit_id': row.audit_id,
                'actor_principal_id': row.actor_principal_id,
                'target_owner_principal_id': row.target_owner_principal_id,
                'operation': row.operation,
                'resource_type': row.resource_type,
                'resource_id': row.resource_id,
                'metadata': row.metadata_json,
                'created_at': row.created_at.isoformat(),
            }
            for row in rows
        ]
    })


def register_access_routes(app: web.Application) -> None:
    app.router.add_get('/v1/me', get_me)
    app.router.add_get('/v1/my-environments', list_my_environments)
    app.router.add_get('/v1/environment-catalog', list_environment_catalog)
    app.router.add_put('/v1/my-environments/{environment_id}/quota', claim_environment_default_quota)
    app.router.add_get('/v1/environment-grants', list_environment_grants)
    app.router.add_get('/v1/access/grants', list_environment_grants)
    app.router.add_get('/v1/access/users', list_access_users)
    app.router.add_get('/v1/access/principal-grants', list_principal_access_grants)
    app.router.add_put(
        '/v1/access/principal-grants/{owner_principal_id}/{grantee_principal_id}',
        put_principal_access_grant,
    )
    app.router.add_delete(
        '/v1/access/principal-grants/{owner_principal_id}/{grantee_principal_id}',
        delete_principal_access_grant,
    )
    app.router.add_put('/v1/access/tokens/{token_id}/app-binding', put_token_app_binding)
    app.router.add_delete('/v1/access/tokens/{token_id}/app-binding', delete_token_app_binding)
    app.router.add_put('/v1/access/users/{principal_id}/app-binding', put_user_app_binding)
    app.router.add_delete('/v1/access/users/{principal_id}/app-binding', delete_user_app_binding)
    app.router.add_put('/v1/access/users/{principal_id}/roles', put_access_user_roles)
    app.router.add_get('/v1/access/users/{principal_id}/quota', get_access_quota)
    app.router.add_put('/v1/access/users/{principal_id}/quota', put_access_quota)
    app.router.add_get('/v1/access/audit', list_access_audit)


__all__ = [
    'list_environment_grants',
    'register_access_routes',
    'upsert_environment_grant',
]
