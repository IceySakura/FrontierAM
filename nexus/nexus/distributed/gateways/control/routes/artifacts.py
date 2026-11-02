from __future__ import annotations

import contextlib
import hashlib

from typing import Any

from aiohttp import web
from pydantic import BaseModel, ConfigDict, Field

from nexus.distributed.access.auth import AuthContext, auth_context
from nexus.distributed.artifacts.store import create_artifact_store
from nexus.distributed.environment.limits import (
    effective_artifact_size_limit,
    effective_environment_resource_limits,
)
from nexus.distributed.errors import DomainConflictError
from nexus.distributed.gateways.common.keys import (
    AUTHORIZATION_KEY,
    CONFIG_KEY,
    SESSION_FACTORY_KEY,
    UPLOAD_TICKET_SIGNER_KEY,
)
from nexus.distributed.gateways.common.serde import to_jsonable
from nexus.distributed.gateways.control.routes._shared import read_model as _read_model
from nexus.distributed.persistence.db import run_idempotent_transaction as _run_idempotent_transaction
from nexus.distributed.repositories.access import AccessRepository
from nexus.distributed.repositories.artifact import ArtifactRepository, CreateArtifactUpload
from nexus.distributed.repositories.environment import EnvironmentRepository
from nexus.distributed.utils.clock import utcnow
from nexus.distributed.utils.identity import (
    ENVIRONMENT_ID_MAX_LENGTH,
    PRINCIPAL_ID_MAX_LENGTH,
    SUBMISSION_IDEMPOTENCY_KEY_MAX_LENGTH,
)


class CreateArtifactUploadPayload(BaseModel):
    model_config = ConfigDict(extra='forbid')

    kind: str
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    size_bytes: int = Field(ge=0)
    idempotency_key: str = Field(min_length=1, max_length=SUBMISSION_IDEMPOTENCY_KEY_MAX_LENGTH)
    owner_principal_id: str | None = Field(default=None, min_length=1, max_length=PRINCIPAL_ID_MAX_LENGTH)
    environment_id: str | None = Field(default=None, min_length=1, max_length=ENVIRONMENT_ID_MAX_LENGTH)


def _request_environment_id(context: AuthContext, explicit_environment_id: str | None) -> str:
    if explicit_environment_id:
        if (
            context.credential_environment_id is not None
            and explicit_environment_id != context.credential_environment_id
        ):
            raise web.HTTPNotFound(text='resource_not_found')
        return explicit_environment_id
    if context.credential_environment_id is not None:
        return context.credential_environment_id
    if context.auth_type == 'disabled':
        return 'default'
    raise web.HTTPUnprocessableEntity(text='environment_id_required')


async def _require_environment_scope(
    session: Any,
    context: AuthContext,
    *,
    owner_principal_id: str,
    environment_id: str,
    scope: str,
) -> None:
    if context.is_admin or context.auth_type == 'disabled':
        return
    await EnvironmentRepository(session).require_grant_scope(
        owner_principal_id,
        environment_id,
        required_scope=scope,
    )


async def create_artifact_upload(request: web.Request) -> web.Response:
    payload = await _read_model(request, CreateArtifactUploadPayload)
    context = auth_context(request)
    policy = request.app[AUTHORIZATION_KEY]
    policy.require_scope(context, 'artifact:write')
    owner = policy.effective_owner(context, payload.owner_principal_id)
    environment_id = _request_environment_id(context, payload.environment_id)
    config = request.app[CONFIG_KEY]
    if payload.size_bytes > config.artifact_upload_max_size_bytes:
        raise web.HTTPRequestEntityTooLarge(
            max_size=config.artifact_upload_max_size_bytes,
            actual_size=payload.size_bytes,
        )
    owner_tenant = context.tenant if owner == context.principal_id else None
    if owner != context.principal_id:
        async with request.app[SESSION_FACTORY_KEY]() as session:
            delegated = await AccessRepository(session).get_user(owner)
        if delegated is None:
            raise web.HTTPNotFound(text='owner_not_found')
        owner_tenant = delegated.tenant
    scope = hashlib.sha256(f'{context.principal_id}\0{owner}\0{environment_id}\0artifact-upload'.encode()).hexdigest()
    store = create_artifact_store(config, access='metadata')
    try:

        async def create_upload(session: Any) -> tuple[Any, Any]:
            environment = None
            environments = EnvironmentRepository(session)
            if context.auth_type != 'disabled':
                environment, grant = await environments.require_submission_access(
                    owner,
                    environment_id,
                    required_scope='artifact:write',
                )
            else:
                grant = None
                with contextlib.suppress(KeyError):
                    environment = await environments.get_environment(environment_id)
            if environment is not None:
                limits = (
                    effective_environment_resource_limits(environment.resource_limits, grant.limits)
                    if grant is not None
                    else environment.resource_limits
                )
                max_size = effective_artifact_size_limit(
                    limits,
                    global_max_size_bytes=config.artifact_upload_max_size_bytes,
                )
                if payload.size_bytes > max_size:
                    raise web.HTTPRequestEntityTooLarge(max_size=max_size, actual_size=payload.size_bytes)
            return await ArtifactRepository(session).create_upload(
                CreateArtifactUpload(
                    owner_principal_id=owner,
                    environment_id=environment_id,
                    owner_tenant=owner_tenant,
                    created_by=context.principal_id,
                    kind=payload.kind,
                    sha256=payload.sha256,
                    size_bytes=payload.size_bytes,
                    idempotency_scope=scope,
                    idempotency_key=payload.idempotency_key,
                    session_ttl=config.artifact_upload_session_ttl,
                    artifact_ttl=config.artifact_default_ttl,
                ),
                store=store,
                now=utcnow(),
            )

        upload, artifact = await _run_idempotent_transaction(
            request.app[SESSION_FACTORY_KEY],
            create_upload,
            conflict_markers=(
                'uq_artifact_uploads_idempotency',
                'distributed_artifact_uploads.idempotency_scope',
            ),
        )
    finally:
        await store.close()
    return web.json_response(
        _upload_response(
            request,
            upload=upload,
            artifact=artifact,
            environment_id=environment_id,
        ),
        status=201 if upload.status == 'pending' else 200,
    )


async def get_artifact_upload(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_scope(context, 'artifact:write')
    environment_id = _request_environment_id(context, request.query.get('environment_id'))
    owner = request.query.get('owner') if context.is_admin else context.principal_id
    owner = owner or context.principal_id
    async with request.app[SESSION_FACTORY_KEY]() as session:
        await _require_environment_scope(
            session,
            context,
            owner_principal_id=owner,
            environment_id=environment_id,
            scope='artifact:write',
        )
        repo = ArtifactRepository(session)
        upload = await repo.get_upload(
            request.match_info['upload_id'],
            owner_principal_id=owner,
            environment_id=environment_id,
        )
        artifact = await repo.get_artifact(
            upload.artifact_id,
            owner_principal_id=owner,
            environment_id=environment_id,
        )
    return web.json_response(
        _upload_response(
            request,
            upload=upload,
            artifact=artifact,
            environment_id=environment_id,
            issue_ticket=False,
        )
    )


async def reissue_artifact_upload_ticket(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_scope(context, 'artifact:write')
    environment_id = _request_environment_id(context, request.query.get('environment_id'))
    owner = request.query.get('owner') if context.is_admin else context.principal_id
    owner = owner or context.principal_id
    async with request.app[SESSION_FACTORY_KEY]() as session:
        await _require_environment_scope(
            session,
            context,
            owner_principal_id=owner,
            environment_id=environment_id,
            scope='artifact:write',
        )
        repo = ArtifactRepository(session)
        upload = await repo.get_upload(
            request.match_info['upload_id'],
            owner_principal_id=owner,
            environment_id=environment_id,
        )
        if upload.expires_at <= utcnow() or upload.status not in {'pending', 'receiving', 'completed'}:
            raise DomainConflictError('artifact_upload_not_retryable')
        artifact = await repo.get_artifact(
            upload.artifact_id,
            owner_principal_id=owner,
            environment_id=environment_id,
        )
    return web.json_response(
        _upload_response(
            request,
            upload=upload,
            artifact=artifact,
            environment_id=environment_id,
        )
    )


async def get_artifact(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_scope(context, 'execution:read')
    environment_id = _request_environment_id(context, request.query.get('environment_id'))
    owner = request.query.get('owner') or context.principal_id
    request.app[AUTHORIZATION_KEY].require_shared_owner_access(context, owner)
    async with request.app[SESSION_FACTORY_KEY]() as session:
        await _require_environment_scope(
            session,
            context,
            owner_principal_id=owner,
            environment_id=environment_id,
            scope='execution:read',
        )
        artifact = await ArtifactRepository(session).get_artifact(
            request.match_info['artifact_id'],
            owner_principal_id=owner,
            environment_id=environment_id,
        )
    result = to_jsonable(artifact)
    result['environment_id'] = environment_id
    if artifact.status == 'ready':
        result['ref'] = {
            'artifact_id': artifact.artifact_id,
            'sha256': artifact.sha256,
            'size_bytes': artifact.size_bytes,
            'kind': artifact.kind,
        }
    return web.json_response(result)


def _upload_response(
    request: web.Request,
    *,
    upload: Any,
    artifact: Any,
    environment_id: str,
    issue_ticket: bool = True,
) -> dict[str, Any]:
    result = {
        'upload': to_jsonable(upload),
        'artifact': to_jsonable(artifact),
        'environment_id': environment_id,
        'upload_url': f'{request.app[CONFIG_KEY].application_public_base_url.rstrip("/")}/v1/artifact-uploads/{upload.upload_id}/content',
    }
    if upload.status == 'completed' and artifact.status == 'ready':
        result['ref'] = {
            'artifact_id': artifact.artifact_id,
            'sha256': artifact.sha256,
            'size_bytes': artifact.size_bytes,
            'kind': artifact.kind,
        }
    if issue_ticket and upload.status != 'completed':
        result['upload_ticket'] = request.app[UPLOAD_TICKET_SIGNER_KEY].issue(
            upload_id=upload.upload_id,
            artifact_id=upload.artifact_id,
            owner_principal_id=upload.owner_principal_id,
            environment_id=environment_id,
            expected_kind=upload.expected_kind,
            expected_size_bytes=upload.expected_size_bytes,
            expected_sha256=upload.expected_sha256,
            now=utcnow(),
        )
    return result


def register_artifact_routes(app: web.Application) -> None:
    app.router.add_post('/v1/artifact-uploads', create_artifact_upload)
    app.router.add_get('/v1/artifact-uploads/{upload_id}', get_artifact_upload)
    app.router.add_post('/v1/artifact-uploads/{upload_id}/ticket', reissue_artifact_upload_ticket)
    app.router.add_get('/v1/artifacts/{artifact_id}', get_artifact)


__all__ = ['register_artifact_routes']
