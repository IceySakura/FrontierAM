from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile

from pathlib import Path
from typing import Any

from aiohttp import web

from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.artifacts.store import StoredObject
from nexus.distributed.config import DistributedConfig
from nexus.distributed.environment.limits import effective_artifact_size_limit
from nexus.distributed.errors import (
    DomainConflictError,
    ResourceNotFoundError,
)
from nexus.distributed.gateways.application.routes.artifact_dependency_build import (
    _upload_dependency_wheelhouse,
)
from nexus.distributed.gateways.application.routes.artifact_shared import (
    InternalArtifactScope,
    _artifact_ref_payload,
    _artifact_transfer_reservation,
    _artifact_upload_deadline,
    _artifact_upload_failure_code,
    _internal_artifact_idempotency_scope,
    _internal_artifact_kind,
    _internal_producer_key,
    _receiving_upload_lease,
    _release_receiving,
    _required_scope_value,
    _resolve_internal_artifact_scope,
    _validate_run_result_size,
    _write_upload_body,
)
from nexus.distributed.gateways.common.keys import (
    ARTIFACT_STORE_KEY,
    ARTIFACT_UPLOAD_COORDINATOR_KEY,
    CONFIG_KEY,
    SESSION_FACTORY_KEY,
)
from nexus.distributed.persistence.db import run_idempotent_transaction
from nexus.distributed.persistence.orm import (
    ArtifactRow,
    BatchRunRow,
    EnvironmentRow,
)
from nexus.distributed.repositories.artifact import ArtifactRepository, CreateArtifactUpload, artifact_object_key
from nexus.distributed.repositories.batch import BatchRepository
from nexus.distributed.utils.clock import utcnow
from nexus.utils.common import to_thread_complete


async def upload_internal_task_list(request: web.Request) -> web.Response:
    return await upload_internal_artifact_content(request)


async def upload_internal_artifact_content(request: web.Request) -> web.Response:
    config = request.app[CONFIG_KEY]
    kind = _internal_artifact_kind(request.match_info.get('artifact_kind', 'task-list'))
    sha256 = request.headers.get('X-Nexus-Sha256', '').strip()
    size_bytes = request.content_length
    if len(sha256) != 64 or any(char not in '0123456789abcdef' for char in sha256):
        raise web.HTTPBadRequest(text='invalid_internal_artifact_sha256')
    if size_bytes is None or size_bytes > config.artifact_upload_max_size_bytes:
        raise web.HTTPRequestEntityTooLarge(
            max_size=config.artifact_upload_max_size_bytes,
            actual_size=size_bytes or 0,
        )
    _validate_run_result_size(config, kind, size_bytes)

    async with _artifact_upload_deadline(config):
        async with _artifact_transfer_reservation(
            request.app[ARTIFACT_UPLOAD_COORDINATOR_KEY].admission,
            direction='upload',
            kind=kind,
            size_bytes=size_bytes,
        ):
            return await _upload_internal_artifact_content_reserved(
                request,
                config=config,
                kind=kind,
                sha256=sha256,
                size_bytes=size_bytes,
            )


async def _upload_internal_artifact_content_reserved(
    request: web.Request,
    *,
    config: DistributedConfig,
    kind: str,
    sha256: str,
    size_bytes: int,
) -> web.Response:
    if kind == 'wheelhouse':
        return await _upload_dependency_wheelhouse(request, sha256=sha256, size_bytes=size_bytes)
    operation = 'run_result:write' if kind == 'run_result' else 'task_list:write'
    store = request.app[ARTIFACT_STORE_KEY]

    async def claim_internal_upload(session: Any) -> tuple[Any, Any, str | None, Any | None, StoredObject]:
        scope = await _resolve_internal_artifact_scope(request, session, operation=operation)
        environment = await session.get(EnvironmentRow, scope.environment_id)
        max_size = effective_artifact_size_limit(
            environment.resource_limits_json if environment is not None else {},
            global_max_size_bytes=config.artifact_upload_max_size_bytes,
        )
        if size_bytes > max_size:
            raise web.HTTPRequestEntityTooLarge(max_size=max_size, actual_size=size_bytes)
        producer_key = await _internal_producer_key(session, scope=scope, kind=kind)
        idempotency_key = hashlib.sha256(producer_key.encode()).hexdigest()
        repo = ArtifactRepository(session)
        upload, _ = await repo.create_upload(
            CreateArtifactUpload(
                owner_principal_id=scope.owner_principal_id,
                environment_id=scope.environment_id,
                owner_tenant=scope.execution.owner_tenant,
                created_by='system:artifact-content',
                kind=kind,
                sha256=sha256,
                size_bytes=size_bytes,
                idempotency_scope=_internal_artifact_idempotency_scope(
                    scope.environment_id,
                    scope.execution_id,
                    scope.attempt_id,
                    kind,
                ),
                idempotency_key=idempotency_key,
                session_ttl=config.artifact_upload_session_ttl,
                artifact_ttl=config.artifact_default_ttl,
                producer_key=producer_key,
            ),
            store=store,
            now=utcnow(),
        )
        artifact_row = await session.get(ArtifactRow, upload.artifact_id)
        if artifact_row is None:
            raise ResourceNotFoundError('artifact_not_found')
        stored_object = StoredObject(uri=artifact_row.uri, etag=artifact_row.provider_etag)
        upload, operation_id = await repo.claim_receiving(
            upload.upload_id,
            owner_principal_id=scope.owner_principal_id,
            environment_id=scope.environment_id,
            operation_owner='application-gateway:internal',
            lease_ttl=config.artifact_upload_lease_ttl,
            now=utcnow(),
        )
        if operation_id is None:
            if kind == 'run_result':
                await repo.link_run_result(
                    upload.artifact_id,
                    execution_id=scope.execution_id,
                    assignment_id=_required_scope_value(scope.assignment_id, 'assignment_id'),
                    attempt_id=_required_scope_value(scope.attempt_id, 'attempt_id'),
                    now=utcnow(),
                )
            artifact = await repo.get_artifact(
                upload.artifact_id,
                owner_principal_id=scope.owner_principal_id,
                environment_id=scope.environment_id,
            )
            if kind == 'task_list':
                await _link_internal_task_list(
                    session,
                    scope=scope,
                    ref=ArtifactRef.model_validate(_artifact_ref_payload(artifact)),
                    config=config,
                )
            return scope, upload, operation_id, artifact, stored_object
        return scope, upload, operation_id, None, stored_object

    scope, upload, operation_id, ready_artifact, stored_object = await run_idempotent_transaction(
        request.app[SESSION_FACTORY_KEY],
        claim_internal_upload,
        conflict_markers=(
            'uq_artifacts_owner_producer',
            'distributed_artifacts.owner_principal_id',
            'uq_artifact_uploads_idempotency',
            'distributed_artifact_uploads.idempotency_scope',
        ),
    )
    if operation_id is None:
        return web.json_response({'ref': _artifact_ref_payload(ready_artifact)})

    spool = Path(config.artifact_upload_spool_dir)
    spool.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'{upload.upload_id}.', suffix='.internal', dir=spool)
    os.close(fd)
    temporary = Path(name)
    try:
        async with _receiving_upload_lease(
            request,
            upload_id=upload.upload_id,
            operation_id=operation_id,
        ) as lease:
            identity = await _write_upload_body(
                request,
                temporary,
                expected_size_bytes=size_bytes,
                expected_sha256=sha256,
                max_size_bytes=config.artifact_upload_max_size_bytes,
            )
            object_key = artifact_object_key(
                owner_principal_id=scope.owner_principal_id,
                artifact_id=upload.artifact_id,
                sha256=sha256,
                kind=kind,
            )
            await lease.renew_now()
            stored = await request.app[ARTIFACT_UPLOAD_COORDINATOR_KEY].put_reserved(
                store,
                temporary,
                object_key=object_key,
                sha256=sha256,
                size_bytes=size_bytes,
                stored_object=stored_object,
                identity=identity,
            )
            await lease.renew_now()
        async with request.app[SESSION_FACTORY_KEY]() as session:
            scope = await _resolve_internal_artifact_scope(request, session, operation=operation)
            repo = ArtifactRepository(session)
            if kind == 'run_result':
                ref = await repo.publish_run_result_ready(
                    upload.upload_id,
                    operation_id=operation_id,
                    execution_id=scope.execution_id,
                    assignment_id=_required_scope_value(scope.assignment_id, 'assignment_id'),
                    attempt_id=_required_scope_value(scope.attempt_id, 'attempt_id'),
                    provider_etag=stored.etag,
                    now=utcnow(),
                )
            else:
                ref = await repo.publish_ready(
                    upload.upload_id,
                    operation_id=operation_id,
                    provider_etag=stored.etag,
                    now=utcnow(),
                )
                await repo.add_ref(
                    ref.artifact_id,
                    owner_principal_id=scope.owner_principal_id,
                    environment_id=scope.environment_id,
                    resource_type='execution',
                    resource_id=scope.execution_id,
                    now=utcnow(),
                )
                await _link_internal_task_list(session, scope=scope, ref=ref, config=config)
            await session.commit()
        return web.json_response({'ref': ref.model_dump(mode='json')}, status=201)
    except BaseException as exc:
        await asyncio.shield(
            _release_receiving(
                request,
                upload_id=upload.upload_id,
                operation_id=operation_id,
                failure_code=_artifact_upload_failure_code(exc),
            )
        )
        raise
    finally:
        await to_thread_complete(temporary.unlink, missing_ok=True)


async def _link_internal_task_list(
    session: Any,
    *,
    scope: InternalArtifactScope,
    ref: ArtifactRef,
    config: DistributedConfig,
) -> None:
    batch = await session.get(BatchRunRow, scope.execution_id)
    if batch is None or not isinstance(batch.task_list_manifest_json, dict):
        raise DomainConflictError('batch_task_list_manifest_not_staged')
    repo = BatchRepository(session, config=config)
    await repo.fence_parent_attempt(scope.execution_id, scope.attempt_id)
    await repo.set_generated_task_list(
        scope.execution_id,
        ref=ref,
        manifest=dict(batch.task_list_manifest_json),
        generation_epoch=batch.generation_epoch,
        now=utcnow(),
    )
