from __future__ import annotations

import asyncio
import os
import tempfile

from pathlib import Path
from typing import Any

from aiohttp import web
from pydantic import ValidationError

from nexus.distributed.artifacts.file_io import read_chunks
from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.artifacts.store import StoredObject
from nexus.distributed.artifacts.tickets import UPLOAD_TICKET_HEADER
from nexus.distributed.config import DistributedConfig
from nexus.distributed.errors import (
    DomainConflictError,
    ResourceNotFoundError,
)
from nexus.distributed.gateways.application.routes.artifact_shared import (
    _artifact_download_deadline,
    _artifact_ref_payload,
    _artifact_transfer_reservation,
    _artifact_upload_deadline,
    _artifact_upload_failure_code,
    _internal_artifact_claims,
    _receiving_upload_lease,
    _release_receiving,
    _resolve_dependency_build_artifact_scope,
    _resolve_internal_artifact_scope,
    _validate_run_result_size,
    _validate_upload_claims,
    _write_upload_body,
)
from nexus.distributed.gateways.common.keys import (
    ARTIFACT_DOWNLOAD_ADMISSION_KEY,
    ARTIFACT_SPOOL_KEY,
    ARTIFACT_STORE_KEY,
    ARTIFACT_UPLOAD_COORDINATOR_KEY,
    CONFIG_KEY,
    SESSION_FACTORY_KEY,
    UPLOAD_TICKET_SIGNER_KEY,
)
from nexus.distributed.persistence.orm import (
    ArtifactRefRow,
    ArtifactRow,
    EnvironmentAllocationRow,
)
from nexus.distributed.repositories.artifact import ArtifactRepository, artifact_object_key
from nexus.distributed.utils.clock import utcnow
from nexus.utils.common import to_thread_complete


_CACHEABLE_ARTIFACT_KINDS = {'source_package', 'nexus_wheel', 'uv_lock', 'wheelhouse', 'task_list', 'data_list'}


async def upload_artifact_content(request: web.Request) -> web.Response:
    config = request.app[CONFIG_KEY]
    raw_ticket = request.headers.get(UPLOAD_TICKET_HEADER)
    if not raw_ticket:
        raise web.HTTPUnauthorized(text='upload_ticket_required')
    try:
        claims = request.app[UPLOAD_TICKET_SIGNER_KEY].verify(raw_ticket, now=utcnow())
    except ValueError as exc:
        raise web.HTTPUnauthorized(text=str(exc)) from exc
    upload_id = request.match_info['upload_id']
    if claims.upload_id != upload_id:
        raise web.HTTPUnauthorized(text='invalid_upload_ticket')
    if request.content_length is not None and request.content_length != claims.expected_size_bytes:
        raise web.HTTPRequestEntityTooLarge(
            max_size=claims.expected_size_bytes,
            actual_size=request.content_length,
        )

    _validate_run_result_size(config, claims.expected_kind, claims.expected_size_bytes)
    async with _artifact_upload_deadline(config):
        async with _artifact_transfer_reservation(
            request.app[ARTIFACT_UPLOAD_COORDINATOR_KEY].admission,
            direction='upload',
            kind=claims.expected_kind,
            size_bytes=claims.expected_size_bytes,
        ):
            return await _upload_artifact_content_reserved(
                request,
                config=config,
                claims=claims,
                upload_id=upload_id,
            )


async def _upload_artifact_content_reserved(
    request: web.Request,
    *,
    config: DistributedConfig,
    claims: Any,
    upload_id: str,
) -> web.Response:
    operation_id: str | None = None
    temporary: Path | None = None
    try:
        async with request.app[SESSION_FACTORY_KEY]() as session:
            repo = ArtifactRepository(session)
            upload, operation_id = await repo.claim_receiving(
                upload_id,
                owner_principal_id=claims.owner_principal_id,
                environment_id=claims.environment_id,
                operation_owner='application-gateway',
                lease_ttl=config.artifact_upload_lease_ttl,
                now=utcnow(),
            )
            artifact_row = await session.get(ArtifactRow, upload.artifact_id)
            if artifact_row is None:
                raise ResourceNotFoundError('artifact_not_found')
            _validate_upload_claims(claims, upload, artifact_row)
            if operation_id is None:
                artifact = await repo.get_artifact(
                    upload.artifact_id,
                    owner_principal_id=claims.owner_principal_id,
                    environment_id=claims.environment_id,
                )
                return web.json_response({'ref': _artifact_ref_payload(artifact)})
            await session.commit()

        spool = Path(config.artifact_upload_spool_dir)
        spool.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f'{upload_id}.', suffix='.upload', dir=spool)
        os.close(fd)
        temporary = Path(name)
        async with _receiving_upload_lease(request, upload_id=upload_id, operation_id=operation_id) as lease:
            identity = await _write_upload_body(
                request,
                temporary,
                expected_size_bytes=claims.expected_size_bytes,
                expected_sha256=claims.expected_sha256,
                max_size_bytes=config.artifact_upload_max_size_bytes,
            )
            object_key = artifact_object_key(
                owner_principal_id=claims.owner_principal_id,
                artifact_id=claims.artifact_id,
                sha256=claims.expected_sha256,
                kind=upload.expected_kind,
            )
            store = request.app[ARTIFACT_STORE_KEY]
            await lease.renew_now()
            stored = await request.app[ARTIFACT_UPLOAD_COORDINATOR_KEY].put_reserved(
                store,
                temporary,
                object_key=object_key,
                sha256=claims.expected_sha256,
                size_bytes=claims.expected_size_bytes,
                stored_object=StoredObject(uri=artifact_row.uri, etag=artifact_row.provider_etag),
                identity=identity,
            )
            await lease.renew_now()
        async with request.app[SESSION_FACTORY_KEY]() as session:
            ref = await ArtifactRepository(session).publish_ready(
                upload_id,
                operation_id=operation_id,
                provider_etag=stored.etag,
                now=utcnow(),
            )
            await session.commit()
        return web.json_response({'ref': ref.model_dump(mode='json')}, status=201)
    except BaseException as exc:
        if operation_id is not None:
            await asyncio.shield(
                _release_receiving(
                    request,
                    upload_id=upload_id,
                    operation_id=operation_id,
                    failure_code=_artifact_upload_failure_code(exc),
                )
            )
        raise
    finally:
        if temporary is not None:
            await to_thread_complete(temporary.unlink, missing_ok=True)


async def download_internal_artifact_content(request: web.Request) -> web.StreamResponse:
    artifact_id = request.match_info['artifact_id']
    try:
        size_bytes = int(request.headers.get('X-Nexus-Artifact-Size', ''))
    except ValueError as exc:
        raise web.HTTPBadRequest(text='invalid_internal_artifact_size') from exc
    try:
        ref = ArtifactRef.model_validate({
            'artifact_id': artifact_id,
            'kind': request.headers.get('X-Nexus-Artifact-Kind', ''),
            'sha256': request.headers.get('X-Nexus-Sha256', ''),
            'size_bytes': size_bytes,
        })
    except ValidationError as exc:
        raise web.HTTPBadRequest(text='invalid_internal_artifact_ref') from exc

    async with request.app[SESSION_FACTORY_KEY]() as session:
        claims = _internal_artifact_claims(request)
        if claims is not None and claims.dependency_build_id is not None:
            build_scope = await _resolve_dependency_build_artifact_scope(
                request,
                session,
                operation='artifact:read',
                artifact_id=artifact_id,
            )
            owner_principal_id = build_scope.owner_principal_id
            environment_id = build_scope.environment_id
        else:
            scope = await _resolve_internal_artifact_scope(
                request,
                session,
                operation='artifact:read',
                artifact_id=artifact_id,
            )
            execution_reference = await session.get(
                ArtifactRefRow,
                {
                    'artifact_id': artifact_id,
                    'resource_type': 'execution',
                    'resource_id': scope.execution_id,
                },
            )
            allocation_reference = await session.get(
                ArtifactRefRow,
                {
                    'artifact_id': artifact_id,
                    'resource_type': 'allocation',
                    'resource_id': scope.credential_allocation_id,
                },
            )
            dependency_reference = None
            allocation = await session.get(EnvironmentAllocationRow, scope.credential_allocation_id)
            if allocation is not None and allocation.dependency_build_id is not None:
                dependency_reference = await session.get(
                    ArtifactRefRow,
                    {
                        'artifact_id': artifact_id,
                        'resource_type': 'dependency_build',
                        'resource_id': allocation.dependency_build_id,
                    },
                )
            if (
                execution_reference is None
                and allocation_reference is None
                and dependency_reference is None
                and scope.execution.result_artifact_id != artifact_id
            ):
                raise web.HTTPNotFound(text='artifact_not_found')
            owner_principal_id = scope.owner_principal_id
            environment_id = scope.environment_id
        stored_object = await ArtifactRepository(session).resolve_stored_object(
            ref,
            owner_principal_id=owner_principal_id,
            environment_id=environment_id,
        )

    _validate_run_result_size(request.app[CONFIG_KEY], ref.kind, ref.size_bytes)
    cacheable = ref.kind in _CACHEABLE_ARTIFACT_KINDS
    async with _artifact_transfer_reservation(
        request.app[ARTIFACT_DOWNLOAD_ADMISSION_KEY],
        direction='download',
        kind=ref.kind,
        size_bytes=0 if cacheable else ref.size_bytes,
    ):
        async with _artifact_download_deadline(request.app[CONFIG_KEY]):
            if cacheable:

                async def populate(path: Path) -> None:
                    await _download_stored_artifact(request, stored_object, ref, path)

                async with request.app[ARTIFACT_SPOOL_KEY].open_cached(
                    sha256=ref.sha256,
                    size_bytes=ref.size_bytes,
                    populate=populate,
                ) as cached:
                    return await _stream_artifact_file(request, ref, cached)

            spool = Path(request.app[CONFIG_KEY].artifact_upload_spool_dir)
            spool.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=f'{artifact_id}.', suffix='.download', dir=spool)
            os.close(fd)
            temporary = Path(name)
            try:
                await _download_stored_artifact(request, stored_object, ref, temporary)
                return await _stream_artifact_file(request, ref, temporary)
            finally:
                await to_thread_complete(temporary.unlink, missing_ok=True)


async def _download_stored_artifact(
    request: web.Request,
    stored_object: Any,
    ref: ArtifactRef,
    destination: Path,
) -> None:
    try:
        await request.app[ARTIFACT_STORE_KEY].get(
            stored_object,
            destination,
            expected_size_bytes=ref.size_bytes,
            expected_sha256=ref.sha256,
        )
    except ValueError as exc:
        if str(exc) not in {'artifact_remote_size_mismatch', 'artifact_remote_sha256_mismatch'}:
            raise
        raise DomainConflictError('artifact_content_identity_mismatch') from exc


async def _stream_artifact_file(request: web.Request, ref: ArtifactRef, path: Path) -> web.StreamResponse:
    response = web.StreamResponse(
        status=200,
        headers={
            'Content-Type': 'application/octet-stream',
            'Content-Length': str(ref.size_bytes),
            'ETag': ref.sha256,
        },
    )
    await response.prepare(request)
    try:
        if request.method != 'HEAD':
            async with read_chunks(path) as chunks:
                async for chunk in chunks:
                    try:
                        await asyncio.wait_for(
                            response.write(chunk),
                            timeout=request.app[CONFIG_KEY].artifact_download_write_idle_timeout,
                        )
                    except asyncio.TimeoutError as exc:
                        response.force_close()
                        raise ConnectionResetError('artifact_download_write_idle_timeout') from exc
        await asyncio.wait_for(
            response.write_eof(), timeout=request.app[CONFIG_KEY].artifact_download_write_idle_timeout
        )
    except BaseException:
        response.force_close()
        if request.transport is not None:
            request.transport.close()
        raise
    return response
