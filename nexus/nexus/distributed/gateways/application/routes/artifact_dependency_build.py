from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile

from pathlib import Path
from typing import Any

from aiohttp import web
from pydantic import ValidationError

from nexus.distributed.artifacts.gateway import (
    NEXUS_WHEEL_PATH_HEADER,
    WHEELHOUSE_METADATA_HEADER,
    WHEELHOUSE_METADATA_SHA256_HEADER,
    WHEELHOUSE_METADATA_VERSION_HEADER,
)
from nexus.distributed.artifacts.store import StoredObject
from nexus.distributed.dependency_build.wheelhouse import WheelhouseMetadata
from nexus.distributed.environment.limits import effective_artifact_size_limit
from nexus.distributed.gateways.application.routes.artifact_shared import (
    _artifact_ref_payload,
    _artifact_upload_failure_code,
    _internal_artifact_idempotency_scope,
    _receiving_upload_lease,
    _release_receiving,
    _resolve_dependency_build_artifact_scope,
    _write_upload_body,
)
from nexus.distributed.gateways.common.keys import (
    ARTIFACT_STORE_KEY,
    ARTIFACT_UPLOAD_COORDINATOR_KEY,
    CONFIG_KEY,
    SESSION_FACTORY_KEY,
)
from nexus.distributed.persistence.db import run_idempotent_transaction
from nexus.distributed.persistence.orm import ArtifactRow, DependencyBuildRow, EnvironmentRow
from nexus.distributed.repositories.artifact import ArtifactRepository, CreateArtifactUpload, artifact_object_key
from nexus.distributed.repositories.dependency_build import DependencyBuildRepository
from nexus.distributed.utils.clock import utcnow
from nexus.distributed.utils.identity import canonical_sha256
from nexus.utils.common import to_thread_complete


_MAX_LEGACY_WHEELHOUSE_METADATA_HEADER_BYTES = 7000


async def stage_dependency_build_wheelhouse_metadata(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        metadata = WheelhouseMetadata.model_validate(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
        raise web.HTTPBadRequest(text='dependency_build_metadata_invalid') from exc

    async with request.app[SESSION_FACTORY_KEY]() as session:
        scope = await _resolve_dependency_build_artifact_scope(
            request,
            session,
            operation='wheelhouse:write',
        )
        if metadata.build_key != scope.build.build_key:
            raise web.HTTPBadRequest(text='dependency_build_metadata_invalid')
        try:
            await DependencyBuildRepository(session).stage_metadata(
                scope.build.dependency_build_id,
                attempt=scope.build.attempt,
                metadata=metadata,
                now=utcnow(),
            )
        except ValueError as exc:
            raise web.HTTPConflict(text=str(exc)) from exc
        await session.commit()
    return web.Response(status=204)


def _wheelhouse_metadata(request: web.Request, build: DependencyBuildRow) -> WheelhouseMetadata:
    metadata_version = request.headers.get(WHEELHOUSE_METADATA_VERSION_HEADER, '').strip()
    metadata_sha256 = request.headers.get(WHEELHOUSE_METADATA_SHA256_HEADER, '').strip()
    if metadata_version or metadata_sha256:
        if not metadata_version or not metadata_sha256 or build.manifest_json is None:
            raise ValueError('staged wheelhouse metadata is incomplete')
        metadata = WheelhouseMetadata.model_validate(build.manifest_json)
        if metadata_version != str(metadata.schema_version) or metadata_sha256 != canonical_sha256(
            metadata.model_dump(mode='json')
        ):
            raise ValueError('staged wheelhouse metadata identity mismatch')
        return metadata

    raw_metadata = request.headers.get(WHEELHOUSE_METADATA_HEADER, '')
    if raw_metadata:
        if len(raw_metadata.encode('utf-8')) > _MAX_LEGACY_WHEELHOUSE_METADATA_HEADER_BYTES:
            raise ValueError('wheelhouse metadata header too large')
        return WheelhouseMetadata.model_validate_json(raw_metadata)

    if build.build_spec_json.get('module_selectors'):
        raise ValueError('legacy wheelhouse metadata cannot represent module selectors')
    return WheelhouseMetadata(
        build_key=build.build_key,
        nexus_wheel_path=request.headers.get(NEXUS_WHEEL_PATH_HEADER, ''),
    )


async def _upload_dependency_wheelhouse(
    request: web.Request,
    *,
    sha256: str,
    size_bytes: int,
) -> web.Response:
    config = request.app[CONFIG_KEY]
    store = request.app[ARTIFACT_STORE_KEY]

    async def claim_upload(
        session: Any,
    ) -> tuple[Any, Any, str | None, Any | None, WheelhouseMetadata, StoredObject]:
        scope = await _resolve_dependency_build_artifact_scope(
            request,
            session,
            operation='wheelhouse:write',
        )
        try:
            metadata = _wheelhouse_metadata(request, scope.build)
            if metadata.build_key != scope.build.build_key:
                raise ValueError('wheelhouse metadata build key mismatch')
        except ValidationError as exc:
            raise web.HTTPBadRequest(text='dependency_build_metadata_invalid') from exc
        except ValueError as exc:
            raise web.HTTPBadRequest(text='dependency_build_metadata_invalid') from exc
        environment = await session.get(EnvironmentRow, scope.environment_id)
        max_size = effective_artifact_size_limit(
            environment.resource_limits_json if environment is not None else {},
            global_max_size_bytes=config.artifact_upload_max_size_bytes,
        )
        if size_bytes > max_size:
            raise web.HTTPRequestEntityTooLarge(max_size=max_size, actual_size=size_bytes)
        attempt = scope.build.attempt
        producer_key = f'dependency-wheelhouse:{scope.build.build_key}:attempt:{attempt}'
        repository = ArtifactRepository(session)
        upload, _ = await repository.create_upload(
            CreateArtifactUpload(
                owner_principal_id=scope.owner_principal_id,
                environment_id=scope.environment_id,
                owner_tenant=None,
                created_by='system:dependency-builder',
                kind='wheelhouse',
                sha256=sha256,
                size_bytes=size_bytes,
                idempotency_scope=_internal_artifact_idempotency_scope(
                    'dependency-build',
                    scope.build.dependency_build_id,
                    attempt,
                    'wheelhouse',
                ),
                idempotency_key=hashlib.sha256(producer_key.encode()).hexdigest(),
                session_ttl=config.artifact_upload_session_ttl,
                artifact_ttl=config.artifact_default_ttl,
                producer_key=producer_key,
            ),
            store=store,
            now=utcnow(),
        )
        artifact_row = await session.get(ArtifactRow, upload.artifact_id)
        if artifact_row is None:
            raise ValueError('artifact_not_found')
        stored_object = StoredObject(uri=artifact_row.uri, etag=artifact_row.provider_etag)
        upload, operation_id = await repository.claim_receiving(
            upload.upload_id,
            owner_principal_id=scope.owner_principal_id,
            environment_id=scope.environment_id,
            operation_owner='application-gateway:dependency-builder',
            lease_ttl=config.artifact_upload_lease_ttl,
            now=utcnow(),
        )
        if operation_id is None:
            artifact = await repository.get_artifact(
                upload.artifact_id,
                owner_principal_id=scope.owner_principal_id,
                environment_id=scope.environment_id,
            )
            return scope, upload, None, artifact, metadata, stored_object
        return scope, upload, operation_id, None, metadata, stored_object

    scope, upload, operation_id, ready_artifact, metadata, stored_object = await run_idempotent_transaction(
        request.app[SESSION_FACTORY_KEY],
        claim_upload,
        conflict_markers=(
            'uq_artifacts_owner_producer',
            'distributed_artifacts.owner_principal_id',
            'uq_artifact_uploads_idempotency',
            'distributed_artifact_uploads.idempotency_scope',
        ),
    )
    if operation_id is None:
        if scope.build.wheelhouse_ref_json != _artifact_ref_payload(
            ready_artifact
        ) or scope.build.manifest_json != metadata.model_dump(mode='json'):
            raise web.HTTPConflict(text='dependency_build_output_not_staged')
        return web.json_response({'ref': _artifact_ref_payload(ready_artifact)})

    spool = Path(config.artifact_upload_spool_dir)
    spool.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'{upload.upload_id}.', suffix='.wheelhouse', dir=spool)
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
                kind='wheelhouse',
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
            scope = await _resolve_dependency_build_artifact_scope(
                request,
                session,
                operation='wheelhouse:write',
            )
            repository = ArtifactRepository(session)
            ref = await repository.publish_ready(
                upload.upload_id,
                operation_id=operation_id,
                provider_etag=stored.etag,
                now=utcnow(),
            )
            await repository.add_ref(
                ref.artifact_id,
                owner_principal_id=scope.owner_principal_id,
                environment_id=scope.environment_id,
                resource_type='dependency_build',
                resource_id=scope.build.dependency_build_id,
                now=utcnow(),
            )
            await DependencyBuildRepository(session).stage_output(
                scope.build.dependency_build_id,
                attempt=scope.build.attempt,
                wheelhouse=ref,
                metadata=metadata,
                now=utcnow(),
            )
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


async def report_dependency_build_failure(request: web.Request) -> web.Response:
    if request.content_length is not None and request.content_length > 8 * 1024:
        raise web.HTTPRequestEntityTooLarge(max_size=8 * 1024, actual_size=request.content_length)
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise web.HTTPBadRequest(text='dependency_build_failure_invalid_json') from exc
    if not isinstance(payload, dict) or set(payload) != {'error_code', 'error_message'}:
        raise web.HTTPBadRequest(text='dependency_build_failure_invalid_fields')
    error_code = payload.get('error_code')
    error_message = payload.get('error_message')
    if not isinstance(error_code, str) or not error_code or len(error_code) > 128:
        raise web.HTTPBadRequest(text='dependency_build_failure_invalid_code')
    if not isinstance(error_message, str) or not error_message or len(error_message) > 2048:
        raise web.HTTPBadRequest(text='dependency_build_failure_invalid_message')

    async with request.app[SESSION_FACTORY_KEY]() as session:
        scope = await _resolve_dependency_build_artifact_scope(
            request,
            session,
            operation='dependency_build:report_failure',
        )
        try:
            await DependencyBuildRepository(session).stage_failure(
                scope.build.dependency_build_id,
                attempt=scope.build.attempt,
                error_code=error_code,
                error_message=error_message,
                now=utcnow(),
            )
        except ValueError as exc:
            raise web.HTTPConflict(text=str(exc)) from exc
        await session.commit()
    return web.Response(status=204)
