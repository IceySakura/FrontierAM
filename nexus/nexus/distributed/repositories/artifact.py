from __future__ import annotations

import hashlib

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import cast

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.distributed.artifacts.models import (
    ArtifactKind,
    ArtifactRecord,
    ArtifactRef,
    ArtifactUploadRecord,
)
from nexus.distributed.artifacts.store import ArtifactStore, StoredObject
from nexus.distributed.errors import (
    DomainConflictError,
    DomainValidationError,
    FencingError,
    ResourceGoneError,
    ResourceNotFoundError,
    TemporaryUnavailableError,
)
from nexus.distributed.persistence.orm import (
    ArtifactRefRow,
    ArtifactRow,
    ArtifactUploadRow,
    AssignmentRow,
    ExecutionRow,
)
from nexus.distributed.utils.identity import PRINCIPAL_ID_MAX_LENGTH, new_id, validate_environment_id


ARTIFACT_KINDS = {
    'source_package',
    'wheel',
    'nexus_wheel',
    'uv_lock',
    'wheelhouse',
    'task_list',
    'data_list',
    'run_result',
}
_CANONICAL_FILENAMES = {
    'source_package': 'source.tar.zst',
    'wheel': 'package.whl',
    'nexus_wheel': 'nexus-agent.whl',
    'uv_lock': 'uv.lock',
    'wheelhouse': 'wheelhouse.tar.zst',
    'task_list': 'tasks.jsonl',
    'data_list': 'data_list.bin',
    'run_result': 'result.json',
}


@dataclass(frozen=True, slots=True)
class CreateArtifactUpload:
    owner_principal_id: str
    environment_id: str
    owner_tenant: str | None
    created_by: str
    kind: str
    sha256: str
    size_bytes: int
    idempotency_scope: str
    idempotency_key: str
    session_ttl: timedelta
    artifact_ttl: timedelta
    producer_key: str | None = None


@dataclass(frozen=True, slots=True)
class UploadMaintenanceClaim:
    upload_id: str
    artifact_id: str
    kind: str
    owner_principal_id: str
    expected_sha256: str
    expected_size_bytes: int
    upload_expires_at: datetime
    previous_status: str
    stored_object: StoredObject
    operation_id: str


@dataclass(frozen=True, slots=True)
class ArtifactDeleteClaim:
    artifact_id: str
    owner_principal_id: str
    size_bytes: int
    stored_object: StoredObject
    operation_id: str


class ArtifactRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_upload(
        self,
        request: CreateArtifactUpload,
        *,
        store: ArtifactStore,
        now: datetime,
    ) -> tuple[ArtifactUploadRecord, ArtifactRecord]:
        if request.kind not in ARTIFACT_KINDS:
            raise DomainValidationError('invalid_artifact_kind')
        if not request.owner_principal_id or len(request.owner_principal_id) > PRINCIPAL_ID_MAX_LENGTH:
            raise DomainValidationError('artifact_scope_required')
        validate_environment_id(request.environment_id)
        if not request.idempotency_scope or len(request.idempotency_scope) > 128:
            raise DomainValidationError('invalid_artifact_idempotency_scope')
        if not request.idempotency_key or len(request.idempotency_key) > 128:
            raise DomainValidationError('invalid_artifact_idempotency_key')
        if request.producer_key is not None and (not request.producer_key or len(request.producer_key) > 256):
            raise DomainValidationError('invalid_artifact_producer_key')
        if len(request.sha256) != 64 or any(char not in '0123456789abcdef' for char in request.sha256):
            raise DomainValidationError('invalid_artifact_sha256')
        if request.size_bytes < 0:
            raise DomainValidationError('invalid_artifact_size')

        existing = await self.session.scalar(
            select(ArtifactUploadRow).where(
                ArtifactUploadRow.idempotency_scope == request.idempotency_scope,
                ArtifactUploadRow.idempotency_key == request.idempotency_key,
            )
        )
        if existing is not None:
            artifact = await self._require_artifact(existing.artifact_id)
            if request.producer_key is not None:
                self._validate_same_artifact(artifact, request)
            self._validate_same_upload(existing, request)
            self._validate_upload_artifact_binding(existing, artifact)
            return _upload_record(existing), _artifact_record(artifact)

        if request.producer_key is not None:
            produced = await self.session.scalar(
                select(ArtifactRow).where(
                    ArtifactRow.owner_principal_id == request.owner_principal_id,
                    ArtifactRow.kind == request.kind,
                    ArtifactRow.producer_key == request.producer_key,
                )
            )
            if produced is not None:
                self._validate_same_artifact(produced, request)
                upload = await self.session.scalar(
                    select(ArtifactUploadRow).where(ArtifactUploadRow.artifact_id == produced.artifact_id)
                )
                if upload is None:
                    raise DomainConflictError('artifact_producer_conflict')
                return _upload_record(upload), _artifact_record(produced)

        artifact_id = new_id('art')
        upload_id = new_id('upl')
        object_key = artifact_object_key(
            owner_principal_id=request.owner_principal_id,
            artifact_id=artifact_id,
            sha256=request.sha256,
            kind=request.kind,
        )
        artifact = ArtifactRow(
            artifact_id=artifact_id,
            owner_principal_id=request.owner_principal_id,
            environment_id=request.environment_id,
            owner_tenant=request.owner_tenant,
            kind=request.kind,
            producer_key=request.producer_key,
            uri=store.object_uri(object_key),
            sha256=request.sha256,
            size_bytes=request.size_bytes,
            status='uploading',
            created_by=request.created_by,
            expires_at=now + request.artifact_ttl,
            created_at=now,
            updated_at=now,
        )
        upload = ArtifactUploadRow(
            upload_id=upload_id,
            artifact_id=artifact_id,
            owner_principal_id=request.owner_principal_id,
            environment_id=request.environment_id,
            idempotency_scope=request.idempotency_scope,
            idempotency_key=request.idempotency_key,
            expected_kind=request.kind,
            expected_sha256=request.sha256,
            expected_size_bytes=request.size_bytes,
            status='pending',
            version=1,
            expires_at=now + request.session_ttl,
            created_at=now,
            updated_at=now,
        )
        self.session.add_all((artifact, upload))
        await self.session.flush()
        return _upload_record(upload), _artifact_record(artifact)

    async def get_upload(
        self,
        upload_id: str,
        *,
        owner_principal_id: str,
        environment_id: str,
    ) -> ArtifactUploadRecord:
        row = await self.session.get(ArtifactUploadRow, upload_id)
        if row is None or row.owner_principal_id != owner_principal_id or row.environment_id != environment_id:
            raise ResourceNotFoundError('artifact_upload_not_found')
        return _upload_record(row)

    async def get_artifact(
        self,
        artifact_id: str,
        *,
        owner_principal_id: str,
        environment_id: str,
    ) -> ArtifactRecord:
        row = await self.session.get(ArtifactRow, artifact_id)
        if row is None or row.owner_principal_id != owner_principal_id or row.environment_id != environment_id:
            raise ResourceNotFoundError('artifact_not_found')
        return _artifact_record(row)

    async def claim_receiving(
        self,
        upload_id: str,
        *,
        owner_principal_id: str,
        environment_id: str,
        operation_owner: str,
        lease_ttl: timedelta,
        now: datetime,
    ) -> tuple[ArtifactUploadRecord, str | None]:
        row = await self.session.scalar(
            select(ArtifactUploadRow).where(ArtifactUploadRow.upload_id == upload_id).with_for_update()
        )
        if row is None or row.owner_principal_id != owner_principal_id or row.environment_id != environment_id:
            raise ResourceNotFoundError('artifact_upload_not_found')
        if row.status == 'completed':
            return _upload_record(row), None
        if row.expires_at <= now:
            raise ResourceGoneError('artifact_upload_expired')
        if row.status == 'receiving' and row.operation_expires_at is not None and row.operation_expires_at > now:
            raise TemporaryUnavailableError('artifact_upload_busy')
        if row.status not in {'pending', 'receiving'}:
            raise DomainConflictError('artifact_upload_not_retryable')
        operation_id = new_id('op')
        row.status = 'receiving'
        row.operation_id = operation_id
        row.operation_owner = operation_owner
        row.operation_expires_at = now + lease_ttl
        row.version += 1
        row.updated_at = now
        await self.session.flush()
        return _upload_record(row), operation_id

    async def publish_ready(
        self,
        upload_id: str,
        *,
        operation_id: str,
        provider_etag: str | None,
        now: datetime,
    ) -> ArtifactRef:
        upload = await self.session.scalar(
            select(ArtifactUploadRow).where(ArtifactUploadRow.upload_id == upload_id).with_for_update()
        )
        if upload is None:
            raise ResourceNotFoundError('artifact_upload_not_found')
        artifact = await self.session.scalar(
            select(ArtifactRow).where(ArtifactRow.artifact_id == upload.artifact_id).with_for_update()
        )
        if artifact is None:
            raise ResourceNotFoundError('artifact_not_found')
        self._validate_upload_artifact_binding(upload, artifact)
        if upload.status == 'completed' and artifact.status == 'ready':
            return _artifact_ref(artifact)
        if upload.status != 'receiving' or upload.operation_id != operation_id:
            raise FencingError('artifact_upload_lease_lost')

        upload.status = 'completed'
        upload.provider_etag = provider_etag
        upload.operation_id = None
        upload.operation_owner = None
        upload.operation_expires_at = None
        upload.version += 1
        upload.updated_at = now
        artifact.status = 'ready'
        artifact.provider_etag = provider_etag
        artifact.updated_at = now
        await self.session.flush()
        return _artifact_ref(artifact)

    async def renew_receiving(
        self,
        upload_id: str,
        *,
        operation_id: str,
        lease_ttl: timedelta,
        now: datetime,
    ) -> bool:
        if lease_ttl <= timedelta(0):
            raise DomainValidationError('artifact_upload_lease_ttl_invalid')
        upload = await self.session.scalar(
            select(ArtifactUploadRow).where(ArtifactUploadRow.upload_id == upload_id).with_for_update()
        )
        if (
            upload is None
            or upload.status != 'receiving'
            or upload.operation_id != operation_id
            or upload.operation_expires_at is None
            or upload.operation_expires_at <= now
        ):
            return False
        upload.operation_expires_at = now + lease_ttl
        upload.version += 1
        upload.updated_at = now
        await self.session.flush()
        return True

    async def publish_run_result_ready(
        self,
        upload_id: str,
        *,
        operation_id: str,
        execution_id: str,
        assignment_id: str,
        attempt_id: str,
        provider_etag: str | None,
        now: datetime,
    ) -> ArtifactRef:
        ref = await self.publish_ready(
            upload_id,
            operation_id=operation_id,
            provider_etag=provider_etag,
            now=now,
        )
        await self.link_run_result(
            ref.artifact_id,
            execution_id=execution_id,
            assignment_id=assignment_id,
            attempt_id=attempt_id,
            now=now,
        )
        return ref

    async def link_run_result(
        self,
        artifact_id: str,
        *,
        execution_id: str,
        assignment_id: str,
        attempt_id: str,
        now: datetime,
    ) -> None:
        execution = await self.session.scalar(
            select(ExecutionRow).where(ExecutionRow.execution_id == execution_id).with_for_update()
        )
        assignment = await self.session.scalar(
            select(AssignmentRow).where(AssignmentRow.assignment_id == assignment_id).with_for_update()
        )
        artifact = await self.session.scalar(
            select(ArtifactRow).where(ArtifactRow.artifact_id == artifact_id).with_for_update()
        )
        if execution is None or assignment is None or artifact is None:
            raise ResourceNotFoundError('artifact_not_found')
        if (
            execution.current_assignment_id != assignment_id
            or execution.current_attempt_id != attempt_id
            or assignment.execution_id != execution_id
            or assignment.attempt_id != attempt_id
            or assignment.owner_principal_id != execution.owner_principal_id
            or assignment.environment_id != execution.environment_id
            or assignment.allocation_id != execution.allocation_id
        ):
            raise FencingError('artifact_execution_scope_mismatch')
        expected_producer_key = f'run-result:{execution_id}:{attempt_id}'
        if (
            artifact.status != 'ready'
            or artifact.kind != 'run_result'
            or artifact.owner_principal_id != execution.owner_principal_id
            or artifact.environment_id != execution.environment_id
            or artifact.producer_key != expected_producer_key
        ):
            raise FencingError('artifact_execution_scope_mismatch')
        if execution.result_artifact_id not in {None, artifact_id}:
            raise DomainConflictError('execution_result_artifact_conflict')
        if execution.result_artifact_id is None:
            execution.result_artifact_id = artifact_id
            execution.version += 1
            execution.updated_at = now
        owner_principal_id = execution.owner_principal_id
        environment_id = execution.environment_id
        assert owner_principal_id is not None
        assert environment_id is not None
        await self.add_ref(
            artifact_id,
            owner_principal_id=owner_principal_id,
            environment_id=environment_id,
            resource_type='execution',
            resource_id=execution_id,
            now=now,
        )
        await self.session.flush()

    async def release_receiving(
        self,
        upload_id: str,
        *,
        operation_id: str,
        failure_code: str | None = None,
        now: datetime,
    ) -> bool:
        values = {
            'status': 'pending',
            'operation_id': None,
            'operation_owner': None,
            'operation_expires_at': None,
            'version': ArtifactUploadRow.version + 1,
            'updated_at': now,
        }
        if failure_code:
            values['failure_count'] = ArtifactUploadRow.failure_count + 1
            values['last_failure_code'] = failure_code[:128]
        result = await self.session.execute(
            update(ArtifactUploadRow)
            .where(
                ArtifactUploadRow.upload_id == upload_id,
                ArtifactUploadRow.status == 'receiving',
                ArtifactUploadRow.operation_id == operation_id,
            )
            .values(**values)
        )
        return int(getattr(result, 'rowcount', 0) or 0) == 1

    async def resolve_ready_ref(
        self,
        ref: ArtifactRef,
        *,
        owner_principal_id: str,
        environment_id: str,
    ) -> ArtifactRow:
        artifact = await self.session.get(ArtifactRow, ref.artifact_id)
        if (
            artifact is None
            or artifact.owner_principal_id != owner_principal_id
            or artifact.environment_id != environment_id
        ):
            raise ResourceNotFoundError('artifact_not_found')
        if artifact.status != 'ready':
            raise DomainConflictError('artifact_not_ready')
        if artifact.kind != ref.kind or artifact.sha256 != ref.sha256 or artifact.size_bytes != ref.size_bytes:
            raise DomainConflictError('artifact_ref_mismatch')
        return artifact

    async def resolve_stored_object(
        self,
        ref: ArtifactRef,
        *,
        owner_principal_id: str,
        environment_id: str,
    ) -> StoredObject:
        artifact = await self.resolve_ready_ref(
            ref,
            owner_principal_id=owner_principal_id,
            environment_id=environment_id,
        )
        return StoredObject(uri=artifact.uri, etag=artifact.provider_etag)

    async def add_ref(
        self,
        artifact_id: str,
        *,
        owner_principal_id: str,
        environment_id: str,
        resource_type: str,
        resource_id: str,
        now: datetime,
    ) -> None:
        key = {
            'artifact_id': artifact_id,
            'resource_type': resource_type,
            'resource_id': resource_id,
        }
        artifact = await self.session.scalar(
            select(ArtifactRow).where(ArtifactRow.artifact_id == artifact_id).with_for_update()
        )
        if (
            artifact is None
            or artifact.owner_principal_id != owner_principal_id
            or artifact.environment_id != environment_id
        ):
            raise ResourceNotFoundError('artifact_not_found')
        if artifact.status != 'ready':
            raise DomainConflictError('artifact_not_ready')
        existing_ref = await self.session.scalar(
            select(ArtifactRefRow)
            .where(
                ArtifactRefRow.artifact_id == artifact_id,
                ArtifactRefRow.resource_type == resource_type,
                ArtifactRefRow.resource_id == resource_id,
            )
            .with_for_update()
        )
        if existing_ref is None:
            self.session.add(ArtifactRefRow(**key, created_at=now))
            await self.session.flush()

    async def claim_upload_maintenance(
        self,
        *,
        operation_owner: str,
        lease_ttl: timedelta,
        limit: int,
        now: datetime,
    ) -> list[UploadMaintenanceClaim]:
        rows = (
            await self.session.scalars(
                select(ArtifactUploadRow)
                .where(
                    or_(
                        (ArtifactUploadRow.status == 'pending') & (ArtifactUploadRow.expires_at <= now),
                        (ArtifactUploadRow.status == 'receiving')
                        & or_(
                            ArtifactUploadRow.operation_expires_at.is_(None),
                            ArtifactUploadRow.operation_expires_at <= now,
                        ),
                        (ArtifactUploadRow.status == 'maintenance')
                        & or_(
                            ArtifactUploadRow.operation_expires_at.is_(None),
                            ArtifactUploadRow.operation_expires_at <= now,
                        ),
                    )
                )
                .order_by(ArtifactUploadRow.updated_at, ArtifactUploadRow.upload_id)
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
        ).all()
        claims: list[UploadMaintenanceClaim] = []
        for row in rows:
            artifact = await self.session.get(ArtifactRow, row.artifact_id)
            if artifact is None:
                continue
            previous_status = 'receiving' if row.status == 'maintenance' else row.status
            operation_id = new_id('op')
            row.status = 'maintenance'
            row.operation_id = operation_id
            row.operation_owner = operation_owner
            row.operation_expires_at = now + lease_ttl
            row.version += 1
            row.updated_at = now
            claims.append(
                UploadMaintenanceClaim(
                    upload_id=row.upload_id,
                    artifact_id=row.artifact_id,
                    kind=artifact.kind,
                    owner_principal_id=row.owner_principal_id,
                    expected_sha256=row.expected_sha256,
                    expected_size_bytes=row.expected_size_bytes,
                    upload_expires_at=row.expires_at,
                    previous_status=previous_status,
                    stored_object=StoredObject(uri=artifact.uri, etag=artifact.provider_etag),
                    operation_id=operation_id,
                )
            )
        await self.session.flush()
        return claims

    async def renew_upload_maintenance(
        self,
        claim: UploadMaintenanceClaim,
        *,
        lease_ttl: timedelta,
        now: datetime,
    ) -> bool:
        if lease_ttl <= timedelta(0):
            raise DomainValidationError('artifact_maintenance_lease_ttl_invalid')
        upload = await self.session.scalar(
            select(ArtifactUploadRow).where(ArtifactUploadRow.upload_id == claim.upload_id).with_for_update()
        )
        if (
            upload is None
            or upload.status != 'maintenance'
            or upload.operation_id != claim.operation_id
            or upload.operation_expires_at is None
            or upload.operation_expires_at <= now
        ):
            return False
        upload.operation_expires_at = now + lease_ttl
        upload.version += 1
        upload.updated_at = now
        await self.session.flush()
        return True

    async def finish_recovered_upload(
        self,
        claim: UploadMaintenanceClaim,
        *,
        provider_etag: str | None,
        now: datetime,
    ) -> bool:
        upload, artifact = await self._locked_maintenance_rows(claim)
        if upload is None or artifact is None:
            return False
        upload.status = 'completed'
        upload.provider_etag = provider_etag
        self._clear_upload_operation(upload)
        upload.version += 1
        upload.updated_at = now
        artifact.status = 'ready'
        artifact.provider_etag = provider_etag
        artifact.updated_at = now
        await self.session.flush()
        return True

    async def finish_abandoned_upload(self, claim: UploadMaintenanceClaim, *, now: datetime) -> bool:
        upload, artifact = await self._locked_maintenance_rows(claim)
        if upload is None or artifact is None:
            return False
        upload.status = 'expired'
        self._clear_upload_operation(upload)
        upload.version += 1
        upload.updated_at = now
        artifact.status = 'deleted'
        artifact.operation_id = None
        artifact.operation_owner = None
        artifact.operation_expires_at = None
        artifact.updated_at = now
        await self.session.flush()
        return True

    async def retry_upload_maintenance(
        self,
        claim: UploadMaintenanceClaim,
        *,
        retry_at: datetime,
        now: datetime,
    ) -> bool:
        upload = await self.session.scalar(
            select(ArtifactUploadRow).where(ArtifactUploadRow.upload_id == claim.upload_id).with_for_update()
        )
        if upload is None or upload.status != 'maintenance' or upload.operation_id != claim.operation_id:
            return False
        upload.status = 'receiving'
        upload.operation_expires_at = retry_at
        upload.version += 1
        upload.updated_at = now
        await self.session.flush()
        return True

    async def reset_upload_pending(self, claim: UploadMaintenanceClaim, *, now: datetime) -> bool:
        upload = await self.session.scalar(
            select(ArtifactUploadRow).where(ArtifactUploadRow.upload_id == claim.upload_id).with_for_update()
        )
        if upload is None or upload.status != 'maintenance' or upload.operation_id != claim.operation_id:
            return False
        upload.status = 'pending'
        self._clear_upload_operation(upload)
        upload.version += 1
        upload.updated_at = now
        await self.session.flush()
        return True

    async def claim_expired_artifacts(
        self,
        *,
        operation_owner: str,
        lease_ttl: timedelta,
        limit: int,
        now: datetime,
    ) -> list[ArtifactDeleteClaim]:
        has_ref = (
            select(ArtifactRefRow.artifact_id).where(ArtifactRefRow.artifact_id == ArtifactRow.artifact_id).exists()
        )
        rows = (
            await self.session.scalars(
                select(ArtifactRow)
                .where(
                    ~has_ref,
                    or_(
                        (ArtifactRow.status == 'ready') & (ArtifactRow.expires_at <= now),
                        (ArtifactRow.status == 'deleting')
                        & or_(
                            ArtifactRow.operation_expires_at.is_(None),
                            ArtifactRow.operation_expires_at <= now,
                        ),
                    ),
                )
                .order_by(ArtifactRow.updated_at, ArtifactRow.artifact_id)
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
        ).all()
        claims = []
        for row in rows:
            operation_id = new_id('op')
            row.status = 'deleting'
            row.operation_id = operation_id
            row.operation_owner = operation_owner
            row.operation_expires_at = now + lease_ttl
            row.updated_at = now
            claims.append(
                ArtifactDeleteClaim(
                    artifact_id=row.artifact_id,
                    owner_principal_id=row.owner_principal_id,
                    size_bytes=row.size_bytes,
                    stored_object=StoredObject(uri=row.uri, etag=row.provider_etag),
                    operation_id=operation_id,
                )
            )
        await self.session.flush()
        return claims

    async def renew_artifact_delete(
        self,
        claim: ArtifactDeleteClaim,
        *,
        lease_ttl: timedelta,
        now: datetime,
    ) -> bool:
        if lease_ttl <= timedelta(0):
            raise DomainValidationError('artifact_maintenance_lease_ttl_invalid')
        row = await self.session.scalar(
            select(ArtifactRow).where(ArtifactRow.artifact_id == claim.artifact_id).with_for_update()
        )
        if (
            row is None
            or row.status != 'deleting'
            or row.operation_id != claim.operation_id
            or row.operation_expires_at is None
            or row.operation_expires_at <= now
        ):
            return False
        row.operation_expires_at = now + lease_ttl
        row.updated_at = now
        await self.session.flush()
        return True

    async def finish_artifact_delete(self, claim: ArtifactDeleteClaim, *, now: datetime) -> bool:
        row = await self.session.scalar(
            select(ArtifactRow).where(ArtifactRow.artifact_id == claim.artifact_id).with_for_update()
        )
        if row is None or row.status != 'deleting' or row.operation_id != claim.operation_id:
            return False
        row.status = 'deleted'
        row.operation_id = None
        row.operation_owner = None
        row.operation_expires_at = None
        row.updated_at = now
        await self.session.flush()
        return True

    async def retry_artifact_delete(
        self,
        claim: ArtifactDeleteClaim,
        *,
        retry_at: datetime,
        now: datetime,
    ) -> bool:
        row = await self.session.scalar(
            select(ArtifactRow).where(ArtifactRow.artifact_id == claim.artifact_id).with_for_update()
        )
        if row is None or row.status != 'deleting' or row.operation_id != claim.operation_id:
            return False
        row.status = 'ready'
        row.operation_id = None
        row.operation_owner = None
        row.operation_expires_at = None
        row.failure_count += 1
        row.expires_at = retry_at
        row.updated_at = now
        await self.session.flush()
        return True

    async def _locked_maintenance_rows(
        self, claim: UploadMaintenanceClaim
    ) -> tuple[ArtifactUploadRow | None, ArtifactRow | None]:
        upload = await self.session.scalar(
            select(ArtifactUploadRow).where(ArtifactUploadRow.upload_id == claim.upload_id).with_for_update()
        )
        if upload is None or upload.status != 'maintenance' or upload.operation_id != claim.operation_id:
            return None, None
        artifact = await self.session.scalar(
            select(ArtifactRow).where(ArtifactRow.artifact_id == claim.artifact_id).with_for_update()
        )
        return upload, artifact

    @staticmethod
    def _clear_upload_operation(upload: ArtifactUploadRow) -> None:
        upload.operation_id = None
        upload.operation_owner = None
        upload.operation_expires_at = None

    async def _require_artifact(self, artifact_id: str) -> ArtifactRow:
        artifact = await self.session.get(ArtifactRow, artifact_id)
        if artifact is None:
            raise ResourceNotFoundError('artifact_not_found')
        return artifact

    @staticmethod
    def _validate_same_upload(row: ArtifactUploadRow, request: CreateArtifactUpload) -> None:
        if (
            row.owner_principal_id != request.owner_principal_id
            or row.environment_id != request.environment_id
            or row.expected_kind != request.kind
            or row.expected_sha256 != request.sha256
            or row.expected_size_bytes != request.size_bytes
        ):
            raise DomainConflictError('artifact_upload_idempotency_conflict')

    @staticmethod
    def _validate_same_artifact(row: ArtifactRow, request: CreateArtifactUpload) -> None:
        if (
            row.owner_principal_id != request.owner_principal_id
            or row.environment_id != request.environment_id
            or row.kind != request.kind
            or row.producer_key != request.producer_key
            or row.sha256 != request.sha256
            or row.size_bytes != request.size_bytes
        ):
            raise DomainConflictError('artifact_producer_conflict')

    @staticmethod
    def _validate_upload_artifact_binding(upload: ArtifactUploadRow, artifact: ArtifactRow) -> None:
        if (
            upload.artifact_id != artifact.artifact_id
            or upload.owner_principal_id != artifact.owner_principal_id
            or upload.environment_id != artifact.environment_id
            or upload.expected_kind != artifact.kind
            or upload.expected_sha256 != artifact.sha256
            or upload.expected_size_bytes != artifact.size_bytes
        ):
            raise FencingError('artifact_upload_binding_mismatch')


def artifact_object_key(*, owner_principal_id: str, artifact_id: str, sha256: str, kind: str) -> str:
    owner_hash = hashlib.sha256(owner_principal_id.encode()).hexdigest()[:24]
    return f'{owner_hash}/{artifact_id}/{sha256}/{_CANONICAL_FILENAMES[kind]}'


def _artifact_ref(row: ArtifactRow) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=row.artifact_id,
        sha256=row.sha256,
        size_bytes=row.size_bytes,
        kind=cast(ArtifactKind, row.kind),
    )


def _artifact_record(row: ArtifactRow) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=row.artifact_id,
        owner_principal_id=row.owner_principal_id,
        kind=row.kind,
        sha256=row.sha256,
        size_bytes=row.size_bytes,
        status=row.status,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _upload_record(row: ArtifactUploadRow) -> ArtifactUploadRecord:
    return ArtifactUploadRecord(
        upload_id=row.upload_id,
        artifact_id=row.artifact_id,
        owner_principal_id=row.owner_principal_id,
        expected_kind=row.expected_kind,
        expected_sha256=row.expected_sha256,
        expected_size_bytes=row.expected_size_bytes,
        status=row.status,
        failure_count=row.failure_count,
        last_failure_code=row.last_failure_code,
        version=row.version,
        expires_at=row.expires_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
