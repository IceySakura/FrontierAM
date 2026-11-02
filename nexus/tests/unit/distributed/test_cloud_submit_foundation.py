from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import zipfile

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.access.taihu import principal_id
from nexus.distributed.artifacts.maintenance import ArtifactMaintenanceReconciler
from nexus.distributed.artifacts.store import (
    CredentialFiles,
    GenericRepositoryArtifactStore,
    LocalArtifactStore,
)
from nexus.distributed.artifacts.tickets import ArtifactContentTicketSigner, UploadTicketSigner
from nexus.distributed.config import DistributedConfig
from nexus.distributed.gateways.application.app import create_application_app
from nexus.distributed.gateways.control.app import create_control_app
from nexus.distributed.persistence.orm import (
    ArtifactQuotaUsageRow,
    ArtifactRow,
    ArtifactUploadRow,
    EnvironmentAllocationRow,
    ExecutionRow,
    create_schema,
)
from nexus.distributed.repositories.artifact import ArtifactRepository, CreateArtifactUpload, artifact_object_key
from nexus.distributed.repositories.environment import EnvironmentRepository
from nexus.distributed.repositories.provider_config import ProviderConfigRepository
from nexus.distributed.submission.source_package import PackageLimits, build_source_package, safe_extract_source_package
from nexus.distributed.submission.specs import SingleSubmission
from nexus.distributed.utils.clock import utcnow


# Upper bound for "this must not hang" waits. The guarded steps cost tens of
# milliseconds; the margin absorbs a loaded CI box without ever masking a real hang.
HANG_GUARD_SECONDS = 30


def _tai_identity_header(*, staff_id: int, application_token: str) -> dict[str, str]:
    protected = base64.urlsafe_b64encode(b'{"alg":"dir","enc":"A256GCM"}').rstrip(b'=')
    payload = json.dumps(
        {
            'StaffId': staff_id,
            'LoginName': f'user-{staff_id}',
            'Expiration': (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        },
        separators=(',', ':'),
    ).encode()
    iv = os.urandom(12)
    encrypted = AESGCM(application_token.encode()).encrypt(iv, payload, protected)

    def encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).rstrip(b'=').decode()

    return {
        'x-tai-identity': '.'.join((
            protected.decode(),
            '',
            encode(iv),
            encode(encrypted[:-16]),
            encode(encrypted[-16:]),
        ))
    }


@pytest.fixture
async def cloud_session_factory(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "cloud.db"}')
    await create_schema(engine)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


async def test_local_artifact_repository_upload_is_idempotent_and_fenced(cloud_session_factory, tmp_path):
    store = LocalArtifactStore(tmp_path / 'store')
    content = tmp_path / 'source.tar.zst'
    content.write_bytes(b'cloud-submit')
    sha256 = hashlib.sha256(content.read_bytes()).hexdigest()
    request = CreateArtifactUpload(
        owner_principal_id='owner-1',
        environment_id='env-1',
        owner_tenant='tenant-1',
        created_by='owner-1',
        kind='source_package',
        sha256=sha256,
        size_bytes=content.stat().st_size,
        idempotency_scope='scope-1',
        idempotency_key='key-1',
        session_ttl=timedelta(hours=1),
        artifact_ttl=timedelta(days=1),
    )
    async with cloud_session_factory() as session:
        repo = ArtifactRepository(session)
        upload, artifact = await repo.create_upload(request, store=store, now=utcnow())
        repeated, _ = await repo.create_upload(request, store=store, now=utcnow())
        assert repeated.upload_id == upload.upload_id
        claimed, operation_id = await repo.claim_receiving(
            upload.upload_id,
            owner_principal_id='owner-1',
            environment_id='env-1',
            operation_owner='test',
            lease_ttl=timedelta(minutes=1),
            now=utcnow(),
        )
        assert claimed.status == 'receiving'
        assert operation_id
        renewed_at = utcnow() + timedelta(seconds=10)
        assert await repo.renew_receiving(
            upload.upload_id,
            operation_id=operation_id,
            lease_ttl=timedelta(minutes=1),
            now=renewed_at,
        )
        renewed_row = await session.get(ArtifactUploadRow, upload.upload_id)
        assert renewed_row is not None
        assert renewed_row.operation_expires_at == renewed_at + timedelta(minutes=1)
        assert not await repo.renew_receiving(
            upload.upload_id,
            operation_id='op-stale',
            lease_ttl=timedelta(minutes=1),
            now=renewed_at,
        )
        assert await repo.release_receiving(
            upload.upload_id,
            operation_id=operation_id,
            failure_code='artifact_content_identity_mismatch',
            now=utcnow(),
        )
        failed_upload = await session.get(ArtifactUploadRow, upload.upload_id)
        assert failed_upload is not None
        assert failed_upload.failure_count == 1
        assert failed_upload.last_failure_code == 'artifact_content_identity_mismatch'
        _, replacement_operation_id = await repo.claim_receiving(
            upload.upload_id,
            owner_principal_id='owner-1',
            environment_id='env-1',
            operation_owner='test-replacement',
            lease_ttl=timedelta(minutes=1),
            now=utcnow(),
        )
        with pytest.raises(ValueError, match='lease_lost'):
            await repo.publish_ready(
                upload.upload_id,
                operation_id=operation_id,
                provider_etag=None,
                now=utcnow(),
            )
        key = artifact_object_key(
            owner_principal_id='owner-1',
            artifact_id=artifact.artifact_id,
            sha256=sha256,
            kind='source_package',
        )
        stored = await store.put(content, object_key=key, sha256=sha256, size_bytes=content.stat().st_size)
        ref = await repo.publish_ready(
            upload.upload_id,
            operation_id=replacement_operation_id,
            provider_etag=stored.etag,
            now=utcnow(),
        )
        assert ref.kind == 'source_package'
        await repo.resolve_ready_ref(ref, owner_principal_id='owner-1', environment_id='env-1')
        await repo.add_ref(
            ref.artifact_id,
            owner_principal_id='owner-1',
            environment_id='env-1',
            resource_type='execution',
            resource_id='exe-1',
            now=utcnow(),
        )
        with pytest.raises(KeyError):
            await repo.resolve_ready_ref(ref, owner_principal_id='owner-1', environment_id='env-2')
        with pytest.raises(KeyError):
            await repo.add_ref(
                ref.artifact_id,
                owner_principal_id='owner-1',
                environment_id='env-2',
                resource_type='execution',
                resource_id='exe-2',
                now=utcnow(),
            )
        await session.commit()


async def test_data_list_artifact_upload_uses_canonical_binary_filename(cloud_session_factory, tmp_path):
    store = LocalArtifactStore(tmp_path / 'store')
    content = b'NXDL-data-list'
    sha256 = hashlib.sha256(content).hexdigest()
    request = CreateArtifactUpload(
        owner_principal_id='owner-data-list',
        environment_id='env-data-list',
        owner_tenant='tenant',
        created_by='owner-data-list',
        kind='data_list',
        sha256=sha256,
        size_bytes=len(content),
        idempotency_scope='scope-data-list',
        idempotency_key='key-data-list',
        session_ttl=timedelta(hours=1),
        artifact_ttl=timedelta(days=1),
    )

    async with cloud_session_factory() as session:
        upload, artifact = await ArtifactRepository(session).create_upload(request, store=store, now=utcnow())

    assert upload.expected_kind == 'data_list'
    assert artifact.kind == 'data_list'
    assert artifact_object_key(
        owner_principal_id='owner-data-list',
        artifact_id=artifact.artifact_id,
        sha256=sha256,
        kind='data_list',
    ).endswith('/data_list.bin')


@pytest.mark.parametrize(
    ('field', 'value', 'error'),
    [
        ('idempotency_scope', 's' * 129, 'invalid_artifact_idempotency_scope'),
        ('idempotency_key', 'k' * 129, 'invalid_artifact_idempotency_key'),
    ],
)
async def test_artifact_repository_rejects_idempotency_fields_longer_than_database_contract(
    cloud_session_factory,
    tmp_path,
    field,
    value,
    error,
):
    request = CreateArtifactUpload(
        owner_principal_id='owner-1',
        environment_id='env-1',
        owner_tenant='tenant-1',
        created_by='owner-1',
        kind='source_package',
        sha256='a' * 64,
        size_bytes=1,
        idempotency_scope='scope-1',
        idempotency_key='key-1',
        session_ttl=timedelta(hours=1),
        artifact_ttl=timedelta(days=1),
    )

    async with cloud_session_factory() as session:
        with pytest.raises(ValueError, match=error):
            await ArtifactRepository(session).create_upload(
                replace(request, **{field: value}),
                store=LocalArtifactStore(tmp_path / 'invalid-artifact-store'),
                now=utcnow(),
            )


async def test_external_artifact_upload_idle_timeout_releases_receiving_claim(cloud_session_factory, tmp_path):
    from nexus.distributed.artifacts.tickets import UPLOAD_TICKET_HEADER

    content = b'slow-upload'
    sha256 = hashlib.sha256(content).hexdigest()
    config = DistributedConfig(
        artifact_local_root=str(tmp_path / 'slow-upload-store'),
        artifact_upload_spool_dir=str(tmp_path / 'slow-upload-spool'),
        artifact_upload_body_idle_timeout=0.05,
        artifact_upload_deadline=timedelta(seconds=10),
        # Lease renewal has its own tests; do not race a SQLite write against this timeout.
        artifact_upload_lease_ttl=timedelta(minutes=5),
        artifact_upload_lease_renew_interval=timedelta(minutes=1),
    )
    store = LocalArtifactStore(tmp_path / 'slow-upload-store')
    async with cloud_session_factory() as session:
        upload, artifact = await ArtifactRepository(session).create_upload(
            CreateArtifactUpload(
                owner_principal_id='owner-slow-upload',
                environment_id='env-slow-upload',
                owner_tenant='tenant',
                created_by='owner-slow-upload',
                kind='source_package',
                sha256=sha256,
                size_bytes=len(content),
                idempotency_scope='slow-upload',
                idempotency_key='upload',
                session_ttl=timedelta(minutes=5),
                artifact_ttl=timedelta(days=1),
            ),
            store=store,
            now=utcnow(),
        )
        await session.commit()
    ticket = UploadTicketSigner(
        keys=config.artifact_ticket_keyring(),
        current_key_id=config.artifact_upload_ticket_current_key_id,
        ttl=config.artifact_upload_ticket_ttl,
    ).issue(
        upload_id=upload.upload_id,
        artifact_id=artifact.artifact_id,
        owner_principal_id='owner-slow-upload',
        environment_id='env-slow-upload',
        expected_kind='source_package',
        expected_size_bytes=len(content),
        expected_sha256=sha256,
        now=utcnow(),
    )

    release_body = asyncio.Event()

    async def slow_body():
        yield content[:1]
        await release_body.wait()
        yield content[1:]

    client = TestClient(TestServer(create_application_app(session_factory=cloud_session_factory, config=config)))
    await client.start_server()
    try:
        response = await client.put(
            f'/v1/artifact-uploads/{upload.upload_id}/content',
            data=slow_body(),
            headers={
                UPLOAD_TICKET_HEADER: ticket,
                'Content-Length': str(len(content)),
            },
        )
        assert response.status == 408
        assert await response.json() == {
            'error': 'artifact_upload_body_idle_timeout',
            'retryable': True,
        }
    finally:
        release_body.set()
        await client.close()

    async with cloud_session_factory() as session:
        row = await session.get(ArtifactUploadRow, upload.upload_id)
        assert row is not None
        assert row.status == 'pending'
        assert row.operation_id is None


async def test_external_artifact_upload_deadline_releases_receiving_claim(
    cloud_session_factory,
    tmp_path,
    monkeypatch,
):
    from nexus.distributed.artifacts.tickets import UPLOAD_TICKET_HEADER
    from nexus.distributed.gateways.application.app import ARTIFACT_STORE_KEY

    content = b'deadline-upload'
    sha256 = hashlib.sha256(content).hexdigest()
    config = DistributedConfig(
        artifact_local_root=str(tmp_path / 'deadline-upload-store'),
        artifact_upload_spool_dir=str(tmp_path / 'deadline-upload-spool'),
        artifact_upload_body_idle_timeout=0.05,
        artifact_upload_deadline=timedelta(milliseconds=150),
        artifact_upload_lease_ttl=timedelta(milliseconds=300),
        artifact_upload_lease_renew_interval=timedelta(milliseconds=50),
    )
    store = LocalArtifactStore(tmp_path / 'deadline-upload-store')
    async with cloud_session_factory() as session:
        upload, artifact = await ArtifactRepository(session).create_upload(
            CreateArtifactUpload(
                owner_principal_id='owner-deadline-upload',
                environment_id='env-deadline-upload',
                owner_tenant='tenant',
                created_by='owner-deadline-upload',
                kind='source_package',
                sha256=sha256,
                size_bytes=len(content),
                idempotency_scope='deadline-upload',
                idempotency_key='upload',
                session_ttl=timedelta(minutes=5),
                artifact_ttl=timedelta(days=1),
            ),
            store=store,
            now=utcnow(),
        )
        await session.commit()
    ticket = UploadTicketSigner(
        keys=config.artifact_ticket_keyring(),
        current_key_id=config.artifact_upload_ticket_current_key_id,
        ttl=config.artifact_upload_ticket_ttl,
    ).issue(
        upload_id=upload.upload_id,
        artifact_id=artifact.artifact_id,
        owner_principal_id='owner-deadline-upload',
        environment_id='env-deadline-upload',
        expected_kind='source_package',
        expected_size_bytes=len(content),
        expected_sha256=sha256,
        now=utcnow(),
    )
    app = create_application_app(session_factory=cloud_session_factory, config=config)

    async def blocked_put(*args, **kwargs):
        await asyncio.sleep(1)
        raise AssertionError('upload deadline did not cancel Store PUT')

    monkeypatch.setattr(app[ARTIFACT_STORE_KEY], 'put', blocked_put)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.put(
            f'/v1/artifact-uploads/{upload.upload_id}/content',
            data=content,
            headers={UPLOAD_TICKET_HEADER: ticket},
        )
        assert response.status == 504
        assert await response.json() == {
            'error': 'artifact_upload_deadline_exceeded',
            'retryable': True,
        }
    finally:
        await client.close()

    async with cloud_session_factory() as session:
        row = await session.get(ArtifactUploadRow, upload.upload_id)
        assert row is not None
        assert row.status == 'pending'
        assert row.operation_id is None


async def test_external_artifact_upload_lease_loss_stops_before_store_publish(
    cloud_session_factory,
    tmp_path,
    monkeypatch,
):
    from nexus.distributed.artifacts.tickets import UPLOAD_TICKET_HEADER

    content = b'lease-loss-upload'
    sha256 = hashlib.sha256(content).hexdigest()
    config = DistributedConfig(
        artifact_local_root=str(tmp_path / 'lease-loss-store'),
        artifact_upload_spool_dir=str(tmp_path / 'lease-loss-spool'),
    )
    store = LocalArtifactStore(tmp_path / 'lease-loss-store')
    async with cloud_session_factory() as session:
        upload, artifact = await ArtifactRepository(session).create_upload(
            CreateArtifactUpload(
                owner_principal_id='owner-lease-loss',
                environment_id='env-lease-loss',
                owner_tenant='tenant',
                created_by='owner-lease-loss',
                kind='source_package',
                sha256=sha256,
                size_bytes=len(content),
                idempotency_scope='lease-loss',
                idempotency_key='upload',
                session_ttl=timedelta(minutes=5),
                artifact_ttl=timedelta(days=1),
            ),
            store=store,
            now=utcnow(),
        )
        await session.commit()
    ticket = UploadTicketSigner(
        keys=config.artifact_ticket_keyring(),
        current_key_id=config.artifact_upload_ticket_current_key_id,
        ttl=config.artifact_upload_ticket_ttl,
    ).issue(
        upload_id=upload.upload_id,
        artifact_id=artifact.artifact_id,
        owner_principal_id='owner-lease-loss',
        environment_id='env-lease-loss',
        expected_kind='source_package',
        expected_size_bytes=len(content),
        expected_sha256=sha256,
        now=utcnow(),
    )
    original_renew = ArtifactRepository.renew_receiving
    renew_calls = 0

    async def lose_after_initial_renew(self, *args, **kwargs):
        nonlocal renew_calls
        renew_calls += 1
        if renew_calls > 1:
            return False
        return await original_renew(self, *args, **kwargs)

    monkeypatch.setattr(ArtifactRepository, 'renew_receiving', lose_after_initial_renew)
    client = TestClient(TestServer(create_application_app(session_factory=cloud_session_factory, config=config)))
    await client.start_server()
    try:
        response = await client.put(
            f'/v1/artifact-uploads/{upload.upload_id}/content',
            data=content,
            headers={UPLOAD_TICKET_HEADER: ticket},
        )
        assert response.status == 409
        assert (await response.json())['error'] == 'artifact_upload_lease_lost'
    finally:
        await client.close()

    async with cloud_session_factory() as session:
        upload_row = await session.get(ArtifactUploadRow, upload.upload_id)
        artifact_row = await session.get(ArtifactRow, artifact.artifact_id)
        assert upload_row is not None
        assert artifact_row is not None
        assert upload_row.status == 'pending'
        assert artifact_row.status == 'uploading'


async def test_artifact_upload_does_not_reserve_owner_item_or_byte_quota(cloud_session_factory, tmp_path):
    store = LocalArtifactStore(tmp_path / 'store')
    request = CreateArtifactUpload(
        owner_principal_id='owner-no-artifact-quota',
        environment_id='env-no-artifact-quota',
        owner_tenant='tenant',
        created_by='owner-no-artifact-quota',
        kind='source_package',
        sha256='a' * 64,
        size_bytes=1024,
        idempotency_scope='no-artifact-quota',
        idempotency_key='first',
        session_ttl=timedelta(hours=1),
        artifact_ttl=timedelta(days=1),
    )

    async with cloud_session_factory() as session:
        repo = ArtifactRepository(session)
        first, _ = await repo.create_upload(request, store=store, now=utcnow())
        second, _ = await repo.create_upload(
            replace(request, sha256='b' * 64, idempotency_key='second'),
            store=store,
            now=utcnow(),
        )
        await session.commit()

    assert first.upload_id != second.upload_id
    async with cloud_session_factory() as session:
        assert await session.get(ArtifactQuotaUsageRow, 'owner-no-artifact-quota') is None


async def test_artifact_maintenance_claims_are_renewable_and_fenced(cloud_session_factory, tmp_path):
    store = LocalArtifactStore(tmp_path / 'maintenance-fencing-store')
    started_at = utcnow() - timedelta(hours=2)
    async with cloud_session_factory() as session:
        repo = ArtifactRepository(session)
        upload, _ = await repo.create_upload(
            CreateArtifactUpload(
                owner_principal_id='owner-maintenance-fencing',
                environment_id='env-maintenance-fencing',
                owner_tenant='tenant',
                created_by='owner-maintenance-fencing',
                kind='source_package',
                sha256='d' * 64,
                size_bytes=10,
                idempotency_scope='maintenance-fencing',
                idempotency_key='upload',
                session_ttl=timedelta(minutes=30),
                artifact_ttl=timedelta(minutes=30),
            ),
            store=store,
            now=started_at,
        )
        upload_claim = (
            await repo.claim_upload_maintenance(
                operation_owner='maintenance-1',
                lease_ttl=timedelta(minutes=1),
                limit=1,
                now=utcnow(),
            )
        )[0]
        renewed_at = utcnow() + timedelta(seconds=10)
        assert await repo.renew_upload_maintenance(
            upload_claim,
            lease_ttl=timedelta(minutes=1),
            now=renewed_at,
        )
        assert not await repo.renew_upload_maintenance(
            replace(upload_claim, operation_id='op-stale'),
            lease_ttl=timedelta(minutes=1),
            now=renewed_at,
        )
        row = await session.get(ArtifactUploadRow, upload.upload_id)
        assert row is not None
        assert row.operation_expires_at == renewed_at + timedelta(minutes=1)
        row.operation_expires_at = utcnow() - timedelta(seconds=1)
        await session.flush()
        replacement_claim = (
            await repo.claim_upload_maintenance(
                operation_owner='maintenance-2',
                lease_ttl=timedelta(minutes=1),
                limit=1,
                now=utcnow(),
            )
        )[0]
        assert replacement_claim.operation_id != upload_claim.operation_id
        assert replacement_claim.previous_status == 'receiving'

        row.status = 'completed'
        row.operation_id = None
        row.operation_owner = None
        row.operation_expires_at = None
        artifact = await session.get(ArtifactRow, row.artifact_id)
        assert artifact is not None
        artifact.status = 'ready'
        artifact.expires_at = utcnow() - timedelta(seconds=1)
        await session.flush()
        delete_claim = (
            await repo.claim_expired_artifacts(
                operation_owner='maintenance-1',
                lease_ttl=timedelta(minutes=1),
                limit=1,
                now=utcnow(),
            )
        )[0]
        assert await repo.renew_artifact_delete(
            delete_claim,
            lease_ttl=timedelta(minutes=1),
            now=renewed_at,
        )
        assert not await repo.renew_artifact_delete(
            replace(delete_claim, operation_id='op-stale'),
            lease_ttl=timedelta(minutes=1),
            now=renewed_at,
        )
        await session.commit()


async def test_artifact_repository_accepts_run_result_and_rejects_dataset_kind(cloud_session_factory, tmp_path):
    store = LocalArtifactStore(tmp_path / 'store')
    request = CreateArtifactUpload(
        owner_principal_id='owner-artifact-kinds',
        environment_id='env-artifact-kinds',
        owner_tenant='tenant',
        created_by='owner-artifact-kinds',
        kind='run_result',
        sha256='c' * 64,
        size_bytes=10,
        idempotency_scope='artifact-kinds',
        idempotency_key='run-result',
        session_ttl=timedelta(hours=1),
        artifact_ttl=timedelta(days=1),
    )

    async with cloud_session_factory() as session:
        repo = ArtifactRepository(session)
        upload, artifact = await repo.create_upload(request, store=store, now=utcnow())
        with pytest.raises(ValueError, match='invalid_artifact_kind'):
            await repo.create_upload(
                replace(request, kind='dataset', idempotency_key='dataset'),
                store=store,
                now=utcnow(),
            )
        await session.commit()

    assert upload.expected_kind == 'run_result'
    assert artifact.kind == 'run_result'


async def test_artifact_maintenance_recovers_completed_remote_put(cloud_session_factory, tmp_path):
    store = LocalArtifactStore(tmp_path / 'store')
    content = tmp_path / 'source.tar.zst'
    content.write_bytes(b'recover-completed-put')
    sha256 = hashlib.sha256(content.read_bytes()).hexdigest()
    started_at = utcnow() - timedelta(hours=2)
    async with cloud_session_factory() as session:
        repo = ArtifactRepository(session)
        upload, artifact = await repo.create_upload(
            CreateArtifactUpload(
                owner_principal_id='owner-recovery',
                environment_id='env-recovery',
                owner_tenant='tenant',
                created_by='owner-recovery',
                kind='source_package',
                sha256=sha256,
                size_bytes=content.stat().st_size,
                idempotency_scope='recovery',
                idempotency_key='completed-put',
                session_ttl=timedelta(hours=4),
                artifact_ttl=timedelta(days=1),
            ),
            store=store,
            now=started_at,
        )
        await repo.claim_receiving(
            upload.upload_id,
            owner_principal_id='owner-recovery',
            environment_id='env-recovery',
            operation_owner='crashed-gateway',
            lease_ttl=timedelta(minutes=1),
            now=started_at,
        )
        key = artifact_object_key(
            owner_principal_id='owner-recovery',
            artifact_id=artifact.artifact_id,
            sha256=sha256,
            kind='source_package',
        )
        await store.put(
            content,
            object_key=key,
            sha256=sha256,
            size_bytes=content.stat().st_size,
        )
        await session.commit()

    config = DistributedConfig(
        artifact_local_root=str(tmp_path / 'store'),
        artifact_maintenance_spool_dir=str(tmp_path / 'maintenance'),
    )
    maintenance = ArtifactMaintenanceReconciler(
        session_factory=cloud_session_factory,
        config=config,
        owner_id='maintenance-test',
        store=store,
    )
    try:
        assert await maintenance.reconcile_once() == 1
    finally:
        await maintenance.close()

    async with cloud_session_factory() as session:
        recovered_upload = await session.get(ArtifactUploadRow, upload.upload_id)
        recovered_artifact = await session.get(ArtifactRow, artifact.artifact_id)
        usage = await session.get(ArtifactQuotaUsageRow, 'owner-recovery')
        assert recovered_upload.status == 'completed'
        assert recovered_artifact.status == 'ready'
        assert usage is None


async def test_artifact_maintenance_claims_one_item_at_a_time(
    cloud_session_factory,
    tmp_path,
    monkeypatch,
):
    observed_limits: list[int] = []
    original_claim = ArtifactRepository.claim_upload_maintenance

    async def recording_claim(self, **kwargs):
        observed_limits.append(kwargs['limit'])
        return await original_claim(self, **kwargs)

    monkeypatch.setattr(ArtifactRepository, 'claim_upload_maintenance', recording_claim)
    maintenance = ArtifactMaintenanceReconciler(
        session_factory=cloud_session_factory,
        config=DistributedConfig(
            artifact_local_root=str(tmp_path / 'claim-one-store'),
            artifact_maintenance_spool_dir=str(tmp_path / 'claim-one-spool'),
            artifact_maintenance_batch_size=7,
        ),
        owner_id='maintenance-test',
    )
    try:
        assert await maintenance.reconcile_once() == 0
    finally:
        await maintenance.close()

    assert observed_limits == [1]


async def test_artifact_maintenance_renews_lease_during_slow_store_read(
    cloud_session_factory,
    tmp_path,
    monkeypatch,
):
    store = LocalArtifactStore(tmp_path / 'slow-maintenance-store')
    content = tmp_path / 'slow-source.tar.zst'
    content.write_bytes(b'slow-maintenance-recovery')
    sha256 = hashlib.sha256(content.read_bytes()).hexdigest()
    started_at = utcnow() - timedelta(hours=2)
    async with cloud_session_factory() as session:
        repo = ArtifactRepository(session)
        upload, artifact = await repo.create_upload(
            CreateArtifactUpload(
                owner_principal_id='owner-slow-maintenance',
                environment_id='env-slow-maintenance',
                owner_tenant='tenant',
                created_by='owner-slow-maintenance',
                kind='source_package',
                sha256=sha256,
                size_bytes=content.stat().st_size,
                idempotency_scope='slow-maintenance',
                idempotency_key='upload',
                session_ttl=timedelta(hours=4),
                artifact_ttl=timedelta(days=1),
            ),
            store=store,
            now=started_at,
        )
        await repo.claim_receiving(
            upload.upload_id,
            owner_principal_id='owner-slow-maintenance',
            environment_id='env-slow-maintenance',
            operation_owner='crashed-gateway',
            lease_ttl=timedelta(milliseconds=50),
            now=started_at,
        )
        key = artifact_object_key(
            owner_principal_id='owner-slow-maintenance',
            artifact_id=artifact.artifact_id,
            sha256=sha256,
            kind='source_package',
        )
        await store.put(content, object_key=key, sha256=sha256, size_bytes=content.stat().st_size)
        await session.commit()

    original_get = store.get
    get_started = asyncio.Event()
    release_get = asyncio.Event()

    async def blocking_get(*args, **kwargs):
        get_started.set()
        await release_get.wait()
        return await original_get(*args, **kwargs)

    monkeypatch.setattr(store, 'get', blocking_get)
    maintenance = ArtifactMaintenanceReconciler(
        session_factory=cloud_session_factory,
        config=DistributedConfig(
            artifact_local_root=str(tmp_path / 'slow-maintenance-store'),
            artifact_maintenance_spool_dir=str(tmp_path / 'slow-maintenance-spool'),
            artifact_maintenance_lease_ttl=timedelta(milliseconds=150),
            artifact_maintenance_lease_renew_interval=timedelta(milliseconds=20),
        ),
        owner_id='maintenance-test',
        store=store,
    )
    task = asyncio.create_task(maintenance.reconcile_once())
    try:
        # These timeouts only guard against a hang; the assertions below carry the
        # actual contract. Keep them far above the real cost (tens of ms) so a busy
        # CI box slowing down SQLite cannot turn a passing run into a failure.
        await asyncio.wait_for(get_started.wait(), timeout=HANG_GUARD_SECONDS)
        await asyncio.sleep(0.25)
        async with cloud_session_factory() as session:
            row = await session.get(ArtifactUploadRow, upload.upload_id)
            assert row is not None
            assert row.operation_expires_at is not None
            assert row.operation_expires_at > utcnow()
        release_get.set()
        assert await asyncio.wait_for(task, timeout=HANG_GUARD_SECONDS) == 1
    finally:
        release_get.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await maintenance.close()


async def test_artifact_maintenance_does_not_publish_unlinked_task_list(cloud_session_factory, tmp_path):
    store = LocalArtifactStore(tmp_path / 'task-list-store')
    content = tmp_path / 'tasks.jsonl'
    content.write_bytes(b'{"item_id":"item-1","run_input":{}}\n')
    sha256 = hashlib.sha256(content.read_bytes()).hexdigest()
    started_at = utcnow() - timedelta(hours=2)
    async with cloud_session_factory() as session:
        repo = ArtifactRepository(session)
        upload, artifact = await repo.create_upload(
            CreateArtifactUpload(
                owner_principal_id='owner-task-list-recovery',
                environment_id='env-task-list-recovery',
                owner_tenant='tenant',
                created_by='system:batch-parent',
                kind='task_list',
                sha256=sha256,
                size_bytes=content.stat().st_size,
                idempotency_scope='batch-task-list',
                idempotency_key='unlinked-task-list',
                session_ttl=timedelta(hours=4),
                artifact_ttl=timedelta(days=1),
            ),
            store=store,
            now=started_at,
        )
        await repo.claim_receiving(
            upload.upload_id,
            owner_principal_id='owner-task-list-recovery',
            environment_id='env-task-list-recovery',
            operation_owner='crashed-batch-parent',
            lease_ttl=timedelta(minutes=1),
            now=started_at,
        )
        key = artifact_object_key(
            owner_principal_id='owner-task-list-recovery',
            artifact_id=artifact.artifact_id,
            sha256=sha256,
            kind='task_list',
        )
        stored = await store.put(content, object_key=key, sha256=sha256, size_bytes=content.stat().st_size)
        await session.commit()

    maintenance = ArtifactMaintenanceReconciler(
        session_factory=cloud_session_factory,
        config=DistributedConfig(
            artifact_local_root=str(tmp_path / 'task-list-store'),
            artifact_maintenance_spool_dir=str(tmp_path / 'task-list-maintenance'),
        ),
        owner_id='maintenance-test',
        store=store,
    )
    try:
        assert await maintenance.reconcile_once() == 1
    finally:
        await maintenance.close()

    async with cloud_session_factory() as session:
        reset_upload = await session.get(ArtifactUploadRow, upload.upload_id)
        unpublished_artifact = await session.get(ArtifactRow, artifact.artifact_id)
        assert reset_upload.status == 'pending'
        assert unpublished_artifact.status == 'uploading'
    with pytest.raises(FileNotFoundError):
        await store.stat(stored)


async def test_generic_repository_store_contract(tmp_path):
    objects: dict[str, bytes] = {}

    async def put(request):
        objects[request.path] = await request.read()
        return web.Response(status=201, headers={'ETag': 'v1'})

    async def head(request):
        value = objects.get(request.path)
        if value is None:
            raise web.HTTPNotFound()
        return web.Response(headers={'Content-Length': str(len(value)), 'ETag': 'v1'})

    async def get(request):
        value = objects.get(request.path)
        if value is None:
            raise web.HTTPNotFound()
        return web.Response(body=value)

    async def delete(request):
        objects.pop(request.path, None)
        return web.Response(status=204)

    app = web.Application()
    app.router.add_put('/repo/{tail:.*}', put)
    app.router.add_head('/repo/{tail:.*}', head)
    app.router.add_get('/repo/{tail:.*}', get, allow_head=False)
    app.router.add_delete('/repo/{tail:.*}', delete)
    server = TestServer(app)
    await server.start_server()
    try:
        source = tmp_path / 'input.bin'
        source.write_bytes(b'generic repository')
        sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        store = GenericRepositoryArtifactStore(
            base_url=str(server.make_url('/')).rstrip('/'),
            repository='repo',
            credentials=CredentialFiles(username_env='TEST_REPO_USER', token_env='TEST_REPO_TOKEN'),
        )
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv('TEST_REPO_USER', 'test-user')
            monkeypatch.setenv('TEST_REPO_TOKEN', 'test-token')
            stored = await store.put(
                source,
                object_key='owner/artifact/sha/file.bin',
                sha256=sha256,
                size_bytes=source.stat().st_size,
            )
            assert (await store.stat(stored)).size_bytes == source.stat().st_size
            target = tmp_path / 'download.bin'
            await store.get(stored, target)
            assert target.read_bytes() == source.read_bytes()
            await store.delete(stored)
            with pytest.raises(FileNotFoundError):
                await store.stat(stored)
        await store.close()
    finally:
        await server.close()


async def test_artifact_upload_coordinator_retries_transient_store_throttling(tmp_path):
    from nexus.distributed.artifacts.store import (
        ArtifactStoreTransientError,
        ArtifactUploadCoordinator,
        StoredObject,
    )

    source = tmp_path / 'result.json'
    source.write_bytes(b'{}')
    delays = []

    class ThrottledStore:
        def __init__(self):
            self.calls = 0

        async def put(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls < 3:
                raise ArtifactStoreTransientError(429)
            return StoredObject(uri='generic://repo/result.json', etag='result')

    async def sleep(delay):
        delays.append(delay)

    store = ThrottledStore()
    coordinator = ArtifactUploadCoordinator(
        total_concurrency=2,
        input_concurrency=1,
        result_concurrency=1,
        retry_attempts=3,
        sleep=sleep,
    )

    stored = await coordinator.put(
        store,
        source,
        kind='run_result',
        object_key='owner/result.json',
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        size_bytes=source.stat().st_size,
    )

    assert stored.etag == 'result'
    assert store.calls == 3
    assert delays == [0.25, 0.5]


async def test_artifact_upload_coordinator_keeps_input_capacity_during_result_burst(tmp_path):
    from nexus.distributed.artifacts.store import ArtifactUploadCoordinator, StoredObject

    source = tmp_path / 'content.bin'
    source.write_bytes(b'x')
    release = asyncio.Event()
    two_started = asyncio.Event()
    active = 0
    active_by_lane = {'input': 0, 'result': 0}
    maximum_active = 0
    maximum_by_lane = {'input': 0, 'result': 0}

    class BlockingStore:
        async def put(self, _path, *, object_key, **_kwargs):
            nonlocal active, maximum_active
            lane = 'result' if object_key.startswith('result/') else 'input'
            active += 1
            active_by_lane[lane] += 1
            maximum_active = max(maximum_active, active)
            maximum_by_lane[lane] = max(maximum_by_lane[lane], active_by_lane[lane])
            if active == 2:
                two_started.set()
            try:
                await release.wait()
                return StoredObject(uri=f'generic://repo/{object_key}')
            finally:
                active -= 1
                active_by_lane[lane] -= 1

    coordinator = ArtifactUploadCoordinator(
        total_concurrency=2,
        input_concurrency=1,
        result_concurrency=1,
        retry_attempts=1,
    )
    store = BlockingStore()

    async def upload(kind, index):
        return await coordinator.put(
            store,
            source,
            kind=kind,
            object_key=f'{"result" if kind == "run_result" else "input"}/{index}',
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            size_bytes=source.stat().st_size,
        )

    tasks = [
        asyncio.create_task(upload('run_result', 1)),
        asyncio.create_task(upload('run_result', 2)),
        asyncio.create_task(upload('source_package', 1)),
        asyncio.create_task(upload('source_package', 2)),
    ]
    await asyncio.wait_for(two_started.wait(), timeout=HANG_GUARD_SECONDS)
    assert active_by_lane == {'input': 1, 'result': 1}
    release.set()
    await asyncio.gather(*tasks)

    assert maximum_active == 2
    assert maximum_by_lane == {'input': 1, 'result': 1}


def test_upload_ticket_rejects_tampering_and_expiry():
    now = utcnow()
    signer = UploadTicketSigner(keys={'v1': 'secret'}, current_key_id='v1', ttl=timedelta(seconds=5))
    ticket = signer.issue(
        upload_id='upl-1',
        artifact_id='art-1',
        owner_principal_id='owner-1',
        environment_id='env-1',
        expected_kind='source_package',
        expected_size_bytes=3,
        expected_sha256='a' * 64,
        now=now,
    )
    claims = signer.verify(ticket, now=now)
    assert claims.upload_id == 'upl-1'
    assert claims.environment_id == 'env-1'
    assert claims.expected_kind == 'source_package'
    with pytest.raises(ValueError, match='invalid_upload_ticket'):
        signer.verify(f'{ticket[:-1]}x', now=now)
    with pytest.raises(ValueError, match='upload_ticket_expired'):
        signer.verify(ticket, now=now + timedelta(seconds=6))


def test_artifact_content_ticket_is_bound_to_allocation_and_attempt_scope():
    now = utcnow()
    signer = ArtifactContentTicketSigner(
        keys={'v1': 'secret'},
        current_key_id='v1',
        ttl=timedelta(minutes=5),
    )

    ticket = signer.issue(
        execution_id='exe-1',
        top_level_execution_id='exe-1',
        environment_id='env-1',
        allocation_id='alloc-1',
        assignment_id='asn-1',
        attempt_id='attempt-1',
        operations=['artifact:read', 'run_result:write'],
        artifact_ids=['art-source', 'art-wheel'],
        now=now,
    )
    claims = signer.verify(ticket, now=now)

    assert claims.execution_id == 'exe-1'
    assert claims.top_level_execution_id == 'exe-1'
    assert claims.environment_id == 'env-1'
    assert claims.allocation_id == 'alloc-1'
    assert claims.assignment_id == 'asn-1'
    assert claims.attempt_id == 'attempt-1'
    assert claims.operations == ['artifact:read', 'run_result:write']
    assert claims.artifact_ids == ['art-source', 'art-wheel']


def test_source_package_is_deterministic_and_extracts_safely(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    (root / 'module.py').write_text('VALUE = 1\n')
    (root / '.env').write_text('secret=value\n')
    first = build_source_package(root, tmp_path / 'first.tar.zst')
    second = build_source_package(root, tmp_path / 'second.tar.zst')
    assert first.sha256 == second.sha256
    target = tmp_path / 'workspace'
    safe_extract_source_package(first.path, target)
    assert (target / 'module.py').read_text() == 'VALUE = 1\n'
    assert not (target / '.env').exists()


def test_source_package_limits_count_archive_entries(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'only.py').write_text('VALUE = 1\n')
    package = build_source_package(root, tmp_path / 'source.tar.zst', limits=PackageLimits(max_files=1))

    target = tmp_path / 'target'
    safe_extract_source_package(package.path, target, limits=PackageLimits(max_files=1))
    assert (target / 'only.py').read_text() == 'VALUE = 1\n'

    with pytest.raises(ValueError, match='archive size limit'):
        safe_extract_source_package(package.path, tmp_path / 'too-small', limits=PackageLimits(max_archive_bytes=1))


def test_source_package_rejects_absolute_directory_symlinks(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    target = tmp_path / 'target'
    target.mkdir()
    (source / 'linked').symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match='source symlink target must be relative'):
        build_source_package(source, tmp_path / 'source.tar.zst')


def test_source_package_treats_old_manifest_name_as_a_regular_file(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / '.nexus-manifest.json').write_text('{}')

    package = build_source_package(source, tmp_path / 'source.tar.zst')
    target = tmp_path / 'target'
    safe_extract_source_package(package.path, target)

    assert (target / '.nexus-manifest.json').read_text() == '{}'


def test_source_package_force_excludes_submit_token_file(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'main.py').write_text('VALUE = 1\n')
    token_file = source / 'submit-token.txt'
    token_file.write_text('submit-secret\n')

    package = build_source_package(
        source,
        tmp_path / 'source.tar.zst',
        excluded_paths=[token_file],
    )
    target = tmp_path / 'target'
    safe_extract_source_package(package.path, target)

    assert (target / 'main.py').is_file()
    assert not (target / 'submit-token.txt').exists()


def test_typed_submission_rejects_internal_driver_fields():
    payload = {
        'submission_version': 1,
        'idempotency_key': 'single-internal-field',
        'source_package': {
            'ref': {'artifact_id': 'source', 'sha256': 'a' * 64, 'size_bytes': 1, 'kind': 'source_package'}
        },
        'execution_environment': {
            'python_version': '3.13',
            'wheel': {'ref': {'artifact_id': 'wheel', 'sha256': 'b' * 64, 'size_bytes': 1, 'kind': 'wheel'}},
        },
        'run_input': {'task_name': 'identity'},
        'driver_config': {'run_trace_sinks': ['other']},
    }

    with pytest.raises(ValueError, match='driver_config'):
        SingleSubmission.model_validate(payload)


async def test_artifact_upload_and_typed_single_submission_gateway_flow(
    cloud_session_factory, tmp_path, monkeypatch, otel_hyway_instrumentor
):
    from opentelemetry.instrumentation.aiohttp_server import AioHttpServerInstrumentor
    from opentelemetry.sdk.trace import TracerProvider

    trace_headers = {
        'traceparent': '00-11111111111111111111111111111111-2222222222222222-01',
        'tracestate': 'nexus=validation',
    }
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    tracer_provider = TracerProvider()
    instrumentor = AioHttpServerInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    keys = {'v1': 'upload-secret'}
    base_config = DistributedConfig(
        artifact_local_root=str(tmp_path / 'store'),
        artifact_upload_spool_dir=str(tmp_path / 'spool'),
        secrets={'artifact_ticket_keys': keys},
        artifact_upload_ticket_current_key_id='v1',
        environment_enabled_providers=['kubernetes'],
        environment_base_image='test-image:latest',
    )
    application = TestClient(
        TestServer(create_application_app(session_factory=cloud_session_factory, config=base_config))
    )
    await application.start_server()
    control_config = base_config.model_copy(
        update={'application_public_base_url': str(application.make_url('/')).rstrip('/')}
    )
    control = TestClient(TestServer(create_control_app(session_factory=cloud_session_factory, config=control_config)))
    await control.start_server()
    try:
        async with cloud_session_factory() as session:
            await ProviderConfigRepository(session).create(
                provider_config_ref='kubernetes/kind',
                config={
                    'provider_type': 'kubernetes',
                    'namespace': 'default',
                    'auth_type': 'in_cluster',
                },
                status='enabled',
                actor='test',
                now=utcnow(),
            )
            environments = EnvironmentRepository(session)
            await environments.create_environment(
                environment_id='kind',
                name='Kind',
                allocation_type='dynamic',
                provider_type='kubernetes',
                provider_config_ref='kubernetes/kind',
                resource_limits={
                    'manager_count': 2,
                    'cpu_millicores': 4_000,
                    'memory_bytes': 8 * 1024**3,
                    'slots': 2,
                    'active_batches': 1,
                },
                default_timeouts={},
                status='enabled',
                actor='test',
                now=utcnow(),
            )
            await environments.upsert_grant(
                principal_id='local-dev',
                environment_id='kind',
                limits={
                    'manager_count': 2,
                    'cpu_millicores': 4_000,
                    'memory_bytes': 8 * 1024**3,
                    'slots': 2,
                    'active_batches': 1,
                },
                enabled=True,
                actor='test',
                now=utcnow(),
            )
            await session.commit()

        async def upload(kind: str, content: bytes) -> dict:
            sha256 = hashlib.sha256(content).hexdigest()
            response = await control.post(
                '/v1/artifact-uploads',
                json={
                    'kind': kind,
                    'sha256': sha256,
                    'size_bytes': len(content),
                    'idempotency_key': f'{kind}-1',
                    'environment_id': 'kind',
                },
            )
            assert response.status == 201
            session = await response.json()
            response = await application.put(
                f'/v1/artifact-uploads/{session["upload"]["upload_id"]}/content',
                data=content,
                headers={'X-Nexus-Upload-Ticket': session['upload_ticket']},
            )
            assert response.status == 201, await response.text()
            return (await response.json())['ref']

        source_ref = await upload('source_package', b'source')
        wheel_path = tmp_path / 'nexus_agent-1.5.0-py3-none-any.whl'
        with zipfile.ZipFile(wheel_path, 'w') as archive:
            archive.writestr(
                'nexus_agent-1.5.0.dist-info/METADATA',
                'Metadata-Version: 2.4\nName: nexus-agent\nVersion: 1.5.0\n',
            )
            archive.writestr(
                'nexus_agent-1.5.0.dist-info/WHEEL',
                'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
            )
        wheel_ref = await upload('nexus_wheel', wheel_path.read_bytes())
        lock_ref = await upload('uv_lock', b'version = 1\n')
        response = await control.post(
            '/v1/submissions/single',
            headers=trace_headers,
            json={
                'submission_version': 1,
                'idempotency_key': 'single-1',
                'environment_id': 'kind',
                'code': {
                    'mode': 'source_package',
                    'source_package': source_ref,
                    'nexus_wheel': wheel_ref,
                    'uv_lock': lock_ref,
                    'extras': [],
                },
                'resources': {
                    'managers': {'count': 1, 'cpu_cores': 1, 'memory_gib': 1, 'slots': 1},
                },
                'run_input': {'task_name': 'identity'},
            },
        )
        assert response.status == 202, await response.text()
        execution = await response.json()
        assert execution['phase'] == 'building_dependencies'
        async with cloud_session_factory() as session:
            row = await session.get(ExecutionRow, execution['execution_id'])
            assert row is not None
            allocation = await session.get(EnvironmentAllocationRow, row.allocation_id)
            assert allocation is not None
            assert row.spec_type == 'nexus_single'
            assert row.owner_principal_id == 'local-dev'
            assert row.environment_id == 'kind'
            assert row.trace_carrier_json is not None
            assert row.trace_carrier_json['traceparent'].startswith('00-11111111111111111111111111111111-')
            assert row.trace_carrier_json['traceparent'].endswith('-01')
            assert row.trace_carrier_json['tracestate'] == 'hyway_slice_id=single-1,nexus=validation'
            assert row.placement_json['labels'] == {'workload': 'execution'}
            assert allocation.status == 'requested'
            assert allocation.workload == 'execution'
            assert allocation.desired_manager_count == 1

        response = await control.post(
            '/v1/submissions/live',
            json={
                'submission_version': 1,
                'idempotency_key': 'live-1',
                'environment_id': 'kind',
                'code': {
                    'mode': 'source_package',
                    'source_package': source_ref,
                    'nexus_wheel': wheel_ref,
                    'uv_lock': lock_ref,
                    'extras': [],
                },
                'resources': {
                    'managers': {'count': 1, 'cpu_cores': 1, 'memory_gib': 1, 'slots': 1},
                },
                'run_input': {'task_name': 'identity'},
                'live': {},
            },
        )
        assert response.status == 202, await response.text()
        live_execution = await response.json()
        assert live_execution['phase'] == 'building_dependencies'
        assert live_execution['result_url'].endswith(f'/v1/executions/{live_execution["execution_id"]}/result')
        assert live_execution['events_url'].endswith(f'/v1/executions/{live_execution["execution_id"]}/live/events')
        assert live_execution['command_url'].endswith(f'/v1/executions/{live_execution["execution_id"]}/commands')
        async with cloud_session_factory() as session:
            row = await session.get(ExecutionRow, live_execution['execution_id'])
            assert row is not None
            assert row.spec_type == 'nexus_live'
            assert row.route_status == 'active'
    finally:
        await control.close()
        await application.close()
        instrumentor.uninstrument()
        tracer_provider.shutdown()


async def test_idempotent_transaction_retries_unique_conflict_with_fresh_session():
    from nexus.distributed.gateways.control.app import _run_idempotent_transaction

    sessions = []

    class FakeSession:
        def __init__(self) -> None:
            self.committed = False
            self.rolled_back = False

        async def __aenter__(self):
            sessions.append(self)
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def commit(self):
            self.committed = True

        async def rollback(self):
            self.rolled_back = True

    class FakeSessionFactory:
        def __call__(self):
            return FakeSession()

    attempts = 0

    async def operation(session):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise IntegrityError('INSERT', {}, RuntimeError('Duplicate entry for key uq_executions_idempotency'))
        return session

    result = await _run_idempotent_transaction(
        FakeSessionFactory(),
        operation,
        conflict_markers=('uq_executions_idempotency',),
    )

    assert result is sessions[1]
    assert len(sessions) == 2
    assert sessions[0].rolled_back is True
    assert sessions[0].committed is False
    assert sessions[1].committed is True


async def test_idempotent_transaction_retries_explicit_aggregate_conflict_with_fresh_session():
    from nexus.distributed.persistence.db import TransactionRetryError, run_idempotent_transaction

    sessions = []

    class FakeSession:
        def __init__(self) -> None:
            self.committed = False
            self.rolled_back = False

        async def __aenter__(self):
            sessions.append(self)
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def commit(self):
            self.committed = True

        async def rollback(self):
            self.rolled_back = True

    class FakeSessionFactory:
        def __call__(self):
            return FakeSession()

    attempts = 0

    async def operation(session):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TransactionRetryError('submission_concurrent_winner')
        return session

    result = await run_idempotent_transaction(
        FakeSessionFactory(),
        operation,
        conflict_markers=('uq_executions_idempotency',),
    )

    assert result is sessions[1]
    assert sessions[0].rolled_back is True
    assert sessions[1].committed is True


async def test_identity_admin_can_manage_user_quota(cloud_session_factory):
    application_token = 't' * 32
    admin_subject = '1000'
    user_subject = '1001'
    config = DistributedConfig(
        auth_mode='identity',
        admin_users=[f'tai/default/{admin_subject}'],
        secrets={
            'tai_application_token': application_token,
        },
    )
    client = TestClient(TestServer(create_control_app(session_factory=cloud_session_factory, config=config)))
    await client.start_server()
    admin_headers = _tai_identity_header(staff_id=int(admin_subject), application_token=application_token)
    user_id = principal_id('tai', 'default', user_subject)
    try:
        response = await client.put(
            f'/v1/access/users/{user_id}/roles',
            headers=admin_headers,
            json={
                'issuer': 'tai',
                'tenant': 'default',
                'subject': user_subject,
                'roles': ['user'],
            },
        )
        assert response.status == 200, await response.text()
        response = await client.put(
            f'/v1/access/users/{user_id}/quota',
            headers=admin_headers,
            json={'limits': {'running_executions': -1}},
        )
        assert response.status == 422
        response = await client.put(
            f'/v1/access/users/{user_id}/quota',
            headers=admin_headers,
            json={'limits': {'unknown_limit': 1}},
        )
        assert response.status == 422
        response = await client.put(
            f'/v1/access/users/{user_id}/quota',
            headers=admin_headers,
            json={'limits': {'running_executions': 3}},
        )
        assert response.status == 200
        quota = await response.json()
        assert quota['limits']['running_executions'] == 3
        assert quota['limits']['queued_executions'] == config.default_user_quota['queued_executions']
        assert quota['overrides'] == {'running_executions': 3}
    finally:
        await client.close()
