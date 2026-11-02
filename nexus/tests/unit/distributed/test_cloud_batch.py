from __future__ import annotations

import asyncio
import hashlib
import json
import signal
import threading

from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.dataset.zip_index import write_nexus_zip_directory_to_file
from nexus.distributed.artifacts.gateway import ArtifactGatewayTransientError
from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.batch import control as batch_control_module
from nexus.distributed.batch.control import (
    BatchControlService,
    _generation_limits,
    _materialize_compose_subprocess,
    _materialize_generator_subprocess,
)
from nexus.distributed.batch.data_list import DataListError, DataListWriter
from nexus.distributed.batch.lifecycle import BatchLifecycleReconciler
from nexus.distributed.batch.materialize import (
    MaterializedTaskList,
    _write_result,
    materialize_compose,
    materialize_generator,
)
from nexus.distributed.config import DistributedConfig
from nexus.distributed.execution.drivers.nexus_batch import NexusBatchDriver
from nexus.distributed.execution.models import ExecutionStatus
from nexus.distributed.persistence.orm import (
    ArtifactRefRow,
    ArtifactRow,
    AssignmentRow,
    BatchItemAttemptRow,
    BatchItemRow,
    BatchRunRow,
    DependencyBuildRow,
    EnvironmentAllocationRow,
    ExecutionRow,
    create_schema,
)
from nexus.distributed.repositories.access import AccessRepository
from nexus.distributed.repositories.artifact import ArtifactRepository
from nexus.distributed.repositories.batch import BatchRepository
from nexus.distributed.repositories.environment import EnvironmentRepository
from nexus.distributed.repositories.execution_phase import ExecutionPhaseRepository
from nexus.distributed.repositories.provider_config import ProviderConfigRepository
from nexus.distributed.repositories.submission import SubmissionRepository, _validate_generation_admission
from nexus.distributed.submission.specs import BatchSubmission
from nexus.distributed.utils.clock import utcnow
from nexus.runner.compose.model import RawRunPart


OWNER = 'principal-batch'
ENVIRONMENT_ID = 'env-batch'
PARENT_ATTEMPT_ID = 'attempt-parent'
PARENT_ASSIGNMENT_ID = 'assignment-parent'


def _dynamic_config(**overrides) -> DistributedConfig:
    overrides.setdefault('environment_enabled_providers', ['kubernetes'])
    return DistributedConfig(**overrides)


@pytest.fixture
async def batch_session_factory(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "batch.db"}')
    await create_schema(engine)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def test_batch_control_blocking_runner_limits_concurrency(batch_session_factory, tmp_path):
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=_dynamic_config(
            batch_control_spool_dir=str(tmp_path / 'spool'),
            batch_control_blocking_concurrency=2,
        ),
    )
    release = threading.Event()
    entered_lock = threading.Lock()
    entered: list[int] = []

    def blocking_helper(index: int) -> int:
        with entered_lock:
            entered.append(index)
        release.wait(timeout=5)
        return index

    tasks = [asyncio.create_task(control._run_blocking('test-helper', blocking_helper, index)) for index in range(3)]
    try:
        for _ in range(100):
            with entered_lock:
                count = len(entered)
            if count == 2:
                break
            await asyncio.sleep(0.01)

        with entered_lock:
            assert len(entered) == 2
        assert sum(task.done() for task in tasks) == 0
    finally:
        release.set()
        assert sorted(await asyncio.gather(*tasks)) == [0, 1, 2]
        await control.close()


async def test_batch_control_blocking_runner_keeps_capacity_until_cancelled_thread_finishes(
    batch_session_factory,
    tmp_path,
):
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=_dynamic_config(
            batch_control_spool_dir=str(tmp_path / 'spool'),
            batch_control_blocking_concurrency=1,
        ),
    )
    first_release = threading.Event()
    first_entered = threading.Event()
    second_entered = threading.Event()

    def first_helper() -> None:
        first_entered.set()
        first_release.wait(timeout=5)

    def second_helper() -> None:
        second_entered.set()

    first = asyncio.create_task(control._run_blocking('first-helper', first_helper))
    try:
        assert await asyncio.to_thread(first_entered.wait, 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        second = asyncio.create_task(control._run_blocking('second-helper', second_helper))
        await asyncio.sleep(0.05)
        assert not second_entered.is_set()

        first_release.set()
        await asyncio.wait_for(second, timeout=1)
        assert second_entered.is_set()
    finally:
        first_release.set()
        await control.close()


async def test_batch_control_blocking_runner_keeps_timeout_watchdog_after_caller_cancellation(
    batch_session_factory,
    tmp_path,
):
    release = threading.Event()
    entered = threading.Event()
    timed_out = asyncio.Event()
    timed_out_operations: list[str] = []

    async def on_blocking_timeout(operation: str) -> None:
        timed_out_operations.append(operation)
        timed_out.set()

    control = BatchControlService(
        session_factory=batch_session_factory,
        config=_dynamic_config(
            batch_control_spool_dir=str(tmp_path / 'spool'),
            batch_control_blocking_concurrency=1,
            manager_executor_hang_timeout=timedelta(seconds=0.05),
        ),
        on_blocking_timeout=on_blocking_timeout,
    )

    def blocking_helper() -> None:
        entered.set()
        release.wait(timeout=5)

    caller = asyncio.create_task(control._run_blocking('cancelled-helper', blocking_helper))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller

        await asyncio.wait_for(timed_out.wait(), timeout=0.5)
        assert timed_out_operations == ['cancelled-helper']
    finally:
        release.set()
        await control.close()


async def test_batch_control_blocking_timeout_requests_manager_self_fence(batch_session_factory, tmp_path):
    timed_out: list[str] = []
    release = threading.Event()

    async def on_blocking_timeout(operation: str) -> None:
        timed_out.append(operation)

    control = BatchControlService(
        session_factory=batch_session_factory,
        config=_dynamic_config(
            batch_control_spool_dir=str(tmp_path / 'spool'),
            manager_executor_hang_timeout=timedelta(milliseconds=20),
        ),
        on_blocking_timeout=on_blocking_timeout,
    )
    try:
        with pytest.raises(TimeoutError, match='batch_blocking_operation_timeout:test-helper'):
            await control._run_blocking('test-helper', release.wait)
        assert timed_out == ['test-helper']
    finally:
        release.set()
        await asyncio.sleep(0.05)
        await control.close()


class FakeArtifactClient:
    def __init__(
        self,
        session_factory,
        contents: dict[str, bytes] | None = None,
        *,
        config: DistributedConfig | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.config = config or _dynamic_config(shard_count=8)
        self.contents = dict(contents or {})
        self.published: list[dict] = []
        self.downloads: list[str] = []
        self.download_deadlines: list[float | None] = []
        self.run_results: list[dict] = []

    async def publish_task_list(
        self,
        *,
        parent_execution_id: str,
        assignment_id: str,
        attempt_id: str,
        path: Path,
        sha256: str,
        size_bytes: int,
    ) -> dict:
        content = path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == sha256
        assert len(content) == size_bytes
        artifact_id = f'artifact-generated-{hashlib.sha256(parent_execution_id.encode()).hexdigest()[:16]}'
        ref = {
            'artifact_id': artifact_id,
            'kind': 'task_list',
            'sha256': sha256,
            'size_bytes': size_bytes,
        }
        now = utcnow()
        async with self.session_factory() as session:
            batch = await session.get(BatchRunRow, parent_execution_id)
            assert batch is not None
            assert batch.generation_status == 'publishing'
            assert batch.task_list_manifest_json is not None
            assert batch.task_list_manifest_json['sha256'] == sha256
            assert batch.task_list_manifest_json['size_bytes'] == size_bytes
            artifact = await session.get(ArtifactRow, artifact_id)
            if artifact is None:
                artifact = ArtifactRow(
                    artifact_id=artifact_id,
                    owner_principal_id=OWNER,
                    environment_id=ENVIRONMENT_ID,
                    owner_tenant='tenant',
                    kind='task_list',
                    producer_key=f'batch-task-list:{parent_execution_id}:{batch.generation_epoch}',
                    uri=f'memory://{artifact_id}',
                    sha256=sha256,
                    size_bytes=size_bytes,
                    provider_etag=None,
                    status='ready',
                    created_by='test',
                    expires_at=None,
                    created_at=now,
                    updated_at=now,
                )
                session.add(artifact)
                await session.flush()
            else:
                assert artifact.sha256 == sha256
                assert artifact.size_bytes == size_bytes
            repo = BatchRepository(session, config=self.config)
            await repo.fence_parent_attempt(parent_execution_id, attempt_id)
            await repo.set_generated_task_list(
                parent_execution_id,
                ref=ArtifactRef.model_validate(ref),
                manifest=dict(batch.task_list_manifest_json),
                generation_epoch=batch.generation_epoch,
                now=now,
            )
            await session.commit()
        self.contents[artifact_id] = content
        self.published.append({
            'parent_execution_id': parent_execution_id,
            'assignment_id': assignment_id,
            'attempt_id': attempt_id,
            'ref': ref,
        })
        return ref

    async def publish_run_result(
        self,
        *,
        execution_id: str,
        assignment_id: str,
        attempt_id: str,
        producer_key: str,
        content: bytes,
    ) -> dict:
        artifact_id = f'artifact-result-{attempt_id}'
        sha256 = hashlib.sha256(content).hexdigest()
        now = utcnow()
        async with self.session_factory() as session:
            artifact = await session.get(ArtifactRow, artifact_id)
            if artifact is None:
                artifact = ArtifactRow(
                    artifact_id=artifact_id,
                    owner_principal_id=OWNER,
                    environment_id=ENVIRONMENT_ID,
                    owner_tenant='tenant',
                    kind='run_result',
                    producer_key=producer_key,
                    uri=f'memory://{artifact_id}',
                    sha256=sha256,
                    size_bytes=len(content),
                    provider_etag=None,
                    status='ready',
                    created_by='test',
                    expires_at=None,
                    created_at=now,
                    updated_at=now,
                )
                session.add(artifact)
                await session.flush()
            else:
                assert artifact.producer_key == producer_key
                assert artifact.sha256 == sha256
                assert artifact.size_bytes == len(content)
            await ArtifactRepository(session).link_run_result(
                artifact_id,
                execution_id=execution_id,
                assignment_id=assignment_id,
                attempt_id=attempt_id,
                now=now,
            )
            await session.commit()
        self.contents[artifact_id] = content
        publication = {
            'execution_id': execution_id,
            'assignment_id': assignment_id,
            'attempt_id': attempt_id,
            'producer_key': producer_key,
            'content': content,
        }
        self.run_results.append(publication)
        return {
            'artifact_id': artifact_id,
            'kind': 'run_result',
            'sha256': sha256,
            'size_bytes': len(content),
        }

    async def download_artifact_to_path(
        self,
        *,
        execution_id: str,
        assignment_id: str,
        attempt_id: str,
        artifact_ref: dict,
        destination: Path,
        deadline: float | None = None,
    ) -> None:
        del execution_id, assignment_id, attempt_id
        content = self.contents[artifact_ref['artifact_id']]
        assert hashlib.sha256(content).hexdigest() == artifact_ref['sha256']
        assert len(content) == artifact_ref['size_bytes']
        destination.write_bytes(content)
        self.downloads.append(artifact_ref['artifact_id'])
        self.download_deadlines.append(deadline)


def _task_list_content(count: int = 2, *, record_version: int = 1) -> bytes:
    return b''.join(
        json.dumps(
            {
                'version': record_version,
                **({'metadata': {}} if record_version == 2 else {}),
                'item_id': f'item-{index + 1:012d}',
                'run_input': {'task_name': 'identity', 'task_config': {'index': index}},
            },
            sort_keys=True,
            separators=(',', ':'),
        ).encode()
        + b'\n'
        for index in range(count)
    )


async def _seed_environment(session, *, allocation_type: str = 'predefined') -> None:
    now = utcnow()
    access = AccessRepository(session)
    await access.upsert_user(
        principal_id=OWNER,
        issuer='tai',
        tenant='tenant',
        subject='batch-user',
        roles={'user'},
        display_name=None,
        actor='admin',
        now=now,
    )
    dynamic = allocation_type == 'dynamic'
    if dynamic:
        await ProviderConfigRepository(session).create(
            provider_config_ref='kubernetes/test',
            config={
                'provider_type': 'kubernetes',
                'namespace': 'default',
                'auth_type': 'in_cluster',
            },
            status='enabled',
            actor='admin',
            now=now,
        )
    environment_limits = {
        'manager_count': 8,
        'cpu_millicores': 16_000,
        'memory_bytes': 32 * 1024**3,
        'slots': 32,
        'active_batches': 2,
        'batch_max_inflight': 8,
        'max_items': 1000,
        'max_task_list_bytes': 1024**2,
        'max_line_bytes': 64 * 1024,
    }
    environments = EnvironmentRepository(session)
    await environments.create_environment(
        environment_id=ENVIRONMENT_ID,
        name=ENVIRONMENT_ID,
        allocation_type=allocation_type,
        provider_type='kubernetes' if dynamic else 'predefined',
        provider_config_ref='kubernetes/test' if dynamic else None,
        resource_limits=environment_limits,
        default_timeouts={
            'generation_timeout_seconds': 30,
            'queue_ttl_seconds': 120,
            'execution_ttl_seconds': 600,
            'stop_timeout_seconds': 10,
            'child_queue_ttl_seconds': 60,
            'timeout_per_run': 30,
        },
        status='enabled',
        actor='admin',
        now=now,
    )
    await environments.upsert_grant(
        principal_id=OWNER,
        environment_id=ENVIRONMENT_ID,
        limits=environment_limits,
        enabled=True,
        actor='admin',
        now=now,
    )


async def _seed_artifact(session, *, artifact_id: str, kind: str, content: bytes) -> ArtifactRef:
    now = utcnow()
    sha256 = hashlib.sha256(content).hexdigest()
    session.add(
        ArtifactRow(
            artifact_id=artifact_id,
            owner_principal_id=OWNER,
            environment_id=ENVIRONMENT_ID,
            owner_tenant='tenant',
            kind=kind,
            producer_key=f'test:{artifact_id}',
            uri=f'memory://{artifact_id}',
            sha256=sha256,
            size_bytes=len(content),
            provider_etag=None,
            status='ready',
            created_by='test',
            expires_at=None,
            created_at=now,
            updated_at=now,
        )
    )
    return ArtifactRef(artifact_id=artifact_id, kind=kind, sha256=sha256, size_bytes=len(content))


async def _bind_parent_attempt(session_factory, parent_execution_id: str) -> None:
    async with session_factory() as session:
        parent = await session.get(ExecutionRow, parent_execution_id)
        assert parent is not None
        now = utcnow()
        parent.current_assignment_id = PARENT_ASSIGNMENT_ID
        parent.current_attempt_id = PARENT_ATTEMPT_ID
        parent.status = ExecutionStatus.RUNNING
        parent.started_at = now
        parent.version += 1
        session.add(
            AssignmentRow(
                assignment_id=PARENT_ASSIGNMENT_ID,
                execution_id=parent.execution_id,
                owner_principal_id=parent.owner_principal_id,
                environment_id=parent.environment_id,
                pool_id=parent.pool_id,
                allocation_id=parent.allocation_id,
                workload=parent.workload,
                allocation_spec_hash=parent.allocation_spec_hash,
                attempt_id=PARENT_ATTEMPT_ID,
                attempt_seq=1,
                manager_id='batch-manager',
                manager_session_id='batch-manager-session',
                status='active',
                slot_cost=1,
                endpoint_ready=False,
                accepted_at=now,
                started_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()


async def _create_local_batch(
    session_factory,
    *,
    retry_failed: int = 0,
    item_count: int = 2,
    record_version: int = 1,
) -> tuple[DistributedConfig, str, FakeArtifactClient]:
    content = _task_list_content(item_count, record_version=record_version)
    async with session_factory() as session:
        await _seed_environment(session)
        task_list_ref = await _seed_artifact(
            session,
            artifact_id='artifact-task-list',
            kind='task_list',
            content=content,
        )
        await session.commit()

    submission = BatchSubmission.model_validate({
        'submission_version': 1,
        'idempotency_key': f'local-batch-{retry_failed}',
        'environment_id': ENVIRONMENT_ID,
        'code': {'mode': 'config_only'},
        'pool': 'execution',
        'generation': {
            'mode': 'local_task_list',
            'task_list': {
                'ref': task_list_ref.model_dump(mode='json'),
                'format': 'jsonl',
                'manifest': {
                    'version': 1,
                    'item_count': item_count,
                    'record_version': record_version,
                    'size_bytes': len(content),
                    'sha256': task_list_ref.sha256,
                    'source': {'mode': 'test'},
                    'datasets': [],
                },
            },
        },
        'batch': {'max_inflight': 2, 'retry_failed': retry_failed, 'fail_fast': False},
    })
    config = _dynamic_config(shard_count=8)
    async with session_factory() as session:
        result = await SubmissionRepository(session, config=config).create(
            submission,
            actor_principal_id=OWNER,
            owner_principal_id=OWNER,
            owner_tenant='tenant',
            now=utcnow(),
        )
        await session.commit()
    await _bind_parent_attempt(session_factory, result.execution.execution_id)
    return (
        config,
        result.execution.execution_id,
        FakeArtifactClient(
            session_factory,
            {task_list_ref.artifact_id: content},
        ),
    )


class MemoryBatchTaskStore:
    backend_name = 'tdsql'

    def __init__(self) -> None:
        self.records = {}
        self.put_batches = []
        self.deleted_parents = []

    async def put_records(self, records) -> None:
        self.put_batches.append(tuple(record.item_id for record in records))
        self.records.update({(record.parent_execution_id, record.item_id): record for record in records})

    async def read_record(self, *, parent_execution_id: str, item_id: str):
        return self.records[(parent_execution_id, item_id)]

    async def delete_records(self, *, parent_execution_id: str) -> None:
        self.deleted_parents.append(parent_execution_id)
        for key in [key for key in self.records if key[0] == parent_execution_id]:
            del self.records[key]

    async def close(self) -> None:
        return None


async def _create_local_v3_batch(
    session_factory,
    tmp_path: Path,
) -> tuple[DistributedConfig, str, FakeArtifactClient, bytes]:
    task_files_root = tmp_path / 'v3-task-files'
    task_files_root.mkdir()
    (task_files_root / 'input.txt').write_text('local payload')
    zip_path = tmp_path / 'v3-task-files.zip'
    write_nexus_zip_directory_to_file(task_files_root, zip_path)
    data_list_path = tmp_path / 'source-data-list.bin'
    with DataListWriter(data_list_path) as writer:
        data_ref = writer.write_frame(item_id='item-local', zip_path=zip_path)
    data_manifest = writer.result()
    data_content = data_list_path.read_bytes()
    records = [
        {
            'version': 3,
            'item_id': 'item-local',
            'run_input': {'task_name': 'identity'},
            'metadata': {'kind': 'local'},
            'data_ref': data_ref.model_dump(mode='json'),
        },
        {
            'version': 3,
            'item_id': 'item-remote',
            'run_input': {'task_name': 'identity'},
            'metadata': {'kind': 'remote'},
            'data_ref': None,
        },
    ]
    task_content = b''.join(
        json.dumps(record, sort_keys=True, separators=(',', ':')).encode() + b'\n' for record in records
    )
    async with session_factory() as session:
        await _seed_environment(session)
        task_list_ref = await _seed_artifact(
            session,
            artifact_id='artifact-v3-task-list',
            kind='task_list',
            content=task_content,
        )
        data_list_ref = await _seed_artifact(
            session,
            artifact_id='artifact-v3-data-list',
            kind='data_list',
            content=data_content,
        )
        await session.commit()

    submission = BatchSubmission.model_validate({
        'submission_version': 1,
        'idempotency_key': 'local-v3-batch',
        'environment_id': ENVIRONMENT_ID,
        'code': {'mode': 'config_only'},
        'pool': 'execution',
        'generation': {
            'mode': 'local_task_list',
            'task_list': {
                'ref': task_list_ref.model_dump(mode='json'),
                'format': 'jsonl',
                'manifest': {
                    'version': 1,
                    'record_version': 3,
                    'item_count': len(records),
                    'size_bytes': len(task_content),
                    'sha256': task_list_ref.sha256,
                    'source': {'mode': 'test'},
                    'datasets': [],
                },
            },
            'data_list': {
                'ref': data_list_ref.model_dump(mode='json'),
                'format': 'nxdl',
                'manifest': {
                    'version': 1,
                    'frame_count': data_manifest.frame_count,
                    'size_bytes': data_manifest.size_bytes,
                    'sha256': data_manifest.sha256,
                },
            },
        },
        'batch': {'max_inflight': 2, 'retry_failed': 0, 'fail_fast': False},
    })
    config = _dynamic_config(shard_count=8)
    async with session_factory() as session:
        result = await SubmissionRepository(session, config=config).create(
            submission,
            actor_principal_id=OWNER,
            owner_principal_id=OWNER,
            owner_tenant='tenant',
            now=utcnow(),
        )
        await session.commit()
    await _bind_parent_attempt(session_factory, result.execution.execution_id)
    return (
        config,
        result.execution.execution_id,
        FakeArtifactClient(
            session_factory,
            {
                task_list_ref.artifact_id: task_content,
                data_list_ref.artifact_id: data_content,
            },
        ),
        data_content,
    )


async def _create_remote_compose_batch(
    session_factory,
    *,
    idempotency_key: str,
    parts: list[dict] | None = None,
    wedata_credential: dict | None = None,
) -> tuple[DistributedConfig, str]:
    async with session_factory() as session:
        await _seed_environment(session)
        await session.commit()
    payload = {
        'submission_version': 1,
        'idempotency_key': idempotency_key,
        'environment_id': ENVIRONMENT_ID,
        'code': {'mode': 'config_only'},
        'pool': 'execution',
        'generation': {
            'mode': 'remote_materialize',
            'source': {
                'type': 'compose',
                'compose': {'parts': parts or [{'kind': 'set', 'value': 'task_name=identity', 'source': 'test'}]},
            },
        },
        'batch': {'max_inflight': 2},
    }
    if wedata_credential is not None:
        payload['data_credentials'] = {'wedata': wedata_credential}
    submission = BatchSubmission.model_validate(payload)
    config = _dynamic_config(shard_count=8)
    async with session_factory() as session:
        result = await SubmissionRepository(session, config=config).create(
            submission,
            actor_principal_id=OWNER,
            owner_principal_id=OWNER,
            owner_tenant='tenant',
            now=utcnow(),
        )
        await session.commit()
    await _bind_parent_attempt(session_factory, result.execution.execution_id)
    return config, result.execution.execution_id


async def test_remote_compose_uses_execution_scoped_wedata_credential(batch_session_factory, monkeypatch):
    config, parent_execution_id = await _create_remote_compose_batch(
        batch_session_factory,
        idempotency_key='remote-with-wedata',
        parts=[{'kind': 'data', 'value': 'research/tasks@latest', 'source': 'test'}],
        wedata_credential={'subject': 'alice', 'key': 'execution-secret', 'id': 'epoch-1'},
    )
    captured = {}
    manager_thread_id = threading.get_ident()
    credential_scan_thread_ids: list[int] = []
    original_requires_wedata = batch_control_module._compose_requires_wedata

    def capture_credential_scan_thread(compose):
        credential_scan_thread_ids.append(threading.get_ident())
        return original_requires_wedata(compose)

    async def fake_materialize(_compose, output_path, **kwargs):
        credential = kwargs['wedata_credential']
        captured['subject'] = credential['subject']
        captured['key'] = credential['key']
        content = _task_list_content(1)
        output_path.write_bytes(content)
        return MaterializedTaskList(
            path=output_path,
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            total_items=1,
        )

    monkeypatch.setattr(batch_control_module, '_materialize_compose_subprocess', fake_materialize)
    monkeypatch.setattr(batch_control_module, '_compose_requires_wedata', capture_credential_scan_thread)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=FakeArtifactClient(batch_session_factory),
    )
    try:
        await control.prepare(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
    finally:
        await control.close()

    assert captured == {'subject': 'alice', 'key': 'execution-secret'}
    assert credential_scan_thread_ids
    assert all(thread_id != manager_thread_id for thread_id in credential_scan_thread_ids)


async def test_remote_compose_does_not_fall_back_to_manager_wedata_environment(
    batch_session_factory,
    monkeypatch,
):
    config, parent_execution_id = await _create_remote_compose_batch(
        batch_session_factory,
        idempotency_key='remote-without-wedata',
        parts=[{'kind': 'data', 'value': 'research/tasks@latest', 'source': 'test'}],
    )
    monkeypatch.setenv('WEDATA_KEY', '{"subject":"host","key":"shared","id":"host"}')
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=FakeArtifactClient(batch_session_factory),
    )
    try:
        with pytest.raises(ValueError, match='batch_data_credential_required:wedata'):
            await control.prepare(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
    finally:
        await control.close()


async def test_batch_driver_leaves_summary_terminalization_to_reconciler():
    batch_control = AsyncMock()
    batch_control.reconcile_once.return_value = True
    trace_registry = AsyncMock()
    reporter = AsyncMock()
    driver = NexusBatchDriver(
        execution_id='batch-1',
        attempt_id='attempt-1',
        owner_principal_id='principal-owner-a',
        batch_control=batch_control,
        trace_registry=trace_registry,
    )

    task = asyncio.create_task(
        driver.start(
            None,
            {'batch': {'retry_failed': 0, 'max_inflight': 2, 'submit_batch_size': 2, 'fail_fast': False}},
            {},
            reporter,
        )
    )
    for _ in range(100):
        if batch_control.reconcile_once.await_count:
            break
        await asyncio.sleep(0.01)

    await asyncio.wait_for(task, timeout=2)
    reporter.complete.assert_not_awaited()
    reporter.fail.assert_not_awaited()
    batch_control.release_parent.assert_awaited_once_with('batch-1')


async def test_batch_driver_does_not_interpret_or_retry_database_errors():
    class MysqlDeadlockError(Exception):
        pass

    deadlock = MysqlDeadlockError(1213, 'Deadlock found when trying to get lock; try restarting transaction')
    batch_control = AsyncMock()
    batch_control.reconcile_once.side_effect = OperationalError('SELECT ... FOR UPDATE', {}, deadlock)
    trace_registry = AsyncMock()
    reporter = AsyncMock()
    driver = NexusBatchDriver(
        execution_id='batch-1',
        attempt_id='attempt-1',
        owner_principal_id='principal-owner-a',
        batch_control=batch_control,
        trace_registry=trace_registry,
    )

    await asyncio.wait_for(
        driver.start(
            None,
            {'batch': {'retry_failed': 0, 'max_inflight': 2, 'submit_batch_size': 2, 'fail_fast': False}},
            {},
            reporter,
        ),
        timeout=2,
    )

    batch_control.reconcile_once.assert_awaited_once()
    reporter.fail.assert_awaited_once()
    assert reporter.fail.await_args.args[0]['type'] == 'OperationalError'
    batch_control.release_parent.assert_awaited_once_with('batch-1')


async def test_batch_driver_uses_configured_reconcile_poll_interval(monkeypatch):
    batch_control = AsyncMock()
    batch_control.reconcile_once.side_effect = [False, True]
    trace_registry = AsyncMock()
    reporter = AsyncMock()
    sleep = AsyncMock()
    monkeypatch.setattr('nexus.distributed.execution.drivers.nexus_batch.asyncio.sleep', sleep)
    driver = NexusBatchDriver(
        execution_id='batch-poll',
        attempt_id='attempt-poll',
        owner_principal_id='principal-owner-a',
        batch_control=batch_control,
        trace_registry=trace_registry,
        reconcile_poll_interval=0.1,
    )

    await driver.start(
        None,
        {'batch': {'retry_failed': 0, 'max_inflight': 2, 'submit_batch_size': 2, 'fail_fast': False}},
        {},
        reporter,
    )

    sleep.assert_awaited_once_with(0.1)


async def test_batch_driver_materializes_without_runtime_trace_scope_ddl():
    calls: list[str] = []
    trace_registry = AsyncMock()
    batch_control = AsyncMock()
    batch_control.ensure_prepared.side_effect = lambda *_args, **_kwargs: calls.append('materialize')
    batch_control.reconcile_once.side_effect = lambda *_args, **_kwargs: calls.append('reconcile') or True
    reporter = AsyncMock()
    driver_config = {'run_trace_backends': {'sinks': ['tdsql']}}
    driver = NexusBatchDriver(
        execution_id='exe_0123456789abcdef0123456789abcdef',
        attempt_id='attempt-parent',
        owner_principal_id='principal-owner-a',
        batch_control=batch_control,
        trace_registry=trace_registry,
    )

    await driver.start(
        None,
        {'batch': {'max_inflight': 2, 'submit_batch_size': 2}},
        driver_config,
        reporter,
    )

    assert calls == ['materialize', 'reconcile']
    trace_registry.prepare_scope.assert_not_awaited()


async def test_batch_driver_bounds_prepare_bridge_by_generation_timeout():
    batch_control = AsyncMock()
    batch_control.reconcile_once.return_value = True
    driver = NexusBatchDriver(
        execution_id='batch-generation-deadline',
        attempt_id='attempt-generation-deadline',
        owner_principal_id='principal-owner-a',
        batch_control=batch_control,
        trace_registry=AsyncMock(),
    )

    await driver.start(
        None,
        {
            'batch': {'max_inflight': 2, 'submit_batch_size': 2},
            'generation_limits': {'timeout_seconds': 30},
        },
        {},
        AsyncMock(),
    )

    batch_control.ensure_prepared.assert_awaited_once_with(
        'batch-generation-deadline',
        parent_attempt_id='attempt-generation-deadline',
        bridge_timeout=32.0,
    )


async def test_remote_generator_timeout_kills_subprocess(tmp_path):
    generator = tmp_path / 'slow_generator.py'
    generator.write_text('import time\ndef generate_configs():\n    time.sleep(30)\n    return []\n')
    output = tmp_path / 'tasks.jsonl'

    with pytest.raises(TimeoutError, match='batch_generator_timeout'):
        await _materialize_generator_subprocess(
            generator,
            output,
            max_items=10,
            max_bytes=1024,
            timeout=1,
        )
    assert not output.exists()


async def test_local_and_remote_generator_materializers_are_byte_identical(tmp_path):
    generator = tmp_path / 'generator.py'
    generator.write_text(
        'from nexus.runner.batch import BatchItem\n'
        'from nexus.runner.single import RunInput\n'
        'def generate_configs():\n'
        '    return [\n'
        '        BatchItem(config=RunInput(task_name="echo", task_config={"message": "你好"}), run_id="case-1"),\n'
        '        BatchItem(config=RunInput(task_name="echo", task_config={"message": "second"})),\n'
        '    ]\n'
        'def get_batch_metadata():\n'
        '    return {"group": "cloud-batch", "tags": ["hyway"]}\n'
    )
    local = materialize_generator(generator, tmp_path / 'local.jsonl')
    remote = await _materialize_generator_subprocess(
        generator,
        tmp_path / 'remote' / 'tasks.jsonl',
        max_items=10,
        max_bytes=1024 * 1024,
        timeout=10,
    )

    assert remote.sha256 == local.sha256
    assert remote.size_bytes == local.size_bytes
    assert remote.total_items == local.total_items
    assert (
        remote.batch_metadata
        == local.batch_metadata
        == {
            'group': 'cloud-batch',
            'tags': ['hyway'],
        }
    )
    assert remote.path.read_bytes() == local.path.read_bytes()


async def test_local_and_remote_compose_materializers_are_byte_identical(tmp_path):
    compose = {
        'parts': [
            {'kind': 'set', 'value': 'task_name=identity', 'source': 'test'},
            {'kind': 'set', 'value': 'task_config.message=hello', 'source': 'test'},
        ],
        'limit': 1,
    }
    local = await materialize_compose(
        [RawRunPart(**part) for part in compose['parts']],
        tmp_path / 'local-compose.jsonl',
        limit=compose['limit'],
        max_items=10,
        max_bytes=1024 * 1024,
    )
    remote = await _materialize_compose_subprocess(
        compose,
        tmp_path / 'remote-compose' / 'tasks.jsonl',
        max_items=10,
        max_bytes=1024 * 1024,
        max_line_bytes=16 * 1024,
        timeout=10,
    )

    assert remote.sha256 == local.sha256
    assert remote.size_bytes == local.size_bytes
    assert remote.total_items == local.total_items
    assert remote.resolved_datasets == local.resolved_datasets
    assert remote.path.read_bytes() == local.path.read_bytes()


async def test_remote_compose_keeps_execution_credential_out_of_request_file(tmp_path, monkeypatch):
    compose = {
        'parts': [
            {'kind': 'set', 'value': 'task_name=identity', 'source': 'test'},
            {'kind': 'set', 'value': 'task_config.message=hello', 'source': 'test'},
        ],
        'limit': 1,
    }
    execution_credential = {'subject': 'alice', 'key': 'execution-secret', 'id': 'epoch-1'}
    original_create_subprocess_exec = batch_control_module.asyncio.create_subprocess_exec
    captured: dict[str, object] = {}

    async def capture_subprocess_boundary(*args, **kwargs):
        request_index = args.index('--compose-request') + 1
        request_content = Path(args[request_index]).read_bytes()
        process_environment = kwargs['env']
        captured['request'] = json.loads(request_content)
        captured['request_has_secret'] = b'execution-secret' in request_content
        captured['command_has_secret'] = any('execution-secret' in str(argument) for argument in args)
        captured['wedata_credential'] = json.loads(process_environment['WEDATA_KEY'])
        return await original_create_subprocess_exec(*args, **kwargs)

    monkeypatch.setattr(batch_control_module.asyncio, 'create_subprocess_exec', capture_subprocess_boundary)
    output = tmp_path / 'credential-compose' / 'tasks.jsonl'
    await _materialize_compose_subprocess(
        compose,
        output,
        max_items=10,
        max_bytes=1024 * 1024,
        max_line_bytes=16 * 1024,
        timeout=10,
        wedata_credential=execution_credential,
    )

    assert captured == {
        'request': {'version': 1, 'compose': compose, 'requires_wedata': False},
        'request_has_secret': False,
        'command_has_secret': False,
        'wedata_credential': execution_credential,
    }
    assert b'execution-secret' not in output.read_bytes()


async def test_remote_compose_rejects_oversized_request_before_spawning_child(tmp_path, monkeypatch):
    spawned = False

    async def fail_if_spawned(*args, **kwargs):  # noqa: ARG001
        nonlocal spawned
        spawned = True
        raise AssertionError('oversized compose request must not spawn a child')

    monkeypatch.setattr(batch_control_module.asyncio, 'create_subprocess_exec', fail_if_spawned)
    output = tmp_path / 'oversized-compose' / 'tasks.jsonl'
    compose = {
        'parts': [
            {
                'kind': 'set',
                'value': 'task_config.message=' + ('x' * (16 * 1024 * 1024)),
                'source': 'test',
            }
        ]
    }

    with pytest.raises(ValueError, match='batch compose request exceeds size limit'):
        await _materialize_compose_subprocess(
            compose,
            output,
            max_items=10,
            max_bytes=1024 * 1024,
            max_line_bytes=16 * 1024,
            timeout=10,
        )

    assert spawned is False
    assert not output.with_name(f'{output.name}.compose.json').exists()


async def test_remote_compose_rejects_local_task_files_before_publishing(tmp_path):
    files = tmp_path / 'local-task-files'
    files.mkdir()
    compose = {
        'parts': [
            {'kind': 'set', 'value': 'task_name=identity', 'source': 'test'},
            {
                'kind': 'set_json',
                'value': f'task_files={json.dumps({"type": "directory", "path": str(files)})}',
                'source': 'test',
            },
        ],
    }
    output = tmp_path / 'remote-compose' / 'tasks.jsonl'

    with pytest.raises(RuntimeError, match='local_compose_task_files_unsupported'):
        await _materialize_compose_subprocess(
            compose,
            output,
            max_items=10,
            max_bytes=1024 * 1024,
            max_line_bytes=16 * 1024,
            timeout=10,
        )

    assert not output.exists()


def test_materialize_result_v3_preserves_resolved_dataset_identity(tmp_path):
    output = tmp_path / 'tasks.jsonl'
    output.write_bytes(b'')
    result_path = tmp_path / 'result.json'
    materialized = MaterializedTaskList(
        path=output,
        sha256=hashlib.sha256(b'').hexdigest(),
        size_bytes=0,
        total_items=0,
        resolved_datasets=(
            {
                'input_uri': 'research/tasks@latest',
                'resolved_uri': 'research/tasks@2026-08-11?split=test',
            },
        ),
    )

    _write_result(result_path, materialized)
    loaded = batch_control_module._load_materialize_result(
        result_path,
        output,
        max_items=10,
        max_bytes=1024,
    )

    assert loaded.resolved_datasets == materialized.resolved_datasets


async def test_materialize_process_group_stops_after_graceful_term(monkeypatch):
    class Process:
        pid = 1234
        returncode = None

        async def wait(self):
            while self.returncode is None:
                await asyncio.sleep(0)
            return self.returncode

    process = Process()
    signals: list[int] = []

    def killpg(pid: int, sent_signal: int) -> None:
        assert pid == process.pid
        if sent_signal == 0:
            if process.returncode is not None:
                raise ProcessLookupError
            return
        signals.append(sent_signal)
        if sent_signal == signal.SIGTERM:
            process.returncode = 0

    monkeypatch.setattr(batch_control_module.os, 'killpg', killpg)

    await batch_control_module._terminate_process_group(process, grace_seconds=0.01)

    assert signals == [signal.SIGTERM]


async def test_materialize_process_group_escalates_term_to_kill(monkeypatch):
    class Process:
        pid = 1234
        returncode = None

        async def wait(self):
            while self.returncode is None:
                await asyncio.sleep(0)
            return self.returncode

    process = Process()
    signals: list[int] = []

    def killpg(pid: int, sent_signal: int) -> None:
        assert pid == process.pid
        if sent_signal == 0:
            return
        signals.append(sent_signal)
        if sent_signal == signal.SIGKILL:
            process.returncode = -signal.SIGKILL

    monkeypatch.setattr(batch_control_module.os, 'killpg', killpg)

    await batch_control_module._terminate_process_group(process, grace_seconds=0.01)

    assert signals == [signal.SIGTERM, signal.SIGKILL]


async def test_remote_generator_scrubs_host_wedata_and_only_injects_execution_credential(tmp_path, monkeypatch):
    generator = tmp_path / 'credential_generator.py'
    generator.write_text(
        'import os\n'
        'from nexus.runner.batch import BatchItem\n'
        'from nexus.runner.single import RunInput\n'
        'def generate_configs():\n'
        '    return [BatchItem(config=RunInput(task_name="identity", task_config={\n'
        '        "wedata": os.getenv("WEDATA_KEY"),\n'
        '        "wedata_file": os.getenv("WEDATA_KEY_FILE"),\n'
        '    }))]\n'
    )
    monkeypatch.setenv('WEDATA_KEY', '{"subject":"host","key":"shared","id":"host"}')
    monkeypatch.setenv('WEDATA_KEY_FILE', '/host/shared-wedata-key')

    without_credential = tmp_path / 'without-credential.jsonl'
    await _materialize_generator_subprocess(
        generator,
        without_credential,
        max_items=10,
        max_bytes=1024 * 1024,
        timeout=10,
    )
    without_task_config = json.loads(without_credential.read_text())['run_input']['task_config']
    assert without_task_config == {'wedata': None, 'wedata_file': None}

    with_credential = tmp_path / 'with-credential.jsonl'
    await _materialize_generator_subprocess(
        generator,
        with_credential,
        max_items=10,
        max_bytes=1024 * 1024,
        timeout=10,
        wedata_credential={'subject': 'alice', 'key': 'execution-secret', 'id': 'epoch-1'},
    )
    with_task_config = json.loads(with_credential.read_text())['run_input']['task_config']
    assert json.loads(with_task_config['wedata']) == {
        'subject': 'alice',
        'key': 'execution-secret',
        'id': 'epoch-1',
    }
    assert with_task_config['wedata_file'] is None


async def test_remote_generator_rejects_nonempty_metadata(tmp_path):
    generator = tmp_path / 'generator.py'
    generator.write_text(
        'from nexus.runner.batch import BatchItem\n'
        'from nexus.runner.single import RunInput\n'
        'def generate_configs():\n'
        '    return [BatchItem(config=RunInput(task_name="echo"), metadata={"group": "unsupported"})]\n'
    )
    output = tmp_path / 'tasks.jsonl'

    with pytest.raises(RuntimeError, match='task-list v1 does not support item metadata'):
        await _materialize_generator_subprocess(
            generator,
            output,
            max_items=10,
            max_bytes=1024,
            timeout=10,
        )

    assert not output.exists()


async def test_remote_generator_task_list_v2_preserves_metadata(tmp_path):
    generator = tmp_path / 'generator.py'
    generator.write_text(
        'from nexus.runner.batch import BatchItem\n'
        'from nexus.runner.single import RunInput\n'
        'def generate_configs():\n'
        '    return [BatchItem(config=RunInput(task_name="echo"), metadata={"group": "preserved"})]\n'
    )
    output = tmp_path / 'tasks.jsonl'

    result = await _materialize_generator_subprocess(
        generator,
        output,
        max_items=10,
        max_bytes=1024,
        timeout=10,
        task_list_version=2,
    )

    assert result.record_version == 2
    assert json.loads(output.read_text())['metadata'] == {'group': 'preserved'}


async def test_remote_generator_bounds_large_stdout_and_stderr(tmp_path):
    generator = tmp_path / 'generator.py'
    generator.write_text(
        'import sys\n'
        'from nexus.runner.batch import BatchItem\n'
        'from nexus.runner.single import RunInput\n'
        'def generate_configs():\n'
        '    print("o" * (2 * 1024 * 1024))\n'
        '    print("e" * (2 * 1024 * 1024), file=sys.stderr)\n'
        '    return [BatchItem(config=RunInput(task_name="echo"))]\n'
    )

    result = await _materialize_generator_subprocess(
        generator,
        tmp_path / 'tasks.jsonl',
        max_items=10,
        max_bytes=1024 * 1024,
        timeout=10,
    )

    assert result.total_items == 1


async def test_remote_generator_result_is_not_confused_by_exit_hook_output(tmp_path):
    generator = tmp_path / 'generator.py'
    generator.write_text(
        'import atexit\n'
        'from nexus.runner.batch import BatchItem\n'
        'from nexus.runner.single import RunInput\n'
        'atexit.register(lambda: print("after-result"))\n'
        'def generate_configs():\n'
        '    return [BatchItem(config=RunInput(task_name="echo"))]\n'
    )

    result = await _materialize_generator_subprocess(
        generator,
        tmp_path / 'tasks.jsonl',
        max_items=10,
        max_bytes=1024 * 1024,
        timeout=10,
    )

    assert result.total_items == 1


async def test_remote_generator_reaps_background_process_group(tmp_path):
    generator = tmp_path / 'generator.py'
    generator.write_text(
        'import subprocess\n'
        'from pathlib import Path\n'
        'from nexus.runner.batch import BatchItem\n'
        'from nexus.runner.single import RunInput\n'
        'def generate_configs():\n'
        '    child = subprocess.Popen(["sleep", "30"])\n'
        '    Path("child.pid").write_text(str(child.pid))\n'
        '    return [BatchItem(config=RunInput(task_name="echo"))]\n'
    )

    await _materialize_generator_subprocess(
        generator,
        tmp_path / 'tasks.jsonl',
        max_items=10,
        max_bytes=1024 * 1024,
        timeout=10,
    )

    child_pid = int((tmp_path / 'child.pid').read_text())
    assert not _process_is_running(child_pid)


async def test_remote_generator_nonzero_exit_cleans_partial_output(tmp_path):
    generator = tmp_path / 'generator.py'
    generator.write_text('def generate_configs():\n    raise RuntimeError("expected-generator-failure")\n')
    output = tmp_path / 'tasks.jsonl'

    with pytest.raises(RuntimeError, match='expected-generator-failure'):
        await _materialize_generator_subprocess(
            generator,
            output,
            max_items=10,
            max_bytes=1024,
            timeout=10,
        )

    assert not output.exists()


async def test_remote_generator_cancellation_reaps_process_group(tmp_path):
    generator = tmp_path / 'generator.py'
    generator.write_text(
        'import subprocess\n'
        'import time\n'
        'from pathlib import Path\n'
        'def generate_configs():\n'
        '    child = subprocess.Popen(["sleep", "30"])\n'
        '    Path("cancel-child.pid").write_text(str(child.pid))\n'
        '    time.sleep(30)\n'
        '    return []\n'
    )
    output = tmp_path / 'tasks.jsonl'
    task = asyncio.create_task(
        _materialize_generator_subprocess(
            generator,
            output,
            max_items=10,
            max_bytes=1024,
            timeout=60,
        )
    )
    pid_file = tmp_path / 'cancel-child.pid'
    for _ in range(100):
        if pid_file.exists():
            break
        await asyncio.sleep(0.02)
    assert pid_file.exists()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not output.exists()
    assert not _process_is_running(int(pid_file.read_text()))


def test_remote_generator_rejects_invalid_result_file(tmp_path):
    output = tmp_path / 'tasks.jsonl'
    output.write_bytes(b'content')
    result_file = tmp_path / 'result.json'
    result_file.write_text(json.dumps({'version': 1, 'sha256': 'a' * 64}))

    with pytest.raises(RuntimeError, match='batch_generator_invalid_result'):
        batch_control_module._load_materialize_result(
            result_file,
            output,
            max_items=10,
            max_bytes=1024,
        )


async def test_remote_generator_can_repeat_with_same_output_path(tmp_path):
    generator = tmp_path / 'generator.py'
    generator.write_text(
        'from nexus.runner.batch import BatchItem\n'
        'from nexus.runner.single import RunInput\n'
        'def generate_configs():\n'
        '    return [BatchItem(config=RunInput(task_name="echo"), run_id="repeat-1")]\n'
    )
    output = tmp_path / 'tasks.jsonl'

    first = await _materialize_generator_subprocess(
        generator,
        output,
        max_items=10,
        max_bytes=1024 * 1024,
        timeout=10,
    )
    first_content = output.read_bytes()
    second = await _materialize_generator_subprocess(
        generator,
        output,
        max_items=10,
        max_bytes=1024 * 1024,
        timeout=10,
    )

    assert second.sha256 == first.sha256
    assert output.read_bytes() == first_content
    assert not list(tmp_path.glob('*.result.json'))


def test_batch_control_generation_limits_preserve_workspace_limit():
    limits = _generation_limits({'max_workspace_bytes': 4096})

    assert limits['max_workspace_bytes'] == 4096


def test_batch_control_generation_limits_default_workspace_limit():
    limits = _generation_limits({})

    assert limits['max_workspace_bytes'] == 4 * 1024**3


@pytest.mark.parametrize(
    ('manifest_updates', 'limits', 'error'),
    [
        (
            {'item_count': 3},
            {'max_items': 2, 'max_task_list_bytes': 100, 'max_workspace_bytes': 100},
            'batch_item_limit_exceeded',
        ),
        (
            {'size_bytes': 101},
            {'max_items': 2, 'max_task_list_bytes': 100, 'max_workspace_bytes': 200},
            'batch_task_list_exceeds_byte_limit',
        ),
        (
            {'size_bytes': 101},
            {'max_items': 2, 'max_task_list_bytes': 200, 'max_workspace_bytes': 100},
            'batch_workspace_limit_exceeded',
        ),
    ],
)
def test_local_task_list_known_limits_are_rejected_during_submission(
    manifest_updates,
    limits,
    error,
):
    manifest = {
        'version': 1,
        'item_count': 2,
        'size_bytes': 100,
        'sha256': 'a' * 64,
        'source': {'mode': 'local_generator'},
        **manifest_updates,
    }
    submission = BatchSubmission.model_validate({
        'idempotency_key': 'local-admission',
        'environment_id': 'predefined',
        'code': {'mode': 'config_only'},
        'pool': 'execution',
        'generation': {
            'mode': 'local_task_list',
            'task_list': {
                'ref': {
                    'artifact_id': 'task-list',
                    'kind': 'task_list',
                    'sha256': manifest['sha256'],
                    'size_bytes': manifest['size_bytes'],
                },
                'manifest': manifest,
            },
        },
        'batch': {'max_inflight': 1},
    })

    with pytest.raises(ValueError, match=error):
        _validate_generation_admission(submission, limits)


def test_remote_compose_known_limit_is_rejected_during_submission():
    submission = BatchSubmission.model_validate({
        'idempotency_key': 'compose-admission',
        'environment_id': 'predefined',
        'code': {'mode': 'config_only'},
        'pool': 'execution',
        'generation': {
            'mode': 'remote_materialize',
            'source': {
                'type': 'compose',
                'compose': {
                    'parts': [{'kind': 'set', 'value': 'task_name=identity', 'source': 'test'}],
                    'limit': 3,
                },
            },
        },
        'batch': {'max_inflight': 1},
    })

    with pytest.raises(ValueError, match='batch_item_limit_exceeded'):
        _validate_generation_admission(
            submission,
            {'max_items': 2, 'max_task_list_bytes': 100, 'max_workspace_bytes': 100},
        )


async def test_remote_generator_rejects_workspace_over_limit(tmp_path):
    generator = tmp_path / 'large_workspace_generator.py'
    generator.write_text(
        'from pathlib import Path\n'
        'def generate_configs():\n'
        '    Path("scratch.bin").write_bytes(b"x" * 4096)\n'
        '    return []\n'
    )
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    output = output_dir / 'tasks.jsonl'

    with pytest.raises(ValueError, match='batch_generator_workspace_limit_exceeded'):
        await _materialize_generator_subprocess(
            generator,
            output,
            max_items=10,
            max_bytes=1024,
            max_workspace_bytes=1024,
            timeout=10,
        )


async def test_remote_compose_preserves_workspace_limit_error(tmp_path):
    compose = {
        'parts': [
            {'kind': 'set', 'value': 'task_name=identity', 'source': 'test'},
            {'kind': 'set', 'value': 'task_config.message=hello', 'source': 'test'},
        ],
    }

    with pytest.raises(ValueError, match='^batch_workspace_limit_exceeded$'):
        await _materialize_compose_subprocess(
            compose,
            tmp_path / 'compose-output' / 'tasks.jsonl',
            max_items=10,
            max_bytes=1024,
            max_line_bytes=1024,
            max_workspace_bytes=1,
            timeout=10,
        )


async def test_remote_generator_rejects_line_over_limit(tmp_path):
    generator = tmp_path / 'large_item_generator.py'
    generator.write_text(
        'from nexus.runner.batch import BatchItem\n'
        'from nexus.runner.single import RunInput\n'
        'def generate_configs():\n'
        '    config = RunInput(task_name="identity", task_config={"payload": "x" * 2048})\n'
        '    return [BatchItem(config=config)]\n'
    )
    output = tmp_path / 'tasks.jsonl'

    with pytest.raises(RuntimeError, match='batch task list item exceeds line size limit'):
        await _materialize_generator_subprocess(
            generator,
            output,
            max_items=10,
            max_bytes=4096,
            max_line_bytes=256,
            timeout=10,
        )
    assert not output.exists()


async def test_batch_control_uses_configured_manager_cache_for_spool(tmp_path):
    spool = tmp_path / 'manager-cache' / 'batch-control'
    control = BatchControlService(
        session_factory=None,
        config=DistributedConfig(batch_control_spool_dir=str(spool)),
        artifact_client=FakeArtifactClient(None),
    )
    try:
        assert Path(control._temporary_root.name).parent == spool
    finally:
        await control.close()


async def test_batch_control_routes_local_file_work_through_bounded_runner(
    batch_session_factory,
    monkeypatch,
):
    config, parent_execution_id, artifact_client = await _create_local_batch(batch_session_factory)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    operations: list[str] = []
    run_blocking = control._run_blocking

    async def recording_runner(operation, function, /, *args, **kwargs):
        operations.append(operation)
        return await run_blocking(operation, function, *args, **kwargs)

    monkeypatch.setattr(control, '_run_blocking', recording_runner)
    try:
        await control.ensure_prepared(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
        await control.fill_window(
            parent_execution_id,
            max_inflight=1,
            submit_batch_size=1,
            parent_attempt_id=PARENT_ATTEMPT_ID,
        )
        await control.release_parent(parent_execution_id)
    finally:
        await control.close()

    assert {
        'workspace-limit',
        'materialize-task-list-chunk',
        'task-list-fingerprint',
        'submit-window',
        'remove-parent-workspace',
    }.issubset(operations)


async def test_v3_batch_prepare_requires_task_store_before_download_or_item_creation(
    batch_session_factory,
    tmp_path,
):
    config, parent_execution_id, artifact_client, _data_content = await _create_local_v3_batch(
        batch_session_factory,
        tmp_path,
    )
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        with pytest.raises(ValueError, match='batch_task_store_required_for_data_list'):
            await control.ensure_prepared(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
    finally:
        await control.close()

    assert artifact_client.downloads == []
    async with batch_session_factory() as session:
        item_count = await session.scalar(
            select(func.count(BatchItemRow.item_id)).where(BatchItemRow.parent_execution_id == parent_execution_id)
        )
    assert item_count == 0


async def test_v3_batch_prepare_reuses_initial_data_list_index_for_materialize_and_finalize(
    batch_session_factory,
    tmp_path,
    monkeypatch,
):
    config, parent_execution_id, artifact_client, data_content = await _create_local_v3_batch(
        batch_session_factory,
        tmp_path,
    )
    task_store = MemoryBatchTaskStore()
    observed_materializing = False
    initial_scan_count = 0
    zip_validation_count = 0
    original_finalize = BatchRepository.commit_data_list_finalization

    from nexus.distributed.batch import control as control_module
    from nexus.distributed.batch import data_list as data_list_module

    original_initial_scan = control_module.scan_data_list
    original_zip_validation = data_list_module.read_zip_index_from_bytes

    def observe_initial_scan(*args, **kwargs):
        nonlocal initial_scan_count
        initial_scan_count += 1
        return original_initial_scan(*args, **kwargs)

    def observe_zip_validation(*args, **kwargs):
        nonlocal zip_validation_count
        zip_validation_count += 1
        return original_zip_validation(*args, **kwargs)

    async def observe_finalize(repository, current_parent_execution_id, **kwargs):
        nonlocal observed_materializing
        batch = await repository.session.get(BatchRunRow, current_parent_execution_id)
        assert batch is not None
        observed_materializing = batch.generation_status == 'materializing'
        await original_finalize(repository, current_parent_execution_id, **kwargs)

    monkeypatch.setattr(BatchRepository, 'commit_data_list_finalization', observe_finalize)
    monkeypatch.setattr(control_module, 'scan_data_list', observe_initial_scan)
    monkeypatch.setattr(data_list_module, 'read_zip_index_from_bytes', observe_zip_validation)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
        task_store=task_store,
    )
    try:
        task_list_path = await control.ensure_prepared(
            parent_execution_id,
            parent_attempt_id=PARENT_ATTEMPT_ID,
        )
        data_list_path = task_list_path.with_name('data_list.bin')
        assert data_list_path.read_bytes() == data_content
        assert artifact_client.downloads == ['artifact-v3-task-list', 'artifact-v3-data-list']
        assert observed_materializing
        assert initial_scan_count == 1
        assert zip_validation_count == 1
        assert {record.record_version for record in task_store.records.values()} == {3}

        async with batch_session_factory() as session:
            batch = await session.get(BatchRunRow, parent_execution_id)
            assert batch is not None
            assert batch.generation_status == 'complete'
    finally:
        await control.close()


async def test_v3_batch_prepare_rejects_data_list_change_during_initial_validation(
    batch_session_factory,
    tmp_path,
    monkeypatch,
):
    config, parent_execution_id, artifact_client, _data_content = await _create_local_v3_batch(
        batch_session_factory,
        tmp_path,
    )
    task_store = MemoryBatchTaskStore()
    original_validate = batch_control_module._validate_data_list_file

    def mutate_header_after_validation(path, **kwargs):
        index = original_validate(path, **kwargs)
        changed = bytearray(path.read_bytes())
        changed[24] ^= 1  # NXDL header reserved field; frame bytes remain unchanged.
        path.write_bytes(changed)
        return index

    monkeypatch.setattr(batch_control_module, '_validate_data_list_file', mutate_header_after_validation)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
        task_store=task_store,
    )
    try:
        with pytest.raises(DataListError) as exc_info:
            await control.ensure_prepared(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
    finally:
        await control.close()

    assert exc_info.value.code == 'invalid_batch_data_list'
    assert exc_info.value.detail == 'data-list file identity changed during validation'
    assert task_store.put_batches == []


async def test_v3_task_store_wait_does_not_hold_control_database_locks(
    batch_session_factory,
    tmp_path,
):
    control_sessions = []

    def tracked_session_factory():
        session = batch_session_factory()
        control_sessions.append(session)
        return session

    class BlockingTaskStore(MemoryBatchTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.active_control_transactions = -1

        async def put_records(self, records) -> None:
            self.active_control_transactions = sum(session.in_transaction() for session in control_sessions)
            self.started.set()
            await self.release.wait()
            await super().put_records(records)

    config, parent_execution_id, artifact_client, _data_content = await _create_local_v3_batch(
        batch_session_factory,
        tmp_path,
    )
    task_store = BlockingTaskStore()
    control = BatchControlService(
        session_factory=tracked_session_factory,
        config=config,
        artifact_client=artifact_client,
        task_store=task_store,
    )
    prepare_task = asyncio.create_task(
        control.ensure_prepared(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
    )

    async def stop_batch() -> bool:
        async with batch_session_factory() as session:
            stopped = await BatchRepository(session, config=config).begin_stop(
                parent_execution_id,
                reason='user_stop',
                now=utcnow(),
            )
            await session.commit()
            return stopped

    stop_task = None
    try:
        await asyncio.wait_for(task_store.started.wait(), timeout=5)
        assert task_store.active_control_transactions == 0
        stop_task = asyncio.create_task(stop_batch())
        assert await asyncio.wait_for(asyncio.shield(stop_task), timeout=1.5)
    finally:
        task_store.release.set()
        await asyncio.gather(prepare_task, *(task for task in (stop_task,) if task is not None), return_exceptions=True)
        await control.close()


async def test_v3_task_store_write_replays_after_stale_control_snapshot(
    batch_session_factory,
    tmp_path,
    monkeypatch,
):
    from nexus.distributed.repositories.batch import MaterializationSnapshotStaleError

    config, parent_execution_id, artifact_client, _data_content = await _create_local_v3_batch(
        batch_session_factory,
        tmp_path,
    )
    task_store = MemoryBatchTaskStore()
    original_commit = BatchRepository.commit_task_list_materialization_chunk
    reject_first_commit = True

    async def stale_once(repository, current_parent_execution_id, **kwargs):
        nonlocal reject_first_commit
        if reject_first_commit:
            reject_first_commit = False
            raise MaterializationSnapshotStaleError(current_parent_execution_id)
        return await original_commit(repository, current_parent_execution_id, **kwargs)

    monkeypatch.setattr(BatchRepository, 'commit_task_list_materialization_chunk', stale_once)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
        task_store=task_store,
    )
    try:
        await control.ensure_prepared(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
    finally:
        await control.close()

    assert task_store.put_batches == [
        ('item-local', 'item-remote'),
        ('item-local', 'item-remote'),
    ]
    assert len(task_store.records) == 2
    async with batch_session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        item_count = await session.scalar(
            select(func.count(BatchItemRow.item_id)).where(BatchItemRow.parent_execution_id == parent_execution_id)
        )
    assert batch is not None
    assert batch.generation_status == 'complete'
    assert batch.materialized_count == 2
    assert item_count == 2


async def test_v3_finalize_rejects_data_list_identity_change_without_rescanning_zip(
    batch_session_factory,
    tmp_path,
    monkeypatch,
):
    config, parent_execution_id, artifact_client, _data_content = await _create_local_v3_batch(
        batch_session_factory,
        tmp_path,
    )
    task_store = MemoryBatchTaskStore()
    original_cross_validate = batch_control_module.cross_validate_task_data_list

    def mutate_after_cross_validation(*args, **kwargs):
        item_count = original_cross_validate(*args, **kwargs)
        data_list_path = kwargs['data_list_index'].manifest.path
        changed = bytearray(data_list_path.read_bytes())
        changed[-1] ^= 1
        data_list_path.write_bytes(changed)
        return item_count

    monkeypatch.setattr(batch_control_module, 'cross_validate_task_data_list', mutate_after_cross_validation)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
        task_store=task_store,
    )
    try:
        with pytest.raises(ValueError, match='invalid_batch_data_list'):
            await control.ensure_prepared(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
    finally:
        await control.close()

    async with batch_session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
    assert batch is not None
    assert batch.generation_status == 'materializing'


async def test_v3_batch_prepare_buffers_incremental_task_records_to_the_configured_write_chunk(
    batch_session_factory,
    tmp_path,
):
    config, parent_execution_id, artifact_client, _data_content = await _create_local_v3_batch(
        batch_session_factory,
        tmp_path,
    )
    task_store = MemoryBatchTaskStore()
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
        task_store=task_store,
    )
    try:
        await control.ensure_prepared(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
    finally:
        await control.close()

    assert task_store.put_batches == [('item-local', 'item-remote')]


@pytest.mark.parametrize('record_version', [1, 2, 3])
async def test_batch_reconciler_cleans_task_records_after_terminal_finalization(
    batch_session_factory,
    tmp_path,
    record_version,
):
    if record_version == 3:
        config, parent_execution_id, artifact_client, _data_content = await _create_local_v3_batch(
            batch_session_factory, tmp_path
        )
    else:
        config, parent_execution_id, artifact_client = await _create_local_batch(
            batch_session_factory, record_version=record_version
        )
    task_store = MemoryBatchTaskStore()
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
        task_store=task_store,
    )
    reconciler = BatchLifecycleReconciler(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
        task_store=task_store,
    )
    try:
        await control.ensure_prepared(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
        await control.reconcile_once(
            parent_execution_id,
            max_inflight=2,
            submit_batch_size=2,
            parent_attempt_id=PARENT_ATTEMPT_ID,
        )
        async with batch_session_factory() as session:
            items = list(
                await session.scalars(
                    select(BatchItemRow)
                    .where(BatchItemRow.parent_execution_id == parent_execution_id)
                    .order_by(BatchItemRow.item_id)
                )
            )
            for item in items:
                child = await session.get(ExecutionRow, item.current_child_execution_id)
                assert child is not None
                child.status = ExecutionStatus.SUCCEEDED
                child.finished_at = utcnow()
                child.version += 1
            await session.commit()

        await reconciler.reconcile_once()

        assert task_store.deleted_parents == [parent_execution_id]
        assert not task_store.records
    finally:
        await reconciler.close()
        await control.close()


@pytest.mark.parametrize('failure_after_delete', [False, True])
@pytest.mark.parametrize('record_version', [1, 2, 3])
async def test_batch_reconciler_retries_cleanup_without_rolling_back_terminal_control_state(
    batch_session_factory,
    tmp_path,
    record_version,
    failure_after_delete,
):
    class FailingOnceTaskStore(MemoryBatchTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.delete_attempts = 0

        async def delete_records(self, *, parent_execution_id: str) -> None:
            self.delete_attempts += 1
            if self.delete_attempts == 1 and not failure_after_delete:
                self.deleted_parents.append(parent_execution_id)
                raise RuntimeError('injected task-store cleanup failure')
            await super().delete_records(parent_execution_id=parent_execution_id)
            if self.delete_attempts == 1:
                raise RuntimeError('injected post-delete cleanup failure')

    if record_version == 3:
        config, parent_execution_id, artifact_client, _data_content = await _create_local_v3_batch(
            batch_session_factory, tmp_path
        )
    else:
        config, parent_execution_id, artifact_client = await _create_local_batch(
            batch_session_factory, record_version=record_version
        )
    task_store = FailingOnceTaskStore()
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
        task_store=task_store,
    )
    reconciler = BatchLifecycleReconciler(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
        task_store=task_store,
    )
    try:
        await control.ensure_prepared(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
        await control.reconcile_once(
            parent_execution_id,
            max_inflight=2,
            submit_batch_size=2,
            parent_attempt_id=PARENT_ATTEMPT_ID,
        )
        async with batch_session_factory() as session:
            items = list(
                await session.scalars(
                    select(BatchItemRow).where(BatchItemRow.parent_execution_id == parent_execution_id)
                )
            )
            for item in items:
                child = await session.get(ExecutionRow, item.current_child_execution_id)
                assert child is not None
                child.status = ExecutionStatus.SUCCEEDED
                child.finished_at = utcnow()
                child.version += 1
            await session.commit()

        await reconciler.reconcile_once()

        async with batch_session_factory() as session:
            parent = await session.get(ExecutionRow, parent_execution_id)
            batch = await session.get(BatchRunRow, parent_execution_id)
        assert parent is not None
        assert parent.status == ExecutionStatus.SUCCEEDED
        assert batch is not None
        assert batch.control_state == 'finalized'
        assert task_store.delete_attempts == 1

        await reconciler.reconcile_once()

        assert task_store.delete_attempts == 2
        assert not task_store.records
    finally:
        await reconciler.close()
        await control.close()


@pytest.mark.parametrize('mutation', ['missing', 'changed'])
async def test_v3_batch_prepared_cache_redownloads_missing_or_changed_data_list_only(
    batch_session_factory,
    tmp_path,
    mutation,
):
    config, parent_execution_id, artifact_client, data_content = await _create_local_v3_batch(
        batch_session_factory,
        tmp_path,
    )
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
        task_store=MemoryBatchTaskStore(),
    )
    try:
        task_list_path = await control.ensure_prepared(
            parent_execution_id,
            parent_attempt_id=PARENT_ATTEMPT_ID,
        )
        data_list_path = task_list_path.with_name('data_list.bin')
        if mutation == 'missing':
            data_list_path.unlink()
        else:
            changed = bytearray(data_list_path.read_bytes())
            changed[-1] ^= 1
            data_list_path.write_bytes(changed)

        await control.ensure_prepared(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)

        assert data_list_path.read_bytes() == data_content
        assert artifact_client.downloads == [
            'artifact-v3-task-list',
            'artifact-v3-data-list',
            'artifact-v3-data-list',
        ]
    finally:
        await control.close()


async def test_batch_control_leaves_child_state_sync_to_reconciler(monkeypatch, batch_session_factory):
    config, parent_execution_id, artifact_client = await _create_local_batch(batch_session_factory)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )

    async def unexpected_sync(*_args, **_kwargs):
        raise AssertionError('batch manager must not synchronize child states')

    monkeypatch.setattr(BatchRepository, 'sync_child_states_page', unexpected_sync)
    try:
        assert (
            await control.reconcile_once(
                parent_execution_id,
                max_inflight=2,
                submit_batch_size=2,
                parent_attempt_id=PARENT_ATTEMPT_ID,
            )
            is False
        )
    finally:
        await control.close()


async def test_batch_reconciler_waits_for_initial_submit_window_before_sync(batch_session_factory):
    config, parent_execution_id, artifact_client = await _create_local_batch(batch_session_factory)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        await control.reconcile_once(
            parent_execution_id,
            max_inflight=2,
            submit_batch_size=1,
            parent_attempt_id=PARENT_ATTEMPT_ID,
        )
        async with batch_session_factory() as session:
            first_item = await session.get(BatchItemRow, (parent_execution_id, 'item-000000000001'))
            assert first_item is not None
            first_child = await session.get(ExecutionRow, first_item.current_child_execution_id)
            assert first_child is not None
            first_child.status = ExecutionStatus.SUCCEEDED
            first_child.finished_at = utcnow()
            first_child.version += 1
            await session.commit()

        changed = await BatchLifecycleReconciler(
            session_factory=batch_session_factory,
            config=config,
            artifact_client=artifact_client,
        ).reconcile_once()

        async with batch_session_factory() as session:
            first_item = await session.get(BatchItemRow, (parent_execution_id, 'item-000000000001'))
            batch = await session.get(BatchRunRow, parent_execution_id)
            assert first_item is not None
            assert first_item.status == 'submitted'
            assert batch is not None
            assert batch.submitted_items == 1
            assert batch.active_items == 1
            assert batch.terminal_items == 0
        assert changed == 0

        await control.reconcile_once(
            parent_execution_id,
            max_inflight=2,
            submit_batch_size=1,
            parent_attempt_id=PARENT_ATTEMPT_ID,
        )
        async with batch_session_factory() as session:
            batch = await session.get(BatchRunRow, parent_execution_id)
            assert batch is not None
            assert batch.submitted_items == 2
            assert batch.active_items == 2
    finally:
        await control.close()


async def test_local_task_list_batch_submits_children_and_reconciler_finalizes(batch_session_factory):
    config, parent_execution_id, artifact_client = await _create_local_batch(batch_session_factory)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        path = await control.prepare(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
        assert path.read_bytes() == _task_list_content()
        parent_root = Path(control._temporary_root.name) / parent_execution_id
        assert parent_root.is_dir()
        assert (
            await control.reconcile_once(
                parent_execution_id,
                max_inflight=2,
                submit_batch_size=2,
                parent_attempt_id=PARENT_ATTEMPT_ID,
            )
            is False
        )

        async with batch_session_factory() as session:
            items = list(
                await session.scalars(
                    select(BatchItemRow)
                    .where(BatchItemRow.parent_execution_id == parent_execution_id)
                    .order_by(BatchItemRow.item_id)
                )
            )
            assert [item.item_id for item in items] == ['item-000000000001', 'item-000000000002']
            assert [item.status for item in items] == ['submitted', 'submitted']
            for item in items:
                child = await session.get(ExecutionRow, item.current_child_execution_id)
                assert child is not None
                assert child.spec_type == 'nexus_single'
                assert child.workload == 'execution'
                assert child.spec_json['run_input']['task_name'] == 'identity'
                assert child.spec_json['batch']['parent_execution_id'] == parent_execution_id
                child.status = ExecutionStatus.SUCCEEDED
                child.finished_at = utcnow()
                child.version += 1
            await session.commit()

        assert (
            await control.reconcile_once(
                parent_execution_id,
                max_inflight=2,
                submit_batch_size=2,
                parent_attempt_id=PARENT_ATTEMPT_ID,
            )
            is False
        )
        await BatchLifecycleReconciler(
            session_factory=batch_session_factory,
            config=config,
            artifact_client=artifact_client,
        ).reconcile_once()
        assert (
            await control.reconcile_once(
                parent_execution_id,
                max_inflight=2,
                submit_batch_size=2,
                parent_attempt_id=PARENT_ATTEMPT_ID,
            )
            is True
        )
        assert not parent_root.exists()
        assert parent_execution_id not in control._prepared_task_lists

        async with batch_session_factory() as session:
            parent = await session.get(ExecutionRow, parent_execution_id)
            batch = await session.get(BatchRunRow, parent_execution_id)
            assert parent is not None
            assert parent.status == ExecutionStatus.SUCCEEDED
            assert parent.result_artifact_id == f'artifact-result-{PARENT_ATTEMPT_ID}'
            assert batch is not None
            assert batch.control_state == 'finalized'
            assert batch.summary_json == {
                'total': 2,
                'succeeded': 2,
                'failed': 0,
                'killed': 0,
                'lost': 0,
                'expired': 0,
                'cancelled': 0,
            }
        assert len(artifact_client.run_results) == 1
        result_document = json.loads(artifact_client.run_results[0]['content'])
        assert result_document == {
            'version': 1,
            'status': 'succeeded',
            'output': {'batch': batch.summary_json},
            'error': None,
        }
    finally:
        await control.close()


async def test_batch_reconcile_hot_path_reads_only_selected_task_records(
    batch_session_factory,
    monkeypatch,
):
    item_count = 200
    config, parent_execution_id, artifact_client = await _create_local_batch(
        batch_session_factory,
        item_count=item_count,
    )
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        path = await control.ensure_prepared(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
        expected_size = path.stat().st_size
        assert artifact_client.downloads == ['artifact-task-list']

        def unexpected_full_identity(_path: Path) -> tuple[int, str]:
            raise AssertionError('steady-state fill_window must not hash the complete task-list')

        monkeypatch.setattr(batch_control_module, '_file_identity', unexpected_full_identity)
        original_open = Path.open
        bytes_read = 0

        class CountingStream:
            def __init__(self, stream) -> None:
                self._stream = stream

            def __enter__(self):
                self._stream.__enter__()
                return self

            def __exit__(self, *args):
                return self._stream.__exit__(*args)

            def read(self, size: int = -1) -> bytes:
                nonlocal bytes_read
                content = self._stream.read(size)
                bytes_read += len(content)
                return content

            def __getattr__(self, name):
                return getattr(self._stream, name)

        def counting_open(current: Path, *args, **kwargs):
            stream = original_open(current, *args, **kwargs)
            mode = args[0] if args else kwargs.get('mode', 'r')
            if current == path and mode == 'rb':
                return CountingStream(stream)
            return stream

        monkeypatch.setattr(Path, 'open', counting_open)
        for _ in range(item_count // 10 + 1):
            assert (
                await control.reconcile_once(
                    parent_execution_id,
                    max_inflight=item_count,
                    submit_batch_size=10,
                    parent_attempt_id=PARENT_ATTEMPT_ID,
                )
                is False
            )

        assert bytes_read == expected_size
        assert artifact_client.downloads == ['artifact-task-list']
        async with batch_session_factory() as session:
            batch = await session.get(BatchRunRow, parent_execution_id)
            assert batch is not None
            assert batch.submitted_items == item_count
    finally:
        await control.close()


async def test_batch_prepared_cache_recovers_when_local_file_is_missing(batch_session_factory):
    config, parent_execution_id, artifact_client = await _create_local_batch(batch_session_factory)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        path = await control.ensure_prepared(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
        path.unlink()

        await control.reconcile_once(
            parent_execution_id,
            max_inflight=1,
            submit_batch_size=1,
            parent_attempt_id=PARENT_ATTEMPT_ID,
        )

        assert path.exists()
        assert artifact_client.downloads == ['artifact-task-list', 'artifact-task-list']
    finally:
        await control.close()


async def test_batch_prepared_cache_recovers_when_local_file_identity_changes(batch_session_factory):
    config, parent_execution_id, artifact_client = await _create_local_batch(batch_session_factory)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        path = await control.ensure_prepared(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
        original = path.read_bytes()
        path.write_bytes(b'X' + original[1:])

        await control.reconcile_once(
            parent_execution_id,
            max_inflight=1,
            submit_batch_size=1,
            parent_attempt_id=PARENT_ATTEMPT_ID,
        )

        assert path.read_bytes() == original
        assert artifact_client.downloads == ['artifact-task-list', 'artifact-task-list']
    finally:
        await control.close()


async def test_batch_prepared_cache_is_keyed_by_artifact_ref_identity(batch_session_factory):
    config, parent_execution_id, artifact_client = await _create_local_batch(batch_session_factory)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        path = await control.ensure_prepared(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
        content = path.read_bytes()
        async with batch_session_factory() as session:
            replacement = await _seed_artifact(
                session,
                artifact_id='artifact-task-list-replacement',
                kind='task_list',
                content=content,
            )
            batch = await session.get(BatchRunRow, parent_execution_id)
            assert batch is not None
            batch.task_list_ref_json = replacement.model_dump(mode='json')
            await session.commit()
        artifact_client.contents[replacement.artifact_id] = content

        await control.reconcile_once(
            parent_execution_id,
            max_inflight=1,
            submit_batch_size=1,
            parent_attempt_id=PARENT_ATTEMPT_ID,
        )

        prepared = control._prepared_task_lists[parent_execution_id]
        assert prepared.artifact_id == replacement.artifact_id
        assert artifact_client.downloads == ['artifact-task-list']
    finally:
        await control.close()


async def test_batch_manager_restart_downloads_and_revalidates_task_list(batch_session_factory):
    config, parent_execution_id, artifact_client = await _create_local_batch(batch_session_factory)
    first = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    first_path = await first.ensure_prepared(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
    await first.close()
    assert not first_path.exists()

    restarted = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        restarted_path = await restarted.ensure_prepared(
            parent_execution_id,
            parent_attempt_id=PARENT_ATTEMPT_ID,
        )
        assert restarted_path.exists()
        assert artifact_client.downloads == ['artifact-task-list', 'artifact-task-list']
    finally:
        await restarted.close()


async def test_task_list_publish_response_loss_recovers_from_atomic_gateway_link(batch_session_factory):
    config, parent_execution_id = await _create_remote_compose_batch(
        batch_session_factory,
        idempotency_key='remote-response-loss',
    )

    class CommitThenLoseResponse(FakeArtifactClient):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.lose_next_response = True

        async def publish_task_list(self, **kwargs) -> dict:
            ref = await super().publish_task_list(**kwargs)
            if self.lose_next_response:
                self.lose_next_response = False
                raise ArtifactGatewayTransientError('task-list upload response lost')
            return ref

    artifact_client = CommitThenLoseResponse(batch_session_factory, config=config)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        path = await control.ensure_prepared(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)

        assert path.exists()
        assert len(artifact_client.published) == 1
        assert artifact_client.downloads == []
        async with batch_session_factory() as session:
            batch = await session.get(BatchRunRow, parent_execution_id)
            assert batch is not None
            assert batch.generation_status == 'complete'
    finally:
        await control.close()


async def test_task_list_publish_retries_same_generation_after_precommit_transient_failure(batch_session_factory):
    config, parent_execution_id = await _create_remote_compose_batch(
        batch_session_factory,
        idempotency_key='remote-publish-retry',
    )

    class FailBeforeCommitOnce(FakeArtifactClient):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.publish_attempts = 0

        async def publish_task_list(self, **kwargs) -> dict:
            self.publish_attempts += 1
            if self.publish_attempts == 1:
                raise ArtifactGatewayTransientError('connection reset before commit')
            return await super().publish_task_list(**kwargs)

    artifact_client = FailBeforeCommitOnce(batch_session_factory, config=config)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        path = await control.ensure_prepared(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)

        assert path.exists()
        assert artifact_client.publish_attempts == 2
        assert len(artifact_client.published) == 1
        assert artifact_client.published[0]['ref']['artifact_id'].startswith('artifact-generated-')
    finally:
        await control.close()


async def test_batch_fill_window_fences_stale_parent_attempt_even_with_prepared_cache(batch_session_factory):
    config, parent_execution_id, artifact_client = await _create_local_batch(batch_session_factory)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        await control.ensure_prepared(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)

        with pytest.raises(ValueError, match='batch_parent_attempt_fenced'):
            await control.fill_window(
                parent_execution_id,
                max_inflight=1,
                submit_batch_size=1,
                parent_attempt_id='stale-attempt',
            )
    finally:
        await control.close()


async def test_batch_fill_window_allows_submit_budget_to_hold_largest_task_record(
    batch_session_factory,
    monkeypatch,
):
    config, parent_execution_id, artifact_client = await _create_local_batch(batch_session_factory)
    max_line_bytes = 256 * 1024**2
    async with batch_session_factory() as session:
        parent = await session.get(ExecutionRow, parent_execution_id)
        assert parent is not None
        parent.spec_json = {
            **parent.spec_json,
            'generation_limits': {
                **parent.spec_json['generation_limits'],
                'max_line_bytes': max_line_bytes,
            },
        }
        await session.commit()

    captured: dict[str, int] = {}
    original_submit_window = BatchRepository.submit_window

    async def capture_submit_budget(self, *args, **kwargs):
        captured['max_batch_bytes'] = kwargs['max_batch_bytes']
        return await original_submit_window(self, *args, **kwargs)

    monkeypatch.setattr(BatchRepository, 'submit_window', capture_submit_budget)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        await control.ensure_prepared(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
        await control.fill_window(
            parent_execution_id,
            max_inflight=1,
            submit_batch_size=1,
            parent_attempt_id=PARENT_ATTEMPT_ID,
        )
    finally:
        await control.close()

    assert captured['max_batch_bytes'] == max_line_bytes


async def test_batch_submission_snapshots_default_workspace_limit(batch_session_factory):
    _, parent_execution_id, _ = await _create_local_batch(batch_session_factory)

    async with batch_session_factory() as session:
        parent = await session.get(ExecutionRow, parent_execution_id)
        allocation = await session.get(EnvironmentAllocationRow, parent.allocation_id)

    assert parent.spec_json['generation_limits']['max_workspace_bytes'] == 4 * 1024**3
    assert allocation.allocation_spec_json['workspace_size_bytes'] == 4 * 1024**3


async def test_batch_parent_permanent_result_publish_error_finalizes_failed(batch_session_factory):
    class PermanentFailureArtifactClient:
        async def publish_run_result(self, **_kwargs):
            raise ValueError('artifact_producer_conflict')

    config, parent_execution_id, artifact_client = await _create_local_batch(batch_session_factory)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        await control.prepare(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
        await control.reconcile_once(
            parent_execution_id,
            max_inflight=2,
            submit_batch_size=2,
            parent_attempt_id=PARENT_ATTEMPT_ID,
        )
        async with batch_session_factory() as session:
            items = list(
                await session.scalars(
                    select(BatchItemRow).where(BatchItemRow.parent_execution_id == parent_execution_id)
                )
            )
            for item in items:
                child = await session.get(ExecutionRow, item.current_child_execution_id)
                assert child is not None
                child.status = ExecutionStatus.SUCCEEDED
                child.finished_at = utcnow()
                child.version += 1
            await session.commit()
        await control.reconcile_once(
            parent_execution_id,
            max_inflight=2,
            submit_batch_size=2,
            parent_attempt_id=PARENT_ATTEMPT_ID,
        )

        await BatchLifecycleReconciler(
            session_factory=batch_session_factory,
            config=config,
            artifact_client=PermanentFailureArtifactClient(),
        ).reconcile_once()

        async with batch_session_factory() as session:
            parent = await session.get(ExecutionRow, parent_execution_id)
            batch = await session.get(BatchRunRow, parent_execution_id)
        assert parent is not None
        assert parent.status == ExecutionStatus.FAILED
        assert parent.error_json == {
            'type': 'ResultPersistenceFailed',
            'message': 'batch run_result could not be persisted',
            'cause': 'ValueError',
        }
        assert batch is not None
        assert batch.control_state == 'finalized'
        assert batch.summary_json['succeeded'] == 2
    finally:
        await control.close()


class _TransientPublishArtifactClient:
    """Always fails ``publish_run_result`` with a transient gateway error.

    Mimics the ``503 artifact_upload_busy:queue_full`` response that
    ``ApplicationGatewayArtifactTransport`` converts into
    :class:`ArtifactGatewayTransientError` so the reconciler exercises the
    Retry-After / exponential backoff path.
    """

    def __init__(self, *, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        self.calls = 0

    async def publish_run_result(self, **_kwargs) -> dict:
        self.calls += 1
        raise ArtifactGatewayTransientError(
            'artifact_upload_busy:queue_full',
            retry_after=self.retry_after,
        )


async def _drive_batch_to_result_publication(
    batch_session_factory,
    *,
    parent_execution_id: str,
    control: BatchControlService,
) -> None:
    await control.prepare(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
    await control.reconcile_once(
        parent_execution_id,
        max_inflight=2,
        submit_batch_size=2,
        parent_attempt_id=PARENT_ATTEMPT_ID,
    )
    async with batch_session_factory() as session:
        items = list(
            await session.scalars(select(BatchItemRow).where(BatchItemRow.parent_execution_id == parent_execution_id))
        )
        for item in items:
            child = await session.get(ExecutionRow, item.current_child_execution_id)
            assert child is not None
            child.status = ExecutionStatus.SUCCEEDED
            child.finished_at = utcnow()
            child.version += 1
        await session.commit()
    await control.reconcile_once(
        parent_execution_id,
        max_inflight=2,
        submit_batch_size=2,
        parent_attempt_id=PARENT_ATTEMPT_ID,
    )


async def test_batch_lifecycle_transient_publish_defers_claim_with_retry_after(batch_session_factory):
    config, parent_execution_id, artifact_client = await _create_local_batch(batch_session_factory)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        await _drive_batch_to_result_publication(
            batch_session_factory,
            parent_execution_id=parent_execution_id,
            control=control,
        )

        transient_client = _TransientPublishArtifactClient(retry_after=2.0)
        reconciler = BatchLifecycleReconciler(
            session_factory=batch_session_factory,
            config=config,
            artifact_client=transient_client,
        )
        try:
            before = utcnow()
            await reconciler.reconcile_once()
            after = utcnow()

            assert transient_client.calls == 1

            async with batch_session_factory() as session:
                batch = await session.get(BatchRunRow, parent_execution_id)
                parent = await session.get(ExecutionRow, parent_execution_id)
            assert batch is not None
            assert parent is not None
            # A transient publish failure must never finalize the parent.
            assert batch.control_state != 'finalized'
            assert parent.status != ExecutionStatus.FAILED
            # Owner/operation cleared, lease pushed to now + retry_after.
            assert batch.lifecycle_owner_id is None
            assert batch.lifecycle_operation_id is None
            assert batch.lifecycle_lease_expires_at is not None
            expected_lower = before + timedelta(seconds=1.5)
            expected_upper = after + timedelta(seconds=2.5)
            assert expected_lower <= batch.lifecycle_lease_expires_at <= expected_upper
            # Reconciler bumped the transient attempt counter for this parent.
            assert reconciler._transient_publish_attempts[parent_execution_id] == 1

            # A follow-up tick inside the defer window must skip the publisher.
            await reconciler.reconcile_once()
            assert transient_client.calls == 1

            async with batch_session_factory() as session:
                batch_after = await session.get(BatchRunRow, parent_execution_id)
            assert batch_after is not None
            assert batch_after.lifecycle_owner_id is None
            assert batch_after.lifecycle_operation_id is None
            assert batch_after.lifecycle_lease_expires_at == batch.lifecycle_lease_expires_at
        finally:
            await reconciler.close()
    finally:
        await control.close()


async def test_batch_lifecycle_transient_publish_exponential_backoff_without_retry_after(batch_session_factory):
    config, parent_execution_id, artifact_client = await _create_local_batch(batch_session_factory)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        await _drive_batch_to_result_publication(
            batch_session_factory,
            parent_execution_id=parent_execution_id,
            control=control,
        )

        reconciler_config = _dynamic_config(
            shard_count=config.shard_count,
            batch_lifecycle_publish_backoff_base=0.5,
            batch_lifecycle_publish_backoff_max=4.0,
        )
        transient_client = _TransientPublishArtifactClient(retry_after=None)
        reconciler = BatchLifecycleReconciler(
            session_factory=batch_session_factory,
            config=reconciler_config,
            artifact_client=transient_client,
        )
        try:
            observed_delays: list[float] = []
            # Base=0.5s, cap=4s → delays should be 0.5, 1.0, 2.0, 4.0, 4.0.
            expected_delays = [0.5, 1.0, 2.0, 4.0, 4.0]
            for expected in expected_delays:
                before = utcnow()
                await reconciler.reconcile_once()
                after = utcnow()

                async with batch_session_factory() as session:
                    batch = await session.get(BatchRunRow, parent_execution_id)
                assert batch is not None
                assert batch.lifecycle_lease_expires_at is not None
                observed_delays.append((batch.lifecycle_lease_expires_at - before).total_seconds())
                lower = before + timedelta(seconds=expected - 0.1)
                upper = after + timedelta(seconds=expected + 0.5)
                assert lower <= batch.lifecycle_lease_expires_at <= upper, (
                    f'delay #{len(observed_delays)} expected≈{expected}s, '
                    f'got lease_expires_at={batch.lifecycle_lease_expires_at}, '
                    f'window=[{lower}, {upper}]'
                )

                # Force the defer window to expire so the next reconcile_once
                # can claim the parent again without wall-clock sleeping.
                async with batch_session_factory() as session:
                    batch_row = await session.get(BatchRunRow, parent_execution_id)
                    assert batch_row is not None
                    batch_row.lifecycle_lease_expires_at = utcnow() - timedelta(seconds=1)
                    await session.commit()

            assert transient_client.calls == len(expected_delays)
            assert reconciler._transient_publish_attempts[parent_execution_id] == len(expected_delays)
        finally:
            await reconciler.close()
    finally:
        await control.close()


async def test_batch_lifecycle_transient_publish_backoff_resets_after_success(batch_session_factory):
    """Attempts counter must be cleared once the parent finally publishes."""

    class _FlakyPublishArtifactClient:
        def __init__(self) -> None:
            self.calls = 0
            self.transient_calls = 0
            self.success_calls = 0

        async def publish_run_result(self, **kwargs) -> dict:
            self.calls += 1
            if self.calls <= 2:
                self.transient_calls += 1
                raise ArtifactGatewayTransientError('artifact_upload_busy:queue_full', retry_after=None)
            self.success_calls += 1
            # Delegate to the fake gateway to actually record the artifact.
            return await artifact_client.publish_run_result(**kwargs)

    config, parent_execution_id, artifact_client = await _create_local_batch(batch_session_factory)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        await _drive_batch_to_result_publication(
            batch_session_factory,
            parent_execution_id=parent_execution_id,
            control=control,
        )

        reconciler_config = _dynamic_config(
            shard_count=config.shard_count,
            batch_lifecycle_publish_backoff_base=0.5,
            batch_lifecycle_publish_backoff_max=2.0,
        )
        flaky_client = _FlakyPublishArtifactClient()
        reconciler = BatchLifecycleReconciler(
            session_factory=batch_session_factory,
            config=reconciler_config,
            artifact_client=flaky_client,
        )
        try:
            for _ in range(2):
                await reconciler.reconcile_once()
                async with batch_session_factory() as session:
                    batch_row = await session.get(BatchRunRow, parent_execution_id)
                    assert batch_row is not None
                    batch_row.lifecycle_lease_expires_at = utcnow() - timedelta(seconds=1)
                    await session.commit()

            assert flaky_client.transient_calls == 2
            assert reconciler._transient_publish_attempts[parent_execution_id] == 2

            await reconciler.reconcile_once()
            assert flaky_client.success_calls == 1
            # Success clears the transient attempts bookkeeping.
            assert parent_execution_id not in reconciler._transient_publish_attempts

            async with batch_session_factory() as session:
                batch = await session.get(BatchRunRow, parent_execution_id)
            assert batch is not None
            assert batch.control_state == 'finalized'
        finally:
            await reconciler.close()
    finally:
        await control.close()


async def test_batch_logical_retry_creates_new_child_attempt(batch_session_factory):
    config, parent_execution_id, artifact_client = await _create_local_batch(batch_session_factory, retry_failed=1)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        await control.reconcile_once(
            parent_execution_id,
            max_inflight=2,
            submit_batch_size=2,
            parent_attempt_id=PARENT_ATTEMPT_ID,
        )
        async with batch_session_factory() as session:
            items = list(
                await session.scalars(
                    select(BatchItemRow)
                    .where(BatchItemRow.parent_execution_id == parent_execution_id)
                    .order_by(BatchItemRow.item_id)
                )
            )
            first_child_id = items[0].current_child_execution_id
            first_child = await session.get(ExecutionRow, first_child_id)
            second_child = await session.get(ExecutionRow, items[1].current_child_execution_id)
            assert first_child is not None
            assert second_child is not None
            first_child.status = ExecutionStatus.FAILED
            first_child.error_json = {'type': 'TaskFailed'}
            first_child.version += 1
            second_child.status = ExecutionStatus.SUCCEEDED
            second_child.version += 1
            await session.commit()

        await BatchLifecycleReconciler(
            session_factory=batch_session_factory,
            config=config,
            artifact_client=artifact_client,
        ).reconcile_once()
        await control.reconcile_once(
            parent_execution_id,
            max_inflight=2,
            submit_batch_size=2,
            parent_attempt_id=PARENT_ATTEMPT_ID,
        )
        async with batch_session_factory() as session:
            item = await session.get(BatchItemRow, (parent_execution_id, 'item-000000000001'))
            assert item is not None
            assert item.current_attempt_no == 1
            assert item.current_child_execution_id != first_child_id
            attempts = list(
                await session.scalars(
                    select(BatchItemAttemptRow).where(
                        BatchItemAttemptRow.parent_execution_id == parent_execution_id,
                        BatchItemAttemptRow.item_id == item.item_id,
                    )
                )
            )
            assert {attempt.attempt_no for attempt in attempts} == {0, 1}
            batch = await session.get(BatchRunRow, parent_execution_id)
            assert batch is not None
            assert batch.submitted_items == 3
    finally:
        await control.close()


async def test_batch_stop_is_deferred_until_reconciler_converges_children(batch_session_factory):
    config, parent_execution_id, artifact_client = await _create_local_batch(batch_session_factory)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        await control.reconcile_once(
            parent_execution_id,
            max_inflight=1,
            submit_batch_size=1,
            parent_attempt_id=PARENT_ATTEMPT_ID,
        )
        async with batch_session_factory() as session:
            parent = await session.get(ExecutionRow, parent_execution_id)
            assert parent is not None
            parent.status = ExecutionStatus.STOPPING
            parent.stop_reason = 'user_stop'
            parent.version += 1
            await session.commit()

        await control.stop_children(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
        async with batch_session_factory() as session:
            parent = await session.get(ExecutionRow, parent_execution_id)
            batch = await session.get(BatchRunRow, parent_execution_id)
            items = list(
                await session.scalars(
                    select(BatchItemRow)
                    .where(BatchItemRow.parent_execution_id == parent_execution_id)
                    .order_by(BatchItemRow.item_id)
                )
            )
            assert parent is not None
            assert parent.status == ExecutionStatus.STOPPING
            assert batch is not None
            assert batch.control_state == 'stopping'
            assert [item.status for item in items] == ['submitted', 'cancelled']
            child = await session.get(ExecutionRow, items[0].current_child_execution_id)
            assert child is not None
            assert child.status == ExecutionStatus.KILLED
            child.status = ExecutionStatus.KILLED
            child.finished_at = utcnow()
            child.version += 1
            await session.commit()

        hook = BatchLifecycleReconciler(session_factory=batch_session_factory, config=config)
        await hook.reconcile_once()
        async with batch_session_factory() as session:
            parent = await session.get(ExecutionRow, parent_execution_id)
            batch = await session.get(BatchRunRow, parent_execution_id)
            assert parent is not None
            assert parent.status == ExecutionStatus.KILLED
            assert batch is not None
            assert batch.control_state == 'finalized'
            assert batch.summary_json['killed'] == 1
            assert batch.summary_json['cancelled'] == 1
    finally:
        await control.close()


async def test_local_task_list_download_retries_transient_gateway_failure_within_generation_deadline(
    batch_session_factory,
):
    config, parent_execution_id, artifact_client = await _create_local_batch(batch_session_factory)

    class FlakyDownloadArtifactClient(FakeArtifactClient):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.attempts = 0

        async def download_artifact_to_path(self, **kwargs) -> None:
            self.attempts += 1
            if self.attempts < 3:
                raise ArtifactGatewayTransientError('temporary task-list read failure')
            await super().download_artifact_to_path(**kwargs)

    flaky_client = FlakyDownloadArtifactClient(batch_session_factory, artifact_client.contents)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=flaky_client,
    )
    try:
        path = await control.prepare(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
        content = path.read_bytes()
    finally:
        await control.close()

    assert content == _task_list_content()
    assert flaky_client.attempts == 3
    assert flaky_client.download_deadlines
    assert all(value is not None for value in flaky_client.download_deadlines)


async def test_batch_generation_uses_one_deadline_across_download_and_chunk_persistence(
    batch_session_factory,
    monkeypatch,
):
    config, parent_execution_id, artifact_client = await _create_local_batch(batch_session_factory)
    async with batch_session_factory() as session:
        parent = await session.get(ExecutionRow, parent_execution_id)
        assert parent is not None
        spec = dict(parent.spec_json)
        generation_limits = dict(spec['generation_limits'])
        generation_limits['timeout_seconds'] = 1
        spec['generation_limits'] = generation_limits
        parent.spec_json = spec
        await session.commit()

    class SlowDownloadArtifactClient(FakeArtifactClient):
        async def download_artifact_to_path(self, **kwargs) -> None:
            await asyncio.sleep(0.6)
            await super().download_artifact_to_path(**kwargs)

    original_materialize = BatchControlService._prepare_materialize_chunk
    delay_next_chunk = True

    async def delayed_materialize(self, *args, **kwargs):
        nonlocal delay_next_chunk
        if delay_next_chunk:
            delay_next_chunk = False
            await asyncio.sleep(0.6)
        return await original_materialize(self, *args, **kwargs)

    monkeypatch.setattr(BatchControlService, '_prepare_materialize_chunk', delayed_materialize)
    slow_control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=SlowDownloadArtifactClient(batch_session_factory, artifact_client.contents),
    )
    try:
        with pytest.raises(TimeoutError, match='batch_generation_timeout'):
            await slow_control.prepare(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
    finally:
        await slow_control.close()

    async with batch_session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        assert batch is not None
        assert batch.generation_status == 'materializing'
        assert batch.materialized_count == 0
        assert batch.materialize_cursor_bytes == 0

    # The retry only verifies recovery from the partial state; give it a fresh, roomy budget
    # so slow CI hosts don't trip the 1s deadline used above.
    async with batch_session_factory() as session:
        parent = await session.get(ExecutionRow, parent_execution_id)
        assert parent is not None
        spec = dict(parent.spec_json)
        spec['generation_limits'] = {**spec['generation_limits'], 'timeout_seconds': 60}
        parent.spec_json = spec
        await session.commit()

    retry_control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        await retry_control.prepare(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
    finally:
        await retry_control.close()

    async with batch_session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        assert batch is not None
        assert batch.generation_status == 'complete'
        assert batch.materialized_count == 2


async def test_batch_generation_deadline_covers_remote_generation_and_artifact_publish(
    batch_session_factory,
    monkeypatch,
):
    async with batch_session_factory() as session:
        await _seed_environment(session)
        await session.commit()
    submission = BatchSubmission.model_validate({
        'submission_version': 1,
        'idempotency_key': 'remote-deadline',
        'environment_id': ENVIRONMENT_ID,
        'code': {'mode': 'config_only'},
        'pool': 'execution',
        'generation': {
            'mode': 'remote_materialize',
            'timeout_seconds': 1,
            'source': {
                'type': 'compose',
                'compose': {
                    'parts': [{'kind': 'set', 'value': 'task_name=identity', 'source': 'test'}],
                    'page_size': 17,
                },
            },
        },
        'batch': {'max_inflight': 2},
    })
    config = _dynamic_config(shard_count=8)
    async with batch_session_factory() as session:
        result = await SubmissionRepository(session, config=config).create(
            submission,
            actor_principal_id=OWNER,
            owner_principal_id=OWNER,
            owner_tenant='tenant',
            now=utcnow(),
        )
        await session.commit()
    parent_execution_id = result.execution.execution_id
    await _bind_parent_attempt(batch_session_factory, parent_execution_id)

    async def slow_materialize_compose(*args, **kwargs):
        assert args[0]['page_size'] == 17
        await asyncio.sleep(0.6)
        content = _task_list_content(1)
        args[1].write_bytes(content)
        return MaterializedTaskList(
            path=args[1],
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            total_items=1,
        )

    class SlowPublishArtifactClient(FakeArtifactClient):
        async def publish_task_list(self, **kwargs) -> dict:
            await asyncio.sleep(0.6)
            return await super().publish_task_list(**kwargs)

    monkeypatch.setattr(batch_control_module, '_materialize_compose_subprocess', slow_materialize_compose)
    slow_control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=SlowPublishArtifactClient(batch_session_factory),
    )
    try:
        with pytest.raises(TimeoutError, match='batch_generation_timeout'):
            await slow_control.prepare(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
    finally:
        await slow_control.close()

    async with batch_session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        assert batch is not None
        assert batch.generation_status == 'publishing'
        assert batch.task_list_manifest_json is not None
        assert batch.task_list_ref_json is None
        assert batch.materialized_count == 0

    async def fast_materialize_compose(*args, **kwargs):
        del kwargs
        content = _task_list_content(1)
        args[1].write_bytes(content)
        return MaterializedTaskList(
            path=args[1],
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            total_items=1,
        )

    monkeypatch.setattr(batch_control_module, '_materialize_compose_subprocess', fast_materialize_compose)
    retry_control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=FakeArtifactClient(batch_session_factory),
    )
    try:
        await retry_control.prepare(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
    finally:
        await retry_control.close()

    async with batch_session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        assert batch is not None
        assert batch.generation_status == 'complete'
        assert batch.materialized_count == 1


async def test_remote_generator_publishes_immutable_task_list(batch_session_factory, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager_thread_id = threading.get_ident()
    selector_digest_thread_ids: list[int] = []
    original_canonical_sha256 = batch_control_module.canonical_sha256

    def capture_selector_digest_thread(value):
        if isinstance(value, list):
            selector_digest_thread_ids.append(threading.get_ident())
        return original_canonical_sha256(value)

    monkeypatch.setattr(batch_control_module, 'canonical_sha256', capture_selector_digest_thread)
    generator = tmp_path / 'tasks.py'
    generator.write_text(
        'from nexus.runner.batch import BatchItem\n'
        'from nexus.runner.single import RunInput\n'
        'def generate_configs():\n'
        '    return [BatchItem(config=RunInput(task_name="identity"), run_id="remote-1")]\n'
        'def get_batch_metadata():\n'
        '    return {"group": "remote", "experiment": "hyway"}\n'
    )
    async with batch_session_factory() as session:
        await _seed_environment(session, allocation_type='dynamic')
        source_ref = await _seed_artifact(
            session,
            artifact_id='artifact-source',
            kind='source_package',
            content=b'source',
        )
        wheel_ref = await _seed_artifact(
            session,
            artifact_id='artifact-wheel',
            kind='nexus_wheel',
            content=b'wheel',
        )
        lock_ref = await _seed_artifact(
            session,
            artifact_id='artifact-lock',
            kind='uv_lock',
            content=b'lock',
        )
        await session.commit()

    submission = BatchSubmission.model_validate({
        'submission_version': 1,
        'idempotency_key': 'remote-generator',
        'environment_id': ENVIRONMENT_ID,
        'code': {
            'mode': 'source_package',
            'source_package': source_ref.model_dump(mode='json'),
            'nexus_wheel': wheel_ref.model_dump(mode='json'),
            'uv_lock': lock_ref.model_dump(mode='json'),
            'extras': [],
        },
        'resources': {
            'batch_manager': {'cpu_cores': 1, 'memory_gib': 1},
            'managers': {'count': 1, 'cpu_cores': 1, 'memory_gib': 1, 'slots': 1},
        },
        'generation': {
            'mode': 'remote_materialize',
            'source': {'type': 'generator', 'path': generator.name},
        },
        'batch': {'max_inflight': 1},
    })
    config = _dynamic_config(shard_count=8)
    async with batch_session_factory() as session:
        result = await SubmissionRepository(session, config=config).create(
            submission,
            actor_principal_id=OWNER,
            owner_principal_id=OWNER,
            owner_tenant='tenant',
            now=utcnow(),
        )
        await session.commit()
    parent_execution_id = result.execution.execution_id
    async with batch_session_factory() as session:
        submitted_batch = await session.get(BatchRunRow, parent_execution_id)
        assert submitted_batch is not None
        base_dependency_build_id = submitted_batch.child_dependency_build_id
        assert base_dependency_build_id is not None
    await _bind_parent_attempt(batch_session_factory, parent_execution_id)
    artifact_client = FakeArtifactClient(batch_session_factory)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        await control.prepare(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
        assert len(artifact_client.published) == 1
        assert artifact_client.published[0]['attempt_id'] == PARENT_ATTEMPT_ID
        async with batch_session_factory() as session:
            batch = await session.get(BatchRunRow, parent_execution_id)
            assert batch is not None
            assert batch.generation_status == 'complete'
            assert batch.total_items == 1
            assert batch.task_list_ref_json['kind'] == 'task_list'
            assert batch.task_list_manifest_json['item_count'] == 1
            assert batch.task_list_manifest_json['batch_metadata'] == {
                'group': 'remote',
                'experiment': 'hyway',
            }
            assert batch.dependency_status == 'building'
            assert batch.child_dependency_build_id != base_dependency_build_id
            assert batch.module_dependencies_json['representative_item_id'] == 'remote-1'
            assert batch.module_dependencies_json['selectors'] == [
                {'kind': 'object', 'object_type': 'tasks', 'value': 'identity'}
            ]
            assert selector_digest_thread_ids
            assert all(thread_id != manager_thread_id for thread_id in selector_digest_thread_ids)
            child_allocation = await session.get(
                EnvironmentAllocationRow,
                batch.child_template_json['allocation_id'],
            )
            augmented_build = await session.get(DependencyBuildRow, batch.child_dependency_build_id)
            assert child_allocation is not None
            assert augmented_build is not None
            assert child_allocation.status == 'requested'
            assert child_allocation.provider_resource_ref_json is None
            assert child_allocation.dependency_build_id == augmented_build.dependency_build_id
            assert augmented_build.build_spec_json['module_selectors'] == [
                {'kind': 'object', 'object_type': 'tasks', 'value': 'identity'}
            ]
            build_artifact_kinds = set(
                await session.scalars(
                    select(ArtifactRow.kind)
                    .join(ArtifactRefRow, ArtifactRefRow.artifact_id == ArtifactRow.artifact_id)
                    .where(
                        ArtifactRefRow.resource_type == 'dependency_build',
                        ArtifactRefRow.resource_id == augmented_build.dependency_build_id,
                    )
                )
            )
            assert build_artifact_kinds == {'source_package', 'nexus_wheel', 'uv_lock'}
            parent = await session.get(ExecutionRow, parent_execution_id)
            assert parent is not None
            assert await ExecutionPhaseRepository(session).project_many([result.execution]) == {
                parent_execution_id: 'building_dependencies'
            }
    finally:
        await control.close()


async def test_generated_task_list_staging_rejects_nondeterministic_retry(batch_session_factory):
    config, parent_execution_id, _ = await _create_local_batch(batch_session_factory)
    async with batch_session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        assert batch is not None
        manifest = dict(batch.task_list_manifest_json)
        repo = BatchRepository(session, config=config)
        await repo.stage_generated_task_list(
            parent_execution_id,
            manifest=manifest,
            generation_epoch=batch.generation_epoch,
            now=utcnow(),
        )
        with pytest.raises(ValueError, match='batch_generation_nondeterministic'):
            await repo.stage_generated_task_list(
                parent_execution_id,
                manifest={**manifest, 'sha256': 'f' * 64},
                generation_epoch=batch.generation_epoch,
                now=utcnow(),
            )


def _process_is_running(pid: int) -> bool:
    try:
        state = Path(f'/proc/{pid}/stat').read_text().split()[2]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state != 'Z'


async def test_batch_parent_attempt_is_fenced_before_materialization(batch_session_factory):
    config, parent_execution_id, artifact_client = await _create_local_batch(batch_session_factory)
    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        with pytest.raises(ValueError, match='batch_parent_attempt_fenced'):
            await control.prepare(parent_execution_id, parent_attempt_id='stale-attempt')
    finally:
        await control.close()


async def test_batch_parent_requires_current_active_assignment_before_materialization(batch_session_factory):
    config, parent_execution_id, artifact_client = await _create_local_batch(batch_session_factory)
    async with batch_session_factory() as session:
        assignment = await session.get(AssignmentRow, PARENT_ASSIGNMENT_ID)
        assert assignment is not None
        assignment.status = 'revoked'
        await session.commit()

    control = BatchControlService(
        session_factory=batch_session_factory,
        config=config,
        artifact_client=artifact_client,
    )
    try:
        with pytest.raises(ValueError, match='batch_parent_assignment_fenced'):
            await control.prepare(parent_execution_id, parent_attempt_id=PARENT_ATTEMPT_ID)
    finally:
        await control.close()
