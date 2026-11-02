from __future__ import annotations

import asyncio
import hashlib
import json
import threading

from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sqlalchemy import event, func, insert, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.dataset.zip_index import write_nexus_zip_directory_to_file
from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.batch.data_list import DataListIndex, DataListWriter, scan_data_list
from nexus.distributed.batch.task_store import decode_task_record
from nexus.distributed.config import DistributedConfig
from nexus.distributed.environment.models import EnvironmentAllocationStatus
from nexus.distributed.errors import DomainValidationError, FencingError
from nexus.distributed.execution.models import (
    AssignmentStatus,
    ExecutionMode,
    ExecutionStatus,
)
from nexus.distributed.persistence.orm import (
    ArtifactRow,
    BatchItemAttemptRow,
    BatchItemRow,
    BatchRunRow,
    DependencyBuildRow,
    EnvironmentAllocationRow,
    ExecutionRow,
    create_schema,
)
from nexus.distributed.repositories.batch import (
    BatchRepository,
    MaterializationSnapshotStaleError,
    cross_validate_task_data_list,
    encode_task_store_record,
    read_materialize_task_list_chunk,
    read_representative_run_input,
)
from nexus.distributed.repositories.execution import CreateExecution, ExecutionRepository
from nexus.distributed.repositories.quota import QuotaRepository
from nexus.distributed.submission.specs import DataListSpec
from nexus.distributed.utils.clock import utcnow


@pytest.fixture
async def session_factory(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "batch-ledger.db"}')
    await create_schema(engine)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


def _task_list(count: int) -> bytes:
    return b''.join(
        json.dumps(
            {
                'version': 1,
                'item_id': f'item-{index + 1:012d}',
                'run_input': {'task_name': 'echo', 'task_config': {'index': index}},
            },
            sort_keys=True,
            separators=(',', ':'),
        ).encode()
        + b'\n'
        for index in range(count)
    )


def _task_list_v2(metadata_items: list[dict]) -> bytes:
    return b''.join(
        json.dumps(
            {
                'version': 2,
                'item_id': f'item-{index + 1:012d}',
                'run_input': {'task_name': 'echo', 'task_config': {'index': index}},
                'metadata': metadata,
            },
            sort_keys=True,
            separators=(',', ':'),
        ).encode()
        + b'\n'
        for index, metadata in enumerate(metadata_items)
    )


def _task_list_v3(
    tmp_path: Path,
    *,
    prefix: str,
    reverse_frames: bool = False,
) -> tuple[bytes, Path, DataListSpec, DataListIndex]:
    records: list[dict] = []
    refs: dict[str, dict] = {}
    data_list_path = tmp_path / f'{prefix}.data_list.bin'
    with DataListWriter(data_list_path) as writer:
        for item_id in ('item-a', 'item-b'):
            task_files_root = tmp_path / f'{prefix}-{item_id}'
            task_files_root.mkdir()
            (task_files_root / 'input.txt').write_text(item_id)
            zip_path = tmp_path / f'{prefix}-{item_id}.zip'
            write_nexus_zip_directory_to_file(task_files_root, zip_path)
            refs[item_id] = writer.write_frame(item_id=item_id, zip_path=zip_path).model_dump(mode='json')
    data_manifest = writer.result()

    item_ids = ('item-b', 'item-a') if reverse_frames else ('item-a', 'item-b')
    for item_id in item_ids:
        records.append({
            'version': 3,
            'item_id': item_id,
            'run_input': {'task_name': 'echo', 'task_config': {'item_id': item_id}},
            'metadata': {'source_item_id': item_id},
            'data_ref': refs[item_id],
        })
    content = b''.join(json.dumps(record, sort_keys=True, separators=(',', ':')).encode() + b'\n' for record in records)
    data_list = DataListSpec.model_validate({
        'ref': {
            'artifact_id': f'artifact-{prefix}-data-list',
            'kind': 'data_list',
            'sha256': data_manifest.sha256,
            'size_bytes': data_manifest.size_bytes,
        },
        'format': 'nxdl',
        'manifest': {
            'version': 1,
            'frame_count': data_manifest.frame_count,
            'sha256': data_manifest.sha256,
            'size_bytes': data_manifest.size_bytes,
        },
    })
    return (
        content,
        data_list_path,
        data_list,
        scan_data_list(
            data_list_path,
            max_frame_bytes=256 * 1024**2,
            max_bytes=data_manifest.size_bytes,
            max_items=2,
        ),
    )


async def _materialize_chunk(
    repository: BatchRepository,
    parent_execution_id: str,
    *,
    task_list_path: Path,
    expected_ref: ArtifactRef,
    chunk_size: int,
    max_items: int,
    now,
    data_list_path: Path | None = None,
    data_list_index: DataListIndex | None = None,
    max_line_bytes: int = 16 * 1024**2,
    max_chunk_bytes: int = 16 * 1024**2,
    max_data_frame_bytes: int = 256 * 1024**2,
    max_task_record_bytes: int = 256 * 1024**2,
):
    snapshot = await repository.snapshot_task_list_materialization(
        parent_execution_id,
        expected_ref=expected_ref,
    )
    await repository.session.commit()
    chunk = read_materialize_task_list_chunk(
        task_list_path,
        cursor=snapshot.cursor_bytes,
        materialized_count=snapshot.materialized_count,
        expected_size_bytes=expected_ref.size_bytes,
        expected_record_version=snapshot.record_version,
        chunk_size=chunk_size,
        max_items=max_items,
        max_line_bytes=max_line_bytes,
        max_chunk_bytes=max_chunk_bytes,
    )
    if chunk.records and repository.task_store is not None:
        records = [
            encode_task_store_record(
                parent_execution_id,
                record,
                data_list_path=data_list_path,
                data_list_index=data_list_index,
                max_data_frame_bytes=max_data_frame_bytes,
                max_task_record_bytes=max_task_record_bytes,
            )
            for record in chunk.records
        ]
        await repository.task_store.put_records(records)
    return await repository.commit_task_list_materialization_chunk(
        parent_execution_id,
        snapshot=snapshot,
        chunk=chunk,
        now=now,
    )


async def _finalize_data_list(
    repository: BatchRepository,
    parent_execution_id: str,
    *,
    task_list_path: Path,
    expected_task_list_ref: ArtifactRef,
    data_list_index: DataListIndex,
    max_items: int,
    max_line_bytes: int,
    now,
) -> None:
    snapshot = await repository.snapshot_data_list_finalization(
        parent_execution_id,
        expected_ref=expected_task_list_ref,
    )
    await repository.session.commit()
    item_count = cross_validate_task_data_list(
        task_list_path,
        expected_task_list_ref=expected_task_list_ref,
        data_list_index=data_list_index,
        max_items=max_items,
        max_line_bytes=max_line_bytes,
    )
    await repository.commit_data_list_finalization(
        parent_execution_id,
        snapshot=snapshot,
        item_count=item_count,
        now=now,
    )


def test_representative_run_input_reads_only_first_validated_task_list_record(tmp_path):
    path = tmp_path / 'representative.jsonl'
    first = _task_list(1)
    path.write_bytes(first + b'{this later record is deliberately invalid}\n')

    representative = read_representative_run_input(path, max_line_bytes=64 * 1024)

    assert representative is not None
    assert representative.item_id == 'item-000000000001'
    assert representative.run_input == {
        'imports': [],
        'task_name': 'echo',
        'task_config': {'index': 0},
        'tool_configs': {},
        'task_files': None,
        'run_id': None,
        'runtime_provider_name': None,
        'runtime_provider_config': {},
        'runtime_configs': {},
        'agent_name': None,
        'agent_config': {},
        'openai_clients': {},
        'processors': {},
        'gateway_trajectory_observers': {},
        'runner': {
            'timeout': None,
            'hold_after_task': None,
            'hold_signal_handlers': True,
            'early_output_path': None,
            'cache': None,
            'use_gateway': False,
            'hooks': {},
            'trace_drain_error_policy': 'warn',
            'task_files_materialization': {'mode': 'none', 'target': 'host', 'timeout': 600},
        },
        'workflow_states': {},
    }


def test_representative_run_input_returns_none_for_empty_task_list(tmp_path):
    path = tmp_path / 'empty.jsonl'
    path.write_bytes(b'')

    assert read_representative_run_input(path, max_line_bytes=64 * 1024) is None


def test_representative_run_input_rejects_invalid_first_record(tmp_path):
    path = tmp_path / 'invalid-first.jsonl'
    path.write_bytes(b'not-json\n' + _task_list(1))

    with pytest.raises(DomainValidationError, match='invalid_task_list_json'):
        read_representative_run_input(path, max_line_bytes=64 * 1024)


async def test_batch_summary_exposes_persisted_control_progress():
    batch = SimpleNamespace(
        parent_execution_id='batch-1',
        generation_mode='remote_compose',
        generation_status='complete',
        materialized_count=12,
        total_items=20,
        submitted_items=10,
        active_items=4,
        terminal_items=6,
        submission_closed=False,
        control_state='stopping',
        stop_reason='user_stop',
        stop_requested_at=utcnow(),
        summary_version=3,
        summary_json={'succeeded': 6},
        task_list_ref_json={'artifact_id': 'artifact-1'},
        task_list_manifest_json={'datasets': [{'resolved_uri': 'namespace/dataset@v1'}]},
        last_progress_at=utcnow(),
    )
    session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(all=lambda: [])))
    repository = BatchRepository(session, config=DistributedConfig())  # type: ignore[arg-type]
    repository._batch_run = AsyncMock(return_value=batch)

    summary = await repository.batch_summary(batch.parent_execution_id)

    assert summary['control_state'] == 'stopping'
    assert summary['stop_reason'] == 'user_stop'
    assert summary['active_items'] == 4
    assert summary['terminal_items'] == 6
    assert summary['task_list_manifest'] == batch.task_list_manifest_json
    assert 'child_execution_status_counts' not in summary
    assert summary['last_progress_at'] == batch.last_progress_at
    session.execute.assert_not_awaited()


async def test_batch_summary_status_distribution_is_explicit_diagnostic():
    batch = SimpleNamespace(
        parent_execution_id='batch-diagnostic-summary',
        generation_mode='local_task_list',
        generation_status='complete',
        materialized_count=10,
        total_items=10,
        submitted_items=10,
        active_items=4,
        terminal_items=6,
        submission_closed=False,
        control_state='active',
        stop_reason=None,
        stop_requested_at=None,
        summary_version=0,
        summary_json=None,
        task_list_ref_json={'artifact_id': 'artifact-1'},
        task_list_manifest_json=None,
        last_progress_at=utcnow(),
    )
    session = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                SimpleNamespace(all=lambda: [('queued', 3, 2), ('running', 4, 0), ('succeeded', 6, 0)]),
                SimpleNamespace(all=lambda: [('submitted', 3), ('running', 4), ('succeeded', 6)]),
            ]
        )
    )
    repository = BatchRepository(session, config=DistributedConfig())  # type: ignore[arg-type]
    repository._batch_run = AsyncMock(return_value=batch)

    summary = await repository.batch_summary(
        batch.parent_execution_id,
        include_status_counts=True,
    )

    assert summary['child_execution_status_counts'] == {'queued': 3, 'running': 4, 'succeeded': 6}
    assert summary['child_waiting_policy_count'] == 2
    assert summary['item_status_counts'] == {'submitted': 3, 'running': 4, 'succeeded': 6}
    assert session.execute.await_count == 2


async def test_has_exhausted_failure_uses_indexed_exists_query():
    session = SimpleNamespace(scalar=AsyncMock(return_value='item-failed'))
    repository = BatchRepository(session, config=DistributedConfig())  # type: ignore[arg-type]

    assert await repository.has_exhausted_failure('batch-fail-fast')

    statement = session.scalar.await_args.args[0]
    sql = str(statement).upper()
    assert 'COUNT(' not in sql
    assert 'LIMIT' in sql


async def test_batch_parent_fence_locks_batch_before_execution_and_assignment():
    calls: list[str] = []
    batch = SimpleNamespace(active_parent_attempt_id=None)
    parent = SimpleNamespace(current_attempt_id='attempt-1', current_assignment_id='assignment-1')
    assignment = SimpleNamespace(
        execution_id='batch-lock-order',
        attempt_id='attempt-1',
        status=AssignmentStatus.ACTIVE,
    )

    class RecordingSession:
        async def scalar(self, statement):
            sql = str(statement)
            if 'FROM distributed_batch_runs' in sql:
                calls.append('batch')
                return batch
            if 'FROM executions' in sql:
                calls.append('execution')
                return parent
            if 'FROM assignments' in sql:
                calls.append('assignment')
                return assignment
            raise AssertionError(sql)

        async def flush(self):
            return None

    repository = BatchRepository(RecordingSession(), config=DistributedConfig())  # type: ignore[arg-type]
    await repository.fence_parent_attempt('batch-lock-order', 'attempt-1')

    assert calls == ['batch', 'execution', 'assignment']


async def _create_batch_run(
    session_factory,
    *,
    parent_execution_id: str,
    task_list: bytes,
    shard_id: int = 0,
) -> ArtifactRef:
    now = utcnow()
    first_record = json.loads(task_list.splitlines()[0]) if task_list else {'version': 1}
    record_version = first_record['version']
    ref = ArtifactRef(
        artifact_id=f'artifact-{parent_execution_id}',
        kind='task_list',
        sha256=hashlib.sha256(task_list).hexdigest(),
        size_bytes=len(task_list),
    )
    async with session_factory() as session:
        session.add(
            BatchRunRow(
                parent_execution_id=parent_execution_id,
                shard_id=shard_id,
                generation_mode='local_task_list',
                generation_status='materializing',
                generation_epoch=1,
                materialized_cursor='0',
                materialize_cursor_bytes=0,
                materialized_count=0,
                task_list_ref_json=ref.model_dump(mode='json'),
                task_list_manifest_json={
                    'version': 1,
                    'record_version': record_version,
                    'item_count': len(task_list.splitlines()),
                    'size_bytes': len(task_list),
                    'sha256': ref.sha256,
                    'source': {'mode': 'local_task_list'},
                    'datasets': [],
                },
                task_list_sha256=ref.sha256,
                owner_principal_id='owner',
                owner_tenant='tenant',
                total_items=len(task_list.splitlines()),
                submitted_items=0,
                active_items=0,
                terminal_items=0,
                submission_closed=False,
                control_state='active',
                summary_version=0,
                last_progress_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()
    return ref


async def test_batch_lifecycle_claim_expires_and_is_fenced_by_operation_id(session_factory):
    await _create_batch_run(session_factory, parent_execution_id='batch-claim-a', task_list=_task_list(1))
    await _create_batch_run(session_factory, parent_execution_id='batch-claim-b', task_list=_task_list(1))
    now = utcnow()

    async with session_factory() as session:
        repository = BatchRepository(session, config=DistributedConfig())
        first_claims = await repository.claim_lifecycle_parents(
            owner_id='reconciler-a',
            lease_ttl=timedelta(seconds=30),
            limit=10,
            now=now,
        )
        await session.commit()

    assert {claim.parent_execution_id for claim in first_claims} == {'batch-claim-a', 'batch-claim-b'}

    async with session_factory() as session:
        repository = BatchRepository(session, config=DistributedConfig())
        assert (
            await repository.claim_lifecycle_parents(
                owner_id='reconciler-b',
                lease_ttl=timedelta(seconds=30),
                limit=10,
                now=now + timedelta(seconds=1),
            )
            == []
        )
        await session.commit()

    async with session_factory() as session:
        repository = BatchRepository(session, config=DistributedConfig())
        second_claims = await repository.claim_lifecycle_parents(
            owner_id='reconciler-b',
            lease_ttl=timedelta(seconds=30),
            limit=10,
            now=now + timedelta(seconds=31),
        )
        await session.commit()

    assert {claim.parent_execution_id for claim in second_claims} == {'batch-claim-a', 'batch-claim-b'}
    first_by_parent = {claim.parent_execution_id: claim for claim in first_claims}
    async with session_factory() as session:
        repository = BatchRepository(session, config=DistributedConfig())
        assert not await repository.renew_lifecycle_claim(
            first_by_parent['batch-claim-a'],
            lease_ttl=timedelta(seconds=30),
            now=now + timedelta(seconds=31),
        )
        assert not await repository.release_lifecycle_claim(
            first_by_parent['batch-claim-a'],
            reconciled_at=now + timedelta(seconds=31),
        )
        await session.commit()


async def test_batch_lifecycle_claim_only_selects_owned_shards(session_factory):
    await _create_batch_run(
        session_factory,
        parent_execution_id='batch-owned-shard',
        task_list=_task_list(1),
        shard_id=2,
    )
    await _create_batch_run(
        session_factory,
        parent_execution_id='batch-other-shard',
        task_list=_task_list(1),
        shard_id=5,
    )

    async with session_factory() as session:
        claims = await BatchRepository(session, config=DistributedConfig()).claim_lifecycle_parents(
            owner_id='reconciler-a',
            lease_ttl=timedelta(seconds=30),
            limit=10,
            shard_ids=[2],
            now=utcnow(),
        )
        await session.commit()

    assert [claim.parent_execution_id for claim in claims] == ['batch-owned-shard']


async def test_batch_lifecycle_cleanup_claim_retries_all_versions_until_reconciled(session_factory):
    await _create_batch_run(
        session_factory,
        parent_execution_id='batch-finalized-v1',
        task_list=_task_list(1),
    )
    v3_task_list = (
        json.dumps({
            'version': 3,
            'item_id': 'item-v3',
            'run_input': {'task_name': 'identity'},
            'metadata': {},
            'data_ref': None,
        }).encode()
        + b'\n'
    )
    await _create_batch_run(
        session_factory,
        parent_execution_id='batch-finalized-v3',
        task_list=v3_task_list,
    )
    async with session_factory() as session:
        for parent_execution_id in ('batch-finalized-v1', 'batch-finalized-v3'):
            batch = await session.get(BatchRunRow, parent_execution_id)
            assert batch is not None
            batch.control_state = 'finalized'
            batch.last_reconciled_at = None
        await session.commit()

    async with session_factory() as session:
        claims = await BatchRepository(session, config=DistributedConfig()).claim_lifecycle_parents(
            owner_id='reconciler-a',
            lease_ttl=timedelta(seconds=30),
            limit=10,
            now=utcnow(),
        )
        await session.commit()

    assert [claim.parent_execution_id for claim in claims] == ['batch-finalized-v1', 'batch-finalized-v3']
    async with session_factory() as session:
        repo = BatchRepository(session, config=DistributedConfig())
        for claim in claims:
            assert await repo.release_lifecycle_claim(claim, reconciled_at=utcnow())
        await session.commit()
    async with session_factory() as session:
        assert not await BatchRepository(session, config=DistributedConfig()).claim_lifecycle_parents(
            owner_id='reconciler-b', lease_ttl=timedelta(seconds=30), limit=10, now=utcnow()
        )


async def test_batch_lifecycle_claim_rotates_quiet_parents_beyond_claim_limit(session_factory):
    parent_ids = ['batch-claim-a', 'batch-claim-b', 'batch-claim-c']
    for parent_execution_id in parent_ids:
        await _create_batch_run(
            session_factory,
            parent_execution_id=parent_execution_id,
            task_list=_task_list(1),
        )
    now = utcnow()
    claimed_parent_ids = []

    for offset in range(len(parent_ids)):
        async with session_factory() as session:
            repository = BatchRepository(session, config=DistributedConfig())
            claims = await repository.claim_lifecycle_parents(
                owner_id='reconciler',
                lease_ttl=timedelta(seconds=30),
                limit=1,
                now=now + timedelta(seconds=offset),
            )
            assert len(claims) == 1
            claimed_parent_ids.append(claims[0].parent_execution_id)
            assert await repository.release_lifecycle_claim(
                claims[0],
                reconciled_at=now + timedelta(seconds=offset),
            )
            await session.commit()

    assert claimed_parent_ids == parent_ids


async def _create_submittable_batch(
    session_factory,
    tmp_path,
    *,
    parent_execution_id: str,
    item_count: int,
    allocation_status: str = EnvironmentAllocationStatus.READY,
    driver_config: dict | None = None,
    task_store=None,
):
    content = _task_list(item_count)
    path = tmp_path / f'{parent_execution_id}.jsonl'
    path.write_bytes(content)
    ref = await _create_batch_run(
        session_factory,
        parent_execution_id=parent_execution_id,
        task_list=content,
    )
    now = utcnow()
    allocation_id = f'allocation-{parent_execution_id}'
    allocation_spec_hash = 'b' * 64
    child_template = {
        'mode': ExecutionMode.JOB,
        'spec_type': 'nexus_single',
        'instance_mode': 'asyncio',
        'workload': 'execution',
        'environment_id': 'environment-1',
        'pool_id': 'execution-pool',
        'allocation_id': allocation_id,
        'allocation_spec_hash': allocation_spec_hash,
        'driver_config': driver_config or {},
        'slot_cost': 1,
        'max_attempts': 2,
        'queue_ttl_seconds': 30,
        'execution_ttl_seconds': 120,
        'stop_timeout_seconds': 15,
        'spec_template': {
            'spec_version': 1,
            'code': {'mode': 'config_only'},
        },
    }
    async with session_factory() as session:
        parent = await ExecutionRepository(session, shard_count=8).create_execution(
            CreateExecution(
                execution_id=parent_execution_id,
                mode=ExecutionMode.JOB,
                spec_type='nexus_batch',
                instance_mode='asyncio',
                spec={
                    'spec_version': 1,
                    'code': {'mode': 'config_only'},
                    'batch': {
                        'child_allocation_id': allocation_id,
                        'child_allocation_spec_hash': allocation_spec_hash,
                        'child_template': child_template,
                    },
                },
                owner_principal_id='owner',
                owner_tenant='tenant',
                created_by='owner',
                environment_id='environment-1',
                pool_id='orchestration-pool',
                allocation_id=f'parent-{allocation_id}',
                workload='orchestration',
                max_attempts=2,
            ),
            now=now,
        )
        parent_row = await session.get(ExecutionRow, parent.execution_id)
        assert parent_row is not None
        parent_row.status = ExecutionStatus.RUNNING
        session.add(
            EnvironmentAllocationRow(
                allocation_id=allocation_id,
                top_level_execution_id=parent_execution_id,
                owner_principal_id='owner',
                environment_id='environment-1',
                pool_id='execution-pool',
                workload='execution',
                status=allocation_status,
                desired_manager_count=2,
                ready_manager_count=2 if allocation_status == EnvironmentAllocationStatus.READY else 0,
                manager_count=2,
                cpu_millicores=2_000,
                memory_bytes=2 * 1024**3,
                slots=4,
                active_batches=0,
                allocation_spec_json={},
                allocation_spec_hash=allocation_spec_hash,
                config_version=1,
                status_version=1,
                last_progress_at=now,
                created_at=now,
                ready_at=now if allocation_status == EnvironmentAllocationStatus.READY else None,
                updated_at=now,
            )
        )
        await session.commit()

    while True:
        async with session_factory() as session:
            repository = BatchRepository(
                session,
                config=DistributedConfig(),
                task_store=task_store,
            )
            result = await _materialize_chunk(
                repository,
                parent_execution_id,
                task_list_path=path,
                expected_ref=ref,
                chunk_size=500,
                max_items=max(1, item_count),
                now=utcnow(),
            )
            await session.commit()
        if result.complete:
            break
    return path, allocation_id


async def test_batch_child_uses_materialized_task_store_record_without_rereading_task_list(session_factory, tmp_path):
    class MemoryTaskStore:
        backend_name = 'tdsql'

        def __init__(self):
            self.records = {}
            self.put_calls = 0

        async def put_records(self, records):
            self.put_calls += 1
            self.records.update({(record.parent_execution_id, record.item_id): record for record in records})

    task_store = MemoryTaskStore()
    parent_execution_id = 'batch-task-store-reference'
    task_list_path, _ = await _create_submittable_batch(
        session_factory,
        tmp_path,
        parent_execution_id=parent_execution_id,
        item_count=1,
        task_store=task_store,
    )
    task_list_path.unlink()

    async with session_factory() as session:
        created = await BatchRepository(
            session,
            config=DistributedConfig(),
            task_store=task_store,
        ).submit_window(
            parent_execution_id,
            task_list_path=task_list_path,
            max_inflight=1,
            submit_batch_size=1,
            now=utcnow(),
        )
        await session.commit()
        child = await session.scalar(
            select(ExecutionRow).where(ExecutionRow.parent_execution_id == parent_execution_id)
        )

    assert created == 1
    assert child is not None
    assert 'run_input' not in child.spec_json
    assert child.spec_json['run_input_ref'] == {
        'version': 1,
        'store': 'tdsql',
        'parent_execution_id': parent_execution_id,
        'item_id': 'item-000000000001',
        'record_sha256': child.spec_json['run_input_ref']['record_sha256'],
    }
    stored = task_store.records[(parent_execution_id, 'item-000000000001')]
    assert stored.record_sha256 == child.spec_json['run_input_ref']['record_sha256']
    assert b'"run_input"' in stored.record_data
    assert task_store.put_calls == 1


async def test_batch_materialize_does_not_advance_control_state_when_task_store_write_fails(
    session_factory,
    tmp_path,
):
    class FailingTaskStore:
        backend_name = 'tdsql'

        async def put_records(self, records):
            assert records
            raise RuntimeError('task store unavailable')

    content = _task_list(1)
    path = tmp_path / 'task-store-failure.jsonl'
    path.write_bytes(content)
    parent_execution_id = 'batch-task-store-failure'
    ref = await _create_batch_run(
        session_factory,
        parent_execution_id=parent_execution_id,
        task_list=content,
    )

    async with session_factory() as session:
        repository = BatchRepository(
            session,
            config=DistributedConfig(),
            task_store=FailingTaskStore(),
        )
        with pytest.raises(RuntimeError, match='task store unavailable'):
            await _materialize_chunk(
                repository,
                parent_execution_id,
                task_list_path=path,
                expected_ref=ref,
                chunk_size=500,
                max_items=1,
                now=utcnow(),
            )
        await session.rollback()

    async with session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        item_count = await session.scalar(
            select(func.count(BatchItemRow.item_id)).where(BatchItemRow.parent_execution_id == parent_execution_id)
        )

    assert batch is not None
    assert batch.materialize_cursor_bytes == 0
    assert batch.materialized_count == 0
    assert item_count == 0


async def test_v3_batch_materialize_requires_task_store_before_creating_items(session_factory, tmp_path):
    parent_execution_id = 'batch-v3-task-store-required'
    content, data_list_path, _data_list, data_list_index = _task_list_v3(
        tmp_path,
        prefix=parent_execution_id,
    )
    task_list_path = tmp_path / f'{parent_execution_id}.jsonl'
    task_list_path.write_bytes(content)
    ref = await _create_batch_run(
        session_factory,
        parent_execution_id=parent_execution_id,
        task_list=content,
    )

    async with session_factory() as session:
        repository = BatchRepository(session, config=DistributedConfig())
        with pytest.raises(DomainValidationError, match='batch_task_store_required_for_data_list'):
            await _materialize_chunk(
                repository,
                parent_execution_id,
                task_list_path=task_list_path,
                expected_ref=ref,
                data_list_path=data_list_path,
                data_list_index=data_list_index,
                chunk_size=500,
                max_items=2,
                max_data_frame_bytes=256 * 1024**2,
                max_task_record_bytes=256 * 1024**2,
                now=utcnow(),
            )
        await session.rollback()

    async with session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        item_count = await session.scalar(
            select(func.count(BatchItemRow.item_id)).where(BatchItemRow.parent_execution_id == parent_execution_id)
        )

    assert batch is not None
    assert batch.generation_status == 'materializing'
    assert batch.materialized_count == 0
    assert item_count == 0


async def test_v3_batch_materialize_writes_nxbr_then_finalizes_after_cross_scan(session_factory, tmp_path):
    class MemoryTaskStore:
        backend_name = 'tdsql'

        def __init__(self):
            self.records = {}

        async def put_records(self, records):
            self.records.update({(record.parent_execution_id, record.item_id): record for record in records})

    task_store = MemoryTaskStore()
    parent_execution_id = 'batch-v3-cross-scan'
    content, data_list_path, _data_list, data_list_index = _task_list_v3(
        tmp_path,
        prefix=parent_execution_id,
    )
    task_list_path = tmp_path / f'{parent_execution_id}.jsonl'
    task_list_path.write_bytes(content)
    ref = await _create_batch_run(
        session_factory,
        parent_execution_id=parent_execution_id,
        task_list=content,
    )

    async with session_factory() as session:
        repository = BatchRepository(session, config=DistributedConfig(), task_store=task_store)
        result = await _materialize_chunk(
            repository,
            parent_execution_id,
            task_list_path=task_list_path,
            expected_ref=ref,
            data_list_path=data_list_path,
            data_list_index=data_list_index,
            chunk_size=500,
            max_items=2,
            max_data_frame_bytes=256 * 1024**2,
            max_task_record_bytes=256 * 1024**2,
            now=utcnow(),
        )
        await session.commit()

    assert result.complete
    assert len(task_store.records) == 2
    stored = task_store.records[(parent_execution_id, 'item-a')]
    decoded = decode_task_record(
        stored,
        expected_sha256=stored.record_sha256,
        max_record_bytes=256 * 1024**2,
        max_frame_bytes=256 * 1024**2,
    )
    assert stored.record_version == 3
    assert decoded.identity['data_ref'] is not None
    assert decoded.frame is not None
    assert decoded.frame.item_id == 'item-a'
    assert decoded.frame.zip_bytes is not None

    async with session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        assert batch is not None
        assert batch.generation_status == 'materializing'
        repository = BatchRepository(session, config=DistributedConfig(), task_store=task_store)
        await _finalize_data_list(
            repository,
            parent_execution_id,
            task_list_path=task_list_path,
            expected_task_list_ref=ref,
            data_list_index=data_list_index,
            max_items=2,
            max_line_bytes=16 * 1024**2,
            now=utcnow(),
        )
        await session.commit()

    async with session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        assert batch is not None
        assert batch.generation_status == 'complete'


async def test_v3_batch_cross_scan_rejects_frame_order_before_generation_complete(session_factory, tmp_path):
    class MemoryTaskStore:
        backend_name = 'tdsql'

        async def put_records(self, _records):
            return None

    task_store = MemoryTaskStore()
    parent_execution_id = 'batch-v3-frame-order'
    content, data_list_path, _data_list, data_list_index = _task_list_v3(
        tmp_path,
        prefix=parent_execution_id,
        reverse_frames=True,
    )
    task_list_path = tmp_path / f'{parent_execution_id}.jsonl'
    task_list_path.write_bytes(content)
    ref = await _create_batch_run(
        session_factory,
        parent_execution_id=parent_execution_id,
        task_list=content,
    )

    async with session_factory() as session:
        repository = BatchRepository(session, config=DistributedConfig(), task_store=task_store)
        result = await _materialize_chunk(
            repository,
            parent_execution_id,
            task_list_path=task_list_path,
            expected_ref=ref,
            data_list_path=data_list_path,
            data_list_index=data_list_index,
            chunk_size=500,
            max_items=2,
            max_data_frame_bytes=256 * 1024**2,
            max_task_record_bytes=256 * 1024**2,
            now=utcnow(),
        )
        assert result.complete
        await session.commit()

    async with session_factory() as session:
        repository = BatchRepository(session, config=DistributedConfig(), task_store=task_store)
        with pytest.raises(DomainValidationError, match='batch_task_list_data_ref_invalid'):
            await _finalize_data_list(
                repository,
                parent_execution_id,
                task_list_path=task_list_path,
                expected_task_list_ref=ref,
                data_list_index=data_list_index,
                max_items=2,
                max_line_bytes=16 * 1024**2,
                now=utcnow(),
            )
        await session.rollback()

    async with session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        child_count = await session.scalar(
            select(func.count(ExecutionRow.execution_id)).where(ExecutionRow.parent_execution_id == parent_execution_id)
        )
        assert batch is not None
        assert batch.generation_status == 'materializing'
        assert child_count == 0


async def test_v3_batch_cross_scan_rejects_same_size_task_list_identity_change(session_factory, tmp_path):
    class MemoryTaskStore:
        backend_name = 'tdsql'

        async def put_records(self, _records):
            return None

    task_store = MemoryTaskStore()
    parent_execution_id = 'batch-v3-task-list-identity'
    content, data_list_path, _data_list, data_list_index = _task_list_v3(
        tmp_path,
        prefix=parent_execution_id,
    )
    task_list_path = tmp_path / f'{parent_execution_id}.jsonl'
    task_list_path.write_bytes(content)
    ref = await _create_batch_run(
        session_factory,
        parent_execution_id=parent_execution_id,
        task_list=content,
    )

    async with session_factory() as session:
        repository = BatchRepository(
            session,
            config=DistributedConfig(),
            task_store=task_store,
        )
        result = await _materialize_chunk(
            repository,
            parent_execution_id,
            task_list_path=task_list_path,
            expected_ref=ref,
            data_list_path=data_list_path,
            data_list_index=data_list_index,
            chunk_size=500,
            max_items=2,
            max_data_frame_bytes=256 * 1024**2,
            max_task_record_bytes=256 * 1024**2,
            now=utcnow(),
        )
        assert result.complete
        await session.commit()

    changed = task_list_path.read_bytes().replace(b'"source_item_id":"item-a"', b'"source_item_id":"item-x"')
    assert len(changed) == len(content)
    assert changed != content
    task_list_path.write_bytes(changed)

    async with session_factory() as session:
        repository = BatchRepository(session, config=DistributedConfig(), task_store=task_store)
        with pytest.raises(DomainValidationError, match='batch_task_list_identity_mismatch'):
            await _finalize_data_list(
                repository,
                parent_execution_id,
                task_list_path=task_list_path,
                expected_task_list_ref=ref,
                data_list_index=data_list_index,
                max_items=2,
                max_line_bytes=16 * 1024**2,
                now=utcnow(),
            )
        await session.rollback()

    async with session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        assert batch is not None
        assert batch.generation_status == 'materializing'


async def test_batch_children_inherit_parent_run_trace_backends(
    session_factory, tmp_path, monkeypatch, otel_hyway_instrumentor
):
    driver_config = {'run_trace_backends': {'sinks': ['tdsql-team-a', 'cos-archive']}}
    parent_carrier = {
        'traceparent': '00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01',
        'tracestate': 'idempotency-key=parent-request,vendor=preserved',
    }
    monkeypatch.setattr('nexus.distributed.repositories.execution.inject_carrier', lambda: parent_carrier)
    parent_execution_id = 'batch-trace-backends'
    task_list_path, _ = await _create_submittable_batch(
        session_factory,
        tmp_path,
        parent_execution_id=parent_execution_id,
        item_count=1,
        driver_config=driver_config,
    )

    async with session_factory() as session:
        created = await BatchRepository(session, config=DistributedConfig()).submit_window(
            parent_execution_id,
            task_list_path=task_list_path,
            max_inflight=1,
            submit_batch_size=1,
            now=utcnow(),
        )
        await session.commit()

        child = await session.scalar(
            select(ExecutionRow).where(ExecutionRow.parent_execution_id == parent_execution_id)
        )

    assert created == 1
    assert child is not None
    assert child.driver_config_json == driver_config
    assert child.trace_carrier_json == {
        'traceparent': parent_carrier['traceparent'],
        'tracestate': 'hyway_instance_id=item-000000000001,idempotency-key=parent-request,vendor=preserved',
    }


async def test_batch_materialization_is_chunked_and_resumes_from_byte_cursor(session_factory, tmp_path):
    content = _task_list(1_201)
    path = tmp_path / 'tasks.jsonl'
    path.write_bytes(content)
    ref = await _create_batch_run(session_factory, parent_execution_id='batch-chunked', task_list=content)

    inserted_per_chunk = []
    while True:
        async with session_factory() as session:
            repository = BatchRepository(session, config=DistributedConfig())
            result = await _materialize_chunk(
                repository,
                'batch-chunked',
                task_list_path=path,
                expected_ref=ref,
                chunk_size=500,
                max_items=2_000,
                now=utcnow(),
            )
            await session.commit()
        inserted_per_chunk.append(result.inserted)
        if result.complete:
            break

    assert inserted_per_chunk == [500, 500, 201]
    async with session_factory() as session:
        batch = await session.get(BatchRunRow, 'batch-chunked')
        count = await session.scalar(
            select(func.count(BatchItemRow.item_id)).where(BatchItemRow.parent_execution_id == 'batch-chunked')
        )
        first = await session.get(
            BatchItemRow,
            {'parent_execution_id': 'batch-chunked', 'item_id': 'item-000000000001'},
        )

    assert batch is not None
    assert batch.generation_status == 'complete'
    assert batch.materialize_cursor_bytes == len(content)
    assert batch.materialized_count == 1_201
    assert count == 1_201
    assert first is not None
    assert first.task_offset == 0
    assert first.task_length > 0
    assert first.status_version == 1


async def test_batch_materialization_stale_snapshot_cannot_advance_cursor_twice(session_factory, tmp_path):
    content = _task_list(2)
    path = tmp_path / 'stale-snapshot-tasks.jsonl'
    path.write_bytes(content)
    parent_execution_id = 'batch-stale-materialization-snapshot'
    ref = await _create_batch_run(
        session_factory,
        parent_execution_id=parent_execution_id,
        task_list=content,
    )

    async with session_factory() as session:
        first_snapshot = await BatchRepository(session, config=DistributedConfig()).snapshot_task_list_materialization(
            parent_execution_id,
            expected_ref=ref,
        )
        await session.commit()
    async with session_factory() as session:
        second_snapshot = await BatchRepository(session, config=DistributedConfig()).snapshot_task_list_materialization(
            parent_execution_id,
            expected_ref=ref,
        )
        await session.commit()

    chunk = read_materialize_task_list_chunk(
        path,
        cursor=first_snapshot.cursor_bytes,
        materialized_count=first_snapshot.materialized_count,
        expected_size_bytes=ref.size_bytes,
        expected_record_version=first_snapshot.record_version,
        chunk_size=1,
        max_items=10,
        max_line_bytes=16 * 1024**2,
        max_chunk_bytes=16 * 1024**2,
    )
    async with session_factory() as session:
        await BatchRepository(session, config=DistributedConfig()).commit_task_list_materialization_chunk(
            parent_execution_id,
            snapshot=first_snapshot,
            chunk=chunk,
            now=utcnow(),
        )
        await session.commit()

    async with session_factory() as session:
        with pytest.raises(MaterializationSnapshotStaleError):
            await BatchRepository(session, config=DistributedConfig()).commit_task_list_materialization_chunk(
                parent_execution_id,
                snapshot=second_snapshot,
                chunk=chunk,
                now=utcnow(),
            )
        await session.rollback()

    async with session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        item_count = await session.scalar(
            select(func.count(BatchItemRow.item_id)).where(BatchItemRow.parent_execution_id == parent_execution_id)
        )
    assert batch is not None
    assert batch.materialized_count == 1
    assert 0 < batch.materialize_cursor_bytes < len(content)
    assert item_count == 1


async def test_batch_materialization_v2_persists_item_metadata(session_factory, tmp_path):
    metadata_items = [
        {'category': 'terminal', 'tags': ['dataset']},
        {'category': 'terminal', 'difficulty': 'hard'},
    ]
    content = _task_list_v2(metadata_items)
    path = tmp_path / 'tasks-v2.jsonl'
    path.write_bytes(content)
    ref = await _create_batch_run(
        session_factory,
        parent_execution_id='batch-v2-metadata',
        task_list=content,
    )

    async with session_factory() as session:
        repository = BatchRepository(session, config=DistributedConfig())
        result = await _materialize_chunk(
            repository,
            'batch-v2-metadata',
            task_list_path=path,
            expected_ref=ref,
            chunk_size=500,
            max_items=10,
            now=utcnow(),
        )
        await session.commit()

    assert result.inserted == 2
    async with session_factory() as session:
        rows = (
            await session.scalars(
                select(BatchItemRow)
                .where(BatchItemRow.parent_execution_id == 'batch-v2-metadata')
                .order_by(BatchItemRow.task_ordinal)
            )
        ).all()

    assert [row.metadata_json for row in rows] == metadata_items


async def test_batch_materialization_rejects_record_version_not_declared_by_manifest(
    session_factory,
    tmp_path,
):
    content = _task_list_v2([{'category': 'terminal'}])
    path = tmp_path / 'tasks-version-mismatch.jsonl'
    path.write_bytes(content)
    ref = await _create_batch_run(
        session_factory,
        parent_execution_id='batch-version-mismatch',
        task_list=content,
    )
    async with session_factory() as session:
        batch = await session.get(BatchRunRow, 'batch-version-mismatch')
        assert batch is not None
        batch.task_list_manifest_json = {**batch.task_list_manifest_json, 'record_version': 1}
        await session.commit()

    async with session_factory() as session:
        repository = BatchRepository(session, config=DistributedConfig())
        with pytest.raises(DomainValidationError) as exc_info:
            await _materialize_chunk(
                repository,
                'batch-version-mismatch',
                task_list_path=path,
                expected_ref=ref,
                chunk_size=500,
                max_items=10,
                now=utcnow(),
            )
    assert exc_info.value.code == 'batch_task_list_record_version_mismatch'


async def test_batch_materialization_page_is_bounded_by_raw_record_bytes(session_factory, tmp_path):
    content = _task_list(3)
    lines = content.splitlines(keepends=True)
    path = tmp_path / 'byte-bounded-tasks.jsonl'
    path.write_bytes(content)
    ref = await _create_batch_run(
        session_factory,
        parent_execution_id='batch-byte-bounded',
        task_list=content,
    )

    async with session_factory() as session:
        repository = BatchRepository(session, config=DistributedConfig())
        result = await _materialize_chunk(
            repository,
            'batch-byte-bounded',
            task_list_path=path,
            expected_ref=ref,
            chunk_size=500,
            max_chunk_bytes=len(lines[0]) + 1,
            max_items=10,
            now=utcnow(),
        )
        await session.commit()

    assert result.inserted == 1
    assert result.cursor_bytes == len(lines[0])
    assert result.complete is False


async def test_batch_materialization_reads_existing_items_with_bounded_selects(session_factory, tmp_path):
    content = _task_list(500)
    path = tmp_path / 'bounded-materialization.jsonl'
    path.write_bytes(content)
    ref = await _create_batch_run(
        session_factory,
        parent_execution_id='batch-materialize-bulk',
        task_list=content,
    )
    statements: list[str] = []
    engine = session_factory.kw['bind']

    def record_statement(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, 'before_cursor_execute', record_statement)
    try:
        async with session_factory() as session:
            repository = BatchRepository(session, config=DistributedConfig())
            result = await _materialize_chunk(
                repository,
                'batch-materialize-bulk',
                task_list_path=path,
                expected_ref=ref,
                chunk_size=500,
                max_items=500,
                now=utcnow(),
            )
            await session.commit()
    finally:
        event.remove(engine.sync_engine, 'before_cursor_execute', record_statement)

    assert result.inserted == 500
    selects = [statement for statement in statements if statement.lstrip().lower().startswith('select')]
    assert len(selects) <= 3, len(selects)


async def test_batch_submit_window_waits_for_ready_child_allocation_and_inherits_server_template(
    session_factory,
    tmp_path,
):
    parent_execution_id = 'batch-submit-ready'
    path, allocation_id = await _create_submittable_batch(
        session_factory,
        tmp_path,
        parent_execution_id=parent_execution_id,
        item_count=3,
        allocation_status=EnvironmentAllocationStatus.REQUESTED,
    )

    async with session_factory() as session:
        repo = BatchRepository(session, config=DistributedConfig())
        assert (
            await repo.submit_window(
                parent_execution_id,
                task_list_path=path,
                max_inflight=3,
                submit_batch_size=50,
                now=utcnow(),
            )
            == 0
        )
        await session.commit()

    async with session_factory() as session:
        allocation = await session.get(EnvironmentAllocationRow, allocation_id)
        assert allocation is not None
        allocation.status = EnvironmentAllocationStatus.READY
        allocation.ready_manager_count = allocation.desired_manager_count
        allocation.ready_at = utcnow()
        await session.commit()

    async with session_factory() as session:
        repo = BatchRepository(session, config=DistributedConfig())
        assert (
            await repo.submit_window(
                parent_execution_id,
                task_list_path=path,
                max_inflight=3,
                submit_batch_size=50,
                now=utcnow(),
            )
            == 3
        )
        await session.commit()

    async with session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        items = (
            await session.scalars(
                select(BatchItemRow)
                .where(BatchItemRow.parent_execution_id == parent_execution_id)
                .order_by(BatchItemRow.item_id)
            )
        ).all()
        children = [await session.get(ExecutionRow, item.current_child_execution_id) for item in items]
        parent = await session.get(ExecutionRow, parent_execution_id)
        assert parent is not None
        parent.status = ExecutionStatus.QUEUED
        quota = await QuotaRepository(session, default_limits={}).refresh_owner('owner', now=utcnow())
        await session.commit()

    assert batch is not None
    assert quota.queued_executions == 1
    assert (batch.submitted_items, batch.active_items, batch.terminal_items) == (3, 3, 0)
    for item, child in zip(items, children, strict=True):
        assert child is not None
        assert child.owner_principal_id == 'owner'
        assert child.owner_tenant == 'tenant'
        assert child.created_by == 'system:batch-parent'
        assert child.environment_id == 'environment-1'
        assert child.pool_id == 'execution-pool'
        assert child.allocation_id == allocation_id
        assert child.workload == 'execution'
        assert child.max_attempts == 2
        assert child.queue_ttl_seconds == 30
        assert child.execution_ttl_seconds == 120
        assert child.stop_timeout_seconds == 15
        assert child.idempotency_scope == f'batch:{parent_execution_id}'
        assert child.idempotency_key == f'batch:{parent_execution_id}:item:{item.item_id}:attempt:0'
        assert child.placement_json == {
            'environment': 'environment-1',
            'pool': 'execution-pool',
            'allocation_id': allocation_id,
            'labels': {'workload': 'execution'},
        }
        assert child.spec_json['code'] == {'mode': 'config_only'}
        assert child.spec_json['run_input']['run_id'] == child.execution_id
        assert child.spec_json['run_input']['task_config']['index'] == item.task_ordinal
        assert child.spec_json['batch'] == {
            'parent_execution_id': parent_execution_id,
            # Inherited from the parent (not the child) so all children of one parent
            # share a time partition on the Boundless trace table.
            'parent_created_at': parent.created_at.isoformat(),
            'item_id': item.item_id,
            'attempt_no': 0,
        }


async def test_batch_child_dependency_binding_is_atomic_idempotent_and_updates_authoritative_template(
    session_factory,
    tmp_path,
):
    parent_execution_id = 'batch-bind-dependency'
    task_list_path, allocation_id = await _create_submittable_batch(
        session_factory,
        tmp_path,
        parent_execution_id=parent_execution_id,
        item_count=1,
        allocation_status=EnvironmentAllocationStatus.REQUESTED,
    )
    now = utcnow()
    module_dependencies = {
        'representative_item_id': 'item-000000000001',
        'selector_digest': 'a' * 64,
        'selectors': [{'kind': 'object', 'object_type': 'tasks', 'value': 'echo'}],
        'module_paths': ['nexus.extensions.tasks.echo'],
        'host_requirements': [],
    }
    async with session_factory() as session:
        parent = await session.get(ExecutionRow, parent_execution_id)
        batch = await session.get(BatchRunRow, parent_execution_id)
        allocation = await session.get(EnvironmentAllocationRow, allocation_id)
        assert parent is not None
        assert batch is not None
        assert allocation is not None
        batch.dependency_status = 'pending'
        batch.child_dependency_build_id = 'build-base'
        batch.child_template_json = deepcopy(parent.spec_json['batch']['child_template'])
        allocation.dependency_build_id = 'build-base'
        allocation.allocation_spec_json = {
            'code': {'mode': 'source_package'},
            'dependency_build': {'dependency_build_id': 'build-base', 'build_key': 'b' * 64},
        }
        session.add(
            DependencyBuildRow(
                dependency_build_id='build-augmented',
                build_key='c' * 64,
                owner_principal_id='owner',
                environment_id='environment-1',
                provider_type='kubernetes',
                provider_config_ref='cluster-a',
                provider_config_snapshot_json={'namespace': 'workloads'},
                status='requested',
                build_spec_json={},
                attempt=0,
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()

    async with session_factory() as session:
        repository = BatchRepository(session, config=DistributedConfig())
        await repository.bind_child_dependency_build(
            parent_execution_id,
            expected_dependency_build_id='build-base',
            dependency_build_id='build-augmented',
            module_dependencies=module_dependencies,
            now=now,
        )
        await session.commit()

    async with session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        allocation = await session.get(EnvironmentAllocationRow, allocation_id)
        assert batch is not None
        assert allocation is not None
        first_hash = allocation.allocation_spec_hash
        assert batch.dependency_status == 'building'
        assert batch.child_dependency_build_id == 'build-augmented'
        assert batch.module_dependencies_json == module_dependencies
        assert allocation.dependency_build_id == 'build-augmented'
        assert allocation.allocation_spec_json['dependency_build'] == {
            'dependency_build_id': 'build-augmented',
            'build_key': 'c' * 64,
        }
        assert batch.child_template_json['allocation_spec_hash'] == first_hash

    async with session_factory() as session:
        repository = BatchRepository(session, config=DistributedConfig())
        await repository.bind_child_dependency_build(
            parent_execution_id,
            expected_dependency_build_id='build-base',
            dependency_build_id='build-augmented',
            module_dependencies=module_dependencies,
            now=now,
        )
        await session.commit()

    async with session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        allocation = await session.get(EnvironmentAllocationRow, allocation_id)
        build = await session.get(DependencyBuildRow, 'build-augmented')
        assert batch is not None
        assert allocation is not None
        assert build is not None
        assert allocation.allocation_spec_hash == first_hash
        build.status = 'ready'
        build.manifest_json = {
            'schema_version': 2,
            'build_key': 'c' * 64,
            'nexus_wheel_path': 'wheels/nexus_agent-1.0.0-py3-none-any.whl',
            'module_names': ['echo'],
        }
        allocation.status = EnvironmentAllocationStatus.READY
        allocation.ready_manager_count = allocation.desired_manager_count
        allocation.ready_at = now
        await session.commit()

    async with session_factory() as session:
        created = await BatchRepository(session, config=DistributedConfig()).submit_window(
            parent_execution_id,
            task_list_path=task_list_path,
            max_inflight=1,
            submit_batch_size=1,
            now=now,
        )
        await session.commit()
        assert created == 1

        child = await session.scalar(
            select(ExecutionRow).where(
                ExecutionRow.parent_execution_id == parent_execution_id,
                ExecutionRow.spec_type == 'nexus_single',
            )
        )
        assert child is not None
        assert child.allocation_spec_hash == first_hash
        batch = await session.get(BatchRunRow, parent_execution_id)
        assert batch is not None
        assert batch.dependency_status == 'ready'
        assert batch.module_dependencies_json['module_names'] == ['echo']


async def test_batch_child_dependency_binding_prepares_user_data_off_event_loop(session_factory, tmp_path):
    parent_execution_id = 'batch-bind-dependency-off-loop'
    _, allocation_id = await _create_submittable_batch(
        session_factory,
        tmp_path,
        parent_execution_id=parent_execution_id,
        item_count=1,
        allocation_status=EnvironmentAllocationStatus.REQUESTED,
    )
    now = utcnow()
    module_dependencies = {
        'representative_item_id': 'item-000000000001',
        'selector_digest': 'a' * 64,
        'selectors': [{'kind': 'object', 'object_type': 'tasks', 'value': 'echo'}],
    }
    async with session_factory() as session:
        parent = await session.get(ExecutionRow, parent_execution_id)
        batch = await session.get(BatchRunRow, parent_execution_id)
        allocation = await session.get(EnvironmentAllocationRow, allocation_id)
        assert parent is not None
        assert batch is not None
        assert allocation is not None
        batch.dependency_status = 'pending'
        batch.child_dependency_build_id = 'build-base'
        batch.child_template_json = deepcopy(parent.spec_json['batch']['child_template'])
        allocation.dependency_build_id = 'build-base'
        allocation.allocation_spec_json = {
            'code': {'mode': 'source_package'},
            'dependency_build': {'dependency_build_id': 'build-base', 'build_key': 'b' * 64},
        }
        session.add(
            DependencyBuildRow(
                dependency_build_id='build-augmented-off-loop',
                build_key='c' * 64,
                owner_principal_id='owner',
                environment_id='environment-1',
                status='requested',
                build_spec_json={},
                attempt=0,
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()

    manager_thread_id = threading.get_ident()
    operations: list[str] = []
    worker_thread_ids: list[int] = []

    async def blocking_runner(operation, function, /, *args, **kwargs):
        operations.append(operation)

        def invoke():
            worker_thread_ids.append(threading.get_ident())
            return function(*args, **kwargs)

        return await asyncio.to_thread(invoke)

    async with session_factory() as session:
        await BatchRepository(
            session,
            config=DistributedConfig(),
            blocking_runner=blocking_runner,
        ).bind_child_dependency_build(
            parent_execution_id,
            expected_dependency_build_id='build-base',
            dependency_build_id='build-augmented-off-loop',
            module_dependencies=module_dependencies,
            now=now,
        )
        await session.commit()

    assert operations == ['bind-child-dependency-build']
    assert worker_thread_ids
    assert set(worker_thread_ids) == {worker_thread_ids[0]}
    assert worker_thread_ids[0] != manager_thread_id


async def test_batch_child_dependency_binding_rejects_provisioned_allocation(session_factory, tmp_path):
    parent_execution_id = 'batch-bind-provisioned'
    _, allocation_id = await _create_submittable_batch(
        session_factory,
        tmp_path,
        parent_execution_id=parent_execution_id,
        item_count=1,
        allocation_status=EnvironmentAllocationStatus.PROVISIONING,
    )
    now = utcnow()
    async with session_factory() as session:
        parent = await session.get(ExecutionRow, parent_execution_id)
        batch = await session.get(BatchRunRow, parent_execution_id)
        allocation = await session.get(EnvironmentAllocationRow, allocation_id)
        assert parent is not None
        assert batch is not None
        assert allocation is not None
        batch.dependency_status = 'pending'
        batch.child_dependency_build_id = 'build-base'
        batch.child_template_json = deepcopy(parent.spec_json['batch']['child_template'])
        allocation.dependency_build_id = 'build-base'
        session.add(
            DependencyBuildRow(
                dependency_build_id='build-too-late',
                build_key='d' * 64,
                owner_principal_id='owner',
                environment_id='environment-1',
                status='requested',
                build_spec_json={},
                attempt=0,
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()

    async with session_factory() as session:
        with pytest.raises(FencingError, match='batch_child_allocation_already_provisioned'):
            await BatchRepository(session, config=DistributedConfig()).bind_child_dependency_build(
                parent_execution_id,
                expected_dependency_build_id='build-base',
                dependency_build_id='build-too-late',
                module_dependencies={'representative_item_id': 'item-000000000001'},
                now=now,
            )


async def test_empty_source_batch_marks_child_dependency_skipped_without_provisioning(session_factory, tmp_path):
    parent_execution_id = 'batch-empty-dependency'
    task_list_path, allocation_id = await _create_submittable_batch(
        session_factory,
        tmp_path,
        parent_execution_id=parent_execution_id,
        item_count=0,
        allocation_status=EnvironmentAllocationStatus.REQUESTED,
    )
    now = utcnow()
    async with session_factory() as session:
        parent = await session.get(ExecutionRow, parent_execution_id)
        batch = await session.get(BatchRunRow, parent_execution_id)
        allocation = await session.get(EnvironmentAllocationRow, allocation_id)
        assert parent is not None
        assert batch is not None
        assert allocation is not None
        batch.dependency_status = 'pending'
        batch.child_dependency_build_id = 'build-base'
        batch.child_template_json = deepcopy(parent.spec_json['batch']['child_template'])
        allocation.dependency_build_id = 'build-base'
        await session.commit()

    async with session_factory() as session:
        repository = BatchRepository(session, config=DistributedConfig())
        await repository.mark_child_dependency_skipped(
            parent_execution_id,
            expected_dependency_build_id='build-base',
            now=now,
        )
        await session.commit()

        batch = await session.get(BatchRunRow, parent_execution_id)
        allocation = await session.get(EnvironmentAllocationRow, allocation_id)
        assert batch is not None
        assert allocation is not None
        assert batch.dependency_status == 'skipped'
        assert batch.module_dependencies_json == {
            'representative_item_id': None,
            'selector_digest': None,
            'selectors': [],
            'module_names': [],
            'host_requirements': [],
            'module_lock_path': None,
            'module_lock_sha256': None,
        }
        assert allocation.status == EnvironmentAllocationStatus.REQUESTED
        assert allocation.provider_resource_ref_json is None

        assert (
            await repository.submit_window(
                parent_execution_id,
                task_list_path=task_list_path,
                max_inflight=1,
                submit_batch_size=1,
                now=now,
            )
            == 0
        )


async def test_batch_submit_window_fails_when_augmented_dependency_build_fails(session_factory, tmp_path):
    parent_execution_id = 'batch-dependency-failed'
    task_list_path, _ = await _create_submittable_batch(
        session_factory,
        tmp_path,
        parent_execution_id=parent_execution_id,
        item_count=1,
        allocation_status=EnvironmentAllocationStatus.REQUESTED,
    )
    now = utcnow()
    async with session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        assert batch is not None
        batch.dependency_status = 'building'
        batch.child_dependency_build_id = 'build-failed'
        batch.module_dependencies_json = {
            'representative_item_id': 'item-000000000001',
            'selector_digest': 'a' * 64,
            'selectors': [{'kind': 'object', 'object_type': 'tasks', 'value': 'echo'}],
        }
        session.add(
            DependencyBuildRow(
                dependency_build_id='build-failed',
                build_key='e' * 64,
                owner_principal_id='owner',
                environment_id='environment-1',
                status='failed',
                build_spec_json={},
                attempt=1,
                error_code='dependency_module_lock_failed',
                error_message='resolver failed',
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()

    async with session_factory() as session:
        repository = BatchRepository(session, config=DistributedConfig())
        with pytest.raises(DomainValidationError) as exc_info:
            await repository.submit_window(
                parent_execution_id,
                task_list_path=task_list_path,
                max_inflight=1,
                submit_batch_size=1,
                now=now,
            )
        assert exc_info.value.code == 'batch_module_dependency_failed'
        await session.commit()

    async with session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        assert batch is not None
        assert batch.dependency_status == 'failed'


async def test_batch_submit_window_waits_when_augmented_dependency_build_will_retry(session_factory, tmp_path):
    parent_execution_id = 'batch-dependency-retry'
    task_list_path, _ = await _create_submittable_batch(
        session_factory,
        tmp_path,
        parent_execution_id=parent_execution_id,
        item_count=1,
        allocation_status=EnvironmentAllocationStatus.REQUESTED,
    )
    now = utcnow()
    async with session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        assert batch is not None
        batch.dependency_status = 'building'
        batch.child_dependency_build_id = 'build-retry'
        batch.module_dependencies_json = {
            'representative_item_id': 'item-000000000001',
            'selector_digest': 'a' * 64,
            'selectors': [{'kind': 'object', 'object_type': 'tasks', 'value': 'echo'}],
        }
        session.add(
            DependencyBuildRow(
                dependency_build_id='build-retry',
                build_key='f' * 64,
                owner_principal_id='owner',
                environment_id='environment-1',
                status='failed',
                build_spec_json={},
                attempt=1,
                next_retry_at=now + timedelta(seconds=15),
                error_code='dependency_module_lock_failed',
                error_message='resolver unavailable',
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()

    async with session_factory() as session:
        repository = BatchRepository(session, config=DistributedConfig())
        assert (
            await repository.submit_window(
                parent_execution_id,
                task_list_path=task_list_path,
                max_inflight=1,
                submit_batch_size=1,
                now=now,
            )
            == 0
        )
        await session.commit()

    async with session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        assert batch is not None
        assert batch.dependency_status == 'building'


async def test_batch_submit_window_bounds_materialized_child_specs_by_task_list_bytes(session_factory, tmp_path):
    parent_execution_id = 'batch-submit-byte-bounded'
    path, _ = await _create_submittable_batch(
        session_factory,
        tmp_path,
        parent_execution_id=parent_execution_id,
        item_count=3,
    )
    first_line = path.read_bytes().splitlines(keepends=True)[0]

    async with session_factory() as session:
        created = await BatchRepository(session, config=DistributedConfig()).submit_window(
            parent_execution_id,
            task_list_path=path,
            max_inflight=3,
            submit_batch_size=3,
            max_batch_bytes=len(first_line) + 1,
            now=utcnow(),
        )
        await session.commit()

    assert created == 1


async def test_batch_submit_window_creates_500_children_with_bounded_sql_statements(session_factory, tmp_path):
    parent_execution_id = 'batch-submit-bulk'
    path, _ = await _create_submittable_batch(
        session_factory,
        tmp_path,
        parent_execution_id=parent_execution_id,
        item_count=500,
    )
    statements: list[str] = []
    engine = session_factory.kw['bind']

    def record_statement(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, 'before_cursor_execute', record_statement)
    try:
        async with session_factory() as session:
            created = await BatchRepository(session, config=DistributedConfig()).submit_window(
                parent_execution_id,
                task_list_path=path,
                max_inflight=500,
                submit_batch_size=500,
                now=utcnow(),
            )
            await session.commit()
    finally:
        event.remove(engine.sync_engine, 'before_cursor_execute', record_statement)

    assert created == 500
    assert len(statements) <= 20, len(statements)

    async with session_factory() as session:
        children = (
            await session.scalars(
                select(ExecutionRow).where(ExecutionRow.spec_type == 'nexus_single').order_by(ExecutionRow.execution_id)
            )
        ).all()
        children[0].last_schedule_error = 'policy_deferred:quota_exhausted'
        children[1].last_schedule_error = 'policy_deferred:hard_brake'
        children[2].last_schedule_error = 'no_capacity'
        await session.commit()
    assert len(children) == 500
    assert all(child.next_event_seq == 2 for child in children)

    async with session_factory() as session:
        summary = await BatchRepository(session, config=DistributedConfig()).batch_summary(
            parent_execution_id,
            include_status_counts=True,
        )

    assert summary['child_execution_status_counts'] == {'queued': 500}
    assert summary['child_waiting_policy_count'] == 2


async def test_batch_finalization_counter_gate_avoids_active_ledger_scan(session_factory, tmp_path):
    parent_execution_id = 'batch-finalization-counter-gate'
    await _create_submittable_batch(
        session_factory,
        tmp_path,
        parent_execution_id=parent_execution_id,
        item_count=3,
    )
    statements: list[str] = []
    engine = session_factory.kw['bind']

    def record_statement(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().lower().startswith('select'):
            statements.append(statement)

    event.listen(engine.sync_engine, 'before_cursor_execute', record_statement)
    try:
        async with session_factory() as session:
            result = await BatchRepository(session, config=DistributedConfig()).finalize_parent_if_ready(
                parent_execution_id,
                now=utcnow(),
            )
            await session.commit()
    finally:
        event.remove(engine.sync_engine, 'before_cursor_execute', record_statement)

    assert result is None
    assert not any('distributed_batch_items' in statement for statement in statements)


async def test_batch_child_sync_locks_batch_before_item_and_execution_rows(session_factory, tmp_path):
    parent_execution_id = 'batch-sync-lock-order'
    path, _ = await _create_submittable_batch(
        session_factory,
        tmp_path,
        parent_execution_id=parent_execution_id,
        item_count=1,
    )
    async with session_factory() as session:
        await BatchRepository(session, config=DistributedConfig()).submit_window(
            parent_execution_id,
            task_list_path=path,
            max_inflight=1,
            submit_batch_size=1,
            now=utcnow(),
        )
        await session.commit()

    statements: list[str] = []
    engine = session_factory.kw['bind']

    def record_statement(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().lower().startswith('select'):
            statements.append(statement)

    event.listen(engine.sync_engine, 'before_cursor_execute', record_statement)
    try:
        async with session_factory() as session:
            await BatchRepository(session, config=DistributedConfig()).sync_child_states_page(
                parent_execution_id,
                after_item_id=None,
                page_size=1,
                retry_failed=0,
                now=utcnow(),
            )
            await session.commit()
    finally:
        event.remove(engine.sync_engine, 'before_cursor_execute', record_statement)

    assert 'FROM distributed_batch_runs' in statements[0]
    assert 'FROM distributed_batch_items' in statements[1]
    assert 'JOIN executions' not in statements[1]
    assert 'FROM executions' in statements[2]
    execution_projection = ' '.join(statements[2].lower().split()).split(' from executions', 1)[0]
    for column in ('driver_config_json', 'placement_json', 'result_json', 'spec_json', 'trace_carrier_json'):
        assert column not in execution_projection


async def test_batch_child_sync_locks_items_but_reads_execution_snapshots_without_locks():
    item = SimpleNamespace(
        item_id='item-1',
        current_child_execution_id='child-1',
        child_execution_status=ExecutionStatus.RUNNING,
        child_execution_version=1,
    )
    child = SimpleNamespace(execution_id='child-1', status=ExecutionStatus.RUNNING, version=1)
    observed_locks: list[tuple[str, bool | None]] = []

    class Rows:
        def __init__(self, values):
            self.values = values

        def all(self):
            return self.values

    class RecordingSession:
        async def scalar(self, _statement):
            return SimpleNamespace()

        async def scalars(self, statement):
            entity = statement.column_descriptions[0]['entity']
            lock = statement._for_update_arg
            observed_locks.append((entity.__name__, None if lock is None else bool(lock.skip_locked)))
            if entity is BatchItemRow:
                return Rows([item])
            if entity is ExecutionRow:
                return Rows([child])
            raise AssertionError(entity)

    page = await BatchRepository(RecordingSession(), config=DistributedConfig()).sync_child_states_page(  # type: ignore[arg-type]
        'batch-skip-locked',
        after_item_id=None,
        page_size=2000,
        retry_failed=0,
        now=utcnow(),
    )

    assert page.complete is True
    assert page.changed == 0
    assert observed_locks == [('BatchItemRow', True), ('ExecutionRow', None)]


async def test_batch_item_detail_includes_ordered_attempts(session_factory, tmp_path):
    parent_execution_id = 'batch-item-detail'
    path, _ = await _create_submittable_batch(
        session_factory,
        tmp_path,
        parent_execution_id=parent_execution_id,
        item_count=1,
    )
    async with session_factory() as session:
        repository = BatchRepository(session, config=DistributedConfig())
        assert (
            await repository.submit_window(
                parent_execution_id,
                task_list_path=path,
                max_inflight=1,
                submit_batch_size=50,
                now=utcnow(),
            )
            == 1
        )
        detail = await repository.get_item(parent_execution_id, 'item-000000000001')
        attempt = await repository.get_attempt(parent_execution_id, 'item-000000000001', 0)

    assert detail['item_id'] == 'item-000000000001'
    assert 'run_id' not in detail
    assert detail['current_attempt_no'] == 0
    assert detail['attempts'] == [attempt]
    assert attempt['attempt_no'] == 0
    assert attempt['child_execution_id'] == detail['current_child_execution_id']


async def test_batch_submit_window_refills_capacity_with_new_logical_attempt(session_factory, tmp_path):
    first_started = utcnow() - timedelta(seconds=20)
    parent_execution_id = 'batch-submit-retry'
    path, _ = await _create_submittable_batch(
        session_factory,
        tmp_path,
        parent_execution_id=parent_execution_id,
        item_count=3,
    )
    async with session_factory() as session:
        repo = BatchRepository(session, config=DistributedConfig())
        assert (
            await repo.submit_window(
                parent_execution_id,
                task_list_path=path,
                max_inflight=2,
                submit_batch_size=50,
                now=utcnow(),
            )
            == 2
        )
        await session.commit()

    async with session_factory() as session:
        first = await session.scalar(
            select(BatchItemRow)
            .where(BatchItemRow.parent_execution_id == parent_execution_id)
            .order_by(BatchItemRow.item_id)
            .limit(1)
        )
        assert first is not None
        first_child = await session.get(ExecutionRow, first.current_child_execution_id)
        assert first_child is not None
        first_child.status = ExecutionStatus.FAILED
        first_child.started_at = first_started
        first_child.error_json = {'type': 'ExpectedFailure'}
        first_child.finished_at = utcnow()
        first_child.version += 1
        first_child_execution_id = first_child.execution_id
        await session.commit()

    async with session_factory() as session:
        repo = BatchRepository(session, config=DistributedConfig())
        page = await repo.sync_child_states_page(
            parent_execution_id,
            after_item_id=None,
            page_size=10,
            retry_failed=1,
            now=utcnow(),
        )
        assert page.changed == 1
        assert (
            await repo.submit_window(
                parent_execution_id,
                task_list_path=path,
                max_inflight=2,
                submit_batch_size=50,
                now=utcnow(),
            )
            == 1
        )
        await session.commit()

    async with session_factory() as session:
        item = await session.get(
            BatchItemRow,
            {'parent_execution_id': parent_execution_id, 'item_id': 'item-000000000001'},
        )
        attempts = (
            await session.scalars(
                select(BatchItemAttemptRow)
                .where(
                    BatchItemAttemptRow.parent_execution_id == parent_execution_id,
                    BatchItemAttemptRow.item_id == 'item-000000000001',
                )
                .order_by(BatchItemAttemptRow.attempt_no)
            )
        ).all()
        batch = await session.get(BatchRunRow, parent_execution_id)
        retried_child = await session.get(ExecutionRow, item.current_child_execution_id if item else '')
        parent_row = await session.get(ExecutionRow, parent_execution_id)

    assert item is not None
    assert item.current_attempt_no == 1
    assert item.status == 'submitted'
    assert item.first_started_at == first_started
    assert item.finished_at is None
    assert not item.finished_at_estimated
    assert [attempt.status for attempt in attempts] == [ExecutionStatus.FAILED, ExecutionStatus.QUEUED]
    assert attempts[0].error_code == 'ExpectedFailure'
    assert retried_child is not None
    assert item.task_name == retried_child.task_name == retried_child.spec_json['run_input']['task_name']
    assert retried_child.execution_id != first_child_execution_id
    assert retried_child.spec_json['run_input']['run_id'] == retried_child.execution_id
    assert parent_row is not None
    assert retried_child.spec_json['batch'] == {
        'parent_execution_id': parent_execution_id,
        # A retry must reuse the parent's timestamp, not its own, so the new attempt
        # writes into the same partition as the original one.
        'parent_created_at': parent_row.created_at.isoformat(),
        'item_id': item.item_id,
        'attempt_no': 1,
    }
    assert retried_child.idempotency_key == (f'batch:{parent_execution_id}:item:{item.item_id}:attempt:1')
    assert batch is not None
    assert (batch.submitted_items, batch.active_items, batch.terminal_items) == (3, 2, 0)

    final_finished = utcnow()
    async with session_factory() as session:
        child = await session.get(ExecutionRow, retried_child.execution_id)
        child.status = ExecutionStatus.SUCCEEDED
        child.started_at = first_started + timedelta(seconds=10)
        child.finished_at = final_finished
        child.version += 1
        await session.commit()
    async with session_factory() as session:
        repo = BatchRepository(session, config=DistributedConfig())
        await repo.sync_child_states_page(
            parent_execution_id, after_item_id=None, page_size=10, retry_failed=1, now=utcnow()
        )
        await session.commit()
        saved = await session.get(BatchItemRow, (parent_execution_id, item.item_id))
        assert saved.first_started_at == first_started
        assert saved.finished_at == final_finished
        assert not saved.finished_at_estimated
        assert saved.status == 'succeeded'
        again = await repo.sync_child_states_page(
            parent_execution_id, after_item_id=None, page_size=10, retry_failed=1, now=utcnow()
        )
        assert again.changed == 0


@pytest.mark.parametrize(
    ('reason', 'unsubmitted_status'),
    [
        ('user_stop', 'cancelled'),
        ('parent_ttl', 'expired'),
    ],
)
async def test_batch_stop_fanout_uses_stable_pages_without_terminalizing_children(
    session_factory,
    reason,
    unsubmitted_status,
):
    content = _task_list(120)
    await _create_batch_run(session_factory, parent_execution_id=f'batch-stop-{reason}', task_list=content)
    parent_execution_id = f'batch-stop-{reason}'
    now = utcnow()
    async with session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        assert batch is not None
        batch.generation_status = 'complete'
        batch.materialized_count = 120
        for index in range(120):
            item_id = f'item-{index + 1:012d}'
            session.add(
                BatchItemRow(
                    parent_execution_id=parent_execution_id,
                    item_id=item_id,
                    task_ordinal=index,
                    task_offset=index,
                    task_length=1,
                    current_attempt_no=0,
                    current_child_execution_id=f'execution-{index:04d}' if index % 2 == 0 else None,
                    spec_sha256='a' * 64,
                    status='submitted' if index % 2 == 0 else 'pending',
                    status_version=1,
                    metadata_json={},
                    created_at=now,
                    updated_at=now,
                )
            )
        await session.commit()

    returned_child_ids = []
    page_sizes = []
    while True:
        async with session_factory() as session:
            repo = BatchRepository(session, config=DistributedConfig())
            await repo.begin_stop(parent_execution_id, reason=reason, now=utcnow())
            page = await repo.next_stop_page(parent_execution_id, page_size=50, now=utcnow())
            await session.commit()
        returned_child_ids.extend(page.child_execution_ids)
        page_sizes.append(page.scanned)
        if page.complete:
            break

    assert page_sizes == [50, 50, 20]
    assert len(returned_child_ids) == 60
    assert len(set(returned_child_ids)) == 60
    async with session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        rows = (
            await session.scalars(
                select(BatchItemRow)
                .where(BatchItemRow.parent_execution_id == parent_execution_id)
                .order_by(BatchItemRow.item_id)
            )
        ).all()

    assert batch is not None
    assert batch.control_state == 'stopping'
    assert batch.stop_reason == reason
    assert batch.stop_cursor == 'item-000000000120'
    assert all(row.status == 'submitted' for row in rows[::2])
    assert all(row.status == unsubmitted_status for row in rows[1::2])
    assert all(row.finished_at is not None and row.first_started_at is None for row in rows[1::2])
    assert all(row.finished_at is None for row in rows[::2])


async def test_batch_child_sync_is_versioned_paged_and_avoids_noop_summary_writes(session_factory):
    content = _task_list(12)
    parent_execution_id = 'batch-sync'
    await _create_batch_run(session_factory, parent_execution_id=parent_execution_id, task_list=content)
    now = utcnow()
    async with session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        assert batch is not None
        batch.generation_status = 'complete'
        batch.materialized_count = 12
        batch.submitted_items = 12
        batch.active_items = 12
        executions = ExecutionRepository(session, shard_count=8)
        for index in range(12):
            item_id = f'item-{index + 1:012d}'
            child = await executions.create_execution(
                CreateExecution(
                    execution_id=f'child-{index:04d}',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={'run_input': {'task_name': 'echo'}},
                    owner_principal_id='owner',
                    max_attempts=2,
                ),
                now=now,
            )
            child_row = await session.get(ExecutionRow, child.execution_id)
            assert child_row is not None
            child_row.status = ExecutionStatus.SUCCEEDED if index < 6 else ExecutionStatus.RUNNING
            child_row.result_artifact_id = f'result-{index:04d}' if index < 6 else None
            child_row.finished_at = now if index < 6 else None
            child_row.version = 2
            session.add(
                BatchItemRow(
                    parent_execution_id=parent_execution_id,
                    item_id=item_id,
                    task_ordinal=index,
                    task_offset=index,
                    task_length=1,
                    current_attempt_no=0,
                    current_child_execution_id=child.execution_id,
                    spec_sha256='a' * 64,
                    status='submitted',
                    status_version=1,
                    child_execution_status=ExecutionStatus.QUEUED,
                    child_execution_version=1,
                    metadata_json={},
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                BatchItemAttemptRow(
                    parent_execution_id=parent_execution_id,
                    item_id=item_id,
                    attempt_no=0,
                    child_execution_id=child.execution_id,
                    spec_sha256='a' * 64,
                    status=ExecutionStatus.QUEUED,
                    status_version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
        await session.commit()

    statements: list[str] = []
    engine = session_factory.kw['bind']

    def record_statement(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, 'before_cursor_execute', record_statement)
    try:
        scanned = []
        changed = 0
        terminal_discovery_lags = []
        while True:
            async with session_factory() as session:
                batch = await session.get(BatchRunRow, parent_execution_id)
                assert batch is not None
                page = await BatchRepository(session, config=DistributedConfig()).sync_child_states_page(
                    parent_execution_id,
                    after_item_id=batch.child_sync_cursor,
                    page_size=4,
                    retry_failed=0,
                    now=now + timedelta(seconds=10),
                )
                await session.commit()
            scanned.append(page.scanned)
            changed += page.changed
            terminal_discovery_lags.extend(page.terminal_discovery_lags)
            if page.complete:
                break
    finally:
        event.remove(engine.sync_engine, 'before_cursor_execute', record_statement)

    assert scanned == [4, 4, 4]
    assert changed == 12
    assert terminal_discovery_lags == [10.0] * 6
    assert len(statements) <= 27, len(statements)
    async with session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        attempts = (
            await session.scalars(
                select(BatchItemAttemptRow)
                .where(BatchItemAttemptRow.parent_execution_id == parent_execution_id)
                .order_by(BatchItemAttemptRow.item_id)
            )
        ).all()
        progress_at = batch.last_progress_at

    assert batch is not None
    assert batch.active_items == 6
    assert batch.terminal_items == 6
    assert batch.child_sync_cursor is None
    assert [attempt.result_artifact_id for attempt in attempts[:6]] == [f'result-{index:04d}' for index in range(6)]

    async with session_factory() as session:
        page = await BatchRepository(session, config=DistributedConfig()).sync_child_states_page(
            parent_execution_id,
            after_item_id=None,
            page_size=12,
            retry_failed=0,
            now=utcnow(),
        )
        await session.commit()
    assert page.changed == 0
    async with session_factory() as session:
        unchanged = await session.get(BatchRunRow, parent_execution_id)
    assert unchanged is not None
    assert unchanged.last_progress_at == progress_at


async def test_batch_child_sync_remains_page_bounded_with_ten_thousand_active_items(session_factory):
    parent_execution_id = 'batch-sync-ten-thousand'
    now = utcnow()
    item_count = 10_000
    page_size = 500
    async with session_factory() as session:
        session.add(
            BatchRunRow(
                parent_execution_id=parent_execution_id,
                shard_id=0,
                generation_mode='local_task_list',
                generation_status='complete',
                generation_epoch=1,
                materialize_cursor_bytes=0,
                materialized_count=item_count,
                owner_principal_id='owner',
                total_items=item_count,
                submitted_items=item_count,
                active_items=item_count,
                terminal_items=0,
                submission_closed=True,
                control_state='active',
                summary_version=0,
                last_progress_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        await session.execute(
            insert(BatchItemRow),
            [
                {
                    'parent_execution_id': parent_execution_id,
                    'item_id': f'item-{index + 1:012d}',
                    'task_ordinal': index,
                    'task_offset': index,
                    'task_length': 1,
                    'current_attempt_no': 0,
                    'current_child_execution_id': f'child-{index:012d}',
                    'spec_sha256': 'a' * 64,
                    'status': 'submitted',
                    'status_version': 1,
                    'child_execution_status': ExecutionStatus.RUNNING,
                    'child_execution_version': 1,
                    'metadata_json': {},
                    'created_at': now,
                    'updated_at': now,
                }
                for index in range(item_count)
            ],
        )
        await session.execute(
            insert(ExecutionRow),
            [
                {
                    'execution_id': f'child-{index:012d}',
                    'created_by': 'system:batch-parent',
                    'version': 1,
                    'shard_id': index % 8,
                    'mode': ExecutionMode.JOB,
                    'spec_type': 'nexus_single',
                    'instance_mode': 'asyncio',
                    'spec_json': {},
                    'driver_config_json': {},
                    'placement_json': {},
                    'status': ExecutionStatus.RUNNING,
                    'route_status': 'none',
                    'slot_cost': 1,
                    'next_attempt_seq': 1,
                    'attempt_count': 1,
                    'max_attempts': 1,
                    'next_event_seq': 1,
                    'priority': 0,
                    'next_schedule_at': now,
                    'created_at': now,
                    'updated_at': now,
                }
                for index in range(page_size)
            ],
        )
        await session.commit()

    async with session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
        assert batch is not None
        first = await BatchRepository(session, config=DistributedConfig()).sync_child_states_page(
            parent_execution_id,
            after_item_id=batch.child_sync_cursor,
            page_size=page_size,
            retry_failed=0,
            now=now,
        )
        await session.commit()

    assert first.scanned == page_size
    assert first.changed == 0
    assert first.complete is False
    async with session_factory() as session:
        batch = await session.get(BatchRunRow, parent_execution_id)
    assert batch is not None
    assert batch.child_sync_cursor == f'item-{page_size:012d}'


@pytest.mark.parametrize(
    ('control_state', 'stop_reason', 'item_statuses', 'total_items', 'expected_status', 'expected_extra'),
    [
        (
            'stopping',
            'user_stop',
            ['succeeded', 'killed', 'cancelled'],
            5,
            ExecutionStatus.KILLED,
            {'cancelled': 2},
        ),
        (
            'stopping',
            'parent_ttl',
            ['succeeded', 'expired', 'expired'],
            5,
            ExecutionStatus.EXPIRED,
            {'expired': 2},
        ),
        (
            'stopping',
            'fail_fast',
            ['failed', 'killed', 'cancelled'],
            3,
            ExecutionStatus.FAILED,
            {},
        ),
    ],
)
async def test_batch_finalization_repairs_summary_and_is_the_parent_terminal_writer(
    session_factory,
    control_state,
    stop_reason,
    item_statuses,
    total_items,
    expected_status,
    expected_extra,
):
    parent_execution_id = f'batch-final-{stop_reason or "success"}'
    content = _task_list(total_items)
    await _create_batch_run(session_factory, parent_execution_id=parent_execution_id, task_list=content)
    now = utcnow()
    async with session_factory() as session:
        parent = await ExecutionRepository(session, shard_count=8).create_execution(
            CreateExecution(
                execution_id=parent_execution_id,
                mode=ExecutionMode.JOB,
                spec_type='nexus_batch',
                instance_mode='asyncio',
                spec={},
                owner_principal_id='owner',
                max_attempts=2,
            ),
            now=now,
        )
        parent_row = await session.get(ExecutionRow, parent.execution_id)
        assert parent_row is not None
        parent_row.status = ExecutionStatus.STOPPING if control_state == 'stopping' else ExecutionStatus.RUNNING
        batch = await session.get(BatchRunRow, parent_execution_id)
        assert batch is not None
        batch.generation_status = 'complete'
        batch.materialized_count = len(item_statuses)
        batch.total_items = total_items
        batch.control_state = control_state
        batch.stop_reason = stop_reason
        batch.stop_requested_at = now if stop_reason else None
        batch.stop_cursor = f'item-{len(item_statuses):012d}' if stop_reason else None
        batch.active_items = 99
        batch.terminal_items = len(item_statuses)
        for index, status in enumerate(item_statuses):
            item_id = f'item-{index + 1:012d}'
            session.add(
                BatchItemRow(
                    parent_execution_id=parent_execution_id,
                    item_id=item_id,
                    task_ordinal=index,
                    task_offset=index,
                    task_length=1,
                    current_attempt_no=0,
                    current_child_execution_id=None,
                    spec_sha256='a' * 64,
                    status=status,
                    status_version=1,
                    metadata_json={},
                    created_at=now,
                    updated_at=now,
                )
            )
        await session.commit()

    async with session_factory() as session:
        result = await BatchRepository(session, config=DistributedConfig()).finalize_parent_if_ready(
            parent_execution_id,
            now=utcnow(),
        )
        await session.commit()

    assert result is not None
    assert result.parent_status == expected_status
    assert result.summary['total'] == total_items
    for key, value in expected_extra.items():
        assert result.summary[key] >= value
    async with session_factory() as session:
        parent = await session.get(ExecutionRow, parent_execution_id)
        batch = await session.get(BatchRunRow, parent_execution_id)

    assert parent is not None
    assert parent.status == expected_status
    assert parent.result_json == {'batch': result.summary}
    assert batch is not None
    assert batch.control_state == 'finalized'
    assert batch.active_items == 0
    assert batch.terminal_items == total_items
    assert batch.summary_json == result.summary


async def test_successful_batch_waits_for_ready_parent_result_artifact(session_factory):
    parent_execution_id = 'batch-final-success-artifact'
    content = _task_list(1)
    await _create_batch_run(session_factory, parent_execution_id=parent_execution_id, task_list=content)
    now = utcnow()
    async with session_factory() as session:
        await ExecutionRepository(session, shard_count=8).create_execution(
            CreateExecution(
                execution_id=parent_execution_id,
                mode=ExecutionMode.JOB,
                spec_type='nexus_batch',
                instance_mode='asyncio',
                spec={},
                owner_principal_id='owner',
                environment_id='environment-1',
                allocation_id='allocation-parent',
                workload='orchestration',
            ),
            now=now,
        )
        parent = await session.get(ExecutionRow, parent_execution_id)
        batch = await session.get(BatchRunRow, parent_execution_id)
        assert parent is not None
        assert batch is not None
        parent.status = ExecutionStatus.RUNNING
        parent.current_assignment_id = 'assignment-parent'
        parent.current_attempt_id = 'attempt-parent'
        batch.generation_status = 'complete'
        batch.materialized_count = 1
        batch.total_items = 1
        batch.terminal_items = 1
        session.add(
            BatchItemRow(
                parent_execution_id=parent_execution_id,
                item_id='item-000000000001',
                task_ordinal=0,
                task_offset=0,
                task_length=1,
                current_attempt_no=0,
                spec_sha256='a' * 64,
                status='succeeded',
                status_version=1,
                metadata_json={},
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()

    async with session_factory() as session:
        result = await BatchRepository(session, config=DistributedConfig()).finalize_parent_if_ready(
            parent_execution_id,
            now=utcnow(),
        )
        await session.commit()

    assert result is not None
    assert result.finalized is False
    assert result.result_artifact_required is True
    assert result.assignment_id == 'assignment-parent'
    assert result.attempt_id == 'attempt-parent'
    async with session_factory() as session:
        parent = await session.get(ExecutionRow, parent_execution_id)
        batch = await session.get(BatchRunRow, parent_execution_id)
    assert parent is not None
    assert parent.status == ExecutionStatus.RUNNING
    assert parent.result_artifact_id is None
    assert batch is not None
    assert batch.control_state == 'active'
    assert batch.summary_json == result.summary

    statements: list[str] = []
    engine = session_factory.kw['bind']

    def record_statement(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().lower().startswith('select'):
            statements.append(statement)

    event.listen(engine.sync_engine, 'before_cursor_execute', record_statement)
    try:
        async with session_factory() as session:
            repeated = await BatchRepository(session, config=DistributedConfig()).finalize_parent_if_ready(
                parent_execution_id,
                now=utcnow(),
            )
            await session.commit()
    finally:
        event.remove(engine.sync_engine, 'before_cursor_execute', record_statement)

    assert repeated is not None
    assert repeated.summary == result.summary
    assert not any('distributed_batch_items' in statement for statement in statements)

    async with session_factory() as session:
        parent = await session.get(ExecutionRow, parent_execution_id)
        assert parent is not None
        parent.current_assignment_id = 'assignment-replacement'
        parent.current_attempt_id = 'attempt-replacement'
        await session.commit()

    async with session_factory() as session:
        with pytest.raises(ValueError, match='batch_parent_result_failure_fenced'):
            await BatchRepository(session, config=DistributedConfig()).fail_parent_result_persistence(
                parent_execution_id,
                summary=result.summary,
                assignment_id=result.assignment_id,
                attempt_id=result.attempt_id,
                cause='ValueError',
                now=utcnow(),
            )

    async with session_factory() as session:
        parent = await session.get(ExecutionRow, parent_execution_id)
        batch = await session.get(BatchRunRow, parent_execution_id)
    assert parent is not None
    assert parent.status == ExecutionStatus.RUNNING
    assert batch is not None
    assert batch.control_state == 'active'

    async with session_factory() as session:
        parent = await session.get(ExecutionRow, parent_execution_id)
        assert parent is not None
        parent.result_artifact_id = 'artifact-parent-result'
        session.add(
            ArtifactRow(
                artifact_id='artifact-parent-result',
                owner_principal_id='owner',
                environment_id='environment-1',
                owner_tenant='tenant',
                kind='run_result',
                producer_key=f'run-result:{parent_execution_id}:attempt-parent',
                uri='memory://artifact-parent-result',
                sha256='b' * 64,
                size_bytes=1,
                provider_etag=None,
                status='ready',
                created_by='test',
                expires_at=None,
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()

    async with session_factory() as session:
        finalized = await BatchRepository(session, config=DistributedConfig()).finalize_parent_if_ready(
            parent_execution_id,
            now=utcnow(),
        )
        await session.commit()
    assert finalized is not None
    assert finalized.finalized is True
    assert finalized.parent_status == ExecutionStatus.SUCCEEDED
