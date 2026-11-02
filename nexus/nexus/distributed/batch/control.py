from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import signal
import sys
import tempfile

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, TypeVar, cast

from nexus.core.module_dependencies import ModuleDependencyError, ModuleSelector, selectors_from_run_input
from nexus.dataset.auth import WedataKey
from nexus.distributed.artifacts.gateway import (
    ApplicationGatewayArtifactTransport,
    ArtifactGatewayTransientError,
    TaskListArtifactGateway,
    create_application_gateway_artifact_transport,
)
from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.artifacts.ports import BatchArtifactClient
from nexus.distributed.batch.data_list import DataListError, DataListIndex, scan_data_list
from nexus.distributed.batch.materialize import COMPOSE_REQUEST_MAX_BYTES, MaterializedTaskList
from nexus.distributed.batch.ports import BatchControlServicePort
from nexus.distributed.batch.task_list import (
    BATCH_ITEM_METADATA_BYTES_ENV,
    DEFAULT_SERVER_BATCH_ITEM_METADATA_BYTES,
    MAX_BATCH_METADATA_BYTES,
    TaskListVersion,
    canonical_batch_metadata,
)
from nexus.distributed.batch.task_store import BatchTaskStore
from nexus.distributed.config.views import BatchControlConfigView
from nexus.distributed.errors import DomainConflictError, DomainValidationError
from nexus.distributed.persistence.orm import BatchRunRow, ExecutionRow
from nexus.distributed.repositories.artifact import ArtifactRepository
from nexus.distributed.repositories.batch import (
    BatchRepository,
    MaterializationSnapshotStaleError,
    MaterializeChunkSnapshot,
    MaterializedTaskListChunk,
    RepresentativeRunInput,
    cross_validate_task_data_list,
    encode_task_store_record,
    read_materialize_task_list_chunk,
    read_representative_run_input,
)
from nexus.distributed.repositories.dependency_build import DependencyBuildRepository
from nexus.distributed.repositories.execution import ExecutionRepository
from nexus.distributed.repositories.execution_credentials import ExecutionCredentialRepository
from nexus.distributed.submission.specs import DataListSpec, ResolvedDatasetVersion, TaskListManifest
from nexus.distributed.utils.clock import utcnow
from nexus.distributed.utils.identity import canonical_sha256
from nexus.utils.platform_metrics import measured
from nexus.utils.timeout import async_timeout


_GENERATOR_LOG_TAIL_BYTES = 64 * 1024
_MATERIALIZE_RESULT_MAX_BYTES = MAX_BATCH_METADATA_BYTES + 64 * 1024
_T = TypeVar('_T')


@dataclass(frozen=True, slots=True)
class _LocalFileFingerprint:
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    inode: int
    device: int


@dataclass(frozen=True, slots=True)
class _VerifiedTaskList:
    generation_epoch: int
    sha256: str
    size_bytes: int
    path: Path
    file_fingerprint: _LocalFileFingerprint


@dataclass(frozen=True, slots=True)
class _PreparedTaskList:
    generation_epoch: int
    artifact_id: str
    kind: str
    sha256: str
    size_bytes: int
    path: Path
    file_fingerprint: _LocalFileFingerprint


@dataclass(frozen=True, slots=True)
class _VerifiedDataList:
    generation_epoch: int
    sha256: str
    size_bytes: int
    path: Path
    file_fingerprint: _LocalFileFingerprint
    index: DataListIndex


@dataclass(frozen=True, slots=True)
class _PreparedDataList:
    generation_epoch: int
    artifact_id: str
    kind: str
    sha256: str
    size_bytes: int
    path: Path
    file_fingerprint: _LocalFileFingerprint


class _BatchPreparationRequiredError(RuntimeError):
    pass


class BatchControlService(BatchControlServicePort):
    def __init__(
        self,
        *,
        session_factory,
        config: BatchControlConfigView,
        artifact_client: BatchArtifactClient | None = None,
        artifact_transport: ApplicationGatewayArtifactTransport | None = None,
        task_store: BatchTaskStore | None = None,
        on_blocking_timeout: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        if artifact_client is not None and artifact_transport is not None:
            raise ValueError('artifact_client and artifact_transport are mutually exclusive')
        self.session_factory = session_factory
        self.config = config
        self._task_store = task_store
        self._artifact_transport = artifact_transport
        self._owns_artifact_transport = False
        self._artifact_client = (
            artifact_client
            if artifact_client is not None
            else TaskListArtifactGateway(artifact_transport)
            if artifact_transport is not None
            else None
        )
        self._prepared_task_lists: dict[str, _PreparedTaskList] = {}
        self._verified_task_lists: dict[str, _VerifiedTaskList] = {}
        self._prepared_data_lists: dict[str, _PreparedDataList] = {}
        self._verified_data_lists: dict[str, _VerifiedDataList] = {}
        self._prepare_locks: dict[str, asyncio.Lock] = {}
        self._blocking_semaphore = asyncio.Semaphore(config.batch_control_blocking_concurrency)
        self._blocking_tasks: set[asyncio.Task[Any]] = set()
        self._blocking_watchdogs: set[asyncio.Task[None]] = set()
        self._on_blocking_timeout = on_blocking_timeout
        spool = Path(config.batch_control_spool_dir)
        spool.mkdir(parents=True, exist_ok=True)
        self._temporary_root = tempfile.TemporaryDirectory(prefix='run-', dir=spool)

    def _repository(self, session) -> BatchRepository:
        return BatchRepository(
            session,
            config=self.config,
            task_store=self._task_store,
            blocking_runner=self._run_blocking,
        )

    async def close(self) -> None:
        self._prepared_task_lists.clear()
        self._verified_task_lists.clear()
        self._prepared_data_lists.clear()
        self._verified_data_lists.clear()
        self._prepare_locks.clear()
        await self._run_blocking('cleanup-temporary-root', self._temporary_root.cleanup)
        if self._owns_artifact_transport and self._artifact_transport is not None:
            await self._artifact_transport.close()
            self._owns_artifact_transport = False

    async def _run_blocking(
        self,
        operation: str,
        function: Callable[..., _T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        await self._blocking_semaphore.acquire()
        try:
            task = asyncio.create_task(
                asyncio.to_thread(function, *args, **kwargs),
                name=f'batch-blocking:{operation}',
            )
        except BaseException:
            self._blocking_semaphore.release()
            raise
        self._blocking_tasks.add(task)
        watchdog = asyncio.create_task(
            self._watch_blocking_task(operation, task),
            name=f'batch-blocking-watchdog:{operation}',
        )
        self._blocking_watchdogs.add(watchdog)

        def discard_done(done: asyncio.Task[Any]) -> None:
            self._blocking_tasks.discard(done)
            self._blocking_semaphore.release()
            watchdog.cancel()
            with suppress(BaseException):
                done.result()

        def discard_watchdog(done: asyncio.Task[None]) -> None:
            self._blocking_watchdogs.discard(done)
            with suppress(BaseException):
                done.result()

        task.add_done_callback(discard_done)
        watchdog.add_done_callback(discard_watchdog)
        try:
            return await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self.config.manager_executor_hang_timeout.total_seconds(),
            )
        except TimeoutError as exc:
            with suppress(asyncio.CancelledError):
                await asyncio.shield(watchdog)
            raise TimeoutError(f'batch_blocking_operation_timeout:{operation}') from exc

    async def _prepare_materialize_chunk(
        self,
        parent_execution_id: str,
        *,
        snapshot: MaterializeChunkSnapshot,
        task_list_path: Path,
        data_list_path: Path | None,
        data_list_index: DataListIndex | None,
        chunk_size: int,
        max_items: int,
        max_line_bytes: int,
        max_chunk_bytes: int,
        max_data_frame_bytes: int,
        max_task_record_bytes: int,
    ) -> MaterializedTaskListChunk:
        if snapshot.record_version == 3:
            if self._task_store is None:
                raise DomainValidationError('batch_task_store_required_for_data_list')
            if data_list_path is None:
                raise DomainValidationError('invalid_batch_data_list', detail='data-list path is required')
            if data_list_index is None:
                raise DomainValidationError('invalid_batch_data_list', detail='data-list index is required')
        chunk = await self._run_blocking(
            'materialize-task-list-chunk',
            read_materialize_task_list_chunk,
            task_list_path,
            cursor=snapshot.cursor_bytes,
            materialized_count=snapshot.materialized_count,
            expected_size_bytes=snapshot.task_list_ref.size_bytes,
            expected_record_version=snapshot.record_version,
            chunk_size=chunk_size,
            max_items=max_items,
            max_line_bytes=max_line_bytes,
            max_chunk_bytes=max_chunk_bytes,
            max_metadata_bytes=self.config.batch_item_metadata_max_size_bytes,
        )
        if not chunk.records or self._task_store is None:
            return chunk

        write_records = []
        write_bytes = 0
        for record in chunk.records:
            task_store_record = await self._run_blocking(
                'encode-task-store-record',
                encode_task_store_record,
                parent_execution_id,
                record,
                data_list_path=data_list_path,
                data_list_index=data_list_index,
                max_data_frame_bytes=max_data_frame_bytes,
                max_task_record_bytes=max_task_record_bytes,
            )
            record_bytes = len(task_store_record.record_data)
            if write_records and write_bytes + record_bytes > self.config.batch_task_store_write_chunk_bytes:
                await self._task_store.put_records(write_records)
                write_records = []
                write_bytes = 0
            write_records.append(task_store_record)
            write_bytes += record_bytes
            if write_bytes >= self.config.batch_task_store_write_chunk_bytes:
                await self._task_store.put_records(write_records)
                write_records = []
                write_bytes = 0
        if write_records:
            await self._task_store.put_records(write_records)
        return chunk

    async def _assert_materialization_files_unchanged(
        self,
        *,
        task_list: _VerifiedTaskList,
        data_list: _VerifiedDataList | None,
    ) -> None:
        if not await self._run_blocking(
            'task-list-fingerprint',
            _fingerprint_matches,
            task_list.path,
            task_list.file_fingerprint,
        ):
            raise ValueError('batch_task_list_identity_mismatch')
        if data_list is not None and not await self._run_blocking(
            'data-list-fingerprint',
            _fingerprint_matches,
            data_list.path,
            data_list.file_fingerprint,
        ):
            raise DataListError('invalid_batch_data_list', detail='data-list file identity changed')

    async def _watch_blocking_task(self, operation: str, task: asyncio.Task[Any]) -> None:
        await asyncio.sleep(self.config.manager_executor_hang_timeout.total_seconds())
        if task.done() or self._on_blocking_timeout is None:
            return
        await self._on_blocking_timeout(operation)

    async def prepare(self, parent_execution_id: str, *, parent_attempt_id: str | None = None) -> Path:
        return await self.ensure_prepared(parent_execution_id, parent_attempt_id=parent_attempt_id)

    async def ensure_prepared(
        self,
        parent_execution_id: str,
        *,
        parent_attempt_id: str | None = None,
        bridge_timeout: float | None = None,
    ) -> Path:
        del bridge_timeout
        lock = self._prepare_locks.setdefault(parent_execution_id, asyncio.Lock())
        async with lock:
            return await self._ensure_prepared(
                parent_execution_id,
                parent_attempt_id=parent_attempt_id,
            )

    async def _ensure_prepared(
        self,
        parent_execution_id: str,
        *,
        parent_attempt_id: str | None,
    ) -> Path:
        async with self.session_factory() as session:
            repo = self._repository(session)
            await repo.fence_parent_attempt(parent_execution_id, parent_attempt_id)
            parent = await session.get(ExecutionRow, parent_execution_id)
            batch = await session.get(BatchRunRow, parent_execution_id)
            if parent is None or batch is None:
                raise KeyError(parent_execution_id)
            if parent_attempt_id is None:
                raise ValueError('batch_parent_attempt_required')
            if not parent.current_assignment_id:
                raise ValueError('batch_parent_assignment_required')
            assignment_id = parent.current_assignment_id
            generation = dict(parent.spec_json.get('generation') or {})
            limits = _generation_limits(parent.spec_json.get('generation_limits'))
            generation_epoch = batch.generation_epoch
            task_list_ref = ArtifactRef.model_validate(batch.task_list_ref_json) if batch.task_list_ref_json else None
            task_list_manifest = dict(batch.task_list_manifest_json) if batch.task_list_manifest_json else None
            control_state = batch.control_state
            await session.commit()

        if control_state != 'active':
            raise ValueError('batch_control_not_active')
        task_list_manifest_model = (
            TaskListManifest.model_validate(task_list_manifest) if task_list_manifest is not None else None
        )
        raw_data_list = generation.get('data_list')
        data_list_spec = DataListSpec.model_validate(raw_data_list) if raw_data_list is not None else None
        if task_list_manifest_model is not None and task_list_manifest_model.record_version == 3:
            if self._task_store is None:
                raise ValueError('batch_task_store_required_for_data_list')
            if data_list_spec is None:
                raise ValueError('invalid_batch_data_list')
        elif data_list_spec is not None:
            raise ValueError('invalid_batch_data_list')
        if task_list_ref is not None and task_list_ref.size_bytes > limits['max_task_list_bytes']:
            raise ValueError('batch_task_list_exceeds_byte_limit')
        if data_list_spec is not None and data_list_spec.ref.size_bytes > limits['max_data_list_bytes']:
            raise ValueError('batch_data_list_exceeds_byte_limit')
        workspace_limit = limits.get('max_workspace_bytes')
        workspace_size = (task_list_ref.size_bytes if task_list_ref is not None else 0) + (
            data_list_spec.ref.size_bytes if data_list_spec is not None else 0
        )
        if workspace_limit is not None and workspace_size > workspace_limit:
            raise ValueError('batch_workspace_limit_exceeded')

        root = Path(self._temporary_root.name) / parent_execution_id
        await self._run_blocking('prepare-workspace', root.mkdir, parents=True, exist_ok=True)
        task_list_path = root / 'tasks.jsonl'
        data_list_path = root / 'data_list.bin'
        prepared = self._prepared_task_lists.get(parent_execution_id)
        prepared_file_matches = (
            await self._run_blocking(
                'task-list-fingerprint',
                _fingerprint_matches,
                prepared.path,
                prepared.file_fingerprint,
            )
            if prepared is not None
            else False
        )
        prepared_data = self._prepared_data_lists.get(parent_execution_id)
        prepared_data_file_matches = (
            await self._run_blocking(
                'data-list-fingerprint',
                _fingerprint_matches,
                prepared_data.path,
                prepared_data.file_fingerprint,
            )
            if prepared_data is not None
            else False
        )
        prepared_data_matches = (
            prepared_data is None
            if data_list_spec is None
            else prepared_data is not None
            and _prepared_data_matches(
                prepared_data,
                generation_epoch=generation_epoch,
                ref=data_list_spec.ref,
            )
            and prepared_data_file_matches
        )
        if (
            prepared is not None
            and task_list_ref is not None
            and _prepared_matches(prepared, generation_epoch=generation_epoch, ref=task_list_ref)
            and prepared_file_matches
            and prepared_data_matches
        ):
            return prepared.path
        if prepared is not None:
            self._prepared_task_lists.pop(parent_execution_id, None)
            if not prepared_file_matches:
                self._verified_task_lists.pop(parent_execution_id, None)
        if prepared_data is not None:
            self._prepared_data_lists.pop(parent_execution_id, None)
            if not prepared_data_file_matches:
                self._verified_data_lists.pop(parent_execution_id, None)

        deadline = _GenerationDeadline(limits['timeout_seconds'])
        async with self.session_factory() as session:
            await self._repository(session).fence_parent_attempt(parent_execution_id, parent_attempt_id)
            batch = await session.get(BatchRunRow, parent_execution_id)
            if batch is not None:
                batch.preparation_started_at = utcnow()
                batch.preparation_deadline_at = batch.preparation_started_at + timedelta(seconds=deadline.remaining())
            await session.commit()
        if task_list_ref is None:
            if generation.get('mode') != 'remote_materialize':
                raise ValueError('batch_task_list_not_available')
            source = generation.get('source')
            if not isinstance(source, dict):
                raise ValueError('batch_generation_source_missing')
            staged_manifest = (
                TaskListManifest.model_validate(task_list_manifest) if task_list_manifest is not None else None
            )
            verified = (
                await self._verified_task_list(
                    parent_execution_id,
                    generation_epoch=generation_epoch,
                    sha256=staged_manifest.sha256,
                    size_bytes=staged_manifest.size_bytes,
                )
                if staged_manifest is not None
                else None
            )
            if verified is None:
                async with deadline.guard():
                    wedata_credential = await self._execution_credential(parent_execution_id, 'wedata')
                    cos_credential = await self._execution_credential(parent_execution_id, 'cos')
                    materialized = await self._materialize_remote(
                        source,
                        task_list_path,
                        limits=limits,
                        wedata_credential=wedata_credential,
                        cos_credential=cos_credential,
                    )
                    deadline.check()
                staged_manifest = TaskListManifest(
                    record_version=materialized.record_version,
                    item_count=materialized.total_items,
                    size_bytes=materialized.size_bytes,
                    sha256=materialized.sha256,
                    source={'mode': 'remote_materialize', 'definition': source},
                    datasets=[ResolvedDatasetVersion.model_validate(item) for item in materialized.resolved_datasets],
                    batch_metadata=materialized.batch_metadata or None,
                )
                verified = await self._remember_verified(
                    parent_execution_id,
                    generation_epoch=generation_epoch,
                    sha256=materialized.sha256,
                    size_bytes=materialized.size_bytes,
                    path=task_list_path,
                )
            else:
                task_list_path = verified.path
            assert staged_manifest is not None
            task_list_manifest = staged_manifest.storage_payload()
            async with deadline.guard():
                async with self.session_factory() as session:
                    repo = self._repository(session)
                    await repo.fence_parent_attempt(parent_execution_id, parent_attempt_id)
                    deadline.check()
                    await repo.stage_generated_task_list(
                        parent_execution_id,
                        manifest=task_list_manifest,
                        generation_epoch=generation_epoch,
                        now=utcnow(),
                    )
                    deadline.check()
                    await session.commit()
            task_list_ref = await self._publish_task_list(
                parent_execution_id=parent_execution_id,
                assignment_id=assignment_id,
                parent_attempt_id=parent_attempt_id,
                generation_epoch=generation_epoch,
                manifest=staged_manifest,
                path=task_list_path,
                deadline=deadline,
            )
            _validate_task_list_manifest_ref(task_list_manifest, task_list_ref)
        else:
            manifest = TaskListManifest.model_validate(task_list_manifest)
            _validate_task_list_manifest_ref(manifest, task_list_ref)
            verified = await self._verified_task_list(
                parent_execution_id,
                generation_epoch=generation_epoch,
                sha256=task_list_ref.sha256,
                size_bytes=task_list_ref.size_bytes,
            )
            if verified is None:
                await self._download_task_list(
                    parent_execution_id=parent_execution_id,
                    assignment_id=assignment_id,
                    parent_attempt_id=parent_attempt_id,
                    task_list_ref=task_list_ref,
                    destination=task_list_path,
                    deadline=deadline,
                )
                verified = await self._remember_verified(
                    parent_execution_id,
                    generation_epoch=generation_epoch,
                    sha256=task_list_ref.sha256,
                    size_bytes=task_list_ref.size_bytes,
                    path=task_list_path,
                )
            else:
                task_list_path = verified.path
        task_list_manifest_model = TaskListManifest.model_validate(task_list_manifest)
        verified_data: _VerifiedDataList | None = None
        if data_list_spec is not None:
            verified_data = await self._verified_data_list(
                parent_execution_id,
                generation_epoch=generation_epoch,
                sha256=data_list_spec.ref.sha256,
                size_bytes=data_list_spec.ref.size_bytes,
            )
            if verified_data is None:
                await self._download_data_list(
                    parent_execution_id=parent_execution_id,
                    assignment_id=assignment_id,
                    parent_attempt_id=parent_attempt_id,
                    data_list_ref=data_list_spec.ref,
                    destination=data_list_path,
                    deadline=deadline,
                )
                verified_data = await self._remember_verified_data_list(
                    parent_execution_id,
                    generation_epoch=generation_epoch,
                    spec=data_list_spec,
                    path=data_list_path,
                    limits=limits,
                )
            else:
                data_list_path = verified_data.path
        async with deadline.guard():
            await self._run_blocking('workspace-limit', _enforce_workspace_limit, root, workspace_limit)
        async with deadline.guard():
            await self._ensure_module_dependencies(
                parent_execution_id=parent_execution_id,
                parent_attempt_id=parent_attempt_id,
                task_list_path=task_list_path,
                max_line_bytes=limits['max_line_bytes'],
            )

        assert verified is not None
        while True:
            async with deadline.guard():
                async with self.session_factory() as session:
                    repo = self._repository(session)
                    await repo.fence_parent_attempt(parent_execution_id, parent_attempt_id)
                    snapshot = await repo.snapshot_task_list_materialization(
                        parent_execution_id,
                        expected_ref=task_list_ref,
                    )
                    deadline.check()
                    await session.commit()
            if snapshot.generation_complete:
                break

            async with deadline.guard():
                await self._assert_materialization_files_unchanged(task_list=verified, data_list=verified_data)
                chunk = await self._prepare_materialize_chunk(
                    parent_execution_id,
                    snapshot=snapshot,
                    task_list_path=task_list_path,
                    data_list_path=data_list_path if data_list_spec is not None else None,
                    data_list_index=verified_data.index if verified_data is not None else None,
                    chunk_size=500,
                    max_items=limits['max_items'],
                    max_line_bytes=limits['max_line_bytes'],
                    max_chunk_bytes=max(16 * 1024 * 1024, limits['max_line_bytes']),
                    max_data_frame_bytes=limits['max_data_frame_bytes'],
                    max_task_record_bytes=limits['max_batch_task_record_bytes'],
                )
                await self._assert_materialization_files_unchanged(task_list=verified, data_list=verified_data)
                deadline.check()

            try:
                async with deadline.guard():
                    async with self.session_factory() as session:
                        repo = self._repository(session)
                        await repo.fence_parent_attempt(parent_execution_id, parent_attempt_id)
                        result = await repo.commit_task_list_materialization_chunk(
                            parent_execution_id,
                            snapshot=snapshot,
                            chunk=chunk,
                            now=utcnow(),
                        )
                        deadline.check()
                        await session.commit()
            except MaterializationSnapshotStaleError:
                continue
            if result.complete:
                break
        if task_list_manifest_model.record_version == 3:
            assert data_list_spec is not None
            assert verified_data is not None
            while True:
                async with deadline.guard():
                    async with self.session_factory() as session:
                        repo = self._repository(session)
                        await repo.fence_parent_attempt(parent_execution_id, parent_attempt_id)
                        finalize_snapshot = await repo.snapshot_data_list_finalization(
                            parent_execution_id,
                            expected_ref=task_list_ref,
                        )
                        deadline.check()
                        await session.commit()

                async with deadline.guard():
                    await self._assert_materialization_files_unchanged(task_list=verified, data_list=verified_data)
                    item_count = await self._run_blocking(
                        'cross-validate-task-data-list',
                        cross_validate_task_data_list,
                        task_list_path,
                        expected_task_list_ref=task_list_ref,
                        data_list_index=verified_data.index,
                        max_items=limits['max_items'],
                        max_line_bytes=limits['max_line_bytes'],
                        max_metadata_bytes=self.config.batch_item_metadata_max_size_bytes,
                    )
                    await self._assert_materialization_files_unchanged(task_list=verified, data_list=verified_data)
                    deadline.check()

                try:
                    async with deadline.guard():
                        async with self.session_factory() as session:
                            repo = self._repository(session)
                            await repo.fence_parent_attempt(parent_execution_id, parent_attempt_id)
                            await repo.commit_data_list_finalization(
                                parent_execution_id,
                                snapshot=finalize_snapshot,
                                item_count=item_count,
                                now=utcnow(),
                            )
                            deadline.check()
                            await session.commit()
                except MaterializationSnapshotStaleError:
                    continue
                break
        fingerprint = await self._run_blocking(
            'task-list-fingerprint',
            _file_fingerprint,
            task_list_path,
            expected_size_bytes=task_list_ref.size_bytes,
        )
        self._prepared_task_lists[parent_execution_id] = _PreparedTaskList(
            generation_epoch=generation_epoch,
            artifact_id=task_list_ref.artifact_id,
            kind=task_list_ref.kind,
            sha256=task_list_ref.sha256,
            size_bytes=task_list_ref.size_bytes,
            path=task_list_path,
            file_fingerprint=fingerprint,
        )
        if data_list_spec is not None:
            data_fingerprint = await self._run_blocking(
                'data-list-fingerprint',
                _file_fingerprint,
                data_list_path,
                expected_size_bytes=data_list_spec.ref.size_bytes,
            )
            self._prepared_data_lists[parent_execution_id] = _PreparedDataList(
                generation_epoch=generation_epoch,
                artifact_id=data_list_spec.ref.artifact_id,
                kind=data_list_spec.ref.kind,
                sha256=data_list_spec.ref.sha256,
                size_bytes=data_list_spec.ref.size_bytes,
                path=data_list_path,
                file_fingerprint=data_fingerprint,
            )
        return task_list_path

    async def _ensure_module_dependencies(
        self,
        *,
        parent_execution_id: str,
        parent_attempt_id: str,
        task_list_path: Path,
        max_line_bytes: int,
    ) -> None:
        async with self.session_factory() as session:
            batch_repository = self._repository(session)
            await batch_repository.fence_parent_attempt(parent_execution_id, parent_attempt_id)
            batch = await session.get(BatchRunRow, parent_execution_id)
            if batch is None:
                raise KeyError(parent_execution_id)
            if batch.dependency_status != 'pending':
                return
            base_dependency_build_id = batch.child_dependency_build_id
            if not base_dependency_build_id:
                raise DomainConflictError('batch_base_dependency_build_missing')

            try:
                representative, selectors, selector_values, selector_digest = await self._run_blocking(
                    'module-dependencies',
                    _read_module_dependency_selectors,
                    task_list_path,
                    max_line_bytes=max_line_bytes,
                    max_metadata_bytes=self.config.batch_item_metadata_max_size_bytes,
                )
            except ModuleDependencyError as exc:
                raise DomainValidationError(exc.code, detail=exc.detail) from exc
            now = utcnow()
            if representative is None:
                await batch_repository.mark_child_dependency_skipped(
                    parent_execution_id,
                    expected_dependency_build_id=base_dependency_build_id,
                    now=now,
                )
                await session.commit()
                return
            dependency_repository = DependencyBuildRepository(session)
            base_build = await dependency_repository.get(base_dependency_build_id)
            if (
                base_build.provider_type is None
                or base_build.provider_config_ref is None
                or base_build.provider_config_snapshot is None
            ):
                raise DomainConflictError('batch_base_dependency_provider_snapshot_missing')
            augmented_spec = base_build.build_spec.model_copy(
                update={'module_selectors': list(selectors)},
            )
            augmented_build, _ = await dependency_repository.get_or_create(
                augmented_spec,
                provider_type=base_build.provider_type,
                provider_config_ref=base_build.provider_config_ref,
                provider_config_snapshot=base_build.provider_config_snapshot,
                now=now,
            )
            artifact_repository = ArtifactRepository(session)
            for ref in (
                augmented_spec.source_package,
                augmented_spec.nexus_wheel,
                augmented_spec.uv_lock,
            ):
                await artifact_repository.add_ref(
                    ref.artifact_id,
                    owner_principal_id=base_build.owner_principal_id,
                    environment_id=base_build.environment_id,
                    resource_type='dependency_build',
                    resource_id=augmented_build.dependency_build_id,
                    now=now,
                )
            module_dependencies = {
                'representative_item_id': representative.item_id,
                'selector_digest': selector_digest,
                'selectors': selector_values,
            }
            await batch_repository.bind_child_dependency_build(
                parent_execution_id,
                expected_dependency_build_id=base_dependency_build_id,
                dependency_build_id=augmented_build.dependency_build_id,
                module_dependencies=module_dependencies,
                now=now,
            )
            await session.commit()

    async def _download_task_list(
        self,
        *,
        parent_execution_id: str,
        assignment_id: str,
        parent_attempt_id: str,
        task_list_ref: ArtifactRef,
        destination: Path,
        deadline: _GenerationDeadline,
    ) -> None:
        attempt = 0
        while True:
            attempt += 1
            try:
                async with deadline.guard():
                    await self._artifact_content_client().download_artifact_to_path(
                        execution_id=parent_execution_id,
                        assignment_id=assignment_id,
                        attempt_id=parent_attempt_id,
                        artifact_ref=task_list_ref.model_dump(mode='json'),
                        destination=destination,
                        deadline=deadline.absolute,
                    )
                return
            except ArtifactGatewayTransientError:
                delay = min(0.25 * (2 ** min(attempt - 1, 5)), 5.0)
                async with deadline.guard():
                    await asyncio.sleep(delay)

    async def _download_data_list(
        self,
        *,
        parent_execution_id: str,
        assignment_id: str,
        parent_attempt_id: str,
        data_list_ref: ArtifactRef,
        destination: Path,
        deadline: _GenerationDeadline,
    ) -> None:
        attempt = 0
        while True:
            attempt += 1
            try:
                async with deadline.guard():
                    await self._artifact_content_client().download_artifact_to_path(
                        execution_id=parent_execution_id,
                        assignment_id=assignment_id,
                        attempt_id=parent_attempt_id,
                        artifact_ref=data_list_ref.model_dump(mode='json'),
                        destination=destination,
                        deadline=deadline.absolute,
                    )
                return
            except ArtifactGatewayTransientError:
                delay = min(0.25 * (2 ** min(attempt - 1, 5)), 5.0)
                async with deadline.guard():
                    await asyncio.sleep(delay)

    async def fill_window(
        self,
        parent_execution_id: str,
        *,
        max_inflight: int,
        submit_batch_size: int,
        parent_attempt_id: str | None = None,
    ) -> int:
        async with self.session_factory() as session:
            repo = self._repository(session)
            await repo.fence_parent_attempt(parent_execution_id, parent_attempt_id)
            batch = await session.get(BatchRunRow, parent_execution_id)
            parent = await session.get(ExecutionRow, parent_execution_id)
            if batch is None or parent is None:
                raise KeyError(parent_execution_id)
            limits = _generation_limits(parent.spec_json.get('generation_limits'))
            task_list_ref = ArtifactRef.model_validate(batch.task_list_ref_json) if batch.task_list_ref_json else None
            generation = parent.spec_json.get('generation')
            raw_data_list = generation.get('data_list') if isinstance(generation, dict) else None
            data_list_spec = DataListSpec.model_validate(raw_data_list) if raw_data_list is not None else None
            prepared = self._prepared_task_lists.get(parent_execution_id)
            if (
                prepared is None
                or task_list_ref is None
                or not _prepared_matches(prepared, generation_epoch=batch.generation_epoch, ref=task_list_ref)
            ):
                self._prepared_task_lists.pop(parent_execution_id, None)
                raise _BatchPreparationRequiredError(parent_execution_id)
            if not await self._run_blocking(
                'task-list-fingerprint',
                _fingerprint_matches,
                prepared.path,
                prepared.file_fingerprint,
            ):
                self._prepared_task_lists.pop(parent_execution_id, None)
                self._verified_task_lists.pop(parent_execution_id, None)
                raise _BatchPreparationRequiredError(parent_execution_id)
            prepared_data = self._prepared_data_lists.get(parent_execution_id)
            if data_list_spec is not None:
                if prepared_data is None or not _prepared_data_matches(
                    prepared_data,
                    generation_epoch=batch.generation_epoch,
                    ref=data_list_spec.ref,
                ):
                    self._prepared_data_lists.pop(parent_execution_id, None)
                    raise _BatchPreparationRequiredError(parent_execution_id)
                if not await self._run_blocking(
                    'data-list-fingerprint',
                    _fingerprint_matches,
                    prepared_data.path,
                    prepared_data.file_fingerprint,
                ):
                    self._prepared_data_lists.pop(parent_execution_id, None)
                    self._verified_data_lists.pop(parent_execution_id, None)
                    raise _BatchPreparationRequiredError(parent_execution_id)
            elif prepared_data is not None:
                self._prepared_data_lists.pop(parent_execution_id, None)
                self._verified_data_lists.pop(parent_execution_id, None)
            try:
                submitted = await repo.submit_window(
                    parent_execution_id,
                    task_list_path=prepared.path,
                    max_inflight=max_inflight,
                    submit_batch_size=submit_batch_size,
                    max_batch_bytes=max(32 * 1024 * 1024, limits['max_line_bytes']),
                    now=utcnow(),
                )
            except DomainValidationError as exc:
                if exc.code == 'batch_module_dependency_failed':
                    await session.commit()
                raise
            await session.commit()
            return submitted

    async def reconcile_once(
        self,
        parent_execution_id: str,
        *,
        max_inflight: int,
        submit_batch_size: int,
        parent_attempt_id: str | None = None,
    ) -> bool:
        control_state = await self._control_state(parent_execution_id, parent_attempt_id=parent_attempt_id)
        if control_state == 'finalized':
            await self.release_parent(parent_execution_id)
            return True
        if control_state == 'stopping':
            await self._drain_stop_pages(parent_execution_id, parent_attempt_id=parent_attempt_id)
            return False
        try:
            await self.fill_window(
                parent_execution_id,
                max_inflight=max_inflight,
                submit_batch_size=submit_batch_size,
                parent_attempt_id=parent_attempt_id,
            )
        except _BatchPreparationRequiredError:
            await self.ensure_prepared(parent_execution_id, parent_attempt_id=parent_attempt_id)
            await self.fill_window(
                parent_execution_id,
                max_inflight=max_inflight,
                submit_batch_size=submit_batch_size,
                parent_attempt_id=parent_attempt_id,
            )
        return False

    async def release_parent(self, parent_execution_id: str) -> None:
        self._prepared_task_lists.pop(parent_execution_id, None)
        self._verified_task_lists.pop(parent_execution_id, None)
        self._prepared_data_lists.pop(parent_execution_id, None)
        self._verified_data_lists.pop(parent_execution_id, None)
        self._prepare_locks.pop(parent_execution_id, None)
        await self._run_blocking(
            'remove-parent-workspace',
            shutil.rmtree,
            Path(self._temporary_root.name) / parent_execution_id,
            ignore_errors=True,
        )

    async def _verified_task_list(
        self,
        parent_execution_id: str,
        *,
        generation_epoch: int,
        sha256: str,
        size_bytes: int,
    ) -> _VerifiedTaskList | None:
        verified = self._verified_task_lists.get(parent_execution_id)
        if (
            verified is not None
            and verified.generation_epoch == generation_epoch
            and verified.sha256 == sha256
            and verified.size_bytes == size_bytes
            and await self._run_blocking(
                'task-list-fingerprint',
                _fingerprint_matches,
                verified.path,
                verified.file_fingerprint,
            )
        ):
            return verified
        self._verified_task_lists.pop(parent_execution_id, None)
        return None

    async def _remember_verified(
        self,
        parent_execution_id: str,
        *,
        generation_epoch: int,
        sha256: str,
        size_bytes: int,
        path: Path,
    ) -> _VerifiedTaskList:
        verified = _VerifiedTaskList(
            generation_epoch=generation_epoch,
            sha256=sha256,
            size_bytes=size_bytes,
            path=path,
            file_fingerprint=await self._run_blocking(
                'task-list-fingerprint',
                _file_fingerprint,
                path,
                expected_size_bytes=size_bytes,
            ),
        )
        self._verified_task_lists[parent_execution_id] = verified
        return verified

    async def _verified_data_list(
        self,
        parent_execution_id: str,
        *,
        generation_epoch: int,
        sha256: str,
        size_bytes: int,
    ) -> _VerifiedDataList | None:
        verified = self._verified_data_lists.get(parent_execution_id)
        if (
            verified is not None
            and verified.generation_epoch == generation_epoch
            and verified.sha256 == sha256
            and verified.size_bytes == size_bytes
            and await self._run_blocking(
                'data-list-fingerprint',
                _fingerprint_matches,
                verified.path,
                verified.file_fingerprint,
            )
        ):
            return verified
        self._verified_data_lists.pop(parent_execution_id, None)
        return None

    async def _remember_verified_data_list(
        self,
        parent_execution_id: str,
        *,
        generation_epoch: int,
        spec: DataListSpec,
        path: Path,
        limits: Mapping[str, int],
    ) -> _VerifiedDataList:
        file_fingerprint = await self._run_blocking(
            'data-list-fingerprint',
            _file_fingerprint,
            path,
            expected_size_bytes=spec.ref.size_bytes,
        )
        index = await self._run_blocking(
            'scan-data-list',
            _validate_data_list_file,
            path,
            spec=spec,
            max_frame_bytes=limits['max_data_frame_bytes'],
            max_bytes=limits['max_data_list_bytes'],
            max_items=limits['max_items'],
        )
        if not await self._run_blocking(
            'data-list-fingerprint',
            _data_list_unchanged,
            path,
            file_fingerprint,
            spec.ref.sha256,
        ):
            raise DataListError('invalid_batch_data_list', detail='data-list file identity changed during validation')
        verified = _VerifiedDataList(
            generation_epoch=generation_epoch,
            sha256=spec.ref.sha256,
            size_bytes=spec.ref.size_bytes,
            path=path,
            file_fingerprint=file_fingerprint,
            index=index,
        )
        self._verified_data_lists[parent_execution_id] = verified
        return verified

    async def _publish_task_list(
        self,
        *,
        parent_execution_id: str,
        assignment_id: str,
        parent_attempt_id: str,
        generation_epoch: int,
        manifest: TaskListManifest,
        path: Path,
        deadline: _GenerationDeadline,
    ) -> ArtifactRef:
        retry_delay = 0.25
        while True:
            try:
                async with deadline.guard():
                    return ArtifactRef.model_validate(
                        await self._artifact_content_client().publish_task_list(
                            parent_execution_id=parent_execution_id,
                            assignment_id=assignment_id,
                            attempt_id=parent_attempt_id,
                            path=path,
                            sha256=manifest.sha256,
                            size_bytes=manifest.size_bytes,
                        )
                    )
            except ArtifactGatewayTransientError:
                linked = await self._linked_task_list_ref(
                    parent_execution_id=parent_execution_id,
                    parent_attempt_id=parent_attempt_id,
                    generation_epoch=generation_epoch,
                    manifest=manifest,
                    deadline=deadline,
                )
                if linked is not None:
                    return linked
                async with deadline.guard():
                    await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 5.0)

    async def _linked_task_list_ref(
        self,
        *,
        parent_execution_id: str,
        parent_attempt_id: str,
        generation_epoch: int,
        manifest: TaskListManifest,
        deadline: _GenerationDeadline,
    ) -> ArtifactRef | None:
        async with deadline.guard():
            async with self.session_factory() as session:
                repo = self._repository(session)
                await repo.fence_parent_attempt(parent_execution_id, parent_attempt_id)
                batch = await session.get(BatchRunRow, parent_execution_id)
                if batch is None:
                    raise KeyError(parent_execution_id)
                if batch.generation_epoch != generation_epoch:
                    raise ValueError('batch_generation_epoch_fenced')
                if batch.task_list_ref_json is None:
                    await session.commit()
                    return None
                linked_manifest = TaskListManifest.model_validate(batch.task_list_manifest_json)
                if linked_manifest.storage_payload() != manifest.storage_payload():
                    raise ValueError('batch_generation_nondeterministic')
                ref = ArtifactRef.model_validate(batch.task_list_ref_json)
                _validate_task_list_manifest_ref(linked_manifest, ref)
                await session.commit()
                return ref

    async def stop_children(self, parent_execution_id: str, *, parent_attempt_id: str | None = None) -> None:
        async with self.session_factory() as session:
            repo = self._repository(session)
            await repo.fence_parent_attempt(parent_execution_id, parent_attempt_id)
            parent = await session.get(ExecutionRow, parent_execution_id)
            batch = await session.get(BatchRunRow, parent_execution_id)
            if parent is None or batch is None:
                raise KeyError(parent_execution_id)
            reason = batch.stop_reason if batch.control_state == 'stopping' else _batch_stop_reason(parent.stop_reason)
            await repo.begin_stop(parent_execution_id, reason=reason, now=utcnow())
            await session.commit()
        await self._drain_stop_pages(parent_execution_id, parent_attempt_id=parent_attempt_id)

    async def _control_state(self, parent_execution_id: str, *, parent_attempt_id: str | None) -> str:
        async with self.session_factory() as session:
            batch = await session.get(BatchRunRow, parent_execution_id)
            if batch is None:
                raise KeyError(parent_execution_id)
            if batch.control_state == 'finalized':
                return batch.control_state
            repo = self._repository(session)
            await repo.fence_parent_attempt(parent_execution_id, parent_attempt_id)
            control_state = batch.control_state
            await session.commit()
            return control_state

    async def _drain_stop_pages(self, parent_execution_id: str, *, parent_attempt_id: str | None) -> None:
        while True:
            async with self.session_factory() as session:
                repo = self._repository(session)
                await repo.fence_parent_attempt(parent_execution_id, parent_attempt_id)
                batch = await session.get(BatchRunRow, parent_execution_id)
                if batch is None or batch.stop_reason is None:
                    raise ValueError('batch_not_stopping')
                page = await repo.next_stop_page(parent_execution_id, page_size=500, now=utcnow())
                execution_repo = ExecutionRepository(session, shard_count=self.config.shard_count)
                await execution_repo.request_stop_many(
                    page.child_execution_ids,
                    stop_reason=batch.stop_reason,
                    now=utcnow(),
                )
                await session.commit()
            if page.complete:
                return

    @measured('materialization', family='stage')
    async def _materialize_remote(
        self,
        source: dict[str, Any],
        task_list_path: Path,
        *,
        limits: dict[str, int],
        wedata_credential: dict[str, Any] | None,
        cos_credential: dict[str, Any] | None = None,
    ) -> MaterializedTaskList:
        timeout = limits['timeout_seconds']
        task_list_version = _task_list_version(source.get('task_list_version', 1))
        if source.get('type') == 'generator':
            return await _materialize_generator_subprocess(
                Path(str(source.get('path') or '')),
                task_list_path,
                max_items=limits['max_items'],
                max_metadata_bytes=self.config.batch_item_metadata_max_size_bytes,
                max_bytes=limits['max_task_list_bytes'],
                max_line_bytes=limits['max_line_bytes'],
                max_workspace_bytes=limits.get('max_workspace_bytes'),
                timeout=timeout,
                task_list_version=task_list_version,
                wedata_credential=wedata_credential,
                cos_credential=cos_credential,
                blocking_runner=self._run_blocking,
            )
        if source.get('type') != 'compose' or not isinstance(source.get('compose'), dict):
            raise ValueError('batch_generation_source_invalid')
        requires_wedata = await self._run_blocking(
            'compose-credential-requirement',
            _compose_requires_wedata,
            source['compose'],
        )
        if 'cos' in _compose_credential_kinds(source['compose']) and cos_credential is None:
            raise ValueError('batch_data_credential_required:cos')
        if requires_wedata and wedata_credential is None:
            raise ValueError('batch_data_credential_required:wedata')
        return await _materialize_compose_subprocess(
            source['compose'],
            task_list_path,
            max_items=limits['max_items'],
            max_metadata_bytes=self.config.batch_item_metadata_max_size_bytes,
            max_bytes=limits['max_task_list_bytes'],
            max_line_bytes=limits['max_line_bytes'],
            max_workspace_bytes=limits.get('max_workspace_bytes'),
            timeout=timeout,
            task_list_version=task_list_version,
            wedata_credential=wedata_credential,
            cos_credential=cos_credential,
            blocking_runner=self._run_blocking,
        )

    async def _execution_credential(self, execution_id: str, kind: str) -> dict[str, Any] | None:
        async with self.session_factory() as session:
            return await ExecutionCredentialRepository(session).get_payload(
                execution_id,
                kind,
                now=utcnow(),
            )

    def _artifact_content_client(self) -> BatchArtifactClient:
        if self._artifact_client is not None:
            return self._artifact_client
        self._artifact_transport = create_application_gateway_artifact_transport(self.config, required=True)
        assert self._artifact_transport is not None
        self._owns_artifact_transport = True
        self._artifact_client = TaskListArtifactGateway(self._artifact_transport)
        return self._artifact_client


class _GenerationDeadline:
    def __init__(self, timeout_seconds: int) -> None:
        self._deadline = asyncio.get_running_loop().time() + timeout_seconds

    def check(self) -> None:
        self.remaining()

    @property
    def absolute(self) -> float:
        return self._deadline

    def remaining(self) -> float:
        remaining = self._deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError('batch_generation_timeout')
        return remaining

    @asynccontextmanager
    async def guard(self) -> AsyncIterator[None]:
        timeout = async_timeout(self.remaining())
        try:
            async with timeout:
                yield
        except TimeoutError:
            if timeout.expired():
                raise TimeoutError('batch_generation_timeout') from None
            raise


def _generation_limits(value: Any) -> dict[str, int]:
    raw = value if isinstance(value, dict) else {}
    defaults = {
        'max_items': 10_000,
        'max_task_list_bytes': 1024**3,
        'max_line_bytes': 16 * 1024**2,
        'timeout_seconds': 3600,
        'max_workspace_bytes': 4 * 1024**3,
        'max_data_frame_bytes': 256 * 1024**2,
        'max_batch_task_record_bytes': 256 * 1024**2,
        'max_data_list_bytes': 4 * 1024**3,
    }
    limits: dict[str, int] = {}
    for key, default in defaults.items():
        candidate = raw.get(key, default)
        if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate < 1:
            raise ValueError(f'invalid_batch_generation_limit:{key}')
        limits[key] = candidate
    return limits


def _batch_stop_reason(execution_stop_reason: str | None) -> str:
    return 'parent_ttl' if execution_stop_reason in {'execution_ttl', 'parent_ttl'} else 'user_stop'


def _validate_task_list_manifest_ref(
    value: TaskListManifest | Mapping[str, Any],
    ref: ArtifactRef,
) -> None:
    manifest = value if isinstance(value, TaskListManifest) else TaskListManifest.model_validate(value)
    if manifest.sha256 != ref.sha256 or manifest.size_bytes != ref.size_bytes:
        raise ValueError('batch_task_list_manifest_identity_mismatch')


def _prepared_matches(
    prepared: _PreparedTaskList,
    *,
    generation_epoch: int,
    ref: ArtifactRef,
) -> bool:
    return (
        prepared.generation_epoch == generation_epoch
        and prepared.artifact_id == ref.artifact_id
        and prepared.kind == ref.kind
        and prepared.sha256 == ref.sha256
        and prepared.size_bytes == ref.size_bytes
    )


def _prepared_data_matches(
    prepared: _PreparedDataList,
    *,
    generation_epoch: int,
    ref: ArtifactRef,
) -> bool:
    return (
        prepared.generation_epoch == generation_epoch
        and prepared.artifact_id == ref.artifact_id
        and prepared.kind == ref.kind
        and prepared.sha256 == ref.sha256
        and prepared.size_bytes == ref.size_bytes
    )


def _validate_data_list_file(
    path: Path,
    *,
    spec: DataListSpec,
    max_frame_bytes: int,
    max_bytes: int,
    max_items: int,
) -> DataListIndex:
    try:
        index = scan_data_list(
            path,
            max_frame_bytes=max_frame_bytes,
            max_bytes=max_bytes,
            max_items=max_items,
        )
    except DataListError:
        raise
    manifest = index.manifest
    if (
        manifest.sha256 != spec.ref.sha256
        or manifest.size_bytes != spec.ref.size_bytes
        or manifest.sha256 != spec.manifest.sha256
        or manifest.size_bytes != spec.manifest.size_bytes
        or manifest.frame_count != spec.manifest.frame_count
    ):
        raise DataListError('invalid_batch_data_list', detail='Artifact identity does not match data-list')
    return index


def _file_fingerprint(path: Path, *, expected_size_bytes: int) -> _LocalFileFingerprint:
    stat = path.stat()
    if stat.st_size != expected_size_bytes:
        raise ValueError('batch_task_list_identity_mismatch')
    return _LocalFileFingerprint(
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        ctime_ns=stat.st_ctime_ns,
        inode=stat.st_ino,
        device=stat.st_dev,
    )


def _fingerprint_matches(path: Path, expected: _LocalFileFingerprint) -> bool:
    try:
        return _file_fingerprint(path, expected_size_bytes=expected.size_bytes) == expected
    except (OSError, ValueError):
        return False


def _data_list_unchanged(path: Path, expected: _LocalFileFingerprint, expected_sha256: str) -> bool:
    """Prove the data-list file was not rewritten while it was being validated.

    ``stat`` alone cannot answer this. An in-place rewrite that preserves the length keeps
    the size, the inode and the device, so ``st_mtime_ns``/``st_ctime_ns`` are the only
    fields left to detect it -- and filesystems such as ext4 derive those from a coarse
    per-jiffy clock, which makes a rewrite landing in the same tick as the original
    materialization completely invisible. Because the cached ``DataListIndex`` hands out raw
    frame offsets into this file, re-reading the digest is the only portable way to show the
    bytes behind those offsets are still the ones that were validated. The cheap stat check
    runs first so an obvious swap fails without hashing the file again.
    """
    if not _fingerprint_matches(path, expected):
        return False
    try:
        size_bytes, sha256 = _file_identity(path)
    except OSError:
        return False
    return size_bytes == expected.size_bytes and sha256 == expected_sha256


def _file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _compose_credential_kinds(compose: Mapping[str, Any]) -> set[str]:
    kinds: set[str] = set()
    for part in compose.get('parts', []):
        if not isinstance(part, Mapping):
            continue
        value = part.get('value')
        if part.get('kind') == 'use' and isinstance(value, str) and value.startswith('data:'):
            value = value[len('data:') :]
        elif part.get('kind') != 'data':
            continue
        if isinstance(value, str):
            kinds.add('cos' if value.startswith('cos://') else 'wedata')
    return kinds


def _compose_requires_wedata(compose: Mapping[str, Any]) -> bool:
    return 'wedata' in _compose_credential_kinds(compose)


def _read_module_dependency_selectors(
    task_list_path: Path,
    *,
    max_line_bytes: int,
    max_metadata_bytes: int = DEFAULT_SERVER_BATCH_ITEM_METADATA_BYTES,
) -> tuple[RepresentativeRunInput | None, tuple[ModuleSelector, ...], list[dict[str, Any]], str | None]:
    representative = read_representative_run_input(
        task_list_path,
        max_line_bytes=max_line_bytes,
        max_metadata_bytes=max_metadata_bytes,
    )
    selectors = selectors_from_run_input(representative.run_input) if representative is not None else ()
    selector_values = [selector.model_dump(mode='json') for selector in selectors]
    selector_digest = canonical_sha256(selector_values) if representative is not None else None
    return representative, selectors, selector_values, selector_digest


@measured('task_generation', family='stage')
async def _materialize_generator_subprocess(
    generator_path: Path,
    output_path: Path,
    *,
    max_items: int,
    max_bytes: int,
    max_line_bytes: int = 16 * 1024**2,
    max_workspace_bytes: int | None = None,
    timeout: int,
    task_list_version: TaskListVersion = 1,
    wedata_credential: dict[str, Any] | None = None,
    cos_credential: dict[str, Any] | None = None,
    blocking_runner: Callable[..., Awaitable[Any]] | None = None,
    max_metadata_bytes: int = DEFAULT_SERVER_BATCH_ITEM_METADATA_BYTES,
) -> MaterializedTaskList:
    generator_path = generator_path.resolve()
    return await _materialize_task_list_subprocess(
        ['--generator', str(generator_path)],
        output_path,
        operation='generator',
        max_items=max_items,
        max_bytes=max_bytes,
        max_line_bytes=max_line_bytes,
        max_metadata_bytes=max_metadata_bytes,
        max_workspace_bytes=max_workspace_bytes,
        timeout=timeout,
        task_list_version=task_list_version,
        wedata_credential=wedata_credential,
        cos_credential=cos_credential,
        blocking_runner=blocking_runner,
    )


async def _materialize_compose_subprocess(
    compose: dict[str, Any],
    output_path: Path,
    *,
    max_items: int,
    max_bytes: int,
    max_line_bytes: int,
    timeout: int,
    max_workspace_bytes: int | None = None,
    task_list_version: TaskListVersion = 1,
    wedata_credential: dict[str, Any] | None = None,
    cos_credential: dict[str, Any] | None = None,
    blocking_runner: Callable[..., Awaitable[Any]] | None = None,
    max_metadata_bytes: int = DEFAULT_SERVER_BATCH_ITEM_METADATA_BYTES,
) -> MaterializedTaskList:
    request_path = output_path.with_name(f'{output_path.name}.compose.json')
    await _run_materialize_blocking(
        blocking_runner,
        'compose-request',
        _write_compose_request,
        request_path,
        compose,
    )
    try:
        return await _materialize_task_list_subprocess(
            ['--compose-request', str(request_path)],
            output_path,
            operation='compose',
            max_items=max_items,
            max_bytes=max_bytes,
            max_line_bytes=max_line_bytes,
            max_metadata_bytes=max_metadata_bytes,
            max_workspace_bytes=max_workspace_bytes,
            timeout=timeout,
            task_list_version=task_list_version,
            wedata_credential=wedata_credential,
            cos_credential=cos_credential,
            blocking_runner=blocking_runner,
        )
    finally:
        await _run_materialize_blocking(
            blocking_runner, 'compose-request-cleanup', request_path.unlink, missing_ok=True
        )


def _write_compose_request(path: Path, compose: dict[str, Any]) -> None:
    payload = {
        'version': 1,
        'compose': compose,
        'requires_wedata': _compose_requires_wedata(compose),
    }
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    if len(content) > COMPOSE_REQUEST_MAX_BYTES:
        raise ValueError('batch compose request exceeds size limit')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


async def _run_materialize_blocking(
    blocking_runner: Callable[..., Awaitable[Any]] | None,
    operation: str,
    function: Callable[..., _T],
    /,
    *args: Any,
    **kwargs: Any,
) -> _T:
    if blocking_runner is not None:
        return await blocking_runner(operation, function, *args, **kwargs)
    return await asyncio.to_thread(function, *args, **kwargs)


async def _materialize_task_list_subprocess(
    source_arguments: list[str],
    output_path: Path,
    *,
    operation: str,
    max_items: int,
    max_bytes: int,
    max_line_bytes: int,
    max_workspace_bytes: int | None,
    timeout: int,
    task_list_version: TaskListVersion,
    wedata_credential: dict[str, Any] | None,
    cos_credential: dict[str, Any] | None = None,
    blocking_runner: Callable[..., Awaitable[Any]] | None,
    max_metadata_bytes: int = DEFAULT_SERVER_BATCH_ITEM_METADATA_BYTES,
) -> MaterializedTaskList:
    await _run_materialize_blocking(
        blocking_runner,
        'materialize-workspace',
        output_path.parent.mkdir,
        parents=True,
        exist_ok=True,
    )
    result_path = output_path.with_name(f'{output_path.name}.result.json')
    await _run_materialize_blocking(blocking_runner, 'materialize-result-cleanup', result_path.unlink, missing_ok=True)
    process_environment = {
        **os.environ,
        'TMPDIR': str(output_path.parent),
        'TMP': str(output_path.parent),
        'TEMP': str(output_path.parent),
        BATCH_ITEM_METADATA_BYTES_ENV: str(max_metadata_bytes),
    }
    for name in ('COS_SECRET_ID', 'COS_SECRET_KEY', 'COS_TOKEN', 'COS_KEY_FILE', 'TAI_PAT', 'TAIHU_PAT'):
        process_environment.pop(name, None)
    process_environment['NEXUS_DATA_ACCESS_SUBMISSION'] = '1'
    if cos_credential is not None:
        process_environment['COS_SECRET_ID'] = cos_credential['secret_id']
        process_environment['COS_SECRET_KEY'] = cos_credential['secret_key']
        if cos_credential.get('token'):
            process_environment['COS_TOKEN'] = cos_credential['token']
    process_environment.pop('WEDATA_KEY', None)
    process_environment.pop('WEDATA_KEY_FILE', None)
    if wedata_credential is not None:
        process_environment['WEDATA_KEY'] = json.dumps(
            WedataKey.from_mapping(wedata_credential, source='execution credential').to_auth_payload(),
            ensure_ascii=False,
            separators=(',', ':'),
        )
    succeeded = False
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            '-m',
            'nexus.distributed.batch.materialize',
            *source_arguments,
            '--output',
            str(output_path),
            '--result',
            str(result_path),
            '--max-items',
            str(max_items),
            '--max-bytes',
            str(max_bytes),
            '--max-line-bytes',
            str(max_line_bytes),
            '--task-list-version',
            str(task_list_version),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=output_path.parent,
            env=process_environment,
            start_new_session=True,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_task = asyncio.create_task(
            _read_bounded_tail(process.stdout),
            name=f'batch-generator-stdout:{process.pid}',
        )
        stderr_task = asyncio.create_task(
            _read_bounded_tail(process.stderr),
            name=f'batch-generator-stderr:{process.pid}',
        )
        try:
            timed_out = not await _wait_for_process_exit(process, timeout=timeout)
        finally:
            await _terminate_process_group(process)
            stdout_tail, stderr_tail = await asyncio.gather(stdout_task, stderr_task)

        if timed_out:
            raise TimeoutError(f'batch_{operation}_timeout')
        if process.returncode != 0:
            detail = _generator_failure_detail(stdout_tail, stderr_tail, process.returncode)
            raise RuntimeError(f'batch_{operation}_failed:{detail}')

        materialized = await _run_materialize_blocking(
            blocking_runner,
            'materialize-result',
            _load_materialize_result,
            result_path,
            output_path,
            max_items=max_items,
            max_bytes=max_bytes,
            task_list_version=task_list_version,
        )
        await _run_materialize_blocking(
            blocking_runner,
            'materialize-result-cleanup',
            result_path.unlink,
            missing_ok=True,
        )
        await _run_materialize_blocking(
            blocking_runner,
            'workspace-limit',
            _enforce_workspace_limit,
            output_path.parent,
            max_workspace_bytes,
            error=(
                'batch_generator_workspace_limit_exceeded'
                if operation == 'generator'
                else 'batch_workspace_limit_exceeded'
            ),
        )
        succeeded = True
        return materialized
    finally:
        await _run_materialize_blocking(
            blocking_runner,
            'materialize-result-cleanup',
            result_path.unlink,
            missing_ok=True,
        )
        if not succeeded:
            await _run_materialize_blocking(
                blocking_runner,
                'materialize-output-cleanup',
                output_path.unlink,
                missing_ok=True,
            )


def _load_materialize_result(
    result_path: Path,
    output_path: Path,
    *,
    max_items: int,
    max_bytes: int,
    task_list_version: TaskListVersion = 1,
) -> MaterializedTaskList:
    try:
        if result_path.stat().st_size > _MATERIALIZE_RESULT_MAX_BYTES:
            raise ValueError('result file is too large')
        payload = json.loads(result_path.read_bytes())
        if not isinstance(payload, dict):
            raise ValueError('result fields are invalid')
        version = payload.get('version')
        if isinstance(version, bool) or version not in {1, 2, 3}:
            raise ValueError('result version is invalid')
        expected_fields = {'version', 'sha256', 'size_bytes', 'total_items'}
        if version == 2:
            expected_fields.add('batch_metadata')
        elif version == 3:
            expected_fields.add('resolved_datasets')
        if set(payload) != expected_fields:
            raise ValueError('result fields are invalid')
        sha256 = payload['sha256']
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in '0123456789abcdef' for character in sha256)
        ):
            raise ValueError('result sha256 is invalid')
        size_bytes = _bounded_result_integer(payload['size_bytes'], maximum=max_bytes)
        total_items = _bounded_result_integer(payload['total_items'], maximum=max_items)
        batch_metadata = payload.get('batch_metadata', {})
        canonical_batch_metadata(batch_metadata)
        resolved_datasets = tuple(
            ResolvedDatasetVersion.model_validate(item).model_dump(mode='json')
            for item in payload.get('resolved_datasets', [])
        )
        materialized = MaterializedTaskList(
            path=output_path,
            sha256=sha256,
            size_bytes=size_bytes,
            total_items=total_items,
            record_version=task_list_version,
            resolved_datasets=resolved_datasets,
            batch_metadata=batch_metadata,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError('batch_generator_invalid_result') from exc
    if _file_identity(output_path) != (materialized.size_bytes, materialized.sha256):
        raise ValueError('batch_generator_output_identity_mismatch')
    return materialized


def _bounded_result_integer(value: object, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > maximum:
        raise ValueError('result integer is invalid')
    return value


def _task_list_version(value: object) -> TaskListVersion:
    if isinstance(value, bool) or not isinstance(value, int) or value not in {1, 2}:
        raise ValueError('unsupported task-list version')
    return cast(TaskListVersion, value)


async def _read_bounded_tail(stream: asyncio.StreamReader) -> bytes:
    tail = bytearray()
    while chunk := await stream.read(64 * 1024):
        tail.extend(chunk)
        if len(tail) > _GENERATOR_LOG_TAIL_BYTES:
            del tail[:-_GENERATOR_LOG_TAIL_BYTES]
    return bytes(tail)


async def _wait_for_process_exit(process: asyncio.subprocess.Process, *, timeout: int) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while process.returncode is None:
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(0.05, remaining))
    return True


def _generator_failure_detail(stdout_tail: bytes, stderr_tail: bytes, returncode: int | None) -> str:
    detail = (stderr_tail or stdout_tail).decode(errors='replace').strip()
    return detail[-4096:] or str(returncode)


def _enforce_workspace_limit(
    workspace: Path,
    max_workspace_bytes: int | None,
    *,
    error: str = 'batch_workspace_limit_exceeded',
) -> None:
    if max_workspace_bytes is not None and _directory_size(workspace) > max_workspace_bytes:
        raise ValueError(error)


def _directory_size(root: Path) -> int:
    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
    return total


async def _terminate_process_group(
    process: asyncio.subprocess.Process,
    *,
    grace_seconds: float = 0.5,
) -> None:
    term_sent = False
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
        term_sent = True
    if term_sent:
        deadline = asyncio.get_running_loop().time() + grace_seconds
        if process.returncode is None:
            with suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining > 0:
            await asyncio.sleep(remaining)
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            pass
        else:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
    if process.returncode is None:
        await process.wait()
