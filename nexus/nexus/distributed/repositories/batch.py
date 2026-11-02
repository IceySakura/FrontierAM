from __future__ import annotations

import hashlib
import json

from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import and_, case, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import load_only

from nexus.distributed.analytics.fields import task_name_from_input
from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.batch.data_list import (
    DataListError,
    DataListIndex,
    DataRef,
    read_indexed_frame,
)
from nexus.distributed.batch.task_list import (
    DEFAULT_SERVER_BATCH_ITEM_METADATA_BYTES,
    canonical_batch_item_metadata,
    is_valid_batch_item_id,
)
from nexus.distributed.batch.task_store import (
    BatchTaskRecord,
    BatchTaskStore,
    validate_batch_task_files_classification,
)
from nexus.distributed.config.views import BatchRepositoryConfigView
from nexus.distributed.dependency_build.models import DependencyBuildStatus
from nexus.distributed.dependency_build.wheelhouse import WheelhouseMetadata
from nexus.distributed.environment.models import EnvironmentAllocationStatus
from nexus.distributed.errors import (
    DomainConflictError,
    DomainValidationError,
    FencingError,
    ResourceNotFoundError,
    TemporaryUnavailableError,
)
from nexus.distributed.execution.models import (
    AssignmentStatus,
    ExecutionStatus,
    RouteStatus,
)
from nexus.distributed.persistence.orm import (
    ArtifactRow,
    AssignmentRow,
    BatchItemAttemptRow,
    BatchItemRow,
    BatchRunRow,
    DependencyBuildRow,
    EnvironmentAllocationRow,
    ExecutionRow,
    InstanceManagerRow,
)
from nexus.distributed.repositories.artifact import ArtifactRepository
from nexus.distributed.repositories.events import append_control_event_row
from nexus.distributed.repositories.execution import CreateExecution, ExecutionRepository
from nexus.distributed.submission.input import InputNormalizationError, normalize_run_input
from nexus.distributed.submission.specs import TaskListManifest
from nexus.distributed.utils.identity import canonical_sha256, new_id


_ACTIVE_CHILD_STATUSES = {'queued', 'assigned', 'starting', 'running', 'waiting_input', 'stopping'}
_TERMINAL_ITEM_STATUSES = {'succeeded', 'failed', 'killed', 'lost', 'expired', 'cancelled'}
_FORBIDDEN_RUN_INPUT_KEYS = {
    'allocation_id',
    'driver_config',
    'environment_id',
    'instance_mode',
    'mode',
    'owner',
    'owner_principal_id',
    'placement',
    'resources',
    'spec_type',
    'workload',
}


@dataclass(frozen=True, slots=True)
class MaterializeChunkResult:
    inserted: int
    cursor_bytes: int
    complete: bool


@dataclass(frozen=True, slots=True)
class MaterializeChunkSnapshot:
    generation_epoch: int
    task_list_ref: ArtifactRef
    record_version: int
    cursor_bytes: int
    materialized_count: int
    total_items: int | None
    generation_complete: bool


class MaterializationSnapshotStaleError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RepresentativeRunInput:
    item_id: str
    run_input: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _TaskListRecord:
    version: int
    item_id: str
    run_input: dict[str, Any]
    metadata: dict[str, Any]
    data_ref: DataRef | None = None

    def identity(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            'version': self.version,
            'item_id': self.item_id,
            'run_input': self.run_input,
        }
        if self.version in {2, 3}:
            value['metadata'] = self.metadata
        if self.version == 3:
            value['data_ref'] = self.data_ref.model_dump(mode='json') if self.data_ref is not None else None
        return value


@dataclass(frozen=True, slots=True)
class MaterializedTaskListChunk:
    records: tuple[tuple[_TaskListRecord, int, int, str], ...]
    cursor_bytes: int
    complete: bool


@dataclass(frozen=True, slots=True)
class _SubmitWindowItem:
    item_id: str
    task_offset: int
    task_length: int
    current_attempt_no: int
    spec_sha256: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _PreparedChildExecution:
    item_id: str
    attempt_no: int
    child_spec_sha256: str
    request: CreateExecution


@dataclass(frozen=True, slots=True)
class _PreparedChildDependencyBinding:
    allocation_spec: dict[str, Any]
    allocation_spec_hash: str
    child_template: dict[str, Any]
    module_dependencies: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StopPageResult:
    scanned: int
    child_execution_ids: tuple[str, ...]
    cursor: str | None
    complete: bool


@dataclass(frozen=True, slots=True)
class SyncChildStatesPageResult:
    scanned: int
    changed: int
    cursor: str | None
    complete: bool
    terminal_discovery_lags: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class BatchFinalizationResult:
    parent_status: str
    summary: dict[str, int]
    finalized: bool = True
    result_artifact_required: bool = False
    assignment_id: str | None = None
    attempt_id: str | None = None


@dataclass(frozen=True, slots=True)
class BatchLifecycleClaim:
    parent_execution_id: str
    owner_id: str
    operation_id: str
    lease_expires_at: datetime


class BatchRepository:
    def __init__(
        self,
        session: AsyncSession,
        *,
        config: BatchRepositoryConfigView,
        task_store: BatchTaskStore | None = None,
        blocking_runner: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self.session = session
        self.config = config
        self.task_store = task_store
        self._blocking_runner = blocking_runner

    async def _run_blocking(self, operation: str, function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        if self._blocking_runner is None:
            return function(*args, **kwargs)
        return await self._blocking_runner(operation, function, *args, **kwargs)

    async def claim_lifecycle_parents(
        self,
        *,
        owner_id: str,
        lease_ttl: timedelta,
        limit: int,
        now: datetime,
        shard_ids: list[int] | None = None,
    ) -> list[BatchLifecycleClaim]:
        if not owner_id:
            raise DomainValidationError('batch_lifecycle_owner_required')
        if lease_ttl <= timedelta(0):
            raise DomainValidationError('batch_lifecycle_lease_ttl_must_be_positive')
        if limit < 1:
            raise DomainValidationError('batch_lifecycle_claim_limit_must_be_positive')
        filters = [
            or_(
                BatchRunRow.control_state.in_(['active', 'stopping']),
                and_(
                    BatchRunRow.control_state == 'finalized',
                    BatchRunRow.last_reconciled_at.is_(None),
                ),
            ),
            or_(
                BatchRunRow.lifecycle_lease_expires_at.is_(None),
                BatchRunRow.lifecycle_lease_expires_at <= now,
            ),
        ]
        if shard_ids is not None:
            if not shard_ids:
                return []
            filters.append(BatchRunRow.shard_id.in_(sorted(set(shard_ids))))
        rows = list(
            await self.session.scalars(
                select(BatchRunRow)
                .where(*filters)
                .order_by(BatchRunRow.last_reconciled_at, BatchRunRow.parent_execution_id)
                .limit(limit)
                .with_for_update(skip_locked=True)
                .execution_options(populate_existing=True)
            )
        )
        claims = []
        for row in rows:
            operation_id = new_id('blc')
            lease_expires_at = now + lease_ttl
            row.lifecycle_owner_id = owner_id
            row.lifecycle_operation_id = operation_id
            row.lifecycle_lease_expires_at = lease_expires_at
            claims.append(
                BatchLifecycleClaim(
                    parent_execution_id=row.parent_execution_id,
                    owner_id=owner_id,
                    operation_id=operation_id,
                    lease_expires_at=lease_expires_at,
                )
            )
        await self.session.flush()
        return claims

    async def renew_lifecycle_claim(
        self,
        claim: BatchLifecycleClaim,
        *,
        lease_ttl: timedelta,
        now: datetime,
    ) -> bool:
        if lease_ttl <= timedelta(0):
            raise DomainValidationError('batch_lifecycle_lease_ttl_must_be_positive')
        result = await self.session.execute(
            update(BatchRunRow)
            .where(
                BatchRunRow.parent_execution_id == claim.parent_execution_id,
                BatchRunRow.lifecycle_owner_id == claim.owner_id,
                BatchRunRow.lifecycle_operation_id == claim.operation_id,
                BatchRunRow.lifecycle_lease_expires_at > now,
            )
            .values(lifecycle_lease_expires_at=now + lease_ttl)
        )
        return int(getattr(result, 'rowcount', 0) or 0) == 1

    async def release_lifecycle_claim(
        self,
        claim: BatchLifecycleClaim,
        *,
        reconciled_at: datetime | None,
    ) -> bool:
        values: dict[str, Any] = {
            'lifecycle_owner_id': None,
            'lifecycle_operation_id': None,
            'lifecycle_lease_expires_at': None,
        }
        if reconciled_at is not None:
            values['last_reconciled_at'] = reconciled_at
        result = await self.session.execute(
            update(BatchRunRow)
            .where(
                BatchRunRow.parent_execution_id == claim.parent_execution_id,
                BatchRunRow.lifecycle_owner_id == claim.owner_id,
                BatchRunRow.lifecycle_operation_id == claim.operation_id,
            )
            .values(**values)
        )
        return int(getattr(result, 'rowcount', 0) or 0) == 1

    async def defer_lifecycle_claim(
        self,
        claim: BatchLifecycleClaim,
        *,
        defer_until: datetime,
    ) -> bool:
        """Release the claim but keep the lease timestamp as a backoff barrier.

        ``claim_lifecycle_parents`` skips rows whose ``lifecycle_lease_expires_at``
        is still in the future, so setting the timestamp to ``defer_until`` while
        clearing the owner/operation lets any reconciler pick the parent up again
        only after the backoff window. ``last_reconciled_at`` is intentionally
        left untouched so the ordering by staleness remains meaningful.
        """
        result = await self.session.execute(
            update(BatchRunRow)
            .where(
                BatchRunRow.parent_execution_id == claim.parent_execution_id,
                BatchRunRow.lifecycle_owner_id == claim.owner_id,
                BatchRunRow.lifecycle_operation_id == claim.operation_id,
            )
            .values(
                lifecycle_owner_id=None,
                lifecycle_operation_id=None,
                lifecycle_lease_expires_at=defer_until,
            )
        )
        return int(getattr(result, 'rowcount', 0) or 0) == 1

    async def fence_lifecycle_claim(self, claim: BatchLifecycleClaim, *, now: datetime) -> BatchRunRow:
        row = await self.session.scalar(
            select(BatchRunRow)
            .where(BatchRunRow.parent_execution_id == claim.parent_execution_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if (
            row is None
            or row.lifecycle_owner_id != claim.owner_id
            or row.lifecycle_operation_id != claim.operation_id
            or row.lifecycle_lease_expires_at is None
            or row.lifecycle_lease_expires_at <= now
        ):
            raise FencingError('batch_lifecycle_claim_lost')
        return row

    async def fence_parent_attempt(self, parent_execution_id: str, attempt_id: str | None) -> None:
        if attempt_id is None:
            raise DomainValidationError('batch_parent_attempt_required')
        batch = await self._batch_run_for_update(parent_execution_id)
        parent = await self.session.scalar(
            select(ExecutionRow).where(ExecutionRow.execution_id == parent_execution_id).with_for_update()
        )
        if parent is None:
            raise ResourceNotFoundError('batch_not_found')
        if parent.current_attempt_id != attempt_id:
            raise FencingError('batch_parent_attempt_fenced')
        if parent.current_assignment_id is None:
            raise FencingError('batch_parent_assignment_fenced')
        assignment = await self.session.scalar(
            select(AssignmentRow).where(AssignmentRow.assignment_id == parent.current_assignment_id).with_for_update()
        )
        if (
            assignment is None
            or assignment.execution_id != parent_execution_id
            or assignment.attempt_id != attempt_id
            or assignment.status != AssignmentStatus.ACTIVE
        ):
            raise FencingError('batch_parent_assignment_fenced')
        batch.active_parent_attempt_id = attempt_id
        await self.session.flush()

    async def set_generated_task_list(
        self,
        parent_execution_id: str,
        *,
        ref: ArtifactRef,
        manifest: dict[str, Any],
        generation_epoch: int,
        now: datetime,
    ) -> None:
        row = await self._batch_run_for_update(parent_execution_id)
        if row.control_state != 'active':
            raise FencingError('batch_control_not_active')
        if generation_epoch != row.generation_epoch:
            raise FencingError('batch_generation_epoch_fenced')
        if (
            manifest.get('version') != 1
            or manifest.get('sha256') != ref.sha256
            or manifest.get('size_bytes') != ref.size_bytes
            or not isinstance(manifest.get('item_count'), int)
            or manifest['item_count'] < 0
        ):
            raise DomainValidationError('invalid_batch_task_list_manifest')
        payload = ref.model_dump(mode='json')
        if row.task_list_ref_json is not None and row.task_list_ref_json != payload:
            raise DomainConflictError('batch_task_list_conflict')
        if row.task_list_manifest_json is not None and row.task_list_manifest_json != manifest:
            raise DomainConflictError('batch_task_list_manifest_conflict')
        row.task_list_ref_json = payload
        row.task_list_manifest_json = dict(manifest)
        row.task_list_sha256 = ref.sha256
        row.total_items = manifest['item_count']
        row.generation_status = 'materializing'
        row.last_progress_at = now
        row.updated_at = now
        parent = await self.session.get(ExecutionRow, parent_execution_id)
        if parent is None or not parent.owner_principal_id or not parent.environment_id:
            raise DomainConflictError('batch_parent_artifact_scope_missing')
        await ArtifactRepository(self.session).add_ref(
            ref.artifact_id,
            owner_principal_id=parent.owner_principal_id,
            environment_id=parent.environment_id,
            resource_type='batch',
            resource_id=parent_execution_id,
            now=now,
        )
        if parent.allocation_id:
            await ArtifactRepository(self.session).add_ref(
                ref.artifact_id,
                owner_principal_id=parent.owner_principal_id,
                environment_id=parent.environment_id,
                resource_type='allocation',
                resource_id=parent.allocation_id,
                now=now,
            )
        await self.session.flush()

    async def stage_generated_task_list(
        self,
        parent_execution_id: str,
        *,
        manifest: dict[str, Any],
        generation_epoch: int,
        now: datetime,
    ) -> None:
        row = await self._batch_run_for_update(parent_execution_id)
        if row.control_state != 'active':
            raise FencingError('batch_control_not_active')
        if generation_epoch != row.generation_epoch:
            raise FencingError('batch_generation_epoch_fenced')
        payload = TaskListManifest.model_validate(manifest).storage_payload()
        if row.task_list_manifest_json is not None and row.task_list_manifest_json != payload:
            raise DomainConflictError('batch_generation_nondeterministic')
        row.task_list_manifest_json = payload
        row.task_list_sha256 = payload['sha256']
        row.total_items = payload['item_count']
        if row.task_list_ref_json is None:
            row.generation_status = 'publishing'
        row.last_progress_at = now
        row.updated_at = now
        await self.session.flush()

    async def snapshot_task_list_materialization(
        self,
        parent_execution_id: str,
        *,
        expected_ref: ArtifactRef,
    ) -> MaterializeChunkSnapshot:
        batch = await self._batch_run_for_update(parent_execution_id)
        if batch.control_state != 'active':
            raise FencingError('batch_control_not_active')
        if batch.task_list_ref_json != expected_ref.model_dump(mode='json'):
            raise FencingError('batch_task_list_ref_mismatch')
        record_version = _manifest_record_version(batch.task_list_manifest_json)
        if record_version == 3 and self.task_store is None:
            raise DomainValidationError('batch_task_store_required_for_data_list')
        return MaterializeChunkSnapshot(
            generation_epoch=batch.generation_epoch,
            task_list_ref=expected_ref,
            record_version=record_version,
            cursor_bytes=int(batch.materialize_cursor_bytes or 0),
            materialized_count=int(batch.materialized_count or 0),
            total_items=batch.total_items,
            generation_complete=batch.generation_status == 'complete',
        )

    async def commit_task_list_materialization_chunk(
        self,
        parent_execution_id: str,
        *,
        snapshot: MaterializeChunkSnapshot,
        chunk: MaterializedTaskListChunk,
        now: datetime,
    ) -> MaterializeChunkResult:
        batch = await self._batch_run_for_update(parent_execution_id)
        if batch.control_state != 'active':
            raise FencingError('batch_control_not_active')
        if batch.task_list_ref_json != snapshot.task_list_ref.model_dump(mode='json'):
            raise FencingError('batch_task_list_ref_mismatch')
        if _manifest_record_version(batch.task_list_manifest_json) != snapshot.record_version:
            raise FencingError('batch_task_list_manifest_fenced')
        if batch.generation_epoch != snapshot.generation_epoch:
            raise FencingError('batch_generation_epoch_fenced')
        if batch.generation_status == 'complete':
            return MaterializeChunkResult(
                inserted=0,
                cursor_bytes=int(batch.materialize_cursor_bytes or 0),
                complete=True,
            )
        if (
            int(batch.materialize_cursor_bytes or 0) != snapshot.cursor_bytes
            or int(batch.materialized_count or 0) != snapshot.materialized_count
            or batch.total_items != snapshot.total_items
        ):
            raise MaterializationSnapshotStaleError(parent_execution_id)

        records = chunk.records
        existing_items: dict[str, BatchItemRow] = {}
        if records:
            existing_items = {
                item.item_id: item
                for item in (
                    await self.session.scalars(
                        select(BatchItemRow).where(
                            BatchItemRow.parent_execution_id == parent_execution_id,
                            BatchItemRow.item_id.in_(record[0].item_id for record in records),
                        )
                    )
                ).all()
            }
        for index, (record, offset, length, spec_sha256) in enumerate(records):
            existing = existing_items.get(record.item_id)
            if existing is None:
                self.session.add(
                    BatchItemRow(
                        parent_execution_id=parent_execution_id,
                        item_id=record.item_id,
                        task_ordinal=snapshot.materialized_count + index,
                        task_offset=offset,
                        task_length=length,
                        current_attempt_no=0,
                        spec_sha256=spec_sha256,
                        status='pending',
                        status_version=1,
                        metadata_json=record.metadata,
                        task_name=task_name_from_input(record.run_input),
                        created_at=now,
                        updated_at=now,
                    )
                )
            elif (
                existing.task_offset != offset
                or existing.task_length != length
                or existing.spec_sha256 != spec_sha256
                or existing.metadata_json != record.metadata
            ):
                raise DomainValidationError('duplicate_batch_item_id', detail=record.item_id)

        inserted = len(records)
        batch.materialize_cursor_bytes = chunk.cursor_bytes
        batch.materialized_cursor = str(chunk.cursor_bytes)
        batch.materialized_count = snapshot.materialized_count + inserted
        batch.updated_at = now
        if inserted:
            batch.last_progress_at = now
        if chunk.complete:
            expected_items = _manifest_item_count(batch.task_list_manifest_json)
            if expected_items is not None and batch.materialized_count != expected_items:
                raise DomainConflictError('batch_task_list_item_count_mismatch')
            batch.total_items = batch.materialized_count if batch.total_items is None else batch.total_items
            if batch.total_items != batch.materialized_count:
                raise DomainConflictError('batch_task_list_item_count_mismatch')
            if snapshot.record_version != 3:
                batch.generation_status = 'complete'
                batch.last_progress_at = now
        await self.session.flush()
        return MaterializeChunkResult(
            inserted=inserted,
            cursor_bytes=chunk.cursor_bytes,
            complete=chunk.complete,
        )

    async def snapshot_data_list_finalization(
        self,
        parent_execution_id: str,
        *,
        expected_ref: ArtifactRef,
    ) -> MaterializeChunkSnapshot:
        snapshot = await self.snapshot_task_list_materialization(parent_execution_id, expected_ref=expected_ref)
        if snapshot.record_version != 3:
            raise DomainValidationError('invalid_batch_task_list_manifest')
        if snapshot.cursor_bytes != expected_ref.size_bytes:
            raise DomainConflictError('batch_task_list_materialization_incomplete')
        return snapshot

    async def commit_data_list_finalization(
        self,
        parent_execution_id: str,
        *,
        snapshot: MaterializeChunkSnapshot,
        item_count: int,
        now: datetime,
    ) -> None:
        batch = await self._batch_run_for_update(parent_execution_id)
        if batch.control_state != 'active':
            raise FencingError('batch_control_not_active')
        if batch.task_list_ref_json != snapshot.task_list_ref.model_dump(mode='json'):
            raise FencingError('batch_task_list_ref_mismatch')
        if _manifest_record_version(batch.task_list_manifest_json) != 3:
            raise DomainValidationError('invalid_batch_task_list_manifest')
        if batch.generation_epoch != snapshot.generation_epoch:
            raise FencingError('batch_generation_epoch_fenced')
        if batch.generation_status == 'complete':
            return
        if (
            int(batch.materialize_cursor_bytes or 0) != snapshot.cursor_bytes
            or int(batch.materialized_count or 0) != snapshot.materialized_count
            or batch.total_items != snapshot.total_items
        ):
            raise MaterializationSnapshotStaleError(parent_execution_id)
        expected_items = _manifest_item_count(batch.task_list_manifest_json)
        if expected_items is not None and item_count != expected_items:
            raise DomainConflictError('batch_task_list_item_count_mismatch')
        if batch.materialized_count != item_count or batch.total_items != item_count:
            raise DomainConflictError('batch_task_list_item_count_mismatch')
        batch.generation_status = 'complete'
        batch.last_progress_at = now
        batch.updated_at = now
        await self.session.flush()

    async def begin_stop(self, parent_execution_id: str, *, reason: str, now: datetime) -> bool:
        if reason not in {'user_stop', 'parent_ttl', 'fail_fast', 'orchestration_error', 'parent_lost'}:
            raise DomainValidationError('invalid_batch_stop_reason')
        batch = await self._batch_run_for_update(parent_execution_id)
        if batch.control_state == 'finalized':
            return False
        if batch.control_state == 'stopping':
            if batch.stop_reason != reason:
                raise DomainConflictError('batch_stop_reason_conflict')
            return False
        if batch.control_state != 'active':
            raise FencingError('batch_control_state_invalid')
        batch.control_state = 'stopping'
        batch.stop_reason = reason
        batch.stop_requested_at = now
        batch.stop_cursor = None
        batch.submission_closed = True
        batch.last_progress_at = now
        batch.updated_at = now
        await self.session.flush()
        return True

    async def next_stop_page(
        self,
        parent_execution_id: str,
        *,
        page_size: int,
        now: datetime,
    ) -> StopPageResult:
        if page_size < 1 or page_size > 10000:
            raise DomainValidationError('batch_stop_page_size_out_of_range')
        batch = await self._batch_run_for_update(parent_execution_id)
        if batch.control_state != 'stopping' or batch.stop_reason is None:
            raise DomainConflictError('batch_not_stopping')
        query = select(BatchItemRow).where(BatchItemRow.parent_execution_id == parent_execution_id)
        if batch.stop_cursor is not None:
            query = query.where(BatchItemRow.item_id > batch.stop_cursor)
        selected = (
            await self.session.scalars(query.order_by(BatchItemRow.item_id).limit(page_size + 1).with_for_update())
        ).all()
        has_more = len(selected) > page_size
        rows = selected[:page_size]
        child_execution_ids: list[str] = []
        for item in rows:
            if item.status in _TERMINAL_ITEM_STATUSES:
                continue
            if item.current_child_execution_id is not None:
                child_execution_ids.append(item.current_child_execution_id)
                continue
            item.status = 'expired' if batch.stop_reason == 'parent_ttl' else 'cancelled'
            item.finished_at = now
            item.finished_at_estimated = False
            item.status_version += 1
            item.updated_at = now
            batch.terminal_items += 1
        if rows:
            batch.stop_cursor = rows[-1].item_id
            batch.last_progress_at = now
            batch.updated_at = now
        await self.session.flush()
        return StopPageResult(
            scanned=len(rows),
            child_execution_ids=tuple(child_execution_ids),
            cursor=batch.stop_cursor,
            complete=not has_more,
        )

    async def sync_child_states_page(
        self,
        parent_execution_id: str,
        *,
        after_item_id: str | None,
        page_size: int,
        retry_failed: int,
        now: datetime,
    ) -> SyncChildStatesPageResult:
        if page_size < 1 or page_size > 10000:
            raise DomainValidationError('batch_child_sync_page_size_out_of_range')
        if retry_failed < 0:
            raise DomainValidationError('batch_retry_failed_must_be_non_negative')
        _validate_batch_item_id(after_item_id)
        batch = await self._batch_run_for_update(parent_execution_id)
        item_query = select(BatchItemRow).where(
            BatchItemRow.parent_execution_id == parent_execution_id,
            BatchItemRow.status.in_(['submitted', 'running']),
        )
        if after_item_id is not None:
            item_query = item_query.where(BatchItemRow.item_id > after_item_id)
        selected_items = (
            await self.session.scalars(
                item_query.order_by(BatchItemRow.item_id).limit(page_size + 1).with_for_update(skip_locked=True)
            )
        ).all()
        has_more = len(selected_items) > page_size
        items = selected_items[:page_size]
        if not items:
            batch.child_sync_cursor = None
            return SyncChildStatesPageResult(scanned=0, changed=0, cursor=after_item_id, complete=True)

        child_ids = [item.current_child_execution_id for item in items if item.current_child_execution_id is not None]
        children = (
            await self.session.scalars(
                select(ExecutionRow)
                .options(
                    load_only(
                        ExecutionRow.execution_id,
                        ExecutionRow.result_artifact_id,
                        ExecutionRow.version,
                        ExecutionRow.status,
                        ExecutionRow.error_json,
                        ExecutionRow.started_at,
                        ExecutionRow.finished_at,
                    )
                )
                .where(ExecutionRow.execution_id.in_(child_ids))
            )
        ).all()
        children_by_id = {child.execution_id: child for child in children}
        rows = [
            (item, children_by_id[item.current_child_execution_id])
            for item in items
            if item.current_child_execution_id in children_by_id
        ]

        changed_rows = [
            (item, child)
            for item, child in rows
            if item.child_execution_version != child.version or item.child_execution_status != child.status
        ]
        if not changed_rows:
            batch.child_sync_cursor = None if not has_more else items[-1].item_id
            return SyncChildStatesPageResult(
                scanned=len(items),
                changed=0,
                cursor=items[-1].item_id,
                complete=not has_more,
            )

        attempts = (
            await self.session.scalars(
                select(BatchItemAttemptRow).where(
                    BatchItemAttemptRow.parent_execution_id == parent_execution_id,
                    BatchItemAttemptRow.item_id.in_([item.item_id for item, _child in changed_rows]),
                )
            )
        ).all()
        attempts_by_key = {(attempt.item_id, attempt.attempt_no): attempt for attempt in attempts}
        changed = 0
        terminal_discovery_lags: list[float] = []
        for item, child in changed_rows:
            old_item_status = item.status
            if child.started_at is not None:
                item.first_started_at = (
                    min(item.first_started_at, child.started_at) if item.first_started_at else child.started_at
                )
            item.child_execution_status = child.status
            item.child_execution_version = child.version
            next_item_status = (
                ('submitted' if child.status in {'queued', 'assigned'} else 'running')
                if child.status in _ACTIVE_CHILD_STATUSES
                else child.status
            )
            attempt = attempts_by_key.get((item.item_id, item.current_attempt_no))
            if attempt is not None:
                attempt.status = child.status
                attempt.result_artifact_id = child.result_artifact_id
                attempt.error_code = _execution_error_code(child)
                attempt.started_at = child.started_at
                attempt.finished_at = child.finished_at
                attempt.status_version += 1
                attempt.updated_at = now

            retryable = child.status in {'failed', 'lost', 'expired'} and item.current_attempt_no < retry_failed
            if retryable:
                item.current_attempt_no += 1
                item.current_child_execution_id = None
                item.child_execution_status = None
                item.child_execution_version = None
                next_item_status = 'pending'

            item.status = next_item_status
            item.finished_at = child.finished_at if next_item_status in _TERMINAL_ITEM_STATUSES else None
            item.finished_at_estimated = False
            item.status_version += 1
            item.updated_at = now
            if old_item_status in {'submitted', 'running'} and next_item_status not in {'submitted', 'running'}:
                batch.active_items = max(0, batch.active_items - 1)
                if next_item_status in _TERMINAL_ITEM_STATUSES:
                    batch.terminal_items += 1
                    if child.finished_at is not None:
                        terminal_discovery_lags.append(max(0.0, (now - child.finished_at).total_seconds()))
            changed += 1

        if changed:
            batch.last_progress_at = now
            batch.updated_at = now
        batch.child_sync_cursor = None if not has_more else items[-1].item_id
        await self.session.flush()
        return SyncChildStatesPageResult(
            scanned=len(items),
            changed=changed,
            cursor=items[-1].item_id,
            complete=not has_more,
            terminal_discovery_lags=tuple(terminal_discovery_lags),
        )

    async def has_exhausted_failure(self, parent_execution_id: str) -> bool:
        failed = await self.session.scalar(
            select(BatchItemRow.item_id)
            .where(
                BatchItemRow.parent_execution_id == parent_execution_id,
                BatchItemRow.status.in_(['failed', 'lost', 'expired']),
            )
            .limit(1)
        )
        return failed is not None

    async def bind_child_dependency_build(
        self,
        parent_execution_id: str,
        *,
        expected_dependency_build_id: str,
        dependency_build_id: str,
        module_dependencies: dict[str, Any],
        now: datetime,
    ) -> None:
        if not expected_dependency_build_id or not dependency_build_id:
            raise DomainValidationError('batch_dependency_build_id_required')
        if not isinstance(module_dependencies, dict) or not module_dependencies:
            raise DomainValidationError('batch_module_dependencies_required')

        batch = await self._batch_run_for_update(parent_execution_id)
        parent = await self.session.scalar(
            select(ExecutionRow).where(ExecutionRow.execution_id == parent_execution_id).with_for_update()
        )
        if parent is None:
            raise ResourceNotFoundError('batch_not_found')
        template = _child_template(parent, batch)
        allocation = await self.session.scalar(
            select(EnvironmentAllocationRow)
            .where(EnvironmentAllocationRow.allocation_id == template['allocation_id'])
            .with_for_update()
        )
        if allocation is None:
            raise TemporaryUnavailableError('batch_child_allocation_unavailable')
        build = await self.session.scalar(
            select(DependencyBuildRow)
            .where(DependencyBuildRow.dependency_build_id == dependency_build_id)
            .with_for_update()
        )
        if build is None:
            raise ResourceNotFoundError('dependency_build_not_found')
        if build.owner_principal_id != batch.owner_principal_id or build.environment_id != allocation.environment_id:
            raise FencingError('batch_dependency_build_identity_mismatch')

        if batch.child_dependency_build_id == dependency_build_id:
            persisted_dependencies = batch.module_dependencies_json
            if (
                not isinstance(persisted_dependencies, dict)
                or any(persisted_dependencies.get(key) != value for key, value in module_dependencies.items())
                or allocation.dependency_build_id != dependency_build_id
            ):
                raise FencingError('batch_dependency_build_retry_mismatch')
            _apply_batch_dependency_build_status(batch, build, now=now)
            batch.updated_at = now
            await self.session.flush()
            return
        if batch.dependency_status != 'pending' or batch.child_dependency_build_id != expected_dependency_build_id:
            raise FencingError('batch_dependency_build_already_bound')

        _validate_child_allocation(parent, template, allocation)
        if (
            allocation.status != EnvironmentAllocationStatus.REQUESTED
            or allocation.provider_resource_ref_json is not None
            or allocation.ready_manager_count != 0
        ):
            raise FencingError('batch_child_allocation_already_provisioned')
        manager_id = await self.session.scalar(
            select(InstanceManagerRow.manager_id)
            .where(InstanceManagerRow.allocation_id == allocation.allocation_id)
            .limit(1)
        )
        if manager_id is not None:
            raise FencingError('batch_child_allocation_already_provisioned')
        if allocation.dependency_build_id != expected_dependency_build_id:
            raise FencingError('batch_child_dependency_build_mismatch')
        prepared = await self._run_blocking(
            'bind-child-dependency-build',
            _prepare_child_dependency_binding,
            allocation_spec=allocation.allocation_spec_json or {},
            child_template=template,
            module_dependencies=module_dependencies,
            expected_dependency_build_id=expected_dependency_build_id,
            dependency_build_id=build.dependency_build_id,
            dependency_build_key=build.build_key,
        )
        allocation.dependency_build_id = build.dependency_build_id
        allocation.allocation_spec_json = prepared.allocation_spec
        allocation.allocation_spec_hash = prepared.allocation_spec_hash
        allocation.status_version += 1
        allocation.last_progress_at = now
        allocation.updated_at = now

        batch.module_dependencies_json = prepared.module_dependencies
        batch.child_dependency_build_id = build.dependency_build_id
        batch.child_template_json = prepared.child_template
        _apply_batch_dependency_build_status(batch, build, now=now)
        batch.last_progress_at = now
        batch.updated_at = now
        await self.session.flush()

    async def mark_child_dependency_skipped(
        self,
        parent_execution_id: str,
        *,
        expected_dependency_build_id: str,
        now: datetime,
    ) -> None:
        batch = await self._batch_run_for_update(parent_execution_id)
        if batch.dependency_status == 'skipped':
            return
        if batch.dependency_status != 'pending' or batch.child_dependency_build_id != expected_dependency_build_id:
            raise FencingError('batch_dependency_build_already_bound')
        parent = await self.session.scalar(
            select(ExecutionRow).where(ExecutionRow.execution_id == parent_execution_id).with_for_update()
        )
        if parent is None:
            raise ResourceNotFoundError('batch_not_found')
        template = _child_template(parent, batch)
        allocation = await self.session.scalar(
            select(EnvironmentAllocationRow)
            .where(EnvironmentAllocationRow.allocation_id == template['allocation_id'])
            .with_for_update()
        )
        if allocation is None:
            raise TemporaryUnavailableError('batch_child_allocation_unavailable')
        _validate_child_allocation(parent, template, allocation)
        if (
            allocation.status != EnvironmentAllocationStatus.REQUESTED
            or allocation.provider_resource_ref_json is not None
            or allocation.ready_manager_count != 0
            or allocation.dependency_build_id != expected_dependency_build_id
        ):
            raise FencingError('batch_child_allocation_already_provisioned')
        manager_id = await self.session.scalar(
            select(InstanceManagerRow.manager_id)
            .where(InstanceManagerRow.allocation_id == allocation.allocation_id)
            .limit(1)
        )
        if manager_id is not None:
            raise FencingError('batch_child_allocation_already_provisioned')
        batch.dependency_status = 'skipped'
        batch.module_dependencies_json = {
            'representative_item_id': None,
            'selector_digest': None,
            'selectors': [],
            'module_names': [],
            'host_requirements': [],
            'module_lock_path': None,
            'module_lock_sha256': None,
        }
        batch.last_progress_at = now
        batch.updated_at = now
        await self.session.flush()

    async def submit_window(
        self,
        parent_execution_id: str,
        *,
        task_list_path: Path,
        max_inflight: int,
        submit_batch_size: int,
        now: datetime,
        max_batch_bytes: int = 32 * 1024 * 1024,
    ) -> int:
        if max_inflight < 1:
            raise DomainValidationError('batch_max_inflight_must_be_positive')
        if submit_batch_size < 1 or submit_batch_size > self.config.batch_submit_batch_size:
            raise DomainValidationError('batch_submit_batch_size_out_of_range')
        if max_batch_bytes < 1:
            raise DomainValidationError('batch_submit_batch_bytes_out_of_range')
        batch = await self._batch_run_for_update(parent_execution_id)
        if batch.control_state != 'active' or batch.submission_closed or batch.generation_status != 'complete':
            return 0
        if not await self._child_dependencies_ready(batch, now=now):
            return 0
        limit = min(submit_batch_size, max(0, max_inflight - int(batch.active_items or 0)))
        if limit <= 0:
            return 0
        parent = await self.session.scalar(
            select(ExecutionRow).where(ExecutionRow.execution_id == parent_execution_id).with_for_update()
        )
        if parent is None:
            raise ResourceNotFoundError('batch_not_found')
        owner_principal_id = parent.owner_principal_id
        if not owner_principal_id:
            raise DomainConflictError('batch_parent_owner_required')
        child_template = _child_template(parent, batch)
        child_allocation = await self.session.get(EnvironmentAllocationRow, child_template['allocation_id'])
        if child_allocation is None:
            raise TemporaryUnavailableError('batch_child_allocation_unavailable')
        _validate_child_allocation(parent, child_template, child_allocation)
        if (
            child_allocation.status != EnvironmentAllocationStatus.READY
            or child_allocation.ready_manager_count < child_allocation.desired_manager_count
        ):
            return 0
        items = (
            await self.session.scalars(
                select(BatchItemRow)
                .where(
                    BatchItemRow.parent_execution_id == parent_execution_id,
                    BatchItemRow.status == 'pending',
                )
                .order_by(BatchItemRow.task_ordinal)
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
        ).all()
        if not items:
            return 0
        item_values = tuple(
            _SubmitWindowItem(
                item_id=item.item_id,
                task_offset=item.task_offset,
                task_length=item.task_length,
                current_attempt_no=item.current_attempt_no,
                spec_sha256=item.spec_sha256,
                metadata=item.metadata_json,
            )
            for item in items
        )
        pending_children = await self._run_blocking(
            'submit-window',
            _prepare_submit_window,
            task_list_path,
            items=item_values,
            max_batch_bytes=max_batch_bytes,
            child_template=child_template,
            parent_execution_id=parent_execution_id,
            parent_created_at=parent.created_at,
            owner_principal_id=owner_principal_id,
            owner_tenant=parent.owner_tenant,
            task_store_backend=self.task_store.backend_name if self.task_store is not None else None,
            now=now,
            max_metadata_bytes=self.config.batch_item_metadata_max_size_bytes,
        )

        children = await ExecutionRepository(self.session, shard_count=self.config.shard_count).create_executions_batch(
            [entry.request for entry in pending_children], now=now
        )
        items_by_id = {item.item_id: item for item in items}
        existing_attempts = (
            await self.session.scalars(
                select(BatchItemAttemptRow).where(
                    BatchItemAttemptRow.parent_execution_id == parent_execution_id,
                    BatchItemAttemptRow.item_id.in_([entry.item_id for entry in pending_children]),
                )
            )
        ).all()
        attempts_by_key = {(attempt.item_id, attempt.attempt_no): attempt for attempt in existing_attempts}
        created = 0
        for prepared, child in zip(pending_children, children, strict=True):
            item = items_by_id[prepared.item_id]
            attempt_no = prepared.attempt_no
            child_spec_sha256 = prepared.child_spec_sha256
            attempt = attempts_by_key.get((item.item_id, attempt_no))
            if attempt is None:
                self.session.add(
                    BatchItemAttemptRow(
                        parent_execution_id=parent_execution_id,
                        item_id=item.item_id,
                        attempt_no=attempt_no,
                        child_execution_id=child.execution_id,
                        spec_sha256=child.spec_sha256 or child_spec_sha256,
                        status=child.status,
                        status_version=1,
                        created_at=now,
                        updated_at=now,
                    )
                )
            elif attempt.child_execution_id != child.execution_id or attempt.spec_sha256 != child_spec_sha256:
                raise DomainConflictError('batch_child_attempt_idempotency_conflict')
            item.current_child_execution_id = child.execution_id
            item.child_execution_status = child.status
            item.child_execution_version = child.version
            item.status = 'submitted'
            item.status_version += 1
            item.updated_at = now
            created += 1
        batch.submitted_items += created
        batch.active_items += created
        if created:
            batch.last_progress_at = now
            batch.updated_at = now
        await self.session.flush()
        return created

    async def _child_dependencies_ready(self, batch: BatchRunRow, *, now: datetime) -> bool:
        if batch.dependency_status in {'not_required', 'ready'}:
            return True
        if batch.dependency_status in {'pending', 'skipped'}:
            return False
        if batch.dependency_status == 'failed':
            raise DomainValidationError(
                'batch_module_dependency_failed',
                detail=batch.child_dependency_build_id,
            )
        if batch.dependency_status != 'building' or not batch.child_dependency_build_id:
            raise DomainConflictError('batch_dependency_status_invalid')
        build = await self.session.get(DependencyBuildRow, batch.child_dependency_build_id)
        if build is None:
            raise DomainConflictError('batch_dependency_build_missing')
        _apply_batch_dependency_build_status(batch, build, now=now)
        if batch.dependency_status == 'failed':
            raise DomainValidationError(
                'batch_module_dependency_failed',
                detail=build.error_code or build.dependency_build_id,
            )
        return batch.dependency_status == 'ready'

    async def finalize_parent_if_ready(
        self,
        parent_execution_id: str,
        *,
        now: datetime,
    ) -> BatchFinalizationResult | None:
        batch = await self._batch_run_for_update(parent_execution_id)
        if batch.control_state == 'finalized':
            parent = await self.session.scalar(
                select(ExecutionRow).where(ExecutionRow.execution_id == parent_execution_id).with_for_update()
            )
            if parent is None:
                raise ResourceNotFoundError('batch_not_found')
            if not isinstance(batch.summary_json, dict):
                raise RuntimeError('finalized_batch_summary_missing')
            return BatchFinalizationResult(
                parent_status=parent.status,
                summary={key: int(value) for key, value in batch.summary_json.items()},
            )
        if batch.control_state not in {'active', 'stopping'}:
            return None

        summary = (
            {key: int(value) for key, value in batch.summary_json.items()}
            if isinstance(batch.summary_json, dict)
            else None
        )
        if summary is None:
            if batch.control_state == 'active':
                if batch.generation_status != 'complete' or batch.total_items != batch.materialized_count:
                    return None
            elif not await self._stop_fanout_complete(batch):
                return None

            materialized_count = int(batch.materialized_count or 0)
            if int(batch.terminal_items or 0) < materialized_count:
                return None

            rows = (
                await self.session.execute(
                    select(BatchItemRow.status, func.count(BatchItemRow.item_id))
                    .where(BatchItemRow.parent_execution_id == parent_execution_id)
                    .group_by(BatchItemRow.status)
                )
            ).all()
            counts = {str(status): int(count) for status, count in rows}
            exact_materialized_count = sum(counts.values())
            exact_terminal_count = sum(counts.get(status, 0) for status in _TERMINAL_ITEM_STATUSES)
            exact_active_count = counts.get('submitted', 0) + counts.get('running', 0)
            batch.active_items = exact_active_count
            batch.terminal_items = exact_terminal_count
            if exact_materialized_count != materialized_count:
                if batch.control_state == 'active' and batch.total_items != exact_materialized_count:
                    raise DomainConflictError('batch_materialized_counter_mismatch')
                batch.materialized_count = exact_materialized_count
                materialized_count = exact_materialized_count
            if exact_terminal_count != exact_materialized_count:
                batch.updated_at = now
                await self.session.flush()
                return None

            total = int(batch.total_items if batch.total_items is not None else materialized_count)
            unmaterialized = max(0, total - materialized_count)
            if batch.control_state == 'active' and unmaterialized:
                raise DomainConflictError('batch_materialized_counter_mismatch')
            if unmaterialized:
                unmaterialized_status = 'expired' if batch.stop_reason == 'parent_ttl' else 'cancelled'
                counts[unmaterialized_status] = counts.get(unmaterialized_status, 0) + unmaterialized
            summary = {
                'total': total,
                'succeeded': counts.get('succeeded', 0),
                'failed': counts.get('failed', 0),
                'killed': counts.get('killed', 0),
                'lost': counts.get('lost', 0),
                'expired': counts.get('expired', 0),
                'cancelled': counts.get('cancelled', 0),
            }
            batch.active_items = 0
            batch.terminal_items = materialized_count
            batch.submission_closed = True
            batch.summary_version += 1
            batch.summary_json = summary
            batch.updated_at = now
            await self.session.flush()

        parent = await self.session.scalar(
            select(ExecutionRow).where(ExecutionRow.execution_id == parent_execution_id).with_for_update()
        )
        if parent is None:
            raise ResourceNotFoundError('batch_not_found')
        parent_status = _batch_parent_terminal_status(batch, summary)
        if parent_status == ExecutionStatus.SUCCEEDED:
            if parent.result_artifact_id is None:
                if parent.current_assignment_id is None or parent.current_attempt_id is None:
                    raise DomainConflictError('batch_parent_result_scope_missing')
                return BatchFinalizationResult(
                    parent_status=parent_status,
                    summary=summary,
                    finalized=False,
                    result_artifact_required=True,
                    assignment_id=parent.current_assignment_id,
                    attempt_id=parent.current_attempt_id,
                )
            artifact = await self.session.get(ArtifactRow, parent.result_artifact_id)
            producer_prefix = f'run-result:{parent_execution_id}:'
            if (
                artifact is None
                or artifact.status != 'ready'
                or artifact.kind != 'run_result'
                or artifact.owner_principal_id != parent.owner_principal_id
                or artifact.environment_id != parent.environment_id
                or not isinstance(artifact.producer_key, str)
                or not artifact.producer_key.startswith(producer_prefix)
                or len(artifact.producer_key) == len(producer_prefix)
            ):
                raise DomainConflictError('batch_parent_result_artifact_invalid')
        error = (
            {
                'type': 'BatchFailed',
                'message': 'batch completed with non-success items',
                'summary': summary,
            }
            if parent_status == ExecutionStatus.FAILED
            else None
        )
        return await self._persist_parent_finalization(
            batch,
            parent,
            parent_status=parent_status,
            summary=summary,
            error=error,
            now=now,
        )

    async def fail_parent_result_persistence(
        self,
        parent_execution_id: str,
        *,
        summary: dict[str, int],
        assignment_id: str | None,
        attempt_id: str | None,
        cause: str,
        now: datetime,
    ) -> BatchFinalizationResult:
        candidate = await self.finalize_parent_if_ready(parent_execution_id, now=now)
        if (
            candidate is None
            or candidate.finalized
            or not candidate.result_artifact_required
            or candidate.summary != summary
            or candidate.assignment_id != assignment_id
            or candidate.attempt_id != attempt_id
        ):
            raise FencingError('batch_parent_result_failure_fenced')
        batch = await self._batch_run_for_update(parent_execution_id)
        parent = await self.session.scalar(
            select(ExecutionRow).where(ExecutionRow.execution_id == parent_execution_id).with_for_update()
        )
        if parent is None:
            raise ResourceNotFoundError('batch_parent_not_found')
        return await self._persist_parent_finalization(
            batch,
            parent,
            parent_status=ExecutionStatus.FAILED,
            summary=summary,
            error={
                'type': 'ResultPersistenceFailed',
                'message': 'batch run_result could not be persisted',
                'cause': cause,
            },
            now=now,
        )

    async def _persist_parent_finalization(
        self,
        batch: BatchRunRow,
        parent: ExecutionRow,
        *,
        parent_status: str,
        summary: dict[str, int],
        error: dict[str, Any] | None,
        now: datetime,
    ) -> BatchFinalizationResult:
        if parent.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.KILLED,
            ExecutionStatus.LOST,
            ExecutionStatus.EXPIRED,
        }:
            if parent.status != parent_status:
                raise DomainConflictError('batch_parent_terminal_status_conflict')
        else:
            old_parent_status = parent.status
            parent.status = parent_status
            parent.version += 1
            parent.result_json = {'batch': summary}
            parent.error_json = deepcopy(error)
            parent.finished_at = parent.finished_at or now
            parent.updated_at = now
            await append_control_event_row(
                self.session,
                parent,
                event_type='execution.status.changed',
                source='reconciler',
                payload={'from': old_parent_status, 'to': parent_status, 'reason': 'batch_finalized'},
                assignment_id=parent.current_assignment_id,
                attempt_id=parent.current_attempt_id,
                now=now,
            )

        assignment = (
            await self.session.get(AssignmentRow, parent.current_assignment_id)
            if parent.current_assignment_id is not None
            else None
        )
        if assignment is not None and assignment.status in {AssignmentStatus.PENDING, AssignmentStatus.ACTIVE}:
            old_assignment_status = assignment.status
            assignment.status = AssignmentStatus.RELEASED
            assignment.endpoint_ready = False
            assignment.finished_at = assignment.finished_at or now
            assignment.updated_at = now
            await append_control_event_row(
                self.session,
                parent,
                event_type='assignment.status.changed',
                source='reconciler',
                payload={'from': old_assignment_status, 'to': assignment.status},
                assignment_id=assignment.assignment_id,
                attempt_id=assignment.attempt_id,
                now=now,
            )

        batch.control_state = 'finalized'
        batch.submission_closed = True
        batch.active_items = 0
        batch.terminal_items = summary['total']
        if batch.summary_json != summary:
            batch.summary_version += 1
        batch.summary_json = summary
        batch.finalized_at = now
        batch.last_progress_at = now
        batch.updated_at = now
        await self.session.flush()
        return BatchFinalizationResult(parent_status=parent_status, summary=summary)

    async def batch_summary(
        self,
        parent_execution_id: str,
        *,
        include_status_counts: bool = False,
    ) -> dict[str, Any]:
        batch = await self._batch_run(parent_execution_id)
        summary = {
            'parent_execution_id': batch.parent_execution_id,
            'generation_mode': batch.generation_mode,
            'generation_status': batch.generation_status,
            'materialized_count': batch.materialized_count,
            'total_items': batch.total_items,
            'submitted_items': batch.submitted_items,
            'active_items': batch.active_items,
            'terminal_items': batch.terminal_items,
            'submission_closed': batch.submission_closed,
            'control_state': batch.control_state,
            'stop_reason': batch.stop_reason,
            'stop_requested_at': batch.stop_requested_at,
            'summary_version': batch.summary_version,
            'summary': batch.summary_json,
            'outcome': _batch_outcome(batch.summary_json),
            'task_list_ref': batch.task_list_ref_json,
            'task_list_manifest': batch.task_list_manifest_json,
            'dependency_status': getattr(batch, 'dependency_status', 'not_required'),
            'module_dependencies': getattr(batch, 'module_dependencies_json', None),
            'child_dependency_build_id': getattr(batch, 'child_dependency_build_id', None),
            'last_progress_at': batch.last_progress_at,
        }
        if include_status_counts:
            child_rows = (
                await self.session.execute(
                    select(
                        ExecutionRow.status,
                        func.count(ExecutionRow.execution_id),
                        func.sum(
                            case(
                                (
                                    and_(
                                        ExecutionRow.status == ExecutionStatus.QUEUED,
                                        ExecutionRow.last_schedule_error.startswith(
                                            'policy_deferred:', autoescape=True
                                        ),
                                    ),
                                    1,
                                ),
                                else_=0,
                            )
                        ),
                    )
                    .select_from(BatchItemRow)
                    .join(ExecutionRow, ExecutionRow.execution_id == BatchItemRow.current_child_execution_id)
                    .where(BatchItemRow.parent_execution_id == parent_execution_id)
                    .group_by(ExecutionRow.status)
                )
            ).all()
            summary['child_execution_status_counts'] = {str(status): int(count) for status, count, _ in child_rows}
            summary['child_waiting_policy_count'] = sum(int(deferred or 0) for _, _, deferred in child_rows)
            summary['item_status_counts'] = {
                str(status): int(count)
                for status, count in (
                    await self.session.execute(
                        select(BatchItemRow.status, func.count(BatchItemRow.item_id))
                        .where(BatchItemRow.parent_execution_id == parent_execution_id)
                        .group_by(BatchItemRow.status)
                    )
                ).all()
            }
        return summary

    async def batch_outcomes(self, parent_execution_ids: list[str]) -> dict[str, str | None]:
        if not parent_execution_ids:
            return {}
        rows = (
            await self.session.execute(
                select(BatchRunRow.parent_execution_id, BatchRunRow.summary_json).where(
                    BatchRunRow.parent_execution_id.in_(parent_execution_ids)
                )
            )
        ).all()
        return {str(parent_execution_id): _batch_outcome(summary) for parent_execution_id, summary in rows}

    async def list_items(
        self,
        parent_execution_id: str,
        *,
        status: str | None,
        after_item_id: str | None,
        limit: int,
        item_id_prefix: str | None = None,
        child_execution_id: str | None = None,
        error_code: str | None = None,
    ) -> list[dict[str, Any]]:
        _validate_batch_item_id(after_item_id)
        _validate_batch_item_filter(item_id_prefix, max_length=64)
        _validate_batch_item_filter(child_execution_id, max_length=64)
        _validate_batch_item_filter(error_code, max_length=128)
        query = select(BatchItemRow).where(BatchItemRow.parent_execution_id == parent_execution_id)
        if status:
            query = query.where(BatchItemRow.status == status)
        if item_id_prefix:
            query = query.where(BatchItemRow.item_id.startswith(item_id_prefix, autoescape=True))
        if child_execution_id:
            query = query.where(BatchItemRow.current_child_execution_id == child_execution_id)
        if error_code:
            matching_attempt = (
                select(BatchItemAttemptRow.item_id)
                .where(
                    BatchItemAttemptRow.parent_execution_id == parent_execution_id,
                    BatchItemAttemptRow.item_id == BatchItemRow.item_id,
                    BatchItemAttemptRow.error_code == error_code,
                )
                .exists()
            )
            query = query.where(matching_attempt)
        if after_item_id is not None:
            query = query.where(BatchItemRow.item_id > after_item_id)
        rows = (await self.session.scalars(query.order_by(BatchItemRow.item_id).limit(limit))).all()
        return [
            {
                'item_id': row.item_id,
                'status': row.status,
                'child_execution_status': row.child_execution_status,
                'current_attempt_no': row.current_attempt_no,
                'current_child_execution_id': row.current_child_execution_id,
                'metadata': row.metadata_json,
            }
            for row in rows
        ]

    async def get_item(self, parent_execution_id: str, item_id: str) -> dict[str, Any]:
        _validate_batch_item_id(item_id)
        row = await self.session.get(BatchItemRow, (parent_execution_id, item_id))
        if row is None:
            raise ResourceNotFoundError('batch_item_not_found')
        attempts = (
            await self.session.scalars(
                select(BatchItemAttemptRow)
                .where(
                    BatchItemAttemptRow.parent_execution_id == parent_execution_id,
                    BatchItemAttemptRow.item_id == item_id,
                )
                .order_by(BatchItemAttemptRow.attempt_no)
            )
        ).all()
        return {
            'item_id': row.item_id,
            'task_ordinal': row.task_ordinal,
            'status': row.status,
            'status_version': row.status_version,
            'child_execution_status': row.child_execution_status,
            'current_attempt_no': row.current_attempt_no,
            'current_child_execution_id': row.current_child_execution_id,
            'spec_sha256': row.spec_sha256,
            'metadata': row.metadata_json,
            'created_at': row.created_at,
            'updated_at': row.updated_at,
            'attempts': [_attempt_detail(attempt) for attempt in attempts],
        }

    async def get_attempt(self, parent_execution_id: str, item_id: str, attempt_no: int) -> dict[str, Any]:
        _validate_batch_item_id(item_id)
        row = await self.session.get(BatchItemAttemptRow, (parent_execution_id, item_id, attempt_no))
        if row is None:
            raise ResourceNotFoundError('batch_item_attempt_not_found')
        return _attempt_detail(row)

    async def _batch_run(self, parent_execution_id: str) -> BatchRunRow:
        row = await self.session.get(BatchRunRow, parent_execution_id)
        if row is None:
            raise ResourceNotFoundError('batch_not_found')
        return row

    async def _batch_run_for_update(self, parent_execution_id: str) -> BatchRunRow:
        row = await self.session.scalar(
            select(BatchRunRow).where(BatchRunRow.parent_execution_id == parent_execution_id).with_for_update()
        )
        if row is None:
            raise ResourceNotFoundError('batch_not_found')
        return row

    async def _stop_fanout_complete(self, batch: BatchRunRow) -> bool:
        if batch.materialized_count == 0:
            return True
        if batch.stop_cursor is None:
            return False
        remaining = await self.session.scalar(
            select(BatchItemRow.item_id)
            .where(
                BatchItemRow.parent_execution_id == batch.parent_execution_id,
                BatchItemRow.item_id > batch.stop_cursor,
            )
            .limit(1)
        )
        return remaining is None


def _child_template(parent: ExecutionRow, batch_run: BatchRunRow | None = None) -> dict[str, Any]:
    if not parent.owner_principal_id:
        raise DomainValidationError('batch_parent_owner_required')
    batch_value = parent.spec_json.get('batch')
    if not isinstance(batch_value, dict):
        raise DomainValidationError('batch_child_template_missing')
    batch: dict[str, Any] = batch_value
    persisted_template = batch_run.child_template_json if batch_run is not None else None
    template = persisted_template if isinstance(persisted_template, dict) else batch.get('child_template')
    if not isinstance(template, dict):
        raise DomainValidationError('batch_child_template_missing')
    required_strings = (
        'mode',
        'spec_type',
        'instance_mode',
        'workload',
        'environment_id',
        'pool_id',
        'allocation_id',
        'allocation_spec_hash',
    )
    if any(not isinstance(template.get(key), str) or not template[key] for key in required_strings):
        raise DomainValidationError('batch_child_template_invalid')
    if (
        template['mode'] != 'job'
        or template['spec_type'] != 'nexus_single'
        or template['instance_mode'] != 'asyncio'
        or template['workload'] != 'execution'
        or batch.get('child_allocation_id') != template['allocation_id']
        or (persisted_template is None and batch.get('child_allocation_spec_hash') != template['allocation_spec_hash'])
    ):
        raise DomainValidationError('batch_child_template_invalid')
    for key in ('slot_cost', 'max_attempts'):
        value = template.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise DomainValidationError('batch_child_template_invalid')
    for key in ('queue_ttl_seconds', 'execution_ttl_seconds', 'stop_timeout_seconds'):
        value = template.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise DomainValidationError('batch_child_template_invalid')
    if not isinstance(template.get('spec_template'), dict):
        raise DomainValidationError('batch_child_template_invalid')
    driver_config = template.get('driver_config', parent.driver_config_json or {})
    if not isinstance(driver_config, dict):
        raise DomainValidationError('batch_child_template_invalid')
    resolved_template = deepcopy(template)
    resolved_template['driver_config'] = deepcopy(driver_config)
    return resolved_template


def _prepare_child_dependency_binding(
    *,
    allocation_spec: dict[str, Any],
    child_template: dict[str, Any],
    module_dependencies: dict[str, Any],
    expected_dependency_build_id: str,
    dependency_build_id: str,
    dependency_build_key: str,
) -> _PreparedChildDependencyBinding:
    prepared_allocation_spec = deepcopy(allocation_spec)
    dependency_snapshot = prepared_allocation_spec.get('dependency_build')
    if (
        not isinstance(dependency_snapshot, dict)
        or dependency_snapshot.get('dependency_build_id') != expected_dependency_build_id
    ):
        raise FencingError('batch_child_dependency_build_mismatch')
    prepared_allocation_spec['dependency_build'] = {
        'dependency_build_id': dependency_build_id,
        'build_key': dependency_build_key,
    }
    allocation_spec_hash = canonical_sha256(prepared_allocation_spec)
    prepared_child_template = deepcopy(child_template)
    prepared_child_template['allocation_spec_hash'] = allocation_spec_hash
    return _PreparedChildDependencyBinding(
        allocation_spec=prepared_allocation_spec,
        allocation_spec_hash=allocation_spec_hash,
        child_template=prepared_child_template,
        module_dependencies=deepcopy(module_dependencies),
    )


def _batch_dependency_status(build: DependencyBuildRow) -> str:
    if build.status == DependencyBuildStatus.READY:
        return 'ready'
    if build.status == DependencyBuildStatus.FAILED and build.next_retry_at is not None:
        return 'building'
    if build.status in {DependencyBuildStatus.FAILED, DependencyBuildStatus.CANCELLED}:
        return 'failed'
    if build.status in {DependencyBuildStatus.REQUESTED, DependencyBuildStatus.BUILDING}:
        return 'building'
    raise DomainConflictError('batch_dependency_build_status_invalid')


def _apply_batch_dependency_build_status(
    batch: BatchRunRow,
    build: DependencyBuildRow,
    *,
    now: datetime,
) -> None:
    status = _batch_dependency_status(build)
    if status == 'ready':
        if not isinstance(build.manifest_json, dict) or not isinstance(batch.module_dependencies_json, dict):
            raise DomainConflictError('batch_dependency_build_metadata_missing')
        try:
            metadata = WheelhouseMetadata.model_validate(build.manifest_json)
        except ValueError as exc:
            raise DomainConflictError('batch_dependency_build_metadata_invalid') from exc
        if metadata.build_key != build.build_key:
            raise DomainConflictError('batch_dependency_build_metadata_mismatch')
        module_dependencies = deepcopy(batch.module_dependencies_json)
        module_dependencies['module_names'] = metadata.module_names
        module_dependencies['module_paths'] = metadata.module_paths
        module_dependencies['host_requirements'] = metadata.module_host_requirements
        module_dependencies['module_lock_path'] = metadata.module_lock_path
        module_dependencies['module_lock_sha256'] = metadata.module_lock_sha256
        batch.module_dependencies_json = module_dependencies
    batch.dependency_status = status
    batch.last_progress_at = now
    batch.updated_at = now


def _validate_child_allocation(
    parent: ExecutionRow,
    template: dict[str, Any],
    allocation: EnvironmentAllocationRow,
) -> None:
    expected = (
        parent.execution_id,
        parent.owner_principal_id,
        template['environment_id'],
        template['pool_id'],
        template['workload'],
        template['allocation_spec_hash'],
    )
    actual = (
        allocation.top_level_execution_id,
        allocation.owner_principal_id,
        allocation.environment_id,
        allocation.pool_id,
        allocation.workload,
        allocation.allocation_spec_hash,
    )
    if actual != expected:
        raise FencingError('batch_child_allocation_identity_mismatch')


def read_representative_run_input(
    task_list_path: Path,
    *,
    max_line_bytes: int,
    max_metadata_bytes: int = DEFAULT_SERVER_BATCH_ITEM_METADATA_BYTES,
) -> RepresentativeRunInput | None:
    if max_line_bytes < 1:
        raise DomainValidationError('batch_task_list_line_limit_invalid')
    with task_list_path.open('rb') as stream:
        line = stream.readline(max_line_bytes + 1)
    if not line:
        return None
    if len(line) > max_line_bytes:
        raise DomainValidationError('batch_task_list_line_too_large', detail='offset=0')
    record = _parse_task_list_record(line, offset=0, max_metadata_bytes=max_metadata_bytes)
    return RepresentativeRunInput(
        item_id=record.item_id,
        run_input=record.run_input,
    )


def read_materialize_task_list_chunk(
    task_list_path: Path,
    *,
    cursor: int,
    materialized_count: int,
    expected_size_bytes: int,
    expected_record_version: int,
    chunk_size: int,
    max_items: int,
    max_line_bytes: int,
    max_chunk_bytes: int,
    max_metadata_bytes: int = DEFAULT_SERVER_BATCH_ITEM_METADATA_BYTES,
) -> MaterializedTaskListChunk:
    if chunk_size < 1 or chunk_size > 1_000:
        raise DomainValidationError('batch_materialize_chunk_size_out_of_range')
    if max_chunk_bytes < 1:
        raise DomainValidationError('batch_materialize_chunk_bytes_out_of_range')
    if task_list_path.stat().st_size != expected_size_bytes:
        raise DomainConflictError('batch_task_list_identity_mismatch')
    complete = False
    seen_item_ids: set[str] = set()
    records: list[tuple[_TaskListRecord, int, int, str]] = []
    chunk_bytes = 0
    with task_list_path.open('rb') as stream:
        stream.seek(cursor)
        while len(records) < chunk_size:
            offset = stream.tell()
            line = stream.readline(max_line_bytes + 1)
            if not line:
                complete = True
                break
            if len(line) > max_line_bytes:
                raise DomainValidationError(
                    'batch_task_list_line_too_large',
                    detail=f'offset={offset}',
                )
            if chunk_bytes + len(line) > max_chunk_bytes:
                stream.seek(offset)
                if not records:
                    raise DomainValidationError(
                        'batch_materialize_record_exceeds_chunk_bytes',
                        detail=f'offset={offset}',
                    )
                break
            if materialized_count + len(records) >= max_items:
                raise DomainValidationError('batch_item_limit_exceeded')
            record = _parse_task_list_record(line, offset=offset, max_metadata_bytes=max_metadata_bytes)
            if record.version != expected_record_version:
                raise DomainValidationError(
                    'batch_task_list_record_version_mismatch',
                    detail=f'offset={offset}',
                )
            if record.item_id in seen_item_ids:
                raise DomainValidationError('duplicate_batch_item_id', detail=record.item_id)
            seen_item_ids.add(record.item_id)
            length = len(line)
            chunk_bytes += length
            records.append((record, offset, length, canonical_sha256(record.identity())))
        next_cursor = stream.tell()
    return MaterializedTaskListChunk(
        records=tuple(records),
        cursor_bytes=next_cursor,
        complete=complete,
    )


def cross_validate_task_data_list(
    task_list_path: Path,
    *,
    expected_task_list_ref: ArtifactRef,
    data_list_index: DataListIndex,
    max_items: int,
    max_line_bytes: int,
    max_metadata_bytes: int = DEFAULT_SERVER_BATCH_ITEM_METADATA_BYTES,
) -> int:
    if task_list_path.stat().st_size != expected_task_list_ref.size_bytes:
        raise DomainConflictError('batch_task_list_identity_mismatch')

    referenced_frames = 0
    item_count = 0
    task_list_digest = hashlib.sha256()
    with task_list_path.open('rb') as stream:
        while True:
            offset = stream.tell()
            line = stream.readline(max_line_bytes + 1)
            if not line:
                break
            if len(line) > max_line_bytes:
                raise DomainValidationError('batch_task_list_line_too_large', detail=f'offset={offset}')
            task_list_digest.update(line)
            if item_count >= max_items:
                raise DomainValidationError('batch_item_limit_exceeded')
            record = _parse_task_list_record(line, offset=offset, max_metadata_bytes=max_metadata_bytes)
            if record.version != 3:
                raise DomainValidationError('batch_task_list_record_version_mismatch', detail=f'offset={offset}')
            if record.data_ref is not None:
                if referenced_frames >= len(data_list_index.frames):
                    raise DomainValidationError(
                        'batch_task_list_data_ref_invalid',
                        detail=f'offset={offset}:orphan task-list reference',
                    )
                frame = data_list_index.frames[referenced_frames]
                if record.data_ref != frame.ref or record.item_id != frame.item_id:
                    raise DomainValidationError(
                        'batch_task_list_data_ref_invalid',
                        detail=f'offset={offset}:frame order or identity mismatch',
                    )
                referenced_frames += 1
            item_count += 1
    if task_list_digest.hexdigest() != expected_task_list_ref.sha256:
        raise DomainValidationError('batch_task_list_identity_mismatch')
    if referenced_frames != len(data_list_index.frames):
        raise DomainValidationError('batch_task_list_data_ref_invalid', detail='unreferenced data-list frame')
    return item_count


def encode_task_store_record(
    parent_execution_id: str,
    record_with_location: tuple[_TaskListRecord, int, int, str],
    *,
    data_list_path: Path | None,
    data_list_index: DataListIndex | None,
    max_data_frame_bytes: int,
    max_task_record_bytes: int,
) -> BatchTaskRecord:
    record, _offset, _length, _spec_sha256 = record_with_location
    try:
        if record.version == 3:
            if data_list_path is None:
                raise DomainValidationError('invalid_batch_data_list', detail='data-list path is required')
            if data_list_index is None:
                raise DomainValidationError('invalid_batch_data_list', detail='data-list index is required')
            frame = (
                read_indexed_frame(
                    data_list_path,
                    record.data_ref,
                    expected_frame=_expected_data_list_frame(data_list_index, record.data_ref),
                    expected_item_id=record.item_id,
                    max_frame_bytes=max_data_frame_bytes,
                )
                if record.data_ref is not None
                else None
            )
            return BatchTaskRecord.from_v3_identity(
                parent_execution_id=parent_execution_id,
                identity=record.identity(),
                frame=frame,
                max_record_bytes=max_task_record_bytes,
                max_frame_bytes=max_data_frame_bytes,
            )
        return BatchTaskRecord.from_identity(
            parent_execution_id=parent_execution_id,
            identity=record.identity(),
        )
    except DataListError as exc:
        raise DomainValidationError(exc.code, detail=exc.detail) from exc
    except ValueError as exc:
        code = str(exc)
        if code.startswith('batch_task_record_'):
            raise DomainValidationError(code) from exc
        raise


def _expected_data_list_frame(data_list_index: DataListIndex, ref: DataRef):
    frame = data_list_index.by_offset.get(ref.offset)
    if frame is None:
        raise DataListError(
            'batch_task_list_data_ref_invalid', detail='referenced frame is absent from data-list index'
        )
    return frame


def _prepare_submit_window(
    task_list_path: Path,
    *,
    items: tuple[_SubmitWindowItem, ...],
    max_batch_bytes: int,
    child_template: dict[str, Any],
    parent_execution_id: str,
    parent_created_at: datetime,
    owner_principal_id: str,
    owner_tenant: str | None,
    task_store_backend: str | None,
    now: datetime,
    max_metadata_bytes: int = DEFAULT_SERVER_BATCH_ITEM_METADATA_BYTES,
) -> tuple[_PreparedChildExecution, ...]:
    pending_children: list[_PreparedChildExecution] = []
    batch_bytes = 0
    stream = task_list_path.open('rb') if task_store_backend is None else None
    try:
        for item in items:
            if task_store_backend is None:
                assert stream is not None
                if batch_bytes + item.task_length > max_batch_bytes:
                    if not pending_children:
                        raise DomainValidationError(
                            'batch_submit_record_exceeds_batch_bytes',
                            detail=item.item_id,
                        )
                    break
                stream.seek(item.task_offset)
                line = stream.read(item.task_length)
                record = _parse_task_list_record(line, offset=item.task_offset, max_metadata_bytes=max_metadata_bytes)
                if (
                    record.item_id != item.item_id
                    or canonical_sha256(record.identity()) != item.spec_sha256
                    or record.metadata != item.metadata
                ):
                    raise FencingError('batch_task_list_item_identity_mismatch')
                batch_bytes += len(line)
            child_execution_id = new_id('exe')
            child_spec = deepcopy(child_template['spec_template'])
            if task_store_backend is None:
                record.run_input['run_id'] = child_execution_id
                child_spec['run_input'] = record.run_input
            else:
                child_spec['run_input_ref'] = {
                    'version': 1,
                    'store': task_store_backend,
                    'parent_execution_id': parent_execution_id,
                    'item_id': item.item_id,
                    'record_sha256': item.spec_sha256,
                }
            child_spec['batch'] = {
                'parent_execution_id': parent_execution_id,
                # Frozen into the spec so every attempt of every child reports the same
                # partition key; the Boundless trace table stores it on each row and
                # requires it in every WHERE clause for partition pruning.
                'parent_created_at': parent_created_at.isoformat(),
                'item_id': item.item_id,
                'attempt_no': item.current_attempt_no,
            }
            child_spec_sha256 = canonical_sha256(child_spec)
            request = CreateExecution(
                execution_id=child_execution_id,
                mode=child_template['mode'],
                spec_type=child_template['spec_type'],
                instance_mode=child_template['instance_mode'],
                spec=child_spec,
                driver_config=deepcopy(child_template['driver_config']),
                placement={
                    'environment': child_template['environment_id'],
                    'pool': child_template['pool_id'],
                    'allocation_id': child_template['allocation_id'],
                    'labels': {'workload': child_template['workload']},
                    **(
                        {'scheduler_policy': child_template['scheduler_policy']}
                        if child_template.get('scheduler_policy')
                        else {}
                    ),
                    **(
                        {'request_policy': child_template['request_policy']}
                        if child_template.get('request_policy')
                        else {}
                    ),
                },
                route_status=RouteStatus.NONE,
                slot_cost=child_template['slot_cost'],
                max_attempts=child_template['max_attempts'],
                queue_ttl_seconds=child_template['queue_ttl_seconds'],
                execution_ttl_seconds=child_template['execution_ttl_seconds'],
                stop_timeout_seconds=child_template['stop_timeout_seconds'],
                idempotency_scope=f'batch:{parent_execution_id}',
                idempotency_key=(f'batch:{parent_execution_id}:item:{item.item_id}:attempt:{item.current_attempt_no}'),
                owner_principal_id=owner_principal_id,
                owner_tenant=owner_tenant,
                created_by='system:batch-parent',
                parent_execution_id=parent_execution_id,
                environment_id=child_template['environment_id'],
                pool_id=child_template['pool_id'],
                allocation_id=child_template['allocation_id'],
                workload=child_template['workload'],
                spec_sha256=child_spec_sha256,
                allocation_spec_hash=child_template['allocation_spec_hash'],
                first_schedulable_at=now,
            )
            pending_children.append(
                _PreparedChildExecution(
                    item_id=item.item_id,
                    attempt_no=item.current_attempt_no,
                    child_spec_sha256=child_spec_sha256,
                    request=request,
                )
            )
    finally:
        if stream is not None:
            stream.close()
    return tuple(pending_children)


def _parse_task_list_record(
    line: bytes,
    *,
    offset: int,
    max_metadata_bytes: int = DEFAULT_SERVER_BATCH_ITEM_METADATA_BYTES,
) -> _TaskListRecord:
    try:
        payload = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DomainValidationError('invalid_task_list_json', detail=f'offset={offset}') from exc
    if not isinstance(payload, dict):
        raise DomainValidationError('invalid_task_list_item', detail=f'offset={offset}')
    version = payload.get('version')
    if isinstance(version, bool):
        raise DomainValidationError('unsupported_task_list_version', detail=f'offset={offset}')
    if version == 1:
        expected_fields = {'version', 'item_id', 'run_input'}
        metadata: dict[str, Any] = {}
        data_ref = None
    elif version in {2, 3}:
        expected_fields = (
            {'version', 'item_id', 'run_input', 'metadata', 'data_ref'}
            if version == 3
            else {'version', 'item_id', 'run_input', 'metadata'}
        )
        raw_metadata = payload.get('metadata')
        try:
            canonical_batch_item_metadata(raw_metadata, max_bytes=max_metadata_bytes)
        except ValueError as exc:
            raise DomainValidationError('invalid_task_list_metadata', detail=f'offset={offset}') from exc
        metadata = deepcopy(raw_metadata)
        if version == 3:
            try:
                data_ref = DataRef.model_validate(payload['data_ref']) if payload.get('data_ref') is not None else None
            except ValueError as exc:
                raise DomainValidationError(
                    'batch_task_list_data_ref_invalid',
                    detail=f'offset={offset}',
                ) from exc
        else:
            data_ref = None
    else:
        raise DomainValidationError('unsupported_task_list_version', detail=f'offset={offset}')
    if set(payload) != expected_fields:
        raise DomainValidationError('invalid_task_list_item', detail=f'offset={offset}')
    item_id = payload.get('item_id')
    run_input = payload.get('run_input')
    if not is_valid_batch_item_id(item_id) or not isinstance(run_input, dict):
        raise DomainValidationError('invalid_task_list_item', detail=f'offset={offset}')
    forbidden = _FORBIDDEN_RUN_INPUT_KEYS.intersection(run_input)
    if forbidden:
        raise DomainValidationError(
            'forbidden_task_list_run_input_field',
            detail=sorted(forbidden)[0],
        )
    try:
        normalized_run_input = normalize_run_input(run_input)
    except InputNormalizationError as exc:
        raise DomainValidationError('invalid_run_input', detail=f'offset={offset}') from exc
    if version == 3:
        try:
            validate_batch_task_files_classification(
                run_input=normalized_run_input,
                has_data_ref=data_ref is not None,
            )
        except ValueError as exc:
            raise DomainValidationError('batch_task_list_data_ref_invalid', detail=f'offset={offset}') from exc
    return _TaskListRecord(
        version=version,
        item_id=item_id,
        run_input=normalized_run_input,
        metadata=metadata,
        data_ref=data_ref,
    )


def _validate_batch_item_id(item_id: str | None) -> None:
    if item_id is not None and not is_valid_batch_item_id(item_id):
        raise DomainValidationError('invalid_batch_item_id')


def _validate_batch_item_filter(value: str | None, *, max_length: int) -> None:
    if value is not None and (not value or len(value) > max_length or not value.isascii()):
        raise DomainValidationError('invalid_batch_item_filter')


def _manifest_item_count(manifest: dict[str, Any] | None) -> int | None:
    if not isinstance(manifest, dict):
        return None
    value = manifest.get('item_count')
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DomainValidationError('invalid_batch_task_list_manifest')
    return value


def _manifest_record_version(manifest: dict[str, Any] | None) -> int:
    if not isinstance(manifest, dict):
        raise DomainValidationError('invalid_batch_task_list_manifest')
    value = manifest.get('record_version', 1)
    if isinstance(value, bool) or value not in {1, 2, 3}:
        raise DomainValidationError('invalid_batch_task_list_manifest')
    return value


def _attempt_detail(row: BatchItemAttemptRow) -> dict[str, Any]:
    return {
        'attempt_no': row.attempt_no,
        'child_execution_id': row.child_execution_id,
        'spec_sha256': row.spec_sha256,
        'status': row.status,
        'status_version': row.status_version,
        'result_artifact_id': row.result_artifact_id,
        'error_code': row.error_code,
        'started_at': row.started_at,
        'finished_at': row.finished_at,
        'created_at': row.created_at,
        'updated_at': row.updated_at,
    }


def _execution_error_code(execution: ExecutionRow) -> str | None:
    error = execution.error_json
    if not isinstance(error, dict):
        return None
    value = error.get('type') or error.get('code')
    return str(value)[:128] if value else None


def _batch_parent_terminal_status(batch: BatchRunRow, summary: dict[str, int]) -> str:
    if batch.control_state == 'stopping':
        return {
            'user_stop': ExecutionStatus.KILLED,
            'parent_ttl': ExecutionStatus.EXPIRED,
            'fail_fast': ExecutionStatus.FAILED,
            'orchestration_error': ExecutionStatus.FAILED,
            'parent_lost': ExecutionStatus.LOST,
        }[batch.stop_reason]
    non_success = sum(summary[key] for key in ('failed', 'killed', 'lost', 'expired', 'cancelled'))
    return ExecutionStatus.FAILED if non_success else ExecutionStatus.SUCCEEDED


def _batch_outcome(summary: dict[str, Any] | None) -> str | None:
    if summary is None:
        return None
    non_success = sum(int(summary.get(status, 0)) for status in ('failed', 'killed', 'lost', 'expired', 'cancelled'))
    if non_success == 0:
        return 'succeeded'
    return 'partial_success' if int(summary.get('succeeded', 0)) > 0 else 'failed'
