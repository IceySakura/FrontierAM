from __future__ import annotations

import asyncio
import logging
import time

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import async_sessionmaker

from nexus.distributed.artifacts.gateway import (
    ApplicationGatewayArtifactTransport,
    RunResultArtifactGateway,
    create_application_gateway_artifact_transport,
)
from nexus.distributed.artifacts.run_result import (
    RunResultPublisher,
    TransientResultPublishError,
    encode_run_result_content,
    run_result_producer_key,
)
from nexus.distributed.batch.task_store import BatchTaskStore
from nexus.distributed.config.views import BatchLifecycleConfigView
from nexus.distributed.execution.models import ExecutionStatus
from nexus.distributed.observability import (
    ControlLoopWork,
    ControlPlaneMetrics,
    OpenTelemetryControlPlaneMetrics,
)
from nexus.distributed.persistence.db import is_database_transaction_conflict
from nexus.distributed.persistence.orm import BatchRunRow, ExecutionRow
from nexus.distributed.ports import ReconcileController
from nexus.distributed.repositories.batch import (
    BatchFinalizationResult,
    BatchLifecycleClaim,
    BatchRepository,
)
from nexus.distributed.repositories.execution import ExecutionRepository
from nexus.distributed.utils.clock import utcnow
from nexus.distributed.utils.leases import LeaseLostError, PeriodicLease


logger = logging.getLogger(__name__)

_BATCH_TRANSACTION_MAX_ATTEMPTS = 3


class _BatchTaskStoreCleanupError(RuntimeError):
    pass


class BatchLifecycle(ReconcileController, Protocol):
    pass


@dataclass(frozen=True, slots=True)
class _BatchLifecyclePageResult:
    changed: int
    scanned: int
    pages: int
    more_work: bool
    terminal_discovery_lags: tuple[float, ...] = ()


class BatchLifecycleReconciler(BatchLifecycle):
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker,
        config: BatchLifecycleConfigView,
        owner_id: str = 'reconciler',
        artifact_client: RunResultPublisher | None = None,
        artifact_transport: ApplicationGatewayArtifactTransport | None = None,
        task_store: BatchTaskStore | None = None,
        metrics: ControlPlaneMetrics | None = None,
    ) -> None:
        if artifact_client is not None and artifact_transport is not None:
            raise ValueError('artifact_client and artifact_transport are mutually exclusive')
        self.session_factory = session_factory
        self.config = config
        self.owner_id = owner_id
        self.metrics = metrics if metrics is not None else OpenTelemetryControlPlaneMetrics()
        self._artifact_transport = artifact_transport
        self._owns_artifact_transport = False
        self._task_store = task_store
        self._last_stats = ControlLoopWork()
        self._full_scan_started_at: dict[str, float] = {}
        self._full_scan_scanned: dict[str, int] = {}
        self._artifact_client = (
            artifact_client
            if artifact_client is not None
            else RunResultArtifactGateway(artifact_transport)
            if artifact_transport is not None
            else None
        )
        self._shard_ids: tuple[int, ...] | None = None
        # Per-parent consecutive transient publish failures. Used to compute an
        # exponential backoff when the gateway does not supply Retry-After. In
        # memory only: a reconciler restart drops the counter, which is
        # harmless because the DB-side defer keeps the backoff window intact.
        self._transient_publish_attempts: dict[str, int] = {}

    async def close(self) -> None:
        if self._task_store is not None:
            await self._task_store.close()
            self._task_store = None
        if self._owns_artifact_transport and self._artifact_transport is not None:
            await self._artifact_transport.close()
            self._owns_artifact_transport = False

    @property
    def last_stats(self) -> ControlLoopWork:
        return self._last_stats

    def set_shard_ids(self, shard_ids: list[int]) -> None:
        self._shard_ids = tuple(sorted(set(shard_ids)))

    async def reconcile_once(self) -> int:
        async with self.session_factory() as session:
            claims = await BatchRepository(session, config=self.config).claim_lifecycle_parents(
                owner_id=self.owner_id,
                lease_ttl=self.config.batch_lifecycle_claim_ttl,
                limit=self.config.batch_lifecycle_parent_limit,
                now=utcnow(),
                shard_ids=list(self._shard_ids) if self._shard_ids is not None else None,
            )
            await session.commit()
        changed = 0
        scanned = 0
        remaining_scan = self.config.batch_lifecycle_scan_limit
        for claim in claims:
            reconciled = False
            deferred_until: datetime | None = None
            try:
                if remaining_scan <= 0:
                    reconciled = True
                    continue
                scan_started = self._full_scan_started_at.setdefault(
                    claim.parent_execution_id,
                    time.monotonic(),
                )
                async with self._lifecycle_lease(claim) as lease:
                    result = await self._reconcile_parent_with_retry(
                        claim,
                        lease=lease,
                        page_budget=self.config.batch_lifecycle_max_pages_per_parent,
                        scan_budget=remaining_scan,
                    )
                reconciled = True
                changed += result.changed
                scanned += result.scanned
                remaining_scan -= result.scanned
                for lag_seconds in result.terminal_discovery_lags:
                    self.metrics.record_batch_terminal_discovery_lag(lag_seconds=lag_seconds)
                self._full_scan_scanned[claim.parent_execution_id] = (
                    self._full_scan_scanned.get(claim.parent_execution_id, 0) + result.scanned
                )
                if not result.more_work:
                    self.metrics.record_batch_full_scan(
                        duration_seconds=max(0.0, time.monotonic() - scan_started),
                        scanned=self._full_scan_scanned.pop(claim.parent_execution_id, 0),
                    )
                    self._full_scan_started_at.pop(claim.parent_execution_id, None)
                # A tick that never touched the run_result publisher (or one that
                # touched it and succeeded) resets the transient backoff counter.
                self._transient_publish_attempts.pop(claim.parent_execution_id, None)
            except LeaseLostError:
                logger.info(
                    'Batch lifecycle claim lost: owner=%s parent_execution_id=%s operation_id=%s',
                    self.owner_id,
                    claim.parent_execution_id,
                    claim.operation_id,
                )
            except _BatchTaskStoreCleanupError:
                logger.exception(
                    'Batch Task Store cleanup failed: owner=%s parent_execution_id=%s operation_id=%s',
                    self.owner_id,
                    claim.parent_execution_id,
                    claim.operation_id,
                )
            except TransientResultPublishError as exc:
                delay = self._transient_publish_delay(claim.parent_execution_id, exc)
                deferred_until = utcnow() + timedelta(seconds=delay)
                logger.warning(
                    'Batch lifecycle publish deferred by transient gateway error: '
                    'owner=%s parent_execution_id=%s operation_id=%s '
                    'delay_seconds=%.3f cause=%s detail=%s',
                    self.owner_id,
                    claim.parent_execution_id,
                    claim.operation_id,
                    delay,
                    type(exc).__name__,
                    str(exc),
                )
            except Exception:
                reconciled = True
                logger.exception(
                    'Batch lifecycle reconcile failed: owner=%s parent_execution_id=%s operation_id=%s',
                    self.owner_id,
                    claim.parent_execution_id,
                    claim.operation_id,
                )
            finally:
                if deferred_until is not None:
                    await self._defer_lifecycle_claim(claim, defer_until=deferred_until)
                else:
                    await self._release_lifecycle_claim(claim, reconciled=reconciled)
        self._prune_full_scan_starts()
        self._last_stats = ControlLoopWork(
            scanned=scanned,
            claimed=len(claims),
            changed=changed,
        )
        return changed

    def _prune_full_scan_starts(self) -> None:
        oldest_allowed = time.monotonic() - max(
            3600.0,
            self.config.batch_lifecycle_claim_ttl.total_seconds() * 2,
        )
        stale = [
            parent_execution_id
            for parent_execution_id, started_at in self._full_scan_started_at.items()
            if started_at < oldest_allowed
        ]
        for parent_execution_id in stale:
            self._full_scan_started_at.pop(parent_execution_id, None)
            self._full_scan_scanned.pop(parent_execution_id, None)
            self._transient_publish_attempts.pop(parent_execution_id, None)
        max_entries = max(1000, self.config.batch_lifecycle_parent_limit * 100)
        while len(self._full_scan_started_at) > max_entries:
            parent_execution_id = next(iter(self._full_scan_started_at))
            self._full_scan_started_at.pop(parent_execution_id, None)
            self._full_scan_scanned.pop(parent_execution_id, None)
            self._transient_publish_attempts.pop(parent_execution_id, None)
        # Independently trim the transient backoff dict in case parents get
        # tracked here but never showed up in the full-scan bookkeeping.
        while len(self._transient_publish_attempts) > max_entries:
            self._transient_publish_attempts.pop(next(iter(self._transient_publish_attempts)), None)

    async def _reconcile_parent_with_retry(
        self,
        claim: BatchLifecycleClaim,
        *,
        lease: PeriodicLease,
        page_budget: int,
        scan_budget: int,
    ) -> _BatchLifecyclePageResult:
        for attempt in range(1, _BATCH_TRANSACTION_MAX_ATTEMPTS + 1):
            try:
                return await self._reconcile_parent(
                    claim,
                    lease=lease,
                    page_budget=page_budget,
                    scan_budget=scan_budget,
                )
            except DBAPIError as exc:
                if not is_database_transaction_conflict(exc) or attempt >= _BATCH_TRANSACTION_MAX_ATTEMPTS:
                    raise
                delay = min(0.05 * (2 ** (attempt - 1)), max(0.0, self.config.reconciler_poll_interval))
                logger.warning(
                    'Retrying batch lifecycle transaction after database conflict: '
                    'owner=%s parent_execution_id=%s operation_id=%s attempt=%s/%s retry_reason=db_lock',
                    self.owner_id,
                    claim.parent_execution_id,
                    claim.operation_id,
                    attempt + 1,
                    _BATCH_TRANSACTION_MAX_ATTEMPTS,
                )
                if delay > 0:
                    await asyncio.sleep(delay)
        raise AssertionError('unreachable')

    def _lifecycle_lease(self, claim: BatchLifecycleClaim) -> PeriodicLease:
        return PeriodicLease(
            renew=lambda: self._renew_lifecycle_claim(claim),
            interval=self.config.batch_lifecycle_claim_renew_interval,
            lost_code='batch_lifecycle_claim_lost',
        )

    async def _renew_lifecycle_claim(self, claim: BatchLifecycleClaim) -> bool:
        async with self.session_factory() as session:
            renewed = await BatchRepository(session, config=self.config).renew_lifecycle_claim(
                claim,
                lease_ttl=self.config.batch_lifecycle_claim_ttl,
                now=utcnow(),
            )
            await session.commit()
        return renewed

    async def _release_lifecycle_claim(self, claim: BatchLifecycleClaim, *, reconciled: bool) -> None:
        try:
            async with self.session_factory() as session:
                await BatchRepository(session, config=self.config).release_lifecycle_claim(
                    claim,
                    reconciled_at=utcnow() if reconciled else None,
                )
                await session.commit()
        except Exception:
            logger.warning(
                'Batch lifecycle claim release failed: owner=%s parent_execution_id=%s operation_id=%s',
                self.owner_id,
                claim.parent_execution_id,
                claim.operation_id,
                exc_info=True,
            )

    async def _defer_lifecycle_claim(
        self,
        claim: BatchLifecycleClaim,
        *,
        defer_until: datetime,
    ) -> None:
        try:
            async with self.session_factory() as session:
                await BatchRepository(session, config=self.config).defer_lifecycle_claim(
                    claim,
                    defer_until=defer_until,
                )
                await session.commit()
        except Exception:
            logger.warning(
                'Batch lifecycle claim defer failed: owner=%s parent_execution_id=%s operation_id=%s',
                self.owner_id,
                claim.parent_execution_id,
                claim.operation_id,
                exc_info=True,
            )

    def _transient_publish_delay(
        self,
        parent_execution_id: str,
        exc: TransientResultPublishError,
    ) -> float:
        """Pick the backoff duration for a transient publish failure.

        Prefer the gateway's Retry-After hint when provided; otherwise apply an
        exponential backoff seeded by ``batch_lifecycle_publish_backoff_base`` and
        capped at ``batch_lifecycle_publish_backoff_max``.
        """
        base = float(self.config.batch_lifecycle_publish_backoff_base)
        cap = float(self.config.batch_lifecycle_publish_backoff_max)
        cap = max(cap, base)
        attempts = self._transient_publish_attempts.get(parent_execution_id, 0) + 1
        self._transient_publish_attempts[parent_execution_id] = attempts
        retry_after = getattr(exc, 'retry_after', None)
        # 2**(attempts-1) saturates at ``cap`` quickly and the counter itself is
        # bounded by _prune_full_scan_starts + resets on successful publishes.
        delay = (
            float(retry_after) if retry_after is not None and retry_after > 0 else base * (2 ** max(0, attempts - 1))
        )
        return max(base, min(cap, delay))

    async def _reconcile_parent(
        self,
        claim: BatchLifecycleClaim,
        *,
        lease: PeriodicLease,
        page_budget: int,
        scan_budget: int,
    ) -> _BatchLifecyclePageResult:
        await lease.ensure_owned()
        parent_execution_id = claim.parent_execution_id
        now = utcnow()
        page_size = self.config.batch_lifecycle_page_size
        publication: BatchFinalizationResult | None = None
        changed = 0
        scanned = 0
        pages = 0
        more_work = False
        terminal_discovery_lags: list[float] = []
        cleanup_pending = False
        async with self.session_factory() as session:
            repo = BatchRepository(session, config=self.config)
            batch = await repo.fence_lifecycle_claim(claim, now=now)
            parent = await session.get(ExecutionRow, parent_execution_id)
            if parent is None:
                return _BatchLifecyclePageResult(changed=0, scanned=0, pages=0, more_work=False)
            if batch.control_state == 'finalized':
                cleanup_pending = _batch_uses_task_store(batch, task_store_available=self._task_store is not None)
                await session.commit()
                if cleanup_pending:
                    changed += await self._cleanup_task_store_records(claim, lease=lease)
                return _BatchLifecyclePageResult(changed=changed, scanned=0, pages=0, more_work=False)
            if batch.control_state == 'active':
                terminal_stop_reason = {
                    ExecutionStatus.FAILED: 'orchestration_error',
                    ExecutionStatus.LOST: 'parent_lost',
                    ExecutionStatus.KILLED: 'user_stop',
                    ExecutionStatus.EXPIRED: 'parent_ttl',
                }.get(parent.status)
                if parent.status == ExecutionStatus.STOPPING:
                    terminal_stop_reason = (
                        'parent_ttl' if parent.stop_reason in {'execution_ttl', 'parent_ttl'} else 'user_stop'
                    )
                if terminal_stop_reason is not None:
                    changed += int(await repo.begin_stop(parent_execution_id, reason=terminal_stop_reason, now=now))
                elif _initial_submit_window_pending(parent, batch):
                    await lease.ensure_owned()
                    await repo.fence_lifecycle_claim(claim, now=utcnow())
                    await session.commit()
                    return _BatchLifecyclePageResult(changed=changed, scanned=0, pages=0, more_work=False)

            retry_failed = (
                0
                if batch.control_state == 'stopping'
                else int((parent.spec_json.get('batch') or {}).get('retry_failed') or 0)
            )
            sync_complete = False
            while pages < page_budget and scanned < scan_budget:
                current_batch = await session.get(BatchRunRow, parent_execution_id)
                if current_batch is None:
                    break
                current_page_size = min(page_size, scan_budget - scanned)
                sync_page = await repo.sync_child_states_page(
                    parent_execution_id,
                    after_item_id=current_batch.child_sync_cursor,
                    page_size=current_page_size,
                    retry_failed=retry_failed,
                    now=now,
                )
                pages += 1
                scanned += sync_page.scanned
                changed += sync_page.changed
                terminal_discovery_lags.extend(sync_page.terminal_discovery_lags)
                sync_complete = sync_page.complete
                if sync_page.complete or sync_page.scanned == 0:
                    break

            batch_policy = parent.spec_json.get('batch') or {}
            batch = await session.get(BatchRunRow, parent_execution_id)
            if (
                batch is not None
                and batch.control_state == 'active'
                and bool(batch_policy.get('fail_fast'))
                and await repo.has_exhausted_failure(parent_execution_id)
            ):
                changed += int(await repo.begin_stop(parent_execution_id, reason='fail_fast', now=now))

            batch = await session.get(BatchRunRow, parent_execution_id)
            stop_complete = batch is None or batch.control_state != 'stopping'
            if batch is not None and batch.control_state == 'stopping':
                execution_repo = ExecutionRepository(session, shard_count=self.config.shard_count)
                while pages < page_budget and scanned < scan_budget:
                    current_page_size = min(page_size, scan_budget - scanned)
                    stop_page = await repo.next_stop_page(
                        parent_execution_id,
                        page_size=current_page_size,
                        now=now,
                    )
                    pages += 1
                    scanned += stop_page.scanned
                    changed += await execution_repo.request_stop_many(
                        stop_page.child_execution_ids,
                        stop_reason=batch.stop_reason,
                        now=now,
                    )
                    changed += int(stop_page.scanned > 0)
                    stop_complete = stop_page.complete
                    if stop_page.complete or stop_page.scanned == 0:
                        break
            more_work = not sync_complete or not stop_complete

            finalized = await repo.finalize_parent_if_ready(parent_execution_id, now=now)
            if finalized is not None:
                if finalized.finalized:
                    cleanup_pending = _batch_uses_task_store(batch, task_store_available=self._task_store is not None)
                    if cleanup_pending:
                        batch.last_reconciled_at = None
                    changed += 1
                    more_work = False
                elif finalized.result_artifact_required:
                    publication = finalized
            await lease.ensure_owned()
            await repo.fence_lifecycle_claim(claim, now=utcnow())
            await session.commit()
        if publication is None:
            if cleanup_pending:
                changed += await self._cleanup_task_store_records(claim, lease=lease)
            return _BatchLifecyclePageResult(
                changed=changed,
                scanned=scanned,
                pages=pages,
                more_work=more_work,
                terminal_discovery_lags=tuple(terminal_discovery_lags),
            )

        try:
            await lease.ensure_owned()
            await self._publish_parent_result(parent_execution_id, publication)
        except TransientResultPublishError:
            raise
        except Exception as exc:
            async with self.session_factory() as session:
                repo = BatchRepository(session, config=self.config)
                await repo.fence_lifecycle_claim(claim, now=utcnow())
                await repo.fail_parent_result_persistence(
                    parent_execution_id,
                    summary=publication.summary,
                    assignment_id=publication.assignment_id,
                    attempt_id=publication.attempt_id,
                    cause=type(exc).__name__,
                    now=utcnow(),
                )
                finalized_batch = await session.get(BatchRunRow, parent_execution_id)
                cleanup_pending = _batch_uses_task_store(
                    finalized_batch, task_store_available=self._task_store is not None
                )
                if cleanup_pending:
                    finalized_batch.last_reconciled_at = None
                await lease.ensure_owned()
                await repo.fence_lifecycle_claim(claim, now=utcnow())
                await session.commit()
            if cleanup_pending:
                changed += await self._cleanup_task_store_records(claim, lease=lease)
            return _BatchLifecyclePageResult(
                changed=changed + 1,
                scanned=scanned,
                pages=pages,
                more_work=False,
                terminal_discovery_lags=tuple(terminal_discovery_lags),
            )
        async with self.session_factory() as session:
            repo = BatchRepository(session, config=self.config)
            await repo.fence_lifecycle_claim(claim, now=utcnow())
            finalized = await repo.finalize_parent_if_ready(parent_execution_id, now=utcnow())
            if finalized is None or not finalized.finalized:
                raise RuntimeError('batch_parent_result_artifact_not_linked')
            finalized_batch = await session.get(BatchRunRow, parent_execution_id)
            cleanup_pending = _batch_uses_task_store(finalized_batch, task_store_available=self._task_store is not None)
            if cleanup_pending:
                finalized_batch.last_reconciled_at = None
            await lease.ensure_owned()
            await repo.fence_lifecycle_claim(claim, now=utcnow())
            await session.commit()
        if cleanup_pending:
            changed += await self._cleanup_task_store_records(claim, lease=lease)
        return _BatchLifecyclePageResult(
            changed=changed + 2,
            scanned=scanned,
            pages=pages,
            more_work=False,
            terminal_discovery_lags=tuple(terminal_discovery_lags),
        )

    async def _cleanup_task_store_records(self, claim: BatchLifecycleClaim, *, lease: PeriodicLease) -> int:
        if self._task_store is None:
            raise _BatchTaskStoreCleanupError('batch_task_store_required_for_cleanup')
        await lease.ensure_owned()
        try:
            await self._task_store.delete_records(parent_execution_id=claim.parent_execution_id)
        except Exception as exc:
            raise _BatchTaskStoreCleanupError('batch_task_store_cleanup_failed') from exc
        await lease.ensure_owned()
        return 1

    async def _publish_parent_result(
        self,
        parent_execution_id: str,
        publication: BatchFinalizationResult,
    ) -> None:
        if publication.assignment_id is None or publication.attempt_id is None:
            raise ValueError('batch_parent_result_scope_missing')
        content = encode_run_result_content(
            status='succeeded',
            output={'batch': publication.summary},
            error=None,
        )
        await self._result_artifact_client().publish_run_result(
            execution_id=parent_execution_id,
            assignment_id=publication.assignment_id,
            attempt_id=publication.attempt_id,
            producer_key=run_result_producer_key(parent_execution_id, publication.attempt_id),
            content=content,
        )

    def _result_artifact_client(self) -> RunResultPublisher:
        if self._artifact_client is None:
            self._artifact_transport = create_application_gateway_artifact_transport(self.config, required=True)
            assert self._artifact_transport is not None
            self._owns_artifact_transport = True
            self._artifact_client = RunResultArtifactGateway(self._artifact_transport)
        return self._artifact_client


def _initial_submit_window_pending(parent: ExecutionRow, batch: BatchRunRow) -> bool:
    if batch.generation_status != 'complete':
        return True
    total_items = int(batch.total_items if batch.total_items is not None else batch.materialized_count or 0)
    if total_items <= 0:
        return False
    batch_policy = parent.spec_json.get('batch') or {}
    max_inflight = max(1, int(batch_policy.get('max_inflight', 200)))
    initial_target = min(total_items, max_inflight)
    return int(batch.submitted_items or 0) < initial_target


def _batch_uses_task_store(batch: BatchRunRow | None, *, task_store_available: bool = False) -> bool:
    manifest = batch.task_list_manifest_json if batch is not None else None
    # All task-list versions use the store when configured; v3 requires it.
    return batch is not None and (
        task_store_available or (isinstance(manifest, dict) and manifest.get('record_version') == 3)
    )


__all__ = ['BatchLifecycle', 'BatchLifecycleReconciler']
