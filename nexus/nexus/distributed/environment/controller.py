from __future__ import annotations

import logging

from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any, TypeVar

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from nexus.distributed.artifacts.tickets import ArtifactContentOperation, ArtifactContentTicketSigner
from nexus.distributed.config.views import EnvironmentConfigView
from nexus.distributed.dependency_build.models import DependencyBuildStatus
from nexus.distributed.environment.allocator import (
    AllocationSpecConflictError,
    EnvironmentProvisioningTimeoutError,
    KubernetesEnvironmentAllocator,
    PredefinedEnvironmentAllocator,
    TkexEnvironmentAllocator,
)
from nexus.distributed.environment.database_secret import EnvironmentDatabaseSecretSynchronizer
from nexus.distributed.environment.kubernetes import WorkloadPermanentError
from nexus.distributed.environment.models import (
    EnvironmentAllocationStatus,
    EnvironmentProviderType,
)
from nexus.distributed.environment.naming import (
    allocation_resource_name,
    dns_name,
)
from nexus.distributed.environment.ports import (
    AllocationSnapshot,
    EnvironmentAllocator,
    WorkloadClientResolverPort,
)
from nexus.distributed.environment.provider_config import KubernetesProviderConfig, TkexProviderConfig
from nexus.distributed.environment.rendering import (
    AllocationWorkloadRenderer as ProviderWorkloadRenderer,
)
from nexus.distributed.environment.rendering import (
    TkexAllocationWorkloadRenderer,
)
from nexus.distributed.execution.models import (
    TERMINAL_EXECUTION_STATUSES,
    ExecutionStatus,
)
from nexus.distributed.instance_manager.liveness import manager_is_healthy_clause
from nexus.distributed.observability import ControlLoopWork
from nexus.distributed.persistence.orm import (
    ArtifactRefRow,
    BatchRunRow,
    DependencyBuildRow,
    EnvironmentAllocationRow,
    EnvironmentRow,
    ExecutionRow,
    InstanceManagerLivenessRow,
    InstanceManagerRow,
)
from nexus.distributed.ports import ReconcileController
from nexus.distributed.repositories.allocation import AllocationRepository, EnvironmentAllocationClaim
from nexus.distributed.repositories.events import append_control_event_row
from nexus.distributed.repositories.quota import QuotaRepository
from nexus.distributed.utils.clock import utcnow
from nexus.distributed.utils.identity import canonical_sha256
from nexus.distributed.utils.leases import LeaseLostError, PeriodicLease


logger = logging.getLogger(__name__)
T = TypeVar('T')


class EnvironmentController(ReconcileController):
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker,
        config: EnvironmentConfigView,
        allocators: dict[str, EnvironmentAllocator] | None = None,
        client_resolver: WorkloadClientResolverPort,
        owner_id: str | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.config = config
        self.owner_id = owner_id or config.environment_controller_owner_id
        self._allocators = dict(allocators or {})
        self.client_resolver = client_resolver
        self._artifact_content_signer = ArtifactContentTicketSigner(
            keys=config.artifact_ticket_keyring(),
            current_key_id=config.artifact_upload_ticket_current_key_id,
            ttl=config.environment_artifact_content_ticket_ttl,
        )
        self._last_environment_stats = ControlLoopWork()

    @property
    def last_environment_stats(self) -> ControlLoopWork:
        return self._last_environment_stats

    async def reconcile_once(self) -> int:
        changed = 0
        processed: set[str] = set()
        for _ in range(self.config.environment_controller_batch_size):
            async with self.session_factory() as session:
                claims = await AllocationRepository(session).claim_allocations(
                    operation_owner=self.owner_id,
                    operation_ttl=self.config.environment_operation_ttl,
                    statuses=None,
                    limit=1,
                    now=utcnow(),
                    exclude_allocation_ids=processed,
                )
                await session.commit()
            if not claims:
                break
            claim = claims[0]
            processed.add(claim.allocation_id)
            snapshot: AllocationSnapshot | None = None
            try:
                async with self._claim_lease(claim) as lease:
                    try:
                        snapshot = await self._load_snapshot(claim.allocation_id)
                        await self._reconcile_claim(claim, snapshot, lease=lease)
                    except (AllocationSpecConflictError, WorkloadPermanentError) as exc:
                        if snapshot is not None:
                            await self._handle_permanent_error(claim, snapshot, exc, lease=lease)
                        else:
                            await self._record_claim_load_error(claim, exc, lease=lease)
                    except LeaseLostError:
                        raise
                    except Exception as exc:
                        logger.exception(
                            'Allocation reconcile failed: owner=%s allocation_id=%s operation_id=%s',
                            self.owner_id,
                            claim.allocation_id,
                            claim.operation_id,
                        )
                        if snapshot is not None:
                            await self._record_error(claim, snapshot, exc, lease=lease)
                        else:
                            await self._record_claim_load_error(claim, exc, lease=lease)
            except LeaseLostError:
                logger.warning(
                    'Allocation lease lost: owner=%s allocation_id=%s operation_id=%s',
                    self.owner_id,
                    claim.allocation_id,
                    claim.operation_id,
                )
            changed += 1
        self._last_environment_stats = ControlLoopWork(
            scanned=len(processed),
            claimed=len(processed),
            changed=changed,
        )
        return changed

    async def close(self) -> None:
        seen: set[int] = set()
        for allocator in self._allocators.values():
            if id(allocator) in seen:
                continue
            seen.add(id(allocator))
            await allocator.close()

    async def _reconcile_claim(
        self,
        claim: EnvironmentAllocationClaim,
        allocation: AllocationSnapshot,
        *,
        lease: PeriodicLease | None = None,
    ) -> None:
        batch_finalized = allocation.batch_control_state in {None, 'finalized'}
        terminal = allocation.execution_status is None or (
            allocation.execution_status in TERMINAL_EXECUTION_STATUSES and batch_finalized
        )
        stop_during_provisioning = allocation.execution_status == ExecutionStatus.STOPPING and allocation.status in {
            EnvironmentAllocationStatus.REQUESTED,
            EnvironmentAllocationStatus.PROVISIONING,
        }
        if (terminal or stop_during_provisioning) and allocation.status not in {
            EnvironmentAllocationStatus.DRAINING,
            EnvironmentAllocationStatus.DELETING,
        }:
            await self._transition(claim, EnvironmentAllocationStatus.DRAINING, lease=lease)
            return
        if (
            allocation.status in {EnvironmentAllocationStatus.REQUESTED, EnvironmentAllocationStatus.PROVISIONING}
            and allocation.created_at is not None
            and allocation.provisioning_timeout_seconds is not None
            and utcnow() >= allocation.created_at + timedelta(seconds=allocation.provisioning_timeout_seconds)
        ):
            raise EnvironmentProvisioningTimeoutError('environment_provisioning_timeout')
        if allocation.status not in {
            EnvironmentAllocationStatus.DRAINING,
            EnvironmentAllocationStatus.DELETING,
        } and allocation.batch_dependency_status not in {None, 'ready', 'not_required'}:
            if allocation.batch_dependency_status == 'failed':
                raise AllocationSpecConflictError('batch_module_dependency_failed')
            await self._complete_observation(
                claim,
                ready_manager_count=allocation.ready_manager_count,
                provider_resource_ref=allocation.provider_resource_ref,
                lease=lease,
            )
            return
        if allocation.status not in {
            EnvironmentAllocationStatus.DRAINING,
            EnvironmentAllocationStatus.DELETING,
        } and allocation.dependency_build_status not in {None, DependencyBuildStatus.READY}:
            retrying_build = (
                allocation.dependency_build_status == DependencyBuildStatus.FAILED
                and allocation.dependency_build_next_retry_at is not None
            )
            if allocation.dependency_build_status == DependencyBuildStatus.CANCELLED or (
                allocation.dependency_build_status == DependencyBuildStatus.FAILED and not retrying_build
            ):
                raise AllocationSpecConflictError(
                    f'dependency_build_{allocation.dependency_build_status}:{allocation.dependency_build_id}'
                )
            await self._complete_observation(
                claim,
                ready_manager_count=allocation.ready_manager_count,
                provider_resource_ref=allocation.provider_resource_ref,
                lease=lease,
            )
            return

        if allocation.status == EnvironmentAllocationStatus.REQUESTED:
            allocator = self._allocator_for(
                allocation.provider_type,
                allocation.provider_config_ref,
                provider_config_snapshot=allocation.provider_config_snapshot,
            )
            await self._ensure_claim_owned(lease)
            await self._reserve_quota(allocation.allocation_id)
            await self._record_provisioning_clock(claim, allocation, started=True)
            observation = await self._provider_call(lease, lambda: allocator.create(allocation))
            if allocation.provider_type == EnvironmentProviderType.PREDEFINED:
                await self._transition(
                    claim,
                    EnvironmentAllocationStatus.READY,
                    ready_manager_count=allocation.desired_manager_count,
                    provider_resource_ref=observation.provider_resource_ref,
                    lease=lease,
                )
            else:
                await self._transition(
                    claim,
                    EnvironmentAllocationStatus.PROVISIONING,
                    ready_manager_count=0,
                    provider_resource_ref=observation.provider_resource_ref,
                    lease=lease,
                )
            return

        if allocation.status in {
            EnvironmentAllocationStatus.PROVISIONING,
            EnvironmentAllocationStatus.READY,
        }:
            allocator = self._allocator_for(
                allocation.provider_type,
                allocation.provider_config_ref,
                provider_config_snapshot=allocation.provider_config_snapshot,
            )
            observation = await self._provider_call(lease, lambda: allocator.observe(allocation))
            if not observation.exists:
                observation = await self._provider_call(lease, lambda: allocator.create(allocation))
            ready_managers = await self._ready_manager_count(allocation)
            provider_ready = observation.provider_ready_replicas >= allocation.desired_manager_count
            if (
                allocation.status == EnvironmentAllocationStatus.PROVISIONING
                and provider_ready
                and (ready_managers == allocation.desired_manager_count)
            ):
                await self._transition(
                    claim,
                    EnvironmentAllocationStatus.READY,
                    ready_manager_count=ready_managers,
                    provider_resource_ref=observation.provider_resource_ref,
                    lease=lease,
                )
            else:
                await self._complete_observation(
                    claim,
                    ready_manager_count=ready_managers,
                    provider_resource_ref=observation.provider_resource_ref,
                    lease=lease,
                )
            return

        if allocation.status == EnvironmentAllocationStatus.DRAINING:
            if (
                allocation.provider_type == EnvironmentProviderType.PREDEFINED
                or allocation.provider_resource_ref is None
            ):
                transitioned = await self._transition(
                    claim,
                    EnvironmentAllocationStatus.RELEASED,
                    lease=lease,
                )
                if transitioned:
                    await self._release_quota(allocation.allocation_id)
            else:
                allocator = self._allocator_for(
                    allocation.provider_type,
                    allocation.provider_config_ref,
                    provider_config_snapshot=allocation.provider_config_snapshot,
                )
                await self._provider_call(lease, lambda: allocator.delete(allocation))
                await self._transition(claim, EnvironmentAllocationStatus.DELETING, lease=lease)
            return

        if allocation.status == EnvironmentAllocationStatus.DELETING:
            if allocation.provider_resource_ref is None:
                transitioned = await self._transition(
                    claim,
                    EnvironmentAllocationStatus.RELEASED,
                    ready_manager_count=0,
                    lease=lease,
                )
                if transitioned:
                    await self._release_quota(allocation.allocation_id)
                return
            allocator = self._allocator_for(
                allocation.provider_type,
                allocation.provider_config_ref,
                provider_config_snapshot=allocation.provider_config_snapshot,
            )
            await self._provider_call(lease, lambda: allocator.delete(allocation))
            observation = await self._provider_call(lease, lambda: allocator.observe_deletion(allocation))
            if observation.exists:
                await self._complete_observation(
                    claim,
                    ready_manager_count=0,
                    provider_resource_ref=observation.provider_resource_ref,
                    lease=lease,
                )
            else:
                transitioned = await self._transition(
                    claim,
                    EnvironmentAllocationStatus.RELEASED,
                    ready_manager_count=0,
                    lease=lease,
                )
                if transitioned:
                    await self._release_quota(allocation.allocation_id)
            return

        await self._complete_observation(
            claim,
            ready_manager_count=allocation.ready_manager_count,
            provider_resource_ref=allocation.provider_resource_ref,
            lease=lease,
        )

    async def _load_snapshot(self, allocation_id: str) -> AllocationSnapshot:
        async with self.session_factory() as session:
            row = await session.get(EnvironmentAllocationRow, allocation_id)
            if row is None:
                raise KeyError(allocation_id)
            environment = await session.get(EnvironmentRow, row.environment_id)
            if environment is None:
                raise RuntimeError('allocation_environment_missing')
            execution = await session.get(ExecutionRow, row.top_level_execution_id)
            batch = (
                await session.get(BatchRunRow, row.top_level_execution_id)
                if execution is not None and execution.spec_type == 'nexus_batch'
                else None
            )
            allocation_spec = dict(row.allocation_spec_json or {})
            dependency_build = None
            if row.dependency_build_id is not None:
                dependency_build = await session.get(DependencyBuildRow, row.dependency_build_id)
                if dependency_build is None:
                    raise AllocationSpecConflictError('allocation_dependency_build_missing')
                if dependency_build.status == DependencyBuildStatus.READY:
                    if dependency_build.wheelhouse_ref_json is None or dependency_build.manifest_json is None:
                        raise AllocationSpecConflictError('ready_dependency_build_output_missing')
                    dependency_snapshot = dict(allocation_spec.get('dependency_build') or {})
                    dependency_snapshot['wheelhouse'] = dict(dependency_build.wheelhouse_ref_json)
                    dependency_snapshot['metadata'] = dict(dependency_build.manifest_json)
                    allocation_spec['dependency_build'] = dependency_snapshot
            snapshot_allocation_type = allocation_spec.get('allocation_type')
            snapshot_provider_type = allocation_spec.get('provider_type')
            snapshot_provider_config_ref = allocation_spec.get('provider_config_ref')
            snapshot_provider_config = allocation_spec.get('provider_config')
            provider_snapshot_complete = (
                isinstance(snapshot_allocation_type, str)
                and bool(snapshot_allocation_type)
                and isinstance(snapshot_provider_type, str)
                and bool(snapshot_provider_type)
                and (
                    snapshot_provider_type == EnvironmentProviderType.PREDEFINED
                    or isinstance(snapshot_provider_config_ref, str)
                    and bool(snapshot_provider_config_ref)
                    and isinstance(snapshot_provider_config, dict)
                )
            )
            if not provider_snapshot_complete and row.config_version != environment.config_version:
                raise AllocationSpecConflictError('allocation_provider_snapshot_missing')
            allocation_type = str(allocation_spec.get('allocation_type') or environment.allocation_type)
            provider_type = str(allocation_spec.get('provider_type') or environment.provider_type)
            provider_config_ref = allocation_spec.get('provider_config_ref', environment.provider_config_ref)
            if provider_config_ref is not None:
                provider_config_ref = str(provider_config_ref)
            execution_spec = dict(execution.spec_json or {}) if execution is not None else {}
            provisioning_timeout = execution_spec.get('provisioning_timeout_seconds')
            if provisioning_timeout is None:
                provisioning_timeout = allocation_spec.get('provisioning_timeout_seconds')
            if provisioning_timeout is None:
                provisioning_timeout = (environment.default_timeouts_json or {}).get('provisioning_timeout_seconds')
            credential_name = None
            credential_token = None
            if provider_type != EnvironmentProviderType.PREDEFINED:
                artifact_ids = list(
                    await session.scalars(
                        select(ArtifactRefRow.artifact_id).where(
                            ArtifactRefRow.resource_type == 'allocation',
                            ArtifactRefRow.resource_id == row.allocation_id,
                        )
                    )
                )
                if dependency_build is not None and dependency_build.wheelhouse_ref_json is not None:
                    artifact_ids.append(str(dependency_build.wheelhouse_ref_json['artifact_id']))
                operations: list[ArtifactContentOperation] = ['artifact:read']
                operations.append('task_list:write' if row.workload == 'orchestration' else 'run_result:write')
                ticket_now = _credential_ticket_epoch(
                    utcnow(),
                    ttl=self.config.environment_artifact_content_ticket_ttl,
                )
                credential_name = dns_name(
                    f'{allocation_resource_name(row.allocation_id)}-artifacts',
                    fallback='nexus-allocation',
                )
                credential_token = self._artifact_content_signer.issue(
                    execution_id=row.top_level_execution_id,
                    top_level_execution_id=row.top_level_execution_id,
                    environment_id=row.environment_id,
                    allocation_id=row.allocation_id,
                    operations=operations,
                    artifact_ids=artifact_ids,
                    now=ticket_now,
                )
            return AllocationSnapshot(
                allocation_id=row.allocation_id,
                top_level_execution_id=row.top_level_execution_id,
                owner_principal_id=row.owner_principal_id,
                environment_id=row.environment_id,
                allocation_type=allocation_type,
                provider_type=provider_type,
                pool_id=row.pool_id,
                workload=row.workload,
                status=row.status,
                desired_manager_count=row.desired_manager_count,
                ready_manager_count=row.ready_manager_count,
                cpu_millicores=row.cpu_millicores,
                memory_bytes=row.memory_bytes,
                slots=row.slots,
                allocation_spec=allocation_spec,
                allocation_spec_hash=row.allocation_spec_hash,
                provider_resource_ref=(
                    dict(row.provider_resource_ref_json) if row.provider_resource_ref_json else None
                ),
                provider_config_ref=provider_config_ref,
                provider_config_snapshot=(
                    dict(snapshot_provider_config) if isinstance(snapshot_provider_config, dict) else None
                ),
                execution_status=execution.status if execution is not None else None,
                batch_control_state=batch.control_state if batch is not None else None,
                batch_dependency_status=(
                    batch.dependency_status if batch is not None and row.workload == 'execution' else None
                ),
                created_at=row.created_at,
                provisioning_timeout_seconds=(int(provisioning_timeout) if provisioning_timeout is not None else None),
                artifact_credential_secret_name=credential_name,
                artifact_credential_token=credential_token,
                dependency_build_id=row.dependency_build_id,
                dependency_build_status=(dependency_build.status if dependency_build is not None else None),
                dependency_build_next_retry_at=(
                    dependency_build.next_retry_at if dependency_build is not None else None
                ),
            )

    async def _record_provisioning_clock(
        self,
        claim: EnvironmentAllocationClaim,
        allocation: AllocationSnapshot,
        *,
        started: bool = False,
    ) -> None:
        deadline = (
            allocation.created_at + timedelta(seconds=allocation.provisioning_timeout_seconds)
            if allocation.created_at is not None and allocation.provisioning_timeout_seconds is not None
            else None
        )
        async with self.session_factory() as session:
            recorded = await AllocationRepository(session).record_provisioning_clock(
                claim,
                now=utcnow(),
                deadline_at=deadline,
                started=started,
            )
            if not recorded:
                raise LeaseLostError('allocation_clock_claim_lost')
            await session.commit()

    def _allocator_for(
        self,
        provider_type: str,
        provider_config_ref: str | None,
        *,
        provider_config_snapshot: dict[str, Any] | None = None,
    ) -> EnvironmentAllocator:
        configured_key = f'{provider_type}:{provider_config_ref or "default"}'
        allocator_key = configured_key
        if provider_config_snapshot is not None:
            allocator_key = f'{configured_key}:{canonical_sha256(provider_config_snapshot)}'
        existing = self._allocators.get(allocator_key) or self._allocators.get(provider_type)
        if existing is None and provider_config_snapshot is not None:
            existing = self._allocators.get(configured_key)
        if existing is not None:
            return existing
        if provider_type == EnvironmentProviderType.PREDEFINED:
            allocator: EnvironmentAllocator = PredefinedEnvironmentAllocator()
        elif provider_type == EnvironmentProviderType.KUBERNETES:
            try:
                resolved = self.client_resolver.resolve(
                    provider_type=provider_type,
                    provider_config_ref=provider_config_ref,
                    snapshot=provider_config_snapshot,
                )
            except ValueError as exc:
                raise AllocationSpecConflictError(str(exc)) from exc
            provider_config = resolved.provider_config
            if not isinstance(provider_config, KubernetesProviderConfig):
                raise AssertionError('resolved Kubernetes provider config has the wrong type')
            renderer = ProviderWorkloadRenderer(
                config=self.config,
                workload_namespace=provider_config.namespace,
                resource_request_ratio=provider_config.manager_resource_request_ratio,
            )
            allocator = KubernetesEnvironmentAllocator(
                client=resolved.client,
                renderer=renderer,
                database_secret_synchronizer=self._database_secret_synchronizer(provider_config.namespace),
            )
        elif provider_type == EnvironmentProviderType.TKEX:
            try:
                resolved = self.client_resolver.resolve(
                    provider_type=provider_type,
                    provider_config_ref=provider_config_ref,
                    snapshot=provider_config_snapshot,
                )
            except ValueError as exc:
                raise AllocationSpecConflictError(str(exc)) from exc
            provider_config = resolved.provider_config
            if not isinstance(provider_config, TkexProviderConfig):
                raise AssertionError('resolved TKEx provider config has the wrong type')
            renderer = TkexAllocationWorkloadRenderer(
                config=self.config,
                cluster_name=provider_config.cluster_name,
                workload_namespace=provider_config.namespace,
                zhiyan_log=provider_config.zhiyan_log,
            )
            allocator = TkexEnvironmentAllocator(
                client=resolved.client,
                renderer=renderer,
                database_secret_synchronizer=self._database_secret_synchronizer(provider_config.namespace),
            )
        else:
            raise ValueError(f'unsupported_environment_provider:{provider_type}')
        self._allocators[allocator_key] = allocator
        return allocator

    def _database_secret_synchronizer(self, namespace: str) -> EnvironmentDatabaseSecretSynchronizer:
        return EnvironmentDatabaseSecretSynchronizer(
            namespace=namespace,
            app_name=self.config.environment_app_name,
            release_name=self.config.environment_release_name,
            secret_name=self.config.environment_database_url_secret_name,
            secret_key=self.config.environment_database_url_secret_key,
            database_url=self.config.read_database_url(),
        )

    async def _ready_manager_count(self, allocation: AllocationSnapshot) -> int:
        if allocation.provider_type == EnvironmentProviderType.PREDEFINED:
            return allocation.desired_manager_count
        healthy_since = utcnow() - self.config.manager_healthy_ttl
        expected_ids = {f'{allocation.allocation_id}-{ordinal}' for ordinal in range(allocation.desired_manager_count)}
        if not expected_ids:
            return 0
        async with self.session_factory() as session:
            manager_ids = set(
                await session.scalars(
                    select(InstanceManagerRow.manager_id)
                    .outerjoin(
                        InstanceManagerLivenessRow,
                        and_(
                            InstanceManagerLivenessRow.manager_id == InstanceManagerRow.manager_id,
                            InstanceManagerLivenessRow.session_id == InstanceManagerRow.session_id,
                        ),
                    )
                    .where(
                        InstanceManagerRow.manager_id.in_(expected_ids),
                        InstanceManagerRow.owner_principal_id == allocation.owner_principal_id,
                        InstanceManagerRow.environment_id == allocation.environment_id,
                        InstanceManagerRow.pool_id == allocation.pool_id,
                        InstanceManagerRow.allocation_id == allocation.allocation_id,
                        InstanceManagerRow.workload == allocation.workload,
                        InstanceManagerRow.allocation_spec_hash == allocation.allocation_spec_hash,
                        InstanceManagerRow.session_id.is_not(None),
                        InstanceManagerRow.draining.is_(False),
                        manager_is_healthy_clause(healthy_since),
                    )
                )
            )
        return len(manager_ids & expected_ids)

    async def _transition(
        self,
        claim: EnvironmentAllocationClaim,
        status: str,
        *,
        ready_manager_count: int | None = None,
        provider_resource_ref: dict[str, Any] | None = None,
        error: Exception | None = None,
        lease: PeriodicLease | None = None,
    ) -> bool:
        await self._finish_claim_lease(lease)
        now = utcnow()
        async with self.session_factory() as session:
            transitioned = await AllocationRepository(session).transition(
                claim,
                status=status,
                ready_manager_count=ready_manager_count,
                provider_resource_ref=provider_resource_ref,
                last_error_code=type(error).__name__ if error is not None else None,
                last_error_message=str(error) if error is not None else None,
                now=now,
            )
            if not transitioned:
                await session.rollback()
                return False
            await session.commit()
            return True

    async def _complete_observation(
        self,
        claim: EnvironmentAllocationClaim,
        *,
        ready_manager_count: int,
        provider_resource_ref: dict[str, Any] | None,
        error: Exception | None = None,
        lease: PeriodicLease | None = None,
    ) -> None:
        await self._finish_claim_lease(lease)
        now = utcnow()
        async with self.session_factory() as session:
            row = await session.get(EnvironmentAllocationRow, claim.allocation_id)
            if (
                row is None
                or row.operation_id != claim.operation_id
                or row.operation_owner != claim.operation_owner
                or row.status_version != claim.status_version
                or row.operation_lease_until is None
                or row.operation_lease_until <= now
            ):
                return
            next_ref = dict(provider_resource_ref) if provider_resource_ref else None
            error_code = type(error).__name__ if error is not None else None
            error_message = str(error)[:2048] if error is not None else None
            changed = (
                row.ready_manager_count != ready_manager_count
                or row.provider_resource_ref_json != next_ref
                or row.last_error_code != error_code
                or row.last_error_message != error_message
            )
            row.ready_manager_count = ready_manager_count
            row.provider_resource_ref_json = next_ref
            row.last_error_code = error_code
            row.last_error_message = error_message
            if changed:
                row.last_progress_at = now
                row.updated_at = now
            row.operation_id = None
            row.operation_owner = None
            row.operation_lease_until = None
            row.status_version += 1
            await session.commit()

    async def _reserve_quota(self, allocation_id: str) -> None:
        async with self.session_factory() as session:
            await QuotaRepository(session, default_limits=self.config.default_user_quota).reserve_allocation(
                allocation_id,
                now=utcnow(),
            )
            await session.commit()

    async def _release_quota(self, allocation_id: str) -> None:
        async with self.session_factory() as session:
            await QuotaRepository(session, default_limits=self.config.default_user_quota).release_allocation(
                allocation_id,
                now=utcnow(),
            )
            await session.commit()

    async def _record_error(
        self,
        claim: EnvironmentAllocationClaim,
        allocation: AllocationSnapshot,
        error: Exception,
        *,
        lease: PeriodicLease | None = None,
    ) -> None:
        await self._complete_observation(
            claim,
            ready_manager_count=allocation.ready_manager_count,
            provider_resource_ref=allocation.provider_resource_ref,
            error=error,
            lease=lease,
        )

    async def _record_claim_load_error(
        self,
        claim: EnvironmentAllocationClaim,
        error: Exception,
        *,
        lease: PeriodicLease | None = None,
    ) -> None:
        async with self.session_factory() as session:
            row = await session.get(EnvironmentAllocationRow, claim.allocation_id)
            ready_manager_count = row.ready_manager_count if row is not None else 0
            provider_resource_ref = (
                dict(row.provider_resource_ref_json) if row is not None and row.provider_resource_ref_json else None
            )
        await self._complete_observation(
            claim,
            ready_manager_count=ready_manager_count,
            provider_resource_ref=provider_resource_ref,
            error=error,
            lease=lease,
        )

    async def _fail_provisioning(
        self,
        claim: EnvironmentAllocationClaim,
        allocation: AllocationSnapshot,
        error: Exception,
        *,
        lease: PeriodicLease | None = None,
    ) -> bool:
        await self._finish_claim_lease(lease)
        now = utcnow()
        async with self.session_factory() as session:
            next_status = await AllocationRepository(session).fail_provisioning(
                claim,
                last_error_code=type(error).__name__,
                last_error_message=str(error),
                now=now,
            )
            if next_status is None:
                await session.rollback()
                return False
            execution = await session.get(ExecutionRow, allocation.top_level_execution_id)
            if execution is not None and execution.status not in TERMINAL_EXECUTION_STATUSES:
                old_status = execution.status
                execution.status = ExecutionStatus.FAILED
                execution.error_json = {
                    'type': 'EnvironmentProvisioningFailed',
                    'message': str(error),
                }
                execution.finished_at = utcnow()
                execution.updated_at = execution.finished_at
                execution.version += 1
                await append_control_event_row(
                    session,
                    execution,
                    event_type='execution.status.changed',
                    source='reconciler',
                    payload={
                        'from': old_status,
                        'to': ExecutionStatus.FAILED,
                        'reason': 'EnvironmentProvisioningFailed',
                    },
                    now=execution.finished_at,
                )
            if next_status == EnvironmentAllocationStatus.RELEASED:
                await QuotaRepository(session, default_limits=self.config.default_user_quota).release_allocation(
                    allocation.allocation_id,
                    now=now,
                )
            await session.commit()
            return True

    async def _handle_permanent_error(
        self,
        claim: EnvironmentAllocationClaim,
        allocation: AllocationSnapshot,
        error: Exception,
        *,
        lease: PeriodicLease | None = None,
    ) -> None:
        if allocation.status in {
            EnvironmentAllocationStatus.REQUESTED,
            EnvironmentAllocationStatus.PROVISIONING,
        }:
            await self._fail_provisioning(claim, allocation, error, lease=lease)
            return
        if allocation.status == EnvironmentAllocationStatus.READY:
            await self._transition(
                claim,
                EnvironmentAllocationStatus.DRAINING,
                ready_manager_count=0,
                provider_resource_ref=allocation.provider_resource_ref,
                error=error,
                lease=lease,
            )
            return
        if allocation.status in {
            EnvironmentAllocationStatus.DRAINING,
            EnvironmentAllocationStatus.DELETING,
        }:
            if allocation.provider_resource_ref is None:
                transitioned = await self._transition(
                    claim,
                    EnvironmentAllocationStatus.RELEASED,
                    ready_manager_count=0,
                    error=error,
                    lease=lease,
                )
                if transitioned:
                    await self._release_quota(allocation.allocation_id)
                return
            await self._record_error(claim, allocation, error, lease=lease)
            return
        await self._record_error(claim, allocation, error, lease=lease)

    def _claim_lease(self, claim: EnvironmentAllocationClaim) -> PeriodicLease:
        async def renew() -> bool:
            async with self.session_factory() as session:
                renewed = await AllocationRepository(session).renew_claim(
                    claim,
                    operation_ttl=self.config.environment_operation_ttl,
                    now=utcnow(),
                )
                if renewed:
                    await session.commit()
                else:
                    await session.rollback()
                return renewed

        return PeriodicLease(
            renew=renew,
            interval=self.config.environment_operation_lease_renew_interval.total_seconds(),
            lost_code='environment_allocation_lease_lost',
        )

    async def _provider_call(
        self,
        lease: PeriodicLease | None,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        if lease is not None:
            await lease.renew_now()
        result = await operation()
        if lease is not None:
            await lease.renew_now()
        return result

    @staticmethod
    async def _ensure_claim_owned(lease: PeriodicLease | None) -> None:
        if lease is not None:
            await lease.ensure_owned()

    @staticmethod
    async def _finish_claim_lease(lease: PeriodicLease | None) -> None:
        if lease is None:
            return
        await lease.renew_now()
        await lease.close()
        await lease.ensure_owned()


def _credential_ticket_epoch(now: datetime, *, ttl: timedelta) -> datetime:
    ttl_seconds = int(ttl.total_seconds())
    if ttl_seconds < 2:
        raise ValueError('environment artifact content ticket TTL must be at least two seconds')
    rotation_seconds = max(1, ttl_seconds // 2)
    aware = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
    bucket = int(aware.timestamp()) // rotation_seconds * rotation_seconds
    return datetime.fromtimestamp(bucket, tz=timezone.utc).replace(tzinfo=None)
