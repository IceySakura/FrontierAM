from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from nexus.distributed.dependency_build.models import DependencyBuildStatus
from nexus.distributed.environment.models import EnvironmentAllocationStatus
from nexus.distributed.execution.models import (
    TERMINAL_EXECUTION_STATUSES,
    ExecutionStatus,
)


_EXECUTION_STATUSES = {
    ExecutionStatus.QUEUED,
    ExecutionStatus.ASSIGNED,
    ExecutionStatus.STARTING,
    ExecutionStatus.RUNNING,
    ExecutionStatus.WAITING_INPUT,
    ExecutionStatus.STOPPING,
    *TERMINAL_EXECUTION_STATUSES,
}
_ALLOCATION_STATUSES = {
    EnvironmentAllocationStatus.REQUESTED,
    EnvironmentAllocationStatus.PROVISIONING,
    EnvironmentAllocationStatus.READY,
    EnvironmentAllocationStatus.DRAINING,
    EnvironmentAllocationStatus.DELETING,
    EnvironmentAllocationStatus.RELEASED,
}
_DEPENDENCY_BUILD_STATUSES = {
    DependencyBuildStatus.REQUESTED,
    DependencyBuildStatus.BUILDING,
    DependencyBuildStatus.READY,
    DependencyBuildStatus.FAILED,
    DependencyBuildStatus.CANCELLED,
}
_BATCH_GENERATION_STATUSES = {'pending', 'publishing', 'materializing', 'complete'}
_BATCH_CONTROL_STATES = {'active', 'stopping', 'finalized'}
_BATCH_DEPENDENCY_STATUSES = {'pending', 'building', 'ready', 'failed', 'skipped', 'not_required'}
_RESULT_ARTIFACT_STATUSES = {'uploading', 'ready'}


class PhaseProjectionError(ValueError):
    """Persisted execution state cannot be projected without guessing."""


@dataclass(frozen=True, slots=True)
class AllocationPhaseState:
    status: str
    dependency_build_status: str | None = None
    dependency_build_next_retry_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class BatchPhaseState:
    generation_status: str
    control_state: str
    submission_closed: bool
    materialized_count: int
    total_items: int | None
    dependency_status: str = 'not_required'


@dataclass(frozen=True, slots=True)
class ExecutionPhaseSnapshot:
    status: str
    spec_type: str
    allocations: tuple[AllocationPhaseState, ...] = ()
    batch: BatchPhaseState | None = None
    result_artifact_status: str | None = None
    policy_deferred: bool = False


def project_execution_phase(snapshot: ExecutionPhaseSnapshot) -> str:
    """Project one reader-facing phase from persisted execution facts."""

    dependency_build_status = _validate_snapshot(snapshot)
    terminal = snapshot.status in TERMINAL_EXECUTION_STATUSES

    if terminal:
        if any(allocation.status != EnvironmentAllocationStatus.RELEASED for allocation in snapshot.allocations):
            return 'releasing'
        return snapshot.status

    if snapshot.batch is not None and snapshot.batch.dependency_status == 'failed':
        return 'dependency_build_failed'
    if snapshot.batch is not None and snapshot.batch.dependency_status == 'building':
        return 'building_dependencies'
    if dependency_build_status in {DependencyBuildStatus.FAILED, DependencyBuildStatus.CANCELLED}:
        return 'dependency_build_failed'
    if dependency_build_status in {DependencyBuildStatus.REQUESTED, DependencyBuildStatus.BUILDING}:
        return 'building_dependencies'

    if any(
        allocation.status in {EnvironmentAllocationStatus.REQUESTED, EnvironmentAllocationStatus.PROVISIONING}
        for allocation in snapshot.allocations
    ):
        return 'waiting_resources'
    if any(
        allocation.status
        in {
            EnvironmentAllocationStatus.DRAINING,
            EnvironmentAllocationStatus.DELETING,
            EnvironmentAllocationStatus.RELEASED,
        }
        for allocation in snapshot.allocations
    ):
        return 'releasing'

    if snapshot.spec_type == 'nexus_batch':
        assert snapshot.batch is not None
        if (
            snapshot.status == ExecutionStatus.STOPPING
            or snapshot.batch.control_state == 'stopping'
            or snapshot.batch.submission_closed
        ):
            return 'finalizing'
        if snapshot.batch.generation_status in {'pending', 'publishing'}:
            return 'generating'
        if snapshot.batch.generation_status == 'materializing':
            return 'materializing'
        return 'running_children'

    if snapshot.result_artifact_status is not None:
        return 'uploading_result'
    if snapshot.status == ExecutionStatus.QUEUED and snapshot.policy_deferred:
        return 'waiting_policy'
    return snapshot.status


def _validate_snapshot(snapshot: ExecutionPhaseSnapshot) -> str | None:
    if snapshot.status not in _EXECUTION_STATUSES:
        raise PhaseProjectionError(f'unknown_execution_status:{snapshot.status}')
    if not snapshot.spec_type:
        raise PhaseProjectionError('invalid_spec_type')

    dependency_build_statuses: set[str] = set()
    for allocation in snapshot.allocations:
        if allocation.status not in _ALLOCATION_STATUSES:
            raise PhaseProjectionError(f'unknown_allocation_status:{allocation.status}')
        if allocation.dependency_build_status is None:
            continue
        if allocation.dependency_build_status not in _DEPENDENCY_BUILD_STATUSES:
            raise PhaseProjectionError(f'unknown_dependency_build_status:{allocation.dependency_build_status}')
        dependency_build_status = allocation.dependency_build_status
        if (
            dependency_build_status == DependencyBuildStatus.FAILED
            and allocation.dependency_build_next_retry_at is not None
        ):
            dependency_build_status = DependencyBuildStatus.BUILDING
        dependency_build_statuses.add(dependency_build_status)
    if len(dependency_build_statuses) > 1:
        if snapshot.spec_type != 'nexus_batch':
            raise PhaseProjectionError(f'conflicting_dependency_build_status:{sorted(dependency_build_statuses)}')
        if dependency_build_statuses.intersection({DependencyBuildStatus.FAILED, DependencyBuildStatus.CANCELLED}):
            dependency_build_statuses = {DependencyBuildStatus.FAILED}
        elif dependency_build_statuses.intersection({DependencyBuildStatus.REQUESTED, DependencyBuildStatus.BUILDING}):
            dependency_build_statuses = {DependencyBuildStatus.BUILDING}
        else:
            dependency_build_statuses = {DependencyBuildStatus.READY}

    if snapshot.result_artifact_status not in {None, *_RESULT_ARTIFACT_STATUSES}:
        raise PhaseProjectionError(f'unknown_result_artifact_status:{snapshot.result_artifact_status}')

    if snapshot.spec_type == 'nexus_batch':
        if snapshot.batch is None:
            raise PhaseProjectionError('batch_state_missing')
        _validate_batch(snapshot)
    elif snapshot.batch is not None:
        raise PhaseProjectionError('non_batch_execution_has_batch_state')

    return next(iter(dependency_build_statuses), None)


def _validate_batch(snapshot: ExecutionPhaseSnapshot) -> None:
    assert snapshot.batch is not None
    batch = snapshot.batch
    if batch.generation_status not in _BATCH_GENERATION_STATUSES:
        raise PhaseProjectionError(f'unknown_batch_generation_status:{batch.generation_status}')
    if batch.control_state not in _BATCH_CONTROL_STATES:
        raise PhaseProjectionError(f'unknown_batch_control_state:{batch.control_state}')
    if batch.dependency_status not in _BATCH_DEPENDENCY_STATUSES:
        raise PhaseProjectionError(f'unknown_batch_dependency_status:{batch.dependency_status}')
    if batch.materialized_count < 0:
        raise PhaseProjectionError('negative_batch_materialized_count')
    if batch.total_items is not None and batch.total_items < 0:
        raise PhaseProjectionError('negative_batch_total_items')
    if batch.total_items is not None and batch.materialized_count > batch.total_items:
        raise PhaseProjectionError('batch_materialized_count_exceeds_total')
    if batch.generation_status == 'complete' and (
        batch.total_items is None or batch.materialized_count != batch.total_items
    ):
        raise PhaseProjectionError('complete_generation_count_mismatch')
    if batch.control_state == 'finalized' and snapshot.status not in TERMINAL_EXECUTION_STATUSES:
        raise PhaseProjectionError('non_terminal_execution_has_finalized_batch')


__all__ = [
    'AllocationPhaseState',
    'BatchPhaseState',
    'ExecutionPhaseSnapshot',
    'PhaseProjectionError',
    'project_execution_phase',
]
