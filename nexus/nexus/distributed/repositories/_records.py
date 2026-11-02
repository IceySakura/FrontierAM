from __future__ import annotations

from copy import deepcopy

from nexus.distributed.execution.models import (
    AssignmentRecord,
    CommandRecord,
    EventRecord,
    ExecutionRecord,
)
from nexus.distributed.execution.runtime_info import normalize_runtime_info
from nexus.distributed.instance_manager.liveness import effective_manager_liveness
from nexus.distributed.instance_manager.models import ManagerRecord
from nexus.distributed.persistence.orm import (
    AssignmentRow,
    ExecutionCommandRow,
    ExecutionEventRow,
    ExecutionRow,
    InstanceManagerLivenessRow,
    InstanceManagerRow,
)


def assignment_record(row: AssignmentRow | None) -> AssignmentRecord | None:
    if row is None:
        return None
    return AssignmentRecord(
        assignment_id=row.assignment_id,
        execution_id=row.execution_id,
        attempt_id=row.attempt_id,
        attempt_seq=row.attempt_seq,
        manager_id=row.manager_id,
        manager_session_id=row.manager_session_id,
        status=row.status,
        slot_cost=row.slot_cost,
        created_at=row.created_at,
        updated_at=row.updated_at,
        endpoint_url=row.endpoint_url,
        endpoint_ready=row.endpoint_ready,
        runtime_info=deepcopy(normalize_runtime_info(row.runtime_info_json)),
        owner_principal_id=row.owner_principal_id,
        environment_id=row.environment_id,
        pool_id=row.pool_id,
        allocation_id=row.allocation_id,
        workload=row.workload,
        allocation_spec_hash=row.allocation_spec_hash,
    )


def execution_record(row: ExecutionRow, current_assignment: AssignmentRow | None = None) -> ExecutionRecord:
    return ExecutionRecord(
        execution_id=row.execution_id,
        shard_id=row.shard_id,
        mode=row.mode,
        spec_type=row.spec_type,
        instance_mode=row.instance_mode,
        spec=deepcopy(row.spec_json),
        driver_config=deepcopy(row.driver_config_json or {}),
        placement=deepcopy(row.placement_json or {}),
        status=row.status,
        route_status=row.route_status,
        slot_cost=row.slot_cost,
        current_assignment_id=row.current_assignment_id,
        current_attempt_id=row.current_attempt_id,
        attempt_count=row.attempt_count,
        max_attempts=row.max_attempts,
        next_schedule_at=row.next_schedule_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
        stop_timeout_seconds=row.stop_timeout_seconds,
        cleanup_done_at=row.cleanup_done_at,
        result_json=deepcopy(row.result_json),
        error_json=deepcopy(row.error_json),
        current_assignment=assignment_record(current_assignment),
        owner_principal_id=row.owner_principal_id,
        owner_tenant=row.owner_tenant,
        created_by=row.created_by,
        parent_execution_id=row.parent_execution_id,
        spec_sha256=row.spec_sha256,
        environment_id=row.environment_id,
        pool_id=row.pool_id,
        allocation_id=row.allocation_id,
        workload=row.workload,
        submission_sha256=row.submission_sha256,
        allocation_spec_hash=row.allocation_spec_hash,
        result_artifact_id=row.result_artifact_id,
        first_schedulable_at=row.first_schedulable_at,
        stop_reason=row.stop_reason,
        queue_expires_at=row.queue_expires_at,
        execution_expires_at=row.execution_expires_at,
        stop_deadline_at=row.stop_deadline_at,
        version=row.version,
    )


def command_record(row: ExecutionCommandRow) -> CommandRecord:
    return CommandRecord(
        command_id=row.command_id,
        execution_id=row.execution_id,
        assignment_id=row.assignment_id,
        attempt_id=row.attempt_id,
        type=row.type,
        status=row.status,
        payload=deepcopy(row.payload_json or {}),
        created_at=row.created_at,
        updated_at=row.updated_at,
        idempotency_key=row.idempotency_key,
    )


def manager_record(
    row: InstanceManagerRow,
    liveness: InstanceManagerLivenessRow | None = None,
) -> ManagerRecord:
    effective = effective_manager_liveness(row, liveness)
    return ManagerRecord(
        manager_id=row.manager_id,
        session_id=row.session_id,
        endpoint_url=row.endpoint_url,
        environment_id=row.environment_id,
        pool_id=row.pool_id,
        workload=row.workload,
        labels=deepcopy(row.labels_json or {}),
        metadata=deepcopy(row.metadata_json or {}),
        draining=row.draining,
        supported_modes=deepcopy(row.supported_modes_json),
        supported_spec_types=deepcopy(row.supported_spec_types_json),
        supported_instance_modes=deepcopy(row.supported_instance_modes_json),
        capacity_total=effective.capacity_total,
        observed_capacity_used=effective.observed_capacity_used,
        last_heartbeat_at=effective.last_heartbeat_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
        bootstrap_version=row.bootstrap_version,
        artifact_ref_version=row.artifact_ref_version,
        supported_spec_versions=deepcopy(row.supported_spec_versions_json or {}),
        owner_principal_id=row.owner_principal_id,
        allocation_id=row.allocation_id,
        allocation_spec_hash=row.allocation_spec_hash,
        capability_snapshot=deepcopy(row.capability_snapshot_json or {}),
    )


def event_record(row: ExecutionEventRow) -> EventRecord:
    return EventRecord(
        event_id=row.event_id,
        execution_id=row.execution_id,
        seq=row.seq,
        assignment_id=row.assignment_id,
        attempt_id=row.attempt_id,
        type=row.type,
        source=row.source,
        visibility=row.visibility,
        payload=deepcopy(row.payload_json),
        event_key=row.event_key,
        created_at=row.created_at,
    )
