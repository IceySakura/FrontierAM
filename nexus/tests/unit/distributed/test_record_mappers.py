from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from nexus.distributed.execution.models import (
    AssignmentRecord,
    CommandRecord,
    EventRecord,
    ExecutionRecord,
)
from nexus.distributed.instance_manager.models import ManagerRecord
from nexus.distributed.persistence.orm import (
    AssignmentRow,
    ExecutionCommandRow,
    ExecutionEventRow,
    ExecutionRow,
    InstanceManagerRow,
)
from nexus.distributed.repositories._records import (
    assignment_record,
    command_record,
    event_record,
    execution_record,
    manager_record,
)
from nexus.distributed.utils.clock import utcnow


_RECORD_FIELDS = {
    AssignmentRecord: {
        'assignment_id',
        'execution_id',
        'attempt_id',
        'attempt_seq',
        'manager_id',
        'manager_session_id',
        'status',
        'slot_cost',
        'created_at',
        'updated_at',
        'endpoint_url',
        'endpoint_ready',
        'runtime_info',
        'owner_principal_id',
        'environment_id',
        'pool_id',
        'allocation_id',
        'workload',
        'allocation_spec_hash',
    },
    ExecutionRecord: {
        'execution_id',
        'shard_id',
        'mode',
        'spec_type',
        'instance_mode',
        'spec',
        'driver_config',
        'placement',
        'status',
        'route_status',
        'slot_cost',
        'current_assignment_id',
        'current_attempt_id',
        'attempt_count',
        'max_attempts',
        'next_schedule_at',
        'created_at',
        'updated_at',
        'stop_timeout_seconds',
        'cleanup_done_at',
        'result_json',
        'error_json',
        'current_assignment',
        'owner_principal_id',
        'owner_tenant',
        'created_by',
        'parent_execution_id',
        'spec_sha256',
        'environment_id',
        'pool_id',
        'allocation_id',
        'workload',
        'submission_sha256',
        'allocation_spec_hash',
        'result_artifact_id',
        'first_schedulable_at',
        'stop_reason',
        'queue_expires_at',
        'execution_expires_at',
        'stop_deadline_at',
        'version',
    },
    CommandRecord: {
        'command_id',
        'execution_id',
        'assignment_id',
        'attempt_id',
        'type',
        'status',
        'payload',
        'created_at',
        'updated_at',
        'idempotency_key',
    },
    ManagerRecord: {
        'manager_id',
        'session_id',
        'endpoint_url',
        'environment_id',
        'pool_id',
        'workload',
        'labels',
        'metadata',
        'draining',
        'supported_modes',
        'supported_spec_types',
        'supported_instance_modes',
        'capacity_total',
        'observed_capacity_used',
        'last_heartbeat_at',
        'created_at',
        'updated_at',
        'bootstrap_version',
        'artifact_ref_version',
        'supported_spec_versions',
        'owner_principal_id',
        'allocation_id',
        'allocation_spec_hash',
        'capability_snapshot',
    },
    EventRecord: {
        'event_id',
        'execution_id',
        'seq',
        'assignment_id',
        'attempt_id',
        'type',
        'source',
        'visibility',
        'payload',
        'event_key',
        'created_at',
    },
}


@pytest.mark.parametrize(('record_type', 'expected'), _RECORD_FIELDS.items())
def test_record_mapper_contract_lists_every_record_field(record_type, expected):
    assert {field.name for field in fields(record_type)} == expected


def test_record_mappers_return_frozen_deep_copied_snapshots():
    now = utcnow()
    assignment_row = AssignmentRow(
        assignment_id='assignment-1',
        execution_id='execution-1',
        owner_principal_id='owner-1',
        environment_id='environment-1',
        pool_id='execution',
        allocation_id='allocation-1',
        workload='execution',
        allocation_spec_hash='a' * 64,
        attempt_id='attempt-1',
        attempt_seq=1,
        manager_id='manager-1',
        manager_session_id='session-1',
        status='active',
        slot_cost=1,
        endpoint_url='http://manager-1',
        endpoint_ready=True,
        runtime_info_json={'runtimes': {'default': {'nested': [1]}}},
        created_at=now,
        updated_at=now,
    )
    execution_row = ExecutionRow(
        execution_id='execution-1',
        owner_principal_id='owner-1',
        owner_tenant='tenant-1',
        created_by='owner-1',
        parent_execution_id='batch-parent-1',
        spec_sha256='b' * 64,
        environment_id='environment-1',
        pool_id='execution',
        allocation_id='allocation-1',
        workload='execution',
        submission_sha256='c' * 64,
        allocation_spec_hash='a' * 64,
        result_artifact_id='artifact-1',
        first_schedulable_at=now,
        stop_reason=None,
        version=3,
        shard_id=1,
        mode='job',
        spec_type='nexus_single',
        instance_mode='asyncio',
        spec_json={'nested': {'values': [1]}},
        driver_config_json={'nested': {'values': [2]}},
        placement_json={'nested': {'values': [3]}},
        status='running',
        route_status='none',
        slot_cost=1,
        current_assignment_id='assignment-1',
        current_attempt_id='attempt-1',
        attempt_count=1,
        max_attempts=2,
        next_schedule_at=now,
        stop_timeout_seconds=30,
        result_json={'nested': {'values': [4]}},
        error_json={'nested': {'values': [5]}},
        created_at=now,
        updated_at=now,
    )
    manager_row = InstanceManagerRow(
        manager_id='manager-1',
        session_id='session-1',
        endpoint_url='http://manager-1',
        labels_json={'nested': {'value': 'label'}},
        owner_principal_id='owner-1',
        environment_id='environment-1',
        pool_id='execution',
        allocation_id='allocation-1',
        workload='execution',
        allocation_spec_hash='a' * 64,
        capability_snapshot_json={'nested': {'values': [1]}},
        bootstrap_version=1,
        artifact_ref_version=1,
        supported_spec_versions_json={'nexus_single': [1]},
        supported_modes_json=['job'],
        supported_spec_types_json=['nexus_single'],
        supported_instance_modes_json=['asyncio'],
        capacity_total=1,
        observed_capacity_used=0,
        metadata_json={'nested': {'values': [2]}},
        last_heartbeat_at=now,
        created_at=now,
        updated_at=now,
    )
    command_row = ExecutionCommandRow(
        command_id='command-1',
        execution_id='execution-1',
        assignment_id='assignment-1',
        attempt_id='attempt-1',
        type='stop',
        status='pending',
        payload_json={'nested': {'values': [1]}},
        created_at=now,
        updated_at=now,
    )
    event_row = ExecutionEventRow(
        event_id=1,
        execution_id='execution-1',
        seq=1,
        assignment_id='assignment-1',
        attempt_id='attempt-1',
        type='execution.created',
        source='control',
        visibility='public',
        payload_json={'nested': {'values': [1]}},
        created_at=now,
    )

    assignment = assignment_record(assignment_row)
    assert assignment is not None
    execution = execution_record(execution_row, assignment_row)
    manager = manager_record(manager_row)
    command = command_record(command_row)
    event = event_record(event_row)

    assignment_row.runtime_info_json['runtimes']['default']['nested'].append(2)
    execution_row.spec_json['nested']['values'].append(2)
    execution_row.driver_config_json['nested']['values'].append(3)
    execution_row.placement_json['nested']['values'].append(4)
    execution_row.result_json['nested']['values'].append(5)
    execution_row.error_json['nested']['values'].append(6)
    manager_row.labels_json['nested']['value'] = 'changed'
    manager_row.metadata_json['nested']['values'].append(3)
    manager_row.supported_spec_versions_json['nexus_single'].append(2)
    manager_row.capability_snapshot_json['nested']['values'].append(2)
    command_row.payload_json['nested']['values'].append(2)
    event_row.payload_json['nested']['values'].append(2)

    assert assignment.runtime_info['runtimes']['default']['nested'] == [1]
    assert execution.spec['nested']['values'] == [1]
    assert execution.driver_config['nested']['values'] == [2]
    assert execution.placement['nested']['values'] == [3]
    assert execution.result_json['nested']['values'] == [4]
    assert execution.error_json['nested']['values'] == [5]
    assert execution.parent_execution_id == 'batch-parent-1'
    assert execution.current_assignment == assignment
    assert manager.labels['nested']['value'] == 'label'
    assert manager.metadata['nested']['values'] == [2]
    assert manager.supported_spec_versions['nexus_single'] == [1]
    assert manager.capability_snapshot['nested']['values'] == [1]
    assert command.payload['nested']['values'] == [1]
    assert event.payload['nested']['values'] == [1]

    with pytest.raises(FrozenInstanceError):
        execution.status = 'failed'
