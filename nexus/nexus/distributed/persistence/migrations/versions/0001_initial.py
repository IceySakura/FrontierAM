from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations


revision = '0001_initial'
down_revision = None


def upgrade(op: Operations) -> None:
    op.create_table(
        'schema_version',
        sa.Column('version', sa.Integer(), primary_key=True),
    )
    op.execute(sa.text('INSERT INTO schema_version (version) VALUES (1)'))

    op.create_table(
        'executions',
        sa.Column('execution_id', sa.String(64), primary_key=True),
        sa.Column('idempotency_scope', sa.String(128), nullable=True),
        sa.Column('idempotency_key', sa.String(128), nullable=True),
        sa.Column('shard_id', sa.Integer(), nullable=False),
        sa.Column('mode', sa.String(16), nullable=False),
        sa.Column('spec_type', sa.String(64), nullable=False),
        sa.Column('instance_mode', sa.String(32), nullable=False),
        sa.Column('spec_json', sa.JSON(), nullable=False),
        sa.Column('driver_config_json', sa.JSON(), nullable=False),
        sa.Column('placement_json', sa.JSON(), nullable=False),
        sa.Column('status', sa.String(32), nullable=False),
        sa.Column('route_status', sa.String(32), nullable=False),
        sa.Column('slot_cost', sa.Integer(), nullable=False),
        sa.Column('current_assignment_id', sa.String(64), nullable=True),
        sa.Column('current_attempt_id', sa.String(64), nullable=True),
        sa.Column('next_attempt_seq', sa.Integer(), nullable=False),
        sa.Column('attempt_count', sa.Integer(), nullable=False),
        sa.Column('max_attempts', sa.Integer(), nullable=False),
        sa.Column('next_event_seq', sa.Integer(), nullable=False),
        sa.Column('priority', sa.Integer(), nullable=False),
        sa.Column('next_schedule_at', sa.DateTime(), nullable=False),
        sa.Column('last_schedule_error', sa.Text(), nullable=True),
        sa.Column('queue_ttl_seconds', sa.Integer(), nullable=True),
        sa.Column('execution_ttl_seconds', sa.Integer(), nullable=True),
        sa.Column('stop_timeout_seconds', sa.Integer(), nullable=True),
        sa.Column('result_json', sa.JSON(), nullable=True),
        sa.Column('error_json', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('finished_at', sa.DateTime(), nullable=True),
        sa.UniqueConstraint('idempotency_scope', 'idempotency_key', name='uq_executions_idempotency'),
    )
    op.create_index('ix_executions_shard_id', 'executions', ['shard_id'])
    op.create_index('ix_executions_status', 'executions', ['status'])
    op.create_index('ix_executions_route_status', 'executions', ['route_status'])
    op.create_index('ix_executions_current_assignment_id', 'executions', ['current_assignment_id'])
    op.create_index('ix_executions_next_schedule_at', 'executions', ['next_schedule_at'])
    op.create_index(
        'idx_executions_sched',
        'executions',
        ['status', 'shard_id', 'next_schedule_at', 'priority', 'created_at'],
    )
    op.create_index('idx_executions_status_updated', 'executions', ['status', 'updated_at'])
    op.create_index('idx_executions_route_status', 'executions', ['route_status', 'status', 'updated_at'])
    op.create_index('idx_executions_current_assignment', 'executions', ['current_assignment_id'])

    op.create_table(
        'assignments',
        sa.Column('assignment_id', sa.String(64), primary_key=True),
        sa.Column('execution_id', sa.String(64), nullable=False),
        sa.Column('attempt_id', sa.String(64), nullable=False),
        sa.Column('attempt_seq', sa.Integer(), nullable=False),
        sa.Column('manager_id', sa.String(128), nullable=False),
        sa.Column('manager_session_id', sa.String(64), nullable=False),
        sa.Column('status', sa.String(32), nullable=False),
        sa.Column('slot_cost', sa.Integer(), nullable=False),
        sa.Column('endpoint_url', sa.String(2048), nullable=True),
        sa.Column('endpoint_ready', sa.Boolean(), nullable=False),
        sa.Column('endpoint_updated_at', sa.DateTime(), nullable=True),
        sa.Column('endpoint_error', sa.Text(), nullable=True),
        sa.Column('runtime_info_json', sa.JSON(), nullable=True),
        sa.Column('last_instance_update_at', sa.DateTime(), nullable=True),
        sa.Column('accepted_at', sa.DateTime(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('finished_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.UniqueConstraint('attempt_id'),
    )
    op.create_index('ix_assignments_execution_id', 'assignments', ['execution_id'])
    op.create_index('ix_assignments_manager_id', 'assignments', ['manager_id'])
    op.create_index('ix_assignments_status', 'assignments', ['status'])
    op.create_index('idx_assignments_manager_status', 'assignments', ['manager_id', 'status', 'updated_at'])
    op.create_index('idx_assignments_execution_status', 'assignments', ['execution_id', 'status', 'updated_at'])
    op.create_index('idx_assignments_status_updated', 'assignments', ['status', 'updated_at'])

    op.create_table(
        'instance_managers',
        sa.Column('manager_id', sa.String(128), primary_key=True),
        sa.Column('session_id', sa.String(64), nullable=False),
        sa.Column('endpoint_url', sa.String(2048), nullable=False),
        sa.Column('environment_type', sa.String(32), nullable=False),
        sa.Column('environment_name', sa.String(128), nullable=False),
        sa.Column('pool_name', sa.String(128), nullable=False),
        sa.Column('labels_json', sa.JSON(), nullable=False),
        sa.Column('draining', sa.Boolean(), nullable=False),
        sa.Column('supported_modes_json', sa.JSON(), nullable=False),
        sa.Column('supported_spec_types_json', sa.JSON(), nullable=False),
        sa.Column('supported_instance_modes_json', sa.JSON(), nullable=False),
        sa.Column('capacity_total', sa.Integer(), nullable=False),
        sa.Column('observed_capacity_used', sa.Integer(), nullable=False),
        sa.Column('local_instances_json', sa.JSON(), nullable=True),
        sa.Column('version', sa.String(64), nullable=True),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
        sa.Column('last_heartbeat_at', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_instance_managers_last_heartbeat_at', 'instance_managers', ['last_heartbeat_at'])
    op.create_index('idx_instance_managers_heartbeat', 'instance_managers', ['last_heartbeat_at'])
    op.create_index('idx_instance_managers_capacity', 'instance_managers', ['capacity_total'])
    op.create_index(
        'idx_instance_managers_environment',
        'instance_managers',
        ['environment_name', 'pool_name', 'last_heartbeat_at'],
    )
    op.create_index('idx_instance_managers_draining', 'instance_managers', ['draining', 'last_heartbeat_at'])

    op.create_table(
        'execution_commands',
        sa.Column('command_id', sa.String(64), primary_key=True),
        sa.Column('execution_id', sa.String(64), nullable=False),
        sa.Column('assignment_id', sa.String(64), nullable=True),
        sa.Column('attempt_id', sa.String(64), nullable=True),
        sa.Column('type', sa.String(64), nullable=False),
        sa.Column('status', sa.String(32), nullable=False),
        sa.Column('payload_json', sa.JSON(), nullable=False),
        sa.Column('idempotency_key', sa.String(128), nullable=True),
        sa.Column('created_by', sa.String(128), nullable=True),
        sa.Column('claimed_by_manager_id', sa.String(128), nullable=True),
        sa.Column('claimed_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('error_json', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.UniqueConstraint('execution_id', 'idempotency_key', name='uq_execution_commands_idempotency'),
    )
    op.create_index('ix_execution_commands_execution_id', 'execution_commands', ['execution_id'])
    op.create_index('ix_execution_commands_assignment_id', 'execution_commands', ['assignment_id'])
    op.create_index('ix_execution_commands_status', 'execution_commands', ['status'])
    op.create_index(
        'idx_execution_commands_assignment_pending',
        'execution_commands',
        ['assignment_id', 'status', 'created_at'],
    )
    op.create_index('idx_execution_commands_execution', 'execution_commands', ['execution_id', 'created_at'])
    op.create_index('idx_execution_commands_status_updated', 'execution_commands', ['status', 'updated_at'])

    op.create_table(
        'execution_events',
        sa.Column('event_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('execution_id', sa.String(64), nullable=False),
        sa.Column('seq', sa.Integer(), nullable=False),
        sa.Column('assignment_id', sa.String(64), nullable=True),
        sa.Column('attempt_id', sa.String(64), nullable=True),
        sa.Column('type', sa.String(64), nullable=False),
        sa.Column('source', sa.String(64), nullable=False),
        sa.Column('visibility', sa.String(16), nullable=False),
        sa.Column('payload_json', sa.JSON(), nullable=False),
        sa.Column('event_key', sa.String(128), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.UniqueConstraint('execution_id', 'seq', name='uq_execution_events_execution_seq'),
        sa.UniqueConstraint('execution_id', 'event_key', name='uq_execution_events_event_key'),
    )
    op.create_index('ix_execution_events_execution_id', 'execution_events', ['execution_id'])
    op.create_index('ix_execution_events_attempt_id', 'execution_events', ['attempt_id'])
    op.create_index('idx_execution_events_created', 'execution_events', ['created_at'])
    op.create_index('idx_execution_events_attempt', 'execution_events', ['attempt_id', 'seq'])

    op.create_table(
        'scheduler_leases',
        sa.Column('shard_group', sa.String(64), primary_key=True),
        sa.Column('shard_id', sa.Integer(), primary_key=True),
        sa.Column('shard_count', sa.Integer(), nullable=False),
        sa.Column('owner_id', sa.String(128), nullable=False),
        sa.Column('lease_expires_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_scheduler_leases_lease_expires_at', 'scheduler_leases', ['lease_expires_at'])
    op.create_index('idx_scheduler_leases_expires', 'scheduler_leases', ['lease_expires_at'])
