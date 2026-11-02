"""Persist complete status groups independently of HTTP requests."""

from __future__ import annotations

from alembic.operations import Operations
from sqlalchemy import JSON, BigInteger, Column, DateTime, String, inspect


revision = '0032_observability_snapshots'
down_revision = '0031_build_list_index'
branch_labels = None
depends_on = None

_TABLE = 'distributed_observability_snapshots'
_TIME_COLUMNS = {
    'executions': ('queue_entered_at',),
    'distributed_dependency_builds': ('deadline_at',),
    'distributed_environment_allocations': (
        'provisioning_started_at',
        'provisioning_deadline_at',
        'release_requested_at',
    ),
    'distributed_batch_runs': ('preparation_started_at', 'preparation_deadline_at'),
}
_INDEXES = (
    (
        'distributed_environment_resource_usage',
        'idx_resource_usage_scope_environment',
        ['scope_type', 'environment_id'],
    ),
    ('instance_managers', 'idx_managers_env_heartbeat', ['environment_id', 'last_heartbeat_at', 'manager_id']),
    (
        'assignments',
        'idx_assignments_manager_session_status',
        ['manager_id', 'manager_session_id', 'status', 'slot_cost'],
    ),
    ('executions', 'idx_executions_env_status_schedule', ['environment_id', 'status', 'next_schedule_at']),
    ('execution_commands', 'idx_commands_status_created', ['status', 'created_at']),
)


def upgrade(op: Operations) -> None:
    inspector = inspect(op.get_bind())
    for table, names in _TIME_COLUMNS.items():
        existing = {column['name'] for column in inspector.get_columns(table)}
        for name in names:
            if name not in existing:
                op.add_column(table, Column(name, DateTime, nullable=True))
    if _TABLE not in inspector.get_table_names():
        op.create_table(
            _TABLE,
            Column('scope_type', String(16), primary_key=True),
            Column('scope_id', String(128), primary_key=True),
            Column('metric_group', String(32), primary_key=True),
            Column('values_json', JSON, nullable=True),
            Column('sampled_at', DateTime, nullable=True),
            Column('last_attempt_at', DateTime, nullable=False),
            Column('collection_status', String(16), nullable=False),
            Column('last_error_code', String(64), nullable=True),
            Column('version', BigInteger, nullable=False),
            Column('publisher_id', String(128), nullable=False),
        )
    for table, name, columns in _INDEXES:
        if name not in {item['name'] for item in inspect(op.get_bind()).get_indexes(table)}:
            op.create_index(name, table, columns, unique=False)


def downgrade(op: Operations) -> None:
    for table, name, _columns in reversed(_INDEXES):
        if name in {item['name'] for item in inspect(op.get_bind()).get_indexes(table)}:
            op.drop_index(name, table_name=table)
    if _TABLE in inspect(op.get_bind()).get_table_names():
        op.drop_table(_TABLE)
    for table, names in reversed(tuple(_TIME_COLUMNS.items())):
        existing = {column['name'] for column in inspect(op.get_bind()).get_columns(table)}
        for name in reversed(names):
            if name in existing:
                op.drop_column(table, name)
