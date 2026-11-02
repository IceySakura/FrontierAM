"""Make Reconciler execution indexes shard-leading."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from alembic.operations import Operations
from sqlalchemy import inspect


revision = '0017_reconciler_shard_indexes'
down_revision = '0016_add_batch_run_shard'
branch_labels = None
depends_on = None


_SHARDED_INDEXES: Mapping[str, Sequence[str]] = {
    'idx_executions_status_updated': ('status', 'shard_id', 'updated_at', 'execution_id'),
    'idx_executions_route_status': ('route_status', 'status', 'shard_id', 'updated_at', 'execution_id'),
    'idx_executions_cleanup': ('status', 'shard_id', 'cleanup_done_at', 'execution_id'),
    'idx_executions_queue_deadline': ('status', 'shard_id', 'queue_expires_at', 'execution_id'),
    'idx_executions_execution_deadline': ('status', 'shard_id', 'execution_expires_at', 'execution_id'),
    'idx_executions_stop_deadline': ('status', 'shard_id', 'stop_deadline_at', 'execution_id'),
}
_LEGACY_INDEXES: Mapping[str, Sequence[str]] = {
    'idx_executions_status_updated': ('status', 'updated_at'),
    'idx_executions_route_status': ('route_status', 'status', 'updated_at'),
    'idx_executions_cleanup': ('status', 'cleanup_done_at'),
    'idx_executions_queue_deadline': ('status', 'queue_expires_at', 'execution_id'),
    'idx_executions_execution_deadline': ('status', 'execution_expires_at', 'execution_id'),
    'idx_executions_stop_deadline': ('status', 'stop_deadline_at', 'execution_id'),
}
_REDUNDANT_INDEXES: Mapping[str, tuple[str, Sequence[str]]] = {
    'ix_executions_current_assignment_id': ('executions', ('current_assignment_id',)),
    'ix_instance_managers_last_heartbeat_at': ('instance_managers', ('last_heartbeat_at',)),
}


def upgrade(op: Operations) -> None:
    _replace_indexes(op, table_name='executions', desired=_SHARDED_INDEXES)
    bind = op.get_bind()
    table_names = set(inspect(bind).get_table_names())
    for index_name, (table_name, _columns) in _REDUNDANT_INDEXES.items():
        if table_name not in table_names:
            continue
        indexes = {index['name'] for index in inspect(bind).get_indexes(table_name)}
        if index_name in indexes:
            op.drop_index(index_name, table_name=table_name)


def downgrade(op: Operations) -> None:
    _replace_indexes(op, table_name='executions', desired=_LEGACY_INDEXES)
    bind = op.get_bind()
    table_names = set(inspect(bind).get_table_names())
    for index_name, (table_name, columns) in _REDUNDANT_INDEXES.items():
        if table_name not in table_names:
            continue
        indexes = {index['name'] for index in inspect(bind).get_indexes(table_name)}
        if index_name not in indexes:
            op.create_index(index_name, table_name, list(columns), unique=False)


def _replace_indexes(
    op: Operations,
    *,
    table_name: str,
    desired: Mapping[str, Sequence[str]],
) -> None:
    bind = op.get_bind()
    if table_name not in set(inspect(bind).get_table_names()):
        return
    existing = {index['name']: tuple(index['column_names']) for index in inspect(bind).get_indexes(table_name)}
    for index_name, columns in desired.items():
        expected = tuple(columns)
        if existing.get(index_name) == expected:
            continue
        if index_name in existing:
            op.drop_index(index_name, table_name=table_name)
        op.create_index(index_name, table_name, list(expected), unique=False)
