"""Keep logical task names and lifecycle times on the existing task tables."""

from __future__ import annotations

from alembic.operations import Operations
from sqlalchemy import Column, DateTime, Text, inspect
from sqlalchemy.dialects import mysql


revision = '0033_business_statistics'
down_revision = '0032_observability_snapshots'
branch_labels = None
depends_on = None

_COLUMNS = {
    'executions': ('task_name',),
    'distributed_batch_items': ('task_name', 'first_started_at', 'finished_at'),
}
_INDEXES = (
    ('executions', 'idx_exec_stats_created', ['parent_execution_id', 'created_at', 'execution_id']),
    ('executions', 'idx_exec_stats_started', ['parent_execution_id', 'started_at', 'execution_id']),
    ('executions', 'idx_exec_stats_finished', ['parent_execution_id', 'finished_at', 'execution_id']),
    ('distributed_batch_items', 'idx_batch_stats_created', ['created_at', 'parent_execution_id', 'item_id']),
    ('distributed_batch_items', 'idx_batch_stats_started', ['first_started_at', 'parent_execution_id', 'item_id']),
    ('distributed_batch_items', 'idx_batch_stats_finished', ['finished_at', 'parent_execution_id', 'item_id']),
)


def upgrade(op: Operations) -> None:
    for table, names in _COLUMNS.items():
        columns = {column['name'] for column in inspect(op.get_bind()).get_columns(table)}
        for name in names:
            if name not in columns:
                kind = (
                    Text().with_variant(mysql.LONGTEXT(collation='utf8mb4_bin'), 'mysql')
                    if name == 'task_name'
                    else DateTime()
                )
                op.add_column(table, Column(name, kind, nullable=True))
    for table, name, columns in _INDEXES:
        if name not in {index['name'] for index in inspect(op.get_bind()).get_indexes(table)}:
            op.create_index(name, table, columns)


def downgrade(op: Operations) -> None:
    for table, name, _ in reversed(_INDEXES):
        if name in {index['name'] for index in inspect(op.get_bind()).get_indexes(table)}:
            op.drop_index(name, table_name=table)
    for table, names in _COLUMNS.items():
        existing = {column['name'] for column in inspect(op.get_bind()).get_columns(table)}
        for name in reversed(names):
            if name in existing:
                op.drop_column(table, name)
