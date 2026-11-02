from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations
from sqlalchemy import inspect


revision = '0006_add_execution_deadlines'
down_revision = '0005_unify_run_identity'


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    table_name = 'executions'
    if table_name not in inspector.get_table_names():
        return

    existing_columns = {column['name'] for column in inspector.get_columns(table_name)}
    for column_name in ('queue_expires_at', 'execution_expires_at', 'stop_deadline_at'):
        if column_name not in existing_columns:
            op.add_column(table_name, sa.Column(column_name, sa.DateTime(), nullable=True))

    existing_indexes = {index['name'] for index in inspect(bind).get_indexes(table_name)}
    indexes = {
        'idx_executions_queue_deadline': ['status', 'queue_expires_at', 'execution_id'],
        'idx_executions_execution_deadline': ['status', 'execution_expires_at', 'execution_id'],
        'idx_executions_stop_deadline': ['status', 'stop_deadline_at', 'execution_id'],
    }
    for index_name, columns in indexes.items():
        if index_name not in existing_indexes:
            op.create_index(index_name, table_name, columns, unique=False)
