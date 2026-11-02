from __future__ import annotations

from alembic.operations import Operations
from sqlalchemy import inspect
from sqlalchemy.dialects import mysql


revision = '0009_batch_item_bin_collation'
down_revision = '0008_batch_item_hot_index'


_BATCH_ITEM_ID_COLUMNS = (
    ('distributed_batch_items', 'item_id', False),
    ('distributed_batch_item_attempts', 'item_id', False),
    ('distributed_batch_runs', 'stop_cursor', True),
    ('distributed_batch_runs', 'child_sync_cursor', True),
)


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    if bind.dialect.name != 'mysql':
        return

    inspector = inspect(bind)
    table_names = set(inspector.get_table_names())
    target_type = mysql.VARCHAR(64, charset='utf8mb4', collation='utf8mb4_bin')
    for table_name, column_name, nullable in _BATCH_ITEM_ID_COLUMNS:
        if table_name not in table_names:
            continue
        columns = {column['name']: column for column in inspect(bind).get_columns(table_name)}
        column = columns.get(column_name)
        if column is None or getattr(column['type'], 'collation', None) == 'utf8mb4_bin':
            continue
        op.alter_column(
            table_name,
            column_name,
            existing_type=column['type'],
            type_=target_type,
            existing_nullable=nullable,
        )
