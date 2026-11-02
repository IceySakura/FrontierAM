from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations
from sqlalchemy import inspect


revision = '0007_add_batch_lifecycle_claim'
down_revision = '0006_add_execution_deadlines'


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    table_name = 'distributed_batch_runs'
    if table_name not in inspector.get_table_names():
        return

    existing_columns = {column['name'] for column in inspector.get_columns(table_name)}
    columns = (
        sa.Column('child_sync_cursor', sa.String(length=64), nullable=True),
        sa.Column('lifecycle_owner_id', sa.String(length=128), nullable=True),
        sa.Column('lifecycle_operation_id', sa.String(length=64), nullable=True),
        sa.Column('lifecycle_lease_expires_at', sa.DateTime(), nullable=True),
    )
    for column in columns:
        if column.name not in existing_columns:
            op.add_column(table_name, column)

    index_name = 'idx_batch_runs_lifecycle_claim'
    existing_indexes = {index['name'] for index in inspect(bind).get_indexes(table_name)}
    if index_name not in existing_indexes:
        op.create_index(
            index_name,
            table_name,
            ['control_state', 'lifecycle_lease_expires_at', 'parent_execution_id'],
            unique=False,
        )
