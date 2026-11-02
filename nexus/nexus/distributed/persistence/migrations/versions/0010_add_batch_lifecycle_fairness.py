from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations
from sqlalchemy import inspect


revision = '0010_batch_lifecycle_fair'
down_revision = '0009_batch_item_bin_collation'


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    table_name = 'distributed_batch_runs'
    inspector = inspect(bind)
    if table_name not in inspector.get_table_names():
        return

    columns = {column['name'] for column in inspector.get_columns(table_name)}
    if 'last_reconciled_at' not in columns:
        op.add_column(table_name, sa.Column('last_reconciled_at', sa.DateTime(), nullable=True))

    inspector = inspect(bind)
    indexes = {index['name']: index for index in inspector.get_indexes(table_name)}
    index_name = 'idx_batch_runs_lifecycle_claim'
    expected_columns = [
        'control_state',
        'last_reconciled_at',
        'lifecycle_lease_expires_at',
        'parent_execution_id',
    ]
    existing = indexes.get(index_name)
    if existing is not None and list(existing.get('column_names') or ()) != expected_columns:
        op.drop_index(index_name, table_name=table_name)
        existing = None
    if existing is None:
        op.create_index(index_name, table_name, expected_columns, unique=False)
