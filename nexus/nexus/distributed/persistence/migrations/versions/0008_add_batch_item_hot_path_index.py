from __future__ import annotations

from alembic.operations import Operations
from sqlalchemy import inspect


revision = '0008_batch_item_hot_index'
down_revision = '0007_add_batch_lifecycle_claim'


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    table_name = 'distributed_batch_items'
    inspector = inspect(bind)
    if table_name not in inspector.get_table_names():
        return

    index_name = 'idx_batch_items_parent_status_ordinal'
    existing_indexes = {index['name'] for index in inspector.get_indexes(table_name)}
    if index_name not in existing_indexes:
        op.create_index(
            index_name,
            table_name,
            ['parent_execution_id', 'status', 'task_ordinal'],
            unique=False,
        )
