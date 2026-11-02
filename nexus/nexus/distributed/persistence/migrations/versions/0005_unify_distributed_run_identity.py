from __future__ import annotations

from alembic.operations import Operations
from sqlalchemy import inspect


revision = '0005_unify_run_identity'
down_revision = '0004_cloud_submit_foundation'


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    table_name = 'distributed_batch_items'
    if table_name not in inspector.get_table_names():
        return
    if 'run_id' not in {column['name'] for column in inspector.get_columns(table_name)}:
        return

    constraint_name = 'uq_batch_items_run_id'
    unique_constraints = {constraint['name'] for constraint in inspector.get_unique_constraints(table_name)}
    indexes = {index['name'] for index in inspector.get_indexes(table_name)}
    with op.batch_alter_table(table_name) as batch_op:
        if constraint_name in unique_constraints:
            batch_op.drop_constraint(constraint_name, type_='unique')
        elif constraint_name in indexes:
            batch_op.drop_index(constraint_name)
        batch_op.drop_column('run_id')
