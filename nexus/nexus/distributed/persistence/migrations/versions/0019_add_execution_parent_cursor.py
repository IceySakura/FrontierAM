"""Add explicit Batch child ownership and Console cursor indexes."""

from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations
from sqlalchemy import inspect


revision = '0019_add_execution_parent_cursor'
down_revision = '0018_precreate_resource_usage'
branch_labels = None
depends_on = None


def upgrade(op: Operations) -> None:
    columns = {column['name'] for column in inspect(op.get_bind()).get_columns('executions')}
    if 'parent_execution_id' not in columns:
        op.add_column('executions', sa.Column('parent_execution_id', sa.String(length=64), nullable=True))

    executions = sa.table(
        'executions',
        sa.column('execution_id', sa.String()),
        sa.column('created_by', sa.String()),
        sa.column('parent_execution_id', sa.String()),
    )
    attempts = sa.table(
        'distributed_batch_item_attempts',
        sa.column('parent_execution_id', sa.String()),
        sa.column('child_execution_id', sa.String()),
    )
    parent_lookup = (
        sa
        .select(attempts.c.parent_execution_id)
        .where(attempts.c.child_execution_id == executions.c.execution_id)
        .limit(1)
        .scalar_subquery()
    )
    op.execute(
        sa
        .update(executions)
        .where(executions.c.created_by == 'system:batch-parent')
        .values(parent_execution_id=parent_lookup)
    )

    _create_index_if_missing(
        op,
        'idx_executions_parent_updated',
        'executions',
        ['parent_execution_id', 'updated_at', 'created_at', 'execution_id'],
    )
    _create_index_if_missing(
        op,
        'idx_executions_owner_parent_updated',
        'executions',
        ['owner_principal_id', 'parent_execution_id', 'updated_at', 'created_at', 'execution_id'],
    )
    _create_index_if_missing(
        op,
        'idx_executions_spec_parent_updated',
        'executions',
        ['spec_type', 'parent_execution_id', 'updated_at', 'created_at', 'execution_id'],
    )
    _create_index_if_missing(
        op,
        'idx_batch_item_attempts_parent_error_item',
        'distributed_batch_item_attempts',
        ['parent_execution_id', 'error_code', 'item_id'],
    )


def downgrade(op: Operations) -> None:
    _drop_index_if_present(op, 'idx_batch_item_attempts_parent_error_item', 'distributed_batch_item_attempts')
    _drop_index_if_present(op, 'idx_executions_spec_parent_updated', 'executions')
    _drop_index_if_present(op, 'idx_executions_owner_parent_updated', 'executions')
    _drop_index_if_present(op, 'idx_executions_parent_updated', 'executions')
    columns = {column['name'] for column in inspect(op.get_bind()).get_columns('executions')}
    if 'parent_execution_id' in columns:
        op.drop_column('executions', 'parent_execution_id')


def _create_index_if_missing(op: Operations, name: str, table_name: str, columns: list[str]) -> None:
    indexes = {index['name'] for index in inspect(op.get_bind()).get_indexes(table_name)}
    if name not in indexes:
        op.create_index(name, table_name, columns)


def _drop_index_if_present(op: Operations, name: str, table_name: str) -> None:
    indexes = {index['name'] for index in inspect(op.get_bind()).get_indexes(table_name)}
    if name in indexes:
        op.drop_index(name, table_name=table_name)
