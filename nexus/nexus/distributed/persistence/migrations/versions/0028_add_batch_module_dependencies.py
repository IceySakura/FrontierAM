"""Add Batch module dependency resolution state."""

from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations


revision = '0028_batch_module_dependencies'
down_revision = '0027_trace_database_accounts'
branch_labels = None
depends_on = None


_TABLE = 'distributed_batch_runs'


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    columns = {column['name'] for column in sa.inspect(bind).get_columns(_TABLE)}
    additions = (
        sa.Column(
            'dependency_status',
            sa.String(length=32),
            nullable=False,
            server_default='not_required',
        ),
        sa.Column('module_dependencies_json', sa.JSON(), nullable=True),
        sa.Column('child_dependency_build_id', sa.String(length=64), nullable=True),
        sa.Column('child_template_json', sa.JSON(), nullable=True),
    )
    for column in additions:
        if column.name not in columns:
            op.add_column(_TABLE, column)


def downgrade(op: Operations) -> None:
    bind = op.get_bind()
    columns = {column['name'] for column in sa.inspect(bind).get_columns(_TABLE)}
    for column_name in (
        'child_template_json',
        'child_dependency_build_id',
        'module_dependencies_json',
        'dependency_status',
    ):
        if column_name in columns:
            op.drop_column(_TABLE, column_name)
