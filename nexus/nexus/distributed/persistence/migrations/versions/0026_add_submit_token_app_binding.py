"""Bind trusted application identities to submit tokens."""

from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations


revision = '0026_submit_token_app_binding'
down_revision = '0025_default_token_quota'
branch_labels = None
depends_on = None


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    columns = {column['name'] for column in sa.inspect(bind).get_columns('distributed_submit_tokens')}
    if 'bound_app_id' not in columns:
        op.add_column(
            'distributed_submit_tokens',
            sa.Column('bound_app_id', sa.String(length=128), nullable=True),
        )
    indexes = {index['name'] for index in sa.inspect(bind).get_indexes('distributed_submit_tokens')}
    if 'uq_submit_tokens_bound_app_id' not in indexes:
        op.create_index(
            'uq_submit_tokens_bound_app_id',
            'distributed_submit_tokens',
            ['bound_app_id'],
            unique=True,
        )


def downgrade(op: Operations) -> None:
    bind = op.get_bind()
    indexes = {index['name'] for index in sa.inspect(bind).get_indexes('distributed_submit_tokens')}
    if 'uq_submit_tokens_bound_app_id' in indexes:
        op.drop_index('uq_submit_tokens_bound_app_id', table_name='distributed_submit_tokens')
    columns = {column['name'] for column in sa.inspect(bind).get_columns('distributed_submit_tokens')}
    if 'bound_app_id' in columns:
        op.drop_column('distributed_submit_tokens', 'bound_app_id')
