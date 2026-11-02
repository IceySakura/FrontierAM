"""Add scoped database storage for distributed platform secrets."""

from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations


revision = '0023_add_config_secret_overrides'
down_revision = '0022_add_grant_scopes'
branch_labels = None
depends_on = None


def upgrade(op: Operations) -> None:
    if 'distributed_config_secret_overrides' in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        'distributed_config_secret_overrides',
        sa.Column('role', sa.String(length=64), nullable=False),
        sa.Column('identity', sa.String(length=128), nullable=False),
        sa.Column('secrets_json', sa.JSON(), nullable=False),
        sa.Column('version', sa.BigInteger(), nullable=False, server_default=sa.text('1')),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('role', 'identity'),
    )


def downgrade(op: Operations) -> None:
    if 'distributed_config_secret_overrides' in sa.inspect(op.get_bind()).get_table_names():
        op.drop_table('distributed_config_secret_overrides')
