"""Add direct principal-to-principal access grants."""

from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations


revision = '0030_principal_access_grants'
down_revision = '0029_run_trace_backend_grants'
branch_labels = None
depends_on = None


_TABLE = 'distributed_principal_access_grants'


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    if _TABLE not in sa.inspect(bind).get_table_names():
        op.create_table(
            _TABLE,
            sa.Column('owner_principal_id', sa.String(length=64), nullable=False),
            sa.Column('grantee_principal_id', sa.String(length=64), nullable=False),
            sa.Column('created_by', sa.String(length=128), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(
                ['owner_principal_id'],
                ['distributed_principals.principal_id'],
                ondelete='CASCADE',
            ),
            sa.ForeignKeyConstraint(
                ['grantee_principal_id'],
                ['distributed_principals.principal_id'],
                ondelete='CASCADE',
            ),
            sa.PrimaryKeyConstraint('owner_principal_id', 'grantee_principal_id'),
        )
    index_names = {index['name'] for index in sa.inspect(bind).get_indexes(_TABLE)}
    if 'idx_principal_access_grants_grantee_owner' not in index_names:
        op.create_index(
            'idx_principal_access_grants_grantee_owner',
            _TABLE,
            ['grantee_principal_id', 'owner_principal_id'],
            unique=False,
        )


def downgrade(op: Operations) -> None:
    if _TABLE in sa.inspect(op.get_bind()).get_table_names():
        op.drop_table(_TABLE)
