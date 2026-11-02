"""Add the administrator-managed ProviderConfig registry."""

from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations


revision = '0021_add_provider_configs'
down_revision = '0020_ensure_config_overrides'
branch_labels = None
depends_on = None


def upgrade(op: Operations) -> None:
    metadata = sa.MetaData()
    table = sa.Table(
        'distributed_provider_configs',
        metadata,
        sa.Column('provider_config_ref', sa.String(512), primary_key=True),
        sa.Column('provider_type', sa.String(32), nullable=False),
        sa.Column('config_json', sa.JSON(), nullable=False),
        sa.Column('status', sa.String(16), nullable=False),
        sa.Column('version', sa.BigInteger(), nullable=False),
        sa.Column('created_by', sa.String(128), nullable=False),
        sa.Column('updated_by', sa.String(128), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Index('idx_provider_configs_status_ref', 'status', 'provider_config_ref'),
    )
    table.create(bind=op.get_bind(), checkfirst=True)
