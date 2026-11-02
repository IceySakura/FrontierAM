from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations
from sqlalchemy import inspect


revision = '0011_dependency_provider_snap'
down_revision = '0010_batch_lifecycle_fair'


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    table_name = 'distributed_dependency_builds'
    inspector = inspect(bind)
    if table_name not in inspector.get_table_names():
        return

    columns = {column['name'] for column in inspector.get_columns(table_name)}
    if 'provider_type' not in columns:
        op.add_column(table_name, sa.Column('provider_type', sa.String(32), nullable=True))
    if 'provider_config_ref' not in columns:
        op.add_column(table_name, sa.Column('provider_config_ref', sa.String(512), nullable=True))
    if 'provider_config_snapshot_json' not in columns:
        op.add_column(table_name, sa.Column('provider_config_snapshot_json', sa.JSON(), nullable=True))

    # Preview builds were created before provider targets were snapshotted. They
    # cannot be resumed safely; ready wheelhouses remain reusable.
    bind.execute(
        sa.text(
            """
            UPDATE distributed_environment_allocations
            SET dependency_build_id = NULL
            WHERE dependency_build_id IN (
                SELECT dependency_build_id
                FROM distributed_dependency_builds
                WHERE status <> 'ready'
                  AND provider_config_snapshot_json IS NULL
            )
            """
        )
    )
    bind.execute(
        sa.text(
            """
            DELETE FROM distributed_artifact_refs
            WHERE resource_type = 'dependency_build'
              AND resource_id IN (
                  SELECT dependency_build_id
                  FROM distributed_dependency_builds
                  WHERE status <> 'ready'
                    AND provider_config_snapshot_json IS NULL
              )
            """
        )
    )
    bind.execute(
        sa.text(
            """
            DELETE FROM distributed_dependency_builds
            WHERE status <> 'ready'
              AND provider_config_snapshot_json IS NULL
            """
        )
    )


def downgrade(op: Operations) -> None:
    bind = op.get_bind()
    table_name = 'distributed_dependency_builds'
    inspector = inspect(bind)
    if table_name not in inspector.get_table_names():
        return
    columns = {column['name'] for column in inspector.get_columns(table_name)}
    for column_name in (
        'provider_config_snapshot_json',
        'provider_config_ref',
        'provider_type',
    ):
        if column_name in columns:
            op.drop_column(table_name, column_name)
