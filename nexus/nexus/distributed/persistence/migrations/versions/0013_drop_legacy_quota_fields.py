from __future__ import annotations

from alembic.operations import Operations
from sqlalchemy import inspect


revision = '0013_drop_legacy_quota_fields'
down_revision = '0012_remove_legacy_env_fields'


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    table_name = 'distributed_quota_usage'
    if table_name not in inspect(bind).get_table_names():
        return
    existing = {column['name'] for column in inspect(bind).get_columns(table_name)}
    drop_names = [
        column_name
        for column_name in ('active_environment_pools', 'desired_manager_replicas')
        if column_name in existing
    ]
    if not drop_names:
        return
    with op.batch_alter_table(table_name) as batch_op:
        for column_name in drop_names:
            batch_op.drop_column(column_name)
