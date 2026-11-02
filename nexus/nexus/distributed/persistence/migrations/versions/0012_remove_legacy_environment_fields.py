from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations
from sqlalchemy import inspect


revision = '0012_remove_legacy_env_fields'
down_revision = '0011_dependency_provider_snap'


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    tables = set(inspect(bind).get_table_names())

    if 'instance_managers' in tables:
        # Legacy manager identity cannot be translated reliably. Managers
        # re-register after this offline migration.
        bind.execute(sa.text('DELETE FROM instance_managers'))
        _drop_index_if_exists(op, bind, 'instance_managers', 'idx_instance_managers_environment')
        _drop_index_if_exists(op, bind, 'instance_managers', 'idx_instance_managers_environment_key')
        _drop_columns(
            op,
            bind,
            'instance_managers',
            ('environment_type', 'environment_name', 'pool_name', 'environment_key', 'workload_class'),
        )
        _require_manager_identity_columns(op, bind)

    if 'executions' in tables:
        _drop_index_if_exists(op, bind, 'executions', 'idx_executions_environment_status')
        _drop_columns(op, bind, 'executions', ('environment_key',))

    for table_name in ('distributed_environment_demands', 'distributed_environment_pools'):
        if table_name in tables:
            op.drop_table(table_name)


def _drop_index_if_exists(
    op: Operations,
    bind,
    table_name: str,
    index_name: str,
) -> None:
    indexes = {index['name'] for index in inspect(bind).get_indexes(table_name)}
    if index_name in indexes:
        op.drop_index(index_name, table_name=table_name)


def _drop_columns(
    op: Operations,
    bind,
    table_name: str,
    column_names: tuple[str, ...],
) -> None:
    existing = {column['name'] for column in inspect(bind).get_columns(table_name)}
    drop_names = [column_name for column_name in column_names if column_name in existing]
    if not drop_names:
        return
    with op.batch_alter_table(table_name) as batch_op:
        for column_name in drop_names:
            batch_op.drop_column(column_name)


def _require_manager_identity_columns(op: Operations, bind) -> None:
    columns = {column['name']: column for column in inspect(bind).get_columns('instance_managers')}
    with op.batch_alter_table('instance_managers') as batch_op:
        for column_name in ('environment_id', 'pool_id', 'workload'):
            column = columns.get(column_name)
            if column is not None and column['nullable']:
                batch_op.alter_column(
                    column_name,
                    existing_type=column['type'],
                    nullable=False,
                )
