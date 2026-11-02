from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations
from sqlalchemy import inspect

from nexus.distributed.persistence.orm import Base


revision = '0004_cloud_submit_foundation'
down_revision = '0004_add_config_overrides'


_NEW_TABLES = (
    'distributed_principals',
    'distributed_environments',
    'distributed_environment_grants',
    'distributed_environment_allocations',
    'distributed_environment_resource_usage',
    'distributed_artifacts',
    'distributed_artifact_refs',
    'distributed_dependency_builds',
    'distributed_artifact_uploads',
    'distributed_artifact_quota_usage',
    'distributed_batch_runs',
    'distributed_batch_items',
    'distributed_batch_item_attempts',
    'distributed_user_roles',
    'distributed_submit_tokens',
    'distributed_user_quota_overrides',
    'distributed_quota_usage',
    'distributed_audit_events',
)


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    _widen_varchar_if_needed(op, bind, 'executions', 'idempotency_key', 256, nullable=True)
    _add_columns(op, bind, 'executions', _execution_columns())
    _add_columns(op, bind, 'assignments', _assignment_columns())
    _add_columns(op, bind, 'instance_managers', _manager_columns())
    # This is an offline identity-contract migration. Legacy managers cannot be
    # translated reliably and must re-register with the new identity fields.
    bind.execute(sa.text('DELETE FROM instance_managers'))
    _drop_index_if_exists(op, bind, 'instance_managers', 'idx_instance_managers_environment')
    _drop_index_if_exists(op, bind, 'instance_managers', 'idx_instance_managers_environment_key')
    _drop_index_if_exists(op, bind, 'executions', 'idx_executions_environment_status')
    _drop_columns(
        op,
        bind,
        'instance_managers',
        ('environment_type', 'environment_name', 'pool_name', 'environment_key', 'workload_class'),
    )
    _drop_columns(op, bind, 'executions', ('environment_key',))
    _require_manager_identity_columns(op, bind)
    _create_index_if_missing(
        op,
        bind,
        'executions',
        'idx_executions_owner_status_updated',
        ['owner_principal_id', 'status', 'updated_at'],
    )
    _create_index_if_missing(
        op,
        bind,
        'executions',
        'idx_executions_owner_cursor',
        ['owner_principal_id', 'created_at', 'execution_id'],
    )
    _create_index_if_missing(
        op,
        bind,
        'executions',
        'idx_executions_allocation_status',
        ['allocation_id', 'status'],
    )
    _create_index_if_missing(
        op,
        bind,
        'assignments',
        'idx_assignments_allocation_status',
        ['allocation_id', 'status', 'updated_at'],
    )
    _create_index_if_missing(
        op,
        bind,
        'instance_managers',
        'idx_instance_managers_allocation',
        ['allocation_id', 'last_heartbeat_at'],
    )
    _create_index_if_missing(
        op,
        bind,
        'instance_managers',
        'idx_instance_managers_environment_pool_workload',
        ['environment_id', 'pool_id', 'workload', 'last_heartbeat_at'],
    )

    existing_tables = set(inspect(bind).get_table_names())
    for table_name in _NEW_TABLES:
        if table_name not in existing_tables:
            Base.metadata.tables[table_name].create(bind=bind, checkfirst=True)
    _add_columns(op, bind, 'distributed_submit_tokens', _submit_token_columns())
    _add_columns(op, bind, 'distributed_artifact_uploads', _artifact_binding_columns())
    _add_columns(op, bind, 'distributed_artifact_uploads', _artifact_upload_observability_columns())
    _add_columns(op, bind, 'distributed_artifacts', _artifact_binding_columns())
    _add_columns(op, bind, 'distributed_artifacts', _artifact_maintenance_columns())
    _add_columns(op, bind, 'distributed_batch_runs', _batch_run_columns())
    _add_columns(op, bind, 'distributed_batch_items', _batch_item_columns())
    _add_columns(op, bind, 'distributed_batch_item_attempts', _batch_attempt_columns())
    _add_columns(op, bind, 'distributed_environment_allocations', _environment_allocation_columns())
    _create_index_if_missing(
        op,
        bind,
        'distributed_artifacts',
        'idx_artifacts_status_operation',
        ['status', 'operation_expires_at'],
    )
    _create_index_if_missing(
        op,
        bind,
        'distributed_artifacts',
        'idx_artifacts_owner_environment_status_created',
        ['owner_principal_id', 'environment_id', 'status', 'created_at'],
    )
    _create_index_if_missing(
        op,
        bind,
        'distributed_artifact_uploads',
        'idx_artifact_uploads_owner_environment_status_created',
        ['owner_principal_id', 'environment_id', 'status', 'created_at'],
    )
    _create_index_if_missing(
        op,
        bind,
        'distributed_batch_runs',
        'idx_batch_runs_control_progress',
        ['control_state', 'last_progress_at'],
    )
    _create_index_if_missing(
        op,
        bind,
        'distributed_batch_runs',
        'idx_batch_runs_generation_progress',
        ['generation_status', 'last_progress_at'],
    )
    _create_index_if_missing(
        op,
        bind,
        'distributed_batch_item_attempts',
        'idx_batch_item_attempts_parent_item_status',
        ['parent_execution_id', 'status', 'item_id', 'attempt_no'],
    )
    _create_index_if_missing(
        op,
        bind,
        'distributed_environment_allocations',
        'idx_allocations_dependency_build',
        ['dependency_build_id'],
    )


def _execution_columns() -> tuple[sa.Column, ...]:
    return (
        sa.Column('owner_principal_id', sa.String(64), nullable=True),
        sa.Column('owner_tenant', sa.String(128), nullable=True),
        sa.Column('created_by', sa.String(128), nullable=False, server_default='system:migration'),
        sa.Column('spec_sha256', sa.String(64), nullable=True),
        sa.Column('environment_id', sa.String(64), nullable=True),
        sa.Column('pool_id', sa.String(64), nullable=True),
        sa.Column('allocation_id', sa.String(64), nullable=True),
        sa.Column('workload', sa.String(32), nullable=True),
        sa.Column('submission_sha256', sa.String(64), nullable=True),
        sa.Column('allocation_spec_hash', sa.String(64), nullable=True),
        sa.Column('result_artifact_id', sa.String(64), nullable=True),
        sa.Column('first_schedulable_at', sa.DateTime(), nullable=True),
        sa.Column('stop_reason', sa.String(32), nullable=True),
        sa.Column('version', sa.BigInteger(), nullable=False, server_default='1'),
    )


def _assignment_columns() -> tuple[sa.Column, ...]:
    return (
        sa.Column('owner_principal_id', sa.String(64), nullable=True),
        sa.Column('environment_id', sa.String(64), nullable=True),
        sa.Column('pool_id', sa.String(64), nullable=True),
        sa.Column('allocation_id', sa.String(64), nullable=True),
        sa.Column('workload', sa.String(32), nullable=True),
        sa.Column('allocation_spec_hash', sa.String(64), nullable=True),
    )


def _manager_columns() -> tuple[sa.Column, ...]:
    return (
        sa.Column('draining_at', sa.DateTime(), nullable=True),
        sa.Column('owner_principal_id', sa.String(64), nullable=True),
        sa.Column('environment_id', sa.String(64), nullable=True),
        sa.Column('pool_id', sa.String(64), nullable=True),
        sa.Column('allocation_id', sa.String(64), nullable=True),
        sa.Column('workload', sa.String(32), nullable=True),
        sa.Column('allocation_spec_hash', sa.String(64), nullable=True),
        sa.Column('capability_snapshot_json', sa.JSON(), nullable=True),
        sa.Column('bootstrap_version', sa.Integer(), nullable=True),
        sa.Column('artifact_ref_version', sa.Integer(), nullable=True),
        sa.Column('supported_spec_versions_json', sa.JSON(), nullable=True),
    )


def _environment_allocation_columns() -> tuple[sa.Column, ...]:
    return (sa.Column('dependency_build_id', sa.String(64), nullable=True),)


def _submit_token_columns() -> tuple[sa.Column, ...]:
    # Older preview databases may contain unbound tokens. They remain unusable;
    # new repository writes always require a concrete Environment.
    return (sa.Column('environment_id', sa.String(64), nullable=True),)


def _artifact_maintenance_columns() -> tuple[sa.Column, ...]:
    return (
        sa.Column('operation_id', sa.String(64), nullable=True),
        sa.Column('operation_owner', sa.String(128), nullable=True),
        sa.Column('operation_expires_at', sa.DateTime(), nullable=True),
        sa.Column('failure_count', sa.Integer(), nullable=False, server_default='0'),
    )


def _artifact_upload_observability_columns() -> tuple[sa.Column, ...]:
    return (
        sa.Column('failure_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_failure_code', sa.String(128), nullable=True),
    )


def _artifact_binding_columns() -> tuple[sa.Column, ...]:
    # Preview databases may contain owner-only artifacts. They remain unreadable
    # through scoped APIs until an administrator migrates their Environment.
    return (sa.Column('environment_id', sa.String(64), nullable=True),)


def _batch_run_columns() -> tuple[sa.Column, ...]:
    return (
        sa.Column('materialize_cursor_bytes', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('task_list_manifest_json', sa.JSON(), nullable=True),
        sa.Column('task_list_sha256', sa.String(64), nullable=True),
        sa.Column('active_items', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('terminal_items', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('control_state', sa.String(32), nullable=False, server_default='active'),
        sa.Column('stop_reason', sa.String(32), nullable=True),
        sa.Column('stop_requested_at', sa.DateTime(), nullable=True),
        sa.Column('stop_cursor', sa.String(64), nullable=True),
        sa.Column('last_progress_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('finalized_at', sa.DateTime(), nullable=True),
    )


def _batch_item_columns() -> tuple[sa.Column, ...]:
    return (
        sa.Column('task_length', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('status_version', sa.BigInteger(), nullable=False, server_default='1'),
    )


def _batch_attempt_columns() -> tuple[sa.Column, ...]:
    return (
        sa.Column('status_version', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('result_artifact_id', sa.String(64), nullable=True),
        sa.Column('error_code', sa.String(128), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('finished_at', sa.DateTime(), nullable=True),
    )


def _add_columns(op: Operations, bind, table_name: str, columns: tuple[sa.Column, ...]) -> None:
    existing = {column['name'] for column in inspect(bind).get_columns(table_name)}
    for column in columns:
        if column.name not in existing:
            op.add_column(table_name, column)


def _create_index_if_missing(
    op: Operations,
    bind,
    table_name: str,
    index_name: str,
    columns: list[str],
) -> None:
    indexes = {index['name'] for index in inspect(bind).get_indexes(table_name)}
    if index_name not in indexes:
        op.create_index(index_name, table_name, columns)


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
            column = columns[column_name]
            if column['nullable']:
                batch_op.alter_column(
                    column_name,
                    existing_type=column['type'],
                    nullable=False,
                )


def _widen_varchar_if_needed(
    op: Operations,
    bind,
    table_name: str,
    column_name: str,
    length: int,
    *,
    nullable: bool,
) -> None:
    if bind.dialect.name == 'sqlite' or table_name not in inspect(bind).get_table_names():
        return
    columns = {column['name']: column for column in inspect(bind).get_columns(table_name)}
    column = columns.get(column_name)
    if column is None:
        return
    current_length = getattr(column['type'], 'length', None)
    if current_length is None or current_length >= length:
        return
    op.alter_column(
        table_name,
        column_name,
        existing_type=column['type'],
        type_=sa.String(length),
        existing_nullable=nullable,
    )
