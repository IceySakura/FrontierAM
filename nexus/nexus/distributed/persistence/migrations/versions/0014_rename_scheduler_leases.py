from __future__ import annotations

from datetime import datetime

from alembic.operations import Operations
from sqlalchemy import inspect, text


revision = '0014_rename_scheduler_leases'
down_revision = '0013_drop_legacy_quota_fields'


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    tables = set(inspect(bind).get_table_names())
    if 'scheduler_leases' in tables and 'shard_leases' not in tables:
        op.rename_table('scheduler_leases', 'shard_leases')
    if 'shard_leases' not in set(inspect(bind).get_table_names()):
        return

    indexes = {index['name'] for index in inspect(bind).get_indexes('shard_leases')}
    for old_name in ('idx_scheduler_leases_expires', 'ix_scheduler_leases_lease_expires_at'):
        if old_name in indexes:
            op.drop_index(old_name, table_name='shard_leases')
    indexes = {index['name'] for index in inspect(bind).get_indexes('shard_leases')}
    if 'idx_shard_leases_expires' not in indexes:
        op.create_index('idx_shard_leases_expires', 'shard_leases', ['lease_expires_at'], unique=False)

    persisted_shard_count = int(bind.scalar(text('SELECT MAX(shard_count) FROM shard_leases')) or 0)

    # Ownership is ephemeral, but shard_count is a monotonic schema invariant.
    # Expired placeholders preserve that baseline while forcing every new
    # process-session owner to acquire fresh leases.
    op.execute(text('DELETE FROM shard_leases'))
    if persisted_shard_count:
        expired_at = datetime(1970, 1, 1)
        bind.execute(
            text(
                'INSERT INTO shard_leases '
                '(shard_group, shard_id, shard_count, owner_id, lease_expires_at, updated_at) '
                'VALUES '
                '(:shard_group, :shard_id, :shard_count, :owner_id, :lease_expires_at, :updated_at)'
            ),
            [
                {
                    'shard_group': shard_group,
                    'shard_id': shard_id,
                    'shard_count': persisted_shard_count,
                    'owner_id': 'migration-expired',
                    'lease_expires_at': expired_at,
                    'updated_at': expired_at,
                }
                for shard_group in ('execution-scheduler', 'execution-reconciler')
                for shard_id in range(persisted_shard_count)
            ],
        )


def downgrade(op: Operations) -> None:
    bind = op.get_bind()
    tables = set(inspect(bind).get_table_names())
    if 'shard_leases' not in tables:
        return
    op.execute(text('DELETE FROM shard_leases'))
    indexes = {index['name'] for index in inspect(bind).get_indexes('shard_leases')}
    if 'idx_shard_leases_expires' in indexes:
        op.drop_index('idx_shard_leases_expires', table_name='shard_leases')
    op.rename_table('shard_leases', 'scheduler_leases')
    indexes = {index['name'] for index in inspect(bind).get_indexes('scheduler_leases')}
    if 'idx_scheduler_leases_expires' not in indexes:
        op.create_index(
            'idx_scheduler_leases_expires',
            'scheduler_leases',
            ['lease_expires_at'],
            unique=False,
        )
