"""Recover historical Batch finish times and identify updated_at estimates."""

from __future__ import annotations

from alembic.operations import Operations
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    String,
    and_,
    column,
    false,
    func,
    inspect,
    or_,
    select,
    table,
    true,
    update,
)


revision = '0034_batch_finish_backfill'
down_revision = '0033_business_statistics'
branch_labels = None
depends_on = None

PAGE_SIZE = 1000
TERMINAL = ('succeeded', 'failed', 'lost', 'expired', 'killed', 'cancelled')


def upgrade_batches(op: Operations):
    connection = op.get_bind()
    if 'finished_at_estimated' not in {
        value['name'] for value in inspect(connection).get_columns('distributed_batch_items')
    }:
        op.add_column(
            'distributed_batch_items',
            Column('finished_at_estimated', Boolean(), nullable=False, server_default=false()),
        )
    yield

    items = table(
        'distributed_batch_items',
        column('parent_execution_id', String(64)),
        column('item_id', String(64)),
        column('current_child_execution_id', String(64)),
        column('status', String(32)),
        column('finished_at', DateTime()),
        column('finished_at_estimated', Boolean()),
        column('updated_at', DateTime()),
    )
    children = table(
        'executions',
        column('execution_id', String(64)),
        column('status', String(32)),
        column('finished_at', DateTime()),
    )
    parent, item = items.c.parent_execution_id, items.c.item_id
    last = connection.execute(select(parent, item).order_by(parent.desc(), item.desc()).limit(1)).first()
    if last is None:
        return
    high_watermark = or_(parent < last[0], and_(parent == last[0], item <= last[1]))
    actual_finish = (
        select(children.c.finished_at)
        .where(children.c.execution_id == items.c.current_child_execution_id, children.c.status == items.c.status)
        .scalar_subquery()
    )
    after = None
    while True:
        lower = or_(parent > after[0], and_(parent == after[0], item > after[1])) if after else true()
        keys = connection.execute(
            select(parent, item).where(lower, high_watermark).order_by(parent, item).limit(PAGE_SIZE)
        ).all()
        if not keys:
            break
        end = keys[-1]
        upper = or_(parent < end[0], and_(parent == end[0], item <= end[1]))
        connection.execute(
            update(items)
            .where(lower, upper, items.c.status.in_(TERMINAL), items.c.finished_at.is_(None))
            .values(
                finished_at=func.coalesce(actual_finish, items.c.updated_at),
                finished_at_estimated=actual_finish.is_(None),
            )
        )
        # The migration runner commits each page; completed values make a retry idempotent.
        yield
        after = end


def upgrade(op: Operations) -> None:
    for _ in upgrade_batches(op):
        pass


def downgrade(op: Operations) -> None:
    raise RuntimeError('Batch finish backfill cannot be downgraded without losing timestamp provenance')
