"""SQL and connection-wait measurements without statement or parameter labels."""

from __future__ import annotations

import time

from contextlib import suppress
from datetime import datetime
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from nexus.distributed.observability.telemetry import PlatformMetrics
from nexus.utils.platform_metrics import current_metric_attributes, current_recorder


class ObservedAsyncSession(AsyncSession):
    _observed_transaction: Any = None

    async def _prepare_connection(self) -> None:
        recorder = current_recorder() or self.info.get('platform_metrics')
        if recorder is None or not getattr(recorder, 'available', True):
            return
        transaction = self.sync_session.get_transaction()
        if transaction is not None and transaction is self._observed_transaction:
            return
        source = self.info.get('platform_metric_source', 'control_db')
        started = time.perf_counter()
        outcome = 'success'
        try:
            await super().connection()
            self._observed_transaction = self.sync_session.get_transaction()
        except BaseException:
            outcome = 'error'
            recorder.record(
                'database_errors', attributes={'operation': 'acquire', 'source': source, 'outcome': outcome}
            )
            raise
        finally:
            recorder.record(
                'database_operations', attributes={'source': source, 'operation': 'acquire', 'outcome': outcome}
            )
            if outcome == 'success':
                recorder.record('database_errors', 0, {'source': source, 'operation': 'acquire', 'outcome': outcome})
            recorder.record(
                'database_acquisition_distribution',
                time.perf_counter() - started,
                {'outcome': outcome, 'source': source, 'operation': 'acquire'},
            )

    async def flush(self, objects=None):
        if self.new or self.dirty or self.deleted:
            await self._prepare_connection()
        return await super().flush(objects)

    async def commit(self):
        if self.new or self.dirty or self.deleted:
            await self._prepare_connection()
        return await super().commit()

    async def execute(self, *args, **kwargs):
        await self._prepare_connection()
        return await super().execute(*args, **kwargs)

    async def scalar(self, *args, **kwargs):
        await self._prepare_connection()
        return await super().scalar(*args, **kwargs)

    async def get(self, *args, **kwargs):
        await self._prepare_connection()
        return await super().get(*args, **kwargs)


def install_database_metrics(
    factory: async_sessionmaker, metrics: PlatformMetrics, *, source: str = 'control_db', slow_seconds: float = 0.5
) -> None:
    if not metrics.available:
        return
    engine = getattr(factory, 'kw', {}).get('bind')
    if not isinstance(engine, AsyncEngine):
        return
    factory.configure(
        info={**(factory.kw.get('info') or {}), 'platform_metrics': metrics, 'platform_metric_source': source}
    )
    sync = engine.sync_engine
    sync._nexus_platform_metrics_default = metrics
    installed = getattr(sync, '_nexus_platform_metrics_installed', False)
    active = {}

    def before(_connection, _cursor, statement, _parameters, context, _executemany):
        recorder = current_recorder() or sync._nexus_platform_metrics_default
        context._nexus_metric_recorder = recorder
        context._nexus_metric_started = time.perf_counter()
        first = statement.lstrip().split(None, 1)[0].lower() if statement.strip() else 'other'
        context._nexus_metric_operation = (
            first if first in {'select', 'insert', 'update', 'delete', 'commit', 'rollback'} else 'other'
        )
        active[recorder] = active.get(recorder, 0) + 1
        recorder.record('database_sql_active', active[recorder], {'source': source})

    def finish(context, *, outcome):
        recorder = context._nexus_metric_recorder
        started = getattr(context, '_nexus_metric_started', None)
        if started is None:
            return
        context._nexus_metric_started = None
        active[recorder] = max(0, active.get(recorder, 0) - 1)
        duration = time.perf_counter() - started
        dimensions = {'source': source, 'operation': context._nexus_metric_operation, 'outcome': outcome}
        recorder.record('database_sql_active', active[recorder], {'source': source})
        if active[recorder] == 0:
            active.pop(recorder, None)
        recorder.record('database_sql_duration_distribution', duration, dimensions)
        recorder.record('database_operations', attributes=dimensions)
        if outcome == 'success':
            recorder.record('database_errors', 0, dimensions)
        if duration >= slow_seconds:
            recorder.record('database_slow_queries', attributes=dimensions)

    def after(_connection, _cursor, _statement, _parameters, context, _executemany):
        finish(context, outcome='success')

    def failed(context):
        if context.execution_context is None:
            return
        recorder = (
            getattr(context.execution_context, '_nexus_metric_recorder', None) or sync._nexus_platform_metrics_default
        )
        recorder.record(
            'database_errors',
            attributes={
                'source': source,
                'operation': getattr(context.execution_context, '_nexus_metric_operation', 'connect'),
                'reason': type(context.original_exception).__name__,
                'outcome': 'error',
            },
        )
        if context.execution_context is not None:
            finish(context.execution_context, outcome='error')

    if not installed:
        sync._nexus_platform_metrics_installed = True
        event.listen(sync, 'before_cursor_execute', before)
        event.listen(sync, 'after_cursor_execute', after)
        event.listen(sync, 'handle_error', failed)

    def pool_values():
        pool = sync.pool
        if not hasattr(pool, 'checkedout'):
            return []
        size = pool.size()
        overflow = getattr(pool, '_max_overflow', None)
        values = [
            (pool.checkedout(), {'source': source, 'kind': 'used'}),
            (size, {'source': source, 'kind': 'base_size'}),
        ]
        if overflow is not None and overflow >= 0:
            values.append((size + overflow, {'source': source, 'kind': 'limit'}))
        return values

    metrics.set_gauge_callback('database_pool', pool_values)


def queue_committed_metric(session: AsyncSession, name: str, value: float = 1, /, **attributes: str) -> None:
    """Queue a state-transition measurement for the root transaction's commit."""
    recorder = current_recorder()
    if recorder is None or not getattr(recorder, 'available', True):
        return
    synchronous = session.sync_session
    key = 'platform_commit_metrics'
    if not synchronous.info.get('platform_commit_hooks'):
        synchronous.info['platform_commit_hooks'] = True

        def committed(current):
            if current.in_nested_transaction():
                return
            for _transaction, owner, metric, amount, labels in current.info.pop(key, []):
                with suppress(Exception):
                    owner.record(metric, amount, labels)

        def rolled_back(current, previous):
            def belongs_to_rollback(transaction):
                while transaction is not None:
                    if transaction is previous:
                        return True
                    transaction = transaction.parent
                return False

            current.info[key] = [entry for entry in current.info.get(key, []) if not belongs_to_rollback(entry[0])]

        event.listen(synchronous, 'after_commit', committed)
        event.listen(synchronous, 'after_soft_rollback', rolled_back)
    transaction = synchronous.get_nested_transaction() or synchronous.get_transaction()
    labels = {**current_metric_attributes(), **attributes}
    synchronous.info.setdefault(key, []).append((transaction, recorder, name, value, labels))


def queue_stage_result(
    session: AsyncSession,
    *,
    stage: str,
    started_at: datetime | None,
    now: datetime,
    outcome: str,
    environment: str,
) -> None:
    labels = {'stage': stage, 'operation': stage, 'outcome': outcome, 'environment': environment}
    queue_committed_metric(session, 'stage_operations', **labels)
    queue_committed_metric(session, 'stage_failures', int(outcome not in {'success', 'cancelled'}), **labels)
    if started_at is not None and now >= started_at:
        queue_committed_metric(session, 'stage_duration_distribution', (now - started_at).total_seconds(), **labels)
