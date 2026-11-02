"""Small, separately budgeted history reads from the configured metrics destination."""

from __future__ import annotations

import asyncio
import copy
import time

from collections import OrderedDict
from datetime import timedelta
from typing import Any

from nexus.distributed.config import DistributedConfig
from nexus.distributed.observability.history_transport import HISTORY_RANGES, HistoryTransport, align_time
from nexus.distributed.observability.sampler import SNAPSHOT_GAUGES
from nexus.distributed.observability.schema import metric_definitions
from nexus.distributed.observability.telemetry import COUNTERS
from nexus.distributed.utils.clock import utcnow


HISTORY_PANELS: dict[str, tuple[str, ...]] = {
    'terminal': ('platform_terminal_results', 'platform_terminal_failures', 'platform_terminal_unknown'),
    'gateway': ('gateway_requests', 'gateway_server_errors', 'gateway_duration_p95'),
    'preparation': ('stage_operations', 'stage_failures', 'stage_duration_p95'),
    'delivery': ('persistence_operations', 'persistence_failures', 'persistence_duration_p95'),
    'database': ('database_operations', 'database_errors', 'database_sql_duration_p95'),
    'control': ('terminal_operations', 'terminal_failures', 'terminal_report_p95'),
}
# A Gauge's kinds and states are distinct quantities, not samples of the same quantity.
# Select one semantic view before Zhiyan performs its time/instance aggregation.
METRIC_VIEWS = {
    'active_executions': [
        {'state': state} for state in ('running', 'queued', 'assigned', 'starting', 'waiting_input', 'stopping')
    ],
    'slots': [{'kind': kind} for kind in ('available', 'reserved', 'total')],
    'queue_depth': [{'kind': kind} for kind in ('schedulable', 'preparing', 'retry_wait')],
    'database_pool': [{'kind': kind} for kind in ('used', 'limit', 'base_size')],
    'database_server': [{'kind': kind} for kind in ('connected', 'running', 'limit')],
    'objects_by_state': [
        {'category': 'inventory', 'kind': 'manager', 'state': state} for state in ('healthy', 'draining')
    ]
    + [
        {'category': category, 'state': state}
        for category, states in {
            'allocations': ('requested', 'provisioning', 'ready', 'draining', 'deleting'),
            'builds': ('requested', 'building', 'failed', 'ready'),
            'uploads': ('pending', 'receiving', 'maintenance'),
        }.items()
        for state in states
    ],
    'pending_commands': [
        {'operation': operation, 'state': state}
        for operation in ('stop.request', 'input.submit', 'other')
        for state in ('pending', 'claimed')
    ],
    'unowned_shards': [{'component': component} for component in ('execution-scheduler', 'execution-reconciler')],
}
for _name in ('stage_overdue', 'stage_unknown_budget'):
    METRIC_VIEWS[_name] = [
        {'stage': stage}
        for stage in (
            'queued',
            'assigned',
            'starting',
            'running',
            'waiting_input',
            'stopping',
            'provisioning',
            'allocation_release',
            'dependency_build',
            'builder_release',
            'batch_preparation',
            'batch_finalization',
            'artifact_upload',
        )
    ]
METRIC_VIEWS['stage_unknown_budget'] = METRIC_VIEWS['stage_unknown_budget'][6:]
HISTORY_CATALOG: dict[str, dict[str, Any]] = {
    definition['metric_enname']: {
        'name': definition['metric_enname'],
        'label': definition['metric_enname'].replace('_', ' '),
        'unit': definition['unit'],
        'aggregation': definition['cal_op'],
        'kind': 'counter' if definition['metric_enname'] in COUNTERS else 'sample',
        'scope': 'global' if definition['metric_enname'] == 'database_server' else 'selected',
        'views': [
            {'id': '.'.join(filters.values()), 'label': ' / '.join(filters.values()), 'filters': filters}
            for filters in METRIC_VIEWS.get(definition['metric_enname'], [])
        ],
    }
    for definition in metric_definitions()
}
HISTORY_PANELS.update({name: (name,) for name in HISTORY_CATALOG})


class MetricHistoryReader:
    def __init__(self, config: DistributedConfig) -> None:
        self.config = config.metrics_history
        self.credential = config.secrets.metrics_query
        self.trace_db_enabled = any(
            backend.kind == 'tdsql' or (backend.sink_class or '').startswith('nexus.runner.trace.tdsql')
            for backend in config.run_trace_sinks.values()
        )
        self._transport = HistoryTransport(self.config, self.credential)
        self._waiters: dict[asyncio.Task, int] = {}
        self._cache: OrderedDict[tuple, tuple[float, dict]] = OrderedDict()
        self._pending: dict[tuple, asyncio.Task] = {}
        self._slots = asyncio.Semaphore(4)

    @property
    def available(self) -> bool:
        return self.config.enabled and self.credential is not None

    async def get(self, *, panel: str, hours: int, filters: dict[str, str], view: str | None = None) -> dict[str, Any]:
        if not self.available:
            return {'source': 'zhiyan', 'data_status': 'unavailable', 'error_code': 'history_not_configured'}
        if panel not in HISTORY_PANELS or hours not in HISTORY_RANGES:
            raise ValueError('invalid_history_query')
        if panel == 'database_server':
            filters = {**filters, 'environment': 'all'}
        if set(filters) - {'environment', 'service', 'stage', 'outcome', 'category', 'source', 'operation', 'route'}:
            raise ValueError('invalid_history_filter')
        if any(
            not value or len(value) > (128 if key == 'route' else 64) or any(ord(char) < 32 for char in value)
            for key, value in filters.items()
        ):
            raise ValueError('invalid_history_filter')
        views = HISTORY_CATALOG.get(panel, {}).get('views', [])
        if views:
            selected = next((item for item in views if item['id'] == (view or views[0]['id'])), None)
            if selected is None or any(
                key in filters and filters[key] != value for key, value in selected['filters'].items()
            ):
                raise ValueError('invalid_history_view')
            filters = {**filters, **selected['filters']}
        elif view is not None:
            raise ValueError('invalid_history_view')
        if panel in SNAPSHOT_GAUGES:
            filters = {'environment': 'all', **filters}
        window_end = align_time(utcnow(), HISTORY_RANGES[hours]) if hours > 24 else None
        key = (panel, hours, tuple(sorted(filters.items())), window_end)
        entry = self._cache.get(key)
        if entry and time.monotonic() - entry[0] < (55 if hours <= 24 else 240):
            self._cache.move_to_end(key)
            return copy.deepcopy(entry[1])
        task = self._pending.get(key)
        if task is None:
            if len(self._pending) >= 16:
                raise TimeoutError('history_query_capacity')
            task = asyncio.create_task(self._load(key, panel, hours, filters))
            self._pending[key] = task
            task.add_done_callback(lambda done: self._finish(key, done))
        self._waiters[task] = self._waiters.get(task, 0) + 1
        try:
            value = await asyncio.shield(task)
            if value.get('data_status') == 'unavailable' and entry is not None:
                retained = copy.deepcopy(entry[1])
                retained.update(
                    data_status='stale',
                    error_code='history_query_failed',
                    failed_intervals=value.get('failed_intervals', []),
                )
                return retained
            return copy.deepcopy(value)
        except Exception:
            if entry is None:
                raise
            value = copy.deepcopy(entry[1])
            value.update(data_status='stale', error_code='history_query_failed')
            return value
        finally:
            self._waiters[task] -= 1
            if not self._waiters[task]:
                self._waiters.pop(task)
                if not task.done():
                    if self._pending.get(key) is task:
                        self._pending.pop(key, None)
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    @property
    def _client(self):
        return self._transport.client

    def _finish(self, key, task):
        if self._pending.get(key) is task:
            self._pending.pop(key, None)
        if not task.cancelled():
            task.exception()

    async def _load(self, key, panel, hours, filters):
        async def run():
            async with self._slots:
                requested_end = utcnow().replace(second=0, microsecond=0)
                now = key[3] or requested_end
                series = await asyncio.gather(
                    *(self._query(metric, now=now, hours=hours, filters=filters) for metric in HISTORY_PANELS[panel])
                )
                for index, item in enumerate(series):
                    item['role'] = (
                        'sample'
                        if len(series) == 1
                        else ('completed', 'failed', 'unknown' if panel == 'terminal' else 'duration')[index]
                    )
                # A missing numerator is unknown; it must not silently become a zero-error sample.
                denominator = _total(series[0]['points']) if len(series) > 1 else None
                numerator = _total(series[1]['points']) if len(series) > 1 else None
                complete = _matching_coverage(series[0]['points'], series[1]['points']) if len(series) > 1 else True
                value = {
                    'source': 'zhiyan',
                    'data_status': 'ready',
                    'queried_at': utcnow().isoformat(),
                    'window_start': (now - timedelta(hours=hours)).isoformat(),
                    'window_end': now.isoformat(),
                    'granularity_seconds': HISTORY_RANGES[hours],
                    'window_alignment': 'complete_buckets',
                    'requested_window_end': requested_end.isoformat(),
                    'retention_days': self.config.retention_days,
                    'retention_verified': self.config.retention_days is not None,
                    'panel': panel,
                    'filters': filters,
                    'series': series,
                    'metric': HISTORY_CATALOG.get(panel),
                    'completed': denominator,
                    'failed': numerator,
                    'failure_ratio': numerator / denominator
                    if complete and denominator and numerator is not None
                    else None,
                }
                has_samples = any(point['value'] is not None for item in series for point in item['points'])
                failures = [failure for item in series for failure in item.get('failed_intervals', [])]
                value['failed_intervals'] = failures
                if failures:
                    value.update(
                        data_status='partial' if has_samples else 'unavailable',
                        error_code='history_query_failed',
                    )
                    if any(item.get('failed_intervals') for item in series[:2]):
                        value['failure_ratio'] = None
                elif not has_samples:
                    value['data_status'] = 'no_samples'
                    if filters.get('source') == 'trace_db' and not self.trace_db_enabled:
                        value.update(data_status='not_configured', error_code='trace_database_not_enabled')
                elif not complete:
                    value['data_status'] = 'partial'
                return value

        value = await asyncio.wait_for(run(), timeout=4 if hours <= 24 else 5)
        if value['data_status'] in {'ready', 'no_samples', 'not_configured'}:
            self._cache[key] = (time.monotonic(), value)
            self._cache.move_to_end(key)
        while len(self._cache) > 256:
            self._cache.popitem(last=False)
        return value

    async def _query(self, metric, *, now, hours, filters):
        return await self._transport.query(metric, now=now, hours=hours, filters=filters)

    async def close(self):
        pending = tuple(self._pending.values())
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await self._transport.close()


def _total(points):
    values = [point['value'] for point in points if point['value'] is not None]
    return sum(values) if values else None


def _matching_coverage(left, right):
    covered = {point['time'] for point in left if point['value'] is not None}
    return bool(covered) and covered == {point['time'] for point in right if point['value'] is not None}
