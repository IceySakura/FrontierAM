"""Explicit Metrics SDK ownership; no interaction with the global OTel provider."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import socket
import threading
import time

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from itertools import islice
from typing import Any

from nexus.distributed.config.observability import PlatformMetricsConfig
from nexus.utils.otel.meters import (
    load_metrics_sdk,
    load_otlp_http_metric_exporter,
    make_observation,
    rejected_data_points,
)


logger = logging.getLogger(__name__)
ATTRIBUTES = frozenset({
    'service',
    'component',
    'loop',
    'environment',
    'pool',
    'operation',
    'stage',
    'outcome',
    'reason',
    'status',
    'route',
    'source',
    'kind',
    'state',
    'work_kind',
    'category',
    'instance_id',
})
COUNTERS = frozenset({
    'gateway_requests',
    'gateway_server_errors',
    'gateway_client_aborted',
    'gateway_stream_connections',
    'gateway_stream_closed',
    'stage_operations',
    'stage_failures',
    'platform_terminal_results',
    'platform_terminal_failures',
    'platform_terminal_unknown',
    'terminal_operations',
    'terminal_failures',
    'instance_exits',
    'capacity_rejections',
    'database_errors',
    'database_operations',
    'database_slow_queries',
    'persistence_operations',
    'persistence_failures',
    'data_integrity_errors',
    'control_loop_ticks',
    'control_loop_work',
    'scheduler_assignments',
    'scheduler_lock_contention',
    'scheduler_no_capacity',
    'scheduler_candidate_queries',
    'reconciler_repairs',
    'observer_runs',
    'metrics_records_dropped',
    'metrics_exports',
})
HISTOGRAMS = frozenset({
    'gateway_duration_distribution',
    'gateway_stream_handshake_distribution',
    'stage_duration_distribution',
    'database_acquisition_distribution',
    'database_sql_duration_distribution',
    'assignment_ack_distribution',
    'terminal_report_distribution',
    'persistence_duration_distribution',
    'control_loop_duration_distribution',
    'control_loop_lag_distribution',
    'control_loop_backoff_distribution',
    'scheduler_transaction_duration_distribution',
    'reconciler_overdue_lag_distribution',
    'reconciler_batch_full_scan_duration_distribution',
    'reconciler_batch_terminal_discovery_lag_distribution',
    'observer_duration_distribution',
})
GAUGES = frozenset({
    'gateway_stream_active',
    'database_pool',
    'database_sql_active',
    'process_info',
    'event_loop_lag',
    'control_loop_success_timestamp',
    'observer_success_timestamp',
    'snapshot_age',
    'active_executions',
    'queue_depth',
    'queue_oldest_age',
    'slots',
    'objects_by_state',
    'object_oldest_age',
    'stage_overdue',
    'stage_unknown_budget',
    'operation_active',
    'operation_oldest_age',
    'operation_overdue',
    'operation_unknown_budget',
    'manager_heartbeat_age',
    'batch_progress_lag',
    'database_server',
    'scheduler_owned_shards',
    'reconciler_owned_shards',
    'pending_commands',
    'command_oldest_age',
    'resource_managers',
    'resource_cpu_millicores',
    'resource_memory_bytes',
    'resource_reserved_slots',
    'resource_active_batches',
    'manager_gap',
    'unowned_shards',
    'metrics_export_success_timestamp',
})
# Zhiyan permits at most ten explicit bounds; configured views use these same bounds.
SECONDS_BUCKETS = (0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 5, 60, 3600)
COUNT_BUCKETS = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000)
GaugeValues = Iterable[tuple[float, Mapping[str, str]]]
OTLP_HEADER_ENV_VARS = ('OTEL_EXPORTER_OTLP_HEADERS', 'OTEL_EXPORTER_OTLP_METRICS_HEADERS')


@contextmanager
def without_global_otlp_headers() -> Iterator[None]:
    """Hide global OTLP header credentials while an exporter reads its configuration.

    OTLP/HTTP exporters below 1.45 let an explicit ``headers`` argument replace the environment;
    1.45 and above merge the two. Those headers authenticate the trace collector, so a merge
    would send them to an unrelated metrics endpoint.
    """
    saved = {name: os.environ.pop(name) for name in OTLP_HEADER_ENV_VARS if name in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


class PlatformMetrics:
    def __init__(
        self,
        config: PlatformMetricsConfig,
        *,
        service: str,
        readers: Sequence[Any] | None = None,
        instance_id: str | None = None,
    ) -> None:
        self.config = config
        self.service = service
        self.process_instance_id = os.getenv('POD_UID') or f'{socket.gethostname()}:{os.getpid()}'
        self.instance_id = instance_id or self.process_instance_id
        self.provider: Any = None
        self.meter: Any = None
        self.error_code: str | None = None
        self.last_export_attempt_at: float | None = None
        self.last_export_success_at: float | None = None
        self.last_export_error_code: str | None = None
        self._closed = False
        self._closing = False
        self._lock = threading.Lock()
        self._series: dict[str, set[tuple]] = {}
        self._values: dict[str, dict[tuple, tuple[float, dict[str, str]]]] = {}
        self._callbacks: dict[str, Callable[[], GaugeValues]] = {}
        self._instruments: dict[str, Any] = {}
        self._active_operations: dict[object, tuple[float, float | None, dict[str, str]]] = {}
        self._operation_labels: dict[tuple, dict[str, str]] = {}
        if not config.enabled:
            return
        try:
            self._initialize(readers)
            self.record('process_info', 1)
            for name in ('operation_active', 'operation_oldest_age', 'operation_overdue', 'operation_unknown_budget'):
                self.set_gauge_callback(name, lambda metric=name: self._operation_gauges(metric))
        except Exception as exc:
            self.error_code = type(exc).__name__
            logger.warning('Platform metrics unavailable: error_type=%s', self.error_code)

    def _initialize(self, readers: Sequence[Any] | None) -> None:
        sdk = load_metrics_sdk()
        histogram_aggregation = sdk.ExplicitBucketHistogramAggregation
        view = sdk.View

        if readers is None:
            import requests

            from requests import certs

            exporter_sdk = load_otlp_http_metric_exporter()
            export_result = exporter_sdk.MetricExportResult
            delta = sdk.AggregationTemporality.DELTA

            owner = self

            class MetricsSession(requests.Session):
                def send(self, request, **kwargs):
                    # A trace client certificate from global OTEL_* is unrelated to this destination.
                    kwargs['cert'] = None
                    response = super().send(request, **kwargs)
                    if response.ok and response.content:
                        rejected = rejected_data_points(response.content)
                        if rejected:
                            owner.record('metrics_records_dropped', rejected, {'reason': 'receiver_rejected'})
                            owner.last_export_error_code = 'partial_success'
                    return response

            class ObservedExporter(exporter_sdk.OTLPMetricExporter):
                def export(self, metrics_data, timeout_millis=10000, **kwargs):
                    owner.last_export_attempt_at = time.time()
                    owner.last_export_error_code = None
                    outcome = 'error'
                    try:
                        result = super().export(metrics_data, timeout_millis=timeout_millis, **kwargs)
                        if result == export_result.SUCCESS and owner.last_export_error_code is None:
                            outcome = 'success'
                            owner.last_export_success_at = time.time()
                            owner.record('metrics_export_success_timestamp', owner.last_export_success_at)
                        elif owner.last_export_error_code is None:
                            owner.last_export_error_code = 'export_failed'
                        return result
                    except Exception:
                        owner.last_export_error_code = 'export_failed'
                        raise
                    finally:
                        owner.record('metrics_exports', attributes={'outcome': outcome})

            session = MetricsSession()
            session.trust_env = False
            with without_global_otlp_headers():
                exporter = ObservedExporter(
                    endpoint=self.config.endpoint,
                    headers={'Content-Type': 'application/x-protobuf'},
                    certificate_file=certs.where(),
                    timeout=self.config.export_timeout_seconds,
                    session=session,
                    preferred_temporality={sdk.Counter: delta, sdk.Histogram: delta},
                )
            readers = [
                sdk.PeriodicExportingMetricReader(
                    exporter,
                    export_interval_millis=self.config.export_interval_seconds * 1000,
                    export_timeout_millis=self.config.export_timeout_seconds * 1000,
                )
            ]
        self.provider = sdk.MeterProvider(
            resource=sdk.Resource({
                'service.name': f'nexus-{self.service}',
                '__zhiyan_app_mark__': self.config.app_mark,
                '__zhiyan_env__': self.config.environment,
                '__zhiyan_instance_mark__': self.instance_id,
                '__zhiyan_data_grain__': 60,
            }),
            metric_readers=readers,
            views=[
                view(
                    instrument_name=name.removesuffix('_distribution'),
                    aggregation=histogram_aggregation(SECONDS_BUCKETS),
                )
                for name in HISTOGRAMS
            ]
            + [
                view(
                    instrument_name='reconciler_batch_full_scan_items',
                    aggregation=histogram_aggregation(COUNT_BUCKETS),
                )
            ],
            shutdown_on_exit=False,
        )
        self.meter = self.provider.get_meter(self.config.meter_name)
        for name in COUNTERS:
            self._instruments[name] = self.meter.create_counter(name, unit='1')
        for name in HISTOGRAMS:
            self._instruments[name] = self.meter.create_histogram(name.removesuffix('_distribution'), unit='s')
        self._instruments['reconciler_batch_full_scan_items_distribution'] = self.meter.create_histogram(
            'reconciler_batch_full_scan_items', unit='1'
        )
        for name in GAUGES:
            self.meter.create_observable_gauge(name, callbacks=[self._gauge_callback(name)])

    @property
    def available(self) -> bool:
        return self.provider is not None and self.error_code is None and not self._closing

    def _dimensions(self, attributes: Mapping[str, str] | None) -> dict[str, str]:
        result = {'service': self.service, 'instance_id': self.process_instance_id}
        for key, value in (attributes or {}).items():
            if key in ATTRIBUTES and isinstance(value, str) and len(value) <= 128:
                result[key] = value
        return result

    def record(self, name: str, value: float = 1, attributes: Mapping[str, str] | None = None) -> None:
        try:
            self._record(name, value, attributes)
        except Exception:
            # SDK failures cannot escape SQL hooks, request handling or operation cleanup.
            if not getattr(self, '_record_failure_reported', False):
                self._record_failure_reported = True
                logger.warning('Platform metric recording failed')

    def _record(self, name: str, value: float, attributes: Mapping[str, str] | None) -> None:
        if (
            not self.available
            or not math.isfinite(value)
            or name not in GAUGES | COUNTERS | HISTOGRAMS | {'reconciler_batch_full_scan_items_distribution'}
        ):
            return
        dimensions = self._dimensions(attributes)
        key = tuple(sorted(dimensions.items()))
        with self._lock:
            series = self._series.setdefault(name, set())
            if key not in series and len(series) >= 2048:
                dropped = self._instruments.get('metrics_records_dropped')
                if dropped is not None:
                    dropped.add(1, {'service': self.service, 'reason': 'series_limit'})
                return
            series.add(key)
            if name in GAUGES:
                self._values.setdefault(name, {})[key] = (value, dimensions)
                return
        instrument = self._instruments.get(name)
        if instrument is not None and value >= 0:
            if name in COUNTERS:
                instrument.add(value, dimensions)
            else:
                instrument.record(value, dimensions)

    def set_gauge_callback(self, name: str, callback: Callable[[], GaugeValues]) -> None:
        if name not in GAUGES:
            raise ValueError('unknown platform gauge')
        with self._lock:
            self._callbacks[name] = callback

    def begin_operation(self, stage: str, *, timeout_seconds: float | None, attributes: Mapping[str, str]):
        if not self.available:
            return None
        labels = self._dimensions({**attributes, 'stage': stage})
        key = tuple(sorted(labels.items()))
        token = object()
        started = time.monotonic()
        deadline = started + timeout_seconds if timeout_seconds is not None else None
        with self._lock:
            if len(self._active_operations) >= 2048 or (
                key not in self._operation_labels and len(self._operation_labels) >= 2048
            ):
                return None
            self._operation_labels[key] = labels
            self._active_operations[token] = (started, deadline, labels)

        def finish():
            with self._lock:
                self._active_operations.pop(token, None)

        return finish

    def _operation_gauges(self, metric: str):
        now = time.monotonic()
        with self._lock:
            active = tuple(self._active_operations.values())
            labels = dict(self._operation_labels)
        grouped = {key: [] for key in labels}
        for started, deadline, dimensions in active:
            grouped[tuple(sorted(dimensions.items()))].append((started, deadline))
        values = []
        for key, items in grouped.items():
            unknown = sum(deadline is None for _, deadline in items)
            if metric == 'operation_active':
                value = len(items)
            elif metric == 'operation_oldest_age':
                value = max((now - start for start, _ in items), default=0)
            elif metric == 'operation_unknown_budget':
                value = unknown
            elif unknown:
                continue
            else:
                value = sum(deadline is not None and deadline < now for _, deadline in items)
            values.append((value, labels[key]))
        return values

    def _gauge_callback(self, name: str):
        def collect(_options):
            if self._closed:
                return []
            with self._lock:
                values = tuple(self._values.get(name, {}).values())
                callback = self._callbacks.get(name)
            if callback is not None:
                try:
                    values = tuple(islice(callback(), 2048))
                except Exception:
                    return []
            return [
                make_observation(value, self._dimensions(attributes))
                for value, attributes in values
                if math.isfinite(value)
            ]

        return collect

    def state(self) -> dict[str, Any]:
        return {
            'enabled': self.config.enabled,
            'available': self.available,
            'error_code': self.error_code,
            'last_export_attempt_at': self.last_export_attempt_at,
            'last_export_success_at': self.last_export_success_at,
            'last_export_error_code': self.last_export_error_code,
        }

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self.provider is None:
            self._closed = True
            return
        done = asyncio.Event()
        loop = asyncio.get_running_loop()

        def shutdown():
            try:
                self.provider.shutdown(timeout_millis=self.config.export_timeout_seconds * 1000)
            except Exception:
                pass
            finally:
                self._closed = True
                if not loop.is_closed():
                    loop.call_soon_threadsafe(done.set)

        threading.Thread(target=shutdown, name='nexus-metrics-shutdown', daemon=True).start()
        try:
            await asyncio.wait_for(done.wait(), timeout=self.config.export_timeout_seconds)
        except TimeoutError:
            logger.warning('Platform metrics shutdown exceeded its time budget')
