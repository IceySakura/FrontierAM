from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from nexus.utils.otel import get_meter


ControlLoopOutcome = Literal['success', 'transient_error', 'fatal_error']


@dataclass(frozen=True, slots=True)
class ControlLoopWork:
    scanned: int = 0
    claimed: int = 0
    changed: int = 0

    def __post_init__(self) -> None:
        if min(self.scanned, self.claimed, self.changed) < 0:
            raise ValueError('control loop work counts must be non-negative')


@dataclass(frozen=True, slots=True)
class ControlLoopSample:
    component: Literal['scheduler', 'reconciler']
    loop: str
    outcome: ControlLoopOutcome
    duration_seconds: float
    lag_seconds: float = 0
    scanned: int = 0
    claimed: int = 0
    changed: int = 0
    backoff_seconds: float = 0

    def __post_init__(self) -> None:
        if not self.loop:
            raise ValueError('control loop name is required')
        for field_name in (
            'duration_seconds',
            'lag_seconds',
            'scanned',
            'claimed',
            'changed',
            'backoff_seconds',
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f'{field_name} must be non-negative')


class ControlPlaneMetrics(Protocol):
    def record_loop(self, sample: ControlLoopSample) -> None: ...

    def set_scheduler_owned_shards(self, count: int) -> None: ...

    def set_reconciler_owned_shards(self, count: int) -> None: ...

    def record_scheduler_assignments(self, count: int) -> None: ...

    def record_scheduler_lock_contention(self, count: int) -> None: ...

    def record_scheduler_no_capacity(self, count: int) -> None: ...

    def record_scheduler_candidate_queries(self, count: int) -> None: ...

    def record_scheduler_transaction_duration(self, *, duration_seconds: float) -> None: ...

    def record_reconciler_overdue_lag(self, *, operation: str, lag_seconds: float) -> None: ...

    def record_reconciler_repair(self, *, reason: str, count: int) -> None: ...

    def record_batch_full_scan(self, *, duration_seconds: float, scanned: int) -> None: ...

    def record_batch_terminal_discovery_lag(self, *, lag_seconds: float) -> None: ...


class OpenTelemetryControlPlaneMetrics(ControlPlaneMetrics):
    """Low-cardinality control-loop metrics shared by Scheduler and Reconciler."""

    def __init__(self, *, meter: Any | None = None) -> None:
        meter = meter or get_meter('nexus.distributed.control-plane')
        self._loop_ticks = meter.create_counter(
            'nexus.distributed.control_loop.ticks',
            unit='{tick}',
        )
        self._loop_duration = meter.create_histogram(
            'nexus.distributed.control_loop.duration',
            unit='s',
        )
        self._loop_lag = meter.create_histogram(
            'nexus.distributed.control_loop.lag',
            unit='s',
        )
        self._loop_backoff = meter.create_histogram(
            'nexus.distributed.control_loop.backoff',
            unit='s',
        )
        self._loop_work = meter.create_counter(
            'nexus.distributed.control_loop.work',
            unit='{item}',
        )
        self._scheduler_owned_shards = meter.create_up_down_counter(
            'nexus.distributed.control_plane.scheduler.owned_shards',
            unit='{shard}',
        )
        self._reconciler_owned_shards = meter.create_up_down_counter(
            'nexus.distributed.control_plane.reconciler.owned_shards',
            unit='{shard}',
        )
        self._scheduler_assignments = meter.create_counter(
            'nexus.distributed.control_plane.scheduler.assignments',
            unit='{assignment}',
        )
        self._scheduler_lock_contention = meter.create_counter(
            'nexus.distributed.control_plane.scheduler.lock_contention',
            unit='{execution}',
        )
        self._scheduler_no_capacity = meter.create_counter(
            'nexus.distributed.control_plane.scheduler.no_capacity',
            unit='{execution}',
        )
        self._scheduler_candidate_queries = meter.create_counter(
            'nexus.distributed.control_plane.scheduler.candidate_queries',
            unit='{query}',
        )
        self._scheduler_transaction_duration = meter.create_histogram(
            'nexus.distributed.control_plane.scheduler.transaction_duration',
            unit='s',
        )
        self._reconciler_overdue_lag = meter.create_histogram(
            'nexus.distributed.control_plane.reconciler.overdue_lag',
            unit='s',
        )
        self._reconciler_repairs = meter.create_counter(
            'nexus.distributed.control_plane.reconciler.repairs',
            unit='{execution}',
        )
        self._batch_full_scan_duration = meter.create_histogram(
            'nexus.distributed.control_plane.reconciler.batch_full_scan.duration',
            unit='s',
        )
        self._batch_full_scan_items = meter.create_histogram(
            'nexus.distributed.control_plane.reconciler.batch_full_scan.items',
            unit='{item}',
        )
        self._batch_terminal_discovery_lag = meter.create_histogram(
            'nexus.distributed.control_plane.reconciler.batch_terminal_discovery_lag',
            unit='s',
        )
        self._owned_shards = 0
        self._reconciler_owned_shard_count = 0

    def record_loop(self, sample: ControlLoopSample) -> None:
        tick_attributes = {
            'component': sample.component,
            'loop': sample.loop,
            'outcome': sample.outcome,
        }
        loop_attributes = {
            'component': sample.component,
            'loop': sample.loop,
        }
        self._loop_ticks.add(1, attributes=tick_attributes)
        self._loop_duration.record(sample.duration_seconds, attributes=tick_attributes)
        self._loop_lag.record(sample.lag_seconds, attributes=loop_attributes)
        if sample.backoff_seconds:
            self._loop_backoff.record(sample.backoff_seconds, attributes=tick_attributes)
        for kind, value in (
            ('scanned', sample.scanned),
            ('claimed', sample.claimed),
            ('changed', sample.changed),
        ):
            self._loop_work.add(
                value,
                attributes={
                    'component': sample.component,
                    'loop': sample.loop,
                    'kind': kind,
                },
            )

    def set_scheduler_owned_shards(self, count: int) -> None:
        if count < 0:
            raise ValueError('owned shard count must be non-negative')
        delta = count - self._owned_shards
        if delta:
            self._scheduler_owned_shards.add(delta, attributes={'component': 'scheduler'})
            self._owned_shards = count

    def set_reconciler_owned_shards(self, count: int) -> None:
        if count < 0:
            raise ValueError('owned shard count must be non-negative')
        delta = count - self._reconciler_owned_shard_count
        if delta:
            self._reconciler_owned_shards.add(delta, attributes={'component': 'reconciler'})
            self._reconciler_owned_shard_count = count

    def record_scheduler_assignments(self, count: int) -> None:
        self._add_non_negative(
            self._scheduler_assignments,
            count,
            attributes={'component': 'scheduler'},
        )

    def record_scheduler_lock_contention(self, count: int) -> None:
        self._add_non_negative(
            self._scheduler_lock_contention,
            count,
            attributes={'component': 'scheduler'},
        )

    def record_scheduler_no_capacity(self, count: int) -> None:
        self._add_non_negative(
            self._scheduler_no_capacity,
            count,
            attributes={'component': 'scheduler'},
        )

    def record_scheduler_candidate_queries(self, count: int) -> None:
        self._add_non_negative(
            self._scheduler_candidate_queries,
            count,
            attributes={'component': 'scheduler'},
        )

    def record_scheduler_transaction_duration(self, *, duration_seconds: float) -> None:
        if duration_seconds < 0:
            raise ValueError('scheduler transaction duration must be non-negative')
        self._scheduler_transaction_duration.record(
            duration_seconds,
            attributes={'component': 'scheduler'},
        )

    def record_reconciler_overdue_lag(self, *, operation: str, lag_seconds: float) -> None:
        if lag_seconds < 0:
            raise ValueError('overdue lag must be non-negative')
        self._reconciler_overdue_lag.record(
            lag_seconds,
            attributes={
                'component': 'reconciler',
                'operation': operation,
            },
        )

    def record_reconciler_repair(self, *, reason: str, count: int) -> None:
        self._add_non_negative(
            self._reconciler_repairs,
            count,
            attributes={
                'component': 'reconciler',
                'reason': reason,
            },
        )

    def record_batch_full_scan(self, *, duration_seconds: float, scanned: int) -> None:
        if duration_seconds < 0 or scanned < 0:
            raise ValueError('batch full-scan metrics must be non-negative')
        attributes = {'component': 'reconciler', 'loop': 'batch'}
        self._batch_full_scan_duration.record(duration_seconds, attributes=attributes)
        self._batch_full_scan_items.record(scanned, attributes=attributes)

    def record_batch_terminal_discovery_lag(self, *, lag_seconds: float) -> None:
        if lag_seconds < 0:
            raise ValueError('batch terminal discovery lag must be non-negative')
        self._batch_terminal_discovery_lag.record(
            lag_seconds,
            attributes={'component': 'reconciler', 'loop': 'batch'},
        )

    @staticmethod
    def _add_non_negative(instrument: Any, value: int, *, attributes: dict[str, str]) -> None:
        if value < 0:
            raise ValueError('metric counter value must be non-negative')
        if value:
            instrument.add(value, attributes=attributes)


class OpenTelemetryExecutionLogMetrics:
    """Low-cardinality metrics for the unified execution log endpoint."""

    def __init__(self, *, meter: Any | None = None) -> None:
        meter = meter or get_meter('nexus.distributed.execution-logs')
        self._queries = meter.create_counter('nexus_execution_log_query_total', unit='{query}')
        self._latency = meter.create_histogram('nexus_zhiyan_log_query_latency_seconds', unit='s')
        self._records = meter.create_histogram('nexus_zhiyan_log_query_records', unit='{record}')
        self._pending = meter.create_counter('nexus_zhiyan_log_ingest_pending_total', unit='{query}')
        self._limited = meter.create_counter('nexus_zhiyan_log_query_limited_total', unit='{query}')
        self._bytes = meter.create_histogram('nexus_execution_log_response_bytes', unit='By')

    def record_query(
        self,
        *,
        source: str,
        result: str,
        latency_seconds: float,
        records: int,
        response_bytes: int,
    ) -> None:
        if min(latency_seconds, records, response_bytes) < 0:
            raise ValueError('execution log metric values must be non-negative')
        attributes = {'source': source, 'result': result}
        self._queries.add(1, attributes=attributes)
        self._latency.record(latency_seconds, attributes={'source': source})
        self._records.record(records, attributes={'source': source})
        self._bytes.record(response_bytes, attributes={'source': source})
        if result == 'ingest_pending':
            self._pending.add(1, attributes={'source': source})
        if result == 'limited':
            self._limited.add(1, attributes={'source': source})


__all__ = [
    'ControlLoopOutcome',
    'ControlLoopSample',
    'ControlLoopWork',
    'ControlPlaneMetrics',
    'OpenTelemetryControlPlaneMetrics',
    'OpenTelemetryExecutionLogMetrics',
]
