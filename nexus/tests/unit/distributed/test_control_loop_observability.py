from __future__ import annotations

import asyncio

from collections import defaultdict

import pytest

from sqlalchemy.exc import OperationalError

from nexus.distributed.artifacts.store import ArtifactStoreTransientError
from nexus.distributed.config import DistributedConfig
from nexus.distributed.observability import ControlLoopSample, ControlLoopWork, OpenTelemetryControlPlaneMetrics
from nexus.distributed.control_plane.reconciler import ReconcilerService
from nexus.distributed.control_plane.scheduler import SchedulerService


class _Instrument:
    def __init__(self) -> None:
        self.calls: list[tuple[float | int, dict[str, str]]] = []

    def add(self, value, attributes=None) -> None:
        self.calls.append((value, dict(attributes or {})))

    def record(self, value, attributes=None) -> None:
        self.calls.append((value, dict(attributes or {})))


class _Meter:
    def __init__(self) -> None:
        self.instruments: dict[str, _Instrument] = defaultdict(_Instrument)

    def create_counter(self, name, **kwargs):
        del kwargs
        return self.instruments[name]

    def create_histogram(self, name, **kwargs):
        del kwargs
        return self.instruments[name]

    def create_up_down_counter(self, name, **kwargs):
        del kwargs
        return self.instruments[name]


class _RecordingMetrics:
    def __init__(self) -> None:
        self.samples: list[ControlLoopSample] = []
        self.owned_shards: list[int] = []
        self.reconciler_owned_shards: list[int] = []
        self.no_capacity: list[int] = []
        self.candidate_queries: list[int] = []
        self.transaction_durations: list[float] = []

    def record_loop(self, sample: ControlLoopSample) -> None:
        self.samples.append(sample)

    def set_scheduler_owned_shards(self, count: int) -> None:
        self.owned_shards.append(count)

    def set_reconciler_owned_shards(self, count: int) -> None:
        self.reconciler_owned_shards.append(count)

    def record_scheduler_assignments(self, count: int) -> None:
        del count

    def record_scheduler_lock_contention(self, count: int) -> None:
        del count

    def record_scheduler_no_capacity(self, count: int) -> None:
        self.no_capacity.append(count)

    def record_scheduler_candidate_queries(self, count: int) -> None:
        self.candidate_queries.append(count)

    def record_scheduler_transaction_duration(self, *, duration_seconds: float) -> None:
        self.transaction_durations.append(duration_seconds)

    def record_reconciler_overdue_lag(self, *, operation: str, lag_seconds: float) -> None:
        del operation, lag_seconds

    def record_reconciler_repair(self, *, reason: str, count: int) -> None:
        del reason, count

    def record_batch_full_scan(self, *, duration_seconds: float, scanned: int) -> None:
        del duration_seconds, scanned

    def record_batch_terminal_discovery_lag(self, *, lag_seconds: float) -> None:
        del lag_seconds


class _NoopController:
    async def reconcile_once(self) -> int:
        return 0

    async def close(self) -> None:
        pass


class _NoopWorkloadClientResolver:
    async def close(self) -> None:
        pass


def test_control_plane_metric_labels_are_bounded_and_stable() -> None:
    meter = _Meter()
    metrics = OpenTelemetryControlPlaneMetrics(meter=meter)

    metrics.record_loop(
        ControlLoopSample(
            component='scheduler',
            loop='assignment',
            outcome='success',
            duration_seconds=0.25,
            lag_seconds=0.01,
            scanned=20,
            claimed=10,
            changed=7,
            backoff_seconds=0,
        )
    )

    assert meter.instruments['nexus.distributed.control_loop.ticks'].calls == [
        (1, {'component': 'scheduler', 'loop': 'assignment', 'outcome': 'success'})
    ]
    assert meter.instruments['nexus.distributed.control_loop.work'].calls == [
        (20, {'component': 'scheduler', 'loop': 'assignment', 'kind': 'scanned'}),
        (10, {'component': 'scheduler', 'loop': 'assignment', 'kind': 'claimed'}),
        (7, {'component': 'scheduler', 'loop': 'assignment', 'kind': 'changed'}),
    ]
    for instrument in meter.instruments.values():
        for _, attributes in instrument.calls:
            assert set(attributes) <= {'component', 'loop', 'outcome', 'kind', 'operation', 'reason'}


def test_control_plane_metrics_track_scheduler_and_reconciler_signals() -> None:
    meter = _Meter()
    metrics = OpenTelemetryControlPlaneMetrics(meter=meter)

    metrics.set_scheduler_owned_shards(3)
    metrics.set_scheduler_owned_shards(2)
    metrics.record_scheduler_assignments(4)
    metrics.record_scheduler_lock_contention(2)
    metrics.record_scheduler_no_capacity(5)
    metrics.record_scheduler_candidate_queries(2)
    metrics.record_scheduler_transaction_duration(duration_seconds=0.125)
    metrics.record_reconciler_overdue_lag(operation='expire_queued', lag_seconds=1.5)
    metrics.record_reconciler_repair(reason='repair_active_assignments', count=3)
    metrics.record_batch_full_scan(duration_seconds=0.4, scanned=500)
    metrics.record_batch_terminal_discovery_lag(lag_seconds=2.5)

    assert meter.instruments['nexus.distributed.control_plane.scheduler.owned_shards'].calls == [
        (3, {'component': 'scheduler'}),
        (-1, {'component': 'scheduler'}),
    ]
    assert meter.instruments['nexus.distributed.control_plane.scheduler.assignments'].calls == [(4, {'component': 'scheduler'})]
    assert meter.instruments['nexus.distributed.control_plane.scheduler.lock_contention'].calls == [(2, {'component': 'scheduler'})]
    assert meter.instruments['nexus.distributed.control_plane.scheduler.no_capacity'].calls == [(5, {'component': 'scheduler'})]
    assert meter.instruments['nexus.distributed.control_plane.scheduler.candidate_queries'].calls == [(2, {'component': 'scheduler'})]
    assert meter.instruments['nexus.distributed.control_plane.scheduler.transaction_duration'].calls == [
        (0.125, {'component': 'scheduler'})
    ]
    assert meter.instruments['nexus.distributed.control_plane.reconciler.overdue_lag'].calls == [
        (1.5, {'component': 'reconciler', 'operation': 'expire_queued'})
    ]
    assert meter.instruments['nexus.distributed.control_plane.reconciler.repairs'].calls == [
        (3, {'component': 'reconciler', 'reason': 'repair_active_assignments'})
    ]
    assert meter.instruments['nexus.distributed.control_plane.reconciler.batch_full_scan.duration'].calls == [
        (0.4, {'component': 'reconciler', 'loop': 'batch'})
    ]
    assert meter.instruments['nexus.distributed.control_plane.reconciler.batch_full_scan.items'].calls == [
        (500, {'component': 'reconciler', 'loop': 'batch'})
    ]
    assert meter.instruments['nexus.distributed.control_plane.reconciler.batch_terminal_discovery_lag'].calls == [
        (2.5, {'component': 'reconciler', 'loop': 'batch'})
    ]


@pytest.mark.asyncio
async def test_scheduler_loop_records_success_work_and_owned_shards(monkeypatch) -> None:
    metrics = _RecordingMetrics()
    service = SchedulerService(
        session_factory=None,  # type: ignore[arg-type]
        config=DistributedConfig(scheduler_poll_interval=0.001),
        owner_id='scheduler-a',
        metrics=metrics,
    )

    async def tick_once() -> int:
        service._shard_leases._owned_shards = {1, 2}
        service.stop()
        return 3

    monkeypatch.setattr(service, 'tick_once', tick_once)

    await asyncio.wait_for(service._run_assignment_loop(), timeout=1)

    assert metrics.owned_shards[-1] == 2
    assert len(metrics.samples) == 1
    assert metrics.samples[0].component == 'scheduler'
    assert metrics.samples[0].loop == 'assignment'
    assert metrics.samples[0].outcome == 'success'
    assert metrics.samples[0].changed == 3


@pytest.mark.asyncio
async def test_reconciler_loop_records_transient_failure_and_backoff() -> None:
    metrics = _RecordingMetrics()

    class EnvironmentController:
        async def reconcile_once(self) -> int:
            return 0

        async def close(self) -> None:
            pass

    class BatchLifecycle:
        async def reconcile_once(self) -> int:
            return 0

    service = ReconcilerService(
        session_factory=None,  # type: ignore[arg-type]
        config=DistributedConfig(
            reconciler_poll_interval=0.001,
            reconciler_failure_backoff=0.001,
            artifact_maintenance_enabled=False,
        ),
        environment_controller=EnvironmentController(),  # type: ignore[arg-type]
        dependency_build_controller=_NoopController(),
        workload_client_resolver=_NoopWorkloadClientResolver(),  # type: ignore[arg-type]
        batch_lifecycle=BatchLifecycle(),
        metrics=metrics,
    )
    calls = 0

    async def tick() -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ArtifactStoreTransientError(detail='temporary')
        service.stop()
        return 2

    await asyncio.wait_for(
        service._run_loop(
            loop_name='artifact-maintenance',
            tick=tick,
            poll_interval=0.001,
            timeout=1,
        ),
        timeout=1,
    )

    assert [sample.outcome for sample in metrics.samples] == ['transient_error', 'success']
    assert metrics.samples[0].backoff_seconds == 0.001
    assert metrics.samples[1].changed == 2


@pytest.mark.asyncio
async def test_reconciler_loop_propagates_permanent_database_failure() -> None:
    metrics = _RecordingMetrics()

    class EnvironmentController:
        async def reconcile_once(self) -> int:
            return 0

        async def close(self) -> None:
            pass

    class BatchLifecycle:
        async def reconcile_once(self) -> int:
            return 0

    class DatabaseError(Exception):
        pass

    service = ReconcilerService(
        session_factory=None,  # type: ignore[arg-type]
        config=DistributedConfig(
            reconciler_poll_interval=0.001,
            reconciler_failure_backoff=0.001,
            artifact_maintenance_enabled=False,
        ),
        environment_controller=EnvironmentController(),  # type: ignore[arg-type]
        dependency_build_controller=_NoopController(),
        workload_client_resolver=_NoopWorkloadClientResolver(),  # type: ignore[arg-type]
        batch_lifecycle=BatchLifecycle(),
        metrics=metrics,
    )

    async def tick() -> int:
        raise OperationalError('select 1', {}, DatabaseError(1049, 'unknown database'))

    with pytest.raises(OperationalError):
        await asyncio.wait_for(
            service._run_loop(
                loop_name='core',
                tick=tick,
                poll_interval=0.001,
                timeout=1,
            ),
            timeout=1,
        )

    assert [sample.outcome for sample in metrics.samples] == ['fatal_error']


@pytest.mark.asyncio
async def test_reconciler_environment_loop_uses_controller_work_counts() -> None:
    metrics = _RecordingMetrics()

    class EnvironmentController:
        last_environment_stats = ControlLoopWork(scanned=8, claimed=5, changed=3)

        async def reconcile_once(self) -> int:
            service.stop()
            return 3

        async def close(self) -> None:
            pass

    class BatchLifecycle:
        async def reconcile_once(self) -> int:
            return 0

        async def close(self) -> None:
            pass

    controller = EnvironmentController()
    service = ReconcilerService(
        session_factory=None,  # type: ignore[arg-type]
        config=DistributedConfig(
            reconciler_poll_interval=0.001,
            artifact_maintenance_enabled=False,
        ),
        environment_controller=controller,  # type: ignore[arg-type]
        dependency_build_controller=_NoopController(),
        workload_client_resolver=_NoopWorkloadClientResolver(),  # type: ignore[arg-type]
        batch_lifecycle=BatchLifecycle(),
        metrics=metrics,
    )

    await asyncio.wait_for(
        service._run_loop(
            loop_name='environment',
            tick=service._environment_tick_once,
            poll_interval=0.001,
            timeout=1,
        ),
        timeout=1,
    )

    assert len(metrics.samples) == 1
    sample = metrics.samples[0]
    assert (sample.component, sample.loop, sample.outcome) == ('reconciler', 'environment', 'success')
    assert (sample.scanned, sample.claimed, sample.changed) == (8, 5, 3)
