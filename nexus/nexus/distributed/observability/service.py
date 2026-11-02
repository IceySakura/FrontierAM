from __future__ import annotations

import time

from contextlib import asynccontextmanager

from nexus.distributed.observability.database import install_database_metrics
from nexus.distributed.observability.metrics import OpenTelemetryControlPlaneMetrics
from nexus.distributed.observability.telemetry import PlatformMetrics
from nexus.utils.platform_metrics import metric_scope, process_recorder, set_process_recorder


@asynccontextmanager
async def service_metrics(config, service, session_factory):
    runtime = PlatformMetrics(config.metrics, service=service)
    previous = process_recorder()
    set_process_recorder(runtime)
    install_database_metrics(session_factory, runtime, slow_seconds=config.slow_sql_log_threshold_ms / 1000)
    try:
        with metric_scope(runtime):
            yield runtime
    finally:
        if process_recorder() is runtime:
            set_process_recorder(previous)
        await runtime.close()


class _Instrument:
    def __init__(self, runtime, name, kind):
        self.runtime = runtime
        self.name = name
        self.kind = kind
        self.value = 0

    def add(self, value, attributes=None):
        if self.kind == 'gauge':
            self.value += value
            self.runtime.record(self.name, self.value, attributes)
        else:
            self.runtime.record(self.name, value, attributes)

    def record(self, value, attributes=None):
        self.runtime.record(self.name, value, attributes)


class _Meter:
    def __init__(self, runtime):
        self.runtime = runtime

    @staticmethod
    def _name(name):
        return name.removeprefix('nexus.distributed.').removeprefix('control_plane.').replace('.', '_')

    def create_counter(self, name, **_kwargs):
        return _Instrument(self.runtime, self._name(name), 'counter')

    def create_histogram(self, name, **_kwargs):
        return _Instrument(self.runtime, self._name(name) + '_distribution', 'histogram')

    def create_up_down_counter(self, name, **_kwargs):
        mapped = self._name(name)
        self.runtime.record(mapped, 0)
        return _Instrument(self.runtime, mapped, 'gauge')


class PlatformControlPlaneMetrics(OpenTelemetryControlPlaneMetrics):
    def __init__(self, runtime: PlatformMetrics):
        super().__init__(meter=_Meter(runtime))
        self.runtime = runtime

    def record_loop(self, sample):
        super().record_loop(sample)
        if sample.outcome == 'success':
            self.runtime.record(
                'control_loop_success_timestamp', time.time(), {'component': sample.component, 'loop': sample.loop}
            )
