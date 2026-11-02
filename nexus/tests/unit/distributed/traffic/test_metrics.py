from __future__ import annotations

from nexus.distributed.traffic.metrics import OpenTelemetryArtifactRouteMetrics


class _Instrument:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, str]]] = []

    def add(self, value: object, *, attributes: dict[str, str]) -> None:
        self.calls.append((value, attributes))

    def record(self, value: object, *, attributes: dict[str, str]) -> None:
        self.calls.append((value, attributes))


class _Meter:
    def __init__(self) -> None:
        self.instruments: dict[str, _Instrument] = {}

    def create_counter(self, name: str, **kwargs: object) -> _Instrument:
        instrument = _Instrument()
        self.instruments[name] = instrument
        return instrument

    def create_histogram(self, name: str, **kwargs: object) -> _Instrument:
        instrument = _Instrument()
        self.instruments[name] = instrument
        return instrument


def test_artifact_route_metrics_use_only_bounded_labels() -> None:
    meter = _Meter()
    metrics = OpenTelemetryArtifactRouteMetrics(meter=meter)

    metrics.record_selected(target='artifact_service', kind='run_result', operation='put', policy_version=2)
    metrics.record_result(
        target='artifact_service',
        kind='run_result',
        operation='put',
        result='success',
        policy_version=2,
    )
    metrics.record_latency(target='artifact_service', operation='put', policy_version=2, latency_seconds=0.25)

    assert meter.instruments['nexus.distributed.artifact.route.selected'].calls == [
        (1, {'target': 'artifact_service', 'kind': 'run_result', 'operation': 'put', 'policy_version': '2'})
    ]
    for instrument in meter.instruments.values():
        for _, attributes in instrument.calls:
            assert set(attributes) <= {'target', 'kind', 'operation', 'result', 'policy_version'}
            assert all(value not in attributes.values() for value in ('secret-token', 'owner/secret'))
