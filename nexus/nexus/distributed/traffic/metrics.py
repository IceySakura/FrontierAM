from __future__ import annotations

from typing import Any, Literal, Protocol

from nexus.utils.otel.meters import get_meter


ArtifactRouteResult = Literal['success', 'not_found', 'transient_error', 'permanent_error', 'validation_error']


class ArtifactRouteMetrics(Protocol):
    def record_selected(self, *, target: str, kind: str, operation: str, policy_version: int) -> None: ...

    def record_result(
        self,
        *,
        target: str,
        kind: str,
        operation: str,
        result: ArtifactRouteResult,
        policy_version: int,
    ) -> None: ...

    def record_latency(
        self,
        *,
        target: str,
        operation: str,
        policy_version: int,
        latency_seconds: float,
    ) -> None: ...


class OpenTelemetryArtifactRouteMetrics:
    """Low-cardinality metrics for routed artifact operations."""

    def __init__(self, *, meter: Any | None = None) -> None:
        meter = meter or get_meter('nexus.distributed.artifact')
        self._selected = meter.create_counter(
            'nexus.distributed.artifact.route.selected',
            unit='{selection}',
        )
        self._result = meter.create_counter(
            'nexus.distributed.artifact.route.result',
            unit='{operation}',
        )
        self._latency = meter.create_histogram(
            'nexus.distributed.artifact.route.latency',
            unit='s',
        )

    def record_selected(self, *, target: str, kind: str, operation: str, policy_version: int) -> None:
        self._selected.add(1, attributes=_attributes(target, kind, operation, policy_version))

    def record_result(
        self,
        *,
        target: str,
        kind: str,
        operation: str,
        result: ArtifactRouteResult,
        policy_version: int,
    ) -> None:
        self._result.add(1, attributes={**_attributes(target, kind, operation, policy_version), 'result': result})

    def record_latency(
        self,
        *,
        target: str,
        operation: str,
        policy_version: int,
        latency_seconds: float,
    ) -> None:
        if latency_seconds < 0:
            raise ValueError('artifact route latency must be non-negative')
        self._latency.record(
            latency_seconds,
            attributes={
                'target': target,
                'operation': operation,
                'policy_version': str(policy_version),
            },
        )


def _attributes(target: str, kind: str, operation: str, policy_version: int) -> dict[str, str]:
    return {
        'target': target,
        'kind': kind,
        'operation': operation,
        'policy_version': str(policy_version),
    }


__all__ = ['ArtifactRouteMetrics', 'ArtifactRouteResult', 'OpenTelemetryArtifactRouteMetrics']
