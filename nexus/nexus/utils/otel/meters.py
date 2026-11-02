"""Meter access facade.

Direct ``opentelemetry`` imports are confined to ``nexus/utils/otel`` (enforced by
``tests/unit/utils/otel/test_architecture.py``), so production code that needs a meter
goes through here instead of importing the SDK itself.

``get_meter`` covers callers that just record against the ambient provider. Callers that
own their own ``MeterProvider`` (see ``nexus/distributed/observability/telemetry.py``)
need SDK types rather than a meter, so the loaders below expose exactly those. Every
loader imports lazily, keeping this module importable without the ``otel`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from opentelemetry.metrics import Meter


def get_meter(name: str, version: str | None = None) -> Meter | _NoOpMeter:
    """Return the OpenTelemetry meter for *name*.

    Falls back to a no-op meter when the OpenTelemetry API is not installed, so that
    instrumented code stays importable without the ``otel`` extra.
    """
    try:
        from opentelemetry import metrics
    except ImportError:
        return _NoOpMeter()
    return metrics.get_meter(name, version or '')


class _NoOpInstrument:
    def add(self, *args: Any, **kwargs: Any) -> None:
        return None

    def record(self, *args: Any, **kwargs: Any) -> None:
        return None


class _NoOpMeter:
    """Minimal stand-in used when ``opentelemetry-api`` is absent."""

    def create_counter(self, *args: Any, **kwargs: Any) -> _NoOpInstrument:
        return _NoOpInstrument()

    def create_histogram(self, *args: Any, **kwargs: Any) -> _NoOpInstrument:
        return _NoOpInstrument()

    def create_up_down_counter(self, *args: Any, **kwargs: Any) -> _NoOpInstrument:
        return _NoOpInstrument()

    def create_gauge(self, *args: Any, **kwargs: Any) -> _NoOpInstrument:
        return _NoOpInstrument()


@dataclass(frozen=True)
class MetricsSdk:
    """SDK symbols needed to build a self-owned ``MeterProvider``."""

    Counter: Any
    Histogram: Any
    MeterProvider: Any
    Resource: Any
    AggregationTemporality: Any
    PeriodicExportingMetricReader: Any
    ExplicitBucketHistogramAggregation: Any
    View: Any


@dataclass(frozen=True)
class OtlpHttpMetricExporterSdk:
    """OTLP/HTTP exporter symbols plus the export result enum used to grade attempts."""

    OTLPMetricExporter: Any
    MetricExportResult: Any


def load_metrics_sdk() -> MetricsSdk:
    """Return the OpenTelemetry metrics SDK symbols."""
    from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
    from opentelemetry.sdk.metrics.export import AggregationTemporality, PeriodicExportingMetricReader
    from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
    from opentelemetry.sdk.resources import Resource

    return MetricsSdk(
        Counter=Counter,
        Histogram=Histogram,
        MeterProvider=MeterProvider,
        Resource=Resource,
        AggregationTemporality=AggregationTemporality,
        PeriodicExportingMetricReader=PeriodicExportingMetricReader,
        ExplicitBucketHistogramAggregation=ExplicitBucketHistogramAggregation,
        View=View,
    )


def load_otlp_http_metric_exporter() -> OtlpHttpMetricExporterSdk:
    """Return the OTLP/HTTP metric exporter class and its export result enum."""
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.metrics.export import MetricExportResult

    return OtlpHttpMetricExporterSdk(
        OTLPMetricExporter=OTLPMetricExporter,
        MetricExportResult=MetricExportResult,
    )


def rejected_data_points(payload: bytes) -> int:
    """Return the data points an OTLP collector rejected in a partial-success response."""
    from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceResponse

    response = ExportMetricsServiceResponse()
    response.ParseFromString(payload)
    return response.partial_success.rejected_data_points


def make_observation(value: float, attributes: dict[str, str]) -> Any:
    """Return an asynchronous-instrument observation."""
    from opentelemetry.metrics import Observation

    return Observation(value, attributes)
