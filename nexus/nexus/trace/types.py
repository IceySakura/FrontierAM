from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


RUN_ID_ATTRIBUTE = '_run_id'
RUN_SPAN_ID_ATTRIBUTE = '_run_span_id'
RUN_KIND_ATTRIBUTE = '_run_kind'


def run_attribution_attributes(
    *,
    run_id: str | None = None,
    run_span_id: str | None = None,
    run_kind: str | None = None,
) -> dict[str, str]:
    """Build the private attributes used to route spans to a run."""
    values = (
        (RUN_ID_ATTRIBUTE, run_id),
        (RUN_SPAN_ID_ATTRIBUTE, run_span_id),
        (RUN_KIND_ATTRIBUTE, run_kind),
    )
    return {key: value for key, value in values if value is not None}


class Event(BaseModel):
    time_unix_nano: int = Field(..., description='time, in nanoseconds since epoch')
    name: str = Field(..., description='event name')
    attributes: dict[str, Any] = Field(..., description='attributes')


class Status(BaseModel):
    code: Literal['UNSET', 'OK', 'ERROR'] = Field('UNSET', description='status code')
    message: str | None = Field(None, description='status message')


# this is only for reference as pydantic doesn't play nice with TypeDict before python 3.12
class Attributes(TypedDict, total=False):
    # span type, e.g. root, llm, tool, task, etc.
    type: str
    hostname: str
    process_id: int
    thread_id: int
    task_id: int
    inputs: dict
    outputs: dict
    project_id: str


# https://github.com/open-telemetry/opentelemetry-proto/blob/main/opentelemetry/proto/trace/v1/trace.proto#L88
class Span(BaseModel):
    trace_id: str = Field(..., description='trace id, 16 bytes = 22 base64 encoded chars')
    span_id: str = Field(..., description='span id, 8 bytes = 11 base64 encoded chars')
    parent_span_id: str = Field(..., description='parent span id')
    name: str = Field(..., description='span name')
    start_time_unix_nano: int = Field(..., description='start time, in nanoseconds since epoch')
    end_time_unix_nano: int | None = Field(None, description='end time, in nanoseconds since epoch')
    attributes: dict = Field(default_factory=dict, description='attributes')
    events: list[Event] = Field(default_factory=list, description='events')
    status: Status = Field(default_factory=Status, description='status')


class RemoteTraceContext(BaseModel):
    """Trace context for distributed tracing across processes.

    Used to propagate trace information from client to server,
    so server-side spans can be linked as children of client spans.
    """

    trace_id: str
    parent_span_id: str
    run_id: str | None = None
    run_span_id: str | None = None
    run_kind: str | None = None
    otel_carrier: dict[str, str] | None = None

    @classmethod
    def from_current_span(cls) -> 'RemoteTraceContext | None':
        """Create RemoteTraceContext from the current Nexus or native OTel span.

        Returns:
            RemoteTraceContext if a current Nexus or native OTel span exists, None otherwise.
        """
        from nexus.trace.tracer import get_current_span
        from nexus.utils.otel.propagation import inject_carrier

        span = get_current_span()
        carrier = inject_carrier() or None
        if span:
            return cls(
                trace_id=span.trace_id,
                parent_span_id=span.span_id,
                run_id=span.attributes.get(RUN_ID_ATTRIBUTE),
                run_span_id=span.attributes.get(RUN_SPAN_ID_ATTRIBUTE),
                run_kind=span.attributes.get(RUN_KIND_ATTRIBUTE),
                otel_carrier=carrier,
            )

        if carrier is None:
            return None
        try:
            _version, trace_id, parent_span_id, _flags = carrier['traceparent'].split('-')
            from nexus.utils.otel.ids import otel_hex_to_nexus_span_id, otel_hex_to_nexus_trace_id

            nexus_trace_id = otel_hex_to_nexus_trace_id(trace_id)
            nexus_parent_span_id = otel_hex_to_nexus_span_id(parent_span_id)
        except (KeyError, ValueError):
            return None

        return cls(
            trace_id=nexus_trace_id,
            parent_span_id=nexus_parent_span_id,
            otel_carrier=carrier,
        )
