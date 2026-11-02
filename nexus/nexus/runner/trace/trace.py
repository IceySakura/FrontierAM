"""Runner-level trace processing, sessions, streams, and sink contracts."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import logging
import threading
import time

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from nexus.trace.controller import SpanController
from nexus.trace.events import SpanEndEvent, SpanEvent, SpanStartEvent, SpanUpdateEvent
from nexus.trace.io import TraceSink, TraceSource
from nexus.trace.processors import SpanProcessor
from nexus.trace.records import SpanEventRecord, span_event_record_from_span
from nexus.trace.scopes import DEFAULT_SCOPE, RUN_SCOPE
from nexus.trace.types import (
    RUN_ID_ATTRIBUTE,
    RUN_KIND_ATTRIBUTE,
    RUN_SPAN_ID_ATTRIBUTE,
    Event,
    Span,
    run_attribution_attributes,
)
from nexus.utils.common import random_urlsafe_id
from nexus.utils.platform_metrics import measured, record


logger = logging.getLogger(__name__)

RUN_KINDS = frozenset({'live', 'single', 'batch'})
RUN_SPAN_NAMES = frozenset(f'run.{kind}' for kind in RUN_KINDS)
RUN_TRACE_METADATA_VERSION = 4
_PRINCIPAL_TRACE_SCOPE_DOMAIN = b'nexus-tdsql-principal-table:v1\0'
_PRINCIPAL_ID_MAX_LENGTH = 64


def principal_trace_scope_id(owner_principal_id: str) -> str:
    """Return the opaque stable trace-storage scope for one persisted owner."""
    if (
        not isinstance(owner_principal_id, str)
        or not owner_principal_id
        or len(owner_principal_id) > _PRINCIPAL_ID_MAX_LENGTH
        or not owner_principal_id.isascii()
        or any(
            character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-'
            for character in owner_principal_id
        )
    ):
        raise ValueError('owner_principal_id must be a bounded ASCII identity')
    return hashlib.sha256(_PRINCIPAL_TRACE_SCOPE_DOMAIN + owner_principal_id.encode('utf-8')).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class RunTraceContext:
    """Stable identity and caller metadata for one run trace."""

    run_id: str
    run_span_id: str
    run_kind: str
    metadata: dict[str, Any] = field(default_factory=dict)
    result_artifact_available: bool = False


@dataclass(frozen=True, slots=True)
class RunTraceScope:
    """A storage scope that selected trace backends may prepare before a run fan-out."""

    run_kind: str
    scope_id: str

    def __post_init__(self) -> None:
        _validate_run_kind(self.run_kind)
        if not isinstance(self.scope_id, str) or not self.scope_id:
            raise ValueError('run trace scope_id must be a non-empty string')


@runtime_checkable
class RunTraceScopePreparer(Protocol):
    async def prepare_scope(self, scope: RunTraceScope) -> None:
        """Idempotently prepare backend storage for one selected scope."""

        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RunTraceProjection:
    """One reader-facing value projected from raw span records."""

    type: str
    data: dict[str, Any]
    time_unix_nano: int


@dataclass(frozen=True, slots=True)
class RunTraceStreamEvent(RunTraceProjection):
    """One cursor-addressable projection stream event."""

    id: int


@dataclass(frozen=True, slots=True)
class RunEvent:
    """One decoded run lifecycle event from the synthetic run span."""

    type: str
    data: dict[str, Any]
    time_unix_nano: int
    raw_record_id: int


class RunTraceSink(TraceSink[RunTraceContext, SpanEventRecord], ABC):
    """Explicit base class for every raw run trace sink."""

    @abstractmethod
    async def start(self, context: RunTraceContext) -> None:
        """Prepare this sink for one run."""

    @abstractmethod
    async def on_event(self, context: RunTraceContext, event: SpanEventRecord) -> None:
        """Accept one ordered raw span event record."""

    @abstractmethod
    async def close(self, context: RunTraceContext) -> None:
        """Flush all accepted records and close the sink."""

    def metadata(self) -> Mapping[str, Any]:
        """Return fast, JSON-serializable metadata after a successful close."""
        return {}


@runtime_checkable
class RunResultSink(Protocol):
    """A result publisher whose lifecycle is independent of raw trace sinks."""

    def capture_output(self, output: Mapping[str, Any]) -> None: ...

    async def publish_result(self, *, status: str, error: Mapping[str, Any] | None = None) -> None: ...

    def is_durable(self) -> bool: ...

    def metadata(self) -> Mapping[str, Any]: ...


class RunTraceSource(TraceSource[Mapping[str, Any], SpanEventRecord], ABC):
    """Explicit base class for every persisted run trace source."""

    @abstractmethod
    async def read_events_after(
        self,
        metadata: Mapping[str, Any],
        cursor: int,
        *,
        limit: int | None = None,
    ) -> list[SpanEventRecord]:
        """Read ordered raw records after a raw cursor."""

    @abstractmethod
    async def read_result(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        """Recover the terminal RunOutput from raw run events."""


class RunTraceSourceUnavailable(RuntimeError):  # noqa: N818
    """The source cannot currently serve an artifact, so fallback is allowed."""


class RunTraceRecordStream(ABC):
    """Active cursorable stream of raw run trace records."""

    @property
    @abstractmethod
    def latest_record_id(self) -> int:
        """Return the newest available raw cursor."""

    @abstractmethod
    def snapshot_after(self, cursor: int, *, limit: int | None = None) -> list[SpanEventRecord]:
        """Return currently buffered records after a cursor."""

    @abstractmethod
    async def wait_after(
        self,
        cursor: int,
        *,
        timeout: float | None = None,
        limit: int | None = None,
    ) -> list[SpanEventRecord]:
        """Wait until records are available after a cursor."""


class RunTraceProjectionStream(ABC):
    """Active cursorable stream of projected run trace values."""

    @property
    @abstractmethod
    def latest_id(self) -> int:
        """Return the newest projection cursor."""

    @abstractmethod
    def snapshot_after(self, cursor: int, *, limit: int | None = None) -> list[RunTraceStreamEvent]:
        """Return projected values after a cursor."""

    @abstractmethod
    async def wait_after(
        self,
        cursor: int,
        *,
        timeout: float | None = None,
        limit: int | None = None,
    ) -> list[RunTraceStreamEvent]:
        """Wait until projected values are available after a cursor."""


class RunTraceProjector(ABC):
    """Project a reconstructed raw trace record into zero or more API values."""

    @abstractmethod
    def project(
        self,
        context: RunTraceContext,
        span: Span,
        event: SpanEvent,
        record: SpanEventRecord,
    ) -> Sequence[RunTraceProjection]:
        """Project one raw span event."""


class RunTraceCursorExpired(RuntimeError):  # noqa: N818
    """The requested in-memory raw cursor has already been evicted."""


class RunTraceProcessingError(RuntimeError):
    """The per-run trace worker could not accept or process a record."""


@dataclass(frozen=True, slots=True)
class _SinkFailure:
    sink: RunTraceSink
    stage: str
    error: BaseException


class RunTraceSinkError(RuntimeError):
    """One or more enabled run trace sinks failed."""

    def __init__(self, failures: Sequence[_SinkFailure]) -> None:
        self.failures = tuple(failures)
        details = '; '.join(f'{item.stage}: {type(item.error).__name__}: {item.error}' for item in failures)
        super().__init__(f'run trace sink failed: {details}')


class RunTraceInMemoryJournal(RunTraceSink, RunTraceRecordStream):
    """Bounded active raw journal used by control-plane consumers."""

    def __init__(self, *, max_records: int = 10_000) -> None:
        if max_records < 1:
            raise ValueError('max_records must be at least 1')
        self._records: deque[SpanEventRecord] = deque(maxlen=max_records)
        self._condition = asyncio.Condition()
        self._started = False
        self._closed = False

    @property
    def latest_record_id(self) -> int:
        return self._records[-1].record_id if self._records else 0

    async def start(self, context: RunTraceContext) -> None:  # noqa: ARG002
        if self._started:
            raise RuntimeError('run trace journal is already started')
        self._started = True

    async def on_event(self, context: RunTraceContext, event: SpanEventRecord) -> None:  # noqa: ARG002
        if not self._started or self._closed:
            raise RuntimeError('run trace journal is not active')
        async with self._condition:
            self._records.append(event)
            self._condition.notify_all()

    async def close(self, context: RunTraceContext) -> None:  # noqa: ARG002
        async with self._condition:
            self._closed = True
            self._condition.notify_all()

    def snapshot_after(self, cursor: int, *, limit: int | None = None) -> list[SpanEventRecord]:
        if cursor < 0:
            raise ValueError('cursor must not be negative')
        if limit is not None and limit < 1:
            raise ValueError('limit must be at least 1')
        if self._records and cursor < self._records[0].record_id - 1:
            raise RunTraceCursorExpired(f'run trace cursor {cursor} is older than {self._records[0].record_id - 1}')
        records = [record for record in self._records if record.record_id > cursor]
        return records[:limit] if limit is not None else records

    async def wait_after(
        self,
        cursor: int,
        *,
        timeout: float | None = None,
        limit: int | None = None,
    ) -> list[SpanEventRecord]:
        available = self.snapshot_after(cursor, limit=limit)
        if available or self._closed:
            return available

        async def wait() -> None:
            async with self._condition:
                await self._condition.wait_for(lambda: self.latest_record_id > cursor or self._closed)

        try:
            if timeout is None:
                await wait()
            else:
                await asyncio.wait_for(wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return []
        return self.snapshot_after(cursor, limit=limit)


def iter_run_events(record: SpanEventRecord, run_span_id: str) -> Iterator[RunEvent]:
    """Decode lifecycle events from one raw synthetic run-span record."""
    if record.span_id != run_span_id or record.event_type != 'update':
        return
    for item in record.event_data.get('events') or []:
        if not isinstance(item, Mapping) or item.get('name') != 'run.event':
            continue
        attributes = item.get('attributes')
        if not isinstance(attributes, Mapping):
            continue
        event_type = attributes.get('event_type')
        event_data = attributes.get('event_data')
        if not isinstance(event_type, str) or not isinstance(event_data, Mapping):
            continue
        event_time = item.get('time_unix_nano')
        yield RunEvent(
            type=event_type,
            data=dict(event_data),
            time_unix_nano=event_time if isinstance(event_time, int) else record.time_unix_nano,
            raw_record_id=record.record_id,
        )


@dataclass(frozen=True, slots=True)
class _QueuedSpanEvent:
    sequence: int
    span: Span
    event: SpanEvent


@dataclass(slots=True)
class _RunTraceState:
    context: RunTraceContext
    loop: asyncio.AbstractEventLoop
    wake_event: asyncio.Event
    processed_condition: asyncio.Condition
    sinks: tuple[RunTraceSink, ...]
    lock: threading.Lock = field(default_factory=threading.Lock)
    queue: deque[_QueuedSpanEvent] = field(default_factory=deque)
    started_sinks: list[RunTraceSink] = field(default_factory=list)
    cleanup_sinks: list[RunTraceSink] = field(default_factory=list)
    failed_sink_ids: set[int] = field(default_factory=set)
    failures: list[_SinkFailure] = field(default_factory=list)
    consumer_task: asyncio.Task[None] | None = None
    worker_task: asyncio.Task[None] | None = None
    close_task: asyncio.Task[None] | None = None
    consumer_cancel_requested: bool = False
    accepting_events: bool = False
    wake_scheduled: bool = False
    closing: bool = False
    accepted_sequence: int = 0
    processed_sequence: int = 0
    processing_error: BaseException | None = None


class RunTraceProcessor(SpanProcessor):
    """Attribute raw span callbacks to active runs and fan them out to sinks."""

    processor_scopes = frozenset({DEFAULT_SCOPE, RUN_SCOPE})

    def __init__(self, *, max_queued_events: int = 100_000) -> None:
        if max_queued_events < 1:
            raise ValueError('max_queued_events must be at least 1')
        self._max_queued_events = max_queued_events
        self._registry_lock = threading.RLock()
        self._states_by_run_span_id: dict[str, _RunTraceState] = {}

    async def register_run(self, *, context: RunTraceContext, sinks: Sequence[RunTraceSink]) -> None:
        _validate_run_kind(context.run_kind)
        if not sinks:
            raise ValueError('run trace requires at least one sink')
        loop = asyncio.get_running_loop()
        state = _RunTraceState(
            context=context,
            loop=loop,
            wake_event=asyncio.Event(),
            processed_condition=asyncio.Condition(),
            sinks=tuple(sinks),
        )
        with self._registry_lock:
            if context.run_span_id in self._states_by_run_span_id:
                raise ValueError(f'run trace span is already registered: {context.run_span_id}')
            self._states_by_run_span_id[context.run_span_id] = state

        try:
            for sink in state.sinks:
                state.cleanup_sinks.append(sink)
                try:
                    await _await_sink_lifecycle(sink.start(context), stage='start')
                    state.started_sinks.append(sink)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    state.failures.append(_SinkFailure(sink=sink, stage='start', error=exc))
                    raise RunTraceSinkError(state.failures) from exc
            state.consumer_task = loop.create_task(
                self._worker_loop(state),
                name=f'run-trace-consumer:{context.run_span_id}',
            )
            state.worker_task = loop.create_task(
                self._supervise_worker(state),
                name=f'run-trace:{context.run_span_id}',
            )
            with state.lock:
                state.accepting_events = True
        except BaseException:
            rollback = loop.create_task(
                self._rollback_registration(state), name=f'run-trace-rollback:{context.run_span_id}'
            )
            with state.lock:
                state.accepting_events = False
                state.closing = True
                state.close_task = rollback
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(rollback)
            raise

    def on_start(self, span: Span, event: SpanStartEvent) -> None:
        self._accept(span, event)

    def on_update(self, span: Span, event: SpanUpdateEvent) -> None:
        self._accept(span, event)

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        self._accept(span, event)

    @measured('trace_flush', family='persistence')
    async def flush_run(self, run_span_id: str) -> None:
        state = self._get_state(run_span_id)
        if state is None:
            return
        self._require_owner_loop(state)
        with state.lock:
            target = state.accepted_sequence
        async with state.processed_condition:
            await state.processed_condition.wait_for(lambda: self._processed_at_least(state, target))
        self._raise_state_error(state)

    async def close_run(self, run_span_id: str) -> None:
        state = self._get_state(run_span_id)
        if state is None:
            return
        self._require_owner_loop(state)
        should_wake = False
        with state.lock:
            if not state.closing:
                state.accepting_events = False
                state.closing = True
                if not state.wake_scheduled:
                    state.wake_scheduled = True
                    should_wake = True
                state.close_task = state.loop.create_task(
                    self._close_run_state(state),
                    name=f'run-trace-close:{run_span_id}',
                )
            close_task = state.close_task
        if should_wake:
            state.wake_event.set()
        if close_task is None:
            raise RuntimeError('run trace close task was not created')
        await asyncio.shield(close_task)

    def _accept(self, span: Span, event: SpanEvent) -> None:
        state = self._state_for_span(span)
        if state is None:
            return
        try:
            queued = _QueuedSpanEvent(
                sequence=0,
                span=_compact_span_snapshot(span, event),
                event=event.model_copy(deep=True),
            )
            should_wake = False
            with state.lock:
                if not state.accepting_events:
                    return
                if len(state.queue) >= self._max_queued_events:
                    raise RunTraceProcessingError(
                        f'run trace queue exceeded {self._max_queued_events} events for {state.context.run_id}'
                    )
                state.accepted_sequence += 1
                state.queue.append(
                    _QueuedSpanEvent(sequence=state.accepted_sequence, span=queued.span, event=queued.event)
                )
                if not state.wake_scheduled:
                    state.wake_scheduled = True
                    should_wake = True
            if should_wake:
                state.loop.call_soon_threadsafe(state.wake_event.set)
        except Exception as exc:
            self._set_processing_error(state, exc)
            logger.exception('Failed to enqueue run trace event')

    async def _worker_loop(self, state: _RunTraceState) -> None:
        while True:
            await state.wake_event.wait()
            while True:
                with state.lock:
                    if state.queue:
                        queued = state.queue.popleft()
                        should_close = False
                    else:
                        queued = None
                        state.wake_event.clear()
                        state.wake_scheduled = False
                        should_close = state.closing
                if queued is None:
                    break
                try:
                    await self._process_queued_event(state, queued)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._set_processing_error(state, exc)
                    logger.exception('Failed to process run trace event')
                finally:
                    async with state.processed_condition:
                        with state.lock:
                            state.processed_sequence = max(state.processed_sequence, queued.sequence)
                        state.processed_condition.notify_all()
            if should_close:
                return

    async def _process_queued_event(self, state: _RunTraceState, queued: _QueuedSpanEvent) -> None:
        record = span_event_record_from_span(record_id=queued.sequence, span=queued.span, event=queued.event)
        for sink in state.started_sinks:
            if id(sink) in state.failed_sink_ids:
                continue
            try:
                await sink.on_event(state.context, record.model_copy(deep=True))
            except asyncio.CancelledError as exc:
                with state.lock:
                    cancel_requested = state.consumer_cancel_requested
                if cancel_requested:
                    raise
                state.failed_sink_ids.add(id(sink))
                state.failures.append(_SinkFailure(sink=sink, stage='write', error=exc))
                logger.warning('Run trace sink %s failed during write', sink, exc_info=exc)
            except Exception as exc:
                state.failed_sink_ids.add(id(sink))
                state.failures.append(_SinkFailure(sink=sink, stage='write', error=exc))
                logger.warning('Run trace sink %s failed during write', sink, exc_info=exc)

    async def _supervise_worker(self, state: _RunTraceState) -> None:
        consumer_task = state.consumer_task
        if consumer_task is None:
            raise RuntimeError('run trace consumer task was not created')
        try:
            await asyncio.shield(consumer_task)
        except asyncio.CancelledError:
            if consumer_task.cancelled():
                with state.lock:
                    state.accepting_events = False
                self._set_processing_error(state, RuntimeError('run trace consumer was unexpectedly cancelled'))
                await self._notify_processing_waiters(state)
                return
            with state.lock:
                state.consumer_cancel_requested = True
            consumer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consumer_task
            raise
        except Exception as exc:
            with state.lock:
                state.accepting_events = False
            self._set_processing_error(state, exc)
            await self._notify_processing_waiters(state)

    @measured('trace_close', family='persistence')
    async def _close_run_state(self, state: _RunTraceState) -> None:
        try:
            if state.worker_task is not None:
                try:
                    await state.worker_task
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._set_processing_error(state, exc)
            await self._close_sinks(state)
        finally:
            self._remove_state(state)
        self._raise_state_error(state)

    async def _rollback_registration(self, state: _RunTraceState) -> None:
        try:
            await self._close_sinks(state)
        finally:
            self._remove_state(state)

    async def _close_sinks(self, state: _RunTraceState) -> None:
        for sink in state.cleanup_sinks:
            try:
                await _await_sink_lifecycle(sink.close(state.context), stage='close')
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                state.failures.append(_SinkFailure(sink=sink, stage='close', error=exc))

    @staticmethod
    async def _notify_processing_waiters(state: _RunTraceState) -> None:
        async with state.processed_condition:
            state.processed_condition.notify_all()

    def _state_for_span(self, span: Span) -> _RunTraceState | None:
        if span.name in RUN_SPAN_NAMES:
            state = self._get_state(span.span_id)
            if state is None:
                return None
            if span.attributes.get('run_id') != state.context.run_id:
                return None
            if span.attributes.get('run_kind') != state.context.run_kind:
                return None
            return state

        run_span_id = span.attributes.get(RUN_SPAN_ID_ATTRIBUTE)
        if not isinstance(run_span_id, str):
            return None
        state = self._get_state(run_span_id)
        if state is None:
            return None
        if span.attributes.get(RUN_ID_ATTRIBUTE) != state.context.run_id:
            return None
        if span.attributes.get(RUN_KIND_ATTRIBUTE) != state.context.run_kind:
            return None
        return state

    def _get_state(self, run_span_id: str) -> _RunTraceState | None:
        with self._registry_lock:
            return self._states_by_run_span_id.get(run_span_id)

    def _remove_state(self, state: _RunTraceState) -> None:
        with self._registry_lock:
            if self._states_by_run_span_id.get(state.context.run_span_id) is state:
                self._states_by_run_span_id.pop(state.context.run_span_id, None)

    @staticmethod
    def _processed_at_least(state: _RunTraceState, target: int) -> bool:
        with state.lock:
            return state.processed_sequence >= target or state.processing_error is not None

    @staticmethod
    def _set_processing_error(state: _RunTraceState, error: BaseException) -> None:
        with state.lock:
            if state.processing_error is None:
                state.processing_error = error
                record('data_integrity_errors', source='trace_processor', reason='processing_failed')

    @staticmethod
    def _raise_state_error(state: _RunTraceState) -> None:
        if state.failures:
            raise RunTraceSinkError(state.failures)
        if state.processing_error is not None:
            raise RunTraceProcessingError(str(state.processing_error)) from state.processing_error

    @staticmethod
    def _require_owner_loop(state: _RunTraceState) -> None:
        if asyncio.get_running_loop() is not state.loop:
            raise RuntimeError('run trace lifecycle methods must execute on the owner event loop')


class RunTraceSession:
    """Own the root run span, enabled sinks, activation context, and close lifecycle."""

    def __init__(
        self,
        *,
        run_id: str,
        run_kind: str,
        processor: RunTraceProcessor,
        backend_sinks: Mapping[str, RunTraceSink],
        internal_sinks: Sequence[RunTraceSink] = (),
        span_controller: SpanController | None = None,
        run_span_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _validate_run_kind(run_kind)
        if not backend_sinks:
            raise ValueError('run trace requires at least one backend sink')
        if any(not name for name in backend_sinks):
            raise ValueError('run trace backend sink names must not be empty')
        all_sinks = [*backend_sinks.values(), *internal_sinks]
        if len({id(sink) for sink in all_sinks}) != len(all_sinks):
            raise ValueError('run trace sinks must not be duplicated')
        self.run_id = run_id
        self.run_kind = run_kind
        self.run_span_id = run_span_id or random_urlsafe_id(8)
        self._result_sinks = {name: sink for name, sink in backend_sinks.items() if isinstance(sink, RunResultSink)}
        self.context = RunTraceContext(
            run_id=run_id,
            run_span_id=self.run_span_id,
            run_kind=run_kind,
            metadata=dict(metadata or {}),
            result_artifact_available='artifact' in self._result_sinks,
        )
        self.processor = processor
        self.span_controller = span_controller or SpanController.get_instance_sync()
        self.backend_sinks = dict(backend_sinks)
        self.internal_sinks = tuple(internal_sinks)
        self._all_sinks = tuple(all_sinks)
        self._started = False
        self._run_span_started = False
        self._closing = False
        self._closed = False
        self._close_succeeded = False
        self._close_task: asyncio.Task[None] | None = None
        self._artifact_envelope: dict[str, Any] | None = None
        self._lifecycle_lock = asyncio.Lock()

    def capture_output(self, output: Mapping[str, Any]) -> None:
        for sink in self._result_sinks.values():
            sink.capture_output(output)

    async def publish_result(self, *, status: str, error: Mapping[str, Any] | None = None) -> None:
        for sink in self._result_sinks.values():
            await sink.publish_result(status=status, error=error)

    def result_artifact_metadata(self) -> dict[str, Any]:
        entries = [
            {'name': name, 'metadata': dict(sink.metadata())}
            for name, sink in self._result_sinks.items()
            if sink.is_durable()
        ]
        if not entries:
            return {}
        return {
            'version': RUN_TRACE_METADATA_VERSION,
            'run_id': self.run_id,
            'run_span_id': self.run_span_id,
            'run_kind': self.run_kind,
            'sinks': entries,
            'trace_complete': False,
        }

    @contextlib.contextmanager
    def activate(self):
        """Make this active run span the parent of task work in the context."""
        if not self._started or self._closing or self._closed:
            raise RuntimeError('run trace session is not active')
        from nexus.trace.tracer import run_span_parent_context, run_trace_context

        with run_trace_context(self.run_id, self.run_span_id, self.run_kind):
            with run_span_parent_context(self.run_span_id):
                yield

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._closing or self._closed:
                raise RuntimeError('run trace session is closing or closed')
            if self._started:
                return
            registered = False
            try:
                await self.processor.register_run(context=self.context, sinks=self._all_sinks)
                registered = True
                self.span_controller.add_event(
                    SpanStartEvent(
                        span_id=self.run_span_id,
                        parent_span_id='',
                        name=f'run.{self.run_kind}',
                        attributes={
                            'run_id': self.run_id,
                            'run_kind': self.run_kind,
                            'type': f'{self.run_kind}_run',
                            **run_attribution_attributes(
                                run_id=self.run_id,
                                run_span_id=self.run_span_id,
                                run_kind=self.run_kind,
                            ),
                        },
                    ),
                    processor_scope=RUN_SCOPE,
                )
                self._run_span_started = True
                await self.processor.flush_run(self.run_span_id)
            except BaseException:
                self._closing = True
                if self._run_span_started:
                    with contextlib.suppress(BaseException):
                        self.span_controller.add_event(
                            SpanEndEvent(span_id=self.run_span_id), processor_scope=RUN_SCOPE
                        )
                    self._run_span_started = False
                if registered:
                    with contextlib.suppress(BaseException):
                        await self.processor.close_run(self.run_span_id)
                self._closed = True
                raise
            self._started = True

    async def emit(self, event_type: str, **data: Any) -> None:
        async with self._lifecycle_lock:
            if not self._started or self._closing or self._closed:
                raise RuntimeError('run trace session is not active')
            self.span_controller.add_event(
                SpanUpdateEvent(
                    span_id=self.run_span_id,
                    events=[
                        Event(
                            time_unix_nano=int(time.time() * 1e9),
                            name='run.event',
                            attributes={'event_type': event_type, 'event_data': data},
                        )
                    ],
                ),
                processor_scope=RUN_SCOPE,
            )
            await self.processor.flush_run(self.run_span_id)

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._close_task is not None:
                close_task = self._close_task
            elif self._closed:
                if not self._close_succeeded:
                    raise RuntimeError('run trace session did not close successfully')
                return
            elif not self._started:
                self._closing = True
                self._closed = True
                raise RuntimeError('run trace session was not started')
            else:
                self._closing = True
                close_task = asyncio.create_task(self._close_impl(), name=f'run-trace-close:{self.run_span_id}')
                self._close_task = close_task
        await asyncio.shield(close_task)

    async def _close_impl(self) -> None:
        try:
            if self._run_span_started:
                self.span_controller.add_event(SpanEndEvent(span_id=self.run_span_id), processor_scope=RUN_SCOPE)
                self._run_span_started = False
            await self.processor.close_run(self.run_span_id)
            self._artifact_envelope = self._build_artifact_envelope()
            self._close_succeeded = True
        finally:
            self._closed = True

    def artifact_metadata(self) -> dict[str, Any]:
        """Return backend source routing metadata after a successful close."""
        if not self._close_succeeded:
            raise RuntimeError('run trace artifact metadata is unavailable before close')
        if self._artifact_envelope is None:
            raise RuntimeError('run trace artifact metadata is unavailable')
        return copy.deepcopy(self._artifact_envelope)

    def _build_artifact_envelope(self) -> dict[str, Any]:
        sink_entries: list[dict[str, Any]] = []
        failures: list[_SinkFailure] = []
        for name, sink in self.backend_sinks.items():
            try:
                metadata = dict(sink.metadata())
                json.dumps(metadata)
                sink_entries.append({'name': name, 'metadata': metadata})
            except asyncio.CancelledError as exc:
                failures.append(_SinkFailure(sink=sink, stage='metadata', error=exc))
            except Exception as exc:
                failures.append(_SinkFailure(sink=sink, stage='metadata', error=exc))
        if failures:
            raise RunTraceSinkError(failures)
        return {
            'version': RUN_TRACE_METADATA_VERSION,
            'run_id': self.run_id,
            'run_span_id': self.run_span_id,
            'run_kind': self.run_kind,
            'sinks': sink_entries,
        }


def _compact_span_snapshot(span: Span, event: SpanEvent) -> Span:
    # SpanUpdate events fire once per streaming delta. The downstream sink path only reads
    # span identity (``trace_id`` / ``span_id`` / ``parent_span_id``) — the record itself
    # carries the update's own ``events`` / ``attributes`` / ``status`` from
    # ``event.model_dump()`` in :func:`span_event_record_from_span`, and never re-reads the
    # retained span. Deep-copying the retained span's attributes here would clone the entire
    # request payload of an ``openai_completion`` span (an inlined image or long chat
    # history is regularly multi-MB) on every streaming delta, saturating the inline
    # dispatcher thread / event loop and starving live SSE ``collect``. Return an identity
    # snapshot instead. Start/End remain O(spans), so we keep their deep copies.
    if isinstance(event, SpanUpdateEvent):
        return Span(
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            name=span.name,
            start_time_unix_nano=span.start_time_unix_nano,
            end_time_unix_nano=span.end_time_unix_nano,
        )
    return Span(
        trace_id=span.trace_id,
        span_id=span.span_id,
        parent_span_id=span.parent_span_id,
        name=span.name,
        start_time_unix_nano=span.start_time_unix_nano,
        end_time_unix_nano=span.end_time_unix_nano,
        attributes=copy.deepcopy(span.attributes),
        events=copy.deepcopy(getattr(event, 'events', None) or []),
        status=span.status.model_copy(deep=True),
    )


async def _await_sink_lifecycle(awaitable: Awaitable[None], *, stage: str) -> None:
    """Separate caller cancellation from a sink that raises CancelledError itself."""
    task = asyncio.ensure_future(awaitable)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError as exc:
        if task.cancelled():
            raise RuntimeError(f'run trace sink {stage} raised CancelledError') from exc
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        else:
            with contextlib.suppress(BaseException):
                task.result()
        raise


def _validate_run_kind(run_kind: str) -> None:
    if run_kind not in RUN_KINDS:
        raise ValueError(f'unsupported run trace kind: {run_kind}')


_global_run_trace_processor: RunTraceProcessor | None = None
_global_run_trace_processor_lock = threading.Lock()


def ensure_run_trace_processor() -> RunTraceProcessor:
    """Return and register the process-wide run trace attribution hub."""
    global _global_run_trace_processor
    with _global_run_trace_processor_lock:
        if _global_run_trace_processor is None:
            _global_run_trace_processor = RunTraceProcessor()
        controller = SpanController.get_instance_sync()
        if _global_run_trace_processor not in controller.processors:
            controller.add_processor(_global_run_trace_processor)
        return _global_run_trace_processor
