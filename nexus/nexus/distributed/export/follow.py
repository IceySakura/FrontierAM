"""Bounded polling of persisted Cloud trace events."""

from __future__ import annotations

import asyncio
import math
import random
import sys
import time

from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, TypeVar

from nexus.distributed.export.types import ExportError
from nexus.distributed.export.writer import JsonlEventWriter
from nexus.distributed.trace.models import TraceStreamKey
from nexus.runner.trace.tdsql.schema import principal_trace_table_name


if TYPE_CHECKING:
    from nexus.runner.trace.tdsql.sink_source import RawTraceReadPlan


_T = TypeVar('_T')
_TRANSIENT_DB_CODES = {1040, 1042, 1047, 1158, 1159, 1160, 1161, 1205, 1213, 1317, 2002, 2003, 2006, 2013, 3024}


class AmbiguousTraceBackendError(ExportError):
    """One principal table is published by several backends, on different schemas.

    Distinct from a plain :class:`ExportError` so the results exporter can drop its
    partition-pruning optimisation instead of failing an export it would otherwise
    complete; the event exporter has no such choice, since the dialect decides whether
    its queries reference a column the target table may not have.
    """


_SUPPORTED_TYPES = {'nexus_batch', 'nexus_single', 'nexus_live'}
_TERMINAL = {'succeeded', 'failed', 'cancelled', 'lost', 'expired', 'killed'}


@dataclass
class _Stream:
    key: TraceStreamKey
    upper: int
    busy: bool = False
    done: bool = False
    next_due: float = 0


class _ReadBudget:
    """Reserve raw bytes through durable commit; oversized rows acquire exclusive access."""

    def __init__(self, capacity: int, batch_target: int) -> None:
        self.capacity = capacity
        self.batch_target = batch_target
        self.used = 0
        self.exclusive = False
        self.waiters: deque[object] = deque()
        self.condition = asyncio.Condition()

    async def acquire(self, size: int) -> None:
        ticket = object()
        amount = max(1, size)
        exclusive = size > self.batch_target
        async with self.condition:
            self.waiters.append(ticket)
            try:
                await self.condition.wait_for(
                    lambda: (
                        self.waiters[0] is ticket
                        and not self.exclusive
                        and (self.used == 0 if exclusive else self.used + amount <= self.capacity)
                    )
                )
                self.used += amount
                self.exclusive = exclusive
            finally:
                self.waiters.remove(ticket)
                self.condition.notify_all()

    async def release(self, size: int) -> None:
        async with self.condition:
            self.used -= max(1, size)
            self.exclusive = False
            self.condition.notify_all()


def _retryable(error: BaseException) -> bool:
    # The source wraps all OperationalError instances, including access denied.
    # Classify the original DB errno instead of retrying the wrapper blindly.
    # Inspect every link, not only the deepest cause: asyncio's TimeoutError is
    # itself raised ``from CancelledError``, so a deepest-only walk classified
    # the client's own statement timeout as non-retryable.
    seen: set[int] = set()
    cause: BaseException | None = error
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
            return True
        if cause.args and isinstance(cause.args[0], int):
            return cause.args[0] in _TRANSIENT_DB_CODES
        cause = cause.__cause__
    return False


async def _read_with_retry(operation: Callable[[], Awaitable[_T]], sleep: Callable[[float], Awaitable[None]]) -> _T:
    for attempt in range(3):
        try:
            return await operation()
        except Exception as exc:
            if not _retryable(exc) or attempt == 2:
                raise ExportError(f'Trace database read failed ({type(exc).__name__})') from exc
            print(f'Trace database read retry {attempt + 1}/2', file=sys.stderr)
            await sleep(0.25 * 2**attempt + random.uniform(0, 0.1))
    raise AssertionError('unreachable')


def _principal_table(metadata: Mapping[str, Any]) -> str | None:
    """The principal trace table this sink wrote, or ``None`` if it wrote none.

    The one place the derivation lives, because the two callers below disagree about how
    much else counts as evidence and that disagreement should be theirs alone, not a
    difference in how each spells the table name. A scope id the writer could not have
    produced yields ``None`` rather than a table name no backend holds.
    """
    principal = metadata.get('principal_scope_id')
    if principal is None:
        return None
    try:
        return principal_trace_table_name(principal)
    except ValueError:
        return None


def _validate_target(execution: Mapping[str, Any], execution_id: str, table: str) -> str:
    if execution.get('execution_id') != execution_id:
        raise ExportError('Execution identity mismatch')
    spec_type = execution.get('spec_type')
    if spec_type not in _SUPPORTED_TYPES:
        raise ExportError('Unsupported Execution spec_type')
    metadata_entries = [metadata for _, metadata in _sink_entries(execution, str(spec_type))]
    if metadata_entries:
        supported_scopes = {None, 'principal_v1', 'batch_principal_v1'}

        def referenced_tables(metadata: Mapping[str, Any]) -> list[Any]:
            references = [metadata[key] for key in ('table', 'trace_table') if metadata.get(key) is not None]
            principal_table = _principal_table(metadata)
            if principal_table is not None:
                references.append(principal_table)
            return references

        # Select the explicitly requested source before considering other sinks.
        # A valid principal sink can coexist with local/shared/other-table sinks.
        for metadata in metadata_entries:
            references = referenced_tables(metadata)
            if (
                metadata.get('storage_scope') in supported_scopes
                and references
                and all(reference == table for reference in references)
            ):
                return str(spec_type)

        for metadata in metadata_entries:
            if metadata.get('storage_scope') not in supported_scopes:
                raise ExportError('Known trace storage scope is unsupported')
            if any(reference != table for reference in referenced_tables(metadata)):
                raise ExportError('Known trace table does not match --trace-table')
        known_local = any(
            any(key in metadata for key in ('path', 'trace_events_path', 'manifest_path', 'artifact_uri'))
            for metadata in metadata_entries
        )
        if known_local:
            raise ExportError('Known trace backend is unsupported; principal TDSQL traces are required')
        # Missing, empty and opaque metadata are not evidence of any backend.
    return str(spec_type)


def _sink_entries(execution: Mapping[str, Any], spec_type: str) -> list[tuple[Any, Mapping[str, Any]]]:
    """The published sink entries, paired with the backend name that produced each one.

    The name matters as much as the metadata: the principal table name is derived from the
    owner alone, so two backends holding the same principal publish the same table under
    different schemas.
    """
    assignment = execution.get('current_assignment')
    runtime = assignment.get('runtime_info') if isinstance(assignment, Mapping) else execution.get('runtime_info')
    artifacts = runtime.get('_artifacts', {}) if isinstance(runtime, Mapping) else {}
    envelope = (
        artifacts.get('live' if spec_type == 'nexus_live' else 'single', {}) if isinstance(artifacts, Mapping) else {}
    )
    sinks = envelope.get('sinks') if isinstance(envelope, Mapping) else None
    if not isinstance(sinks, list):
        return []
    entries: list[tuple[Any, Mapping[str, Any]]] = []
    for sink in sinks:
        if not isinstance(sink, Mapping):
            continue
        metadata = sink.get('metadata')
        if isinstance(metadata, Mapping):
            entries.append((sink.get('name'), metadata))
    return entries


def resolve_partition_key(
    execution: Mapping[str, Any],
    spec_type: str,
    table: str,
    *,
    backend: str | None = None,
) -> str | None:
    """Read the Boundless partition key out of the envelope the writer left behind.

    Only Boundless tables carry ``parent_created_at``, and the writer records it precisely
    so readers can prune; see the sink's ``metadata()``. Deriving it here rather than
    asking for ``--parent-created-at`` removes a failure the caller could not detect: a
    forgotten flag used to select the classic dialect, which skips the partition filter
    and scans every day in the retention window while still returning correct rows.

    The table name cannot identify the schema on its own: it is derived from the owner
    principal, so a Run writing to both a classic and a Boundless backend publishes the
    same table twice. Taking whichever entry happens to carry a key would hand a Boundless
    partition filter to a classic connection, whose table has no such column. ``backend``
    names the sink to read -- the same name the caller used to obtain its database
    credentials -- and an ambiguous envelope is reported rather than guessed.

    Returns ``None`` for classic TDSQL, where the column does not exist.
    """
    from nexus.runner.trace.tdsql.sink_source import normalize_parent_created_at

    entries = _sink_entries(execution, spec_type)
    candidates: list[Any] = []
    for name, metadata in entries:
        # Only sinks that published this very table can speak for its schema. A sink
        # without a principal scope did not write it -- an artifact sink, or a trace sink on
        # the shared table -- and treating one as a candidate made its absent partition key
        # look like a second backend disagreeing about the dialect, which reported an
        # ambiguity for an Execution that writes the table exactly once.
        #
        # Deliberately stricter than the table references _validate_target accepts: only a
        # principal sink publishes parent_created_at at all, so a sink naming this table
        # any other way could only ever contribute a None and vote for classic.
        if _principal_table(metadata) != table:
            continue
        if backend is not None and name != backend:
            continue
        candidates.append(metadata.get('parent_created_at'))
    if backend is not None and entries and not candidates:
        # A name the envelope never published points at the wrong Execution or the wrong
        # backend; either way the dialect below would be a guess.
        raise ExportError('Execution has no trace sink named by --trace-backend')
    if len({value is None for value in candidates}) > 1:
        raise AmbiguousTraceBackendError('Execution writes this trace table on several backends; pass --trace-backend')
    for value in candidates:
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise ExportError('Execution reports an unusable trace partition key')
        try:
            return normalize_parent_created_at(value)
        except ValueError:
            raise ExportError('Execution reports an unusable trace partition key') from None
    return None


@dataclass(frozen=True, slots=True)
class TraceBackendChoice:
    """Which schema the export must speak, and the partition key that goes with it.

    The dialect and the partition key are one decision, not two: on Boundless every query
    carries ``parent_created_at``, and on classic TDSQL the column does not exist. Returning
    them together stops a caller from pairing a key resolved from one backend's envelope
    with a connection opened against another, whose table would reject the predicate.
    """

    dialect: str
    partition_key: str | None

    def source_class(self) -> type:
        """The registered source for this dialect.

        Selected by class rather than by a ``dialect=`` argument, matching how the sinks are
        registered: a build that predates Boundless cannot name the class at all, whereas a
        flag would let it construct a reader for a backend it knows nothing about.
        """
        from nexus.runner.trace.tdsql.sink_source import TDSQLBoundlessTraceSource, TDSQLTraceSource

        return TDSQLBoundlessTraceSource if self.partition_key is not None else TDSQLTraceSource

    def scope(self, *, table: str, execution_id: str, spec_type: str) -> dict[str, str]:
        """Read scope for ``table``, carrying the partition key only where it exists."""
        from nexus.runner.trace.tdsql.schema import principal_scope_id_from_table

        scope = {
            'storage_scope': 'principal_v1',
            'principal_scope_id': principal_scope_id_from_table(table),
            'parent_execution_id' if spec_type == 'nexus_batch' else 'run_id': execution_id,
        }
        if self.partition_key is not None:
            scope['parent_created_at'] = self.partition_key
        return scope


def resolve_trace_backend(
    execution: Mapping[str, Any],
    spec_type: str,
    table: str,
    *,
    backend: str | None = None,
    override_partition_key: str | None = None,
) -> TraceBackendChoice:
    """Decide the trace schema from the Execution rather than from the caller's flags.

    The dialect used to be inferred from whether a partition key had been found, which is
    not evidence of anything: the principal table name is derived from the owner alone, so
    a Run writing to both a classic and a Boundless backend publishes the same name twice,
    and taking whichever entry carried a key handed a Boundless predicate to a classic
    connection. ``backend`` names the sink to believe -- the same name the caller used to
    obtain its credentials -- and :func:`resolve_partition_key` reports an ambiguous
    envelope instead of guessing.

    ``override_partition_key`` serves ``--parent-created-at``, for the case where the
    envelope is unavailable; naming a key is itself a statement that the target is
    Boundless.
    """
    from nexus.runner.trace.tdsql.schema import BOUNDLESS_DIALECT, TDSQL_DIALECT

    partition_key = override_partition_key or resolve_partition_key(execution, spec_type, table, backend=backend)
    return TraceBackendChoice(
        dialect=BOUNDLESS_DIALECT if partition_key is not None else TDSQL_DIALECT,
        partition_key=partition_key,
    )


def _envelope_publishes_table(
    execution: Mapping[str, Any],
    spec_type: str,
    table: str,
    *,
    backend: str | None,
) -> bool:
    """Whether the envelope names the selected sink for this very table.

    ``resolve_trace_backend`` returns ``None`` both for a classic table and
    for an Execution whose envelope is unavailable. Only the first is
    authoritative; the second must be resolved by probing the schema.
    """
    try:
        entries = _sink_entries(execution, spec_type)
    except Exception:
        return False
    for name, metadata in entries:
        if backend is not None and name != backend:
            continue
        if _principal_table(metadata) == table:
            return True
    return False


def _envelope_has_entries(execution: Mapping[str, Any], spec_type: str) -> bool:
    try:
        return bool(_sink_entries(execution, spec_type))
    except Exception:
        return False


async def resolve_trace_read_scope(
    execution: Mapping[str, Any],
    spec_type: str,
    table: str,
    *,
    backend: str | None = None,
    override_partition_key: str | None = None,
    probe: Callable[[], Awaitable[bool | None]],
    fetch_execution: Callable[[str], Awaitable[Mapping[str, Any]]] | None = None,
    fail_closed_on_unknown: bool = False,
) -> TraceBackendChoice:
    """Resolve one Execution's trace dialect and partition key.

    Shared by the event (traces) and results exporters so the two cannot drift:
    the envelope wins when it names this table, otherwise the schema is probed
    once. ``override_partition_key`` is an explicit operator assertion and is
    validated rather than trusted blindly; a failed probe is fail-closed only
    for that assertion (or when the caller opts in), because a normal read can
    still be correct without pruning.
    """
    if override_partition_key is not None:
        partitioned = await probe()
        if partitioned is False:
            raise ExportError('Trace partition key was supplied for a non-partitioned table')
        if partitioned is None:
            raise ExportError('Cannot determine trace table partitioning')
        from nexus.runner.trace.tdsql.schema import BOUNDLESS_DIALECT

        return TraceBackendChoice(dialect=BOUNDLESS_DIALECT, partition_key=override_partition_key)

    if backend is not None and _envelope_has_entries(execution, spec_type):
        # An explicitly named backend must exist in the envelope even when the
        # probe could have guessed the dialect: a wrong name means the caller
        # is pointed at another Execution/backend.
        return resolve_trace_backend(execution, spec_type, table, backend=backend)

    if _envelope_publishes_table(execution, spec_type, table, backend=backend):
        return resolve_trace_backend(execution, spec_type, table, backend=backend)

    partitioned = await probe()
    from nexus.runner.trace.tdsql.schema import BOUNDLESS_DIALECT, TDSQL_DIALECT

    if partitioned is None:
        if fail_closed_on_unknown:
            raise ExportError('Cannot determine trace table partitioning')
        return TraceBackendChoice(dialect=TDSQL_DIALECT, partition_key=None)
    if not partitioned:
        return TraceBackendChoice(dialect=TDSQL_DIALECT, partition_key=None)

    source: Mapping[str, Any] = execution
    parent_id = execution.get('parent_execution_id')
    parent_lookup = False
    if spec_type == 'nexus_single' and isinstance(parent_id, str) and parent_id:
        parent_lookup = True
        if fetch_execution is None:
            raise ExportError('Partitioned trace table requires a parent execution lookup')
        source = await fetch_execution(parent_id)
    created_at = source.get('created_at') if isinstance(source, Mapping) else None
    if not isinstance(created_at, str) or not created_at:
        if parent_lookup:
            raise ExportError('Partitioned trace table requires created_at for the parent execution')
        raise ExportError('Partitioned trace table requires created_at for the execution')
    from nexus.runner.trace.tdsql.sink_source import normalize_parent_created_at

    return TraceBackendChoice(
        dialect=BOUNDLESS_DIALECT,
        partition_key=normalize_parent_created_at(created_at),
    )


async def export_trace_events(
    *,
    api: Any,
    source: Any,
    writer: JsonlEventWriter,
    execution_id: str,
    scope_metadata: Mapping[str, Any],
    follow: bool,
    poll_interval: float = 1.0,
    discovery_interval: float = 5.0,
    page_size: int = 500,
    concurrency: int = 4,
    queue_size: int = 4,
    batch_target_bytes: int = 8 * 1024 * 1024,
    max_buffered_bytes: int = 32 * 1024 * 1024,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    prepare_scope: Callable[[], Awaitable[Mapping[str, Any]]] | None = None,
) -> None:
    """Own writer lifetime; cancellation preserves committed progress and is re-raised.

    The discovery task and fixed read workers share a fair semaphore. Every stream
    stays busy until its page has been durably committed by the single consumer.
    Complete bodies are read only after a metadata plan reserves their raw bytes.
    The reservation lasts through the writer ACK; an oversized event drains and
    excludes all other payload reads until its complete record is committed.
    """
    for value in (poll_interval, discovery_interval):
        if not math.isfinite(value) or value <= 0:
            raise ExportError('Polling intervals must be positive and finite')
    for value in (page_size, concurrency, queue_size, batch_target_bytes, max_buffered_bytes):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ExportError('Page, concurrency, and queue limits must be positive integers')
    if page_size > 500 or batch_target_bytes > max_buffered_bytes:
        raise ExportError('Event page or byte budget is invalid')
    if not callable(getattr(source, 'plan_raw_event_read', None)):
        raise ExportError('Trace source does not support bounded raw read planning')
    budget = _ReadBudget(max_buffered_bytes, batch_target_bytes)
    scope = dict(scope_metadata)
    if scope.get('storage_scope') == 'batch_principal_v1':
        scope['storage_scope'] = 'principal_v1'
    principal_scope_id = scope.get('principal_scope_id', '')
    try:
        table = principal_trace_table_name(principal_scope_id)
    except ValueError:
        raise ExportError(f'invalid Trace Database table: {principal_scope_id!r}') from None
    if scope.get('storage_scope') != 'principal_v1':
        raise ExportError('Only principal trace storage is supported')
    execution = await api.get_execution(execution_id)
    spec_type = _validate_target(execution, execution_id, table)
    scope_key = 'parent_execution_id' if spec_type == 'nexus_batch' else 'run_id'
    if scope.get(scope_key) != execution_id or (spec_type != 'nexus_batch' and scope.get('parent_execution_id')):
        raise ExportError('Trace scope does not match target Execution')

    streams: dict[TraceStreamKey, _Stream] = {}
    ready: deque[_Stream] = deque()
    jobs: asyncio.Queue[_Stream] = asyncio.Queue(maxsize=concurrency)
    events: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=queue_size)
    semaphore = asyncio.Semaphore(concurrency)
    discovery_done = False
    discovery_round = 0
    busy = 0
    last_status = execution.get('status')
    last_commit_time = 'unknown'
    last_read_seconds = 0.0
    last_discovery_seconds = 0.0
    tasks: list[asyncio.Task[Any]] = []

    async def query(operation: Callable[[], Awaitable[_T]]) -> _T:
        async def once() -> _T:
            async with semaphore:
                return await operation()

        return await _read_with_retry(once, sleep)

    async def discover() -> None:
        while True:
            after = None
            while True:
                started = clock()
                rows = await query(lambda after=after: source.discover_raw_streams(scope, after=after, limit=page_size))
                ack = asyncio.get_running_loop().create_future()
                await events.put(('discovery', (rows, ack, clock() - started)))
                await ack
                if len(rows) < page_size:
                    break
                next_after = (rows[-1]['run_id'], rows[-1]['run_span_id'])
                if next_after == after:
                    raise ExportError('Trace discovery cursor did not advance')
                after = next_after
            await events.put(('sweep', None))
            if not follow:
                return
            await sleep(discovery_interval)

    async def worker() -> None:
        while True:
            stream = await jobs.get()
            cursor = writer.cursor_for(stream.key)
            metadata = {**scope, 'run_id': stream.key.run_id, 'run_span_id': stream.key.run_span_id}
            started = clock()
            plan: RawTraceReadPlan = await query(
                lambda metadata=metadata, cursor=cursor, stream=stream: source.plan_raw_event_read(
                    metadata,
                    cursor,
                    limit=page_size,
                    target_bytes=batch_target_bytes,
                    through_record_id=None if follow else stream.upper,
                )
            )
            await budget.acquire(plan.size_bytes)
            rows: list[dict[str, Any]] = []
            try:
                if plan.record_sizes:
                    rows = await query(
                        lambda metadata=metadata, cursor=cursor, plan=plan: source.read_raw_events_after(
                            metadata,
                            cursor,
                            limit=len(plan.record_sizes),
                            through_record_id=plan.through_record_id,
                        )
                    )
                    actual_sizes = tuple((row['record_id'], len(row['event_data'].encode('utf-8'))) for row in rows)
                    if actual_sizes != plan.record_sizes:
                        raise ExportError('Trace records changed after their read plan')
                ack = asyncio.get_running_loop().create_future()
                await events.put(('page', (stream, cursor, rows, clock() - started, plan.has_more, ack)))
                await ack
            finally:
                rows.clear()
                await budget.release(plan.size_bytes)

    async def observe() -> None:
        while True:
            await sleep(10)
            target = await api.get_execution(execution_id)  # CloudApi owns its bounded retries.
            if _validate_target(target, execution_id, table) != spec_type:
                raise ExportError('Execution spec_type changed during export')
            await events.put(('status', target.get('status')))

    async def guarded(operation: Awaitable[None]) -> None:
        try:
            await operation
        except Exception as exc:
            await events.put(('error', exc))

    async def writer_operation(operation: Awaitable[None]) -> None:
        io = asyncio.ensure_future(operation)
        try:
            await asyncio.shield(io)
        except asyncio.CancelledError:
            # Stop producers immediately, before waiting for durable writer I/O.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await io
            raise

    async def heartbeat() -> None:
        while True:
            await sleep(10)
            await events.put(('heartbeat', None))

    async def observation(**extra: Any) -> None:
        await writer_operation(
            writer.update_observation({
                'discovered_streams': len(streams),
                'discovery_round': discovery_round,
                'execution_status': last_status,
                'read_seconds': last_read_seconds,
                'discovery_seconds': last_discovery_seconds,
                **extra,
            })
        )

    async def next_event() -> tuple[str, Any]:
        # A fixed pair of waiters permits an injected clock/sleep without creating
        # a task per stream. Both waiters are always drained on cancellation.
        get = asyncio.create_task(events.get())
        tick = asyncio.ensure_future(sleep(min(poll_interval, 0.1)))
        try:
            await asyncio.wait((get, tick), return_when=asyncio.FIRST_COMPLETED)
            return get.result() if get.done() else ('tick', None)
        finally:
            for task in (get, tick):
                if not task.done():
                    task.cancel()
            await asyncio.gather(get, tick, return_exceptions=True)

    reason = 'error'
    primary_error: BaseException | None = None
    try:
        await writer_operation(writer.open())
        if prepare_scope is not None:
            # Validate checkpoint identity before opening any database connection.
            scope.update(await prepare_scope())
        await observation()
        tasks = [
            asyncio.create_task(guarded(discover())),
            asyncio.create_task(guarded(observe())),
            asyncio.create_task(guarded(heartbeat())),
        ]
        tasks.extend(asyncio.create_task(guarded(worker())) for _ in range(concurrency))
        while True:
            for _ in range(len(ready)):
                stream = ready.popleft()
                if stream.done:
                    continue
                ready.append(stream)
                if busy >= concurrency + queue_size or jobs.full() or events.full():
                    break
                if not stream.busy and stream.next_due <= clock():
                    stream.busy = True
                    busy += 1
                    jobs.put_nowait(stream)
            if not follow and discovery_done and all(stream.done for stream in streams.values()):
                reason = 'snapshot'
                break
            kind, payload = await next_event()
            if kind == 'error':
                raise payload
            if kind == 'discovery':
                rows, ack, elapsed = payload
                last_discovery_seconds = elapsed
                previous_count = len(streams)
                for row in rows:
                    key = TraceStreamKey('principal_v1', table, row['run_id'], row['run_span_id'])
                    upper = row['max_record_id']
                    if isinstance(upper, bool) or not isinstance(upper, int) or upper < 1:
                        raise ExportError('Invalid discovered stream upper bound')
                    if key not in streams:
                        stream = _Stream(key, upper)
                        streams[key] = stream
                        ready.append(stream)
                await observation()
                if len(streams) != previous_count:
                    print(f'Trace export: discovered streams={len(streams)}', file=sys.stderr)
                ack.set_result(None)
            elif kind == 'sweep':
                discovery_done = True
                discovery_round += 1
                await observation(last_complete_discovery_at=time.time())
            elif kind == 'page':
                stream, cursor, rows, elapsed, has_more, ack = payload
                last_read_seconds = elapsed
                if not follow and any(row['record_id'] > stream.upper for row in rows):
                    raise ExportError('Snapshot read exceeded captured upper bound')
                if rows:
                    await writer_operation(writer.commit_page(stream.key, expected_cursor=cursor, rows=rows))
                    last_commit_time = datetime.now(timezone.utc).isoformat()
                if not follow:
                    stream.done = writer.cursor_for(stream.key) >= stream.upper
                    if not stream.done and not has_more:
                        raise ExportError('Snapshot read stopped before captured upper bound')
                stream.next_due = clock() + (0 if has_more else poll_interval)
                stream.busy = False
                busy -= 1
                rows.clear()
                ack.set_result(None)
            elif kind == 'status':
                last_status = payload
                await observation()
            elif kind == 'heartbeat':
                await observation()
                message = (
                    f'Trace export: target={execution_id} streams={len(streams)} records={writer.committed_records} '
                    f'bytes={writer.committed_bytes} read_seconds={last_read_seconds:.3f} '
                    f'discovery_seconds={last_discovery_seconds:.3f} last_commit={last_commit_time}'
                )
                if last_status in _TERMINAL and writer.committed_records == 0:
                    message += '; target terminal, no visible trace; backend metadata may be unknown'
                print(message, file=sys.stderr)
    except BaseException as exc:
        primary_error = exc
        if isinstance(exc, asyncio.CancelledError):
            reason = 'interrupted'
    finally:

        async def teardown() -> None:
            try:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                await writer.close(reason=reason)

        # Shield the complete teardown, not just writer I/O. The first signal
        # can arrive during producer cleanup after normal snapshot processing.
        cleanup = asyncio.create_task(teardown())
        cancelled = isinstance(primary_error, asyncio.CancelledError)
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
                if primary_error is None or isinstance(primary_error, asyncio.CancelledError):
                    reason = 'interrupted'
            except BaseException:
                break  # Retrieve the cleanup failure below without masking the primary error.
        cleanup_error: BaseException | None = None
        try:
            cleanup.result()
        except BaseException as exc:
            cleanup_error = exc
        if cancelled:
            cause = (
                primary_error
                if primary_error is not None and not isinstance(primary_error, asyncio.CancelledError)
                else cleanup_error
            )
            raise asyncio.CancelledError from cause
        if primary_error is not None:
            if cleanup_error is not None:
                primary_error.__context__ = cleanup_error
            raise primary_error
        if cleanup_error is not None:
            raise cleanup_error
        print(f'Trace export stopped: {reason}; committed records={writer.committed_records}', file=sys.stderr)
