from __future__ import annotations

import asyncio
import json

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from nexus.distributed.export_output import JsonlEventWriter
from nexus.distributed.export_types import ExportError
from nexus.runner.trace.tdsql.sink_source import RawTraceReadPlan
from tests.unit.distributed._trace_export_fixtures import (
    OTHER_PRINCIPAL_SCOPE_ID,
    PARTITION_KEY_ISO,
    PARTITION_KEY_SQL,
    PRINCIPAL_SCOPE_ID,
    envelope_execution,
    metadata_execution,
    principal_scope,
    sink,
)
from tests.unit.distributed.test_cloud_event_export_output import TABLE, _event, _source


SCOPE = principal_scope(TABLE, parent_execution_id='batch-1')


class Api:
    async def get_execution(self, execution_id):
        return {'execution_id': execution_id, 'spec_type': 'nexus_batch', 'status': 'succeeded'}


class Source:
    def __init__(self):
        self.rows: list[dict[str, Any]] = [_event(1), _event(3)]
        self.discoveries = 0
        self.cursors = []
        self.failures = 0

    async def discover_raw_streams(self, metadata, *, after, limit):
        if after is not None:
            return []
        self.discoveries += 1
        keys = list(dict.fromkeys((r['run_id'], r['run_span_id']) for r in self.rows))
        return [
            {
                'run_id': run,
                'run_span_id': span,
                'max_record_id': max(
                    r['record_id'] for r in self.rows if (r['run_id'], r['run_span_id']) == (run, span)
                ),
            }
            for run, span in keys
        ][:limit]

    async def read_raw_events_after(self, metadata, cursor, *, limit, through_record_id=None):
        self.cursors.append(cursor)
        if self.failures:
            self.failures -= 1
            raise ConnectionResetError('secret-password')
        return [
            r
            for r in self.rows
            if r['run_id'] == metadata['run_id']
            and r['run_span_id'] == metadata['run_span_id']
            and r['record_id'] > cursor
            and (through_record_id is None or r['record_id'] <= through_record_id)
        ][:limit]

    async def plan_raw_event_read(self, metadata, cursor, *, limit, target_bytes, through_record_id=None):
        rows = [
            r
            for r in self.rows
            if r['run_id'] == metadata['run_id']
            and r['run_span_id'] == metadata['run_span_id']
            and r['record_id'] > cursor
            and (through_record_id is None or r['record_id'] <= through_record_id)
        ]
        selected = []
        total = 0
        for row in rows:
            size = len(row['event_data'].encode())
            if len(selected) == limit or (selected and total + size > target_bytes):
                return RawTraceReadPlan(tuple(selected), True)
            selected.append((row['record_id'], size))
            total += size
        return RawTraceReadPlan(tuple(selected), False)


def _exception_chain(error: BaseException) -> list[BaseException]:
    pending = [error]
    chain: list[BaseException] = []
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        chain.append(current)
        pending.extend(link for link in (current.__cause__, current.__context__) if link is not None)
    return chain


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), 3)


async def run(writer, source, **kwargs):
    from nexus.distributed import export_events

    return await export_events.export_trace_events(
        api=Api(),
        source=source,
        writer=writer,
        execution_id='batch-1',
        scope_metadata=SCOPE,
        poll_interval=0.01,
        discovery_interval=0.02,
        **kwargs,
    )


async def test_follow_terminal_discovers_late_smaller_child_and_same_execution_span(tmp_path):
    writer = JsonlEventWriter(tmp_path / 'events', source=_source())
    source = Source()
    task = asyncio.create_task(run(writer, source, follow=True))
    try:
        await until(lambda: writer.committed_records == 2)
        source.rows.extend([_event(1, run_id='a-child', run_span_id='a-span'), _event(1, run_span_id='other-attempt')])
        await until(lambda: writer.committed_records == 4)
        assert not task.done()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert json.loads(writer.manifest.read_text())['status'] == 'interrupted'
    assert len(writer.output.read_text().splitlines()) == 4


async def test_transient_read_retries_same_cursor_without_skipping(tmp_path):
    writer = JsonlEventWriter(tmp_path / 'events', source=_source(), follow=False)
    source = Source()
    source.failures = 1
    await run(writer, source, follow=False, page_size=1)
    assert source.cursors[:2] == [0, 0]
    assert [json.loads(line)['record_id'] for line in writer.output.read_text().splitlines()] == [1, 3]


async def test_snapshot_fixed_upper_bound_excludes_new_rows(tmp_path):
    class Growing(Source):
        async def read_raw_events_after(self, *args, **kwargs):
            self.rows.append(_event(100 + len(self.rows)))
            return await super().read_raw_events_after(*args, **kwargs)

    writer = JsonlEventWriter(tmp_path / 'events', source=_source(), follow=False)
    await run(writer, Growing(), follow=False, page_size=1)
    assert len(writer.output.read_text().splitlines()) == 2
    assert json.loads(writer.manifest.read_text())['status'] == 'snapshot'


async def test_snapshot_empty_before_captured_highwater_is_error(tmp_path):
    class Missing(Source):
        async def read_raw_events_after(self, *args, **kwargs):
            return []

    writer = JsonlEventWriter(tmp_path / 'events', source=_source(), follow=False)
    with pytest.raises(ExportError, match='changed after their read plan'):
        await run(writer, Missing(), follow=False)
    assert not writer.output.exists()


async def test_byte_budget_pages_and_oversized_row_preserve_snapshot_tail(tmp_path):
    source = Source()
    source.rows = [_event(i) for i in range(1, 7)]
    source.rows[2]['event_data'] = json.dumps({'large': '中' * 5000})
    calls = []
    original = source.read_raw_events_after

    async def read(*args, **kwargs):
        rows = await original(*args, **kwargs)
        calls.append([r['record_id'] for r in rows])
        return rows

    source.read_raw_events_after = read
    writer = JsonlEventWriter(tmp_path / 'events', source=_source(), follow=False)
    await run(writer, source, follow=False, batch_target_bytes=256, max_buffered_bytes=1024)
    assert [3] in calls
    assert all(len(c) == 1 for c in calls if 3 in c)
    actual = [json.loads(line) for line in writer.output.read_text().splitlines()]
    assert actual == source.rows
    assert json.loads(writer.manifest.read_text())['status'] == 'snapshot'


async def test_read_budget_oversized_waiter_drains_and_blocks_later_readers():
    from nexus.distributed.export.follow import _ReadBudget

    budget = _ReadBudget(100, 50)
    await budget.acquire(40)
    large = asyncio.create_task(budget.acquire(200))
    await until(lambda: len(budget.waiters) == 1)
    small = asyncio.create_task(budget.acquire(10))
    await until(lambda: len(budget.waiters) == 2)
    assert not large.done()
    assert not small.done()
    await budget.release(40)
    await large
    assert budget.used == 200
    assert budget.exclusive
    assert not small.done()
    await budget.release(200)
    await small
    assert budget.used == 10
    assert not budget.exclusive
    await budget.release(10)
    assert budget.used == 0


async def test_cancelled_budget_waiter_does_not_block_other_streams():
    from nexus.distributed.export.follow import _ReadBudget

    budget = _ReadBudget(100, 50)
    await budget.acquire(40)
    cancelled = asyncio.create_task(budget.acquire(200))
    await until(lambda: len(budget.waiters) == 1)
    remaining = asyncio.create_task(budget.acquire(20))
    await until(lambda: len(budget.waiters) == 2)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    await remaining
    assert budget.used == 60
    await budget.release(40)
    await budget.release(20)
    assert budget.used == 0
    assert not budget.waiters


async def test_control_only_api_supported_and_result_requires_application():
    from aiohttp import ClientSession

    from nexus.distributed.export import CloudApi

    async with ClientSession() as session:
        api = CloudApi(session, control_url='https://control.invalid', token='token')
        with pytest.raises(ExportError, match='Application'):
            await api.get_result('exe')


async def test_api_429_respects_retry_after():
    from aiohttp import ClientSession, web
    from aiohttp.test_utils import TestServer

    from nexus.distributed.export import CloudApi

    attempts = 0
    delays = []

    async def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return web.json_response({}, status=429, headers={'Retry-After': '12'})
        return web.json_response({'execution_id': 'exe'})

    async def sleep(delay):
        delays.append(delay)

    app = web.Application()
    app.router.add_get('/v1/executions/exe', handler)
    async with TestServer(app) as server, ClientSession() as session:
        api = CloudApi(session, control_url=str(server.make_url('/')), token='secret', sleep=sleep)
        assert await api.get_execution('exe') == {'execution_id': 'exe'}
    assert len(delays) == 1
    assert delays[0] >= 12


@pytest.mark.parametrize(
    ('metadata', 'message'),
    [
        ({'storage_scope': 'shared', 'trajectory_group_id': 'g'}, 'scope'),
        ({'storage_scope': 'principal_v1', 'principal_scope_id': 'f' * 32}, 'table'),
        ({'path': '/local/trace.jsonl'}, 'backend'),
    ],
)
def test_known_runtime_backend_metadata_rejects_incompatible_scope(metadata, message):
    from nexus.distributed.export.follow import _validate_target

    execution = {
        'execution_id': 'exe',
        'spec_type': 'nexus_single',
        'runtime_info': {'_artifacts': {'single': {'sinks': [{'name': 'custom', 'metadata': metadata}]}}},
    }
    with pytest.raises(ExportError, match=message):
        _validate_target(execution, 'exe', TABLE)


def test_known_principal_runtime_metadata_allows_configured_name():
    from nexus.distributed.export.follow import _validate_target

    execution = {
        'execution_id': 'exe',
        'spec_type': 'nexus_single',
        'runtime_info': {'_artifacts': {'single': {'sinks': [{'name': 'my-db', 'metadata': dict(SCOPE)}]}}},
    }
    assert _validate_target(execution, 'exe', TABLE) == 'nexus_single'


async def test_slow_writer_bounds_queries_and_hot_stream_does_not_starve_discovery(tmp_path):
    class Measured(Source):
        def __init__(self):
            super().__init__()
            self.active = 0
            self.peak = 0
            self.pending = set()
            self.peak_pending = 0
            self.rows = [_event(i) for i in range(1, 100)]
            self.rows.append(_event(1, run_id='b-child', run_span_id='b-span'))

        async def discover_raw_streams(self, metadata, *, after, limit):
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.001)
            self.active -= 1
            if after is not None:
                keys = list(dict.fromkeys((r['run_id'], r['run_span_id']) for r in self.rows))
                keys = keys[keys.index(after) + 1 :][:limit]
                return [{'run_id': run, 'run_span_id': span, 'max_record_id': 1} for run, span in keys]
            return await super().discover_raw_streams(metadata, after=after, limit=limit)

        async def read_raw_events_after(self, metadata, *args, **kwargs):
            key = (metadata['run_id'], metadata['run_span_id'])
            assert key not in self.pending, 'a stream was read before its previous page ACK'
            self.pending.add(key)
            self.peak_pending = max(self.peak_pending, len(self.pending))
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.001)
            self.active -= 1
            rows = await super().read_raw_events_after(metadata, *args, **kwargs)
            if not rows:
                self.pending.remove(key)
            return rows

    source = Measured()

    class SlowWriter(JsonlEventWriter):
        async def commit_page(self, stream, **kwargs):
            await asyncio.sleep(0.005)
            await super().commit_page(stream, **kwargs)
            source.pending.remove((stream.run_id, stream.run_span_id))

    writer = SlowWriter(tmp_path / 'events', source=_source())
    task = asyncio.create_task(run(writer, source, follow=True, page_size=2, concurrency=2, queue_size=1))
    try:
        await until(lambda: writer.committed_records >= 5)
        source.rows.append(_event(1, run_id='a-child', run_span_id='late'))
        await until(
            lambda: any(json.loads(line)['run_id'] == 'a-child' for line in writer.output.read_text().splitlines())
        )
        assert writer.committed_records < 100
        assert source.peak <= 2
        assert source.peak_pending <= 3
        assert source.discoveries > 1
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.parametrize(('errno', 'attempts'), [(1045, 1), (1142, 1), (1146, 1), (2006, 3)])
async def test_database_original_errno_controls_finite_retry(tmp_path, errno, attempts, capsys):
    class SqlError(Exception):
        pass

    class Failed(Source):
        async def read_raw_events_after(self, metadata, cursor, **kwargs):
            self.cursors.append(cursor)
            try:
                raise SqlError(errno, 'secret-password')
            except SqlError as exc:
                raise RuntimeError('wrapped secret-password') from exc

    async def immediate(delay):
        await asyncio.sleep(0 if 0.25 <= delay < 1 else delay)

    source = Failed()
    writer = JsonlEventWriter(tmp_path / 'events', source=_source())
    with pytest.raises(ExportError, match='database read failed') as raised:
        await run(writer, source, follow=True, sleep=immediate)
    assert 'secret-password' not in str(raised.value)
    assert source.cursors == [0] * attempts
    assert writer.committed_records == 0
    assert json.loads(writer.manifest.read_text())['status'] == 'error'
    assert 'secret-password' not in capsys.readouterr().err


async def test_cancellation_during_real_writer_io_waits_and_commits(tmp_path, monkeypatch):
    import threading

    import nexus.distributed.export.writer as output_module

    entered = threading.Event()
    release = threading.Event()
    original = output_module._write_all

    def blocked(stream, data):
        entered.set()
        assert release.wait(3)
        original(stream, data)

    monkeypatch.setattr(output_module, '_write_all', blocked)
    writer = JsonlEventWriter(tmp_path / 'events', source=_source())
    task = asyncio.create_task(run(writer, Source(), follow=True))
    await until(entered.is_set)
    task.cancel()
    await asyncio.sleep(0.02)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert writer.committed_records == 2
    assert json.loads(writer.manifest.read_text())['status'] == 'interrupted'


async def test_discovery_uses_last_database_key_without_python_sorting(tmp_path):
    class Collated(Source):
        def __init__(self):
            super().__init__()
            self.rows = [_event(1, run_span_id=span) for span in ('a', 'Z', 'z1')]
            self.afters = []

        async def discover_raw_streams(self, metadata, *, after, limit):
            self.afters.append(after)
            if after is None:
                return [{'run_id': 'child-1', 'run_span_id': span, 'max_record_id': 1} for span in ('a', 'Z')]
            assert after == ('child-1', 'Z')
            return [{'run_id': 'child-1', 'run_span_id': 'z1', 'max_record_id': 1}]

    source = Collated()
    writer = JsonlEventWriter(tmp_path / 'events', source=_source(), follow=False)
    await run(writer, source, follow=False, page_size=2)
    assert writer.committed_records == 3
    assert source.afters == [None, ('child-1', 'Z')]


@pytest.mark.parametrize('status', ['succeeded', 'killed'])
async def test_injected_clock_empty_terminal_heartbeats_without_stopping(tmp_path, capsys, monkeypatch, status):
    async def get_execution(self, execution_id):
        return {'execution_id': execution_id, 'spec_type': 'nexus_batch', 'status': status}

    monkeypatch.setattr(Api, 'get_execution', get_execution)

    class Clock:
        now = 0.0

        def __init__(self):
            self.waiters = []

        def __call__(self):
            return self.now

        async def sleep(self, delay):
            future = asyncio.get_running_loop().create_future()
            self.waiters.append((self.now + delay, future))
            await future

        def advance(self, delay):
            self.now += delay
            for deadline, future in self.waiters:
                if deadline <= self.now and not future.done():
                    future.set_result(None)
            self.waiters = [(deadline, future) for deadline, future in self.waiters if not future.done()]

    clock = Clock()
    source = Source()
    source.rows.clear()
    writer = JsonlEventWriter(tmp_path / 'events', source=_source())
    task = asyncio.create_task(run(writer, source, follow=True, clock=clock, sleep=clock.sleep))
    try:
        await until(lambda: source.discoveries == 1)
        await until(lambda: any(deadline == 0.02 and not future.done() for deadline, future in clock.waiters))
        clock.advance(10)
        await until(lambda: source.discoveries >= 2)
        await until(lambda: json.loads(writer.manifest.read_text())['observed']['discovery_round'] >= 2)
        assert not task.done()
        assert writer.committed_records == 0
        assert 'target terminal, no visible trace' in capsys.readouterr().err
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_network_oserror_errno_is_retryable_without_confusing_database_access_denied():
    from nexus.distributed.export.follow import _retryable

    assert _retryable(ConnectionResetError(104, 'reset'))
    assert not _retryable(RuntimeError(1045, 'access denied'))


async def test_stale_queued_page_is_rejected_using_captured_read_cursor(tmp_path):
    from nexus.distributed.export_types import TraceStreamKey

    writer = JsonlEventWriter(tmp_path / 'events', source=_source())

    class Stale(Source):
        async def read_raw_events_after(self, metadata, cursor, **kwargs):
            rows = await super().read_raw_events_after(metadata, cursor, **kwargs)
            key = TraceStreamKey('principal_v1', TABLE, metadata['run_id'], metadata['run_span_id'])
            await writer.commit_page(key, expected_cursor=cursor, rows=rows)
            return rows

    with pytest.raises(ExportError, match='cursor'):
        await run(writer, Stale(), follow=True)
    assert writer.committed_records == 2
    assert json.loads(writer.manifest.read_text())['status'] == 'error'


@pytest.mark.parametrize('metadata', [None, {}, {'opaque': 'custom extension metadata'}])
def test_missing_or_unrecognized_sink_metadata_remains_unknown(metadata):
    from nexus.distributed.export.follow import _validate_target

    sink = {'name': 'custom'}
    if metadata is not None:
        sink['metadata'] = metadata
    execution = {
        'execution_id': 'exe',
        'spec_type': 'nexus_single',
        'runtime_info': {'_artifacts': {'single': {'sinks': [sink]}}},
    }
    assert _validate_target(execution, 'exe', TABLE) == 'nexus_single'


@pytest.mark.parametrize('fail_read', [False, True])
async def test_first_cancellation_during_producer_teardown_always_closes_writer(tmp_path, fail_read):
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    class DelayedCleanupApi(Api):
        calls = 0

        async def get_execution(self, execution_id):
            self.calls += 1
            if self.calls > 1:
                try:
                    await asyncio.Event().wait()
                finally:
                    cleanup_started.set()
                    await release_cleanup.wait()
            return await super().get_execution(execution_id)

    api = DelayedCleanupApi()

    observer_started = False

    async def sleep(delay):
        nonlocal observer_started
        if delay == 10 and not observer_started:
            observer_started = True
            await asyncio.sleep(0)
        else:
            await asyncio.sleep(delay)

    class WaitForObserver(Source):
        async def read_raw_events_after(self, *args, **kwargs):
            await until(lambda: api.calls > 1)
            if fail_read:
                raise ValueError('primary read failure')
            return await super().read_raw_events_after(*args, **kwargs)

    from nexus.distributed.export_events import export_trace_events

    writer = JsonlEventWriter(tmp_path / 'events', source=_source(), follow=False)
    task = asyncio.create_task(
        export_trace_events(
            api=api,
            source=WaitForObserver(),
            writer=writer,
            execution_id='batch-1',
            scope_metadata=SCOPE,
            follow=False,
            sleep=sleep,
        )
    )
    try:
        await asyncio.wait_for(cleanup_started.wait(), 3)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done(), 'teardown cancellation must wait for producer cleanup and writer closure'
    finally:
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        if fail_read:
            # asyncio.Task can insert another CancelledError wrapper on Python 3.10.
            chain = _exception_chain(raised.value)
            assert any(isinstance(error, ExportError) and 'database read failed' in str(error) for error in chain)
            assert any(isinstance(error, ValueError) and str(error) == 'primary read failure' for error in chain)

    # Use the real OS lock and same output name to prove teardown released ownership.
    import fcntl

    with Path(f'{writer.output}.lock').open('rb') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    assert not writer.output.exists(), 'cancelled snapshot must not publish successful output'


def test_current_assignment_trace_metadata_is_authoritative_over_flattened_compatibility():
    from nexus.distributed.export.follow import _validate_target

    wrong = {'storage_scope': 'principal_v1', 'principal_scope_id': 'f' * 32}
    execution = {
        'execution_id': 'exe',
        'spec_type': 'nexus_single',
        'runtime_info': {'_artifacts': {'single': {'sinks': [{'metadata': dict(SCOPE)}]}}},
        'current_assignment': {'runtime_info': {'_artifacts': {'single': {'sinks': [{'metadata': wrong}]}}}},
    }
    with pytest.raises(ExportError, match='table'):
        _validate_target(execution, 'exe', TABLE)


@pytest.mark.parametrize(
    'other_metadata',
    [
        {'storage_scope': 'shared', 'trajectory_group_id': 'g'},
        {'storage_scope': 'principal_v1', 'principal_scope_id': 'f' * 32},
        {'trace_events_path': '/local/trace.jsonl'},
    ],
)
def test_selected_principal_table_can_coexist_with_other_trace_sinks(other_metadata):
    from nexus.distributed.export.follow import _validate_target

    execution = {
        'execution_id': 'exe',
        'spec_type': 'nexus_single',
        'runtime_info': {
            '_artifacts': {
                'single': {
                    'sinks': [
                        {'name': 'other', 'metadata': other_metadata},
                        {'name': 'selected', 'metadata': dict(SCOPE)},
                    ]
                }
            }
        },
    }
    assert _validate_target(execution, 'exe', TABLE) == 'nexus_single'


def test_explicit_table_mismatch_is_rejected_even_without_storage_scope():
    from nexus.distributed.export.follow import _validate_target

    execution = {
        'execution_id': 'exe',
        'spec_type': 'nexus_single',
        'runtime_info': {
            '_artifacts': {'single': {'sinks': [{'metadata': {'table': 'trajectory_span_events_pr_' + 'f' * 32}}]}}
        },
    }
    with pytest.raises(ExportError, match='table'):
        _validate_target(execution, 'exe', TABLE)


async def test_primary_read_error_survives_writer_close_failure(tmp_path):
    class FailingClose(JsonlEventWriter):
        async def close(self, *, reason):
            await super().close(reason=reason)
            raise ExportError('secondary close failure')

    class FailingRead(Source):
        async def read_raw_events_after(self, *args, **kwargs):
            raise ValueError('primary read failure')

    writer = FailingClose(tmp_path / 'events', source=_source())
    with pytest.raises(ExportError, match='database read failed') as raised:
        await run(writer, FailingRead(), follow=True)
    assert isinstance(raised.value.__cause__, ValueError)
    assert isinstance(raised.value.__context__, ExportError)
    assert 'secondary close failure' in str(raised.value.__context__)


# --- partition key resolution --------------------------------------------------------
# The writer records parent_created_at in its envelope so readers can prune. Taking it
# from there rather than from a flag removes a failure the caller cannot observe: a
# forgotten flag used to select the classic dialect, which returns the same rows while
# reading every day partition in the retention window.


def _envelope_execution(metadata: dict[str, Any], *, spec_type: str = 'nexus_single') -> dict[str, Any]:
    return metadata_execution(metadata, spec_type=spec_type)


def test_partition_key_comes_from_the_envelope() -> None:
    from nexus.distributed.export.follow import resolve_partition_key

    execution = _envelope_execution({**SCOPE, 'parent_created_at': '2026-09-21T10:30:00'})

    # Normalised on the way out: MySQL DATETIME rejects the ISO 'T' separator.
    assert resolve_partition_key(execution, 'nexus_single', TABLE) == '2026-09-21 10:30:00.000000'


def test_classic_tables_resolve_to_no_partition_key() -> None:
    """Absence is the classic schema, not an error -- those tables have no such column."""
    from nexus.distributed.export.follow import resolve_partition_key

    execution = _envelope_execution(dict(SCOPE))

    assert resolve_partition_key(execution, 'nexus_single', TABLE) is None


def test_partition_key_ignores_sinks_for_other_tables() -> None:
    from nexus.distributed.export.follow import resolve_partition_key

    other = {
        'storage_scope': 'principal_v1',
        'principal_scope_id': 'b' * 32,
        'parent_created_at': '2026-09-20T09:00:00',
    }
    execution = _envelope_execution(other)

    # Reporting another table's key would prune to a partition holding none of our rows.
    assert resolve_partition_key(execution, 'nexus_single', TABLE) is None


@pytest.mark.parametrize('value', ['', '   ', 'not-a-datetime', 17])
def test_unusable_envelope_partition_key_is_rejected(value: Any) -> None:
    """Better to stop than to quietly fall back to scanning every partition."""
    from nexus.distributed.export.follow import resolve_partition_key

    execution = _envelope_execution({**SCOPE, 'parent_created_at': value})

    with pytest.raises(ExportError, match='partition key'):
        resolve_partition_key(execution, 'nexus_single', TABLE)


def test_missing_envelope_resolves_to_no_partition_key() -> None:
    from nexus.distributed.export.follow import resolve_partition_key

    assert resolve_partition_key({'execution_id': 'exe'}, 'nexus_single', TABLE) is None


# --- one table name, two backends ----------------------------------------------------
# The principal table name is derived from the owner alone, so a Run writing to both a
# classic and a Boundless backend publishes the same name twice under different schemas.
# Reading a key out of whichever entry carries one would hand a Boundless partition
# predicate to a classic connection, whose table has no such column -- 1054, not a wider
# scan. The backend name is the only thing that separates them, and the envelope has it.

_OTHER_PARTITION_KEY = PARTITION_KEY_ISO


def _two_backend_execution(*, spec_type: str = 'nexus_single') -> dict[str, Any]:
    return envelope_execution(
        sink('tdsql', dict(SCOPE)),
        sink('tdsql-boundless', {**SCOPE, 'parent_created_at': _OTHER_PARTITION_KEY}),
        spec_type=spec_type,
    )


def test_one_table_on_two_backends_is_not_guessed() -> None:
    from nexus.distributed.export.follow import AmbiguousTraceBackendError, resolve_partition_key

    with pytest.raises(AmbiguousTraceBackendError, match='several backends'):
        resolve_partition_key(_two_backend_execution(), 'nexus_single', TABLE)


@pytest.mark.parametrize(
    ('backend', 'expected'),
    [('tdsql', None), ('tdsql-boundless', '2026-09-21 10:30:00.000000')],
)
def test_the_named_backend_decides_the_schema(backend: str, expected: str | None) -> None:
    """Same Execution, same table, opposite dialects -- the name is what tells them apart."""
    from nexus.distributed.export.follow import resolve_partition_key

    resolved = resolve_partition_key(_two_backend_execution(), 'nexus_single', TABLE, backend=backend)

    assert resolved == expected


def test_a_backend_the_envelope_never_published_is_rejected() -> None:
    """Silently falling back to the classic dialect here would be another guess."""
    from nexus.distributed.export.follow import resolve_partition_key

    with pytest.raises(ExportError, match='no trace sink named'):
        resolve_partition_key(_two_backend_execution(), 'nexus_single', TABLE, backend='tdsql-elsewhere')


def test_a_named_backend_writing_another_table_is_rejected() -> None:
    """The name and the table must agree; one backend can serve several principals."""
    from nexus.distributed.export.follow import resolve_partition_key

    execution = _envelope_execution(
        principal_scope(parent_created_at=_OTHER_PARTITION_KEY)
        | {
            'principal_scope_id': OTHER_PRINCIPAL_SCOPE_ID,
        }
    )

    with pytest.raises(ExportError, match='no trace sink named'):
        resolve_partition_key(execution, 'nexus_single', TABLE, backend='my-db')


# --- the backend choice: dialect and partition key as one decision --------------------
# Inferring the dialect from whether a key was found is what let a Boundless predicate
# reach a classic connection, whose table has no such column.


def test_the_backend_choice_pairs_the_dialect_with_its_partition_key() -> None:
    from nexus.distributed.export.follow import resolve_trace_backend
    from nexus.runner.trace.tdsql.schema import BOUNDLESS_DIALECT
    from nexus.runner.trace.tdsql.sink_source import TDSQLBoundlessTraceSource

    execution = _envelope_execution({**SCOPE, 'parent_created_at': PARTITION_KEY_ISO})

    choice = resolve_trace_backend(execution, 'nexus_single', TABLE)

    assert choice.dialect == BOUNDLESS_DIALECT
    assert choice.partition_key == PARTITION_KEY_SQL
    # Selected by class, matching how the sinks are registered.
    assert choice.source_class() is TDSQLBoundlessTraceSource


def test_a_classic_backend_choice_carries_no_partition_key() -> None:
    """The classic table has no such column, so the reader must not name it."""
    from nexus.distributed.export.follow import resolve_trace_backend
    from nexus.runner.trace.tdsql.schema import TDSQL_DIALECT
    from nexus.runner.trace.tdsql.sink_source import TDSQLTraceSource

    choice = resolve_trace_backend(_envelope_execution(dict(SCOPE)), 'nexus_single', TABLE)

    assert choice.dialect == TDSQL_DIALECT
    assert choice.partition_key is None
    assert choice.source_class() is TDSQLTraceSource
    assert 'parent_created_at' not in choice.scope(table=TABLE, execution_id='exe', spec_type='nexus_single')


def test_an_explicit_partition_key_overrides_the_envelope() -> None:
    """--parent-created-at serves the case where the envelope is unavailable."""
    from nexus.distributed.export.follow import resolve_trace_backend
    from nexus.runner.trace.tdsql.schema import BOUNDLESS_DIALECT

    choice = resolve_trace_backend(
        _envelope_execution(dict(SCOPE)),
        'nexus_single',
        TABLE,
        override_partition_key=PARTITION_KEY_SQL,
    )

    assert choice.dialect == BOUNDLESS_DIALECT
    assert choice.partition_key == PARTITION_KEY_SQL


def test_a_backend_choice_scope_names_the_principal_and_the_run() -> None:
    from nexus.distributed.export.follow import resolve_trace_backend

    execution = _envelope_execution({**SCOPE, 'parent_created_at': PARTITION_KEY_ISO})
    choice = resolve_trace_backend(execution, 'nexus_single', TABLE)

    assert choice.scope(table=TABLE, execution_id='exe', spec_type='nexus_single') == {
        'storage_scope': 'principal_v1',
        'principal_scope_id': PRINCIPAL_SCOPE_ID,
        'run_id': 'exe',
        'parent_created_at': PARTITION_KEY_SQL,
    }
    # A Batch is addressed by its parent, so the key naming the Execution differs.
    assert 'parent_execution_id' in choice.scope(table=TABLE, execution_id='exe', spec_type='nexus_batch')


def test_the_backend_choice_refuses_a_table_no_writer_could_have_produced() -> None:
    """``removeprefix`` used to return an unrelated name whole, as though it were a scope id."""
    from nexus.distributed.export.follow import resolve_trace_backend

    choice = resolve_trace_backend(_envelope_execution(dict(SCOPE)), 'nexus_single', TABLE)

    with pytest.raises(ValueError, match='principal trace table'):
        choice.scope(table='some_other_table', execution_id='exe', spec_type='nexus_single')


def test_an_ambiguous_envelope_is_not_resolved_into_a_dialect() -> None:
    from nexus.distributed.export.follow import AmbiguousTraceBackendError, resolve_trace_backend

    with pytest.raises(AmbiguousTraceBackendError, match='several backends'):
        resolve_trace_backend(_two_backend_execution(), 'nexus_single', TABLE)


def test_a_single_backend_still_resolves_without_being_named() -> None:
    """The flag stays optional: nothing is ambiguous until two backends share the table."""
    from nexus.distributed.export.follow import resolve_partition_key

    execution = _envelope_execution({**SCOPE, 'parent_created_at': _OTHER_PARTITION_KEY})

    assert resolve_partition_key(execution, 'nexus_single', TABLE) == '2026-09-21 10:30:00.000000'


def _mixed_sink_execution(*sinks: dict[str, Any]) -> dict[str, Any]:
    return envelope_execution(*sinks)


_TRACE_SINK = sink('tdsql-boundless', {**SCOPE, 'parent_created_at': _OTHER_PARTITION_KEY})
# Published alongside the trace sink on an ordinary Run, and carrying no principal scope.
_ARTIFACT_SINK = sink('artifact', {'kind': 'artifact', 'uri': 'cos://bucket/key'})


def test_a_sink_that_did_not_write_this_table_is_not_a_backend_for_it() -> None:
    """Only sinks bearing this table's principal scope can speak for its schema.

    An artifact sink has no principal scope, so it never wrote this table -- but it was
    still counted as a candidate, and its absent partition key read as a second backend
    disagreeing about the dialect. Every Run publishing artifacts alongside traces was
    reported ambiguous, which is to say it could not be exported without naming a backend.
    """
    from nexus.distributed.export.follow import resolve_partition_key

    execution = _mixed_sink_execution(_TRACE_SINK, _ARTIFACT_SINK)

    assert resolve_partition_key(execution, 'nexus_single', TABLE) == '2026-09-21 10:30:00.000000'


def test_an_envelope_with_no_trace_sink_for_this_table_resolves_to_nothing() -> None:
    """Absence is the classic answer -- no key, hence no partition predicate."""
    from nexus.distributed.export.follow import resolve_partition_key

    assert resolve_partition_key(_mixed_sink_execution(_ARTIFACT_SINK), 'nexus_single', TABLE) is None


def test_another_principal_table_is_not_a_backend_for_this_one() -> None:
    """Two owners, two tables: the other one's key would prune away all our rows."""
    from nexus.distributed.export.follow import resolve_partition_key

    other = {
        'name': 'tdsql-boundless',
        'metadata': {
            'storage_scope': 'principal_v1',
            'principal_scope_id': 'c' * 32,
            'parent_created_at': _OTHER_PARTITION_KEY,
        },
    }

    assert resolve_partition_key(_mixed_sink_execution(other), 'nexus_single', TABLE) is None


def test_two_backends_on_this_table_are_still_ambiguous_among_other_sinks() -> None:
    """Narrowing the candidates must not lose the ambiguity the narrowing is for."""
    from nexus.distributed.export.follow import AmbiguousTraceBackendError, resolve_partition_key

    classic = {'name': 'tdsql', 'metadata': dict(SCOPE)}
    execution = _mixed_sink_execution(_TRACE_SINK, _ARTIFACT_SINK, classic)

    with pytest.raises(AmbiguousTraceBackendError, match='several backends'):
        resolve_partition_key(execution, 'nexus_single', TABLE)

    # And naming one still settles it.
    assert (
        resolve_partition_key(execution, 'nexus_single', TABLE, backend='tdsql-boundless')
        == '2026-09-21 10:30:00.000000'
    )


def test_checkpoint_identity_records_the_backend_only_when_named() -> None:
    """Two backends serve one table and their rows are not interchangeable, so a resume
    must not switch between them -- while checkpoints written before the option stay
    resumable."""
    import argparse

    from nexus.distributed.export.cli import _source_identity

    def identity(backend: str | None) -> dict[str, Any]:
        args = argparse.Namespace(
            control_url='https://control.example',
            execution_id='exe',
            db_host='db.example',
            db_port=3306,
            db_name='runner',
            trace_table=TABLE,
            trace_backend=backend,
        )
        return _source_identity(args, 'nexus_single')

    assert 'trace_backend' not in identity(None)
    assert identity('tdsql-boundless')['trace_backend'] == 'tdsql-boundless'
    assert identity('tdsql') != identity('tdsql-boundless')


# --- partition key for a whole-Batch export -----------------------------------------
# NexusBatchDriver publishes only runtime_info.batch, never the _artifacts envelope the
# resolution above reads, so a Batch parent has none and the export used to scan every day
# partition. The parent's own created_at is the answer: the repository freezes it into every
# child spec, so all rows beneath the parent carry exactly that value.
#
# Which makes the schema probe load-bearing rather than defensive. The predicate is not
# merely redundant on a classic table -- the column is absent, so it fails the query with
# 1054. Losing the key costs a wider scan; adding it wrongly costs the export.

_BATCH_PARENT_CREATED_AT = '2026-09-23T02:15:30.123456+00:00'
_BATCH_PARTITION_KEY = '2026-09-23 02:15:30.123456'


def _batch_parent_execution(execution_id: str = 'exe_parent') -> dict[str, Any]:
    """A Batch parent as the driver really leaves it: runtime_info.batch and nothing else."""
    return {
        'execution_id': execution_id,
        'spec_type': 'nexus_batch',
        'status': 'succeeded',
        'attempt_count': 1,
        'created_at': _BATCH_PARENT_CREATED_AT,
        'current_assignment': {'runtime_info': {'batch': {'generation_status': 'complete'}}},
    }


class _ProbeReader:
    """Reader that answers the schema probe without a database."""

    def __init__(self, *, boundless: bool) -> None:
        self.table = TABLE
        self.parent_created_at: str | None = None
        self._boundless = boundless
        self.probes = 0
        self.used_partitions = []

    async def has_partition_column(self) -> bool:
        self.probes += 1
        return self._boundless

    async def read_many(self, ids):
        from nexus.distributed.export.types import TraceResult

        self.used_partitions.append(self.parent_created_at)
        return {ids[0]: TraceResult(output={'success': True})}


async def _resolve_for_batch(reader: Any, execution: dict[str, Any], tmp_path: Path) -> None:
    from nexus.distributed.export.results import export_execution

    class Api:
        async def get_execution(self, execution_id):
            return execution

        async def get_batch_summary(self, execution_id):
            return {
                'generation_status': 'complete',
                'submission_closed': True,
                'summary_version': 1,
                'materialized_count': 1,
                'total_items': 1,
                'terminal_items': 1,
                'item_status_counts': {'succeeded': 1},
            }

        async def get_batch_items(self, execution_id, cursor):
            return {
                'items': [
                    {
                        'item_id': 'item',
                        'status': 'succeeded',
                        'current_attempt_no': 0,
                        'current_child_execution_id': 'child',
                    }
                ],
                'next_cursor': None,
            }

        async def get_control_events(self, execution_id):
            return []

    await export_execution(
        api=Api(),
        execution_id=execution['execution_id'],
        output=tmp_path / 'out.jsonl',
        force=False,
        trace_reader=reader,
    )


def test_batch_parent_has_no_envelope_to_resolve() -> None:
    """The premise: envelope resolution cannot serve a Batch parent, so a fallback is needed."""
    from nexus.distributed.export.follow import resolve_partition_key

    assert resolve_partition_key(_batch_parent_execution(), 'nexus_batch', TABLE) is None


async def test_batch_export_fills_the_partition_key_from_the_parent(tmp_path: Path) -> None:
    reader = _ProbeReader(boundless=True)

    await _resolve_for_batch(reader, _batch_parent_execution(), tmp_path)

    assert reader.used_partitions == [_BATCH_PARTITION_KEY]
    assert reader.parent_created_at is None
    assert reader.probes == 1


async def test_batch_export_leaves_classic_tables_unpruned(tmp_path: Path) -> None:
    """A classic table has no such column: the predicate would fail the query, not widen it."""
    reader = _ProbeReader(boundless=False)

    await _resolve_for_batch(reader, _batch_parent_execution(), tmp_path)

    assert reader.used_partitions == [None]
    assert reader.parent_created_at is None
    assert reader.probes == 1


async def test_batch_partition_key_matches_what_the_writer_stored(tmp_path: Path) -> None:
    """Both sides must derive the key from the parent's created_at, or pruning finds nothing.

    The writer's value is not re-derived here but taken the way the repository builds it,
    so this fails if either side changes how the parent timestamp reaches a row.
    """
    from nexus.runner.trace.tdsql.sink_source import normalize_parent_created_at

    parent = _batch_parent_execution()
    parent_created_at = datetime.fromisoformat(parent['created_at'])
    # repositories/batch.py freezes exactly this into every child spec, and nexus_single.py
    # hands it to the sink, which stores it on each row.
    writer_side = normalize_parent_created_at(parent_created_at.isoformat())

    reader = _ProbeReader(boundless=True)
    await _resolve_for_batch(reader, parent, tmp_path)

    assert reader.used_partitions == [writer_side]
    assert reader.parent_created_at is None


async def test_single_child_export_does_not_borrow_its_own_created_at(tmp_path: Path) -> None:
    from nexus.distributed.export.results import TdsqlTraceReader, export_execution
    from nexus.distributed.export.types import TraceResult

    class Reader(TdsqlTraceReader):
        async def has_partition_column(self):
            return True

        async def read_many(self, ids, **kwargs):
            assert ids == ['exe_child']
            assert self.parent_created_at == _BATCH_PARTITION_KEY
            assert kwargs['parent_execution_id'] == 'exe_parent'
            return {'exe_child': TraceResult(output={'success': True})}

    class Api:
        async def get_execution(self, execution_id):
            if execution_id == 'exe_parent':
                return _batch_parent_execution()
            return {
                'execution_id': execution_id,
                'spec_type': 'nexus_single',
                'status': 'succeeded',
                'attempt_count': 1,
                'parent_execution_id': 'exe_parent',
                'created_at': '2026-09-24T04:00:00+00:00',
            }

        async def get_control_events(self, execution_id):
            return []

        async def get_result(self, execution_id):
            pytest.fail('DB already has the result')

    reader = Reader(host='db.test', port=3306, user='reader', password='unused', database='trace', table=TABLE)
    await export_execution(
        api=Api(), execution_id='exe_child', output=tmp_path / 'out.jsonl', force=False, trace_reader=reader
    )
    assert reader.parent_created_at is None


@pytest.mark.parametrize('spec_type', ['nexus_single', 'nexus_live'])
@pytest.mark.parametrize('partition_key', [None, PARTITION_KEY_SQL])
async def test_single_and_live_results_use_a_configured_trace_reader(tmp_path, spec_type, partition_key):
    from nexus.distributed.export.results import export_execution
    from nexus.distributed.export.types import TraceResult

    class Reader(_ProbeReader):
        async def read_many(self, ids):
            assert ids == ['exe']
            return {'exe': TraceResult(output={'success': True, 'task_output': {'value': 42}})}

    reader = Reader(boundless=True)
    reader.parent_created_at = partition_key

    class Api:
        async def get_execution(self, execution_id):
            # One backend publishes this table, so the result reader can trust
            # the envelope and skip the schema probe for the no-override case.
            return envelope_execution(
                sink('tdsql-boundless', {**SCOPE, 'parent_created_at': PARTITION_KEY_SQL}),
                spec_type=spec_type,
                execution_id=execution_id,
                status='succeeded',
                attempt_count=1,
                created_at=PARTITION_KEY_ISO,
            )

        async def get_result(self, execution_id):
            pytest.fail('DB output must take precedence over the result API')

        async def get_control_events(self, execution_id):
            return []

    output = tmp_path / 'results.jsonl'
    await export_execution(api=Api(), execution_id='exe', output=output, force=False, trace_reader=reader)
    assert json.loads(output.read_text())['task_output'] == {'value': 42}
    # An envelope key is used directly; an explicit override is validated once.
    assert reader.probes == (1 if partition_key is not None else 0)
    assert reader.parent_created_at == partition_key
