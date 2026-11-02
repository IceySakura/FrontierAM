"""Disk-prefetch pipeline (ResultSpillFile) and client streaming download tests."""

from __future__ import annotations

import asyncio
import json

from typing import Any, cast

import pytest

from nexus.distributed.export import results
from nexus.distributed.export.client import CloudApi
from nexus.distributed.export.types import ExportError
from nexus.distributed.export.writer import AtomicJsonlOutput, ResultSpillFile, ResultSpillPool


class Api:
    """Fake CloudApi exercising both fetch paths with shared bookkeeping."""

    def __init__(self, outputs: dict[str, Any]):
        self.outputs = outputs
        self.executions = {
            execution_id: {
                'execution_id': execution_id,
                'spec_type': 'nexus_single',
                'status': 'succeeded',
                'attempt_count': 1,
                'current_attempt_id': f'attempt-{execution_id}',
                'created_at': '2026-10-13T00:00:00',
            }
            for execution_id in outputs
        }
        self.result_calls: list[str] = []
        self.ram_calls: list[str] = []
        self.file_calls: list[str] = []
        self.max_concurrent_ram = 0
        self._ram_in_flight = 0
        self.download_started = asyncio.Event()
        self.gate: asyncio.Event | None = None

    async def get_execution(self, execution_id):
        return dict(self.executions[execution_id])

    async def get_control_events(self, execution_id):
        return []

    async def get_result(self, execution_id):
        self.result_calls.append(execution_id)
        self.ram_calls.append(execution_id)
        self._ram_in_flight += 1
        self.max_concurrent_ram = max(self.max_concurrent_ram, self._ram_in_flight)
        try:
            if self.gate is not None:
                await self.gate.wait()
            return self.outputs.get(execution_id)
        finally:
            self._ram_in_flight -= 1

    async def get_result_to_file(self, execution_id, fileobj, chunk_size=1 << 20, scan=None):
        self.result_calls.append(execution_id)
        self.file_calls.append(execution_id)
        self.download_started.set()
        if self.gate is not None:
            await self.gate.wait()
        value = self.outputs.get(execution_id)
        if value is None:
            return None
        data = json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()
        fileobj.write(data)
        if scan is not None:
            scan.feed(data)
        return len(data)


def _record_plan(execution_id: str, output: Any) -> results._RecordPlan:
    single = results._SingleExport(
        execution_id=execution_id,
        run_id=execution_id,
        status='succeeded' if output is not None else 'failed',
        retry_count=0,
        context={},
    )
    return results._RecordPlan(single=single)


async def _run_export(tmp_path, api: Api, plans) -> list[dict[str, Any]]:
    writer = AtomicJsonlOutput(tmp_path / 'out.jsonl', force=True)
    await writer.open()
    written = await results._write_records(writer, cast(Any, api), plans, spill=ResultSpillPool(tmp_path / 'out.jsonl'))
    await writer.close_and_validate(expected_count=written, require_sorted=False)
    await writer.publish()
    return [json.loads(line) for line in (tmp_path / 'out.jsonl').read_text().splitlines()]


async def test_spill_pipeline_matches_plan_order_and_drops_files(tmp_path):
    outputs = {
        f'exe-{i}': {'success': True, 'task_output': {'value': f'v{i}'}, 'processor_results': {'i': i}}
        for i in range(4)
    }
    api = Api(outputs)
    plans = [_record_plan(execution_id, output) for execution_id, output in outputs.items()]
    rows = await _run_export(tmp_path, api, plans)

    assert [row['run_id'] for row in rows] == list(outputs)
    # Every API result uses the spill path, including the first record. This
    # keeps a single large response out of the legacy read-bytes path.
    assert api.ram_calls == []
    assert api.file_calls == list(outputs)
    assert not list(tmp_path.glob('out.jsonl.prefetch-*'))


async def test_spill_pipeline_missing_result_matches_missing(tmp_path):
    outputs = {'exe-a': {'success': True, 'task_output': {'v': 1}}, 'exe-b': None}
    api = Api(outputs)
    plans = [_record_plan(execution_id, output) for execution_id, output in outputs.items()]
    plans[1].single.status = 'failed'
    plans[1].single.error = {'type': 'TaskFailed'}
    rows = await _run_export(tmp_path, api, plans)

    by_id = {row['run_id']: row for row in rows}
    assert by_id['exe-b']['task_output'] is None
    assert by_id['exe-b']['error'] == {'type': 'TaskFailed'}


async def test_spill_pipeline_keeps_single_record_in_ram_while_downloading(tmp_path):
    outputs = {f'exe-{i}': {'success': True, 'task_output': {'v': i}} for i in range(4)}
    api = Api(outputs)
    saw_download_during_write = False

    class Writer(AtomicJsonlOutput):
        async def write_records(self, records):
            nonlocal saw_download_during_write
            saw_download_during_write = saw_download_during_write or api.download_started.is_set()
            await super().write_records(records)

        async def write_row_chunks(self, chunks, **kwargs):
            nonlocal saw_download_during_write
            saw_download_during_write = saw_download_during_write or api.download_started.is_set()
            await super().write_row_chunks(chunks, **kwargs)

    writer = Writer(tmp_path / 'out.jsonl', force=True)
    await writer.open()
    plans = [_record_plan(execution_id, output) for execution_id, output in outputs.items()]
    written = await results._write_records(writer, cast(Any, api), plans, spill=ResultSpillPool(writer.output))
    await writer.close_and_validate(expected_count=written, require_sorted=False)

    assert api.max_concurrent_ram == 0  # no record uses the legacy in-RAM fetch path
    assert saw_download_during_write  # prefetch overlapped the write


async def test_cancel_cleans_up_spill_files(tmp_path):
    outputs = {f'exe-{i}': {'success': True, 'task_output': {'v': i}} for i in range(4)}
    api = Api(outputs)
    api.gate = asyncio.Event()  # block every fetch
    writer = AtomicJsonlOutput(tmp_path / 'out.jsonl', force=True)
    await writer.open()
    plans = [_record_plan(execution_id, output) for execution_id, output in outputs.items()]
    task = asyncio.create_task(
        results._write_records(writer, cast(Any, api), plans, spill=ResultSpillPool(tmp_path / 'out.jsonl'))
    )
    await asyncio.sleep(0.1)  # let the pipeline start head + prefetch
    assert api.result_calls  # head fetch and prefetch download both started
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert not list(tmp_path.glob('out.jsonl.prefetch-*'))


async def test_prefetch_download_disk_error_fails_the_export(tmp_path):
    outputs = {'exe-a': {'success': True, 'task_output': {'v': 1}}, 'exe-b': {'success': True, 'task_output': {'v': 2}}}
    api = Api(outputs)

    async def boom(execution_id, fileobj, chunk_size=1 << 20, scan=None):
        fileobj.write(b'partial')
        raise OSError(28, 'No space left on device')

    api.get_result_to_file = boom  # type: ignore[method-assign]
    writer = AtomicJsonlOutput(tmp_path / 'out.jsonl', force=True)
    await writer.open()
    plans = [_record_plan(execution_id, output) for execution_id, output in outputs.items()]
    with pytest.raises(ExportError, match='cannot write prefetch output'):
        await results._write_records(writer, cast(Any, api), plans, spill=ResultSpillPool(tmp_path / 'out.jsonl'))
    await writer.abort()
    assert not list(tmp_path.glob('out.jsonl.prefetch-*'))


def test_open_sweeps_leftover_prefetch_files(tmp_path):
    output = tmp_path / 'out.jsonl'
    leftover = output.with_name('out.jsonl.prefetch-9')
    leftover.write_bytes(b'stale')

    async def run():
        writer = AtomicJsonlOutput(output, force=True)
        await writer.open()
        await writer.abort()

    asyncio.run(run())
    assert not leftover.exists()


def test_spill_file_load_and_discard(tmp_path):
    file = ResultSpillFile(tmp_path / 'out.jsonl', 1)
    body = json.dumps({'success': True}).encode()
    file.path.write_bytes(body)
    assert file.load_raw() == body
    file.discard()
    assert not file.path.exists()
    file.discard()  # idempotent


def test_spill_pool_cleanup_removes_unconsumed_files(tmp_path):
    pool = ResultSpillPool(tmp_path / 'out.jsonl')
    first = pool.new_file()
    second = pool.new_file()
    first.path.write_bytes(b'a')
    second.path.write_bytes(b'b')
    pool.cleanup()
    assert not first.path.exists()
    assert not second.path.exists()
    pool.cleanup()  # idempotent


# ---------------------------------------------------------------------------
# client.get_result_to_file: streaming download with retry semantics
# ---------------------------------------------------------------------------


class _FakeContent:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def read(self, n):
        if not self._chunks:
            return b''
        chunk = self._chunks.pop(0)
        if isinstance(chunk, BaseException):
            raise chunk
        return chunk


class _FakeResponse:
    def __init__(self, status, chunks=(), headers=None):
        self.status = status
        self.headers = headers or {}
        self._chunks = list(chunks)
        self.content = _FakeContent(chunks)

    async def read(self):
        parts = []
        while self._chunks and not isinstance(self._chunks[0], BaseException):
            parts.append(self._chunks.pop(0))
        return b''.join(parts)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append({'url': url, 'headers': headers, 'timeout': timeout})
        return self._responses.pop(0)


def _client(responses) -> CloudApi:
    async def _no_sleep(_seconds):
        return None

    session = _FakeSession(responses)
    client = CloudApi(
        cast(Any, session),
        control_url='http://control',
        application_url='http://app',
        token='token',
        sleep=_no_sleep,
    )
    client.session_calls = session.calls  # type: ignore[attr-defined]
    return client


def _connection_error():
    import aiohttp

    return aiohttp.ClientConnectionError('connection lost')


async def test_get_result_to_file_streams_body(tmp_path):
    target = tmp_path / 'body.bin'
    file = target.open('wb')
    client = _client([_FakeResponse(200, [b'{"success":', b' true}'])])
    length = await client.get_result_to_file('exe-1', file)
    file.close()
    assert length == len(b'{"success": true}')
    assert target.read_bytes() == b'{"success": true}'


async def test_get_result_to_file_404_returns_none(tmp_path):
    file = (tmp_path / 'body.bin').open('wb')
    client = _client([_FakeResponse(404)])
    assert await client.get_result_to_file('exe-1', file) is None
    file.close()


async def test_get_result_to_file_retries_transient_and_truncates(tmp_path):
    target = tmp_path / 'body.bin'
    file = target.open('wb')
    client = _client([
        _FakeResponse(500),
        _FakeResponse(200, [b'{"succ', _connection_error(), b'ess": true}']),
        _FakeResponse(200, [b'{"success": true}']),
    ])
    length = await client.get_result_to_file('exe-1', file)
    file.close()
    assert length == len(b'{"success": true}')
    # partial bytes from the failed mid-stream attempt must not survive
    assert target.read_bytes() == b'{"success": true}'


async def test_get_result_to_file_exhausts_retries(tmp_path):
    file = (tmp_path / 'body.bin').open('wb')
    client = _client([_FakeResponse(503) for _ in range(3)])
    with pytest.raises(ExportError, match='failed after 3 attempts'):
        await client.get_result_to_file('exe-1', file)
    file.close()


async def test_get_result_to_file_other_client_errors_raise_immediately(tmp_path):
    file = (tmp_path / 'body.bin').open('wb')
    client = _client([_FakeResponse(403)])
    with pytest.raises(ExportError, match='HTTP 403'):
        await client.get_result_to_file('exe-1', file)
    file.close()


async def test_result_download_uses_idle_read_timeout_not_wall_clock(tmp_path):
    client = _client([_FakeResponse(200, [b'{"success": true}'])])
    file = (tmp_path / 'body.bin').open('wb')
    length = await client.get_result_to_file('exe-1', file)
    file.close()
    assert length == len(b'{"success": true}')
    timeout = client.session_calls[0]['timeout']
    assert timeout is not None and timeout.total is None
    assert timeout.connect == 30 and timeout.sock_read == 900.0


async def test_result_stream_timeout_env_override(monkeypatch):
    from nexus.distributed.export.client import _result_stream_timeout

    monkeypatch.delenv('NEXUS_EXPORT_RESULT_SOCK_READ_SECONDS', raising=False)
    timeout = _result_stream_timeout()
    assert timeout.total is None
    assert timeout.connect == 30 and timeout.sock_connect == 30
    assert timeout.sock_read == 900.0

    monkeypatch.setenv('NEXUS_EXPORT_RESULT_SOCK_READ_SECONDS', '120')
    assert _result_stream_timeout().sock_read == 120.0

    monkeypatch.setenv('NEXUS_EXPORT_RESULT_SOCK_READ_SECONDS', 'nonsense')
    assert _result_stream_timeout().sock_read == 900.0

    monkeypatch.setenv('NEXUS_EXPORT_RESULT_SOCK_READ_SECONDS', '0')
    assert _result_stream_timeout().sock_read == 1.0


async def test_get_result_passes_streaming_timeout_to_get_json(monkeypatch):
    client = _client([_FakeResponse(200, [b'{"success": true}'])])
    assert await client.get_result('exe-1') == {'success': True}
    assert client.session_calls[0]['timeout'] is not None and client.session_calls[0]['timeout'].total is None


async def test_get_result_to_file_survives_slow_legacy_server(tmp_path, monkeypatch):
    """A legacy buffered server can stay silent for minutes before responding.

    With the old session-wide total=300s this aborted (and re-downloaded three
    times); the idle-read bound must tolerate it while still tripping when the
    silence exceeds the configured sock_read.
    """
    import aiohttp
    from aiohttp import web

    async def slow_headers(request):
        await asyncio.sleep(1.2)
        return web.Response(body=b'{"success": true, "task_output": {"n": 1}}')

    app = web.Application()
    app.router.add_get('/v1/executions/exe/result', slow_headers)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    base = f'http://127.0.0.1:{runner.addresses[0][1]}'
    try:
        async with aiohttp.ClientSession() as session:
            api = CloudApi(session, control_url='http://control', application_url=base, token='t')
            target = tmp_path / 'body.bin'
            with target.open('wb') as fileobj:
                length = await api.get_result_to_file('exe', fileobj)
            assert length == len(b'{"success": true, "task_output": {"n": 1}}')

            monkeypatch.setenv('NEXUS_EXPORT_RESULT_SOCK_READ_SECONDS', '0.5')
            with target.open('wb') as fileobj:
                with pytest.raises(ExportError, match='failed after 3 attempts'):
                    await api.get_result_to_file('exe', fileobj)
    finally:
        await runner.cleanup()


async def test_get_result_fallback_path_bounds_idle_reads(tmp_path, monkeypatch):
    import aiohttp
    from aiohttp import web

    async def slow_headers(request):
        await asyncio.sleep(1.2)
        return web.Response(body=b'{"success": true}')

    app = web.Application()
    app.router.add_get('/v1/executions/exe/result', slow_headers)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '127.0.0.1', 0).start()
    base = f'http://127.0.0.1:{runner.addresses[0][1]}'
    try:
        async with aiohttp.ClientSession() as session:
            api = CloudApi(session, control_url='http://control', application_url=base, token='t')
            assert await api.get_result('exe') == {'success': True}

            monkeypatch.setenv('NEXUS_EXPORT_RESULT_SOCK_READ_SECONDS', '0.5')
            with pytest.raises(ExportError, match='failed after 3 attempts'):
                await api.get_result('exe')
    finally:
        await runner.cleanup()


@pytest.mark.parametrize('style', ['legacy-json-response', 'streamed-raw-artifact'])
async def test_prefetched_result_accepts_legacy_and_streamed_bodies(tmp_path, style):
    output = {'success': True, 'task_output': {'note': '中文\U0001f600'}, 'processor_results': {'score': 1}}
    if style == 'legacy-json-response':
        body = json.dumps(output).encode()  # ensure_ascii + ", "/": " separators
    else:
        body = json.dumps(output, ensure_ascii=False, separators=(',', ':')).encode()

    spill = ResultSpillFile(tmp_path / 'out.jsonl', 1)
    spill.path.write_bytes(body)
    with spill.open_view() as view:
        assert results._prefetched_result('exe', view) == output
