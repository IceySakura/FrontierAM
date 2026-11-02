from __future__ import annotations

import asyncio
import hashlib
import json

from types import SimpleNamespace

import pytest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.artifacts.run_result import encode_run_result_content
from nexus.distributed.config import DistributedConfig
from nexus.distributed.errors import TemporaryUnavailableError
from nexus.distributed.gateways.application.app import create_application_app
from nexus.distributed.gateways.application.routes import executions
from nexus.distributed.gateways.application.routes._result_pipeline import (
    RESULT_PIPELINE_KEY,
    ResultEncoder,
    ResultPipeline,
)
from nexus.distributed.gateways.common.keys import ARTIFACT_SPOOL_KEY, ARTIFACT_STORE_KEY


async def test_result_queue_is_bounded_and_cancelled_waiters_release_their_place():
    pipeline = ResultPipeline(DistributedConfig(artifact_run_result_max_inflight=1, artifact_run_result_max_waiters=1))
    async with pipeline.admit():
        waiter = asyncio.create_task(pipeline.admit().__aenter__())
        await asyncio.sleep(0)
        assert pipeline.waiting == 1
        with pytest.raises(TemporaryUnavailableError, match='result_queue_full'):
            async with pipeline.admit():
                pytest.fail('queue limit was not enforced')
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert pipeline.waiting == 0
    assert pipeline.active == 0
    async with pipeline.admit():
        assert pipeline.active == 1


async def test_result_queue_timeout_does_not_leak_admission():
    pipeline = ResultPipeline(
        DistributedConfig(artifact_run_result_max_inflight=1, artifact_run_result_admission_timeout=0.01)
    )
    async with pipeline.admit():
        with pytest.raises(TemporaryUnavailableError, match='result_queue_timeout'):
            async with pipeline.admit():
                pytest.fail('queue wait did not time out')
        assert pipeline.active == 1
        assert pipeline.waiting == 0
    assert pipeline.active == 0


def fixture_app(tmp_path, **overrides):
    output = {'z': '中文🙂' * 20, 'a': {'success': True, 'error': None}}
    content = encode_run_result_content(
        status='succeeded', output=output, error=None, compression='zstd', sort_keys=False
    )
    ref = ArtifactRef(
        artifact_id='result', kind='run_result', size_bytes=len(content), sha256=hashlib.sha256(content).hexdigest()
    )

    class Store:
        async def get(self, stored_object, destination, **kwargs):
            destination.write_bytes(content)

        async def close(self):
            pass

    config = DistributedConfig(artifact_upload_spool_dir=str(tmp_path), **overrides)
    app = create_application_app(session_factory=None, config=config)
    app[ARTIFACT_STORE_KEY] = Store()
    return app, ref, output


@pytest.mark.parametrize('method', ['GET', 'HEAD'])
async def test_result_stream_preserves_json_headers_and_bounds_write_chunks(tmp_path, monkeypatch, method):
    app, ref, output = fixture_app(tmp_path, artifact_run_result_response_chunk_bytes=31)
    writes = []
    original = web.StreamResponse.write

    async def write(response, data):
        writes.append(len(data))
        return await original(response, data)

    finished = asyncio.Event()

    async def handler(request):
        try:
            return await executions._read_result_response(request, stored_object=None, ref=ref)
        finally:
            finished.set()

    app.router.add_get('/fixture-result', handler)
    monkeypatch.setattr(web.StreamResponse, 'write', write)
    async with TestClient(TestServer(app)) as client:
        response = await client.request(method, '/fixture-result', headers={'Origin': 'https://example.test'})
        expected = json.dumps(output, ensure_ascii=False, separators=(',', ':')).encode()
        assert response.status == 200
        assert 'Content-Length' not in response.headers
        if method == 'GET':
            assert response.headers['Transfer-Encoding'] == 'chunked'
        assert response.headers['Content-Type'] == 'application/json; charset=utf-8'
        assert response.headers['Access-Control-Allow-Origin'] == 'https://example.test'
        assert await response.read() == (expected if method == 'GET' else b'')
        if method == 'GET':
            assert max(writes) <= 31
            assert sum(writes) == len(expected)
        else:
            assert writes == []
        await asyncio.wait_for(finished.wait(), timeout=2)
        assert app[RESULT_PIPELINE_KEY].active == 0
        assert app[ARTIFACT_SPOOL_KEY].reserved_bytes == 0
        assert not list(tmp_path.glob('*.result'))


async def test_slow_send_keeps_request_and_spool_but_releases_cpu(tmp_path, monkeypatch):
    app, ref, _ = fixture_app(tmp_path, artifact_run_result_max_inflight=1)
    spool = app[ARTIFACT_SPOOL_KEY]
    await spool.start()
    entered = asyncio.Event()
    sending = asyncio.Event()
    calls = 0

    async def send(request, chunks):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await sending.wait()
        return web.Response(body=b''.join([chunk async for chunk in chunks]))

    monkeypatch.setattr(executions, '_send_result_response', send)
    request = SimpleNamespace(app=app)
    first = asyncio.create_task(executions._read_result_response(request, stored_object=None, ref=ref))
    await asyncio.wait_for(entered.wait(), timeout=2)
    pipeline = app[RESULT_PIPELINE_KEY]
    assert pipeline.active == 1
    assert await app[RESULT_PIPELINE_KEY].transform(lambda: 'available') == 'available'
    assert spool.cached_bytes == ref.size_bytes
    assert list((tmp_path / 'cache').iterdir())
    second = asyncio.create_task(executions._read_result_response(request, stored_object=None, ref=ref))
    await asyncio.sleep(0)
    assert pipeline.waiting == 1
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await asyncio.wait_for(second, timeout=2)
    assert pipeline.active == pipeline.waiting == 0
    await spool.close()


async def test_compressed_result_ignores_legacy_whole_document_memory_estimate(tmp_path, monkeypatch):
    app, ref, output = fixture_app(tmp_path, artifact_run_result_memory_budget_bytes=1)
    spool = app[ARTIFACT_SPOOL_KEY]
    await spool.start()

    async def send(request, chunks):
        return web.Response(body=b''.join([chunk async for chunk in chunks]))

    monkeypatch.setattr(executions, '_send_result_response', send)
    try:
        response = await executions._read_result_response(SimpleNamespace(app=app), stored_object=None, ref=ref)
        assert json.loads(response.body) == output
        assert app[RESULT_PIPELINE_KEY].active == 0
        assert spool.reserved_bytes == 0
        assert not list(tmp_path.glob('*.result'))
    finally:
        await app[RESULT_PIPELINE_KEY].close()
        await spool.close()


async def test_download_admission_is_released_while_transform_is_still_running(tmp_path, monkeypatch):
    import threading

    from nexus.distributed.gateways.common.keys import ARTIFACT_DOWNLOAD_ADMISSION_KEY

    app, ref, _ = fixture_app(tmp_path, artifact_download_result_concurrency=1)
    spool = app[ARTIFACT_SPOOL_KEY]
    await spool.start()
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    original_encode = ResultEncoder.prepare
    store = app[ARTIFACT_STORE_KEY]
    original_get = store.get
    downloads = 0

    def encode(*args):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return original_encode(*args)

    async def get(*args, **kwargs):
        nonlocal downloads
        await original_get(*args, **kwargs)
        downloads += 1

    async def send(request, chunks):
        return web.Response(body=b''.join([chunk async for chunk in chunks]))

    monkeypatch.setattr(ResultEncoder, 'prepare', encode)
    monkeypatch.setattr(executions, '_send_result_response', send)
    monkeypatch.setattr(store, 'get', get)
    request = SimpleNamespace(app=app)
    tasks = [asyncio.create_task(executions._read_result_response(request, stored_object=None, ref=ref))]
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        tasks.append(asyncio.create_task(executions._read_result_response(request, stored_object=None, ref=ref)))
        await asyncio.sleep(0.05)
        async with app[ARTIFACT_DOWNLOAD_ADMISSION_KEY].reserve('run_result', size_bytes=0):
            assert spool.cached_bytes == ref.size_bytes
            assert downloads == 1
            assert not tasks[0].done()
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=3)
        await spool.close()
    assert app[RESULT_PIPELINE_KEY].active == 0


async def test_stalled_response_write_closes_transport(tmp_path, monkeypatch):
    app, _, _ = fixture_app(tmp_path, artifact_download_write_idle_timeout=0.01)
    transport_closed = asyncio.Event()
    response_closed = asyncio.Event()

    class Response:
        def __init__(self, *, headers):
            self.headers = headers

        async def prepare(self, request):
            pass

        async def write(self, data):
            await asyncio.Event().wait()

        def force_close(self):
            response_closed.set()

    monkeypatch.setattr(executions.web, 'StreamResponse', Response)
    request = SimpleNamespace(app=app, headers={}, method='GET', transport=SimpleNamespace(close=transport_closed.set))

    async def chunks():
        yield b'result'

    with pytest.raises(asyncio.TimeoutError):
        await executions._send_result_response(request, chunks())
    assert response_closed.is_set()
    assert transport_closed.is_set()


async def test_result_cpu_pool_preserves_context_and_leaves_default_executor_free():
    import contextvars
    import threading

    from nexus.utils.common import to_thread_complete

    value = contextvars.ContextVar('result-test', default='missing')
    token = value.set('request-context')
    pipeline = ResultPipeline(DistributedConfig(artifact_run_result_decode_concurrency=1))
    entered = threading.Event()
    release = threading.Event()

    def transform():
        entered.set()
        assert release.wait(5)
        return value.get(), threading.current_thread().name

    task = asyncio.create_task(pipeline.transform(transform))
    try:
        while not entered.is_set():
            await asyncio.sleep(0.001)
        assert await asyncio.wait_for(to_thread_complete(lambda: 42), 1) == 42
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    result, name = await pipeline.transform(lambda: (value.get(), threading.current_thread().name))
    assert result == 'request-context'
    assert name.startswith('result-cpu')
    await pipeline.close()
    value.reset(token)
    with pytest.raises(RuntimeError, match='closed'):
        await pipeline.transform(lambda: None)


@pytest.mark.parametrize('method', ['GET', 'HEAD'])
async def test_response_encodes_only_as_client_consumes(tmp_path, monkeypatch, method):
    app, ref, output = fixture_app(tmp_path, artifact_run_result_response_chunk_bytes=31)
    spool = app[ARTIFACT_SPOOL_KEY]
    pipeline = app[RESULT_PIPELINE_KEY]
    await spool.start()
    writes = []
    encoded = 0
    writing = asyncio.Event()
    release = asyncio.Event()
    original = ResultEncoder.next_chunk

    def next_chunk(encoder):
        nonlocal encoded
        encoded += 1
        assert method != 'HEAD', 'HEAD must not encode the response body'
        return original(encoder)

    class Response:
        def __init__(self, *, headers):
            self.headers = headers

        async def prepare(self, request):
            pass

        async def write(self, chunk):
            writes.append(chunk)
            writing.set()
            await release.wait()

        async def write_eof(self):
            pass

        def force_close(self):
            pass

    monkeypatch.setattr(ResultEncoder, 'next_chunk', next_chunk)
    monkeypatch.setattr(executions.web, 'StreamResponse', Response)
    request = SimpleNamespace(app=app, headers={}, method=method, transport=None)
    task = asyncio.create_task(executions._read_result_response(request, stored_object=None, ref=ref))
    try:
        if method == 'GET':
            await asyncio.wait_for(writing.wait(), timeout=2)
            assert encoded == 1
            assert pipeline.active == 1
            assert spool.cached_bytes == ref.size_bytes
            assert list((tmp_path / 'cache').iterdir())
            assert await asyncio.wait_for(pipeline.transform(lambda: True), timeout=1)
            assert encoded == 1
        release.set()
        await task
        assert b''.join(writes) == (
            json.dumps(output, ensure_ascii=False, separators=(',', ':')).encode() if method == 'GET' else b''
        )
        assert pipeline.active == 0
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await pipeline.close()
        await spool.close()


async def test_large_string_encoding_never_dumps_the_entire_output(tmp_path, monkeypatch):
    output = {'blob': '中文🙂\\\n"' * 100_000, 'a': [1, 2, None, True]}
    path = tmp_path / 'result'
    path.write_bytes(encode_run_result_content(status='succeeded', output=output, error=None, sort_keys=False))
    expected = json.dumps(output, ensure_ascii=False, separators=(',', ':')).encode()
    original = json.dumps
    original_quote = json.encoder.encode_basestring_ascii

    def bounded_quote(value):
        assert len(value) <= 16 * 1024
        return original_quote(value)

    def bounded_dumps(value, **kwargs):
        assert not isinstance(value, dict)
        if isinstance(value, str):
            assert len(value) <= 16 * 1024
        return original(value, **kwargs)

    monkeypatch.setattr(json, 'dumps', bounded_dumps)
    monkeypatch.setattr(json.encoder, 'encode_basestring_ascii', bounded_quote)
    pipeline = ResultPipeline(DistributedConfig(artifact_run_result_response_chunk_bytes=1024 * 1024))
    try:
        async with pipeline.stream(path, timings={}) as stream:
            received = bytearray()
            async for chunk in stream:
                assert 0 < len(chunk) <= 1024 * 1024
                received.extend(chunk)
        assert bytes(received) == expected
    finally:
        await pipeline.close()


async def test_cancelled_encoder_retains_spool_until_worker_and_cleanup_finish(tmp_path, monkeypatch):
    import threading

    app, ref, _ = fixture_app(tmp_path, artifact_run_result_response_chunk_bytes=31)
    spool = app[ARTIFACT_SPOOL_KEY]
    pipeline = app[RESULT_PIPELINE_KEY]
    await spool.start()
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    encoders = []
    original = ResultEncoder.next_chunk

    def blocked(encoder):
        encoders.append(encoder)
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return original(encoder)

    async def send(request, chunks):
        return web.Response(body=b''.join([chunk async for chunk in chunks]))

    monkeypatch.setattr(ResultEncoder, 'next_chunk', blocked)
    monkeypatch.setattr(executions, '_send_result_response', send)
    task = asyncio.create_task(executions._read_result_response(SimpleNamespace(app=app), stored_object=None, ref=ref))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()
        assert pipeline.active == 1
        assert spool.cached_bytes == ref.size_bytes
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(encoders) == 1
        assert encoders[0]._chunks is None
        assert pipeline.active == 0
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await pipeline.close()
        await spool.close()


async def test_encoding_error_after_headers_terminates_incomplete_json(tmp_path, monkeypatch):
    import aiohttp

    app, ref, _ = fixture_app(tmp_path, artifact_run_result_response_chunk_bytes=31)
    finished = asyncio.Event()
    calls = 0
    original = ResultEncoder.next_chunk

    def fail_after_chunk(encoder):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError('injected encoding failure')
        return original(encoder)

    async def handler(request):
        try:
            return await executions._read_result_response(request, stored_object=None, ref=ref)
        finally:
            finished.set()

    app.router.add_get('/fixture-result', handler)
    monkeypatch.setattr(ResultEncoder, 'next_chunk', fail_after_chunk)
    async with TestClient(TestServer(app)) as client:
        response = await client.get('/fixture-result')
        assert response.status == 200
        with pytest.raises(aiohttp.ClientPayloadError):
            await response.read()
        await asyncio.wait_for(finished.wait(), timeout=2)
        assert app[RESULT_PIPELINE_KEY].active == 0
        assert not list(tmp_path.glob('*.result'))


async def test_result_deadline_includes_continuously_progressing_send(tmp_path, monkeypatch):
    from datetime import timedelta

    app, ref, _ = fixture_app(
        tmp_path,
        artifact_run_result_response_chunk_bytes=8,
        artifact_download_write_idle_timeout=0.1,
        artifact_download_deadline=timedelta(seconds=0.5),
    )
    spool = app[ARTIFACT_SPOOL_KEY]
    pipeline = app[RESULT_PIPELINE_KEY]
    await spool.start()
    closed = asyncio.Event()
    writes = []

    class Response:
        def __init__(self, *, headers):
            self.headers = headers

        async def prepare(self, request):
            pass

        async def write(self, chunk):
            writes.append(chunk)
            await asyncio.sleep(0.02)

        async def write_eof(self):
            pytest.fail('incomplete response must not finish successfully')

        def force_close(self):
            closed.set()

    monkeypatch.setattr(executions.web, 'StreamResponse', Response)
    request = SimpleNamespace(app=app, headers={}, method='GET', transport=SimpleNamespace(close=closed.set))
    try:
        with pytest.raises(web.HTTPGatewayTimeout, match='Gateway Timeout'):
            await executions._read_result_response(request, stored_object=None, ref=ref)
        assert writes
        assert closed.is_set()
        assert pipeline.active == 0
        assert spool.reserved_bytes == 0
    finally:
        await pipeline.close()
        await spool.close()
