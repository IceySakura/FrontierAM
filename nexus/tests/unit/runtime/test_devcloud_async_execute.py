"""Delivery uncertainty must never replay a desktop action."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import threading
import time
import uuid

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from nexus.runtime.providers.devcloud import async_execute
from nexus.runtime.providers.devcloud.client import DevCloudRuntime


def response(status, body):
    return MagicMock(status_code=status, text=json.dumps(body), json=lambda: body)


@pytest.fixture
def job(monkeypatch):
    value = uuid.uuid4()
    monkeypatch.setattr(async_execute.uuid, 'uuid4', lambda: value)
    monkeypatch.setattr(async_execute, '_POLL_INTERVAL', 0.001)
    return value.hex


def completed(job, *, code=0, http=200):
    text = json.dumps({'returncode': code, 'output': 'once', 'error': ''})
    return response(200, {'job_id': job, 'state': 'completed', 'http_status': http, 'text': text})


@pytest.mark.asyncio
@pytest.mark.parametrize('lost_response', [TimeoutError(), aiohttp.ServerDisconnectedError(), response(503, {})])
async def test_lost_submission_and_poll_errors_only_query_same_job(monkeypatch, job, lost_response):
    post = AsyncMock(
        side_effect=[
            lost_response,
            response(404, {}),
            TimeoutError(),
            response(503, {}),
            response(200, {'job_id': job, 'state': 'running'}),
            completed(job),
        ]
    )
    monkeypatch.setattr(async_execute.http_client, 'post', post)
    body, _, status = await async_execute.execute(
        'http://vm:5000', {'command': ['echo', 'once']}, proxy_url='http://proxy', timeout=30
    )
    assert status == 200
    assert body['output'] == 'once'
    calls = post.await_args_list
    assert [c.args[0] for c in calls].count('http://vm:5000/execute') == 1
    assert all(c.args[0] == 'http://vm:5000/file' for c in calls[1:])
    assert {c.kwargs['data']['file_path'] for c in calls[1:]} == {f'/tmp/nexus-devcloud-execute-{job}/state.json'}
    assert all(c.kwargs['timeout'].total <= 10 for c in calls)
    assert all(c.kwargs['proxy'] == 'http://proxy' for c in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(('code', 'status'), [(0, 'Finished'), (7, 'Error'), (124, 'TimeLimitExceeded')])
async def test_async_mode_preserves_command_exit_status(monkeypatch, job, code, status):
    post = AsyncMock(side_effect=[response(200, {'returncode': 0}), completed(job, code=code)])
    monkeypatch.setattr(async_execute.http_client, 'post', post)
    runtime = DevCloudRuntime(base_url='http://vm:5000', execute_mode='async')
    result = await runtime.run_command(command='echo once', timeout=20)
    assert result.status.value == status
    assert result.return_code == code
    assert result.stdout == 'once'


@pytest.mark.asyncio
async def test_result_deadline_never_resubmits(monkeypatch, job):
    post = AsyncMock(
        side_effect=lambda url, **kw: (
            response(200, {'returncode': 0}) if url.endswith('/execute') else response(404, {})
        )
    )
    monkeypatch.setattr(async_execute.http_client, 'post', post)
    with pytest.raises(TimeoutError, match=f'{job}.*state unknown, not resubmitted'):
        await async_execute.execute('http://vm:5000', {}, proxy_url=None, timeout=0.03)
    assert sum(c.args[0].endswith('/execute') for c in post.await_args_list) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('recovers', [True, False])
async def test_truncated_state_only_retries_reads_within_deadline(monkeypatch, job, recovers):
    broken = MagicMock(status_code=200)
    broken.json.side_effect = json.JSONDecodeError('truncated', '{', 1)
    polls = 0

    async def post(url, **kwargs):
        nonlocal polls
        if url.endswith('/execute'):
            return response(200, {'returncode': 0})
        polls += 1
        return completed(job) if recovers and polls == 2 else broken

    mocked = AsyncMock(side_effect=post)
    monkeypatch.setattr(async_execute.http_client, 'post', mocked)
    if recovers:
        body, _, status = await async_execute.execute('http://vm:5000', {}, proxy_url=None, timeout=1)
        assert status == 200
        assert body['output'] == 'once'
    else:
        with pytest.raises(TimeoutError, match='state unknown, not resubmitted'):
            await async_execute.execute('http://vm:5000', {}, proxy_url=None, timeout=0.02)
    assert sum(c.args[0].endswith('/execute') for c in mocked.await_args_list) == 1
    assert polls >= 2


@pytest.mark.asyncio
async def test_cancellation_propagates_without_retry(monkeypatch, job):
    polled = asyncio.Event()

    async def post(url, **kwargs):
        if url.endswith('/execute'):
            return response(200, {'returncode': 0})
        polled.set()
        return await asyncio.Future()

    mocked = AsyncMock(side_effect=post)
    monkeypatch.setattr(async_execute.http_client, 'post', mocked)
    task = asyncio.create_task(async_execute.execute('http://vm:5000', {}, proxy_url=None, timeout=20))
    await polled.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert mocked.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'state',
    [
        {'job_id': 'wrong', 'state': 'completed'},
        {'state': 'failed', 'error': 'local execute disconnected'},
        {'state': 'unexpected'},
    ],
)
async def test_invalid_or_failed_job_is_not_success(monkeypatch, job, state):
    post = AsyncMock(side_effect=[response(200, {'returncode': 0}), response(200, {'job_id': job, **state})])
    monkeypatch.setattr(async_execute.http_client, 'post', post)
    with pytest.raises(RuntimeError, match='DevCloud async execute'):
        await async_execute.execute('http://vm:5000', {}, proxy_url=None, timeout=10)
    assert post.await_count == 2


@pytest.mark.asyncio
async def test_bootstrap_rejected_is_not_replayed_or_polled(monkeypatch):
    post = AsyncMock(return_value=response(200, {'returncode': 127, 'error': 'python3 missing'}))
    monkeypatch.setattr(async_execute.http_client, 'post', post)
    with pytest.raises(RuntimeError, match='bootstrap failed'):
        await async_execute.execute('http://vm:5000', {}, proxy_url=None, timeout=10)
    assert post.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(('provider_budget', 'expected'), [(5400, 150), (60, 60)])
async def test_command_result_wait_has_bounded_budget(monkeypatch, provider_budget, expected):
    execute = AsyncMock(return_value=({'returncode': 0}, '{"returncode":0}', 200))
    monkeypatch.setattr(async_execute, 'execute', execute)
    runtime = DevCloudRuntime(base_url='http://vm:5000', request_timeout=provider_budget, execute_mode='async')
    await runtime.run_command(command='sleep 20', timeout=120)
    assert execute.await_args.kwargs['timeout'] == expected


def test_detached_worker_executes_once_and_preserves_http_error():
    """Exercise the real bootstrap + worker against a local OSWorld stand-in."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            time.sleep(0.2)
            self.send_response(502)
            self.end_headers()
            self.wfile.write(b'{"error":"upstream failed"}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    job_id = uuid.uuid4().hex
    directory = Path(f'/tmp/nexus-devcloud-execute-{job_id}')
    payload = {'command': ['echo', 'exactly once'], 'shell': False}
    launch = async_execute.build_launch_payload(
        job_id,
        {
            'job_id': job_id,
            'url': f'http://127.0.0.1:{server.server_port}/execute',
            'payload': payload,
            'timeout': 5,
        },
    )
    try:
        for _ in range(2):
            subprocess.run(launch['command'], check=True, timeout=5)
        deadline = time.monotonic() + 5
        state = {}
        while time.monotonic() < deadline:
            state = json.loads((directory / 'state.json').read_text())
            if state['state'] == 'completed':
                break
            time.sleep(0.01)
        assert state['state'] == 'completed'
        assert state['http_status'] == 502
        assert json.loads(state['text']) == {'error': 'upstream failed'}
        assert requests == [payload]
        assert directory.stat().st_mode & 0o777 == 0o700
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        shutil.rmtree(directory, ignore_errors=True)
