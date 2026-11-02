import asyncio
import json

from unittest.mock import AsyncMock

import pytest

from nexus.extensions.tasks.os_world_v2.execution.runtime import OSWorldTaskRuntime
from nexus.extensions.tasks.os_world_v2.adapter.control_reads import PollReadRuntime, read_control
from nexus.runtime.providers.ags.handle import AGSInstanceHandle
from nexus.utils.async_http import HTTPResponse


def response(status=200, data=None):
    text = (
        '<html>502 Bad Gateway</html>'
        if status >= 400
        else json.dumps(data or {'returncode': 0, 'output': 'RUNNING\n'})
    )
    return HTTPResponse(status, {}, text, text.encode(), 'https://fixture.invalid/execute', 'utf-8')


@pytest.fixture
def runtime():
    return OSWorldTaskRuntime(AGSInstanceHandle('fixture', 'secret', 'example.invalid'))


@pytest.mark.asyncio
async def test_single_502_reuses_read_policy_and_preserves_result(runtime, monkeypatch, tmp_path):
    runtime._http_request = AsyncMock(side_effect=[response(502), response()])
    monkeypatch.setattr('nexus.extensions.tasks.os_world_v2.execution.runtime.asyncio.sleep', AsyncMock())
    result = await read_control(runtime, 'echo RUNNING', label='cc_pid_exit', timeout=30, diagnostic_dir=tmp_path)
    assert result.return_code == 0
    assert result.stdout == 'RUNNING\n'
    assert runtime._http_request.await_count == 2
    calls = runtime._http_request.call_args_list
    assert calls[0].kwargs['json'] == calls[1].kwargs['json']
    assert all(call.kwargs['headers']['Connection'] == 'close' for call in calls)
    assert 'echo RUNNING' not in (tmp_path / 'control-reads.jsonl').read_text()


@pytest.mark.asyncio
async def test_persistent_502_has_five_attempts_not_nested_fifteen(runtime, monkeypatch):
    runtime._http_request = AsyncMock(return_value=response(502))
    monkeypatch.setattr('nexus.extensions.tasks.os_world_v2.execution.runtime.asyncio.sleep', AsyncMock())
    with pytest.raises(RuntimeError, match='HTTP 502'):
        await read_control(runtime, 'echo RUNNING', label='cc_pid_exit', timeout=30)
    assert runtime._http_request.await_count == 5


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [400, 401, 403])
async def test_permanent_http_errors_are_not_retried(runtime, status):
    runtime._http_request = AsyncMock(return_value=response(status))
    with pytest.raises(RuntimeError, match=f'HTTP {status}'):
        await read_control(runtime, 'echo RUNNING', label='cc_pid_exit', timeout=30)
    assert runtime._http_request.await_count == 1


@pytest.mark.asyncio
async def test_command_error_and_invalid_json_do_not_retry(runtime):
    runtime._http_request = AsyncMock(return_value=response(data={'returncode': 2, 'error': 'bad path'}))
    result = await read_control(runtime, 'test -f /not-found', label='cc_pid_exit', timeout=30)
    assert result.return_code == 2
    assert runtime._http_request.await_count == 1
    runtime._http_request = AsyncMock(return_value=HTTPResponse(200, {}, 'invalid', b'invalid', '', 'utf-8'))
    with pytest.raises(RuntimeError, match='invalid JSON'):
        await read_control(runtime, 'echo RUNNING', label='cc_pid_exit', timeout=30)
    assert runtime._http_request.await_count == 1


@pytest.mark.asyncio
async def test_wrapper_does_not_retry_writes_or_modified_read(runtime):
    runtime.run_command = AsyncMock(side_effect=RuntimeError('HTTP 502'))
    wrapper = PollReadRuntime(runtime, 'echo RUNNING', None)
    for command in ['echo write >> /tmp/state', 'echo RUNNING; touch /tmp/state']:
        with pytest.raises(RuntimeError, match='HTTP 502'):
            await wrapper.run_command(command=command, timeout=10)
    assert runtime.run_command.await_count == 2


@pytest.mark.asyncio
async def test_cancellation_propagates_without_retry(runtime):
    runtime._http_request = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await read_control(runtime, 'echo RUNNING', label='cc_pid_exit', timeout=30)
    assert runtime._http_request.await_count == 1


@pytest.mark.asyncio
async def test_total_deadline_includes_backoff(runtime):
    runtime._http_request = AsyncMock(return_value=response(502))
    with pytest.raises(TimeoutError):
        await read_control(runtime, 'echo RUNNING', label='cc_pid_exit', timeout=0.05)
    assert runtime._http_request.await_count == 1
