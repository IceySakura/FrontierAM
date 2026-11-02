from __future__ import annotations

import asyncio

from unittest.mock import AsyncMock

import pytest

from nexus.runtime.client.base import Runtime
from nexus.runtime.client.remote import RemoteRuntime
from nexus.runtime.executors import command as command_executor
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandRequest, RunCommandResponse
from nexus.runtime.models import InvokeRequest
from nexus.runtime.providers.ags.code_interpreter import AGSCodeInterpreterRuntime
from nexus.runtime.providers.ags.osworld import AGSOSWorldRuntime
from nexus.runtime.providers.devcloud.client import DevCloudRuntime
from tests.unit.runtime.test_wait_for_job_race import _FakeHTTPResponse, _ok_response


def test_run_command_model_preserves_default_and_explicit_none():
    assert RunCommandRequest(command='true').timeout == 180
    request = RunCommandRequest(command='true', timeout=None)
    assert request.timeout is None
    assert RunCommandRequest.model_validate_json(request.model_dump_json()).timeout is None


@pytest.mark.parametrize('timeout_control_id', [None, 'unbounded-command'])
async def test_command_executor_accepts_no_deadline(monkeypatch, isolated_shell_home, timeout_control_id):
    real_async_timeout = command_executor.async_timeout
    deadlines = []

    def capture_timeout(timeout):
        deadlines.append(timeout)
        return real_async_timeout(timeout)

    monkeypatch.setattr(command_executor, 'async_timeout', capture_timeout)
    response = await asyncio.wait_for(
        command_executor.run_command(
            RunCommandRequest(command='echo completed', timeout=None, timeout_control_id=timeout_control_id)
        ),
        timeout=5,
    )

    assert deadlines == [None]
    assert response.status == CommandRunStatus.Finished
    assert response.return_code == 0
    assert response.stdout.strip() == 'completed'
    assert timeout_control_id not in command_executor._COMMAND_TIMEOUTS


@pytest.mark.parametrize('timeout_kwargs', [{}, {'timeout': None}, {'timeout': 12}])
async def test_command_client_disables_invoke_deadline_only_for_explicit_none(timeout_kwargs):
    requests = []

    class RecordingRuntime(Runtime):
        async def _invoke(self, request: InvokeRequest):
            requests.append(request)
            return RunCommandResponse(status=CommandRunStatus.Finished, return_code=0)

    runtime = RecordingRuntime()
    runtime.default_venv_id = 'worker-venv'
    response = await runtime.run_command('echo completed', **timeout_kwargs)

    assert response.return_code == 0
    assert len(requests) == 1
    request = requests[0]
    expected_timeout = timeout_kwargs.get('timeout', 180)
    assert request.disable_invoke_wait_timeout is (expected_timeout is None)
    assert request.venv_id == 'worker-venv'
    args, kwargs = request.unpack_payload()
    assert args[0]['timeout'] == expected_timeout
    assert kwargs == {}


@pytest.mark.parametrize('sio_connected', [False, True])
async def test_unbounded_remote_command_outlives_default_rpc_wait(monkeypatch, sio_connected):
    runtime = RemoteRuntime(base_url='http://sandbox:8000', request_timeout=0.01, auto_connect_ws=False)
    runtime._ws_connected = sio_connected
    runtime.config.poll_interval = 0.01
    runtime.config.backstop_poll_interval = 0.01
    captured = {}
    polls = 0

    async def fake_post(url, **kwargs):
        captured['wire'] = kwargs['json']
        assert kwargs['timeout'].total == 0.01
        response = _FakeHTTPResponse({})
        response.status_code = 202
        return response

    async def fake_get(url, **kwargs):
        nonlocal polls
        polls += 1
        assert kwargs['timeout'].total == 0.01
        rid = captured['wire']['request_id']
        if polls < 5:
            return _FakeHTTPResponse({'request_id': rid, 'status': 'running'})
        response = _ok_response(rid, RunCommandResponse(status=CommandRunStatus.Finished, return_code=0))
        return _FakeHTTPResponse({'request_id': rid, 'status': 'done', 'response': response.model_dump()})

    monkeypatch.setattr(runtime, '_try_auto_connect_sio', AsyncMock())
    monkeypatch.setattr('nexus.runtime.client.remote.http_client.post', fake_post)
    monkeypatch.setattr('nexus.runtime.client.remote.http_client.get', fake_get)

    response = await asyncio.wait_for(runtime.run_command('echo completed', timeout=None), timeout=2)

    assert response.return_code == 0
    assert polls == 5
    assert 'disable_invoke_wait_timeout' not in captured['wire']
    request = InvokeRequest.model_validate(captured['wire'])
    args, _ = request.unpack_payload()
    assert args[0]['timeout'] is None


@pytest.mark.parametrize('sio_connected', [False, True])
async def test_unbounded_remote_command_wait_can_be_cancelled(monkeypatch, sio_connected):
    runtime = RemoteRuntime(base_url='http://sandbox:8000', auto_connect_ws=False)
    runtime._ws_connected = sio_connected
    runtime.config.poll_interval = 0.01
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def fake_post(url, **kwargs):
        response = _FakeHTTPResponse({})
        response.status_code = 202
        return response

    async def fake_get(url, **kwargs):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    monkeypatch.setattr(runtime, '_try_auto_connect_sio', AsyncMock())
    monkeypatch.setattr('nexus.runtime.client.remote.http_client.post', fake_post)
    monkeypatch.setattr('nexus.runtime.client.remote.http_client.get', fake_get)
    task = asyncio.create_task(runtime.run_command('long-running-command', timeout=None))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert cancelled.is_set()


@pytest.mark.parametrize('runtime_type', [AGSOSWorldRuntime, DevCloudRuntime])
def test_osworld_command_accepts_none_without_shell_timeout(runtime_type):
    payload = runtime_type._build_osworld_execute_payload(
        command='echo completed', cwd='/workspace', timeout=None, stdin=None, extra_env={'KEY': 'value'}
    )

    command = payload['command'][2]
    assert 'timeout --kill-after' not in command
    assert 'echo completed' in command
    assert 'cd /workspace' in command
    assert 'export KEY=value' in command


def test_code_interpreter_command_preserves_none_timeout():
    script = AGSCodeInterpreterRuntime._build_run_command_script('echo completed', None, None, None, None)
    assert 'timeout = None' in script
    compile(script, '<run-command>', 'exec')
