"""Keep failed invoke responses out of asynchronous job polling."""

import asyncio
import json

from typing import Any
from unittest.mock import AsyncMock

import pytest

from nexus.core.context import ExecutionContext
from nexus.extensions.openai_clients.hytuner import capture_hytuner_execution_timeouts
from nexus.runtime.client.remote import RemoteRuntime, http_client
from nexus.runtime.models import (
    InvokeRequest,
    RemoteInvokeError,
    invoke_response_from_exception,
    pack_result,
)
from nexus.utils.async_http import HTTPResponse
from nexus.utils.timeout import async_timeout


def _response(status_code: int, payload: dict[str, Any] | str, *, path: str = '/api/v1/invoke') -> HTTPResponse:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return HTTPResponse(
        status=status_code,
        headers={},
        text=text,
        content=text.encode(),
        url=f'http://sandbox:8000{path}',
        encoding='utf-8',
    )


@pytest.mark.parametrize('status_code', [201, 400, 401, 403, 404, 422, 500, 502, 503, 504])
@pytest.mark.parametrize('disable_invoke_wait_timeout', [False, True])
async def test_invoke_http_error_never_enters_job_wait(
    monkeypatch: pytest.MonkeyPatch, status_code: int, disable_invoke_wait_timeout: bool
) -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000', request_timeout=45000, auto_connect_ws=False)
    post = AsyncMock(return_value=_response(status_code, {'error': 'No connected worker for venv-test'}))
    wait = AsyncMock(side_effect=AssertionError('A failed POST must not enter asynchronous job waiting'))
    monkeypatch.setattr(http_client, 'post', post)
    monkeypatch.setattr(runtime, '_wait_for_job', wait)

    with pytest.raises(RemoteInvokeError) as caught:
        await runtime._invoke(
            InvokeRequest(
                request_id='failed-control',
                import_path='nexus.runtime.executors.command::resume_command_timeout',
                disable_invoke_wait_timeout=disable_invoke_wait_timeout,
            )
        )

    message = str(caught.value)
    assert f'HTTP {status_code}' in message, 'The failure must preserve its HTTP status'
    assert 'failed-control' in message, 'The failure must identify the invoke request'
    assert 'resume_command_timeout' in message, 'The failure must identify the remote operation'
    assert 'No connected worker for venv-test' in message, 'The server diagnosis must reach the caller'
    post.assert_awaited_once()
    wait.assert_not_awaited()


@pytest.mark.parametrize('status_code', [200, 202])
async def test_invoke_preserves_synchronous_and_asynchronous_success(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000', auto_connect_ws=False)
    payload = (
        pack_result('completed', request_id='valid-invoke').model_dump()
        if status_code == 200
        else {'request_id': 'valid-invoke', 'status': 'running'}
    )
    monkeypatch.setattr(http_client, 'post', AsyncMock(return_value=_response(status_code, payload)))
    wait = AsyncMock(return_value='completed')
    monkeypatch.setattr(runtime, '_wait_for_job', wait)

    result = await runtime._invoke(InvokeRequest(request_id='valid-invoke', import_path='test::run'))

    assert result == 'completed', 'Both supported HTTP statuses must preserve the invoke result'
    assert wait.await_count == (1 if status_code == 202 else 0), 'Only HTTP 202 may enter job waiting'


@pytest.mark.parametrize('sio_connected', [False, True])
@pytest.mark.parametrize('request_timeout', [None, 45000.0])
@pytest.mark.parametrize('status_code', [204, 401, 403, 404, 500, 503])
async def test_poll_http_error_exits_with_or_without_a_cumulative_deadline(
    monkeypatch: pytest.MonkeyPatch, sio_connected: bool, request_timeout: float | None, status_code: int
) -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000', auto_connect_ws=False, poll_interval=0.0)
    runtime._ws_connected = sio_connected
    runtime.config.backstop_poll_interval = 0.0
    get = AsyncMock(
        return_value=_response(
            status_code, {'error': 'Job missing-control not found'}, path='/api/v1/invoke/missing-control'
        )
    )
    monkeypatch.setattr(http_client, 'get', get)

    with pytest.raises(RemoteInvokeError) as caught:
        await asyncio.wait_for(
            runtime._wait_for_job('missing-control', request_timeout=request_timeout, timeout=runtime._timeout),
            timeout=0.1,
        )

    message = str(caught.value)
    assert f'HTTP {status_code}' in message, 'Polling failures must preserve their HTTP status'
    assert 'missing-control' in message, 'Polling failures must identify the missing job'
    assert 'Job missing-control not found' in message, 'The server diagnosis must not be lost'
    get.assert_awaited_once()
    assert not runtime._event_listeners.get('invoke:done:missing-control'), 'Failure must remove the SIO listener'


@pytest.mark.parametrize('payload', ['upstream unavailable', {'error': 'worker unavailable', 'traceback': 'x' * 10000}])
async def test_invoke_http_error_preserves_a_bounded_server_diagnosis(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any] | str
) -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000', auto_connect_ws=False)
    monkeypatch.setattr(http_client, 'post', AsyncMock(return_value=_response(500, payload)))
    monkeypatch.setattr(runtime, '_wait_for_job', AsyncMock(side_effect=AssertionError('Unexpected job polling')))

    with pytest.raises(RemoteInvokeError) as caught:
        await runtime._invoke(InvokeRequest(request_id='diagnostic-control', import_path='test::run'))

    expected = payload if isinstance(payload, str) else payload['error']
    assert expected in str(caught.value), 'JSON and plain-text errors must retain their server diagnosis'
    assert len(str(caught.value)) < 1500, 'A large remote traceback must not be copied into the error message'


async def test_successful_http_transport_preserves_the_remote_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000', auto_connect_ws=False)
    payload = invoke_response_from_exception(ValueError('remote failure'), request_id='remote-error').model_dump()
    monkeypatch.setattr(http_client, 'post', AsyncMock(return_value=_response(200, payload)))

    with pytest.raises(ValueError, match='remote failure'):
        await runtime._invoke(InvokeRequest(request_id='remote-error', import_path='test::run'))


@pytest.mark.parametrize('failed_action', ['pause', 'resume'])
async def test_timer_http_error_restores_budget_and_releases_control_locks(
    monkeypatch: pytest.MonkeyPatch, failed_action: str
) -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000', auto_connect_ws=False)
    runtime.default_venv_id = 'venv-test'
    monkeypatch.setattr(runtime, 'drain_trace', AsyncMock())
    failed = False

    async def post(_url: str, **kwargs: Any) -> HTTPResponse:
        nonlocal failed
        request = kwargs['json']
        if not failed and request['import_path'].endswith(f'::{failed_action}_command_timeout'):
            failed = True
            return _response(500, {'error': 'No connected worker for venv-test'})
        return _response(200, pack_result(True, request_id=request['request_id']).model_dump())

    monkeypatch.setattr(http_client, 'post', post)
    context = ExecutionContext(runtime_command_timeout_id='timer-test')
    context._runtimes['main'] = runtime
    async with context:
        async with async_timeout(1.0) as runner:
            context.timeout_handle = runner
            controller = capture_hytuner_execution_timeouts('request-test', track_shared_pause=True)
            assert controller is not None, 'A live Nexus context must provide an execution budget'
            if failed_action == 'resume':
                await controller.pause()
            with pytest.raises(RemoteInvokeError, match='HTTP 500'):
                await asyncio.wait_for(
                    controller.pause() if failed_action == 'pause' else controller.resume(), timeout=0.2
                )
            assert runner.when() is not None, 'A failed timer transition must restore the local runner deadline'

            await asyncio.wait_for(controller.pause(), timeout=0.2)
            assert runner.when() is None, 'A later pause must acquire the released control locks'
            await asyncio.wait_for(controller.resume(), timeout=0.2)
            assert runner.when() is not None, 'A later resume must restore the runner deadline'
