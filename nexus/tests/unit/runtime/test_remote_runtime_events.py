import asyncio
import logging

from unittest.mock import AsyncMock

import aiohttp
import pytest

from nexus.runtime.client.remote import RemoteRuntime, http_client
from nexus.runtime.models import InvokeRequest, pack_result


class _TraceBarrierSio:
    def __init__(self, response: dict | None = None) -> None:
        self.response = response or {'ok': True, 'span_batch_id': 0}
        self.calls: list[tuple[str, dict, float]] = []

    async def call(self, event: str, data: dict, *, timeout: float) -> dict:
        self.calls.append((event, data, timeout))
        return self.response


class _HealthResponse:
    status_code = 200

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _DisconnectSio:
    def __init__(self, *, shutdown_error: Exception | None = None) -> None:
        self.shutdown = AsyncMock(side_effect=shutdown_error)
        self.eio = AsyncMock()


class _ChildCancelledShutdownSio:
    def __init__(self) -> None:
        self.eio = AsyncMock()

    async def shutdown(self) -> None:
        shutdown_future = asyncio.get_running_loop().create_future()
        shutdown_future.cancel()
        await shutdown_future


class _BlockingShutdownSio:
    def __init__(self) -> None:
        self.shutdown_started = asyncio.Event()
        self.release_shutdown = asyncio.Event()
        self.shutdown_calls = 0
        self.eio = AsyncMock()

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.shutdown_started.set()
        await self.release_shutdown.wait()


class _ChildCancelledConnectSio:
    def __init__(self) -> None:
        self.connected = False
        self.disconnect_calls = 0

    async def connect(self, *_args: object, **_kwargs: object) -> None:
        connect_future = asyncio.get_running_loop().create_future()
        connect_future.cancel()
        await connect_future

    async def disconnect(self) -> None:
        self.disconnect_calls += 1


class _BlockingConnectSio:
    def __init__(self) -> None:
        self.connected = False
        self.started = asyncio.Event()

    async def connect(self, *_args: object, **_kwargs: object) -> None:
        self.started.set()
        await asyncio.Event().wait()


class _ChildCancelledTraceBarrierSio:
    async def call(self, *_args: object, **_kwargs: object) -> dict:
        call_future = asyncio.get_running_loop().create_future()
        call_future.cancel()
        await call_future


def _trace_flush_runtime() -> RemoteRuntime:
    runtime = RemoteRuntime(base_url='http://sandbox:8000')
    runtime._trace_flush_supported = True
    return runtime


@pytest.mark.asyncio
async def test_ws_disconnect_closes_engineio_session_when_socketio_is_already_disconnected() -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000')
    sio = runtime._ensure_sio_client()
    session = aiohttp.ClientSession()
    sio.eio.http = session
    sio.eio.external_http = False

    assert sio.connected is False

    await runtime.ws_disconnect()

    assert session.closed is True


@pytest.mark.asyncio
async def test_ws_disconnect_is_idempotent() -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000')
    sio = runtime._ensure_sio_client()
    session = aiohttp.ClientSession()
    sio.eio.http = session
    sio.eio.external_http = False

    await runtime.ws_disconnect()
    await runtime.ws_disconnect()

    assert session.closed is True


@pytest.mark.asyncio
async def test_ws_disconnect_cleans_engineio_after_socketio_shutdown_failure() -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000')
    sio = _DisconnectSio(shutdown_error=RuntimeError('shutdown failed'))
    runtime._sio = sio  # type: ignore[assignment]

    for _ in range(2):
        with pytest.raises(RuntimeError, match='shutdown failed'):
            await runtime.ws_disconnect()

    assert sio.eio.disconnect.await_count == 2
    sio.eio.disconnect.assert_awaited_with(abort=True)


@pytest.mark.asyncio
async def test_ws_disconnect_contains_child_shutdown_cancellation() -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000')
    sio = _ChildCancelledShutdownSio()
    runtime._sio = sio  # type: ignore[assignment]

    await runtime.ws_disconnect()

    sio.eio.disconnect.assert_awaited_once_with(abort=True)


@pytest.mark.asyncio
async def test_ws_disconnect_serializes_lifecycle_calls() -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000')
    sio = _BlockingShutdownSio()
    runtime._sio = sio  # type: ignore[assignment]

    first_disconnect = asyncio.create_task(runtime.ws_disconnect())
    await sio.shutdown_started.wait()
    second_disconnect = asyncio.create_task(runtime.ws_disconnect())
    await asyncio.sleep(0)

    assert sio.shutdown_calls == 1
    assert second_disconnect.done() is False

    sio.release_shutdown.set()
    await asyncio.gather(first_disconnect, second_disconnect)

    assert sio.shutdown_calls == 2


@pytest.mark.asyncio
async def test_ws_disconnect_preserves_external_cancellation() -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000')
    sio = _BlockingShutdownSio()
    runtime._sio = sio  # type: ignore[assignment]

    disconnect_task = asyncio.create_task(runtime.ws_disconnect())
    await sio.shutdown_started.wait()
    disconnect_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await disconnect_task


def test_socketio_client_reconnects_in_background_by_default() -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000')

    assert runtime._ensure_sio_client().reconnection is True


@pytest.mark.asyncio
async def test_ws_connect_converts_child_cancellation(caplog: pytest.LogCaptureFixture) -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000', initial_connect_attempts=1)
    runtime._sio = _ChildCancelledConnectSio()  # type: ignore[assignment]

    with caplog.at_level(logging.ERROR, logger='nexus.runtime.client.remote'):
        with pytest.raises(RuntimeError, match='failed to connect'):
            await runtime.ws_connect()

    assert '[nexus-cancellation]' in caplog.text
    assert 'phase=sio_connect' in caplog.text
    assert 'task_cancelling=0' in caplog.text


@pytest.mark.asyncio
async def test_ws_connect_preserves_external_cancellation() -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000', initial_connect_attempts=1)
    sio = _BlockingConnectSio()
    runtime._sio = sio  # type: ignore[assignment]
    connect_task = asyncio.create_task(runtime.ws_connect())
    await sio.started.wait()

    connect_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await connect_task


@pytest.mark.asyncio
async def test_runtime_http_pool_expires_idle_connections_before_uvicorn() -> None:
    session = http_client._get_session()
    try:
        assert session.connector is not None
        assert session.connector._keepalive_timeout == 4.0
    finally:
        await http_client.close()


@pytest.mark.asyncio
async def test_auto_connect_child_cancellation_falls_back_to_http(caplog: pytest.LogCaptureFixture) -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000')
    sio = _ChildCancelledConnectSio()
    runtime._sio = sio  # type: ignore[assignment]

    with caplog.at_level(logging.ERROR, logger='nexus.runtime.client.remote'):
        await runtime._try_auto_connect_sio()

    assert runtime._sio_auto_connect_disabled is True
    assert runtime._sio is None
    assert sio.disconnect_calls == 1
    assert '[nexus-cancellation]' in caplog.text
    assert 'phase=sio_auto_connect' in caplog.text
    assert 'task_cancelling=0' in caplog.text


@pytest.mark.asyncio
async def test_auto_connect_child_cancellation_propagates_without_task_cancelling_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000')
    sio = _ChildCancelledConnectSio()
    runtime._sio = sio  # type: ignore[assignment]
    current_task = asyncio.current_task
    monkeypatch.setattr(asyncio, 'current_task', lambda: object())

    try:
        with pytest.raises(asyncio.CancelledError):
            await runtime._try_auto_connect_sio()
    finally:
        monkeypatch.setattr(asyncio, 'current_task', current_task)

    assert runtime._sio_auto_connect_disabled is False
    assert runtime._sio is sio


@pytest.mark.asyncio
async def test_auto_connect_external_task_cancellation_propagates() -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000')
    sio = _BlockingConnectSio()
    runtime._sio = sio  # type: ignore[assignment]
    connect_task = asyncio.create_task(runtime._try_auto_connect_sio())
    await sio.started.wait()

    connect_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await connect_task
    assert runtime._sio_auto_connect_disabled is False
    assert runtime._sio is sio


@pytest.mark.asyncio
async def test_auto_connect_preserves_asyncio_timeout() -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000')
    sio = _BlockingConnectSio()
    runtime._sio = sio  # type: ignore[assignment]

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await runtime._try_auto_connect_sio()

    assert runtime._sio_auto_connect_disabled is False
    assert runtime._sio is sio


@pytest.mark.asyncio
async def test_span_subscription_keeps_builtin_replay(monkeypatch) -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000')
    replayed: list[dict] = []
    observed: list[dict] = []

    monkeypatch.setattr(runtime, '_handle_span_event', replayed.append)
    runtime.on('span', observed.append)

    handler = runtime._ensure_sio_client().handlers['/']['span']
    payload = {'batch_id': 1, 'items': [{'type': 'UPDATE', 'span_id': 'span-1'}]}
    await handler(payload)

    assert replayed == payload['items']
    assert observed == [payload]


@pytest.mark.asyncio
async def test_forced_client_reset_keeps_event_subscriptions() -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000')
    observed: list[dict] = []
    runtime.on('llm_request', observed.append)
    old_sio = runtime._ensure_sio_client()

    await runtime._force_reset_sio_client()
    new_sio = runtime._ensure_sio_client()
    await new_sio.handlers['/']['llm_request']({'request_id': 'R1'})

    assert new_sio is not old_sio
    assert observed == [{'request_id': 'R1'}]


@pytest.mark.asyncio
async def test_log_subscription_keeps_builtin_log_handler(monkeypatch) -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000')
    logged: list[dict] = []
    observed: list[dict] = []

    monkeypatch.setattr(runtime, '_handle_runtime_log', logged.append)
    runtime.on('log', observed.append)

    handler = runtime._ensure_sio_client().handlers['/']['log']
    payload = {'items': [{'level': 'INFO', 'message': 'hello'}]}
    await handler(payload)

    assert logged == payload['items']
    assert observed == [payload]


@pytest.mark.asyncio
async def test_drain_trace_waits_for_socketio_barrier() -> None:
    runtime = _trace_flush_runtime()
    sio = _TraceBarrierSio()
    runtime._sio = sio  # type: ignore[assignment]
    runtime._ws_connected = True

    await runtime.drain_trace(timeout=3.0)

    assert sio.calls == [('trace:flush', {}, 3.0)]


@pytest.mark.asyncio
async def test_trace_flush_child_cancellation_becomes_operation_error(caplog: pytest.LogCaptureFixture) -> None:
    runtime = _trace_flush_runtime()
    runtime._sio = _ChildCancelledTraceBarrierSio()  # type: ignore[assignment]
    runtime._ws_connected = True

    with caplog.at_level(logging.ERROR, logger='nexus.runtime.client.remote'):
        with pytest.raises(RuntimeError, match='phase=trace_flush'):
            await runtime.drain_trace(timeout=3.0)

    assert '[nexus-cancellation]' in caplog.text
    assert 'phase=trace_flush' in caplog.text
    assert 'base_url=http://sandbox:8000' in caplog.text
    assert 'request_id=None' in caplog.text
    assert 'task_cancelling=0' in caplog.text


@pytest.mark.asyncio
async def test_http_poll_child_cancellation_becomes_operation_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000', poll_interval=0.0)

    async def cancelled_get(*_args: object, **_kwargs: object) -> None:
        get_future = asyncio.get_running_loop().create_future()
        get_future.cancel()
        await get_future

    monkeypatch.setattr(http_client, 'get', cancelled_get)

    with caplog.at_level(logging.ERROR, logger='nexus.runtime.client.remote'):
        with pytest.raises(RuntimeError, match='phase=invoke_poll_get.*request_id=request-456'):
            await runtime._wait_for_job(
                'request-456',
                request_timeout=1.0,
                timeout=runtime._timeout,
            )

    assert '[nexus-cancellation]' in caplog.text
    assert 'phase=invoke_poll_get' in caplog.text
    assert 'base_url=http://sandbox:8000' in caplog.text
    assert 'request_id=request-456' in caplog.text
    assert 'task_cancelling=0' in caplog.text


@pytest.mark.asyncio
async def test_http_poll_external_task_cancellation_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000', poll_interval=0.0)
    request_started = asyncio.Event()

    async def blocking_get(*_args: object, **_kwargs: object) -> None:
        request_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(http_client, 'get', blocking_get)
    poll_task = asyncio.create_task(
        runtime._wait_for_job(
            'request-external-cancel',
            request_timeout=10.0,
            timeout=runtime._timeout,
        )
    )
    await request_started.wait()

    poll_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await poll_task


@pytest.mark.asyncio
async def test_drain_trace_does_not_wait_for_span_compatibility_listener(monkeypatch) -> None:
    runtime = _trace_flush_runtime()
    handler = runtime._ensure_sio_client().handlers['/']['span']
    runtime._sio = _TraceBarrierSio({'ok': True, 'span_batch_id': 7})  # type: ignore[assignment]
    runtime._ws_connected = True
    replay_started = asyncio.Event()
    release_replay = asyncio.Event()

    async def slow_dispatch(_event, *_args, **_kwargs) -> None:  # noqa: ANN001
        replay_started.set()
        await release_replay.wait()

    monkeypatch.setattr(runtime, '_dispatch_event_listeners', slow_dispatch)
    replay_task = asyncio.create_task(handler({'batch_id': 7, 'items': []}))
    await replay_started.wait()
    drain_task = asyncio.create_task(runtime.drain_trace(timeout=3.0))
    try:
        await asyncio.wait_for(asyncio.shield(drain_task), timeout=0.1)
        assert replay_task.done() is False
    finally:
        release_replay.set()
        await replay_task
        if not drain_task.done():
            await drain_task


@pytest.mark.asyncio
async def test_drain_trace_rejects_a_gap_after_the_first_replayed_batch() -> None:
    runtime = _trace_flush_runtime()
    handler = runtime._ensure_sio_client().handlers['/']['span']
    await handler({'batch_id': 5, 'items': []})
    await handler({'batch_id': 7, 'items': [], 'barrier': True})
    runtime._sio = _TraceBarrierSio({'ok': True, 'span_batch_id': 7})  # type: ignore[assignment]
    runtime._ws_connected = True

    with pytest.raises(RuntimeError, match='expected 6, received 7'):
        await runtime.drain_trace()


@pytest.mark.asyncio
async def test_duplicate_span_batch_is_not_replayed_twice(monkeypatch) -> None:
    runtime = _trace_flush_runtime()
    replayed: list[dict] = []
    monkeypatch.setattr(runtime, '_handle_span_event', replayed.append)
    handler = runtime._ensure_sio_client().handlers['/']['span']
    item = {'type': 'START', 'span_id': 'span-1'}

    await handler({'batch_id': 3, 'items': [item]})
    await handler({'batch_id': 3, 'items': [item]})
    runtime._sio = _TraceBarrierSio({'ok': True, 'span_batch_id': 3})  # type: ignore[assignment]
    runtime._ws_connected = True

    assert replayed == [item]
    with pytest.raises(RuntimeError, match='duplicate or out-of-order span batch 3'):
        await runtime.drain_trace()


@pytest.mark.asyncio
async def test_negative_span_batch_id_is_rejected_without_replay(monkeypatch) -> None:
    runtime = _trace_flush_runtime()
    replayed: list[dict] = []
    monkeypatch.setattr(runtime, '_handle_span_event', replayed.append)
    handler = runtime._ensure_sio_client().handlers['/']['span']

    await handler({'batch_id': -1, 'items': [{'type': 'START', 'span_id': 'span-1'}]})
    runtime._sio = _TraceBarrierSio({'ok': True, 'span_batch_id': 0})  # type: ignore[assignment]
    runtime._ws_connected = True

    assert replayed == []
    with pytest.raises(RuntimeError, match='invalid batch_id'):
        await runtime.drain_trace()


@pytest.mark.asyncio
async def test_drain_trace_accepts_a_reconnected_connection_without_a_span_gap() -> None:
    runtime = _trace_flush_runtime()
    sio = runtime._ensure_sio_client()
    runtime._ws_connected = True

    await sio.handlers['/']['disconnect']()
    reconnected_sio = _TraceBarrierSio({'ok': True, 'span_batch_id': 0})
    runtime._sio = reconnected_sio  # type: ignore[assignment]
    runtime._ws_connected = True

    await runtime.drain_trace()

    assert reconnected_sio.calls == [('trace:flush', {}, runtime.config.socketio_request_timeout)]


@pytest.mark.asyncio
async def test_drain_trace_surfaces_local_span_replay_failure(monkeypatch) -> None:
    runtime = _trace_flush_runtime()
    handler = runtime._ensure_sio_client().handlers['/']['span']

    def fail_replay(_event) -> None:  # noqa: ANN001
        raise ValueError('invalid remote span')

    monkeypatch.setattr(runtime, '_handle_span_event', fail_replay)
    await handler({'batch_id': 3, 'items': [{'type': 'UPDATE', 'span_id': 'missing'}]})
    runtime._sio = _TraceBarrierSio({'ok': True, 'span_batch_id': 3})  # type: ignore[assignment]
    runtime._ws_connected = True

    with pytest.raises(RuntimeError, match='invalid remote span'):
        await runtime.drain_trace()


@pytest.mark.asyncio
async def test_drain_trace_surfaces_malformed_span_batch() -> None:
    runtime = _trace_flush_runtime()
    handler = runtime._ensure_sio_client().handlers['/']['span']
    await handler({'batch_id': 4, 'items': None})
    runtime._sio = _TraceBarrierSio({'ok': True, 'span_batch_id': 4})  # type: ignore[assignment]
    runtime._ws_connected = True

    with pytest.raises(RuntimeError, match='items must be a list'):
        await runtime.drain_trace()


@pytest.mark.asyncio
async def test_drain_trace_rejects_failed_barrier() -> None:
    runtime = _trace_flush_runtime()
    runtime._sio = _TraceBarrierSio({'ok': False, 'error': 'flush failed'})  # type: ignore[assignment]
    runtime._ws_connected = True

    with pytest.raises(RuntimeError, match='flush failed'):
        await runtime.drain_trace()


@pytest.mark.asyncio
async def test_drain_trace_connects_socketio_before_first_remote_operation(monkeypatch) -> None:
    runtime = _trace_flush_runtime()
    sio = _TraceBarrierSio()
    connect_calls = 0

    async def connect() -> _TraceBarrierSio:
        nonlocal connect_calls
        connect_calls += 1
        runtime._sio = sio  # type: ignore[assignment]
        runtime._ws_connected = True
        return sio

    monkeypatch.setattr(runtime, '_ensure_sio_connected', connect)

    await runtime.drain_trace(timeout=3.0)

    assert connect_calls == 1
    assert sio.calls == [('trace:flush', {}, 3.0)]


@pytest.mark.asyncio
async def test_drain_trace_rejects_socketio_connected_only_after_remote_operation(monkeypatch) -> None:
    runtime = _trace_flush_runtime()
    sio = _TraceBarrierSio()

    async def skip_auto_connect() -> None:
        return None

    class Response:
        status_code = 200

        def json(self) -> dict:
            return pack_result('ok', request_id='request-1').model_dump()

    async def post(*_args, **_kwargs) -> Response:
        return Response()

    async def connect() -> _TraceBarrierSio:
        runtime._sio = sio  # type: ignore[assignment]
        runtime._ws_connected = True
        return sio

    monkeypatch.setattr(runtime, '_try_auto_connect_sio', skip_auto_connect)
    monkeypatch.setattr(runtime, '_ensure_sio_connected', connect)
    monkeypatch.setattr(http_client, 'post', post)

    assert await runtime._invoke(InvokeRequest.build('unit::call', request_id='request-1')) == 'ok'
    with pytest.raises(RuntimeError, match='remote operation started before Socket.IO'):
        await runtime.drain_trace(timeout=3.0)


@pytest.mark.asyncio
async def test_drain_trace_skips_legacy_runtime_without_flush_capability(monkeypatch) -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000')
    health_calls: list[str] = []

    async def get_health(url: str, **_kwargs) -> _HealthResponse:
        health_calls.append(url)
        return _HealthResponse({'status': 'ok', 'uptime_seconds': 10.0})

    monkeypatch.setattr('nexus.runtime.client.remote.http_client.get', get_health)

    await runtime.drain_trace()

    assert health_calls == ['http://sandbox:8000/health']


@pytest.mark.asyncio
async def test_drain_trace_enforces_advertised_flush_capability(monkeypatch) -> None:
    runtime = RemoteRuntime(base_url='http://sandbox:8000')
    runtime._sio = _TraceBarrierSio({'ok': False, 'error': 'flush failed'})  # type: ignore[assignment]
    runtime._ws_connected = True
    health_calls: list[str] = []

    async def get_health(url: str, **_kwargs) -> _HealthResponse:
        health_calls.append(url)
        return _HealthResponse({'status': 'ok', 'capabilities': {'trace_flush': 1}})

    monkeypatch.setattr('nexus.runtime.client.remote.http_client.get', get_health)

    with pytest.raises(RuntimeError, match='flush failed'):
        await runtime.drain_trace()

    assert health_calls == ['http://sandbox:8000/health']


def test_runtime_span_replay_marks_controller_event_as_replay(monkeypatch) -> None:
    from nexus.trace import SpanController

    runtime = RemoteRuntime(base_url='http://sandbox:8000')
    controller = SpanController()
    calls: list[tuple[str, str | None]] = []
    original_add_event = controller.add_event

    def record_origin(event, **kwargs):  # noqa: ANN001, ANN003
        calls.append((event.type, kwargs.get('origin')))
        original_add_event(event, **kwargs)

    monkeypatch.setattr(controller, 'add_event', record_origin)
    monkeypatch.setattr(SpanController, 'get_instance_sync', lambda: controller)

    runtime._handle_span_event({'type': 'START', 'span_id': 'span-1', 'parent_span_id': '', 'name': 'runtime'})
    runtime._handle_span_event({'type': 'END', 'span_id': 'span-1'})

    assert calls == [('START', 'replay'), ('END', 'replay')]
