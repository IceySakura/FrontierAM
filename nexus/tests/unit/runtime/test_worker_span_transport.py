from __future__ import annotations

from typing import Any

import pytest

from nexus.runtime.server.worker.sdk import WorkerSDK, _SIOSpanProcessor
from nexus.trace import SpanController
from nexus.utils.truncate import DEFAULT_MARKER, DEFAULT_MAX_BYTES


class _FakeWorkerSio:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, dict[str, Any], str, float]] = []

    async def call(
        self,
        event: str,
        data: dict[str, Any],
        *,
        namespace: str,
        timeout: float,
    ) -> dict[str, bool]:
        self.calls.append((event, data, namespace, timeout))
        if self.error is not None:
            raise self.error
        return {'ok': True}


class _SpanEvent:
    def __init__(self, span_id: str) -> None:
        self.span_id = span_id

    def model_dump(self) -> dict[str, str]:
        return {'type': 'UPDATE', 'span_id': self.span_id}


def test_worker_span_processor_default_queue_capacity_is_2000() -> None:
    processor = _SIOSpanProcessor(_FakeWorkerSio(), 'worker-1')

    assert processor._emitter._queue.maxsize == 2000


@pytest.mark.asyncio
async def test_worker_span_processor_batches_events_and_flushes_with_ack() -> None:
    sio = _FakeWorkerSio()
    processor = _SIOSpanProcessor(sio, 'worker-1', flush_interval=60)

    processor.on_update(None, _SpanEvent('span-1'))
    processor.on_update(None, _SpanEvent('span-2'))
    await processor.flush()

    assert sio.calls == [
        (
            'span',
            {
                'batch_id': 1,
                'items': [
                    {'type': 'UPDATE', 'span_id': 'span-1', 'worker_id': 'worker-1'},
                    {'type': 'UPDATE', 'span_id': 'span-2', 'worker_id': 'worker-1'},
                ],
            },
            '/worker',
            60.0,
        )
    ]


@pytest.mark.asyncio
async def test_worker_span_processor_surfaces_transport_failure() -> None:
    processor = _SIOSpanProcessor(_FakeWorkerSio(OSError('worker socket failed')), 'worker-1')
    processor.on_end(None, _SpanEvent('span-1'))

    with pytest.raises(OSError, match='worker socket failed'):
        await processor.flush()


@pytest.mark.asyncio
async def test_worker_span_processor_truncates_attributes_before_first_socket_hop() -> None:
    class LargeSpanEvent:
        def model_dump(self) -> dict[str, Any]:
            return {
                'type': 'UPDATE',
                'span_id': 'span-1',
                'attributes': {'payload': 'x' * 300_000},
            }

    sio = _FakeWorkerSio()
    processor = _SIOSpanProcessor(sio, 'worker-1', flush_interval=60)

    processor.on_update(None, LargeSpanEvent())
    await processor.flush()

    payload = sio.calls[0][1]['items'][0]['attributes']['payload']
    assert DEFAULT_MARKER in payload
    assert len(payload.encode('utf-8')) <= DEFAULT_MAX_BYTES


@pytest.mark.asyncio
async def test_worker_sdk_stop_unregisters_span_processor(monkeypatch) -> None:
    SpanController._instance = None
    controller = SpanController.get_instance_sync()
    sdk = WorkerSDK('http://runtime.invalid', 'venv-1')
    controller.add_processor(sdk._span_processor)
    sdk._span_processor_registered = True
    closed = False

    async def close() -> None:
        nonlocal closed
        closed = True

    monkeypatch.setattr(sdk._span_processor, 'close', close)

    await sdk.stop()

    assert sdk._span_processor not in controller.processors
    assert closed is True


def test_worker_socketio_auth_advertises_trace_flush_capability() -> None:
    sdk = WorkerSDK('http://runtime.invalid', 'venv-1')

    assert sdk._connect_auth() == {
        'venv_id': 'venv-1',
        'worker_id': 'worker-venv-1',
        'capabilities': {'trace_flush': 1},
    }


@pytest.mark.asyncio
async def test_worker_execute_flushes_trace_before_returning(monkeypatch) -> None:
    sdk = WorkerSDK('http://runtime.invalid', 'venv-1')
    calls: list[str] = []

    async def dispatch(method: str, params: dict[str, Any]) -> dict[str, Any]:
        calls.append(f'dispatch:{method}')
        return params

    async def flush() -> None:
        calls.append('flush')

    monkeypatch.setattr(sdk, '_dispatch', dispatch)
    monkeypatch.setattr(sdk._span_processor, 'flush', flush)
    handler = sdk._sio.handlers['/worker']['execute']

    response = await handler({'method': 'tool.call', 'params': {'value': 1}})

    assert calls == ['dispatch:tool.call', 'flush']
    assert response == {'success': True, 'data': {'value': 1}}


@pytest.mark.asyncio
async def test_worker_execute_preserves_business_failure_when_trace_flush_also_fails(monkeypatch) -> None:
    sdk = WorkerSDK('http://runtime.invalid', 'venv-1')

    async def fail_dispatch(method: str, params: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001
        raise ValueError('business failed')

    async def fail_flush() -> None:
        raise OSError('trace failed')

    monkeypatch.setattr(sdk, '_dispatch', fail_dispatch)
    monkeypatch.setattr(sdk._span_processor, 'flush', fail_flush)
    handler = sdk._sio.handlers['/worker']['execute']

    response = await handler({'method': 'tool.call', 'params': {}})

    assert response == {'success': False, 'error': 'business failed'}


@pytest.mark.asyncio
async def test_worker_invoke_preserves_business_failure_when_trace_flush_also_fails(monkeypatch) -> None:
    import nexus.runtime.models as runtime_models

    from nexus.runtime.models import InvokeResponse, unpack_response

    sdk = WorkerSDK('http://runtime.invalid', 'venv-1')
    emitted: dict[str, Any] = {}

    async def fail_invoke(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        raise ValueError('business failed')

    async def fail_flush() -> None:
        raise OSError('trace failed')

    async def capture_emit(event: str, data: dict[str, Any], *, namespace: str) -> None:
        emitted.update(event=event, data=data, namespace=namespace)

    monkeypatch.setattr(runtime_models, 'execute_invoked_func', fail_invoke)
    monkeypatch.setattr(sdk._span_processor, 'flush', fail_flush)
    monkeypatch.setattr(sdk._sio, 'emit', capture_emit)
    handler = sdk._sio.handlers['/worker']['invoke']

    await handler({'import_path': 'example::call', 'request_id': 'request-1', 'payload': ''})

    assert emitted['event'] == 'invoke:result'
    assert emitted['namespace'] == '/worker'
    with pytest.raises(ValueError, match='business failed'):
        unpack_response(InvokeResponse.model_validate(emitted['data']))


@pytest.mark.asyncio
async def test_worker_trace_flush_handler_surfaces_processor_failure(monkeypatch) -> None:
    sdk = WorkerSDK('http://runtime.invalid', 'venv-1')

    async def fail_flush() -> None:
        raise OSError('worker trace unavailable')

    monkeypatch.setattr(sdk._span_processor, 'flush', fail_flush)
    handler = sdk._sio.handlers['/worker']['trace:flush']

    response = await handler({})

    assert response == {'ok': False, 'error': 'OSError: worker trace unavailable'}


def _runtime_app(monkeypatch):  # noqa: ANN001, ANN202
    from nexus.core import executor_registry
    from nexus.utils import log as nexus_log

    monkeypatch.setattr(nexus_log, 'setup_logging', lambda **_kwargs: None)
    monkeypatch.setattr(executor_registry, 'get_all', lambda: {})
    from nexus.runtime.server import app as runtime_app

    return runtime_app


def test_runtime_health_advertises_trace_flush_capability(monkeypatch) -> None:
    runtime_app = _runtime_app(monkeypatch)

    assert runtime_app.health()['capabilities'] == {'trace_flush': 1}


@pytest.mark.asyncio
async def test_base_trace_flush_skips_legacy_worker_without_capability(monkeypatch) -> None:
    runtime_app = _runtime_app(monkeypatch)
    monkeypatch.setattr(runtime_app, '_worker_sids', {})
    monkeypatch.setattr(runtime_app, '_worker_capabilities', {}, raising=False)
    calls: list[str] = []

    async def call_worker(_event, _data, *, to, namespace, timeout):  # noqa: ANN001, ANN202, ARG001
        calls.append(to)
        return {'ok': False, 'error': 'legacy worker has no trace:flush handler'}

    async def barrier() -> int:
        return 7

    async def otel_flush() -> bool:
        return True

    monkeypatch.setattr(runtime_app.sio, 'call', call_worker)
    monkeypatch.setattr(runtime_app.span_broadcaster, 'barrier', barrier)
    monkeypatch.setattr(runtime_app, 'async_force_flush_best_effort', otel_flush)
    await runtime_app.worker_connect('legacy-sid', {}, {'worker_id': 'legacy-worker', 'venv_id': 'legacy-venv'})

    response = await runtime_app.trace_flush('host-sid')

    assert response == {'ok': True, 'span_batch_id': 7}
    assert calls == []


@pytest.mark.asyncio
async def test_base_trace_flush_requires_advertised_worker_flush_to_succeed(monkeypatch) -> None:
    runtime_app = _runtime_app(monkeypatch)
    monkeypatch.setattr(runtime_app, '_worker_sids', {})
    monkeypatch.setattr(runtime_app, '_worker_capabilities', {}, raising=False)
    calls: list[str] = []

    async def call_worker(_event, _data, *, to, namespace, timeout):  # noqa: ANN001, ANN202, ARG001
        calls.append(to)
        return {'ok': False, 'error': 'worker flush failed'}

    monkeypatch.setattr(runtime_app.sio, 'call', call_worker)
    await runtime_app.worker_connect(
        'new-sid',
        {},
        {
            'worker_id': 'new-worker',
            'venv_id': 'new-venv',
            'capabilities': {'trace_flush': 1},
        },
    )

    response = await runtime_app.trace_flush('host-sid')

    assert response == {'ok': False, 'error': 'RuntimeError: worker flush failed'}
    assert calls == ['new-sid']
