"""OSWorld bridge protocol regressions."""

from __future__ import annotations

import asyncio
import gc
import logging
import weakref

from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
import socketio

from fastapi.dependencies import models as dependency_models

from nexus.runtime.bridge.adapters.ags_osworld import (
    NEXUS_AGENT_COMPACTION_FIELD,
    NEXUS_COMPACTION_DETECTOR_FIELD,
    LLMRequestError,
    RPCReplayGuard,
    RSIOLLMBridge,
    RSIOLLMConfig,
    bridge_error_payload,
)
from nexus.runtime.context import SIOBridge
from nexus.runtime.server.services.anthropic_proxy.server import AnthropicProxy


class FakeSIO:
    """Bridge-like transport. Records emits, dispatches llm_request to a
    server callback, and lets tests drop/reorder/inject responses."""

    def __init__(self) -> None:
        self._handlers: dict[str, list] = {}
        self.requests: list[dict] = []
        self.cancels: list[dict] = []
        self.server = None  # async def server(bridge, envelope) -> None
        self.cancel_server = None  # async def cancel_server(bridge, envelope) -> None
        self.drop_request_attempts: set[int] = set()  # attempts to silently drop
        self.emit_should_fail = False

    def on(self, event, handler=None):
        def deco(fn):
            self._handlers.setdefault(event, []).append(fn)
            return fn

        return deco if handler is None else deco(handler)

    async def emit(self, event, data=None):
        if event == 'llm_request':
            assert isinstance(data, dict), 'LLM requests must carry a dictionary envelope'
            self.requests.append(data)
            if self.emit_should_fail:
                raise ConnectionError('simulated disconnect')
            if data.get('rsio', {}).get('attempt') in self.drop_request_attempts:
                return
            if self.server is not None:
                await self.server(self, data)
        elif event == 'llm_cancel':
            self.cancels.append(data)
            if self.cancel_server is not None:
                await self.cancel_server(self, data)

    async def deliver(self, event, data):
        for h in list(self._handlers.get(event, ())):
            await h(data)


def _fast_cfg() -> RSIOLLMConfig:
    return RSIOLLMConfig(ack_timeout=0.05, report_interval=0.02, report_timeout=0.1, cancel_timeout=0.1)


@pytest.mark.asyncio
async def test_normal_flow_returns_response(caplog):
    sio = FakeSIO()

    async def server(bridge, env):
        rid = env['request_id']
        await bridge.deliver('llm_response', {'request_id': rid, 'rsio': {'request_id': rid, 'kind': 'running'}})
        await bridge.deliver(
            'llm_response',
            {
                'request_id': rid,
                'success': True,
                'response': {'ok': 1},
                'rsio': {'request_id': rid, 'kind': 'final'},
            },
        )

    sio.server = server
    bridge = RSIOLLMBridge(cast(SIOBridge, sio), _fast_cfg())
    with caplog.at_level(logging.DEBUG, logger='nexus.runtime.bridge.adapters.ags_osworld'):
        result = await bridge.request({'model': 'm', 'messages': []}, timeout=2)
    assert result == {'ok': 1}
    # request_id lives at the top level; rsio carries attempt + deadline_at.
    req = sio.requests[0]
    assert req['request_id']
    assert req['rsio']['attempt'] == 1
    assert 'deadline_at' in req['rsio']
    assert 'request_id' not in req['rsio']  # mirror removed
    routine_records = [record for record in caplog.records if record.getMessage().startswith('rsio.worker:')]
    assert routine_records
    assert all(record.levelno == logging.DEBUG for record in routine_records)


@pytest.mark.asyncio
async def test_dropped_first_request_is_resent():
    sio = FakeSIO()
    sio.drop_request_attempts = {1}
    executed = 0

    async def server(bridge, env):
        nonlocal executed
        executed += 1
        rid = env['request_id']
        await bridge.deliver(
            'llm_response',
            {
                'request_id': rid,
                'success': True,
                'response': {'ok': 1},
                'rsio': {'request_id': rid, 'kind': 'final'},
            },
        )

    sio.server = server
    bridge = RSIOLLMBridge(cast(SIOBridge, sio), _fast_cfg())
    result = await bridge.request({'model': 'm'}, timeout=2)
    assert result == {'ok': 1}
    assert executed == 1
    assert len(sio.requests) >= 2  # resent after ack_timeout


@pytest.mark.asyncio
async def test_running_heartbeat_prevents_resend():
    sio = FakeSIO()

    async def server(bridge, env):
        rid = env['request_id']
        # Only attempt 1 should ever reach the server.
        assert env['rsio']['attempt'] == 1
        for _ in range(4):
            await asyncio.sleep(0.03)
            await bridge.deliver('llm_response', {'request_id': rid, 'rsio': {'request_id': rid, 'kind': 'running'}})
        await bridge.deliver(
            'llm_response',
            {
                'request_id': rid,
                'success': True,
                'response': {'ok': 1},
                'rsio': {'request_id': rid, 'kind': 'final'},
            },
        )

    sio.server = server
    bridge = RSIOLLMBridge(cast(SIOBridge, sio), _fast_cfg())
    result = await bridge.request({'model': 'm'}, timeout=2)
    assert result == {'ok': 1}
    assert len(sio.requests) == 1


@pytest.mark.asyncio
async def test_business_timeout_raises():
    sio = FakeSIO()

    async def server(bridge, env):
        return  # never respond

    sio.server = server

    async def cancel_server(bridge, env):
        await bridge.deliver('llm_cancelled', {'request_id': env['request_id'], 'cancelled': True})

    sio.cancel_server = cancel_server
    bridge = RSIOLLMBridge(sio, _fast_cfg())
    with pytest.raises(TimeoutError):
        await bridge.request({'model': 'm'}, timeout=0.2)
    assert [item['request_id'] for item in sio.cancels] == [sio.requests[0]['request_id']]


@pytest.mark.asyncio
async def test_caller_cancellation_waits_for_host_retry_chain_to_stop():
    sio = FakeSIO()
    host_started = asyncio.Event()
    host_stopped = asyncio.Event()
    host_task = None

    async def old_host_retry_chain():
        host_started.set()
        try:
            await asyncio.Future()
        finally:
            await asyncio.sleep(0.02)
            host_stopped.set()

    async def server(_bridge, _env):
        nonlocal host_task
        host_task = asyncio.create_task(old_host_retry_chain())

    async def cancel_server(bridge, env):
        assert host_task is not None
        host_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await host_task
        assert host_stopped.is_set()
        await bridge.deliver('llm_cancelled', {'request_id': env['request_id'], 'cancelled': True})

    sio.server = server
    sio.cancel_server = cancel_server
    bridge = RSIOLLMBridge(sio, _fast_cfg())
    request_task = asyncio.create_task(bridge.request({'model': 'm'}, timeout=2))
    await host_started.wait()

    request_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request_task

    assert host_stopped.is_set()
    assert len(sio.cancels) == 1


@pytest.mark.asyncio
async def test_handler_error_final_raises():
    sio = FakeSIO()

    async def server(bridge, env):
        rid = env['request_id']
        await bridge.deliver(
            'llm_response',
            {
                'request_id': rid,
                'success': False,
                'error': 'boom',
                'rsio': {'request_id': rid, 'kind': 'final'},
            },
        )

    sio.server = server
    bridge = RSIOLLMBridge(cast(SIOBridge, sio), _fast_cfg())
    with pytest.raises(LLMRequestError, match='boom') as excinfo:
        await bridge.request({'model': 'm'}, timeout=2)
    assert excinfo.value.error_code is None


@pytest.mark.asyncio
async def test_handler_error_final_carries_error_code():
    sio = FakeSIO()

    async def server(bridge, env):
        rid = env['request_id']
        await bridge.deliver(
            'llm_response',
            {
                'request_id': rid,
                'success': False,
                'error': 'over the limit',
                'error_code': 'context_length_exceeded',
                'rsio': {'request_id': rid, 'kind': 'final'},
            },
        )

    sio.server = server
    bridge = RSIOLLMBridge(cast(SIOBridge, sio), _fast_cfg())
    with pytest.raises(LLMRequestError) as excinfo:
        await bridge.request({'model': 'm'}, timeout=2)
    assert excinfo.value.error_code == 'context_length_exceeded'


def test_bridge_error_payload_renders_a_coded_failure_as_a_client_error():
    coded = LLMRequestError('over the limit', error_code='context_length_exceeded')

    # 400, not 500: codex retries a 5xx ~30 times and discards its body, so a refusal returned as
    # a server error is both invisible downstream and hugely amplified.
    assert bridge_error_payload(coded) == (
        400,
        {
            'error': {
                'message': '[context_length_exceeded] over the limit',
                'type': 'invalid_request_error',
                'code': 'context_length_exceeded',
            }
        },
    )


def test_bridge_error_payload_leaves_an_uncoded_failure_alone():
    assert bridge_error_payload(LLMRequestError('boom')) is None
    assert bridge_error_payload(ValueError('boom')) is None


@pytest.mark.asyncio
async def test_duplicate_final_ignored():
    sio = FakeSIO()

    async def server(bridge, env):
        rid = env['request_id']
        payload = {
            'request_id': rid,
            'success': True,
            'response': {'ok': 1},
            'rsio': {'request_id': rid, 'kind': 'final'},
        }
        await bridge.deliver('llm_response', payload)
        await bridge.deliver('llm_response', payload)  # second one must be ignored

    sio.server = server
    bridge = RSIOLLMBridge(cast(SIOBridge, sio), _fast_cfg())
    result = await bridge.request({'model': 'm'}, timeout=2)
    assert result == {'ok': 1}


@pytest.mark.asyncio
async def test_lazy_reconnect_replay_after_emit_failure():
    sio = FakeSIO()
    sio.emit_should_fail = True

    async def server(bridge, env):
        rid = env['request_id']
        await bridge.deliver(
            'llm_response',
            {
                'request_id': rid,
                'success': True,
                'response': {'ok': 1},
                'rsio': {'request_id': rid, 'kind': 'final'},
            },
        )

    sio.server = server
    bridge = RSIOLLMBridge(cast(SIOBridge, sio), _fast_cfg())
    task = asyncio.create_task(bridge.request({'model': 'm'}, timeout=2))
    await asyncio.sleep(0.15)  # first emit failed; connection "down"
    sio.emit_should_fail = False  # reconnected
    result = await task
    assert result == {'ok': 1}


@pytest.mark.asyncio
async def test_missing_rsio_response_treated_as_final():
    sio = FakeSIO()

    async def server(bridge, env):
        rid = env['request_id']
        await bridge.deliver('llm_response', {'request_id': rid, 'success': True, 'response': {'ok': 1}})

    sio.server = server
    bridge = RSIOLLMBridge(cast(SIOBridge, sio), _fast_cfg())
    result = await bridge.request({'model': 'm'}, timeout=2)
    assert result == {'ok': 1}


@pytest.mark.asyncio
@pytest.mark.parametrize('rsio_enabled', [True, False])
async def test_disabled_business_timeout_waits_for_delayed_response(rsio_enabled, monkeypatch):
    monkeypatch.delenv('NEXUS_DISABLE_LLM_RSIO', raising=False)
    sio = FakeSIO()
    delivery_tasks: list[asyncio.Task] = []

    async def server(bridge, env):
        async def deliver_response():
            await asyncio.sleep(0.1)
            rid = env['request_id']
            await bridge.deliver(
                'llm_response',
                {
                    'request_id': rid,
                    'success': True,
                    'response': {'ok': 1},
                    'rsio': {'request_id': rid, 'kind': 'final'},
                },
            )

        delivery_tasks.append(asyncio.create_task(deliver_response()))

    sio.server = server
    bridge = RSIOLLMBridge(
        cast(SIOBridge, sio),
        RSIOLLMConfig(enable=rsio_enabled, ack_timeout=0.2, report_interval=0.05, report_timeout=0.3),
    )
    result = await asyncio.wait_for(bridge.request({'model': 'm'}, timeout=None), timeout=0.5)
    await asyncio.gather(*delivery_tasks)

    assert result == {'ok': 1}
    if rsio_enabled:
        assert sio.requests[0]['rsio']['deadline_at'] is None
    else:
        assert 'rsio' not in sio.requests[0]


@pytest.fixture
def raw_transport(monkeypatch: pytest.MonkeyPatch) -> tuple[socketio.AsyncClient, asyncio.Queue[dict[str, Any]]]:
    monkeypatch.delenv('NEXUS_DISABLE_LLM_RSIO', raising=False)
    raw_sio = socketio.AsyncClient()
    requests: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def emit(event: str, data: dict[str, Any], namespace: str | None = None) -> None:
        assert event == 'llm_request', 'The transport should only receive LLM requests'
        assert namespace == '/worker', 'Both wrappers must use the worker namespace'
        requests.put_nowait(data)

    monkeypatch.setattr(raw_sio, 'emit', AsyncMock(side_effect=emit))
    return raw_sio, requests


@pytest.mark.asyncio
@pytest.mark.parametrize('channels', [None, ('channel-a', 'channel-b')])
async def test_real_wrappers_share_overlapping_requests_and_accept_legacy_finals(
    raw_transport: tuple[socketio.AsyncClient, asyncio.Queue[dict[str, Any]]],
    channels: tuple[str, str] | None,
) -> None:
    raw_sio, requests = raw_transport
    bridges = [RSIOLLMBridge(SIOBridge(raw_sio, '/worker'), channel_id=channel) for channel in channels or (None, None)]
    params = {'model': 'm', NEXUS_AGENT_COMPACTION_FIELD: True, NEXUS_COMPACTION_DETECTOR_FIELD: 'test'}
    tasks = [asyncio.create_task(bridge.request(params, timeout=None)) for bridge in bridges]
    try:
        envelopes = [await asyncio.wait_for(requests.get(), timeout=1) for _ in tasks]
        assert all(not task.done() for task in tasks), 'Both requests must overlap before either final is delivered'
        assert bridges[0]._pending is bridges[1]._pending, 'Wrappers sharing one socket must share the response table'
        assert len(bridges[0]._pending) == 2, 'Both overlapping requests must be registered'
        assert set(raw_sio.handlers['/worker']) == {'llm_response', 'llm_cancelled'}
        for index, envelope in reversed(list(enumerate(envelopes))):
            assert envelope['rsio']['deadline_at'] is None, 'A disabled timeout must stay disabled on the wire'
            assert 'channel_id' not in envelope, 'Channel routing metadata must not become a model parameter'
            if channels is None:
                assert 'channel_id' not in envelope['rsio'], 'Existing callers must not emit a channel field'
            else:
                assert envelope['rsio']['channel_id'] == channels[index], 'Each request must carry its own channel'
            for field in (NEXUS_AGENT_COMPACTION_FIELD, NEXUS_COMPACTION_DETECTOR_FIELD):
                assert envelope[field] == params[field], 'Compaction metadata must survive bridge forwarding'
            await raw_sio.handlers['/worker']['llm_response'](
                {
                    'request_id': envelope['request_id'],
                    'success': True,
                    'response': {'index': index},
                }
            )
        assert await asyncio.gather(*tasks) == [{'index': 0}, {'index': 1}], 'Finals must route by request ID'
        assert not bridges[0]._pending, 'Completed requests must leave no shared pending entries'
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_close_cancels_only_owned_request_and_ignores_late_finals(
    raw_transport: tuple[socketio.AsyncClient, asyncio.Queue[dict[str, Any]]],
) -> None:
    raw_sio, requests = raw_transport
    bridge_a = RSIOLLMBridge(SIOBridge(raw_sio, '/worker'), channel_id='a')
    bridge_b = RSIOLLMBridge(SIOBridge(raw_sio, '/worker'), channel_id='b')
    tasks = [asyncio.create_task(bridge.request({'model': 'm'}, timeout=None)) for bridge in (bridge_a, bridge_b)]
    try:
        envelopes = [await asyncio.wait_for(requests.get(), timeout=1) for _ in tasks]
        assert all(not task.done() for task in tasks), 'Close must run while both requests are still pending'
        pending_a = bridge_a._pending[envelopes[0]['request_id']]
        pending_b = bridge_b._pending[envelopes[1]['request_id']]
        bridge_a.close()
        bridge_a.close()
        assert pending_a.future.cancelled(), 'Close must cancel the owned response future'
        assert not pending_b.future.done(), 'Close must not cancel a sibling bridge response future'
        assert set(bridge_b._pending) == {pending_b.request_id}, 'Close must only remove owned shared entries'
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(tasks[0], timeout=1)
        for envelope in envelopes:
            final = {'request_id': envelope['request_id'], 'success': True, 'response': {'channel': 'b'}}
            await raw_sio.handlers['/worker']['llm_response'](final)
            await raw_sio.handlers['/worker']['llm_response'](final)
        assert await asyncio.wait_for(tasks[1], timeout=1) == {'channel': 'b'}, 'The surviving bridge must complete'
        with pytest.raises(RuntimeError, match='closed'):
            await bridge_a.request({'model': 'm'})
        assert requests.empty(), 'A closed bridge must not emit or retry a request'
        assert not bridge_a._owned_requests, 'Closed bridge task tracking must be empty'
        assert not bridge_b._owned_requests, 'Completed bridge task tracking must be empty'
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        bridge_a.close()
        bridge_b.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('rsio_enabled', [True, False])
async def test_close_during_blocked_emit_cancels_driver_without_retry(
    rsio_enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('NEXUS_DISABLE_LLM_RSIO', raising=False)
    sio = FakeSIO()
    entered_emit = asyncio.Event()
    left_emit = asyncio.Event()

    async def server(transport: FakeSIO, envelope: dict[str, Any]) -> None:
        entered_emit.set()
        try:
            await asyncio.Event().wait()
        finally:
            left_emit.set()

    sio.server = server
    bridge = RSIOLLMBridge(cast(SIOBridge, sio), RSIOLLMConfig(enable=rsio_enabled, ack_timeout=0.01))
    request_task = asyncio.create_task(bridge.request({'model': 'm'}, timeout=None))
    try:
        await asyncio.wait_for(entered_emit.wait(), timeout=1)
        pending = next(iter(bridge._pending.values()))
        bridge.close()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(request_task, timeout=1)
        assert left_emit.is_set(), 'Close must interrupt a blocked emit, not just cancel its response future'
        assert pending.future.cancelled(), 'Close must cancel the response future in both modes'
        assert not bridge._pending, 'Close must release shared pending state'
        assert not bridge._owned_requests, 'Close must release owned driver tasks'
        assert len(sio.requests) == 1, 'Close must prevent further retries'
    finally:
        request_task.cancel()
        await asyncio.gather(request_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_caller_cancellation_removes_owned_and_shared_pending(
    raw_transport: tuple[socketio.AsyncClient, asyncio.Queue[dict[str, Any]]],
) -> None:
    raw_sio, requests = raw_transport
    bridge = RSIOLLMBridge(SIOBridge(raw_sio, '/worker'))
    task = asyncio.create_task(bridge.request({'model': 'm'}, timeout=None))
    await asyncio.wait_for(requests.get(), timeout=1)
    pending = next(iter(bridge._pending.values()))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pending.future.cancelled(), 'Caller cancellation must also cancel the shielded response future'
    assert not bridge._pending, 'Caller cancellation must remove shared pending state'
    assert not bridge._owned_requests, 'Caller cancellation must remove owned driver tasks'
    bridge.close()


@pytest.mark.asyncio
async def test_retries_preserve_channel_and_close_stops_driver(
    raw_transport: tuple[socketio.AsyncClient, asyncio.Queue[dict[str, Any]]],
) -> None:
    raw_sio, requests = raw_transport
    bridge = RSIOLLMBridge(SIOBridge(raw_sio, '/worker'), _fast_cfg(), channel_id='retry-channel')
    task = asyncio.create_task(bridge.request({'model': 'm'}, timeout=None))
    try:
        first = await asyncio.wait_for(requests.get(), timeout=1)
        retry = await asyncio.wait_for(requests.get(), timeout=1)
        assert retry['request_id'] == first['request_id'], 'Retry must preserve host deduplication identity'
        assert retry['rsio']['attempt'] == 2, 'The retry must increment its attempt'
        assert retry['rsio']['channel_id'] == 'retry-channel', 'Retries must preserve the routing channel'
        bridge.close()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert requests.empty(), 'The closed request must not emit more retries'
        assert not bridge._pending, 'The retry driver must remove shared pending state'
        assert not bridge._owned_requests, 'The retry driver must remove owned task tracking'
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('rsio_enabled', [True, False])
async def test_close_after_error_final_during_emit_consumes_future_exception(
    rsio_enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('NEXUS_DISABLE_LLM_RSIO', raising=False)
    sio = FakeSIO()
    final_received = asyncio.Event()
    loop = asyncio.get_running_loop()
    original_handler = loop.get_exception_handler()
    unhandled_errors: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda event_loop, context: unhandled_errors.append(context))

    async def server(transport: FakeSIO, envelope: dict[str, Any]) -> None:
        await transport.deliver(
            'llm_response', {'request_id': envelope['request_id'], 'success': False, 'error': 'host failed'}
        )
        final_received.set()
        await asyncio.Event().wait()

    sio.server = server
    bridge = RSIOLLMBridge(cast(SIOBridge, sio), RSIOLLMConfig(enable=rsio_enabled))
    task = asyncio.create_task(bridge.request({'model': 'm'}, timeout=None))
    try:
        await asyncio.wait_for(final_received.wait(), timeout=1)
        bridge.close()
        with pytest.raises(asyncio.CancelledError):
            await task
        gc.collect()
        assert not unhandled_errors, 'A final racing with close must not leave an unretrieved future exception'
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        loop.set_exception_handler(original_handler)


@pytest.mark.parametrize('disable_mode', ['config', 'environment'])
def test_channel_in_legacy_mode_raises(disable_mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('NEXUS_DISABLE_LLM_RSIO', raising=False)
    if disable_mode == 'environment':
        monkeypatch.setenv('NEXUS_DISABLE_LLM_RSIO', '1')
    config = RSIOLLMConfig(enable=disable_mode != 'config')
    with pytest.raises(ValueError, match='channel_id.*RSIO'):
        RSIOLLMBridge(cast(SIOBridge, FakeSIO()), config, channel_id='session-channel')


@pytest.mark.asyncio
async def test_legacy_anthropic_proxy_cycles_do_not_retain_bridges_or_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    # This specifically exercises the default proxy, whose pending state must
    # remain isolated from the opt-in cancellation extension.
    from nexus.runtime.server.services.rsio_llm_bridge import _SHARED_HANDLER_REGISTERED, _SHARED_PENDING

    # NOTE(wallyshi): FastAPI 0.141 caches route closures; exclude its bounded caches when testing bridge retention.
    for name in ('_is_gen_callable_cached', '_is_async_gen_callable_cached', '_is_coroutine_callable_cached'):
        cached_classifier = getattr(dependency_models, name, None)
        if cached_classifier is not None:
            monkeypatch.setattr(dependency_models, name, cached_classifier.__wrapped__)
    gc.collect()
    pending_count = len(_SHARED_PENDING)
    handler_count = len(_SHARED_HANDLER_REGISTERED)
    raw_sio = socketio.AsyncClient()
    raw_reference = weakref.ref(raw_sio)
    bridge_references = []
    proxy_references = []
    handler = None
    for _ in range(12):
        proxy = AnthropicProxy(SIOBridge(raw_sio, '/worker'))
        app = proxy.create_app()
        bridge_references.append(weakref.ref(proxy.bridge))
        proxy_references.append(weakref.ref(proxy))
        current_handler = raw_sio.handlers['/worker']['llm_response']
        if handler is not None:
            assert current_handler is handler, 'Legacy proxy cycles must reuse the same socket handler'
        handler = current_handler
        await proxy.stop()
        del app, proxy
        gc.collect()
        assert all(reference() is None for reference in bridge_references), 'Stopped legacy bridges must be collectible'
        assert all(reference() is None for reference in proxy_references), 'Stopped legacy proxies must be collectible'
    assert len(_SHARED_PENDING) == pending_count + 1, 'One live socket must have only one pending table'
    assert len(_SHARED_HANDLER_REGISTERED) == handler_count + 1, 'One live socket must have only one handler marker'
    del raw_sio
    gc.collect()
    assert raw_reference() is None, 'Shared dispatch must not retain the underlying socket'
    assert len(_SHARED_PENDING) == pending_count, 'Dead sockets must leave no global pending table'
    assert len(_SHARED_HANDLER_REGISTERED) == handler_count, 'Dead sockets must leave no global handler marker'


@pytest.mark.asyncio
async def test_rpc_replay_guard_deduplicates_inflight_requests() -> None:
    guard = RPCReplayGuard()
    handler_started = asyncio.Event()
    release_handler = asyncio.Event()
    call_count = 0

    async def handler() -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        handler_started.set()
        await release_handler.wait()
        return {'call_count': call_count}

    first_task = asyncio.create_task(guard.run('rpc:call_function', 'req-1', handler))
    await handler_started.wait()
    second_task = asyncio.create_task(guard.run('rpc:call_function', 'req-1', handler))

    await asyncio.sleep(0)
    release_handler.set()

    first_result, second_result = await asyncio.gather(first_task, second_task)

    assert first_result == {'call_count': 1}
    assert second_result == {'call_count': 1}
    assert call_count == 1


@pytest.mark.asyncio
async def test_rpc_replay_guard_replays_completed_results() -> None:
    guard = RPCReplayGuard()
    call_count = 0

    async def handler() -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        return {'call_count': call_count}

    first_result = await guard.run('rpc:call_tool', 'req-2', handler)
    second_result = await guard.run('rpc:call_tool', 'req-2', handler)

    assert first_result == {'call_count': 1}
    assert second_result == {'call_count': 1}
    assert call_count == 1


@pytest.mark.asyncio
async def test_rpc_replay_guard_discards_completed_results() -> None:
    guard = RPCReplayGuard()
    call_count = 0

    async def handler() -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        return {'call_count': call_count}

    assert await guard.run('llm_request', 'req-delivered', handler) == {'call_count': 1}

    assert await guard.discard('llm_request', 'req-delivered') is True
    assert await guard.discard('llm_request', 'req-delivered') is False
    assert await guard.run('llm_request', 'req-delivered', handler) == {'call_count': 2}


@pytest.mark.asyncio
async def test_rpc_replay_guard_discards_completed_results_after_delay() -> None:
    guard = RPCReplayGuard()
    call_count = 0

    async def handler() -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        return {'call_count': call_count}

    assert await guard.run('llm_request', 'req-delayed', handler) == {'call_count': 1}
    assert await guard.discard('llm_request', 'req-delayed', delay=0.01) is True

    assert await guard.run('llm_request', 'req-delayed', handler) == {'call_count': 1}
    await asyncio.sleep(0.02)
    assert await guard.run('llm_request', 'req-delayed', handler) == {'call_count': 2}


@pytest.mark.asyncio
async def test_rpc_replay_guard_delayed_discard_does_not_remove_replacement() -> None:
    guard = RPCReplayGuard()
    call_count = 0

    async def handler() -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        return {'call_count': call_count}

    assert await guard.run('llm_request', 'req-replaced', handler) == {'call_count': 1}
    assert await guard.discard('llm_request', 'req-replaced', delay=0.01) is True
    assert await guard.discard('llm_request', 'req-replaced') is True
    assert await guard.run('llm_request', 'req-replaced', handler) == {'call_count': 2}

    await asyncio.sleep(0.02)
    assert await guard.run('llm_request', 'req-replaced', handler) == {'call_count': 2}


@pytest.mark.asyncio
async def test_rpc_replay_guard_does_not_discard_inflight_results() -> None:
    guard = RPCReplayGuard()
    handler_started = asyncio.Event()
    release_handler = asyncio.Event()

    async def handler() -> str:
        handler_started.set()
        await release_handler.wait()
        return 'done'

    task = asyncio.create_task(guard.run('llm_request', 'req-inflight', handler))
    await handler_started.wait()

    assert await guard.discard('llm_request', 'req-inflight') is False
    assert 'llm_request:req-inflight' in guard._entries

    release_handler.set()
    assert await task == 'done'


@pytest.mark.asyncio
async def test_rpc_replay_guard_cleans_up_cancelled_owner() -> None:
    guard = RPCReplayGuard()
    handler_started = asyncio.Event()
    call_count = 0

    async def blocked_handler() -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        handler_started.set()
        await asyncio.Future()
        return {'call_count': call_count}

    owner_task = asyncio.create_task(guard.run('rpc:call_function', 'req-3', blocked_handler))
    await handler_started.wait()

    duplicate_task = asyncio.create_task(guard.run('rpc:call_function', 'req-3', blocked_handler))
    await asyncio.sleep(0)

    owner_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await owner_task
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(duplicate_task, timeout=0.5)

    async def retry_handler() -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        return {'call_count': call_count}

    retry_result = await guard.run('rpc:call_function', 'req-3', retry_handler)

    assert retry_result == {'call_count': 2}
    assert call_count == 2


@pytest.mark.asyncio
async def test_rpc_replay_guard_failed_owner_does_not_leak_future_exception() -> None:
    """A failing owner with no duplicate must not leave an unretrieved future.

    Reproduces the "Future exception was never retrieved" log noise: the owner
    stores the exception on its replay future and re-raises, but never awaits that
    future itself. Without the retrieving done-callback the pruned/GC'd future logs
    a spurious asyncio error for every failed request.
    """

    class BoomError(RuntimeError):
        pass

    async def failing_handler() -> dict[str, int]:
        raise BoomError('context length exceeded')

    guard = RPCReplayGuard()

    # Owner runs and raises; the exception must propagate to the caller unchanged.
    with pytest.raises(BoomError):
        await guard.run('rpc:call_function', 'req-boom', failing_handler)

    # Done-callbacks are scheduled via loop.call_soon; let them run.
    await asyncio.sleep(0)

    # Grab the stored future before pruning removes it, then confirm its exception
    # was already retrieved by the done-callback (so GC would not warn).
    entry = guard._entries.get('rpc:call_function:req-boom')
    assert entry is not None
    stored_future = entry.future
    assert stored_future.done()
    assert stored_future._log_traceback is False  # asyncio's "needs retrieval" flag cleared

    # Dropping and collecting the future must not raise or warn.
    del entry
    del stored_future
    gc.collect()
