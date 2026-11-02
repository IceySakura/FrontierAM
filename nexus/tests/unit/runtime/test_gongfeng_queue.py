"""Tests for GongfengRuntimeProvider quota-queue waiting support.

Covers the client-side adapter for the sandbox v3 backend's quota_full_policy=queue
behavior: queued status detection, polling until scheduled, queue_wait_max timeout
and best-effort cancellation, queue-failure remark prefix matching, watchdog status
narrowing, and the acquire-level override of queue_wait_max.

Network is fully stubbed by patching ``_call_api`` (or higher-level helpers when
clearer), and ``_create_runtime`` / ``_start_watchdog`` are patched out where they
would otherwise pull in aiohttp / spawn tasks that outlive the test.
"""

from __future__ import annotations

import asyncio
import logging

from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from nexus.runtime.client.base import RuntimeUnhealthyError
from nexus.runtime.client.remote import RemoteRuntime
from nexus.runtime.providers import gongfeng as gf
from nexus.runtime.providers.gongfeng import (
    RUNTIME_STATUS_PENDING,
    RUNTIME_STATUS_QUEUED,
    RUNTIME_STATUS_SCHEDULED,
    RUNTIME_STATUS_STOPPED,
    GongfengRuntimeProvider,
    RuntimeQueueFailedError,
    RuntimeQueueWaitTimeoutError,
)


def _make_provider(**overrides: Any) -> GongfengRuntimeProvider:
    """Build a provider with safe defaults for unit tests (no network, no stderr noise)."""
    kwargs: dict[str, Any] = {
        'api_base_url': 'http://gongfeng.example.com',
        'token': 'unit-test-token',
        'image': 'unit-test/image:latest',
        'print_port_mappings': False,
    }
    kwargs.update(overrides)
    return GongfengRuntimeProvider(**kwargs)


def _patch_runtime_creation(provider: GongfengRuntimeProvider) -> tuple[MagicMock, MagicMock]:
    """Patch _create_runtime to return a bare MagicMock and stub out _start_watchdog.

    Returns the (create_runtime_mock, start_watchdog_mock) pair so tests can assert
    against them. The mock runtime exposes port_mappings (real dict) so the production
    code's ``runtime.port_mappings[port] = url`` assignment works as in real flow.
    """
    fake_runtime = MagicMock()
    fake_runtime.port_mappings = {}
    create_mock = MagicMock(return_value=fake_runtime)
    start_wd_mock = MagicMock()
    provider._create_runtime = create_mock  # type: ignore[method-assign]
    provider._start_watchdog = start_wd_mock  # type: ignore[method-assign]
    return create_mock, start_wd_mock


@pytest.mark.asyncio
async def test_list_runtimes_forwards_batch_id_filter() -> None:
    provider = _make_provider()
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={
            'data': {
                'runtimes': [{'runtime_id': 'runtime-1', 'batch_id': 'batch-a'}],
                'total': 0,
            }
        }
    )

    runtimes, total = await provider.list_runtimes(batch_id='batch-a', page=2, per_page=5000)

    assert runtimes == [{'runtime_id': 'runtime-1', 'batch_id': 'batch-a'}]
    assert total == 0
    provider._call_api.assert_awaited_once_with(
        '/sandbox/v3/runtime/list',
        {'page': 2, 'per_page': 5000, 'batch_id': 'batch-a'},
    )


@pytest.mark.asyncio
async def test_batch_id_is_omitted_when_not_configured() -> None:
    provider = _make_provider()
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={'code': 0, 'data': {'runtime_id': 'runtime-1', 'url': 'http://x', 'port_mappings': {}}}
    )

    await provider._start_runtime(image='img', runtime_id='runtime-1')
    await provider._stop_runtime(['runtime-1'])

    assert provider.batch_id is None
    start_payload = provider._call_api.await_args_list[0].args[1]
    assert 'batch_id' not in start_payload
    assert provider._call_api.await_args_list[1] == call('/sandbox/v3/runtime/stop', {'runtime_ids': ['runtime-1']})


@pytest.mark.asyncio
async def test_configured_batch_id_is_sent_on_start() -> None:
    provider = _make_provider(batch_id='batch-a')
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={'code': 0, 'data': {'runtime_id': 'runtime-1', 'url': 'http://x', 'port_mappings': {}}}
    )

    await provider._start_runtime(image='img', runtime_id='runtime-1')

    assert provider._call_api.await_args.args[1]['batch_id'] == 'batch-a'


@pytest.mark.asyncio
async def test_stop_runtime_retries_transient_api_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _make_provider(batch_id='batch-a')
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        side_effect=[RuntimeError('temporary-1'), RuntimeError('temporary-2'), {'data': {}}]
    )
    sleep = AsyncMock()
    monkeypatch.setattr(gf.asyncio, 'sleep', sleep)

    await provider._stop_runtime(['runtime-1'])

    assert provider._call_api.await_count == 3
    assert (
        provider._call_api.await_args_list
        == [
            call(
                '/sandbox/v3/runtime/stop',
                {'runtime_ids': ['runtime-1'], 'batch_id': 'batch-a'},
            )
        ]
        * 3
    )
    assert [call.args[0] for call in sleep.await_args_list] == [0.5, 1.0]


@pytest.mark.asyncio
async def test_release_keeps_runtime_tracked_when_stop_fails() -> None:
    provider = _make_provider()
    provider._runtime_ids['main'] = 'runtime-1'
    provider._active_runtimes['main'] = MagicMock()
    provider._active_runtimes['main'].ws_disconnect = AsyncMock()
    provider._stop_runtime = AsyncMock(side_effect=RuntimeError('stop failed'))  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match='stop failed'):
        await provider.release('main')

    assert provider._runtime_ids == {'main': 'runtime-1'}
    assert 'main' in provider._active_runtimes


@pytest.mark.asyncio
async def test_release_all_keeps_runtimes_tracked_when_stop_fails() -> None:
    provider = _make_provider()
    provider._runtime_ids.update({'main': 'runtime-1', 'worker': 'runtime-2'})
    provider._stop_runtime = AsyncMock(side_effect=RuntimeError('stop failed'))  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match='stop failed'):
        await provider.release_all()

    assert provider._runtime_ids == {'main': 'runtime-1', 'worker': 'runtime-2'}


# ====================== _start_runtime parsing ======================


@pytest.mark.asyncio
async def test_start_runtime_infers_queued_from_empty_url() -> None:
    """Real Start responses carry no status field — empty url alone must signal queued.

    Validated against the live sandbox v3 API: the queued path returns
    ``{"code":0,"msg":"queued, ...","data":{"runtime_id":"...","url":"","port_mappings":{}}}``
    with no ``status`` field. Inferring from url emptiness is the only robust signal.
    """
    provider = _make_provider()
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={
            'code': 0,
            'msg': 'queued, please poll GetRuntime until status=scheduled',
            'data': {'runtime_id': 'rid-queued-1', 'url': '', 'port_mappings': {}},
        }
    )

    runtime_id, url, port_mappings, status = await provider._start_runtime(image='img')

    assert runtime_id == 'rid-queued-1'
    assert url == ''
    assert port_mappings == {}
    assert status == RUNTIME_STATUS_QUEUED


@pytest.mark.asyncio
async def test_start_runtime_infers_scheduled_from_populated_url() -> None:
    """A populated url means the runtime is fully scheduled (no status field needed)."""
    provider = _make_provider()
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={
            'code': 0,
            'msg': '',
            'data': {
                'runtime_id': 'rid-scheduled-1',
                'url': 'http://nexus-xxx.example.com:8053',
                'port_mappings': {},
            },
        }
    )

    runtime_id, url, port_mappings, status = await provider._start_runtime(image='img')

    assert runtime_id == 'rid-scheduled-1'
    assert url == 'http://nexus-xxx.example.com:8053'
    assert status == RUNTIME_STATUS_SCHEDULED


@pytest.mark.asyncio
async def test_start_runtime_rejects_missing_runtime_id() -> None:
    """Missing runtime_id is unrecoverable — must not silently proceed."""
    provider = _make_provider()
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={'code': 0, 'data': {'url': 'http://x', 'port_mappings': {}}}
    )

    with pytest.raises(RuntimeError, match='missing runtime_id'):
        await provider._start_runtime(image='img')


# ====================== acquire: ambiguous start cleanup ======================


@pytest.mark.asyncio
async def test_acquire_start_failure_stops_client_generated_runtime_id() -> None:
    provider = _make_provider()
    provider._start_runtime = AsyncMock(side_effect=RuntimeError('start timed out'))  # type: ignore[method-assign]
    stop_mock = AsyncMock()
    provider._stop_runtime = stop_mock  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match='start timed out'):
        await provider.acquire('runtime-start-timeout')

    requested_runtime_id = provider._start_runtime.await_args.kwargs['runtime_id']
    assert requested_runtime_id.startswith('nexus-')
    stop_mock.assert_awaited_once_with([requested_runtime_id])


@pytest.mark.asyncio
async def test_acquire_start_failure_does_not_stop_caller_owned_runtime_id() -> None:
    provider = _make_provider()
    provider._start_runtime = AsyncMock(side_effect=RuntimeError('start rejected'))  # type: ignore[method-assign]
    stop_mock = AsyncMock()
    provider._stop_runtime = stop_mock  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match='start rejected'):
        await provider.acquire('runtime-custom-id', runtime_id='caller-owned-runtime')

    assert provider._start_runtime.await_args.kwargs['runtime_id'] == 'caller-owned-runtime'
    stop_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_acquire_cancelled_during_start_stops_client_generated_runtime_id() -> None:
    provider = _make_provider()
    start_entered = asyncio.Event()

    async def wait_until_cancelled(**_: Any) -> tuple[str, str, dict[str, str], str]:
        start_entered.set()
        await asyncio.Future()
        raise AssertionError('unreachable')

    provider._start_runtime = AsyncMock(side_effect=wait_until_cancelled)  # type: ignore[method-assign]
    stop_mock = AsyncMock()
    provider._stop_runtime = stop_mock  # type: ignore[method-assign]

    task = asyncio.create_task(provider.acquire('runtime-start-cancelled'))
    await start_entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    requested_runtime_id = provider._start_runtime.await_args.kwargs['runtime_id']
    assert requested_runtime_id.startswith('nexus-')
    stop_mock.assert_awaited_once_with([requested_runtime_id])


@pytest.mark.asyncio
async def test_acquire_start_cleanup_failure_preserves_start_error() -> None:
    provider = _make_provider()
    provider._start_runtime = AsyncMock(side_effect=RuntimeError('start timed out'))  # type: ignore[method-assign]
    provider._stop_runtime = AsyncMock(side_effect=RuntimeError('stop failed'))  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match='start timed out'):
        await provider.acquire('runtime-cleanup-failed')


# ====================== acquire: default is response-driven (always polls) ======================


@pytest.mark.asyncio
async def test_acquire_with_default_polls_until_scheduled(monkeypatch: pytest.MonkeyPatch) -> None:
    """queue_wait_max=0 (default) is response-driven: backend queue → poll (no deadline) until scheduled.

    Regression guard for the legacy behavior where the default refused to wait and raised instead
    of polling. The client must now key purely off the empty url (status=queued) and wait it out,
    regardless of why the backend enqueued (token quota queue, high-load admission throttle, ...).
    """
    monkeypatch.setattr(gf, 'QUEUE_POLL_INTERVAL', 0.01)
    provider = _make_provider()  # queue_wait_max defaults to 0 == "no client-side deadline"
    provider._start_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value=('rid-default', '', {}, RUNTIME_STATUS_QUEUED)
    )
    # Two polls (still queued, then scheduled) proves the default does not fail-fast after one
    # iteration — i.e. a zero budget is treated as unbounded, not as an immediate deadline.
    get_responses = [
        {'status': RUNTIME_STATUS_QUEUED},
        {
            'status': RUNTIME_STATUS_SCHEDULED,
            'url': 'http://allocated.example.com:32080',
            'port_mappings': {},
        },
    ]
    provider._get_runtime = AsyncMock(side_effect=get_responses)  # type: ignore[method-assign]
    create_mock, _ = _patch_runtime_creation(provider)

    runtime = await provider.acquire('runtime-a')

    assert runtime is create_mock.return_value
    assert provider._get_runtime.await_count == 2
    # runtime_id tracked throughout so release()/release_all() can cancel a mid-wait queue entry.
    assert provider._runtime_ids['runtime-a'] == 'rid-default'
    assert provider._active_runtimes['runtime-a'] is runtime


# ====================== acquire: polls until scheduled ======================


@pytest.mark.asyncio
async def test_acquire_polls_until_scheduled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Queued → queued → scheduled with url should produce a usable runtime."""
    monkeypatch.setattr(gf, 'QUEUE_POLL_INTERVAL', 0.01)
    provider = _make_provider(queue_wait_max=5)
    provider._start_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value=('rid-poll', '', {}, RUNTIME_STATUS_QUEUED)
    )
    get_responses = [
        {'status': RUNTIME_STATUS_QUEUED, 'enqueued_at': '2026-05-27 11:00:00'},
        {'status': RUNTIME_STATUS_QUEUED, 'enqueued_at': '2026-05-27 11:00:00'},
        {
            'status': RUNTIME_STATUS_SCHEDULED,
            'url': 'http://allocated.example.com:32080',
            'port_mappings': {'8023': 'http://allocated.example.com:32081'},
        },
    ]
    provider._get_runtime = AsyncMock(side_effect=get_responses)  # type: ignore[method-assign]
    create_mock, start_wd_mock = _patch_runtime_creation(provider)

    runtime = await provider.acquire('runtime-poll')

    assert runtime is create_mock.return_value
    assert provider._get_runtime.await_count == 3
    create_mock.assert_called_once()
    assert create_mock.call_args.args[0] == 'http://allocated.example.com:32080'
    start_wd_mock.assert_called_once_with('runtime-poll', runtime, 'rid-poll')
    assert provider._active_runtimes['runtime-poll'] is runtime
    assert runtime.port_mappings == {8023: 'http://allocated.example.com:32081'}


# ====================== acquire: pending phase narration ======================

_PROVISIONING_LINE = 'resources granted, now provisioning the sandbox'
_PENDING_HEARTBEAT = 'provisioning sandbox (resources already granted)'
_QUEUED_HEARTBEAT = 'still queued for resources'


@pytest.mark.asyncio
async def test_acquire_announces_pending_once_before_scheduled(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """queued → pending → scheduled narrates the phase change exactly once."""
    monkeypatch.setattr(gf, 'QUEUE_POLL_INTERVAL', 0.01)
    provider = _make_provider(queue_wait_max=5)
    provider._start_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value=('rid-pending', '', {}, RUNTIME_STATUS_QUEUED)
    )
    get_responses = [
        {'status': RUNTIME_STATUS_QUEUED},
        {'status': RUNTIME_STATUS_PENDING},
        {
            'status': RUNTIME_STATUS_SCHEDULED,
            'url': 'http://allocated.example.com:32080',
            'port_mappings': {},
        },
    ]
    provider._get_runtime = AsyncMock(side_effect=get_responses)  # type: ignore[method-assign]
    create_mock, _ = _patch_runtime_creation(provider)

    with caplog.at_level(logging.DEBUG, logger='nexus.runtime.providers.gongfeng'):
        runtime = await provider.acquire('runtime-pending')

    assert runtime is create_mock.return_value
    messages = [r.getMessage() for r in caplog.records]
    assert len([m for m in messages if _PROVISIONING_LINE in m]) == 1
    assert any('started after queuing' in m for m in messages)


@pytest.mark.asyncio
async def test_pending_heartbeat_replaces_queued_wording(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Consecutive pending polls keep one transition line and switch the heartbeat wording."""
    monkeypatch.setattr(gf, 'QUEUE_POLL_INTERVAL', 0.01)
    monkeypatch.setattr(gf, 'QUEUE_PROGRESS_LOG_INTERVAL', 0.001)
    provider = _make_provider(queue_wait_max=5)
    provider._start_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value=('rid-pending-long', '', {}, RUNTIME_STATUS_QUEUED)
    )
    get_responses = [
        {'status': RUNTIME_STATUS_PENDING},
        {'status': RUNTIME_STATUS_PENDING},
        {'status': RUNTIME_STATUS_PENDING},
        {
            'status': RUNTIME_STATUS_SCHEDULED,
            'url': 'http://allocated.example.com:32080',
            'port_mappings': {},
        },
    ]
    provider._get_runtime = AsyncMock(side_effect=get_responses)  # type: ignore[method-assign]
    _patch_runtime_creation(provider)

    with caplog.at_level(logging.DEBUG, logger='nexus.runtime.providers.gongfeng'):
        await provider.acquire('runtime-pending-long')

    messages = [r.getMessage() for r in caplog.records]
    assert len([m for m in messages if _PROVISIONING_LINE in m]) == 1
    assert [m for m in messages if _PENDING_HEARTBEAT in m]
    assert not [m for m in messages if _QUEUED_HEARTBEAT in m]


@pytest.mark.asyncio
async def test_unregistered_status_keeps_polling_without_pending_wording(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An unknown intermediate status is waited out, not treated as death nor as pending.

    Guards the backend's freedom to add in-flight statuses: the SDK only short-circuits on the
    explicit dead allowlist, so anything it has never heard of must fall through to the poll.
    """
    monkeypatch.setattr(gf, 'QUEUE_POLL_INTERVAL', 0.01)
    monkeypatch.setattr(gf, 'QUEUE_PROGRESS_LOG_INTERVAL', 0.001)
    provider = _make_provider(queue_wait_max=5)
    provider._start_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value=('rid-future', '', {}, RUNTIME_STATUS_QUEUED)
    )
    get_responses = [
        {'status': 'some_future_state'},
        {'status': 'some_future_state'},
        {
            'status': RUNTIME_STATUS_SCHEDULED,
            'url': 'http://allocated.example.com:32080',
            'port_mappings': {},
        },
    ]
    provider._get_runtime = AsyncMock(side_effect=get_responses)  # type: ignore[method-assign]
    create_mock, _ = _patch_runtime_creation(provider)

    with caplog.at_level(logging.DEBUG, logger='nexus.runtime.providers.gongfeng'):
        runtime = await provider.acquire('runtime-future')

    assert runtime is create_mock.return_value
    assert provider._get_runtime.await_count == 3
    messages = [r.getMessage() for r in caplog.records]
    assert not [m for m in messages if _PROVISIONING_LINE in m]
    assert not [m for m in messages if _PENDING_HEARTBEAT in m]
    assert [m for m in messages if _QUEUED_HEARTBEAT in m]


# ====================== acquire: queue failure via remark ======================


@pytest.mark.asyncio
async def test_acquire_raises_queue_failed_on_queue_timeout_remark(monkeypatch: pytest.MonkeyPatch) -> None:
    """status=stopped + remark='queue_timeout' surfaces as RuntimeQueueFailedError."""
    monkeypatch.setattr(gf, 'QUEUE_POLL_INTERVAL', 0.01)
    provider = _make_provider(queue_wait_max=5)
    provider._start_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value=('rid-failed', '', {}, RUNTIME_STATUS_QUEUED)
    )
    provider._get_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value={'status': RUNTIME_STATUS_STOPPED, 'remark': 'queue_timeout'}
    )
    _patch_runtime_creation(provider)

    with pytest.raises(RuntimeQueueFailedError, match='queue_failure=True'):
        await provider.acquire('runtime-failed')


@pytest.mark.asyncio
async def test_acquire_handles_stop_failed_suffix_in_remark(monkeypatch: pytest.MonkeyPatch) -> None:
    """remark='queue_give_up(stop_failed:...)' must match via startswith, not ==."""
    monkeypatch.setattr(gf, 'QUEUE_POLL_INTERVAL', 0.01)
    provider = _make_provider(queue_wait_max=5)
    provider._start_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value=('rid-give-up', '', {}, RUNTIME_STATUS_QUEUED)
    )
    provider._get_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value={
            'status': RUNTIME_STATUS_STOPPED,
            'remark': 'queue_give_up(stop_failed:connection timeout to k8s)',
        }
    )
    _patch_runtime_creation(provider)

    with pytest.raises(RuntimeQueueFailedError, match='queue_failure=True'):
        await provider.acquire('runtime-give-up')


@pytest.mark.asyncio
async def test_acquire_stopped_with_unrelated_remark_still_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any status=stopped during queue wait is terminal (queue_failure flag tracks remark match)."""
    monkeypatch.setattr(gf, 'QUEUE_POLL_INTERVAL', 0.01)
    provider = _make_provider(queue_wait_max=5)
    provider._start_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value=('rid-stopped', '', {}, RUNTIME_STATUS_QUEUED)
    )
    provider._get_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value={'status': RUNTIME_STATUS_STOPPED, 'remark': 'user_delete'}
    )
    _patch_runtime_creation(provider)

    with pytest.raises(RuntimeQueueFailedError, match='queue_failure=False'):
        await provider.acquire('runtime-stopped-other')


# ====================== acquire: wait_max exhaustion ======================


@pytest.mark.asyncio
async def test_acquire_raises_wait_timeout_and_cancels_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    """queue_wait_max exhausted while still queued → cancel queue entry + RuntimeQueueWaitTimeoutError."""
    monkeypatch.setattr(gf, 'QUEUE_POLL_INTERVAL', 0.05)
    provider = _make_provider(queue_wait_max=0.2)
    provider._start_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value=('rid-timeout', '', {}, RUNTIME_STATUS_QUEUED)
    )
    provider._get_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value={'status': RUNTIME_STATUS_QUEUED}
    )
    stop_mock = AsyncMock()
    provider._stop_runtime = stop_mock  # type: ignore[method-assign]
    _patch_runtime_creation(provider)

    with pytest.raises(RuntimeQueueWaitTimeoutError, match='queue_wait_max=0.2'):
        await provider.acquire('runtime-timeout')

    stop_mock.assert_awaited_once_with(['rid-timeout'])


# ====================== acquire: cancellation cleanup ======================


@pytest.mark.asyncio
async def test_acquire_cancelled_during_queue_calls_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """asyncio cancellation during queue wait must best-effort cancel the queue entry."""
    monkeypatch.setattr(gf, 'QUEUE_POLL_INTERVAL', 0.5)
    provider = _make_provider(queue_wait_max=60)
    provider._start_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value=('rid-cancelled', '', {}, RUNTIME_STATUS_QUEUED)
    )
    provider._get_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value={'status': RUNTIME_STATUS_QUEUED}
    )
    stop_mock = AsyncMock()
    provider._stop_runtime = stop_mock  # type: ignore[method-assign]
    _patch_runtime_creation(provider)

    task = asyncio.create_task(provider.acquire('runtime-cancelled'))
    # Yield so the task can enter the polling loop and reach the sleep.
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    stop_mock.assert_awaited_once_with(['rid-cancelled'])


# ====================== acquire: per-call override ======================


@pytest.mark.asyncio
async def test_acquire_override_queue_wait_max_via_acquire_param(monkeypatch: pytest.MonkeyPatch) -> None:
    """acquire(queue_wait_max=N) overrides the provider default to cap this single call's wait."""
    monkeypatch.setattr(gf, 'QUEUE_POLL_INTERVAL', 0.01)
    provider = _make_provider()  # provider default 0 == no deadline; override caps it
    provider._start_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value=('rid-override', '', {}, RUNTIME_STATUS_QUEUED)
    )
    provider._get_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value={
            'status': RUNTIME_STATUS_SCHEDULED,
            'url': 'http://allocated.example.com:32080',
            'port_mappings': {},
        }
    )
    _patch_runtime_creation(provider)

    runtime = await provider.acquire('runtime-override', queue_wait_max=5)

    assert runtime is provider._create_runtime.return_value
    provider._get_runtime.assert_awaited()


# ====================== watchdog: runtime health probe ======================


@pytest.mark.asyncio
async def test_watchdog_does_not_mark_unhealthy_on_successful_health_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _make_provider(watchdog_poll_interval=0.01)
    response = MagicMock(status_code=200)
    monkeypatch.setattr(gf.http_client, 'get', AsyncMock(return_value=response))
    fake_runtime = MagicMock()
    fake_runtime.base_url = 'http://runtime.example.com'
    fake_runtime.is_unhealthy = False
    fake_runtime.mark_unhealthy = MagicMock()

    task = asyncio.create_task(provider._watchdog_loop('n', fake_runtime, 'rid-q'))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    fake_runtime.mark_unhealthy.assert_not_called()


@pytest.mark.asyncio
async def test_watchdog_resets_streak_on_runtime_paused(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Runtime.Paused is a confirmed live sandbox, same as 2xx: reset the streak, never kill.

    The proxy answers 503 + code 260503 for a pause. That code is shared with other faults, so
    the decision must key off error.reason. Two inconclusive misses, then a pause, must not
    leave the counter at 2 — the next single miss stays under the threshold.
    """
    monkeypatch.setattr(gf, 'WATCHDOG_HEALTH_FAILURE_THRESHOLD', 3)
    provider = _make_provider(watchdog_poll_interval=0.01)
    probes = {'n': 0}
    # Two inconclusive 503s, one pause (must reset), two more 503s, then block.
    # Threshold is 3, so a pause that only increments would already kill on probe 3.
    reasons = [
        'Platform.NodeAgentUnavailable',
        'Platform.NodeAgentUnavailable',
        'Runtime.Paused',
        'Platform.NodeAgentUnavailable',
        'Platform.NodeAgentUnavailable',
    ]

    async def probe(_url: str, **_kwargs: Any) -> MagicMock:
        index = probes['n']
        probes['n'] += 1
        if index >= len(reasons):
            await asyncio.Event().wait()
        response = MagicMock(status_code=503)
        response.json.return_value = {'code': 260503, 'error': {'reason': reasons[index]}}
        return response

    monkeypatch.setattr(gf.http_client, 'get', probe)

    class _FakeRuntime:
        def __init__(self) -> None:
            self.base_url = 'http://runtime.example.com'
            self.is_unhealthy = False

        def mark_unhealthy(self, _reason: str) -> None:
            self.is_unhealthy = True

    fake_runtime = _FakeRuntime()
    with caplog.at_level(logging.INFO, logger='nexus.runtime.providers.gongfeng'):
        task = asyncio.create_task(provider._watchdog_loop('n', fake_runtime, 'rid-paused'))
        for _ in range(50):
            if probes['n'] > len(reasons):
                break
            await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert probes['n'] > len(reasons)
    assert fake_runtime.is_unhealthy is False
    assert any(
        "Sandbox 'n': paused (runtime_id=rid-paused)" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_watchdog_marks_unhealthy_on_confirmed_sandbox_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _make_provider(watchdog_poll_interval=0.01)
    response = MagicMock(status_code=502)
    response.json.return_value = {
        'code': 260422,
        'error': {'reason': 'Process.NonZeroExit', 'exit_code': 137, 'trace_id': 'tid-exit'},
    }
    monkeypatch.setattr(gf.http_client, 'get', AsyncMock(return_value=response))

    fake_runtime = RemoteRuntime(base_url='http://runtime.example.com', auto_connect_ws=False)
    await asyncio.wait_for(provider._watchdog_loop('n', fake_runtime, 'rid-stopped'), timeout=1.0)

    assert fake_runtime.is_unhealthy is True
    unhealthy = fake_runtime.unhealthy_reason or ''
    assert 'Process.NonZeroExit' in unhealthy
    assert 'exit_code=137' in unhealthy
    # The death message must tell the task owner what to do next and how to report it.
    assert 'next step' in unhealthy
    assert 'tid-exit' in unhealthy
    assert 'exit_code=137' in (fake_runtime.unhealthy_reason or '')
    with pytest.raises(RuntimeUnhealthyError) as captured:
        fake_runtime._raise_if_unhealthy()
    assert captured.value.reason == 'Process.NonZeroExit'
    assert captured.value.code == 260422
    assert captured.value.exit_code == 137
    assert captured.value.trace_id == 'tid-exit'


@pytest.mark.asyncio
async def test_watchdog_probe_budget_outlasts_proxy_diagnose_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    """A slow-but-diagnosed 502 must still kill on the FIRST probe, not via the failure counter.

    A dead sandbox makes the proxy burn its full dial budget before it can attach
    reason+exit_code, so the 502 arrives tens of seconds late. If the per-probe budget expires
    first the probe is cancelled client-side and the death degrades to
    WATCHDOG_HEALTH_FAILURE_THRESHOLD × poll_interval. The budget is scaled down here to keep
    the test fast; the real-world invariant (budget > the 30s dial in the sandbox repo's
    proxy/proxy.go) is asserted on the shipped constant at the end.
    """
    shipped_budget = gf.WATCHDOG_HEALTH_TIMEOUT
    probe_budget = 0.2
    monkeypatch.setattr(gf, 'WATCHDOG_HEALTH_TIMEOUT', probe_budget)
    provider = _make_provider(watchdog_poll_interval=0.01)

    seen_timeouts: list[float | None] = []

    async def slow_diagnosed_probe(url: str, **kwargs: Any) -> MagicMock:
        seen_timeouts.append(getattr(kwargs.get('timeout'), 'total', None))
        # Answers after the proxy's (simulated) dial + diagnose, still inside the budget.
        await asyncio.sleep(probe_budget / 4)
        response = MagicMock(status_code=502)
        response.json.return_value = {
            'code': 260422,
            'error': {'reason': 'Process.OOMKilled', 'exit_code': 137},
        }
        return response

    monkeypatch.setattr(gf.http_client, 'get', slow_diagnosed_probe)

    fake_runtime = RemoteRuntime(base_url='http://runtime.example.com', auto_connect_ws=False)
    await asyncio.wait_for(provider._watchdog_loop('n', fake_runtime, 'rid-oom'), timeout=1.0)

    assert fake_runtime.is_unhealthy is True
    assert 'Process.OOMKilled' in (fake_runtime.unhealthy_reason or '')
    assert 'exit_code=137' in (fake_runtime.unhealthy_reason or '')
    # Died on the first probe: the fast path never entered the consecutive-failure counter.
    assert len(seen_timeouts) == 1
    assert seen_timeouts == [probe_budget]
    # The shipped budget must outlast the sandbox proxy's 30s dial plus its diagnose round-trip.
    assert shipped_budget > 30.0
    with pytest.raises(RuntimeUnhealthyError) as captured:
        fake_runtime._raise_if_unhealthy()
    assert captured.value.reason == 'Process.OOMKilled'
    assert captured.value.code == 260422
    assert captured.value.exit_code == 137


@pytest.mark.asyncio
@pytest.mark.parametrize('username', [None, 'fixture-user'])
async def test_platform_token_user_header_is_explicit(monkeypatch: pytest.MonkeyPatch, username: str | None) -> None:
    monkeypatch.delenv('GONGFENG_USERNAME', raising=False)
    provider = _make_provider(username=username)
    response = MagicMock(status_code=200, text='{"code": 0, "data": {}}')
    response.json.return_value = {'code': 0, 'data': {}}
    post = AsyncMock(return_value=response)
    monkeypatch.setattr(gf.http_client, 'post', post)
    await provider._call_api('/sandbox/v3/runtime/list', {'runtime_ids': ['fixture-runtime']})
    headers = post.call_args.kwargs['headers']
    assert headers.get('X-Username') == username


def test_platform_user_can_be_injected_by_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GONGFENG_USERNAME', 'fixture-user')
    assert _make_provider().username == 'fixture-user'
    assert _make_provider(username='explicit-user').username == 'explicit-user'
