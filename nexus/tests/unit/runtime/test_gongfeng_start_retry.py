"""Tests for GongfengRuntimeProvider sandbox creation and its opt-in retry (start_retry).

The sandbox error contract (https://iwiki.woa.com/p/4041018198) allows a retriable start
failure to be recreated at most once, no sooner than 30s later, and the backend rejects a
reused runtime_id with 260409 even after the earlier attempt failed. Retry is off by default
because higher layers (batch retry_failed, task loops) may already recreate the sandbox.
These tests stub ``_start_runtime`` / ``_stop_runtime`` and replace ``asyncio.sleep`` so no
network or real waiting is involved.
"""

from __future__ import annotations

import asyncio

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.runtime.providers import gongfeng as gf
from nexus.runtime.providers.gongfeng import (
    RUNTIME_STATUS_SCHEDULED,
    START_RETRY_JITTER,
    START_RETRY_MAX,
    START_RETRY_MIN_DELAY,
    GongfengRuntimeProvider,
    SandboxApiError,
)

SANDBOX_URL = 'http://sandbox.local:8080'
DO_NOT_RETRY_NOTICE = 'do not retry it at a higher layer'


def _make_provider(**overrides: Any) -> GongfengRuntimeProvider:
    kwargs: dict[str, Any] = {
        'api_base_url': 'http://gongfeng.example.com',
        'token': 'unit-test-token',
        'image': 'unit-test/image:latest',
        'print_port_mappings': False,
    }
    kwargs.update(overrides)
    provider = GongfengRuntimeProvider(**kwargs)
    fake_runtime = MagicMock()
    fake_runtime.port_mappings = {}
    provider._create_runtime = MagicMock(return_value=fake_runtime)  # type: ignore[method-assign]
    provider._start_watchdog = MagicMock()  # type: ignore[method-assign]
    return provider


def _api_error(code: int, reason: str = '', msg: str = 'failed to create sandbox: boom') -> SandboxApiError:
    return SandboxApiError(
        f'Gongfeng API error (code={code}, reason={reason}): {msg}', code=code, reason=reason, trace_id='tid-1'
    )


class _FakeBackend:
    """Mimic the backend's runtime_id bookkeeping and record the call order.

    Every start records its runtime_id first, exactly like the backend's runtimes row; a
    second start with a recorded id answers 260409. ``failures`` is consumed one per start.
    """

    def __init__(self, failures: list[BaseException]) -> None:
        self.failures = list(failures)
        self.recorded: set[str] = set()
        self.events: list[tuple[str, Any]] = []

    async def start(self, **kwargs: Any) -> tuple[str, str, dict[int, str], str]:
        runtime_id = kwargs['runtime_id']
        self.events.append(('start', runtime_id))
        if runtime_id in self.recorded:
            raise _api_error(260409, 'Runtime.AlreadyExists', 'runtime already exists')
        self.recorded.add(runtime_id)
        if self.failures:
            raise self.failures.pop(0)
        return runtime_id, SANDBOX_URL, {}, RUNTIME_STATUS_SCHEDULED

    async def stop(self, runtime_ids: list[str]) -> None:
        self.events.append(('stop', list(runtime_ids)))

    @property
    def started_ids(self) -> list[str]:
        return [value for kind, value in self.events if kind == 'start']


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def _record_sleep(delay: float, *_: Any, **__: Any) -> None:
        recorded.append(delay)

    monkeypatch.setattr(gf.asyncio, 'sleep', _record_sleep)
    return recorded


def _wire(provider: GongfengRuntimeProvider, backend: _FakeBackend) -> None:
    provider._start_runtime = AsyncMock(side_effect=backend.start)  # type: ignore[method-assign]
    provider._stop_runtime = AsyncMock(side_effect=backend.stop)  # type: ignore[method-assign]


# ====================== default: start once, no retry ======================


@pytest.mark.asyncio
async def test_start_retry_is_off_by_default(sleeps: list[float]) -> None:
    provider = _make_provider()
    failure = _api_error(260503, 'Platform.RuntimeError')
    backend = _FakeBackend([failure])
    _wire(provider, backend)

    with pytest.raises(SandboxApiError) as excinfo:
        await provider.acquire('main')

    assert excinfo.value is failure
    assert DO_NOT_RETRY_NOTICE not in str(excinfo.value)
    (runtime_id,) = backend.started_ids
    assert backend.events == [('start', runtime_id), ('stop', [runtime_id])]
    assert sleeps == []


@pytest.mark.asyncio
async def test_failed_first_start_surfaces_the_real_error_not_already_exists(sleeps: list[float]) -> None:
    """Drive the real ``_start_runtime`` through a backend that keeps failed runtime_ids.

    Before the fix the same payload was resent and the caller only saw 260409. With retry off
    the caller now sees the original failure and can decide; a higher-layer acquire() mints a
    fresh runtime_id, so its own retry no longer collides either.
    """
    provider = _make_provider()
    recorded: list[str] = []
    stopped: list[str] = []

    async def _fake_call_api(endpoint: str, payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        if endpoint == '/sandbox/v3/runtime/stop':
            stopped.extend(payload['runtime_ids'])
            return {'code': 0}
        assert endpoint == '/sandbox/v3/runtime/start'
        runtime_id = payload['runtime_id']
        if runtime_id in recorded:
            raise _api_error(260409, 'Runtime.AlreadyExists', 'runtime already exists')
        recorded.append(runtime_id)
        if len(recorded) == 1:
            raise _api_error(260503, 'Platform.RuntimeError', 'failed to create sandbox: CreateContainerError')
        return {'code': 0, 'data': {'runtime_id': runtime_id, 'url': SANDBOX_URL, 'port_mappings': {}}}

    provider._call_api = _fake_call_api  # type: ignore[method-assign]

    with pytest.raises(SandboxApiError) as excinfo:
        await provider.acquire('main')
    assert excinfo.value.code == 260503
    assert stopped == [recorded[0]]

    await provider.acquire('main')
    assert len(recorded) == 2
    assert recorded[0] != recorded[1]
    assert provider.get_runtime_id('main') == recorded[1]


@pytest.mark.asyncio
async def test_start_runtime_posts_once_even_for_retriable_code() -> None:
    provider = _make_provider(start_retry=True)
    failure = _api_error(260503, 'Platform.RuntimeError')
    provider._call_api = AsyncMock(side_effect=failure)  # type: ignore[method-assign]

    with pytest.raises(SandboxApiError) as excinfo:
        await provider._start_runtime(image='img', runtime_id='nexus-' + '0' * 32)

    assert excinfo.value is failure
    provider._call_api.assert_awaited_once()


# ====================== start_retry enabled ======================


@pytest.mark.asyncio
async def test_retriable_start_failure_recreates_under_fresh_runtime_id(sleeps: list[float]) -> None:
    provider = _make_provider(start_retry=True)
    backend = _FakeBackend([_api_error(260503, 'Platform.RuntimeError')])
    _wire(provider, backend)

    await provider.acquire('main')

    first_id, second_id = backend.started_ids
    assert first_id != second_id
    assert backend.events == [('start', first_id), ('stop', [first_id]), ('start', second_id)]
    assert provider.get_runtime_id('main') == second_id
    assert len(sleeps) == 1
    assert START_RETRY_MIN_DELAY <= sleeps[0] <= START_RETRY_MIN_DELAY + START_RETRY_JITTER


@pytest.mark.asyncio
async def test_caller_owned_runtime_id_is_left_alone_but_the_recreation_id_is_stopped(
    sleeps: list[float],
) -> None:
    """A caller-supplied id stays the caller's to clean up; the id nexus mints on retry does not.

    Skipping the cleanup for the recreation id would leak a sandbox nexus itself asked for.
    """
    provider = _make_provider(start_retry=True)
    failures = [_api_error(260503, 'Platform.RuntimeError'), _api_error(260503, 'Platform.RuntimeError')]
    backend = _FakeBackend(failures)
    _wire(provider, backend)

    with pytest.raises(SandboxApiError):
        await provider.acquire('main', runtime_id='caller-owned-runtime')

    first_id, second_id = backend.started_ids
    assert first_id == 'caller-owned-runtime'
    assert second_id.startswith('nexus-')
    assert backend.events == [
        ('start', 'caller-owned-runtime'),
        ('start', second_id),
        ('stop', [second_id]),
    ]


@pytest.mark.parametrize('value', [True, 'true', '1'])
@pytest.mark.asyncio
async def test_start_retry_can_be_enabled_per_acquire(value: Any, sleeps: list[float]) -> None:
    provider = _make_provider()
    backend = _FakeBackend([_api_error(260503, 'Platform.RuntimeError')])
    _wire(provider, backend)

    await provider.acquire('main', start_retry=value)

    assert len(backend.started_ids) == 2


@pytest.mark.asyncio
async def test_start_retry_can_be_disabled_per_acquire(sleeps: list[float]) -> None:
    provider = _make_provider(start_retry=True)
    backend = _FakeBackend([_api_error(260503, 'Platform.RuntimeError')])
    _wire(provider, backend)

    with pytest.raises(SandboxApiError):
        await provider.acquire('main', start_retry=False)

    assert len(backend.started_ids) == 1


def test_start_retry_rejects_values_that_are_not_booleans() -> None:
    with pytest.raises(ValueError, match='start_retry'):
        _make_provider(start_retry='sometimes')


@pytest.mark.parametrize(
    ('code', 'reason'),
    [
        (260503, 'Platform.RuntimeError'),
        (260504, 'Runtime.StartupTimeout'),
        (260429, 'Quota.RateLimited'),
    ],
)
@pytest.mark.asyncio
async def test_contract_retriable_codes_recreate_once(code: int, reason: str, sleeps: list[float]) -> None:
    provider = _make_provider(start_retry=True)
    backend = _FakeBackend([_api_error(code, reason)])
    _wire(provider, backend)

    await provider.acquire('main')

    assert len(backend.started_ids) == 2


@pytest.mark.asyncio
async def test_failure_after_recreation_tells_callers_not_to_retry(
    sleeps: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    provider = _make_provider(start_retry=True)
    first = _api_error(260503, 'Platform.RuntimeError', 'first')
    second = _api_error(260503, 'Schedule.InsufficientResource', 'second')
    backend = _FakeBackend([first, second])
    _wire(provider, backend)

    with caplog.at_level('WARNING', logger='nexus.runtime.providers.gongfeng'):
        with pytest.raises(SandboxApiError) as excinfo:
            await provider.acquire('main')

    raised = excinfo.value
    assert raised.__cause__ is second
    assert (raised.code, raised.reason, raised.trace_id) == (second.code, second.reason, second.trace_id)
    assert str(raised).startswith(str(second))
    assert DO_NOT_RETRY_NOTICE in str(raised)
    assert any('do not retry this failure at a higher layer' in r.getMessage() for r in caplog.records)

    assert len(backend.started_ids) == START_RETRY_MAX + 1
    first_id, second_id = backend.started_ids
    assert backend.events == [
        ('start', first_id),
        ('stop', [first_id]),
        ('start', second_id),
        ('stop', [second_id]),
    ]
    assert len(sleeps) == START_RETRY_MAX
    assert provider.get_runtime_id('main') is None


@pytest.mark.asyncio
async def test_non_retriable_failure_after_recreation_also_carries_the_notice(sleeps: list[float]) -> None:
    provider = _make_provider(start_retry=True)
    second = _api_error(260400, 'Request.InvalidParameter', 'bad image')
    backend = _FakeBackend([_api_error(260503, 'Platform.RuntimeError'), second])
    _wire(provider, backend)

    with pytest.raises(SandboxApiError) as excinfo:
        await provider.acquire('main')

    assert excinfo.value.code == 260400
    assert DO_NOT_RETRY_NOTICE in str(excinfo.value)
    assert len(backend.started_ids) == 2


@pytest.mark.parametrize(
    'failure',
    [
        _api_error(260500, 'Platform.UnknownError'),
        _api_error(260400, 'Request.InvalidParameter'),
        _api_error(260409, 'Runtime.AlreadyExists'),
        _api_error(260412, 'Snapshot.NotReady'),
        _api_error(260422, 'Process.OOMKilled'),
        _api_error(260429, 'Quota.Exceeded'),
        _api_error(260499, 'Request.Canceled'),
        _api_error(260777, ''),
        RuntimeError('API request timeout for /sandbox/v3/runtime/start: failed to create sandbox'),
    ],
    ids=lambda exc: getattr(exc, 'reason', '') or type(exc).__name__,
)
@pytest.mark.asyncio
async def test_non_retriable_failures_fail_after_one_start(failure: BaseException, sleeps: list[float]) -> None:
    provider = _make_provider(start_retry=True)
    backend = _FakeBackend([failure])
    _wire(provider, backend)

    with pytest.raises(type(failure)) as excinfo:
        await provider.acquire('main')

    assert excinfo.value is failure
    (runtime_id,) = backend.started_ids
    assert backend.events == [('start', runtime_id), ('stop', [runtime_id])]
    assert sleeps == []


@pytest.mark.asyncio
async def test_cancel_during_retry_backoff_does_not_start_again(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _make_provider(start_retry=True)
    backend = _FakeBackend([_api_error(260503, 'Platform.RuntimeError')])
    _wire(provider, backend)
    backoff_entered = asyncio.Event()

    async def _block_sleep(*_: Any, **__: Any) -> None:
        backoff_entered.set()
        await asyncio.Future()

    monkeypatch.setattr(gf.asyncio, 'sleep', _block_sleep)

    task = asyncio.create_task(provider.acquire('main'))
    await backoff_entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    (runtime_id,) = backend.started_ids
    assert backend.events == [('start', runtime_id), ('stop', [runtime_id])]
