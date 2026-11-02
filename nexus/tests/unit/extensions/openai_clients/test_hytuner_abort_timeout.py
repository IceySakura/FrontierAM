"""Task-deadline handling around the HyTuner abort/resume park.

Regression coverage for a rollout crash: ``asyncio.Timeout.reschedule`` raises
``RuntimeError`` once the deadline has already fired, and the unguarded restore
call let that RuntimeError escape into the agent loop. Escaping there is worse
than a lost deadline -- it reaches ``Timeout.__aexit__`` ahead of the pending
``CancelledError``, so the cancellation is never converted into a
``TimeoutError`` and later surfaces bare, outside the timeout scope.
"""

import asyncio

from types import SimpleNamespace

import pytest

from nexus.core.context import ExecutionContext
from nexus.extensions.openai_clients import hytuner


class _ExpiringTimeoutHandle:
    """Stands in for an ``asyncio.Timeout`` whose deadline has already fired."""

    def __init__(self) -> None:
        self.reschedule_calls: list[float | None] = []

    def when(self) -> float:
        return 1000.0

    def reschedule(self, when: float | None) -> None:
        self.reschedule_calls.append(when)
        raise RuntimeError('Cannot change state of expiring Timeout')


class _PauseRejectingHandle:
    """Wraps a live ``asyncio.Timeout`` that refuses ``reschedule``, as it does once EXPIRING.

    The wrapped deadline stays armed, so the park it fails to suspend really does get
    cancelled by it.
    """

    def __init__(self, timeout: 'asyncio.Timeout') -> None:
        self._timeout = timeout

    def when(self) -> float | None:
        return self._timeout.when()

    def reschedule(self, when: float | None) -> None:
        raise RuntimeError('Cannot change state of expiring Timeout')


@pytest.fixture
def stub_context(monkeypatch):
    context = ExecutionContext()
    monkeypatch.setattr(hytuner, 'require_context', lambda: context)
    return context


def test_pause_reports_nothing_to_restore_when_deadline_already_fired(stub_context):
    stub_context.timeout_handle = _ExpiringTimeoutHandle()

    assert hytuner._pause_task_timeout('req-1') is None


def test_pause_reports_nothing_to_restore_without_a_timeout_handle(stub_context):
    assert hytuner._pause_task_timeout('req-1') is None


def test_restore_is_a_noop_when_the_pause_never_took_effect():
    hytuner._restore_task_timeout(None, 1.0, 'req-1')


def test_restore_survives_a_deadline_that_fired_while_parked():
    handle = _ExpiringTimeoutHandle()

    hytuner._restore_task_timeout(hytuner._PausedDeadline(handle, 1000.0), 2.0, 'req-1')

    assert handle.reschedule_calls == [1002.0]


async def test_park_suspends_the_deadline_and_extends_it_by_the_wait(stub_context):
    """A park longer than the remaining budget must not consume the budget."""
    abort_event = asyncio.Event()
    asyncio.get_running_loop().call_later(0.12, abort_event.set)

    async with asyncio.timeout(0.05) as handle:
        stub_context.timeout_handle = handle
        deadline_before = handle.when()

        wait_duration = await hytuner._wait_for_abort_resume(abort_event, 'req-1')

        assert wait_duration >= 0.12
        assert handle.when() == pytest.approx(deadline_before + wait_duration)
        # The 50 ms budget is still unspent after a 120 ms park.
        await asyncio.sleep(0.01)


async def test_park_over_an_expired_deadline_skips_the_restore(stub_context):
    """The regression itself: a restore paired with a failed pause raised RuntimeError."""
    handle = _ExpiringTimeoutHandle()
    stub_context.timeout_handle = handle
    abort_event = asyncio.Event()
    asyncio.get_running_loop().call_soon(abort_event.set)

    await hytuner._wait_for_abort_resume(abort_event, 'req-1')

    assert handle.reschedule_calls == [None], 'a pause that did not take effect must not be restored'


async def test_a_deadline_that_cannot_be_paused_still_times_the_task_out(stub_context):
    """A failed pause must leave the real deadline to fire and surface as TimeoutError.

    Anything other than a ``CancelledError`` reaching ``Timeout.__aexit__`` leaves the
    cancellation pending, to surface bare outside the scope. That is the crash this
    module guards against, so assert on the exception the scope actually raises.
    """
    abort_event = asyncio.Event()  # never set, so only the deadline can end the park

    async def park_under_a_deadline() -> None:
        async with asyncio.timeout(0.05) as handle:
            stub_context.timeout_handle = _PauseRejectingHandle(handle)
            await hytuner._wait_for_abort_resume(abort_event, 'req-1')

    with pytest.raises(TimeoutError):
        await park_under_a_deadline()


async def test_cancellation_while_parked_restores_the_deadline(monkeypatch):
    """Cancellation must not leave the task running with a suspended deadline."""
    context = ExecutionContext()
    monkeypatch.setattr(hytuner, 'require_context', lambda: context)
    abort_event = asyncio.Event()  # never set, so the park blocks
    observed: dict[str, float | None] = {}

    async def parked() -> None:
        async with asyncio.timeout(10) as handle:
            context.timeout_handle = handle
            observed['before'] = handle.when()
            try:
                await hytuner._wait_for_abort_resume(abort_event, 'req-1')
            finally:
                observed['after'] = handle.when()

    loop = asyncio.get_running_loop()
    started = loop.time()
    task = asyncio.create_task(parked())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Bound the assertion by the time this test actually took rather than by a fixed
    # tolerance, so a loaded machine cannot make it flaky.
    elapsed = loop.time() - started

    assert observed['after'] is not None, 'the suspended deadline must be re-armed'
    assert observed['before'] <= observed['after'] <= observed['before'] + elapsed


def test_token_output_inference_time_prefers_reported_value():
    output = SimpleNamespace(extra_fields={'_inference_time': 1.25})

    assert hytuner._token_output_inference_time(output, 9.0) == pytest.approx(1.25)


def test_token_output_inference_time_falls_back_for_invalid_value():
    output = SimpleNamespace(extra_fields={'_inference_time': 'not-a-number'})

    assert hytuner._token_output_inference_time(output, 2.5) == pytest.approx(2.5)
