import asyncio

from types import SimpleNamespace

import pytest

from nexus.utils.timeout import (
    InferenceTimeoutError,
    TimeBudget,
    await_completion_with_time_budget,
    credit_runner_timeout_for_response,
    response_inference_time,
)


def test_response_inference_time_prefers_provider_value():
    response = SimpleNamespace(provider_specific_fields={'inference_time': 1.5})

    assert response_inference_time(response, 9.0) == pytest.approx(1.5)


def test_response_inference_time_falls_back_when_missing_or_invalid():
    assert response_inference_time(SimpleNamespace(provider_specific_fields=None), 2.0) == pytest.approx(2.0)
    invalid = SimpleNamespace(provider_specific_fields={'inference_time': float('nan')})
    assert response_inference_time(invalid, 3.0) == pytest.approx(3.0)


def test_runner_timeout_is_extended_by_non_inference_time(monkeypatch):
    class TimeoutHandle:
        def __init__(self):
            self.deadline = 100.0

        def when(self):
            return self.deadline

        def reschedule(self, deadline):
            self.deadline = deadline

    handle = TimeoutHandle()
    context = SimpleNamespace(timeout_handle=handle)
    monkeypatch.setattr('nexus.core.context.require_context', lambda: context)
    response = SimpleNamespace(provider_specific_fields={'inference_time': 2.0})

    used = credit_runner_timeout_for_response(response, wall_time=5.0)

    assert used == pytest.approx(2.0)
    assert handle.deadline == pytest.approx(103.0)


def test_runner_timeout_is_unchanged_without_reported_time(monkeypatch):
    class TimeoutHandle:
        def __init__(self):
            self.deadline = 100.0

        def when(self):
            return self.deadline

        def reschedule(self, deadline):
            self.deadline = deadline

    handle = TimeoutHandle()
    context = SimpleNamespace(timeout_handle=handle)
    monkeypatch.setattr('nexus.core.context.require_context', lambda: context)

    used = credit_runner_timeout_for_response(
        SimpleNamespace(provider_specific_fields=None),
        wall_time=5.0,
    )

    assert used == pytest.approx(5.0)
    assert handle.deadline == pytest.approx(100.0)


def test_runner_timeout_does_not_double_credit_previously_paused_time(monkeypatch):
    class TimeoutHandle:
        def __init__(self):
            self.deadline = 100.0

        def when(self):
            return self.deadline

        def reschedule(self, deadline):
            self.deadline = deadline

    handle = TimeoutHandle()
    context = SimpleNamespace(timeout_handle=handle)
    monkeypatch.setattr('nexus.core.context.require_context', lambda: context)
    response = SimpleNamespace(
        provider_specific_fields={
            'inference_time': 2.0,
            '_runner_timeout_credited': 1.5,
        }
    )

    used = credit_runner_timeout_for_response(response, wall_time=5.0)

    assert used == pytest.approx(2.0)
    assert handle.deadline == pytest.approx(101.5)


def test_runner_timeout_credit_is_post_hoc_and_cannot_rescue_an_expired_deadline(monkeypatch):
    class ExpiredTimeoutHandle:
        def when(self):
            return 100.0

        def reschedule(self, deadline):
            raise RuntimeError('Cannot change state of expired Timeout')

    context = SimpleNamespace(timeout_handle=ExpiredTimeoutHandle())
    monkeypatch.setattr('nexus.core.context.require_context', lambda: context)
    response = SimpleNamespace(provider_specific_fields={'inference_time': 2.0})

    assert credit_runner_timeout_for_response(response, wall_time=5.0) == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_completion_budget_uses_reported_time_after_success():
    budget = TimeBudget(0.1)

    async def completion():
        await asyncio.sleep(0.02)
        return SimpleNamespace(provider_specific_fields={'inference_time': 0.01})

    await await_completion_with_time_budget(budget, completion)

    assert budget.elapsed == pytest.approx(0.01)
    assert budget.remaining() == pytest.approx(0.09)


@pytest.mark.asyncio
async def test_completion_budget_expires_post_hoc_when_reported_time_uses_the_budget():
    budget = TimeBudget(0.01)

    async def completion():
        return SimpleNamespace(provider_specific_fields={'inference_time': 0.01})

    with pytest.raises(InferenceTimeoutError):
        await await_completion_with_time_budget(budget, completion)

    assert budget.elapsed == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_completion_budget_times_out_a_call_that_does_not_return():
    budget = TimeBudget(0.01)

    async def completion():
        await asyncio.Event().wait()

    with pytest.raises(InferenceTimeoutError):
        await await_completion_with_time_budget(budget, completion)

    assert budget.exhausted()


@pytest.mark.asyncio
async def test_completion_budget_charges_wall_time_when_call_fails(monkeypatch):
    budget = TimeBudget(10.0)
    timestamps = iter([100.0, 103.0])
    monkeypatch.setattr('nexus.utils.timeout.time', SimpleNamespace(monotonic=lambda: next(timestamps)))

    async def completion():
        raise RuntimeError('provider failed')

    with pytest.raises(RuntimeError, match='provider failed'):
        await await_completion_with_time_budget(budget, completion)

    assert budget.elapsed == pytest.approx(3.0)
    assert budget.remaining() == pytest.approx(7.0)


@pytest.mark.asyncio
async def test_completion_budget_preserves_provider_timeout(monkeypatch):
    budget = TimeBudget(10.0)
    timestamps = iter([100.0, 102.0])
    monkeypatch.setattr('nexus.utils.timeout.time', SimpleNamespace(monotonic=lambda: next(timestamps)))
    provider_timeout = TimeoutError('provider request timed out')

    async def completion():
        raise provider_timeout

    with pytest.raises(TimeoutError, match='provider request timed out') as exc_info:
        await await_completion_with_time_budget(budget, completion)

    assert exc_info.value is provider_timeout
    assert not isinstance(exc_info.value, InferenceTimeoutError)
    assert budget.elapsed == pytest.approx(2.0)
    assert not budget.exhausted()
