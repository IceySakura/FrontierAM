import asyncio
import time

from contextvars import Context
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.core.context import ExecutionContext
from nexus.extensions.openai_clients import hytuner
from nexus.extensions.openai_clients.hytuner import capture_hytuner_execution_timeouts
from nexus.utils import timeout as timeout_utils
from nexus.utils.timeout import (
    TimeBudget,
    async_timeout,
    await_completion_with_time_budget,
    credit_runner_timeout_for_response,
    with_timeout,
)


def test_capture_without_execution_context_returns_none() -> None:
    assert capture_hytuner_execution_timeouts('standalone') is None
    assert timeout_utils.capture_execution_timeouts('standalone') is None


@pytest.mark.parametrize('timeout_sec', [None, 0.5])
async def test_with_timeout_preserves_none_context_manager_result(timeout_sec) -> None:
    async with with_timeout(timeout_sec) as result:
        assert result is None


async def test_generic_capture_pauses_inference_without_a_nexus_context() -> None:
    async with with_timeout(0.1):
        inference = timeout_utils._ACTIVE_TIMEOUT_SCOPES.get()[-1]
        assert capture_hytuner_execution_timeouts('inference-only') is None
        controller = timeout_utils.capture_execution_timeouts('inference-only')
        await controller.pause()
        assert inference.handle.when() is None
        await asyncio.sleep(0.15)
        await controller.resume()
        assert inference.handle.when() is not None


@pytest.mark.parametrize('provider_error', [TimeoutError('provider timeout'), RuntimeError('provider failure')])
async def test_unbounded_budget_preserves_provider_exception_object_and_type(monkeypatch, provider_error) -> None:
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        timeout_utils, 'time', SimpleNamespace(monotonic=lambda: clock.now, perf_counter=lambda: clock.now)
    )
    budget = TimeBudget(None)

    async def completion():
        clock.now = 3.0
        raise provider_error

    with pytest.raises(type(provider_error)) as raised:
        await await_completion_with_time_budget(budget, completion)
    assert raised.value is provider_error
    assert budget.elapsed == pytest.approx(3.0)


async def test_capture_propagates_errors_in_an_existing_context(monkeypatch) -> None:
    context = ExecutionContext()

    def fail_context_lookup():
        raise RuntimeError('context integration failed')

    async with context:
        monkeypatch.setattr(hytuner, 'require_context', fail_context_lookup)
        with pytest.raises(RuntimeError, match='context integration failed'):
            capture_hytuner_execution_timeouts('broken')


async def test_runner_only_controller_leaves_inference_deadline_armed() -> None:
    context = ExecutionContext()
    async with context:
        async with async_timeout(1.0) as runner:
            context.timeout_handle = runner
            async with with_timeout(1.0):
                inference = timeout_utils._ACTIVE_TIMEOUT_SCOPES.get()[-1]
                deadline = inference.handle.when()
                controller = capture_hytuner_execution_timeouts('runner-only', include_inference=False)
                await controller.pause()
                assert runner.when() is None
                assert inference.handle.when() == deadline
                await controller.resume()
                assert inference.handle.when() == deadline


async def test_captured_timeouts_can_be_controlled_without_an_active_context() -> None:
    context = ExecutionContext()
    async with context:
        async with async_timeout(0.1) as runner:
            context.timeout_handle = runner
            async with with_timeout(0.1):
                inference = timeout_utils._ACTIVE_TIMEOUT_SCOPES.get()[-1]
                controller = capture_hytuner_execution_timeouts('captured')
                assert controller is not None
                control = Context().run(asyncio.create_task, _pause_and_resume(controller))
                await control
                assert runner.when() is not None
                assert inference.handle.when() is not None


async def test_tracking_upgrade_restores_default_first_lease_from_a_context_free_controller() -> None:
    runtime = SimpleNamespace(default_venv_id='worker', invoke=AsyncMock(return_value=True), drain_trace=AsyncMock())
    context = ExecutionContext(runtime_command_timeout_id='mixed-command')
    context._runtimes['main'] = runtime
    async with context:
        async with async_timeout(1.0) as runner:
            context.timeout_handle = runner
            group = hytuner._AbortTimeoutPauseGroup()
            context.runtime_shared.set(hytuner._ABORT_TIMEOUT_PAUSE_GROUP_KEY, group)
            await group.acquire('default-first')
            controller = capture_hytuner_execution_timeouts('tracking-second', track_shared_pause=True)
            await controller.pause()
            await group.release('default-first')
            assert runner.when() is None
            resume_task = Context().run(asyncio.create_task, controller.resume())
            await resume_task
            assert runner.when() is not None
    assert [call.args[0].rsplit('::', 1)[-1] for call in runtime.invoke.await_args_list] == [
        'pause_command_timeout',
        'resume_command_timeout',
    ]


async def _pause_and_resume(controller) -> None:
    await controller.pause()
    await asyncio.sleep(0.15)
    await controller.resume()


async def test_nested_leases_preserve_remaining_runner_and_inference_budgets() -> None:
    context = ExecutionContext()
    async with context:
        async with async_timeout(0.1) as runner:
            context.timeout_handle = runner
            async with with_timeout(0.1):
                inference = timeout_utils._ACTIVE_TIMEOUT_SCOPES.get()[-1]
                controller = capture_hytuner_execution_timeouts('nested')
                assert controller is not None
                loop = asyncio.get_running_loop()
                remaining = (runner.when() - loop.time(), inference.handle.when() - loop.time())
                await controller.pause()
                await controller.pause()
                await asyncio.sleep(0.15)
                await controller.resume()
                assert runner.when() is None
                assert inference.handle.when() is None
                await controller.resume()
                restored = (runner.when() - loop.time(), inference.handle.when() - loop.time())
                assert restored == pytest.approx(remaining, abs=0.01)
                assert inference.holders == 0
                with pytest.raises(RuntimeError, match='underflow'):
                    await controller.resume()


async def test_parallel_controllers_share_runner_and_runtime_until_last_resume() -> None:
    runtime = SimpleNamespace(default_venv_id='worker', invoke=AsyncMock(return_value=True), drain_trace=AsyncMock())
    context = ExecutionContext(runtime_command_timeout_id='command')
    context._runtimes['main'] = runtime
    async with context:
        async with async_timeout(0.1) as runner:
            context.timeout_handle = runner
            first = capture_hytuner_execution_timeouts('first')
            second = capture_hytuner_execution_timeouts('second')
            await first.pause()
            await second.pause()
            await first.resume()
            assert runner.when() is None
            await asyncio.sleep(0.15)
            await second.resume()
            assert runner.when() is not None
    assert [call.args[0].rsplit('::', 1)[-1] for call in runtime.invoke.await_args_list] == [
        'pause_command_timeout',
        'resume_command_timeout',
    ]


async def test_shared_group_counts_overlapping_and_repeated_pause_union(monkeypatch) -> None:
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(time, 'perf_counter', lambda: clock.now)
    context = ExecutionContext()
    async with context:
        first = capture_hytuner_execution_timeouts('first', track_shared_pause=True)
        second = capture_hytuner_execution_timeouts('second', track_shared_pause=True)
        clock.now = 1.0
        await first.pause()
        clock.now = 2.0
        await second.pause()
        clock.now = 4.0
        await first.resume()
        clock.now = 6.0
        await second.resume()
        clock.now = 10.0
        await first.pause()
        await first.pause()
        clock.now = 12.0
        await first.resume()
        await first.resume()
        group = context.runtime_shared.get_optional(hytuner._ABORT_TIMEOUT_PAUSE_GROUP_KEY)
        assert group.get_paused_time(0.0) == pytest.approx(7.0)


async def test_failed_completion_excludes_paused_time_from_budget() -> None:
    context = ExecutionContext()
    budget = TimeBudget(0.1)

    async def completion():
        controller = timeout_utils.capture_execution_timeouts('failed')
        await controller.pause()
        try:
            await asyncio.sleep(0.15)
        finally:
            await controller.resume()
        raise RuntimeError('backend failed')

    async with context:
        with pytest.raises(RuntimeError, match='backend failed'):
            await await_completion_with_time_budget(budget, completion)
    assert budget.remaining() > 0.04


async def test_later_unrelated_timeout_is_not_captured() -> None:
    context = ExecutionContext()
    async with context:
        controller = capture_hytuner_execution_timeouts('generation')
        await controller.pause()
        try:
            with pytest.raises(TimeoutError):
                async with with_timeout(0.01):
                    await asyncio.Event().wait()
        finally:
            await controller.resume()


async def test_response_credit_deduplicates_shared_pause_union_while_runner_is_paused(monkeypatch) -> None:
    initial_deadline = asyncio.get_running_loop().time() + 100.0

    class TimeoutHandle:
        deadline = initial_deadline

        def when(self):
            return self.deadline

        def reschedule(self, value):
            self.deadline = value

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(time, 'perf_counter', lambda: clock.now)
    context = ExecutionContext(timeout_handle=TimeoutHandle())
    response = SimpleNamespace(provider_specific_fields={'inference_time': 5.0})
    async with context:
        first = capture_hytuner_execution_timeouts('first', track_shared_pause=True)
        second = capture_hytuner_execution_timeouts('second', track_shared_pause=True)
        clock.now = 10.0
        await first.pause()
        await second.pause()
        clock.now = 20.0
        await second.resume()
        assert context.timeout_handle.when() is None
        credit_runner_timeout_for_response(response, wall_time=20.0, started_at=0.0)
        assert response.provider_specific_fields['_runner_timeout_credited'] == pytest.approx(10.0)
        clock.now = 30.0
        await first.resume()
        assert context.timeout_handle.when() == pytest.approx(initial_deadline + 25.0)


@pytest.mark.parametrize('operation', ['pause', 'resume'])
async def test_cancellation_settles_remote_update_and_cleans_lease(operation) -> None:
    rpc_started = asyncio.Event()
    rpc_allowed = asyncio.Event()

    async def invoke(target, *args, **kwargs):
        if target.endswith(f'::{operation}_command_timeout'):
            rpc_started.set()
            await rpc_allowed.wait()
        return True

    runtime = SimpleNamespace(default_venv_id='worker', invoke=AsyncMock(side_effect=invoke), drain_trace=AsyncMock())
    context = ExecutionContext(runtime_command_timeout_id='command')
    context._runtimes['main'] = runtime
    async with context:
        async with async_timeout(1.0) as runner:
            context.timeout_handle = runner
            async with with_timeout(1.0):
                inference = timeout_utils._ACTIVE_TIMEOUT_SCOPES.get()[-1]
                controller = capture_hytuner_execution_timeouts('cancel')
                if operation == 'resume':
                    await controller.pause()
                pending = asyncio.create_task(getattr(controller, operation)())
                try:
                    await asyncio.wait_for(rpc_started.wait(), timeout=1.0)
                    for _ in range(3):
                        pending.cancel()
                        await asyncio.sleep(0)
                        assert not pending.done()
                        assert runner.when() is None
                        assert inference.handle.when() is None
                    rpc_allowed.set()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                    assert runner.when() is not None
                    assert inference.handle.when() is not None
                    assert inference.holders == 0
                    group = context.runtime_shared.get_optional(hytuner._ABORT_TIMEOUT_PAUSE_GROUP_KEY)
                    assert group._holder_count == 0
                finally:
                    rpc_allowed.set()
                    await asyncio.gather(pending, return_exceptions=True)
    assert [call.args[0].rsplit('::', 1)[-1] for call in runtime.invoke.await_args_list] == [
        'pause_command_timeout',
        'resume_command_timeout',
    ]


async def test_failed_runtime_pause_restores_local_scopes() -> None:
    runtime = SimpleNamespace(
        default_venv_id='worker', invoke=AsyncMock(side_effect=[False, True]), drain_trace=AsyncMock()
    )
    context = ExecutionContext(runtime_command_timeout_id='missing')
    context._runtimes['main'] = runtime
    async with context:
        async with async_timeout(1.0) as runner:
            context.timeout_handle = runner
            async with with_timeout(1.0):
                inference = timeout_utils._ACTIVE_TIMEOUT_SCOPES.get()[-1]
                controller = capture_hytuner_execution_timeouts('error')
                with pytest.raises(RuntimeError, match='missing'):
                    await controller.pause()
                assert runner.when() is not None
                assert inference.handle.when() is not None
                assert inference.holders == 0


async def test_expired_inference_deadline_is_not_revived() -> None:
    context = ExecutionContext()
    async with context:
        with pytest.raises(TimeoutError):  # noqa: PT012
            async with with_timeout(10.0):
                scope = timeout_utils._ACTIVE_TIMEOUT_SCOPES.get()[-1]
                deadline = asyncio.get_running_loop().time() - 1.0
                scope.handle.reschedule(deadline)
                controller = capture_hytuner_execution_timeouts('expired', track_shared_pause=True)
                await controller.pause()
                assert scope.handle.when() == deadline
                assert scope.holders == 0
                await controller.resume()


async def test_expired_runner_deadline_is_not_revived() -> None:
    context = ExecutionContext()
    async with context:
        with pytest.raises(TimeoutError):  # noqa: PT012
            async with async_timeout(10.0) as runner:
                context.timeout_handle = runner
                deadline = asyncio.get_running_loop().time() - 1.0
                runner.reschedule(deadline)
                controller = capture_hytuner_execution_timeouts('expired', track_shared_pause=True)
                await controller.pause()
                assert runner.when() == deadline
                await controller.resume()


@pytest.mark.parametrize('credit_while_paused', [False, True])
async def test_default_abort_preserves_response_credit_without_shared_pause_accounting(
    monkeypatch, credit_while_paused
):
    class TimeoutHandle:
        deadline = 1000.0

        def when(self):
            return self.deadline

        def reschedule(self, deadline):
            self.deadline = deadline

    clock = SimpleNamespace(now=1.0)
    monkeypatch.setattr(time, 'perf_counter', lambda: clock.now)
    context = ExecutionContext(timeout_handle=TimeoutHandle())
    abort_event = asyncio.Event()
    response = SimpleNamespace(provider_specific_fields={'inference_time': 2.0})
    async with context:
        waiting = asyncio.create_task(hytuner._wait_for_abort_resume(abort_event, 'default-accounting'))
        try:
            for _ in range(1000):
                if context.timeout_handle.when() is None:
                    break
                await asyncio.sleep(0)
            else:
                pytest.fail('Default abort wait did not pause the runner')
            clock.now = 4.0
            if not credit_while_paused:
                abort_event.set()
                await waiting
            used = credit_runner_timeout_for_response(response, wall_time=5.0, started_at=1.0)
            assert used == pytest.approx(2.0)
            assert response.provider_specific_fields == {'inference_time': 2.0}
            if credit_while_paused:
                assert context.timeout_handle.when() is None
                clock.now = 5.0
                abort_event.set()
                await waiting
            assert context.timeout_handle.when() == pytest.approx(1004.0 if credit_while_paused else 1006.0)
        finally:
            abort_event.set()
            await asyncio.gather(waiting, return_exceptions=True)


async def test_default_abort_park_leaves_inference_deadline_to_expire() -> None:
    context = ExecutionContext()
    abort_event = asyncio.Event()
    waiting = None
    async with context:
        async with async_timeout(1.0) as runner:
            context.timeout_handle = runner
            try:
                with pytest.raises(TimeoutError):  # noqa: PT012
                    async with with_timeout(0.05):
                        inference = timeout_utils._ACTIVE_TIMEOUT_SCOPES.get()[-1]
                        deadline = inference.handle.when()
                        waiting = asyncio.create_task(hytuner._wait_for_abort_resume(abort_event, 'default-inference'))
                        for _ in range(1000):
                            if context.timeout_handle.when() is None:
                                break
                            await asyncio.sleep(0)
                        else:
                            pytest.fail('Default abort wait did not pause the runner')
                        assert runner.when() is None
                        assert inference.handle.when() == deadline
                        await asyncio.Event().wait()
            finally:
                abort_event.set()
                if waiting is not None:
                    await waiting
            assert runner.when() is not None
