"""Tests for hold-after-task logic in single runner."""

import asyncio
import json
import signal
import time

from contextlib import nullcontext
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from pydantic import BaseModel

from nexus.dataset.preparation import TaskFilesMaterialization
from nexus.runner.single import RunHooks, RunInput, RunnerConfig, _hold_sandbox, run
from nexus.runtime.providers.base import RuntimeProvider
from nexus.trace.processors.base import SpanProcessor


class FakeTaskInput(BaseModel):
    pass


async def _fake_task(_input: FakeTaskInput) -> str:
    return 'done'


class FakeRuntime:
    def __init__(self, base_url: str = 'http://10.0.0.1:32000'):
        self.base_url = base_url

    async def ws_disconnect(self) -> None:
        pass


class FakeProvider(RuntimeProvider):
    """Minimal concrete provider for testing hold logic."""

    def __init__(self) -> None:
        super().__init__()
        self.released: list[str] = []

    async def acquire(self, name: str, **params: Any) -> Any:
        runtime = FakeRuntime()
        self._active_runtimes[name] = runtime
        return runtime

    async def _release_impl(self, name: str) -> None:
        self.released.append(name)
        self._active_runtimes.pop(name, None)


@pytest.fixture
def provider() -> FakeProvider:
    p = FakeProvider()
    p._active_runtimes['main'] = FakeRuntime()
    return p


@pytest.fixture
def run_input_with_hold():
    return RunInput(
        task_name='test_task',
        task_config={},
        runtime_provider_name='fake',
        runner=RunnerConfig(hold_after_task=0.1),
    )


@pytest.fixture
def run_input_no_hold():
    return RunInput(
        task_name='test_task',
        task_config={},
        runtime_provider_name='fake',
        runner=RunnerConfig(),
    )


def _run_patches(provider, task_fn=None, processors=None):
    """Return a combined context manager that patches runner dependencies."""
    if task_fn is None:
        task_fn = _fake_task
    if processors is None:
        processors = {}

    class _Patches:
        def __enter__(self):
            self._p1 = patch('nexus.runner.single.task_registry.get', return_value=task_fn)
            self._p2 = patch(
                'nexus.runner.single.runtime_provider_registry.get',
                return_value=lambda **kwargs: provider,
            )
            self._p3 = patch('nexus.runner.single._create_processors', return_value=processors)
            self._p1.start()
            self._p2.start()
            self._p3.start()
            return self

        def __exit__(self, *args):
            self._p3.stop()
            self._p2.stop()
            self._p1.stop()

    return _Patches()


class TestRunnerConfigHold:
    def test_hold_after_task_default_none(self) -> None:
        config = RunnerConfig()
        assert config.hold_after_task is None

    def test_hold_after_task_accepts_float(self) -> None:
        config = RunnerConfig(hold_after_task=1800.0)
        assert config.hold_after_task == 1800.0

    def test_hold_after_task_accepts_int(self) -> None:
        config = RunnerConfig(hold_after_task=60)
        assert config.hold_after_task == 60

    def test_early_output_path_default_none(self) -> None:
        config = RunnerConfig()
        assert config.early_output_path is None

    def test_hold_signal_handlers_default_enabled(self) -> None:
        config = RunnerConfig()
        assert config.hold_signal_handlers is True


class TestHoldSuppressLifecycle:
    @pytest.mark.asyncio
    async def test_suppress_enabled_before_task(self, provider, run_input_with_hold) -> None:
        suppress_states: list[bool] = []

        async def tracking_task(_input: FakeTaskInput) -> str:
            suppress_states.append(provider._suppress_release)
            return 'done'

        with _run_patches(provider, task_fn=tracking_task):
            await run(run_input_with_hold)

        assert suppress_states == [True]

    @pytest.mark.asyncio
    async def test_suppress_cleared_after_hold(self, provider, run_input_with_hold) -> None:
        with _run_patches(provider):
            await run(run_input_with_hold)

        assert provider._suppress_release is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize('stage', ['host_files', 'ready', 'processor'])
    @pytest.mark.parametrize('report_setup_error', [False, True])
    async def test_setup_failure_releases_runtime_before_reporting(
        self, provider, run_input_with_hold, stage, report_setup_error
    ) -> None:
        failure = ValueError('setup failed')
        hooks = RunHooks()
        if report_setup_error:

            def report(_run_input, _error, _output):
                assert provider._suppress_release is False
                assert provider.released == ['main']

            hooks.on_initialization_error = report
        if stage == 'host_files':
            run_input_with_hold.runner.task_files_materialization = TaskFilesMaterialization(mode='cwd', target='host')
        elif stage == 'ready':
            hooks.on_context_ready = AsyncMock(side_effect=failure)
        processor_scope = AsyncMock()
        processor_scope.__aenter__.side_effect = failure
        processor_patch = (
            patch('nexus.runner.single._processor_scope', return_value=processor_scope)
            if stage == 'processor'
            else nullcontext()
        )
        with (
            _run_patches(provider),
            patch('nexus.runner.single._hold_sandbox', new_callable=AsyncMock) as hold,
            processor_patch,
        ):
            if not report_setup_error and stage != 'host_files':
                with pytest.raises(ValueError, match='setup failed'):
                    await run(run_input_with_hold, hooks=hooks)
            else:
                output = await run(run_input_with_hold, hooks=hooks)
                assert not output.success
            if report_setup_error or stage != 'host_files':
                hold.assert_not_awaited()

        assert provider._suppress_release is False
        assert provider.released == ['main']
        assert not provider.get_active_runtimes()

    @pytest.mark.asyncio
    async def test_task_cancellation_releases_held_runtime(self, provider, run_input_with_hold) -> None:
        started = asyncio.Event()

        async def task(_input: FakeTaskInput) -> str:
            started.set()
            await asyncio.Event().wait()
            return 'unreachable'

        def close(_ctx):
            assert provider._suppress_release is False
            raise RuntimeError('closing failed')

        hooks = RunHooks(on_context_closing=close)
        with _run_patches(provider, task_fn=task):
            job = asyncio.create_task(run(run_input_with_hold, hooks=hooks))
            await asyncio.wait_for(started.wait(), timeout=5)
            job.cancel()
            with pytest.raises(asyncio.CancelledError):
                await job

        assert provider.released == ['main']
        assert not provider.get_active_runtimes()
        assert provider._suppress_release is False
        assert hooks.hook_failures[0].stage == 'context_closing'


class FinalizingProcessor(SpanProcessor):
    def __init__(self) -> None:
        self.finalized = False

    def finalize(self) -> None:
        self.finalized = True

    def get_result(self):
        return {'finalized': self.finalized}


class TestHoldEarlyOutput:
    @pytest.mark.asyncio
    async def test_early_output_written_after_processors_finalize(self, provider, tmp_path) -> None:
        output_path = tmp_path / 'run-output.json'
        processor = FinalizingProcessor()
        seen_during_hold = False

        async def fake_hold(_runtime_provider, _timeout, *, install_signal_handlers=True) -> None:
            nonlocal seen_during_hold
            assert install_signal_handlers is True
            seen_during_hold = True
            assert output_path.exists()
            payload = json.loads(output_path.read_text())
            assert payload['task_output'] == 'done'
            assert payload['processor_results']['probe'] == {'finalized': True}

        run_input = RunInput(
            task_name='test_task',
            task_config={},
            runtime_provider_name='fake',
            runner=RunnerConfig(hold_after_task=30.0, early_output_path=str(output_path)),
        )

        with (
            _run_patches(provider, processors={'probe': processor}),
            patch('nexus.runner.single._hold_sandbox', side_effect=fake_hold),
        ):
            await run(run_input)

        assert seen_during_hold is True


class TestHoldTimeout:
    @pytest.mark.asyncio
    async def test_hold_exits_on_timeout(self, provider) -> None:
        run_input = RunInput(
            task_name='test_task',
            task_config={},
            runtime_provider_name='fake',
            runner=RunnerConfig(hold_after_task=0.2),
        )
        with _run_patches(provider):
            start = time.monotonic()
            await run(run_input)
            elapsed = time.monotonic() - start

        assert elapsed >= 0.2
        assert elapsed < 2.0


class TestHoldSignalExit:
    @pytest.mark.asyncio
    async def test_hold_exits_on_signal_event(self, provider) -> None:
        captured_callbacks: dict[int, Any] = {}

        def fake_add_signal_handler(sig, callback):
            captured_callbacks[sig] = callback

        def fake_remove_signal_handler(sig):
            captured_callbacks.pop(sig, None)

        run_input = RunInput(
            task_name='test_task',
            task_config={},
            runtime_provider_name='fake',
            runner=RunnerConfig(hold_after_task=30.0),
        )

        async def fire_signal():
            await asyncio.sleep(0.05)
            if signal.SIGINT in captured_callbacks:
                captured_callbacks[signal.SIGINT]()

        loop = asyncio.get_event_loop()
        with (
            _run_patches(provider),
            patch.object(loop, 'add_signal_handler', side_effect=fake_add_signal_handler),
            patch.object(loop, 'remove_signal_handler', side_effect=fake_remove_signal_handler),
        ):
            trigger = asyncio.ensure_future(fire_signal())
            start = time.monotonic()
            await run(run_input)
            elapsed = time.monotonic() - start
            await trigger

        assert elapsed < 5.0

    @pytest.mark.asyncio
    async def test_hold_can_skip_signal_handlers(self, provider) -> None:
        loop = asyncio.get_event_loop()

        def fail_add_signal_handler(*_args, **_kwargs):
            raise AssertionError('signal handlers should not be installed')

        with patch.object(loop, 'add_signal_handler', side_effect=fail_add_signal_handler):
            start = time.monotonic()
            await _hold_sandbox(provider, 0.01, install_signal_handlers=False)
            elapsed = time.monotonic() - start

        assert elapsed < 1.0


class TestHoldInfoOutput:
    @pytest.mark.asyncio
    async def test_hold_prints_runtime_info(self, provider, run_input_with_hold, capsys) -> None:
        with _run_patches(provider):
            await run(run_input_with_hold)

        captured = capsys.readouterr()
        assert 'HOLD MODE' in captured.out
        assert 'http://10.0.0.1:32000' in captured.out


class TestNoHoldByDefault:
    @pytest.mark.asyncio
    async def test_no_hold_when_not_configured(self, provider, run_input_no_hold) -> None:
        with _run_patches(provider):
            start = time.monotonic()
            await run(run_input_no_hold)
            elapsed = time.monotonic() - start

        assert elapsed < 1.0
        assert provider._suppress_release is False
