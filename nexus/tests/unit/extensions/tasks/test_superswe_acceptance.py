import ast
import asyncio
import copy
import textwrap
import uuid

from types import SimpleNamespace

import pytest

from nexus.core import TaskRunInput, TaskRunOutput, task_registry
from nexus.core.context import ExecutionContext, require_context
from nexus.extensions.tasks.swe import acceptance
from nexus.runner.single import RunHookError, RunHooks, RunInput, RunOutput, run
from nexus.runtime.providers.base import RuntimeProvider


SOURCE_PATH = 'nexus/extensions/tasks/swe/constants.py'
CONTROLLER_ENVIRONMENT = {
    'SWE_AGENT_BASH_BACKEND': 'stateless',
    'NEXUS_CALL_DEADLINE': '3600',
    'NEXUS_RETRY_TOTAL_TIMEOUT': '7200',
    'NEXUS_LOG_EXEMPT_SIZE': '200000',
}


class ProbeRuntime:
    def __init__(self, name, fingerprint, events):
        self.name = name
        self.fingerprint = copy.deepcopy(fingerprint)
        self.events = events
        self.alive = True
        self.failure = None
        self.wait = False
        self.started = asyncio.Event()

    async def exec(self, code):
        assert self.alive, 'Source inspection ran after release'
        ast.parse('async def probe():\n' + textwrap.indent(code, '    '))
        self.events.append(f'inspect:{self.name}')
        self.started.set()
        if self.wait:
            await asyncio.Event().wait()
        if self.failure is not None:
            raise self.failure
        return copy.deepcopy(self.fingerprint)

    async def drain_trace(self):
        pass


class ProbeProvider(RuntimeProvider):
    def __init__(self, fingerprint):
        super().__init__()
        self.fingerprint = fingerprint
        self.events = []
        self.release_failure = None

    async def acquire(self, name, **params):
        runtime = ProbeRuntime(name, self.fingerprint, self.events)
        self._active_runtimes[name] = runtime
        return runtime

    async def _release_impl(self, name):
        self.events.append(f'release:{name}')
        if self.release_failure is not None:
            error, self.release_failure = self.release_failure, None
            raise error
        runtime = self._active_runtimes.pop(name, None)
        if runtime is not None:
            runtime.alive = False


@pytest.fixture
def setup(monkeypatch):
    for key, value in CONTROLLER_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    fingerprint = acceptance.source_fingerprint([SOURCE_PATH])
    provider = ProbeProvider(fingerprint)
    context = ExecutionContext(runtime_provider=provider, task_name='source-verification')
    hook = acceptance.SuperSWEAcceptanceHook(
        fingerprint['source_hashes'], CONTROLLER_ENVIRONMENT, fingerprint['package_version']
    )
    return context, provider, hook


def test_requires_explicit_source_hashes():
    with pytest.raises(ValueError, match='explicit source hashes'):
        acceptance.SuperSWEAcceptanceHook({}, {}, '1.8.21')


def test_requires_explicit_package_version():
    with pytest.raises(ValueError, match='explicit package version'):
        acceptance.SuperSWEAcceptanceHook({SOURCE_PATH: 'expected'}, {}, '')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('mismatch', 'message'),
    [
        ('source', 'Controller loaded unexpected Nexus source files'),
        ('environment', 'Controller environment differs'),
        ('provider', 'Acceptance requires a runtime provider'),
    ],
)
async def test_controller_failure_leaves_release_unchanged(setup, monkeypatch, mismatch, message):
    context, provider, hook = setup
    original = provider.release
    if mismatch == 'source':
        hook.expected_hashes = {SOURCE_PATH: 'unexpected'}
    elif mismatch == 'environment':
        monkeypatch.setenv('NEXUS_CALL_DEADLINE', '1')
    else:
        context.runtime_provider = None
    with pytest.raises(ValueError, match=message):
        await hook.on_context_ready(context)
    await hook.on_context_closing(context)
    assert provider.release == original
    assert hook.get_result()['passed'] is False


@pytest.mark.asyncio
async def test_inspects_both_workers_before_task_releases_them(setup):
    context, provider, hook = setup
    original = provider.release
    async with context:
        await hook.on_context_ready(context)
        for name in ('main', 'eval'):
            await context.get_runtime(name)
            assert await context.release_runtime(name) is True
        assert not context._runtimes
        await hook.on_run_output(context, RunOutput(success=True))
        await hook.on_context_closing(context)
    assert provider.events == ['inspect:main', 'release:main', 'inspect:eval', 'release:eval']
    assert set(hook.get_result()['workers']) == {'main', 'eval'}
    assert hook.get_result()['passed'] is True
    assert provider.release == original


@pytest.mark.asyncio
async def test_checks_workers_left_for_context_cleanup(setup):
    context, provider, hook = setup
    async with context:
        await hook.on_context_ready(context)
        await context.get_runtime('eval')
        await hook.on_run_output(context, RunOutput(success=True))
        await hook.on_context_closing(context)
    assert provider.events == ['inspect:eval', 'release:eval']
    assert hook.get_result()['passed'] is True


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['source', 'version', 'unavailable'])
async def test_worker_verification_failure_still_releases_resources(setup, failure):
    context, provider, hook = setup
    async with context:
        await hook.on_context_ready(context)
        worker = await context.get_runtime('main')
        if failure == 'source':
            worker.fingerprint['source_hashes'] = {}
        elif failure == 'version':
            worker.fingerprint['package_version'] = '0.0.0'
        else:
            worker.failure = RuntimeError('Worker disconnected')
        assert await context.release_runtime('main') is True
        with pytest.raises(ValueError, match='Worker source verification failed'):
            await hook.on_run_output(context, RunOutput(success=True))
        await hook.on_context_closing(context)
    assert not worker.alive
    assert hook.get_result()['worker_errors']['main']
    assert hook.get_result()['passed'] is False


@pytest.mark.asyncio
async def test_checks_all_remaining_workers_and_keeps_each_error(setup):
    context, provider, hook = setup
    async with context:
        await hook.on_context_ready(context)
        first = await context.get_runtime('main')
        await context.get_runtime('eval')
        first.fingerprint['source_hashes'] = {}
        with pytest.raises(ValueError, match='Worker source verification failed'):
            await hook.on_run_output(context, RunOutput(success=True))
        assert provider.events == ['inspect:main', 'inspect:eval']
        assert 'main' in hook.get_result()['worker_errors']
        await hook.on_context_closing(context)
    assert provider.get_active_runtimes() == {}


@pytest.mark.asyncio
async def test_release_error_preserves_original_exception_and_failed_acceptance(setup):
    context, provider, hook = setup
    error = OSError('Cleanup failed')
    async with context:
        await hook.on_context_ready(context)
        await context.get_runtime('main')
        provider.release_failure = error
        with pytest.raises(OSError, match='Cleanup failed') as captured:
            await context.release_runtime('main')
        assert captured.value is error
        with pytest.raises(ValueError, match='Worker release failed'):
            await hook.on_run_output(context, RunOutput(success=True))
        assert hook.get_result()['release_errors']['main'] == 'OSError: Cleanup failed'
        await hook.on_context_closing(context)
    assert provider.get_active_runtimes() == {}


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel', [False, True])
async def test_interrupted_probe_does_not_prevent_release(setup, monkeypatch, cancel):
    context, provider, hook = setup
    original = provider.release
    async with context:
        await hook.on_context_ready(context)
        worker = await context.get_runtime('main')
        worker.wait = True
        if not cancel:
            monkeypatch.setattr(
                acceptance,
                'asyncio',
                SimpleNamespace(wait_for=lambda awaitable, timeout: asyncio.wait_for(awaitable, timeout=0.01)),
            )
        releasing = asyncio.create_task(context.release_runtime('main'))
        await worker.started.wait()
        if cancel:
            releasing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await releasing
        else:
            assert await releasing is True
        assert not worker.alive
        assert hook.get_result()['worker_errors']['main'].startswith('CancelledError' if cancel else 'TimeoutError')
        assert hook.get_result()['passed'] is False
        await hook.on_context_closing(context)
    assert provider.release == original


@pytest.mark.asyncio
async def test_preserves_suppressed_and_forced_release_behavior(setup):
    context, provider, hook = setup
    async with context:
        await hook.on_context_ready(context)
        worker = await context.get_runtime('main')
        provider._suppress_release = True
        assert await context.release_runtime('main') is False
        assert worker.alive
        assert await context.release_runtime('main', force=True) is True
        assert not worker.alive
        await hook.on_run_output(context, RunOutput(success=True))
        await hook.on_context_closing(context)
    assert hook.get_result()['passed'] is True


@pytest.mark.asyncio
async def test_replacement_worker_cannot_hide_an_earlier_mismatch(setup):
    context, _provider, hook = setup
    async with context:
        await hook.on_context_ready(context)
        first = await context.get_runtime('main')
        first.fingerprint['source_hashes'] = {}
        await context.replace_runtime('main')
        await context.release_runtime('main')
        with pytest.raises(ValueError, match='Worker source verification failed'):
            await hook.on_run_output(context, RunOutput(success=True))
        await hook.on_context_closing(context)
    assert 'main' in hook.get_result()['worker_errors']
    assert hook.get_result()['passed'] is False


@pytest.mark.asyncio
async def test_missing_workers_cannot_pass_acceptance(setup):
    context, _provider, hook = setup
    await hook.on_context_ready(context)
    with pytest.raises(ValueError, match='No acquired worker'):
        await hook.on_run_output(context, RunOutput(success=True))
    await hook.on_context_closing(context)
    assert hook.get_result()['passed'] is False


@pytest.mark.asyncio
@pytest.mark.parametrize('fail_source', [False, True])
async def test_runner_preserves_acceptance_evidence_after_task_cleanup(register_test_component, setup, fail_source):
    _context, provider, template = setup
    task_name = f'acceptance_lifecycle_{uuid.uuid4().hex}'
    original = provider.release

    @register_test_component(task_registry, task_name)
    async def task(_input: TaskRunInput) -> TaskRunOutput:
        ctx = require_context()
        worker = await ctx.get_runtime('main')
        if fail_source:
            worker.fingerprint['source_hashes'] = {}
        await ctx.release_runtime('main')
        return TaskRunOutput()

    def install_provider(ctx):
        ctx.runtime_provider = provider

    callbacks = RunHooks(on_context_ready=install_provider)
    output = await run(
        RunInput(
            task_name=task_name,
            runner={
                'hooks': {
                    'sources': {
                        'name': 'superswe_acceptance',
                        'required': True,
                        'args': {
                            'expected_hashes': template.expected_hashes,
                            'controller_environment': CONTROLLER_ENVIRONMENT,
                            'expected_package_version': template.expected_package_version,
                        },
                    }
                }
            },
        ),
        hooks=callbacks,
    )
    assert output.success is True
    assert output.hook_results['sources']['passed'] is not fail_source
    assert provider.events == ['inspect:main', 'release:main']
    assert provider.release == original
    if fail_source:
        assert 'sources.on_run_output' in output.hook_errors
        with pytest.raises(RunHookError):
            callbacks.raise_if_failed()
    else:
        assert output.hook_errors == {}
        callbacks.raise_if_failed()
