"""OSWorld sandbox regressions."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.tasks.os_world.data.task_source import OSWorldBatchTask
from nexus.extensions.tasks.os_world.execution import large_sandbox
from nexus.extensions.tasks.os_world.execution import large_sandbox as bootstrap
from nexus.extensions.tasks.os_world.execution import sandbox_policy as policy


@pytest.fixture
def environment(monkeypatch, tmp_path):
    for name, value in {
        'AGS_SECRET_ID': 'test-id',
        'AGS_SECRET_KEY': 'test-key',
        'AIHUB_API_KEY': 'sk-test-only',
        'MODEL_SERVICE_API_APP_ID': 'test-app',
        'MODEL_SERVICE_API_APP_KEY': 'test-key',
        'AGS_OSWORLD_TOOL_ID': 'sdt-47kozq7f',
        'OSWORLD_EXAMPLES_ROOT': str(tmp_path),
        'OSWORLD_TEST_META_PATH': str(tmp_path / 'meta.json'),
        'OSWORLD_ENABLE_PROXY': '0',
        'OSWORLD_VALIDATE_EVALUATOR_HOST_DEPS': '0',
        'OSWORLD_SESSION_NAMESPACE': str(tmp_path),
        'CC_THINKING_PRESET': 'hunyuan',
        'CC_GUIDE_VERSION': 'v2',
        'CC_GUIDE_FILE': 'computer_use_guide_v2.md',
    }.items():
        monkeypatch.setenv(name, value)
    for name in ('CC_DISALLOWED_TOOLS', 'CC_MARKER_DETECTION', 'NEXUS_TAMPER_THINKING_INDEX'):
        monkeypatch.delenv(name, raising=False)


def task(domain='multi_apps', task_id=policy.INSTRUCTOR_TASK_ID):
    return OSWorldBatchTask(domain=domain, task_id=task_id, instruction='Task instruction', task_file='unused.json')


@pytest.mark.usefixtures('environment')
@pytest.mark.parametrize(
    ('domain', 'task_id', 'expected'),
    [
        ('multi_apps', policy.INSTRUCTOR_TASK_ID, True),
        ('gimp', policy.GIMP_UPSCALE_TASK_ID, True),
        ('gimp', policy.INSTRUCTOR_TASK_ID, False),
        ('multi_apps', policy.GIMP_UPSCALE_TASK_ID, False),
        ('gimp', 'another-task', False),
    ],
)
def test_policy_requires_exact_domain_and_task_id(domain, task_id, expected):
    assert policy.uses_large_sandbox(task(domain, task_id)) is expected


def provider():
    result = bootstrap.OSWorldV1LargeRuntimeProvider(secret_id='test', secret_key='test')
    result.configure('main', tool_id=policy.LARGE_TOOL_ID)
    return result


@pytest.mark.usefixtures('environment')
@pytest.mark.asyncio
async def test_pinned_cli_only(monkeypatch):
    runtime = object()
    monkeypatch.setattr(bootstrap.AGSRuntimeProvider, 'acquire', AsyncMock(return_value=runtime))
    prepare = AsyncMock()
    monkeypatch.setattr(bootstrap, '_prepare_v1_guest', prepare)
    run = AsyncMock(return_value=(0, 'PINNED_CC_VERSION=2.1.205 (Claude Code)\n', ''))
    monkeypatch.setattr(bootstrap, '_run_detached', run)
    instance = provider()
    assert await instance.acquire('main') is runtime
    assert await instance.acquire('main') is runtime
    prepare.assert_awaited_once()
    run.assert_awaited_once()
    script = run.call_args.args[1]
    assert '@anthropic-ai/claude-code@2.1.205' in script
    assert '20*1024**3' in script
    for forbidden in ['pip install', 'InstructorEmbedding', 'torch', 'datasets']:
        assert forbidden not in script
    # The script still gets the same overall budget; it is now polled rather than
    # awaited inside one request that the gateway would cut at ~120s.
    assert bootstrap._BOOTSTRAP_TIMEOUT_SECONDS == 900
    assert bootstrap._STEP_TIMEOUT_SECONDS < 120


@pytest.mark.usefixtures('environment')
@pytest.mark.asyncio
@pytest.mark.parametrize('result', [(0, '2.2.0 (Claude Code)', ''), (1, '', 'failed')])
async def test_bootstrap_failure_releases_sandbox(monkeypatch, result):
    monkeypatch.setattr(bootstrap.AGSRuntimeProvider, 'acquire', AsyncMock(return_value=object()))
    monkeypatch.setattr(bootstrap, '_prepare_v1_guest', AsyncMock())
    detached = AsyncMock(return_value=result)
    monkeypatch.setattr(bootstrap, '_run_detached', detached)
    monkeypatch.setattr(bootstrap.asyncio, 'sleep', AsyncMock())
    instance = provider()
    instance.release = AsyncMock()
    with pytest.raises(RuntimeError, match='pinned CC bootstrap failed'):
        await instance.acquire('main')
    instance.release.assert_awaited_once_with('main')
    assert not instance._bootstrapped
    # A persistent failure is retried, then surfaced -- the sandbox is released once.
    assert detached.await_count == bootstrap._BOOTSTRAP_ATTEMPTS


@pytest.mark.usefixtures('environment')
@pytest.mark.asyncio
async def test_unvalidated_template_rejected_before_provisioning(monkeypatch):
    acquire = AsyncMock()
    monkeypatch.setattr(bootstrap.AGSRuntimeProvider, 'acquire', acquire)
    with pytest.raises(ValueError, match='restricted'):
        await provider().acquire('main', tool_id='another-image')
    acquire.assert_not_awaited()


@pytest.mark.usefixtures('environment')
@pytest.mark.asyncio
async def test_guest_alignment_is_scoped():
    runtime = SimpleNamespace(run_command=AsyncMock(return_value=SimpleNamespace(return_code=0)))
    await bootstrap._prepare_v1_guest(runtime)
    assert runtime.run_command.await_count == 2
    setup, check = runtime.run_command.call_args_list
    assert 'chown user:user /home/user' in setup.kwargs['command']
    assert 'chpasswd' in setup.kwargs['command']
    assert 'chown -R' not in setup.kwargs['command']
    assert setup.kwargs['stdin'] == bootstrap._IMAGE_LOGIN_DEFAULT + '\n'
    assert check.kwargs['command'].startswith('test -w /home/user')


@pytest.mark.usefixtures('environment')
@pytest.mark.asyncio
async def test_guest_alignment_failure_releases_sandbox(monkeypatch):
    runtime = SimpleNamespace(run_command=AsyncMock(return_value=SimpleNamespace(return_code=1)))
    monkeypatch.setattr(bootstrap.AGSRuntimeProvider, 'acquire', AsyncMock(return_value=runtime))
    instance = provider()
    instance.release = AsyncMock()
    with pytest.raises(RuntimeError, match='home/sudo alignment failed'):
        await instance.acquire('main')
    instance.release.assert_awaited_once_with('main')
    assert runtime.run_command.await_count == 1


@pytest.mark.usefixtures('environment')
@pytest.mark.asyncio
async def test_release_clears_ready_state(monkeypatch):
    release = AsyncMock()
    monkeypatch.setattr(bootstrap.AGSRuntimeProvider, '_release_impl', release)
    instance = provider()
    instance._bootstrapped['main'] = object()
    await instance.release('main')
    assert not instance._bootstrapped
    release.assert_awaited_once_with('main')


@pytest.mark.usefixtures('environment')
@pytest.mark.parametrize(
    'agent', ['codex_native_bridge', 'claude_code_native_bridge', 'cua_gpt_agent', 'cua_anthropic_agent']
)
@pytest.mark.parametrize('selected', [task(), task(domain='gimp', task_id=policy.GIMP_UPSCALE_TASK_ID)])
def test_large_policy_changes_only_runtime_for_all_native_agents(selected, agent):
    original = {
        'agent_name': agent,
        'task_config': {'task_id': selected.task_id},
        'agent_config': {'marker': 'preserved'},
        'openai_clients': {'main': {'name': 'fixture'}},
        'runtime_provider_name': 'ags',
        'runtime_provider_config': {},
        'runtime_configs': {'main': {'tool_id': 'original'}},
        'imports': [],
    }
    actual = policy.apply_sandbox_policy(selected, deepcopy(original))
    assert (
        actual['runtime_provider_config']['cli_harness']
        == {
            'codex_native_bridge': 'codex',
            'claude_code_native_bridge': 'cc',
            'cua_gpt_agent': 'bare',
            'cua_anthropic_agent': 'bare',
        }[agent]
    )
    assert actual['runtime_configs']['main']['tool_id'] == policy.LARGE_TOOL_ID
    for key in ['task_config', 'agent_config', 'openai_clients']:
        assert actual[key] == original[key]
    assert policy.apply_sandbox_policy(selected, deepcopy(actual)) == actual
    assert actual['imports'].count(policy.LARGE_PROVIDER_MODULE) == 1


@pytest.mark.usefixtures('environment')
@pytest.mark.asyncio
async def test_codex_large_guest_has_same_disk_check_without_cc(monkeypatch):
    runtime = object()
    monkeypatch.setattr(bootstrap.AGSRuntimeProvider, 'acquire', AsyncMock(return_value=runtime))
    prepare = AsyncMock()
    monkeypatch.setattr(bootstrap, '_prepare_v1_guest', prepare)
    run = AsyncMock(return_value=(0, 'GUEST_FREE_BYTES=999999999999\n', ''))
    monkeypatch.setattr(bootstrap, '_run_detached', run)
    instance = bootstrap.OSWorldV1LargeRuntimeProvider(secret_id='test', secret_key='test', cli_harness='codex')
    instance.configure('main', tool_id=policy.LARGE_TOOL_ID)
    assert await instance.acquire('main') is runtime
    assert await instance.acquire('main') is runtime
    prepare.assert_awaited_once()
    run.assert_awaited_once()
    script = run.call_args.args[1]
    assert '20*1024**3' in script
    for forbidden in ['claude', 'npm install', 'pip install', 'torch', 'InstructorEmbedding']:
        assert forbidden not in script


@pytest.mark.usefixtures('environment')
@pytest.mark.asyncio
async def test_codex_disk_validation_failure_releases_guest(monkeypatch):
    monkeypatch.setattr(bootstrap.AGSRuntimeProvider, 'acquire', AsyncMock(return_value=object()))
    monkeypatch.setattr(bootstrap, '_prepare_v1_guest', AsyncMock())
    monkeypatch.setattr(bootstrap, '_run_detached', AsyncMock(return_value=(None, '', '')))
    monkeypatch.setattr(bootstrap.asyncio, 'sleep', AsyncMock())
    instance = bootstrap.OSWorldV1LargeRuntimeProvider(secret_id='test', secret_key='test', cli_harness='codex')
    instance.configure('main', tool_id=policy.LARGE_TOOL_ID)
    instance.release = AsyncMock()
    with pytest.raises(RuntimeError, match='disk validation failed'):
        await instance.acquire('main')
    instance.release.assert_awaited_once_with('main')


@pytest.mark.usefixtures('environment')
def test_unknown_harness_rejected():
    with pytest.raises(ValueError, match='cli_harness'):
        bootstrap.OSWorldV1LargeRuntimeProvider(secret_id='test', secret_key='test', cli_harness='other')


class _Resp:
    def __init__(self, rc: int | None, out: str = '', err: str = '') -> None:
        self.return_code = rc
        self.stdout = out
        self.stderr = err

    def model_dump(self) -> dict[str, Any]:
        return {'return_code': self.return_code, 'stdout': self.stdout, 'stderr': self.stderr}


class _Guest:
    """Minimal guest that runs the detached script after `finish_after` polls."""

    def __init__(self, *, script_rc: int = 0, log: str = '', finish_after: int = 1) -> None:
        self.script_rc = script_rc
        self.log = log
        self.finish_after = finish_after
        self.polls = 0
        self.timeouts: list[float] = []
        self.commands: list[str] = []

    async def run_command(self, command: str, **kwargs: Any) -> _Resp:
        self.commands.append(command)
        self.timeouts.append(float(kwargs.get('timeout', 0)))
        if command.startswith('rm -rf'):
            return _Resp(0)
        if command.startswith('setsid'):
            return _Resp(0, 'LAUNCHED\n')
        if command.startswith('if [ -f'):
            self.polls += 1
            if self.polls >= self.finish_after:
                return _Resp(0, f'RC={self.script_rc}')
            return _Resp(0, 'RUNNING')
        if command.startswith('cat '):
            return _Resp(0, self.log)
        return _Resp(0)


@pytest.mark.asyncio
async def test_detached_bootstrap_never_holds_a_request_past_the_gateway_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(large_sandbox.asyncio, 'sleep', _noop_sleep)
    guest = _Guest(script_rc=0, log='PINNED_CC_VERSION=2.1.205 (Claude Code)\n', finish_after=6)

    rc, out, _ = await large_sandbox._run_detached(guest, 'echo hi', poll_interval=0.0)

    assert rc == 0
    assert 'PINNED_CC_VERSION' in out
    # The whole point: no single request may approach the ~120s gateway cut.
    assert guest.timeouts, 'no requests were issued'
    assert max(guest.timeouts) <= 60.0, f'a request could outlive the gateway: {max(guest.timeouts)}s'
    # Work is polled, not awaited inside one call.
    assert guest.polls >= 6


@pytest.mark.asyncio
async def test_detached_bootstrap_clears_a_stale_sentinel_before_launching() -> None:
    guest = _Guest(script_rc=0, log='GUEST_FREE_BYTES=999\n')

    await large_sandbox._run_detached(guest, 'echo hi', poll_interval=0.0)

    assert guest.commands[0].startswith('rm -rf '), 'stale rc file could be read as this run'
    assert guest.commands[0].index('rm -rf') < guest.commands[0].index('mkdir -p')


@pytest.mark.asyncio
async def test_detached_bootstrap_times_out_instead_of_polling_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(large_sandbox.asyncio, 'sleep', _noop_sleep)
    guest = _Guest(finish_after=10**9)

    with pytest.raises(TimeoutError):
        await large_sandbox._run_detached(guest, 'echo hi', overall_timeout=0.0, poll_interval=0.0)


@pytest.mark.asyncio
async def test_bootstrap_retries_on_a_transient_failure_without_recreating_the_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(large_sandbox.asyncio, 'sleep', _noop_sleep)
    provider = large_sandbox.OSWorldV1LargeRuntimeProvider.__new__(large_sandbox.OSWorldV1LargeRuntimeProvider)
    provider.cli_harness = 'cc'
    calls = {'n': 0}

    async def flaky(_runtime: Any, _script: str, **_kw: Any) -> tuple[int, str, str]:
        calls['n'] += 1
        if calls['n'] == 1:
            raise RuntimeError('Server disconnected')
        return 0, 'PINNED_CC_VERSION=2.1.205 (Claude Code)\n', ''

    monkeypatch.setattr(large_sandbox, '_run_detached', flaky)
    sentinel = object()

    await provider._bootstrap_with_retries(sentinel)

    assert calls['n'] == 2, 'a transient disconnect must not fail the task'


@pytest.mark.asyncio
async def test_bootstrap_gives_up_after_the_attempt_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(large_sandbox.asyncio, 'sleep', _noop_sleep)
    provider = large_sandbox.OSWorldV1LargeRuntimeProvider.__new__(large_sandbox.OSWorldV1LargeRuntimeProvider)
    provider.cli_harness = 'cc'
    calls = {'n': 0}

    async def always_fail(_runtime: Any, _script: str, **_kw: Any) -> tuple[int, str, str]:
        calls['n'] += 1
        raise RuntimeError('Server disconnected')

    monkeypatch.setattr(large_sandbox, '_run_detached', always_fail)

    with pytest.raises(RuntimeError, match='Server disconnected'):
        await provider._bootstrap_with_retries(object())

    assert calls['n'] == large_sandbox._BOOTSTRAP_ATTEMPTS


@pytest.mark.asyncio
async def test_bootstrap_rejects_a_wrong_pinned_cc_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """The version contract must stay enforced; retrying must not paper over it."""
    monkeypatch.setattr(large_sandbox.asyncio, 'sleep', _noop_sleep)
    provider = large_sandbox.OSWorldV1LargeRuntimeProvider.__new__(large_sandbox.OSWorldV1LargeRuntimeProvider)
    provider.cli_harness = 'cc'

    async def wrong_version(_runtime: Any, _script: str, **_kw: Any) -> tuple[int, str, str]:
        return 0, 'PINNED_CC_VERSION=9.9.9 (Claude Code)\n', ''

    monkeypatch.setattr(large_sandbox, '_run_detached', wrong_version)

    with pytest.raises(RuntimeError, match='pinned CC bootstrap failed'):
        await provider._bootstrap_with_retries(object())


async def _noop_sleep(_seconds: float) -> None:
    return None
