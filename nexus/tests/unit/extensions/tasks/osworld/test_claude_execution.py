"""OSWorld claude execution regressions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.agents.claude_code.native_bridge import agent_runtime
from nexus.extensions.agents.claude_code.native_bridge.adapters.osworld import (
    OSWorldClaudeNativeBridgeAdapter as GenericAdapter,
)
from nexus.extensions.agents.claude_code.native_bridge.adapters.posix import _SessionPaths
from nexus.extensions.agents.claude_code.native_bridge.adapters.shell import build_claude_code_env
from nexus.extensions.agents.claude_code.native_bridge.adapters.types import ClaudeRunRequest
from nexus.extensions.tasks.os_world.execution.claude_code import (
    _BASH_PROCESS_GUARD,
    _GNU_GREP_HOOK,
    OSWorldClaudeExecutionProfile,
    OSWorldClaudeNativeBridgeAdapter,
)
from nexus.runtime.bridge.adapters.base import BridgeLaunchResult
from nexus.runtime.executors.models.command import RunCommandResponse
from nexus.runtime.providers.ags.osworld import AGSOSWorldRuntime


@pytest.mark.parametrize(
    ('overrides', 'timeout', 'turns'),
    [
        ({}, 77, 9),
        ({'timeout': 3}, 3, 9),
        ({'max_turns': 1}, 77, 1),
        ({'timeout': 4, 'max_turns': 2}, 4, 2),
    ],
)
def test_standard_budget_inputs_override_only_when_explicit(overrides, timeout, turns):
    from nexus.extensions.agents.claude_code.agent import ClaudeCodeNativeBridgeInput

    options = {'cc_timeout': 77, 'max_turns': 9}
    inp = ClaudeCodeNativeBridgeInput(
        user_prompt='fixture', execution_profile='osworld', execution_options=options, **overrides
    )
    profile = inp.build_execution_profile()
    assert profile.options.timeout == timeout
    assert profile.options.max_turns == turns
    assert profile.bridge_options(None)['timeout'] == timeout
    assert options == {'cc_timeout': 77, 'max_turns': 9}
    restored = ClaudeCodeNativeBridgeInput.model_validate(inp.model_dump(exclude_unset=True))
    assert restored.build_execution_profile().options.timeout == timeout
    assert 'cc_timeout' not in profile.options.model_dump()


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf')])
def test_profile_rejects_invalid_top_level_time_budget(timeout):
    from nexus.extensions.agents.claude_code.agent import ClaudeCodeNativeBridgeInput

    with pytest.raises(ValueError, match='timeout'):
        ClaudeCodeNativeBridgeInput(user_prompt='fixture', execution_profile='osworld', timeout=timeout)


def test_profile_must_opt_in_to_top_level_timeout():
    from nexus.extensions.agents.claude_code.native_bridge.adapters.base import (
        build_execution_profile,
        execution_profile_registry,
    )

    name = 'test_no_budget_alias'
    if name not in execution_profile_registry:
        execution_profile_registry.register(name)(lambda *args: object())
    with pytest.raises(ValueError, match='timeout must be configured through execution_options'):
        build_execution_profile(name, {}, {}, 'fixture', input_overrides={'timeout': 3})


def test_timeout_has_one_internal_name_and_standard_key_wins():
    from nexus.extensions.tasks.os_world.execution.claude_code import OSWorldExecutionOptions

    options = OSWorldExecutionOptions.model_validate({'timeout': 12, 'cc_timeout': 77})
    assert options.timeout == 12
    assert 'cc_timeout' not in options.model_dump()
    assert OSWorldExecutionOptions().timeout == 1800


@pytest.mark.asyncio
async def test_agent_entry_passes_standard_budgets_to_real_profile(monkeypatch):
    from nexus.extensions.agents.claude_code.agent import ClaudeCodeNativeBridgeInput, run_claude_code_native_bridge

    captured = {}

    async def execute(self, client):
        captured.update(timeout=self.options.timeout, max_turns=self.options.max_turns)
        return {'success': True, 'output': '', 'stderr': '', 'exit_code': 0, 'execution_time': 0}

    monkeypatch.setattr(OSWorldClaudeExecutionProfile, 'run', execute)
    monkeypatch.setattr(
        'nexus.extensions.agents.claude_code.agent.require_context',
        lambda: SimpleNamespace(get_openai_client=lambda: object()),
    )
    inp = ClaudeCodeNativeBridgeInput(
        user_prompt='fixture',
        execution_profile='osworld',
        timeout=7,
        max_turns=1,
        execution_options={'cc_timeout': 77, 'max_turns': 9},
    )
    await run_claude_code_native_bridge(inp)
    assert captured == {'timeout': 7, 'max_turns': 1}


def test_gnu_grep_hook_preserves_input_and_uses_system_grep(tmp_path):
    payload = {
        'tool_input': {
            'command': 'printf \'date="2026"\\n\' | grep -iEo '
            """'(date|published|updated|copyright)[a-zA-Z:_-]* ?[=:] ?"?[^",}]{0,60}'""",
            'timeout': 10000,
            'description': 'unchanged description',
        }
    }
    result = subprocess.run(
        [sys.executable, '-c', _GNU_GREP_HOOK], input=json.dumps(payload), text=True, capture_output=True, check=True
    )
    output = json.loads(result.stdout)['hookSpecificOutput']
    assert output['hookEventName'] == 'PreToolUse'
    assert 'permissionDecision' not in output
    updated = output['updatedInput']
    assert updated['description'] == payload['tool_input']['description']
    assert updated['timeout'] == payload['tool_input']['timeout']
    assert updated['command'].endswith(payload['tool_input']['command'])
    # Stand in for CC's function. The Hook must override it after snapshot load.
    execution = subprocess.run(
        ['bash', '-c', 'grep() { exit 77; }; ' + updated['command']],
        text=True,
        capture_output=True,
        check=True,
    )
    assert '2026' in execution.stdout


def test_gnu_grep_hook_rejects_malformed_input():
    result = subprocess.run(
        [sys.executable, '-c', _GNU_GREP_HOOK], input='{"tool_input":{}}', text=True, capture_output=True
    )
    assert result.returncode == 2
    assert not result.stdout


@pytest.mark.asyncio
async def test_gnu_grep_and_process_guard_share_settings(tmp_path):
    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    adapter.runtime = LocalRuntime()
    adapter.diagnostics = {}
    adapter.system_grep = True
    adapter.process_guard = True
    result = await adapter._install_guard(request(), str(tmp_path))
    settings = json.loads((tmp_path / 'settings.json').read_text())
    hooks = settings['hooks']['PreToolUse'][0]['hooks']
    assert len(hooks) == 2
    assert '/g ' in hooks[0]['command']
    assert '/gnu-grep ' in hooks[1]['command']
    assert result.extra_cli_args == ['--settings', str(tmp_path / 'settings.json')]


@pytest.mark.asyncio
async def test_standard_runner_keeps_stdin_prompt_and_hooks(monkeypatch, tmp_path):
    adapter = OSWorldClaudeNativeBridgeAdapter(object.__new__(AGSOSWorldRuntime))
    adapter.runtime = LocalRuntime()
    adapter.process_guard = True
    assert adapter.system_grep is True
    adapter._observe = AsyncMock()
    adapter._capture_guard = AsyncMock()
    prompt = 'task input on stdin'

    async def common_run(req):
        assert req.task == ''
        assert Path(adapter._prompt_file).read_text() == prompt
        settings = json.loads(Path(req.extra_cli_args[-1]).read_text())
        assert len(settings['hooks']['PreToolUse'][0]['hooks']) == 2
        command = adapter._build_launch_command(req, _SessionPaths(str(tmp_path / 'session')))
        assert f'< {adapter._prompt_file}' in command
        assert prompt not in command
        assert 'systemd-run' not in command
        return 'completed'

    common = AsyncMock(side_effect=common_run)
    monkeypatch.setattr(GenericAdapter, 'run_claude', common)
    assert await adapter.run_claude(request(prompt)) == 'completed'
    common.assert_awaited_once()
    adapter._observe.assert_awaited_once_with('before_agent')
    adapter._capture_guard.assert_awaited_once()
    assert adapter._prompt_file is None


@pytest.mark.asyncio
async def test_missing_gnu_grep_fails_before_launch(tmp_path):
    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    adapter.runtime = SimpleNamespace(
        run_command=AsyncMock(return_value=SimpleNamespace(return_code=1, stdout='')), setup_upload_bytes=AsyncMock()
    )
    adapter.system_grep = True
    with pytest.raises(RuntimeError, match='GNU grep'):
        await adapter._install_guard(request(), str(tmp_path))
    adapter.runtime.setup_upload_bytes.assert_not_awaited()


def request(prompt='test', **kwargs):
    return ClaudeRunRequest(
        claude_executable=kwargs.pop('claude_executable', '/usr/bin/claude'),
        launch=BridgeLaunchResult(base_url='http://test', local_base_url='http://localhost:9281'),
        task=prompt,
        working_dir=kwargs.pop('working_dir', '/tmp'),
        timeout=15,
        **kwargs,
    )


class LocalRuntime:
    """Local commands and uploads; no model, network or real AGS instance."""

    def __init__(self):
        self.commands = []
        self.uploads = []

    async def run_command(self, command, timeout=30, **kwargs):
        self.commands.append(command)
        process = await asyncio.create_subprocess_exec(
            'bash',
            '-c',
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        return RunCommandResponse(
            status='Finished' if process.returncode == 0 else 'Error',
            return_code=process.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode(),
        )

    async def setup_upload_bytes(self, path, content, **kwargs):
        self.uploads.append(path)
        Path(path).write_bytes(content)
        return SimpleNamespace(status_code=200)


@pytest.mark.asyncio
async def test_real_shell_receives_large_prompt_on_stdin_without_argv_collision(tmp_path):
    marker = 'nexus_prompt_collision_probe_' + os.urandom(8).hex()
    # Unique synthetic marker, not a real application name or broad pkill.
    prompt = marker + '\n' + '中文 shell \' "$HOME" ; gimp chrome\n' * 10000
    executable = tmp_path / 'fake-claude'
    executable.write_text(
        f'#!{sys.executable}\n'
        'import hashlib,json,os,sys\n'
        'content=sys.stdin.buffer.read()\n'
        'marker=content.splitlines()[0]\n'
        'parent=open(f"/proc/{os.getppid()}/cmdline","rb").read()\n'
        'own=open("/proc/self/cmdline","rb").read()\n'
        'assert marker not in parent and marker not in own, "prompt exposed to pkill -f"\n'
        'print(json.dumps({"type":"result","subtype":"success","result":hashlib.sha256(content).hexdigest()}))\n'
    )
    executable.chmod(0o700)
    runtime = LocalRuntime()
    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    adapter.runtime = runtime
    adapter._prompt_file = None
    result = await adapter.run_claude(
        request(
            prompt,
            claude_executable=str(executable),
            working_dir=str(tmp_path),
        )
    )
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout)['result'] == hashlib.sha256(prompt.encode()).hexdigest()
    assert all(marker not in command for command in runtime.commands)
    assert max(map(len, runtime.commands)) < 10000
    assert all(not Path(path).parent.exists() for path in runtime.uploads)


@pytest.mark.parametrize('failure', ['upload', 'checksum'])
@pytest.mark.asyncio
async def test_staging_failure_prevents_launch_and_cleans_prompt(failure):
    runtime = SimpleNamespace(
        run_command=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout=''),
                SimpleNamespace(return_code=0, stdout='wrong-checksum'),
                SimpleNamespace(return_code=0, stdout=''),
            ]
        ),
        setup_upload_bytes=AsyncMock(return_value=SimpleNamespace(status_code=500 if failure == 'upload' else 200)),
    )
    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    adapter.runtime, adapter._prompt_file = runtime, None
    with pytest.raises(RuntimeError, match='upload'):
        await adapter.run_claude(request())
    commands = [call.kwargs['command'] for call in runtime.run_command.await_args_list]
    assert not any('nohup' in command for command in commands)
    assert commands[-1].startswith('rm -rf /tmp/nexus-cc-input-')
    assert adapter._prompt_file is None


@pytest.mark.parametrize('override', [None, '0', '1'])
def test_cli_watchdog_matches_reference_without_changing_generic_default(override):
    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    adapter._prompt_file = '/tmp/private-prompt'
    extra = {} if override is None else {'CLAUDE_ENABLE_STREAM_WATCHDOG': override}
    command = adapter._build_launch_command(request(extra_env=extra), _SessionPaths('/tmp/test-session'))
    if override is None:
        assert 'CLAUDE_ENABLE_STREAM_WATCHDOG' not in command
    else:
        assert f'CLAUDE_ENABLE_STREAM_WATCHDOG={override}' in command
    assert build_claude_code_env(local_bridge_base_url='http://test')['CLAUDE_ENABLE_STREAM_WATCHDOG'] == '0'
    assert GenericAdapter.run_claude is not OSWorldClaudeNativeBridgeAdapter.run_claude


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('rc', 'subtype', 'text', 'expected_success', 'expected_status'),
    [
        (0, 'success', 'completed', True, 'finished'),
        (0, 'success', 'TASK_INFEASIBLE', True, 'infeasible'),
        (1, 'error_max_turns', '', True, 'max_iterations'),
        (1, 'max_turns', '', True, 'max_iterations'),
        (0, 'error_max_turns', 'TASK_DONE', True, 'max_iterations'),
        (0, 'max_turns', '', True, 'max_iterations'),
        (-1, 'error_max_turns', '', False, 'tool_error'),
        (1, 'success', 'TASK_DONE', False, 'tool_error'),
        (-1, 'success', 'TASK_DONE', False, 'tool_error'),
    ],
)
async def test_profile_maps_budget_and_preserves_failure_diagnostics(
    monkeypatch,
    tmp_path,
    rc,
    subtype,
    text,
    expected_success,
    expected_status,
):
    runtime = object.__new__(AGSOSWorldRuntime)
    raw = json.dumps({'type': 'result', 'subtype': subtype, 'result': text})
    runner = AsyncMock(
        return_value={
            'success': rc == 0,
            'exit_code': rc,
            'output': raw,
            'stderr': 'diagnostic',
        }
    )
    monkeypatch.setattr(agent_runtime, 'setup_and_run_native_bridge_claude_code', runner)
    profile = OSWorldClaudeExecutionProfile(
        {'guide_text': 'Guide', 'working_dir': '/home/user'},
        {'osworld_task_id': 'test', 'osworld_domain': 'os', 'trajectory_dir': str(tmp_path / 'os__test')},
        'Task',
    )
    result = await profile.execute(runtime=runtime, launch=request().launch, timeout=15)
    assert result['success'] is expected_success
    assert result['exit_status'] == expected_status
    assert result['cc_return_code'] == rc
    factory = runner.call_args.kwargs['adapter_factory']
    assert isinstance(factory(runtime), OSWorldClaudeNativeBridgeAdapter)
    directory = tmp_path / 'os__test'
    assert (directory / 'cc_prompt.txt').read_text() == 'Guide\nTask\n'
    assert (directory / 'cc_stream.jsonl').read_text() == raw
    assert (directory / 'cc_stderr.txt').read_text() == 'diagnostic'
    assert json.loads((directory / 'cc_trajectory.json').read_text())['success'] is expected_success


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('rc', 'subtype', 'expected'),
    [(0, None, True), (1, None, True), (1, 'success', True), (-1, None, False), (1, 'error_during_execution', False)],
)
async def test_stderr_turn_limit_is_not_done_or_transport_success(monkeypatch, rc, subtype, expected):
    from nexus.extensions.tasks.os_world.task import _marker_from_agent_output

    runner = AsyncMock(
        return_value={
            'success': rc == 0,
            'exit_code': rc,
            'output': json.dumps({'type': 'result', 'subtype': subtype, 'result': 'TASK_DONE'}),
            'stderr': 'Error: Reached max turns (200)',
        }
    )
    monkeypatch.setattr(agent_runtime, 'setup_and_run_native_bridge_claude_code', runner)
    profile = OSWorldClaudeExecutionProfile({'guide_text': '', 'working_dir': '/tmp'}, {}, 'Task')
    result = await profile.execute(runtime=object.__new__(AGSOSWorldRuntime), launch=request().launch, timeout=15)
    assert result['success'] is expected
    assert result['exit_status'] == ('max_iterations' if expected else 'tool_error')
    assert _marker_from_agent_output(SimpleNamespace(**result)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['read', 'cleanup', 'both', 'none'])
async def test_exception_diagnostics_precede_cleanup_and_preserve_primary(monkeypatch, failure):
    from nexus.extensions.agents.claude_code.native_bridge.adapters.posix import BasePosixClaudeNativeBridgeAdapter

    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    events = []

    async def command(command, **kwargs):
        events.append('read')
        if failure in {'read', 'both'}:
            raise ConnectionError('disconnected')
        return SimpleNamespace(return_code=0, stdout='partial output')

    adapter.runtime = SimpleNamespace(run_command=command)

    async def cleanup(self, paths, *, terminate):
        events.append('cleanup')
        if failure in {'cleanup', 'both'}:
            raise RuntimeError('secondary cleanup failure')

    monkeypatch.setattr(BasePosixClaudeNativeBridgeAdapter, '_run_cleanup', cleanup)
    primary = ValueError('primary poll failure')

    async def invocation():
        try:
            raise primary
        finally:
            await adapter._run_cleanup(_SessionPaths('/tmp/test'), terminate=True)

    with pytest.raises(ValueError, match='primary poll failure') as caught:
        await invocation()
    assert caught.value is primary
    assert events == ['read'] * 5 + ['cleanup']
    assert adapter.diagnostics['primary_error_type'] == 'ValueError'
    if failure in {'read', 'both'}:
        assert len(adapter.diagnostics['errors']) == 4
    else:
        assert adapter.diagnostics['stdout'] == 'partial output'
    if failure in {'cleanup', 'both'}:
        assert adapter.diagnostics['cleanup_error_type'] == 'RuntimeError'


@pytest.mark.asyncio
async def test_cleanup_failure_without_primary_stays_fatal(monkeypatch):
    from nexus.extensions.agents.claude_code.native_bridge.adapters.posix import BasePosixClaudeNativeBridgeAdapter

    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    adapter.runtime = SimpleNamespace(run_command=AsyncMock(return_value=SimpleNamespace(return_code=1)))
    monkeypatch.setattr(
        BasePosixClaudeNativeBridgeAdapter, '_run_cleanup', AsyncMock(side_effect=RuntimeError('cleanup'))
    )
    with pytest.raises(RuntimeError, match='cleanup'):
        await adapter._run_cleanup(_SessionPaths('/tmp/test'), terminate=True)


@pytest.mark.asyncio
async def test_cancelled_invocation_preserves_cancellation_after_diagnostics(monkeypatch, tmp_path):
    from nexus.extensions.agents.claude_code.native_bridge.adapters.posix import BasePosixClaudeNativeBridgeAdapter

    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    adapter.diagnostic_dir = tmp_path
    adapter.runtime = SimpleNamespace(run_command=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout='tail')))
    cleanup = AsyncMock(side_effect=RuntimeError('secondary cleanup error'))
    monkeypatch.setattr(BasePosixClaudeNativeBridgeAdapter, '_run_cleanup', cleanup)
    started = asyncio.Event()

    async def invocation():
        try:
            started.set()
            await asyncio.Future()
        finally:
            await adapter._run_cleanup(_SessionPaths('/tmp/test'), terminate=True)

    task = asyncio.create_task(invocation())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert adapter.diagnostics['stdout'] == 'tail'
    assert adapter.diagnostics['primary_error_type'] == 'CancelledError'
    cleanup.assert_awaited_once()
    assert (tmp_path / 'cc_stream.jsonl').read_text() == 'tail'
    saved = json.loads((tmp_path / 'cc_diagnostics.json').read_text())
    assert saved['primary_error_type'] == 'CancelledError'
    assert saved['cleanup_error_type'] == 'RuntimeError'


@pytest.mark.asyncio
async def test_profile_recovers_partial_output_without_overwriting_primary(monkeypatch, tmp_path):
    runtime = object.__new__(AGSOSWorldRuntime)
    raw = json.dumps({'type': 'result', 'subtype': 'max_turns', 'result': 'partial'})

    async def runner(**kwargs):
        adapter = kwargs['adapter_factory'](runtime)
        adapter.diagnostics = {'stdout': raw, 'stderr': 'Reached max turns', 'primary_error_type': 'ConnectionError'}
        return {'success': False, 'exit_code': -1, 'output': '', 'stderr': 'Failed to run Claude Code: disconnected'}

    monkeypatch.setattr(agent_runtime, 'setup_and_run_native_bridge_claude_code', runner)
    profile = OSWorldClaudeExecutionProfile(
        {'guide_text': '', 'working_dir': '/tmp'},
        {'trajectory_dir': str(tmp_path / 'os__test'), 'osworld_domain': 'os', 'osworld_task_id': 'test'},
        'Task',
    )
    result = await profile.execute(runtime=runtime, launch=request().launch, timeout=15)
    assert result['success'] is False
    assert result['exit_status'] == 'tool_error'
    assert result['stderr'].startswith('Failed to run Claude Code: disconnected')
    assert result['output'] == raw
    assert result['cc_return_code'] == -1
    assert (tmp_path / 'os__test' / 'cc_stream.jsonl').read_text() == raw


@pytest.mark.asyncio
async def test_observation_failure_does_not_change_baseline_poll_or_cleanup(monkeypatch):
    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    adapter.diagnostics = {}
    adapter.runtime = SimpleNamespace(run_command=AsyncMock(side_effect=ConnectionError('observation failed')))
    await adapter._observe('before_agent')
    assert adapter.diagnostics['before_agent']['error_type'] == 'ConnectionError'
    runtime = SimpleNamespace(
        run_command=AsyncMock(
            return_value=SimpleNamespace(return_code=0, status='Finished', stdout='GONE\n', stderr='')
        )
    )
    adapter.runtime = runtime
    assert await adapter._poll_until_done(_SessionPaths('/tmp/test'), 10) == (None, False, True)
    # GONE re-checks the exit-code sentinel exactly once (race between process
    # exit and the file landing), then gives up — not the removed three-probe loop.
    assert runtime.run_command.await_count == 2
    assert adapter.runtime is runtime
    from nexus.extensions.agents.claude_code.native_bridge.adapters.posix import BasePosixClaudeNativeBridgeAdapter

    # The task adapter may take an opt-in artifact snapshot, but must delegate
    # the exact same paths/termination policy to the generic cleanup.
    cleanup = AsyncMock()
    monkeypatch.setattr(BasePosixClaudeNativeBridgeAdapter, '_cleanup_session', cleanup)
    paths = _SessionPaths('/tmp/test')
    await adapter._cleanup_session(paths, terminate=True)
    cleanup.assert_awaited_once_with(paths, terminate=True)
    assert (
        OSWorldClaudeNativeBridgeAdapter._build_kill_command is BasePosixClaudeNativeBridgeAdapter._build_kill_command
    )


@pytest.mark.parametrize(
    ('command', 'deny'),
    [
        ('pkill -f code', True),
        ('killall chrome', True),
        ('pgrep -f x | xargs kill', True),
        ('python3 -c "os.kill(12,15); print(\'/proc\')"', True),
        ('printf OK', False),
        ('ls /proc', False),
        ('kill "$(cat /tmp/child.pid)"', False),
    ],
)
def test_guard_policy_and_redacted_audit(tmp_path, command, deny):
    script = tmp_path / 'g'
    script.write_text(_BASH_PROCESS_GUARD)
    result = subprocess.run(
        [sys.executable, '-I', str(script)],
        input=json.dumps({'tool_use_id': 'x', 'tool_input': {'command': command}}),
        capture_output=True,
        text=True,
        check=True,
    )
    if deny:
        assert json.loads(result.stdout)['hookSpecificOutput']['permissionDecision'] == 'deny'
    else:
        assert result.stdout == ''
    event = json.loads((tmp_path / 'guard.jsonl').read_text())
    assert event['decision'] == ('deny' if deny else 'continue')
    assert 'command' not in event


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('confirmed', 'execution_error', 'terminal_error', 'expected'),
    [(True, False, False, True), (False, False, False, False), (True, True, False, False), (True, False, True, False)],
)
async def test_wall_budget_is_evaluable_only_after_confirmed_stop(
    monkeypatch, tmp_path, confirmed, execution_error, terminal_error, expected
):
    async def runner(**kwargs):
        adapter = kwargs['adapter_factory'](kwargs['runtime'])
        adapter.diagnostics = {
            'budget_reason': 'cc_timeout',
            'budget_stop_confirmed': confirmed,
            'budget_execution_error': execution_error,
        }
        raw = json.dumps(
            {
                'type': 'result',
                'subtype': 'success',
                'is_error': True,
                'api_error_status': 401,
                'result': 'API rejected',
            }
            if terminal_error
            else {'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': 'TASK_DONE'}]}}
        )
        return {'success': False, 'exit_code': -1, 'output': raw, 'stderr': '[timeout]'}

    monkeypatch.setattr(agent_runtime, 'setup_and_run_native_bridge_claude_code', runner)
    profile = OSWorldClaudeExecutionProfile(
        {'guide_text': '', 'working_dir': '/tmp'}, {'trajectory_dir': str(tmp_path)}, 'Task'
    )
    result = await profile.execute(runtime=object.__new__(AGSOSWorldRuntime), launch=request().launch, timeout=1)
    assert result['success'] is expected
    assert result['exit_status'] == ('max_iterations' if expected else 'tool_error')
    assert result['cc_return_code'] == -1  # Never fabricate CLI exit 0 or DONE.
    from nexus.extensions.tasks.os_world.task import _marker_from_agent_output

    assert _marker_from_agent_output(SimpleNamespace(**result)) is None
    assert json.loads((tmp_path / 'cc_trajectory.json').read_text())['success'] is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('return_code', 'status', 'stdout', 'confirmed'),
    [
        (0, 'Finished', '{"confirmed": true}', True),
        (1, 'Finished', '', False),
        (0, 'Failed', '{"confirmed": true}', False),
        (0, 'Finished', '', False),
        (0, 'Finished', '{"confirmed": false}', False),
    ],
)
async def test_wall_budget_requires_valid_stop_evidence(monkeypatch, tmp_path, return_code, status, stdout, confirmed):
    from nexus.extensions.agents.claude_code.native_bridge.adapters.posix import BasePosixClaudeNativeBridgeAdapter

    monkeypatch.setattr(
        BasePosixClaudeNativeBridgeAdapter, '_poll_until_done', AsyncMock(return_value=(None, True, False))
    )
    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    adapter.diagnostics = {}
    adapter.diagnostic_dir = tmp_path
    adapter._capture_diagnostics = AsyncMock()
    adapter.runtime = SimpleNamespace(
        run_command=AsyncMock(
            return_value=SimpleNamespace(
                return_code=return_code, status=status, stdout=stdout, stderr='helper evidence'
            )
        )
    )
    if confirmed:
        assert await adapter._poll_until_done(_SessionPaths('/tmp/test'), 10) == (None, True, False)
        assert adapter.diagnostics['budget_stop_confirmed']
    else:
        with pytest.raises(RuntimeError, match='CC time-budget termination'):
            await adapter._poll_until_done(_SessionPaths('/tmp/test'), 10)
        assert adapter.diagnostics['budget_execution_error']
        assert not adapter.diagnostics.get('budget_stop_confirmed')
    evidence = json.loads((tmp_path / 'cc_diagnostics.json').read_text())
    assert evidence['budget_termination'] == {
        'return_code': return_code,
        'stdout': stdout,
        'stderr': 'helper evidence',
    }


@pytest.mark.asyncio
async def test_transport_timeout_is_not_wall_budget(monkeypatch):
    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    adapter.diagnostics = {}
    adapter.runtime = SimpleNamespace(run_command=AsyncMock(side_effect=TimeoutError('transport')))
    with pytest.raises(TimeoutError, match='transport'):
        await adapter._poll_until_done(_SessionPaths('/tmp/unused'), 100)
    assert 'budget_reason' not in adapter.diagnostics


@pytest.mark.asyncio
async def test_unavailable_sandbox_at_deadline_is_not_scored(monkeypatch):
    from nexus.extensions.agents.claude_code.native_bridge.adapters.posix import BasePosixClaudeNativeBridgeAdapter

    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    adapter.diagnostics = {}
    adapter.runtime = SimpleNamespace(
        run_command=AsyncMock(return_value=SimpleNamespace(return_code=0, status='Finished', stdout='GONE', stderr=''))
    )

    async def blocked(self, paths, timeout):
        await asyncio.sleep(100)

    monkeypatch.setattr(BasePosixClaudeNativeBridgeAdapter, '_poll_until_done', blocked)
    with pytest.raises(RuntimeError, match='unavailable'):
        await adapter._poll_until_done(_SessionPaths('/tmp/unused'), 0.01)
    assert not adapter.diagnostics.get('budget_stop_confirmed')


def test_api_error_has_priority_over_max_turns_text():
    from nexus.extensions.tasks.os_world.execution.claude_code import _terminal_api_failure

    assert _terminal_api_failure(
        json.dumps({
            'type': 'result',
            'subtype': 'success',
            'is_error': True,
            'result': 'Reached max turns',
        })
    )
    assert not _terminal_api_failure(
        json.dumps({
            'type': 'result',
            'subtype': 'error_max_turns',
            'is_error': True,
        })
    )


@pytest.mark.asyncio
async def test_guard_installed_only_when_explicitly_requested(tmp_path):
    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    adapter.runtime = LocalRuntime()
    adapter.diagnostics = {}
    original = request()
    assert await adapter._install_guard(original, str(tmp_path)) is original
    assert not adapter.runtime.uploads
    adapter.process_guard = True
    guarded = await adapter._install_guard(original, str(tmp_path))
    assert guarded.extra_cli_args == ['--settings', str(tmp_path / 'settings.json')]
    config = json.loads((tmp_path / 'settings.json').read_text())
    assert config['hooks']['PreToolUse'][0]['matcher'] == 'Bash'
    assert 'ExitPlanMode' not in json.dumps(config)
