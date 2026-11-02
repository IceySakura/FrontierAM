"""OSWorld codex execution regressions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.agents.codex.native_bridge.adapters import osworld as codex_adapter
from nexus.extensions.agents.codex.native_bridge.adapters.posix import _SessionPaths
from nexus.extensions.agents.codex.native_bridge.adapters.types import CodexRunRequest
from nexus.extensions.tasks.os_world.execution.codex import TaskCodexAdapter, _ExecutionError
from nexus.runtime.bridge.adapters.base import BridgeLaunchResult
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


@pytest.mark.asyncio
@pytest.mark.parametrize('version', ['16.20.2', '22.0.0'])
async def test_node_preparation_returns_validated_guest_path(version):
    payload = {'PATH': '/opt/node/bin:/usr/bin', 'node': '/opt/node/bin/node', 'version': version}
    runtime = SimpleNamespace(
        run_command=AsyncMock(
            return_value=SimpleNamespace(
                status=CommandRunStatus.Finished,
                return_code=0,
                stdout='noise\nNEXUS_CODEX_NODE_ENV=' + json.dumps(payload),
            )
        )
    )
    assert await codex_adapter.prepare_node_env(runtime) == {'PATH': payload['PATH']}
    runtime.run_command.assert_awaited_once_with(command=codex_adapter.NODE_BOOTSTRAP, timeout=900.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'payload',
    [
        '{}',
        'not-json',
        '{"PATH":"/usr/bin","node":"node","version":"22"}',
        '{"PATH":"/usr/bin","node":"/bin/node","version":"14.0"}',
        '{"PATH":"bad\\npath","node":"/bin/node","version":"22"}',
    ],
)
async def test_node_preparation_rejects_invalid_guest_environment(payload):
    runtime = SimpleNamespace(
        run_command=AsyncMock(
            return_value=SimpleNamespace(
                status=CommandRunStatus.Finished,
                return_code=0,
                stdout='NEXUS_CODEX_NODE_ENV=' + payload,
            )
        )
    )
    with pytest.raises(RuntimeError, match='invalid environment'):
        await codex_adapter.prepare_node_env(runtime)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('rc', 'stdout', 'message'),
    [
        (1, '', 'preparation failed'),
        (0, 'no marker', 'did not report'),
    ],
)
async def test_node_preparation_reports_failed_or_missing_result(rc, stdout, message):
    runtime = SimpleNamespace(
        run_command=AsyncMock(
            return_value=SimpleNamespace(
                status=CommandRunStatus.Finished,
                return_code=rc,
                stdout=stdout,
            )
        )
    )
    with pytest.raises(RuntimeError, match=message):
        await codex_adapter.prepare_node_env(runtime)


@pytest.mark.asyncio
@pytest.mark.parametrize('enabled', [False, True])
async def test_node_preparation_is_scoped_to_benchmark_install(monkeypatch, enabled):
    adapter = object.__new__(codex_adapter.OSWorldCodexNativeBridgeAdapter)
    adapter.runtime = object()
    adapter.benchmark_policy = enabled
    adapter._has_npm = AsyncMock(return_value=True)
    prepare = AsyncMock(return_value={'PATH': '/opt/node/bin:/usr/bin'})
    monkeypatch.setattr(codex_adapter, 'prepare_node_env', prepare)
    await adapter.prepare_install(SimpleNamespace(binary_url=None))
    assert prepare.await_count == int(enabled)
    if enabled:
        assert adapter.execution_env == {'PATH': '/opt/node/bin:/usr/bin'}


class LocalRuntime:
    def __init__(self):
        self.commands = []

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
        Path(path).write_bytes(content)
        return SimpleNamespace(status_code=200)


def adapter(runtime, diagnostic_dir=None):
    result = object.__new__(TaskCodexAdapter)
    result.runtime = runtime
    result.execution_env = {}
    result.diagnostics = {}
    result.diagnostic_dir = diagnostic_dir
    result.poll_interval = 0.01
    return result


def request(executable='/bin/false', task='prompt', working_dir='/tmp'):
    return CodexRunRequest(
        codex_executable=executable,
        launch=BridgeLaunchResult(base_url='http://test', local_base_url='http://test'),
        task=task,
        working_dir=working_dir,
        timeout=10,
        version='0.154.0',
        npm_registry=None,
        strict_cleanup=True,
        developer_instructions='developer instructions',
    )


@pytest.mark.asyncio
async def test_actual_shell_uses_stdin_and_preserves_identities_before_cleanup(tmp_path):
    executable = tmp_path / 'fake-codex'
    executable.write_text(
        f'#!{sys.executable}\n'
        'import sys,os,json,hashlib,time\n'
        'raw=sys.stdin.buffer.read()\n'
        'assert sys.argv[-1]=="-"\n'
        'assert raw not in open("/proc/self/cmdline","rb").read()\n'
        'assert raw not in open(f"/proc/{os.getppid()}/cmdline","rb").read()\n'
        'time.sleep(0.15)\n'
        'print(json.dumps({"type":"turn.completed","hash":hashlib.sha256(raw).hexdigest()}))\n'
        'open(sys.argv[sys.argv.index("-o")+1],"w").write("done")\n'
    )
    executable.chmod(0o700)
    runtime = LocalRuntime()
    worker = adapter(runtime, tmp_path / 'diagnostics')
    marker = 'prompt-marker-' + os.urandom(8).hex()
    prompt = marker + '中文 chrome gimp " $HOME\n' * 15000
    result = await worker.run_codex(request(str(executable), prompt, str(tmp_path)))
    assert result.exit_code == 0, result.stderr
    assert result.output == 'done'
    assert (
        json.loads(result.stdout)['hash']
        == hashlib.sha256(('developer instructions\n\n' + prompt).encode()).hexdigest()
    )
    assert all(marker not in command for command in runtime.commands)
    evidence = json.loads((tmp_path / 'diagnostics/codex_diagnostics.json').read_text())
    assert evidence['wrapper_pid'].strip().isdigit()
    assert evidence['codex_pid'].strip().isdigit()
    assert evidence['wrapper_start'].isdigit()
    assert evidence['codex_start'].isdigit()
    assert evidence['exit_code_file'].strip() == '0'
    assert evidence['last_events'][-1]['type'] == 'turn.completed'
    assert evidence['captured_before_cleanup_at'] <= evidence['cleanup_finished_at']
    assert not Path(evidence['session_dir']).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('states', 'stage'),
    [
        (['GONE'] * 3, 'process_missing'),
        (['PROBE_FAILED'] * 3, 'probe'),
        (['PID_METADATA_INVALID'] * 3, 'probe'),
        (['PID_REUSED'] * 3, 'probe'),
        (['EXIT_INVALID'] * 3, 'probe'),
        (['GONE', 'PROBE_FAILED', 'GONE'], 'probe'),
    ],
)
async def test_only_repeated_successful_gone_probes_confirm_disappearance(states, stage):
    worker = adapter(object())
    worker._probe = AsyncMock(side_effect=[{'state': state} for state in states])
    with pytest.raises(_ExecutionError) as caught:
        await worker._poll(_SessionPaths('/tmp/test'), 5)
    assert caught.value.stage == stage
    assert worker._probe.await_count == 3


@pytest.mark.asyncio
async def test_transient_probe_failure_recovers_without_relaunch():
    worker = adapter(object())
    worker._probe = AsyncMock(
        side_effect=[
            {'state': 'PROBE_FAILED'},
            {'state': 'RUNNING'},
            {'state': 'DONE', 'exit_code': 0},
        ]
    )
    assert await worker._poll(_SessionPaths('/tmp/test'), 5) == 0


@pytest.mark.asyncio
async def test_error_command_stdout_is_not_trusted_as_gone():
    runtime = SimpleNamespace(
        run_command=AsyncMock(
            return_value=RunCommandResponse(
                status='Error',
                return_code=1,
                stdout='{"state":"GONE"}',
                stderr='transport error',
            )
        )
    )
    worker = adapter(runtime)
    assert (await worker._probe(_SessionPaths('/tmp/test')))['state'] == 'PROBE_FAILED'
    assert worker.diagnostics['probes'][-1]['return_code'] == 1
    assert worker.diagnostics['probes'][-1]['stdout'] == '{"state":"GONE"}'


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['timeout', 'process_missing', 'probe'])
async def test_failure_is_captured_before_termination_and_does_not_get_masked(tmp_path, stage):
    runtime = SimpleNamespace(
        run_command=AsyncMock(
            return_value=RunCommandResponse(
                status='Finished',
                return_code=0,
                stdout='LAUNCHED',
            )
        )
    )
    worker = adapter(runtime, tmp_path)
    worker._prepare = AsyncMock()
    worker._poll = AsyncMock(side_effect=_ExecutionError(stage, 'primary failure'))
    worker._probe = AsyncMock(return_value={'state': 'RUNNING', 'wrapper': {'matches': True, 'pid': 123, 'pgid': 123}})
    events = []

    async def capture(paths):
        events.append('capture')
        worker.diagnostics['stdout_tail'] = '{"type":"item.completed"}'
        worker.diagnostics['stderr_tail'] = 'saved stderr'
        worker._save()

    async def terminate(paths):
        events.append('terminate')
        assert (tmp_path / 'codex_diagnostics.json').exists()
        raise RuntimeError('secondary termination failure')

    worker._capture = capture
    worker._terminate_session = terminate
    result = await worker.run_codex(request())
    assert events == ['capture', 'terminate']
    assert result.failure_stage == stage
    assert result.failure_kind == 'infrastructure'
    assert result.timed_out is (stage == 'timeout')
    assert 'primary failure' in result.stderr
    assert 'saved stderr' in result.stderr
    assert result.stdout
    assert worker.diagnostics['cleanup_error']['type'] == 'RuntimeError'
    # The only execution command was the initial launch, never a replayed GUI action.
    assert runtime.run_command.await_count == 1


@pytest.mark.asyncio
async def test_cancelled_run_captures_before_cleanup_and_propagates():
    runtime = SimpleNamespace(
        run_command=AsyncMock(
            return_value=RunCommandResponse(
                status='Finished',
                return_code=0,
                stdout='LAUNCHED',
            )
        )
    )
    worker = adapter(runtime)
    worker._prepare = AsyncMock()
    worker._poll = AsyncMock(side_effect=asyncio.CancelledError())
    worker._capture = AsyncMock()
    worker._probe = AsyncMock(return_value={'state': 'PROBE_FAILED'})
    with pytest.raises(asyncio.CancelledError):
        await worker.run_codex(request())
    worker._capture.assert_awaited_once()
    assert worker.diagnostics['failure_kind'] == 'cancelled'
    assert worker.diagnostics['termination_unconfirmed'] is True


@pytest.mark.asyncio
async def test_upload_failure_prevents_launch(tmp_path):
    runtime = LocalRuntime()
    runtime.setup_upload_bytes = AsyncMock(return_value=SimpleNamespace(status_code=500))
    worker = adapter(runtime, tmp_path / 'diagnostics')
    result = await worker.run_codex(request(working_dir=str(tmp_path)))
    assert result.failure_stage == 'staging'
    assert result.failure_kind == 'infrastructure'
    assert not any('nohup' in command for command in runtime.commands)


def test_cleanup_targets_verified_wrapper_pid_not_launcher_pid():
    worker = adapter(object())
    paths = _SessionPaths('/tmp/safe-session')
    command = worker._build_kill_command(paths)
    assert '/tmp/safe-session/wrapper.pid' in command
    assert '/tmp/safe-session/pid ' not in command


@pytest.mark.asyncio
async def test_invalid_pid_is_not_confirmed_process_disappearance(tmp_path):
    paths = _SessionPaths(str(tmp_path))
    (tmp_path / 'wrapper.pid').write_text('not-a-pid')
    worker = adapter(LocalRuntime())
    result = await worker._probe(paths)
    assert result['state'] == 'PID_METADATA_INVALID'


@pytest.mark.asyncio
@pytest.mark.parametrize('exit_code', [0, 1])
@pytest.mark.parametrize('cleanup_timeout', [False, True])
async def test_cleanup_failure_preserves_execution_result(tmp_path, exit_code, cleanup_timeout):
    runtime = SimpleNamespace(
        run_command=AsyncMock(
            return_value=RunCommandResponse(
                status='Finished',
                return_code=0,
                stdout='LAUNCHED',
            )
        )
    )
    worker = adapter(runtime, tmp_path)
    worker._prepare = AsyncMock()
    worker._poll = AsyncMock(return_value=exit_code)
    worker._capture = AsyncMock()
    runtime.run_command.side_effect = [
        RunCommandResponse(status='Finished', return_code=0, stdout='LAUNCHED'),
        RunCommandResponse(status='Finished', return_code=0, stdout='{"type":"turn.completed"}'),
        RunCommandResponse(status='Finished', return_code=0, stdout=''),
        RunCommandResponse(status='Finished', return_code=0, stdout='done'),
        TimeoutError() if cleanup_timeout else RunCommandResponse(status='Error', return_code=1, stdout=''),
    ]
    result = await worker.run_codex(request())
    assert result.failure_stage == ('process' if exit_code else None)
    assert result.failure_kind == ('agent' if exit_code else None)
    assert result.exit_code == exit_code
    assert result.stdout == '{"type":"turn.completed"}'
    assert result.output == 'done'
    assert result.stderr == ''
    assert worker.diagnostics['exit_code'] == exit_code
    assert worker.diagnostics['return_code'] == exit_code
    assert worker.diagnostics['cleanup_error']['type'] == ('TimeoutError' if cleanup_timeout else 'RuntimeError')


@pytest.mark.asyncio
async def test_profile_can_proceed_to_scoring_after_directory_cleanup_timeout(monkeypatch, tmp_path):
    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.os_world.execution.codex_profile import OSWorldCodexExecutionProfile

    stdout = '{"type":"item.completed","item":{"type":"agent_message","text":"TASK_DONE"}}\n{"type":"turn.completed"}'
    runtime = SimpleNamespace(
        osworld_use_proxy=False,
        run_command=AsyncMock(
            side_effect=[
                RunCommandResponse(status='Finished', return_code=0, stdout='codex-cli 0.154.0'),
                RunCommandResponse(status='Finished', return_code=0, stdout='LAUNCHED'),
                RunCommandResponse(status='Finished', return_code=0, stdout=stdout),
                RunCommandResponse(status='Finished', return_code=0, stdout=''),
                RunCommandResponse(status='Finished', return_code=0, stdout='TASK_DONE'),
                TimeoutError('directory cleanup timed out'),
            ]
        ),
    )
    worker = adapter(runtime, tmp_path)
    worker.ensure_codex = AsyncMock(return_value='/bin/codex')
    worker._prepare = AsyncMock()
    worker._poll = AsyncMock(return_value=0)
    worker._capture = AsyncMock()
    monkeypatch.setattr('nexus.extensions.tasks.os_world.execution.codex.TaskCodexAdapter', lambda *a, **kw: worker)
    monkeypatch.setattr(
        'nexus.extensions.tasks.os_world.execution.codex_profile.require_ags_osworld_runtime', lambda r: r
    )
    profile = OSWorldCodexExecutionProfile(
        {'guide_text': 'Complete the task.', 'model_catalog_preset': 'gpt6_catalog_v8'}, {}, 'fixture'
    )
    async with ExecutionContext(task_name='os_world') as ctx:
        result = await profile.execute(runtime=runtime, launch=request().launch, timeout=10)
        assert ctx.tool_state['osworld_agent_output'].exit_status == 'finished'
    assert result['success'] is True
    assert result['exit_code'] == 0
    assert result['final_answer'] == 'TASK_DONE'
    assert worker.diagnostics['cleanup_error']['type'] == 'TimeoutError'


@pytest.mark.asyncio
async def test_confirmed_timeout_records_evidence_without_relaunch(tmp_path):
    runtime = SimpleNamespace(
        run_command=AsyncMock(
            return_value=RunCommandResponse(
                status='Finished',
                return_code=0,
                stdout='LAUNCHED',
            )
        )
    )
    worker = adapter(runtime, tmp_path)
    worker._prepare = AsyncMock()
    worker._poll = AsyncMock(side_effect=_ExecutionError('timeout', 'budget exhausted'))
    worker._probe = AsyncMock(return_value={'wrapper': {'matches': True, 'pid': 123, 'pgid': 123}})
    worker._capture = AsyncMock()
    worker._terminate_session = AsyncMock()
    result = await worker.run_codex(request())
    assert worker.diagnostics['termination_confirmed'] is True
    worker._terminate_session.assert_awaited_once()
    worker._poll.assert_awaited_once()
    # Only initial launch and directory cleanup; no task replay or new kill.
    assert runtime.run_command.await_count == 2
    assert result.timed_out


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['context', 'api', 'timeout', 'unconfirmed_timeout', 'install', 'finished'])
async def test_profile_records_standard_outcome_before_return_or_raise(monkeypatch, outcome):
    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.os_world.execution.codex_profile import OSWorldCodexOptions, _execute_osworld

    runtime = SimpleNamespace(
        run_command=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout='codex-cli 0.154.0')),
        osworld_use_proxy=False,
    )
    options = OSWorldCodexOptions(user_prompt='fixture', guide_version='v1', model_catalog_preset='gpt6_catalog_v8')
    stdout = {
        'context': '{"type":"turn.failed","error":{"message":"[context_length_exceeded] context full"}}',
        'api': '{"type":"turn.failed","error":{"message":"upstream unavailable"}}',
        'finished': '{"type":"item.completed","item":{"type":"agent_message","text":"TASK_DONE"}}\n'
        '{"type":"turn.completed"}',
    }.get(outcome, '')
    worker = SimpleNamespace(
        ensure_codex=AsyncMock(
            return_value='/bin/codex', side_effect=RuntimeError('npm unavailable') if outcome == 'install' else None
        ),
        execution_env={},
        diagnostics={'termination_confirmed': outcome == 'timeout'},
        run_codex=AsyncMock(
            return_value=SimpleNamespace(
                exit_code=0 if outcome == 'finished' else -1,
                stdout=stdout,
                stderr='fixture stderr',
                timed_out='timeout' in outcome,
                failure_kind=None if outcome == 'finished' else 'infrastructure',
                failure_stage='timeout' if 'timeout' in outcome else 'process',
            )
        ),
    )
    monkeypatch.setattr('nexus.extensions.tasks.os_world.execution.codex.TaskCodexAdapter', lambda *a, **kw: worker)
    monkeypatch.setattr(
        'nexus.extensions.tasks.os_world.execution.codex_profile.require_ags_osworld_runtime', lambda r: r
    )
    ctx = ExecutionContext(task_name='os_world')
    async with ctx:
        if outcome in {'api', 'unconfirmed_timeout', 'install'}:
            with pytest.raises(RuntimeError):
                await _execute_osworld(runtime=runtime, launch=request().launch, timeout=10, inp=options)
        else:
            await _execute_osworld(runtime=runtime, launch=request().launch, timeout=10, inp=options)
        stage = ctx.tool_state['osworld_agent_output']
    assert (
        stage.exit_status
        == {
            'context': 'context_length_exceeded',
            'api': 'inference_error',
            'timeout': 'agent_timeout',
            'unconfirmed_timeout': 'runtime_error_by_env',
            'install': 'environment_error',
            'finished': 'finished',
        }[outcome]
    )
