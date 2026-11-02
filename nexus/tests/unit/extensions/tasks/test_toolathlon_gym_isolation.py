from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nexus.extensions.tasks.mcp_benchmark.evaluators.toolathlon import ToolathlonRuntimeEvaluator
from nexus.extensions.tasks.toolathlon_gym import task as toolathlon_task
from nexus.runtime.executors.command import CommandRunStatus


def _command_result(*, return_code: int = 0, stdout: str = '', stderr: str = '') -> SimpleNamespace:
    return SimpleNamespace(
        status=CommandRunStatus.Finished,
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
    )


def test_legacy_task_without_binding_defaults_to_no_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    # The legacy env switch is deliberately ignored. A row must carry
    # data_binding to opt into the new topology.
    monkeypatch.setenv('TOOLATHLON_GYM_ISOLATE_MCP_SERVER', '1')
    assert toolathlon_task._binding_isolation(None) is None


def test_binding_selects_isolation_without_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('TOOLATHLON_GYM_ISOLATE_MCP_SERVER', raising=False)
    assert toolathlon_task._binding_isolation({'isolation': 'backend_mock'}) == 'backend_mock'
    assert toolathlon_task._binding_isolation({'isolation': 'split_workspace'}) == 'split_workspace'


@pytest.mark.asyncio
async def test_workspace_sync_copies_archive_between_runtimes() -> None:
    class SourceRuntime:
        async def run_command(self, **kwargs):
            assert kwargs['command'].endswith('tar czf - . | base64 -w0')
            return _command_result(stdout='dGVzdC10YXItZGF0YQ==\n')

    class DestinationRuntime:
        def __init__(self) -> None:
            self.commands: list[str] = []
            self.uploads = []

        async def run_command(self, **kwargs):
            self.commands.append(kwargs['command'])
            return _command_result()

        async def upload_file(self, items):
            self.uploads.extend(items)
            return SimpleNamespace(results=[SimpleNamespace(success=True, message='')])

    destination = DestinationRuntime()
    await toolathlon_task._sync_workspace_dir(
        SourceRuntime(),
        destination,
        remote_dir='/workspace/case/agent_workspace',
    )

    assert destination.uploads[0].path == '/workspace/case/agent_workspace/__ws_sync.tar.gz'
    assert destination.uploads[0].content == 'dGVzdC10YXItZGF0YQ=='
    assert destination.uploads[0].encoding == 'base64'
    assert any('tar -xzf' in command for command in destination.commands)


class _RecordingDestination:
    """Destination runtime that records whether it was mutated."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.uploads: list[Any] = []

    async def run_command(self, **kwargs):
        self.commands.append(kwargs['command'])
        return _command_result()

    async def upload_file(self, items):
        self.uploads.extend(items)
        return SimpleNamespace(results=[SimpleNamespace(success=True, message='')])

    @property
    def destructive_commands(self) -> list[str]:
        return [command for command in self.commands if 'rm -rf' in command]


@pytest.mark.asyncio
async def test_workspace_sync_raises_when_pack_times_out() -> None:
    """A timed-out pack must never be mistaken for an empty workspace."""

    class TimingOutSource:
        async def run_command(self, **kwargs):
            return SimpleNamespace(
                status=CommandRunStatus.TimeLimitExceeded,
                return_code=None,
                stdout='',
                stderr='',
            )

    destination = _RecordingDestination()
    with pytest.raises(RuntimeError, match='failed to pack workspace'):
        await toolathlon_task._sync_workspace_dir(
            TimingOutSource(),
            destination,
            remote_dir='/workspace/case/agent_workspace',
            clean_destination=True,
        )

    # The regression this guards: clean_destination wiped the destination before
    # the source archive was validated, destroying the workspace it mirrors.
    assert destination.destructive_commands == []
    assert destination.uploads == []


@pytest.mark.asyncio
async def test_workspace_sync_raises_when_pack_returns_nonzero() -> None:
    class FailingSource:
        async def run_command(self, **kwargs):
            return _command_result(return_code=2, stderr='tar: write error: No space left on device')

    destination = _RecordingDestination()
    with pytest.raises(RuntimeError, match='No space left on device'):
        await toolathlon_task._sync_workspace_dir(
            FailingSource(),
            destination,
            remote_dir='/workspace/case/agent_workspace',
            clean_destination=True,
        )

    assert destination.destructive_commands == []


@pytest.mark.asyncio
async def test_workspace_sync_raises_on_empty_archive() -> None:
    """`tar czf - .` yields a valid archive even for an empty dir, so an empty
    payload is always a failure and must not silently skip the copy."""

    class EmptyOutputSource:
        async def run_command(self, **kwargs):
            return _command_result(return_code=0, stdout='   \n')

    destination = _RecordingDestination()
    with pytest.raises(RuntimeError, match='unexpectedly empty'):
        await toolathlon_task._sync_workspace_dir(
            EmptyOutputSource(),
            destination,
            remote_dir='/workspace/case/agent_workspace',
            clean_destination=True,
        )

    assert destination.destructive_commands == []


@pytest.mark.asyncio
async def test_workspace_sync_cleans_destination_only_after_pack_succeeds() -> None:
    class SourceRuntime:
        async def run_command(self, **kwargs):
            return _command_result(stdout='dGVzdC10YXItZGF0YQ==\n')

    destination = _RecordingDestination()
    await toolathlon_task._sync_workspace_dir(
        SourceRuntime(),
        destination,
        remote_dir='/workspace/case/agent_workspace',
        clean_destination=True,
    )

    # On the success path the wipe still happens, and it precedes extraction so
    # the mirror stays exact (deletions propagate).
    assert len(destination.destructive_commands) == 1
    wipe_index = next(i for i, c in enumerate(destination.commands) if 'rm -rf' in c)
    extract_index = next(i for i, c in enumerate(destination.commands) if 'tar -xzf' in c)
    assert wipe_index < extract_index


@pytest.mark.asyncio
async def test_bound_artifact_sync_only_copies_declared_outputs() -> None:
    class SourceRuntime:
        async def run_command(self, **kwargs):
            command = kwargs['command']
            assert 'result.json' in command
            assert 'secret.db' not in command
            assert '.python_tmp' not in command
            return _command_result(stdout='dGVzdC10YXItZGF0YQ==\n')

    class DestinationRuntime:
        def __init__(self) -> None:
            self.commands: list[str] = []
            self.uploads = []

        async def run_command(self, **kwargs):
            self.commands.append(kwargs['command'])
            return _command_result()

        async def upload_file(self, items):
            self.uploads.extend(items)
            return SimpleNamespace(results=[SimpleNamespace(success=True, message='')])

    destination = DestinationRuntime()
    await toolathlon_task._sync_workspace_artifacts(
        SourceRuntime(),
        destination,
        remote_dir='/workspace/case/agent_workspace',
        artifacts=['result.json'],
    )

    assert destination.uploads[0].path.endswith('/__artifact_sync.tar.gz')
    assert any('tar -xzf' in command for command in destination.commands)


@pytest.mark.asyncio
async def test_bound_artifact_sync_skips_missing_output_without_failing() -> None:
    """A missing artifact is a rollout outcome, not an infra failure.

    Raising here turned every "agent forgot to write result.json" run into an
    UNHANDLED_ERROR with no reward, throwing away the checks it did satisfy.
    """

    class SourceRuntime:
        async def run_command(self, **kwargs):
            # The shell reports the gap on stderr but exits 0 with no archive.
            return _command_result(return_code=0, stdout='', stderr='missing output artifacts: result.json')

    destination = _RecordingDestination()
    await toolathlon_task._sync_workspace_artifacts(
        SourceRuntime(),
        destination,
        remote_dir='/workspace/case/agent_workspace',
        artifacts=['result.json'],
    )

    # Nothing was transferred, and crucially the task was allowed to continue to
    # evaluation so the evaluator can score the run.
    assert destination.uploads == []
    assert destination.commands == []


@pytest.mark.asyncio
async def test_bound_artifact_sync_still_fails_on_real_pack_error() -> None:
    """Genuine pack failures (timeout, permissions) must stay fatal."""

    class TimingOutSource:
        async def run_command(self, **kwargs):
            return SimpleNamespace(
                status=CommandRunStatus.TimeLimitExceeded,
                return_code=None,
                stdout='',
                stderr='',
            )

    with pytest.raises(RuntimeError, match='failed to pack bound output artifacts'):
        await toolathlon_task._sync_workspace_artifacts(
            TimingOutSource(),
            _RecordingDestination(),
            remote_dir='/workspace/case/agent_workspace',
            artifacts=['result.json'],
        )

    class FailingSource:
        async def run_command(self, **kwargs):
            return _command_result(return_code=1, stderr='tar: Cannot open: Permission denied')

    with pytest.raises(RuntimeError, match='Permission denied'):
        await toolathlon_task._sync_workspace_artifacts(
            FailingSource(),
            _RecordingDestination(),
            remote_dir='/workspace/case/agent_workspace',
            artifacts=['result.json'],
        )


@pytest.mark.asyncio
async def test_bound_artifact_sync_transfers_the_artifacts_that_exist() -> None:
    """With several declared artifacts, the present ones must still be copied."""

    class PartialSource:
        async def run_command(self, **kwargs):
            command = kwargs['command']
            # Both names are probed; only the archive contents differ at runtime.
            assert 'a.json' in command
            assert 'b.json' in command
            return _command_result(stdout='dGVzdC10YXItZGF0YQ==\n', stderr='missing output artifacts: b.json')

    destination = _RecordingDestination()
    await toolathlon_task._sync_workspace_artifacts(
        PartialSource(),
        destination,
        remote_dir='/workspace/case/agent_workspace',
        artifacts=['a.json', 'b.json'],
    )

    assert destination.uploads[0].path.endswith('/__artifact_sync.tar.gz')
    assert any('tar -xzf' in command for command in destination.commands)


@pytest.mark.asyncio
async def test_backend_mock_main_runtime_removes_baked_task_data() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.command = ''

        async def run_command(self, **kwargs):
            self.command = kwargs['command']
            return _command_result()

    runtime = Runtime()
    await toolathlon_task._prepare_backend_mock_agent_workspace(
        runtime,
        case_root='/workspace/mcp-benchmark/toolathlon/mock/case',
        agent_workspace='/workspace/mcp-benchmark/toolathlon/mock/case/agent_workspace',
    )

    assert 'rm -rf /workspace/mcp-benchmark/toolathlon/mock/case' in runtime.command
    assert '/opt/mcp-benchmarks/datasets/toolathlon/tasks' in runtime.command
    assert runtime.command.endswith('mkdir -p /workspace/mcp-benchmark/toolathlon/mock/case/agent_workspace')


def test_split_binding_applies_per_server_runtime() -> None:
    servers = [
        {'name': 'excel', 'runtime_name': 'old'},
        {'name': 'snowflake', 'runtime_name': 'old'},
        {'name': 'emails', 'runtime_name': 'old'},
    ]
    hybrids = toolathlon_task._apply_bound_server_placements(
        servers,
        placements={'excel': 'main', 'snowflake': 'backend', 'emails': 'hybrid'},
    )

    assert [server['runtime_name'] for server in servers] == ['main', 'mcp_backend', 'mcp_backend']
    assert hybrids == {'emails'}


@pytest.mark.asyncio
async def test_workspace_clean_mirror_preserves_deletions() -> None:
    class SourceRuntime:
        async def run_command(self, **kwargs):
            return _command_result(stdout='dGVzdC10YXItZGF0YQ==\n')

    class DestinationRuntime:
        def __init__(self) -> None:
            self.commands = []

        async def run_command(self, **kwargs):
            self.commands.append(kwargs['command'])
            return _command_result()

        async def upload_file(self, items):
            return SimpleNamespace(results=[SimpleNamespace(success=True, message='')])

    destination = DestinationRuntime()
    await toolathlon_task._sync_workspace_dir(
        SourceRuntime(),
        destination,
        remote_dir='/workspace/case/agent_workspace',
        clean_destination=True,
    )

    assert any('find /workspace/case/agent_workspace -mindepth 1' in command for command in destination.commands)


@pytest.mark.asyncio
async def test_hybrid_bridge_mirrors_before_and_after_call(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []

    class Manager:
        def __init__(self):
            self.tools = {'emails_send_email': SimpleNamespace(server_name='emails')}

        async def call_tool(self, name, arguments=None):
            events.append(('call', name, arguments))
            return _command_result()

    async def fake_sync(src, dst, *, remote_dir, clean_destination=False):
        events.append(('sync', src, dst, remote_dir, clean_destination))

    manager = Manager()
    ctx = SimpleNamespace(_mcp_manager=manager)
    monkeypatch.setattr(toolathlon_task, '_sync_workspace_dir', fake_sync)
    toolathlon_task._install_hybrid_workspace_bridge(
        ctx,
        hybrid_servers={'emails'},
        agent_runtime='main',
        backend_runtime='backend',
        agent_workspace='/workspace/case/agent_workspace',
    )

    await manager.call_tool('emails_send_email', {'attachments': ['report.xlsx']})

    assert events == [
        ('sync', 'main', 'backend', '/workspace/case/agent_workspace', True),
        ('call', 'emails_send_email', {'attachments': ['report.xlsx']}),
        ('sync', 'backend', 'main', '/workspace/case/agent_workspace', True),
    ]


@pytest.mark.asyncio
async def test_toolathlon_evaluator_honors_harbor_continuous_reward() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.commands: list[str] = []

        async def run_command(self, **kwargs):
            self.commands.append(kwargs['command'])
            if kwargs['command'].startswith('rm -f --'):
                return _command_result()
            if kwargs['command'] == 'bash tests/test.sh':
                return _command_result(return_code=0, stdout='verifier completed')
            return _command_result(stdout='{"reward": 0.375}')

    runtime = Runtime()
    result = await ToolathlonRuntimeEvaluator().evaluate(
        runtime=runtime,
        command='bash tests/test.sh',
        cwd='/workspace/case',
        env={},
        timeout_s=600,
        reward_json_path='/logs/verifier/reward.json',
    )

    assert result.score == 0.375
    assert result.passed is False
    assert runtime.commands == [
        'rm -f -- /logs/verifier/reward.json',
        'bash tests/test.sh',
        'cat -- /logs/verifier/reward.json',
    ]


@pytest.mark.asyncio
async def test_native_toolathlon_evaluator_ignores_agent_written_reward_file() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.commands: list[str] = []

        async def run_command(self, **kwargs):
            self.commands.append(kwargs['command'])
            # A malicious/stale reward file exists, but native evaluators must
            # never read it.
            return _command_result(return_code=1, stdout='native verifier failed')

    runtime = Runtime()
    result = await ToolathlonRuntimeEvaluator().evaluate(
        runtime=runtime,
        command='python -m evaluation.main',
        cwd='/workspace/case',
        env={},
        timeout_s=600,
    )

    assert result.score == 0.0
    assert result.passed is False
    assert runtime.commands == ['python -m evaluation.main']


@pytest.mark.asyncio
async def test_native_toolathlon_evaluator_uses_successful_exit_code() -> None:
    class Runtime:
        async def run_command(self, **kwargs):
            return _command_result(return_code=0)

    result = await ToolathlonRuntimeEvaluator().evaluate(
        runtime=Runtime(),
        command='python -m evaluation.main',
        cwd='/workspace/case',
        env={},
        timeout_s=600,
    )

    assert result.score == 1.0
    assert result.passed is True


@pytest.mark.asyncio
async def test_continuous_evaluator_clears_stale_reward_and_fails_closed_when_missing() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.commands: list[str] = []

        async def run_command(self, **kwargs):
            self.commands.append(kwargs['command'])
            if kwargs['command'].startswith('rm -f --'):
                return _command_result()
            if kwargs['command'] == 'bash tests/test.sh':
                return _command_result(return_code=0, stdout='verifier forgot reward')
            return _command_result(return_code=1, stderr='No such file')

    runtime = Runtime()
    result = await ToolathlonRuntimeEvaluator().evaluate(
        runtime=runtime,
        command='bash tests/test.sh',
        cwd='/workspace/case',
        env={},
        timeout_s=600,
        reward_json_path='/logs/verifier/reward.json',
    )

    assert result.score == 0.0
    assert result.passed is False
    assert 'reward_error' in result.details


@pytest.mark.asyncio
async def test_continuous_evaluator_aborts_when_reward_cleanup_fails() -> None:
    class Runtime:
        async def run_command(self, **kwargs):
            return _command_result(return_code=1, stderr='permission denied')

    with pytest.raises(RuntimeError, match='failed to reset evaluator reward file'):
        await ToolathlonRuntimeEvaluator().evaluate(
            runtime=Runtime(),
            command='bash tests/test.sh',
            cwd='/workspace/case',
            env={},
            timeout_s=600,
            reward_json_path='/logs/verifier/reward.json',
        )


@pytest.mark.asyncio
async def test_unisolated_continuous_evaluator_is_rejected_before_command_runs() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.commands = []

        async def run_command(self, **kwargs):
            self.commands.append(kwargs['command'])
            return _command_result()

    runtime = Runtime()
    task_input = SimpleNamespace(
        evaluator=SimpleNamespace(
            type='toolathlon_runtime',
            config={'command': 'bash tests/test.sh', 'cwd': '/workspace/case', 'env': {}, 'timeout_s': 600.0},
        )
    )
    with pytest.raises(RuntimeError, match='requires an isolated backend_mock'):
        await toolathlon_task._evaluate(
            task_input,
            runtime=runtime,
            case_root='/workspace/case',
            agent_workspace='/workspace/case/agent_workspace',
            launch_time='',
            res_log_file=None,
            continuous_reward_allowed=False,
        )

    # Reject before mutating evaluation data/state or running the verifier.
    assert runtime.commands == []
