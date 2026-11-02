"""V2 Codex execution keeps the agent result when guest cleanup fails."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.core.context import ExecutionContext
from nexus.extensions.agents.codex.native_bridge.adapters.types import CodexRunRequest
from nexus.extensions.tasks.os_world_v2.adapter.codex import OSWorldV2CodexProfile
from nexus.extensions.tasks.os_world_v2.adapter.codex_hitl import HITLCodexAdapter
from nexus.extensions.tasks.os_world_v2.execution.codex import TaskCodexAdapter
from nexus.runtime.bridge.adapters.base import BridgeLaunchResult
from nexus.runtime.executors.models.command import RunCommandResponse


def adapter(cls, runtime, diagnostic_dir):
    result = object.__new__(cls)
    result.runtime = runtime
    result.execution_env = {}
    result.diagnostics = {}
    result.diagnostic_dir = diagnostic_dir
    result.poll_interval = 0.01
    return result


def request():
    return CodexRunRequest(
        codex_executable='/bin/false',
        launch=BridgeLaunchResult(base_url='http://test', local_base_url='http://test'),
        task='prompt',
        working_dir='/tmp',
        timeout=10,
        version='0.154.0',
        npm_registry=None,
        strict_cleanup=True,
        developer_instructions='developer instructions',
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('exit_code', [0, 1])
@pytest.mark.parametrize('cleanup_timeout', [False, True])
@pytest.mark.parametrize('cls', [TaskCodexAdapter, HITLCodexAdapter])
async def test_cleanup_failure_preserves_execution_result(tmp_path, exit_code, cleanup_timeout, cls):
    runtime = SimpleNamespace(run_command=AsyncMock())
    worker = adapter(cls, runtime, tmp_path)
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
    assert worker.diagnostics['exit_code'] == exit_code
    assert worker.diagnostics['return_code'] == exit_code
    assert worker.diagnostics['cleanup_error']['type'] == ('TimeoutError' if cleanup_timeout else 'RuntimeError')


@pytest.mark.asyncio
async def test_profile_can_proceed_to_scoring_after_directory_cleanup_timeout(monkeypatch, tmp_path):
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
    worker = adapter(TaskCodexAdapter, runtime, tmp_path)
    worker.ensure_codex = AsyncMock(return_value='/bin/codex')
    worker._prepare = AsyncMock()
    worker._poll = AsyncMock(return_value=0)
    worker._capture = AsyncMock()
    monkeypatch.setattr(
        'nexus.extensions.tasks.os_world_v2.execution.codex.TaskCodexAdapter', lambda *a, **kw: worker
    )
    monkeypatch.setattr(
        'nexus.extensions.tasks.os_world_v2.execution.codex_profile.require_ags_osworld_runtime', lambda r: r
    )
    profile = OSWorldV2CodexProfile(
        {'guide_text': 'Complete the task.', 'model_catalog_preset': 'gpt6_catalog_v8'}, {}, 'fixture'
    )
    async with ExecutionContext(task_name='os_world_v2') as ctx:
        result = await profile.execute(runtime=runtime, launch=request().launch, timeout=10)
        assert ctx.tool_state['osworld_agent_output'].exit_status == 'finished'
    assert result['success'] is True
    assert result['exit_code'] == 0
    assert result['final_answer'] == 'TASK_DONE'
    assert worker.diagnostics['cleanup_error']['type'] == 'TimeoutError'
