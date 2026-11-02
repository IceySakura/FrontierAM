from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.tasks.os_world_v2.outputs import CONTRACT_ACTIVE_STATE


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [False, True])
async def test_v2_uses_shared_publisher_and_preserves_failed_artifact_reference(tmp_path, monkeypatch, failure):
    from nexus.extensions.tasks.os_world_v2.data.artifacts import RunArtifacts
    from nexus.extensions.tasks.os_world_v2 import task
    from nexus.extensions.tasks.os_world_v2.adapter import artifacts

    assert artifacts.RunArtifacts is RunArtifacts
    stage = SimpleNamespace(
        root=tmp_path,
        files=tmp_path,
        reference={'status': 'pending', 'uri': None},
        finish=lambda **_: None,
        publish=AsyncMock(return_value={'status': 'published', 'uri': 'artifact:test'}),
    )
    context = SimpleNamespace(tool_state={}, run_id='test', runtime_provider=None)
    monkeypatch.setattr(task, 'require_context', lambda: context)
    monkeypatch.setattr(artifacts, 'RunArtifacts', lambda **_: stage)

    async def run(_):
        assert context.tool_state['osworld_artifact_publisher'] is stage
        if failure:
            raise RuntimeError('original setup failure')
        return SimpleNamespace(artifacts=None)

    monkeypatch.setattr(task, 'run', run)
    config = task.OSWorldV2TaskInput(task_id='007', v21={}, run_agent=AsyncMock())
    if failure:
        # Hook-managed run: the failure propagates and the hook reports the partial output.
        context.tool_state[CONTRACT_ACTIVE_STATE] = True
        with pytest.raises(RuntimeError, match='original setup failure'):
            await task.run_os_world_v2(config)
        assert context.tool_state.pop(CONTRACT_ACTIVE_STATE) is True
        partial = context.tool_state.pop('osworld_partial_task_output')
        assert partial['score'] is None
        assert partial['artifacts'] is stage.reference
        assert partial['status'] == 'agent_error'
        assert partial['agent_output']['exit_status'] == 'environment_error'
        assert not hasattr(context, 'partial_task_output')
    else:
        result = await task.run_os_world_v2(config)
        assert result.artifacts['uri'] == 'artifact:test'
    assert context.tool_state.pop('osworld_output_publisher') is stage
    # Shared until the output hook projects it; provider teardown fills it later.
    assert context.tool_state.pop('osworld_release_records') == []
    assert context.tool_state == {}
    stage.publish.assert_awaited_once()


@pytest.mark.asyncio
async def test_v2_standalone_failure_returns_classified_task_output(tmp_path, monkeypatch):
    """Without the output-contract hook the task itself reports a spec exit_status."""
    from nexus.extensions.tasks.os_world_v2 import task
    from nexus.extensions.tasks.os_world_v2.adapter import artifacts

    stage = SimpleNamespace(
        root=tmp_path,
        files=tmp_path,
        reference={'status': 'pending', 'uri': None},
        finish=lambda **_: None,
        publish=AsyncMock(return_value={'status': 'published', 'uri': 'artifact:test'}),
    )
    context = SimpleNamespace(tool_state={}, run_id='test', runtime_provider=None)
    monkeypatch.setattr(task, 'require_context', lambda: context)
    monkeypatch.setattr(artifacts, 'RunArtifacts', lambda **_: stage)

    async def run(_):
        # e.g. the agent input model rejects an injected agent_config value.
        context.tool_state['osworld_output_stage'] = 'agent'
        raise ValueError('statusbench-unclassifiable-value')

    monkeypatch.setattr(task, 'run', run)
    result = await task.run_os_world_v2(task.OSWorldV2TaskInput(task_id='007', v21={}, run_agent=AsyncMock()))
    assert result.status == 'agent_error'
    assert result.score is None
    assert result.agent_output['exit_status'] == 'unknown_error'
    assert result.agent_output['exit_code'] == -1
    assert 'statusbench-unclassifiable-value' in result.error
    assert 'osworld_partial_task_output' not in context.tool_state
