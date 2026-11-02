"""V2 cleanup cannot replace a task result with a Stop API exception."""

import json

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.tasks.os_world_v2 import task
from nexus.extensions.tasks.os_world_v2.data.artifacts import RunArtifacts
from nexus.extensions.tasks.os_world_v2.execution import release
from nexus.runtime.providers.ags.provider import AGSRuntimeProvider


def provider():
    owner = AGSRuntimeProvider(secret_id='dummy', secret_key='dummy')
    owner._instance_ids['main'] = 'owned-instance'
    owner._tool_ids['main'] = 'sdt-test'
    owner._active_runtimes['main'] = SimpleNamespace()
    owner._close_runtime = AsyncMock()
    return owner


@pytest.mark.asyncio
@pytest.mark.parametrize('task_fails', [False, True])
@pytest.mark.parametrize('stop_fails', [False, True])
async def test_runner_result_survives_stop_and_records_status_after_publication(
    tmp_path, monkeypatch, task_fails, stop_fails
):
    from nexus.runner import single

    owner = provider()
    events = []
    monkeypatch.setattr(single, '_create_runtime_provider', lambda _: owner)
    monkeypatch.setattr('nexus.extensions.tasks.os_world_v2.data.artifacts.NEXUS_HOME', tmp_path)

    async def publish(self):
        events.append('task_artifacts')
        self.reference.update(status='published', uri='artifact:///fixture/task/')
        return self.reference

    async def publish_details(self, output):
        events.append('runner_artifacts')
        return {'status': 'published', 'uri': 'artifact:///fixture/details/'}

    monkeypatch.setattr(RunArtifacts, 'publish', publish)
    monkeypatch.setattr(RunArtifacts, 'publish_run_output_details', publish_details)

    def stop(*args):
        events.append('stop')
        if stop_fails:
            raise RuntimeError('cleanup must not replace primary outcome')
        return {}

    monkeypatch.setattr(release, '_stop_once', stop)

    async def execute(_):
        events.append('task')
        if task_fails:
            raise ValueError('original task failure')
        return task.OSWorldV2TaskOutput(completed=True, score=0.75, task_id='025', domain='tasks', instruction='fixture')

    monkeypatch.setattr(task, 'run', execute)
    config = single.RunInput.model_validate({
        'task_name': 'os_world_v2',
        'task_config': {'task_id': '025', 'v21': {}},
        'runner': {'hooks': {'output_contract': {'name': 'osworld_v2_output_contract'}}},
    })
    output = await single.run(config, hooks=single.RunHooks(task_input_overrides={'run_agent': AsyncMock()}))
    assert output.success is not task_fails
    assert events == ['task', 'task_artifacts', 'runner_artifacts', 'stop']
    assert output.task_output['score'] == (None if task_fails else 0.75)
    assert output.error.message == 'original task failure' if task_fails else output.error is None
    status = output.task_output['artifacts']['release'][0]
    assert status['status'] == ('stop_failed' if stop_fails else 'stop_accepted')
    assert status['instance_id'] == 'owned-instance'
    encoded = json.loads(output.model_dump_json())
    assert encoded['task_output']['artifacts']['release'] == [status]
