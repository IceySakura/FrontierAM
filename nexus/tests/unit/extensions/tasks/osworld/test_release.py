"""OSWorld cleanup cannot replace a task result with a Stop API exception."""

import asyncio
import json
import subprocess
import sys
import threading
import time

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nexus.extensions.tasks.os_world.execution import release
from nexus.runtime.providers.ags.provider import AGSRuntimeProvider


def provider():
    owner = AGSRuntimeProvider(secret_id='dummy', secret_key='dummy')
    owner._instance_ids['main'] = 'owned-instance'
    owner._tool_ids['main'] = 'sdt-test'
    owner._active_runtimes['main'] = SimpleNamespace()
    owner._close_runtime = AsyncMock()
    return owner


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['success', 'failure', 'pending'])
async def test_stop_called_once_with_bounded_status_and_record(tmp_path, monkeypatch, mode):
    owner = provider()
    records = []
    gate = threading.Event()
    calls = []
    monkeypatch.setattr(release, 'STOP_WAIT_SECONDS', 0.1)

    def stop(current, instance, sandbox):
        assert owner._close_runtime.await_count == 1
        calls.append(instance)
        if mode == 'failure':
            error = RuntimeError('do not log credential secret')
            error.code = 'FailedOperation.Timeout'
            raise error
        if mode == 'pending':
            gate.wait(5)
        return {'request_id': 'request-fixture'}

    monkeypatch.setattr(release, '_stop_once', stop)
    release.install_release(owner, 'main', records, tmp_path)
    started = time.monotonic()
    try:
        await owner.release_all()
        assert time.monotonic() - started < 1
        assert len(records) == 1
        expected = {'success': 'stop_accepted', 'failure': 'stop_failed', 'pending': 'stop_pending'}[mode]
        assert records[0]['status'] == expected
        assert records[0]['instance_id'] == 'owned-instance'
        assert json.loads((tmp_path / 'release-status.json').read_text()) == records
        assert 'credential secret' not in (tmp_path / 'release-status.json').read_text()
        await owner.release_all()
        assert calls == ['owned-instance']
        assert not owner._active_runtimes
        assert not owner._instance_ids
        if mode == 'pending':
            gate.set()
            await asyncio.sleep(0.1)
            assert records[0]['status'] == 'stop_pending'
    finally:
        gate.set()


@pytest.mark.asyncio
async def test_external_instance_only_detaches(tmp_path, monkeypatch):
    owner = provider()
    owner._external_instance_names.add('main')
    records = []
    stop = Mock()
    monkeypatch.setattr(release, '_stop_once', stop)
    release.install_release(owner, 'main', records, tmp_path)
    await owner.release_all()
    stop.assert_not_called()
    assert records[0]['status'] == 'detached'


@pytest.mark.asyncio
async def test_local_close_failure_does_not_prevent_stop(tmp_path, monkeypatch):
    owner = provider()
    owner._close_runtime.side_effect = OSError('socket close failed')
    monkeypatch.setattr(release, '_stop_once', lambda *args: {})
    records = []
    release.install_release(owner, 'main', records, tmp_path)
    await owner.release_all()
    assert records[0]['status'] == 'stop_accepted'
    assert records[0]['local_cleanup_error_type'] == 'OSError'


@pytest.mark.asyncio
async def test_unmodified_provider_keeps_default_release_strategy(monkeypatch):
    owner = provider()
    stop = AsyncMock()
    monkeypatch.setattr(owner, '_stop_instance_with_retries', stop)
    await owner.release_all()
    stop.assert_awaited_once_with('main', 'owned-instance')


@pytest.mark.asyncio
async def test_template_stop_once_and_dynamic_tool_deferred(tmp_path, monkeypatch):
    owner = provider()
    sandbox = SimpleNamespace(kill=Mock(return_value=True))
    owner._template_sandboxes['main'] = sandbox
    owner._tools_to_delete['main'] = 'created-tool'
    records = []
    release.install_release(owner, 'main', records, tmp_path)
    await owner.release_all()
    sandbox.kill.assert_called_once_with(request_timeout=5)
    assert records[0]['status'] == 'stop_accepted'
    assert records[0]['tool_cleanup_pending'] == 'created-tool'


def test_dedicated_sdk_has_short_timeout_no_retry(monkeypatch):
    from tencentcloud.ags.v20250920 import ags_client

    owner = provider()
    client = SimpleNamespace(StopSandboxInstance=Mock(return_value=SimpleNamespace(RequestId='r1')))
    factory = Mock(return_value=client)
    monkeypatch.setattr(ags_client, 'AgsClient', factory)
    assert release._stop_once(owner, 'owned-instance', None) == {'request_id': 'r1'}
    profile = factory.call_args.args[2]
    assert profile.httpProfile.reqTimeout == 5
    assert profile.retryer is None
    client.StopSandboxInstance.assert_called_once()
    assert client.StopSandboxInstance.call_args.args[0].InstanceId == 'owned-instance'
    assert owner._client is None


def test_hung_stop_does_not_block_asyncio_run_or_process_exit(tmp_path):
    source = """
import asyncio,time
from pathlib import Path
from nexus.runtime.providers.ags.provider import AGSRuntimeProvider
from nexus.extensions.tasks.os_world.execution import release
release.STOP_WAIT_SECONDS=0.1
release._stop_once=lambda *args:time.sleep(60)
async def main():
    provider=AGSRuntimeProvider(secret_id='dummy',secret_key='dummy')
    provider._instance_ids['main']='test'
    rows=[]
    release.install_release(provider,'main',rows,Path(__import__('sys').argv[1]))
    await provider.release('main')
    assert rows[0]['status']=='stop_pending'
asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, '-c', source, str(tmp_path)],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_cancellation_does_not_skip_stop_or_change_to_success(tmp_path, monkeypatch):
    owner = provider()
    started = threading.Event()
    gate = threading.Event()

    def stop(*args):
        started.set()
        gate.wait(5)
        return {}

    monkeypatch.setattr(release, '_stop_once', stop)
    rows = []
    release.install_release(owner, 'main', rows, tmp_path)
    job = asyncio.create_task(owner.release_all())
    try:
        while not started.is_set():
            await asyncio.sleep(0.01)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        assert rows[0]['status'] == 'stop_pending'
        assert not owner._instance_ids
    finally:
        gate.set()


@pytest.mark.asyncio
@pytest.mark.parametrize('task_fails', [False, True])
@pytest.mark.parametrize('stop_fails', [False, True])
async def test_runner_result_survives_stop_and_records_status_after_publication(
    tmp_path, monkeypatch, task_fails, stop_fails
):
    from nexus.extensions.tasks.os_world import task
    from nexus.extensions.tasks.os_world.data.artifacts import RunArtifacts
    from nexus.runner import single

    owner = provider()
    events = []
    monkeypatch.setattr(single, '_create_runtime_provider', lambda _: owner)
    monkeypatch.setattr('nexus.extensions.tasks.os_world.data.artifacts.NEXUS_HOME', tmp_path)

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
        return task.OSWorldTaskOutput(completed=True, score=0.75, task_id='025', domain='tasks', instruction='fixture')

    monkeypatch.setattr(task, '_run_os_world', execute)
    config = single.RunInput.model_validate({
        'task_name': 'os_world',
        'task_config': {'task_id': '025'},
        'runner': {'hooks': {'output_contract': {'name': 'osworld_output_contract'}}},
    })
    output = await single.run(config, hooks=single.RunHooks(task_input_overrides={'run_agent': AsyncMock()}))
    assert output.success is not task_fails
    assert events == ['task', 'task_artifacts', 'runner_artifacts', 'stop']
    assert output.task_output['score'] == (None if task_fails else 0.75)
    assert output.error.message == 'original task failure' if task_fails else output.error is None
    status = output.task_output['artifacts']['release'][0]
    assert status['status'] == ('stop_failed' if stop_fails else 'stop_accepted')
    assert status['instance_id'] == 'owned-instance'
    # The final response carries teardown status even though the immutable
    # task/runner archives were uploaded before teardown.
    encoded = json.loads(output.model_dump_json())
    assert encoded['task_output']['artifacts']['release'] == [status]
