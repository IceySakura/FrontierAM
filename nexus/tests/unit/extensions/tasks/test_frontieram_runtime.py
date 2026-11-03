"""Lifecycle and command-boundary tests; no Docker or model service is called."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nexus.extensions.tasks.frontieram_offline import assets, mcp, runtime
from nexus.runtime.models import InvokeRequest


@pytest.mark.asyncio
@pytest.mark.parametrize('operation_fails', [False, True])
async def test_repeated_cancellation_waits_for_lifecycle_thread(operation_fails):
    started, release, finished = asyncio.Event(), threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def operation():
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=3)
        finished.set()
        if operation_fails:
            raise RuntimeError('synthetic lifecycle failure')

    running = asyncio.create_task(runtime._blocking(operation))
    await started.wait()
    try:
        for _ in range(3):
            running.cancel()
            await asyncio.sleep(0)
            assert not running.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, timeout=3)
    assert finished.is_set()


def test_asset_drift_is_rejected_before_module_import(tmp_path, monkeypatch):
    image = tmp_path / 'image_lock.json'
    image.write_text(json.dumps({'image_id': assets.IMAGE_ID}))
    code = tmp_path / 'campaign_runner_v2.py'
    code.write_text('VALUE = 42\n')
    monkeypatch.setattr(
        assets, 'ASSET_SHA256', {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (image, code)}
    )
    locked = assets.LockedAssets(tmp_path)
    assert locked.controller.VALUE == 42
    code.write_text('raise AssertionError("unreviewed code must not run")\n')
    with pytest.raises(assets.AssetDriftError, match='Frozen module changed'):
        _ = locked.controller


def _offline_runtime(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    output = tmp_path / 'output'
    output.mkdir()
    events = []
    controller = SimpleNamespace(
        verify_configuration=lambda info, image: {'safe': True},
        freeze=lambda *args: events.append('export') or {'sha256': 'frozen'},
    )
    docker = SimpleNamespace(
        prefix=['/trusted/docker', '--host', 'unix:///trusted/socket'],
        cpuset='0',
        cgroup_parent=None,
        verify=lambda info, image: controller.verify_configuration(info, image),
        inspect=lambda name: {'Id': 'a' * 64},
        pause=lambda name: events.append('pause'),
        remove=lambda name: events.append('remove'),
    )
    locked = SimpleNamespace(source_root=source, controller=controller, image_id=assets.IMAGE_ID)
    value = runtime.OfflineDockerRuntime(
        assets=locked,
        docker=docker,
        container='nexus-frontieram-' + 'b' * 32,
        output_dir=output,
        configuration_checks={'safe': True},
    )
    return value, events


@pytest.mark.asyncio
async def test_unknown_invoke_is_rejected_before_unpickling(tmp_path):
    value, _ = _offline_runtime(tmp_path)
    with pytest.raises(ValueError, match='only the run_command'):
        await value._invoke(InvokeRequest(import_path='os::system', payload='not-even-base64'))


@pytest.mark.asyncio
async def test_deadline_cannot_be_renewed_and_mcp_only_receives_lease(tmp_path):
    value, _ = _offline_runtime(tmp_path)
    await value.initialize_lease()
    deadline = time.monotonic() + 60
    config = value.mcp_config(deadline_monotonic=deadline)
    assert config['name'] == 'workspace'
    assert config['enabled_tools'] == ['exec']
    assert config['args'][-2] == '--lease-file'
    assert json.loads(Path(config['args'][-1]).read_text())['deadline_monotonic'] == deadline
    assert Path(config['args'][-1]).stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match='cannot be changed'):
        value.mcp_config(deadline_monotonic=deadline + 1)
    await value.close()


@pytest.mark.asyncio
async def test_pause_precedes_export_and_release_does_not_resume(tmp_path):
    value, events = _offline_runtime(tmp_path)
    await value.initialize_lease()
    await value.pause()
    result = await value.freeze()
    assert result == {'sha256': 'frozen'}
    assert events == ['pause', 'pause', 'export']
    with pytest.raises(RuntimeError, match='no longer open'):
        value.mcp_config(deadline_monotonic=time.monotonic() + 30)
    await value.close()
    assert events == ['pause', 'pause', 'export']
    assert not value._lease_path.exists()


@pytest.mark.asyncio
async def test_watchdog_pause_race_accepts_only_verified_owned_frozen_state(tmp_path):
    value, events = _offline_runtime(tmp_path)
    await value.initialize_lease()

    def raced_pause(name):
        raise RuntimeError('already paused by watchdog')

    value.docker.pause = raced_pause
    value.docker.inspect = lambda name: {'Id': 'a' * 64, 'State': {'Running': True, 'Paused': True}}
    assert (await value.pause())['paused'] is True
    await value.freeze()
    assert events == ['export']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'state',
    [
        {'Id': 'b' * 64, 'State': {'Running': True, 'Paused': True}},
        {'Id': 'a' * 64, 'State': {'Running': True, 'Paused': False}},
        {'Id': 'a' * 64, 'State': {'Running': False, 'Paused': True}},
    ],
)
async def test_failed_pause_does_not_hide_replacement_or_running_container(tmp_path, state):
    value, _ = _offline_runtime(tmp_path)
    await value.initialize_lease()

    def failed_pause(name):
        raise RuntimeError('pause failed')

    value.docker.pause = failed_pause
    value.docker.inspect = lambda name: state
    with pytest.raises(RuntimeError, match='pause failed'):
        await value.pause()
    assert not value._paused


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'extra',
    [
        {'cwd': '/Users'},
        {'stdin': 'payload'},
        {'extra_env': {'PATH': '/host'}},
        {'timeout': float('nan')},
        {'timeout_control_id': 'renew-me'},
        {'unrecognized': True},
    ],
)
async def test_standard_command_cannot_expand_the_execution_boundary(tmp_path, extra):
    value, events = _offline_runtime(tmp_path)
    await value.initialize_lease()
    value.set_deadline(deadline_monotonic=time.monotonic() + 60)
    with pytest.raises(ValueError, match='command|Command|Offline|Unsupported'):
        await value._run_command({'command': 'true', 'timeout': 10, **extra})
    assert events == []
    await value.close()


def test_expired_mcp_request_never_dispatches():
    dispatch = Mock()
    raw_mcp = SimpleNamespace(
        validate_arguments=lambda value: ('echo bounded', 120, 1000),
        bounded_run=dispatch,
        DockerWorkspace=lambda *args: SimpleNamespace(prefix=['docker'], container='fixed'),
    )
    workspace = mcp.DeadlineWorkspace(
        SimpleNamespace(docker_mcp=raw_mcp),
        {
            'deadline_monotonic': time.monotonic() - 1,
            'docker_path': 'docker',
            'docker_host': 'unix:///local/socket',
            'container': 'fixed',
        },
    )
    with pytest.raises(RuntimeError, match='no command was dispatched'):
        workspace.execute({'command': 'echo bounded'})
    dispatch.assert_not_called()


def test_mcp_command_time_is_capped_by_remaining_shared_budget(monkeypatch):
    dispatch = Mock(return_value={'exit_code': 0, 'timed_out': False, 'output': ''})
    raw_mcp = SimpleNamespace(
        validate_arguments=lambda value: ('echo bounded', 120, 1000),
        bounded_run=dispatch,
        DockerWorkspace=lambda *args: SimpleNamespace(prefix=['docker'], container='fixed'),
    )
    monkeypatch.setattr(mcp.time, 'monotonic', lambda: 100.0)
    workspace = mcp.DeadlineWorkspace(
        SimpleNamespace(docker_mcp=raw_mcp),
        {
            'deadline_monotonic': 102.5,
            'docker_path': 'docker',
            'docker_host': 'unix:///local/socket',
            'container': 'fixed',
        },
    )
    workspace.execute({'command': 'echo bounded'})
    argv, host_timeout, limit = dispatch.call_args.args
    assert argv[:5] == ['docker', 'exec', '-i', '--workdir', '/workspace']
    assert '--signal=KILL' in argv
    assert '2.5' in argv
    assert host_timeout == 3.5
    assert limit == 1000


def test_mcp_rejects_world_readable_lease_before_loading_assets(tmp_path, monkeypatch):
    lease = tmp_path / 'lease.json'
    lease.write_text('{}')
    lease.chmod(0o644)
    constructor = Mock()
    monkeypatch.setattr(mcp, 'LockedAssets', constructor)
    with pytest.raises(ValueError, match='private, owned regular file'):
        mcp.DeadlineWorkspace.from_lease(lease)
    constructor.assert_not_called()


@pytest.mark.asyncio
async def test_provider_cleans_container_when_public_staging_fails(tmp_path, monkeypatch):
    value, events = _offline_runtime(tmp_path)
    value.assets.public_tar = tmp_path / 'public.tar'
    value.assets.public_tar.write_bytes(b'fixture')
    value.assets.verify = lambda: None
    value.docker.create = lambda *args: events.append('create') or {'safe': True}
    value.docker.run = Mock(side_effect=RuntimeError('synthetic copy failure'))
    value.assets.controller.Docker = lambda *args: value.docker
    binary = tmp_path / 'docker'
    binary.touch()
    monkeypatch.setattr(runtime, 'LockedAssets', lambda source_root: value.assets)
    provider = runtime.FrontierAMOfflineProvider(tmp_path, 'unix:///local/socket', docker_path=binary)
    with pytest.raises(RuntimeError, match='synthetic copy failure'):
        await provider.acquire('main', output_dir=tmp_path / 'failed')
    assert events == ['create', 'remove']
    assert provider.get_active_runtimes() == {}
    assert not list((tmp_path / 'failed').glob('*.lease.json'))
    assert provider._acquiring_names == set()


@pytest.mark.asyncio
async def test_concurrent_acquire_cannot_replace_owned_runtime(tmp_path, monkeypatch):
    value, events = _offline_runtime(tmp_path)
    value.docker.create = lambda *args: events.append('create') or {'safe': True}
    value.assets.controller.Docker = lambda *args: value.docker
    binary = tmp_path / 'docker'
    binary.touch()
    monkeypatch.setattr(runtime, 'LockedAssets', lambda source_root: value.assets)
    provider = runtime.FrontierAMOfflineProvider(tmp_path, 'unix:///local/socket', docker_path=binary)
    result = await asyncio.gather(
        provider.acquire('main', output_dir=tmp_path / 'parallel', stage_public=False),
        provider.acquire('main', output_dir=tmp_path / 'parallel', stage_public=False),
        return_exceptions=True,
    )
    assert sum(isinstance(item, runtime.OfflineDockerRuntime) for item in result) == 1
    assert sum(isinstance(item, ValueError) for item in result) == 1
    assert events == ['create']
    await provider.release_all()
    assert events == ['create', 'remove']
