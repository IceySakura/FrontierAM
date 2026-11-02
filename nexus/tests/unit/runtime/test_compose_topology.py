from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import tarfile

from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus.runtime.topology.compose import (
    _TIMEOUT_ERRORS,
    ComposeTopology,
    _parse_compose_ps,
    _shielded_best_effort_cleanup,
)
from nexus.runtime.topology.models import ComposeTopologySpec


IMAGE = f'mirrors.tencent.com/rl-task/service@sha256:{"a" * 64}'
AGENT_HELPER_DIR = Path(__file__).parents[3] / 'deployment' / 'docker' / 'runtime-bin'


class _FakeRuntime:
    def __init__(self, *, definition_sha256: str, ps_outputs: list[str] | None = None) -> None:
        self.definition_sha256 = definition_sha256
        self.ps_outputs = list(ps_outputs or [])
        self.commands: list[tuple[str, float, str | None]] = []
        self.command_envs: list[dict[str, str]] = []
        self.uploads = []
        self.downloads = []
        self.download_content = b'downloaded'
        self.fail_down = False
        self.timeout_down = False
        self.hang_down = False
        self.block_down = False
        self.down_started = asyncio.Event()
        self.down_release = asyncio.Event()
        self.hang_cleanup = False
        self.cleanup_started = asyncio.Event()
        self.cleanup_cancelled = False
        self.exec_return_code = 0
        self.kill_already_stopped = False
        self.kill_multiple_already_stopped = False
        self.kill_other_error = False
        self.fail_compose_cp = False
        self.compose_version = '2.35.1'
        self.remote_path_valid = True

    async def run_command(self, command: str, *, timeout: float, stdin: str | None = None, **_kwargs):
        self.commands.append((command, timeout, stdin))
        self.command_envs.append(_kwargs.get('extra_env') or {})
        if command == 'docker compose version --short':
            return SimpleNamespace(return_code=0, stdout=f'{self.compose_version}\n', stderr='')
        if command.startswith('if [ ! -') and not self.remote_path_valid:
            return SimpleNamespace(return_code=1, stdout='', stderr='symlink rejected')
        if command.startswith('sha256sum '):
            return SimpleNamespace(
                return_code=0,
                stdout=f'{self.definition_sha256.removeprefix("sha256:")}  definition.yaml\n',
                stderr='',
            )
        if ' ps --all --format json' in command:
            output = self.ps_outputs.pop(0) if self.ps_outputs else '[]'
            return SimpleNamespace(return_code=0, stdout=output, stderr='')
        if ' down ' in command and self.block_down:
            self.down_started.set()
            await self.down_release.wait()
        if ' down ' in command and self.hang_down:
            await asyncio.Event().wait()
        if command == 'rm -rf /cleanup' and self.hang_cleanup:
            self.cleanup_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cleanup_cancelled = True
                raise
        if ' down ' in command and self.timeout_down and '--timeout 0' not in command:
            return SimpleNamespace(return_code=None, stdout='', stderr='graceful timeout')
        if ' down ' in command and self.fail_down:
            return SimpleNamespace(return_code=1, stdout='', stderr='down failed')
        if ' cp ' in command and self.fail_compose_cp:
            return SimpleNamespace(return_code=1, stdout='', stderr='container rootfs is marked read-only')
        if command.endswith(' kill') and self.kill_already_stopped:
            return SimpleNamespace(
                return_code=1,
                stdout='',
                stderr='Error response from daemon: cannot kill container: abc: container abc is not running',
            )
        if command.endswith(' kill') and self.kill_multiple_already_stopped:
            return SimpleNamespace(
                return_code=1,
                stdout='',
                stderr=(
                    ' Container project-main-1  Killing\n'
                    ' Container project-db-1  Error while Killing\n'
                    'Error response from daemon: cannot kill container: abc: container abc is not running\n'
                    ' Container project-main-1  Error while Killing\n'
                    'Error response from daemon: cannot kill container: def: container def is not running'
                ),
            )
        if command.endswith(' kill') and self.kill_other_error:
            return SimpleNamespace(return_code=1, stdout='', stderr='Error response from daemon: permission denied')
        if ' exec -T ' in command:
            return SimpleNamespace(
                return_code=self.exec_return_code,
                stdout='',
                stderr='service failed' if self.exec_return_code else '',
            )
        return SimpleNamespace(return_code=0, stdout='', stderr='')

    async def upload_file(self, files, overwrite=True):
        self.uploads.append((files, overwrite))
        return SimpleNamespace(results=[SimpleNamespace(path=item.path, success=True, message='') for item in files])

    async def download_file(self, files):
        self.downloads.append(files)
        return SimpleNamespace(
            files=[
                SimpleNamespace(
                    path=files[0].path,
                    content=base64.b64encode(self.download_content).decode(),
                    encoding='base64',
                )
            ],
            errors=None,
        )


def _fixture(tmp_path: Path) -> tuple[Path, ComposeTopologySpec]:
    task_dir = tmp_path / 'task'
    (task_dir / 'environment' / 'config').mkdir(parents=True)
    definition = task_dir / 'environment' / 'docker-compose.lock.yaml'
    definition.write_text('services:\n  main:\n    image: test\n')
    (task_dir / 'environment' / 'config' / 'app.yaml').write_text('mode: test\n')
    digest = f'sha256:{hashlib.sha256(definition.read_bytes()).hexdigest()}'
    spec = ComposeTopologySpec(
        definition='environment/docker-compose.lock.yaml',
        definition_sha256=digest,
        primary_service='main',
        startup_timeout_sec=30,
        shutdown_timeout_sec=10,
        services={
            'main': {'image': IMAGE, 'has_healthcheck': True},
            'db': {'image': IMAGE, 'has_healthcheck': True},
            'migrate': {'image': IMAGE, 'one_shot': True},
        },
        bind_paths=('environment/config',),
    )
    return task_dir, spec


@pytest.mark.asyncio
async def test_topology_stages_starts_and_waits_for_ready_services(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(
        definition_sha256=spec.definition_sha256,
        ps_outputs=[
            json.dumps([
                {'Service': 'main', 'State': 'running', 'Health': 'starting'},
                {'Service': 'db', 'State': 'running', 'Health': 'healthy'},
                {'Service': 'migrate', 'State': 'running'},
            ]),
            '\n'.join([
                json.dumps({'Service': 'main', 'State': 'running', 'Health': 'healthy'}),
                json.dumps({'Service': 'db', 'State': 'running', 'Health': 'healthy'}),
                json.dumps({'Service': 'migrate', 'State': 'exited', 'ExitCode': 0}),
            ]),
        ],
    )
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-attempt-1',
        poll_interval_sec=0,
    )

    await topology.stage()
    await topology.start()
    status = await topology.wait_ready()

    assert status.state == 'ready'
    assert status.services['migrate'].exit_code == 0
    commands = [command for command, _, _ in runtime.commands]
    assert not any('docker compose' in command and ' pull' in command for command in commands)
    assert any('docker compose' in command and ' up --detach --remove-orphans' in command for command in commands)
    assert runtime.uploads

    runtime.ps_outputs.append(
        json.dumps([
            {'Service': 'main', 'State': 'running', 'Health': 'unhealthy'},
            {'Service': 'db', 'State': 'running', 'Health': 'healthy'},
            {'Service': 'migrate', 'State': 'exited', 'ExitCode': 0},
        ])
    )
    assert (await topology.inspect()).state == 'failed'


@pytest.mark.asyncio
async def test_topology_does_not_verify_staged_definition_digest(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(definition_sha256=f'sha256:{"b" * 64}')
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-digest-metadata-only',
    )

    await topology.stage()

    commands = [command for command, _, _ in runtime.commands]
    assert not any(command.startswith('sha256sum ') for command in commands)


@pytest.mark.asyncio
async def test_topology_stages_private_primary_environment_override(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    spec = spec.model_copy(update={'primary_environment': {'TASK_TOKEN': 'secret-value'}})
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-primary-env',
    )

    await topology.stage()
    await topology.start()

    commands = [command for command, _, _ in runtime.commands]
    assert all('secret-value' not in command for command in commands)
    assert any('chmod 600' in command and '.nexus-runtime.override.json' in command for command in commands)
    assert any(
        '--file /nexus/compose/task-primary-env/.nexus-runtime.override.json' in command and ' up --detach' in command
        for command in commands
    )
    uploaded_items = [item for files, _ in runtime.uploads for item in files]
    override_item = next(item for item in uploaded_items if item.path.endswith('.nexus-runtime.override.json'))
    override = json.loads(base64.b64decode(override_item.content).decode())
    assert override == {'services': {'main': {'environment': {'TASK_TOKEN': 'secret-value'}}}}


@pytest.mark.asyncio
async def test_topology_stages_trusted_agent_transport_mounts(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    spec = spec.model_copy(update={'agent_transport': 'socket-v1'})
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-agent-transport',
        agent_helper_dir=AGENT_HELPER_DIR,
    )

    await topology.stage()

    uploaded_items = [item for files, _ in runtime.uploads for item in files]
    override_item = next(item for item in uploaded_items if item.path.endswith('.nexus-runtime.override.json'))
    override = json.loads(base64.b64decode(override_item.content).decode())
    volumes = override['services']['main']['volumes']
    assert volumes == [
        {
            'type': 'bind',
            'source': '/nix',
            'target': '/nix',
            'read_only': True,
            'bind': {'create_host_path': False},
        },
        {
            'type': 'bind',
            'source': topology.agent_transport_host_dir,
            'target': '/opt/nexus-agent',
            'bind': {'create_host_path': False},
        },
    ]
    assert topology.project_name == 'task-agent-transport'
    assert list(topology.compose_files) == [
        '/nexus/compose/task-agent-transport/environment/docker-compose.lock.yaml',
        '/nexus/compose/task-agent-transport/.nexus-runtime.override.json',
    ]
    assert topology.agent_transport_host_dir.startswith('/tmp/nexus-agent/')
    assert topology.agent_transport_container_dir == '/opt/nexus-agent'
    uploaded_paths = {item.path for item in uploaded_items}
    assert topology.socket_bridge_host_path in uploaded_paths
    assert not any(path.endswith('nexus-compose-exec') for path in uploaded_paths)


@pytest.mark.asyncio
async def test_topology_rejects_failed_one_shot_and_still_deletes_resources(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(
        definition_sha256=spec.definition_sha256,
        ps_outputs=[
            json.dumps([
                {'Service': 'main', 'State': 'running', 'Health': 'healthy'},
                {'Service': 'db', 'State': 'running', 'Health': 'healthy'},
                {'Service': 'migrate', 'State': 'exited', 'ExitCode': 2},
            ])
        ],
    )
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-attempt-2',
        poll_interval_sec=0,
    )
    await topology.stage()
    await topology.start()

    with pytest.raises(RuntimeError, match='one-shot service .* exited with code 2'):
        await topology.wait_ready()

    await topology.delete()
    assert any(' down --timeout 10 --volumes --remove-orphans' in command for command, _, _ in runtime.commands)
    assert any(command.startswith('rm -rf ') for command, _, _ in runtime.commands)


@pytest.mark.asyncio
async def test_cancelled_start_preserves_cancellation_without_collecting_logs(tmp_path: Path) -> None:
    class _BlockingStartRuntime(_FakeRuntime):
        def __init__(self, *, definition_sha256: str) -> None:
            super().__init__(definition_sha256=definition_sha256)
            self.up_started = asyncio.Event()

        async def run_command(self, command: str, *, timeout: float, stdin: str | None = None, **kwargs):
            if ' up --detach --remove-orphans' in command:
                self.commands.append((command, timeout, stdin))
                self.up_started.set()
                await asyncio.Event().wait()
            if ' logs ' in command:
                return SimpleNamespace(return_code=0, stdout='service log', stderr='')
            return await super().run_command(command, timeout=timeout, stdin=stdin, **kwargs)

    task_dir, spec = _fixture(tmp_path)
    runtime = _BlockingStartRuntime(definition_sha256=spec.definition_sha256)
    topology = ComposeTopology(runtime=runtime, spec=spec, task_dir=task_dir, project_name='task-cancelled-start')
    await topology.stage()

    start = asyncio.create_task(topology.start())
    await asyncio.wait_for(runtime.up_started.wait(), timeout=1)
    start.cancel()

    with pytest.raises(asyncio.CancelledError):
        await start

    assert topology._state == 'failed'
    assert not any(' logs ' in command for command, _, _ in runtime.commands)


@pytest.mark.asyncio
async def test_delete_preserves_down_error_and_remains_retryable(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    runtime.fail_down = True
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-attempt-3',
    )
    await topology.stage()
    await topology.start()

    with pytest.raises(RuntimeError, match='down failed'):
        await topology.delete()

    commands_after_failure = [command for command, _, _ in runtime.commands]
    assert not any(command.startswith('rm -rf ') for command in commands_after_failure)

    runtime.fail_down = False
    await topology.delete()
    assert any(command.startswith('rm -rf ') for command, _, _ in runtime.commands)


@pytest.mark.asyncio
async def test_concurrent_delete_callers_share_one_deletion(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-attempt-single-flight-delete',
    )
    await topology.stage()
    await topology.start()
    runtime.block_down = True

    first = asyncio.create_task(topology.delete())
    await asyncio.wait_for(runtime.down_started.wait(), timeout=1)
    second = asyncio.create_task(topology.delete())
    await asyncio.sleep(0)

    down_commands = [command for command, _, _ in runtime.commands if ' down ' in command]
    assert len(down_commands) == 1

    runtime.down_release.set()
    await asyncio.gather(first, second)
    assert topology._state == 'deleted'


@pytest.mark.asyncio
async def test_delete_forces_cleanup_after_graceful_timeout(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    runtime.timeout_down = True
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-attempt-force-delete',
    )
    await topology.stage()
    await topology.start()

    await topology.delete()

    commands = [command for command, _, _ in runtime.commands]
    assert any(' kill' in command for command in commands)
    assert any(' down --timeout 0 --volumes --remove-orphans' in command for command in commands)
    assert any(command.startswith('rm -rf ') for command in commands)


@pytest.mark.asyncio
async def test_delete_has_total_timeout_without_waiting_for_cancellation(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-attempt-total-delete-timeout',
    )
    await topology.stage()
    await topology.start()

    runtime.hang_down = True
    timeout_calls = 0

    def cleanup_timeout() -> float:
        nonlocal timeout_calls
        timeout_calls += 1
        return 0.05

    topology._cleanup_abandon_timeout = cleanup_timeout

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(topology.delete(), timeout=5)

    await asyncio.sleep(0)
    assert timeout_calls == 1
    assert topology._state == 'failed'
    commands_after_abandonment = list(runtime.commands)

    await topology.delete()

    assert runtime.commands == commands_after_abandonment
    assert timeout_calls == 1


@pytest.mark.asyncio
async def test_delete_from_cancelled_finally_uses_total_timeout(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-attempt-cancel-before-delete',
    )
    await topology.stage()
    await topology.start()

    runtime.hang_down = True
    timeout_calls = 0

    def cleanup_timeout() -> float:
        nonlocal timeout_calls
        timeout_calls += 1
        return 0.05

    topology._cleanup_abandon_timeout = cleanup_timeout
    agent_work = asyncio.Event()

    async def run_with_cleanup() -> None:
        try:
            await agent_work.wait()
        finally:
            await topology.delete()

    task = asyncio.create_task(run_with_cleanup())
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)

    await asyncio.sleep(0)
    assert timeout_calls == 1
    assert topology._state == 'failed'


@pytest.mark.asyncio
async def test_cancelled_delete_abandons_a_deletion_that_never_returns(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-attempt-abandon',
    )
    await topology.stage()
    await topology.start()

    runtime.hang_down = True
    topology._cleanup_abandon_timeout = lambda: 0.05

    delete = asyncio.create_task(topology.delete())
    await asyncio.sleep(0)
    delete.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(delete, timeout=5)


@pytest.mark.asyncio
async def test_cancelled_delete_still_waits_for_cleanup_that_finishes(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-attempt-abandon-ok',
    )
    await topology.stage()
    await topology.start()

    delete = asyncio.create_task(topology.delete())
    await asyncio.sleep(0)
    delete.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(delete, timeout=5)

    assert any(command.startswith('rm -rf ') for command, _, _ in runtime.commands)


def test_cleanup_timeout_errors_include_asyncio_compatibility_type() -> None:
    assert TimeoutError in _TIMEOUT_ERRORS
    assert asyncio.TimeoutError in _TIMEOUT_ERRORS


@pytest.mark.asyncio
async def test_best_effort_cleanup_has_total_timeout() -> None:
    runtime = _FakeRuntime(definition_sha256='sha256:unused')
    runtime.hang_cleanup = True

    await asyncio.wait_for(
        _shielded_best_effort_cleanup(runtime, '/cleanup', timeout=0.05),
        timeout=1,
    )

    await asyncio.sleep(0)
    assert runtime.cleanup_started.is_set()
    assert runtime.cleanup_cancelled is True


@pytest.mark.asyncio
async def test_cancelled_best_effort_cleanup_uses_remaining_timeout() -> None:
    runtime = _FakeRuntime(definition_sha256='sha256:unused')
    runtime.hang_cleanup = True

    cleanup = asyncio.create_task(_shielded_best_effort_cleanup(runtime, '/cleanup', timeout=0.05))
    await asyncio.wait_for(runtime.cleanup_started.wait(), timeout=1)
    cleanup.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(cleanup, timeout=1)

    await asyncio.sleep(0)
    assert runtime.cleanup_cancelled is True


@pytest.mark.asyncio
async def test_delete_tolerates_force_kill_when_container_is_already_stopped(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    runtime.timeout_down = True
    runtime.kill_already_stopped = True
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-force-kill-stopped',
    )
    await topology.stage()
    await topology.start()

    await topology.delete()

    commands = [command for command, _, _ in runtime.commands]
    assert any(command.endswith(' kill') for command in commands)
    assert any(' down --timeout 0 --volumes --remove-orphans' in command for command in commands)
    assert topology._state == 'deleted'


@pytest.mark.asyncio
async def test_delete_tolerates_compose_progress_errors_when_all_containers_are_stopped(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    runtime.timeout_down = True
    runtime.kill_multiple_already_stopped = True
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-force-kill-multiple-stopped',
    )
    await topology.stage()
    await topology.start()

    await topology.delete()

    assert topology._state == 'deleted'


@pytest.mark.asyncio
async def test_delete_preserves_non_idempotent_force_kill_error(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    runtime.timeout_down = True
    runtime.kill_other_error = True
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-force-kill-error',
    )
    await topology.stage()
    await topology.start()

    with pytest.raises(RuntimeError, match='permission denied'):
        await topology.delete()


@pytest.mark.asyncio
async def test_service_exec_forwards_workdir_user_env_and_stdin(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-attempt-4',
    )
    await topology.stage()
    await topology.start()

    result = await topology.service('main').exec(
        'printf "%s" "$TOKEN"',
        cwd='/workdir',
        env={'TOKEN': 'secret value'},
        user='agent',
        stdin='prompt',
        timeout=90,
    )

    command, timeout, stdin = runtime.commands[-1]
    assert 'exec -T --workdir /workdir --user agent --env' in command
    assert 'secret value' not in command
    assert runtime.command_envs[-1] == {'TOKEN': 'secret value'}
    assert "main sh -c 'printf" in command
    assert timeout == 90
    assert stdin == 'prompt'
    assert result.return_code == 0

    runtime.exec_return_code = 7
    failed = await topology.service('main').exec('exit 7')
    assert failed.return_code == 7


@pytest.mark.asyncio
async def test_service_file_upload_and_download_use_controller_staging(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-attempt-5',
    )
    await topology.stage()
    await topology.start()
    source = tmp_path / 'source.bin'
    source.write_bytes(b'uploaded')
    target = tmp_path / 'downloaded.bin'

    await topology.service('db').upload_file(source, '/tmp/source.bin')
    await topology.service('db').download_file('/tmp/result.bin', target)

    assert target.read_bytes() == b'downloaded'
    commands = [command for command, _, _ in runtime.commands]
    assert any(' cp ' in command and 'db:/tmp/source.bin' in command for command in commands)
    assert any(' cp db:/tmp/result.bin ' in command for command in commands)
    assert not any(' exec -T --user root db ' in command for command in commands)


@pytest.mark.asyncio
async def test_service_transfers_fall_back_to_exec_for_read_only_tmpfs(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-read-only-tmpfs-transfer',
    )
    await topology.stage()
    await topology.start()
    runtime.fail_compose_cp = True

    source_file = tmp_path / 'source.bin'
    source_file.write_bytes(b'uploaded')
    source_dir = tmp_path / 'tests'
    source_dir.mkdir()
    (source_dir / 'test.sh').write_text('#!/bin/sh\n')
    (source_dir / '.hidden').write_text('hidden')
    (source_dir / 'empty').mkdir()

    service = topology.service('main')
    await service.upload_file(source_file, '/tests/source.bin')
    await service.upload_dir(source_dir, '/tests')
    uploaded_archives = [
        base64.b64decode(item.content or '')
        for files, _overwrite in runtime.uploads
        for item in files
        if item.path.endswith('/payload.tar.gz')
    ]
    assert len(uploaded_archives) == 1
    with tarfile.open(fileobj=io.BytesIO(uploaded_archives[0]), mode='r:gz') as archive:
        assert archive.getnames() == ['.hidden', 'empty', 'test.sh']

    downloaded_file = tmp_path / 'downloaded.bin'
    await service.download_file('/logs/artifacts/result.bin', downloaded_file)
    assert downloaded_file.read_bytes() == b'downloaded'

    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode='w:gz') as archive:
        payload = b'artifact'
        info = tarfile.TarInfo('./nested/result.bin')
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    runtime.download_content = archive_buffer.getvalue()
    downloaded_dir = tmp_path / 'artifacts'
    await service.download_dir('/logs/artifacts', downloaded_dir)
    assert (downloaded_dir / 'nested' / 'result.bin').read_bytes() == b'artifact'

    commands = [command for command, _, _ in runtime.commands]
    cp_commands = [command for command in commands if ' cp ' in command]
    fallback_commands = [command for command in commands if ' exec -T --user root main ' in command]
    assert len(cp_commands) == 4
    assert len(fallback_commands) == 4
    assert len([command for command in commands if command.startswith('rm -rf -- ')]) == 2
    assert any('source.bin' in command and ' < ' in command for command in fallback_commands)
    assert any('/nix/runtime/nix-env/bin/tar' in command and 'xzf -' in command for command in fallback_commands)
    assert any('/nix/runtime/nix-env/bin/tar' in command and 'czf -' in command for command in fallback_commands)
    assert any(' > ' in command and 'result.bin' in command for command in fallback_commands)


@pytest.mark.asyncio
async def test_service_transfer_reports_cp_and_exec_fallback_failures(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-transfer-double-failure',
    )
    await topology.stage()
    await topology.start()
    runtime.fail_compose_cp = True
    runtime.exec_return_code = 127
    source = tmp_path / 'source.bin'
    source.write_bytes(b'uploaded')

    with pytest.raises(RuntimeError, match='docker compose cp failed.*in-service fallback also failed'):
        await topology.service('main').upload_file(source, '/tests/source.bin')


@pytest.mark.asyncio
async def test_exec_fallback_download_rejects_archive_links(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-transfer-link-rejection',
    )
    await topology.stage()
    await topology.start()
    runtime.fail_compose_cp = True

    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode='w:gz') as archive:
        link = tarfile.TarInfo('./unsafe-link')
        link.type = tarfile.SYMTYPE
        link.linkname = '/etc/passwd'
        archive.addfile(link)
    runtime.download_content = archive_buffer.getvalue()

    with pytest.raises(ValueError, match='contains a link'):
        await topology.service('main').download_dir('/logs/artifacts', tmp_path / 'artifacts')


def test_compose_ps_parser_accepts_array_and_ndjson() -> None:
    array = json.dumps([
        {'Service': 'main', 'State': 'running', 'Health': 'healthy'},
        {'Service': 'migrate', 'State': 'exited', 'ExitCode': 0},
    ])
    ndjson = '\n'.join([
        json.dumps({'service': 'main', 'state': 'running', 'health': 'healthy'}),
        json.dumps({'service': 'migrate', 'state': 'exited', 'exit_code': 0}),
    ])

    assert _parse_compose_ps(array) == _parse_compose_ps(ndjson)


def test_topology_rejects_invalid_project_name(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)

    with pytest.raises(ValueError, match='Invalid Compose project name'):
        ComposeTopology(
            runtime=object(),
            spec=spec,
            task_dir=task_dir,
            project_name='Invalid Project',
        )


@pytest.mark.asyncio
async def test_stage_revalidates_bind_paths_and_rejects_symlinks(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    (task_dir / 'environment' / 'linked').symlink_to('/etc')
    payload = spec.model_dump()
    payload['bind_paths'] = ('environment/linked',)
    symlink_spec = ComposeTopologySpec.model_validate(payload)
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    topology = ComposeTopology(
        runtime=runtime,
        spec=symlink_spec,
        task_dir=task_dir,
        project_name='task-attempt-6',
    )

    with pytest.raises(ValueError, match='must not contain symlinks'):
        await topology.stage()

    assert (await topology.inspect()).state == 'new'


@pytest.mark.asyncio
async def test_service_operations_require_started_topology(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-attempt-7',
    )

    with pytest.raises(RuntimeError, match='require a started topology'):
        await topology.service('main').exec('true')


@pytest.mark.asyncio
async def test_service_download_rejects_controller_symlink_before_file_api(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    runtime.remote_path_valid = False
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-attempt-8',
    )
    await topology.stage()
    await topology.start()

    with pytest.raises(RuntimeError, match='contains symlinks'):
        await topology.service('main').download_file('/tmp/result', tmp_path / 'result')

    assert runtime.downloads == []


@pytest.mark.asyncio
async def test_stage_requires_supported_compose_version(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    runtime.compose_version = '2.19.9'
    topology = ComposeTopology(
        runtime=runtime,
        spec=spec,
        task_dir=task_dir,
        project_name='task-attempt-9',
    )

    with pytest.raises(RuntimeError, match=r'Docker Compose 2\.20\.0\+ is required'):
        await topology.stage()


@pytest.mark.asyncio
async def test_topology_starts_dependency_layers_in_bounded_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_dir, spec = _fixture(tmp_path)
    spec = spec.model_copy(
        update={
            'services': {
                'main': spec.services['main'].model_copy(update={'depends_on': ('db',)}),
                'db': spec.services['db'],
                'migrate': spec.services['migrate'].model_copy(update={'depends_on': ('db',)}),
            }
        }
    )
    runtime = _FakeRuntime(
        definition_sha256=spec.definition_sha256,
        ps_outputs=[
            json.dumps([{'Service': 'db', 'State': 'running', 'Health': 'healthy'}]),
            json.dumps([
                {'Service': 'db', 'State': 'running', 'Health': 'healthy'},
                {'Service': 'main', 'State': 'running', 'Health': 'healthy'},
            ]),
            json.dumps([
                {'Service': 'db', 'State': 'running', 'Health': 'healthy'},
                {'Service': 'main', 'State': 'running', 'Health': 'healthy'},
                {'Service': 'migrate', 'State': 'exited', 'ExitCode': 0},
            ]),
        ],
    )
    monkeypatch.setenv('NEXUS_COMPOSE_START_BATCH_SIZE', '1')
    topology = ComposeTopology(runtime=runtime, spec=spec, task_dir=task_dir, project_name='task-batched-start')

    await topology.stage()
    await topology.start()

    starts = [command for command, _, _ in runtime.commands if ' up --detach --remove-orphans' in command]
    assert len(starts) == 3
    assert starts[0].endswith(' --no-deps db')
    assert starts[1].endswith(' --no-deps main')
    assert starts[2].endswith(' --no-deps migrate')


@pytest.mark.asyncio
async def test_topology_retries_transient_start_container_hook_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_dir, spec = _fixture(tmp_path)

    class _TransientRuntime(_FakeRuntime):
        attempts = 0

        async def run_command(self, command: str, *, timeout: float, stdin: str | None = None, **kwargs):
            if ' up --detach --remove-orphans' in command:
                self.attempts += 1
                self.commands.append((command, timeout, stdin))
                self.command_envs.append(kwargs.get('extra_env') or {})
                if self.attempts == 1:
                    return SimpleNamespace(
                        return_code=1,
                        stdout='',
                        stderr='error running startContainer hook #0: hook ran past specified timeout of 60.0s',
                    )
            return await super().run_command(command, timeout=timeout, stdin=stdin, **kwargs)

    runtime = _TransientRuntime(definition_sha256=spec.definition_sha256)
    monkeypatch.setenv('NEXUS_COMPOSE_START_RETRY_ATTEMPTS', '2')
    monkeypatch.setenv('NEXUS_COMPOSE_START_RETRY_BACKOFF_SEC', '0')
    topology = ComposeTopology(runtime=runtime, spec=spec, task_dir=task_dir, project_name='task-retry-start')

    await topology.stage()
    await topology.start()

    assert runtime.attempts == 2


@pytest.mark.asyncio
async def test_topology_uses_retry_settings_from_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_dir, spec = _fixture(tmp_path)
    spec = spec.model_copy(
        update={
            'startup_timeout_sec': 300,
            'start_retry_attempts': 2,
            'start_retry_backoff_sec': 8.0,
            'start_retry_jitter_max': 60.0,
        }
    )

    class _TransientRuntime(_FakeRuntime):
        attempts = 0

        async def run_command(self, command: str, *, timeout: float, stdin: str | None = None, **kwargs):
            if ' up --detach --remove-orphans' in command:
                self.attempts += 1
                self.commands.append((command, timeout, stdin))
                self.command_envs.append(kwargs.get('extra_env') or {})
                if self.attempts == 1:
                    return SimpleNamespace(
                        return_code=1,
                        stdout='',
                        stderr='error running startContainer hook #0: hook ran past specified timeout of 60.0s',
                    )
            return await super().run_command(command, timeout=timeout, stdin=stdin, **kwargs)

    runtime = _TransientRuntime(definition_sha256=spec.definition_sha256)
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr('nexus.runtime.topology.compose.random.uniform', lambda _low, high: high)
    monkeypatch.setattr('nexus.runtime.topology.compose.asyncio.sleep', fake_sleep)
    topology = ComposeTopology(runtime=runtime, spec=spec, task_dir=task_dir, project_name='task-config-retry')

    await topology.stage()
    await topology.start()

    assert runtime.attempts == 2
    assert sleeps == [68.0]


@pytest.mark.asyncio
async def test_compose_retry_environment_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_dir, spec = _fixture(tmp_path)
    spec = spec.model_copy(
        update={
            'startup_timeout_sec': 300,
            'start_retry_attempts': 3,
            'start_retry_backoff_sec': 8.0,
            'start_retry_jitter_max': 60.0,
        }
    )

    class _TransientRuntime(_FakeRuntime):
        attempts = 0

        async def run_command(self, command: str, *, timeout: float, stdin: str | None = None, **kwargs):
            if ' up --detach --remove-orphans' in command:
                self.attempts += 1
                self.commands.append((command, timeout, stdin))
                self.command_envs.append(kwargs.get('extra_env') or {})
                if self.attempts == 1:
                    return SimpleNamespace(
                        return_code=1,
                        stdout='',
                        stderr='error running startContainer hook #0: hook ran past specified timeout of 60.0s',
                    )
            return await super().run_command(command, timeout=timeout, stdin=stdin, **kwargs)

    runtime = _TransientRuntime(definition_sha256=spec.definition_sha256)
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setenv('NEXUS_COMPOSE_START_RETRY_ATTEMPTS', '2')
    monkeypatch.setenv('NEXUS_COMPOSE_START_RETRY_BACKOFF_SEC', '2')
    monkeypatch.setenv('NEXUS_COMPOSE_START_RETRY_JITTER_MAX', '4')
    monkeypatch.setattr('nexus.runtime.topology.compose.random.uniform', lambda _low, high: high)
    monkeypatch.setattr('nexus.runtime.topology.compose.asyncio.sleep', fake_sleep)
    topology = ComposeTopology(runtime=runtime, spec=spec, task_dir=task_dir, project_name='task-env-retry')

    await topology.stage()
    await topology.start()

    assert runtime.attempts == 2
    assert sleeps == [6.0]


@pytest.mark.asyncio
async def test_topology_applies_independent_retry_jitter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_dir, spec = _fixture(tmp_path)
    spec = spec.model_copy(update={'startup_timeout_sec': 300})

    class _TransientRuntime(_FakeRuntime):
        attempts = 0

        async def run_command(self, command: str, *, timeout: float, stdin: str | None = None, **kwargs):
            if ' up --detach --remove-orphans' in command:
                self.attempts += 1
                self.commands.append((command, timeout, stdin))
                self.command_envs.append(kwargs.get('extra_env') or {})
                if self.attempts == 1:
                    return SimpleNamespace(
                        return_code=1,
                        stdout='',
                        stderr='error running startContainer hook #0: hook ran past specified timeout of 60.0s',
                    )
            return await super().run_command(command, timeout=timeout, stdin=stdin, **kwargs)

    runtime = _TransientRuntime(definition_sha256=spec.definition_sha256)
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setenv('NEXUS_COMPOSE_START_RETRY_ATTEMPTS', '2')
    monkeypatch.setenv('NEXUS_COMPOSE_START_RETRY_BACKOFF_SEC', '8')
    monkeypatch.setenv('NEXUS_COMPOSE_START_RETRY_JITTER_MAX', '120')
    monkeypatch.setattr('nexus.runtime.topology.compose.random.uniform', lambda _low, high: high)
    monkeypatch.setattr('nexus.runtime.topology.compose.asyncio.sleep', fake_sleep)
    topology = ComposeTopology(runtime=runtime, spec=spec, task_dir=task_dir, project_name='task-retry-jitter')

    await topology.stage()
    await topology.start()

    assert runtime.attempts == 2
    assert sleeps == [128.0]


@pytest.mark.asyncio
async def test_topology_start_failure_collects_service_logs(tmp_path: Path) -> None:
    task_dir, spec = _fixture(tmp_path)

    class _FailingRuntime(_FakeRuntime):
        async def run_command(self, command: str, *, timeout: float, stdin: str | None = None, **kwargs):
            if ' up --detach --remove-orphans' in command:
                self.commands.append((command, timeout, stdin))
                self.command_envs.append(kwargs.get('extra_env') or {})
                return SimpleNamespace(return_code=1, stdout='', stderr='service startup failed')
            if ' logs --no-color --tail 200' in command:
                self.commands.append((command, timeout, stdin))
                self.command_envs.append(kwargs.get('extra_env') or {})
                return SimpleNamespace(return_code=0, stdout='backend | diagnostic output', stderr='')
            return await super().run_command(command, timeout=timeout, stdin=stdin, **kwargs)

    runtime = _FailingRuntime(definition_sha256=spec.definition_sha256)
    topology = ComposeTopology(runtime=runtime, spec=spec, task_dir=task_dir, project_name='task-start-logs')

    await topology.stage()
    with pytest.raises(RuntimeError, match=r'Compose service logs:\nbackend \| diagnostic output'):
        await topology.start()

    assert any(' logs --no-color --tail 200' in command for command, _, _ in runtime.commands)


@pytest.mark.asyncio
async def test_topology_retries_hook_timeout_found_only_in_compose_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_dir, spec = _fixture(tmp_path)

    class _DiagnosticRuntime(_FakeRuntime):
        attempts = 0

        async def run_command(self, command: str, *, timeout: float, stdin: str | None = None, **kwargs):
            if ' up --detach --remove-orphans' in command:
                self.attempts += 1
                self.commands.append((command, timeout, stdin))
                self.command_envs.append(kwargs.get('extra_env') or {})
                if self.attempts == 1:
                    return SimpleNamespace(
                        return_code=1,
                        stdout='',
                        stderr='dependency failed to start: container backend exited (255)',
                    )
            if ' logs --no-color --tail 80 --since ' in command:
                self.commands.append((command, timeout, stdin))
                self.command_envs.append(kwargs.get('extra_env') or {})
                return SimpleNamespace(
                    return_code=0,
                    stdout='error running startContainer hook #0: hook ran past specified timeout of 60.0s',
                    stderr='',
                )
            return await super().run_command(command, timeout=timeout, stdin=stdin, **kwargs)

    runtime = _DiagnosticRuntime(definition_sha256=spec.definition_sha256)
    monkeypatch.setenv('NEXUS_COMPOSE_START_RETRY_ATTEMPTS', '2')
    monkeypatch.setenv('NEXUS_COMPOSE_START_RETRY_BACKOFF_SEC', '0')
    topology = ComposeTopology(runtime=runtime, spec=spec, task_dir=task_dir, project_name='task-log-retry-start')

    await topology.stage()
    await topology.start()

    assert runtime.attempts == 2
    assert any(' logs --no-color --tail 80 --since ' in command for command, _, _ in runtime.commands)


@pytest.mark.asyncio
async def test_topology_does_not_retry_application_dependency_failure_from_stale_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_dir, spec = _fixture(tmp_path)

    class _ApplicationFailureRuntime(_FakeRuntime):
        attempts = 0

        async def run_command(self, command: str, *, timeout: float, stdin: str | None = None, **kwargs):
            if ' up --detach --remove-orphans' in command:
                self.attempts += 1
                self.commands.append((command, timeout, stdin))
                self.command_envs.append(kwargs.get('extra_env') or {})
                return SimpleNamespace(
                    return_code=1,
                    stdout='',
                    stderr='dependency failed to start: container emails exited (1)',
                )
            if ' logs --no-color' in command:
                self.commands.append((command, timeout, stdin))
                self.command_envs.append(kwargs.get('extra_env') or {})
                return SimpleNamespace(
                    return_code=0,
                    stdout='old error: startContainer hook #0: hook ran past specified timeout of 60.0s',
                    stderr='',
                )
            return await super().run_command(command, timeout=timeout, stdin=stdin, **kwargs)

    runtime = _ApplicationFailureRuntime(definition_sha256=spec.definition_sha256)
    monkeypatch.setenv('NEXUS_COMPOSE_START_RETRY_ATTEMPTS', '3')
    monkeypatch.setenv('NEXUS_COMPOSE_START_RETRY_BACKOFF_SEC', '0')
    topology = ComposeTopology(runtime=runtime, spec=spec, task_dir=task_dir, project_name='task-app-failure')

    await topology.stage()
    with pytest.raises(RuntimeError, match=r'emails exited \(1\)'):
        await topology.start()

    assert runtime.attempts == 1
    assert not any(' logs --no-color --tail 80' in command for command, _, _ in runtime.commands)


@pytest.mark.asyncio
async def test_topology_uses_compose_startup_jitter_before_start_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_dir, spec = _fixture(tmp_path)
    spec = spec.model_copy(update={'startup_timeout_sec': 1, 'startup_jitter_max': 12.0})
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr('nexus.runtime.topology.compose.random.uniform', lambda _low, high: high)
    monkeypatch.setattr('nexus.runtime.topology.compose.asyncio.sleep', fake_sleep)
    topology = ComposeTopology(runtime=runtime, spec=spec, task_dir=task_dir, project_name='task-jitter-start')

    await topology.stage()
    await topology.start()

    assert sleeps == [12.0]
    assert topology._startup_deadline is not None


@pytest.mark.asyncio
async def test_compose_jitter_environment_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_dir, spec = _fixture(tmp_path)
    spec = spec.model_copy(update={'startup_jitter_max': 5.0})
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setenv('NEXUS_COMPOSE_STARTUP_JITTER_MAX', '30')
    monkeypatch.setattr('nexus.runtime.topology.compose.random.uniform', lambda _low, high: high)
    monkeypatch.setattr('nexus.runtime.topology.compose.asyncio.sleep', fake_sleep)
    topology = ComposeTopology(runtime=runtime, spec=spec, task_dir=task_dir, project_name='task-jitter-override')

    await topology.stage()
    await topology.start()

    assert sleeps == [30.0]


@pytest.mark.asyncio
async def test_topology_start_gate_rejects_capacity_below_service_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_dir, spec = _fixture(tmp_path)
    runtime = _FakeRuntime(definition_sha256=spec.definition_sha256)
    monkeypatch.setenv('NEXUS_COMPOSE_START_MAX_INFLIGHT_SERVICES', '2')
    monkeypatch.setenv('NEXUS_COMPOSE_START_GATE_DIR', str(tmp_path / 'gate'))
    topology = ComposeTopology(runtime=runtime, spec=spec, task_dir=task_dir, project_name='task-gate-too-small')

    await topology.stage()
    with pytest.raises(ValueError, match=r'\(2\) must be at least the topology service count \(3\)'):
        await topology.start()

    commands = [command for command, _, _ in runtime.commands]
    assert not any(' pull' in command or ' up ' in command for command in commands)


@pytest.mark.asyncio
async def test_topology_start_gate_bounds_200_tasks_by_service_weight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _MeasuredRuntime(_FakeRuntime):
        active_services = 0
        peak_services = 0
        guard = asyncio.Lock()

        def __init__(self, *, definition_sha256: str, service_weight: int) -> None:
            super().__init__(definition_sha256=definition_sha256)
            self.service_weight = service_weight

        async def run_command(self, command: str, *, timeout: float, stdin: str | None = None, **kwargs):
            if ' up --detach --remove-orphans' in command:
                async with self.guard:
                    type(self).active_services += self.service_weight
                    type(self).peak_services = max(type(self).peak_services, type(self).active_services)
                await asyncio.sleep(0.01)
                try:
                    return await super().run_command(command, timeout=timeout, stdin=stdin, **kwargs)
                finally:
                    async with self.guard:
                        type(self).active_services -= self.service_weight
            return await super().run_command(command, timeout=timeout, stdin=stdin, **kwargs)

    monkeypatch.setenv('NEXUS_COMPOSE_START_MAX_INFLIGHT_SERVICES', '48')
    monkeypatch.setenv('NEXUS_COMPOSE_START_GATE_DIR', str(tmp_path / 'gate'))
    _MeasuredRuntime.active_services = 0
    _MeasuredRuntime.peak_services = 0
    topologies = []
    for index in range(200):
        task_dir, spec = _fixture(tmp_path / f'case-{index}')
        spec = spec.model_copy(update={'startup_timeout_sec': 1})
        service_weight = len(spec.services)
        runtime = _MeasuredRuntime(definition_sha256=spec.definition_sha256, service_weight=service_weight)
        topology = ComposeTopology(
            runtime=runtime,
            spec=spec,
            task_dir=task_dir,
            project_name=f'task-gated-{index}',
        )
        await topology.stage()
        topologies.append(topology)

    await asyncio.gather(*(topology.start() for topology in topologies))

    assert 0 < _MeasuredRuntime.peak_services <= 48
