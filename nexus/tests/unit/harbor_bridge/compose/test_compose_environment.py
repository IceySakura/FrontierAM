from __future__ import annotations

import base64

from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus.extensions.harbor_bridge.compose import ComposePlan, ComposeServiceSummary, topology_spec_from_plan
from nexus.extensions.harbor_bridge.compose.environment import ComposeHarborEnvironment, ComposeServiceRuntime
from nexus.runtime.executors.models.file import FileItem


IMAGE = f'mirrors.tencent.com/rl-task/service@sha256:{"a" * 64}'
CONTROLLER_IMAGE = f'mirrors.tencent.com/rl-task/controller@sha256:{"b" * 64}'


class _FakeService:
    def __init__(self, name: str) -> None:
        self.name = name
        self.commands: list[dict] = []
        self.uploads: list[tuple[bytes, str]] = []
        self.directory_uploads: list[tuple[Path, str]] = []
        self.downloads: list[tuple[str, Path]] = []
        self.listed_files: list[str] = []
        self.download_content = b'downloaded'
        self.stopped = False

    async def exec(self, command: str, **kwargs):
        self.commands.append({'command': command, **kwargs})
        stdout = '\n'.join(self.listed_files) if ' -type f' in command else self.name
        return SimpleNamespace(return_code=0, stdout=stdout, stderr='')

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.uploads.append((Path(source_path).read_bytes(), target_path))

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        self.directory_uploads.append((Path(source_dir), target_dir))

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        target = Path(target_path)
        self.downloads.append((source_path, target))
        target.write_bytes(self.download_content)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / 'result.txt').write_text(self.name)

    async def stop(self, *, timeout: int = 30) -> None:
        self.stopped = True


class _FakeTopology:
    def __init__(self) -> None:
        self.services = {name: _FakeService(name) for name in ('app', 'db')}

    def service(self, name: str):
        return self.services[name]


def _plan() -> ComposePlan:
    return ComposePlan(
        definition='environment/docker-compose.lock.yaml',
        definition_sha256=f'sha256:{"c" * 64}',
        controller_image=CONTROLLER_IMAGE,
        primary_service='app',
        primary_image=IMAGE,
        workdir='/workdir',
        environment={'TASK_MODE': 'evaluation'},
        startup_timeout_sec=180,
        startup_jitter_max=17.5,
        start_retry_attempts=3,
        start_retry_backoff_sec=8.0,
        start_retry_jitter_max=60.0,
        shutdown_timeout_sec=30,
        services={
            'app': ComposeServiceSummary(image=IMAGE, has_healthcheck=True),
            'db': ComposeServiceSummary(image=IMAGE),
        },
        bind_paths=['environment/config'],
    )


def test_topology_spec_from_plan_keeps_only_runtime_contract() -> None:
    spec = topology_spec_from_plan(_plan())

    assert spec.primary_service == 'app'
    assert spec.startup_jitter_max == 17.5
    assert spec.start_retry_attempts == 3
    assert spec.start_retry_backoff_sec == 8.0
    assert spec.start_retry_jitter_max == 60.0
    assert spec.services['app'].image == IMAGE
    assert spec.services['app'].has_healthcheck is True
    assert spec.services['app'].depends_on == ()
    assert spec.bind_paths == ('environment/config',)
    assert spec.primary_environment == {'TASK_MODE': 'evaluation'}
    assert 'controller_image' not in type(spec).model_fields


@pytest.mark.asyncio
async def test_compose_service_runtime_translates_structured_file_operations() -> None:
    service = _FakeService('app')
    runtime = ComposeServiceRuntime(service)

    upload = await runtime.upload_file([
        FileItem(path='/workdir/input.bin', content=base64.b64encode(b'payload').decode(), encoding='base64')
    ])
    download = await runtime.download_file([FileItem(path='/workdir/output.bin', encoding='base64')])

    assert upload.results[0].success is True
    assert service.uploads == [(b'payload', '/workdir/input.bin')]
    assert download.errors is None
    assert base64.b64decode(download.files[0].content or '') == b'downloaded'


@pytest.mark.asyncio
async def test_compose_environment_routes_logical_main_and_sidecars(tmp_path: Path) -> None:
    topology = _FakeTopology()
    environment = object.__new__(ComposeHarborEnvironment)
    environment._topology = topology
    environment._primary_service = 'app'
    environment._default_timeout_sec = 30
    environment.default_user = None
    environment._merge_env = lambda env: {'TASK_MODE': 'evaluation', **(env or {})}

    main_result = await environment.service_exec('pwd', service='main', cwd='/workdir')
    sidecar_result = await environment.service_exec('dump-db', service='db', env={'FORMAT': 'sql'})
    target_dir = tmp_path / 'db'
    await environment.service_download_dir('/tmp/results', target_dir, service='db')
    await environment.stop_service('db')

    assert main_result.stdout == 'app'
    assert topology.services['app'].commands[0]['cwd'] == '/workdir'
    assert sidecar_result.stdout == 'db'
    assert topology.services['db'].commands[0]['env'] == {'FORMAT': 'sql'}
    assert (target_dir / 'result.txt').read_text() == 'db'
    assert topology.services['db'].stopped is True


@pytest.mark.asyncio
async def test_primary_operations_do_not_require_service_tmp(tmp_path: Path) -> None:
    topology = _FakeTopology()
    environment = object.__new__(ComposeHarborEnvironment)
    environment._topology = topology
    environment._primary_service = 'app'
    environment._default_timeout_sec = 30
    environment.default_user = None
    environment._merge_env = lambda env: {'PERSISTENT': '1', **(env or {})}
    source_dir = tmp_path / 'tests'
    source_dir.mkdir()
    (source_dir / 'test.sh').write_text('true')
    downloaded_dir = tmp_path / 'downloaded'
    tail_path = tmp_path / 'tail.txt'

    exec_result = await environment.exec(
        'printenv TOKEN',
        cwd='/workspace',
        env={'TOKEN': 'secret'},
        user='root',
    )
    await environment.upload_dir(source_dir, '/tests')
    await environment.download_dir('/workspace/output', downloaded_dir)
    await environment.download_file_tail('/logs/agent/stdout.txt', tail_path, max_bytes=1024)

    assert exec_result.stdout == 'app'
    primary = topology.services['app']
    assert primary.commands[0]['env'] == {'PERSISTENT': '1', 'TOKEN': 'secret'}
    assert primary.commands[0]['cwd'] == '/workspace'
    assert primary.commands[0]['user'] == 'root'
    assert primary.directory_uploads == [(source_dir, '/tests')]
    assert (downloaded_dir / 'result.txt').read_text() == 'app'
    assert tail_path.read_text() == 'app'
    assert '/nix/runtime/nix-env/bin/tail' in primary.commands[-1]['command']
    assert 'command -v tail' in primary.commands[-1]['command']
    assert not any('/tmp/.harbor_' in call['command'] for call in primary.commands)


@pytest.mark.asyncio
async def test_filtered_downloads_use_controller_staging_without_service_tmp(tmp_path: Path) -> None:
    topology = _FakeTopology()
    topology.services['app'].listed_files = ['./keep.txt', './skip.tmp', './protected.log']
    topology.services['db'].listed_files = ['./keep.json', './secret.bin']
    environment = object.__new__(ComposeHarborEnvironment)
    environment._topology = topology
    environment._primary_service = 'app'
    environment._default_timeout_sec = 30
    environment.default_user = None
    environment._merge_env = lambda env: dict(env or {})

    main_target = tmp_path / 'main'
    await environment.download_dir_filtered(
        source_dir='/logs',
        target_dir=main_target,
        include=['*.txt'],
        protect=['protected.log'],
    )
    sidecar_target = tmp_path / 'sidecar'
    await environment.service_download_dir_with_exclusions(
        source_dir='/dumps',
        target_dir=sidecar_target,
        exclude=['*.bin'],
        service='db',
    )

    assert [source for source, _ in topology.services['app'].downloads] == [
        '/logs/keep.txt',
        '/logs/protected.log',
    ]
    assert [source for source, _ in topology.services['db'].downloads] == ['/dumps/keep.json']
    assert (main_target / 'keep.txt').read_bytes() == b'downloaded'
    assert (main_target / 'protected.log').read_bytes() == b'downloaded'
    assert (sidecar_target / 'keep.json').read_bytes() == b'downloaded'
    assert not any(
        '/tmp/.hb-transfer-' in call['command'] for service in topology.services.values() for call in service.commands
    )


@pytest.mark.asyncio
async def test_compose_environment_preserves_service_command_failure() -> None:
    topology = _FakeTopology()
    environment = object.__new__(ComposeHarborEnvironment)
    environment._topology = topology
    environment._primary_service = 'app'
    environment._default_timeout_sec = 30
    environment.default_user = None
    environment._merge_env = lambda env: dict(env or {})

    async def failed_command(*_args, **_kwargs):
        return SimpleNamespace(return_code=7, stdout='', stderr='task command failed')

    topology.services['db'].exec = failed_command

    result = await environment.service_exec('collect', service='db')

    assert result.return_code == 7
    assert result.stderr == 'task command failed'


@pytest.mark.asyncio
async def test_compose_environment_propagates_service_infrastructure_error() -> None:
    topology = _FakeTopology()
    environment = object.__new__(ComposeHarborEnvironment)
    environment._topology = topology
    environment._primary_service = 'app'
    environment._default_timeout_sec = 30
    environment.default_user = None
    environment._merge_env = lambda env: dict(env or {})

    async def disconnected_runtime(*_args, **_kwargs):
        raise ConnectionError('controller disconnected')

    topology.services['db'].exec = disconnected_runtime

    with pytest.raises(ConnectionError, match='controller disconnected'):
        await environment.service_exec('collect', service='db')


@pytest.mark.asyncio
async def test_compose_environment_rejects_missing_service_exit_code() -> None:
    topology = _FakeTopology()
    environment = object.__new__(ComposeHarborEnvironment)
    environment._topology = topology
    environment._primary_service = 'app'
    environment._default_timeout_sec = 30
    environment.default_user = None
    environment._merge_env = lambda env: dict(env or {})

    async def timed_out_command(*_args, **_kwargs):
        return SimpleNamespace(return_code=None, stdout='', stderr='timed out')

    topology.services['db'].exec = timed_out_command

    with pytest.raises(RuntimeError, match="service 'db'.*did not return an exit code: timed out"):
        await environment.service_exec('collect', service='db')


def test_compose_environment_advertises_compose_capability(tmp_path: Path) -> None:
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    environment_dir = tmp_path / 'environment'
    environment_dir.mkdir()
    trial_paths = TrialPaths(tmp_path / 'trial')
    trial_paths.mkdir()

    environment = ComposeHarborEnvironment(
        topology=_FakeTopology(),
        primary_service='app',
        persistent_environment={'TASK_MODE': 'evaluation'},
        environment_dir=environment_dir,
        environment_name='compose-task',
        session_id='compose-session',
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(
            docker_image=CONTROLLER_IMAGE,
            workdir='/workdir',
        ),
    )

    assert environment.capabilities.docker_compose is True
    assert environment.capabilities.mounted is False
