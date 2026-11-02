from __future__ import annotations

import base64
import io
import json
import tarfile

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from harbor.environments.base import ExecResult
from harbor.environments.docker import ENV_COMPOSE_NAME
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from tests.unit.harbor_trial.test_harbor_trial import FakeRuntime, FakeRuntimeProvider, make_task_dir
from yaml import safe_load

from nexus.core import ExecutionContext
from nexus.extensions.harbor_trial.context import HarborTrialBinding, harbor_trial_binding
from nexus.extensions.harbor_trial.environment import NexusEnvironment


@pytest.fixture
async def environment_factory(tmp_path: Path):
    provider = FakeRuntimeProvider()
    provider.configure('main', image='example/outer:v1')
    binding = HarborTrialBinding(execution_context=ExecutionContext(runtime_provider=provider), run_agent=AsyncMock())
    task_dir = make_task_dir(tmp_path)
    environment_dir = task_dir / 'environment'
    paths = TrialPaths(tmp_path / 'trial')
    paths.mkdir()

    def create(*, docker_image=None, compose=True, dockerfile=False, extra_compose=False, **env_kwargs):
        if compose:
            (environment_dir / 'docker-compose.yaml').write_text('services:\n  main:\n    command: sleep infinity\n')
        if dockerfile:
            (environment_dir / 'Dockerfile').write_text('FROM ubuntu:24.04\n')
        extra_paths = []
        if extra_compose:
            overlay = tmp_path / 'extra.yaml'
            overlay.write_text('services:\n  main:\n    command: sleep infinity\n')
            extra_paths.append(overlay)
        return NexusEnvironment(
            environment_dir=environment_dir,
            environment_name='build-test',
            session_id='trial-build-test',
            trial_paths=paths,
            task_env_config=EnvironmentConfig(docker_image=docker_image, build_timeout_sec=321, **env_kwargs),
            extra_docker_compose=extra_paths,
        )

    async with harbor_trial_binding(binding):
        yield create, provider


@pytest.mark.parametrize('docker_image', [None, 'example/task:v1'])
@pytest.mark.parametrize('dockerfile', [False, True])
@pytest.mark.parametrize('force_build', [False, True])
async def test_compose_uses_native_image_selection(environment_factory, docker_image, dockerfile, force_build):
    create, provider = environment_factory
    environment = create(docker_image=docker_image, dockerfile=dockerfile)
    await environment.start(force_build=force_build)
    target = environment.agent_execution_target()
    assert target is not None
    prebuilt = bool(docker_image) and not (force_build and dockerfile)
    template = f'docker-compose-{"prebuilt" if prebuilt else "build"}.yaml'
    runtime = provider.runtimes['main']
    assert f'/harbor/compose/{template}' in target.compose_files
    assert f'/harbor/compose/{template}' in runtime.uploaded_files
    assert target.compose_env.get('PREBUILT_IMAGE_NAME') == (docker_image if prebuilt else None)
    assert provider.acquire_calls[0][1]['image'] == 'example/outer:v1'
    assert provider.acquire_calls[0][1]['enable_dind'] is True
    builds = [command for command in runtime.commands if command.endswith(' build')]
    assert len(builds) == 1
    assert template in builds[0]
    assert '--no-cache' not in builds[0]
    await environment.stop(delete=True)


@pytest.mark.parametrize('extra_compose', [False, True], ids=['task-compose', 'extra-compose'])
async def test_compose_accepts_custom_build_path_without_root_dockerfile(
    environment_factory, monkeypatch, extra_compose
):
    create, provider = environment_factory
    environment = create(compose=not extra_compose, extra_compose=extra_compose)
    app = environment.environment_dir / 'app'
    app.mkdir()
    (app / 'Dockerfile.runtime').write_text('FROM ubuntu:24.04\n')
    compose_path = (
        environment.extra_docker_compose_paths[0]
        if extra_compose
        else environment.environment_dir / 'docker-compose.yaml'
    )
    compose_path.write_text(
        'services:\n  main:\n    image: task-main:local\n    build:\n'
        '      context: ./app\n      dockerfile: Dockerfile.runtime\n'
    )
    dind = environment._dind
    assert dind is not None
    definition = {
        'services': {
            'main': {
                'image': 'task-main:local',
                'build': {'context': '/harbor/environment/app', 'dockerfile': 'Dockerfile.runtime'},
            }
        }
    }
    resolve = AsyncMock(return_value=definition)
    monkeypatch.setattr(dind, '_resolved_compose_definition', resolve)

    await environment.start(force_build=False)

    assert not (environment.environment_dir / 'Dockerfile').exists()
    assert environment.task_env_config.docker_image is None
    assert resolve.await_count == 2
    runtime = provider.runtimes['main']
    if extra_compose:
        assert runtime.uploaded_files['/harbor/compose/docker-compose-extra-0.yaml'] == compose_path.read_text()
    archives = [content for path, content in runtime.uploaded_files.items() if path.endswith('.tar.gz')]
    assert len(archives) == 1
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(archives[0])), mode='r:gz') as archive:
        assert './app/Dockerfile.runtime' in archive.getnames()
    assert any(command.endswith(' build') for command in runtime.commands)
    assert any('--project-directory /harbor/environment' in command for command in runtime.commands)
    await environment.stop(delete=True)


@pytest.mark.parametrize(
    'services',
    [
        {'main': {'image': 'task-main:local', 'build': {'context': './app'}}},
        {
            'main': {'image': 'example/task:v1'},
            'sidecar': {'image': 'sidecar:local', 'build': {'context': './sidecar'}},
            'db': {'image': 'postgres:17'},
        },
        {'main': {'image': 'example/task:v1'}, 'db': {'image': 'postgres:17'}},
    ],
    ids=['main-image-and-build', 'sidecar-image-and-build', 'prebuilt-only'],
)
async def test_compose_delegates_build_selection_to_compose(environment_factory, monkeypatch, services):
    create, _ = environment_factory
    environment = create(docker_image='example/task:v1')
    dind = environment._dind
    assert dind is not None
    definition = {'services': services}

    async def compose_exec(subcommand, timeout_sec=None):
        return ExecResult(
            stdout=json.dumps(definition) if subcommand == ['config', '--format', 'json'] else '',
            return_code=0,
        )

    execute = AsyncMock(side_effect=compose_exec)
    monkeypatch.setattr(dind, '_compose_exec', execute)
    await environment.start(force_build=False)

    assert [call.args[0] for call in execute.await_args_list] == [
        ['config', '--format', 'json'],
        ['config', '--format', 'json'],
        ['build'],
        ['up', '-d'],
    ]
    assert execute.await_args_list[2].kwargs == {'timeout_sec': None}
    assert execute.await_args_list[3].kwargs == {'timeout_sec': None}
    await environment.stop(delete=True)


@pytest.mark.parametrize(
    'definition',
    [{}, {'services': {'db': {'image': 'postgres:17'}}}, {'services': {'main': {'command': 'sleep infinity'}}}],
    ids=['missing-services', 'missing-main', 'missing-image-and-build'],
)
async def test_compose_rejects_invalid_effective_definition_before_build(environment_factory, monkeypatch, definition):
    create, provider = environment_factory
    environment = create()
    dind = environment._dind
    assert dind is not None
    monkeypatch.setattr(dind, '_resolved_compose_definition', AsyncMock(return_value=definition))
    with pytest.raises(ValueError, match='Harbor Compose'):
        await environment.start(force_build=False)
    assert not any(
        command.endswith(' build') or ' up ' in command
        for command in provider.runtimes['main'].commands
    )
    await environment.stop(delete=True)


@pytest.mark.parametrize('failure', ['config', 'build'])
async def test_compose_failure_does_not_start_services(environment_factory, monkeypatch, failure):
    create, _ = environment_factory
    environment = create(docker_image='example/task:v1')
    dind = environment._dind
    assert dind is not None

    async def compose_exec(subcommand, timeout_sec=None):
        if subcommand[0] == failure:
            return ExecResult(stderr='invalid build definition', return_code=1)
        return ExecResult(stdout='{"services":{"main":{"image":"example/task:v1"}}}', return_code=0)

    execute = AsyncMock(side_effect=compose_exec)
    monkeypatch.setattr(dind, '_compose_exec', execute)
    with pytest.raises(RuntimeError, match=f'docker compose {failure} failed'):
        await environment.start(force_build=False)
    assert all(call.args[0][0] != 'up' for call in execute.await_args_list)
    await environment.stop(delete=True)


@pytest.mark.parametrize('docker_image', [None, 'example/task:v1'])
@pytest.mark.parametrize('dockerfile', [False, True])
@pytest.mark.parametrize('force_build', [False, True])
async def test_no_compose_selects_direct_or_single_service_dind(
    environment_factory, docker_image, dockerfile, force_build
):
    create, provider = environment_factory
    if not docker_image and not dockerfile:
        with pytest.raises(FileNotFoundError, match='has no environment definition'):
            create(compose=False)
        assert provider.acquire_calls == []
        return

    environment = create(compose=False, docker_image=docker_image, dockerfile=dockerfile)
    provider.configure('main', image='example/outer:v1', runtime_image='example/nexus-runtime:v1')
    await environment.start(force_build=force_build)
    await environment.start(force_build=force_build)

    use_dind = not docker_image or (force_build and dockerfile)
    assert len(provider.acquire_calls) == 1
    params = provider.acquire_calls[0][1]
    assert params['enable_dind'] is use_dind
    assert params['image'] == ('example/outer:v1' if use_dind else docker_image)
    assert params['runtime_image'] == 'example/nexus-runtime:v1'
    assert environment._uses_compose is use_dind
    target = environment.agent_execution_target()
    commands = provider.runtimes['main'].commands
    if use_dind:
        assert target is not None
        assert target.service == 'main'
        assert '/harbor/compose/docker-compose-build.yaml' in target.compose_files
        assert '/harbor/environment/docker-compose.yaml' not in target.compose_files
        assert 'PREBUILT_IMAGE_NAME' not in target.compose_env
        assert any(command.endswith(' build') for command in commands)
        assert any(command.endswith(' up -d') for command in commands)
    else:
        assert target is None
        assert not any('docker compose' in command for command in commands)
    await environment.stop(delete=True)
    assert provider.release_calls == ['main']


@pytest.mark.parametrize('docker_image', [None, 'example/task:v1'], ids=['dockerfile-only', 'force-build'])
async def test_single_service_dind_reuses_main_lifecycle(environment_factory, tmp_path, monkeypatch, docker_image):
    create, provider = environment_factory
    environment = create(
        compose=False,
        dockerfile=True,
        docker_image=docker_image,
        workdir='/workspace',
        env={'TASK_VALUE': 'inner-value'},
        cpus=3,
        memory_mb=5120,
    )
    original_run_command = FakeRuntime.run_command

    async def run_command(runtime, **kwargs):
        result = await original_run_command(runtime, **kwargs)
        if ' config --format json' in kwargs['command']:
            result.stdout = json.dumps(safe_load(runtime.uploaded_files['/harbor/compose/docker-compose-build.yaml']))
        return result

    monkeypatch.setattr(FakeRuntime, 'run_command', run_command)
    await environment.start(force_build=bool(docker_image))
    params = provider.acquire_calls[0][1]
    assert params['image'] == 'example/outer:v1'
    assert params['enable_dind'] is True
    assert params['environment'] == {}
    assert params['resource_spec'] == {'cpu': 3, 'memory': 5}
    assert 'working_dir' not in params

    runtime = provider.runtimes['main']
    target = environment.agent_execution_target()
    assert target is not None
    assert target.service == 'main'
    assert target.compose_env['CONTEXT_DIR'] == '/harbor/environment'
    for path in target.compose_files:
        definition = safe_load(runtime.uploaded_files[path])
        assert set(definition['services']) == {'main'}
    template = safe_load(runtime.uploaded_files['/harbor/compose/docker-compose-build.yaml'])
    assert template['services']['main']['build'] == {'context': '${CONTEXT_DIR}'}
    assert template['services']['main']['command'] == ['sh', '-c', 'sleep infinity']
    startup_env = safe_load(runtime.uploaded_files[f'/harbor/compose/{ENV_COMPOSE_NAME}'])
    assert startup_env['services']['main']['environment']['TASK_VALUE'] == 'inner-value'
    archives = [content for path, content in runtime.uploaded_files.items() if path.endswith('.tar.gz')]
    assert len(archives) == 1
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(archives[0])), mode='r:gz') as archive:
        assert './Dockerfile' in archive.getnames()

    await environment.exec('cat result.txt')
    assert any('exec -T -w /workspace' in command and 'main bash -lc' in command for command in runtime.commands)
    source = tmp_path / 'input.txt'
    source.write_text('input')
    await environment.upload_file(source, '/workspace/input.txt')
    assert any(' cp ' in command and 'main:/workspace/input.txt' in command for command in runtime.commands)
    runtime.default_download = b'result'
    destination = tmp_path / 'result.txt'
    await environment.download_file('/workspace/result.txt', destination)
    assert destination.read_bytes() == b'result'
    assert any(' cp main:/workspace/result.txt ' in command for command in runtime.commands)
    await environment.stop(delete=True)
    assert any(command.endswith(' down --volumes --remove-orphans') for command in runtime.commands)
    assert provider.release_calls == ['main']
