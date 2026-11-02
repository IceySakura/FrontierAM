from __future__ import annotations

import base64
import io
import json
import tarfile

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harbor.models.agent.context import AgentContext
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from harbor.trial.multi_step import MultiStepTrial
from harbor.trial.single_step import SingleStepTrial
from harbor.trial.trial import Trial

from nexus.core import AgentRunOutput, ExecutionContext
from nexus.extensions.agents.installed_common.execution import get_agent_execution_target
from nexus.extensions.harbor_trial.agent import NexusAgent
from nexus.extensions.harbor_trial.context import HarborTrialBinding, harbor_trial_binding
from nexus.extensions.harbor_trial.environment import NexusEnvironment
from nexus.extensions.harbor_trial.task import HarborTrialTaskInput, harbor_trial_task
from nexus.runtime.context import get_execution_user
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse
from nexus.runtime.executors.models.file import FileItem
from nexus.runtime.providers.base import RuntimeProvider


class FakeAgentOutput(AgentRunOutput):
    final_answer: str


class FakeRuntime:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.download_requests: list[list[FileItem]] = []
        self.downloaded_files: dict[str, bytes] = {}
        self.default_download: bytes | None = None
        self.uploaded_files: dict[str, str] = {}

    async def run_command(self, *, command, timeout, cwd=None, extra_env=None):
        rendered = f'cd {cwd} && {command}' if cwd else command
        self.commands.append(rendered)
        stdout = ''
        if command == 'pwd':
            stdout = '/workspace\n'
        elif command.startswith("printf 'NO_PROXY=%s\\nno_proxy=%s\\n'"):
            stdout = 'NO_PROXY=outer.internal,localhost\nno_proxy=legacy.internal,outer.internal\n'
        elif 'docker compose' in command and ' config --format json' in command:
            main_source = (
                '"build":{"context":"/harbor/environment"}'
                if 'docker-compose-build.yaml' in command
                else '"image":"example/task:v1"'
            )
            stdout = (
                '{"services":{'
                f'"main":{{{main_source},"environment":{{"NO_PROXY":"task.internal"}}}},'
                '"db":{"build":{"context":"/harbor/environment/db"},"hostname":"database.internal",'
                '"environment":{"no_proxy":"db.internal"},'
                '"networks":{"default":{"aliases":["database","data.internal"]}}}'
                '}}'
            )
        return RunCommandResponse(status=CommandRunStatus.Finished, stdout=stdout, stderr='', return_code=0)

    async def upload_file(self, *, files, overwrite):
        for file in files:
            self.uploaded_files[file.path] = file.content
        return SimpleNamespace(results=[SimpleNamespace(path=file.path, success=True, message=None) for file in files])

    async def download_file(self, *, files):
        self.download_requests.append(files)
        downloaded = []
        errors = []
        for file in files:
            content = self.downloaded_files.get(file.path, self.default_download)
            if content is None:
                errors.append(SimpleNamespace(path=file.path, success=False, message='File does not exist'))
                continue
            downloaded.append(
                SimpleNamespace(
                    path=file.path,
                    content=base64.b64encode(content).decode('ascii'),
                    encoding='base64',
                )
            )
        return SimpleNamespace(files=downloaded, errors=errors or None)


class FakeRuntimeProvider(RuntimeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.acquire_calls = []
        self.release_calls = []
        self.runtimes: dict[str, FakeRuntime] = {}

    async def acquire(self, name: str, **params):
        merged = self.get_params(name, **params)
        self.acquire_calls.append((name, merged))
        runtime = FakeRuntime()
        self.runtimes[name] = runtime
        self._active_runtimes[name] = runtime
        return runtime

    async def _release_impl(self, name: str) -> None:
        self.release_calls.append(name)
        self._active_runtimes.pop(name, None)


def make_task_dir(root: Path, *, compose: bool = False) -> Path:
    task_dir = root / 'multi-step'
    (task_dir / 'environment').mkdir(parents=True)
    if compose:
        (task_dir / 'environment' / 'docker-compose.yaml').write_text(
            'services:\n  db:\n    image: postgres:17\n  main:\n    depends_on: [db]\n'
        )
    for name in ('first', 'second'):
        step_dir = task_dir / 'steps' / name
        (step_dir / 'tests').mkdir(parents=True)
        (step_dir / 'instruction.md').write_text(f'Do {name}.\n')
        (step_dir / 'tests' / 'test.sh').write_text('#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n')
    (task_dir / 'task.toml').write_text(
        '[environment]\n'
        'docker_image = "example/task:v1"\n'
        'workdir = "/workspace"\n'
        'cpus = 3\n'
        'memory_mb = 5120\n\n'
        '[[steps]]\n'
        'name = "first"\n\n'
        '[[steps]]\n'
        'name = "second"\n'
    )
    return task_dir


@pytest.mark.asyncio
async def test_nexus_agent_delegates_each_step_without_resume(tmp_path: Path) -> None:
    calls = []

    async def run_agent(**kwargs):
        calls.append({**kwargs, 'execution_user': get_execution_user(), 'target': get_agent_execution_target()})
        return FakeAgentOutput(final_answer=f'answer-{len(calls)}')

    run_agent.nexus_meta = {
        'name': 'supported',
        'meta': {'supports_execution_target': True, 'supports_execution_user': True},
    }
    context = ExecutionContext(task_config={}, tool_state={})
    binding = HarborTrialBinding(execution_context=context, run_agent=run_agent)
    async with harbor_trial_binding(binding):
        agent = NexusAgent(logs_dir=tmp_path / 'logs')
        environment = SimpleNamespace(
            task_env_config=SimpleNamespace(workdir='/workspace'),
            default_user='agent',
            exec=AsyncMock(),
        )
        first_context = AgentContext()
        second_context = AgentContext()
        await agent.run('first instruction', environment, first_context)
        assert first_context.is_empty()
        agent.populate_context_post_run(first_context)
        environment.default_user = 1001
        await agent.run('second instruction', environment, second_context)
        assert second_context.is_empty()
        agent.populate_context_post_run(second_context)
        agent.populate_context_post_run(first_context)

    assert agent.capabilities.resume is False
    assert [call['user_prompt'] for call in calls] == ['first instruction', 'second instruction']
    assert [call['execution_user'] for call in calls] == [None, None]
    assert [call['target'].user for call in calls] == ['agent', 1001]
    assert all(call['target'].kind == 'direct' for call in calls)
    assert all(call['working_dir'] == '/workspace' for call in calls)
    assert get_execution_user() is None
    assert first_context.metadata == {'nexus_agent_output': {'final_answer': 'answer-1'}}
    assert second_context.metadata == {'nexus_agent_output': {'final_answer': 'answer-2'}}


@pytest.mark.asyncio
async def test_nexus_environment_owns_single_runtime_lifecycle(tmp_path: Path, caplog) -> None:
    provider = FakeRuntimeProvider()
    context = ExecutionContext(runtime_provider=provider, run_id='run-1')
    task_dir = make_task_dir(tmp_path)
    trial_paths = TrialPaths(tmp_path / 'trial')
    trial_paths.mkdir()

    binding = HarborTrialBinding(execution_context=context, run_agent=AsyncMock())
    async with harbor_trial_binding(binding):
        environment = NexusEnvironment(
            environment_dir=task_dir / 'environment',
            environment_name='multi-step',
            session_id='trial-env',
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(
                docker_image='example/task:v1',
                workdir='/workspace',
                cpus=3,
                memory_mb=5120,
                storage_mb=10240,
            ),
        )
        await environment.start(force_build=False)
        await environment.stop(delete=True)

    assert 'Ignoring [environment].storage_mb=10240' in caplog.text
    assert provider.acquire_calls == [
        (
            'main',
            {
                'image': 'example/task:v1',
                'environment': {},
                'enable_dind': False,
                'resource_spec': {'cpu': 3, 'memory': 5},
                'working_dir': '/workspace',
                'run_id': 'run-1',
            },
        )
    ]
    assert provider.release_calls == ['main']


@pytest.mark.asyncio
async def test_nexus_environment_requires_an_environment_definition(tmp_path: Path) -> None:
    task_dir = make_task_dir(tmp_path)
    trial_paths = TrialPaths(tmp_path / 'trial')
    trial_paths.mkdir()
    binding = HarborTrialBinding(execution_context=ExecutionContext(), run_agent=AsyncMock())

    async with harbor_trial_binding(binding):
        with pytest.raises(FileNotFoundError, match='has no environment definition'):
            NexusEnvironment(
                environment_dir=task_dir / 'environment',
                environment_name='direct',
                session_id='trial-direct',
                trial_paths=trial_paths,
                task_env_config=EnvironmentConfig(workdir='/workspace'),
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'outer_params',
    [{}, {'image': 'example/outer:v1', 'runtime_image': 'example/nexus-runtime:v1'}],
    ids=['defer-to-provider', 'configured-outer-images'],
)
async def test_nexus_environment_runs_compose_inside_dind_runtime(tmp_path: Path, outer_params: dict[str, str]) -> None:
    provider = FakeRuntimeProvider()
    provider.configure('main', **outer_params)
    context = ExecutionContext(runtime_provider=provider, run_id='run-1')
    task_dir = make_task_dir(tmp_path, compose=True)
    trial_paths = TrialPaths(tmp_path / 'trial')
    trial_paths.mkdir()

    binding = HarborTrialBinding(execution_context=context, run_agent=AsyncMock())
    async with harbor_trial_binding(binding):
        environment = NexusEnvironment(
            environment_dir=task_dir / 'environment',
            environment_name='multi-step',
            session_id='trial-env',
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(
                docker_image='example/task:v1',
                workdir='/workspace',
                cpus=4,
                memory_mb=8192,
            ),
        )
        await environment.start(force_build=False)
        result = await environment.exec('cat /workspace/result.txt')
        target = environment.agent_execution_target()
        await environment.stop(delete=True)

    assert result.return_code == 0
    assert provider.acquire_calls == [
        (
            'main',
            {
                **outer_params,
                'environment': {},
                'enable_dind': True,
                'resource_spec': {'cpu': 4, 'memory': 8},
                'run_id': 'run-1',
            },
        )
    ]
    commands = provider.runtimes['main'].commands
    assert any('docker compose' in command and command.endswith(' build') for command in commands)
    assert any('docker compose' in command and ' up -d' in command for command in commands)
    assert all(
        '--project-directory /harbor/environment' in command for command in commands if 'docker compose' in command
    )
    readiness_commands = [command for command in commands if command.endswith('exec -T main true')]
    assert len(readiness_commands) == 1
    assert 'while ' not in readiness_commands[0]
    assert 'sleep 2' not in readiness_commands[0]
    assert any(
        'docker compose' in command and "exec -T -w /workspace main bash -lc 'cat /workspace/result.txt'" in command
        for command in commands
    )
    assert any('docker compose' in command and ' down --volumes --remove-orphans' in command for command in commands)
    assert target is not None
    assert target.service == 'main'
    assert target.project_directory == '/harbor/environment'
    assert '/harbor/compose/docker-compose-prebuilt.yaml' in target.compose_files
    assert target.compose_files[-1] == '/harbor/compose/docker-compose-no-proxy.json'
    assert target.compose_env['PREBUILT_IMAGE_NAME'] == 'example/task:v1'
    no_proxy_override = provider.runtimes['main'].uploaded_files['/harbor/compose/docker-compose-no-proxy.json']
    assert json.loads(no_proxy_override) == {
        'services': {
            'db': {
                'environment': {
                    'NO_PROXY': (
                        'outer.internal,localhost,legacy.internal,db.internal,127.0.0.1,::1,db,database.internal,'
                        'database,data.internal,main'
                    ),
                    'no_proxy': (
                        'outer.internal,localhost,legacy.internal,db.internal,127.0.0.1,::1,db,database.internal,'
                        'database,data.internal,main'
                    ),
                }
            },
            'main': {
                'environment': {
                    'NO_PROXY': (
                        'outer.internal,localhost,legacy.internal,task.internal,127.0.0.1,::1,db,database.internal,'
                        'database,data.internal,main'
                    ),
                    'no_proxy': (
                        'outer.internal,localhost,legacy.internal,task.internal,127.0.0.1,::1,db,database.internal,'
                        'database,data.internal,main'
                    ),
                }
            },
        }
    }
    assert provider.release_calls == ['main']


@pytest.mark.asyncio
async def test_nexus_environment_builds_compose_main_from_dockerfile(tmp_path: Path) -> None:
    provider = FakeRuntimeProvider()
    context = ExecutionContext(runtime_provider=provider, run_id='run-build')
    task_dir = make_task_dir(tmp_path, compose=True)
    (task_dir / 'environment' / 'Dockerfile').write_text('FROM ubuntu:24.04\nWORKDIR /workspace\n')
    trial_paths = TrialPaths(tmp_path / 'trial')
    trial_paths.mkdir()

    binding = HarborTrialBinding(execution_context=context, run_agent=AsyncMock())
    async with harbor_trial_binding(binding):
        environment = NexusEnvironment(
            environment_dir=task_dir / 'environment',
            environment_name='build-main',
            session_id='trial-build',
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(workdir='/workspace'),
        )
        await environment.start(force_build=False)
        target = environment.agent_execution_target()
        await environment.stop(delete=True)

    assert provider.acquire_calls == [
        (
            'main',
            {
                'environment': {},
                'enable_dind': True,
                'run_id': 'run-build',
            },
        )
    ]
    commands = provider.runtimes['main'].commands
    assert any('docker compose' in command and command.endswith(' build') for command in commands)
    assert any('docker compose' in command and ' up -d' in command for command in commands)
    assert target is not None
    assert target.project_directory == '/harbor/environment'
    assert '/harbor/compose/docker-compose-build.yaml' in target.compose_files
    assert 'PREBUILT_IMAGE_NAME' not in target.compose_env
    assert provider.release_calls == ['main']


@pytest.mark.asyncio
async def test_nexus_agent_selects_execution_target_without_changing_agent_input(tmp_path: Path) -> None:
    calls = []

    async def run_agent(**kwargs):
        calls.append((kwargs, get_agent_execution_target(), get_execution_user()))
        return FakeAgentOutput(final_answer='done')

    run_agent.nexus_meta = {  # type: ignore[attr-defined]
        'name': 'supported',
        'type': 'agent',
        'meta': {'supports_execution_target': True, 'supports_execution_user': True},
    }
    target = SimpleNamespace(service='main', user='agent')
    environment = SimpleNamespace(
        task_env_config=SimpleNamespace(workdir='/workspace'),
        default_user='agent',
        agent_execution_target=lambda user: SimpleNamespace(service=target.service, user=user),
    )
    binding = HarborTrialBinding(execution_context=ExecutionContext(task_config={}, tool_state={}), run_agent=run_agent)
    async with harbor_trial_binding(binding):
        await NexusAgent(logs_dir=tmp_path / 'logs').run('implement', environment, AgentContext())

    kwargs, execution_target, runtime_user = calls[0]
    assert kwargs == {'user_prompt': 'implement', 'working_dir': '/workspace'}
    assert execution_target.service == 'main'
    assert execution_target.user == 'agent'
    assert runtime_user is None
    assert get_agent_execution_target() is None
    assert get_execution_user() is None


@pytest.mark.asyncio
async def test_nexus_agent_rejects_execution_target_for_unsupported_agent(tmp_path: Path) -> None:
    run_agent = AsyncMock()
    run_agent.nexus_meta = {'name': 'unsupported', 'type': 'agent', 'meta': {}}
    environment = SimpleNamespace(
        task_env_config=SimpleNamespace(workdir='/workspace'),
        default_user='agent',
        agent_execution_target=lambda user: SimpleNamespace(service='main', user=user),
    )
    binding = HarborTrialBinding(execution_context=ExecutionContext(task_config={}, tool_state={}), run_agent=run_agent)

    async with harbor_trial_binding(binding):
        with pytest.raises(ValueError, match="Agent 'unsupported' does not support the selected execution target"):
            await NexusAgent(logs_dir=tmp_path / 'logs').run('implement', environment, AgentContext())

    run_agent.assert_not_awaited()
    assert get_agent_execution_target() is None
    assert get_execution_user() is None


@pytest.mark.asyncio
async def test_nexus_environment_exec_supports_numeric_uid(tmp_path: Path) -> None:
    provider = FakeRuntimeProvider()
    context = ExecutionContext(runtime_provider=provider)
    task_dir = make_task_dir(tmp_path)
    trial_paths = TrialPaths(tmp_path / 'trial')
    trial_paths.mkdir()

    binding = HarborTrialBinding(execution_context=context, run_agent=AsyncMock())
    async with harbor_trial_binding(binding):
        environment = NexusEnvironment(
            environment_dir=task_dir / 'environment',
            environment_name='multi-step',
            session_id='trial-env',
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(docker_image='example/task:v1', workdir='/workspace'),
        )
        await environment.start(force_build=False)
        await environment.exec('whoami', user=1001)

    assert provider.runtimes['main'].commands[-1] == (
        'su "$(getent passwd 1001 | cut -d: -f1)" -s /bin/sh -c \'cd /workspace && whoami\''
    )


@pytest.mark.asyncio
async def test_nexus_environment_exec_grants_non_root_user_access_to_scoped_env(tmp_path: Path) -> None:
    provider = FakeRuntimeProvider()
    context = ExecutionContext(runtime_provider=provider)
    task_dir = make_task_dir(tmp_path)
    trial_paths = TrialPaths(tmp_path / 'trial')
    trial_paths.mkdir()

    binding = HarborTrialBinding(execution_context=context, run_agent=AsyncMock())
    async with harbor_trial_binding(binding):
        environment = NexusEnvironment(
            environment_dir=task_dir / 'environment',
            environment_name='multi-step',
            session_id='trial-env',
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(docker_image='example/task:v1', workdir='/workspace'),
        )
        await environment.start(force_build=False)
        await environment.exec('printf "$TOKEN"', env={'TOKEN': 'secret'}, user='agent')

    commands = provider.runtimes['main'].commands
    permission_command = next(command for command in commands if command.startswith('chown agent '))
    assert ' && chmod 700 ' in permission_command
    assert ' && chmod 600 ' in permission_command
    assert any(
        command.startswith("su agent -s /bin/sh -c '") and '.harbor_exec_env_' in command for command in commands
    )


@pytest.mark.asyncio
async def test_nexus_environment_downloads_files_through_runtime_api(tmp_path: Path) -> None:
    provider = FakeRuntimeProvider()
    context = ExecutionContext(runtime_provider=provider)
    task_dir = make_task_dir(tmp_path)
    trial_paths = TrialPaths(tmp_path / 'trial')
    trial_paths.mkdir()

    binding = HarborTrialBinding(execution_context=context, run_agent=AsyncMock())
    async with harbor_trial_binding(binding):
        environment = NexusEnvironment(
            environment_dir=task_dir / 'environment',
            environment_name='multi-step',
            session_id='trial-env',
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(docker_image='example/task:v1', workdir='/workspace'),
        )
        await environment.start(force_build=False)
        runtime = provider.runtimes['main']

        binary_content = b'\x00\xffnexus\n'
        runtime.downloaded_files['/workspace/result.bin'] = binary_content
        file_target = tmp_path / 'downloads' / 'result.bin'
        await environment.download_file('/workspace/result.bin', file_target)

        archive_buffer = io.BytesIO()
        archive_content = b'artifact-content'
        with tarfile.open(fileobj=archive_buffer, mode='w:gz') as archive:
            entry = tarfile.TarInfo('nested/artifact.bin')
            entry.size = len(archive_content)
            archive.addfile(entry, io.BytesIO(archive_content))
        runtime.default_download = archive_buffer.getvalue()
        directory_target = tmp_path / 'artifacts'
        await environment.download_dir('/logs/artifacts', directory_target)

    assert file_target.read_bytes() == binary_content
    assert (directory_target / 'nested' / 'artifact.bin').read_bytes() == archive_content
    assert runtime.download_requests[0][0].path == '/workspace/result.bin'
    assert runtime.download_requests[0][0].encoding == 'base64'
    assert runtime.download_requests[1][0].path.startswith('/tmp/.harbor_download_')
    assert runtime.download_requests[1][0].encoding == 'base64'
    assert not any(command.startswith('base64 ') for command in runtime.commands)


@pytest.mark.asyncio
async def test_separate_verifier_environment_uses_its_own_runtime(tmp_path: Path) -> None:
    provider = FakeRuntimeProvider()
    context = ExecutionContext(runtime_provider=provider, run_id='run-1')
    task_dir = make_task_dir(tmp_path)
    trial_paths = TrialPaths(tmp_path / 'trial')
    trial_paths.mkdir()

    binding = HarborTrialBinding(execution_context=context, run_agent=AsyncMock())
    async with harbor_trial_binding(binding):
        agent_environment = NexusEnvironment(
            environment_dir=task_dir / 'environment',
            environment_name='multi-step',
            session_id='trial__env',
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(docker_image='example/agent:v1', workdir='/workspace'),
        )
        verifier_environment = NexusEnvironment(
            environment_dir=task_dir / 'steps' / 'first' / 'tests',
            environment_name='multi-step',
            session_id='trial__verifier__step-first',
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(docker_image='example/verifier:v1', workdir='/tests'),
        )
        await agent_environment.start(force_build=False)
        await verifier_environment.start(force_build=False)
        await verifier_environment.stop(delete=True)
        await agent_environment.stop(delete=True)

    runtime_names = [name for name, _params in provider.acquire_calls]
    assert runtime_names[0] == 'main'
    assert runtime_names[1].startswith('verifier_trial__verifier__step-first_')
    assert runtime_names[0] != runtime_names[1]
    assert provider.acquire_calls[1][1]['image'] == 'example/verifier:v1'
    assert any(
        'tar xzf' in command and '-C /tests' in command for command in provider.runtimes[runtime_names[1]].commands
    )
    assert provider.release_calls == [runtime_names[1], 'main']


@pytest.mark.asyncio
@pytest.mark.parametrize('force_build', [False, True])
@pytest.mark.parametrize('agent_name', [None, 'claude_code', 'codex'])
async def test_task_runs_native_trial_and_preserves_step_agent_outputs(
    tmp_path: Path, monkeypatch, force_build, agent_name
) -> None:
    task_dir = make_task_dir(tmp_path)
    provider = FakeRuntimeProvider()
    context = ExecutionContext(runtime_provider=provider, task_config={}, tool_state={})
    created = []

    class FakeResult:
        exception_info = None
        verifier_result = SimpleNamespace(rewards={'reward': 0.75, 'quality': 1})
        step_results = [
            SimpleNamespace(
                step_name='first',
                agent_result=SimpleNamespace(metadata={'nexus_agent_output': {'final_answer': 'one'}}),
            ),
            SimpleNamespace(
                step_name='second',
                agent_result=SimpleNamespace(metadata={'nexus_agent_output': {'final_answer': 'two'}}),
            ),
        ]
        agent_result = None

        def model_dump(self, *, mode):
            assert mode == 'json'
            return {'task_name': 'multi-step', 'step_count': 2}

    class FakeTrial:
        @classmethod
        async def create(cls, config):
            from harbor.models.task.task import Task

            created.append(config)
            trial = cls()
            trial.task = Task(task_dir=config.task.path)
            return trial

        async def run(self):
            return FakeResult()

    monkeypatch.setattr('harbor.trial.trial.Trial', FakeTrial)

    async def run_agent(**kwargs):
        return FakeAgentOutput(final_answer='unused')

    if agent_name is not None:
        run_agent.nexus_meta = {'name': agent_name, 'meta': {}}

    task_input = HarborTrialTaskInput(
        task_dir=str(task_dir),
        instance_id='example/task',
        run_agent=run_agent,
        force_build=force_build,
    )
    async with context:
        output = await harbor_trial_task(task_input)

    assert output.score == 0.75
    assert output.rewards == {'reward': 0.75, 'quality': 1.0}
    assert output.step_agent_outputs == [
        {'step_name': 'first', 'output': {'final_answer': 'one'}},
        {'step_name': 'second', 'output': {'final_answer': 'two'}},
    ]
    expected_class = 'NexusCodexAgent' if agent_name == 'codex' else 'NexusAgent'
    assert created[0].agent.import_path == f'nexus.extensions.harbor_trial.agent:{expected_class}'
    assert created[0].agent.resume_trajectory is False
    assert created[0].environment.import_path == 'nexus.extensions.harbor_trial.environment:NexusEnvironment'
    assert created[0].environment.force_build is force_build


@pytest.mark.asyncio
async def test_native_trial_factory_selects_multi_step_trial(tmp_path: Path) -> None:
    task_dir = make_task_dir(tmp_path)
    provider = FakeRuntimeProvider()
    context = ExecutionContext(runtime_provider=provider)
    binding = HarborTrialBinding(execution_context=context, run_agent=AsyncMock())

    config = TrialConfig(
        task=TaskConfig(path=task_dir),
        trials_dir=tmp_path / 'trials',
        agent=AgentConfig(import_path='nexus.extensions.harbor_trial.agent:NexusAgent'),
        environment=TrialEnvironmentConfig(import_path='nexus.extensions.harbor_trial.environment:NexusEnvironment'),
    )
    async with harbor_trial_binding(binding):
        trial = await Trial.create(config)

    assert isinstance(trial, MultiStepTrial)
    assert isinstance(trial.agent, NexusAgent)
    assert isinstance(trial.agent_environment, NexusEnvironment)


@pytest.mark.asyncio
async def test_native_trial_factory_allows_single_step_task(tmp_path: Path) -> None:
    task_dir = tmp_path / 'single-step'
    (task_dir / 'environment').mkdir(parents=True)
    (task_dir / 'tests').mkdir()
    (task_dir / 'task.toml').write_text('[environment]\ndocker_image = "example/task:v1"\n')
    (task_dir / 'instruction.md').write_text('Do one thing.\n')
    (task_dir / 'tests' / 'test.sh').write_text('#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n')

    provider = FakeRuntimeProvider()
    context = ExecutionContext(runtime_provider=provider)
    binding = HarborTrialBinding(execution_context=context, run_agent=AsyncMock())
    config = TrialConfig(
        task=TaskConfig(path=task_dir),
        trials_dir=tmp_path / 'trials',
        agent=AgentConfig(import_path='nexus.extensions.harbor_trial.agent:NexusAgent'),
        environment=TrialEnvironmentConfig(import_path='nexus.extensions.harbor_trial.environment:NexusEnvironment'),
    )

    async with harbor_trial_binding(binding):
        trial = await Trial.create(config)

    assert isinstance(trial, SingleStepTrial)
