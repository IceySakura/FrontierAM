from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from harbor.models.task.config import EnvironmentConfig, TaskConfig
from harbor.models.trial.paths import TrialPaths
from harbor.trial.trial import Trial
from tests.unit.harbor_trial.test_harbor_trial import make_task_dir
from tests.unit.harbor_trial.test_network import (
    ALLOWLIST,
    DENY_ALL,
    OFFLINE,
    PUBLIC,
    RecordingGongfengProvider,
    make_config,
)

from nexus.core import ExecutionContext
from nexus.extensions.harbor_trial.context import HarborTrialBinding, harbor_trial_binding
from nexus.extensions.harbor_trial.environment import NexusEnvironment
from nexus.extensions.harbor_trial.network import (
    network_capabilities,
    runtime_network_params,
    validate_separate_verifier_networks,
)
from nexus.extensions.harbor_trial.task import HarborTrialTaskInput, harbor_trial_task


def test_dind_capabilities_match_daytona_static_isolation():
    capabilities = network_capabilities(RecordingGongfengProvider(), compose_mode=True)
    assert capabilities.disable_internet
    assert not capabilities.dynamic_network_policy
    assert all(not value for key, value in capabilities.model_dump().items() if key.startswith('network_allowlist'))


@pytest.mark.parametrize('policy', [PUBLIC, OFFLINE])
def test_dind_outer_sandbox_stays_online(policy):
    provider = RecordingGongfengProvider()
    assert runtime_network_params(policy, provider, compose_mode=True) == {'network_policy': None}


def test_dind_rejects_allowlist_instead_of_widening_policy():
    with pytest.raises(ValueError, match="allowlist.*not supported.*DinD"):
        runtime_network_params(ALLOWLIST, RecordingGongfengProvider(), compose_mode=True)


@pytest.mark.parametrize('source', ['provider', 'runtime'])
def test_dind_rejects_offline_outer_overrides(source):
    provider = RecordingGongfengProvider(network_policy=DENY_ALL if source == 'provider' else None)
    if source == 'runtime':
        provider.configure('main', network_policy=DENY_ALL)
    with pytest.raises(ValueError, match='online outer Sandbox'):
        runtime_network_params(OFFLINE, provider, runtime_name='main', compose_mode=True)


@pytest.mark.parametrize('mode', ['compose', 'extra-compose', 'dockerfile-only', 'force-build'])
async def test_dind_rejects_allowlist_before_acquire(tmp_path, mode):
    provider = RecordingGongfengProvider()
    task_dir = make_task_dir(tmp_path, compose=mode == 'compose')
    if mode in {'dockerfile-only', 'force-build'}:
        (task_dir / 'environment' / 'Dockerfile').write_text('FROM ubuntu:24.04\n')
    extra_paths = []
    if mode == 'extra-compose':
        overlay = tmp_path / 'extra.yaml'
        overlay.write_text('services:\n  main:\n    command: sleep infinity\n')
        extra_paths.append(overlay)

    def create_environment():
        return NexusEnvironment(
            environment_dir=task_dir / 'environment',
            environment_name='test',
            session_id='trial-env',
            trial_paths=TrialPaths(tmp_path / 'trial'),
            task_env_config=EnvironmentConfig(docker_image=None if mode == 'dockerfile-only' else 'example/task:v1'),
            extra_docker_compose=extra_paths,
            network_policy=ALLOWLIST,
        )

    binding = HarborTrialBinding(execution_context=ExecutionContext(runtime_provider=provider), run_agent=AsyncMock())
    async with harbor_trial_binding(binding):
        if mode == 'force-build':
            environment = create_environment()
            with pytest.raises(ValueError, match='allowlist.*not supported by nexus'):
                await environment.start(force_build=True)
        else:
            with pytest.raises(ValueError, match='allowlist.*not supported by nexus'):
                create_environment()
    assert provider.acquire_calls == []


async def test_force_build_checks_outer_override_after_selecting_dind(tmp_path):
    provider = RecordingGongfengProvider(network_policy={'default_action': 'allow'})
    task_dir = make_task_dir(tmp_path)
    (task_dir / 'environment' / 'Dockerfile').write_text('FROM ubuntu:24.04\n')
    paths = TrialPaths(tmp_path / 'trial')
    paths.mkdir()
    binding = HarborTrialBinding(execution_context=ExecutionContext(runtime_provider=provider), run_agent=AsyncMock())
    async with harbor_trial_binding(binding):
        environment = NexusEnvironment(
            environment_dir=task_dir / 'environment',
            environment_name='test',
            session_id='trial-env',
            trial_paths=paths,
            task_env_config=EnvironmentConfig(docker_image='example/task:v1'),
            network_policy=OFFLINE,
        )
        await environment.start(force_build=True)
        assert provider.acquire_calls[0][1]['network_policy'] is None
        assert provider.acquire_calls[0][1]['enable_dind'] is True
        await environment.stop(delete=True)


@pytest.mark.parametrize('phase', ['agent', 'verifier'])
async def test_native_dind_rejects_phase_switch_before_start(tmp_path, phase):
    config = make_config(
        tmp_path,
        '[environment]\ndocker_image = "example/task:v1"\n'
        f'[{phase}]\nnetwork_mode = "no-network"\n',
        compose=True,
    )
    provider = RecordingGongfengProvider()
    binding = HarborTrialBinding(execution_context=ExecutionContext(runtime_provider=provider), run_agent=AsyncMock())
    async with harbor_trial_binding(binding):
        with pytest.raises(ValueError, match='cannot change network policy'):
            await Trial.create(config)
    assert provider.acquire_calls == []


@pytest.mark.parametrize('definition', ['compose', 'dockerfile-only'])
@pytest.mark.parametrize('step_level', [False, True])
async def test_task_preflights_separate_dind_allowlist_before_agent_run(tmp_path, monkeypatch, definition, step_level):
    config = make_config(
        tmp_path,
        '[environment]\ndocker_image = "example/task:v1"\n'
        + ('[[steps]]\nname = "first"\n[steps.verifier.environment]\n' if step_level else '[verifier.environment]\n')
        + ('docker_image = "example/verifier:v1"\n' if definition == 'compose' else '')
        + 'network_mode = "allowlist"\nallowed_hosts = ["example.com"]\n',
        multi_step=step_level,
    )
    tests_dir = config.task.path / 'steps' / 'first' / 'tests' if step_level else config.task.path / 'tests'
    if definition == 'compose':
        (tests_dir / 'docker-compose.yaml').write_text('services:\n  main:\n    command: sleep infinity\n')
    else:
        (tests_dir / 'Dockerfile').write_text('FROM ubuntu:24.04\n')
    provider = RecordingGongfengProvider()
    run_agent = AsyncMock(nexus_meta={})
    run_trial = AsyncMock()
    monkeypatch.setattr(Trial, 'run', run_trial)
    async with ExecutionContext(runtime_provider=provider):
        with pytest.raises(ValueError, match='allowlist.*not supported.*DinD'):
            await harbor_trial_task(HarborTrialTaskInput(task_dir=str(config.task.path), run_agent=run_agent))
    assert provider.acquire_calls == []
    run_agent.assert_not_awaited()
    run_trial.assert_not_awaited()


@pytest.mark.parametrize('step_level', [False, True])
def test_separate_dind_preflight_accepts_online_outer_override(tmp_path, step_level):
    config = make_config(
        tmp_path,
        '[environment]\ndocker_image = "example/task:v1"\n'
        + ('[[steps]]\nname = "first"\n[steps.verifier.environment]\n' if step_level else '[verifier.environment]\n')
        + 'network_mode = "no-network"\n',
        multi_step=step_level,
    )
    tests_dir = config.task.path / 'steps' / 'first' / 'tests' if step_level else config.task.path / 'tests'
    (tests_dir / 'Dockerfile').write_text('FROM ubuntu:24.04\n')
    provider = RecordingGongfengProvider(network_policy={'default_action': 'allow'})
    task = TaskConfig.model_validate_toml((config.task.path / 'task.toml').read_text())
    validate_separate_verifier_networks(task, config, provider)


def test_separate_preflight_ignores_agent_force_build_and_extra_compose(tmp_path):
    config = make_config(
        tmp_path,
        '[environment]\ndocker_image = "example/task:v1"\n'
        '[verifier.environment]\ndocker_image = "example/verifier:v1"\n'
        'network_mode = "allowlist"\nallowed_hosts = ["example.com"]\n',
    )
    (config.task.path / 'tests' / 'Dockerfile').write_text('FROM ubuntu:24.04\n')
    overlay = tmp_path / 'agent-extra.yaml'
    overlay.write_text('services:\n  main:\n    command: sleep infinity\n')
    config.environment.force_build = True
    config.environment.extra_docker_compose = [overlay]
    task = TaskConfig.model_validate_toml((config.task.path / 'task.toml').read_text())
    validate_separate_verifier_networks(task, config, RecordingGongfengProvider())
