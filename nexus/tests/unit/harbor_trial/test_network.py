from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from harbor.models.task.config import EnvironmentConfig, NetworkPolicy, TaskConfig
from harbor.models.task.verifier_mode import resolve_effective_verifier_env_config
from harbor.models.trial.config import AgentConfig, TrialConfig
from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
from harbor.models.trial.config import TaskConfig as TrialTaskConfig
from harbor.models.trial.paths import TrialPaths
from harbor.trial.trial import Trial
from tests.unit.harbor_trial.test_harbor_trial import FakeRuntime, FakeRuntimeProvider, make_task_dir

from nexus.core import ExecutionContext
from nexus.extensions.harbor_trial.context import HarborTrialBinding, harbor_trial_binding
from nexus.extensions.harbor_trial.environment import NexusEnvironment
from nexus.extensions.harbor_trial.network import (
    network_capabilities,
    runtime_network_params,
    validate_separate_verifier_networks,
)
from nexus.extensions.harbor_trial.task import HarborTrialTaskInput, harbor_trial_task
from nexus.runtime.providers.gongfeng import GongfengRuntimeProvider


PUBLIC = NetworkPolicy()
OFFLINE = NetworkPolicy(network_mode='no-network')
ALLOWLIST = NetworkPolicy(
    network_mode='allowlist',
    allowed_hosts=['example.com', '*.example.org', '192.0.2.1', '198.51.100.0/24'],
)
DENY_ALL = {'default_action': 'deny', 'egress': []}


class RecordingGongfengProvider(GongfengRuntimeProvider):
    def __init__(self, **kwargs):
        super().__init__(api_base_url='http://sandbox.invalid', token='unit-test-token', **kwargs)
        self.acquire_calls = []
        self.release_calls = []
        self.runtimes = {}

    async def acquire(self, name, **params):
        merged = self.get_params(name, **params)
        self.acquire_calls.append((name, merged))
        runtime = FakeRuntime()
        self.runtimes[name] = runtime
        self._active_runtimes[name] = runtime
        return runtime

    async def _release_impl(self, name):
        self.release_calls.append(name)
        self._active_runtimes.pop(name, None)


def make_config(root: Path, content: str, *, compose: bool = False, multi_step: bool = False) -> TrialConfig:
    task_dir = make_task_dir(root, compose=compose)
    if not multi_step:
        (task_dir / 'instruction.md').write_text('Produce a result.\n')
        (task_dir / 'tests').mkdir()
        (task_dir / 'tests' / 'test.sh').write_text('#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n')
    (task_dir / 'task.toml').write_text(content)
    return TrialConfig(
        task=TrialTaskConfig(path=task_dir),
        trial_name='network-test',
        trials_dir=root / 'trials',
        agent=AgentConfig(import_path='nexus.extensions.harbor_trial.agent:NexusAgent'),
        environment=TrialEnvironmentConfig(import_path='nexus.extensions.harbor_trial.environment:NexusEnvironment'),
    )


@pytest.mark.parametrize('pool', [None, '', 'gongfeng', 'suanli', 'taiji', 'suanli.pool.custom'])
def test_network_capabilities_do_not_restrict_scheduling_names(pool):
    capabilities = network_capabilities(RecordingGongfengProvider(runtime_provider=pool))
    assert capabilities.disable_internet
    assert capabilities.network_allowlist
    assert capabilities.network_allowlist_hostnames
    assert capabilities.network_allowlist_wildcard_hostnames
    assert capabilities.network_allowlist_ipv4_addresses
    assert capabilities.network_allowlist_ipv4_cidrs
    assert not capabilities.network_allowlist_ipv6_addresses
    assert not capabilities.network_allowlist_ipv6_cidrs
    assert not capabilities.dynamic_network_policy


@pytest.mark.parametrize('provider', [None, FakeRuntimeProvider()])
@pytest.mark.parametrize('policy', [OFFLINE, ALLOWLIST])
def test_unsupported_provider_rejects_isolation(provider, policy):
    assert not network_capabilities(provider).disable_internet
    with pytest.raises(ValueError, match='network isolation requires'):
        runtime_network_params(policy, provider)


@pytest.mark.parametrize('target', ['2001:db8::1', '2001:db8::/32'])
def test_ipv6_is_rejected(target):
    with pytest.raises(ValueError, match='IPv6'):
        runtime_network_params(
            NetworkPolicy(network_mode='allowlist', allowed_hosts=[target]), RecordingGongfengProvider()
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('mode', 'policy'),
    [(mode, policy) for mode in ['direct', 'compose', 'dockerfile-only', 'force-build'] for policy in [PUBLIC, OFFLINE]]
    + [('direct', ALLOWLIST)],
)
@pytest.mark.parametrize('pool', [None, '', 'gongfeng', 'suanli', 'taiji', 'suanli.pool.custom'])
async def test_environment_applies_policy_at_runtime_creation(tmp_path, mode, policy, pool):
    provider = RecordingGongfengProvider(runtime_provider=pool)
    context = ExecutionContext(runtime_provider=provider)
    task_dir = make_task_dir(tmp_path, compose=mode == 'compose')
    if mode in {'dockerfile-only', 'force-build'}:
        (task_dir / 'environment' / 'Dockerfile').write_text('FROM ubuntu:24.04\n')
    paths = TrialPaths(tmp_path / 'trial')
    paths.mkdir()
    binding = HarborTrialBinding(execution_context=context, run_agent=AsyncMock())
    async with harbor_trial_binding(binding):
        environment = NexusEnvironment(
            environment_dir=task_dir / 'environment',
            environment_name='test',
            session_id='trial-env',
            trial_paths=paths,
            task_env_config=EnvironmentConfig(docker_image=None if mode == 'dockerfile-only' else 'example/task:v1'),
            network_policy=policy,
        )
        await environment.start(force_build=mode == 'force-build')
        await environment.start(force_build=mode == 'force-build')
        await environment.set_network_policy(policy.model_copy(deep=True))
        with pytest.raises(ValueError, match='cannot change network policy'):
            await environment.set_network_policy(OFFLINE if policy == PUBLIC else PUBLIC)
        await environment.stop(delete=True)

    assert len(provider.acquire_calls) == 1
    name, params = provider.acquire_calls[0]
    assert name == 'main'
    assert params['enable_dind'] == (mode != 'direct')
    expected = (
        None
        if mode != 'direct' or policy == PUBLIC
        else {
            'default_action': 'deny',
            'egress': [{'action': 'allow', 'target': target} for target in policy.allowed_hosts],
        }
    )
    assert params['network_policy'] == expected
    assert provider.release_calls == ['main']
    if mode != 'direct':
        runtime = provider.runtimes['main']
        assert any(' up -d' in command for command in runtime.commands)
        target = environment.agent_execution_target()
        assert target is not None
        network_template = '/harbor/compose/docker-compose-no-network.yaml'
        assert (network_template in target.compose_files) == (policy == OFFLINE)
        if policy == OFFLINE:
            assert runtime.uploaded_files[network_template] == 'services:\n  main:\n    network_mode: none\n'
            assert target.compose_files[-1] == network_template
            for command in runtime.commands:
                if 'docker compose' in command:
                    assert network_template in command


@pytest.mark.parametrize('source', ['provider', 'runtime'])
@pytest.mark.parametrize(('policy', 'override'), [(PUBLIC, DENY_ALL), (OFFLINE, {'default_action': 'allow'})])
def test_conflicting_defaults_are_not_silently_overridden(source, policy, override):
    provider = RecordingGongfengProvider(network_policy=override if source == 'provider' else None)
    if source == 'runtime':
        provider.configure('main', network_policy=override)
    original_default = deepcopy(provider.network_policy)
    original_params = deepcopy(provider.get_params('main'))
    with pytest.raises(ValueError, match='conflicts with Harbor'):
        runtime_network_params(policy, provider, runtime_name='main')
    assert provider.network_policy == original_default
    assert provider.get_params('main') == original_params


def test_matching_policy_and_empty_overrides_preserve_task_isolation():
    provider = RecordingGongfengProvider(network_policy=DENY_ALL)
    provider.configure('main', network_policy={})
    assert runtime_network_params(OFFLINE, provider, runtime_name='main') == {'network_policy': DENY_ALL}
    provider.configure('main', network_policy=DENY_ALL)
    assert runtime_network_params(OFFLINE, provider, runtime_name='main') == {'network_policy': DENY_ALL}


@pytest.mark.asyncio
async def test_conflicting_runtime_config_fails_before_acquire(tmp_path):
    provider = RecordingGongfengProvider()
    provider.configure('main', network_policy={'default_action': 'allow'})
    context = ExecutionContext(runtime_provider=provider)
    task_dir = make_task_dir(tmp_path)
    async with harbor_trial_binding(HarborTrialBinding(execution_context=context, run_agent=AsyncMock())):
        environment = NexusEnvironment(
            environment_dir=task_dir / 'environment',
            environment_name='test',
            session_id='trial-env',
            trial_paths=TrialPaths(tmp_path / 'trial'),
            task_env_config=EnvironmentConfig(docker_image='example/task:v1'),
            network_policy=OFFLINE,
        )
        with pytest.raises(ValueError, match='runtime_configs.main.network_policy conflicts'):
            await environment.start(force_build=False)
    assert provider.acquire_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('provider', 'policy'),
    [
        (FakeRuntimeProvider(), OFFLINE),
        (RecordingGongfengProvider(), NetworkPolicy(network_mode='allowlist', allowed_hosts=['2001:db8::1'])),
    ],
)
async def test_environment_rejects_unsupported_network_before_acquire(tmp_path, provider, policy):
    task_dir = make_task_dir(tmp_path)
    context = ExecutionContext(runtime_provider=provider)
    async with harbor_trial_binding(HarborTrialBinding(execution_context=context, run_agent=AsyncMock())):
        with pytest.raises(ValueError, match='not supported by nexus'):
            NexusEnvironment(
                environment_dir=task_dir / 'environment',
                environment_name='test',
                session_id='trial-env',
                trial_paths=TrialPaths(tmp_path / 'trial'),
                task_env_config=EnvironmentConfig(docker_image='example/task:v1'),
                network_policy=policy,
            )
    assert provider.acquire_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'phase',
    [
        '[agent]\nnetwork_mode = "no-network"\n',
        '[verifier]\nenvironment_mode = "shared"\nnetwork_mode = "no-network"\n',
        '[[steps]]\nname = "first"\n[[steps]]\nname = "second"\n[steps.agent]\nnetwork_mode = "no-network"\n',
        '[[steps]]\nname = "first"\n[[steps]]\nname = "second"\n[steps.verifier]\nnetwork_mode = "no-network"\n',
    ],
)
async def test_native_trial_rejects_shared_phase_switch_before_start(tmp_path, phase):
    config = make_config(
        tmp_path, '[environment]\ndocker_image = "example/task:v1"\n' + phase, multi_step='[[steps]]' in phase
    )
    provider = RecordingGongfengProvider()
    binding = HarborTrialBinding(execution_context=ExecutionContext(runtime_provider=provider), run_agent=AsyncMock())
    async with harbor_trial_binding(binding):
        with pytest.raises(ValueError, match='cannot change network policy'):
            await Trial.create(config)
    assert provider.acquire_calls == []


@pytest.mark.asyncio
async def test_native_trial_rejects_shared_allowlist_changes(tmp_path):
    config = make_config(
        tmp_path,
        '''[environment]
docker_image = "example/task:v1"
network_mode = "allowlist"
allowed_hosts = ["example.com"]
[verifier]
network_mode = "allowlist"
allowed_hosts = ["example.org"]
''',
    )
    provider = RecordingGongfengProvider()
    binding = HarborTrialBinding(execution_context=ExecutionContext(runtime_provider=provider), run_agent=AsyncMock())
    async with harbor_trial_binding(binding):
        with pytest.raises(ValueError, match='cannot change network policy'):
            await Trial.create(config)
    assert provider.acquire_calls == []


@pytest.mark.asyncio
async def test_native_shared_multi_step_direct_inherits_static_allowlist(tmp_path):
    config = make_config(
        tmp_path,
        '''[environment]
docker_image = "example/task:v1"
network_mode = "allowlist"
allowed_hosts = ["EXAMPLE.COM.", "*.example.org", "192.0.2.1", "198.51.100.0/24"]
[[steps]]
name = "first"
[[steps]]
name = "second"
''',
        multi_step=True,
    )
    provider = RecordingGongfengProvider()
    binding = HarborTrialBinding(execution_context=ExecutionContext(runtime_provider=provider), run_agent=AsyncMock())
    async with harbor_trial_binding(binding):
        trial = await Trial.create(config)
        try:
            assert trial.agent_environment.network_policy == ALLOWLIST
            await trial.agent_environment.start(force_build=False)
            await trial.agent_environment.stop(delete=True)
        finally:
            trial._close_logger_handler()
    assert provider.acquire_calls[0][1]['network_policy']['egress'] == [
        {'action': 'allow', 'target': target} for target in ALLOWLIST.allowed_hosts
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('compose', [False, True], ids=['direct-agent', 'compose-agent'])
@pytest.mark.parametrize('verifier_compose', [False, True], ids=['direct-verifier', 'compose-verifier'])
@pytest.mark.parametrize('pool', [None, 'gongfeng'])
async def test_native_separate_verifier_uses_its_own_policy(tmp_path, compose, verifier_compose, pool):
    config = make_config(
        tmp_path,
        '''[environment]
docker_image = "example/agent:v1"
network_mode = "public"
[verifier.environment]
docker_image = "example/verifier:v1"
network_mode = "no-network"
''',
        compose=compose,
    )
    if verifier_compose:
        (config.task.path / 'tests' / 'docker-compose.yaml').write_text('services:\n  main:\n    command: sleep infinity\n')
    provider = RecordingGongfengProvider(runtime_provider=pool)
    binding = HarborTrialBinding(execution_context=ExecutionContext(runtime_provider=provider), run_agent=AsyncMock())
    async with harbor_trial_binding(binding):
        trial = await Trial.create(config)
        try:
            validate_separate_verifier_networks(trial.task.config, config, provider)
            await trial.agent_environment.start(force_build=False)
            await trial.agent_environment.stop(delete=True)
            env_config = resolve_effective_verifier_env_config(trial.task.config, None)
            plan = trial._network_plan(None)
            async with trial._separate_verifier_env(env_config, key='task', plan=plan) as environment:
                assert environment.network_policy == OFFLINE
        finally:
            trial._close_logger_handler()
    assert len(provider.acquire_calls) == 2
    (agent_name, agent_params), (verifier_name, verifier_params) = provider.acquire_calls
    assert agent_name == 'main'
    assert verifier_name != agent_name
    assert agent_params['network_policy'] is None
    assert verifier_params['network_policy'] == (None if verifier_compose else DENY_ALL)
    assert verifier_params['enable_dind'] == verifier_compose
    if not verifier_compose:
        assert verifier_params['image'] == 'example/verifier:v1'
    assert provider.release_calls == [agent_name, verifier_name]


@pytest.mark.parametrize(
    'content',
    [
        '''[environment]
docker_image = "example/task:v1"
[verifier]
environment_mode = "separate"
network_mode = "no-network"
''',
        '''[environment]
docker_image = "example/task:v1"
[[steps]]
name = "first"
[[steps]]
name = "second"
[steps.verifier]
network_mode = "no-network"
[steps.verifier.environment]
docker_image = "example/verifier:v1"
''',
    ],
)
@pytest.mark.asyncio
async def test_task_preflights_separate_phase_switch_before_trial_run(tmp_path, monkeypatch, content):
    config = make_config(tmp_path, content, multi_step='[[steps]]' in content)
    provider = RecordingGongfengProvider()
    run_agent = AsyncMock(nexus_meta={})
    run_trial = AsyncMock()
    monkeypatch.setattr(Trial, 'run', run_trial)
    async with ExecutionContext(runtime_provider=provider):
        with pytest.raises(ValueError, match='cannot change network policy'):
            await harbor_trial_task(HarborTrialTaskInput(task_dir=str(config.task.path), run_agent=run_agent))
    assert provider.acquire_calls == []
    run_agent.assert_not_awaited()
    run_trial.assert_not_awaited()


@pytest.mark.parametrize(
    ('provider', 'verifier_network', 'match'),
    [
        (FakeRuntimeProvider(), 'network_mode = "no-network"', 'network isolation requires'),
        (RecordingGongfengProvider(), 'network_mode = "allowlist"\nallowed_hosts = ["2001:db8::1"]', 'IPv6'),
        (RecordingGongfengProvider(network_policy=DENY_ALL), 'network_mode = "public"', 'conflicts with Harbor'),
    ],
)
def test_separate_preflight_rejects_unsupported_policy(tmp_path, provider, verifier_network, match):
    config = make_config(
        tmp_path,
        '[environment]\ndocker_image = "example/task:v1"\n'
        '[verifier.environment]\ndocker_image = "example/verifier:v1"\n' + verifier_network,
    )
    task = TaskConfig.model_validate_toml((config.task.path / 'task.toml').read_text())
    with pytest.raises(ValueError, match=match):
        validate_separate_verifier_networks(task, config, provider)
    assert provider.acquire_calls == []


def test_separate_multi_step_may_use_different_static_baselines(tmp_path):
    config = make_config(
        tmp_path,
        '''[environment]
docker_image = "example/task:v1"
network_mode = "public"
[verifier.environment]
docker_image = "example/verifier:v1"
network_mode = "no-network"
[[steps]]
name = "first"
[[steps]]
name = "second"
[steps.verifier.environment]
docker_image = "example/verifier:v2"
network_mode = "allowlist"
allowed_hosts = ["example.com"]
''',
        multi_step=True,
    )
    task = TaskConfig.model_validate_toml((config.task.path / 'task.toml').read_text())
    validate_separate_verifier_networks(task, config, RecordingGongfengProvider())
