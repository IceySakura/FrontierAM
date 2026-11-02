"""Static Harbor network policies for direct and DinD-backed environments."""

from __future__ import annotations

from typing import Any

from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import NetworkMode, NetworkPolicy, TaskConfig
from harbor.models.task.paths import TaskPaths
from harbor.models.task.verifier_mode import (
    resolve_effective_verifier_env_config,
    resolve_step_verifier_mode,
    resolve_task_verifier_mode,
)
from harbor.models.trial.config import TrialConfig
from harbor.trial.network_policy import resolve_trial_network_plan

from nexus.runtime.providers.base import RuntimeProvider
from nexus.runtime.providers.gongfeng import GongfengRuntimeProvider


def network_capabilities(provider: RuntimeProvider | None, *, compose_mode: bool = False) -> EnvironmentCapabilities:
    # The backend handles network-policy routing regardless of the scheduling name.
    supported = isinstance(provider, GongfengRuntimeProvider)
    allowlist = supported and not compose_mode
    return EnvironmentCapabilities(
        disable_internet=supported,
        network_allowlist=allowlist,
        network_allowlist_hostnames=allowlist,
        network_allowlist_wildcard_hostnames=allowlist,
        network_allowlist_ipv4_addresses=allowlist,
        network_allowlist_ipv4_cidrs=allowlist,
        dynamic_network_policy=False,
        mounted=False,
        docker_compose=True,
    )


def runtime_network_params(
    policy: NetworkPolicy,
    provider: RuntimeProvider | None,
    *,
    runtime_name: str | None = None,
    compose_mode: bool = False,
) -> dict[str, Any]:
    """Route Direct policy to the Sandbox; keep DinD's build Sandbox online."""
    if policy.network_mode != NetworkMode.PUBLIC:
        if not network_capabilities(provider, compose_mode=compose_mode).disable_internet:
            raise ValueError(
                'Harbor network isolation requires GongfengRuntimeProvider; other providers are unsupported'
            )
        if compose_mode and policy.network_mode == NetworkMode.ALLOWLIST:
            raise ValueError("network_mode='allowlist' is not supported by nexus DinD environments")
        if any(':' in host for host in policy.allowed_hosts):
            raise ValueError('Harbor allowed_hosts contains IPv6 targets unsupported by NexusEnvironment')

    if not isinstance(provider, GongfengRuntimeProvider):
        return {}

    network_policy = (
        None
        if compose_mode or policy.network_mode == NetworkMode.PUBLIC
        else {
            'default_action': 'deny',
            'egress': [{'action': 'allow', 'target': host} for host in policy.allowed_hosts],
        }
    )
    expected = network_policy or {'default_action': 'allow', 'egress': []}
    configured_policies = [('runtime_provider_config', provider.network_policy)]
    if runtime_name is not None:
        configured_policies.append((
            f'runtime_configs.{runtime_name}',
            provider.get_params(runtime_name).get('network_policy'),
        ))
    for source, configured in configured_policies:
        normalized = provider._validate_network_policy(configured)
        if normalized is not None and normalized != expected:
            requirement = (
                'Harbor DinD requires an online outer Sandbox for image pulls and builds'
                if compose_mode
                else f'Harbor network_mode={policy.network_mode.value!r}'
            )
            raise ValueError(
                f'{source}.network_policy conflicts with {requirement}; '
                'remove the override and configure network policy in the Harbor task'
            )
    return {'network_policy': network_policy}


def validate_separate_verifier_networks(
    task_config: TaskConfig,
    trial_config: TrialConfig,
    provider: RuntimeProvider,
) -> None:
    """Check separate baselines before Trial starts the Agent environment.

    Harbor already checks Shared Verifier and Agent step policies during Trial
    creation, but validates Separate Verifier policies only when verifying.
    """
    for step in task_config.steps or [None]:
        mode = (
            resolve_step_verifier_mode(task_config, step)
            if step is not None
            else resolve_task_verifier_mode(task_config)
        )
        plan = resolve_trial_network_plan(
            task_config,
            trial_config.agent,
            trial_config.environment,
            step,
            verifier_mode=mode,
        )
        if plan.verifier_env_baseline is None:
            continue
        label = f'Step {step.name!r} separate verifier' if step is not None else 'Separate verifier'
        if plan.verifier_phase != plan.verifier_env_baseline:
            raise ValueError(
                f'{label} network policy differs from its environment baseline; '
                'NexusEnvironment cannot change network policy after start. '
                'Configure the policy in [verifier.environment] or [steps.verifier.environment].'
            )
        env_config = resolve_effective_verifier_env_config(task_config, step)
        assert env_config is not None
        assert trial_config.task.path is not None
        paths = TaskPaths(trial_config.task.path)
        environment_dir = paths.tests_dir
        if step is not None and paths.step_tests_dir(step.name).exists():
            environment_dir = paths.step_tests_dir(step.name)
        # Native Separate Verifier always starts with force_build=False and no extra Compose files.
        compose_mode = (environment_dir / 'docker-compose.yaml').exists() or not env_config.docker_image
        runtime_network_params(plan.verifier_env_baseline, provider, compose_mode=compose_mode)
