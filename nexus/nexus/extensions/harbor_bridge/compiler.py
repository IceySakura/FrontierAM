"""Compile one Harbor task into a task-intrinsic Nexus Dataset Item body.

This module owns the task-to-runtime translation shared by the modern
``t3-harbor:///`` Dataset Source and the legacy environment-driven generator.
Agent, LLM, credentials, campaign policy, and Batch orchestration deliberately
remain outside this layer.
"""

from __future__ import annotations

import logging
import math
import os
import re
import tomllib

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from nexus.extensions.harbor_bridge.agent_trajectory import trajectory_processor_config
from nexus.extensions.harbor_bridge.browser import browser_agent_kwargs
from nexus.extensions.harbor_bridge.compose import compile_compose_plan
from nexus.extensions.harbor_bridge.mcp import mcp_agent_kwargs


logger = logging.getLogger(__name__)

DEFAULT_GONGFENG_CPU = 2
DEFAULT_GONGFENG_MEMORY_GB = 10
DEFAULT_RUNTIME_PROVIDER = 'gongfeng'
DEFAULT_TIMEOUT_MULTIPLIER = 1.0

_ENV_TEMPLATE_RE = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}')
_WARNED_STORAGE_REQUESTS: set[tuple[str, int]] = set()

# Providers whose sandbox API expresses an accelerator request as one
# ``resource_spec.gpu_card_type`` value (sandbox v3 CreateServing ``gpuType``).
_GPU_CARD_TYPE_PROVIDERS: frozenset[str] = frozenset({'gongfeng', 'suanli', 'taiji'})


@dataclass(frozen=True)
class LoadedHarborTask:
    """Raw T3 extensions paired with Harbor's validated Task model."""

    config: dict[str, Any]
    task: Any


def _load_raw_task_config(task_dir: Path) -> dict[str, Any]:
    with (task_dir / 'task.toml').open('rb') as config_file:
        config = tomllib.load(config_file)
    if config.get('steps'):
        raise ValueError('Harbor multi-step tasks are not supported by harbor_bridge')
    return config


def load_task_definition(task_dir: Path) -> tuple[dict[str, Any], str]:
    """Compatibility loader for callers that only need raw files."""
    config = _load_raw_task_config(task_dir)

    instruction = (task_dir / 'instruction.md').read_text()
    if not instruction.strip():
        raise ValueError('instruction.md is empty')
    return config, instruction


def load_harbor_task(task_dir: Path) -> LoadedHarborTask:
    """Read raw T3 extensions and construct Harbor's Task exactly once."""
    from harbor.models.task.task import Task

    config = _load_raw_task_config(task_dir)
    task = Task(task_dir)
    if not task.instruction.strip():
        raise ValueError('instruction.md is empty')
    return LoadedHarborTask(config=config, task=task)


def resolve_env_value(value: Any) -> str:
    """Resolve every ``${VAR}`` / ``${VAR:-default}`` template in one value."""
    raw = str(value)

    def replace(match: re.Match[str]) -> str:
        name, default = match.groups()
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(f"Environment variable '{name}' not found in host environment")

    return _ENV_TEMPLATE_RE.sub(replace, raw)


def resolve_env_mapping(env: dict[str, Any]) -> dict[str, str]:
    """Resolve task environment values with one shared template implementation."""
    return {key: resolve_env_value(value) for key, value in env.items()}


def _get_task_enable_dind(config: dict[str, Any]) -> bool:
    """Return the task's explicit DinD requirement."""
    environment = config.get('environment', {}) or {}
    value = environment.get('enable_dind', False)
    if not isinstance(value, bool):
        raise ValueError(f'environment.enable_dind must be a boolean, got {value!r}')
    return value


def _build_boot_config(
    config: dict,
    *,
    command_override: str = '',
) -> dict[str, Any]:
    """Compute the optional image-boot action for the serialized task input."""
    boot_section = config.get('environment', {}).get('boot', {}) or {}
    boot_cmd = command_override.strip() or str(boot_section.get('command', '')).strip()
    if not boot_cmd:
        return {}

    boot_env = resolve_env_mapping(config.get('environment', {}).get('env', {}) or {})
    return {
        'boot_command': boot_cmd,
        'boot_env': boot_env,
    }


def _resolve_runtime_provider_for_network(
    configured_provider: str | None,
    network_policy: dict[str, Any] | None,
) -> str:
    """Use suanli for network policies and reject conflicting explicit overrides."""
    if network_policy is None:
        return configured_provider or DEFAULT_RUNTIME_PROVIDER

    provider = (configured_provider or '').strip().lower()
    # Taiji pins ``runtime_provider=taiji`` and carries the same
    # ``network_policy`` field into the CreateServing sandbox config, so it
    # cannot be routed through the suanli scheduling path.
    if provider == 'taiji':
        return 'taiji'
    if provider not in ('', 'suanli'):
        raise ValueError(
            'Harbor network isolation requires GONGFENG_RUNTIME_PROVIDER=suanli because '
            'Gongfeng network_policy is enforced only on the suanli scheduling path.'
        )
    return 'suanli'


def _build_static_network_policy(
    environment_config: Any,
    phase_config: Any,
    *,
    role: str,
    offline_allowlist: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """Resolve one Harbor environment/phase policy into a static Gongfeng policy."""
    baseline = environment_config.resolve_baseline()
    phase = phase_config.explicit_phase_policy()
    if phase is not None and phase != baseline:
        raise ValueError(
            f'{role} network policy changes from {baseline.network_mode.value!r} at startup '
            f'to {phase.network_mode.value!r} during execution, but Gongfeng cannot change '
            'network_policy after sandbox creation.'
        )

    effective = phase or baseline
    mode = effective.network_mode.value
    if mode == 'public':
        return None

    harbor_targets = list(effective.allowed_hosts)
    unsupported = [target for target in harbor_targets if ':' in target]
    if unsupported:
        raise ValueError(f'{role} allowed_hosts contains IPv6 targets unsupported by Gongfeng: {unsupported}')

    targets = list(dict.fromkeys([*harbor_targets, *offline_allowlist]))
    return {
        'default_action': 'deny',
        'egress': [{'action': 'allow', 'target': target} for target in targets],
    }


def _build_resource_spec(
    environment_config: Any,
    *,
    fallback_cpu: int,
    fallback_memory_gb: int,
    role: str,
    runtime_provider: str = DEFAULT_RUNTIME_PROVIDER,
) -> dict[str, Any]:
    """Map Harbor resources to the selected provider's cpu/memory/gpu resource spec."""
    if environment_config.os.value != 'linux':
        raise ValueError(f'{role} environment.os={environment_config.os.value!r} is not supported by Gongfeng')
    if environment_config.tpu is not None:
        raise ValueError(f'{role} TPU requirements are not supported by Gongfeng')
    gpu_types = [str(card_type).strip() for card_type in (environment_config.gpu_types or [])]
    if any(not card_type for card_type in gpu_types):
        raise ValueError(f'{role} gpu_types must contain non-empty card type names')
    if gpu_types and runtime_provider not in _GPU_CARD_TYPE_PROVIDERS:
        raise ValueError(f'{role} gpu_types cannot be guaranteed by {runtime_provider}: {gpu_types}')
    if len(gpu_types) > 1:
        raise ValueError(
            f'{role} gpu_types declares multiple card types {gpu_types}; '
            f'{runtime_provider} accepts exactly one card type per runtime'
        )
    if environment_config.storage_mb is not None:
        warning_key = (role, int(environment_config.storage_mb))
        if warning_key not in _WARNED_STORAGE_REQUESTS:
            _WARNED_STORAGE_REQUESTS.add(warning_key)
            logger.warning(
                '%s requests storage_mb=%s, but Gongfeng has no configurable storage quota; '
                'the value cannot be enforced. Further identical warnings are suppressed.',
                role,
                environment_config.storage_mb,
            )

    gpus = environment_config.gpus or 0
    if gpus < 0:
        raise ValueError(f'{role} gpus must be non-negative for Gongfeng, got {gpus}')

    cpu = environment_config.cpus if environment_config.cpus is not None else fallback_cpu
    memory_mb = environment_config.memory_mb if environment_config.memory_mb is not None else fallback_memory_gb * 1024
    if cpu <= 0 or cpu > 32:
        raise ValueError(f'{role} cpus must be between 1 and 32 for Gongfeng, got {cpu}')
    if memory_mb <= 0:
        raise ValueError(f'{role} memory must be positive for Gongfeng, got {memory_mb} MB')
    memory_gb = max(math.ceil(memory_mb / 1024), 2)
    max_memory_gb = 123 if gpus else 64
    if memory_gb > max_memory_gb:
        resource_kind = 'GPU sandbox' if gpus else 'CPU sandbox'
        raise ValueError(f'{role} memory must be at most {max_memory_gb} GB for {resource_kind}, got {memory_mb} MB')

    spec: dict[str, Any] = {'cpu': int(cpu), 'memory': int(memory_gb)}
    if gpus:
        spec['gpu'] = int(gpus)
        if gpu_types:
            spec['gpu_card_type'] = gpu_types[0]
    elif gpu_types:
        raise ValueError(f'{role} declares gpu_types {gpu_types} without requesting gpus')
    return spec


def _resolve_harbor_env(environment_config: Any) -> dict[str, str]:
    """Resolve Harbor environment values with composite-template support."""
    return resolve_env_mapping(environment_config.env or {})


def _validate_service_references(task_config: Any, *, compose_enabled: bool) -> None:
    """Reject sidecar references when no Compose topology can provide them."""
    from harbor.constants import MAIN_SERVICE_NAME

    if compose_enabled:
        return
    for artifact in task_config.artifacts:
        service = getattr(artifact, 'service', None) if not isinstance(artifact, str) else None
        if service not in (None, MAIN_SERVICE_NAME):
            raise ValueError(
                f'Harbor Bridge artifact service={service!r} is not supported for non-Compose tasks; '
                'use service="main" or native Compose.'
            )

    for hook in task_config.verifier.collect:
        if hook.service != MAIN_SERVICE_NAME:
            raise ValueError(
                f'Harbor Bridge collect service={hook.service!r} is not supported for non-Compose tasks; '
                'use service="main" or native Compose.'
            )


def _validate_separate_verifier_features(task_config: Any, task_dir: Path) -> None:
    """Reject separate-verifier features that Gongfeng Bridge cannot preserve."""

    for hook in task_config.verifier.collect:
        if isinstance(hook.user, int):
            raise ValueError('Separate verifier collect numeric UID is not supported by Gongfeng Bridge')

    if isinstance(task_config.verifier.user, int):
        raise ValueError('Separate verifier numeric UID is not supported by Gongfeng Bridge')

    verifier_environment = task_config.verifier.environment
    if verifier_environment is not None and not verifier_environment.docker_image:
        dockerfile = task_dir / 'tests' / 'Dockerfile'
        detail = f' ({dockerfile})' if dockerfile.exists() else ''
        raise ValueError(
            'Separate verifier requires a prebuilt [verifier.environment].docker_image; '
            f'building the tests Dockerfile is not supported{detail}.'
        )


def _get_docker_image(config: dict, task_dir: Path | None = None) -> str:
    """Require the task to name a complete prebuilt image."""
    raw_docker_image = (config.get('environment', {}) or {}).get('docker_image')
    docker_image = raw_docker_image.strip() if isinstance(raw_docker_image, str) else ''
    if docker_image:
        return docker_image

    location = f' for task {task_dir}' if task_dir is not None else ''
    raise ValueError(
        f'Harbor Bridge requires an explicit [environment].docker_image{location}. '
        'Dockerfile and Docker Compose builds are not supported; prebuild and publish '
        'the complete task image, then set environment.docker_image.'
    )


def _get_workdir(config: dict, task_dir: Path | None = None) -> str:
    """Require one explicit absolute container workdir."""
    raw_workdir = (config.get('environment', {}) or {}).get('workdir')
    workdir = raw_workdir.strip() if isinstance(raw_workdir, str) else ''
    location = f' for task {task_dir}' if task_dir is not None else ''
    if not workdir:
        raise ValueError(f'T3-Harbor requires an explicit [environment].workdir{location}')
    if not workdir.startswith('/'):
        raise ValueError(f'T3-Harbor environment.workdir must be an absolute container path{location}: {workdir!r}')
    return workdir


def _get_metadata_instance_id(config: dict[str, Any]) -> str | None:
    """Read the canonical task ID from ``[metadata].instance_id`` when present."""
    metadata = config.get('metadata', {}) or {}
    if not isinstance(metadata, dict):
        raise ValueError('[metadata] must be a table')
    raw_instance_id = metadata.get('instance_id')
    if raw_instance_id is None:
        return None
    if not isinstance(raw_instance_id, str) or not raw_instance_id.strip():
        raise ValueError('metadata.instance_id must be a non-empty string')
    return raw_instance_id.strip()


def _require_bool(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f'{location} must be a boolean, got {value!r}')
    return value


def _scaled_positive_timeout(
    config: dict[str, Any],
    section: str,
    *,
    default: float,
    multiplier: float,
) -> float:
    raw_section = config.get(section, {}) or {}
    if not isinstance(raw_section, dict):
        raise ValueError(f'[{section}] must be a table')
    raw_timeout = raw_section.get('timeout_sec', default)
    if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, (int, float)):
        raise ValueError(f'{section}.timeout_sec must be a number, got {raw_timeout!r}')
    timeout = float(raw_timeout) * multiplier
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f'{section}.timeout_sec must produce a positive finite timeout, got {timeout!r}')
    return timeout


def _get_skip_environment_upload(config: dict) -> bool:
    """Return whether runtime should skip the default workspace overlay."""
    env_config = config.get('environment', {}) or {}
    if 'skip_upload' in env_config:
        return _require_bool(env_config.get('skip_upload'), 'environment.skip_upload')
    return True


def _get_environment_uploads(config: dict) -> list[dict[str, str]]:
    """Validate and normalize explicit ``[environment.upload]`` items."""
    environment = config.get('environment', {}) or {}
    if not isinstance(environment, dict):
        raise ValueError('[environment] must be a table')

    upload_config = environment.get('upload')
    if upload_config is None:
        return []
    if not isinstance(upload_config, dict):
        raise ValueError('environment.upload must be a table')

    has_source = 'source' in upload_config
    has_items = 'items' in upload_config
    if has_source == has_items:
        raise ValueError('environment.upload must define exactly one of source or items')

    if has_source:
        raw_items: list[Any] = [upload_config]
    else:
        items = upload_config.get('items')
        if not isinstance(items, list) or not items:
            raise ValueError('environment.upload.items must be a non-empty array')
        raw_items = items

    normalized: list[dict[str, str]] = []
    for index, item in enumerate(raw_items):
        location = 'environment.upload' if has_source else f'environment.upload.items[{index}]'
        if not isinstance(item, dict):
            raise ValueError(f'{location} must be a table')

        raw_source = item.get('source')
        if not isinstance(raw_source, str) or not raw_source.strip():
            raise ValueError(f'{location}.source must be a non-empty string')
        source = raw_source.strip()
        source_path = PurePosixPath(source)
        if source_path.is_absolute() or '..' in source_path.parts:
            raise ValueError(f'{location}.source must be a safe task-relative path: {source!r}')

        raw_target = item.get('target')
        if raw_target is None:
            target = ''
        elif not isinstance(raw_target, str) or not raw_target.strip():
            raise ValueError(f'{location}.target must be a non-empty absolute container path when provided')
        else:
            target = raw_target.strip()
            if not PurePosixPath(target).is_absolute():
                raise ValueError(f'{location}.target must be an absolute container path: {target!r}')

        normalized.append({'source': source, 'target': target})
    return normalized


@dataclass(frozen=True)
class CompiledHarborTask:
    """Task-intrinsic partial RunInput compiled from one Harbor task."""

    task_name: str
    task_config: dict[str, Any]
    runtime_configs: dict[str, dict[str, Any]]
    runtime_provider: str
    metadata: dict[str, Any]

    def dataset_body(self) -> dict[str, Any]:
        """Return the partial RunInput owned by a Dataset Item."""
        body: dict[str, Any] = {
            'task_config': dict(self.task_config),
            'runtime_configs': self.runtime_configs,
        }
        if self.task_config.get('needs_agent_trajectory'):
            body['processors'] = trajectory_processor_config()
        if self.runtime_provider != DEFAULT_RUNTIME_PROVIDER:
            body['runtime_provider_config'] = {'runtime_provider': self.runtime_provider}
        return body


def compile_harbor_task(
    task_dir: Path,
    config: dict[str, Any],
    harbor_task: Any,
    *,
    fallback_cpu: int = DEFAULT_GONGFENG_CPU,
    fallback_memory_gb: int = DEFAULT_GONGFENG_MEMORY_GB,
    timeout_multiplier: float = DEFAULT_TIMEOUT_MULTIPLIER,
    configured_runtime_provider: str | None = None,
    enable_dind_override: bool | None = None,
    offline_allowlist: tuple[str, ...] = (),
    boot_command_override: str = '',
    verifier_boot_command_override: str = '',
) -> CompiledHarborTask:
    """Compile task-specific Harbor settings without Agent, LLM, credentials, or run mode."""
    from harbor.models.task.verifier_mode import (
        VerifierEnvironmentMode,
        resolve_effective_verifier_env_config,
        resolve_task_verifier_mode,
    )

    task_config_model = harbor_task.config
    instruction = harbor_task.instruction
    verifier_mode = resolve_task_verifier_mode(task_config_model)
    browser_kwargs = browser_agent_kwargs(config)
    if browser_kwargs and verifier_mode != VerifierEnvironmentMode.SHARED:
        raise ValueError(
            'T3-Harbor [environment.browser] requires verifier.environment_mode="shared"; '
            'a separate verifier cannot access the main runtime browser session'
        )
    docker_image = _get_docker_image(config, task_dir)
    workdir = _get_workdir(config, task_dir)
    # Accelerator declarations are provider-specific: the Gongfeng family pins one
    # sandbox-v3 card type, other bridges keep rejecting what they cannot enforce.
    requested_provider = (configured_runtime_provider or DEFAULT_RUNTIME_PROVIDER).strip().lower()
    resolved_environment = _resolve_harbor_env(task_config_model.environment)
    compose_plan = compile_compose_plan(
        task_dir,
        config,
        harbor_task,
        controller_image=docker_image,
        workdir=workdir,
        environment=resolved_environment,
        verifier_mode=verifier_mode.value,
    )
    enable_dind = enable_dind_override if enable_dind_override is not None else _get_task_enable_dind(config)
    if compose_plan is not None and not enable_dind:
        raise ValueError('Native Compose requires effective environment.enable_dind = true')
    _validate_service_references(task_config_model, compose_enabled=compose_plan is not None)
    main_network_policy = _build_static_network_policy(
        task_config_model.environment,
        task_config_model.agent,
        role='agent',
        offline_allowlist=offline_allowlist,
    )
    if verifier_mode == VerifierEnvironmentMode.SHARED:
        shared_verifier_policy = _build_static_network_policy(
            task_config_model.environment,
            task_config_model.verifier,
            role='shared verifier',
            offline_allowlist=offline_allowlist,
        )
        if shared_verifier_policy != main_network_policy:
            raise ValueError(
                'Shared Agent and Verifier resolved to different static network policies, '
                'but Gongfeng cannot change network_policy after sandbox creation.'
            )

    main_resource_spec = _build_resource_spec(
        task_config_model.environment,
        fallback_cpu=fallback_cpu,
        fallback_memory_gb=fallback_memory_gb,
        role='agent',
        runtime_provider=requested_provider,
    )
    main_boot_config = _build_boot_config(config, command_override=boot_command_override)
    main_runtime_config: dict[str, Any] = {
        'image': docker_image,
        'resource_spec': main_resource_spec,
        'environment': {} if compose_plan is not None else resolved_environment,
        'enable_dind': enable_dind,
        **({'network_policy': main_network_policy} if main_network_policy is not None else {}),
    }
    main_runtime_config['working_dir'] = '/nexus/control' if compose_plan is not None else workdir
    if compose_plan is not None:
        # Native Compose Agents execute through runtime-side helpers; ship HyCode too.
        main_runtime_config['worker_venv'] = {'modules': ['claude_code', 'codex', 'hycode']}

    verifier_network_policy: dict[str, Any] | None = None
    verifier_runtime_config: dict[str, Any] | None = None
    verifier_config = config.get('verifier', {}) or {}
    if not isinstance(verifier_config, dict):
        raise ValueError('[verifier] must be a table')
    skip_tests_upload = _require_bool(
        verifier_config.get('skip_tests_upload', False),
        'verifier.skip_tests_upload',
    )
    # Off unless the task says it grades process rather than deliverables. Most rubrics
    # read the workspace, and materializing a trajectory they ignore costs an upload per
    # run and puts the Agent's transcript in front of a judge that has no use for it.
    needs_agent_trajectory = _require_bool(
        verifier_config.get('needs_agent_trajectory', False),
        'verifier.needs_agent_trajectory',
    )
    verifier_boot_config: dict[str, Any] = {}
    if verifier_mode == VerifierEnvironmentMode.SEPARATE:
        _validate_separate_verifier_features(task_config_model, task_dir)
        verifier_environment = resolve_effective_verifier_env_config(task_config_model, step_cfg=None)
        if verifier_environment is None:
            raise RuntimeError('Separate verifier did not resolve an effective environment')

        verifier_network_policy = _build_static_network_policy(
            verifier_environment,
            task_config_model.verifier,
            role='verifier',
            offline_allowlist=offline_allowlist,
        )
        verifier_image = verifier_environment.docker_image or (
            docker_image if task_config_model.verifier.environment is None else ''
        )
        if not verifier_image:
            raise ValueError(f'Separate verifier requires a prebuilt docker_image for task {task_dir}')
        verifier_workdir = workdir if task_config_model.verifier.environment is None else verifier_environment.workdir
        verifier_runtime_config = {
            'image': verifier_image,
            'resource_spec': _build_resource_spec(
                verifier_environment,
                fallback_cpu=fallback_cpu,
                fallback_memory_gb=fallback_memory_gb,
                role='verifier',
                runtime_provider=requested_provider,
            ),
            'environment': _resolve_harbor_env(verifier_environment),
            'enable_dind': False,
            **({'network_policy': verifier_network_policy} if verifier_network_policy is not None else {}),
        }
        if verifier_workdir:
            verifier_runtime_config['working_dir'] = verifier_workdir
        skip_tests_upload = (
            skip_tests_upload or harbor_task.paths.discovered_test_path_for(verifier_environment.os) is None
        )
        raw_verifier_environment = (
            config
            if task_config_model.verifier.environment is None
            else {'environment': config.get('verifier', {}).get('environment', {}) or {}}
        )
        verifier_boot_config = {
            f'verifier_{key}': value
            for key, value in _build_boot_config(
                raw_verifier_environment,
                command_override=verifier_boot_command_override,
            ).items()
        }
        if task_config_model.verifier.environment is not None and not verifier_boot_config:
            logger.warning(
                'Separate verifier image %s will run without its Docker ENTRYPOINT because '
                'Gongfeng replaces it with nexus-runtime. If the image needs startup '
                'initialization, declare [verifier.environment.boot].command.',
                verifier_image,
            )

    runtime_provider = _resolve_runtime_provider_for_network(
        configured_runtime_provider,
        verifier_network_policy or main_network_policy,
    )
    runtime_configs: dict[str, dict[str, Any]] = {'main': main_runtime_config}
    if verifier_runtime_config is not None:
        runtime_configs['verifier'] = verifier_runtime_config

    agent_timeout = _scaled_positive_timeout(config, 'agent', default=900.0, multiplier=timeout_multiplier)
    test_timeout = _scaled_positive_timeout(config, 'verifier', default=600.0, multiplier=timeout_multiplier)

    task_name = task_dir.name
    metadata_instance_id = _get_metadata_instance_id(config)
    task_config = {
        'instance_id': task_name,
        'task_dir': str(task_dir),
        'instruction': instruction,
        'mcp_servers': mcp_agent_kwargs(config).get('mcp_servers', []),
        **browser_kwargs,
        **({'compose_plan': compose_plan.model_dump(mode='json')} if compose_plan is not None else {}),
        'workdir': workdir,
        'agent_timeout': agent_timeout,
        'test_timeout': test_timeout,
        'verifier_mode': verifier_mode.value,
        'skip_tests_upload': skip_tests_upload,
        'needs_agent_trajectory': needs_agent_trajectory,
        'skip_environment_upload': _get_skip_environment_upload(config),
        'environment_uploads': _get_environment_uploads(config),
        'network_isolation_enabled': main_network_policy is not None,
        'verifier_network_isolation_enabled': verifier_network_policy is not None,
        **main_boot_config,
        **verifier_boot_config,
    }
    return CompiledHarborTask(
        task_name=task_name,
        task_config=task_config,
        runtime_configs=runtime_configs,
        runtime_provider=runtime_provider,
        metadata={
            'task_name': task_name,
            'docker_image': docker_image,
            **(
                {
                    'execution_kind': 'compose',
                    'compose_definition_sha256': compose_plan.definition_sha256,
                    'compose_controller_image': compose_plan.controller_image,
                    'compose_primary_service': compose_plan.primary_service,
                    'compose_primary_image': compose_plan.primary_image,
                    'compose_services': sorted(compose_plan.services),
                }
                if compose_plan is not None
                else {}
            ),
            **({'instance_id': metadata_instance_id} if metadata_instance_id is not None else {}),
        },
    )


__all__ = [
    'CompiledHarborTask',
    'LoadedHarborTask',
    'compile_harbor_task',
    'load_harbor_task',
    'load_task_definition',
]
