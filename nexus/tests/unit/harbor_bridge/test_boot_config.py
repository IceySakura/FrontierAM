import asyncio
import contextlib
import json
import re
import shutil
import subprocess
import traceback
import types

from pathlib import Path

import pytest

from nexus.extensions.harbor_bridge import compiler, generator
from nexus.extensions.harbor_bridge.task import (
    BOOT_HEALTHCHECK_INTERVAL_SEC,
    BOOT_HEALTHCHECK_RETRIES,
    _agent_watchdog_timeout,
    _boot_image_services,
    _build_boot_launch_command,
    _non_numeric_reward_entries,
    _read_optional_verifier_stdout,
    _restore_tests_executable_bits,
    _run_boot_healthcheck,
    _run_harbor_verifier,
    _split_reward_payload,
    _validate_env_mapping,
    _validate_reward_contract,
)


def test_restore_tests_executable_bits_marks_staged_scripts_executable(tmp_path):
    script = tmp_path / 'tests' / 'helpers' / 'test.source.sh'
    script.parent.mkdir(parents=True)
    script.write_text('#!/bin/sh\n', encoding='utf-8')
    script.chmod(0o644)

    _restore_tests_executable_bits(tmp_path / 'tests')

    assert script.stat().st_mode & 0o111 == 0o111


def test_agent_watchdog_adds_internal_setup_and_cleanup_grace():
    assert _agent_watchdog_timeout(600) == 900.0


def test_validate_env_mapping_rejects_invalid_names():
    with pytest.raises(ValueError, match='Invalid environment variable name'):
        _validate_env_mapping({'SAFE; touch /tmp/pwned': 'x'})


def test_validate_env_mapping_normalizes_values_without_shell_interpolation():
    assert _validate_env_mapping({'SAFE_NAME': 'hello; touch /tmp/pwned'}) == {'SAFE_NAME': 'hello; touch /tmp/pwned'}


@pytest.mark.asyncio
async def test_humanlaya_boot_receives_trusted_judge_aliases(monkeypatch):
    captured = {}

    async def run_command(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(return_code=0, stdout='', stderr='')

    for name, value in {
        'YOUR_API_KEY': 'secret-key',
        'YOUR_BASE_URL': 'https://judge.example/v1',
        'JUDGE_MODEL': 'judge-model',
    }.items():
        monkeypatch.setenv(name, value)

    task_input = types.SimpleNamespace(
        boot_command='bash /environment/start-services.sh',
        boot_env={'TASK_MCP_URL': 'http://127.0.0.1:8000/mcp'},
    )
    await _boot_image_services(types.SimpleNamespace(run_command=run_command), task_input)

    assert captured['extra_env'] == {
        'TASK_MCP_URL': 'http://127.0.0.1:8000/mcp',
        'YOUR_API_KEY': 'secret-key',
        'YOUR_BASE_URL': 'https://judge.example/v1',
        'JUDGE_MODEL': 'judge-model',
    }


def write_valid_task(parent: Path, name: str, instruction: str = 'Complete the task.') -> Path:
    task_dir = parent / name
    (task_dir / 'environment').mkdir(parents=True)
    (task_dir / 'tests').mkdir()
    (task_dir / 'task.toml').write_text('')
    (task_dir / 'instruction.md').write_text(instruction)
    (task_dir / 'tests' / 'test.sh').write_text('#!/bin/sh\n')
    return task_dir


def test_docker_image_must_be_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv('HARBOR_DEFAULT_IMAGE', 'unsafe-fallback:v1')

    with pytest.raises(ValueError, match=r'requires an explicit \[environment\]\.docker_image') as exc_info:
        compiler._get_docker_image({'environment': {}}, tmp_path)

    assert str(tmp_path) in str(exc_info.value)
    assert 'prebuild and publish' in str(exc_info.value)


@pytest.mark.parametrize('definition_name', ['Dockerfile', 'docker-compose.yaml', 'docker-compose.yml'])
def test_dockerfile_and_compose_do_not_fall_back_to_incomplete_images(tmp_path, definition_name):
    environment_dir = tmp_path / 'environment'
    environment_dir.mkdir()
    (environment_dir / definition_name).write_text('FROM python:3.13\n')

    with pytest.raises(ValueError, match='Dockerfile and Docker Compose builds are not supported'):
        compiler._get_docker_image({'environment': {}}, tmp_path)


def test_explicit_docker_image_is_used_verbatim_after_trimming():
    assert compiler._get_docker_image({'environment': {'docker_image': ' agent:v1 '}}) == 'agent:v1'


def test_workdir_must_be_explicit_and_absolute(tmp_path):
    with pytest.raises(ValueError, match=r'requires an explicit \[environment\]\.workdir'):
        compiler._get_workdir({'environment': {}}, tmp_path)
    with pytest.raises(ValueError, match='must be an absolute container path'):
        compiler._get_workdir({'environment': {'workdir': 'relative/path'}}, tmp_path)

    assert compiler._get_workdir({'environment': {'workdir': ' /workspace/task '}}, tmp_path) == '/workspace/task'


def test_boot_config_is_compiled_without_copying_native_healthcheck():
    config = {
        'environment': {
            'env': {'TASK_ID': 'abc'},
            'boot': {'command': '/usr/local/bin/start.sh'},
            'healthcheck': {
                'command': 'curl -f http://127.0.0.1:9900/health',
                'interval_sec': 5.0,
                'timeout_sec': 10.0,
                'retries': 60,
                'start_period_sec': 30.0,
            },
        }
    }

    boot = compiler._build_boot_config(config)

    assert boot == {
        'boot_command': '/usr/local/bin/start.sh',
        'boot_env': {'TASK_ID': 'abc'},
    }
    assert not any('healthcheck' in key for key in boot)


def test_boot_config_ignores_non_schema_boot_fields():
    config = {
        'environment': {
            'boot': {
                'command': '/boot.sh',
                'ignored_command': 'test -f /tmp/ready',
                'ignored_url': 'http://localhost:1/ready',
                'ignored_retries': 7,
                'ignored_interval_sec': 2.5,
                'ignored_timeout_sec': 1.25,
                'ignored_start_period_sec': 4.0,
            },
            'healthcheck': {
                'command': 'curl -f http://127.0.0.1:9900/health',
                'interval_sec': 5.0,
                'timeout_sec': 10.0,
                'retries': 60,
                'start_period_sec': 30.0,
            },
        }
    }

    boot = compiler._build_boot_config(config)

    assert boot == {'boot_command': '/boot.sh', 'boot_env': {}}


def test_boot_config_allows_command_without_healthcheck():
    boot = compiler._build_boot_config({'environment': {'boot': {'command': '/boot.sh'}}})

    assert boot == {'boot_command': '/boot.sh', 'boot_env': {}}


def test_healthcheck_without_boot_is_not_serialized_into_task_input():
    config = {
        'environment': {
            'env': {'SERVICE': 'api'},
            'healthcheck': {'command': 'test -f /tmp/ready'},
        }
    }

    assert compiler._build_boot_config(config) == {}


@pytest.mark.asyncio
async def test_boot_healthcheck_widens_harbor_defaults():
    from harbor.models.task.config import HealthcheckConfig

    healthcheck = HealthcheckConfig(command='test -f /tmp/ready')
    captured = {}

    async def run_healthcheck(config=None):
        captured['config'] = config

    environment = types.SimpleNamespace(
        task_env_config=types.SimpleNamespace(healthcheck=healthcheck),
        run_healthcheck=run_healthcheck,
    )

    await _run_boot_healthcheck(environment)

    effective = captured['config']
    assert effective.interval_sec == BOOT_HEALTHCHECK_INTERVAL_SEC
    assert effective.retries == BOOT_HEALTHCHECK_RETRIES
    assert effective.timeout_sec == healthcheck.timeout_sec
    assert effective.command == healthcheck.command


@pytest.mark.asyncio
async def test_boot_healthcheck_preserves_explicit_task_policy():
    from harbor.models.task.config import HealthcheckConfig

    healthcheck = HealthcheckConfig(
        command='curl -f http://127.0.0.1:9900/health',
        interval_sec=2.5,
        timeout_sec=7.0,
        retries=12,
        start_period_sec=30.0,
    )
    captured = {}

    async def run_healthcheck(config=None):
        captured['config'] = config

    environment = types.SimpleNamespace(
        task_env_config=types.SimpleNamespace(healthcheck=healthcheck),
        run_healthcheck=run_healthcheck,
    )

    await _run_boot_healthcheck(environment)

    assert captured['config'] is healthcheck
    assert captured['config'].model_dump() == healthcheck.model_dump()


def test_boot_launcher_propagates_fast_child_failure(tmp_path):
    command = _build_boot_launch_command(
        'echo boot exploded >&2; exit 23',
        log_dir=str(tmp_path / 'logs'),
        startup_grace_sec=1.0,
    )

    result = subprocess.run(
        command,
        shell=True,
        executable='/bin/bash',
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 23
    assert 'boot process exited during startup with code 23' in result.stdout
    assert 'boot exploded' in result.stdout


def test_boot_launcher_falls_back_to_sh_without_bash(tmp_path):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name in ('mkdir', 'rm', 'setsid', 'nohup', 'sleep', 'sh'):
        executable = shutil.which(name)
        assert executable is not None
        (bin_dir / name).symlink_to(executable)

    selected_shell_path = tmp_path / 'selected-shell.txt'
    command = _build_boot_launch_command(
        f'printf "%s" "$0" > {selected_shell_path}',
        log_dir=str(tmp_path / 'logs'),
        startup_grace_sec=0.1,
    )

    result = subprocess.run(
        command,
        shell=True,
        executable='/bin/sh',
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
        env={'PATH': str(bin_dir), 'BASH': '/missing/bash'},
    )

    assert result.returncode == 0
    assert selected_shell_path.read_text().endswith('/sh')
    assert 'shell=' in result.stdout
    assert 'boot_shell_args="-c"' in command


def test_boot_launcher_fails_explicitly_without_usable_shell(tmp_path):
    command = _build_boot_launch_command(
        'true',
        log_dir=str(tmp_path / 'logs'),
        startup_grace_sec=0,
    )

    assert 'No usable bash or sh found for Harbor boot command' in command
    assert 'exit 127' in command


def test_composite_env_templates_resolve_multiple_variables(monkeypatch):
    monkeypatch.setenv('JUDGE_APP_ID', 'app')
    monkeypatch.setenv('JUDGE_APP_KEY', 'key')
    monkeypatch.delenv('JUDGE_MODEL', raising=False)

    assert compiler.resolve_env_mapping({
        'OPENAI_API_KEY': '${JUDGE_APP_ID}:${JUDGE_APP_KEY}',
        'JUDGE_MODEL': '${JUDGE_MODEL:-default-model}',
        'LITERAL': '{"thinking": true}',
    }) == {
        'OPENAI_API_KEY': 'app:key',
        'JUDGE_MODEL': 'default-model',
        'LITERAL': '{"thinking": true}',
    }


def test_composite_env_template_requires_missing_variable(monkeypatch):
    monkeypatch.delenv('MISSING_JUDGE_KEY', raising=False)

    with pytest.raises(ValueError, match='MISSING_JUDGE_KEY'):
        compiler.resolve_env_mapping({'OPENAI_API_KEY': '${MISSING_JUDGE_KEY}'})


def test_skip_environment_upload_defaults_and_explicit_overrides():
    assert compiler._get_skip_environment_upload({'version': 'T3-RL-v1.0', 'environment': {}}) is True
    assert compiler._get_skip_environment_upload({'schema_version': '1.2', 'environment': {}}) is True
    assert (
        compiler._get_skip_environment_upload({'version': 'T3-RL-v1.0', 'environment': {'skip_upload': False}}) is False
    )
    assert (
        compiler._get_skip_environment_upload({'schema_version': '1.2', 'environment': {'skip_upload': True}}) is True
    )


@pytest.mark.parametrize('value', ['false', 'true', 0, 1, None])
def test_boolean_bridge_config_rejects_coercion(value):
    with pytest.raises(ValueError, match='must be a boolean'):
        compiler._require_bool(value, 'test.flag')


@pytest.mark.parametrize('value', ['false', 0, None])
def test_skip_environment_upload_requires_a_real_boolean(value):
    with pytest.raises(ValueError, match='environment.skip_upload must be a boolean'):
        compiler._get_skip_environment_upload({'environment': {'skip_upload': value}})


def test_timeout_config_preserves_fractions_and_applies_multiplier():
    assert (
        compiler._scaled_positive_timeout(
            {'agent': {'timeout_sec': 0.5}},
            'agent',
            default=900.0,
            multiplier=2.0,
        )
        == 1.0
    )


@pytest.mark.parametrize('value', [True, '1', 0, -1, float('nan'), float('inf')])
def test_timeout_config_rejects_invalid_values(value):
    with pytest.raises(ValueError, match=r'agent\.timeout_sec'):
        compiler._scaled_positive_timeout(
            {'agent': {'timeout_sec': value}},
            'agent',
            default=900.0,
            multiplier=1.0,
        )


@pytest.mark.parametrize(
    ('memory_mb', 'gpus', 'message'),
    [
        (-1, 0, 'memory must be positive'),
        (1024, -1, 'gpus must be non-negative'),
    ],
)
def test_resource_spec_rejects_negative_resources(memory_mb, gpus, message):
    environment = types.SimpleNamespace(
        os=types.SimpleNamespace(value='linux'),
        tpu=None,
        gpu_types=None,
        storage_mb=None,
        cpus=2,
        memory_mb=memory_mb,
        gpus=gpus,
    )

    with pytest.raises(ValueError, match=message):
        compiler._build_resource_spec(
            environment,
            fallback_cpu=2,
            fallback_memory_gb=4,
            role='agent',
        )


def test_environment_upload_config_uses_items_array():
    config = {
        'environment': {
            'upload': {
                'items': [
                    {'source': 'environment/debug_data', 'target': '/app/debug_data'},
                    {'source': 'environment/docker-compose.yml'},
                ]
            }
        }
    }

    assert compiler._get_environment_uploads(config) == [
        {'source': 'environment/debug_data', 'target': '/app/debug_data'},
        {'source': 'environment/docker-compose.yml', 'target': ''},
    ]


def _gpu_environment(*, gpus: int, gpu_types):
    return types.SimpleNamespace(
        os=types.SimpleNamespace(value='linux'),
        tpu=None,
        gpu_types=gpu_types,
        storage_mb=None,
        cpus=2,
        memory_mb=4096,
        gpus=gpus,
    )


@pytest.mark.parametrize('provider', ['gongfeng', 'suanli', 'taiji'])
def test_resource_spec_maps_single_gpu_type_to_sandbox_card_type(provider):
    spec = compiler._build_resource_spec(
        _gpu_environment(gpus=1, gpu_types=['H20']),
        fallback_cpu=2,
        fallback_memory_gb=4,
        role='agent',
        runtime_provider=provider,
    )

    assert spec == {'cpu': 2, 'memory': 4, 'gpu': 1, 'gpu_card_type': 'H20'}


def test_resource_spec_rejects_gpu_types_for_providers_without_card_types():
    with pytest.raises(ValueError, match='cannot be guaranteed by docker'):
        compiler._build_resource_spec(
            _gpu_environment(gpus=1, gpu_types=['H20']),
            fallback_cpu=2,
            fallback_memory_gb=4,
            role='agent',
            runtime_provider='docker',
        )


def test_resource_spec_rejects_multiple_gpu_types():
    with pytest.raises(ValueError, match='multiple card types'):
        compiler._build_resource_spec(
            _gpu_environment(gpus=2, gpu_types=['H20', 'B200']),
            fallback_cpu=2,
            fallback_memory_gb=4,
            role='agent',
            runtime_provider='taiji',
        )


def test_resource_spec_rejects_gpu_types_without_gpus():
    with pytest.raises(ValueError, match='without requesting gpus'):
        compiler._build_resource_spec(
            _gpu_environment(gpus=0, gpu_types=['H20']),
            fallback_cpu=2,
            fallback_memory_gb=4,
            role='agent',
            runtime_provider='taiji',
        )


def test_resource_spec_rejects_empty_gpu_type():
    with pytest.raises(ValueError, match='non-empty card type'):
        compiler._build_resource_spec(
            _gpu_environment(gpus=1, gpu_types=['  ']),
            fallback_cpu=2,
            fallback_memory_gb=4,
            role='agent',
            runtime_provider='taiji',
        )


def test_resource_spec_allows_123gb_for_gpu_sandbox():
    environment = _gpu_environment(gpus=1, gpu_types=['H20'])
    environment.memory_mb = 123 * 1024

    spec = compiler._build_resource_spec(
        environment,
        fallback_cpu=2,
        fallback_memory_gb=4,
        role='agent',
        runtime_provider='taiji',
    )

    assert spec == {'cpu': 2, 'memory': 123, 'gpu': 1, 'gpu_card_type': 'H20'}


def test_resource_spec_keeps_64gb_limit_for_cpu_sandbox():
    environment = _gpu_environment(gpus=0, gpu_types=None)
    environment.memory_mb = 65 * 1024

    with pytest.raises(ValueError, match='at most 64 GB for CPU sandbox'):
        compiler._build_resource_spec(
            environment,
            fallback_cpu=2,
            fallback_memory_gb=4,
            role='agent',
            runtime_provider='gongfeng',
        )


def test_resource_spec_rejects_more_than_123gb_for_gpu_sandbox():
    environment = _gpu_environment(gpus=1, gpu_types=['H20'])
    environment.memory_mb = 124 * 1024

    with pytest.raises(ValueError, match='at most 123 GB for GPU sandbox'):
        compiler._build_resource_spec(
            environment,
            fallback_cpu=2,
            fallback_memory_gb=4,
            role='agent',
            runtime_provider='taiji',
        )


def test_environment_upload_config_allows_single_direct_item():
    config = {'environment': {'upload': {'source': 'environment/debug_data', 'target': '/debug'}}}

    assert compiler._get_environment_uploads(config) == [{'source': 'environment/debug_data', 'target': '/debug'}]


def test_environment_upload_config_absent_is_empty():
    assert compiler._get_environment_uploads({'environment': {}}) == []


@pytest.mark.parametrize(
    ('upload', 'message'),
    [
        ([], 'environment.upload must be a table'),
        ({}, 'exactly one of source or items'),
        ({'source': 'environment/a', 'items': [{'source': 'environment/b'}]}, 'exactly one of source or items'),
        ({'items': 'bad'}, 'items must be a non-empty array'),
        ({'items': {'source': 'environment/a'}}, 'items must be a non-empty array'),
        ({'items': []}, 'items must be a non-empty array'),
        ({'items': ['bad']}, r'items\[0\] must be a table'),
        ({'items': [{}]}, r'items\[0\]\.source must be a non-empty string'),
        ({'source': '/absolute/source'}, 'source must be a safe task-relative path'),
        ({'source': '../solution'}, 'source must be a safe task-relative path'),
        ({'source': 'environment/a', 'target': 'relative/target'}, 'target must be an absolute container path'),
        ({'source': 'environment/a', 'target': ''}, 'target must be a non-empty absolute container path'),
    ],
)
def test_environment_upload_config_rejects_invalid_shapes(upload, message):
    with pytest.raises(ValueError, match=message):
        compiler._get_environment_uploads({'environment': {'upload': upload}})


def test_task_declared_runtime_image_is_ignored_by_compiler(tmp_path):
    task_dir = write_valid_task(tmp_path, 'runtime-image')
    (task_dir / 'task.toml').write_text(
        """schema_version = "1.2"

[environment]
docker_image = "agent:v1"
runtime_image = "task-owned-runtime:forbidden"
workdir = "/app"

[verifier]
environment_mode = "shared"
"""
    )
    loaded = compiler.load_harbor_task(task_dir)

    compiled = compiler.compile_harbor_task(task_dir, loaded.config, loaded.task)

    assert compiled.runtime_configs['main']['image'] == 'agent:v1'
    assert 'runtime_image' not in compiled.runtime_configs['main']


def test_compile_maps_main_and_separate_verifier_gpu_types(tmp_path):
    task_dir = write_valid_task(tmp_path, 'separate-verifier-gpu')
    (task_dir / 'task.toml').write_text(
        """schema_version = "1.2"

[environment]
docker_image = "agent:v1"
workdir = "/workspace"
gpus = 1
gpu_types = ["H20"]
memory_mb = 125952

[verifier]
environment_mode = "separate"

[verifier.environment]
docker_image = "verifier:v1"
workdir = "/workspace"
gpus = 1
gpu_types = ["H20"]
memory_mb = 125952
"""
    )
    loaded = compiler.load_harbor_task(task_dir)

    compiled = compiler.compile_harbor_task(
        task_dir,
        loaded.config,
        loaded.task,
        configured_runtime_provider='taiji',
    )

    expected = {'cpu': 2, 'memory': 123, 'gpu': 1, 'gpu_card_type': 'H20'}
    assert compiled.runtime_configs['main']['resource_spec'] == expected
    assert compiled.runtime_configs['verifier']['resource_spec'] == expected


def test_non_numeric_reward_extras_recover_quietly_but_real_failures_stay_loud():
    from pydantic import BaseModel, ValidationError

    class _VerifierResult(BaseModel):
        """Mirrors how harbor types VerifierResult.rewards."""

        rewards: dict[str, float | int]

    numeric, extra = _split_reward_payload({'reward': 0.5, 'steps': 3, 'answer_found': True, 'judge_reason': 'matched'})
    assert numeric == {'reward': 0.5, 'steps': 3}
    assert extra == {'answer_found': True, 'judge_reason': 'matched'}

    # answer_found / judge_reason next to the score: recoverable, so it must not warn.
    with pytest.raises(ValidationError) as extras_only:
        _VerifierResult(rewards={'reward': 0.5, 'judge_reason': 'matched'})
    assert _non_numeric_reward_entries(extras_only.value) == {'judge_reason'}

    # rewards itself being the wrong shape is structural, not an extras problem.
    with pytest.raises(ValidationError) as structural:
        _VerifierResult(rewards='not-a-mapping')
    assert _non_numeric_reward_entries(structural.value) == set()


@pytest.mark.parametrize('reward', [-1, 0, 0.5, 1])
def test_reward_contract_accepts_system_error_and_unit_interval(reward):
    _validate_reward_contract({'reward': reward})


@pytest.mark.parametrize(
    ('rewards', 'message'),
    [
        ({}, 'must contain'),
        ({'reward': True}, 'must be a number'),
        ({'reward': float('nan')}, 'must be finite'),
        ({'reward': float('inf')}, 'must be finite'),
        ({'reward': -0.5}, 'must be -1 or within'),
        ({'reward': 1.1}, 'must be -1 or within'),
        ({'reward': 0.5, 'metric': float('nan')}, 'must be finite'),
    ],
)
def test_reward_contract_rejects_missing_nonfinite_and_out_of_range_values(rewards, message):
    with pytest.raises(ValueError, match=message):
        _validate_reward_contract(rewards)


def test_reward_contract_checks_raw_json_before_pydantic_coercion():
    with pytest.raises(ValueError, match='must be a number'):
        _validate_reward_contract({'reward': 1.0}, raw_payload={'reward': True})


@pytest.mark.asyncio
async def test_verifier_receives_requested_agent_trajectory_path(monkeypatch, tmp_path):
    from harbor.models.task.config import TaskOS

    from nexus.extensions.harbor_bridge.agent_result import T3_AGENT_RESULT_ENV
    from nexus.extensions.harbor_bridge.agent_trajectory import T3_AGENT_TRAJECTORY_ENV

    captured: dict[str, object] = {}

    class _Verifier:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def verify(self):
            return types.SimpleNamespace(rewards={'reward': 1.0})

    monkeypatch.setattr('harbor.verifier.verifier.Verifier', _Verifier)
    trial_paths = types.SimpleNamespace(
        reward_json_path=tmp_path / 'reward.json',
        test_stdout_path=tmp_path / 'stdout.txt',
    )
    harbor_task = types.SimpleNamespace(
        config=types.SimpleNamespace(verifier=types.SimpleNamespace(user=None, env={'JUDGE_MODEL': 'test'})),
        paths=types.SimpleNamespace(tests_dir=tmp_path / 'tests'),
    )
    environment = types.SimpleNamespace(
        os=TaskOS.LINUX,
        with_default_user=lambda _user: contextlib.nullcontext(),
    )

    await _run_harbor_verifier(
        harbor_task=harbor_task,
        trial_paths=trial_paths,
        environment=environment,
        skip_tests_upload=False,
        timeout_sec=10,
        needs_agent_trajectory=True,
    )

    assert captured['verifier_env'] == {
        'JUDGE_MODEL': 'test',
        T3_AGENT_RESULT_ENV: '/logs/agent/result.json',
        T3_AGENT_TRAJECTORY_ENV: '/logs/agent/trajectory.json',
    }
    assert captured['include_logs'] == ['stdout.txt']
    assert captured['exclude_logs'] is None


@pytest.mark.asyncio
async def test_verifier_reward_only_mode_excludes_all_optional_logs(monkeypatch, tmp_path):
    from harbor.models.task.config import TaskOS

    captured: dict[str, object] = {}

    class _Verifier:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def verify(self):
            return types.SimpleNamespace(rewards={'reward': 1.0})

    monkeypatch.setattr('harbor.verifier.verifier.Verifier', _Verifier)
    trial_paths = types.SimpleNamespace(
        reward_json_path=tmp_path / 'reward.json',
        test_stdout_path=tmp_path / 'test-stdout.txt',
    )
    harbor_task = types.SimpleNamespace(
        config=types.SimpleNamespace(verifier=types.SimpleNamespace(user=None, env={})),
        paths=types.SimpleNamespace(tests_dir=tmp_path / 'tests'),
    )
    environment = types.SimpleNamespace(
        os=TaskOS.LINUX,
        with_default_user=lambda _user: contextlib.nullcontext(),
    )

    rewards, _, test_stdout = await _run_harbor_verifier(
        harbor_task=harbor_task,
        trial_paths=trial_paths,
        environment=environment,
        skip_tests_upload=False,
        timeout_sec=10,
        include_test_stdout=False,
    )

    assert rewards == {'reward': 1.0}
    assert test_stdout == ''
    assert captured['include_logs'] is None
    assert captured['exclude_logs'] == ['*']


@pytest.mark.asyncio
async def test_verifier_system_error_reward_fails_task(monkeypatch, tmp_path):
    from harbor.models.task.config import TaskOS

    secret = 'judge-secret-value'
    reward_json = tmp_path / 'reward.json'
    reward_json.write_text(
        json.dumps({
            'reward': -1,
            'error': f'missing required trajectory; authorization=Bearer {secret}',
            'api_key': secret,
        }),
        encoding='utf-8',
    )
    test_stdout = tmp_path / 'stdout.txt'
    test_stdout.write_text(f'grader failed with token={secret}\nlast useful line\n', encoding='utf-8')

    class _Verifier:
        def __init__(self, **_kwargs):
            pass

        async def verify(self):
            return types.SimpleNamespace(rewards={'reward': -1})

    monkeypatch.setattr('harbor.verifier.verifier.Verifier', _Verifier)
    trial_paths = types.SimpleNamespace(
        reward_json_path=reward_json,
        test_stdout_path=test_stdout,
    )
    harbor_task = types.SimpleNamespace(
        config=types.SimpleNamespace(verifier=types.SimpleNamespace(user=None, env={'JUDGE_API_KEY': secret})),
        paths=types.SimpleNamespace(tests_dir=tmp_path / 'tests'),
    )
    environment = types.SimpleNamespace(
        os=TaskOS.LINUX,
        with_default_user=lambda _user: contextlib.nullcontext(),
    )

    with pytest.raises(RuntimeError, match='Verifier reported a system error with reward=-1') as exc_info:
        await _run_harbor_verifier(
            harbor_task=harbor_task,
            trial_paths=trial_paths,
            environment=environment,
            skip_tests_upload=False,
            timeout_sec=10,
        )

    message = str(exc_info.value)
    assert 'missing required trajectory' in message
    assert 'last useful line' in message
    assert secret not in message
    assert '<redacted>' in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'exception_type',
    [
        pytest.param('RewardFileNotFoundError', id='missing'),
        pytest.param('RewardFileEmptyError', id='empty'),
        pytest.param('VerifierOutputParseError', id='invalid'),
    ],
)
async def test_unusable_reward_reports_redacted_verifier_output(monkeypatch, tmp_path, exception_type):
    from harbor.models.task.config import TaskOS
    from harbor.verifier import verifier as verifier_module

    secret = 'reward-diagnostic-secret'
    test_stdout = tmp_path / 'stdout.txt'
    test_stdout.write_text(
        f'initial noise\nverifier failed because token={secret}\nfinal diagnostic line\n',
        encoding='utf-8',
    )

    class _Verifier:
        def __init__(self, **_kwargs):
            pass

        async def verify(self):
            raise getattr(verifier_module, exception_type)(f'reward unavailable with api_key={secret}')

    monkeypatch.setattr('harbor.verifier.verifier.Verifier', _Verifier)
    trial_paths = types.SimpleNamespace(
        reward_json_path=tmp_path / 'reward.json',
        test_stdout_path=test_stdout,
    )
    harbor_task = types.SimpleNamespace(
        config=types.SimpleNamespace(verifier=types.SimpleNamespace(user=None, env={'JUDGE_API_KEY': secret})),
        paths=types.SimpleNamespace(tests_dir=tmp_path / 'tests'),
    )
    environment = types.SimpleNamespace(
        os=TaskOS.LINUX,
        with_default_user=lambda _user: contextlib.nullcontext(),
    )

    with pytest.raises(RuntimeError, match='Verifier did not produce a usable reward') as exc_info:
        await _run_harbor_verifier(
            harbor_task=harbor_task,
            trial_paths=trial_paths,
            environment=environment,
            skip_tests_upload=False,
            timeout_sec=10,
        )

    message = str(exc_info.value)
    rendered_traceback = ''.join(
        traceback.format_exception(type(exc_info.value), exc_info.value, exc_info.value.__traceback__)
    )
    assert exception_type in message
    assert 'final diagnostic line' in message
    assert secret not in message
    assert secret not in rendered_traceback
    assert '<redacted>' in message


@pytest.mark.asyncio
async def test_reward_only_failure_captures_bounded_remote_stdout_tail(monkeypatch, tmp_path):
    from harbor.models.task.config import TaskOS
    from harbor.verifier.verifier import RewardFileNotFoundError

    class _Verifier:
        def __init__(self, **_kwargs):
            pass

        async def verify(self):
            raise RewardFileNotFoundError('missing reward')

    class _Environment:
        os = TaskOS.LINUX

        def __init__(self):
            self.tail_requests = []

        def with_default_user(self, _user):
            return contextlib.nullcontext()

        async def download_file_tail(self, source_path, target_path, *, max_bytes):
            self.tail_requests.append((source_path, target_path, max_bytes))
            target_path.write_bytes(b'final remote diagnostic')

    monkeypatch.setattr('harbor.verifier.verifier.Verifier', _Verifier)
    trial_paths = types.SimpleNamespace(
        reward_json_path=tmp_path / 'reward.json',
        test_stdout_path=tmp_path / 'test-stdout.txt',
    )
    harbor_task = types.SimpleNamespace(
        config=types.SimpleNamespace(verifier=types.SimpleNamespace(user=None, env={})),
        paths=types.SimpleNamespace(tests_dir=tmp_path / 'tests'),
    )
    environment = _Environment()

    with pytest.raises(RuntimeError, match='final remote diagnostic'):
        await _run_harbor_verifier(
            harbor_task=harbor_task,
            trial_paths=trial_paths,
            environment=environment,
            skip_tests_upload=False,
            timeout_sec=10,
            include_test_stdout=False,
        )

    assert environment.tail_requests == [
        (
            '/logs/verifier/test-stdout.txt',
            trial_paths.test_stdout_path,
            10000,
        )
    ]


@pytest.mark.asyncio
async def test_structural_verifier_validation_error_is_not_recovered(monkeypatch, tmp_path):
    from harbor.models.task.config import TaskOS
    from pydantic import BaseModel, ValidationError

    class _VerifierResult(BaseModel):
        rewards: dict[str, float | int]

    with pytest.raises(ValidationError) as structural:
        _VerifierResult(rewards='not-a-mapping')

    class _Verifier:
        def __init__(self, **_):
            pass

        async def verify(self):
            raise structural.value

    monkeypatch.setattr('harbor.verifier.verifier.Verifier', _Verifier)
    reward_path = tmp_path / 'reward.json'
    reward_path.write_text('{"reward": 1.0}')
    trial_paths = types.SimpleNamespace(
        reward_json_path=reward_path,
        test_stdout_path=tmp_path / 'stdout.txt',
    )
    harbor_task = types.SimpleNamespace(
        config=types.SimpleNamespace(
            verifier=types.SimpleNamespace(user=None, env={}),
        ),
        paths=types.SimpleNamespace(tests_dir=tmp_path / 'tests'),
    )
    environment = types.SimpleNamespace(
        os=TaskOS.LINUX,
        with_default_user=lambda _user: contextlib.nullcontext(),
    )

    with pytest.raises(ValidationError):
        await _run_harbor_verifier(
            harbor_task=harbor_task,
            trial_paths=trial_paths,
            environment=environment,
            skip_tests_upload=False,
            timeout_sec=10,
        )


@pytest.mark.asyncio
async def test_verifier_timeout_wraps_complete_verification(monkeypatch, tmp_path):
    from harbor.models.task.config import TaskOS

    class _Verifier:
        def __init__(self, **_kwargs):
            pass

        async def verify(self):
            await asyncio.sleep(1)

    monkeypatch.setattr('harbor.verifier.verifier.Verifier', _Verifier)
    trial_paths = types.SimpleNamespace(
        reward_json_path=tmp_path / 'reward.json',
        test_stdout_path=tmp_path / 'stdout.txt',
    )
    harbor_task = types.SimpleNamespace(
        config=types.SimpleNamespace(verifier=types.SimpleNamespace(user=None, env={})),
        paths=types.SimpleNamespace(tests_dir=tmp_path / 'tests'),
    )
    environment = types.SimpleNamespace(
        os=TaskOS.LINUX,
        with_default_user=lambda _user: contextlib.nullcontext(),
    )

    with pytest.raises(TimeoutError, match='Verifier timed out after 0.01s'):
        await _run_harbor_verifier(
            harbor_task=harbor_task,
            trial_paths=trial_paths,
            environment=environment,
            skip_tests_upload=False,
            timeout_sec=0.01,
        )


@pytest.mark.asyncio
async def test_verifier_stdout_replaces_non_utf8_bytes(monkeypatch, tmp_path):
    from harbor.models.task.config import TaskOS

    class _Verifier:
        def __init__(self, **_kwargs):
            pass

        async def verify(self):
            return types.SimpleNamespace(rewards={'reward': 1.0})

    monkeypatch.setattr('harbor.verifier.verifier.Verifier', _Verifier)
    reward_path = tmp_path / 'reward.json'
    reward_path.write_text('{"reward": 1.0}', encoding='utf-8')
    test_stdout_path = tmp_path / 'stdout.txt'
    test_stdout_path.write_bytes(b'before' + bytes([0xFF]) + b'after')
    trial_paths = types.SimpleNamespace(
        reward_json_path=reward_path,
        test_stdout_path=test_stdout_path,
    )
    harbor_task = types.SimpleNamespace(
        config=types.SimpleNamespace(verifier=types.SimpleNamespace(user=None, env={})),
        paths=types.SimpleNamespace(tests_dir=tmp_path / 'tests'),
    )
    environment = types.SimpleNamespace(
        os=TaskOS.LINUX,
        with_default_user=lambda _user: contextlib.nullcontext(),
    )

    rewards, _, test_stdout = await _run_harbor_verifier(
        harbor_task=harbor_task,
        trial_paths=trial_paths,
        environment=environment,
        skip_tests_upload=False,
        timeout_sec=10,
    )

    assert rewards == {'reward': 1.0}
    assert test_stdout == f'before{chr(0xFFFD)}after'


def test_optional_verifier_stdout_reads_only_requested_bytes(tmp_path):
    test_stdout_path = tmp_path / 'stdout.txt'
    test_stdout_path.write_bytes(b'abcd' + bytes([0xFF]))

    assert _read_optional_verifier_stdout(test_stdout_path, max_bytes=4) == 'abcd'


def test_optional_verifier_stdout_read_error_does_not_escape(monkeypatch, tmp_path):
    test_stdout_path = tmp_path / 'stdout.txt'

    def fail_open(*_args, **_kwargs):
        raise PermissionError('simulated unreadable log')

    monkeypatch.setattr(test_stdout_path.__class__, 'open', fail_open)

    assert _read_optional_verifier_stdout(test_stdout_path) == ''


def test_task_validation_aggregates_all_failures(monkeypatch, tmp_path):
    monkeypatch.delenv('HARBOR_SKIP_INVALID_TASKS', raising=False)

    bad_toml = tmp_path / 'bad-toml'
    bad_toml.mkdir()
    (bad_toml / 'task.toml').write_text('[environment\n')

    missing_instruction = tmp_path / 'missing-instruction'
    missing_instruction.mkdir()
    (missing_instruction / 'task.toml').write_text('')

    with pytest.raises(ValueError, match='Found 2 invalid Harbor task') as exc_info:
        generator._load_valid_tasks([bad_toml, missing_instruction])

    message = str(exc_info.value)
    assert 'Found 2 invalid Harbor task(s)' in message
    assert str(bad_toml) in message
    assert str(missing_instruction) in message
    assert 'HARBOR_SKIP_INVALID_TASKS=1' in message


def test_parallel_task_validation_preserves_order(monkeypatch, tmp_path):
    task_a = write_valid_task(tmp_path, 'a')
    task_b = write_valid_task(tmp_path, 'b')
    monkeypatch.setenv('HARBOR_LOAD_WORKERS', '4')

    loaded = generator._load_valid_tasks([task_a, task_b])

    assert [task_dir for task_dir, _, _ in loaded] == [task_a, task_b]


def test_skip_invalid_tasks_warns_and_total_matches_generated_set(monkeypatch, tmp_path, caplog):
    write_valid_task(tmp_path, 'valid')
    invalid = tmp_path / 'invalid'
    invalid.mkdir()
    (invalid / 'task.toml').write_text('')

    monkeypatch.setenv('HARBOR_DATASET_PATH', str(tmp_path))
    monkeypatch.setenv('HARBOR_SKIP_INVALID_TASKS', '1')
    for name in ('TASK_NAMES', 'HARBOR_TASK_INCLUDE', 'TASK_SKIP', 'TASK_LIMIT'):
        monkeypatch.delenv(name, raising=False)

    with caplog.at_level('WARNING'):
        assert generator.get_total_count() == 1

    assert f'Skipping invalid Harbor task {invalid}' in caplog.text


def test_flat_task_discovery_only_scans_immediate_children(monkeypatch, tmp_path):
    immediate = write_valid_task(tmp_path, 'immediate')
    nested_root = tmp_path / 'nested'
    nested = write_valid_task(nested_root, 'deep-task')
    monkeypatch.setenv('HARBOR_FLAT_TASKS', '1')
    monkeypatch.delenv('TASK_NAMES', raising=False)

    discovered = generator._discover_task_dirs([tmp_path])

    assert immediate in discovered
    assert nested not in discovered


def test_run_ids_are_stable_across_task_additions_and_dataset_order(tmp_path):
    dataset_a = tmp_path / 'dataset-a'
    dataset_b = tmp_path / 'dataset-b'
    task_a = write_valid_task(dataset_a, 'task-a')
    task_b = write_valid_task(dataset_b, 'task-b')
    unique_task = write_valid_task(dataset_a, 'unique-task')
    valid_tasks = [(task_a, {}, 'a'), (task_b, {}, 'b'), (unique_task, {}, 'unique')]

    initial_ids = generator._build_run_ids([(unique_task, {}, 'unique')])
    run_ids = generator._build_run_ids(valid_tasks)
    reversed_ids = generator._build_run_ids(list(reversed(valid_tasks)))

    assert run_ids[unique_task] == initial_ids[unique_task]
    assert run_ids == reversed_ids
    assert len(set(run_ids.values())) == len(valid_tasks)
    assert all(re.fullmatch(r'[A-Za-z0-9_-]+', run_id) for run_id in run_ids.values())


def test_run_ids_reject_ambiguous_cross_root_identity(tmp_path):
    dataset_a = tmp_path / 'a' / 'dataset'
    dataset_b = tmp_path / 'b' / 'dataset'
    task_a = write_valid_task(dataset_a, 'shared-task')
    task_b = write_valid_task(dataset_b, 'shared-task')
    valid_tasks = [(task_a, {}, 'a'), (task_b, {}, 'b')]

    with pytest.raises(ValueError, match='canonical task content are duplicated'):
        generator._build_run_ids(valid_tasks)


def test_run_ids_do_not_collide_after_nexus_sanitization(tmp_path):
    dataset = tmp_path / 'dataset'
    dotted = write_valid_task(dataset, 'a.b')
    underscored = write_valid_task(dataset, 'a_b')
    valid_tasks = [(dotted, {}, 'dotted'), (underscored, {}, 'underscored')]

    run_ids = generator._build_run_ids(valid_tasks)

    assert len(set(run_ids.values())) == 2
    assert all(re.sub(r'[^a-zA-Z0-9_-]', '_', run_id) == run_id for run_id in run_ids.values())


def test_offline_allowlist_is_parsed_into_deduped_targets():
    assert generator._parse_offline_allowlist('') == ()
    assert generator._parse_offline_allowlist(
        'mirrors.tencent.com:80, llm-api.model-eval.woa.com:80,mirrors.tencent.com:80'
    ) == ('mirrors.tencent.com:80', 'llm-api.model-eval.woa.com:80', 'mirrors.tencent.com:80')


def test_network_policy_forces_suanli_when_provider_is_unset():
    policy = {'default_action': 'deny', 'egress': []}

    # None (env unset) upgrades to suanli for isolated tasks; without a policy it stays gongfeng.
    assert compiler._resolve_runtime_provider_for_network(None, policy) == 'suanli'
    assert compiler._resolve_runtime_provider_for_network('suanli', policy) == 'suanli'
    assert compiler._resolve_runtime_provider_for_network(None, None) == 'gongfeng'


def test_offline_network_policy_rejects_explicit_non_suanli_provider():
    policy = {'default_action': 'deny', 'egress': []}

    with pytest.raises(ValueError, match='requires GONGFENG_RUNTIME_PROVIDER=suanli'):
        compiler._resolve_runtime_provider_for_network('gongfeng', policy)


def test_network_policy_keeps_explicit_taiji_provider():
    policy = {'default_action': 'deny', 'egress': []}

    # Taiji pins runtime_provider=taiji, so an isolated task cannot be routed to suanli.
    assert compiler._resolve_runtime_provider_for_network('taiji', policy) == 'taiji'
    assert compiler._resolve_runtime_provider_for_network('Taiji', policy) == 'taiji'
    assert compiler._resolve_runtime_provider_for_network('taiji', None) == 'taiji'


def test_network_policy_still_rejects_docker_provider():
    policy = {'default_action': 'deny', 'egress': []}

    with pytest.raises(ValueError, match='requires GONGFENG_RUNTIME_PROVIDER=suanli'):
        compiler._resolve_runtime_provider_for_network('docker', policy)
