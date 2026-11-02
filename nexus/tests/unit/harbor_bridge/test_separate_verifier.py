from __future__ import annotations

import json
import subprocess

from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from urllib.parse import parse_qs

import pytest

from nexus.core.context import ExecutionContext
from nexus.extensions.harbor_bridge import compiler, generator
from nexus.extensions.harbor_bridge import task as task_module
from nexus.runtime.client.base import RuntimeUnhealthyError
from nexus.runtime.providers.base import RuntimeProvider
from nexus.runtime.providers.gongfeng import GongfengRuntimeProvider


def test_runtime_directory_reset_falls_back_to_service_tools(tmp_path: Path) -> None:
    reset_dir = tmp_path / 'reset'
    reset_dir.mkdir()
    (reset_dir / 'visible.txt').write_text('visible')
    (reset_dir / '.hidden').write_text('hidden')
    command = task_module._reset_runtime_directories_command(
        [str(reset_dir)],
        [str(reset_dir)],
        runtime_bin='/missing-nexus-runtime-bin',
    )

    subprocess.run(['/bin/sh', '-c', command], check=True)

    assert list(reset_dir.iterdir()) == []
    assert reset_dir.stat().st_mode & 0o777 == 0o777


@pytest.mark.asyncio
async def test_runtime_directory_preparation_explicitly_runs_as_root() -> None:
    environment = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout='', stderr='')))

    await task_module._prepare_runtime_directories(
        environment,
        workdir='/workspace',
        reset_dirs=['/logs/agent', '/tests'],
        writable_dirs=['/logs/agent'],
    )

    environment.exec.assert_awaited_once()
    command = environment.exec.await_args.args[0]
    assert '[ ! -d /workspace ]' in command
    assert 'for dir in /logs/agent /tests' in command
    assert environment.exec.await_args.kwargs == {'timeout_sec': 60, 'user': 'root'}


@pytest.mark.asyncio
async def test_artifact_collection_downloads_main_before_stopping_for_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harbor.models.task.config import TaskOS
    from harbor.trial import artifact_handler as artifact_module

    events: list[tuple[str, object]] = []

    class FakeArtifactHandler:
        def __init__(self, **_kwargs):
            pass

        def begin_collection(self):
            events.append(('begin', None))

        def sidecar_services(self):
            return {'db'}

        async def download_artifacts(self, _environment, _target, **kwargs):
            events.append(('download', set(kwargs['services'])))

    class FakeEnvironment:
        os = TaskOS.LINUX
        capabilities = SimpleNamespace(docker_compose=True)

        async def stop_service(self, service):
            events.append(('stop', service))

    monkeypatch.setattr(artifact_module, 'ArtifactHandler', FakeArtifactHandler)

    await task_module._collect_artifacts(
        FakeEnvironment(),
        SimpleNamespace(config=SimpleNamespace(artifacts=[])),
        SimpleNamespace(artifacts_dir=tmp_path / 'artifacts'),
    )

    assert events == [
        ('begin', None),
        ('download', {'main'}),
        ('stop', 'main'),
        ('download', {'db'}),
    ]


def _write_task(
    root: Path,
    *,
    separate: bool = True,
    verifier_network: str = 'public',
    artifact_service: str = 'main',
    collect_service: str = 'main',
    explicit_verifier_environment: bool = True,
) -> Path:
    task_dir = root / 'task'
    (task_dir / 'environment').mkdir(parents=True)
    (task_dir / 'tests').mkdir()
    mode = 'environment_mode = "separate"' if separate else 'environment_mode = "shared"'
    verifier_environment = (
        f'''

[verifier.environment]
docker_image = "verifier:v1"
workdir = "/verify"
cpus = 3
memory_mb = 6144
network_mode = "{verifier_network}"
{'allowed_hosts = ["judge.woa.com"]' if verifier_network == 'allowlist' else ''}

[verifier.environment.env]
MODE = "evaluation"
'''
        if separate and explicit_verifier_environment
        else ''
    )
    (task_dir / 'task.toml').write_text(
        f'''schema_version = "1.2"
artifacts = [{{ source = "/tmp/model.patch", service = "{artifact_service}" }}]

[environment]
docker_image = "agent:v1"
workdir = "/app"
cpus = 2
memory_mb = 4096
network_mode = "public"

[verifier]
{mode}
timeout_sec = 120

[[verifier.collect]]
service = "{collect_service}"
command = "cp /tmp/source.patch /tmp/model.patch"
timeout_sec = 30
{verifier_environment}
'''
    )
    (task_dir / 'instruction.md').write_text('Complete the task.\n')
    (task_dir / 'tests' / 'test.sh').write_text('#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n')
    return task_dir


def _configure_generator_env(monkeypatch: pytest.MonkeyPatch, task_dir: Path) -> None:
    monkeypatch.setenv('HARBOR_DATASET_PATH', str(task_dir))
    monkeypatch.setenv('GONGFENG_RUNTIME_TOKEN', 'test-token')
    for name in (
        'GONGFENG_RUNTIME_PROVIDER',
        'GONGFENG_PURPOSE',
        'GONGFENG_STARTUP_JITTER_MAX',
        'OPENAI_CACHE_CONTROL_ENABLED',
        'OPENAI_CACHE_CONTROL_STRATEGY',
        'OPENAI_LOG_LLM_USAGE',
        'OPENAI_MAX_RETRIES',
        'OPENAI_MAX_TOKENS',
        'OPENAI_DISABLE_THINKING',
        'OPENAI_THINKING_TYPE',
        'OPENAI_THINKING_DISPLAY',
        'OPENAI_THINKING_EFFORT',
        'ANTHROPIC_MODEL',
        'AGENT_CONFIG_JSON',
        'AGENT_NAME',
        'CLAUDE_CODE_THINKING_PRESET',
        'CLAUDE_CODE_CONVERSION_LOG_LEVEL',
        'API_FORCE_IDLE_TIMEOUT',
        'API_TIMEOUT_MS',
        'CLAUDE_STREAM_IDLE_TIMEOUT_MS',
        'HARBOR_ENABLE_DIND',
        'HARBOR_OFFLINE_ALLOWLIST',
        'TASK_NAMES',
        'HARBOR_TASK_INCLUDE',
        'TASK_SKIP',
        'TASK_LIMIT',
        'RUN_ORACLE',
        'RUN_NOP',
    ):
        monkeypatch.delenv(name, raising=False)


def test_generator_builds_distinct_main_and_verifier_runtime_configs(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, verifier_network='allowlist')
    _configure_generator_env(monkeypatch, task_dir)

    item = next(generator.generate_configs())

    assert item.config.task_config['verifier_mode'] == 'separate'
    assert item.config.task_config['skip_tests_upload'] is False
    assert 'image' not in item.config.runtime_provider_config
    assert item.config.runtime_provider_config['runtime_provider'] == 'suanli'
    assert item.config.runtime_configs['main'] == {
        'image': 'agent:v1',
        'resource_spec': {'cpu': 2, 'memory': 4},
        'working_dir': '/app',
        'environment': {},
        'enable_dind': False,
    }
    assert item.config.runtime_configs['verifier'] == {
        'image': 'verifier:v1',
        'resource_spec': {'cpu': 3, 'memory': 6},
        'working_dir': '/verify',
        'environment': {'MODE': 'evaluation'},
        'enable_dind': False,
        'network_policy': {
            'default_action': 'deny',
            'egress': [{'action': 'allow', 'target': 'judge.woa.com'}],
        },
    }


def test_generator_resolves_composite_separate_verifier_env(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path)
    task_toml = task_dir / 'task.toml'
    task_toml.write_text(
        task_toml.read_text().replace(
            'MODE = "evaluation"',
            'AUTH = "${JUDGE_APP_ID}:${JUDGE_APP_KEY}"',
        )
    )
    _configure_generator_env(monkeypatch, task_dir)
    monkeypatch.setenv('JUDGE_APP_ID', 'judge-app')
    monkeypatch.setenv('JUDGE_APP_KEY', 'judge-key')

    item = next(generator.generate_configs())

    assert item.config.runtime_configs['verifier']['environment'] == {
        'AUTH': 'judge-app:judge-key',
    }


def test_generator_shared_mode_only_configures_main(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    _configure_generator_env(monkeypatch, task_dir)

    item = next(generator.generate_configs())

    assert item.config.task_config['verifier_mode'] == 'shared'
    assert set(item.config.runtime_configs) == {'main'}


def test_compiler_does_not_promote_system_prompt_into_task_config(tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    (task_dir / 'system_prompt.md').write_text('Agent-specific prompt.\n')
    loaded = compiler.load_harbor_task(task_dir)

    compiled = compiler.compile_harbor_task(task_dir, loaded.config, loaded.task)

    assert 'system_prompt' not in compiled.task_config


def test_compiler_rejects_browser_with_separate_verifier(tmp_path):
    task_dir = _write_task(tmp_path, separate=True)
    with (task_dir / 'task.toml').open('a') as task_file:
        task_file.write(
            """

[environment.browser]
start_url = "https://example.com/task"
"""
        )
    loaded = compiler.load_harbor_task(task_dir)

    with pytest.raises(ValueError, match='requires verifier.environment_mode="shared"'):
        compiler.compile_harbor_task(task_dir, loaded.config, loaded.task)


@pytest.mark.parametrize(('env_name', 'mode'), [('RUN_ORACLE', 'oracle'), ('RUN_NOP', 'nop')])
def test_generator_converts_legacy_mode_env_once(monkeypatch, tmp_path, env_name, mode):
    task_dir = _write_task(tmp_path, separate=False)
    _configure_generator_env(monkeypatch, task_dir)
    monkeypatch.setenv(env_name, 'true')

    item = next(generator.generate_configs())

    assert item.config.task_config['mode'] == mode
    assert 'run_oracle' not in item.config.task_config
    assert 'run_nop' not in item.config.task_config


def test_generator_rejects_conflicting_legacy_mode_env(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    _configure_generator_env(monkeypatch, task_dir)
    monkeypatch.setenv('RUN_ORACLE', 'true')
    monkeypatch.setenv('RUN_NOP', 'true')

    with pytest.raises(ValueError, match='cannot both be true'):
        next(generator.generate_configs())


def test_boot_command_does_not_enable_dind_implicitly(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    task_toml = task_dir / 'task.toml'
    task_toml.write_text(
        task_toml.read_text().replace(
            '\n[verifier]\n',
            '\n[environment.boot]\ncommand = "/app/start.sh"\n\n[verifier]\n',
        )
    )
    _configure_generator_env(monkeypatch, task_dir)

    item = next(generator.generate_configs())

    assert item.config.task_config['boot_command'] == '/app/start.sh'
    assert item.config.runtime_configs['main']['enable_dind'] is False


def test_task_must_enable_dind_explicitly(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    task_toml = task_dir / 'task.toml'
    task_toml.write_text(
        task_toml.read_text().replace(
            'network_mode = "public"\n',
            'network_mode = "public"\nenable_dind = true\n',
            1,
        )
    )
    _configure_generator_env(monkeypatch, task_dir)

    item = next(generator.generate_configs())

    assert item.config.runtime_configs['main']['enable_dind'] is True


def test_harbor_enable_dind_env_explicitly_overrides_task(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    task_toml = task_dir / 'task.toml'
    task_toml.write_text(
        task_toml.read_text().replace(
            'network_mode = "public"\n',
            'network_mode = "public"\nenable_dind = true\n',
            1,
        )
    )
    _configure_generator_env(monkeypatch, task_dir)
    monkeypatch.setenv('HARBOR_ENABLE_DIND', 'false')

    item = next(generator.generate_configs())

    assert item.config.runtime_configs['main']['enable_dind'] is False


def test_generator_rejects_invalid_harbor_enable_dind_env(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    _configure_generator_env(monkeypatch, task_dir)
    monkeypatch.setenv('HARBOR_ENABLE_DIND', 'automatic')

    with pytest.raises(ValueError, match='HARBOR_ENABLE_DIND must be a boolean'):
        next(generator.generate_configs())


def test_generator_rejects_non_boolean_task_enable_dind(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    task_toml = task_dir / 'task.toml'
    task_toml.write_text(
        task_toml.read_text().replace(
            'network_mode = "public"\n',
            'network_mode = "public"\nenable_dind = "yes"\n',
            1,
        )
    )
    _configure_generator_env(monkeypatch, task_dir)

    with pytest.raises(ValueError, match='environment.enable_dind must be a boolean'):
        next(generator.generate_configs())


def test_generator_passes_explicit_gongfeng_purpose(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    _configure_generator_env(monkeypatch, task_dir)
    monkeypatch.setenv('GONGFENG_PURPOSE', 'prod')

    item = next(generator.generate_configs())

    assert item.config.runtime_provider_config['purpose'] == 'prod'


def test_generator_passes_gongfeng_startup_jitter_max(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    _configure_generator_env(monkeypatch, task_dir)
    monkeypatch.setenv('GONGFENG_STARTUP_JITTER_MAX', '120.5')

    item = next(generator.generate_configs())

    assert item.config.runtime_provider_config['startup_jitter_max'] == 120.5


def test_generator_rejects_negative_gongfeng_startup_jitter_max(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    _configure_generator_env(monkeypatch, task_dir)
    monkeypatch.setenv('GONGFENG_STARTUP_JITTER_MAX', '-1')

    with pytest.raises(ValueError, match='GONGFENG_STARTUP_JITTER_MAX must be >= 0'):
        next(generator.generate_configs())


def test_generator_rejects_invalid_gongfeng_purpose(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    _configure_generator_env(monkeypatch, task_dir)
    monkeypatch.setenv('GONGFENG_PURPOSE', 'invalid')

    with pytest.raises(ValueError, match='GONGFENG_PURPOSE must be prod, eval, or test'):
        next(generator.generate_configs())


def test_append_auth_query_preserves_and_replaces_routing_fields():
    updated = generator._append_auth_query(
        'secret?provider=aws_third&cache_task_id=old',
        cache_task_id='new cache',
    )

    base, _, raw_query = updated.partition('?')
    assert base == 'secret'
    assert parse_qs(raw_query) == {
        'provider': ['aws_third'],
        'cache_task_id': ['new cache'],
    }


def test_generator_enables_task_scoped_litellm_prompt_cache(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    _configure_generator_env(monkeypatch, task_dir)
    monkeypatch.setenv('OPENAI_CLIENT_NAME', 'litellm')
    monkeypatch.setenv('OPENAI_API_KEY', 'secret?provider=aws_third')

    item = next(generator.generate_configs())

    client_args = item.config.openai_clients['main']['config']['client_args']
    _, _, raw_query = client_args['api_key'].partition('?')
    cache_task_id = parse_qs(raw_query)['cache_task_id'][0]
    assert cache_task_id == item.metadata['cache_task_id']
    assert cache_task_id == generator._make_cache_task_id(item.run_id)
    assert client_args['cache_control_enabled'] is True
    assert client_args['cache_control_strategy'] == 'claude_code'
    assert client_args['log_llm_usage'] is True
    assert client_args['max_retries'] == 3


def test_generator_can_disable_litellm_prompt_cache(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    _configure_generator_env(monkeypatch, task_dir)
    monkeypatch.setenv('OPENAI_CLIENT_NAME', 'litellm')
    monkeypatch.setenv('OPENAI_API_KEY', 'secret?provider=aws_third')
    monkeypatch.setenv('OPENAI_CACHE_CONTROL_ENABLED', 'false')

    item = next(generator.generate_configs())

    client_args = item.config.openai_clients['main']['config']['client_args']
    assert client_args['api_key'] == 'secret?provider=aws_third'
    assert 'cache_control_enabled' not in client_args
    assert 'cache_control_strategy' not in client_args
    assert 'cache_task_id' not in item.metadata


def test_opus48_request_args_match_factory_thinking_preset(monkeypatch):
    monkeypatch.delenv('OPENAI_MAX_TOKENS', raising=False)
    monkeypatch.delenv('OPENAI_DISABLE_THINKING', raising=False)
    monkeypatch.delenv('OPENAI_THINKING_TYPE', raising=False)
    monkeypatch.delenv('OPENAI_THINKING_DISPLAY', raising=False)
    monkeypatch.delenv('OPENAI_THINKING_EFFORT', raising=False)

    request_args = generator._build_request_args('bedrock/anthropic.claude-opus-4-8', None)

    assert request_args == {
        'model': 'bedrock/anthropic.claude-opus-4-8',
        'max_tokens': 128000,
        'thinking': {'type': 'adaptive', 'display': 'summarized'},
        'output_config': {'effort': 'xhigh'},
    }


def test_opus48_request_args_can_disable_thinking_and_override_max_tokens(monkeypatch):
    monkeypatch.setenv('OPENAI_DISABLE_THINKING', 'true')
    monkeypatch.setenv('OPENAI_MAX_TOKENS', '64000')

    request_args = generator._build_request_args('bedrock/anthropic.claude-opus-4-8', None)

    assert request_args == {
        'model': 'bedrock/anthropic.claude-opus-4-8',
        'max_tokens': 64000,
        'thinking': {'type': 'disabled'},
    }


def test_generator_keeps_agent_config_generic_for_opus48(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    _configure_generator_env(monkeypatch, task_dir)
    monkeypatch.setenv('AGENT_NAME', 'claude_code')
    monkeypatch.setenv('OPENAI_CLIENT_NAME', 'litellm')
    monkeypatch.setenv('OPENAI_MODEL', 'bedrock/anthropic.claude-opus-4-8')

    item = next(generator.generate_configs())

    assert item.config.openai_clients['main']['config']['request_args'] == {
        'model': 'bedrock/anthropic.claude-opus-4-8',
        'max_tokens': 128000,
        'thinking': {'type': 'adaptive', 'display': 'summarized'},
        'output_config': {'effort': 'xhigh'},
    }
    assert item.config.openai_clients['main']['config']['enforced_request_args'] == {
        'max_tokens': 128000,
    }
    assert item.config.agent_config == {
        'max_iterations': 200,
        'working_dir': '/app',
        'timeout': 900.0,
    }


def test_generator_does_not_enforce_claude_budget_for_other_agents(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    _configure_generator_env(monkeypatch, task_dir)
    monkeypatch.setenv('AGENT_NAME', 'terminus2')
    monkeypatch.setenv('OPENAI_CLIENT_NAME', 'litellm')
    monkeypatch.setenv('OPENAI_MODEL', 'bedrock/anthropic.claude-opus-4-8')

    item = next(generator.generate_configs())

    assert 'enforced_request_args' not in item.config.openai_clients['main']['config']


@pytest.mark.parametrize('agent_name', ['main_agent', 'demo_agent'])
def test_generator_uses_registry_names_without_deriving_agent_imports(monkeypatch, tmp_path, agent_name):
    task_dir = _write_task(tmp_path, separate=False)
    _configure_generator_env(monkeypatch, task_dir)
    monkeypatch.setenv('AGENT_NAME', agent_name)

    item = next(generator.generate_configs())

    assert item.config.agent_name == agent_name
    assert item.config.imports == [
        'nexus.extensions.harbor_bridge.task',
        'nexus.extensions.harbor_bridge.noauth_client',
    ]


def test_generator_merges_generic_agent_config_override(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    _configure_generator_env(monkeypatch, task_dir)
    monkeypatch.setenv('AGENT_CONFIG_JSON', '{"parser_name":"xml","max_iterations":12}')

    item = next(generator.generate_configs())

    assert item.config.agent_config == {
        'max_iterations': 12,
        'working_dir': '/app',
        'timeout': 900.0,
        'parser_name': 'xml',
    }


def test_generator_requires_explicit_workdir(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    task_toml = task_dir / 'task.toml'
    task_toml.write_text(task_toml.read_text().replace('workdir = "/app"\n', ''))
    _configure_generator_env(monkeypatch, task_dir)

    with pytest.raises(ValueError, match=r'requires an explicit \[environment\]\.workdir'):
        next(generator.generate_configs())


def test_generator_rejects_shared_verifier_network_switch(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    task_toml = task_dir / 'task.toml'
    task_toml.write_text(
        task_toml.read_text().replace(
            'environment_mode = "shared"\ntimeout_sec = 120',
            'environment_mode = "shared"\ntimeout_sec = 120\nnetwork_mode = "no-network"',
        )
    )
    _configure_generator_env(monkeypatch, task_dir)

    with pytest.raises(ValueError, match='cannot change network_policy'):
        next(generator.generate_configs())


def test_generator_separate_mode_without_environment_reuses_image_in_new_runtime(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, explicit_verifier_environment=False)
    _configure_generator_env(monkeypatch, task_dir)

    item = next(generator.generate_configs())

    assert item.config.runtime_configs['main']['image'] == 'agent:v1'
    assert item.config.runtime_configs['verifier']['image'] == 'agent:v1'
    assert item.config.runtime_configs['main'] is not item.config.runtime_configs['verifier']


def test_generator_uses_prebaked_tests_when_host_script_is_absent(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path)
    (task_dir / 'tests' / 'test.sh').unlink()
    _configure_generator_env(monkeypatch, task_dir)

    item = next(generator.generate_configs())

    assert item.config.task_config['skip_tests_upload'] is True


def test_generator_wires_separate_verifier_boot_config(monkeypatch, tmp_path):
    from harbor.models.task.task import Task

    task_dir = _write_task(tmp_path)
    task_toml = task_dir / 'task.toml'
    task_toml.write_text(
        task_toml.read_text()
        + """

[verifier.environment.boot]
command = "/verify/start.sh"

[verifier.environment.healthcheck]
command = "test -f /tmp/verifier-ready"
retries = 10
interval_sec = 1
timeout_sec = 2
"""
    )
    _configure_generator_env(monkeypatch, task_dir)

    item = next(generator.generate_configs())

    assert item.config.task_config['verifier_boot_command'] == '/verify/start.sh'
    assert not any(key.startswith('verifier_healthcheck_') for key in item.config.task_config)
    healthcheck = Task(task_dir).config.verifier.environment.healthcheck
    assert healthcheck is not None
    assert healthcheck.command == 'test -f /tmp/verifier-ready'
    assert healthcheck.retries == 10


def test_generator_wires_separate_verifier_healthcheck_without_boot(monkeypatch, tmp_path):
    from harbor.models.task.task import Task

    task_dir = _write_task(tmp_path)
    task_toml = task_dir / 'task.toml'
    task_toml.write_text(
        task_toml.read_text()
        + """

[verifier.environment.healthcheck]
command = "test -f /tmp/verifier-ready"
"""
    )
    _configure_generator_env(monkeypatch, task_dir)

    item = next(generator.generate_configs())

    assert 'verifier_boot_command' not in item.config.task_config
    assert not any(key.startswith('verifier_healthcheck_') for key in item.config.task_config)
    healthcheck = Task(task_dir).config.verifier.environment.healthcheck
    assert healthcheck is not None
    assert healthcheck.command == 'test -f /tmp/verifier-ready'
    assert healthcheck.retries == 3
    assert healthcheck.interval_sec == 5.0
    assert healthcheck.timeout_sec == 30.0


def test_generator_rejects_sidecar_artifacts(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, artifact_service='database')
    _configure_generator_env(monkeypatch, task_dir)

    with pytest.raises(ValueError, match='artifact service=.*not supported'):
        next(generator.generate_configs())


def test_generator_rejects_shared_non_compose_sidecar_artifacts(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False, artifact_service='database')
    _configure_generator_env(monkeypatch, task_dir)

    with pytest.raises(ValueError, match='artifact service=.*not supported'):
        next(generator.generate_configs())


def test_generator_rejects_shared_non_compose_sidecar_collect(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False, collect_service='database')
    _configure_generator_env(monkeypatch, task_dir)

    with pytest.raises(ValueError, match='collect service=.*not supported'):
        next(generator.generate_configs())


@pytest.mark.asyncio
async def test_collect_hooks_reject_non_compose_sidecar_at_runtime() -> None:
    environment = SimpleNamespace(capabilities=SimpleNamespace(docker_compose=False))
    hook = SimpleNamespace(service='database', command='dump-data', timeout_sec=30, user=None)
    harbor_task = SimpleNamespace(config=SimpleNamespace(verifier=SimpleNamespace(collect=[hook])))

    with pytest.raises(ValueError, match='requires native Compose'):
        await task_module._run_collect_hooks(environment, harbor_task)


def test_resource_spec_rounds_memory_up():
    from harbor.models.task.config import EnvironmentConfig

    spec = compiler._build_resource_spec(
        EnvironmentConfig(cpus=2, memory_mb=2500),
        fallback_cpu=1,
        fallback_memory_gb=2,
        role='verifier',
    )

    assert spec == {'cpu': 2, 'memory': 3}


def test_command_failure_guard_rejects_nonzero_and_missing_results():
    with pytest.raises(RuntimeError, match='Prepare failed with code 1'):
        task_module._raise_for_command_failure(
            SimpleNamespace(return_code=1, stdout='', stderr='permission denied'),
            'Prepare',
        )
    with pytest.raises(RuntimeError, match='Prepare failed with code None'):
        task_module._raise_for_command_failure(None, 'Prepare')


@pytest.mark.asyncio
async def test_git_safe_directory_is_scoped_to_task_workdir():
    events: list[str] = []
    runtime = _FakeRuntime('main', events)

    await task_module._configure_git_safe_directory(runtime, '/app')

    assert 'git config --global --add safe.directory /app' in runtime.commands[-1]
    assert 'safe.directory=*' not in runtime.commands[-1]


def test_static_allowlist_network_policy_maps_to_gongfeng(monkeypatch):
    from harbor.models.task.config import AgentConfig, EnvironmentConfig, NetworkMode

    monkeypatch.delenv('HARBOR_OFFLINE_ALLOWLIST', raising=False)
    policy = compiler._build_static_network_policy(
        EnvironmentConfig(
            network_mode=NetworkMode.ALLOWLIST,
            allowed_hosts=['judge.woa.com', '*.example.com'],
        ),
        AgentConfig(),
        role='verifier',
    )

    assert policy == {
        'default_action': 'deny',
        'egress': [
            {'action': 'allow', 'target': 'judge.woa.com'},
            {'action': 'allow', 'target': '*.example.com'},
        ],
    }


class _FakeRuntime:
    def __init__(self, name: str, events: list[str]):
        self.name = name
        self.events = events
        self.commands: list[str] = []
        self.uploads: list[tuple[list[Any], bool]] = []

    async def run_command(self, *, command: str, timeout: float):
        self.commands.append(command)
        self.events.append(f'command:{self.name}')
        stdout = '/app\n' if command == 'pwd' else ''
        return SimpleNamespace(return_code=0, stdout=stdout, stderr='')

    async def upload_file(self, files: list[Any], overwrite: bool = False):
        self.uploads.append((files, overwrite))
        for item in files:
            self.events.append(f'upload:{self.name}:{item.path}')
        return SimpleNamespace(results=[SimpleNamespace(path=item.path, success=True, message='') for item in files])

    async def drain_trace(self, timeout: float | None = None):
        self.events.append(f'drain-trace:{self.name}:{timeout}')


class _FakeContext:
    def __init__(self, events: list[str]):
        self.events = events
        self.runtime_provider = object()
        self.task_config: dict[str, Any] = {}
        self.tool_state: dict[str, Any] = {}
        self.runtimes = {
            'main': _FakeRuntime('main', events),
            'verifier': _FakeRuntime('verifier', events),
        }

    async def get_runtime(self, name: str = 'main'):
        self.events.append(f'acquire:{name}')
        return self.runtimes[name]

    def get_runtime_id(self, name: str = 'main'):
        return f'runtime-{name}'

    async def release_runtime(self, name: str = 'main', *, force: bool = False):
        self.events.append(f'release-force:{name}:{force}')
        self.events.append(f'release:{name}')
        return True


class _FakeEnvironment:
    def __init__(self, *, runtime: _FakeRuntime, task_env_config: Any, **_: Any):
        self.runtime = runtime
        self.os = task_env_config.os
        self.events = runtime.events

    async def start(self, force_build: bool = False):
        self.events.append(f'start:{self.runtime.name}')

    async def run_healthcheck(self):
        self.events.append(f'healthcheck:{self.runtime.name}')

    async def stop(self, delete: bool = True):
        self.events.append(f'stop:{self.runtime.name}')

    async def empty_dirs(self, *_: Any, **__: Any):
        self.events.append(f'empty:{self.runtime.name}')
        return SimpleNamespace(return_code=0, stdout='', stderr='')

    async def ensure_dirs(self, *_: Any, **__: Any):
        self.events.append(f'ensure:{self.runtime.name}')
        return SimpleNamespace(return_code=0, stdout='', stderr='')

    async def exec(self, command: str, *, timeout_sec: int | None = None, **__: Any):
        if 'T3-Harbor workdir does not exist' in command:
            return await self.runtime.run_command(command=command, timeout=float(timeout_sec or 60))
        self.events.append(f'collect:{self.runtime.name}')
        return SimpleNamespace(return_code=0, stdout='', stderr='')


class _FakeArtifactHandler:
    def __init__(self, events: list[str]):
        self.events = events

    async def upload_artifacts(self, environment: _FakeEnvironment, *_: Any, **__: Any):
        self.events.append(f'upload-artifacts:{environment.runtime.name}')


@pytest.mark.asyncio
async def test_collect_hook_nonzero_exit_fails_case():
    environment = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=7, stdout='', stderr='collect failed'))
    )
    hook = SimpleNamespace(service='main', command='collect-results', timeout_sec=30, user='root')
    harbor_task = SimpleNamespace(config=SimpleNamespace(verifier=SimpleNamespace(collect=[hook])))

    with pytest.raises(RuntimeError, match='collect failed'):
        await task_module._run_collect_hooks(environment, harbor_task)


@pytest.mark.asyncio
async def test_collect_hook_execution_error_fails_case():
    environment = SimpleNamespace(exec=AsyncMock(side_effect=TimeoutError('collect timeout')))
    hook = SimpleNamespace(service='main', command='collect-results', timeout_sec=30, user='root')
    harbor_task = SimpleNamespace(config=SimpleNamespace(verifier=SimpleNamespace(collect=[hook])))

    with pytest.raises(RuntimeError, match='collect timeout'):
        await task_module._run_collect_hooks(environment, harbor_task)


@pytest.mark.asyncio
async def test_task_requires_workdir_to_exist_in_image(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    events: list[str] = []
    context = _FakeContext(events)
    commands: list[str] = []

    async def run_command(*, command: str, timeout: float):
        commands.append(command)
        return SimpleNamespace(return_code=1, stdout='', stderr='missing workdir')

    context.runtimes['main'].run_command = run_command
    monkeypatch.setattr(task_module, 'require_context', lambda: context)

    with pytest.raises(RuntimeError, match='Prepare Harbor runtime directories'):
        await task_module.harbor_bridge_task(
            task_module.HarborBridgeTaskInput(
                task_dir=str(task_dir),
                workdir='/appp',
                verifier_mode='shared',
                run_agent=AsyncMock(),
            )
        )

    assert len(commands) == 1
    assert '[ ! -d /appp ]' in commands[0]
    assert 'mkdir -p /appp' not in commands[0]


@pytest.mark.asyncio
async def test_task_runs_separate_verifier_in_second_runtime(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path)
    events: list[str] = []
    context = _FakeContext(events)
    span_updates: list[dict[str, Any]] = []
    phase_calls: list[str] = []
    span_lifecycle: list[str] = []

    def record_phase(name: str | None = None, **_: Any):
        if name is not None:
            phase_calls.append(name)
            span_lifecycle.append(name)
        return nullcontext()

    def get_task_span_id() -> str:
        span_lifecycle.append('get_task_span_id')
        return 'task-span'

    async def run_agent(**_: Any):
        events.append('agent')
        return SimpleNamespace(
            exit_status='finished',
            final_answer='separate answer',
            main_trajectory=[{'role': 'assistant', 'content': 'separate answer'}],
        )

    async def collect_artifacts(*_: Any, **__: Any):
        events.append('collect-artifacts')
        return (
            _FakeArtifactHandler(events),
            SimpleNamespace(artifacts_dir=PurePosixPath('/logs/artifacts')),
        )

    async def run_verifier(*, environment: _FakeEnvironment, **_: Any):
        assert _['needs_agent_trajectory'] is True
        events.append(f'verify:{environment.runtime.name}')
        return {'reward': 0.5}, {}, 'passed'

    import nexus.extensions.harbor_bridge.environment as environment_module

    monkeypatch.setattr(task_module, 'require_context', lambda: context)
    monkeypatch.setattr(task_module, 'get_current_span_id', get_task_span_id)
    monkeypatch.setattr(
        task_module,
        'update_span',
        lambda _span_id, *, attributes: span_updates.append(attributes),
    )
    monkeypatch.setattr(task_module, '_collect_artifacts', collect_artifacts)
    monkeypatch.setattr(task_module, '_run_harbor_verifier', run_verifier)
    monkeypatch.setattr(task_module, 'phase', record_phase)
    monkeypatch.setattr(environment_module, 'GongfengEnvironment', _FakeEnvironment)

    output = await task_module.harbor_bridge_task(
        task_module.HarborBridgeTaskInput(
            task_dir=str(task_dir),
            workdir='/app',
            verifier_mode='separate',
            needs_agent_trajectory=True,
            run_agent=run_agent,
        )
    )

    assert output.score == 0.5
    assert len(span_updates) == 1
    assert span_updates[0]['hytuner.runtime_id'] == 'runtime-main'
    assert span_updates[0]['hytuner.eval_runtime_id'] == 'runtime-verifier'
    assert span_updates[0]['hytuner.termination_reason'] == 'ENV_DONE'
    assert span_updates[0]['hytuner.reward_time'] >= 0
    # Optional subphases do not affect the expected top-level phase order.
    assert [name for name in phase_calls if name.count('.') == 1] == [
        'task.init',
        'task.agent',
        'task.handoff',
        'task.scoring',
        'task.cleanup',
    ]
    assert span_lifecycle[:2] == ['get_task_span_id', 'task.init']
    assert events.index('start:main') < events.index('healthcheck:main') < events.index('agent')
    assert events.index('agent') < events.index('collect-artifacts')
    assert events.index('collect-artifacts') < events.index('release:main')
    assert events.index('release:main') < events.index('acquire:verifier')
    assert events.index('start:verifier') < events.index('healthcheck:verifier')
    assert events.index('healthcheck:verifier') < events.index('upload:verifier:/logs/agent/result.json')
    assert events.index('upload:verifier:/logs/agent/result.json') < events.index(
        'upload:verifier:/logs/agent/trajectory.json'
    )
    assert events.index('upload:verifier:/logs/agent/trajectory.json') < events.index('upload-artifacts:verifier')
    assert events.index('healthcheck:verifier') < events.index('upload-artifacts:verifier')
    assert events.index('acquire:verifier') < events.index('upload-artifacts:verifier')
    assert events.index('upload-artifacts:verifier') < events.index('verify:verifier')
    assert events.index('verify:verifier') < events.index('release:verifier')
    assert any(
        'for dir in /logs/agent /logs/verifier /logs/artifacts /tests' in command
        and '"$find_bin" "$dir" -mindepth 1 -maxdepth 1' in command
        and 'command -v "$name"' in command
        for command in context.runtimes['main'].commands
    )
    assert not any('rm -rf /tests' in command for command in context.runtimes['main'].commands)
    assert any(
        '"$chmod_bin" 0777 /logs/agent /logs/verifier /logs/artifacts' in command
        for command in context.runtimes['main'].commands
    )
    assert not any(
        '"$chmod_bin" 0777 /logs/agent /logs/verifier /logs/artifacts /tests' in command
        for command in context.runtimes['main'].commands
    )
    assert not any(event.startswith('upload:main:/logs/agent/result.json') for event in events)
    (result_files, result_overwrite) = context.runtimes['verifier'].uploads[0]
    (trajectory_files, trajectory_overwrite) = context.runtimes['verifier'].uploads[1]
    assert result_overwrite is True
    assert trajectory_overwrite is True
    assert json.loads(result_files[0].content)['final_answer'] == 'separate answer'
    assert json.loads(trajectory_files[0].content)['messages'] == [{'role': 'assistant', 'content': 'separate answer'}]
    assert 'release-force:main:True' in events


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('verifier_mode', 'handoff_subphases'),
    [
        (
            'separate',
            [
                'task.handoff.artifact_collect_time',
                'task.handoff.artifact_collect_time',
                'task.handoff.release_main_time',
                'task.handoff.sandbox_init_time',
                'task.handoff.env_setup_time',
            ],
        ),
        (
            'shared',
            ['task.handoff.artifact_collect_time', 'task.handoff.env_setup_time', 'task.handoff.env_setup_time'],
        ),
    ],
)
async def test_task_opens_handoff_and_scoring_subphases_in_order(
    monkeypatch, tmp_path, verifier_mode, handoff_subphases
):
    task_dir = _write_task(tmp_path, separate=verifier_mode == 'separate')
    events: list[str] = []
    context = _FakeContext(events)
    phase_calls: list[str] = []

    def record_phase(name: str | None = None, **_: Any):
        if name is not None:
            phase_calls.append(name)
        return nullcontext()

    async def run_agent(**_: Any):
        return SimpleNamespace(exit_status='finished', final_answer='done', main_trajectory=[])

    async def collect_artifacts(*_: Any, **__: Any):
        return (
            _FakeArtifactHandler(events),
            SimpleNamespace(artifacts_dir=PurePosixPath('/logs/artifacts')),
        )

    async def run_verifier(**_: Any):
        return {'reward': 1.0}, {}, ''

    import nexus.extensions.harbor_bridge.environment as environment_module

    monkeypatch.setattr(task_module, 'require_context', lambda: context)
    monkeypatch.setattr(task_module, 'get_current_span_id', lambda: 'task-span')
    monkeypatch.setattr(task_module, 'update_span', lambda _span_id, *, attributes: None)
    monkeypatch.setattr(task_module, '_collect_artifacts', collect_artifacts)
    monkeypatch.setattr(task_module, '_run_harbor_verifier', run_verifier)
    monkeypatch.setattr(task_module, 'phase', record_phase)
    monkeypatch.setattr(environment_module, 'GongfengEnvironment', _FakeEnvironment)

    await task_module.harbor_bridge_task(
        task_module.HarborBridgeTaskInput(
            task_dir=str(task_dir),
            workdir='/app',
            verifier_mode=verifier_mode,
            run_agent=run_agent,
        )
    )

    assert [name for name in phase_calls if name.startswith(('task.handoff', 'task.scoring'))] == [
        'task.handoff',
        *handoff_subphases,
        'task.scoring',
        'task.scoring.check_score_time',
    ]


@pytest.mark.asyncio
async def test_task_rejects_stale_serialized_verifier_mode_before_runtime(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path)
    events: list[str] = []
    context = _FakeContext(events)
    task_scopes: list[Path] = []

    async def materialize_task_files(*_: Any, destination: Path, **__: Any) -> Path:
        destination.mkdir(parents=True)
        task_scopes.append(destination.parent)
        return task_dir

    async def run_agent(**_: Any):
        return SimpleNamespace()

    monkeypatch.setattr(task_module, 'require_context', lambda: context)
    monkeypatch.setattr(task_module, '_materialize_task_directory', materialize_task_files)

    with pytest.raises(ValueError, match='regenerate the RunInput'):
        await task_module.harbor_bridge_task(
            task_module.HarborBridgeTaskInput(
                task_dir=str(task_dir),
                workdir='/app',
                verifier_mode='shared',
                run_agent=run_agent,
            )
        )
    assert events == []
    assert task_scopes
    assert not task_scopes[0].exists()


@pytest.mark.asyncio
async def test_task_rejects_browser_with_separate_verifier_before_runtime(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=True)
    events: list[str] = []
    context = _FakeContext(events)
    monkeypatch.setattr(task_module, 'require_context', lambda: context)

    with pytest.raises(ValueError, match='browser tasks require a shared Verifier'):
        await task_module.harbor_bridge_task(
            task_module.HarborBridgeTaskInput(
                task_dir=str(task_dir),
                workdir='/app',
                verifier_mode='separate',
                browser={'start_url': 'https://example.com/task'},
                run_agent=AsyncMock(),
            )
        )

    assert events == []


@pytest.mark.parametrize(
    ('result', 'expected_reason'),
    [
        (task_module.T3AgentResult(mode='agent', status='success'), 'ENV_DONE'),
        (task_module.T3AgentResult(mode='agent', status='timeout'), 'TASK_TIMEOUT'),
        (
            task_module.T3AgentResult(mode='agent', status='incomplete', agent_exit_status='max_iterations'),
            'MAX_STEPS',
        ),
        (
            task_module.T3AgentResult(mode='agent', status='incomplete', agent_exit_status='truncation'),
            'TRUNCATION',
        ),
        (
            task_module.T3AgentResult(mode='agent', status='error', agent_exit_status='tool_error'),
            'FORMAT_ERROR',
        ),
        (
            task_module.T3AgentResult(
                mode='agent',
                status='error',
                failure_kind='infrastructure',
                failure_stage='proxy',
            ),
            'INITIALIZATION_ERROR',
        ),
        (task_module.T3AgentResult(mode='agent', status='error'), 'UNHANDLED_ERROR'),
    ],
)
def test_agent_result_maps_to_task_termination_span(result, expected_reason):
    assert task_module.agent_termination_reason(result) == expected_reason


@pytest.mark.asyncio
async def test_shared_task_runs_collect_hook_before_verifier(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    events: list[str] = []
    context = _FakeContext(events)
    captured_agent_kwargs: dict[str, Any] = {}
    span_updates: list[dict[str, Any]] = []

    async def run_agent(**kwargs: Any):
        captured_agent_kwargs.update(kwargs)
        events.append('agent')
        return SimpleNamespace(final_answer='shared answer', exit_status='finished')

    async def run_verifier(*, environment: _FakeEnvironment, **_: Any):
        assert _['needs_agent_trajectory'] is True
        events.append(f'verify:{environment.runtime.name}')
        return {'reward': 1.0}, {}, 'passed'

    import nexus.extensions.harbor_bridge.environment as environment_module

    monkeypatch.setattr(task_module, 'require_context', lambda: context)
    monkeypatch.setattr(task_module, 'get_current_span_id', lambda: 'task-span')
    monkeypatch.setattr(
        task_module,
        'update_span',
        lambda _span_id, *, attributes: span_updates.append(attributes),
    )
    monkeypatch.setattr(
        task_module,
        'get_processor',
        lambda _processor_type: SimpleNamespace(
            snapshot_result=lambda: {
                'trajectories': [
                    {
                        'messages': [{'role': 'assistant', 'content': 'shared answer from tracker'}],
                        'tools': [],
                    }
                ]
            }
        ),
    )
    monkeypatch.setattr(task_module, '_run_harbor_verifier', run_verifier)
    monkeypatch.setattr(environment_module, 'GongfengEnvironment', _FakeEnvironment)

    output = await task_module.harbor_bridge_task(
        task_module.HarborBridgeTaskInput(
            task_dir=str(task_dir),
            workdir='/app',
            verifier_mode='shared',
            needs_agent_trajectory=True,
            run_agent=run_agent,
        )
    )

    assert output.score == 1.0
    assert output.extra_fields == {}
    assert context.tool_state['working_dir'] == '/app'
    assert len(span_updates) == 1
    assert span_updates[0]['hytuner.runtime_id'] == 'runtime-main'
    assert span_updates[0]['hytuner.termination_reason'] == 'ENV_DONE'
    assert span_updates[0]['hytuner.reward_time'] >= 0
    assert 'hytuner.observation_time' not in span_updates[0]
    assert events.index('agent') < events.index('collect:main')
    assert events.index('agent') < events.index('drain-trace:main:5.0') < events.index('collect:main')
    assert events.index('collect:main') < events.index('upload:main:/logs/agent/result.json')
    assert events.index('upload:main:/logs/agent/result.json') < events.index('upload:main:/logs/agent/trajectory.json')
    assert events.index('upload:main:/logs/agent/trajectory.json') < events.index('verify:main')
    assert events.index('collect:main') < events.index('verify:main')
    assert 'acquire:verifier' not in events
    (result_files, result_overwrite) = context.runtimes['main'].uploads[0]
    (trajectory_files, trajectory_overwrite) = context.runtimes['main'].uploads[1]
    assert result_overwrite is True
    assert trajectory_overwrite is True
    assert json.loads(result_files[0].content)['final_answer'] == 'shared answer'
    assert json.loads(trajectory_files[0].content)['messages'] == [
        {'role': 'assistant', 'content': 'shared answer from tracker'}
    ]
    assert captured_agent_kwargs == {
        'user_prompt': 'Complete the task.\n',
        'working_dir': '/app',
        'timeout': 600.0,
        'mcp_servers': [],
    }


@pytest.mark.asyncio
async def test_shared_browser_session_is_passed_from_agent_to_verifier(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    events: list[str] = []
    context = _FakeContext(events)
    browser = {
        'start_url': 'https://example.com/task',
        'headless': True,
        'viewport_width': 1280,
        'viewport_height': 720,
        'nav_timeout_ms': 10000,
        'runtime_name': 'main',
    }

    async def run_agent(**kwargs: Any):
        assert kwargs['browser'] == browser
        context.tool_state['browser_session_id'] = 'browser-1'
        return SimpleNamespace(final_answer='done')

    async def run_verifier(*, environment: _FakeEnvironment, **kwargs: Any):
        assert environment.runtime.name == 'main'
        assert kwargs['browser'] == browser
        assert kwargs['tool_state'] == {'browser_session_id': 'browser-1', 'working_dir': '/app'}
        return {'reward': 1.0}, {}, 'passed'

    import nexus.extensions.harbor_bridge.environment as environment_module

    monkeypatch.setattr(task_module, 'require_context', lambda: context)
    monkeypatch.setattr(task_module, '_run_harbor_verifier', run_verifier)
    monkeypatch.setattr(environment_module, 'GongfengEnvironment', _FakeEnvironment)

    output = await task_module.harbor_bridge_task(
        task_module.HarborBridgeTaskInput(
            task_dir=str(task_dir),
            workdir='/app',
            verifier_mode='shared',
            browser=browser,
            run_agent=run_agent,
        )
    )

    assert output.reward == 1.0


@pytest.mark.asyncio
async def test_agent_process_failure_still_runs_verifier(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    events: list[str] = []
    context = _FakeContext(events)

    async def run_agent(**_: Any):
        events.append('agent')
        return SimpleNamespace(success=False, exit_code=1, stderr='agent process failed')

    async def run_verifier(*, environment: _FakeEnvironment, **_: Any):
        events.append(f'verify:{environment.runtime.name}')
        return {'reward': 0.0}, {}, 'failed'

    import nexus.extensions.harbor_bridge.environment as environment_module

    monkeypatch.setattr(task_module, 'require_context', lambda: context)
    monkeypatch.setattr(task_module, '_run_harbor_verifier', run_verifier)
    monkeypatch.setattr(environment_module, 'GongfengEnvironment', _FakeEnvironment)

    output = await task_module.harbor_bridge_task(
        task_module.HarborBridgeTaskInput(
            task_dir=str(task_dir),
            workdir='/app',
            verifier_mode='shared',
            run_agent=run_agent,
        )
    )

    assert output.score == 0.0
    assert 'verify:main' in events
    (uploaded_files, _) = context.runtimes['main'].uploads[0]
    agent_result = json.loads(uploaded_files[0].content)
    assert agent_result['status'] == 'error'
    assert agent_result['failure_kind'] is None


@pytest.mark.asyncio
async def test_agent_infrastructure_failure_skips_verifier(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    events: list[str] = []
    context = _FakeContext(events)
    run_verifier = AsyncMock()

    async def run_agent(**_: Any):
        return SimpleNamespace(
            success=False,
            exit_code=1,
            stderr='LLM authentication failed',
            failure_kind='infrastructure',
            failure_stage='llm',
        )

    import nexus.extensions.harbor_bridge.environment as environment_module

    monkeypatch.setattr(task_module, 'require_context', lambda: context)
    monkeypatch.setattr(task_module, '_run_harbor_verifier', run_verifier)
    monkeypatch.setattr(environment_module, 'GongfengEnvironment', _FakeEnvironment)

    with pytest.raises(task_module.AgentInfrastructureError, match='LLM authentication failed'):
        await task_module.harbor_bridge_task(
            task_module.HarborBridgeTaskInput(
                task_dir=str(task_dir),
                workdir='/app',
                verifier_mode='shared',
                run_agent=run_agent,
            )
        )

    run_verifier.assert_not_awaited()


@pytest.mark.asyncio
async def test_inference_error_exit_status_skips_verifier(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    events: list[str] = []
    context = _FakeContext(events)
    run_verifier = AsyncMock()

    async def run_agent(**_: Any):
        return SimpleNamespace(
            exit_status='inference_error',
            final_answer='Provider request failed after retries',
        )

    import nexus.extensions.harbor_bridge.environment as environment_module

    monkeypatch.setattr(task_module, 'require_context', lambda: context)
    monkeypatch.setattr(task_module, '_run_harbor_verifier', run_verifier)
    monkeypatch.setattr(environment_module, 'GongfengEnvironment', _FakeEnvironment)

    with pytest.raises(task_module.AgentInfrastructureError, match='inference_error'):
        await task_module.harbor_bridge_task(
            task_module.HarborBridgeTaskInput(
                task_dir=str(task_dir),
                workdir='/app',
                verifier_mode='shared',
                run_agent=run_agent,
            )
        )

    run_verifier.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_exception_is_infrastructure_failure(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    events: list[str] = []
    context = _FakeContext(events)
    run_verifier = AsyncMock()

    async def run_agent(**_: Any):
        raise ConnectionError('runtime disconnected')

    import nexus.extensions.harbor_bridge.environment as environment_module

    monkeypatch.setattr(task_module, 'require_context', lambda: context)
    monkeypatch.setattr(task_module, '_run_harbor_verifier', run_verifier)
    monkeypatch.setattr(environment_module, 'GongfengEnvironment', _FakeEnvironment)

    with pytest.raises(task_module.AgentInfrastructureError, match='runtime disconnected'):
        await task_module.harbor_bridge_task(
            task_module.HarborBridgeTaskInput(
                task_dir=str(task_dir),
                workdir='/app',
                verifier_mode='shared',
                run_agent=run_agent,
            )
        )

    run_verifier.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('delivery', 'expected_prompt'),
    [
        (
            'instruction_prefix',
            '<IMPORTANT>\nDo not copy the answer.\n</IMPORTANT>\n\nComplete the task.\n',
        ),
        (
            'instruction_suffix',
            'Complete the task.\n\n<IMPORTANT>\nDo not copy the answer.\n</IMPORTANT>',
        ),
    ],
)
async def test_task_applies_anti_hack_policy_to_agent_instruction(
    monkeypatch,
    tmp_path,
    delivery,
    expected_prompt,
):
    task_dir = _write_task(tmp_path, separate=False)
    events: list[str] = []
    context = _FakeContext(events)
    captured_prompt = ''

    async def run_agent(**kwargs: Any):
        nonlocal captured_prompt
        captured_prompt = kwargs['user_prompt']
        events.append('agent')
        return SimpleNamespace()

    async def run_verifier(*, environment: _FakeEnvironment, **_: Any):
        events.append(f'verify:{environment.runtime.name}')
        return {'reward': 1.0}, {}, 'passed'

    import nexus.extensions.harbor_bridge.environment as environment_module

    monkeypatch.setattr(task_module, 'require_context', lambda: context)
    monkeypatch.setattr(task_module, '_run_harbor_verifier', run_verifier)
    monkeypatch.setattr(environment_module, 'GongfengEnvironment', _FakeEnvironment)

    await task_module.harbor_bridge_task(
        task_module.HarborBridgeTaskInput(
            task_dir=str(task_dir),
            workdir='/app',
            verifier_mode='shared',
            anti_hack_policy={
                'id': 'code-anti-hack-v1',
                'delivery': delivery,
                'prompt': '<IMPORTANT>\nDo not copy the answer.\n</IMPORTANT>',
            },
            run_agent=run_agent,
        )
    )

    assert captured_prompt == expected_prompt


@pytest.mark.asyncio
async def test_task_detector_only_policy_does_not_change_instruction(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    events: list[str] = []
    context = _FakeContext(events)
    captured_prompt = ''
    span_updates: list[dict[str, Any]] = []

    async def run_agent(**kwargs: Any):
        nonlocal captured_prompt
        captured_prompt = kwargs['user_prompt']
        events.append('agent')
        return SimpleNamespace(
            exit_status='finished',
            final_answer='done',
            main_trajectory=[{'role': 'assistant', 'content': 'done'}],
        )

    async def run_verifier(*, environment: _FakeEnvironment, **_: Any):
        events.append(f'verify:{environment.runtime.name}')
        return {'reward': 1.0}, {}, 'passed'

    import nexus.extensions.harbor_bridge.environment as environment_module

    monkeypatch.setattr(task_module, 'require_context', lambda: context)
    monkeypatch.setattr(task_module, 'get_current_span_id', lambda: 'task-span')
    monkeypatch.setattr(
        task_module,
        'update_span',
        lambda _span_id, *, attributes: span_updates.append(attributes),
    )
    monkeypatch.setattr(task_module, '_run_harbor_verifier', run_verifier)
    monkeypatch.setattr(environment_module, 'GongfengEnvironment', _FakeEnvironment)

    output = await task_module.harbor_bridge_task(
        task_module.HarborBridgeTaskInput(
            task_dir=str(task_dir),
            workdir='/app',
            verifier_mode='shared',
            anti_hack_policy={
                'id': 'code-anti-hack-v1',
                'delivery': 'none',
                'prompt': '',
                'trajectory_strategy': 'restricted-source-access-v1',
            },
            run_agent=run_agent,
        )
    )

    assert captured_prompt == 'Complete the task.\n'
    assert output.score == 1.0
    assert output.reward == 1.0
    assert output.extra_fields == {
        'is_hack': False,
        'anti_hack_status': 'completed',
    }
    assert span_updates[0]['hytuner.termination_reason'] == 'ENV_DONE'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('delivery', 'prompt'),
    [
        ('instruction_suffix', 'Do not copy the answer.'),
        ('none', ''),
    ],
)
async def test_task_reports_trajectory_strategy_hit_without_changing_verifier_result(
    monkeypatch,
    tmp_path,
    delivery,
    prompt,
):
    task_dir = _write_task(tmp_path, separate=False)
    with (task_dir / 'task.toml').open('a', encoding='utf-8') as task_file:
        task_file.write(
            '\n[other]\n'
            'repo = "owner/project"\n'
            'base_commit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n'
            'task_type = "SWE"\n'
        )
    events: list[str] = []
    context = _FakeContext(events)
    span_updates: list[dict[str, Any]] = []

    async def run_agent(**_: Any):
        events.append('agent')
        return SimpleNamespace(
            exit_status='finished',
            final_answer='done',
            main_trajectory=[
                {
                    'role': 'assistant',
                    'tool_calls': [
                        {
                            'function': {
                                'name': 'exec_command',
                                'arguments': {'cmd': ['bash', '-lc', 'git fetch origin pull/123/head']},
                            }
                        }
                    ],
                },
                {'role': 'tool', 'content': 'From github.com:owner/project'},
                {'role': 'assistant', 'content': 'done'},
            ],
        )

    async def run_verifier(*, environment: _FakeEnvironment, **_: Any):
        events.append(f'verify:{environment.runtime.name}')
        return {'reward': 1.0}, {}, 'passed'

    import nexus.extensions.harbor_bridge.environment as environment_module

    monkeypatch.setattr(task_module, 'require_context', lambda: context)
    monkeypatch.setattr(task_module, 'get_current_span_id', lambda: 'task-span')
    monkeypatch.setattr(
        task_module,
        'update_span',
        lambda _span_id, *, attributes: span_updates.append(attributes),
    )
    monkeypatch.setattr(task_module, '_run_harbor_verifier', run_verifier)
    monkeypatch.setattr(environment_module, 'GongfengEnvironment', _FakeEnvironment)

    output = await task_module.harbor_bridge_task(
        task_module.HarborBridgeTaskInput(
            task_dir=str(task_dir),
            instance_id='owner__project-123',
            workdir='/app',
            verifier_mode='shared',
            anti_hack_policy={
                'id': 'code-anti-hack-v1',
                'delivery': delivery,
                'prompt': prompt,
                'trajectory_strategy': 'restricted-source-access-v1',
            },
            run_agent=run_agent,
        )
    )

    assert output.score == 1.0
    assert output.reward == 1.0
    assert output.rewards['reward'] == 1.0
    assert output.extra_fields == {
        'is_hack': True,
        'anti_hack_status': 'completed',
    }
    assert span_updates[0]['hytuner.termination_reason'] == 'ENV_DONE'
    assert not any(event == 'upload:main:/logs/agent/trajectory.json' for event in events)


@pytest.mark.asyncio
async def test_task_rejects_unknown_trajectory_strategy_before_runtime(monkeypatch):
    events: list[str] = []
    context = _FakeContext(events)
    monkeypatch.setattr(task_module, 'require_context', lambda: context)

    with pytest.raises(ValueError, match='Unknown anti-hack trajectory strategy'):
        await task_module.harbor_bridge_task(
            task_module.HarborBridgeTaskInput(
                task_dir='unused',
                workdir='/app',
                verifier_mode='shared',
                anti_hack_policy={
                    'id': 'unknown-policy',
                    'prompt': 'Notice.',
                    'trajectory_strategy': 'unknown-strategy',
                },
                run_agent=AsyncMock(),
            )
        )

    assert events == []


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['oracle', 'nop'])
async def test_task_rejects_anti_hack_prompt_outside_agent_mode(monkeypatch, mode):
    span_updates: list[dict[str, Any]] = []
    monkeypatch.setattr(task_module, 'get_current_span_id', lambda: 'task-span')
    monkeypatch.setattr(
        task_module,
        'update_span',
        lambda _span_id, *, attributes: span_updates.append(attributes),
    )

    with pytest.raises(ValueError, match='anti_hack_policy is supported only for Agent runs'):
        await task_module.harbor_bridge_task(
            task_module.HarborBridgeTaskInput(
                task_dir='unused',
                workdir='/app',
                mode=mode,
                anti_hack_policy={
                    'id': 'code-anti-hack-v1',
                    'prompt': 'Do not copy the answer.',
                },
                run_agent=AsyncMock(),
            )
        )

    assert span_updates == [{'hytuner.termination_reason': 'INITIALIZATION_ERROR'}]


@pytest.mark.asyncio
async def test_oracle_transport_failure_is_classified_as_environment_error(monkeypatch, tmp_path):
    task_dir = _write_task(tmp_path, separate=False)
    events: list[str] = []
    context = _FakeContext(events)
    span_updates: list[dict[str, Any]] = []
    run_verifier = AsyncMock()

    async def fail_oracle(*_args, **_kwargs):
        raise RuntimeUnhealthyError('oracle runtime disconnected')

    import nexus.extensions.harbor_bridge.environment as environment_module

    monkeypatch.setattr(task_module, 'require_context', lambda: context)
    monkeypatch.setattr(task_module, 'get_current_span_id', lambda: 'task-span')
    monkeypatch.setattr(
        task_module,
        'update_span',
        lambda _span_id, *, attributes: span_updates.append(attributes),
    )
    monkeypatch.setattr(task_module, '_run_oracle', fail_oracle)
    monkeypatch.setattr(task_module, '_run_harbor_verifier', run_verifier)
    monkeypatch.setattr(environment_module, 'GongfengEnvironment', _FakeEnvironment)

    with pytest.raises(RuntimeUnhealthyError, match='oracle runtime disconnected'):
        await task_module.harbor_bridge_task(
            task_module.HarborBridgeTaskInput(
                task_dir=str(task_dir),
                workdir='/app',
                mode='oracle',
                verifier_mode='shared',
                run_agent=AsyncMock(),
            )
        )

    run_verifier.assert_not_awaited()
    assert any(
        '"$chmod_bin" 0777 /logs/agent /logs/verifier /logs/artifacts' in command
        for command in context.runtimes['main'].commands
    )
    assert span_updates == [
        {
            'hytuner.runtime_id': 'runtime-main',
            'hytuner.termination_reason': 'ENV_ERROR',
        }
    ]


class _ReleaseProvider(RuntimeProvider):
    def __init__(self, *, fail: bool = False, fail_all: bool = False):
        super().__init__()
        self.fail = fail
        self.fail_all = fail_all
        self.released: list[str] = []

    async def acquire(self, name: str, **params: Any):
        raise NotImplementedError

    async def _release_impl(self, name: str):
        self.released.append(name)
        if self.fail:
            raise RuntimeError('release failed')

    async def _release_all_impl(self):
        if self.fail_all:
            raise RuntimeError('release all failed')


@pytest.mark.asyncio
async def test_execution_context_release_runtime_updates_cache_only_after_success():
    provider = _ReleaseProvider()
    context = ExecutionContext(runtime_provider=provider)
    context._runtimes['main'] = cast(Any, object())

    await context.release_runtime('main')

    assert provider.released == ['main']
    assert 'main' not in context._runtimes

    failing_provider = _ReleaseProvider(fail=True)
    failing_context = ExecutionContext(runtime_provider=failing_provider)
    failing_context._runtimes['main'] = cast(Any, object())
    with pytest.raises(RuntimeError, match='release failed'):
        await failing_context.release_runtime('main')
    assert 'main' in failing_context._runtimes


@pytest.mark.asyncio
async def test_execution_context_retains_cache_when_release_is_suppressed():
    provider = _ReleaseProvider()
    provider._suppress_release = True
    context = ExecutionContext(runtime_provider=provider)
    context._runtimes['main'] = cast(Any, object())

    assert await context.release_runtime('main') is False
    assert 'main' in context._runtimes
    assert provider.released == []

    assert await context.release_runtime('main', force=True) is True
    assert 'main' not in context._runtimes
    assert provider.released == ['main']


@pytest.mark.asyncio
async def test_execution_context_preserves_primary_error_when_cleanup_fails():
    context = ExecutionContext(runtime_provider=_ReleaseProvider(fail_all=True))

    with pytest.raises(ValueError, match='primary failure'):
        async with context:
            raise ValueError('primary failure')


@pytest.mark.asyncio
async def test_execution_context_reports_cleanup_error_without_primary_error():
    context = ExecutionContext(runtime_provider=_ReleaseProvider(fail_all=True))

    with pytest.raises(RuntimeError, match='release all failed'):
        async with context:
            pass


@pytest.mark.asyncio
async def test_runtime_release_retries_transient_failure():
    class _RetryContext:
        def __init__(self):
            self.attempts = 0

        async def release_runtime(self, name: str, *, force: bool = False):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError('transient stop failure')
            return True

    context = _RetryContext()

    assert await task_module._release_runtime_with_retry(context, 'verifier') is True
    assert context.attempts == 2


@pytest.mark.asyncio
async def test_gongfeng_release_all_retains_tracking_on_stop_failure():
    provider: Any = object.__new__(GongfengRuntimeProvider)
    provider._runtime_ids = {'main': 'runtime-main', 'verifier': 'runtime-verifier'}
    provider._active_runtimes = {}
    provider._watchdog_tasks = {}
    provider._stop_watchdog = AsyncMock()
    provider._stop_runtime = AsyncMock(side_effect=RuntimeError('batch stop failed'))

    with pytest.raises(RuntimeError, match='batch stop failed'):
        await provider._release_all_impl()

    assert provider._runtime_ids == {
        'main': 'runtime-main',
        'verifier': 'runtime-verifier',
    }


@pytest.mark.asyncio
async def test_gongfeng_release_one_retains_tracking_on_stop_failure():
    provider: Any = object.__new__(GongfengRuntimeProvider)
    runtime = SimpleNamespace(ws_disconnect=AsyncMock())
    provider._runtime_ids = {'main': 'runtime-main'}
    provider._active_runtimes = {'main': runtime}
    provider._watchdog_tasks = {}
    provider._stop_watchdog = AsyncMock()
    provider._stop_runtime = AsyncMock(side_effect=RuntimeError('stop failed'))

    with pytest.raises(RuntimeError, match='stop failed'):
        await provider._release_impl('main')

    assert provider._runtime_ids == {'main': 'runtime-main'}
    assert provider._active_runtimes == {'main': runtime}
