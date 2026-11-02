from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from nexus.extensions.harbor_bridge import compiler, generator
from nexus.extensions.harbor_bridge import task as task_module
from nexus.extensions.harbor_bridge.compose import (
    ComposeAdmissionError,
    ComposeConfigError,
    ComposeDefinitionError,
    compose_config_from_task,
)
from nexus.extensions.harbor_bridge.identity import task_content_sha256


MAIN_IMAGE = f'mirrors.tencent.com/rl-task/main@sha256:{"a" * 64}'
DB_IMAGE = f'mirrors.tencent.com/rl-task/db@sha256:{"b" * 64}'
MIGRATE_IMAGE = f'mirrors.tencent.com/rl-task/migrate@sha256:{"c" * 64}'
CONTROLLER_IMAGE = f'mirrors.tencent.com/rl-task/compose-controller@sha256:{"d" * 64}'


def _write_compose_task(
    root: Path,
    *,
    enable_dind: bool = True,
    verifier_mode: str = 'shared',
    primary_service: str = 'main',
    collect_service: str = 'db',
    controller_image: str = CONTROLLER_IMAGE,
    with_boot: bool = False,
    compose_overrides: dict | None = None,
    extension_overrides: dict | None = None,
) -> Path:
    task_dir = root / 'compose-task'
    (task_dir / 'environment' / 'config').mkdir(parents=True)
    (task_dir / 'tests').mkdir()
    (task_dir / 'environment' / 'config' / 'app.yaml').write_text('mode: test\n')

    compose = {
        'services': {
            'main': {
                'image': MAIN_IMAGE,
                'working_dir': '/workdir',
                'entrypoint': ['/app/start'],
                'depends_on': {
                    'db': {'condition': 'service_healthy'},
                    'migrate': {'condition': 'service_completed_successfully'},
                },
                'volumes': ['workspace:/workdir', './config:/etc/task:ro'],
            },
            'db': {
                'image': DB_IMAGE,
                'healthcheck': {'test': ['CMD-SHELL', 'db-ready']},
            },
            'migrate': {
                'image': MIGRATE_IMAGE,
                'restart': 'no',
                'depends_on': {'db': {'condition': 'service_healthy'}},
            },
        },
        'volumes': {'workspace': {}},
        'networks': {'default': {}},
    }
    if compose_overrides:
        _deep_update(compose, compose_overrides)
    (task_dir / 'environment' / 'docker-compose.lock.yaml').write_text(
        yaml.safe_dump(compose, sort_keys=False),
        encoding='utf-8',
    )

    extension = {
        'definition': 'environment/docker-compose.lock.yaml',
        'primary_service': primary_service,
        'startup_timeout_sec': 180,
        'shutdown_timeout_sec': 30,
        'one_shot_services': [],
        **(extension_overrides or {}),
    }
    extension_toml = '\n'.join(f'{key} = {_toml_value(value)}' for key, value in extension.items())
    boot = '\n[environment.boot]\ncommand = "/environment/t3_boot.sh"\n' if with_boot else ''
    (task_dir / 'task.toml').write_text(
        f'''schema_version = "1.2"
artifacts = [{{ service = "db", source = "/tmp/database.sql", destination = "db/database.sql" }}]

[environment]
docker_image = "{controller_image}"
workdir = "/workdir"
cpus = 8
memory_mb = 16384
enable_dind = {str(enable_dind).lower()}
network_mode = "public"

[environment.env]
TASK_MODE = "evaluation"

[environment.healthcheck]
command = "test -f /workdir/.initialized"
timeout_sec = 10
interval_sec = 2
retries = 30
{boot}
[agent]
timeout_sec = 3600
user = "root"

[verifier]
environment_mode = "{verifier_mode}"
timeout_sec = 1800

[[verifier.collect]]
service = "{collect_service}"
command = "pg_dump -U app app > /tmp/database.sql"
timeout_sec = 60

[extensions.nexus.compose]
{extension_toml}
''',
        encoding='utf-8',
    )
    (task_dir / 'instruction.md').write_text('Repair the service.\n')
    (task_dir / 'tests' / 'test.sh').write_text('#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n')
    return task_dir


def _compile(task_dir: Path):
    loaded = compiler.load_harbor_task(task_dir)
    return compiler.compile_harbor_task(task_dir, loaded.config, loaded.task)


def _deep_update(target: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _toml_value(value):
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        return '[' + ', '.join(_toml_value(item) for item in value) + ']'
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def test_compiler_builds_frozen_compose_plan(tmp_path: Path) -> None:
    task_dir = _write_compose_task(tmp_path)

    compiled = _compile(task_dir)
    plan = compiled.task_config['compose_plan']

    assert plan['schema_version'] == 'nexus-compose-plan-v1'
    assert plan['definition'] == 'environment/docker-compose.lock.yaml'
    assert plan['definition_sha256'].startswith('sha256:')
    assert plan['controller_image'] == CONTROLLER_IMAGE
    assert plan['primary_service'] == 'main'
    assert plan['primary_image'] == MAIN_IMAGE
    assert plan['workdir'] == '/workdir'
    assert plan['environment'] == {'TASK_MODE': 'evaluation'}
    assert plan['one_shot_services'] == ['migrate']
    assert plan['bind_paths'] == ['environment/config']
    assert plan['services']['main']['depends_on'] == ['db', 'migrate']
    assert plan['services']['db']['has_healthcheck'] is True
    assert plan['services']['migrate']['one_shot'] is True
    assert compiled.metadata['execution_kind'] == 'compose'
    assert compiled.metadata['compose_controller_image'] == CONTROLLER_IMAGE
    assert compiled.metadata['compose_primary_service'] == 'main'
    assert compiled.metadata['compose_primary_image'] == MAIN_IMAGE
    assert compiled.metadata['compose_services'] == ['db', 'main', 'migrate']
    assert compiled.runtime_configs['main']['image'] == CONTROLLER_IMAGE
    assert compiled.runtime_configs['main']['working_dir'] == '/nexus/control'
    assert compiled.runtime_configs['main']['environment'] == {}
    assert compiled.runtime_configs['main']['worker_venv'] == {'modules': ['claude_code', 'codex', 'hycode']}


def test_compose_config_is_absent_for_single_container_task() -> None:
    assert compose_config_from_task({'environment': {}}) is None


def test_compose_config_rejects_unknown_fields() -> None:
    with pytest.raises(ComposeConfigError, match='Extra inputs'):
        compose_config_from_task({
            'extensions': {
                'nexus': {
                    'compose': {
                        'definition': 'environment/docker-compose.lock.yaml',
                        'primary_service': 'main',
                        'runtime_build': True,
                    }
                }
            }
        })


def test_compose_definition_rejects_service_names_with_surrounding_whitespace(tmp_path: Path) -> None:
    task_dir = _write_compose_task(
        tmp_path,
        compose_overrides={
            'services': {
                ' main ': {
                    'image': MAIN_IMAGE,
                }
            }
        },
    )

    with pytest.raises(ComposeDefinitionError, match='surrounding whitespace'):
        _compile(task_dir)


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'enable_dind': False}, 'enable_dind'),
        ({'primary_service': 'missing'}, 'not present'),
        ({'primary_service': 'migrate'}, 'cannot be a one-shot'),
        ({'collect_service': 'missing'}, 'collect references unknown'),
    ],
)
def test_compose_task_level_contract_is_fail_closed(tmp_path: Path, kwargs: dict, message: str) -> None:
    task_dir = _write_compose_task(tmp_path, **kwargs)

    with pytest.raises((ComposeConfigError, ComposeAdmissionError), match=message):
        _compile(task_dir)


def test_compose_allows_separate_verifier_and_sidecar_collection(tmp_path: Path) -> None:
    task_dir = _write_compose_task(tmp_path, verifier_mode='separate', collect_service='db')

    compiled = _compile(task_dir)

    assert compiled.task_config['verifier_mode'] == 'separate'
    assert compiled.runtime_configs['verifier']['image'] == CONTROLLER_IMAGE
    assert compiled.task_config['compose_plan']['primary_service'] == 'main'


def test_compose_allows_primary_boot_command_and_override(tmp_path: Path) -> None:
    task_dir = _write_compose_task(tmp_path, with_boot=True)

    compiled = _compile(task_dir)
    assert compiled.task_config['boot_command'] == '/environment/t3_boot.sh'

    loaded = compiler.load_harbor_task(task_dir)
    overridden = compiler.compile_harbor_task(
        task_dir,
        loaded.config,
        loaded.task,
        boot_command_override='/opt/task/pre-agent.sh',
    )
    assert overridden.task_config['boot_command'] == '/opt/task/pre-agent.sh'


def test_compose_start_retry_config_propagates_to_runtime_spec(tmp_path: Path) -> None:
    task_dir = _write_compose_task(
        tmp_path,
        extension_overrides={
            'start_retry_attempts': 3,
            'start_retry_backoff_sec': 8.0,
            'start_retry_jitter_max': 60.0,
        },
    )

    plan = task_module.ComposePlan.model_validate(_compile(task_dir).task_config['compose_plan'])
    assert plan.start_retry_attempts == 3
    assert plan.start_retry_backoff_sec == 8.0
    assert plan.start_retry_jitter_max == 60.0

    from nexus.extensions.harbor_bridge.compose import topology_spec_from_plan

    spec = topology_spec_from_plan(plan)
    assert spec.start_retry_attempts == 3
    assert spec.start_retry_backoff_sec == 8.0
    assert spec.start_retry_jitter_max == 60.0


@pytest.mark.parametrize(
    ('extension_overrides', 'message'),
    [
        ({'start_retry_attempts': 0}, 'start_retry_attempts'),
        ({'start_retry_backoff_sec': -1.0}, 'start_retry_backoff_sec'),
        ({'start_retry_jitter_max': -1.0}, 'start_retry_jitter_max'),
    ],
)
def test_compose_rejects_invalid_start_retry_config(tmp_path: Path, extension_overrides: dict, message: str) -> None:
    task_dir = _write_compose_task(tmp_path, extension_overrides=extension_overrides)

    with pytest.raises(ComposeConfigError, match=message):
        _compile(task_dir)


def test_compose_timeouts_have_no_bridge_maximum(tmp_path: Path) -> None:
    task_dir = _write_compose_task(
        tmp_path,
        extension_overrides={
            'startup_timeout_sec': 86400,
            'shutdown_timeout_sec': 3600,
        },
    )

    plan = task_module.ComposePlan.model_validate(_compile(task_dir).task_config['compose_plan'])
    assert plan.startup_timeout_sec == 86400
    assert plan.shutdown_timeout_sec == 3600

    from nexus.extensions.harbor_bridge.compose import topology_spec_from_plan

    spec = topology_spec_from_plan(plan)
    assert spec.startup_timeout_sec == 86400
    assert spec.shutdown_timeout_sec == 3600


def test_compose_task_rejects_legacy_dind_override_that_disables_controller(tmp_path: Path) -> None:
    task_dir = _write_compose_task(tmp_path)
    loaded = compiler.load_harbor_task(task_dir)

    with pytest.raises(ValueError, match='effective environment.enable_dind'):
        compiler.compile_harbor_task(
            task_dir,
            loaded.config,
            loaded.task,
            enable_dind_override=False,
        )


@pytest.mark.parametrize(
    ('overrides', 'message'),
    [
        ({'services': {'main': {'build': {'context': '.'}}}}, 'runtime build'),
        ({'services': {'main': {'privileged': True}}}, 'cannot be privileged'),
        ({'services': {'main': {'network_mode': 'host'}}}, 'network_mode=host'),
        ({'services': {'main': {'volumes': ['/var/run/docker.sock:/var/run/docker.sock']}}}, 'Docker socket'),
        ({'services': {'main': {'use_api_socket': True}}}, 'use_api_socket'),
        ({'services': {'main': {'volumes_from': ['db']}}}, 'volumes_from'),
        ({'services': {'main': {'label_file': '../../labels.env'}}}, 'unsupported fields'),
        ({'services': {'main': {'runtime': 'nvidia'}}}, 'unsupported fields'),
        ({'services': {'main': {'device_cgroup_rules': ['c 1:3 mr']}}}, 'unsupported fields'),
        ({'services': {'main': {'security_opt': ['seccomp=unconfined']}}}, 'unsupported security_opt'),
        ({'services': {'main': {'extra_hosts': ['host.docker.internal:host-gateway']}}}, 'host-gateway'),
        (
            {'services': {'main': {'environment': {'INHERITED': None}}}},
            'must define a concrete value',
        ),
        (
            {'services': {'main': {'environment': ['INHERITED']}}},
            'invalid environment entry',
        ),
        (
            {'volumes': {'workspace': {'driver_opts': {'type': 'none', 'device': '/', 'o': 'bind'}}}},
            'driver_opts',
        ),
        ({'services': {'main': {'volumes': ['missing:/workdir']}}}, 'undeclared volume'),
        ({'services': {'main': {'volumes': ['foo/bar:/workdir']}}}, 'invalid named volume'),
        (
            {'services': {'main': {'volumes': [{'type': 'bind', 'source': './config', 'target': 'relative'}]}}},
            'absolute container path',
        ),
        ({'services': {'main': {'volumes': ['workspace:/nix/runtime']}}}, 'reserved for Nexus Runtime'),
        ({'services': {'main': {'volumes': ['workspace:/opt/nexus-agent/cache']}}}, 'reserved for Nexus Runtime'),
        ({'services': {'main': {'tmpfs': ['/nix/runtime:mode=1777']}}}, 'reserved for Nexus Runtime'),
        ({'services': {'main': {'tmpfs': ['/nix/store']}}}, 'reserved for Nexus Runtime'),
        ({'services': {'main': {'tmpfs': ['/nix/store/cache']}}}, 'reserved for Nexus Runtime'),
        ({'services': {'main': {'tmpfs': ['/nix/runtime/nix-env/cache']}}}, 'reserved for Nexus Runtime'),
        ({'services': {'main': {'tmpfs': '/opt/nexus-agent/cache'}}}, 'reserved for Nexus Runtime'),
        ({'services': {'main': {'tmpfs': ['relative']}}}, 'absolute container path'),
        ({'services': {'main': {'tmpfs': {'target': '/tmp'}}}}, 'string or list of strings'),
        ({'services': {'main': {'networks': ['missing']}}}, 'undeclared networks'),
        ({'include': ['unsafe.yaml']}, 'unsupported top-level'),
    ],
)
def test_compose_definition_admission_rejects_unsafe_features(
    tmp_path: Path,
    overrides: dict,
    message: str,
) -> None:
    task_dir = _write_compose_task(tmp_path, compose_overrides=overrides)

    with pytest.raises(ComposeAdmissionError, match=message):
        _compile(task_dir)


def test_compose_allows_non_reserved_service_tmpfs(tmp_path: Path) -> None:
    task_dir = _write_compose_task(
        tmp_path,
        compose_overrides={
            'services': {
                'main': {
                    'tmpfs': [
                        '/tmp:size=64m,mode=1777',
                        '/logs',
                        '/tests',
                        '/nix/runtime/nexus/nexus/extensions/tasks:mode=1777',
                        '/nix/runtime/nexus/tests:mode=1777',
                    ]
                }
            }
        },
    )

    assert _compile(task_dir).task_config['compose_plan']['primary_service'] == 'main'


def test_compose_allows_host_coupled_compatibility_features(tmp_path: Path) -> None:
    task_dir = _write_compose_task(
        tmp_path,
        compose_overrides={
            'name': 'fixed-topology',
            'services': {
                'main': {
                    'container_name': 'fixed-main',
                    'image': 'mirrors.tencent.com/rl-task/main:latest',
                    'environment': {
                        'FIXED': 'value',
                        'EXPANDED': '${CONTROLLER_VALUE:-fallback}',
                    },
                    'volumes': [
                        '/root:/controller-root',
                        '${HOST_DATA_DIR:-/data}:/host-data',
                    ],
                }
            },
            'volumes': {'workspace': {'external': True, 'name': 'fixed-workspace'}},
            'networks': {'default': {'external': True, 'name': 'fixed-network'}},
        },
    )

    plan = _compile(task_dir).task_config['compose_plan']

    assert plan['primary_image'].endswith(':latest')
    assert plan['bind_paths'] == []


def test_compose_allows_common_extended_service_fields(tmp_path: Path) -> None:
    task_dir = _write_compose_task(
        tmp_path,
        compose_overrides={
            'services': {
                'main': {
                    'deploy': {'resources': {'limits': {'cpus': '2', 'memory': '2G'}}},
                    'env_file': ['./runtime.env'],
                    'configs': ['app-config'],
                    'cap_drop': ['ALL'],
                    'security_opt': ['no-new-privileges:true'],
                    'shm_size': '1g',
                    'ulimits': {'nofile': {'soft': 65536, 'hard': 65536}},
                }
            },
            'configs': {'app-config': {'file': './app.conf'}},
        },
    )

    assert _compile(task_dir).task_config['compose_plan']['primary_service'] == 'main'


def test_compose_definition_rejects_dependency_cycle(tmp_path: Path) -> None:
    task_dir = _write_compose_task(
        tmp_path,
        compose_overrides={'services': {'db': {'depends_on': {'main': {'condition': 'service_started'}}}}},
    )

    with pytest.raises(ComposeAdmissionError, match='contains a cycle'):
        _compile(task_dir)


def test_compose_healthy_dependency_requires_healthcheck(tmp_path: Path) -> None:
    task_dir = _write_compose_task(
        tmp_path,
        compose_overrides={'services': {'db': {'healthcheck': None}}},
    )

    with pytest.raises(ComposeAdmissionError, match='has no healthcheck'):
        _compile(task_dir)


@pytest.mark.parametrize(
    'healthcheck',
    [
        {'disable': True},
        {'test': ['NONE']},
    ],
)
def test_compose_healthy_dependency_rejects_disabled_healthcheck(
    tmp_path: Path,
    healthcheck: dict,
) -> None:
    task_dir = _write_compose_task(
        tmp_path,
        compose_overrides={'services': {'db': {'healthcheck': healthcheck}}},
    )

    with pytest.raises(ComposeAdmissionError, match='has no healthcheck'):
        _compile(task_dir)


def test_compose_dependency_flags_must_be_boolean(tmp_path: Path) -> None:
    task_dir = _write_compose_task(
        tmp_path,
        compose_overrides={
            'services': {
                'main': {
                    'depends_on': {
                        'db': {
                            'condition': 'service_healthy',
                            'required': 'yes',
                        }
                    }
                }
            }
        },
    )

    with pytest.raises(ComposeAdmissionError, match='must be a boolean'):
        _compile(task_dir)


def test_compose_lock_allows_escaped_runtime_shell_variables(tmp_path: Path) -> None:
    task_dir = _write_compose_task(
        tmp_path,
        compose_overrides={'services': {'main': {'command': 'echo $$HOME'}}},
    )

    assert _compile(task_dir).task_config['compose_plan']['primary_service'] == 'main'


def test_compose_definition_path_cannot_escape_task(tmp_path: Path) -> None:
    task_dir = _write_compose_task(
        tmp_path,
        extension_overrides={'definition': '../outside.yaml'},
    )

    with pytest.raises(ComposeDefinitionError, match='safe task-relative'):
        _compile(task_dir)


def test_compose_bind_source_cannot_be_a_symlink(tmp_path: Path) -> None:
    task_dir = _write_compose_task(
        tmp_path,
        compose_overrides={'services': {'main': {'volumes': ['./linked:/etc/task:ro']}}},
    )
    (task_dir / 'environment' / 'linked').symlink_to(
        task_dir / 'environment' / 'config',
        target_is_directory=True,
    )

    with pytest.raises(ComposeAdmissionError, match='must not contain symlinks'):
        _compile(task_dir)


def test_compose_controller_and_primary_images_are_independent(tmp_path: Path) -> None:
    task_dir = _write_compose_task(tmp_path)

    compiled = _compile(task_dir)

    assert compiled.runtime_configs['main']['image'] == CONTROLLER_IMAGE
    assert compiled.task_config['compose_plan']['primary_image'] == MAIN_IMAGE


def test_compose_controller_and_service_images_may_use_tags(tmp_path: Path) -> None:
    task_dir = _write_compose_task(
        tmp_path,
        controller_image='mirrors.tencent.com/rl-task/compose-controller:latest',
        compose_overrides={'services': {'main': {'image': 'mirrors.tencent.com/rl-task/main:latest'}}},
    )

    compiled = _compile(task_dir)

    assert compiled.runtime_configs['main']['image'].endswith(':latest')
    assert compiled.task_config['compose_plan']['primary_image'].endswith(':latest')


def test_compose_primary_workdir_must_match_environment(tmp_path: Path) -> None:
    task_dir = _write_compose_task(
        tmp_path,
        compose_overrides={'services': {'main': {'working_dir': '/different'}}},
    )

    with pytest.raises(ComposeAdmissionError, match='environment.workdir must match'):
        _compile(task_dir)


def test_compose_lock_and_bind_files_change_task_identity(tmp_path: Path) -> None:
    task_dir = _write_compose_task(tmp_path)
    before = task_content_sha256(task_dir)

    config_path = task_dir / 'environment' / 'config' / 'app.yaml'
    config_path.write_text('mode: changed\n')
    after_bind_change = task_content_sha256(task_dir)
    assert after_bind_change != before

    compose_path = task_dir / 'environment' / 'docker-compose.lock.yaml'
    compose_path.write_text(compose_path.read_text() + '\n# changed\n')
    assert task_content_sha256(task_dir) != after_bind_change


def test_compose_bind_file_mode_changes_task_identity(tmp_path: Path) -> None:
    task_dir = _write_compose_task(tmp_path)
    config_path = task_dir / 'environment' / 'config' / 'app.yaml'
    before = task_content_sha256(task_dir)

    config_path.chmod(0o755)

    assert task_content_sha256(task_dir) != before


def test_compose_agent_transport_is_opt_in_at_runtime(tmp_path: Path) -> None:
    task_dir = _write_compose_task(tmp_path)
    compose_plan = task_module.ComposePlan.model_validate(_compile(task_dir).task_config['compose_plan'])

    from nexus.extensions.harbor_bridge.compose import topology_spec_from_plan

    assert topology_spec_from_plan(compose_plan).agent_transport == 'none'
    assert topology_spec_from_plan(compose_plan, agent_transport='socket-v1').agent_transport == 'socket-v1'


@pytest.mark.asyncio
async def test_compose_rejects_unsupported_agent_before_runtime_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_dir = _write_compose_task(tmp_path)
    compose_plan = _compile(task_dir).task_config['compose_plan']

    async def unsupported_agent(**_kwargs):
        raise AssertionError('unsupported Agent must not run')

    unsupported_agent.nexus_meta = {'name': 'terminus2', 'meta': {}}

    def unexpected_context():
        raise AssertionError('runtime context must not be requested')

    monkeypatch.setattr(task_module, 'require_context', unexpected_context)

    with pytest.raises(ValueError, match="Agent 'terminus2' does not support Native Compose execution"):
        await task_module.harbor_bridge_task(
            task_module.HarborBridgeTaskInput(
                task_dir=str(task_dir),
                workdir='/workdir',
                compose_plan=compose_plan,
                mode='agent',
                run_agent=unsupported_agent,
            )
        )


@pytest.mark.asyncio
async def test_authoritative_task_definition_cannot_remove_fail_closed_plan(tmp_path: Path, monkeypatch) -> None:
    task_dir = _write_compose_task(tmp_path)
    context = SimpleNamespace(runtime_provider=object(), task_files=None)
    monkeypatch.setattr(task_module, 'require_context', lambda: context)

    with pytest.raises(RuntimeError, match='Serialized compose_plan does not match'):
        await task_module.harbor_bridge_task(
            task_module.HarborBridgeTaskInput(
                task_dir=str(task_dir),
                workdir='/workdir',
                compose_plan=None,
                run_agent=AsyncMock(),
            )
        )


@pytest.mark.asyncio
async def test_compose_agent_receives_service_target_and_runs_shared_verifier(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_dir = _write_compose_task(tmp_path)
    compose_plan = _compile(task_dir).task_config['compose_plan']
    events: list[str] = []
    captured_agent_kwargs = {}
    captured_execution_target = None

    class FakeRuntime:
        async def run_command(self, **_):
            events.append('controller-command')
            return SimpleNamespace(return_code=0, stdout='', stderr='')

    class FakeServiceRuntime:
        async def run_command(self, **_):
            events.append('primary-command')
            return SimpleNamespace(return_code=0, stdout='', stderr='')

        async def upload_file(self, files, overwrite=True):
            events.extend(f'primary-upload:{item.path}' for item in files)
            return SimpleNamespace(
                results=[SimpleNamespace(path=item.path, success=True, message='') for item in files]
            )

    class FakeTopology:
        instance = None

        def __init__(self, *, runtime, spec, task_dir, project_name, agent_helper_dir=None):
            self.runtime = runtime
            self.spec = spec
            self.task_dir = task_dir
            self.project_name = project_name
            self.agent_helper_dir = agent_helper_dir
            self.controller_dir = f'/nexus/compose/{project_name}'
            self.compose_files = (
                f'{self.controller_dir}/lock.yaml',
                f'{self.controller_dir}/override.json',
            )
            self.agent_transport_host_dir = '/tmp/nexus-agent/test'
            self.agent_transport_container_dir = '/opt/nexus-agent'
            self.socket_bridge_host_path = '/tmp/nexus-agent/test/bin/nexus-socket-bridge'
            self.socket_bridge_container_path = '/opt/nexus-agent/bin/nexus-socket-bridge'
            FakeTopology.instance = self

        async def stage(self):
            events.append('topology-stage')

        async def start(self):
            events.append('topology-start')

        async def wait_ready(self):
            events.append('topology-ready')

        async def delete(self):
            events.append('topology-delete')

    class FakeEnvironment:
        capabilities = SimpleNamespace(docker_compose=True)

        def __init__(self, *, task_env_config, **_):
            self.os = task_env_config.os
            self.primary_runtime = FakeServiceRuntime()

        async def start(self, force_build=False):
            events.append('environment-start')

        async def run_healthcheck(self):
            events.append('healthcheck')

        async def exec(self, *_, **__):
            events.append('collect-main')
            return SimpleNamespace(return_code=0, stdout='', stderr='')

        async def service_exec(self, *_, service, **__):
            events.append(f'collect:{service}')
            return SimpleNamespace(return_code=0, stdout='', stderr='')

        async def empty_dirs(self, *_, **__):
            events.append('empty-verifier')
            return SimpleNamespace(return_code=0, stdout='', stderr='')

        async def stop(self, delete=True):
            events.append('environment-stop')

    class FakeContext:
        runtime_provider = object()
        task_files = None
        run_id = 'Run/Attempt:1'
        task_config = {}
        tool_state = {}

        async def get_runtime(self, name='main'):
            assert name == 'main'
            events.append('controller-acquire')
            return FakeRuntime()

    async def run_verifier(**_):
        events.append('verify')
        return {'reward': 1.0}, {}, 'passed'

    async def boot_primary(runtime, *_args, **_kwargs):
        assert isinstance(runtime, FakeServiceRuntime)
        events.append('boot-primary')

    async def run_agent(**kwargs):
        nonlocal captured_execution_target

        from nexus.extensions.agents.installed_common.execution import get_agent_execution_target

        captured_agent_kwargs.update(kwargs)
        captured_execution_target = get_agent_execution_target()
        events.append('agent')
        return SimpleNamespace(final_answer='done')

    run_agent.nexus_meta = {'meta': {'supports_execution_target': True}}

    import nexus.extensions.harbor_bridge.compose.environment as compose_environment_module

    monkeypatch.setattr(task_module, 'require_context', lambda: FakeContext())
    monkeypatch.setattr(task_module, 'ComposeTopology', FakeTopology)
    monkeypatch.setattr(compose_environment_module, 'ComposeHarborEnvironment', FakeEnvironment)
    monkeypatch.setattr(task_module, '_boot_image_services', boot_primary)
    monkeypatch.setattr(task_module, '_run_harbor_verifier', run_verifier)

    output = await task_module.harbor_bridge_task(
        task_module.HarborBridgeTaskInput(
            task_dir=str(task_dir),
            workdir='/workdir',
            compose_plan=compose_plan,
            mode='agent',
            run_agent=run_agent,
        )
    )

    assert output.score == 1.0
    assert events.index('topology-stage') < events.index('topology-start') < events.index('topology-ready')
    assert events.index('topology-ready') < events.index('collect-main') < events.index('boot-primary')
    assert events.index('boot-primary') < events.index('healthcheck') < events.index('agent')
    assert events.index('agent') < events.index('collect:db') < events.index('verify') < events.index('topology-delete')
    assert 'primary-command' not in events
    assert 'controller-command' not in events
    assert FakeTopology.instance.spec.agent_transport == 'socket-v1'
    assert FakeTopology.instance.spec.services['main'].image == MAIN_IMAGE
    assert FakeTopology.instance.project_name.startswith('compose-task-run-attempt-1-')
    assert captured_agent_kwargs['working_dir'] == '/workdir'
    assert captured_execution_target.service == 'main'
    assert captured_execution_target.project_directory == FakeTopology.instance.controller_dir
    assert captured_execution_target.user == 'root'


def test_legacy_generator_can_skip_compose_admission_failure(tmp_path: Path, monkeypatch) -> None:
    task_dir = _write_compose_task(
        tmp_path,
        compose_overrides={'services': {'main': {'build': {'context': '.'}}}},
    )
    monkeypatch.setenv('HARBOR_DATASET_PATH', str(task_dir))
    monkeypatch.setenv('GONGFENG_RUNTIME_TOKEN', 'test-token')
    monkeypatch.setenv('HARBOR_SKIP_INVALID_TASKS', 'true')
    monkeypatch.delenv('HARBOR_ENABLE_DIND', raising=False)
    monkeypatch.delenv('TASK_NAMES', raising=False)
    monkeypatch.delenv('HARBOR_TASK_INCLUDE', raising=False)

    assert list(generator.generate_configs()) == []


def test_legacy_generator_can_skip_missing_compose_identity_input(tmp_path: Path, monkeypatch) -> None:
    task_dir = _write_compose_task(tmp_path)
    (task_dir / 'environment' / 'docker-compose.lock.yaml').unlink()
    monkeypatch.setenv('HARBOR_DATASET_PATH', str(task_dir))
    monkeypatch.setenv('GONGFENG_RUNTIME_TOKEN', 'test-token')
    monkeypatch.setenv('HARBOR_SKIP_INVALID_TASKS', 'true')
    monkeypatch.delenv('TASK_NAMES', raising=False)
    monkeypatch.delenv('HARBOR_TASK_INCLUDE', raising=False)

    assert list(generator.generate_configs()) == []
    assert generator.get_total_count() == 0


def test_separate_verifier_collects_main_then_freezes_before_sidecars_and_release() -> None:
    source = Path(task_module.__file__).read_text()
    collect_function_pos = source.index('async def _collect_artifacts(')
    main_services_pos = source.index('services={MAIN_SERVICE_NAME}', collect_function_pos)
    stop_pos = source.index('await environment.stop_service(MAIN_SERVICE_NAME)', main_services_pos)
    sidecar_services_pos = source.index('services=sidecar_services', stop_pos)
    assert main_services_pos < stop_pos < sidecar_services_pos

    separate_pos = source.index('if verifier_mode == VerifierEnvironmentMode.SEPARATE:')
    collect_pos = source.index('artifact_handler, main_paths = await _collect_artifacts(', separate_pos)
    delete_pos = source.index('await compose_topology.delete()', collect_pos)
    release_pos = source.index("_release_runtime_with_retry(ctx, 'main'", delete_pos)
    assert collect_pos < delete_pos < release_pos
