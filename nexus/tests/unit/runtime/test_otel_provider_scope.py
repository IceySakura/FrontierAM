from __future__ import annotations

import ast

from pathlib import Path
from typing import Any

import pytest

from nexus.runtime.providers.base import RuntimeProvider


class _Provider(RuntimeProvider):
    async def acquire(self, name: str, **params: Any) -> Any:
        raise NotImplementedError

    async def _release_impl(self, name: str) -> None:
        raise NotImplementedError


def _enable_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://trusted.invalid:4318')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_HEADERS', 'authorization=trusted')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_PROTOCOL', 'http/protobuf')


def test_base_provider_does_not_inject_managed_otel_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_otel(monkeypatch)
    environment = {'USER_VALUE': 'kept', 'OTEL_SERVICE_NAME': 'user-owned'}

    params = _Provider().get_params('main', environment=environment)

    assert params == {'environment': environment}
    assert params['environment'] is environment


def test_ags_provider_has_no_otel_specific_integration() -> None:
    path = Path('nexus/runtime/providers/ags/provider.py')
    tree = ast.parse(path.read_text(), filename=str(path))
    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(module.startswith('nexus.utils.otel') for module in imported_modules)
    assert '_managed_otel_environment' not in path.read_text()
    assert '_merge_env_overlay' not in path.read_text()


@pytest.mark.asyncio
async def test_docker_provider_uses_legacy_runner_when_host_otel_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nexus.runtime.providers import docker

    _enable_otel(monkeypatch)
    provider = docker.DockerRuntimeProvider(
        image='runtime-image',
        docker_host='gateway.invalid',
        print_port_mappings=False,
        auto_connect_ws=False,
    )
    commands: list[str] = []

    async def noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def false(*_args: Any, **_kwargs: Any) -> bool:
        return False

    async def port(*_args: Any, **_kwargs: Any) -> int:
        return 32000

    async def run_legacy(command: str) -> str:
        commands.append(command)
        return ''

    async def reject_otel_runner(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError('Docker provider must not select an OTel-specific shell runner')

    monkeypatch.setattr(provider, '_ensure_runtime_image', noop)
    monkeypatch.setattr(provider, '_ensure_runtime_container', noop)
    monkeypatch.setattr(provider, '_wait_for_container_ready', noop)
    monkeypatch.setattr(docker, 'container_exists', false)
    monkeypatch.setattr(docker, 'get_container_port', port)
    monkeypatch.setattr(docker, 'run_shell_cmd', run_legacy)
    monkeypatch.setattr(docker, '_run_shell_cmd_redacted', reject_otel_runner, raising=False)

    await provider.acquire('main', environment={'USER_VALUE': 'kept'})

    assert len(commands) == 1
    assert 'USER_VALUE=kept' in commands[0]


@pytest.mark.asyncio
async def test_gongfeng_provider_injects_managed_http_otel_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nexus.runtime.providers.gongfeng import GongfengRuntimeProvider

    _enable_otel(monkeypatch)
    monkeypatch.setenv(
        'NEXUS_OTEL_RUNTIME_EXPORTER_OTLP_ENDPOINT',
        'http://runtime-collector.invalid:4318',
    )
    provider = GongfengRuntimeProvider(
        api_base_url='http://gongfeng.invalid',
        token='token',
        image='runtime-image',
        auto_connect_ws=False,
        startup_jitter_max=0,
    )
    captured: dict[str, Any] = {}

    async def start_runtime(**kwargs: Any):
        captured.update(kwargs)
        return 'runtime-id', 'http://runtime.invalid', {}, 'scheduled'

    monkeypatch.setattr(provider, '_start_runtime', start_runtime)

    await provider.acquire(
        'main',
        environment={
            'USER_VALUE': 'kept',
            'OTEL_SERVICE_NAME': 'forged',
            'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://forged.invalid:4318',
            'OTEL_EXPORTER_OTLP_HEADERS': 'authorization=forged',
            'NEXUS_OTEL_RUNTIME_EXPORTER_OTLP_ENDPOINT': 'http://forged-runtime.invalid:4318',
        },
    )

    environment = captured['environment']
    assert environment['USER_VALUE'] == 'kept'
    assert environment['OTEL_SERVICE_NAME'] == 'nexus-runtime-base'
    assert environment['OTEL_EXPORTER_OTLP_ENDPOINT'] == 'http://runtime-collector.invalid:4318'
    assert environment['OTEL_EXPORTER_OTLP_HEADERS'] == 'authorization=trusted'
    assert environment['NEXUS_OTEL_RUNTIME_EXPORTER_OTLP_ENDPOINT'] == 'http://runtime-collector.invalid:4318'
    assert environment['OTEL_EXPORTER_OTLP_PROTOCOL'] == 'http/protobuf'
    assert environment['NEXUS_OTEL_PROCESS_ROLE'] == 'runtime-base'
