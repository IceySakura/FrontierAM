from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.runtime.providers import gongfeng as gf
from nexus.runtime.providers.docker import DockerRuntimeProvider
from nexus.runtime.providers.gongfeng import GongfengRuntimeProvider


def _make_gongfeng_provider(**overrides) -> GongfengRuntimeProvider:
    return GongfengRuntimeProvider(
        api_base_url='http://gongfeng.example.com',
        token='unit-test-token',
        image='unit-test/image:latest',
        print_port_mappings=False,
        **overrides,
    )


@pytest.mark.asyncio
async def test_gongfeng_start_runtime_injects_dind_payload_and_environment() -> None:
    provider = _make_gongfeng_provider()
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={'code': 0, 'data': {'runtime_id': 'rid-dind', 'url': 'http://runtime', 'port_mappings': {}}}
    )

    await provider._start_runtime(
        image='img',
        environment={'EXISTING': 'value', 'NEXUS_ENABLE_DIND': 'false'},
        enable_dind=True,
    )

    _endpoint, payload = provider._call_api.await_args.args
    assert payload['enable_dind'] is True
    assert payload['environment']['EXISTING'] == 'value'
    assert payload['environment']['PYTHON_VERSION'] == provider.python_version
    assert payload['environment']['NEXUS_ENABLE_DIND'] == 'true'


@pytest.mark.asyncio
async def test_gongfeng_start_runtime_leaves_dind_payload_absent_by_default() -> None:
    provider = _make_gongfeng_provider()
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={'code': 0, 'data': {'runtime_id': 'rid-default', 'url': 'http://runtime', 'port_mappings': {}}}
    )

    await provider._start_runtime(image='img', environment={'EXISTING': 'value'})

    _endpoint, payload = provider._call_api.await_args.args
    assert 'enable_dind' not in payload
    assert payload['environment']['EXISTING'] == 'value'
    assert payload['environment']['PYTHON_VERSION'] == provider.python_version
    assert 'NEXUS_ENABLE_DIND' not in payload['environment']


@pytest.mark.asyncio
async def test_gongfeng_acquire_passes_enable_dind_to_start_runtime() -> None:
    provider = _make_gongfeng_provider()
    provider._start_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value=('rid-acquire', 'http://runtime', {}, gf.RUNTIME_STATUS_SCHEDULED)
    )
    fake_runtime = MagicMock()
    fake_runtime.port_mappings = {}
    provider._create_runtime = MagicMock(return_value=fake_runtime)  # type: ignore[method-assign]
    provider._start_watchdog = MagicMock()  # type: ignore[method-assign]

    runtime = await provider.acquire('main', enable_dind='true')

    assert runtime is fake_runtime
    assert provider._start_runtime.await_args.kwargs['enable_dind'] is True


@pytest.mark.asyncio
async def test_gongfeng_runtime_config_enable_dind_reaches_start_runtime() -> None:
    provider = _make_gongfeng_provider()
    provider.configure('main', enable_dind=True)
    provider._start_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value=('rid-config', 'http://runtime', {}, gf.RUNTIME_STATUS_SCHEDULED)
    )
    fake_runtime = MagicMock()
    fake_runtime.port_mappings = {}
    provider._create_runtime = MagicMock(return_value=fake_runtime)  # type: ignore[method-assign]
    provider._start_watchdog = MagicMock()  # type: ignore[method-assign]

    await provider.acquire('main')

    assert provider._start_runtime.await_args.kwargs['enable_dind'] is True


@pytest.mark.asyncio
async def test_gongfeng_runtime_config_can_disable_provider_default_enable_dind() -> None:
    provider = _make_gongfeng_provider(enable_dind=True)
    provider.configure('main', enable_dind=False)
    provider._start_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value=('rid-config', 'http://runtime', {}, gf.RUNTIME_STATUS_SCHEDULED)
    )
    fake_runtime = MagicMock()
    fake_runtime.port_mappings = {}
    provider._create_runtime = MagicMock(return_value=fake_runtime)  # type: ignore[method-assign]
    provider._start_watchdog = MagicMock()  # type: ignore[method-assign]

    await provider.acquire('main')

    assert provider._start_runtime.await_args.kwargs['enable_dind'] is False


@pytest.mark.asyncio
async def test_gongfeng_acquire_rejects_ambiguous_enable_dind() -> None:
    provider = _make_gongfeng_provider()

    with pytest.raises(ValueError, match='enable_dind must be a boolean'):
        await provider.acquire('main', enable_dind='sometimes')


@pytest.mark.asyncio
async def test_local_docker_provider_rejects_enable_dind() -> None:
    provider = DockerRuntimeProvider(image='unit-test/image:latest')

    with pytest.raises(ValueError, match='only supported by the Gongfeng/gVisor runtime path'):
        await provider.acquire('main', enable_dind=True)


def test_local_docker_provider_rejects_provider_level_enable_dind() -> None:
    with pytest.raises(ValueError, match='only supported by the Gongfeng/gVisor runtime path'):
        DockerRuntimeProvider(image='unit-test/image:latest', enable_dind=True)
