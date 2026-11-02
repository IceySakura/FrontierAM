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
async def test_gongfeng_start_runtime_injects_disable_aslr_payload() -> None:
    provider = _make_gongfeng_provider()
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={'code': 0, 'data': {'runtime_id': 'rid-aslr', 'url': 'http://runtime', 'port_mappings': {}}}
    )

    await provider._start_runtime(image='img', runtime_disable_aslr=True)

    _endpoint, payload = provider._call_api.await_args.args
    assert payload['runtime_disable_aslr'] is True


@pytest.mark.asyncio
async def test_gongfeng_start_runtime_omits_disable_aslr_by_default() -> None:
    provider = _make_gongfeng_provider()
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={'code': 0, 'data': {'runtime_id': 'rid-default', 'url': 'http://runtime', 'port_mappings': {}}}
    )

    await provider._start_runtime(image='img')

    _endpoint, payload = provider._call_api.await_args.args
    assert 'runtime_disable_aslr' not in payload


@pytest.mark.asyncio
async def test_gongfeng_acquire_passes_disable_aslr_to_start_runtime() -> None:
    provider = _make_gongfeng_provider()
    provider._start_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value=('rid-acquire', 'http://runtime', {}, gf.RUNTIME_STATUS_SCHEDULED)
    )
    fake_runtime = MagicMock()
    fake_runtime.port_mappings = {}
    provider._create_runtime = MagicMock(return_value=fake_runtime)  # type: ignore[method-assign]
    provider._start_watchdog = MagicMock()  # type: ignore[method-assign]

    runtime = await provider.acquire('main', runtime_disable_aslr='true')

    assert runtime is fake_runtime
    assert provider._start_runtime.await_args.kwargs['runtime_disable_aslr'] is True


@pytest.mark.asyncio
async def test_gongfeng_runtime_config_disable_aslr_reaches_start_runtime() -> None:
    provider = _make_gongfeng_provider()
    provider.configure('main', runtime_disable_aslr=True)
    provider._start_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value=('rid-config', 'http://runtime', {}, gf.RUNTIME_STATUS_SCHEDULED)
    )
    fake_runtime = MagicMock()
    fake_runtime.port_mappings = {}
    provider._create_runtime = MagicMock(return_value=fake_runtime)  # type: ignore[method-assign]
    provider._start_watchdog = MagicMock()  # type: ignore[method-assign]

    await provider.acquire('main')

    assert provider._start_runtime.await_args.kwargs['runtime_disable_aslr'] is True


@pytest.mark.asyncio
async def test_gongfeng_runtime_config_can_disable_provider_default_disable_aslr() -> None:
    provider = _make_gongfeng_provider(runtime_disable_aslr=True)
    provider.configure('main', runtime_disable_aslr=False)
    provider._start_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value=('rid-config', 'http://runtime', {}, gf.RUNTIME_STATUS_SCHEDULED)
    )
    fake_runtime = MagicMock()
    fake_runtime.port_mappings = {}
    provider._create_runtime = MagicMock(return_value=fake_runtime)  # type: ignore[method-assign]
    provider._start_watchdog = MagicMock()  # type: ignore[method-assign]

    await provider.acquire('main')

    assert provider._start_runtime.await_args.kwargs['runtime_disable_aslr'] is False


@pytest.mark.asyncio
async def test_gongfeng_acquire_rejects_ambiguous_disable_aslr() -> None:
    provider = _make_gongfeng_provider()

    with pytest.raises(ValueError, match='runtime_disable_aslr must be a boolean'):
        await provider.acquire('main', runtime_disable_aslr='sometimes')


@pytest.mark.asyncio
async def test_local_docker_provider_rejects_disable_aslr() -> None:
    provider = DockerRuntimeProvider(image='unit-test/image:latest')

    with pytest.raises(ValueError, match='runtime_disable_aslr is only supported by the Gongfeng/gVisor runtime path'):
        await provider.acquire('main', runtime_disable_aslr=True)


def test_local_docker_provider_rejects_provider_level_disable_aslr() -> None:
    with pytest.raises(ValueError, match='runtime_disable_aslr is only supported by the Gongfeng/gVisor runtime path'):
        DockerRuntimeProvider(image='unit-test/image:latest', runtime_disable_aslr=True)
