from __future__ import annotations

from typing import Any

import pytest

from nexus.core.context import ExecutionContext
from nexus.extensions.tasks.aidata.engine.models import AidataService, AidataServiceConnection
from nexus.extensions.tasks.aidata.runtime.services import setup_services
from nexus.runtime.providers.base import RuntimeProvider


class _FakeRuntime:
    def __init__(self, port_mappings: dict[int, str] | None = None) -> None:
        self.port_mappings = dict(port_mappings or {})


class _RefreshingProvider(RuntimeProvider):
    def __init__(self, runtime: _FakeRuntime, runtime_infos: list[dict[str, Any]]) -> None:
        super().__init__()
        self.runtime = runtime
        self.runtime_infos = list(runtime_infos)
        self.info_calls: list[str] = []

    async def acquire(self, name: str, **params: Any) -> Any:
        return self.runtime

    async def _release_impl(self, name: str) -> None:
        self._active_runtimes.pop(name, None)

    async def get_runtime_info(self, name: str) -> dict[str, Any] | None:
        self.info_calls.append(name)
        if not self.runtime_infos:
            return None
        return self.runtime_infos.pop(0)


class _NonRefreshingProvider(RuntimeProvider):
    def __init__(self, runtime: _FakeRuntime) -> None:
        super().__init__()
        self.runtime = runtime

    async def acquire(self, name: str, **params: Any) -> Any:
        return self.runtime

    async def _release_impl(self, name: str) -> None:
        return None


def _service(*, timeout_s: float = 1.0, poll_interval_s: float = 0.0) -> AidataService:
    return AidataService(
        name='mock-server-maimai',
        runtime_name='service_mock_server_maimai',
        exposed_ports=[20015],
        ready_timeout_s=timeout_s,
        ready_poll_interval_s=poll_interval_s,
        connections=[
            AidataServiceConnection(
                placeholder='__MS_MAIMAI_URL__',
                internal_port=20015,
                value_template='http://{host}:{port}/mcp',
            )
        ],
    )


@pytest.mark.asyncio
async def test_setup_services_refreshes_delayed_port_mapping() -> None:
    runtime = _FakeRuntime()
    provider = _RefreshingProvider(
        runtime,
        runtime_infos=[
            {'status': 'scheduled', 'port_mappings': {}},
            {'status': 'scheduled', 'port_mappings': {'20015': 'http://sandbox.example:32015'}},
        ],
    )
    ctx = ExecutionContext(runtime_provider=provider)

    async with ctx:
        replacements = await setup_services([_service()])

    assert provider.info_calls == ['service_mock_server_maimai', 'service_mock_server_maimai']
    assert runtime.port_mappings == {20015: 'http://sandbox.example:32015'}
    assert replacements.text == {'__MS_MAIMAI_URL__': 'http://sandbox.example:32015/mcp'}


@pytest.mark.asyncio
async def test_setup_services_does_not_refresh_existing_port_mapping() -> None:
    runtime = _FakeRuntime({20015: 'http://sandbox.example:32015'})
    provider = _RefreshingProvider(runtime, runtime_infos=[])
    ctx = ExecutionContext(runtime_provider=provider)

    async with ctx:
        replacements = await setup_services([_service()])

    assert provider.info_calls == []
    assert replacements.text == {'__MS_MAIMAI_URL__': 'http://sandbox.example:32015/mcp'}


@pytest.mark.asyncio
async def test_setup_services_preserves_missing_mapping_error_after_refresh_timeout() -> None:
    runtime = _FakeRuntime()
    provider = _RefreshingProvider(runtime, runtime_infos=[{'status': 'scheduled', 'port_mappings': {}}])
    ctx = ExecutionContext(runtime_provider=provider)

    with pytest.raises(RuntimeError, match=r'missing port mapping for internal_port=20015; available=\[\]'):
        async with ctx:
            await setup_services([_service(timeout_s=0.0)])

    assert provider.info_calls == ['service_mock_server_maimai']


@pytest.mark.asyncio
async def test_setup_services_waits_for_all_required_port_mappings() -> None:
    runtime = _FakeRuntime()
    provider = _RefreshingProvider(
        runtime,
        runtime_infos=[
            {'port_mappings': {'20015': 'http://sandbox.example:32015'}},
            {'port_mappings': {'20008': 'http://sandbox.example:32008'}},
        ],
    )
    service = AidataService(
        name='mock-servers',
        runtime_name='service_mock_servers',
        exposed_ports=[20015, 20008],
        ready_timeout_s=1.0,
        ready_poll_interval_s=0.0,
        connections=[
            AidataServiceConnection(placeholder='__MAIMAI__', internal_port=20015),
            AidataServiceConnection(placeholder='__NOTES__', internal_port=20008),
        ],
    )
    ctx = ExecutionContext(runtime_provider=provider)

    async with ctx:
        replacements = await setup_services([service])

    assert provider.info_calls == ['service_mock_servers', 'service_mock_servers']
    assert runtime.port_mappings == {
        20015: 'http://sandbox.example:32015',
        20008: 'http://sandbox.example:32008',
    }
    assert replacements.text == {
        '__MAIMAI__': 'http://sandbox.example:32015',
        '__NOTES__': 'http://sandbox.example:32008',
    }


@pytest.mark.asyncio
async def test_setup_services_fails_immediately_when_provider_cannot_refresh() -> None:
    runtime = _FakeRuntime()
    provider = _NonRefreshingProvider(runtime)
    ctx = ExecutionContext(runtime_provider=provider)

    with pytest.raises(RuntimeError, match=r'missing port mapping for internal_port=20015; available=\[\]'):
        async with ctx:
            await setup_services([_service(timeout_s=30.0)])
