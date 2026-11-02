"""User-facing pause / resume on GongfengRuntimeProvider.

Callers pass the runtime name used with acquire(). The default name is ``main``.
The provider looks up the sandbox id and sends that as ``runtime_id``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from nexus.runtime.providers.gongfeng import GongfengRuntimeProvider, SandboxApiError


def _make_provider(**overrides: Any) -> GongfengRuntimeProvider:
    kwargs: dict[str, Any] = {
        'api_base_url': 'http://gongfeng.example.com',
        'token': 'unit-test-token',
        'image': 'unit-test/image:latest',
        'print_port_mappings': False,
    }
    kwargs.update(overrides)
    return GongfengRuntimeProvider(**kwargs)


@pytest.mark.asyncio
async def test_pause_defaults_to_main() -> None:
    provider = _make_provider()
    provider._runtime_ids['main'] = 'nexus-abc'
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={'code': 0, 'data': {'status': 'paused', 'runtime_id': 'nexus-abc'}}
    )

    result = await provider.pause()

    assert result == {'status': 'paused', 'runtime_id': 'nexus-abc'}
    provider._call_api.assert_awaited_once_with(
        '/sandbox/v3/runtime/pause',
        {'runtime_id': 'nexus-abc'},
    )


@pytest.mark.asyncio
async def test_resume_uses_the_given_name() -> None:
    provider = _make_provider()
    provider._runtime_ids['worker'] = 'nexus-worker'
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={'code': 0, 'data': {'status': 'scheduled', 'runtime_id': 'nexus-worker'}}
    )

    result = await provider.resume('worker')

    assert result == {'status': 'scheduled', 'runtime_id': 'nexus-worker'}
    provider._call_api.assert_awaited_once_with(
        '/sandbox/v3/runtime/resume',
        {'runtime_id': 'nexus-worker'},
    )


@pytest.mark.asyncio
async def test_resume_reconnects_socketio_after_the_api_call() -> None:
    provider = _make_provider()
    provider._runtime_ids['main'] = 'nexus-abc'
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={'code': 0, 'data': {'status': 'scheduled', 'runtime_id': 'nexus-abc'}}
    )
    runtime = AsyncMock()
    provider._active_runtimes['main'] = runtime

    await provider.resume()

    runtime.ws_connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_skips_socketio_when_the_provider_has_no_client() -> None:
    provider = _make_provider()
    provider._runtime_ids['main'] = 'nexus-abc'
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={'code': 0, 'data': {'status': 'scheduled', 'runtime_id': 'nexus-abc'}}
    )

    result = await provider.resume()

    assert result == {'status': 'scheduled', 'runtime_id': 'nexus-abc'}


@pytest.mark.asyncio
async def test_resume_does_not_reconnect_when_the_api_fails() -> None:
    provider = _make_provider()
    provider._runtime_ids['main'] = 'nexus-abc'
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        side_effect=SandboxApiError(
            'rejected',
            code=260422,
            reason='Runtime.PauseNotSupported',
            msg='resume is not supported',
        )
    )
    runtime = AsyncMock()
    provider._active_runtimes['main'] = runtime

    with pytest.raises(SandboxApiError):
        await provider.resume()

    runtime.ws_connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_pause_unknown_name_does_not_call_api() -> None:
    provider = _make_provider()
    provider._call_api = AsyncMock()  # type: ignore[method-assign]

    with pytest.raises(ValueError, match='main'):
        await provider.pause()

    provider._call_api.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_pause_surfaces_sandbox_api_error() -> None:
    provider = _make_provider()
    provider._runtime_ids['main'] = 'nexus-abc'
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        side_effect=SandboxApiError(
            'rejected',
            code=260422,
            reason='Runtime.PauseNotSupported',
            msg='pause is not supported',
        )
    )

    with pytest.raises(SandboxApiError) as exc_info:
        await provider.pause()

    assert exc_info.value.code == 260422
    assert exc_info.value.reason == 'Runtime.PauseNotSupported'
