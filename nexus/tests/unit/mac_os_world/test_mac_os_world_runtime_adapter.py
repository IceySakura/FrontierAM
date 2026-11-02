from __future__ import annotations

import asyncio

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_runtime_adapter_delegates_non_screenshot_executors() -> None:
    from nexus.extensions.tasks.mac_os_world.runtime_adapter import MacOSWorldRuntimeAdapter

    calls = []

    class Runtime:
        async def call_executor(self, executor_name, arg):
            calls.append((executor_name, arg))
            return {'returncode': 0}

    adapter = MacOSWorldRuntimeAdapter(Runtime(), screenshot_timeout=0.1)

    result = await adapter.call_executor('run_command', {'command': 'echo ok'})

    assert result == {'returncode': 0}
    assert calls == [('run_command', {'command': 'echo ok'})]


@pytest.mark.asyncio
async def test_runtime_adapter_bounds_only_screenshot_wall_clock() -> None:
    from nexus.extensions.tasks.mac_os_world.runtime_adapter import MacOSWorldRuntimeAdapter

    cancelled = asyncio.Event()

    class Runtime:
        async def gui_screenshot(self):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

    adapter = MacOSWorldRuntimeAdapter(Runtime(), screenshot_timeout=0.01)

    result = await adapter.call_executor('gui_screenshot', {})

    assert result['success'] is False
    assert result['image'] is None
    assert result['format'] == 'png'
    assert result['error'] == 'macOSWorld screenshot exceeded 0.01s'
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_runtime_adapter_preserves_runtime_lifecycle_and_identity() -> None:
    from nexus.extensions.tasks.mac_os_world.runtime_adapter import MacOSWorldRuntimeAdapter

    class Runtime:
        base_url = 'http://runtime.example'
        metadata = {'assetId': 'asset-1'}
        port_mappings = {5000: 'http://runtime.example'}
        default_venv_id = 'venv-1'

        async def drain_trace(self, timeout=None):
            return ('drained', timeout)

    runtime = Runtime()
    adapter = MacOSWorldRuntimeAdapter(runtime, screenshot_timeout=20)

    assert adapter.wrapped_runtime is runtime
    assert adapter.base_url == 'http://runtime.example'
    assert adapter.metadata == {'assetId': 'asset-1'}
    assert adapter.port_mappings == {5000: 'http://runtime.example'}
    assert adapter.default_venv_id == 'venv-1'
    assert await adapter.drain_trace(timeout=3) == ('drained', 3)


def test_runtime_adapter_temporarily_replaces_only_requested_cached_runtime() -> None:
    from nexus.extensions.tasks.mac_os_world.runtime_adapter import (
        MacOSWorldRuntimeAdapter,
        mac_os_world_runtime_adapter,
    )

    original = object()
    other = object()
    ctx = SimpleNamespace(_runtimes={'main': original, 'other': other})

    with mac_os_world_runtime_adapter(ctx, 'main', original, screenshot_timeout=20) as adapter:
        assert isinstance(adapter, MacOSWorldRuntimeAdapter)
        assert ctx._runtimes == {'main': adapter, 'other': other}
    assert ctx._runtimes == {'main': original, 'other': other}


def test_runtime_adapter_does_not_overwrite_runtime_replaced_inside_scope() -> None:
    from nexus.extensions.tasks.mac_os_world.runtime_adapter import mac_os_world_runtime_adapter

    original = object()
    replacement = object()
    ctx = SimpleNamespace(_runtimes={'main': original})

    with mac_os_world_runtime_adapter(ctx, 'main', original, screenshot_timeout=20):
        ctx._runtimes['main'] = replacement

    assert ctx._runtimes['main'] is replacement


@pytest.mark.asyncio
async def test_runtime_adapter_restores_original_after_screenshot_timeout() -> None:
    from nexus.extensions.tasks.mac_os_world.runtime_adapter import mac_os_world_runtime_adapter

    class Runtime:
        async def gui_screenshot(self):
            await asyncio.sleep(60)

    original = Runtime()
    ctx = SimpleNamespace(_runtimes={'main': original})

    with mac_os_world_runtime_adapter(ctx, 'main', original, screenshot_timeout=0.01) as adapter:
        response = await adapter.gui_screenshot()
        assert response.success is False
        assert ctx._runtimes['main'] is adapter

    assert ctx._runtimes['main'] is original
