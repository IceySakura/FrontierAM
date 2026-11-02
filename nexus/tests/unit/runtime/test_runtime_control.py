"""Tests for RuntimeControl contextvar."""

from __future__ import annotations

from typing import Any

import pytest

from nexus.runtime.context import (
    RuntimeControl,
    get_runtime_control,
    require_runtime_control,
    set_runtime_control,
)


class _SharedValueSIO:
    def __init__(self, ctrl: RuntimeControl) -> None:
        self.ctrl = ctrl
        self.requests = 0

    async def emit(self, event: str, payload: dict[str, Any]) -> None:
        assert event == 'shared-value:request'
        self.requests += 1
        self.ctrl.provide_shared_value(payload['request_id'], {'value': 'from-host'})


class TestRuntimeControlContextvar:
    def test_default_is_none(self):
        assert get_runtime_control() is None

    def test_require_raises_when_not_set(self):
        with pytest.raises(RuntimeError, match='No RuntimeControl available'):
            require_runtime_control()

    def test_set_and_get(self):
        ctrl = RuntimeControl()
        token = set_runtime_control(ctrl)
        try:
            assert get_runtime_control() is ctrl
            assert require_runtime_control() is ctrl
        finally:
            # Reset to None (cleanup)
            from nexus.runtime.context import _runtime_control

            _runtime_control.reset(token)

    def test_runtime_control_has_executor_methods(self):
        """RuntimeControl should expose key executor convenience methods."""
        ctrl = RuntimeControl()
        assert hasattr(ctrl, 'run_command')
        assert hasattr(ctrl, 'create_ipython_session')
        assert hasattr(ctrl, 'create_browser_session')
        assert hasattr(ctrl, 'mcp_start_proxy')
        assert hasattr(ctrl, 'call_executor')


@pytest.mark.asyncio
async def test_runtime_shared_value_is_requested_once_and_cached() -> None:
    ctrl = RuntimeControl()
    sio = _SharedValueSIO(ctrl)
    ctrl.sio = sio

    first = await ctrl.get_shared_value('example', dict, decode=dict)
    second = await ctrl.get_shared_value('example', dict, decode=dict)

    assert first == {'value': 'from-host'}
    assert second is first
    assert sio.requests == 1


class DummySIO:
    async def call(self, event, data=None, namespace=None, **kwargs):  # noqa: ANN001, ANN202, ARG002
        return {'native': True}

    def on(self, event, handler=None, namespace=None):  # noqa: ANN001, ANN202, ARG002
        return handler


def test_siobridge_delegates_call_to_native_sio() -> None:
    import asyncio

    from nexus.runtime.context import SIOBridge

    bridge = SIOBridge(DummySIO(), namespace='/worker')

    result = asyncio.run(bridge.call('some_event', {'x': 1}))

    assert result == {'native': True}
