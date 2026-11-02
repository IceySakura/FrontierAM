"""LocalToolExecutor 的 Windows 执行后端选择测试（不连真实 WAA）。

验证：os=='windows' 时懒获取 WAA runtime、构造 WaaBackend、注入 tool_state 且不注入
bwrap 沙箱；os=='macos' 时不触碰 runtime、维持 sandbox_wrap_fn 老路。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from nexus.extensions.agents.local_one_agent.context import (
    LocalContext,
    LocalPermission,
    LocalRuntime,
    LocalWorkspace,
)
from nexus.extensions.agents.local_one_agent.executor import LocalToolExecutor
from nexus.extensions.agents.local_one_agent.waa_backend import WaaBackend
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


_MOD = 'nexus.extensions.agents.local_one_agent.executor'


def _ctx(os_name: str = 'windows', shell: str = 'powershell') -> LocalContext:
    return LocalContext(
        workspace=LocalWorkspace(path=r'C:\Users\dev\project', scope='conversation'),
        permission=LocalPermission(level='basic'),
        runtime=LocalRuntime(os=os_name, arch='x86_64', shell=shell),
    )


def _fake_backend() -> WaaBackend:
    backend = WaaBackend.__new__(WaaBackend)
    backend._raw = AsyncMock()
    backend._nexus = AsyncMock()
    backend._sandbox = None
    # executor 会对新 backend 跑 ybsbr 探测（sandbox.probe → backend.run_command）;给一个干净返回。
    backend.run_command = AsyncMock(
        return_value=RunCommandResponse(status=CommandRunStatus.Finished, return_code=0, stdout='ybsbr 1.0.0')
    )
    return backend


def _runtime_ctx() -> SimpleNamespace:
    return SimpleNamespace(
        run_id='r',
        tool_configs={},
        tool_state={},
        trace_id=None,
        get_runtime=AsyncMock(return_value=object()),
    )


class TestEnsureWaaBackend:
    @pytest.mark.asyncio
    async def test_windows_builds_backend(self):
        executor = LocalToolExecutor([], local_context=_ctx('windows'))
        fake = _fake_backend()
        runtime_ctx = _runtime_ctx()
        with (
            patch(f'{_MOD}.require_context', return_value=runtime_ctx),
            patch(f'{_MOD}.WaaBackend', return_value=fake) as mk,
        ):
            await executor._ensure_waa_backend()

        assert executor._waa_backend is fake
        # 命令 shell 固定 PowerShell，构造时不再传 shell 参数
        mk.assert_called_once_with(runtime_ctx.get_runtime.return_value)
        assert 'shell' not in mk.call_args.kwargs

    @pytest.mark.asyncio
    async def test_windows_backend_cached(self):
        executor = LocalToolExecutor([], local_context=_ctx('windows'))
        fake = _fake_backend()
        with (
            patch(f'{_MOD}.require_context', return_value=_runtime_ctx()),
            patch(f'{_MOD}.WaaBackend', return_value=fake) as mk,
        ):
            await executor._ensure_waa_backend()
            await executor._ensure_waa_backend()
        mk.assert_called_once()

    @pytest.mark.asyncio
    async def test_macos_never_touches_runtime(self):
        executor = LocalToolExecutor([], local_context=_ctx('macos', shell='zsh'))
        with patch(f'{_MOD}.require_context') as rc:
            await executor._ensure_waa_backend()
        rc.assert_not_called()
        assert executor._waa_backend is None


class TestBuildToolContextBackendSelection:
    def test_windows_injects_waa_backend_no_sandbox(self):
        executor = LocalToolExecutor([], local_context=_ctx('windows'))
        executor._waa_backend = _fake_backend()
        with patch(f'{_MOD}.require_context', return_value=_runtime_ctx()):
            tool_ctx = executor._build_tool_context()

        assert isinstance(tool_ctx.tool_state.get('waa_backend'), WaaBackend)
        assert 'sandbox_wrap_fn' not in tool_ctx.tool_state

    def test_macos_injects_sandbox_no_waa(self):
        executor = LocalToolExecutor([], local_context=_ctx('macos', shell='zsh'))
        with (
            patch(f'{_MOD}.require_context', return_value=_runtime_ctx()),
            patch.object(LocalToolExecutor, 'sandbox') as sandbox_prop,
        ):
            sandbox_prop.is_available.return_value = True
            tool_ctx = executor._build_tool_context()

        assert 'waa_backend' not in tool_ctx.tool_state
        assert 'sandbox_wrap_fn' in tool_ctx.tool_state
