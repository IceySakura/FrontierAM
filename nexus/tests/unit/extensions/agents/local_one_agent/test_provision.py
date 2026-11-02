"""Tests for local_one_agent 的环境依赖预置钩子（cold/warm）。"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from nexus.extensions.agents.local_one_agent.agent import _provision_env
from nexus.extensions.agents.local_one_agent.context import (
    LocalContext,
    LocalEnv,
    LocalPermission,
    LocalRuntime,
    LocalWorkspace,
)


def _ctx(ws_path: str, env: LocalEnv) -> LocalContext:
    return LocalContext(
        workspace=LocalWorkspace(path=ws_path, scope='conversation'),
        permission=LocalPermission(level='basic'),
        runtime=LocalRuntime(os='macos', arch='arm64', shell='zsh'),
        env=env,
    )


class _FakeProc:
    returncode = 0

    async def communicate(self):
        return (b'ok', None)


class TestProvisionEnv:
    @pytest.mark.asyncio
    async def test_cold_is_noop(self, tmp_path):
        ctx = _ctx(str(tmp_path), LocalEnv(mode='cold', python_requirements='requirements.txt'))
        with patch('asyncio.create_subprocess_exec', new=AsyncMock()) as spawn:
            await _provision_env(ctx)
            spawn.assert_not_called()

    @pytest.mark.asyncio
    async def test_warm_missing_files_no_crash(self, tmp_path):
        ctx = _ctx(str(tmp_path), LocalEnv(mode='warm', python_requirements='nope.txt', node_manifest='nope.json'))
        with patch('asyncio.create_subprocess_exec', new=AsyncMock()) as spawn:
            await _provision_env(ctx)
            spawn.assert_not_called()

    @pytest.mark.asyncio
    async def test_warm_installs_python_requirements(self, tmp_path):
        (tmp_path / 'requirements.txt').write_text('openpyxl\n', encoding='utf-8')
        ctx = _ctx(str(tmp_path), LocalEnv(mode='warm', python_requirements='requirements.txt'))
        with patch('asyncio.create_subprocess_exec', new=AsyncMock(return_value=_FakeProc())) as spawn:
            await _provision_env(ctx)
            spawn.assert_called_once()
            args = spawn.call_args.args
            assert args[:4] == ('python3', '-m', 'pip', 'install')
            assert str(tmp_path / 'requirements.txt') in args

    @pytest.mark.asyncio
    async def test_warm_installs_node_manifest(self, tmp_path):
        (tmp_path / 'package.json').write_text('{"dependencies":{}}', encoding='utf-8')
        ctx = _ctx(str(tmp_path), LocalEnv(mode='warm', node_manifest='package.json'))
        with patch('asyncio.create_subprocess_exec', new=AsyncMock(return_value=_FakeProc())) as spawn:
            await _provision_env(ctx)
            spawn.assert_called_once()
            assert spawn.call_args.args[:2] == ('npm', 'install')
