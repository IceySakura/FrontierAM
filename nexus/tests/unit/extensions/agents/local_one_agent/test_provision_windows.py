"""Windows/WAA provisioning 测试（不连真实 WAA）。

patch 掉 WaaBackend 构造与 require_context，用 AsyncMock 断言：
- skill：本地 skill 目录逐文件上传到 VM 的 skill_path；
- warm：在 VM 内跑 pip/npm；cold 不跑库安装。

node/python runtime 由 VM 镜像预装，provisioning 不再上传 runtime。
"""

from __future__ import annotations

from types import SimpleNamespace
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
from nexus.extensions.agents.local_one_agent.waa_backend import WaaBackend
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


_MOD = 'nexus.extensions.agents.local_one_agent.agent'


def _ctx(env: LocalEnv, skill_path: str | None = None, windows_sandbox: bool = True) -> LocalContext:
    return LocalContext(
        workspace=LocalWorkspace(path=r'C:\Users\dev\project', scope='conversation'),
        permission=LocalPermission(level='basic'),
        runtime=LocalRuntime(os='windows', arch='x86_64', shell='powershell'),
        env=env,
        skill_path=skill_path,
        windows_sandbox=windows_sandbox,
    )


def _fake_backend() -> WaaBackend:
    backend = WaaBackend.__new__(WaaBackend)
    backend._raw = AsyncMock()
    backend._nexus = AsyncMock()
    backend.run_command = AsyncMock(
        return_value=RunCommandResponse(status=CommandRunStatus.Finished, return_code=0, stdout='ok')
    )
    backend.upload_file = AsyncMock()
    backend.write_file = AsyncMock()
    backend.create_folder = AsyncMock()
    return backend


def _patches(backend: WaaBackend):
    ctx = SimpleNamespace(get_runtime=AsyncMock(return_value=object()))
    return (
        patch(f'{_MOD}.WaaBackend', return_value=backend),
        patch(f'{_MOD}.require_context', return_value=ctx),
    )


class TestSkillProvisioning:
    @pytest.mark.asyncio
    async def test_no_skill_path_no_upload(self):
        backend = _fake_backend()
        p1, p2 = _patches(backend)
        with p1, p2:
            await _provision_env(_ctx(LocalEnv(mode='cold'), skill_path=None))
        backend.upload_file.assert_not_awaited()


class TestYbsbrProvisioning:
    @pytest.mark.asyncio
    async def test_profiles_uploaded_and_init_dirs_created(self):
        backend = _fake_backend()
        p1, p2 = _patches(backend)
        with p1, p2:
            await _provision_env(_ctx(LocalEnv(mode='cold')))

        # 两个 profile 走 write_file（生成内容）
        written = [c.args[0] for c in backend.write_file.await_args_list]
        assert any(t.endswith('profile_main.jsonc') for t in written)
        assert any(t.endswith('profile_net.jsonc') for t in written)
        # init 目录（含 profiles 目录）走 create_folder
        folders = [c.args[0] for c in backend.create_folder.await_args_list]
        assert any(f.endswith(r'\profiles') for f in folders)

    @pytest.mark.asyncio
    async def test_disabled_skips_profiles(self):
        backend = _fake_backend()
        p1, p2 = _patches(backend)
        with p1, p2:
            await _provision_env(_ctx(LocalEnv(mode='cold'), windows_sandbox=False))
        backend.write_file.assert_not_awaited()


class TestWarmDeps:
    @pytest.mark.asyncio
    async def test_warm_runs_pip_and_npm_in_vm(self):
        backend = _fake_backend()
        p1, p2 = _patches(backend)
        env = LocalEnv(mode='warm', python_requirements='requirements.txt', node_manifest='sub/package.json')
        with p1, p2:
            await _provision_env(_ctx(env))

        commands = [c.args[0] for c in backend.run_command.await_args_list]
        assert any('pip install -r' in c and 'requirements.txt' in c for c in commands)
        assert any('npm install' in c for c in commands)

    @pytest.mark.asyncio
    async def test_cold_skips_pip_npm(self):
        backend = _fake_backend()
        p1, p2 = _patches(backend)
        env = LocalEnv(mode='cold', python_requirements='requirements.txt', node_manifest='package.json')
        with p1, p2:
            await _provision_env(_ctx(env))

        commands = [c.args[0] for c in backend.run_command.await_args_list]
        assert not any('pip install' in c for c in commands)
        assert not any('npm install' in c for c in commands)


class TestWorkspaceSeed:
    @pytest.mark.asyncio
    async def test_local_seed_dir_uploaded_to_workspace(self, tmp_path):
        seed = tmp_path / 'seed'
        (seed / 'sub').mkdir(parents=True)
        (seed / 'requirements.txt').write_text('six\n')
        (seed / 'sub' / 'data.txt').write_text('x')
        backend = _fake_backend()
        p1, p2 = _patches(backend)
        env = LocalEnv(mode='warm', python_requirements='requirements.txt', local_seed_dir=str(seed))
        with p1, p2:
            await _provision_env(_ctx(env))

        targets = [c.args[0] for c in backend.upload_file.await_args_list]
        assert r'C:\Users\dev\project\requirements.txt' in targets
        assert r'C:\Users\dev\project\sub\data.txt' in targets

    @pytest.mark.asyncio
    async def test_no_seed_dir_no_workspace_upload(self):
        backend = _fake_backend()
        p1, p2 = _patches(backend)
        with p1, p2:
            await _provision_env(_ctx(LocalEnv(mode='cold')))
        # 无 seed dir、无 skill → 不应有任何 workspace 文件上传（profile 走 write_file，不计入）
        backend.upload_file.assert_not_awaited()
