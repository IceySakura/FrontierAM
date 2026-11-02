"""ybsbr Windows 沙箱测试（不连真实 WAA）。

覆盖:profile 渲染（readonly/basic/网络档）、profile/init 目录清单、命令包装（含 install 选放网 profile）、
探测可用性与降级。
"""

from __future__ import annotations

import json

from unittest.mock import AsyncMock

import pytest

from nexus.extensions.agents.local_one_agent.context import LocalPermission, LocalWorkspace
from nexus.extensions.agents.local_one_agent.windows_sandbox import (
    MAIN_PROFILE_PATH,
    NET_PROFILE_PATH,
    YBSBR_EXE,
    WindowsSandbox,
    build_profile,
)
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


_WS = r'C:\Users\dev\project'


def _sandbox(level: str = 'basic', enabled: bool = True) -> WindowsSandbox:
    return WindowsSandbox(
        LocalWorkspace(path=_WS, scope='conversation'),
        LocalPermission(level=level),
        enabled=enabled,
    )


class TestBuildProfile:
    def test_basic_grants_workspace_write(self):
        profile = json.loads(build_profile(_WS, 'basic', 'forbidden'))
        perms = profile['exec']['file_permissions']
        assert perms == [{'path': _WS, 'access': 'write'}]
        assert profile['exec']['cwd'] == _WS
        assert profile['exec']['network_mode'] == 'forbidden'
        assert profile['exec']['use_sandbox'] is True
        assert profile['exec']['windows_sandbox_level'] == 'restricted_token'
        assert profile['exec']['windows_sandbox_private_desktop'] is False

    def test_readonly_has_no_writable_dirs(self):
        profile = json.loads(build_profile(_WS, 'readonly', 'forbidden'))
        assert profile['exec']['file_permissions'] == []

    def test_network_mode_threaded(self):
        assert json.loads(build_profile(_WS, 'basic', 'allow_all'))['exec']['network_mode'] == 'allow_all'

    def test_init_dirs_absolute(self):
        init = json.loads(build_profile(_WS, 'basic', 'forbidden'))['init']
        assert all(v.startswith('C:\\') for v in init.values())

    def test_forward_slash_workspace_normalized(self):
        profile = json.loads(build_profile('C:/Users/dev/project', 'basic', 'forbidden'))
        assert profile['exec']['cwd'] == _WS


class TestProvisioningInputs:
    def test_iter_profiles_main_and_net(self):
        pairs = dict(_sandbox('basic').iter_profiles())
        assert set(pairs) == {MAIN_PROFILE_PATH, NET_PROFILE_PATH}
        assert json.loads(pairs[MAIN_PROFILE_PATH])['exec']['network_mode'] == 'forbidden'
        assert json.loads(pairs[NET_PROFILE_PATH])['exec']['network_mode'] == 'allow_all'
        # 两档文件权限一致（install 只放网、不放文件权限,对齐 Linux）
        assert (
            json.loads(pairs[MAIN_PROFILE_PATH])['exec']['file_permissions']
            == json.loads(pairs[NET_PROFILE_PATH])['exec']['file_permissions']
        )

    def test_init_dirs_include_profiles_dir(self):
        dirs = WindowsSandbox.init_dirs()
        assert any(d.endswith(r'\profiles') for d in dirs)
        assert len(dirs) == 4


class TestSelectAndWrap:
    def test_select_profile_install_uses_net(self):
        sb = _sandbox()
        assert sb.select_profile('pip install six') == NET_PROFILE_PATH
        assert sb.select_profile('npm install') == NET_PROFILE_PATH
        assert sb.select_profile('python app.py') == MAIN_PROFILE_PATH
        assert sb.select_profile('Get-ChildItem') == MAIN_PROFILE_PATH

    def test_wrap_normal_uses_main_profile(self):
        wrapped = _sandbox().wrap('powershell -NoProfile -EncodedCommand AAA', 'python app.py')
        assert (
            wrapped
            == f'call "{YBSBR_EXE}" --profile="{MAIN_PROFILE_PATH}" exec -- powershell -NoProfile -EncodedCommand AAA'
        )

    def test_wrap_install_uses_net_profile(self):
        wrapped = _sandbox().wrap('powershell -NoProfile -EncodedCommand BBB', 'pip install six')
        assert f'--profile="{NET_PROFILE_PATH}"' in wrapped
        assert wrapped.startswith(f'call "{YBSBR_EXE}"')
        assert wrapped.endswith('exec -- powershell -NoProfile -EncodedCommand BBB')


def _backend(run_return=None, raises: Exception | None = None):
    backend = AsyncMock()
    if raises is not None:
        backend.run_command = AsyncMock(side_effect=raises)
    else:
        backend.run_command = AsyncMock(return_value=run_return)
    return backend


class TestProbe:
    @pytest.mark.asyncio
    async def test_probe_success_activates(self):
        sb = _sandbox()
        ok = await sb.probe(
            _backend(RunCommandResponse(status=CommandRunStatus.Finished, return_code=0, stdout='ybsbr 1.0.0'))
        )
        assert ok is True
        assert sb.is_active() is True

    @pytest.mark.asyncio
    async def test_probe_nonzero_degrades(self):
        sb = _sandbox()
        ok = await sb.probe(
            _backend(RunCommandResponse(status=CommandRunStatus.Error, return_code=1, stderr='not found'))
        )
        assert ok is False
        assert sb.is_active() is False

    @pytest.mark.asyncio
    async def test_probe_exception_degrades(self):
        sb = _sandbox()
        assert await sb.probe(_backend(raises=RuntimeError('boom'))) is False
        assert sb.is_active() is False

    @pytest.mark.asyncio
    async def test_disabled_skips_probe(self):
        sb = _sandbox(enabled=False)
        backend = _backend(RunCommandResponse(status=CommandRunStatus.Finished, return_code=0, stdout='ybsbr'))
        assert await sb.probe(backend) is False
        backend.run_command.assert_not_awaited()
        assert sb.is_active() is False

    @pytest.mark.asyncio
    async def test_probe_cached(self):
        sb = _sandbox()
        backend = _backend(RunCommandResponse(status=CommandRunStatus.Finished, return_code=0, stdout='ok'))
        await sb.probe(backend)
        await sb.probe(backend)
        backend.run_command.assert_awaited_once()
