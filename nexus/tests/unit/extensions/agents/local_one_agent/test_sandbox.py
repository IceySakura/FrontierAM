import shutil

import pytest

from nexus.extensions.agents.local_one_agent.context import LocalPermission, LocalRuntime, LocalWorkspace
from nexus.extensions.agents.local_one_agent.sandbox import LocalSandbox


_FAKE_BINS = {'bwrap': '/usr/bin/bwrap', 'unshare': '/usr/bin/unshare'}


def _with_bwrap(monkeypatch):
    monkeypatch.setattr(shutil, 'which', lambda name: _FAKE_BINS.get(name))
    # bwrap 存在且**可用**（可用性探测打桩为 True，避免真跑 /usr/bin/bwrap）。
    monkeypatch.setattr(LocalSandbox, '_bwrap_usable', staticmethod(lambda _path: True))


@pytest.fixture
def workspace():
    return LocalWorkspace(path='/workspace/project', scope='conversation')


@pytest.fixture
def macos_runtime():
    # 训练在 Linux 容器里用 bwrap 模拟 macOS，故 os='macos' 才开沙箱（B1）。
    return LocalRuntime(os='macos', arch='arm64', shell='zsh')


@pytest.fixture
def windows_runtime():
    return LocalRuntime(os='windows', arch='x86_64', shell='powershell')


class TestLocalSandbox:
    def test_build_bwrap_command_basic(self, workspace, macos_runtime, monkeypatch):
        _with_bwrap(monkeypatch)
        sandbox = LocalSandbox(workspace=workspace, runtime=macos_runtime, permission=LocalPermission(level='basic'))
        result = sandbox.wrap_command('echo hello')
        assert 'bwrap' in result
        # root + Docker 默认 caps 场景必需，否则 make / slave 失败（gVisor 实测）。
        assert '--unshare-user' in result
        assert '--ro-bind' in result
        # --dev-bind 让 /dev/null 等设备节点在沙箱内可用。
        assert '--dev-bind /dev /dev' in result
        assert f'--bind {workspace.path} {workspace.path}' in result
        # 网络隔离改由外层 unshare 负责（bwrap --unshare-net 在 gVisor 拉 lo 失败）。
        assert 'unshare --user --map-root-user --net' in result
        assert '--unshare-net' not in result
        assert '--die-with-parent' in result
        assert 'echo hello' in result

    def test_build_bwrap_command_readonly(self, workspace, macos_runtime, monkeypatch):
        _with_bwrap(monkeypatch)
        sandbox = LocalSandbox(workspace=workspace, runtime=macos_runtime, permission=LocalPermission(level='readonly'))
        result = sandbox.wrap_command('ls -la')
        assert '--unshare-user' in result
        assert '--ro-bind' in result
        assert 'unshare --user --map-root-user --net' in result
        assert f'--bind {workspace.path}' not in result

    def test_install_escape_allows_network(self, workspace, macos_runtime, monkeypatch):
        """install-escape（Part Ⅰ §6）：安装类命令放网（不套外层网络隔离 unshare）。"""
        _with_bwrap(monkeypatch)
        sandbox = LocalSandbox(workspace=workspace, runtime=macos_runtime, permission=LocalPermission(level='basic'))
        for cmd in (
            'npm install lodash',
            'pip install requests',
            'pnpm i',
            'yarn add react',
            'python -m pip install requests',
            'python3 -m pip install requests',
        ):
            result = sandbox.wrap_command(cmd)
            assert '--map-root-user' not in result, cmd
            assert '--unshare-net' not in result, cmd

    def test_non_install_command_isolates_network(self, workspace, macos_runtime, monkeypatch):
        _with_bwrap(monkeypatch)
        sandbox = LocalSandbox(workspace=workspace, runtime=macos_runtime, permission=LocalPermission(level='basic'))
        result = sandbox.wrap_command('curl http://example.com')
        # 外层 unshare 建空 netns 实现网络隔离。
        assert 'unshare --user --map-root-user --net' in result

    def test_net_isolation_degrades_without_unshare(self, workspace, macos_runtime, monkeypatch):
        """缺 unshare 时：仍做 FS 隔离（bwrap），但不套外层网络隔离，且不阻断命令。"""
        monkeypatch.setattr(shutil, 'which', lambda name: '/usr/bin/bwrap' if name == 'bwrap' else None)
        monkeypatch.setattr(LocalSandbox, '_bwrap_usable', staticmethod(lambda _path: True))
        sandbox = LocalSandbox(workspace=workspace, runtime=macos_runtime, permission=LocalPermission(level='basic'))
        result = sandbox.wrap_command('curl http://example.com')
        assert 'bwrap' in result
        assert '--unshare-user' in result
        assert '--map-root-user' not in result

    def test_no_bwrap_on_windows(self, workspace, windows_runtime, monkeypatch):
        # B2：windows 不开 bwrap，即使容器里有 bwrap。
        _with_bwrap(monkeypatch)
        sandbox = LocalSandbox(workspace=workspace, runtime=windows_runtime, permission=LocalPermission(level='basic'))
        assert sandbox.wrap_command('echo hello') == 'echo hello'
        assert sandbox.is_available() is False

    def test_bwrap_not_installed_fallback(self, workspace, macos_runtime, monkeypatch):
        monkeypatch.setattr(shutil, 'which', lambda _name: None)
        sandbox = LocalSandbox(workspace=workspace, runtime=macos_runtime, permission=LocalPermission(level='basic'))
        assert sandbox.wrap_command('echo hello') == 'echo hello'

    def test_is_available_macos_with_bwrap(self, workspace, macos_runtime, monkeypatch):
        _with_bwrap(monkeypatch)
        sandbox = LocalSandbox(workspace=workspace, runtime=macos_runtime, permission=LocalPermission(level='basic'))
        assert sandbox.is_available() is True

    def test_is_available_macos_without_bwrap(self, workspace, macos_runtime, monkeypatch):
        monkeypatch.setattr(shutil, 'which', lambda _name: None)
        sandbox = LocalSandbox(workspace=workspace, runtime=macos_runtime, permission=LocalPermission(level='basic'))
        assert sandbox.is_available() is False

    def test_bwrap_present_but_unusable_degrades(self, workspace, macos_runtime, monkeypatch):
        """装了 bwrap 但无法创建 namespace（容器未放开 userns）→ 干净降级，不阻断、不 wrap。"""
        monkeypatch.setattr(shutil, 'which', lambda name: '/usr/bin/bwrap' if name == 'bwrap' else None)
        monkeypatch.setattr(LocalSandbox, '_bwrap_usable', staticmethod(lambda _path: False))
        sandbox = LocalSandbox(workspace=workspace, runtime=macos_runtime, permission=LocalPermission(level='basic'))
        assert sandbox.is_available() is False
        assert sandbox.wrap_command('echo hello') == 'echo hello'

    def test_tmpfs_mount(self, workspace, macos_runtime, monkeypatch):
        _with_bwrap(monkeypatch)
        sandbox = LocalSandbox(workspace=workspace, runtime=macos_runtime, permission=LocalPermission(level='basic'))
        result = sandbox.wrap_command('python script.py')
        assert '--tmpfs /tmp' in result

    def test_m4_quotes_workspace_with_spaces(self, macos_runtime, monkeypatch):
        # M4：含空格的 workspace 路径必须被 quote，不能被 ' '.join 错误拆分。
        _with_bwrap(monkeypatch)
        ws = LocalWorkspace(path='/Users/me/My Project', scope='conversation')
        sandbox = LocalSandbox(workspace=ws, runtime=macos_runtime, permission=LocalPermission(level='basic'))
        result = sandbox.wrap_command('echo hi')
        # 被 quote 后应出现 'My Project' 整体（带引号），而非裸空格拆开。
        assert "'/Users/me/My Project'" in result
        # 命令本身也整体 quote 成 sh -c 的单参数。
        assert "'echo hi'" in result
