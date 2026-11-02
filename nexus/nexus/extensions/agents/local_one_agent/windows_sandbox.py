"""ybsbr sandbox wrapper for Windows/WAA bash commands.

Windows 版的 OS 级隔离,对标 Linux 的 ``sandbox.py``（bwrap）:

- Linux：``bwrap <args> -- sh -c "<cmd>"``,围栏走命令行参数（``--ro-bind`` / ``--bind``）。
- Windows：``ybsbr.exe --profile=<jsonc> exec -- <cmd>``,围栏走 **profile 文件**。

因 workspace 每 task 不同,profile 内容随之不同,故不能用静态文件——由 ``build_profile`` 按
workspace + 权限档实时渲染,provisioning 阶段上传到 VM 固定目录（见 ``agent.py``）,命令执行时用
``--profile=<VM 路径>`` 引用。

隔离语义与 Linux 1:1 对齐:
- ``basic`` → workspace 可写（``file_permissions``）;``readonly`` → 无可写目录。
- 默认断网（``network_mode: forbidden``）;命中安装类命令切「放网 profile」（对齐 Linux install-escape,
  **只放网、不放文件权限**——pip/npm 仍只能装进 workspace）。

缺失/不可用时优雅降级为「仅权限引擎」（探测 ``ybsbr --version`` 失败则不包装,命令直接执行）。
"""

from __future__ import annotations

import json
import logging

from typing import TYPE_CHECKING, Literal

from nexus.extensions.agents.local_one_agent.context import LocalPermission, LocalWorkspace
from nexus.extensions.agents.local_one_agent.sandbox import _is_install_command


if TYPE_CHECKING:
    from nexus.extensions.agents.local_one_agent.waa_backend import WaaBackend


logger = logging.getLogger(__name__)

# VM 内 ybsbr 落点（镜像预装,exe 与 yb_sandbox_runtime.dll 同目录）。
YBSBR_EXE = r'C:\Program Files\cli-dist\ybsbr.exe'

# 沙箱私有目录（独立于 workspace,避免模型 ls 工作区时看到沙箱内部文件）。
YBSBR_BASE_DIR = r'C:\Users\Public\oneagent_ybsbr'
_PROFILES_DIR = YBSBR_BASE_DIR + r'\profiles'
_INIT_APP_DIR = YBSBR_BASE_DIR + r'\app'
_INIT_TMP_DIR = YBSBR_BASE_DIR + r'\tmp'
_INIT_DATA_DIR = YBSBR_BASE_DIR + r'\data'

# 两个 profile:仅 network_mode 不同（主档断网,net 档放网供安装命令用）。
MAIN_PROFILE_PATH = _PROFILES_DIR + r'\profile_main.jsonc'
NET_PROFILE_PATH = _PROFILES_DIR + r'\profile_net.jsonc'

# 装库镜像源（加速 pip/npm,无害；对非安装命令也无副作用）。
_MIRROR_ENV = {
    'NPM_CONFIG_REGISTRY': 'https://mirrors.cloud.tencent.com/npm/',
    'PIP_INDEX_URL': 'https://mirrors.cloud.tencent.com/pypi/simple/',
}

NetworkMode = Literal['allow_all', 'forbidden', 'proxy']


def _norm_win(path: str) -> str:
    """归一到 Windows 反斜杠绝对路径口径。"""
    return path.replace('/', '\\')


def build_profile(
    workspace_path: str,
    level: Literal['readonly', 'basic'],
    network_mode: NetworkMode,
) -> str:
    """按 workspace + 权限档 + 网络模式渲染 ybsbr profile（返回 JSON 文本,合法 jsonc）。

    - ``basic``：workspace 可写；``readonly``：无可写目录（越界写由 ybsbr RestrictedToken 挡）。
    - ``restricted_token`` 档:系统目录只读/可执行（powershell/python/node 可跑）,写受限于
      ``file_permissions``。
    - ``private_desktop`` 关:bash 无 GUI,私有桌面可能让子进程见不到桌面而卡,求稳。
    """
    ws = _norm_win(workspace_path)
    file_permissions = [{'path': ws, 'access': 'write'}] if level == 'basic' else []
    profile = {
        'init': {
            'app_dir': _INIT_APP_DIR,
            'global_tmp_dir': _INIT_TMP_DIR,
            'sdk_data_dir': _INIT_DATA_DIR,
        },
        'exec': {
            'use_sandbox': True,
            'cwd': ws,
            'timeout_ms': 0,
            'network_mode': network_mode,
            'env_overrides': dict(_MIRROR_ENV),
            'env_path_overrides': [],
            'file_permissions': file_permissions,
            'windows_sandbox_level': 'restricted_token',
            'windows_sandbox_private_desktop': False,
        },
    }
    return json.dumps(profile, indent=2, ensure_ascii=False)


class WindowsSandbox:
    """把 bash 命令用 ybsbr 包装（Windows OS 级隔离）,对标 ``LocalSandbox``。

    - 惰性探测:首个命令执行前跑一次 ``ybsbr --version``,不可用则整体降级为不包装。
    - profile / init 目录内容由本类计算,provisioning 负责真正建目录 + 上传（见 agent.py）。
    """

    def __init__(
        self,
        workspace: LocalWorkspace,
        permission: LocalPermission,
        *,
        enabled: bool = True,
    ) -> None:
        self._workspace = workspace
        self._permission = permission
        self._enabled = enabled
        # 三态可用性:None=未探测,True/False=探测结论（缓存）。
        self._available: bool | None = None

    # ---- provisioning 侧:告诉 agent.py 要建哪些目录、上传哪些 profile ----

    @staticmethod
    def init_dirs() -> list[str]:
        """ybsbr 运行所需目录（含 profile 目录）,provisioning 逐个 create_folder。"""
        return [_INIT_APP_DIR, _INIT_TMP_DIR, _INIT_DATA_DIR, _PROFILES_DIR]

    def iter_profiles(self) -> list[tuple[str, str]]:
        """返回 ``[(VM 路径, 内容)]``,provisioning 逐个 write_file 上传。"""
        level = self._permission.level
        ws = self._workspace.path
        return [
            (MAIN_PROFILE_PATH, build_profile(ws, level, 'forbidden')),
            (NET_PROFILE_PATH, build_profile(ws, level, 'allow_all')),
        ]

    # ---- 执行侧:探测可用性 + 包装命令 ----

    @property
    def available(self) -> bool | None:
        """探测结论（None=未探测）。"""
        return self._available

    async def probe(self, backend: WaaBackend) -> bool:
        """跑一次 ``ybsbr --version`` 探测可用性并缓存。

        必须在 backend 尚未注入本 sandbox 时调用,否则会被自身 wrap 递归包装。
        缺失/失败即降级——训练侧可接受「仅权限引擎」这一明确档位。
        """
        if not self._enabled:
            self._available = False
            return False
        if self._available is not None:
            return self._available
        try:
            # 探测经 backend.run_command → 被 PowerShell -EncodedCommand 承载,故用 PowerShell 的调用
            # 操作符 & 而非 cmd 的 call（run_command 恒走 powershell,与 wrap 的 cmd 口径不同）。
            resp = await backend.run_command(f'& "{YBSBR_EXE}" --version')
        except Exception as exc:  # noqa: BLE001  探测失败即降级,不阻断
            logger.warning('ybsbr 探测异常,Windows 沙箱降级为仅权限引擎: %s', exc)
            self._available = False
            return False
        ok = resp.return_code in (0, None) and bool((resp.stdout or '').strip())
        self._available = ok
        if ok:
            logger.info('ybsbr 可用,Windows 沙箱开启: %s', (resp.stdout or '').strip())
        else:
            logger.warning(
                'ybsbr 不可用（rc=%s）,Windows 沙箱降级为仅权限引擎: %s',
                resp.return_code,
                (resp.stderr or resp.stdout or '')[-200:],
            )
        return ok

    def is_active(self) -> bool:
        """当前是否会真正包装命令（启用 + 探测通过）。"""
        return self._enabled and self._available is True

    def select_profile(self, command: str) -> str:
        """安装类命令用放网 profile,其余用断网主 profile（对齐 Linux install-escape）。"""
        return NET_PROFILE_PATH if _is_install_command(command) else MAIN_PROFILE_PATH

    def wrap(self, wire_command: str, command: str) -> str:
        """把已编码的 PowerShell wire 用 ybsbr 套在最外层（cmd.exe 口径）。

        ``command`` 是模型原始命令,仅用于选 profile（安装类→放网）;实际执行的是 ``wire_command``。
        WAA ``/execute`` 外层是 cmd.exe:用 ``call`` 前缀承载,规避行首引号被 ``cmd /c`` 吞掉的坑
        （ybsbr.exe 路径含空格必须加引号）。 最终等价于
        ``[cd /d <cwd> &&] call "ybsbr.exe" --profile="P" exec -- powershell -EncodedCommand <b64>``。
        """
        profile = self.select_profile(command)
        return f'call "{YBSBR_EXE}" --profile="{profile}" exec -- {wire_command}'
