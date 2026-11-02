"""Windows 执行后端：把 bash/read/edit/write 的叶子操作转发到 WAA Runtime。

设计见 ``docs/developing/local-agent/03-windows-waa-execution-design.md``。

要点：
- ``os == 'windows'`` 时由 ``LocalToolExecutor`` 构造本对象并注入 ``tool_state['waa_backend']``。
- 命令**恒走 PowerShell**：WAA ``/execute`` 外层是 cmd.exe，我们把命令包成
  ``powershell -EncodedCommand``（UTF-16LE base64）作为「要执行的命令」传进去，等价于
  ``cd /d {cwd} && powershell -EncodedCommand ...`` —— cmd 先切目录再拉起 powershell，cwd 正常生效。
  不支持 cmd 直传：模型命令与权限引擎都是 PowerShell 口径（cmdlet/别名），cmd.exe 跑不了。
  编码仅在 wire 层，模型看到的仍是原始命令，训推形状不受影响。
- 文件读写走 raw 端点 ``get_file`` / ``setup_create_file``。
"""

from __future__ import annotations

import base64
import logging
import ntpath

from typing import TYPE_CHECKING

from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


if TYPE_CHECKING:
    from nexus.extensions.agents.local_one_agent.windows_sandbox import WindowsSandbox
    from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime, AGSWindowsAgentArenaRuntime


logger = logging.getLogger(__name__)

# wire 层 PATH 前置注入的额外目录。 node/python runtime 由 VM 镜像预装并已在系统 PATH，
# 故默认无需再注入；保留该常量作为「provisioning 落地额外工具目录」的扩展点，缺省为空。
WAA_DEFAULT_PATH_DIRS: tuple[str, ...] = ()


def _build_prelude(path_dirs: tuple[str, ...], inject_utf8: bool) -> list[str]:
    """构造每条命令前置的 PowerShell 语句（PATH + UTF-8 适配）。"""
    lines: list[str] = []
    if path_dirs:
        prefix = ';'.join(path_dirs)
        lines.append(f"$env:PATH = '{prefix};' + $env:PATH")
    if inject_utf8:
        # 工具层适配 Windows 控制台编码：默认代码页（cp936）下 python 等 `print` 中文会
        # UnicodeEncodeError。 令子进程按 UTF-8 读写，把控制台代码页切 65001，并让 PowerShell
        # 以 UTF-8 读/写管道，尽量让非 ASCII 输出被 WAA /execute 正确捕获。
        lines.append("$env:PYTHONUTF8 = '1'")
        lines.append("$env:PYTHONIOENCODING = 'utf-8'")
        lines.append('chcp 65001 > $null 2>&1')
        lines.append(
            'try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; '
            '$OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}'
        )
    return lines


def _prepend_prelude(command: str, path_dirs: tuple[str, ...], inject_utf8: bool) -> str:
    """在命令前注入 PATH / UTF-8 前缀（同一 PowerShell 进程内对后续命令生效）。"""
    lines = _build_prelude(path_dirs, inject_utf8)
    if not lines:
        return command
    return '; '.join(lines) + '; ' + command


def _encode_powershell(command: str) -> str:
    """把命令包成 ``powershell -EncodedCommand``，避免引号 / 特殊字符在 cmd.exe 层被破坏。"""
    encoded = base64.b64encode(command.encode('utf-16-le')).decode('ascii')
    return f'powershell -NoProfile -EncodedCommand {encoded}'


class WaaBackend:
    """把叶子执行 / 文件操作转发到 Windows VM（WAA）的薄封装。"""

    def __init__(
        self,
        runtime: object,
        *,
        path_dirs: tuple[str, ...] = WAA_DEFAULT_PATH_DIRS,
        inject_utf8: bool = True,
    ) -> None:
        # 惰性导入：ags provider 包会拉起 e2b 等重依赖，仅在真正构造 backend（windows）时才引入，
        # 避免污染 Linux/macOS 路径与纯 Python 单测的导入链。
        from nexus.runtime.providers.ags.waa import require_raw_waa_runtime

        # runtime 可能是 raw 或 nexus 形态；统一取 raw（文件端点在 raw 上），命令走 nexus.call_executor。
        self._raw: AGSRawWindowsAgentArenaRuntime = require_raw_waa_runtime(runtime)
        self._nexus: AGSWindowsAgentArenaRuntime = self._raw.to_nexus()
        self._path_dirs = path_dirs
        self._inject_utf8 = inject_utf8
        # ybsbr OS 级沙箱（可选）:探测通过后由 executor 注入;None 或不可用则命令不包装。
        self._sandbox: WindowsSandbox | None = None

    def set_sandbox(self, sandbox: WindowsSandbox | None) -> None:
        """注入 ybsbr 沙箱包装器。 须在 sandbox 探测（probe）完成后调用,避免探测命令被自身包装。"""
        self._sandbox = sandbox

    async def run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        extra_env: dict[str, str] | None = None,
        timeout: float | None = None,  # noqa: ARG002  预留：WAA /execute 暂不透传 timeout
    ) -> RunCommandResponse:
        """在 Windows VM 内执行命令（恒走 PowerShell），返回与本地 run_command 同构的响应。

        沙箱激活时（ybsbr 探测通过）,PowerShell wire 再被 ybsbr 套在最外层做 OS 级隔离;
        未注入或探测未过则不包装（优雅降级为仅权限引擎）。 探测命令本身在 set_sandbox 前执行,不受包装。
        """
        wire_command = _encode_powershell(_prepend_prelude(command, self._path_dirs, self._inject_utf8))
        if self._sandbox is not None and self._sandbox.is_active():
            wire_command = self._sandbox.wrap(wire_command, command)
        result = await self._nexus.call_executor(
            'run_command',
            {'command': wire_command, 'cwd': cwd, 'extra_env': extra_env or {}},
        )
        if isinstance(result, RunCommandResponse):
            return result
        if isinstance(result, dict):
            return RunCommandResponse.model_validate(result)
        # 兜底：非预期返回不阻断，转成 Error 让工具照常渲染。
        logger.warning('WaaBackend.run_command: unexpected result type %s', type(result).__name__)
        return RunCommandResponse(status=CommandRunStatus.Error, stderr='WAA run_command returned unexpected payload')

    async def read_file(self, path: str) -> str:
        """读取 VM 内文件文本（UTF-8，容错解码）。"""
        data = await self._raw.get_file(path)
        return data.decode('utf-8', errors='replace')

    async def read_bytes(self, path: str) -> bytes:
        """读取 VM 内文件原始字节（产物回收用，二进制安全，不做解码）。"""
        return await self._raw.get_file(path)

    async def write_file(self, path: str, content: str) -> None:
        """写入 VM 内文件（覆盖）。

        走 setup_upload 发 UTF-8 字节流，而非 setup_create_file（JSON 传字符串）——后者服务端
        按字符串写文件时对非 ASCII（如中文）会静默失败仍返回 200，导致 write 误报成功、文件实际缺失。
        字节流绕开服务端字符串编码，与 read_file（get_file 取字节）对称。父目录先幂等创建，
        兜底 .tool_output 等嵌套路径（setup_upload 不保证建父目录）。
        """
        parent = ntpath.dirname(path)
        if parent:
            await self._raw.setup_create_folder(parent)
        await self._raw.setup_upload(path, content.encode('utf-8'))

    async def upload_file(self, vm_path: str, data: bytes) -> None:
        """把本地字节流上传到 VM 指定路径（provisioning / task seed 用）。

        setup_upload 不保证建父目录；与 write_file 对称，先幂等创建父目录。
        """
        parent = ntpath.dirname(vm_path)
        if parent:
            await self._raw.setup_create_folder(parent)
        await self._raw.setup_upload(vm_path, data)

    async def create_folder(self, vm_path: str) -> None:
        """在 VM 内创建目录（幂等由 WAA setup 侧保证）。"""
        await self._raw.setup_create_folder(vm_path)
