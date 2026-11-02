"""Tests for WaaBackend — Windows 执行后端（转发命令/文件到 WAA VM）。

不连真实 WAA：直接构造 WaaBackend 实例并 mock 其 _raw / _nexus 句柄，
验证命令编码、文件读写、上传等 wire 层行为。
"""

from __future__ import annotations

import base64

from types import SimpleNamespace
from unittest.mock import AsyncMock

from nexus.extensions.agents.local_one_agent.waa_backend import WaaBackend, _encode_powershell
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


def _make_backend(path_dirs: tuple[str, ...] = (), inject_utf8: bool = False) -> WaaBackend:
    """绕过 __init__（需真实 runtime），手工装配 mock 句柄。

    默认 path_dirs 为空、不注入 UTF-8，便于对命令做精确断言；前缀注入单独测。
    """
    backend = WaaBackend.__new__(WaaBackend)
    backend._raw = AsyncMock()
    backend._nexus = AsyncMock()
    backend._path_dirs = path_dirs
    backend._inject_utf8 = inject_utf8
    backend._sandbox = None
    return backend


def _decode_ps(wire: str) -> str:
    prefix = 'powershell -NoProfile -EncodedCommand '
    assert wire.startswith(prefix)
    return base64.b64decode(wire[len(prefix) :]).decode('utf-16-le')


class TestEncodePowershell:
    def test_roundtrip(self):
        wire = _encode_powershell('Get-ChildItem -Path C:\\Users')
        assert _decode_ps(wire) == 'Get-ChildItem -Path C:\\Users'


class TestRunCommand:
    async def test_always_encodes_powershell(self):
        backend = _make_backend()
        backend._nexus.call_executor = AsyncMock(
            return_value=RunCommandResponse(status=CommandRunStatus.Finished, return_code=0, stdout='ok')
        )
        resp = await backend.run_command('Get-ChildItem', cwd=r'C:\ws')

        assert resp.stdout == 'ok'
        name, payload = backend._nexus.call_executor.call_args.args
        assert name == 'run_command'
        assert _decode_ps(payload['command']) == 'Get-ChildItem'
        assert payload['cwd'] == r'C:\ws'
        assert payload['extra_env'] == {}

    async def test_cmd_style_command_still_wrapped_in_powershell(self):
        # 即使模型发的是 cmd 风格命令（dir），也恒经 powershell 承载，不做 cmd 直传。
        backend = _make_backend()
        backend._nexus.call_executor = AsyncMock(
            return_value=RunCommandResponse(status=CommandRunStatus.Finished, return_code=0, stdout='ok')
        )
        await backend.run_command('dir')

        _name, payload = backend._nexus.call_executor.call_args.args
        assert payload['command'].startswith('powershell -NoProfile -EncodedCommand ')
        assert _decode_ps(payload['command']) == 'dir'

    async def test_path_dirs_injected_before_command(self):
        # WAA /execute 新进程读不到注册表 PATH，故每条命令在 wire 层前置 $env:PATH。
        backend = _make_backend(path_dirs=(r'C:\tools\node',))
        backend._nexus.call_executor = AsyncMock(
            return_value=RunCommandResponse(status=CommandRunStatus.Finished, return_code=0, stdout='ok')
        )
        await backend.run_command('node --version')

        _name, payload = backend._nexus.call_executor.call_args.args
        decoded = _decode_ps(payload['command'])
        assert decoded == "$env:PATH = 'C:\\tools\\node;' + $env:PATH; node --version"

    async def test_utf8_prelude_injected(self):
        # 工具层适配：注入 PYTHONUTF8/PYTHONIOENCODING + Console 编码，规避中文 print 崩溃。
        backend = _make_backend(inject_utf8=True)
        backend._nexus.call_executor = AsyncMock(
            return_value=RunCommandResponse(status=CommandRunStatus.Finished, return_code=0, stdout='ok')
        )
        await backend.run_command('python app.py')

        _name, payload = backend._nexus.call_executor.call_args.args
        decoded = _decode_ps(payload['command'])
        assert "$env:PYTHONUTF8 = '1'" in decoded
        assert "$env:PYTHONIOENCODING = 'utf-8'" in decoded
        assert '[Console]::OutputEncoding = [System.Text.Encoding]::UTF8' in decoded
        assert decoded.endswith('; python app.py')

    async def test_dict_result_validated(self):
        backend = _make_backend()
        backend._nexus.call_executor = AsyncMock(return_value={'status': 'Finished', 'return_code': 0, 'stdout': 'hi'})
        resp = await backend.run_command('Get-Process')
        assert isinstance(resp, RunCommandResponse)
        assert resp.stdout == 'hi'

    async def test_unexpected_payload_becomes_error(self):
        backend = _make_backend()
        backend._nexus.call_executor = AsyncMock(return_value=12345)
        resp = await backend.run_command('Get-Process')
        assert resp.status == CommandRunStatus.Error
        assert 'unexpected' in (resp.stderr or '').lower()


class TestFileOps:
    async def test_read_file_decodes_bytes(self):
        backend = _make_backend()
        backend._raw.get_file = AsyncMock(return_value='héllo'.encode())
        assert await backend.read_file(r'C:\ws\a.txt') == 'héllo'

    async def test_read_file_lossy_decode(self):
        backend = _make_backend()
        backend._raw.get_file = AsyncMock(return_value=b'\xff\xfe bad')
        # 不抛异常，容错解码
        assert isinstance(await backend.read_file(r'C:\ws\a.txt'), str)

    async def test_write_file_uploads_utf8_bytes_and_creates_parent(self):
        # write 走 setup_upload（字节流），非 setup_create_file（字符串）；非 ASCII 才能正确落盘。
        backend = _make_backend()
        backend._raw.setup_create_folder = AsyncMock()
        backend._raw.setup_upload = AsyncMock()
        await backend.write_file(r'C:\ws\sub\a.txt', 'héllo 你好')
        backend._raw.setup_create_folder.assert_awaited_once_with(r'C:\ws\sub')
        backend._raw.setup_upload.assert_awaited_once_with(r'C:\ws\sub\a.txt', 'héllo 你好'.encode('utf-8'))

    async def test_upload_file(self):
        backend = _make_backend()
        backend._raw.setup_create_folder = AsyncMock()
        backend._raw.setup_upload = AsyncMock()
        await backend.upload_file(r'C:\tools\node.zip', b'\x00\x01')
        backend._raw.setup_create_folder.assert_awaited_once_with(r'C:\tools')
        backend._raw.setup_upload.assert_awaited_once_with(r'C:\tools\node.zip', b'\x00\x01')

    async def test_create_folder(self):
        backend = _make_backend()
        backend._raw.setup_create_folder = AsyncMock()
        await backend.create_folder(r'C:\tools')
        backend._raw.setup_create_folder.assert_awaited_once_with(r'C:\tools')


class TestSandboxWrap:
    async def test_active_sandbox_wraps_wire(self):
        backend = _make_backend()
        backend._nexus.call_executor = AsyncMock(
            return_value=RunCommandResponse(status=CommandRunStatus.Finished, return_code=0, stdout='ok')
        )
        backend.set_sandbox(SimpleNamespace(is_active=lambda: True, wrap=lambda wire, cmd: f'WRAP[{wire}]'))
        await backend.run_command('python app.py')

        _name, payload = backend._nexus.call_executor.call_args.args
        # ybsbr 包在最外层,内层仍是 powershell 编码 wire（编码只在 wire 层,模型形状不变）
        assert payload['command'].startswith('WRAP[powershell -NoProfile -EncodedCommand ')

    async def test_inactive_sandbox_no_wrap(self):
        backend = _make_backend()
        backend._nexus.call_executor = AsyncMock(
            return_value=RunCommandResponse(status=CommandRunStatus.Finished, return_code=0, stdout='ok')
        )
        backend.set_sandbox(SimpleNamespace(is_active=lambda: False, wrap=lambda wire, cmd: f'WRAP[{wire}]'))
        await backend.run_command('python app.py')

        _name, payload = backend._nexus.call_executor.call_args.args
        assert payload['command'].startswith('powershell -NoProfile -EncodedCommand ')
