"""Tests for the Windows (WAA) branch of local_one_agent bash/read/edit/write tools.

不连真实 WAA：注入一个「假 backend」（WaaBackend 实例但公共方法被替换成 AsyncMock），
验证 os=='windows' 时四个工具走 backend 转发、输出形状与 v3 对齐、错误/大输出兜底正确。
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nexus.core.tool.types import ToolContext
from nexus.extensions.agents.local_one_agent.tools.bash import LocalBashInput, bash_tool
from nexus.extensions.agents.local_one_agent.tools.edit import LocalEditInput, edit_tool
from nexus.extensions.agents.local_one_agent.tools.read import LocalReadInput, read_tool
from nexus.extensions.agents.local_one_agent.tools.write import LocalWriteInput, write_tool
from nexus.extensions.agents.local_one_agent.waa_backend import WaaBackend
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


def _fake_backend() -> WaaBackend:
    """构造 WaaBackend 实例（绕过 __init__），公共方法置为 AsyncMock。"""
    backend = WaaBackend.__new__(WaaBackend)
    backend._raw = AsyncMock()
    backend._nexus = AsyncMock()
    backend.run_command = AsyncMock()
    backend.read_file = AsyncMock()
    backend.write_file = AsyncMock()
    return backend


def _ctx(backend: WaaBackend, working_dir: str = r'C:\ws') -> ToolContext:
    return ToolContext(run_id='r', tool_configs={}, tool_state={'waa_backend': backend, 'working_dir': working_dir})


def _ok(stdout: str = '', stderr: str = '', rc: int = 0) -> RunCommandResponse:
    return RunCommandResponse(
        status=CommandRunStatus.Finished, return_code=rc, stdout=stdout, stderr=stderr, execution_time=0.1
    )


class TestBashWaa:
    async def test_forwards_and_formats(self):
        backend = _fake_backend()
        backend.run_command.return_value = _ok(stdout='hello')
        result = await bash_tool(LocalBashInput(command='Get-ChildItem'), _ctx(backend))

        backend.run_command.assert_awaited_once()
        assert backend.run_command.await_args.args[0] == 'Get-ChildItem'
        text = result.output[0]['text']
        assert 'hello' in text
        assert 'exit code 0' in text
        assert result.meta['exit_code'] == 0

    async def test_timeout(self):
        backend = _fake_backend()
        backend.run_command.return_value = RunCommandResponse(
            status=CommandRunStatus.TimeLimitExceeded, stdout='partial', execution_time=60.0
        )
        result = await bash_tool(LocalBashInput(command='sleep'), _ctx(backend))
        assert 'timed out' in result.output[0]['text'].lower()
        assert result.meta.get('timeout') is True

    async def test_error(self):
        backend = _fake_backend()
        backend.run_command.return_value = RunCommandResponse(
            status=CommandRunStatus.Error, stderr='boom'
        )
        result = await bash_tool(LocalBashInput(command='bad'), _ctx(backend))
        assert 'boom' in result.output[0]['text']
        assert result.meta.get('error') is True

    async def test_large_output_written_back(self):
        backend = _fake_backend()
        backend.run_command.return_value = _ok(stdout='x' * 60000)
        result = await bash_tool(LocalBashInput(command='cat big'), _ctx(backend))

        backend.write_file.assert_awaited_once()
        saved_path = backend.write_file.await_args.args[0]
        assert saved_path.startswith(r'C:\ws\.tool_output')
        assert result.meta['output_file'] == saved_path
        assert 'Output too large' in result.output[0]['text']

    async def test_large_output_writeback_failure_falls_back(self):
        backend = _fake_backend()
        backend.run_command.return_value = _ok(stdout='x' * 60000)
        backend.write_file.side_effect = RuntimeError('vm down')
        result = await bash_tool(LocalBashInput(command='cat big'), _ctx(backend))
        # 写回失败 → 退化为截断，仍带 exit code 尾注，不抛异常
        assert 'output_file' not in result.meta
        assert 'exit code 0' in result.output[0]['text']


class TestReadWaa:
    async def test_cat_n_render(self):
        backend = _fake_backend()
        backend.read_file.return_value = 'a\nb\nc'
        result = await read_tool(LocalReadInput(file_path=r'C:\ws\f.txt'), _ctx(backend))
        text = result.output[0]['text']
        assert '     1\ta' in text
        assert '     3\tc' in text
        backend.read_file.assert_awaited_once_with(r'C:\ws\f.txt')

    async def test_offset_limit(self):
        backend = _fake_backend()
        backend.read_file.return_value = '\n'.join(str(i) for i in range(1, 11))
        result = await read_tool(LocalReadInput(file_path=r'C:\ws\f.txt', offset=3, limit=2), _ctx(backend))
        text = result.output[0]['text']
        assert '     3\t3' in text
        assert '     4\t4' in text
        assert 'more lines' in text

    async def test_unsupported_extension_skips_read(self):
        backend = _fake_backend()
        result = await read_tool(LocalReadInput(file_path=r'C:\ws\a.pdf'), _ctx(backend))
        backend.read_file.assert_not_awaited()
        assert result.output[0]['text']

    async def test_binary_content(self):
        backend = _fake_backend()
        backend.read_file.return_value = 'abc\x00def'
        result = await read_tool(LocalReadInput(file_path=r'C:\ws\f.bin'), _ctx(backend))
        assert 'binary' in result.output[0]['text'].lower() or result.meta

    async def test_read_error(self):
        backend = _fake_backend()
        backend.read_file.side_effect = RuntimeError('not found')
        result = await read_tool(LocalReadInput(file_path=r'C:\ws\missing.txt'), _ctx(backend))
        assert 'Error' in result.output[0]['text']


class TestWriteWaa:
    async def test_write_success(self):
        backend = _fake_backend()
        result = await write_tool(LocalWriteInput(file_path=r'C:\ws\a.txt', content='l1\nl2'), _ctx(backend))
        backend.write_file.assert_awaited_once_with(r'C:\ws\a.txt', 'l1\nl2')
        assert 'Successfully wrote' in result.output[0]['text']

    async def test_write_error(self):
        backend = _fake_backend()
        backend.write_file.side_effect = RuntimeError('disk full')
        result = await write_tool(LocalWriteInput(file_path=r'C:\ws\a.txt', content='x'), _ctx(backend))
        assert 'Error' in result.output[0]['text']


class TestEditWaa:
    async def test_edit_replace_first(self):
        backend = _fake_backend()
        backend.read_file.return_value = 'foo bar foo'
        result = await edit_tool(
            LocalEditInput(file_path=r'C:\ws\a.txt', old_string='foo', new_string='baz'), _ctx(backend)
        )
        # 未 replace_all 且出现多次 → 报歧义错误，不写回
        assert 'appears 2 times' in result.output[0]['text']
        backend.write_file.assert_not_awaited()

    async def test_edit_replace_all(self):
        backend = _fake_backend()
        backend.read_file.return_value = 'foo bar foo'
        result = await edit_tool(
            LocalEditInput(file_path=r'C:\ws\a.txt', old_string='foo', new_string='baz', replace_all=True),
            _ctx(backend),
        )
        backend.write_file.assert_awaited_once_with(r'C:\ws\a.txt', 'baz bar baz')
        assert 'Successfully edited' in result.output[0]['text']
        assert '2 replacement' in result.output[0]['text']

    async def test_edit_unique(self):
        backend = _fake_backend()
        backend.read_file.return_value = 'hello world'
        await edit_tool(
            LocalEditInput(file_path=r'C:\ws\a.txt', old_string='world', new_string='there'), _ctx(backend)
        )
        backend.write_file.assert_awaited_once_with(r'C:\ws\a.txt', 'hello there')

    async def test_edit_no_change(self):
        backend = _fake_backend()
        result = await edit_tool(
            LocalEditInput(file_path=r'C:\ws\a.txt', old_string='x', new_string='x'), _ctx(backend)
        )
        assert result.meta.get('error', {}).get('code') == 'NO_CHANGES'
        backend.read_file.assert_not_awaited()

    async def test_edit_not_found(self):
        backend = _fake_backend()
        backend.read_file.return_value = 'hello'
        result = await edit_tool(
            LocalEditInput(file_path=r'C:\ws\a.txt', old_string='zzz', new_string='y'), _ctx(backend)
        )
        assert 'not found' in result.output[0]['text']
        backend.write_file.assert_not_awaited()

    async def test_edit_unsupported_ext(self):
        backend = _fake_backend()
        result = await edit_tool(
            LocalEditInput(file_path=r'C:\ws\a.ipynb', old_string='a', new_string='b'), _ctx(backend)
        )
        backend.read_file.assert_not_awaited()
        assert result.output[0]['text']
