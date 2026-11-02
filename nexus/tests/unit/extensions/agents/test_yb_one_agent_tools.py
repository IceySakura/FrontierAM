"""Tests for yb_one_agent tools (v2): bash, read, edit, write.

These tests target the v2 implementations (``*_v2``), which retain a
``description`` field on every input. The newer ``bash_tool``/``read_tool``/
``edit_tool``/``write_tool`` (without ``description``) are thin wrappers that
forward to v2, so v2 coverage transitively covers them.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from nexus.core import ToolContext
from nexus.extensions.agents.yb_one_agent import tool as tool_module
from nexus.extensions.agents.yb_one_agent.tools import (
    MAX_RESPONSE_LEN,
    OUTPUT_THRESHOLD,
    PREVIEW_HEAD,
    PREVIEW_TAIL,
    TRUNCATED_MESSAGE,
    BashInputV2,
    EditInputV2,
    ReadInputV2,
    WriteInputV2,
    _generate_preview,
    _smart_cut,
    bash_tool_v2,
    edit_tool_v2,
    maybe_truncate,
    read_tool_v2,
    write_tool_v2,
)
from nexus.runtime.executors import file as file_executors_pkg
from nexus.runtime.executors.command import CommandRunStatus, RunCommandResponse
from nexus.runtime.executors.file import (
    DownloadFileResponse,
    FileItem,
    FileOperationResult,
    UploadFileResponse,
)


# ============== Helpers ==============

# Patch targets: read/edit/write tools use local imports from nexus.runtime.executors,
# so we patch on the executor modules directly.
_PATCH_RUN_CMD = 'nexus.extensions.agents.yb_one_agent.tools.run_command'
_PATCH_DOWNLOAD = 'nexus.runtime.executors.file.download_file'
_PATCH_UPLOAD = 'nexus.runtime.executors.file.upload_file'


def _ctx(**tool_state_kw) -> ToolContext:
    """Create a minimal ToolContext with optional tool_state entries."""
    return ToolContext(tool_configs={}, tool_state=tool_state_kw)


def _text(resp) -> str:
    """Extract text from a ToolResponse."""
    return resp.output[0]['text']


# ============== maybe_truncate ==============


class TestMaybeTruncate:
    def test_short_content_unchanged(self):
        assert maybe_truncate('hello') == 'hello'

    def test_exact_max_length_unchanged(self):
        content = 'a' * MAX_RESPONSE_LEN
        assert maybe_truncate(content) == content

    def test_over_max_length_truncated(self):
        content = 'a' * (MAX_RESPONSE_LEN + 100)
        result = maybe_truncate(content)
        assert result.endswith(TRUNCATED_MESSAGE)
        assert len(result) == MAX_RESPONSE_LEN + len(TRUNCATED_MESSAGE)

    def test_custom_max_length(self):
        result = maybe_truncate('abcdefgh', max_length=5)
        assert result == 'abcde' + TRUNCATED_MESSAGE


# ============== Bash Tool ==============


class TestBashTool:
    async def test_successful_command(self):
        ctx = _ctx()
        fake_response = RunCommandResponse(
            status=CommandRunStatus.Finished,
            execution_time=0.5,
            return_code=0,
            stdout='hello world',
            stderr='',
        )

        with patch(
            'nexus.extensions.agents.yb_one_agent.tools.run_command',
            new_callable=AsyncMock,
            return_value=fake_response,
        ) as mock_cmd:
            resp = await bash_tool_v2(BashInputV2(description='test', command='echo hello world'), ctx)

        assert 'hello world' in _text(resp)
        assert 'exit code 0' in _text(resp)
        assert resp.meta['exit_code'] == 0
        assert resp.meta['execution_time'] == 0.5

        # Verify run_command was called with correct args
        call_args = mock_cmd.call_args[0][0]
        assert call_args.command == 'echo hello world'
        assert call_args.timeout == 60  # DEFAULT_TIMEOUT

    async def test_command_with_stderr(self):
        ctx = _ctx()
        fake_response = RunCommandResponse(
            status=CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='output',
            stderr='warning msg',
        )

        with patch(
            'nexus.extensions.agents.yb_one_agent.tools.run_command',
            new_callable=AsyncMock,
            return_value=fake_response,
        ):
            resp = await bash_tool_v2(BashInputV2(description='test', command='cmd'), ctx)

        text = _text(resp)
        assert 'output' in text
        assert '[stderr]' in text
        assert 'warning msg' in text

    async def test_command_with_nonzero_exit(self):
        ctx = _ctx()
        fake_response = RunCommandResponse(
            status=CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=1,
            stdout='',
            stderr='not found',
        )

        with patch(
            'nexus.extensions.agents.yb_one_agent.tools.run_command',
            new_callable=AsyncMock,
            return_value=fake_response,
        ):
            resp = await bash_tool_v2(BashInputV2(description='test', command='false'), ctx)

        assert resp.meta['exit_code'] == 1
        assert 'exit code 1' in _text(resp)

    async def test_command_no_output(self):
        ctx = _ctx()
        fake_response = RunCommandResponse(
            status=CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='',
            stderr='',
        )

        with patch(
            'nexus.extensions.agents.yb_one_agent.tools.run_command',
            new_callable=AsyncMock,
            return_value=fake_response,
        ):
            resp = await bash_tool_v2(BashInputV2(description='test', command='true'), ctx)

        assert '(no output)' in _text(resp)

    async def test_command_timeout(self):
        ctx = _ctx()
        fake_response = RunCommandResponse(
            status=CommandRunStatus.TimeLimitExceeded,
            execution_time=5.0,
            stdout='partial output',
            stderr='',
        )

        with patch(
            'nexus.extensions.agents.yb_one_agent.tools.run_command',
            new_callable=AsyncMock,
            return_value=fake_response,
        ):
            resp = await bash_tool_v2(BashInputV2(description='test', command='sleep 999', timeout=5000), ctx)

        assert resp.meta['exit_code'] == -1
        assert resp.meta['timeout'] is True
        assert 'timed out' in _text(resp).lower()
        assert 'partial output' in _text(resp)

    async def test_command_timeout_with_stderr(self):
        ctx = _ctx()
        fake_response = RunCommandResponse(
            status=CommandRunStatus.TimeLimitExceeded,
            execution_time=5.0,
            stdout='',
            stderr='timeout warning',
        )

        with patch(
            'nexus.extensions.agents.yb_one_agent.tools.run_command',
            new_callable=AsyncMock,
            return_value=fake_response,
        ):
            resp = await bash_tool_v2(BashInputV2(description='test', command='sleep 999'), ctx)

        assert 'timeout warning' in _text(resp)

    async def test_command_error(self):
        ctx = _ctx()
        fake_response = RunCommandResponse(
            status=CommandRunStatus.Error,
            stderr='Permission denied',
        )

        with patch(
            'nexus.extensions.agents.yb_one_agent.tools.run_command',
            new_callable=AsyncMock,
            return_value=fake_response,
        ):
            resp = await bash_tool_v2(BashInputV2(description='test', command='bad'), ctx)

        assert resp.meta['exit_code'] == -1
        assert resp.meta['error'] is True
        assert 'Permission denied' in _text(resp)

    async def test_command_error_no_stderr(self):
        ctx = _ctx()
        fake_response = RunCommandResponse(
            status=CommandRunStatus.Error,
            stderr='',
        )

        with patch(
            'nexus.extensions.agents.yb_one_agent.tools.run_command',
            new_callable=AsyncMock,
            return_value=fake_response,
        ):
            resp = await bash_tool_v2(BashInputV2(description='test', command='bad'), ctx)

        assert 'Command execution failed' in _text(resp)

    async def test_timeout_ms_conversion(self):
        ctx = _ctx()
        fake_response = RunCommandResponse(
            status=CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='ok',
            stderr='',
        )

        with patch(
            'nexus.extensions.agents.yb_one_agent.tools.run_command',
            new_callable=AsyncMock,
            return_value=fake_response,
        ) as mock_cmd:
            await bash_tool_v2(BashInputV2(description='test', command='echo ok', timeout=30000), ctx)

        call_args = mock_cmd.call_args[0][0]
        assert call_args.timeout == 30.0  # 30000ms -> 30s

    async def test_timeout_max_cap(self):
        ctx = _ctx()
        fake_response = RunCommandResponse(
            status=CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='ok',
            stderr='',
        )

        with patch(
            'nexus.extensions.agents.yb_one_agent.tools.run_command',
            new_callable=AsyncMock,
            return_value=fake_response,
        ) as mock_cmd:
            await bash_tool_v2(BashInputV2(description='test', command='echo ok', timeout=999999), ctx)

        call_args = mock_cmd.call_args[0][0]
        assert call_args.timeout == 600.0  # capped at MAX_TIMEOUT_MS

    async def test_working_dir_from_context(self):
        ctx = _ctx(working_dir='/workspace/project')
        fake_response = RunCommandResponse(
            status=CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='ok',
            stderr='',
        )

        with patch(
            'nexus.extensions.agents.yb_one_agent.tools.run_command',
            new_callable=AsyncMock,
            return_value=fake_response,
        ) as mock_cmd:
            await bash_tool_v2(BashInputV2(description='test', command='ls'), ctx)

        call_args = mock_cmd.call_args[0][0]
        assert call_args.cwd == '/workspace/project'

    async def test_output_over_threshold_persisted(self, tmp_path):
        """超过 OUTPUT_THRESHOLD（16K）→ 落盘 + 预览，而非内联全量。"""
        ctx = _ctx(working_dir=str(tmp_path))
        huge_stdout = 'x' * (OUTPUT_THRESHOLD + 1000)
        fake_response = RunCommandResponse(
            status=CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout=huge_stdout,
            stderr='',
        )

        with patch(
            'nexus.extensions.agents.yb_one_agent.tools.run_command',
            new_callable=AsyncMock,
            return_value=fake_response,
        ):
            resp = await bash_tool_v2(BashInputV2(description='test', command='cat bigfile'), ctx)

        assert _text(resp).startswith('Output too large. Full output saved to: ')
        assert resp.meta['output_file'].endswith('.txt')


# ============== Bash Tool env from tool_state ==============


class TestBashToolEnv:
    """env is read from ``context.tool_state['env']`` and passed as ``extra_env``."""

    async def test_env_from_tool_state_v2(self):
        ctx = _ctx(env={'MY_VAR': 'hello', 'OTHER': 'world'})
        fake_response = RunCommandResponse(
            status=CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='ok',
            stderr='',
        )

        with patch(
            'nexus.extensions.agents.yb_one_agent.tools.run_command',
            new_callable=AsyncMock,
            return_value=fake_response,
        ) as mock_cmd:
            await bash_tool_v2(BashInputV2(description='test', command='echo $MY_VAR'), ctx)

        call_args = mock_cmd.call_args[0][0]
        assert call_args.extra_env == {'MY_VAR': 'hello', 'OTHER': 'world'}

    async def test_env_from_tool_state_non_v2(self):
        ctx = _ctx(env={'MY_VAR': 'hello'})
        fake_response = RunCommandResponse(
            status=CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='ok',
            stderr='',
        )

        with patch(
            'nexus.extensions.agents.yb_one_agent.tool.run_command',
            new_callable=AsyncMock,
            return_value=fake_response,
        ) as mock_cmd:
            await tool_module.bash_tool(tool_module.BashInput(command='echo $MY_VAR'), ctx)

        call_args = mock_cmd.call_args[0][0]
        assert call_args.extra_env == {'MY_VAR': 'hello'}

    async def test_no_env_in_tool_state_defaults_to_empty(self):
        ctx = _ctx()
        fake_response = RunCommandResponse(
            status=CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='ok',
            stderr='',
        )

        with patch(
            'nexus.extensions.agents.yb_one_agent.tools.run_command',
            new_callable=AsyncMock,
            return_value=fake_response,
        ) as mock_cmd:
            await bash_tool_v2(BashInputV2(description='test', command='echo hi'), ctx)

        call_args = mock_cmd.call_args[0][0]
        assert call_args.extra_env == {}

    async def test_env_combined_with_working_dir(self):
        ctx = _ctx(working_dir='/custom/dir', env={'FOO': 'bar'})
        fake_response = RunCommandResponse(
            status=CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='ok',
            stderr='',
        )

        with patch(
            'nexus.extensions.agents.yb_one_agent.tools.run_command',
            new_callable=AsyncMock,
            return_value=fake_response,
        ) as mock_cmd:
            await bash_tool_v2(BashInputV2(description='test', command='echo hi'), ctx)

        call_args = mock_cmd.call_args[0][0]
        assert call_args.cwd == '/custom/dir'
        assert call_args.extra_env == {'FOO': 'bar'}


# ============== Bash Tool env from tool_state ==============


class TestBashToolEnv:
    """env is read from ``context.tool_state['env']`` and passed as ``extra_env``."""

    async def test_env_from_tool_state_v2(self):
        ctx = _ctx(env={'MY_VAR': 'hello', 'OTHER': 'world'})
        fake_response = RunCommandResponse(
            status=CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='ok',
            stderr='',
        )

        with patch(
            'nexus.extensions.agents.yb_one_agent.tools.run_command',
            new_callable=AsyncMock,
            return_value=fake_response,
        ) as mock_cmd:
            await bash_tool_v2(BashInputV2(description='test', command='echo $MY_VAR'), ctx)

        call_args = mock_cmd.call_args[0][0]
        assert call_args.extra_env == {'MY_VAR': 'hello', 'OTHER': 'world'}

    async def test_env_from_tool_state_non_v2(self):
        ctx = _ctx(env={'MY_VAR': 'hello'})
        fake_response = RunCommandResponse(
            status=CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='ok',
            stderr='',
        )

        with patch(
            'nexus.extensions.agents.yb_one_agent.tool.run_command',
            new_callable=AsyncMock,
            return_value=fake_response,
        ) as mock_cmd:
            await tool_module.bash_tool(tool_module.BashInput(command='echo $MY_VAR'), ctx)

        call_args = mock_cmd.call_args[0][0]
        assert call_args.extra_env == {'MY_VAR': 'hello'}

    async def test_no_env_in_tool_state_defaults_to_empty(self):
        ctx = _ctx()
        fake_response = RunCommandResponse(
            status=CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='ok',
            stderr='',
        )

        with patch(
            'nexus.extensions.agents.yb_one_agent.tools.run_command',
            new_callable=AsyncMock,
            return_value=fake_response,
        ) as mock_cmd:
            await bash_tool_v2(BashInputV2(description='test', command='echo hi'), ctx)

        call_args = mock_cmd.call_args[0][0]
        assert call_args.extra_env == {}

    async def test_env_combined_with_working_dir(self):
        ctx = _ctx(working_dir='/custom/dir', env={'FOO': 'bar'})
        fake_response = RunCommandResponse(
            status=CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='ok',
            stderr='',
        )

        with patch(
            'nexus.extensions.agents.yb_one_agent.tools.run_command',
            new_callable=AsyncMock,
            return_value=fake_response,
        ) as mock_cmd:
            await bash_tool_v2(BashInputV2(description='test', command='echo hi'), ctx)

        call_args = mock_cmd.call_args[0][0]
        assert call_args.cwd == '/custom/dir'
        assert call_args.extra_env == {'FOO': 'bar'}


# ============== Read Tool ==============


def _make_download_response(content: str | None = None, error: str | None = None) -> DownloadFileResponse:
    """Helper to create a DownloadFileResponse."""
    if error:
        return DownloadFileResponse(
            files=[],
            errors=[FileOperationResult(path='/test', success=False, message=error)],
        )
    return DownloadFileResponse(
        files=[FileItem(path='/test/file.py', content=content or '', encoding='utf8')],
        errors=None,
    )


class TestReadOutputLimit:
    def test_max_chars_is_accepted_but_hidden_from_agent_schema(self):
        params = tool_module.ReadInput.model_validate({'file_path': '/test/file.py', 'max_chars': 500})

        assert params.max_chars == 500
        assert 'max_chars' not in tool_module.ReadInput.model_json_schema()['properties']

    def test_max_chars_defaults_to_max_response_len(self):
        assert tool_module.ReadInput(file_path='/test/file.py').max_chars == MAX_RESPONSE_LEN

    async def test_read_uses_internal_max_chars(self, monkeypatch: pytest.MonkeyPatch):
        content = '\n'.join(f'line-{i}-{"x" * 80}' for i in range(1, 31))
        monkeypatch.setattr(
            file_executors_pkg, 'download_file', AsyncMock(return_value=_make_download_response(content))
        )

        response = await tool_module.read_tool(
            tool_module.ReadInput(file_path='/test/file.py', max_chars=500),
            _ctx(),
        )

        assert len(_text(response)) == 500 + len(TRUNCATED_MESSAGE)
        assert _text(response).endswith(TRUNCATED_MESSAGE)


class TestReadTool:
    async def test_read_file_success(self, monkeypatch: pytest.MonkeyPatch):
        ctx = _ctx()
        content = 'line1\nline2\nline3'
        mock_resp = _make_download_response(content)

        monkeypatch.setattr(file_executors_pkg, 'download_file', AsyncMock(return_value=mock_resp))
        resp = await read_tool_v2(ReadInputV2(description='test', file_path='/test/file.py'), ctx)

        text = _text(resp)
        assert '1\tline1' in text
        assert '2\tline2' in text
        assert '3\tline3' in text

    async def test_read_file_with_offset_and_limit(self, monkeypatch: pytest.MonkeyPatch):
        ctx = _ctx()
        content = '\n'.join(f'line{i}' for i in range(1, 11))  # 10 lines
        mock_resp = _make_download_response(content)

        monkeypatch.setattr(file_executors_pkg, 'download_file', AsyncMock(return_value=mock_resp))
        resp = await read_tool_v2(ReadInputV2(description='test', file_path='/test/file.py', offset=3, limit=2), ctx)

        text = _text(resp)
        assert '3\tline3' in text
        assert '4\tline4' in text
        assert 'line2' not in text
        assert 'line5' not in text
        assert '6 more lines' in text

    async def test_read_file_not_found(self, monkeypatch: pytest.MonkeyPatch):
        ctx = _ctx()
        mock_resp = _make_download_response(error='File does not exist')

        monkeypatch.setattr(file_executors_pkg, 'download_file', AsyncMock(return_value=mock_resp))
        resp = await read_tool_v2(ReadInputV2(description='test', file_path='/nonexistent'), ctx)

        assert 'Error' in _text(resp)
        assert 'File does not exist' in _text(resp)

    async def test_read_empty_file(self, monkeypatch: pytest.MonkeyPatch):
        ctx = _ctx()
        mock_resp = _make_download_response('')

        monkeypatch.setattr(file_executors_pkg, 'download_file', AsyncMock(return_value=mock_resp))
        resp = await read_tool_v2(ReadInputV2(description='test', file_path='/test/empty.py'), ctx)

        # Should not error, just return empty formatted content
        assert resp.output is not None

    async def test_read_exception_handling(self, monkeypatch: pytest.MonkeyPatch):
        ctx = _ctx()

        monkeypatch.setattr(file_executors_pkg, 'download_file', AsyncMock(side_effect=RuntimeError('connection lost')))
        resp = await read_tool_v2(ReadInputV2(description='test', file_path='/test/file.py'), ctx)

        assert 'Error' in _text(resp)
        assert 'connection lost' in _text(resp)

    async def test_read_default_limit_clips_at_2000(self, monkeypatch: pytest.MonkeyPatch):
        """Verify default limit is 2000 lines (lines beyond limit are dropped)."""
        import nexus.extensions.agents.yb_one_agent.tools as tool_module

        ctx = _ctx()
        total_lines = 2005
        content = '\n'.join('.' for _ in range(total_lines))
        mock_resp = _make_download_response(content)

        monkeypatch.setattr(file_executors_pkg, 'download_file', AsyncMock(return_value=mock_resp))
        # Bypass truncation to test the limit logic independently
        monkeypatch.setattr(tool_module, 'maybe_truncate', lambda s, **kw: s)

        resp = await read_tool_v2(ReadInputV2(description='test', file_path='/test/big.py'), ctx)

        text = _text(resp)
        # The formatted output should contain line 2000 but not 2001
        assert '2000\t' in text
        assert '2001\t' not in text
        assert '5 more lines' in text


# ============== Edit Tool ==============


class TestEditTool:
    async def test_edit_single_replacement(self, monkeypatch: pytest.MonkeyPatch):
        ctx = _ctx()
        original = 'hello world\ngoodbye world'
        download_resp = _make_download_response(original)
        upload_resp = UploadFileResponse(
            results=[FileOperationResult(path='/test/file.py', success=True, message='ok')]
        )

        monkeypatch.setattr(file_executors_pkg, 'download_file', AsyncMock(return_value=download_resp))
        mock_upload = AsyncMock(return_value=upload_resp)
        monkeypatch.setattr(file_executors_pkg, 'upload_file', mock_upload)

        resp = await edit_tool_v2(
            EditInputV2(description='test', file_path='/test/file.py', old_string='hello', new_string='hi'),
            ctx,
        )

        assert 'Successfully edited' in _text(resp)
        assert '1 replacement' in _text(resp)

        # Verify the uploaded content
        uploaded_content = mock_upload.call_args[0][0].files[0].content
        assert uploaded_content == 'hi world\ngoodbye world'

    async def test_edit_replace_all(self, monkeypatch: pytest.MonkeyPatch):
        ctx = _ctx()
        original = 'foo bar foo baz foo'
        download_resp = _make_download_response(original)
        upload_resp = UploadFileResponse(
            results=[FileOperationResult(path='/test/file.py', success=True, message='ok')]
        )

        monkeypatch.setattr(file_executors_pkg, 'download_file', AsyncMock(return_value=download_resp))
        mock_upload = AsyncMock(return_value=upload_resp)
        monkeypatch.setattr(file_executors_pkg, 'upload_file', mock_upload)

        resp = await edit_tool_v2(
            EditInputV2(
                description='test', file_path='/test/file.py', old_string='foo', new_string='qux', replace_all=True
            ),
            ctx,
        )

        assert 'Successfully edited' in _text(resp)
        assert '3 replacement' in _text(resp)
        uploaded_content = mock_upload.call_args[0][0].files[0].content
        assert uploaded_content == 'qux bar qux baz qux'

    async def test_edit_old_string_not_found(self, monkeypatch: pytest.MonkeyPatch):
        ctx = _ctx()
        download_resp = _make_download_response('hello world')

        monkeypatch.setattr(file_executors_pkg, 'download_file', AsyncMock(return_value=download_resp))

        resp = await edit_tool_v2(
            EditInputV2(description='test', file_path='/test/file.py', old_string='nonexistent', new_string='new'),
            ctx,
        )

        assert 'old_string not found' in _text(resp)

    async def test_edit_ambiguous_match(self, monkeypatch: pytest.MonkeyPatch):
        ctx = _ctx()
        original = 'foo bar foo'  # 'foo' appears twice
        download_resp = _make_download_response(original)

        monkeypatch.setattr(file_executors_pkg, 'download_file', AsyncMock(return_value=download_resp))

        resp = await edit_tool_v2(
            EditInputV2(description='test', file_path='/test/file.py', old_string='foo', new_string='baz'),
            ctx,
        )

        assert 'appears 2 times' in _text(resp)

    async def test_edit_file_not_found(self, monkeypatch: pytest.MonkeyPatch):
        ctx = _ctx()
        download_resp = _make_download_response(error='File does not exist')

        monkeypatch.setattr(file_executors_pkg, 'download_file', AsyncMock(return_value=download_resp))

        resp = await edit_tool_v2(
            EditInputV2(description='test', file_path='/nonexistent', old_string='a', new_string='b'),
            ctx,
        )

        assert 'Error' in _text(resp)

    async def test_edit_exception_handling(self, monkeypatch: pytest.MonkeyPatch):
        ctx = _ctx()

        monkeypatch.setattr(file_executors_pkg, 'download_file', AsyncMock(side_effect=RuntimeError('disk full')))

        resp = await edit_tool_v2(
            EditInputV2(description='test', file_path='/test/file.py', old_string='a', new_string='b'),
            ctx,
        )

        assert 'Error' in _text(resp)
        assert 'disk full' in _text(resp)

    async def test_edit_upload_failure_result(self, monkeypatch: pytest.MonkeyPatch):
        ctx = _ctx()
        download_resp = _make_download_response('hello world')
        upload_resp = UploadFileResponse(
            results=[FileOperationResult(path='/readonly/file.py', success=False, message='read-only filesystem')]
        )

        monkeypatch.setattr(file_executors_pkg, 'download_file', AsyncMock(return_value=download_resp))
        monkeypatch.setattr(file_executors_pkg, 'upload_file', AsyncMock(return_value=upload_resp))

        resp = await edit_tool_v2(
            EditInputV2(description='test', file_path='/readonly/file.py', old_string='hello', new_string='hi'),
            ctx,
        )

        assert 'Error' in _text(resp)
        assert 'read-only filesystem' in _text(resp)
        assert 'Successfully edited' not in _text(resp)


# ============== Write Tool ==============


class TestWriteTool:
    async def test_write_success(self, monkeypatch: pytest.MonkeyPatch):
        ctx = _ctx()
        upload_resp = UploadFileResponse(results=[FileOperationResult(path='/test/new.py', success=True, message='ok')])

        mock_upload = AsyncMock(return_value=upload_resp)
        monkeypatch.setattr(file_executors_pkg, 'upload_file', mock_upload)

        resp = await write_tool_v2(
            WriteInputV2(description='test', file_path='/test/new.py', content='line1\nline2\nline3'),
            ctx,
        )

        assert 'Successfully wrote' in _text(resp)
        assert '3 lines' in _text(resp)

        uploaded_content = mock_upload.call_args[0][0].files[0].content
        assert uploaded_content == 'line1\nline2\nline3'

    async def test_write_single_line(self, monkeypatch: pytest.MonkeyPatch):
        ctx = _ctx()
        upload_resp = UploadFileResponse(
            results=[FileOperationResult(path='/test/one.txt', success=True, message='ok')]
        )

        monkeypatch.setattr(file_executors_pkg, 'upload_file', AsyncMock(return_value=upload_resp))

        resp = await write_tool_v2(
            WriteInputV2(description='test', file_path='/test/one.txt', content='single line'),
            ctx,
        )

        assert '1 lines' in _text(resp)

    async def test_write_exception_handling(self, monkeypatch: pytest.MonkeyPatch):
        ctx = _ctx()

        monkeypatch.setattr(
            file_executors_pkg, 'upload_file', AsyncMock(side_effect=PermissionError('read-only filesystem'))
        )

        resp = await write_tool_v2(
            WriteInputV2(description='test', file_path='/readonly/file.py', content='data'),
            ctx,
        )

        assert 'Error' in _text(resp)
        assert 'read-only filesystem' in _text(resp)

    async def test_write_upload_failure_result(self, monkeypatch: pytest.MonkeyPatch):
        ctx = _ctx()
        upload_resp = UploadFileResponse(
            results=[FileOperationResult(path='/readonly/file.py', success=False, message='read-only filesystem')]
        )

        monkeypatch.setattr(file_executors_pkg, 'upload_file', AsyncMock(return_value=upload_resp))

        resp = await write_tool_v2(
            WriteInputV2(description='test', file_path='/readonly/file.py', content='data'),
            ctx,
        )

        assert 'Error' in _text(resp)
        assert 'read-only filesystem' in _text(resp)
        assert 'Successfully wrote' not in _text(resp)


# ============== v3: bash 输出过大处理 ==============


def _bash_response(stdout: str, stderr: str = '', return_code: int = 0) -> RunCommandResponse:
    return RunCommandResponse(
        status=CommandRunStatus.Finished,
        execution_time=0.1,
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
    )


class TestSmartCutPreview:
    """纯函数：换行感知截断 + 有界 head/tail 预览。"""

    def test_short_output_returned_as_is(self):
        s = 'line1\nline2\nline3'
        assert _generate_preview(s) == s

    def test_total_bounded_for_many_single_long_lines(self):
        """grep 大量单行文件：250 行 × 1000 字符也不会爆，预览有硬上限。"""
        output = '\n'.join('x' * 1000 for _ in range(250))  # ~250KB
        preview = _generate_preview(output)
        # 头 + 尾 + 分隔符，远小于 250KB
        assert len(preview) <= PREVIEW_HEAD + PREVIEW_TAIL + len('\n...\n') + 10

    def test_single_over_long_line_hard_cut(self):
        """base64/混淆代码：整段无换行，只能硬截，但仍有界。"""
        output = 'A' * 100_000
        preview = _generate_preview(output)
        assert len(preview) <= PREVIEW_HEAD + PREVIEW_TAIL + len('\n...\n')
        assert preview.startswith('A' * PREVIEW_HEAD)

    def test_normal_multiline_no_half_line(self):
        """正常多行：head 砍在换行处、tail 从换行后起，绝不留半行。"""
        output = '\n'.join(f'line-{i:04d}' for i in range(2000))  # 每行短，总量超预算
        preview = _generate_preview(output)
        head, _, tail = preview.partition('\n...\n')
        # head 的每一行都是完整的 line-XXXX；tail 同理
        assert all(line.startswith('line-') for line in head.split('\n') if line)
        assert all(line.startswith('line-') for line in tail.split('\n') if line)

    def test_tail_preserved(self):
        """末尾内容（报错/最终结果常在这里）保留在预览里。"""
        output = '\n'.join(f'line-{i:04d}' for i in range(2000))
        preview = _generate_preview(output)
        assert 'line-1999' in preview

    def test_smart_cut_head_aligns_to_newline(self):
        s = 'aaaa\n' * 1000  # 5000 chars, 行短
        cut = _smart_cut(s, PREVIEW_HEAD)
        assert cut.endswith('aaaa')  # 整行结束，无半行
        assert len(cut) <= PREVIEW_HEAD

    def test_smart_cut_tail_aligns_to_newline(self):
        s = 'aaaa\n' * 1000
        cut = _smart_cut(s, PREVIEW_TAIL, from_end=True)
        assert cut.startswith('aaaa')  # 从行首开始，无半行
        assert len(cut) <= PREVIEW_TAIL


class TestBashLargeOutput:
    async def test_large_output_saved_to_file(self, tmp_path):
        """超阈值 → 完整落盘到 <cwd>/.tool_output/ + canonical 预览文案。"""
        ctx = _ctx(working_dir=str(tmp_path))
        huge = '\n'.join(f'line{i}' for i in range(8000))  # 远超 30K 字符
        assert len(huge) > OUTPUT_THRESHOLD

        with patch(_PATCH_RUN_CMD, new_callable=AsyncMock, return_value=_bash_response(huge)):
            resp = await bash_tool_v2(BashInputV2(description='test', command='cat big'), ctx)

        text = _text(resp)
        assert text.startswith('Output too large. Full output saved to: ')
        assert 'Preview:' in text
        # 预览有界，不会把全量塞回来（含文案前缀 + 落盘路径的余量）
        assert len(text) <= PREVIEW_HEAD + PREVIEW_TAIL + 400
        assert resp.meta['output_file'].endswith('.txt')
        assert '.tool_output' in resp.meta['output_file']
        assert resp.meta['output_size'] == len(huge)
        # 落盘文件存在且内容完整
        with open(resp.meta['output_file'], encoding='utf-8') as f:
            assert f.read() == huge

    async def test_output_under_threshold_returned_full(self, tmp_path):
        """≤ 阈值：给全量，不截断、不落盘。"""
        ctx = _ctx(working_dir=str(tmp_path))
        body = '\n'.join(f'line{i}' for i in range(1000))  # 远小于 16K
        assert len(body) < OUTPUT_THRESHOLD

        with patch(_PATCH_RUN_CMD, new_callable=AsyncMock, return_value=_bash_response(body)):
            resp = await bash_tool_v2(BashInputV2(description='test', command='cat mid'), ctx)

        text = _text(resp)
        assert 'Output too large' not in text
        assert TRUNCATED_MESSAGE not in text
        assert 'line0' in text  # 全量：首行
        assert 'line999' in text  # 全量：尾行
        assert 'output_file' not in resp.meta

    async def test_save_failure_falls_back_to_truncate(self, tmp_path):
        """落盘失败（OSError）→ 退回字符截断，不崩溃。"""
        ctx = _ctx(working_dir=str(tmp_path))
        huge = 'x' * (OUTPUT_THRESHOLD + 5000)

        with (
            patch(_PATCH_RUN_CMD, new_callable=AsyncMock, return_value=_bash_response(huge)),
            patch(
                'nexus.extensions.agents.yb_one_agent.tools.os.makedirs',
                side_effect=OSError('read-only fs'),
            ),
        ):
            resp = await bash_tool_v2(BashInputV2(description='test', command='cat big'), ctx)

        text = _text(resp)
        assert TRUNCATED_MESSAGE in text
        assert 'Output too large' not in text
        assert 'output_file' not in resp.meta

    async def test_small_output_not_saved(self, tmp_path):
        ctx = _ctx(working_dir=str(tmp_path))
        with patch(_PATCH_RUN_CMD, new_callable=AsyncMock, return_value=_bash_response('tiny')):
            resp = await bash_tool_v2(BashInputV2(description='test', command='echo tiny'), ctx)

        assert 'Output too large' not in _text(resp)
        assert 'output_file' not in resp.meta


# ============== v3: read/edit 统一 unsupported-file-type 门禁 ==============


class TestUnsupportedFileType:
    @pytest.mark.parametrize(
        ('path', 'expected_ext'),
        [
            ('/w/report.docx', '.docx'),
            ('/w/sheet.xlsx', '.xlsx'),
            ('/w/slides.pptx', '.pptx'),
            ('/w/scan.pdf', '.pdf'),
            ('/w/photo.PNG', '.png'),
            ('/w/pic.jpg', '.jpg'),
            ('/w/notebook.ipynb', '.ipynb'),
        ],
    )
    async def test_read_rejects_unsupported_extensions(self, monkeypatch, path, expected_ext):
        ctx = _ctx()
        # download 不应被调用（门禁在下载前拦截）
        mock_download = AsyncMock(side_effect=AssertionError('download_file should not be called'))
        monkeypatch.setattr(file_executors_pkg, 'download_file', mock_download)

        resp = await read_tool_v2(ReadInputV2(description='test', file_path=path), ctx)

        assert _text(resp) == f'Unsupported file type: {expected_ext}'
        assert resp.meta['error']['code'] == 'UNSUPPORTED_FILE_TYPE'
        assert resp.meta['error']['extension'] == expected_ext

    async def test_read_rejects_binary_content(self, monkeypatch):
        ctx = _ctx()
        binary_content = 'PK\x03\x04\x00\x00binary garbage'
        mock_resp = _make_download_response(binary_content)
        monkeypatch.setattr(file_executors_pkg, 'download_file', AsyncMock(return_value=mock_resp))

        resp = await read_tool_v2(ReadInputV2(description='test', file_path='/w/blob'), ctx)

        assert _text(resp) == 'Unsupported file type: binary'
        assert resp.meta['error']['code'] == 'UNSUPPORTED_FILE_TYPE'

    async def test_read_allows_plain_text(self, monkeypatch):
        ctx = _ctx()
        mock_resp = _make_download_response('hello\nworld')
        monkeypatch.setattr(file_executors_pkg, 'download_file', AsyncMock(return_value=mock_resp))

        resp = await read_tool_v2(ReadInputV2(description='test', file_path='/w/notes.txt'), ctx)

        assert 'hello' in _text(resp)

    async def test_edit_rejects_ipynb(self, monkeypatch):
        ctx = _ctx()
        mock_download = AsyncMock(side_effect=AssertionError('download_file should not be called'))
        monkeypatch.setattr(file_executors_pkg, 'download_file', mock_download)

        resp = await edit_tool_v2(
            EditInputV2(description='test', file_path='/w/nb.ipynb', old_string='a', new_string='b'),
            ctx,
        )

        assert _text(resp) == 'Unsupported file type: .ipynb'
        assert resp.meta['error']['code'] == 'UNSUPPORTED_FILE_TYPE'


# ============== v3: edit old_string == new_string ==============


class TestEditNoChanges:
    async def test_edit_no_changes_rejected(self, monkeypatch):
        ctx = _ctx()
        mock_download = AsyncMock(side_effect=AssertionError('download_file should not be called'))
        monkeypatch.setattr(file_executors_pkg, 'download_file', mock_download)

        resp = await edit_tool_v2(
            EditInputV2(description='test', file_path='/w/file.py', old_string='same', new_string='same'),
            ctx,
        )

        assert _text(resp) == 'No changes made: old_string and new_string are identical.'
        assert resp.meta['error']['code'] == 'NO_CHANGES'
