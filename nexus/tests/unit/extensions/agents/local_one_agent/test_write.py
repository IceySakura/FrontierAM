"""Tests for local_one_agent write tool (mkdir parent + delegates to v3 write_tool_v2)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nexus.core.tool.types import ToolContext
from nexus.extensions.agents.local_one_agent.tools.write import LocalWriteInput, write_tool
from nexus.runtime.executors import file as file_executors_pkg
from nexus.runtime.executors.file import FileOperationResult, UploadFileResponse


@pytest.fixture
def tool_context():
    return ToolContext(run_id='test-run', tool_configs={}, tool_state={})


def _upload(success: bool, path: str, message: str = 'ok') -> UploadFileResponse:
    return UploadFileResponse(results=[FileOperationResult(path=path, success=success, message=message)])


class TestLocalWriteInput:
    def test_fields(self):
        inp = LocalWriteInput(file_path='/test.py', content='hello')
        assert inp.file_path == '/test.py'
        assert inp.content == 'hello'


class TestAutoCreateParentDirs:
    async def test_creates_parent_dirs(self, tool_context, tmp_path, monkeypatch):
        nested_path = str(tmp_path / 'a' / 'b' / 'c' / 'file.txt')
        monkeypatch.setattr(file_executors_pkg, 'upload_file', AsyncMock(return_value=_upload(True, nested_path)))
        result = await write_tool(LocalWriteInput(file_path=nested_path, content='hello'), tool_context)
        assert 'Successfully wrote' in result.output[0]['text']
        assert (tmp_path / 'a' / 'b' / 'c').is_dir()

    async def test_existing_parent_no_error(self, tool_context, tmp_path, monkeypatch):
        existing_dir = tmp_path / 'existing'
        existing_dir.mkdir()
        file_path = str(existing_dir / 'file.txt')
        monkeypatch.setattr(file_executors_pkg, 'upload_file', AsyncMock(return_value=_upload(True, file_path)))
        result = await write_tool(LocalWriteInput(file_path=file_path, content='hello'), tool_context)
        assert 'Successfully wrote' in result.output[0]['text']


class TestWriteUploadError:
    async def test_upload_error_reported(self, tool_context, tmp_path, monkeypatch):
        file_path = str(tmp_path / 'file.txt')
        monkeypatch.setattr(
            file_executors_pkg, 'upload_file', AsyncMock(return_value=_upload(False, file_path, 'Disk full'))
        )
        result = await write_tool(LocalWriteInput(file_path=file_path, content='hello'), tool_context)
        assert 'Error' in result.output[0]['text']
        assert 'Disk full' in result.output[0]['text']


class TestWriteLineCount:
    async def test_line_count_in_output(self, tool_context, tmp_path, monkeypatch):
        file_path = str(tmp_path / 'file.txt')
        monkeypatch.setattr(file_executors_pkg, 'upload_file', AsyncMock(return_value=_upload(True, file_path)))
        result = await write_tool(LocalWriteInput(file_path=file_path, content='line1\nline2\nline3'), tool_context)
        assert '3 lines' in result.output[0]['text']
