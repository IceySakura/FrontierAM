"""Tests for local_one_agent edit tool (delegates to harness v3 edit_tool_v2)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nexus.core.tool.types import ToolContext
from nexus.extensions.agents.local_one_agent.tools.edit import LocalEditInput, edit_tool
from nexus.runtime.executors import file as file_executors_pkg
from nexus.runtime.executors.file import (
    DownloadFileResponse,
    FileItem,
    FileOperationResult,
    UploadFileResponse,
)


@pytest.fixture
def tool_context():
    return ToolContext(run_id='test-run', tool_configs={}, tool_state={})


def _download(content: str) -> DownloadFileResponse:
    return DownloadFileResponse(files=[FileItem(path='/x', content=content, encoding='utf8')], errors=None)


def _upload_ok() -> UploadFileResponse:
    return UploadFileResponse(results=[FileOperationResult(path='/x', success=True, message='ok')])


class TestLocalEditInput:
    def test_inherits_fields(self):
        inp = LocalEditInput(file_path='/test.py', old_string='a', new_string='b')
        assert inp.file_path == '/test.py'
        assert inp.old_string == 'a'
        assert inp.new_string == 'b'
        assert inp.replace_all is False


class TestUnsupportedGate:
    async def test_ipynb_rejected(self, tool_context):
        result = await edit_tool(
            LocalEditInput(file_path='/workspace/notebook.ipynb', old_string='a', new_string='b'),
            tool_context,
        )
        assert result.meta['ok'] is False
        assert result.meta['error']['code'] == 'UNSUPPORTED_FILE_TYPE'
        assert result.output[0]['text'] == 'Unsupported file type: .ipynb'


class TestNoChanges:
    async def test_same_string_rejected(self, tool_context):
        result = await edit_tool(
            LocalEditInput(file_path='/workspace/test.py', old_string='same', new_string='same'),
            tool_context,
        )
        assert result.meta['ok'] is False
        assert result.meta['error']['code'] == 'NO_CHANGES'
        assert result.output[0]['text'] == 'No changes made: old_string and new_string are identical.'


class TestEditDelegation:
    async def test_successful_edit(self, tool_context, monkeypatch):
        monkeypatch.setattr(file_executors_pkg, 'download_file', AsyncMock(return_value=_download('hello world')))
        monkeypatch.setattr(file_executors_pkg, 'upload_file', AsyncMock(return_value=_upload_ok()))
        result = await edit_tool(
            LocalEditInput(file_path='/workspace/test.py', old_string='hello', new_string='hi'),
            tool_context,
        )
        assert 'Successfully edited' in result.output[0]['text']

    async def test_file_not_found(self, tool_context, monkeypatch):
        resp = DownloadFileResponse(
            files=[], errors=[FileOperationResult(path='/x', success=False, message='File does not exist')]
        )
        monkeypatch.setattr(file_executors_pkg, 'download_file', AsyncMock(return_value=resp))
        result = await edit_tool(
            LocalEditInput(file_path='/workspace/missing.py', old_string='a', new_string='b'),
            tool_context,
        )
        assert 'Error' in result.output[0]['text']

    async def test_old_string_not_found(self, tool_context, monkeypatch):
        monkeypatch.setattr(file_executors_pkg, 'download_file', AsyncMock(return_value=_download('hello world')))
        result = await edit_tool(
            LocalEditInput(file_path='/workspace/test.py', old_string='not_here', new_string='b'),
            tool_context,
        )
        assert 'not found' in result.output[0]['text']

    async def test_ambiguous_match(self, tool_context, monkeypatch):
        monkeypatch.setattr(file_executors_pkg, 'download_file', AsyncMock(return_value=_download('a a a')))
        result = await edit_tool(
            LocalEditInput(file_path='/workspace/test.py', old_string='a', new_string='b'),
            tool_context,
        )
        assert 'appears 3 times' in result.output[0]['text']
