"""Tests for local_one_agent read tool.

read 现继承 harness v3 ``read_tool_v2``（统一 unsupported-file-type 门禁在 v3），
local 层只覆盖「图片 → COS URL 文本」（Part Ⅰ §4）。 本测试聚焦该覆盖 + 委托链路。
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nexus.core.tool.types import ToolContext
from nexus.extensions.agents.local_one_agent.tools.read import LocalReadInput, _mock_cos_url, read_tool
from nexus.runtime.executors import file as file_executors_pkg
from nexus.runtime.executors.file import DownloadFileResponse, FileItem, FileOperationResult


def _ctx():
    return ToolContext(run_id='test', tool_configs={}, tool_state={})


def _download_resp(content: str) -> DownloadFileResponse:
    return DownloadFileResponse(files=[FileItem(path='/x', content=content, encoding='utf8')], errors=None)


class TestLocalReadInput:
    def test_fields(self):
        inp = LocalReadInput(file_path='/test.py')
        assert inp.file_path == '/test.py'
        assert inp.offset == 1
        assert inp.limit == 2000
        assert inp.description == ''


class TestImageCosUrl:
    def test_mock_cos_url_deterministic(self):
        url1 = _mock_cos_url('/w/a.png')
        url2 = _mock_cos_url('/w/a.png')
        assert url1 == url2
        assert url1.startswith('cos-mock://')
        assert _mock_cos_url('/w/b.png') != url1

    @pytest.mark.parametrize('path', ['/w/photo.png', '/w/pic.JPG', '/w/scan.jpeg', '/w/a.webp', '/w/b.bmp'])
    async def test_image_returns_cos_url_text(self, path):
        result = await read_tool(LocalReadInput(file_path=path), _ctx())
        text = result.output[0]['text']
        assert 'is a local image' in text
        assert 'cos-mock://' in text
        assert result.meta['ok'] is True
        assert result.meta['file_type'] == 'image'


class TestUnsupportedDelegatedToV3:
    @pytest.mark.parametrize(
        ('path', 'ext'),
        [
            ('/w/doc.docx', '.docx'),
            ('/w/sheet.xlsx', '.xlsx'),
            ('/w/slides.pptx', '.pptx'),
            ('/w/scan.pdf', '.pdf'),
            ('/w/nb.ipynb', '.ipynb'),
        ],
    )
    async def test_office_pdf_ipynb_rejected(self, path, ext):
        result = await read_tool(LocalReadInput(file_path=path), _ctx())
        assert result.meta['ok'] is False
        assert result.meta['error']['code'] == 'UNSUPPORTED_FILE_TYPE'
        assert result.output[0]['text'] == f'Unsupported file type: {ext}'

    async def test_binary_content_rejected(self, monkeypatch):
        monkeypatch.setattr(
            file_executors_pkg, 'download_file', AsyncMock(return_value=_download_resp('hello\x00world'))
        )
        result = await read_tool(LocalReadInput(file_path='/w/blob.dat'), _ctx())
        assert result.meta['ok'] is False
        assert result.output[0]['text'] == 'Unsupported file type: binary'


class TestTextDelegatedToV3:
    async def test_text_file_read(self, monkeypatch):
        monkeypatch.setattr(
            file_executors_pkg, 'download_file', AsyncMock(return_value=_download_resp('hello world\nline 2'))
        )
        result = await read_tool(LocalReadInput(file_path='/w/readme.md'), _ctx())
        assert 'hello world' in result.output[0]['text']

    async def test_text_file_not_found(self, monkeypatch):
        resp = DownloadFileResponse(
            files=[], errors=[FileOperationResult(path='/w/x', success=False, message='File does not exist')]
        )
        monkeypatch.setattr(file_executors_pkg, 'download_file', AsyncMock(return_value=resp))
        result = await read_tool(LocalReadInput(file_path='/w/missing.txt'), _ctx())
        assert 'Error' in result.output[0]['text']
