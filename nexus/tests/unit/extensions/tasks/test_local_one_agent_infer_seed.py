"""local_one_agent_infer 依赖文件 seed / 产物回收测试（os 感知）。

覆盖两条执行环境：
- macos/linux（backend=None）：输入 shutil.copy 到本地 workspace，产物本地读回。
- windows/WAA（mock backend）：输入经 upload_file 上传到 VM，产物经 read_bytes 读回。

不连真实 WAA：windows 分支用 AsyncMock 断言上传目标路径 / 回收字节。
"""

from __future__ import annotations

import base64

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.tasks.local_one_agent_infer.task import (
    _collect_output_files,
    _seed_files_only,
    _strip_leading_dirs,
    _task_os,
    _vm_join,
)


class TestHelpers:
    def test_vm_join_normalizes_separators(self):
        assert _vm_join(r'C:\Users\Public\ws', 'sub/data.txt') == r'C:\Users\Public\ws\sub\data.txt'
        assert _vm_join('C:/Users/Public/ws', 'a.txt') == r'C:\Users\Public\ws\a.txt'

    def test_strip_leading_dirs_keeps_deeper_layers(self):
        # files/<task_id>/... 前两级剥掉，保留更深层级。
        assert _strip_leading_dirs('files/Data-001/sub/a.pdf') == Path('sub/a.pdf')
        # 层级不足 n+1 时退回纯文件名。
        assert _strip_leading_dirs('data/report.pdf') == Path('report.pdf')

    def test_task_os_windows(self):
        ti = SimpleNamespace(local_context={'runtime': {'os': 'windows'}})
        assert _task_os(ti) == 'windows'  # type: ignore[arg-type]

    def test_task_os_defaults_macos(self):
        ti = SimpleNamespace(local_context={})
        assert _task_os(ti) == 'macos'  # type: ignore[arg-type]


class TestSeedLocal:
    @pytest.mark.asyncio
    async def test_local_seed_copies_into_workspace(self, tmp_path):
        bench = tmp_path / 'bench'
        (bench / 'data').mkdir(parents=True)
        (bench / 'data' / 'report.pdf').write_bytes(b'PDFDATA')
        ws = tmp_path / 'ws'
        turn = {'input_file_list': ['data/report.pdf'], 'query': '看 docs/report.pdf'}

        await _seed_files_only(turn, bench, str(ws), 'task-1', backend=None)

        assert (ws / 'report.pdf').read_bytes() == b'PDFDATA'
        # query 引用了子目录路径，同步铺一份
        assert (ws / 'docs' / 'report.pdf').read_bytes() == b'PDFDATA'

    @pytest.mark.asyncio
    async def test_local_seed_missing_file_skipped(self, tmp_path):
        bench = tmp_path / 'bench'
        bench.mkdir()
        ws = tmp_path / 'ws'
        turn = {'input_file_list': ['nope.txt'], 'query': 'x'}
        await _seed_files_only(turn, bench, str(ws), 'task-1', backend=None)
        assert not ws.exists() or not any(ws.iterdir())


class TestSeedWindows:
    @pytest.mark.asyncio
    async def test_windows_seed_uploads_vm_paths(self, tmp_path):
        bench = tmp_path / 'bench'
        (bench / 'data').mkdir(parents=True)
        (bench / 'data' / 'report.pdf').write_bytes(b'PDFDATA')
        backend = SimpleNamespace(
            create_folder=AsyncMock(),
            upload_file=AsyncMock(),
        )
        turn = {'input_file_list': ['data/report.pdf'], 'query': '看 docs/report.pdf'}

        await _seed_files_only(turn, bench, r'C:\Users\Public\ws', 'task-1', backend=backend)

        backend.create_folder.assert_awaited_once_with(r'C:\Users\Public\ws')
        targets = [c.args[0] for c in backend.upload_file.await_args_list]
        payloads = [c.args[1] for c in backend.upload_file.await_args_list]
        assert r'C:\Users\Public\ws\report.pdf' in targets
        assert r'C:\Users\Public\ws\docs\report.pdf' in targets
        assert all(p == b'PDFDATA' for p in payloads)


class TestCollectLocal:
    @pytest.mark.asyncio
    async def test_local_collect_reads_back(self, tmp_path):
        ws = tmp_path / 'ws'
        ws.mkdir()
        (ws / 'out.txt').write_bytes(b'RESULT')
        result = await _collect_output_files([{'path': 'out.txt', 'type': 'file'}], str(ws), 'task-1', backend=None)
        assert result == {'out.txt': base64.b64encode(b'RESULT').decode('utf-8')}

    @pytest.mark.asyncio
    async def test_local_collect_directory_is_recursive(self, tmp_path):
        ws = tmp_path / 'ws'
        (ws / 'out' / 'nested').mkdir(parents=True)
        (ws / 'out' / 'a.txt').write_bytes(b'A')
        (ws / 'out' / 'nested' / 'b.txt').write_bytes(b'B')
        result = await _collect_output_files([{'path': 'out', 'type': 'directory'}], str(ws), 'task-1', backend=None)
        assert result == {
            'out/a.txt': base64.b64encode(b'A').decode('utf-8'),
            'out/nested/b.txt': base64.b64encode(b'B').decode('utf-8'),
        }


class TestCollectWindows:
    @pytest.mark.asyncio
    async def test_windows_collect_relative_to_workspace(self):
        backend = SimpleNamespace(read_bytes=AsyncMock(return_value=b'XLSX'))
        result = await _collect_output_files(
            [{'path': r'C:\Users\Public\ws\out.xlsx', 'type': 'file'}],
            r'C:\Users\Public\ws',
            'task-1',
            backend=backend,
        )
        assert result == {'out.xlsx': base64.b64encode(b'XLSX').decode('utf-8')}
        backend.read_bytes.assert_awaited_once_with(r'C:\Users\Public\ws\out.xlsx')

    @pytest.mark.asyncio
    async def test_windows_collect_outside_workspace_uses_basename(self):
        backend = SimpleNamespace(read_bytes=AsyncMock(return_value=b'DATA'))
        result = await _collect_output_files(
            [{'path': r'C:\other\thing.csv', 'type': 'file'}], r'C:\Users\Public\ws', 'task-1', backend=backend
        )
        assert result == {'thing.csv': base64.b64encode(b'DATA').decode('utf-8')}

    @pytest.mark.asyncio
    async def test_windows_collect_relative_path_joined(self):
        backend = SimpleNamespace(read_bytes=AsyncMock(return_value=b'REL'))
        result = await _collect_output_files(
            [{'path': 'sub/out.bin', 'type': 'file'}], r'C:\Users\Public\ws', 'task-1', backend=backend
        )
        # VM backslashes are normalised to posix separators so nested dirs are recreated on disk.
        assert result == {'sub/out.bin': base64.b64encode(b'REL').decode('utf-8')}
        backend.read_bytes.assert_awaited_once_with(r'C:\Users\Public\ws\sub\out.bin')
