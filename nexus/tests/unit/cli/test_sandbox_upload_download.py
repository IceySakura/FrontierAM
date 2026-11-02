"""Tests for nexus sandbox upload/download CLI commands.

Interface spec:
- upload: read local file → detect encoding → call RemoteRuntime.upload_file
- download: call RemoteRuntime.download_file → write local file

These functions call asyncio.run() internally (like sandbox_exec).
"""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from nexus.runtime.executors.models.file import DownloadFileResponse, FileItem, FileOperationResult, UploadFileResponse


@pytest.fixture(autouse=True)
def _mock_http_client_close():
    """Prevent http_client.close() from hitting real aiohttp."""
    with patch('nexus.utils.async_http.http_client.close', new_callable=AsyncMock):
        yield


class TestSandboxUpload:
    """nexus sandbox upload <base_url> <local_path> <remote_path>"""

    def _make_args(self, base_url: str, local_path: str, remote_path: str, timeout: float = 180):
        from argparse import Namespace

        return Namespace(
            base_url=base_url,
            local_path=local_path,
            remote_path=remote_path,
            timeout=timeout,
        )

    def test_upload_text_file(self, tmp_path: Path):
        """Text files are uploaded with utf8 encoding."""
        local_file = tmp_path / 'script.sh'
        local_file.write_text('#!/bin/bash\necho hello\n')

        mock_response = UploadFileResponse(
            results=[FileOperationResult(path='/workspace/script.sh', success=True)]
        )

        with patch('nexus.runtime.client.remote.RemoteRuntime.upload_file', new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = mock_response

            from nexus.cli.commands.sandbox import sandbox_upload

            args = self._make_args('http://10.0.0.1:32000', str(local_file), '/workspace/script.sh')
            exit_code = sandbox_upload(args)

            assert exit_code == 0
            mock_upload.assert_called_once()
            files = mock_upload.call_args.kwargs['files']
            assert len(files) == 1
            assert files[0].path == '/workspace/script.sh'
            assert files[0].encoding == 'utf8'
            assert files[0].content == '#!/bin/bash\necho hello\n'

    def test_upload_binary_file(self, tmp_path: Path):
        """Binary files are uploaded with base64 encoding."""
        local_file = tmp_path / 'data.bin'
        binary_content = bytes(range(256))
        local_file.write_bytes(binary_content)

        mock_response = UploadFileResponse(
            results=[FileOperationResult(path='/workspace/data.bin', success=True)]
        )

        with patch('nexus.runtime.client.remote.RemoteRuntime.upload_file', new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = mock_response

            from nexus.cli.commands.sandbox import sandbox_upload

            args = self._make_args('http://10.0.0.1:32000', str(local_file), '/workspace/data.bin')
            exit_code = sandbox_upload(args)

            assert exit_code == 0
            files = mock_upload.call_args.kwargs['files']
            assert files[0].encoding == 'base64'
            assert base64.b64decode(files[0].content) == binary_content

    def test_upload_empty_file(self, tmp_path: Path):
        """Empty files can be uploaded (utf8, empty content)."""
        local_file = tmp_path / 'empty.txt'
        local_file.touch()

        mock_response = UploadFileResponse(
            results=[FileOperationResult(path='/workspace/empty.txt', success=True)]
        )

        with patch('nexus.runtime.client.remote.RemoteRuntime.upload_file', new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = mock_response

            from nexus.cli.commands.sandbox import sandbox_upload

            args = self._make_args('http://10.0.0.1:32000', str(local_file), '/workspace/empty.txt')
            exit_code = sandbox_upload(args)

            assert exit_code == 0
            files = mock_upload.call_args.kwargs['files']
            assert files[0].encoding == 'utf8'
            assert files[0].content == ''

    def test_upload_nonexistent_file(self, tmp_path: Path, capsys):
        """Upload of nonexistent local file prints error and returns 1."""
        from nexus.cli.commands.sandbox import sandbox_upload

        args = self._make_args('http://10.0.0.1:32000', str(tmp_path / 'nope.txt'), '/workspace/nope.txt')
        exit_code = sandbox_upload(args)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert captured.err.startswith('error: ')
        assert 'not found' in captured.err.lower()

    def test_upload_large_file_warning(self, tmp_path: Path, capsys):
        """Files >10MB print a size warning to stderr but still proceed."""
        local_file = tmp_path / 'large.bin'
        local_file.write_bytes(b'\x00' * (11 * 1024 * 1024))

        mock_response = UploadFileResponse(
            results=[FileOperationResult(path='/workspace/large.bin', success=True)]
        )

        with patch('nexus.runtime.client.remote.RemoteRuntime.upload_file', new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = mock_response

            from nexus.cli.commands.sandbox import sandbox_upload

            args = self._make_args('http://10.0.0.1:32000', str(local_file), '/workspace/large.bin')
            exit_code = sandbox_upload(args)

            assert exit_code == 0
            captured = capsys.readouterr()
            assert 'warning:' in captured.err.lower()
            assert '10' in captured.err  # mentions size threshold

    def test_upload_server_failure(self, tmp_path: Path, capsys):
        """Server-side failure is reported and returns 1."""
        local_file = tmp_path / 'script.sh'
        local_file.write_text('echo fail')

        mock_response = UploadFileResponse(
            results=[FileOperationResult(path='/workspace/script.sh', success=False, message='Permission denied')]
        )

        with patch('nexus.runtime.client.remote.RemoteRuntime.upload_file', new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = mock_response

            from nexus.cli.commands.sandbox import sandbox_upload

            args = self._make_args('http://10.0.0.1:32000', str(local_file), '/workspace/script.sh')
            exit_code = sandbox_upload(args)

            assert exit_code == 1
            captured = capsys.readouterr()
            assert captured.err.startswith('error: ')
            assert 'permission denied' in captured.err.lower()

    def test_upload_connection_error(self, tmp_path: Path, capsys):
        """Connection failure gives friendly error."""
        import aiohttp

        local_file = tmp_path / 'script.sh'
        local_file.write_text('echo hi')

        with patch('nexus.runtime.client.remote.RemoteRuntime.upload_file', new_callable=AsyncMock) as mock_upload:
            mock_upload.side_effect = aiohttp.ClientError('Connection refused')

            from nexus.cli.commands.sandbox import sandbox_upload

            args = self._make_args('http://10.0.0.1:32000', str(local_file), '/workspace/script.sh')
            exit_code = sandbox_upload(args)

            assert exit_code == 1
            captured = capsys.readouterr()
            assert 'cannot reach sandbox' in captured.err.lower()


class TestSandboxDownload:
    """nexus sandbox download <base_url> <remote_path> <local_path>"""

    def _make_args(self, base_url: str, remote_path: str, local_path: str, timeout: float = 180):
        from argparse import Namespace

        return Namespace(
            base_url=base_url,
            remote_path=remote_path,
            local_path=local_path,
            timeout=timeout,
        )

    def test_download_text_file(self, tmp_path: Path):
        """Text file is written to local path (server returns base64)."""
        local_file = tmp_path / 'output.txt'
        text_content = 'result data\n'

        mock_response = DownloadFileResponse(
            files=[
                FileItem(
                    path='/workspace/output.txt',
                    content=base64.b64encode(text_content.encode()).decode(),
                    encoding='base64',
                )
            ],
            errors=None,
        )

        with patch(
            'nexus.runtime.client.remote.RemoteRuntime.download_file', new_callable=AsyncMock
        ) as mock_download:
            mock_download.return_value = mock_response

            from nexus.cli.commands.sandbox import sandbox_download

            args = self._make_args('http://10.0.0.1:32000', '/workspace/output.txt', str(local_file))
            exit_code = sandbox_download(args)

            assert exit_code == 0
            assert local_file.read_bytes() == text_content.encode()

    def test_download_binary_file(self, tmp_path: Path):
        """Binary file is decoded from base64 and written as bytes."""
        local_file = tmp_path / 'image.png'
        binary_content = bytes(range(256))

        mock_response = DownloadFileResponse(
            files=[
                FileItem(
                    path='/workspace/image.png', content=base64.b64encode(binary_content).decode(), encoding='base64'
                )
            ],
            errors=None,
        )

        with patch(
            'nexus.runtime.client.remote.RemoteRuntime.download_file', new_callable=AsyncMock
        ) as mock_download:
            mock_download.return_value = mock_response

            from nexus.cli.commands.sandbox import sandbox_download

            args = self._make_args('http://10.0.0.1:32000', '/workspace/image.png', str(local_file))
            exit_code = sandbox_download(args)

            assert exit_code == 0
            assert local_file.read_bytes() == binary_content

    def test_download_creates_parent_dirs(self, tmp_path: Path):
        """Download creates intermediate directories if missing."""
        local_file = tmp_path / 'nested' / 'deep' / 'output.txt'
        text_content = 'data\n'

        mock_response = DownloadFileResponse(
            files=[
                FileItem(
                    path='/workspace/output.txt',
                    content=base64.b64encode(text_content.encode()).decode(),
                    encoding='base64',
                )
            ],
            errors=None,
        )

        with patch(
            'nexus.runtime.client.remote.RemoteRuntime.download_file', new_callable=AsyncMock
        ) as mock_download:
            mock_download.return_value = mock_response

            from nexus.cli.commands.sandbox import sandbox_download

            args = self._make_args('http://10.0.0.1:32000', '/workspace/output.txt', str(local_file))
            exit_code = sandbox_download(args)

            assert exit_code == 0
            assert local_file.read_bytes() == text_content.encode()

    def test_download_remote_file_not_found(self, tmp_path: Path, capsys):
        """Remote file not found prints error and returns 1."""
        local_file = tmp_path / 'nope.txt'

        mock_response = DownloadFileResponse(
            files=[],
            errors=[FileOperationResult(path='/workspace/nope.txt', success=False, message='File not found')],
        )

        with patch(
            'nexus.runtime.client.remote.RemoteRuntime.download_file', new_callable=AsyncMock
        ) as mock_download:
            mock_download.return_value = mock_response

            from nexus.cli.commands.sandbox import sandbox_download

            args = self._make_args('http://10.0.0.1:32000', '/workspace/nope.txt', str(local_file))
            exit_code = sandbox_download(args)

            assert exit_code == 1
            captured = capsys.readouterr()
            assert captured.err.startswith('error: ')
            assert 'not found' in captured.err.lower()

    def test_download_to_stdout(self, capsys):
        """When local_path is '-', text content is printed to stdout."""
        text_content = 'line1\nline2\n'
        mock_response = DownloadFileResponse(
            files=[
                FileItem(
                    path='/workspace/log.txt',
                    content=base64.b64encode(text_content.encode()).decode(),
                    encoding='base64',
                )
            ],
            errors=None,
        )

        with patch(
            'nexus.runtime.client.remote.RemoteRuntime.download_file', new_callable=AsyncMock
        ) as mock_download:
            mock_download.return_value = mock_response

            from nexus.cli.commands.sandbox import sandbox_download

            args = self._make_args('http://10.0.0.1:32000', '/workspace/log.txt', '-')
            exit_code = sandbox_download(args)

            assert exit_code == 0
            captured = capsys.readouterr()
            assert captured.out == 'line1\nline2\n'

    def test_download_binary_to_stdout(self):
        """Binary download to stdout writes raw bytes to stdout.buffer."""
        import io
        from unittest.mock import MagicMock

        binary_content = bytes(range(128))
        mock_response = DownloadFileResponse(
            files=[
                FileItem(
                    path='/workspace/data.bin', content=base64.b64encode(binary_content).decode(), encoding='base64'
                )
            ],
            errors=None,
        )

        with patch(
            'nexus.runtime.client.remote.RemoteRuntime.download_file', new_callable=AsyncMock
        ) as mock_download:
            mock_download.return_value = mock_response

            mock_buffer = io.BytesIO()
            mock_stdout = MagicMock()
            mock_stdout.write.side_effect = UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid')
            mock_stdout.buffer = mock_buffer

            with patch('nexus.cli.commands.sandbox.sys.stdout', mock_stdout):
                from nexus.cli.commands.sandbox import sandbox_download

                args = self._make_args('http://10.0.0.1:32000', '/workspace/data.bin', '-')
                exit_code = sandbox_download(args)

            assert exit_code == 0
            assert mock_buffer.getvalue() == binary_content

    def test_download_connection_error(self, tmp_path: Path, capsys):
        """Connection failure gives friendly error."""
        import aiohttp

        with patch(
            'nexus.runtime.client.remote.RemoteRuntime.download_file', new_callable=AsyncMock
        ) as mock_download:
            mock_download.side_effect = aiohttp.ClientError('Connection refused')

            from nexus.cli.commands.sandbox import sandbox_download

            args = self._make_args('http://10.0.0.1:32000', '/workspace/out.txt', str(tmp_path / 'out.txt'))
            exit_code = sandbox_download(args)

            assert exit_code == 1
            captured = capsys.readouterr()
            assert 'cannot reach sandbox' in captured.err.lower()

    def test_download_timeout_error(self, tmp_path: Path, capsys):
        """Timeout gives friendly error."""
        import asyncio

        with patch(
            'nexus.runtime.client.remote.RemoteRuntime.download_file', new_callable=AsyncMock
        ) as mock_download:
            mock_download.side_effect = asyncio.TimeoutError()

            from nexus.cli.commands.sandbox import sandbox_download

            args = self._make_args('http://10.0.0.1:32000', '/workspace/out.txt', str(tmp_path / 'out.txt'), timeout=60)
            exit_code = sandbox_download(args)

            assert exit_code == 1
            captured = capsys.readouterr()
            assert 'timed out' in captured.err.lower()
