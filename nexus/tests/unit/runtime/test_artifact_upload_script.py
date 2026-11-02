import ast
import asyncio
import base64
import hashlib
import json
import threading

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus.runner.hooks.gui_recording import GuiRecordingHook
from nexus.runtime.bridge.runtime_kind import NativeBridgeRuntimeKind
from nexus.runtime.executors.models.command import RunCommandResponse
from nexus.runtime.utils import _artifact_upload


@pytest.fixture
def cos_server(request):
    received = []
    options = getattr(request, 'param', {})

    class Handler(BaseHTTPRequestHandler):
        def do_PUT(self):
            data = self.rfile.read(int(self.headers['Content-Length']))
            received.append((self.path, dict(self.headers), data))
            self.send_response(options.get('status', 200))
            if options.get('etag', True):
                self.send_header('ETag', '"test-etag"')
            self.send_header('Content-Length', '0')
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/video?signature=test', received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_upload_streams_exact_file_range_and_preserves_signed_headers(tmp_path, cos_server, monkeypatch):
    url, received = cos_server
    path = tmp_path / 'video.mp4'
    path.write_bytes(bytes(range(50)))
    monkeypatch.setattr(_artifact_upload, 'CHUNK_SIZE', 4)
    result = _artifact_upload.upload_file(
        {
            'path': str(path),
            'offset': 5,
            'length': 13,
            'url': url,
            'timeout': 5,
            'headers': {'Content-Type': 'video/mp4', 'x-cos-meta-example': 'keep'},
        }
    )
    assert result == {'etag': '"test-etag"'}
    target, headers, data = received[0]
    assert target == '/video?signature=test'
    assert headers['Content-Type'] == 'video/mp4'
    assert headers['X-Cos-Meta-Example'] == 'keep'
    assert headers['Content-Length'] == '13'
    assert 'Transfer-Encoding' not in headers
    assert data == bytes(range(5, 18))
    with pytest.raises(ValueError, match='range'):
        _artifact_upload.upload_file({'path': str(path), 'offset': 49, 'length': 2})


@pytest.mark.asyncio
async def test_host_bootstrap_runs_in_sandbox_python_and_returns_only_metadata(tmp_path, cos_server):
    url, received = cos_server
    content = b'video content' * 100
    path = tmp_path / "recording with 'quotes'.mp4"
    path.write_bytes(content)
    commands = []

    async def run_command(*, command, timeout):
        commands.append(command)
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        return RunCommandResponse(
            status='Finished',
            return_code=process.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode(),
        )

    hook = GuiRecordingHook(upload_timeout=5)
    hook._runtime = SimpleNamespace(native_bridge_kind=NativeBridgeRuntimeKind.AGS_OSWORLD, run_command=run_command)
    metadata = await hook._run_upload_script(action='inspect', path=str(path))
    assert metadata == {
        'success': True,
        'size_bytes': len(content),
        'content_md5': base64.b64encode(hashlib.md5(content).digest()).decode(),
    }
    result = await hook._run_upload_script(
        action='upload',
        path=str(path),
        url=url,
        method='PUT',
        headers={},
        length=len(content),
        timeout=5,
    )
    assert result == {'success': True, 'etag': '"test-etag"', 'source_deleted': True}
    assert received[0][2] == content
    assert not path.exists()
    assert len(commands) == 2


@pytest.mark.parametrize('cos_server', [{'status': 500}, {'etag': False}], indirect=True)
def test_upload_failure_preserves_source(tmp_path, cos_server, capsys):
    url, _received = cos_server
    path = tmp_path / 'video.mp4'
    path.write_bytes(b'video')
    _artifact_upload.main(json.dumps({'action': 'upload', 'path': str(path), 'length': 5, 'url': url, 'timeout': 5}))
    result = json.loads(capsys.readouterr().out.split(':', 1)[1])
    assert result['success'] is False
    assert path.read_bytes() == b'video'


def test_multipart_deletes_source_only_after_last_part(tmp_path, cos_server, capsys):
    url, received = cos_server
    path = tmp_path / 'video.mp4'
    path.write_bytes(b'abcdef')
    for offset, last in [(0, False), (3, True)]:
        _artifact_upload.main(
            json.dumps(
                {
                    'action': 'upload',
                    'path': str(path),
                    'url': url,
                    'timeout': 5,
                    'offset': offset,
                    'length': 3,
                    'delete_source': last,
                }
            )
        )
        result = json.loads(capsys.readouterr().out.split(':', 1)[1])
        assert result['success'] is True
        assert path.exists() is not last
    assert [item[2] for item in received] == [b'abc', b'def']


def test_ready_upload_deletes_source_without_put(tmp_path, capsys):
    path = tmp_path / 'video.mp4'
    path.write_bytes(b'video')
    _artifact_upload.main(json.dumps({'action': 'upload', 'path': str(path), 'already_uploaded': True}))
    assert json.loads(capsys.readouterr().out.split(':', 1)[1]) == {'success': True, 'source_deleted': True}
    assert not path.exists()


def test_upload_script_supports_sandbox_python_37():
    ast.parse(Path(_artifact_upload.__file__).read_text(), feature_version=(3, 7))
