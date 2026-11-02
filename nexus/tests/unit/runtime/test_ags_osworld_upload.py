from __future__ import annotations

import pytest

from nexus.runtime.providers.ags.handle import AGSInstanceHandle
from nexus.runtime.providers.ags.osworld import AGSRawOSWorldRuntime


@pytest.mark.asyncio
async def test_osworld_setup_upload_uses_multipart_file_field(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    async def fake_request(self, method: str, path: str = '', *, port: int | None = None, **kwargs):
        captured['method'] = method
        captured['path'] = path
        captured['port'] = port
        captured['kwargs'] = kwargs

        class Response:
            text = 'File Uploaded: 3 bytes'
            status_code = 200
            ok = True
            url = 'https://example'
            headers = {}
            content = b''

        return Response()

    monkeypatch.setattr(AGSRawOSWorldRuntime, 'request', fake_request)
    rt = AGSRawOSWorldRuntime(AGSInstanceHandle(instance_id='iid', access_token='token', domain='example.com'))

    text = await rt.setup_upload('/tmp/a.txt', b'abc', filename='a.txt')

    assert text == 'File Uploaded: 3 bytes'
    assert captured['method'] == 'POST'
    assert captured['path'] == '/setup/upload'
    assert captured['port'] == 5000
    data = captured['kwargs']['data']
    assert data is not None
