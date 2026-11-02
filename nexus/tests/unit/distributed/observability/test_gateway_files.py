from __future__ import annotations

import pytest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from nexus.distributed.config import DistributedConfig
from nexus.distributed.gateways.common.platform_metrics import MeasuredFileResponse, install_platform_metrics


@pytest.mark.asyncio
async def test_measured_file_transfers_preserve_ranges_and_connection_reuse(tmp_path, caplog):
    path = tmp_path / 'asset.js'
    content = b'const example = 1;\n' * 1000
    path.write_bytes(content)

    async def file(_request):
        return MeasuredFileResponse(path)

    app = web.Application()
    install_platform_metrics(app, session_factory=None, config=DistributedConfig(), service='file-test')
    app.router.add_get('/asset', file)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        first = await client.get('/asset', headers={'Range': 'bytes=0-99'})
        assert first.status == 206
        assert await first.read() == content[:100]
        second = await client.get('/asset')
        assert second.status == 200
        assert await second.read() == content
        third = await client.get('/asset', headers={'If-None-Match': second.headers['ETag']})
        assert third.status == 304
        assert await third.read() == b''
    finally:
        await client.close()
    assert not [record for record in caplog.records if record.levelname in {'ERROR', 'CRITICAL'}]
