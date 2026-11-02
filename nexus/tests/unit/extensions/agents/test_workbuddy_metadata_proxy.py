"""Verify opt-in capture preserves the request and redacts authentication headers."""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

import httpx
import pytest


pytest.importorskip('fastapi', reason='Metadata proxy tests require Nexus runtime server dependencies')

from nexus.extensions.agents.workbuddy.metadata_proxy import WorkBuddyMetadataProxy


@pytest.mark.asyncio
async def test_proxy_captures_original_metadata_and_keeps_request_body(tmp_path: Path) -> None:
    observed = []

    class Bridge:
        async def request(self, body: dict[str, Any], timeout: float) -> dict[str, Any]:
            observed.append(body)
            return {'id': 'response-1', 'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]}

    proxy = WorkBuddyMetadataProxy(sio=None, capture_dir=str(tmp_path))
    app = proxy._create_app()
    proxy.bridge = Bridge()
    body = {
        'model': 'visible-model',
        'messages': [{'role': 'user', 'content': 'read fixture'}],
        'metadata': {'query_source': 'subagent', 'parent_tool_call_id': 'spawn-1'},
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://proxy') as client:
        response = await client.post(
            '/v1/chat/completions',
            json=body,
            headers={'Authorization': 'Bearer test-secret', 'X-Api-Key': 'test-key', 'X-Request-ID': 'request-1'},
        )
    assert response.status_code == 200
    assert len(observed) == 1
    assert {key: value for key, value in observed[0].items() if key != 'request_meta'} == body
    assert observed[0]['request_meta']['request_id'] == 'request-1'
    capture = json.loads(next(tmp_path.glob('*.request.json')).read_text())
    assert capture['body'] == body
    assert capture['headers']['authorization'] == '[REDACTED]'
    assert capture['headers']['x-api-key'] == '[REDACTED]'
    assert capture['headers']['x-request-id'] == 'request-1'


@pytest.mark.asyncio
async def test_non_completion_route_does_not_capture(tmp_path: Path) -> None:
    proxy = WorkBuddyMetadataProxy(sio=None, capture_dir=str(tmp_path))
    app = proxy._create_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://proxy') as client:
        response = await client.get('/v1/models')
    assert response.status_code == 200
    assert list(tmp_path.iterdir()) == []
