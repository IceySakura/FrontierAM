from __future__ import annotations

import inspect
import json

import httpx
import pytest

from nexus.extensions.agents.codex.agent import CodexInput
from nexus.extensions.agents.codex.installer import DEFAULT_CODEX_VERSION
from nexus.extensions.agents.codex.runtime_setup import setup_and_run_codex
from nexus.runtime.server.services.responses_proxy.codex import build_codex_request_meta
from nexus.runtime.server.services.responses_proxy.server import ResponsesProxy


class _Bridge:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def request(self, params: dict, timeout: float) -> dict:
        assert timeout == 300
        self.requests.append(params)
        return {'id': 'response-1', 'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]}


def _message(text: str) -> dict:
    return {
        'type': 'message',
        'role': 'user',
        'content': [{'type': 'input_text', 'text': text}],
    }


def test_codex_default_version_pins_legacy_release() -> None:
    """Omitted or null versions preserve the legacy default instead of adopting an installed CLI's version."""
    assert DEFAULT_CODEX_VERSION == '0.114.0'
    assert CodexInput(user_prompt='test', working_dir='/workspace').version == DEFAULT_CODEX_VERSION
    assert CodexInput(user_prompt='test', working_dir='/workspace', version=None).version == DEFAULT_CODEX_VERSION
    assert inspect.signature(setup_and_run_codex).parameters['version'].default == DEFAULT_CODEX_VERSION


@pytest.mark.parametrize('version', ['0.114.0', '0.146.1', '0.147.0', '0.148.0'])
def test_responses_proxy_accepts_nearby_codex_versions(version: str) -> None:
    proxy = ResponsesProxy(sio=object(), codex_version=version)

    assert proxy.codex_version == version


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('version', 'expected_payload'),
    [
        ('0.114.0', {'object': 'list', 'data': []}),
        ('0.115.0', {'object': 'list', 'data': []}),
        ('0.146.1', {'models': []}),
        ('0.147.0', {'models': []}),
        ('0.148.0', {'models': []}),
    ],
)
async def test_models_endpoint_uses_version_appropriate_catalog(version: str, expected_payload: dict) -> None:
    proxy = ResponsesProxy(sio=object(), codex_version=version)
    transport = httpx.ASGITransport(app=proxy._create_app())

    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.get('/v1/models')

    assert response.status_code == 200
    assert response.json() == expected_payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('version', 'metadata_location'),
    [('0.114.0', 'none'), ('0.146.1', 'client'), ('0.147.0', 'header'), ('0.148.0', 'client')],
)
async def test_proxy_attaches_version_tolerant_codex_request_metadata(
    version: str,
    metadata_location: str,
) -> None:
    proxy = ResponsesProxy(sio=object(), codex_version=version)
    bridge = _Bridge()
    proxy.bridge = bridge
    transport = httpx.ASGITransport(app=proxy._create_app())
    turn_metadata = {
        'request_kind': 'turn',
        'session_id': 'root-thread',
        'thread_id': 'root-thread',
        'window_id': 'root-thread:0',
    }
    body = {'model': 'test', 'input': [_message('work')]}
    headers = {'user-agent': f'codex_cli_rs/{version}'}
    if metadata_location == 'client':
        body['client_metadata'] = {'x-codex-turn-metadata': json.dumps(turn_metadata)}
    elif metadata_location == 'header':
        headers['x-codex-turn-metadata'] = json.dumps(turn_metadata)

    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.post('/v1/responses', headers=headers, json=body)

    assert response.status_code == 200
    request_meta = bridge.requests[0]['request_meta']
    assert request_meta['scaffold'] == 'codex'
    assert request_meta['scaffold_version'] == version
    assert isinstance(request_meta['request_id'], str)
    assert request_meta['request_id']
    if metadata_location == 'none':
        assert request_meta['codex'] == {}
        assert request_meta['codex_metadata_source'] == 'none'
    else:
        assert request_meta['codex'] == turn_metadata
        assert request_meta['codex_metadata_source'] in {'client_metadata', 'turn_metadata'}


def test_request_metadata_prefers_even_incomplete_header_identity_over_client_identity() -> None:
    header_metadata = {'request_kind': 'turn'}
    client_metadata = {
        'request_kind': 'turn',
        'session_id': 'root-thread',
        'thread_id': 'child-thread',
        'window_id': 'child-thread:0',
        'parent_thread_id': 'root-thread',
    }

    request_meta = build_codex_request_meta(
        {'client_metadata': {'x-codex-turn-metadata': json.dumps(client_metadata)}},
        {'x-codex-turn-metadata': json.dumps(header_metadata)},
        configured_version='0.147.0',
        request_id='provider-1',
        compaction_evidence=None,
    )

    assert request_meta['codex'] == header_metadata
    assert request_meta['codex_metadata_source'] == 'turn_metadata'


def test_native_compaction_metadata_marks_summary_inference_for_gateway_accounting() -> None:
    request_meta = build_codex_request_meta(
        {},
        {
            'x-codex-turn-metadata': json.dumps({
                'request_kind': 'compaction',
                'session_id': 'root-thread',
                'thread_id': 'root-thread',
                'window_id': 'root-thread:0',
            })
        },
        configured_version='0.147.0',
        request_id='provider-compaction',
        compaction_evidence=None,
    )

    assert request_meta['invocation_kind'] == 'summary_inference'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('normalize_metadata', 'expected_prefix'),
    [
        (False, 'Chunk ID: random-value\nWall time: 0.1234 seconds\n'),
        (True, 'Chunk ID: deterministic\nWall time: deterministic seconds\n'),
    ],
)
async def test_proxy_only_normalizes_exec_command_metadata_when_enabled(
    normalize_metadata: bool,
    expected_prefix: str,
) -> None:
    proxy = ResponsesProxy(
        sio=object(),
        codex_version='0.146.1',
        normalize_exec_command_metadata=normalize_metadata,
    )
    bridge = _Bridge()
    proxy.bridge = bridge
    transport = httpx.ASGITransport(app=proxy._create_app())
    body = {
        'model': 'test',
        'input': [
            {
                'type': 'function_call',
                'call_id': 'call_exec',
                'name': 'exec_command',
                'arguments': '{"cmd":"true"}',
            },
            {
                'type': 'function_call_output',
                'call_id': 'call_exec',
                'output': 'Chunk ID: random-value\nWall time: 0.1234 seconds\nProcess exited with code 0\n',
            },
        ],
    }

    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.post('/v1/responses', json=body)

    assert response.status_code == 200
    assert bridge.requests[0]['messages'][1]['content'].startswith(expected_prefix)
