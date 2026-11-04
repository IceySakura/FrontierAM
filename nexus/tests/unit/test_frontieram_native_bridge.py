from __future__ import annotations

import json
import time

import httpx
import pytest

from nexus.extensions.tasks.frontieram_offline.client import BoundaryError
from nexus.extensions.tasks.frontieram_offline.native_bridge import (
    FLAT_NAME,
    NativeResponsesBridge,
    inbound_response,
    outbound_request,
    responses_events,
    usable_response,
)


POLICY = {
    'code': 0,
    'data': {
        'dispositions': [{'code': 3003, 'is_retryable': True}],
        'backoff': {'retryable_http_status': [429, 500, 502, 503, 504]},
    },
}
WORKSPACE = {
    'type': 'namespace',
    'name': 'mcp__workspace',
    'tools': [{'type': 'function', 'name': 'exec', 'parameters': {'type': 'object'}}],
}


def codex_request(**extra):
    return {
        'model': 'fixture-model',
        'instructions': 'be brief',
        'input': [
            {'type': 'message', 'role': 'user', 'content': 'hello'},
            {'type': 'reasoning', 'id': 'rs_1', 'summary': []},
            {'type': 'function_call', 'namespace': 'mcp__workspace', 'name': 'exec', 'call_id': 'c1', 'arguments': '{}'},
            {'type': 'function_call_output', 'call_id': 'c1', 'output': 'ok'},
        ],
        'tools': [WORKSPACE, {'type': 'function', 'name': 'request_user_input'}],
        'reasoning': {'effort': 'high'},
        'client_metadata': {'thread_id': 't'},
        'stream': True,
        **extra,
    }


def completed(name=FLAT_NAME, arguments='{"command": "ls"}', status='completed'):
    return {
        'id': 'resp_1',
        'status': status,
        'output': [{'type': 'function_call', 'id': 'fc_1', 'call_id': 'c2', 'name': name, 'arguments': arguments}],
        'usage': {'output_tokens': 3},
    }


def test_outbound_request_keeps_only_the_flattened_workspace_tool_and_v4_settings():
    events = []
    body = outbound_request(codex_request(), max_output_tokens=64000, record=lambda kind, **f: events.append(kind))
    assert body['tools'] == [{'type': 'function', 'parameters': {'type': 'object'}, 'name': FLAT_NAME}]
    assert [i['type'] for i in body['input']] == ['message', 'function_call', 'function_call_output']
    assert body['input'][1]['name'] == FLAT_NAME and 'namespace' not in body['input'][1]
    assert body['stream'] is False and body['store'] is False and body['max_output_tokens'] == 64000
    assert 'reasoning' not in body and 'client_metadata' not in body
    assert events == ['tool_schema_filtered']


def test_outbound_request_rejects_foreign_tools_in_history():
    request = codex_request()
    request['input'].append({'type': 'function_call', 'name': 'shell', 'call_id': 'x', 'arguments': '{}'})
    with pytest.raises(BoundaryError, match='history_tool_identity'):
        outbound_request(request, max_output_tokens=64000, record=lambda *a, **k: None)


@pytest.mark.parametrize(
    ('payload', 'reason'),
    [
        (completed(status='incomplete'), 'incomplete_response'),
        (completed(name='mcp__workspace<parameter=command'), 'rejected_tool'),
        (completed(arguments='{not json'), 'rejected_tool'),
        (completed(status='failed'), 'failed_response'),
        (completed(), None),
    ],
)
def test_usable_response(payload, reason):
    assert usable_response(payload, lambda *a, **k: None) == reason


def test_events_restore_the_namespace_codex_dispatches_by():
    stream = responses_events(inbound_response(completed())).decode()
    done = [json.loads(line[6:]) for line in stream.splitlines() if line.startswith('data: ')]
    item = next(e['item'] for e in done if e['type'] == 'response.output_item.done')
    assert (item['name'], item['namespace']) == ('exec', 'mcp__workspace')
    assert done[-1]['type'] == 'response.completed'


def bridge(handler, **kwargs):
    return NativeResponsesBridge(
        upstream_url='http://fixture.invalid/v1',
        api_key='not-a-real-key',
        token='t' * 32,
        deadline=time.monotonic() + 600,
        retry_policy=POLICY,
        transport=httpx.MockTransport(handler),
        sleep=lambda _s: _noop(),
        jitter=lambda _low, _high: 0.0,
        **kwargs,
    )


async def _noop():
    return None


@pytest.mark.asyncio
async def test_unusable_samples_are_redrawn_with_the_identical_payload():
    bodies = []
    replies = [completed(status='incomplete'), completed(name='mcp__workspace<parameter=command'), completed()]

    def handler(request):
        bodies.append(request.content)
        return httpx.Response(200, json=replies[len(bodies) - 1])

    value = bridge(handler)
    try:
        payload = await value.complete(outbound_request(codex_request(), max_output_tokens=64000, record=value.record))
        assert payload['status'] == 'completed'
        assert value.http_attempts == 3 and value.retry_count == 2 and not value.errors
        assert bodies[0] == bodies[1] == bodies[2]
    finally:
        await value.aclose()


@pytest.mark.asyncio
async def test_persistent_truncation_ends_the_request_after_three_retries():
    value = bridge(lambda request: httpx.Response(200, json=completed(status='incomplete')))
    try:
        with pytest.raises(BoundaryError, match='incomplete_response'):
            await value.complete(outbound_request(codex_request(), max_output_tokens=64000, record=value.record))
        assert value.http_attempts == 4
        assert [e['category'] for e in value.errors] == ['incomplete_response']
    finally:
        await value.aclose()


@pytest.mark.asyncio
async def test_rate_limit_is_retried_and_other_client_errors_are_not():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={'error': {'code': '3003'}})
        return httpx.Response(400, json={'error': {'code': '2001'}})

    value = bridge(handler)
    try:
        with pytest.raises(BoundaryError, match='upstream_http'):
            await value.complete(outbound_request(codex_request(), max_output_tokens=64000, record=value.record))
        assert len(calls) == 2 and value.retry_count == 1
    finally:
        await value.aclose()


@pytest.mark.asyncio
async def test_http_budget_is_shared_by_retries():
    value = bridge(lambda request: httpx.Response(200, json=completed(status='incomplete')), max_http_attempts=2)
    try:
        with pytest.raises(BoundaryError, match='retry_budget|http_budget'):
            await value.complete(outbound_request(codex_request(), max_output_tokens=64000, record=value.record))
        assert value.http_attempts == 2
    finally:
        await value.aclose()
