"""Local-only checks for task budgets and the native Nexus protocol boundary."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import json
import time

import httpx
import pytest

from nexus.core import OpenAIConfig
from nexus.extensions.tasks.frontieram_offline.client import BoundaryError, BoundedOpenAIClient
from nexus.extensions.tasks.frontieram_offline.proxy import OfflineResponsesProxy, validate_tool_response
from nexus.trace import processor_context, span_context
from nexus.trace.processors.base import SpanProcessor
from nexus.trace.processors.token_counter import TokenCounterProcessor


POLICY = {
    'code': 0,
    'data': {
        'backoff': {'retryable_http_status': [429, 500, 502, 503, 504]},
        'dispositions': [
            {'code': 3006, 'is_retryable': True},
            {'code': 1010, 'is_retryable': True},
            {'code': 1000, 'is_retryable': False},
        ],
    },
}
TOOL_NAME = 'mcp__workspace__exec'
TOOLS = [{'type': 'function', 'function': {'name': TOOL_NAME, 'parameters': {'type': 'object'}}}]
TOKEN = 'local-fixture-token-only'


def completion(*, arguments=None, call_id='call-1', name=TOOL_NAME, finish_reason=None):
    message = {'role': 'assistant', 'content': 'complete'}
    if arguments is not None:
        message['tool_calls'] = [
            {'type': 'function', 'id': call_id, 'function': {'name': name, 'arguments': arguments}},
        ]
    return {
        'id': 'fixture-completion',
        'object': 'chat.completion',
        'created': 1,
        'model': 'fixture-model',
        'choices': [
            {
                'index': 0,
                'finish_reason': finish_reason or ('tool_calls' if arguments is not None else 'stop'),
                'message': message,
            },
        ],
        'usage': {'prompt_tokens': 4, 'completion_tokens': 2, 'total_tokens': 6},
    }


def client_for(handler, **kwargs):
    return BoundedOpenAIClient(
        OpenAIConfig(
            client_args={'api_key': 'not-a-real-key', 'base_url': 'http://fixture.invalid/v1', 'max_retries': 15},
            request_args={'model': 'fixture-model'},
        ),
        deadline=kwargs.pop('deadline', time.monotonic() + 30),
        retry_policy=kwargs.pop('retry_policy', POLICY),
        transport=httpx.MockTransport(handler),
        jitter=lambda _low, _high: 0.0,
        **kwargs,
    )


async def ask(client):
    return await client.create_completion(messages=[{'role': 'user', 'content': 'fixture'}])


@pytest.mark.asyncio
async def test_native_sdk_retry_is_not_multiplied_and_payload_is_identical():
    payloads = []

    def handler(request):
        payloads.append(request.content)
        if len(payloads) == 1:
            return httpx.Response(500, json={'error': {'code': 3006, 'message': 'provider-private-text'}})
        return httpx.Response(200, json=completion())

    client = client_for(handler)
    try:
        result = await ask(client)
        assert result.choices[0].finish_reason == 'stop'
        assert client.client.max_retries == 0
        assert client.config.client_args['max_retries'] == 1
        assert client.http_attempts == 2
        assert client.retry_count == 1
        assert payloads[0] == payloads[1]
        body = json.loads(payloads[0])
        assert body['model'] == 'fixture-model'
        assert body['max_tokens'] == 64000
        assert body['stream'] is False
        assert body['n'] == 1
        assert 'provider-private-text' not in json.dumps(client.events)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_cache_task_id_rides_on_every_attempt_of_one_conversation():
    authorizations = []

    def handler(request):
        authorizations.append(request.headers['authorization'])
        if len(authorizations) == 1:
            return httpx.Response(500, json={'error': {'code': 3006}})
        return httpx.Response(200, json=completion())

    client = BoundedOpenAIClient(
        OpenAIConfig(
            client_args={
                'api_key': 'not-a-real-key',
                'base_url': 'http://fixture.invalid/v1',
                'cache_task_id': 'frontieram-attempt-1',
            },
            request_args={'model': 'fixture-model'},
        ),
        deadline=time.monotonic() + 30,
        retry_policy=POLICY,
        transport=httpx.MockTransport(handler),
        jitter=lambda _low, _high: 0.0,
    )
    try:
        await ask(client)
        await ask(client)
        assert len(authorizations) == 3
        assert set(authorizations) == {'Bearer not-a-real-key?cache_task_id=frontieram-attempt-1'}
        assert client.cache_task_id == 'frontieram-attempt-1'
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(('status', 'code'), [(500, 1000), (500, 99999), (401, 3006), (400, 3006), (501, 3006)])
async def test_only_documented_status_and_gateway_code_retries(status, code):
    client = client_for(lambda _: httpx.Response(status, json={'error': {'code': code, 'message': 'private-text'}}))
    try:
        with pytest.raises(BoundaryError, match='upstream_http') as error:
            await ask(client)
        assert 'private-text' not in str(error.value)
        assert client.http_attempts == 1
        assert client.retry_count == 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_retry_cap_is_four_actual_http_attempts():
    client = client_for(lambda _: httpx.Response(503, json={'error': {'code': 3006}}))
    try:
        with pytest.raises(BoundaryError, match='upstream_http'):
            await ask(client)
        assert client.http_attempts == 4
        assert client.retry_count == 3
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_total_http_cap_is_shared_across_logical_requests():
    client = client_for(lambda _: httpx.Response(200, json=completion()), max_http_attempts=2)
    try:
        await ask(client)
        await ask(client)
        with pytest.raises(BoundaryError, match='http_budget'):
            await ask(client)
        assert client.http_attempts == 2
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_retry_after_cannot_outlive_development_deadline():
    client = client_for(
        lambda _: httpx.Response(429, headers={'retry-after': '500'}, json={'error': {'code': 1010}}),
    )
    try:
        with pytest.raises(BoundaryError, match='retry_budget'):
            await ask(client)
        assert client.http_attempts == 1
        assert client.retry_count == 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_late_success_is_rejected_before_tool_dispatch():
    clock = [100.0]

    def handler(_request):
        clock[0] = 102.0
        return httpx.Response(200, json=completion(arguments='{"command":"do-not-run"}'))

    client = client_for(handler, deadline=101.0, clock=lambda: clock[0])
    try:
        with pytest.raises(BoundaryError, match='deadline'):
            await ask(client)
        assert client.http_attempts == 1
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_inflight_request_is_cancelled_at_immutable_deadline():
    async def handler(_request):
        await asyncio.sleep(2)
        return httpx.Response(200, json=completion())

    client = client_for(handler, deadline=time.monotonic() + 0.03)
    try:
        with pytest.raises(BoundaryError, match='deadline'):
            await ask(client)
        assert client.http_attempts == 1
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_stop_prevents_all_new_requests():
    client = client_for(lambda _: httpx.Response(200, json=completion()))
    client.stop()
    try:
        with pytest.raises(BoundaryError, match='stopped'):
            await ask(client)
        assert client.http_attempts == 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_client_override_cannot_replace_model_or_output_policy():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=completion())

    client = client_for(handler)
    try:
        with pytest.raises(BoundaryError, match='model_override'):
            await client.create_completion([], model='different-model')
        await client.create_completion([], max_tokens=999999, reasoning_effort='max', verbosity='high')
        assert bodies[0]['max_tokens'] == 64000
        assert 'reasoning_effort' not in bodies[0]
        assert 'verbosity' not in bodies[0]
    finally:
        await client.aclose()


@pytest.mark.parametrize('arguments', ['[]', 'null', 'NaN', 'bad json', [], 42, {'x': float('nan')}])
def test_structurally_invalid_arguments_never_reach_native_events(arguments):
    with pytest.raises(BoundaryError):
        validate_tool_response(completion(arguments=arguments), {TOOL_NAME}, lambda *_args, **_kwargs: None)


def test_ordinary_tool_field_errors_are_left_for_mcp_feedback():
    validate_tool_response(
        completion(arguments='{"command":"fixture","timeout_seconds":121,"unknown":true}'),
        {TOOL_NAME},
        lambda *_args, **_kwargs: None,
    )


@pytest.mark.parametrize('kind', ['length', 'empty_choices', 'tool_name', 'duplicate_id'])
def test_incomplete_or_ambiguous_tool_responses_are_rejected(kind):
    response = completion(arguments='{}')
    if kind == 'length':
        response['choices'][0]['finish_reason'] = 'length'
    elif kind == 'empty_choices':
        response['choices'] = []
    elif kind == 'tool_name':
        response['choices'][0]['message']['tool_calls'][0]['function']['name'] = 'unknown-private-name'
    else:
        response['choices'][0]['message']['tool_calls'] *= 2
    with pytest.raises(BoundaryError) as error:
        validate_tool_response(response, {TOOL_NAME}, lambda *_args, **_kwargs: None)
    assert 'unknown-private-name' not in str(error.value)


def request_body():
    return {
        'model': 'fixture-model',
        'input': [{'type': 'message', 'role': 'user', 'content': 'fixture'}],
        'tools': [{'type': 'function', 'name': TOOL_NAME, 'parameters': {'type': 'object'}}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('arguments', 'call_id'), [({'command': 'fixture'}, None), ('{"timeout_seconds":121}', 'call-1')]
)
async def test_native_client_proxy_keeps_compatible_tools_and_fixed_error_feedback(arguments, call_id):
    client = client_for(lambda _: httpx.Response(200, json=completion(arguments=arguments, call_id=call_id)))
    proxy = OfflineResponsesProxy(client, token=TOKEN)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.create_app()), base_url='http://local'
        ) as http:
            response = await http.post(
                '/v1/responses', headers={'authorization': f'Bearer {TOKEN}'}, json=request_body()
            )
        assert response.status_code == 200, response.text
        payloads = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        calls = [event['item'] for event in payloads if event.get('item', {}).get('type') == 'function_call']
        assert len(calls) == 1
        assert calls[0]['call_id']
        assert json.loads(calls[0]['arguments']) == (
            arguments if isinstance(arguments, dict) else json.loads(arguments)
        )
        assert any(event['type'] == 'response.completed' for event in payloads)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_proxy_auth_compaction_and_length_guards_do_not_leak_payloads():
    client = client_for(
        lambda _: httpx.Response(200, json=completion(arguments='{"command":"PRIVATE"}', finish_reason='length')),
    )
    proxy = OfflineResponsesProxy(client, token=TOKEN)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.create_app()), base_url='http://local'
        ) as http:
            unauthorized = await http.post('/v1/responses', json=request_body())
            assert unauthorized.status_code == 401
            assert client.http_attempts == 0
            compact = await http.post('/v1/responses/compact', headers={'authorization': f'Bearer {TOKEN}'}, json={})
            assert compact.status_code == 400
            assert client.http_attempts == 0
            response = await http.post(
                '/v1/responses', headers={'authorization': f'Bearer {TOKEN}'}, json=request_body()
            )
            assert response.status_code == 400
            assert 'PRIVATE' not in response.text
            assert 'response.output_item' not in response.text
            assert 'incomplete_response' in response.text
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('with_tools', [True, False])
async def test_proxy_forwards_local_compaction_within_the_same_budget(with_tools):
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=completion())

    client = client_for(handler, max_http_attempts=3)
    proxy = OfflineResponsesProxy(client, token=TOKEN)
    body = request_body()
    if not with_tools:
        body['tools'] = []
        body['tool_choice'] = 'auto'
    body['input'] += [
        {'type': 'function_call', 'call_id': 'call-1', 'name': TOOL_NAME, 'arguments': '{"command":"ls"}'},
        {'type': 'function_call_output', 'call_id': 'call-1', 'output': 'task'},
        {'type': 'message', 'role': 'user', 'content': 'You are performing a CONTEXT CHECKPOINT COMPACTION.'},
    ]
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.create_app()), base_url='http://local'
        ) as http:
            response = await http.post('/v1/responses', headers={'authorization': f'Bearer {TOKEN}'}, json=body)
        assert response.status_code == 200, response.text
        assert client.http_attempts == 1
        assert [e['kind'] for e in client.events].count('compaction') == 1
        assert not any(key.startswith('_nexus') for key in payloads[0])
        assert ('tools' in payloads[0]) is with_tools
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_proxy_still_requires_the_workspace_tool_outside_compaction():
    client = client_for(lambda _: httpx.Response(200, json=completion()))
    proxy = OfflineResponsesProxy(client, token=TOKEN)
    body = request_body()
    body['tools'] = []
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.create_app()), base_url='http://local'
        ) as http:
            response = await http.post('/v1/responses', headers={'authorization': f'Bearer {TOKEN}'}, json=body)
        assert response.status_code == 400
        assert client.http_attempts == 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_proxy_does_not_advertise_native_host_tools():
    client = client_for(lambda _: httpx.Response(200, json=completion()))
    proxy = OfflineResponsesProxy(client, token=TOKEN)
    body = copy.deepcopy(request_body())
    body['tools'][0]['name'] = 'exec_command'
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.create_app()), base_url='http://local'
        ) as http:
            response = await http.post('/v1/responses', headers={'authorization': f'Bearer {TOKEN}'}, json=body)
        assert response.status_code == 400
        assert client.http_attempts == 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_real_codex_namespaced_workspace_survives_only_schema_filtering():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=completion(arguments='{"command":"fixture"}'))

    client = client_for(handler)
    proxy = OfflineResponsesProxy(client, token=TOKEN)
    body = request_body()
    body['tools'] = [
        {'type': 'function', 'name': name, 'parameters': {'type': 'object'}}
        for name in ('list_mcp_resources', 'list_mcp_resource_templates', 'read_mcp_resource', 'request_user_input')
    ]
    body['tools'].append(
        {
            'type': 'namespace',
            'name': 'mcp__workspace',
            'tools': [{'type': 'function', 'name': 'exec', 'parameters': {'type': 'object'}}],
        }
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.create_app()), base_url='http://local'
        ) as http:
            result = await http.post('/v1/responses', headers={'authorization': f'Bearer {TOKEN}'}, json=body)
        assert result.status_code == 200, result.text
        assert [tool['function']['name'] for tool in bodies[0]['tools']] == [TOOL_NAME]
        items = [
            json.loads(line[6:])['item']
            for line in result.text.splitlines()
            if line.startswith('data: ') and '"item"' in line
        ]
        call = next(item for item in items if item.get('type') == 'function_call')
        assert call['namespace'] == 'mcp__workspace'
        assert call['name'] == 'exec'
        assert len([event for event in client.events if event['kind'] == 'tool_schema_filtered']) == 4
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('location', ['history', 'response'])
async def test_filtered_resource_tool_is_not_allowed_in_history_or_model_response(location):
    client = client_for(lambda _: httpx.Response(200, json=completion(arguments='{}', name='list_mcp_resources')))
    proxy = OfflineResponsesProxy(client, token=TOKEN)
    body = request_body()
    body['tools'].append({'type': 'function', 'name': 'list_mcp_resources', 'parameters': {'type': 'object'}})
    if location == 'history':
        body['input'].extend(
            [
                {'type': 'function_call', 'name': 'list_mcp_resources', 'call_id': 'fake-call', 'arguments': '{}'},
                {'type': 'function_call_output', 'call_id': 'fake-call', 'output': 'fake'},
            ]
        )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.create_app()), base_url='http://local'
        ) as http:
            result = await http.post('/v1/responses', headers={'authorization': f'Bearer {TOKEN}'}, json=body)
        assert result.status_code == 400
        assert client.http_attempts == (0 if location == 'history' else 1)
        assert 'response.output_item' not in result.text
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(('namespace', 'name'), [('workspace', 'exec'), ('mcp__workspace', 'other'), ('other', 'exec')])
async def test_workspace_namespace_and_leaf_name_must_match_exactly(namespace, name):
    client = client_for(lambda _: httpx.Response(200, json=completion()))
    proxy = OfflineResponsesProxy(client, token=TOKEN)
    body = request_body()
    body['tools'] = [
        {'type': 'namespace', 'name': namespace, 'tools': [{'type': 'function', 'name': name, 'parameters': {}}]},
    ]
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.create_app()), base_url='http://local'
        ) as http:
            result = await http.post('/v1/responses', headers={'authorization': f'Bearer {TOKEN}'}, json=body)
        assert result.status_code == 400
        assert client.http_attempts == 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_real_loopback_proxy_start_stop_and_authentication():
    client = client_for(lambda _: httpx.Response(200, json=completion()))
    proxy = OfflineResponsesProxy(client, token=TOKEN)
    try:
        await proxy.start()
        async with httpx.AsyncClient(trust_env=False) as http:
            result = await http.get(proxy.base_url + '/models', headers={'authorization': f'Bearer {TOKEN}'})
        assert result.status_code == 200
        assert result.json() == {'models': []}
    finally:
        await proxy.stop()
        await client.aclose()


class _CompletionSpans(SpanProcessor):
    def __init__(self):
        self.spans = []

    def on_end(self, span, _event):
        if span.name == 'openai_completion':
            self.spans.append(span)


@pytest.mark.asyncio
@pytest.mark.parametrize('transport', ['empty_task_context', 'loopback_http'])
async def test_native_usage_and_parent_trace_stay_with_concurrent_proxy_owner(transport):
    """Exercise the actual native observe/token_counter, not a synthetic total."""

    async def owner(label, input_tokens, output_tokens):
        async def handler(_request):
            await asyncio.sleep(0.02)
            body = completion()
            body['usage'] = {
                'prompt_tokens': input_tokens,
                'completion_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
            }
            return httpx.Response(200, json=body)

        client = client_for(handler)
        counter, recorder = TokenCounterProcessor(), _CompletionSpans()
        with processor_context([counter, recorder]), span_context('owner-' + label) as parent:
            parent_id = parent.span_id
            parent_trace = parent.span.trace_id
            proxy = OfflineResponsesProxy(client, token=TOKEN)
            try:
                if transport == 'empty_task_context':
                    # Reproduce a callback scheduled outside Nexus's run scope.
                    call = proxy.bridge.request({'messages': [{'role': 'user', 'content': label}], 'tools': TOOLS})
                    await asyncio.create_task(call, context=contextvars.Context())
                else:
                    await proxy.start()

                    async def send_http():
                        async with httpx.AsyncClient(trust_env=False) as http:
                            result = await http.post(
                                proxy.base_url + '/responses',
                                headers={'authorization': f'Bearer {TOKEN}'},
                                json=request_body(),
                            )
                        assert result.status_code == 200

                    await asyncio.create_task(send_http(), context=contextvars.Context())
            finally:
                await proxy.stop()
                await client.aclose()
        result = counter.get_result()
        assert list(result) == ['main']
        assert result['main']['prompt_tokens'] == input_tokens
        assert result['main']['completion_tokens'] == output_tokens
        assert result['main']['llm_call_count'] == 1
        assert len(recorder.spans) == 1
        assert recorder.spans[0].parent_span_id == parent_id
        assert recorder.spans[0].trace_id == parent_trace
        return recorder.spans[0].trace_id

    trace_a, trace_b = await asyncio.gather(owner('a', 4, 2), owner('b', 17, 5))
    assert trace_a != trace_b
