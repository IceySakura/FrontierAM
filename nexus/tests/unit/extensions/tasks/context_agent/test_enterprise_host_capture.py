import asyncio

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.tasks.context_agent.enterprisebench_native.host_judge_capture import (
    ResponsesCapture,
    attach_capture,
)
from nexus.extensions.tasks.context_agent.enterprisebench_native.judge_monitor import JudgeMonitor


async def test_capture_preserves_raw_transport_and_original_response():
    response = {'id': 'response-1', 'status': 'completed', 'output': []}
    headers = {'x-ms-region': 'region-1'}
    raw_response = SimpleNamespace(parse=lambda: response, http_response=SimpleNamespace(headers=headers))
    request = {'model': 'judge', 'input': [{'role': 'user', 'content': 'grade this'}]}
    transport = AsyncMock(return_value=raw_response)
    client = SimpleNamespace(
        endpoint='https://judge.example/responses',
        config=SimpleNamespace(client_args={}),
        client=SimpleNamespace(
            api_key='test-key', responses=SimpleNamespace(with_raw_response=SimpleNamespace(create=transport))
        ),
    )
    records = ResponsesCapture()
    attach_capture(client, records, JudgeMonitor())

    result = await client.client.responses.with_raw_response.create(**request)

    assert result is raw_response
    assert result.parse() is response
    assert result.http_response.headers is headers
    transport.assert_awaited_once_with(**request)
    assert [event['type'] for event in records.data['stream']] == ['request', 'response', 'request_end']
    assert records.data['stream'][1]['response']['id'] == 'response-1'


async def test_task_capture_preserves_sdk_headers_retries_and_shared_client(monkeypatch):
    import httpx

    from nexus.core import OpenAIConfig
    from nexus.extensions.openai_clients import openai_responses
    from nexus.extensions.tasks.context_agent.cache_clients import ContextAgentOpenAIResponsesClient
    from nexus.extensions.tasks.context_agent.enterprisebench_native.task import _YamlAuthoritativeJudgeClient

    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, json={'error': {'message': 'retry'}})
        return httpx.Response(
            200,
            headers={'x-ms-region': 'judge-region'},
            json={
                'id': 'response-capture',
                'object': 'response',
                'created_at': 0,
                'model': 'judge',
                'status': 'completed',
                'output': [
                    {
                        'id': 'answer',
                        'type': 'message',
                        'role': 'assistant',
                        'status': 'completed',
                        'content': [{'type': 'output_text', 'text': 'pass', 'annotations': []}],
                    }
                ],
            },
        )

    monkeypatch.setattr(openai_responses.asyncio, 'sleep', AsyncMock())
    client = ContextAgentOpenAIResponsesClient(
        OpenAIConfig(
            client_args={
                'api_key': 'capture-test-key',
                'base_url': 'https://judge.example/v1',
                'http_client': httpx.AsyncClient(transport=httpx.MockTransport(respond)),
                'max_retries': 2,
            },
            request_args={'model': 'judge'},
        )
    )
    records = ResponsesCapture()
    judge = _YamlAuthoritativeJudgeClient(client, api_records=records)
    try:
        response = await judge.create_completion(messages=[{'role': 'user', 'content': 'grade this'}])
        assert response.choices[0].message.content == 'pass'
        assert len(requests) == 2
        assert judge._client.client is not client.client
        assert judge._client.client._resource is client.client
        events = records.data['stream']
        assert [event['type'] for event in events] == [
            'request',
            'error',
            'request_end',
            'request',
            'response',
            'request_end',
        ]
        assert 'judge-region' in str(events)
        assert 'capture-test-key' not in str(events)
    finally:
        await client.client.close()


async def test_capture_records_deadline_cancellation_without_mutating_shared_sdk():
    started = asyncio.Event()

    async def create(**kwargs):
        started.set()
        await asyncio.Event().wait()

    transport = AsyncMock(side_effect=create)
    resource = SimpleNamespace(with_raw_response=SimpleNamespace(create=transport))
    sdk = SimpleNamespace(api_key='test-key', responses=resource)
    client = SimpleNamespace(
        endpoint='https://judge.example/responses',
        config=SimpleNamespace(client_args={}),
        client=sdk,
    )
    records = ResponsesCapture()
    attach_capture(client, records, JudgeMonitor())
    task = asyncio.create_task(client.client.responses.create(model='judge', input=[]))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert sdk.responses is resource
    assert sdk.responses.with_raw_response.create is transport
    assert [event['type'] for event in records.data['stream']] == ['request', 'error', 'request_end']
    assert records.data['stream'][1]['error']['type'] == 'CancelledError'
