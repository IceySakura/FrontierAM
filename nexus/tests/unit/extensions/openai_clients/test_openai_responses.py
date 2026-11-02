from __future__ import annotations

import asyncio
import hashlib
import json
import warnings

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

from openai.types.responses.response import Response
from openai.types.responses.response_computer_tool_call import ResponseComputerToolCall
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from openai.types.responses.response_usage import ResponseUsage
from pydantic import BaseModel, ConfigDict

from nexus.core import ContextLengthExceededError, OpenAIConfig, openai_client_registry
from nexus.core.types import ExtendedChatCompletion
from nexus.extensions.agents.main.agent import _prepare_messages_for_inference
from nexus.extensions.openai_clients import openai_responses as responses_module
from nexus.extensions.openai_clients.openai_responses import (
    EncryptedContentArchiveError,
    OpenAIResponsesClient,
    OpenAIResponsesRequestError,
    convert_response,
)
from nexus.trace import Span, SpanEndEvent, SpanProcessor, processor_context


class SDKLikeObject(BaseModel):
    model_config = ConfigDict(extra='allow')


class CapturingProcessor(SpanProcessor):
    def __init__(self) -> None:
        self.ended: list[Span] = []

    def on_end(self, span: Span, _event: SpanEndEvent) -> None:
        self.ended.append(span)


def _response(
    *,
    response_id: str = 'resp_123',
    status: str = 'completed',
    output: list[Any] | None = None,
    error: Any = None,
    incomplete_details: Any = None,
    usage: Any = None,
    model: str = 'gpt-cua',
) -> dict[str, Any]:
    return {
        'id': response_id,
        'object': 'response',
        'created_at': 123,
        'model': model,
        'status': status,
        'output': output or [],
        'error': error,
        'incomplete_details': incomplete_details,
        'metadata': {'request_kind': 'test'},
        'usage': usage
        or {
            'input_tokens': 10,
            'input_tokens_details': {'cached_tokens': 3, 'cache_write_tokens': 2},
            'output_tokens': 4,
            'output_tokens_details': {'reasoning_tokens': 1},
            'total_tokens': 14,
        },
    }


def _text_response(text: str = 'done') -> dict[str, Any]:
    return _response(
        output=[
            {
                'id': 'msg_1',
                'type': 'message',
                'status': 'completed',
                'role': 'assistant',
                'content': [
                    {'type': 'output_text', 'text': text, 'annotations': []},
                    {'type': 'refusal', 'refusal': 'cannot disclose'},
                ],
            }
        ]
    )


def _sdk_response(output: list[Any], *, usage: Any = None) -> Response:
    return Response.model_construct(
        id='resp_sdk',
        created_at=456,
        model='gpt-sdk',
        object='response',
        output=output,
        parallel_tool_calls=True,
        status='completed',
        tool_choice='auto',
        tools=[],
        usage=usage,
    )


def _fake_async_openai(
    create: AsyncMock,
    init_calls: list[dict[str, Any]],
    compact: AsyncMock | None = None,
) -> type:
    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            init_calls.append(kwargs)
            self.responses = SimpleNamespace(create=create, compact=compact)

    return FakeAsyncOpenAI


def _make_client(
    monkeypatch: pytest.MonkeyPatch,
    create: AsyncMock,
    *,
    client_args: dict[str, Any] | None = None,
    request_args: dict[str, Any] | None = None,
    compact: AsyncMock | None = None,
) -> tuple[OpenAIResponsesClient, list[dict[str, Any]]]:
    init_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create, init_calls, compact))
    client = OpenAIResponsesClient(
        OpenAIConfig(
            client_args={'api_key': 'test-key', **(client_args or {})},
            request_args=request_args if request_args is not None else {'model': 'gpt-cua'},
        )
    )
    return client, init_calls


def _status_error(status: int, body: Any) -> openai.APIStatusError:
    request = httpx.Request('POST', 'https://api.openai.test/v1/responses')
    response = httpx.Response(status, request=request)
    if status == 429:
        return openai.RateLimitError('rate limited', response=response, body=body)
    if status == 400:
        return openai.BadRequestError('bad request', response=response, body=body)
    return openai.APIStatusError('provider failed', response=response, body=body)


def test_openai_responses_is_registered() -> None:
    assert openai_client_registry.get('openai_responses') is OpenAIResponsesClient


def _encrypted_tool_response(response_id: str, call_id: str) -> dict[str, Any]:
    return _response(
        response_id=response_id,
        output=[
            {
                'id': 'reasoning_' + call_id,
                'type': 'reasoning',
                'summary': [],
                'encrypted_content': 'opaque_' + call_id,
            },
            {
                'id': 'tool_' + call_id,
                'type': 'function_call',
                'call_id': call_id,
                'name': 'bash',
                'arguments': '{"command":"true"}',
                'status': 'completed',
            },
        ],
    )


def _make_encrypted_client(monkeypatch: pytest.MonkeyPatch, create: AsyncMock) -> OpenAIResponsesClient:
    client, init_calls = _make_client(
        monkeypatch,
        create,
        client_args={
            'require_encrypted_reasoning': True,
            'capture_raw_io': True,
            'max_retries': 3,
        },
        request_args={
            'model': 'muse-spark-1.3',
            'store': True,
            'max_output_tokens': 131072,
            'include': ['reasoning.encrypted_content'],
            'reasoning': {'effort': 'max'},
        },
    )
    assert 'require_encrypted_reasoning' not in init_calls[0]
    return client


async def _prime_encrypted_history(client: OpenAIResponsesClient) -> list[dict[str, Any]]:
    history = [{'role': 'user', 'content': 'Original task instructions.'}]
    first = await client.create_completion(messages=history)
    history.append(first.choices[0].message.model_dump(exclude_none=True))
    history.append({'role': 'tool', 'tool_call_id': 'call_first', 'content': 'first observation'})
    return history


async def test_encrypted_history_preserves_all_native_state_and_original_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _encrypted_tool_response('resp_first', 'call_first')
    create = AsyncMock(side_effect=[first, _encrypted_tool_response('resp_second', 'call_second')])
    client = _make_encrypted_client(monkeypatch, create)
    history = await _prime_encrypted_history(client)
    await client.create_completion(messages=history)
    requests = [call.kwargs for call in create.await_args_list]
    assert requests[1]['input'] == requests[0]['input'] + first['output'] + [
        {'type': 'function_call_output', 'call_id': 'call_first', 'output': 'first observation'}
    ]
    assert {k: v for k, v in requests[0].items() if k != 'input'} == {
        k: v for k, v in requests[1].items() if k != 'input'
    }
    assert all('previous_response_id' not in request for request in requests)


@pytest.mark.parametrize(
    'damage', ['missing_ciphertext', 'pruned_prompt', 'changed_native_output', 'missing_native_output']
)
async def test_encrypted_history_rejects_loss_before_the_next_model_call(
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    create = AsyncMock(return_value=_encrypted_tool_response('resp_first', 'call_first'))
    client = _make_encrypted_client(monkeypatch, create)
    history = await _prime_encrypted_history(client)
    if damage == 'missing_ciphertext':
        history[1]['provider_specific_fields']['responses_output_items'][0].pop('encrypted_content')
    elif damage == 'pruned_prompt':
        history.pop(0)
    elif damage == 'changed_native_output':
        history[1]['provider_specific_fields']['responses_output_items'][1]['arguments'] = '{"command":"false"}'
    else:
        history[1].pop('provider_specific_fields')
    with pytest.raises(OpenAIResponsesRequestError):
        await client.create_completion(messages=history)
    assert create.await_count == 1


@pytest.mark.parametrize('ciphertext', [None, '', {'opaque': 'wrong type'}])
async def test_encrypted_history_rejects_missing_native_output_state_without_resampling(
    monkeypatch: pytest.MonkeyPatch,
    ciphertext: Any,
) -> None:
    response = _encrypted_tool_response('resp_first', 'call_first')
    response['output'][0]['encrypted_content'] = ciphertext
    create = AsyncMock(return_value=response)
    client = _make_encrypted_client(monkeypatch, create)
    with pytest.raises(OpenAIResponsesRequestError):
        await client.create_completion(messages=[{'role': 'user', 'content': 'Task.'}])
    assert create.await_count == 1
    assert client._next_encrypted_prefix is None


@pytest.mark.parametrize('reasoning_tokens', [None, 3])
async def test_encrypted_history_rejects_unaccounted_reasoning(
    monkeypatch: pytest.MonkeyPatch, reasoning_tokens: Any
) -> None:
    response = _encrypted_tool_response('resp_first', 'call_first')
    response['output'].pop(0)
    response['usage']['output_tokens_details']['reasoning_tokens'] = reasoning_tokens
    create = AsyncMock(return_value=response)
    client = _make_encrypted_client(monkeypatch, create)
    with pytest.raises(OpenAIResponsesRequestError):
        await client.create_completion(messages=[{'role': 'user', 'content': 'Task.'}])
    assert create.await_count == 1


async def test_encrypted_history_accepts_zero_reasoning_without_inventing_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _encrypted_tool_response('resp_first', 'call_first')
    response['output'].pop(0)
    response['usage']['output_tokens_details']['reasoning_tokens'] = 0
    create = AsyncMock(return_value=response)
    client = _make_encrypted_client(monkeypatch, create)
    await client.create_completion(messages=[{'role': 'user', 'content': 'Task.'}])
    assert create.await_count == 1


async def test_encrypted_history_allows_rejected_response_retry_without_appending_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _encrypted_tool_response('resp_first', 'call_first')
    rejected = _response(
        response_id='resp_rejected',
        output=[
            {'id': 'reasoning_rejected', 'type': 'reasoning', 'summary': [], 'encrypted_content': 'opaque_rejected'}
        ],
    )
    create = AsyncMock(side_effect=[first, rejected, _encrypted_tool_response('resp_second', 'call_second')])
    client = _make_encrypted_client(monkeypatch, create)
    history = await _prime_encrypted_history(client)
    await client.create_completion(messages=history)
    await client.create_completion(messages=history)
    requests = [call.kwargs for call in create.await_args_list]
    assert requests[1] == requests[2]
    assert all(item.get('id') != 'reasoning_rejected' for item in requests[2]['input'])


@pytest.mark.parametrize(
    'request_changes',
    [
        {'previous_response_id': 'resp_foreign'},
        {'conversation': 'conv_foreign'},
        {'include': []},
        {'include': 'reasoning.encrypted_content'},
        {'extra_body': {'include': []}},
    ],
)
async def test_encrypted_history_rejects_conflicting_request_controls(
    monkeypatch: pytest.MonkeyPatch,
    request_changes: dict[str, Any],
) -> None:
    create = AsyncMock()
    client = _make_encrypted_client(monkeypatch, create)
    with pytest.raises(OpenAIResponsesRequestError):
        await client.create_completion(messages=[{'role': 'user', 'content': 'Task.'}], **request_changes)
    assert create.await_count == 0


def test_encrypted_history_cannot_enable_stateful_continuation(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match='stateful_previous_response'):
        _make_client(
            monkeypatch,
            AsyncMock(),
            client_args={'stateful_previous_response': True, 'require_encrypted_reasoning': True},
        )


@pytest.mark.parametrize(
    'rejected_output',
    [
        [{'id': 'reasoning_only', 'type': 'reasoning', 'summary': []}],
        [{'id': 'bad_args', 'type': 'function_call', 'call_id': 'call_rejected', 'name': 'bash', 'arguments': ''}],
    ],
)
async def test_stateful_retry_of_unaccepted_turn_keeps_original_parent(
    monkeypatch: pytest.MonkeyPatch,
    rejected_output: list[dict[str, Any]],
) -> None:
    def tool_response(response_id: str, call_id: str) -> dict[str, Any]:
        return _response(
            response_id=response_id,
            output=[
                {
                    'id': f'item_{call_id}',
                    'type': 'function_call',
                    'call_id': call_id,
                    'name': 'bash',
                    'arguments': '{"command":"true"}',
                }
            ],
        )

    create = AsyncMock(
        side_effect=[
            tool_response('resp_first', 'call_first'),
            _response(response_id='resp_rejected', output=rejected_output),
            tool_response('resp_accepted', 'call_accepted'),
            _text_response(),
        ]
    )
    client, _ = _make_client(monkeypatch, create, client_args={'stateful_previous_response': True})
    history = [{'role': 'user', 'content': 'Use tools to solve the task.'}]
    first = await client.create_completion(messages=history)
    history.append(first.choices[0].message.model_dump(exclude_none=True))
    history.append({'role': 'tool', 'tool_call_id': 'call_first', 'content': 'done'})

    # SWE retries malformed output without appending that response to history.
    await client.create_completion(messages=history)
    accepted = await client.create_completion(messages=json.loads(json.dumps(history)))
    history.append(accepted.choices[0].message.model_dump(exclude_none=True))
    history.append({'role': 'tool', 'tool_call_id': 'call_accepted', 'content': 'done'})
    await client.create_completion(messages=history)

    requests = [call.kwargs for call in create.await_args_list]
    assert 'previous_response_id' not in requests[0]
    assert requests[1]['previous_response_id'] == 'resp_first'
    assert requests[2]['previous_response_id'] == 'resp_first'
    assert requests[2]['input'] == [{'type': 'function_call_output', 'call_id': 'call_first', 'output': 'done'}]
    assert requests[3]['previous_response_id'] == 'resp_accepted'
    assert requests[3]['input'] == [{'type': 'function_call_output', 'call_id': 'call_accepted', 'output': 'done'}]


async def test_stateful_retry_of_initial_prompt_sends_complete_prompt_again(monkeypatch: pytest.MonkeyPatch) -> None:
    create = AsyncMock(side_effect=[_response(response_id='resp_unused'), _text_response()])
    client, _ = _make_client(monkeypatch, create, client_args={'stateful_previous_response': True})
    history = [{'role': 'system', 'content': 'System instructions.'}, {'role': 'user', 'content': 'Task.'}]
    await client.create_completion(messages=history)
    await client.create_completion(messages=history)
    requests = [call.kwargs for call in create.await_args_list]
    assert all('previous_response_id' not in request for request in requests)
    assert requests[0]['input'] == requests[1]['input']
    assert len(requests[1]['input']) == 2


async def test_stateful_failed_request_does_not_advance_retry_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    create = AsyncMock(
        side_effect=[
            _response(
                response_id='resp_first',
                output=[
                    {
                        'id': 'item_first',
                        'type': 'function_call',
                        'call_id': 'call_first',
                        'name': 'bash',
                        'arguments': '{"command":"true"}',
                    }
                ],
            ),
            _status_error(400, {'error': {'message': 'invalid parameter'}}),
            _text_response(),
        ]
    )
    client, _ = _make_client(monkeypatch, create, client_args={'stateful_previous_response': True})
    history = [{'role': 'user', 'content': 'Task.'}]
    first = await client.create_completion(messages=history)
    history.append(first.choices[0].message.model_dump(exclude_none=True))
    history.append({'role': 'tool', 'tool_call_id': 'call_first', 'content': 'done'})
    with pytest.raises(OpenAIResponsesRequestError):
        await client.create_completion(messages=history)
    await client.create_completion(messages=history)
    assert [call.kwargs['previous_response_id'] for call in create.await_args_list[1:]] == ['resp_first', 'resp_first']


async def test_stateless_repeated_history_keeps_full_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    create = AsyncMock(side_effect=[_text_response(), _text_response()])
    client, _ = _make_client(monkeypatch, create)
    history = [{'role': 'user', 'content': 'Task.'}]
    await client.create_completion(messages=history)
    await client.create_completion(messages=history)
    requests = [call.kwargs for call in create.await_args_list]
    assert all('previous_response_id' not in request for request in requests)
    assert requests[0]['input'] == requests[1]['input']


def test_convert_text_refusal_usage_and_provider_state() -> None:
    response = _text_response()
    completion = convert_response(response, 'gpt-cua')

    assert isinstance(completion, ExtendedChatCompletion)
    assert completion.id == 'resp_123'
    assert completion.created == 123
    assert completion.model == 'gpt-cua'
    assert completion.choices[0].finish_reason == 'stop'
    assert completion.choices[0].message.content == 'done'
    assert completion.choices[0].message.refusal == 'cannot disclose'
    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 10
    assert completion.usage.completion_tokens == 4
    assert completion.usage.total_tokens == 14
    assert completion.usage.cache_read_input_tokens == 3
    assert completion.usage.cache_creation_input_tokens == 2
    assert completion.usage.prompt_tokens_details is not None
    assert completion.usage.prompt_tokens_details.cached_tokens == 3
    assert completion.usage.prompt_tokens_details.cache_creation_tokens == 2
    assert completion.usage.completion_tokens_details is not None
    assert completion.usage.completion_tokens_details.reasoning_tokens == 1
    assert completion.provider_specific_fields == {
        'provider': 'openai',
        'protocol': 'responses',
        'response_id': 'resp_123',
        'response_model': 'gpt-cua',
        'response_status': 'completed',
        'response_output': _text_response()['output'],
        'response_error': None,
        'response_incomplete_details': None,
        'response_state': {key: value for key, value in response.items() if key != 'output'},
    }


def test_convert_function_call_to_standard_tool_call() -> None:
    completion = convert_response(
        _response(
            output=[
                {
                    'id': 'fc_1',
                    'type': 'function_call',
                    'call_id': 'call_weather',
                    'name': 'get_weather',
                    'arguments': '{"city":"Paris"}',
                    'status': 'completed',
                }
            ]
        ),
        'gpt-cua',
    )

    call = completion.choices[0].message.tool_calls[0]
    assert call.id == 'call_weather'
    assert call.type == 'function'
    assert call.function.name == 'get_weather'
    assert call.function.arguments == '{"city":"Paris"}'
    assert completion.choices[0].finish_reason == 'tool_calls'


@pytest.mark.parametrize(
    'item',
    [
        {
            'id': 'patch_1',
            'type': 'apply_patch_call',
            'call_id': 'call_patch_1',
            'operation': {'type': 'delete_file', 'path': 'example.txt'},
        },
        {
            'id': 'custom_1',
            'type': 'custom_tool_call',
            'call_id': 'call_custom_1',
            'name': 'custom_tool',
            'input': 'payload',
        },
        {
            'id': 'local_shell_1',
            'type': 'local_shell_call',
            'call_id': 'call_local_shell_1',
            'action': {'type': 'exec', 'command': ['pwd']},
        },
        {
            'id': 'mcp_1',
            'type': 'mcp_approval_request',
            'name': 'remote_tool',
            'server_label': 'server',
            'arguments': '{}',
        },
        {
            'id': 'shell_1',
            'type': 'shell_call',
            'call_id': 'call_shell_1',
            'action': {'commands': ['pwd']},
        },
        {'id': 'web_search_1', 'type': 'web_search_call'},
        {'id': 'file_search_1', 'type': 'file_search_call'},
        {'id': 'code_interpreter_1', 'type': 'code_interpreter_call'},
        {'id': 'image_generation_1', 'type': 'image_generation_call'},
        {'id': 'mcp_call_1', 'type': 'mcp_call'},
        {'id': 'mcp_list_tools_1', 'type': 'mcp_list_tools'},
        {'id': 'future_1', 'type': 'future_output_item'},
    ],
)
def test_convert_rejects_response_tool_items_outside_computer_use_scope(item: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match='Unsupported OpenAI Responses output item type') as exc_info:
        convert_response(_response(output=[item]), 'gpt-cua')

    assert str(exc_info.value) == f'Unsupported OpenAI Responses output item type {item["type"]!r}'


@pytest.mark.parametrize('item', [None, 7, 'invalid', [], {}, {'type': None}, {'type': 7}])
def test_convert_rejects_response_items_without_a_valid_type(item: Any) -> None:
    with pytest.raises(ValueError, match='OpenAI Responses output item requires a supported type'):
        convert_response(_response(output=[item]), 'gpt-cua')


@pytest.mark.parametrize(
    'item_type',
    [
        'apply_patch_call',
        'custom_tool_call_output',
        'local_shell_call',
        'mcp_approval_response',
        'shell_call_output',
        'web_search_call',
        'file_search_call',
        'code_interpreter_call',
        'image_generation_call',
        'mcp_call',
        'mcp_list_tools',
        'future_input_item',
    ],
)
def test_responses_input_rejects_items_outside_computer_use_scope(item_type: str) -> None:
    with pytest.raises(ValueError, match='Unsupported OpenAI Responses input item type') as exc_info:
        responses_module._responses_input([{'type': item_type}])

    assert str(exc_info.value) == f'Unsupported OpenAI Responses input item type {item_type!r}'


@pytest.mark.parametrize('item_type', [None, 7])
def test_responses_input_rejects_typed_items_without_a_valid_type(item_type: Any) -> None:
    with pytest.raises(ValueError, match='OpenAI Responses input item requires a supported type'):
        responses_module._responses_input([{'type': item_type}])


def test_responses_input_preserves_native_compaction_item() -> None:
    item = {'type': 'compaction', 'id': 'cmp_1', 'encrypted_content': 'opaque-state'}

    assert responses_module._responses_input([item]) == [item]


def test_responses_input_preserves_native_agent_message() -> None:
    item = {
        'type': 'agent_message',
        'id': 'amsg_child_task',
        'author': '/root',
        'recipient': '/root/child',
        'content': [
            {'type': 'input_text', 'text': 'Message Type: NEW_TASK\nPayload:\n'},
            {'type': 'encrypted_content', 'encrypted_content': 'opaque-child-task'},
        ],
    }

    assert responses_module._responses_input([item]) == [item]


def test_responses_input_drops_only_computer_round_pruned_by_image_history_limit() -> None:
    messages: list[dict[str, Any]] = [{'role': 'system', 'content': 'Use the computer tool.'}]
    for round_index in range(1, 8):
        call_id = f'call_{round_index}'
        messages.extend(
            [
                {
                    'role': 'assistant',
                    'content': None,
                    'provider_specific_fields': {
                        'responses_output_items': [
                            {
                                'type': 'computer_call',
                                'id': f'computer_{round_index}',
                                'call_id': call_id,
                                'actions': [{'type': 'click', 'x': round_index, 'y': round_index}],
                                'pending_safety_checks': [],
                            }
                        ]
                    },
                },
                {'role': 'tool', 'tool_call_id': call_id, 'content': f'executed round {round_index}'},
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'Current desktop screenshot.'},
                        {
                            'type': 'image_url',
                            'image_url': {'url': f'data:image/png;base64,shot-{round_index}'},
                        },
                    ],
                },
            ]
        )

    prepared = _prepare_messages_for_inference(messages, image_history_limit=6)
    converted = responses_module._responses_input(prepared)

    computer_calls = [item for item in converted if item.get('type') == 'computer_call']
    computer_outputs = [item for item in converted if item.get('type') == 'computer_call_output']
    assert [item['call_id'] for item in computer_calls] == [f'call_{index}' for index in range(2, 8)]
    assert [item['call_id'] for item in computer_outputs] == [f'call_{index}' for index in range(2, 8)]
    assert [item['output']['image_url'] for item in computer_outputs] == [
        f'data:image/png;base64,shot-{index}' for index in range(2, 8)
    ]
    assert 'previous image omitted' not in json.dumps(converted)


def test_responses_input_still_rejects_unmarked_missing_computer_screenshot() -> None:
    computer_call = {
        'type': 'computer_call',
        'id': 'computer_1',
        'call_id': 'call_1',
        'actions': [{'type': 'click', 'x': 1, 'y': 1}],
        'pending_safety_checks': [],
    }
    messages = [
        {
            'role': 'assistant',
            'content': None,
            'provider_specific_fields': {'responses_output_items': [computer_call]},
        },
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'executed'},
        {'role': 'user', 'content': 'No screenshot is available.'},
    ]

    with pytest.raises(ValueError, match="computer_call 'call_1' requires a screenshot result"):
        responses_module._responses_input(messages)


@pytest.mark.parametrize('item_type', ['function_call', 'computer_call'])
def test_convert_rejects_tool_items_without_provider_call_id(item_type: str) -> None:
    item: dict[str, Any] = {'id': 'provider_item_id', 'type': item_type}
    if item_type == 'function_call':
        item.update({'name': 'tool', 'arguments': '{}'})
    else:
        item.update({'actions': [], 'pending_safety_checks': []})

    with pytest.raises(ValueError, match='call_id'):
        convert_response(_response(output=[item]), 'gpt-cua')


@pytest.mark.parametrize(
    ('action_field', 'action_value', 'expected_actions'),
    [
        ('action', {'type': 'click', 'x': 10, 'y': 20}, [{'type': 'click', 'x': 10, 'y': 20}]),
        (
            'actions',
            [{'type': 'click', 'x': 10, 'y': 20}, {'type': 'type', 'text': 'hello'}],
            [{'type': 'click', 'x': 10, 'y': 20}, {'type': 'type', 'text': 'hello'}],
        ),
    ],
)
def test_convert_dict_computer_call_normalizes_singular_and_plural_actions(
    action_field: str,
    action_value: Any,
    expected_actions: list[dict[str, Any]],
) -> None:
    item = {
        'id': 'computer_1',
        'type': 'computer_call',
        'call_id': 'call_computer',
        action_field: action_value,
        'pending_safety_checks': [{'id': 'safe_1', 'code': 'confirm'}],
        'status': 'completed',
    }
    completion = convert_response(_response(output=[item]), 'gpt-cua')

    call = completion.choices[0].message.tool_calls[0]
    arguments = json.loads(call.function.arguments)
    assert call.id == 'call_computer'
    assert call.function.name == 'computer'
    assert set(arguments) == {'actions', 'pending_safety_checks', 'raw_item'}
    assert arguments['actions'] == expected_actions
    assert arguments['pending_safety_checks'] == [{'id': 'safe_1', 'code': 'confirm'}]
    assert arguments['raw_item'] == item
    assert completion.choices[0].finish_reason == 'tool_calls'


def test_convert_sdk_like_computer_call_without_binding_to_generated_class() -> None:
    item = SDKLikeObject.model_validate(
        {
            'id': 'computer_sdk',
            'type': 'computer_call',
            'call_id': 'call_sdk',
            'action': {'type': 'screenshot'},
            'pending_safety_checks': [],
            'status': 'completed',
        }
    )
    response = SDKLikeObject.model_validate(_response(output=[item]))

    completion = convert_response(response, 'gpt-cua')

    call = completion.choices[0].message.tool_calls[0]
    assert json.loads(call.function.arguments)['actions'] == [{'type': 'screenshot'}]


def test_convert_real_sdk_singular_computer_call_and_json_serializes_complete_state() -> None:
    item = ResponseComputerToolCall.model_validate(
        {
            'id': 'computer_sdk',
            'type': 'computer_call',
            'call_id': 'call_sdk',
            'action': {
                'type': 'drag',
                'path': [
                    {'x': 1, 'y': 2},
                    {'x': 3, 'y': 4},
                ],
            },
            'pending_safety_checks': [{'id': 'safe_1', 'code': 'confirm', 'message': 'Confirm this action.'}],
            'status': 'completed',
        }
    )
    usage = ResponseUsage.model_validate(
        {
            'input_tokens': 8,
            'input_tokens_details': {'cached_tokens': 2, 'cache_write_tokens': 0},
            'output_tokens': 5,
            'output_tokens_details': {'reasoning_tokens': 3},
            'total_tokens': 13,
        }
    )

    completion = convert_response(_sdk_response([item], usage=usage), 'ignored-model')

    arguments = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)
    assert arguments['actions'] == [
        {
            'type': 'drag',
            'path': [
                {'x': 1, 'y': 2},
                {'x': 3, 'y': 4},
            ],
        }
    ]
    assert arguments['pending_safety_checks'] == [
        {'id': 'safe_1', 'code': 'confirm', 'message': 'Confirm this action.'}
    ]
    assert completion.id == 'resp_sdk'
    assert completion.model == 'gpt-sdk'
    assert completion.usage.cache_read_input_tokens == 2
    assert completion.usage.completion_tokens_details.reasoning_tokens == 3
    json.dumps(completion.provider_specific_fields)


def test_convert_loose_sdk_ga_plural_actions_does_not_fabricate_singular_action() -> None:
    item = ResponseComputerToolCall.model_construct(
        id='computer_ga',
        type='computer_call',
        call_id='call_ga',
        actions=[
            {'type': 'click', 'x': 10, 'y': 20},
            {'type': 'type', 'text': 'ordered'},
        ],
        pending_safety_checks=[],
        status='completed',
    )

    completion = convert_response(_sdk_response([item]), 'gpt-sdk')

    arguments = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)
    assert arguments['actions'] == [
        {'type': 'click', 'x': 10, 'y': 20},
        {'type': 'type', 'text': 'ordered'},
    ]
    assert 'action' not in arguments['raw_item']


def test_convert_reasoning_summary_and_complete_encrypted_items() -> None:
    reasoning_item = {
        'id': 'rs_1',
        'type': 'reasoning',
        'summary': [
            {'type': 'summary_text', 'text': 'Inspect the current window.'},
            {'type': 'summary_text', 'text': 'Then click the button.'},
        ],
        'encrypted_content': 'opaque-encrypted-reasoning',
        'status': 'completed',
    }
    completion = convert_response(
        _response(
            output=[
                reasoning_item,
                {
                    'type': 'message',
                    'content': [{'type': 'output_text', 'text': 'working'}],
                    'role': 'assistant',
                    'status': 'completed',
                },
            ]
        ),
        'gpt-cua',
    )

    message = completion.choices[0].message
    assert message.reasoning_content == 'Inspect the current window.\nThen click the button.'
    assert message.reasoning_items == [reasoning_item]
    assert message.content == 'working'


def test_convert_ignores_top_level_reasoning_summary_mode_string() -> None:
    reasoning_item = {
        'id': 'rs_1',
        'type': 'reasoning',
        'summary': [{'type': 'summary_text', 'text': 'Inspect the repo.'}],
        'encrypted_content': 'ENC',
        'status': 'completed',
    }
    payload = _response(
        output=[
            reasoning_item,
            {
                'type': 'message',
                'content': [{'type': 'output_text', 'text': 'ok'}],
                'role': 'assistant',
                'status': 'completed',
            },
        ],
        model='gpt-6-astra',
    )
    payload['reasoning'] = {'effort': 'xhigh', 'summary': 'auto'}
    completion = convert_response(payload, 'gpt-6-astra')
    message = completion.choices[0].message
    assert message.reasoning_content == 'Inspect the repo.'
    assert 'auto' not in (message.reasoning_content or '')


def test_convert_ignores_top_level_reasoning_summary_detailed_mode() -> None:
    """``summary: detailed`` is a request enum; never persist it as thinking text."""
    payload = _response(
        output=[
            {
                'type': 'message',
                'content': [{'type': 'output_text', 'text': 'ok'}],
                'role': 'assistant',
                'status': 'completed',
            },
        ],
        model='gpt-6-astra',
    )
    payload['reasoning'] = {'effort': 'xhigh', 'summary': 'detailed'}
    payload['output'] = [
        {
            'id': 'rs_1',
            'type': 'reasoning',
            'summary': 'detailed',
            'encrypted_content': 'ENC',
            'status': 'completed',
        },
        *payload['output'],
    ]
    completion = convert_response(payload, 'gpt-6-astra')
    message = completion.choices[0].message
    assert message.reasoning_content is None
    assert message.reasoning_items is not None


def test_convert_does_not_mint_thinking_from_request_echo_only() -> None:
    """First turns often have no output reasoning items; the request echo must
    not become reasoning_content or the proxy will mint a fake rs_* and 404."""
    payload = _response(
        output=[
            {
                'type': 'message',
                'content': [{'type': 'output_text', 'text': 'ok'}],
                'role': 'assistant',
                'status': 'completed',
            },
        ],
        model='gpt-6-astra',
    )
    payload['reasoning'] = {'effort': 'xhigh', 'summary': 'detailed'}
    completion = convert_response(payload, 'gpt-6-astra')
    message = completion.choices[0].message
    assert message.reasoning_content is None
    assert not message.reasoning_items


def test_convert_real_sdk_reasoning_and_multiple_outputs_preserves_order() -> None:
    first_reasoning = ResponseReasoningItem.model_validate(
        {
            'id': 'rs_sdk_1',
            'type': 'reasoning',
            'summary': [{'type': 'summary_text', 'text': 'First reason.'}],
            'content': [{'type': 'reasoning_text', 'text': 'Private detail.'}],
            'encrypted_content': 'encrypted-one',
            'status': 'completed',
        }
    )
    second_reasoning = ResponseReasoningItem.model_validate(
        {
            'id': 'rs_sdk_2',
            'type': 'reasoning',
            'summary': [{'type': 'summary_text', 'text': 'Second reason.'}],
            'encrypted_content': 'encrypted-two',
            'status': 'completed',
        }
    )
    output = [
        first_reasoning,
        {
            'id': 'msg_1',
            'type': 'message',
            'content': [
                {'type': 'output_text', 'text': 'part one'},
                {'type': 'output_text', 'text': 'part two'},
            ],
        },
        {
            'id': 'fc_1',
            'type': 'function_call',
            'call_id': 'call_1',
            'name': 'first',
            'arguments': '{ "kept" : true }',
        },
        second_reasoning,
        {
            'id': 'msg_2',
            'type': 'message',
            'content': [{'type': 'output_text', 'text': 'part three'}],
        },
        {
            'id': 'computer_1',
            'type': 'computer_call',
            'call_id': 'call_2',
            'actions': [],
        },
        {
            'id': 'fc_2',
            'type': 'function_call',
            'call_id': 'call_3',
            'name': 'last',
            'arguments': '{}',
        },
    ]

    with warnings.catch_warnings(record=True) as serialization_warnings:
        warnings.simplefilter('always')
        completion = convert_response(_sdk_response(output), 'gpt-sdk')

    message = completion.choices[0].message
    assert serialization_warnings == []
    assert message.content == 'part one\npart two\npart three'
    assert message.reasoning_content == 'First reason.\nSecond reason.'
    assert [item['id'] for item in message.reasoning_items] == ['rs_sdk_1', 'rs_sdk_2']
    assert message.reasoning_items[0]['content'] == [{'type': 'reasoning_text', 'text': 'Private detail.'}]
    assert [call.id for call in message.tool_calls] == ['call_1', 'call_2', 'call_3']
    assert message.tool_calls[0].function.arguments == '{ "kept" : true }'
    assert completion.choices[0].finish_reason == 'tool_calls'
    assert [item['id'] for item in completion.provider_specific_fields['response_output']] == [
        'rs_sdk_1',
        'msg_1',
        'fc_1',
        'rs_sdk_2',
        'msg_2',
        'computer_1',
        'fc_2',
    ]


def test_convert_incomplete_response_uses_length_finish_reason() -> None:
    details = {'reason': 'max_output_tokens'}
    completion = convert_response(
        _response(status='incomplete', incomplete_details=details, output=[]),
        'gpt-cua',
    )

    assert completion.choices[0].finish_reason == 'length'
    assert completion.provider_specific_fields['response_incomplete_details'] == details


@pytest.mark.parametrize('reason', ['max_output_tokens', 'content_filter'])
def test_convert_incomplete_response_preserves_each_official_reason(reason: str) -> None:
    completion = convert_response(
        _response(status='incomplete', incomplete_details={'reason': reason}, output=[]),
        'gpt-cua',
    )

    expected = 'length' if reason == 'max_output_tokens' else 'content_filter'
    assert completion.choices[0].finish_reason == expected
    assert completion.provider_specific_fields['response_incomplete_details'] == {'reason': reason}


@pytest.mark.parametrize(
    ('reason', 'expected_finish_reason'),
    [
        ('future_provider_reason', 'future_provider_reason'),
        (None, 'incomplete'),
        ('', 'incomplete'),
    ],
)
def test_incomplete_finish_reason_precedes_valid_tool_calls(
    reason: str | None,
    expected_finish_reason: str,
) -> None:
    details = {'reason': reason} if reason is not None else None
    completion = convert_response(
        _response(
            status='incomplete',
            incomplete_details=details,
            output=[
                {
                    'id': 'computer_partial',
                    'type': 'computer_call',
                    'call_id': 'call_partial',
                    'actions': [{'type': 'click', 'x': 1, 'y': 2}],
                    'pending_safety_checks': [],
                }
            ],
        ),
        'gpt-cua',
    )

    assert completion.choices[0].finish_reason == expected_finish_reason
    assert completion.choices[0].message.tool_calls[0].id == 'call_partial'


def test_incomplete_response_preserves_but_does_not_convert_malformed_partial_tool_call() -> None:
    partial_item = {
        'id': 'computer_partial',
        'type': 'computer_call',
        'actions': [{'type': 'click'}],
        'pending_safety_checks': [{'id': 'pending'}],
    }
    completion = convert_response(
        _response(
            status='incomplete',
            incomplete_details={'reason': 'max_output_tokens'},
            output=[partial_item],
        ),
        'gpt-cua',
    )

    assert completion.choices[0].finish_reason == 'length'
    assert completion.choices[0].message.tool_calls is None
    assert completion.provider_specific_fields['response_output'] == [partial_item]


def test_convert_missing_usage_and_zero_usage() -> None:
    no_usage = _response()
    no_usage['usage'] = None
    assert convert_response(no_usage, 'gpt-cua').usage is None

    zero_usage = _response(
        usage={
            'input_tokens': 0,
            'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0},
            'output_tokens': 0,
            'output_tokens_details': {'reasoning_tokens': 0},
            'total_tokens': 0,
        }
    )
    usage = convert_response(zero_usage, 'gpt-cua').usage
    assert usage is not None
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0
    assert usage.cache_read_input_tokens == 0
    assert usage.cache_creation_input_tokens == 0


@pytest.mark.asyncio
async def test_client_passes_response_items_without_chat_conversion_and_honors_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = AsyncMock(return_value=_text_response())
    client, init_calls = _make_client(
        monkeypatch,
        create,
        client_args={
            'base_url': 'https://openai.test/v1',
            'capture_raw_io': True,
            'max_retries': 7,
        },
        request_args={
            'model': 'config-model',
            'input': 'must-not-win',
            'instructions': 'config instructions',
            'previous_response_id': 'resp_config',
            'tools': [{'type': 'function', 'name': 'config_tool'}],
            'reasoning': {'effort': 'medium'},
        },
    )
    client.config.enforced_request_args['response_format'] = {'type': 'json_object'}
    item = {
        'type': 'computer_call_output',
        'call_id': 'call_1',
        'output': {'type': 'computer_screenshot', 'image_url': 'data:image/png;base64,AAAA'},
    }
    native_message = {
        'type': 'message',
        'role': 'user',
        'content': [{'type': 'input_text', 'text': 'Continue from the screenshot.'}],
    }
    input_items = [item, native_message]

    completion = await client._create_completion(
        input_items,
        model='call-model',
        instructions='call instructions',
        previous_response_id='resp_call',
        tools=[{'type': 'computer'}],
        reasoning={'effort': 'high', 'summary': 'concise'},
    )

    assert isinstance(completion, ExtendedChatCompletion)
    create.assert_awaited_once()
    request = create.await_args.kwargs
    assert request['input'] is input_items
    assert request['model'] == 'call-model'
    assert request['instructions'] == 'call instructions'
    assert request['previous_response_id'] == 'resp_call'
    assert request['tools'] == [{'type': 'computer'}]
    assert request['reasoning'] == {'effort': 'high', 'summary': 'concise'}
    assert 'response_format' not in request
    assert request['text'] == {'format': {'type': 'json_object'}}
    assert init_calls == [
        {
            'api_key': 'test-key',
            'base_url': 'https://openai.test/v1',
            'max_retries': 0,
        }
    ]


@pytest.mark.asyncio
async def test_client_forwards_native_compaction_request(monkeypatch: pytest.MonkeyPatch) -> None:
    compact_response = {
        'id': 'resp_compact',
        'output': [
            {
                'type': 'compaction',
                'id': 'cmp_1',
                'encrypted_content': 'opaque-compaction-state',
            }
        ],
        'usage': {'input_tokens': 12, 'output_tokens': 3, 'total_tokens': 15},
    }
    compact = AsyncMock(return_value=compact_response)
    client, _ = _make_client(
        monkeypatch,
        AsyncMock(return_value=_text_response()),
        request_args={'model': 'configured-model'},
        compact=compact,
    )
    client.config.enforced_request_args['model'] = 'enforced-model'
    records: list[dict[str, Any]] = []
    client._record_provider_call = records.append

    result = await client.compact_response(
        {
            'input': [{'type': 'message', 'role': 'user', 'content': 'history'}],
            'model': 'request-model',
        }
    )

    assert result == compact_response
    compact.assert_awaited_once_with(
        input=[{'type': 'message', 'role': 'user', 'content': 'history'}],
        model='enforced-model',
    )
    assert records[0]['endpoint'].endswith('/responses/compact')
    assert records[0]['attempts'] == [{'attempt': 1, 'status': 'success'}]


@pytest.mark.asyncio
async def test_client_translates_chat_max_tokens_to_responses_output_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = AsyncMock(return_value=_text_response())
    client, _ = _make_client(
        monkeypatch,
        create,
        request_args={'model': 'gpt-cua', 'max_output_tokens': 2048},
    )

    await client._create_completion(
        [{'role': 'user', 'content': 'Solve the problem.'}],
        max_tokens=1024,
    )

    request = create.await_args.kwargs
    assert 'max_tokens' not in request
    assert request['max_output_tokens'] == 2048


@pytest.mark.asyncio
async def test_client_translates_max_completion_tokens_to_responses_output_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = AsyncMock(return_value=_text_response())
    client, _ = _make_client(
        monkeypatch,
        create,
        request_args={'model': 'hermes-gpt'},
    )

    await client._create_completion(
        [{'role': 'user', 'content': 'Solve the problem.'}],
        max_completion_tokens=4096,
    )

    request = create.await_args.kwargs
    assert 'max_completion_tokens' not in request
    assert request['max_output_tokens'] == 4096


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('configured_reasoning', 'reasoning_effort', 'expected_reasoning'),
    [
        (None, 'high', {'effort': 'high'}),
        (
            {'effort': 'medium', 'summary': 'auto', 'context': 'all_turns'},
            'high',
            {'effort': 'high', 'summary': 'auto', 'context': 'all_turns'},
        ),
        ({'effort': 'medium', 'summary': 'auto'}, None, {'effort': 'medium', 'summary': 'auto'}),
        (None, None, None),
    ],
)
async def test_client_translates_chat_reasoning_effort_to_responses_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    configured_reasoning: dict[str, Any] | None,
    reasoning_effort: str | None,
    expected_reasoning: dict[str, Any] | None,
) -> None:
    create = AsyncMock(return_value=_text_response())
    request_args: dict[str, Any] = {'model': 'gpt-cua'}
    if configured_reasoning is not None:
        request_args['reasoning'] = configured_reasoning
    client, _ = _make_client(monkeypatch, create, request_args=request_args)

    await client._create_completion(
        [{'role': 'user', 'content': 'Solve the problem.'}],
        reasoning_effort=reasoning_effort,
    )

    request = create.await_args.kwargs
    assert 'reasoning_effort' not in request
    if expected_reasoning is None:
        assert 'reasoning' not in request
    else:
        assert request['reasoning'] == expected_reasoning


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('configured_text', 'verbosity', 'expected_text'),
    [
        (None, 'low', {'verbosity': 'low'}),
        (
            {'verbosity': 'medium', 'format': {'type': 'text'}},
            'low',
            {'verbosity': 'low', 'format': {'type': 'text'}},
        ),
        ({'verbosity': 'medium'}, None, {'verbosity': 'medium'}),
        (None, None, None),
    ],
)
async def test_client_translates_chat_verbosity_to_responses_text(
    monkeypatch: pytest.MonkeyPatch,
    configured_text: dict[str, Any] | None,
    verbosity: str | None,
    expected_text: dict[str, Any] | None,
) -> None:
    create = AsyncMock(return_value=_text_response())
    request_args: dict[str, Any] = {'model': 'gpt-cua'}
    if configured_text is not None:
        request_args['text'] = configured_text
    client, _ = _make_client(monkeypatch, create, request_args=request_args)

    await client._create_completion(
        [{'role': 'user', 'content': 'Answer concisely.'}],
        verbosity=verbosity,
    )

    request = create.await_args.kwargs
    assert 'verbosity' not in request
    if expected_text is None:
        assert 'text' not in request
    else:
        assert request['text'] == expected_text


@pytest.mark.parametrize(
    ('response_format', 'expected_format'),
    [
        (
            {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'answer',
                    'description': 'Structured answer.',
                    'schema': {'type': 'object'},
                    'strict': True,
                },
            },
            {
                'type': 'json_schema',
                'name': 'answer',
                'description': 'Structured answer.',
                'schema': {'type': 'object'},
                'strict': True,
            },
        ),
        ({'type': 'json_object'}, {'type': 'json_object'}),
        ({'type': 'text'}, {'type': 'text'}),
    ],
)
def test_normalize_chat_response_format_to_responses_text_format(
    response_format: dict[str, Any],
    expected_format: dict[str, Any],
) -> None:
    request = {
        'response_format': response_format,
        'text': {'verbosity': 'low', 'format': {'type': 'text'}},
    }

    responses_module._normalize_chat_request_controls(request)

    assert 'response_format' not in request
    assert request['text'] == {'verbosity': 'low', 'format': expected_format}


@pytest.mark.parametrize(
    ('response_format', 'error_type'),
    [
        ('json', TypeError),
        ({'type': 'json_schema'}, TypeError),
        ({'type': 'unsupported'}, ValueError),
    ],
)
def test_normalize_chat_response_format_rejects_invalid_values(
    response_format: Any,
    error_type: type[Exception],
) -> None:
    request = {'response_format': response_format}

    with pytest.raises(error_type):
        responses_module._normalize_chat_request_controls(request)


@pytest.mark.asyncio
async def test_client_flattens_main_agent_function_tool_schema_for_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = AsyncMock(return_value=_text_response())
    client, _ = _make_client(monkeypatch, create)
    main_agent_tools = [
        {
            'type': 'function',
            'function': {
                'name': 'protocol_step',
                'description': 'Advance the protocol by one step.',
                'parameters': {
                    'type': 'object',
                    'properties': {'step': {'type': 'integer'}},
                    'required': ['step'],
                },
                'strict': True,
            },
        }
    ]

    await client._create_completion(
        [{'role': 'user', 'content': 'Run the protocol.'}],
        tools=main_agent_tools,
    )

    assert create.await_args.kwargs['tools'] == [
        {
            'type': 'function',
            'name': 'protocol_step',
            'description': 'Advance the protocol by one step.',
            'parameters': {
                'type': 'object',
                'properties': {'step': {'type': 'integer'}},
                'required': ['step'],
            },
            'strict': True,
        }
    ]


@pytest.mark.asyncio
async def test_client_defaults_missing_function_tool_strict_to_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AIHub upgrades omitted strict to true; default false preserves omit-optional semantics."""
    create = AsyncMock(return_value=_text_response())
    client, _ = _make_client(monkeypatch, create)

    await client._create_completion(
        [{'role': 'user', 'content': 'Look something up.'}],
        tools=[
            {
                'type': 'function',
                'function': {
                    'name': 'collaboration__spawn_agent',
                    'description': 'Spawn an agent.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'task_name': {'type': 'string'},
                            'message': {'type': 'string'},
                            'model': {'type': 'string'},
                            'reasoning_effort': {'type': 'string'},
                        },
                        'required': ['task_name', 'message'],
                        'additionalProperties': False,
                    },
                },
            }
        ],
    )

    tool = create.await_args.kwargs['tools'][0]
    assert tool['name'] == 'collaboration__spawn_agent'
    assert tool['strict'] is False
    assert tool['parameters']['required'] == ['task_name', 'message']


@pytest.mark.asyncio
async def test_client_preserves_explicit_function_tool_strict_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = AsyncMock(return_value=_text_response())
    client, _ = _make_client(monkeypatch, create)

    await client._create_completion(
        [{'role': 'user', 'content': 'Call demo.'}],
        tools=[
            {
                'type': 'function',
                'name': 'demo_tool',
                'strict': False,
                'parameters': {
                    'type': 'object',
                    'properties': {'name': {'type': 'string'}},
                    'required': ['name'],
                },
            }
        ],
    )

    assert create.await_args.kwargs['tools'][0]['strict'] is False


@pytest.mark.asyncio
async def test_client_omits_none_tools_from_main_agent_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = AsyncMock(return_value=_text_response())
    client, _ = _make_client(monkeypatch, create)

    completion = await client._create_completion(
        [{'role': 'user', 'content': 'Answer without tools.'}],
        tools=None,
    )

    assert completion.choices[0].message.content == 'done'
    assert 'tools' not in create.await_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize('tool_type', ['apply_patch', 'custom', 'local_shell', 'mcp', 'shell'])
async def test_client_rejects_native_tools_outside_computer_use_scope(
    monkeypatch: pytest.MonkeyPatch,
    tool_type: str,
) -> None:
    create = AsyncMock(return_value=_text_response())
    client, _ = _make_client(monkeypatch, create)

    with pytest.raises(OpenAIResponsesRequestError) as exc_info:
        await client._create_completion(
            [{'role': 'user', 'content': 'Run the tool.'}],
            tools=[{'type': tool_type}],
        )

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__) == f'Unsupported OpenAI Responses native tool type {tool_type!r}'
    create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('container_kind', ['tuple', 'generator'])
async def test_client_rejects_unsupported_native_tool_iterables_before_network(
    monkeypatch: pytest.MonkeyPatch,
    container_kind: str,
) -> None:
    create = AsyncMock(return_value=_text_response())
    client, _ = _make_client(monkeypatch, create)
    tool = {'type': 'shell'}
    tools = (tool,) if container_kind == 'tuple' else (item for item in [tool])

    with pytest.raises(OpenAIResponsesRequestError) as exc_info:
        await client._create_completion([{'role': 'user', 'content': 'Run the tool.'}], tools=tools)

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__) == "Unsupported OpenAI Responses native tool type 'shell'"
    create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('container_kind', ['tuple', 'generator'])
async def test_client_materializes_supported_native_tool_iterables(
    monkeypatch: pytest.MonkeyPatch,
    container_kind: str,
) -> None:
    create = AsyncMock(return_value=_text_response())
    client, _ = _make_client(monkeypatch, create)
    tool = {'type': 'computer'}
    tools = (tool,) if container_kind == 'tuple' else (item for item in [tool])

    await client._create_completion([{'role': 'user', 'content': 'Use the computer.'}], tools=tools)

    assert create.await_args.kwargs['tools'] == [{'type': 'computer'}]


@pytest.mark.asyncio
async def test_client_converts_main_agent_tool_history_to_responses_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = AsyncMock(return_value=_text_response())
    client, _ = _make_client(monkeypatch, create)
    messages = [
        {'role': 'system', 'content': 'Use tools when needed.'},
        {'role': 'user', 'content': 'Inspect the repository.'},
        {
            'role': 'assistant',
            'content': 'I will inspect it.',
            'tool_calls': [
                {
                    'id': 'call_1',
                    'type': 'function',
                    'function': {'name': 'shell', 'arguments': '{"cmd":"pwd"}'},
                }
            ],
        },
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': '/workspace/nexus'},
    ]

    completion = await client._create_completion(messages)

    assert isinstance(completion, ExtendedChatCompletion)
    assert create.await_args.kwargs['input'] == [
        {'role': 'system', 'content': 'Use tools when needed.'},
        {'role': 'user', 'content': 'Inspect the repository.'},
        {'role': 'assistant', 'content': 'I will inspect it.'},
        {
            'type': 'function_call',
            'call_id': 'call_1',
            'name': 'shell',
            'arguments': '{"cmd":"pwd"}',
        },
        {
            'type': 'function_call_output',
            'call_id': 'call_1',
            'output': '/workspace/nexus',
        },
    ]


@pytest.mark.parametrize('history_format', ['chat', 'replayed', 'native', 'sdk'])
@pytest.mark.parametrize('stateful', [False, True])
async def test_client_aligns_out_of_order_parallel_tool_outputs(
    monkeypatch: pytest.MonkeyPatch,
    history_format: str,
    stateful: bool,
) -> None:
    calls = [
        {'type': 'function_call', 'call_id': call_id, 'name': 'Read', 'arguments': json.dumps({'file_path': path})}
        for call_id, path in [('call_env', '/pkg/environment/Dockerfile'), ('call_tests', '/pkg/tests/Dockerfile')]
    ]
    outputs = [
        {'type': 'function_call_output', 'call_id': 'call_tests', 'output': 'WORKDIR /tests'},
        {'type': 'function_call_output', 'call_id': 'call_env', 'output': 'WORKDIR /app'},
    ]
    if history_format == 'native':
        messages = calls + outputs
    else:
        assistant = convert_response(_response(output=calls)).choices[0].message.model_dump(exclude_none=True)
        if history_format == 'chat':
            assistant.pop('provider_specific_fields')
        messages = [
            {'role': 'user', 'content': 'Read both Dockerfiles.'},
            assistant,
            *({'role': 'tool', 'tool_call_id': item['call_id'], 'content': item['output']} for item in outputs),
        ]
        if history_format == 'sdk':
            messages = [SDKLikeObject(**item) for item in messages]
    before = responses_module._json_value(messages)
    create = AsyncMock(return_value=_text_response())
    client, _ = _make_client(monkeypatch, create, client_args={'stateful_previous_response': stateful})
    if stateful:
        client._last_response_id = 'resp_parent'

    await client.create_completion(messages)

    request = create.await_args.kwargs
    sent_items = responses_module._json_value(request['input'])
    assert [item for item in sent_items if item.get('type') == 'function_call_output'] == outputs[::-1]
    assert responses_module._json_value(messages) == before
    if stateful:
        assert request['previous_response_id'] == 'resp_parent'
        if history_format != 'native':
            assert sent_items == outputs[::-1]


def test_responses_input_aligns_each_output_block_without_moving_other_items() -> None:
    reasoning = {'type': 'reasoning', 'id': 'rs_1', 'summary': [], 'encrypted_content': 'opaque'}
    first = {'type': 'function_call', 'call_id': 'call_1', 'name': 'Read', 'arguments': '{}'}
    second = {'type': 'computer_call', 'call_id': 'call_2', 'actions': []}
    first_output = {'type': 'function_call_output', 'call_id': 'call_1', 'output': 'text'}
    second_output = {
        'type': 'computer_call_output',
        'call_id': 'call_2',
        'output': {'type': 'computer_screenshot', 'image_url': 'data:image/png;base64,AAAA'},
    }
    separator = {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Continue.'}]}
    third = {**first, 'call_id': 'call_3'}
    fourth = {**first, 'call_id': 'call_4'}
    third_output = {**first_output, 'call_id': 'call_3'}
    fourth_output = {**first_output, 'call_id': 'call_4'}
    messages = [
        reasoning,
        first,
        second,
        second_output,
        first_output,
        separator,
        third,
        fourth,
        fourth_output,
        third_output,
    ]
    original = list(messages)

    converted = responses_module._responses_input(messages)

    assert converted == [
        reasoning,
        first,
        second,
        first_output,
        second_output,
        separator,
        third,
        fourth,
        third_output,
        fourth_output,
    ]
    assert messages == original
    assert converted[0] is reasoning
    assert converted[2] is second
    # Already ordered native input is passed through without copying.
    assert responses_module._responses_input(converted) is converted
    # Results separated by another message must never be moved across it.
    separated = [first, second, second_output, separator, first_output]
    assert responses_module._responses_input(separated) == separated
    # A stateful delta without source calls has no basis for reordering.
    delta = [second_output, first_output]
    assert responses_module._responses_input(delta) is delta


def test_responses_input_reused_call_id_preserves_previous_round_order() -> None:
    def call(call_id: str, arguments: str) -> dict[str, Any]:
        return {'type': 'function_call', 'call_id': call_id, 'name': 'Read', 'arguments': arguments}

    def output(call_id: str, value: str) -> dict[str, Any]:
        return {'type': 'function_call_output', 'call_id': call_id, 'output': value}

    history = [
        call('call_0', 'a1'),
        call('call_1', 'b1'),
        output('call_0', 'A1'),
        output('call_1', 'B1'),
        call('call_0', 'a2'),
        output('call_0', 'A2'),
    ]

    converted = responses_module._responses_input(history)

    assert [item['output'] for item in converted if item['type'] == 'function_call_output'] == ['A1', 'B1', 'A2']
    assert converted is history


@pytest.mark.parametrize('mode', ['full', 'stateful', 'encrypted', 'native'])
@pytest.mark.parametrize('first_outputs_reversed', [False, True])
@pytest.mark.parametrize('second_call_ids', [('call_0',), ('call_1', 'call_0')])
async def test_reused_parallel_call_ids_preserve_history_across_three_requests(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    first_outputs_reversed: bool,
    second_call_ids: tuple[str, ...],
) -> None:
    def response(round_name: str, call_ids: tuple[str, ...]) -> dict[str, Any]:
        return _response(
            response_id='resp_' + round_name,
            output=[
                {
                    'type': 'reasoning',
                    'id': 'rs_' + round_name,
                    'summary': [],
                    'encrypted_content': 'opaque_' + round_name,
                },
                *(
                    {
                        'type': 'function_call',
                        'id': f'fc_{round_name}_{call_id}',
                        'call_id': call_id,
                        'name': 'Read',
                        'arguments': json.dumps({'file_path': f'/{round_name}/{call_id}'}),
                    }
                    for call_id in call_ids
                ),
            ],
        )

    def outputs(call_ids: tuple[str, ...], round_number: int) -> list[dict[str, Any]]:
        return [
            {'type': 'function_call_output', 'call_id': call_id, 'output': f'{call_id}-round-{round_number}'}
            for call_id in call_ids
        ]

    first_ids = ('call_0', 'call_1')
    first_response = response('first', first_ids)
    second_response = response('second', second_call_ids)
    create = AsyncMock(side_effect=[first_response, second_response, response('done', ())])
    if mode == 'encrypted':
        client = _make_encrypted_client(monkeypatch, create)
    else:
        client, _ = _make_client(monkeypatch, create, client_args={'stateful_previous_response': mode == 'stateful'})
    history: list[Any] = [{'role': 'user', 'content': 'Read the requested files.'}]

    def append_round(completion: ExtendedChatCompletion, result_items: list[dict[str, Any]]) -> None:
        if mode == 'native':
            history.extend(completion.choices[0].message.provider_specific_fields['responses_output_items'])
            history.extend(result_items)
        else:
            history.append(completion.choices[0].message.model_dump(exclude_none=True))
            history.extend(
                {'role': 'tool', 'tool_call_id': item['call_id'], 'content': item['output']} for item in result_items
            )

    first = await client.create_completion(history)
    first_outputs = outputs(first_ids, 1)
    append_round(first, first_outputs[::-1] if first_outputs_reversed else first_outputs)
    second = await client.create_completion(history)
    second_outputs = outputs(second_call_ids, 2)
    append_round(second, second_outputs[::-1])
    original_history = json.loads(json.dumps(history))
    await client.create_completion(history)

    requests = [call.kwargs for call in create.await_args_list]
    assert len(requests) == 3
    assert history == original_history
    if mode == 'stateful':
        assert requests[1]['input'] == first_outputs
        assert requests[2]['input'] == second_outputs
        assert [request['previous_response_id'] for request in requests[1:]] == ['resp_first', 'resp_second']
    else:
        assert requests[1]['input'] == requests[0]['input'] + first_response['output'] + first_outputs
        assert requests[2]['input'] == requests[1]['input'] + second_response['output'] + second_outputs


def test_responses_input_uses_output_text_for_assistant_content_parts() -> None:
    converted = responses_module._responses_input(
        [
            {'role': 'system', 'content': 'You are CodeBuddy.'},
            {
                'role': 'user',
                'content': [{'type': 'text', 'text': 'Inspect the formatter.'}],
            },
            {
                'role': 'assistant',
                'content': [{'type': 'text', 'text': 'I will inspect it.'}],
                'tool_calls': [
                    {
                        'id': 'call_1',
                        'type': 'function',
                        'function': {'name': 'Bash', 'arguments': '{"command":"ls"}'},
                    }
                ],
            },
        ]
    )

    assert converted[1]['content'] == [{'type': 'input_text', 'text': 'Inspect the formatter.'}]
    assert converted[2] == {
        'role': 'assistant',
        'content': [{'type': 'output_text', 'text': 'I will inspect it.'}],
    }
    assert converted[3]['type'] == 'function_call'


@pytest.mark.asyncio
async def test_client_preserves_structured_image_function_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = AsyncMock(return_value=_text_response())
    client, _ = _make_client(monkeypatch, create)
    image_url = 'data:image/png;base64,' + 'A' * 2048
    messages = [
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                {
                    'id': 'call_view',
                    'type': 'function',
                    'function': {'name': 'view_image', 'arguments': '{"path":"/tmp/image.png"}'},
                }
            ],
        },
        {
            'role': 'tool',
            'tool_call_id': 'call_view',
            'content': [
                {'type': 'image_url', 'image_url': {'url': image_url, 'detail': 'high'}},
            ],
        },
    ]

    await client._create_completion(messages)

    assert create.await_args.kwargs['input'] == [
        {'role': 'assistant', 'content': ''},
        {
            'type': 'function_call',
            'call_id': 'call_view',
            'name': 'view_image',
            'arguments': '{"path":"/tmp/image.png"}',
        },
        {
            'type': 'function_call_output',
            'call_id': 'call_view',
            'output': [
                {'type': 'input_image', 'image_url': image_url, 'detail': 'high'},
            ],
        },
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('api_key', 'separator'),
    [
        ('plain-secret', '?'),
        ('app:secret?provider=azure_openai', '&'),
    ],
)
async def test_cache_task_id_is_hashed_into_sdk_key_and_raw_echo_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    api_key: str,
    separator: str,
) -> None:
    cache_task_id = 'chrome-b'
    cache_hash = hashlib.md5(cache_task_id.encode()).hexdigest()
    routed_key = f'{api_key}{separator}cache_task_id={cache_hash}'
    response = _text_response()
    response['metadata'] = {
        'routing_echo': f'provider returned {routed_key}',
        'hash_echo': cache_hash,
    }
    create = AsyncMock(return_value=response)
    client, init_calls = _make_client(
        monkeypatch,
        create,
        client_args={
            'api_key': api_key,
            'cache_task_id': cache_task_id,
            'capture_raw_io': True,
        },
    )

    completion = await client._create_completion([{'role': 'user', 'content': 'hello'}])

    assert init_calls == [{'api_key': routed_key, 'max_retries': 0}]
    persisted = json.dumps(completion.provider_specific_fields)
    assert routed_key in persisted
    assert cache_hash in persisted


def test_empty_cache_task_id_does_not_rewrite_key_or_reach_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    create = AsyncMock(return_value=_text_response())
    _client, init_calls = _make_client(
        monkeypatch,
        create,
        client_args={
            'api_key': 'official-secret',
            'cache_task_id': '',
        },
    )

    assert init_calls == [{'api_key': 'official-secret', 'max_retries': 0}]


def test_gateway_routing_is_merged_into_sdk_key_and_not_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = AsyncMock(return_value=_text_response())
    _client, init_calls = _make_client(
        monkeypatch,
        create,
        client_args={
            'api_key': 'official-secret',
            'gateway_routing': {
                'provider': 'azure',
                'model': 'gpt-5.6-sol',
                'timeout': 1800,
            },
        },
    )

    assert init_calls == [
        {
            'api_key': 'official-secret?provider=azure&model=gpt-5.6-sol&timeout=1800',
            'max_retries': 0,
        }
    ]


@pytest.mark.asyncio
async def test_ensure_v1_path_controls_sdk_base_and_recorded_responses_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = AsyncMock(return_value=_text_response())
    client, init_calls = _make_client(
        monkeypatch,
        create,
        client_args={
            'base_url': 'https://gateway.test/custom',
            'ensure_v1_path': True,
        },
    )

    completion = await client._create_completion([{'role': 'user', 'content': 'hello'}])

    assert init_calls == [
        {
            'api_key': 'test-key',
            'base_url': 'https://gateway.test/custom/v1',
            'max_retries': 0,
        }
    ]
    assert completion.provider_specific_fields['provider_call']['endpoint'] == (
        'https://gateway.test/custom/v1/responses'
    )


@pytest.mark.asyncio
async def test_client_uses_config_model_when_method_model_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    create = AsyncMock(return_value=_text_response())
    client, _ = _make_client(monkeypatch, create, request_args={'model': 'config-model'})

    await client._create_completion([{'role': 'user', 'content': 'hello'}])

    assert create.await_args.kwargs['model'] == 'config-model'


@pytest.mark.asyncio
async def test_exact_endpoint_controls_sdk_base_url_and_never_falls_back_to_public_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = AsyncMock(return_value=_text_response())
    client, init_calls = _make_client(
        monkeypatch,
        create,
        client_args={
            'base_url': 'https://api.openai.com/v1',
            'endpoint': 'https://gateway.test/custom/v1/responses',
            'capture_raw_io': True,
            'ensure_v1_path': True,
        },
    )

    completion = await client._create_completion([{'role': 'user', 'content': 'hello'}])

    assert init_calls == [
        {
            'api_key': 'test-key',
            'base_url': 'https://gateway.test/custom/v1/',
            'max_retries': 0,
        }
    ]
    assert completion.provider_specific_fields['provider_call']['endpoint'] == (
        'https://gateway.test/custom/v1/responses'
    )


@pytest.mark.parametrize(
    'endpoint',
    [
        'https://gateway.test/custom/v1/responses?api_key=secret',
        'https://user:secret@gateway.test/custom/v1/responses',
        'https://gateway.test/custom/v1/chat/completions',
        'not-a-url',
    ],
)
def test_exact_endpoint_rejects_unsafe_or_non_responses_urls(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
) -> None:
    create = AsyncMock(return_value=_text_response())
    init_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create, init_calls))

    with pytest.raises(ValueError, match='endpoint'):
        OpenAIResponsesClient(
            OpenAIConfig(
                client_args={'api_key': 'test-key', 'endpoint': endpoint},
                request_args={'model': 'gpt-cua'},
            )
        )

    assert init_calls == []


@pytest.mark.asyncio
async def test_missing_model_and_stream_true_are_recorded_complete_preflight_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for request_args, call_kwargs in (({}, {}), ({'model': 'gpt-cua'}, {'stream': True})):
        create = AsyncMock(return_value=_text_response())
        client, _ = _make_client(monkeypatch, create, request_args=request_args)
        records: list[dict[str, Any]] = []
        client._record_provider_call = records.append

        with pytest.raises(OpenAIResponsesRequestError):
            await client._create_completion([{'role': 'user', 'content': 'hello'}], **call_kwargs)

        assert create.await_count == 0
        assert len(records) == 1
        assert records[0]['attempts'] == []
        assert records[0]['error']['message'] in {
            'openai_responses requires a model in request_args or create_completion(model=...)',
            'openai_responses does not support stream=true',
        }


@pytest.mark.asyncio
async def test_background_true_is_rejected_before_network_and_records_preflight_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = AsyncMock(return_value=_response(status='queued'))
    client, _ = _make_client(monkeypatch, create)
    records: list[dict[str, Any]] = []
    client._record_provider_call = records.append

    with pytest.raises(OpenAIResponsesRequestError):
        await client._create_completion(
            [{'role': 'user', 'content': 'background'}],
            background=True,
        )

    assert create.await_count == 0
    assert len(records) == 1
    assert records[0]['attempts'] == []
    assert records[0]['error']['message'] == 'openai_responses does not support background=true'


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['failed', 'completed'])
async def test_failed_or_error_response_records_call_then_raises_safe_public_error(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    provider_error = {'code': 'provider_failed', 'message': 'credential test-key was rejected'}
    response_error = provider_error if status == 'completed' else None
    create = AsyncMock(
        return_value=_response(
            status=status,
            error=response_error,
            output=[{'type': 'message', 'content': [{'type': 'output_text', 'text': 'test-key'}]}],
        )
    )
    client, _ = _make_client(monkeypatch, create, client_args={'capture_raw_io': True})
    records: list[dict[str, Any]] = []
    client._record_provider_call = records.append

    with pytest.raises(OpenAIResponsesRequestError, match='OpenAI Responses request failed') as caught:
        await client._create_completion([{'role': 'user', 'content': 'run'}])

    assert 'test-key' not in str(caught.value)
    assert len(records) == 1
    assert records[0]['attempts'][0]['status'] == 'error'
    assert records[0]['error']['message'] == 'OpenAI Responses returned a non-final or failed response'
    assert 'test-key' in json.dumps(records[0])


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['queued', 'in_progress', 'cancelled'])
async def test_nonfinal_response_status_is_not_silently_normalized(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    create = AsyncMock(return_value=_response(status=status))
    client, _ = _make_client(monkeypatch, create)
    records: list[dict[str, Any]] = []
    client._record_provider_call = records.append

    with pytest.raises(OpenAIResponsesRequestError):
        await client._create_completion([{'role': 'user', 'content': 'run'}])

    assert len(records) == 1
    assert records[0]['attempts'][0]['status'] == 'error'


@pytest.mark.asyncio
async def test_retryable_errors_keep_failed_attempt_evidence_in_one_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_error = _status_error(500, {'error': {'message': 'retry secret-token'}})
    create = AsyncMock(side_effect=[first_error, _text_response()])
    client, _ = _make_client(
        monkeypatch,
        create,
        client_args={'capture_raw_io': True, 'max_retries': 2, 'api_key': 'secret-token'},
    )
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, 'sleep', sleep)

    completion = await client._create_completion([{'role': 'user', 'content': 'retry'}])

    assert create.await_count == 2
    sleep.assert_awaited_once()
    provider_call = completion.provider_specific_fields['provider_call']
    assert [attempt['status'] for attempt in provider_call['attempts']] == ['error', 'success']
    assert provider_call['attempts'][0]['status_code'] == 500
    assert provider_call['attempts'][0]['response']['body']['error']['message'] == 'retry <redacted>'
    assert provider_call['response']['id'] == 'resp_123'


@pytest.mark.asyncio
@pytest.mark.parametrize('error_code', ['server_error', 'rate_limit_exceeded', 'vector_store_timeout'])
async def test_failed_response_with_transient_official_error_retries_as_one_logical_attempt_each(
    monkeypatch: pytest.MonkeyPatch,
    error_code: str,
) -> None:
    failed_response = _response(
        status='failed',
        error={'code': error_code, 'message': 'retry secret-token'},
    )
    create = AsyncMock(side_effect=[failed_response, _text_response()])
    client, _ = _make_client(
        monkeypatch,
        create,
        client_args={'capture_raw_io': True, 'max_retries': 2, 'api_key': 'secret-token'},
    )
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, 'sleep', sleep)

    completion = await client._create_completion([{'role': 'user', 'content': 'retry'}])

    assert create.await_count == 2
    sleep.assert_awaited_once()
    provider_call = completion.provider_specific_fields['provider_call']
    assert [attempt['status'] for attempt in provider_call['attempts']] == ['error', 'success']
    assert provider_call['attempts'][0]['response']['status'] == 'failed'
    assert provider_call['attempts'][0]['response']['error'] == {
        'code': error_code,
        'message': 'retry <redacted>',
    }
    assert provider_call['response']['status'] == 'completed'


@pytest.mark.asyncio
async def test_failed_response_with_permanent_official_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_response = _response(
        status='failed',
        error={'code': 'invalid_prompt', 'message': 'invalid secret-token'},
    )
    create = AsyncMock(return_value=failed_response)
    client, _ = _make_client(
        monkeypatch,
        create,
        client_args={'capture_raw_io': True, 'max_retries': 3, 'api_key': 'secret-token'},
    )
    records: list[dict[str, Any]] = []
    client._record_provider_call = records.append
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, 'sleep', sleep)

    with pytest.raises(OpenAIResponsesRequestError) as caught:
        await client._create_completion([{'role': 'user', 'content': 'invalid'}])

    assert 'secret-token' not in str(caught.value)
    assert create.await_count == 1
    sleep.assert_not_awaited()
    assert len(records) == 1
    assert len(records[0]['attempts']) == 1
    assert records[0]['attempts'][0]['response']['error'] == {
        'code': 'invalid_prompt',
        'message': 'invalid secret-token',
    }


@pytest.mark.asyncio
async def test_ordinary_4xx_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    create = AsyncMock(side_effect=_status_error(400, {'error': {'message': 'invalid request'}}))
    client, _ = _make_client(monkeypatch, create, client_args={'capture_raw_io': True, 'max_retries': 3})
    records: list[dict[str, Any]] = []
    client._record_provider_call = records.append

    with pytest.raises(OpenAIResponsesRequestError):
        await client._create_completion([{'role': 'user', 'content': 'bad'}])

    assert create.await_count == 1
    assert records[0]['attempts'][0]['status_code'] == 400
    assert records[0]['attempts'][0]['response']['body'] == {'error': {'message': 'invalid request'}}


@pytest.mark.asyncio
async def test_connection_timeout_and_rate_limit_are_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    request = httpx.Request('POST', 'https://openai.test/v1/responses')
    errors: list[BaseException] = [
        openai.APIConnectionError(message='connection failed', request=request),
        openai.APITimeoutError(request),
        _status_error(429, {'error': 'busy'}),
    ]
    create = AsyncMock(side_effect=[*errors, _text_response()])
    client, _ = _make_client(monkeypatch, create, client_args={'max_retries': 4})
    monkeypatch.setattr(asyncio, 'sleep', AsyncMock())

    completion = await client._create_completion([{'role': 'user', 'content': 'retry'}])

    assert create.await_count == 4
    assert [attempt['status'] for attempt in completion.provider_specific_fields['provider_call']['attempts']] == [
        'error',
        'error',
        'error',
        'success',
    ]


@pytest.mark.asyncio
async def test_cancellation_during_retry_backoff_records_call_and_reraises_same_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request('POST', 'https://openai.test/v1/responses')
    create = AsyncMock(side_effect=openai.APIConnectionError(message='connection failed', request=request))
    client, _ = _make_client(
        monkeypatch,
        create,
        client_args={'api_key': 'backoff-secret', 'max_retries': 2},
    )
    records: list[dict[str, Any]] = []
    client._record_provider_call = records.append
    cancellation = asyncio.CancelledError('backoff-secret interrupted sleep')
    monkeypatch.setattr(asyncio, 'sleep', AsyncMock(side_effect=cancellation))

    with pytest.raises(asyncio.CancelledError) as caught:
        await client._create_completion([{'role': 'user', 'content': 'retry'}])

    assert caught.value is cancellation
    assert str(caught.value) == 'backoff-secret interrupted sleep'
    assert len(records) == 1
    assert records[0]['attempts'][0]['status'] == 'error'
    assert records[0]['error']['type'] == 'CancelledError'
    assert 'backoff-secret' in json.dumps(records[0])


@pytest.mark.asyncio
async def test_cancellation_is_recorded_completely_and_same_exception_is_reraised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancellation = asyncio.CancelledError('test-key interrupted request')
    create = AsyncMock(side_effect=cancellation)
    client, _ = _make_client(monkeypatch, create, client_args={'capture_raw_io': True})
    records: list[dict[str, Any]] = []
    client._record_provider_call = records.append

    with pytest.raises(asyncio.CancelledError) as caught:
        await client._create_completion([{'role': 'user', 'content': 'cancel'}])

    assert caught.value is cancellation
    assert str(caught.value) == 'test-key interrupted request'
    assert len(records) == 1
    assert records[0]['attempts'][0]['error_type'] == 'CancelledError'
    assert records[0]['error']['type'] == 'CancelledError'
    assert 'test-key' in json.dumps(records[0])


@pytest.mark.asyncio
async def test_context_length_error_is_normalized_after_provider_call_recording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = AsyncMock(
        side_effect=_status_error(
            400,
            {'error': {'code': 'context_length_exceeded', 'message': 'maximum context length exceeded'}},
        )
    )
    client, _ = _make_client(monkeypatch, create, client_args={'capture_raw_io': True})
    records: list[dict[str, Any]] = []
    client._record_provider_call = records.append

    with pytest.raises(ContextLengthExceededError, match='OpenAI Responses context length exceeded'):
        await client._create_completion([{'role': 'user', 'content': 'too long'}])

    assert len(records) == 1
    assert records[0]['attempts'][0]['status'] == 'error'


@pytest.mark.asyncio
async def test_capture_false_keeps_lossless_provider_state_but_omits_raw_provider_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = AsyncMock(return_value=_text_response('echo test-key'))
    client, _ = _make_client(monkeypatch, create, client_args={'capture_raw_io': False})

    completion = await client._create_completion([{'role': 'user', 'content': 'hello'}])

    fields = completion.provider_specific_fields
    assert fields['response_output'][0]['content'][0]['text'] == 'echo test-key'
    assert fields['response_status'] == 'completed'
    assert fields['response_id'] == 'resp_123'
    assert fields['response_model'] == 'gpt-cua'
    assert 'request' not in fields['provider_call']
    assert 'response' not in fields['provider_call']


@pytest.mark.asyncio
async def test_missing_response_id_records_failure_instead_of_fabricating_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _text_response()
    response.pop('id')
    create = AsyncMock(return_value=response)
    client, _ = _make_client(monkeypatch, create)
    records: list[dict[str, Any]] = []
    client._record_provider_call = records.append

    with pytest.raises(OpenAIResponsesRequestError):
        await client._create_completion([{'role': 'user', 'content': 'hello'}])

    assert len(records) == 1
    assert records[0]['attempts'][0]['status'] == 'error'


@pytest.mark.asyncio
async def test_required_provider_state_preserves_configured_credential_echoes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _text_response()
    response['id'] = 'credential-id'
    response['model'] = 'credential-model'
    create = AsyncMock(return_value=response)
    client, _ = _make_client(
        monkeypatch,
        create,
        client_args={
            'api_key': 'safe-key',
            'default_query': {
                'response_id_credential': 'credential-id',
                'model_credential': 'credential-model',
            },
            'capture_raw_io': False,
        },
    )

    completion = await client._create_completion([{'role': 'user', 'content': 'hello'}])

    assert completion.id == 'credential-id'
    assert completion.model == 'credential-model'
    assert completion.provider_specific_fields['response_id'] == 'credential-id'
    assert completion.provider_specific_fields['response_model'] == 'credential-model'


@pytest.mark.asyncio
async def test_operational_computer_arguments_persist_exactly_in_state_and_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = {
        'id': 'computer_1',
        'type': 'computer_call',
        'call_id': 'call_1',
        'actions': [
            {'type': 'click', 'x': 11, 'y': 22},
            {'type': 'type', 'text': 'typed-secret'},
        ],
        'pending_safety_checks': [
            {
                'id': 'safety-secret',
                'code': 'confirm',
                'message': 'typed-secret',
            }
        ],
        'future_field': {'credential_echo': 'typed-secret'},
    }
    create = AsyncMock(return_value=_response(output=[item]))
    client, _ = _make_client(
        monkeypatch,
        create,
        client_args={
            'api_key': 'typed-secret',
            'default_headers': {'X-Safety-ID': 'safety-secret'},
            'default_query': {'credential': 11},
            'capture_raw_io': True,
        },
    )
    processor = CapturingProcessor()

    with processor_context([processor]):
        completion = await client.create_completion([{'role': 'user', 'content': 'run'}])

    arguments = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)
    assert arguments == {
        'actions': item['actions'],
        'pending_safety_checks': item['pending_safety_checks'],
        'raw_item': item,
    }

    fields = completion.provider_specific_fields
    assert fields['response_output'][0]['actions'][0]['x'] == 11
    assert fields['response_output'][0]['actions'][1]['text'] == 'typed-secret'
    assert fields['response_output'][0]['pending_safety_checks'][0]['id'] == 'safety-secret'
    provider_output = fields['provider_call']['response']['output'][0]
    assert provider_output['actions'][0]['x'] == 11
    assert provider_output['actions'][1]['text'] == '<redacted>'
    assert provider_output['pending_safety_checks'][0]['id'] == 'safety-secret'
    assert provider_output['pending_safety_checks'][0]['message'] == '<redacted>'
    assert provider_output['future_field']['credential_echo'] == '<redacted>'

    span_output = processor.ended[-1].attributes['outputs']
    serialized_trace = json.dumps(span_output)
    assert 'typed-secret' in serialized_trace
    assert 'safety-secret' in serialized_trace
    assert '"x": 11' in serialized_trace


@pytest.mark.asyncio
async def test_operational_incomplete_reason_stays_raw_in_all_persisted_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = AsyncMock(
        return_value=_response(
            status='incomplete',
            incomplete_details={'reason': 'content_filter'},
            output=[],
        )
    )
    client, _ = _make_client(
        monkeypatch,
        create,
        client_args={
            'api_key': 'content_filter',
            'capture_raw_io': True,
        },
    )
    processor = CapturingProcessor()

    with processor_context([processor]):
        completion = await client.create_completion([{'role': 'user', 'content': 'run'}])

    assert completion.choices[0].finish_reason == 'content_filter'
    fields = completion.provider_specific_fields
    assert fields['response_incomplete_details'] == {'reason': 'content_filter'}
    assert fields['response_state']['incomplete_details'] == {'reason': 'content_filter'}
    assert fields['provider_call']['response']['incomplete_details'] == {'reason': '<redacted>'}
    assert 'content_filter' in json.dumps(processor.ended[-1].attributes['outputs'])


@pytest.mark.asyncio
async def test_operational_reasoning_items_stay_exact_in_persisted_copies_and_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reasoning_item = {
        'id': 'rs_1',
        'type': 'reasoning',
        'summary': [
            {
                'type': 'summary_text',
                'text': 'raw-summary-secret',
            }
        ],
        'content': [
            {
                'type': 'reasoning_text',
                'text': 'private reasoning detail',
            }
        ],
        'encrypted_content': 'opaque-encrypted-secret',
        'status': 'completed',
    }
    create = AsyncMock(return_value=_response(output=[reasoning_item]))
    client, _ = _make_client(
        monkeypatch,
        create,
        client_args={
            'api_key': 'opaque-encrypted-secret',
            'default_headers': {'X-Reasoning-Credential': 'raw-summary-secret'},
            'capture_raw_io': True,
        },
    )
    processor = CapturingProcessor()

    with processor_context([processor]):
        completion = await client.create_completion([{'role': 'user', 'content': 'run'}])

    message = completion.choices[0].message
    assert message.reasoning_content == 'raw-summary-secret'
    assert message.reasoning_items == [reasoning_item]

    fields = completion.provider_specific_fields
    persisted_item = fields['response_output'][0]
    assert persisted_item['summary'][0]['text'] == 'raw-summary-secret'
    assert persisted_item['encrypted_content'] == 'opaque-encrypted-secret'
    provider_item = fields['provider_call']['response']['output'][0]
    assert provider_item['summary'][0]['text'] == '<redacted>'
    assert provider_item['encrypted_content'] == '<redacted>'
    assert provider_item['content'] == persisted_item['content']

    serialized_trace = json.dumps(processor.ended[-1].attributes['outputs'])
    assert 'raw-summary-secret' in serialized_trace
    assert 'opaque-encrypted-secret' in serialized_trace


def test_encrypted_content_archive_preserves_recovery_metadata(tmp_path: Any) -> None:
    archive = responses_module._encrypted_content_archive(
        _response(
            response_id='resp_archive',
            output=[{'id': 'rs_1', 'type': 'reasoning', 'encrypted_content': 'encrypted-one'}],
        ),
        headers={'X-MS-Region': 'East US 2'},
        requested_model='gpt-5.6-sol',
        endpoint='/v1/responses',
        request_sequence=1,
        gateway_cache_task_id='affinity-hash',
    )
    archive_file = responses_module._write_encrypted_content_archive(tmp_path, archive)
    persisted = json.loads(archive_file.read_text())

    assert tmp_path.stat().st_mode & 0o777 == 0o700
    assert archive_file.stat().st_mode & 0o777 == 0o600
    assert persisted['region'] == 'East US 2'
    reasoning_block = persisted['reasoning_blocks'][0]
    assert reasoning_block['output_index'] == 0
    assert reasoning_block['reasoning_id'] == 'rs_1'
    assert reasoning_block['encrypted_content'] == 'encrypted-one'


@pytest.mark.asyncio
async def test_encrypted_content_archive_missing_region_is_preserved_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    response = _response(
        response_id='resp_archive',
        output=[{'id': 'rs_1', 'type': 'reasoning', 'encrypted_content': 'encrypted-preserved'}],
    )
    client, _ = _make_client(
        monkeypatch,
        AsyncMock(),
        client_args={'encrypted_content_archive_dir': str(tmp_path), 'max_retries': 3},
    )
    raw_create = AsyncMock(
        return_value=SimpleNamespace(
            http_response=SimpleNamespace(headers={}),
            parse=lambda: response,
        )
    )
    client.client.responses.with_raw_response = SimpleNamespace(create=raw_create)

    with pytest.raises(EncryptedContentArchiveError, match='x-ms-region'):
        await client.create_completion([{'role': 'user', 'content': 'run'}])

    assert raw_create.await_count == 1
    archive = json.loads(next(tmp_path.glob('*.enc.json')).read_text())
    assert archive['reasoning_blocks'][0]['encrypted_content'] == 'encrypted-preserved'
    assert archive['missing_fields'] == ['x-ms-region']


@pytest.mark.parametrize('archive_enabled', [False, True])
@pytest.mark.parametrize('native', [False, True])
async def test_request_deadline_applies_with_and_without_encrypted_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, archive_enabled: bool, native: bool
) -> None:
    monkeypatch.setenv('NEXUS_CALL_DEADLINE', '0.01')

    async def stalled_request(*_args: Any, **_kwargs: Any) -> None:
        await asyncio.Event().wait()

    create = AsyncMock(side_effect=stalled_request)
    client, _ = _make_client(
        monkeypatch,
        create,
        client_args={
            'max_retries': 1,
            **({'encrypted_content_archive_dir': str(tmp_path)} if archive_enabled else {}),
        },
    )
    client.client.responses.with_raw_response = SimpleNamespace(create=create)
    client.client.post = create

    call = (
        client.create_responses({'input': 'run'})
        if native
        else client.create_completion([{'role': 'user', 'content': 'run'}])
    )
    with pytest.raises(OpenAIResponsesRequestError) as caught:
        await asyncio.wait_for(call, timeout=1.0)

    assert isinstance(caught.value.__cause__, TimeoutError)
    assert 'client-side deadline' in str(caught.value.__cause__)
    assert create.await_count == 1
    assert not list(tmp_path.glob('*.enc.json'))


async def test_request_deadline_includes_async_archive_response_parsing(monkeypatch, tmp_path):
    monkeypatch.setenv('NEXUS_CALL_DEADLINE', '0.01')

    async def stalled_parse():
        await asyncio.Event().wait()

    client, _ = _make_client(
        monkeypatch,
        AsyncMock(),
        client_args={'encrypted_content_archive_dir': str(tmp_path), 'max_retries': 1},
    )
    client.client.responses.with_raw_response = SimpleNamespace(
        create=AsyncMock(return_value=SimpleNamespace(parse=stalled_parse))
    )
    with pytest.raises(OpenAIResponsesRequestError) as caught:
        await asyncio.wait_for(client.create_completion([{'role': 'user', 'content': 'run'}]), timeout=1.0)
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert not list(tmp_path.glob('*.enc.json'))


@pytest.mark.parametrize('native', [False, True])
async def test_retry_budget_bounds_native_and_completion_attempts(monkeypatch, native):
    monkeypatch.setenv('NEXUS_CALL_DEADLINE', '0')
    monkeypatch.setenv('NEXUS_RETRY_TOTAL_TIMEOUT', '0.5')
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(responses_module, 'time', SimpleNamespace(monotonic=lambda: clock.now))

    async def backoff(delay):
        clock.now += delay

    monkeypatch.setattr(responses_module.asyncio, 'sleep', backoff)
    create = AsyncMock(side_effect=_status_error(503, {'error': {'code': 'server_error'}}))
    client, _ = _make_client(monkeypatch, create, client_args={'max_retries': 3})
    client.client.post = create
    with pytest.raises(OpenAIResponsesRequestError) as caught:
        if native:
            await client.create_responses({'input': 'run'})
        else:
            await client.create_completion([{'role': 'user', 'content': 'run'}])
    assert 'retry total budget' in str(caught.value.__cause__)
    assert create.await_count == 1


@pytest.mark.parametrize('complete_reasoning', [False, True])
async def test_encrypted_archive_keeps_required_reasoning_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, complete_reasoning: bool
) -> None:
    response = _encrypted_tool_response('resp_archive', 'call_archive')
    if not complete_reasoning:
        response['output'].pop(0)
    create = AsyncMock(
        return_value=SimpleNamespace(
            http_response=SimpleNamespace(headers={'x-ms-region': 'test-region'}),
            parse=lambda: response,
        )
    )
    client, _ = _make_client(
        monkeypatch,
        AsyncMock(),
        client_args={
            'encrypted_content_archive_dir': str(tmp_path),
            'require_encrypted_reasoning': True,
            'max_retries': 3,
        },
        request_args={'model': 'test-model', 'include': ['reasoning.encrypted_content']},
    )
    client.client.responses.with_raw_response = SimpleNamespace(create=create)
    messages = [{'role': 'user', 'content': 'run'}]

    if complete_reasoning:
        completion = await client.create_completion(messages)
        archive_info = completion.provider_specific_fields['encrypted_content_archive']
        assert archive_info['archive_complete'] is True
        assert archive_info['encrypted_block_count'] == 1
        assert archive_info['region'] == 'test-region'
    else:
        with pytest.raises(OpenAIResponsesRequestError) as caught:
            await client.create_completion(messages)
        assert 'native reasoning state' in str(caught.value.__cause__)

    assert create.await_count == 1
    archive = json.loads(next(tmp_path.glob('*.enc.json')).read_text())
    assert archive['output_order'] == [
        {'output_index': i, 'type': item['type'], 'id': item['id']} for i, item in enumerate(response['output'])
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('region', [None, 'East US 2'])
async def test_archive_transport_preserves_native_and_completion_results(monkeypatch, tmp_path, native, region):
    body = _response(
        response_id='resp_archive_pair',
        output=[{'id': 'rs_1', 'type': 'reasoning', 'encrypted_content': 'opaque-state'}],
    )
    client, _ = _make_client(
        monkeypatch, AsyncMock(), client_args={'encrypted_content_archive_dir': str(tmp_path), 'max_retries': 3},
    )
    headers = {'x-ms-region': region} if region else {}
    raw = AsyncMock(return_value=SimpleNamespace(
        http_response=SimpleNamespace(headers=headers), parse=lambda: body,
    ))
    native_post = AsyncMock(return_value=httpx.Response(200, json=body, headers=headers))
    client.client.responses.with_raw_response = SimpleNamespace(create=raw)
    client.client.post = native_post

    async def call():
        if native:
            return await client.create_responses({'input': 'test'})
        return await client.create_completion([{'role': 'user', 'content': 'test'}])
    if region is None:
        with pytest.raises(EncryptedContentArchiveError, match='x-ms-region'):
            await call()
    else:
        result = await call()
        if native:
            assert result == body  # Archive metadata must not alter native provider output.
        else:
            assert result.provider_specific_fields['encrypted_content_archive']['region'] == region
    assert (native_post if native else raw).await_count == 1
    client.client.responses.create.assert_not_awaited()
    archive = json.loads(next(tmp_path.glob('*.enc.json')).read_text())
    assert archive['reasoning_blocks'][0]['encrypted_content'] == 'opaque-state'
    assert archive['archive_complete'] == bool(region)


def test_debug_env_does_not_install_credential_logging_transport(monkeypatch):
    monkeypatch.setenv('NEXUS_DUMP_LLM_REQ', '1')
    # The fake SDK has no private _client._transport: construction must not touch it.
    client, _ = _make_client(monkeypatch, AsyncMock())
    assert not hasattr(client.client, '_client')


@pytest.mark.asyncio
async def test_real_sdk_transport_does_not_log_authorization(monkeypatch, caplog):
    monkeypatch.setenv('NEXUS_DUMP_LLM_REQ', '1')
    secret = 'authorization-secret-sentinel-12345'
    observed = []

    async def handle(request):
        observed.append(request.headers['authorization'])
        return httpx.Response(200, json=_response(output=[]))
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    client = OpenAIResponsesClient(OpenAIConfig(
        client_args={'api_key': secret, 'base_url': 'https://fixture/v1', 'http_client': http_client},
        request_args={'model': 'test'},
    ))
    try:
        await client._create_completion([{'role': 'user', 'content': 'test'}])
        assert observed == ['Bearer ' + secret]
        assert secret not in caplog.text
        assert '[NEXUS_DUMP_LLM_REQ]' not in caplog.text
    finally:
        await client.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('native', [False, True])
async def test_content_policy_handling_preserves_default_retry_contract(monkeypatch, native):
    from nexus.core.openai_client import ContentPolicyViolationError
    error = _status_error(500, {'error': {'code': 'content_policy_violation'}})
    create = AsyncMock(side_effect=error)
    client, _ = _make_client(monkeypatch, create, client_args={'max_retries': 2})
    client.client.post = AsyncMock(side_effect=error)
    monkeypatch.setattr(responses_module.asyncio, 'sleep', AsyncMock())
    if native:
        with pytest.raises(ContentPolicyViolationError):
            await client.create_responses({'input': 'test'})
        assert client.client.post.await_count == 1
    else:
        with pytest.raises(OpenAIResponsesRequestError):
            await client._create_completion([{'role': 'user', 'content': 'test'}])
        assert create.await_count == 2


@pytest.mark.asyncio
async def test_concurrent_archives_keep_per_request_sequence(monkeypatch, tmp_path):
    client, _ = _make_client(
        monkeypatch, AsyncMock(), client_args={'encrypted_content_archive_dir': str(tmp_path)},
    )
    arrived = asyncio.Event()
    count = 0

    async def raw_create(**kwargs):
        nonlocal count
        count += 1
        number = count
        if count == 2:
            arrived.set()
        await arrived.wait()
        return SimpleNamespace(
            http_response=SimpleNamespace(headers={'x-ms-region': 'region'}),
            parse=lambda: _response(response_id=f'resp-{number}', output=[]),
        )
    client.client.responses.with_raw_response = SimpleNamespace(create=raw_create)
    await asyncio.gather(
        client._create_completion([{'role': 'user', 'content': 'one'}]),
        client._create_completion([{'role': 'user', 'content': 'two'}]),
    )
    archives = [json.loads(p.read_text()) for p in tmp_path.glob('*.enc.json')]
    assert sorted(a['request_sequence'] for a in archives) == [1, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('region', [None, 'East US 2'])
async def test_archive_transport_preserves_native_and_completion_results(monkeypatch, tmp_path, native, region):
    body = _response(
        response_id='resp_archive_pair',
        output=[{'id': 'rs_1', 'type': 'reasoning', 'encrypted_content': 'opaque-state'}],
    )
    client, _ = _make_client(
        monkeypatch, AsyncMock(), client_args={'encrypted_content_archive_dir': str(tmp_path), 'max_retries': 3},
    )
    headers = {'x-ms-region': region} if region else {}
    raw = AsyncMock(return_value=SimpleNamespace(
        http_response=SimpleNamespace(headers=headers), parse=lambda: body,
    ))
    native_post = AsyncMock(return_value=httpx.Response(200, json=body, headers=headers))
    client.client.responses.with_raw_response = SimpleNamespace(create=raw)
    client.client.post = native_post

    async def call():
        if native:
            return await client.create_responses({'input': 'test'})
        return await client.create_completion([{'role': 'user', 'content': 'test'}])
    if region is None:
        with pytest.raises(EncryptedContentArchiveError, match='x-ms-region'):
            await call()
    else:
        result = await call()
        if native:
            assert result == body  # Archive metadata must not alter native provider output.
        else:
            assert result.provider_specific_fields['encrypted_content_archive']['region'] == region
    assert (native_post if native else raw).await_count == 1
    client.client.responses.create.assert_not_awaited()
    archive = json.loads(next(tmp_path.glob('*.enc.json')).read_text())
    assert archive['reasoning_blocks'][0]['encrypted_content'] == 'opaque-state'
    assert archive['archive_complete'] == bool(region)


def test_debug_env_does_not_install_credential_logging_transport(monkeypatch):
    monkeypatch.setenv('NEXUS_DUMP_LLM_REQ', '1')
    # The fake SDK has no private _client._transport: construction must not touch it.
    client, _ = _make_client(monkeypatch, AsyncMock())
    assert not hasattr(client.client, '_client')


@pytest.mark.asyncio
async def test_real_sdk_transport_does_not_log_authorization(monkeypatch, caplog):
    monkeypatch.setenv('NEXUS_DUMP_LLM_REQ', '1')
    secret = 'authorization-secret-sentinel-12345'
    observed = []

    async def handle(request):
        observed.append(request.headers['authorization'])
        return httpx.Response(200, json=_response(output=[]))
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    client = OpenAIResponsesClient(OpenAIConfig(
        client_args={'api_key': secret, 'base_url': 'https://fixture/v1', 'http_client': http_client},
        request_args={'model': 'test'},
    ))
    try:
        await client._create_completion([{'role': 'user', 'content': 'test'}])
        assert observed == ['Bearer ' + secret]
        assert secret not in caplog.text
        assert '[NEXUS_DUMP_LLM_REQ]' not in caplog.text
    finally:
        await client.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('native', [False, True])
async def test_content_policy_handling_preserves_default_retry_contract(monkeypatch, native):
    from nexus.core.openai_client import ContentPolicyViolationError
    error = _status_error(500, {'error': {'code': 'content_policy_violation'}})
    create = AsyncMock(side_effect=error)
    client, _ = _make_client(monkeypatch, create, client_args={'max_retries': 2})
    client.client.post = AsyncMock(side_effect=error)
    monkeypatch.setattr(responses_module.asyncio, 'sleep', AsyncMock())
    if native:
        with pytest.raises(ContentPolicyViolationError):
            await client.create_responses({'input': 'test'})
        assert client.client.post.await_count == 1
    else:
        with pytest.raises(OpenAIResponsesRequestError):
            await client._create_completion([{'role': 'user', 'content': 'test'}])
        assert create.await_count == 2


@pytest.mark.asyncio
async def test_concurrent_archives_keep_per_request_sequence(monkeypatch, tmp_path):
    client, _ = _make_client(
        monkeypatch, AsyncMock(), client_args={'encrypted_content_archive_dir': str(tmp_path)},
    )
    arrived = asyncio.Event()
    count = 0

    async def raw_create(**kwargs):
        nonlocal count
        count += 1
        number = count
        if count == 2:
            arrived.set()
        await arrived.wait()
        return SimpleNamespace(
            http_response=SimpleNamespace(headers={'x-ms-region': 'region'}),
            parse=lambda: _response(response_id=f'resp-{number}', output=[]),
        )
    client.client.responses.with_raw_response = SimpleNamespace(create=raw_create)
    await asyncio.gather(
        client._create_completion([{'role': 'user', 'content': 'one'}]),
        client._create_completion([{'role': 'user', 'content': 'two'}]),
    )
    archives = [json.loads(p.read_text()) for p in tmp_path.glob('*.enc.json')]
    assert sorted(a['request_sequence'] for a in archives) == [1, 2]


@pytest.mark.asyncio
async def test_response_state_and_output_reconstruct_all_raw_top_level_fields_without_raw_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _text_response('state-secret')
    response.update(
        {
            'metadata': {
                'request_kind': 'test',
                'credential_echo': 'state-secret',
            },
            'future_top_level': {
                'nested': 'state-secret',
                'url': 'https://user:state-secret@future.test/path?api_key=state-secret',
            },
        }
    )
    create = AsyncMock(return_value=response)
    client, _ = _make_client(
        monkeypatch,
        create,
        client_args={
            'api_key': 'state-secret',
            'capture_raw_io': False,
        },
    )

    completion = await client._create_completion([{'role': 'user', 'content': 'hello'}])

    fields = completion.provider_specific_fields
    reconstructed = {
        **fields['response_state'],
        'output': fields['response_output'],
    }
    assert set(reconstructed) == set(response)
    assert reconstructed['metadata'] == {
        'request_kind': 'test',
        'credential_echo': 'state-secret',
    }
    assert reconstructed['future_top_level'] == {
        'nested': 'state-secret',
        'url': 'https://user:state-secret@future.test/path?api_key=state-secret',
    }
    assert reconstructed['output'][0]['content'][0]['text'] == 'state-secret'
    assert reconstructed['usage']['input_tokens'] == 10
    assert 'request' not in fields['provider_call']
    assert 'response' not in fields['provider_call']


@pytest.mark.asyncio
async def test_client_serializes_large_response_once_then_reuses_complete_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screenshot = f'data:image/png;base64,{"A" * 20000}'
    response = _response(
        output=[
            {
                'id': 'msg_large',
                'type': 'message',
                'content': [
                    {'type': 'output_text', 'text': screenshot},
                ],
            }
        ]
    )
    create = AsyncMock(return_value=response)
    client, _ = _make_client(
        monkeypatch,
        create,
        client_args={'capture_raw_io': True},
    )
    original_json_value = responses_module._json_value
    root_serializations = 0

    def spy_json_value(value: Any) -> Any:
        nonlocal root_serializations
        if value is response:
            root_serializations += 1
        return original_json_value(value)

    monkeypatch.setattr(responses_module, '_json_value', spy_json_value)

    completion = await client._create_completion([{'role': 'user', 'content': 'large'}])

    assert root_serializations == 1
    fields = completion.provider_specific_fields
    assert fields['response_output'][0]['content'][0]['text'] == screenshot
    assert fields['provider_call']['response']['output'][0]['content'][0]['text'] == screenshot
    assert len(fields['provider_call']['response']['output'][0]['content'][0]['text']) == len(screenshot)


@pytest.mark.asyncio
async def test_capture_true_redacts_credentials_and_preserves_screenshot_data_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screenshot = 'data:image/png;base64,AAAABBBB'
    response = _response(
        output=[
            {
                'type': 'message',
                'content': [
                    {
                        'type': 'output_text',
                        'text': (
                            'api-key=super-secret header-secret cookie-secret query-secret '
                            'https://echo:super-secret@openai.test/path?api_key=query-secret'
                        ),
                    },
                    {'type': 'output_text', 'text': screenshot},
                ],
            },
            {
                'type': 'computer_call',
                'call_id': 'call_1',
                'actions': [{'type': 'type', 'text': 'super-secret'}],
                'pending_safety_checks': [],
            },
        ],
        usage={
            'input_tokens': 11,
            'input_tokens_details': {'cached_tokens': 2, 'cache_write_tokens': 1},
            'output_tokens': 4,
            'output_tokens_details': {'reasoning_tokens': 1},
            'total_tokens': 15,
        },
    )
    create = AsyncMock(return_value=response)
    client, _ = _make_client(
        monkeypatch,
        create,
        client_args={
            'api_key': 'super-secret',
            'base_url': 'https://echo:super-secret@openai.test/v1?api_key=query-secret',
            'default_headers': {
                'Authorization': 'Bearer header-secret',
                'Cookie': 'sid=cookie-secret',
            },
            'default_query': {'access_token': 'query-secret', 'credential': 11},
            'capture_raw_io': True,
        },
    )
    request_input = [
        {
            'role': 'user',
            'content': [
                {'type': 'input_text', 'text': 'visit https://openai.test/page?api_key=query-secret'},
                {'type': 'input_image', 'image_url': screenshot},
            ],
        }
    ]

    completion = await client._create_completion(
        request_input,
        extra_headers={'Authorization': 'Bearer per-call-secret', 'Cookie': 'sid=per-call-cookie'},
        extra_query={'access_token': 'per-call-query'},
    )

    fields = completion.provider_specific_fields
    provider_call = fields['provider_call']
    serialized = json.dumps(provider_call)
    for secret in (
        'super-secret',
        'header-secret',
        'cookie-secret',
        'query-secret',
        'per-call-secret',
        'per-call-cookie',
        'per-call-query',
    ):
        assert secret not in serialized
    assert provider_call['endpoint'] == 'https://openai.test/v1/responses?api_key=<redacted>'
    assert provider_call['request']['input'][0]['content'][0]['text'] == (
        'visit https://openai.test/page?api_key=<redacted>'
    )
    assert provider_call['request']['input'][0]['content'][1]['image_url'] == screenshot
    assert provider_call['response']['output'][0]['content'][1]['text'] == screenshot
    assert provider_call['response']['usage']['input_tokens'] == 11
    assert provider_call['response']['usage']['output_tokens'] == 4
    assert provider_call['response']['usage']['total_tokens'] == 15


@pytest.mark.parametrize(
    ('enforced', 'request_args', 'expected'),
    [
        # The reported case: an enforced text.format must survive a request-level
        # response_format, which normalization writes into the same composite key.
        (
            {'text': {'format': {'type': 'text'}}},
            {'response_format': {'type': 'json_object'}},
            {'format': {'type': 'text'}},
        ),
        # ... without discarding a sibling the translation produced.
        (
            {'text': {'format': {'type': 'text'}}},
            {'verbosity': 'low', 'response_format': {'type': 'json_object'}},
            {'format': {'type': 'text'}, 'verbosity': 'low'},
        ),
        # An enforced sibling leaves the translated format alone.
        (
            {'text': {'verbosity': 'high'}},
            {'response_format': {'type': 'json_object'}},
            {'format': {'type': 'json_object'}, 'verbosity': 'high'},
        ),
    ],
)
def test_enforced_nested_controls_outrank_normalized_values(
    enforced: dict[str, Any],
    request_args: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    request = dict(request_args)
    responses_module._normalize_chat_request_controls(request)
    responses_module._reapply_enforced_nested_controls(request, enforced)

    assert request['text'] == expected


def test_enforced_reasoning_effort_outranks_a_translated_one() -> None:
    request = {'reasoning_effort': 'high'}
    responses_module._normalize_chat_request_controls(request)
    assert request['reasoning'] == {'effort': 'high'}

    responses_module._reapply_enforced_nested_controls(request, {'reasoning': {'effort': 'low'}})
    assert request['reasoning'] == {'effort': 'low'}


def test_reapply_enforced_nested_controls_is_a_noop_without_a_policy() -> None:
    request = {'response_format': {'type': 'json_object'}}
    responses_module._normalize_chat_request_controls(request)
    before = {key: dict(value) if isinstance(value, dict) else value for key, value in request.items()}

    responses_module._reapply_enforced_nested_controls(request, None)
    responses_module._reapply_enforced_nested_controls(request, {})
    responses_module._reapply_enforced_nested_controls(request, {'temperature': 0.5})

    assert request == before


@pytest.mark.asyncio
async def test_create_completion_keeps_enforced_text_format_over_a_request_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: the policy wins at the wire, and the record matches the wire.

    Enforced args are merged before normalization so that values they inject get
    translated too. That ordering lets a request-level ``response_format`` land in
    ``text.format`` after the merge and silently replace the enforced value, while
    the recorded effective args still showed the policy -- a log that disagrees
    with what was actually sent.
    """
    create = AsyncMock(return_value=_text_response())
    client, _ = _make_client(
        monkeypatch,
        create,
        client_args={'capture_raw_io': True},
        request_args={'model': 'gpt-5'},
    )
    client.config.enforced_request_args['text'] = {'format': {'type': 'text'}}
    records: list[dict[str, Any]] = []
    client._record_provider_call = records.append

    await client._create_completion(
        [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'hi'}]}],
        response_format={'type': 'json_schema', 'json_schema': {'name': 'answer', 'schema': {'type': 'object'}}},
        verbosity='low',
    )

    create.assert_awaited_once()
    sent = create.await_args.kwargs
    assert 'response_format' not in sent
    # The enforced format survives translation; the translated sibling survives it.
    assert sent['text'] == {'format': {'type': 'text'}, 'verbosity': 'low'}

    assert records, 'capture_raw_io should have recorded the provider call'
    recorded_request = records[-1]['request']
    assert recorded_request['text'] == sent['text']
    assert 'response_format' not in recorded_request


@pytest.mark.asyncio
async def test_create_completion_translates_response_format_without_a_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an enforced value the caller's own format is what goes out."""
    create = AsyncMock(return_value=_text_response())
    client, _ = _make_client(
        monkeypatch,
        create,
        client_args={'capture_raw_io': True},
        request_args={'model': 'gpt-5'},
    )
    records: list[dict[str, Any]] = []
    client._record_provider_call = records.append

    await client._create_completion(
        [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'hi'}]}],
        response_format={'type': 'json_object'},
    )

    sent = create.await_args.kwargs
    assert 'response_format' not in sent
    assert sent['text'] == {'format': {'type': 'json_object'}}
    assert records[-1]['request']['text'] == sent['text']
