import asyncio
import copy
import hashlib
import re

from urllib.parse import parse_qs

import pytest

from litellm import AmazonAnthropicClaudeConfig, BadRequestError
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from tenacity import wait_none

from nexus.core import OpenAIConfig
from nexus.extensions.openai_clients.litellm import (
    LiteLLMClient,
    _copy_provider_reasoning_signature_to_thinking_blocks,
    _deduplicate_gemini_reasoning_replay,
    _ensure_object_tool_parameters,
    _has_mid_conversation_system,
    _model_supports_mid_conversation_system,
    _restore_gemini_thought_signatures,
    _sanitize_request_kwargs,
    apply_cache_control,
    apply_cache_control_claude_code,
)


def _no_arg_tool():
    return {'type': 'function', 'function': {'name': 'memo_clear', 'parameters': {'type': 'object', 'properties': {}}}}


def test_cache_task_id_is_added_to_api_key_routing_query() -> None:
    client = LiteLLMClient(
        OpenAIConfig(
            client_args={
                'api_key': 'secret?provider=aws_third',
                'cache_task_id': 'stable task',
            },
            request_args={'model': 'bedrock/anthropic.claude-opus-4-8'},
        )
    )

    base, _, query = client.client_args['api_key'].partition('?')
    assert base == 'secret'
    assert parse_qs(query) == {
        'provider': ['aws_third'],
        'cache_task_id': ['stable task'],
    }
    assert 'cache_task_id' not in client.client_args


def test_request_id_can_be_derived_from_cache_task_id() -> None:
    client = LiteLLMClient(
        OpenAIConfig(
            client_args={
                'api_key': 'secret?provider=aws_third&model=anthropic.claude-opus-5',
                'cache_task_id': 'stable-cache-id',
                'request_id_from_cache_task_id': True,
            },
            request_args={'model': 'api_aws_third_anthropic.claude-opus-5'},
        )
    )

    _, _, query = client.client_args['api_key'].partition('?')
    expected_request_id = hashlib.md5(
        b'stable-cache-id:routed-auth',
        usedforsecurity=False,
    ).hexdigest()
    assert parse_qs(query) == {
        'provider': ['aws_third'],
        'model': ['anthropic.claude-opus-5'],
        'cache_task_id': ['stable-cache-id'],
        'request_id': [expected_request_id],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize('api_key', ['secret', 'secret?provider=ali', 'secret?provider=ali&model=x'])
async def test_provider_is_pinned_on_api_key_and_not_forwarded(monkeypatch, api_key) -> None:
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return FakeModelResponse(
            {
                'id': 'chatcmpl-provider',
                'created': 1,
                'model': kwargs['model'],
                'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'ok'}}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
            }
        )

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': api_key, 'provider': 'deepseek', 'cache_task_id': 'task-1', 'max_retries': 1},
            request_args={'model': 'openai/deepseek-flash'},
        )
    )

    base, _, query = client.client_args['api_key'].partition('?')
    assert base == 'secret'
    expected = {'provider': ['deepseek'], 'cache_task_id': ['task-1']}
    if 'model=x' in api_key:
        expected['model'] = ['x']
    assert parse_qs(query) == expected
    assert 'provider' not in client.client_args

    await client.create_completion(messages=[{'role': 'user', 'content': 'hi'}])
    assert len(calls) == 1
    assert 'provider' not in calls[0]
    assert parse_qs(calls[0]['api_key'].partition('?')[2]) == expected


def test_provider_requires_api_key() -> None:
    with pytest.raises(ValueError, match='provider requires a non-empty api_key'):
        LiteLLMClient(OpenAIConfig(client_args={'provider': 'deepseek'}, request_args={'model': 'openai/x'}))


def test_gemini_reasoning_replay_prefers_the_signed_tool_call_without_mutating_captured_cot() -> None:
    messages = [
        {
            'role': 'assistant',
            'content': '',
            'reasoning_content': 'captured plaintext CoT',
            'reasoning_items': [
                {
                    'type': 'reasoning',
                    'id': 'rs_synthetic',
                    'summary': [{'type': 'summary_text', 'text': 'captured plaintext CoT'}],
                }
            ],
            'thinking_blocks': [
                {
                    'type': 'thinking',
                    'thinking': 'captured plaintext CoT',
                    'signature': 'sig-tool',
                }
            ],
            'tool_calls': [
                {
                    'id': 'call_1__thought__sig-tool',
                    'type': 'function',
                    'function': {'name': 'bash', 'arguments': '{}'},
                }
            ],
        }
    ]

    replay = _deduplicate_gemini_reasoning_replay(messages)

    assert replay[0]['tool_calls'] == messages[0]['tool_calls']
    assert 'reasoning_content' not in replay[0]
    assert 'reasoning_items' not in replay[0]
    assert 'thinking_blocks' not in replay[0]
    assert messages[0]['reasoning_content'] == 'captured plaintext CoT'
    assert messages[0]['thinking_blocks'][0]['signature'] == 'sig-tool'


def test_gemini_reasoning_replay_reads_signature_restored_onto_short_tool_call_id() -> None:
    messages = [
        {
            'role': 'assistant',
            'content': '',
            'reasoning_content': 'duplicate plaintext',
            'reasoning_items': [
                {
                    'type': 'reasoning',
                    'id': 'rs_synthetic',
                    'summary': [{'type': 'summary_text', 'text': 'duplicate plaintext'}],
                }
            ],
            'thinking_blocks': [
                {'type': 'thinking', 'thinking': 'duplicate plaintext', 'signature': 'sig-tool'},
            ],
            'tool_calls': [
                {
                    'id': 'call_1',
                    'type': 'function',
                    'function': {'name': 'bash', 'arguments': '{}'},
                }
            ],
        }
    ]

    restored = _restore_gemini_thought_signatures(messages, {'call_1': 'sig-tool'})
    replay = _deduplicate_gemini_reasoning_replay(restored)

    assert replay[0]['tool_calls'][0]['provider_specific_fields'] == {'thought_signature': 'sig-tool'}
    assert 'reasoning_content' not in replay[0]
    assert 'reasoning_items' not in replay[0]
    assert 'thinking_blocks' not in replay[0]
    assert messages[0]['thinking_blocks'][0]['signature'] == 'sig-tool'


def test_gemini_reasoning_replay_uses_message_level_signatures_as_signed_representation() -> None:
    messages = [
        {
            'role': 'assistant',
            'content': 'answer',
            'reasoning_content': 'duplicate summary',
            'reasoning_items': [
                {
                    'type': 'reasoning',
                    'id': 'rs_synthetic',
                    'summary': [{'type': 'summary_text', 'text': 'duplicate summary'}],
                }
            ],
            'provider_specific_fields': {'thought_signatures': ['sig-message']},
        }
    ]

    replay = _deduplicate_gemini_reasoning_replay(messages)

    assert replay[0]['provider_specific_fields'] == messages[0]['provider_specific_fields']
    assert 'reasoning_content' not in replay[0]
    assert 'reasoning_items' not in replay[0]


def test_gemini_reasoning_replay_keeps_one_native_part_for_a_shared_message_and_block_signature() -> None:
    from litellm.llms.vertex_ai.gemini.transformation import _gemini_convert_messages_with_history

    signature = 'sig-shared'
    messages = [
        {
            'role': 'assistant',
            'content': 'answer',
            'reasoning_content': 'prior thinking',
            'thinking_blocks': [
                {'type': 'thinking', 'thinking': 'prior thinking', 'signature': signature},
            ],
            'provider_specific_fields': {'thought_signatures': [signature]},
        }
    ]

    replay = _deduplicate_gemini_reasoning_replay(messages)
    contents = _gemini_convert_messages_with_history(
        replay,
        model='gemini-4-argon',
        custom_llm_provider='vertex_ai',
    )
    parts = [part for content in contents for part in content.get('parts', [])]

    assert replay[0]['thinking_blocks'] == messages[0]['thinking_blocks']
    assert 'provider_specific_fields' not in replay[0]
    assert [part.get('thoughtSignature') for part in parts].count(signature) == 1


def test_gemini_reasoning_replay_retains_distinct_message_level_signatures() -> None:
    from litellm.llms.vertex_ai.gemini.transformation import _gemini_convert_messages_with_history

    block_signature = 'sig-block'
    text_signature = 'sig-text'
    messages = [
        {
            'role': 'assistant',
            'content': 'answer',
            'thinking_blocks': [
                {'type': 'thinking', 'thinking': 'prior thinking', 'signature': block_signature},
            ],
            'provider_specific_fields': {
                'thought_signatures': [block_signature, text_signature, text_signature],
            },
        }
    ]

    replay = _deduplicate_gemini_reasoning_replay(messages)
    contents = _gemini_convert_messages_with_history(
        replay,
        model='gemini-4-argon',
        custom_llm_provider='vertex_ai',
    )
    parts = [part for content in contents for part in content.get('parts', [])]

    assert replay[0]['provider_specific_fields']['thought_signatures'] == [text_signature]
    assert [part.get('thoughtSignature') for part in parts].count(block_signature) == 1
    assert [part.get('thoughtSignature') for part in parts].count(text_signature) == 1


def test_gemini_reasoning_replay_prefers_a_signed_thinking_block_without_a_tool_call() -> None:
    messages = [
        {
            'role': 'assistant',
            'content': 'answer',
            'reasoning_content': 'duplicate summary',
            'reasoning_items': [
                {
                    'type': 'reasoning',
                    'id': 'rs_synthetic',
                    'summary': [{'type': 'summary_text', 'text': 'duplicate summary'}],
                }
            ],
            'thinking_blocks': [
                {'type': 'thinking', 'thinking': 'signed thought', 'signature': 'sig-text'},
            ],
        }
    ]

    replay = _deduplicate_gemini_reasoning_replay(messages)

    assert replay[0]['thinking_blocks'] == messages[0]['thinking_blocks']
    assert 'reasoning_content' not in replay[0]
    assert 'reasoning_items' not in replay[0]


def test_gemini_reasoning_replay_keeps_plaintext_when_no_signed_representation_exists() -> None:
    messages = [
        {
            'role': 'assistant',
            'content': 'answer',
            'reasoning_content': 'only available reasoning',
            'reasoning_items': [
                {
                    'type': 'reasoning',
                    'id': 'rs_synthetic',
                    'summary': [{'type': 'summary_text', 'text': 'only available reasoning'}],
                }
            ],
        }
    ]

    assert _deduplicate_gemini_reasoning_replay(messages) == messages


def test_gemini_reasoning_replay_preserves_native_encrypted_items_and_distinct_signatures() -> None:
    messages = [
        {
            'role': 'assistant',
            'content': '',
            'reasoning_content': 'duplicate plaintext',
            'reasoning_items': [
                {'type': 'reasoning', 'id': 'rs_native', 'encrypted_content': 'enc'},
                {'type': 'reasoning', 'id': 'rs_synthetic', 'summary': []},
            ],
            'thinking_blocks': [
                {'type': 'thinking', 'thinking': 'distinct thought', 'signature': 'sig-text'},
            ],
            'tool_calls': [
                {
                    'id': 'call_1__thought__sig-tool',
                    'type': 'function',
                    'function': {'name': 'bash', 'arguments': '{}'},
                }
            ],
        }
    ]

    replay = _deduplicate_gemini_reasoning_replay(messages)

    assert replay[0]['reasoning_items'] == [
        {'type': 'reasoning', 'id': 'rs_native', 'encrypted_content': 'enc'},
    ]
    assert replay[0]['thinking_blocks'] == messages[0]['thinking_blocks']
    assert 'reasoning_content' not in replay[0]


def test_ensure_object_tool_parameters_drops_empty_properties():
    normalized = _ensure_object_tool_parameters([_no_arg_tool()])

    assert normalized[0]['function']['parameters'] == {'type': 'object'}


def test_ensure_object_tool_parameters_coerces_missing_parameters_to_object():
    normalized = _ensure_object_tool_parameters([{'type': 'function', 'function': {'name': 'no_params'}}])

    assert normalized[0]['function']['parameters'] == {'type': 'object'}


def test_ensure_object_tool_parameters_leaves_well_formed_schema_untouched():
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'write_file',
                'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}},
            },
        }
    ]

    normalized = _ensure_object_tool_parameters(tools)

    assert normalized is tools


def test_sanitize_request_kwargs_drops_top_p_for_bedrock_anthropic_when_temperature_is_set():
    request_kwargs = {
        'model': 'anthropic.claude-sonnet-4-6',
        'custom_llm_provider': 'bedrock',
        'temperature': 0.7,
        'top_p': 0.95,
        'messages': [{'role': 'user', 'content': 'hi'}],
    }

    sanitized = _sanitize_request_kwargs(request_kwargs)

    assert sanitized is not request_kwargs
    assert sanitized['temperature'] == 0.7
    assert 'top_p' not in sanitized
    assert request_kwargs['top_p'] == 0.95


def test_sanitize_request_kwargs_keeps_top_p_for_non_bedrock_requests():
    request_kwargs = {
        'model': 'gpt-4.1',
        'custom_llm_provider': 'openai',
        'temperature': 0.7,
        'top_p': 0.95,
        'messages': [{'role': 'user', 'content': 'hi'}],
    }

    sanitized = _sanitize_request_kwargs(request_kwargs)

    assert sanitized is request_kwargs
    assert sanitized['top_p'] == 0.95


def test_apply_cache_control_tags_tool_message_at_message_level():
    messages = apply_cache_control(
        [{'role': 'tool', 'tool_call_id': 'toolu_1', 'content': 'file contents'}],
        last_n_messages=1,
    )

    assert messages == [
        {
            'role': 'tool',
            'tool_call_id': 'toolu_1',
            'content': 'file contents',
            'cache_control': {'type': 'ephemeral'},
        }
    ]


def test_apply_cache_control_claude_code_tags_tool_message_at_message_level():
    messages = apply_cache_control_claude_code(
        [{'role': 'tool', 'tool_call_id': 'toolu_1', 'content': 'file contents'}]
    )

    assert messages == [
        {
            'role': 'tool',
            'tool_call_id': 'toolu_1',
            'content': 'file contents',
            'cache_control': {'type': 'ephemeral'},
        }
    ]


def test_bedrock_final_payload_preserves_cache_control_on_converted_tool_result():
    messages = apply_cache_control_claude_code(
        [
            {'role': 'user', 'content': 'Read the file.'},
            {
                'role': 'assistant',
                'content': 'Reading.',
                'tool_calls': [
                    {
                        'id': 'toolu_1',
                        'type': 'function',
                        'function': {'name': 'Read', 'arguments': '{"file_path":"README.md"}'},
                    }
                ],
            },
            {'role': 'tool', 'tool_call_id': 'toolu_1', 'content': 'file contents'},
        ]
    )

    payload = AmazonAnthropicClaudeConfig().transform_request(
        model='invoke/anthropic.claude-opus-4-8',
        messages=messages,
        optional_params={'max_tokens': 1024, 'tools': [_no_arg_tool()]},
        litellm_params={},
        headers={},
    )

    assert payload['messages'][-1]['content'][0] == {
        'type': 'tool_result',
        'tool_use_id': 'toolu_1',
        'content': 'file contents',
        'cache_control': {'type': 'ephemeral'},
    }


def test_copy_provider_reasoning_signature_fills_existing_thinking_block():
    response_json = {
        'choices': [
            {
                'message': {
                    'thinking_blocks': [{'type': 'thinking', 'thinking': '', 'signature': None}],
                    'provider_specific_fields': {'reasoningContent': {'signature': 'provider-signature'}},
                }
            }
        ]
    }

    normalized = _copy_provider_reasoning_signature_to_thinking_blocks(response_json)

    assert normalized['choices'][0]['message']['thinking_blocks'] == [
        {'type': 'thinking', 'thinking': '', 'signature': 'provider-signature'}
    ]


class FakeModelResponse:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return self.payload


class FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _path_keys(value, prefix=()):
    if isinstance(value, dict):
        out = {prefix}
        for key, child in value.items():
            out |= _path_keys(child, (*prefix, key))
        return out
    if isinstance(value, list):
        out = {prefix}
        for index, child in enumerate(value):
            out |= _path_keys(child, (*prefix, index))
        return out
    return {prefix}


def _rich_response_payload(response_id='chatcmpl-rich'):
    return {
        'id': response_id,
        'created': 1,
        'model': 'gpt-test',
        'object': 'chat.completion',
        'choices': [
            {
                'index': 0,
                'finish_reason': 'tool_calls',
                'message': {
                    'role': 'assistant',
                    'content': 'builder response',
                    'reasoning_content': 'reasoning text',
                    'thinking_blocks': [
                        {
                            'type': 'thinking',
                            'thinking': '',
                            'signature': 'claude-signature',
                        }
                    ],
                    'provider_specific_fields': {
                        'reasoning': {'encrypted_content': 'opaque-openai-reasoning'},
                        'reasoningContent': {'signature': 'claude-signature'},
                    },
                    'tool_calls': [
                        {
                            'id': 'call_1',
                            'type': 'function',
                            'function': {'name': 'write_file', 'arguments': '{"path":"/tmp/a"}'},
                        }
                    ],
                },
            }
        ],
        'usage': {
            'prompt_tokens': 2,
            'completion_tokens': 3,
            'total_tokens': 5,
            'completion_tokens_details': {
                'reasoning_tokens': 1,
                'text_tokens': 2,
                'image_tokens': 0,
                'video_tokens': 0,
            },
            'prompt_tokens_details': {
                'audio_tokens': 0,
                'cached_tokens': 0,
                'image_tokens': 0,
                'video_tokens': 0,
            },
        },
    }


@pytest.mark.asyncio
async def test_litellm_stream_true_uses_streaming_and_returns_full_response(monkeypatch):
    calls = []
    builder_calls = []
    chunks = [
        FakeModelResponse(
            {
                'id': 'chatcmpl-stream',
                'created': 1,
                'model': 'gpt-test',
                'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'hello'}}],
            }
        ),
        FakeModelResponse(
            {
                'id': 'chatcmpl-stream',
                'created': 1,
                'model': 'gpt-test',
                'choices': [
                    {
                        'index': 0,
                        'delta': {
                            'content': ' world',
                            'provider_specific_fields': {'reasoning': {'encrypted_content': 'opaque-openai-reasoning'}},
                        },
                        'finish_reason': 'stop',
                    }
                ],
                'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5},
            }
        ),
    ]

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        assert kwargs['stream'] is True
        return FakeAsyncStream(chunks)

    def fake_stream_chunk_builder(received_chunks, messages):
        builder_calls.append({'chunks': received_chunks, 'messages': messages})
        return FakeModelResponse(
            {
                'id': 'chatcmpl-stream-built',
                'created': 1,
                'model': 'gpt-test',
                'choices': [
                    {
                        'index': 0,
                        'finish_reason': 'stop',
                        'message': {
                            'role': 'assistant',
                            'content': 'builder response',
                            'provider_specific_fields': {'reasoning': {'encrypted_content': 'opaque-openai-reasoning'}},
                        },
                    }
                ],
                'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5},
            }
        )

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    monkeypatch.setattr(
        'nexus.extensions.openai_clients.litellm.litellm.stream_chunk_builder', fake_stream_chunk_builder
    )

    client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'test', 'stream': True, 'max_retries': 1},
            request_args={'model': 'gpt-test'},
        )
    )

    response = await client.create_completion(messages=[{'role': 'user', 'content': 'hi'}])

    assert len(calls) == 1
    assert calls[0]['stream_options']['include_usage'] is True
    assert builder_calls == [{'chunks': chunks, 'messages': [{'role': 'user', 'content': 'hi'}]}]
    assert response.id == 'chatcmpl-stream-built'
    assert response.choices[0].message.content == 'builder response'
    assert response.choices[0].finish_reason == 'stop'
    assert response.choices[0].message.provider_specific_fields == {
        'reasoning': {'encrypted_content': 'opaque-openai-reasoning'}
    }
    assert response.choices[0].message.thinking_blocks is None
    assert response.usage.total_tokens == 5


@pytest.mark.asyncio
async def test_litellm_stream_true_copies_provider_reasoning_signature_to_thinking_blocks(monkeypatch):
    stream_chunks = [
        FakeModelResponse(
            {
                'id': 'chatcmpl-stream',
                'created': 1,
                'model': 'bedrock/anthropic.claude-opus-4-8',
                'choices': [
                    {
                        'index': 0,
                        'delta': {
                            'role': 'assistant',
                            'provider_specific_fields': {
                                'reasoningContent': {
                                    'signature': 'claude-4-8-stream-signature',
                                }
                            },
                            'thinking_blocks': [
                                {
                                    'type': 'thinking',
                                    'thinking': '',
                                    'signature': 'claude-4-8-stream-signature',
                                }
                            ],
                        },
                        'finish_reason': 'stop',
                    }
                ],
                'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5},
            }
        )
    ]

    async def fake_acompletion(**kwargs):
        return FakeAsyncStream(stream_chunks)

    def fake_stream_chunk_builder(received_chunks, messages):
        assert received_chunks == stream_chunks
        return FakeModelResponse(
            {
                'id': 'chatcmpl-stream-built',
                'created': 1,
                'model': 'bedrock/anthropic.claude-opus-4-8',
                'choices': [
                    {
                        'index': 0,
                        'finish_reason': 'stop',
                        'message': {
                            'role': 'assistant',
                            'content': 'ok',
                            'thinking_blocks': None,
                            'provider_specific_fields': {
                                'reasoningContent': {
                                    'signature': 'claude-4-8-stream-signature',
                                }
                            },
                        },
                    }
                ],
                'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5},
            }
        )

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    monkeypatch.setattr(
        'nexus.extensions.openai_clients.litellm.litellm.stream_chunk_builder', fake_stream_chunk_builder
    )

    client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'test', 'stream': True, 'max_retries': 1},
            request_args={'model': 'bedrock/anthropic.claude-opus-4-8'},
        )
    )

    response = await client.create_completion(messages=[{'role': 'user', 'content': 'hi'}])

    assert response.choices[0].message.thinking_blocks is not None
    assert response.choices[0].message.thinking_blocks[0].thinking == ''
    assert response.choices[0].message.thinking_blocks[0].signature == 'claude-4-8-stream-signature'
    assert response.choices[0].message.provider_specific_fields == {
        'reasoningContent': {'signature': 'claude-4-8-stream-signature'}
    }


@pytest.mark.asyncio
async def test_litellm_stream_and_non_stream_outputs_have_same_path_keys(monkeypatch):
    final_payload = _rich_response_payload()
    stream_chunks = [
        FakeModelResponse(
            {
                'id': 'chatcmpl-stream',
                'created': 1,
                'model': 'gpt-test',
                'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'builder '}}],
            }
        ),
        FakeModelResponse(
            {
                'id': 'chatcmpl-stream',
                'created': 1,
                'model': 'gpt-test',
                'choices': [{'index': 0, 'delta': {'content': 'response'}, 'finish_reason': 'tool_calls'}],
                'usage': final_payload['usage'],
            }
        ),
    ]

    async def fake_acompletion(**kwargs):
        if kwargs.get('stream'):
            return FakeAsyncStream(stream_chunks)
        return FakeModelResponse(final_payload)

    def fake_stream_chunk_builder(received_chunks, messages):
        assert received_chunks == stream_chunks
        return FakeModelResponse(final_payload)

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    monkeypatch.setattr(
        'nexus.extensions.openai_clients.litellm.litellm.stream_chunk_builder', fake_stream_chunk_builder
    )

    non_stream_client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'test', 'stream': False, 'max_retries': 1},
            request_args={'model': 'gpt-test'},
        )
    )
    stream_client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'test', 'stream': True, 'max_retries': 1},
            request_args={'model': 'gpt-test'},
        )
    )

    non_stream_response = await non_stream_client.create_completion(messages=[{'role': 'user', 'content': 'hi'}])
    stream_response = await stream_client.create_completion(messages=[{'role': 'user', 'content': 'hi'}])

    assert _path_keys(stream_response.model_dump()) == _path_keys(non_stream_response.model_dump())


@pytest.mark.asyncio
async def test_litellm_stream_true_preserves_raw_finish_reason_when_builder_defaults_to_stop(monkeypatch):
    stream_chunks = [
        FakeModelResponse(
            {
                'id': 'chatcmpl-stream',
                'created': 1,
                'model': 'gpt-test',
                'choices': [
                    {
                        'index': 0,
                        'delta': {
                            'role': 'assistant',
                            'tool_calls': [
                                {
                                    'id': 'call_1',
                                    'type': 'function',
                                    'function': {'name': 'write_file', 'arguments': ''},
                                    'index': 0,
                                }
                            ],
                        },
                    }
                ],
            }
        ),
        FakeModelResponse(
            {
                'id': 'chatcmpl-stream',
                'created': 1,
                'model': 'gpt-test',
                'choices': [
                    {
                        'index': 0,
                        'delta': {
                            'tool_calls': [
                                {
                                    'id': None,
                                    'type': 'function',
                                    'function': {'name': None, 'arguments': '{"path":"/tmp/a"}'},
                                    'index': 0,
                                }
                            ]
                        },
                        'finish_reason': 'tool_calls',
                    }
                ],
                'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5},
            }
        ),
    ]

    async def fake_acompletion(**kwargs):
        return FakeAsyncStream(stream_chunks)

    def fake_stream_chunk_builder(received_chunks, messages):
        assert received_chunks == stream_chunks
        return FakeModelResponse(
            {
                'id': 'chatcmpl-stream-built',
                'created': 1,
                'model': 'gpt-test',
                'choices': [
                    {
                        'index': 0,
                        'finish_reason': 'stop',
                        'message': {
                            'role': 'assistant',
                            'content': None,
                            'tool_calls': [
                                {
                                    'id': 'call_1',
                                    'type': 'function',
                                    'function': {'name': 'write_file', 'arguments': '{"path":"/tmp/a"}'},
                                }
                            ],
                        },
                    }
                ],
                'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5},
            }
        )

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    monkeypatch.setattr(
        'nexus.extensions.openai_clients.litellm.litellm.stream_chunk_builder',
        fake_stream_chunk_builder,
    )

    client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'test', 'stream': True, 'max_retries': 1},
            request_args={'model': 'gpt-test'},
        )
    )

    response = await client.create_completion(messages=[{'role': 'user', 'content': 'hi'}])

    assert response.choices[0].finish_reason == 'tool_calls'


@pytest.mark.asyncio
async def test_litellm_stream_true_preserves_explicit_stream_options(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return FakeAsyncStream(
            [
                FakeModelResponse(
                    {
                        'id': 'chatcmpl-stream',
                        'created': 1,
                        'model': 'gpt-test',
                        'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'hello'}}],
                    }
                )
            ]
        )

    def fake_stream_chunk_builder(received_chunks, messages):
        return FakeModelResponse(
            {
                'id': 'chatcmpl-stream',
                'created': 1,
                'model': 'gpt-test',
                'choices': [
                    {
                        'index': 0,
                        'finish_reason': 'stop',
                        'message': {'role': 'assistant', 'content': 'hello'},
                    }
                ],
            }
        )

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    monkeypatch.setattr(
        'nexus.extensions.openai_clients.litellm.litellm.stream_chunk_builder', fake_stream_chunk_builder
    )

    client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'test', 'stream': True, 'max_retries': 1},
            request_args={'model': 'gpt-test', 'stream_options': {'include_usage': False, 'custom': 'value'}},
        )
    )

    await client.create_completion(messages=[{'role': 'user', 'content': 'hi'}])

    assert len(calls) == 1
    assert calls[0]['stream_options'] == {'include_usage': False, 'custom': 'value'}


@pytest.mark.asyncio
async def test_litellm_stream_false_keeps_non_streaming_path(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        assert 'stream' not in kwargs
        return FakeModelResponse(
            {
                'id': 'chatcmpl-full',
                'created': 1,
                'model': 'gpt-test',
                'choices': [
                    {
                        'index': 0,
                        'finish_reason': 'stop',
                        'message': {'role': 'assistant', 'content': 'full response'},
                    }
                ],
                'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5},
            }
        )

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)

    client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'test', 'stream': False, 'max_retries': 1}, request_args={'model': 'gpt-test'}
        )
    )

    response = await client.create_completion(messages=[{'role': 'user', 'content': 'hi'}])

    assert len(calls) == 1
    assert response.id == 'chatcmpl-full'
    assert response.choices[0].message.content == 'full response'


def _ok_response():
    return FakeModelResponse(
        {
            'id': 'chatcmpl-ok',
            'created': 1,
            'model': 'gpt-test',
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'ok'}}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
        }
    )


@pytest.mark.asyncio
async def test_litellm_maps_nexus_session_routing_params_to_headers(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)

    client = LiteLLMClient(
        OpenAIConfig(
            client_args={
                'api_key': 'test',
                'max_retries': 1,
                'nexus_trace_id': 'stable-session',
                'nexus_llm_lb_strategy': 'ConsistentHash',
            },
            request_args={
                'model': 'openai/hy4',
                'extra_headers': {'X-Test': 'preserved'},
            },
        )
    )

    await client.create_completion(messages=[{'role': 'user', 'content': 'hi'}])

    assert 'nexus_trace_id' not in calls[0]
    assert 'nexus_llm_lb_strategy' not in calls[0]
    assert calls[0]['extra_headers'] == {
        'X-Test': 'preserved',
        'SessionID': 'stable-session',
        'StrategyType': 'ConsistentHash',
    }


@pytest.mark.asyncio
async def test_explicit_session_header_overrides_nexus_default(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)

    client = LiteLLMClient(
        OpenAIConfig(
            client_args={
                'api_key': 'test',
                'max_retries': 1,
                'nexus_trace_id': 'derived-session',
                'nexus_llm_lb_strategy': 'ConsistentHash',
            },
            request_args={
                'model': 'openai/hy4',
                'extra_headers': {
                    'SessionID': 'explicit-session',
                    'X-Campaign': '0811',
                },
            },
        )
    )

    await client.create_completion(messages=[{'role': 'user', 'content': 'hi'}])

    assert calls[0]['extra_headers'] == {
        'SessionID': 'explicit-session',
        'StrategyType': 'ConsistentHash',
        'X-Campaign': '0811',
    }


@pytest.mark.asyncio
async def test_litellm_normalizes_empty_tool_properties_for_gemini(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)

    client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'test', 'stream': False, 'max_retries': 1},
            request_args={'model': 'api_google_gemini-3.1-pro-preview'},
        )
    )

    await client.create_completion(messages=[{'role': 'user', 'content': 'hi'}], tools=[_no_arg_tool()])

    assert len(calls) == 1
    assert calls[0]['tools'][0]['function']['parameters'] == {'type': 'object'}


@pytest.mark.asyncio
async def test_litellm_leaves_empty_tool_properties_untouched_for_non_gemini(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)

    client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'test', 'stream': False, 'max_retries': 1},
            request_args={'model': 'gpt-test'},
        )
    )

    await client.create_completion(messages=[{'role': 'user', 'content': 'hi'}], tools=[_no_arg_tool()])

    assert len(calls) == 1
    assert calls[0]['tools'][0]['function']['parameters'] == {'type': 'object', 'properties': {}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('model', 'parallel_tool_calls', 'forwarded'),
    [
        ('api_google_gemini-3.1-pro-preview', False, None),
        ('api_google_gemini-3.1-pro-preview', True, True),
        ('gpt-test', False, False),
    ],
)
async def test_litellm_drops_parallel_tool_calls_false_only_for_gemini(
    monkeypatch, model, parallel_tool_calls, forwarded
):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)

    client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'test', 'stream': False, 'max_retries': 1},
            request_args={'model': model},
        )
    )

    await client.create_completion(
        messages=[{'role': 'user', 'content': 'hi'}],
        tools=[_no_arg_tool(), _no_arg_tool()],
        parallel_tool_calls=parallel_tool_calls,
    )

    assert calls[0].get('parallel_tool_calls') == forwarded


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('model', 'forwarded'),
    [
        ('gemini/gemini-4-argon', None),
        ('gpt-test', 'medium'),
    ],
)
async def test_litellm_drops_unsupported_verbosity_only_for_gemini(monkeypatch, model, forwarded):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'test', 'stream': False, 'max_retries': 1},
            request_args={'model': model},
        )
    )

    await client.create_completion(messages=[{'role': 'user', 'content': 'hi'}], verbosity='medium')

    assert calls[0].get('verbosity') == forwarded


@pytest.mark.asyncio
async def test_litellm_applies_enforced_args_after_legacy_override(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'test', 'max_retries': 1},
            request_args={'model': 'config-model'},
            enforced_request_args={
                'model': 'forced-model',
                'temperature': 0.2,
                'extra_body': {'vendor_mode': 'forced'},
            },
        )
    )

    await client.create_completion(
        messages=[{'role': 'user', 'content': 'hi'}],
        temperature=0.9,
        extra_body={'model': 'extra-model', 'vendor_mode': 'call'},
        _request_args_override={'model': 'override-model'},
    )

    assert calls[0]['model'] == 'forced-model'
    assert calls[0]['temperature'] == 0.2
    assert calls[0]['extra_body'] == {'vendor_mode': 'forced'}


@pytest.mark.asyncio
async def test_litellm_strip_cache_control_removes_all_markers(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'test', 'max_retries': 1, 'strip_cache_control': True},
            request_args={'model': 'gemini/gemini-test'},
        )
    )
    ephemeral = {'type': 'ephemeral'}
    messages = [
        {'role': 'system', 'content': [{'type': 'text', 'text': 'sys', 'cache_control': ephemeral}]},
        {'role': 'user', 'content': 'hi'},
        {'role': 'tool', 'tool_call_id': 't1', 'content': 'r', 'cache_control': ephemeral},
    ]
    tools = [{**_no_arg_tool(), 'cache_control': ephemeral}]

    await client.create_completion(messages=messages, tools=tools)

    assert 'strip_cache_control' not in calls[0]
    assert calls[0]['messages'][0]['content'] == [{'type': 'text', 'text': 'sys'}]
    assert calls[0]['messages'][1] == {'role': 'user', 'content': 'hi'}
    assert 'cache_control' not in calls[0]['messages'][2]
    assert 'cache_control' not in calls[0]['tools'][0]
    assert messages[0]['content'][0]['cache_control'] == ephemeral
    assert tools[0]['cache_control'] == ephemeral


@pytest.mark.asyncio
async def test_litellm_strip_cache_control_keeps_tools_none(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'test', 'max_retries': 1, 'strip_cache_control': True},
            request_args={'model': 'gemini/gemini-test'},
        )
    )

    await client.create_completion(
        messages=[
            {'role': 'user', 'content': [{'type': 'text', 'text': 'hi', 'cache_control': {'type': 'ephemeral'}}]}
        ],
        tools=None,
    )

    assert calls[0]['messages'][0]['content'] == [{'type': 'text', 'text': 'hi'}]
    assert calls[0].get('tools') is None


# ============== Mid-Conversation System Tests ==============


def test_has_mid_conversation_system_detects_system_after_user():
    messages = [
        {'role': 'system', 'content': 'prefix'},
        {'role': 'user', 'content': 'hi'},
        {'role': 'system', 'content': 'mid'},
    ]
    assert _has_mid_conversation_system(messages) is True


def test_has_mid_conversation_system_false_for_prefix_only():
    messages = [
        {'role': 'system', 'content': 'a'},
        {'role': 'system', 'content': 'b'},
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': 'hello'},
    ]
    assert _has_mid_conversation_system(messages) is False


def test_has_mid_conversation_system_false_for_empty():
    assert _has_mid_conversation_system([]) is False


def test_has_mid_conversation_system_false_for_no_system():
    messages = [{'role': 'user', 'content': 'hi'}]
    assert _has_mid_conversation_system(messages) is False


def test_model_supports_mid_conversation_system_false_for_unknown():
    assert _model_supports_mid_conversation_system('unknown-model-xyz', 'bedrock') is False


@pytest.mark.skipif(
    not _model_supports_mid_conversation_system('anthropic.claude-opus-4-8', 'bedrock'),
    reason='litellm model map does not include supports_mid_conversation_system (requires >=1.91.2)',
)
def test_model_supports_mid_conversation_system_for_bedrock_opus_4_8():
    assert _model_supports_mid_conversation_system('anthropic.claude-opus-4-8', 'bedrock') is True


def test_model_supports_mid_conversation_system_false_for_none():
    assert _model_supports_mid_conversation_system(None, 'bedrock') is False


def _transform_with_mid_system(messages, model='claude-opus-4-8', optional_params=None):
    """Run AnthropicConfig.transform_request with _preserve_mid_system=True."""
    import litellm as _litellm

    from nexus.extensions.openai_clients.litellm import _preserve_mid_system

    orig_modify = _litellm.modify_params
    _litellm.modify_params = True
    token = _preserve_mid_system.set(True)
    try:
        return AnthropicConfig().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params or {'max_tokens': 1024},
            litellm_params={},
            headers={},
        )
    finally:
        _preserve_mid_system.reset(token)
        _litellm.modify_params = orig_modify


def test_mid_system_tool_result_then_system_preserved():
    """tool_result → system: system must appear after tool_result, not merged."""
    messages = [
        {'role': 'system', 'content': 'Initial system prompt.'},
        {'role': 'user', 'content': 'Read the file.'},
        {
            'role': 'assistant',
            'content': 'I will read it.',
            'tool_calls': [
                {
                    'id': 'toolu_1',
                    'type': 'function',
                    'function': {'name': 'Read', 'arguments': '{"file_path":"README.md"}'},
                }
            ],
        },
        {'role': 'tool', 'tool_call_id': 'toolu_1', 'content': 'file contents'},
        {'role': 'system', 'content': 'Runtime instructions after the tool result.'},
    ]

    payload = _transform_with_mid_system(messages)

    assert payload['system'] == [{'type': 'text', 'text': 'Initial system prompt.'}]
    roles = [m['role'] for m in payload['messages']]
    assert roles == ['user', 'assistant', 'user', 'system']
    assert payload['messages'][-2]['content'][0]['type'] == 'tool_result'
    assert payload['messages'][-1] == {
        'role': 'system',
        'content': [{'type': 'text', 'text': 'Runtime instructions after the tool result.'}],
    }


def test_bedrock_invoke_final_payload_preserves_mid_system():
    """Validate the final Bedrock Invoke request body, not only the base Anthropic transform."""
    from nexus.extensions.openai_clients.litellm import _preserve_mid_system

    messages = [
        {'role': 'system', 'content': 'Top.'},
        {'role': 'user', 'content': 'Start.'},
        {'role': 'assistant', 'content': 'Working.'},
        {'role': 'system', 'content': 'Runtime reminder.'},
        {'role': 'user', 'content': 'Continue.'},
    ]

    token = _preserve_mid_system.set(True)
    try:
        payload = AmazonAnthropicClaudeConfig().transform_request(
            model='invoke/anthropic.claude-opus-4-8',
            messages=messages,
            optional_params={'max_tokens': 1024},
            litellm_params={},
            headers={},
        )
    finally:
        _preserve_mid_system.reset(token)

    assert payload['anthropic_version'] == 'bedrock-2023-05-31'
    assert payload['system'] == [{'type': 'text', 'text': 'Top.'}]
    assert [message['role'] for message in payload['messages']] == ['user', 'assistant', 'system', 'user']
    assert payload['messages'][2] == {
        'role': 'system',
        'content': [{'type': 'text', 'text': 'Runtime reminder.'}],
    }


def test_bedrock_final_payload_drops_only_billing_system_block():
    from nexus.extensions.openai_clients.litellm import _preserve_mid_system

    messages = [
        {
            'role': 'system',
            'content': [
                {'type': 'text', 'text': 'x-anthropic-billing-header: cc'},
                {
                    'type': 'text',
                    'text': 'Top policy.',
                    'cache_control': {'type': 'ephemeral'},
                },
            ],
        },
        {'role': 'user', 'content': 'Start.'},
        {'role': 'system', 'content': 'Runtime reminder.'},
    ]

    preserve_token = _preserve_mid_system.set(True)
    try:
        payload = AmazonAnthropicClaudeConfig().transform_request(
            model='invoke/anthropic.claude-opus-4-8',
            messages=messages,
            optional_params={'max_tokens': 1024},
            litellm_params={},
            headers={},
        )
    finally:
        _preserve_mid_system.reset(preserve_token)

    assert payload['system'] == [
        {
            'type': 'text',
            'text': 'Top policy.',
            'cache_control': {'type': 'ephemeral'},
        }
    ]
    assert [message['role'] for message in payload['messages']] == ['user', 'system']


def test_mid_system_before_tool_history_keeps_request_wide_tool_handling():
    """Tool history after a mid-system must still trigger Anthropic's request-wide tool handling."""
    messages = [
        {'role': 'system', 'content': 'Top.'},
        {'role': 'user', 'content': 'Before the reminder.'},
        {'role': 'system', 'content': 'Runtime reminder.'},
        {
            'role': 'assistant',
            'content': 'Calling a tool.',
            'tool_calls': [
                {
                    'id': 'toolu_after_mid',
                    'type': 'function',
                    'function': {'name': 'Read', 'arguments': '{}'},
                }
            ],
        },
        {'role': 'tool', 'tool_call_id': 'toolu_after_mid', 'content': 'result'},
    ]

    payload = _transform_with_mid_system(messages)

    assert payload.get('tools')
    assert [message['role'] for message in payload['messages']] == ['user', 'system', 'assistant', 'user']
    assert payload['messages'][2]['content'][-1]['type'] == 'tool_use'
    assert payload['messages'][3]['content'][0]['type'] == 'tool_result'


def test_mid_system_user_system_user_no_cross_boundary_merge():
    """user → system → user: the two user messages must NOT be merged across the system boundary."""
    messages = [
        {'role': 'system', 'content': 'Top.'},
        {'role': 'user', 'content': 'First user message.'},
        {'role': 'system', 'content': 'Mid-conversation instruction.'},
        {'role': 'user', 'content': 'Second user message.'},
    ]

    payload = _transform_with_mid_system(messages)

    assert payload['system'] == [{'type': 'text', 'text': 'Top.'}]
    roles = [m['role'] for m in payload['messages']]
    assert roles == ['user', 'system', 'user']
    assert payload['messages'][0]['content'] == [{'type': 'text', 'text': 'First user message.'}]
    assert payload['messages'][1] == {
        'role': 'system',
        'content': [{'type': 'text', 'text': 'Mid-conversation instruction.'}],
    }
    assert payload['messages'][2]['content'] == [{'type': 'text', 'text': 'Second user message.'}]


def test_mid_system_multiple_consecutive_system_messages():
    """Multiple consecutive mid-system messages preserved in order."""
    messages = [
        {'role': 'system', 'content': 'Top.'},
        {'role': 'user', 'content': 'Hello.'},
        {'role': 'system', 'content': 'Mid 1.'},
        {'role': 'system', 'content': 'Mid 2.'},
        {'role': 'assistant', 'content': 'Hi.'},
    ]

    payload = _transform_with_mid_system(messages)

    assert payload['system'] == [{'type': 'text', 'text': 'Top.'}]
    roles = [m['role'] for m in payload['messages']]
    assert roles == ['user', 'system', 'system', 'assistant']
    assert payload['messages'][1]['content'] == [{'type': 'text', 'text': 'Mid 1.'}]
    assert payload['messages'][2]['content'] == [{'type': 'text', 'text': 'Mid 2.'}]


def test_mid_system_no_mid_system_without_contextvar():
    """Without ContextVar, original litellm behavior: all system aggregated to top."""
    messages = [
        {'role': 'system', 'content': 'Top.'},
        {'role': 'user', 'content': 'Hi.'},
        {'role': 'assistant', 'content': 'Hello.'},
    ]

    payload = AnthropicConfig().transform_request(
        model='claude-opus-4-8',
        messages=messages,
        optional_params={'max_tokens': 1024},
        litellm_params={},
        headers={},
    )

    assert payload['system'] == [{'type': 'text', 'text': 'Top.'}]
    roles = [m['role'] for m in payload['messages']]
    assert roles == ['user', 'assistant']


def test_mid_system_prefix_system_still_goes_to_top():
    """Leading system messages are extracted to the top-level system field."""
    messages = [
        {'role': 'system', 'content': 'System A.'},
        {'role': 'system', 'content': 'System B.'},
        {'role': 'user', 'content': 'Go.'},
        {'role': 'system', 'content': 'Mid.'},
    ]

    payload = _transform_with_mid_system(messages)

    assert payload['system'] == [
        {'type': 'text', 'text': 'System A.'},
        {'type': 'text', 'text': 'System B.'},
    ]
    roles = [m['role'] for m in payload['messages']]
    assert roles == ['user', 'system']


@pytest.mark.asyncio
async def test_litellm_invoke_prefix_for_bedrock_mid_system(monkeypatch):
    """Bedrock + supported model + mid-system → model gets invoke/ prefix."""
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    monkeypatch.setattr(
        'nexus.extensions.openai_clients.litellm._model_supports_mid_conversation_system', lambda m, p: True
    )

    client = LiteLLMClient(
        OpenAIConfig(
            client_args={
                'api_key': 'k?provider=aws_third',
                'stream': False,
                'max_retries': 1,
            },
            request_args={'model': 'anthropic.claude-opus-4-8-20250714-v1:0'},
        )
    )

    messages = [
        {'role': 'system', 'content': 'Top.'},
        {'role': 'user', 'content': 'Go.'},
        {'role': 'system', 'content': 'Mid.'},
    ]

    await client.create_completion(messages=messages)

    assert len(calls) == 1
    assert 'invoke/' in calls[0]['model']
    assert calls[0]['custom_llm_provider'] == 'bedrock'


@pytest.mark.asyncio
async def test_litellm_bedrock_prefixed_model_keeps_provider_when_forcing_invoke(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    monkeypatch.setattr(
        'nexus.extensions.openai_clients.litellm._model_supports_mid_conversation_system', lambda m, p: True
    )

    client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'test', 'stream': False, 'max_retries': 1},
            request_args={'model': 'bedrock/anthropic.claude-opus-4-8'},
        )
    )

    await client.create_completion(
        messages=[
            {'role': 'system', 'content': 'Top.'},
            {'role': 'user', 'content': 'Go.'},
            {'role': 'system', 'content': 'Mid.'},
        ]
    )

    assert calls[0]['model'] == 'invoke/anthropic.claude-opus-4-8'
    assert calls[0]['custom_llm_provider'] == 'bedrock'


@pytest.mark.asyncio
async def test_litellm_explicit_converse_model_switches_to_invoke_for_mid_system(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    monkeypatch.setattr(
        'nexus.extensions.openai_clients.litellm._model_supports_mid_conversation_system', lambda m, p: True
    )

    client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'test', 'stream': False, 'max_retries': 1},
            request_args={'model': 'bedrock/converse/anthropic.claude-opus-4-8'},
        )
    )

    await client.create_completion(
        messages=[
            {'role': 'system', 'content': 'Top.'},
            {'role': 'user', 'content': 'Go.'},
            {'role': 'system', 'content': 'Mid.'},
        ]
    )

    assert calls[0]['model'] == 'invoke/anthropic.claude-opus-4-8'
    assert calls[0]['custom_llm_provider'] == 'bedrock'


@pytest.mark.asyncio
async def test_litellm_no_invoke_prefix_for_unsupported_model(monkeypatch):
    """Unsupported model: no route change, system aggregated by litellm default."""
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    monkeypatch.setattr(
        'nexus.extensions.openai_clients.litellm._model_supports_mid_conversation_system', lambda m, p: False
    )

    client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'k?provider=aws_third', 'stream': False, 'max_retries': 1},
            request_args={'model': 'some-unknown-model'},
        )
    )

    messages = [
        {'role': 'system', 'content': 'Top.'},
        {'role': 'user', 'content': 'Go.'},
        {'role': 'system', 'content': 'Mid.'},
    ]

    await client.create_completion(messages=messages)

    assert len(calls) == 1
    assert 'invoke/' not in calls[0]['model']


@pytest.mark.asyncio
async def test_litellm_no_invoke_prefix_without_mid_system(monkeypatch):
    """No mid-conversation system → no route change regardless of model support."""
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    monkeypatch.setattr(
        'nexus.extensions.openai_clients.litellm._model_supports_mid_conversation_system', lambda m, p: True
    )

    client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'k?provider=aws_third', 'stream': False, 'max_retries': 1},
            request_args={'model': 'anthropic.claude-opus-4-8-20250714-v1:0'},
        )
    )

    messages = [
        {'role': 'system', 'content': 'Top.'},
        {'role': 'user', 'content': 'Go.'},
    ]

    await client.create_completion(messages=messages)

    assert len(calls) == 1
    assert 'invoke/' not in calls[0]['model']


@pytest.mark.asyncio
async def test_litellm_mid_system_contextvar_isolation(monkeypatch):
    """Concurrent requests: one with mid-system, one without. ContextVar must isolate."""
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        await asyncio.sleep(0.01)
        return _ok_response()

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    monkeypatch.setattr(
        'nexus.extensions.openai_clients.litellm._model_supports_mid_conversation_system', lambda m, p: True
    )

    client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'k?provider=aws_third', 'stream': False, 'max_retries': 1},
            request_args={'model': 'anthropic.claude-opus-4-8-20250714-v1:0'},
        )
    )

    messages_with_mid = [
        {'role': 'system', 'content': 'Top.'},
        {'role': 'user', 'content': 'Go.'},
        {'role': 'system', 'content': 'Mid.'},
    ]
    messages_without_mid = [
        {'role': 'system', 'content': 'Top.'},
        {'role': 'user', 'content': 'Hi.'},
    ]

    await asyncio.gather(
        client.create_completion(messages=messages_with_mid),
        client.create_completion(messages=messages_without_mid),
    )

    assert len(calls) == 2
    has_invoke = any('invoke/' in c['model'] for c in calls)
    has_no_invoke = any('invoke/' not in c['model'] for c in calls)
    assert has_invoke
    assert has_no_invoke


# ============== Gemini mid-conversation system + trace cache_task_id ==============


def test_gemini_mid_conversation_system_demoted_to_user_in_place():
    from nexus.extensions.openai_clients.litellm import _demote_mid_conversation_system_for_gemini

    messages = [
        {'role': 'system', 'content': 'Stable system prompt.'},
        {'role': 'user', 'content': 'Run the task.'},
        {'role': 'assistant', 'content': 'ok'},
        {
            'role': 'system',
            'content': [
                {'type': 'text', 'text': '<total_tokens>1</total_tokens>', 'cache_control': {'type': 'ephemeral'}}
            ],
        },
    ]
    out = _demote_mid_conversation_system_for_gemini(messages, 'gemini/gemini-4-argon')
    assert [m['role'] for m in out] == ['system', 'user', 'assistant', 'user']
    assert out[0] is messages[0]
    assert out[3]['content'] == [{'type': 'text', 'text': '<total_tokens>1</total_tokens>'}]


def test_mid_conversation_system_untouched_for_non_marker_gemini():
    from nexus.extensions.openai_clients.litellm import _demote_mid_conversation_system_for_gemini

    messages = [
        {'role': 'system', 'content': 'a'},
        {'role': 'user', 'content': 'b'},
        {'role': 'system', 'content': 'c'},
    ]
    assert _demote_mid_conversation_system_for_gemini(messages, 'gemini/gemini-3-pro') is messages
    assert _demote_mid_conversation_system_for_gemini(messages, 'gpt-test') is messages


def _gemini_hop(
    *, native, token=None, content=None, reasoning=None, tool_calls=None, prompt=100, completion=10, cached=0
):
    fields = {'native_finish_reason': native}
    if token:
        fields['gemini_continuation_token'] = token
    return FakeModelResponse(
        {
            'id': 'gen',
            'created': 1,
            'model': 'gemini-4-argon',
            'object': 'chat.completion',
            'choices': [
                {
                    'index': 0,
                    'finish_reason': 'stop',
                    'provider_specific_fields': fields,
                    'message': {
                        'role': 'assistant',
                        'content': content,
                        'reasoning_content': reasoning,
                        'thinking_blocks': [{'type': 'thinking', 'thinking': reasoning}] if reasoning else None,
                        'tool_calls': tool_calls,
                    },
                }
            ],
            'usage': {
                'prompt_tokens': prompt,
                'completion_tokens': completion,
                'total_tokens': prompt + completion,
                'prompt_tokens_details': {'cached_tokens': cached},
                'completion_tokens_details': {'reasoning_tokens': completion if reasoning else 0},
            },
        }
    )


def _assert_minted(ids, issued):
    """The client handed out ids of its own: distinct, ``call_`` + 28 hex, none of them litellm's."""
    assert all(re.fullmatch(r'call_[0-9a-f]{28}', call_id) for call_id in ids), ids
    assert len(set(ids)) == len(ids)
    assert not set(ids) & set(issued)


def _gemini_client(**client_args):
    return LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'k', 'max_retries': 1, **client_args},
            request_args={'model': 'gemini/gemini-4-argon', 'max_tokens': 128000},
        )
    )


@pytest.mark.asyncio
async def test_gemini_continuation_loop_resumes_with_token_and_merges(monkeypatch):
    call = {
        'id': 'call_1__thought__sig',
        'type': 'function',
        'index': 0,
        'function': {'name': 'Bash', 'arguments': '{"command":"python3 big.py"}'},
    }
    hops = [
        _gemini_hop(
            native='CONTINUATION', token='tok-1', reasoning='plan part 1. ', prompt=1000, completion=500, cached=200
        ),
        _gemini_hop(
            native='CONTINUATION', token='tok-2', reasoning='plan part 2.', prompt=1500, completion=400, cached=1000
        ),
        _gemini_hop(native='STOP', tool_calls=[call], prompt=1900, completion=50, cached=1500),
    ]
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return hops[len(calls) - 1]

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    response = await _gemini_client().create_completion(messages=[{'role': 'user', 'content': 'hi'}])

    assert len(calls) == 3
    assert 'continuationToken' not in (calls[0].get('extra_body') or {})
    assert calls[1]['extra_body']['continuationToken'] == 'tok-1'
    assert calls[2]['extra_body']['continuationToken'] == 'tok-2'
    assert calls[1]['messages'] == calls[0]['messages']
    assert calls[1]['max_tokens'] == 128000

    choice = response.choices[0]
    assert choice.finish_reason == 'tool_calls'
    assert choice.message.reasoning_content == 'plan part 1. plan part 2.'
    _assert_minted([c.id for c in choice.message.tool_calls], ['call_1'])
    assert choice.message.tool_calls[0].provider_specific_fields['thought_signature'] == 'sig'
    # usage keeps per-request (context) semantics: first hop's prompt/cached, output summed over hops.
    assert response.usage.prompt_tokens == 1000
    assert response.usage.prompt_tokens_details.cached_tokens == 200
    assert response.usage.completion_tokens == 950
    assert response.usage.completion_tokens_details.reasoning_tokens == 900
    assert response.usage.total_tokens == 1950
    info = choice.provider_specific_fields['gemini_continuation']
    assert info['hops'] == 3
    assert info['unresolved'] is False
    assert info['context_prompt_tokens'] == 1000
    # What was billed: every hop resent the prompt.
    assert info['billed_prompt_tokens'] == 4400
    assert info['billed_cached_tokens'] == 2700
    assert [h['prompt_tokens'] for h in info['hop_usage']] == [1000, 1500, 1900]
    assert all(c['max_tokens'] == 128000 for c in calls)
    assert 'gemini_continuation_token' not in choice.provider_specific_fields


@pytest.mark.asyncio
async def test_gemini_max_tokens_hop_is_returned_as_length(monkeypatch):
    # max_tokens <= per-request cap: Gemini stops with MAX_TOKENS; nexus must not resume, the caller decides.
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _gemini_hop(native='MAX_TOKENS', token='tok-1', reasoning='still thinking', completion=100)

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    client = _gemini_client()
    client.config.request_args['max_tokens'] = 100
    response = await client.create_completion(messages=[{'role': 'user', 'content': 'hi'}])

    assert len(calls) == 1
    choice = response.choices[0]
    assert choice.finish_reason == 'length'
    assert choice.provider_specific_fields['native_finish_reason'] == 'MAX_TOKENS'
    assert 'gemini_continuation_token' not in choice.provider_specific_fields
    assert choice.message.reasoning_content == 'still thinking'
    assert response.usage.completion_tokens == 100


@pytest.mark.asyncio
async def test_gemini_continuation_then_max_tokens_stops_with_length(monkeypatch):
    calls = []
    hops = [
        _gemini_hop(native='CONTINUATION', token='tok-1', reasoning='a', completion=65536),
        _gemini_hop(native='MAX_TOKENS', token='tok-2', reasoning='b', completion=34464),
    ]

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return hops[len(calls) - 1]

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    client = _gemini_client()
    client.config.request_args['max_tokens'] = 100000
    response = await client.create_completion(messages=[{'role': 'user', 'content': 'hi'}])

    assert len(calls) == 2
    assert [c['max_tokens'] for c in calls] == [100000, 100000]
    assert calls[1]['extra_body']['continuationToken'] == 'tok-1'
    choice = response.choices[0]
    assert choice.finish_reason == 'length'
    assert choice.provider_specific_fields['native_finish_reason'] == 'MAX_TOKENS'
    assert choice.provider_specific_fields['gemini_continuation']['unresolved'] is True
    assert choice.message.reasoning_content == 'ab'
    assert response.usage.completion_tokens == 100000


@pytest.mark.asyncio
async def test_gemini_continuation_loop_in_streaming_mode(monkeypatch):
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import ModelResponseIterator

    class _Log:
        model_call_details: dict = {}  # noqa: RUF012
        optional_params: dict = {}  # noqa: RUF012

    parser = ModelResponseIterator(streaming_response=None, sync_stream=False, logging_obj=_Log())
    calls = []
    hop_payloads = [
        _gemini_hop(native='CONTINUATION', token='tok-1', reasoning='part 1. ', prompt=100, completion=50).payload,
        _gemini_hop(native='STOP', content='answer', prompt=150, completion=10).payload,
    ]
    raw_candidates = [
        [
            {'index': 0, 'continuationToken': 'ckpt'},
            {'index': 0, 'finishReason': 'CONTINUATION', 'continuationToken': 'tok-1'},
        ],
        [{'index': 0, 'continuationToken': 'ckpt-2'}, {'index': 0, 'finishReason': 'STOP'}],
    ]

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        i = len(calls) - 1
        assert kwargs['stream'] is True
        # Mimic litellm's Gemini stream parser seeing the raw chunks of this hop.
        for candidate in raw_candidates[i]:
            parser.chunk_parser({'candidates': [candidate]})
        return FakeAsyncStream(
            [
                FakeModelResponse(
                    {
                        'id': 'gen',
                        'choices': [{'index': 0, 'delta': {'content': 'x'}, 'finish_reason': 'stop'}],
                    }
                )
            ]
        )

    built = iter(hop_payloads)

    def fake_stream_chunk_builder(received_chunks, messages):
        payload = copy.deepcopy(next(built))
        payload['choices'][0].pop('provider_specific_fields', None)
        return FakeModelResponse(payload)

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    monkeypatch.setattr(
        'nexus.extensions.openai_clients.litellm.litellm.stream_chunk_builder', fake_stream_chunk_builder
    )
    response = await _gemini_client(stream=True).create_completion(messages=[{'role': 'user', 'content': 'hi'}])

    assert len(calls) == 2
    assert calls[1]['extra_body']['continuationToken'] == 'tok-1'
    choice = response.choices[0]
    assert choice.message.reasoning_content == 'part 1. '
    assert choice.message.content == 'answer'
    assert choice.provider_specific_fields['gemini_continuation']['hops'] == 2
    assert response.usage.prompt_tokens == 100
    assert response.usage.completion_tokens == 60
    assert choice.provider_specific_fields['gemini_continuation']['billed_prompt_tokens'] == 250


@pytest.mark.asyncio
async def test_gemini_continuation_hop_cap_and_opt_out(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _gemini_hop(native='CONTINUATION', token=f'tok-{len(calls)}', reasoning='r')

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    capped = await _gemini_client(gemini_continuation_max_hops=2).create_completion(
        messages=[{'role': 'user', 'content': 'hi'}]
    )
    assert len(calls) == 2
    assert capped.choices[0].finish_reason == 'length'

    calls.clear()
    plain = _gemini_client(gemini_continuation=False)
    assert 'gemini_continuation' not in plain.client_args
    await plain.create_completion(messages=[{'role': 'user', 'content': 'hi'}])
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_gemini_single_stop_hop_passes_through(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _gemini_hop(native='STOP', content='done')

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    response = await _gemini_client().create_completion(messages=[{'role': 'user', 'content': 'hi'}])
    assert len(calls) == 1
    assert response.choices[0].message.content == 'done'
    assert 'gemini_continuation' not in (response.choices[0].provider_specific_fields or {})


def test_gemini_transform_exposes_native_finish_and_token():
    import httpx

    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
    from litellm.types.utils import ModelResponse

    class _Log:
        model_call_details: dict = {}
        optional_params: dict = {}

        def post_call(self, **kwargs):
            pass

    raw = {
        'candidates': [
            {
                'content': {'role': 'model', 'parts': [{'text': 'thinking...', 'thought': True}]},
                'finishReason': 'CONTINUATION',
                'continuationToken': 'opaque',
            }
        ],
        'usageMetadata': {'promptTokenCount': 10, 'candidatesTokenCount': 5, 'totalTokenCount': 15},
    }
    result = VertexGeminiConfig()._transform_google_generate_content_to_openai_model_response(
        completion_response=raw,
        model_response=ModelResponse(),
        model='gemini-4-argon',
        logging_obj=_Log(),
        raw_response=httpx.Response(200, json=raw),
    )
    fields = result.choices[0].provider_specific_fields
    assert fields['native_finish_reason'] == 'CONTINUATION'
    assert fields['gemini_continuation_token'] == 'opaque'


@pytest.mark.asyncio
async def test_gemini_continuation_usage_is_context_size_for_proxies(monkeypatch):
    """Anthropic proxy input + cache_read must equal the real context, not the billed sum over hops."""
    from nexus.runtime.server.services.anthropic_proxy.converter import openai_to_anthropic

    hops = [
        _gemini_hop(
            native='CONTINUATION', token='tok-1', reasoning='a', prompt=200_000, completion=30_000, cached=150_000
        ),
        _gemini_hop(native='STOP', content='done', prompt=230_000, completion=500, cached=200_000),
    ]
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return hops[len(calls) - 1]

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    response = await _gemini_client().create_completion(messages=[{'role': 'user', 'content': 'hi'}])

    anthropic = openai_to_anthropic(response.model_dump(), 'gemini-4-argon')
    usage = anthropic.usage
    assert usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens == 200_000
    assert usage.output_tokens == 30_500


def test_billed_prompt_usage_drives_token_counter_and_langfuse():
    from nexus.core.types import billed_prompt_usage
    from nexus.extensions.openai_clients.litellm import _merge_gemini_continuation_hops

    merged = _merge_gemini_continuation_hops(
        [
            _gemini_hop(
                native='CONTINUATION', token='t', reasoning='a', prompt=1000, completion=500, cached=100
            ).payload,
            _gemini_hop(native='STOP', content='b', prompt=1500, completion=50, cached=900).payload,
        ],
        unresolved=False,
    )
    assert merged['usage']['prompt_tokens'] == 1000
    assert billed_prompt_usage(merged) == (2500, 1000)
    assert billed_prompt_usage({'choices': [{'provider_specific_fields': {}}], 'usage': {}}) is None

    from types import SimpleNamespace

    from nexus.trace.processors.token_counter import TokenCounterProcessor

    counter = TokenCounterProcessor()
    span = SimpleNamespace(name='openai_completion', attributes={'_client_key': 'llm', 'outputs': merged})
    counter.on_end(span, None)
    stats = counter.stat_dict['llm']
    assert stats['prompt_tokens'] == 2500
    assert stats['cached_prompt_tokens'] == 1000
    assert stats['completion_tokens'] == 550

    pytest.importorskip('langfuse')
    from nexus.trace.processors.langfuse_tracker import _convert_usage, _parse_response

    lf = _convert_usage(_parse_response(merged)[3])
    assert lf == {'input': 2500, 'output': 550, 'total': 3050, 'cached_tokens': 1000}


# ============== Gemini thought signature on tool call ids ==============

_SIG_A = 'AY89a1+first/signature=='
_SIG_B = 'AY89a1+second/signature=='


def _gemini_tool_call(call_id, name='Bash', index=None):
    call = {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': '{"command":"ls"}'}}
    if index is not None:
        call['index'] = index
    return call


def _gemini_tool_history(assistant_calls, tool_ids):
    return [
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': None, 'tool_calls': assistant_calls},
        *[{'role': 'tool', 'tool_call_id': tool_id, 'content': 'ok'} for tool_id in tool_ids],
    ]


def test_strip_gemini_thought_signature_moves_it_from_the_id_to_the_tool_call():
    from nexus.extensions.openai_clients.litellm import _strip_gemini_thought_signatures

    signatures: dict[str, str] = {}
    response = {
        'choices': [
            {
                'message': {
                    'role': 'assistant',
                    'tool_calls': [
                        _gemini_tool_call(f'call_a__thought__{_SIG_A}'),
                        _gemini_tool_call(f'call_b__thought__{_SIG_B}', name='Read'),
                    ],
                }
            }
        ]
    }

    _strip_gemini_thought_signatures(response, signatures)

    calls = response['choices'][0]['message']['tool_calls']
    assert [call['id'] for call in calls] == ['call_a', 'call_b']
    assert [call['provider_specific_fields']['thought_signature'] for call in calls] == [_SIG_A, _SIG_B]
    assert signatures == {'call_a': _SIG_A, 'call_b': _SIG_B}


def test_strip_gemini_thought_signature_leaves_unsuffixed_ids_alone():
    from nexus.extensions.openai_clients.litellm import _strip_gemini_thought_signatures

    signatures: dict[str, str] = {}
    response = {'choices': [{'message': {'role': 'assistant', 'tool_calls': [_gemini_tool_call('call_plain')]}}]}

    _strip_gemini_thought_signatures(response, signatures)

    call = response['choices'][0]['message']['tool_calls'][0]
    assert call['id'] == 'call_plain'
    assert 'provider_specific_fields' not in call
    assert signatures == {}


def test_strip_gemini_thought_signature_cache_is_bounded():
    from nexus.extensions.openai_clients.litellm import (
        GEMINI_THOUGHT_SIGNATURE_CACHE_SIZE,
        _strip_gemini_thought_signatures,
    )

    signatures: dict[str, str] = {}
    overflow = 5
    for i in range(GEMINI_THOUGHT_SIGNATURE_CACHE_SIZE + overflow):
        _strip_gemini_thought_signatures(
            {'choices': [{'message': {'tool_calls': [_gemini_tool_call(f'call_{i}__thought__{_SIG_A}')]}}]},
            signatures,
        )

    assert len(signatures) == GEMINI_THOUGHT_SIGNATURE_CACHE_SIZE
    assert f'call_{overflow - 1}' not in signatures
    assert f'call_{overflow}' in signatures


def test_restore_gemini_thought_signature_puts_it_back_without_touching_the_caller():
    from nexus.extensions.openai_clients.litellm import _restore_gemini_thought_signatures

    messages = _gemini_tool_history([_gemini_tool_call('call_a'), _gemini_tool_call('call_b', name='Read')], ['call_a'])

    restored = _restore_gemini_thought_signatures(messages, {'call_a': _SIG_A, 'call_b': _SIG_B})

    assert [call['provider_specific_fields']['thought_signature'] for call in restored[1]['tool_calls']] == [
        _SIG_A,
        _SIG_B,
    ]
    assert all('provider_specific_fields' not in call for call in messages[1]['tool_calls'])
    assert restored[0] is messages[0]
    assert restored[2] is messages[2]


def test_restore_gemini_thought_signature_is_a_noop_without_a_cached_signature():
    from nexus.extensions.openai_clients.litellm import _restore_gemini_thought_signatures

    messages = _gemini_tool_history([_gemini_tool_call('call_a')], ['call_a'])

    assert _restore_gemini_thought_signatures(messages, {}) is messages
    assert _restore_gemini_thought_signatures(messages, {'call_other': _SIG_A}) is messages


def test_restore_gemini_thought_signature_keeps_a_signature_the_caller_already_sent():
    from nexus.extensions.openai_clients.litellm import _restore_gemini_thought_signatures

    call = {**_gemini_tool_call('call_a'), 'provider_specific_fields': {'thought_signature': _SIG_B}}
    messages = _gemini_tool_history([call], ['call_a'])

    assert _restore_gemini_thought_signatures(messages, {'call_a': _SIG_A}) is messages


@pytest.mark.parametrize('n_calls', [1, 2])
def test_gemini_request_body_is_unchanged_by_moving_the_signature_off_the_id(n_calls):
    """Gemini must receive exactly what the suffixed id used to produce, signature placement included."""
    from litellm.llms.vertex_ai.gemini.transformation import _gemini_convert_messages_with_history

    names = ['Bash', 'Read'][:n_calls]
    sigs = [_SIG_A, _SIG_B][:n_calls]
    suffixed_ids = [f'call_{i}__thought__{sig}' for i, sig in enumerate(sigs)]
    clean_ids = [f'call_{i}' for i in range(n_calls)]

    from_suffixed = _gemini_convert_messages_with_history(
        messages=copy.deepcopy(
            _gemini_tool_history(
                [_gemini_tool_call(call_id, name=name) for call_id, name in zip(suffixed_ids, names, strict=True)],
                suffixed_ids,
            )
        ),
        model='gemini-4-argon',
    )
    from_clean = _gemini_convert_messages_with_history(
        messages=copy.deepcopy(
            _gemini_tool_history(
                [
                    {**_gemini_tool_call(call_id, name=name), 'provider_specific_fields': {'thought_signature': sig}}
                    for call_id, name, sig in zip(clean_ids, names, sigs, strict=True)
                ],
                clean_ids,
            )
        ),
        model='gemini-4-argon',
    )

    assert from_clean == from_suffixed
    assert [part['thoughtSignature'] for part in from_clean[1]['parts']] == sigs


@pytest.mark.asyncio
async def test_gemini_client_hands_out_minted_ids_and_replays_the_signature(monkeypatch):
    calls = []
    responses = [
        _gemini_hop(
            native='STOP',
            tool_calls=[
                _gemini_tool_call(f'call_a__thought__{_SIG_A}', index=0),
                _gemini_tool_call(f'call_b__thought__{_SIG_B}', name='Read', index=1),
            ],
        ),
        _gemini_hop(native='STOP', content='done'),
    ]

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return responses[len(calls) - 1]

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    client = _gemini_client()

    first = await client.create_completion(messages=[{'role': 'user', 'content': 'hi'}])
    ids = [call.id for call in first.choices[0].message.tool_calls]
    _assert_minted(ids, ['call_a', 'call_b'])
    assert [call.provider_specific_fields['thought_signature'] for call in first.choices[0].message.tool_calls] == [
        _SIG_A,
        _SIG_B,
    ]

    # A scaffold replays the ids it was given and keeps nothing else: the signature has to come back from the client.
    replay = _gemini_tool_history([_gemini_tool_call(ids[0]), _gemini_tool_call(ids[1], name='Read')], ids)
    await client.create_completion(messages=replay)

    sent = calls[1]['messages'][1]['tool_calls']
    assert [call['provider_specific_fields']['thought_signature'] for call in sent] == [_SIG_A, _SIG_B]
    assert [call['id'] for call in sent] == ids
    assert all('provider_specific_fields' not in call for call in replay[1]['tool_calls'])


@pytest.mark.asyncio
async def test_gemini_client_replays_the_signature_after_a_cached_turn(monkeypatch):
    from nexus.utils.cache import cache_context

    calls = []
    responses = [
        _gemini_hop(native='STOP', tool_calls=[_gemini_tool_call(f'call_a__thought__{_SIG_A}', index=0)]),
        _gemini_hop(native='STOP', content='done'),
    ]

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return responses[len(calls) - 1]

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    first_turn = [{'role': 'user', 'content': 'hi'}]
    recorded: dict = {}
    with cache_context(recorded):
        await _gemini_client().create_completion(messages=first_turn)

    # A rerun gets a fresh client: turn one comes back from the cache, so only turn two reaches the model.
    client = _gemini_client()
    with cache_context(recorded):
        replayed = await client.create_completion(messages=first_turn)
        call_id = replayed.choices[0].message.tool_calls[0].id
        await client.create_completion(messages=_gemini_tool_history([_gemini_tool_call(call_id)], [call_id]))

    assert len(calls) == 2
    assert calls[1]['messages'][1]['tool_calls'][0]['provider_specific_fields']['thought_signature'] == _SIG_A


@pytest.mark.asyncio
async def test_gemini_stream_events_carry_the_minted_tool_call_ids(monkeypatch):
    from nexus.trace import processor_context, span_context
    from nexus.trace.events import SpanUpdateEvent
    from nexus.trace.processors import SpanProcessor

    class _Recorder(SpanProcessor):
        def __init__(self):
            self.events = []

        def on_update(self, _span, event):
            if isinstance(event, SpanUpdateEvent):
                self.events.extend(event.events or [])

    raw_chunks = [
        {
            'id': 'gen',
            'choices': [
                {'index': 0, 'delta': {'tool_calls': [_gemini_tool_call(f'call_a__thought__{_SIG_A}', index=0)]}}
            ],
        },
        {'id': 'gen', 'choices': [{'index': 0, 'delta': {'tool_calls': [_gemini_tool_call('call_b', index=1)]}}]},
    ]
    built_from = []

    async def fake_acompletion(**kwargs):
        return FakeAsyncStream([FakeModelResponse(copy.deepcopy(chunk)) for chunk in raw_chunks])

    def fake_stream_chunk_builder(received_chunks, messages):
        built_from.extend(chunk.model_dump() for chunk in received_chunks)
        return _gemini_hop(
            native='STOP',
            tool_calls=[_gemini_tool_call(f'call_a__thought__{_SIG_A}', index=0), _gemini_tool_call('call_b', index=1)],
        )

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    monkeypatch.setattr(
        'nexus.extensions.openai_clients.litellm.litellm.stream_chunk_builder', fake_stream_chunk_builder
    )
    recorder = _Recorder()
    with processor_context([recorder]), span_context('openai_completion'):
        response = await _gemini_client(stream=True).create_completion(messages=[{'role': 'user', 'content': 'hi'}])

    streamed_ids = [
        event.attributes['llm.tool_call.id']
        for event in recorder.events
        if event.name == 'chat.completion.tool_call.delta'
    ]
    _assert_minted(streamed_ids, ['call_a', 'call_b'])
    assert [call.id for call in response.choices[0].message.tool_calls] == streamed_ids
    assert all('__thought__' not in repr(event.attributes) for event in recorder.events)
    # The chunks litellm assembles keep the signature: the final strip still has to read it off the id.
    assert built_from == raw_chunks


@pytest.mark.asyncio
async def test_gemini_continuation_merge_still_yields_minted_ids(monkeypatch):
    hops = [
        _gemini_hop(native='CONTINUATION', token='tok-1', reasoning='plan. '),
        _gemini_hop(native='STOP', tool_calls=[_gemini_tool_call(f'call_a__thought__{_SIG_A}', index=0)]),
    ]
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return hops[len(calls) - 1]

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    response = await _gemini_client().create_completion(messages=[{'role': 'user', 'content': 'hi'}])

    choice = response.choices[0]
    assert choice.finish_reason == 'tool_calls'
    _assert_minted([call.id for call in choice.message.tool_calls], ['call_a'])
    assert choice.message.tool_calls[0].provider_specific_fields['thought_signature'] == _SIG_A
    assert choice.provider_specific_fields['gemini_continuation']['hops'] == 2


def test_strip_gemini_thought_signature_mints_the_ids_the_stream_events_already_used():
    from nexus.extensions.openai_clients.litellm import _bare_gemini_stream_chunk, _strip_gemini_thought_signatures

    renamed: dict[str, str] = {}
    chunk = {'choices': [{'index': 0, 'delta': {'tool_calls': [_gemini_tool_call(f'call_a__thought__{_SIG_A}', index=0)]}}]}
    streamed_id = _bare_gemini_stream_chunk(chunk, renamed)['choices'][0]['delta']['tool_calls'][0]['id']

    signatures: dict[str, str] = {}
    # litellm reusing one id for two calls of the same response
    tool_calls = [_gemini_tool_call(f'call_a__thought__{_SIG_A}'), _gemini_tool_call('call_a', name='Read')]
    response = {'choices': [{'message': {'role': 'assistant', 'tool_calls': tool_calls}}]}
    _strip_gemini_thought_signatures(response, signatures, renamed)

    ids = [call['id'] for call in response['choices'][0]['message']['tool_calls']]
    assert ids[0] == streamed_id
    _assert_minted(ids, ['call_a'])
    assert signatures == {ids[0]: _SIG_A}
    # stream_chunk_builder still assembles litellm's ids, which is what the final strip maps from
    assert chunk['choices'][0]['delta']['tool_calls'][0]['id'] == f'call_a__thought__{_SIG_A}'


def _repeating_id_hops():
    """Two tool turns on which litellm issues the same id under different signatures, then a closing reply."""
    return [
        _gemini_hop(
            native='STOP',
            tool_calls=[
                _gemini_tool_call(f'call_dup__thought__{_SIG_A}', index=0),
                _gemini_tool_call('call_dup2', name='Read', index=1),
            ],
        ),
        _gemini_hop(native='STOP', tool_calls=[_gemini_tool_call(f'call_dup__thought__{_SIG_B}', index=0)]),
        _gemini_hop(native='STOP', content='done'),
    ]


async def _converse(client, rounds):
    """Drive a scaffold that replays only the ids it was handed: each reply and its tool results join the history."""
    messages = [{'role': 'user', 'content': 'hi'}]
    handed = []
    for _ in range(rounds):
        reply = (await client.create_completion(messages=copy.deepcopy(messages))).choices[0].message
        ids = [call.id for call in reply.tool_calls or []]
        handed.append(ids)
        if not ids:
            break
        messages.append({
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {'id': call.id, 'type': 'function', 'function': {'name': call.function.name, 'arguments': call.function.arguments}}
                for call in reply.tool_calls
            ],
        })
        messages.extend({'role': 'tool', 'tool_call_id': call_id, 'content': 'ok'} for call_id in ids)
    return handed


def _fake_gemini(monkeypatch, hops):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return hops[len(calls) - 1]

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    return calls


@pytest.mark.asyncio
async def test_gemini_client_keeps_signatures_apart_when_litellm_repeats_an_id(monkeypatch):
    from litellm.llms.vertex_ai.gemini.transformation import _gemini_convert_messages_with_history

    calls = _fake_gemini(monkeypatch, _repeating_id_hops())
    handed = await _converse(_gemini_client(), rounds=3)

    _assert_minted([*handed[0], *handed[1]], ['call_dup', 'call_dup2'])
    final = calls[2]['messages']
    assert final[1]['tool_calls'][0]['provider_specific_fields']['thought_signature'] == _SIG_A
    assert 'provider_specific_fields' not in final[1]['tool_calls'][1]
    assert final[4]['tool_calls'][0]['provider_specific_fields']['thought_signature'] == _SIG_B

    # What Gemini receives: each signature once, on its own call, and every result paired with its call by name
    contents = _gemini_convert_messages_with_history(messages=copy.deepcopy(final), model='gemini-4-argon')
    calls_sent = [part for content in contents if content['role'] == 'model' for part in content['parts']]
    assert [(part['function_call']['name'], part.get('thoughtSignature')) for part in calls_sent] == [
        ('Bash', _SIG_A),
        ('Read', None),
        ('Bash', _SIG_B),
    ]
    results = [part['function_response']['name'] for content in contents for part in content['parts'] if 'function_response' in part]
    assert results == ['Bash', 'Read', 'Bash']


@pytest.mark.asyncio
async def test_gemini_minted_ids_keep_every_request_a_prefix_of_the_next(monkeypatch):
    calls = _fake_gemini(monkeypatch, _repeating_id_hops())
    await _converse(_gemini_client(), rounds=3)

    sent = [call['messages'] for call in calls]
    assert len(sent) == 3
    for earlier, later in zip(sent, sent[1:], strict=False):
        assert later[: len(earlier)] == earlier


@pytest.mark.asyncio
async def test_gemini_cache_replay_hands_back_the_recorded_ids(monkeypatch):
    from nexus.utils.cache import cache_context

    calls = _fake_gemini(monkeypatch, _repeating_id_hops())
    recorded: dict = {}
    with cache_context(recorded):
        # the run dies before its third turn
        recorded_ids = await _converse(_gemini_client(), rounds=2)

    with cache_context(recorded):
        replayed_ids = await _converse(_gemini_client(), rounds=3)

    assert replayed_ids[:2] == recorded_ids
    # both recorded turns come back from the cache: only turn three reaches the model
    assert len(calls) == 3
    final = calls[2]['messages']
    assert final[1]['tool_calls'][0]['provider_specific_fields']['thought_signature'] == _SIG_A
    assert final[4]['tool_calls'][0]['provider_specific_fields']['thought_signature'] == _SIG_B


@pytest.mark.asyncio
async def test_non_gemini_tool_call_ids_are_left_as_issued(monkeypatch):
    _fake_gemini(monkeypatch, [_gemini_hop(native='STOP', tool_calls=[_gemini_tool_call('toolu_1', index=0)])])
    client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'k', 'max_retries': 1},
            request_args={'model': 'anthropic/claude-opus-4-8', 'max_tokens': 1024},
        )
    )

    response = await client.create_completion(messages=[{'role': 'user', 'content': 'hi'}])

    assert [call.id for call in response.choices[0].message.tool_calls] == ['toolu_1']


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [True, False])
async def test_bad_request_is_not_retried_with_default_max_retries(monkeypatch, stream):
    # A text-only model rejecting an image part fails identically on every retry.
    monkeypatch.setattr('nexus.utils.common.wait_exponential_jitter', lambda **_kwargs: wait_none())
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        raise BadRequestError(
            message="ZaiException - messages.content.type 参数非法，取值范围 ['text']",
            model='zai/glm-5.3',
            llm_provider='zai',
        )

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    client = LiteLLMClient(
        OpenAIConfig(client_args={'api_key': 'k', 'stream': stream}, request_args={'model': 'zai/glm-5.3'})
    )
    image = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,iVBORw0KGgo='}}

    with pytest.raises(BadRequestError, match='参数非法'):
        await client.create_completion(messages=[{'role': 'user', 'content': [image]}])

    assert len(calls) == 1
