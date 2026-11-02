import json

from typing import Any

import pytest

import nexus.extensions.openai_clients.vertex_anthropic as vertex_anthropic_module

from nexus.extensions.openai_clients.vertex_anthropic import (
    _StreamAccumulator,
    _build_payload,
    _convert_response,
    apply_cache_control_claude_code,
    convert_messages_to_anthropic,
    convert_tools_to_anthropic,
)


def test_legacy_conversion_helpers_are_reexported():
    helper_names = {
        '_content_text',
        '_convert_assistant_message',
        '_convert_image_url',
        '_convert_text_part',
        '_convert_tool_choice',
        '_convert_tool_result_content',
        '_convert_user_content',
        '_default_max_tokens_for_model',
        '_normalize_output_config',
        '_normalize_thinking',
        '_parse_sse_event',
        '_read_stream_response',
        '_safe_json_loads',
        '_stop_reason_to_finish_reason',
        '_tag_message_cache_control',
        '_thinking_enabled',
        '_tool_call_to_dict',
    }

    assert helper_names <= set(vars(vertex_anthropic_module))


def test_explicit_computer_tool_is_flattened():
    converted = convert_tools_to_anthropic([
        {
            'type': 'computer_20251124',
            'function': {
                'name': 'computer',
                'parameters': {'display_width_px': 1280, 'display_height_px': 720},
            },
        }
    ])

    assert converted is not None
    assert converted[0]['type'] == 'computer_20251124'
    assert converted[0]['name'] == 'computer'
    assert converted[0]['display_width_px'] == 1280
    assert converted[0]['display_height_px'] == 720


def test_explicit_bash_tool_is_flattened():
    converted = convert_tools_to_anthropic([
        {
            'type': 'bash_20250124',
            'function': {
                'name': 'bash',
                'parameters': {},
            },
        }
    ])

    assert converted == [{'type': 'bash_20250124', 'name': 'bash'}]


@pytest.mark.parametrize('tool_type', ['computer_20241022', 'computer_20990101', 'computer', 'arbitrary_tool'])
def test_unsupported_non_function_tool_types_are_rejected(tool_type):
    with pytest.raises(ValueError, match=tool_type):
        convert_tools_to_anthropic([
            {
                'type': tool_type,
                'function': {
                    'name': 'computer',
                    'parameters': {'display_width_px': 1280, 'display_height_px': 720},
                },
            }
        ])


def test_function_named_computer_stays_custom():
    converted = convert_tools_to_anthropic([
        {
            'type': 'function',
            'function': {'name': 'computer', 'parameters': {'type': 'object'}},
        }
    ])

    assert converted is not None
    assert 'type' not in converted[0]
    assert converted[0]['name'] == 'computer'
    assert converted[0]['input_schema'] == {'type': 'object'}


def test_build_payload_preserves_signed_thinking_and_uses_output_config_effort():
    payload = _build_payload(
        messages=[
            {'role': 'system', 'content': 'You are coding.'},
            {'role': 'user', 'content': 'Inspect the repo.'},
            {
                'role': 'assistant',
                'content': 'I will inspect it.',
                'thinking_blocks': [{'type': 'thinking', 'thinking': 'Need file list.', 'signature': 'sig-1'}],
                'tool_calls': [
                    {
                        'id': 'toolu_1',
                        'type': 'function',
                        'function': {'name': 'bash', 'arguments': '{"cmd": "ls"}'},
                    }
                ],
            },
            {'role': 'tool', 'tool_call_id': 'toolu_1', 'content': 'README.md'},
        ],
        request_args={
            'model': 'claude-opus-4-8',
            'max_tokens': 16000,
            'thinking': {'type': 'adaptive', 'display': 'summarized'},
            'output_config': {'effort': 'xhigh'},
            'top_k': 64,
            'metadata': {'trace': 'gcp-smoke'},
            'tools': [
                {
                    'type': 'function',
                    'function': {
                        'name': 'bash',
                        'description': 'Run a command.',
                        'parameters': {'type': 'object', 'properties': {'cmd': {'type': 'string'}}},
                    },
                }
            ],
            'tool_choice': {'type': 'function', 'function': {'name': 'bash'}},
        },
    )

    assert payload['thinking'] == {'type': 'adaptive', 'display': 'summarized'}
    assert payload['output_config'] == {'effort': 'xhigh'}
    assert payload['stream'] is False
    assert payload['top_k'] == 64
    assert payload['metadata'] == {'trace': 'gcp-smoke'}
    assert 'reasoning_effort' not in payload
    assert 'tool_choice' not in payload
    assert payload['system'] == 'You are coding.'
    assert payload['tools'] == [
        {
            'name': 'bash',
            'description': 'Run a command.',
            'input_schema': {'type': 'object', 'properties': {'cmd': {'type': 'string'}}},
        }
    ]

    assistant = payload['messages'][1]
    assert assistant['role'] == 'assistant'
    assert assistant['content'][0] == {'type': 'thinking', 'thinking': 'Need file list.', 'signature': 'sig-1'}
    assert assistant['content'][2]['type'] == 'tool_use'
    assert assistant['content'][2]['input'] == {'cmd': 'ls'}

    tool_result = payload['messages'][2]
    assert tool_result == {
        'role': 'user',
        'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_1', 'content': 'README.md'}],
    }


def test_anthropic_conversion_preserves_valid_mid_conversation_system_messages_for_opus_4_8():
    payload = _build_payload(
        messages=[
            {'role': 'system', 'content': 'Stable system prompt.'},
            {'role': 'user', 'content': 'Run the task.'},
            {'role': 'developer', 'content': 'Runtime reminder.'},
            {'role': 'assistant', 'content': 'Continuing.'},
        ],
        request_args={'model': 'anthropic.claude-opus-4-8'},
    )

    assert payload['system'] == 'Stable system prompt.'
    assert payload['messages'] == [
        {'role': 'user', 'content': 'Run the task.'},
        {'role': 'system', 'content': 'Runtime reminder.'},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Continuing.'}]},
    ]


def test_assistant_message_replays_native_anthropic_content_verbatim():
    # Anthropic occasionally returns interleaved blocks such as [thinking, text, thinking, text].
    # The original ordering (and each thinking signature) must be replayed exactly, otherwise the
    # follow-up request is rejected.
    interleaved = [
        {'type': 'thinking', 'thinking': 'step one', 'signature': 'sig-1'},
        {'type': 'text', 'text': 'partial answer'},
        {'type': 'thinking', 'thinking': 'step two', 'signature': 'sig-2'},
        {'type': 'text', 'text': 'final answer'},
    ]
    _system, messages = convert_messages_to_anthropic([
        {
            'role': 'assistant',
            'content': 'partial answerfinal answer',
            'thinking_blocks': [
                {'type': 'thinking', 'thinking': 'step one', 'signature': 'sig-1'},
                {'type': 'thinking', 'thinking': 'step two', 'signature': 'sig-2'},
            ],
            'provider_specific_fields': {'anthropic_content': interleaved},
        }
    ])

    assistant = messages[0]
    assert assistant['role'] == 'assistant'
    # Order preserved verbatim, not collapsed into thinking-then-text.
    assert assistant['content'] == interleaved
    # A defensive copy is returned so mutating the payload does not corrupt stored history.
    assert assistant['content'] is not interleaved
    assert assistant['content'][0] is not interleaved[0]


def test_assistant_message_falls_back_to_reassembly_without_native_content():
    _system, messages = convert_messages_to_anthropic([
        {
            'role': 'assistant',
            'content': 'working',
            'thinking_blocks': [{'type': 'thinking', 'thinking': 'reasoning', 'signature': 'sig-a'}],
            'tool_calls': [{'id': 'toolu_1', 'function': {'name': 'bash', 'arguments': '{"cmd": "ls"}'}}],
        }
    ])

    assistant = messages[0]
    assert [block['type'] for block in assistant['content']] == ['thinking', 'text', 'tool_use']
    assert assistant['content'][0] == {'type': 'thinking', 'thinking': 'reasoning', 'signature': 'sig-a'}
    assert assistant['content'][1] == {'type': 'text', 'text': 'working'}
    assert assistant['content'][2]['input'] == {'cmd': 'ls'}


def test_assistant_message_ignores_empty_native_content_and_reassembles():
    _system, messages = convert_messages_to_anthropic([
        {
            'role': 'assistant',
            'content': 'hello',
            'provider_specific_fields': {'anthropic_content': []},
        }
    ])

    assert messages[0]['content'] == [{'type': 'text', 'text': 'hello'}]


def test_anthropic_conversion_preserves_mid_conversation_system_after_tool_result():
    system, messages = convert_messages_to_anthropic(
        [
            {'role': 'assistant', 'tool_calls': [{'id': 'toolu_1', 'function': {'name': 'bash', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'toolu_1', 'content': 'done'},
            {'role': 'system', 'content': 'The user added a requirement.'},
        ],
        model='claude-opus-4-8',
    )

    assert system is None
    assert [message['role'] for message in messages] == ['assistant', 'user', 'system']


def test_anthropic_conversion_uses_legacy_system_hoisting_for_unsupported_models():
    system, messages = convert_messages_to_anthropic(
        [
            {'role': 'system', 'content': 'Stable system prompt.'},
            {'role': 'user', 'content': 'Run the task.'},
            {'role': 'system', 'content': 'Runtime reminder.'},
        ],
        model='claude-sonnet-4-6',
    )

    assert system == 'Stable system prompt.\nRuntime reminder.'
    assert messages == [{'role': 'user', 'content': 'Run the task.'}]


def test_anthropic_conversion_uses_legacy_system_hoisting_for_invalid_placement():
    system, messages = convert_messages_to_anthropic(
        [
            {'role': 'system', 'content': 'Stable system prompt.'},
            {'role': 'user', 'content': 'Run the task.'},
            {'role': 'assistant', 'content': 'Working.'},
            {'role': 'system', 'content': 'Runtime reminder.'},
            {'role': 'user', 'content': 'Continue.'},
        ],
        model='claude-opus-4-8',
    )

    assert system == 'Stable system prompt.\nRuntime reminder.'
    assert [message['role'] for message in messages] == ['user', 'assistant', 'user']


def test_build_payload_preserves_cache_control_on_converted_tool_result():
    messages = apply_cache_control_claude_code([
        {'role': 'user', 'content': 'Inspect the repo.'},
        {'role': 'tool', 'tool_call_id': 'toolu_1', 'content': 'README.md'},
    ])

    payload = _build_payload(
        messages=messages,
        request_args={'model': 'claude-opus-4-8'},
    )

    assert payload['messages'][1]['content'][0] == {
        'type': 'tool_result',
        'tool_use_id': 'toolu_1',
        'content': 'README.md',
        'cache_control': {'type': 'ephemeral'},
    }


def test_build_payload_translates_reasoning_effort_to_output_config():
    payload = _build_payload(
        messages=[{'role': 'user', 'content': 'hi'}],
        request_args={
            'model': 'claude-opus-4-8',
            'thinking': {'type': 'adaptive'},
            'reasoning_effort': 'xhigh',
        },
    )

    assert payload['output_config'] == {'effort': 'xhigh'}
    assert 'reasoning_effort' not in payload


def test_build_payload_defaults_opus48_to_128k_output_tokens():
    payload = _build_payload(
        messages=[{'role': 'user', 'content': 'hi'}],
        request_args={'model': 'claude-opus-4-8'},
    )

    assert payload['max_tokens'] == 128000


def test_build_payload_keeps_default_for_unknown_models():
    payload = _build_payload(
        messages=[{'role': 'user', 'content': 'hi'}],
        request_args={'model': 'some-other-model'},
    )

    assert payload['max_tokens'] == 64000


def test_build_payload_can_enable_streaming():
    payload = _build_payload(
        messages=[{'role': 'user', 'content': 'hi'}],
        request_args={'model': 'claude-opus-4-8', 'stream': True},
    )

    assert payload['stream'] is True


def test_convert_response_preserves_signed_thinking_tool_calls_and_usage():
    completion = _convert_response(
        {
            'id': 'msg_123',
            'model': 'claude-opus-4-8',
            'content': [
                {'type': 'thinking', 'thinking': 'Use a tool.', 'signature': 'sig-2'},
                {'type': 'text', 'text': 'Checking.'},
                {'type': 'tool_use', 'id': 'toolu_2', 'name': 'bash', 'input': {'cmd': 'pwd'}},
            ],
            'stop_reason': 'tool_use',
            'usage': {
                'input_tokens': 10,
                'output_tokens': 5,
                'cache_creation_input_tokens': 3,
                'cache_read_input_tokens': 7,
                'output_tokens_details': {'thinking_tokens': 2},
            },
        },
        'claude-opus-4-8',
    )

    assert completion.id == 'msg_123'
    assert completion.choices[0].finish_reason == 'tool_calls'
    message = completion.choices[0].message
    assert message.content == 'Checking.'
    assert message.thinking_blocks is not None
    assert message.thinking_blocks[0].signature == 'sig-2'
    assert message.tool_calls is not None
    assert message.tool_calls[0].id == 'toolu_2'
    assert message.tool_calls[0].function.name == 'bash'
    assert json.loads(message.tool_calls[0].function.arguments) == {'cmd': 'pwd'}
    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 20
    assert completion.usage.completion_tokens == 5
    assert completion.usage.total_tokens == 25
    assert completion.usage.prompt_tokens_details is not None
    assert completion.usage.prompt_tokens_details.cache_creation_tokens == 3
    assert completion.usage.prompt_tokens_details.cached_tokens == 7
    assert completion.usage.cache_creation_input_tokens == 3
    assert completion.usage.cache_read_input_tokens == 7
    assert completion.usage.completion_tokens_details is not None
    assert completion.usage.completion_tokens_details.reasoning_tokens == 2


def test_convert_response_retains_successful_native_response_envelope():
    response = {
        'id': 'msg_native_success',
        'model': 'claude-opus-4-8',
        'type': 'message',
        'role': 'assistant',
        'content': [{'type': 'text', 'text': 'Native answer.'}],
        'stop_reason': 'end_turn',
        'stop_sequence': None,
        'usage': {'input_tokens': 10, 'output_tokens': 3},
    }

    completion = _convert_response(response, 'claude-opus-4-8')

    assert completion.choices[0].message.content == 'Native answer.'
    assert completion.choices[0].finish_reason == 'stop'
    assert completion.provider_specific_fields == {
        'native_response': {'protocol': 'anthropic', 'body': response},
    }


def test_convert_response_retains_empty_refusal_native_response_envelope():
    response = {
        'id': 'msg_native_refusal',
        'model': 'claude-opus-4-8',
        'type': 'message',
        'role': 'assistant',
        'content': [],
        'stop_reason': 'refusal',
        'stop_details': {'type': 'refusal', 'category': 'cyber'},
        'usage': {'input_tokens': 10, 'output_tokens': 2},
    }

    completion = _convert_response(response, 'claude-opus-4-8')

    assert completion.choices[0].message.content is None
    assert completion.choices[0].message.refusal == 'The model refused to fulfill the request.'
    assert completion.choices[0].finish_reason == 'stop'
    assert completion.provider_specific_fields == {
        'native_response': {'protocol': 'anthropic', 'body': response},
    }


def test_convert_response_excludes_raw_sse_events_without_mutating_source_response():
    response = {
        'id': 'msg_native_stream',
        'content': [{'type': 'text', 'text': 'Streamed answer.'}],
        'stop_reason': 'end_turn',
        'usage': {'input_tokens': 4, 'output_tokens': 2},
        'raw_sse_events': ['data: {"type":"message_start"}'],
    }
    expected_source = {
        'id': 'msg_native_stream',
        'content': [{'type': 'text', 'text': 'Streamed answer.'}],
        'stop_reason': 'end_turn',
        'usage': {'input_tokens': 4, 'output_tokens': 2},
        'raw_sse_events': ['data: {"type":"message_start"}'],
    }

    completion = _convert_response(response, 'claude-opus-4-8')

    assert response == expected_source
    assert completion.provider_specific_fields == {
        'native_response': {
            'protocol': 'anthropic',
            'body': {
                'id': 'msg_native_stream',
                'content': [{'type': 'text', 'text': 'Streamed answer.'}],
                'stop_reason': 'end_turn',
                'usage': {'input_tokens': 4, 'output_tokens': 2},
            },
        },
    }


def test_convert_response_maps_refusal_stop_details_to_openai_refusal():
    stop_details = {
        'type': 'refusal',
        'category': 'cyber',
        'explanation': 'I cannot help with that request.',
    }

    completion = _convert_response(
        {
            'id': 'msg_refusal',
            'model': 'claude-opus-4-8',
            'content': [],
            'stop_reason': 'refusal',
            'stop_details': stop_details,
            'usage': {'input_tokens': 10, 'output_tokens': 2},
        },
        'claude-opus-4-8',
    )

    choice = completion.choices[0]
    assert choice.finish_reason == 'stop'
    assert choice.message.content is None
    assert choice.message.refusal == 'I cannot help with that request.'
    assert choice.provider_specific_fields == {
        'stop_reason': 'refusal',
        'stop_sequence': None,
        'stop_details': stop_details,
    }


def test_convert_response_uses_nonempty_fallback_for_refusal_without_explanation():
    completion = _convert_response(
        {
            'id': 'msg_refusal',
            'content': [],
            'stop_reason': 'refusal',
            'stop_details': {'type': 'refusal', 'category': 'unknown'},
            'usage': {},
        },
        'claude-opus-4-8',
    )

    assert completion.choices[0].message.refusal == 'The model refused to fulfill the request.'


@pytest.mark.parametrize('tool_input', [['not', 'an', 'object'], 'null'])
def test_convert_response_rejects_non_object_tool_input(tool_input: Any) -> None:
    with pytest.raises(ValueError, match='Tool arguments'):
        _convert_response(
            {
                'id': 'msg_invalid_tool',
                'content': [{'type': 'tool_use', 'id': 'toolu_invalid', 'name': 'bash', 'input': tool_input}],
                'stop_reason': 'tool_use',
                'usage': {},
            },
            'claude-opus-4-8',
        )


def test_stream_accumulator_reconstructs_thinking_text_and_tool_input():
    accumulator = _StreamAccumulator()
    for event in [
        {
            'type': 'message_start',
            'message': {
                'id': 'msg_stream',
                'model': 'claude-opus-4-8',
                'type': 'message',
                'role': 'assistant',
                'content': [],
                'usage': {'input_tokens': 10, 'output_tokens': 1},
            },
        },
        {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'thinking', 'thinking': ''}},
        {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'thinking_delta', 'thinking': 'Plan.'}},
        {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'signature_delta', 'signature': 'sig'}},
        {'type': 'content_block_stop', 'index': 0},
        {'type': 'content_block_start', 'index': 1, 'content_block': {'type': 'text', 'text': ''}},
        {'type': 'content_block_delta', 'index': 1, 'delta': {'type': 'text_delta', 'text': 'Editing.'}},
        {'type': 'content_block_stop', 'index': 1},
        {
            'type': 'content_block_start',
            'index': 2,
            'content_block': {'type': 'tool_use', 'id': 'toolu_1', 'name': 'edit', 'input': {}},
        },
        {
            'type': 'content_block_delta',
            'index': 2,
            'delta': {'type': 'input_json_delta', 'partial_json': '{"old_string":'},
        },
        {
            'type': 'content_block_delta',
            'index': 2,
            'delta': {'type': 'input_json_delta', 'partial_json': ' "a", "new_string": "b"}'},
        },
        {'type': 'content_block_stop', 'index': 2},
        {
            'type': 'message_delta',
            'delta': {'stop_reason': 'tool_use', 'stop_sequence': None},
            'usage': {'output_tokens': 20, 'output_tokens_details': {'thinking_tokens': 5}},
        },
        {'type': 'message_stop'},
    ]:
        accumulator.handle(event)

    completion = _convert_response(accumulator.build('claude-opus-4-8'), 'claude-opus-4-8')

    message = completion.choices[0].message
    assert message.content == 'Editing.'
    assert message.thinking_blocks is not None
    assert message.thinking_blocks[0].thinking == 'Plan.'
    assert message.thinking_blocks[0].signature == 'sig'
    assert message.tool_calls is not None
    assert json.loads(message.tool_calls[0].function.arguments) == {'old_string': 'a', 'new_string': 'b'}
    assert completion.choices[0].finish_reason == 'tool_calls'
    assert completion.usage is not None
    assert completion.usage.completion_tokens_details is not None
    assert completion.usage.completion_tokens_details.reasoning_tokens == 5


def test_stream_accumulator_preserves_refusal_stop_details():
    accumulator = _StreamAccumulator()
    accumulator.handle({
        'type': 'message_start',
        'message': {
            'id': 'msg_stream_refusal',
            'model': 'claude-opus-4-8',
            'content': [],
            'usage': {'input_tokens': 4, 'output_tokens': 0},
        },
    })
    accumulator.handle({
        'type': 'message_delta',
        'delta': {
            'stop_reason': 'refusal',
            'stop_details': {'type': 'refusal', 'explanation': 'Request refused.'},
        },
        'usage': {'output_tokens': 2},
    })
    accumulator.handle({'type': 'message_stop'})

    completion = _convert_response(accumulator.build('claude-opus-4-8'), 'claude-opus-4-8')

    assert completion.choices[0].message.refusal == 'Request refused.'
    assert completion.choices[0].provider_specific_fields == {
        'stop_reason': 'refusal',
        'stop_sequence': None,
        'stop_details': {'type': 'refusal', 'explanation': 'Request refused.'},
    }


@pytest.mark.parametrize('arguments', ['{invalid', '[]', 'null'])
def test_legacy_tool_argument_parser_rejects_malformed_or_non_object_json(arguments: str) -> None:
    with pytest.raises(ValueError, match='Tool arguments'):
        vertex_anthropic_module._safe_json_loads(arguments)


@pytest.mark.parametrize('partial_json', ['{invalid', '[]', 'null'])
def test_stream_accumulator_rejects_malformed_or_non_object_tool_input(partial_json: str) -> None:
    accumulator = _StreamAccumulator()
    accumulator.handle({
        'type': 'content_block_start',
        'index': 0,
        'content_block': {'type': 'tool_use', 'id': 'toolu_1', 'name': 'edit', 'input': {}},
    })
    accumulator.handle({
        'type': 'content_block_delta',
        'index': 0,
        'delta': {'type': 'input_json_delta', 'partial_json': partial_json},
    })

    with pytest.raises(ValueError, match='tool block'):
        accumulator.handle({'type': 'content_block_stop', 'index': 0})


@pytest.mark.parametrize('tool_input', [[], 'null'])
def test_stream_accumulator_rejects_non_object_initial_tool_input(tool_input: Any) -> None:
    accumulator = _StreamAccumulator()
    accumulator.handle({
        'type': 'content_block_start',
        'index': 0,
        'content_block': {'type': 'tool_use', 'id': 'toolu_1', 'name': 'edit', 'input': tool_input},
    })

    with pytest.raises(ValueError, match='tool block'):
        accumulator.handle({'type': 'content_block_stop', 'index': 0})
