import pytest

from pydantic import ValidationError

from nexus.extensions.harbor_bridge.bad_pattern_operator.legacy import run_legacy_deferred_monitors
from nexus.runtime.server.services.anthropic_proxy.converter import (
    anthropic_to_openai,
    convert_messages_to_openai,
    openai_to_anthropic,
)
from nexus.runtime.server.services.anthropic_proxy.models import (
    AnthropicServerTool,
    MessagesRequest,
    Tool,
)


def _messages_request_payload(tools: list[dict]):
    return {
        'model': 'claude-sonnet-4-6',
        'max_tokens': 1024,
        'messages': [{'role': 'user', 'content': 'Search the web'}],
        'tools': tools,
    }


def test_client_tool_still_requires_description_and_input_schema():
    with pytest.raises(ValidationError) as exc_info:
        MessagesRequest.model_validate(_messages_request_payload([{'name': 'lookup'}]))

    errors = {(tuple(error['loc']), error['type']) for error in exc_info.value.errors()}
    assert (('tools', 0, 'Tool', 'description'), 'missing') in errors
    assert (('tools', 0, 'Tool', 'input_schema'), 'missing') in errors


def test_client_tool_converts_to_openai_function_tool():
    request = MessagesRequest.model_validate(
        _messages_request_payload(
            [
                {
                    'name': 'lookup',
                    'description': 'Look up a record.',
                    'input_schema': {
                        'type': 'object',
                        'properties': {'query': {'type': 'string'}},
                        'required': ['query'],
                    },
                }
            ]
        )
    )

    assert isinstance(request.tools[0], Tool)
    openai_request = anthropic_to_openai(request)

    assert openai_request['tools'] == [
        {
            'type': 'function',
            'function': {
                'name': 'lookup',
                'description': 'Look up a record.',
                'parameters': {
                    'type': 'object',
                    'properties': {'query': {'type': 'string'}},
                    'required': ['query'],
                },
            },
        }
    ]


@pytest.mark.parametrize('tool_type', ['web_search_20250305', 'web_search_20260209'])
def test_web_search_server_tool_does_not_require_client_tool_schema(tool_type):
    request = MessagesRequest.model_validate(
        _messages_request_payload([{'type': tool_type, 'name': 'web_search', 'max_uses': 5}])
    )

    assert isinstance(request.tools[0], AnthropicServerTool)
    openai_request = anthropic_to_openai(request)

    assert openai_request['tools'] == [{'type': tool_type, 'name': 'web_search', 'max_uses': 5}]


def test_mixed_client_and_web_search_tools_preserve_each_tool_shape():
    request = MessagesRequest.model_validate(
        _messages_request_payload(
            [
                {'type': 'web_search_20250305', 'name': 'web_search', 'max_uses': 8},
                {
                    'name': 'lookup',
                    'description': 'Look up a record.',
                    'input_schema': {'type': 'object', 'properties': {}, 'required': []},
                },
            ]
        )
    )

    openai_request = anthropic_to_openai(request)

    assert openai_request['tools'][0] == {'type': 'web_search_20250305', 'name': 'web_search', 'max_uses': 8}
    assert openai_request['tools'][1]['type'] == 'function'
    assert openai_request['tools'][1]['function']['name'] == 'lookup'


def test_system_prompt_defaults_to_system_message():
    request = MessagesRequest.model_validate(
        {
            'model': 'claude-sonnet-4-6',
            'max_tokens': 1024,
            'system': 'Follow the policy.',
            'messages': [{'role': 'user', 'content': 'Search the web'}],
        }
    )

    openai_request = anthropic_to_openai(request)

    assert openai_request['messages'][0] == {'role': 'system', 'content': 'Follow the policy.'}


def test_system_prompt_blocks_preserve_boundaries_and_cache_control():
    request = MessagesRequest.model_validate(
        {
            'model': 'claude-opus-4-8',
            'max_tokens': 1024,
            'system': [
                {'type': 'text', 'text': 'x-anthropic-billing-header: cc'},
                {
                    'type': 'text',
                    'text': 'Follow the actual policy.',
                    'cache_control': {'type': 'ephemeral'},
                },
            ],
            'messages': [{'role': 'user', 'content': 'Run the task.'}],
        }
    )

    openai_request = anthropic_to_openai(request)

    assert openai_request['messages'][0] == {
        'role': 'system',
        'content': [
            {'type': 'text', 'text': 'x-anthropic-billing-header: cc'},
            {
                'type': 'text',
                'text': 'Follow the actual policy.',
                'cache_control': {'type': 'ephemeral'},
            },
        ],
    }


def test_system_message_after_user_is_accepted_and_preserved():
    request = MessagesRequest.model_validate(
        {
            'model': 'claude-sonnet-4-6',
            'max_tokens': 1024,
            'messages': [
                {'role': 'user', 'content': 'Run the task.'},
                {'role': 'system', 'content': 'Additional runtime instructions.'},
            ],
        }
    )

    openai_request = anthropic_to_openai(request)

    assert openai_request['messages'] == [
        {'role': 'user', 'content': 'Run the task.'},
        {'role': 'system', 'content': 'Additional runtime instructions.'},
    ]


def test_each_anthropic_response_has_a_fresh_message_id():
    response = {
        'id': 'cmpl-session-id',
        'choices': [
            {
                'message': {'role': 'assistant', 'content': 'First turn.'},
                'finish_reason': 'stop',
            }
        ],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1},
    }

    first = openai_to_anthropic(response, 'test-model')
    second = openai_to_anthropic(response, 'test-model')

    assert first.id.startswith('msg_')
    assert second.id.startswith('msg_')
    assert first.id != second.id


def test_tool_result_conversion_preserves_explicit_error_state():
    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'tool_result',
                    'tool_use_id': 'toolu_1',
                    'content': 'String to replace not found in file.',
                    'is_error': True,
                }
            ],
        }
    ]

    converted = convert_messages_to_openai(messages)

    assert converted == [
        {
            'role': 'tool',
            'tool_call_id': 'toolu_1',
            'content': [{'type': 'text', 'text': 'String to replace not found in file.'}],
            'is_error': True,
        }
    ]


def test_tool_result_conversion_does_not_invent_error_state():
    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'tool_result',
                    'tool_use_id': 'toolu_1',
                    'content': 'updated successfully',
                }
            ],
        }
    ]

    converted = convert_messages_to_openai(messages)

    assert 'is_error' not in converted[0]


def test_converted_hycode_edit_failures_reach_deferred_penalty_family():
    messages = []
    for index in range(2):
        messages.extend(
            [
                {
                    'role': 'assistant',
                    'content': [
                        {
                            'type': 'tool_use',
                            'id': f'toolu_{index}',
                            'name': 'Edit',
                            'input': {
                                'file_path': '/tmp/example.py',
                                'old_string': f'before-{index}',
                                'new_string': 'after',
                            },
                        }
                    ],
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'tool_result',
                            'tool_use_id': f'toolu_{index}',
                            'content': 'String to replace not found in file.',
                            'is_error': True,
                        }
                    ],
                },
            ]
        )

    converted = convert_messages_to_openai(messages)
    detection = run_legacy_deferred_monitors(
        converted,
        enabled_families={'str_replace_old_eq_new'},
        extra_fields={'messages': converted},
    )['str_replace_old_eq_new']

    assert detection['edit_call_failure_count'] == 1
    assert detection['edit_call_never_recovered_count'] == 1
    assert detection['old_eq_new_count'] == 2
