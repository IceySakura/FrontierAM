"""OSWorld bridge anthropic tools regressions."""

from __future__ import annotations

import base64
import json
import logging

import pytest

from pydantic import ValidationError

from nexus.core.types import ExtendedChatCompletion, ExtendedChoice, ExtendedMessage, ExtendedUsage
from nexus.runtime.bridge.adapters.ags_osworld import (
    _convert_document_to_openai_parts,
    _redact_data_urls,
    anthropic_to_openai,
    openai_to_anthropic,
)
from nexus.runtime.server.services.anthropic_proxy.models import AnthropicServerTool, MessagesRequest, Tool


def _messages_request_payload(tools: list[dict]):
    return {
        'model': 'claude-sonnet-4-6',
        'max_tokens': 1024,
        'messages': [{'role': 'user', 'content': 'Search the web'}],
        'tools': tools,
    }


@pytest.mark.parametrize('string_arguments', [False, True])
@pytest.mark.parametrize('tool_count', [1, 2])
def test_openai_tool_calls_preserve_names_ids_and_arguments(string_arguments, tool_count) -> None:
    expected = [
        {'type': 'tool_use', 'id': 'call-bash', 'name': 'Bash', 'input': {'command': 'echo 你好'}},
        {'type': 'tool_use', 'id': 'call-read', 'name': 'Read', 'input': {'file_path': '/tmp/example.txt'}},
    ][:tool_count]
    calls = [
        {
            'id': tool['id'],
            'type': 'function',
            'function': {
                'name': tool['name'],
                'arguments': json.dumps(tool['input']) if string_arguments else tool['input'],
            },
        }
        for tool in expected
    ]
    response = {
        'choices': [
            {
                'message': {'role': 'assistant', 'content': 'Working.', 'tool_calls': calls},
                'finish_reason': 'tool_calls',
            }
        ],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
    }

    result = openai_to_anthropic(response, model='test-model')

    assert result.content[0].model_dump() == {'type': 'text', 'text': 'Working.'}
    assert [block.model_dump() for block in result.content[1:]] == expected
    assert result.stop_reason == 'tool_use'


def test_conversion_log_redaction_drops_base64_payloads() -> None:
    payload = 'sensitive-base64-payload' * 100
    value = {
        'content': [
            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{payload}'}},
            {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': payload}},
            {'type': 'document', 'source': {'type': 'base64', 'media_type': 'application/pdf', 'data': payload}},
        ]
    }

    redacted = _redact_data_urls(value)

    summary = redacted['content'][0]['image_url']['url']
    assert payload not in summary
    assert str(len(payload)) in summary
    source_summary = redacted['content'][1]['source']['data']
    assert payload not in source_summary
    assert str(len(payload)) in source_summary
    document_summary = redacted['content'][2]['source']['data']
    assert payload not in document_summary
    assert str(len(payload)) in document_summary


def test_pdf_document_block_is_accepted_and_converted_once_and_redacted(monkeypatch, caplog) -> None:
    rendered = [
        {'type': 'text', 'text': '[PDF page 1 of 1]'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,rendered'}},
    ]
    render_calls = []

    def fake_render(document):
        render_calls.append(document)
        return rendered

    monkeypatch.setattr(
        'nexus.runtime.bridge.adapters.ags_osworld._convert_document_to_openai_parts',
        fake_render,
    )
    encoded = base64.b64encode(b'%PDF-test').decode()
    request = MessagesRequest.model_validate(
        {
            'model': 'claude-opus-4-8',
            'max_tokens': 1024,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'Inspect this document.'},
                        {
                            'type': 'document',
                            'source': {
                                'type': 'base64',
                                'media_type': 'application/pdf',
                                'data': encoded,
                            },
                        },
                    ],
                }
            ],
        }
    )

    with caplog.at_level(logging.DEBUG, logger='nexus.osworld.anthropic_proxy.converter'):
        converted = anthropic_to_openai(request, document_mode='images')

    assert converted['messages'] == [
        {
            'role': 'user',
            'content': [{'type': 'text', 'text': 'Inspect this document.'}, *rendered],
        }
    ]
    assert len(render_calls) == 1
    assert encoded not in caplog.text


def test_pdf_document_inside_tool_result_is_converted(monkeypatch) -> None:
    rendered = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,rendered'}}]
    monkeypatch.setattr(
        'nexus.runtime.bridge.adapters.ags_osworld._convert_document_to_openai_parts',
        lambda _document: rendered,
    )
    request = MessagesRequest.model_validate(
        {
            'model': 'claude-opus-4-8',
            'max_tokens': 1024,
            'messages': [
                {
                    'role': 'assistant',
                    'content': [{'type': 'tool_use', 'id': 'read-pdf', 'name': 'Read', 'input': {}}],
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'tool_result',
                            'tool_use_id': 'read-pdf',
                            'content': [
                                {'type': 'text', 'text': 'PDF file read'},
                                {
                                    'type': 'document',
                                    'source': {
                                        'type': 'base64',
                                        'media_type': 'application/pdf',
                                        'data': base64.b64encode(b'%PDF-test').decode(),
                                    },
                                },
                            ],
                        }
                    ],
                },
            ],
        }
    )

    converted = anthropic_to_openai(request, document_mode='images')

    assert converted['messages'][1] == {
        'role': 'tool',
        'tool_call_id': 'read-pdf',
        'content': [{'type': 'text', 'text': 'PDF file read'}, *rendered],
    }


def test_pdf_document_invalid_base64_returns_actionable_fallback() -> None:
    parts = _convert_document_to_openai_parts(
        {
            'type': 'document',
            'source': {'type': 'base64', 'media_type': 'application/pdf', 'data': 'not-base64'},
        }
    )

    assert len(parts) == 1
    assert parts[0]['type'] == 'text'
    assert 'pdftoppm' in parts[0]['text']


def test_pdf_document_oversize_returns_actionable_fallback(monkeypatch) -> None:
    monkeypatch.setattr('nexus.runtime.bridge.adapters.ags_osworld._PDF_MAX_BYTES', 8)

    parts = _convert_document_to_openai_parts(
        {
            'type': 'document',
            'source': {
                'type': 'base64',
                'media_type': 'application/pdf',
                'data': base64.b64encode(b'123456789').decode(),
            },
        }
    )

    assert parts[0]['type'] == 'text'
    assert 'exceeds' in parts[0]['text']


def test_pdf_document_parse_failure_returns_actionable_fallback() -> None:
    parts = _convert_document_to_openai_parts(
        {
            'type': 'document',
            'source': {
                'type': 'base64',
                'media_type': 'application/pdf',
                'data': base64.b64encode(b'not a PDF').decode(),
            },
        }
    )

    assert parts[0]['type'] == 'text'
    assert 'pdftoppm' in parts[0]['text']


def test_pdf_document_is_rendered_to_one_image_per_page() -> None:
    pymupdf = pytest.importorskip('pymupdf')
    pdf = pymupdf.open()
    for page_number in (1, 2):
        page = pdf.new_page(width=200, height=100)
        page.insert_text((20, 50), f'Page {page_number}')
    encoded = base64.b64encode(pdf.tobytes()).decode()
    pdf.close()

    parts = _convert_document_to_openai_parts(
        {
            'type': 'document',
            'source': {'type': 'base64', 'media_type': 'application/pdf', 'data': encoded},
        }
    )

    assert [part['type'] for part in parts] == ['text', 'image_url', 'text', 'image_url']
    for part in parts[1::2]:
        url = part['image_url']['url']
        assert url.startswith('data:image/png;base64,')
        assert base64.b64decode(url.split(',', 1)[1]).startswith(b'\x89PNG\r\n\x1a\n')


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
    openai_request = anthropic_to_openai(request, document_mode='images')

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


def test_native_computer_tool_is_rejected_at_chat_conversion_not_global_validation():
    request = MessagesRequest.model_validate(
        _messages_request_payload(
            [
                {
                    'type': 'computer_20251124',
                    'name': 'computer',
                    'description': 'Control the desktop.',
                    'input_schema': {'type': 'object', 'properties': {}, 'required': []},
                    'display_width_px': 1920,
                    'display_height_px': 1080,
                }
            ]
        )
    )
    with pytest.raises(ValueError, match='computer_20251124'):
        anthropic_to_openai(request)


def test_regular_computer_function_tool_remains_supported():
    request = MessagesRequest.model_validate(
        _messages_request_payload(
            [
                {
                    'name': 'computer',
                    'description': 'Control the desktop.',
                    'input_schema': {'type': 'object', 'properties': {}, 'required': []},
                }
            ]
        )
    )

    openai_request = anthropic_to_openai(request, document_mode='images')

    assert openai_request['tools'][0]['function']['name'] == 'computer'


def test_openai_to_anthropic_prefers_top_level_extended_cache_usage():
    completion = ExtendedChatCompletion(
        id='msg_1',
        created=1,
        model='claude-fable-5',
        choices=[
            ExtendedChoice(
                finish_reason='stop',
                index=0,
                message=ExtendedMessage(role='assistant', content='ok'),
            )
        ],
        usage=ExtendedUsage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_creation_input_tokens=30,
            cache_read_input_tokens=70,
        ),
    )

    response = openai_to_anthropic(completion, 'claude-fable-5')

    assert response.usage.input_tokens == 0
    assert response.usage.output_tokens == 20
    assert response.usage.cache_creation_input_tokens == 30
    assert response.usage.cache_read_input_tokens == 70


@pytest.mark.parametrize('arguments', ['{invalid', '[]', 'null'])
def test_openai_to_anthropic_rejects_malformed_or_non_object_tool_arguments(arguments: str):
    completion = ExtendedChatCompletion.model_validate(
        {
            'id': 'msg_tool',
            'created': 1,
            'model': 'claude-fable-5',
            'choices': [
                {
                    'finish_reason': 'tool_calls',
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': None,
                        'tool_calls': [
                            {
                                'id': 'toolu_1',
                                'type': 'function',
                                'function': {'name': 'computer', 'arguments': arguments},
                            }
                        ],
                    },
                }
            ],
        }
    )

    with pytest.raises(ValueError, match='tool arguments'):
        openai_to_anthropic(completion, 'claude-fable-5')


@pytest.mark.parametrize('tool_type', ['web_search_20250305', 'web_search_20260209'])
def test_web_search_server_tool_does_not_require_client_tool_schema(tool_type):
    request = MessagesRequest.model_validate(
        _messages_request_payload([{'type': tool_type, 'name': 'web_search', 'max_uses': 5}])
    )

    assert isinstance(request.tools[0], AnthropicServerTool)
    openai_request = anthropic_to_openai(request, document_mode='images')

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

    openai_request = anthropic_to_openai(request, document_mode='images')

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

    openai_request = anthropic_to_openai(request, document_mode='images')

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

    openai_request = anthropic_to_openai(request, document_mode='images')

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

    openai_request = anthropic_to_openai(request, document_mode='images')

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


def _parallel_tool_result_request(tool_result_ids: list[str]) -> MessagesRequest:
    return MessagesRequest.model_validate(
        {
            'model': 'claude-sonnet-4-6',
            'max_tokens': 1024,
            'messages': [
                {
                    'role': 'assistant',
                    'content': [
                        {
                            'type': 'tool_use',
                            'id': tool_id,
                            'name': 'lookup',
                            'input': {'query': tool_id},
                        }
                        for tool_id in ['call-a', 'call-b', 'call-c']
                    ],
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'tool_result',
                            'tool_use_id': tool_id,
                            'content': f'result-{tool_id}',
                        }
                        for tool_id in tool_result_ids
                    ],
                },
            ],
        }
    )


def _converted_tool_result_ids(request: MessagesRequest) -> list[str]:
    return [
        message['tool_call_id']
        for message in anthropic_to_openai(request, document_mode='images')['messages']
        if message['role'] == 'tool'
    ]


def test_parallel_tool_results_are_aligned_with_assistant_tool_call_order():
    request = _parallel_tool_result_request(['call-b', 'call-a', 'call-c'])

    assert _converted_tool_result_ids(request) == ['call-a', 'call-b', 'call-c']


def _tool_order_request(messages: list[dict]) -> MessagesRequest:
    return MessagesRequest.model_validate(
        {
            'model': 'claude-sonnet-4-6',
            'max_tokens': 1024,
            'messages': messages,
        }
    )


def _assistant_calls(*tool_ids: str) -> dict:
    return {
        'role': 'assistant',
        'content': [
            {'type': 'tool_use', 'id': tool_id, 'name': 'lookup', 'input': {'query': tool_id}} for tool_id in tool_ids
        ],
    }


def _tool_results(*tool_ids: str, text: str | None = None) -> dict:
    content = [{'type': 'tool_result', 'tool_use_id': tool_id, 'content': f'result-{tool_id}'} for tool_id in tool_ids]
    if text is not None:
        content.insert(1, {'type': 'text', 'text': text})
    return {'role': 'user', 'content': content}


def test_adjacent_tool_result_messages_form_one_contiguous_batch():
    request = _tool_order_request(
        [
            _assistant_calls('call-z', 'call-a'),
            _tool_results('call-a'),
            _tool_results('call-z'),
        ]
    )

    assert _converted_tool_result_ids(request) == ['call-z', 'call-a']


def test_tool_results_reorder_without_losing_ordinary_user_content():
    request = _tool_order_request(
        [
            _assistant_calls('call-z', 'call-a'),
            _tool_results('call-a', 'call-z', text='keep this user text'),
        ]
    )

    messages = anthropic_to_openai(request, document_mode='images')['messages']

    assert [message['tool_call_id'] for message in messages if message['role'] == 'tool'] == ['call-z', 'call-a']
    assert messages[-1] == {'role': 'user', 'content': 'keep this user text'}


def test_unknown_duplicate_and_missing_tool_results_are_preserved_stably():
    request = _tool_order_request(
        [
            _assistant_calls('call-z', 'call-a', 'call-missing'),
            _tool_results('unknown-before', 'call-a', 'call-z', 'call-a', 'unknown-after'),
        ]
    )

    messages = anthropic_to_openai(request, document_mode='images')['messages']

    assert len(messages) == 6
    assert [message['tool_call_id'] for message in messages[1:]] == [
        'unknown-before',
        'call-z',
        'call-a',
        'call-a',
        'unknown-after',
    ]


def test_ordinary_user_content_breaks_tool_result_reordering_batch():
    request = _tool_order_request(
        [
            _assistant_calls('call-z', 'call-a'),
            _tool_results('call-a', text='do not cross this boundary'),
            _tool_results('call-z'),
        ]
    )

    messages = anthropic_to_openai(request, document_mode='images')['messages']

    assert [message['role'] for message in messages] == ['assistant', 'tool', 'user', 'tool']
    assert _converted_tool_result_ids(request) == ['call-a', 'call-z']
