import json

from nexus.runtime.server.services.gemini_proxy.converter import (
    gemini_request_to_openai,
    gemini_response_to_openai,
    openai_request_to_gemini,
    openai_response_to_gemini,
)


def test_gemini_request_to_openai_preserves_tool_call_id_for_tool_response():
    request_body = {
        'contents': [
            {
                'role': 'user',
                'parts': [{'text': 'Write hello.txt'}],
            },
            {
                'role': 'model',
                'parts': [
                    {
                        'functionCall': {
                            'id': 'call_write_1',
                            'name': 'write_file',
                            'args': {
                                'file_path': '/tmp/hello.txt',
                                'content': 'Hello, Nexus!',
                            },
                        }
                    }
                ],
            },
            {
                'role': 'user',
                'parts': [
                    {
                        'functionResponse': {
                            'id': 'call_write_1',
                            'name': 'write_file',
                            'response': {'ok': True},
                        }
                    }
                ],
            },
        ]
    }

    params = gemini_request_to_openai(
        '/v1/publishers/google/models/gemini-2.5-flash:generateContent',
        request_body,
    )

    assert params['model'] == 'gemini-2.5-flash'
    assert params['messages'] == [
        {'role': 'user', 'content': 'Write hello.txt'},
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                {
                    'id': 'call_write_1',
                    'type': 'function',
                    'function': {
                        'name': 'write_file',
                        'arguments': json.dumps(
                            {
                                'file_path': '/tmp/hello.txt',
                                'content': 'Hello, Nexus!',
                            }
                        ),
                    },
                }
            ],
        },
        {
            'role': 'tool',
            'tool_call_id': 'call_write_1',
            'content': '{"ok": true}',
        },
    ]


def test_openai_request_to_gemini_omits_tool_call_id_for_upstream_api_requests():
    messages = [
        {'role': 'system', 'content': 'System prompt'},
        {'role': 'user', 'content': 'Write hello.txt'},
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                {
                    'id': 'call_write_1',
                    'type': 'function',
                    'function': {
                        'name': 'write_file',
                        'arguments': '{"file_path": "/tmp/hello.txt", "content": "Hello, Nexus!"}',
                    },
                }
            ],
        },
        {
            'role': 'tool',
            'tool_call_id': 'call_write_1',
            'content': '{"ok": true}',
        },
    ]

    payload = openai_request_to_gemini(messages=messages)

    assert payload['systemInstruction'] == {'parts': [{'text': 'System prompt'}]}
    assert payload['contents'][0] == {
        'role': 'user',
        'parts': [{'text': 'Write hello.txt'}],
    }

    assistant_parts = payload['contents'][1]['parts']
    assert payload['contents'][1]['role'] == 'model'
    assert assistant_parts[0] == {'text': ''}
    assert assistant_parts[1] == {
        'functionCall': {
            'name': 'write_file',
            'args': {
                'file_path': '/tmp/hello.txt',
                'content': 'Hello, Nexus!',
            },
        }
    }
    assert 'id' not in assistant_parts[1]['functionCall']

    assert payload['contents'][2] == {
        'role': 'user',
        'parts': [
            {
                'functionResponse': {
                    'name': 'write_file',
                    'response': {'ok': True},
                }
            }
        ],
    }


def test_openai_response_to_gemini_keeps_tool_call_id_for_cli_follow_up_requests():
    response_body = {
        'id': 'chatcmpl_test',
        'created': 1_736_800_000,
        'model': 'gemini-2.5-flash',
        'choices': [
            {
                'index': 0,
                'finish_reason': 'tool_calls',
                'message': {
                    'role': 'assistant',
                    'content': '',
                    'tool_calls': [
                        {
                            'id': 'call_write_1',
                            'type': 'function',
                            'function': {
                                'name': 'write_file',
                                'arguments': '{"file_path": "/tmp/hello.txt", "content": "Hello, Nexus!"}',
                            },
                        }
                    ],
                },
            }
        ],
        'usage': {
            'prompt_tokens': 10,
            'completion_tokens': 4,
            'total_tokens': 14,
        },
    }

    gemini_response = openai_response_to_gemini(response_body, model='gemini-2.5-flash')

    assert gemini_response['candidates'][0]['content'] == {
        'role': 'model',
        'parts': [
            {
                'functionCall': {
                    'id': 'call_write_1',
                    'name': 'write_file',
                    'args': {
                        'file_path': '/tmp/hello.txt',
                        'content': 'Hello, Nexus!',
                    },
                }
            }
        ],
    }
    assert gemini_response['usageMetadata'] == {
        'promptTokenCount': 10,
        'candidatesTokenCount': 4,
        'totalTokenCount': 14,
    }


def test_gemini_response_to_openai_converts_function_call_arguments_to_json_string():
    response_body = {
        'candidates': [
            {
                'content': {
                    'role': 'model',
                    'parts': [
                        {
                            'functionCall': {
                                'name': 'write_file',
                                'args': {
                                    'file_path': '/tmp/hello.txt',
                                    'content': 'Hello, Nexus!',
                                },
                            }
                        }
                    ],
                },
                'finishReason': 'STOP',
            }
        ],
        'usageMetadata': {
            'promptTokenCount': 10,
            'candidatesTokenCount': 4,
            'totalTokenCount': 14,
        },
    }

    openai_response = gemini_response_to_openai(response_body, model='gemini-2.5-flash')

    message = openai_response['choices'][0]['message']
    assert message['role'] == 'assistant'
    assert message['content'] == ''
    assert len(message['tool_calls']) == 1
    assert message['tool_calls'][0]['function'] == {
        'name': 'write_file',
        'arguments': '{"file_path": "/tmp/hello.txt", "content": "Hello, Nexus!"}',
    }
    assert openai_response['usage'] == {
        'prompt_tokens': 10,
        'completion_tokens': 4,
        'total_tokens': 14,
    }
