from __future__ import annotations

import copy
import json

import pytest

from nexus.gateway.trajectory_assembler import TrajectoryAssembler
from nexus.runtime.server.services.responses_proxy.converter import (
    ResponsesProxyConversionError,
    ToolIdentity,
    _extract_custom_tool_input,
    _normalize_reasoning_item,
    openai_to_responses_events,
    responses_to_openai,
)


def test_responses_string_input_is_not_duplicated() -> None:
    result = responses_to_openai({'model': 'gpt-test', 'input': 'hello'})

    assert result['messages'] == [{'role': 'user', 'content': 'hello'}]


def test_responses_tool_choice_is_normalized_for_chat_clients():
    tools = [
        {'type': 'custom', 'name': 'shell.command', 'description': 'run a command'},
        {
            'type': 'function',
            'name': 'lookup',
            'description': 'look up a record',
            'parameters': {'type': 'object', 'properties': {}},
        },
    ]

    def convert(tool_choice):
        return responses_to_openai({'tools': tools, 'tool_choice': tool_choice})

    assert convert('auto')['tool_choice'] == 'auto'
    assert convert({'type': 'auto'})['tool_choice'] == 'auto'
    assert convert({'type': 'auto', 'disable_parallel_tool_use': False})['tool_choice'] == 'auto'
    assert convert({'type': 'required'})['tool_choice'] == 'required'
    assert convert({'type': 'function', 'name': 'lookup'})['tool_choice'] == {
        'type': 'function',
        'function': {'name': 'lookup'},
    }
    assert convert({'type': 'custom', 'name': 'shell.command'})['tool_choice'] == {
        'type': 'function',
        'function': {'name': 'shell_command'},
    }
    with pytest.raises(ResponsesProxyConversionError, match='tool_choice_tool_not_found'):
        convert({'type': 'function', 'name': 'missing'})


def test_openai_to_responses_events_preserves_reasoning_items():
    events = openai_to_responses_events(
        {
            'id': 'resp_test',
            'choices': [
                {
                    'message': {
                        'role': 'assistant',
                        'content': 'done',
                        'reasoning_items': [
                            {
                                'type': 'reasoning',
                                'id': 'rs_1',
                                'summary': [{'type': 'summary_text', 'text': 'summary'}],
                                'encrypted_content': 'encrypted-payload',
                                'status': 'completed',
                            }
                        ],
                    }
                }
            ],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12},
        }
    )

    reasoning_events = [event for event in events if event.get('item', {}).get('type') == 'reasoning']
    assert reasoning_events == [
        {
            'type': 'response.output_item.done',
            'item': {
                'type': 'reasoning',
                'id': 'rs_1',
                'summary': [{'type': 'summary_text', 'text': 'summary'}],
                'encrypted_content': 'encrypted-payload',
                'status': 'completed',
            },
        }
    ]


def test_openai_to_responses_events_emits_plaintext_reasoning_content():
    """HY3/GLM: only reasoning_content → emit a reasoning item (minted id)."""
    events = openai_to_responses_events({
        'id': 'resp_test',
        'choices': [
            {
                'message': {
                    'role': 'assistant',
                    'content': 'done',
                    'reasoning_content': 'plain summary',
                }
            },
        ]
    })

    reasoning_events = [event for event in events if event.get('item', {}).get('type') == 'reasoning']
    assert len(reasoning_events) == 1
    item = reasoning_events[0]['item']
    assert item['summary'] == [{'type': 'summary_text', 'text': 'plain summary'}]
    assert str(item['id']).startswith('rs_')
    assert 'encrypted_content' not in item


def test_openai_to_responses_events_skips_plaintext_wrap_when_disabled():
    events = openai_to_responses_events(
        {
            'id': 'resp_test',
            'choices': [
                {
                    'message': {
                        'role': 'assistant',
                        'content': 'done',
                        'reasoning_content': 'plain summary',
                    }
                },
            ],
        },
        wrap_plaintext_reasoning_content=False,
    )

    reasoning_events = [event for event in events if event.get('item', {}).get('type') == 'reasoning']
    assert reasoning_events == []


def test_openai_to_responses_events_keeps_native_reasoning_items_when_wrap_disabled():
    events = openai_to_responses_events(
        {
            'id': 'resp_test',
            'choices': [
                {
                    'message': {
                        'role': 'assistant',
                        'content': 'done',
                        'reasoning_content': 'plain summary',
                        'reasoning_items': [
                            {
                                'type': 'reasoning',
                                'id': 'rs_provider',
                                'encrypted_content': 'enc',
                                'summary': [],
                            }
                        ],
                    }
                }
            ],
        },
        wrap_plaintext_reasoning_content=False,
    )

    reasoning_events = [event for event in events if event.get('item', {}).get('type') == 'reasoning']
    assert len(reasoning_events) == 1
    assert reasoning_events[0]['item']['id'] == 'rs_provider'
    assert reasoning_events[0]['item']['encrypted_content'] == 'enc'


def test_normalize_reasoning_item_mints_id_for_encrypted_blob_without_id():
    item = _normalize_reasoning_item({
        'type': 'reasoning',
        'encrypted_content': 'encrypted-payload',
    })
    assert item is not None
    assert str(item['id']).startswith('rs_')
    assert item['encrypted_content'] == 'encrypted-payload'


def test_normalize_reasoning_item_mints_id_for_plaintext_summary():
    item = _normalize_reasoning_item({
        'type': 'reasoning',
        'summary': [{'type': 'summary_text', 'text': 'summary'}],
    })
    assert item is not None
    assert str(item['id']).startswith('rs_')
    assert 'encrypted_content' not in item


def test_normalize_reasoning_item_keeps_provider_id():
    item = _normalize_reasoning_item({
        'type': 'reasoning',
        'id': 'rs_provider',
        'encrypted_content': 'enc',
        'summary': [],
    })
    assert item == {
        'type': 'reasoning',
        'id': 'rs_provider',
        'summary': [],
        'encrypted_content': 'enc',
    }


def test_normalize_reasoning_item_drops_encrypted_blob_without_id_when_wrap_disabled():
    assert _normalize_reasoning_item(
        {
            'type': 'reasoning',
            'encrypted_content': 'encrypted-payload',
        },
        wrap_plaintext_reasoning_content=False,
    ) is None


def test_normalize_reasoning_item_does_not_mint_id_for_plaintext_summary_when_wrap_disabled():
    item = _normalize_reasoning_item(
        {
            'type': 'reasoning',
            'summary': [{'type': 'summary_text', 'text': 'summary'}],
        },
        wrap_plaintext_reasoning_content=False,
    )
    assert item is not None
    assert 'id' not in item
    assert 'encrypted_content' not in item


def test_openai_to_responses_events_omits_encrypted_reasoning_without_id_when_wrap_disabled():
    events = openai_to_responses_events(
        {
            'id': 'resp_test',
            'choices': [
                {
                    'message': {
                        'role': 'assistant',
                        'content': 'done',
                        'reasoning_items': [
                            {
                                'type': 'reasoning',
                                'encrypted_content': 'encrypted-payload',
                            }
                        ],
                    }
                }
            ],
        },
        wrap_plaintext_reasoning_content=False,
    )

    reasoning_events = [event for event in events if event.get('item', {}).get('type') == 'reasoning']
    assert reasoning_events == []


def test_responses_to_openai_attaches_reasoning_to_assistant_message():
    result = responses_to_openai(
        {
            'model': 'gpt-5.5',
            'input': [
                {
                    'type': 'reasoning',
                    'id': 'rs_1',
                    'summary': [{'type': 'summary_text', 'text': 'summary'}],
                    'encrypted_content': 'encrypted-payload',
                    'status': 'completed',
                },
                {
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': 'I will call a tool.'}],
                },
                {
                    'type': 'function_call',
                    'call_id': 'call_1',
                    'name': 'shell',
                    'arguments': '{"cmd": "ls"}',
                },
            ],
        }
    )

    assert result['messages'] == [
        {
            'role': 'assistant',
            'content': 'I will call a tool.',
            'reasoning_items': [
                {
                    'type': 'reasoning',
                    'id': 'rs_1',
                    'summary': [{'type': 'summary_text', 'text': 'summary'}],
                    'encrypted_content': 'encrypted-payload',
                    'status': 'completed',
                }
            ],
            # Plaintext mirror of reasoning_items' summary text, attached for chat models
            # (HY3/GLM) that read reasoning_content; GPT-5.5/Responses ignores it (dropped on
            # the litellm responses path). See _attach_reasoning_content.
            'reasoning_content': 'summary',
            'tool_calls': [
                {
                    'id': 'call_1',
                    'type': 'function',
                    'function': {'name': 'shell', 'arguments': '{"cmd": "ls"}'},
                }
            ],
        }
    ]


def test_responses_to_openai_creates_assistant_holder_for_reasoning_only_tool_call():
    result = responses_to_openai(
        {
            'input': [
                {
                    'type': 'reasoning',
                    'id': 'rs_1',
                    'summary': [],
                    'encrypted_content': 'encrypted-payload',
                },
                {
                    'type': 'function_call',
                    'call_id': 'call_1',
                    'name': 'shell',
                    'arguments': '{}',
                },
            ],
        }
    )

    assert result['messages'] == [
        {
            'role': 'assistant',
            'content': '',
            'reasoning_items': [
                {
                    'type': 'reasoning',
                    'id': 'rs_1',
                    'summary': [],
                    'encrypted_content': 'encrypted-payload',
                }
            ],
            'tool_calls': [
                {
                    'id': 'call_1',
                    'type': 'function',
                    'function': {'name': 'shell', 'arguments': '{}'},
                }
            ],
        }
    ]


_MCP_SEARCH_HISTORY = [
    {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'find it'}]},
    {
        'type': 'function_call',
        'call_id': 'call_search',
        'name': 'search',
        'namespace': 'mcp__search_tools',
        'arguments': '{"query": "q"}',
    },
    {'type': 'function_call_output', 'call_id': 'call_search', 'output': 'results'},
]
_MCP_SEARCH_TOOLS = [
    {
        'type': 'namespace',
        'name': 'mcp__search_tools',
        'tools': [{'type': 'function', 'name': 'search', 'parameters': {'type': 'object', 'properties': {}}}],
    },
]


def _history_tool_call_names(params: dict) -> list[str]:
    return [
        call['function']['name']
        for message in params['messages']
        for call in message.get('tool_calls') or []
    ]


def test_namespaced_history_call_keeps_flattened_name_without_request_tools() -> None:
    # Codex local compaction requests carry the history but no tools.
    params = responses_to_openai({'input': _MCP_SEARCH_HISTORY})

    assert 'tools' not in params
    assert _history_tool_call_names(params) == ['mcp__search_tools__search']


def test_namespaced_history_call_name_matches_with_and_without_request_tools() -> None:
    turn = responses_to_openai({'input': _MCP_SEARCH_HISTORY, 'tools': _MCP_SEARCH_TOOLS})
    compaction = responses_to_openai({'input': _MCP_SEARCH_HISTORY})

    assert [tool['function']['name'] for tool in turn['tools']] == ['mcp__search_tools__search']
    assert _history_tool_call_names(compaction) == _history_tool_call_names(turn)


def test_responses_to_openai_preserves_native_compaction_item_verbatim() -> None:
    compaction_item = {
        'type': 'compaction',
        'id': 'cmp_1',
        'encrypted_content': 'opaque-compaction-state',
    }

    result = responses_to_openai({
        'instructions': 'system instructions',
        'input': [
            {'type': 'message', 'role': 'user', 'content': 'retained user message'},
            compaction_item,
        ],
    })

    assert result['messages'] == [
        {'role': 'system', 'content': 'system instructions'},
        {'role': 'user', 'content': 'retained user message'},
        compaction_item,
    ]


def test_openai_to_responses_events_preserves_tool_call_order_and_argument_strings():
    events = openai_to_responses_events(
        {
            'id': 'resp_test',
            'choices': [
                {
                    'message': {
                        'role': 'assistant',
                        'tool_calls': [
                            {
                                'id': 'call_1',
                                'type': 'function',
                                'function': {
                                    'name': 'str_replace_editor',
                                    'arguments': '{"command":"str_replace","old_str":"old","new_str":"new"}',
                                },
                            },
                            {
                                'id': 'call_2',
                                'type': 'function',
                                'function': {
                                    'name': 'bash',
                                    'arguments': '{"command":"pytest -q"}',
                                },
                            },
                        ],
                    }
                }
            ],
        }
    )

    calls = [event['item'] for event in events if event.get('item', {}).get('type') == 'function_call']
    assert [call['call_id'] for call in calls] == ['call_1', 'call_2']
    assert calls[0]['arguments'] == '{"command":"str_replace","old_str":"old","new_str":"new"}'
    assert calls[1]['arguments'] == '{"command":"pytest -q"}'


def test_responses_to_openai_preserves_function_call_order_and_argument_strings():
    result = responses_to_openai(
        {
            'input': [
                {
                    'type': 'function_call',
                    'call_id': 'call_1',
                    'name': 'str_replace_editor',
                    'arguments': '{"command":"str_replace","old_str":"old","new_str":"new"}',
                },
                {
                    'type': 'function_call',
                    'call_id': 'call_2',
                    'name': 'bash',
                    'arguments': '{"command":"pytest -q"}',
                },
            ],
        }
    )

    tool_calls = result['messages'][0]['tool_calls']
    assert [tool_call['id'] for tool_call in tool_calls] == ['call_1', 'call_2']
    assert tool_calls[0]['function']['arguments'] == '{"command":"str_replace","old_str":"old","new_str":"new"}'
    assert tool_calls[1]['function']['arguments'] == '{"command":"pytest -q"}'


def test_responses_to_openai_keeps_tool_result_adjacent_when_codex_inserts_warning():
    result = responses_to_openai(
        {
            'input': [
                {
                    'type': 'function_call',
                    'call_id': 'call_patch',
                    'name': 'exec_command',
                    'arguments': '{"cmd":"apply_patch ..."}',
                },
                {
                    'type': 'message',
                    'role': 'user',
                    'content': [
                        {
                            'type': 'input_text',
                            'text': 'Warning: apply_patch was requested via exec_command.',
                        }
                    ],
                },
                {
                    'type': 'function_call_output',
                    'call_id': 'call_patch',
                    'output': 'Success. Updated the file.',
                },
            ]
        }
    )

    assert [message['role'] for message in result['messages']] == ['assistant', 'tool', 'user']
    assert result['messages'][1]['tool_call_id'] == 'call_patch'
    assert result['messages'][2]['content'] == 'Warning: apply_patch was requested via exec_command.'


@pytest.mark.parametrize('tool_name_mappings', [None, {'shell': ToolIdentity(tool_type='function', name='shell')}])
def test_custom_tool_round_trips_as_custom_tool_call(tool_name_mappings):
    # Request: a Responses custom tool (Codex freeform apply_patch) is collected and downgraded
    # to a function on the way to the chat host.
    custom_tools: set[str] = set()
    params = responses_to_openai(
        {'tools': [{'type': 'custom', 'name': 'apply_patch', 'description': 'freeform'}]},
        collect_custom_tools=custom_tools,
    )
    assert custom_tools == {'apply_patch'}
    assert params['tools'][0]['type'] == 'function'

    # Response: the host returns a function_call for that name → must be restored to
    # custom_tool_call (Codex >=0.136 only accepts custom_tool_call for apply_patch).
    events = openai_to_responses_events(
        {
            'id': 'resp_1',
            'choices': [
                {
                    'message': {
                        'tool_calls': [
                            {
                                'id': 'call_1',
                                'function': {'name': 'apply_patch', 'arguments': '{"input": "*** Begin Patch"}'},
                            },
                            {'id': 'call_2', 'function': {'name': 'shell', 'arguments': '{"command": "ls"}'}},
                        ]
                    }
                }
            ],
            'usage': {},
        },
        custom_tool_names=custom_tools,
        tool_name_mappings=tool_name_mappings,
    )
    items = [e['item'] for e in events if e['type'] == 'response.output_item.done']
    assert items[0] == {
        'type': 'custom_tool_call',
        'call_id': 'call_1',
        'name': 'apply_patch',
        'input': '*** Begin Patch',
    }
    # Non-custom tool stays a function_call.
    assert items[1]['type'] == 'function_call'
    assert items[1]['name'] == 'shell'


@pytest.mark.parametrize(
    ('chat_name', 'arguments', 'expected_item'),
    [
        (
            'edit',
            '{"value":42}',
            {'type': 'function_call', 'call_id': 'call_edit', 'name': 'edit', 'arguments': '{"value":42}'},
        ),
        (
            'editor__edit',
            '{"input":"replace this line"}',
            {
                'type': 'custom_tool_call',
                'call_id': 'call_edit',
                'name': 'edit',
                'namespace': 'editor',
                'input': 'replace this line',
            },
        ),
    ],
)
def test_mapped_tool_types_win_over_custom_names_with_shared_bare_name(chat_name, arguments, expected_item):
    custom_tools: set[str] = set()
    tool_name_mappings: dict[str, ToolIdentity] = {}
    params = responses_to_openai(
        {
            'tools': [
                {
                    'type': 'function',
                    'name': 'edit',
                    'parameters': {
                        'type': 'object',
                        'properties': {'value': {'type': 'integer'}},
                        'required': ['value'],
                    },
                },
                {'type': 'namespace', 'name': 'editor', 'tools': [{'type': 'custom', 'name': 'edit'}]},
            ],
        },
        collect_custom_tools=custom_tools,
        collect_tool_name_mappings=tool_name_mappings,
    )
    assert [tool['function']['name'] for tool in params['tools']] == ['edit', 'editor__edit']
    assert custom_tools == {'edit'}

    events = openai_to_responses_events(
        {
            'choices': [{
                'message': {
                    'tool_calls': [{
                        'id': 'call_edit',
                        'type': 'function',
                        'function': {'name': chat_name, 'arguments': arguments},
                    }],
                },
            }],
        },
        custom_tool_names=custom_tools,
        tool_name_mappings=tool_name_mappings,
    )

    items = [event['item'] for event in events if event['type'] == 'response.output_item.done']
    assert items == [expected_item]


@pytest.mark.parametrize(('chat_name', 'namespace'), [('editor__edit', 'editor'), ('renderer__edit', 'renderer')])
def test_namespaced_custom_tools_with_shared_name_round_trip(chat_name, namespace):
    custom_tools: set[str] = set()
    tool_name_mappings: dict[str, ToolIdentity] = {}
    params = responses_to_openai(
        {
            'tools': [
                {'type': 'namespace', 'name': 'editor', 'tools': [{'type': 'custom', 'name': 'edit'}]},
                {'type': 'namespace', 'name': 'renderer', 'tools': [{'type': 'custom', 'name': 'edit'}]},
            ],
        },
        collect_custom_tools=custom_tools,
        collect_tool_name_mappings=tool_name_mappings,
    )
    assert [tool['function']['name'] for tool in params['tools']] == ['editor__edit', 'renderer__edit']

    events = openai_to_responses_events(
        {
            'choices': [{
                'message': {
                    'tool_calls': [{
                        'id': 'call_edit',
                        'type': 'function',
                        'function': {'name': chat_name, 'arguments': '{"input":"PROBE_NAMESPACE"}'},
                    }],
                },
            }],
        },
        custom_tool_names=custom_tools,
        tool_name_mappings=tool_name_mappings,
    )

    items = [event['item'] for event in events if event['type'] == 'response.output_item.done']
    assert items == [{
        'type': 'custom_tool_call',
        'call_id': 'call_edit',
        'name': 'edit',
        'namespace': namespace,
        'input': 'PROBE_NAMESPACE',
    }]


def test_codex_0114_classic_tools_round_trip_without_renaming():
    """0.114 classic catalog: top-level function + custom, no namespace.

    Chat names must stay identical so pre-namespace proxy behaviour is preserved.
    """
    custom: set[str] = set()
    mappings: dict[str, ToolIdentity] = {}
    params = responses_to_openai(
        {
            'tools': [
                {
                    'type': 'function',
                    'name': 'exec_command',
                    'description': 'run a command',
                    'parameters': {'type': 'object', 'properties': {'cmd': {'type': 'string'}}},
                },
                {
                    'type': 'function',
                    'name': 'write_stdin',
                    'parameters': {'type': 'object', 'properties': {'data': {'type': 'string'}}},
                },
                {'type': 'custom', 'name': 'apply_patch', 'description': 'freeform patch'},
            ]
        },
        collect_custom_tools=custom,
        collect_tool_name_mappings=mappings,
    )
    assert [tool['function']['name'] for tool in params['tools']] == [
        'exec_command',
        'write_stdin',
        'apply_patch',
    ]
    assert custom == {'apply_patch'}
    assert mappings['exec_command'].tool_type == 'function'
    assert mappings['apply_patch'].tool_type == 'custom'
    assert mappings['apply_patch'].namespace is None

    events = openai_to_responses_events(
        {
            'id': 'resp_114',
            'choices': [{
                'message': {
                    'tool_calls': [
                        {
                            'id': 'call_cmd',
                            'function': {'name': 'exec_command', 'arguments': '{"cmd":"ls"}'},
                        },
                        {
                            'id': 'call_patch',
                            'function': {'name': 'apply_patch', 'arguments': '{"input": "*** Begin Patch"}'},
                        },
                    ]
                }
            }],
        },
        custom_tool_names=custom,
        tool_name_mappings=mappings,
    )
    items = [event['item'] for event in events if event['type'] == 'response.output_item.done']
    assert items[0] == {
        'type': 'function_call',
        'call_id': 'call_cmd',
        'name': 'exec_command',
        'arguments': '{"cmd":"ls"}',
    }
    assert 'namespace' not in items[0]
    assert items[1] == {
        'type': 'custom_tool_call',
        'call_id': 'call_patch',
        'name': 'apply_patch',
        'input': '*** Begin Patch',
    }


def test_tool_calls_default_to_function_call_without_custom_names():
    # Zero-regression: with no custom_tool_names, every tool_call stays a function_call.
    events = openai_to_responses_events(
        {
            'id': 'resp_1',
            'choices': [
                {
                    'message': {
                        'tool_calls': [
                            {'id': 'call_1', 'function': {'name': 'apply_patch', 'arguments': '{"input": "p"}'}},
                        ]
                    }
                }
            ],
            'usage': {},
        }
    )
    items = [e['item'] for e in events if e['type'] == 'response.output_item.done']
    assert items[0]['type'] == 'function_call'


def test_extract_custom_tool_input_handles_off_schema_arguments():
    assert _extract_custom_tool_input('{"input": "p"}') == 'p'  # normal wrapper
    assert _extract_custom_tool_input('*** Begin Patch') == '*** Begin Patch'  # bare patch
    assert _extract_custom_tool_input(None) == ''  # None must not become "null"
    assert _extract_custom_tool_input('{"patch": "p"}') == 'p'  # off-schema dict → sole string


IMAGE_DATA_URL = 'data:image/jpeg;base64,/9j/4AAQSkZJRg'


def test_image_tool_output_reaches_the_model_as_an_image():
    """``view_image`` must land as an image_url part, not flattened base64 text."""
    result = responses_to_openai({
        'input': [
            {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'score?'}]},
            {'type': 'function_call', 'call_id': 'c1', 'name': 'view_image', 'arguments': '{"path":"/w/i.jpg"}'},
            {
                'type': 'function_call_output',
                'call_id': 'c1',
                'output': [{'type': 'input_image', 'image_url': IMAGE_DATA_URL, 'detail': 'original'}],
            },
        ],
    })

    tool_message = next(m for m in result['messages'] if m['role'] == 'tool')
    assert tool_message['tool_call_id'] == 'c1'
    assert tool_message['content'] == [
        {'type': 'image_url', 'image_url': {'url': IMAGE_DATA_URL, 'detail': 'original'}}
    ]


def test_text_only_tool_output_stays_a_plain_string():
    """Text tool outputs stay a string so existing host clients keep working."""
    result = responses_to_openai({
        'input': [
            {'type': 'function_call', 'call_id': 'c1', 'name': 'exec_command', 'arguments': '{}'},
            {
                'type': 'function_call_output',
                'call_id': 'c1',
                'output': [{'type': 'output_text', 'text': 'exit 0'}, {'type': 'output_text', 'text': 'done'}],
            },
        ],
    })

    tool_message = next(m for m in result['messages'] if m['role'] == 'tool')
    assert tool_message['content'] == 'exit 0\ndone'


def test_image_part_without_url_degrades_to_text():
    """An empty url would fail the whole request at the backend, so send it as text."""
    result = responses_to_openai({
        'input': [
            {'type': 'function_call', 'call_id': 'c1', 'name': 'view_image', 'arguments': '{}'},
            {
                'type': 'function_call_output',
                'call_id': 'c1',
                'output': [{'type': 'input_image'}, {'type': 'output_text', 'text': 'caption'}],
            },
        ],
    })

    tool_message = next(m for m in result['messages'] if m['role'] == 'tool')
    assert tool_message['content'] == [
        {'type': 'text', 'text': '{"type": "input_image"}'},
        {'type': 'text', 'text': 'caption'},
    ]


def test_response_lite_additional_tools_round_trip_custom_and_namespace():
    tool_name_mappings: dict[str, ToolIdentity] = {}
    custom_tools: set[str] = set()
    params = responses_to_openai(
        {
            'model': 'gpt-5.6-sol',
            'input': [
                {
                    'type': 'additional_tools',
                    'role': 'developer',
                    'tools': [
                        {'type': 'custom', 'name': 'exec', 'description': 'Run JavaScript.'},
                        {
                            'type': 'function',
                            'name': 'wait',
                            'description': 'Wait for a yielded cell.',
                            'strict': False,
                            'parameters': {
                                'type': 'object',
                                'properties': {'cell_id': {'type': 'string'}},
                                'required': ['cell_id'],
                                'additionalProperties': False,
                            },
                        },
                        {
                            'type': 'namespace',
                            'name': 'collaboration',
                            'description': 'Subagent lifecycle tools.',
                            'tools': [
                                {
                                    'type': 'function',
                                    'name': 'spawn_agent',
                                    'description': 'Spawn a child agent.',
                                    'strict': False,
                                    'parameters': {
                                        'type': 'object',
                                        'properties': {
                                            'task_name': {'type': 'string'},
                                            'message': {'type': 'string'},
                                        },
                                        'required': ['task_name', 'message'],
                                        'additionalProperties': False,
                                    },
                                }
                            ],
                        },
                    ],
                },
                {
                    'type': 'message',
                    'role': 'developer',
                    'content': [{'type': 'input_text', 'text': 'Base instructions.'}],
                },
                {
                    'type': 'message',
                    'role': 'user',
                    'content': [{'type': 'input_text', 'text': 'Delegate this task.'}],
                },
            ],
            'tools': None,
            'tool_choice': 'auto',
            'parallel_tool_calls': False,
        },
        collect_custom_tools=custom_tools,
        collect_tool_name_mappings=tool_name_mappings,
    )

    assert params['messages'] == [
        {'role': 'system', 'content': 'Base instructions.'},
        {'role': 'user', 'content': 'Delegate this task.'},
    ]
    assert [tool['function']['name'] for tool in params['tools']] == [
        'exec',
        'wait',
        'collaboration__spawn_agent',
    ]
    assert params['tools'][1]['function']['strict'] is False
    assert params['tools'][2]['function']['strict'] is False
    assert custom_tools == {'exec'}
    assert tool_name_mappings == {
        'exec': ToolIdentity(tool_type='custom', name='exec'),
        'wait': ToolIdentity(tool_type='function', name='wait'),
        'collaboration__spawn_agent': ToolIdentity(
            tool_type='function',
            name='spawn_agent',
            namespace='collaboration',
        ),
    }

    events = openai_to_responses_events(
        {
            'id': 'resp_lite',
            'choices': [
                {
                    'message': {
                        'tool_calls': [
                            {
                                'id': 'call_exec',
                                'function': {'name': 'exec', 'arguments': '{"input":"text(true)"}'},
                            },
                            {
                                'id': 'call_spawn',
                                'function': {
                                    'name': 'collaboration__spawn_agent',
                                    'arguments': '{"task_name":"child","message":"inspect"}',
                                },
                            },
                        ]
                    }
                }
            ],
        },
        custom_tool_names=custom_tools,
        tool_name_mappings=tool_name_mappings,
    )
    items = [event['item'] for event in events if event['type'] == 'response.output_item.done']
    assert items == [
        {
            'type': 'custom_tool_call',
            'call_id': 'call_exec',
            'name': 'exec',
            'input': 'text(true)',
        },
        {
            'type': 'function_call',
            'call_id': 'call_spawn',
            'namespace': 'collaboration',
            'name': 'spawn_agent',
            'arguments': '{"task_name":"child","message":"inspect"}',
        },
    ]


def test_response_lite_replay_uses_the_same_flattened_tool_names():
    tool_name_mappings: dict[str, ToolIdentity] = {}
    result = responses_to_openai(
        {
            'input': [
                {
                    'type': 'additional_tools',
                    'role': 'developer',
                    'tools': [
                        {'type': 'custom', 'name': 'exec'},
                        {
                            'type': 'namespace',
                            'name': 'collaboration',
                            'tools': [
                                {
                                    'type': 'function',
                                    'name': 'spawn_agent',
                                    'parameters': {'type': 'object', 'properties': {}},
                                }
                            ],
                        },
                    ],
                },
                {'type': 'custom_tool_call', 'call_id': 'call_exec', 'name': 'exec', 'input': 'text(true)'},
                {'type': 'custom_tool_call_output', 'call_id': 'call_exec', 'output': 'true'},
                {
                    'type': 'function_call',
                    'call_id': 'call_spawn',
                    'namespace': 'collaboration',
                    'name': 'spawn_agent',
                    'arguments': '{"task_name":"child","message":"inspect"}',
                },
                {'type': 'function_call_output', 'call_id': 'call_spawn', 'output': '{"agent_id":"agent-1"}'},
            ]
        },
        collect_tool_name_mappings=tool_name_mappings,
    )

    assert result['messages'][0]['tool_calls'][0]['function']['name'] == 'exec'
    assert result['messages'][2]['tool_calls'][0]['function']['name'] == 'collaboration__spawn_agent'
    assert result['messages'][1] == {'role': 'tool', 'tool_call_id': 'call_exec', 'content': 'true'}
    assert result['messages'][3] == {
        'role': 'tool',
        'tool_call_id': 'call_spawn',
        'content': '{"agent_id":"agent-1"}',
    }


def test_response_lite_agent_messages_remain_native():
    agent_messages = [
        {
            'type': 'agent_message',
            'id': 'amsg_child_task',
            'author': '/root',
            'recipient': '/root/probe_child',
            'content': [
                {
                    'type': 'input_text',
                    'text': 'Message Type: NEW_TASK\nTask name: /root/probe_child\nSender: /root\nPayload:\n',
                },
                {'type': 'encrypted_content', 'encrypted_content': 'opaque-child-task'},
            ],
        },
        {
            'type': 'agent_message',
            'id': 'amsg_child_result',
            'author': '/root/probe_child',
            'recipient': '/root',
            'content': [
                {
                    'type': 'input_text',
                    'text': (
                        'Message Type: FINAL_ANSWER\n'
                        'Task name: /root\n'
                        'Sender: /root/probe_child\n'
                        'Payload:\n'
                        'child complete'
                    ),
                }
            ],
        },
    ]
    result = responses_to_openai({'input': agent_messages})

    assert result['messages'] == agent_messages


def test_response_lite_image_detail_is_preserved_for_chat_completions():
    result = responses_to_openai(
        {
            'input': [
                {
                    'type': 'message',
                    'role': 'user',
                    'content': [
                        {'type': 'input_text', 'text': 'Inspect this image.'},
                        {
                            'type': 'input_image',
                            'image_url': 'data:image/png;base64,AAAA',
                            'detail': 'original',
                        },
                    ],
                }
            ]
        }
    )

    assert result['messages'] == [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Inspect this image.'},
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': 'data:image/png;base64,AAAA',
                        'detail': 'original',
                    },
                },
            ],
        }
    ]


def test_unsupported_input_audio_is_reported_instead_of_silently_dropped(caplog):
    result = responses_to_openai(
        {
            'input': [
                {
                    'type': 'message',
                    'role': 'user',
                    'content': [{'type': 'input_audio', 'audio_url': 'data:audio/wav;base64,AAAA'}],
                }
            ]
        }
    )

    assert result['messages'] == [{'role': 'user', 'content': ''}]
    assert 'dropping unsupported message content type=input_audio' in caplog.text


def test_response_request_controls_are_mapped_to_chat_completions():
    result = responses_to_openai(
        {
            'reasoning': {
                'effort': 'xhigh',
                'summary': 'auto',
                'context': 'all_turns',
            },
            'text': {
                'verbosity': 'low',
                'format': {
                    'type': 'json_schema',
                    'name': 'codex_output_schema',
                    'strict': True,
                    'schema': {
                        'type': 'object',
                        'properties': {'answer': {'type': 'string'}},
                        'required': ['answer'],
                        'additionalProperties': False,
                    },
                },
            },
        }
    )

    assert result['reasoning_effort'] == 'xhigh'
    assert result['verbosity'] == 'low'
    assert result['response_format'] == {
        'type': 'json_schema',
        'json_schema': {
            'name': 'codex_output_schema',
            'strict': True,
            'schema': {
                'type': 'object',
                'properties': {'answer': {'type': 'string'}},
                'required': ['answer'],
                'additionalProperties': False,
            },
        },
    }


def test_namespace_flattening_avoids_top_level_name_collisions():
    tool_name_mappings: dict[str, ToolIdentity] = {}
    params = responses_to_openai(
        {
            'tools': [{'type': 'function', 'name': 'collaboration__spawn_agent'}],
            'input': [
                {
                    'type': 'additional_tools',
                    'role': 'developer',
                    'tools': [
                        {
                            'type': 'namespace',
                            'name': 'collaboration',
                            'tools': [{'type': 'function', 'name': 'spawn_agent'}],
                        }
                    ],
                }
            ],
        },
        collect_tool_name_mappings=tool_name_mappings,
    )

    chat_names = [tool['function']['name'] for tool in params['tools']]
    assert chat_names[0] == 'collaboration__spawn_agent'
    assert chat_names[1].startswith('collaboration__spawn_agent__')
    assert len(chat_names[1]) <= 64
    assert tool_name_mappings[chat_names[1]] == ToolIdentity(
        tool_type='function',
        name='spawn_agent',
        namespace='collaboration',
    )


def test_duplicate_top_level_and_lite_tool_declarations_are_coalesced():
    result = responses_to_openai(
        {
            'tools': [{'type': 'function', 'name': 'wait'}],
            'input': [
                {
                    'type': 'additional_tools',
                    'role': 'developer',
                    'tools': [{'type': 'function', 'name': 'wait'}],
                }
            ],
        }
    )

    assert [tool['function']['name'] for tool in result['tools']] == ['wait']


def test_client_tool_search_declaration_becomes_mapped_chat_function():
    tool_name_mappings: dict[str, ToolIdentity] = {}
    result = responses_to_openai(
        {
            'tools': [
                {
                    'type': 'tool_search',
                    'execution': 'client',
                    'description': 'Search deferred tools.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {'type': 'string'},
                            'limit': {'type': 'number'},
                        },
                        'required': ['query'],
                        'additionalProperties': False,
                    },
                }
            ],
            'tool_choice': 'auto',
            'parallel_tool_calls': False,
        },
        collect_tool_name_mappings=tool_name_mappings,
    )

    assert result['tools'] == [
        {
            'type': 'function',
            'function': {
                'name': 'tool_search',
                'description': 'Search deferred tools.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {'type': 'string'},
                        'limit': {'type': 'number'},
                    },
                    'required': ['query'],
                    'additionalProperties': False,
                },
            },
        }
    ]
    assert result['tool_choice'] == 'auto'
    assert result['parallel_tool_calls'] is False
    assert tool_name_mappings == {
        'tool_search': ToolIdentity(
            tool_type='tool_search',
            name='tool_search',
            execution='client',
        )
    }


def test_client_tool_search_provider_call_restores_native_response_item():
    events = openai_to_responses_events(
        {
            'id': 'resp-search',
            'choices': [
                {
                    'message': {
                        'tool_calls': [
                            {
                                'id': 'call-search-1',
                                'function': {
                                    'name': 'tool_search',
                                    'arguments': '{"query":"spawn agent","limit":1}',
                                },
                            }
                        ]
                    }
                }
            ],
        },
        tool_name_mappings={
            'tool_search': ToolIdentity(
                tool_type='tool_search',
                name='tool_search',
                execution='client',
            )
        },
    )

    output_items = [event['item'] for event in events if event['type'] == 'response.output_item.done']
    assert output_items == [
        {
            'type': 'tool_search_call',
            'call_id': 'call-search-1',
            'execution': 'client',
            'arguments': {'query': 'spawn agent', 'limit': 1},
        }
    ]


def test_client_tool_search_name_collision_remains_reversible():
    tool_name_mappings: dict[str, ToolIdentity] = {}
    params = responses_to_openai(
        {
            'tools': [
                {'type': 'function', 'name': 'tool_search'},
                {
                    'type': 'tool_search',
                    'execution': 'client',
                    'parameters': {
                        'type': 'object',
                        'properties': {'query': {'type': 'string'}},
                        'required': ['query'],
                    },
                },
            ]
        },
        collect_tool_name_mappings=tool_name_mappings,
    )

    chat_names = [tool['function']['name'] for tool in params['tools']]
    assert chat_names[0] == 'tool_search'
    assert chat_names[1].startswith('tool_search__')
    assert tool_name_mappings[chat_names[0]] == ToolIdentity(tool_type='function', name='tool_search')
    assert tool_name_mappings[chat_names[1]] == ToolIdentity(
        tool_type='tool_search',
        name='tool_search',
        execution='client',
    )

    events = openai_to_responses_events(
        {
            'id': 'resp-collision',
            'choices': [
                {
                    'message': {
                        'tool_calls': [
                            {
                                'id': 'call-ordinary',
                                'function': {'name': chat_names[0], 'arguments': '{}'},
                            },
                            {
                                'id': 'call-search',
                                'function': {'name': chat_names[1], 'arguments': '{"query":"agent"}'},
                            },
                        ]
                    }
                }
            ],
        },
        tool_name_mappings=tool_name_mappings,
    )

    output_items = [event['item'] for event in events if event['type'] == 'response.output_item.done']
    assert output_items == [
        {
            'type': 'function_call',
            'call_id': 'call-ordinary',
            'name': 'tool_search',
            'arguments': '{}',
        },
        {
            'type': 'tool_search_call',
            'call_id': 'call-search',
            'execution': 'client',
            'arguments': {'query': 'agent'},
        },
    ]


def _make_tool_search_request(model: str | None) -> dict:
    loaded_tool = {
        'type': 'function',
        'name': 'calculate',
        'description': 'Evaluate an operation.',
        'defer_loading': True,
        'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}}},
    }
    request = {
        'tools': [{'type': 'tool_search', 'execution': 'client'}],
        'input': [
            {'type': 'message', 'role': 'user', 'content': 'calculate'},
            {
                'type': 'tool_search_call',
                'call_id': 'search',
                'execution': 'client',
                'arguments': {'query': 'calculator'},
            },
            {
                'type': 'tool_search_output',
                'call_id': 'search',
                'execution': 'client',
                'status': 'completed',
                'tools': [{'type': 'namespace', 'name': 'calculator', 'tools': [loaded_tool]}],
            },
        ],
    }
    if model is not None:
        request['model'] = model
    return request


@pytest.mark.parametrize('model', ['model-name', 'provider/model-name', 'HY4-test', 'openai/hy4-test', None])
def test_standard_chat_tool_search_uses_text_and_eager_schemas(model: str | None) -> None:
    request = _make_tool_search_request(model)
    original = copy.deepcopy(request)

    params = responses_to_openai(request)

    result = params['messages'][2]
    assert result['tool_call_id'] == 'search'
    assert isinstance(result['content'], str)
    discovered = json.loads(result['content'])
    assert discovered['status'] == 'completed'
    assert discovered['execution'] == 'client'
    assert discovered['tools'][0]['function']['name'] == 'calculator__calculate'
    assert discovered['tools'][0]['function']['parameters']['properties']['a'] == {'type': 'integer'}
    assert all('defer_loading' not in tool['function'] for tool in params['tools'])
    assert 'defer_loading' not in discovered['tools'][0]['function']
    assert request == original


@pytest.mark.parametrize(
    ('model', 'supports_tool_references', 'expected_references'),
    [
        ('model-name', True, True),
        ('model-name', False, False),
        ('HY4-test', True, True),
        ('HY4-test', False, False),
    ],
)
def test_tool_search_reference_capability_is_independent_of_model_name(
    model: str, supports_tool_references: bool, expected_references: bool
) -> None:
    params = responses_to_openai(_make_tool_search_request(model), supports_tool_references=supports_tool_references)

    result = params['messages'][2]
    if expected_references:
        assert result['content'] == [{'type': 'tool_reference', 'name': 'calculator__calculate'}]
        assert params['tools'][1]['function']['defer_loading'] is True
    else:
        assert isinstance(result['content'], str)
        assert 'defer_loading' not in params['tools'][1]['function']


@pytest.mark.parametrize(
    ('supports_tool_references', 'tool_search_output_format', 'expected_segments'),
    [(True, 'tool_reference', 1), (True, 'tool_search_output', 1), (False, 'tool_reference', 2)],
)
def test_search_schema_loading_preserves_backend_segment_semantics(
    supports_tool_references: bool, tool_search_output_format: str, expected_segments: int
) -> None:
    request = _make_tool_search_request('model-name')
    first_params = responses_to_openai(
        {**request, 'input': request['input'][:1]},
        supports_tool_references=supports_tool_references,
        tool_search_output_format=tool_search_output_format,
    )
    second_params = responses_to_openai(
        request,
        supports_tool_references=supports_tool_references,
        tool_search_output_format=tool_search_output_format,
    )
    assembler = TrajectoryAssembler(scaffold_policy='codex')
    assembler.record_completion(
        first_params['messages'], second_params['messages'][1], request_kwargs={'tools': first_params['tools']}
    )
    assembler.record_completion(
        second_params['messages'],
        {'role': 'assistant', 'content': 'answer'},
        request_kwargs={'tools': second_params['tools']},
    )

    trajectory = assembler.freeze(session_id='search-schema-loading', state='completed')
    assert len(trajectory.to_segment_node_lists()) == expected_segments


@pytest.mark.parametrize('empty_tools', [False, True])
def test_tool_search_output_embeds_chat_schemas_and_preserves_images(empty_tools: bool) -> None:
    request = _make_tool_search_request('HY4-VLM-test')
    request['input'][0]['content'] = [
        {'type': 'input_text', 'text': 'Inspect this image and use the calculator.'},
        {'type': 'input_image', 'image_url': 'https://example.invalid/image.png'},
    ]
    if empty_tools:
        request['input'][-1]['tools'] = []
    original = copy.deepcopy(request)
    mappings: dict[str, ToolIdentity] = {}

    params = responses_to_openai(
        request,
        collect_tool_name_mappings=mappings,
        supports_tool_references=True,
        tool_search_output_format='tool_search_output',
    )

    discovered_tools = params['tools'][1:]
    assert params['messages'][2] == {
        'role': 'tool',
        'tool_call_id': 'search',
        'content': [{'type': 'tool_search_output', 'tools': discovered_tools}],
    }, 'VLM tool search results must embed schemas rather than references'
    assert params['messages'][0]['content'][1] == {
        'type': 'image_url',
        'image_url': {'url': 'https://example.invalid/image.png'},
    }, 'Tool search conversion must preserve image content'
    assert request == original, 'Conversion must not mutate the Responses request'
    if not empty_tools:
        assert discovered_tools[0]['function']['defer_loading'] is True, 'Loaded schemas must remain deferred'
        assert discovered_tools[0]['function']['parameters']['properties']['a'] == {'type': 'integer'}, (
            'Inline search results must retain parameter schemas'
        )
        assert mappings['calculator__calculate'].namespace == 'calculator', 'Namespace identity must survive discovery'


@pytest.mark.parametrize(
    ('supports_tool_references', 'tool_search_output_format', 'error_code'),
    [
        (True, 'invalid', 'tool_search_output_format_invalid'),
        (False, 'tool_search_output', 'tool_search_output_requires_deferred_loading'),
    ],
)
def test_tool_search_output_rejects_incompatible_configuration(
    supports_tool_references: bool, tool_search_output_format: str, error_code: str
) -> None:
    with pytest.raises(ResponsesProxyConversionError, match=error_code):
        responses_to_openai(
            _make_tool_search_request('HY4-VLM-test'),
            supports_tool_references=supports_tool_references,
            tool_search_output_format=tool_search_output_format,
        )


def test_client_tool_search_replay_exposes_loaded_tools_and_preserves_pairing():
    tool_name_mappings: dict[str, ToolIdentity] = {}
    loaded_tools = [
        {
            'type': 'function',
            'name': 'get_shipping_eta',
            'description': 'Look up shipping ETA.',
            'defer_loading': True,
            'parameters': {
                'type': 'object',
                'properties': {'order_id': {'type': 'string'}},
                'required': ['order_id'],
                'additionalProperties': False,
            },
        },
        {
            'type': 'namespace',
            'name': 'multi_agent_v1',
            'description': 'Subagent lifecycle tools.',
            'tools': [
                {
                    'type': 'function',
                    'name': 'spawn_agent',
                    'description': 'Spawn a child agent.',
                    'defer_loading': True,
                    'parameters': {
                        'type': 'object',
                        'properties': {'message': {'type': 'string'}},
                        'required': ['message'],
                        'additionalProperties': False,
                    },
                }
            ],
        },
    ]
    params = responses_to_openai(
        {
            'model': 'model-name',
            'tools': [
                {
                    'type': 'tool_search',
                    'execution': 'client',
                    'parameters': {
                        'type': 'object',
                        'properties': {'query': {'type': 'string'}},
                        'required': ['query'],
                    },
                }
            ],
            'input': [
                {
                    'type': 'tool_search_call',
                    'call_id': 'call-search-1',
                    'execution': 'client',
                    'arguments': {'query': 'spawn agent', 'limit': 2},
                },
                {
                    'type': 'tool_search_output',
                    'call_id': 'call-search-1',
                    'status': 'completed',
                    'execution': 'client',
                    'tools': loaded_tools,
                },
            ],
        },
        collect_tool_name_mappings=tool_name_mappings,
        supports_tool_references=True,
    )

    assert [tool['function']['name'] for tool in params['tools']] == [
        'tool_search',
        'get_shipping_eta',
        'multi_agent_v1__spawn_agent',
    ]
    assert tool_name_mappings == {
        'tool_search': ToolIdentity(
            tool_type='tool_search',
            name='tool_search',
            execution='client',
        ),
        'get_shipping_eta': ToolIdentity(tool_type='function', name='get_shipping_eta'),
        'multi_agent_v1__spawn_agent': ToolIdentity(
            tool_type='function',
            name='spawn_agent',
            namespace='multi_agent_v1',
        ),
    }
    assert params['messages'][0] == {
        'role': 'assistant',
        'content': '',
        'tool_calls': [
            {
                'id': 'call-search-1',
                'type': 'function',
                'function': {
                    'name': 'tool_search',
                    'arguments': '{"query": "spawn agent", "limit": 2}',
                },
            }
        ],
    }
    assert params['messages'][1]['role'] == 'tool'
    assert params['messages'][1]['tool_call_id'] == 'call-search-1'
    assert params['messages'][1]['content'] == [
        {'type': 'tool_reference', 'name': 'get_shipping_eta'},
        {'type': 'tool_reference', 'name': 'multi_agent_v1__spawn_agent'},
    ]
    assert all(tool['function']['defer_loading'] is True for tool in params['tools'][1:])


@pytest.mark.parametrize('tool_type', ['function', 'custom'])
@pytest.mark.parametrize('defer_loading', [True, False, None])
def test_callable_tools_preserve_boolean_defer_loading(tool_type, defer_loading):
    tool = {'type': tool_type, 'name': 'echo'}
    if defer_loading is not None:
        tool['defer_loading'] = defer_loading
    params = responses_to_openai({'tools': [tool], 'input': 'hello'}, supports_tool_references=True)

    converted = params['tools'][0]
    assert 'defer_loading' not in converted
    if defer_loading is None:
        assert 'defer_loading' not in converted['function']
    else:
        assert converted['function']['defer_loading'] is defer_loading


@pytest.mark.parametrize('namespace', ['mcp', 'mcp_' + 'x' * 70])
def test_tool_search_references_use_mapped_names_for_collisions_and_long_names(namespace):
    tool_name_mappings: dict[str, ToolIdentity] = {}
    loaded_tool = {'type': 'function', 'name': 'echo', 'defer_loading': True}
    params = responses_to_openai(
        {
            'model': 'model-name',
            'tools': [
                {'type': 'tool_search', 'execution': 'client'},
                {'type': 'function', 'name': f'{namespace}__echo'},
            ],
            'input': [
                {
                    'type': 'tool_search_call',
                    'call_id': 'search',
                    'execution': 'client',
                    'arguments': {'query': 'echo'},
                },
                {
                    'type': 'tool_search_output',
                    'call_id': 'search',
                    'status': 'completed',
                    'execution': 'client',
                    'tools': [{'type': 'namespace', 'name': namespace, 'tools': [loaded_tool, loaded_tool]}],
                },
                {
                    'type': 'function_call',
                    'call_id': 'echo',
                    'namespace': namespace,
                    'name': 'echo',
                    'arguments': '{}',
                },
            ],
        },
        collect_tool_name_mappings=tool_name_mappings,
        supports_tool_references=True,
    )

    chat_name = params['tools'][2]['function']['name']
    assert chat_name != params['tools'][1]['function']['name']
    assert len(chat_name) <= 64
    assert tool_name_mappings[chat_name] == ToolIdentity(tool_type='function', name='echo', namespace=namespace)
    assert params['messages'][1]['content'] == [{'type': 'tool_reference', 'name': chat_name}]
    assert params['messages'][2]['tool_calls'][0]['function']['name'] == chat_name


def test_empty_tool_search_result_preserves_status():
    params = responses_to_openai({
        'tools': [{'type': 'tool_search', 'execution': 'client'}],
        'input': [
            {
                'type': 'tool_search_output',
                'call_id': 'search',
                'status': 'completed',
                'execution': 'client',
                'tools': [],
            }
        ],
    })

    assert json.loads(params['messages'][0]['content']) == {
        'status': 'completed',
        'execution': 'client',
        'tools': [],
    }


def test_client_tool_search_replay_preserves_multiple_call_order():
    result = responses_to_openai(
        {
            'tools': [{'type': 'tool_search', 'execution': 'client'}],
            'input': [
                {
                    'type': 'tool_search_call',
                    'call_id': 'call-search-1',
                    'execution': 'client',
                    'arguments': {'query': 'first'},
                },
                {
                    'type': 'tool_search_output',
                    'call_id': 'call-search-1',
                    'status': 'completed',
                    'execution': 'client',
                    'tools': [],
                },
                {
                    'type': 'tool_search_call',
                    'call_id': 'call-search-2',
                    'execution': 'client',
                    'arguments': {'query': 'second'},
                },
                {
                    'type': 'tool_search_output',
                    'call_id': 'call-search-2',
                    'status': 'completed',
                    'execution': 'client',
                    'tools': [],
                },
            ],
        }
    )

    assert [
        message.get('tool_calls', [{}])[0].get('id') if message['role'] == 'assistant' else message['tool_call_id']
        for message in result['messages']
    ] == ['call-search-1', 'call-search-1', 'call-search-2', 'call-search-2']


@pytest.mark.parametrize(
    ('request_body', 'error_code'),
    [
        (
            {'tools': [{'type': 'tool_search', 'execution': 'server'}]},
            'tool_search_execution_unsupported',
        ),
        (
            {
                'input': [
                    {
                        'type': 'tool_search_call',
                        'call_id': 'call-search-1',
                        'execution': 'server',
                        'arguments': {'query': 'server search'},
                    }
                ]
            },
            'tool_search_execution_unsupported',
        ),
        (
            {
                'input': [
                    {
                        'type': 'tool_search_output',
                        'call_id': 'call-search-1',
                        'status': 'completed',
                        'execution': 'server',
                        'tools': [],
                    }
                ]
            },
            'tool_search_execution_unsupported',
        ),
        (
            {
                'input': [
                    {
                        'type': 'tool_search_call',
                        'execution': 'client',
                        'arguments': {'query': 'missing id'},
                    }
                ]
            },
            'tool_search_call_id_missing',
        ),
        (
            {
                'input': [
                    {
                        'type': 'tool_search_output',
                        'status': 'completed',
                        'execution': 'client',
                        'tools': [],
                    }
                ]
            },
            'tool_search_call_id_missing',
        ),
        (
            {
                'input': [
                    {
                        'type': 'tool_search_output',
                        'call_id': 'call-search-1',
                        'status': 'completed',
                        'execution': 'client',
                        'tools': {'not': 'a list'},
                    }
                ]
            },
            'tool_search_tools_invalid',
        ),
    ],
)
def test_invalid_client_tool_search_protocol_is_rejected(request_body: dict, error_code: str):
    with pytest.raises(ValueError, match=error_code):
        responses_to_openai(request_body)


@pytest.mark.parametrize(
    ('first_content', 'first_has_tool_call'),
    [('first answer', False), ('first answer', True), ('', True)],
    ids=['message-only', 'message-and-tool', 'tool-only'],
)
def test_new_reasoning_does_not_attach_to_the_previous_assistant(
    first_content: str, first_has_tool_call: bool,
) -> None:
    items = [
        {'type': 'reasoning', 'id': 'rs_first', 'summary': [{'type': 'summary_text', 'text': 'first thought'}]},
    ]
    if first_content:
        items.append({'type': 'message', 'role': 'assistant', 'content': first_content})
    if first_has_tool_call:
        items.append({'type': 'function_call', 'call_id': 'call_first', 'name': 'shell', 'arguments': '{}'})
    items.extend([
        {'type': 'reasoning', 'id': 'rs_second', 'summary': [{'type': 'summary_text', 'text': 'second thought'}]},
        {'type': 'message', 'role': 'assistant', 'content': 'second answer'},
    ])
    messages = responses_to_openai({'input': items})['messages']
    assert [(message.get('reasoning_content'), message['content']) for message in messages] == [
        ('first thought', first_content),
        ('second thought', 'second answer'),
    ]
    assert [[item['id'] for item in message['reasoning_items']] for message in messages] == [
        ['rs_first'],
        ['rs_second'],
    ]
    if first_has_tool_call:
        assert messages[0]['tool_calls'][0]['id'] == 'call_first'
    assert 'tool_calls' not in messages[1]


def test_consecutive_reasoning_items_stay_with_the_same_assistant() -> None:
    items = [
        {'type': 'reasoning', 'id': 'rs_first', 'summary': [{'type': 'summary_text', 'text': 'first thought'}]},
        {'type': 'reasoning', 'id': 'rs_second', 'summary': [], 'encrypted_content': 'opaque'},
        {'type': 'message', 'role': 'assistant', 'content': 'answer'},
    ]
    messages = responses_to_openai({'input': items})['messages']
    assert messages == [
        {
            'role': 'assistant',
            'content': 'answer',
            'reasoning_content': 'first thought',
            'reasoning_items': items[:2],
        }
    ]


def test_function_tool_image_output_remains_structured():
    image_url = 'data:image/png;base64,' + 'A' * 2048

    result = responses_to_openai({
        'input': [
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
    })

    assert result['messages'][1] == {
        'role': 'tool',
        'tool_call_id': 'call_view',
        'content': [
            {'type': 'image_url', 'image_url': {'url': image_url, 'detail': 'high'}},
        ],
    }


def test_exec_command_output_metadata_normalization_is_explicit_and_non_mutating() -> None:
    request = {
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
                'output': (
                    'Chunk ID: a1b2c3\nWall time: 0.1923 seconds\nProcess exited with code 0\nFinal output:\ndone\n'
                ),
            },
            {
                'type': 'function_call',
                'call_id': 'call_other',
                'name': 'read_file',
                'arguments': '{"path":"result.txt"}',
            },
            {
                'type': 'function_call_output',
                'call_id': 'call_other',
                'output': 'Chunk ID: semantic-content\nWall time: must-stay\n',
            },
        ]
    }

    from nexus.runtime.server.services.responses_proxy import converter

    normalized = converter.normalize_exec_command_output_metadata(request)

    assert normalized is not request
    assert normalized['input'][1]['output'].startswith('Chunk ID: deterministic\nWall time: deterministic seconds\n')
    assert normalized['input'][1]['output'].endswith('Final output:\ndone\n')
    assert normalized['input'][3]['output'] == 'Chunk ID: semantic-content\nWall time: must-stay\n'
    assert request['input'][1]['output'].startswith('Chunk ID: a1b2c3\nWall time: 0.1923 seconds\n')
