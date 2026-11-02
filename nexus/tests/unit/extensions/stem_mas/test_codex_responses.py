import pytest

from nexus.extensions.tasks.stem_mas.patches.codex_responses import McpNamespaceToolCodec
from nexus.runtime.server.services.responses_proxy.converter import ToolIdentity


def test_namespace_tools_are_flattened_and_restored(monkeypatch) -> None:
    codec = McpNamespaceToolCodec()
    monkeypatch.setenv(codec.allowed_tools_env, '{"lean_lsp":["lean_run_code"]}')
    tools = [
        {
            'type': 'namespace',
            'name': 'lean_lsp',
            'tools': [
                {'type': 'function', 'name': 'lean_run_code', 'parameters': {'type': 'object'}},
                {'type': 'function', 'name': 'lean_goal', 'parameters': {'type': 'object'}},
            ],
        }
    ]

    flattened = codec.flatten_namespace_tools(tools)

    assert [tool['name'] for tool in flattened] == ['mcp__lean_lsp__lean_run_code']
    events = [
        {
            'type': 'response.output_item.done',
            'item': {'type': 'function_call', 'name': flattened[0]['name']},
        }
    ]
    restored = codec.restore_response_events(events)
    assert restored[0]['item']['namespace'] == 'mcp__lean_lsp__'
    assert restored[0]['item']['name'] == 'lean_run_code'


def test_malformed_tool_names_are_not_guessed() -> None:
    codec = McpNamespaceToolCodec()

    assert codec.split_tool_name('lean_lsplean_run_code') is None
    assert codec.split_tool_name('mcp__lean_lsp_lean_run_code') is None
    assert codec.split_tool_name('mcp__lean_lsp__lean_run_code') == ('lean_lsp', 'lean_run_code')


def test_input_tool_call_namespace_is_flattened_without_mutation() -> None:
    codec = McpNamespaceToolCodec()
    original = {'type': 'function_call', 'namespace': 'mcp__lean_lsp__', 'name': 'lean_run_code'}

    flattened = codec.flatten_namespaced_input([original])

    assert flattened == [{'type': 'function_call', 'name': 'mcp__lean_lsp__lean_run_code'}]
    assert original['namespace'] == 'mcp__lean_lsp__'


def test_response_lite_collaboration_namespace_is_left_for_base_converter() -> None:
    codec = McpNamespaceToolCodec()
    tool_name_mappings: dict[str, ToolIdentity] = {}
    result = codec.responses_to_openai(
        {
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
                },
                {
                    'type': 'function_call',
                    'call_id': 'call-spawn',
                    'namespace': 'collaboration',
                    'name': 'spawn_agent',
                    'arguments': '{}',
                },
            ]
        },
        collect_tool_name_mappings=tool_name_mappings,
    )

    assert result['tools'][0]['function']['name'] == 'collaboration__spawn_agent'
    assert result['messages'][0]['tool_calls'][0]['function']['name'] == 'collaboration__spawn_agent'
    events = codec.openai_to_responses_events(
        {
            'choices': [
                {
                    'message': {
                        'tool_calls': [
                            {
                                'id': 'call-spawn',
                                'function': {'name': 'collaboration__spawn_agent', 'arguments': '{}'},
                            }
                        ]
                    }
                }
            ]
        },
        tool_name_mappings=tool_name_mappings,
    )
    call_items = [event['item'] for event in events if event['type'] == 'response.output_item.done']
    assert call_items == [
        {
            'type': 'function_call',
            'call_id': 'call-spawn',
            'namespace': 'collaboration',
            'name': 'spawn_agent',
            'arguments': '{}',
        }
    ]


def test_configured_allowlist_denies_unlisted_namespace(monkeypatch) -> None:
    codec = McpNamespaceToolCodec()
    monkeypatch.setenv(codec.allowed_tools_env, '{"lean_lsp":["lean_run_code"]}')

    assert codec.allowed_namespace_tools('search_tools') == set()


@pytest.mark.parametrize('raw_config', ['not-json', '[]', '{"lean_lsp":"lean_run_code"}', '{"lean_lsp":[1]}'])
def test_invalid_allowlist_fails_closed(monkeypatch, raw_config: str) -> None:
    codec = McpNamespaceToolCodec()
    monkeypatch.setenv(codec.allowed_tools_env, raw_config)

    with pytest.raises(ValueError, match=codec.allowed_tools_env):
        codec.allowed_namespace_tools('lean_lsp')
