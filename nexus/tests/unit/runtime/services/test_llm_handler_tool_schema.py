from nexus.runtime.server.services.llm_handler import sanitize_openai_params


def _fn(name, parameters):
    return {'type': 'function', 'function': {'name': name, 'description': 'd', 'parameters': parameters}}


def _params(parameters):
    out = sanitize_openai_params({'model': 'm', 'messages': [], 'tools': [_fn('t', parameters)]})
    return out['tools'][0]['function']['parameters']


def test_zero_argument_tool_drops_empty_properties():
    """Vertex rejects `{'type': 'object', 'properties': {}}`.

    It answers "functionDeclaration parameters schema should be of type OBJECT"
    and refuses the entire request, so one zero-argument tool anywhere in the
    list stops the agent taking any step. Playwright MCP emits this shape for
    browser_snapshot, browser_close and friends.

    This must be enforced host-side: the runtime-side proxies convert tools
    inside the container, from a published runtime image that may predate any
    fix in that converter.
    """
    assert _params({'type': 'object', 'properties': {}}) == {'type': 'object'}


def test_populated_properties_survive():
    schema = {'type': 'object', 'properties': {'x': {'type': 'string'}}, 'required': ['x']}
    assert _params(schema) == schema


def test_nested_empty_properties_are_dropped_too():
    assert _params({'type': 'object', 'properties': {'o': {'type': 'object', 'properties': {}}}}) == {
        'type': 'object',
        'properties': {'o': {'type': 'object'}},
    }


def test_missing_type_is_filled_in():
    assert _params({'properties': {}}) == {'type': 'object'}


def test_tool_without_parameters_gets_object_type():
    out = sanitize_openai_params({'model': 'm', 'messages': [], 'tools': [{'type': 'function', 'function': {'name': 't'}}]})
    assert out['tools'][0]['function']['parameters'] == {'type': 'object'}


def test_mixed_batch_keeps_both_tools():
    out = sanitize_openai_params({
        'model': 'm',
        'messages': [],
        'tools': [
            _fn('browser_snapshot', {'type': 'object', 'properties': {}}),
            _fn('browser_click', {'type': 'object', 'properties': {'ref': {'type': 'string'}}}),
        ],
    })
    assert [t['function']['parameters'] for t in out['tools']] == [
        {'type': 'object'},
        {'type': 'object', 'properties': {'ref': {'type': 'string'}}},
    ]


def test_requests_without_tools_are_untouched():
    assert 'tools' not in sanitize_openai_params({'model': 'm', 'messages': []})
    assert sanitize_openai_params({'model': 'm', 'messages': [], 'tools': []})['tools'] == []


def test_non_function_entries_pass_through():
    """Anything that is not a function tool is left exactly as it arrived."""
    tools = [{'type': 'web_search'}, 'weird-but-not-ours']
    assert sanitize_openai_params({'model': 'm', 'messages': [], 'tools': tools})['tools'] == tools


def test_unsupported_completion_fields_still_dropped():
    out = sanitize_openai_params({'model': 'm', 'messages': [], 'store': True, 'service_tier': 'auto'})
    assert 'store' not in out
    assert 'service_tier' not in out


def test_function_strict_is_stripped():
    """Codex/CodeBuddy set function.strict; AIHub Claude rejects it as tools.N.custom.strict."""
    out = sanitize_openai_params({
        'model': 'claude-opus-4-8',
        'messages': [],
        'tools': [
            {
                'type': 'function',
                'function': {
                    'name': 'exec_command',
                    'description': 'run',
                    'parameters': {'type': 'object', 'properties': {'cmd': {'type': 'string'}}},
                    'strict': False,
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'Agent',
                    'description': 'spawn',
                    'parameters': {'type': 'object', 'properties': {}},
                    'strict': True,
                },
            },
        ],
    })
    assert [t['function'].get('strict', '<missing>') for t in out['tools']] == ['<missing>', '<missing>']
    assert out['tools'][0]['function']['name'] == 'exec_command'
    assert out['tools'][1]['function']['parameters'] == {'type': 'object'}
