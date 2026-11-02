from nexus.runtime.server.services.anthropic_proxy.converter import anthropic_to_openai, convert_tools_to_openai
from nexus.runtime.server.services.anthropic_proxy.models import MessagesRequest


def _params(input_schema):
    tools = convert_tools_to_openai([{'name': 't', 'description': 'd', 'input_schema': input_schema}])
    return tools[0]['function']['parameters']


def _params_via_request(input_schema):
    """Same thing, but through the request model the proxy actually validates.

    `_params` hands a bare dict straight to the converter and so never exercises
    `ToolInputSchema`. Every keyword the pydantic layer drops is invisible to it.
    """
    request = MessagesRequest.model_validate(
        {
            'model': 'test-model',
            'messages': [{'role': 'user', 'content': 'hi'}],
            'tools': [{'name': 't', 'description': 'd', 'input_schema': input_schema}],
        }
    )
    return anthropic_to_openai(request)['tools'][0]['function']['parameters']


def test_zero_argument_tool_drops_empty_properties():
    """Vertex rejects `{'type': 'object', 'properties': {}}`.

    It answers "functionDeclaration parameters schema should be of type OBJECT"
    and refuses the entire request, so a single zero-argument tool anywhere in
    the list stops the agent taking any step at all. Playwright MCP emits this
    shape for browser_snapshot, browser_close and friends. A bare
    `{'type': 'object'}` is accepted.
    """
    assert _params({'type': 'object', 'properties': {}}) == {'type': 'object'}


def test_zero_argument_tool_keeps_other_keys():
    assert _params({'type': 'object', 'properties': {}, 'required': []}) == {
        'type': 'object',
        'required': [],
    }


def test_nested_empty_properties_are_dropped_too():
    assert _params({'type': 'object', 'properties': {'o': {'type': 'object', 'properties': {}}}}) == {
        'type': 'object',
        'properties': {'o': {'type': 'object'}},
    }


def test_populated_properties_survive():
    schema = {'type': 'object', 'properties': {'x': {'type': 'string'}}, 'required': ['x']}
    assert _params(schema) == schema


def test_missing_type_is_filled_in():
    """An empty input_schema must still name a type, or Vertex refuses it."""
    assert _params({}) == {'type': 'object'}


def test_unsupported_keys_are_still_stripped():
    schema = {
        '$schema': 'http://json-schema.org/draft-07/schema#',
        'type': 'object',
        'properties': {'x': {'type': 'string', 'default': 'a'}},
        'additionalProperties': False,
    }
    assert _params(schema) == {'type': 'object', 'properties': {'x': {'type': 'string'}}}


# The tests above hand bare dicts to the converter, so they say nothing about
# what `ToolInputSchema` keeps. These go through `MessagesRequest`.


def test_defs_survive_request_validation():
    """A dropped `$defs` leaves `$ref` dangling and breaks the tool call.

    This is the shape pydantic-backed MCP servers emit for a nested model, and
    it is by far the most common way `$ref` reaches the proxy. With `$defs`
    gone the parser cannot type `query`, forwards the model's JSON as a string,
    and the server answers "is not of type 'object'".
    """
    defs = {'Q': {'type': 'object', 'properties': {'domain': {'type': 'string'}}, 'required': ['domain']}}
    params = _params_via_request(
        {
            '$defs': defs,
            'type': 'object',
            'properties': {'query': {'$ref': '#/$defs/Q'}},
            'required': ['query'],
        }
    )

    assert params['$defs'] == defs
    assert params['properties']['query'] == {'$ref': '#/$defs/Q'}


def test_draft07_definitions_survive_request_validation():
    """Draft-07 servers anchor `$ref` at `definitions` rather than `$defs`."""
    defs = {'Q': {'type': 'object', 'properties': {'domain': {'type': 'string'}}}}
    params = _params_via_request(
        {
            'definitions': defs,
            'type': 'object',
            'properties': {'query': {'$ref': '#/definitions/Q'}},
        }
    )

    assert params['definitions'] == defs


def test_other_schema_keywords_survive_request_validation():
    """The drop was indiscriminate: any keyword outside the three declared fields.

    `additionalProperties` is expected to be absent -- `_clean_schema` strips it
    deliberately for Vertex, which is a separate and intended behaviour.
    """
    params = _params_via_request(
        {
            'type': 'object',
            'title': 'ToolArguments',
            'properties': {'x': {'type': 'string'}},
            'allOf': [{'required': ['x']}],
            'patternProperties': {'^x_': {'type': 'string'}},
            'additionalProperties': False,
        }
    )

    assert params['title'] == 'ToolArguments'
    assert params['allOf'] == [{'required': ['x']}]
    assert params['patternProperties'] == {'^x_': {'type': 'string'}}
    assert 'additionalProperties' not in params


def test_declared_fields_still_round_trip_through_request():
    schema = {'type': 'object', 'properties': {'x': {'type': 'string'}}, 'required': ['x']}
    assert _params_via_request(schema) == schema


def test_native_schema_preservation_is_explicit_and_does_not_materialize_defaults():
    from copy import deepcopy

    for schema in [
        {},
        {'type': 'object', 'properties': {}},
        {'type': 'object', 'properties': {}, 'required': []},
        {
            '$schema': 'https://json-schema.org/draft/2020-12/schema',
            'type': 'object',
            'additionalProperties': False,
            'properties': {'mode': {'type': 'string', 'default': 'native'}},
        },
    ]:
        original = deepcopy(schema)
        request = MessagesRequest.model_validate(
            {
                'model': 'test-model',
                'messages': [{'role': 'user', 'content': 'task'}],
                'tools': [{'name': 'fixture', 'description': 'Native tool.', 'input_schema': schema}],
            }
        )
        result = anthropic_to_openai(request, preserve_tool_schema=True)
        assert result['tools'][0]['function']['parameters'] == original
        result['tools'][0]['function']['parameters']['unexpected'] = True
        assert schema == original
        assert anthropic_to_openai(request, preserve_tool_schema=True)['tools'][0]['function']['parameters'] == original


def test_schema_preservation_does_not_change_messages_or_model_routing():
    request = MessagesRequest.model_validate(
        {
            'model': 'test-model',
            'system': 'System',
            'messages': [{'role': 'user', 'content': 'task'}],
            'tools': [
                {'name': 'fixture', 'description': 'Native tool.', 'input_schema': {'type': 'object', 'properties': {}}}
            ],
        }
    )
    legacy = anthropic_to_openai(request)
    native = anthropic_to_openai(request, preserve_tool_schema=True)
    legacy.pop('tools')
    native.pop('tools')
    assert legacy == native
