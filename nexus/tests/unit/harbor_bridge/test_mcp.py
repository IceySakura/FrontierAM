import pytest

from nexus.core.tool import MCPServerSpec
from nexus.extensions.harbor_bridge.mcp import (
    T3MCPServerConfig,
    mcp_agent_kwargs,
    translate_mcp_to_nexus,
)


def test_translate_http_keeps_url_headers_timeout_and_routing():
    servers = [
        {
            'name': 'canyon',
            'transport': 'streamable-http',
            'url': 'http://localhost:9900/mcp',
            'headers': {'Authorization': '${MCP_AUTHORIZATION}', 'X-Source': 't3-harbor'},
            'timeout_s': 30.0,
            'runtime_name': 'main',
        }
    ]

    result = translate_mcp_to_nexus(servers)

    assert len(result) == 1
    spec = MCPServerSpec.model_validate(result[0])
    assert spec.name == 'canyon'
    assert spec.client.transport.value == 'streamable_http'
    assert spec.client.url == 'http://localhost:9900/mcp'
    assert spec.client.headers == {
        'Authorization__env': 'MCP_AUTHORIZATION',
        'X-Source': 't3-harbor',
    }
    assert spec.client.timeout_s == 30.0
    assert spec.runtime_name == 'main'
    assert spec.nexus_managed is False


def test_translate_sse_keeps_transport():
    result = translate_mcp_to_nexus([
        {
            'name': 'events',
            'transport': 'sse',
            'url': 'http://localhost/events',
        }
    ])

    assert MCPServerSpec.model_validate(result[0]).client.transport.value == 'sse'


def test_translate_stdio_keeps_t3_extension_fields():
    result = translate_mcp_to_nexus([
        {
            'name': 'word',
            'transport': 'stdio',
            'command': 'uv',
            'args': ['run', 'x'],
            'env': {'MODE': 'test'},
            'cwd': '/workspace/project',
            'timeout_s': 60.0,
            'prefix': 'word_',
            'runtime_name': 'main',
            'nexus_managed': False,
        }
    ])

    spec = MCPServerSpec.model_validate(result[0])
    assert spec.client.transport.value == 'stdio'
    assert spec.client.command == 'uv'
    assert spec.client.args == ['run', 'x']
    assert spec.client.env == {'MODE': 'test'}
    assert spec.client.cwd == '/workspace/project'
    assert spec.client.timeout_s == 60.0
    assert spec.prefix == 'word_'
    assert spec.runtime_name == 'main'
    assert spec.nexus_managed is False


def test_translate_real_toolathlon_shape():
    result = translate_mcp_to_nexus([
        {
            'name': 'terminal',
            'runtime_name': 'main',
            'transport': 'stdio',
            'command': 'bash',
            'args': ['-c', 'exec uv --directory /opt/server run server'],
            'env': {'PG_HOST': '127.0.0.1', 'PG_PORT': '5432'},
            'cwd': '/workspace/mcp-benchmark/case/agent_workspace',
            'timeout_s': 100.0,
        }
    ])

    spec = MCPServerSpec.model_validate(result[0])
    assert spec.client.command == 'bash'
    assert spec.client.env['PG_HOST'] == '127.0.0.1'
    assert spec.client.cwd == '/workspace/mcp-benchmark/case/agent_workspace'
    assert spec.client.timeout_s == 100.0
    assert spec.runtime_name == 'main'
    assert spec.nexus_managed is False


def test_http_input_alias_uses_t3_normalization():
    config = {'environment': {'mcp_servers': [{'name': 'a', 'transport': 'http', 'url': 'http://h/mcp'}]}}

    result = mcp_agent_kwargs(config)

    assert result['mcp_servers'][0]['client']['transport'] == 'streamable_http'


def test_translate_allows_mock_secret_placeholders():
    result = translate_mcp_to_nexus([
        {
            'name': 'mock-header',
            'transport': 'streamable-http',
            'url': 'http://localhost/mcp',
            'headers': {'Authorization': 'placeholder', 'X-Source': 'mock'},
        },
        {
            'name': 'mock-env',
            'transport': 'stdio',
            'command': 'python',
            'env': {'API_TOKEN': 'placeholder', 'TOKEN_PATH': ''},
        },
    ])

    http = MCPServerSpec.model_validate(result[0])
    stdio = MCPServerSpec.model_validate(result[1])
    assert http.client.headers == {'Authorization': 'placeholder', 'X-Source': 'mock'}
    assert stdio.client.env == {'API_TOKEN': 'placeholder', 'TOKEN_PATH': ''}


def test_translate_preserves_literal_mcp_values():
    result = translate_mcp_to_nexus([
        {
            'name': 'literal-header',
            'transport': 'streamable-http',
            'url': 'http://localhost/mcp',
            'headers': {'Authorization': 'Bearer literal'},
        },
        {
            'name': 'literal-env',
            'transport': 'stdio',
            'command': 'python',
            'env': {'API_TOKEN': 'literal'},
        },
    ])

    http = MCPServerSpec.model_validate(result[0])
    stdio = MCPServerSpec.model_validate(result[1])
    assert http.client.headers == {'Authorization': 'Bearer literal'}
    assert stdio.client.env == {'API_TOKEN': 'literal'}


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('runtime_name', 'verifier'),
        ('nexus_managed', True),
    ],
)
def test_task_cannot_override_nexus_mcp_routing(field, value):
    server = {'name': 'a', 'transport': 'stdio', 'command': 'node', field: value}

    with pytest.raises(ValueError, match=field):
        translate_mcp_to_nexus([server])


def test_task_mcp_requires_explicit_transport():
    config = {'environment': {'mcp_servers': [{'name': 'a', 'url': 'http://h/mcp'}]}}

    with pytest.raises(ValueError, match='transport must be explicitly set'):
        mcp_agent_kwargs(config)


def test_task_mcp_accepts_legacy_transport_alias():
    config = {'environment': {'mcp_servers': [{'name': 'a', 'transport': 'streamable_http', 'url': 'http://h/mcp'}]}}

    result = mcp_agent_kwargs(config)

    assert result['mcp_servers'][0]['client']['transport'] == 'streamable_http'


def test_task_mcp_requires_array_of_tables():
    with pytest.raises(ValueError, match='must be an array of tables'):
        mcp_agent_kwargs({'environment': {'mcp_servers': {'name': 'a'}}})


def test_agent_kwargs_empty_when_task_declares_no_server():
    assert mcp_agent_kwargs({}) == {}
    assert mcp_agent_kwargs({'environment': {'mcp_servers': []}}) == {}


@pytest.mark.parametrize(
    ('server', 'message'),
    [
        ({'name': '', 'transport': 'stdio', 'command': 'node'}, 'name must contain'),
        ({'name': 'a b', 'transport': 'stdio', 'command': 'node'}, 'name must contain'),
        ({'name': 'a', 'transport': 'stdio', 'command': 'node', 'cwd': 'relative'}, 'absolute container path'),
        ({'name': 'a', 'transport': 'stdio', 'command': 'node', 'timeout_s': 0.0}, 'positive finite'),
        ({'name': 'a', 'transport': 'stdio', 'command': 'node', 'unknown': True}, 'Extra inputs'),
    ],
)
def test_task_mcp_rejects_invalid_t3_extensions(server, message):
    with pytest.raises(ValueError, match=message):
        translate_mcp_to_nexus([server])


def test_task_mcp_rejects_duplicate_names():
    server = {'name': 'same', 'transport': 'stdio', 'command': 'node'}

    with pytest.raises(ValueError, match='duplicates MCP server'):
        translate_mcp_to_nexus([server, server])


def test_explicit_t3_model_translates_without_reparsing():
    server = T3MCPServerConfig(
        name='a',
        transport='stdio',
        command='node',
        env={'API_TOKEN': '${MCP_API_TOKEN}'},
    )

    result = translate_mcp_to_nexus([server])[0]
    assert result['name'] == 'a'
    assert result['client']['env'] == {'API_TOKEN__env': 'MCP_API_TOKEN'}


def test_task_mcp_translation_ignores_legacy_host_toggle(monkeypatch):
    monkeypatch.setenv('HARBOR_MCP_INJECT', 'off')
    config = {'environment': {'mcp_servers': [{'name': 'a', 'transport': 'http', 'url': 'http://h/mcp'}]}}

    assert [server['name'] for server in mcp_agent_kwargs(config)['mcp_servers']] == ['a']
