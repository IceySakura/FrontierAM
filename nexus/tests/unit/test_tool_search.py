"""Tests for client-side Tool Search."""

from __future__ import annotations

import json
import sys
import types

from typing import Any

import pytest

from pydantic import BaseModel, ValidationError

from nexus.core.agent import WithToolSearch
from nexus.core.context import ExecutionContext
from nexus.core.tool import (
    ToolContext,
    ToolExecutor,
    ToolExposure,
    ToolMeta,
    ToolResponse,
    ToolSearchConfig,
    ToolType,
)


_EXT_KEY = 'nexus.extensions'
if _EXT_KEY not in sys.modules:
    _stub = types.ModuleType(_EXT_KEY)
    import nexus as _nexus_pkg

    _stub.__path__ = [p + '/extensions' for p in _nexus_pkg.__path__]  # type: ignore[union-attr]
    sys.modules[_EXT_KEY] = _stub


class _DummyInput(BaseModel):
    value: str = 'default'


class _RequiredInput(BaseModel):
    value: str


def _make_tool_func(
    name: str,
    *,
    registry_name: str | None = None,
    description: str | None = None,
    input_model: type[BaseModel] = _DummyInput,
) -> Any:
    async def _tool(params: Any, context: ToolContext) -> ToolResponse:
        return ToolResponse(output=[{'type': 'text', 'text': f'{name}:{params.value}'}])

    _tool.__tool_input_model__ = input_model
    _tool.__tool_meta__ = ToolMeta(
        name=name,
        description=description or f'Tool {name}',
        tool_type=ToolType.LOCAL,
        input_model=input_model,
        func=_tool,
        parallel_func=None,
    )
    _tool.nexus_meta = {'type': 'tool', 'name': registry_name or name}
    return _tool


def _output_json(response: ToolResponse) -> dict[str, Any]:
    part = response.output[0]
    assert isinstance(part, dict)
    assert part['type'] == 'text'
    return json.loads(part['text'])


class _InputWithToolSearch(WithToolSearch):
    value: str


class _FakeMCPTool:
    full_name = 'mcp_lookup'
    tool_name = 'lookup'
    description = 'Lookup records from a fake MCP server'
    input_schema = {
        'type': 'object',
        'properties': {
            'query': {'type': 'string'},
        },
        'required': ['query'],
    }


class _FakeMCPManager:
    def __init__(self) -> None:
        self.tools = {'mcp_lookup': _FakeMCPTool()}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    async def call_tool(self, full_name: str, arguments: dict[str, Any] | None = None) -> ToolResponse:
        call_arguments = arguments or {}
        self.calls.append((full_name, call_arguments))
        return ToolResponse(output=[{'type': 'text', 'text': f'{full_name}:{call_arguments["query"]}'}])

    async def close(self) -> None:
        self.closed = True
        self.tools.clear()


class _FakeMCPToolWithServer:
    def __init__(self, full_name: str, server_name: str, tool_name: str) -> None:
        self.full_name = full_name
        self.server_name = server_name
        self.tool_name = tool_name
        self.description = f'{server_name} {tool_name}'
        self.input_schema = {'type': 'object', 'properties': {}}


class _MultiServerMCPManager:
    def __init__(self) -> None:
        self.tools = {
            'github_create_issue': _FakeMCPToolWithServer('github_create_issue', 'github', 'create_issue'),
            'github_list_repos': _FakeMCPToolWithServer('github_list_repos', 'github', 'list_repos'),
            'slack_post': _FakeMCPToolWithServer('slack_post', 'slack', 'post'),
        }

    async def close(self) -> None:
        self.tools.clear()


def test_tool_search_config_defaults():
    config = ToolSearchConfig()

    assert config.enabled is False
    assert config.exposure_overrides == {}
    assert config.defer_all_mcp is False


def test_tool_search_config_parses_exposure_values():
    config = ToolSearchConfig.model_validate({
        'enabled': True,
        'exposure_overrides': {'alpha': 'deferred', 'beta': 'direct'},
        'defer_all_mcp': True,
    })

    assert config.enabled is True
    assert config.exposure_overrides['alpha'] == ToolExposure.DEFERRED
    assert config.exposure_overrides['beta'] == ToolExposure.DIRECT
    assert config.defer_all_mcp is True


def test_with_tool_search_mixin_accepts_config_dict():
    value = _InputWithToolSearch.model_validate({
        'value': 'x',
        'tool_search': {'enabled': True, 'exposure_overrides': {'alpha': 'deferred'}},
    })

    assert value.tool_search is not None
    assert value.tool_search.enabled is True
    assert value.tool_search.exposure_overrides['alpha'] == ToolExposure.DEFERRED


def test_tool_context_accepts_executor_reference():
    executor = ToolExecutor([])
    context = ToolContext(executor=executor, tool_configs={}, tool_state={})

    assert context.executor is executor
    with pytest.raises(TypeError):
        context.tool_configs['x'] = 'y'


def test_bridge_tools_are_registered():
    import nexus.extensions.tools.tool_search  # noqa: F401

    from nexus.core import tool_registry

    assert tool_registry.get('tool_search').nexus_meta['name'] == 'tool_search'
    assert tool_registry.get('tool_describe').nexus_meta['name'] == 'tool_describe'
    assert tool_registry.get('tool_call').nexus_meta['name'] == 'tool_call'


async def test_tool_search_enabled_exposes_direct_and_bridge_only():
    direct = _make_tool_func('direct_tool')
    deferred = _make_tool_func('deferred_tool')
    executor = ToolExecutor(
        [direct, deferred],
        allowed_tool_names=['direct_tool', 'deferred_tool'],
        tool_search_config=ToolSearchConfig(
            enabled=True,
            exposure_overrides={'deferred_tool': ToolExposure.DEFERRED},
        ),
    )

    schema_names = [schema['function']['name'] for schema in executor.get_tool_schemas()]

    assert schema_names == ['direct_tool', 'tool_search', 'tool_describe', 'tool_call']
    assert executor.available_schema_names() == ['direct_tool', 'tool_call', 'tool_describe', 'tool_search']


async def test_non_bridge_tool_context_does_not_include_executor_when_tool_search_enabled():
    captured_contexts: list[ToolContext] = []

    async def _tool(params: Any, context: ToolContext) -> ToolResponse:
        captured_contexts.append(context)
        return ToolResponse(output=[{'type': 'text', 'text': f'alpha:{params.value}'}])

    _tool.__tool_input_model__ = _DummyInput
    _tool.__tool_meta__ = ToolMeta(
        name='alpha',
        description='Tool alpha',
        tool_type=ToolType.LOCAL,
        input_model=_DummyInput,
        func=_tool,
        parallel_func=None,
    )
    _tool.nexus_meta = {'type': 'tool', 'name': 'alpha'}

    executor = ToolExecutor(
        [_tool],
        allowed_tool_names=['alpha'],
        tool_search_config=ToolSearchConfig(enabled=True),
    )

    ctx = ExecutionContext(openai_clients={})
    async with ctx:
        await executor.call('alpha', {'value': 'ok'})

    assert captured_contexts[0].executor is None
    assert captured_contexts[0].tool_configs['_deferred_tool_names'] == []


async def test_bridge_tool_context_includes_executor_when_tool_search_enabled():
    import nexus.extensions.tools.tool_search  # noqa: F401

    executor = ToolExecutor([], allowed_tool_names=[], tool_search_config=ToolSearchConfig(enabled=True))

    ctx = ExecutionContext(openai_clients={})
    async with ctx:
        response = await executor.call('tool_search', {'query': 'anything', 'max_results': 10})

    payload = _output_json(response)
    assert payload == {'matches': [], 'total': 0}


async def test_tool_search_returns_single_schema_on_unique_match():
    import nexus.extensions.tools.tool_search  # noqa: F401

    from nexus.core import tool_registry

    alpha = _make_tool_func('alpha_search', description='Search project files quickly')
    beta = _make_tool_func('beta_browser', description='Click browser buttons')
    executor = ToolExecutor(
        [beta, alpha],
        allowed_tool_names=['beta_browser', 'alpha_search'],
        tool_search_config=ToolSearchConfig(
            enabled=True,
            exposure_overrides={
                'alpha_search': ToolExposure.DEFERRED,
                'beta_browser': ToolExposure.DEFERRED,
            },
        ),
    )

    ctx = ExecutionContext(openai_clients={})
    async with ctx:
        search = tool_registry.get('tool_search')
        single_response = await search(
            search.__tool_input_model__.model_validate({'query': 'project files', 'max_results': 5}),
            ToolContext(executor=executor, tool_configs={}, tool_state={}),
        )
        single_payload = _output_json(single_response)

    assert single_payload['total'] == 1
    assert single_payload['matches'][0]['name'] == 'alpha_search'
    assert 'input_schema' in single_payload['matches'][0]


async def test_tool_search_honors_model_requested_max_results():
    import nexus.extensions.tools.tool_search  # noqa: F401

    from nexus.core import tool_registry

    tools = [_make_tool_func(f'tool_{i}', description=f'Tool number {i}') for i in range(5)]
    executor = ToolExecutor(
        tools,
        allowed_tool_names=[f'tool_{i}' for i in range(5)],
        tool_search_config=ToolSearchConfig(
            enabled=True,
            exposure_overrides={f'tool_{i}': ToolExposure.DEFERRED for i in range(5)},
        ),
    )

    ctx = ExecutionContext(openai_clients={})
    async with ctx:
        search = tool_registry.get('tool_search')
        response = await search(
            search.__tool_input_model__.model_validate({'query': 'number', 'max_results': 2}),
            ToolContext(executor=executor, tool_configs={}, tool_state={}),
        )
        payload = _output_json(response)

    # The model decides how many matches it wants; total still reflects all deferred tools.
    assert payload['total'] == 5
    assert len(payload['matches']) == 2


async def test_tool_search_matches_acronym_and_camelcase_names():
    import nexus.extensions.tools.tool_search  # noqa: F401

    from nexus.core import tool_registry

    http = _make_tool_func('HTTPServer', description='Serve requests')
    reader = _make_tool_func('readFileV2', description='Read a file')
    executor = ToolExecutor(
        [http, reader],
        allowed_tool_names=['HTTPServer', 'readFileV2'],
        tool_search_config=ToolSearchConfig(
            enabled=True,
            exposure_overrides={'HTTPServer': ToolExposure.DEFERRED, 'readFileV2': ToolExposure.DEFERRED},
        ),
    )

    ctx = ExecutionContext(openai_clients={})
    async with ctx:
        search = tool_registry.get('tool_search')
        server_resp = await search(
            search.__tool_input_model__.model_validate({'query': 'server', 'max_results': 10}),
            ToolContext(executor=executor, tool_configs={}, tool_state={}),
        )
        read_resp = await search(
            search.__tool_input_model__.model_validate({'query': 'file', 'max_results': 10}),
            ToolContext(executor=executor, tool_configs={}, tool_state={}),
        )

    # Acronym ("HTTPServer" -> "server") and camelCase ("readFileV2" -> "file")
    # tokens are matched, total == 1 auto-describes.
    assert _output_json(server_resp)['matches'][0]['name'] == 'HTTPServer'
    assert _output_json(read_resp)['matches'][0]['name'] == 'readFileV2'


async def test_tool_search_required_term_does_not_match_unrelated_substring():
    import nexus.extensions.tools.tool_search  # noqa: F401

    from nexus.core import tool_registry

    sheet = _make_tool_func('spreadsheet_export', description='Export spreadsheets')
    executor = ToolExecutor(
        [sheet],
        allowed_tool_names=['spreadsheet_export'],
        tool_search_config=ToolSearchConfig(
            enabled=True,
            exposure_overrides={'spreadsheet_export': ToolExposure.DEFERRED},
        ),
    )

    ctx = ExecutionContext(openai_clients={})
    async with ctx:
        search = tool_registry.get('tool_search')
        response = await search(
            # "+read" is a hard requirement; "read" is only a substring of
            # "sp-read-sheet" and must NOT count as a match.
            search.__tool_input_model__.model_validate({'query': '+read', 'max_results': 10}),
            ToolContext(executor=executor, tool_configs={}, tool_state={}),
        )
        payload = _output_json(response)

    assert payload['total'] == 0
    assert payload['matches'] == []


async def test_unknown_exposure_override_is_ignored():
    # An override naming a tool that does not exist has no effect: it is simply
    # not matched at call time, so construction succeeds and real tools stay visible.
    local_tool = _make_tool_func('local_tool')
    executor = ToolExecutor(
        [local_tool],
        allowed_tool_names=['local_tool'],
        tool_search_config=ToolSearchConfig(
            enabled=True,
            exposure_overrides={'nope': ToolExposure.DEFERRED},
        ),
    )

    ctx = ExecutionContext(openai_clients={})
    async with ctx:
        schema_names = [schema['function']['name'] for schema in executor.get_tool_schemas()]

    assert 'local_tool' in schema_names
    assert executor.get_deferred_tool_infos() == []


async def test_tool_describe_and_tool_call_use_schema_visible_name():
    import nexus.extensions.tools.tool_search  # noqa: F401

    from nexus.core import tool_registry

    deferred = _make_tool_func('original_name', registry_name='pkg/original')
    executor = ToolExecutor(
        [deferred],
        allowed_tool_names=['pkg/original'],
        tool_search_config=ToolSearchConfig(
            enabled=True,
            exposure_overrides={'pkg/original': ToolExposure.DEFERRED},
        ),
    )
    executor.set_tool_override('pkg/original', name='VisibleTool')

    describe = tool_registry.get('tool_describe')
    call = tool_registry.get('tool_call')
    context = ToolContext(executor=executor, tool_configs={}, tool_state={})

    ctx = ExecutionContext(openai_clients={})
    async with ctx:
        describe_response = await describe(
            describe.__tool_input_model__.model_validate({'tool_name': 'VisibleTool'}),
            context,
        )
        call_response = await call(
            call.__tool_input_model__.model_validate({
                'tool_name': 'VisibleTool',
                'params': {'value': 'ok'},
            }),
            context,
        )

    describe_payload = _output_json(describe_response)
    assert describe_payload['name'] == 'VisibleTool'
    assert describe_payload['input_schema']['type'] == 'object'
    assert call_response.output[0]['text'] == 'original_name:ok'


async def test_tool_call_returns_model_correctable_validation_error():
    import nexus.extensions.tools.tool_search  # noqa: F401

    from nexus.core import tool_registry

    deferred = _make_tool_func('needs_value', input_model=_RequiredInput)
    executor = ToolExecutor(
        [deferred],
        allowed_tool_names=['needs_value'],
        tool_search_config=ToolSearchConfig(
            enabled=True,
            exposure_overrides={'needs_value': ToolExposure.DEFERRED},
        ),
    )

    call = tool_registry.get('tool_call')
    ctx = ExecutionContext(openai_clients={})
    async with ctx:
        response = await call(
            call.__tool_input_model__.model_validate({'tool_name': 'needs_value', 'params': {}}),
            ToolContext(executor=executor, tool_configs={}, tool_state={}),
        )

    assert response.output[0]['text'].startswith("Tool error: Invalid arguments for tool 'needs_value'")
    assert 'required=[value: string]' in response.output[0]['text']


def test_tool_search_rejects_direct_deferred_schema_name_collision():
    direct = _make_tool_func('shared', registry_name='direct')
    deferred = _make_tool_func('shared', registry_name='deferred')
    executor = ToolExecutor(
        [direct, deferred],
        allowed_tool_names=['direct', 'deferred'],
        tool_search_config=ToolSearchConfig(
            enabled=True,
            exposure_overrides={'deferred': ToolExposure.DEFERRED},
        ),
    )

    with pytest.raises(ValueError, match='Duplicate schema tool name'):
        executor.get_tool_schemas()


def test_tool_search_available_schema_names_rejects_direct_deferred_schema_name_collision():
    direct = _make_tool_func('shared', registry_name='direct')
    deferred = _make_tool_func('shared', registry_name='deferred')
    executor = ToolExecutor(
        [direct, deferred],
        allowed_tool_names=['direct', 'deferred'],
        tool_search_config=ToolSearchConfig(
            enabled=True,
            exposure_overrides={'deferred': ToolExposure.DEFERRED},
        ),
    )

    with pytest.raises(ValueError, match='Duplicate schema tool name'):
        executor.available_schema_names()


def test_tool_search_validate_tool_call_rejects_direct_deferred_schema_name_collision():
    direct = _make_tool_func('shared', registry_name='direct')
    deferred = _make_tool_func('shared', registry_name='deferred')
    executor = ToolExecutor(
        [direct, deferred],
        allowed_tool_names=['direct', 'deferred'],
        tool_search_config=ToolSearchConfig(
            enabled=True,
            exposure_overrides={'deferred': ToolExposure.DEFERRED},
        ),
    )

    with pytest.raises(ValueError, match='Duplicate schema tool name'):
        executor.validate_tool_call('shared', {'value': 'x'})


def test_tool_search_deferred_infos_rejects_direct_deferred_schema_name_collision():
    direct = _make_tool_func('shared', registry_name='direct')
    deferred = _make_tool_func('shared', registry_name='deferred')
    executor = ToolExecutor(
        [direct, deferred],
        allowed_tool_names=['direct', 'deferred'],
        tool_search_config=ToolSearchConfig(
            enabled=True,
            exposure_overrides={'deferred': ToolExposure.DEFERRED},
        ),
    )

    with pytest.raises(ValueError, match='Duplicate schema tool name'):
        executor.get_deferred_tool_infos()


def test_tool_search_describe_rejects_direct_deferred_schema_name_collision():
    direct = _make_tool_func('shared', registry_name='direct')
    deferred = _make_tool_func('shared', registry_name='deferred')
    executor = ToolExecutor(
        [direct, deferred],
        allowed_tool_names=['direct', 'deferred'],
        tool_search_config=ToolSearchConfig(
            enabled=True,
            exposure_overrides={'deferred': ToolExposure.DEFERRED},
        ),
    )

    with pytest.raises(ValueError, match='Duplicate schema tool name'):
        executor.describe_deferred_tool('shared')


def test_tool_search_rejects_deferred_bridge_schema_name_collision():
    deferred = _make_tool_func('tool_call', registry_name='deferred')
    executor = ToolExecutor(
        [deferred],
        allowed_tool_names=['deferred'],
        tool_search_config=ToolSearchConfig(
            enabled=True,
            exposure_overrides={'deferred': ToolExposure.DEFERRED},
        ),
    )

    with pytest.raises(ValueError, match='Duplicate schema tool name'):
        executor.get_tool_schemas()


async def test_deferred_tool_requires_call_discovered_tool():
    deferred = _make_tool_func('deferred_tool')
    executor = ToolExecutor(
        [deferred],
        allowed_tool_names=['deferred_tool'],
        tool_search_config=ToolSearchConfig(
            enabled=True,
            exposure_overrides={'deferred_tool': ToolExposure.DEFERRED},
        ),
    )

    with pytest.raises(KeyError):
        executor.validate_tool_call('deferred_tool', {'value': 'x'})

    ctx = ExecutionContext(openai_clients={})
    async with ctx:
        response = await executor.call_discovered_tool('deferred_tool', {'value': 'x'})

    assert response.output[0]['text'] == 'deferred_tool:x'


async def test_defer_all_mcp_exposes_bridge_tools_and_discovers_mcp_tools():
    local_tool = _make_tool_func('local_tool')
    executor = ToolExecutor(
        [local_tool],
        allowed_tool_names=['local_tool', 'mcp_lookup'],
        tool_search_config=ToolSearchConfig(enabled=True, defer_all_mcp=True),
    )
    manager = _FakeMCPManager()

    ctx = ExecutionContext(openai_clients={})
    async with ctx:
        ctx._mcp_manager = manager
        ctx.mcp_tools = dict(manager.tools)
        schema_names = [schema['function']['name'] for schema in executor.get_tool_schemas()]
        deferred_infos = executor.get_deferred_tool_infos()
        response = await executor.call_discovered_tool('mcp_lookup', {'query': 'abc'})

    assert schema_names == ['local_tool', 'tool_search', 'tool_describe', 'tool_call']
    assert deferred_infos == [
        {
            'name': 'mcp_lookup',
            'canonical_name': 'mcp_lookup',
            'description': 'Lookup records from a fake MCP server',
            'source': 'mcp',
        }
    ]
    assert manager.calls == [('mcp_lookup', {'query': 'abc'})]
    assert manager.closed is True
    assert manager.tools == {}
    assert ctx._mcp_manager is None
    assert ctx.mcp_tools == {}
    assert response.output[0]['text'] == 'mcp_lookup:abc'


async def test_call_discovered_tool_rejects_bridge_tools():
    executor = ToolExecutor(
        [],
        allowed_tool_names=[],
        tool_search_config=ToolSearchConfig(enabled=True),
    )

    ctx = ExecutionContext(openai_clients={})
    async with ctx:
        with pytest.raises(KeyError) as exc_info:
            await executor.call_discovered_tool('tool_call', {'tool_name': 'tool_call', 'params': {}})

    assert 'tool_call' not in exc_info.value.available_schema_names


async def test_call_discovered_tool_unknown_name_lists_only_discovered_tools():
    deferred = _make_tool_func('deferred_tool')
    executor = ToolExecutor(
        [deferred],
        allowed_tool_names=['deferred_tool'],
        tool_search_config=ToolSearchConfig(
            enabled=True,
            exposure_overrides={'deferred_tool': ToolExposure.DEFERRED},
        ),
    )

    ctx = ExecutionContext(openai_clients={})
    async with ctx:
        with pytest.raises(KeyError) as exc_info:
            await executor.call_discovered_tool('missing_tool', {})

    assert exc_info.value.available_schema_names == ['deferred_tool']


def test_describe_deferred_tool_returns_input_schema_only():
    deferred = _make_tool_func('deferred_tool')
    executor = ToolExecutor(
        [deferred],
        allowed_tool_names=['deferred_tool'],
        tool_search_config=ToolSearchConfig(
            enabled=True,
            exposure_overrides={'deferred_tool': ToolExposure.DEFERRED},
        ),
    )

    info = executor.describe_deferred_tool('deferred_tool')

    assert 'input_schema' in info
    assert info['input_schema']['type'] == 'object'
    assert 'schema' not in info


async def test_tool_describe_does_not_swallow_executor_runtime_errors():
    import nexus.extensions.tools.tool_search  # noqa: F401

    from nexus.core import tool_registry

    class BrokenExecutor:
        def describe_deferred_tool(self, tool_name: str) -> dict[str, Any]:
            raise RuntimeError('boom')

    describe = tool_registry.get('tool_describe')
    with pytest.raises(RuntimeError, match='boom'):
        await describe(
            describe.__tool_input_model__.model_validate({'tool_name': 'anything'}),
            ToolContext(executor=BrokenExecutor(), tool_configs={}, tool_state={}),
        )


def test_tool_search_input_max_results_bounds():
    from nexus.extensions.tools.tool_search.tools import ToolSearchInput

    # 默认值降到 5
    assert ToolSearchInput().max_results == 5
    # 合法边界:1 与 10
    assert ToolSearchInput(max_results=1).max_results == 1
    assert ToolSearchInput(max_results=10).max_results == 10
    # 越界:0(低于 ge=1)与 11(高于 le=10)被拒
    with pytest.raises(ValidationError):
        ToolSearchInput(max_results=0)
    with pytest.raises(ValidationError):
        ToolSearchInput(max_results=11)


def test_tool_search_config_inject_reminder_default_true():
    assert ToolSearchConfig().inject_reminder is True
    assert ToolSearchConfig.model_validate({'inject_reminder': False}).inject_reminder is False


async def test_tool_search_star_query_returns_no_matches():
    import nexus.extensions.tools.tool_search  # noqa: F401

    from nexus.core import tool_registry

    alpha = _make_tool_func('alpha_search', description='Search project files quickly')
    beta = _make_tool_func('beta_browser', description='Click browser buttons')
    executor = ToolExecutor(
        [beta, alpha],
        allowed_tool_names=['beta_browser', 'alpha_search'],
        tool_search_config=ToolSearchConfig(
            enabled=True,
            exposure_overrides={
                'alpha_search': ToolExposure.DEFERRED,
                'beta_browser': ToolExposure.DEFERRED,
            },
        ),
    )

    ctx = ExecutionContext(openai_clients={})
    async with ctx:
        search = tool_registry.get('tool_search')
        response = await search(
            search.__tool_input_model__.model_validate({'query': '*', 'max_results': 5}),
            ToolContext(executor=executor, tool_configs={}, tool_state={}),
        )
        payload = _output_json(response)

    assert payload['total'] == 0
    assert payload['matches'] == []


async def test_build_reminder_lists_local_tools_and_mcp_server_counts():
    local_tool = _make_tool_func('grep_search', description='Search code')
    executor = ToolExecutor(
        [local_tool],
        allowed_tool_names=['grep_search', 'github_create_issue', 'github_list_repos', 'slack_post'],
        tool_search_config=ToolSearchConfig(
            enabled=True,
            defer_all_mcp=True,
            exposure_overrides={'grep_search': ToolExposure.DEFERRED},
        ),
    )

    ctx = ExecutionContext(openai_clients={})
    async with ctx:
        ctx._mcp_manager = _MultiServerMCPManager()
        # mcp_tools mirrors the manager registry so allowed_tool_names resolves MCP tools.
        ctx.mcp_tools = dict(ctx._mcp_manager.tools)
        reminder = executor.build_tool_search_reminder()

    assert reminder is not None
    assert 'grep_search' in reminder
    assert 'github (2 tools)' in reminder
    assert 'slack (1 tool' in reminder
    assert 'create_issue' not in reminder
    assert 'list_repos' not in reminder
    assert 'slack_post' not in reminder


async def test_build_reminder_local_only_omits_mcp_line():
    local_tool = _make_tool_func('grep_search', description='Search code')
    executor = ToolExecutor(
        [local_tool],
        allowed_tool_names=['grep_search'],
        tool_search_config=ToolSearchConfig(
            enabled=True,
            exposure_overrides={'grep_search': ToolExposure.DEFERRED},
        ),
    )

    ctx = ExecutionContext(openai_clients={})
    async with ctx:
        reminder = executor.build_tool_search_reminder()

    assert reminder is not None
    assert 'grep_search' in reminder
    assert 'MCP servers' not in reminder


def test_build_reminder_returns_none_when_disabled():
    local_tool = _make_tool_func('grep_search')
    executor = ToolExecutor([local_tool], allowed_tool_names=['grep_search'])
    assert executor.build_tool_search_reminder() is None


def test_build_reminder_returns_none_when_inject_reminder_false():
    local_tool = _make_tool_func('grep_search')
    executor = ToolExecutor(
        [local_tool],
        allowed_tool_names=['grep_search'],
        tool_search_config=ToolSearchConfig(
            enabled=True,
            inject_reminder=False,
            exposure_overrides={'grep_search': ToolExposure.DEFERRED},
        ),
    )
    assert executor.build_tool_search_reminder() is None


def test_build_reminder_returns_none_when_no_deferred_tools():
    direct = _make_tool_func('direct_tool')
    executor = ToolExecutor(
        [direct],
        allowed_tool_names=['direct_tool'],
        tool_search_config=ToolSearchConfig(enabled=True),
    )
    assert executor.build_tool_search_reminder() is None


async def test_build_reminder_mcp_only_omits_hidden_tools_line():
    executor = ToolExecutor(
        [],
        allowed_tool_names=['github_create_issue', 'github_list_repos', 'slack_post'],
        tool_search_config=ToolSearchConfig(enabled=True, defer_all_mcp=True),
    )

    ctx = ExecutionContext(openai_clients={})
    async with ctx:
        ctx._mcp_manager = _MultiServerMCPManager()
        # mcp_tools mirrors the manager registry so allowed_tool_names resolves MCP tools.
        ctx.mcp_tools = dict(ctx._mcp_manager.tools)
        reminder = executor.build_tool_search_reminder()

    assert reminder is not None
    assert 'Hidden tools:' not in reminder
    assert 'github (2 tools)' in reminder
    assert 'slack (1 tool)' in reminder


def test_append_reminder_to_user_content_str_and_list_and_none():
    from nexus.extensions.agents.main.agent import _append_reminder_to_user_content

    # str user prompt → 换行拼接
    assert _append_reminder_to_user_content('do the thing', 'REMINDER') == 'do the thing\n\nREMINDER'

    # None reminder → 原样返回
    assert _append_reminder_to_user_content('do the thing', None) == 'do the thing'

    # list user prompt → 追加一个 text 段
    parts = [{'type': 'text', 'text': 'do the thing'}]
    result = _append_reminder_to_user_content(parts, 'REMINDER')
    assert result == [
        {'type': 'text', 'text': 'do the thing'},
        {'type': 'text', 'text': 'REMINDER'},
    ]
    # 原列表不被原地修改
    assert parts == [{'type': 'text', 'text': 'do the thing'}]

    # None reminder with list content returns the same object unchanged
    assert _append_reminder_to_user_content(parts, None) is parts


@pytest.mark.parametrize('include_schema', [False, True])
def test_context_agent_search_reports_required_parameters_only_for_its_executor(include_schema: bool) -> None:
    from nexus.extensions.agents.context_agent.tools.general.executor import _ContextAgentToolExecutor

    tool = _make_tool_func('required_tool', input_model=_RequiredInput)
    context_executor = _ContextAgentToolExecutor([tool])
    shared_executor = ToolExecutor([tool])
    info = context_executor._tool_info_for_canonical_name('required_tool', include_schema)
    assert info['required'] == ['value']
    assert ('input_schema' in info) is include_schema
    assert 'required' not in shared_executor._tool_info_for_canonical_name('required_tool', include_schema)
