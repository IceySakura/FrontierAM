"""Tools and MCP servers a run declares on top of its persona's own.

Both paths fail quietly if unguarded, and in the same way -- the model simply
never calls the tool:

* an MCP server offering a bare name a built-in tool already uses collides only
  when the schemas are built, which is every round rather than once;
* the schemas are captured at assembly, so a server loaded after that never
  reaches the model at all;
* a worker-side entry in the user-side shape is skipped by ``claude mcp add``
  with a log line and nothing else.

The declarations themselves are checked when a run is generated, so a typo stops
a batch instead of costing one.
"""

import asyncio
import json
import re

from pathlib import Path
from typing import Any

import pytest

from nexus.core import ExecutionContext
from nexus.core.tool.mcp.manager import MCPToolManager
from nexus.core.tool.mcp.models import ConnectedMCPTool, MCPServerSpec
from nexus.extensions.tasks.hitl.user_agent import mcp_config
from nexus.extensions.tasks.hitl.user_agent.personas import DEFAULT_TOOLS, get_persona, registered_personas
from nexus.extensions.tasks.hitl.user_agent.tools import toolset
from nexus.extensions.tasks.hitl.user_agent.tools import acting as acting_tools


def mcp_tool(full_name: str) -> ConnectedMCPTool:
    return ConnectedMCPTool(
        full_name=full_name,
        server_name='some-server',
        tool_name=full_name,
        description='a tool an MCP server offers',
        input_schema={'type': 'object', 'properties': {}},
        server_key='key',
    )


def with_mcp_tools(*full_names: str):
    """A context already holding these MCP tools, as ``load_mcp_tools`` leaves it."""

    async def run(body: Any) -> Any:
        async with ExecutionContext(tool_state={'working_dir': '/testbed'}) as context:
            manager = MCPToolManager(context)
            context._mcp_manager = manager
            for name in full_names:
                manager._tools[name] = mcp_tool(name)
            return body(context)

    return run


def write_json(path: Path, payload: Any) -> str:
    path.write_text(json.dumps(payload), encoding='utf-8')
    return str(path)


# -- tools a run asks for beyond the persona's -----------------------------


def test_a_tool_outside_both_tables_is_still_refused() -> None:
    """The error names what is available, so it says whether the tool exists at all."""
    with pytest.raises(toolset.ToolSetError) as caught:
        toolset.build(['send_message', 'not_a_tool'])

    message = str(caught.value)
    assert 'send_message' in message, 'the built-in tools are not named'
    assert 'str_replace_editor' in message, 'the editor is not named'


def test_a_declared_optional_tool_reaches_the_model() -> None:
    tools = toolset.build(['send_message', 'str_replace_editor'])

    assert 'str_replace_editor' in tools.schema_names


def test_the_editor_reads_the_working_dir_this_run_sets() -> None:
    """It is a tool from elsewhere in the tree, so the state key has to match.

    ``file_editor`` reads ``tool_state['working_dir']`` and otherwise falls back to
    ``/testbed``, which for this recipe is a different repository.
    """
    tools = toolset.build(['send_message', 'str_replace_editor'])

    state = toolset.tool_state(tools, working_dir='/workspace')

    assert state[acting_tools.WORKING_DIR_STATE] == '/workspace'


def test_extras_are_added_once() -> None:
    """``build`` refuses a name listed twice, so threading must not repeat one."""
    assert toolset.tool_names(('send_message',), {}, ('str_replace_editor',)) == [
        'send_message',
        'str_replace_editor',
    ]
    assert toolset.tool_names(('send_message', 'str_replace_editor'), {}, ('str_replace_editor',)) == [
        'send_message',
        'str_replace_editor',
    ]


def test_the_default_tools_can_look_without_being_able_to_write() -> None:
    """``read`` is the editor's ``view`` without the four commands that write.

    So a user gets the line range, the directory listing and the image -- the last is
    how it checks a rendered page or a chart -- and has no way to change the work. A
    persona that wants to change the work asks for the editor by name; what any one
    persona declares is its own business.
    """
    assert 'read' in DEFAULT_TOOLS
    assert 'str_replace_editor' not in DEFAULT_TOOLS, 'writing is asked for, not default'


@pytest.mark.skip(reason='the editor lists a directory by shelling out to find, which a CI image need not carry')
def test_read_views_a_directory_and_an_image_the_way_the_editor_does() -> None:
    """The three things ``cat -n`` could not do, which is why this shares ``view``.

    Reading an image as text gives a placeholder or a base64 blob; neither is the
    picture. A directory listing was not available at all.
    """
    import asyncio
    import tempfile

    directory = Path(tempfile.mkdtemp())
    (directory / 'a.py').write_text('one\ntwo\nthree\n', encoding='utf-8')
    (directory / 'sub').mkdir()
    (directory / 'shot.png').write_bytes(b'\x89PNG\r\n\x1a\n' + bytes(400))

    async def view(**kwargs: Any) -> Any:
        async with ExecutionContext(tool_state={'working_dir': str(directory)}) as context:
            return await acting_tools.read_tool(acting_tools.ReadInput(**kwargs), context)

    from nexus.extensions.tasks.hitl.user_agent.dispatch import content_of

    listing = content_of(asyncio.run(view(path='.')))
    assert isinstance(listing, str)
    assert 'sub' in listing, 'a directory is not listed'

    image = content_of(asyncio.run(view(path='shot.png')))
    assert isinstance(image, list), 'an image arrived as text'
    assert [part['type'] for part in image] == ['text', 'image_url']

    ranged = content_of(asyncio.run(view(path='a.py', view_range=[2, 3])))
    assert isinstance(ranged, str)
    assert 'two' in ranged and 'one' not in ranged, 'the line range is not applied'


def test_a_path_the_user_mistyped_comes_back_as_an_answer() -> None:
    """``view`` raises for these two; ``cat -n`` answered with a message.

    Both are the user mistyping, so they stay answers rather than becoming errors it
    has to recover from.
    """
    import asyncio
    import tempfile

    directory = Path(tempfile.mkdtemp())
    (directory / 'a.py').write_text('one\n', encoding='utf-8')

    async def view(**kwargs: Any) -> Any:
        async with ExecutionContext(tool_state={'working_dir': str(directory)}) as context:
            return await acting_tools.read_tool(acting_tools.ReadInput(**kwargs), context)

    for kwargs in ({'path': 'nope.py'}, {'path': 'a.py', 'view_range': [99, 100]}):
        answer = asyncio.run(view(**kwargs))
        text = answer.output[0]['text']
        assert 'Cannot view' in text, f'{kwargs} did not answer: {text[:60]}'


# -- MCP tools and the two ways they go missing ----------------------------


def test_a_server_offering_a_name_a_tool_already_uses_is_refused_at_assembly() -> None:
    """Unguarded this raises on every round instead, from inside the executor.

    The MCP layer checks a new name against the tool registry, but this set
    registers as ``hitl/user/read`` while the model calls it ``read``, so a bare
    ``read`` passes that check.
    """

    def body(_context: Any) -> Any:
        with pytest.raises(toolset.ToolSetError) as caught:
            toolset.build(['send_message', 'read'])
        return str(caught.value)

    message = asyncio.run(with_mcp_tools('read')(body))

    assert 'read' in message
    assert 'prefix' in message, 'the message does not say how to fix it'


def test_a_prefixed_server_tool_coexists_with_the_built_in_ones() -> None:
    """The ordinary case: the default prefix is the server's own name."""

    def body(_context: Any) -> Any:
        return toolset.build(['send_message', 'read']).schema_names

    names = asyncio.run(with_mcp_tools('docs_read')(body))

    assert set(names) == {'send_message', 'read', 'docs_read'}


def test_a_server_loaded_after_assembly_never_reaches_the_model() -> None:
    """Why the agent loads MCP first. This pins the order rather than the wish.

    ``ToolSet.schemas`` is captured once, so a tool that arrives later is callable
    through the executor and invisible in the request.
    """

    async def run() -> tuple[Any, Any]:
        async with ExecutionContext(tool_state={'working_dir': '/testbed'}) as context:
            tools = toolset.build(['send_message'])
            manager = MCPToolManager(context)
            context._mcp_manager = manager
            manager._tools['docs_read'] = mcp_tool('docs_read')
            return tools.schema_names, tuple(schema['function']['name'] for schema in tools.executor.get_tool_schemas())

    frozen, live = asyncio.run(run())

    assert 'docs_read' not in frozen, 'a late server would appear to work'
    assert 'docs_read' in live, 'the executor is expected to see it, which is what makes this silent'


def test_the_agent_loads_its_servers_before_it_builds_the_tool_set() -> None:
    """The consequence of the test above, at the one call site that has to obey it.

    Read from the source because running the agent needs a sandbox. Reordering the
    two calls leaves every test green and every declared server invisible, which is
    what this exists to stop.
    """
    import inspect as inspect_mod

    from nexus.extensions.tasks.hitl.agents import agent_with_user as agent_module

    source = inspect_mod.getsource(agent_module.run_agent_with_user)
    load = source.index('load_mcp_tools(extra_servers=')
    build = source.index('tools = toolset.build(')

    assert load < build, 'the servers are loaded after the schemas are captured, so the model never sees them'


# -- the user-side declaration --------------------------------------------


def test_a_flat_stdio_entry_becomes_the_nested_shape() -> None:
    entry = mcp_config.normalise_user_entry(
        {'name': 'docs', 'command': 'python', 'args': ['-m', 'docs'], 'env': {'TOKEN': 'x'}},
        source='T',
    )

    assert entry == {
        'name': 'docs',
        'client': {'transport': 'stdio', 'command': 'python', 'args': ['-m', 'docs'], 'env': {'TOKEN': 'x'}},
        'runtime_name': 'main',
    }
    assert MCPServerSpec.model_validate(entry).client.command == 'python'


def test_a_server_runs_in_the_sandbox_unless_it_says_otherwise() -> None:
    """The connection is proxied from inside the sandbox, so the runtime parses it.

    A host-side server is the exception and has to say so, because a streaming
    response parsed on the host is a different code path.
    """
    default = mcp_config.normalise_user_entry({'name': 'a', 'command': 'c'}, source='T')
    on_host = mcp_config.normalise_user_entry({'name': 'b', 'command': 'c', 'runtime_name': None}, source='T')

    assert default['runtime_name'] == mcp_config.DEFAULT_RUNTIME == 'main'
    assert 'runtime_name' not in on_host
    assert MCPServerSpec.model_validate(on_host).runtime_name is None


def test_a_url_entry_becomes_an_http_client() -> None:
    entry = mcp_config.normalise_user_entry({'name': 'remote', 'url': 'http://host/mcp'}, source='T')

    assert entry['client']['url'] == 'http://host/mcp'
    assert MCPServerSpec.model_validate(entry).client.transport.value == 'aiohttp_streamable_http'


def test_an_entry_with_no_way_to_reach_the_server_is_refused() -> None:
    with pytest.raises(mcp_config.MCPConfigError, match='client, a url, or a command'):
        mcp_config.normalise_user_entry({'name': 'nothing'}, source='T')


# -- the worker-side declaration ------------------------------------------


def test_the_user_shape_is_refused_for_the_worker() -> None:
    """``claude mcp add`` reads none of it: the server is skipped and logged.

    Left to Claude Code this reads as a worker that never used the tool.
    """
    with pytest.raises(mcp_config.MCPConfigError, match='does not read'):
        mcp_config.check_worker_entry(
            {'name': 'search', 'client': {'transport': 'stdio', 'command': 'python'}},
            source='T',
        )


def test_env_is_refused_for_the_worker_and_says_where_it_goes() -> None:
    """There is no ``--env`` on the command; the server inherits the worker's."""
    with pytest.raises(mcp_config.MCPConfigError, match='worker environment'):
        mcp_config.check_worker_entry({'name': 's', 'command': 'c', 'env': {'TOKEN': 'x'}}, source='T')


def test_a_key_the_worker_side_ignores_is_refused() -> None:
    """``cwd`` and ``timeout_s`` are read on the user side and dropped here."""
    with pytest.raises(mcp_config.MCPConfigError, match='cwd'):
        mcp_config.check_worker_entry({'name': 's', 'command': 'c', 'cwd': '/tmp'}, source='T')


def test_a_worker_entry_keeps_only_what_reaches_claude_code() -> None:
    assert mcp_config.check_worker_entry({'name': 's', 'command': 'c', 'args': ['-m', 'x']}, source='T') == {
        'name': 's',
        'command': 'c',
        'args': ['-m', 'x'],
    }


# -- what a run is given, and how a mistake in it surfaces -----------------


@pytest.mark.parametrize('side', ['user', 'worker'])
def test_a_shell_placeholder_is_refused_because_nothing_expands_it(side: str) -> None:
    """A credential written as ``${VAR}`` reaches the server as that text.

    The stdio transport merges the environment as given, so the server fails on a
    key that looks present.
    """
    entry = {'name': 's', 'command': 'c', 'args': ['--key', '${SEARCH_API_KEY}']}
    check = mcp_config.normalise_user_entry if side == 'user' else mcp_config.check_worker_entry

    with pytest.raises(mcp_config.MCPConfigError, match=r'SEARCH_API_KEY'):
        check(entry, source='T')


def test_an_unset_declaration_means_no_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(mcp_config.USER_ENV, raising=False)
    monkeypatch.delenv(mcp_config.WORKER_ENV, raising=False)

    assert mcp_config.user_servers() == []
    assert mcp_config.worker_servers() == []


def test_a_path_naming_no_file_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Returning nothing would run the whole batch without the servers."""
    missing = tmp_path / 'typo.json'
    monkeypatch.setenv(mcp_config.USER_ENV, str(missing))

    with pytest.raises(mcp_config.MCPConfigError, match=str(missing)):
        mcp_config.user_servers()


@pytest.mark.parametrize('payload', [{'name': 'one'}, 'a string', 3])
def test_a_declaration_that_is_not_a_list_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, payload: Any
) -> None:
    monkeypatch.setenv(mcp_config.USER_ENV, write_json(tmp_path / 'servers.json', payload))

    with pytest.raises(mcp_config.MCPConfigError, match='list of servers'):
        mcp_config.user_servers()


def test_the_same_name_twice_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The MCP layer would raise on the second one; this says which file to fix."""
    monkeypatch.setenv(
        mcp_config.WORKER_ENV,
        write_json(tmp_path / 'w.json', [{'name': 's', 'command': 'a'}, {'name': 's', 'command': 'b'}]),
    )

    with pytest.raises(mcp_config.MCPConfigError, match='more than once'):
        mcp_config.worker_servers()


# -- the built-in search server -------------------------------------------


def test_the_built_in_search_server_is_declared_like_any_other(monkeypatch: pytest.MonkeyPatch) -> None:
    """Its shape and its interpreter are pinned in ``actor/test_worker_mcp.py``, where
    the resolution happens. This is the generator's part: the entry lands in the
    worker config, and the key it needs lands in the worker environment.
    """
    from nexus.extensions.tasks.hitl.utils.generator_util import _build_worker_config

    for key, value in (
        ('HITL_WORKER_API_KEY', 'k'),
        ('HITL_WORKER_BASE_URL', 'http://x/v1'),
        ('HITL_WORKER_MODEL', 'm'),
        ('HITL_WORKER_MCP_SEARCH', '1'),
        ('SEARCH_API_KEY', 'sk-test'),
    ):
        monkeypatch.setenv(key, value)
    monkeypatch.delenv(mcp_config.WORKER_ENV, raising=False)

    config = _build_worker_config('default', session_id='s')

    search = next(entry for entry in config['mcp_servers'] if entry['name'] == 'search')
    assert search['args'] == ['-m', 'nexus.extensions.tasks.hitl.mcp.search_server']
    assert config['extra_env']['SEARCH_API_KEY'] == 'sk-test', 'the key travels as an environment variable'


def test_the_search_server_is_not_registered_unless_it_is_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every recorded run so far has it off, so this is the path they take."""
    from nexus.extensions.tasks.hitl.utils.generator_util import _build_worker_config

    for key, value in (
        ('HITL_WORKER_API_KEY', 'k'),
        ('HITL_WORKER_BASE_URL', 'http://x/v1'),
        ('HITL_WORKER_MODEL', 'm'),
    ):
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('HITL_WORKER_MCP_SEARCH', raising=False)
    monkeypatch.delenv(mcp_config.WORKER_ENV, raising=False)

    config = _build_worker_config('default', session_id='s')

    assert 'mcp_servers' not in config


def test_the_search_server_needs_its_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from nexus.extensions.tasks.hitl.utils.generator_util import _build_worker_config

    for key, value in (
        ('HITL_WORKER_API_KEY', 'k'),
        ('HITL_WORKER_BASE_URL', 'http://x/v1'),
        ('HITL_WORKER_MODEL', 'm'),
        ('HITL_WORKER_MCP_SEARCH', '1'),
    ):
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('SEARCH_API_KEY', raising=False)

    with pytest.raises(ValueError, match='SEARCH_API_KEY'):
        _build_worker_config('default', session_id='s')


def test_a_run_with_no_servers_carries_no_such_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The backends forward the key when present, so an absent one must be absent."""
    from nexus.extensions.tasks.hitl.utils.generator_util import _build_worker_config

    for key, value in (
        ('HITL_WORKER_API_KEY', 'k'),
        ('HITL_WORKER_BASE_URL', 'http://x/v1'),
        ('HITL_WORKER_MODEL', 'm'),
    ):
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('HITL_WORKER_MCP_SEARCH', raising=False)
    monkeypatch.delenv(mcp_config.WORKER_ENV, raising=False)

    assert 'mcp_servers' not in _build_worker_config('default', session_id='s')


def test_both_backends_forward_the_servers_they_are_given() -> None:
    """A key neither backend forwards is a switch that reads as working.

    ``mcp_search`` was one: it was set and nothing read it.
    """
    from nexus.extensions.tasks.hitl.actor.print_claude_code.backend import _SETUP_KEYS
    from nexus.extensions.tasks.hitl.actor.pty_claude_code.runtime_adapter import RUNTIME_CONFIG_KEYS

    assert 'mcp_servers' in _SETUP_KEYS
    assert 'mcp_servers' in RUNTIME_CONFIG_KEYS


# -- what a run tells the user it can check with ----------------------------


def test_a_set_that_can_check_something_is_told_how_to_check() -> None:
    """``str_replace_editor`` is from elsewhere in the tree and declares no usage.

    A persona holding it without ``bash`` or ``read`` would otherwise reach the model
    with nothing said about checking at all. What each tool does is its own
    ``description``, so the doc itself is one constant.
    """
    editing = dict(toolset.build(['send_message', 'str_replace_editor']).usage_docs)
    reading = dict(toolset.build(['send_message', 'read']).usage_docs)
    neither = dict(toolset.build(['send_message', 'accept']).usage_docs)

    assert editing['verification'] == acting_tools.VERIFICATION_DOC
    assert reading['verification'] == acting_tools.VERIFICATION_DOC
    assert 'verification' not in neither


def test_an_unregistered_persona_name_is_refused_by_name() -> None:
    """How a recipe ships its own persona, and how that goes wrong.

    ``register_persona`` runs on import, and ``load_persona_dirs`` on the call. A run
    naming a persona neither of those reached gets only the built-ins.
    """
    from nexus.extensions.tasks.hitl.user_agent.personas import PersonaError

    with pytest.raises(PersonaError) as caught:
        get_persona('reviewer_never_registered')
    assert 'default' in str(caught.value), 'the error names what is registered'


def test_the_generic_agent_resolves_a_persona_from_its_own_directories(tmp_path: Path) -> None:
    """``persona_dirs`` is what a run configured with a persona file of its own sets.

    The agent resolves the name after loading those directories, so a file there
    reaches ``get_persona`` the same way a built-in does.
    """
    from nexus.extensions.tasks.hitl.agents.agent_with_user import AgentWithUserInput
    from nexus.extensions.tasks.hitl.user_agent.personas import load_persona_dirs

    (tmp_path / 'shipped_by_a_run.md').write_text('A person from outside.\n', encoding='utf-8')
    assert 'persona_dirs' in AgentWithUserInput.model_fields

    load_persona_dirs([str(tmp_path)])

    assert get_persona('shipped_by_a_run').prompt.strip() == 'A person from outside.'


def test_a_run_with_nothing_to_look_with_gets_no_verification_section() -> None:
    """The section is about tools, so with none of them it is not written."""
    tools = toolset.build(['send_message', 'accept'])

    assert 'verification' not in dict(tools.usage_docs)


# -- the threshold, said one way --------------------------------------------


def test_the_agent_takes_the_context_window_and_a_share_of_it() -> None:
    """One name for one number, and the one the rest of the tree uses.

    Two agents once disagreed: one read a threshold in tokens, the other a window
    times a percentage. Both went to the same ``Budget``, so the same number meant two
    different thresholds depending on which one read it, and nothing was raised.
    """
    from nexus.extensions.tasks.hitl.agents.agent_with_user import AgentWithUserInput

    assert 'context_window' in AgentWithUserInput.model_fields
    assert 'compact_pct' in AgentWithUserInput.model_fields
    assert 'compact_at' not in AgentWithUserInput.model_fields, 'it still takes the old threshold'


def test_the_window_and_its_share_give_the_threshold() -> None:
    """Below 100% on purpose: the size read lags the current turn's appends."""
    from nexus.extensions.tasks.hitl.agents.agent_with_user import AgentWithUserInput
    from nexus.extensions.tasks.hitl.user_agent.compact import Budget

    def threshold(**kwargs: object) -> Budget:
        held = AgentWithUserInput(  # type: ignore[arg-type]
            user_prompt='x', working_dir='/testbed', requirements=['x'], **kwargs
        )
        return Budget(window=held.context_window, pct=held.compact_pct)

    assert threshold(context_window=250_000, compact_pct=80).threshold == 200_000
    assert threshold(context_window=62_500, compact_pct=80).threshold == 50_000
    assert not threshold().enabled, 'no window leaves compaction off'
