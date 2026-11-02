"""Worker MCP servers: what reaches ``claude mcp add``, and when it does not.

``claude mcp add`` records a command and runs it later, when the worker's Claude Code
starts the server. Two consequences are pinned here:

* the interpreter is resolved against the worker's ``PATH``, not the one that
  registered it, so a bare ``python`` has to be replaced sandbox-side;
* a registration that fails is logged, not raised -- Claude Code reports a failed
  *registration* but never a failed *launch*, so both read as a worker that did not
  use the tool.

Nothing here starts a server. What a real run adds is whether the launch works,
which needs a sandbox.
"""

import sys

from typing import Any

import pytest

from nexus.extensions.tasks.hitl.actor.mcp import resolve_in_sandbox, resolve_interpreters
from nexus.extensions.tasks.hitl.user_agent import mcp_config


def test_a_bare_interpreter_becomes_the_one_this_process_runs() -> None:
    """``python`` on the worker's PATH may be one without ``mcp`` installed."""
    resolved = resolve_interpreters([{'name': 'search', 'command': 'python', 'args': ['-m', 'x']}])

    assert resolved[0]['command'] == sys.executable
    assert resolved[0]['args'] == ['-m', 'x'], 'only the command is rewritten'


@pytest.mark.parametrize('command', ['python', 'python3'])
def test_both_spellings_of_a_bare_interpreter_are_resolved(command: str) -> None:
    resolved = resolve_interpreters([{'name': 's', 'command': command}])

    assert resolved[0]['command'] == sys.executable


@pytest.mark.parametrize('command', ['/usr/bin/python3', 'bash', 'node', 'uvx'])
def test_anything_else_is_left_alone(command: str) -> None:
    """An absolute path is a choice, and a server that is not Python has nothing
    to resolve."""
    resolved = resolve_interpreters([{'name': 's', 'command': command}])

    assert resolved[0]['command'] == command


def test_the_input_is_not_mutated() -> None:
    """The caller's config is the worker config, which other code reads after this."""
    original: list[dict[str, Any]] = [{'name': 's', 'command': 'python'}]

    resolve_interpreters(original)

    assert original[0]['command'] == 'python'


def test_nothing_to_resolve_is_not_an_error() -> None:
    assert resolve_interpreters(None) == []
    assert resolve_interpreters([]) == []


# -- the user's own servers, which carry the command one level down ---------


def test_the_nested_user_shape_is_resolved_too() -> None:
    """``load_mcp_tools`` takes ``{name, client: {command, ...}}``. Reading only the
    top level left a user server pointing at whatever ``python`` the sandbox had."""
    resolved = resolve_interpreters(
        [{'name': 'search', 'client': {'transport': 'stdio', 'command': 'python', 'args': ['-m', 'x']}}]
    )

    assert resolved[0]['client']['command'] == sys.executable
    assert resolved[0]['client']['args'] == ['-m', 'x']
    assert resolved[0]['client']['transport'] == 'stdio', 'the rest of the client is untouched'


def test_a_nested_absolute_command_is_left_alone() -> None:
    entry = {'name': 's', 'client': {'transport': 'stdio', 'command': '/usr/bin/python3'}}

    assert resolve_interpreters([entry])[0]['client']['command'] == '/usr/bin/python3'


def test_a_nested_entry_is_not_mutated() -> None:
    original: list[dict[str, Any]] = [{'name': 's', 'client': {'command': 'python'}}]

    resolve_interpreters(original)

    assert original[0]['client']['command'] == 'python'


@pytest.mark.asyncio
async def test_the_user_side_resolves_in_the_sandbox_not_on_the_host() -> None:
    """Measured on 0821: the host's interpreter is not the sandbox's, and only the
    venv a run builds carries this package. Resolving host-side named a path that
    does not exist there, and the session failed after it was already open."""
    invoked: list[Any] = []

    class FakeRuntime:
        async def invoke(self, target: Any, /, *args: Any) -> Any:
            invoked.append(target)
            return [{'name': 's', 'client': {'command': '/sandbox/python'}}]

    resolved = await resolve_in_sandbox(FakeRuntime(), [{'name': 's', 'client': {'command': 'python'}}])

    assert invoked == [resolve_interpreters], 'the resolution has to happen in the sandbox'
    assert resolved[0]['client']['command'] == '/sandbox/python'


@pytest.mark.asyncio
async def test_a_sandbox_round_trip_is_skipped_when_there_is_nothing_to_resolve() -> None:
    """An absolute command is a choice, and a round trip costs a sandbox call."""

    class FakeRuntime:
        async def invoke(self, target: Any, /, *args: Any) -> Any:
            raise AssertionError('nothing needed resolving')

    entries = [{'name': 's', 'client': {'command': '/usr/bin/python3'}}]

    assert await resolve_in_sandbox(FakeRuntime(), entries) == entries


@pytest.mark.asyncio
async def test_no_servers_needs_no_runtime_at_all() -> None:
    assert await resolve_in_sandbox(None, None) == []
    assert await resolve_in_sandbox(None, []) == []


# -- the built-in search server --------------------------------------------


def test_the_built_in_server_passes_the_check_a_declared_one_gets() -> None:
    """It used to be appended past this check, and would have failed it: its command
    was ``bash -c`` with a ``${PYTHON_VERSION}`` the check refuses. A built-in nobody
    could write by hand is a built-in whose rules are wrong.
    """
    from nexus.extensions.tasks.hitl.utils.generator_util import _search_server

    checked = mcp_config.check_worker_entry(_search_server(), source='T')

    assert checked['name'] == 'search'


def test_the_built_in_server_resolves_to_an_interpreter_that_has_mcp() -> None:
    """Its module needs ``mcp`` and ``httpx``. Both are in the runtime's own
    interpreter, and in a worker venv, which seeds ``nexus-agent[runtime]``."""
    from nexus.extensions.tasks.hitl.utils.generator_util import _search_server

    resolved = resolve_interpreters([_search_server()])[0]

    assert resolved['command'] == sys.executable
    assert resolved['args'] == ['-m', 'nexus.extensions.tasks.hitl.mcp.search_server']


def test_the_generator_puts_the_checked_entry_in_the_worker_config(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert set(search) == mcp_config.WORKER_KEYS, 'a key Claude Code ignores would be dropped silently'
    assert config['extra_env']['SEARCH_API_KEY'] == 'sk-test', 'the server reads its key from the environment'
