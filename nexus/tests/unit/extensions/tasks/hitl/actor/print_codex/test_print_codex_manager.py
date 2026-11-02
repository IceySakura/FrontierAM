"""Offline tests for the labelled print-mode Codex backend.

A fake runtime stands in for the sandbox: ``invoke`` records which sandbox
function was called with which kwargs and returns a canned result, so every
assertion is about *this* backend's logic -- no sandbox, no Codex CLI, no
credentials.

Three things are under test:

* **the event stream is read as the manager's turn dict** -- the thread id, the
  final message, the ordered steps, and what a named failure does to the turn;
* **attribution is per session**, via a label fixed when the session's proxy
  opens, which is what allows concurrent sessions;
* **what reaches the sandbox** -- the catalog rewrite and the Codex version are
  what decide whether the worker is offered tools at all.
"""

import asyncio
import json

from typing import Any
from unittest.mock import patch

import pytest

from tests.unit.extensions.tasks.hitl.actor.invoke_double import read_invoke

from nexus.extensions.tasks.hitl.actor.base import SessionMessage
from nexus.extensions.tasks.hitl.actor.labelling.handler import LabelledLLMRequestHandler
from nexus.extensions.tasks.hitl.actor.print_codex import PrintCodexSessionManager
from nexus.extensions.tasks.hitl.actor.print_codex.backend import (
    final_message,
    interpret_turn,
    steps_of,
    thread_id_of,
)
from nexus.extensions.tasks.hitl.trace import labels


THREAD = 'thread_abc123'


def stream(*events: dict[str, Any]) -> str:
    """Render events as ``codex exec --json`` line-delimited output."""
    return '\n'.join(json.dumps(event) for event in events)


def started(thread_id: str = THREAD) -> dict[str, Any]:
    return {'type': 'thread.started', 'thread_id': thread_id}


def said(text: str) -> dict[str, Any]:
    return {'type': 'item.completed', 'item': {'item_type': 'agent_message', 'text': text}}


def ran(command: str) -> dict[str, Any]:
    return {'type': 'item.completed', 'item': {'item_type': 'command_execution', 'command': command}}


def edited(*paths: str) -> dict[str, Any]:
    return {
        'type': 'item.completed',
        'item': {'item_type': 'file_change', 'changes': [{'path': path} for path in paths]},
    }


def failed(message: str) -> dict[str, Any]:
    return {'type': 'turn.failed', 'error': {'message': message}}


class StubClient:
    """The client seat on the handler, which nothing here calls through."""


class FakeRuntime:
    """Records sandbox invocations and replays canned turn output."""

    default_venv_id = 'worker-venv'

    def __init__(self, *, turns: list[dict[str, Any]] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.turns = turns or []
        self._ports = iter(range(9001, 9100))

    async def ws_connect(self) -> None:
        pass

    def on(self, event: str, listener: Any) -> None:
        pass

    def off(self, event: str, listener: Any = None) -> None:
        pass

    def listener_count(self, event: str) -> int:
        return 0

    async def emit(self, event: str, payload: Any = None) -> None:
        pass

    async def invoke(self, target: Any, **kwargs: Any) -> Any:
        name, kwargs = read_invoke(target, kwargs)
        self.calls.append((name, kwargs))
        if name == 'open_session':
            return {'proxy_port': next(self._ports), 'label': kwargs['label'], 'codex_home': '/root/.h'}
        if name == 'run_turn':
            if self.turns:
                return self.turns.pop(0)
            return {'output': stream(started(), said('done')), 'exit_code': 0, 'execution_time': 1.0}
        return {}

    def kwargs_for(self, name: str) -> list[dict[str, Any]]:
        return [kwargs for called, kwargs in self.calls if called == name]

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


async def make_manager(runtime: FakeRuntime, **kwargs: Any) -> PrintCodexSessionManager:
    """A manager wired to the fake runtime, bypassing ExecutionContext."""
    manager = PrintCodexSessionManager(
        llm_handler=LabelledLLMRequestHandler(runtime=runtime, openai_client=StubClient()),
        **kwargs,
    )
    manager._runtime = runtime  # setup() would need a real ExecutionContext
    return manager


def user(content: str) -> SessionMessage:
    return SessionMessage(kind='user', content=content)


@pytest.mark.asyncio
async def test_concurrent_starts_install_once_and_turns_remain_parallel(monkeypatch):
    runtime = FakeRuntime()
    invoke = runtime.invoke
    initializing = peak_initializing = running = 0
    all_running = asyncio.Event()

    async def yielding_invoke(target, **kwargs):
        nonlocal initializing, peak_initializing, running
        name, _ = read_invoke(target, kwargs)
        if name in {'setup_env', 'open_session'}:
            initializing += 1
            peak_initializing = max(peak_initializing, initializing)
            try:
                await asyncio.sleep(0)
                return await invoke(target, **kwargs)
            finally:
                initializing -= 1
        if name == 'run_turn':
            running += 1
            if running == 3:
                all_running.set()
            await all_running.wait()
        return await invoke(target, **kwargs)

    monkeypatch.setattr(runtime, 'invoke', yielding_invoke)
    manager = await make_manager(runtime)
    sessions = await asyncio.wait_for(
        asyncio.gather(*(manager.start(name=name) for name in ('a', 'b', 'c'))), timeout=2
    )
    assert len(runtime.kwargs_for('setup_env')) == 1
    assert len(runtime.kwargs_for('open_session')) == 3
    assert peak_initializing == 1
    assert [manager.get(session.name) for session in sessions] == sessions
    results = await asyncio.wait_for(
        asyncio.gather(*(manager.send(session, user('Work')) for session in sessions)), timeout=2
    )
    assert running == 3
    assert all(result.kind == 'completed' for result in results)


@pytest.mark.asyncio
async def test_concurrent_duplicate_name_opens_one_session(monkeypatch):
    runtime = FakeRuntime()
    invoke = runtime.invoke

    async def yielding_invoke(target, **kwargs):
        await asyncio.sleep(0)
        return await invoke(target, **kwargs)

    monkeypatch.setattr(runtime, 'invoke', yielding_invoke)
    manager = await make_manager(runtime)
    results = await asyncio.gather(manager.start(name='a'), manager.start(name='a'), return_exceptions=True)
    assert sum(isinstance(result, ValueError) for result in results) == 1
    assert len(runtime.kwargs_for('open_session')) == 1
    assert manager.get('a').name == 'a'


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel', [False, True])
async def test_interrupted_initialization_releases_lock(monkeypatch, cancel):
    runtime = FakeRuntime()
    invoke = runtime.invoke
    entered = asyncio.Event()
    attempts = 0

    async def interrupt_once(target, **kwargs):
        nonlocal attempts
        name, _ = read_invoke(target, kwargs)
        if name == 'setup_env':
            attempts += 1
            if attempts == 1:
                entered.set()
                if cancel:
                    await asyncio.Event().wait()
                raise RuntimeError('Installation interrupted')
        return await invoke(target, **kwargs)

    monkeypatch.setattr(runtime, 'invoke', interrupt_once)
    manager = await make_manager(runtime)
    first = asyncio.create_task(manager.start(name='a'))
    await asyncio.wait_for(entered.wait(), timeout=2)
    if cancel:
        first.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        await first
    session = await asyncio.wait_for(manager.start(name='a'), timeout=2)
    assert manager.get('a') is session
    assert attempts == 2
    assert len(runtime.kwargs_for('open_session')) == 1


@pytest.mark.asyncio
async def test_independent_managers_can_initialize_in_parallel(monkeypatch):
    entered = 0
    both_entered = asyncio.Event()

    async def setup_both(target, **kwargs):
        nonlocal entered
        name, _ = read_invoke(target, kwargs)
        if name == 'setup_env':
            entered += 1
            if entered == 2:
                both_entered.set()
            await both_entered.wait()
        return {'proxy_port': 9001}

    runtimes = [FakeRuntime(), FakeRuntime()]
    for runtime in runtimes:
        monkeypatch.setattr(runtime, 'invoke', setup_both)
    managers = [await make_manager(runtime) for runtime in runtimes]
    await asyncio.wait_for(asyncio.gather(*(manager.start(name='a') for manager in managers)), timeout=2)
    assert entered == 2


# --- reading the event stream ---------------------------------------------


def test_the_thread_id_is_read_from_thread_started():
    """Every turn reports it, a resumed one included, so it is read back each time."""
    assert thread_id_of([started(), said('hi')]) == THREAD


def test_a_stream_naming_no_thread_yields_an_empty_id():
    assert thread_id_of([said('hi')]) == ''


def test_the_final_message_is_the_last_thing_the_worker_said():
    """Its claim about the turn, which the user simulator reads."""
    assert final_message([started(), said('first'), ran('ls'), said('last')]) == 'last'


def test_steps_keep_what_was_said_and_run_in_order():
    """The rendering reads this list, so its order is the order of the report."""
    steps = steps_of([started(), said('looking'), ran('pytest -q'), edited('a.py', 'b.py'), said('done')])

    assert [step['type'] for step in steps] == ['text', 'tool_use', 'tool_use', 'text']
    assert steps[1] == {'type': 'tool_use', 'name': 'command_execution', 'detail': 'pytest -q'}
    assert steps[2] == {'type': 'tool_use', 'name': 'file_change', 'detail': 'a.py, b.py'}


def test_an_item_the_schema_names_no_field_for_is_reported_by_name():
    """``mcp_tool_call`` and ``todo_list`` have no documented payload field."""
    events = [
        {'type': 'item.completed', 'item': {'item_type': 'mcp_tool_call', 'server': 'x'}},
        {'type': 'item.completed', 'item': {'item_type': 'todo_list', 'items': []}},
    ]

    assert steps_of(events) == [
        {'type': 'tool_use', 'name': 'mcp_tool_call', 'detail': ''},
        {'type': 'tool_use', 'name': 'todo_list', 'detail': ''},
    ]


def test_a_named_failure_makes_the_turn_unsuccessful_despite_exit_zero():
    """Codex exits 0 on a turn its own stream reports as failed."""
    turn = interpret_turn(
        {'output': stream(started(), said('partial'), failed('rate limited')), 'exit_code': 0},
        session_id='a',
    )

    assert turn['success'] is False
    assert turn['failure'] == 'rate limited'
    assert turn['thread_id'] == THREAD


def test_a_missing_exit_code_counts_as_success():
    turn = interpret_turn({'output': stream(started(), said('done'))}, session_id='a')

    assert turn['exit_code'] == 0
    assert turn['success'] is True


# --- session lifecycle ----------------------------------------------------


@pytest.mark.asyncio
async def test_starting_a_session_opens_its_own_labelled_proxy():
    """One proxy per session is what makes the label immutable, and so safe."""
    runtime = FakeRuntime()
    manager = await make_manager(runtime)

    session = await manager.start(name='a')

    opened = runtime.kwargs_for('open_session')
    assert len(opened) == 1
    assert opened[0]['label'] == labels.worker_label('a')
    assert session.label == labels.worker_label('a')


@pytest.mark.asyncio
async def test_a_session_id_is_empty_until_its_first_turn_reports_a_thread():
    """Codex mints the thread id, so there is none to hold at start."""
    runtime = FakeRuntime()
    manager = await make_manager(runtime)

    session = await manager.start(name='a')
    assert session.session_id == ''

    await manager.send(session, user('go'))
    assert session.session_id == THREAD


@pytest.mark.asyncio
async def test_the_first_turn_starts_a_thread_and_the_next_resumes_it():
    """``resume`` is what carries the history; a fresh exec would have none."""
    runtime = FakeRuntime()
    manager = await make_manager(runtime)
    session = await manager.start(name='a')

    await manager.send(session, user('first'))
    await manager.send(session, user('second'))

    threads = [call['thread_id'] for call in runtime.kwargs_for('run_turn')]
    assert threads == ['', THREAD]


@pytest.mark.asyncio
async def test_a_turn_reporting_no_thread_ends_the_run():
    """The next message would reach a fresh Codex with no memory of this one."""
    runtime = FakeRuntime(turns=[{'output': stream(said('done')), 'exit_code': 0}])
    manager = await make_manager(runtime)
    session = await manager.start(name='a')

    result = await manager.send(session, user('go'))

    assert result.kind == 'terminal'


@pytest.mark.asyncio
async def test_a_turn_killed_at_its_timeout_ends_the_run():
    """A resumed thread stalls the same way: the session's own proxy is what hangs."""
    runtime = FakeRuntime(turns=[{'output': stream(started(), said('')), 'exit_code': 124, 'timed_out': True}])
    manager = await make_manager(runtime)
    session = await manager.start(name='a')

    result = await manager.send(session, user('go'))

    assert result.kind == 'terminal'
    assert session.session_id == THREAD


@pytest.mark.asyncio
async def test_each_session_gets_a_distinct_label():
    runtime = FakeRuntime()
    manager = await make_manager(runtime)

    await manager.start(name='a')
    await manager.start(name='b')

    opened = [call['label'] for call in runtime.kwargs_for('open_session')]
    assert opened == [labels.worker_label('a'), labels.worker_label('b')]


@pytest.mark.asyncio
async def test_session_name_with_a_slash_is_rejected():
    """A slash still produces a well-formed label, so filtering would mismatch."""
    manager = await make_manager(FakeRuntime())

    with pytest.raises(labels.LabelError):
        await manager.start(name='a/b')


@pytest.mark.asyncio
async def test_a_turn_carries_the_port_of_the_session_it_belongs_to():
    """Several sessions share one ``config.toml``, so the port cannot come from it.

    The shared file holds whichever port was written last. A turn that read it would
    reach another session's proxy and go out under that session's label.
    """
    from nexus.extensions.tasks.hitl.actor.print_codex import runtime as sandbox

    commands: list[str] = []

    class Result:
        stdout = ''
        stderr = ''
        return_code = 0

    async def record(request):
        commands.append(request.command)
        return Result()

    sandbox._sessions.clear()
    sandbox._sessions['a'] = sandbox._CodexSession(proxy=None, port=31001, label='x', codex_home='/root/.codex')
    sandbox._sessions['b'] = sandbox._CodexSession(proxy=None, port=31002, label='y', codex_home='/root/.codex')
    entry, node = sandbox._codex_entry, sandbox._node_path
    sandbox._codex_entry, sandbox._node_path = '/tmp/codex.js', '/tmp/node'
    try:
        with patch('nexus.runtime.executors.command.run_command', record):
            await sandbox.run_turn(session_id='a', prompt='pa', working_dir='/tmp')
            await sandbox.run_turn(session_id='b', prompt='pb', working_dir='/tmp')
    finally:
        sandbox._sessions.clear()
        sandbox._codex_entry, sandbox._node_path = entry, node

    assert 'model_providers.proxy.base_url="http://127.0.0.1:31001/v1"' in commands[0]
    assert 'model_providers.proxy.base_url="http://127.0.0.1:31002/v1"' in commands[1]


# --- what reaches the sandbox --------------------------------------------


@pytest.mark.asyncio
async def test_the_session_is_opened_with_the_catalog_it_was_configured_with():
    """The version reaches the sandbox alongside it: the proxy serves ``/v1/models``
    in the shape that version of Codex expects.
    """
    runtime = FakeRuntime()
    catalog = {'models': [{'slug': 'gpt-5.6-sol', 'tool_mode': 'direct'}]}
    manager = await make_manager(
        runtime,
        default_config={
            'version': '0.154.0',
            'model_catalog_json': catalog,
            'codex_model': 'gpt-5.6-sol',
        },
    )

    await manager.start(name='a')

    opened = runtime.kwargs_for('open_session')[0]['config']
    assert opened['version'] == '0.154.0'
    assert opened['model_catalog_json'] == catalog
    assert opened['codex_model'] == 'gpt-5.6-sol'


def test_the_config_names_the_effort_before_the_first_table():
    """Codex sends the effort it reads here; the host client's request args do not
    override it. Unset it takes the catalog's default_reasoning_level, so an xhigh run
    and a max run would send the same thing.

    A top-level key written after a table header belongs to that table, which is why
    the file is parsed rather than the line matched."""
    import tempfile

    from pathlib import Path

    import tomllib

    from nexus.extensions.tasks.hitl.actor.print_codex.runtime import _write_config

    home = Path(tempfile.mkdtemp())
    _write_config(home, proxy_port=1234, config={'codex_model': 'gpt-5.6-sol', 'reasoning_effort': 'none'})
    config = tomllib.loads((home / 'config.toml').read_text())

    assert config['model_reasoning_effort'] == 'none'
    assert config['model_provider'] == 'proxy'

    bare = Path(tempfile.mkdtemp())
    _write_config(bare, proxy_port=1234, config={'codex_model': 'gpt-5.6-sol'})
    assert 'model_reasoning_effort' not in tomllib.loads((bare / 'config.toml').read_text())


def test_the_config_keeps_request_user_input_off_the_wire_by_default():
    """Codex offers that tool and then answers a call to it itself with
    ``unavailable in Default mode``, since no one is at a terminal here.

    The file is parsed rather than the line matched: ``[tools...]`` is a table, so every
    top-level key has to precede it or it becomes one of its fields.
    """
    import tempfile
    import tomllib

    from pathlib import Path

    from nexus.extensions.tasks.hitl.actor.print_codex.runtime import _write_config

    off = Path(tempfile.mkdtemp())
    _write_config(off, proxy_port=1234, config={'codex_model': 'gpt-6.1-sol'})
    parsed = tomllib.loads((off / 'config.toml').read_text())

    assert parsed['tools']['experimental_request_user_input']['enabled'] is False
    # The table did not swallow the keys written before it.
    assert parsed['model'] == 'gpt-6.1-sol'
    assert parsed['model_provider'] == 'proxy'
    assert parsed['model_providers']['proxy']['wire_api'] == 'responses'

    on = Path(tempfile.mkdtemp())
    _write_config(on, proxy_port=1234, config={'codex_model': 'gpt-6.1-sol', 'user_input_tools': True})
    assert 'tools' not in tomllib.loads((on / 'config.toml').read_text())


@pytest.mark.asyncio
async def test_a_session_configured_with_neither_opens_under_the_bundled_catalog():
    """No catalog key is what leaves the one Codex ships in effect."""
    runtime = FakeRuntime()
    manager = await make_manager(runtime)

    await manager.start(name='a')

    opened = runtime.kwargs_for('open_session')[0]['config']
    assert not opened.get('version')
    assert not opened.get('model_catalog_json')


@pytest.mark.asyncio
async def test_the_install_carries_the_same_version_the_session_opens_with():
    """A catalog written for one version and a CLI installed at another disagree."""
    runtime = FakeRuntime()
    manager = await make_manager(runtime, default_config={'version': '0.154.0'})

    await manager.start(name='a')

    assert runtime.kwargs_for('setup_env')[0]['version'] == '0.154.0'
    assert runtime.kwargs_for('open_session')[0]['config']['version'] == '0.154.0'


@pytest.mark.asyncio
async def test_the_worker_environment_names_neither_the_session_nor_the_harness():
    """The worker process holds no variable describing the run it serves."""
    runtime = FakeRuntime()
    manager = await make_manager(runtime)
    session = await manager.start(name='a', metadata={'instance_id': 'inst-7'})

    await manager.send(session, user('go'))

    env = runtime.kwargs_for('run_turn')[0]['extra_env']
    assert 'inst-7' not in env.values()
    assert [key for key in env if 'SESSION' in key or 'BACKEND' in key or key.startswith('HITL')] == []


@pytest.mark.asyncio
async def test_a_turn_runs_in_the_worker_venv_and_owns_its_own_deadline():
    """The sandbox runs a whole agent loop, not one RPC."""
    runtime = FakeRuntime()
    manager = await make_manager(runtime)
    session = await manager.start(name='a')

    await manager.send(session, user('go'))

    call = runtime.kwargs_for('run_turn')[0]
    assert call['venv_id'] == 'worker-venv'
    assert call['disable_invoke_wait_timeout'] is True
