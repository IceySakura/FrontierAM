"""Offline tests for the labelled print-mode worker backend.

A fake runtime stands in for the sandbox: ``invoke`` records which sandbox
function was called with which kwargs and returns a canned result, so every
assertion is about *this* backend's logic -- no sandbox, no Claude binary, no
credentials.

Two things are under test, and they are separate concerns:

* **behaviour outside trace must match the legacy print manager**, because those
  details were arrived at by fixing real failures: resume-vs-start, sending no
  prompt on an answer resume, a deferred question making the turn unsuccessful
  despite exit 0, a missing exit code counting as success, the config-key
  whitelist, and the Claude Code environment defaults;
* **attribution is per session**, via a label fixed when the session's proxy
  opens, which is what allows concurrent sessions.
"""

import asyncio

from typing import Any

import pytest

from nexus.extensions.tasks.hitl.actor.base import SessionMessage
from nexus.extensions.tasks.hitl.actor.labelling.handler import LabelledLLMRequestHandler
from nexus.extensions.tasks.hitl.actor.print_claude_code import PrintClaudeCodeSessionManager
from nexus.extensions.tasks.hitl.actor.print_claude_code import runtime as sandbox
from nexus.extensions.tasks.hitl.actor.print_claude_code.backend import (
    extract_assistant_messages,
    parse_deferred_tools,
)
from nexus.extensions.tasks.hitl.trace import labels
from tests.unit.extensions.tasks.hitl.actor.invoke_double import read_invoke


def stream(*events: dict[str, Any]) -> str:
    """Render events as Claude Code's line-delimited stream-json output."""
    import json

    return '\n'.join(json.dumps(event) for event in events)


def assistant_text(text: str) -> dict[str, Any]:
    return {'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': text}]}}


def ask_user(tool_id: str, question: str = 'Which auth method?') -> dict[str, Any]:
    return {
        'type': 'assistant',
        'message': {
            'content': [
                {
                    'type': 'tool_use',
                    'id': tool_id,
                    'name': 'AskUserQuestion',
                    'input': {'questions': [{'question': question, 'header': 'Auth'}]},
                }
            ]
        },
    }


def tool_result(tool_id: str) -> dict[str, Any]:
    return {'type': 'user', 'message': {'content': [{'type': 'tool_result', 'tool_use_id': tool_id}]}}


class StubClient:
    """The client seat on the handler, which nothing here calls through.

    A real object rather than None: the handler enables labels on whatever it is
    given, and ``None`` is not a legal client anyway -- the base handler reads
    ``config.request_args`` off it.
    """


class FakeRuntime:
    """Records sandbox invocations and replays canned turn output."""

    default_venv_id = 'worker-venv'
    """Non-None so a request that drops it is visible. A turn outside the venv finds
    none of the sessions the backend registered."""

    def __init__(self, *, turns: list[dict[str, Any]] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.turns = turns or []
        self.connects = 0
        self._ports = iter(range(9001, 9100))

    async def ws_connect(self) -> None:
        self.connects += 1

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
            return {'proxy_port': next(self._ports), 'label': kwargs['label']}
        if name == 'run_turn':
            if self.turns:
                return self.turns.pop(0)
            return {'output': stream(assistant_text('done')), 'exit_code': 0, 'execution_time': 1.0}
        return {}

    def kwargs_for(self, name: str) -> list[dict[str, Any]]:
        return [kwargs for called, kwargs in self.calls if called == name]

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


async def make_manager(runtime: FakeRuntime, **kwargs: Any) -> PrintClaudeCodeSessionManager:
    """A manager wired to the fake runtime, bypassing ExecutionContext."""
    manager = PrintClaudeCodeSessionManager(
        llm_handler=LabelledLLMRequestHandler(runtime=runtime, openai_client=StubClient()),
        **kwargs,
    )
    manager._runtime = runtime  # setup() would need a real ExecutionContext
    return manager


# --- session lifecycle -----------------------------------------------------


@pytest.mark.asyncio
async def test_starting_a_session_opens_its_own_labelled_proxy():
    """One proxy per session is what makes the label immutable, and so safe."""
    runtime = FakeRuntime()
    manager = await make_manager(runtime)

    session = await manager.start(name='a')

    opened = runtime.kwargs_for('open_session')
    assert len(opened) == 1
    assert opened[0]['label'] == labels.worker_label('a')
    assert opened[0]['session_id'] == session.session_id
    assert session.label == labels.worker_label('a')


@pytest.mark.asyncio
async def test_each_session_gets_a_distinct_label_and_proxy():
    runtime = FakeRuntime()
    manager = await make_manager(runtime)

    await manager.start(name='a')
    await manager.start(name='b')

    labels_opened = [call['label'] for call in runtime.kwargs_for('open_session')]
    assert labels_opened == [labels.worker_label('a'), labels.worker_label('b')]


@pytest.mark.asyncio
async def test_session_name_with_a_slash_is_rejected():
    """A slash still produces a well-formed label, so filtering would mismatch."""
    manager = await make_manager(FakeRuntime())

    with pytest.raises(labels.LabelError):
        await manager.start(name='a/b')


@pytest.mark.asyncio
async def test_duplicate_session_name_is_rejected():
    manager = await make_manager(FakeRuntime())
    await manager.start(name='a')

    with pytest.raises(ValueError, match='already exists'):
        await manager.start(name='a')


@pytest.mark.asyncio
async def test_session_id_uses_the_claude_code_namespace():
    """Claude Code rejects ids it did not mint in its own shape."""
    manager = await make_manager(FakeRuntime())

    session = await manager.start(name='a')

    assert len(session.session_id) == 36 and session.session_id.count('-') == 4


# --- turn behaviour, aligned with the legacy manager -----------------------


@pytest.mark.asyncio
async def test_first_turn_starts_and_later_turns_resume():
    runtime = FakeRuntime()
    manager = await make_manager(runtime)
    session = await manager.start(name='a')

    await manager.send(session, SessionMessage(content='one'))
    await manager.send(session, SessionMessage(content='two'))

    resumes = [call['resume'] for call in runtime.kwargs_for('run_turn')]
    assert resumes == [False, True]


@pytest.mark.asyncio
async def test_a_deferred_question_makes_the_turn_unsuccessful():
    """Exit 0 is not enough: the turn is unfinished until the question is answered."""
    runtime = FakeRuntime(turns=[{'output': stream(assistant_text('thinking'), ask_user('tool-1')), 'exit_code': 0}])
    manager = await make_manager(runtime)
    session = await manager.start(name='a')

    result = await manager.send(session, SessionMessage(content='go'))

    assert result.kind == 'needs_input'
    assert result.user_input_request is not None
    assert result.user_input_request.source == 'hook'
    assert session.pending_parent_tool_use_id == 'tool-1'


@pytest.mark.asyncio
async def test_answering_writes_the_answer_and_sends_no_prompt():
    """The answer file is what Claude resumes on; a prompt would start afresh."""
    runtime = FakeRuntime(
        turns=[
            {'output': stream(ask_user('tool-1')), 'exit_code': 0},
            {'output': stream(assistant_text('finished')), 'exit_code': 0},
        ]
    )
    manager = await make_manager(runtime)
    session = await manager.start(name='a')
    await manager.send(session, SessionMessage(content='go'))

    result = await manager.send(session, SessionMessage(content='OAuth'))

    written = runtime.kwargs_for('write_answer')
    assert written and written[0]['tool_use_id'] == 'tool-1'
    assert written[0]['answers'] == {'Which auth method?': 'OAuth'}
    assert runtime.kwargs_for('run_turn')[1]['prompt'] == ''
    assert result.kind == 'completed'
    assert session.pending_parent_tool_use_id is None


@pytest.mark.asyncio
async def test_explicit_answers_in_metadata_take_precedence():
    runtime = FakeRuntime(
        turns=[
            {'output': stream(ask_user('tool-1')), 'exit_code': 0},
            {'output': stream(assistant_text('ok')), 'exit_code': 0},
        ]
    )
    manager = await make_manager(runtime)
    session = await manager.start(name='a')
    await manager.send(session, SessionMessage(content='go'))

    await manager.send(session, SessionMessage(content='ignored', metadata={'answers': {'Q': 'A'}}))

    assert runtime.kwargs_for('write_answer')[0]['answers'] == {'Q': 'A'}


@pytest.mark.asyncio
async def test_a_missing_exit_code_counts_as_success():
    """A finished process sometimes reports no status; the output is still complete."""
    runtime = FakeRuntime(turns=[{'output': stream(assistant_text('done')), 'exit_code': None}])
    manager = await make_manager(runtime)
    session = await manager.start(name='a')

    result = await manager.send(session, SessionMessage(content='go'))

    assert result.kind == 'completed'
    assert result.raw['exit_code'] == 0


@pytest.mark.asyncio
async def test_a_crashed_turn_is_not_mistaken_for_a_pending_question():
    """Truncated stdout can leave an unanswered tool_use that means "we stopped"."""
    runtime = FakeRuntime(turns=[{'output': stream(ask_user('tool-1')), 'exit_code': 1}])
    manager = await make_manager(runtime)
    session = await manager.start(name='a')

    result = await manager.send(session, SessionMessage(content='go'))

    assert result.user_input_request is None, 'a crash is not a question'
    assert result.kind == 'completed', 'the process died, not the session; the next turn resumes'
    assert not result.session_lost
    assert result.delivered


@pytest.mark.asyncio
async def test_an_unreachable_sandbox_is_a_lost_session_not_an_exception():
    """A run whose sandbox stopped answering has to report what it has.

    Raising here unwinds the agent, and the task then records an empty patch for
    work already on disk -- measured on the PTY backend in run 0812g, where two
    instances lost hours that way. The same call can fail for either backend, so the
    same result shape covers it.
    """
    runtime = FakeRuntime()
    manager = await make_manager(runtime)
    session = await manager.start(name='a')

    async def boom(*_args: Any, **_kwargs: Any):
        raise TimeoutError('invoke: job abc timed out')

    runtime.invoke = boom  # type: ignore[method-assign]

    result = await manager.send(session, SessionMessage(content='go'))

    assert result.kind == 'terminal', 'the caller must be able to stop rather than send again'
    assert result.session_lost
    assert 'timed out' in str(result.metadata['error'])
    assert session.turns == 1, 'the turn consumed a message, so a capped loop cannot spin here'
    assert manager.report(result), 'a lost turn still reports, or the loop shows the user nothing'


@pytest.mark.asyncio
async def test_a_turn_carries_its_label_and_message_kind():
    """The turn's own metadata carries them, so nothing reconstructs them later.

    The manager kept a second copy of every turn in a per-session trajectory; the
    run-level chat tracker already holds the conversations, keyed by this label.
    """
    runtime = FakeRuntime()
    manager = await make_manager(runtime)
    session = await manager.start(name='a')

    result = await manager.send(session, SessionMessage(content='go', kind='goal'))

    assert result.metadata['label'] == labels.worker_label('a')
    assert result.metadata['message_kind'] == 'goal'
    assert result.metadata['turn'] == 1
    assert result.metadata['name'] == 'a'


# --- config and environment passthrough ------------------------------------


@pytest.mark.asyncio
async def test_claude_code_environment_defaults_match_the_legacy_manager():
    """Including the [1m] suffix, which is what enables the extended window."""
    manager = await make_manager(FakeRuntime(), default_config={'cc_model': 'claude-opus-5'})
    runtime = manager._runtime
    session = await manager.start(name='a')

    await manager.send(session, SessionMessage(content='go'))

    env = runtime.kwargs_for('run_turn')[0]['extra_env']
    assert env['ANTHROPIC_MODEL'] == 'claude-opus-5[1m]'
    assert env['CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC'] == '1'
    assert env['CLAUDE_CODE_ATTRIBUTION_HEADER'] == '0'
    assert env['CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS'] == '1'


@pytest.mark.asyncio
async def test_an_already_extended_model_is_not_double_suffixed():
    manager = await make_manager(FakeRuntime(), default_config={'cc_model': 'claude-opus-5[1m]'})
    runtime = manager._runtime
    session = await manager.start(name='a')

    await manager.send(session, SessionMessage(content='go'))

    assert runtime.kwargs_for('run_turn')[0]['extra_env']['ANTHROPIC_MODEL'] == 'claude-opus-5[1m]'


@pytest.mark.asyncio
async def test_the_worker_environment_names_neither_the_session_nor_the_harness():
    """The worker process holds no variable describing the run it serves.

    ``session.metadata`` is not exported either.
    """
    manager = await make_manager(FakeRuntime())
    runtime = manager._runtime
    session = await manager.start(name='a', metadata={'instance_id': 'inst-7'})

    await manager.send(session, SessionMessage(content='go'))

    env = runtime.kwargs_for('run_turn')[0]['extra_env']
    assert 'inst-7' not in env.values()
    assert [key for key in env if 'SESSION' in key or 'BACKEND' in key or key.startswith('HITL')] == []


@pytest.mark.asyncio
async def test_per_session_config_overrides_the_default():
    manager = await make_manager(FakeRuntime(), default_config={'timeout': 100})
    runtime = manager._runtime
    session = await manager.start(name='a', config={'timeout': 5})

    await manager.send(session, SessionMessage(content='go'))

    assert runtime.kwargs_for('run_turn')[0]['timeout'] == 5.0


@pytest.mark.asyncio
async def test_a_turn_declares_that_the_sandbox_owns_the_deadline():
    """``timeout`` bounds a whole agent loop in the sandbox. The runtime client's own
    invoke wait is set for one RPC, and left on it cuts the turn first and reports it
    as lost while it is still running.
    """
    manager = await make_manager(FakeRuntime())
    runtime = manager._runtime
    session = await manager.start(name='a')

    await manager.send(session, SessionMessage(content='go'))

    assert runtime.kwargs_for('run_turn')[0]['disable_invoke_wait_timeout'] is True
    assert runtime.kwargs_for('run_turn')[0]['venv_id'] == runtime.default_venv_id


@pytest.mark.asyncio
async def test_environment_setup_happens_once_for_many_sessions():
    runtime = FakeRuntime()
    manager = await make_manager(runtime)

    await manager.start(name='a')
    await manager.start(name='b')

    assert runtime.names().count('setup_env') == 1


# --- concurrency -----------------------------------------------------------


@pytest.mark.asyncio
async def test_each_session_opens_its_proxy_under_its_own_label():
    runtime = FakeRuntime()
    manager = await make_manager(runtime)

    await manager.start(name='a')
    await manager.start(name='b')

    assert [call['label'] for call in runtime.kwargs_for('open_session')] == [
        labels.worker_label('a'),
        labels.worker_label('b'),
    ]


@pytest.mark.asyncio
async def test_concurrent_turns_reach_the_sandbox_with_their_own_sessions():
    """The point of per-session proxies: two turns in flight, no shared slot."""
    runtime = FakeRuntime()
    manager = await make_manager(runtime)
    first = await manager.start(name='a')
    second = await manager.start(name='b')

    batch = [(first, SessionMessage(content='ta')), (second, SessionMessage(content='tb'))]
    assert len([turn async for turn in manager.run_turns(batch)]) == 2

    turns = runtime.kwargs_for('run_turn')
    by_session = {call['session_id']: call['prompt'] for call in turns}
    assert len(by_session) == 2
    assert sorted(by_session.values()) == ['ta', 'tb']


# --- trace filtering -------------------------------------------------------


@pytest.mark.asyncio
async def test_session_traces_do_not_absorb_a_sibling_whose_name_it_prefixes():
    """``traces(a)`` must not return ``a2``'s conversation.

    A session label has no terminator, so prefix filtering would match every
    session whose name it prefixes -- silently, since both labels are
    well-formed. That would duplicate a sibling's trajectory into this session's
    export.
    """
    from nexus.trace import processor_context
    from nexus.trace.processors.chat_tracker import ChatTrackerProcessor

    manager = await make_manager(FakeRuntime())
    session_a = await manager.start(name='a')
    await manager.start(name='a2')

    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    tracker.snapshot_result = lambda: {  # type: ignore[method-assign]
        'trajectories': [
            {'messages': [{'role': 'assistant', 'content': 'A'}], 'role': labels.worker_label('a')},
            {'messages': [{'role': 'assistant', 'content': 'A2'}], 'role': labels.worker_label('a2')},
        ]
    }

    with processor_context([tracker], finalize_on_exit=False):
        kept = manager.traces(session_a)

    assert [t['messages'][-1]['content'] for t in kept['trajectories']] == ['A']


# --- teardown --------------------------------------------------------------


@pytest.mark.asyncio
async def test_closing_a_session_stops_its_proxy():
    runtime = FakeRuntime()
    manager = await make_manager(runtime)
    session = await manager.start(name='a')

    await manager.close(session)

    assert runtime.kwargs_for('close_session')[0]['session_id'] == session.session_id
    with pytest.raises(KeyError):
        manager.get('a')


@pytest.mark.asyncio
async def test_shutdown_tears_down_the_sandbox_environment():
    runtime = FakeRuntime()
    manager = await make_manager(runtime)
    await manager.start(name='a')

    await manager.shutdown()

    assert 'teardown_env' in runtime.names()
    with pytest.raises(KeyError):
        manager.get('a')


@pytest.mark.asyncio
async def test_an_injected_handler_is_not_released_by_this_manager():
    """It belongs to whoever passed it in; releasing it would break them."""
    runtime = FakeRuntime()
    manager = await make_manager(runtime)

    await manager.shutdown()  # must not raise despite owning no handler


# --- the stdout parsing, which the turn semantics rest on ------------------


def test_parse_deferred_tools_ignores_an_answered_question():
    stdout = stream(ask_user('tool-1'), tool_result('tool-1'))

    assert parse_deferred_tools(stdout) is None


def test_parse_deferred_tools_returns_the_latest_unanswered_question():
    """With several, Claude is blocked on the last one."""
    stdout = stream(ask_user('tool-1', 'first'), tool_result('tool-1'), ask_user('tool-2', 'second'))

    deferred = parse_deferred_tools(stdout)

    assert deferred is not None
    assert deferred['tool_use_id'] == 'tool-2'
    assert deferred['input']['questions'][0]['question'] == 'second'


def test_parse_deferred_tools_survives_non_json_lines():
    """Sandbox stdout carries log noise alongside the event stream."""
    stdout = 'not json\n' + stream(ask_user('tool-1')) + '\nalso not json'

    assert parse_deferred_tools(stdout) is not None


def test_extract_assistant_messages_skips_tool_only_events():
    """The last item should be the worker's final spoken message."""
    stdout = stream(assistant_text('first'), ask_user('tool-1'), assistant_text('last'))

    assert extract_assistant_messages(stdout) == ['first', 'last']


def test_sandbox_run_turn_refuses_without_an_installation():
    """The install check comes first: no binary means nothing can run at all."""
    with pytest.raises(RuntimeError, match='not installed'):
        asyncio.run(sandbox.run_turn(session_id='never-opened', prompt='x'))


def test_sandbox_run_turn_refuses_a_session_with_no_proxy(monkeypatch):
    """A turn whose session was never opened has nowhere to send its requests.

    Patches the module's installed-path marker so the *second* guard is the one
    under test rather than the first.
    """
    monkeypatch.setattr(sandbox, '_claude_path', '/tmp/claude-code/claude')

    with pytest.raises(RuntimeError, match='has no proxy'):
        asyncio.run(sandbox.run_turn(session_id='never-opened', prompt='x'))


def test_sandbox_setup_is_a_no_op_for_the_settings_it_already_ran(monkeypatch, tmp_path):
    """The install is what makes a second setup return early.

    Everything below it -- the config patches, the pre-start scripts, the MCP
    servers -- is skipped on that path, so a second caller with the same settings
    loses nothing.
    """
    installed = tmp_path / 'claude'
    installed.write_text('#!/bin/sh\n')
    settings = {'mcp_servers': [{'name': 'search', 'command': 'python', 'args': ['-m', 'x']}]}
    monkeypatch.setattr(sandbox, '_claude_path', str(installed))
    monkeypatch.setattr(sandbox, '_setup_signature', _signature_of(settings))

    assert asyncio.run(sandbox.setup_env(**settings)) == {'claude_path': str(installed)}


def test_sandbox_setup_refuses_settings_the_container_cannot_take(monkeypatch, tmp_path):
    """Measured as a silent drop: the MCP registration sits below the early return,
    so a second session asking for a server got a container without it and nothing
    said so."""
    installed = tmp_path / 'claude'
    installed.write_text('#!/bin/sh\n')
    monkeypatch.setattr(sandbox, '_claude_path', str(installed))
    monkeypatch.setattr(sandbox, '_setup_signature', _signature_of({'mcp_servers': None}))

    with pytest.raises(ValueError, match='already set up with different settings'):
        asyncio.run(sandbox.setup_env(mcp_servers=[{'name': 'search', 'command': 'python'}]))


def _signature_of(settings: dict) -> str:
    """The signature ``setup_env`` computes for ``settings``, with its defaults."""
    import inspect
    import json

    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(sandbox.setup_env).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    return json.dumps({**defaults, **settings}, sort_keys=True, default=str)


def test_sandbox_open_session_refuses_to_relabel_an_open_session(monkeypatch):
    """A session's label is fixed for its proxy's lifetime.

    Allowing a second label would reintroduce the mutable slot this design
    exists to remove: turns already in flight would start reporting the new
    label. Re-opening with the *same* label stays idempotent, because the
    backend may reopen after a reconnect.
    """
    existing = sandbox._SessionProxy(proxy=object(), port=9999, label=labels.worker_label('a'))
    monkeypatch.setitem(sandbox._sessions, 'sid-1', existing)

    same = asyncio.run(sandbox.open_session(session_id='sid-1', label=labels.worker_label('a')))
    assert same == {'proxy_port': 9999, 'label': labels.worker_label('a')}

    with pytest.raises(ValueError, match='already open with label'):
        asyncio.run(sandbox.open_session(session_id='sid-1', label=labels.worker_label('b')))


def test_sandbox_open_session_requires_a_label():
    """An unlabelled session's trajectories could not be attributed to anyone."""
    with pytest.raises(ValueError, match='label must be non-empty'):
        asyncio.run(sandbox.open_session(session_id='sid-1', label=''))


def test_deploying_the_hook_keeps_hooks_that_were_already_there(monkeypatch, tmp_path):
    """A hook the image or a config patch installed survives, and ours is added.

    ``PreToolUse`` is the event this backend claims, and the one a caller is most
    likely to have used too. Replacing the whole ``hooks`` key would drop theirs
    without raising -- the worker would simply stop doing whatever their hook did.
    """
    import json

    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setattr(sandbox, '_HOOKS_DIR', str(tmp_path / 'hooks'))
    monkeypatch.setattr(sandbox, '_ANSWERS_DIR', str(tmp_path / 'answers'))
    settings = tmp_path / '.claude' / 'settings.json'
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps(
            {
                'model': 'theirs',
                'hooks': {
                    'PostToolUse': [{'matcher': 'Bash', 'hooks': [{'type': 'command', 'command': 'mine'}]}],
                    'PreToolUse': [{'matcher': 'Write', 'hooks': [{'type': 'command', 'command': 'theirs'}]}],
                },
            }
        )
    )

    sandbox._deploy_ask_user_hook()

    written = json.loads(settings.read_text())
    matchers = [entry['matcher'] for entry in written['hooks']['PreToolUse']]
    assert matchers == ['Write', 'AskUserQuestion', 'ExitPlanMode']
    assert 'PostToolUse' in written['hooks']
    assert written['model'] == 'theirs'

    # Called again on a container that already has it: no duplicate entries.
    sandbox._deploy_ask_user_hook()
    again = json.loads(settings.read_text())
    assert [entry['matcher'] for entry in again['hooks']['PreToolUse']] == matchers
