"""Offline tests for the interactive PTY Claude Code worker backend.

A fake Runtime stands in for the sandbox: ``invoke`` records the target function
and its kwargs and returns a canned upstream result dict. That keeps every
assertion here about *this backend's* logic, with no sandbox, no Claude binary
and no model credentials.

The upstream contract these fakes emulate is verified separately in
``test_pty_runtime_contract.py``, which runs the real runtime functions.
"""

import time

from types import SimpleNamespace
from typing import Any

import pytest

from nexus.extensions.tasks.hitl.actor.base import STATUS_ANSWER_MISSED, SessionMessage, TurnResult
from nexus.extensions.tasks.hitl.actor.factory import make_worker_manager
from nexus.extensions.tasks.hitl.actor.pty_claude_code import PtyClaudeCodeSessionManager
from tests.unit.extensions.tasks.hitl.actor.invoke_double import read_invoke


INTERACTION = {
    'interaction_id': 'tooluse-1',
    'tool_name': 'AskUserQuestion',
    'tool_input': {
        'questions': [
            {
                'question': 'Which auth method?',
                'header': 'Auth',
                'options': [{'label': 'OAuth'}, {'label': 'API key'}],
            }
        ]
    },
    'turn_id': 1,
    # Upstream sends this on every request; the manager turns it into the deadline
    # after which an answer is no longer worth sending.
    'timeout_seconds': 1200.0,
}


def _completed(**overrides: Any) -> dict[str, Any]:
    """A finished turn, in the shape ``_format_runtime_turn_result`` returns.

    There is deliberately no top-level ``completed`` key: the upstream reports that
    through ``success`` and puts ``completed`` only on the ``pty_turn`` event
    (``interactive/runtime.py:468-490``). A double that carries one anyway lets a
    caller read a field production never sends.
    """
    result = {
        'success': True,
        'status': 'completed',
        'session_id': 'runtime-side-id',
        'output': 'done the thing',
        'stderr': '',
        'exit_code': 0,
        'events': [{'type': 'pty_turn', 'completed': True}],
        'turn_id': 1,
        'claude_pid': 4321,
        'latest_screen': 'rendered screen',
        'recent_events': [{'type': 'tool_use', 'name': 'Edit'}],
        'event_cursor': {'latest_seq': 7},
    }
    return {**result, **overrides}


def _needs_input(**overrides: Any) -> dict[str, Any]:
    result = _completed(
        status='needs_user_input',
        # Upstream really does report success here, with an empty output.
        success=True,
        output='',
        interaction=dict(INTERACTION),
        events=[{'type': 'pty_interaction_requested'}],
    )
    return {**result, **overrides}


def _terminal(status: str = 'timeout', **overrides: Any) -> dict[str, Any]:
    result = _completed(
        status=status,
        success=False,
        output='partial work',
        stderr=f'PTY turn not completed (reason={status})',
        exit_code=-1,
        events=[{'type': 'pty_turn', 'completed': False}],
    )
    return {**result, **overrides}


class StubClient:
    """The client seat on the handler; nothing here calls through it."""


class FakeRuntime:
    """Records invocations; returns queued upstream result dicts."""

    default_venv_id = 'worker-venv'
    """Non-None so a request that drops it is visible. A turn outside the venv finds
    none of the sessions the backend registered."""

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self.results = list(results or [])
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.handlers: list[str] = []
        self.connects = 0

    async def ws_connect(self) -> None:
        self.connects += 1

    def on(self, event: str, _handler: Any) -> None:
        self.handlers.append(event)

    def off(self, event: str, _handler: Any) -> None:
        self.handlers.remove(event)

    async def invoke(self, target: Any, /, *_args: Any, **kwargs: Any) -> dict[str, Any]:
        name, kwargs = read_invoke(target, kwargs)
        self.calls.append((name, kwargs))
        # Only turn-shaped calls draw from the queue; session opening and closing
        # would otherwise consume the results the turns are waiting for.
        if name not in ('run_claude_code_interactive_session_turn', 'respond_and_wait_claude_code_interactive_session'):
            return {'session_id': kwargs.get('session_id', ''), 'label': kwargs.get('label', ''), 'proxy_port': 9001}
        if not self.results:
            return _completed()
        return self.results.pop(0)

    def turn_calls(self) -> list[dict[str, Any]]:
        return [kw for name, kw in self.calls if name == 'run_claude_code_interactive_session_turn']

    def respond_calls(self) -> list[dict[str, Any]]:
        return [kw for name, kw in self.calls if name == 'respond_and_wait_claude_code_interactive_session']


@pytest.fixture
def manager_factory(monkeypatch):
    """Build a set-up manager wired to a FakeRuntime."""

    def build(results: list[dict[str, Any]] | None = None, **manager_kwargs: Any):
        runtime = FakeRuntime(results)

        async def get_runtime(name='main'):
            assert name == manager_kwargs.get('runtime_name', 'main')
            return runtime

        ctx = SimpleNamespace(
            get_runtime=get_runtime,
            get_openai_client=lambda _name='main': StubClient(),
        )
        # The manager did `from nexus.core import require_context`, so the name to
        # replace is the one bound in its module, not the one in nexus.core.
        monkeypatch.setattr(
            'nexus.extensions.tasks.hitl.actor.pty_claude_code.manager.require_context',
            lambda: ctx,
        )

        manager = PtyClaudeCodeSessionManager(default_working_dir='/testbed', **manager_kwargs)
        return manager, runtime

    return build


@pytest.mark.asyncio
async def test_setup_connects_and_registers_llm_handler(manager_factory):
    manager, runtime = manager_factory()

    await manager.setup()

    assert runtime.connects == 1
    assert runtime.handlers == ['llm_request']

    await manager.setup()  # idempotent
    assert runtime.connects == 1


@pytest.mark.asyncio
async def test_send_before_setup_raises(manager_factory):
    manager, _ = manager_factory()
    manager_session = None
    with pytest.raises(RuntimeError, match='not set up'):
        await manager.send(manager_session, SessionMessage(content='hi'))


@pytest.mark.asyncio
async def test_first_send_starts_a_turn_with_the_input_bridge_enabled(manager_factory):
    # Regression guard: enable_input_bridge defaults to False upstream, and
    # without it AskUserQuestion is denied unattended — the worker would simply
    # never ask anything, with no error anywhere.
    manager, runtime = manager_factory([_completed()])
    await manager.setup()
    session = await manager.start(name='worker')

    await manager.send(session, SessionMessage(content='fix the bug'))

    assert len(runtime.turn_calls()) == 1
    call = runtime.turn_calls()[0]
    assert call['enable_input_bridge'] is True
    assert call['prompt'] == 'fix the bug'
    assert call['working_dir'] == '/testbed'
    assert call['session_id'] == session.session_id


@pytest.mark.asyncio
async def test_completed_turn_reports_no_user_input_request(manager_factory):
    # An ordinary turn must not carry a request at all: a non-None value here
    # would make every turn look pending to a caller that tests presence.
    manager, _ = manager_factory([_completed()])
    await manager.setup()
    session = await manager.start(name='worker')

    result = await manager.send(session, SessionMessage(content='go'))

    assert result.kind == 'completed'
    assert result.user_input_request is None
    assert result.output == 'done the thing'
    assert session.pending_interaction_id is None
    assert session.turns == 1


@pytest.mark.asyncio
async def test_pending_input_is_reported_as_a_hook_request(manager_factory):
    manager, _ = manager_factory([_needs_input()])
    await manager.setup()
    session = await manager.start(name='worker')

    result = await manager.send(session, SessionMessage(content='implement it'))

    request = result.user_input_request
    assert request is not None
    # source, not presence, is what distinguishes "must answer" from "may act".
    assert request.source == 'hook'
    assert request.hook is not None
    assert request.hook['name'] == 'AskUserQuestion'
    assert request.hook['input']['questions'][0]['question'] == 'Which auth method?'
    assert 'Which auth method?' in (request.hint or '')
    assert session.pending_interaction_id == 'tooluse-1'

    # Upstream reports success=True with an empty output while blocked, so a
    # success-first branch would misread this as a finished turn.
    assert result.kind == 'needs_input'
    assert result.output == ''


@pytest.mark.asyncio
async def test_next_send_answers_the_pending_hook(manager_factory):
    manager, runtime = manager_factory([_needs_input(), _completed(output='implemented OAuth')])
    await manager.setup()
    session = await manager.start(name='worker')

    await manager.send(session, SessionMessage(content='implement it'))
    result = await manager.send(session, SessionMessage(content='OAuth'))

    assert len(runtime.respond_calls()) == 1
    answer_call = runtime.respond_calls()[0]
    assert answer_call['interaction_id'] == 'tooluse-1'
    assert answer_call['answer'] == 'OAuth'
    assert answer_call['session_id'] == session.session_id

    # Only one new turn was started; the answer resumed the pending one.
    assert len(runtime.turn_calls()) == 1
    assert result.user_input_request is None
    assert session.pending_interaction_id is None
    assert result.metadata['answered_interaction_id'] == 'tooluse-1'


@pytest.mark.asyncio
async def test_an_unreachable_runtime_is_a_lost_session_not_an_exception(manager_factory):
    """A run whose sandbox stopped answering has to be able to report what it has.

    Measured on run 0812g: the provider returned 500 until Claude Code gave up and
    exited, the sandbox worker went with it, and the invoke never returned. Raising
    from here unwound the whole agent, so the task recorded `patch_text=''` for two
    instances that had been running for hours -- the files were on disk, and nothing
    collected them.
    """
    manager, runtime = manager_factory([_needs_input()])
    await manager.setup()
    session = await manager.start(name='worker')
    await manager.send(session, SessionMessage(content='implement it'))

    async def boom(*_args: Any, **_kwargs: Any):
        raise RuntimeError('bridge gone')

    runtime.invoke = boom  # type: ignore[method-assign]
    result = await manager.send(session, SessionMessage(content='OAuth'))

    assert result.kind == 'terminal', 'the caller must tell this apart from a refused message'
    assert result.session_lost
    assert result.delivered, 'the message reached the sandbox; the session died afterwards'
    assert 'bridge gone' in str(result.metadata['error'])
    assert manager.report(result), 'a lost turn still reports, or the loop shows the user nothing'
    # A stale id would route the following send into a dead interaction.
    assert session.pending_interaction_id is None
    # The turn is counted: it consumed a message, and a silent no-op would let a
    # loop that caps turns spin here forever.
    assert session.turns == 2


@pytest.mark.parametrize('status', ['timeout', 'process_exit'])
@pytest.mark.asyncio
async def test_incomplete_turn_is_terminal(manager_factory, status):
    # Upstream tears the session down for these, so the caller must stop rather
    # than retry: the next send would silently get a fresh, historyless Claude.
    manager, _ = manager_factory([_terminal(status)])
    await manager.setup()
    session = await manager.start(name='worker')

    result = await manager.send(session, SessionMessage(content='go'))

    assert result.kind == 'terminal'
    assert result.user_input_request is None
    assert result.metadata['status'] == status


@pytest.mark.asyncio
async def test_what_the_pty_observes_is_carried_and_degrades_to_empty(manager_factory):
    """The sandbox's own turn material stays in ``raw`` rather than widening the result.

    Every backend returns the same class, so the driver has one shape to read; what
    only a PTY can see is reached through this backend's projector.
    """
    manager, _ = manager_factory([_completed()])
    await manager.setup()
    session = await manager.start(name='worker')

    result = await manager.send(session, SessionMessage(content='go'))

    assert type(result) is TurnResult, 'a per-backend subclass would need per-backend readers'
    assert result.raw['latest_screen'] == 'rendered screen'
    assert result.raw['recent_events'] == [{'type': 'tool_use', 'name': 'Edit'}]
    assert result.raw['event_cursor'] == {'latest_seq': 7}
    assert manager.report(result).count('Edit') == 1, 'the projector reads the events'

    # An older sandbox omits them; the projector must not raise on the absence.
    stripped = _completed()
    for key in ('latest_screen', 'recent_events', 'event_cursor'):
        del stripped[key]
    bare_manager, _ = manager_factory([stripped])
    await bare_manager.setup()
    bare = await bare_manager.send(await bare_manager.start(name='worker'), SessionMessage(content='go'))

    assert bare.kind == 'completed'
    assert bare_manager.report(bare), 'a turn with no events still reports something'


@pytest.mark.asyncio
async def test_the_guard_blocks_a_command_the_worker_does_not_have(manager_factory):
    """Blocked and allowed forms together, since the guard is one predicate.

    ``/clear`` discards the conversation the run depends on and ``/resume`` reattaches
    elsewhere. ``/plan`` is refused for a different reason: nothing installed it, and
    an unknown command produces no assistant turn, so the turn waits out its whole
    timeout -- measured at 900 seconds and the session on run 0821a. A message that
    only mentions a command is prose.
    """
    blocked = ['/clear', '/compact', '/RESUME extra args', '/plan add type hints']
    allowed = ['fix the bug', '/simplify the new tests', 'explain /compact to me']

    manager, runtime = manager_factory(
        [_completed()] * len(allowed),
        default_config={'builtin_commands': ['simplify']},
    )
    await manager.setup()
    session = await manager.start(name='worker')

    for text in blocked:
        result = await manager.send(session, SessionMessage(content=text))
        assert result.kind == 'refused', text
        assert not result.delivered, text
        assert not result.session_lost, f'the session is untouched, so {text!r} must not read as lost'
        assert result.metadata['blocked'] is True, text

    # Session opening happened at start(); the guard must stop the turn itself,
    # and a refused message is not a turn.
    assert runtime.turn_calls() == []
    assert runtime.respond_calls() == []
    assert session.turns == 0

    for text in allowed:
        assert (await manager.send(session, SessionMessage(content=text))).kind == 'completed', text

    assert len(runtime.turn_calls()) == len(allowed)


def test_a_run_offers_no_harness_command_unless_it_names_one(manager_factory):
    """The harness's own commands are the run's choice, not a default it inherits."""
    manager, _ = manager_factory()

    assert dict(manager.builtin_commands) == {}
    assert dict(manager.commands) == {}


def test_a_named_harness_command_joins_the_run_s_own_skills(manager_factory):
    manager, _ = manager_factory(
        default_config={'builtin_commands': ['simplify'], 'worker_commands': {'verify': 'run it from clean'}},
    )

    assert sorted(manager.commands) == ['simplify', 'verify']


def test_a_harness_command_the_harness_does_not_have_is_refused(manager_factory):
    """Sent, it would answer ``Unknown command`` and spend a turn on nothing."""
    manager, _ = manager_factory(default_config={'builtin_commands': ['plan']})

    with pytest.raises(ValueError, match='unknown built-in'):
        _ = manager.commands


@pytest.mark.asyncio
async def test_turns_share_one_session_id(manager_factory):
    manager, runtime = manager_factory([_completed(), _completed()])
    await manager.setup()
    session = await manager.start(name='worker')

    first = await manager.send(session, SessionMessage(content='first'))
    second = await manager.send(session, SessionMessage(content='second'))

    session_ids = {call['session_id'] for call in runtime.turn_calls()}
    assert session_ids == {session.session_id}
    assert session.turns == 2
    # A finished turn must not read as a lost session: a caller that stops on that
    # ends the conversation after one exchange, and the loop looks like a worker
    # that only ever answers once.
    assert (first.kind, second.kind) == ('completed', 'completed')


@pytest.mark.asyncio
async def test_each_turn_records_which_interaction_it_answered(manager_factory):
    """The turn's own metadata carries it, so nothing has to be reconstructed later.

    The manager kept a second copy of every turn in a per-session trajectory; the
    run-level chat tracker already holds the conversations, and the only thing read
    back off that copy was the pending question's text, which the session now holds
    directly.
    """
    manager, _ = manager_factory([_needs_input(), _completed()])
    await manager.setup()
    session = await manager.start(name='worker')

    asked = await manager.send(session, SessionMessage(content='implement it'))
    answered = await manager.send(session, SessionMessage(content='OAuth'))

    assert (asked.metadata['turn'], answered.metadata['turn']) == (1, 2)
    assert asked.metadata['answered_interaction_id'] == '', 'the first turn answered nothing'
    assert answered.metadata['answered_interaction_id'] == 'tooluse-1'
    assert session.pending_questions == [], 'cleared once answered'


# -- an answer the worker was no longer waiting for -------------------------
#
# The sandbox hook waits a bounded time and then answers the question itself, so
# the worker has moved on. Nothing about the result says so: it reads as a normal
# completed turn, and the user believes it was answered.


@pytest.mark.asyncio
async def test_a_pending_question_carries_the_deadline_it_must_be_answered_by(manager_factory):
    manager, _ = manager_factory([_needs_input()])
    await manager.setup()
    session = await manager.start(name='worker')

    before = time.monotonic()
    await manager.send(session, SessionMessage(content='implement it'))

    assert session.pending_deadline is not None
    remaining = session.pending_deadline - before
    assert 1199 <= remaining <= 1201, f'the deadline should be the hook wait, got {remaining}'


@pytest.mark.asyncio
async def test_an_answer_past_the_deadline_is_not_sent_at_all(manager_factory):
    """The hook has already answered for them, so the round trip is wasted."""
    manager, runtime = manager_factory([_needs_input()])
    await manager.setup()
    session = await manager.start(name='worker')
    await manager.send(session, SessionMessage(content='implement it'))
    session.pending_deadline = time.monotonic() - 1.0
    calls_before = len(runtime.calls)

    result = await manager.send(session, SessionMessage(content='OAuth'))

    assert len(runtime.calls) == calls_before, 'the answer was sent to a hook that had stopped waiting'
    assert result.kind == 'refused'
    assert not result.delivered
    assert not result.session_lost, 'the session still serves the next message'
    assert result.raw['status'] == STATUS_ANSWER_MISSED


@pytest.mark.asyncio
async def test_the_user_is_told_the_outcome_without_being_told_the_mechanism(manager_factory):
    """A timeout is not something the user can act on, and naming it would say
    that someone is standing between them and the worker."""
    manager, _ = manager_factory([_needs_input()])
    await manager.setup()
    session = await manager.start(name='worker')
    await manager.send(session, SessionMessage(content='implement it'))
    session.pending_deadline = time.monotonic() - 1.0

    result = await manager.send(session, SessionMessage(content='OAuth'))

    reason = str(result.raw['reason']).lower()
    assert 'did not reach them' in reason, 'the user is not told their answer was lost'
    assert 'themselves' in reason, 'the user is not told the worker chose for itself'
    for leaked in ('timeout', 'deadline', 'hook', 'second', 'expired'):
        assert leaked not in reason, f'the reason names the mechanism: {leaked!r}'


@pytest.mark.asyncio
@pytest.mark.parametrize('reported', ['expired', 'unknown'])
async def test_an_answer_upstream_discarded_is_reported_as_missed(manager_factory, reported: str):
    """The second guard: the deadline is our arithmetic, this is upstream's fact.

    Without it a discarded answer reads as a completed turn, and the run records a
    conversation that did not happen.
    """
    manager, _ = manager_factory([_needs_input(), _completed(interaction_response_status=reported)])
    await manager.setup()
    session = await manager.start(name='worker')
    await manager.send(session, SessionMessage(content='implement it'))

    result = await manager.send(session, SessionMessage(content='OAuth'))

    assert result.kind == 'refused'
    assert result.raw['status'] == STATUS_ANSWER_MISSED
    assert session.turns == 1, 'a turn the worker did not run for us must not count'


@pytest.mark.asyncio
async def test_an_answer_that_landed_is_an_ordinary_turn(manager_factory):
    """``written`` is the normal case, and older upstreams send no status at all."""
    manager, _ = manager_factory([_needs_input(), _completed(interaction_response_status='written')])
    await manager.setup()
    session = await manager.start(name='worker')
    await manager.send(session, SessionMessage(content='implement it'))

    result = await manager.send(session, SessionMessage(content='OAuth'))

    assert result.kind == 'completed'
    assert result.delivered
    assert session.turns == 2


@pytest.mark.asyncio
async def test_a_pending_question_with_no_stated_deadline_is_left_unbudgeted(manager_factory):
    """Better to send and let upstream judge than to guess a deadline."""
    interaction = {key: value for key, value in INTERACTION.items() if key != 'timeout_seconds'}
    manager, runtime = manager_factory([_needs_input(interaction=interaction), _completed()])
    await manager.setup()
    session = await manager.start(name='worker')
    await manager.send(session, SessionMessage(content='implement it'))

    assert session.pending_deadline is None
    result = await manager.send(session, SessionMessage(content='OAuth'))

    assert result.kind == 'completed', 'an unbudgeted answer is still sent'


@pytest.mark.asyncio
async def test_start_rejects_blank_and_duplicate_names(manager_factory):
    manager, _ = manager_factory()
    await manager.setup()

    with pytest.raises(ValueError, match='non-empty'):
        await manager.start(name='')

    await manager.start(name='worker')
    with pytest.raises(ValueError, match='already exists'):
        await manager.start(name='worker')


@pytest.mark.asyncio
async def test_start_merges_default_and_per_session_config(manager_factory):
    manager, _ = manager_factory(default_config={'timeout': 111, 'version': 'x'})
    await manager.setup()

    session = await manager.start(name='worker', config={'timeout': 222})

    assert session.config['timeout'] == 222
    assert session.config['version'] == 'x'


@pytest.mark.asyncio
async def test_config_reaches_the_runtime_and_unknown_keys_are_dropped(manager_factory):
    manager, runtime = manager_factory(
        [_completed()],
        default_config={'timeout': 99, 'haiku_model': 'small', 'cc_model': 'claude-opus-4-8'},
    )
    await manager.setup()
    session = await manager.start(name='worker')

    await manager.send(session, SessionMessage(content='go'))

    call = runtime.turn_calls()[0]
    assert call['timeout'] == 99
    # Not parameters of the upstream primitive: haiku_model configures the host-side
    # LLM handler, cc_model travels as an environment variable.
    assert 'haiku_model' not in call
    assert 'cc_model' not in call
    assert call['extra_env']['ANTHROPIC_MODEL'] == 'claude-opus-4-8[1m]'


@pytest.mark.asyncio
async def test_default_turn_timeout_is_applied_when_unset(manager_factory):
    manager, runtime = manager_factory([_completed()])
    await manager.setup()
    session = await manager.start(name='worker')

    await manager.send(session, SessionMessage(content='go'))

    # Interactive turns need much longer than the upstream 3600s default.
    assert runtime.turn_calls()[0]['timeout'] == 7200


@pytest.mark.asyncio
async def test_a_turn_declares_that_the_sandbox_owns_the_deadline(manager_factory):
    """The runtime client's own invoke wait is set for one RPC, and a turn is a whole
    agent loop. Measured with it left on: two runs of ``dev-chain-lang-0916a`` had a
    turn cut at 7230s and reported as a lost session while it was still running.
    """
    manager, runtime = manager_factory([_needs_input(), _completed()])
    await manager.setup()
    session = await manager.start(name='worker')

    await manager.send(session, SessionMessage(content='go'))
    await manager.send(session, SessionMessage(content='OAuth', kind='answer'))

    assert runtime.turn_calls()[0]['disable_invoke_wait_timeout'] is True
    assert runtime.respond_calls()[0]['disable_invoke_wait_timeout'] is True
    # A ready-made request does not inherit the client's default, and a turn outside
    # the venv finds none of the sessions start() registered.
    assert runtime.turn_calls()[0]['venv_id'] == runtime.default_venv_id
    assert runtime.respond_calls()[0]['venv_id'] == runtime.default_venv_id


@pytest.mark.asyncio
async def test_close_and_shutdown_release_resources(manager_factory):
    manager, runtime = manager_factory([_completed(), {}, {}])
    await manager.setup()
    session = await manager.start(name='worker')
    await manager.send(session, SessionMessage(content='go'))

    await manager.close(session)
    assert [name for name, _ in runtime.calls][-1] == 'close_claude_code_interactive_session'
    with pytest.raises(KeyError):
        manager.get('worker')

    await manager.shutdown()
    assert [name for name, _ in runtime.calls][-1] == 'close_all_claude_code_interactive_sessions'
    assert runtime.handlers == []


@pytest.mark.asyncio
async def test_traces_selects_only_this_session(manager_factory):
    manager, _ = manager_factory([_completed()])
    await manager.setup()
    session = await manager.start(name='worker')
    await manager.send(session, SessionMessage(content='go'))

    from nexus.trace import processor_context
    from nexus.trace.processors.chat_tracker import ChatTrackerProcessor

    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    tracker.snapshot_result = lambda: {  # type: ignore[method-assign]
        'trajectories': [
            {'role': session.label, 'messages': [{'role': 'assistant', 'content': 'MINE'}]},
            {'role': 'worker/_/other', 'messages': [{'role': 'assistant', 'content': 'THEIRS'}]},
        ]
    }

    with processor_context([tracker], finalize_on_exit=False):
        traces = manager.traces(session)

    assert [t['messages'][-1]['content'] for t in traces['trajectories']] == ['MINE']


@pytest.mark.asyncio
async def test_foreign_session_type_is_rejected(manager_factory):
    from nexus.extensions.tasks.hitl.actor.base import Session

    manager, _ = manager_factory()
    await manager.setup()
    foreign = Session(name='x', session_id='y', backend='claude_code')

    with pytest.raises(TypeError, match='PtyClaudeCodeSession'):
        await manager.send(foreign, SessionMessage(content='go'))


def test_factory_builds_the_backend_by_name():
    manager = make_worker_manager('pty_claude_code', default_working_dir='/testbed')

    assert isinstance(manager, PtyClaudeCodeSessionManager)
    assert manager.backend == 'pty_claude_code'
