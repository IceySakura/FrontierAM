"""Contract tests: this backend against the *real* upstream PTY runtime.

``test_manager.py`` fakes the runtime entirely, so it proves the manager's logic
but not that the logic matches upstream. These tests close that gap: the adapter,
manager, and the genuine
``nexus.extensions.agents.claude_code.interactive.runtime`` functions all run,
with only the lowest layer — the PTY process itself — replaced. If upstream
changes a status string, a result key, or its session bookkeeping, these fail.

Still fully offline: no Claude binary, no sandbox, no model credentials.
"""

import json

from types import SimpleNamespace
from typing import Any

import pytest

from nexus.extensions.agents.claude_code.interactive import hooks
from nexus.extensions.agents.claude_code.interactive import runtime as interactive_runtime
from nexus.extensions.tasks.hitl.actor.base import SessionMessage
from nexus.extensions.tasks.hitl.actor.pty_claude_code import PtyClaudeCodeSessionManager, runtime_adapter
from tests.unit.extensions.tasks.hitl.actor.invoke_double import resolve_invoke


class StubClient:
    """The client seat on the handler; nothing here calls through it."""


class StubPty:
    """Minimal PTY stand-in: asks questions, then finishes the same turn.

    ``questions`` defaults to three because that is what the worker asked on every
    real call (run 0812e). A one-question stub is the single shape the upstream
    hook accepts as prose, so it cannot see an answer that the hook would deny.

    ``respond_and_wait`` runs the answer through the genuine
    ``hooks.build_hook_output``, so a shape the hook rejects fails here instead of
    in a sandbox.
    """

    def __init__(
        self,
        *,
        pid: int = 4321,
        expected_answer: str = 'OAuth',
        questions: tuple[str, ...] = ('Which auth method?', 'Which store?', 'Which expiry?'),
        wait_timeout: float = runtime_adapter.DEFAULT_HOOK_WAIT_SECONDS,
    ) -> None:
        self._pid = pid
        self._expected_answer = expected_answer
        self._questions = questions
        self._wait_timeout = wait_timeout
        self.turns = 0
        self.answers: list[tuple[str, str | None]] = []
        self.decisions: list[str] = []
        self.closed = False

    @property
    def claude_pid(self) -> int:
        return self._pid

    @property
    def tool_input(self) -> dict[str, Any]:
        return {'questions': [{'question': question} for question in self._questions]}

    async def send_turn(self, prompt: str, *, timeout: float | None = None) -> dict[str, Any]:
        del prompt, timeout
        self.turns += 1
        return {
            'status': 'needs_user_input',
            'completed': False,
            'turn_id': self.turns,
            'interaction': {
                'interaction_id': f'tooluse-{self.turns}',
                'tool_name': 'AskUserQuestion',
                'tool_input': self.tool_input,
                'turn_id': self.turns,
                # The sandbox hook writes this on every request it files
                # (``interactive/hooks.py:359``), and the manager budgets against it.
                'timeout_seconds': self._wait_timeout,
            },
            'screen_tail': '',
            'latest_screen': 'question on screen',
            'elapsed': 0.1,
            'reason': 'needs_user_input',
            'claude_pid': self._pid,
        }

    async def respond_and_wait(self, interaction_id: str, answer: str | None) -> dict[str, Any]:
        self.answers.append((interaction_id, answer))
        decision = hooks.build_hook_output('AskUserQuestion', self.tool_input, answer)
        self.decisions.append(str(decision['hookSpecificOutput']['permissionDecision']))
        return {
            'status': 'completed',
            'completed': True,
            'turn_id': self.turns,
            'final_message': f'implemented {answer}',
            'screen_tail': '',
            'latest_screen': 'work done',
            'elapsed': 0.2,
            'reason': 'stop_hook',
            'claude_pid': self._pid,
        }

    async def close(self) -> None:
        self.closed = True


class StubBridge:
    """The proxy's LLM bridge, which is what carries the session label."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def request(self, openai_params: dict[str, Any], timeout: float = 300) -> dict[str, Any]:
        del timeout
        self.requests.append(openai_params)
        return {'choices': [{'message': {'role': 'assistant', 'content': 'ok'}}]}


class StubProxy:
    """Event sink matching the surface the upstream runtime drives.

    Carries a ``bridge`` because the real ``AnthropicProxy`` sets one in
    ``start()`` and the labelled session wraps it.
    """

    port = 9001

    def __init__(self) -> None:
        self.stopped = False
        self._seq = 0
        self.bridge: Any = StubBridge()

    def mark_turn_start(self) -> None:
        return

    def get_event_delta(self, max_events: int = 1000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del max_events
        return [], {'after_seq': self._seq, 'latest_seq': self._seq, 'count': 0, 'truncated': False}

    def commit_event_cursor(self, seq: int) -> None:
        self._seq = max(self._seq, seq)

    async def stop(self) -> None:
        self.stopped = True


class PassthroughRuntime:
    """Runs invoked functions in-process instead of shipping them to a sandbox."""

    default_venv_id = None

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def ws_connect(self) -> None:
        return

    def on(self, event: str, _handler: Any) -> None:
        return

    def off(self, event: str, _handler: Any) -> None:
        return

    async def invoke(self, target: Any, /, *args: Any, **kwargs: Any) -> Any:
        # This double runs the real runtime function, so the callable has to come back
        # out of an InvokeRequest rather than being looked up by name.
        called, kwargs = resolve_invoke(target, kwargs)
        self.calls.append(called.__name__)
        return await called(*args, **kwargs)


@pytest.fixture
def pty_stub(monkeypatch):
    """Install a stub PTY behind the real runtime and isolate its registry."""
    # The runtime keeps live sessions in a module-global dict; tests must not
    # inherit one another's entries.
    interactive_runtime._PTY_SESSIONS.clear()
    stub = StubPty()
    wrapper = interactive_runtime._InteractiveRuntimeSession(pty_session=stub, proxy=StubProxy())

    async def fake_prepare(_session_id, **_kwargs):
        return wrapper

    monkeypatch.setattr(interactive_runtime, '_prepare_runtime_session', fake_prepare)
    yield stub
    interactive_runtime._PTY_SESSIONS.clear()


@pytest.fixture
def manager(monkeypatch):
    runtime = PassthroughRuntime()

    async def get_runtime(name='main'):
        assert name == 'main'
        return runtime

    ctx = SimpleNamespace(get_runtime=get_runtime, get_openai_client=lambda _name='main': StubClient())
    monkeypatch.setattr(
        'nexus.extensions.tasks.hitl.actor.pty_claude_code.manager.require_context',
        lambda: ctx,
    )
    return PtyClaudeCodeSessionManager(
        default_working_dir='/testbed',
    ), runtime


@pytest.mark.asyncio
async def test_hook_question_and_answer_complete_one_turn(manager, pty_stub):
    mgr, runtime = manager
    await mgr.setup()
    session = await mgr.start(name='worker')

    asked = await mgr.send(session, SessionMessage(content='implement auth'))

    # Upstream really does report success while blocked; the kind is what separates
    # "waiting on an answer" from "finished".
    assert asked.kind == 'needs_input'
    assert asked.output == ''
    assert asked.user_input_request is not None
    assert asked.user_input_request.source == 'hook'
    assert session.pending_interaction_id == 'tooluse-1'
    assert asked.raw['latest_screen'] == 'question on screen'

    answered = await mgr.send(session, SessionMessage(content='OAuth'))

    assert answered.kind == 'completed'
    assert 'OAuth' in answered.output
    assert answered.user_input_request is None
    assert session.pending_interaction_id is None
    # The answer is reshaped for the hook, which is covered on its own below; here
    # only that it reached the pending interaction carrying what the user said.
    assert [interaction_id for interaction_id, _ in pty_stub.answers] == ['tooluse-1']
    assert 'OAuth' in str(pty_stub.answers[0][1])

    # Answering resumes the pending turn rather than starting a new one, so the
    # PTY only ever saw one prompt.
    assert pty_stub.turns == 1
    assert asked.metadata['turn_id'] == answered.metadata['turn_id'] == 1

    # open_session first: the proxy must exist and be labelled before any turn.
    assert runtime.calls == [
        'open_session',
        'run_claude_code_interactive_session_turn',
        'respond_and_wait_claude_code_interactive_session',
    ]


@pytest.mark.asyncio
async def test_session_is_reused_across_turns(manager, pty_stub):
    mgr, _ = manager
    await mgr.setup()
    session = await mgr.start(name='worker')

    first = await mgr.send(session, SessionMessage(content='first'))
    answered = await mgr.send(session, SessionMessage(content='OAuth'))
    second = await mgr.send(session, SessionMessage(content='now the tests'))

    # One live PTY for the whole conversation, keyed by our session id.
    assert list(interactive_runtime._PTY_SESSIONS) == [session.session_id]
    assert second.raw['claude_pid'] == 4321
    assert pty_stub.turns == 2
    assert session.turns == 3
    # Read off the real formatter, which reports a finished turn through `success`
    # and carries no top-level `completed`. A terminal reading of any of these ends
    # the conversation after one exchange.
    assert [result.session_lost for result in (first, answered, second)] == [False, False, False]


@pytest.mark.asyncio
async def test_close_removes_the_runtime_side_session(manager, pty_stub):
    mgr, _ = manager
    await mgr.setup()
    session = await mgr.start(name='worker')
    await mgr.send(session, SessionMessage(content='go'))
    assert session.session_id in interactive_runtime._PTY_SESSIONS

    await mgr.close(session)

    assert interactive_runtime._PTY_SESSIONS == {}
    assert pty_stub.closed is True


@pytest.mark.asyncio
async def test_incomplete_turn_tears_down_the_runtime_session(manager, monkeypatch):
    # Upstream closes and de-registers the PTY when a turn neither completes nor
    # asks for input, which is why the loop must treat this as terminal: a retry
    # would silently get a fresh Claude with no history.
    interactive_runtime._PTY_SESSIONS.clear()
    stub = StubPty()

    async def timed_out(prompt: str, *, timeout: float | None = None) -> dict[str, Any]:
        del prompt, timeout
        return {
            'status': 'timeout',
            'completed': False,
            'turn_id': 1,
            'final_message': '',
            'screen_tail': 'partial',
            'latest_screen': 'partial',
            'reason': 'timeout',
            'claude_pid': 4321,
        }

    stub.send_turn = timed_out  # type: ignore[method-assign]
    wrapper = interactive_runtime._InteractiveRuntimeSession(pty_session=stub, proxy=StubProxy())

    async def fake_prepare(_session_id, **_kwargs):
        return wrapper

    monkeypatch.setattr(interactive_runtime, '_prepare_runtime_session', fake_prepare)

    mgr, _ = manager
    await mgr.setup()
    session = await mgr.start(name='worker')

    result = await mgr.send(session, SessionMessage(content='go'))

    assert result.kind == 'terminal'
    assert result.user_input_request is None
    assert interactive_runtime._PTY_SESSIONS == {}
    interactive_runtime._PTY_SESSIONS.clear()


@pytest.mark.asyncio
async def test_input_bridge_flag_reaches_the_runtime(manager, pty_stub, monkeypatch):
    # Guard the whole path, not just the adapter: without this flag the upstream
    # builds no hook bridge and the worker can never ask anything.
    seen: dict[str, Any] = {}
    original = interactive_runtime._prepare_runtime_session

    async def recording_prepare(session_id, **kwargs):
        seen.update(kwargs)
        return await original(session_id, **kwargs)

    monkeypatch.setattr(interactive_runtime, '_prepare_runtime_session', recording_prepare)

    mgr, _ = manager
    await mgr.setup()
    session = await mgr.start(name='worker')
    await mgr.send(session, SessionMessage(content='go'))

    assert seen['enable_input_bridge'] is True


@pytest.mark.asyncio
async def test_the_hook_timeouts_this_backend_sets_are_ones_the_runtime_accepts(manager, pty_stub, monkeypatch):
    """A user simulator reads the repository between turns, so upstream's own 700s
    wait is short. The pair is validated by the runtime -- the command timeout has
    to exceed the wait, or the hook is killed while it is still waiting -- so it is
    checked here against the real function rather than against our copy of it.
    """
    seen: dict[str, Any] = {}
    original = interactive_runtime._prepare_runtime_session

    async def recording_prepare(session_id, **kwargs):
        seen.update(kwargs)
        return await original(session_id, **kwargs)

    monkeypatch.setattr(interactive_runtime, '_prepare_runtime_session', recording_prepare)

    mgr, _ = manager
    await mgr.setup()
    session = await mgr.start(name='worker')
    await mgr.send(session, SessionMessage(content='go'))

    assert seen['hook_wait_timeout'] == 1200.0
    assert seen['hook_command_timeout'] == 1250.0
    # Raises when the pair is invalid, which is the reason for asserting both.
    interactive_runtime._coerce_hook_timeouts(seen['hook_wait_timeout'], seen['hook_command_timeout'])


@pytest.mark.asyncio
async def test_the_request_states_the_window_the_answer_has(manager, pty_stub):
    """The manager budgets against this, so a payload without it leaves it blind."""
    mgr, _ = manager
    await mgr.setup()
    session = await mgr.start(name='worker')

    asked = await mgr.send(session, SessionMessage(content='go'))

    assert asked.awaiting_answer
    assert asked.user_input_request is not None
    assert asked.user_input_request.metadata['hook_timeout_seconds'] == 1200.0
    assert session.pending_deadline is not None


# -- the answer shape the upstream hook will actually accept --------------


@pytest.fixture
def one_question_pty(monkeypatch):
    """A stub that asks a single question, which the hook accepts as prose."""
    interactive_runtime._PTY_SESSIONS.clear()
    stub = StubPty(questions=('Which auth method?',))
    wrapper = interactive_runtime._InteractiveRuntimeSession(pty_session=stub, proxy=StubProxy())

    async def fake_prepare(_session_id, **_kwargs):
        return wrapper

    monkeypatch.setattr(interactive_runtime, '_prepare_runtime_session', fake_prepare)
    yield stub
    interactive_runtime._PTY_SESSIONS.clear()


@pytest.mark.asyncio
async def test_an_answer_to_several_questions_is_accepted_by_the_real_hook(manager, pty_stub):
    """Run 0812e: 11 of 11 AskUserQuestion calls were denied and none was answered.

    The worker asked three questions each time and the user replied in one
    paragraph. ``_normalise_answers`` takes prose only when there is exactly one
    question, so every answer became "no user is available" -- while the worker
    kept working, so nothing raised and no score moved.
    """
    mgr, _ = manager
    await mgr.setup()
    session = await mgr.start(name='worker')
    await mgr.send(session, SessionMessage(content='add token auth'))

    await mgr.send(session, SessionMessage(content='Use OAuth, keep it in redis, expire in a day.'))

    assert pty_stub.decisions == ['allow'], 'the hook must accept what the manager sent'


@pytest.mark.asyncio
async def test_the_answer_carries_one_entry_per_question(manager, pty_stub):
    """The reply lands on the first question; the rest point back to it, so the worker
    is not shown the same paragraph once per question."""
    mgr, _ = manager
    await mgr.setup()
    session = await mgr.start(name='worker')
    await mgr.send(session, SessionMessage(content='add token auth'))
    answer_text = 'Use OAuth, keep it in redis, expire in a day.'

    await mgr.send(session, SessionMessage(content=answer_text))

    _, sent = pty_stub.answers[0]
    assert json.loads(str(sent)) == {
        'Which auth method?': answer_text,
        'Which store?': '(answered above)',
        'Which expiry?': '(answered above)',
    }


@pytest.mark.asyncio
async def test_a_single_question_is_answered_in_prose(manager, one_question_pty):
    """One question needs no JSON, and prose is what a person would type."""
    mgr, _ = manager
    await mgr.setup()
    session = await mgr.start(name='worker')
    await mgr.send(session, SessionMessage(content='add token auth'))

    await mgr.send(session, SessionMessage(content='OAuth please'))

    assert one_question_pty.answers[0][1] == 'OAuth please'
    assert one_question_pty.decisions == ['allow']


@pytest.mark.asyncio
async def test_the_questions_are_forgotten_once_answered(manager, pty_stub):
    """A stale list would shape the next answer against the wrong questions."""
    mgr, _ = manager
    await mgr.setup()
    session = await mgr.start(name='worker')
    await mgr.send(session, SessionMessage(content='add token auth'))
    assert session.pending_questions == ['Which auth method?', 'Which store?', 'Which expiry?']

    await mgr.send(session, SessionMessage(content='OAuth, redis, one day.'))

    assert session.pending_questions == []
    assert session.pending_interaction_id is None


def test_hook_answer_is_prose_for_a_single_question() -> None:
    assert runtime_adapter.hook_answer('use OAuth', ['Which auth flow?']) == 'use OAuth'


def test_hook_answer_does_not_repeat_the_reply_across_questions() -> None:
    """The reply lands on the first question; the rest point back to it.

    Round-tripped through the upstream hook to show it accepts the shape and the reply
    is not shown once per question.
    """
    questions = ['Which auth flow?', 'Which database?', 'Which UI kit?']
    tool_input = {'questions': [{'question': text} for text in questions]}
    reply = 'use OAuth, postgres, and whatever you like for the UI'

    shaped = runtime_adapter.hook_answer(reply, questions)
    answers = hooks._normalise_answers(shaped, tool_input)

    assert answers == {
        'Which auth flow?': reply,
        'Which database?': '(answered above)',
        'Which UI kit?': '(answered above)',
    }
    assert shaped.count(reply) == 1, 'the reply appears once, not once per question'
