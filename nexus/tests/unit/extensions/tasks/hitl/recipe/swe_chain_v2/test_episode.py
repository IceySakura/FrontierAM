"""A whole conversation, driven through the real parts.

Only the model and the worker are faked; the tool executor, the dispatcher, the
handlers, the queue, the hook and the prompts are the real ones. These are the cases
the per-module tests cannot reach: a requirement that never arrives in the history, a
tool result that never gets written, a queue that advances when it should not.

The loop is ``agents/agent_with_user``'s and the recipe's part is
:class:`~nexus.extensions.tasks.hitl.recipe.swe_chain_v2.hook.SWEChainHook`, so what
these cover is the two together.
"""

import asyncio
import json

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from nexus.core import ExecutionContext
from nexus.extensions.tasks.hitl.actor.base import TurnKind, TurnResult
from nexus.extensions.tasks.hitl.agents.agent_with_user import (
    RunFinish,
    RunStart,
    RunState,
    notify_hooks,
    build_handlers,
)
from nexus.extensions.tasks.hitl.recipe.swe_chain_v2 import hook as chain_hook
from nexus.extensions.tasks.hitl.user_agent.dispatch import (
    Dispatcher,
    call_name,
    check_handlers,
    read_response,
    tool_message,
)
from nexus.extensions.tasks.hitl.user_agent.personas import get_persona
from nexus.extensions.tasks.hitl.user_agent.prompts import (
    NUDGE,
    build_opening_message,
    build_system_prompt,
    environment_facts,
    reminder,
    reminder_message,
)
from nexus.extensions.tasks.hitl.user_agent.language import detect
from nexus.extensions.tasks.hitl.user_agent.queue import Requirements
from nexus.extensions.tasks.hitl.user_agent.reminders import Reminders
from nexus.extensions.tasks.hitl.user_agent.tools import toolset


WORKER = 'worker'


class Call:
    def __init__(self, name: str, call_id: str, arguments: dict[str, Any]) -> None:
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=json.dumps(arguments))


class Message:
    def __init__(self, calls: list[Call]) -> None:
        self.tool_calls = calls
        self.role = 'assistant'
        self.content = None

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {'role': 'assistant', 'tool_calls': [{'id': c.id} for c in (self.tool_calls or [])]}


class ScriptedModel:
    """Plays a list of (tool, arguments) batches, repeating the last."""

    def __init__(self, script: list[list[tuple[str, dict[str, Any]]]]) -> None:
        self.script = script
        self.turns = 0
        self.prompts: list[list[dict]] = []

    async def create_completion(self, messages: list[dict], tools: list[dict] | None = None, **_: Any) -> Any:
        batch = self.script[min(self.turns, len(self.script) - 1)]
        self.turns += 1
        self.prompts.append([dict(m) for m in messages])
        calls = [Call(name, f't{self.turns}_{i}', args) for i, (name, args) in enumerate(batch)]
        return SimpleNamespace(choices=[SimpleNamespace(message=Message(calls))])


@dataclass(frozen=True)
class FakeTurn(TurnResult):
    """A worker turn, built from the real result type.

    Subclassed rather than duck-typed: a hand-written stand-in stops carrying
    whatever the interface gains, and the loop then fails on the attribute instead
    of on the behaviour under test. ``kind`` gains a default so the cases here can
    stay about what the worker said.
    """

    kind: TurnKind = 'completed'
    text: str = 'I changed it.'

    @property
    def has_output(self) -> bool:
        return bool(self.text)


class FakeManager:
    product = 'Claude Code'
    commands: tuple[str, ...] = ()
    command_prefix = '/'

    def __init__(self, results: list[FakeTurn] | None = None) -> None:
        self.results = results or [FakeTurn()]
        self.index = 0
        self.sent: list[str] = []

    async def send(self, session: Any, message: Any) -> FakeTurn:
        self.sent.append(message.content)
        result = self.results[min(self.index, len(self.results) - 1)]
        self.index += 1
        return result

    def report(self, result: TurnResult, *, detail: str = 'default') -> str:
        text = result.text if isinstance(result, FakeTurn) else result.output
        return f'<worker_report>{text}</worker_report>'


@dataclass
class Episode:
    history: list[dict[str, Any]]
    state: RunState
    manager: FakeManager
    model: ScriptedModel
    hook: Any

    @property
    def turns(self) -> int:
        return self.state.turns

    @property
    def stopped(self) -> bool:
        return self.state.stopped

    @property
    def text(self) -> str:
        return '\n'.join(str(m.get('content', '')) for m in self.history)

    @property
    def tool_results(self) -> str:
        return '\n'.join(str(m['content']) for m in self.history if m.get('role') == 'tool')

    @property
    def reminders(self) -> str:
        """Reminders, which arrive as their own user messages rather than on a tool result.

        A tool result can be tens of thousands of characters (20036 on run
        0812f), so anything appended to one is not read.
        """
        return '\n'.join(
            str(m.get('content', ''))
            for m in self.history
            if m.get('role') == 'user' and '<system-reminder>' in str(m.get('content', ''))
        )


def run_episode(
    script: list[list[tuple[str, dict[str, Any]]]],
    *,
    requirements: list[str],
    results: list[FakeTurn] | None = None,
    persona: str = 'default',
    max_turns: int = 12,
    hook: Any = None,
) -> Episode:
    """Drive one conversation the way the agent function does."""
    chosen = get_persona(persona)
    model = ScriptedModel(script)
    manager = FakeManager(results)
    session = object()
    run_hook = chain_hook.SWEChainHook(instance_id='test') if hook is None else hook

    tools = toolset.build(list(chosen.tools))
    queue = Requirements.build(requirements)
    language = detect('\n'.join(queue.segments))
    reminders = Reminders(recalls=chosen.recalls)

    history: list[dict[str, Any]] = [
        {
            'role': 'system',
            'content': build_system_prompt(
                chosen,
                usage_docs=tools.usage_docs,
                environment=environment_facts(working_dir='/testbed', workers=[WORKER]),
            ),
        },
        {
            'role': 'user',
            'content': build_opening_message(queue.current, language=language.name),
        },
    ]
    cap = max_turns
    context = ExecutionContext(tool_state=toolset.tool_state(tools, working_dir='/testbed'))
    state = RunState(queue=queue, manager=manager, session=session, natural_language=language)

    built = build_handlers(state, hooks=[run_hook])
    handlers = {name: handler for name, handler in built.items() if name in tools.schema_names}
    check_handlers(handlers, set(tools.schema_names))
    dispatcher = Dispatcher(executor=tools.executor, handlers=handlers)

    async def go() -> None:
        async with context:
            await notify_hooks([run_hook], RunStart(state=state), state)
            for _ in range(cap * 2):
                if state.turns >= cap:
                    break
                message, calls = read_response(
                    await model.create_completion(messages=history, tools=list(tools.schemas))
                )
                history.append(message)
                if not calls:
                    state.idle_turns += 1
                    history.append({'role': 'user', 'content': reminder(NUDGE)})
                    continue
                state.turns += 1
                appended: list[dict[str, Any]] = []
                for call in calls:
                    result = await dispatcher.dispatch(call)
                    history.append(tool_message(call, result.content))
                    appended.extend(result.messages)
                    recall = reminders.after_tool(call_name(call))
                    if recall:
                        appended.append(reminder_message(recall))
                history.extend(appended)
                if state.stopped:
                    break

    asyncio.run(go())
    return Episode(history=history, state=state, manager=manager, model=model, hook=run_hook)


def send(text: str, intent: str = 'request', **extra: Any) -> tuple[str, dict[str, Any]]:
    """A call the handler accepts. ``draft`` is filled because a call without one is
    refused, so a test that omits it exercises that guard rather than its own subject."""
    return ('send_message', {'intent': intent, 'message': text, 'draft': 'what I make of it', **extra})


def accept(reason: str = 'satisfied', summary: str = 'good', **extra: Any) -> tuple[str, dict[str, Any]]:
    """``checked`` is filled for the same reason ``draft`` is: a `satisfied` accept
    without it is refused, so a test omitting it exercises that guard instead."""
    return ('accept', {'reason': reason, 'summary': summary, 'checked': 'went through each part', **extra})


def look(path: str = '/testbed') -> tuple[str, dict[str, Any]]:
    return ('bash', {'command': f'ls {path}'})


# -- a run from end to end -----------------------------------------------


def test_one_requirement_reaches_the_worker_and_the_run_ends() -> None:
    episode = run_episode([[send('it crashes on empty input')], [accept()]], requirements=['Fix the crash.'])

    assert episode.stopped
    assert episode.manager.sent == ['it crashes on empty input']
    assert episode.state.stop_reason == 'satisfied'
    assert episode.state.worker_rounds == 1


def test_a_three_requirement_run_walks_the_whole_queue() -> None:
    episode = run_episode(
        [[send('first')], [accept()], [send('next')], [accept()], [send('last')], [accept()]],
        requirements=['one', 'two', 'three'],
    )

    assert episode.stopped
    assert len(episode.manager.sent) == 3
    assert [entry['reason'] for entry in episode.state.accepted] == ['satisfied'] * 3


def test_each_released_requirement_reaches_the_model_not_just_the_queue() -> None:
    """The handler returns it as a message; a loop that drops those still 'passes'
    every queue test while the model never learns what else was wanted."""
    episode = run_episode(
        [[send('about the first')], [accept()], [send('about the second')], [accept()]],
        requirements=['FIRST REQUIREMENT', 'SECOND REQUIREMENT'],
    )

    assert 'FIRST REQUIREMENT' in episode.text
    assert 'SECOND REQUIREMENT' in episode.text


def test_the_requirement_is_not_what_the_worker_receives() -> None:
    requirement = 'ERC20DetailedWrapper applies bytes32ToString to an already-string value.'

    episode = run_episode([[send('the token name comes out as garbage')], [accept()]], requirements=[requirement])

    assert episode.manager.sent == ['the token name comes out as garbage']
    assert requirement not in episode.manager.sent[0]


def test_a_run_that_never_accepts_stops_at_the_cap() -> None:
    episode = run_episode([[send('again')]], requirements=['x'], max_turns=3)

    assert not episode.stopped
    assert episode.turns == 3
    assert episode.state.stop_reason == ''


def test_a_lost_worker_ends_the_run_instead_of_being_talked_to_again() -> None:
    """The run has to stop so the task can still collect what was written.

    The script would keep sending forever. A lost session must end the loop on its
    own, under a reason distinct from accepting, or a run killed by a provider
    outage grades as a finished conversation.
    """
    from nexus.extensions.tasks.hitl.user_agent.tools.orchestration import (
        SESSION_LOST_NOTICE,
        SESSION_LOST_REASON,
    )

    episode = run_episode(
        [[send('please fix it')], [send('are you there?')], [accept()]],
        requirements=['x'],
        results=[FakeTurn(kind='terminal', text='')],
    )

    assert episode.stopped
    assert episode.state.stop_reason == SESSION_LOST_REASON
    assert episode.state.stop_reason != 'satisfied'
    assert episode.state.session_lost, 'derived from the stop reason rather than tracked twice'
    assert len(episode.manager.sent) == 1, 'nothing is sent after the session is gone'
    assert SESSION_LOST_NOTICE in episode.tool_results, 'the user is told why it stopped'


def test_a_refused_message_does_not_end_the_run() -> None:
    """A refused message is not a lost worker: the user can try something else.

    The two were the same shape under ``success``/``session_lost``, so this case
    could only assert that the run continued, not that the user was told why.
    """
    from nexus.extensions.tasks.hitl.user_agent.tools.orchestration import NOT_DELIVERED

    episode = run_episode(
        [[send('/compact')], [send('actually, fix the crash')], [accept()]],
        requirements=['x'],
        results=[
            FakeTurn(kind='refused', text='', raw={'reason': 'slash command would reset the session'}),
            FakeTurn(text='fixed'),
        ],
    )

    assert episode.stopped
    assert episode.state.stop_reason == 'satisfied'
    assert not episode.state.session_lost
    assert len(episode.manager.sent) == 2
    assert NOT_DELIVERED in episode.tool_results, 'the user is told it never arrived'
    assert 'slash command' in episode.tool_results, 'and why'
    assert episode.state.worker_rounds == 1, 'a message that was never sent is not a round'


def test_an_answer_the_worker_never_got_is_not_told_to_try_again() -> None:
    """Every other refusal is fixed by resending; this one is not.

    The worker answered its own question and moved on, so the user has to react to
    a decision rather than resend. Telling them to resend would have them answer a
    question nobody is asking any more, and leaving ``awaiting_answer`` set would
    have ``accept`` refused for a question that is already resolved.
    """
    from nexus.extensions.tasks.hitl.actor.base import STATUS_ANSWER_MISSED
    from nexus.extensions.tasks.hitl.actor.pty_claude_code.manager import ANSWER_MISSED_REASON
    from nexus.extensions.tasks.hitl.user_agent.tools.orchestration import NOT_DELIVERED

    episode = run_episode(
        [[send('use OAuth')], [send('what did you pick?')], [accept()]],
        requirements=['x'],
        results=[
            FakeTurn(kind='refused', text='', raw={'status': STATUS_ANSWER_MISSED, 'reason': ANSWER_MISSED_REASON}),
            FakeTurn(text='I went with OAuth'),
        ],
    )

    assert episode.stopped, 'the run continues: the session is untouched'
    assert ANSWER_MISSED_REASON in episode.tool_results
    assert NOT_DELIVERED not in episode.tool_results, 'resending is the wrong advice here'
    assert not episode.state.awaiting_answer, 'the question is resolved, so accept must not be refused'
    assert episode.state.worker_rounds == 1


def test_giving_up_is_a_different_outcome_from_being_satisfied() -> None:
    episode = run_episode(
        [[send('this is still wrong')], [accept('gave_up', 'not worth more of my time')]], requirements=['x']
    )

    assert episode.state.stop_reason == 'gave_up'


def test_prose_with_no_action_is_nudged_and_does_not_spend_a_turn() -> None:
    """A turn that reached nobody is not one of the user's turns."""
    episode = run_episode([[], [send('right, sorry')], [accept()]], requirements=['x'])

    assert episode.stopped
    assert NUDGE in episode.text
    assert episode.manager.sent == ['right, sorry']
    assert episode.state.idle_turns == 1
    assert episode.model.turns == 4, 'four requests: the prose, the message, the held accept, the accept'
    assert episode.state.turns == 3, 'the prose turn is not among them'


# -- every call is answered ----------------------------------------------


def test_two_calls_in_one_turn_both_get_a_result() -> None:
    """An assistant message with two calls and one reply is malformed, and the
    endpoint rejects it on the following request."""
    episode = run_episode([[send('one'), send('two')], [accept()]], requirements=['x'])

    answered = [m['tool_call_id'] for m in episode.history if m.get('role') == 'tool']
    assert len(answered) == len(set(answered))
    assert {'t1_0', 't1_1'} <= set(answered)


def test_nothing_comes_between_a_batch_of_calls_and_its_answers() -> None:
    """A non-tool message among the answers makes the next request a 400.

    Measured on run dev-chain-hook-0911a: three `bash` calls in one turn, a reminder
    after the first answer, and azure gpt-5.6-sol named the two left unanswered.
    """
    episode = run_episode([[look(), look(), look()], [accept()]], requirements=['x'])

    for index, message in enumerate(episode.history):
        ids = [call['id'] for call in message.get('tool_calls') or []]
        if len(ids) < 2:
            continue
        answers = []
        for entry in episode.history[index + 1 :]:
            if entry.get('role') != 'tool':
                break
            answers.append(entry.get('tool_call_id'))
        assert set(ids) <= set(answers), f'message {index} left {sorted(set(ids) - set(answers))} unanswered'


def test_a_call_the_worker_fails_still_gets_a_result() -> None:
    class Broken(FakeManager):
        async def send(self, session: Any, message: Any) -> FakeTurn:
            raise RuntimeError('worker is gone')

    episode = run_episode([[send('hello')]], requirements=['x'], max_turns=1)
    episode.manager.__class__ = Broken

    assert any(m.get('role') == 'tool' for m in episode.history)


def test_an_unknown_worker_is_reported_to_the_model() -> None:
    episode = run_episode([[send('hello', to='nobody')]], requirements=['x'], max_turns=1)

    assert 'nobody' in episode.tool_results
    assert episode.manager.sent == []


def test_an_empty_message_is_refused_without_reaching_the_worker() -> None:
    episode = run_episode([[send('   ')]], requirements=['x'], max_turns=1)

    assert episode.manager.sent == []
    assert 'empty' in episode.tool_results


def test_a_message_with_no_draft_is_refused_and_can_be_sent_again() -> None:
    episode = run_episode(
        [[send('fix it', draft='')], [send('fix it')], [accept()]],
        requirements=['x'],
    )

    assert episode.manager.sent == ['fix it']
    assert episode.state.messages_sent == 1
    assert episode.stopped


# -- the guards ----------------------------------------------------------


def test_a_pending_question_blocks_acceptance_until_it_is_answered() -> None:
    episode = run_episode(
        [[send('fix it')], [accept()], [send('the second one', 'answer')], [accept()]],
        requirements=['x'],
        results=[FakeTurn(kind='needs_input', text='which format?'), FakeTurn(text='done')],
    )

    assert episode.stopped
    assert len(episode.manager.sent) == 2, 'the refused acceptance sent nothing, then the answer did'
    assert len(episode.state.accepted) == 1


def test_an_acceptance_that_accounts_for_nothing_is_refused() -> None:
    """A `satisfied` accept needs ``checked``, since that field is where the whole
    requirement gets gone through. The refusal leaves the requirement open."""
    episode = run_episode(
        [[send('build all four parts')], [accept(checked='')], [accept()]],
        requirements=['x'],
    )

    assert episode.stopped
    assert len(episode.state.accepted) == 1, 'the refused acceptance recorded nothing'


def test_giving_up_needs_no_account_of_the_parts() -> None:
    """``gave_up`` does not claim the work is done, so it carries no ``checked``."""
    episode = run_episode(
        [[send('build all four parts')], [accept(reason='gave_up', checked='')]],
        requirements=['x'],
    )

    assert episode.stopped
    assert [entry['reason'] for entry in episode.state.accepted] == ['gave_up']


def test_a_worker_claiming_completion_draws_a_reminder_to_check() -> None:
    from nexus.extensions.tasks.hitl.user_agent.prompts import HANDOVER_LOOK

    episode = run_episode([[send('fix it')], [accept()]], requirements=['x'])

    assert HANDOVER_LOOK in episode.reminders
    assert HANDOVER_LOOK not in episode.tool_results, 'not buried in the tool output'


def test_a_persona_is_recalled_after_a_message_reaches_the_worker() -> None:
    """The heavy reminder lists the persona's own lines, in its own reminder message."""
    episode = run_episode([[send('fix it')], [accept()]], requirements=['x'], persona='default')

    for line in get_persona('default').recalls:
        assert line in episode.reminders
        assert line not in episode.tool_results, 'not buried in the tool output'


# -- what the model was given --------------------------------------------


def test_the_working_directory_reaches_the_system_prompt() -> None:
    episode = run_episode([[accept()]], requirements=['x'], max_turns=1)

    assert '/testbed' in episode.history[0]['content']


def test_the_persona_reaches_the_system_prompt() -> None:
    episode = run_episode([[accept()]], requirements=['x'], persona='default', max_turns=1)

    assert get_persona('default').prompt in episode.history[0]['content']


def test_every_built_in_persona_can_drive_a_run() -> None:
    """A persona declaring a tool the assembly does not offer would otherwise only
    fail on whichever turn the model first reached for it."""
    from nexus.extensions.tasks.hitl.user_agent.personas import registered_personas

    for name in registered_personas():
        episode = run_episode([[send('hi')], [accept()]], requirements=['x'], persona=name)
        assert episode.stopped, name


# -- what a handover asks the user to go and do ----------------------------


def test_a_handover_tells_the_user_where_the_run_stands() -> None:
    """The user paces itself off its own progress rather than off the worker's claims."""
    episode = run_episode(
        [[send('first')], [accept()], [send('second')], [accept()]],
        requirements=['first', 'second'],
    )

    assert 'request 1' in episode.reminders
    assert 'request 2' in episode.reminders
    assert 'first time' in episode.reminders


def test_a_handover_reminder_names_no_command_of_its_own() -> None:
    """The workspace need not be a git repository."""
    episode = run_episode([[send('fix it')], [accept()]], requirements=['x'])

    assert 'request 1' in episode.reminders
    assert 'git diff' not in episode.reminders


# -- what the hook records ------------------------------------------------


def records(episode: Episode) -> dict[str, Any]:
    """What the hook wrote onto the run by the time it finished."""
    asyncio.run(notify_hooks([episode.hook], RunFinish(state=episode.state), episode.state))
    return episode.state.hook_results['swe_chain']


def test_the_hook_records_the_rounds_that_reached_the_worker() -> None:
    episode = run_episode(
        [[send('one')], [send('two')], [accept()]],
        requirements=['x'],
    )

    assert records(episode) == {'instance_id': 'test', 'rounds': 2}


def test_a_round_the_backend_refused_is_not_counted() -> None:
    """A refused send reached nobody."""
    episode = run_episode(
        [[send('/compact')], [send('fix it')], [accept()]],
        requirements=['x'],
        results=[FakeTurn(kind='refused', text='', raw={'reason': 'slash command'}), FakeTurn(text='done')],
    )

    assert records(episode)['rounds'] == 1
