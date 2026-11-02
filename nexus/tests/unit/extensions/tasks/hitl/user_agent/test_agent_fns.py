"""The two ready-made agent functions, and what a host can actually reach.

The risk these cover is not the loop -- that is tested where it lives -- but the
contract with whatever calls them. An agent input drops any field it does not
declare (``runner/single.py:395-402``), silently, so what a host passes and what
arrives are only equal if the declarations line up. And a run that fails partway has
to report what it managed rather than raising through the task.
"""

import asyncio

from typing import Any

import pytest

from nexus.core import ExecutionContext, agent_registry
from nexus.extensions.tasks.hitl.actor.base import TurnResult
from nexus.extensions.tasks.hitl.agents.agent_with_user import (
    AgentWithUserInput,
    AgentWithUserOutput,
    RunFinish,
    RunStart,
    WorkerDone,
    run_agent_with_user,
)
from nexus.extensions.tasks.hitl.user_agent.hooks import HookEvent, HookOutcome
from nexus.runner.single import _build_input
from nexus.trace.context import context_processors
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor


@pytest.fixture
def labelling_tracker() -> Any:
    """A run-level tracker in the mode labels need.

    Both agents refuse to start without one, so a test about anything later has to
    install it first.
    """
    token = context_processors.set([ChatTrackerProcessor(tools_mode='branch_last')])
    yield
    context_processors.reset(token)


# -- registration --------------------------------------------------------


def test_the_agent_is_reachable_by_the_name_a_generator_would_use() -> None:
    assert agent_registry.get('hitl/agent_with_user') is not None


# -- the input contract with a host --------------------------------------


def test_the_host_facing_agent_needs_only_what_installed_agents_already_take() -> None:
    required = {name for name, field in AgentWithUserInput.model_fields.items() if field.is_required()}

    assert required == {'user_prompt', 'working_dir'}


def test_a_host_passing_only_a_prompt_and_a_directory_builds() -> None:
    built = _build_input(AgentWithUserInput, {'user_prompt': 'fix the bug', 'working_dir': '/app'})

    assert built.user_prompt == 'fix the bug'
    assert built.working_dir == '/app'
    assert built.requirements == []


def test_a_field_the_agent_does_not_declare_is_dropped_by_the_runner() -> None:
    """Not a defect to fix here, but the reason working_dir cannot be inferred."""
    built = _build_input(
        AgentWithUserInput,
        {'user_prompt': 'x', 'working_dir': '/app', 'max_iterations': 30, 'tools': ['read']},
    )

    assert not hasattr(built, 'max_iterations')
    assert not hasattr(built, 'tools')


def test_a_missing_working_directory_fails_loudly_rather_than_defaulting() -> None:
    with pytest.raises(Exception, match='working_dir'):
        _build_input(AgentWithUserInput, {'user_prompt': 'x'})


def test_a_requirement_list_wins_over_the_single_prompt() -> None:
    """What the chain task hands over, and what any multi-turn task can."""
    built = _build_input(
        AgentWithUserInput,
        {'requirements': ['one', 'two', 'three'], 'user_prompt': 'combined', 'working_dir': '/testbed'},
    )

    assert built.requirements == ['one', 'two', 'three']


def test_hooks_are_declared_so_a_recipe_can_name_its_own_logic() -> None:
    built = _build_input(
        AgentWithUserInput,
        {
            'user_prompt': 'x',
            'working_dir': '/testbed',
            'hooks': [{'name': 'swe_chain', 'args': {'instance_id': 'a'}}],
        },
    )

    assert built.hooks == [{'name': 'swe_chain', 'args': {'instance_id': 'a'}}]


# -- failure reporting ---------------------------------------------------


def run(agent: Any, agent_input: Any) -> Any:
    return asyncio.run(agent(agent_input))


def test_a_run_with_no_requirement_at_all_is_refused_before_any_worker_starts() -> None:
    with pytest.raises(ValueError, match='requirements or a user_prompt'):
        run(run_agent_with_user, AgentWithUserInput(user_prompt='  ', working_dir='/app'))


def test_a_failure_reaching_the_worker_is_reported_not_raised() -> None:
    """A task must be able to score a failed run, so this returns rather than raises."""
    output = run(
        run_agent_with_user,
        AgentWithUserInput(user_prompt='fix it', working_dir='/app', worker_backend='print_claude_code'),
    )

    assert isinstance(output, AgentWithUserOutput)
    assert not output.success
    assert output.stderr, 'the reason is reported'
    assert output.run_meta['status'] == 'error'
    assert output.exit_code == 1
    assert output.run_meta['total_steps'] == 1, 'what was attempted is still known'


def test_a_multi_requirement_run_reports_a_failure_the_same_way() -> None:
    output = run(
        run_agent_with_user,
        AgentWithUserInput(requirements=['one', 'two'], user_prompt='', working_dir='/testbed'),
    )

    assert isinstance(output, AgentWithUserOutput)
    assert not output.success
    assert output.stderr
    assert output.run_meta['total_steps'] == 2


def test_a_hook_that_cannot_be_resolved_raises_before_a_sandbox_is_paid_for() -> None:
    """A misconfigured hook is a configuration error, unlike a run that fell over."""
    from nexus.extensions.tasks.hitl.user_agent.hooks import HookError

    with pytest.raises(HookError, match='no_such_hook'):
        run(
            run_agent_with_user,
            AgentWithUserInput(user_prompt='x', working_dir='/app', hooks=[{'name': 'no_such_hook'}]),
        )


def test_an_unknown_worker_backend_is_reported_rather_than_raising(labelling_tracker) -> None:
    async def go() -> Any:
        async with ExecutionContext(tool_state={}):
            return await run_agent_with_user(
                AgentWithUserInput(user_prompt='x', working_dir='/app', worker_backend='not_a_backend')
            )

    output = asyncio.run(go())

    assert not output.success
    assert 'not_a_backend' in output.stderr


def test_an_unknown_persona_is_reported_rather_than_raising(labelling_tracker) -> None:
    async def go() -> Any:
        async with ExecutionContext(tool_state={}):
            return await run_agent_with_user(
                AgentWithUserInput(user_prompt='x', working_dir='/app', persona='nonexistent')
            )

    output = asyncio.run(go())

    assert not output.success
    assert 'nonexistent' in output.stderr


def test_a_missing_execution_context_is_reported_like_any_other_failure() -> None:
    """The check used to sit above the try, so this raised through the task."""
    output = run(
        run_agent_with_user,
        AgentWithUserInput(user_prompt='x', working_dir='/app'),
    )

    assert not output.success
    assert 'ExecutionContext' in output.stderr
    assert output.run_meta['status'] == 'error'


# -- output shape --------------------------------------------------------


def test_the_host_facing_output_keeps_the_installed_agent_shape() -> None:
    fields = set(AgentWithUserOutput.model_fields)

    assert {'success', 'output', 'stderr', 'exit_code', 'execution_time'} <= fields, (
        'a host that already reads an installed agent output keeps working'
    )


def test_the_trajectories_are_read_off_the_object_not_out_of_a_dump() -> None:
    """They are excluded from the dump: the run record already carries the same
    conversations under ``processor_results.chat``, which is where an exported run is
    read from. A holder of this object -- the reward-hacking detector -- still gets them."""
    output = AgentWithUserOutput(
        success=True,
        output='',
        stderr='',
        exit_code=0,
        execution_time=0.0,
        trajectories={'user': {'traces': {'trajectories': [{'role': 'user'}]}}},
    )

    assert output.trajectories['user'], 'the attribute reads'
    assert 'trajectories' not in output.model_dump(), 'a dump does not carry a second copy'


def test_giving_up_and_being_satisfied_are_distinguishable_in_the_output() -> None:
    def ended(reason: str) -> AgentWithUserOutput:
        return AgentWithUserOutput(
            success=True, output='', stderr='', exit_code=0, execution_time=0.0, run_meta={'stop_reason': reason}
        )

    assert ended('satisfied').run_meta['stop_reason'] != ended('gave_up').run_meta['stop_reason']


def test_what_a_hook_returned_and_what_it_broke_on_are_both_reported() -> None:
    output = AgentWithUserOutput(
        success=True,
        output='',
        stderr='',
        exit_code=0,
        execution_time=0.0,
        hook_results={'mytopic': {'artifact_bytes': 12}},
        hook_failures=[f'{RunFinish.name}: RuntimeError: no such path'],
    )

    dumped = output.model_dump()
    assert dumped['hook_results']['mytopic']['artifact_bytes'] == 12
    assert RunFinish.name in dumped['hook_failures'][0]


# -- the calls the agents make into a manager ----------------------------


def _one_send(result: Any, *, message: str = 'hi') -> Any:
    """Run the generic ``send_message`` handler once over ``result``.

    ``state.manager.sends`` counts what reached the backend.
    """
    from nexus.core import ToolResponse
    from nexus.extensions.tasks.hitl.agents.agent_with_user import RunState, build_handlers
    from nexus.extensions.tasks.hitl.user_agent.queue import Requirements

    class Manager:
        sends = 0

        async def send(self, session: Any, message: Any) -> Any:
            Manager.sends += 1
            return result

        def report(self, turn: Any) -> str:
            return '<worker_progress>done</worker_progress>'

    state = RunState(queue=Requirements.build(['one', 'two']), manager=Manager(), session=object())
    handler = build_handlers(state)['send_message']
    sent = asyncio.run(handler(Call(), ToolResponse(output=[]), {'message': message, 'draft': 'd'}))
    return state, sent


class Call:
    """The shape a provider returns for one tool call."""

    id = 'c1'

    class function:  # noqa: N801 - matches the openai call shape
        name = 'send_message'
        arguments = '{}'


def test_a_message_the_backend_refused_is_not_counted_as_a_round() -> None:
    """``TurnResult.delivered`` is false when a backend blocks the message.

    The session is untouched, so a run that counted the round would report a
    conversation the worker was never part of, and the user would be told its words
    arrived.
    """
    from nexus.extensions.tasks.hitl.actor.base import TurnResult
    from nexus.extensions.tasks.hitl.user_agent.tools.orchestration import NOT_DELIVERED

    state, sent = _one_send(TurnResult(kind='refused', output='', raw={'reason': '/foo is not registered'}))

    assert state.worker_rounds == 0 and state.messages_sent == 0
    assert NOT_DELIVERED in sent.content
    assert '/foo is not registered' in sent.content
    assert '<sent>' not in sent.content, 'the user must not read its own message back as delivered'


def test_a_late_answer_is_not_answered_with_advice_to_resend() -> None:
    """``answer_missed`` carries a reason already addressed to the user."""
    from nexus.extensions.tasks.hitl.actor.base import STATUS_ANSWER_MISSED, TurnResult
    from nexus.extensions.tasks.hitl.user_agent.tools.orchestration import NOT_DELIVERED

    result = TurnResult(
        kind='refused',
        output='',
        raw={'status': STATUS_ANSWER_MISSED, 'reason': 'The worker answered it itself and moved on.'},
    )
    state, sent = _one_send(result)

    assert state.worker_rounds == 0
    assert NOT_DELIVERED not in sent.content
    assert 'answered it itself' in sent.content


def test_a_worker_handing_work_back_draws_a_reminder_to_look_at_the_change() -> None:
    """The generic agent gets the handover reminder the chain recipe has."""
    from nexus.extensions.tasks.hitl.actor.base import TurnResult
    from nexus.extensions.tasks.hitl.user_agent.prompts import HANDOVER_LOOK

    state, sent = _one_send(TurnResult(kind='completed', output='built it', raw={}))

    assert state.handovers == 1
    reminders = ' '.join(str(message.get('content')) for message in sent.messages)
    assert HANDOVER_LOOK in reminders
    assert 'request 1' in reminders and 'first time' in reminders
    assert 'git diff' not in reminders, 'the workspace need not be a git repository'


def test_a_question_from_the_worker_is_not_a_handover() -> None:
    """A turn blocked on a question has handed nothing back to look at."""
    from nexus.extensions.tasks.hitl.actor.base import TurnResult
    from nexus.extensions.tasks.hitl.user_agent.prompts import HANDOVER_LOOK

    state, sent = _one_send(TurnResult(kind='needs_input', output='which one?', raw={}))

    assert state.awaiting_answer and state.handovers == 0
    reminders = ' '.join(str(message.get('content')) for message in sent.messages)
    assert HANDOVER_LOOK not in reminders


def test_a_delivered_message_is_followed_by_the_language_it_has_to_be_in() -> None:
    """Both ways a worker turn can come back: with work, and with a question."""
    from nexus.extensions.tasks.hitl.actor.base import TurnResult

    for result in (
        TurnResult(kind='completed', output='built it', raw={}),
        TurnResult(kind='needs_input', output='which one?', raw={}),
    ):
        state, sent = _one_send(result)
        reminders = ' '.join(str(message.get('content')) for message in sent.messages)
        assert state.natural_language.name in reminders


def test_a_message_in_the_wrong_language_never_reaches_the_worker() -> None:
    """Refusing after the send would have delivered it and then complained."""
    from nexus.extensions.tasks.hitl.actor.base import TurnResult

    state, sent = _one_send(TurnResult(kind='completed', output='built it', raw={}), message='分页不对')

    assert type(state.manager).sends == 0
    assert state.messages_sent == 0 and state.worker_rounds == 0
    assert state.natural_language.name in sent.content


# -- the hooks the agent calls --------------------------------------------


class RecordingHook:
    """Records the events it received, in order.

    On the class, not the instance: the agent resolves and builds its own hooks, so an
    instance a test holds is never one that ran.
    """

    events = (RunStart, WorkerDone, RunFinish)
    seen: list[str] = []

    def __init__(self, tag: str = 'one') -> None:
        self.tag = tag

    async def on_event(self, event: HookEvent) -> HookOutcome | None:
        RecordingHook.seen.append(f'{self.tag}:{event.name}')
        if isinstance(event, RunFinish):
            return HookOutcome(results={self.tag: len(RecordingHook.seen)})
        return None

    async def on_failure(self, event: HookEvent, error: Exception) -> bool:
        return False


def test_a_runs_events_reach_every_hook_the_agent_built(labelling_tracker) -> None:
    """Asserting which events arrived rather than 'nothing raised': an event the agent
    never fires leaves a subscriber silently inert. Two entries of one class also say
    the agent builds an instance per entry rather than sharing one.

    There is no sandbox here, so the worker never comes up. That happens before
    ``run.start``, leaving ``run.finish`` as the only event of the run.
    """
    RecordingHook.seen = []
    path = f'{RecordingHook.__module__}::RecordingHook'

    async def go() -> Any:
        async with ExecutionContext(tool_state={}):
            return await run_agent_with_user(
                AgentWithUserInput(
                    user_prompt='x',
                    working_dir='/app',
                    hooks=[
                        {'name': path, 'args': {'tag': 'first'}},
                        {'name': path, 'args': {'tag': 'second'}},
                    ],
                )
            )

    output = asyncio.run(go())

    assert not output.success, 'the worker cannot start without a sandbox'
    assert output.hook_failures == [], 'a failing run is not a failing hook'
    assert RecordingHook.seen == [
        f'first:{RunFinish.name}',
        f'second:{RunFinish.name}',
    ], 'both are closed even on the failure path, in the order configured'
    assert set(output.hook_results) == {'first', 'second'}


class RequiredEverywhere:
    """A hook that ends the run when it fails, resolved by import path from below."""

    events = (RunStart, RunFinish)

    async def on_event(self, event: HookEvent) -> None:
        raise RuntimeError('artifact missing')

    async def on_failure(self, event: HookEvent, error: Exception) -> bool:
        return True


def test_a_raising_hook_is_recorded_rather_than_ending_the_run() -> None:
    """The run it decorates has an outcome of its own, which a task must still score.

    A hook configured after the one that raised still runs.
    """
    from nexus.extensions.tasks.hitl.agents.agent_with_user import RunState, notify_hooks
    from nexus.extensions.tasks.hitl.user_agent.queue import Requirements

    class Broken:
        events = (RunFinish,)

        async def on_event(self, event: HookEvent) -> None:
            raise RuntimeError('no such path')

        async def on_failure(self, event: HookEvent, error: Exception) -> bool:
            return False

    class Later:
        events = (RunFinish,)

        async def on_event(self, event: HookEvent) -> HookOutcome:
            return HookOutcome(results={'later': True})

        async def on_failure(self, event: HookEvent, error: Exception) -> bool:
            return False

    state = RunState(queue=Requirements.build(['one']))
    asyncio.run(notify_hooks([Broken(), Later()], RunFinish(state=state), state))

    assert state.hook_failures == [f'{RunFinish.name}: RuntimeError: no such path']
    assert state.hook_results == {'later': True}
    assert not state.error, 'the run itself is untouched'


def test_a_hook_can_end_the_run_on_its_own_failure() -> None:
    """The opposite default, asked of the hook at the moment it fails.

    ``on_failure`` sees which event went wrong and what was raised, so a hook whose
    upload matters can end the run while its logging on another event does not.
    """
    from nexus.extensions.tasks.hitl.agents.agent_with_user import RunState, notify_hooks
    from nexus.extensions.tasks.hitl.user_agent.hooks import RequiredHookError
    from nexus.extensions.tasks.hitl.user_agent.queue import Requirements

    class MustUpload:
        events = (RunFinish,)

        async def on_event(self, event: HookEvent) -> None:
            raise RuntimeError('no such path')

        async def on_failure(self, event: HookEvent, error: Exception) -> bool:
            return isinstance(error, RuntimeError)

    class Later:
        events = (RunFinish,)
        ran = False

        async def on_event(self, event: HookEvent) -> None:
            Later.ran = True

        async def on_failure(self, event: HookEvent, error: Exception) -> bool:
            return False

    state = RunState(queue=Requirements.build(['one']))

    with pytest.raises(RequiredHookError, match='no such path') as caught:
        asyncio.run(notify_hooks([MustUpload(), Later()], RunFinish(state=state), state))

    assert caught.value.event == RunFinish.name
    assert isinstance(caught.value.cause, RuntimeError)
    assert state.hook_failures == [f'{RunFinish.name}: RuntimeError: no such path'], 'recorded either way'
    assert not Later.ran, 'the hooks after it do not run'


def test_one_hook_can_be_fatal_on_one_event_and_not_another() -> None:
    """Which a flag on the class cannot express: the same hook logs on one event and
    uploads on another, and only the upload is worth ending the run over."""
    from nexus.extensions.tasks.hitl.agents.agent_with_user import RunState, notify_hooks
    from nexus.extensions.tasks.hitl.user_agent.hooks import RequiredHookError
    from nexus.extensions.tasks.hitl.user_agent.queue import Requirements

    class LogsThenUploads:
        events = (RunStart, RunFinish)

        async def on_event(self, event: HookEvent) -> None:
            raise RuntimeError('failed')

        async def on_failure(self, event: HookEvent, error: Exception) -> bool:
            return isinstance(event, RunFinish)

    state = RunState(queue=Requirements.build(['one']))
    hooks = [LogsThenUploads()]

    asyncio.run(notify_hooks(hooks, RunStart(state=state), state))

    with pytest.raises(RequiredHookError):
        asyncio.run(notify_hooks(hooks, RunFinish(state=state), state))

    assert len(state.hook_failures) == 2, 'both were recorded; only the second stopped anything'


def test_a_hooks_failure_does_not_mask_why_the_run_died(labelling_tracker) -> None:
    """Two failures at once: the run's own, and the hook that then also failed.

    ``stderr`` keeps the run's, which is the one that explains the other. The hook's is
    still in ``hook_failures``.
    """

    async def go() -> Any:
        async with ExecutionContext(tool_state={}):
            return await run_agent_with_user(
                AgentWithUserInput(
                    user_prompt='x',
                    working_dir='/app',
                    hooks=[{'name': f'{RequiredEverywhere.__module__}::RequiredEverywhere'}],
                )
            )

    output = asyncio.run(go())

    assert not output.success
    assert output.run_meta['status'] == 'error'
    assert 'No runtime provider' in output.stderr, 'the run got no sandbox, which came first'
    assert output.hook_failures == [f'{RunFinish.name}: RuntimeError: artifact missing']


def test_a_run_that_finished_is_failed_by_its_hook() -> None:
    """A hook ending the run at ``run.finish`` has to reach the output as a failure.

    The agent catches it in its ``finally``; asserting the catch is there as well, since
    driving the real one needs a sandbox and a run that reaches the end.
    """
    import inspect

    from nexus.extensions.tasks.hitl.agents import agent_with_user
    from nexus.extensions.tasks.hitl.agents.agent_with_user import RunState, notify_hooks
    from nexus.extensions.tasks.hitl.user_agent.hooks import RequiredHookError
    from nexus.extensions.tasks.hitl.user_agent.queue import Requirements

    source = inspect.getsource(agent_with_user.run_agent_with_user)
    assert 'except RequiredHookError' in source, 'a raise here would escape past the output'
    assert 'state.error = state.error or str(exc)' in source

    state = RunState(queue=Requirements.build(['one']), stopped=True, stop_reason='satisfied')
    assert state.succeeded, 'the run itself went fine'

    with pytest.raises(RequiredHookError, match='artifact missing'):
        asyncio.run(notify_hooks([RequiredEverywhere()], RunFinish(state=state), state))

    state.error = state.error or 'x'
    assert not state.succeeded and state.status == 'error'


def test_an_event_the_hook_did_not_subscribe_to_is_not_delivered() -> None:
    from nexus.extensions.tasks.hitl.agents.agent_with_user import RunState, notify_hooks
    from nexus.extensions.tasks.hitl.user_agent.queue import Requirements

    class Sparse:
        events = (RunFinish,)
        seen: list[str] = []

        async def on_event(self, event: HookEvent) -> None:
            Sparse.seen.append(event.name)

        async def on_failure(self, event: HookEvent, error: Exception) -> bool:
            return False

    state = RunState(queue=Requirements.build(['one']))
    hooks = [Sparse()]

    asyncio.run(notify_hooks(hooks, RunStart(state=state), state))
    asyncio.run(notify_hooks(hooks, RunFinish(state=state), state))
    asyncio.run(notify_hooks([], RunFinish(state=state), state))

    assert Sparse.seen == [RunFinish.name]
    assert state.hook_failures == []


def test_what_a_hook_returns_reaches_the_run() -> None:
    from nexus.extensions.tasks.hitl.agents.agent_with_user import WORKER_NAME, RunState, notify_hooks
    from nexus.extensions.tasks.hitl.user_agent.queue import Requirements

    class Recorder:
        events = (WorkerDone,)

        async def on_event(self, event: HookEvent) -> HookOutcome:
            assert isinstance(event, WorkerDone)
            return HookOutcome(results={'mine': event.result.kind, 'worker': event.worker})

        async def on_failure(self, event: HookEvent, error: Exception) -> bool:
            return False

    state = RunState(queue=Requirements.build(['one']))
    done = WorkerDone(state=state, worker=WORKER_NAME, result=TurnResult(kind='completed', output='x', raw={}))
    asyncio.run(notify_hooks([Recorder()], done, state))

    assert state.hook_results == {'mine': 'completed', 'worker': WORKER_NAME}


@pytest.mark.parametrize(
    'module',
    ['nexus.extensions.tasks.hitl.agents.agent_with_user'],
)
def test_the_agents_start_a_session_with_a_keyword_every_manager_accepts(module: str) -> None:
    """``SessionManager.start`` takes ``**kwargs``, so a wrong keyword is not a
    type error -- it reaches the manager and raises there, on a real run, after a
    sandbox has been paid for. This checks the name against every backend."""
    import ast
    import importlib
    import inspect as inspect_mod

    from nexus.extensions.tasks.hitl.actor.registry import worker_manager_registry

    source = inspect_mod.getsource(importlib.import_module(module))
    keywords: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == 'start':
            keywords |= {kw.arg for kw in node.keywords if kw.arg}
    assert keywords, f'{module} never starts a session'

    for backend, manager_cls in worker_manager_registry.get_all().items():
        accepted = set(inspect_mod.signature(manager_cls.start).parameters)
        unknown = keywords - accepted
        assert not unknown, f'{module} passes {sorted(unknown)} to start(); {backend} accepts {sorted(accepted)}'


# -- trace attribution ----------------------------------------------------


@pytest.mark.parametrize(
    'module',
    ['nexus.extensions.tasks.hitl.agents.agent_with_user'],
)
def test_the_agents_label_their_own_turns(module: str) -> None:
    """A user turn has to carry a label, or no processor can attribute it.

    The label rides on the completion span, read off ``_chat_phase``: a stock client
    has no such attribute, so ``enable_labels`` has to rebind the class and the
    request has to be made inside a ``labelled`` block. Nothing raises when either is
    missing -- the chat tracker, the usage counters and the Langfuse upload all fall
    back to ``UNATTRIBUTED``, so the user's turns merge with whatever else is
    unlabelled and only the export shows it, after the run.

    Measured on run 0816a before this: 14 of 60 completions exported under
    ``_unattributed``, all of them the user's.

    Checked in the source because the alternative is a full run: the client is
    fetched from the context, and a stub client that records its own class rebinding
    would still not prove the request was made inside the block.
    """
    import importlib
    import inspect as inspect_mod

    source = inspect_mod.getsource(importlib.import_module(module))

    assert 'enable_labels(' in source, f'{module} never gives its client a _chat_phase'
    assert 'labelled(' in source, f'{module} never opens a labelled block'
    # Order matters: labelled() only sets a contextvar, and a client whose class was
    # never rebound does not read it.
    assert source.index('enable_labels(') < source.index('labelled('), (
        f'{module} opens its labelled block before enabling labels on the client'
    )


def test_the_agent_refuses_a_run_whose_tracker_cannot_carry_labels() -> None:
    """It labels its user turns, so it needs the mode a label survives.

    Against a ``global_first`` tracker it would otherwise run to completion, exporting
    nothing for either side, the first sign of it being an empty record after the run.
    """

    async def go() -> Any:
        async with ExecutionContext(tool_state={}):
            return await run_agent_with_user(AgentWithUserInput(user_prompt='fix it', working_dir='/app'))

    token = context_processors.set([ChatTrackerProcessor(tools_mode='global_first')])
    try:
        output = asyncio.run(go())
    finally:
        context_processors.reset(token)

    assert 'branch_last' in output.stderr, f'the mode is what it stopped on, got: {output.stderr[:120]}'
