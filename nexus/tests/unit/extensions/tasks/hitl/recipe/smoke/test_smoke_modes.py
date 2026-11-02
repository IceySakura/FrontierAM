"""The two ways a smoke run drives the user, and what each one can check.

A smoke run tests plumbing, so the risk it carries is an instruction that never
arrives: an injection on a turn nothing reaches, a scripted call for a tool nothing
offers. Both would report a broken mechanism as a failing check, with the ask itself
missing. Those refusals are what most of this file pins.

The other half is the drift the two loops invite. ``agents/agent_with_user`` is the
production loop and this recipe writes its own, so a change to when the production loop
compacts or reminds does not reach here. ``test_the_smoke_loop_still_does_what_the_
production_loop_does`` compares the two by the calls they make.

No sandbox is started. What a real run adds is whether the worker acts on any of it.
"""

import asyncio
import inspect

from typing import Any

import pytest

from nexus.core import ExecutionContext, agent_registry
from nexus.extensions.tasks.hitl.agents import agent_with_user as production
from nexus.extensions.tasks.hitl.recipe.smoke import agent as smoke
from nexus.extensions.tasks.hitl.recipe.smoke.agent import (
    Injection,
    ScriptedTurn,
    SmokeAgentInput,
    SmokeAgentOutput,
    check_injections,
    check_script,
    run_smoke_agent,
    scripted_call,
)
from nexus.trace.context import context_processors
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor


@pytest.fixture
def labelling_tracker() -> Any:
    """A run-level tracker in the mode labels need; the agent refuses to start without."""
    token = context_processors.set([ChatTrackerProcessor(tools_mode='branch_last')])
    yield
    context_processors.reset(token)


def run(agent_input: SmokeAgentInput) -> Any:
    async def go() -> Any:
        async with ExecutionContext(tool_state={}):
            return await run_smoke_agent(agent_input)

    return asyncio.run(go())


# -- registration ----------------------------------------------------------


def test_the_agent_is_reachable_by_the_name_the_generator_uses() -> None:
    assert agent_registry.get(smoke.AGENT_NAME) is not None


def test_the_output_keeps_the_shape_the_task_reads() -> None:
    """The smoke task reads ``AgentWithUserOutput`` fields off whatever it is given."""
    assert issubclass(SmokeAgentOutput, production.AgentWithUserOutput)
    assert 'mode' in SmokeAgentOutput.model_fields


# -- a scripted turn is the live path's call -------------------------------


def test_a_scripted_step_becomes_the_call_shape_a_dispatcher_takes() -> None:
    """The point of replay: only where the call comes from changes. The dispatcher, the
    handlers and the history entry are the live path's."""
    call = scripted_call(ScriptedTurn(tool='send_message', args={'message': 'hi'}), index=1)

    assert call.function.name == 'send_message'
    assert call.function.arguments == '{"message": "hi"}'
    assert call.type == 'function'
    assert call.id


def test_a_step_with_no_arguments_is_still_a_valid_call() -> None:
    """``accept`` takes none, and json.dumps({}) has to be '{}' rather than empty."""
    call = scripted_call(ScriptedTurn(tool='accept'), index=2)

    assert call.function.arguments == '{}'


def test_each_scripted_call_gets_its_own_id() -> None:
    """A tool reply is matched to its call by id; two the same would pair wrongly."""
    first = scripted_call(ScriptedTurn(tool='accept'), index=1)
    second = scripted_call(ScriptedTurn(tool='accept'), index=2)

    assert first.id != second.id


# -- a script the run cannot execute is refused -----------------------------


def test_a_script_calling_a_tool_the_run_does_not_offer_is_refused() -> None:
    """The dispatcher reports an unknown tool as content, which a model reads and works
    around. A script reads nothing, so the step is simply lost."""
    with pytest.raises(ValueError, match='browse'):
        check_script([ScriptedTurn(tool='browse')], offered=['send_message', 'accept'])


def test_an_empty_script_is_refused() -> None:
    """A replay with no steps would start a sandbox and do nothing in it."""
    with pytest.raises(ValueError, match='at least one'):
        check_script([], offered=['send_message'])


def test_a_script_of_offered_tools_is_accepted() -> None:
    check_script(
        [ScriptedTurn(tool='send_message', args={'message': 'x'}), ScriptedTurn(tool='accept')],
        offered=['send_message', 'accept', 'read'],
    )


# -- an injection no turn reaches is refused --------------------------------


def test_an_injection_past_the_turn_cap_is_refused() -> None:
    """It would never fire, and the check looking for what it asked for would fail for
    the wrong reason."""
    with pytest.raises(ValueError, match='past the cap'):
        check_injections([Injection(turn=9, text='search now')], cap=4)


def test_an_injection_on_turn_zero_is_refused() -> None:
    """Turns are 1-based, matching the log. Turn 0 reads as "before the run"."""
    with pytest.raises(ValueError, match='1-based'):
        check_injections([Injection(turn=0, text='x')], cap=4)


def test_two_injections_on_one_turn_are_refused() -> None:
    """One would silently win, and which one depends on ordering."""
    with pytest.raises(ValueError, match='more than once'):
        check_injections([Injection(turn=2, text='a'), Injection(turn=2, text='b')], cap=4)


def test_injections_within_the_cap_are_accepted() -> None:
    check_injections([Injection(turn=1, text='a'), Injection(turn=4, text='b')], cap=4)


def test_no_injections_is_the_ordinary_live_run() -> None:
    check_injections([], cap=4)


# -- the two modes do not read each other's fields --------------------------


def test_a_replay_carrying_injections_is_refused(labelling_tracker) -> None:
    """There is no model in a replay to instruct, so they would be ignored."""
    output = run(
        SmokeAgentInput(
            user_prompt='x',
            working_dir='/app',
            mode='replay',
            script=[{'tool': 'accept'}],
            injections=[{'turn': 1, 'text': 'search'}],
        )
    )

    assert not output.success
    assert 'live mode only' in output.stderr


def test_a_live_run_carrying_a_script_is_refused(labelling_tracker) -> None:
    """Otherwise the script looks configured and never runs."""
    output = run(
        SmokeAgentInput(user_prompt='x', working_dir='/app', mode='live', script=[{'tool': 'accept'}])
    )

    assert not output.success
    assert 'replay mode only' in output.stderr


def test_a_replay_with_no_script_is_refused(labelling_tracker) -> None:
    output = run(SmokeAgentInput(user_prompt='x', working_dir='/app', mode='replay'))

    assert not output.success
    assert 'at least one' in output.stderr


# -- failures are reported, as the production agent reports them ------------


def test_a_failure_reaching_the_worker_is_reported_not_raised(labelling_tracker) -> None:
    """A smoke task has to be able to run its command checks against what a failed
    conversation left behind."""
    output = run(
        SmokeAgentInput(
            user_prompt='x',
            working_dir='/app',
            mode='replay',
            script=[{'tool': 'accept'}],
            worker_backend='not_a_backend',
        )
    )

    assert isinstance(output, SmokeAgentOutput)
    assert not output.success
    assert 'not_a_backend' in output.stderr
    assert output.mode == 'replay', 'the mode is reported even by a run that failed'


def test_the_mode_reaches_the_output(labelling_tracker) -> None:
    """The task reports it because the other numbers are not comparable across modes."""
    output = run(SmokeAgentInput(user_prompt='x', working_dir='/app', mode='live'))

    assert output.mode == 'live'


# -- the two loops must not drift apart ------------------------------------


LOOP_CALLS = (
    'reminders.after_tool',
    'budget.exceeded',
    'state.stopped',
    'dispatcher.dispatch',
    'reminder_message',
    'tool_message',
    'compact(',
)


@pytest.mark.parametrize('expression', LOOP_CALLS)
def test_the_smoke_loop_still_does_what_the_production_loop_does(expression: str) -> None:
    """Two turn loops for one conversation, which is what this recipe chose.

    Read from the source because the alternative is a real run of each. A production
    loop that grows a step this one lacks means a smoke run stops testing the thing it
    was pointed at -- silently, since a loop that reminds less still finishes.
    """
    production_source = inspect.getsource(production.run_agent_with_user)
    smoke_source = inspect.getsource(smoke.run_smoke_agent)

    if expression not in production_source:
        pytest.skip(f'{expression} is no longer in the production loop')
    assert expression in smoke_source, f'the production loop calls {expression} and the smoke loop does not'


def test_both_loops_share_the_run_state_and_its_handlers() -> None:
    """What keeps the drift to the loop itself: sending, accepting and reporting are
    one implementation."""
    smoke_source = inspect.getsource(smoke)

    assert 'from nexus.extensions.tasks.hitl.agents.agent_with_user import' in smoke_source
    assert smoke.RunState is production.RunState
    assert smoke.build_handlers is production.build_handlers
