"""Resolving run hooks, and what happens when one is declared wrong.

A hook is the seam a topic reaches through, so the failures worth covering are the
silent ones: a name that resolves to nothing, and a subscription to an event the
calling agent never fires. Either would leave a hook that never runs and never
reports why.

The events themselves belong to the agent that fires them, so these use
``agent_with_user``'s.
"""

import asyncio
import pathlib
import sys
import textwrap

from typing import Any

import pytest

from nexus.extensions.tasks.hitl.agents.agent_with_user import EVENTS, RunFinish, RunStart, WorkerDone
from nexus.extensions.tasks.hitl.user_agent.hooks import (
    HookError,
    HookEvent,
    HookOutcome,
    resolve_hooks,
    run_hook_registry,
)


def resolve(*configs: Any) -> Any:
    """``resolve_hooks`` against the events this agent fires."""
    return resolve_hooks(list(configs), events=EVENTS)


def one(*configs: Any) -> Any:
    return resolve(*configs)[0]


# -- nothing configured ----------------------------------------------------


@pytest.mark.parametrize('configs', [None, []], ids=['none', 'empty'])
def test_a_run_that_names_no_hook_gets_none(configs: Any) -> None:
    assert resolve_hooks(configs, events=EVENTS) == []


# -- the two resolution paths ----------------------------------------------


class Probe:
    """A hook used by both resolution paths below."""

    events = (RunFinish,)

    def __init__(self, tag: str = 'default') -> None:
        self.tag = tag

    async def on_event(self, event: HookEvent) -> HookOutcome:
        return HookOutcome(results={'probe': self.tag})

    async def on_failure(self, event: HookEvent, error: Exception) -> bool:
        return False


def register_once(name: str) -> None:
    if name not in run_hook_registry.get_all():
        run_hook_registry.register(name)(Probe)


def test_a_registered_name_resolves() -> None:
    register_once('probe_registered')

    hook = one({'name': 'probe_registered', 'args': {'tag': 'x'}})

    assert isinstance(hook, Probe)
    assert hook.tag == 'x', 'args reach the constructor'


def test_an_import_path_resolves_the_same_way() -> None:
    """What a topic uses when it registers nothing."""
    hook = one({'name': f'{__name__}::Probe', 'args': {'tag': 'y'}})

    assert isinstance(hook, Probe)
    assert hook.tag == 'y'


def test_both_paths_reach_the_same_class() -> None:
    register_once('probe_both_paths')

    by_name, by_path = resolve({'name': 'probe_both_paths'}, {'name': f'{__name__}::Probe'})

    assert type(by_name) is type(by_path)
    assert by_name.events == by_path.events


def test_each_configured_hook_gets_its_own_instance() -> None:
    """Two entries naming one class are two objects, each with the args it was given."""
    first, second = resolve(
        {'name': f'{__name__}::Probe', 'args': {'tag': 'first'}},
        {'name': f'{__name__}::Probe', 'args': {'tag': 'second'}},
    )

    assert first is not second
    assert (first.tag, second.tag) == ('first', 'second')


def test_registering_one_name_twice_is_refused() -> None:
    """Two hooks under one name would make which of them runs arbitrary."""
    from nexus.core.registry import DuplicateEntryError

    register_once('probe_duplicate')

    with pytest.raises(DuplicateEntryError):
        run_hook_registry.register('probe_duplicate')(Probe)


def test_a_hook_importing_nothing_from_hitl_still_works(tmp_path: pathlib.Path) -> None:
    """A topic writing plain Python: the event classes arrive as arguments."""
    module = tmp_path / 'foreign_topic_hook.py'
    module.write_text(
        textwrap.dedent("""
            class Ducky:
                def __init__(self, subscribe_to):
                    self.events = (subscribe_to,)

                async def on_event(self, event):
                    return None

                async def on_failure(self, event, error):
                    return False
        """),
        encoding='utf-8',
    )
    sys.path.insert(0, str(tmp_path))
    try:
        hook = one({'name': 'foreign_topic_hook::Ducky', 'args': {'subscribe_to': RunFinish}})
    finally:
        sys.path.remove(str(tmp_path))

    assert 'import' not in module.read_text(encoding='utf-8')
    assert hook.events == (RunFinish,)


# -- the refusals ----------------------------------------------------------


def test_an_unknown_registered_name_names_what_is_available() -> None:
    with pytest.raises(HookError, match='unknown hook'):
        resolve({'name': 'no_such_hook_anywhere'})


def test_an_import_path_pointing_at_nothing_is_refused() -> None:
    with pytest.raises(HookError, match='could not be imported'):
        resolve({'name': f'{__name__}::NoSuchClass'})


def test_a_config_with_no_name_is_refused() -> None:
    with pytest.raises(HookError, match='no name'):
        resolve({'args': {'tag': 'x'}})


def test_args_that_are_not_a_mapping_are_refused() -> None:
    with pytest.raises(HookError, match='must be a mapping'):
        resolve({'name': f'{__name__}::Probe', 'args': 3})


class AnotherAgentsEvent(HookEvent):
    name = 'hitl/somebody_else:worker.done'


class NoMethod:
    events = (RunStart,)


class NoFailureMethod:
    events = (RunStart,)

    async def on_event(self, event: HookEvent) -> None:
        return None


class NoEvents:
    async def on_event(self, event: HookEvent) -> None:
        return None

    async def on_failure(self, event: HookEvent, error: Exception) -> bool:
        return False


class ForeignEvent:
    events = (AnotherAgentsEvent,)

    async def on_event(self, event: HookEvent) -> None:
        return None

    async def on_failure(self, event: HookEvent, error: Exception) -> bool:
        return False


@pytest.mark.parametrize(
    ('cls', 'message'),
    [
        (NoMethod, 'defines no on_event'),
        (NoFailureMethod, 'defines no on_failure'),
        (NoEvents, 'subscribes to no event'),
        (ForeignEvent, 'never fires'),
    ],
    ids=['no on_event', 'no on_failure', 'no subscription', "another agent's event"],
)
def test_a_hook_that_could_never_fire_is_refused(cls: type, message: str) -> None:
    """An event belonging to a different agent is the silent failure this exists to
    catch: the object resolves, the run proceeds, and nothing the topic wrote ever
    executes."""
    with pytest.raises(HookError, match=message):
        resolve({'name': f'{__name__}::{cls.__name__}'})


# -- the events belong to the agent ----------------------------------------


def test_an_event_of_another_agent_is_a_different_class() -> None:
    """Two agents can each have a ``worker.done`` without a hook confusing them."""
    assert AnotherAgentsEvent not in EVENTS
    assert AnotherAgentsEvent.name != WorkerDone.name


def test_every_event_the_agent_declares_is_one_it_fires() -> None:
    """A class in ``EVENTS`` that no run reaches would be a position a topic can
    subscribe to and never see fire."""
    import inspect

    from nexus.extensions.tasks.hitl.agents import agent_with_user

    source = inspect.getsource(agent_with_user)

    for cls in EVENTS:
        assert f'{cls.__name__}(state=state' in source, f'{cls.__name__} is declared but no run fires it'


# -- what a hook hands back ------------------------------------------------


def test_an_outcome_carries_what_the_run_should_record() -> None:
    outcome = HookOutcome(results={'mine': 1})

    assert outcome.results == {'mine': 1}


def test_a_hook_can_be_driven_through_its_one_method() -> None:
    hook = one({'name': f'{__name__}::Probe', 'args': {'tag': 'z'}})

    outcome = asyncio.run(hook.on_event(None))  # type: ignore[arg-type]

    assert outcome.results == {'probe': 'z'}
