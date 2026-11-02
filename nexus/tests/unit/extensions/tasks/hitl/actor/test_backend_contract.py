"""What a recipe may assume of any worker backend.

The factory selects a backend by name and hands the recipe a
:class:`SessionManager`, so a recipe is written once against that interface. These
pin the parts of it a signature cannot express, and they run against every
registered backend rather than a named one, so adding a backend either satisfies
them or fails here.

What used to be checked by reading each manager's source -- "does it mention
``session_lost``?" -- is now structural: ``kind`` is required, so a backend cannot
omit the outcome, and ``SessionManager`` is abstract, so it cannot omit a method.
"""

import inspect

from typing import get_type_hints

import pytest

from nexus.core import UserInputRequest
from nexus.extensions.tasks.hitl.actor.base import (
    SessionManager,
    TurnKind,
    TurnResult,
)
from nexus.extensions.tasks.hitl.actor.factory import backends, make_worker_manager


BACKENDS = backends()

# What the v2 recipe calls on a manager, beyond what the ABC already requires.
RECIPE_USES = ('setup', 'start', 'send', 'report', 'traces', 'close', 'shutdown')


def test_there_is_at_least_one_backend_to_check() -> None:
    """Otherwise every parametrized test below silently passes on an empty set."""
    assert BACKENDS


@pytest.mark.parametrize('backend', BACKENDS)
def test_every_backend_is_constructible_from_the_same_arguments(backend: str) -> None:
    """One caller selects any backend by name, so construction cannot differ."""
    manager = make_worker_manager(backend, default_working_dir='/testbed')

    assert manager.backend == backend
    assert isinstance(manager, SessionManager)


@pytest.mark.parametrize('backend', BACKENDS)
def test_every_backend_offers_what_a_recipe_calls(backend: str) -> None:
    manager = make_worker_manager(backend)
    missing = [name for name in RECIPE_USES if not callable(getattr(manager, name, None))]

    assert not missing, f'{backend} cannot be driven by a recipe: missing {missing}'


@pytest.mark.parametrize('backend', BACKENDS)
def test_a_backend_that_can_block_a_turn_can_also_report_losing_the_session(backend: str) -> None:
    """Both outcomes are reachable, because both are values of one field.

    A backend holding a session across turns is the same thing that can lose one --
    torn down by the runtime, or unreachable because the sandbox stopped answering.
    Reporting the first and not the second leaves a recipe unable to tell "answer
    this" from "this is gone", so it keeps sending: run 0812g spent 2h47m that way
    and recorded an empty patch.

    Read off the source because reaching a lost session needs a sandbox. What is
    checked is that the manager selects among the kinds rather than hard-coding one:
    a backend returning only ``completed`` would satisfy the type and never stop.
    """
    manager_cls = type(make_worker_manager(backend))
    module = inspect.getmodule(manager_cls)
    assert module is not None
    source = inspect.getsource(module)

    can_block = 'user_input_request=' in source
    if not can_block:
        pytest.skip(f'{backend} cannot block a turn, so it holds no session to lose')

    assert "'terminal'" in source, (
        f'{backend} can block a turn on a session but never reports losing one; '
        'a recipe driving it cannot stop when the session is gone'
    )
    assert "'needs_input'" in source, f'{backend} blocks turns but never says so'


# -- the turn contract a recipe branches on --------------------------------


def test_the_outcome_is_required_rather_than_defaulted() -> None:
    """A default outcome is what makes an incomplete backend harmless to construct.

    ``success`` used to default, so a backend that never set it reported every turn
    as failed-but-recoverable and a loop stopping on a lost session never stopped.
    """
    fields = TurnResult.__dataclass_fields__

    with pytest.raises(TypeError):
        TurnResult()  # type: ignore[call-arg]
    assert get_type_hints(TurnResult)['kind'] is TurnKind
    assert fields['user_input_request'].default is None
    assert fields['output'].default == ''


def test_the_four_outcomes_are_distinguishable() -> None:
    """A driver reads these three properties; two kinds sharing all three collide.

    ``refused`` and ``terminal`` are the pair that matters: retrying is right for one
    and pointless for the other.
    """
    reads = {
        kind: (result.awaiting_answer, result.session_lost, result.delivered)
        for kind in ('completed', 'needs_input', 'terminal', 'refused')
        if (result := TurnResult(kind=kind))
    }

    assert len(set(reads.values())) == 4, f'two kinds a driver cannot tell apart: {reads}'


@pytest.mark.parametrize('backend', BACKENDS)
def test_every_backend_reports_a_turn_with_output(backend: str) -> None:
    """The user simulator is shown this; an empty report shows it nothing."""
    manager = make_worker_manager(backend)

    rendered = manager.report(TurnResult(kind='completed', output='I changed three files.'))

    assert 'I changed three files.' in rendered
    assert rendered.strip()


@pytest.mark.parametrize('backend', BACKENDS)
def test_a_blocked_turn_reports_the_question_rather_than_a_claim(backend: str) -> None:
    """The distinction a recipe cannot recover once the report is text."""
    manager = make_worker_manager(backend)
    blocked = TurnResult(
        kind='needs_input',
        output='I got as far as the parser.',
        user_input_request=UserInputRequest(source='hook', hint='Which auth method?'),
    )

    rendered = manager.report(blocked)

    assert blocked.awaiting_answer
    assert 'Which auth method?' in rendered
    assert '<worker_question>' in rendered
    assert '<final_response>' not in rendered, 'a blocked turn has claimed nothing yet'


@pytest.mark.parametrize('backend', BACKENDS)
def test_a_refused_turn_still_reports_something(backend: str) -> None:
    """A message that never left must not read as a worker that said nothing."""
    manager = make_worker_manager(backend)

    rendered = manager.report(TurnResult(kind='refused', raw={'reason': 'slash command'}))

    assert rendered.strip()


# -- concurrency is available on every backend -----------------------------


@pytest.mark.parametrize('backend', BACKENDS)
def test_every_backend_can_drive_several_sessions_at_once(backend: str) -> None:
    """``run_turns`` is concrete on the ABC, so this is not a capability to check for.

    A recipe that fans out does not have to ask whether the manager it was given
    supports it.
    """
    manager = make_worker_manager(backend)

    assert callable(manager.run_turns)
    assert type(manager).run_turns is SessionManager.run_turns, f'{backend} overrides the shared driver'


def test_the_factory_rejects_a_name_it_cannot_build() -> None:
    """A typo in HITL_WORKER_BACKEND must fail at construction, not at first send."""
    with pytest.raises(ValueError, match='Unknown worker backend'):
        make_worker_manager('pty_clade_code')
