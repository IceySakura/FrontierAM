"""What may be sent to a worker harness, and why the check is a whitelist.

Measured on run 0821a: the user sent one unregistered command, Claude Code answered it
locally without producing an assistant turn, and the turn waited out its full 900-second
timeout before the PTY session was torn down -- ending the run with the requirement
unstarted. A denylist cannot cover that, because the name is not knowable in advance.

So these cover three groups: the commands a run has, the near-misses that cost the same
timeout as an unknown command, and the registered ones that are refused anyway.
"""

import pytest

from nexus.extensions.tasks.hitl.actor.input_spec import (
    check_outgoing,
    command_name,
    first_token,
)
from nexus.extensions.tasks.hitl.actor.print_claude_code.spec import (
    DENIED_COMMANDS as DENYLIST,
)
from nexus.extensions.tasks.hitl.actor.print_claude_code.spec import (
    UNSUPPORTED_COMMANDS as UNSUPPORTED,
)


HAS = frozenset({'verify', 'simplify', 'code-review', 'probe-echo'})


def allowed(text: str, *, prefix: str = '/') -> bool:
    return bool(check_outgoing(text, allowed=HAS, prefix=prefix, denied=DENYLIST, unsupported=UNSUPPORTED))


def reason(text: str, *, prefix: str = '/') -> str:
    return check_outgoing(text, allowed=HAS, prefix=prefix, denied=DENYLIST, unsupported=UNSUPPORTED).reason


# -- ordinary messages are the common case --------------------------------


@pytest.mark.parametrize(
    'text',
    [
        'fix the auth module',
        'explain what /compact does',
        'the file at /tmp/x.py is the one I mean',
        'why did you change src/main.py?',
    ],
)
def test_prose_is_sent_even_when_it_names_a_command(text: str) -> None:
    """Only the first token is a command, so a slash mid-sentence is prose."""
    assert allowed(text)


# -- the commands a run has ------------------------------------------------


def test_a_command_the_run_has_is_sent() -> None:
    assert allowed('/verify')


def test_a_command_carries_an_instruction_after_it() -> None:
    """The whole point of the form: the routine and what to apply it to, at once."""
    assert allowed('/verify the auth module, especially the token refresh')


@pytest.mark.parametrize('text', ['/VERIFY', '/Verify', '/verify.', '/verify,', '  \n\n/verify'])
def test_a_command_is_recognised_through_case_punctuation_and_leading_space(text: str) -> None:
    """A user writes the name, not a token: refusing these would cost a real turn."""
    assert allowed(text), text


# -- an unregistered command is the failure this exists for ---------------


def test_an_unknown_command_is_refused_before_it_reaches_the_sandbox() -> None:
    assert not allowed('/definitely-not-a-command')


def test_the_refusal_lists_what_the_worker_does_have() -> None:
    """Being told no without being told what works costs another turn to find out."""
    said = reason('/definitely-not-a-command')

    assert '/verify' in said
    assert '/code-review' in said


def test_a_run_with_no_commands_says_so_rather_than_listing_nothing() -> None:
    verdict = check_outgoing('/anything', allowed=frozenset(), prefix='/')

    assert not verdict.allowed
    assert 'no commands' in verdict.reason


# -- near-misses cost the same timeout, so they are refused too -----------


def test_a_backtick_wrapped_command_is_refused_with_the_reason() -> None:
    """What a user writes after reading the name in backticks. Claude Code reads it as
    prose and does nothing, so the intent is lost silently.
    """
    verdict = check_outgoing('`/verify`', allowed=HAS, prefix='/')

    assert not verdict.allowed
    assert 'backticks' in verdict.reason


def test_a_backtick_wrapped_command_with_more_after_it_is_also_refused() -> None:
    assert not allowed('`/verify` the auth module')


def test_a_path_at_the_start_is_refused_and_told_to_move( ) -> None:
    """``/tmp/x.py`` is prose to a person and a command to the harness."""
    verdict = check_outgoing('/tmp/foo.py is broken', allowed=HAS, prefix='/')

    assert not verdict.allowed
    assert 'Put the path later' in verdict.reason


def test_the_prefix_alone_names_nothing() -> None:
    assert not allowed('/')


# -- registered but refused anyway ----------------------------------------


@pytest.mark.parametrize('name', ['clear', 'resume', 'fork', 'exit', 'background'])
def test_a_command_that_would_end_the_session_is_refused(name: str) -> None:
    """The manager holds one long-lived session; these discard or detach it."""
    verdict = check_outgoing(f'/{name}', allowed=HAS | {name}, prefix='/', denied=DENYLIST, unsupported=UNSUPPORTED)

    assert not verdict.allowed, name
    assert 'cannot be sent' in verdict.reason


@pytest.mark.parametrize('name', ['compact', 'context', 'cost', 'model', 'goal'])
def test_a_command_that_produces_no_reply_is_refused(name: str) -> None:
    """Turn completion is signalled by the Stop hook, which these never fire."""
    verdict = check_outgoing(f'/{name}', allowed=HAS | {name}, prefix='/', denied=DENYLIST, unsupported=UNSUPPORTED)

    assert not verdict.allowed, name


def test_goal_is_refused_although_it_does_produce_work() -> None:
    """Refused on purpose rather than by omission.

    It holds state across turns, so it has to be cleared once reached -- and that clear
    produces no reply -- and a second one opens a confirmation screen waiting on a
    keypress the host cannot send.
    """
    assert 'goal' in UNSUPPORTED
    assert not allowed('/goal all the tests pass')


def test_listing_a_dangerous_command_does_not_make_it_sendable() -> None:
    """The two checks are separate: a run cannot whitelist its way past the denylist."""
    assert not check_outgoing('/clear', allowed={'clear'}, prefix='/', denied=DENYLIST, unsupported=UNSUPPORTED).allowed


# -- a harness with no command form ---------------------------------------


def test_a_harness_with_no_prefix_refuses_a_slash_message() -> None:
    """It would reach the worker as text it will not act on, spending the turn."""
    verdict = check_outgoing('/verify', allowed=HAS, prefix='')

    assert not verdict.allowed
    assert 'no commands' in verdict.reason


def test_a_harness_with_no_prefix_still_takes_ordinary_messages() -> None:
    assert check_outgoing('fix the auth module', allowed=HAS, prefix='').allowed


# -- the pieces -----------------------------------------------------------


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('/verify now', 'verify'),
        ('/verify', 'verify'),
        ('verify', ''),
        ('/', ''),
        ('', ''),
        ('/tmp/x.py broke', 'tmp/x.py'),
    ],
)
def test_command_name_reads_the_first_token_only(text: str, expected: str) -> None:
    assert command_name(text, prefix='/') == expected


def test_first_token_strips_the_tail_marks_a_sentence_leaves() -> None:
    assert first_token('/verify.') == '/verify'
    assert first_token('/verify!') == '/verify'


def test_the_two_refusal_sets_do_not_overlap() -> None:
    """A name in both would make the reason given depend on check order."""
    assert not DENYLIST & UNSUPPORTED
