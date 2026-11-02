"""The two checks a parallel run adds: what one label holds, and what ran at once.

The concurrency itself is covered by ``actor/test_run_turns.py`` and the label
mechanism by ``trace/test_span_labels.py``; what is pinned here is only the reading of
what those left behind, and the ways that reading can be wrong.
"""

from typing import Any

import pytest

from nexus.extensions.tasks.hitl.recipe.parallel.task import (
    ParallelCheck,
    _check_label,
    _check_overlap,
    _check_reacted,
    _content_text,
    _trajectory_text,
    peak_inflight,
)


def trajectory(label: str, *texts: str, calls: tuple[str, ...] = ()) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{'role': 'user', 'content': text} for text in texts]
    for arguments in calls:
        messages.append(
            {
                'role': 'assistant',
                'tool_calls': [{'function': {'name': 'send_message', 'arguments': arguments}}],
            }
        )
    return {'role': label, 'messages': messages}


def exported(*trajectories: dict[str, Any]) -> dict[str, Any]:
    return {'trajectories': list(trajectories)}


# -- the label check --------------------------------------------------------


def test_a_label_holding_its_own_text_and_not_the_others_passes() -> None:
    result = _check_label(
        ParallelCheck(name='alpha kept to itself', label='worker/alpha', holds='ALPHA', lacks='BETA'),
        exported(trajectory('worker/alpha', 'write ALPHA'), trajectory('worker/beta', 'write BETA')),
    )

    assert result.passed
    assert result.kind == 'label'


def test_text_leaking_from_the_other_session_fails() -> None:
    """The whole point of the run: two sessions must not be able to quote each other."""
    result = _check_label(
        ParallelCheck(name='alpha kept to itself', label='worker/alpha', holds='ALPHA', lacks='BETA'),
        exported(trajectory('worker/alpha', 'write ALPHA', 'and also BETA')),
    )

    assert not result.passed
    assert 'BETA' in result.detail
    assert 'never saw it' in result.detail


def test_text_that_should_be_there_and_is_not_fails() -> None:
    result = _check_label(
        ParallelCheck(name='alpha did its work', label='worker/alpha', holds='ALPHA'),
        exported(trajectory('worker/alpha', 'nothing of the sort')),
    )

    assert not result.passed
    assert 'is not in what worker/alpha recorded' in result.detail


def test_both_going_wrong_at_once_says_so() -> None:
    result = _check_label(
        ParallelCheck(name='alpha', label='worker/alpha', holds='ALPHA', lacks='BETA'),
        exported(trajectory('worker/alpha', 'BETA only')),
    )

    assert not result.passed
    assert 'absent' in result.detail
    assert 'leaked' in result.detail


def test_a_label_that_exported_nothing_names_the_ones_that_did() -> None:
    """A misspelt session name would otherwise read as a session that stayed silent."""
    result = _check_label(
        ParallelCheck(name='gamma', label='worker/gamma', holds='X'),
        exported(trajectory('worker/alpha', 'a'), trajectory('worker/beta', 'b')),
    )

    assert not result.passed
    assert 'worker/alpha' in result.detail
    assert 'worker/beta' in result.detail


def test_a_check_naming_only_what_must_be_absent_still_works() -> None:
    passed = _check_label(
        ParallelCheck(name='no leak', label='worker/alpha', lacks='BETA'),
        exported(trajectory('worker/alpha', 'ALPHA')),
    )
    failed = _check_label(
        ParallelCheck(name='no leak', label='worker/alpha', lacks='BETA'),
        exported(trajectory('worker/alpha', 'BETA')),
    )

    assert passed.passed
    assert not failed.passed


def test_what_a_session_was_told_is_found_in_a_tool_call() -> None:
    """On the user's side the message travels as a call's arguments, not as content."""
    text = _trajectory_text(trajectory('user/orchestrator', calls=('{"to": "alpha", "message": "write ALPHA"}',)))

    assert 'ALPHA' in text


@pytest.mark.parametrize(
    ('content', 'expected'),
    [
        ('plain', 'plain'),
        ([{'type': 'text', 'text': 'first'}, {'type': 'text', 'text': 'second'}], 'first\nsecond'),
        (None, ''),
        ([], ''),
        ([{'type': 'image'}], ''),
    ],
)
def test_a_message_reads_whichever_shape_it_arrived_in(content: object, expected: str) -> None:
    """The worker's side of every recorded run arrives as blocks, the user's as strings."""
    assert _content_text(content) == expected


# -- the overlap check ------------------------------------------------------


def test_two_sessions_recorded_running_at_once_passes() -> None:
    overlaps = [{'session': 'alpha', 'inflight': ['beta']}, {'session': 'beta', 'inflight': []}]

    result = _check_overlap(ParallelCheck(name='ran together', overlap=2), overlaps)

    assert result.passed
    assert '2 session(s) ran at once' in result.detail


def test_sessions_that_only_ever_ran_alone_fails() -> None:
    """What a run looks like when the turns were serialised somewhere."""
    overlaps = [{'session': 'alpha', 'inflight': []}, {'session': 'beta', 'inflight': []}]

    result = _check_overlap(ParallelCheck(name='ran together', overlap=2), overlaps)

    assert not result.passed
    assert '1 session(s) ran at once, wanted 2' in result.detail


def test_no_recorded_turns_fails_rather_than_passing_vacuously() -> None:
    """A run that fell over before sending anything must not read as a passing check."""
    result = _check_overlap(ParallelCheck(name='ran together', overlap=2), [])

    assert not result.passed
    assert 'no turns' in result.detail


@pytest.mark.parametrize(
    ('overlaps', 'expected'),
    [
        ([], 0),
        ([{'inflight': []}], 1),
        ([{'inflight': ['b']}, {'inflight': []}], 2),
        ([{'inflight': ['b', 'c']}, {'inflight': ['c']}, {'inflight': []}], 3),
        ([{'inflight': None}], 1),
    ],
)
def test_the_peak_counts_the_finishing_session_too(overlaps: list[dict], expected: int) -> None:
    """``inflight`` names the others; the one finishing was running as well."""
    assert peak_inflight(overlaps) == expected


# -- the reacted check ------------------------------------------------------


def test_a_follow_up_sent_while_others_ran_passes() -> None:
    """The affordance being checked: answer the first to finish without waiting."""
    followed = [{'step': 1, 'session': 'alpha', 'while_running': ['beta', 'gamma'], 'kind': 'completed'}]

    result = _check_reacted(ParallelCheck(name='answered early', reacted=True), followed)

    assert result.passed
    assert 'alpha while beta/gamma ran' in result.detail


def test_a_follow_up_sent_only_after_the_batch_drained_fails() -> None:
    """A correct send, but not the affordance: nothing was still running beside it."""
    followed = [{'step': 1, 'session': 'alpha', 'while_running': [], 'kind': 'completed'}]

    result = _check_reacted(ParallelCheck(name='answered early', reacted=True), followed)

    assert not result.passed
    assert 'after the batch had drained' in result.detail


def test_one_reactive_follow_up_among_several_is_enough() -> None:
    """The last session to finish has nobody left beside it, so some follow-up always
    drains; the check asks whether any of them did not."""
    followed = [
        {'session': 'alpha', 'while_running': ['beta'], 'kind': 'completed'},
        {'session': 'beta', 'while_running': [], 'kind': 'completed'},
    ]

    assert _check_reacted(ParallelCheck(name='n', reacted=True), followed).passed


def test_no_follow_up_at_all_fails_rather_than_passing_vacuously() -> None:
    result = _check_reacted(ParallelCheck(name='n', reacted=True), [])

    assert not result.passed
    assert 'no follow-up was sent' in result.detail


# -- which check a set of fields means --------------------------------------


@pytest.mark.parametrize(
    ('fields', 'expected'),
    [
        ({'label': 'worker/a', 'holds': 'X'}, 'label'),
        ({'overlap': 2}, 'overlap'),
        ({'reacted': True}, 'reacted'),
        ({'tool': 'mcp__search'}, 'tool'),
        ({'says': 'DONE'}, 'says'),
        ({'command': 'test -f /x'}, 'command'),
    ],
)
def test_a_check_reports_the_kind_its_fields_name(fields: dict, expected: str) -> None:
    assert ParallelCheck(name='n', **fields).kind() == expected


def test_a_check_naming_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match='sets none of'):
        ParallelCheck(name='empty').kind()


def test_holds_alone_is_not_a_check() -> None:
    """``holds`` belongs to a label; on its own there is nothing to look in."""
    with pytest.raises(ValueError, match='sets none of'):
        ParallelCheck(name='dangling', holds='ALPHA').kind()
