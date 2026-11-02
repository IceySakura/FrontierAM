"""The experience rating, and the one thing it must never do: end a run.

A refusing endpoint, a reply in prose, and a reply short one dimension each have to
come back as an unrated result rather than an exception.
"""

import asyncio

from types import SimpleNamespace
from typing import Any

import pytest

from nexus.extensions.tasks.hitl.user_agent.rating import (
    ATTEMPTS,
    DIMENSIONS,
    GROUPS,
    SCALE,
    ExperienceRating,
    dimension_lines,
    note_key,
    parse,
    rate_experience,
)


def a_rating(**overrides: int) -> dict[str, int]:
    """Every dimension at the good end, with whatever the caller wants moved."""
    return {name: overrides.get(name, SCALE) for name in DIMENSIONS}


def as_json(scores: dict[str, int], note: str = 'round 2 dropped the export helper') -> str:
    parts = ', '.join(f'"{name}": {value}' for name, value in scores.items())
    notes = ', '.join(f'"{note_key(group)}": "{note}"' for group in GROUPS)
    return f'<experience>\n{{{parts}, {notes}}}\n</experience>'


class Replies:
    """Answers with each reply in turn, raising whatever is in place of one."""

    def __init__(self, *replies: str | Exception) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict[str, Any]]] = []

    async def create_completion(self, messages: list[dict]) -> Any:
        self.seen.append(list(messages))
        reply = self.replies[min(len(self.seen) - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def test_a_reply_in_the_asked_for_shape_is_read_back():
    scores = a_rating(flattery=1, boundary=3)
    client = Replies(as_json(scores, note='it kept agreeing with me in round 4'))

    result = asyncio.run(rate_experience([{'role': 'user', 'content': 'do it'}], client=client))

    assert result.rated is True
    assert result.scores == scores
    assert result.notes == dict.fromkeys(GROUPS, 'it kept agreeing with me in round 4')
    assert result.attempts == 1


def test_the_history_is_continued_rather_than_replaced():
    history = [{'role': 'system', 'content': 'persona'}, {'role': 'user', 'content': 'do it'}]
    client = Replies(as_json(a_rating()))

    asyncio.run(rate_experience(history, client=client))

    sent = client.seen[0]
    assert sent[:2] == history
    assert len(sent) == 3
    assert 'plain_words' in str(sent[-1]['content'])


def test_prose_is_retried_and_the_second_reply_stands():
    client = Replies('it was fine overall, I would work with it again', as_json(a_rating(stuck_looping=2)))

    result = asyncio.run(rate_experience([], client=client))

    assert result.rated is True
    assert result.attempts == 2
    assert result.scores['stuck_looping'] == 2


def test_the_retry_carries_the_reply_that_failed():
    client = Replies('no json here', as_json(a_rating()))

    asyncio.run(rate_experience([], client=client))

    second = client.seen[1]
    assert second[-2] == {'role': 'assistant', 'content': 'no json here'}
    assert 'did not parse' in str(second[-1]['content'])


def test_a_rating_missing_a_dimension_does_not_count_as_a_rating():
    partial = a_rating()
    partial.pop('stance_drift')
    client = Replies(as_json(partial))

    result = asyncio.run(rate_experience([], client=client))

    assert result.rated is False
    assert result.scores == {}
    assert 'stance_drift' in result.error


def test_a_rating_missing_a_group_note_does_not_count_as_a_rating():
    """The notes are the point of the rating, so one missing is not a rating."""
    parts = ', '.join(f'"{name}": {value}' for name, value in a_rating().items())
    client = Replies(f'<experience>\n{{{parts}, "{note_key(GROUPS[0])}": "fine"}}\n</experience>')

    result = asyncio.run(rate_experience([], client=client))

    assert result.rated is False
    assert note_key(GROUPS[1]) in result.error


def test_attempts_run_out_and_the_last_reply_is_kept():
    client = Replies('still prose')

    result = asyncio.run(rate_experience([], client=client))

    assert result.rated is False
    assert result.attempts == ATTEMPTS
    assert len(client.seen) == ATTEMPTS
    assert result.raw == 'still prose'


def test_an_endpoint_that_raises_every_time_returns_rather_than_propagates():
    client = Replies(RuntimeError('rate limited'))

    result = asyncio.run(rate_experience([], client=client))

    assert result.rated is False
    assert result.error == 'RuntimeError: rate limited'


def test_one_failed_request_does_not_end_the_attempts():
    client = Replies(RuntimeError('rate limited'), as_json(a_rating()))

    result = asyncio.run(rate_experience([], client=client))

    assert result.rated is True
    assert result.attempts == 2


@pytest.mark.parametrize(
    ('payload', 'wanted'),
    [
        ('{"plain_words": 0}', f'outside 1 to {SCALE}'),
        ('{"plain_words": 6}', f'outside 1 to {SCALE}'),
        ('{"plain_words": "high"}', 'not a number'),
        ('{"plain_words": true}', 'not a number'),
    ],
)
def test_a_score_that_is_not_a_number_in_range_is_refused(payload: str, wanted: str):
    with pytest.raises(ValueError, match=wanted):
        parse(payload)


def test_every_dimension_is_offered_with_all_five_scores_named():
    """A score with no anchor is one the model picks blind."""
    text = dimension_lines()

    for name, dimension in DIMENSIONS.items():
        assert f'- {name}, ' in text
        for score, anchor in enumerate(dimension.anchors, 1):
            assert f'{score} = {anchor}' in text


def test_every_dimension_belongs_to_a_declared_group():
    assert {dimension.group for dimension in DIMENSIONS.values()} == set(GROUPS)


def test_an_unrated_result_is_what_a_run_that_was_never_asked_carries():
    assert ExperienceRating().rated is False
    assert ExperienceRating().scores == {}
    assert ExperienceRating().notes == {}
