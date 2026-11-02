"""Offline tests for how a chain's entries are grouped into released requirements."""

import pytest

from nexus.extensions.tasks.hitl.recipe.swe_chain_v2.requirement_groups import (
    GROUP_GUIDE,
    ITEM_TAG,
    group_indices,
    group_requirements,
    merge_statements,
)


def test_size_one_releases_the_chain_unchanged():
    """The default, and what every run before grouping existed did."""
    statements = ['first', 'second', 'third']

    assert group_requirements(statements, size=1) == statements


def test_a_group_takes_up_to_size_entries_in_dataset_order():
    indices = group_indices(['a', 'b', 'c', 'd', 'e'], size=3, max_chars=1000)

    assert indices == [[0, 1, 2], [3, 4]]


def test_a_group_stops_early_when_the_next_entry_would_pass_max_chars():
    indices = group_indices(['a' * 60, 'b' * 60, 'c' * 60], size=3, max_chars=100)

    assert indices == [[0], [1], [2]]


def test_an_entry_longer_than_max_chars_is_released_on_its_own():
    """Truncating it would hand over a requirement missing what it asked for."""
    indices = group_indices(['a' * 500, 'b'], size=3, max_chars=100)

    assert indices == [[0], [1]]


def test_every_entry_lands_in_exactly_one_group():
    statements = [f'req{position}' * position for position in range(1, 12)]

    indices = group_indices(statements, size=3, max_chars=40)

    assert sorted(index for group in indices for index in group) == list(range(len(statements)))


def test_a_grouped_requirement_tags_each_entry_and_says_how_to_use_them():
    text = merge_statements(['change the reader', 'undo that change'], [0, 1])

    assert GROUP_GUIDE in text
    assert text.count(f'<{ITEM_TAG}>\nchange the reader\n</{ITEM_TAG}>') == 1
    assert text.count(f'<{ITEM_TAG}>\nundo that change\n</{ITEM_TAG}>') == 1


def test_a_lone_entry_is_released_as_written():
    """A group of one carries no ordering to explain."""
    assert merge_statements(['just this'], [0]) == 'just this'


def test_a_size_below_one_is_refused():
    with pytest.raises(ValueError, match='at least 1'):
        group_indices(['a'], size=0)
