"""Requirements released as groups rather than one per dataset entry.

A chain's entries are commit-sized: 45% of them are under 600 characters, and some
ask for a single typo. Releasing each on its own puts the user in a round for work a
person would have mentioned in passing. Grouping consecutive entries hands over a
requirement the size a person would actually send.

The split is positional -- ``size`` consecutive entries, capped by ``max_chars`` --
so nothing here reads what a requirement says. Entries keep their dataset order,
which is the order the original commits landed, and a group may hold one that
supersedes or repairs another. ``GROUP_GUIDE`` is what tells the user so.
"""

import logging

from collections.abc import Sequence

from nexus.extensions.tasks.hitl.utils.text import tagged


logger = logging.getLogger(__name__)


DEFAULT_GROUP_SIZE = 1
"""Entries per released requirement. 1 releases the chain as the dataset holds it."""

DEFAULT_GROUP_MAX_CHARS = 3000
"""Characters a group may reach before the next entry starts a new one.

An entry longer than this on its own is still released on its own, never truncated.
"""

GROUP_TAG = 'requirements'
"""Wraps a released requirement holding more than one entry."""

ITEM_TAG = 'item'
"""Wraps one entry inside a group, in the order it is to be asked for."""

GROUP_GUIDE = (
    f'Each <{ITEM_TAG}> is one thing you want, in the order to ask for it. Items that '
    f'stand on their own can share a message. Where an item changes or fixes what an '
    f'earlier item asked for, ask for the earlier one and wait for it to come back '
    f'before raising the later one.'
)
"""How a grouped requirement is to be used. Opens the group."""


def group_indices(
    statements: Sequence[str],
    *,
    size: int = DEFAULT_GROUP_SIZE,
    max_chars: int = DEFAULT_GROUP_MAX_CHARS,
) -> list[list[int]]:
    """Which entries go in each released requirement, as index lists.

    A group takes up to ``size`` entries and stops early when the next one would
    carry it past ``max_chars``. Every entry lands in exactly one group, in dataset
    order.
    """
    if size < 1:
        raise ValueError(f'group size must be at least 1, got {size}')
    if max_chars < 1:
        raise ValueError(f'max_chars must be positive, got {max_chars}')

    groups: list[list[int]] = []
    current: list[int] = []
    length = 0
    for index, statement in enumerate(statements):
        if current and (len(current) >= size or length + len(statement) > max_chars):
            groups.append(current)
            current = []
            length = 0
        current.append(index)
        length += len(statement)
    if current:
        groups.append(current)
    return groups


def merge_statements(statements: Sequence[str], indices: Sequence[int]) -> str:
    """One requirement's text: the entry itself, or the group's entries tagged."""
    chosen = [statements[index] for index in indices]
    if len(chosen) == 1:
        return chosen[0]
    items = '\n'.join(tagged(ITEM_TAG, text) for text in chosen)
    return tagged(GROUP_TAG, f'{GROUP_GUIDE}\n\n{items}')


def group_requirements(
    statements: Sequence[str],
    *,
    size: int = DEFAULT_GROUP_SIZE,
    max_chars: int = DEFAULT_GROUP_MAX_CHARS,
) -> list[str]:
    """The chain as the user receives it: one string per released requirement."""
    if size == 1:
        return list(statements)
    groups = group_indices(statements, size=size, max_chars=max_chars)
    merged = [merge_statements(statements, indices) for indices in groups]
    logger.info(
        '[swe_chain_v2] grouped %d requirements into %d (size<=%d, chars<=%d)',
        len(statements),
        len(merged),
        size,
        max_chars,
    )
    return merged


__all__ = [
    'DEFAULT_GROUP_MAX_CHARS',
    'DEFAULT_GROUP_SIZE',
    'GROUP_GUIDE',
    'GROUP_TAG',
    'ITEM_TAG',
    'group_indices',
    'group_requirements',
    'merge_statements',
]
