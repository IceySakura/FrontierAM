"""Expanded prompt partitioning preserves message ownership and validation errors."""

import numpy as np
import pytest

from nexus.gateway.mm_expansion import partition_expanded_ids


@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize(
    ('raw_messages', 'expanded', 'expected'),
    [
        ([], [], []),
        ([[]], [], [[]]),
        ([[], []], [], [[], []]),
        ([[1, 2], [2, 2]], [1, 2, 2, 2], [[1, 2], [2, 2]]),
        ([[99]], [99, 99, 99], [[99, 99, 99]]),
        ([[99], [1], [99]], [99, 99, 1, 99, 99, 99], [[99, 99], [1], [99, 99, 99]]),
        ([[], [1, 99], [], []], [1, 99, 99], [[], [1, 99, 99], [], []]),
        (
            [[1, 99], [], [2, 99, 3], []],
            [1, 99, 99, 2, 99, 99, 99, 3],
            [[1, 99, 99], [], [2, 99, 99, 99, 3], []],
        ),
    ],
)
def test_partition_preserves_message_boundaries_and_array_views(
    raw_messages: list[list[int]], expanded: list[int], expected: list[list[int]], strided: bool
) -> None:
    raw_ids = [np.array(message, dtype=np.int32) for message in raw_messages]
    prompt_ids = np.array(expanded, dtype=np.int32)
    if strided:
        prompt_ids = np.repeat(prompt_ids, 2)[::2]

    parts = partition_expanded_ids(raw_ids, prompt_ids, 99)

    assert len(parts) == len(expected), 'empty messages must retain their own partitions'
    for part, values in zip(parts, expected, strict=True):
        np.testing.assert_array_equal(part, values)
        assert part.dtype == prompt_ids.dtype, 'partitioning must retain the processor token dtype'
        if part.size:
            assert np.shares_memory(part, prompt_ids), 'partitions must remain views of the processor buffer'
    np.testing.assert_array_equal(prompt_ids, expanded)
    for raw, message in zip(raw_ids, raw_messages, strict=True):
        np.testing.assert_array_equal(raw, message)


@pytest.mark.parametrize(
    ('raw_messages', 'expanded', 'error'),
    [
        ([[1, 2]], [1, 3, 4], 'non-image token'),
        ([[1]], [], 'non-image token'),
        ([[1, 99, 2]], [1, 99, 99, 3], 'non-image token'),
        ([[1, 99, 2, 99]], [1, 99], 'non-image token'),
        ([[99, 99]], [99, 99, 99], 'no image-token run'),
        ([[99], [], [99]], [99, 99], 'no image-token run'),
        ([[], [99]], [], 'no image-token run'),
        ([[1, 99]], [1, 2], 'no image-token run'),
        ([[1, 99, 99, 2]], [1, 99, 99, 2], 'no image-token run'),
        ([[99, 1]], [99, 99, 1, 99, 99], '2 unpartitioned token'),
        ([[]], [99, 99, 99], '3 unpartitioned token'),
        ([], [1, 99, 99], '3 unpartitioned token'),
    ],
)
def test_partition_rejects_the_first_mismatch_or_unclaimed_tokens(
    raw_messages: list[list[int]], expanded: list[int], error: str
) -> None:
    raw_ids = [np.array(message, dtype=np.int32) for message in raw_messages]
    prompt_ids = np.array(expanded, dtype=np.int32)

    with pytest.raises(ValueError, match=error):
        partition_expanded_ids(raw_ids, prompt_ids, 99)
