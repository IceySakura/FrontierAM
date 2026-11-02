"""The two export-time deduplication rules, and the wrapper over them.

The rules are exercised by calling them directly. Nothing applies them during a
run: they are for post-processing, which reads an exported jsonl.
"""

from nexus.extensions.tasks.hitl.trace.dedup import (
    _deduplicate_by_message_prefix,
    _deduplicate_by_tool_call_ids,
    _deduplicate_trajectories,
    deduplicate_tracker_result,
)


SYSTEM = {'role': 'system', 'content': 'sys'}
USER = {'role': 'user', 'content': 'fix the failing test'}


def _assistant(*tool_ids: str) -> dict:
    return {
        'role': 'assistant',
        'content': '',
        'tool_calls': [
            {'id': tid, 'type': 'function', 'function': {'name': 'Edit', 'arguments': '{}'}} for tid in tool_ids
        ],
    }


def _tool(tool_id: str) -> dict:
    return {'role': 'tool', 'tool_call_id': tool_id, 'content': 'ok'}


def _has_trailing_unanswered_tool_call(trajectory: list[dict]) -> bool:
    """True when the last message is an assistant turn whose tool calls got no result."""
    called: set[str] = set()
    answered: set[str] = set()
    for message in trajectory:
        if message.get('role') == 'assistant':
            called.update(call['id'] for call in message.get('tool_calls') or [])
        elif message.get('role') == 'tool':
            answered.add(message.get('tool_call_id'))
    return bool(called - answered)


def _prefix_chain(depth: int) -> list[list[dict]]:
    """Mimic ChatTrackerProcessor.dump_trajectories output for one linear conversation.

    Each LLM call leaves its assistant output as a leaf; the next call attaches the
    tool result as a *sibling* of that leaf. DFS therefore emits every intermediate
    snapshot shortest-first, each a strict prefix of the next.
    """
    trajectories: list[list[dict]] = []
    current = [SYSTEM, USER]
    for index in range(depth):
        current = [*current, _assistant(f't{index}')]
        trajectories.append(list(current))
        current = [*current, _tool(f't{index}')]
    trajectories.append([*current, {'role': 'assistant', 'content': 'done'}])
    return trajectories


def test_prefix_dedup_keeps_only_the_longest_of_a_chain():
    # Regression: the sort key used to receive the (index, trajectory) pair instead
    # of the trajectory, flattening every key to -0. Order then degraded to DFS
    # order (shortest first), so partial snapshots were kept.
    trajectories = _prefix_chain(4)

    kept = _deduplicate_by_message_prefix(list(trajectories))

    assert len(kept) == 1
    assert kept[0] == trajectories[-1]


def test_prefix_dedup_is_independent_of_input_order():
    trajectories = _prefix_chain(3)
    longest = trajectories[-1]

    for candidate in (trajectories, list(reversed(trajectories))):
        kept = _deduplicate_by_message_prefix(list(candidate))
        assert kept == [longest]


def test_dedup_drops_trailing_unanswered_tool_calls_from_a_chain():
    trajectories = _prefix_chain(4)
    assert sum(_has_trailing_unanswered_tool_call(t) for t in trajectories) == 4

    kept = _deduplicate_trajectories(list(trajectories))

    assert not any(_has_trailing_unanswered_tool_call(t) for t in kept)


def test_dedup_keeps_genuinely_diverging_branches():
    shared = [SYSTEM, USER, _assistant('a'), _tool('a')]
    first = [*shared, _assistant('b'), _tool('b'), {'role': 'assistant', 'content': 'took route B'}]
    second = [*shared, _assistant('c'), _tool('c'), {'role': 'assistant', 'content': 'took route C'}]

    kept = _deduplicate_trajectories([first, second])

    assert kept == [first, second]


def test_dedup_preserves_original_order_of_survivors():
    keep_first = [SYSTEM, USER, _assistant('a'), _tool('a'), {'role': 'assistant', 'content': 'first'}]
    keep_second = [SYSTEM, USER, _assistant('b'), _tool('b'), {'role': 'assistant', 'content': 'second'}]
    partial = [SYSTEM, USER, _assistant('a')]

    kept = _deduplicate_trajectories([keep_first, partial, keep_second])

    assert kept == [keep_first, keep_second]


def test_dedup_is_idempotent():
    trajectories = _prefix_chain(4)

    once = _deduplicate_trajectories(list(trajectories))
    twice = _deduplicate_trajectories(list(once))

    assert once == twice


def test_dedup_passes_through_single_and_empty_inputs():
    assert _deduplicate_trajectories([]) == []
    single = _prefix_chain(2)[0]
    assert _deduplicate_trajectories([single]) == [single]


def test_tool_call_id_dedup_drops_strict_subsets():
    subset = [SYSTEM, USER, _assistant('a'), _tool('a')]
    superset = [SYSTEM, USER, _assistant('a'), _tool('a'), _assistant('b'), _tool('b')]

    kept = _deduplicate_by_tool_call_ids([subset, superset])

    assert kept == [superset]


def test_tool_call_id_dedup_keeps_disjoint_and_equal_sets():
    left = [SYSTEM, USER, _assistant('a'), _tool('a')]
    right = [SYSTEM, USER, _assistant('b'), _tool('b')]

    assert _deduplicate_by_tool_call_ids([left, right]) == [left, right]
    assert _deduplicate_by_tool_call_ids([left, list(left)]) == [left, list(left)]


def test_dedup_handles_branch_last_trajectory_dicts():
    chain = _prefix_chain(3)
    wrapped = [{'messages': messages, 'tools': None} for messages in chain]

    kept = _deduplicate_trajectories(list(wrapped))

    assert kept == [wrapped[-1]]


# --- the wrapper that post-processing calls --------------------------------


def test_the_wrapper_applies_both_rules_and_keeps_other_keys():
    chain = _prefix_chain(3)

    result = deduplicate_tracker_result({'trajectories': chain, 'tools': ['bash']})

    assert len(result['trajectories']) < len(chain)
    assert result['tools'] == ['bash']


def test_the_wrapper_does_not_mutate_its_input():
    chain = _prefix_chain(3)
    original = {'trajectories': chain}

    deduplicate_tracker_result(original)

    assert len(original['trajectories']) == len(chain)
