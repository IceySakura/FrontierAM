"""What a trace holds once, and what it holds a marker for.

The upstream tracker records the full message list on every generation, so N calls
upload N + (N-1) + ... + 1 messages. Measured on run 0816a: 297524 messages of
which 1102 were distinct, one instance's trace weighing 1222 MB against a real
history of 14 MB -- and Langfuse refuses to read a trace past 80 MB, so the record
becomes unreadable rather than merely large.

These pin that a message is uploaded once and referred to afterwards, that a marker
says where to look, and that nothing the deduplicator does not recognise is
altered.
"""

import json

from typing import Any

import pytest

from nexus.extensions.tasks.hitl.trace.langfuse.upload_dedup import (
    DIGEST_SIZE,
    DeduplicatingSpan,
    MessageLedger,
    deduplicate,
    fingerprint,
)


def message(role: str, content: str) -> dict[str, Any]:
    return {'role': role, 'content': content}


class FakeSpan:
    """Records what would reach the SDK, and nests the way a real span does."""

    def __init__(self, name: str = 'root') -> None:
        self.name = name
        self.children: list[FakeSpan] = []
        self.input: Any = None
        self.updates: list[dict[str, Any]] = []

    def start_observation(self, **kwargs: Any) -> 'FakeSpan':
        child = FakeSpan(str(kwargs.get('name') or '?'))
        child.input = kwargs.get('input')
        self.children.append(child)
        return child

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


# -- the ledger ----------------------------------------------------------------


def test_a_repeated_message_is_uploaded_once() -> None:
    ledger = MessageLedger()
    first = [message('system', 'S'), message('user', 'U')]
    second = [*first, message('assistant', 'A')]

    ledger.rewrite(first, where='w Step 1')
    out = ledger.rewrite(second, where='w Step 2')

    assert out == ['<same as w Step 1>', '<same as w Step 1>', message('assistant', 'A')]
    assert ledger.summary() == {'messages_uploaded': 3, 'messages_referenced': 2, 'messages_total': 5}


def test_a_marker_names_the_label_as_well_as_the_step() -> None:
    """Every label numbers its steps from 1, so a step alone locates nothing."""
    ledger = MessageLedger()
    shared = message('system', 'S')

    ledger.rewrite([shared], where='worker/main/worker Step 1')
    out = ledger.rewrite([shared], where='user/orchestrator Step 1')

    assert out == ['<same as worker/main/worker Step 1>']


def test_identity_is_content_and_not_position() -> None:
    """Concurrent sessions and sub-calls arrive interleaved, so lists are not
    prefixes of one another: on the PTY run 41 of 205 worker spans were shorter
    than the one before and 42 differed inside the shared prefix."""
    ledger = MessageLedger()

    ledger.rewrite([message('system', 'A'), message('user', 'B')], where='w Step 1')
    # A shorter list whose first entry differs -- a sub-call with its own prompt.
    out = ledger.rewrite([message('system', 'DIFFERENT')], where='w Step 2')

    assert out == [message('system', 'DIFFERENT')], 'a different message must be uploaded'
    assert ledger.uploaded == 3


def test_a_reordered_message_is_recognised_as_the_same_one() -> None:
    ledger = MessageLedger()

    ledger.rewrite([{'role': 'user', 'content': 'C'}], where='w Step 1')
    out = ledger.rewrite([{'content': 'C', 'role': 'user'}], where='w Step 2')

    assert out == ['<same as w Step 1>']


def test_the_digest_is_wide_enough_for_a_run() -> None:
    """A collision would report one message's content under another's marker."""
    assert DIGEST_SIZE >= 8
    assert len(fingerprint(message('user', 'x'))) == DIGEST_SIZE * 2


# -- the whole input payload ---------------------------------------------------


def test_a_tool_set_is_uploaded_once_and_pointed_at_afterwards() -> None:
    ledger = MessageLedger()
    tools = [{'name': 'Bash'}, {'name': 'Read'}]

    first = deduplicate({'messages': [message('system', 'S')], 'tools': tools}, ledger, where='w Step 1')
    second = deduplicate({'messages': [message('system', 'S')], 'tools': tools}, ledger, where='w Step 2')

    assert first['tools'] == tools
    assert second['tools'] == '<same 2 tools as w Step 1>'


def test_a_different_tool_set_is_uploaded_in_full() -> None:
    """A sub-call carries its own tools, which the upstream comment asks to keep."""
    ledger = MessageLedger()

    deduplicate({'messages': [], 'tools': [{'name': 'Bash'}, {'name': 'Read'}]}, ledger, where='w Step 1')
    out = deduplicate({'messages': [], 'tools': [{'name': 'Bash'}]}, ledger, where='w Step 2')

    assert out['tools'] == [{'name': 'Bash'}]


@pytest.mark.parametrize(
    'payload',
    [
        {'prompt': 'a subagent prompt'},
        {'note': 'SWE-bench agent run'},
        {'content': 'a system prompt span'},
        'a bare string',
        None,
    ],
)
def test_an_input_it_does_not_recognise_is_untouched(payload: Any) -> None:
    """Most observations are not generations; their inputs must reach Langfuse whole."""
    assert deduplicate(payload, MessageLedger(), where='w Step 1') == payload


def test_the_first_upload_is_never_truncated() -> None:
    """Truncation is the other lever and this is not it: content stays whole."""
    ledger = MessageLedger()
    big = message('tool', 'X' * 40_000)

    out = deduplicate({'messages': [big]}, ledger, where='w Step 1')

    assert out['messages'] == [big]


# -- the wrapper ---------------------------------------------------------------


def test_the_wrapper_rewrites_what_reaches_the_sdk() -> None:
    root = FakeSpan()
    span = DeduplicatingSpan(root, MessageLedger(), scope='worker/main/worker')

    span.start_observation(name='Step 1', input={'messages': [message('system', 'S')]})
    span.start_observation(name='Step 2', input={'messages': [message('system', 'S'), message('user', 'U')]})

    assert root.children[0].input['messages'] == [message('system', 'S')]
    assert root.children[1].input['messages'] == ['<same as worker/main/worker Step 1>', message('user', 'U')]


def test_wrapping_is_inherited_so_a_subagent_is_covered() -> None:
    """A subagent's generations hang off `prev_generation`, not off the label span."""
    root = FakeSpan()
    ledger = MessageLedger()
    span = DeduplicatingSpan(root, ledger, scope='w')

    generation = span.start_observation(name='Step 1', input={'messages': [message('system', 'S')]})
    tool = generation.start_observation(name='Tool: Agent', input={'prompt': 'go'})
    subagent = tool.start_observation(name='Subagent: x', input={'prompt': 'go'})
    subagent.start_observation(name='Step 1', input={'messages': [message('system', 'S')]})

    assert isinstance(subagent, DeduplicatingSpan)
    nested = root.children[0].children[0].children[0].children[0]
    assert nested.input['messages'] == ['<same as w Step 1>']


def test_the_wrapper_passes_everything_else_through() -> None:
    root = FakeSpan('the-real-name')
    span = DeduplicatingSpan(root, MessageLedger())

    assert span.name == 'the-real-name'


def test_an_observation_with_no_input_is_started_anyway() -> None:
    root = FakeSpan()
    span = DeduplicatingSpan(root, MessageLedger())

    span.start_observation(name='Tools Definition')

    assert [child.name for child in root.children] == ['Tools Definition']


def test_update_is_rewritten_too() -> None:
    """`_create_system_user_prompts` reaches the trace through `update`."""
    root = FakeSpan()
    ledger = MessageLedger()
    span = DeduplicatingSpan(root, ledger, scope='w')

    span.start_observation(name='Step 1', input={'messages': [message('system', 'S')]})
    span.update(name='Step 2', input={'messages': [message('system', 'S')]})

    assert root.updates[0]['input']['messages'] == ['<same as w Step 1>']


# -- the scale it exists for ---------------------------------------------------


def test_a_long_conversation_stops_growing_quadratically() -> None:
    """40 calls over a growing history: the upload should track the history, not
    the sum of every prefix."""
    ledger = MessageLedger()
    history = [message('system', 'S' * 4000)]
    before = after = 0

    for step in range(1, 41):
        history.append(message('user', f'u{step}' * 500))
        payload = {'messages': list(history)}
        before += len(json.dumps(payload))
        after += len(json.dumps(deduplicate(payload, ledger, where=f'w Step {step}')))

    assert after < before / 8
    assert ledger.uploaded == len(history)
