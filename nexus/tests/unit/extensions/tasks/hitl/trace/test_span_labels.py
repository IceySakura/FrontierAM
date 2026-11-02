"""Tests for per-session trace labels carried on spans.

The label mechanism replaces "install this session's processors, run, uninstall"
with "tag the span, let one tracker sort it out". These tests pin the two facts
that decide whether that works, and the one that decides when it breaks.

Why this file exists at all: the label rides on ``_chat_phase``, which
``@observe`` reads off the *client instance* (``core/openai_client.py:136``),
not from a contextvar. That looks unsafe under concurrency and mostly isn't --
but the boundary is narrow enough that it needs to be written down as a test
rather than a comment.
"""

import asyncio

import pytest

from nexus.trace import processor_context
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor
from nexus.trace.tracer import observe


SYS_WORKER = {'role': 'system', 'content': 'You are Claude Code.'}
TASK = {'role': 'user', 'content': 'fix the auth bug'}
SYS_USER = {'role': 'system', 'content': 'You are a user simulator.'}

# Long enough that the slow call is still open when the fast one finishes, which
# is the whole point: a label that only works when calls happen to serialise
# proves nothing.
SLOW = 0.10
FAST = 0.01
DISPATCH_SETTLE = 0.6


class LabelledClient:
    """Stand-in for OpenAICompatibleClient.

    Only ``_chat_phase`` matters here: it is in the real client's
    ``@observe(self_attrs=[...])`` list, so setting it puts the value on the
    span. Subclassing the real client would drag in config plumbing that has
    nothing to do with what is under test.
    """

    _chat_phase = None

    @observe('openai_completion', self_attrs=['_chat_phase'])
    async def create_completion(self, messages, reply, delay=0.0):
        if delay:
            await asyncio.sleep(delay)
        return {'choices': [{'message': {'role': 'assistant', 'content': reply}}]}


async def labelled_call(client, label, messages, reply, delay=0.0):
    """Label immediately before the call -- see test_await_between_... for why."""
    client._chat_phase = label
    return await client.create_completion(messages, reply, delay)


def labels_by_reply(tracker_or_processor) -> dict[str, str | None]:
    """Map each trajectory's final assistant text to its label.

    Keyed by reply rather than position because trajectory order follows tree
    traversal, not call order.
    """
    result = tracker_or_processor.get_result() if hasattr(tracker_or_processor, 'get_result') else tracker_or_processor
    out: dict[str, str | None] = {}
    for trajectory in result.get('trajectories') or []:
        if isinstance(trajectory, dict):
            messages = trajectory['messages']
            out[str(messages[-1].get('content'))] = trajectory.get('role')
        else:
            # global_first exports bare message lists, which have nowhere to put
            # a label at all (see test_global_first_mode_cannot_carry_labels).
            out[str(trajectory[-1].get('content'))] = None
    return out


# --- one tracker, several conversations ------------------------------------


@pytest.mark.asyncio
async def test_one_tracker_separates_workers_and_host():
    """The point of the whole design: no per-session processor registration."""
    tracker = ChatTrackerProcessor(tools_mode='branch_last')

    with processor_context([tracker], finalize_on_exit=False):
        worker_a, worker_b, host = LabelledClient(), LabelledClient(), LabelledClient()
        await asyncio.gather(
            labelled_call(worker_a, 'worker:a', [SYS_WORKER, TASK], 'A1', SLOW),
            labelled_call(worker_b, 'worker:b', [SYS_WORKER, TASK], 'B1', FAST),
            labelled_call(host, 'user', [SYS_USER, {'role': 'user', 'content': 'plan'}], 'U1', FAST),
        )
        await asyncio.sleep(DISPATCH_SETTLE)

    assert labels_by_reply(tracker) == {'A1': 'worker:a', 'B1': 'worker:b', 'U1': 'user'}


@pytest.mark.asyncio
async def test_identical_opening_still_gets_distinct_labels():
    """Parallel members sampling one task open byte-identically.

    This is the case that makes attribution unrecoverable from tree shape
    alone, so it is the case the label has to carry.
    """
    tracker = ChatTrackerProcessor(tools_mode='branch_last')

    with processor_context([tracker], finalize_on_exit=False):
        a, b = LabelledClient(), LabelledClient()
        await asyncio.gather(
            labelled_call(a, 'worker:a', [SYS_WORKER, TASK], 'A1', SLOW),
            labelled_call(b, 'worker:b', [SYS_WORKER, TASK], 'B1', FAST),
        )
        await asyncio.sleep(DISPATCH_SETTLE)

    assert labels_by_reply(tracker) == {'A1': 'worker:a', 'B1': 'worker:b'}


@pytest.mark.asyncio
async def test_identical_reply_does_not_merge_labels():
    """Even a byte-identical assistant reply keeps two labelled leaves.

    Output nodes are always appended as new children, never merged, so each
    call leaves its own leaf. If that ever changed, two members answering
    identically would collapse into one trajectory under one label.
    """
    tracker = ChatTrackerProcessor(tools_mode='branch_last')

    with processor_context([tracker], finalize_on_exit=False):
        a, b = LabelledClient(), LabelledClient()
        await asyncio.gather(
            labelled_call(a, 'worker:a', [SYS_WORKER, TASK], 'SAME', SLOW),
            labelled_call(b, 'worker:b', [SYS_WORKER, TASK], 'SAME', FAST),
        )
        await asyncio.sleep(DISPATCH_SETTLE)

    trajectories = tracker.get_result()['trajectories']
    assert len(trajectories) == 2
    assert sorted(t.get('role') for t in trajectories) == ['worker:a', 'worker:b']


@pytest.mark.asyncio
async def test_second_turn_keeps_its_own_label():
    """A member continuing its conversation stays under its own label."""
    tracker = ChatTrackerProcessor(tools_mode='branch_last')

    with processor_context([tracker], finalize_on_exit=False):
        a, b = LabelledClient(), LabelledClient()
        await asyncio.gather(
            labelled_call(a, 'worker:a', [SYS_WORKER, TASK], 'A1', SLOW),
            labelled_call(b, 'worker:b', [SYS_WORKER, TASK], 'B1', FAST),
        )
        await asyncio.gather(
            labelled_call(
                a,
                'worker:a',
                [SYS_WORKER, TASK, {'role': 'assistant', 'content': 'A1'}, {'role': 'user', 'content': 'next'}],
                'A2',
                FAST,
            ),
            labelled_call(
                b,
                'worker:b',
                [SYS_WORKER, TASK, {'role': 'assistant', 'content': 'B1'}, {'role': 'user', 'content': 'next'}],
                'B2',
                FAST,
            ),
        )
        await asyncio.sleep(DISPATCH_SETTLE)

    labels = labels_by_reply(tracker)
    assert labels['A2'] == 'worker:a'
    assert labels['B2'] == 'worker:b'


# --- the processor the recipe configures ------------------------------------


@pytest.mark.asyncio
async def test_the_run_level_processor_carries_labels():
    """The configuration the worker actually runs under.

    The recipe names ``chat_tracker`` with ``tools_mode='branch_last'``, so this
    is the processor a label has to survive.
    """
    tracker = ChatTrackerProcessor(tools_mode='branch_last')

    with processor_context([tracker], finalize_on_exit=False):
        a, b = LabelledClient(), LabelledClient()
        await asyncio.gather(
            labelled_call(a, 'worker:a', [SYS_WORKER, TASK], 'A1', SLOW),
            labelled_call(b, 'worker:b', [SYS_WORKER, TASK], 'B1', FAST),
        )
        await asyncio.sleep(DISPATCH_SETTLE)

    assert labels_by_reply(tracker) == {'A1': 'worker:a', 'B1': 'worker:b'}


# --- the two ways it breaks ------------------------------------------------


@pytest.mark.asyncio
async def test_global_first_mode_cannot_carry_labels():
    """A guard, not an aspiration: ``global_first`` exports bare lists.

    It is ``ChatTrackerProcessor``'s default, so a recipe that omits
    ``tools_mode`` gets it. Asserting the limitation here means the switch shows
    up as a test change rather than as silently unlabelled data.
    """
    tracker = ChatTrackerProcessor(tools_mode='global_first')

    with processor_context([tracker], finalize_on_exit=False):
        a, b = LabelledClient(), LabelledClient()
        await asyncio.gather(
            labelled_call(a, 'worker:a', [SYS_WORKER, TASK], 'A1', SLOW),
            labelled_call(b, 'worker:b', [SYS_WORKER, TASK], 'B1', FAST),
        )
        await asyncio.sleep(DISPATCH_SETTLE)

    trajectories = tracker.get_result()['trajectories']
    assert all(isinstance(t, list) for t in trajectories)


@pytest.mark.asyncio
async def test_await_between_label_and_call_misattributes_a_shared_client():
    """The invariant, stated as its violation.

    ``@observe`` snapshots ``self_attrs`` synchronously before awaiting the
    wrapped call, which is why labelling a shared client works at all. Yield
    control between the write and the call and the other task's label wins.

    Both fixes are already covered above: a per-session client instance (no
    shared state to race) or writing the label immediately before the call.
    This test exists so that adding an await -- an async rate limiter, an async
    auth refresh -- fails loudly here instead of silently mislabelling
    production trajectories.
    """
    tracker = ChatTrackerProcessor(tools_mode='branch_last')

    async def call_with_await_in_between(client, label, messages, reply, delay):
        client._chat_phase = label
        await asyncio.sleep(FAST)  # the violation
        return await client.create_completion(messages, reply, delay)

    with processor_context([tracker], finalize_on_exit=False):
        shared = LabelledClient()  # one instance for both members
        await asyncio.gather(
            call_with_await_in_between(shared, 'worker:a', [SYS_WORKER, TASK], 'A1', SLOW),
            call_with_await_in_between(shared, 'worker:b', [SYS_WORKER, TASK], 'B1', 0.0),
        )
        await asyncio.sleep(DISPATCH_SETTLE)

    labels = labels_by_reply(tracker)
    assert labels['A1'] == 'worker:b', 'expected the documented misattribution'
    assert labels['B1'] == 'worker:b'


@pytest.mark.asyncio
async def test_await_between_label_and_call_is_safe_with_per_session_clients():
    """Same await, separate instances: correct. Isolates the cause to sharing."""
    tracker = ChatTrackerProcessor(tools_mode='branch_last')

    async def call_with_await_in_between(client, label, messages, reply, delay):
        client._chat_phase = label
        await asyncio.sleep(FAST)
        return await client.create_completion(messages, reply, delay)

    with processor_context([tracker], finalize_on_exit=False):
        a, b = LabelledClient(), LabelledClient()
        await asyncio.gather(
            call_with_await_in_between(a, 'worker:a', [SYS_WORKER, TASK], 'A1', SLOW),
            call_with_await_in_between(b, 'worker:b', [SYS_WORKER, TASK], 'B1', 0.0),
        )
        await asyncio.sleep(DISPATCH_SETTLE)

    assert labels_by_reply(tracker) == {'A1': 'worker:a', 'B1': 'worker:b'}
