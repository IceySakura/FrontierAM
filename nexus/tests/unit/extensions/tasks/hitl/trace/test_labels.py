"""Tests for per-session trace labels.

The label mechanism replaces "install this session's processors, run the turn,
uninstall them" with "tag the request, let one tracker sort it out". Three
things decide whether that holds, and each is asserted here rather than left in
a comment:

* a label survives being written on a client shared by concurrent sessions,
* an *absent* label is never left absent, because an untagged call inherits a
  shared ancestor's tag,
* filtering by label is a free function, so post-processing can reuse it.

``test_span_labels.py`` covers the tracker end (does the tag reach the exported
trajectory). This file covers the utilities.
"""

import asyncio

import pytest

from nexus.extensions.tasks.hitl.trace.labels import (
    LABEL_FIELD,
    UNATTRIBUTED,
    LabelError,
    LabelReadingClient,
    assert_no_label_fields,
    current_label,
    enable_labels,
    filter_by_label,
    group_by_label,
    is_wellformed,
    label_of,
    labelled,
    strip_label,
    unattributed_count,
    user_label,
    validate_name,
    worker_label,
)


class FakeClient(LabelReadingClient):
    """A client whose ``_chat_phase`` reads the active label."""


def trajectory(label: str | None, last: str) -> dict:
    """A branch_last-shaped trajectory, which is the only shape carrying labels."""
    entry: dict = {'messages': [{'role': 'assistant', 'content': last}], 'tools': None}
    if label is not None:
        entry['role'] = label
    return entry


# --- names and label construction ------------------------------------------


def test_a_label_names_its_actor_and_its_session():
    assert worker_label('a') == 'worker/a'
    assert worker_label('worker') == 'worker/worker'


def test_user_label_defaults_to_orchestrator():
    assert user_label() == 'user/orchestrator'
    assert user_label('reviewer') == 'user/reviewer'


@pytest.mark.parametrize('bad', ['a/b', '/a', 'a/', 'g/1'])
def test_names_containing_the_separator_are_rejected(bad: str):
    """A slash shifts every later segment, so filtering matches the wrong sessions.

    The result is still a *well-formed* label, so nothing downstream would
    notice -- which is why this has to be refused at the entry point.
    """
    with pytest.raises(LabelError):
        validate_name(bad)
    with pytest.raises(LabelError):
        worker_label(bad)


def test_empty_names_are_rejected():
    with pytest.raises(LabelError):
        validate_name('')
    with pytest.raises(LabelError):
        worker_label('')


@pytest.mark.parametrize(
    ('label', 'expected'),
    [
        ('user/orchestrator', True),
        ('worker/a', True),
        (UNATTRIBUTED, True),
        ('worker', False),  # too few segments
        ('user/a/b', False),  # too many
        ('worker/', False),  # empty segment
        ('other/a', False),  # unknown actor
        ('', False),
    ],
)
def test_is_wellformed(label: str, expected: bool):
    assert is_wellformed(label) is expected


# --- the context manager ---------------------------------------------------


def test_labelled_sets_and_restores():
    client = FakeClient()

    with labelled(client, 'worker/a') as resolved:
        assert resolved == 'worker/a'
        assert client._chat_phase == 'worker/a'

    assert client._chat_phase is None


def test_labelled_substitutes_sentinel_for_an_absent_label():
    """Leaving it unset is not an option: see test_untagged_call_inherits_...

    An untagged call inherits the nearest tagged ancestor, and ancestors are
    shared between sessions, so "no label" silently reads as a sibling's label.
    """
    client = FakeClient()

    with labelled(client, None) as resolved:
        assert resolved == UNATTRIBUTED
        assert client._chat_phase == UNATTRIBUTED

    with labelled(client, '') as resolved:
        assert resolved == UNATTRIBUTED


def test_labelled_restores_on_exception():
    client = FakeClient()

    with labelled(client, 'outer'):
        with pytest.raises(RuntimeError):
            with labelled(client, 'inner'):
                raise RuntimeError('boom')
        assert client._chat_phase == 'outer'

    assert client._chat_phase is None


def test_the_label_attribute_cannot_be_assigned():
    """Assignment would be a silent no-op, so it raises instead.

    The old mechanism *was* assignment, which is exactly what broke under
    overlap, so code still doing it should fail at the point of misuse.
    """
    client = FakeClient()

    with pytest.raises(AttributeError, match='context-scoped'):
        client._chat_phase = 'worker/a'


def test_labelled_nests_on_one_client():
    """A recipe-level wrapper around a handler-level one must unwind cleanly."""
    client = FakeClient()

    with labelled(client, 'outer'):
        with labelled(client, 'inner'):
            assert client._chat_phase == 'inner'
        assert client._chat_phase == 'outer'

    assert client._chat_phase is None


@pytest.mark.asyncio
async def test_concurrent_sessions_on_one_client_each_keep_their_own_label():
    """One shared client, twelve sessions -- the case that removes per-session clients.

    Each session reads its label on both sides of the await standing in for the
    model call. Both reads are needed and so are three or more sessions: with two,
    the shorter one finishes inside the longer one's await, so even a shared
    attribute saved and restored on exit hands the outer session the right label
    back. Overlap needs a third to enter before the second leaves.
    """
    client = FakeClient()
    reads: dict[str, list[str | None]] = {}

    async def one(session: str, delay: float):
        with labelled(client, worker_label(session)) as resolved:
            observed = [client._chat_phase]
            await asyncio.sleep(delay)
            observed.append(client._chat_phase)
            reads[session] = [*observed, resolved]

    await asyncio.gather(*(one(f's{i}', (7 - i % 7) * 0.005) for i in range(12)))

    for session, observed in reads.items():
        assert observed == [worker_label(session)] * 3, f'{session} saw {observed}'


@pytest.mark.asyncio
async def test_a_live_label_does_not_outlive_overlapping_blocks():
    """The defect a save/restore label had, kept as a regression test.

    Storing the label on the client meant a block captured whatever was there as
    its "previous" -- which, under overlap, is another session's *live* label. The
    later block's exit then wrote that label back, and it outlived both: one
    session's label was still set after both blocks had returned, ready for the next
    unlabelled call to inherit.

    The interleaving is specific: B must enter while A is inside, and A must exit
    first. Equal delays do not produce it.
    """
    client = FakeClient()

    async def first():
        with labelled(client, worker_label('a')):
            await asyncio.sleep(0.02)

    async def second():
        await asyncio.sleep(0.01)  # enter while `first` is inside
        with labelled(client, worker_label('b')):
            await asyncio.sleep(0.05)  # `first` exits during this

    await asyncio.gather(first(), second())

    assert client._chat_phase is None, 'a label outlived the block that set it'


@pytest.mark.asyncio
async def test_a_host_block_survives_worker_blocks_nested_inside_it():
    """A user-agent turn that dispatches workers keeps its own label afterwards.

    This is the orchestration shape: the host holds a block, workers run inside
    it, then the host makes its own call. With client-stored labels the workers'
    exits could overwrite the host's.
    """
    client = FakeClient()

    async def worker(session: str, delay: float):
        with labelled(client, worker_label(session)):
            await asyncio.sleep(delay)

    with labelled(client, user_label()):
        await asyncio.gather(worker('a', 0.03), worker('b', 0.0))
        assert client._chat_phase == user_label()

    assert client._chat_phase is None


def test_current_label_reports_the_active_label():
    assert current_label() is None

    with labelled(FakeClient(), worker_label('a')):
        assert current_label() == worker_label('a')

    assert current_label() is None


# --- envelope handling -----------------------------------------------------


def test_strip_label_removes_the_field_and_leaves_the_rest():
    label, remaining = strip_label({LABEL_FIELD: 'worker/a', 'model': 'm', 'messages': []})

    assert label == 'worker/a'
    assert remaining == {'model': 'm', 'messages': []}


def test_strip_label_does_not_mutate_its_input():
    params = {LABEL_FIELD: 'worker/a', 'model': 'm'}

    strip_label(params)

    assert LABEL_FIELD in params


@pytest.mark.parametrize('value', [None, '', 123])
def test_strip_label_treats_unusable_values_as_absent(value):
    label, remaining = strip_label({LABEL_FIELD: value, 'model': 'm'})

    assert label is None
    assert LABEL_FIELD not in remaining, 'the field must be removed even when unusable'


def test_assert_no_label_fields_rejects_a_surviving_private_field():
    """``sanitize_openai_params`` filters by a fixed denylist.

    An unknown key is therefore *not* dropped -- it is forwarded to the provider
    as a completion parameter. This assertion is the only thing standing between
    a forgotten pop and that.
    """
    assert_no_label_fields({'model': 'm'})

    with pytest.raises(AssertionError, match='must not reach the provider'):
        assert_no_label_fields({'model': 'm', LABEL_FIELD: 'worker/a'})


# --- reading back ----------------------------------------------------------


def test_label_of_returns_none_for_bare_message_lists():
    """``global_first`` exports lists, which have nowhere to carry a label."""
    assert label_of([{'role': 'assistant', 'content': 'x'}]) is None
    assert label_of(trajectory(None, 'x')) is None
    assert label_of(trajectory('worker/a', 'x')) == 'worker/a'


def test_filtering_selects_one_session_and_never_a_neighbour():
    """``a``/``a2`` and ``w1``/``w10`` are the pairs where one session label prefixes
    another, so a prefix match would take the extra trajectory and nothing downstream
    would notice. Non-label keys are asserted too, since an export that lost its tool
    schema on the way through would otherwise look like a passing filter.
    """
    result = {
        'trajectories': [
            trajectory(worker_label('a'), 'A'),
            trajectory(worker_label('a2'), 'A2'),
            trajectory(worker_label('w1'), 'W1'),
            trajectory(worker_label('w10'), 'W10'),
            trajectory(user_label(), 'U'),
        ],
        'tools': ['bash'],
    }

    def kept(label):
        return [entry['messages'][-1]['content'] for entry in filter_by_label(result, label)['trajectories']]

    assert kept(worker_label('a')) == ['A'], 'a2 shares the prefix'
    assert kept(worker_label('w1')) == ['W1'], 'w10 shares the prefix'
    assert kept(user_label()) == ['U']
    assert filter_by_label(result, worker_label('a'))['tools'] == ['bash']
    assert len(result['trajectories']) == 5, 'the input is not mutated'


def test_filtering_tolerates_a_missing_trajectories_key():
    assert filter_by_label({}, worker_label('a')) == {}


def test_an_unlabelled_trajectory_is_never_selected():
    """``global_first`` exports bare lists, which carry no label. Matching '' against
    them would hand one session another's history."""
    result = {'trajectories': [trajectory(None, 'orphan'), [{'role': 'assistant', 'content': 'bare'}]]}

    assert filter_by_label(result, worker_label('a'))['trajectories'] == []


def test_group_by_label_buckets_unlabelled_under_the_sentinel():
    result = {
        'trajectories': [
            trajectory('worker/a', 'A1'),
            trajectory('worker/a', 'A2'),
            trajectory(None, 'orphan'),
        ]
    }

    buckets = group_by_label(result)

    assert sorted(buckets) == ['_unattributed', 'worker/a']
    assert len(buckets['worker/a']) == 2


def test_unattributed_count_for_the_export_time_assertion():
    labelled_only = {'trajectories': [trajectory('worker/a', 'A')]}
    with_orphans = {'trajectories': [trajectory('worker/a', 'A'), trajectory(None, 'x')]}

    assert unattributed_count(labelled_only) == 0
    assert unattributed_count(with_orphans) == 1
    # global_first lists carry no label at all, so every one is unattributable
    assert unattributed_count({'trajectories': [[{'role': 'assistant', 'content': 'x'}]]}) == 1


class StubOpenAIClient:
    """Stands in for OpenAICompatibleClient: no ``_chat_phase`` of its own."""

    def __init__(self, name: str = 'main') -> None:
        self.name = name


def test_a_stock_client_carries_no_label():
    # The premise of enable_labels: @observe reads _chat_phase off the instance,
    # and a client that never had one exports every span unlabelled.
    with labelled(None, worker_label('a')):
        assert getattr(StubOpenAIClient(), '_chat_phase', None) is None


def test_enable_labels_makes_the_active_label_readable():
    client = enable_labels(StubOpenAIClient())

    assert client._chat_phase is None
    with labelled(None, worker_label('a')):
        assert client._chat_phase == 'worker/a'
    assert client._chat_phase is None


def test_enable_labels_returns_the_same_object_and_keeps_its_state():
    client = StubOpenAIClient(name='secondary')

    assert enable_labels(client) is client
    assert client.name == 'secondary'
    assert isinstance(client, StubOpenAIClient)


def test_enable_labels_leaves_other_instances_alone():
    enabled = enable_labels(StubOpenAIClient())
    untouched = StubOpenAIClient()

    with labelled(None, worker_label('a')):
        assert enabled._chat_phase == worker_label('a')
        assert getattr(untouched, '_chat_phase', None) is None


def test_enable_labels_is_idempotent():
    client = enable_labels(StubOpenAIClient())
    first = type(client)

    assert type(enable_labels(client)) is first


def test_enable_labels_skips_a_client_that_already_reads_labels():
    client = FakeClient()

    assert enable_labels(client) is client
    assert type(client) is FakeClient
