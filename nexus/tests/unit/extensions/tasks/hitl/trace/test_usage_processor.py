"""Token accounting read off the completion spans, grouped by trace label.

These cover what the manager-interface accounting could not:

* the user agent's own calls and its compaction summaries are counted, because
  they go through the same client under the user's label -- v2 recorded neither,
  and a run's report showed only the worker;
* the counters outlive the worker manager. Run 0812g read ``usage_summary`` after
  ``shutdown()`` had dropped the handler and recorded ``{"worker": {}}`` for a run
  that made 119 requests;
* the label is the grouping key, so one client serving the user and several
  workers still separates them. Grouping by ``_client_key``, as the upstream
  ``token_counter`` does, merges all of them into one bucket.
"""

import asyncio

import pytest

from nexus.extensions.tasks.hitl.trace.labels import LabelReadingClient, labelled, user_label, worker_label
from nexus.extensions.tasks.hitl.trace.usage import HITLUsageProcessor, UsageTracker, combine
from nexus.extensions.tasks.hitl.trace.usage.parse import parse_usage
from nexus.extensions.tasks.hitl.trace.usage.tracker import SUMMED_FIELDS
from nexus.trace import observe, processor_context


def usage_block(*, prompt: int, completion: int, cache_read: int = 0, cache_creation: int = 0, reasoning: int = 0):
    """A Vertex/Anthropic usage block, in the shape that client emits."""
    return {
        'prompt_tokens': prompt,
        'completion_tokens': completion,
        'total_tokens': prompt + completion,
        'prompt_tokens_details': {'cached_tokens': cache_read, 'cache_creation_tokens': cache_creation},
        'cache_read_input_tokens': cache_read,
        'cache_creation_input_tokens': cache_creation,
        'completion_tokens_details': {'reasoning_tokens': reasoning},
    }


class Client(LabelReadingClient):
    """Stands in for OpenAICompatibleClient.

    ``self_attrs`` repeats the real client's list, which is what puts the label and
    the usage on the span. Subclassing the real client would drag in config
    plumbing unrelated to what is measured here.
    """

    _client_key = 'main'

    @observe('openai_completion', self_attrs=['_client_key', '_chat_phase'])
    async def create_completion(self, messages, usage=None, model='claude-opus-5', delay=0.0):
        if delay:
            await asyncio.sleep(delay)
        return {
            'choices': [{'message': {'role': 'assistant', 'content': 'ok'}}],
            'model': model,
            'usage': usage if usage is not None else usage_block(prompt=100, completion=20),
        }


def test_the_run_totals_and_each_label_carry_the_same_fields_as_the_legacy_report() -> None:
    """A reader of the legacy per-role summary can read this one.

    The legacy recipes report ``{'human': summary(), 'worker': summary()}``. Adding
    a field here and not there, or dropping one, means two reports of the same run
    that cannot be compared or summed.

    The run totals carry the summed counters only. Per-model detail and the per-call
    series belong to a label -- concatenating two labels' call rows would read as one
    conversation, and their peaks are not a run's peak but two separate ones.
    """
    processor = HITLUsageProcessor(default_profile='bedrock_claude')
    client = Client()

    async def drive():
        with processor_context([processor], finalize_on_exit=False):
            with labelled(client, user_label()):
                await client.create_completion([], usage=usage_block(prompt=100, completion=20))

    asyncio.run(drive())
    result = processor.get_result()

    legacy_fields = set(UsageTracker().summary())
    assert set(result['by_label'][user_label()]) == legacy_fields
    assert set(result['totals']) == set(SUMMED_FIELDS), 'totals leave per-label detail to each label'
    assert set(result['totals']).isdisjoint({'by_model', 'series', 'peak_input_tokens'})


def test_one_client_serving_the_user_and_two_workers_keeps_them_apart() -> None:
    """The reason the label is the key rather than ``_client_key``.

    Every session here runs through one client instance, which is what the label
    design established. Grouped by client key all nine calls would land in a single
    ``main`` bucket, and a run could not say what the user cost versus a worker.
    """
    processor = HITLUsageProcessor(default_profile='bedrock_claude')
    client = Client()

    async def drive():
        with processor_context([processor], finalize_on_exit=False):
            with labelled(client, user_label()):
                for _ in range(4):
                    await client.create_completion([], usage=usage_block(prompt=2000, completion=150))

            async def worker(name: str, calls: int, delay: float):
                with labelled(client, worker_label(name)):
                    for _ in range(calls):
                        await client.create_completion([], usage=usage_block(prompt=5000, completion=600), delay=delay)

            # Concurrent, and interleaved: the spans end out of order.
            await asyncio.gather(worker('a', 2, 0.02), worker('b', 3, 0.01))

    asyncio.run(drive())
    result = processor.get_result()
    by_label = result['by_label']

    assert sorted(by_label) == [user_label(), worker_label('a'), worker_label('b')]
    assert by_label[user_label()]['calls'] == 4
    assert by_label[user_label()]['input_tokens'] == 8000
    assert by_label[worker_label('a')]['calls'] == 2
    assert by_label[worker_label('b')]['calls'] == 3
    assert result['totals']['calls'] == 9
    assert result['totals']['input_tokens'] == 8000 + 10000 + 15000


def test_cache_creation_survives_because_the_profile_reads_it() -> None:
    """The field the upstream ``token_counter`` drops.

    Vertex reports cache creation separately from cache reads and the two are
    priced differently, so a report without it cannot produce a cost. This is what
    the profile is for; a fixed set of keys would lose it.
    """
    processor = HITLUsageProcessor(default_profile='bedrock_claude')
    client = Client()

    async def drive():
        with processor_context([processor], finalize_on_exit=False):
            with labelled(client, worker_label('a')):
                await client.create_completion(
                    [],
                    usage=usage_block(prompt=10000, completion=500, cache_read=8000, cache_creation=1500, reasoning=90),
                )

    asyncio.run(drive())
    summary = processor.get_result()['by_label'][worker_label('a')]

    assert summary['cache_read_tokens'] == 8000
    assert summary['cache_creation_tokens'] == 1500
    assert summary['reasoning_tokens'] == 90
    # The hit rate a reader computes from these two.
    assert summary['cache_read_tokens'] / summary['input_tokens'] == 0.8


def test_a_call_made_outside_any_label_is_reported_rather_than_dropped() -> None:
    """An unlabelled call is a wiring fault worth seeing in the report.

    Dropping it would make the totals quietly disagree with the provider's bill,
    which is the one number this exists to reconcile against.
    """
    processor = HITLUsageProcessor()
    client = Client()

    async def drive():
        with processor_context([processor], finalize_on_exit=False):
            await client.create_completion([], usage=usage_block(prompt=700, completion=70))

    asyncio.run(drive())
    result = processor.get_result()

    assert result['by_label'], 'the call must land somewhere'
    label, summary = next(iter(result['by_label'].items()))
    assert label.startswith('_'), f'an unlabelled call is bucketed under a sentinel, got {label!r}'
    assert summary['input_tokens'] == 700
    assert result['totals']['input_tokens'] == 700


def test_a_span_that_is_not_a_completion_contributes_nothing() -> None:
    """Every span in the run reaches on_end, including ones with no usage at all."""
    processor = HITLUsageProcessor()

    class Other:
        @observe('some_other_span')
        async def work(self):
            return {'no': 'usage'}

    async def drive():
        with processor_context([processor], finalize_on_exit=False):
            await Other().work()

    asyncio.run(drive())

    assert processor.get_result()['by_label'] == {}
    assert processor.get_result()['totals']['calls'] == 0


def test_combining_summaries_is_available_to_a_reader_of_the_record() -> None:
    """The totals are reproducible from the parts, so a reader can re-slice them."""
    parts = [
        {'calls': 2, 'input_tokens': 100, 'output_tokens': 10, 'by_model': {'m': {}}},
        {'calls': 3, 'input_tokens': 200, 'output_tokens': 20},
    ]

    totals = combine(parts)

    assert totals['calls'] == 5
    assert totals['input_tokens'] == 300
    assert totals['output_tokens'] == 30
    assert 'by_model' not in totals
    assert totals['reasoning_tokens'] == 0, 'a field no part reported is zero, not missing'


def test_each_client_is_read_with_its_own_profile() -> None:
    """One run, two providers, which is the ordinary case rather than an edge one.

    Measured on run 0812g: the user simulator was on Gemini through a plain
    OpenAI-compatible endpoint while the worker was on Vertex Claude. Vertex reports
    ``cache_creation_input_tokens`` at the top of the usage block and the default
    profile does not look there, so a single run-wide profile reads one of the two
    with the wrong one -- and a lost cache-creation counter cannot be recovered from
    the record afterwards.
    """
    processor = HITLUsageProcessor(profiles={'worker': 'bedrock_claude'}, default_profile='default')

    class Human(LabelReadingClient):
        _client_key = 'human'

        @observe('openai_completion', self_attrs=['_client_key', '_chat_phase'])
        async def create_completion(self):
            # A plain OpenAI-compatible response: cache reads only, under details.
            return {
                'model': 'gemini-3.1-pro',
                'usage': {
                    'prompt_tokens': 5000,
                    'completion_tokens': 300,
                    'prompt_tokens_details': {'cached_tokens': 1000},
                },
            }

    class Worker(LabelReadingClient):
        _client_key = 'worker'

        @observe('openai_completion', self_attrs=['_client_key', '_chat_phase'])
        async def create_completion(self):
            # Vertex/Anthropic: both cache counters at the top of the block.
            return {
                'model': 'claude-opus-5',
                'usage': {
                    'prompt_tokens': 20000,
                    'completion_tokens': 800,
                    'cache_read_input_tokens': 16000,
                    'cache_creation_input_tokens': 900,
                },
            }

    async def drive():
        with processor_context([processor], finalize_on_exit=False):
            human, worker = Human(), Worker()
            with labelled(human, user_label()):
                await human.create_completion()
            with labelled(worker, worker_label('w')):
                await worker.create_completion()

    asyncio.run(drive())
    result = processor.get_result()

    user = result['by_label'][user_label()]
    assert user['input_tokens'] == 5000
    assert user['cache_read_tokens'] == 1000

    worker_usage = result['by_label'][worker_label('w')]
    assert worker_usage['input_tokens'] == 20000
    assert worker_usage['cache_read_tokens'] == 16000
    assert worker_usage['cache_creation_tokens'] == 900, 'the default profile would read this as zero'

    assert result['profiles'] == {user_label(): 'default', worker_label('w'): 'bedrock_claude'}


def test_a_client_the_run_did_not_name_falls_back_to_the_default_profile() -> None:
    """A run adds a client -- a judge, a rewriter -- without touching the mapping."""
    processor = HITLUsageProcessor(profiles={'worker': 'bedrock_claude'})

    assert processor.profile_for('worker') == 'bedrock_claude'
    assert processor.profile_for('human') == 'default'
    assert processor.profile_for('rubric_eval') == 'default'
    assert processor.profile_for(None) == 'default', 'a span with no client key is still counted'


def test_a_counter_a_provider_left_at_zero_does_not_hide_the_one_it_filled() -> None:
    """Both cache keys are always present, so the first one is not the answer.

    ``ExtendedUsage`` declares ``cache_read_input_tokens: int = 0``, so a provider
    reporting only ``prompt_tokens_details.cached_tokens`` still ships a zero at the
    top of the block. Run 0817a: 1338 of 1352 glm calls carried a cached_tokens of
    up to 13312 and every one of them exported as zero.
    """
    trace = parse_usage(
        'default',
        {
            'usage': {
                'prompt_tokens': 15433,
                'completion_tokens': 190,
                'prompt_tokens_details': {'cached_tokens': 13312, 'cache_creation_tokens': None},
                'cache_read_input_tokens': 0,
                'cache_creation_input_tokens': 0,
            }
        },
    )

    assert trace.cache_read_tokens == 13312
    assert trace.billed_input_tokens == 2121, 'the point of reading cache_read at all'


def test_a_provider_that_cached_nothing_still_reports_a_reading() -> None:
    """Zero and absent stay apart: 0 is what happened, absent is unusable usage.

    Run 0817a's 651 sonnet calls sent no cache_control and really did cache nothing.
    """
    cached_nothing = parse_usage(
        'default',
        {'usage': {'prompt_tokens': 5514, 'completion_tokens': 301, 'cache_read_input_tokens': 0}},
    )
    assert cached_nothing.valid
    assert cached_nothing.cache_read_tokens == 0

    assert not parse_usage('default', {'usage': {'prompt_tokens': 10}}).valid, 'no output counter at all'
    zero_output = parse_usage('default', {'usage': {'prompt_tokens': 10, 'completion_tokens': 0}})
    assert zero_output.valid, 'a call that emitted nothing is still a call'


@pytest.mark.parametrize('profile', ['default', 'bedrock_claude', 'naci_glm'])
def test_every_profile_the_generator_can_pass_is_accepted(profile: str) -> None:
    """The generator resolves the worker's usage profile and keys it by client."""
    processor = HITLUsageProcessor(profiles={'worker': profile})
    client = Client()

    async def drive():
        with processor_context([processor], finalize_on_exit=False):
            with labelled(client, worker_label('a')):
                await client.create_completion([], usage=usage_block(prompt=500, completion=50))

    asyncio.run(drive())
    result = processor.get_result()

    assert processor.profile_for('worker') == profile
    assert result['by_label'][worker_label('a')]['input_tokens'] == 500


# -- the per-turn series --------------------------------------------------


def test_each_turn_is_recorded_as_a_series_the_totals_cannot_show() -> None:
    """One row per turn, because one turn is one completion.

    The user agent's loop makes exactly one ``create_completion`` per turn
    (``agent_with_user.py:332``), and run 0816c bears it out: turns equalled
    calls on the user's label for all three instances (57/57, 9/9, 30/30). So a
    completion *is* a turn and no extra signal is needed to group them.

    What the totals cannot say: a turn's ``input_tokens`` is the length of the
    history it was sent, so the series is the growth curve. On 0816c one worker
    averaged 183k input over 264 calls, which says nothing about whether it
    plateaued or was still climbing.
    """
    processor = HITLUsageProcessor()
    client = Client()

    async def drive():
        with processor_context([processor], finalize_on_exit=False):
            with labelled(client, user_label()):
                for prompt in (1000, 4000, 9000):
                    await client.create_completion(
                        [], usage=usage_block(prompt=prompt, completion=50, cache_read=prompt // 2)
                    )

    asyncio.run(drive())
    label = processor.get_result()['by_label'][user_label()]

    assert label['calls'] == 3
    assert label['series']['input'] == '1000,4000,9000'
    assert label['series']['cache_read'] == '500,2000,4500'
    assert label['peak_input_tokens'] == 9000, 'the longest the history ever got'


def test_the_series_renders_as_a_few_lines_however_many_turns_there_were() -> None:
    """The shape is chosen for reading, not only for bytes.

    One JSON array per turn renders as one line per number: 264 turns of the widest
    label on run 0816c came to 2116 lines of a pretty-printed record. One string per
    counter is 10 lines for the same data, and smaller.
    """
    import json

    processor = HITLUsageProcessor()
    client = Client()

    async def drive():
        with processor_context([processor], finalize_on_exit=False):
            with labelled(client, worker_label('a')):
                for index in range(264):
                    await client.create_completion([], usage=usage_block(prompt=180000 + index, completion=500))

    asyncio.run(drive())
    label = processor.get_result()['by_label'][worker_label('a')]

    rendered = json.dumps(label['series'], indent=2).count('\n') + 1
    assert rendered <= 12, f'the series renders as {rendered} lines'
    assert label['series']['input'].count(',') == 263, 'and still holds every turn'


def test_a_turn_with_no_usable_usage_leaves_no_point_in_the_series() -> None:
    """A zero point would read as a turn that sent an empty history."""
    processor = HITLUsageProcessor()
    client = Client()

    async def drive():
        with processor_context([processor], finalize_on_exit=False):
            with labelled(client, user_label()):
                await client.create_completion([], usage={'prompt_tokens': 10})
                await client.create_completion([], usage=usage_block(prompt=70, completion=5))

    asyncio.run(drive())
    label = processor.get_result()['by_label'][user_label()]

    assert label['invalid_calls'] == 1
    assert label['calls'] == 1
    assert label['series']['input'] == '70'


def test_the_series_is_capped_so_a_long_run_cannot_grow_the_record_without_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two ends are kept, the middle is dropped, and the gap is counted.

    Patches a reachable limit in: the shipped one is beyond reach.
    """
    from nexus.extensions.tasks.hitl.trace.usage import tracker as tracker_module

    limit = 40
    monkeypatch.setattr(tracker_module, 'SERIES_LIMIT', limit)

    processor = HITLUsageProcessor()
    client = Client()
    turns = limit + 40

    async def drive():
        with processor_context([processor], finalize_on_exit=False):
            with labelled(client, user_label()):
                for index in range(turns):
                    await client.create_completion([], usage=usage_block(prompt=100 + index, completion=5))

    asyncio.run(drive())
    label = processor.get_result()['by_label'][user_label()]
    points = label['series']['input'].split(',')

    assert label['calls'] == turns
    assert len(points) == limit
    assert label['series_dropped'] == 40
    assert points[0] == '100', 'the run still says where it started'
    assert points[-1] == str(100 + turns - 1), 'and where it ended'
    assert label['peak_input_tokens'] == 100 + turns - 1, 'the peak survives truncation'


def test_a_run_longer_than_any_observed_keeps_every_point() -> None:
    """No real run reaches the cap, so no real series has a gap.

    The longest single label across the recorded batches made 411 calls.
    """
    from nexus.extensions.tasks.hitl.trace.usage.tracker import SERIES_LIMIT

    turns = 500
    assert turns < SERIES_LIMIT

    processor = HITLUsageProcessor()
    client = Client()

    async def drive():
        with processor_context([processor], finalize_on_exit=False):
            with labelled(client, user_label()):
                for index in range(turns):
                    await client.create_completion([], usage=usage_block(prompt=100 + index, completion=5))

    asyncio.run(drive())
    label = processor.get_result()['by_label'][user_label()]

    assert label['series_dropped'] == 0
    assert len(label['series']['input'].split(',')) == turns


def test_each_turn_records_how_long_the_history_it_was_sent_was() -> None:
    """The final length of a trajectory, without reading the trajectory.

    The chat block already holds every message, so the count is derivable -- but on
    run 0816c that block was 7.13 MB against usage's 1.2 KB, and the widest worker's
    532 messages were only findable by parsing all of it. The completion span carries
    ``inputs.messages``, so the count is free here.

    A maximum rather than the last value: a history does not only grow. Compaction
    shortens it, and a worker's sub-calls carry their own shorter lists -- on 0816c
    one worker's six trajectories ran 300/370/440/464/510/532 messages.
    """
    processor = HITLUsageProcessor()
    client = Client()

    async def drive():
        with processor_context([processor], finalize_on_exit=False):
            with labelled(client, user_label()):
                for count in (2, 8, 4):
                    await client.create_completion(
                        [{'role': 'user', 'content': 'x'}] * count,
                        usage=usage_block(prompt=100 * count, completion=10),
                    )

    asyncio.run(drive())
    label = processor.get_result()['by_label'][user_label()]

    assert label['series']['messages'] == '2,8,4'
    assert label['peak_messages'] == 8, 'the longest the history got, not the last length'
