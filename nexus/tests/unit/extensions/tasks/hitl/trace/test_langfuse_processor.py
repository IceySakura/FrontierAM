"""One Langfuse trace per run, one span per agent.

These cover what installing the upstream ``langfuse_tracker`` run-level would get
wrong, each measured on the agents a real run produces (``user/orchestrator`` and
``worker/_/worker``, from run 0813c, plus the subagents a worker spawns):

* a pending tool span belongs to the agent that opened it, so an interleaved call
  from another agent no longer discards it and its result no longer lands on the
  wrong tool;
* a subagent is its own record, so the several a worker runs at once no longer share
  one step counter and one pending-tool slot;
* ``model``, the tool catalogue and the token counters are per label rather than
  first- or last-write-wins.

The fake Langfuse client mirrors the one in
``tests/unit/trace/processors/test_langfuse_tracker.py``: the real ``Langfuse``
would need credentials and a server, and what these tests read is the shape of
the observation tree, which the fake records faithfully.
"""

import importlib
import sys
import types

from typing import Any

import pytest

from nexus.trace.events import SpanEndEvent
from nexus.trace.processors.base import span_processor_registry
from nexus.trace.types import Span


USER = 'user/orchestrator'
WORKER = 'worker/_/worker'
OTHER_WORKER = 'worker/_/second'

SYS_USER = {'role': 'system', 'content': 'You are playing a person who wants something built'}
SYS_WORKER = {'role': 'system', 'content': "You are Claude Code, Anthropic's official CLI"}
SYS_SUBAGENT = {'role': 'system', 'content': 'You are an Explore subagent'}


class FakeObservation:
    def __init__(self, **kwargs: Any) -> None:
        self.children: list[FakeObservation] = []
        self.ended = False
        self.__dict__.update(kwargs)

    def start_observation(self, **kwargs: Any) -> 'FakeObservation':
        child = FakeObservation(**kwargs)
        self.children.append(child)
        return child

    def update(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    def end(self) -> None:
        self.ended = True


class FakeLangfuse:
    instances: list['FakeLangfuse'] = []

    def __init__(self, **kwargs: Any) -> None:
        self.root: FakeObservation | None = None
        self.scores: list[dict[str, Any]] = []
        self.flushed = False
        FakeLangfuse.instances.append(self)

    def create_trace_id(self, seed: str | None = None) -> str:
        self.seed = seed
        return 'trace-1'

    def start_observation(self, **kwargs: Any) -> FakeObservation:
        self.root = FakeObservation(**kwargs)
        return self.root

    def create_score(self, **kwargs: Any) -> None:
        self.scores.append(kwargs)

    def flush(self) -> None:
        self.flushed = True


@pytest.fixture
def langfuse(monkeypatch: pytest.MonkeyPatch) -> type[FakeLangfuse]:
    """Import the processor against a fake Langfuse client.

    ``propagate_attributes`` stays real: it attaches an OpenTelemetry context, and
    a stub would hide a leak between that context and the trace it names.
    """
    import langfuse as real_langfuse

    FakeLangfuse.instances.clear()
    for name in list(sys.modules):
        if name == 'nexus.trace.processors.langfuse_tracker' or name.startswith(
            'nexus.extensions.tasks.hitl.trace.langfuse'
        ):
            sys.modules.pop(name, None)
    span_processor_registry._registry.pop('langfuse_tracker', None)
    span_processor_registry._registry.pop('multiagent_langfuse', None)

    stub = types.ModuleType('langfuse')
    stub.Langfuse = FakeLangfuse
    stub.propagate_attributes = real_langfuse.propagate_attributes
    types_stub = types.ModuleType('langfuse.types')
    types_stub.TraceContext = real_langfuse.types.TraceContext

    monkeypatch.setitem(sys.modules, 'langfuse', stub)
    monkeypatch.setitem(sys.modules, 'langfuse.types', types_stub)
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'pk')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'sk')
    return FakeLangfuse


@pytest.fixture
def processor_cls(langfuse: type[FakeLangfuse]) -> Any:
    module = importlib.import_module('nexus.extensions.tasks.hitl.trace.langfuse.processor')
    return module.MultiAgentLangfuseProcessor


@pytest.fixture
def base_cls(langfuse: type[FakeLangfuse]) -> Any:
    module = importlib.import_module('nexus.trace.processors.langfuse_tracker')
    return module.LangfuseTrackerProcessor


def completion(
    label: str | None,
    system: dict[str, Any],
    *,
    span_id: str,
    calls: tuple[tuple[str, str], ...] = (),
    history: tuple[dict[str, Any], ...] = (),
    model: str = 'test-model',
    user: str = 'the task',
) -> Span:
    """A completion span as the tracer emits it, labelled or not.

    ``user`` is the opening user message. An agent with no tool call yet -- a fresh
    conversation, and a subagent's first call -- is told apart by it, so a subagent must
    carry its own, as it does in a real run.
    """
    message: dict[str, Any] = {'role': 'assistant', 'content': 'ok'}
    if calls:
        message['tool_calls'] = [
            {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': '{"description": "probe"}'}}
            for name, call_id in calls
        ]
    attributes: dict[str, Any] = {
        'inputs': {
            'messages': [system, {'role': 'user', 'content': user}, *history],
            'kwargs': {'tools': [{'function': {'name': 'Agent', 'description': 'delegate'}}]},
            'model': model,
        },
        'outputs': {
            'model': model,
            'choices': [{'message': message}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
        },
    }
    if label:
        attributes['_chat_phase'] = label
    return Span(
        trace_id='t',
        span_id=span_id,
        parent_span_id='',
        name='openai_completion',
        start_time_unix_nano=1_000_000_000,
        end_time_unix_nano=2_000_000_000,
        attributes=attributes,
    )


def tool_result(call_id: str, name: str, output: str) -> tuple[dict[str, Any], ...]:
    """The two messages a finished tool call adds to the next request."""
    return (
        {
            'role': 'assistant',
            'content': None,
            'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': '{}'}}],
        },
        {'role': 'tool', 'tool_call_id': call_id, 'content': output},
    )


def task_span(score: float = 0.42) -> Span:
    return Span(
        trace_id='t',
        span_id='task',
        parent_span_id='',
        name='task.hitl',
        start_time_unix_nano=0,
        end_time_unix_nano=3_000_000_000,
        attributes={'type': 'task', 'outputs': {'score': score}},
    )


def send(processor: Any, *spans: Span) -> None:
    for span in spans:
        processor.on_end(span, SpanEndEvent())


def names(node: FakeObservation) -> list[str]:
    """Every observation name under ``node``, indented by depth."""
    collected: list[str] = []

    def walk(current: FakeObservation, depth: int) -> None:
        collected.append('  ' * depth + str(getattr(current, 'name', '?')))
        for child in current.children:
            walk(child, depth + 1)

    walk(node, 0)
    return collected


def find(node: FakeObservation, name: str) -> list[FakeObservation]:
    matches = [node] if str(getattr(node, 'name', '')) == name else []
    for child in node.children:
        matches.extend(find(child, name))
    return matches


def unclosed(node: FakeObservation) -> list[str]:
    found = [] if node.ended else [str(getattr(node, 'name', '?'))]
    for child in node.children:
        found.extend(unclosed(child))
    return found


# -- the shape ------------------------------------------------------------


def test_each_label_becomes_a_span_beside_the_others(processor_cls: Any, langfuse: type[FakeLangfuse]) -> None:
    """One trace holds the run; the agents are its children, not each other's."""
    processor = processor_cls(trace_name='ins', trace_metadata={'instance_id': 'ins'})

    send(
        processor,
        completion(USER, SYS_USER, span_id='s0', calls=(('send_message', 'u1'),)),
        completion(WORKER, SYS_WORKER, span_id='s1', calls=(('Bash', 'w1'),)),
        completion(OTHER_WORKER, SYS_WORKER, span_id='s2', calls=(('Bash', 'x1'),)),
    )
    processor.finalize()

    assert len(langfuse.instances) == 1, 'one run must upload one trace'
    root = langfuse.instances[0].root
    assert [child.name for child in root.children] == [USER, WORKER, OTHER_WORKER]


def test_two_workers_sharing_a_system_prompt_stay_apart(processor_cls: Any, langfuse: type[FakeLangfuse]) -> None:
    """Two Claude Code workers open byte-identically; the label tells them apart."""
    processor = processor_cls(trace_name='ins', trace_metadata={'instance_id': 'ins'})

    send(
        processor,
        completion(WORKER, SYS_WORKER, span_id='s0', calls=(('Bash', 'a1'),)),
        completion(OTHER_WORKER, SYS_WORKER, span_id='s1', calls=(('Bash', 'b1'),)),
    )
    processor.finalize()

    root = langfuse.instances[0].root
    assert [child.name for child in root.children] == [WORKER, OTHER_WORKER]


def test_a_second_conversation_under_a_label_is_a_sibling_not_a_child(
    processor_cls: Any, langfuse: type[FakeLangfuse]
) -> None:
    """A subagent -- or the worker's loop resuming after a compaction -- opens its own
    span beside the worker's, not nested inside it.

    Attribution is by content prefix and carries no parent-child link, and a
    post-compaction continuation is indistinguishable from a subagent, so the tree claims
    no hierarchy: every conversation under a label is a top-level sibling. The subagent's
    opening call carries no tool_call id yet, so it is told from the worker by its own
    opening message -- which is why it must be distinct.
    """
    processor = processor_cls(trace_name='ins', trace_metadata={'instance_id': 'ins'})

    send(
        processor,
        completion(WORKER, SYS_WORKER, span_id='s0', user='build the thing', calls=(('Agent', 'w1'),)),
        completion(WORKER, SYS_SUBAGENT, span_id='s1', user='explore the tree', calls=(('Bash', 's1'),)),
    )
    processor.finalize()

    root = langfuse.instances[0].root
    assert [child.name for child in root.children] == [WORKER, f'{WORKER} · 1'], (
        'both conversations sit directly under the trace, side by side'
    )


# -- the four defects, as regressions -------------------------------------


def test_each_label_keeps_its_own_tool_results(processor_cls: Any, langfuse: type[FakeLangfuse]) -> None:
    """A call from another agent used to discard the pending tool span.

    ``_attach_tool_results_for`` clears the pending generation on every call, so
    with one shared copy the user's ``send_message`` span was closed empty by the
    worker's turn and its result, arriving a turn later, had nothing to attach to.
    """
    processor = processor_cls(trace_name='ins', trace_metadata={'instance_id': 'ins'})

    send(
        processor,
        completion(USER, SYS_USER, span_id='s0', calls=(('send_message', 'u1'),)),
        completion(WORKER, SYS_WORKER, span_id='s1', calls=(('Bash', 'w1'),)),
        completion(
            WORKER,
            SYS_WORKER,
            span_id='s2',
            history=tool_result('w1', 'Bash', 'the worker saw pytest output'),
            calls=(('Edit', 'w2'),),
        ),
        completion(
            USER,
            SYS_USER,
            span_id='s3',
            history=tool_result('u1', 'send_message', 'the worker replied to the user'),
            calls=(('read', 'u2'),),
        ),
    )
    processor.finalize()

    root = langfuse.instances[0].root
    assert find(root, 'Tool: send_message')[0].output == 'the worker replied to the user'
    assert find(root, 'Tool: Bash')[0].output == 'the worker saw pytest output'


def test_a_concurrent_agent_does_not_disturb_a_subagents_record(processor_cls: Any) -> None:
    """Another agent's interleaved turn must not touch a subagent's step counter.

    The stack the upstream tracker kept held one subagent, so an unrelated turn
    overwrote it. Each agent now has its own record, and a turn only advances its own.
    """
    processor = processor_cls(trace_name='ins', trace_metadata={'instance_id': 'ins'})

    send(
        processor,
        completion(WORKER, SYS_WORKER, span_id='s0', user='build', calls=(('Agent', 'w1'),)),
        completion(WORKER, SYS_SUBAGENT, span_id='s1', user='explore', calls=(('Bash', 'sa1'),)),
        completion(USER, SYS_USER, span_id='s2', user='be brief', calls=(('send_message', 'u1'),)),
        completion(
            WORKER,
            SYS_SUBAGENT,
            span_id='s3',
            user='explore',
            history=tool_result('sa1', 'Bash', 'tree listing'),
            calls=(('Read', 'sa2'),),
        ),
    )

    worker = processor._records_by_label[WORKER]
    assert [record.state['_ctx_stack'][0].step for record in worker] == [1, 2], (
        "the worker took one turn, its subagent two, and the user's turn touched neither"
    )


def test_a_worker_continues_its_own_record_after_a_subagent(processor_cls: Any) -> None:
    """A worker's next turn goes back to the worker, not into the subagent it spawned."""
    processor = processor_cls(trace_name='ins', trace_metadata={'instance_id': 'ins'})

    send(
        processor,
        completion(WORKER, SYS_WORKER, span_id='s0', user='build', calls=(('Agent', 'w1'),)),
        completion(WORKER, SYS_SUBAGENT, span_id='s1', user='explore', calls=(('Bash', 'sa1'),)),
        completion(
            WORKER,
            SYS_WORKER,
            span_id='s2',
            user='build',
            history=tool_result('w1', 'Agent', 'subagent done'),
            calls=(('Edit', 'w2'),),
        ),
    )

    worker = processor._records_by_label[WORKER]
    assert len(worker) == 2, 'the worker and its one subagent'
    assert worker[0].state['_ctx_stack'][0].step == 2, "the worker's own second turn returns to it"
    assert worker[1].state['_ctx_stack'][0].step == 1, 'the subagent kept its single turn'


def test_model_tools_and_tokens_are_reported_per_label(processor_cls: Any) -> None:
    """One copy made these first- or last-write-wins across every agent."""
    processor = processor_cls(trace_name='ins', trace_metadata={'instance_id': 'ins'})

    send(
        processor,
        completion(USER, SYS_USER, span_id='s0', calls=(('send_message', 'u1'),), model='glm-5.1'),
        completion(WORKER, SYS_WORKER, span_id='s1', calls=(('Bash', 'w1'),), model='claude-opus-5'),
        completion(WORKER, SYS_WORKER, span_id='s2', calls=(('Edit', 'w2'),), model='claude-opus-5'),
    )

    by_label = processor.get_result()['by_label']
    assert by_label[USER]['model'] == 'glm-5.1'
    assert by_label[WORKER]['model'] == 'claude-opus-5'
    assert by_label[USER]['total_usage']['total'] == 15
    assert by_label[WORKER]['total_usage']['total'] == 30


def test_the_active_label_is_reported_up_to_date(processor_cls: Any) -> None:
    """``get_result`` mid-run must not report the active label's stale snapshot.

    The inherited code rebinds ``_model_name`` rather than mutating it, so the
    label's saved dict holds the value from when it was last activated.
    """
    processor = processor_cls(trace_name='ins', trace_metadata={'instance_id': 'ins'})

    send(processor, completion(WORKER, SYS_WORKER, span_id='s0', calls=(('Bash', 'w1'),), model='claude-opus-5'))

    assert processor.get_result()['by_label'][WORKER]['model'] == 'claude-opus-5'


# -- toolless utility calls (a title, a summary) --------------------------


def test_a_toolless_call_is_bucketed_as_aux_not_a_subagent(processor_cls: Any, langfuse: type[FakeLangfuse]) -> None:
    """Claude Code's title call issues no tool and arrives first; it must not take the
    main name or open a sibling conversation, and the real worker stays the main span."""
    processor = processor_cls(trace_name='ins', trace_metadata={'instance_id': 'ins'})

    send(
        processor,
        completion(WORKER, SYS_WORKER, span_id='t0', user='generate a title for this session', calls=()),
        completion(WORKER, SYS_WORKER, span_id='w0', user='build the thing', calls=(('Bash', 'w1'),)),
    )
    processor.finalize()

    child_names = [str(child.name) for child in langfuse.instances[0].root.children]
    assert child_names == [WORKER, f'{WORKER} · aux'], (
        'the tool-using worker keeps the label name; the toolless call is only the aux bucket'
    )


def test_toolless_calls_share_one_aux_bucket(processor_cls: Any) -> None:
    """Every toolless call under a label funnels into the one aux record, so a run does
    not sprout a subagent per title or summary."""
    processor = processor_cls(trace_name='ins', trace_metadata={'instance_id': 'ins'})

    send(
        processor,
        completion(WORKER, SYS_WORKER, span_id='w0', user='build the thing', calls=(('Bash', 'w1'),)),
        completion(WORKER, SYS_WORKER, span_id='t0', user='generate a title', calls=()),
        completion(WORKER, SYS_WORKER, span_id='t1', user='summarize the run', calls=()),
    )

    assert len(processor._records_by_label[WORKER]) == 1, 'only the main agent is a record'
    assert processor._aux_by_label[WORKER].state['_ctx_stack'][0].step == 2, 'both toolless calls share the aux bucket'


# -- closing --------------------------------------------------------------


def test_every_span_is_closed_and_the_score_is_uploaded_once(processor_cls: Any, langfuse: type[FakeLangfuse]) -> None:
    """The score belongs to the run, and each agent span to one label.

    Reusing the inherited ``finalize`` per label would upload the score once per
    label and, because it returns early without a trace, leave every agent span
    open.
    """
    processor = processor_cls(trace_name='ins', trace_metadata={'instance_id': 'ins'})

    send(
        processor,
        completion(USER, SYS_USER, span_id='s0', calls=(('send_message', 'u1'),)),
        completion(WORKER, SYS_WORKER, span_id='s1', calls=(('Agent', 'w1'),)),
        completion(WORKER, SYS_SUBAGENT, span_id='s2', calls=(('Bash', 's1'),)),
        task_span(),
    )
    processor.finalize()

    client = langfuse.instances[0]
    assert unclosed(client.root) == []
    assert len(client.scores) == 1
    assert client.scores[0]['value'] == pytest.approx(0.42)


def test_a_run_with_no_completions_finalizes_quietly(processor_cls: Any, langfuse: type[FakeLangfuse]) -> None:
    processor = processor_cls(trace_name='ins', trace_metadata={'instance_id': 'ins'})

    processor.finalize()

    assert langfuse.instances == [], 'nothing to record means no trace to open'
    assert processor.get_result()['by_label'] == {}


def test_a_task_span_before_any_completion_still_scores(processor_cls: Any, langfuse: type[FakeLangfuse]) -> None:
    """Evaluation can finish before the trace exists; the score must survive."""
    processor = processor_cls(trace_name='ins', trace_metadata={'instance_id': 'ins'})

    send(processor, task_span(0.25))
    assert langfuse.instances == [], 'a task span alone must not open a trace'

    send(processor, completion(USER, SYS_USER, span_id='s0', calls=(('send_message', 'u1'),)))
    processor.finalize()

    assert [score['value'] for score in langfuse.instances[0].scores] == [pytest.approx(0.25)]


# -- unlabelled calls ------------------------------------------------------


def test_an_unlabelled_call_is_recorded_under_the_sentinel(processor_cls: Any, langfuse: type[FakeLangfuse]) -> None:
    """A bucket, not an error.

    The rubrics evaluator runs outside the agent function and is deliberately
    unlabelled, so it lands here every run. Naming it keeps it visible instead of
    attributing its turns to whichever agent shares its opening message.
    """
    from nexus.extensions.tasks.hitl.trace.labels import UNATTRIBUTED

    processor = processor_cls(trace_name='ins', trace_metadata={'instance_id': 'ins'})

    send(
        processor,
        completion(USER, SYS_USER, span_id='s0', calls=(('send_message', 'u1'),)),
        completion(None, SYS_USER, span_id='s1', calls=(('score', 'e1'),)),
    )
    processor.finalize()

    assert [child.name for child in langfuse.instances[0].root.children] == [USER, UNATTRIBUTED]
    assert UNATTRIBUTED in processor.get_result()['by_label']


def test_a_run_without_labels_uploads_the_tree_the_base_class_would(
    processor_cls: Any, base_cls: Any, langfuse: type[FakeLangfuse]
) -> None:
    """Only the agent span's name differs, and it says what it knows.

    Every recipe that configures the upstream tracker keeps its behaviour, so
    this processor can be adopted without changing what those runs upload.
    """
    trees = {}
    for name, cls in (('base', base_cls), ('hitl', processor_cls)):
        langfuse.instances.clear()
        processor = cls(trace_name='plain', trace_metadata={'instance_id': 'plain'})
        send(
            processor,
            completion(None, SYS_WORKER, span_id='p0', calls=(('Bash', 'b1'),)),
            completion(None, SYS_WORKER, span_id='p1', history=tool_result('b1', 'Bash', 'output')),
            task_span(),
        )
        processor.finalize()
        trees[name] = names(langfuse.instances[0].root)

    from nexus.extensions.tasks.hitl.trace.labels import UNATTRIBUTED

    renamed = [line.replace(UNATTRIBUTED, 'Agent') for line in trees['hitl']]
    assert renamed == trees['base']


# -- the OpenTelemetry context ---------------------------------------------


def _propagated() -> dict[str, Any]:
    """The Langfuse attributes currently attached to the OpenTelemetry context."""
    from opentelemetry import context as otel_context_api

    current = otel_context_api.get_current() or {}
    return {str(key): value for key, value in current.items() if 'langfuse.propagated' in str(key)}


def test_finalize_detaches_the_context_it_attached(processor_cls: Any) -> None:
    """``_ensure_init`` attaches an OTel context; finalize has to give it back.

    The context is process-global (``otel_context_api.attach``), so one left
    attached names every span created afterwards -- including another instance's,
    in a batch that runs several in one process. The inherited ``finalize`` detaches
    it; this processor replaces that method wholesale, so it has to do it too.
    """
    before = _propagated()
    processor = processor_cls(trace_name='ins', session_id='ins', trace_metadata={'instance_id': 'ins'})

    send(processor, completion(USER, SYS_USER, span_id='s0'), task_span())
    assert _propagated(), 'the run should have attached its own context'

    processor.finalize()

    assert _propagated() == before, 'finalize left an OpenTelemetry context attached'


def test_one_runs_context_does_not_outlive_it_into_the_next(processor_cls: Any) -> None:
    """Two runs in one process must not inherit each other's session.

    A batch shares a process across instances. With the context left attached, the
    second run's spans carry the first run's ``session_id`` and ``trace_name``,
    which is what detaches a span from the trace it belongs to.
    """
    first = processor_cls(trace_name='first', session_id='first', trace_metadata={'instance_id': 'first'})
    send(first, completion(USER, SYS_USER, span_id='a0'), task_span())
    first.finalize()

    leaked = _propagated()
    assert 'first' not in str(leaked.values()), f'the first run is still named by the context: {leaked}'
