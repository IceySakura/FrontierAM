from __future__ import annotations

import importlib
import logging
import sys
import types

from typing import Any

import pytest

from nexus.trace.events import SpanEndEvent
from nexus.trace.processors.base import span_processor_registry
from nexus.trace.types import Span


def _forget_tracker_module() -> None:
    sys.modules.pop('nexus.trace.processors.langfuse_tracker', None)
    span_processor_registry._registry.pop('langfuse_tracker', None)


class FakeObservation:
    def __init__(self, **kwargs: Any):
        self.children: list[FakeObservation] = []
        self.ended = False
        self.updates: list[dict[str, Any]] = []
        self.__dict__.update(kwargs)

    def start_observation(self, **kwargs: Any) -> FakeObservation:
        child = FakeObservation(**kwargs)
        self.children.append(child)
        return child

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)
        for key, value in kwargs.items():
            if key == 'metadata' and isinstance(getattr(self, 'metadata', None), dict) and isinstance(value, dict):
                self.metadata.update(value)
            else:
                setattr(self, key, value)

    def end(self) -> None:
        self.ended = True


class FakePropagateContext:
    def __init__(self, store: list[dict[str, Any]], kwargs: dict[str, Any]):
        self._store = store
        self._kwargs = kwargs

    def __enter__(self) -> FakePropagateContext:
        self._store.append(self._kwargs)
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        return None


class FakeLangfuse:
    instances: list[FakeLangfuse] = []
    propagations: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.root: FakeObservation | None = None
        self.scores: list[dict[str, Any]] = []
        self.flushed = False
        self.__class__.instances.append(self)

    def create_trace_id(self, seed: str | None = None) -> str:
        self.seed = seed
        return 'trace-123'

    def start_observation(self, **kwargs: Any) -> FakeObservation:
        self.root = FakeObservation(**kwargs)
        return self.root

    def create_score(self, **kwargs: Any) -> None:
        self.scores.append(kwargs)

    def flush(self) -> None:
        self.flushed = True


def _install_fake_langfuse(monkeypatch) -> None:
    FakeLangfuse.instances.clear()
    FakeLangfuse.propagations.clear()
    _forget_tracker_module()

    langfuse_module = types.ModuleType('langfuse')
    langfuse_module.Langfuse = FakeLangfuse
    langfuse_module.propagate_attributes = lambda **kwargs: FakePropagateContext(
        FakeLangfuse.propagations,
        kwargs,
    )

    types_module = types.ModuleType('langfuse.types')

    class TraceContext:
        def __init__(self, trace_id: str, parent_span_id: str):
            self.trace_id = trace_id
            self.parent_span_id = parent_span_id

    types_module.TraceContext = TraceContext

    monkeypatch.setitem(sys.modules, 'langfuse', langfuse_module)
    monkeypatch.setitem(sys.modules, 'langfuse.types', types_module)


def _completion_span(inputs: dict[str, Any], outputs: dict[str, Any], span_id: str) -> Span:
    return Span(
        trace_id='trace',
        span_id=span_id,
        parent_span_id='',
        name='openai_completion',
        start_time_unix_nano=1_000_000_000,
        end_time_unix_nano=2_000_000_000,
        attributes={'inputs': inputs, 'outputs': outputs},
    )


def _task_span(outputs: dict[str, Any]) -> Span:
    return Span(
        trace_id='trace',
        span_id='task',
        parent_span_id='',
        name='task.demo',
        start_time_unix_nano=0,
        end_time_unix_nano=3_000_000_000,
        attributes={'type': 'task', 'outputs': outputs},
    )


def test_langfuse_tracker_adds_taiji_metadata_turns_tags_and_score(monkeypatch) -> None:
    _install_fake_langfuse(monkeypatch)
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'pk')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'sk')
    monkeypatch.setenv('LANGFUSE_HOST', 'https://langfuse.taiji.example.com')
    monkeypatch.setenv('LANGFUSE_BASE_URL', 'https://wrong.example.com')
    monkeypatch.setenv('TJ_INSTANCE_ID', 'ins-001')
    monkeypatch.setenv('TJ_TASK_ID', 'task-001')
    monkeypatch.setenv('TJ_TASK_NAME', 'demo-task')
    monkeypatch.setenv('TJ_WORKSPACE_ID', 'ws-001')

    module = importlib.import_module('nexus.trace.processors.langfuse_tracker')
    tracker = module.LangfuseTrackerProcessor(tags=['manual'], trace_metadata={'episode': 7})

    system = {'role': 'system', 'content': 'You are an agent.'}
    user = {'role': 'user', 'content': 'Do the task.'}
    tool_call = {
        'id': 'call-1',
        'type': 'function',
        'function': {'name': 'search', 'arguments': '{"query": "nexus"}'},
    }

    tracker.on_end(
        _completion_span(
            {'messages': [system, user], 'kwargs': {'tools': []}},
            {
                'model': 'openai/test-model',
                'choices': [{'message': {'role': 'assistant', 'content': '', 'tool_calls': [tool_call]}}],
                'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5},
            },
            'first',
        ),
        SpanEndEvent(),
    )
    tracker.on_end(
        _completion_span(
            {
                'messages': [
                    system,
                    user,
                    {'role': 'assistant', 'content': None, 'tool_calls': [tool_call]},
                    {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'result'},
                ],
                'kwargs': {'tools': []},
            },
            {'model': 'openai/test-model', 'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]},
            'second',
        ),
        SpanEndEvent(),
    )
    tracker.on_end(_task_span({'score': 0.75, 'detail': 'ok'}), SpanEndEvent())

    tracker.finalize()

    fake = FakeLangfuse.instances[0]
    assert fake.kwargs['host'] == 'https://langfuse.taiji.example.com'
    assert FakeLangfuse.propagations[0]['session_id'] == 'ins-001-ep7'
    assert FakeLangfuse.propagations[0]['trace_name'] == 'ins-001'
    assert FakeLangfuse.propagations[0]['tags'] == [
        'manual',
        'ins-001',
        'task-001',
        'demo-task',
        'ws-001',
        'test-model',
    ]

    assert fake.root is not None
    assert fake.root.name == 'ins-001'
    root_metadata = fake.root.metadata
    assert root_metadata['contract'] == 'episode_trace_generation_turn'
    assert root_metadata['episode'] == 7
    assert root_metadata['episode_id'] == 'ins-001-ep7'
    assert root_metadata['instance_id'] == 'ins-001'
    assert root_metadata['task_id'] == 'task-001'
    assert root_metadata['task_name'] == 'demo-task'
    assert root_metadata['workspace_id'] == 'ws-001'
    assert root_metadata['score'] == 0.75

    agent = fake.root.children[0]
    generations = [child for child in agent.children if getattr(child, 'as_type', None) == 'generation']
    first_generation = generations[0]
    assert first_generation.metadata['turn'] == 1
    assert first_generation.metadata['episode'] == 7
    assert first_generation.metadata['episode_id'] == 'ins-001-ep7'
    assert first_generation.metadata['contract'] == 'episode_trace_generation_turn'
    final_generation = generations[-1]
    assert final_generation.name == 'Final Response'
    assert final_generation.metadata['is_final'] is True
    assert final_generation.metadata['turn'] == 2
    assert final_generation.metadata['episode'] == 7
    assert final_generation.metadata['episode_id'] == 'ins-001-ep7'
    assert final_generation.metadata['contract'] == 'episode_trace_generation_turn'

    tool_observation = first_generation.children[0]
    assert tool_observation.metadata['turn'] == 1
    assert tool_observation.metadata['step'] == 1
    assert tool_observation.metadata['global_step'] == 1
    assert tool_observation.metadata['episode'] == 7
    assert tool_observation.metadata['episode_id'] == 'ins-001-ep7'

    assert fake.scores == [
        {
            'trace_id': 'trace-123',
            'name': 'episode_reward',
            'value': 0.75,
            'comment': 'instance=ins-001, episode=7, task=demo-task',
        }
    ]
    assert fake.flushed is True


def test_langfuse_tracker_skips_non_finite_score(monkeypatch) -> None:
    _install_fake_langfuse(monkeypatch)
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'pk')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'sk')

    module = importlib.import_module('nexus.trace.processors.langfuse_tracker')
    tracker = module.LangfuseTrackerProcessor()

    tracker.on_end(
        _completion_span(
            {'messages': [{'role': 'user', 'content': 'Do the task.'}], 'kwargs': {'tools': []}},
            {'model': 'openai/test-model', 'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]},
            'first',
        ),
        SpanEndEvent(),
    )
    tracker.on_end(_task_span({'score': float('nan')}), SpanEndEvent())
    tracker.finalize()

    fake = FakeLangfuse.instances[0]
    assert fake.scores == []
    assert 'score' not in fake.root.metadata


def test_langfuse_tracker_adds_episode_metadata_to_subagent_spans(monkeypatch) -> None:
    _install_fake_langfuse(monkeypatch)
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'pk')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'sk')
    monkeypatch.setenv('TJ_INSTANCE_ID', 'ins-sub')

    module = importlib.import_module('nexus.trace.processors.langfuse_tracker')
    tracker = module.LangfuseTrackerProcessor(trace_metadata={'episode': 3})

    main_system = {'role': 'system', 'content': 'main system prompt'}
    sub_system = {'role': 'system', 'content': 'subagent system prompt'}
    user = {'role': 'user', 'content': 'Do the task.'}
    agent_call = {
        'id': 'agent-call-1',
        'type': 'function',
        'function': {
            'name': 'Agent',
            'arguments': '{"description": "inspect", "prompt": "look closer"}',
        },
    }
    # Subagent transitions are only tracked for tool-carrying calls; tool-less utility
    # sub-calls (e.g. "generate a title") must not be mistaken for a subagent.
    tools = [{'type': 'function', 'function': {'name': 'Agent', 'description': 'spawn a subagent'}}]

    tracker.on_end(
        _completion_span(
            {'messages': [main_system, user], 'kwargs': {'tools': tools}},
            {'model': 'openai/test-model', 'choices': [{'message': {'role': 'assistant', 'tool_calls': [agent_call]}}]},
            'main-first',
        ),
        SpanEndEvent(),
    )
    tracker.on_end(
        _completion_span(
            {'messages': [sub_system, user], 'kwargs': {'tools': tools}},
            {'model': 'openai/test-model', 'choices': [{'message': {'role': 'assistant', 'content': 'sub result'}}]},
            'sub-first',
        ),
        SpanEndEvent(),
    )
    tracker.on_end(
        _completion_span(
            {
                'messages': [
                    main_system,
                    user,
                    {'role': 'assistant', 'content': None, 'tool_calls': [agent_call]},
                    {'role': 'tool', 'tool_call_id': 'agent-call-1', 'content': 'sub result'},
                ],
                'kwargs': {'tools': tools},
            },
            {'model': 'openai/test-model', 'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]},
            'main-second',
        ),
        SpanEndEvent(),
    )
    tracker.finalize()

    fake = FakeLangfuse.instances[0]
    assert fake.root is not None
    main_agent = fake.root.children[0]
    main_generation = next(child for child in main_agent.children if getattr(child, 'as_type', None) == 'generation')
    agent_tool = main_generation.children[0]
    assert agent_tool.metadata['turn'] == 1
    assert agent_tool.metadata['episode'] == 3
    assert agent_tool.metadata['episode_id'] == 'ins-sub-ep3'

    subagent_wrapper = agent_tool.children[0]
    assert subagent_wrapper.metadata['episode'] == 3
    assert subagent_wrapper.metadata['episode_id'] == 'ins-sub-ep3'
    assert subagent_wrapper.metadata['parent_turn'] == 1
    assert subagent_wrapper.metadata['parent_global_step'] == 1

    sub_generation = next(
        child for child in subagent_wrapper.children if getattr(child, 'as_type', None) == 'generation'
    )
    assert sub_generation.metadata['turn'] == 1
    assert sub_generation.metadata['episode'] == 3
    assert sub_generation.metadata['episode_id'] == 'ins-sub-ep3'


def test_langfuse_tracker_uploads_raw_openai_reasoning_items(monkeypatch) -> None:
    _install_fake_langfuse(monkeypatch)
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'pk')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'sk')

    module = importlib.import_module('nexus.trace.processors.langfuse_tracker')
    tracker = module.LangfuseTrackerProcessor()

    system = {'role': 'system', 'content': 'You are an agent.'}
    user = {'role': 'user', 'content': 'Do the task.'}
    encrypted_content = 'raw-gpt55-encrypted-content'
    reasoning_item = {
        'id': 'rs_123',
        'type': 'reasoning',
        'summary': [{'type': 'summary_text', 'text': 'called search'}],
        'encrypted_content': encrypted_content,
        'status': 'completed',
    }
    tool_call = {
        'id': 'call-1',
        'type': 'function',
        'function': {'name': 'search', 'arguments': '{"query": "nexus"}'},
    }

    tracker.on_end(
        _completion_span(
            {'messages': [system, user], 'kwargs': {'tools': []}},
            {
                'model': 'openai/gpt-5.5',
                'choices': [
                    {
                        'message': {
                            'role': 'assistant',
                            'content': '',
                            'reasoning_items': [reasoning_item],
                            'tool_calls': [tool_call],
                        }
                    }
                ],
            },
            'first',
        ),
        SpanEndEvent(),
    )
    tracker.on_end(
        _completion_span(
            {
                'messages': [
                    system,
                    user,
                    {
                        'role': 'assistant',
                        'content': None,
                        'reasoning_items': [reasoning_item],
                        'tool_calls': [tool_call],
                    },
                    {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'result'},
                ],
                'kwargs': {'tools': []},
            },
            {'model': 'openai/gpt-5.5', 'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]},
            'second',
        ),
        SpanEndEvent(),
    )

    assert FakeLangfuse.instances, 'expected at least one FakeLangfuse instance'
    fake = FakeLangfuse.instances[0]
    assert fake.root.children, 'expected at least one root child (agent)'
    agent = fake.root.children[0]
    generations = [child for child in agent.children if getattr(child, 'as_type', None) == 'generation']
    assert len(generations) >= 2, f'expected at least 2 generations, got {len(generations)}'
    first_generation = generations[0]
    second_generation = generations[1]

    assert first_generation.output['reasoning_items'][0]['encrypted_content'] == encrypted_content
    assert first_generation.output['reasoning_items'][0]['tool_call_ids'] == ['call-1']
    # `input.messages` is the full cumulative history, so locate the replayed assistant turn
    # instead of assuming a fixed index.
    replayed_assistant = next(
        (msg
         for msg in second_generation.input['messages']
         if msg.get('role') == 'assistant' and msg.get('reasoning_items')),
        None,
    )
    assert replayed_assistant is not None, 'no assistant message with reasoning_items found in input history'
    assert replayed_assistant['reasoning_items'][0]['encrypted_content'] == encrypted_content
    assert [entry['id'] for entry in second_generation.input['input_reasoning_items']] == ['rs_123']
    assert second_generation.input['input_reasoning_items'][0]['encrypted_content'] == encrypted_content
    assert second_generation.input['input_reasoning_items'][0]['message_index'] == 2
    assert 'reasoning_items' not in second_generation.input
    assert second_generation.metadata['reasoning_round_trip_rate'] == '1/1'
    assert second_generation.metadata['reasoning_round_trip'] == [
        {
            'id': 'rs_123',
            'match_on': 'encrypted_content',
            'found': True,
            'tool_call_ids': ['call-1'],
            'landed_tool_call_ids': ['call-1'],
            'landed_at_message_index': 2,
            'landed_at_reasoning_index': 0,
        }
    ]


def test_langfuse_tracker_uploads_all_historical_input_reasoning_items(monkeypatch) -> None:
    _install_fake_langfuse(monkeypatch)
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'pk')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'sk')

    module = importlib.import_module('nexus.trace.processors.langfuse_tracker')
    tracker = module.LangfuseTrackerProcessor()

    def _item(item_id: str, blob: str) -> dict[str, Any]:
        return {
            'id': item_id,
            'type': 'reasoning',
            'encrypted_content': blob,
            'status': 'completed',
        }

    first = _item('rs_1', 'enc-1')
    second = _item('rs_2', 'enc-2')
    first_call = {'id': 'call-1', 'type': 'function', 'function': {'name': 'search', 'arguments': '{}'}}
    second_call = {'id': 'call-2', 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'}}
    system = {'role': 'system', 'content': 'You are an agent.'}
    user = {'role': 'user', 'content': 'Do the task.'}

    tracker.on_end(
        _completion_span(
            {'messages': [system, user], 'kwargs': {'tools': []}},
            {
                'model': 'openai/gpt-6-astra',
                'choices': [
                    {
                        'message': {
                            'role': 'assistant',
                            'content': '',
                            'reasoning_items': [first],
                            'tool_calls': [first_call],
                        }
                    }
                ],
            },
            'turn-1',
        ),
        SpanEndEvent(),
    )
    tracker.on_end(
        _completion_span(
            {
                'messages': [
                    system,
                    user,
                    {
                        'role': 'assistant',
                        'content': None,
                        'reasoning_items': [first],
                        'tool_calls': [first_call],
                    },
                    {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'result-1'},
                ],
                'kwargs': {'tools': []},
            },
            {
                'model': 'openai/gpt-6-astra',
                'choices': [
                    {
                        'message': {
                            'role': 'assistant',
                            'content': '',
                            'reasoning_items': [second],
                            'tool_calls': [second_call],
                        }
                    }
                ],
            },
            'turn-2',
        ),
        SpanEndEvent(),
    )
    tracker.on_end(
        _completion_span(
            {
                'messages': [
                    system,
                    user,
                    {
                        'role': 'assistant',
                        'content': None,
                        'reasoning_items': [first],
                        'tool_calls': [first_call],
                    },
                    {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'result-1'},
                    {
                        'role': 'assistant',
                        'content': None,
                        'reasoning_items': [second],
                        'tool_calls': [second_call],
                    },
                    {'role': 'tool', 'tool_call_id': 'call-2', 'content': 'result-2'},
                ],
                'kwargs': {'tools': []},
            },
            {'model': 'openai/gpt-6-astra', 'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]},
            'turn-3',
        ),
        SpanEndEvent(),
    )

    fake = FakeLangfuse.instances[0]
    agent = fake.root.children[0]
    generations = [child for child in agent.children if getattr(child, 'as_type', None) == 'generation']
    third = generations[2]
    uploaded = third.input['input_reasoning_items']
    assert [entry['id'] for entry in uploaded] == ['rs_1', 'rs_2']
    assert [entry['encrypted_content'] for entry in uploaded] == ['enc-1', 'enc-2']
    assert [entry['message_index'] for entry in uploaded] == [2, 4]


def test_suppress_otel_detach_log_filters_only_detach_record() -> None:
    """The scoped filter drops the benign detach ERROR and nothing else, then detaches."""
    _forget_tracker_module()
    module = importlib.import_module('nexus.trace.processors.langfuse_tracker')

    otel_logger = logging.getLogger('opentelemetry.context')
    seen: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record.getMessage())

    handler = _Capture()
    otel_logger.addHandler(handler)
    try:
        with module._suppressed_otel_detach_log():
            # This is exactly what opentelemetry.context.detach logs on a cross-context token.
            otel_logger.error('Failed to detach context')
            otel_logger.error('some other unrelated otel error')
        # Filter must be removed on exit -> this record is NOT dropped.
        assert otel_logger.filters == []
        otel_logger.error('Failed to detach context')
    finally:
        otel_logger.removeHandler(handler)

    assert 'Failed to detach context' in seen  # the post-context record got through
    assert 'some other unrelated otel error' in seen  # unrelated records never filtered
    # Inside the context, the detach message was filtered out (only the unrelated one + the
    # post-context one remain).
    assert seen.count('Failed to detach context') == 1


def _install_fake_langfuse_with_logging_detach(monkeypatch) -> None:
    """Like _install_fake_langfuse, but the propagate ctx __exit__ mimics OpenTelemetry:

    it logs 'Failed to detach context' on the opentelemetry.context logger (and returns
    normally), reproducing the cross-context detach the real SDK produces.
    """
    _install_fake_langfuse(monkeypatch)
    otel_logger = logging.getLogger('opentelemetry.context')

    class LoggingPropagateContext:
        def __init__(self, **_kwargs: Any):
            pass

        def __enter__(self) -> LoggingPropagateContext:
            return self

        def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
            otel_logger.error('Failed to detach context')
            return

    langfuse_module = sys.modules['langfuse']
    langfuse_module.propagate_attributes = lambda **kwargs: LoggingPropagateContext(**kwargs)


def test_langfuse_finalize_suppresses_cross_context_detach_error(monkeypatch) -> None:
    _install_fake_langfuse_with_logging_detach(monkeypatch)
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'pk')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'sk')

    module = importlib.import_module('nexus.trace.processors.langfuse_tracker')
    tracker = module.LangfuseTrackerProcessor(trace_metadata={'episode': 1})

    system = {'role': 'system', 'content': 'You are an agent.'}
    user = {'role': 'user', 'content': 'Do the task.'}
    tracker.on_end(
        _completion_span(
            {'messages': [system, user], 'kwargs': {'tools': []}},
            {'model': 'openai/test-model', 'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]},
            'first',
        ),
        SpanEndEvent(),
    )

    otel_logger = logging.getLogger('opentelemetry.context')
    leaked: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            leaked.append(record.getMessage())

    handler = _Capture()
    otel_logger.addHandler(handler)
    try:
        tracker.finalize()
    finally:
        otel_logger.removeHandler(handler)

    # The detach ERROR emitted during finalize's __exit__ must have been filtered out,
    # and the filter must not linger on the logger afterwards.
    assert 'Failed to detach context' not in leaked
    assert otel_logger.filters == []
    assert FakeLangfuse.instances[0].flushed is True


def test_langfuse_tracker_supports_real_sdk_v4_api() -> None:
    langfuse = pytest.importorskip('langfuse')
    from langfuse import Langfuse, propagate_attributes

    assert hasattr(langfuse, 'propagate_attributes')
    assert hasattr(Langfuse, 'create_trace_id')
    assert hasattr(Langfuse, 'start_observation')
    assert hasattr(Langfuse, 'create_score')
    assert hasattr(Langfuse, 'flush')
    assert not hasattr(Langfuse, 'score')

    _forget_tracker_module()
    module = importlib.import_module('nexus.trace.processors.langfuse_tracker')

    assert module.Langfuse is Langfuse
    assert module.propagate_attributes is propagate_attributes


def test_langfuse_trace_url_prefers_project_path() -> None:
    from nexus.trace.processors.langfuse_tracker import _langfuse_trace_url

    assert _langfuse_trace_url('http://lf.example/', 'proj-1', 'tid') == 'http://lf.example/project/proj-1/traces/tid'
    assert _langfuse_trace_url('http://lf.example', None, 'tid') == 'http://lf.example/trace/tid'
    assert _langfuse_trace_url('http://lf.example', 'proj-1', None) is None


def test_langfuse_trace_url_maps_taiji_api_host_to_ui() -> None:
    from nexus.trace.processors.langfuse_tracker import _langfuse_trace_url

    assert (
        _langfuse_trace_url(
            'http://langfuse-taiji-api.woa.com',
            'cmt4mssz9001fzy06o7dmgy1e',
            'tid-1',
        )
        == 'https://langfuse-taiji.woa.com/project/cmt4mssz9001fzy06o7dmgy1e/traces/tid-1'
    )


def test_fetch_langfuse_project_id_caches_per_host_and_key(monkeypatch) -> None:
    from nexus.trace.processors import langfuse_tracker as module

    monkeypatch.delenv('LANGFUSE_PROJECT_ID', raising=False)
    module._PROJECT_ID_CACHE.clear()
    module._PROJECT_ID_STARTED.clear()
    calls: list[str] = []

    class _Resp:
        def read(self) -> bytes:
            return b'{"data":[{"id":"proj-cached"}]}'

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

    def _urlopen(request, timeout=None):  # noqa: ANN001
        calls.append(request.full_url)
        return _Resp()

    monkeypatch.setattr(module.urllib.request, 'urlopen', _urlopen)
    first = module._fetch_langfuse_project_id('http://lf.example/', 'pk', 'sk')
    second = module._fetch_langfuse_project_id('http://lf.example', 'pk', 'sk')
    assert first == second == 'proj-cached'
    assert calls == ['http://lf.example/api/public/projects']


def test_schedule_langfuse_project_id_does_not_block_the_caller(monkeypatch) -> None:
    import threading
    import time

    from nexus.trace.processors import langfuse_tracker as module

    monkeypatch.delenv('LANGFUSE_PROJECT_ID', raising=False)
    module._PROJECT_ID_CACHE.clear()
    module._PROJECT_ID_STARTED.clear()
    entered = threading.Event()
    release = threading.Event()

    class _Resp:
        def read(self) -> bytes:
            return b'{"data":[{"id":"proj-later"}]}'

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

    def _urlopen(request, timeout=None):  # noqa: ANN001
        entered.set()
        assert release.wait(timeout=2)
        return _Resp()

    monkeypatch.setattr(module.urllib.request, 'urlopen', _urlopen)
    started = time.monotonic()
    resolved = module._schedule_langfuse_project_id('http://lf.example', 'pk', 'sk', 'tid')
    elapsed = time.monotonic() - started

    assert resolved is None
    assert elapsed < 0.5
    assert entered.wait(timeout=1)
    # A second caller while the lookup is in flight must not open another request.
    assert module._schedule_langfuse_project_id('http://lf.example/', 'pk', 'sk', 'tid') is None
    release.set()

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and module._cached_project_id(('http://lf.example', 'pk')) is None:
        time.sleep(0.01)
    assert module._cached_project_id(('http://lf.example', 'pk')) == 'proj-later'

    tracker = module.LangfuseTrackerProcessor(public_key='pk', secret_key='sk', host='http://lf.example')
    tracker._trace_id = 'tid'
    tracker._project_id = None
    assert tracker.get_result()['trace_url'] == 'http://lf.example/project/proj-later/traces/tid'
