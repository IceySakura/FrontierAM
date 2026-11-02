from __future__ import annotations

import copy

from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from nexus.gateway.observers import TrajectoryUpdate, create_trajectory_observers
from nexus.gateway.observers.langfuse import LangfuseTrajectoryObserver
from nexus.gateway.trajectory import (
    Chain,
    InvocationFacts,
    JoinLink,
    ProviderFacts,
    SegmentBoundary,
    SpawnLink,
    TokenAccount,
    Trajectory,
    TrajectoryNode,
)


class FakeHandle:
    def __init__(
        self,
        client: Any,
        name: str,
        kind: str,
        *,
        parent: Any = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.name = name
        self.kind = kind
        self.parent = parent
        self.payload = payload or {}
        self.children: list[FakeHandle] = []
        self.ended = False
        self.end_time_ns: int | None = None
        self._otel_span = SimpleNamespace(_start_time=None)

    def start_observation(self, *, name: str, as_type: str = 'span', **payload: Any) -> FakeHandle:
        child = FakeHandle(self.client, name, as_type, parent=self, payload=payload)
        self.children.append(child)
        self.client.handles.append(child)
        return child

    def end(self, *, end_time: int | None = None) -> FakeHandle:
        self.ended = True
        self.end_time_ns = end_time
        return self


class FakeLangfuse:
    def __init__(self) -> None:
        self.handles: list[FakeHandle] = []
        self.scores: list[dict[str, Any]] = []
        self.flush_count = 0
        self.trace_seeds: list[str | None] = []

    def create_trace_id(self, *, seed: str | None = None) -> str:
        self.trace_seeds.append(seed)
        return f'trace-{len(self.trace_seeds)}'

    def start_observation(self, *, name: str, as_type: str = 'span', **payload: Any) -> FakeHandle:
        handle = FakeHandle(self, name, as_type, payload=payload)
        self.handles.append(handle)
        return handle

    def create_score(self, **score: Any) -> None:
        self.scores.append(score)

    def flush(self) -> None:
        self.flush_count += 1


def _node(
    node_id: int,
    parent: int | None,
    generated_by: str,
    message: dict[str, Any],
    *,
    start_time: float,
    end_time: float,
    request_kwargs: dict[str, Any] | None = None,
    provider_facts: ProviderFacts | None = None,
    request_message_overrides: dict[int, dict[str, Any]] | None = None,
) -> TrajectoryNode:
    return TrajectoryNode(
        node_id=node_id,
        message_id=100 + node_id,
        parent=parent,
        generated_by=generated_by,
        children=[],
        original_message=message,
        request_kwargs=request_kwargs,
        provider_facts=provider_facts,
        request_message_overrides=request_message_overrides or {},
        timeline=[
            {
                'event_type': 'llm_request' if generated_by == 'llm' else generated_by,
                'start_time': start_time,
                'end_time': end_time,
                'status': 'ok',
            }
        ],
    )


def _trajectory(session_id: str = 'run-1') -> Trajectory:
    agent_call = {
        'id': 'spawn-1',
        'type': 'function',
        'function': {'name': 'Agent', 'arguments': '{"description":"worker"}'},
    }
    nodes = {
        0: _node(0, None, 'client', {'role': 'system', 'content': 'main system'}, start_time=100, end_time=101),
        1: _node(1, 0, 'client', {'role': 'user', 'content': 'fix the issue'}, start_time=102, end_time=103),
        2: _node(
            2,
            1,
            'llm',
            {'role': 'assistant', 'content': '', 'tool_calls': [agent_call]},
            start_time=104,
            end_time=106,
            request_kwargs={'temperature': 0.2, 'tools': [{'type': 'function', 'function': {'name': 'Agent'}}]},
            provider_facts=ProviderFacts(
                model='openai/main-model',
                response_id='response-2',
                usage={'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12},
                finish_reason='tool_calls',
                inference_time=2.0,
            ),
        ),
        3: _node(3, None, 'client', {'role': 'system', 'content': 'child system'}, start_time=107, end_time=108),
        4: _node(
            4,
            3,
            'client',
            {'role': 'user', 'content': 'inspect repository'},
            start_time=109,
            end_time=110,
        ),
        5: _node(
            5,
            4,
            'llm',
            {'role': 'assistant', 'content': 'child answer'},
            start_time=111,
            end_time=113,
            provider_facts=ProviderFacts(model='openai/child-model'),
        ),
        6: _node(
            6,
            5,
            'client',
            {'role': 'tool', 'tool_call_id': 'search-1', 'name': 'search', 'content': 'search result'},
            start_time=114,
            end_time=114.5,
        ),
        7: _node(
            7,
            2,
            'observation',
            {'role': 'tool', 'tool_call_id': 'spawn-1', 'name': 'Agent', 'content': 'child result'},
            start_time=115,
            end_time=116,
        ),
        8: _node(
            8,
            7,
            'llm',
            {'role': 'assistant', 'content': 'done'},
            start_time=117,
            end_time=120,
            request_kwargs={'temperature': 0.1},
            provider_facts=ProviderFacts(model='openai/main-model'),
            request_message_overrides={1: {'role': 'user', 'content': 'rewritten task'}},
        ),
    }
    for parent, child in ((0, 1), (1, 2), (2, 7), (3, 4), (4, 5), (5, 6), (7, 8)):
        nodes[parent].children.append(child)
    nodes[2].templated_message = '<assistant>tool call</assistant>'
    nodes[2].token_account = TokenAccount(token_ids=np.array([10, 11, 12], dtype=np.int32), gen_start=1, gen_len=2)
    nodes[2].as_sent_variants = [{'provider': 'openai', 'response_id': 'response-2'}]
    nodes[2].segment_boundary = SegmentBoundary(starts_new_segment=True, kind='conversation')
    nodes[2].selection = 'selected'
    nodes[2].invocation = InvocationFacts(iteration=1, retry=0)
    nodes[2].reward = 0.25
    return Trajectory(
        session_id=session_id,
        state='completed',
        nodes=nodes,
        roots=[0, 3],
        chains={
            'main': Chain(chain_id='main', roots=[0]),
            'worker': Chain(
                chain_id='worker',
                kind='subagent',
                parent_chain_id='main',
                roots=[3],
                spawn=SpawnLink(node=2, tool_use_id='spawn-1'),
                join=JoinLink(node=7, tool_use_id='spawn-1', kind='inline'),
                detector='declared',
                evidence={'source': 'request_meta'},
            ),
        },
        timeline=[{'event_type': 'session', 'start_time': 99, 'end_time': 121}],
        final_reward=0.75,
        termination_reason='ENV_DONE',
        metrics={'llm_inference_time': 5.0},
        extra={'benchmark': 'swe'},
    )


def _handle_for_chain(client: FakeLangfuse, chain_id: str) -> FakeHandle:
    return next(
        handle
        for handle in client.handles
        if handle.payload.get('metadata', {}).get('object_type') == 'trajectory_chain'
        and handle.payload['metadata']['chain_id'] == chain_id
    )


def _handle_for_node(client: FakeLangfuse, node_id: int) -> FakeHandle:
    return next(
        handle
        for handle in client.handles
        if handle.payload.get('metadata', {}).get('object_type') == 'trajectory_node'
        and handle.payload['metadata']['node_id'] == node_id
    )


def test_final_export_maps_trajectory_chains_and_nodes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('TJ_INSTANCE_ID', 'ins-1')
    client = FakeLangfuse()
    observer = LangfuseTrajectoryObserver(tags=('manual',), metadata={'episode': 3}, client=client)
    trajectory = _trajectory()

    observer.on_final(trajectory)

    trace = next(handle for handle in client.handles if handle.parent is None)
    main_chain = _handle_for_chain(client, 'main')
    worker_chain = _handle_for_chain(client, 'worker')
    assert len(client.handles) == (
        1 + len(trajectory.chains) + len(trajectory.nodes) + len(trajectory.to_segment_node_lists())
    )
    assert main_chain.kind == 'agent', 'main chains must be exported as Langfuse agents'
    assert worker_chain.kind == 'agent', 'subagent chains must be exported as Langfuse agents'
    assert main_chain.parent is trace
    assert worker_chain.parent is main_chain
    for node_id in (0, 1, 2, 7, 8):
        assert _handle_for_node(client, node_id).parent is main_chain
    for node_id in (3, 4, 5, 6):
        assert _handle_for_node(client, node_id).parent is worker_chain
    assert trace.payload['metadata']['root_node_ids'] == [0, 3]
    assert trace.payload['metadata']['node_count'] == len(trajectory.nodes)
    assert worker_chain.payload['metadata']['member_node_ids'] == [3, 4, 5, 6]
    assert worker_chain.payload['metadata']['spawn'] == {
        'node': 2,
        'tool_use_id': 'spawn-1',
        'candidates': [],
    }
    assert worker_chain.payload['metadata']['join'] == {
        'node': 7,
        'tool_use_id': 'spawn-1',
        'kind': 'inline',
    }
    assert all(handle.ended for handle in client.handles)


def test_flattened_export_preserves_physical_edges_without_mutating_nodes() -> None:
    client = FakeLangfuse()
    trajectory = _trajectory()
    original_nodes = copy.deepcopy(trajectory.nodes)

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    np.testing.assert_equal(
        {node_id: asdict(node) for node_id, node in trajectory.nodes.items()},
        {node_id: asdict(node) for node_id, node in original_nodes.items()},
    )
    for node_id, node in original_nodes.items():
        handle = _handle_for_node(client, node_id)
        assert handle.parent is _handle_for_chain(client, trajectory.node_chain(node_id))
        assert handle.payload['metadata']['parent_node_id'] == node.parent
        assert handle.payload['metadata']['child_node_ids'] == node.children


def test_client_tool_message_maps_to_tool_observation() -> None:
    client = FakeLangfuse()
    trajectory = _trajectory()
    assert trajectory.nodes[6].generated_by == 'client', 'fixture must match the production resent-history path'

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    tool = _handle_for_node(client, 6)
    assert tool.kind == 'tool', 'resent client tool messages must remain Langfuse tools'
    assert tool.name == 'Tool Result: search [node 6]'


def test_nodes_without_declared_chains_attach_to_trace() -> None:
    client = FakeLangfuse()
    trajectory = _trajectory()
    trajectory.chains = {}

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    trace = next(handle for handle in client.handles if handle.parent is None)
    assert len(client.handles) == 1 + len(trajectory.nodes)
    for node_id, node in trajectory.nodes.items():
        handle = _handle_for_node(client, node_id)
        assert handle.parent is trace
        assert handle.payload['metadata']['parent_node_id'] == node.parent
        assert handle.payload['metadata']['child_node_ids'] == node.children


def test_cross_chain_physical_edge_is_preserved_as_metadata_not_visual_parent() -> None:
    client = FakeLangfuse()
    trajectory = _trajectory()
    trajectory.nodes[3].parent = 2
    trajectory.nodes[2].children.insert(0, 3)
    trajectory.roots.remove(3)

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    worker_chain = _handle_for_chain(client, 'worker')
    worker_root = _handle_for_node(client, 3)
    assert worker_root.parent is worker_chain
    assert worker_root.payload['metadata']['parent_node_id'] == 2
    assert _handle_for_node(client, 2).payload['metadata']['child_node_ids'] == [3, 7]


def test_invalid_chain_parent_relationship_falls_back_to_trace() -> None:
    client = FakeLangfuse()
    trajectory = _trajectory()
    trajectory.chains['main'].parent_chain_id = 'worker'

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    trace = next(handle for handle in client.handles if handle.parent is None)
    main_chain = _handle_for_chain(client, 'main')
    worker_chain = _handle_for_chain(client, 'worker')
    assert main_chain.parent is trace
    assert worker_chain.parent is trace
    assert main_chain.payload['metadata']['parent_chain_id'] == 'worker'
    assert main_chain.payload['metadata']['display_parent_chain_id'] is None
    assert worker_chain.payload['metadata']['parent_chain_id'] == 'main'
    assert worker_chain.payload['metadata']['display_parent_chain_id'] is None


def test_node_payload_uses_only_recorded_local_facts() -> None:
    client = FakeLangfuse()
    trajectory = _trajectory()

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    llm = _handle_for_node(client, 2)
    final_llm = _handle_for_node(client, 8)
    tool_result = _handle_for_node(client, 6)
    assert llm.kind == 'generation'
    assert llm.payload['output'] == trajectory.nodes[2].original_message
    assert llm.payload['input']['request_kwargs'] == trajectory.nodes[2].request_kwargs
    assert llm.payload['input']['messages'] == [trajectory.nodes[index].original_message for index in (0, 1)]
    assert final_llm.payload['input']['request_message_overrides'] == [
        {'node_id': 1, 'message': {'role': 'user', 'content': 'rewritten task'}}
    ]
    assert tool_result.kind == 'tool'
    assert tool_result.payload['input'] is None
    assert tool_result.payload['output'] == trajectory.nodes[6].original_message
    assert llm.payload['model'] == 'openai/main-model'
    assert llm.payload['usage_details'] == {'input': 10, 'output': 2, 'total': 12, 'cached_tokens': 0}
    assert llm.payload['metadata']['token_account']['token_count'] == 3
    assert llm.payload['metadata']['token_account']['gen_len'] == 2
    assert llm.payload['metadata']['segment_boundary']['kind'] == 'conversation'
    assert llm.payload['metadata']['invocation'] == {'iteration': 1, 'retry': 0}


def test_export_uses_trajectory_timestamps_for_trace_chains_and_nodes() -> None:
    client = FakeLangfuse()

    LangfuseTrajectoryObserver(client=client).on_final(_trajectory())

    trace = next(handle for handle in client.handles if handle.parent is None)
    main_chain = _handle_for_chain(client, 'main')
    worker_chain = _handle_for_chain(client, 'worker')
    assert trace._otel_span._start_time == 99_000_000_000
    assert trace.end_time_ns == 121_000_000_000
    assert main_chain._otel_span._start_time == 100_000_000_000
    assert main_chain.end_time_ns == 120_000_000_000
    assert worker_chain._otel_span._start_time == 107_000_000_000
    assert worker_chain.end_time_ns == 114_500_000_000
    assert _handle_for_node(client, 2)._otel_span._start_time == 104_000_000_000
    assert _handle_for_node(client, 2).end_time_ns == 106_000_000_000


def test_missing_input_times_use_first_request_without_changing_archive() -> None:
    client = FakeLangfuse()
    trajectory = _trajectory()
    for node_id in (0, 1, 3, 4, 6, 7):
        trajectory.nodes[node_id].timeline = []
    original = copy.deepcopy(trajectory.dump())

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    for node_id, milliseconds in ((0, 103998), (1, 103999), (3, 110998), (4, 110999), (7, 116999)):
        handle = _handle_for_node(client, node_id)
        assert handle._otel_span._start_time == handle.end_time_ns == milliseconds * 1_000_000
        assert handle.payload['metadata']['timestamp_source'] == 'inferred_first_request'
        assert handle.payload['metadata']['timeline'] == []
    terminal_tool = _handle_for_node(client, 6)
    assert terminal_tool._otel_span._start_time is terminal_tool.end_time_ns is None
    assert terminal_tool.payload['metadata']['timestamp_source'] == 'missing'
    assert terminal_tool.payload['metadata']['export_timestamp_semantics'] == 'sdk_default_not_execution_time'
    assert 'level' not in terminal_tool.payload
    assert 'status_message' not in terminal_tool.payload
    assert _handle_for_node(client, 2).payload['metadata']['timestamp_source'] == 'trajectory_tree'
    assert _handle_for_node(client, 2).end_time_ns == 106_000_000_000
    assert _handle_for_chain(client, 'worker')._otel_span._start_time == 110_998_000_000
    assert trajectory.dump() == original


def test_missing_inputs_use_earliest_request_across_branches_not_tree_order() -> None:
    client = FakeLangfuse()
    trajectory = _trajectory()
    for node_id in (0, 1):
        trajectory.nodes[node_id].timeline = []
    trajectory.nodes[9] = _node(
        9, 1, 'llm', {'role': 'assistant', 'content': 'earlier request'}, start_time=103, end_time=105
    )
    trajectory.nodes[1].children.append(9)

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    for node_id in (0, 1):
        handle = _handle_for_node(client, node_id)
        assert handle._otel_span._start_time == handle.end_time_ns == 103_000_000_000 - (2 - node_id) * 1_000_000
        assert handle.payload['metadata']['anchor_node_id'] == 9
        assert handle.payload['metadata']['timestamp_source'] == 'inferred_first_request'


def test_missing_input_without_chains_can_still_use_physical_request_path() -> None:
    client = FakeLangfuse()
    trajectory = _trajectory()
    trajectory.chains = {}
    for node_id in (0, 1, 6):
        trajectory.nodes[node_id].timeline = []

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    assert _handle_for_node(client, 0)._otel_span._start_time == 103_998_000_000
    terminal_tool = _handle_for_node(client, 6)
    assert terminal_tool._otel_span._start_time is None
    assert terminal_tool.payload['metadata']['timestamp_source'] == 'missing'


def test_missing_llm_times_are_not_replaced_with_input_fallback() -> None:
    client = FakeLangfuse()
    trajectory = _trajectory()
    trajectory.nodes[2].timeline = []

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    handle = _handle_for_node(client, 2)
    assert handle._otel_span._start_time is None
    assert handle.end_time_ns is None
    assert handle.payload['metadata']['timestamp_source'] == 'missing'


def test_missing_times_without_request_or_chain_evidence_remain_missing() -> None:
    client = FakeLangfuse()
    trajectory = _trajectory()
    for node in trajectory.nodes.values():
        node.timeline = []

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    for node_id in trajectory.nodes:
        handle = _handle_for_node(client, node_id)
        assert handle._otel_span._start_time is None
        assert handle.end_time_ns is None
        assert handle.payload['metadata']['timestamp_source'] == 'missing'


def test_recorded_first_seen_point_is_preserved_over_derived_request_time() -> None:
    client = FakeLangfuse()
    trajectory = _trajectory()
    trajectory.nodes[0].timeline = [
        {
            'event_type': 'input_first_seen',
            'start_time': 103.0,
            'end_time': 103.0,
            'duration': 0.0,
        }
    ]

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    handle = _handle_for_node(client, 0)
    assert handle._otel_span._start_time == handle.end_time_ns == 103_000_000_000
    assert handle.payload['metadata']['timestamp_source'] == 'trajectory_tree'
    assert handle.payload['metadata']['timeline'] == trajectory.nodes[0].timeline


def test_invalid_trajectory_timestamp_fails_before_export() -> None:
    client = FakeLangfuse()
    trajectory = _trajectory()
    trajectory.nodes[2].timeline[0]['start_time'] = 107
    trajectory.nodes[2].timeline[0]['end_time'] = 106

    with pytest.raises(ValueError, match='node 2 timeline ends before it starts'):
        LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    assert client.handles == []


def test_final_export_scores_flushes_and_deduplicates_trajectory() -> None:
    client = FakeLangfuse()
    observer = LangfuseTrajectoryObserver(client=client)
    trajectory = _trajectory()

    observer.on_final(trajectory)
    observer.on_final(trajectory)

    assert len(client.trace_seeds) == 1
    assert client.scores == [
        {
            'trace_id': 'trace-1',
            'name': 'trajectory_reward',
            'value': 0.75,
            'comment': 'trajectory=run-1',
        }
    ]
    assert client.flush_count == 1


def test_observer_is_final_only_and_rejects_realtime_configuration() -> None:
    observer = LangfuseTrajectoryObserver(client=FakeLangfuse())

    observer.on_update(TrajectoryUpdate(_trajectory(), revision=1))

    assert observer.observes_updates is False
    assert observer._finalized_ids == {}
    with pytest.raises(ValueError, match='final_only=True'):
        LangfuseTrajectoryObserver(final_only=False, client=FakeLangfuse())


def test_registry_constructs_final_only_observer() -> None:
    observers = create_trajectory_observers({
        'langfuse': {
            'name': 'langfuse',
            'args': {'trace_name': 'Trajectory Trace', 'final_only': True, 'tags': ['gateway']},
        }
    })

    observer = observers['langfuse']
    assert isinstance(observer, LangfuseTrajectoryObserver)
    assert observer.config.trace_name == 'Trajectory Trace'
    assert observer.config.final_only is True
    assert observer.config.tags == ('gateway',)


def test_payload_truncation_does_not_mutate_trajectory() -> None:
    trajectory = _trajectory()
    original = copy.deepcopy(trajectory.nodes[2])

    LangfuseTrajectoryObserver(max_content_chars=32, client=FakeLangfuse()).on_final(trajectory)

    assert trajectory.nodes[2].original_message == original.original_message
    assert trajectory.nodes[2].request_kwargs == original.request_kwargs
    assert trajectory.nodes[2].request_message_overrides == original.request_message_overrides
