from __future__ import annotations

import copy

import numpy as np
import pytest

from tests.unit.gateway.test_langfuse_trajectory_observer import (
    FakeHandle,
    FakeLangfuse,
    _handle_for_chain,
    _handle_for_node,
    _node,
    _trajectory,
)

from nexus.dashboard.projection import project_trajectory
from nexus.gateway.observers.langfuse import LangfuseTrajectoryObserver
from nexus.gateway.trajectory import Chain, SegmentBoundary, TokenAccount, Trajectory, TrajectoryNode


def _segments(client: FakeLangfuse) -> dict[str, FakeHandle]:
    return {
        handle.payload['metadata']['segment_id']: handle
        for handle in client.handles
        if handle.payload.get('metadata', {}).get('object_type') == 'trajectory_segment'
    }


def _linear_trajectory() -> Trajectory:
    roles = ['user', 'assistant', 'tool', 'user', 'assistant', 'tool']
    nodes = {
        node_id: TrajectoryNode(
            node_id=node_id,
            message_id=node_id,
            parent=node_id - 1 if node_id else None,
            children=[node_id + 1] if node_id + 1 < len(roles) else [],
            generated_by='llm' if role == 'assistant' else 'client',
            original_message={'role': role, 'content': f'Node {node_id}'},
            token_account=TokenAccount(token_ids=np.full(3, node_id, dtype=np.int64)),
        )
        for node_id, role in enumerate(roles)
    }
    nodes[1].token_account = TokenAccount(token_ids=np.array([1, 2, 3], dtype=np.int64), gen_start=1, gen_len=2)
    nodes[4].token_account = TokenAccount(token_ids=np.array([4, 5, 6], dtype=np.int64), gen_start=1, gen_len=1)
    return Trajectory(
        session_id='multi-turn',
        nodes=nodes,
        roots=[0],
        chains={'main': Chain(chain_id='main', roots=[0])},
    )


def test_segment_summaries_match_dashboard_without_changing_nodes_or_usage() -> None:
    trajectory = _trajectory()
    original = copy.deepcopy(trajectory.dump())
    expected = project_trajectory(original)['segments']
    client = FakeLangfuse()
    observer = LangfuseTrajectoryObserver(client=client)

    observer.on_final(trajectory)
    observer.on_final(trajectory)

    segments = _segments(client)
    assert set(segments) == {item['segment_id'] for item in expected}
    trace = next(handle for handle in client.handles if handle.parent is None)
    assert trace.payload['metadata']['segment_ids'] == [item['segment_id'] for item in expected]
    assert trace.payload['metadata']['segment_count'] == len(segments)
    for item in expected:
        segment = segments[item['segment_id']]
        assert segment.kind == 'span'
        assert segment.name == f'Segment {item["segment_id"]}: {item["kind"]}'
        assert segment.parent is _handle_for_chain(client, item['chain_id'])
        assert segment.children == []
        assert segment.ended
        assert {key: segment.payload['metadata'][key] for key in item} == item
        assert segment.payload['input']['source_node_ids'] == item['source_node_ids']
        assert segment.payload['output']['target_node_ids'] == item['target_node_ids']
        assert segment.payload['output']['boundary'] == item['boundary']
        assert 'usage_details' not in segment.payload
        assert 'model' not in segment.payload
    for chain_id in trajectory.chains:
        metadata = _handle_for_chain(client, chain_id).payload['metadata']
        assert metadata['segment_ids'] == [item['segment_id'] for item in expected if item['chain_id'] == chain_id]
        assert metadata['segment_count'] == len(metadata['segment_ids'])
    targets = {node_id: item['segment_id'] for item in expected for node_id in item['target_node_ids']}
    for node_id, node in trajectory.nodes.items():
        handle = _handle_for_node(client, node_id)
        assert handle.parent is _handle_for_chain(client, trajectory.node_chain(node_id))
        assert handle.payload['metadata'].get('segment_id') == targets.get(node_id)
        assert handle.payload['metadata']['parent_node_id'] == node.parent
        assert handle.payload['metadata']['child_node_ids'] == node.children
    assert len([handle for handle in client.handles if handle.kind == 'generation']) == 3
    assert len(client.scores) == client.flush_count == len(client.trace_seeds) == 1
    assert trajectory.dump() == original
    assert not trajectory.segment_training_data


@pytest.mark.parametrize(
    'boundary',
    [
        SegmentBoundary(starts_new_segment=True, kind='summary_inference', causes=('compaction',)),
        SegmentBoundary(starts_new_segment=True, kind='compaction_resume'),
        SegmentBoundary(kind='render_restart', evidence={'physical_only': True}),
    ],
)
def test_initial_segment_and_explicit_splits_use_canonical_target_membership(boundary: SegmentBoundary) -> None:
    trajectory = _trajectory()
    trajectory.nodes[2].segment_boundary = None
    trajectory.nodes[8].segment_boundary = boundary
    client = FakeLangfuse()

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    segments = _segments(client)
    initial, resumed = segments['main@2'], segments['main@8']
    assert initial.name == 'Segment main@2: conversation'
    assert initial.payload['metadata']['boundary'] is None
    assert initial.payload['metadata']['target_node_ids'] == [2]
    assert resumed.payload['metadata']['target_node_ids'] == [8]
    assert resumed.payload['metadata']['boundary'] == boundary.to_dict()
    assert resumed.payload['input']['source_node_ids'] == [0, 1, 2, 7, 8]
    assert resumed.payload['input']['context_node_ids'] == [0, 1, 2, 7]
    assert initial.payload['input']['trailing_node_ids'] == [7]
    assert _handle_for_node(client, 2).payload['metadata']['segment_id'] == 'main@2'
    assert _handle_for_node(client, 8).payload['metadata']['segment_id'] == 'main@8'
    assert _handle_for_node(client, 2).payload['metadata']['segment_anchor_node_id'] is None
    assert _handle_for_node(client, 8).payload['metadata']['segment_anchor_node_id'] == (
        8 if boundary.starts_new_segment else None
    )
    assert 'segment_id' not in _handle_for_node(client, 7).payload['metadata']
    assert _handle_for_chain(client, 'main').payload['metadata']['segment_count'] == 2


def test_shared_parent_context_keeps_its_own_target_segment() -> None:
    trajectory = _trajectory()
    trajectory.nodes[3].parent = 2
    trajectory.nodes[2].children.insert(0, 3)
    trajectory.roots.remove(3)
    client = FakeLangfuse()

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    worker = _segments(client)['worker@5']
    assert worker.payload['input']['source_node_ids'] == [0, 1, 2, 3, 4, 5, 6]
    assert worker.payload['input']['context_node_ids'] == [0, 1, 2, 3, 4]
    assert worker.payload['input']['trailing_node_ids'] == [6]
    assert worker.payload['output']['target_node_ids'] == [5]
    assert _handle_for_node(client, 2).payload['metadata']['segment_id'] == 'main@2'
    assert _handle_for_node(client, 5).payload['metadata']['segment_id'] == 'worker@5'
    assert _handle_for_node(client, 3).parent is _handle_for_chain(client, 'worker')


def test_shared_system_context_presentation_does_not_change_segment_sources() -> None:
    trajectory = _trajectory()
    trajectory.chains['main'].roots.append(3)
    trajectory.chains['worker'].roots = [4]
    trajectory.nodes[9] = _node(9, 3, 'client', {'role': 'user', 'content': 'other task'}, start_time=109, end_time=110)
    trajectory.nodes[10] = _node(10, 9, 'llm', {'role': 'assistant', 'content': 'answer'}, start_time=112, end_time=113)
    trajectory.nodes[3].children.append(9)
    trajectory.nodes[9].children.append(10)
    trajectory.chains['other'] = Chain(chain_id='other', kind='subagent', roots=[9], parent_chain_id='main')
    original = copy.deepcopy(trajectory.dump())
    client = FakeLangfuse()

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    segments = _segments(client)
    assert segments['worker@5'].payload['input']['source_node_ids'] == [3, 4, 5, 6]
    assert segments['other@10'].payload['input']['source_node_ids'] == [3, 9, 10]
    shared = [handle for handle in client.handles if handle.payload['metadata']['object_type'] == 'shared_context']
    assert len(shared) == 2
    assert all(handle.payload['metadata']['source_node_id'] == 3 for handle in shared)
    assert all('segment_id' not in handle.payload['metadata'] for handle in shared)
    assert segments['worker@5'].parent is _handle_for_chain(client, 'worker')
    assert segments['other@10'].parent is _handle_for_chain(client, 'other')
    assert trajectory.dump() == original


def test_rejected_sibling_remains_visible_without_a_canonical_segment_id() -> None:
    trajectory = _trajectory()
    trajectory.nodes[9] = _node(9, 7, 'llm', {'role': 'assistant', 'content': 'rejected'}, start_time=117, end_time=119)
    trajectory.nodes[7].children.append(9)
    trajectory.nodes[9].selection = 'rejected_sibling'
    trajectory.nodes[8].selection = 'selected'
    client = FakeLangfuse()

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    for segment in _segments(client).values():
        assert 9 not in segment.payload['metadata']['source_node_ids']
    rejected = _handle_for_node(client, 9)
    assert rejected.kind == 'generation'
    assert 'segment_id' not in rejected.payload['metadata']
    assert rejected.parent is _handle_for_chain(client, 'main')


def test_segment_time_extent_excludes_shared_context_and_trailing_observations() -> None:
    client = FakeLangfuse()
    LangfuseTrajectoryObserver(client=client).on_final(_trajectory())

    segments = _segments(client)
    assert segments['main@2']._otel_span._start_time == 104_000_000_000
    assert segments['main@2'].end_time_ns == 120_000_000_000
    assert segments['worker@5']._otel_span._start_time == 111_000_000_000
    assert segments['worker@5'].end_time_ns == 113_000_000_000
    assert segments['worker@5'].payload['metadata']['timestamps_complete'] is True
    assert (
        segments['worker@5'].payload['metadata']['time_semantics'] == 'target_llm_extent_not_segment_execution_duration'
    )


@pytest.mark.parametrize('missing_nodes', [(2,), (2, 8)])
def test_missing_target_times_are_marked_incomplete_not_invented(missing_nodes: tuple[int, ...]) -> None:
    trajectory = _trajectory()
    for node_id in missing_nodes:
        trajectory.nodes[node_id].timeline = []
    client = FakeLangfuse()

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    segment = _segments(client)['main@2']
    metadata = segment.payload['metadata']
    assert metadata['timestamps_complete'] is False
    if len(missing_nodes) == 1:
        assert segment._otel_span._start_time == 117_000_000_000
        assert segment.end_time_ns == 120_000_000_000
        assert metadata['timestamp_source'] == 'derived_target_node_bounds'
    else:
        assert segment._otel_span._start_time is segment.end_time_ns is None
        assert metadata['timestamp_source'] == 'missing'
        assert metadata['export_timestamp_semantics'] == 'sdk_default_not_execution_time'


def test_segment_tokens_match_dashboard_and_training_without_counting_tail() -> None:
    trajectory = _linear_trajectory()
    trajectory.nodes[5].token_account = None
    (canonical,) = trajectory.to_segment_node_lists()
    header = trajectory.prepare_segment_training_data([(canonical, True)])[canonical.segment_id]
    expected = project_trajectory(trajectory.dump())['segments'][0]['tokens']
    client = FakeLangfuse()

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    segment = _segments(client)['main@1']
    assert (
        segment.payload['metadata']['tokens']
        == expected
        == {
            'prompt': header.prompt_length,
            'response': header.response_length,
            'generated': sum(stop - start for start, stop in header.generated_ranges),
        }
        == {'prompt': 4, 'response': 11, 'generated': 3}
    )
    assert segment.payload['output']['tokens'] == expected
    assert segment.payload['metadata']['tokens_available'] is True
    assert 'usage_details' not in segment.payload
    assert not trajectory.segment_training_data


@pytest.mark.parametrize('node_id', [0, 1, 2, 4])
@pytest.mark.parametrize('missing_account', [False, True])
def test_missing_segment_token_data_is_unknown_not_zero(node_id: int, missing_account: bool) -> None:
    trajectory = _linear_trajectory()
    trajectory.nodes[node_id].token_account = None if missing_account else TokenAccount()
    client = FakeLangfuse()

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    segment = _segments(client)['main@1']
    assert segment.payload['metadata']['tokens'] is None
    assert segment.payload['metadata']['tokens_available'] is False
    assert segment.payload['output']['tokens'] is None


def test_known_empty_segment_tokens_remain_zero() -> None:
    trajectory = _linear_trajectory()
    for node in trajectory.nodes.values():
        node.token_account = TokenAccount(token_ids=np.empty(0, dtype=np.int64))
    client = FakeLangfuse()

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    segment = _segments(client)['main@1']
    assert segment.payload['metadata']['tokens'] == {'prompt': 0, 'response': 0, 'generated': 0}
    assert segment.payload['metadata']['tokens_available'] is True


@pytest.mark.parametrize('empty', [False, True])
def test_empty_or_undeclared_trajectories_do_not_invent_segments(empty: bool) -> None:
    trajectory = Trajectory(session_id='empty', state='failed') if empty else _trajectory()
    trajectory.chains = {}
    client = FakeLangfuse()

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    assert _segments(client) == {}
    trace = next(handle for handle in client.handles if handle.parent is None)
    assert trace.payload['metadata']['segment_count'] == 0
    assert trace.payload['metadata']['segment_ids'] == []
    assert len(client.handles) == 1 + len(trajectory.nodes)
    for node_id in trajectory.nodes:
        assert 'segment_id' not in _handle_for_node(client, node_id).payload['metadata']
