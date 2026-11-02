import copy
import shutil
import subprocess

from pathlib import Path

import numpy as np
import pytest

from tests.unit.dashboard.trajectory_fixtures import branched_trajectory

from nexus.dashboard.projection import project_trajectory
from nexus.gateway.trajectory import Chain, SegmentBoundary, TokenAccount, Trajectory, TrajectoryNode


def test_segments_javascript():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js is not installed; segment JavaScript tests require Node.js')
    result = subprocess.run(
        [node, '--test', str(Path(__file__).with_name('test_segments.cjs'))],
        capture_output=True,
        text=True,
        encoding='utf-8',
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f'Segment JavaScript tests failed:\n{result.stdout}\n{result.stderr}'


def compaction_trajectory():
    roles = [
        'system',
        'system',
        'user',
        'user',
        'assistant',
        'tool',
        'user',
        'assistant',
        'user',
        'assistant',
        'tool',
        'user',
        'assistant',
        'user',
        'assistant',
    ]
    parents = [None, 0, 1, 2, 3, 4, 5, 6, 3, 8, 9, 10, 11, 3, 13]
    nodes = {}
    for node_id, (role, parent) in enumerate(zip(roles, parents, strict=True)):
        nodes[node_id] = TrajectoryNode(
            node_id=node_id,
            message_id=node_id,
            parent=parent,
            children=[],
            generated_by='llm' if role == 'assistant' else 'client',
            original_message={'role': role, 'content': f'Node {node_id}'},
        )
        if parent is not None:
            nodes[parent].children.append(node_id)
    for node_id, kind in [
        (7, 'summary_inference'),
        (9, 'compaction_resume'),
        (12, 'summary_inference'),
        (14, 'compaction_resume'),
    ]:
        nodes[node_id].segment_boundary = SegmentBoundary(
            starts_new_segment=True,
            kind=kind,
            causes=('render_context_discontinuity', 'compaction'),
        )
    return Trajectory(
        session_id='two-compactions',
        state='completed',
        nodes=nodes,
        roots=[0],
        chains={'main': Chain(chain_id='main', roots=[0])},
    )


def test_segments_show_five_compositions_without_training_history_twice():
    trajectory = compaction_trajectory()
    data = trajectory.dump()
    original = copy.deepcopy(data)
    segments = project_trajectory(data)['segments']
    assert [segment['segment_id'] for segment in segments] == ['main@4', 'main@7', 'main@9', 'main@12', 'main@14']
    assert [segment['source_node_ids'] for segment in segments] == [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 8, 9, 10],
        [0, 1, 2, 3, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 13, 14],
    ]
    assert [segment['target_node_ids'] for segment in segments] == [[4], [7], [9], [12], [14]]
    assert [segment['first_response_node_index'] for segment in segments] == [4, 7, 5, 8, 5]
    assert segments[0]['kind'] == 'conversation'
    assert segments[0]['boundary'] is None
    assert segments[1]['boundary'] == data['nodes'][7]['segment_boundary']
    assert all(segment['tokens'] is None for segment in segments)
    assert data == original
    assert not trajectory.segment_training_data


def linear_trajectory():
    roles = ['user', 'assistant', 'tool', 'user', 'assistant', 'tool']
    nodes = {}
    for node_id, role in enumerate(roles):
        nodes[node_id] = TrajectoryNode(
            node_id=node_id,
            message_id=node_id,
            parent=node_id - 1 if node_id else None,
            children=[node_id + 1] if node_id + 1 < len(roles) else [],
            generated_by='llm' if role == 'assistant' else 'client',
            original_message={'role': role, 'content': f'Node {node_id}'},
            token_account=TokenAccount(token_ids=np.array([node_id] * 3, dtype=np.int32)),
        )
    nodes[1].token_account = TokenAccount(token_ids=np.array([1, 2, 3], dtype=np.int32), gen_start=1, gen_len=2)
    nodes[4].token_account = TokenAccount(token_ids=np.array([4, 5, 6], dtype=np.int32), gen_start=1, gen_len=1)
    return Trajectory(
        session_id='multi-turn',
        state='completed',
        nodes=nodes,
        roots=[0],
        chains={'main': Chain(chain_id='main', roots=[0])},
    )


def test_multi_turn_tokens_match_training_header_and_exclude_trailing_observation():
    trajectory = linear_trajectory()
    (node_list,) = trajectory.to_segment_node_lists()
    header = trajectory.prepare_segment_training_data([(node_list, True)])[node_list.segment_id]
    (segment,) = project_trajectory(trajectory.dump())['segments']
    assert segment['source_node_ids'] == [0, 1, 2, 3, 4, 5]
    assert segment['target_node_ids'] == [1, 4]
    assert (
        segment['tokens']
        == {
            'prompt': header.prompt_length,
            'response': header.response_length,
            'generated': sum(stop - start for start, stop in header.generated_ranges),
        }
        == {'prompt': 4, 'response': 11, 'generated': 3}
    )
    trajectory.nodes[5].token_account = None
    assert project_trajectory(trajectory.dump())['segments'][0]['tokens'] == segment['tokens']


@pytest.mark.parametrize('node_id', [0, 1, 2, 4])
@pytest.mark.parametrize('missing_account', [False, True])
def test_incomplete_token_data_is_unknown_not_zero(node_id, missing_account):
    trajectory = linear_trajectory()
    trajectory.nodes[node_id].token_account = None if missing_account else TokenAccount()
    assert project_trajectory(trajectory.dump())['segments'][0]['tokens'] is None


def test_known_empty_tokens_are_zero_not_unknown():
    trajectory = linear_trajectory()
    for node in trajectory.nodes.values():
        node.token_account = TokenAccount(token_ids=np.array([], dtype=np.int32))
    assert project_trajectory(trajectory.dump())['segments'][0]['tokens'] == {
        'prompt': 0,
        'response': 0,
        'generated': 0,
    }


def test_physical_only_boundary_splits_even_without_semantic_start():
    trajectory = linear_trajectory()
    trajectory.nodes[4].segment_boundary = SegmentBoundary(
        kind='render_restart',
        evidence={'physical_only': True},
    )
    segments = project_trajectory(trajectory.dump())['segments']
    assert [segment['target_node_ids'] for segment in segments] == [[1], [4]]
    assert segments[1]['kind'] == 'render_restart'


def test_multiple_chains_keep_canonical_order_parent_context_and_reject_siblings():
    trajectory = branched_trajectory()
    segments = project_trajectory(trajectory.dump())['segments']
    canonical = trajectory.to_segment_node_lists()
    assert [segment['segment_id'] for segment in segments] == [s.segment_id for s in canonical]
    for segment, node_list in zip(segments, canonical, strict=True):
        assert segment['source_node_ids'] == [node.node_id for node in node_list.nodes]
        assert segment['target_node_ids'] == [node.node_id for node in node_list.response_nodes]
    assert all(15 not in segment['source_node_ids'] for segment in segments)
    child = next(segment for segment in segments if segment['chain_id'] == 'research')
    assert child['source_node_ids'][:3] == [0, 1, 2]
    assert 2 not in child['target_node_ids']
    targets = [node_id for segment in segments for node_id in segment['target_node_ids']]
    assert len(targets) == len(set(targets))


def test_no_declared_chains_does_not_invent_segments():
    trajectory = linear_trajectory()
    trajectory.chains = {}
    assert project_trajectory(trajectory.dump())['segments'] == []


def test_empty_trajectory_has_no_segments():
    assert project_trajectory(Trajectory(session_id='empty', state='failed').dump())['segments'] == []
