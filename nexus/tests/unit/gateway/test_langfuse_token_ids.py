from __future__ import annotations

import copy

import numpy as np
import pytest

from tests.unit.gateway.test_langfuse_segments import _linear_trajectory, _segments
from tests.unit.gateway.test_langfuse_trajectory_observer import (
    FakeLangfuse,
    _handle_for_node,
    _node,
    _trajectory,
)

from nexus.gateway.observers.langfuse import LangfuseTrajectoryObserver
from nexus.gateway.trajectory import Chain, SegmentBoundary, TokenAccount, Trajectory
from nexus.gateway.trajectory_store import NodePayloadRef


@pytest.mark.parametrize('node_id', [0, 1, 2, 4, 5])
@pytest.mark.parametrize('dtype', ['int32', 'int64'])
def test_node_exports_recorded_ids_and_generated_range_without_mutation(node_id: int, dtype: str) -> None:
    trajectory = _linear_trajectory()
    account = trajectory.nodes[node_id].token_account
    assert account is not None
    assert account.token_ids is not None
    account.token_ids = account.token_ids.astype(dtype)
    account.token_ids.setflags(write=False)
    account.raw_token_ids = np.array([999], dtype=np.int64)
    account.logprobs = np.full(account.gen_len, -0.1, dtype=np.float64)
    original = copy.deepcopy(trajectory.dump())
    client = FakeLangfuse()

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    exported = _handle_for_node(client, node_id).payload['metadata']['token_account']
    assert exported == {
        'token_ids': account.token_ids.tolist(),
        'gen_start': account.gen_start,
        'gen_len': account.gen_len,
        'token_count': 3,
        'trailing_token_count': 3 - account.gen_start - account.gen_len,
        'has_routed_experts': False,
        'has_payload_ref': False,
        'num_images': account.num_images,
    }
    assert (
        exported['token_ids'][account.gen_start : account.gen_start + account.gen_len] == account.generated_ids.tolist()
    )
    assert trajectory.dump() == original
    assert account.raw_token_ids.tolist() == [999]
    exported['token_ids'].append(123)
    assert trajectory.dump() == original


@pytest.mark.parametrize('has_payload_ref', [False, True])
def test_missing_ids_remain_unknown_without_raw_fallback_or_store_fetch(has_payload_ref: bool) -> None:
    trajectory = _linear_trajectory()
    trajectory.nodes[1].token_account = TokenAccount(
        payload_ref=NodePayloadRef() if has_payload_ref else None,
        raw_token_ids=np.array([99, 100], dtype=np.int64),
        gen_start=2,
        gen_len=3,
    )
    client = FakeLangfuse()

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    exported = _handle_for_node(client, 1).payload['metadata']['token_account']
    assert exported['token_ids'] is None
    assert exported['token_count'] is None
    assert exported['trailing_token_count'] is None
    assert exported['gen_start'] == 2
    assert exported['gen_len'] == 3
    assert exported['has_payload_ref'] is has_payload_ref


def test_known_empty_ids_are_available_and_not_missing() -> None:
    trajectory = _linear_trajectory()
    trajectory.nodes[1].token_account = TokenAccount(token_ids=np.empty(0, dtype=np.int64))
    client = FakeLangfuse()

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    exported = _handle_for_node(client, 1).payload['metadata']['token_account']
    assert exported['token_ids'] == []
    assert exported['token_count'] == exported['trailing_token_count'] == 0


def test_message_only_node_does_not_invent_a_token_account() -> None:
    trajectory = _linear_trajectory()
    trajectory.nodes[1].token_account = None
    client = FakeLangfuse()

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    assert 'token_account' not in _handle_for_node(client, 1).payload['metadata']


def test_large_ids_survive_dump_reload_and_default_export_in_full() -> None:
    trajectory = _linear_trajectory()
    trajectory.nodes[1].token_account = TokenAccount(
        token_ids=np.arange(20_000, dtype=np.int64), gen_start=3, gen_len=19_990
    )
    reloaded = Trajectory.from_dump(trajectory.dump())
    client = FakeLangfuse()

    LangfuseTrajectoryObserver(client=client).on_final(reloaded)

    exported = _handle_for_node(client, 1).payload['metadata']['token_account']
    assert exported['token_ids'] == list(range(20_000))
    assert exported['token_count'] == 20_000
    assert exported['gen_start'] == 3
    assert exported['gen_len'] == 19_990
    assert exported['trailing_token_count'] == 7


def test_explicit_content_limit_still_marks_truncated_metadata() -> None:
    trajectory = _linear_trajectory()
    trajectory.nodes[1].token_account = TokenAccount(token_ids=np.arange(2000, dtype=np.int64))
    original = copy.deepcopy(trajectory.dump())
    client = FakeLangfuse()

    LangfuseTrajectoryObserver(max_content_chars=4000, client=client).on_final(trajectory)

    handle = next(handle for handle in client.handles if handle.kind == 'generation')
    assert len(handle.payload['metadata']['truncated_json']) == 4000
    assert handle.payload['metadata']['original_chars'] > 4000
    assert trajectory.dump() == original


def test_segments_reference_node_tokens_without_copying_shared_context_arrays() -> None:
    trajectory = _linear_trajectory()
    trajectory.nodes[4].segment_boundary = SegmentBoundary(starts_new_segment=True, kind='summary_inference')
    client = FakeLangfuse()

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    segments = _segments(client)
    assert segments['main@1'].payload['output']['target_node_ids'] == [1]
    assert segments['main@4'].payload['output']['target_node_ids'] == [4]
    assert 1 in segments['main@4'].payload['input']['context_node_ids']
    for segment in segments.values():
        for part in ('input', 'output', 'metadata'):
            assert 'token_ids' not in segment.payload[part]
    for node_id, node in trajectory.nodes.items():
        assert node.token_account is not None
        assert node.token_account.token_ids is not None
        assert _handle_for_node(client, node_id).payload['metadata']['token_account']['token_ids'] == (
            node.token_account.token_ids.tolist()
        )
    assert not trajectory.segment_training_data


def test_hidden_shared_system_node_keeps_ids_on_context_references() -> None:
    trajectory = _trajectory()
    trajectory.chains['main'].roots.append(3)
    trajectory.chains['worker'].roots = [4]
    trajectory.nodes[3].token_account = TokenAccount(token_ids=np.array([101, 102], dtype=np.int64))
    trajectory.nodes[9] = _node(9, 3, 'client', {'role': 'user', 'content': 'other task'}, start_time=109, end_time=110)
    trajectory.nodes[10] = _node(10, 9, 'llm', {'role': 'assistant', 'content': 'answer'}, start_time=112, end_time=113)
    trajectory.nodes[3].children.append(9)
    trajectory.nodes[9].children.append(10)
    trajectory.chains['other'] = Chain(chain_id='other', kind='subagent', roots=[9], parent_chain_id='main')
    client = FakeLangfuse()

    LangfuseTrajectoryObserver(client=client).on_final(trajectory)

    assert not any(handle.payload['metadata'].get('node_id') == 3 for handle in client.handles)
    shared = [handle for handle in client.handles if handle.payload['metadata']['object_type'] == 'shared_context']
    assert len(shared) == 2
    for handle in shared:
        assert handle.payload['metadata']['source_node_id'] == 3
        assert handle.payload['metadata']['token_account']['token_ids'] == [101, 102]
        assert handle.payload['metadata']['token_account']['gen_len'] == 0
        assert 'usage_details' not in handle.payload
