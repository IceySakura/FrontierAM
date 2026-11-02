"""Canonical Trajectory contract and direct-freeze behavior.

Covers the unified per-node TokenAccount, path-concat reconstruction,
assembly-owned token attribution, and render forks.
"""

import copy
import json
import pickle

import numpy as np
import pytest

from nexus.gateway.trajectory import (
    Chain,
    SegmentBoundary,
    SegmentNodeList,
    SegmentTrainingData,
    TokenAccount,
    Trajectory,
    TrajectoryNode,
)
from nexus.gateway.trajectory_assembler import TrajectoryAssembler
from nexus.gateway.trajectory_store import InMemoryTrajectoryRecordStore, NodePayloadRecord, NodePayloadRef
from nexus.gateway.types import PromptRender


SYSTEM = {'role': 'system', 'content': 'sys'}
USER = {'role': 'user', 'content': 'question'}


def _training(raw_prompt_ids, response_ids, logprobs=None):
    training = {
        'prompt_ids': list(raw_prompt_ids),
        'raw_prompt_ids': list(raw_prompt_ids),
        'response_ids': list(response_ids),
    }
    if logprobs is not None:
        training['response_logprobs'] = list(logprobs)
    return training


def _publish_segment_training_data(trajectory, **updates):
    trajectory.segment_training_data = trajectory.prepare_segment_training_data(**updates)


def _materialize(assembler: TrajectoryAssembler, session_id: str = 'traj-test'):
    return assembler.freeze(session_id=session_id, state='completed')


def _node_with_generated_tokens(gen_len: int = 3) -> TrajectoryNode:
    return TrajectoryNode(
        node_id=7,
        message_id=11,
        parent=None,
        generated_by='llm',
        children=[],
        token_account=TokenAccount(token_ids=np.array(range(gen_len), dtype=np.int32), gen_len=gen_len),
    )


def _segment_kind(node_list: SegmentNodeList) -> str:
    boundary = node_list.response_nodes[0].segment_boundary
    return boundary.kind if boundary is not None and boundary.kind is not None else 'conversation'


class ByteTok:
    """1 byte == 1 token; ASCII fixtures keep spans human-checkable."""

    def encode(self, text, add_special_tokens=False):
        return list(text.encode('utf-8'))

    def decode(self, ids):
        return bytes(ids).decode('utf-8', errors='ignore')


def _enc(text: str) -> list[int]:
    return list(text.encode('utf-8'))


def _render(messages: list[str], generation_prompt: str) -> PromptRender:
    return _render_ids(
        list(zip(messages, (_enc(message) for message in messages), strict=True)),
        (generation_prompt, _enc(generation_prompt)),
    )


def _render_ids(
    messages: list[tuple[str, list[int]]],
    generation_prompt: tuple[str, list[int]],
) -> PromptRender:
    texts = [text for text, _ in messages]
    message_ids = [list(token_ids) for _, token_ids in messages]
    generation_text, generation_ids = generation_prompt
    return PromptRender(
        text=''.join(texts) + generation_text,
        token_ids=np.array(
            [token for token_ids in message_ids for token in token_ids] + list(generation_ids), dtype=np.int32
        ),
        templated_messages=texts,
        message_token_ids=[np.array(part, dtype=np.int32) for part in message_ids],
        generation_prompt=generation_text,
        generation_prompt_ids=np.array(generation_ids, dtype=np.int32),
    )


def _record_tokens(
    assembler: TrajectoryAssembler,
    *,
    messages: list[dict],
    output_message: dict,
    message_spans: list[tuple[str, list[int]]],
    generation_prompt: tuple[str, list[int]],
    response_ids: list[int],
    completion_text: str,
    logprobs: list[float] | None = None,
    **kwargs,
) -> None:
    render = _render_ids(message_spans, generation_prompt)
    assembler.record_completion(
        messages=messages,
        output_message=output_message,
        token_training=_training(render.token_ids, response_ids, logprobs=logprobs),
        completion_text=completion_text,
        prompt_render=render,
        **kwargs,
    )


def _linear_token_session() -> TrajectoryAssembler:
    """3-turn linear tool conversation with a growing token ledger."""
    assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
    tools = [{'type': 'function', 'function': {'name': 'calc', 'parameters': {}}}]

    turn1 = {
        'role': 'assistant',
        'content': 'use tool',
        'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {'name': 'calc', 'arguments': '{"a": 1}'}}],
    }
    _record_tokens(
        assembler,
        messages=[SYSTEM, USER],
        output_message=turn1,
        message_spans=[('S', [10]), ('U', [11])],
        generation_prompt=('A', [12]),
        response_ids=[20, 21],
        completion_text='R1',
        request_kwargs={'tools': tools},
        logprobs=[-0.1, -0.2],
        response_meta={
            'model': 'm1',
            'response_id': 'r1',
            'usage': {'prompt_tokens': 3},
            'inference_time': 1.25,
            'wall_inference_time': 2.0,
            'abort_wait_time': 0.25,
            'num_preempted': 2,
        },
    )
    # production shape: the tool result arrives via the resent history
    # (record_observation has no production callers)
    tool_msg = {'role': 'tool', 'content': '2', 'tool_call_id': 'c1'}

    turn2 = {'role': 'assistant', 'content': 'answer is 2'}
    _record_tokens(
        assembler,
        messages=[SYSTEM, USER, turn1, tool_msg],
        output_message=turn2,
        message_spans=[('S', [10]), ('U', [11]), ('AR1', [12, 20, 21]), ('T', [30])],
        generation_prompt=('A', [40]),
        response_ids=[50, 51, 52],
        completion_text='R2',
        request_kwargs={'tools': tools},
        logprobs=[-0.3, -0.4, -0.5],
        response_meta={'model': 'm1', 'response_id': 'r2', 'usage': {'prompt_tokens': 7}},
    )
    return assembler


def _compaction_token_session(k: int = 2) -> TrajectoryAssembler:
    """k replacement-style compactions -> k+1 segments."""
    assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
    _record_tokens(
        assembler,
        messages=[SYSTEM, USER],
        output_message={'role': 'assistant', 'content': 'turn-0'},
        message_spans=[('S', [10]), ('U', [11])],
        generation_prompt=('', []),
        response_ids=[12, 13],
        completion_text='turn-0',
    )
    for i in range(1, k + 1):
        summary = {'role': 'user', 'content': f'summary {i}', '_is_compacted_summary': True}
        reply = {'role': 'assistant', 'content': f'resume {i}'}
        _record_tokens(
            assembler,
            messages=[SYSTEM, USER, summary],
            output_message=reply,
            message_spans=[('S', [10]), ('U', [11]), (f'C{i}', [100 + i])],
            generation_prompt=('', []),
            response_ids=[110 + i, 120 + i],
            completion_text=f'resume {i}',
        )
    return assembler


class TestLinearTokenSession:
    def test_token_ownership_and_reconstruction(self):
        assembler = _linear_token_session()
        trajectory = _materialize(assembler)

        llm_nodes = [n for n in trajectory.iter_nodes() if n.generated_by == 'llm']
        assert len(llm_nodes) == 2
        first, second = llm_nodes
        assert first.token_account.token_ids.tolist() == [12, 20, 21]
        assert first.token_account.gen_len == 2
        assert first.token_account.glue_ids.tolist() == [12]
        assert first.token_account.generated_ids.tolist() == [20, 21]
        assert second.token_account.token_ids.tolist() == [40, 50, 51, 52]
        assert second.token_account.generated_ids.tolist() == [50, 51, 52]
        input_nodes = [node for node in trajectory.iter_nodes() if node.generated_by != 'llm']
        assert [node.token_account.token_ids.tolist() for node in input_nodes] == [[10], [11], [30]]
        node_list = trajectory.to_segment_node_lists(chains='main')[0]
        assert [node.node_id for node in node_list.response_nodes] == [2, 4]

    def test_semantic_annotations_and_facts(self):
        trajectory = _materialize(_linear_token_session())
        llm = [n for n in trajectory.iter_nodes() if n.generated_by == 'llm']
        assert llm[0].provider_facts.model == 'm1'
        assert llm[0].provider_facts.usage == {'prompt_tokens': 3}
        assert llm[0].provider_facts.inference_time == 1.25
        assert llm[0].provider_facts.wall_inference_time == 2.0
        assert llm[0].provider_facts.abort_wait_time == 0.25
        assert llm[0].provider_facts.num_preempted == 2
        assert llm[0].request_kwargs is not None
        assert 'tools' in llm[0].request_kwargs
        assert llm[0].request_kwargs == llm[1].request_kwargs
        assert llm[0].request_kwargs is not llm[1].request_kwargs
        assert llm[0].selection is None  # no multi-response sibling group
        assert trajectory.node_chain(llm[1].node_id) == 'main'
        assert list(trajectory.chains) == ['main']
        # session timeline events are node-anchored; none fed here
        assert trajectory.timeline == []

    def test_request_kwargs_preserve_null_absent_and_provider_extensions(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')
        assembler.record_completion(
            messages=[USER],
            output_message={'role': 'assistant', 'content': 'done'},
            request_kwargs={
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'provider_extension': {'enabled': False},
            },
        )

        node = next(node for node in _materialize(assembler).iter_nodes() if node.generated_by == 'llm')

        assert node.request_kwargs == {
            'tools': None,
            'tool_choice': None,
            'response_format': None,
            'provider_extension': {'enabled': False},
        }

    def test_roundtrips_preserve_canonical_composition(self):
        trajectory = _materialize(_linear_token_session())
        expected = [
            (node_list.segment_id, tuple(node.node_id for node in node_list.nodes))
            for node_list in trajectory.to_segment_node_lists()
        ]

        pickled = pickle.loads(pickle.dumps(trajectory))
        dumped = type(trajectory).from_dump(trajectory.dump())
        for restored in (pickled, dumped):
            assert [
                (node_list.segment_id, tuple(node.node_id for node in node_list.nodes))
                for node_list in restored.to_segment_node_lists()
            ] == expected
        llm = [n for n in dumped.iter_nodes() if n.generated_by == 'llm']
        assert llm[0].request_kwargs == llm[1].request_kwargs
        assert llm[0].request_kwargs is not llm[1].request_kwargs

    def test_dump_schema_version(self):
        trajectory = _materialize(_linear_token_session())
        data = trajectory.dump()
        assert data['schema_version'] == 'gateway-trajectory-v1'
        assert 'latest_messages' not in data
        assert 'kwargs_pool' not in data
        assert 'resolved_leaves' not in data
        assert not hasattr(trajectory, 'resolved_leaves')
        llm_dump = [n for n in data['nodes'] if n['generated_by'] == 'llm']
        assert llm_dump[0]['request_kwargs'] == llm_dump[1]['request_kwargs']
        assert llm_dump[0]['token_account']['token_ids']['__nexus_gateway_type__'] == 'ndarray'
        assert llm_dump[0]['token_account']['gen_start'] == 1
        assert llm_dump[0]['token_account']['gen_len'] == 2
        assert 'post_observation_token_ids' not in llm_dump[0]['token_account']
        restored = type(trajectory).from_dump(data)
        legacy_data = {**data, 'latest_messages': [{'role': 'user', 'content': 'legacy cache'}]}
        legacy_restored = type(trajectory).from_dump(legacy_data)
        assert not hasattr(legacy_restored, 'latest_messages')
        account = restored.nodes[llm_dump[0]['node_id']].token_account
        assert account is not None
        assert account.token_ids.tolist() == [12, 20, 21]
        assert not hasattr(account, 'post_observation_token_ids')
        data['resolved_leaves'] = [{'node_id': 9999}]
        assert [item.segment_id for item in type(trajectory).from_dump(data).to_segment_node_lists()] == [
            item.segment_id for item in trajectory.to_segment_node_lists()
        ]
        llm_dump[0]['token_account']['post_observation_token_ids'] = [999]
        assert [item.segment_id for item in type(trajectory).from_dump(data).to_segment_node_lists()] == [
            item.segment_id for item in trajectory.to_segment_node_lists()
        ]
        data['schema_version'] = 'gateway-trajectory-v0'
        with pytest.raises(ValueError, match='schema_version'):
            type(trajectory).from_dump(data)

    def test_from_dump_reads_the_previous_pooled_v1_form(self):
        trajectory = _materialize(_linear_token_session())
        data = trajectory.dump()
        llm_items = [item for item in data['nodes'] if 'request_kwargs' in item]
        data['kwargs_pool'] = [llm_items[0].pop('request_kwargs')]
        for item in llm_items:
            item.pop('request_kwargs', None)
            item['request_kwargs_ref'] = 0

        restored = type(trajectory).from_dump(data)
        restored_llm = [node for node in restored.iter_nodes() if node.request_kwargs is not None]

        assert restored_llm[0].request_kwargs == restored_llm[1].request_kwargs
        assert restored_llm[0].request_kwargs is not restored_llm[1].request_kwargs


class TestContractValidation:
    """Fail-loud guards on malformed accounts and persisted data."""

    def test_children_are_materialized_and_round_trip_in_assembly_order(self):
        trajectory = _materialize(_linear_token_session())

        assert trajectory.nodes[0].children == [1]
        assert trajectory.nodes[4].children == []

        dumped = trajectory.dump()
        assert dumped['nodes'][0]['children'] == [1]
        restored = type(trajectory).from_dump(dumped)
        assert restored.nodes[0].children == [1]

    def test_token_account_rejects_invalid_generated_range(self):
        with pytest.raises(ValueError, match='gen_len'):
            TokenAccount(token_ids=np.array([1, 2], dtype=np.int32), gen_len=3)
        with pytest.raises(ValueError, match='gen_start'):
            TokenAccount(token_ids=np.array([1, 2], dtype=np.int32), gen_start=3)
        with pytest.raises(ValueError, match=r'gen_start 2 \+ gen_len 2'):
            TokenAccount(token_ids=np.array([1, 2, 3], dtype=np.int32), gen_start=2, gen_len=2)

    def test_token_account_retains_range_after_hot_cache_release(self):
        account = TokenAccount(token_ids=None, gen_start=2, gen_len=3)

        assert (account.gen_start, account.gen_len) == (2, 3)

    @pytest.mark.parametrize(
        'field_name', ['token_ids', 'logprobs', 'topk_indices', 'topk_values', 'routed_experts', 'multi_modal_items']
    )
    def test_token_account_detects_empty_store_payload_without_mutation(self, field_name: str) -> None:
        ref = NodePayloadRef()
        account = TokenAccount(
            payload_ref=ref,
            raw_token_ids=np.array([7, 8], dtype=np.int32),
            gen_start=1,
            gen_len=2,
            num_images=1,
            has_routed_experts=True,
        )
        assert not account.has_inline_store_payload, 'Refs and compact metadata are not inline Store payloads'
        value = {} if field_name == 'multi_modal_items' else []
        setattr(account, field_name, value)
        original = pickle.dumps(account)

        assert account.has_inline_store_payload, 'Even an empty Store field must be discarded explicitly'
        assert pickle.dumps(account) == original, 'Checking payload state must not mutate the account'
        assert getattr(account, field_name) is value, 'Checking payload state must preserve the payload object'

        account.discard_store_payload()

        assert not account.has_inline_store_payload, 'Discarding Store fields must clear the computed payload state'
        assert account.payload_ref is ref, 'Discarding inline fields must retain the Store ref'

    @pytest.mark.parametrize('with_account', [False, True])
    def test_trajectory_without_inline_store_payload_is_read_only(self, with_account: bool) -> None:
        trajectory = Trajectory(session_id='compact')
        assert not trajectory.has_inline_store_payload, 'An empty trajectory has no inline Store payload'
        node = _node_with_generated_tokens()
        node.token_account = (
            TokenAccount(raw_token_ids=np.array([1], dtype=np.int32), gen_len=3) if with_account else None
        )
        trajectory.nodes[node.node_id] = node
        original = pickle.dumps(trajectory)

        assert not trajectory.has_inline_store_payload, 'Missing accounts and metadata-only accounts are compact'
        assert pickle.dumps(trajectory) == original, 'Checking payload state must not mutate the trajectory'

    @pytest.mark.parametrize('node_id', [0, 4, 5])
    def test_trajectory_detects_inline_store_payload_in_any_node(self, node_id: int) -> None:
        trajectory = _materialize(_linear_token_session())
        _publish_segment_training_data(
            trajectory, initializations=[(node_list, True) for node_list in trajectory.to_segment_node_lists()]
        )
        trajectory.nodes[5] = TrajectoryNode(
            node_id=5, message_id=5, parent=4, generated_by='tool', children=[], token_account=TokenAccount()
        )
        trajectory.nodes[4].children.append(5)
        trajectory.discard_inline_payload()
        assert not trajectory.has_inline_store_payload, 'Discarded trajectories must contain no inline Store payload'
        account = trajectory.nodes[node_id].token_account
        assert account is not None, 'The selected node must have a token account'
        account.logprobs = []
        original = pickle.dumps(trajectory)

        assert trajectory.has_inline_store_payload, 'Check all nodes, including those outside training segments'
        assert pickle.dumps(trajectory) == original, 'Checking payload state must not mutate the trajectory'
        assert trajectory.nodes[node_id].token_account is account, (
            'Checking payload state must preserve account identity'
        )

    def test_token_account_hydrates_and_discards_store_payload_without_validation(self):
        account = TokenAccount(raw_token_ids=np.array([7, 8], dtype=np.int32), gen_start=1, gen_len=2, num_images=1)
        payload = {
            'token_ids': [10, 11, 12],
            'logprobs': [-0.1],
            'topk_indices': [[1, 2]],
            'topk_values': [[-0.2, -0.3]],
            'routed_experts': [[3, 4]],
            'multi_modal_items': {'images': [{'pixel_values': [0.5]}]},
        }

        account.hydrate_store_payload(payload)

        assert account.has_inline_store_payload, 'Hydrated Store fields must be detected'
        assert account.token_ids is payload['token_ids']
        assert account.logprobs is payload['logprobs']
        assert account.topk_indices is payload['topk_indices']
        assert account.topk_values is payload['topk_values']
        assert account.routed_experts is payload['routed_experts']
        assert account.multi_modal_items is payload['multi_modal_items']

        account.discard_store_payload()

        assert not account.has_inline_store_payload, 'Discarded Store fields must no longer be detected'
        assert account.token_ids is None
        assert account.logprobs is None
        assert account.topk_indices is None
        assert account.topk_values is None
        assert account.routed_experts is None
        assert account.multi_modal_items is None
        assert account.raw_token_ids.tolist() == [7, 8]
        assert (account.gen_start, account.gen_len, account.num_images) == (1, 2, 1)

    def test_dump_and_restore_inline_store_payloads(self):
        account = TokenAccount(
            token_ids=np.array([10, 11], dtype=np.int32),
            gen_len=2,
            has_routed_experts=True,
            logprobs=np.array([-0.1, -0.2], dtype=np.float32),
            topk_indices=np.array([[1, 2], [3, 4]], dtype=np.int32),
            topk_values=np.array([[-0.1, -0.2], [-0.3, -0.4]], dtype=np.float32),
            routed_experts=[[5], [6]],
            multi_modal_items={'images': [{'name': 'image-0'}]},
        )
        trajectory = Trajectory(
            session_id='inline-payloads',
            nodes={
                0: TrajectoryNode(
                    node_id=0,
                    message_id=0,
                    parent=None,
                    generated_by='llm',
                    children=[],
                    token_account=account,
                )
            },
            roots=[0],
            chains={'main': Chain(chain_id='main', roots=[0])},
        )

        dumped = trajectory.dump()
        assert all(
            name in dumped['nodes'][0]['token_account']
            for name in ('logprobs', 'topk_indices', 'topk_values', 'routed_experts', 'multi_modal_items')
        )
        restored = Trajectory.from_dump(json.loads(json.dumps(dumped)))
        restored_account = restored.nodes[0].token_account
        assert restored_account is not None
        np.testing.assert_array_equal(restored_account.logprobs, account.logprobs)
        np.testing.assert_array_equal(restored_account.topk_indices, account.topk_indices)
        np.testing.assert_array_equal(restored_account.topk_values, account.topk_values)
        assert restored_account.routed_experts == account.routed_experts
        assert restored_account.multi_modal_items == account.multi_modal_items

    def test_freezer_deep_copies_public_snapshot_inputs(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')
        timeline = [{'type': 'event', 'attributes': {'nested': 1}}]
        metrics = {'nested': {'value': 1}}
        extra = {'nested': {'value': 1}}

        trajectory = assembler.freeze(
            session_id='snapshot',
            session_timeline=timeline,
            metrics=metrics,
            extra=extra,
        )
        timeline[0]['attributes']['nested'] = 2
        metrics['nested']['value'] = 2
        extra['nested']['value'] = 2

        assert trajectory.timeline[0]['attributes']['nested'] == 1
        assert trajectory.metrics['nested']['value'] == 1
        assert trajectory.extra['nested']['value'] == 1

    async def test_dump_tagged_values_are_json_safe_and_lossless(self) -> None:
        trajectory = Trajectory(
            session_id='tagged',
            nodes={
                0: TrajectoryNode(
                    node_id=0,
                    message_id=0,
                    parent=None,
                    generated_by='llm',
                    children=[],
                    original_message={'role': 'assistant', 'content': 'done'},
                    request_kwargs={
                        'binary': b'\x00\xff',
                        'coordinates': (1, 2),
                        '__nexus_gateway_type__': 'wire-value',
                    },
                    token_account=TokenAccount(token_ids=np.array([10], dtype=np.int32), gen_len=1),
                )
            },
            roots=[0],
            chains={'main': Chain(chain_id='main', roots=[0])},
        )

        dumped = trajectory.dump()
        restored = Trajectory.from_dump(json.loads(json.dumps(dumped)))

        assert restored.nodes[0].request_kwargs == {
            'binary': b'\x00\xff',
            'coordinates': (1, 2),
            '__nexus_gateway_type__': 'wire-value',
        }

    async def test_payload_export_inlines_node_payloads(self) -> None:
        from nexus.gateway.trajectory import dump_trajectory_with_payloads, load_trajectory_with_payloads

        store = InMemoryTrajectoryRecordStore('export')
        [ref] = await store.put_many([NodePayloadRecord('export', 0, np.array([10], dtype=np.int32))])
        trajectory = Trajectory(
            session_id='export',
            roots=[0],
            chains={'main': Chain(chain_id='main', roots=[0])},
            nodes={
                0: TrajectoryNode(
                    node_id=0,
                    message_id=0,
                    parent=None,
                    generated_by='llm',
                    children=[],
                    token_account=TokenAccount(payload_ref=ref, gen_len=1),
                )
            },
        )

        dumped = dump_trajectory_with_payloads(trajectory, {0: (await store.get_many([ref]))[0]})
        restored, payloads = load_trajectory_with_payloads(json.loads(json.dumps(dumped)))
        assert restored.nodes[0].token_account is not None
        assert payloads == {0: {'token_ids': [10]}}

    def test_dump_keeps_media_out_of_the_artifact(self):
        """The JSON artifact is an observability view; a node's media lives in its payload
        record and is fetched by ref, so a large trajectory crosses a process boundary
        without its images.
        """
        first_url = 'data:image/png;base64,d2lyZS1maXJzdA=='
        second_url = 'data:image/png;base64,d2lyZS1zZWNvbmQ='
        first_user = {
            'role': 'user',
            'content': [{'type': 'image_url', 'image_url': {'url': first_url}}],
        }
        second_user = {
            'role': 'user',
            'content': [{'type': 'image_url', 'image_url': {'url': second_url}}],
        }
        trajectory = Trajectory(
            session_id='multimodal-pool',
            nodes={
                0: TrajectoryNode(
                    node_id=0,
                    message_id=0,
                    parent=None,
                    generated_by='client',
                    children=[1],
                    original_message=first_user,
                    token_account=TokenAccount(token_ids=np.array([1], dtype=np.int32), num_images=1),
                ),
                1: TrajectoryNode(
                    node_id=1,
                    message_id=1,
                    parent=0,
                    generated_by='llm',
                    children=[2],
                    original_message={'role': 'assistant', 'content': 'first'},
                    token_account=TokenAccount(token_ids=np.array([10], dtype=np.int32), gen_len=1),
                ),
                2: TrajectoryNode(
                    node_id=2,
                    message_id=2,
                    parent=1,
                    generated_by='client',
                    children=[3],
                    original_message=second_user,
                    token_account=TokenAccount(token_ids=np.array([2], dtype=np.int32), num_images=1),
                ),
                3: TrajectoryNode(
                    node_id=3,
                    message_id=3,
                    parent=2,
                    generated_by='llm',
                    children=[],
                    original_message={'role': 'assistant', 'content': 'second'},
                    token_account=TokenAccount(token_ids=np.array([11], dtype=np.int32), gen_len=1),
                ),
            },
            roots=[0],
            chains={'main': Chain(chain_id='main', roots=[0])},
        )

        dumped = trajectory.dump()
        serialized = json.dumps(dumped)

        restored = Trajectory.from_dump(json.loads(serialized))
        assert restored.nodes[0].original_message == first_user
        assert restored.nodes[2].original_message == second_user
        assert restored.nodes[0].token_account is not None
        assert restored.nodes[2].token_account is not None
        # The count is the one media fact the artifact keeps, so a restored trajectory can
        # still say how many images a message owned without the payload.
        assert restored.nodes[0].token_account.num_images == 1
        assert restored.nodes[1].token_account.num_images == 0

    def test_dump_rejects_the_first_unsupported_value_with_its_path(self):
        trajectory = Trajectory(session_id='invalid', extra={'nested': {'bad': object()}})

        with pytest.raises(TypeError, match=r'\$\.extra\.nested\.bad'):
            trajectory.dump()

    def test_dump_rejects_unsupported_materialized_wire_values(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')
        assembler.record_completion(
            messages=[USER],
            output_message={'role': 'assistant', 'content': 'done'},
            request_kwargs={'provider_extension': {'bad': object()}},
        )
        trajectory = _materialize(assembler)

        with pytest.raises(TypeError, match=r'\$\.nodes\[\d+\]\.request_kwargs\.provider_extension\.bad'):
            trajectory.dump()

    def test_from_dump_rejects_bad_structure(self):
        trajectory = _materialize(_linear_token_session())
        base = trajectory.dump()

        import copy as _c

        broken = _c.deepcopy(base)
        broken['nodes'][1]['parent'] = 999
        with pytest.raises(ValueError, match='parent 999'):
            type(trajectory).from_dump(broken)

        broken = _c.deepcopy(base)
        broken['nodes'][0]['children'] = []
        with pytest.raises(ValueError, match='children do not match parent links'):
            type(trajectory).from_dump(broken)

        broken = _c.deepcopy(base)
        broken['nodes'][1]['node_id'] = broken['nodes'][0]['node_id']
        with pytest.raises(ValueError, match='duplicate node id'):
            type(trajectory).from_dump(broken)

        broken = _c.deepcopy(base)
        llm_item = next(item for item in broken['nodes'] if 'request_kwargs' in item)
        broken['kwargs_pool'] = [llm_item.pop('request_kwargs')]
        llm_item['request_kwargs_ref'] = 1
        with pytest.raises(ValueError, match='request_kwargs_ref'):
            type(trajectory).from_dump(broken)

        broken = _c.deepcopy(base)
        broken['nodes'][0]['parent'] = broken['nodes'][-1]['node_id']  # root points at its own descendant
        with pytest.raises(ValueError, match='cycle|does not exist'):
            type(trajectory).from_dump(broken)

        broken = _c.deepcopy(base)
        broken['chains'].append(_c.deepcopy(broken['chains'][0]))
        with pytest.raises(ValueError, match='duplicate chain id'):
            type(trajectory).from_dump(broken)

        broken = _c.deepcopy(base)
        broken['roots'].append(broken['roots'][0])
        with pytest.raises(ValueError, match='duplicate trajectory root'):
            type(trajectory).from_dump(broken)

        broken = _c.deepcopy(base)
        broken['chains'][0]['roots'].append(broken['chains'][0]['roots'][0])
        with pytest.raises(ValueError, match='more than one chain entry'):
            type(trajectory).from_dump(broken)

    def test_from_dump_rejects_request_override_outside_ancestors(self):
        trajectory = _materialize(_linear_token_session())
        data = trajectory.dump()
        llm_item = next(item for item in data['nodes'] if item['generated_by'] == 'llm')
        llm_item['request_message_overrides'] = [
            {
                'node_id': llm_item['node_id'],
                'message': {'role': 'user', 'content': 'not an ancestor'},
            }
        ]

        with pytest.raises(ValueError, match='request overrides outside its ancestors'):
            type(trajectory).from_dump(data)

    def test_from_dump_rejects_non_json_dictionary_keys(self):
        trajectory = _materialize(_linear_token_session())
        data = trajectory.dump()
        llm_items = [n for n in data['nodes'] if 'request_kwargs' in n]
        llm_items[0]['request_kwargs'] = {1: 'x'}
        with pytest.raises(TypeError, match='dictionary keys must be strings'):
            type(trajectory).from_dump(data)

    def test_segment_node_lists_reject_unknown_chain_selector(self):
        trajectory = _materialize(_linear_token_session())
        with pytest.raises(ValueError, match="'all', 'main'"):
            trajectory.to_segment_node_lists(chains='mian')
        with pytest.raises(ValueError, match='unknown chain ids'):
            trajectory.to_segment_node_lists(chains=['nope'])


class TestSegmentTrainingData:
    def test_header_excludes_nodes_after_the_source_response(self):
        trajectory = Trajectory(
            session_id='terminal-observation',
            nodes={
                0: TrajectoryNode(
                    node_id=0,
                    message_id=0,
                    parent=None,
                    generated_by='client',
                    children=[1],
                    token_account=TokenAccount(token_ids=np.array([10, 11], dtype=np.int32), gen_len=0),
                ),
                1: TrajectoryNode(
                    node_id=1,
                    message_id=1,
                    parent=0,
                    generated_by='llm',
                    children=[2],
                    token_account=TokenAccount(token_ids=np.array([12], dtype=np.int32), gen_len=1),
                ),
                2: TrajectoryNode(
                    node_id=2,
                    message_id=2,
                    parent=1,
                    generated_by='observation',
                    children=[],
                    token_account=TokenAccount(token_ids=np.array([13, 14, 15], dtype=np.int32), gen_len=0),
                ),
            },
            roots=[0],
            chains={'main': Chain(chain_id='main', roots=[0])},
        )
        node_list = trajectory.to_segment_node_lists(chains='main')[0]

        replacement = trajectory.prepare_segment_training_data(initializations=[(node_list, True)])
        trajectory.segment_training_data = replacement

        assert [node.node_id for node in node_list.nodes] == [0, 1, 2]
        assert node_list.last_response_node is trajectory.nodes[1]
        assert replacement[node_list.segment_id].response_length == 1
        assert replacement[node_list.segment_id].generated_ranges == ((0, 1),)
        source = pickle.dumps(trajectory)
        debug = trajectory.to_debug_dict()
        assert debug['segments'] == [
            {
                'segment_id': 'main@1',
                'chain_id': 'main',
                'node_ids': [0, 1, 2],
                'first_response_node_index': 1,
                'token_ids': [10, 11, 12, 13, 14, 15],
                'token_count': 6,
            }
        ], 'Debug segments include trailing observations beyond the training response'
        assert debug['nodes'][1]['token_ids'] == [12]
        assert 'segment_training_data' not in debug
        debug['nodes'][1]['token_ids'].append(99)
        assert pickle.dumps(trajectory) == source, 'Debug token lists must not mutate live accounts'

    def test_header_uses_expanded_lengths_for_multiturn_images(self):
        pytest.importorskip('torch')
        trajectory = Trajectory(
            session_id='multiturn-images',
            nodes={
                0: TrajectoryNode(
                    node_id=0,
                    message_id=0,
                    parent=None,
                    generated_by='client',
                    children=[1],
                    token_account=TokenAccount(
                        token_ids=np.array([10, 101, 102], dtype=np.int32), raw_token_ids=np.array([10], dtype=np.int32)
                    ),
                ),
                1: TrajectoryNode(
                    node_id=1,
                    message_id=1,
                    parent=0,
                    generated_by='llm',
                    children=[2],
                    token_account=TokenAccount(token_ids=np.array([11, 12], dtype=np.int32), gen_start=1, gen_len=1),
                ),
                2: TrajectoryNode(
                    node_id=2,
                    message_id=2,
                    parent=1,
                    generated_by='observation',
                    children=[3],
                    token_account=TokenAccount(
                        token_ids=np.array([13, 103, 104, 105], dtype=np.int32),
                        raw_token_ids=np.array([13], dtype=np.int32),
                    ),
                ),
                3: TrajectoryNode(
                    node_id=3,
                    message_id=3,
                    parent=2,
                    generated_by='llm',
                    children=[],
                    token_account=TokenAccount(token_ids=np.array([14, 15], dtype=np.int32), gen_start=1, gen_len=1),
                ),
            },
            roots=[0],
            chains={'main': Chain(chain_id='main', roots=[0])},
        )
        node_list = trajectory.to_segment_node_lists(chains='main')[0]

        replacement = trajectory.prepare_segment_training_data(initializations=[(node_list, True)])
        record = replacement[node_list.segment_id]

        assert record.prompt_length == 4
        assert record.response_length == 7
        assert record.generated_ranges == ((0, 1), (6, 7))
        # The per-token mask now lives on the training side; the header carries only
        # whether this segment trains at all.
        assert record.training_enabled is True

    def test_compaction_node_lists_preserve_occurrence_order_and_response_boundary(self):
        trajectory = _materialize(_compaction_token_session(2))

        node_lists = trajectory.to_segment_node_lists(chains='main')

        assert [node_list.segment_id for node_list in node_lists] == ['main@2', 'main@4', 'main@6']
        assert [node_list.chain_id for node_list in node_lists] == ['main', 'main', 'main']
        assert [[node.node_id for node in node_list.response_nodes] for node_list in node_lists] == [[2], [4], [6]]
        assert all(
            node_list.nodes[node_list.first_response_node_index] is node_list.response_nodes[0]
            for node_list in node_lists
        )
        debug_segments = trajectory.to_debug_dict()['segments']
        assert [segment['segment_id'] for segment in debug_segments] == ['main@2', 'main@4', 'main@6']
        assert [segment['token_ids'] for segment in debug_segments] == [
            [10, 11, 12, 13],
            [10, 11, 101, 111, 121],
            [10, 11, 102, 112, 122],
        ], 'Every compacted segment exports its own complete context'

    def test_node_lists_capture_composition_without_materializing_tokens(self):
        trajectory = _materialize(_linear_token_session())
        node_list = trajectory.to_segment_node_lists(chains='main')[0]

        trajectory.segment_training_data = trajectory.prepare_segment_training_data(
            initializations=[(node_list, True)],
        )
        record = trajectory.segment_training_data[node_list.segment_id]

        assert isinstance(node_list, SegmentNodeList)
        assert set(vars(node_list)) == {'segment_id', 'chain_id', 'nodes', 'first_response_node_index'}
        assert record.node_list is node_list
        assert not hasattr(record, 'response_ids')
        assert not hasattr(record, 'response_mask')
        assert not hasattr(record, 'canonical_segment_fingerprint')
        assert record.prompt_length == 3
        assert record.response_length == 7
        assert record.generated_ranges == ((0, 2), (4, 7))
        assert [node.node_id for node in node_list.nodes] == [0, 1, 2, 3, 4]
        assert [node.node_id for node in node_list.response_nodes] == [2, 4]
        assert node_list.last_response_node is trajectory.nodes[4]

    def test_segment_training_data_preserves_the_canonical_header(self):
        trajectory = _materialize(_linear_token_session())
        node_list = trajectory.to_segment_node_lists()[0]

        _publish_segment_training_data(trajectory, initializations=[(node_list, True)])

        record = trajectory.segment_training_data[node_list.segment_id]
        assert isinstance(record, SegmentTrainingData)
        assert record.node_list is node_list
        assert (record.prompt_length, record.response_length) == (3, 7)
        assert record.generated_ranges == ((0, 2), (4, 7))
        assert record.training_enabled is True

        assert set(vars(record)) == {
            'node_list',
            'prompt_length',
            'response_length',
            'generated_ranges',
            'training_enabled',
        }

    def test_segment_node_lists_pickle_preserves_shared_node_identity_across_records(self):
        trajectory = _materialize(_compaction_token_session(2), session_id='stable-occurrences')
        node_lists = trajectory.to_segment_node_lists(chains='main')

        _publish_segment_training_data(
            trajectory,
            initializations=[(node_list, True) for node_list in node_lists],
        )
        restored = pickle.loads(pickle.dumps(trajectory))
        records = list(restored.segment_training_data.values())

        assert [[node.node_id for node in record.node_list.nodes] for record in records] == [
            [0, 1, 2],
            [0, 1, 3, 4],
            [0, 1, 5, 6],
        ]
        assert all(restored.nodes[node.node_id] is node for record in records for node in record.node_list.nodes)
        assert all(record.node_list.nodes[0] is restored.nodes[0] for record in records)
        assert all(record.node_list.nodes[1] is restored.nodes[1] for record in records)

    def test_segment_training_data_initialization_is_single_shot(self):
        trajectory = _materialize(_linear_token_session())
        node_list = trajectory.to_segment_node_lists()[0]
        _publish_segment_training_data(trajectory, initializations=[(node_list, True)])
        before = pickle.dumps(trajectory.segment_training_data)

        with pytest.raises(ValueError, match='already published'):
            trajectory.prepare_segment_training_data(initializations=[(node_list, True)])

        assert pickle.dumps(trajectory.segment_training_data) == before

    def test_segment_training_data_rejects_nodes_from_another_trajectory(self):
        trajectory = _materialize(_linear_token_session(), session_id='owner')
        foreign = _materialize(_linear_token_session(), session_id='foreign')
        foreign_node_list = foreign.to_segment_node_lists()[0]

        with pytest.raises(ValueError, match='nodes from another trajectory'):
            trajectory.prepare_segment_training_data(initializations=[(foreign_node_list, True)])

    def test_segment_training_data_rejects_active_trajectory(self):
        active = _linear_token_session().freeze(session_id='active', state='active')
        active_node_list = active.to_segment_node_lists()[0]
        with pytest.raises(ValueError, match='completed trajectory'):
            active.prepare_segment_training_data(initializations=[(active_node_list, True)])

    def test_segment_training_data_pickle_deepcopy_and_dump_contract(self):
        trajectory = _materialize(_linear_token_session())
        node_list = trajectory.to_segment_node_lists()[0]
        _publish_segment_training_data(
            trajectory,
            initializations=[(node_list, True)],
        )

        for restored in (copy.deepcopy(trajectory), pickle.loads(pickle.dumps(trajectory))):
            record = restored.segment_training_data[node_list.segment_id]
            assert record.training_enabled is True
            assert all(restored.nodes[node.node_id] is node for node in record.node_list.nodes)

        with pytest.raises(TypeError, match='segment training data.*pickle or Ray serialization'):
            trajectory.dump()
        empty = _materialize(_linear_token_session())
        assert Trajectory.from_dump(empty.dump()).segment_training_data == {}


class TestCompactionSession:
    @pytest.mark.parametrize(
        ('invocation_kind', 'expected_rule'),
        [
            ('summary_inference', 'nexus_summary_inference'),
            ('compaction_resume', 'nexus_compaction_resume_explicit'),
        ],
    )
    def test_zero_message_request_attaches_the_explicit_compaction_boundary_to_the_output(
        self, invocation_kind, expected_rule
    ):
        assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
        assembler.record_completion(
            messages=[],
            output_message={'role': 'assistant', 'content': invocation_kind},
            request_meta={'invocation_kind': invocation_kind},
        )

        trajectory = _materialize(assembler)
        output = trajectory.nodes[0]

        assert trajectory.roots == [0]
        assert output.segment_boundary == SegmentBoundary(
            starts_new_segment=True,
            kind=invocation_kind,
            causes=('compaction',),
            evidence={'boundary_rules': [expected_rule]},
        )
        assert [_segment_kind(node_list) for node_list in trajectory.to_segment_node_lists()] == [invocation_kind]

    @pytest.mark.parametrize('k', [1, 2, 3])
    def test_k_compactions_create_k_plus_one_canonical_segments(self, k):
        assembler = _compaction_token_session(k)
        trajectory = _materialize(assembler)
        node_lists = trajectory.to_segment_node_lists()
        assert [_segment_kind(node_list) for node_list in node_lists] == ['conversation'] + ['compaction_resume'] * k
        assert [len(node_list.response_nodes) for node_list in node_lists] == [1] * (k + 1)

    def test_chain_subtrees_and_boundaries(self):
        trajectory = _materialize(_compaction_token_session(2))
        # replacement history shares the [system, user] prefix -> one forest root
        assert list(trajectory.chains) == ['main']
        assert trajectory.chains['main'].roots == trajectory.roots
        resume_nodes = [
            n
            for n in trajectory.iter_nodes()
            if n.segment_boundary is not None and n.segment_boundary.starts_new_segment
        ]
        assert len(resume_nodes) == 2
        for node in resume_nodes:
            assert node.segment_boundary.kind == 'compaction_resume'
            assert any('compaction' in cause for cause in node.segment_boundary.causes)
        resume_roots = [trajectory.nodes[node.parent] for node in resume_nodes]
        common_ancestor = resume_roots[0].parent
        assert common_ancestor is not None
        assert all(root.parent == common_ancestor for root in resume_roots)
        siblings = trajectory.nodes[common_ancestor].children
        assert all(root.node_id in siblings for root in resume_roots)
        assert any(
            trajectory.nodes[node_id].generated_by == 'llm'
            for node_id in siblings
            if node_id not in {root.node_id for root in resume_roots}
        )

    def test_summary_inference_and_resume_keep_distinct_boundary_kinds(self):
        assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
        _record_tokens(
            assembler,
            messages=[SYSTEM, USER],
            output_message={'role': 'assistant', 'content': 'turn-0'},
            message_spans=[('S', [10]), ('U', [11])],
            generation_prompt=('', []),
            response_ids=[12, 13],
            completion_text='turn-0',
        )
        _record_tokens(
            assembler,
            messages=[
                SYSTEM,
                USER,
                {'role': 'assistant', 'content': 'turn-0'},
                {'role': 'user', 'content': 'summarize now'},
            ],
            output_message={'role': 'assistant', 'content': 'the summary'},
            message_spans=[('S', [10]), ('U', [11]), ('turn-0', [12, 13]), ('Q', [20])],
            generation_prompt=('', []),
            response_ids=[30, 31],
            completion_text='the summary',
            request_meta={'invocation_kind': 'summary_inference'},
        )
        summary = {'role': 'user', 'content': 'the summary', '_is_compacted_summary': True}
        _record_tokens(
            assembler,
            messages=[SYSTEM, USER, summary],
            output_message={'role': 'assistant', 'content': 'resume'},
            message_spans=[('S', [10]), ('U', [11]), ('C', [40])],
            generation_prompt=('', []),
            response_ids=[50, 51],
            completion_text='resume',
            request_meta={'invocation_kind': 'compaction_resume'},
        )
        trajectory = _materialize(assembler)
        node_lists = trajectory.to_segment_node_lists()
        assert [_segment_kind(node_list) for node_list in node_lists] == [
            'conversation',
            'summary_inference',
            'compaction_resume',
        ]
        boundaries = [
            node.segment_boundary.kind
            for node in trajectory.iter_nodes()
            if node.segment_boundary is not None and node.segment_boundary.starts_new_segment
        ]
        assert boundaries == ['summary_inference', 'compaction_resume']
        summary_node = next(
            node
            for node in trajectory.iter_nodes()
            if node.segment_boundary is not None and node.segment_boundary.kind == 'summary_inference'
        )
        resume_node = next(
            node
            for node in trajectory.iter_nodes()
            if node.segment_boundary is not None and node.segment_boundary.kind == 'compaction_resume'
        )
        old_turn = trajectory.nodes[trajectory.nodes[summary_node.parent].parent]
        resume_root = trajectory.nodes[resume_node.parent]
        assert old_turn.parent == resume_root.parent
        assert {old_turn.node_id, resume_root.node_id}.issubset(trajectory.nodes[old_turn.parent].children)

    def test_flattened_summary_root_preserves_call_order(self):
        assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
        _record_tokens(
            assembler,
            messages=[SYSTEM, USER],
            output_message={'role': 'assistant', 'content': 'tool call'},
            message_spans=[('S', [10]), ('U', [11])],
            generation_prompt=('', []),
            response_ids=[12, 13],
            completion_text='tool call',
        )
        _record_tokens(
            assembler,
            messages=[{'role': 'user', 'content': 'flattened tool result'}],
            output_message={'role': 'assistant', 'content': 'the summary'},
            message_spans=[('Q', [20])],
            generation_prompt=('', []),
            response_ids=[30, 31],
            completion_text='the summary',
            request_meta={'invocation_kind': 'summary_inference'},
        )
        summary = {'role': 'assistant', 'content': 'the summary', '_is_compacted_summary': True}
        _record_tokens(
            assembler,
            messages=[SYSTEM, USER, summary],
            output_message={'role': 'assistant', 'content': 'resume'},
            message_spans=[('S', [10]), ('U', [11]), ('C', [40])],
            generation_prompt=('', []),
            response_ids=[50, 51],
            completion_text='resume',
            request_meta={'invocation_kind': 'compaction_resume'},
        )

        trajectory = _materialize(assembler)
        expected = ['conversation', 'summary_inference', 'compaction_resume']

        assert [_segment_kind(node_list) for node_list in trajectory.to_segment_node_lists()] == expected

    def test_reward_slots(self):
        trajectory = _materialize(_compaction_token_session(1))
        llm = [n for n in trajectory.iter_nodes() if n.generated_by == 'llm']
        llm[0].reward = 0.5
        trajectory.final_reward = 1.0
        node_lists = trajectory.to_segment_node_lists()
        assert [node.reward for node in node_lists[0].response_nodes] == [0.5]
        assert [node.reward for node in node_lists[1].response_nodes] == [None]


class TestRenderRestart:
    """A stream contradiction is a fork in the tree."""

    def test_render_fork_starts_at_changed_message(self):
        assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
        _record_tokens(
            assembler,
            messages=[SYSTEM, USER],
            output_message={'role': 'assistant', 'content': 't0'},
            message_spans=[('S', [10]), ('U', [11])],
            generation_prompt=('', []),
            response_ids=[12],
            completion_text='t0',
        )
        _record_tokens(
            assembler,
            messages=[SYSTEM, USER, {'role': 'assistant', 'content': 't0'}, {'role': 'user', 'content': 'q2'}],
            output_message={'role': 'assistant', 'content': 't1'},
            message_spans=[('S', [10]), ('U2', [99]), ('t0', []), ('Q2', [])],
            generation_prompt=('A', [55]),
            response_ids=[77],
            completion_text='t1',
        )
        trajectory = _materialize(assembler)
        _, llm2 = [n for n in trajectory.iter_nodes() if n.generated_by == 'llm']
        assert llm2.segment_boundary.kind == 'render_restart'
        assert 'render_divergence' in llm2.segment_boundary.causes
        system_node = next(node for node in trajectory.iter_nodes() if node.original_message == SYSTEM)
        assert len(system_node.children) == 2
        node_lists = trajectory.to_segment_node_lists()
        assert [_segment_kind(node_list) for node_list in node_lists] == ['conversation', 'render_restart']

    def test_restart_preserves_semantic_kind(self):
        # a physical divergence must not erase policy semantics: if a
        # summary-inference call really restarted the stream, its kind stays,
        # the divergence rides on causes, and the fork derives from the cause
        assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
        _record_tokens(
            assembler,
            messages=[SYSTEM, USER],
            output_message={'role': 'assistant', 'content': 't0'},
            message_spans=[('S', [10]), ('U', [11])],
            generation_prompt=('', []),
            response_ids=[12],
            completion_text='t0',
        )
        _record_tokens(
            assembler,
            messages=[
                SYSTEM,
                USER,
                {'role': 'assistant', 'content': 't0'},
                {'role': 'user', 'content': 'summarize now'},
            ],
            output_message={'role': 'assistant', 'content': 'the summary'},
            message_spans=[('S2', [70]), ('U2', [71]), ('t0', []), ('Q', [])],
            generation_prompt=('A', [72]),
            response_ids=[80],
            completion_text='the summary',
            request_meta={'invocation_kind': 'summary_inference'},
        )
        trajectory = _materialize(assembler)
        summary_llm = [n for n in trajectory.iter_nodes() if n.generated_by == 'llm'][1]
        assert summary_llm.segment_boundary.kind == 'summary_inference'
        assert 'render_divergence' in summary_llm.segment_boundary.causes
        restart_root = summary_llm
        while restart_root.parent is not None:
            restart_root = trajectory.nodes[restart_root.parent]
        assert len(trajectory.roots) == 2
        assert restart_root.node_id in trajectory.roots
        node_lists = trajectory.to_segment_node_lists()
        assert [_segment_kind(node_list) for node_list in node_lists] == ['conversation', 'summary_inference']

    def test_restart_mid_path_forks_under_deepest_match(self):
        assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
        p1 = 'S:sys\nU:question\nA:'
        assembler.record_completion(
            messages=[SYSTEM, USER],
            output_message={'role': 'assistant', 'content': 'ok'},
            token_training=_training(_enc(p1), _enc('ok')),
            completion_text='ok',
            prompt_render=_render(['S:sys', '\nU:question'], '\nA:'),
        )
        p2 = 'S:sys' + '\nU2:question\nA:ok\nU:q2\nA:'
        assembler.record_completion(
            messages=[SYSTEM, USER, {'role': 'assistant', 'content': 'ok'}, {'role': 'user', 'content': 'q2'}],
            output_message={'role': 'assistant', 'content': 'done'},
            token_training=_training(_enc(p2), _enc('done')),
            completion_text='done',
            prompt_render=_render(['S:sys', '\nU2:question', '\nA:ok', '\nU:q2'], '\nA:'),
        )
        trajectory = _materialize(assembler)
        system_node = next(n for n in trajectory.iter_nodes() if n.original_message.get('role') == 'system')
        _, llm2 = [n for n in trajectory.iter_nodes() if n.generated_by == 'llm'][:2]
        assert len(system_node.children) == 2
        assert llm2.segment_boundary.kind == 'render_restart'
        assert llm2.token_account.glue_ids.tolist() == _enc('\nA:')
        node_lists = trajectory.to_segment_node_lists()
        assert [_segment_kind(node_list) for node_list in node_lists] == ['conversation', 'render_restart']
        assert [node.original_message for node in node_lists[1].nodes] == [
            SYSTEM,
            USER,
            {'role': 'assistant', 'content': 'ok'},
            {'role': 'user', 'content': 'q2'},
            {'role': 'assistant', 'content': 'done'},
        ]
        again = type(trajectory).from_dump(trajectory.dump())
        assert [item.segment_id for item in again.to_segment_node_lists()] == [item.segment_id for item in node_lists]


class TestWireVariants:
    def test_normalized_resend_keeps_raw_variant_without_forking(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')
        decorated = {'role': 'assistant', 'content': 'ok', 'provider_specific_fields': {'cached': True}}
        assembler.record_completion(
            messages=[SYSTEM, USER],
            output_message=decorated,
            token_training=_training(_enc('S:sys\nU:question\nA:'), _enc('ok')),
            completion_text='ok',
            prompt_render=_render(['S:sys', '\nU:question'], '\nA:'),
        )
        resent = {'role': 'assistant', 'content': 'ok'}
        assembler.record_completion(
            messages=[SYSTEM, USER, resent, {'role': 'user', 'content': 'next'}],
            output_message={'role': 'assistant', 'content': 'done'},
            token_training=_training(_enc('S:sys\nU:question\nA:ok\nU:next\nA:'), _enc('done')),
            completion_text='done',
            prompt_render=_render(['S:sys', '\nU:question', '\nA:ok', '\nU:next'], '\nA:'),
        )

        trajectory = _materialize(assembler)
        first_llm = next(node for node in trajectory.iter_nodes() if node.generated_by == 'llm')
        assert first_llm.original_message == decorated
        assert first_llm.as_sent_variants == [
            {
                'message': resent,
                'first_request_seq': 1,
                'occurrences': 1,
            }
        ]
        assert len(trajectory.roots) == 1
        assert len(trajectory.to_segment_node_lists()[0].response_nodes) == 2
        assert trajectory.to_messages() == [
            SYSTEM,
            USER,
            resent,
            {'role': 'user', 'content': 'next'},
            {'role': 'assistant', 'content': 'done'},
        ]

    def test_normalized_resend_and_render_restart_create_physical_segments(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')
        decorated = {'role': 'assistant', 'content': 'ok', 'provider_specific_fields': {'cached': True}}
        _record_tokens(
            assembler,
            messages=[SYSTEM, USER],
            output_message=decorated,
            message_spans=[('S', [10]), ('U', [11])],
            generation_prompt=('', []),
            response_ids=[12],
            completion_text='ok',
        )
        resent = {'role': 'assistant', 'content': 'ok'}
        final = {'role': 'assistant', 'content': 'done'}
        _record_tokens(
            assembler,
            messages=[SYSTEM, USER, resent, {'role': 'user', 'content': 'next'}],
            output_message=final,
            message_spans=[('S2', [90]), ('U', [11]), ('A', [12]), ('Q', [20])],
            generation_prompt=('', []),
            response_ids=[30],
            completion_text='done',
        )

        node_lists = _materialize(assembler).to_segment_node_lists()

        assert [_segment_kind(node_list) for node_list in node_lists] == ['conversation', 'render_restart']
        assert [node.original_message for node in node_lists[1].nodes] == [
            SYSTEM,
            USER,
            resent,
            {'role': 'user', 'content': 'next'},
            final,
        ]

    def test_render_fanout_preserves_each_logical_leaf(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')
        first = {'role': 'assistant', 'content': 'first'}
        _record_tokens(
            assembler,
            messages=[SYSTEM, USER],
            output_message=first,
            message_spans=[('S', [10]), ('U', [11])],
            generation_prompt=('', []),
            response_ids=[12],
            completion_text='first',
        )

        expected = []
        for label, system_ids, response_ids in (
            ('normal', [10], [40]),
            ('fork-b', [20], [41]),
            ('fork-c', [21], [42]),
        ):
            user = {'role': 'user', 'content': label}
            reply = {'role': 'assistant', 'content': f'{label}-out'}
            _record_tokens(
                assembler,
                messages=[SYSTEM, USER, first, user],
                output_message=reply,
                message_spans=[('S', system_ids), ('U', [11]), ('A', [12]), ('Q', [30])],
                generation_prompt=('', []),
                response_ids=response_ids,
                completion_text=f'{label}-out',
            )
            expected.append([SYSTEM, USER, first, user, reply])

        node_lists = _materialize(assembler).to_segment_node_lists()

        assert len(node_lists) == 3
        assert {tuple(node.original_message['content'] for node in node_list.nodes) for node_list in node_lists} == {
            *(tuple(message['content'] for message in messages) for messages in expected),
        }

    def test_explicit_observation_keeps_its_timeline_across_a_later_fork(self):
        assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
        first = {'role': 'assistant', 'content': 'call'}
        decorated = {
            'role': 'tool',
            'content': 'result',
            'tool_call_id': 'c1',
            'provider_specific_fields': {'cached': True},
        }
        resent = {'role': 'tool', 'content': 'result', 'tool_call_id': 'c1'}
        _record_tokens(
            assembler,
            messages=[SYSTEM, USER],
            output_message=first,
            message_spans=[('S', [10]), ('U', [11])],
            generation_prompt=('', []),
            response_ids=[12],
            completion_text='call',
        )
        assembler.record_observation(
            message=decorated,
            parent_messages=[SYSTEM, USER, first],
            start_time=2,
            end_time=3,
            status='OK',
        )
        _record_tokens(
            assembler,
            messages=[SYSTEM, USER, first, resent, {'role': 'user', 'content': 'next'}],
            output_message={'role': 'assistant', 'content': 'done'},
            message_spans=[('S2', [99]), ('U', [11]), ('A', [12]), ('O', [13]), ('N', [14])],
            generation_prompt=('', []),
            response_ids=[15],
            completion_text='done',
        )

        trajectory = _materialize(assembler)
        node_lists = trajectory.to_segment_node_lists()
        events = [
            event
            for node in trajectory.iter_nodes()
            for event in node.timeline
            if event.get('event_type') == 'observation'
        ]

        assert len(node_lists) == 2
        assert [event['status'] for event in events] == ['OK']


class TestSidechainProjection:
    def test_sidechain_render_restart_creates_physical_segments(self):
        assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
        meta = {'declared_chain': {'chain_id': 'worker', 'parent_chain_id': 'main', 'kind': 'worker'}}
        system = {'role': 'system', 'content': 'child'}
        user = {'role': 'user', 'content': 'task'}
        decorated = {'role': 'assistant', 'content': 'ok', 'provider_specific_fields': {'cached': True}}
        _record_tokens(
            assembler,
            messages=[system, user],
            output_message=decorated,
            message_spans=[('S', [10]), ('U', [11])],
            generation_prompt=('', []),
            response_ids=[12],
            completion_text='ok',
            request_meta=meta,
        )
        resent = {'role': 'assistant', 'content': 'ok'}
        final = {'role': 'assistant', 'content': 'done'}
        _record_tokens(
            assembler,
            messages=[system, user, resent, {'role': 'user', 'content': 'next'}],
            output_message=final,
            message_spans=[('S', [10]), ('U2', [99]), ('A', [12]), ('Q', [20])],
            generation_prompt=('', []),
            response_ids=[30],
            completion_text='done',
            request_meta=meta,
        )

        node_lists = _materialize(assembler).to_segment_node_lists()

        assert [_segment_kind(node_list) for node_list in node_lists] == ['conversation', 'render_restart']
        assert [node.original_message for node in node_lists[1].nodes] == [
            system,
            user,
            resent,
            {'role': 'user', 'content': 'next'},
            final,
        ]

    def test_nested_sidechain_restart_excludes_parent_turn(self):
        assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
        meta = {'declared_chain': {'chain_id': 'worker', 'parent_chain_id': 'main', 'kind': 'worker'}}
        spawn = {'role': 'assistant', 'content': 'spawn'}
        child = {'role': 'user', 'content': 'child'}
        first = {'role': 'assistant', 'content': 'first'}
        next_user = {'role': 'user', 'content': 'next'}
        second = {'role': 'assistant', 'content': 'second'}
        calls = (
            ([SYSTEM, USER], spawn, [('S', [10]), ('U', [11])], [12], None),
            (
                [SYSTEM, USER, spawn, child],
                first,
                [('S', [10]), ('U', [11]), ('A', [12]), ('C', [20])],
                [21],
                meta,
            ),
            (
                [SYSTEM, USER, spawn, child, first, next_user],
                second,
                [('S2', [99]), ('U', [11]), ('A', [12]), ('C', [20]), ('F', [21]), ('N', [30])],
                [31],
                meta,
            ),
        )
        for messages, output, spans, response_ids, request_meta in calls:
            _record_tokens(
                assembler,
                messages=messages,
                output_message=output,
                message_spans=spans,
                generation_prompt=('', []),
                response_ids=response_ids,
                completion_text=output['content'],
                request_meta=request_meta,
            )

        trajectory = _materialize(assembler)
        node_lists = trajectory.to_segment_node_lists()
        worker = [node_list for node_list in node_lists if node_list.chain_id == 'worker']

        assert len(node_lists) == 3
        assert len(worker) == 2
        assert sum(len(node_list.response_nodes) for node_list in worker) == 2
        assert [node_list.chain_id for node_list in node_lists].count('main') == 1
        assert [node_list.chain_id for node_list in node_lists].count('worker') == 2
        assert all(node_list.segment_id.startswith(f'{node_list.chain_id}@') for node_list in node_lists)


class TestSegmentContinuity:
    def test_message_lane_treats_absent_and_none_tools_equally(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')
        first = {'role': 'assistant', 'content': 'first'}
        assembler.record_completion(
            messages=[USER],
            output_message=first,
            request_kwargs={},
        )
        assembler.record_completion(
            messages=[USER, first, {'role': 'user', 'content': 'next'}],
            output_message={'role': 'assistant', 'content': 'second'},
            request_kwargs={'tools': None},
        )

        trajectory = _materialize(assembler)

        assert len(trajectory.to_segment_node_lists()) == 1

    def test_sidechain_dedup_assigns_each_response_once(self):
        trajectory = Trajectory(
            session_id='legacy-order',
            nodes={
                0: TrajectoryNode(
                    node_id=0,
                    message_id=0,
                    parent=None,
                    generated_by='llm',
                    children=[1, 2],
                    original_message={'role': 'assistant', 'content': 'shared'},
                ),
                1: TrajectoryNode(
                    node_id=1,
                    message_id=1,
                    parent=0,
                    generated_by='llm',
                    children=[],
                    original_message={'role': 'assistant', 'content': 'A'},
                ),
                2: TrajectoryNode(
                    node_id=2,
                    message_id=2,
                    parent=0,
                    generated_by='llm',
                    children=[],
                    original_message={'role': 'assistant', 'content': 'B'},
                ),
            },
            roots=[0],
            chains={'worker': Chain(chain_id='worker', kind='worker', roots=[0])},
        )

        assert [
            [node.node_id for node in node_list.response_nodes] for node_list in trajectory.to_segment_node_lists()
        ] == [[0, 1], [2]]

    def test_mixed_representation_splits_not_drops(self):
        # token -> message -> token must split into three physical segments;
        # the message-level turn must not be silently dropped
        assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
        _record_tokens(
            assembler,
            messages=[SYSTEM, USER],
            output_message={'role': 'assistant', 'content': 't0'},
            message_spans=[('S', [10]), ('U', [11])],
            generation_prompt=('', []),
            response_ids=[12],
            completion_text='t0',
        )
        assembler.record_completion(
            messages=[SYSTEM, USER, {'role': 'assistant', 'content': 't0'}, {'role': 'user', 'content': 'q2'}],
            output_message={'role': 'assistant', 'content': 'm1'},  # message-level call
        )
        _record_tokens(
            assembler,
            messages=[
                SYSTEM,
                USER,
                {'role': 'assistant', 'content': 't0'},
                {'role': 'user', 'content': 'q2'},
                {'role': 'assistant', 'content': 'm1'},
                {'role': 'user', 'content': 'q3'},
            ],
            output_message={'role': 'assistant', 'content': 't2'},
            message_spans=[
                ('S', [10]),
                ('U', [11]),
                ('t0', [12]),
                ('Q2', [20]),
                ('m1', [21]),
                ('Q3', [22]),
            ],
            generation_prompt=('A', [23]),
            response_ids=[30],
            completion_text='t2',
        )
        node_lists = _materialize(assembler).to_segment_node_lists()
        assert len(node_lists) == 3
        assert [len(node_list.response_nodes) for node_list in node_lists] == [1, 1, 1]
        assert all(
            'training_representation_discontinuity' in node_list.response_nodes[0].segment_boundary.causes
            for node_list in node_lists[1:]
        )

    def test_tools_without_a_render_change_do_not_split(self):
        assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
        tools_a = [{'type': 'function', 'function': {'name': 'a', 'parameters': {}}}]
        tools_b = [{'type': 'function', 'function': {'name': 'b', 'parameters': {}}}]
        _record_tokens(
            assembler,
            messages=[SYSTEM, USER],
            output_message={'role': 'assistant', 'content': 't0'},
            message_spans=[('S', [10]), ('U', [11])],
            generation_prompt=('', []),
            response_ids=[12],
            completion_text='t0',
            request_kwargs={'tools': tools_a},
        )
        _record_tokens(
            assembler,
            messages=[SYSTEM, USER, {'role': 'assistant', 'content': 't0'}, {'role': 'user', 'content': 'q2'}],
            output_message={'role': 'assistant', 'content': 't1'},
            message_spans=[('S', [10]), ('U', [11]), ('t0', [12]), ('Q2', [20])],
            generation_prompt=('', []),
            response_ids=[30],
            completion_text='t1',
            request_kwargs={'tools': tools_b},
        )
        node_lists = _materialize(assembler).to_segment_node_lists()
        assert len(node_lists) == 1
        assert len(node_lists[0].response_nodes) == 2

    def test_message_backend_tools_change_is_frozen_as_a_boundary(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')
        first = {'role': 'assistant', 'content': 't0'}
        assembler.record_completion(
            messages=[SYSTEM, USER],
            output_message=first,
            request_kwargs={'tools': [{'type': 'function', 'function': {'name': 'a', 'parameters': {}}}]},
        )
        assembler.record_completion(
            messages=[SYSTEM, USER, first, {'role': 'user', 'content': 'q2'}],
            output_message={'role': 'assistant', 'content': 't1'},
            request_kwargs={'tools': [{'type': 'function', 'function': {'name': 'b', 'parameters': {}}}]},
        )

        trajectory = _materialize(assembler)
        node_lists = trajectory.to_segment_node_lists()
        assert [_segment_kind(node_list) for node_list in node_lists] == ['conversation', 'render_restart']
        assert 'render_context_discontinuity' in node_lists[1].response_nodes[0].segment_boundary.causes

    def test_deferred_tool_discovery_preserves_the_message_segment(self) -> None:
        assembler = TrajectoryAssembler(scaffold_policy='codex')
        tools = [{'type': 'function', 'function': {'name': 'tool_search'}}]
        discovered = {'type': 'function', 'function': {'name': 'calculate', 'defer_loading': True}}
        first = {'role': 'assistant', 'content': 'search'}
        reference = {'role': 'tool', 'content': [{'type': 'tool_reference', 'name': 'calculate'}]}
        assembler.record_completion([SYSTEM, USER], first, request_kwargs={'tools': tools})
        assembler.record_completion(
            [SYSTEM, USER, first, reference],
            {'role': 'assistant', 'content': 'calculate'},
            request_kwargs={'tools': [*tools, discovered]},
        )

        trajectory = _materialize(assembler)
        assert len(trajectory.to_segment_node_lists()) == 1
        responses = [node for node in trajectory.iter_nodes() if node.generated_by == 'llm']
        assert responses[0].request_kwargs['tools'] == tools
        assert responses[1].request_kwargs['tools'] == [*tools, discovered]

    @pytest.mark.parametrize(
        'changed_tools',
        [
            [{'type': 'function', 'function': {'name': 'search'}}],
            [
                {'type': 'function', 'function': {'name': 'calculate', 'defer_loading': True}},
                {'type': 'function', 'function': {'name': 'search'}},
            ],
            [
                {'type': 'function', 'function': {'name': 'search'}},
                {'type': 'function', 'function': {'name': 'calculate', 'defer_loading': False}},
            ],
            [
                {'type': 'function', 'function': {'name': 'search'}},
                {
                    'type': 'function',
                    'function': {'name': 'calculate', 'defer_loading': True, 'description': 'changed'},
                },
            ],
            [
                {'type': 'function', 'function': {'name': 'search'}},
                {'type': 'function', 'function': {'name': 'calculate', 'defer_loading': True}},
                {'type': 'function', 'function': {'name': 'calculate', 'defer_loading': True}},
            ],
        ],
        ids=['removed', 'reordered', 'made-eager', 'schema-changed', 'duplicate-name'],
    )
    def test_existing_tool_changes_restart_the_message_segment(self, changed_tools: list[dict]) -> None:
        assembler = TrajectoryAssembler(scaffold_policy='codex')
        tools = [
            {'type': 'function', 'function': {'name': 'search'}},
            {'type': 'function', 'function': {'name': 'calculate', 'defer_loading': True}},
        ]
        first = {'role': 'assistant', 'content': 'search'}
        assembler.record_completion([SYSTEM, USER], first, request_kwargs={'tools': tools})
        history = [SYSTEM, USER, first, {'role': 'tool', 'content': 'result'}]
        assembler.record_completion(
            history, {'role': 'assistant', 'content': 'answer'}, request_kwargs={'tools': changed_tools}
        )

        responses = [node for node in assembler.trajectory.iter_nodes() if node.generated_by == 'llm']
        assert responses[1].segment_boundary.kind == 'render_restart'

    def test_deferred_tool_discovery_reuses_the_sampled_prefix(self) -> None:
        assembler = TrajectoryAssembler(scaffold_policy='codex')
        tools = [{'type': 'function', 'function': {'name': 'tool_search'}}]
        first = {'role': 'assistant', 'content': 'search'}
        _record_tokens(
            assembler,
            messages=[SYSTEM, USER],
            output_message=first,
            message_spans=[('S', [10]), ('U', [11])],
            generation_prompt=('', []),
            response_ids=[12],
            completion_text='search',
            request_kwargs={'tools': tools},
        )
        history = [SYSTEM, USER, first, {'role': 'tool', 'content': [{'type': 'tool_reference', 'name': 'calculate'}]}]
        discovered = {'type': 'function', 'function': {'name': 'calculate', 'defer_loading': True}}

        prefix = assembler.frozen_prefix(history, tools=[*tools, discovered])

        assert prefix is not None
        assert prefix.count == 3
        assert [list(token_ids) for token_ids in prefix.rollout_message_token_ids] == [[10], [11], [12]]
        assert (
            assembler.frozen_prefix(history, tools=[*tools, {'type': 'function', 'function': {'name': 'other'}}])
            is None
        )

        second = {'role': 'assistant', 'content': 'calculate'}
        _record_tokens(
            assembler,
            messages=history,
            output_message=second,
            message_spans=[('S', [10]), ('U', [11]), ('search', [12]), ('reference', [13])],
            generation_prompt=('', []),
            response_ids=[14],
            completion_text='calculate',
            request_kwargs={'tools': [*tools, discovered]},
        )
        assert len(_materialize(assembler).to_segment_node_lists()) == 1
        changed = {'type': 'function', 'function': {**discovered['function'], 'description': 'changed'}}
        changed_prefix = assembler.frozen_prefix([*history, second], tools=[*tools, changed])
        assert changed_prefix is not None
        assert changed_prefix.node_ids == prefix.node_ids

    def test_deferred_tool_addition_cannot_resolve_an_existing_reference(self) -> None:
        assembler = TrajectoryAssembler(scaffold_policy='codex')
        reference = {'role': 'tool', 'content': [{'type': 'tool_reference', 'name': 'calculate'}]}
        first = {'role': 'assistant', 'content': 'search'}
        tools = [{'type': 'function', 'function': {'name': 'tool_search'}}]
        assembler.record_completion([SYSTEM, USER, reference], first, request_kwargs={'tools': tools})
        discovered = {'type': 'function', 'function': {'name': 'calculate', 'defer_loading': True}}
        assembler.record_completion(
            [SYSTEM, USER, reference, first],
            {'role': 'assistant', 'content': 'answer'},
            request_kwargs={'tools': [*tools, discovered]},
        )

        responses = [node for node in assembler.trajectory.iter_nodes() if node.generated_by == 'llm']
        assert responses[1].segment_boundary.kind == 'render_restart'

    def test_deferred_tool_addition_with_changed_tokens_still_restarts(self) -> None:
        assembler = TrajectoryAssembler(scaffold_policy='codex')
        tools = [{'type': 'function', 'function': {'name': 'tool_search'}}]
        first = {'role': 'assistant', 'content': 'search'}
        _record_tokens(
            assembler,
            messages=[SYSTEM, USER],
            output_message=first,
            message_spans=[('S', [10]), ('U', [11])],
            generation_prompt=('', []),
            response_ids=[12],
            completion_text='search',
            request_kwargs={'tools': tools},
        )
        discovered = {'type': 'function', 'function': {'name': 'calculate', 'defer_loading': True}}
        _record_tokens(
            assembler,
            messages=[SYSTEM, USER, first, {'role': 'tool', 'content': 'reference'}],
            output_message={'role': 'assistant', 'content': 'calculate'},
            message_spans=[('changed', [99]), ('U', [11]), ('search', [12]), ('reference', [13])],
            generation_prompt=('', []),
            response_ids=[14],
            completion_text='calculate',
            request_kwargs={'tools': [*tools, discovered]},
        )

        node_lists = _materialize(assembler).to_segment_node_lists()
        assert [_segment_kind(node_list) for node_list in node_lists] == ['conversation', 'render_restart']
        assert 'render_divergence' in node_lists[1].response_nodes[0].segment_boundary.causes


class TestTimeline:
    def test_node_keeps_timeline(self):
        trajectory = _materialize(_linear_token_session())
        llm = [n for n in trajectory.iter_nodes() if n.generated_by == 'llm']
        event = {'source': 'session', 'type': 'inference', 'start': 1.0, 'end': 2.0}

        llm[0].timeline.append(event)

        assert llm[0].timeline[-1] == event
