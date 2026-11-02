"""Literal baseline oracles for the Direct-Trajectory migration.

The fixtures deliberately use small, hand-numbered token streams.  Do not
derive expected streams through trajectory helpers: these tests are the
behavioral contract the direct assembler must preserve.
"""

from __future__ import annotations

import base64
import pickle

from types import SimpleNamespace

import numpy as np
import pytest

from _fakes import memory_store

from nexus.gateway.session import GatewaySession
from nexus.gateway.trajectory import (
    Chain,
    JoinLink,
    SegmentBoundary,
    SpawnLink,
    TokenAccount,
    Trajectory,
    TrajectoryNode,
)
from nexus.gateway.trajectory_assembler import TrajectoryAssembler
from nexus.gateway.types import PromptRender


EVENT_USER = {'role': 'user', 'content': 'event prompt'}
EVENT_FIRST = {'role': 'assistant', 'content': 'first answer'}
EVENT_OBSERVATION = {'role': 'tool', 'content': 'tool result'}
EVENT_FINAL = {'role': 'assistant', 'content': 'final answer'}


def _list_token_dump(trajectory: Trajectory) -> dict:
    """Compare the historical literal token oracle without changing its expected values."""
    dumped = trajectory.dump()
    for node in dumped['nodes']:
        if (account := node.get('token_account')) is not None and 'token_ids' in account:
            array = account['token_ids']
            assert array['__nexus_gateway_type__'] == 'ndarray', 'numeric fields must use array encoding'
            account['token_ids'] = np.frombuffer(base64.b64decode(array['data']), dtype=array['dtype']).tolist()
    return dumped


class LifecycleStubProcessor:
    is_token_level = False

    def __init__(self, messages: list[dict]):
        self._messages = iter(messages)

    async def create_completion(self, **_kwargs):
        return SimpleNamespace(
            message=next(self._messages),
            prompt_ids=None,
            raw_prompt_ids=None,
            token_ids=None,
            log_probs=None,
            routed_experts=None,
            stop_reason='stop',
            usage=None,
            id=None,
            created=None,
            model=None,
            object=None,
            system_fingerprint=None,
            service_tier=None,
            choice_logprobs=None,
            choice_provider_specific_fields=None,
            extra_fields={},
            completion_text=None,
            prompt_render=None,
        )


def _record_event_tokens(
    assembler: TrajectoryAssembler,
    messages: list[dict],
    output: dict,
    spans: list[tuple[str, list[int]]],
    generation: tuple[str, list[int]],
    response_ids: list[int],
    request_meta: dict | None = None,
) -> None:
    span_text = [text for text, _ in spans]
    span_ids = [token_ids for _, token_ids in spans]
    generation_text, generation_ids = generation
    prompt_ids = [token for token_ids in span_ids for token in token_ids] + generation_ids
    assembler.record_completion(
        messages=messages,
        output_message=output,
        token_training={
            'prompt_ids': prompt_ids,
            'raw_prompt_ids': prompt_ids,
            'response_ids': response_ids,
        },
        completion_text=output['content'],
        prompt_render=PromptRender(
            text=''.join(span_text) + generation_text,
            token_ids=np.array(prompt_ids, dtype=np.int32),
            templated_messages=span_text,
            message_token_ids=[np.array(part, dtype=np.int32) for part in span_ids],
            generation_prompt=generation_text,
            generation_prompt_ids=np.array(generation_ids, dtype=np.int32),
        ),
        request_meta=request_meta,
    )


def _event_path_assembler() -> TrajectoryAssembler:
    assembler = TrajectoryAssembler(scaffold_policy='generic')
    _record_event_tokens(
        assembler,
        [EVENT_USER],
        EVENT_FIRST,
        [('U:event prompt', [10])],
        ('A:', [11]),
        [12],
    )
    assembler.record_observation(message=EVENT_OBSERVATION, parent_messages=[EVENT_USER, EVENT_FIRST])
    _record_event_tokens(
        assembler,
        [EVENT_USER, EVENT_FIRST, EVENT_OBSERVATION],
        EVENT_FINAL,
        [
            ('U:event prompt', [10]),
            ('A:first answer', [11, 12]),
            ('T:tool result', [13, 14, 15]),
        ],
        ('', []),
        [16],
    )
    return assembler


def _freeze_event_path(*, state: str) -> Trajectory:
    assembler = _event_path_assembler()
    trajectory = assembler.freeze(
        session_id='event-path-golden',
        state=state,
        termination_reason='ENV_DONE' if state == 'completed' else None,
        metrics={'elapsed': 3.5} if state == 'completed' else None,
        extra={'runtime_id': 'event-runtime'},
    )
    if state == 'completed':
        trajectory.final_reward = 0.75
    return trajectory


def _golden_multi_agent(*, state: str = 'completed') -> Trajectory:
    """Two physical roots; child execution is linked only logically to main.

    Main's first sampled token is 12.  The next request must consume 13, 14,
    and 15 as observation/context tokens before sampling 16.  The child has
    prompt [20, 21] and samples [22, 23].
    """
    return Trajectory(
        session_id='golden-multi-agent',
        state=state,
        nodes={
            0: TrajectoryNode(
                node_id=0,
                message_id=0,
                parent=None,
                generated_by='client',
                children=[1],
                original_message={'role': 'user', 'content': 'main prompt'},
                token_account=TokenAccount(token_ids=np.array([10], dtype=np.int32)),
            ),
            1: TrajectoryNode(
                node_id=1,
                message_id=1,
                parent=0,
                generated_by='llm',
                children=[2],
                original_message={'role': 'assistant', 'content': 'spawn child'},
                token_account=TokenAccount(token_ids=np.array([11, 12], dtype=np.int32), gen_start=1, gen_len=1),
            ),
            2: TrajectoryNode(
                node_id=2,
                message_id=2,
                parent=1,
                generated_by='observation',
                children=[3],
                original_message={'role': 'tool', 'content': 'child returned'},
                token_account=TokenAccount(token_ids=np.array([13, 14, 15], dtype=np.int32)),
            ),
            3: TrajectoryNode(
                node_id=3,
                message_id=3,
                parent=2,
                generated_by='llm',
                children=[],
                original_message={'role': 'assistant', 'content': 'main final'},
                token_account=TokenAccount(token_ids=np.array([16], dtype=np.int32), gen_len=1),
            ),
            4: TrajectoryNode(
                node_id=4,
                message_id=4,
                parent=None,
                generated_by='client',
                children=[5],
                original_message={'role': 'user', 'content': 'child prompt'},
                token_account=TokenAccount(token_ids=np.array([20], dtype=np.int32)),
            ),
            5: TrajectoryNode(
                node_id=5,
                message_id=5,
                parent=4,
                generated_by='llm',
                children=[],
                original_message={'role': 'assistant', 'content': 'child final'},
                token_account=TokenAccount(token_ids=np.array([21, 22, 23], dtype=np.int32), gen_start=1, gen_len=2),
            ),
        },
        roots=[0, 4],
        chains={
            'main': Chain(chain_id='main', roots=[0]),
            'child': Chain(
                chain_id='child',
                kind='worker',
                parent_chain_id='main',
                roots=[4],
                spawn=SpawnLink(node=1, tool_use_id='spawn-1'),
                join=JoinLink(node=2, kind='notify'),
                detector='golden-oracle',
            ),
        },
        termination_reason='ENV_DONE',
        final_reward=1.25,
        metrics={'elapsed': 2.5},
        extra={'runtime_id': 'golden-runtime'},
    )


def _node_lists_by_chain(trajectory: Trajectory):
    return {node_list.chain_id: node_list for node_list in trajectory.to_segment_node_lists()}


def _response_axis(node_list) -> tuple[list[int], list[int]]:
    token_ids: list[int] = []
    response_mask: list[int] = []
    last_index = node_list.nodes.index(node_list.last_response_node)
    for index, node in enumerate(node_list.nodes[node_list.first_response_node_index : last_index + 1]):
        account = node.token_account
        if account is None or account.token_ids is None:
            continue
        start = account.gen_start if index == 0 else 0
        owned = account.token_ids[start:]
        generated_start = account.gen_start - start
        token_ids.extend(owned)
        response_mask.extend(
            [0] * generated_start + [1] * account.gen_len + [0] * (len(owned) - generated_start - account.gen_len)
        )
    return token_ids, response_mask


def test_event_path_dump_roundtrips():
    trajectory = _freeze_event_path(state='completed')
    expected_dump = {
        'schema_version': 'gateway-trajectory-v1',
        'session_id': 'event-path-golden',
        'state': 'completed',
        'roots': [0],
        'nodes': [
            {
                'node_id': 0,
                'message_id': 0,
                'parent': None,
                'children': [1],
                'generated_by': 'client',
                'original_message': EVENT_USER,
                'templated_message': 'U:event prompt',
                'token_account': {'token_ids': [10], 'has_routed_experts': False},
            },
            {
                'node_id': 1,
                'message_id': 1,
                'parent': 0,
                'children': [2],
                'generated_by': 'llm',
                'original_message': EVENT_FIRST,
                'templated_message': 'A:first answer',
                'token_account': {'token_ids': [11, 12], 'gen_start': 1, 'gen_len': 1, 'has_routed_experts': False},
                'provider_facts': {'finish_reason': 'stop'},
                'timeline': [
                    {
                        'event_type': 'llm_request',
                        'is_sidechain': False,
                        'status': 'OK',
                        'attributes': {'type': 'agent_inference'},
                    }
                ],
            },
            {
                'node_id': 2,
                'message_id': 2,
                'parent': 1,
                'children': [3],
                'generated_by': 'observation',
                'original_message': EVENT_OBSERVATION,
                'templated_message': 'T:tool result',
                'token_account': {'token_ids': [13, 14, 15], 'has_routed_experts': False},
                'timeline': [
                    {
                        'event_type': 'observation',
                        'is_sidechain': False,
                        'attributes': {'type': 'agent_observation'},
                    }
                ],
            },
            {
                'node_id': 3,
                'message_id': 3,
                'parent': 2,
                'children': [],
                'generated_by': 'llm',
                'original_message': EVENT_FINAL,
                'templated_message': 'final answer',
                'token_account': {'token_ids': [16], 'gen_len': 1, 'has_routed_experts': False},
                'provider_facts': {'finish_reason': 'stop'},
                'timeline': [
                    {
                        'event_type': 'llm_request',
                        'is_sidechain': False,
                        'status': 'OK',
                        'attributes': {'type': 'agent_inference'},
                    }
                ],
            },
        ],
        'chains': [{'chain_id': 'main', 'kind': 'main', 'parent_chain_id': None, 'roots': [0]}],
        'timeline': [],
        'final_reward': 0.75,
        'termination_reason': 'ENV_DONE',
        'metrics': {'elapsed': 3.5},
        'extra': {'runtime_id': 'event-runtime'},
    }
    assert _list_token_dump(trajectory) == expected_dump
    for restored in (Trajectory.from_dump(expected_dump), pickle.loads(pickle.dumps(trajectory))):
        assert _list_token_dump(restored) == expected_dump
        assert [item.segment_id for item in restored.to_segment_node_lists()] == ['main@1']


async def test_gateway_session_active_snapshot_is_isolated_and_end_returns_the_live_trajectory():
    metadata = {'run_id': 'run-1', 'nested': {'value': 1}}
    session = GatewaySession(
        'lifecycle-golden',
        LifecycleStubProcessor([EVENT_FIRST]),
        memory_store('lifecycle-golden'),
        metadata=metadata,
        scaffold_policy='generic',
    )
    metadata['nested']['value'] = 99
    await session.create_completion([EVENT_USER])

    active = session.get_trajectory()
    assert active.dump()['session_id'] == 'lifecycle-golden'
    assert active.dump()['state'] == 'active'
    assert active.dump()['final_reward'] is None
    assert active.dump()['termination_reason'] is None
    assert active.dump()['metrics'] == {}
    assert active.dump()['extra'] == {'run_id': 'run-1', 'nested': {'value': 1}}
    assert active.to_messages() == [EVENT_USER, EVENT_FIRST]
    assert [node.node_id for node in active.iter_nodes()] == [0, 1]
    assert [node.parent for node in active.iter_nodes()] == [None, 0]
    assert [node.children for node in active.iter_nodes()] == [[1], []]
    assert [node.original_message for node in active.iter_nodes()] == [EVENT_USER, EVENT_FIRST]
    assert active.to_segment_node_lists()[0].segment_id == 'main@1'

    active.nodes[0].original_message['content'] = 'mutated snapshot'
    active.extra['nested']['value'] = 77
    fresh_active = session.get_trajectory()
    assert fresh_active.nodes[0].original_message == EVENT_USER
    assert fresh_active.extra == {'run_id': 'run-1', 'nested': {'value': 1}}

    metrics = {'score': {'value': 3.5}}
    final_extra = {'runtime_id': 'runtime-1', 'nested_extra': {'value': 2}}
    completed = session.end(
        termination_reason='ENV_DONE',
        final_reward=0.75,
        metrics=metrics,
        extra=final_extra,
    )
    metrics['score']['value'] = 100.0
    final_extra['nested_extra']['value'] = 200

    assert session.is_ended is True
    assert completed is session.segment_assembler.trajectory
    assert completed.dump()['session_id'] == 'lifecycle-golden'
    assert completed.dump()['state'] == 'completed'
    assert completed.dump()['final_reward'] == 0.75
    assert completed.dump()['termination_reason'] == 'ENV_DONE'
    assert completed.dump()['metrics'] == {'score': {'value': 3.5}}
    assert completed.dump()['extra'] == {
        'run_id': 'run-1',
        'nested': {'value': 1},
        'runtime_id': 'runtime-1',
        'nested_extra': {'value': 2},
    }
    assert completed.to_messages() == [EVENT_USER, EVENT_FIRST]
    assert [node.node_id for node in completed.iter_nodes()] == [0, 1]
    assert [node.parent for node in completed.iter_nodes()] == [None, 0]
    assert [node.children for node in completed.iter_nodes()] == [[1], []]
    assert [node.original_message for node in completed.iter_nodes()] == [EVENT_USER, EVENT_FIRST]
    assert completed.to_segment_node_lists()[0].segment_id == 'main@1'


def test_golden_multi_agent_tokens_masks_ids_and_logical_links_are_literal():
    trajectory = _golden_multi_agent()

    node_lists = _node_lists_by_chain(trajectory)
    main = node_lists['main']
    child = node_lists['child']

    assert [item.segment_id for item in trajectory.to_segment_node_lists()] == ['child@5', 'main@1']
    assert [node.node_id for node in child.response_nodes] == [5]
    assert _response_axis(child) == ([22, 23], [1, 1])
    assert [node.node_id for node in main.response_nodes] == [1, 3]
    assert _response_axis(main) == ([12, 13, 14, 15, 16], [1, 0, 0, 0, 1])
    assert trajectory.chains['child'].spawn == SpawnLink(node=1, tool_use_id='spawn-1')
    assert trajectory.chains['child'].join == JoinLink(node=2, kind='notify')


def test_golden_dump_round_trip_and_pickle_preserve_the_literal_contract():
    trajectory = _golden_multi_agent()
    dumped = trajectory.dump()

    assert dumped['schema_version'] == 'gateway-trajectory-v1'
    assert dumped['roots'] == [0, 4]
    assert dumped['chains'] == [
        {'chain_id': 'main', 'kind': 'main', 'parent_chain_id': None, 'roots': [0]},
        {
            'chain_id': 'child',
            'kind': 'worker',
            'parent_chain_id': 'main',
            'roots': [4],
            'spawn': {'node': 1, 'tool_use_id': 'spawn-1', 'candidates': []},
            'join': {'node': 2, 'tool_use_id': None, 'kind': 'notify'},
            'detector': 'golden-oracle',
        },
    ]
    assert [node['token_account'] for node in _list_token_dump(trajectory)['nodes']] == [
        {'token_ids': [10], 'has_routed_experts': False},
        {'token_ids': [11, 12], 'gen_start': 1, 'gen_len': 1, 'has_routed_experts': False},
        {'token_ids': [13, 14, 15], 'has_routed_experts': False},
        {'token_ids': [16], 'gen_len': 1, 'has_routed_experts': False},
        {'token_ids': [20], 'has_routed_experts': False},
        {'token_ids': [21, 22, 23], 'gen_start': 1, 'gen_len': 2, 'has_routed_experts': False},
    ]

    for restored in (Trajectory.from_dump(dumped), pickle.loads(pickle.dumps(trajectory))):
        assert restored.dump() == dumped
        assert [_response_axis(item) for item in restored.to_segment_node_lists()] == [
            ([22, 23], [1, 1]),
            ([12, 13, 14, 15, 16], [1, 0, 0, 0, 1]),
        ]


def test_active_snapshot_keeps_ids_and_payload():
    active = _golden_multi_agent(state='active')

    assert active.state == 'active'
    assert [item.segment_id for item in active.to_segment_node_lists()] == ['child@5', 'main@1']
    assert [_response_axis(item)[0] for item in active.to_segment_node_lists()] == [
        [22, 23],
        [12, 13, 14, 15, 16],
    ]


def test_compaction_crossing_starts_a_new_segment_without_retraining_the_first_response():
    trajectory = _golden_multi_agent()
    trajectory.nodes[3].segment_boundary = SegmentBoundary(
        starts_new_segment=True,
        kind='compaction_resume',
        causes=('compaction',),
    )

    main = [item for item in trajectory.to_segment_node_lists() if item.chain_id == 'main']
    assert [item.segment_id for item in main] == ['main@1', 'main@3']
    assert [_response_axis(item) for item in main] == [([12], [1]), ([16], [1])]
    assert main[1].response_nodes[0].segment_boundary.kind == 'compaction_resume'


SHARED_SYSTEM = {'role': 'system', 'content': 'shared system'}
SHARED_USER = {'role': 'user', 'content': 'shared user'}
SHARED_ANSWER = {'role': 'assistant', 'content': 'shared answer'}
CONTINUATION_USER = {'role': 'user', 'content': 'continuation user'}
CONTINUATION_ANSWER = {'role': 'assistant', 'content': 'continuation answer'}
DECLARED_CHILD = {
    'declared_chain': {
        'chain_id': 'child',
        'parent_chain_id': 'main',
        'kind': 'worker',
    }
}
DECLARED_A = {
    'declared_chain': {
        'chain_id': 'A',
        'parent_chain_id': 'main',
        'kind': 'worker',
    }
}
DECLARED_B = {
    'declared_chain': {
        'chain_id': 'B',
        'parent_chain_id': 'A',
        'kind': 'worker',
    }
}


# Captured verbatim from the detached 6312f04 baseline. These expected values
# must remain independent of the Trajectory produced by the candidate branch.
BASELINE_MAIN_CHILD_CHAINS = [
    {'chain_id': 'main', 'kind': 'main', 'parent_chain_id': None, 'roots': [0]},
    {
        'chain_id': 'child',
        'detector': 'declared_chain',
        'evidence': {
            'anchor_node': 2,
            'attributes': {'parent_chain_id': 'main'},
            'correlation': {
                'fork_node_index': 2,
                'fork_segment_index': 0,
                'parent_chain_id': 'main',
            },
        },
        'kind': 'worker',
        'parent_chain_id': 'main',
        'roots': [3],
    },
]
BASELINE_NESTED_AB_CHAINS = [
    {'chain_id': 'main', 'kind': 'main', 'parent_chain_id': None, 'roots': []},
    {
        'chain_id': 'A',
        'detector': 'declared_chain',
        'evidence': {
            'attributes': {'parent_chain_id': 'main'},
            'correlation': {'parent_chain_id': 'main'},
        },
        'kind': 'worker',
        'parent_chain_id': 'main',
        'roots': [0],
    },
    {
        'chain_id': 'B',
        'detector': 'declared_chain',
        'evidence': {
            'anchor_node': 2,
            'attributes': {'parent_chain_id': 'A'},
            'correlation': {
                'fork_node_index': 2,
                'fork_segment_index': 0,
                'parent_chain_id': 'A',
            },
        },
        'kind': 'worker',
        'parent_chain_id': 'A',
        'roots': [3],
    },
]


def _baseline_shared_prefix_nodes(*, token_level: bool, first_is_sidechain: bool) -> list[dict]:
    if token_level:
        return [
            {
                'children': [1],
                'generated_by': 'client',
                'message_id': 0,
                'node_id': 0,
                'original_message': {'content': 'shared system', 'role': 'system'},
                'parent': None,
                'templated_message': '<S>',
                'token_account': {'token_ids': [10], 'has_routed_experts': False},
            },
            {
                'children': [2],
                'generated_by': 'client',
                'message_id': 1,
                'node_id': 1,
                'original_message': {'content': 'shared user', 'role': 'user'},
                'parent': 0,
                'templated_message': '<U>',
                'token_account': {'token_ids': [11], 'has_routed_experts': False},
            },
            {
                'children': [3],
                'generated_by': 'llm',
                'message_id': 2,
                'node_id': 2,
                'original_message': {'content': 'shared answer', 'role': 'assistant'},
                'parent': 1,
                'provider_facts': {'finish_reason': 'stop'},
                'templated_message': '<A>shared answer',
                'timeline': [
                    {
                        'attributes': {'type': 'agent_inference'},
                        'event_type': 'llm_request',
                        'is_sidechain': first_is_sidechain,
                        'status': 'OK',
                    }
                ],
                'token_account': {'gen_len': 1, 'gen_start': 1, 'token_ids': [12, 13], 'has_routed_experts': False},
            },
            {
                'children': [4],
                'generated_by': 'client',
                'message_id': 3,
                'node_id': 3,
                'original_message': {'content': 'continuation user', 'role': 'user'},
                'parent': 2,
                'templated_message': '<N>',
                'token_account': {'token_ids': [14], 'has_routed_experts': False},
            },
            {
                'children': [],
                'generated_by': 'llm',
                'message_id': 4,
                'node_id': 4,
                'original_message': {'content': 'continuation answer', 'role': 'assistant'},
                'parent': 3,
                'provider_facts': {'finish_reason': 'stop'},
                'templated_message': '<B>continuation answer',
                'timeline': [
                    {
                        'attributes': {'type': 'agent_inference'},
                        'event_type': 'llm_request',
                        'is_sidechain': True,
                        'status': 'OK',
                    }
                ],
                'token_account': {'gen_len': 1, 'gen_start': 1, 'token_ids': [15, 16], 'has_routed_experts': False},
            },
        ]
    return [
        {
            'children': [1],
            'generated_by': 'client',
            'message_id': 0,
            'node_id': 0,
            'original_message': {'content': 'shared system', 'role': 'system'},
            'parent': None,
        },
        {
            'children': [2],
            'generated_by': 'client',
            'message_id': 1,
            'node_id': 1,
            'original_message': {'content': 'shared user', 'role': 'user'},
            'parent': 0,
        },
        {
            'children': [3],
            'generated_by': 'llm',
            'message_id': 2,
            'node_id': 2,
            'original_message': {'content': 'shared answer', 'role': 'assistant'},
            'parent': 1,
            'provider_facts': {'finish_reason': 'stop'},
            'timeline': [
                {
                    'attributes': {'type': 'agent_inference'},
                    'event_type': 'llm_request',
                    'is_sidechain': first_is_sidechain,
                    'status': 'OK',
                }
            ],
        },
        {
            'children': [4],
            'generated_by': 'client',
            'message_id': 3,
            'node_id': 3,
            'original_message': {'content': 'continuation user', 'role': 'user'},
            'parent': 2,
        },
        {
            'children': [],
            'generated_by': 'llm',
            'message_id': 4,
            'node_id': 4,
            'original_message': {'content': 'continuation answer', 'role': 'assistant'},
            'parent': 3,
            'provider_facts': {'finish_reason': 'stop'},
            'timeline': [
                {
                    'attributes': {'type': 'agent_inference'},
                    'event_type': 'llm_request',
                    'is_sidechain': True,
                    'status': 'OK',
                }
            ],
        },
    ]


def _baseline_shared_prefix_dump(*, chains: list[dict], nodes: list[dict]) -> dict:
    return {
        'chains': chains,
        'extra': {},
        'final_reward': None,
        'metrics': {},
        'nodes': nodes,
        'roots': [0],
        'schema_version': 'gateway-trajectory-v1',
        'session_id': '',
        'state': 'completed',
        'termination_reason': None,
        'timeline': [],
    }


def _shared_prefix_trajectory(*, first_meta: dict | None, second_meta: dict | None, token_level: bool) -> Trajectory:
    assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
    if token_level:
        _record_event_tokens(
            assembler,
            [SHARED_SYSTEM, SHARED_USER],
            SHARED_ANSWER,
            [('<S>', [10]), ('<U>', [11])],
            ('<A>', [12]),
            [13],
            request_meta=first_meta,
        )
        _record_event_tokens(
            assembler,
            [SHARED_SYSTEM, SHARED_USER, SHARED_ANSWER, CONTINUATION_USER],
            CONTINUATION_ANSWER,
            [
                ('<S>', [10]),
                ('<U>', [11]),
                ('<A>shared answer', [12, 13]),
                ('<N>', [14]),
            ],
            ('<B>', [15]),
            [16],
            request_meta=second_meta,
        )
    else:
        assembler.record_completion(
            messages=[SHARED_SYSTEM, SHARED_USER],
            output_message=SHARED_ANSWER,
            request_meta=first_meta,
        )
        assembler.record_completion(
            messages=[SHARED_SYSTEM, SHARED_USER, SHARED_ANSWER, CONTINUATION_USER],
            output_message=CONTINUATION_ANSWER,
            request_meta=second_meta,
        )
    return assembler.freeze(state='completed')


def _shared_prefix_structure(trajectory: Trajectory) -> list[tuple]:
    return [
        (
            node.node_id,
            node.message_id,
            node.parent,
            node.children,
            node.generated_by,
            node.original_message,
        )
        for node in trajectory.iter_nodes()
    ]


def _shared_prefix_segments(trajectory: Trajectory) -> list[dict]:
    result = []
    for node_list in trajectory.to_segment_node_lists():
        token_level = node_list.response_nodes[0].token_account is not None
        response_ids, response_mask = _response_axis(node_list)
        result.append(
            {
                'segment_id': node_list.segment_id,
                'chain_id': node_list.chain_id,
                'node_ids': [node.node_id for node in node_list.response_nodes],
                'messages': [node.original_message for node in node_list.nodes],
                'num_turns': len(node_list.response_nodes),
                'response_ids': response_ids if token_level else None,
                'response_mask': response_mask if token_level else None,
            }
        )
    return result


def _assert_shared_prefix_physical_tree(trajectory: Trajectory) -> None:
    assert trajectory.roots == [0]
    assert _shared_prefix_structure(trajectory) == [
        (0, 0, None, [1], 'client', SHARED_SYSTEM),
        (1, 1, 0, [2], 'client', SHARED_USER),
        (2, 2, 1, [3], 'llm', SHARED_ANSWER),
        (3, 3, 2, [4], 'client', CONTINUATION_USER),
        (4, 4, 3, [], 'llm', CONTINUATION_ANSWER),
    ]


@pytest.mark.parametrize('token_level', [False, True], ids=['message', 'token'])
def test_main_shared_prefix_remains_main_when_declared_child_continues(token_level):
    trajectory = _shared_prefix_trajectory(first_meta=None, second_meta=DECLARED_CHILD, token_level=token_level)

    assert _list_token_dump(trajectory) == _baseline_shared_prefix_dump(
        chains=BASELINE_MAIN_CHILD_CHAINS,
        nodes=_baseline_shared_prefix_nodes(token_level=token_level, first_is_sidechain=False),
    )
    _assert_shared_prefix_physical_tree(trajectory)
    assert list(trajectory.chains) == ['main', 'child']
    assert trajectory.chains['main'].roots == [0]
    child_chain = trajectory.chains['child']
    assert (child_chain.kind, child_chain.parent_chain_id, child_chain.roots, child_chain.detector) == (
        'worker',
        'main',
        [3],
        'declared_chain',
    )
    assert child_chain.evidence['anchor_node'] == 2
    assert [trajectory.node_chain(node_id) for node_id in range(5)] == [
        'main',
        'main',
        'main',
        'child',
        'child',
    ]
    child_response, child_mask, main_response, main_mask = (
        ([16], [1], [13], [1]) if token_level else (None, None, None, None)
    )
    assert _shared_prefix_segments(trajectory) == [
        {
            'segment_id': 'child@4',
            'chain_id': 'child',
            'node_ids': [4],
            'messages': [SHARED_SYSTEM, SHARED_USER, SHARED_ANSWER, CONTINUATION_USER, CONTINUATION_ANSWER],
            'num_turns': 1,
            'response_ids': child_response,
            'response_mask': child_mask,
        },
        {
            'segment_id': 'main@2',
            'chain_id': 'main',
            'node_ids': [2],
            'messages': [SHARED_SYSTEM, SHARED_USER, SHARED_ANSWER],
            'num_turns': 1,
            'response_ids': main_response,
            'response_mask': main_mask,
        },
    ]


def test_nested_declared_chain_a_to_b_full_dump_matches_baseline():
    trajectory = _shared_prefix_trajectory(first_meta=DECLARED_A, second_meta=DECLARED_B, token_level=False)

    assert _list_token_dump(trajectory) == _baseline_shared_prefix_dump(
        chains=BASELINE_NESTED_AB_CHAINS,
        nodes=_baseline_shared_prefix_nodes(token_level=False, first_is_sidechain=True),
    )
    assert [trajectory.node_chain(node_id) for node_id in range(5)] == ['A', 'A', 'A', 'B', 'B']
    assert [
        (node_list.segment_id, [node.node_id for node in node_list.response_nodes])
        for node_list in trajectory.to_segment_node_lists()
    ] == [
        ('B@4', [4]),
        ('A@2', [2]),
    ]


@pytest.mark.parametrize('token_level', [False, True], ids=['message', 'token'])
def test_synthetic_declared_child_prefix_collapses_to_main_when_main_continues(token_level):
    trajectory = _shared_prefix_trajectory(first_meta=DECLARED_CHILD, second_meta=None, token_level=token_level)

    _assert_shared_prefix_physical_tree(trajectory)
    assert trajectory.dump()['chains'] == [
        {
            'chain_id': 'main',
            'kind': 'main',
            'parent_chain_id': None,
            'roots': [0],
            'evidence': {'attributes': {'parent_chain_id': 'main'}},
        }
    ]
    assert [trajectory.node_chain(node_id) for node_id in range(5)] == ['main'] * 5
    response_ids, response_mask = ([13, 14, 15, 16], [1, 0, 0, 1]) if token_level else (None, None)
    assert _shared_prefix_segments(trajectory) == [
        {
            'segment_id': 'main@2',
            'chain_id': 'main',
            'node_ids': [2, 4],
            'messages': [SHARED_SYSTEM, SHARED_USER, SHARED_ANSWER, CONTINUATION_USER, CONTINUATION_ANSWER],
            'num_turns': 2,
            'response_ids': response_ids,
            'response_mask': response_mask,
        }
    ]


def test_shared_llm_node_is_a_training_target_once_across_plain_forks():
    assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
    system = {'role': 'system', 'content': 'you are an agent'}
    user = {'role': 'user', 'content': 'do the task'}
    shared = {'role': 'assistant', 'content': 'shared response'}
    assembler.record_completion(messages=[system, user], output_message=shared)
    assembler.record_completion(
        messages=[system, user, shared, {'role': 'user', 'content': 'left'}],
        output_message={'role': 'assistant', 'content': 'left response'},
    )
    assembler.record_completion(
        messages=[system, user, shared, {'role': 'user', 'content': 'right'}],
        output_message={'role': 'assistant', 'content': 'right response'},
    )

    trajectory = assembler.freeze(state='completed')
    assert trajectory.roots == [0]
    assert _shared_prefix_structure(trajectory) == [
        (0, 0, None, [1], 'client', system),
        (1, 1, 0, [2], 'client', user),
        (2, 2, 1, [3, 5], 'llm', shared),
        (3, 3, 2, [4], 'client', {'role': 'user', 'content': 'left'}),
        (4, 4, 3, [], 'llm', {'role': 'assistant', 'content': 'left response'}),
        (5, 5, 2, [6], 'client', {'role': 'user', 'content': 'right'}),
        (6, 6, 5, [], 'llm', {'role': 'assistant', 'content': 'right response'}),
    ]
    assert trajectory.dump()['chains'] == [{'chain_id': 'main', 'kind': 'main', 'parent_chain_id': None, 'roots': [0]}]
    assert [trajectory.node_chain(node_id) for node_id in range(7)] == ['main'] * 7
    node_lists = trajectory.to_segment_node_lists()
    assert [
        (node_list.segment_id, [node.node_id for node in node_list.response_nodes]) for node_list in node_lists
    ] == [('main@2', [2, 4]), ('main@6', [6])]
    assert [node.node_id for node_list in node_lists for node in node_list.response_nodes].count(2) == 1


def test_shared_prefix_fanout_keeps_response_ownership_distinct():
    assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
    system = {'role': 'system', 'content': 'you are an agent'}
    user = {'role': 'user', 'content': 'do the task'}
    shared = {'role': 'assistant', 'content': 'shared response'}
    left_user = {'role': 'user', 'content': 'left'}
    left_reply = {'role': 'assistant', 'content': 'left response'}
    right_user = {'role': 'user', 'content': 'right'}
    right_reply = {'role': 'assistant', 'content': 'right response'}
    _record_event_tokens(
        assembler,
        [system, user],
        shared,
        [('<S>', [7]), ('<U>', [7])],
        ('<A>', [7]),
        [7],
    )
    _record_event_tokens(
        assembler,
        [system, user, shared, left_user],
        left_reply,
        [('<S>', [7]), ('<U>', [7]), ('<A>shared response', [7, 7]), ('<L>', [7])],
        ('<A>', [7]),
        [7],
    )
    _record_event_tokens(
        assembler,
        [system, user, shared, right_user],
        right_reply,
        [('<S>', [7]), ('<U>', [7]), ('<A>shared response', [7, 7]), ('<R>', [7])],
        ('<A>', [7]),
        [7],
    )

    trajectory = assembler.freeze(state='completed')
    left, right = trajectory.to_segment_node_lists()

    assert [
        (left.segment_id, [node.node_id for node in left.response_nodes]),
        (right.segment_id, [node.node_id for node in right.response_nodes]),
    ] == [
        ('main@2', [2, 4]),
        ('main@6', [6]),
    ]
    assert _response_axis(left)[0] == [7, 7, 7, 7]
    assert _response_axis(right)[0] == [7]


def test_rejected_sibling_is_not_exported():
    trajectory = _golden_multi_agent()
    trajectory.nodes[5].selection = 'rejected_sibling'
    node_lists = trajectory.to_segment_node_lists()
    assert [node_list.segment_id for node_list in node_lists] == ['main@1']
    assert [node.node_id for node in node_lists[0].response_nodes] == [1, 3]


def test_unknown_chain_fails_loudly():
    trajectory = _golden_multi_agent()

    with pytest.raises(ValueError, match='unknown chain ids'):
        trajectory.to_segment_node_lists(chains=['missing'])


def test_direct_assembler_exposes_one_live_active_trajectory_before_freeze():
    """Task 2 contract: committed records are immediately visible on one Trajectory.

    Baseline deliberately has no such public direct handle; the expected RED
    failure is ``AttributeError: 'TrajectoryAssembler' object has no attribute
    'trajectory'``.  The later direct implementation must satisfy this without
    changing the golden public schema above.
    """
    assembler = TrajectoryAssembler(scaffold_policy='generic')
    assembler.record_completion(
        messages=[{'role': 'user', 'content': 'prompt'}],
        output_message={'role': 'assistant', 'content': 'answer'},
    )

    assert assembler.trajectory.state == 'active'
    assert [node.node_id for node in assembler.trajectory.iter_nodes()] == [0, 1]
