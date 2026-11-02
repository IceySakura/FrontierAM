"""Nexus declared-chain and compaction regression tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.unit.gateway.helpers import freeze_assembler

from nexus.gateway.trajectory_assembler import TrajectoryAssembler
from nexus.gateway.types import PromptRender


SYSTEM = {'role': 'system', 'content': 'nexus scaffold'}
USER = {'role': 'user', 'content': 'do the task'}


def _segment_kind(node_list) -> str:
    boundary = node_list.response_nodes[0].segment_boundary
    return boundary.kind if boundary is not None and boundary.kind is not None else 'conversation'


def test_declared_chain_exports_as_sidechain_end_to_end():
    worker_meta = {
        'declared_chain': {
            'chain_id': 'chain-7',
            'parent_chain_id': 'chain-root',
            'kind': 'worker',
        }
    }

    assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
    # Main lane has no declared chain.
    assembler.record_completion(
        messages=[SYSTEM, USER],
        output_message={'role': 'assistant', 'content': 'main answer'},
    )
    # Declared worker is an independent root.
    assembler.record_completion(
        messages=[{'role': 'system', 'content': 'worker scaffold'}, {'role': 'user', 'content': 'child task'}],
        output_message={'role': 'assistant', 'content': 'child answer'},
        request_meta=worker_meta,
    )

    trajectory = freeze_assembler(assembler)
    node_lists = trajectory.to_segment_node_lists()
    main = [node_list for node_list in node_lists if node_list.chain_id == 'main']
    side = [node_list for node_list in node_lists if node_list.chain_id != 'main']

    assert len(main) == 1
    assert len(side) == 1
    chain = trajectory.chains[side[0].chain_id]
    assert chain.kind == 'worker'
    assert chain.detector == 'declared_chain'
    assert chain.parent_chain_id == 'chain-root'


def _call(call_id: str, name: str, arguments: dict) -> dict:
    return {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}


def _tool(call_id: str, payload: dict) -> dict:
    return {'role': 'tool', 'tool_call_id': call_id, 'content': json.dumps(payload)}


def _subagent_meta(agent_id: str) -> dict:
    return {
        'agent_id': agent_id,
        'parent_agent_id': 'main',
        'declared_chain': {
            'chain_id': agent_id,
            'parent_chain_id': 'main',
            'kind': 'subagent',
            'spawn_order': int(agent_id.rsplit('-', 1)[1]),
        },
    }


def _completed(agent_id: str) -> dict:
    return {'agent_id': agent_id, 'name': None, 'status': 'completed', 'final_answer': 'ok', 'exit_status': 'finished'}


class TestNexusSubagentLinks:
    def test_spawn_and_wait_results_resolve_chain_spawn_and_join(self) -> None:
        assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
        spawn_turn = {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                _call('spawn-1', 'spawn_agent', {'prompt': 'task one'}),
                _call('spawn-2', 'spawn_agent', {'prompt': 'task two'}),
            ],
        }
        history = [SYSTEM, USER]
        assembler.record_completion(history, spawn_turn)
        for agent_id, prompt in (('agent-1', 'task one'), ('agent-2', 'task two')):
            assembler.record_completion(
                [{'role': 'system', 'content': 'worker'}, {'role': 'user', 'content': prompt}],
                {'role': 'assistant', 'content': f'{agent_id} done'},
                request_meta=_subagent_meta(agent_id),
            )
        wait_any = {
            'role': 'assistant',
            'content': None,
            'tool_calls': [_call('wait-1', 'wait_agent', {'agent_ids': ['agent-1', 'agent-2'], 'return_when': 'any'})],
        }
        history += [
            spawn_turn,
            _tool('spawn-1', {'agent_id': 'agent-1', 'name': None, 'status': 'running'}),
            _tool('spawn-2', {'agent_id': 'agent-2', 'name': None, 'status': 'running'}),
        ]
        assembler.record_completion(history, wait_any)
        wait_rest = {
            'role': 'assistant',
            'content': None,
            'tool_calls': [_call('wait-2', 'wait_agent', {'agent_ids': ['agent-2']})],
        }
        history += [
            wait_any,
            _tool('wait-1', {'completed': [_completed('agent-1')], 'failed': [], 'pending': ['agent-2']}),
        ]
        assembler.record_completion(history, wait_rest)
        history += [wait_rest, _tool('wait-2', {'completed': [_completed('agent-2')], 'failed': [], 'pending': []})]
        assembler.record_completion(history, {'role': 'assistant', 'content': 'final'})

        trajectory = freeze_assembler(assembler)
        spawn_node = next(
            node.node_id
            for node in trajectory.iter_nodes()
            if node.original_message.get('tool_calls') == spawn_turn['tool_calls']
        )
        first, second = trajectory.chains['agent-1'], trajectory.chains['agent-2']

        assert list(trajectory.chains) == ['main', 'agent-1', 'agent-2']
        assert (first.kind, first.parent_chain_id) == ('subagent', 'main')
        assert (first.spawn.node, first.spawn.tool_use_id) == (spawn_node, 'spawn-1')
        assert (second.spawn.node, second.spawn.tool_use_id) == (spawn_node, 'spawn-2')
        assert (first.join.tool_use_id, first.join.kind) == ('wait-1', 'poll')
        assert (second.join.tool_use_id, second.join.kind) == ('wait-2', 'poll')
        assert trajectory.nodes[second.join.node].original_message['tool_call_id'] == 'wait-2'
        assert 'unresolved_correlations' not in trajectory.chains['main'].evidence
        assert 'spawn_order' not in first.evidence['attributes']

    def test_timed_out_wait_leaves_the_pending_child_unjoined(self) -> None:
        assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
        spawn_turn = {
            'role': 'assistant',
            'content': None,
            'tool_calls': [_call('spawn-1', 'spawn_agent', {'prompt': 't'})],
        }
        wait_turn = {
            'role': 'assistant',
            'content': None,
            'tool_calls': [_call('wait-1', 'wait_agent', {'agent_ids': ['agent-1']})],
        }
        assembler.record_completion([SYSTEM, USER], spawn_turn)
        assembler.record_completion(
            [{'role': 'system', 'content': 'worker'}, {'role': 'user', 'content': 't'}],
            {'role': 'assistant', 'content': 'child'},
            request_meta=_subagent_meta('agent-1'),
        )
        history = [
            SYSTEM,
            USER,
            spawn_turn,
            _tool('spawn-1', {'agent_id': 'agent-1', 'name': None, 'status': 'running'}),
        ]
        assembler.record_completion(history, wait_turn)
        history += [
            wait_turn,
            _tool('wait-1', {'completed': [], 'failed': [], 'pending': ['agent-1'], 'timed_out': True}),
        ]
        assembler.record_completion(history, {'role': 'assistant', 'content': 'gave up'})

        child = freeze_assembler(assembler).chains['agent-1']

        assert child.spawn.tool_use_id == 'spawn-1'
        assert child.join is None


def test_identical_declared_chains_keep_separate_attribution():
    assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
    first_reply = {'role': 'assistant', 'content': 'identical reply'}
    first_messages = [SYSTEM, USER]
    worker_1 = {'declared_chain': {'chain_id': 'worker-1', 'parent_chain_id': 'main', 'kind': 'worker'}}
    worker_2 = {'declared_chain': {'chain_id': 'worker-2', 'parent_chain_id': 'main', 'kind': 'worker'}}
    assembler.record_completion(first_messages, first_reply, request_meta=worker_1)
    assembler.record_completion(first_messages, first_reply, request_meta=worker_2)
    assembler.record_completion(
        first_messages + [first_reply, {'role': 'user', 'content': 'continue'}],
        {'role': 'assistant', 'content': 'worker-2 final'},
        request_meta=worker_2,
    )

    node_lists = freeze_assembler(assembler).to_segment_node_lists()
    by_agent = {node_list.chain_id: node_list for node_list in node_lists}

    assert set(by_agent) == {'worker-1', 'worker-2'}
    assert len(by_agent['worker-1'].response_nodes) == 1
    assert len(by_agent['worker-2'].response_nodes) == 2
    assert (
        sum(
            node.original_message.get('content') == 'identical reply'
            for node_list in node_lists
            for node in node_list.response_nodes
        )
        == 2
    )


def test_reused_declared_chain_id_under_a_different_parent_gets_a_collision_safe_id():
    assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
    first = {'declared_chain': {'chain_id': 'worker', 'parent_chain_id': 'parent-a', 'kind': 'worker'}}
    second = {'declared_chain': {'chain_id': 'worker', 'parent_chain_id': 'parent-b', 'kind': 'worker'}}
    assembler.record_completion(
        [{'role': 'system', 'content': 'worker a'}, {'role': 'user', 'content': 'task a'}],
        {'role': 'assistant', 'content': 'answer a'},
        request_meta=first,
    )
    assembler.record_completion(
        [{'role': 'system', 'content': 'worker b'}, {'role': 'user', 'content': 'task b'}],
        {'role': 'assistant', 'content': 'answer b'},
        request_meta=second,
    )

    trajectory = freeze_assembler(assembler)

    assert list(trajectory.chains) == ['main', 'worker', 'worker@3']
    assert trajectory.chains['worker'].parent_chain_id == 'parent-a'
    assert trajectory.chains['worker@3'].parent_chain_id == 'parent-b'
    assert trajectory.chains['worker@3'].evidence['collides_with'] == 'worker'


def test_declared_sidechain_keeps_its_own_compaction_boundary():
    assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
    worker = {'declared_chain': {'chain_id': 'worker-1', 'parent_chain_id': 'main', 'kind': 'worker'}}
    before = {'role': 'assistant', 'content': 'before compaction'}
    assembler.record_completion(
        [SYSTEM, USER],
        before,
        request_meta=worker,
    )
    assembler.record_completion(
        [SYSTEM, USER, before, {'role': 'user', 'content': 'worker summary', '_is_compacted_summary': True}],
        {'role': 'assistant', 'content': 'after compaction'},
        request_meta=worker,
    )

    node_lists = freeze_assembler(assembler).to_segment_node_lists()

    assert [_segment_kind(node_list) for node_list in node_lists] == ['conversation', 'compaction_resume']
    assert {node_list.chain_id for node_list in node_lists} == {'worker-1'}


@pytest.mark.parametrize('k', [1, 2, 3])
def test_nexus_k_compaction_resumes_yield_k_plus_one_segments(k):
    assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
    assembler.record_completion(messages=[SYSTEM, USER], output_message={'role': 'assistant', 'content': 'turn-0'})

    for i in range(1, k + 1):
        summary = {'role': 'user', 'content': f'summary {i}', '_is_compacted_summary': True}
        reply = {'role': 'assistant', 'content': f'resume {i}'}
        # Real Nexus compaction replaces the prior history. Every compacted
        # request therefore forks from the stable [system, first-user] prefix
        # instead of appending its summary to the preceding compacted path.
        assembler.record_completion(messages=[SYSTEM, USER, summary], output_message=reply)

    node_lists = freeze_assembler(assembler).to_segment_node_lists()

    assert len(node_lists) == k + 1
    assert [_segment_kind(node_list) for node_list in node_lists] == ['conversation'] + ['compaction_resume'] * k
    assert [len(node_list.response_nodes) for node_list in node_lists] == [1] * (k + 1)


def _training(raw_prompt_ids, response_ids):
    return {
        'prompt_ids': raw_prompt_ids,
        'raw_prompt_ids': raw_prompt_ids,
        'response_ids': response_ids,
    }


def _record_tokens(assembler, messages, output, spans, response_ids):
    texts = [text for text, _ in spans]
    token_parts = [ids for _, ids in spans]
    prompt_ids = [token for ids in token_parts for token in ids]
    render = PromptRender(
        text=''.join(texts),
        token_ids=np.array(prompt_ids, dtype=np.int32),
        templated_messages=texts,
        message_token_ids=[np.array(part, dtype=np.int32) for part in token_parts],
        generation_prompt='',
        generation_prompt_ids=np.array([], dtype=np.int32),
    )
    assembler.record_completion(
        messages=messages,
        output_message=output,
        token_training=_training(prompt_ids, response_ids),
        completion_text=output.get('content') or '',
        prompt_render=render,
    )


def test_nexus_compaction_segments_carry_response_only_masks_under_tokens():
    k = 2
    assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
    expected_responses = []

    _record_tokens(
        assembler,
        [SYSTEM, USER],
        {'role': 'assistant', 'content': 'turn-0'},
        [('S', [10]), ('U', [11])],
        [12, 13],
    )
    expected_responses.append([12, 13])

    for i in range(1, k + 1):
        summary = {'role': 'user', 'content': f'summary {i}', '_is_compacted_summary': True}
        reply = {'role': 'assistant', 'content': f'resume {i}'}
        prompt_ids = [100 * i, 100 * i + 1, 100 * i + 2]
        response_ids = [100 * i + 50, 100 * i + 51]
        _record_tokens(
            assembler,
            [SYSTEM, USER, summary],
            reply,
            [('S', [prompt_ids[0]]), ('U', [prompt_ids[1]]), ('C', [prompt_ids[2]])],
            response_ids,
        )
        expected_responses.append(response_ids)

    trajectory = freeze_assembler(assembler)
    node_lists = trajectory.to_segment_node_lists()
    records = trajectory.prepare_segment_training_data((node_list, True) for node_list in node_lists)

    assert len(node_lists) == k + 1
    assert [_segment_kind(node_list) for node_list in node_lists] == [
        'conversation',
        'compaction_resume',
        'compaction_resume',
    ]
    assert [record.response_length for record in records.values()] == [len(response) for response in expected_responses]
    assert [record.generated_ranges for record in records.values()] == [
        ((0, len(response)),) for response in expected_responses
    ]


def test_declared_compaction_under_physical_coincidence_has_no_split_cause():
    """keep_recent-style compaction: the rewritten (shorter) history inevitably trips the
    physical divergence detectors at the same request that carries the declared
    ``invocation_kind='compaction_resume'``. The declared boundary wins: the segment is
    compaction_resume AND its split_cause stays None, so downstream labeling reports
    CONTEXT_MANAGEMENT instead of a physical restart."""
    assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
    _record_tokens(
        assembler,
        [SYSTEM, USER],
        {'role': 'assistant', 'content': 't0'},
        [('S', [10]), ('U', [11])],
        [20],
    )
    _record_tokens_with_meta(
        assembler,
        [SYSTEM, USER],
        {'role': 'assistant', 'content': 't2'},
        [('S', [10]), ('U', [11])],
        [22],
        request_meta={'invocation_kind': 'compaction_resume'},
    )

    node_lists = freeze_assembler(assembler).to_segment_node_lists()

    assert [_segment_kind(node_list) for node_list in node_lists] == ['conversation', 'compaction_resume']
    boundary = node_lists[1].response_nodes[0].segment_boundary
    assert boundary.causes == ('compaction',)
    assert boundary.evidence.get('physical_only') is not True


def _record_tokens_with_meta(assembler, messages, output, spans, response_ids, request_meta):
    texts = [text for text, _ in spans]
    token_parts = [ids for _, ids in spans]
    prompt_ids = [token for ids in token_parts for token in ids]
    render = PromptRender(
        text=''.join(texts),
        token_ids=np.array(prompt_ids, dtype=np.int32),
        templated_messages=texts,
        message_token_ids=[np.array(part, dtype=np.int32) for part in token_parts],
        generation_prompt='',
        generation_prompt_ids=np.array([], dtype=np.int32),
    )
    assembler.record_completion(
        messages=messages,
        output_message=output,
        token_training=_training(prompt_ids, response_ids),
        completion_text=output.get('content') or '',
        prompt_render=render,
        request_meta=request_meta,
    )
