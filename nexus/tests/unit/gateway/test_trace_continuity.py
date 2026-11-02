from dataclasses import replace

import numpy as np

from tests.unit.gateway.helpers import freeze_assembler, latest_node_id

from nexus.gateway.scaffold_policy import GenericScaffoldPolicy, RequestEvent, ResponseEvent, SidechainMembership
from nexus.gateway.trajectory import TokenAccount
from nexus.gateway.trajectory_assembler import TrajectoryAssembler
from nexus.gateway.types import PromptRender


def _training(raw_prompt_ids, response_ids):
    return {
        'prompt_ids': raw_prompt_ids,
        'raw_prompt_ids': raw_prompt_ids,
        'response_ids': response_ids,
    }


def _record_tokens(
    assembler, messages, output, spans, generation, response_ids, request_meta=None, request_kwargs=None
):
    texts = [text for text, _ in spans]
    message_ids = [ids for _, ids in spans]
    generation_text, generation_ids = generation
    render = PromptRender(
        text=''.join(texts) + generation_text,
        token_ids=np.array([token for ids in message_ids for token in ids] + generation_ids, dtype=np.int32),
        templated_messages=texts,
        message_token_ids=[np.array(part, dtype=np.int32) for part in message_ids],
        generation_prompt=generation_text,
        generation_prompt_ids=np.array(generation_ids, dtype=np.int32),
    )
    assembler.record_completion(
        messages=messages,
        output_message=output,
        request_kwargs=request_kwargs,
        token_training=_training(render.token_ids, response_ids),
        completion_text=output.get('content') or '',
        prompt_render=render,
        request_meta=request_meta,
    )


def _live_last_node(assembler):
    node_id = latest_node_id(assembler)
    return assembler.trajectory.nodes[node_id]


def _children(assembler, node):
    return [assembler.trajectory.nodes[node_id] for node_id in node.children]


def _segment_kind(node_list):
    boundary = node_list.response_nodes[0].segment_boundary
    return boundary.kind if boundary is not None and boundary.kind is not None else 'conversation'


class TestTokenContinuity:
    def test_partial_previous_token_account_keeps_ancestor_prompt_fallback(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')
        user = {'role': 'user', 'content': 'start'}
        reply = {'role': 'assistant', 'content': 'first'}
        _record_tokens(assembler, [user], reply, [('U', [1])], ('A', [2]), [3])
        previous_id = latest_node_id(assembler)
        assembler.trajectory.nodes[previous_id].token_account = TokenAccount()
        request_facts = assembler._inspect_request([user, reply], None, None)

        boundary = assembler._physical_boundary(
            request_facts=request_facts,
            current_id=previous_id,
            token_training={'raw_prompt_ids': [1]},
        )

        assert boundary is None

        divergence = assembler._physical_boundary(
            request_facts=request_facts,
            current_id=previous_id,
            token_training={'raw_prompt_ids': [9]},
        )
        assert divergence is not None
        assert divergence.boundary.causes == ('render_divergence',)

    def test_exact_prefix_keeps_turns_in_one_training_chunk(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')
        user = {'role': 'user', 'content': 'start'}
        first = {'role': 'assistant', 'content': 'first'}
        _record_tokens(
            assembler,
            [user],
            first,
            [('U', [1])],
            ('A', [2]),
            [3],
        )
        _record_tokens(
            assembler,
            [user, first, {'role': 'user', 'content': 'continue'}],
            {'role': 'assistant', 'content': 'second'},
            [('U', [1]), ('Afirst', [2, 3]), ('Q', [])],
            ('A', [4]),
            [5],
        )

        trajectory = freeze_assembler(assembler)
        node_lists = trajectory.to_segment_node_lists()
        records = trajectory.prepare_segment_training_data((node_list, True) for node_list in node_lists)

        assert len(node_lists) == 1
        assert len(node_lists[0].response_nodes) == 2
        assert next(iter(records.values())).response_length == 3
        assert next(iter(records.values())).generated_ranges == ((0, 1), (2, 3))

    def test_equivalent_token_render_reuses_the_semantic_path(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')
        user = {'role': 'user', 'content': 'start'}
        first = {'role': 'assistant', 'content': 'first'}
        _record_tokens(assembler, [user], first, [('U', [1])], ('A', [2]), [3])

        _record_tokens(
            assembler,
            [user, first, {'role': 'user', 'content': 'continue'}],
            {'role': 'assistant', 'content': 'second'},
            [('U re-rendered', [1]), ('Afirst re-rendered', [2, 3]), ('Q', [4])],
            ('A', [5]),
            [6],
        )

        node_lists = freeze_assembler(assembler).to_segment_node_lists()

        assert len(assembler.trajectory.roots) == 1
        root = assembler.trajectory.nodes[assembler.trajectory.roots[0]]
        assert root.templated_message == 'U'
        assert assembler.trajectory.nodes[root.children[0]].templated_message == 'Afirst'
        assert len(node_lists) == 1
        assert len(node_lists[0].response_nodes) == 2

    def test_non_prefix_prompt_splits_one_semantic_segment_into_safe_chunks(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')
        user = {'role': 'user', 'content': 'start'}
        first = {'role': 'assistant', 'content': 'first'}
        _record_tokens(
            assembler,
            [user],
            first,
            [('U', [1])],
            ('A', [2]),
            [3],
        )
        _record_tokens(
            assembler,
            [user, first, {'role': 'user', 'content': 'continue'}],
            {'role': 'assistant', 'content': 'second'},
            [('U2', [90]), ('first', []), ('Q', [])],
            ('A', [91]),
            [92],
        )

        node_lists = freeze_assembler(assembler).to_segment_node_lists()

        assert [len(node_list.response_nodes) for node_list in node_lists] == [1, 1]
        assert [_segment_kind(node_list) for node_list in node_lists] == ['conversation', 'render_restart']

    def test_consumed_render_is_not_extended_retroactively(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')
        user = {'role': 'user', 'content': 'start'}
        first = {'role': 'assistant', 'content': 'first'}
        second_user = {'role': 'user', 'content': 'second question'}
        _record_tokens(assembler, [user], first, [('U', [1])], ('A', [2]), [3])
        _record_tokens(
            assembler,
            [user, first, second_user],
            {'role': 'assistant', 'content': 'second'},
            [('U', [1]), ('Afirst', [2, 3]), ('Q2', [4])],
            ('A', [5]),
            [6],
        )

        _record_tokens(
            assembler,
            [user, first, {'role': 'user', 'content': 'third question'}],
            {'role': 'assistant', 'content': 'third'},
            [('U', [1]), ('Afirst!', [2, 3, 7]), ('Q3', [8])],
            ('A', [9]),
            [10],
        )

        user_node = assembler.trajectory.nodes[assembler.trajectory.roots[0]]
        first_nodes = [node for node in _children(assembler, user_node) if node.original_message == first]
        assert len(first_nodes) == 2
        original = next(node for node in first_nodes if node.generated_by == 'llm')
        assert original.token_account is not None
        assert original.token_account.token_ids.tolist() == [2, 3]
        node_lists = freeze_assembler(assembler).to_segment_node_lists()
        assert _segment_kind(node_lists[-1]) == 'conversation'
        assert _live_last_node(assembler).segment_boundary is None

    def test_closure_frames_the_next_span_and_never_extends_a_retry(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')
        user = {'role': 'user', 'content': 'start'}
        reply = {'role': 'assistant', 'content': 'same'}
        _record_tokens(assembler, [user], reply, [('U', [1])], ('A', [2]), [3])
        _record_tokens(assembler, [user], reply, [('U', [1])], ('A', [2]), [3])

        user_node = assembler.trajectory.nodes[assembler.trajectory.roots[0]]
        retries = _children(assembler, user_node)
        assert len(retries) == 2
        # The block closure rides at the head of the next span (leading framing);
        # the retry's frozen span is resent byte for byte.
        _record_tokens(
            assembler,
            [user, reply, {'role': 'user', 'content': 'continue'}],
            {'role': 'assistant', 'content': 'next'},
            [('U', [1]), ('Asame', [2, 3]), ('!Q', [4, 5])],
            ('A', [6]),
            [7],
        )

        assert retries[0].token_account is not None
        assert retries[1].token_account is not None
        assert retries[0].token_account.token_ids.tolist() == [2, 3]
        assert retries[1].token_account.token_ids.tolist() == [2, 3]
        continuation = next(node for retry in retries for node in _children(assembler, retry))
        assert continuation.original_message['content'] == 'continue'
        assert continuation.templated_message == '!Q'
        assert continuation.token_account.token_ids.tolist() == [4, 5]

    def test_continuation_stays_on_the_matching_declared_chain(self):
        assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
        user = {'role': 'user', 'content': 'start'}
        reply = {'role': 'assistant', 'content': 'same'}
        worker_1 = {'declared_chain': {'chain_id': 'worker-1', 'kind': 'worker'}}
        worker_2 = {'declared_chain': {'chain_id': 'worker-2', 'kind': 'worker'}}
        _record_tokens(assembler, [user], reply, [('U', [1])], ('A', [2]), [3], worker_1)
        _record_tokens(assembler, [user], reply, [('U', [1])], ('A', [2]), [3, 4], worker_2)

        # worker-1's continuation resends worker-1's frozen span byte for byte;
        # the physically longer worker-2 realization must not attract it.
        _record_tokens(
            assembler,
            [user, reply, {'role': 'user', 'content': 'continue'}],
            {'role': 'assistant', 'content': 'next'},
            [('U', [1]), ('Asame', [2, 3]), ('!Q', [5, 6])],
            ('A', [7]),
            [8],
            worker_1,
        )

        user_node = assembler.trajectory.nodes[assembler.trajectory.roots[0]]
        by_worker = {}
        for node_id in user_node.children:
            memberships = assembler._runtime_state.facts.by_node[node_id].facts_of_type(SidechainMembership)
            if memberships:
                by_worker[memberships[0][1].chain_id] = assembler.trajectory.nodes[node_id]
        assert by_worker['worker-1'].token_account.token_ids.tolist() == [2, 3]
        assert by_worker['worker-2'].token_account.token_ids.tolist() == [2, 3, 4]
        assert [child.original_message['content'] for child in _children(assembler, by_worker['worker-1'])] == [
            'continue'
        ]
        assert by_worker['worker-2'].children == []

    def test_message_backend_splits_when_render_context_changes(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')
        user = {'role': 'user', 'content': 'start'}
        first = {'role': 'assistant', 'content': 'first'}
        assembler.record_completion(
            messages=[user],
            output_message=first,
            request_kwargs={'tools': [{'type': 'function', 'function': {'name': 'first_tool'}}]},
        )
        assembler.record_completion(
            messages=[user, first, {'role': 'user', 'content': 'continue'}],
            output_message={'role': 'assistant', 'content': 'second'},
            request_kwargs={'tools': [{'type': 'function', 'function': {'name': 'different_tool'}}]},
        )

        node_lists = freeze_assembler(assembler).to_segment_node_lists()

        assert [len(node_list.response_nodes) for node_list in node_lists] == [1, 1]
        assert [_segment_kind(node_list) for node_list in node_lists] == ['conversation', 'render_restart']
        assert 'render_context_discontinuity' in _live_last_node(assembler).segment_boundary.causes

    def test_message_backend_never_continues_across_cache_contexts(self):
        class RequestPolicy(GenericScaffoldPolicy):
            def inspect_event(self, event):
                facts = super().inspect_event(event)
                request = event.request if isinstance(event, ResponseEvent) else event
                if not isinstance(request, RequestEvent):
                    return facts
                request_id = (request.request_meta or {}).get('request_id')
                return replace(facts, context_key=request_id)

        assembler = TrajectoryAssembler(scaffold_policy=RequestPolicy(policy_id='request-keyed'))
        user = {'role': 'user', 'content': 'start'}
        first = {'role': 'assistant', 'content': 'first'}
        assembler.record_completion(
            messages=[user],
            output_message=first,
            request_meta={'request_id': 'request-1'},
        )
        assembler.record_completion(
            messages=[user, first, {'role': 'user', 'content': 'continue'}],
            output_message={'role': 'assistant', 'content': 'second'},
            request_meta={'request_id': 'request-2'},
        )

        node_lists = freeze_assembler(assembler).to_segment_node_lists()

        assert [len(node_list.response_nodes) for node_list in node_lists] == [1, 1]
        assert node_lists[1].response_nodes[0].segment_boundary.causes == ('render_context_discontinuity',)

    def test_mixed_token_and_message_turns_split_without_dropping_a_response(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')
        user = {'role': 'user', 'content': 'start'}
        first = {'role': 'assistant', 'content': 'token response'}
        second = {'role': 'assistant', 'content': 'message response'}
        _record_tokens(
            assembler,
            [user],
            first,
            [('U', [1])],
            ('A', [2]),
            [3],
        )
        assembler.record_completion(
            messages=[user, first, {'role': 'user', 'content': 'continue'}],
            output_message=second,
        )

        node_lists = freeze_assembler(assembler).to_segment_node_lists()

        assert [len(node_list.response_nodes) for node_list in node_lists] == [1, 1]
        assert node_lists[0].response_nodes[0].token_account.generated_ids.tolist() == [3]
        assert node_lists[1].response_nodes[-1].original_message == second
        assert _segment_kind(node_lists[1]) == 'representation_restart'


class TestDivergenceAttribution:
    def test_truncated_resend_is_tagged_terminator_residue(self):
        policy = GenericScaffoldPolicy(policy_id='test-terminator-residue')
        policy.block_terminator_ids = (9,)
        assembler = TrajectoryAssembler(scaffold_policy=policy)
        user = {'role': 'user', 'content': 'start'}
        _record_tokens(assembler, [user], {'role': 'assistant', 'content': 'same'}, [('U', [1])], ('A', [2]), [3, 4])

        # The resend keeps only the sampled prefix [3]; the template closes the
        # unfinished turn with the declared block terminator 9.
        _record_tokens(
            assembler,
            [user, {'role': 'assistant', 'content': 'sa'}, {'role': 'user', 'content': 'continue'}],
            {'role': 'assistant', 'content': 'next'},
            [('U', [1]), ('Asa', [2, 3, 9]), ('Q', [5])],
            ('A', [6]),
            [7],
        )

        assert 'render_divergence' in _live_last_node(assembler).segment_boundary.causes
        assert _live_last_node(assembler).segment_boundary.evidence['divergence'] == 'terminator-residue'

    def test_tool_change_divergence_is_tagged_context_change(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')
        user = {'role': 'user', 'content': 'start'}
        first = {'role': 'assistant', 'content': 'first'}
        tools_a = [{'type': 'function', 'function': {'name': 'a'}}]
        tools_b = [{'type': 'function', 'function': {'name': 'b'}}]
        _record_tokens(assembler, [user], first, [('TaU', [11, 1])], ('A', [2]), [3], request_kwargs={'tools': tools_a})
        _record_tokens(
            assembler,
            [user, first, {'role': 'user', 'content': 'continue'}],
            {'role': 'assistant', 'content': 'second'},
            [('TbU', [12, 1]), ('Afirst', [2, 3]), ('Q', [5])],
            ('A', [6]),
            [7],
            request_kwargs={'tools': tools_b},
        )

        assert 'render_divergence' in _live_last_node(assembler).segment_boundary.causes
        assert _live_last_node(assembler).segment_boundary.evidence['divergence'] == 'context-change'

    def test_realization_fork_is_tagged_assumption_violation(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')
        user = {'role': 'user', 'content': 'start'}
        first = {'role': 'assistant', 'content': 'first'}
        _record_tokens(assembler, [user], first, [('U', [1])], ('A', [2]), [3])

        # The same semantic history re-renders to different bytes.
        _record_tokens(
            assembler,
            [user, first, {'role': 'user', 'content': 'continue'}],
            {'role': 'assistant', 'content': 'second'},
            [('U', [1]), ('Afirst?', [2, 99]), ('Q', [5])],
            ('A', [6]),
            [7],
        )

        assert 'render_divergence' in _live_last_node(assembler).segment_boundary.causes
        assert _live_last_node(assembler).segment_boundary.evidence['divergence'] == 'assumption-violation'

    def test_edited_history_is_tagged_history_rewrite(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')

        user = {'role': 'user', 'content': 'start'}
        first = {'role': 'assistant', 'content': 'first'}
        _record_tokens(assembler, [user], first, [('U', [1])], ('A', [2]), [3])

        _record_tokens(
            assembler,
            [user, {'role': 'assistant', 'content': 'edited'}, {'role': 'user', 'content': 'continue'}],
            {'role': 'assistant', 'content': 'second'},
            [('U', [1]), ('Aedited', [2, 40]), ('Q', [5])],
            ('A', [6]),
            [7],
        )

        boundary = _live_last_node(assembler).segment_boundary
        assert boundary.causes == ('history_rewrite',)
        assert boundary.evidence['divergence'] == 'history-rewrite'
        assert boundary.evidence['path_divergence_message_index'] == 1

    def test_accumulated_assistant_turn_is_tagged_history_rewrite(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')
        system = {'role': 'system', 'content': 'policy'}
        user = {'role': 'user', 'content': 'start'}
        first = {
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                {'id': 'call-a', 'type': 'function', 'function': {'name': 'A', 'arguments': '{}'}},
                {'id': 'call-b', 'type': 'function', 'function': {'name': 'B', 'arguments': '{}'}},
            ],
        }
        _record_tokens(assembler, [system, user], first, [('S', [1]), ('U', [2])], ('A', [3]), [4])

        accumulated = {
            **first,
            'tool_calls': [
                *first['tool_calls'],
                {'id': 'call-c', 'type': 'function', 'function': {'name': 'C', 'arguments': '{}'}},
            ],
        }

        _record_tokens(
            assembler,
            [system, user, accumulated, {'role': 'tool', 'tool_call_id': 'call-c', 'content': 'done'}],
            {'role': 'assistant', 'content': 'second'},
            [('S', [1]), ('U', [2]), ('A accumulated', [40]), ('T', [5])],
            ('A', [6]),
            [7],
        )

        boundary = _live_last_node(assembler).segment_boundary
        assert boundary.causes == ('history_rewrite',)
        assert boundary.evidence['divergence'] == 'history-rewrite'
        assert boundary.evidence['path_divergence_message_index'] == 2
