from __future__ import annotations

import asyncio
import copy
import importlib.util

from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from _fakes import memory_store
from tests.unit.gateway.helpers import latest_node_id

from nexus.gateway.runtime_state import RuntimeFactsLedger, RuntimeState
from nexus.gateway.scaffold_policy import (
    AnchoredIdentity,
    BoundAnchor,
    BoundaryFact,
    ChildPromptCandidate,
    DeliveryReceipt,
    EventFacts,
    GenericScaffoldPolicy,
    InputMessage,
    NotificationDelivery,
    ObservationEvent,
    ObservationMessage,
    OutputMessage,
    RequestEvent,
    ResponseEvent,
    ResponsePolicy,
    RetrievalIntent,
    SidechainMembership,
    SpawnIntent,
    build_scaffold_policy,
    protocol_message_identity,
    register_scaffold_policy,
)
from nexus.gateway.session import GatewaySession
from nexus.gateway.trajectory_assembler import TrajectoryAssembler
from nexus.gateway.trajectory_freeze import freeze_trajectory
from nexus.gateway.types import PromptRender


def _non_dfs_snapshot_pair():
    class SnapshotPolicy(GenericScaffoldPolicy):
        def inspect_event(self, event):
            facts = super().inspect_event(event)
            if not isinstance(event, RequestEvent):
                return facts
            anchors = tuple(InputMessage(index) for index in range(len(event.messages)))
            boundary_facts = tuple(
                BoundaryFact(
                    rule_name='canonical-boundary',
                    anchor=anchor,
                    segment_kind='conversation',
                    cause='explicit',
                )
                for anchor, message in zip(anchors, event.messages, strict=True)
                if message.get('_boundary')
            )
            return replace(
                facts,
                spawn_intents=tuple(
                    SpawnIntent(anchor, f'spawn-{index}', f'prompt-{index}') for index, anchor in enumerate(anchors)
                ),
                retrieval_intents=tuple(
                    RetrievalIntent(anchor, f'poll-{index}', f'child-{index}') for index, anchor in enumerate(anchors)
                ),
                delivery_receipts=tuple(
                    DeliveryReceipt(anchor, f'delivery-{index}', True) for index, anchor in enumerate(anchors)
                ),
                notification_deliveries=tuple(
                    NotificationDelivery(anchor, f'child-{index}') for index, anchor in enumerate(anchors)
                ),
                child_prompts=tuple(
                    ChildPromptCandidate(anchor, f'prompt-{index}', index) for index, anchor in enumerate(anchors)
                ),
                boundary_facts=boundary_facts,
            )

    assembler = TrajectoryAssembler(
        scaffold_policy=SnapshotPolicy(policy_id='test-snapshot-pair'),
        session_id='snapshot-pair',
    )
    root_a = {'role': 'user', 'content': 'root-a'}
    reply_a = {'role': 'assistant', 'content': 'reply-a'}
    root_b = {'role': 'user', 'content': 'root-b'}
    reply_b = {'role': 'assistant', 'content': 'reply-b'}
    continuation_a = {'role': 'user', 'content': 'continuation-a', '_boundary': True}
    final_a = {'role': 'assistant', 'content': 'final-a'}
    assembler.record_completion([root_a], reply_a)
    assembler.record_completion([root_b], reply_b)
    assembler.record_completion([root_a, reply_a, continuation_a], final_a)
    assembler.finalize_segment()
    return assembler


class ConflictingBoundaryPolicy(GenericScaffoldPolicy):
    def inspect_event(self, event):
        facts = super().inspect_event(event)
        if not isinstance(event, RequestEvent):
            return facts
        boundary_index = next(
            (index for index in range(len(event.messages) - 1, -1, -1) if event.messages[index].get('_boundary')),
            None,
        )
        if boundary_index is None:
            return facts
        anchor = InputMessage(boundary_index)
        return replace(
            facts,
            boundary_facts=(
                BoundaryFact('summary', anchor, 'summary_inference', 'compaction'),
                BoundaryFact('resume', anchor, 'compaction_resume', 'compaction'),
            ),
        )


def test_scaffold_policy_selection_is_explicit_and_duplicate_registration_fails():
    with pytest.raises(ValueError, match='requires an explicit scaffold_policy'):
        build_scaffold_policy(None)
    with pytest.raises(ValueError, match="Unknown gateway scaffold_policy='missing'"):
        build_scaffold_policy('missing')

    policy = GenericScaffoldPolicy(policy_id='test-explicit-policy')
    register_scaffold_policy(policy)
    assert build_scaffold_policy('test-explicit-policy') is policy
    with pytest.raises(ValueError, match="Duplicate gateway scaffold_policy='test-explicit-policy'"):
        register_scaffold_policy(policy)

    assert build_scaffold_policy('generic').policy_id == 'generic'
    assert build_scaffold_policy('nexus_main_agent').policy_id == 'nexus_main_agent'

    with pytest.raises(ValueError, match='requires an explicit scaffold_policy'):
        TrajectoryAssembler()
    with pytest.raises(ValueError, match='requires an explicit scaffold_policy'):
        GatewaySession('missing-policy', object(), memory_store('missing-policy'))


def test_retired_tree_policy_modules_and_constructor_alias_are_absent():
    for module_name in (
        'nexus.gateway.assembly_tree',
        'nexus.gateway.materializer',
        'nexus.gateway.trace_policy',
        'nexus.gateway.trace_policy_engine',
    ):
        assert importlib.util.find_spec(module_name) is None

    assert importlib.util.find_spec('nexus.gateway.trajectory_freeze') is not None
    with pytest.raises(TypeError, match='trace_policy'):
        TrajectoryAssembler(trace_policy='generic')


def test_scaffold_policy_protocol_is_stateless_and_event_only():
    policy = GenericScaffoldPolicy(policy_id='test-event-only')
    assert policy.response_policy is ResponsePolicy.KEEP_ALL
    assert policy.block_terminator_ids == ()
    assert policy.message_key == 'protocol-message-v1'
    assert policy.tools_key == 'exact-tools-v1'

    facts = policy.inspect_event(
        RequestEvent(
            messages=({'role': 'user', 'content': 'hello'},),
            tools=None,
            request_meta=None,
        )
    )

    assert isinstance(facts, EventFacts)
    assert 'trajectory' not in {item.name for item in fields(EventFacts)}
    assert 'runtime_state' not in {item.name for item in fields(EventFacts)}


def test_trajectory_assembler_context_key_public_api_returns_inspected_context_value():
    assembler = TrajectoryAssembler(scaffold_policy='generic')
    context = assembler.extract_context(
        messages=[{'role': 'user', 'content': 'hello'}],
        request_meta={'request_id': 'request-1'},
    )

    assert hasattr(assembler, 'context_key')
    assert assembler.context_key(context) == ('generic', 'main', False)


def test_generic_policy_identity_ignores_decorations_and_canonicalizes_tool_arguments():
    policy = GenericScaffoldPolicy(policy_id='test-protocol-identity')
    decorated = {
        'role': 'assistant',
        'content': None,
        'tool_calls': [
            {
                'id': 'call-1',
                'index': 0,
                'type': 'function',
                'function': {'name': 'read', 'arguments': '{"b": 2, "a": 1}'},
            }
        ],
        'provider_specific_fields': {'refusal': None},
    }
    resent = {
        'role': 'assistant',
        'content': None,
        'tool_calls': [
            {
                'id': 'call-1',
                'type': 'function',
                'function': {'name': 'read', 'arguments': '{"a": 1, "b": 2}'},
            }
        ],
    }

    first = policy.inspect_event(RequestEvent((decorated,), None, None)).message_identities[0].value
    second = policy.inspect_event(RequestEvent((resent,), None, None)).message_identities[0].value
    assert first == second

    empty_content = {**resent, 'content': ''}
    assert protocol_message_identity(empty_content) == protocol_message_identity(resent)

    assert (
        policy.inspect_event(RequestEvent((), None, None)).tools_identity
        != policy.inspect_event(RequestEvent((), [], None)).tools_identity
    )


def test_input_message_anchors_not_policy_tuple_order_drive_runtime_path_matching():
    class ReverseIdentityPolicy(GenericScaffoldPolicy):
        def inspect_event(self, event):
            facts = super().inspect_event(event)
            if isinstance(event, RequestEvent):
                return replace(facts, message_identities=tuple(reversed(facts.message_identities)))
            return facts

    def record_tokens(assembler, messages, output, spans, generation, response_ids):
        generation_text, generation_ids = generation
        render = PromptRender(
            text=''.join(text for text, _ in spans) + generation_text,
            token_ids=np.array(
                [token for _, token_ids in spans for token in token_ids] + generation_ids, dtype=np.int32
            ),
            templated_messages=[text for text, _ in spans],
            message_token_ids=[np.array(token_ids, dtype=np.int32) for _, token_ids in spans],
            generation_prompt=generation_text,
            generation_prompt_ids=np.array(generation_ids, dtype=np.int32),
        )
        assembler.record_completion(
            messages=messages,
            output_message=output,
            token_training={
                'prompt_ids': render.token_ids,
                'raw_prompt_ids': render.token_ids,
                'response_ids': response_ids,
            },
            completion_text=output['content'],
            prompt_render=render,
        )

    assembler = TrajectoryAssembler(
        scaffold_policy=ReverseIdentityPolicy(policy_id='test-reverse-input-identities'),
    )
    user = {'role': 'user', 'content': 'start'}
    first = {'role': 'assistant', 'content': 'first'}
    followup = {'role': 'user', 'content': 'continue'}
    final = {'role': 'assistant', 'content': 'final'}
    history = [user, first, followup]
    record_tokens(assembler, [user], first, [('U1', [1])], ('A1', [2]), [3])
    record_tokens(
        assembler,
        history,
        final,
        [('U2', [10]), ('first', [2, 3]), ('Q', [4])],
        ('A2', [5]),
        [6],
    )

    assert [assembler.trajectory.nodes[node_id].message_id for node_id in assembler.trajectory.roots] == [0, 0]
    assert assembler.frozen_prefix(history) is not None

    assembler.record_observation('result', parent_messages=history)
    observation_id = latest_node_id(assembler)
    assert assembler.trajectory.nodes[observation_id].parent == 4


def test_frozen_prefix_ignores_a_semantic_match_without_physical_bytes():
    assembler = TrajectoryAssembler(scaffold_policy='generic')
    system = {'role': 'system', 'content': 'shared system prompt'}
    user = {'role': 'user', 'content': 'original conversation'}
    output = {'role': 'assistant', 'content': 'answer'}
    render = PromptRender(
        text='rendered user prompt',
        token_ids=np.array([10, 11], dtype=np.int32),
        templated_messages=['', 'rendered user prompt'],
        message_token_ids=[np.empty(0, dtype=np.int32), np.array([10, 11], dtype=np.int32)],
        generation_prompt='',
        generation_prompt_ids=np.array([], dtype=np.int32),
    )
    assembler.record_completion(
        messages=[system, user],
        output_message=output,
        token_training={
            'prompt_ids': [10, 11],
            'raw_prompt_ids': [10, 11],
            'response_ids': [12],
        },
        completion_text='answer',
        prompt_render=render,
    )

    compacted = [system, {'role': 'user', 'content': 'compacted summary'}]
    assert assembler.frozen_prefix(compacted) is None


@pytest.mark.parametrize(
    'observation_kwargs',
    [
        {},
        {'request_meta': {}},
        {'tools': []},
        {'start_time': 2.0, 'end_time': 3.0, 'status': 'OK'},
    ],
)
def test_observation_without_parent_messages_is_rejected(observation_kwargs):
    assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
    assembler.record_completion(
        messages=[{'role': 'user', 'content': 'prompt'}],
        output_message={'role': 'assistant', 'content': 'answer'},
    )

    with pytest.raises(ValueError, match='requires parent_messages'):
        assembler.record_observation('implicit', **observation_kwargs)
    assert len(assembler.trajectory.nodes) == 2


def test_request_context_can_be_reused_for_observation_without_reusing_request_anchors():
    assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
    user = {'role': 'user', 'content': 'prompt'}
    reply = {'role': 'assistant', 'content': 'answer'}
    context = assembler.extract_context([user], request_meta={'request_id': 'request-1'})
    assembler.record_completion([user], reply, context=context)

    assembler.record_observation(
        'result',
        parent_messages=[user, reply],
        context=context,
        start_time=2.0,
        end_time=3.0,
        status='OK',
    )

    observation_id = latest_node_id(assembler)
    observation = assembler.trajectory.nodes[observation_id]
    assert observation.original_message == {'role': 'tool', 'content': 'result'}
    assert observation.timeline == [
        {
            'event_type': 'observation',
            'request_id': 'request-1',
            'is_sidechain': False,
            'start_time': 2.0,
            'end_time': 3.0,
            'duration': 1.0,
            'status': 'OK',
            'attributes': {'type': 'agent_observation'},
        }
    ]


def test_supplied_context_scopes_explicit_observation_parent_lookup():
    assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
    user = {'role': 'user', 'content': 'child prompt'}
    reply = {'role': 'assistant', 'content': 'child answer'}
    meta = {'declared_chain': {'chain_id': 'child', 'parent_chain_id': 'main', 'kind': 'worker'}}
    context = assembler.extract_context([user], request_meta=meta)
    assembler.record_completion([user], reply, context=context)

    assembler.record_observation('result', parent_messages=[user, reply], context=context)

    observation_id = latest_node_id(assembler)
    assert assembler.trajectory.nodes[observation_id].parent == 1
    assert assembler._runtime_state.facts.by_node[observation_id].context_key == context.context_key


def test_supplied_context_never_replaces_current_message_identity_or_prefix():
    assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
    stale = {'role': 'user', 'content': 'stale'}
    actual = {'role': 'user', 'content': 'actual'}
    reply = {'role': 'assistant', 'content': 'answer'}
    context = assembler.extract_context([stale])
    render = PromptRender(
        text='actual',
        token_ids=np.array([10], dtype=np.int32),
        templated_messages=['actual'],
        message_token_ids=[np.array([10], dtype=np.int32)],
        generation_prompt='',
        generation_prompt_ids=np.array([], dtype=np.int32),
    )
    assembler.record_completion(
        [actual],
        reply,
        token_training={'prompt_ids': [10], 'raw_prompt_ids': [10], 'response_ids': [11]},
        completion_text='answer',
        prompt_render=render,
        context=context,
    )

    actual_prefix = assembler.frozen_prefix([actual])
    assert actual_prefix is not None
    assert actual_prefix.count == 1
    assert assembler.frozen_prefix([stale]) is None

    assembler.record_completion([actual], {'role': 'assistant', 'content': 'second'})
    assert len(assembler.trajectory.roots) == 1


def test_supplied_lane_ownership_applies_to_response_while_event_local_facts_use_current_message():
    class LocalFactPolicy(GenericScaffoldPolicy):
        def inspect_event(self, event):
            facts = super().inspect_event(event)
            if not isinstance(event, RequestEvent):
                return facts
            return replace(
                facts,
                child_prompts=tuple(
                    ChildPromptCandidate(InputMessage(index), message['content'], index)
                    for index, message in enumerate(event.messages)
                ),
            )

    policy = LocalFactPolicy(policy_id='context-projection')
    assembler = TrajectoryAssembler(scaffold_policy=policy)
    stale = {'role': 'user', 'content': 'stale'}
    actual = {'role': 'user', 'content': 'actual'}
    context = replace(
        assembler.extract_context([stale]),
        context_key=('context-projection', 'child', True),
        membership=build_scaffold_policy('nexus_main_agent')
        .inspect_event(
            RequestEvent(
                (stale,),
                None,
                {'declared_chain': {'chain_id': 'child', 'parent_chain_id': 'main', 'kind': 'worker'}},
            )
        )
        .membership,
    )
    assembler.record_completion([actual], {'role': 'assistant', 'content': 'answer'}, context=context)

    input_facts = assembler._runtime_state.facts.by_node[0]
    output_facts = assembler._runtime_state.facts.by_node[1]
    assert input_facts.facts_of_type(ChildPromptCandidate)[0][1].value == 'actual'
    assert output_facts.context_key == context.context_key
    assert output_facts.facts_of_type(type(context.membership))[0][1] == context.membership


def test_supplied_message_derived_context_is_authoritative_for_response_and_observation_timeline():
    class MessageContextPolicy(GenericScaffoldPolicy):
        def inspect_event(self, event):
            facts = super().inspect_event(event)
            request = event.request if isinstance(event, ResponseEvent) else event
            if isinstance(request, RequestEvent):
                label = request.messages[0]['content'] if request.messages else 'empty'
                is_child = label == 'child-context'
                membership = SidechainMembership('child', 'worker', 'main', 'message-context') if is_child else None
                timeline = dict(facts.timeline)
                if timeline:
                    timeline['request_id'] = f'request:{label}'
                    timeline['is_sidechain'] = is_child
                return replace(
                    facts,
                    context_key=('message-context', label, is_child),
                    request_id=f'request:{label}',
                    membership=membership,
                    timeline=tuple(timeline.items()),
                )
            return facts

    assembler = TrajectoryAssembler(scaffold_policy=MessageContextPolicy(policy_id='message-context'))
    context = assembler.extract_context([{'role': 'user', 'content': 'child-context'}])
    history = [{'role': 'user', 'content': 'actual-main-message'}, {'role': 'assistant', 'content': 'answer'}]
    assembler.record_completion(history[:1], history[1], context=context)
    assembler.record_observation('result', parent_messages=history, context=context)

    llm_timeline = assembler.trajectory.nodes[1].timeline[0]
    observation_id = latest_node_id(assembler)
    observation_timeline = assembler.trajectory.nodes[observation_id].timeline[0]
    for timeline in (llm_timeline, observation_timeline):
        assert timeline['request_id'] == 'request:child-context'
        assert timeline['is_sidechain'] is True


def test_supplied_context_does_not_carry_an_old_message_marker_to_the_current_request():
    assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
    old_summary = {'role': 'assistant', 'content': 'old', '_is_compacted_summary': True}
    context = assembler.extract_context([old_summary])

    assembler.record_completion(
        [{'role': 'user', 'content': 'plain'}],
        {'role': 'assistant', 'content': 'answer'},
        context=context,
    )

    output_id = latest_node_id(assembler)
    assert assembler.trajectory.nodes[output_id].segment_boundary is None


def test_zero_message_request_materializes_the_boundary_from_supplied_context():
    assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
    context = assembler.extract_context([], request_meta={'invocation_kind': 'summary_inference'})

    assembler.record_completion([], {'role': 'assistant', 'content': 'summary'}, context=context)

    output_id = latest_node_id(assembler)
    boundary = assembler.trajectory.nodes[output_id].segment_boundary
    assert boundary is not None
    assert boundary.kind == 'summary_inference'
    assert boundary.causes == ('compaction',)
    assert boundary.evidence == {'boundary_rules': ['nexus_summary_inference']}


@pytest.mark.parametrize(
    ('delivery_kind', 'delivery_fact_type'),
    [('inline', DeliveryReceipt), ('notify', NotificationDelivery)],
)
def test_observation_keeps_anchored_delivery_facts_for_freeze(delivery_kind, delivery_fact_type):
    class ObservationDeliveryPolicy(GenericScaffoldPolicy):
        def inspect_event(self, event):
            facts = super().inspect_event(event)
            if isinstance(event, ObservationEvent):
                if delivery_kind == 'inline':
                    return replace(
                        facts,
                        delivery_receipts=(DeliveryReceipt(ObservationMessage(), 'spawn-1', True),),
                    )
                return replace(
                    facts,
                    notification_deliveries=(NotificationDelivery(ObservationMessage(), 'child'),),
                )

            request = event.request if isinstance(event, ResponseEvent) else event
            if not isinstance(request, RequestEvent):
                return facts
            meta = request.request_meta or {}
            is_child = meta.get('agent_id') == 'child'
            membership = (
                SidechainMembership('child', 'worker', 'main', 'test-contextless-delivery') if is_child else None
            )
            context_key = ('observation-delivery', 'child' if is_child else 'main', is_child)
            if isinstance(event, ResponseEvent):
                spawn_intents = (
                    (SpawnIntent(OutputMessage(), 'spawn-1', 'child prompt'),)
                    if event.message.get('content') == 'spawn child'
                    else ()
                )
                return replace(
                    facts,
                    context_key=context_key,
                    membership=membership,
                    spawn_intents=spawn_intents,
                )
            return replace(
                facts,
                context_key=context_key,
                membership=membership,
                child_prompts=tuple(
                    ChildPromptCandidate(InputMessage(index), message.get('content') or '', index)
                    for index, message in enumerate(event.messages)
                ),
            )

    assembler = TrajectoryAssembler(
        scaffold_policy=ObservationDeliveryPolicy(policy_id='observation-delivery'),
    )
    assembler.record_completion(
        [{'role': 'user', 'content': 'child prompt'}],
        {'role': 'assistant', 'content': 'child done'},
        request_meta={'agent_id': 'child'},
    )
    main_history = [{'role': 'user', 'content': 'main'}, {'role': 'assistant', 'content': 'spawn child'}]
    assembler.record_completion(main_history[:1], main_history[1])
    assembler.record_observation('delivered', parent_messages=main_history)

    observation_id = latest_node_id(assembler)
    facts = assembler._runtime_state.facts.by_node[observation_id]
    assert len(facts.facts_of_type(delivery_fact_type)) == 1

    trajectory = assembler.freeze(state='completed')
    child = trajectory.chains['child']
    assert child.join is not None
    assert child.join.kind == delivery_kind
    assert child.join.node == observation_id


def test_runtime_state_shape_has_no_shadow_topology_or_message_sidecar():
    assert {item.name for item in fields(RuntimeFactsLedger)} == {'by_node'}
    assert {item.name for item in fields(RuntimeState)} == {
        'facts',
        'llm_by_message_id',
        'next_node_id',
        'next_message_id',
        'next_request_sequence',
        'next_fact_sequence',
        'pending_boundary',
    }
    forbidden = {'message', 'messages', 'node', 'nodes', 'parent', 'children', 'root', 'roots', 'adjacency'}
    for runtime_type in (RuntimeFactsLedger, RuntimeState):
        assert forbidden.isdisjoint(item.name for item in fields(runtime_type))


def test_runtime_ledger_binds_every_correlation_fact_to_the_live_node_id():
    class CorrelationPolicy(GenericScaffoldPolicy):
        def inspect_event(self, event):
            facts = super().inspect_event(event)
            if isinstance(event, RequestEvent) and event.messages:
                return replace(
                    facts,
                    spawn_intents=(SpawnIntent(InputMessage(0), 'spawn-1', 'child prompt'),),
                    retrieval_intents=(RetrievalIntent(InputMessage(0), 'poll-1', 'child-1'),),
                    delivery_receipts=(DeliveryReceipt(InputMessage(0), 'spawn-1', True),),
                    notification_deliveries=(NotificationDelivery(InputMessage(0), 'child-1'),),
                    child_prompts=(ChildPromptCandidate(InputMessage(0), 'child prompt', 0),),
                )
            return facts

    assembler = TrajectoryAssembler(scaffold_policy=CorrelationPolicy(policy_id='test-bound-facts'))
    assembler.record_completion(
        messages=[{'role': 'user', 'content': 'prompt'}],
        output_message={'role': 'assistant', 'content': 'answer'},
    )

    node_facts = assembler._runtime_state.facts.by_node[0]
    for fact_type in (SpawnIntent, RetrievalIntent, DeliveryReceipt, NotificationDelivery, ChildPromptCandidate):
        facts = node_facts.facts_of_type(fact_type)
        assert len(facts) == 1
        assert facts[0][1].anchor == BoundAnchor(0)
    assert node_facts.facts_of_type(ChildPromptCandidate)[0][1].order == 0


@pytest.mark.parametrize('invalid_fact_kind', ['spawn', 'boundary'])
def test_policy_anchored_facts_must_bind_to_an_identity_from_the_same_event(invalid_fact_kind):
    class InvalidAnchorPolicy(GenericScaffoldPolicy):
        def inspect_event(self, event):
            facts = super().inspect_event(event)
            if not isinstance(event, RequestEvent):
                return facts
            if invalid_fact_kind == 'spawn':
                return replace(
                    facts,
                    spawn_intents=(SpawnIntent(InputMessage(99), 'lost-spawn', 'child prompt'),),
                )
            return replace(
                facts,
                boundary_facts=(BoundaryFact('lost-boundary', InputMessage(99), 'summary', 'compaction'),),
            )

    assembler = TrajectoryAssembler(scaffold_policy=InvalidAnchorPolicy(policy_id=f'invalid-{invalid_fact_kind}'))

    with pytest.raises(ValueError, match='anchor is not emitted by this event'):
        assembler.record_completion(
            messages=[{'role': 'user', 'content': 'prompt'}],
            output_message={'role': 'assistant', 'content': 'answer'},
        )

    assert assembler.trajectory.nodes == {}
    assert assembler._runtime_state.facts.by_node == {}


@pytest.mark.parametrize('invalid_event_kind', ['request', 'response', 'observation'])
def test_policy_message_identity_contract_is_event_specific(invalid_event_kind):
    class InvalidIdentityPolicy(GenericScaffoldPolicy):
        def inspect_event(self, event):
            facts = super().inspect_event(event)
            if invalid_event_kind == 'request' and isinstance(event, RequestEvent):
                return replace(facts, message_identities=(AnchoredIdentity(OutputMessage(), 'wrong'),))
            if invalid_event_kind == 'response' and isinstance(event, ResponseEvent):
                return replace(facts, message_identities=(AnchoredIdentity(InputMessage(0), 'wrong'),))
            if invalid_event_kind == 'observation' and isinstance(event, ObservationEvent):
                return replace(facts, message_identities=(AnchoredIdentity(OutputMessage(), 'wrong'),))
            return facts

    assembler = TrajectoryAssembler(scaffold_policy=InvalidIdentityPolicy(policy_id=f'invalid-{invalid_event_kind}'))
    if invalid_event_kind == 'observation':
        assembler.record_completion(
            messages=[{'role': 'user', 'content': 'prompt'}],
            output_message={'role': 'assistant', 'content': 'answer'},
        )

        def operation():
            assembler.record_observation(
                'result',
                parent_messages=[{'role': 'user', 'content': 'prompt'}, {'role': 'assistant', 'content': 'answer'}],
            )

    else:

        def operation():
            assembler.record_completion(
                messages=[{'role': 'user', 'content': 'prompt'}],
                output_message={'role': 'assistant', 'content': 'answer'},
            )

    with pytest.raises(ValueError, match=f'{invalid_event_kind.capitalize()}Event message identities'):
        operation()


def test_nexus_main_agent_policy_emits_only_declared_chain_and_compaction_facts():
    policy = build_scaffold_policy('nexus_main_agent')
    facts = policy.inspect_event(
        RequestEvent(
            messages=(
                {'role': 'user', 'content': 'parent'},
                {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        {
                            'id': 'spawn-1',
                            'type': 'function',
                            'function': {'name': 'spawn_subagent', 'arguments': '{"prompt":"child"}'},
                        }
                    ],
                },
                {'role': 'user', 'content': 'summary', '_is_compacted_summary': True},
            ),
            tools=None,
            request_meta={
                'declared_chain': {'chain_id': 'worker-1', 'kind': 'worker'},
                'parent_agent_id': 'implicit-parent-must-not-leak',
            },
        )
    )

    assert facts.membership is not None
    assert facts.membership.chain_id == 'worker-1'
    assert facts.declared_claim is not None
    assert facts.boundary_facts
    assert facts.spawn_intents == ()
    assert facts.retrieval_intents == ()
    assert facts.delivery_receipts == ()
    assert facts.notification_deliveries == ()
    assert facts.parent_claim is None


def test_freeze_preserves_runtime_node_ids_for_a_non_dfs_forest():
    assembler = _non_dfs_snapshot_pair()
    trajectory, runtime_state = copy.deepcopy((assembler.trajectory, assembler._runtime_state))
    assert trajectory.roots == [0, 2]
    assert trajectory.nodes[0].children == [1]
    assert trajectory.nodes[1].children == [4]
    assert trajectory.nodes[4].children == [5]
    assert trajectory.nodes[2].children == [3]
    semantic_facts = {
        facts.identity: trajectory.nodes[node_id].original_message['content']
        for node_id, facts in runtime_state.facts.by_node.items()
    }
    llm_by_message_id = dict(runtime_state.llm_by_message_id)
    scalar_counters = (
        runtime_state.next_node_id,
        runtime_state.next_message_id,
        runtime_state.next_request_sequence,
        runtime_state.next_fact_sequence,
    )
    pending_boundary = copy.deepcopy(runtime_state.pending_boundary)

    frozen = freeze_trajectory(
        trajectory,
        runtime_state,
        assembler._policy.response_policy,
        copy_inputs=False,
    )

    assert frozen is trajectory
    assert frozen.roots == [0, 2]
    assert [(node.node_id, node.parent, node.original_message['content']) for node in frozen.nodes.values()] == [
        (0, None, 'root-a'),
        (1, 0, 'reply-a'),
        (2, None, 'root-b'),
        (3, 2, 'reply-b'),
        (4, 1, 'continuation-a'),
        (5, 4, 'final-a'),
    ]
    assert frozen.chains['main'].roots == [0, 2]
    assert {
        facts.identity: frozen.nodes[node_id].original_message['content']
        for node_id, facts in runtime_state.facts.by_node.items()
    } == semantic_facts

    for node_id, facts in runtime_state.facts.by_node.items():
        anchored_occurrences = [
            occurrence.fact for occurrence in facts.occurrences if hasattr(occurrence.fact, 'anchor')
        ]
        assert all(fact.anchor == BoundAnchor(node_id) for fact in anchored_occurrences)
    assert runtime_state.facts.by_node[5].boundary_claims == (('canonical-boundary', 4),)
    assert runtime_state.llm_by_message_id == llm_by_message_id
    assert (
        runtime_state.next_node_id,
        runtime_state.next_message_id,
        runtime_state.next_request_sequence,
        runtime_state.next_fact_sequence,
    ) == scalar_counters
    assert runtime_state.pending_boundary == pending_boundary


def test_repeated_export_snapshots_keep_the_original_runtime_node_ids():
    assembler = _non_dfs_snapshot_pair()
    expected = [
        (0, None, 'root-a'),
        (1, 0, 'reply-a'),
        (2, None, 'root-b'),
        (3, 2, 'reply-b'),
        (4, 1, 'continuation-a'),
        (5, 4, 'final-a'),
    ]

    first = assembler.freeze()
    second = assembler.freeze()

    for snapshot in (first, second, assembler.trajectory):
        assert [
            (node.node_id, node.parent, node.original_message['content']) for node in snapshot.iter_nodes()
        ] == expected
        assert snapshot.roots == [0, 2]


def test_late_validation_failure_preserves_the_directly_committed_input_prefix():
    assembler = TrajectoryAssembler(
        scaffold_policy=ConflictingBoundaryPolicy(policy_id='test-nontransactional-boundary-conflict')
    )

    with pytest.raises(ValueError, match='incompatible segment kinds'):
        assembler.record_completion(
            messages=[{'role': 'user', 'content': 'prompt', '_boundary': True}],
            output_message={'role': 'assistant', 'content': 'answer'},
        )

    assert assembler.trajectory.roots == [0]
    assert list(assembler.trajectory.nodes) == [0]
    assert assembler.trajectory.nodes[0].original_message == {
        'role': 'user',
        'content': 'prompt',
        '_boundary': True,
    }
    assert list(assembler._runtime_state.facts.by_node) == [0]


@pytest.mark.asyncio
async def test_post_end_observation_finalize_and_repeat_end_keep_baseline_behavior():
    session = GatewaySession('ended', object(), memory_store('ended'), scaffold_policy='generic')
    session.end()

    with pytest.raises(ValueError, match='requires parent_messages'):
        session.record_observation('late')
    session.finalize_segment()
    trajectory = session.end('updated')

    assert trajectory.termination_reason == 'updated'
    assert list(trajectory.iter_nodes()) == []
    assert session.segment_assembler._runtime_state.pending_boundary is not None
    with pytest.raises(RuntimeError, match='already ended'):
        await session.create_completion([{'role': 'user', 'content': 'late'}])


@pytest.mark.asyncio
async def test_post_end_completion_rejects_before_copying_caller_wire_values():
    class CopyBomb:
        def __deepcopy__(self, _memo):
            raise AssertionError('must not copy an ended request')

    session = GatewaySession('ended-copy-order', object(), memory_store('ended-copy-order'), scaffold_policy='generic')
    session.end()

    with pytest.raises(RuntimeError, match='already ended'):
        await session.create_completion([{'role': 'user', 'content': CopyBomb()}])


@pytest.mark.asyncio
async def test_active_snapshot_is_committed_only_and_finish_rejects_in_flight_request():
    class ControlledProcessor:
        is_token_level = False

        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def create_completion(self, **_kwargs):
            self.started.set()
            await self.release.wait()
            return SimpleNamespace(
                message={'role': 'assistant', 'content': 'answer'},
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

    processor = ControlledProcessor()
    session = GatewaySession('in-flight', processor, memory_store('in-flight'), scaffold_policy='generic')
    completion = asyncio.create_task(session.create_completion([{'role': 'user', 'content': 'prompt'}]))
    await processor.started.wait()

    assert session.get_trajectory().nodes == {}
    with pytest.raises(RuntimeError, match='in-flight'):
        session.end()
    assert session.is_ended is False

    processor.release.set()
    await completion
    assert [node.original_message['content'] for node in session.end().iter_nodes()] == ['prompt', 'answer']
