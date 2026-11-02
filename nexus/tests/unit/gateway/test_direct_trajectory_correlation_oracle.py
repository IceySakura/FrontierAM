"""Baseline golden matrix for spawned-sidechain correlation.

The production Nexus/generic profiles intentionally no longer enable implicit
spawn inference.  This module instantiates the historical semantics locally so
the Direct-Trajectory resolver has an executable parity oracle without global
profile registration or production changes.
"""

from __future__ import annotations

import json

from dataclasses import replace

import pytest

from nexus.gateway.scaffold_policy import (
    BoundaryFact,
    ChildPromptCandidate,
    DeliveryReceipt,
    GenericScaffoldPolicy,
    InputMessage,
    NotificationDelivery,
    OutputMessage,
    ParentClaim,
    RequestEvent,
    ResponseEvent,
    ResponsePolicy,
    RetrievalIntent,
    SidechainMembership,
    SpawnIntent,
)
from nexus.gateway.trajectory_assembler import TrajectoryAssembler


SYSTEM = {'role': 'system', 'content': 'main system'}
USER = {'role': 'user', 'content': 'main task'}
CHILD_SYSTEM = {'role': 'system', 'content': 'child system'}
CHILD_PROMPT = 'inspect the repository'
CHILD_USER = {'role': 'user', 'content': CHILD_PROMPT}
LAUNCH_ACK = 'Async agent launched successfully'


class CorrelationPolicy(GenericScaffoldPolicy):
    response_policy = ResponsePolicy.PREFER_EXTENDED_ELSE_LAST

    @staticmethod
    def _text_blocks(content):
        if isinstance(content, str):
            return [content]
        if not isinstance(content, list):
            return []
        blocks = []
        for item in content:
            if isinstance(item, str):
                blocks.append(item)
            elif isinstance(item, dict):
                blocks.extend(CorrelationPolicy._text_blocks(item.get('text')))
                blocks.extend(CorrelationPolicy._text_blocks(item.get('content')))
        return blocks

    @staticmethod
    def _tool_argument(function, key):
        try:
            arguments = json.loads(function.get('arguments') or '{}')
        except ValueError:
            return ''
        value = arguments.get(key) if isinstance(arguments, dict) else None
        return value if isinstance(value, str) else ''

    @classmethod
    def _message_facts(cls, messages, anchors):
        spawns = []
        retrievals = []
        receipts = []
        notifications = []
        for message, anchor in zip(messages, anchors, strict=True):
            for call in message.get('tool_calls') or []:
                function = call.get('function') or {}
                call_id = call.get('id')
                if not isinstance(call_id, str):
                    continue
                if function.get('name') in ('Agent', 'Task'):
                    prompt = ' '.join(cls._tool_argument(function, 'prompt').split())
                    if prompt:
                        spawns.append(SpawnIntent(anchor, call_id, prompt))
                elif function.get('name') == 'TaskOutput':
                    agent_id = cls._tool_argument(function, 'task_id')
                    if agent_id:
                        retrievals.append(RetrievalIntent(anchor, call_id, agent_id))
            if message.get('role') == 'tool' and isinstance(message.get('tool_call_id'), str):
                blocks = cls._text_blocks(message.get('content'))
                text = '\n'.join(blocks)
                delivered = bool(blocks) and not any(block.lstrip().startswith(LAUNCH_ACK) for block in blocks)
                if '<retrieval_status>' in text:
                    delivered = '<retrieval_status>success' in text
                receipts.append(DeliveryReceipt(anchor, message['tool_call_id'], delivered))
            if message.get('role') == 'user':
                text = ' '.join(cls._text_blocks(message.get('content'))).strip()
                if (
                    text.startswith('<task-notification>')
                    and text.endswith('</task-notification>')
                    and '<status>completed' in text
                    and '<task-id>' in text
                    and '</task-id>' in text
                ):
                    agent_id = text.split('<task-id>', 1)[1].split('</task-id>', 1)[0]
                    notifications.append(NotificationDelivery(anchor, agent_id))
        return tuple(spawns), tuple(retrievals), tuple(receipts), tuple(notifications)

    def inspect_event(self, event):
        facts = super().inspect_event(event)
        request = event.request if isinstance(event, ResponseEvent) else event
        if not isinstance(request, RequestEvent):
            return facts
        request_facts = super().inspect_event(request)
        meta = request.request_meta or {}
        agent_id = str(meta['agent_id']) if meta.get('agent_id') is not None else None
        parent_agent_id = (
            str(meta.get('parent_agent_id') or meta.get('parent_id'))
            if meta.get('parent_agent_id') is not None or meta.get('parent_id') is not None
            else None
        )
        membership = (
            SidechainMembership(
                chain_id=agent_id,
                kind='worker',
                parent_chain_id=parent_agent_id,
                detector='context:agent_id',
                attributes=(('parent_agent_id', parent_agent_id),) if parent_agent_id is not None else (),
            )
            if agent_id is not None
            else None
        )
        context_key = (self.policy_id, agent_id, agent_id is not None)
        parent_claim = ParentClaim(parent_agent_id) if parent_agent_id is not None else None
        iteration = int(meta['iteration']) if meta.get('iteration') is not None else None
        retry = int(meta['retry']) if meta.get('retry') is not None else None
        if isinstance(event, ResponseEvent):
            anchors = (OutputMessage(),)
            message_facts = self._message_facts((event.message,), anchors)
            return replace(
                facts,
                tools_identity=request_facts.tools_identity,
                context_key=context_key,
                membership=membership,
                spawn_intents=message_facts[0],
                retrieval_intents=message_facts[1],
                delivery_receipts=message_facts[2],
                notification_deliveries=message_facts[3],
                parent_claim=parent_claim,
                iteration=iteration,
                retry=retry,
            )
        anchors = tuple(InputMessage(index) for index in range(len(event.messages)))
        message_facts = self._message_facts(event.messages, anchors)
        boundary_facts = ()
        if meta.get('invocation_kind') == 'compaction_resume' and anchors:
            boundary_facts = (
                BoundaryFact(
                    'test_compaction_resume',
                    anchors[-1],
                    'compaction_resume',
                    'compaction',
                    from_request_context=True,
                    at_request_edge=True,
                ),
            )
        child_prompts = ()
        if membership is not None:
            for index, message in enumerate(event.messages):
                if message.get('role') != 'user':
                    continue
                blocks = self._text_blocks(message.get('content'))
                prompt = next(
                    (block for block in blocks if not block.lstrip().startswith('<system-reminder>')),
                    blocks[0] if blocks else None,
                )
                if prompt is not None:
                    child_prompts = (ChildPromptCandidate(InputMessage(index), ' '.join(prompt.split()), 0),)
                break
        return replace(
            facts,
            context_key=context_key,
            membership=membership,
            spawn_intents=message_facts[0],
            retrieval_intents=message_facts[1],
            delivery_receipts=message_facts[2],
            notification_deliveries=message_facts[3],
            parent_claim=parent_claim,
            boundary_facts=boundary_facts,
            child_prompts=child_prompts,
            iteration=iteration,
            retry=retry,
        )


def _assembler() -> TrajectoryAssembler:
    return TrajectoryAssembler(scaffold_policy=CorrelationPolicy(policy_id='test-direct-trajectory-spawned'))


def _record(
    assembler: TrajectoryAssembler,
    messages: list[dict],
    output: dict,
    meta: dict | None = None,
) -> None:
    assembler.record_completion(messages=messages, output_message=output, request_meta=meta)


def _freeze(assembler: TrajectoryAssembler, *, state: str = 'completed'):
    return assembler.freeze(session_id='correlation-oracle', state=state)


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        'id': call_id,
        'type': 'function',
        'function': {'name': name, 'arguments': json.dumps(arguments, sort_keys=True)},
    }


def _spawn(*calls: tuple[str, str]) -> dict:
    return {
        'role': 'assistant',
        'content': '',
        'tool_calls': [_tool_call(call_id, 'Agent', {'prompt': prompt}) for call_id, prompt in calls],
    }


def _child(
    assembler: TrajectoryAssembler,
    *,
    agent_id: str = 'agent-1',
    prompt: str = CHILD_PROMPT,
    parent_agent_id: str | None = None,
    system: dict | None = None,
    reply: str = 'child done',
) -> None:
    meta = {'agent_id': agent_id}
    if parent_agent_id is not None:
        meta['parent_agent_id'] = parent_agent_id
    _record(
        assembler,
        [system or CHILD_SYSTEM, {'role': 'user', 'content': prompt}],
        {'role': 'assistant', 'content': reply},
        meta,
    )


def _notification(agent_id: str, result: str = 'delivered') -> dict:
    return {
        'role': 'user',
        'content': (
            f'<task-notification><task-id>{agent_id}</task-id>'
            f'<status>completed</status><result>{result}</result></task-notification>'
        ),
    }


@pytest.mark.parametrize(
    ('delivery', 'join_id'),
    [('inline', 'spawn-1'), ('poll', 'poll-1'), ('notify', None)],
)
def test_inline_poll_and_notify_resolve_literal_spawn_and_join_links(delivery: str, join_id: str | None):
    assembler = _assembler()
    spawn = _spawn(('spawn-1', CHILD_PROMPT))
    launch_ack = {'role': 'tool', 'tool_call_id': 'spawn-1', 'content': f'{LAUNCH_ACK}.'}
    _record(assembler, [SYSTEM, USER], spawn)
    _child(assembler)

    if delivery == 'inline':
        joined = {'role': 'tool', 'tool_call_id': 'spawn-1', 'content': 'child result'}
        _record(assembler, [SYSTEM, USER, spawn, joined], {'role': 'assistant', 'content': 'main done'})
    elif delivery == 'poll':
        poll = {
            'role': 'assistant',
            'content': '',
            'tool_calls': [_tool_call('poll-1', 'TaskOutput', {'task_id': 'agent-1'})],
        }
        joined = {
            'role': 'tool',
            'tool_call_id': 'poll-1',
            'content': '<retrieval_status>success</retrieval_status><output>child result</output>',
        }
        _record(assembler, [SYSTEM, USER, spawn, launch_ack], poll)
        _record(assembler, [SYSTEM, USER, spawn, launch_ack, poll, joined], {'role': 'assistant', 'content': 'done'})
    else:
        joined = _notification('agent-1')
        _record(assembler, [SYSTEM, USER, spawn, launch_ack, joined], {'role': 'assistant', 'content': 'done'})

    trajectory = _freeze(assembler)
    chain = trajectory.chains['agent-1']
    assert chain.parent_chain_id == 'main'
    assert chain.spawn is not None
    assert chain.spawn.tool_use_id == 'spawn-1'
    assert chain.spawn.node is not None
    assert trajectory.nodes[chain.spawn.node].original_message == spawn
    assert chain.join is not None
    assert chain.join.kind == delivery
    assert chain.join.tool_use_id == join_id
    assert chain.join.node is not None
    assert trajectory.nodes[chain.join.node].original_message == joined


@pytest.mark.parametrize(
    ('case', 'tail'),
    [
        (
            'launch-ack',
            [{'role': 'tool', 'tool_call_id': 'spawn-1', 'content': f'{LAUNCH_ACK}.'}],
        ),
        (
            'poll-not-ready',
            [
                {
                    'role': 'assistant',
                    'content': '',
                    'tool_calls': [_tool_call('poll-1', 'TaskOutput', {'task_id': 'agent-1'})],
                },
                {
                    'role': 'tool',
                    'tool_call_id': 'poll-1',
                    'content': '<retrieval_status>pending</retrieval_status>',
                },
            ],
        ),
        (
            'wrong-agent',
            [
                {
                    'role': 'assistant',
                    'content': '',
                    'tool_calls': [_tool_call('poll-other', 'TaskOutput', {'task_id': 'agent-2'})],
                },
                {
                    'role': 'tool',
                    'tool_call_id': 'poll-other',
                    'content': '<retrieval_status>success</retrieval_status><output>wrong</output>',
                },
            ],
        ),
        (
            'prefixed-non-delivery',
            [{'role': 'tool', 'tool_call_id': 'spawn-1', 'content': f'{LAUNCH_ACK}; child is still running'}],
        ),
    ],
)
def test_launch_ack_not_ready_wrong_agent_and_configured_non_delivery_are_not_joins(case: str, tail: list[dict]):
    del case
    assembler = _assembler()
    spawn = _spawn(('spawn-1', CHILD_PROMPT))
    _record(assembler, [SYSTEM, USER], spawn)
    _child(assembler)
    _record(assembler, [SYSTEM, USER, spawn, *tail], {'role': 'assistant', 'content': 'continue'})

    trajectory = _freeze(assembler)
    chain = trajectory.chains['agent-1']
    assert chain.spawn is not None
    assert chain.spawn.tool_use_id == 'spawn-1'
    assert chain.join is None
    assert chain.evidence['correlation']['joined'] is False


def test_child_recorded_before_spawn_still_correlates_against_the_full_forest():
    assembler = _assembler()
    spawn = _spawn(('spawn-1', CHILD_PROMPT))
    _child(assembler)
    _record(assembler, [SYSTEM, USER], spawn)

    trajectory = _freeze(assembler)
    chain = trajectory.chains['agent-1']
    assert chain.spawn is not None
    assert chain.spawn.tool_use_id == 'spawn-1'
    assert chain.spawn.node is not None
    assert trajectory.nodes[chain.spawn.node].original_message == spawn


def test_terminal_spawn_without_a_followup_is_still_the_spawn_node():
    assembler = _assembler()
    spawn = _spawn(('spawn-terminal', CHILD_PROMPT))
    _record(assembler, [SYSTEM, USER], spawn)
    _child(assembler)

    chain = _freeze(assembler).chains['agent-1']
    assert chain.spawn is not None
    assert chain.spawn.tool_use_id == 'spawn-terminal'
    assert chain.spawn.node is not None
    assert chain.join is None


def test_duplicate_spawn_occurrences_keep_the_id_but_preserve_node_ambiguity():
    assembler = _assembler()
    first = _spawn(('spawn-duplicate', CHILD_PROMPT))
    second = _spawn(('spawn-duplicate', CHILD_PROMPT))
    _record(assembler, [SYSTEM, USER], first)
    _record(assembler, [SYSTEM, USER], second)
    _child(assembler)

    trajectory = _freeze(assembler)
    chain = trajectory.chains['agent-1']
    assert chain.spawn is not None
    assert chain.spawn.tool_use_id == 'spawn-duplicate'
    assert chain.spawn.node is None
    assert [node.selection for node in trajectory.iter_nodes() if node.generated_by == 'llm'] == [
        'rejected_sibling',
        'selected',
        None,
    ]


def test_client_replays_of_a_spawn_keep_the_original_llm_node():
    assembler = _assembler()
    spawn = _spawn(('spawn-replayed', CHILD_PROMPT))
    _record(assembler, [SYSTEM, USER], spawn)
    _record(
        assembler,
        [SYSTEM, USER, spawn, {'role': 'tool', 'tool_call_id': 'other-call', 'content': 'other result'}],
        {'role': 'assistant', 'content': 'continued'},
    )
    _child(assembler)

    trajectory = _freeze(assembler)
    chain = trajectory.chains['agent-1']
    assert chain.spawn is not None
    assert chain.spawn.tool_use_id == 'spawn-replayed'
    assert chain.spawn.node is not None
    assert trajectory.nodes[chain.spawn.node].generated_by == 'llm'
    assert trajectory.nodes[chain.spawn.node].original_message == spawn


def test_exact_prompt_match_wins_over_an_earlier_prefix_candidate():
    assembler = _assembler()
    spawn = _spawn(('spawn-prefix', 'inspect'), ('spawn-exact', CHILD_PROMPT))
    _record(assembler, [SYSTEM, USER], spawn)
    _child(assembler)

    chain = _freeze(assembler).chains['agent-1']
    assert chain.spawn is not None
    assert chain.spawn.tool_use_id == 'spawn-exact'
    assert chain.spawn.candidates == []


def test_reverse_order_parallel_returns_resolve_to_their_own_children():
    assembler = _assembler()
    prompt_1 = 'inspect dates'
    prompt_2 = 'inspect users'
    spawn = _spawn(('spawn-1', prompt_1), ('spawn-2', prompt_2))
    return_2 = {'role': 'tool', 'tool_call_id': 'spawn-2', 'content': 'users result'}
    return_1 = {'role': 'tool', 'tool_call_id': 'spawn-1', 'content': 'dates result'}
    _record(assembler, [SYSTEM, USER], spawn)
    _child(assembler, agent_id='agent-1', prompt=prompt_1)
    _child(assembler, agent_id='agent-2', prompt=prompt_2)
    _record(
        assembler,
        [SYSTEM, USER, spawn, return_2, return_1],
        {'role': 'assistant', 'content': 'main done'},
    )

    trajectory = _freeze(assembler)
    first = trajectory.chains['agent-1']
    second = trajectory.chains['agent-2']
    assert first.spawn is not None
    assert first.spawn.tool_use_id == 'spawn-1'
    assert first.join is not None
    assert first.join.tool_use_id == 'spawn-1'
    assert trajectory.nodes[first.join.node].original_message == return_1
    assert second.spawn is not None
    assert second.spawn.tool_use_id == 'spawn-2'
    assert second.join is not None
    assert second.join.tool_use_id == 'spawn-2'
    assert trajectory.nodes[second.join.node].original_message == return_2


def test_nested_child_resolves_its_parent_chain_and_parent_owned_spawn():
    assembler = _assembler()
    nested_prompt = 'inspect nested state'
    parent_spawn = _spawn(('spawn-child', nested_prompt))
    _record(
        assembler,
        [CHILD_SYSTEM, CHILD_USER],
        parent_spawn,
        {'agent_id': 'parent-agent'},
    )
    _child(
        assembler,
        agent_id='child-agent',
        parent_agent_id='parent-agent',
        prompt=nested_prompt,
        system={'role': 'system', 'content': 'nested child'},
    )

    trajectory = _freeze(assembler)
    chain = trajectory.chains['child-agent']
    assert chain.parent_chain_id == 'parent-agent'
    assert chain.spawn is not None
    assert chain.spawn.tool_use_id == 'spawn-child'
    assert chain.spawn.node is not None
    assert trajectory.node_chain(chain.spawn.node) == 'parent-agent'


def test_identical_child_prompts_are_scoped_to_the_declared_parent_agent():
    assembler = _assembler()
    spawns = {}
    joins = {}
    for index in (1, 2):
        parent = f'parent-{index}'
        spawn = _spawn((f'spawn-{index}', CHILD_PROMPT))
        join = {'role': 'tool', 'tool_call_id': f'spawn-{index}', 'content': f'result-{index}'}
        parent_messages = [
            {'role': 'system', 'content': parent},
            {'role': 'user', 'content': f'parent task {index}'},
        ]
        _record(assembler, parent_messages, spawn, {'agent_id': parent})
        _record(
            assembler, [*parent_messages, spawn, join], {'role': 'assistant', 'content': 'done'}, {'agent_id': parent}
        )
        spawns[parent] = spawn
        joins[parent] = join
    for index in (1, 2):
        _child(
            assembler,
            agent_id=f'child-{index}',
            parent_agent_id=f'parent-{index}',
            system={'role': 'system', 'content': f'child system {index}'},
        )

    trajectory = _freeze(assembler)
    for index in (1, 2):
        chain = trajectory.chains[f'child-{index}']
        parent = f'parent-{index}'
        assert chain.parent_chain_id == parent
        assert chain.spawn is not None
        assert chain.spawn.tool_use_id == f'spawn-{index}'
        assert trajectory.nodes[chain.spawn.node].original_message == spawns[parent]
        assert chain.join is not None
        assert chain.join.tool_use_id == f'spawn-{index}'
        assert trajectory.nodes[chain.join.node].original_message == joins[parent]


def test_rewritten_multi_roots_keep_the_intersection_spawn_and_join():
    assembler = _assembler()
    spawn = _spawn(('spawn-common', CHILD_PROMPT))
    joined = {'role': 'tool', 'tool_call_id': 'spawn-common', 'content': 'child result'}
    _record(assembler, [SYSTEM, USER], spawn)
    _child(assembler, system=CHILD_SYSTEM, reply='first reply')
    _child(assembler, system={'role': 'system', 'content': 'rewritten child'}, reply='rewritten reply')
    _record(assembler, [SYSTEM, USER, spawn, joined], {'role': 'assistant', 'content': 'done'})

    trajectory = _freeze(assembler)
    chain = trajectory.chains['agent-1']
    assert len(chain.roots) == 2
    assert chain.spawn is not None
    assert chain.spawn.tool_use_id == 'spawn-common'
    assert chain.join is not None
    assert chain.join.tool_use_id == 'spawn-common'
    assert chain.parent_chain_id == 'main'


def test_rewritten_multi_roots_union_disjoint_spawns_as_explicit_ambiguity():
    assembler = _assembler()
    prompt_1 = 'first rewritten task'
    prompt_2 = 'second rewritten task'
    spawn = _spawn(('spawn-1', prompt_1), ('spawn-2', prompt_2))
    _record(assembler, [SYSTEM, USER], spawn)
    _child(assembler, prompt=prompt_1, system={'role': 'system', 'content': 'child root 1'})
    _child(assembler, prompt=prompt_2, system={'role': 'system', 'content': 'child root 2'})

    trajectory = _freeze(assembler)
    chain = trajectory.chains['agent-1']
    assert len(chain.roots) == 2
    assert chain.spawn is not None
    assert chain.spawn.tool_use_id is None
    assert chain.spawn.candidates == ['spawn-1', 'spawn-2']
    assert chain.spawn.node is not None
    assert trajectory.nodes[chain.spawn.node].original_message == spawn


def test_mixed_main_and_sidechain_siblings_keep_both_branches_without_cross_attribution():
    assembler = _assembler()
    shared = {'role': 'user', 'content': CHILD_PROMPT}
    _record(assembler, [CHILD_SYSTEM, shared], {'role': 'assistant', 'content': 'main reply'})
    _record(
        assembler,
        [CHILD_SYSTEM, shared],
        {'role': 'assistant', 'content': 'child reply'},
        {'agent_id': 'agent-1'},
    )

    trajectory = _freeze(assembler)
    assert set(trajectory.chains) == {'main', 'agent-1'}
    assert [node_list.chain_id for node_list in trajectory.to_segment_node_lists()] == ['agent-1', 'main']
    assert trajectory.chains['agent-1'].spawn is None
    assert trajectory.chains['agent-1'].join is None


def test_rejected_sidechain_sibling_does_not_create_a_second_export_or_link():
    assembler = _assembler()
    spawn = _spawn(('spawn-1', CHILD_PROMPT))
    dangling = {'role': 'assistant', 'content': 'dangling'}
    selected = {'role': 'assistant', 'content': 'selected'}
    _record(assembler, [SYSTEM, USER], spawn)
    _record(assembler, [CHILD_SYSTEM, CHILD_USER], selected, {'agent_id': 'agent-1'})
    _record(assembler, [CHILD_SYSTEM, CHILD_USER], dangling, {'agent_id': 'agent-1'})
    _record(
        assembler,
        [CHILD_SYSTEM, CHILD_USER, selected, {'role': 'user', 'content': 'continue'}],
        {'role': 'assistant', 'content': 'extended'},
        {'agent_id': 'agent-1'},
    )

    trajectory = _freeze(assembler)
    child_llm = [
        node
        for node in trajectory.iter_nodes()
        if node.generated_by == 'llm' and trajectory.node_chain(node.node_id) == 'agent-1'
    ]
    assert {node.original_message['content']: node.selection for node in child_llm} == {
        'selected': 'selected',
        'dangling': 'rejected_sibling',
        'extended': None,
    }
    assert len([item for item in trajectory.to_segment_node_lists() if item.chain_id == 'agent-1']) == 1
    assert trajectory.chains['agent-1'].spawn is not None
    assert trajectory.chains['agent-1'].spawn.tool_use_id == 'spawn-1'


def test_spawn_before_compaction_and_return_after_compaction_keep_cross_root_links():
    assembler = _assembler()
    spawn = _spawn(('spawn-1', CHILD_PROMPT))
    joined = {'role': 'tool', 'tool_call_id': 'spawn-1', 'content': 'child after compaction'}
    _record(assembler, [SYSTEM, USER], spawn)
    _child(assembler)
    _record(
        assembler,
        [{'role': 'user', 'content': 'compacted context'}, joined],
        {'role': 'assistant', 'content': 'main after compaction'},
        {'invocation_kind': 'compaction_resume'},
    )

    trajectory = _freeze(assembler)
    chain = trajectory.chains['agent-1']
    assert chain.parent_chain_id == 'main'
    assert chain.spawn is not None
    assert chain.spawn.tool_use_id == 'spawn-1'
    assert trajectory.nodes[chain.spawn.node].original_message == spawn
    assert chain.join is not None
    assert chain.join.tool_use_id == 'spawn-1'
    assert chain.join.kind == 'inline'
    assert trajectory.nodes[chain.join.node].original_message == joined
    main_segments = [item for item in trajectory.to_segment_node_lists() if item.chain_id == 'main']
    boundaries = [item.response_nodes[0].segment_boundary for item in main_segments]
    assert [boundary.kind if boundary is not None else 'conversation' for boundary in boundaries] == [
        'conversation',
        'compaction_resume',
    ]
