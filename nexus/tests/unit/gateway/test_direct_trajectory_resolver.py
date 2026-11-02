from __future__ import annotations

import pickle

from dataclasses import replace

import pytest

from nexus.gateway.runtime_state import NodeRuntimeFacts, RuntimeFactOccurrence, RuntimeState
from nexus.gateway.scaffold_policy import (
    BoundAnchor,
    ChildPromptCandidate,
    DeclaredChainClaim,
    DeliveryReceipt,
    GenericScaffoldPolicy,
    InputMessage,
    NotificationDelivery,
    ParentClaim,
    RequestEvent,
    ResponseEvent,
    ResponsePolicy,
    RetrievalIntent,
    SidechainMembership,
    SpawnIntent,
)
from nexus.gateway.trajectory import Chain, JoinLink, SegmentBoundary, SpawnLink, Trajectory, TrajectoryNode
from nexus.gateway.trajectory_assembler import TrajectoryAssembler
from nexus.gateway.trajectory_freeze import freeze_trajectory


def _node(
    node_id: int,
    *,
    parent: int | None,
    children: list[int] | None = None,
    generated_by: str = 'client',
    message_id: int | None = None,
    content: str | None = None,
) -> TrajectoryNode:
    return TrajectoryNode(
        node_id=node_id,
        message_id=node_id if message_id is None else message_id,
        parent=parent,
        generated_by=generated_by,
        children=list(children or []),
        original_message={'role': 'assistant' if generated_by == 'llm' else 'user', 'content': content or str(node_id)},
    )


def _state(trajectory: Trajectory, facts: dict[int, NodeRuntimeFacts]) -> RuntimeState:
    state = RuntimeState()
    state.facts.by_node = facts
    state.next_node_id = max(trajectory.nodes, default=-1) + 1
    state.next_message_id = max((node.message_id for node in trajectory.nodes.values()), default=-1) + 1
    return state


def _facts(
    node_id: int,
    *,
    membership: SidechainMembership | None = None,
    child_prompt: str | None = None,
    spawn_intents=(),
    retrieval_intents=(),
    delivery_receipts=(),
    notification_deliveries=(),
    declared_claim=None,
    parent_claim=None,
    child_prompts=(),
    occurrences=(),
    context_key=None,
) -> NodeRuntimeFacts:
    prompts = list(child_prompts)
    if child_prompt is not None:
        prompts.append(ChildPromptCandidate(BoundAnchor(node_id), child_prompt, 0))
    occurrence_list = list(occurrences)
    existing_types = {type(occurrence.fact) for occurrence in occurrence_list}
    values = (
        membership,
        *spawn_intents,
        *retrieval_intents,
        *delivery_receipts,
        *notification_deliveries,
        declared_claim,
        parent_claim,
        *prompts,
    )
    sequence = max((occurrence.sequence for occurrence in occurrence_list), default=-1) + 1
    for value in values:
        if value is not None and type(value) not in existing_types:
            occurrence_list.append(RuntimeFactOccurrence(sequence, value))
            sequence += 1
    return NodeRuntimeFacts(
        identity=('node', node_id),
        context_key=context_key if context_key is not None else membership.chain_id if membership else 'main',
        occurrences=tuple(occurrence_list),
    )


def test_runtime_ids_are_preserved_before_anonymous_chain_identity_and_reference_resolution():
    anonymous = SidechainMembership(chain_id='', kind='worker', detector='anonymous-worker')
    trajectory = Trajectory(
        session_id='canonical-first',
        nodes={
            10: _node(10, parent=None, children=[40], content='child root'),
            40: _node(40, parent=10, generated_by='llm', content='child answer'),
            20: _node(20, parent=None, children=[30], content='main root'),
            30: _node(30, parent=20, generated_by='llm', content='main answer'),
        },
        roots=[10, 20],
    )
    trajectory.nodes[40].request_message_overrides = {10: {'role': 'user', 'content': 'wire child root'}}
    state = _state(
        trajectory,
        {
            10: _facts(10, membership=anonymous, child_prompt='child prompt'),
            40: _facts(40, membership=anonymous),
            20: _facts(20),
            30: _facts(30),
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)

    assert list(frozen.nodes) == [10, 40, 20, 30]
    assert frozen.roots == [10, 20]
    assert frozen.nodes[40].request_message_overrides == {10: {'role': 'user', 'content': 'wire child root'}}
    assert list(frozen.chains) == ['main', 'chain@10']
    assert frozen.chains['chain@10'].roots == [10]


@pytest.mark.parametrize(
    ('first_chain_id', 'second_chain_id', 'expected_chain_id'),
    [('', 'chain@0', 'chain@0'), ('chain@2', '', 'chain@2')],
    ids=['anonymous-then-named', 'named-then-anonymous'],
)
def test_compatible_named_and_anonymous_chain_ids_reuse_the_same_registered_chain(
    first_chain_id, second_chain_id, expected_chain_id
):
    first = SidechainMembership(
        chain_id=first_chain_id,
        kind='worker',
        detector='flag',
        attributes=(('invocation_kind', 'worker'),),
    )
    second = SidechainMembership(
        chain_id=second_chain_id,
        kind='worker',
        detector='flag',
        attributes=(('invocation_kind', 'worker'),),
    )
    trajectory = Trajectory(
        session_id='compatible-provenance-collision',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
        },
        roots=[0, 2],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=first),
            1: _facts(1, membership=first),
            2: _facts(2, membership=second),
            3: _facts(3, membership=second),
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)

    assert list(frozen.chains) == ['main', expected_chain_id]
    assert frozen.chains[expected_chain_id] == Chain(
        chain_id=expected_chain_id,
        kind='worker',
        parent_chain_id=None,
        roots=[0, 2],
        detector='flag',
        evidence={'attributes': {'invocation_kind': 'worker'}},
    )


def test_freeze_rejects_a_physical_node_that_is_not_reachable_from_real_roots():
    trajectory = Trajectory(
        session_id='unreachable',
        nodes={
            0: _node(0, parent=None),
            1: _node(1, parent=None),
        },
        roots=[0],
    )
    state = _state(trajectory, {0: _facts(0), 1: _facts(1)})

    with pytest.raises(ValueError, match='not reachable from trajectory roots'):
        freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)


def test_freeze_handles_1099_node_linear_history_without_recursion():
    nodes = {index: _node(index, parent=index - 1 if index else None, children=[index + 1]) for index in range(1099)}
    nodes[1099] = _node(1099, parent=1098, generated_by='llm', content='done')
    facts = {index: NodeRuntimeFacts(identity=('node', index), context_key='main') for index in range(1100)}
    live = Trajectory(session_id='long', nodes=nodes, roots=[0])
    state = RuntimeState()
    state.facts.by_node = facts
    state.next_node_id = 1100
    state.next_message_id = 1100

    frozen = freeze_trajectory(live, state, ResponsePolicy.KEEP_ALL)

    assert len(frozen.nodes) == 1100
    assert frozen.roots == [0]
    assert list(frozen.nodes) == list(range(1100))
    assert frozen.nodes[1099].parent == 1098
    assert frozen.to_messages()[-1] == {'role': 'assistant', 'content': 'done'}


def test_chain_membership_stops_at_the_first_llm_frontier_so_nested_children_form_their_own_chain():
    parent = SidechainMembership(chain_id='parent', kind='worker', detector='agent-id')
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='nested-membership',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, children=[2], generated_by='llm'),
            2: _node(2, parent=1, children=[3], generated_by='observation'),
            3: _node(3, parent=2, generated_by='llm'),
        },
        roots=[0],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=parent),
            1: _facts(1, membership=parent),
            2: _facts(2, membership=child),
            3: _facts(3, membership=child),
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)

    assert list(frozen.chains) == ['main', 'parent', 'child']
    assert frozen.chains['parent'].roots == [0]
    assert frozen.chains['child'].roots == [2]
    assert [frozen.node_chain(node_id) for node_id in range(4)] == ['parent', 'parent', 'child', 'child']


def test_ambiguous_shared_prefix_correlates_each_child_from_the_shared_anchor():
    child_a = SidechainMembership(chain_id='A', kind='worker', detector='agent-id')
    child_b = SidechainMembership(chain_id='B', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='ambiguous-shared-prefix-correlation',
        nodes={
            0: _node(0, parent=None, children=[1], content='main prompt'),
            1: _node(1, parent=0, generated_by='llm', content='spawn source'),
            2: _node(2, parent=None, children=[3, 4], content='child prompt'),
            3: _node(3, parent=2, generated_by='llm', content='answer a'),
            4: _node(4, parent=2, generated_by='llm', content='answer b'),
        },
        roots=[0, 2],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0),
            1: _facts(
                1,
                spawn_intents=(SpawnIntent(BoundAnchor(1), 'spawn-a', 'child prompt'),),
            ),
            2: _facts(2, child_prompt='child prompt'),
            3: _facts(3, membership=child_a),
            4: _facts(4, membership=child_b),
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)

    assert list(frozen.chains) == ['main', 'A', 'B']
    assert frozen.chains['main'].roots == [0, 2]
    for chain_id, root_id in (('A', 3), ('B', 4)):
        assert frozen.chains[chain_id] == Chain(
            chain_id=chain_id,
            kind='worker',
            parent_chain_id='main',
            roots=[root_id],
            spawn=SpawnLink(node=1, tool_use_id='spawn-a'),
            detector='agent-id',
            evidence={
                'anchor_node': 2,
                'correlation': {
                    'spawn_tool_use_id': 'spawn-a',
                    'joined': False,
                },
            },
        )


def test_ambiguous_shared_prefix_without_correlation_keeps_the_registered_parent():
    child_a = SidechainMembership(chain_id='A', kind='worker', detector='agent-id')
    child_b = SidechainMembership(chain_id='B', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='ambiguous-shared-prefix-no-correlation',
        nodes={
            0: _node(0, parent=None, children=[1, 2], content='shared prefix'),
            1: _node(1, parent=0, generated_by='llm', content='answer a'),
            2: _node(2, parent=0, generated_by='llm', content='answer b'),
        },
        roots=[0],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0),
            1: _facts(1, membership=child_a),
            2: _facts(2, membership=child_b),
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)

    assert list(frozen.chains) == ['main', 'A', 'B']
    assert frozen.chains['A'].parent_chain_id == 'main'
    assert frozen.chains['B'].parent_chain_id == 'main'
    assert frozen.chains['A'].evidence == {'anchor_node': 0}
    assert frozen.chains['B'].evidence == {'anchor_node': 0}


def test_nested_same_wire_children_register_by_physical_parent_in_dfs_order():
    parent_a = SidechainMembership(chain_id='A', kind='worker', detector='flag')
    parent_b = SidechainMembership(chain_id='B', kind='worker', detector='flag')
    child = SidechainMembership(
        chain_id='child',
        kind='worker',
        parent_chain_id='claimed',
        detector='flag',
        attributes=(('parent_agent_id', 'claimed'),),
    )
    trajectory = Trajectory(
        session_id='nested-physical-parent-registration',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, children=[2], generated_by='llm'),
            2: _node(2, parent=1, children=[3], generated_by='observation'),
            3: _node(3, parent=2, generated_by='llm'),
            4: _node(4, parent=None, children=[5]),
            5: _node(5, parent=4, children=[6], generated_by='llm'),
            6: _node(6, parent=5, children=[7], generated_by='observation'),
            7: _node(7, parent=6, generated_by='llm'),
        },
        roots=[0, 4],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=parent_a),
            1: _facts(1, membership=parent_a),
            2: _facts(2, membership=child),
            3: _facts(3, membership=child, parent_claim=ParentClaim('claimed')),
            4: _facts(4, membership=parent_b),
            5: _facts(5, membership=parent_b),
            6: _facts(6, membership=child),
            7: _facts(7, membership=child, parent_claim=ParentClaim('claimed')),
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)

    assert list(frozen.chains) == ['main', 'A', 'child', 'B', 'child@6']
    assert frozen.chains['A'].roots == [0]
    assert frozen.chains['B'].roots == [4]
    assert frozen.chains['child'] == Chain(
        chain_id='child',
        kind='worker',
        parent_chain_id='A',
        roots=[2],
        detector='flag',
        evidence={
            'anchor_node': 1,
            'attributes': {'parent_agent_id': 'claimed'},
            'correlation': {
                'fork_node_index': 1,
                'fork_segment_index': 0,
                'parent_agent_id': 'claimed',
            },
        },
    )
    assert frozen.chains['child@6'] == Chain(
        chain_id='child@6',
        kind='worker',
        parent_chain_id='B',
        roots=[6],
        detector='flag',
        evidence={
            'anchor_node': 5,
            'collides_with': 'child',
            'attributes': {'parent_agent_id': 'claimed'},
            'correlation': {
                'fork_node_index': 1,
                'fork_segment_index': 0,
                'parent_agent_id': 'claimed',
            },
        },
    )


def test_chain_identity_ignores_parent_and_attributes_while_attribute_conflicts_are_evidence():
    first = SidechainMembership(
        chain_id='worker',
        kind='worker',
        parent_chain_id='parent-a',
        detector='agent-id',
        attributes=(('color', 'red'), ('stable', 1)),
    )
    second = SidechainMembership(
        chain_id='worker',
        kind='worker',
        parent_chain_id='parent-b',
        detector='agent-id',
        attributes=(('color', 'blue'), ('stable', 1)),
    )
    trajectory = Trajectory(
        session_id='membership-identity',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
        },
        roots=[0, 2],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=first),
            1: _facts(1, membership=first),
            2: _facts(2, membership=second),
            3: _facts(3, membership=second),
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)

    assert list(frozen.chains) == ['main', 'worker']
    assert frozen.chains['worker'].roots == [0, 2]
    assert frozen.chains['worker'].evidence['attributes'] == {'stable': 1}
    assert frozen.chains['worker'].evidence['attribute_candidates'] == {'color': ['red', 'blue']}


def test_mutually_exclusive_explicit_declared_parents_split_a_reused_named_chain_id():
    membership = SidechainMembership(chain_id='worker', kind='worker', detector='declared_chain')
    trajectory = Trajectory(
        session_id='declared-parent-collision',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
        },
        roots=[0, 2],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=membership),
            1: _facts(
                1,
                membership=membership,
                declared_claim=DeclaredChainClaim('worker', 'worker', 'parent-a'),
            ),
            2: _facts(2, membership=membership),
            3: _facts(
                3,
                membership=membership,
                declared_claim=DeclaredChainClaim('worker', 'worker', 'parent-b'),
            ),
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)

    assert list(frozen.chains) == ['main', 'worker', 'worker@2']
    assert frozen.chains['worker'].roots == [0]
    assert frozen.chains['worker'].parent_chain_id == 'parent-a'
    assert frozen.chains['worker@2'].roots == [2]
    assert frozen.chains['worker@2'].parent_chain_id == 'parent-b'
    assert frozen.chains['worker@2'].evidence['collides_with'] == 'worker'


def test_shared_branch_registration_uses_the_first_frontier_declared_parent_before_link_ambiguity():
    worker_a = SidechainMembership(
        chain_id='worker',
        kind='worker',
        parent_chain_id='A',
        detector='declared_chain',
        attributes=(('parent_chain_id', 'A'),),
    )
    worker_b = SidechainMembership(
        chain_id='worker',
        kind='worker',
        parent_chain_id='B',
        detector='declared_chain',
        attributes=(('parent_chain_id', 'B'),),
    )
    trajectory = Trajectory(
        session_id='shared-branch-declared-parent-registration',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm', content='independent A'),
            2: _node(2, parent=None, children=[3, 4], content='shared request'),
            3: _node(3, parent=2, generated_by='llm', content='B first'),
            4: _node(4, parent=2, generated_by='llm', content='A second'),
        },
        roots=[0, 2],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=worker_a),
            1: _facts(
                1,
                membership=worker_a,
                declared_claim=DeclaredChainClaim('worker', 'worker', 'A'),
            ),
            2: _facts(2),
            3: _facts(
                3,
                membership=worker_b,
                declared_claim=DeclaredChainClaim('worker', 'worker', 'B'),
            ),
            4: _facts(
                4,
                membership=worker_a,
                declared_claim=DeclaredChainClaim('worker', 'worker', 'A'),
            ),
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)

    assert list(frozen.chains) == ['main', 'worker', 'worker@2']
    assert frozen.chains['worker'].parent_chain_id == 'A'
    assert frozen.chains['worker'].roots == [0]
    assert frozen.chains['worker@2'] == Chain(
        chain_id='worker@2',
        kind='worker',
        parent_chain_id=None,
        roots=[2],
        detector='declared_chain',
        evidence={
            'collides_with': 'worker',
            'attribute_candidates': {'parent_chain_id': ['B', 'A']},
            'parent_chain_candidates': ['A', 'B'],
            'correlation_candidates': [
                {'parent_chain_id': 'A'},
                {'parent_chain_id': 'B'},
            ],
        },
    )


def test_external_parent_claims_do_not_split_general_named_membership_identity():
    membership = SidechainMembership(chain_id='worker', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='external-parent-no-collision',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
        },
        roots=[0, 2],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=membership),
            1: _facts(1, membership=membership, parent_claim=ParentClaim('external-a')),
            2: _facts(2, membership=membership),
            3: _facts(3, membership=membership, parent_claim=ParentClaim('external-b')),
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)

    assert list(frozen.chains) == ['main', 'worker']
    assert frozen.chains['worker'].roots == [0, 2]
    assert frozen.chains['worker'].parent_chain_id is None
    assert frozen.chains['worker'].evidence == {}


def test_rejected_sibling_remains_in_the_complete_chain_partition_before_selection():
    membership = SidechainMembership(chain_id='worker', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='rejected-membership',
        nodes={
            0: _node(0, parent=None, children=[1, 2]),
            1: _node(1, parent=0, generated_by='llm', content='dangling'),
            2: _node(2, parent=0, children=[3], generated_by='llm', content='extended'),
            3: _node(3, parent=2, content='continuation'),
        },
        roots=[0],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=membership),
            1: replace(_facts(1, membership=membership), context_key='worker-context'),
            2: replace(_facts(2, membership=membership), context_key='worker-context'),
            3: _facts(3, membership=membership),
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.PREFER_EXTENDED_ELSE_LAST)

    assert frozen.nodes[1].selection == 'rejected_sibling'
    assert frozen.nodes[2].selection == 'selected'
    assert frozen.chains['worker'].roots == [0]
    assert frozen.node_chain(1) == 'worker'
    assert frozen.node_chain(2) == 'worker'


def test_reused_physical_node_preserves_creation_ownership_and_appends_anchored_chronology():
    class ReusedFactsPolicy(GenericScaffoldPolicy):
        def inspect_event(self, event):
            facts = super().inspect_event(event)
            if isinstance(event, ResponseEvent):
                parent = event.request.request_meta['parent']
                membership = SidechainMembership(
                    chain_id='worker',
                    kind='worker',
                    parent_chain_id=parent,
                    detector='test',
                )
                return replace(
                    facts,
                    membership=membership,
                    declared_claim=DeclaredChainClaim('worker', 'worker', parent),
                    parent_claim=ParentClaim(parent),
                )
            if not isinstance(event, RequestEvent) or not event.messages:
                return facts
            parent = event.request_meta['parent']
            membership = SidechainMembership(
                chain_id='worker',
                kind='worker',
                parent_chain_id=parent,
                detector='test',
            )
            return replace(
                facts,
                membership=membership,
                declared_claim=DeclaredChainClaim('worker', 'worker', parent),
                parent_claim=ParentClaim(parent),
                delivery_receipts=(DeliveryReceipt(InputMessage(0), 'spawn-1', False),),
            )

    assembler = TrajectoryAssembler(scaffold_policy=ReusedFactsPolicy(policy_id='reused-facts'))
    prompt = {'role': 'user', 'content': 'same physical prompt'}
    assembler.record_completion([prompt], {'role': 'assistant', 'content': 'first'}, request_meta={'parent': 'a'})
    assembler.record_completion([prompt], {'role': 'assistant', 'content': 'second'}, request_meta={'parent': 'b'})

    node_facts = assembler._runtime_state.facts.by_node[0]
    occurrences = node_facts.occurrences
    assert [occurrence.sequence for occurrence in occurrences] == sorted(
        occurrence.sequence for occurrence in occurrences
    )
    assert [
        occurrence.fact.parent_chain_id
        for occurrence in occurrences
        if isinstance(occurrence.fact, SidechainMembership)
    ] == ['a']
    assert [
        occurrence.fact.parent_chain_id for occurrence in occurrences if isinstance(occurrence.fact, DeclaredChainClaim)
    ] == ['a']
    assert [
        occurrence.fact.parent_chain_id for occurrence in occurrences if isinstance(occurrence.fact, ParentClaim)
    ] == ['a']
    assert [
        occurrence.fact.delivered for occurrence in occurrences if isinstance(occurrence.fact, DeliveryReceipt)
    ] == [False, False]
    assert [occurrence.fact.anchor for occurrence in occurrences if isinstance(occurrence.fact, DeliveryReceipt)] == [
        BoundAnchor(0),
        BoundAnchor(0),
    ]
    assert len(node_facts.facts_of_type(DeliveryReceipt)) == 2
    chain = assembler.freeze().chains['worker']
    assert chain.parent_chain_id is None
    assert chain.evidence == {
        'parent_chain_candidates': ['a', 'b'],
        'correlation_candidates': [
            {'parent_chain_id': 'b'},
            {'parent_chain_id': 'a'},
        ],
    }


def test_reused_membership_occurrences_aggregate_attributes_without_changing_identity():
    red = SidechainMembership(
        chain_id='worker',
        kind='worker',
        detector='agent-id',
        attributes=(('color', 'red'), ('stable', 1)),
    )
    blue = replace(red, attributes=(('color', 'blue'), ('stable', 1)))
    trajectory = Trajectory(
        session_id='membership-occurrences',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
        },
        roots=[0],
    )
    llm_facts = _facts(
        1,
        membership=red,
        occurrences=(RuntimeFactOccurrence(0, red), RuntimeFactOccurrence(1, blue)),
    )
    state = _state(trajectory, {0: _facts(0, membership=red), 1: llm_facts})

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['worker']

    assert chain.evidence['attributes'] == {'stable': 1}
    assert chain.evidence['attribute_candidates'] == {'color': ['red', 'blue']}


def test_each_selected_chain_leaf_contributes_its_own_spawn_candidate_before_union_ambiguity():
    membership = SidechainMembership(chain_id='worker', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='per-leaf-correlation',
        nodes={
            0: _node(0, parent=None, children=[1, 2]),
            1: _node(1, parent=0, generated_by='llm', content='leaf a'),
            2: _node(2, parent=0, generated_by='llm', content='leaf b'),
            3: _node(3, parent=None, generated_by='llm', content='spawn source'),
        },
        roots=[0, 3],
    )
    source_facts = _facts(
        3,
        spawn_intents=(
            SpawnIntent(BoundAnchor(3), 'spawn-a', 'prompt a'),
            SpawnIntent(BoundAnchor(3), 'spawn-b', 'prompt b'),
        ),
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=membership),
            1: _facts(1, membership=membership, child_prompt='prompt a'),
            2: _facts(2, membership=membership, child_prompt='prompt b'),
            3: source_facts,
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['worker']

    assert chain.spawn is not None
    assert chain.spawn.tool_use_id is None
    assert chain.spawn.candidates == ['spawn-b', 'spawn-a']
    assert chain.spawn.node == 3
    assert [candidate['spawn_tool_use_id'] for candidate in chain.evidence['correlation_candidates']] == [
        'spawn-b',
        'spawn-a',
    ]


def test_equal_message_id_spawn_sources_keep_baseline_reverse_forest_traversal_order():
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='equal-message-spawn-order',
        nodes={
            0: _node(0, parent=None, generated_by='llm', message_id=10, content='first source'),
            1: _node(1, parent=None, generated_by='llm', message_id=10, content='second source'),
            2: _node(2, parent=None, children=[3], content='child prompt'),
            3: _node(3, parent=2, generated_by='llm', content='child answer'),
        },
        roots=[0, 1, 2],
    )
    state = _state(
        trajectory,
        {
            0: _facts(
                0,
                spawn_intents=(SpawnIntent(BoundAnchor(0), 'spawn-first', 'child prompt'),),
            ),
            1: _facts(
                1,
                spawn_intents=(SpawnIntent(BoundAnchor(1), 'spawn-second', 'child prompt'),),
            ),
            2: _facts(2, membership=child, child_prompt='child prompt'),
            3: _facts(3, membership=child),
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['child']

    assert chain.spawn is not None
    assert chain.spawn.candidates == ['spawn-second', 'spawn-first']


def test_multi_root_spawn_intersection_drops_inline_joins_for_non_common_spawn_ids():
    membership = SidechainMembership(chain_id='worker', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='intersection-join-filter',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
            4: _node(4, parent=None, generated_by='llm', content='spawn x/y'),
            5: _node(5, parent=None, generated_by='llm', content='spawn y/z'),
            6: _node(6, parent=None, message_id=100, content='x delivery'),
            7: _node(7, parent=None, message_id=200, content='z delivery'),
        },
        roots=[0, 2, 4, 5, 6, 7],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=membership),
            1: _facts(1, membership=membership, child_prompt='prompt a'),
            2: _facts(2, membership=membership),
            3: _facts(3, membership=membership, child_prompt='prompt b'),
            4: _facts(
                4,
                spawn_intents=(
                    SpawnIntent(BoundAnchor(4), 'x', 'prompt a'),
                    SpawnIntent(BoundAnchor(4), 'y', 'prompt a'),
                ),
            ),
            5: _facts(
                5,
                spawn_intents=(
                    SpawnIntent(BoundAnchor(5), 'y', 'prompt b'),
                    SpawnIntent(BoundAnchor(5), 'z', 'prompt b'),
                ),
            ),
            6: _facts(
                6,
                delivery_receipts=(DeliveryReceipt(BoundAnchor(6), 'x', True),),
            ),
            7: _facts(
                7,
                delivery_receipts=(DeliveryReceipt(BoundAnchor(7), 'z', True),),
            ),
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['worker']

    assert chain.spawn is not None
    assert chain.spawn.tool_use_id == 'y'
    assert chain.join is not None
    assert chain.join.node is None
    assert chain.join.tool_use_id is None
    assert chain.join.kind == 'inline'
    assert all('parent_chain_id' not in candidate for candidate in chain.evidence['correlation_candidates'])


def test_parent_chain_correlation_excludes_the_entire_entry_subtree_including_nested_child_facts():
    parent = SidechainMembership(chain_id='parent', kind='worker', detector='agent-id')
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='exclude-entry-subtree',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, children=[2], generated_by='llm', content='parent leaf'),
            2: _node(2, parent=1, children=[3], content='nested entry'),
            3: _node(3, parent=2, generated_by='llm', content='nested child source'),
        },
        roots=[0],
    )
    nested_source = _facts(
        3,
        membership=child,
        spawn_intents=(SpawnIntent(BoundAnchor(3), 'nested-spawn', 'parent prompt'),),
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=parent, child_prompt='parent prompt'),
            1: _facts(1, membership=parent),
            2: _facts(2, membership=child),
            3: nested_source,
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)

    assert frozen.chains['parent'].spawn is None
    assert frozen.chains['child'].spawn is None


def test_declared_parent_precedes_a_different_local_parent_agent_claim():
    declared = SidechainMembership(chain_id='declared', kind='worker', detector='agent-id')
    local = SidechainMembership(chain_id='local', kind='worker', detector='agent-id')
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='declared-parent-precedence',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
            4: _node(4, parent=None, children=[5]),
            5: _node(5, parent=4, generated_by='llm'),
        },
        roots=[0, 2, 4],
    )
    child_facts = _facts(
        5,
        membership=child,
        declared_claim=DeclaredChainClaim('child', 'worker', 'declared'),
        parent_claim=ParentClaim('local'),
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=declared),
            1: _facts(1, membership=declared),
            2: _facts(2, membership=local),
            3: _facts(3, membership=local),
            4: _facts(4, membership=child),
            5: child_facts,
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['child']

    assert chain.parent_chain_id == 'declared'
    assert chain.evidence == {'correlation': {'parent_chain_id': 'declared', 'parent_agent_id': 'local'}}


def test_local_parent_agent_rescope_clears_out_of_scope_spawn_but_retains_mismatch_evidence():
    local = SidechainMembership(chain_id='local', kind='worker', detector='agent-id')
    wrong = SidechainMembership(chain_id='wrong', kind='worker', detector='agent-id')
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='local-parent-rescope',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
            4: _node(4, parent=None, children=[5]),
            5: _node(5, parent=4, generated_by='llm'),
        },
        roots=[0, 2, 4],
    )
    wrong_facts = _facts(
        3,
        membership=wrong,
        spawn_intents=(SpawnIntent(BoundAnchor(3), 'wrong-spawn', 'child prompt'),),
    )
    child_facts = _facts(
        5,
        membership=child,
        child_prompt='child prompt',
        parent_claim=ParentClaim('local'),
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=local),
            1: _facts(1, membership=local),
            2: _facts(2, membership=wrong),
            3: wrong_facts,
            4: _facts(4, membership=child),
            5: child_facts,
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['child']

    assert chain.parent_chain_id == 'local'
    assert chain.spawn is None
    assert chain.evidence['spawn_parent_mismatch'] is True
    assert chain.evidence['correlation']['spawn_tool_use_id'] == 'wrong-spawn'
    assert chain.evidence['correlation']['parent_agent_id'] == 'local'


@pytest.mark.parametrize('child_roots', [(4, 6), (6, 4)])
def test_local_parent_scope_keeps_valid_spawn_and_join_when_another_anchor_mismatches(child_roots):
    local = SidechainMembership(chain_id='local', kind='worker', detector='agent-id')
    wrong = SidechainMembership(chain_id='wrong', kind='worker', detector='agent-id')
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='per-anchor-local-scope',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm', content='valid source'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm', content='wrong source'),
            4: _node(4, parent=None, children=[5]),
            5: _node(5, parent=4, generated_by='llm', content='valid child'),
            6: _node(6, parent=None, children=[7]),
            7: _node(7, parent=6, generated_by='llm', content='wrong child'),
        },
        roots=[0, 2, *child_roots],
    )
    valid_source = _facts(
        1,
        membership=local,
        spawn_intents=(SpawnIntent(BoundAnchor(1), 'spawn-valid', 'valid prompt'),),
        delivery_receipts=(DeliveryReceipt(BoundAnchor(1), 'spawn-valid', True),),
    )
    wrong_source = _facts(
        3,
        membership=wrong,
        spawn_intents=(SpawnIntent(BoundAnchor(3), 'spawn-wrong', 'wrong prompt'),),
        delivery_receipts=(DeliveryReceipt(BoundAnchor(3), 'spawn-wrong', True),),
    )
    valid_child = _facts(
        5,
        membership=child,
        child_prompt='valid prompt',
        parent_claim=ParentClaim('local'),
    )
    wrong_child = _facts(
        7,
        membership=child,
        child_prompt='wrong prompt',
        parent_claim=ParentClaim('local'),
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=local),
            1: valid_source,
            2: _facts(2, membership=wrong),
            3: wrong_source,
            4: _facts(4, membership=child),
            5: valid_child,
            6: _facts(6, membership=child),
            7: wrong_child,
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)
    chain = frozen.chains['child']

    assert chain.parent_chain_id == 'local'
    assert chain.spawn is not None
    assert chain.spawn.tool_use_id == 'spawn-valid'
    assert frozen.nodes[chain.spawn.node].original_message['content'] == 'valid source'
    assert chain.join is not None
    assert chain.join.tool_use_id == 'spawn-valid'
    assert chain.join.kind == 'inline'
    assert frozen.nodes[chain.join.node].original_message['content'] == 'valid source'
    assert chain.evidence['spawn_parent_mismatch'] is True
    mismatched = [
        candidate
        for candidate in chain.evidence['correlation_candidates']
        if candidate.get('spawn_tool_use_id') == 'spawn-wrong'
    ]
    assert len(mismatched) == 1
    assert mismatched[0]['joined'] is False
    assert 'spawn_parent_mismatch' not in mismatched[0]
    assert 'join_parent_mismatch' not in mismatched[0]


def test_local_parent_scope_keeps_partial_spawn_and_join_matches_without_mismatch_evidence():
    local = SidechainMembership(chain_id='local', kind='worker', detector='agent-id')
    wrong = SidechainMembership(chain_id='wrong', kind='worker', detector='agent-id')
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='partial-local-scope',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm', content='local source'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm', content='wrong source'),
            4: _node(4, parent=None, children=[5]),
            5: _node(5, parent=4, generated_by='llm', content='child'),
        },
        roots=[0, 2, 4],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=local),
            1: _facts(
                1,
                membership=local,
                spawn_intents=(SpawnIntent(BoundAnchor(1), 'spawn-local', 'shared prompt'),),
                delivery_receipts=(DeliveryReceipt(BoundAnchor(1), 'spawn-local', True),),
            ),
            2: _facts(2, membership=wrong),
            3: _facts(
                3,
                membership=wrong,
                spawn_intents=(SpawnIntent(BoundAnchor(3), 'spawn-wrong', 'shared prompt'),),
                delivery_receipts=(DeliveryReceipt(BoundAnchor(3), 'spawn-wrong', True),),
            ),
            4: _facts(4, membership=child),
            5: _facts(
                5,
                membership=child,
                child_prompt='shared prompt',
                parent_claim=ParentClaim('local'),
            ),
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)
    chain = frozen.chains['child']

    assert chain.parent_chain_id == 'local'
    assert chain.spawn is not None
    assert chain.spawn.tool_use_id == 'spawn-local'
    assert frozen.nodes[chain.spawn.node].original_message['content'] == 'local source'
    assert chain.join is not None
    assert chain.join.tool_use_id == 'spawn-local'
    assert chain.join.kind == 'inline'
    assert frozen.nodes[chain.join.node].original_message['content'] == 'local source'
    assert 'spawn_parent_mismatch' not in chain.evidence
    assert 'join_parent_mismatch' not in chain.evidence
    assert chain.evidence['correlation'] == {
        'spawn_tool_use_id': 'spawn-local',
        'parent_agent_id': 'local',
        'joined': True,
        'join_tool_use_id': 'spawn-local',
        'join_kind': 'inline',
    }


def test_spawn_owner_precedes_join_owner_and_clears_the_out_of_scope_join():
    spawn_owner = SidechainMembership(chain_id='spawn-owner', kind='worker', detector='agent-id')
    join_owner = SidechainMembership(chain_id='join-owner', kind='worker', detector='agent-id')
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='spawn-before-join',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
            4: _node(4, parent=None, children=[5]),
            5: _node(5, parent=4, generated_by='llm'),
        },
        roots=[0, 2, 4],
    )
    spawn_facts = _facts(
        1,
        membership=spawn_owner,
        spawn_intents=(SpawnIntent(BoundAnchor(1), 'spawn-1', 'child prompt'),),
    )
    join_facts = _facts(
        3,
        membership=join_owner,
        notification_deliveries=(NotificationDelivery(BoundAnchor(3), 'child'),),
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=spawn_owner),
            1: spawn_facts,
            2: _facts(2, membership=join_owner),
            3: join_facts,
            4: _facts(4, membership=child),
            5: _facts(5, membership=child, child_prompt='child prompt'),
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['child']

    assert chain.parent_chain_id == 'spawn-owner'
    assert chain.spawn is not None
    assert chain.spawn.tool_use_id == 'spawn-1'
    assert chain.join is None
    assert chain.evidence['correlation']['joined'] is False
    assert 'join_parent_mismatch' not in chain.evidence


def test_join_owner_precedes_the_structural_parent_chain():
    structural = SidechainMembership(chain_id='structural', kind='worker', detector='agent-id')
    join_owner = SidechainMembership(chain_id='join-owner', kind='worker', detector='agent-id')
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='join-before-structural',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, children=[2], generated_by='llm'),
            2: _node(2, parent=1, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
            4: _node(4, parent=None, children=[5]),
            5: _node(5, parent=4, generated_by='llm'),
        },
        roots=[0, 4],
    )
    join_facts = _facts(
        5,
        membership=join_owner,
        notification_deliveries=(NotificationDelivery(BoundAnchor(5), 'child'),),
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=structural),
            1: _facts(1, membership=structural),
            2: _facts(2, membership=child),
            3: _facts(3, membership=child),
            4: _facts(4, membership=join_owner),
            5: join_facts,
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['child']

    assert chain.parent_chain_id == 'join-owner'
    assert chain.join is not None
    assert chain.join.kind == 'notify'
    assert chain.join.node == 5


def test_declared_parent_ambiguity_does_not_suppress_an_unambiguous_join():
    worker_a = SidechainMembership(chain_id='worker', kind='worker', detector='declared')
    worker_b = SidechainMembership(chain_id='worker', kind='worker', detector='declared')
    trajectory = Trajectory(
        session_id='declared-parent-ambiguity-with-join',
        nodes={
            0: _node(0, parent=None, children=[1, 2]),
            1: _node(1, parent=0, generated_by='llm', content='answer a'),
            2: _node(2, parent=0, generated_by='llm', content='answer b'),
            3: _node(3, parent=None, message_id=50, content='notification'),
        },
        roots=[0, 3],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0),
            1: _facts(
                1,
                membership=worker_a,
                declared_claim=DeclaredChainClaim('worker', 'worker', 'parent-a'),
            ),
            2: _facts(
                2,
                membership=worker_b,
                declared_claim=DeclaredChainClaim('worker', 'worker', 'parent-b'),
            ),
            3: _facts(
                3,
                notification_deliveries=(NotificationDelivery(BoundAnchor(3), 'worker'),),
            ),
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['worker']

    assert chain.parent_chain_id is None
    assert chain.evidence['parent_chain_candidates'] == ['parent-a', 'parent-b']
    assert chain.join == JoinLink(node=3, tool_use_id=None, kind='notify')


def test_structural_parent_precedes_an_external_parent_agent_claim():
    structural = SidechainMembership(chain_id='structural', kind='worker', detector='agent-id')
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='structural-before-external',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, children=[2], generated_by='llm'),
            2: _node(2, parent=1, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
        },
        roots=[0],
    )
    child_facts = _facts(3, membership=child, parent_claim=ParentClaim('external-parent'))
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=structural),
            1: _facts(1, membership=structural),
            2: _facts(2, membership=child),
            3: child_facts,
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['child']

    assert chain.parent_chain_id == 'structural'
    assert chain.evidence == {
        'anchor_node': 1,
        'correlation': {
            'fork_node_index': 1,
            'fork_segment_index': 0,
            'parent_agent_id': 'external-parent',
        },
    }


def test_external_parent_agent_is_the_last_parent_precedence_rung():
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='external-parent',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
        },
        roots=[0],
    )
    child_facts = _facts(1, membership=child, parent_claim=ParentClaim('external-parent'))
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=child),
            1: child_facts,
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['child']

    assert chain.parent_chain_id == 'external-parent'
    assert chain.evidence == {'correlation': {'parent_agent_id': 'external-parent'}}


def test_nonmember_parent_claim_with_same_context_does_not_leak_into_a_sidechain():
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='nonmember-parent-claim',
        nodes={
            0: _node(0, parent=None, generated_by='llm', content='main'),
            1: _node(1, parent=None, generated_by='llm', content='child'),
        },
        roots=[0, 1],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, parent_claim=ParentClaim('wrong-main-claim'), context_key='shared-context'),
            1: _facts(1, membership=child, context_key='shared-context'),
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['child']

    assert chain.parent_chain_id is None
    assert chain.evidence == {}


def test_conflicting_parent_agents_across_named_chain_roots_are_not_published_per_anchor():
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='conflicting-parent-agents',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
        },
        roots=[0, 2],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=child),
            1: _facts(1, membership=child, parent_claim=ParentClaim('parent-a')),
            2: _facts(2, membership=child),
            3: _facts(3, membership=child, parent_claim=ParentClaim('parent-b')),
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['child']

    assert chain.parent_chain_id is None
    assert chain.evidence == {}


def test_parent_agents_are_published_independently_for_distinct_sidechain_contexts():
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='distinct-parent-agent-contexts',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
        },
        roots=[0, 2],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=child, context_key='context-a'),
            1: _facts(
                1,
                membership=child,
                parent_claim=ParentClaim('parent-a'),
                context_key='context-a',
            ),
            2: _facts(2, membership=child, context_key='context-b'),
            3: _facts(
                3,
                membership=child,
                parent_claim=ParentClaim('parent-b'),
                context_key='context-b',
            ),
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['child']

    assert chain.parent_chain_id is None
    assert chain.evidence == {
        'parent_chain_candidates': ['parent-a', 'parent-b'],
        'correlation_candidates': [
            {'parent_agent_id': 'parent-a'},
            {'parent_agent_id': 'parent-b'},
        ],
    }


def test_latest_raw_receipt_per_tool_is_chosen_before_success_filtering_on_the_same_node():
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='latest-raw-receipt',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm', message_id=10),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
        },
        roots=[0, 2],
    )
    spawn = SpawnIntent(BoundAnchor(1), 'spawn-1', 'child prompt')
    delivered = DeliveryReceipt(BoundAnchor(1), 'spawn-1', True)
    superseding_false = DeliveryReceipt(BoundAnchor(1), 'spawn-1', False)
    source_facts = _facts(
        1,
        spawn_intents=(spawn,),
        delivery_receipts=(delivered, superseding_false),
        occurrences=(
            RuntimeFactOccurrence(0, spawn),
            RuntimeFactOccurrence(1, delivered),
            RuntimeFactOccurrence(2, superseding_false),
        ),
    )
    state = _state(
        trajectory,
        {
            0: _facts(0),
            1: source_facts,
            2: _facts(2, membership=child),
            3: _facts(3, membership=child, child_prompt='child prompt'),
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['child']

    assert chain.spawn is not None
    assert chain.spawn.tool_use_id == 'spawn-1'
    assert chain.join is None
    assert chain.evidence['correlation']['joined'] is False


def test_equal_message_id_successful_deliveries_remain_an_unresolved_join():
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='delivery-tie',
        nodes={
            0: _node(0, parent=None, children=[1, 2]),
            1: _node(1, parent=0, message_id=50, content='notification one'),
            2: _node(2, parent=0, message_id=50, content='notification two'),
            3: _node(3, parent=None, children=[4]),
            4: _node(4, parent=3, generated_by='llm'),
        },
        roots=[0, 3],
    )
    first_notification = _facts(
        1,
        notification_deliveries=(NotificationDelivery(BoundAnchor(1), 'child'),),
    )
    second_notification = _facts(
        2,
        notification_deliveries=(NotificationDelivery(BoundAnchor(2), 'child'),),
    )
    declared_child = _facts(
        4,
        membership=child,
        declared_claim=DeclaredChainClaim('child', 'worker', 'main'),
    )
    state = _state(
        trajectory,
        {
            0: _facts(0),
            1: first_notification,
            2: second_notification,
            3: _facts(3, membership=child),
            4: declared_child,
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['child']

    assert chain.parent_chain_id == 'main'
    assert chain.spawn is None
    assert chain.join is not None
    assert chain.join.node is None
    assert chain.join.tool_use_id is None
    assert chain.join.kind == 'notify'
    assert 'join_candidates' not in chain.evidence


def test_repeated_notification_fact_on_one_physical_node_keeps_a_unique_join_back_edge():
    child = SidechainMembership(chain_id='child', kind='worker', detector='declared')
    notification = NotificationDelivery(BoundAnchor(1), 'child')
    trajectory = Trajectory(
        session_id='repeated-notification',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, message_id=50, content='notification'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
        },
        roots=[0, 2],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0),
            1: _facts(
                1,
                occurrences=(
                    RuntimeFactOccurrence(10, notification),
                    RuntimeFactOccurrence(20, notification),
                ),
            ),
            2: _facts(2, membership=child),
            3: _facts(
                3,
                membership=child,
                declared_claim=DeclaredChainClaim('child', 'worker', 'main'),
            ),
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['child']

    assert chain.join is not None
    assert chain.join.node == 1
    assert chain.join.kind == 'notify'
    assert 'join_candidates' not in chain.evidence


def test_distinct_successful_deliveries_on_one_latest_node_remain_ambiguous_independent_of_fact_order():
    child = SidechainMembership(chain_id='child', kind='worker', parent_chain_id='main', detector='agent-id')

    def freeze_with_receipt_order(receipts):
        trajectory = Trajectory(
            session_id='same-node-delivery-ambiguity',
            nodes={
                0: _node(0, parent=None, children=[1]),
                1: _node(1, parent=0, generated_by='llm', message_id=50, content='delivery source'),
                2: _node(2, parent=None, children=[3]),
                3: _node(3, parent=2, generated_by='llm', content='child answer'),
            },
            roots=[0, 2],
        )
        spawn = SpawnIntent(BoundAnchor(1), 'spawn-a', 'child prompt')
        retrieval = RetrievalIntent(BoundAnchor(1), 'poll-b', 'child')
        occurrences = [RuntimeFactOccurrence(0, spawn), RuntimeFactOccurrence(1, retrieval)]
        occurrences.extend(
            RuntimeFactOccurrence(sequence, receipt) for sequence, receipt in enumerate(receipts, start=2)
        )
        state = _state(
            trajectory,
            {
                0: _facts(0),
                1: _facts(1, occurrences=tuple(occurrences)),
                2: _facts(2, membership=child, child_prompt='child prompt'),
                3: _facts(3, membership=child),
            },
        )
        return freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)

    inline = DeliveryReceipt(BoundAnchor(1), 'spawn-a', True)
    poll = DeliveryReceipt(BoundAnchor(1), 'poll-b', True)
    inline_first = freeze_with_receipt_order([inline, poll])
    poll_first = freeze_with_receipt_order([poll, inline])

    assert inline_first.dump() == poll_first.dump()
    child_chain = inline_first.chains['child']
    assert child_chain.join == JoinLink(node=1, tool_use_id=None, kind=None)
    assert child_chain.evidence['correlation']['joined'] is True
    assert 'join_tool_use_id' not in child_chain.evidence['correlation']
    assert 'join_kind' not in child_chain.evidence['correlation']


def test_equal_message_id_receipts_for_one_tool_remain_ambiguous_across_nodes():
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='receipt-tie',
        nodes={
            0: _node(0, parent=None, children=[1, 2]),
            1: _node(1, parent=0, message_id=50, content='first delivery'),
            2: _node(2, parent=0, message_id=50, content='second delivery'),
            3: _node(3, parent=None, children=[4]),
            4: _node(4, parent=3, generated_by='llm'),
        },
        roots=[0, 3],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0),
            1: _facts(
                1,
                spawn_intents=(SpawnIntent(BoundAnchor(1), 'spawn-1', 'child prompt'),),
                delivery_receipts=(DeliveryReceipt(BoundAnchor(1), 'spawn-1', True),),
            ),
            2: _facts(
                2,
                delivery_receipts=(DeliveryReceipt(BoundAnchor(2), 'spawn-1', True),),
            ),
            3: _facts(3, membership=child),
            4: _facts(4, membership=child, child_prompt='child prompt'),
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['child']

    assert chain.join is not None
    assert chain.join.node is None
    assert chain.join.tool_use_id == 'spawn-1'
    assert chain.join.kind == 'inline'
    assert 'join_candidates' not in chain.evidence


def test_equal_message_true_false_receipts_keep_runtime_ids_and_remain_raw_ambiguity():
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='mixed-receipt-tie',
        nodes={
            0: _node(0, parent=None, children=[2, 1], content='spawn source'),
            1: _node(1, parent=0, message_id=50, content='successful delivery'),
            2: _node(2, parent=0, message_id=50, content='failed delivery'),
            3: _node(3, parent=None, children=[4]),
            4: _node(4, parent=3, generated_by='llm'),
        },
        roots=[0, 3],
    )
    state = _state(
        trajectory,
        {
            0: _facts(
                0,
                spawn_intents=(SpawnIntent(BoundAnchor(0), 'spawn-1', 'child prompt'),),
            ),
            1: _facts(
                1,
                delivery_receipts=(DeliveryReceipt(BoundAnchor(1), 'spawn-1', True),),
            ),
            2: _facts(
                2,
                delivery_receipts=(DeliveryReceipt(BoundAnchor(2), 'spawn-1', False),),
            ),
            3: _facts(3, membership=child),
            4: _facts(4, membership=child, child_prompt='child prompt'),
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)
    chain = frozen.chains['child']
    correlation = chain.evidence['correlation']

    assert chain.spawn is not None
    assert chain.spawn.tool_use_id == 'spawn-1'
    assert chain.join is None
    assert correlation['delivery_ambiguous'] is True
    assert correlation['delivery_candidates'] == [
        {'node': 1, 'tool_use_id': 'spawn-1', 'delivered': True, 'message_id': 50},
        {'node': 2, 'tool_use_id': 'spawn-1', 'delivered': False, 'message_id': 50},
    ]
    assert 'joined' not in correlation
    dumped = frozen.dump()
    assert Trajectory.from_dump(dumped).dump() == dumped


@pytest.mark.parametrize('source_roots', [(0, 2), (2, 0)])
@pytest.mark.parametrize(
    'scenario',
    ['same-message-wrong-false', 'later-wrong-false', 'unrelated-conflict', 'wrong-parent-retrieval'],
)
def test_raw_delivery_facts_are_scoped_before_reduction(scenario, source_roots):
    local = SidechainMembership(chain_id='local', kind='worker', detector='agent-id')
    wrong = SidechainMembership(chain_id='wrong', kind='worker', detector='agent-id')
    child = SidechainMembership(chain_id='child', kind='worker', detector='declared')
    local_message_id = 40 if scenario == 'later-wrong-false' else 50
    trajectory = Trajectory(
        session_id=f'raw-scope-{scenario}',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm', message_id=local_message_id, content='local source'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm', message_id=50, content='wrong source'),
            4: _node(4, parent=None, children=[5]),
            5: _node(5, parent=4, generated_by='llm', content='child answer'),
        },
        roots=[*source_roots, 4],
    )
    local_spawn = ()
    local_retrieval = ()
    wrong_retrieval = ()
    local_receipts = ()
    wrong_receipts = ()
    child_prompt = None
    expected_join_kind = None
    expected_tool_id = None
    if scenario == 'same-message-wrong-false':
        child_prompt = 'child prompt'
        expected_join_kind = 'inline'
        expected_tool_id = 'tool-1'
        local_spawn = (SpawnIntent(BoundAnchor(1), 'tool-1', child_prompt),)
        local_receipts = (DeliveryReceipt(BoundAnchor(1), 'tool-1', True),)
        wrong_receipts = (DeliveryReceipt(BoundAnchor(3), 'tool-1', False),)
    elif scenario == 'later-wrong-false':
        expected_join_kind = 'poll'
        expected_tool_id = 'tool-1'
        local_retrieval = (RetrievalIntent(BoundAnchor(1), 'tool-1', 'child'),)
        local_receipts = (DeliveryReceipt(BoundAnchor(1), 'tool-1', True),)
        wrong_receipts = (DeliveryReceipt(BoundAnchor(3), 'tool-1', False),)
    elif scenario == 'unrelated-conflict':
        child_prompt = 'child prompt'
        local_spawn = (SpawnIntent(BoundAnchor(1), 'relevant-tool', child_prompt),)
        local_receipts = (DeliveryReceipt(BoundAnchor(1), 'unrelated-tool', True),)
        wrong_receipts = (DeliveryReceipt(BoundAnchor(3), 'unrelated-tool', False),)
    else:
        wrong_retrieval = (RetrievalIntent(BoundAnchor(3), 'wrong-poll', 'child'),)
        local_receipts = (DeliveryReceipt(BoundAnchor(1), 'wrong-poll', True),)

    state = _state(
        trajectory,
        {
            0: _facts(0, membership=local),
            1: _facts(
                1,
                membership=local,
                spawn_intents=local_spawn,
                retrieval_intents=local_retrieval,
                delivery_receipts=local_receipts,
            ),
            2: _facts(2, membership=wrong),
            3: _facts(
                3,
                membership=wrong,
                retrieval_intents=wrong_retrieval,
                delivery_receipts=wrong_receipts,
            ),
            4: _facts(4, membership=child),
            5: _facts(
                5,
                membership=child,
                child_prompt=child_prompt,
                declared_claim=DeclaredChainClaim('child', 'worker', 'local'),
            ),
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)
    chain = frozen.chains['child']
    correlation = chain.evidence['correlation']

    assert chain.parent_chain_id == 'local'
    if expected_join_kind is not None:
        assert chain.join is not None
        assert chain.join.kind == expected_join_kind
        assert chain.join.tool_use_id == expected_tool_id
        assert frozen.nodes[chain.join.node].original_message['content'] == 'local source'
    else:
        assert chain.join is None
        assert 'delivery_ambiguous' not in correlation
        assert 'delivery_candidates' not in correlation
    dumped = frozen.dump()
    assert Trajectory.from_dump(dumped).dump() == dumped


@pytest.mark.parametrize('source_children', [(1, 2), (2, 1)])
def test_stale_success_does_not_infer_join_parent_after_a_later_false_receipt(source_children):
    source = SidechainMembership(chain_id='source', kind='worker', detector='agent-id')
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='stale-join-parent',
        nodes={
            0: _node(0, parent=None, children=list(source_children)),
            1: _node(1, parent=0, generated_by='llm', message_id=40, content='earlier success'),
            2: _node(2, parent=0, generated_by='llm', message_id=50, content='later false'),
            3: _node(3, parent=None, children=[4]),
            4: _node(4, parent=3, generated_by='llm', content='child answer'),
        },
        roots=[0, 3],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=source),
            1: _facts(
                1,
                membership=source,
                retrieval_intents=(RetrievalIntent(BoundAnchor(1), 'poll-1', 'child'),),
                delivery_receipts=(DeliveryReceipt(BoundAnchor(1), 'poll-1', True),),
            ),
            2: _facts(
                2,
                membership=source,
                delivery_receipts=(DeliveryReceipt(BoundAnchor(2), 'poll-1', False),),
            ),
            3: _facts(3, membership=child),
            4: _facts(4, membership=child),
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)
    chain = frozen.chains['child']

    assert chain.parent_chain_id is None
    assert chain.join is None
    assert 'parent_resolution' not in chain.evidence
    dumped = frozen.dump()
    assert Trajectory.from_dump(dumped).dump() == dumped


def test_latest_successful_delivery_uses_message_id_not_dfs_node_order():
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='delivery-message-id',
        nodes={
            0: _node(0, parent=None, children=[1, 2]),
            1: _node(1, parent=0, message_id=200, content='chronologically latest'),
            2: _node(2, parent=0, message_id=100, content='later dfs node'),
            3: _node(3, parent=None, children=[4]),
            4: _node(4, parent=3, generated_by='llm'),
        },
        roots=[0, 3],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0),
            1: _facts(
                1,
                notification_deliveries=(NotificationDelivery(BoundAnchor(1), 'child'),),
            ),
            2: _facts(
                2,
                notification_deliveries=(NotificationDelivery(BoundAnchor(2), 'child'),),
            ),
            3: _facts(3, membership=child),
            4: _facts(
                4,
                membership=child,
                declared_claim=DeclaredChainClaim('child', 'worker', 'main'),
            ),
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['child']

    assert chain.join is not None
    assert chain.join.node == 1
    assert chain.join.kind == 'notify'


def test_notification_only_declared_chain_resolves_without_a_child_prompt():
    child = SidechainMembership(chain_id='child', kind='worker', detector='declared')
    trajectory = Trajectory(
        session_id='notification-only',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, message_id=12),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
        },
        roots=[0, 2],
    )
    notification = _facts(
        1,
        notification_deliveries=(NotificationDelivery(BoundAnchor(1), 'child'),),
    )
    declared_child = _facts(
        3,
        membership=child,
        declared_claim=DeclaredChainClaim('child', 'worker', 'main'),
    )
    state = _state(
        trajectory,
        {
            0: _facts(0),
            1: notification,
            2: _facts(2, membership=child),
            3: declared_child,
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['child']

    assert chain.parent_chain_id == 'main'
    assert chain.spawn is None
    assert chain.join is not None
    assert chain.join.node == 1
    assert chain.join.kind == 'notify'


def test_child_prompt_candidate_order_precedes_storage_order():
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='ordered-child-prompt',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
        },
        roots=[0, 2],
    )
    source = _facts(
        1,
        spawn_intents=(
            SpawnIntent(BoundAnchor(1), 'spawn-later', 'later prompt'),
            SpawnIntent(BoundAnchor(1), 'spawn-first', 'first prompt'),
        ),
    )
    child_facts = _facts(
        3,
        membership=child,
        child_prompts=(
            ChildPromptCandidate(BoundAnchor(3), 'later prompt', 5),
            ChildPromptCandidate(BoundAnchor(3), 'first prompt', 0),
        ),
    )
    state = _state(
        trajectory,
        {
            0: _facts(0),
            1: source,
            2: _facts(2, membership=child),
            3: child_facts,
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['child']

    assert chain.spawn is not None
    assert chain.spawn.tool_use_id == 'spawn-first'


def test_structural_fork_evidence_counts_the_parent_path_and_compaction_segments():
    structural = SidechainMembership(chain_id='structural', kind='worker', detector='agent-id')
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='structural-fork',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, children=[2], generated_by='llm'),
            2: _node(2, parent=1, children=[3], generated_by='observation'),
            3: _node(3, parent=2, children=[4], generated_by='llm'),
            4: _node(4, parent=3, children=[5]),
            5: _node(5, parent=4, generated_by='llm'),
        },
        roots=[0],
    )
    trajectory.nodes[3].segment_boundary = SegmentBoundary(
        starts_new_segment=True,
        kind='compaction_resume',
        causes=('compaction',),
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=structural),
            1: _facts(1, membership=structural),
            2: _facts(2, membership=structural),
            3: _facts(3, membership=structural),
            4: _facts(4, membership=child),
            5: _facts(5, membership=child),
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['child']

    assert chain.parent_chain_id == 'structural'
    assert chain.evidence['correlation']['fork_node_index'] == 3
    assert chain.evidence['correlation']['fork_segment_index'] == 1


def test_declared_self_parent_is_preserved_for_baseline_dump_parity():
    child = SidechainMembership(
        chain_id='child',
        kind='worker',
        parent_chain_id='child',
        detector='declared_chain',
        attributes=(('parent_chain_id', 'child'),),
    )
    trajectory = Trajectory(
        session_id='self-parent',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
        },
        roots=[0],
    )
    child_facts = _facts(
        1,
        membership=child,
        declared_claim=DeclaredChainClaim('child', 'worker', 'child'),
    )
    state = _state(trajectory, {0: _facts(0, membership=child), 1: child_facts})

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['child']

    assert chain.parent_chain_id == 'child'
    assert chain.evidence == {
        'attributes': {'parent_chain_id': 'child'},
        'correlation': {'parent_chain_id': 'child'},
    }


def test_parent_claim_self_is_evidence_but_not_a_parent_candidate():
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='self-parent-agent-claim',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
        },
        roots=[0],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=child),
            1: _facts(1, membership=child, parent_claim=ParentClaim('child')),
        },
    )

    chain = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).chains['child']

    assert chain.parent_chain_id is None
    assert chain.evidence == {'correlation': {'parent_agent_id': 'child'}}


def test_declared_logical_parent_cycle_is_preserved_for_baseline_dump_parity():
    first = SidechainMembership(
        chain_id='first',
        kind='worker',
        parent_chain_id='second',
        detector='declared_chain',
        attributes=(('parent_chain_id', 'second'),),
    )
    second = SidechainMembership(
        chain_id='second',
        kind='worker',
        parent_chain_id='first',
        detector='declared_chain',
        attributes=(('parent_chain_id', 'first'),),
    )
    trajectory = Trajectory(
        session_id='logical-parent-cycle',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
        },
        roots=[0, 2],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=first),
            1: _facts(
                1,
                membership=first,
                declared_claim=DeclaredChainClaim('first', 'worker', 'second'),
            ),
            2: _facts(2, membership=second),
            3: _facts(
                3,
                membership=second,
                declared_claim=DeclaredChainClaim('second', 'worker', 'first'),
            ),
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)

    assert list(frozen.chains) == ['main', 'first', 'second']
    assert frozen.chains['first'].parent_chain_id == 'second'
    assert frozen.chains['first'].evidence == {
        'attributes': {'parent_chain_id': 'second'},
        'correlation': {'parent_chain_id': 'second'},
    }
    assert frozen.chains['second'].parent_chain_id == 'first'
    assert frozen.chains['second'].evidence == {
        'attributes': {'parent_chain_id': 'first'},
        'correlation': {'parent_chain_id': 'first'},
    }


@pytest.mark.parametrize(
    ('second_kind', 'second_detector'),
    [('reviewer', 'declared'), ('worker', 'tool-call')],
    ids=['kind-conflict', 'detector-conflict'],
)
def test_named_chain_kind_or_detector_conflict_creates_a_collision_chain(second_kind, second_detector):
    first = SidechainMembership(chain_id='duplicate', kind='worker', detector='declared')
    second = SidechainMembership(chain_id='duplicate', kind=second_kind, detector=second_detector)
    trajectory = Trajectory(
        session_id='chain-profile-collision',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
        },
        roots=[0, 2],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=first),
            1: _facts(1, membership=first),
            2: _facts(2, membership=second),
            3: _facts(3, membership=second),
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)

    assert list(frozen.chains) == ['main', 'duplicate', 'duplicate@2']
    assert frozen.chains['duplicate'].roots == [0]
    assert frozen.chains['duplicate@2'].roots == [2]
    assert frozen.chains['duplicate@2'].kind == second_kind
    assert frozen.chains['duplicate@2'].detector == second_detector
    assert frozen.chains['duplicate@2'].evidence['collides_with'] == 'duplicate'


def test_named_chain_profile_conflict_under_a_shared_parent_splits_only_the_conflicting_child():
    first = SidechainMembership(chain_id='duplicate', kind='worker', detector='declared')
    second = SidechainMembership(chain_id='duplicate', kind='reviewer', detector='declared')
    trajectory = Trajectory(
        session_id='nested-chain-profile-collision',
        nodes={
            0: _node(0, parent=None, children=[1, 2]),
            1: _node(1, parent=0, generated_by='llm'),
            2: _node(2, parent=0, generated_by='llm'),
        },
        roots=[0],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=first),
            1: _facts(1, membership=first),
            2: _facts(2, membership=second),
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)

    assert frozen.chains['main'].roots == []
    assert frozen.chains['duplicate'].roots == [0]
    assert frozen.chains['duplicate@2'].roots == [2]
    assert frozen.chains['duplicate@2'].parent_chain_id == 'duplicate'
    assert frozen.chains['duplicate@2'].kind == 'reviewer'


def test_named_chain_collisions_are_deterministic_and_create_a_fresh_bucket_per_root():
    memberships = (
        SidechainMembership(chain_id='duplicate', kind='worker', detector='one'),
        SidechainMembership(chain_id='duplicate', kind='worker', detector='two'),
        SidechainMembership(chain_id='duplicate@2', kind='worker', detector='three'),
        SidechainMembership(chain_id='duplicate', kind='worker', detector='two'),
    )
    declared_parents = ('parent-a', 'parent-b', 'parent-c', 'parent-b')
    nodes: dict[int, TrajectoryNode] = {}
    roots: list[int] = []
    fact_map: dict[int, NodeRuntimeFacts] = {}
    for index, (membership, declared_parent) in enumerate(zip(memberships, declared_parents, strict=True)):
        root_id = index * 2
        llm_id = root_id + 1
        nodes[root_id] = _node(root_id, parent=None, children=[llm_id])
        nodes[llm_id] = _node(llm_id, parent=root_id, generated_by='llm')
        roots.append(root_id)
        fact_map[root_id] = _facts(root_id, membership=membership)
        fact_map[llm_id] = _facts(
            llm_id,
            membership=membership,
            declared_claim=DeclaredChainClaim(membership.chain_id, membership.kind, declared_parent),
        )
    trajectory = Trajectory(session_id='chain-collisions', nodes=nodes, roots=roots)

    frozen = freeze_trajectory(trajectory, _state(trajectory, fact_map), ResponsePolicy.KEEP_ALL)

    assert list(frozen.chains) == ['main', 'duplicate', 'duplicate@2', 'duplicate@2@4', 'duplicate@6']
    assert frozen.chains['duplicate'].roots == [0]
    assert frozen.chains['duplicate@2'].roots == [2]
    assert frozen.chains['duplicate@2@4'].roots == [4]
    assert frozen.chains['duplicate@6'].roots == [6]
    assert frozen.chains['duplicate@2'].evidence['collides_with'] == 'duplicate'
    assert frozen.chains['duplicate@2@4'].evidence['collides_with'] == 'duplicate@2'
    assert frozen.chains['duplicate@6'].evidence['collides_with'] == 'duplicate'


def test_named_chain_collision_bucket_is_not_reused_for_a_later_matching_parent():
    membership = SidechainMembership(chain_id='worker', kind='worker', detector='declared')
    trajectory = Trajectory(
        session_id='repeat-collision-parent',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
            4: _node(4, parent=None, children=[5]),
            5: _node(5, parent=4, generated_by='llm'),
        },
        roots=[0, 2, 4],
    )
    fact_map = {root_id: _facts(root_id, membership=membership) for root_id in (0, 2, 4)}
    fact_map.update({
        1: _facts(
            1,
            membership=membership,
            declared_claim=DeclaredChainClaim('worker', 'worker', 'parent-a'),
        ),
        3: _facts(
            3,
            membership=membership,
            declared_claim=DeclaredChainClaim('worker', 'worker', 'parent-b'),
        ),
        5: _facts(
            5,
            membership=membership,
            declared_claim=DeclaredChainClaim('worker', 'worker', 'parent-b'),
        ),
    })

    frozen = freeze_trajectory(trajectory, _state(trajectory, fact_map), ResponsePolicy.KEEP_ALL)

    assert list(frozen.chains) == ['main', 'worker', 'worker@2', 'worker@4']
    assert frozen.chains['worker'].roots == [0]
    assert frozen.chains['worker@2'].roots == [2]
    assert frozen.chains['worker@4'].roots == [4]


def test_named_chain_declared_parent_does_not_upgrade_an_unparented_base_chain():
    membership = SidechainMembership(chain_id='worker', kind='worker', detector='declared')
    trajectory = Trajectory(
        session_id='unparented-base-collision',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
        },
        roots=[0, 2],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=membership),
            1: _facts(1, membership=membership),
            2: _facts(2, membership=membership),
            3: _facts(
                3,
                membership=membership,
                declared_claim=DeclaredChainClaim('worker', 'worker', 'parent-a'),
            ),
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)

    assert list(frozen.chains) == ['main', 'worker', 'worker@2']
    assert frozen.chains['worker'].roots == [0]
    assert frozen.chains['worker@2'].roots == [2]


def test_shared_prefix_declared_parent_conflicts_stay_in_one_chain_with_baseline_dump_order():
    assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent', session_id='shared-prefix-parent-conflict')
    prompt = [{'role': 'user', 'content': 'same'}]
    assembler.record_completion(
        prompt,
        {'role': 'assistant', 'content': 'answer a'},
        request_meta={'declared_chain': {'chain_id': 'worker', 'kind': 'worker', 'parent_chain_id': 'parent-a'}},
    )
    assembler.record_completion(
        prompt,
        {'role': 'assistant', 'content': 'answer b'},
        request_meta={'declared_chain': {'chain_id': 'worker', 'kind': 'worker', 'parent_chain_id': 'parent-b'}},
    )

    frozen = assembler.freeze()
    worker_dump = next(chain for chain in frozen.dump()['chains'] if chain['chain_id'] == 'worker')

    assert list(frozen.chains) == ['main', 'worker']
    assert frozen.roots == [0]
    assert frozen.nodes[0].children == [1, 2]
    assert worker_dump == {
        'chain_id': 'worker',
        'kind': 'worker',
        'parent_chain_id': None,
        'roots': [0],
        'detector': 'declared_chain',
        'evidence': {
            'attribute_candidates': {'parent_chain_id': ['parent-a', 'parent-b']},
            'parent_chain_candidates': ['parent-a', 'parent-b'],
            'correlation_candidates': [
                {'parent_chain_id': 'parent-b'},
                {'parent_chain_id': 'parent-a'},
            ],
        },
    }


def test_preoccupied_named_collision_id_raises_instead_of_silently_adding_a_suffix():
    worker_a = SidechainMembership(chain_id='worker', kind='worker', detector='declared')
    occupied = SidechainMembership(chain_id='worker@4', kind='worker', detector='declared')
    worker_b = SidechainMembership(chain_id='worker', kind='worker', detector='declared')
    trajectory = Trajectory(
        session_id='occupied-named-collision',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
            4: _node(4, parent=None, children=[5]),
            5: _node(5, parent=4, generated_by='llm'),
        },
        roots=[0, 2, 4],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=worker_a),
            1: _facts(
                1,
                membership=worker_a,
                declared_claim=DeclaredChainClaim('worker', 'worker', 'parent-a'),
            ),
            2: _facts(2, membership=occupied),
            3: _facts(3, membership=occupied),
            4: _facts(4, membership=worker_b),
            5: _facts(
                5,
                membership=worker_b,
                declared_claim=DeclaredChainClaim('worker', 'worker', 'parent-b'),
            ),
        },
    )

    with pytest.raises(ValueError, match=r"sidechain id collision: 'worker@4'"):
        freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)


def test_preoccupied_anonymous_chain_id_raises_instead_of_silently_adding_a_suffix():
    occupied = SidechainMembership(chain_id='chain@2', kind='named', detector='declared')
    anonymous = SidechainMembership(chain_id='', kind='anonymous', detector='anonymous')
    trajectory = Trajectory(
        session_id='occupied-anonymous-collision',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
        },
        roots=[0, 2],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0, membership=occupied),
            1: _facts(1, membership=occupied),
            2: _facts(2, membership=anonymous),
            3: _facts(3, membership=anonymous),
        },
    )

    with pytest.raises(ValueError, match=r"anonymous chain id collision: 'chain@2'"):
        freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)


@pytest.mark.parametrize('delivery_kind', ['poll', 'notify'])
def test_collision_bucket_uses_wire_agent_id_before_parent_scoping(delivery_kind):
    parent_a = SidechainMembership(chain_id='parent-a', kind='worker', detector='agent-id')
    parent_b = SidechainMembership(chain_id='parent-b', kind='worker', detector='agent-id')
    worker = SidechainMembership(chain_id='worker', kind='worker', detector='declared')
    trajectory = Trajectory(
        session_id=f'collision-{delivery_kind}',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm', content=f'{delivery_kind} source a'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm', content=f'{delivery_kind} source b'),
            4: _node(4, parent=None, children=[5]),
            5: _node(5, parent=4, generated_by='llm', content='worker a'),
            6: _node(6, parent=None, children=[7]),
            7: _node(7, parent=6, generated_by='llm', content='worker b'),
        },
        roots=[0, 2, 4, 6],
    )

    def source_facts(node_id, membership, suffix):
        if delivery_kind == 'poll':
            return _facts(
                node_id,
                membership=membership,
                retrieval_intents=(RetrievalIntent(BoundAnchor(node_id), f'poll-{suffix}', 'worker'),),
                delivery_receipts=(DeliveryReceipt(BoundAnchor(node_id), f'poll-{suffix}', True),),
            )
        return _facts(
            node_id,
            membership=membership,
            notification_deliveries=(NotificationDelivery(BoundAnchor(node_id), 'worker'),),
        )

    state = _state(
        trajectory,
        {
            0: _facts(0, membership=parent_a),
            1: source_facts(1, parent_a, 'a'),
            2: _facts(2, membership=parent_b),
            3: source_facts(3, parent_b, 'b'),
            4: _facts(4, membership=worker),
            5: _facts(
                5,
                membership=worker,
                declared_claim=DeclaredChainClaim('worker', 'worker', 'parent-a'),
            ),
            6: _facts(6, membership=worker),
            7: _facts(
                7,
                membership=worker,
                declared_claim=DeclaredChainClaim('worker', 'worker', 'parent-b'),
            ),
        },
    )

    frozen = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)
    collision_id = next(
        chain_id for chain_id, chain in frozen.chains.items() if chain.evidence.get('collides_with') == 'worker'
    )
    collision = frozen.chains[collision_id]

    assert collision.parent_chain_id == 'parent-b'
    assert collision.join is not None
    assert collision.join.kind == delivery_kind
    assert frozen.nodes[collision.join.node].original_message['content'] == f'{delivery_kind} source b'
    if delivery_kind == 'poll':
        assert collision.join.tool_use_id == 'poll-b'
    else:
        assert collision.join.tool_use_id is None


def test_canonical_phase_rejects_a_runtime_fact_anchor_outside_the_physical_forest():
    trajectory = Trajectory(
        session_id='invalid-runtime-anchor',
        nodes={0: _node(0, parent=None, generated_by='llm')},
        roots=[0],
    )
    invalid = _facts(
        0,
        spawn_intents=(SpawnIntent(BoundAnchor(99), 'spawn-1', 'child'),),
    )
    state = _state(trajectory, {0: invalid})

    with pytest.raises(ValueError, match='runtime fact anchor 99 does not reference a trajectory node'):
        freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)


def test_resolver_output_is_stable_across_refreeze_dump_and_pickle_round_trip():
    child = SidechainMembership(chain_id='child', kind='worker', detector='agent-id')
    trajectory = Trajectory(
        session_id='resolver-stability',
        nodes={
            0: _node(0, parent=None, children=[1]),
            1: _node(1, parent=0, generated_by='llm'),
            2: _node(2, parent=None, children=[3]),
            3: _node(3, parent=2, generated_by='llm'),
        },
        roots=[0, 2],
    )
    state = _state(
        trajectory,
        {
            0: _facts(0),
            1: _facts(
                1,
                spawn_intents=(SpawnIntent(BoundAnchor(1), 'spawn-1', 'child prompt'),),
            ),
            2: _facts(2, membership=child),
            3: _facts(3, membership=child, child_prompt='child prompt'),
        },
    )
    expected = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL).dump()

    rebuilt = freeze_trajectory(trajectory, state, ResponsePolicy.KEEP_ALL)

    assert rebuilt.dump() == expected
    assert pickle.loads(pickle.dumps(rebuilt)).dump() == expected
    assert Trajectory.from_dump(expected).dump() == expected
