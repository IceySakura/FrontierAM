"""Deterministic topology fixtures for dashboard regression tests."""

from __future__ import annotations

from nexus.gateway.observers.file import FileTrajectoryObserver
from nexus.gateway.trajectory import Chain, JoinLink, SegmentBoundary, SpawnLink, Trajectory, TrajectoryNode


def branched_trajectory() -> Trajectory:
    # IDs are stable because projection tests assert grouping and lane ownership.
    topology = [
        (0, None, 'system'),
        (1, 0, 'user'),
        (2, 1, 'assistant'),
        (3, 2, 'assistant'),
        (4, 2, 'assistant'),
        (5, 2, 'assistant'),
        (6, 3, 'tool'),
        (7, 6, 'tool'),
        (8, 7, 'tool'),
        (9, 4, 'tool'),
        (10, 5, 'tool'),
        (11, 8, 'assistant'),
        (12, 9, 'assistant'),
        (13, 10, 'tool'),
        (14, 13, 'assistant'),
        (15, 14, 'assistant'),
        (16, 14, 'assistant'),
    ]
    tools = {2: 'spawn_agent', 3: 'Read', 4: 'Read', 5: 'Read'}
    nodes = {}
    for node_id, parent, role in topology:
        node = TrajectoryNode(
            node_id=node_id,
            message_id=node_id,
            parent=parent,
            children=[],
            generated_by='llm' if role == 'assistant' else 'observation' if role == 'tool' else 'client',
            original_message={'role': role, 'content': f'Node {node_id}'},
        )
        if node_id in tools:
            node.original_message['tool_calls'] = [{
                'id': f'call-{node_id}',
                'type': 'function',
                'function': {'name': tools[node_id], 'arguments': '{}'},
            }]
        nodes[node_id] = node
        if parent is not None:
            nodes[parent].children.append(node_id)
    nodes[14].segment_boundary = SegmentBoundary(
        starts_new_segment=True,
        kind='compaction',
        causes=('context_compaction',),
        evidence={'source': 'test-fixture'},
    )
    nodes[15].selection = 'rejected_sibling'
    nodes[16].selection = 'selected'
    return Trajectory(
        session_id='test / branched-topology',
        nodes=nodes,
        roots=[0],
        chains={
            'main': Chain(chain_id='main', roots=[0]),
            'research': Chain(
                chain_id='research', kind='subagent', roots=[3], parent_chain_id='main',
                spawn=SpawnLink(node=2, tool_use_id='call-2'), join=JoinLink(node=13, kind='poll'),
            ),
            'tests': Chain(
                chain_id='tests', kind='subagent', roots=[4], parent_chain_id='main',
                spawn=SpawnLink(node=2, tool_use_id='call-2'), join=JoinLink(node=13, kind='poll'),
            ),
        },
        final_reward=1.0,
        termination_reason='completed',
        extra={'synthetic': True},
    )


def write_trajectory(directory: str) -> None:
    FileTrajectoryObserver(directory).on_final(branched_trajectory())
