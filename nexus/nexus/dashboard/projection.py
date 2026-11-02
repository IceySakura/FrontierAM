"""Presentation-only graph projection; never infer causality from timestamps."""

from __future__ import annotations

from typing import Any

from nexus.gateway.trajectory import SegmentNodeList, Trajectory


def _segment_tokens(segment: SegmentNodeList) -> dict[str, int] | None:
    last = segment.nodes.index(segment.last_response_node)
    accounts = [node.token_account for node in segment.nodes[: last + 1]]
    if any(account is None or account.token_ids is None for account in accounts):
        return None
    lengths = [len(account.token_ids) for account in accounts if account is not None and account.token_ids is not None]
    first = segment.first_response_node_index
    first_account = accounts[first]
    assert first_account is not None
    return {
        'prompt': sum(lengths[:first]) + first_account.gen_start,
        'response': sum(lengths[first:]) - first_account.gen_start,
        'generated': sum(account.gen_len for account in accounts[first:] if account is not None),
    }


def _project_segments(trajectory: Trajectory) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    for segment in trajectory.to_segment_node_lists():
        boundary = segment.response_nodes[0].segment_boundary
        segments.append({
            'segment_id': segment.segment_id,
            'chain_id': segment.chain_id,
            'source_node_ids': [node.node_id for node in segment.nodes],
            'target_node_ids': [node.node_id for node in segment.response_nodes],
            'first_response_node_index': segment.first_response_node_index,
            'kind': (boundary.kind or 'boundary') if boundary is not None else 'conversation',
            'boundary': boundary.to_dict() if boundary is not None else None,
            'tokens': _segment_tokens(segment),
        })
    return segments


def _label(node: dict[str, Any]) -> str:
    message = node['original_message']
    tools = []
    for call in message.get('tool_calls') or []:
        if isinstance(call, dict):
            tools.append(str(call.get('function', {}).get('name', 'tool')))
    content = message.get('content')
    if isinstance(content, list):
        tools.extend(
            str(part.get('name', 'tool'))
            for part in content
            if isinstance(part, dict) and part.get('type') == 'tool_use'
        )
    if tools:
        return ', '.join(tools)
    return str(message.get('name') or message.get('role') or node['generated_by'])


def project_trajectory(data: dict[str, Any]) -> dict[str, Any]:
    """Keep original IDs; aggregate only unambiguous, linear, same-label runs.

    Spawn/join facts target a chain, not a guessed leaf or timestamp. An explicit
    chain endpoint allows the UI to draw that relation without inventing an
    exact child completion node, which JoinLink does not record.
    """
    trajectory = Trajectory.from_dump(data)
    nodes = {node['node_id']: node for node in data['nodes']}
    chains = {chain['chain_id']: chain for chain in data['chains']}
    membership = {node_id: trajectory.node_chain(node_id) for node_id in nodes}
    lane_ids = list(chains)
    for chain_id in membership.values():
        if chain_id not in lane_ids:
            lane_ids.append(chain_id)
    if 'main' in lane_ids:
        lane_ids.remove('main')
        lane_ids.insert(0, 'main')
    lanes = []
    for chain_id in lane_ids:
        chain = chains.get(chain_id)
        lanes.append({
            'id': chain_id,
            'kind': chain['kind'] if chain else 'unclassified',
            'parent_chain_id': chain.get('parent_chain_id') if chain else None,
            'declared': chain is not None,
            'node_ids': [node_id for node_id in sorted(nodes) if membership[node_id] == chain_id],
            'facts': chain,
        })
    anchors = set(data['roots'])
    warnings = []
    links = []
    for chain_id, chain in chains.items():
        anchors.update(chain['roots'])
        for relation in ('spawn', 'join'):
            link = chain.get(relation)
            if link is None:
                continue
            node_id = link.get('node')
            if node_id is None:
                warnings.append(f'{chain_id}: unresolved {relation}; evidence retained in chain details')
                continue
            anchors.add(node_id)
            node_endpoint, chain_endpoint = f'n:{node_id}', f'c:{chain_id}'
            links.append({
                'source': node_endpoint if relation == 'spawn' else chain_endpoint,
                'target': chain_endpoint if relation == 'spawn' else node_endpoint,
                'kind': relation,
                'chain_id': chain_id,
                'facts': link,
            })
    for node in nodes.values():
        if node.get('segment_boundary') or node.get('selection') == 'rejected_sibling':
            anchors.add(node['node_id'])
        if len(node['children']) != 1:
            anchors.add(node['node_id'])
        if node['parent'] is not None and membership[node['parent']] != membership[node['node_id']]:
            anchors.update((node['parent'], node['node_id']))
    raw_nodes: list[dict[str, Any]] = [
        {
            'id': f'n:{node_id}',
            'node_ids': [node_id],
            'lane': membership[node_id],
            'label': _label(node),
            'kind': 'boundary' if node.get('segment_boundary') else node['generated_by'],
            'selection': node.get('selection'),
            'boundary': node.get('segment_boundary'),
        }
        for node_id, node in sorted(nodes.items())
    ]
    physical = [
        {
            'source': f'n:{node["parent"]}',
            'target': f'n:{node_id}',
            'kind': 'physical',
        }
        for node_id, node in nodes.items()
        if node['parent'] is not None
    ]
    overview = []
    owners: dict[int, dict[str, Any]] = {}
    for item in raw_nodes:
        node_id = item['node_ids'][0]
        parent_id = nodes[node_id]['parent']
        previous = owners.get(parent_id) if parent_id is not None else None
        if (
            previous is not None
            and node_id not in anchors
            and parent_id not in anchors
            and previous['lane'] == item['lane']
            and previous['label'] == item['label']
            and previous['kind'] == item['kind']
            and previous['selection'] == item['selection']
        ):
            previous['node_ids'].append(node_id)
            owners[node_id] = previous
        else:
            group = {**item, 'node_ids': list(item['node_ids'])}
            overview.append(group)
            owners[node_id] = group
    projected_edges = []
    seen = set()
    for edge in physical + links:
        source, target = edge['source'], edge['target']
        if source.startswith('n:'):
            source = owners[int(source[2:])]['id']
        if target.startswith('n:'):
            target = owners[int(target[2:])]['id']
        key = (source, target, edge['kind'])
        if source != target and key not in seen:
            seen.add(key)
            projected_edges.append({**edge, 'source': source, 'target': target})
    return {
        'layout': 'assembly-order-not-time',
        'lanes': lanes,
        'raw': {'nodes': raw_nodes, 'edges': physical + links},
        'overview': {'nodes': overview, 'edges': projected_edges},
        'segments': _project_segments(trajectory),
        'warnings': warnings,
    }
