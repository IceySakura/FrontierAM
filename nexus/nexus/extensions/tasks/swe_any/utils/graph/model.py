"""Entity/relation graph model — schema accessors + JSON IO.

The lightweight, protobuf-free core of the pipeline: ``scip`` + ``parse``
construct the graph from a SCIP index, while ``select`` (and later Mask/Eval)
slice / consume it via JSON IO. Depends only on ``networkx`` and ``.symbol``,
so any stage can import it without pulling the SCIP/protobuf toolchain (that
lives in the ``scip`` package). Nodes carry a single ``info: dict``; edges carry
``infos: list[dict]`` so parallel relation facts on the same ``(src, dst)`` stay
separate (plain ``DiGraph``, not MultiDiGraph).

Entity/relation catalogue (E01 enclosing, E02 implementation, E03 reference) is
documented in the task package's ``README.md``.
"""

from __future__ import annotations

import json

from pathlib import Path

import networkx as nx

from .symbol import parse_symbol


def graph_info(graph: nx.DiGraph) -> dict:
    return graph.graph.setdefault('info', {})


def node_info(graph: nx.DiGraph, symbol: str) -> dict:
    return graph.nodes[symbol].setdefault('info', {})


def edge_infos(graph: nx.DiGraph, src: str, dst: str) -> list[dict]:
    # List (not a single dict): each parallel relation fact on this DiGraph edge
    # is one entry, so enclosing + reference on the same (src, dst) are both kept.
    return graph.edges[src, dst].setdefault('infos', [])


def ensure_node(graph: nx.DiGraph, symbol: str, info: dict | None = None) -> None:
    """Create ``symbol`` with ``info`` if missing; leave an existing node alone."""
    if symbol not in graph:
        graph.add_node(symbol, info=dict(info or {}))


def add_edge_info(graph: nx.DiGraph, src: str, dst: str, info: dict) -> None:
    """Append one relation fact onto edge ``src->dst``'s ``infos`` list."""
    if not src or not dst or src == dst:
        return
    ensure_node(graph, src, {'external': True})
    ensure_node(graph, dst, {'external': True})
    if not graph.has_edge(src, dst):
        # infos: one dict per logical edge fact (preserves multi-relation / multi-edge).
        graph.add_edge(src, dst, infos=[info])
        return
    edge_infos(graph, src, dst).append(info)


def filter_graph(
    graph: nx.DiGraph,
    *,
    include_external: bool = True,
    include_tests: bool = True,
) -> nx.DiGraph:
    """Return a new DiGraph keeping only nodes/edges matching the filters.

    When ``include_external`` is false, drop ``external`` nodes and edge facts
    with ``dst_external``. When ``include_tests`` is false, drop ``is_test``
    nodes (and edges that touch them). Always returns a fresh graph.
    """
    if include_external and include_tests:
        out = nx.DiGraph()
        graph_info(out).update(dict(graph_info(graph)))
        for n, data in graph.nodes(data=True):
            out.add_node(n, info=dict(data.get('info') or {}))
        for src, dst, data in graph.edges(data=True):
            out.add_edge(src, dst, infos=[dict(f) for f in (data.get('infos') or [])])
        return out

    out = nx.DiGraph()
    graph_info(out).update(dict(graph_info(graph)))
    keep: set[str] = set()
    for n, data in graph.nodes(data=True):
        info = dict(data.get('info') or {})
        if not include_external and info.get('external'):
            continue
        if not include_tests and info.get('is_test'):
            continue
        keep.add(n)
        out.add_node(n, info=info)

    for src, dst, data in graph.edges(data=True):
        if src not in keep or dst not in keep:
            continue
        infos = []
        for fact in data.get('infos') or []:
            if not include_external and fact.get('dst_external'):
                continue
            infos.append(dict(fact))
        if infos:
            out.add_edge(src, dst, infos=infos)
    return out


def _to_data(graph: nx.DiGraph) -> dict:
    """Serialize ``graph`` to a JSON-friendly dict (round-trippable via ``_from_data``)."""
    nodes = [{'id': n, 'info': dict(data.get('info') or {})} for n, data in graph.nodes(data=True)]
    edges = []
    for src, dst, data in graph.edges(data=True):
        infos = [dict(f) for f in (data.get('infos') or [])]
        if infos:
            edges.append({'src': src, 'dst': dst, 'infos': infos})
    return {
        'info': dict(graph_info(graph)),
        'nodes': nodes,
        'edges': edges,
    }


def _from_data(data: dict) -> nx.DiGraph:
    """Rebuild a ``DiGraph`` from ``_to_data`` output (extra keys are ignored)."""
    graph = nx.DiGraph()
    graph_info(graph).update(dict(data.get('info') or {}))

    for node in data.get('nodes') or []:
        node_id = node.get('id')
        if not node_id:
            continue
        graph.add_node(node_id, info=dict(node.get('info') or {}))

    for edge in data.get('edges') or []:
        src, dst = edge.get('src'), edge.get('dst')
        if not src or not dst:
            continue
        infos = [dict(f) for f in (edge.get('infos') or [])]
        ensure_node(graph, src, {'external': True})
        ensure_node(graph, dst, {'external': True})
        graph.add_edge(src, dst, infos=infos)

    return graph


def dumps_json(graph: nx.DiGraph, *, indent: int = 2) -> str:
    """Serialize *graph* to a UTF-8 JSON string (trailing newline)."""
    return json.dumps(_to_data(graph), indent=indent, ensure_ascii=False) + '\n'


def write_json(graph: nx.DiGraph, path: str | Path, *, indent: int = 2) -> None:
    """Write *graph* as UTF-8 JSON."""
    Path(path).write_text(dumps_json(graph, indent=indent), encoding='utf-8')


def read_json(path: str | Path) -> nx.DiGraph:
    """Load a graph previously written by ``write_json`` / ``dumps_json``."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return _from_data(data)


def qualified_name(graph: nx.DiGraph, symbol: str) -> str:
    """Human-readable dotted name for a node (module + owner path)."""
    info = node_info(graph, symbol) if symbol in graph else {}
    ps = parse_symbol(symbol)
    base = ps.qualified_name or info.get('name') or ps.display_name or symbol
    module = info.get('module') or ''
    if module:
        return f'{module}.{base}'
    if ps.namespace:
        return f'{ps.namespace}.{base}'
    return base


def iter_edge_facts(graph: nx.DiGraph):
    """Yield ``(src, dst, info_dict)`` for each fact in each edge's ``infos``."""
    for src, dst, data in graph.edges(data=True):
        infos: list[dict] = data.get('infos') or []
        for info in infos:
            yield src, dst, info


def _edge_has_type(graph: nx.DiGraph, src: str, dst: str, types: tuple[str, ...]) -> bool:
    return any(f.get('type') in types for f in edge_infos(graph, src, dst))


def callers(graph: nx.DiGraph, node: str, *, types: tuple[str, ...] = ('reference',)) -> set[str]:
    """Nodes that use *node* via an edge carrying one of *types* (default reference).

    ``reference`` edges are call/use sites (``src`` references ``dst``), so
    ``callers(g, f)`` is "who calls/uses f". Pass more types (e.g. adding
    ``implementation``) to widen the relation.
    """
    if node not in graph:
        return set()
    return {u for u in graph.predecessors(node) if _edge_has_type(graph, u, node, types)}


def callees(graph: nx.DiGraph, node: str, *, types: tuple[str, ...] = ('reference',)) -> set[str]:
    """Entities *node* uses via an edge carrying one of *types* (default reference)."""
    if node not in graph:
        return set()
    return {v for v in graph.successors(node) if _edge_has_type(graph, node, v, types)}
