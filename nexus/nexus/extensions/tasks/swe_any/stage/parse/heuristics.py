"""Build the entity graph from a SCIP index — Parse's deterministic step.

Reads a ``.scip`` protobuf (``scip.scip_pb2``) and produces the entity/relation
graph defined by the ``graph`` package. Entity = function or class; relations are
``enclosing`` (E01), ``implementation`` (E02), ``reference`` (E03). Together with
``scip`` this is the SCIP/protobuf-dependent side of Parse — consumers of the
graph model (select, mask) depend on ``graph`` instead.

The Python-convention ``is_test`` re-tagging Parse runs after the build lives in the
shared, protobuf-free ``classify_tests`` module (``annotate_tests``), so Eval can
reuse the same test predicates without importing Parse.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from ...utils.graph import (
    METHOD,
    TYPE,
    ParsedSymbol,
    add_edge_info,
    graph_info,
    node_info,
    parse_symbol,
    read_json,
    write_json,
)


ROLE_DEFINITION = 0x1
ROLE_IMPORT = 0x2
ROLE_TEST = 0x20

KIND_CLASS = 'class'
KIND_FUNCTION = 'function'

Range = tuple[int, int, int, int]

try:
    from ...utils.scip import scip_pb2
except ImportError as e:
    raise ImportError(
        'scip_pb2 is missing or broken. It is a committed, pre-generated module; regenerate it with '
        '`python -m grpc_tools.protoc -I scip_proto --python_out=. scip_proto/scip.proto` from the scip package '
        '(requires grpcio-tools + protobuf).'
    ) from e


def norm_range(r) -> Range | None:
    vals = list(r)
    if len(vals) == 3:
        return (vals[0], vals[1], vals[0], vals[2])
    if len(vals) == 4:
        return (vals[0], vals[1], vals[2], vals[3])
    return None


def contains(outer: Range, pos: tuple[int, int]) -> bool:
    sl, sc, el, ec = outer
    line, col = pos
    if line < sl or line > el:
        return False
    if line == sl and col < sc:
        return False
    return not (line == el and col > ec)


def range_size(r: Range) -> tuple[int, int]:
    """Return ``(line_span, col_span)`` so same-line ties prefer the tighter column range."""
    return (r[2] - r[0], r[3] - r[1])


def entity_kind(ps: ParsedSymbol) -> str | None:
    last = ps.last
    if last is None:
        return None
    if last.suffix == TYPE:
        return KIND_CLASS
    if last.suffix == METHOD:
        return KIND_FUNCTION
    return None


def innermost(graph: nx.DiGraph, funcs: list[str], pos: tuple[int, int]) -> str | None:
    best: str | None = None
    best_size = None
    for symbol in funcs:
        encl = node_info(graph, symbol).get('enclosing_range')
        if not encl or len(encl) != 4:
            continue
        r = (encl[0], encl[1], encl[2], encl[3])
        if contains(r, pos):
            size = range_size(r)
            if best_size is None or size < best_size:
                best, best_size = symbol, size
    return best


def load_index(path: str | Path) -> scip_pb2.Index:  # ty: ignore[unresolved-attribute]
    """Read a ``.scip`` file into an ``Index`` protobuf message."""
    index = scip_pb2.Index()  # ty: ignore[unresolved-attribute]
    with open(path, 'rb') as f:
        index.ParseFromString(f.read())
    return index


def build_graph(
    index: scip_pb2.Index | str | Path,  # ty: ignore[unresolved-attribute]
    *,
    graph_path: str | Path | None = None,
) -> nx.DiGraph:
    """Build a NetworkX ``DiGraph`` from a SCIP index or a path to ``parse_index.scip``.

    If ``graph_path`` already exists, load and return it. Otherwise build from
    the index and, when ``graph_path`` is set, write the complete graph there.
    Filtering external or test entities belongs to downstream stages.
    """
    if graph_path is not None and Path(graph_path).is_file():
        return read_json(graph_path)

    if not isinstance(index, scip_pb2.Index):  # ty: ignore[unresolved-attribute]
        index = load_index(index)

    graph = _build_graph(index)
    if graph_path is not None:
        write_json(graph, graph_path)
    return graph


def _build_graph(index: scip_pb2.Index) -> nx.DiGraph:  # ty: ignore[unresolved-attribute]
    graph = nx.DiGraph()
    graph_info(graph).update({
        'project_root': index.metadata.project_root,
        'tool': (f'{index.metadata.tool_info.name} {index.metadata.tool_info.version}'),
    })

    # Pass 1: definition sites + entity nodes.
    def_info: dict[str, tuple[str, int, Range | None, int]] = {}
    for doc in index.documents:
        for occ in doc.occurrences:
            if not (occ.symbol_roles & ROLE_DEFINITION):
                continue
            rng = norm_range(occ.range)
            if rng is None:
                continue
            encl = norm_range(occ.enclosing_range) if occ.enclosing_range else None
            def_info[occ.symbol] = (doc.relative_path, rng[0], encl, occ.symbol_roles)

    for doc in index.documents:
        for si in doc.symbols:
            ps = parse_symbol(si.symbol)
            kind = entity_kind(ps)
            if kind is None:
                continue
            file, def_line, encl, roles = def_info.get(si.symbol, (doc.relative_path, None, None, 0))
            graph.add_node(
                si.symbol,
                info={
                    'kind': kind,
                    'name': ps.display_name,
                    'module': ps.namespace,
                    'file': file,
                    'owner': ps.owner_symbol(),
                    'def_line': def_line,
                    'enclosing_range': list(encl) if encl is not None else None,
                    'is_test': bool(roles & ROLE_TEST),
                    'documentation': '\n'.join(si.documentation).strip(),
                    'external': False,
                },
            )

    def add_rel(
        src: str,
        dst: str,
        rel_type: str,
        *,
        external: bool | None = None,
        evidence: str = '',
    ) -> None:
        if external is None:
            external = True if dst not in graph else bool(node_info(graph, dst).get('external', False))
        add_edge_info(
            graph,
            src,
            dst,
            {
                'type': rel_type,
                'dst_external': bool(external),
                'evidence': evidence,
            },
        )

    # Pass 2: enclosing - SCIP symbol descriptor hierarchy.
    for doc in index.documents:
        for si in doc.symbols:
            if si.symbol not in graph:
                continue
            owner = parse_symbol(si.symbol).owner_symbol()
            if not owner or owner not in graph:
                continue
            add_rel(owner, si.symbol, 'enclosing', external=False, evidence='descriptor')

    # Pass 3: implementation - relationships.is_implementation.
    for doc in index.documents:
        for si in doc.symbols:
            if si.symbol not in graph:
                continue
            for rel in si.relationships:
                if not rel.is_implementation:
                    continue
                if rel.symbol not in graph and entity_kind(parse_symbol(rel.symbol)) is None:
                    continue
                add_rel(
                    si.symbol,
                    rel.symbol,
                    'implementation',
                    evidence='is_implementation',
                )

    # Pass 4: reference - occurrence inside a function enclosing_range.
    enclosers: dict[str, list[str]] = {}
    for doc in index.documents:
        for si in doc.symbols:
            if si.symbol not in graph:
                continue
            info = node_info(graph, si.symbol)
            if info.get('kind') != KIND_FUNCTION:
                continue
            encl = info.get('enclosing_range')
            if not encl:
                continue
            enclosers.setdefault(info.get('file') or '', []).append(si.symbol)

    for doc in index.documents:
        funcs = enclosers.get(doc.relative_path, [])
        if not funcs:
            continue
        for occ in doc.occurrences:
            if occ.symbol_roles & (ROLE_DEFINITION | ROLE_IMPORT):
                continue
            if occ.symbol in graph and not node_info(graph, occ.symbol).get('external'):
                target_kind = node_info(graph, occ.symbol).get('kind')
            else:
                target_kind = entity_kind(parse_symbol(occ.symbol))
            if target_kind is None:
                continue
            rng = norm_range(occ.range)
            if rng is None:
                continue
            enclosing = innermost(graph, funcs, (rng[0], rng[1]))
            if enclosing is None or enclosing == occ.symbol:
                continue
            add_rel(
                enclosing,
                occ.symbol,
                'reference',
                evidence='occurrence',
            )

    return graph
