"""Lightweight entity-graph library — protobuf-free model, symbol parsing, JSON IO.

Shared by ``parse`` (which builds the graph from a SCIP index) and ``select``
(and later Mask/Eval, which slice / consume it). Depends only on ``networkx``
and the stdlib, so importing it never pulls the SCIP/protobuf toolchain — that
lives in the ``scip`` package.

    from nexus.extensions.tasks.swe_any.utils.graph import node_info, read_json

:mod:`.model` holds the graph schema accessors + JSON IO; :mod:`.symbol` parses
SCIP symbol strings (graph node ids). Both are re-exported here.
"""

from __future__ import annotations

from .model import (
    add_edge_info,
    callees,
    callers,
    dumps_json,
    edge_infos,
    ensure_node,
    filter_graph,
    graph_info,
    iter_edge_facts,
    node_info,
    qualified_name,
    read_json,
    write_json,
)
from .symbol import (
    MACRO,
    META,
    METHOD,
    NAMESPACE,
    PARAMETER,
    TERM,
    TYPE,
    TYPE_PARAMETER,
    Descriptor,
    ParsedSymbol,
    parse_symbol,
)


__all__ = [
    'MACRO',
    'META',
    'METHOD',
    'NAMESPACE',
    'PARAMETER',
    'TERM',
    'TYPE',
    'TYPE_PARAMETER',
    'Descriptor',
    'ParsedSymbol',
    'add_edge_info',
    'callees',
    'callers',
    'dumps_json',
    'edge_infos',
    'ensure_node',
    'filter_graph',
    'graph_info',
    'iter_edge_facts',
    'node_info',
    'parse_symbol',
    'qualified_name',
    'read_json',
    'write_json',
]
