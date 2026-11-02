"""SWE-Any Select — layered-BFS slice extraction (:mod:`.heuristics`) plus the
Claude Code agentic prompt (:mod:`.prompts`) that audits each slice.

Only the heuristic API is re-exported here; import ``build_select_prompt`` from
:mod:`.prompts` directly, mirroring how Parse keeps its own prompt beside its graph
and indexer modules.
"""

from __future__ import annotations

from .heuristics import (
    candidate_nodes,
    component_cap,
    component_quotas,
    node_entry,
    path_excluded,
    sample_seeds,
    seed_components,
    select_subgraph,
    undirected_projection,
)
from .runner import (
    prepare_select_heuristic_context,
    run_select_agentic_slice,
    run_select_feature,
    run_select_heuristic_slice,
)
from .schema import SelectAgenticRow, SelectStatusRow, SelectTaskInput, SelectTaskOutput


__all__ = [
    'SelectAgenticRow',
    'SelectStatusRow',
    'SelectTaskInput',
    'SelectTaskOutput',
    'candidate_nodes',
    'component_cap',
    'component_quotas',
    'node_entry',
    'path_excluded',
    'prepare_select_heuristic_context',
    'run_select_agentic_slice',
    'run_select_feature',
    'run_select_heuristic_slice',
    'sample_seeds',
    'seed_components',
    'select_subgraph',
    'undirected_projection',
]
