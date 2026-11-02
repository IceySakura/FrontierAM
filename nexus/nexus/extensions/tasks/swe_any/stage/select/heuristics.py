"""Select heuristic — layered BFS subgraph extraction, no LLM.

Given a full entity graph (``parse_graph.json`` from Parse), grow a *suspected
feature* node set from an explicit seed via depth-layered BFS with per-depth
dropout / quotas. The host writes per-slice ``sgNN/select_heuristic.json`` then a
parent ``select_heuristic.jsonl``; a downstream Claude Code agentic pass checks feature
consistency.

Algorithm (per slice):
1. Filter candidates with ``include_tests`` / ``include_external``.
2. Seed is always selected and marked used (depth 0).
3. For each deeper layer: collect unused neighbours of the previous frontier →
   mark them used → shuffle → Bernoulli ``drop_bfs_prob`` → raise to
   ``min_bfs_per_depth`` (take-as-many-as-available if short) → cap at
   ``max_bfs_per_depth`` → apply the global ``max_bfs_nodes`` cap.
4. Stop when a layer keeps nothing or ``max_bfs_nodes`` is hit.

Seed sampling works per connected component of a restricted pool: nodes excluded
by ``exclude_globs`` are dropped, and a component must be sized within
``[min_component_nodes, max_component_nodes]`` to seed anything. That keeps
peripheral islands (a util plus its only test, say) from producing degenerate
slices. Excluded nodes stay in the graph — generated code is a poor place to
start a slice but a legitimate part of the feature a slice grows into.

How many slices a repo is worth is derived, not fixed: each component is capped at
``ceil(size / max_bfs_nodes)`` slices, and ``max_subgraphs`` is a repo-level budget
those caps are apportioned against. So a 1200-node repo gets 20 slices and a 230-node
repo gets 4.
"""

from __future__ import annotations

import random

from collections.abc import Sequence
from fnmatch import fnmatch

import networkx as nx

from ...utils.constants import (
    DEFAULT_DROP_BFS_PROB,
    DEFAULT_MAX_BFS_NODES,
    DEFAULT_MAX_BFS_PER_DEPTH,
    DEFAULT_MAX_COMPONENT_NODES,
    DEFAULT_MAX_SUBGRAPHS,
    DEFAULT_MIN_BFS_PER_DEPTH,
    DEFAULT_MIN_COMPONENT_NODES,
)
from ...utils.graph import node_info, qualified_name


def _file_of(graph: nx.DiGraph, n: str) -> str:
    return node_info(graph, n).get('file') or ''


def path_excluded(file: str, exclude_globs: Sequence[str]) -> bool:
    """True when the repo-relative *file* matches any glob.

    ``fnmatch`` has no path semantics — ``*`` matches ``/`` as well — so both
    ``pkg/gen/*`` and ``pkg/gen/**`` cover the whole subtree.
    """
    return bool(file) and any(fnmatch(file, pattern) for pattern in exclude_globs)


def candidate_nodes(
    graph: nx.DiGraph,
    *,
    include_tests: bool = True,
    include_external: bool = False,
) -> set[str]:
    """Return nodes eligible for a subgraph."""
    keep: set[str] = set()
    for n in graph.nodes:
        info = node_info(graph, n)
        if not include_external and info.get('external'):
            continue
        if not include_tests and info.get('is_test'):
            continue
        keep.add(n)
    return keep


def undirected_projection(graph: nx.DiGraph, allowed: set[str]) -> nx.Graph:
    """Undirected graph over *allowed* nodes with edges kept only among them."""
    undirected = nx.Graph()
    undirected.add_nodes_from(allowed)
    for u, v in graph.edges():
        if u in allowed and v in allowed and u != v:
            undirected.add_edge(u, v)
    return undirected


def seed_components(
    graph: nx.DiGraph,
    allowed: set[str],
    *,
    exclude_globs: Sequence[str] = (),
    min_component_nodes: int = DEFAULT_MIN_COMPONENT_NODES,
    max_component_nodes: int = DEFAULT_MAX_COMPONENT_NODES,
) -> list[set[str]]:
    """Return the connected components eligible to seed slices, largest first.

    Components are computed *after* dropping ``exclude_globs`` nodes: excluding
    generated code splits the graph differently, and the size window has to see
    that split. Only components sized within the window are returned.
    """
    pool = {n for n in allowed if not path_excluded(_file_of(graph, n), exclude_globs)}
    undirected = undirected_projection(graph, pool)
    eligible: list[set[str]] = [
        set(c) for c in nx.connected_components(undirected) if min_component_nodes <= len(c) <= max_component_nodes
    ]
    eligible.sort(key=len, reverse=True)
    return eligible


def component_cap(size: int, *, max_bfs_nodes: int) -> int:
    """Most slices a component of *size* nodes is worth.

    ``ceil(size / max_bfs_nodes)`` is the number of disjoint full-size slices the
    component could hold, so the slice budget scales with the repo instead of every
    repo getting the same fixed count.
    """
    return max(1, -(-size // max(max_bfs_nodes, 1)))


def component_quotas(sizes: Sequence[int], *, max_bfs_nodes: int, max_subgraphs: int) -> list[int]:
    """Split a repo-level budget of ``max_subgraphs`` slices across components.

    Each component's cap is :func:`component_cap`. When the caps fit in the budget
    every component gets its cap; otherwise the budget is apportioned in proportion
    to the caps (largest-remainder, so the total lands exactly on ``max_subgraphs``
    and the biggest components absorb the rounding). Components can end up with 0
    when there are more of them than slices to hand out.
    """
    caps = [component_cap(size, max_bfs_nodes=max_bfs_nodes) for size in sizes]
    total = sum(caps)
    if total <= max_subgraphs:
        return caps
    exact = [max_subgraphs * cap / total for cap in caps]
    quotas = [int(share) for share in exact]
    leftover = max_subgraphs - sum(quotas)
    ranked = sorted(range(len(caps)), key=lambda i: (quotas[i] - exact[i], -caps[i], i))
    for i in ranked[:leftover]:
        quotas[i] += 1
    return quotas


def sample_seeds(
    graph: nx.DiGraph,
    *,
    include_tests: bool = True,
    include_external: bool = False,
    rng: random.Random | None = None,
    allowed: set[str] | None = None,
    exclude_globs: Sequence[str] = (),
    min_component_nodes: int = DEFAULT_MIN_COMPONENT_NODES,
    max_component_nodes: int = DEFAULT_MAX_COMPONENT_NODES,
    components: list[set[str]] | None = None,
    max_bfs_nodes: int = DEFAULT_MAX_BFS_NODES,
    max_subgraphs: int = DEFAULT_MAX_SUBGRAPHS,
) -> list[str]:
    """Draw one seed per slice, splitting ``max_subgraphs`` across the components.

    Quotas come from :func:`component_quotas`. Within a component every node is
    equally likely — deliberately not weighted by file or directory, so a package
    contributes seeds in proportion to its entity count. Draws inside a component are
    without replacement, since two identical seeds would only buy two identical
    agentic calls. Callers running many slices over one graph can pass precomputed
    ``components`` from :func:`seed_components`.
    """
    rng = rng or random.Random()
    if components is None:
        if allowed is None:
            allowed = candidate_nodes(graph, include_tests=include_tests, include_external=include_external)
        components = seed_components(
            graph,
            allowed,
            exclude_globs=exclude_globs,
            min_component_nodes=min_component_nodes,
            max_component_nodes=max_component_nodes,
        )
    if not components:
        raise ValueError(
            f'no connected component sized within [{min_component_nodes}, {max_component_nodes}] '
            f'survives exclude_globs={list(exclude_globs)}; widen the window or drop exclusions'
        )
    quotas = component_quotas([len(c) for c in components], max_bfs_nodes=max_bfs_nodes, max_subgraphs=max_subgraphs)
    seeds: list[str] = []
    for component, quota in zip(components, quotas, strict=True):
        if quota > 0:
            seeds.extend(rng.sample(sorted(component), min(quota, len(component))))
    return seeds


def node_entry(graph: nx.DiGraph, n: str) -> dict:
    """One archived node object (id + human-readable metadata) for *n*.

    Shared by the heuristic slice (``nodes``) and the agentic feature (``feature_nodes``)
    so both sides carry the identical node shape.
    """
    info = node_info(graph, n)
    return {
        'id': n,
        'name': qualified_name(graph, n),
        'kind': info.get('kind'),
        'module': info.get('module'),
        'file': info.get('file') or '',
        'def_line': info.get('def_line'),
        'is_test': bool(info.get('is_test')),
        'documentation': info.get('documentation') or '',
    }


def _layered_bfs(
    undirected: nx.Graph,
    seed: str,
    allowed: set[str],
    *,
    max_bfs_nodes: int,
    drop_bfs_prob: float,
    min_bfs_per_depth: int,
    max_bfs_per_depth: int,
    rng: random.Random,
) -> list[str]:
    """Grow from *seed*; return selected nodes in discovery order."""
    if seed not in allowed:
        raise ValueError(f'seed not in candidate set: {seed!r}')
    if not 0.0 <= drop_bfs_prob <= 1.0:
        raise ValueError(f'drop_bfs_prob must be in [0, 1], got {drop_bfs_prob}')
    if min_bfs_per_depth < 1 or max_bfs_per_depth < 1:
        raise ValueError('min_bfs_per_depth and max_bfs_per_depth must be >= 1')
    if min_bfs_per_depth > max_bfs_per_depth:
        raise ValueError(f'min_bfs_per_depth ({min_bfs_per_depth}) > max_bfs_per_depth ({max_bfs_per_depth})')

    selected: list[str] = [seed]
    used: set[str] = {seed}
    frontier: list[str] = [seed]

    while frontier and len(selected) < max_bfs_nodes:
        # 1) Collect next-layer candidates (not yet used).
        layer: list[str] = []
        seen_layer: set[str] = set()
        for cur in frontier:
            for nb in undirected.neighbors(cur):
                if nb not in allowed or nb in used or nb in seen_layer:
                    continue
                seen_layer.add(nb)
                layer.append(nb)

        if not layer:
            break

        # Design choice: every scanned node becomes permanently used, including
        # nodes later removed by dropout or caps. This favors a small, diverse
        # slice over filling max_bfs_nodes by revisiting overflow from hub layers.
        used.update(layer)

        # 2) Shuffle, then Bernoulli dropout.
        rng.shuffle(layer)
        kept: list[str] = [n for n in layer if rng.random() >= drop_bfs_prob]
        kept_set = set(kept)
        dropped = [n for n in layer if n not in kept_set]

        # 3) Raise to min_bfs_per_depth (take as many as available).
        if len(kept) < min_bfs_per_depth and dropped:
            need = min(min_bfs_per_depth - len(kept), len(dropped))
            kept.extend(dropped[:need])

        # 4) Cap at max_bfs_per_depth.
        if len(kept) > max_bfs_per_depth:
            kept = kept[:max_bfs_per_depth]

        # 5) Global node cap.
        room = max_bfs_nodes - len(selected)
        if len(kept) > room:
            kept = kept[:room]

        if not kept:
            break

        selected.extend(kept)
        frontier = kept

    return selected


def select_subgraph(
    graph: nx.DiGraph,
    *,
    seed: str,
    max_bfs_nodes: int = DEFAULT_MAX_BFS_NODES,
    drop_bfs_prob: float = DEFAULT_DROP_BFS_PROB,
    min_bfs_per_depth: int = DEFAULT_MIN_BFS_PER_DEPTH,
    max_bfs_per_depth: int = DEFAULT_MAX_BFS_PER_DEPTH,
    include_tests: bool = True,
    include_external: bool = False,
    rng: random.Random | None = None,
    rng_seed: int | None = None,
    allowed: set[str] | None = None,
    undirected: nx.Graph | None = None,
    component_size: int = 0,
) -> dict:
    """Cut a suspected-feature node set from *graph* starting at *seed*.

    *seed* is required. Callers that run many slices over the same graph may
    pass precomputed ``allowed`` / ``undirected`` to avoid O(V+E) rebuilds, plus
    the seed's ``component_size`` for post-hoc stratified analysis.
    Returns a JSON-friendly slice dict::

        {
          "seed": <str>,
          "meta": {...},                    # knobs + counts; no free-text reason
          "heuristic_nodes": [{id, name, kind, ...}, ...],
        }
    """
    if not seed:
        raise ValueError('seed is required')

    if rng is None:
        rng = random.Random(rng_seed)

    if allowed is None:
        allowed = candidate_nodes(graph, include_tests=include_tests, include_external=include_external)
    if seed not in allowed:
        raise ValueError(
            f'seed {seed!r} is not a candidate under include_tests={include_tests}, include_external={include_external}'
        )

    if undirected is None:
        undirected = undirected_projection(graph, allowed)
    selected = _layered_bfs(
        undirected,
        seed,
        allowed,
        max_bfs_nodes=max_bfs_nodes,
        drop_bfs_prob=drop_bfs_prob,
        min_bfs_per_depth=min_bfs_per_depth,
        max_bfs_per_depth=max_bfs_per_depth,
        rng=rng,
    )

    nodes = [node_entry(graph, n) for n in selected]
    files = sorted({n['file'] for n in nodes if n['file']})
    return {
        'seed': seed,
        'meta': {
            'max_bfs_nodes': max_bfs_nodes,
            'drop_bfs_prob': drop_bfs_prob,
            'min_bfs_per_depth': min_bfs_per_depth,
            'max_bfs_per_depth': max_bfs_per_depth,
            'include_tests': include_tests,
            'include_external': include_external,
            'rng_seed': rng_seed,
            'component_size': component_size,
            'node_count': len(nodes),
            'file_count': len(files),
        },
        'heuristic_nodes': nodes,
    }
