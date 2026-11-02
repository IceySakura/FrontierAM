"""Schemas and host-side output shaping for the SWE-Any Select stage.

Normalizes the agent's raw JSON into the archived agentic row (``agentic_nodes`` etc.)
and joins the per-slice heuristic / agentic / status pieces into the flat unified
``select.jsonl`` row (which carries the ``completed`` / ``runtime_error`` status too).
"""

from __future__ import annotations

import json
import logging

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, WithScore
from nexus.extensions.tasks.swe_any.utils import constants


logger = logging.getLogger(__name__)


class SelectTaskInput(TaskRunInput, WithAgentRunner):
    """Business inputs for one repo's Select run (N slices, internal workers)."""

    instance: dict[str, Any] | None = None
    instance_path: str = ''
    """Optional one-row JSONL; ignored when ``instance`` is provided."""

    # Input graph (host path produced by Parse's download).
    repo_name: str = 'repo'
    graph_path: str = ''
    """Host path to parse_graph.json (from ``swe_any_parse``)."""
    seed_policy_path: str = ''
    """Host path to Parse's parse_seed_policy.json. Empty = ``output_dir/<repo>/parse_seed_policy.json``; missing = no exclusions."""
    output_dir: str = './outputs/swe_any'

    # Sandbox-side layout: repo source for Claude to navigate.
    repo_dir: str = constants.SANDBOX_REPO_DIR
    """Sandbox path to the repo source in the baked per-repo image (SWE-Env ``workdir``)."""
    base_commit: str = ''
    """Git commit restored before this stage; hydrated from the source SWE-Env instance."""
    swe_any_dir: str = constants.SANDBOX_SWE_ANY_DIR
    heuristic_filename: str = constants.SELECT_HEURISTIC_FILENAME
    agentic_filename: str = constants.SELECT_AGENTIC_FILENAME
    heuristic_jsonl_filename: str = constants.SELECT_HEURISTIC_JSONL_FILENAME
    agentic_jsonl_filename: str = constants.SELECT_AGENTIC_JSONL_FILENAME
    unified_jsonl_filename: str = constants.SELECT_JSONL_FILENAME
    runtime_name: str = 'main'

    # Baked per-repo image (the workload image). Empty = rely on the provider default.
    image: str = ''
    image_override: str | None = None

    # Repo-level slice budget (not a fixed count): apportioned across components,
    # each capped at ceil(size / max_bfs_nodes).
    max_subgraphs: int = Field(default=constants.DEFAULT_MAX_SUBGRAPHS, ge=1)
    max_workers: int = Field(default=constants.DEFAULT_MAX_WORKERS, ge=1)
    # Seeds are always sampled; there is no explicit-seed input. Reproduce a run
    # with rng_seed, and shape the pool with the knobs below.
    rng_seed: int | None = None

    # Heuristic knobs — layered BFS.
    max_bfs_nodes: int = Field(default=constants.DEFAULT_MAX_BFS_NODES, ge=1)
    drop_bfs_prob: float = Field(default=constants.DEFAULT_DROP_BFS_PROB, ge=0.0, le=1.0)
    min_bfs_per_depth: int = Field(default=constants.DEFAULT_MIN_BFS_PER_DEPTH, ge=1)
    max_bfs_per_depth: int = Field(default=constants.DEFAULT_MAX_BFS_PER_DEPTH, ge=1)
    include_tests: bool = True
    include_external: bool = False

    # Seed pool knobs. The component window is measured after exclude_globs.
    min_component_nodes: int = Field(default=constants.DEFAULT_MIN_COMPONENT_NODES, ge=1)
    max_component_nodes: int = Field(default=constants.DEFAULT_MAX_COMPONENT_NODES, ge=1)
    exclude_globs: list[str] = Field(default_factory=list)
    """Repo-relative globs barred from seeding (merged with Parse's seed policy)."""

    # Size window for the agent's `feature_node_ids` list, stated in the prompt and enforced
    # on the returned agentic output. The lower bound is off by default.
    min_final_nodes: int = Field(default=constants.DEFAULT_MIN_FINAL_NODES, ge=1)
    max_final_nodes: int = Field(default=constants.DEFAULT_MAX_FINAL_NODES, ge=1)

    # Agentic-pass knobs (Claude Code). The timeout tracks CLAUDE_CODE_TIMEOUT, which is
    # what the YAML and the batch generator pass in.
    agent_timeout: float = Field(default=3600.0, gt=0)
    # Skip early release so another stage in the same ExecutionContext can reuse it.
    # ExecutionContext.__aexit__ still releases all runtimes on success or failure.
    release_runtime: bool = False

    @model_validator(mode='after')
    def validate_limits(self) -> SelectTaskInput:
        if self.min_bfs_per_depth > self.max_bfs_per_depth:
            raise ValueError(
                f'min_bfs_per_depth ({self.min_bfs_per_depth}) > max_bfs_per_depth ({self.max_bfs_per_depth})'
            )
        if self.min_component_nodes > self.max_component_nodes:
            raise ValueError(
                f'min_component_nodes ({self.min_component_nodes}) > max_component_nodes ({self.max_component_nodes})'
            )
        if self.min_final_nodes > self.max_final_nodes:
            raise ValueError(f'min_final_nodes ({self.min_final_nodes}) > max_final_nodes ({self.max_final_nodes})')
        return self


class SelectTaskOutput(TaskRunOutput, WithScore):
    """Result of one repo's Select run (aggregated over N slices)."""

    repo_name: str
    num_subgraphs: int = 0
    """Slices actually run — derived from component sizes, not requested."""
    num_completed: int = 0
    num_agentic: int = 0
    """Slices the agent resolved (``success`` / ``source == 'agentic'``) — the agent-success count."""
    num_abandoned: int = 0
    """Slices that ran but resolved no feature (``completed`` yet not ``success`` — they carry an
    ``error``: abandon or schema)."""
    cover_rate_mean: float = 0.0
    """Mean per-slice ``cover_rate`` over agentic slices (0 when none) — the average fraction of
    an agentic feature's ``agentic_nodes`` that came from its heuristic slice. See
    ``SelectAgenticRow.cover_rate``."""
    seeds: list[str] = Field(default_factory=list)
    # Host JSONL archives: per-half aggregates + the unified deliverable (carries status too).
    heuristic_path: str | None = None
    agentic_path: str | None = None
    unified_path: str | None = None  # select.jsonl — the canonical deliverable (Mask reads this)
    runtime_error: str | None = None


class SelectAgenticRow(BaseModel):
    """One Select agentic row — the agent's final per-slice output, nothing else.

    Written verbatim (no filtering) to ``sgNN/select_agentic.json`` / ``select_agentic.jsonl``.
    Execution status (``completed`` / ``runtime_error``) is deliberately kept out, in
    :class:`SelectStatusRow`, so this stays pure content.

    ``feature`` (the description) / ``agentic_nodes`` (the node objects) carry payload only
    when the agent resolved a coherent feature (``success`` / ``source == 'agentic'``); every
    other outcome archives an empty pair plus an ``error``. The agent itself emits ids; the host
    enriches them into node objects (same shape as the heuristic ``heuristic_nodes``) via
    :func:`stage.select.node_entry`.
    """

    index: int
    feature: str = ''
    agentic_nodes: list[dict] = Field(default_factory=list)
    """The resolved feature's function nodes (``node_entry`` objects, each carrying ``id``);
    the payload Mask consumes. Empty unless ``success`` (``source == 'agentic'``)."""
    cover_rate: float = 0.0
    """Fraction of the agentic feature's ``agentic_nodes`` that came from the heuristic
    slice (``final_in_slice / len(agentic_nodes)``; 0.0 when the list is empty).

    Read it as slice/feature *overlap*, not as drift: a sparse slice (BFS touches a
    couple of functions per file) scores low even when every id lives in a file the slice
    already covered.
    """
    source: str = ''
    """Provenance (shared with Mask / Eval): ``'agentic'`` when the agent resolved a coherent
    feature, else ``''``. Select has no heuristic fallback, so it is never ``'heuristic'`` — an
    unresolved slice simply yields no feature. Kept in sync with ``success`` (``success ==
    (source != '')``)."""
    success: bool = False
    """Deliverable outcome = ``source != ''`` (the agent resolved a coherent feature). Decoupled
    from execution status: an agent crash is a soft ``warning`` (``completed`` stays True), so
    ``success`` is driven purely by whether a feature was produced. The business reason for
    ``success=False`` on a completed run (abandoned / schema) is ``error``."""
    error: str | None = None
    """Business reason no feature was resolved: ``abandoned: ...`` or ``schema: ...`` (the agent ran
    and declined / failed validation). ``None`` when the agent instead crashed (that is a
    ``warning``, on :class:`SelectStatusRow`) or on success. Never overlaps ``runtime_error``."""


class SelectStatusRow(BaseModel):
    """Per-slice execution status, kept out of :class:`SelectAgenticRow` so the agentic
    files stay pure output. Merged into the unified ``select.jsonl`` row + the task-level counts.

    ``completed`` = the slice ran without an UNEXPECTED interruption (an infra/machinery crash,
    e.g. the heuristic subgraph build); ``runtime_error`` holds that interruption. A soft failure —
    notably the agent invoke/parse/download failing — is NOT an interruption: it lands in
    ``warnings`` (``completed`` stays True) and only surfaces as ``success=False`` (no feature),
    never as a ``runtime_error``.
    """

    index: int
    completed: bool = False
    runtime_error: str | None = None
    warnings: list[str] = Field(default_factory=list)


class SelectUnifiedRow(BaseModel):
    """Schema for one ``select.jsonl`` row — the lean per-slice deliverable Mask reads.

    Single source of truth for the unified field set: the agent's result (``source`` / ``nodes``
    / ``feature`` / ``success`` / ``error``) plus diagnostics (``warnings``) and the per-slice
    status (``completed`` / ``runtime_error``). Heuristic detail (``seed`` / ``meta`` /
    ``heuristic_nodes``) and ``cover_rate`` stay in the per-half ``select_{heuristic,agentic}.jsonl``
    files, recoverable by joining on ``index``. Built by :func:`select_unified_rows`.
    """

    index: int
    source: str = ''
    nodes: list[dict] = Field(default_factory=list)
    """The resolved feature's function nodes (the agentic pass's ``agentic_nodes``); Mask takes
    their ids. Empty unless ``success``."""
    feature: str = ''
    success: bool = False
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    completed: bool = False
    runtime_error: str | None = None


def load_seed_policy(path: Path) -> list[str]:
    """Read ``exclude_globs`` from a Parse seed policy; missing/corrupt → no exclusions."""
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:  # noqa: BLE001 - a hand-edited or truncated policy must not fail Select
        logger.warning('swe_any_select ignoring unreadable seed policy %s: %s', path, exc)
        return []
    globs = payload.get('exclude_globs') if isinstance(payload, dict) else None
    if not isinstance(globs, list):
        return []
    return [g for g in globs if isinstance(g, str) and g]


def normalize_select_agentic_row(
    raw: dict[str, Any],
    *,
    graph_function_ids: set[str],
    slice_ids: set[str],
    node_builder: Callable[[str], dict[str, Any]],
    min_final_nodes: int = constants.DEFAULT_MIN_FINAL_NODES,
    max_final_nodes: int = constants.DEFAULT_MAX_FINAL_NODES,
) -> dict[str, Any]:
    """Normalize the agentic-pass fields; anything short of a resolved feature yields an empty row.

    The prompt's two-shape contract is enforced: a resolved feature keeps ``feature`` and
    ``agentic_nodes``, sets ``source = 'agentic'`` and ``success = True``; everything else
    archives an empty pair plus ``success = False`` and an ``error`` (leaving ``source = ''``).
    The agent's output is accepted only when every agent-emitted id identifies a function in
    the working (filtered) graph and the list size sits inside the window the prompt asked for.
    The surviving ids are enriched into node objects via ``node_builder`` (same shape as the
    heuristic ``heuristic_nodes``); ``cover_rate`` is computed on the ids.
    """
    unresolved = {'feature': '', 'source': '', 'agentic_nodes': [], 'cover_rate': 0.0}

    if raw.get('abandoned') is True:
        reason = str(raw.get('abandon_reason') or '').strip() or 'no reason given'
        return {**unresolved, 'success': False, 'error': f'abandoned: {reason}'}

    model_resolved = raw.get('resolved')
    if model_resolved is not True:
        # The prompt offers exactly two shapes; a bare false means the model
        # gave up without taking the abandon exit. Record it as an abandon so
        # the row still explains itself.
        return {
            **unresolved,
            'success': False,
            'error': 'abandoned: model reported resolved=false without taking the abandon exit',
        }

    reasons: list[str] = []
    ids_raw = raw.get('feature_node_ids')
    if not isinstance(ids_raw, list):
        reasons.append('feature_node_ids is not a list')
        ids_raw = []

    feature_node_ids: list[str] = []
    for entity_id in ids_raw:
        if not isinstance(entity_id, str) or entity_id not in graph_function_ids:
            reasons.append(f'invalid feature_node_ids id: {entity_id!r}')
            continue
        if entity_id not in feature_node_ids:
            feature_node_ids.append(entity_id)
    # Counted after dedupe and id validation, so the row is judged on what survived.
    if not min_final_nodes <= len(feature_node_ids) <= max_final_nodes:
        reasons.append(f'feature_node_ids size {len(feature_node_ids)} outside {min_final_nodes}..{max_final_nodes}')

    if reasons:
        logger.warning('swe_any_select agentic schema check failed: %s', '; '.join(reasons))
        return {**unresolved, 'success': False, 'error': f'schema: {"; ".join(reasons)}'}

    final_in_slice = sum(1 for entity_id in feature_node_ids if entity_id in slice_ids)
    return {
        'feature': str(raw.get('feature') or ''),
        'source': 'agentic',
        'agentic_nodes': [node_builder(entity_id) for entity_id in feature_node_ids],
        'cover_rate': (final_in_slice / len(feature_node_ids)) if feature_node_ids else 0.0,
        'success': True,
        'error': None,
    }


def select_unified_rows(heuristic_rows: list[dict], agentic_rows: list[dict], status_rows: list[dict]) -> list[dict]:
    """Project each slice into a :class:`SelectUnifiedRow` (dumped to dict), joined on ``index``.

    The three per-slice pieces (heuristic / agentic / status) are looked up by ``index`` and
    mapped onto the explicit schema — the field set is owned by :class:`SelectUnifiedRow`, not by
    a ``**``-merge, so nothing leaks in from the halves by accident.
    """
    by_agentic = {a.get('index'): a for a in agentic_rows}
    by_status = {s.get('index'): s for s in status_rows}
    rows: list[dict] = []
    for heuristic in heuristic_rows:
        index = heuristic['index']  # every heuristic row carries an int index (join key)
        agentic = by_agentic.get(index, {})
        status = by_status.get(index, {})
        rows.append(
            SelectUnifiedRow(
                index=index,
                source=agentic.get('source', ''),
                nodes=agentic.get('agentic_nodes', []),
                feature=agentic.get('feature', ''),
                success=agentic.get('success', False),
                error=agentic.get('error'),
                warnings=status.get('warnings', []),
                completed=status.get('completed', False),
                runtime_error=status.get('runtime_error'),
            ).model_dump()
        )
    return rows


__all__ = [
    'SelectAgenticRow',
    'SelectStatusRow',
    'SelectTaskInput',
    'SelectTaskOutput',
    'SelectUnifiedRow',
    'load_seed_policy',
    'normalize_select_agentic_row',
    'select_unified_rows',
]
