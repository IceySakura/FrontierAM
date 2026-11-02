"""Schemas and host-side output shaping for the SWE-Any Mask stage.

Reads the resolved Select features Mask consumes and builds the per-feature unified rows
(status inlined; plus the empty-row variants used on initial state / hard failure).
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, WithScore
from nexus.extensions.tasks.swe_any.utils import constants


class MaskTaskInput(TaskRunInput, WithAgentRunner):
    """Business inputs for one repo's Mask run (draft mask per feature + agent refine)."""

    instance: dict[str, Any] | None = None
    instance_path: str = ''
    """Optional one-row JSONL; ignored when ``instance`` is provided."""

    repo_name: str = 'repo'
    graph_path: str = ''
    """Host path to parse_graph.json (from ``swe_any_parse``); uploaded to the sandbox for ``run_mask``."""
    select_path: str = ''
    """Host path to Select's unified select.jsonl (its resolved features). Empty = ``output_dir/<repo>/select.jsonl``."""
    output_dir: str = './outputs/swe_any'

    # Sandbox-side layout.
    repo_dir: str = constants.SANDBOX_REPO_DIR
    base_commit: str = ''
    """Git commit restored before this stage; hydrated from the source SWE-Env instance."""
    swe_any_dir: str = constants.SANDBOX_SWE_ANY_DIR
    graph_filename: str = constants.GRAPH_FILENAME
    heuristic_filename: str = constants.MASK_HEURISTIC_FILENAME
    agentic_filename: str = constants.MASK_AGENTIC_FILENAME
    heuristic_jsonl_filename: str = constants.MASK_HEURISTIC_JSONL_FILENAME
    agentic_jsonl_filename: str = constants.MASK_AGENTIC_JSONL_FILENAME
    unified_jsonl_filename: str = constants.MASK_JSONL_FILENAME
    runtime_name: str = 'main'

    # Baked per-repo image (the workload image). Empty = rely on the provider default.
    image: str = ''
    image_override: str | None = None

    max_features: int = Field(default=0, ge=0)
    """Cap on features to mask (0 = every resolved feature from Select)."""
    max_workers: int = Field(default=constants.DEFAULT_MAX_WORKERS, ge=1)
    use_agent: bool = True
    """Run the Claude Code refine pass over each draft mask patch. Unlike Eval, Mask still accepts
    the deterministic draft when this is false or the refine does not produce an applicable patch."""

    agent_timeout: float = Field(default=3600.0, gt=0)
    request_timeout: float = 900.0
    """Host-side ceiling on each in-sandbox ``run_mask`` invoke (deterministic, fast)."""
    release_runtime: bool = False


class MaskTaskOutput(TaskRunOutput, WithScore):
    """Result of one repo's Mask run (aggregated over features)."""

    repo_name: str
    num_features: int = 0
    """Resolved Select features fed to Mask (the funnel denominator)."""
    num_applicable: int = 0
    """Features with a usable mask/gold — a clean run committed a patch that passed the cumulative
    apply-check (= the ``success`` count)."""
    num_agentic: int = 0
    """Subset of ``num_applicable`` whose CHOSEN mask came from the agent refine
    (``source == 'agentic'``, vs the heuristic-draft fallback)."""
    num_agent_failed: int = 0
    """Features whose refine agent soft-failed (invoke/parse/download) — still may be
    ``success=true`` via the heuristic draft. Observability only; does not affect score."""
    cover_rate_mean: float = 0.0
    """Mean per-feature ``cover_rate`` over agentic features (0 when none) — the average fraction
    of the refine's touched files that stayed within the feature's masked files."""
    unified_path: str | None = None  # mask.jsonl — the canonical deliverable (Eval reads this)
    runtime_error: str | None = None


class MaskUnifiedRow(BaseModel):
    """Schema for one ``mask.jsonl`` row — the lean per-feature deliverable Eval reads.

    Single source of truth for the unified field set: payload (``nodes`` / ``files`` /
    ``feature_mask_patch`` / ``feature_gold_patch``) + outcome (``source`` /
    ``success`` / ``error``) + diagnostics (``warnings``) + status (``completed`` /
    ``runtime_error``). ``nodes`` are the feature's function ids (Select's node objects reduced to
    ids); ``feature_`` on the patches distinguishes the feature-side mask from Eval's ``test_``
    patches. A patch is committed to ``feature_mask_patch`` / ``feature_gold_patch`` / ``source``
    only when it passes ``git apply --check``, so ``success`` means a clean run with a committed
    patch. ``deletes`` (heuristic plan) / ``cover_rate`` (``mask_agentic.json``) are not inlined —
    recoverable by joining on ``index``. Built by :func:`mask_unified_row`.
    """

    index: int
    nodes: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    feature_mask_patch: str = ''
    feature_gold_patch: str = ''
    source: str = ''
    success: bool = False
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    completed: bool = False
    runtime_error: str | None = None


def load_resolved_features(path: Path, *, max_features: int = 0) -> list[tuple[int, str, list[str]]]:
    """Read ``(slice index, description, node ids)`` for each resolved Select feature."""
    if not path.is_file():
        return []
    out: list[tuple[int, str, list[str]]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # index is the Select slice id (sgNN) Mask reuses to co-locate its artifacts, so a row
        # the agent didn't resolve (``source != 'agentic'``) or missing index/nodes is skipped.
        if row.get('source') == 'agentic' and isinstance(row.get('nodes'), list) and isinstance(row.get('index'), int):
            # select.jsonl stores the feature as node objects (``nodes``); Mask only needs their ids.
            feature_node_ids = [n['id'] for n in row['nodes'] if isinstance(n, dict) and isinstance(n.get('id'), str)]
            if feature_node_ids:
                out.append((row['index'], str(row.get('feature') or ''), feature_node_ids))
    if max_features > 0:
        out = out[:max_features]
    return out


def empty_mask_row(index: int, feature_node_ids: list[str]) -> dict[str, Any]:
    """A per-feature mask working row with no result yet (initial state / on failure).

    Accumulates content + status; ``mask_unified_row`` reads it to build the unified row.
    ``cover_rate`` is a working field (it shapes ``mask_agentic.json``) but is NOT emitted on the
    unified row; whether the chosen patch applies is a local in ``_mask_one`` that feeds ``success``."""
    return {
        'index': index,
        'nodes': feature_node_ids,
        'files': [],
        'source': '',
        'success': False,
        'error': None,
        'cover_rate': 0.0,
        'warnings': [],
        'completed': False,
        'runtime_error': None,
    }


def mask_unified_row(row: dict[str, Any], *, mask_patch: str, gold_patch: str) -> dict[str, Any]:
    """Build a :class:`MaskUnifiedRow` (dumped to dict) from the chosen patch pair + working row.

    The working *row* carries the content + status; this maps the wanted fields onto the explicit
    schema (which owns the field set) — the chosen ``mask_patch`` / ``gold_patch`` come in as args.
    """
    return MaskUnifiedRow(
        index=row['index'],
        nodes=row.get('nodes', []),
        files=row.get('files', []),
        feature_mask_patch=mask_patch,
        feature_gold_patch=gold_patch,
        source=row.get('source', ''),
        success=row.get('success', False),
        error=row.get('error'),
        warnings=row.get('warnings', []),
        completed=row.get('completed', False),
        runtime_error=row.get('runtime_error'),
    ).model_dump()


def empty_mask_pieces(index: int, feature_node_ids: list[str], msg: str | None = None) -> dict[str, Any]:
    """The three per-feature rows for a feature that never produced a mask (initial / hard failure)."""
    row = empty_mask_row(index, feature_node_ids)
    row['runtime_error'] = msg
    return {
        'heuristic': {'index': index, 'nodes': feature_node_ids},
        'agentic': {'index': index, 'cover_rate': 0.0, 'agentic_mask_patch': '', 'agentic_gold_patch': ''},
        'unified': mask_unified_row(row, mask_patch='', gold_patch=''),
    }


__all__ = [
    'MaskTaskInput',
    'MaskTaskOutput',
    'MaskUnifiedRow',
    'empty_mask_pieces',
    'empty_mask_row',
    'load_resolved_features',
    'mask_unified_row',
]
