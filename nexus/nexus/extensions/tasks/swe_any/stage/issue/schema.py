"""Schemas and host-side output shaping for the SWE-Any Issue stage.

Joins the masked features with successful structured-runner Eval oracles, plus the Select one-line
description when present, into the per-feature inputs the Issue agent needs.
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, WithScore
from nexus.extensions.tasks.swe_any.utils import constants


# A problem statement shorter than this is treated as the agent not really producing one.
_MIN_PROBLEM_STATEMENT_CHARS = 40


class IssueTaskInput(TaskRunInput, WithAgentRunner):
    """Business inputs for one repo's Issue run (agentic-only: write a non-leaking problem statement)."""

    instance: dict[str, Any] | None = None
    instance_path: str = ''
    """Optional one-row JSONL; ignored when ``instance`` is provided."""

    repo_name: str = 'repo'
    select_path: str = ''
    """Host path to Select's ``select.jsonl`` (the one-line feature description). Empty = default."""
    mask_path: str = ''
    """Host path to Mask's ``mask.jsonl`` (feature mask/gold patches). Empty = default."""
    eval_path: str = ''
    """Host path to Eval's ``eval.jsonl`` (the oracle patches). Empty = default."""
    output_dir: str = './outputs/swe_any'

    # Sandbox-side layout (agentic-only, so only the agentic + unified artifacts).
    repo_dir: str = constants.SANDBOX_REPO_DIR
    base_commit: str = ''
    """Git commit restored before this stage; hydrated from the source SWE-Env instance."""
    swe_any_dir: str = constants.SANDBOX_SWE_ANY_DIR
    agentic_filename: str = constants.ISSUE_AGENTIC_FILENAME
    agentic_jsonl_filename: str = constants.ISSUE_AGENTIC_JSONL_FILENAME
    unified_jsonl_filename: str = constants.ISSUE_JSONL_FILENAME
    runtime_name: str = 'main'

    image: str = ''
    image_override: str | None = None

    feature_indices: list[int] = Field(default_factory=list)
    """Optional exact feature indices to write (empty = all usable features)."""
    max_features: int = Field(default=0, ge=0)
    """Cap on features to write issues for (0 = every masked+oracled feature)."""
    max_workers: int = Field(default=constants.DEFAULT_MAX_WORKERS, ge=1)
    agent_timeout: float = Field(default=3600.0, gt=0)
    request_timeout: float = 900.0
    """Host-side ceiling on each in-sandbox git op (masked-state setup / cleanup)."""
    release_runtime: bool = False


class IssueTaskOutput(TaskRunOutput, WithScore):
    """Result of one repo's Issue run (aggregated over features)."""

    repo_name: str
    num_features: int = 0
    """Masked+oracled features fed to Issue (the funnel denominator)."""
    num_applicable: int = 0
    """Features with a usable problem statement (``success`` — the agent resolved a non-trivial one)."""
    num_agent_failed: int = 0
    """Features whose issue agent soft-failed (invoke/parse/download). Observability only."""
    unified_path: str | None = None  # issue.jsonl — the canonical deliverable (Export reads this)
    runtime_error: str | None = None


class IssueUnifiedRow(BaseModel):
    """Schema for one ``issue.jsonl`` row — the per-feature problem statement Export reads.

    Agentic-only stage, so ``source`` is ``'agentic'`` (and ``success = source != ''``) when the
    agent produced a non-trivial ``problem_statement``, else ``''`` with a business ``error``. A soft
    agent failure (invoke/parse/download) → ``warnings``, ``completed`` stays true (mirrors Select).
    ``feature`` / patches are not inlined — join Mask/Eval on ``index``. Built by :func:`issue_unified_row`.
    """

    index: int
    problem_statement: str = ''
    source: str = ''
    success: bool = False
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    completed: bool = False
    runtime_error: str | None = None


def _rows_by_index(path: Path) -> dict[int, dict[str, Any]]:
    """Read a ``*.jsonl`` into ``{index: row}`` (skips blank/index-less lines)."""
    out: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row.get('index'), int):
            out[row['index']] = row
    return out


def load_issue_features(
    select_path: Path,
    mask_path: Path,
    eval_path: Path,
    *,
    feature_indices: list[int] | None = None,
    max_features: int = 0,
) -> list[dict[str, Any]]:
    """Join Mask (``success``) ⋈ Eval (structured-runner ``success``) into Issue inputs.

    Returns one dict per feature that has BOTH a usable mask and a usable oracle, carrying the
    reference patches the prompt shows (``feature_gold_patch`` = the reference implementation,
    ``test_gold_patch`` = the forward grade oracle), the masking patches used to build the working
    tree (``feature_mask_patch`` + ``test_mask_patch``), and the Select one-line description when
    available.
    """
    select_rows = _rows_by_index(select_path)
    mask_rows = _rows_by_index(mask_path)
    eval_rows = _rows_by_index(eval_path)
    selected_indices = set(feature_indices or [])

    out: list[dict[str, Any]] = []
    for index in sorted(mask_rows):
        if selected_indices and index not in selected_indices:
            continue
        mask = mask_rows[index]
        ev = eval_rows.get(index)
        if not (mask.get('success') and ev and ev.get('success') and str(ev.get('test_runner_script') or '').strip()):
            continue
        feature_ids = [f for f in (mask.get('nodes') or []) if isinstance(f, str)]
        out.append({
            'index': index,
            'feature_desc': str((select_rows.get(index) or {}).get('feature') or ''),
            'feature_ids': feature_ids,
            'feature_mask_patch': str(mask.get('feature_mask_patch') or ''),
            'feature_gold_patch': str(mask.get('feature_gold_patch') or ''),
            'test_mask_patch': str(ev.get('test_mask_patch') or ''),
            'test_gold_patch': str(ev.get('test_gold_patch') or ''),
        })
    if max_features > 0:
        out = out[:max_features]
    return out


def normalize_issue_agentic(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate the agent's ``{problem_statement}`` → outcome fields (``problem_statement``/``source``/``success``/``error``).

    A non-trivial statement resolves the item (``source='agentic'``, ``success=True``); anything
    shorter is a business failure (empty row + ``error``), not a crash.
    """
    statement = str(raw.get('problem_statement') or '').strip()
    if len(statement) < _MIN_PROBLEM_STATEMENT_CHARS:
        return {
            'problem_statement': '',
            'source': '',
            'success': False,
            'error': f'problem_statement too short ({len(statement)} < {_MIN_PROBLEM_STATEMENT_CHARS} chars)',
        }
    return {'problem_statement': statement, 'source': 'agentic', 'success': True, 'error': None}


def empty_issue_row(index: int) -> dict[str, Any]:
    """A per-feature Issue working row with no result yet (initial state / on failure)."""
    return {
        'index': index,
        'problem_statement': '',
        'source': '',
        'success': False,
        'error': None,
        'warnings': [],
        'completed': False,
        'runtime_error': None,
    }


def issue_unified_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project the working *row* onto the explicit :class:`IssueUnifiedRow` schema (dumped to dict)."""
    return IssueUnifiedRow(
        index=row['index'],
        problem_statement=row.get('problem_statement', ''),
        source=row.get('source', ''),
        success=row.get('success', False),
        error=row.get('error'),
        warnings=row.get('warnings', []),
        completed=row.get('completed', False),
        runtime_error=row.get('runtime_error'),
    ).model_dump()


def empty_issue_pieces(index: int, msg: str | None = None) -> dict[str, Any]:
    """The two per-feature rows for a feature that produced no issue (initial / hard failure)."""
    row = empty_issue_row(index)
    row['runtime_error'] = msg
    return {
        'agentic': {'index': index, 'problem_statement': ''},
        'unified': issue_unified_row(row),
    }


__all__ = [
    'IssueTaskInput',
    'IssueTaskOutput',
    'IssueUnifiedRow',
    'empty_issue_pieces',
    'empty_issue_row',
    'issue_unified_row',
    'load_issue_features',
    'normalize_issue_agentic',
]
