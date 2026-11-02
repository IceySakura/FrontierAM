"""Schemas and host-side output shaping for the SWE-Any Export stage."""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from nexus.core import TaskRunInput, TaskRunOutput, WithScore
from nexus.extensions.tasks.swe_any.utils import constants
from nexus.extensions.tasks.swe_any.utils.source_edit import patch_test_targets
from nexus.extensions.tasks.swe_any.utils.test_runner import RUNNER_PROTOCOL_VERSION, derive_structured_reports


class ExportTaskInput(TaskRunInput):
    """Business inputs for one repo's Export run (assemble → execution-grade → Harbor emit)."""

    instance: dict[str, Any] | None = None
    instance_path: str = ''

    repo_name: str = 'repo'
    mask_path: str = ''
    eval_path: str = ''
    issue_path: str = ''
    output_dir: str = './outputs/swe_any'

    repo_dir: str = constants.SANDBOX_REPO_DIR
    base_commit: str = ''
    """Git commit restored before grading; hydrated from the source SWE-Env instance."""
    runtime_name: str = 'main'
    image: str = ''
    image_override: str | None = None

    use_hack_check: bool = False
    """Reserved: run an agent to flag instances solvable by non-standard means (residual build info,
    online lookup). NOT implemented — setting true raises, so the config cannot silently claim a
    dataset was hack-screened."""

    max_features: int = Field(default=0, ge=0)
    max_workers: int = Field(default=constants.DEFAULT_MAX_WORKERS, ge=1)
    grade_timeout: float = 1800.0
    """Host-side ceiling on each in-sandbox grade invoke (two pytest runs)."""
    release_runtime: bool = False


class ExportTaskOutput(TaskRunOutput, WithScore):
    """Result of one repo's Export run (aggregated over instances)."""

    repo_name: str
    num_features: int = 0
    """Features fed to Export (usable mask + oracle + issue)."""
    num_exported: int = 0
    """Features that graded VALID (non-empty FAIL_TO_PASS, no regression) and were written to Harbor."""
    harbor_dir: str | None = None
    unified_path: str | None = None  # export.jsonl
    runtime_error: str | None = None


class ExportUnifiedRow(BaseModel):
    """Schema for one ``export.jsonl`` row — the per-instance grade result.

    ``success`` = the instance graded VALID (non-empty ``fail_to_pass`` and no pregold-pass →
    postgold-fail regression), i.e. a usable, execution-validated task was emitted to Harbor.
    Follows the framework outcome/status axes; ``harbor_path`` is the emitted directory (when valid).
    Built by :func:`export_unified_row`.
    """

    index: int
    instance_id: str = ''
    success: bool = False
    fail_to_pass: list[str] = Field(default_factory=list)
    pass_to_pass: list[str] = Field(default_factory=list)
    harbor_path: str | None = None
    error: str | None = None
    error_kind: str | None = None
    """Machine-readable failure category; keeps infrastructure/execution failures distinct from a
    validly executed oracle that does not gate the feature."""
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


def load_export_features(
    mask_path: Path, eval_path: Path, issue_path: Path, *, max_features: int = 0
) -> list[dict[str, Any]]:
    """Join Mask ⋈ Eval ⋈ Issue on ``index`` for every feature ready to export (issue ``success``).

    Returns one dict per feature carrying every patch Export needs — the bug patches
    (``feature_mask_patch`` + ``test_mask_patch``), the feature gold (``feature_gold_patch``), the
    forward test oracle (``test_gold_patch``), required agent-authored ``test_runner_script``,
    ``test_files``, and the ``problem_statement``. Gating on issue ``success`` implies the mask +
    oracle were usable too.
    """
    mask_rows = _rows_by_index(mask_path)
    eval_rows = _rows_by_index(eval_path)
    issue_rows = _rows_by_index(issue_path)

    out: list[dict[str, Any]] = []
    for index in sorted(issue_rows):
        issue = issue_rows[index]
        mask = mask_rows.get(index)
        ev = eval_rows.get(index)
        if not (issue.get('success') and mask and mask.get('success') and ev and ev.get('success')):
            continue
        test_gold_patch = str(ev.get('test_gold_patch') or '')
        test_runner_script = str(ev.get('test_runner_script') or '').strip()
        if not test_runner_script:
            continue
        # Re-derive from test_gold instead of trusting persisted Eval rows. This makes a corrected
        # Export safe to rerun over older artifacts whose test_files also contained mask-only paths.
        test_files = patch_test_targets(test_gold_patch)
        if not test_files:
            continue
        out.append({
            'index': index,
            'problem_statement': str(issue.get('problem_statement') or ''),
            'feature_mask_patch': str(mask.get('feature_mask_patch') or ''),
            'feature_gold_patch': str(mask.get('feature_gold_patch') or ''),
            'test_mask_patch': str(ev.get('test_mask_patch') or ''),
            'test_gold_patch': test_gold_patch,
            'test_runner_script': test_runner_script,
            'test_files': test_files,
        })
    if max_features > 0:
        out = out[:max_features]
    return out


def export_instance_id(index: int, instance: dict[str, Any]) -> str:
    """The Harbor task name / SWE-bench ``instance_id`` for this item: ``swe_any.<repo>.sgNN``.

    Split out so the caller can stamp the id cheaply (before grading) without assembling the whole
    record; :func:`build_swebench_instance` reuses it.
    """
    repo = str(instance.get('repo') or instance.get('repo_name') or '')
    repo_name = str(instance.get('repo_name') or instance.get('instance_id') or repo or 'repo')
    return f'swe_any.{repo_name}.sg{int(index):02d}'


def build_swebench_instance(
    record: dict[str, Any],
    instance: dict[str, Any],
    *,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
) -> dict[str, Any]:
    """Assemble a ``SWEbenchInstance``-shaped dict from a joined *record* + the SWE-Env *instance*.

    ``patch`` (gold fix) = ``feature_gold_patch``; ``test_patch`` (the forward grade-time oracle) =
    ``test_gold_patch``; bug state = ``feature_mask_patch`` + ``test_mask_patch`` (kept alongside for
    the Harbor Dockerfile). Every row carries the exact Eval-authored ``test_runner_script`` that
    Export graded.
    """
    repo = str(instance.get('repo') or instance.get('repo_name') or '')
    base_commit = str(instance.get('base_commit') or (instance.get('base') or {}).get('sha') or '')
    return {
        'dataset_type': 'swe-any',
        'instance_id': export_instance_id(int(record['index']), instance),
        'repo': repo,
        'base_commit': base_commit,
        'problem_statement': record['problem_statement'],
        'patch': record['feature_gold_patch'],
        'test_patch': record['test_gold_patch'],
        'FAIL_TO_PASS': fail_to_pass,
        'PASS_TO_PASS': pass_to_pass,
        'image': str(instance.get('image') or ''),
        'language': str(instance.get('language') or 'python'),
        'workdir': str(instance.get('workdir') or '/testbed'),
        # Bug state (not part of SWE-bench proper, but the Harbor Dockerfile bakes it in):
        'feature_mask_patch': record['feature_mask_patch'],
        'test_mask_patch': record['test_mask_patch'],
        'test_gold_patch': record['test_gold_patch'],
        'test_runner_script': str(record['test_runner_script']),
        'test_runner_protocol': RUNNER_PROTOCOL_VERSION,
        'test_files': record['test_files'],
    }


def empty_export_row(index: int) -> dict[str, Any]:
    """A per-instance Export working row with no result yet (initial state / on failure)."""
    return {
        'index': index,
        'instance_id': '',
        'success': False,
        'fail_to_pass': [],
        'pass_to_pass': [],
        'harbor_path': None,
        'error': None,
        'error_kind': None,
        'warnings': [],
        'completed': False,
        'runtime_error': None,
    }


def export_unified_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project the working *row* onto the explicit :class:`ExportUnifiedRow` schema (dumped to dict)."""
    return ExportUnifiedRow(
        index=row['index'],
        instance_id=row.get('instance_id', ''),
        success=row.get('success', False),
        fail_to_pass=row.get('fail_to_pass', []),
        pass_to_pass=row.get('pass_to_pass', []),
        harbor_path=row.get('harbor_path'),
        error=row.get('error'),
        error_kind=row.get('error_kind'),
        warnings=row.get('warnings', []),
        completed=row.get('completed', False),
        runtime_error=row.get('runtime_error'),
    ).model_dump()


__all__ = [
    'ExportTaskInput',
    'ExportTaskOutput',
    'ExportUnifiedRow',
    'build_swebench_instance',
    'derive_structured_reports',
    'empty_export_row',
    'export_instance_id',
    'export_unified_row',
    'load_export_features',
]
