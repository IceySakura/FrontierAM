"""Schemas and host-side output shaping for the SWE-Any Eval stage.

Reads the masked features Eval consumes and builds the per-feature unified rows
(status inlined; plus the empty-row variants used on initial state / hard failure).
"""

from __future__ import annotations

import json

from enum import IntEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, StrictInt, field_validator, model_validator

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, WithScore
from nexus.extensions.tasks.swe_any.utils import constants


class EvalTaskInput(TaskRunInput, WithAgentRunner):
    """Business inputs for one repo's Eval run (mechanical hints, agent refine, agent verify)."""

    instance: dict[str, Any] | None = None
    instance_path: str = ''
    """Optional one-row JSONL; ignored when ``instance`` is provided."""

    repo_name: str = 'repo'
    graph_path: str = ''
    """Host path to parse_graph.json (from Parse); uploaded to the sandbox for run_eval."""
    mask_path: str = ''
    """Host path to Mask's unified mask.jsonl (its masked features). Empty = ``output_dir/<repo>/mask.jsonl``."""
    output_dir: str = './outputs/swe_any'

    # Sandbox-side layout.
    repo_dir: str = constants.SANDBOX_REPO_DIR
    base_commit: str = ''
    """Git commit restored before this stage; hydrated from the source SWE-Env instance."""
    swe_any_dir: str = constants.SANDBOX_SWE_ANY_DIR
    graph_filename: str = constants.GRAPH_FILENAME
    heuristic_filename: str = constants.EVAL_HEURISTIC_FILENAME
    agentic_filename: str = constants.EVAL_AGENTIC_FILENAME
    heuristic_jsonl_filename: str = constants.EVAL_HEURISTIC_JSONL_FILENAME
    agentic_jsonl_filename: str = constants.EVAL_AGENTIC_JSONL_FILENAME
    unified_jsonl_filename: str = constants.EVAL_JSONL_FILENAME
    runtime_name: str = 'main'

    image: str = ''
    image_override: str | None = None

    max_features: int = Field(default=0, ge=0)
    """Cap on features to build tests for (0 = every masked feature)."""
    max_workers: int = Field(default=constants.DEFAULT_MAX_WORKERS, ge=1)

    include_feature_tests: bool = True
    """Whether a test Select pulled into the feature may still serve as an oracle
    candidate. Mask keeps tests intact, so such a test still exercises the masked
    feature — kept by default; set false to treat it as part of the feature and drop it."""
    agent_timeout: float = Field(default=3600.0, gt=0)
    request_timeout: float = 900.0
    """Host-side ceiling on each in-sandbox ``run_eval`` invoke (deterministic, fast)."""
    release_runtime: bool = False


class EvalTaskOutput(TaskRunOutput, WithScore):
    """Result of one repo's Eval run (aggregated over features)."""

    repo_name: str
    num_features: int = 0
    """Masked features fed to Eval — mask.jsonl rows with a mask (the funnel denominator)."""
    num_applicable: int = 0
    """Features with a usable runner-backed oracle accepted by the independent verifier."""
    num_agentic: int = 0
    """Agent-authored applicable oracles; equal to ``num_applicable`` for runner-only Eval."""
    num_agent_failed: int = 0
    """Features whose oracle agent soft-failed (invoke/parse/download)."""
    num_verifier_rejected: int = 0
    """Applicable agent-authored oracles assigned any non-zero verifier verdict."""
    unified_path: str | None = None  # eval.jsonl — chosen test patches + structured runner per feature
    runtime_error: str | None = None


class EvalVerifierVerdictCode(IntEnum):
    """Stable primary issue taxonomy emitted by the Eval verifier."""

    OTHER = -1
    CLEAN = 0
    ORACLE_LEAKAGE = 1
    OUT_OF_SCOPE_OR_DESTRUCTIVE = 2
    OVERCONSTRAINED = 3
    TOO_WEAK = 4
    EASILY_GAMED = 5
    RUNNER_INTEGRITY = 6
    NON_HERMETIC_OR_FLAKY = 7


class EvalVerifierVerdict(BaseModel):
    """Strict second-agent review stored as one object in the unified Eval row."""

    model_config = {'extra': 'forbid'}

    verdict: StrictInt
    rationale: str
    findings: list[str] = Field(default_factory=list)

    @field_validator('verdict')
    @classmethod
    def validate_verdict(cls, value: int) -> int:
        try:
            EvalVerifierVerdictCode(value)
        except ValueError as exc:
            allowed = ', '.join(str(code.value) for code in EvalVerifierVerdictCode)
            raise ValueError(f'verdict must be one of: {allowed}') from exc
        return value

    @field_validator('rationale')
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('rationale must be non-empty')
        return value

    @field_validator('findings')
    @classmethod
    def normalize_findings(cls, value: list[str]) -> list[str]:
        return [finding.strip() for finding in value if finding.strip()]

    @model_validator(mode='after')
    def validate_findings_match_verdict(self) -> EvalVerifierVerdict:
        if self.verdict == EvalVerifierVerdictCode.CLEAN and self.findings:
            raise ValueError('clean verdict must not contain findings')
        if self.verdict != EvalVerifierVerdictCode.CLEAN and not self.findings:
            raise ValueError('non-clean verdict must contain at least one finding')
        return self


class EvalUnifiedRow(BaseModel):
    """Schema for one ``eval.jsonl`` row — the per-feature test-side deliverable.

    Single source of truth for the unified field set. The oracle is TWO patches that are NOT
    inverses of each other plus a required structured runner and independent verifier verdict:

    - ``test_mask_patch``: deletes/hides EVERY existing repo test that exercises the feature (so the
      solver can't read the oracle). Applied to the solver-facing state.
    - ``test_gold_patch``: the FORWARD grade-time test set — a generalized / newly-authored oracle
      (plus optional feature-independent P2P guards). It does NOT restore the masked originals
      verbatim; it may replace them with behaviour-level tests.
    - ``test_runner_script``: the agent-authored, grade-only shell adapter that runs the oracle in
      this repository's real environment and emits the versioned structured JSONL protocol.
    - ``verifier``: the second agent's strict ``verdict`` / ``rationale`` / ``findings`` review.
      Verdict ``0`` is clean, positive values are known issue classes, and ``-1`` is other.

    Plus ``test_files`` (executable targets touched by ``test_gold_patch``; Python support files such
    as ``__init__.py`` / ``conftest.py`` are excluded) + outcome (``source`` / ``success`` / ``error``)
    + diagnostics (``warnings``) + status (``completed`` / ``runtime_error``). Export keeps only paths
    that exist post mask+gold, runs every test in them on the masked vs gold state, and derives
    FAIL_TO_PASS / PASS_TO_PASS by execution (P2F ⇒ the instance is rejected). Eval stores no
    precomputed per-test ids; the structured runner emits stable ids during execution.
    ``success`` means a clean run with a committed, apply-checked oracle. Feature nodes are not
    inlined — join Mask's ``mask.jsonl`` on ``index``. Built by :func:`eval_unified_row`.
    """

    index: int
    test_mask_patch: str = ''
    test_gold_patch: str = ''
    test_runner_script: str = ''
    verifier: EvalVerifierVerdict | None = None
    source: str = ''
    success: bool = False
    error: str | None = None
    test_files: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    completed: bool = False
    runtime_error: str | None = None


def load_masked_features(path: Path, *, max_features: int = 0) -> list[tuple[int, list[str], str, str]]:
    """Read ``(index, feature ids, feature_mask_patch, feature_gold_patch)`` per usable feature.

    Reads Mask's unified ``mask.jsonl``; only rows with a committed, apply-checked mask
    (``success`` — a real ``feature_mask_patch`` / ``feature_gold_patch``) are eval targets. Gate
    on ``success``, NOT on ``files``: ``files`` is the heuristic plan's masked-file list and stays
    non-empty even when no candidate patch applied (``success=false`` / empty ``feature_mask_patch``),
    so gating on it would build oracles for features that were never actually masked. The two feature
    patches are surfaced as read-only context for the Eval oracle agent (it reads them by path to
    understand WHAT the feature is), not to re-apply.
    """
    if not path.is_file():
        return []
    out: list[tuple[int, list[str], str, str]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('success') and isinstance(row.get('index'), int) and isinstance(row.get('nodes'), list):
            feature = [f for f in row['nodes'] if isinstance(f, str)]
            feature_mask_patch = str(row.get('feature_mask_patch') or '')
            feature_gold_patch = str(row.get('feature_gold_patch') or '')
            if feature:
                out.append((row['index'], feature, feature_mask_patch, feature_gold_patch))
    if max_features > 0:
        out = out[:max_features]
    return out


def empty_eval_row(index: int) -> dict[str, Any]:
    """A per-feature eval working row with no result yet (initial state / on failure).

    Accumulates outcome + status; ``eval_unified_row`` reads it to build the unified row."""
    return {
        'index': index,
        'source': '',
        'success': False,
        'error': None,
        'test_files': [],
        'warnings': [],
        'completed': False,
        'runtime_error': None,
    }


def eval_unified_row(row: dict[str, Any], *, patches: dict[str, Any]) -> dict[str, Any]:
    """Build an :class:`EvalUnifiedRow` (dumped to dict): the chosen oracle + outcome + status.

    The working *row* carries the outcome (``source`` / ``success`` / ``error`` / ``test_files``)
    and status; the chosen test patches come in via *patches*. Fields are mapped onto the explicit
    schema, which owns the field set.
    """
    return EvalUnifiedRow(
        index=row['index'],
        test_mask_patch=patches.get('test_mask_patch', ''),
        test_gold_patch=patches.get('test_gold_patch', ''),
        test_runner_script=patches.get('test_runner_script', ''),
        verifier=patches.get('verifier'),
        source=row.get('source', ''),
        success=row.get('success', False),
        error=row.get('error'),
        test_files=row.get('test_files', []),
        warnings=row.get('warnings', []),
        completed=row.get('completed', False),
        runtime_error=row.get('runtime_error'),
    ).model_dump()


def parse_eval_verifier_verdict(text: str) -> dict[str, Any]:
    """Parse the verifier's strict JSON artifact and derive its non-author-controlled gate."""
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError('verifier output must be a JSON object')
    return EvalVerifierVerdict.model_validate(raw).model_dump()


def empty_eval_pieces(index: int, msg: str | None = None) -> dict[str, Any]:
    """The three per-feature rows for a feature that produced no oracle (initial / hard failure)."""
    row = empty_eval_row(index)
    row['runtime_error'] = msg
    return {
        'heuristic': {'index': index},
        'agentic': {
            'index': index,
            'agentic_mask_patch': '',
            'agentic_gold_patch': '',
            'agentic_runner_script': '',
            'agentic_test_files': [],
        },
        'unified': eval_unified_row(row, patches={}),
    }


__all__ = [
    'EvalTaskInput',
    'EvalTaskOutput',
    'EvalUnifiedRow',
    'EvalVerifierVerdict',
    'EvalVerifierVerdictCode',
    'empty_eval_pieces',
    'empty_eval_row',
    'eval_unified_row',
    'load_masked_features',
    'parse_eval_verifier_verdict',
]
