"""Task contracts for the additive three-stage SWE-Any feature pipeline.

Manifest rows and ``feature_result.json`` remain JSON/dict contracts.  The
pipeline inputs carry only orchestration-owned fields plus per-stage override
mappings; pipeline task implementations must merge those mappings into the
existing stage ``TaskInput`` models before invoking a stage runner.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator, model_validator

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, WithScore
from nexus.extensions.tasks.swe_any.stage.eval.schema import EvalTaskInput
from nexus.extensions.tasks.swe_any.stage.export.schema import ExportTaskInput
from nexus.extensions.tasks.swe_any.stage.issue.schema import IssueTaskInput
from nexus.extensions.tasks.swe_any.stage.mask.schema import MaskTaskInput
from nexus.extensions.tasks.swe_any.stage.parse.schema import ParseTaskInput
from nexus.extensions.tasks.swe_any.stage.select.schema import SelectTaskInput
from nexus.extensions.tasks.swe_any.utils import constants


_STAGE_INPUT_TYPES = {
    'parse': ParseTaskInput,
    'select': SelectTaskInput,
    'mask': MaskTaskInput,
    'eval': EvalTaskInput,
    'issue': IssueTaskInput,
    'export': ExportTaskInput,
}
_PRE_FEATURE_STAGES = frozenset({'parse', 'select'})
_FEATURE_STAGES = frozenset({'select', 'mask', 'eval', 'issue', 'export'})


def _validate_stage_configs(
    configs: dict[str, dict[str, Any]],
    *,
    allowed_stages: frozenset[str],
) -> dict[str, dict[str, Any]]:
    unknown_stages = set(configs) - allowed_stages
    if unknown_stages:
        raise ValueError(f'unsupported stage_configs keys: {sorted(unknown_stages)}')

    for stage, overrides in configs.items():
        if not isinstance(overrides, dict):
            raise ValueError(f'stage_configs.{stage} must be a mapping')
        allowed_fields = set(_STAGE_INPUT_TYPES[stage].model_fields) - {'run_agent'}
        unknown_fields = set(overrides) - allowed_fields
        if unknown_fields:
            raise ValueError(
                f'unknown {stage} TaskInput overrides: {sorted(unknown_fields)} '
                f'(authoritative model: {_STAGE_INPUT_TYPES[stage].__name__})'
            )
    return configs


class PipelinePreFeatureTaskInput(TaskRunInput):
    """Prepare one repository: Parse, heuristic Select, and a feature manifest."""

    instance: dict[str, Any] | None = None
    instance_path: str = ''
    repo_name: str = 'repo'
    output_dir: str = './outputs/swe_any'

    repo_dir: str = constants.SANDBOX_REPO_DIR
    base_commit: str = ''
    swe_any_dir: str = constants.SANDBOX_SWE_ANY_DIR
    runtime_name: str = 'main'
    image: str = ''
    image_override: str | None = None
    release_runtime: bool = True

    manifest_filename: str = constants.FEATURE_MANIFEST_FILENAME
    stage_configs: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {'parse': {}, 'select': {'max_workers': 1}}
    )

    @field_validator('stage_configs')
    @classmethod
    def validate_stage_configs(cls, value: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return _validate_stage_configs(value, allowed_stages=_PRE_FEATURE_STAGES)


class PipelinePreFeatureTaskOutput(TaskRunOutput, WithScore):
    """Repo-level preparation result consumed by the feature generator."""

    repo_name: str
    num_features: int = 0
    graph_path: str | None = None
    manifest_path: str | None = None
    runtime_error: str | None = None


class PipelineFeatureTaskInput(TaskRunInput, WithAgentRunner):
    """Run one manifest row through Select agentic, Mask, Eval, Issue, and Export."""

    instance: dict[str, Any]
    repo_name: str
    index: int = Field(ge=0)
    graph_path: str
    heuristic_row: dict[str, Any]
    heuristic_error: str | None = None
    """Pre-feature heuristic failure, if this slice could not be constructed."""
    select_settings: dict[str, Any] = Field(default_factory=dict)
    """Deterministic Select settings captured by pre-feature for graph-context reconstruction."""
    manifest_path: str = ''
    output_dir: str = './outputs/swe_any'

    repo_dir: str = constants.SANDBOX_REPO_DIR
    base_commit: str = ''
    swe_any_dir: str = constants.SANDBOX_SWE_ANY_DIR
    runtime_name: str = 'main'
    image: str = ''
    image_override: str | None = None
    release_runtime: bool = True

    stage_configs: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {
            'select': {'max_workers': 1},
            'mask': {'max_workers': 1},
            'eval': {'max_workers': 1},
            'issue': {'max_workers': 1},
            'export': {'max_workers': 1},
        }
    )

    @field_validator('stage_configs')
    @classmethod
    def validate_stage_configs(cls, value: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return _validate_stage_configs(value, allowed_stages=_FEATURE_STAGES)

    @model_validator(mode='after')
    def validate_manifest_row(self) -> PipelineFeatureTaskInput:
        if not self.instance:
            raise ValueError('instance must contain the source SWE-Env repository record')
        if not self.graph_path:
            raise ValueError('graph_path is required')
        heuristic_index = self.heuristic_row.get('index')
        if heuristic_index != self.index:
            raise ValueError(f'heuristic_row index {heuristic_index!r} does not match feature index {self.index}')
        return self


class PipelineFeatureTaskOutput(TaskRunOutput, WithScore):
    """Terminal result for one independently recoverable feature."""

    repo_name: str
    index: int = Field(ge=0)
    status: str
    terminal_stage: str | None = None
    feature_result_path: str | None = None
    instance_id: str | None = None
    error: str | None = None


class PipelinePostFeatureTaskInput(TaskRunInput):
    """Deterministically reduce every feature batch record into dataset artifacts."""

    output_dir: str = './outputs/swe_any'
    feature_batch_results_path: str
    feature_runs: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator('feature_runs')
    @classmethod
    def validate_feature_runs(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, int]] = set()
        for record in value:
            run_id = record.get('run_id')
            batch_status = record.get('batch_status')
            repo_name = record.get('repo_name')
            index = record.get('index')
            instance = record.get('instance')
            feature_result_path = record.get('feature_result_path')
            if not isinstance(run_id, str) or not run_id:
                raise ValueError('each feature_runs record requires run_id')
            if not isinstance(batch_status, str) or not batch_status:
                raise ValueError(f'feature_runs record {run_id!r} requires batch_status')
            if not isinstance(repo_name, str) or not repo_name:
                raise ValueError('each feature_runs record requires repo_name')
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise ValueError(f'feature_runs record for {repo_name!r} requires a non-negative integer index')
            if not isinstance(instance, dict):
                raise ValueError(f'feature_runs record {run_id!r} requires the source instance')
            if feature_result_path is not None and not isinstance(feature_result_path, str):
                raise ValueError(f'feature_runs record {run_id!r} feature_result_path must be a string or null')
            key = (repo_name, index)
            if key in seen:
                raise ValueError(f'duplicate feature_runs record: {repo_name}/{constants.slice_dirname(index)}')
            seen.add(key)
        return value


class PipelinePostFeatureTaskOutput(TaskRunOutput, WithScore):
    """Dataset-level deterministic collection result."""

    num_repos: int = 0
    num_features: int = 0
    num_batch_failed: int = 0
    num_missing_results: int = 0
    num_corrupt_results: int = 0
    num_select_resolved: int = 0
    num_masked: int = 0
    num_oracled: int = 0
    num_issues: int = 0
    num_exported: int = 0
    instances_path: str | None = None
    harbor_dir: str | None = None
    dataset_toml_path: str | None = None
    runtime_error: str | None = None


__all__ = [
    'PipelineFeatureTaskInput',
    'PipelineFeatureTaskOutput',
    'PipelinePostFeatureTaskInput',
    'PipelinePostFeatureTaskOutput',
    'PipelinePreFeatureTaskInput',
    'PipelinePreFeatureTaskOutput',
]
