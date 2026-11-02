"""Feature-level SWE-Any pipeline orchestration.

The additive path consists of repo-level preparation, one task per feature, and
deterministic post-feature reduction. The legacy six-stage pipeline remains in
``nexus.extensions.tasks.swe_any.task``.
"""

from nexus.extensions.tasks.swe_any.pipeline.feature import run_swe_any_pipeline_feature
from nexus.extensions.tasks.swe_any.pipeline.post_feature import run_swe_any_pipeline_post_feature
from nexus.extensions.tasks.swe_any.pipeline.pre_feature import run_swe_any_pipeline_pre_feature
from nexus.extensions.tasks.swe_any.pipeline.schema import (
    PipelineFeatureTaskInput,
    PipelineFeatureTaskOutput,
    PipelinePostFeatureTaskInput,
    PipelinePostFeatureTaskOutput,
    PipelinePreFeatureTaskInput,
    PipelinePreFeatureTaskOutput,
)


__all__ = [
    'PipelineFeatureTaskInput',
    'PipelineFeatureTaskOutput',
    'PipelinePostFeatureTaskInput',
    'PipelinePostFeatureTaskOutput',
    'PipelinePreFeatureTaskInput',
    'PipelinePreFeatureTaskOutput',
    'run_swe_any_pipeline_feature',
    'run_swe_any_pipeline_post_feature',
    'run_swe_any_pipeline_pre_feature',
]
