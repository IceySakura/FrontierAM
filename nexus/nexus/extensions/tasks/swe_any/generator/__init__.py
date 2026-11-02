"""``nexus run-batch`` generators for stages and feature pipelines.

``stage/<stage>.py`` contains the legacy per-stage generators, while
``pipeline/{pre_feature,feature,post_feature}.py`` contains the feature-pipeline
generators. They start from the matching config subtree and share the
``..utils.batch_config`` merge pipeline.
"""

from __future__ import annotations
