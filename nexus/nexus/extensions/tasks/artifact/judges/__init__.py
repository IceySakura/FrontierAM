"""Judges sub-package for artifact task.

Lightweight public exports only. Judge dispatch lives in :mod:`registry` —
each strategy registers itself and is reached by name via
:func:`registry.get_judge` / :func:`registry.run_judge`. There is no central
dispatch dict to keep in sync.
"""

from __future__ import annotations

from nexus.extensions.tasks.artifact.judges.base import (
    AggregationMode,
    JudgeConfig,
    JudgeExecutionError,
    JudgeFailureKind,
    JudgeInput,
    JudgeResult,
    RubricResultItem,
    aggregate_judge_results,
)


__all__ = [
    'AggregationMode',
    'JudgeConfig',
    'JudgeExecutionError',
    'JudgeFailureKind',
    'JudgeInput',
    'JudgeResult',
    'RubricResultItem',
    'aggregate_judge_results',
]
