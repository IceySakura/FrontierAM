"""Evaluator factory for R2 STEM evaluation methods."""

from nexus.extensions.agents.base.evaluation import Evaluator
from nexus.extensions.projects.r2_stem.evaluator.llm_judge import LLMJudge


def build_evaluator(evaluator_type: str) -> Evaluator:
    """Build an evaluator by evaluation method name."""
    if evaluator_type == 'llm_judge':
        return LLMJudge()
    raise ValueError(f'Unsupported evaluator_type: {evaluator_type}')


__all__ = ['build_evaluator']
