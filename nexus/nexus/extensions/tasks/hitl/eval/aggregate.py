"""Combining several evaluators' results into one score for a run.

Independent of which evaluators ran: the input is a mapping of name to
:class:`EvaluationResult`, so a new evaluator is scored by naming it in
``score_policy`` and needs no change here.

``score_policy`` picks which result the run is scored on -- one evaluator by name,
or the mean of those that completed. ``unit_score_policy`` picks between the two
numbers the unit-test evaluator reports: the official all-or-nothing verdict, or
the fraction of tests that passed.
"""

import logging

from typing import Any, Literal

from nexus.extensions.tasks.hitl.eval.result import EvaluationResult


logger = logging.getLogger(__name__)

SWEUnitScorePolicy = Literal['official', 'fraction']
SWEScorePolicy = Literal['unit_test', 'patch_rubrics', 'mean']


def aggregate_swe_evaluations(
    results: dict[str, EvaluationResult],
    *,
    score_policy: SWEScorePolicy = 'mean',
    unit_score_policy: SWEUnitScorePolicy = 'official',
) -> tuple[float, bool, dict[str, Any]]:
    """Aggregate SWE recipe evaluators into first-class score/resolved fields."""

    eval_details = {name: result.model_dump() for name, result in results.items()}
    unit_result = results.get('unit_test')
    rubric_result = results.get('patch_rubrics')

    resolved = bool(_unit_official(unit_result).get('resolved', False)) if unit_result else False

    if score_policy == 'unit_test':
        score = _unit_score(unit_result, unit_score_policy)
    elif score_policy == 'patch_rubrics':
        score = _result_score(rubric_result)
    else:
        scores: list[float] = []
        if unit_result and unit_result.completed:
            scores.append(_unit_score(unit_result, unit_score_policy))
        if rubric_result and rubric_result.completed:
            scores.append(_result_score(rubric_result))
        score = sum(scores) / len(scores) if scores else 0.0

    eval_details['aggregate'] = {
        'score_policy': score_policy,
        'unit_score_policy': unit_score_policy,
        'resolved_source': 'unit_test.official',
        'score': score,
        'resolved': resolved,
    }
    return score, resolved, eval_details


def _unit_official(result: EvaluationResult | None) -> dict[str, Any]:
    if result is None:
        return {}
    official = result.details.get('official')
    return official if isinstance(official, dict) else {}


def _unit_score(result: EvaluationResult | None, policy: SWEUnitScorePolicy) -> float:
    if result is None:
        return 0.0
    if policy == 'fraction':
        fraction = result.details.get('fraction')
        if isinstance(fraction, dict):
            return float(fraction.get('score', 0.0) or 0.0)
    return float(_unit_official(result).get('score', result.score or 0.0) or 0.0)


def _result_score(result: EvaluationResult | None) -> float:
    if result is None:
        return 0.0
    return float(result.score or 0.0)


__all__ = ['SWEScorePolicy', 'SWEUnitScorePolicy', 'aggregate_swe_evaluations']
