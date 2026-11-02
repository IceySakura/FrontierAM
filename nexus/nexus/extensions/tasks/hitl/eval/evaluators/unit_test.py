"""Official SWE unit tests as an evaluator.

Runs the instance's own test suite against the produced patch in a sandbox the
caller has already prepared, and reports both scores the suite supports: the
official all-or-nothing verdict, and the fraction of tests that pass. Which of the
two a run is scored on is the aggregation step's decision, not this one's.

The fraction is the SWE task's own ``_compute_partial_score``, so both paths score a
report the same way.
"""

import json
import logging

from typing import Any

from nexus.extensions.tasks.hitl.eval.result import EvaluationResult
from nexus.extensions.tasks.swe.eval import _compute_partial_score, evaluate_and_grade_in_runtime
from nexus.runtime import Runtime


logger = logging.getLogger(__name__)


async def run_swe_unit_test_eval(
    runtime: Runtime,
    *,
    patch_text: str,
    instance: dict[str, Any],
    instance_id: str,
    timeout_seconds: int,
) -> EvaluationResult:
    """Run official SWE tests once and expose strict + fractional scores."""

    try:
        result = await evaluate_and_grade_in_runtime(
            runtime,
            patch_text,
            instance,  # ty: ignore[invalid-argument-type]
            instance_id,
            timeout_seconds,
            include_tests_status=True,
        )
    except Exception as exc:
        logger.exception('[hitl/swe/eval] unit_test evaluation failed')
        return EvaluationResult(
            name='unit_test',
            completed=False,
            score=0.0,
            error=f'{type(exc).__name__}: {exc}',
        )

    report_json = result.get('report_json', '') or ''
    instance_report = _instance_report(report_json, instance_id)
    official_resolved = bool(result.get('resolved', False))
    official_score = float(result.get('score', 1.0 if official_resolved else 0.0) or 0.0)
    fraction_score = _compute_partial_score(instance_report)

    return EvaluationResult(
        name='unit_test',
        completed=bool(result.get('completed', False)),
        score=official_score,
        error=result.get('error'),
        details={
            'official': {
                'score': official_score,
                'resolved': official_resolved,
            },
            'fraction': {
                'score': fraction_score,
                **_test_status_counts(instance_report),
            },
            'report_json': report_json,
            'test_output': result.get('test_output', ''),
            'apply_log': result.get('apply_log', ''),
            'error_code': result.get('error_code'),
        },
    )


def _instance_report(report_json: str, instance_id: str) -> dict[str, Any]:
    if not report_json:
        return {}
    try:
        report = json.loads(report_json)
    except json.JSONDecodeError:
        return {}
    value = report.get(instance_id)
    return value if isinstance(value, dict) else {}


def _test_status_counts(instance_report: dict[str, Any]) -> dict[str, int]:
    tests_status = instance_report.get('tests_status') or {}
    f2p = tests_status.get('FAIL_TO_PASS', {})
    p2p = tests_status.get('PASS_TO_PASS', {})
    return {
        'fail_to_pass_success': len(f2p.get('success', [])),
        'fail_to_pass_failure': len(f2p.get('failure', [])),
        'pass_to_pass_success': len(p2p.get('success', [])),
        'pass_to_pass_failure': len(p2p.get('failure', [])),
    }


__all__ = ['run_swe_unit_test_eval']
