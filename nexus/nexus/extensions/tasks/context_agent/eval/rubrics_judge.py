"""Rubrics-based evaluation — thin wrapper around :mod:`.judge`."""

from __future__ import annotations

from .judge import judge


async def evaluate(model_output: str, task: dict, llm) -> dict:
    """Evaluate *model_output* against the ``rubrics`` field in *task*.

    Delegates to the existing :func:`.judge.judge` so all
    rubrics-based datasets (CLB, CLB-Life, Pluto20, CL-eval-120, …)
    share the same grading logic and return ``criteria_passed`` /
    ``criteria_total`` alongside ``requirement_status``.
    """
    rubrics: list[str] = task.get('rubrics', [])
    return await judge(model_output, rubrics, llm)


async def evaluate_accuracy(model_output: str, task: dict, llm) -> dict:
    """Grade each rubric independently and return the fraction passed."""
    rubrics: list[str] = task.get('rubrics', [])
    result = await judge(model_output, rubrics, llm)
    if not model_output or not str(model_output).strip():
        return result

    statuses = [str(value).strip().lower() for value in result.get('requirement_status') or []]
    if len(statuses) != len(rubrics) or any(value not in {'yes', 'no'} for value in statuses):
        return {
            **result,
            'score': None,
            'rationale': (
                'Judge failed (rubric_accuracy): expected one yes/no requirement status '
                f'per rubric, got {len(statuses)} for {len(rubrics)}'
            ),
            'reward_error': True,
            'graded_on': 'final_message',
        }
    passed = sum(value == 'yes' for value in statuses)
    result.update(
        score=passed / len(rubrics) if rubrics else 0.0,
        graded_on='final_message',
        criteria_passed=passed,
        criteria_total=len(rubrics),
    )
    return result
