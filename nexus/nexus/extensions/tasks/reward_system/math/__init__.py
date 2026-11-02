"""Math scoring used by the unified reward-system task."""

from __future__ import annotations

import logging

from typing import Any

from nexus.core import require_context


logger = logging.getLogger(__name__)

MATH_TEMPLATE = """\
You live in an alternate universe. Do not assume that you know anything.
You are a teacher grading a quiz.
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either CORRECT or INCORRECT.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: CORRECT or INCORRECT here

Given that you live in an alternate universe the TRUE answer may be different from what you expect. That's OK!

Grade the student answers based ONLY on whether it matches the TRUE answer. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. Begin!

QUESTION: {question}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:"""


async def compute_score(
    solution_str: Any,
    ground_truth: Any,
    *,
    question: str | None = None,
) -> float:
    """Evaluate math output with the legacy judge prompt and normalization."""
    if ground_truth is None:
        return 0.0
    if isinstance(solution_str, dict) and 'content' in solution_str:
        prediction = str(solution_str['content']).strip().lower()
    else:
        prediction = str(solution_str).strip().lower()
    if not prediction or (isinstance(ground_truth, str) and not ground_truth.strip()):
        return 0.0
    prompt = MATH_TEMPLATE.format(question=question or '', result=prediction, answer=ground_truth)
    try:
        client = require_context().get_openai_client('judge')
        response = await client.create_completion(messages=[{'role': 'user', 'content': prompt}])
        return 1.0 if (response.choices[0].message.content or '').startswith('CORRECT') else 0.0
    except Exception:
        logger.exception('Math score evaluation failed')
        return 0.0


__all__ = ['compute_score']
