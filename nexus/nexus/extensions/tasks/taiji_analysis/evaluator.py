"""Taiji Analysis evaluator — LLM Judge for root cause consistency."""

import re

from dataclasses import dataclass

from nexus.core import require_context
from nexus.extensions.tasks.taiji_analysis.prompt import TAIJI_JUDGE_PROMPT, TAIJI_JUDGE_SYSTEM_PROMPT


@dataclass
class JudgeResult:
    score: float
    reasoning: str


async def evaluate_root_cause(prediction: str, reference: str, question: str) -> JudgeResult:
    """Judge whether prediction and reference reports identify the same root cause.

    Returns JudgeResult with score (1.0 if consistent, 0.0 otherwise) and reasoning.
    """
    if not prediction.strip():
        return JudgeResult(score=0.0, reasoning='Empty prediction')

    prompt = TAIJI_JUDGE_PROMPT.format(
        question=question,
        report_a=reference,
        report_b=prediction,
    )

    ctx = require_context()
    client = ctx.get_openai_client('judge')
    response = await client.create_completion(
        messages=[
            {'role': 'system', 'content': TAIJI_JUDGE_SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ]
    )

    output = response.choices[0].message.content or ''
    score_match = re.search(r'<is_consistent>\s*([01])\s*</is_consistent>', output)
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', output, re.DOTALL)

    score = float(score_match.group(1)) if score_match else 0.0
    reasoning = reasoning_match.group(1).strip() if reasoning_match else output.strip()

    return JudgeResult(score=score, reasoning=reasoning)
