from __future__ import annotations

import logging

from typing import Any

from nexus.core import require_context
from nexus.extensions.tasks.mcp_benchmark.types import EvalResult


logger = logging.getLogger(__name__)

CLAIM_JUDGE_SYSTEM_PROMPT = (
    'You are an impartial judge evaluating whether an AI assistant response satisfies a factual claim. '
    'Respond with exactly one of FULLY_SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED.'
)

SCORE_MAP = {
    'FULLY_SUPPORTED': 1.0,
    'PARTIALLY_SUPPORTED': 0.5,
    'NOT_SUPPORTED': 0.0,
}


class LLMJudgeEvaluator:
    async def evaluate(self, *, final_answer: str, config: dict[str, Any]) -> EvalResult:
        claims: list[str] = config.get('claims', [])
        threshold: float = config.get('threshold', 0.75)
        judge_client_name: str = config.get('judge_client', 'judge')

        if not claims:
            return EvalResult(passed=False, score=0.0, details={'error': 'no claims provided'})
        if not final_answer:
            return EvalResult(passed=False, score=0.0, details={'error': 'empty final answer'})

        client = require_context().get_openai_client(judge_client_name)
        claim_scores: list[dict[str, Any]] = []
        total = 0.0

        for claim in claims:
            response = await client.create_completion(
                messages=[
                    {'role': 'system', 'content': CLAIM_JUDGE_SYSTEM_PROMPT},
                    {'role': 'user', 'content': f'CLAIM: {claim}\n\nASSISTANT RESPONSE:\n{final_answer}'},
                ]
            )
            verdict = (response.choices[0].message.content or '').strip().upper()
            score = 0.0
            for key, value in SCORE_MAP.items():
                if key in verdict:
                    score = value
                    break
            total += score
            claim_scores.append({'claim': claim, 'verdict': verdict, 'score': score})

        coverage = total / len(claims)
        return EvalResult(
            passed=coverage >= threshold,
            score=coverage,
            details={
                'claims': claim_scores,
                'coverage_score': coverage,
                'threshold': threshold,
            },
        )
