"""Score screenshot aesthetics using the shared VLM capability."""

from __future__ import annotations

import logging

from nexus.extensions.tasks.artifact.judges.base import JudgeInput, JudgeResult
from nexus.extensions.tasks.artifact.judges.capabilities.llm import call_judge_llm
from nexus.extensions.tasks.artifact.judges.capabilities.parse import sum_tag_scores
from nexus.extensions.tasks.artifact.judges.capabilities.prompts import load_prompt, render_prompt
from nexus.extensions.tasks.artifact.judges.registry import register_judge


logger = logging.getLogger(__name__)

NAME = 'aesthetic'

# 5 dims × 0-5 each. Weight (3,1,1,1,1) — dim 1 (AI-template signal) is the
# source prompt's "most important" dimension, and its scoring anchors explicitly
# call for severe penalties ("严厉惩罚"); the code-review judge already
# precedents 2x weighting, 3x here reflects that stronger emphasis.
SCORE_TAGS = ('ai_template', 'color_system', 'composition', 'style', 'typography')
SCORE_WEIGHTS = (3.0, 1.0, 1.0, 1.0, 1.0)
# Weighted max: 5*3 + 5*1*4 = 35 → normalise weighted sum to [0, 1].
SCORE_MAX = 35.0

VLM_MAX_ATTEMPTS = 1
"""Outer retry attempts for the VLM call. 1 (off) because the underlying
``OpenAICompatibleClient`` already retries transport-level failures."""


@register_judge(
    NAME, requires=('vlm',), capabilities=('screenshot',), model_files={'vlm': 'gemini-3.8-flash-aihub-2.0.yaml'}
)
async def compute_score(inp: JudgeInput) -> JudgeResult:
    """Return the normalized aesthetic score and raw model feedback."""
    shots = inp.sandbox_ctx.screenshots.for_vlm() if inp.sandbox_ctx else []
    if not shots:
        return JudgeResult(
            judge_name=NAME,
            score=0.0,
            error='no screenshot available for VLM scoring',
            failure_kind='no_sandbox',
        )

    prompt_text = render_prompt(load_prompt('judge_aesthetic.txt'), query=inp.problem_statement)

    # call_judge_llm requires the explicit vlm role, retries, and raises
    # JudgeExecutionError('vlm_failed') on failure — the wrapper maps it.
    llm_output = await call_judge_llm(
        inp,
        role='vlm',
        prompt=prompt_text,
        images=shots,
        max_attempts=VLM_MAX_ATTEMPTS,
        label=NAME,
    )

    logger.debug('aesthetic: VLM raw output:\n%s', llm_output[:1000])

    normalised, evidence = sum_tag_scores(llm_output, SCORE_TAGS, SCORE_MAX, weights=SCORE_WEIGHTS)

    return JudgeResult(
        judge_name=NAME,
        score=round(normalised, 4),
        passed=None,
        evidence=evidence,
        raw_output=llm_output,
        metadata={
            'vlm_normalized': normalised,
            'vlm_raw_output': llm_output,
            'score_tags': list(SCORE_TAGS),
            'score_weights': list(SCORE_WEIGHTS),
            'score_max': SCORE_MAX,
        },
    )


__all__ = ['compute_score']
