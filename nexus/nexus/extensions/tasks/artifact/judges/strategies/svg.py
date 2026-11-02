"""Score SVG structure and aesthetics on [1, 5], normalized as (sum - 2) / 8.

Use screenshots prepared by the registry wrapper.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re

from typing import Any

from nexus.extensions.tasks.artifact.judges.base import JudgeInput, JudgeResult
from nexus.extensions.tasks.artifact.judges.capabilities.llm import call_judge_llm
from nexus.extensions.tasks.artifact.judges.capabilities.parse import coerce_score, loads_json_lenient
from nexus.extensions.tasks.artifact.judges.capabilities.prompts import load_prompt, render_prompt
from nexus.extensions.tasks.artifact.judges.registry import register_judge


logger = logging.getLogger(__name__)

NAME = 'svg'
SCORE_MIN = 1.0
SCORE_MAX = 5.0

VLM_MAX_ATTEMPTS = 1
"""Outer LLM-call attempts; scrutiny parsing runs after the call."""


def _extract_dim(value: Any) -> tuple[float | None, str]:
    """Return a dimension score and reason; invalid scores become None."""
    raw = value.get('score') if isinstance(value, dict) else value
    reason = str(value.get('reasoning', '') or '') if isinstance(value, dict) else ''
    return coerce_score(raw, low=SCORE_MIN, high=SCORE_MAX), reason


# Fallback brace spans containing BOTH keys, in either order — used only when
# the lenient outermost-brace slice fails (e.g. extra prose braces around the
# object). Keeps svg robust to messy VLM output without re-implementing fence logic.
_SCRUTINY_FALLBACK_RES = (
    re.compile(r'\{[\s\S]*"structure"[\s\S]*"aesthetics"[\s\S]*\}'),
    re.compile(r'\{[\s\S]*"aesthetics"[\s\S]*"structure"[\s\S]*\}'),
)


def _parse_scrutiny_json(text: str) -> tuple[float, float, str, str] | None:
    """Extract both scrutiny dimensions and reasons, or None for invalid scores.

    Allow fences and surrounding prose; both scores must lie in [1, 5].
    """
    data, _err = loads_json_lenient(text)
    if not (isinstance(data, dict) and 'structure' in data and 'aesthetics' in data):
        data = None
        for pattern in _SCRUTINY_FALLBACK_RES:
            m = pattern.search(text or '')
            if not m:
                continue
            with contextlib.suppress(json.JSONDecodeError):
                data = json.loads(m.group(0))
                break
    if not isinstance(data, dict) or 'structure' not in data or 'aesthetics' not in data:
        return None

    structure, struct_reason = _extract_dim(data.get('structure', {}))
    aesthetics, aesth_reason = _extract_dim(data.get('aesthetics', {}))
    if structure is None or aesthetics is None:
        # 两维都要有合法分数才能合成 —— 缺一维就不是一次有效评审。
        return None
    return structure, aesthetics, struct_reason, aesth_reason


@register_judge(
    NAME, requires=('vlm',), capabilities=('screenshot',), model_files={'vlm': 'gemini-3.8-flash-aihub-2.0.yaml'}
)
async def compute_score(inp: JudgeInput) -> JudgeResult:
    """Evaluate SVG screenshots and return the normalized scrutiny score."""
    shots = inp.sandbox_ctx.screenshots.for_vlm() if inp.sandbox_ctx else []
    if not shots:
        return JudgeResult(
            judge_name=NAME,
            score=0.0,
            error='no screenshot available for VLM scoring',
            failure_kind='no_sandbox',
        )

    prompt_text = render_prompt(load_prompt('judge_svg.txt'), query=inp.problem_statement)

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

    logger.debug('svg: VLM raw output:\n%s', llm_output[:1000])

    parsed = _parse_scrutiny_json(llm_output)
    if parsed is None:
        return JudgeResult(
            judge_name=NAME,
            score=0.0,
            error='Failed to parse scrutiny JSON',
            failure_kind='parse_failed',
            raw_output=llm_output,
            evidence='parse_failed',
        )

    structure, aesthetics, struct_reason, aesth_reason = parsed
    # Zero-based normalization: sum∈[2,10] → (sum-2)/8 ∈ [0,1].
    # 1+1 → 0 (rendered but bad), 5+5 → 1. A strict floor, matching webeval.
    raw_sum = structure + aesthetics  # 2..10
    normalised = round(max(0.0, min(1.0, (raw_sum - 2.0) / 8.0)), 4)

    evidence = (
        f'structure={structure:.1f}/5, aesthetics={aesthetics:.1f}/5, '
        f'sum={raw_sum:.1f}/10, zero_based_normalised={normalised}'
    )

    return JudgeResult(
        judge_name=NAME,
        score=normalised,
        passed=None,
        evidence=evidence,
        raw_output=llm_output,
        metadata={
            'structure': structure,
            'aesthetics': aesthetics,
            'structure_reasoning': struct_reason,
            'aesthetics_reasoning': aesth_reason,
            'raw_sum': raw_sum,
            'normalised': normalised,
        },
    )


__all__ = ['compute_score']
