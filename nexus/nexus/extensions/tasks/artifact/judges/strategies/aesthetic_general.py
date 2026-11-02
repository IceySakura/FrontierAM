"""Gate strict general-agent scoring on the aesthetic verdict.

Reuse prepared resources. A failed or zero aesthetic verdict skips the agent
without adding a missing score. Otherwise average both scores equally.
Propagate evaluator failures and retain both verdicts in child_results.
"""

from __future__ import annotations

import logging

from dataclasses import replace

from nexus.extensions.tasks.artifact.judges.base import JudgeInput, JudgeResult
from nexus.extensions.tasks.artifact.judges.registry import get_judge, register_judge


logger = logging.getLogger(__name__)

NAME = 'aesthetic_general'


@register_judge(
    NAME,
    requires=('agent', 'vlm'),
    capabilities=('screenshot',),
    model_files={'agent': 'deepseek-flash-aihub-2.0.yaml', 'vlm': 'gemini-3.8-flash-aihub-2.0.yaml'},
)
async def compute_score(inp: JudgeInput) -> JudgeResult:
    """Run the aesthetic gate, then strict general scoring if the gate passes."""
    sandbox_url = inp.sandbox_ctx.sandbox_url if inp.sandbox_ctx else None
    shots = inp.sandbox_ctx.screenshots.for_vlm() if inp.sandbox_ctx else []

    # Reuse the prepared sandbox and screenshots for child strategies.
    aesthetic_res = await get_judge('aesthetic').fn(inp)
    aesthetic_res.metadata.setdefault('sandbox_url', sandbox_url or '')
    if shots:
        aesthetic_res.metadata['candidate_screenshot_b64'] = shots[0]

    # Stop on an aesthetic failure or zero score.
    gate_failed = bool(aesthetic_res.failure_kind)  # no_sandbox / vlm_failed / 等
    gate_zero = (aesthetic_res.score or 0.0) <= 0.0
    if gate_failed or gate_zero:
        reason = aesthetic_res.failure_kind or 'aesthetic_score_zero'
        logger.info(
            '[%s] aesthetic gate 未通过（%s, score=%s）→ 跳过 general_agent',
            inp.instance_id[:12],
            reason,
            aesthetic_res.score,
        )
        return JudgeResult(
            judge_name=NAME,
            # Do not add a zero for the skipped general score.
            score=aesthetic_res.score,
            passed=aesthetic_res.passed,
            evidence=f'gate={reason} :: {aesthetic_res.evidence}'[:1000],
            # Propagate evaluator failures to the top-level verdict.
            error=aesthetic_res.error,
            failure_kind=aesthetic_res.failure_kind,
            child_results=[aesthetic_res],
            metadata={
                'general_skipped': reason,
                'sandbox_url': sandbox_url or '',
                'aesthetic': aesthetic_res.metadata,
            },
        )

    # Use strict general scoring in the same runtime.
    general_res = await get_judge('general_agent').fn(
        replace(inp, instance={**dict(inp.instance or {}), 'general_agent_lite': True})
    )
    general_res.metadata.setdefault('sandbox_url', sandbox_url or '')
    if shots:
        general_res.metadata.setdefault('candidate_screenshot_b64', shots[0])

    # Average the two completed scores equally.
    legs = [aesthetic_res, general_res]
    final = round(sum(r.score for r in legs) / len(legs), 4)
    passed_flags = [r.passed for r in legs if r.passed is not None]

    return JudgeResult(
        judge_name=NAME,
        score=final,
        passed=all(passed_flags) if passed_flags else None,
        evidence=f'aesthetic={aesthetic_res.score} general={general_res.score} → mean={final}',
        # Keep general-agent failures visible on the top-level verdict.
        error=general_res.error,
        failure_kind=general_res.failure_kind,
        child_results=legs,
        metadata={
            'sandbox_url': sandbox_url or '',
            'aesthetic': aesthetic_res.metadata,
            'general_agent': general_res.metadata,
        },
    )


__all__ = ['compute_score']
