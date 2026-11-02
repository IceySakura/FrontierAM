"""Per-turn cleanup strategy for the build_cc (Claude Code + Docker-in-Docker) scenario.

Runs after each verifier turn (inline mode) via code_mt_task's
eval_cleanup_strategy hook (task.py:_run_eval_cleanup, called in the per-turn
`finally`). Signature mirrors the builder cleanup (keyword-only ctx/runtime/
task_input/eval_result) so the task can drive it identically.

build_cc is fundamentally different from builder/tb/swe: the agent is
`claude_code`, the runtime has Docker-in-Docker enabled, and the graded artifact
is an INNER docker image (by convention `cc_built:latest`) that Claude Code
builds inside the pod's dockerd. The judge (tests/test.sh) `docker image
inspect`s that image in the same dockerd.

Between MT turns we remove the scratch container and prune dangling layers, but
we intentionally KEEP the built image (cc_built:latest) so turn N+1 inherits
turn N's build progress. Heavy repos cannot finish a fresh build inside a single
turn; wiping the image every turn made the grader see no_built_image. The
`docker` CLI here talks to the SAME pod-local dockerd that the agent used,
because DinD runs one dockerd per pod (gongfeng injects it at provider level;
see gongfeng.py enable_dind).

This module is referenced ONLY by build_cc task.toml files
(eval_cleanup_strategy=...dind_cleanup::cleanup). Every other code_mt subset
leaves eval_cleanup_strategy empty, so task.py short-circuits before importing
this — zero impact on builder/tb/swe.
"""

from __future__ import annotations

import logging

from typing import Any


logger = logging.getLogger(__name__)


# Default name of the inner image the agent is asked to produce. The judge reads
# the same default (CC_BUILT_IMAGE env / .cc_eval_target file / this literal).
DEFAULT_BUILT_IMAGE = 'cc_built:latest'
DEFAULT_ENV_CONTAINER = 'cc_env'


async def cleanup(
    *,
    ctx: Any,
    runtime: Any,
    task_input: Any,
    eval_result: dict[str, Any],
) -> None:
    """Remove scratch container + prune dangling layers between turns; KEEP built image.

    Best-effort: every docker command is guarded so a missing image/container or
    a transient dockerd hiccup can never abort the MT loop. task.py already wraps
    this call in `contextlib.suppress(Exception)`, but we keep it self-contained.
    """
    # Backfill feedback-template fields so the next turn's mt_feedback.md render
    # never KeyErrors when reward.json was missing or the judge crashed.
    if 'stage' not in eval_result:
        eval_result['stage'] = 'reward_missing' if not eval_result.get('test_output') else 'judge_crashed'
    eval_result.setdefault('built_image', _built_image(ctx))
    eval_result.setdefault('error', '')

    img = _built_image(ctx)
    container = _env_container(ctx)

    # KEEP the built image (cc_built:latest) across turns so turn N+1 inherits
    # turn N's build progress — heavy repos can't finish a fresh build in one
    # turn, and wiping it each turn caused no_built_image at grading. Only remove
    # the scratch container and prune dangling layers to reclaim space.
    cmd = (
        f'docker rm -f {container} 2>/dev/null; '
        'docker container prune -f 2>/dev/null; '
        'docker image prune -f 2>/dev/null; '
        'true'
    )
    try:
        await runtime.run_command(command=cmd, timeout=120)
    except Exception as e:  # noqa: BLE001 - cleanup must never abort the loop
        logger.warning('[dind_cleanup] docker wipe failed (non-fatal): %s', e)

    logger.debug(
        '[dind_cleanup] wiped inner docker (image=%s, container=%s, stage=%s, attempt=%s)',
        img,
        container,
        eval_result.get('stage', '?'),
        eval_result.get('attempt', '?'),
    )


def _built_image(ctx: Any) -> str:
    tc = getattr(ctx, 'task_config', {}) or {}
    return tc.get('cc_built_image') or tc.get('extra', {}).get('cc_built_image') or DEFAULT_BUILT_IMAGE


def _env_container(ctx: Any) -> str:
    tc = getattr(ctx, 'task_config', {}) or {}
    return tc.get('cc_env_container') or tc.get('extra', {}).get('cc_env_container') or DEFAULT_ENV_CONTAINER
