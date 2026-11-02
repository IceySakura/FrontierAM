"""Builder per-turn cleanup strategy for code_mt_task.

Runs after each verifier turn (inline mode), before the next MT attempt.

Two guarantees we owe the next turn:
  1. `/testbed` is back at `base_commit` — the judge applies `golden.patch` to
     decide PASS/FAIL, and if the judge crashed mid-flow that patch may still
     be applied to the working tree.
  2. `/tmp/golden_patch.patch` and `/tmp/test_patch.patch` exist and match the
     dataset (the agent may have deleted, truncated, or rewritten them — and
     the eval.sh template depends on knowing the canonical content even though
     the agent is forbidden from `git apply`-ing them).

We intentionally do not touch `/tmp/eval.sh` — the agent may want to iterate
on its own script across turns. The verifier reads it as input, never owns it.
"""

from __future__ import annotations

import base64
import logging

from typing import Any


logger = logging.getLogger(__name__)


_WORKDIR = '/testbed'


async def builder_cleanup(
    *,
    ctx: Any,
    runtime: Any,
    task_input: Any,
    eval_result: dict[str, Any],
) -> None:
    """Reset /testbed + rewrite /tmp/*.patch helpers for the next turn.

    Also backfills builder-specific eval_result fields the dataset's
    mt_feedback.md references — when reward.json is missing or the grader
    crashed mid-flow, the next turn's feedback template still needs these
    keys present with useful diagnostic values.
    """
    # 0. Backfill builder-template fields referenced by mt_feedback.md.
    if 'stage' not in eval_result:
        eval_result['stage'] = 'reward_missing' if not eval_result.get('test_output') else 'judge_crashed'
    eval_result.setdefault('base_output_tail', eval_result.get('test_output', '')[-2000:])
    eval_result.setdefault('eval_script_content', '')
    eval_result.setdefault('anti_hack_reason', '')
    eval_result.setdefault('error', '')

    # 1. Restore /testbed to base_commit (judge may have applied golden_patch).
    workdir = task_input.workdir or _WORKDIR
    await runtime.run_command(
        command=(f'cd {workdir} && git checkout -- . 2>/dev/null; git clean -fd 2>/dev/null; true'),
        timeout=60,
    )

    # 2. Re-materialize /tmp/golden_patch.patch and /tmp/test_patch.patch from
    #    task_config. We pull from `ctx.task_config` (which merges task_input
    #    + task_input.extra at task entry) so this works regardless of where
    #    the dataset put the patch payloads.
    tc = ctx.task_config
    golden = tc.get('golden_patch') or tc.get('extra', {}).get('golden_patch', '')
    test = tc.get('test_patch') or tc.get('extra', {}).get('test_patch', '')

    # Pipe the patch bytes via stdin (request body) rather than embedding them in
    # the `bash -c` argv: a large patch overflows execve ARG_MAX ("Argument list
    # too long") and the rewrite silently fails, leaving the next turn without a
    # canonical patch file. base64 keeps the piped payload newline/quote-safe.
    if golden and golden.strip():
        await runtime.run_command(
            command='base64 -d > /tmp/golden_patch.patch',
            stdin=base64.b64encode(golden.encode('utf-8')).decode('ascii'),
            timeout=30,
        )
    if test and test.strip():
        await runtime.run_command(
            command='base64 -d > /tmp/test_patch.patch',
            stdin=base64.b64encode(test.encode('utf-8')).decode('ascii'),
            timeout=30,
        )

    logger.debug(
        '[builder_cleanup] reset /testbed and rewrote /tmp/*.patch (stage=%s, attempt=%s)',
        eval_result.get('stage', '?'),
        eval_result.get('attempt', '?'),
    )
