"""Review artifact source without deployment or browser tools."""

from __future__ import annotations

import logging

from typing import Any

from nexus.extensions.tasks.artifact.agent_setup.claude_config import JUDGE_ROOT, default_code_review_claude_input
from nexus.extensions.tasks.artifact.judges.base import JudgeInput, JudgeResult
from nexus.extensions.tasks.artifact.judges.capabilities.agent import call_judge_agent
from nexus.extensions.tasks.artifact.judges.capabilities.deploy import upload_archive_to_runtime
from nexus.extensions.tasks.artifact.judges.capabilities.parse import sum_tag_scores
from nexus.extensions.tasks.artifact.judges.capabilities.prompts import load_prompt, render_prompt
from nexus.extensions.tasks.artifact.judges.registry import register_judge


logger = logging.getLogger(__name__)

NAME = 'code_quality'

REWARD_FILE = f'{JUDGE_ROOT}/reward_code_quality.txt'
# prompt 出 <implementation> + <engineering>，2:1 加权归一到 SCORE_MAX。
SCORE_TAGS = ('implementation', 'engineering')
SCORE_WEIGHTS = (2.0, 1.0)
SCORE_MAX = 12.0


@register_judge(NAME, requires=('agent',), model_files={'agent': 'deepseek-flash-aihub-2.0.yaml'})
async def compute_score(inp: JudgeInput) -> JudgeResult:
    """Run a source-review agent and normalize its weighted score tags."""
    if not inp.archive_b64:
        return JudgeResult(
            judge_name=NAME,
            score=0.0,
            evidence='no artifact to review',
            error='code_quality requires JudgeInput.archive_b64',
            failure_kind='no_artifact',
        )

    # 默认不装 playwright MCP —— 读源码不需要浏览器，装它只多 30 秒握手和一堆用不上的工具。
    claude_input = default_code_review_claude_input()
    agent_working_dir = claude_input.working_dir

    task_prompt = render_prompt(
        load_prompt('judge_static_code_review.txt'),
        query=inp.problem_statement,
        working_dir=agent_working_dir,
        reward_file=REWARD_FILE,
    )

    async def _upload_code(runtime: Any) -> None:
        _count, upload_err = await upload_archive_to_runtime(runtime, inp.archive_b64, agent_working_dir)
        if upload_err:
            raise RuntimeError(f'code upload failed: {upload_err}')

    # call_judge_agent resolves the client, runs the agent on inp.runtime, reads
    # the reward file, and raises JudgeExecutionError(agent_failed/parse_failed)
    # on failure — the wrapper maps it.
    run = await call_judge_agent(
        inp,
        prompt=task_prompt,
        reward_file=REWARD_FILE,
        client_role='agent',
        with_playwright=False,
        with_skill=False,
        pre_run_setup=_upload_code,
    )

    cc = run.cc
    metadata: dict[str, Any] = {'trajectories': cc.trajectories, 'execution_time': cc.execution_time}

    try:
        score, evidence = sum_tag_scores(run.raw, SCORE_TAGS, SCORE_MAX, weights=SCORE_WEIGHTS)
    except Exception as exc:  # noqa: BLE001
        logger.error('code_quality: failed to parse result: %s', exc)
        return JudgeResult(
            judge_name=NAME,
            score=0.0,
            error=f'parse failed: {exc}',
            failure_kind='parse_failed',
            raw_output=cc.output,
            metadata=metadata,
        )

    return JudgeResult(
        judge_name=NAME,
        score=round(score, 4),
        passed=None,
        evidence=evidence,
        raw_output=cc.output,
        metadata=metadata,
    )


__all__ = ['compute_score']
