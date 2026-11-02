"""Mock hack agent that verifies git history truncation worked correctly.

Runs inside the runtime after infer_runtime_init to check that:
1. `git log --all --oneline` shows base_commit as the latest (first) entry
2. No future commits leaked past the truncation boundary
"""

import logging

from typing import cast

from nexus.core import agent_registry, require_context
from nexus.extensions.agents.swe_agent.agent import ExitStatus, SWEAgentInput, SWEAgentOutput
from nexus.extensions.tasks.swe.constants import SWEbenchInstance


logger = logging.getLogger(__name__)


@agent_registry.register('mock_hack_agent')
async def run_mock_hack_agent(agent_input: SWEAgentInput) -> SWEAgentOutput:
    """Verify git history truncation instead of solving the problem."""
    del agent_input
    ctx = require_context()
    runtime = await ctx.get_runtime()
    instance = cast(SWEbenchInstance, ctx.task_config.get('instance'))

    base_commit = instance.get('base_commit', '')
    if not base_commit:
        base_commit = instance.get('base', {}).get('sha', '')
    working_dir = instance.get('workdir', '/testbed')

    if not base_commit:
        raise RuntimeError('No base_commit available, cannot verify truncation')

    # Check 1: git log --all --oneline — first line should be base_commit
    log_resp = await runtime.run_command('git log --all --oneline', cwd=working_dir, timeout=60)
    log_output = (log_resp.stdout or '').strip()
    lines = [line for line in log_output.splitlines() if line.strip()]

    if not lines:
        raise RuntimeError(f'git log --all --oneline returned no output (stderr: {log_resp.stderr})')

    first_sha = lines[0].split()[0]
    if not base_commit.startswith(first_sha):
        raise RuntimeError(
            f'Git truncation check FAILED: latest commit {first_sha} does not match base_commit {base_commit[:12]}'
        )

    # Check 2: no future commits leaked — every SHA in log should be an ancestor of base_commit
    rev_resp = await runtime.run_command(f'git rev-list {base_commit}', cwd=working_dir, timeout=60)
    ancestor_shas = set((rev_resp.stdout or '').strip().splitlines())

    leaked = []
    for line in lines:
        sha_short = line.split()[0]
        if not any(full.startswith(sha_short) for full in ancestor_shas):
            leaked.append(line)

    if leaked:
        raise RuntimeError(
            f'Git truncation check FAILED: {len(leaked)} commits not in base_commit ancestry: {leaked[:10]}'
        )

    logger.info(
        'Git truncation check PASSED: latest=%s, total_commits=%d, ancestors=%d',
        first_sha,
        len(lines),
        len(ancestor_shas),
    )
    return SWEAgentOutput(exit_status=ExitStatus.FINISHED)
