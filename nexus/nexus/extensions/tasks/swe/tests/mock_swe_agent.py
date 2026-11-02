"""Mock SWE agent for test runs that applies the instance patch directly."""

import logging

from typing import cast

from nexus.core import agent_registry, require_context
from nexus.extensions.agents.swe_agent.agent import ExitStatus, SWEAgentInput, SWEAgentOutput
from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.dataset_handler import get_handler


logger = logging.getLogger(__name__)


def _write_and_apply_patch(patch_text: str, patch_file: str, working_dir: str) -> dict | None:
    """Write patch to file and apply it. Executed inside the runtime via invoke.

    This avoids ARG_MAX limits by using Python file I/O instead of shell heredoc.
    """
    import subprocess

    from pathlib import Path

    # Write patch to file using Python (no ARG_MAX limit)
    Path(patch_file).write_text(patch_text)

    # Apply patch with fallbacks
    apply_cmds = [
        f'git apply --verbose {patch_file}',
        f'git apply --verbose --reject {patch_file}',
        f'patch --batch --fuzz=5 -p1 -i {patch_file}',
    ]

    last_result = None
    for cmd in apply_cmds:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        last_result = {
            'command': cmd,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }
        if result.returncode == 0:
            break

    # Cleanup
    Path(patch_file).unlink(missing_ok=True)

    return last_result


@agent_registry.register('mock_swe_agent')
async def run_mock_swe_agent(agent_input: SWEAgentInput) -> SWEAgentOutput:
    """Apply the instance patch directly in the runtime and exit."""
    del agent_input
    ctx = require_context()
    runtime = await ctx.get_runtime()
    instance = cast(SWEbenchInstance, ctx.task_config.get('instance'))
    patch_text = instance.get('patch') or instance.get('fix_patch', '')
    working_dir = instance.get('workdir', '/testbed')

    # make sure the problem statement is handled correctly
    dataset_handler = get_handler(instance.get('dataset_type'))
    assert ctx.task_config.get('problem_statement') == dataset_handler.get_problem_statement(instance)

    patch_file = '/tmp/patch.diff'

    # Use invoke to avoid ARG_MAX limit (patch text sent via HTTP body, not shell args)
    result = await runtime.invoke(
        _write_and_apply_patch,
        patch_text,
        patch_file,
        working_dir,
    )

    logger.info(f'Patch apply result: {result}')

    # Check if patch was applied successfully
    if result and result.get('return_code') != 0:
        logger.warning('Mock agent patch apply failed: %s', result)
        return SWEAgentOutput(exit_status=ExitStatus.TOOL_ERROR)

    return SWEAgentOutput(exit_status=ExitStatus.FINISHED)
