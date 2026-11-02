"""Dataset handler for SWE-Env instances.

Reuses the swe_task orchestration (sandbox lifecycle, model-patch apply, scoring) and ports only SWE-Env's evaluation core.

For inference:
    1. reset --hard repo_dict.base_commit
    2. clean untracked files
    3. prevent git future hacking

For evaluation:
    1. apply the model/oracle patch
    2. write the eval script to /tmp/eval.sh
    3. run the eval script and extract the exit code
"""

import logging

from typing import Any

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.infer import (
    run_git_truncate,
    run_init_cmds_in_runtime,
)
from nexus.extensions.tasks.swe.swe_env.utils import (
    TEST_OUTPUT_END,
    TEST_OUTPUT_START,
    _EVAL_SCRIPT_PATH,
    get_eval_report,
    language_to_ext,
    postprocess_eval_script,
)
from nexus.runtime import Runtime


logger = logging.getLogger(__name__)


# =============================================================================
# Runtime setup helpers
# =============================================================================


async def _reset_and_clean(runtime: Runtime, instance: SWEbenchInstance) -> None:
    """Reset and clean the repo to the base commit.

    Reset to base_commit -> clean untracked files.
    """
    working_dir = instance.get('workdir', '/testbed')
    base_commit = instance.get('base_commit', 'HEAD')
    reset_cmd = f'git reset --hard {base_commit}'
    setup_cmds = [
        f'git config --global --add safe.directory {working_dir}',
        "git config --local user.name 'swe-env'",
        "git config --local user.email 'swe-env@anonymous.com'",
        'git config --local commit.gpgsign false',
        f'{reset_cmd} && git clean -fd',
    ]
    setup_log = await run_init_cmds_in_runtime(runtime, setup_cmds, working_dir=working_dir)
    logger.debug('SWE-Env init commands executed:\n%s', setup_log)


# =============================================================================
# DatasetHandler interface
# =============================================================================


def get_problem_statement(instance: SWEbenchInstance) -> str:
    return instance['problem_statement']


def get_repo_language(instance: SWEbenchInstance) -> str | None:
    language = instance.get('language') or instance.get('repo_dict', {}).get('language') or 'python'
    return language_to_ext(language)


async def infer_runtime_init(instance: SWEbenchInstance, runtime: Runtime, **kwargs: Any) -> None:
    """Reset and clean the repo to the base commit, and truncate git history to prevent peeking at future changes."""
    working_dir = instance.get('workdir', '/testbed')
    base_commit = instance.get('base_commit', 'HEAD')

    # 1. Reset and clean the repo to the base commit.
    await _reset_and_clean(runtime, instance)

    if kwargs.get('remove_git_dir', False):
        from nexus.extensions.tasks.swe.infer import remove_git_dir

        await remove_git_dir(runtime, working_dir)
        return

    # 2. Drop every configured remote, then purge any leftover remote-tracking refs.
    # sed uses '\#...#' so its regex delimiter isn't '/', avoiding slash escaping.
    remove_remotes_cmds = [
        'git remote | while read -r r; do git remote remove "$r"; done 2>/dev/null || true',
        'rm -rf .git/refs/remotes',
        "sed -i '\\#refs/remotes/#d' .git/packed-refs 2>/dev/null || true",
    ]
    await run_init_cmds_in_runtime(runtime, remove_remotes_cmds, working_dir=working_dir)

    # 3. Truncate git history to prevent peeking at future changes
    await run_git_truncate(runtime, base_commit, working_dir)
    logger.debug('SWE-Env inference init finished')
    return


async def eval_runtime_init(instance: SWEbenchInstance, runtime: Runtime) -> None:
    """Reset and clean the repo to the base commit."""
    await _reset_and_clean(runtime, instance)


def get_eval_script(instance: SWEbenchInstance) -> str:
    """Compose the script the runtime runs after the model patch is applied."""
    # Materialize test_function into EVAL_SCRIPT_PATH (test_cmd runs `bash` on it).
    # The single-quoted heredoc delimiter disables all expansion, so the (multi-line, special-char) body is written verbatim. We write it before TEST_OUTPUT_START so `set -x` tracing of the heredoc stays out of the parsed test-output region.
    eval_script = postprocess_eval_script(instance.get('eval_script', ''))
    write_eval_script_cmd = '\n'.join([
        f"cat > {_EVAL_SCRIPT_PATH} << 'NEXUS_EVAL_SCRIPT_EOF'",
        eval_script,
        'NEXUS_EVAL_SCRIPT_EOF',
    ])
    return '\n'.join([
        '#!/bin/bash',
        write_eval_script_cmd,
        'set -x',
        f": '{TEST_OUTPUT_START}'",
        f'bash {_EVAL_SCRIPT_PATH}',
        'echo ">>>>>>>>>> Evaluation Exit Code: $? <<<<<<<<<<"',
        f": '{TEST_OUTPUT_END}'",
    ])


def get_eval_report_map(
    instance: SWEbenchInstance,
    prediction: dict[str, str],
    test_log_path: str,
    include_tests_status: bool,
) -> dict[str, Any]:
    """Grade the run and shape it as {instance_id: {resolved, tests_status, ...}}."""
    instance_id = prediction.get('instance_id', instance.get('instance_id', ''))
    report_map: dict[str, Any] = {
        instance_id: {
            'patch_is_None': prediction.get('model_patch') is None,
            'patch_exists': bool(prediction.get('model_patch')),
            'patch_successfully_applied': True,
            'resolved': False,
        }
    }

    report = get_eval_report(test_log_path)
    report_map[instance_id]['resolved'] = bool(report.get('resolved', False))
    report_map[instance_id]['status'] = report.get('status', 'unresolved')
    if include_tests_status:
        report_map[instance_id]['tests_status'] = report.get('tests_status', {})
    return report_map
