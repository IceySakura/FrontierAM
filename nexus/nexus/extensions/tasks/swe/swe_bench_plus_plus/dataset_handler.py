"""
Dataset handler for SWE-Bench-plus-plus dataset.
"""

from typing import Any

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_bench_plus_plus.constants import (
    get_ext_from_language,
)
from nexus.extensions.tasks.swe.swe_bench_plus_plus.grading import get_eval_report
from nexus.extensions.tasks.swe.swe_bench_plus_plus.test_spec import make_test_spec
from nexus.runtime import Runtime


def get_eval_script(instance: SWEbenchInstance) -> str:
    """
    Generate the evaluation script for the instance.
    """
    test_spec = make_test_spec(instance)
    return test_spec.eval_script


def get_eval_report_map(
    instance: SWEbenchInstance, prediction: dict[str, str], test_log_path: str, include_tests_status: bool
) -> dict[str, Any]:
    """
    Get the evaluation report map.
    """
    test_spec = make_test_spec(instance)
    return get_eval_report(test_spec, prediction, test_log_path, include_tests_status)


def get_problem_statement(instance: SWEbenchInstance) -> str:
    """
    Get the problem statement for the instance.
    """
    return instance.get('problem_statement') or instance.get('problem') or ''


def get_repo_language(instance: SWEbenchInstance) -> str | None:
    """
    Get the repository language.
    """
    language = instance.get('language')
    if language:
        return get_ext_from_language(language)
    return None


async def infer_runtime_init(
    instance: SWEbenchInstance, runtime: Runtime, *, skip_git_truncate: bool = False, **kwargs: Any
) -> None:
    """Initialize runtime for SWE-Bench-plus-plus inference.

    Reset repository to base_commit and clean up untracked files to ensure
    git diff only captures agent's changes.

    The images may have:
    1. Modified tracked files
    2. Extra untracked files (e.g., pip install artifacts like "=1.2.0")

    Both would pollute the model_patch if not handled.
    """

    base_commit = instance.get('base_commit', '')
    working_dir = instance.get('workdir', '/testbed')

    # 1. Reset tracked files to base_commit state
    if base_commit:
        reset_cmd = f'git reset --hard {base_commit}'
        await runtime.call_executor('run_command', {'command': reset_cmd, 'cwd': working_dir})

    # 2. Clean untracked files and directories to ensure clean git diff
    # -f: force, -d: directories, -x: also remove ignored files
    clean_cmd = 'git clean -fdx'
    await runtime.call_executor('run_command', {'command': clean_cmd, 'cwd': working_dir})

    # 3. Remove .git or truncate git history
    if kwargs.get('remove_git_dir', False):
        from nexus.extensions.tasks.swe.infer import remove_git_dir

        await remove_git_dir(runtime, working_dir)
        return

    # Remove all remote refs (including symbolic refs like origin/HEAD) before truncation
    # This prevents "fatal: bad object refs/remotes/origin/HEAD" error during git gc
    # when the underlying ref it points to gets deleted by git truncate
    # Also remove remote config to prevent git pull/fetch from re-downloading remote info
    remove_remotes_cmd = (
        'git remote remove origin 2>/dev/null || true && '
        'rm -rf .git/refs/remotes && '
        "sed -i '/refs\\/remotes\\//d' .git/packed-refs 2>/dev/null || true"
    )
    await runtime.call_executor('run_command', {'command': remove_remotes_cmd, 'cwd': working_dir})

    # 4. Truncate git history to prevent peeking at future changes
    from nexus.extensions.tasks.swe.infer import run_git_truncate

    if not skip_git_truncate:
        await run_git_truncate(runtime, base_commit, working_dir)


async def eval_runtime_init(instance: SWEbenchInstance, runtime: Runtime) -> None:
    """Initialize runtime for SWE-Bench-plus-plus evaluation.

    No special initialization needed for evaluation.
    """
    return
