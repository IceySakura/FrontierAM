"""
Dataset handler for SWE-Bench-Multilingual dataset.
"""

from typing import Any

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_bench_multilingual.grading import get_eval_report
from nexus.extensions.tasks.swe.swe_bench_multilingual.test_spec import make_test_spec
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
    return instance.get('problem_statement') or ''


def get_repo_language(instance: SWEbenchInstance) -> str | None:
    """
    Get the repository language.
    Infer from 'repo' field using MAP_REPO_TO_LANGUAGE
    """
    repo = instance.get('repo')
    if repo:
        from nexus.extensions.tasks.swe.swe_bench_multilingual.constants import get_language_from_repo

        return get_language_from_repo(repo)

    return None


async def infer_runtime_init(
    instance: SWEbenchInstance, runtime: Runtime, *, skip_git_truncate: bool = False, **kwargs: Any
) -> None:
    """Initialize runtime for SWE-Bench-Multilingual inference."""
    from nexus.extensions.tasks.swe.infer import run_git_truncate

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

    if not skip_git_truncate:
        await run_git_truncate(runtime, base_commit, working_dir)


async def eval_runtime_init(instance: SWEbenchInstance, runtime: Runtime) -> None:
    """Initialize runtime for SWE-Bench-Multilingual evaluation.

    No special initialization needed for evaluation.
    """
    return
