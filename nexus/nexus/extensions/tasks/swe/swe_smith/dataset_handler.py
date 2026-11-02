import logging

from typing import Any

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.infer import apply_patch_and_commit
from nexus.extensions.tasks.swe.swe_smith.grading import get_eval_report
from nexus.extensions.tasks.swe.swe_smith.test_spec import make_test_spec
from nexus.runtime import Runtime


logger = logging.getLogger(__name__)


def get_eval_script(instance: SWEbenchInstance) -> str:
    """Generate eval script for a SWE-Smith instance."""
    test_spec = make_test_spec(instance)
    return test_spec.eval_script


def get_eval_report_map(
    instance: SWEbenchInstance,
    prediction: dict[str, str],
    test_log_path: str,
    include_tests_status: bool,
) -> dict[str, Any]:
    """Get evaluation report for a SWE-Smith instance."""
    test_spec = make_test_spec(instance)
    return get_eval_report(test_spec, prediction, test_log_path, include_tests_status)


def get_problem_statement(instance: SWEbenchInstance) -> str:
    return instance.get('problem_statement') or instance.get('problem') or ''


def get_repo_language(instance: SWEbenchInstance) -> str | None:
    """Get the primary language for a repository from its profile."""
    from nexus.extensions.tasks.swe.swe_smith.test_spec import (
        get_repo_language as _get_repo_language,
    )

    return _get_repo_language(instance)


async def infer_runtime_init(instance: SWEbenchInstance, runtime: Runtime, **kwargs: Any) -> None:
    """Initialize runtime for SWE-Smith inference.

    SWE-Smith images contain correct code, so we need to apply the bug patch
    to restore the buggy state before inference and evaluation.

    This also includes git history sanitization to prevent models from hacking
    via git diff/log/reflog. The sanitization creates an orphan branch with the
    buggy state as the only commit.

    Note: This function records the new sanitized commit in instance['runtime_base_commit'], which the task uses as the diff base for patch extraction. We must NOT write it to instance['base_commit']: the eval runtime is a fresh sandbox without that orphan commit, and base_commit drives a `git checkout base_commit` there (task.py), which would fail. Using runtime_base_commit also makes patch extraction robust to the agent committing during rollout (a fixed base diff captures all changes, unlike `git diff HEAD`).

    Args:
        instance: The SWE-Smith instance. Will be modified in place.
        runtime: The runtime to initialize.
    """
    bug_patch = instance.get('bug_patch', '')
    if not bug_patch:
        return

    working_dir = instance.get('workdir', '/testbed')

    # Apply only (no commit): the git history sanitization below makes its own commit.
    log_output = await apply_patch_and_commit(runtime, bug_patch, '/tmp/bug_patch.diff', working_dir)
    logger.debug('SWE-Smith init commands executed:\n%s', log_output)

    if kwargs.get('remove_git_dir', False):
        from nexus.extensions.tasks.swe.infer import remove_git_dir

        await remove_git_dir(runtime, working_dir)
        return

    # Extract the new sanitized commit from the last line (output of git rev-parse HEAD) and record it as the patch-extraction diff base.
    from nexus.extensions.tasks.swe.infer import run_git_reinit

    runtime_base_commit = await run_git_reinit(runtime, working_dir)
    instance['runtime_base_commit'] = runtime_base_commit
    logger.debug('SWE-Smith updated runtime_base_commit: %s', runtime_base_commit)


async def eval_runtime_init(instance: SWEbenchInstance, runtime: Runtime) -> None:
    """Initialize runtime for SWE-Smith evaluation.

    SWE-Smith images contain correct code, so we need to apply the bug patch
    to restore the buggy state before evaluation. The model's patch will then
    be applied on top of this buggy state.

    Note: Unlike infer_runtime_init, we don't sanitize git history here since
    the model has already been invoked and we just need the buggy state for testing.

    Args:
        instance: The SWE-Smith instance.
        runtime: The runtime to initialize.
    """
    bug_patch = instance.get('bug_patch', '')
    if not bug_patch:
        return

    working_dir = instance.get('workdir', '/testbed')

    # Apply only (no commit): the model patch is applied on top of this buggy state by the framework.
    log_output = await apply_patch_and_commit(runtime, bug_patch, '/tmp/bug_patch.diff', working_dir)
    logger.debug('SWE-Smith eval init commands executed:\n%s', log_output)
