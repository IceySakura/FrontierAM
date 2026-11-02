"""Dataset handler for SWE-Cosmos instances.

Reuses the swe_task orchestration (sandbox lifecycle, model-patch apply, scoring) and ports only SWE-Cosmos's evaluation core. The buggy task state is reconstructed inside the runtime by applying, each followed by its own commit:

    reset --hard repo_dict.base_commit
      -> init_patch (repo_dict.patch)
      -> init_test_patch (conditional on apply_init_test_patch)
      -> bug patch (instance.patch)
      -> instance test_patch (optional)

For inference we then drop git history via GIT_REINIT_SCRIPT (a fresh single-commit repo), so the agent cannot recover the fix by walking back revisions. For evaluation we keep history, record the bug commit (captured before the instance test_patch) and the test files to revert, and fold the post-model-patch test-file revert into the eval script (so it runs after the model patch the framework applies).
"""

import logging

from typing import Any, cast

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.infer import (
    apply_patch_and_commit,
    get_head_commit,
    run_git_reinit,
    run_init_cmds_in_runtime,
)
from nexus.extensions.tasks.swe.swe_cosmos.profiles import RepoProfile, registry
from nexus.extensions.tasks.swe.swe_cosmos.utils import (
    _BUG_COMMIT_KEY,
    _REVERT_FILES_KEY,
    get_eval_report,
    get_patch_set,
    language_to_ext,
)
from nexus.runtime import Runtime


logger = logging.getLogger(__name__)


# =============================================================================
# Profile / repo_dict helpers
# =============================================================================


def _build_profile(instance: SWEbenchInstance) -> RepoProfile:
    """Each SWE-Cosmos row carries the original reference instance under repo_dict."""
    repo_dict = instance.get('repo_dict')
    if not repo_dict:
        raise ValueError(f'SWE-Cosmos instance {instance.get("instance_id")!r} is missing the repo_dict field')

    return registry.get(repo_dict)


# =============================================================================
# Runtime setup helpers
# =============================================================================


async def _setup_to_bug(runtime: Runtime, profile: RepoProfile, instance: SWEbenchInstance) -> None:
    """Clean the repo and reconstruct the buggy task state up to (and including) the bug patch.

    Stages: reset to base_commit -> init_patch -> init_test_patch (conditional) -> bug patch, each committed.

    The instance test_patch is intentionally NOT applied here: evaluation must capture the bug commit before it (so test files can be reverted to that state), and the caller applies it as a separate stage.
    """
    working_dir = profile.workdir
    reset_cmd = 'git reset --hard'
    if profile.commit:
        reset_cmd += f' {profile.commit}'
    setup_cmds = [
        f'git config --global --add safe.directory {profile.workdir}',
        "git config --local user.name 'swe-cosmos'",
        "git config --local user.email 'swe-cosmos@anonymous.com'",
        'git config --local commit.gpgsign false',
        f'{reset_cmd} && git clean -fd',
    ]
    setup_log = await run_init_cmds_in_runtime(runtime, setup_cmds, working_dir=working_dir)
    logger.debug('SWE-Cosmos setup commands executed:\n%s', setup_log)

    await apply_patch_and_commit(runtime, profile.init_patch, '/tmp/init_patch.diff', working_dir, 'Apply init patch')
    if profile.apply_init_test_patch:
        await apply_patch_and_commit(
            runtime, profile.init_test_patch, '/tmp/init_test_patch.diff', working_dir, 'Apply init test patch'
        )
    await apply_patch_and_commit(runtime, instance.get('patch'), '/tmp/bug_patch.diff', working_dir, 'Apply bug patch')


# =============================================================================
# DatasetHandler interface
# =============================================================================


def get_problem_statement(instance: SWEbenchInstance) -> str:
    return instance['problem_statement']


def get_repo_language(instance: SWEbenchInstance) -> str | None:
    language = instance.get('language') or instance.get('repo_dict', {}).get('language') or 'python'
    return language_to_ext(language)


async def infer_runtime_init(instance: SWEbenchInstance, runtime: Runtime, **kwargs: Any) -> None:
    """Reconstruct the buggy task state, then drop git history via GIT_REINIT_SCRIPT.

    History is dropped (not truncated) because SWE-Cosmos bugs are synthesized on top of a reference commit: leaving any prior history would let the agent recover the fix by walking back a few revisions.

    The new base is the GIT_REINIT_SCRIPT "Initial commit". We record its SHA in instance['runtime_base_commit'] so patch extraction diffs against it rather than HEAD — that keeps the extracted patch correct even if the agent commits during rollout (HEAD would then have moved past the buggy state).
    """
    profile = _build_profile(instance)
    working_dir = profile.workdir

    await _setup_to_bug(runtime, profile, instance)
    await apply_patch_and_commit(
        runtime, instance.get('test_patch'), '/tmp/inst_test_patch.diff', working_dir, 'Apply instance test patch'
    )

    if kwargs.get('remove_git_dir', False):
        from nexus.extensions.tasks.swe.infer import remove_git_dir

        await remove_git_dir(runtime, working_dir)
        return

    # Drop git history and record the fresh single-commit SHA as the patch-extraction diff base.
    runtime_base_commit = await run_git_reinit(runtime, working_dir)
    instance['runtime_base_commit'] = runtime_base_commit
    logger.debug('SWE-Cosmos updated runtime_base_commit: %s', runtime_base_commit)


async def eval_runtime_init(instance: SWEbenchInstance, runtime: Runtime) -> None:
    """Reconstruct the buggy task state and record what get_eval_script needs.

    Records the bug commit SHA and the test files to revert. The model patch is
    applied by the framework afterwards; get_eval_script reverts those test files
    back to the bug commit so the model cannot game grading by editing tests.
    """
    profile = _build_profile(instance)
    working_dir = profile.workdir

    # Reconstruct the buggy task state up to (and including) the bug patch.
    await _setup_to_bug(runtime, profile, instance)

    # Capture the bug commit BEFORE the instance test_patch via a dedicated git rev-parse, so the eval script can revert test files back to the bug state (discarding any model edits to them).
    bug_commit = await get_head_commit(runtime, working_dir)

    # instance.test_patch carries the F2P test edits; SWE-Cosmos commits them as "Remove F2P Tests" so the revert set below can restore the canonical tests.
    removed_test_files: list[str] = []
    inst_test_patch = instance.get('test_patch')
    if inst_test_patch:
        await apply_patch_and_commit(
            runtime, inst_test_patch, '/tmp/inst_test_patch.diff', working_dir, 'Remove F2P Tests'
        )
        removed_test_files = [pf.path for pf in get_patch_set(inst_test_patch)]

    revert_files = sorted(set[str](removed_test_files))
    # Stash private keys for get_eval_script. They are not part of the SWEbenchInstance schema, so cast to a plain dict to write them (TypedDict is closed under ty).
    instance_extra = cast(dict[str, Any], instance)
    instance_extra[_BUG_COMMIT_KEY] = bug_commit
    instance_extra[_REVERT_FILES_KEY] = revert_files


def get_eval_script(instance: SWEbenchInstance) -> str:
    """Compose the script the runtime runs after the model patch is applied."""
    profile = _build_profile(instance)
    return profile.get_eval_script(cast(dict[str, Any], instance))


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

    profile = _build_profile(instance)
    report = get_eval_report(cast(dict[str, Any], instance), profile, test_log_path)
    report_map[instance_id]['resolved'] = bool(report.get('resolved', False))
    report_map[instance_id]['status'] = report.get('status', 'unresolved')
    if include_tests_status:
        report_map[instance_id]['tests_status'] = report.get('tests_status', {})
    return report_map
