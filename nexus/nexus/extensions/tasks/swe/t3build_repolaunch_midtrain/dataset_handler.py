import json
import re

from typing import Any, cast

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.t3build_repolaunch_midtrain.constants import (
    KEY_INSTANCE_ID,
    get_repo_ext,
)
from nexus.extensions.tasks.swe.t3build_repolaunch_midtrain.grading import get_eval_report
from nexus.extensions.tasks.swe.t3build_repolaunch_midtrain.test_spec import TestSpec, make_eval_script_list
from nexus.runtime import Runtime


def _has_new_file_in_patch(diff_text: str) -> bool:
    """Check if the patch contains new files"""
    chunks = re.split(r'^diff --git ', diff_text, flags=re.MULTILINE)
    for chunk in chunks:
        if not chunk.strip():
            continue
        chunk = 'diff --git ' + chunk
        is_new_file = re.search(r'^new file mode \d+', chunk, flags=re.MULTILINE)
        has_dev_null = re.search(r'^--- /dev/null$', chunk, flags=re.MULTILINE)
        has_added_file = re.search(r'^\+\+\+ b/.*', chunk, flags=re.MULTILINE)
        if (is_new_file or has_dev_null) and has_added_file:
            return True
    return False


def get_eval_script(instance: SWEbenchInstance) -> str:
    """
    Get the evaluation script for a given instance.

    Standard format:
    - test_cmd from spec_dict.test_cmd
    - log_parser_name from spec_dict.log_parser_name

    Args:
        instance (SWEbenchInstance): The instance to get the evaluation script for
    Returns:
        str: The evaluation script
    """
    base_commit = instance.get('base_commit', '')
    test_patch = instance.get('test_patch', '')

    env_name = 'testbed'
    repo_directory = f'/{env_name}'

    # Standard format: use spec_dict
    specs = instance.get('spec_dict')
    run_all_tests = bool(_has_new_file_in_patch(test_patch))

    eval_script_list = make_eval_script_list(
        instance, specs, env_name, repo_directory, base_commit, test_patch, run_all_tests
    )

    return '\n'.join(['#!/bin/bash', 'set -uxo pipefail'] + eval_script_list) + '\n'


def get_eval_report_map(
    instance: SWEbenchInstance, prediction: dict[str, str], test_log_path: str, include_tests_status: bool
) -> dict[str, Any]:
    """
    Get the evaluation report for a given instance
    """
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance['repo']
    version = instance.get('version', 'latest')

    def _from_json_or_obj(key: str) -> Any:
        """If key points to string, load with json"""
        instance_dict = cast(dict[str, Any], instance)
        if key not in instance_dict:
            return []
        if isinstance(instance_dict[key], str):
            return json.loads(instance_dict[key])
        return instance_dict[key]

    pass_to_pass = _from_json_or_obj('PASS_TO_PASS')
    fail_to_pass = _from_json_or_obj('FAIL_TO_PASS')

    test_spec = TestSpec(
        instance=instance,
        instance_id=instance_id,
        repo=repo,
        version=version,
        eval_script_list=[],
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language=get_repo_ext(instance),
    )

    return get_eval_report(test_spec, prediction, test_log_path, include_tests_status)


def get_problem_statement(instance: SWEbenchInstance) -> str:
    return instance.get('problem_statement') or instance.get('problem') or ''


def get_repo_language(instance: SWEbenchInstance) -> str | None:
    language = get_repo_ext(instance)
    if language == 'unknown':
        return None
    return language


async def infer_runtime_init(
    instance: SWEbenchInstance, runtime: Runtime, *, skip_git_truncate: bool = False, **kwargs: Any
) -> None:
    """t3build-repolaunch-midtrain images contain pre-configured environment.

    Reset repository to base_commit and clean up untracked files to ensure
    git diff only captures agent's changes.

    The t3build images may have:
    1. Modified tracked files (e.g., gold patch already applied)
    2. Extra untracked files (e.g., Django-1.4.1/, ez_setup.py)

    Both would pollute the model_patch if not handled.

    Args:
        instance: The SWE-bench instance being processed.
        runtime: The runtime environment to initialize.
        skip_git_truncate: If True, skip git history truncation.
    """
    from nexus.extensions.tasks.swe.infer import run_git_truncate

    working_dir = '/testbed'
    base_commit = instance.get('base_commit', '')

    # 0. Configure git safe.directory to avoid ownership issues in container
    # (container user may differ from /testbed directory owner)
    safe_dir_cmd = f'git config --global --add safe.directory {working_dir}'
    await runtime.call_executor('run_command', {'command': safe_dir_cmd, 'cwd': working_dir})

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
    # t3build images have remote refs that cause "bad object refs/remotes/origin/HEAD" error
    # during git gc, because the truncate script deletes the target ref but not the symbolic ref
    # Also remove remote config to prevent git pull/fetch from re-downloading remote info
    remove_remotes_cmd = (
        'git remote remove origin 2>/dev/null || true && '
        'rm -rf .git/refs/remotes && '
        "sed -i '/refs\\/remotes\\//d' .git/packed-refs 2>/dev/null || true"
    )
    await runtime.call_executor('run_command', {'command': remove_remotes_cmd, 'cwd': working_dir})

    # 4. Truncate git history to prevent peeking at future changes via git log/diff/reflog
    if not skip_git_truncate:
        await run_git_truncate(runtime, base_commit, working_dir)
