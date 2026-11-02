import json
import re

from typing import Any, cast

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.scale_swe.grading import get_eval_report
from nexus.extensions.tasks.swe.t3build_repolaunch_midtrain.constants import (
    KEY_INSTANCE_ID,
    get_repo_ext,
)
from nexus.extensions.tasks.swe.t3build_repolaunch_midtrain.test_spec import (
    TestSpec,
    make_eval_script_list,
)
from nexus.runtime import Runtime


def _has_new_file_in_patch(diff_text: str) -> bool:
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


def _get_new_files_in_patch(diff_text: str) -> list[str]:
    """Return the list of files that the patch adds from scratch (--- /dev/null)."""
    new_files = []
    chunks = re.split(r'^diff --git ', diff_text, flags=re.MULTILINE)
    for chunk in chunks:
        if not chunk.strip():
            continue
        chunk = 'diff --git ' + chunk
        is_new = re.search(r'^new file mode \d+', chunk, flags=re.MULTILINE) or re.search(
            r'^--- /dev/null$', chunk, flags=re.MULTILINE
        )
        if is_new:
            m = re.search(r'^\+\+\+ b/(.+)$', chunk, flags=re.MULTILINE)
            if m:
                new_files.append(m.group(1).rstrip('\r'))
    return new_files


def get_eval_script(instance: SWEbenchInstance) -> str:
    """Generate the evaluation script for a Scale-SWE instance.

    Differences from t3build_repolaunch_midtrain:
    1. repo_directory comes from instance['workdir'] (/workspace/<repo>, not /testbed).
    2. Before applying test_patch, any files that the patch adds are removed from the
       working tree first.  This prevents 'git apply' from failing with
       "already exists in working directory" when the model's patch happened to create
       the same test file (with different content/test names).
    """
    base_commit = instance.get('base_commit', '')
    test_patch = instance.get('test_patch', '')

    repo_directory = instance.get('workdir', '/testbed')
    env_name = repo_directory.lstrip('/')

    specs = instance.get('spec_dict')
    run_all_tests = bool(_has_new_file_in_patch(test_patch))

    eval_script_list = make_eval_script_list(
        instance, specs, env_name, repo_directory, base_commit, test_patch, run_all_tests
    )

    # Insert `rm -f` for every file that test_patch adds as a new file, immediately
    # before the `git apply` command.  This ensures the reference test file always
    # wins over whatever the model may have created at the same path.
    new_files = _get_new_files_in_patch(test_patch)
    if new_files:
        for i, cmd in enumerate(eval_script_list):
            if 'git apply' in cmd:
                for j, f in enumerate(new_files):
                    eval_script_list.insert(i + j, f'rm -f {f}')
                break

    return '\n'.join(['#!/bin/bash', 'set -uxo pipefail'] + eval_script_list) + '\n'


def get_eval_report_map(
    instance: SWEbenchInstance, prediction: dict[str, str], test_log_path: str, include_tests_status: bool
) -> dict[str, Any]:
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance['repo']
    version = instance.get('version', 'latest')

    def _from_json_or_obj(key: str) -> Any:
        instance_dict = cast(dict[str, Any], instance)
        if key not in instance_dict:
            return []
        if isinstance(instance_dict[key], str):
            return json.loads(instance_dict[key])
        return instance_dict[key]

    pass_to_pass = _from_json_or_obj('PASS_TO_PASS')
    fail_to_pass = _from_json_or_obj('FAIL_TO_PASS')

    test_patch = instance.get('test_patch', '')
    has_new_test_file = _has_new_file_in_patch(test_patch)

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

    return get_eval_report(
        test_spec, prediction, test_log_path, include_tests_status, has_new_test_file=has_new_test_file
    )


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
    """Initialize the inference container for Scale-SWE.

    Scale-SWE images are pre-built at parent_commit (buggy state).
    We still reset + clean to ensure a pristine starting point for the agent,
    then truncate git history to prevent peeking at future commits.

    Mirrors t3build_repolaunch_midtrain.infer_runtime_init, but uses
    instance['workdir'] instead of the hardcoded '/testbed'.
    """
    from nexus.extensions.tasks.swe.infer import run_git_truncate

    working_dir = instance.get('workdir', '/testbed')
    base_commit = instance.get('base_commit', '')

    # Allow git operations regardless of directory ownership inside the container
    safe_dir_cmd = f'git config --global --add safe.directory {working_dir}'
    await runtime.call_executor('run_command', {'command': safe_dir_cmd, 'cwd': working_dir})

    # Reset tracked files to base_commit (parent_commit / buggy state)
    if base_commit:
        reset_cmd = f'git reset --hard {base_commit}'
        await runtime.call_executor('run_command', {'command': reset_cmd, 'cwd': working_dir})

    # Remove untracked files so git diff only captures the agent's changes
    clean_cmd = 'git clean -fdx'
    await runtime.call_executor('run_command', {'command': clean_cmd, 'cwd': working_dir})

    if kwargs.get('remove_git_dir', False):
        from nexus.extensions.tasks.swe.infer import remove_git_dir

        await remove_git_dir(runtime, working_dir)
        return

    # Remove remote refs to avoid "bad object refs/remotes/origin/HEAD" during gc
    remove_remotes_cmd = (
        'git remote remove origin 2>/dev/null || true && '
        'rm -rf .git/refs/remotes && '
        "sed -i '/refs\\/remotes\\//d' .git/packed-refs 2>/dev/null || true"
    )
    await runtime.call_executor('run_command', {'command': remove_remotes_cmd, 'cwd': working_dir})

    # Truncate git history so the agent cannot peek at the fix via git log/diff
    if not skip_git_truncate:
        await run_git_truncate(runtime, base_commit, working_dir)
