import json
import re

from dataclasses import dataclass
from typing import Any, cast

from unidiff import PatchSet

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.t3build_swe_bench_live.constants import (
    END_TEST_OUTPUT,
    KEY_INSTANCE_ID,
    START_TEST_OUTPUT,
    get_repo_ext,
)


@dataclass
class TestSpec:
    """
    A dataclass that represents a test specification for a single instance of t3build_swe_bench_live.
    """

    instance: SWEbenchInstance
    instance_id: str
    repo: str
    version: str
    eval_script_list: list[str]
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    language: str

    @property
    def eval_script(self):
        return '\n'.join(['#!/bin/bash', 'set -uxo pipefail'] + self.eval_script_list) + '\n'


def safe_patch_set(patch: str) -> PatchSet:
    """
    Safely create a PatchSet, with fallback to handle malformed patches.
    """
    try:
        return PatchSet(patch)
    except Exception:
        cleaned_patch = re.sub(r'\n{2,}', '\n', patch)
        return PatchSet(cleaned_patch)


def get_modified_files(patch: str) -> list[str]:
    """
    Get the list of modified files in a patch
    """
    source_files = []
    for file in safe_patch_set(patch):
        if file.source_file != '/dev/null':
            source_files.append(file.source_file)
    return [x[2:] for x in source_files if x.startswith('a/')]


def get_test_cmds(instance) -> list:
    """Get test commands from instance.

    Priority:
    1. spec_dict.test_cmd (standard format)
    2. Default fallback
    """
    # Standard format: spec_dict.test_cmd
    spec_dict = instance.get('spec_dict')
    if spec_dict:
        test_cmd = spec_dict.get('test_cmd', '')
        if test_cmd:
            return [test_cmd] if isinstance(test_cmd, str) else test_cmd

    # Default fallback
    return ['pytest -rA --tb=long']


def make_eval_script_list_common(
    instance, specs, env_name, repo_directory, base_commit, test_patch, run_all_tests
) -> list:
    """
    Applies the test patch and runs the tests (common version for non-Python repos).
    """
    heredoc_delimiter = 'EOF_114329324912'
    test_files = get_modified_files(test_patch)

    if test_files:
        reset_tests_command = f'git checkout {base_commit} {" ".join(test_files)}'
    else:
        reset_tests_command = 'echo "No test files to reset"'

    build_commands = []
    if specs and 'build' in specs:
        build_commands.extend(specs['build'])

    apply_test_patch_command = (
        f"git apply --verbose --reject - <<'{heredoc_delimiter}'\n{test_patch}\n{heredoc_delimiter}"
    )
    test_commands = get_test_cmds(instance)
    return [
        'export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt',
        f'cd {repo_directory}',
        f'git config --global --add safe.directory {repo_directory}',
        f'cd {repo_directory}',
        reset_tests_command,
        apply_test_patch_command,
        *build_commands,
        f": '{START_TEST_OUTPUT}'",
        *test_commands,
        f"set +x; echo '{END_TEST_OUTPUT}'",
        reset_tests_command,
    ]


def make_eval_script_list_py(
    instance, specs, env_name, repo_directory, base_commit, test_patch, run_all_tests=False
) -> list | None:
    """
    Applies the test patch and runs the tests (Python version).

    If spec_dict contains test_cmd, use it directly without conda activation.
    Otherwise, fall back to conda-based environment.
    """
    heredoc_delimiter = 'EOF_114329324912'
    test_files = get_modified_files(test_patch)
    reset_tests_command = f'git checkout {base_commit} {" ".join(test_files)}'
    apply_test_patch_command = f"git apply -v - <<'{heredoc_delimiter}'\n{test_patch}\n{heredoc_delimiter}"

    # Check if spec_dict has explicit test_cmd - if so, use simpler script without conda
    has_explicit_test_cmd = specs and specs.get('test_cmd')

    if has_explicit_test_cmd:
        # Use test_cmd directly without conda
        test_commands = get_test_cmds(instance)

        eval_commands = [
            'export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt',
            f'cd {repo_directory}',
            f'git config --global --add safe.directory {repo_directory}',
            'git status',
            'git show',
            f'git -c core.fileMode=false diff {base_commit}',
        ]

        if specs and 'eval_commands' in specs:
            eval_commands += specs['eval_commands']

        build_commands = []
        if specs and 'build' in specs:
            build_commands.extend(specs['build'])

        eval_commands += [
            reset_tests_command,
            apply_test_patch_command,
            *build_commands,
            f": '{START_TEST_OUTPUT}'",
            *test_commands,
            f"set +x; echo '{END_TEST_OUTPUT}'",
            reset_tests_command,
        ]
        return eval_commands
    return None


def make_eval_script_list(
    instance, specs, env_name, repo_directory, base_commit, test_patch, run_all_tests=False
) -> list:
    """
    Applies the test patch and runs the tests.
    """
    ext = get_repo_ext(instance)
    common_func = make_eval_script_list_common
    func = {
        'py': make_eval_script_list_py,
    }.get(ext, common_func)
    result = func(instance, specs, env_name, repo_directory, base_commit, test_patch, run_all_tests)
    # Fallback to common if py version returns None
    if result is None:
        result = common_func(instance, specs, env_name, repo_directory, base_commit, test_patch, run_all_tests)
    return result


def make_test_spec(
    instance: SWEbenchInstance,
) -> TestSpec:
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance['repo']
    version = instance.get('version', 'latest')
    base_commit = instance['base_commit']
    test_patch = instance['test_patch']

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

    env_name = 'testbed'
    repo_directory = f'/{env_name}'
    specs = instance.get('spec_dict')

    eval_script_list = make_eval_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch)

    return TestSpec(
        instance=instance,
        instance_id=instance_id,
        repo=repo,
        eval_script_list=eval_script_list,
        version=version,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language=get_repo_ext(instance),
    )
