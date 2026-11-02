import json
import re

from dataclasses import dataclass
from typing import Any, cast

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_gym.constants import (
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    MAP_REPO_VERSION_TO_SPECS,
    NON_TEST_EXTS,
    PASS_TO_PASS,
)


@dataclass
class TestSpec:
    """
    A dataclass that represents a test specification for a single instance of SWE-bench.
    """

    instance_id: str
    repo: str
    version: str
    eval_script_list: list[str]
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    language: str

    @property
    def eval_script(self):
        return '\n'.join(['#!/bin/bash', 'set -xo pipefail'] + self.eval_script_list) + '\n'
        # Don't exit early because we need to revert tests at the end


DIFF_MODIFIED_FILE_REGEX = r'--- a/(.*)'


def get_test_directives(instance: SWEbenchInstance) -> list:
    """
    Get test directives from the test_patch of a task instance

    Args:
        instance (dict): task instance
    Returns:
        directives (list): List of test directives
    """
    # For seq2seq code repos, testing command is fixed
    if instance['repo'] == 'swe-bench/humaneval':
        return ['test.py']

    # Get test directives from test patch and remove non-test files
    diff_pat = r'diff --git a/.* b/(.*)'
    test_patch = instance['test_patch']
    directives = re.findall(diff_pat, test_patch)
    directives = [d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)]

    # For Django tests, remove extension + "tests/" prefix and convert slashes to dots (module referencing)
    if instance['repo'] == 'django/django':
        directives_transformed = []
        for d in directives:
            d = d[: -len('.py')] if d.endswith('.py') else d
            d = d[len('tests/') :] if d.startswith('tests/') else d
            d = d.replace('/', '.')
            directives_transformed.append(d)
        directives = directives_transformed

    return directives


def make_test_command(instance):
    if instance['repo'] == 'python/mypy':
        pattern = r'\[case ([^\]]+)\]'
        test_keys = re.findall(pattern, instance['test_patch'])
        test_keys_or = ' or '.join(test_keys)
        return (
            cast(str, MAP_REPO_VERSION_TO_SPECS[instance['repo']][instance['version']]['test_cmd'])
            + ' '
            + f'"{test_keys_or}"'
        )
    return ' '.join([
        cast(str, MAP_REPO_VERSION_TO_SPECS[instance['repo'].lower()][instance['version']]['test_cmd']),
        *get_test_directives(instance),
    ])


def make_eval_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch):
    """
    Applies the test patch and runs the tests.
    """
    heredoc_delimiter = 'EOF_114329324912'
    test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch)
    # Reset test files to the state they should be in before the patch.
    reset_tests_command = f'git checkout {base_commit} {" ".join(test_files)}'
    apply_test_patch_command = f"git apply -v - <<'{heredoc_delimiter}'\n{test_patch}\n{heredoc_delimiter}"
    test_command = make_test_command(instance)
    eval_commands = [
        'source /opt/miniconda3/bin/activate',
        f'conda activate {env_name}',
        f'cd {repo_directory}',
    ]
    if 'eval_commands' in specs:
        eval_commands += specs['eval_commands']
    eval_commands += [
        f'git config --global --add safe.directory {repo_directory}',  # for nonroot user
        f'cd {repo_directory}',
        # This is just informational, so we have a record
        'git status',
        'git show',
        f'git diff {base_commit}',
        'source /opt/miniconda3/bin/activate',
        f'conda activate {env_name}',
    ]
    if 'install' in specs:
        eval_commands.append(specs['install'])
    eval_commands += [
        reset_tests_command,
        apply_test_patch_command,
        test_command,
        reset_tests_command,  # Revert tests after done, leave the repo in the same state as before
    ]
    return eval_commands


def make_test_spec(instance: SWEbenchInstance) -> TestSpec:
    instance_id = instance[KEY_INSTANCE_ID]
    # if there's capital letters in the repo name, convert to lowercase
    if instance_id != instance_id.lower():
        instance_id = instance_id.lower()
    repo = instance['repo'].lower()
    version = instance['version']
    base_commit = instance['base_commit']
    test_patch = instance['test_patch']

    def _from_json_or_obj(key: str) -> Any:
        """If key points to string, load with json"""
        instance_dict = cast(dict[str, Any], instance)
        if isinstance(instance_dict[key], str):
            return json.loads(instance_dict[key])
        return instance_dict[key]

    try:
        pass_to_pass = _from_json_or_obj(PASS_TO_PASS)
    except Exception as e:
        print(f'Error parsing PASS_TO_PASS for instance {instance_id}: {e}. PASS_TO_PASS: {instance[PASS_TO_PASS]}')
        pass_to_pass = []

    try:
        fail_to_pass = _from_json_or_obj(FAIL_TO_PASS)
    except Exception as e:
        print(f'Error parsing FAIL_TO_PASS for instance {instance_id}: {e}. FAIL_TO_PASS: {instance[FAIL_TO_PASS]}')
        fail_to_pass = []

    env_name = 'testbed'
    repo_directory = f'/{env_name}'
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]

    eval_script_list = make_eval_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch)

    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        eval_script_list=eval_script_list,
        version=version,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language='py',
    )


def get_repo_language(instance: SWEbenchInstance) -> str | None:
    if instance['repo'] not in MAP_REPO_VERSION_TO_SPECS:
        return None
    return 'py'
