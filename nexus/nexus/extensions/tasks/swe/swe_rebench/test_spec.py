import json

from dataclasses import dataclass
from io import StringIO
from typing import Any, cast

from unidiff import PatchSet

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_rebench.constants import (
    END_TEST_OUTPUT,
    KEY_INSTANCE_ID,
    MAP_REPO_TO_EXT,
    MAP_REPO_VERSION_TO_SPECS,
    NON_TEST_EXTS,
    START_TEST_OUTPUT,
)


@dataclass
class TestSpec:
    """
    A dataclass that represents a test specification for a single instance of SWE-bench.
    """

    instance_id: str
    repo: str
    version: str | None
    eval_script_list: list[str]
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    language: str
    spec_dict: dict | None = None

    @property
    def eval_script(self):
        return '\n'.join(['#!/bin/bash', 'set -uxo pipefail'] + self.eval_script_list) + '\n'


def get_modified_files(patch: str) -> list[str]:
    """
    Get the list of modified files in a patch
    """
    source_files = []
    for file in PatchSet(patch):
        if file.source_file != '/dev/null':
            source_files.append(file.source_file)
    return [x[2:] for x in source_files if x.startswith('a/')]


def get_test_cmds(instance) -> list:
    test_cmd = MAP_REPO_VERSION_TO_SPECS[instance['repo']][instance['version']]['test_cmd']
    return [test_cmd] if isinstance(test_cmd, str) else cast(list, test_cmd)


def get_changed_files(patch: str) -> list[str]:
    """Extract changed or added filenames from a git patch, excluding deleted files."""

    files = []
    patch_set: PatchSet = PatchSet(StringIO(patch))

    for patched_file in patch_set:
        # Skip deleted files
        if not patched_file.is_removed_file:
            files.append(patched_file.path)

    return files


def get_default_test_directives(test_patch: str) -> list[str]:
    directives = get_changed_files(test_patch)
    return [d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)]


def get_test_directives(instance) -> list[str]:
    """
    Get test directives from the test_patch of a task instance

    Args:
        instance (dict): task instance
    Returns:
        directives (list): List of test directives
    """
    # HumanEvalFix: For seq2seq code repos, testing command is fixed
    if any(x == instance['repo'] for x in ['swe-bench/humaneval', 'swe-bench/humanevalfix-python']):
        return ['test.py']
    if any(x == instance['repo'] for x in ['swe-bench/humanevalfix-go', 'swe-bench/humanevalfix-java']):
        return []
    if instance['repo'] == 'swe-bench/humanevalfix-js':
        return ['test.js']
    if instance['repo'] == 'nebius/nebo':
        return ['']

    # Get test directives from test patch and remove non-test files
    test_patch = instance['test_patch']
    directives = get_default_test_directives(test_patch)

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


def make_eval_script_list_common(instance, specs, env_name, repo_directory, base_commit, test_patch) -> list:
    """
    Applies the test patch and runs the tests.
    """
    heredoc_delimiter = 'EOF_114329324912'
    test_files = get_modified_files(test_patch)
    # Reset test files to the state they should be in before the patch.
    if test_files:
        reset_tests_command = f'git checkout {base_commit} {" ".join(test_files)}'
    else:
        reset_tests_command = 'echo "No test files to reset"'

    build_commands = []
    if 'build' in specs:
        build_commands.extend(specs['build'])

    apply_test_patch_command = (
        f"git apply --verbose --reject - <<'{heredoc_delimiter}'\n{test_patch}\n{heredoc_delimiter}"
    )
    test_commands = get_test_cmds(instance)
    return [
        f'cd {repo_directory}',
        f'git config --global --add safe.directory {repo_directory}',  # for nonroot user
        f'cd {repo_directory}',
        # This is just informational, so we have a record
        # f"git status",
        # f"git show",
        # f"git -c core.fileMode=false diff {base_commit}",
        reset_tests_command,
        apply_test_patch_command,
        *build_commands,
        f": '{START_TEST_OUTPUT}'",
        *test_commands,
        f": '{END_TEST_OUTPUT}'",
        reset_tests_command,
    ]


def make_eval_script_list_py(instance, specs, env_name, repo_directory, base_commit, test_patch) -> list:
    """
    Applies the test patch and runs the tests.
    """
    heredoc_delimiter = 'EOF_114329324912'
    test_files = get_modified_files(test_patch)
    # Reset test files to the state they should be in before the patch.
    reset_tests_command = f'git checkout {base_commit} {" ".join(test_files)}'
    apply_test_patch_command = f"git apply -v - <<'{heredoc_delimiter}'\n{test_patch}\n{heredoc_delimiter}"
    if 'test_cmd' in specs:
        test_cmd = specs['test_cmd']
    else:
        test_cmd = MAP_REPO_VERSION_TO_SPECS[instance['repo']][instance['version']]['test_cmd']
    test_command = ' '.join([
        cast(str, test_cmd),
        *get_test_directives(instance),
    ])
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
        f'git -c core.fileMode=false diff {base_commit}',
        'source /opt/miniconda3/bin/activate',
        f'conda activate {env_name}',
    ]
    if 'install' in specs:
        eval_commands.append(specs['install'])
    eval_commands += [
        reset_tests_command,
        apply_test_patch_command,
        f": '{START_TEST_OUTPUT}'",
        test_command,
        f": '{END_TEST_OUTPUT}'",
        reset_tests_command,  # Revert tests after done, leave the repo in the same state as before
    ]
    return eval_commands


def make_eval_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch) -> list:
    """
    Applies the test patch and runs the tests.
    """
    ext = MAP_REPO_TO_EXT[instance['repo']]
    common_func = make_eval_script_list_common
    func = {
        'py': make_eval_script_list_py,
    }.get(ext, common_func)
    return func(instance, specs, env_name, repo_directory, base_commit, test_patch)


def make_test_spec(
    instance: SWEbenchInstance,
) -> TestSpec:
    """
    Make a test spec for a given SWE instance

    Args:
        instance (SWEInstance): The SWE instance to make a test spec for

    Returns:
        TestSpec: The test spec for the given SWE instance
    """
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance['repo']
    version = instance.get('version')
    base_commit = instance['base_commit']
    test_patch = instance['test_patch']

    def _from_json_or_obj(key: str) -> Any:
        """If key points to string, load with json"""
        instance_dict = cast(dict[str, Any], instance)
        if key not in instance_dict:
            # If P2P, F2P keys not found, it's a validation instance
            return []
        if isinstance(instance_dict[key], str):
            return json.loads(instance_dict[key])
        return instance_dict[key]

    pass_to_pass = _from_json_or_obj('PASS_TO_PASS')
    fail_to_pass = _from_json_or_obj('FAIL_TO_PASS')

    env_name = 'testbed'
    repo_directory = f'/{env_name}'
    if 'spec_dict' in instance:
        spec_dict = instance['spec_dict']
        specs = spec_dict
    else:
        spec_dict = None
        specs = MAP_REPO_VERSION_TO_SPECS[repo][version]

    eval_script_list = make_eval_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch)

    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        eval_script_list=eval_script_list,
        version=version,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language=MAP_REPO_TO_EXT[repo],
        spec_dict=spec_dict,
    )
