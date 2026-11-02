"""
Test specification for SWE-Bench-Multilingual dataset.

This module provides functionality to create test specifications for instances
across multiple programming languages.
"""

import json
import re

from dataclasses import dataclass
from typing import Any, cast

from unidiff import PatchSet

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_bench_multilingual import carbon_3098, fmt_3901, valkey_928
from nexus.extensions.tasks.swe.swe_bench_multilingual.constants import (
    END_TEST_OUTPUT,
    KEY_INSTANCE_ID,
    MAP_REPO_TO_EXT,
    MAP_REPO_VERSION_TO_SPECS,
    NON_TEST_EXTS,
    START_TEST_OUTPUT,
    get_language_from_repo,
)


@dataclass
class TestSpec:
    """
    A dataclass that represents a test specification for a single instance of SWE-bench-Multilingual.
    """

    instance_id: str
    repo: str
    version: str | None
    eval_script_list: list[str]
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    language: str

    @property
    def eval_script(self):
        return '\n'.join(['#!/bin/bash', 'set -uxo pipefail'] + self.eval_script_list) + '\n'


def _from_json_or_obj(instance: SWEbenchInstance, key: str) -> Any:
    """If key points to string, load with json."""
    instance_dict = cast(dict[str, Any], instance)
    if key not in instance_dict:
        # If P2P, F2P keys not found, it's a validation instance
        return []
    if isinstance(instance_dict[key], str):
        return json.loads(instance_dict[key])
    return instance_dict[key]


def get_modified_files(patch: str) -> list[str]:
    """
    Get the list of modified files in a patch
    """
    source_files = []
    for file in PatchSet(patch):
        if file.source_file != '/dev/null':
            source_files.append(file.source_file)
    return [x[2:] for x in source_files if x.startswith('a/')]


def _get_patch_file_paths(patch: str) -> set[str]:
    return {file.path for file in PatchSet(patch)}


def _get_patch_file_text(patch: str, path: str) -> str | None:
    for file in PatchSet(patch):
        if file.path == path:
            return str(file)
    return None


def _get_cucumber_feature_paths(test_names: list[str]) -> list[str]:
    paths = []
    for test_name in test_names:
        match = re.match(r'([^:]+\.feature):\d+\s+Scenario:', test_name)
        if match and match.group(1) not in paths:
            paths.append(match.group(1))
    return paths


def normalize_test_patch(instance: SWEbenchInstance, test_patch: str) -> str:
    """
    Apply small dataset-specific repairs to the evaluation test patch.

    jekyll__jekyll-8771 expects a cucumber scenario added in the golden patch,
    but that feature hunk is absent from the dataset test_patch. Without it,
    the fixed line number points at a different pre-existing scenario.
    """
    if valkey_928.is_target_instance(instance):
        return valkey_928.normalize_test_patch(instance, test_patch)

    if instance.get(KEY_INSTANCE_ID) != 'jekyll__jekyll-8771':
        return test_patch

    missing_patch = ''
    test_patch_paths = _get_patch_file_paths(test_patch)
    for path in _get_cucumber_feature_paths(_from_json_or_obj(instance, 'FAIL_TO_PASS')):
        if path in test_patch_paths:
            continue
        patch_file_text = _get_patch_file_text(instance.get('patch', ''), path)
        if patch_file_text:
            missing_patch += patch_file_text

    return missing_patch + test_patch


def get_test_directives(instance: SWEbenchInstance) -> list[str]:
    """
    Get test directives from the test_patch of a task instance

    Args:
        instance (dict): task instance
    Returns:
        directives (list): List of test directives
    """
    # Get test directives from test patch and remove non-test files
    diff_pat = r'diff --git a/.* b/(.*)'
    test_patch = normalize_test_patch(instance, instance['test_patch'])
    directives = re.findall(diff_pat, test_patch)
    return [d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)]


def get_test_cmds(instance: SWEbenchInstance) -> list[str]:
    """
    Get test commands for an instance from MAP_REPO_VERSION_TO_SPECS.

    Args:
        instance: SWE-bench-Multilingual instance

    Returns:
        List of test commands

    Raises:
        ValueError: If repo or version not found in MAP_REPO_VERSION_TO_SPECS
    """
    repo = instance['repo']
    version = instance.get('version', '')

    # Get from MAP_REPO_VERSION_TO_SPECS
    if repo not in MAP_REPO_VERSION_TO_SPECS:
        raise ValueError(f"Repo '{repo}' not found in MAP_REPO_VERSION_TO_SPECS")

    if version not in MAP_REPO_VERSION_TO_SPECS[repo]:
        available_versions = list(MAP_REPO_VERSION_TO_SPECS[repo].keys())[:5]
        raise ValueError(
            f"Version '{version}' not found for repo '{repo}'. Available versions: {available_versions}..."
        )

    test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version].get('test_cmd', [])

    if not test_cmd:
        raise ValueError(f'No test_cmd configured for {repo} version {version}')

    # Normalize to list
    if isinstance(test_cmd, str):
        return [test_cmd]

    # Ensure we return a list of strings
    if isinstance(test_cmd, list):
        return test_cmd

    # Fallback for unexpected types
    return [str(test_cmd)]


def _get_spec_cmds(instance: SWEbenchInstance, key: str) -> list[str]:
    """Return a normalized command list from the instance's version spec."""
    repo = instance['repo']
    version = instance.get('version', '')

    if repo not in MAP_REPO_VERSION_TO_SPECS:
        return []

    if version not in MAP_REPO_VERSION_TO_SPECS[repo]:
        return []

    commands = MAP_REPO_VERSION_TO_SPECS[repo][version].get(key, [])

    if not commands:
        return []

    if isinstance(commands, str):
        return [commands]

    if isinstance(commands, list):
        return [str(command) for command in commands]

    return [str(commands)]


def get_install_cmds(instance: SWEbenchInstance) -> list[str]:
    """Return compatibility setup commands currently run during evaluation.

    ``install`` is normally an image-setup concern in upstream SWE-bench, but
    multilingual evaluation has executed it here since MR !265. Keep that
    behavior for compatibility with instances such as apache__druid-13704.
    """
    return _get_spec_cmds(instance, 'install')


def get_build_cmds(instance: SWEbenchInstance) -> list[str]:
    """Return commands that must rebuild the patched repository before tests.

    Upstream SWE-bench runs ``build`` after applying the candidate and test
    patches. Omitting this step can make evaluation execute stale binaries from
    the prebuilt image instead of the submitted patch.
    """
    return _get_spec_cmds(instance, 'build')


def make_eval_script_list(instance: SWEbenchInstance, env_name: str, repo_directory: str) -> list[str]:
    """
    Create evaluation script for a multilingual instance.
    This uses get_test_cmds to retrieve test commands from MAP_REPO_VERSION_TO_SPECS
    or generates them from FAIL_TO_PASS/PASS_TO_PASS if not configured.

    Args:
        instance: SWE-bench-Multilingual instance
        env_name: Environment name (e.g., 'testbed')
        repo_directory: Repository directory path

    Returns:
        List of commands to execute
    """
    if valkey_928.is_target_instance(instance):
        return valkey_928.make_eval_script_list(instance, repo_directory)

    base_commit = instance['base_commit']
    test_patch = normalize_test_patch(instance, instance['test_patch'])
    heredoc_delimiter = 'EOF_114329324912'

    test_files = get_modified_files(test_patch)

    # Reset test files to the state they should be in before the patch.
    if test_files:
        reset_tests_command = f'git checkout {base_commit} {" ".join(test_files)}'
    else:
        reset_tests_command = 'echo "No test files to reset"'

    apply_test_patch_command = (
        f"git apply --verbose --reject - <<'{heredoc_delimiter}'\n{test_patch}\n{heredoc_delimiter}"
    )

    # Get test commands from MAP_REPO_VERSION_TO_SPECS or generate them
    test_commands = get_test_cmds(instance)
    # Preserve the existing eval-time install compatibility, then restore the
    # upstream build phase so candidate changes are compiled before testing.
    install_commands = get_install_cmds(instance)
    build_commands = get_build_cmds(instance)
    build_phase_commands = (
        [
            # Build failures must stop evaluation. Otherwise the script can
            # continue into tests and execute stale binaries from the image.
            'set -e',
            *build_commands,
            # Test commands may intentionally manage/inspect their own exit codes.
            'set +e',
        ]
        if build_commands
        else []
    )

    setup_commands: list[str] = []
    cleanup_commands: list[str] = []
    if carbon_3098.is_target_instance(instance):
        setup_commands = carbon_3098.make_setup_commands()
    elif fmt_3901.is_target_instance(instance):
        setup_commands = fmt_3901.make_setup_commands()
        # fmt_3901 patches harness sources outside test_patch, so `reset_tests_command` does not
        # cover them; restore them explicitly to leave the tree as graded.
        cleanup_commands = fmt_3901.make_cleanup_commands()

    return [
        f'cd {repo_directory}',
        f'git config --global --add safe.directory {repo_directory}',  # for nonroot user
        f'cd {repo_directory}',
        reset_tests_command,
        apply_test_patch_command,
        *install_commands,
        *setup_commands,
        *build_phase_commands,
        f": '{START_TEST_OUTPUT}'",
        *test_commands,  # Use test commands from get_test_cmds
        f": '{END_TEST_OUTPUT}'",
        reset_tests_command,
        *cleanup_commands,
    ]


def make_test_spec(instance: SWEbenchInstance) -> TestSpec:
    """
    Create a test specification for a SWE-bench-Multilingual instance.

    Args:
        instance: A SWE-bench-Multilingual instance

    Returns:
        TestSpec: Test specification for the instance
    """
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance['repo']
    version = instance.get('version')

    if valkey_928.is_target_instance(instance):
        valkey_928.validate_instance(instance, cast(str, instance.get('test_patch')))

    pass_to_pass = _from_json_or_obj(instance, 'PASS_TO_PASS')
    fail_to_pass = _from_json_or_obj(instance, 'FAIL_TO_PASS')

    env_name = 'testbed'
    repo_directory = f'/{env_name}'

    eval_script_list = make_eval_script_list(instance, env_name, repo_directory)

    # Get language from repo
    language = get_language_from_repo(repo)
    if language is None:
        language = MAP_REPO_TO_EXT.get(repo, 'unknown')

    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        eval_script_list=eval_script_list,
        version=version,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language=language,
    )
