import json
import re

from typing import Any, cast

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_bench_turing.constants import (
    KEY_INSTANCE_ID,
    get_repo_ext,
)
from nexus.extensions.tasks.swe.swe_bench_turing.grading import get_eval_report
from nexus.extensions.tasks.swe.swe_bench_turing.test_spec import TestSpec, make_eval_script_list


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
    Get the evaluation script for a given instance

    Args:
        instance (SWEbenchInstance): The instance to get the evaluation script for
    Returns:
        str: The evaluation script
    """
    base_commit = instance['base_commit']
    test_patch = instance['test_patch']

    env_name = 'testbed'
    repo_directory = f'/{env_name}'
    specs = instance['spec_dict']

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
    version = instance['version']

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
