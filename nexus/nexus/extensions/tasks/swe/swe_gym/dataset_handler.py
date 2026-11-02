import json

from typing import Any, cast

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_gym.constants import (
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    MAP_REPO_VERSION_TO_SPECS,
    PASS_TO_PASS,
)
from nexus.extensions.tasks.swe.swe_gym.grading import get_eval_report
from nexus.extensions.tasks.swe.swe_gym.test_spec import TestSpec, make_eval_script_list


def get_eval_script(instance: SWEbenchInstance) -> str:
    instance_id = instance[KEY_INSTANCE_ID]
    # if there's capital letters in the repo name, convert to lowercase
    if instance_id != instance_id.lower():
        instance_id = instance_id.lower()
    repo = instance['repo'].lower()
    version = instance['version']
    base_commit = instance['base_commit']
    test_patch = instance['test_patch']

    env_name = 'testbed'
    repo_directory = f'/{env_name}'
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]

    eval_script_list = make_eval_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch)

    return '\n'.join(['#!/bin/bash', 'set -xo pipefail'] + eval_script_list) + '\n'


def get_eval_report_map(
    instance: SWEbenchInstance, prediction: dict[str, str], test_log_path: str, include_tests_status: bool
) -> dict[str, Any]:
    instance_id = instance[KEY_INSTANCE_ID]
    # if there's capital letters in the repo name, convert to lowercase
    if instance_id != instance_id.lower():
        instance_id = instance_id.lower()
    repo = instance['repo'].lower()
    version = instance['version']

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

    test_spec = TestSpec(
        instance_id=instance_id,
        repo=repo,
        version=version,
        eval_script_list=[],
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language='py',
    )
    return get_eval_report(test_spec, prediction, test_log_path, include_tests_status)


def get_problem_statement(instance: SWEbenchInstance) -> str:
    return instance.get('problem_statement') or instance.get('problem') or ''


def get_repo_language(instance: SWEbenchInstance) -> str | None:
    if instance['repo'] not in MAP_REPO_VERSION_TO_SPECS:
        return None
    return 'py'
