import json

from typing import Any, cast

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_rebench.constants import KEY_INSTANCE_ID, MAP_REPO_TO_EXT, MAP_REPO_VERSION_TO_SPECS
from nexus.extensions.tasks.swe.swe_rebench.grading import get_eval_report
from nexus.extensions.tasks.swe.swe_rebench.test_spec import TestSpec, make_eval_script_list


def get_eval_script(instance: SWEbenchInstance) -> str:
    repo = instance['repo']
    version = instance.get('version')
    base_commit = instance['base_commit']
    test_patch = instance['test_patch']

    env_name = 'testbed'
    repo_directory = f'/{env_name}'
    specs = instance['spec_dict'] if 'spec_dict' in instance else MAP_REPO_VERSION_TO_SPECS[repo][version]

    eval_script_list = make_eval_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch)

    return '\n'.join(['#!/bin/bash', 'set -uxo pipefail'] + eval_script_list) + '\n'


def get_eval_report_map(
    instance: SWEbenchInstance, prediction: dict[str, str], test_log_path: str, include_tests_status: bool
) -> dict[str, Any]:
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance['repo']
    version = instance.get('version')

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

    spec_dict = instance.get('spec_dict', None)

    test_spec = TestSpec(
        instance_id=instance_id,
        repo=repo,
        version=version,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        eval_script_list=[],
        language=MAP_REPO_TO_EXT[repo],
        spec_dict=spec_dict,
    )
    return get_eval_report(test_spec, prediction, test_log_path, include_tests_status)


def get_problem_statement(instance: SWEbenchInstance) -> str:
    return instance.get('problem_statement') or instance.get('problem') or ''


def get_repo_language(instance: SWEbenchInstance) -> str | None:
    return MAP_REPO_TO_EXT.get(instance['repo'], None)
