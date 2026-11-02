"""
Test specification generation for SWE-Smith dataset.

SWE-Smith differs from SWE-bench in that:
1. Docker images contain the CORRECT code (not buggy code)
2. Bug patches must be applied BEFORE inference and evaluation
3. Log parsers are embedded in RepoProfile classes (via registry)

Note: The bug patch application during evaluation is handled by `_evaluate_patch`
in eval.py, which checks for DATASET_SWE_SMITH and applies bug patch before model patch.
"""

import json

from dataclasses import dataclass
from typing import Any, cast

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_smith.constants import (
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    PASS_TO_PASS,
    TEST_OUTPUT_END,
    TEST_OUTPUT_START,
)
from nexus.extensions.tasks.swe.swe_smith.profiles import registry


@dataclass
class TestSpec:
    """
    A dataclass that represents a test specification for a single instance of SWE-Smith.
    """

    instance_id: str
    repo: str
    version: str
    eval_script_list: list[str]
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    language: str
    dataset_type: str = 'swe-smith'

    @property
    def eval_script(self):
        return '\n'.join(['#!/bin/bash', 'set -uxo pipefail'] + self.eval_script_list) + '\n'


def make_test_spec(instance: SWEbenchInstance) -> TestSpec:
    """
    Generate a TestSpec for a SWE-Smith instance.

    Unlike SWE-bench, SWE-Smith images contain correct code.
    The bug patch application is handled by _evaluate_patch in eval.py.
    """
    dataset_type = instance.get('dataset_type', 'swe-smith')
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance_id.rsplit('.', 1)[0]

    # Get profile for this repo
    rp = registry.get(repo)

    # Get test command from profile (cast to dict for type compatibility)
    test_cmd, _ = rp.get_test_cmd(cast(dict[str, Any], instance))

    # Parse FAIL_TO_PASS and PASS_TO_PASS from instance
    def _from_json_or_obj(key: str) -> list[str]:
        instance_dict = cast(dict[str, Any], instance)
        if key not in instance_dict:
            return []
        val = instance_dict[key]
        if isinstance(val, str):
            return json.loads(val)
        return val

    fail_to_pass = _from_json_or_obj(FAIL_TO_PASS)
    pass_to_pass = _from_json_or_obj(PASS_TO_PASS)

    # Build eval script (only runs tests, patch application handled by _evaluate_patch)
    repo_directory = '/testbed'
    eval_script_list = make_eval_script_list(
        repo_directory=repo_directory,
        test_cmd=test_cmd,
    )

    # Get language from profile
    language = rp.exts[0].lstrip('.') if rp.exts else 'py'

    return TestSpec(
        dataset_type=dataset_type,
        instance_id=instance_id,
        repo=repo,
        version='',  # SWE-Smith doesn't use version
        eval_script_list=eval_script_list,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language=language,
    )


def make_eval_script_list(
    repo_directory: str,
    test_cmd: str,
) -> list[str]:
    """
    Generate the evaluation script commands for SWE-Smith.

    The eval flow in _evaluate_patch for SWE-Smith is:
    1. Apply bug patch (restore buggy state) - handled by _evaluate_patch
    2. Apply model patch - handled by _evaluate_patch
    3. Execute this eval script (just runs tests)
    """
    return [
        f'cd {repo_directory}',
        f'git config --global --add safe.directory {repo_directory}',
        'git status',
        # Run tests
        f": '{TEST_OUTPUT_START}'",
        test_cmd,
        f": '{TEST_OUTPUT_END}'",
    ]


def get_repo_language(instance: SWEbenchInstance) -> str | None:
    """Get the primary language for a repository from its profile."""
    try:
        rp = registry.get_from_inst(cast(dict[str, Any], instance))
        return rp.exts[0].lstrip('.') if rp.exts else None
    except KeyError:
        return None
