"""SWE-Env repository profile.

Built dynamically from a reference instance (repo_dict), supporting the pre-built SWE-Env dataset. Ported and slimmed from swecosmos.profiles.swe_env.
"""

import re
import shlex

from dataclasses import dataclass, field
from typing import Any

from nexus.extensions.tasks.swe.swe_cosmos.profiles.base import RepoProfile
from nexus.extensions.tasks.swe.swe_cosmos.utils import (
    TEST_OUTPUT_END,
    TEST_OUTPUT_START,
    _BUG_COMMIT_KEY,
    _EVAL_SCRIPT_PATH,
    _REVERT_FILES_KEY,
    get_patch_set,
)


_EXIT_CODE_RE = re.compile(r'>>>>>>>>>> Evaluation Exit Code: (\d+) <<<<<<<<<<')

_SWE_ENV_PROFILE_CACHE: dict[str, 'SWEEnvProfile'] = {}


@dataclass
class SWEEnvProfile(RepoProfile):
    """Profile for SWE-Env repositories."""

    dataset: str = 'SWE-Cosmos/SWE-Env'
    # SWE-Env forbids applying the init test patch during setup: the eval script itself inserts the test content at run time.
    apply_init_test_patch: bool = False
    ref_problem_statement: str | None = None
    test_files: list[str] = field(default_factory=list)
    # Pre-built eval script body (repo_dict.eval_script).
    test_function: str = ''
    # Written to EVAL_SCRIPT_PATH by the handler.
    test_cmd: str = f'bash {_EVAL_SCRIPT_PATH}'

    @classmethod
    def from_dict(cls, repo_dict: dict[str, Any]) -> 'SWEEnvProfile':
        """Build a profile from a reference instance dict.

        Expected repo_dict keys: instance_id, split, repo, base_commit, image/docker_image, workdir, patch, test_patch, language, problem_statement, eval_script.
        """
        ref_instance_id = repo_dict['instance_id']
        split = repo_dict['split']
        cache_key = f'swe-env.{split}.{ref_instance_id}'

        if cache_key in _SWE_ENV_PROFILE_CACHE:
            return _SWE_ENV_PROFILE_CACHE[cache_key]

        repo_field = repo_dict['repo']
        owner, repo = repo_field.split('/', 1) if '/' in repo_field else ('tencent-hunyuan-t3', repo_field)

        init_patch = repo_dict.get('patch')
        init_files = [x.path for x in get_patch_set(init_patch)] if init_patch else []
        init_test_patch = repo_dict.get('test_patch')
        init_test_files = [x.path for x in get_patch_set(init_test_patch)] if init_test_patch else []
        test_files = list(set(init_test_files) - set(init_files))

        ref_problem_statement = repo_dict.get('problem_statement') or repo_dict.get('problem', '')
        profile = cls(
            owner=owner,
            repo=repo,
            commit=repo_dict['base_commit'],
            split=split,
            ref_instance_id=ref_instance_id,
            docker_image=repo_dict.get('image') or repo_dict.get('docker_image'),
            language=repo_dict.get('language') or 'python',
            workdir=repo_dict.get('workdir', '/testbed'),
            init_patch=init_patch,
            init_test_patch=init_test_patch,
            test_files=test_files,
            test_function=repo_dict['eval_script'],
            ref_problem_statement=ref_problem_statement,
        )
        _SWE_ENV_PROFILE_CACHE[cache_key] = profile
        return profile

    def get_eval_script(self, instance: dict[str, Any]) -> str:
        """Get the test command for the repository."""
        # Materialize test_function into EVAL_SCRIPT_PATH (test_cmd runs `bash` on it).
        # The single-quoted heredoc delimiter disables all expansion, so the (multi-line, special-char) body is written verbatim. We write it before TEST_OUTPUT_START so `set -x` tracing of the heredoc stays out of the parsed test-output region.
        write_eval_script_cmd = '\n'.join([
            f"cat > {_EVAL_SCRIPT_PATH} << 'NEXUS_EVAL_SCRIPT_EOF'",
            self.test_function,
            'NEXUS_EVAL_SCRIPT_EOF',
        ])
        revert_files_cmd = ''
        bug_commit = instance.get(_BUG_COMMIT_KEY, '')
        revert_files = instance.get(_REVERT_FILES_KEY, [])
        if bug_commit and revert_files:
            revert_files_cmd = '\n'.join([
                f'git checkout {bug_commit} -- {shlex.quote(file)} 2>/dev/null || true' for file in revert_files
            ])
        return '\n'.join([
            '#!/bin/bash',
            write_eval_script_cmd,
            revert_files_cmd,
            'set -x',
            f": '{TEST_OUTPUT_START}'",
            self.test_cmd,
            'echo ">>>>>>>>>> Evaluation Exit Code: $? <<<<<<<<<<"',
            f": '{TEST_OUTPUT_END}'",
        ])

    def log_parser(self, log: str | None) -> dict[str, str]:
        """Determine the test status of the overall test (special for SWE-Env, only one test case)."""
        matched = _EXIT_CODE_RE.search(log or '')
        if matched and int(matched.group(1)) == 0:
            return {'all': 'PASSED'}
        return {'all': 'FAILED'}
